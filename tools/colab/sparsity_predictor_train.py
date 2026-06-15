#!/usr/bin/env python3
"""sparsity_predictor_train — train a per-layer FFN active-block predictor.

EXPERIMENTAL (the DENOMINATOR lever). Decode reads every FFN weight per token,
but for a given token most FFN neurons contribute ~nothing. If a cheap
predictor says "blocks {i,j,k} of the FFN-down weight matter for this token,"
the runtime skips the rest → fewer weight bytes read/token → direct tps gain
that MULTIPLIES with speculative decode.

This trains, per transformer layer, a tiny MLP:
    predictor_l : RMSNorm(residual_l)  ->  P(block b is active)   for b in blocks
where a block (default 256 contiguous intermediate channels) is "active" if any
of its silu*up activations exceed a magnitude threshold (so skipping it changes
the FFN-down output negligibly). Trained with BCE; we report recall at a target
skip rate (the runtime cares about recall: never skip a truly-active block).

EXPECTED INPUT (produced by a LOCAL capture, NOT yet built — see after-steps):
  --ffn-dir <dir>/shard_*.parquet, each row one sequence:
    layer            : int32          (which transformer layer)
    norm_in_q        : bytes int8     (n_tok, hidden)  RMSNorm(residual) input
    norm_in_scale    : f32
    act_blockmax_q   : bytes int8     (n_tok, n_blocks) per-block max|silu*up|
    act_blockmax_scale : f32
    shape            : list[int]      [n_tok, hidden, n_blocks]

Because the local FFN capture does not exist yet, this file is committed for the
final-push notebook's Track C (default OFF) and for audit. Output: one small
predictor per layer (npz) + a recall/skip-rate report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pyarrow.parquet as pq
except ImportError:
    sys.stderr.write("needs torch + pyarrow\n")
    sys.exit(1)


class BlockPredictor(nn.Module):
    """hidden -> hidden//4 -> n_blocks logits. Tiny: its own forward must cost
    far less than the FFN bytes it lets us skip."""

    def __init__(self, hidden: int, n_blocks: int):
        super().__init__()
        mid = max(64, hidden // 4)
        self.net = nn.Sequential(
            nn.Linear(hidden, mid), nn.GELU(), nn.Linear(mid, n_blocks)
        )

    def forward(self, x):
        return self.net(x)


def _deq(row, stem):
    q = np.frombuffer(row[f"{stem}_q"], dtype=np.int8).astype(np.float32)
    return (q * float(row[f"{stem}_scale"]))


def load_layer(shards, layer, act_thresh):
    """Return (X: [N,hidden], Y: [N,n_blocks] in {0,1}) for one layer."""
    xs, ys = [], []
    for sh in shards:
        t = pq.read_table(sh)
        cols = t.column_names
        for i in range(t.num_rows):
            r = {c: t[c][i].as_py() for c in cols}
            if int(r["layer"]) != layer:
                continue
            n_tok, hidden, n_blocks = (int(x) for x in r["shape"])
            x = _deq(r, "norm_in").reshape(n_tok, hidden)
            a = _deq(r, "act_blockmax").reshape(n_tok, n_blocks)
            xs.append(x)
            ys.append((a > act_thresh).astype(np.float32))
    if not xs:
        return None, None
    return np.concatenate(xs), np.concatenate(ys)


def main() -> int:
    ap = argparse.ArgumentParser(prog="sparsity_predictor_train")
    ap.add_argument("--ffn-dir", required=True)
    ap.add_argument("--frozen", required=True, help="for hidden dim (token_embd shape)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--act-thresh", type=float, default=0.05)
    ap.add_argument("--target-skip", type=float, default=0.5,
                    help="report recall when skipping this fraction of blocks")
    args = ap.parse_args()

    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        dev = "cpu"
    if dev == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        dev = "cpu"

    hidden = int(np.load(args.frozen)["token_embd"].shape[0])
    shards = sorted(Path(args.ffn_dir).glob("shard_*.parquet"))
    if not shards:
        sys.stderr.write(f"no shards in {args.ffn_dir}\n")
        return 1
    # discover layers present
    layers = set()
    for sh in shards[:4]:
        t = pq.read_table(sh)
        for i in range(t.num_rows):
            layers.add(int(t["layer"][i].as_py()))
    layers = sorted(layers)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    report = {}
    for layer in layers:
        X, Y = load_layer(shards, layer, args.act_thresh)
        if X is None:
            continue
        n_blocks = Y.shape[1]
        Xt = torch.from_numpy(X).to(dev)
        Yt = torch.from_numpy(Y).to(dev)
        model = BlockPredictor(hidden, n_blocks).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        N = Xt.shape[0]
        for ep in range(args.epochs):
            perm = torch.randperm(N, device=dev)
            for s in range(0, N, 4096):
                idx = perm[s : s + 4096]
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(model(Xt[idx]), Yt[idx])
                loss.backward()
                opt.step()
        # recall at target skip rate: skip the (target_skip) lowest-prob blocks,
        # measure fraction of truly-active blocks we KEPT.
        with torch.no_grad():
            p = torch.sigmoid(model(Xt))
            k_keep = max(1, int(round((1 - args.target_skip) * n_blocks)))
            topk = p.topk(k_keep, dim=-1).indices
            kept = torch.zeros_like(p)
            kept.scatter_(1, topk, 1.0)
            active = Yt > 0.5
            recall = (kept.bool() & active).sum().float() / active.sum().clamp(min=1)
        report[layer] = float(recall)
        np.savez(
            Path(args.out_dir) / f"sparsity_l{layer}.npz",
            w0=model.net[0].weight.detach().cpu().numpy(),
            b0=model.net[0].bias.detach().cpu().numpy(),
            w1=model.net[2].weight.detach().cpu().numpy(),
            b1=model.net[2].bias.detach().cpu().numpy(),
            block_size=np.int32(args.block_size),
            n_blocks=np.int32(n_blocks),
        )
        print(f"layer {layer}: recall@skip{args.target_skip:.0%} = {recall:.3f} "
              f"(n_blocks={n_blocks}, N={N})")
    import json
    json.dump(report, open(Path(args.out_dir) / "recall_report.json", "w"), indent=2)
    avg = sum(report.values()) / max(len(report), 1)
    print(f"avg recall@skip{args.target_skip:.0%} = {avg:.3f} over {len(report)} layers")
    print("HIGH recall (>0.99) at high skip = big BW cut at ~no quality loss.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
