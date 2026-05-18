"""EAGLE-3 baseline head + side-by-side compare with EAGLE-4.

EAGLE-3: 4-input fusion (no shared-expert input), no mask head, no calib.
Trained on the same per-layer captures EAGLE-4 sees.

  python bench.py train --parquet data/*.parquet --frozen v2lite_frozen.npz \\
                        --ckpt-dir ckpt/eagle3
  python bench.py compare --eagle3 ckpt/eagle3/latest.npz \\
                          --eagle4 ckpt/eagle4/best.npz \\
                          --frozen v2lite_frozen.npz \\
                          --parquet data/heldout/*.parquet
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as mxoptim
import numpy as np
import pyarrow.parquet as pq

import eagle4


# ---------------------------------------------------------------------------
# EAGLE-3 head: 4-input fusion, no mask
# ---------------------------------------------------------------------------
class Eagle3Head(nn.Module):
    def __init__(self, token_embd, lm_head, output_norm):
        super().__init__()
        self._token_embd = token_embd
        self._lm_head = lm_head
        self._output_norm = output_norm
        self.in_proj = nn.Linear(4 * eagle4.HIDDEN_DIM, eagle4.HIDDEN_DIM, bias=False)
        self.block = eagle4._Block()
        self.residual_gate = mx.zeros((1,))

    def trainable_parameters(self):
        p = self.parameters()
        for k in ("_token_embd", "_lm_head", "_output_norm"):
            p.pop(k, None)
        return p

    def __call__(self, prev_tok, h_low, h_mid, h_high):
        B, S = prev_tok.shape
        attn_mask = (mx.eye(S) - 1.0) * 1e9
        embed_table = mx.transpose(self._token_embd, (1, 0))
        prev_embed = embed_table[prev_tok]
        x = mx.concatenate([prev_embed, h_low, h_mid, h_high], axis=-1)
        x = self.in_proj(x)
        x = self.block(x, attn_mask)
        baseline = mx.fast.rms_norm(h_high, self._output_norm, eagle4.RMS_EPS)
        draft_hidden = baseline.astype(x.dtype) + self.residual_gate * x
        return draft_hidden @ self._lm_head, draft_hidden


def build_eagle3(frozen: Path) -> Eagle3Head:
    z = np.load(frozen)
    return Eagle3Head(mx.array(z["token_embd"]), mx.array(z["lm_head"]), mx.array(z["output_norm"]))


# ---------------------------------------------------------------------------
# EAGLE-3 trainer — token CE + aux MSE, no mask BCE
# ---------------------------------------------------------------------------
def _iter_e3(shards: list[Path], batch_size: int, seq_len: int, epochs: int, seed: int = 0):
    rng = random.Random(seed)
    rows = []
    for s in shards:
        t = pq.read_table(s)
        for i in range(t.num_rows):
            rows.append(
                {
                    "prev": t["prev_token"][i].as_py(),
                    "next": t["next_token"][i].as_py(),
                    "low": t["hidden_low"][i].as_py(),
                    "mid": t["hidden_mid"][i].as_py(),
                    "high": t["hidden_high"][i].as_py(),
                }
            )
    print(f"[bench] loaded {len(rows)} records", flush=True)
    H = eagle4.HIDDEN_DIM
    per_batch = batch_size * seq_len
    for epoch in range(epochs):
        rng.shuffle(rows)
        for i in range(0, len(rows) - per_batch + 1, per_batch):
            sub = rows[i : i + per_batch]
            prev = np.array([r["prev"] for r in sub], dtype=np.int32).reshape(batch_size, seq_len)
            nxt = np.array([r["next"] for r in sub], dtype=np.int32).reshape(batch_size, seq_len)
            low = np.frombuffer(b"".join(r["low"] for r in sub), dtype=np.float16).reshape(batch_size, seq_len, H)
            mid = np.frombuffer(b"".join(r["mid"] for r in sub), dtype=np.float16).reshape(batch_size, seq_len, H)
            hi = np.frombuffer(b"".join(r["high"] for r in sub), dtype=np.float16).reshape(batch_size, seq_len, H)
            yield {
                "prev": mx.array(prev),
                "next": mx.array(nxt),
                "low": mx.array(low).astype(mx.float32),
                "mid": mx.array(mid).astype(mx.float32),
                "high": mx.array(hi).astype(mx.float32),
                "epoch": epoch,
            }


def train_eagle3(parquet_paths: list[Path], frozen: Path, ckpt_dir: Path,
                 epochs: int = 4, batch_size: int = 32, seq_len: int = 16,
                 lr: float = 3e-4, aux_weight: float = 0.5):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    head = build_eagle3(frozen)
    print(f"[bench] EAGLE-3 head built (4-input fusion, no mask, aux={aux_weight})", flush=True)

    def loss_fn(head, b):
        tok, draft_h = head(b["prev"], b["low"], b["mid"], b["high"])
        B, S, V = tok.shape
        pos_mask = mx.concatenate([mx.zeros((B, 3)), mx.ones((B, S - 3))], axis=1).reshape(-1)
        ce_per = nn.losses.cross_entropy(tok.reshape(-1, V), b["next"].reshape(-1), reduction="none")
        ce = (ce_per * pos_mask).sum() / mx.maximum(pos_mask.sum(), mx.array(1.0))
        baseline = mx.fast.rms_norm(b["high"], head._output_norm, eagle4.RMS_EPS)
        mse_per = ((draft_h - baseline) ** 2).mean(axis=-1).reshape(-1)
        mse = (mse_per * pos_mask).sum() / mx.maximum(pos_mask.sum(), mx.array(1.0))
        return ce + aux_weight * mse

    grad_fn = nn.value_and_grad(head, loss_fn)
    opt = mxoptim.AdamW(learning_rate=lr, weight_decay=0.01)
    step = 0
    t0 = time.time()
    for batch in _iter_e3(parquet_paths, batch_size, seq_len, epochs):
        loss, grads = grad_fn(head, batch)
        opt.update(head, grads)
        mx.eval(head.parameters(), opt.state, loss)
        step += 1
        if step % 25 == 0 or step == 1:
            print(f"step={step} epoch={batch['epoch']} loss={float(loss):.3f} gate={float(head.residual_gate[0]):.3f} wall={time.time()-t0:.0f}s", flush=True)
        if step % 200 == 0:
            eagle4.save_ckpt(head, ckpt_dir / "latest.npz", step)

    eagle4.save_ckpt(head, ckpt_dir / "latest.npz", step)
    print(f"[bench] EAGLE-3 train done: {step} steps in {time.time()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------
def _eval_eagle3(ckpt: Path, frozen: Path, shards: list[Path], max_records: int = 5000) -> dict:
    head = build_eagle3(frozen)
    eagle4.load_ckpt(head, ckpt)
    H = eagle4.HIDDEN_DIM
    n = n_target = n_corpus = 0
    for shard in shards:
        t = pq.read_table(shard)
        take = min(max_records - n, t.num_rows)
        if take <= 0:
            break
        prev = np.array([t["prev_token"][i].as_py() for i in range(take)], dtype=np.int32)
        nxt = np.array([t["next_token"][i].as_py() for i in range(take)], dtype=np.int32)
        low = np.frombuffer(b"".join(t["hidden_low"][i].as_py() for i in range(take)), dtype=np.float16).reshape(take, H)
        mid = np.frombuffer(b"".join(t["hidden_mid"][i].as_py() for i in range(take)), dtype=np.float16).reshape(take, H)
        hi = np.frombuffer(b"".join(t["hidden_high"][i].as_py() for i in range(take)), dtype=np.float16).reshape(take, H)
        ph = mx.array(prev).reshape(1, take)
        lo_a = mx.array(low).astype(mx.float32).reshape(1, take, H)
        md_a = mx.array(mid).astype(mx.float32).reshape(1, take, H)
        hi_a = mx.array(hi).astype(mx.float32).reshape(1, take, H)
        tok, _ = head(ph, lo_a, md_a, hi_a)
        baseline = mx.fast.rms_norm(hi_a.reshape(take, H), head._output_norm, eagle4.RMS_EPS)
        target_logits = baseline @ head._lm_head.astype(mx.float32)
        mx.eval(tok, target_logits)
        head_arg = np.argmax(np.array(tok).reshape(take, -1), axis=-1)
        target_arg = np.array(mx.argmax(target_logits, axis=-1))
        n_target += int((head_arg == target_arg).sum())
        n_corpus += int((head_arg == nxt).sum())
        n += take
        if n >= max_records:
            break
    n = max(n, 1)
    return {"scored": n, "top1_vs_target": n_target / n, "top1_vs_corpus": n_corpus / n}


def compare(eagle3_ckpt: Path, eagle4_ckpt: Path, frozen: Path, shards: list[Path], max_records: int = 5000) -> dict:
    print("[bench] EAGLE-3 baseline:")
    e3 = _eval_eagle3(eagle3_ckpt, frozen, shards, max_records)
    for k, v in e3.items():
        print(f"  {k:24s} {v}")
    print("[bench] EAGLE-4:")
    e4 = eagle4.evaluate(eagle4_ckpt, frozen, shards, max_records)
    for k, v in e4.items():
        if k != "mask_topk_per_layer_recall":
            print(f"  {k:24s} {v}")
    return {"eagle3": e3, "eagle4": e4}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(prog="bench")
    sub = p.add_subparsers(dest="cmd", required=True)

    tp = sub.add_parser("train", help="train EAGLE-3 baseline")
    tp.add_argument("--parquet", nargs="+", type=Path, required=True)
    tp.add_argument("--frozen", type=Path, required=True)
    tp.add_argument("--ckpt-dir", type=Path, required=True)
    tp.add_argument("--epochs", type=int, default=4)
    tp.add_argument("--batch-size", type=int, default=32)
    tp.add_argument("--seq-len", type=int, default=16)
    tp.add_argument("--lr", type=float, default=3e-4)
    tp.add_argument("--aux-weight", type=float, default=0.5)

    cp = sub.add_parser("compare", help="EAGLE-3 vs EAGLE-4 side-by-side on the same held-out")
    cp.add_argument("--eagle3", type=Path, required=True)
    cp.add_argument("--eagle4", type=Path, required=True)
    cp.add_argument("--frozen", type=Path, required=True)
    cp.add_argument("--parquet", nargs="+", type=Path, required=True)
    cp.add_argument("--max-records", type=int, default=5000)
    cp.add_argument("--out", type=Path, default=Path("bench_results.json"))

    args = p.parse_args()
    if args.cmd == "train":
        train_eagle3(args.parquet, args.frozen, args.ckpt_dir,
                     epochs=args.epochs, batch_size=args.batch_size,
                     seq_len=args.seq_len, lr=args.lr, aux_weight=args.aux_weight)
    elif args.cmd == "compare":
        r = compare(args.eagle3, args.eagle4, args.frozen, args.parquet, args.max_records)
        args.out.write_text(json.dumps(r, indent=2))
        print(f"\n[bench] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
