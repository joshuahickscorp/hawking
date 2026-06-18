"""On-device preference tuning (SimPO) for RWKV-7-0.4B on Apple MPS, on top of the
SFT checkpoint. Uses the parity-verified pure-torch forward.

Why SimPO (not vanilla DPO): vanilla DPO needs a frozen *reference* model copy in
memory alongside the policy — a second 0.4B (~1.8GB fp32) that won't fit on a 19GB
box during training. SimPO is reference-free: it maximizes the length-normalized
log-prob margin between chosen and rejected with a target margin gamma, so only one
model is resident. (Loss: -log_sigmoid(beta*(logp_c/|c| - logp_r/|r|) - gamma).)

Pairs come from rwkv7_dpo_build_pairs (chosen = gold answer, rejected = the SFT'd
student's own sampled generation). Tokenization matches the SFT trainer.

Usage (after SFT + pair build):
    python rwkv7_dpo_torch.py --sft-state artifacts/rwkv7_posttrain/sft_out/final/state_dict.pt \
        --pairs artifacts/rwkv7_posttrain/dpo.jsonl --out artifacts/rwkv7_posttrain/dpo_out \
        --device mps --last-n-layers 16 --max-length 448
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from rwkv7_load_weights import load_rwkv7  # noqa: E402
from rwkv7_torch_model import RWKV7Config  # noqa: E402
from rwkv7_sft_torch import load_tokenizer, freeze_to_last_n  # noqa: E402

EOS = 0  # <|rwkv_tokenizer_end_of_text|>


def encode_pair(tok, prompt_user: str, completion: str, max_length: int):
    """Return (input_ids, comp_start) for [0]+User/Assistant prompt + completion + [0]."""
    prompt = [EOS] + tok.encodeBytes(f"User: {prompt_user}\n\nAssistant:".encode("utf-8"))
    comp = tok.encodeBytes(f" {completion}".encode("utf-8")) + [EOS]
    ids = (prompt + comp)[:max_length]
    return ids, len(prompt)


def seq_logp(model, ids, comp_start, device):
    """Sum log-prob of the completion tokens (and their count) under the policy."""
    x = torch.tensor([ids], dtype=torch.long, device=device)
    hidden = model(x, return_final_hidden=True)  # [1,T,n_embd]
    # token t predicts t+1; completion tokens are positions [comp_start, T)
    logits = model.lm_head(hidden[0, comp_start - 1:-1])  # predict comp tokens
    targets = x[0, comp_start:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return tok_logp.sum(), tok_logp.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-state", required=True, help="SFT checkpoint state_dict.pt")
    ap.add_argument("--pairs", default=str(ROOT / "artifacts/rwkv7_posttrain/dpo.jsonl"))
    ap.add_argument("--hf-dir", default=str(ROOT / "models/rwkv7-g1-04-hf"))
    ap.add_argument("--out", default=str(ROOT / "artifacts/rwkv7_posttrain/dpo_out"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-length", type=int, default=448)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=2.0)       # SimPO margin scale
    ap.add_argument("--gamma", type=float, default=0.5)      # SimPO target margin
    ap.add_argument("--last-n-layers", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=50, help="opt steps; overwrites <out>/latest")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--use-chunked", action="store_true", help="chunked-scan WKV-7 (7-9x faster fwd+bwd)")
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tok = load_tokenizer(Path(args.hf_dir))
    rows = [json.loads(l) for l in open(args.pairs)]
    if args.max_rows:
        rows = rows[: args.max_rows]
    pairs = []
    for r in rows:
        u = r.get("user") or r.get("prompt", "")
        if not r.get("chosen") or not r.get("rejected") or not u:
            continue
        ci, cs = encode_pair(tok, u, r["chosen"], args.max_length)
        ri, rs = encode_pair(tok, u, r["rejected"], args.max_length)
        if len(ci) <= cs or len(ri) <= rs:  # need >=1 completion token each
            continue
        pairs.append((ci, cs, ri, rs))
    print(f"[data] {len(pairs)} usable preference pairs")

    cfg = RWKV7Config(use_chunked=args.use_chunked, chunk_size=args.chunk_size)
    if args.use_chunked:
        print(f"[model] chunked-scan enabled (chunk_size={args.chunk_size}) — 7-9x faster fwd+bwd")
    model = load_rwkv7(args.hf_dir + "/model.safetensors", cfg, device=args.device, dtype=torch.float32)
    model.load_state_dict(torch.load(args.sft_state, map_location=args.device), strict=True)
    print(f"[model] loaded SFT weights from {args.sft_state}")
    model.grad_checkpoint = True
    model.train()
    if args.last_n_layers:
        freeze_to_last_n(model, args.last_n_layers)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.95))

    def simpo_loss(pair):
        ci, cs, ri, rs = pair
        lc, nc = seq_logp(model, ci, cs, args.device)
        lr, nr = seq_logp(model, ri, rs, args.device)
        margin = args.beta * (lc / max(nc, 1) - lr / max(nr, 1)) - args.gamma
        return -F.logsigmoid(margin), (lc / max(nc, 1)).item(), (lr / max(nr, 1)).item()

    if args.dry_run:
        print("[dry-run] 3 SimPO steps")
        for k in range(3):
            loss, lc, lr = simpo_loss(pairs[k % len(pairs)])
            loss.backward(); opt.step(); opt.zero_grad()
            print(f"  step {k}: loss={loss.item():.4f}  logp_chosen={lc:.3f} logp_rejected={lr:.3f} "
                  f"(margin {'+' if lc>lr else '-'})")
        print("[dry-run] OK")
        return

    os.makedirs(args.out, exist_ok=True)
    n_steps = 0
    ema = None
    t0 = time.time()
    for epoch in range(args.epochs):
        opt.zero_grad()
        for i, pair in enumerate(pairs):
            loss, lc, lr = simpo_loss(pair)
            (loss / args.grad_accum).backward()
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if args.device == "mps":
                torch.mps.empty_cache()  # per-example: same OOM guard as SFT hardening
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad()
                n_steps += 1
                if n_steps % args.log_every == 0:
                    print(f"[ep{epoch} step {n_steps}] loss={ema:.4f} margin_acc={'win' if lc>lr else 'lose'} "
                          f"({i+1}/{len(pairs)} pairs, {(time.time()-t0)/60:.0f}min)", flush=True)
                if args.save_every and n_steps % args.save_every == 0:
                    _save_ckpt(model, args.out, "latest")
    _save_ckpt(model, args.out, "final")
    print(f"[done] DPO {n_steps} steps, final loss={ema:.4f}")


def _save_ckpt(model, out, tag):
    d = Path(out) / tag
    d.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().float() for k, v in model.state_dict().items()}, d / "state_dict.pt")
    print(f"  [save] {d/'state_dict.pt'}", flush=True)


if __name__ == "__main__":
    main()
