#!/usr/bin/env python3
"""Correctness gate for batched RWKV-7 SFT training.

A right-padded batched forward must produce the SAME logits at real token
positions as running each sequence individually — otherwise `--batch-size > 1`
in rwkv7_train_draft.py would silently train on corrupted signal. This holds
because RWKV-7's recurrence is strictly left-to-right: pad tokens appended at
the end of shorter sequences cannot perturb earlier positions, and rows in a
batch never interact.

Runs CPU-only on a tiny model (no GPU contention with a live sweep), exercising
both the chunked and non-chunked WKV paths, including chunk-boundary cases.
Exit 0 = pass.
"""
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rwkv7_torch_model import RWKV7Config, RWKV7Model  # noqa: E402


def tiny_cfg(use_chunked: bool, chunk_size: int) -> RWKV7Config:
    return RWKV7Config(
        n_embd=128, n_layer=3, n_ff=256, head_dim=64, n_head=2,
        vocab_size=256, use_chunked=use_chunked, chunk_size=chunk_size,
    )


def per_position_logits(model, ids):
    x = torch.tensor([ids], dtype=torch.long)
    hidden = model(x, return_final_hidden=True)  # [1, T, H]
    return model.lm_head(hidden[0]).float()      # [T, V]


def batched_logits(model, seqs, pad_id=0):
    B = len(seqs)
    T = max(len(s) for s in seqs)
    x = torch.full((B, T), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    hidden = model(x, return_final_hidden=True)  # [B, T, H]
    return [model.lm_head(hidden[i, : len(s)]).float() for i, s in enumerate(seqs)]


def run_case(use_chunked, chunk_size, seqs, tol=2e-4):
    torch.manual_seed(0)
    model = RWKV7Model(tiny_cfg(use_chunked, chunk_size))
    model.eval()
    with torch.no_grad():
        indiv = [per_position_logits(model, s) for s in seqs]
        batched = batched_logits(model, seqs)
    worst = 0.0
    for a, b in zip(indiv, batched):
        worst = max(worst, (a - b).abs().max().item())
    tag = f"chunked={use_chunked} cs={chunk_size} lens={[len(s) for s in seqs]}"
    ok = worst <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {tag}: worst |Δlogit|={worst:.3e} (tol {tol:.0e})")
    return ok


def main():
    # Mixed lengths; include lengths that straddle chunk boundaries (cs=4 -> 5,7,9).
    seqs = [
        [3, 7, 1, 9, 200, 13, 42],          # len 7
        [5, 5, 5, 1, 2, 3, 4, 8, 99, 17, 6],  # len 11
        [10, 20, 30, 40, 50],                # len 5
        [1, 2, 3, 4, 5, 6, 7, 8, 9],          # len 9
    ]
    cases = [
        (False, 32, seqs),
        (True, 4, seqs),    # chunk size 4 vs lens {5,7,9,11}: exercises partial last chunk
        (True, 32, seqs),   # chunk >= all lens: single-chunk path
    ]
    all_ok = True
    print("RWKV-7 batched-vs-individual forward equivalence:")
    for uc, cs, ss in cases:
        all_ok &= run_case(uc, cs, ss)
    print("RESULT:", "ALL PASS" if all_ok else "FAILURE")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
