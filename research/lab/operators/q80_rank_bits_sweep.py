#!/usr/bin/env python3
"""Sweep (rank, bits) for Q80's failing organs against the MEASURED 0.8604 break-even.

Why this exists: the CLAMP145 prescription fits the bit budget with room to spare
(complete_physical_bpw 0.7714 of a 1.5 ceiling, expert_local_bpw 0.7472 of a 1.3012
allowance = 1.74x unused) yet fails coherence, because the rung ladder tops out at
rank 192 / 3-bit factors and yields gate 0.8926 / up 0.8277 / down 0.8128.

gate clears the measured bar 0.8604. up and down do not. rank is capped by
min(rows, W.shape[0], W.shape[1]) = min(2921, 512, 2048) = 512, so there is real
headroom above 192. The question is exactly: what rung reaches 0.8604, and what does
it cost in expert BPW.

Materializes only the two (layer, expert) pairs the prior prescription already named,
so it skips the multi-minute census entirely.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/scammermike/Downloads/hawking")

from lab.operators.ascension_dual_gravity_worker import _mean_row_cosine
from lab.operators.doctor6.rungs import awq_scale, quant_act_svd, random_orthogonal
from lab.operators.ascension_qwen30_activation_weighted_svd_repack import collect_expert_activations
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map

CAPTURE = Path(os.environ["CAPTURE"])
MODEL_DIR = Path(os.environ["MODEL_DIR"])
BAR = 0.8604
EXPERT_BPW_ALLOWANCE = 1.3011578470468521  # from QWEN80_BIT_BUDGET_LEDGER

LE_PAIRS = [(10, 453), (3, 494)]
COMPONENTS = ("gate_proj", "up_proj", "down_proj")
GRID = [(192, 3), (256, 3), (320, 3), (384, 3), (256, 4), (320, 4), (384, 4), (448, 4)]


def organ_cosine(W, X, rank, bits, key):
    """Same transform chain l2_mixed_prec uses, so the numbers are directly comparable."""
    s = awq_scale(W, X, alpha=0.5)
    W_awq = (W * s[None, :]).astype(np.float32)
    R = random_orthogonal(W.shape[1], seed=0xD0C6 ^ (hash(key) & 0xFFFFFFFF))
    W_rot = (W_awq @ R).astype(np.float32)
    X_rot = X @ R
    r = max(1, min(rank, X_rot.shape[0], W_rot.shape[0], W_rot.shape[1]))
    rec_rot, nbytes = quant_act_svd(W_rot, X_rot, rank=r, bits=bits)
    W_hat = ((rec_rot @ R.T) / np.maximum(s[None, :], 1e-12)).astype(np.float32)
    bpw = 8.0 * nbytes / max(W.size, 1)
    cos = _mean_row_cosine(X @ W.T, X @ W_hat.T)
    return float(cos), float(bpw), int(r)


def main():
    wanted = set(LE_PAIRS)
    cache = Path(os.environ.get("XCACHE", "/tmp/q80_sweep_xcache.npz"))
    if cache.exists():
        print(f"[sweep] loading cached X from {cache}", flush=True)
        z = np.load(cache)
        by_le = {(int(k.split("_")[0]), int(k.split("_")[1])): z[k] for k in z.files}
    else:
        print(f"[sweep] materializing {wanted} (no cache; this is the 1.38GB JSON parse)", flush=True)
        by_le, prov = collect_expert_activations(CAPTURE, wanted_keys=wanted)
        np.savez(cache, **{f"{L}_{E}": v for (L, E), v in by_le.items()})
        print(f"[sweep] cached X to {cache}", flush=True)
    wmap = load_weight_map(MODEL_DIR)
    out = {
        "bar": BAR,
        "expert_bpw_allowance": EXPERT_BPW_ALLOWANCE,
        "grid": [list(g) for g in GRID],
        "organs": [],
    }
    for (layer, expert) in LE_PAIRS:
        X = np.asarray(by_le[(layer, expert)], dtype=np.float32)
        for comp in COMPONENTS:
            key = f"model.layers.{layer}.mlp.experts.{expert}.{comp}.weight"
            W = np.asarray(load_tensor(MODEL_DIR, wmap, key), dtype=np.float32)
            rec = {
                "organ_key": key, "component": comp, "layer": layer, "expert": expert,
                "rows": int(X.shape[0]), "W_shape": list(W.shape), "rungs": [],
            }
            print(f"\n=== {key}  rows={X.shape[0]}  W={W.shape}", flush=True)
            for rank, bits in GRID:
                cos, bpw, r_eff = organ_cosine(W, X, rank, bits, key)
                ok = cos >= BAR
                affordable = bpw <= EXPERT_BPW_ALLOWANCE
                verdict = "PASS" if (ok and affordable) else ("over-budget" if ok else "fail")
                rec["rungs"].append({
                    "rank": rank, "bits": bits, "rank_effective": r_eff,
                    "cosine": cos, "expert_bpw": bpw,
                    "clears_bar": ok, "within_allowance": affordable, "verdict": verdict,
                })
                print(f"  r{rank:<4} b{bits}  eff_r={r_eff:<4} cos={cos:.4f} "
                      f"bpw={bpw:.4f}  {verdict}", flush=True)
            out["organs"].append(rec)
    dest = os.environ.get("OUT", "/tmp/q80_rank_bits_sweep.json")
    Path(dest).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}", flush=True)


if __name__ == "__main__":
    main()
