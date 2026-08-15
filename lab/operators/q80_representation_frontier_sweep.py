#!/usr/bin/env python3
"""Matched binary vs uniform-Qn sweep on Q80 experts, against the measured 0.8604 bar.

The low-rank family is refuted at this operating point (measured: every rung clearing 0.8604
costs >= 2.03 expert BPW against a 1.3012 allowance, while the binary incumbent reaches 0.8926
far cheaper). So the open question is whether a *cheaper* family converts bits to cosine well
enough to lift up_proj over the bar inside the allowance.

The prescription's apparent "2-bit is worse than 1-bit" is confounded: the allocator assigns
2 bits only to the organs it judges hardest. This runs 1-bit and 2-bit on the SAME organs,
which is the matched comparison that claim needs.

Uses the X cache written by q80_rank_bits_sweep.py, so it costs seconds, not the 15-minute
1.38 GB JSON parse.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/Users/scammermike/Downloads/hawking")

from lab.operators.ascension_dual_gravity_worker import _mean_row_cosine
from lab.operators.doctor6.rungs import quant_binary, quant_uniform, quant_residual
from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map

MODEL_DIR = Path(os.environ["MODEL_DIR"])
XCACHE = Path(os.environ["XCACHE"])
BAR = 0.8604
ALLOWANCE = 1.3011578470468521

LE_PAIRS = [(10, 453), (3, 494)]
# down_proj input is the post-SwiGLU intermediate (512-dim), not the captured 2048-dim hidden,
# so it cannot be scored from this capture. gate/up share the hidden and are scorable.
COMPONENTS = ("gate_proj", "up_proj")


def score(W, X, rec, nbytes):
    bpw = 8.0 * nbytes / max(W.size, 1)
    cos = _mean_row_cosine(X @ W.T, X @ rec.astype(np.float32).T)
    return float(cos), float(bpw)


def main():
    z = np.load(XCACHE)
    by_le = {(int(k.split("_")[0]), int(k.split("_")[1])): z[k] for k in z.files}
    wmap = load_weight_map(MODEL_DIR)
    out = {"bar": BAR, "expert_bpw_allowance": ALLOWANCE, "organs": []}

    for (layer, expert) in LE_PAIRS:
        X = np.asarray(by_le[(layer, expert)], dtype=np.float32)
        for comp in COMPONENTS:
            key = f"model.layers.{layer}.mlp.experts.{expert}.{comp}.weight"
            W = np.asarray(load_tensor(MODEL_DIR, wmap, key), dtype=np.float32)
            rec_out = {"organ_key": key, "component": comp, "rows": int(X.shape[0]),
                       "W_shape": list(W.shape), "candidates": []}
            print(f"\n=== {key}  rows={X.shape[0]}  W={tuple(W.shape)}", flush=True)

            trials = [("binary_g", lambda W=W: quant_binary(W))]
            for b in (2, 3, 4):
                trials.append((f"uniform_b{b}", lambda W=W, b=b: quant_uniform(W, bits=b)))
            for frac in (0.02, 0.05):
                trials.append((f"binary+resid_{int(frac*100)}pct",
                               lambda W=W, f=frac: quant_residual(W, outlier_ratio=f)))

            for name, fn in trials:
                try:
                    rec, nbytes = fn()
                except Exception as exc:  # noqa: BLE001
                    print(f"  {name:<22} ERROR {exc}", flush=True)
                    continue
                cos, bpw = score(W, X, rec, nbytes)
                ok, aff = cos >= BAR, bpw <= ALLOWANCE
                verdict = "PASS" if (ok and aff) else ("over-budget" if ok else "fail")
                rec_out["candidates"].append({
                    "codec": name, "cosine": cos, "expert_bpw": bpw,
                    "clears_bar": ok, "within_allowance": aff, "verdict": verdict})
                print(f"  {name:<22} cos={cos:.4f} bpw={bpw:.4f}  {verdict}", flush=True)
            out["organs"].append(rec_out)

    dest = os.environ.get("OUT", "/tmp/q80_binary_uniform_sweep.json")
    Path(dest).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}", flush=True)


if __name__ == "__main__":
    main()
