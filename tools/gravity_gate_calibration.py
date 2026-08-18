#!/usr/bin/env python3
"""Calibrate the adequacy gate against artifacts whose END-TO-END verdict is known.

The gate declares a candidate adequate when every axis lands within margin of a SAME-TENSOR
honest-Q4 reference. That is a proxy. It has never been checked against ground truth, and it
is load-bearing: the sub-1.0 bound rests on it.

Ground truth available on disk:
  mixed-q3mlp-v1  complete 3.6138647373, MLP 3.2500251321, non-MLP 4.2501427135
                  recipe: mlp.{gate,up,down}_proj = HGRAVU01 uniform_q3_group64,
                  reconstruct_to_q4 false, everything else copied from mixed-2p0-v1
                  VERDICT: COHERENT end to end (clears France-Paris and 17x19=323)
  uniform-q4-v1   complete 4.255954555664, VERDICT: COHERENT

So uniform q3 at group 64, applied to all 192 MLP tensors, DEMONSTRABLY preserves end-to-end
behaviour. If the gate rejects that same construction per tensor, the gate is stricter than
coherence, and every bound derived from it is correspondingly conservative. If the gate
accepts it, the gate is calibrated at this point and the bound stands.

This does not tell us the gate is right in general -- one coherent point cannot do that. It
tells us whether the gate is WRONG at a point where the answer is already known, which is the
only cheap check available.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gravity_doctor_gate import load_tensor, load_X, axes, c_uniform, AXIS_MARGIN  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=64, help="artifact group size (G0 packs g64)")
    ap.add_argument("--layers", default="0,15,31,47,63")
    a = ap.parse_args()
    g = a.group
    layers = [int(x) for x in a.layers.split(",")]

    print(f"reference = same-tensor uniform q4 group {g}; candidate = uniform q3 group {g},")
    print("which is exactly what mixed-q3mlp-v1 packs into all 192 MLP tensors and which is")
    print("COHERENT end to end at 3.6138647373 complete BPW.\n")
    print(f"{'tensor':<30}{'observed':>10}{'probed':>10}{'worst_u':>10}{'gain':>10}   verdict")
    rows, rejects = [], 0
    for l in layers:
        for cls in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            name = f"language_model.model.layers.{l}.{cls}.weight"
            try:
                W = load_tensor(name).astype(np.float32)
            except Exception as e:
                print(f"{cls}@L{l:<26} SKIP {e}")
                continue
            X = load_X(l) if W.shape[1] == 5120 else None
            if X is None:
                X = np.random.default_rng(l).standard_normal((256, W.shape[1])).astype(np.float32)
            ref = axes(W, c_uniform(W, 4, g), X, seed=None)
            cand = axes(W, c_uniform(W, 3, g), X, seed=None)
            bad = [k for k, m in AXIS_MARGIN.items() if cand[k] < ref[k] - m]
            rejects += bool(bad)
            rows.append((f"{cls}@L{l}", ref, cand, bad))
            print(f"{cls+'@L'+str(l):<30}{cand['observed']:>10.6f}{cand['probed']:>10.6f}"
                  f"{cand['worst_unit']:>10.6f}{cand['gain']:>10.6f}   "
                  f"{'ADEQUATE' if not bad else 'reject: ' + ','.join(bad)}")
            print(f"{'  (q4 reference)':<30}{ref['observed']:>10.6f}{ref['probed']:>10.6f}"
                  f"{ref['worst_unit']:>10.6f}{ref['gain']:>10.6f}")

    n = len(rows)
    print(f"\n{rejects} of {n} MLP tensors REJECTED by the gate.")
    if rejects:
        print("The gate is STRICTER THAN DEMONSTRATED COHERENCE at this point: it rejects a")
        print("construction that is packed into a coherent artifact and runs end to end.")
        print("Consequence: every bound derived from the same-tensor-Q4 bar is CONSERVATIVE,")
        print("and the per-organ Doctor bar owed by G025 should be calibrated against measured")
        print("end-to-end behaviour rather than assumed equal to Q4 on every organ.")
        worst = {k: min(c[k] - r[k] for _, r, c, _ in rows) for k in AXIS_MARGIN}
        print("\nhow far below the Q4 reference a COHERENT construction actually sits:")
        for k, v in worst.items():
            print(f"  {k:<12} worst deficit {v:>10.6f}   current margin {AXIS_MARGIN[k]:.2f}"
                  f"   -> margin would need to be >= {abs(v):.4f}")
    else:
        print("The gate ACCEPTS the coherent construction. Calibrated at this point; the")
        print("same-tensor-Q4 bar is not over-strict here and the bound stands as stated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
