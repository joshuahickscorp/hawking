#!/usr/bin/env python3
"""G074: choose exceptions by functional consequence, and prove it against magnitude.

The verify is a head-to-head at EQUAL exception budget, so every rule below picks
the same NUMBER of exceptions from the same tensor and the same base, and only the
ranking differs. Nothing here is compared at equal error or equal anything else,
because that would let a rule win by spending more.

Three rules, in increasing knowledge of what the tensor is for:

  MAGNITUDE   |w|            the classic outlier rule -- keep the biggest weights
  RESIDUAL    |w - q(w)|     keep the weights the quantizer got most wrong
  FUNCTIONAL  d_j*(w-q(w))^2 keep the weights whose error the OUTPUT actually feels

The middle rule matters. Without it a functional win could be dismissed as "of
course, magnitude does not know about the quantizer" -- RESIDUAL knows exactly
that and nothing about activations, so the gap between RESIDUAL and FUNCTIONAL
isolates what the activation statistics contribute on their own.

d is fitted on one half of the captured rows and error scored on the other.

  ./tools/gravity_exception_selection.py --out receipts/.../G074_EXCEPTIONS.json
"""
from __future__ import annotations
import argparse, json, math, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor, quantize_group  # noqa: E402
from gravity_phase_transition import ORGANS, acts  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITES = ["mlp.gate_proj", "mlp.down_proj", "linear_attn.out_proj", "self_attn.q_proj"]


def out_rel(x, w, wh):
    y = x @ w.T
    return float(np.linalg.norm(x @ wh.T - y) / np.linalg.norm(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-rows", type=int, default=1536)
    ap.add_argument("--act-rows", type=int, default=768)
    ap.add_argument("--base-bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--budgets", default="0.25,0.5,1.0")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    budgets = [float(b) for b in a.budgets.split(",")]

    out, ratios = [], {}
    for organ in SITES:
        suffix, site, width, prov, layers = ORGANS[organ]
        L = layers[-1]
        w = load_tensor(f"language_model.model.layers.{L}.{suffix}").astype(np.float32)[:a.tensor_rows]
        x = acts(site, L, width, a.act_rows)
        half = x.shape[0] // 2
        xf, xe = x[:half], x[half:]
        d = (xf.astype(np.float64) ** 2).sum(0)
        d = (d / d.mean()).astype(np.float32)
        rows, cols = w.shape
        n = rows * cols
        base, _ = quantize_group(w, a.base_bits, a.group)
        R = w - base
        e_base = out_rel(xe, w, base)
        idx = math.ceil(math.log2(n))

        rules = {"MAGNITUDE": np.abs(w).astype(np.float64),
                 "RESIDUAL": np.abs(R).astype(np.float64),
                 "FUNCTIONAL": (R.astype(np.float64) ** 2) * d[None, :]}
        entry = {"organ": organ, "layer": L, "site": site, "provenance": prov,
                 "shape": [rows, cols], "base_output_rel_fro": e_base, "budgets": []}
        for B in budgets:
            k = int(B * n // (16 + idx))
            res = {}
            for name, sc in rules.items():
                wh = base.copy()
                sel = np.argpartition(sc.ravel(), -k)[-k:]
                wh.ravel()[sel] = w.ravel()[sel]
                res[name] = out_rel(xe, w, wh)
                del wh
            red = {kk: (e_base - v) / e_base * 100.0 for kk, v in res.items()}
            entry["budgets"].append({
                "budget_bits_per_elem": B, "exceptions": k,
                "exception_frac_of_tensor": k / n,
                "err": res, "error_reduction_pct": red,
                "functional_over_magnitude_error_ratio": res["MAGNITUDE"] / res["FUNCTIONAL"],
                "functional_over_residual_error_ratio": res["RESIDUAL"] / res["FUNCTIONAL"]})
            ratios.setdefault("vs_magnitude", []).append(res["MAGNITUDE"] / res["FUNCTIONAL"])
            ratios.setdefault("vs_residual", []).append(res["RESIDUAL"] / res["FUNCTIONAL"])
            print(f"{organ:<22}L{L:<3} B={B:<5} k={k:<8} base {e_base:.5f}  "
                  f"MAG {res['MAGNITUDE']:.5f}  RESID {res['RESIDUAL']:.5f}  "
                  f"FUNC {res['FUNCTIONAL']:.5f}   func/mag {res['MAGNITUDE']/res['FUNCTIONAL']:.2f}x")
        out.append(entry)
        del w, x, R, base

    vm, vr = ratios["vs_magnitude"], ratios["vs_residual"]
    print(f"\nFUNCTIONAL vs MAGNITUDE error ratio: mean {np.mean(vm):.2f}x, "
          f"worst {np.min(vm):.2f}x, best {np.max(vm):.2f}x over {len(vm)} cells")
    print(f"FUNCTIONAL vs RESIDUAL  error ratio: mean {np.mean(vr):.2f}x, "
          f"worst {np.min(vr):.2f}x, best {np.max(vr):.2f}x")
    print(f"functional wins every cell: vs magnitude {all(r>1 for r in vm)}, "
          f"vs residual {all(r>1 for r in vr)}")

    doc = {
        "schema": "hawking.nos.exception_selection.v1",
        "obligation": "G074 -- functional exception allocation measured against magnitude",
        "rules": {"MAGNITUDE": "|w|, the classic outlier rule",
                  "RESIDUAL": "|w - q(w)|, the weights the quantizer got most wrong",
                  "FUNCTIONAL": "d_j*(w-q(w))^2, the weights whose error the output feels"},
        "why_residual_is_included": ("without it a functional win could be dismissed as 'magnitude "
                                     "does not know about the quantizer'. RESIDUAL knows exactly "
                                     "that and nothing about activations, so the gap between "
                                     "RESIDUAL and FUNCTIONAL isolates what the activation "
                                     "statistics contribute on their own."),
        "equal_budget": "every rule picks the SAME number of exceptions from the same tensor and "
                        "the same base; only the ranking differs",
        "holdout": "d fitted on the first half of captured rows, error scored on the second",
        "sites": out,
        "summary": {"functional_over_magnitude_mean": float(np.mean(vm)),
                    "functional_over_magnitude_worst": float(np.min(vm)),
                    "functional_over_magnitude_best": float(np.max(vm)),
                    "functional_over_residual_mean": float(np.mean(vr)),
                    "functional_over_residual_worst": float(np.min(vr)),
                    "wins_every_cell_vs_magnitude": bool(all(r > 1 for r in vm)),
                    "wins_every_cell_vs_residual": bool(all(r > 1 for r in vr))},
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
