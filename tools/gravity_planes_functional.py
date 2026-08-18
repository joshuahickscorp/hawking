#!/usr/bin/env python3
"""G069: fit each plane against FUNCTIONAL error instead of the weight residual.

G067 measured that the planes family FLOORS -- at the top of a five-plane sweep it
still sits 26-60x the flat code's error, and it is the only one of the three live
families with real curvature for that reason. The open question that leaves is
whether the floor belongs to the FAMILY or to the FITTING RULE, because the
existing ladder is greedy residual binarization in WEIGHT space and this
obligation says to fit against functional error.

The two objectives differ by one matrix:

  weight space      || W - W_hat ||_F^2
  function space    || X W^T - X W_hat^T ||_F^2  =  sum_r (w_r - w_hat_r) H (w_r - w_hat_r)^T

with H = X^T X. Taking the DIAGONAL of H -- the per-input-channel activation
energy -- reduces this to a weighted least squares, and then the plane fit has a
closed form that differs from the current one in exactly one place:

  signs   unchanged, sign(r) is still optimal because every weight d_j > 0
  scale   sum_j d_j |r_j| / sum_j d_j   instead of the plain mean |r_j|

So the functional fit is the SAME ladder with an activation-weighted scale. That
is the whole change, and it is why d=None must reproduce the existing weight-space
fitter bit for bit -- checked below, because a "new" fitter that is quietly a
different algorithm would make every comparison here meaningless.

Ternary planes are included since the obligation names them: symbols {-1,0,+1}
with p_j = sign(r_j) when |r_j| > s/2, alternating with the scale solve.

The activation energy is fitted on one half of the captured rows and the error
measured on the OTHER half. In-sample fitting is the exact trap that made the Q30
fits worthless, so the split is not optional.

  ./tools/gravity_planes_functional.py --out receipts/.../G069_PLANES_FUNCTIONAL.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor, quantize_group  # noqa: E402
from gravity_planes_ladder import binary_planes, SCALE_BITS  # noqa: E402
from gravity_phase_transition import ORGANS, acts  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG2_3 = float(np.log2(3.0))


def planes_fit(w, k, group, d=None, ternary=False, iters=3):
    rows, cols = w.shape
    assert cols % group == 0
    r = w.reshape(rows, cols // group, group).astype(np.float32).copy()
    dd = (np.ones((1, cols // group, group), np.float32) if d is None
          else d.reshape(1, cols // group, group).astype(np.float32))
    approx = np.zeros_like(r)
    for _ in range(k):
        a = np.abs(r)
        s = (dd * a).sum(2, keepdims=True) / dd.sum(2, keepdims=True)
        if ternary:
            for _ in range(iters):
                p = np.where(a > s / 2.0, np.sign(r), 0.0).astype(np.float32)
                m = p != 0
                num = (dd * a * m).sum(2, keepdims=True)
                den = (dd * m).sum(2, keepdims=True)
                s = np.where(den > 0, num / np.maximum(den, 1e-12), s)
            step = s * p
        else:
            step = s * np.where(r >= 0, 1.0, -1.0).astype(np.float32)
        approx += step
        r -= step
    per_symbol = 2.0 if ternary else 1.0   # raw packed, same convention as flat
    return approx.reshape(rows, cols), k * (per_symbol + SCALE_BITS / group)


def out_rel(x, w, w_hat):
    y = x @ w.T
    return float(np.linalg.norm(x @ w_hat.T - y) / np.linalg.norm(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1024)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    g = a.group

    # The control that makes every later comparison meaningful.
    rng = np.random.default_rng(7)
    t = rng.standard_normal((64, 256)).astype(np.float32)
    same = all(np.array_equal(planes_fit(t, k, 64)[0], binary_planes(t, k, 64)[0])
               for k in (1, 2, 3))
    print(f"CONTROL: d=None reproduces the weight-space fitter exactly: {same}")
    assert same, "the new fitter is not a superset of the old one"

    rows_out = []
    for organ, (suffix, site, width, prov, layers) in ORGANS.items():
        L = layers[-1]
        w = load_tensor(f"language_model.model.layers.{L}.{suffix}").astype(np.float32)
        x = acts(site, L, width, a.rows)
        half = x.shape[0] // 2
        xf, xe = x[:half], x[half:]          # fit on one half, score on the other
        d = (xf.astype(np.float64) ** 2).sum(0)
        d = (d / d.mean()).astype(np.float32)
        pts = []
        for k in (1, 2, 3, 4, 5):
            for tag, dv, tern in (("planes weight-fit", None, False),
                                  ("planes function-fit", d, False)):
                wh, bits = planes_fit(w, k, g, dv, tern)
                pts.append({"scheme": f"{tag} k{k}", "bits_per_elem": bits,
                            "output_rel_fro": out_rel(xe, w, wh)})
                del wh
        for k in (1, 2, 3):
            wh, bits = planes_fit(w, k, g, d, True)
            pts.append({"scheme": f"ternary function-fit k{k}", "bits_per_elem": bits,
                        "entropy_coded_bits_per_elem": k * (LOG2_3 + SCALE_BITS / g),
                        "output_rel_fro": out_rel(xe, w, wh)})
            del wh
        for b in (2, 3, 4, 5):
            wh, _ = quantize_group(w, b, g)
            pts.append({"scheme": f"flat q{b}", "bits_per_elem": b + SCALE_BITS / g,
                        "output_rel_fro": out_rel(xe, w, wh)})
            del wh
        gain = []
        for k in (1, 2, 3, 4, 5):
            a0 = next(p for p in pts if p["scheme"] == f"planes weight-fit k{k}")["output_rel_fro"]
            a1 = next(p for p in pts if p["scheme"] == f"planes function-fit k{k}")["output_rel_fro"]
            gain.append((a0 - a1) / a0)
        rows_out.append({"organ": organ, "layer": L, "site": site, "site_provenance": prov,
                         "eval_rows": int(xe.shape[0]), "fit_rows": int(xf.shape[0]),
                         "points": pts, "function_fit_error_reduction_by_k": gain})
        print(f"\n{organ} L{L} ({site}, {prov})  fit {xf.shape[0]} / eval {xe.shape[0]} rows")
        print(f"  {'k':>2}{'bits':>7}{'weight-fit':>12}{'function-fit':>14}{'reduction':>11}")
        for i, k in enumerate((1, 2, 3, 4, 5)):
            a0 = next(p for p in pts if p["scheme"] == f"planes weight-fit k{k}")
            a1 = next(p for p in pts if p["scheme"] == f"planes function-fit k{k}")
            print(f"  {k:>2}{a0['bits_per_elem']:>7.3f}{a0['output_rel_fro']:>12.5f}"
                  f"{a1['output_rel_fro']:>14.5f}{gain[i]*100:>10.2f}%")
        for k in (1, 2, 3):
            p = next(p for p in pts if p["scheme"] == f"ternary function-fit k{k}")
            print(f"  ternary k{k}  {p['bits_per_elem']:.3f} bits raw "
                  f"({p['entropy_coded_bits_per_elem']:.3f} entropy-coded)  "
                  f"{p['output_rel_fro']:.5f}")
        del w, x

    allg = [g for r in rows_out for g in r["function_fit_error_reduction_by_k"]]
    print(f"\nfunction-fit error reduction over weight-fit: mean {np.mean(allg)*100:.2f}%, "
          f"worst {np.min(allg)*100:.2f}%, best {np.max(allg)*100:.2f}%")

    doc = {
        "schema": "hawking.nos.planes_functional_fit.v1",
        "obligation": "G069 -- planes fitted to functional error, not residual weights",
        "derivation": ("||X(W-What)^T||_F^2 = sum_r (w_r-what_r) H (w_r-what_r)^T with H = X^T X. "
                       "The diagonal of H is the per-input-channel activation energy and reduces "
                       "this to weighted least squares, whose plane fit differs from the weight-"
                       "space one in exactly one place: the scale becomes sum_j d_j|r_j| / sum_j "
                       "d_j instead of the plain mean. Signs are unchanged because every d_j > 0."),
        "control": {"d_none_reproduces_weight_space_fitter": same,
                    "why": "a 'new' fitter that is quietly a different algorithm would make every "
                           "comparison here meaningless"},
        "holdout": "activation energy fitted on the first half of the captured rows, error measured "
                   "on the second half. In-sample fitting is the trap that made the Q30 fits "
                   "worthless, so the split is not optional.",
        "bits_convention": "raw packed, same as every other family here: binary 1 bit/symbol, "
                           "ternary 2 bits/symbol, plus 16 bits of scale per group. The log2(3) "
                           "entropy floor for ternary is carried alongside, not substituted in.",
        "sites": rows_out,
        "function_fit_reduction_mean": float(np.mean(allg)),
        "function_fit_reduction_worst": float(np.min(allg)),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
