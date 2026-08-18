#!/usr/bin/env python3
"""Search the group scale against the JOINT objective: gate score at fixed CODED bits.

Absmax and MSE are two points on a one-parameter family, s = r * max|w| / bound, and they sit
at opposite ends of a trade (AMENDMENT 2026-08-17m): absmax concentrates the histogram and
codes cheaply, MSE spreads it and reconstructs better. Both were chosen against objectives
that ignore the codec. Neither is required to be the joint optimum.

The unexplored half of the family is r > 1, i.e. deliberately OVER-scaling so the codes do not
reach the bound. That trades fidelity for two things at once: a more concentrated histogram,
and -- unlike any r <= 1 -- a code range that is genuinely NARROWER than the container. The
narrow range is what fixed-bits-per-tile needed and never had, so this is also the only way
that dead parallel form could come back.

The honest currency is CODED bits per element: the entropy-coded stream under a 12-bit
quantized model, plus the f16 group scale. Comparing at nominal bit width would hide the whole
effect. Every point is scored on the four-axis gate against a same-tensor honest Q4 reference,
and the deliverable is the PARETO FRONTIER -- which r gives the best gate at each cost -- not
a single winner.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gravity_doctor_gate import load_tensor, load_X, axes, c_uniform  # noqa: E402
from gravity_parallel_code import quantized_freqs, rans_bits          # noqa: E402

GROUP = 64


def quant_ratio(W, bits, r, group=GROUP):
    m, n = W.shape
    g = W.reshape(-1, group)
    bound = (1 << (bits - 1)) - 1
    s = np.maximum(np.abs(g).max(1, keepdims=True) * r / bound, 1e-30)
    s = s.astype(np.float16).astype(np.float32)
    q = np.clip(np.rint(g / s), -bound, bound)
    return (q * s).reshape(m, n), (q + bound).astype(np.uint16).ravel()


def coded_bits(sym, prec=12):
    h = np.bincount(sym, minlength=int(sym.max()) + 1)
    return rans_bits(h, quantized_freqs(h, prec)) / sym.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="mlp.down_proj@31")
    ap.add_argument("--bits", default="2,3,4")
    ap.add_argument("--ratios", default="0.7,0.85,1.0,1.2,1.5,2.0,3.0")
    a = ap.parse_args()
    cls, l = a.site.split("@"); l = int(l)
    W = load_tensor(f"language_model.model.layers.{l}.{cls}.weight").astype(np.float32)
    X = load_X(l) if W.shape[1] == 5120 else None
    if X is None:
        X = np.random.default_rng(l).standard_normal((256, W.shape[1])).astype(np.float32)
    ref = axes(W, c_uniform(W, 4, GROUP), X, seed=None)
    M = {"observed": 0.02, "probed": 0.02, "worst_unit": 0.10, "gain": 0.02}
    print(f"{a.site}, group {GROUP}. r=1.0 is absmax, the incumbent. r>1 over-scales so the")
    print(f"codes never reach the bound. Cost is CODED bits (rANS, 12-bit model) + f16 scale.\n")
    print(f"q4 g64 reference: observed {ref['observed']:.6f} probed {ref['probed']:.6f} "
          f"worst_u {ref['worst_unit']:.6f} gain {ref['gain']:.6f}\n")
    print(f"{'bits':>5}{'r':>6}{'coded b/w':>11}{'+scale':>9}{'range':>7}"
          f"{'observed':>10}{'probed':>10}{'worst_u':>10}{'gain':>10}{'worst deficit':>15}")
    pts = []
    for b in (int(x) for x in a.bits.split(",")):
        for r in (float(x) for x in a.ratios.split(",")):
            Wh, sym = quant_ratio(W, b, r)
            ax = axes(W, Wh, X, seed=None)
            cb = coded_bits(sym)
            tot = cb + 8 * 2 / GROUP
            rng = int(sym.max()) - int(sym.min()) + 1
            defi = min(ax[k] - (ref[k] - m) for k, m in M.items())
            pts.append((tot, defi, b, r, rng, ax))
            print(f"{b:>5}{r:>6.2f}{cb:>11.4f}{tot:>9.4f}{rng:>7}"
                  f"{ax['observed']:>10.6f}{ax['probed']:>10.6f}{ax['worst_unit']:>10.6f}"
                  f"{ax['gain']:>10.6f}{defi:>15.6f}")

    print("\nPARETO FRONTIER (no other point is both cheaper and better):")
    pts.sort()
    best = -9e9
    front = []
    for tot, defi, b, r, rng, ax in pts:
        if defi > best:
            best = defi
            front.append((tot, defi, b, r, rng))
    print(f"{'+scale b/w':>11}{'worst deficit':>15}{'bits':>6}{'r':>6}{'range':>7}   note")
    for tot, defi, b, r, rng in front:
        note = "ADEQUATE" if defi >= 0 else ""
        if r == 1.0:
            note += "  <- absmax incumbent"
        if rng < (1 << b):
            note += f"  range {rng} < {1<<b}: fixed-bits viable"
        print(f"{tot:>11.4f}{defi:>15.6f}{b:>6}{r:>6.2f}{rng:>7}   {note}")
    inc = [p for p in pts if p[2] == 3 and p[3] == 1.0][0]
    better = [p for p in pts if p[0] <= inc[0] and p[1] > inc[1]]
    print(f"\nincumbent q3 absmax: {inc[0]:.4f} b/w, worst deficit {inc[1]:.6f}")
    if better:
        p = max(better, key=lambda x: x[1])
        print(f"DOMINATES IT: q{p[2]} r={p[3]} at {p[0]:.4f} b/w, deficit {p[1]:.6f} "
              f"({p[1]-inc[1]:+.6f} at {p[0]-inc[0]:+.4f} bits)")
    else:
        print("nothing in this family is both cheaper-or-equal AND better than absmax q3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
