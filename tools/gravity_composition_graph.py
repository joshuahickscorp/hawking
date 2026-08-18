#!/usr/bin/env python3
"""G068: are the compression axes orthogonal, or do they eat each other?

Every stacked-candidate projection this campaign has made assumed multiplicativity
-- 0.8 x 0.7 x 0.7 x 0.8 -- and the obligation is explicit that multiplicativity
must be MEASURED, not assumed. So this measures the interaction directly on a
factorial grid rather than reasoning about it.

The test. If two axes are independent, their effects are additive in log error:

  log e(a, b) - log e(a0, b0)  ==  [log e(a, b0) - log e(a0, b0)]
                                 + [log e(a0, b) - log e(a0, b0)]

The interaction is the amount by which that fails, in log2:

  I = log2 e(a,b) + log2 e(a0,b0) - log2 e(a,b0) - log2 e(a0,b)

  |I| small          ORTHOGONAL    stacking is multiplicative, as assumed
  I < 0              SYNERGISTIC   together they beat the product of their parts
  I > 0, small       REDUNDANT     they overlap; the second buys less than alone
  I > 0, large       CONFLICTING   the second axis undoes the first

A threshold has to be named before looking or the labels are decoration. 0.1 log2
units is one tenth of a halving of error, which is below the spread this harness
shows between adjacent layers of the same organ, so anything under it is not
distinguishable from site noise.

Axes measurable from what exists, and the ones that are not are reported UNKNOWN
with the reason rather than omitted:

  NUMERICAL   quantizer bit depth
  STRUCTURAL  scale group size
  ALGEBRAIC   channel-scale gauge fold, W diag(s) with diag(1/s) folded into the
              upstream norm -- the G032 mechanism, exact by construction

  DEPTH / STATE / ENTROPY  UNKNOWN: no candidate on the board varies depth or
              state, and the entropy axis is a coding stage that leaves the
              dequantized values identical, so it cannot move a functional error
              at all and would need TOKEN_NS to be scored.

  ./tools/gravity_composition_graph.py --out receipts/.../G068_COMPOSITION.json
"""
from __future__ import annotations
import argparse, json, itertools, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor, quantize_group  # noqa: E402
from gravity_phase_transition import ORGANS, acts  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
BITS = [2, 3, 4]          # NUMERICAL
GROUPS = [64, 256]        # STRUCTURAL
FOLD = [False, True]      # ALGEBRAIC
THRESH = 0.1              # log2 units, named before looking


def out_rel(x, w, w_hat):
    y = x @ w.T
    return float(np.linalg.norm(x @ w_hat.T - y) / np.linalg.norm(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=768)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    sites, cells = [], {}
    for organ, (suffix, site, width, prov, layers) in ORGANS.items():
        L = layers[-1]
        w = load_tensor(f"language_model.model.layers.{L}.{suffix}").astype(np.float32)
        x = acts(site, L, width, a.rows)
        half = x.shape[0] // 2
        xf, xe = x[:half], x[half:]
        # Channel scale from the FIT half only; the gauge is exact so the fold
        # cannot change the function, only how quantization error lands.
        e = np.sqrt((xf.astype(np.float64) ** 2).mean(0))
        s = (e / e.mean()) ** a.alpha
        s = np.where(s > 0, s, 1.0).astype(np.float32)
        grid = {}
        for b, g, f in itertools.product(BITS, GROUPS, FOLD):
            if f:
                wh = quantize_group(w * s, b, g)[0] / s
            else:
                wh = quantize_group(w, b, g)[0]
            grid[(b, g, f)] = out_rel(xe, w, wh)
            del wh
        cells[(organ, L)] = grid
        sites.append({"organ": organ, "layer": L, "site": site, "provenance": prov,
                      "grid": [{"bits": b, "group": g, "fold": f, "err": grid[(b, g, f)]}
                               for (b, g, f) in grid]})
        print(f"{organ:<22} L{L:<3} grid {len(grid)} cells")
        del w, x

    base = (BITS[-1], GROUPS[0], False)
    pairs = {"NUMERICAL x STRUCTURAL": [], "NUMERICAL x ALGEBRAIC": [],
             "STRUCTURAL x ALGEBRAIC": []}
    for key, grid in cells.items():
        lg = {k: float(np.log2(v)) for k, v in grid.items()}
        for b in BITS:
            if b == base[0]:
                continue
            for g in GROUPS:
                if g == base[1]:
                    continue
                pairs["NUMERICAL x STRUCTURAL"].append(
                    lg[(b, g, False)] + lg[base] - lg[(b, base[1], False)] - lg[(base[0], g, False)])
            pairs["NUMERICAL x ALGEBRAIC"].append(
                lg[(b, base[1], True)] + lg[base] - lg[(b, base[1], False)] - lg[(base[0], base[1], True)])
        for g in GROUPS:
            if g == base[1]:
                continue
            pairs["STRUCTURAL x ALGEBRAIC"].append(
                lg[(base[0], g, True)] + lg[base] - lg[(base[0], g, False)] - lg[(base[0], base[1], True)])

    def label(m):
        if abs(m) < THRESH:
            return "ORTHOGONAL"
        if m < 0:
            return "SYNERGISTIC"
        return "REDUNDANT" if m < 3 * THRESH else "CONFLICTING"

    graph = {}
    print(f"\n{'pair':<26}{'mean I':>10}{'worst I':>10}{'n':>5}  label   (threshold {THRESH} log2)")
    for k, v in pairs.items():
        m = float(np.mean(v)); worst = float(max(v, key=abs))
        graph[k] = {"interaction_log2_mean": m, "interaction_log2_worst": worst,
                    "n_cells": len(v), "label": label(m),
                    "label_at_worst_cell": label(worst)}
        print(f"{k:<26}{m:>10.4f}{worst:>10.4f}{len(v):>5}  {graph[k]['label']}"
              f"  (worst cell: {graph[k]['label_at_worst_cell']})")
    for k in ("DEPTH x anything", "STATE x anything", "ENTROPY x anything"):
        graph[k] = {"label": "UNKNOWN", "reason":
                    ("no live candidate varies depth or state (G062/G064 both refuted), and the "
                     "entropy stage leaves dequantized values identical so it cannot move a "
                     "functional error -- scoring it needs TOKEN_NS, not this harness")}
        print(f"{k:<26}{'--':>10}{'--':>10}{'--':>5}  UNKNOWN")

    doc = {
        "schema": "hawking.nos.composition_graph.v1",
        "obligation": "G068 -- multi-axis interaction; multiplicativity measured, not assumed",
        "interaction_definition": "I = log2 e(a,b) + log2 e(a0,b0) - log2 e(a,b0) - log2 e(a0,b); "
                                  "zero means the two axes compose multiplicatively in error",
        "threshold_log2": THRESH,
        "threshold_justification": "one tenth of a halving of error, below the spread this harness "
                                   "shows between adjacent layers of the same organ, so anything "
                                   "under it is not distinguishable from site noise. Named before "
                                   "looking so the labels are not decoration.",
        "baseline_cell": {"bits": base[0], "group": base[1], "fold": base[2]},
        "axes": {"NUMERICAL": f"quantizer bit depth {BITS}",
                 "STRUCTURAL": f"scale group size {GROUPS}",
                 "ALGEBRAIC": f"channel-scale gauge fold at alpha={a.alpha}, exact by construction"},
        "holdout": "channel scale fitted on the first half of captured rows, error scored on the "
                   "second half",
        "GRAVITY_COMPOSITION_GRAPH": graph,
        "sites": sites,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
