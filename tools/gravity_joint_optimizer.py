#!/usr/bin/env python3
"""G129: search SCALE x CODEC at fixed CODED bits, and reproduce the absmax law.

The directive states a compiler law: MSE-optimal scaling improves local fidelity but
RAISES CODE ENTROPY, so after entropy coding a cruder absmax scale can afford more
precision at the same coded size. The acceptance says that law must be REPRODUCED BY
THE SEARCH, not assumed -- so this searches scale per group, measures what each
choice costs in CODED bits, and scores both in FUNCTION space on real activations.

Three quantities per candidate, and the third is the one that matters:

  weight MSE     what an MSE-optimal scale minimises, by construction
  coded bits     empirical symbol entropy + the scale stream. An MSE scale that
                 spreads the histogram pays here.
  output error   relative Frobenius of W x on real captured activations -- the only
                 axis Doctor cares about.

The law is reproduced if MSE wins on weight MSE, LOSES on coded bits, and the two
cross when compared at equal coded size.
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor  # noqa: E402
from gravity_phase_transition import ORGANS, acts  # noqa: E402


def quant(w, bits, group, alpha):
    """alpha scales the absmax denominator: 1.0 is plain absmax, <1 shrinks the step."""
    rows, cols = w.shape
    g = w.reshape(rows, cols // group, group)
    qmax = (1 << (bits - 1)) - 1
    am = np.abs(g).max(axis=2, keepdims=True)
    s = np.where(am > 0, am * alpha / qmax, 1.0).astype(np.float32)
    codes = np.clip(np.rint(g / s), -qmax - 1, qmax).astype(np.int32)
    return (codes * s).reshape(rows, cols), codes


def coded_bits(codes, bits, group, scale_bits=16):
    c = codes.ravel() + (1 << (bits - 1))
    n = np.bincount(c, minlength=1 << bits).astype(np.float64)
    p = n[n > 0] / n.sum()
    H = float(-(p * np.log2(p)).sum())
    return H + scale_bits / group, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organs", default="mlp.gate_proj,self_attn.q_proj")
    ap.add_argument("--bits", type=int, default=3)
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--rows", type=int, default=512)
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    ALPHAS = [round(x, 3) for x in np.arange(0.60, 1.01, 0.05)]

    sites = []
    for organ in a.organs.split(","):
        suffix, site, width, prov, layers = ORGANS[organ]
        L = layers[-1]
        w = load_tensor(f"language_model.model.layers.{L}.{suffix}").astype(np.float32)[:1536]
        x = acts(site, L, width, a.rows)
        y = x @ w.T; ny = np.linalg.norm(y)
        rows = []
        for al in ALPHAS:
            wq, codes = quant(w, a.bits, a.group, al)
            cb, H = coded_bits(codes, a.bits, a.group)
            rows.append({"alpha": al, "weight_mse": float(((w - wq) ** 2).mean()),
                         "symbol_entropy": H, "coded_bits_per_elem": cb,
                         "output_rel_fro": float(np.linalg.norm(x @ wq.T - y) / ny)})
            del wq, codes
        best_mse = min(rows, key=lambda r: r["weight_mse"])
        best_out = min(rows, key=lambda r: r["output_rel_fro"])
        absmax = next(r for r in rows if r["alpha"] == 1.0)
        sites.append({"organ": organ, "layer": L, "sweep": rows,
                      "absmax": absmax, "mse_optimal": best_mse, "output_optimal": best_out})
        print(f"\n{organ} L{L}  bits={a.bits} group={a.group}")
        print(f"{'alpha':>7}{'weight MSE':>14}{'entropy':>10}{'coded b/elem':>14}{'output err':>12}")
        for r in rows:
            tag = ""
            if r is absmax: tag += "  <- absmax"
            if r is best_mse: tag += "  <- min weight MSE"
            if r is best_out: tag += "  <- min OUTPUT err"
            print(f"{r['alpha']:>7.2f}{r['weight_mse']:>14.3e}{r['symbol_entropy']:>10.4f}"
                  f"{r['coded_bits_per_elem']:>14.4f}{r['output_rel_fro']:>12.5f}{tag}")
        del w, x, y

    print(f"\n{'organ':<20}{'absmax coded':>14}{'MSE coded':>12}{'MSE pays':>10}"
          f"{'absmax out':>12}{'MSE out':>10}")
    law = []
    for s in sites:
        am, ms = s["absmax"], s["mse_optimal"]
        pays = ms["coded_bits_per_elem"] - am["coded_bits_per_elem"]
        law.append({"organ": s["organ"],
                    "mse_wins_weight_mse": ms["weight_mse"] < am["weight_mse"],
                    "mse_costs_coded_bits": pays,
                    "mse_pays_in_bits": pays > 0,
                    "absmax_output": am["output_rel_fro"], "mse_output": ms["output_rel_fro"],
                    "absmax_beats_mse_on_output": am["output_rel_fro"] < ms["output_rel_fro"]})
        print(f"{s['organ']:<20}{am['coded_bits_per_elem']:>14.4f}{ms['coded_bits_per_elem']:>12.4f}"
              f"{pays:>+10.4f}{am['output_rel_fro']:>12.5f}{ms['output_rel_fro']:>10.5f}")

    doc = {"schema": "hawking.nos.joint_scale_codec_search.v1",
           "obligation": "G129 -- joint search at FIXED CODED BITS; reproduce the absmax law",
           "bits": a.bits, "group": a.group, "alphas": ALPHAS, "sites": sites,
           "law_check": law,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, cwd=ROOT if (ROOT := pathlib.Path(__file__).resolve().parents[1]) else None).stdout.strip()}
    if a.out:
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
