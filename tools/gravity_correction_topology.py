#!/usr/bin/env python3
"""G070: which correction TOPOLOGY is worth its index, at a matched bit budget.

A correction plane is not one mechanism, it is a choice of WHERE the corrections
live, and each choice pays a different index. Scattered entries carry a full
coordinate each; a whole row carries one row number for thousands of values. So
comparing topologies by functional gain alone is meaningless -- at a fixed budget
the topologies buy wildly different COUNTS of corrected weights, and that
trade is the entire question.

Every topology here is therefore given the SAME total bits per element and asked
what it can do with them. Nothing is compared at "the same number of corrections",
because that would hide the index cost the obligation exists to price.

  SCATTERED  N entries, each 16 value bits + ceil(log2(n_elem)) index bits
  ROW        N whole rows, each cols*16 + ceil(log2(rows))
  COLUMN     N whole columns, each rows*16 + ceil(log2(cols))
  BLOCK      N scale-groups of 64, each 64*16 + ceil(log2(n_groups))
  LOW-RANK   rank r, r*(rows+cols)*16, no index at all
  PLANE      one function-fitted binary plane, 1 + 16/group, priced at its own
             natural cost since it cannot be sized to an arbitrary budget

Selection is by FUNCTIONAL importance, not magnitude: each candidate is ranked by
d_j * resid^2, the actual output error it contributes, with d the per-input-channel
activation energy. Ranking by |w| would pick the weights that are large rather
than the ones that matter, which is the same weight-space mistake G069 corrected.

Low-rank gets the weighted optimum rather than a plain SVD: with D = diag(d),
tr((R-L) D (R-L)^T) = ||R D^0.5 - L D^0.5||_F^2, so the truncated SVD of R D^0.5
unscaled by D^-0.5 is exactly optimal.

Held out: d fitted on one half of the captured rows, error scored on the other.

  ./tools/gravity_correction_topology.py --out receipts/.../G070_TOPOLOGY.json
"""
from __future__ import annotations
import argparse, json, math, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor, quantize_group  # noqa: E402
from gravity_phase_transition import ORGANS, acts  # noqa: E402
from gravity_planes_functional import planes_fit  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITES = ["mlp.gate_proj", "mlp.down_proj", "linear_attn.out_proj", "self_attn.q_proj"]


def out_rel(x, w, w_hat):
    y = x @ w.T
    return float(np.linalg.norm(x @ w_hat.T - y) / np.linalg.norm(y))


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

    out_sites = []
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
        # Per-element functional error contribution.
        contrib = (R.astype(np.float64) ** 2) * d[None, :]

        idx_scat = math.ceil(math.log2(n))
        idx_row = math.ceil(math.log2(rows))
        idx_col = math.ceil(math.log2(cols))
        ngroups = n // a.group
        idx_blk = math.ceil(math.log2(ngroups))

        entry = {"organ": organ, "layer": L, "site": site, "provenance": prov,
                 "slice_shape": [rows, cols], "base": f"flat q{a.base_bits} g{a.group}",
                 "base_output_rel_fro": e_base, "eval_rows": int(xe.shape[0]), "budgets": []}

        for B in budgets:
            bits = B * n
            res = []

            k = int(bits // (16 + idx_scat))
            wh = base.copy()
            if k > 0:
                flat = np.argpartition(contrib.ravel(), -k)[-k:]
                wh.ravel()[flat] = w.ravel()[flat]
            res.append({"topology": "SCATTERED", "count": k, "unit": "entries",
                        "index_bits_per_unit": idx_scat, "elements_corrected": k,
                        "err": out_rel(xe, w, wh)})
            del wh

            k = int(bits // (cols * 16 + idx_row))
            wh = base.copy()
            if k > 0:
                sel = np.argpartition(contrib.sum(1), -k)[-k:]
                wh[sel] = w[sel]
            res.append({"topology": "ROW", "count": k, "unit": "rows",
                        "index_bits_per_unit": idx_row, "elements_corrected": k * cols,
                        "err": out_rel(xe, w, wh)})
            del wh

            k = int(bits // (rows * 16 + idx_col))
            wh = base.copy()
            if k > 0:
                sel = np.argpartition(contrib.sum(0), -k)[-k:]
                wh[:, sel] = w[:, sel]
            res.append({"topology": "COLUMN", "count": k, "unit": "columns",
                        "index_bits_per_unit": idx_col, "elements_corrected": k * rows,
                        "err": out_rel(xe, w, wh)})
            del wh

            k = int(bits // (a.group * 16 + idx_blk))
            wh = base.reshape(rows, cols // a.group, a.group).copy()
            wv = w.reshape(rows, cols // a.group, a.group)
            if k > 0:
                gc = contrib.reshape(rows, cols // a.group, a.group).sum(2)
                sel = np.argpartition(gc.ravel(), -k)[-k:]
                ri, gi = np.unravel_index(sel, gc.shape)
                wh[ri, gi] = wv[ri, gi]
            res.append({"topology": "BLOCK", "count": k, "unit": f"groups of {a.group}",
                        "index_bits_per_unit": idx_blk, "elements_corrected": k * a.group,
                        "err": out_rel(xe, w, wh.reshape(rows, cols))})
            del wh, wv

            r = int(bits // ((rows + cols) * 16))
            wh = base.copy()
            if r > 0:
                sq = np.sqrt(d)
                U, S, Vt = np.linalg.svd((R * sq[None, :]).astype(np.float32),
                                         full_matrices=False)
                wh += (U[:, :r] * S[:r]) @ Vt[:r] / sq[None, :]
                del U, S, Vt
            res.append({"topology": "LOW-RANK", "count": r, "unit": "rank",
                        "index_bits_per_unit": 0, "elements_corrected": n if r else 0,
                        "err": out_rel(xe, w, wh)})
            del wh

            for t in res:
                t["budget_bits_per_elem"] = B
                t["index_overhead_frac"] = (
                    t["count"] * t["index_bits_per_unit"] / bits if t["count"] else 0.0)
                t["error_reduction_pct"] = (e_base - t["err"]) / e_base * 100.0
            entry["budgets"].append({"budget_bits_per_elem": B, "topologies": res})

        pl, plb = planes_fit(R, 1, a.group, d, False)
        entry["plane_at_natural_cost"] = {
            "topology": "PLANE", "budget_bits_per_elem": plb,
            "err": out_rel(xe, w, base + pl),
            "error_reduction_pct": (e_base - out_rel(xe, w, base + pl)) / e_base * 100.0,
            "note": "priced at its own natural cost; it cannot be sized to an arbitrary budget"}
        del pl
        out_sites.append(entry)

        print(f"\n{organ} L{L}  slice {rows}x{cols}  base q{a.base_bits} err {e_base:.5f}")
        for bg in entry["budgets"]:
            print(f"  budget {bg['budget_bits_per_elem']} b/elem")
            for t in sorted(bg["topologies"], key=lambda r: r["err"]):
                print(f"    {t['topology']:<10}{t['count']:>9} {t['unit']:<14}"
                      f"idx {t['index_overhead_frac']*100:5.1f}%  err {t['err']:.5f}  "
                      f"{t['error_reduction_pct']:+6.2f}%")
        p = entry["plane_at_natural_cost"]
        print(f"  PLANE at {p['budget_bits_per_elem']:.3f} b/elem  err {p['err']:.5f}  "
              f"{p['error_reduction_pct']:+.2f}%")
        del w, x, R, base, contrib

    winners = {}
    for B in budgets:
        tally = {}
        for s in out_sites:
            bg = next(b for b in s["budgets"] if b["budget_bits_per_elem"] == B)
            best = min(bg["topologies"], key=lambda r: r["err"])
            tally[best["topology"]] = tally.get(best["topology"], 0) + 1
        winners[str(B)] = tally
        print(f"\nbudget {B}: winner tally {tally}")

    doc = {
        "schema": "hawking.nos.correction_topology.v1",
        "obligation": "G070 -- correction topology costed jointly with its index",
        "method": ("every topology gets the SAME total bits per element and is asked what it can do "
                   "with them, so the index cost is inside the comparison rather than beside it. "
                   "Selection is by d_j*resid^2, the output error each candidate actually "
                   "contributes -- ranking by |w| would pick large weights instead of important "
                   "ones, the weight-space mistake G069 corrected."),
        "low_rank_optimality": ("with D=diag(d), tr((R-L)D(R-L)^T) = ||R D^0.5 - L D^0.5||_F^2, so "
                                "the truncated SVD of R D^0.5 rescaled by D^-0.5 is exactly optimal "
                                "for this weighting, not an approximation"),
        "holdout": "d fitted on the first half of captured rows, error scored on the second",
        "slice_note": f"measured on a {a.tensor_rows}-row slice of each tensor; index costs are "
                      f"computed for THAT shape so the accounting is self-consistent",
        "sites": out_sites, "winner_tally_by_budget": winners,
        "kernel_cost_status": (
            "NOT MEASURED HERE. This closes the mathematical half. The verify also requires the "
            "winning topology to be shown against its KERNEL cost, and the only scattered-"
            "correction kernel in the tree, strand_outlier_correct, does one atomic_fetch_add per "
            "outlier into the output row -- serialising every entry that shares a row. It has ZERO "
            "references from qwen38_hybrid_decode.rs (verified by name search, 38 of 554 declared "
            "kernels are bound there), so no correction topology executes on this model today."),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
