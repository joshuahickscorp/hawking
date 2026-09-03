#!/usr/bin/env python3
"""Summarize /tmp/g1-functional-distillation/run.jsonl into tables."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

JSONL = Path("/tmp/g1-functional-distillation/run.jsonl")
OUT = Path("/tmp/g1-functional-distillation/summary.json")
G0_S0 = 0.4078534106896186
N_MLP = 192


def load():
    cells, extras, cal, amp = [], defaultdict(list), None, None
    for line in JSONL.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        k = r.get("kind")
        if k == "cell":
            cells.append(r)
        elif k == "calibration":
            cal = r
        elif k == "amplification":
            amp = r
        else:
            extras[k].append(r)
    return cells, extras, cal, amp


def s0(r, who="vs_bf16"):
    return r["scores"]["s0"][who]["flat_cosine"]


def main():
    cells, extras, cal, amp = load()
    # index
    by = {}
    for r in cells:
        if r.get("source") != "bf16":
            continue
        key = (r["layer"], r["role"], r["codec"]["bits"], r["codec"]["g"], r["codec"]["family"], r["objective"])
        by[key] = r

    layers = sorted({r["layer"] for r in cells})
    roles = ["gate", "up", "down", "out"]
    grid = []
    for layer in layers:
        for role in roles:
            for bits, g, fam in ((2, 64, "hgravu"), (3, 64, "hgravu"), (4, 64, "hq30"), (1, 128, "binary")):
                rec = {"layer": layer, "role": role, "bits": bits, "g": g, "family": fam}
                ok = True
                for obj in ("weight_absmax", "weight_mse", "func_bf16", "func_g0"):
                    r = by.get((layer, role, bits, g, fam, obj))
                    if r is None:
                        ok = False
                        rec[obj] = None
                    else:
                        rec[obj] = {
                            "s0_bf16": s0(r, "vs_bf16"),
                            "s0_g0": s0(r, "vs_g0"),
                            "s0_row": r["scores"]["s0"]["vs_bf16"]["mean_row_cosine"],
                            "s0_rel": r["scores"]["s0"]["vs_bf16"]["rel_l2"],
                            "prompt_bf16": r["scores"]["prompt"]["vs_bf16"]["flat_cosine"],
                            "evenodd_bf16": r["scores"]["evenodd"]["vs_bf16"]["flat_cosine"],
                            "evenodd_row": r["scores"]["evenodd"]["vs_bf16"]["mean_row_cosine"],
                            "frac_not_abs": r.get("meta", {}).get("frac_groups_not_absmax"),
                            "nominal_bpw": r["nominal_bpw"],
                        }
                if rec["weight_absmax"] and rec["func_bf16"]:
                    rec["d_func_bf16_vs_abs"] = rec["func_bf16"]["s0_bf16"] - rec["weight_absmax"]["s0_bf16"]
                    rec["d_func_g0_vs_abs_g0"] = rec["func_g0"]["s0_g0"] - rec["weight_absmax"]["s0_g0"]
                    rec["d_weight_mse_vs_abs"] = rec["weight_mse"]["s0_bf16"] - rec["weight_absmax"]["s0_bf16"]
                if ok:
                    grid.append(rec)

    # projected S0 product: treat sample of 11 layers as representative of 64
    # 64 layers * 3 mlp roles = 192. Our sample is 11*3=33 mlp tensors.
    def proj_product(obj, bits, g, fam, roles_=("gate", "up", "down")):
        vals = []
        for rec in grid:
            if rec["bits"] == bits and rec["g"] == g and rec["family"] == fam and rec["role"] in roles_:
                if rec[obj]:
                    vals.append(rec[obj]["s0_bf16"])
        if not vals:
            return None
        geo = math.exp(sum(math.log(max(v, 1e-15)) for v in vals) / len(vals))
        # ESTIMATED product over 192 assuming sample geo mean
        return {
            "n_sample": len(vals),
            "geo_mean": geo,
            "min": min(vals),
            "max": max(vals),
            "est_product_192": geo ** 192,
            "est_product_label": "ESTIMATED from sample geo-mean ** 192",
        }

    projections = {}
    for bits, g, fam, tag in (
        (4, 64, "hq30", "q4g64"),
        (3, 64, "hgravu", "q3g64"),
        (2, 64, "hgravu", "q2g64"),
        (1, 128, "binary", "bin128"),
    ):
        for obj in ("weight_absmax", "weight_mse", "func_bf16", "func_g0"):
            projections[f"{tag}/{obj}"] = proj_product(obj, bits, g, fam)

    # n-sweep saturation
    nsweeps = []
    for rec in extras.get("n_sweep", []):
        rows = rec["rows"]
        byn = defaultdict(dict)
        for row in rows:
            byn[row["n_fit"]][row["objective"]] = row["vs_bf16"]["flat_cosine"]
        sat = []
        for n, d in sorted(byn.items()):
            if "func_bf16" in d and "weight_absmax" in d:
                sat.append(
                    {
                        "n_fit": n,
                        "abs": d["weight_absmax"],
                        "func": d["func_bf16"],
                        "delta": d["func_bf16"] - d["weight_absmax"],
                    }
                )
        nsweeps.append({"layer": rec["layer"], "role": rec["role"], "curve": sat})

    # LOO
    loos = []
    for rec in extras.get("loo", []):
        dlt = []
        for row in rec["rows"]:
            if row["objective"] == "func_bf16":
                # find matching absmax
                ab = next(
                    x
                    for x in rec["rows"]
                    if x["objective"] == "weight_absmax" and x["leave_prompt"] == row["leave_prompt"]
                )
                dlt.append(
                    {
                        "leave": row["leave_prompt"],
                        "abs": ab["vs_bf16"]["flat_cosine"],
                        "func": row["vs_bf16"]["flat_cosine"],
                        "delta": row["vs_bf16"]["flat_cosine"] - ab["vs_bf16"]["flat_cosine"],
                    }
                )
        loos.append({"layer": rec["layer"], "role": rec["role"], "per_prompt": dlt})

    # composed mlp
    composed = []
    for rec in extras.get("composed_mlp", []):
        g0 = rec["g0_vs_bf16"]
        best = defaultdict(dict)
        for row in rec["replacements"]:
            best[(row["bits"], row["role"], row["objective"])] = row
        composed.append(
            {
                "layer": rec["layer"],
                "g0_vs_bf16_flat": g0["flat_cosine"],
                "amp_write_over_H": rec["amp_write_over_H"],
                "amp_write_over_H_g0": rec["amp_write_over_H_g0"],
                "replacements": rec["replacements"],
            }
        )

    # levels
    levels = []
    for rec in extras.get("levels", []):
        levels.append(
            {
                "layer": rec["layer"],
                "role": rec["role"],
                "bits": rec["bits"],
                "obj": rec["objective"],
                "s0_bf16": rec["scores"]["s0"]["vs_bf16"]["flat_cosine"],
                "s0_g0": rec["scores"]["s0"]["vs_g0"]["flat_cosine"],
                "bpw": rec["nominal_bpw"],
            }
        )

    src_g0 = [r for r in cells if r.get("source") == "g0"]

    out = {
        "n_cells": len(cells),
        "n_grid": len(grid),
        "layers": layers,
        "calibration_pack_max_abs": None if cal is None else cal.get("g0_pack_vs_requant_4rows"),
        "grid": grid,
        "projections": projections,
        "n_sweeps": nsweeps,
        "loos": loos,
        "composed": [
            {k: v for k, v in c.items() if k != "replacements"}
            | {
                "q3_gate_abs": next(
                    (
                        x["vs_bf16"]["flat_cosine"]
                        for x in next(cc["replacements"] for cc in extras.get("composed_mlp", []) if cc["layer"] == c["layer"])
                        if x["bits"] == 3 and x["role"] == "gate" and x["objective"] == "weight_absmax"
                    ),
                    None,
                ),
                "q3_gate_func": next(
                    (
                        x["vs_bf16"]["flat_cosine"]
                        for x in next(cc["replacements"] for cc in extras.get("composed_mlp", []) if cc["layer"] == c["layer"])
                        if x["bits"] == 3 and x["role"] == "gate" and x["objective"] == "func_bf16"
                    ),
                    None,
                ),
                "q3_down_abs": next(
                    (
                        x["vs_bf16"]["flat_cosine"]
                        for x in next(cc["replacements"] for cc in extras.get("composed_mlp", []) if cc["layer"] == c["layer"])
                        if x["bits"] == 3 and x["role"] == "down" and x["objective"] == "weight_absmax"
                    ),
                    None,
                ),
                "q3_down_func": next(
                    (
                        x["vs_bf16"]["flat_cosine"]
                        for x in next(cc["replacements"] for cc in extras.get("composed_mlp", []) if cc["layer"] == c["layer"])
                        if x["bits"] == 3 and x["role"] == "down" and x["objective"] == "func_bf16"
                    ),
                    None,
                ),
            }
            for c in composed
        ],
        "levels": levels,
        "exceptions": extras.get("exceptions", []),
        "partitions": extras.get("partition", []),
        "src_g0_n": len(src_g0),
        "amplification_present": amp is not None,
        "g0_s0_cited": G0_S0,
    }
    if amp is not None:
        out["amp_product"] = amp.get("product_H_norm_ratios_1_to_63")
        out["amp_L63_over_L0"] = amp.get("layer63_mean_H_over_layer0")
        out["amp_write"] = amp.get("write_detail")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT} grid={len(grid)} cells={len(cells)}")
    # print compact Q3 table
    print("\nQ3 g64 s0 flat vs_bf16  absmax | w_mse | func_bf16 | func_g0 | d_func")
    for rec in grid:
        if rec["bits"] != 3 or rec["g"] != 64:
            continue
        if not rec.get("func_bf16"):
            continue
        print(
            f"L{rec['layer']:02d} {rec['role']:4}  "
            f"{rec['weight_absmax']['s0_bf16']:.6f}  "
            f"{rec['weight_mse']['s0_bf16']:.6f}  "
            f"{rec['func_bf16']['s0_bf16']:.6f}  "
            f"{rec['func_g0']['s0_bf16']:.6f}  "
            f"{rec['d_func_bf16_vs_abs']:+.6f}"
        )
    print("\nprojections:")
    for k, v in projections.items():
        if v:
            print(f"  {k:28} n={v['n_sample']:2} geo={v['geo_mean']:.6f} min={v['min']:.6f} est192={v['est_product_192']:.6e}")


if __name__ == "__main__":
    main()
