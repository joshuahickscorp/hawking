#!/usr/bin/env python3
"""G033 / G-PLANES: does a ladder of cheap binary planes beat a flat code at equal bits?

W ~ s1*P1 + s2*P2 + ... with each Pi in {-1,+1} and si a per-group scale. Fitted
greedily: each plane takes the sign of what the previous planes left behind. The
appeal is that a plane is the cheapest possible symbol, and the ladder can stop
when fidelity suffices instead of paying a fixed width everywhere.

The comparison this obligation asks for is at EQUAL TOTAL BITS, so both sides are
counted the same way and neither gets to omit its scales:

  k binary planes, group g:   k * (1 + 16/g) bits/elem
  flat b-bit, group g:            b + 16/g   bits/elem

At g=64 that is 1.25k against b+0.25. Two planes cost 2.50 bits/elem and flat q2
costs 2.25; three planes cost 3.75 and flat q3 costs 3.25. So the ladder is
already paying MORE than the flat code one step below it, and it has to beat that
step's hold to be worth anything.

G031 predicts this family fails on decode ALU regardless of the bit result --
every plane is another decode and another MAC per weight, against a 0.810
ps/element budget that q3 already blows by 5.6%. That is why the cheap offline
question is asked first: if the ladder does not win on bits, the ALU question
never has to be paid for.

  ./tools/gravity_planes_ladder.py --out receipts/ascent-2026-08-16/G033_PLANES_LADDER.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gravity_xform_hadamard import load_tensor, quantize_group, cosine, rel_fro  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCALE_BITS = 16  # one f16 per group per plane, same convention as the flat codec


def binary_planes(w, k, group):
    """Greedy residual binarization. Returns (approximation, bits_per_elem)."""
    rows, cols = w.shape
    assert cols % group == 0
    r = w.reshape(rows, cols // group, group).astype(np.float32).copy()
    approx = np.zeros_like(r)
    for _ in range(k):
        signs = np.where(r >= 0, 1.0, -1.0).astype(np.float32)
        # Optimal per-group scale for a sign plane is mean|r| in L2.
        s = np.abs(r).mean(axis=2, keepdims=True)
        approx += s * signs
        r -= s * signs
    bits = k * (1.0 + SCALE_BITS / group)
    return approx.reshape(rows, cols), bits


def flat_bits(b, group):
    return b + SCALE_BITS / group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,31,63")
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--planes", default="1,2,3,4")
    ap.add_argument("--flat-bits", default="2,3,4")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    ks = [int(x) for x in a.planes.split(",")]
    bs = [int(x) for x in a.flat_bits.split(",")]
    names = [f"language_model.model.layers.{l}.mlp.{o}.weight"
             for l in [int(x) for x in a.layers.split(",")]
             for o in ("gate_proj", "up_proj", "down_proj")]

    per_tensor, agg = [], {}
    for name in names:
        w = load_tensor(name).astype(np.float32)
        row = {"tensor": name, "shape": list(w.shape), "planes": [], "flat": []}
        for k in ks:
            ap_, bits = binary_planes(w, k, a.group)
            e = {"k": k, "bits_per_elem": bits, "hold_cosine": cosine(w, ap_),
                 "rel_fro": rel_fro(w, ap_)}
            row["planes"].append(e)
            agg.setdefault(("planes", k), []).append(e)
            del ap_
        for b in bs:
            deq, _ = quantize_group(w, b, a.group)
            e = {"bits": b, "bits_per_elem": flat_bits(b, a.group),
                 "hold_cosine": cosine(w, deq), "rel_fro": rel_fro(w, deq)}
            row["flat"].append(e)
            agg.setdefault(("flat", b), []).append(e)
            del deq
        per_tensor.append(row)
        del w
        print(f"  {name.split('layers.')[1]}")

    def mean(key, field):
        v = agg[key]
        return sum(x[field] for x in v) / len(v)

    table = []
    for k in ks:
        table.append({"scheme": f"{k} binary planes", "bits_per_elem": mean(("planes", k), "bits_per_elem"),
                      "mean_hold": mean(("planes", k), "hold_cosine"),
                      "min_hold": min(x["hold_cosine"] for x in agg[("planes", k)])})
    for b in bs:
        table.append({"scheme": f"flat q{b}", "bits_per_elem": mean(("flat", b), "bits_per_elem"),
                      "mean_hold": mean(("flat", b), "hold_cosine"),
                      "min_hold": min(x["hold_cosine"] for x in agg[("flat", b)])})
    table.sort(key=lambda r: r["bits_per_elem"])

    print(f"\n{'scheme':<18}{'bits/elem':>10}{'mean hold':>12}{'min hold':>11}")
    for r in table:
        print(f"{r['scheme']:<18}{r['bits_per_elem']:>10.4f}{r['mean_hold']:>12.6f}{r['min_hold']:>11.6f}")

    # Verdict: for each plane rung, is there a flat code costing NO MORE bits that
    # holds at least as well?
    verdicts = []
    for k in ks:
        pb = mean(("planes", k), "bits_per_elem")
        ph = mean(("planes", k), "hold_cosine")
        beaten_by = [r for r in table
                     if r["scheme"].startswith("flat")
                     and r["bits_per_elem"] <= pb + 1e-9
                     and r["mean_hold"] >= ph]
        verdicts.append({"k": k, "bits_per_elem": pb, "mean_hold": ph,
                         "beaten_by_a_cheaper_or_equal_flat_code": bool(beaten_by),
                         "beaten_by": [r["scheme"] for r in beaten_by]})
    print()
    for v in verdicts:
        if v["beaten_by_a_cheaper_or_equal_flat_code"]:
            print(f"  {v['k']} planes ({v['bits_per_elem']:.4f} b/elem, hold {v['mean_hold']:.6f}) "
                  f"LOSES to {', '.join(v['beaten_by'])}")
        else:
            print(f"  {v['k']} planes ({v['bits_per_elem']:.4f} b/elem, hold {v['mean_hold']:.6f}) "
                  f"is not beaten by any flat code at or below its width")

    all_lose = all(v["beaten_by_a_cheaper_or_equal_flat_code"] for v in verdicts)
    doc = {
        "schema": "hawking.nos.gplanes_ladder.v1",
        "obligation": "G033 -- G-PLANES progressive structured planes",
        "method": "Greedy residual binarization, per-group L2-optimal scale (mean|r|). Both sides "
                  "count their scale streams: k planes cost k*(1+16/g), flat b-bit costs b+16/g.",
        "group": a.group,
        "table": table,
        "verdicts": verdicts,
        "verdict": ("REFUTED at equal bits: every plane rung is beaten by a flat code costing no "
                    "more." if all_lose else
                    "At least one plane rung is not beaten by an equal-or-cheaper flat code."),
        "alu_note": ("Independent of the bit result, G031 flags this family on decode ALU: each "
                     "plane is another decode and another MAC per weight against a measured "
                     "0.810 ps/element budget (CODEC_ALU_COST.json). A ladder that merely tied on "
                     "bits would still lose on time."),
        "per_tensor": per_tensor,
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
