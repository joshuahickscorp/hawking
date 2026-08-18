#!/usr/bin/env python3
"""G125: the cost vector gains its sixth axis, and the ranking rule gains teeth.

G041 emitted (B, M, F, L, R) and carried T as null-with-reason, because no Tabula
instrument existed. One exists now (G123), so T can be filled -- and filling it
changes the ranking, which is the whole reason the axis was demanded.

THE RULE: a candidate that improves B while worsening T CANNOT be ranked ahead
silently. Not "is penalised", not "is weighted" -- the comparison is REFUSED and the
regression is named, because a weighting would let a large B win purchase a T
regression at some exchange rate, and no such rate has ever been established.

T IS PROVISIONAL AND SAYS SO. G123 recovers the abliterated direction with a 200x
null separation, and its drift ladder misses the recorded values by a constant 2.5x.
Every T below carries that flag; a provisional number that does not announce itself
is worse than a null.

  ./tools/cost_vector_t.py --out receipts/.../G125_COST_VECTOR_T.json
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
# B, M, L measured in G041; T from the codec each artifact's body uses, measured in G123.
CAND = {
  "uniform-q4-v1":               {"B": 4.255955, "M": 14_297_694_680, "L_ns": 30_549_917,
                                  "tps": 32.73, "body_codec": "q4_group64"},
  "compact-q3attn-r1p2-v1":      {"B": 3.344821, "M": 11_245_158_827, "L_ns": 33_586_042,
                                  "tps": 29.77, "body_codec": "q3_group64"},
  "g032-chanscale-a025-compact": {"B": 3.344826, "M": 11_245_158_827, "L_ns": 33_593_042,
                                  "tps": 29.77, "body_codec": "q3_group64"},
}
TABULA = {"q4_group64": 28.63, "q3_group64": 67.55, "q2_group64": 162.08}
GEMV_ELEMS = 25_622_424_508
KV_PER_POS = 131_072


def vector(name, c, seq=128):
    return {"candidate": name,
            "B_bits_per_weight": c["B"], "M_dram_bytes_per_token": c["M"],
            "F_flops_per_token": 2 * GEMV_ELEMS, "L_ns_per_token": c["L_ns"], "L_tps": c["tps"],
            "R_resident_bytes": c["M"] + seq * KV_PER_POS,
            "T_tabula_drift_x": TABULA[c["body_codec"]],
            "T_provisional": True,
            "T_basis": f"{c['body_codec']} measured on the abliterated full_attention_out at L63 "
                       "(G123). PROVISIONAL: that instrument's ladder misses the recorded "
                       "11-12/25-27/64-67 by a constant 2.5x."}


def compare(a, b):
    """Returns (verdict, notes). Lower is better on every axis."""
    dB = b["B_bits_per_weight"] - a["B_bits_per_weight"]
    dT = b["T_tabula_drift_x"] - a["T_tabula_drift_x"]
    dL = b["L_ns_per_token"] - a["L_ns_per_token"]
    notes = [f"B {a['B_bits_per_weight']:.6f} -> {b['B_bits_per_weight']:.6f} ({dB:+.6f})",
             f"T {a['T_tabula_drift_x']:.2f}x -> {b['T_tabula_drift_x']:.2f}x ({dT:+.2f})",
             f"L {a['L_ns_per_token']/1e6:.3f} -> {b['L_ns_per_token']/1e6:.3f} ms ({dL/1e6:+.3f})"]
    if dB < 0 and dT > 0:
        return "REFUSED", notes + [
            "B IMPROVES AND T REGRESSES. This comparison is refused, not weighted: a weighting "
            "would let a large enough B win purchase a T regression at some exchange rate, and no "
            "such rate has ever been established between bits and patient identity."]
    if dB < 0 and dT <= 0 and dL <= 0:
        return "B_DOMINATES", notes
    return "NO_SILENT_RANK", notes + ["mixed axes; no single-axis ordering applies"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()
    vecs = {n: vector(n, c) for n, c in CAND.items()}
    print(f"{'candidate':<32}{'B':>10}{'L ms':>9}{'TPS':>7}{'T drift':>10}{'R GB':>8}")
    for n, v in vecs.items():
        print(f"{n:<32}{v['B_bits_per_weight']:>10.6f}{v['L_ns_per_token']/1e6:>9.3f}"
              f"{v['L_tps']:>7.2f}{v['T_tabula_drift_x']:>9.2f}x{v['R_resident_bytes']/1e9:>8.2f}")
    base = vecs["uniform-q4-v1"]
    results = []
    print()
    for n, v in vecs.items():
        if n == "uniform-q4-v1":
            continue
        verdict, notes = compare(base, v)
        results.append({"from": "uniform-q4-v1", "to": n, "verdict": verdict, "notes": notes})
        print(f"uniform-q4-v1 -> {n}: {verdict}")
        for x in notes:
            print(f"    {x}")

    # The constructed T-regression the verify asks for: B improves a lot, T worsens a little.
    synth = dict(base); synth = json.loads(json.dumps(base))
    synth["candidate"] = "CONSTRUCTED: half the bits, 1% worse T"
    synth["B_bits_per_weight"] = base["B_bits_per_weight"] / 2
    synth["T_tabula_drift_x"] = base["T_tabula_drift_x"] * 1.01
    v2, n2 = compare(base, synth)
    results.append({"from": "uniform-q4-v1", "to": synth["candidate"], "verdict": v2, "notes": n2})
    print(f"\nCONSTRUCTED T-REGRESSION (B halved, T worse by 1%): {v2}")
    for x in n2:
        print(f"    {x}")

    doc = {"schema": "hawking.nos.cost_vector_t.v1",
           "obligation": "G125 -- cost vector extended to (B, M, F, L, R, T)",
           "axes": {"B": "stored bits per weight, complete", "M": "DRAM bytes per token",
                    "F": "physical flops per token", "L": "ns per token, measured on device",
                    "R": "resident bytes at the harness sequence length",
                    "T": "Tabula drift, x the parent, from G123"},
           "candidates": vecs, "comparisons": results,
           "the_rule": ("a candidate that improves B while worsening T is REFUSED, not weighted. A "
                        "weighting would let a large enough B win purchase a T regression at some "
                        "exchange rate, and no such rate has ever been established between bits and "
                        "patient identity."),
           "T_is_provisional": ("G123 recovers the direction with a 200x null separation but its "
                                "ladder misses the recorded values by a constant 2.5x. Every T here "
                                "carries the flag; a provisional number that does not announce "
                                "itself is worse than a null."),
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, cwd=ROOT).stdout.strip()}
    if a.out:
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
