#!/usr/bin/env python3
"""G039: entropy coding is cheap in bits and expensive in tile-aligned random access.

The acceptance asks for the measured entropy of the stored format AND a decode
path aligned to the kernel's tile access, then verifies on bits/weight after
lossless coding PLUS the random-access decode cost per tile. Those two halves
pull in opposite directions and the campaign has only ever measured the first.

The measured half, from G024_RATE_DISTORTION_BOUND: the q3 symbol stream carries
2.2964 bits of order-0 entropy against a 3.0-bit nominal width, and specialized
order-1 context models find at most 0.0013 more. So lossless coding genuinely
buys ~0.70 bits/elem of code, before scales.

The unmeasured half is what that costs to consume. An entropy coder is sequential
by construction: symbol k cannot be decoded without the decoder state after
symbol k-1. The geo_tpr64 GEMV is the opposite -- every thread jumps to its own
offset and reads 8 weights, and 401 of the genome's GEMVs depend on that geometry.
Reconciling them costs one of two things, and this prices both against the
MEASURED 0.8092 ps/element ALU budget from CODEC_ALU_COST.

  ENTRY POINTS   store a resumable state per access point. Fewer weights per
                 entry point means more index bits.
  AMPLIFICATION  decode forward from the nearest entry point to reach the
                 weights actually wanted. Coarser entry points mean more wasted
                 symbol decodes per useful weight.

  ./tools/gravity_tile_entropy_feasibility.py --out receipts/ascent-2026-08-16/G039_TILE_ENTROPY.json
"""
from __future__ import annotations
import argparse, json, math, pathlib, subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]

# geo_tpr64_tg128, read off the kernel source. Each thread walks
#   for (col = lane_in_row * 8; col < cols; col += 512)
# so it consumes 8 contiguous weights per access, 64 lanes per row.
WEIGHTS_PER_ACCESS = 8
COLS = 5120
NOMINAL_BITS = 3.0
SCALE_BITS_PER_ELEM = 16.0 / 64          # one f16 per group of 64
Q3_TOTAL_BITS = NOMINAL_BITS + SCALE_BITS_PER_ELEM   # 3.25, the incumbent
CODED_BITS = 2.2964                      # measured order-0 entropy of q3 symbols
ALU_BUDGET_PS = 0.8092                   # measured, CODEC_ALU_COST.json
Q3_UNPACK_PS = 0.867                     # measured: q3 already FAILS the budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-bits", type=int, default=14,
                    help="bits to store one resumable entry point. 14 is generous: a bit offset "
                         "into a row's coded stream needs ~13.5, and that assumes the coder state "
                         "is reset at each entry point, which itself costs coding efficiency.")
    ap.add_argument("--out", type=pathlib.Path)
    a = ap.parse_args()

    rows = []
    for wpe in (8, 16, 32, 64, 128, 256, 512, COLS):
        entry_points = COLS / wpe
        index_bits_per_elem = entry_points * a.state_bits / COLS
        total_bits = CODED_BITS + index_bits_per_elem + SCALE_BITS_PER_ELEM
        # A thread wants 8 contiguous weights starting at a uniformly random
        # offset inside its entry-point span, and must decode from the entry
        # point forward to reach them.
        mean_start = max(0.0, (wpe - WEIGHTS_PER_ACCESS) / 2.0)
        symbols_decoded = mean_start + WEIGHTS_PER_ACCESS
        amplification = symbols_decoded / WEIGHTS_PER_ACCESS
        rows.append({
            "weights_per_entry_point": wpe,
            "entry_points_per_row": entry_points,
            "index_bits_per_elem": index_bits_per_elem,
            "total_bits_per_elem": total_bits,
            "beats_q3_on_bits": total_bits < Q3_TOTAL_BITS,
            "mean_symbols_decoded_per_8_useful": symbols_decoded,
            "decode_amplification": amplification,
            # Even granting rANS a decode as cheap as a q3 nibble unpack -- which
            # it is not, being a multiply, a renormalize and a table lookup
            # against a shift and a mask -- amplification alone multiplies it.
            "optimistic_ps_per_elem": Q3_UNPACK_PS * amplification,
            "clears_alu_budget": Q3_UNPACK_PS * amplification <= ALU_BUDGET_PS,
        })

    both = [r for r in rows if r["beats_q3_on_bits"] and r["clears_alu_budget"]]
    print(f"measured coded entropy {CODED_BITS:.4f} b/elem against {NOMINAL_BITS:.1f} nominal; "
          f"incumbent q3 total {Q3_TOTAL_BITS:.4f} b/elem")
    print(f"measured ALU budget {ALU_BUDGET_PS:.4f} ps/elem; q3's own unpack already costs "
          f"{Q3_UNPACK_PS:.3f} and FAILS it\n")
    print(f"{'w/entry':>8}{'index b/e':>11}{'total b/e':>11}{'bits win':>10}"
          f"{'amplify':>9}{'ps/elem*':>10}{'ALU ok':>8}")
    for r in rows:
        print(f"{r['weights_per_entry_point']:>8}{r['index_bits_per_elem']:>11.4f}"
              f"{r['total_bits_per_elem']:>11.4f}{str(r['beats_q3_on_bits']):>10}"
              f"{r['decode_amplification']:>9.2f}{r['optimistic_ps_per_elem']:>10.3f}"
              f"{str(r['clears_alu_budget']):>8}")
    print("\n* optimistic: charges rANS the SAME per-symbol cost as a q3 nibble unpack, which is "
          "generous by construction.")
    print(f"\nconfigurations that beat q3 on bits AND clear the ALU budget: {len(both)}")

    doc = {
        "schema": "hawking.nos.tile_entropy_feasibility.v1",
        "obligation": "G039 -- lossless redundancy layer with a decode path aligned to tile access",
        "kernel_geometry": {
            "kernel": "qwen_uniform_q4_group64_matvec_geo_tpr64_tg128 and its q3 sibling",
            "access": "each thread reads 8 contiguous weights at col = lane*8 + k*512, 64 lanes "
                      "per row. Read from the kernel source, not assumed.",
            "why_it_matters": "401 of the genome's GEMVs use this geometry and F6 measured it at "
                              "88% of the bandwidth roof, i.e. on the bandwidth/ALU balance point.",
        },
        "measured_inputs": {
            "coded_bits_per_elem": CODED_BITS,
            "coded_bits_source": "order-0 entropy of the real q3 symbol stream; order-1 context "
                                 "models add at most 0.0013 (G024_BOUND_REVIEW.json)",
            "incumbent_total_bits_per_elem": Q3_TOTAL_BITS,
            "alu_budget_ps_per_elem": ALU_BUDGET_PS,
            "q3_unpack_ps_per_elem": Q3_UNPACK_PS,
        },
        "state_bits_per_entry_point": a.state_bits,
        "sweep": rows,
        "configurations_winning_both": len(both),
        "verdict": (
            "REFUTED FOR THIS KERNEL GEOMETRY. The two halves of the acceptance cannot be "
            "satisfied at once. Fine entry points keep decode cheap but the index costs more bits "
            "than the entropy coding saves; coarse entry points keep the bits but multiply decode "
            "by the amplification factor, and q3's far simpler unpack ALREADY fails the ALU budget "
            "at amplification 1.0. There is no crossing point, and the sweep is charged "
            "optimistically at every step."),
        "what_would_change_it": [
            "A kernel geometry whose threads consume CONTIGUOUS runs rather than strided 8-weight "
            "windows would cut amplification to ~1. That means retiling the GEMV all 401 GEMVs "
            "depend on, against a kernel already at 88% of the bandwidth roof.",
            "A codec whose decode is genuinely cheaper per symbol than a q3 nibble unpack. rANS is "
            "not: a state multiply, a renormalize and a table lookup against a shift and a mask.",
            "Bandwidth becoming the binding resource again -- which would require the ALU side to "
            "get faster, not the byte side to get smaller.",
        ],
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=ROOT).stdout.strip(),
    }
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
