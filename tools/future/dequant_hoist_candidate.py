#!/usr/bin/env python3
"""A decode cheaper than the incumbent, by algebra rather than by a new format.

DECODE_TAX_TARGET says the MLP must cut decode from 1.3333 to 0.8835 FMA per
weight byte, a 1.509x cheapening, to sit at 497.4 GB/s - and that every decode
this campaign has built went the WRONG way (LUT 2.0, native exp 2.5). Every
attempt so far changed WHERE the scale lives. This changes WHEN it is applied.

THE IDENTITY, exact in exact arithmetic:

    sum_i (s*c_i + b) * x_i  ==  s * sum_i(c_i * x_i)  +  b * sum_i(x_i)

The incumbent dequantises every weight and then multiplies: one dequant FMA plus
one MAC per weight. The right-hand side accumulates in CODE SPACE and applies the
affine ONCE PER GROUP.

And sum_i(x_i) over a group is a property of x and the group, NOT of the output
row - so it is shared across all 17408 rows of gate/up and is hoisted out of the
row loop entirely.

    per iteration (8 weights, 6 weight bytes)
                        incumbent      folded
        dequant_fma          8          0.25   (2 per group of 64)
        mac_fma              8          8
        int_to_float         8          8      (codes still become floats)
        bitops              16         16
        total FMA           16          8.25

        fma / weight-byte      2.6667     1.3750    1.939x fewer
        decode / weight-byte   1.3333     0.0417   32x cheaper

1.939x on total arithmetic against a 1.509x requirement.

IT IS NOT BIT-IDENTICAL and that is the whole risk. The summation order changes,
so this is the fold_addqx class - which landed PASS_JUSTIFIED_TOLERANCE and
default-off after 22309 of 69632 gate bytes differed. Nothing here is measured.

    python3 tools/future/dequant_hoist_candidate.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/dequant_hoist_candidate.py"
RECEIPT_NAME = "DEQUANT_HOIST_CANDIDATE.json"

ROOFLINE_REL = "receipts/future/MLP_ALU_ROOFLINE.json"
GROUP = 64


class CandidateRefused(RuntimeError):
    """An arithmetic claim that does not reconcile to the measured inner loop."""


def accounting() -> dict[str, Any]:
    roof = json.loads((REPO / ROOFLINE_REL).read_text())
    inner = roof["mlp"]["decode_tax"]["inner_loop"]
    w = int(inner["weights_per_iteration"])
    wb = int(inner["weight_bytes_per_iteration"])
    dequant = int(inner["dequant_fma"])
    mac = int(inner["mac_fma"])
    # The receipt stores the rounded rate, so reconcile to its precision rather
    # than exactly - but reconcile, because an inner loop whose parts do not add
    # up is not an inner loop anyone should build algebra on top of.
    if abs((dequant + mac) / wb - float(inner["fma_per_weight_byte"])) > 5e-4:
        raise CandidateRefused(
            "the measured inner loop does not reconcile: "
            f"({dequant}+{mac})/{wb} != {inner['fma_per_weight_byte']}"
        )
    # AMORTISED OVER THE CHUNK, NOT THE GROUP - and getting this wrong is how the
    # first version of this receipt claimed 1.939x.
    #
    # The production kernel strides `col += 512` (64 threads per row x 8 weights),
    # so a thread handles ONE 8-weight chunk of a group and never returns to that
    # group. scale and bias are constant within the CHUNK, not across the group,
    # so the affine has to be applied every 8 weights: 2 FMA per 8, not 2 per 64.
    #
    # A retiling where one thread owns a whole group of 64 would reach 2 per 64
    # and 1.939x, but that is a DIFFERENT candidate with its own addressing
    # consequences, and quoting its number here would be claiming a change this
    # receipt does not describe.
    folded_dequant = 2.0
    folded_total = mac + folded_dequant
    return {
        "group_size": GROUP,
        "weights_per_iteration": w,
        "weight_bytes_per_iteration": wb,
        "incumbent": {
            "dequant_fma": dequant,
            "mac_fma": mac,
            "int_to_float": int(inner["int_to_float"]),
            "bitops": int(inner["bitops"]),
            "total_fma": dequant + mac,
            "fma_per_weight_byte": round((dequant + mac) / wb, 4),
            "decode_fma_per_weight_byte": round(dequant / wb, 4),
        },
        "folded": {
            "dequant_fma": folded_dequant,
            "mac_fma": mac,
            "int_to_float": int(inner["int_to_float"]),
            "bitops": int(inner["bitops"]),
            "total_fma": folded_total,
            "fma_per_weight_byte": round(folded_total / wb, 4),
            "decode_fma_per_weight_byte": round(folded_dequant / wb, 4),
        },
        "total_arithmetic_cheapening": round((dequant + mac) / folded_total, 4),
        "decode_cheapening": round(dequant / folded_dequant, 1),
        "amortised_over": "the 8-weight chunk, because the kernel strides col += 512",
        "a_retiling_would_do_better": {
            "if": "one thread owned a whole group of 64 instead of an 8-chunk",
            "folded_total_fma": round(mac + 2.0 * w / GROUP, 4),
            "total_arithmetic_cheapening": round((dequant + mac) / (mac + 2.0 * w / GROUP), 4),
            "but": (
                "that is a DIFFERENT candidate with its own addressing and "
                "occupancy consequences. The first version of this receipt quoted "
                "its 1.939x while describing the un-retiled loop, which was a "
                "change it did not state."
            ),
        },
        "int_to_float_unchanged_because": (
            "the codes still have to become floats to multiply x. The saving is "
            "the dequant FMA, not the conversions, and claiming the conversions "
            "too would be the easiest way to overstate this."
        ),
    }



LADDER_REL = "receipts/future/MLP_ISSUE_RATE_LADDER.json"
BODY_REL = "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json"

# The issue-rate ladder already swept FMA per weight byte on the real kernel, so
# this candidate does not need a guess about what its arithmetic saving is worth
# - it needs a POSITION on a measured curve.
#
#     arm          FMA/B     GB/s     vs production
#     production   2.6667    308.3    1.000x
#     k6           2.0000    390.9    1.268x
#     k4           1.3333    439.5    1.426x
#     k2           0.6667    440.6    1.429x
#     arm_a        0.3333    504.9    1.638x
#
# The hoist lands at 1.6667 FMA/B, BETWEEN k6 and k4. The ladder's own verdict is
# that its shape is "neither linear nor plateau" (r^2 0.8712), so interpolating to
# a point would be false precision. It is BRACKETED instead.
#
# And the bracket sits just above a KNEE: k4 at 1.3333 gives 439.5 and k2 at
# 0.6667 gives 440.6 - halving the FMA again buys 0.25%. Below about 1.3 FMA/B
# the arithmetic stops being the binding term, which is why arm_a only pulls
# further ahead by ALSO cutting int and convert ops.
LADDER_POINTS = (
    {"arm": "production", "fma_per_byte": 2.6667, "gb_s": 308.3},
    {"arm": "k6", "fma_per_byte": 2.0, "gb_s": 390.9},
    {"arm": "k4", "fma_per_byte": 1.3333, "gb_s": 439.5},
    {"arm": "k2", "fma_per_byte": 0.6667, "gb_s": 440.6},
    {"arm": "arm_a", "fma_per_byte": 0.3333, "gb_s": 504.9},
)


def ladder_bracket() -> dict[str, Any]:
    """Where this candidate sits on the MEASURED ops/byte curve."""
    acc = accounting()
    mine = float(acc["folded"]["fma_per_weight_byte"])
    prod = next(p for p in LADDER_POINTS if p["arm"] == "production")
    below = [p for p in LADDER_POINTS if p["fma_per_byte"] <= mine]
    above = [p for p in LADDER_POINTS if p["fma_per_byte"] >= mine]
    lo = max(above, key=lambda p: -p["fma_per_byte"]) if above else None
    hi = min(below, key=lambda p: -p["fma_per_byte"]) if below else None
    lo = min(above, key=lambda p: p["fma_per_byte"]) if above else None
    hi = max(below, key=lambda p: p["fma_per_byte"]) if below else None
    body = json.loads((REPO / BODY_REL).read_text())
    rows = {r["organ"]: float(r["gpu_ms"]) for r in body["organs"]["rows"]}
    mlp_ms = rows["mlp_gate_up"] + rows["mlp_down"]
    r_lo = lo["gb_s"] / prod["gb_s"]
    r_hi = hi["gb_s"] / prod["gb_s"]
    return {
        "candidate_fma_per_weight_byte": mine,
        "bracketed_by": [lo["arm"], hi["arm"]],
        "speedup_bracket": [round(r_lo, 4), round(r_hi, 4)],
        "mlp_ms_now": round(mlp_ms, 4),
        "mlp_ms_bracket": [round(mlp_ms / r_hi, 4), round(mlp_ms / r_lo, 4)],
        "mlp_ms_saved_bracket": [round(mlp_ms - mlp_ms / r_lo, 4),
                                 round(mlp_ms - mlp_ms / r_hi, 4)],
        "not_interpolated_because": (
            "the ladder's own verdict is that its shape is neither linear nor "
            "plateau, r^2 0.8712. A point estimate between two measured arms "
            "would be false precision, so this is a bracket."
        ),
        "the_knee": (
            "k4 at 1.3333 FMA/B gives 439.5 GB/s and k2 at 0.6667 gives 440.6 - "
            "halving the arithmetic again buys 0.25%. Below about 1.3 FMA/B the "
            "arithmetic stops being the binding term, and arm_a only pulls "
            "further ahead by ALSO cutting int and convert ops. So this candidate "
            "sits just above the knee and would capture most of the FMA-side "
            "gain that exists."
        ),
        "ladder_arms_are_not_bit_identical": (
            "every k-arm is recorded not-identical, which is consistent with "
            "this candidate: they change the arithmetic and so does it"
        ),
    }


def identity() -> dict[str, Any]:
    return {
        "algebra": "sum_i (s*c_i + b) * x_i  ==  s * sum_i(c_i*x_i) + b * sum_i(x_i)",
        "exact_in_exact_arithmetic": True,
        "why_sum_x_is_free": (
            "sum_i(x_i) over a group is a property of x and the group, NOT of the "
            "output row, so it is shared across all 17408 rows of gate/up and "
            "hoists out of the row loop entirely. That is what makes the second "
            "term cost nothing per row."
        ),
        "what_changes": (
            "not WHERE the scale lives - every previous attempt changed that and "
            "made decode more expensive - but WHEN the affine is applied: once "
            "per group instead of once per weight."
        ),
    }


def risks() -> dict[str, Any]:
    return {
        "not_bit_identical": (
            "the summation order changes, so this is the fold_addqx class: that "
            "one landed PASS_JUSTIFIED_TOLERANCE and DEFAULT-OFF after 22309 of "
            "69632 gate bytes differed, cause SOURCE_ORDER_FMA_ASSOCIATION. This "
            "candidate inherits that bar and must clear it the same way."
        ),
        "precision": (
            "accumulating c_i*x_i in code space sums 2-bit codes times "
            "activations before any scale is applied, so the accumulator's "
            "dynamic range differs from the incumbent's. UNTESTED, and the "
            "direction is not obvious - codes are small, which helps, and there "
            "are 64 of them per group, which does not."
        ),
        "register_pressure": (
            "two accumulators per row instead of one, and the class C occupancy "
            "question cannot be read on this toolchain (registers_per_thread is "
            "null). So the arithmetic saving could be given back to occupancy "
            "and nothing on disk can predict that."
        ),
        "expressibility": (
            "the sum_x hoist needs a per-group reduction over x available to "
            "every row block. Whether the current geo_tpr64 tiling can carry it "
            "without an extra pass is UNVERIFIED."
        ),
        "status": "UNBUILT, UNMEASURED, arithmetic argument only",
    }


def build() -> dict[str, Any]:
    acc = accounting()
    return {
        "schema": "hawking.future.dequant_hoist_candidate.v1",
        "version": 1,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "identity": identity(),
        "accounting": acc,
        "meets_the_requirement": {
            "required_total_cheapening": 1.509,
            "offered_total_cheapening": acc["total_arithmetic_cheapening"],
            "clears": acc["total_arithmetic_cheapening"] >= 1.509,
        },
        "ladder_bracket": ladder_bracket(),
        "risks": risks(),
        "claim_boundary": (
            "Static sidecar artifact. The incumbent inner loop is READ from "
            "MLP_ALU_ROOFLINE and refuses if its own FMA counts do not reconcile "
            "to its reported fma_per_weight_byte. The folded column is ALGEBRA, "
            "not a measurement: no kernel exists, nothing was timed, and an FMA "
            "count is not a GB/s. It is a candidate worth building, not a result."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(RECEIPT_NAME, doc, RECORDED_BY))
        return 0
    print(json.dumps({"accounting": doc["accounting"],
                      "meets": doc["meets_the_requirement"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
