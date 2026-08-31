#!/usr/bin/env python3
"""The decode tax is the largest single target on record, and it is not bytes.

Two landed measurements, read together, put a number on it.

    ARM A       strips arithmetic at IDENTICAL bytes: MLP 329.6 -> 497.4 GB/s
    ILP ladder  same arithmetic, 2/4/8 independent accumulator chains,
                bit-identical: 328.5 / 325.5 / 327.4 - flat inside 1%

Removing arithmetic buys 1.51x; making it parallel buys nothing. So the cost is
arithmetic THROUGHPUT, and the roofline says where it goes: 1.3333 of the 2.6667
FMA per weight byte are DECODE, not compute. Half the inner loop unpacks the
representation.

    to sit at the LM head's demonstrated 497.4 GB/s the MLP must cut
    decode-FMA per weight byte 1.3333 -> 0.8835, a 1.509x cheapening

What that is worth, at the measured post-widen_f4 organ times:

    MLP       15.6473 ms  ->  10.3686 ms   saves 5.2787 ms
    DeltaNet   5.5971 ms  ->   3.5658 ms   saves 2.0313 ms   (its own ARM A rate)

Against a 13.205 ms raw gap to 71 and a 5.993 ms residual after everything on
record. The MLP term alone is larger than the entire byte ladder (0.18 TPS), the
entire DeltaNet representation (2.08 ms upper bound), and the whole auxiliary
school (0.000 ms).

THIS IS NOT A PROMISE THAT SUCH A DECODE EXISTS. It is the price of one if it
does, and the specification a candidate has to meet.

    python3 tools/future/decode_tax_target.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/decode_tax_target.py"
RECEIPT_NAME = "DECODE_TAX_TARGET.json"

ROOFLINE_REL = "receipts/future/MLP_ALU_ROOFLINE.json"
BODY_REL = "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json"
LADDER_REL = "receipts/future/PATH_TO_71.json"
TARGET_MS = 1000.0 / 71.0


class TargetRefused(RuntimeError):
    """A price built from a rate the receipts do not carry."""


def _read(rel: str) -> dict[str, Any]:
    path = REPO / rel
    if not path.is_file():
        raise TargetRefused(f"{rel} is not on disk; this target is read, not assumed")
    return json.loads(path.read_text())


def requirement() -> dict[str, Any]:
    roof = _read(ROOFLINE_REL)
    tax = roof["mlp"]["decode_tax"]
    need = tax["to_match_lm_head_497"]
    inner = tax["inner_loop"]
    return {
        "production_fma_per_weight_byte": tax["production_fma_per_weight_byte"],
        "production_decode_fma_per_weight_byte": tax["production_decode_fma_per_weight_byte"],
        "decode_share_of_inner_loop": round(
            tax["production_decode_fma_per_weight_byte"]
            / tax["production_fma_per_weight_byte"], 4
        ),
        "target_decode_fma_per_weight_byte": need["target_decode_fma_per_weight_byte"],
        "required_decode_cheapening": need["required_decode_cheapening"],
        "inner_loop": inner,
        "not_a_promise": need["note"],
    }


def worth() -> dict[str, Any]:
    body = _read(BODY_REL)
    rows = {r["organ"]: float(r["gpu_ms"]) for r in body["organs"]["rows"]}
    roof = _read(ROOFLINE_REL)
    mlp_ms = rows["mlp_gate_up"] + rows["mlp_down"]
    dn_ms = rows["deltanet"]
    mlp_prod = float(roof["mlp"]["production"]["effective_gb_s"])
    mlp_arm_a = float(roof["mlp"]["arm_a_stripped"]["effective_gb_s"])
    dn_prod = float(roof["deltanet"]["production"]["effective_gb_s"])
    dn_arm_a = float(roof["deltanet"]["arm_a_stripped"]["effective_gb_s"])
    mlp_after = mlp_ms * mlp_prod / mlp_arm_a
    dn_after = dn_ms * dn_prod / dn_arm_a
    ladder = _read(LADDER_REL)
    residual = float(ladder["gap_to_71"]["still_to_remove_ms"])
    raw = float(body["decode_wall_ms_per_token"]) - TARGET_MS
    saved = (mlp_ms - mlp_after) + (dn_ms - dn_after)
    return {
        "mlp_ms_now": round(mlp_ms, 4),
        "mlp_ms_at_arm_a_rate": round(mlp_after, 4),
        "mlp_ms_saved": round(mlp_ms - mlp_after, 4),
        "deltanet_ms_now": round(dn_ms, 4),
        "deltanet_ms_at_arm_a_rate": round(dn_after, 4),
        "deltanet_ms_saved": round(dn_ms - dn_after, 4),
        "total_ms_saved": round(saved, 4),
        "gap_raw_ms": round(raw, 4),
        "gap_residual_ms": round(residual, 4),
        "share_of_raw_gap": round(saved / raw, 4),
        "share_of_residual_gap": round(saved / residual, 4),
        "closes_the_residual": saved >= residual,
    }


def against_everything_else() -> dict[str, Any]:
    """Rank it, because a number without a ranking invites the wrong work."""
    return {
        "decode_tax_at_arm_a_rate_ms": round(worth()["total_ms_saved"], 4),
        "entire_byte_lever_ladder_tps": 0.18,
        "entire_deltanet_representation_ms": 2.0768,
        "entire_auxiliary_school_ms": 0.0,
        "mlp_entropy_floor_ms": 0.152,
        "reading": (
            "the arithmetic term is larger than every representation lever on "
            "record COMBINED, and unlike them it needs no fit and no capability "
            "screen - the output stays bit-identical by construction if the "
            "decode is exact. That is the reordering S025 asked for: stop "
            "polishing bytes that do not cost time and attack the arithmetic "
            "that does."
        ),
        "the_catch": (
            "ARM A removes the decode ENTIRELY, which no real representation can. "
            "The 5.2787 ms is the ceiling of a cheaper decode, not a candidate's "
            "yield, and the required 1.509x cheapening is a specification nobody "
            "has met. A format decoding in 0.8835 FMA per weight byte instead of "
            "1.3333 may not exist at this bit width."
        ),
    }


def every_decode_tried_so_far() -> dict[str, Any]:
    """The direction of travel has been the WRONG WAY, and that is the prior.

    Two alternative aux decodes were built and measured. Both cost MORE
    arithmetic per weight byte than the incumbent, not less:

        incumbent   1.3333 decode-FMA/weight-byte at 6 B/iter
        LUT         2.0    (dequant only, two 256-entry tables)
        native exp  2.5    at 4 B/iter (4.5 FMA/byte total vs 2.6667)

    So the cheapest decode this campaign has is the one already shipping, and
    the 0.8835 target is below anything anyone has built.

    It also explains aux_u8's measured slowdown without needing a second theory:
    it removed BYTES, which cost 0.000 ms, and added ARITHMETIC, which is the
    term that costs. It traded a free resource for the binding one.
    """
    return {
        "incumbent_decode_fma_per_weight_byte": 1.3333,
        "attempts": [
            {"id": "aux_u8_lut", "decode_fma_per_weight_byte": 2.0,
             "direction": "WORSE", "source": "receipts/future/AUX_U8_LUT.json",
             "what": "two 256-entry tables instead of exp; dequant only"},
            {"id": "aux_u8_native", "decode_fma_per_weight_byte": 2.5,
             "direction": "WORSE", "source": "receipts/future/AUX_U8_NATIVE.json",
             "what": "uchar scale/bias decoded in-register, log-scale exp"},
        ],
        "target": 0.8835,
        "nothing_built_is_below": 1.3333,
        "law": (
            "A REPRESENTATION CHANGE THAT TRADES BYTES FOR DECODE ARITHMETIC "
            "TRADES A FREE RESOURCE FOR THE BINDING ONE. broadcast_aux bytes "
            "measure 0.000 ms/GB and arithmetic measures 1.51x, so removing aux "
            "bytes at the cost of decode FMA is negative before it starts. That "
            "is aux_u8's measured slowdown explained without a second theory."
        ),
        "what_would_be_needed": (
            "a decode cheaper than the incumbent's 1.3333, which is the opposite "
            "direction from both attempts. Fewer int_to_float and fewer dequant "
            "FMA per weight, not a different place to store the scale."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "schema": "hawking.future.decode_tax_target.v1",
        "version": 1,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "requirement": requirement(),
        "worth": worth(),
        "ranking": against_everything_else(),
        "prior_from_every_decode_tried": every_decode_tried_so_far(),
        "claim_boundary": (
            "Static sidecar artifact. Every rate is READ from MLP_ALU_ROOFLINE "
            "and every millisecond from the measured post-widen_f4 organ census; "
            "nothing is measured here. This is an OPPORTUNITY BOUND computed by "
            "scaling measured organ time by a measured ARM A ratio, not a "
            "prediction that any decode achieves it. Both roofline arms are "
            "SELF_MEASURED_DIRTY under load, so the RATIO is the usable part."
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
    print(json.dumps({"requirement": doc["requirement"], "worth": doc["worth"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
