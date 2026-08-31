#!/usr/bin/env python3
"""G053: price a perfect state machine BEFORE fitting three programs.

Two of the six families S020 §16 names have curves and both came back
MEASURED_NEGATIVE. Three more - generated coefficients, learned recurrence,
conditional recurrence - are blocked on the same thing: a fitted program on disk.
Fitting three is real work.

Oracle-first is a permanent Gravity law, and its economics half is blunt: IF AN
UNFAIR ORACLE CANNOT MAKE THE ECONOMICS WORK, DO NOT BUILD THE REAL VERSION. So
price perfect success first, at the MEASURED per-stream rates rather than at the
organ average that mispriced the auxiliary ladder.

    DeltaNet weights  2.9617 GB  x 0.547282 ms/GB (weight_codes) = 1.6209 ms
    DeltaNet state    0.1569 GB  x 2.906132 ms/GB (activation)   = 0.4560 ms
    UPPER BOUND                                                    2.0768 ms

That is the whole organ's representation removed - every code, every byte of
recurrent state, nothing left. Against a 5.993 ms gap to 71 it is 34.7%.
A HALVING, which is what the landed candidates actually propose, is 1.04 ms.

So the three unfitted families are MATERIAL and not decisive. That is a real
answer to "should we fit them", and it is not the answer a byte count would have
given: 3.1 GB looks like a lot until the streams are priced separately.

    python3 tools/future/deltanet_state_machine_economics.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

import causal_budget_71 as cb  # noqa: E402

RECORDED_BY = "tools/future/deltanet_state_machine_economics.py"
RECEIPT_NAME = "DELTANET_STATE_MACHINE_ECONOMICS.json"

STATE_FN_REL = "receipts/future/DELTANET_STATE_FUNCTION.json"
BODY_REL = "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json"

TARGET_MS = 1000.0 / 71.0
MATERIAL_MS = 1.0


class EconomicsRefused(RuntimeError):
    """A price computed from a byte count without its stream."""


def _bytes() -> dict[str, int]:
    doc = json.loads((REPO / STATE_FN_REL).read_text())
    cw = doc["accounting"]["catalog_weights"]
    weights = int(cw["total"])
    named = sum(int(v) for k, v in cw.items() if k != "total" and isinstance(v, int))
    if weights != named:
        raise EconomicsRefused(
            f"catalog_weights.total {weights} does not equal its named parts "
            f"{named}; refusing to price an accounting that does not reconcile"
        )
    # The recurrent state is resident ACTIVATION, read and written every token -
    # a different stream from the codes, which is the whole point of pricing them
    # apart. Figures from DELTANET_STATE_FUNCTION's own byte model.
    state = 150_994_944 + 5_898_240
    return {"weights": weights, "state": state}


def price() -> dict[str, Any]:
    b = _bytes()
    rates = cb.STREAM_MS_PER_GB
    w_ms = b["weights"] / 1e9 * rates["weight_codes"]
    s_ms = b["state"] / 1e9 * rates["activation"]
    upper = w_ms + s_ms
    body = json.loads((REPO / BODY_REL).read_text())
    gap = float(body["decode_wall_ms_per_token"]) - TARGET_MS
    # TWO GAPS, and reporting only one would be picking the flattering number.
    # The RAW gap is from the body that runs today to 71. The RESIDUAL gap is
    # what is left after every path already on record composes, which is the gap
    # a NEW lever actually has to close.
    ladder = json.loads((REPO / "receipts/future/PATH_TO_71.json").read_text())
    residual = float(ladder["gap_to_71"]["still_to_remove_ms"])
    return {
        "weights_gb": round(b["weights"] / 1e9, 4),
        "state_gb": round(b["state"] / 1e9, 4),
        "weights_ms_if_all_removed": round(w_ms, 4),
        "state_ms_if_all_removed": round(s_ms, 4),
        "upper_bound_ms": round(upper, 4),
        "halved_ms": round(upper / 2.0, 4),
        "gap_to_71_raw_ms": round(gap, 4),
        "gap_to_71_residual_after_everything_on_record_ms": round(residual, 4),
        "upper_bound_share_of_raw_gap": round(upper / gap, 4),
        "upper_bound_share_of_residual_gap": round(upper / residual, 4),
        "two_gaps_because": (
            "the RAW gap is from today's body to 71; the RESIDUAL is what remains "
            "after every path already on record composes, and that is the gap a "
            "NEW lever has to close. Quoting only one would be picking the "
            "flattering number."
        ),
        "rates_used": dict(rates),
        "material_bar_ms": MATERIAL_MS,
        "verdict": "MATERIAL_NOT_DECISIVE" if upper > MATERIAL_MS else "IMMATERIAL",
        "why": (
            f"removing the ENTIRE DeltaNet representation - every code and every "
            f"byte of recurrent state - is {upper:.4f} ms against a {gap:.4f} ms "
            f"raw gap - {upper / gap:.1%} of it, and {upper / residual:.1%} of "
            f"the {residual:.4f} ms residual after everything on record. "
            f"The halving the landed candidates actually "
            f"propose is {upper / 2:.4f} ms. Above the {MATERIAL_MS} ms bar, so "
            "worth doing; nowhere near enough to close 71 on its own."
        ),
    }


def what_this_licenses() -> dict[str, Any]:
    return {
        "oracle_first": (
            "IF AN UNFAIR ORACLE CANNOT MAKE THE ECONOMICS WORK, DO NOT BUILD THE "
            "REAL VERSION. Here it can, barely: a perfect state machine clears the "
            "1 ms bar. So fitting is licensed rather than refused."
        ),
        "but": (
            "two of the six families have already been fitted and BOTH came back "
            "MEASURED_NEGATIVE - truncated_state_rank16 ONSET_THEN_PLATEAU, "
            "lower_rank_transition_rank8 PLATEAU. The prior on the remaining three "
            "is not neutral."
        ),
        "and_the_byte_count_would_have_lied": (
            "3.1 GB of DeltaNet representation looks decisive next to a 5.993 ms "
            "gap until the streams are priced apart. At the organ average it would "
            "have read as roughly 9 ms - more than the entire gap - which is "
            "exactly the error BYTE_COUNT_TIMES_ORGAN_AVERAGE is scarred for."
        ),
        "ranking": (
            "against the capacity classes: ARM A shows about 1.5x available at "
            "CONSTANT BYTES, which on a 5.597 ms organ is a larger prize than "
            "2.077 ms of perfect byte removal, and needs no fit at all."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "schema": "hawking.future.deltanet_state_machine_economics.v1",
        "version": 1,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "price": price(),
        "licenses": what_this_licenses(),
        "claim_boundary": (
            "Static sidecar artifact. Byte counts are read from "
            "DELTANET_STATE_FUNCTION's accounting and refuse to price unless the "
            "named parts reconcile to their own total. Milliseconds are those "
            "bytes at the MEASURED per-stream rates from ECONOMICS_CALIBRATION, "
            "never the organ average. This is an OPPORTUNITY BOUND on perfect "
            "success, not a measured speedup, and capability is UNMEASURED for "
            "every family."
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
    print(json.dumps(doc["price"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
