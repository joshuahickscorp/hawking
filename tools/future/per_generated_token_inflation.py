#!/usr/bin/env python3
"""Every `*_per_generated_token` field in the resident is inflated by (P+N)/G.

The resident divides totals that span PREFILL AND DECODE by the number of
GENERATED tokens. With a 12-token prompt and 128 generated tokens there are
12 + 127 = 139 forward passes but only 128 generated tokens, so every such
field reads 139/128 = 1.085938x too high.

This is the same arithmetic that L42 found in complete-vs-decode TPS, showing
up a second time in the byte and dispatch ledgers instead of in the timing. It
is worth naming as a class rather than fixing one field, because it silently
corrupts any denominator built from these numbers — and it did: it briefly
looked like an 8.6% correction to the active-byte anchor and moved the whole
milestone ladder.

Confirmed to 7 ppm against the catalog census, and to the last digit on
dispatches:

    reported dispatches_per_generated_token   1046.84375
    964 x 139/128                             1046.84375

The `decode` sub-block is NOT affected: it divides decode-only totals by decode
steps and is the field to read.

    python3 tools/future/per_generated_token_inflation.py --record
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402

RECEIPT = REPO / "receipts" / "future" / "PER_GENERATED_TOKEN_INFLATION.json"
SOURCE = "crates/hawking-core/examples/ascension_qwen38_resident.rs"

# The observation the correction is derived from.
PROMPT_TOKENS = 12
GENERATED_TOKENS = 128
DECODE_STEPS = 127
FORWARD_PASSES = PROMPT_TOKENS + DECODE_STEPS      # 139

CATALOG_ACTIVE_BYTES_PER_PASS = 9_878_901_136      # from MLP_BYTE_CENSUS.json
REPORTED_ACTIVE_BYTES = 10_727_793_881.75
REPORTED_DISPATCHES = 1046.84375
TRUE_DISPATCHES_UNFUSED = 964.0
TRUE_DISPATCHES_FUSED = 628.0

AFFECTED_FIELDS = (
    "active_bytes_per_token",
    "active_weight_bytes_per_generated_token",
    "dispatches_per_generated_token",
    "gpu_ns_per_generated_token",
    "complete_wall_ns_per_generated_token",
)
UNAFFECTED = (
    "metrics.decode.dispatches / metrics.decode.steps",
    "metrics.decode.wall_ns / metrics.decode.steps",
    "metrics.decode.gpu_ns / metrics.decode.steps",
    "decode_tps",
)


def factor() -> float:
    return FORWARD_PASSES / GENERATED_TOKENS


def correct(reported: float) -> float:
    """Undo the inflation: reported x G / (P + N)."""
    return reported / factor()


def checks() -> list[dict[str, Any]]:
    f = factor()
    predicted_bytes = CATALOG_ACTIVE_BYTES_PER_PASS * f
    return [
        {
            "field": "active_bytes_per_token",
            "reported": REPORTED_ACTIVE_BYTES,
            "predicted_if_inflated": round(predicted_bytes, 2),
            "residual_ppm": round(
                (REPORTED_ACTIVE_BYTES / predicted_bytes - 1) * 1e6, 1),
            "corrected": round(correct(REPORTED_ACTIVE_BYTES)),
            "residual_bytes": round(correct(REPORTED_ACTIVE_BYTES))
                              - CATALOG_ACTIVE_BYTES_PER_PASS,
            "residual_note": (
                "about 69 KB, and NOT rounding. The resident reports "
                "resident_weight_bytes 10,554,259,456 against a catalog stored "
                "total of 10,554,328,856 — a 69,400 byte gap of the same size. "
                "Something small (headers, a metadata tensor) is counted by one "
                "and not the other. Left OPEN and visible rather than rounded "
                "into agreement."
            ),
            "independent_ground_truth": CATALOG_ACTIVE_BYTES_PER_PASS,
            "ground_truth_source": "receipts/future/MLP_BYTE_CENSUS.json — a "
                                   "per-tensor sum over the HQ38M20 catalog, "
                                   "derived without reference to the resident",
            "verdict": "CONFIRMED",
        },
        {
            "field": "dispatches_per_generated_token",
            "reported": REPORTED_DISPATCHES,
            "predicted_if_inflated": TRUE_DISPATCHES_UNFUSED * f,
            "residual_ppm": 0.0,
            "corrected": round(correct(REPORTED_DISPATCHES), 5),
            "independent_ground_truth": TRUE_DISPATCHES_UNFUSED,
            "ground_truth_source": "metrics.decode.dispatches / decode steps, "
                                   "and independently a static encode-path walk "
                                   "in tools/future/tps_budget.py",
            "verdict": "CONFIRMED_EXACTLY",
        },
    ]


def corrected_anchors() -> dict[str, Any]:
    return {
        "active_weight_bytes_per_token": CATALOG_ACTIVE_BYTES_PER_PASS,
        "active_gb_per_token": round(CATALOG_ACTIVE_BYTES_PER_PASS / 1e9, 4),
        "production_dispatches_per_token": TRUE_DISPATCHES_FUSED,
        "unfused_dispatches_per_token": TRUE_DISPATCHES_UNFUSED,
        "roof_tps_at_clean_gemv_703_5": round(
            703.5 / (CATALOG_ACTIVE_BYTES_PER_PASS / 1e9), 2),
        "roof_tps_at_published_peak_819": round(
            819.0 / (CATALOG_ACTIVE_BYTES_PER_PASS / 1e9), 2),
        "byte_ms_at_clean_gemv": round(
            CATALOG_ACTIVE_BYTES_PER_PASS / 1e9 / 703.5 * 1000, 3),
        "byte_share_of_28_44ms_token": round(
            CATALOG_ACTIVE_BYTES_PER_PASS / 1e9 / 703.5 * 1000 / 28.44, 4),
        "implied_production_bandwidth_gb_s": round(
            CATALOG_ACTIVE_BYTES_PER_PASS / 1e9 / (28.44 / 1000), 1),
    }


def build() -> dict[str, Any]:
    a = corrected_anchors()
    return {
        "schema": "hawking.future.per_generated_token_inflation.v1",
        "version": 1,
        "recorded_by": "tools/future/per_generated_token_inflation.py",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "defect": {
            "class": "PER_GENERATED_TOKEN_DIVIDES_BY_THE_WRONG_DENOMINATOR",
            "source": SOURCE,
            "code": "active_weight_bytes_total / generated_count, and "
                    "dispatches / generated_count",
            "problem": "the numerators span prefill AND decode; the denominator "
                       "counts generated tokens only",
            "inflation_factor": round(factor(), 6),
            "for_this_run": f"P={PROMPT_TOKENS}, N={DECODE_STEPS}, "
                            f"G={GENERATED_TOKENS}, passes={FORWARD_PASSES}",
            "grows_with": "prompt length; a long prompt inflates these fields "
                          "without bound, and a very long generation hides it",
        },
        "affected_fields": list(AFFECTED_FIELDS),
        "unaffected_fields": list(UNAFFECTED),
        "checks": checks(),
        "corrected_anchors": a,
        "what_this_reverses": {
            "claim": "active bytes per token are 10.73 GB, so the clean-GEMV "
                     "roof is 65.58 TPS and 71 TPS requires bytes to fall",
            "made_in": "receipts/future/RESIDENT_TOKEN_BUDGET.json v2 and the "
                       "commits that recorded it",
            "correction": f"active bytes per token are "
                          f"{CATALOG_ACTIVE_BYTES_PER_PASS:,}, the roof is "
                          f"{a['roof_tps_at_clean_gemv_703_5']} TPS, and 71 TPS "
                          "is back below the roof and reachable by executor "
                          "recovery alone",
            "why_it_was_believed": "the resident is the authority on its own "
                                   "byte ledger, and its field is named "
                                   "active_bytes_per_token",
            "what_caught_it": "an independent per-tensor catalog census that "
                              "reconciled to 9,878,901,136 and disagreed",
        },
        "law": (
            "The same prefill-over-generated-tokens arithmetic that L42 found in "
            "complete-vs-decode TPS is present in the byte and dispatch ledgers. "
            "When one accounting artifact is found, look for the same "
            "denominator everywhere else before trusting any figure that shares "
            "it. A field named per_token is not per_token unless its numerator "
            "and denominator count the same events."
        ),
        "fix_in_the_resident": (
            "Either divide by forward passes, or rename these fields to say "
            "what they measure (per_generated_token_including_prefill_work). "
            "Not applied here: the resident source is not this module's to "
            "change, and the correction factor is recorded so every consumer "
            "can undo it."
        ),
        "claim_boundary": (
            "Arithmetic over one recorded run plus an independent catalog "
            "census. The inflation factor is exact for that run's P, N and G "
            "and must be recomputed for any other run. No hardware measurement "
            "is asserted."
        ),
    }


def record() -> Path:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(build(), indent=1, sort_keys=True, default=str) + "\n")
    return RECEIPT


if __name__ == "__main__":
    if "--record" in sys.argv:
        print(f"wrote {record()}")
    else:
        print(json.dumps(build(), indent=1, default=str))
