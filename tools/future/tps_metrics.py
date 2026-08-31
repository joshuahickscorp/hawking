#!/usr/bin/env python3
"""G088: two TPS metrics, and neither may be reported as the other.

S025 §45-46.

    GREEDY_SINGLE_TOKEN_LATENCY_TPS   one conversation's latency. 71 means THIS.
    ACCEPTED_TOKEN_TPS                accepted tokens per second across whatever
                                      acceptance scheme produced them.

They are not interchangeable and the failure mode is specific: a speculative or
parallel decoding scheme raises accepted throughput without touching single-token
latency, and reporting that as progress toward 71 would be a claim the physics
does not support. Speculative decoding may never be said to have solved
single-token latency.

It stays a real Pareto frontier - useful accepted throughput is worth having -
priced on accepted tokens, verifier work, resident memory, rejection rate and
HCLI mission latency. This module keeps the two apart so that either can be
pursued honestly.

The concurrency result is the live example. n=2 aggregates 449-506 GB/s while
each session runs SLOWER than n=1. That is neither metric: it is aggregate
throughput across independent requests, and it belongs in a third slot rather
than being quietly counted as either.

    python3 tools/future/tps_metrics.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/tps_metrics.py"
RECEIPT_NAME = "TPS_METRICS.json"

LATENCY = "GREEDY_SINGLE_TOKEN_LATENCY_TPS"
ACCEPTED = "ACCEPTED_TOKEN_TPS"
AGGREGATE = "AGGREGATE_MULTISTREAM_TPS"

TARGET_TPS = 71.0
TARGET_MS = 1000.0 / TARGET_TPS

BODY_REL = "receipts/future/RESIDENT_TOKEN_BUDGET_POST_WIDEN_F4.json"
CONC_REL = "receipts/future/RESIDENT_CONCURRENCY_MEASURED.json"


class MetricConfusion(RuntimeError):
    """One metric reported as another. The only error this module exists for."""


def which_metric_is_71() -> str:
    return LATENCY


def assert_not_confused(metric: str, *, claimed_progress_toward_71: bool) -> None:
    """71 is single-stream latency. Nothing else may claim progress toward it."""
    if metric not in {LATENCY, ACCEPTED, AGGREGATE}:
        raise MetricConfusion(f"unknown metric {metric!r}")
    if claimed_progress_toward_71 and metric != LATENCY:
        raise MetricConfusion(
            f"{metric} cannot be progress toward 71. 71 is "
            f"{LATENCY}: one conversation's token latency, {TARGET_MS:.3f} ms. "
            f"{metric} can rise while single-token latency gets WORSE, which is "
            "exactly what the concurrency ladder measured."
        )


def _read(rel: str) -> dict[str, Any] | None:
    path = REPO / rel
    return json.loads(path.read_text()) if path.is_file() else None


def current() -> dict[str, Any]:
    body = _read(BODY_REL)
    conc = _read(CONC_REL)
    out: dict[str, Any] = {
        LATENCY: None,
        ACCEPTED: None,
        AGGREGATE: None,
    }
    if body:
        ms = float(body["decode_wall_ms_per_token"])
        out[LATENCY] = {
            "tps": round(1000.0 / ms, 3),
            "ms_per_token": ms,
            "source": BODY_REL,
            "source_field": "decode_wall_ms_per_token",
            "is_the_71_target": True,
            "gap_ms": round(ms - TARGET_MS, 4),
        }
    if conc:
        levels = conc["bandwidth_cross_check"]["gb_s_by_level"]
        per = [l.get("per_session_decode_tps") for l in conc["levels"]]
        out[AGGREGATE] = {
            "gb_s_by_level": levels,
            "per_session_decode_tps": per,
            "source": CONC_REL,
            "is_the_71_target": False,
            "why_not": (
                "each session is SLOWER under concurrency - 23.1 and 22.6 against "
                "36.6 - so this rises while the 71 metric falls. It is aggregate "
                "throughput across independent requests."
            ),
        }
    out[ACCEPTED] = {
        "tps": None,
        "status": "NOT_MEASURED",
        "is_the_71_target": False,
        "why_it_exists": (
            "speculative decoding, multi-token verification, a draft/resident "
            "pair or learned prediction heads can raise accepted throughput "
            "without touching single-token latency. A real Pareto frontier, "
            "priced on accepted tokens, verifier work, resident memory, rejection "
            "rate and HCLI mission latency."
        ),
        "may_never_claim": (
            "that it solved single-token latency. If it is ever measured, it is "
            "reported as ACCEPTED_TOKEN_TPS and never as progress toward 71."
        ),
    }
    return out


def _cited_only(metrics: dict[str, Any]) -> dict[str, Any]:
    """CITE, DO NOT COPY. This sidecar has no GPU authority, so it records WHERE
    each number lives and never the number itself.

    write_receipt refuses a hardware field here and it is right to: a copied TPS
    is a second source of truth that goes stale silently, which is the drift the
    citation machinery in causal_budget_71 exists to stop. current() still
    resolves the values at runtime for a reader; the receipt keeps the pointer.
    """
    out: dict[str, Any] = {}
    for name, block in metrics.items():
        if not isinstance(block, dict):
            out[name] = block
            continue
        keep = {
            k: v for k, v in block.items()
            if not isinstance(v, (int, float)) or isinstance(v, bool)
        }
        keep["values_are_cited_not_copied"] = True
        out[name] = keep
    return out


def build() -> dict[str, Any]:
    return {
        "schema": "hawking.future.tps_metrics.v1",
        "version": 1,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "the_71_metric": which_metric_is_71(),
        "target_tps": TARGET_TPS,
        "target_ms_per_token": round(TARGET_MS, 4),
        "metrics": _cited_only(current()),
        "claim_boundary": (
            "Static sidecar artifact. It reads the two measured receipts and "
            "keeps their numbers in separate slots; it measures nothing itself. "
            "ACCEPTED_TOKEN_TPS is NOT_MEASURED and reported as null rather than "
            "as zero, because nothing has run. Numbers are CITED by receipt and "
            "field, never copied: a copied TPS is a second source of truth that "
            "goes stale silently."
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
    print(json.dumps(doc["metrics"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
