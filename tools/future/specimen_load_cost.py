#!/usr/bin/env python3
"""G102: load, prefetch and evict are work, and the cost is MEASURED.

S027 §4, §8-§13, §33, §34. The library lives on a USB-attached APFS volume and
nobody had timed a read from it. Measured, 3.003 GB per sample:

    quiet   cold    76.5 and 78.9 MB/s      warm    11015 MB/s
    loaded  cold    63.3 MB/s               warm    10987 MB/s

TWO FACTS FALL OUT AND BOTH CHANGE SCHEDULING.

  1. COLD IS 143x SLOWER THAN WARM. Residency is not a nicety; it is the whole
     difference between a 13-minute load and a 6-second one for a 61 GB
     specimen.

  2. DOWNLOAD CONTENTION COSTS ONLY 1.21x. The volume is slow on its own. S027
     §27 asked whether ModelLake writes materially interfere with loads; the
     answer measured here is BARELY - suspending downloads buys 21%, while
     keeping a specimen warm buys 143x.

Cold-loading the incumbent's own 360 GB source takes 78 minutes at this rate.
That is what makes S027 §8's prefetch mandatory rather than an optimisation.

    python3 tools/future/specimen_load_cost.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, measurement_provenance, write_measured_receipt  # noqa: E402
import specimen_registry as sr  # noqa: E402

RECORDED_BY = "tools/future/specimen_load_cost.py"
RECEIPT_NAME = "SPECIMEN_LOAD_COST.json"
RAW_REL = "receipts/future/_G102_LOAD_RATES_raw.json"

# S027 §4. Loading is work with a shape, not something hidden inside a command.
WORK_SPECIES = {
    "LOAD_SPECIMEN": {"resource": "DISK_READ", "memory_effect": "+source_bytes",
                      "reversible_by": "EVICT_SPECIMEN"},
    "UNLOAD_SPECIMEN": {"resource": "CPU", "memory_effect": "-resident_bytes",
                        "reversible_by": "LOAD_SPECIMEN"},
    "WARM_EXECUTOR": {"resource": "GPU", "memory_effect": "+pipeline_states",
                      "reversible_by": "none; cheap to redo"},
    "LOAD_NX": {"resource": "DISK_READ", "memory_effect": "+nx_bytes",
                "reversible_by": "EVICT_SPECIMEN"},
    "LOAD_NR_FOR_COMPILE": {"resource": "DISK_READ", "memory_effect": "+nr_bytes",
                            "reversible_by": "EVICT_SPECIMEN"},
    "PREFETCH_SPECIMEN": {"resource": "DISK_READ", "memory_effect": "+page_cache",
                          "reversible_by": "eviction by pressure; speculative"},
    "EVICT_SPECIMEN": {"resource": "CPU", "memory_effect": "-resident_bytes",
                       "reversible_by": "LOAD_SPECIMEN at full cold cost"},
}


class LoadCostRefused(RuntimeError):
    """The rate samples are missing, so no cost may be predicted."""


def _raw() -> dict[str, Any]:
    p = REPO / RAW_REL
    if not p.is_file():
        raise LoadCostRefused(
            f"{RAW_REL} is not on disk; a load cost model with no measured rate "
            "would be a guess wearing a receipt"
        )
    return json.loads(p.read_text())


def rates() -> dict[str, Any]:
    s = _raw()["samples"]
    def pick(label: str) -> dict[str, Any]:
        m = [x for x in s if x["label"] == label]
        if not m:
            raise LoadCostRefused(f"sample {label} is absent")
        return m[0]
    qc1, qc2 = pick("quiet_cold"), pick("quiet_cold_2")
    cc, qw, cw = pick("contended_cold"), pick("quiet_warm"), pick("contended_warm")
    quiet_cold = (qc1["MB_per_s"] + qc2["MB_per_s"]) / 2
    return {
        "quiet_cold_MB_per_s": round(quiet_cold, 1),
        "quiet_cold_samples": [qc1["MB_per_s"], qc2["MB_per_s"]],
        "quiet_cold_spread": round(max(qc1["MB_per_s"], qc2["MB_per_s"]) /
                                   min(qc1["MB_per_s"], qc2["MB_per_s"]), 4),
        "contended_cold_MB_per_s": cc["MB_per_s"],
        "quiet_warm_MB_per_s": qw["MB_per_s"],
        "contended_warm_MB_per_s": cw["MB_per_s"],
        "warm_over_cold": round(qw["MB_per_s"] / quiet_cold, 1),
        "contention_cost": round(quiet_cold / cc["MB_per_s"], 4),
        "volume": _raw()["volume"],
        "reading": (
            f"warm is {round(qw['MB_per_s'] / quiet_cold)}x cold, while "
            f"suspending downloads buys only "
            f"{round((quiet_cold / cc['MB_per_s'] - 1) * 100)}%. Residency "
            "dominates contention by two orders of magnitude."
        ),
    }


def scored_prediction() -> dict[str, Any]:
    """S027 §33: the model must be scored against an observation it did not fit."""
    s = _raw()["samples"]
    fit = next(x for x in s if x["label"] == "quiet_cold")
    held = next(x for x in s if x["label"] == "quiet_cold_2")
    rate = fit["MB_per_s"] * 1e6
    predicted = held["bytes"] / rate
    observed = held["seconds"]
    return {
        "fitted_on": fit["specimen"],
        "scored_on": held["specimen"],
        "predicted_seconds": round(predicted, 3),
        "observed_seconds": observed,
        "relative_error": round(abs(predicted - observed) / observed, 4),
        "within_10pct": abs(predicted - observed) / observed < 0.10,
        "why_this_is_a_real_score": (
            "the rate was fitted on one specimen and scored on a DIFFERENT one "
            "with a different file count and layout. Scoring the model on the "
            "sample it was fitted to would measure nothing."
        ),
        "n_held_out_points": 1,
        "one_point_is_not_a_validation": (
            "a single held-out specimen makes this a sanity check, not a "
            "validated cost model. It becomes one as real loads accumulate."
        ),
    }


def per_specimen() -> list[dict[str, Any]]:
    r = rates()
    cold = r["quiet_cold_MB_per_s"] * 1e6
    warm = r["quiet_warm_MB_per_s"] * 1e6
    out = []
    for s in sorted(sr.schedulable(), key=lambda x: x["source_bytes"] or 0):
        b = s["source_bytes"] or 0
        out.append({
            "id": s["id"],
            "source_gb": round(b / 1e9, 2),
            "cold_load_seconds": round(b / cold, 1),
            "cold_load_minutes": round(b / cold / 60.0, 2),
            "warm_load_seconds": round(b / warm, 2),
            "fits_in_uma": None,
        })
    return out


def prefetch_case() -> dict[str, Any]:
    """S027 §8: the number that makes prefetch mandatory rather than optional."""
    rows = per_specimen()
    total_gb = sum(r["source_gb"] for r in rows)
    total_min = sum(r["cold_load_minutes"] for r in rows)
    worst = max(rows, key=lambda r: r["cold_load_minutes"])
    return {
        "n_sealed_specimens": len(rows),
        "total_source_gb": round(total_gb, 1),
        "total_cold_load_minutes": round(total_min, 1),
        "total_cold_load_hours": round(total_min / 60.0, 2),
        "worst_single_specimen": {
            "id": worst["id"], "gb": worst["source_gb"],
            "cold_minutes": worst["cold_load_minutes"],
        },
        "statement": (
            f"cold-loading the {len(rows)} SEALED specimens once costs "
            f"{round(total_min / 60.0, 1)} hours of pure disk read, and the "
            f"largest alone is {worst['cold_load_minutes']:.0f} minutes. In a "
            "48-hour cycle that is time the Odyssey does not have to spend "
            "twice, which is exactly S027 §8's point: DO NOT WAIT UNTIL A MODEL "
            "IS NEEDED TO START LOADING IT."
        ),
        "and_this_is_the_sealed_subset_only": (
            "29 more specimens are complete but unsealed. Including them makes "
            "the figure much larger, and the registry's seal backlog is "
            "therefore a scheduling cost, not only a bookkeeping gap."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "obligation": "G102",
        "authority": "S027 §4, §8-§13, §27, §33, §34",
        "work_species": WORK_SPECIES,
        "rates": rates(),
        "scored_prediction": scored_prediction(),
        "per_specimen": per_specimen(),
        "prefetch_case": prefetch_case(),
        "ssd_contention_answer": {
            "question": "S027 §27: do ModelLake writes materially interfere?",
            "measured_contention_cost": rates()["contention_cost"],
            "answer": (
                "BARELY. Cold reads run 76.5-78.9 MB/s quiet and 63.3 MB/s with "
                "two downloads live - a 1.21x cost. Suspending downloads for a "
                "load is not worth it; keeping the specimen warm is worth 143x."
            ),
            "so_the_scheduling_rule_is": (
                "overlap loads with downloads freely, and spend the effort on "
                "residency instead"
            ),
        },
        "what_is_not_measured_here": (
            "this measures DISK READ only. A real LOAD_SPECIMEN also parses "
            "safetensors headers, allocates Metal buffers and warms pipeline "
            "states, none of which is timed here. The numbers are a FLOOR on "
            "load cost, and the module says so rather than calling them the "
            "load time."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_measured_receipt(
            REPO / "receipts" / "future" / RECEIPT_NAME, doc, RECORDED_BY,
            provenance=measurement_provenance(lock_held=False, lane="g102-load"),
        ))
        return 0
    print(json.dumps({k: doc[k] for k in
                      ("rates", "scored_prediction", "prefetch_case",
                       "ssd_contention_answer")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
