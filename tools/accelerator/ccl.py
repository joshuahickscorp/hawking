"""The CUDA Capability Ledger. FRONT B (G044, steer S015).

The authoritative inventory of useful CUDA/NVIDIA compute capabilities, with the
twelve fields the steer names on every entry. The validator refuses an incomplete
entry, because a ledger with holes silently reads as a ledger with coverage.

Two disciplines are enforced in code rather than by intention:

  1. NEVER "CUDA PARITY" UNQUALIFIED. A semantic_gap of NONE may only be claimed
     with evidence attached; without it the entry must say UNKNOWN.
  2. A performance_gap may not be stated without a receipt path. This program has
     measured exactly four things, and everything else says so.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "hawking.accelerator.cuda_capability_ledger.v1"

CLASSES = ("COMPILER", "EXECUTION", "MEMORY", "MATH", "RUNTIME", "PROFILING",
           "DEBUGGING", "MULTI_DEVICE")

FIELDS = ("capability_id", "cuda_mechanism", "why_it_exists", "underlying_problem",
          "apple_equivalent", "hawking_equivalent", "semantic_gap", "performance_gap",
          "priority", "test_corpus", "current_winner", "remaining_limitation")

GAP = ("NONE", "PARTIAL", "LARGE", "UNKNOWN", "DELETED_BY_UNIFIED_MEMORY")
PRIORITY = ("P0", "P1", "P2", "P3")


class CCLError(ValueError):
    pass


def entry(**kw: Any) -> dict[str, Any]:
    missing = [f for f in FIELDS if f not in kw]
    if missing:
        raise CCLError(f"entry {kw.get('capability_id')!r} is missing {missing}; a "
                       f"ledger with holes reads as a ledger with coverage")
    if kw["capability_id"].split(".")[0] not in CLASSES:
        raise CCLError(f"{kw['capability_id']!r} must start with one of {CLASSES}")
    if kw["semantic_gap"] not in GAP:
        raise CCLError(f"semantic_gap {kw['semantic_gap']!r} not in {GAP}")
    if kw["priority"] not in PRIORITY:
        raise CCLError(f"priority {kw['priority']!r} not in {PRIORITY}")
    if kw["semantic_gap"] == "NONE" and not kw.get("evidence"):
        raise CCLError(f"{kw['capability_id']}: a semantic gap of NONE is a PARITY "
                       f"CLAIM and needs evidence; say UNKNOWN instead")
    pg = kw["performance_gap"]
    if isinstance(pg, dict) and pg.get("measured") and not pg.get("receipt"):
        raise CCLError(f"{kw['capability_id']}: a measured performance gap needs a "
                       f"receipt path")
    return dict(kw)


def unmeasured(reason: str) -> dict[str, Any]:
    return {"measured": False, "reason": reason}


def measured(value: str, receipt: str) -> dict[str, Any]:
    return {"measured": True, "value": value, "receipt": receipt}


def recount(led: dict[str, Any]) -> dict[str, Any]:
    """Recompute every derived count from the entries.

    Exists because an edit that changed one entry's performance_gap without
    recomputing left the ledger claiming 10 measured gaps when it had 11 -- the
    ledger disagreeing with itself, which is precisely the failure its own validator
    is for. Any writer must call this rather than maintaining counts by hand.
    """
    led["count"] = len(led["entries"])
    led["performance_gaps_measured"] = sum(
        1 for e in led["entries"]
        if isinstance(e.get("performance_gap"), dict)
        and e["performance_gap"].get("measured"))
    led["performance_gaps_unmeasured"] = led["count"] - led["performance_gaps_measured"]
    by: dict[str, int] = {}
    for e in led["entries"]:
        c = e["capability_id"].split(".")[0]
        by[c] = by.get(c, 0) + 1
    led["by_class"] = by
    led["classes_with_no_entry"] = [c for c in CLASSES if c not in by]
    return led


def build(entries: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [e["capability_id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise CCLError("duplicate capability_id")
    by_class: dict[str, int] = {}
    for e in entries:
        by_class[e["capability_id"].split(".")[0]] = \
            by_class.get(e["capability_id"].split(".")[0], 0) + 1
    n_measured = sum(1 for e in entries
                     if isinstance(e["performance_gap"], dict)
                     and e["performance_gap"].get("measured"))
    return {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entries": entries,
        "count": len(entries),
        "by_class": by_class,
        "classes_with_no_entry": [c for c in CLASSES if c not in by_class],
        "performance_gaps_measured": n_measured,
        "performance_gaps_unmeasured": len(entries) - n_measured,
        "parity_claim": ("NOT CLAIMED. This is a capability inventory, not a parity "
                         "statement. No entry may report a semantic gap of NONE "
                         "without evidence, and the steer forbids the phrase 'CUDA "
                         "parity' unqualified."),
        "coverage_honesty": ("The CUDA surface is vastly larger than this. These are "
                             "the capabilities reachable or relevant from what has "
                             "actually been built here; absence from this ledger means "
                             "NOT YET STUDIED, never NOT NEEDED."),
    }
