#!/usr/bin/env python3
"""Attack the 30m PASS. A verdict nobody tried to break is not a verdict.

Every control here mutates the sealed 30m transcript or the receipts its
conditions rest on, and asserts the judge FLIPS. A control that does not flip
the judge is either a leak in the evaluator or a bad control, and this module
records one of each -- the second was mine.

    python3 tools/future/trial_negative_controls.py --record
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import autonomy_trial as at  # noqa: E402
from _common import REPO  # noqa: E402

# Lowercase "30m" is the tracked spelling. An earlier invocation passed
# AUTONOMY_TIMELINE_30M.json and macOS, being case-insensitive, silently resolved
# it to this same file -- so the run wrote where it was meant to, by luck. On a
# case-sensitive filesystem that argument would have created a second file and
# the judge would have read a stale transcript. Use the tracked spelling.
TIMELINE = REPO / "receipts" / "future" / "AUTONOMY_TIMELINE_30m.json"
RECEIPT = REPO / "receipts" / "future" / "TRIAL_NEGATIVE_CONTROLS.json"


def _doc() -> dict[str, Any]:
    return json.loads(TIMELINE.read_text())


def _strip_kinds(doc: dict[str, Any], kinds: tuple[str, ...]) -> dict[str, Any]:
    d = copy.deepcopy(doc)
    d["events"] = [e for e in d["events"] if e.get("kind") not in kinds]
    return d


def _strip_cites(doc: dict[str, Any], needle: str) -> dict[str, Any]:
    d = copy.deepcopy(doc)
    for e in d["events"]:
        if e.get("cites"):
            e["cites"] = [c for c in e["cites"] if needle not in c]
    return d


EVIDENCE_REMOVAL: tuple[tuple[str, tuple[str, ...], Callable[..., Any]], ...] = (
    ("overlap_detached_work", ("detached_started", "detached_completed"),
     at.eval_overlap_detached_work),
    ("use_negative_science", ("negative_science_query", "negative_science_refusal"),
     at.eval_use_negative_science),
    ("alter_priority_from_evidence", ("priority_altered",),
     at.eval_alter_priority_from_evidence),
    ("refill_work", ("work_refilled",), at.eval_refill_work),
    ("ingest_completed_result", ("result_ingested",), at.eval_ingest_completed_result),
)

CITATION_REMOVAL: tuple[tuple[str, str, Callable[..., Any]], ...] = (
    ("mutation_proposed_and_rolled_back", "MUTATION_ENGINE.json",
     at.eval_mutation_proposed_and_rolled_back),
    ("status_causality_challenged", "STATUS_CAUSALITY_CHALLENGE.json",
     at.eval_status_causality_challenged),
)


def run() -> dict[str, Any]:
    doc = _doc()
    results: list[dict[str, Any]] = []

    # 1. The seal must notice the single edit that would matter most.
    tampered = copy.deepcopy(doc)
    tampered["elapsed_s"] = 1800
    before = at.timeline_internal_seal_state(doc)
    after = at.timeline_internal_seal_state(tampered)
    results.append({
        "control": "seal_detects_the_elapsed_lie",
        "what": "rewrite elapsed_s from the real 565 to the full 1800",
        "untouched_verifies": before.get("verifies"),
        "tampered_verifies": after.get("verifies"),
        "flipped": before.get("verifies") is True and after.get("verifies") is False,
    })

    # 2. Removing the evidence must remove the verdict.
    for cond, kinds, fn in EVIDENCE_REMOVAL:
        r = fn(at.TimelineView(_strip_kinds(doc, kinds), "30m"))
        results.append({
            "control": f"strip_events_{cond}",
            "what": f"delete every {' / '.join(kinds)} event",
            "met_after": bool(r.get("met")),
            "flipped": not r.get("met"),
        })

    # 3. Removing the CITATION must remove the verdict, since these two
    #    conditions are judged from a cited receipt rather than an event label.
    for cond, needle, fn in CITATION_REMOVAL:
        r = fn(at.TimelineView(_strip_cites(doc, needle), "30m"))
        results.append({
            "control": f"strip_citation_{cond}",
            "what": f"delete every cite naming {needle}",
            "met_after": bool(r.get("met")),
            "flipped": not r.get("met"),
        })

    # 4. Corrupting the cited receipt's CONTENT must remove the verdict.
    mut = REPO / "receipts/future/MUTATION_ENGINE.json"
    orig = mut.read_text()
    try:
        d = json.loads(orig)
        d.setdefault("proofs", {})["all_hold"] = False
        mut.write_text(json.dumps(d, indent=1, sort_keys=True))
        r = at.eval_mutation_proposed_and_rolled_back(at.TimelineView(doc, "30m"))
        results.append({
            "control": "corrupt_receipt_mutation_all_hold_false",
            "what": "set proofs.all_hold false in the receipt the judge reads",
            "met_after": bool(r.get("met")),
            "flipped": not r.get("met"),
            "detail": r.get("detail"),
        })
    finally:
        mut.write_text(orig)

    sc = REPO / "receipts/future/STATUS_CAUSALITY_CHALLENGE.json"
    orig = sc.read_text()
    try:
        for label, key in (("blank_historical_cases", "historical_cases"),
                           ("blank_supported_fixtures", "supported_fixtures")):
            d = json.loads(orig)
            d[key] = []
            sc.write_text(json.dumps(d, indent=1, sort_keys=True))
            r = at.eval_status_causality_challenged(at.TimelineView(doc, "30m"))
            results.append({
                "control": f"corrupt_receipt_status_{label}",
                "what": f"empty {key} in the receipt the judge reads",
                "met_after": bool(r.get("met")),
                "flipped": not r.get("met"),
            })
    finally:
        sc.write_text(orig)

    leaks = [r for r in results if not r["flipped"]]
    return {
        "schema": "hawking.future.trial_negative_controls.v1",
        "version": 1,
        "recorded_by": "tools/future/trial_negative_controls.py",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "trial": "30m",
        "timeline": str(TIMELINE.relative_to(REPO)),
        "n_controls": len(results),
        "n_leaks": len(leaks),
        "all_controls_flipped_the_judge": not leaks,
        "controls": results,
        "leaks": leaks,
        "a_bad_control_of_my_own": {
            "what": "a first pass emptied challenges/labels/verdicts/results in "
                    "STATUS_CAUSALITY_CHALLENGE.json and reported a LEAK when the "
                    "evaluator still passed",
            "why_it_was_wrong": "the evaluator reads historical_cases and "
                                "supported_fixtures. Those keys were never touched, "
                                "so nothing had been removed.",
            "the_lesson": "a negative control is itself a probe and can be narrow. "
                          "Emptying the wrong key proves nothing and looks exactly "
                          "like a leak.",
        },
        "real_limitation_the_controls_do_not_fix": {
            "condition": "status_causality_challenged",
            "what": "it rests on historical_cases and supported_fixtures, which "
                    "are replayed history and named fixtures. The condition is "
                    "therefore satisfied without any status label produced DURING "
                    "the trial being challenged.",
            "class": "fixture-only success",
            "not_a_leak_because": "the negative controls do flip it, so the "
                                  "evaluator reads what it claims to read",
            "still_weak_because": "a trial could pass this condition while its own "
                                  "run produced no causal challenge at all",
            "what_would_close_it": "require at least one OVERREACHING verdict whose "
                                   "subject label was recorded during this trial",
        },
        "claim_boundary": (
            "These controls test the JUDGE, not the resident. They show that "
            "removing evidence removes the verdict. They do not show the trial's "
            "behaviour was good; a separate finding records that the same PASS "
            "contains a 477-second idle the judge cannot see."
        ),
    }


def record() -> Path:
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(run(), indent=1, sort_keys=True) + "\n")
    return RECEIPT


if __name__ == "__main__":
    doc = run()
    if "--record" in sys.argv:
        print(f"wrote {record()}")
    for r in doc["controls"]:
        print(f"  {'ok ' if r['flipped'] else 'LEAK'}  {r['control']}")
    print(f"\n{doc['n_controls']} controls, {doc['n_leaks']} leaks")
    raise SystemExit(0 if doc["all_controls_flipped_the_judge"] else 1)
