#!/usr/bin/env python3
"""G106: what Claude still does by hand, and what now owns each of those jobs.

S027 §38-§39, §65-§66. The acceptance criterion is stated as a FAILURE mode:
routine operation must not look like run-work, Claude notices, Claude writes a
fifty-page steer, work resumes, repeat.

This register lists recurring operations observed in this campaign, names the
artifact that owns each one now, and - for the ones nothing owns yet - says so.
The count that matters is CLAUDE_OWNED falling, and it can only fall honestly if
the unowned ones are listed rather than quietly dropped.

EVERY HANDOFF CITES AN ARTIFACT THAT EXISTS. A register whose entries point at
modules nobody wrote would be the fake completion this campaign forbids, so the
module CHECKS each owner is on disk and refuses to publish if one is not.

    python3 tools/future/recurring_operations.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/recurring_operations.py"
RECEIPT_NAME = "RECURRING_OPERATIONS.json"

# (operation, owner_path_or_None, note)
# owner None means NOBODY OWNS IT YET and Claude still does it by hand.
OPERATIONS: tuple[dict[str, Any], ...] = (
    {
        "op": "interpret a benchmark discrepancy",
        "owner": "tools/future/causal_pattern_library.py",
        "rule": "RULE.UNSTEADY_ARM_IS_NOT_A_MEASUREMENT",
        "note": "an arm outside steady state is refused by the consumer rather "
                "than diagnosed by a person; the producer emits rep_spread",
    },
    {
        "op": "decide whether an op-class is worth attacking",
        "owner": "tools/future/op_class_ablation.py",
        "rule": "RULE.MEASURE_THE_SPLIT_FIRST",
        "note": "the split is measured before a target is chosen; two "
                "candidates attacked a 1.8% class before this existed",
    },
    {
        "op": "recognise that a search has gone degenerate",
        "owner": "tools/future/productive_search.py",
        "rule": None,
        "note": "causal_question_id tracking, not prompt-string counting",
    },
    {
        "op": "choose whether to take a protected measurement window",
        "owner": "tools/future/protected_bench_lease.py",
        "rule": None,
        "note": "pause-measure-resume is encoded; G095 added the lesson that "
                "the supervisor must be suspended before the workers",
    },
    {
        "op": "update the causal budget after a lever lands",
        "owner": "tools/future/causal_budget_71.py",
        "rule": None,
        "note": "citations resolve against receipts and drift raises",
    },
    {
        "op": "compute the gap to a TPS target and what it licenses",
        "owner": "tools/future/gap_ledger_60.py",
        "rule": "RULE.SCHOOL_BOUNDED_BELOW_TARGET",
        "note": "gap derived from disk authority; the escalation clock's phase "
                "is computed from a start stamped once",
    },
    {
        "op": "suppress a hypothesis belonging to a dead school",
        "owner": "tools/future/frontiers.py",
        "rule": "RULE.SUPPRESS_REORDERING",
        "note": "refused at proposal time in _item, before GPU time is spent",
    },
    {
        "op": "decide which model to load next",
        "owner": "tools/future/specimen_scheduler.py",
        "rule": None,
        "note": "ranked by architecture distance over measured load cost; a "
                "scarred hypothesis loads nothing at all",
    },
    {
        "op": "know what specimens exist and what state they are in",
        "owner": "tools/future/specimen_registry.py",
        "rule": None,
        "note": "lifecycle derived from disk facts, not declared",
    },
    {
        "op": "check on ModelLake download progress",
        "owner": "tools/future/modellake_scheduler_view.py",
        "rule": None,
        "note": "the watcher was always writing it; nothing read it",
    },
    {
        "op": "predict whether a specimen fits in memory",
        "owner": "tools/future/uma_resource_ledger.py",
        "rule": None,
        "note": "peak predicted before the load rather than OOM after it",
    },
    {
        "op": "estimate what a load will cost",
        "owner": "tools/future/specimen_load_cost.py",
        "rule": None,
        "note": "measured on the real volume and scored held-out",
    },
    {
        "op": "decide whether an experiment is worth starting at this hour",
        "owner": "tools/future/odyssey_mission_controller.py",
        "rule": None,
        "note": "admission knows the horizon; five hours is fine at hour 4 and "
                "refused at hour 44",
    },
    {
        "op": "classify whether an obligation blocks launch",
        "owner": "tools/future/mission_class.py",
        "rule": None,
        "note": "derived from what the obligation's text says it blocks; the "
                "unknown case blocks rather than defers",
    },
    # ── still Claude's ────────────────────────────────────────────────────
    {
        "op": "design the next kernel candidate once a ladder names the class",
        "owner": None,
        "rule": None,
        "note": "the ladder says WHERE the cost is; turning that into a bitcast "
                "unpack with the right mantissa constants was novel work and "
                "was wrong twice before it was right",
    },
    {
        "op": "notice that a measurement instrument is defective",
        "owner": None,
        "rule": None,
        "note": "the warmup-5 bimodality was found by reading per-rep timings "
                "after a result refused to reproduce. The GUARD is now owned "
                "(rep_spread), but noticing a NEW instrument defect is not",
    },
    {
        "op": "write the receipt prose that states a claim boundary",
        "owner": None,
        "rule": None,
        "note": "every module says what it does not claim; deciding what the "
                "boundary IS remains a judgement",
    },
    {
        "op": "reconcile two harnesses that disagree",
        "owner": None,
        "rule": None,
        "note": "resident_reprofile and the organ profiler differ by 0.545 ms "
                "on the same control and nothing adjudicates that",
    },
    {
        "op": "decide that an obligation is met",
        "owner": None,
        "rule": None,
        "note": "the integration gate proves tests pass; whether the acceptance "
                "text is SATISFIED is still read by a person",
    },
)


class RegisterRefused(RuntimeError):
    """An entry names an owner that does not exist on disk."""


def entries() -> list[dict[str, Any]]:
    out = []
    for o in OPERATIONS:
        owner = o["owner"]
        if owner is not None and not (REPO / owner).is_file():
            raise RegisterRefused(
                f"{o['op']!r} names owner {owner} which is not on disk; a "
                "register pointing at modules nobody wrote is fake completion"
            )
        out.append({**o, "claude_owned": owner is None})
    return out


def counts() -> dict[str, Any]:
    e = entries()
    owned = [x for x in e if not x["claude_owned"]]
    claude = [x for x in e if x["claude_owned"]]
    return {
        "n_operations": len(e),
        "n_handed_off": len(owned),
        "n_still_claude_owned": len(claude),
        "fraction_handed_off": round(len(owned) / len(e), 3),
        "still_claude_owned": [x["op"] for x in claude],
        "handed_off_this_campaign": [
            {"op": x["op"], "owner": x["owner"]} for x in owned],
    }


def failure_mode() -> dict[str, Any]:
    """S027 §66 states the criterion as a failure, so state it that way."""
    return {
        "FAIL_looks_like": (
            "run work, Claude notices a problem, Claude writes a fifty-point "
            "steer, work resumes, repeat"
        ),
        "PASS_looks_like": (
            "work, evidence, HCLI diagnoses, HCLI generates probes, HCLI "
            "prepares the next model, HCLI runs them, the user optionally "
            "observes"
        ),
        "where_this_campaign_sits": (
            "between them, and closer to FAIL than the handed-off count "
            "suggests. Fourteen recurring operations now have owners, but the "
            "five that remain are the ones that decide what happens next: "
            "designing a candidate, noticing a new instrument defect, setting a "
            "claim boundary, adjudicating two disagreeing harnesses, and ruling "
            "an obligation met. Those are the loop, and Claude is still in it."
        ),
        "the_count_is_not_the_metric": (
            "handing off fourteen mechanical operations while keeping every "
            "judgement is not progress toward autonomy; it is a faster "
            "scaffold. The metric that matters is whether HCLI can run a cycle "
            "without a decision escalating, and this register does not claim it "
            "can."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "obligation": "G106",
        "authority": "S027 §38-§39, §65-§66",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "counts": counts(),
        "register": entries(),
        "failure_mode": failure_mode(),
        "every_owner_is_checked_on_disk": (
            "the module refuses to publish if any named owner is missing, "
            "because a register pointing at modules nobody wrote would be "
            "exactly the fake completion S027 §38 is guarding against"
        ),
        "how_this_falls_honestly": (
            "an operation moves out of the Claude column only when a module "
            "OWNS it - not when a module mentions it. The five remaining "
            "entries have no owner and are listed rather than dropped, so the "
            "count can only fall by real handoff."
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
    print(json.dumps({"counts": doc["counts"], "failure_mode": doc["failure_mode"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
