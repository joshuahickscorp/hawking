#!/usr/bin/env python3
"""G086: what was ACTUALLY runnable at the moment of the wait.

Two 30m timelines are currently indistinguishable, and that is the defect:

    archived 477 s gap   next_work_left t=88,  exhausted True, n 0, ids []
                         ...and the run ended with TWELVE frontiers holding novel
                         work the driver had missed. The claim was WRONG.

    run three 482 s gap  next_work_left t=81,  exhausted True, n 0, ids []
                         ...and the work found afterwards was created by the
                         detached job the loop was waiting for.

Same signal, opposite meanings. `exhausted` is the DRIVER'S OWN CLAIM about the
frontier, and in the archived case it was false. A judge keyed on it acquits the
original defect - I tried it, and the negative control caught it.

The missing evidence is not another opinion. It is a per-frontier snapshot taken
AT the wait: for each frontier, how many entries it holds, how many survive the
scar filter, how many are already launched this run, and therefore how many were
RUNNABLE at that instant. A wait with zero runnable is a wait. A wait with
runnable work is the failure, and now the timeline says which.

    python3 tools/future/runnability_snapshot.py --explain
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/runnability_snapshot.py"
RECEIPT_NAME = "RUNNABILITY_SNAPSHOT.json"

WAIT_JUSTIFIED = "WAIT_JUSTIFIED"
IDLE_WITH_RUNNABLE_WORK = "IDLE_WITH_RUNNABLE_WORK"
UNEVIDENCED = "UNEVIDENCED"


class SnapshotRefused(RuntimeError):
    """A snapshot that asserts runnability without having looked."""


def snapshot(
    frontiers: Sequence[Mapping[str, Any]],
    *,
    already_launched: Sequence[str],
    scar_dead: Sequence[str],
    t_s: float,
) -> dict[str, Any]:
    """Per-frontier counts at ONE instant. Every number is derived, none asserted."""
    if frontiers is None:
        raise SnapshotRefused(
            "the frontier set was not supplied; a snapshot that did not look is "
            "the driver's claim again, which is what this replaces"
        )
    launched = {str(x) for x in already_launched}
    dead = {str(x) for x in scar_dead}
    rows = []
    total_runnable = 0
    for f in frontiers:
        entries = [str(e) for e in (f.get("entry_ids") or []) if str(e)]
        live = [e for e in entries if e not in dead]
        fresh = [e for e in live if e not in launched]
        total_runnable += len(fresh)
        rows.append({
            "frontier_id": str(f.get("id") or f.get("frontier_id") or ""),
            "n_entries": len(entries),
            "n_after_scars": len(live),
            "n_not_yet_launched": len(fresh),
            "runnable_ids": fresh[:8],
        })
    return {
        "t_s": t_s,
        "n_frontiers": len(rows),
        "n_runnable": total_runnable,
        "runnable_by_frontier": rows,
        "counts_are_derived_not_claimed": True,
    }


def classify_wait(snap: Mapping[str, Any], *, waiting_on: Sequence[Any]) -> dict[str, Any]:
    """A wait with zero runnable is a wait. With runnable work it is the failure."""
    if "n_runnable" not in snap:
        return {
            "verdict": UNEVIDENCED,
            "why": (
                "no runnability snapshot was taken at this wait, so the timeline "
                "cannot say whether work existed. This is the state both 30m "
                "timelines are in, and it is why they are indistinguishable."
            ),
        }
    n = int(snap["n_runnable"])
    if not waiting_on:
        return {
            "verdict": IDLE_WITH_RUNNABLE_WORK if n else UNEVIDENCED,
            "why": "a wait with no open handle is not a wait; it is an end",
        }
    if n == 0:
        return {
            "verdict": WAIT_JUSTIFIED,
            "n_runnable": 0,
            "why": (
                f"{len(waiting_on)} handle(s) in flight and NOTHING runnable across "
                f"{snap['n_frontiers']} frontiers - every entry is scar-dead or "
                "already launched this run. Waiting on the only live work is not "
                "idling."
            ),
        }
    return {
        "verdict": IDLE_WITH_RUNNABLE_WORK,
        "n_runnable": n,
        "why": (
            f"{n} runnable entries existed while the loop waited. This is the "
            "failure the no-idle law is for, and the snapshot names which "
            "frontiers held them rather than leaving it to be inferred."
        ),
        "runnable_by_frontier": [
            r for r in snap["runnable_by_frontier"] if r["n_not_yet_launched"]
        ][:8],
    }


def why_the_two_timelines_are_indistinguishable() -> dict[str, Any]:
    return {
        "archived_477s": {
            "signal": "next_work_left t=88, exhausted True, n 0, ids []",
            "truth": "the run ended with twelve frontiers holding novel work the "
                     "driver had missed, so the claim was FALSE",
        },
        "run_three_482s": {
            "signal": "next_work_left t=81, exhausted True, n 0, ids []",
            "truth": "the work found afterwards was created by the detached job "
                     "the loop was waiting for; UNKNOWN whether anything was "
                     "runnable at t=81",
        },
        "conclusion": (
            "identical signal, opposite meanings. `exhausted` is the driver's own "
            "claim and in one case it was wrong, so no judge keyed on it can "
            "separate them - a rule that acquits run three acquits the archived "
            "defect too, which the negative control demonstrated."
        ),
        "what_closes_it": (
            "a per-frontier runnability snapshot taken AT the wait. Not another "
            "opinion about the timeline; a count taken at the instant in question."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "schema": "hawking.future.runnability_snapshot.v1",
        "version": 1,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "verdicts": [WAIT_JUSTIFIED, IDLE_WITH_RUNNABLE_WORK, UNEVIDENCED],
        "indistinguishable_timelines": why_the_two_timelines_are_indistinguishable(),
        "claim_boundary": (
            "Static sidecar artifact. This defines and tests the snapshot; it does "
            "not retrofit one onto timelines recorded before it existed. Both 30m "
            "runs stay UNEVIDENCED on this axis, which is the honest verdict - "
            "neither can be convicted nor acquitted from what they recorded."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--explain", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(RECEIPT_NAME, doc, RECORDED_BY))
        return 0
    print(json.dumps(doc["indistinguishable_timelines"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
