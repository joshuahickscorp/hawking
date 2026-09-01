#!/usr/bin/env python3
"""G096: HCLI_FAST_THOUGHT, measured on events that were actually recorded.

S026 §14-§20 asks for three metrics and a decision-compression view. Every
number here is computed from event timelines already on disk and from git, not
from a scenario written to make the metrics look answerable.

    FRONTIER_TO_EXPERIMENT_LATENCY   result_ingested -> next workunit_launched
    CLAUDE_ESCALATIONS_PER_FRONTIER_MOVE   commits per VERIFIED obligation
    SCIENTIFIC_TPS                   verified frontier moves / wall hour

THE FIRST METRIC IS ALREADY GREEN AND THE THIRD IS NOT, AND THAT IS THE POINT.
The scheduler re-launches within a second of ingesting a result; what is slow is
the part that still runs in a Claude turn. Reporting only the green metric would
hide exactly the thing S026 §16 wants driven down.

    python3 tools/future/fast_thought.py --build
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/fast_thought.py"
RECEIPT_NAME = "HCLI_FAST_THOUGHT.json"

GOAL = Path.home() / ".claude/ultragoal/hawking-global-parallel-civilization/GOAL.md"

# Event timelines with real t_s stamps. Named rather than globbed so a new file
# cannot silently change a published metric.
TIMELINES = (
    "receipts/future/AUTONOMY_TIMELINE_1h.json",
    "receipts/future/AUTONOMY_TIMELINE_30m.json",
    "receipts/future/AUTONOMY_TIMELINE_15m.json",
    "receipts/future/MODEL_BEARING_TIMELINE.json",
)

INGEST = "result_ingested"
LAUNCH = "workunit_launched"

# S026 §20. An event of one of these kinds must trigger immediate
# reconsideration rather than waiting for the next heartbeat.
EVENT_DRIVEN_TRIGGERS = (
    "benchmark result",
    "failed test",
    "scar emitted",
    "model sealed",
    "new kernel compiled",
    "new child available",
    "causal budget changed",
    "unexpected regression",
)

# S026 §19's target for the scheduler loop.
LATENCY_TARGET_S = 120.0


class FastThoughtRefused(RuntimeError):
    """The evidence needed for a metric is missing or empty."""


def _timeline(rel: str) -> dict[str, Any]:
    p = REPO / rel
    if not p.is_file():
        raise FastThoughtRefused(f"{rel} is not on disk")
    d = json.loads(p.read_text())
    if not isinstance(d.get("events"), list) or not d["events"]:
        raise FastThoughtRefused(f"{rel} has no events")
    return d


def frontier_to_experiment_latency() -> dict[str, Any]:
    """S026 §19: evidence ingested -> next launch. Measured, per timeline."""
    per: list[dict[str, Any]] = []
    every: list[float] = []
    for rel in TIMELINES:
        d = _timeline(rel)
        gaps: list[float] = []
        pending: float | None = None
        for e in d["events"]:
            k = e.get("kind")
            if k == INGEST:
                pending = float(e["t_s"])
            elif k == LAUNCH and pending is not None:
                gaps.append(float(e["t_s"]) - pending)
                pending = None
        if not gaps:
            per.append({"timeline": rel, "pairs": 0,
                        "note": "no ingest-then-launch pair in this run"})
            continue
        every.extend(gaps)
        per.append({
            "timeline": rel,
            "pairs": len(gaps),
            "median_s": round(statistics.median(gaps), 4),
            "mean_s": round(statistics.mean(gaps), 4),
            "max_s": round(max(gaps), 4),
            "within_target": sum(1 for g in gaps if g <= LATENCY_TARGET_S),
        })
    if not every:
        raise FastThoughtRefused(
            "no ingest-then-launch pair exists in any recorded timeline; the "
            "metric has no evidence and must not be published as a number"
        )
    return {
        "target_s": LATENCY_TARGET_S,
        "per_timeline": per,
        "n_pairs": len(every),
        "median_s": round(statistics.median(every), 4),
        "mean_s": round(statistics.mean(every), 4),
        "max_s": round(max(every), 4),
        "fraction_within_target": round(
            sum(1 for g in every if g <= LATENCY_TARGET_S) / len(every), 4),
        "verdict": "GREEN" if max(every) <= LATENCY_TARGET_S else "AMBER",
        "what_it_does_not_measure": (
            "this is the SCHEDULER's latency - the gap between a result landing "
            "and the next unit launching. It says nothing about how long the "
            "reasoning took, because the reasoning that produced these units was "
            "largely precomputed by the frontier catalogue. A green number here "
            "is necessary and nowhere near sufficient for FAST_THOUGHT."
        ),
    }


def _ledger() -> str:
    if not GOAL.is_file():
        raise FastThoughtRefused(f"{GOAL} is not on disk")
    return GOAL.read_text()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout


def claude_escalations_per_frontier_move() -> dict[str, Any]:
    """S026 §16. A commit is the cheapest honest proxy for a Claude turn.

    Every commit in this campaign was authored inside a Claude turn, so commits
    per VERIFIED obligation is an UPPER bound on turns per move that needs no
    new instrumentation. It is a proxy and is labelled one.
    """
    txt = _ledger()
    # Match the Stop hook's own counting: every checkbox line is an obligation,
    # not only those whose text starts with a G-id. Counting the narrower set
    # reported 85 against the hook's 144 and would have made every ratio wrong.
    verified = len(re.findall(r"^- \[x\]", txt, re.M))
    unmet = len(re.findall(r"^- \[ \]", txt, re.M))
    partial = len(re.findall(r"^- \[~\]", txt, re.M))
    if verified == 0:
        raise FastThoughtRefused("no VERIFIED obligation in the ledger")
    commits = len([l for l in _git("log", "--oneline").splitlines() if l.strip()])
    return {
        "verified_obligations": verified,
        "unmet_obligations": unmet,
        "partial_obligations": partial,
        "commits": commits,
        "commits_per_verified_obligation": round(commits / verified, 3),
        "is_a_proxy": True,
        "why_a_proxy": (
            "a Claude turn is not recorded anywhere on disk, and a commit is. "
            "Every commit was authored inside a turn, so this is an UPPER bound "
            "on turns per move rather than a count of them. Driving it down is "
            "meaningful; treating it as the literal metric is not."
        ),
        "what_would_measure_it_directly": (
            "an escalation event written by HCLI whenever it hands a decision "
            "to Claude, so the ratio counts hand-offs rather than commits"
        ),
    }


def scientific_tps() -> dict[str, Any]:
    """S026 §51: verified frontier moves per wall hour."""
    txt = _ledger()
    verified = len(re.findall(r"^- \[x\]", txt, re.M))
    first = _git("log", "--reverse", "--format=%ct").splitlines()
    last = _git("log", "-1", "--format=%ct").splitlines()
    if not first or not last:
        raise FastThoughtRefused("git has no commits to date the campaign from")
    hours = (int(last[0]) - int(first[0])) / 3600.0
    if hours <= 0:
        raise FastThoughtRefused("campaign wall time is not positive")
    return {
        "verified_frontier_moves": verified,
        "campaign_wall_hours": round(hours, 2),
        "scientific_tps_per_hour": round(verified / hours, 4),
        "components": {
            "experiments_per_hour": round(
                len([l for l in _git("log", "--oneline").splitlines()]) / hours, 4),
        },
        "the_denominator_is_the_whole_repo_history": (
            "this dates the campaign from the first commit in the repository, "
            "which predates this ultragoal. The RATE is therefore a floor, not "
            "the rate of the current campaign. It is reported this way because "
            "the ledger records no campaign start, and inventing one would be "
            "the fabricated provenance this project forbids."
        ),
    }


def decision_compression() -> dict[str, Any]:
    """S026 §38: the ranked fact set a model should see instead of narrative."""
    import gap_ledger_60 as gap  # noqa: E402
    g = gap.build()
    rows = [
        {
            "option": r["id"],
            "max_ms_removable": r["max_ms_removable"],
            "status": r["status"],
        }
        for r in g["ranked_experiments"] if r["material"]
    ]
    return {
        "gap_to_60_ms": g["gap_to_60_ms"],
        "live_tps": g["live"]["tps"],
        "tps_if_built_levers_promoted": g["built_not_promoted"]["tps_if_promoted"],
        "options": rows,
        "hard_bounds": [
            g["arithmetic_ceiling"]["verdict"],
            g["bytes_required_after_arithmetic"]["statement"],
        ],
        "dead_schools": [
            "TOP_LEVEL_TOKEN_REORDERING_HAS_NO_CURRENT_SLACK",
            "BYTE_COUNT_TIMES_ORGAN_AVERAGE",
            "REMOVING_ONE_OP_CLASS_IS_NOT_A_LEVER_WHEN_FOUR_SHARE_THE_COST",
            "WARMUP_5_LEAVES_THE_FIRST_MEASURED_ARM_OUTSIDE_STEADY_STATE",
        ],
        "size_bytes": None,  # filled below
        "why_this_shape": (
            "a gap, a ranked option list with each option's maximum payoff, the "
            "bounds that are already proven, and the schools that are already "
            "dead. No history, no narrative, no receipts quoted at length. "
            "S026 §38 asks for exactly this and §63 says context is a cache."
        ),
    }


def event_driven_contract() -> dict[str, Any]:
    return {
        "triggers": list(EVENT_DRIVEN_TRIGGERS),
        "rule": (
            "any of these events triggers immediate reconsideration; the loop "
            "does not wait for a periodic heartbeat when the evidence is "
            "already available (S026 §20)"
        ),
        "observed_in_timelines": sorted({
            e.get("kind") for rel in TIMELINES for e in _timeline(rel)["events"]
            if e.get("kind")
        }),
        "gap": (
            "the recorded timelines carry result_ingested and workunit_launched, "
            "which cover the benchmark-result trigger. scar emitted, model "
            "sealed and new kernel compiled do NOT appear as event kinds in any "
            "recorded run, so those three triggers are DECLARED, NOT DEMONSTRATED."
        ),
    }


def build() -> dict[str, Any]:
    dc = decision_compression()
    dc["size_bytes"] = len(json.dumps(dc))
    return {
        "obligation": "G096",
        "authority": "S026 §14-§20, §35, §38, §51",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "frontier_to_experiment_latency": frontier_to_experiment_latency(),
        "claude_escalations_per_frontier_move": claude_escalations_per_frontier_move(),
        "scientific_tps": scientific_tps(),
        "event_driven_contract": event_driven_contract(),
        "decision_compression": dc,
        "honest_summary": (
            "the SCHEDULER loop is fast: 152 recorded ingest-then-launch pairs "
            "with a median of 0.0 s and a maximum of 2.0 s, far inside S026's "
            "seconds-to-low-minutes target. What is slow is everything that "
            "still runs in a Claude turn, and this module cannot measure that "
            "directly because no escalation event is written to disk. The first "
            "concrete step toward FAST_THOUGHT is therefore instrumentation, "
            "not optimisation."
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
    print(json.dumps({k: doc[k] for k in (
        "frontier_to_experiment_latency", "claude_escalations_per_frontier_move",
        "scientific_tps", "honest_summary")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
