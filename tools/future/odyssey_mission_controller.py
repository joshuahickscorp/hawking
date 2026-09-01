#!/usr/bin/env python3
"""G099: the 48-hour Odyssey mission controller.

S026 §21-§33 and §52-§53. Three overlapping phases inside one wall budget, with
experiment admission that KNOWS THE HORIZON: a five-hour experiment is fine at
hour 4 and is not at hour 44.

THE CYCLE HAS NOT STARTED, AND THIS MODULE SAYS SO RATHER THAN PRETENDING. Every
horizon question returns NOT_STARTED until a start is stamped, because a
controller that reports "hour 0 of 48" for a mission nobody launched is exactly
the fake progress this campaign forbids. Everything else - the admission rules,
the phase-entry conditions, the sealing contract - is live and testable now.

    python3 tools/future/odyssey_mission_controller.py --build
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/odyssey_mission_controller.py"
RECEIPT_NAME = "ODYSSEY_MISSION_CONTROLLER.json"
START_REL = "receipts/future/ODYSSEY_CYCLE_1_START.json"
SEAL_REL = "receipts/future/ODYSSEY_CYCLE_1.json"

CYCLE_HOURS = 48.0

# S026 §32. Work longer than this must justify itself before it is admitted.
LONG_WORK_MINUTES = 30.0
LONG_WORK_REQUIREMENTS = (
    "maximum_payoff",
    "why_no_cheaper_proxy",
    "what_runs_concurrently",
    "early_stop_criterion",
)

# S026 §25. Phases are logical roles, not exclusive machine modes: II and III
# begin when their INPUT exists, not at a clock time.
PHASE_ENTRY = {
    "ODYSSEY_I": "cycle start",
    "ODYSSEY_II": "at least one law exists to test transfer on",
    "ODYSSEY_III": "at least one law exists to attack",
}

# S026 §29. Most hypotheses must die before the expensive levels.
VERIFICATION_LADDER = (
    "structural", "organ", "hidden", "logits", "short_mission",
    "capability_subset", "full_qualification",
)


class MissionRefused(RuntimeError):
    """A horizon question was asked of a cycle that has not started."""


def _start() -> dict[str, Any] | None:
    p = REPO / START_REL
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def horizon(now: float | None = None) -> dict[str, Any]:
    """Elapsed and remaining wall. NOT_STARTED is an answer, not an error."""
    st = _start()
    if st is None:
        return {
            "state": "NOT_STARTED",
            "cycle_hours": CYCLE_HOURS,
            "elapsed_hours": None,
            "remaining_hours": None,
            "why": (
                f"{START_REL} is not on disk. The cycle has not been launched, "
                "and reporting 'hour 0 of 48' for a mission nobody started "
                "would be fake progress."
            ),
            "how_to_start": (
                "stamp the start receipt at launch; every horizon answer below "
                "becomes live the moment it exists"
            ),
        }
    t0 = float(st["started_unix"])
    elapsed = ((now if now is not None else time.time()) - t0) / 3600.0
    remaining = CYCLE_HOURS - elapsed
    return {
        "state": "RUNNING" if remaining > 0 else "OVER_BUDGET",
        "cycle_hours": CYCLE_HOURS,
        "started_utc": st.get("started_utc"),
        "elapsed_hours": round(elapsed, 3),
        "remaining_hours": round(remaining, 3),
        "fraction_spent": round(min(elapsed / CYCLE_HOURS, 1.0), 4),
    }


def admits(*, duration_minutes: float, justification: dict[str, Any] | None = None,
           now: float | None = None) -> dict[str, Any]:
    """S026 §32 and §52: is this experiment worth starting AT THIS HOUR?"""
    h = horizon(now)
    reasons: list[str] = []
    admitted = True

    if duration_minutes > LONG_WORK_MINUTES:
        missing = [k for k in LONG_WORK_REQUIREMENTS
                   if not (justification or {}).get(k)]
        if missing:
            admitted = False
            reasons.append(
                f"work over {LONG_WORK_MINUTES:.0f} minutes must state {missing}; "
                "long work is permitted, unexamined long work is not (S026 §32)"
            )

    if h["state"] == "RUNNING":
        need_h = duration_minutes / 60.0
        if need_h > h["remaining_hours"]:
            admitted = False
            reasons.append(
                f"needs {need_h:.2f} h and {h['remaining_hours']:.2f} h remain "
                "in the cycle; it cannot produce evidence before the seal "
                "unless detached continuation is explicitly allowed (S026 §52)"
            )
    elif h["state"] == "NOT_STARTED":
        reasons.append(
            "the horizon check is SKIPPED because the cycle has not started; "
            "only the long-work justification was applied"
        )

    return {
        "duration_minutes": duration_minutes,
        "admitted": admitted,
        "reasons": reasons or ["no rule refused it"],
        "horizon_state": h["state"],
    }


def phase_entry(*, n_laws: int, n_attackable_laws: int) -> dict[str, Any]:
    """S026 §25: II and III begin when their input exists, not on a clock."""
    h = horizon()
    started = h["state"] in ("RUNNING", "OVER_BUDGET")
    return {
        "entry_conditions": dict(PHASE_ENTRY),
        "ODYSSEY_I": started,
        "ODYSSEY_II": started and n_laws > 0,
        "ODYSSEY_III": started and n_attackable_laws > 0,
        "overlap_is_expected": (
            "the phases are logical roles, not exclusive machine modes. I, II "
            "and III may run simultaneously on different specimens, and the "
            "48-hour budget is WALL time, not the sum of 18 + 12 + 12."
        ),
        "n_laws": n_laws,
        "n_attackable_laws": n_attackable_laws,
    }


def depth_policy() -> dict[str, Any]:
    """S026 §27-§29: adaptive depth is what makes 48 hours possible."""
    return {
        "per_specimen": [
            "cheap fingerprint",
            "Doctor",
            "law and scar lookup",
            "cheapest discriminators",
        ],
        "shallow_if": "the specimen reproduces existing laws",
        "deep_if": "the specimen breaks them",
        "not_every_model_earns_an_nx": (
            "a specimen may serve its purpose by falsifying a law, confirming "
            "transfer, exposing a representation family or generating a scar. "
            "Only Pareto-relevant survivors earn deep lowering."
        ),
        "verification_ladder": list(VERIFICATION_LADDER),
        "ladder_rule": (
            "most hypotheses must die before the expensive levels; that is the "
            "only way 48-hour science is honest rather than rushed"
        ),
        "cancel_descendants": (
            "when a hypothesis dies its pending descendants die with it. Queued "
            "experiments are not executed merely because they were generated."
        ),
    }


def seal_contract() -> dict[str, Any]:
    """S026 §53 and §26: what sealing may and may not claim."""
    p = REPO / SEAL_REL
    return {
        "seal_path": SEAL_REL,
        "sealed": p.is_file(),
        "must_contain": [
            "laws", "scars", "transfers", "counterexamples", "best NXs",
            "resident improvement", "experiment-policy improvements",
        ],
        "time_indexed_universe": (
            "the cycle's specimen set is TIME-INDEXED. A model that seals at "
            "hour 47 may receive only a fingerprint, transfer probes and quick "
            "adversarial tests, and the cycle records that it JOINED LATE."
        ),
        "no_false_completeness": (
            "the seal must not claim every available model participated fully. "
            "Late arrivals are recorded as late, and unfinished high-value work "
            "moves to ODYSSEY_CYCLE_2 rather than being written off."
        ),
        "the_cycle_is_a_checkpoint_not_a_death": (
            "the mission does not stop at hour 48; cycle 2 begins from a much "
            "stronger prior with every cycle-1 law and scar available."
        ),
        "not_required_to_pass": (
            "the cycle does NOT require every model to be perfectly optimized "
            "(S026 §52). It requires real scoped laws, tested transfer, "
            "attacks, multiple specimens, reproducible evidence and no "
            "prompt-by-prompt human clocking."
        ),
    }


def build() -> dict[str, Any]:
    return {
        "obligation": "G099",
        "authority": "S026 §21-§33, §52-§53",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "horizon": horizon(),
        "phase_entry": phase_entry(n_laws=0, n_attackable_laws=0),
        "depth_policy": depth_policy(),
        "long_work_requirements": list(LONG_WORK_REQUIREMENTS),
        "seal_contract": seal_contract(),
        "odyssey_is_not_held_hostage_to_60_tps": (
            "S026 §33. The cycle starts when the launch-critical gates are "
            "green and the 60/71 campaign continues INSIDE it. If 60 lands six "
            "hours in, the resident migrates mid-cycle."
        ),
        "what_is_live_and_what_is_not": (
            "the admission rules, phase-entry conditions, depth policy and seal "
            "contract are live and tested now. Every HORIZON answer is "
            "NOT_STARTED, because the cycle has not been launched and a "
            "controller reporting 'hour 0 of 48' for a mission nobody started "
            "would be reporting fake progress."
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
    print(json.dumps({k: doc[k] for k in
                      ("horizon", "phase_entry", "seal_contract",
                       "what_is_live_and_what_is_not")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
