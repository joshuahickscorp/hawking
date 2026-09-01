"""G114: how often does Claude decide something the resident should have decided?

S030 §48-§49 and S033: Claude audits, challenges, inspects novel architecture and
repairs HCLI when HCLI breaks. Claude does NOT routinely decide which
representation is next, what model to load, what experiment to run, what failed,
or what a result means. This measures whether that is true, from RECORDED EVENTS.

TWO STREAMS, BOTH WITH A CLOCK.

    git       every landed obligation is a FRONTIER MOVE, with an author time
    sovereign every parsed iteration that produced EXECUTED work is a RESIDENT
              DECISION - the resident chose the experiment and the lab ran it

The G106 register is the input that says WHICH operations count: an op with
claude_owned=true is one whose exercise is an intervention by definition.

THE NUMBER IS NOT FLATTERING AND IS NOT MEANT TO BE. Claude authors nearly every
landing, so interventions per frontier move sits near 1.0 today. That is the
honest reading of a campaign where the resident runs experiments and Claude lands
code. The metric exists so the trend is visible, not so the first reading looks
good.

WHAT THIS MODULE HAD TO FIX BEFORE IT COULD MEASURE ANYTHING. The sovereign log's
first 83 entries carry only t_s - seconds since that run's start - so nothing in
them could be placed on a wall clock and the PER HOUR half of the obligation was
unmeasurable from the resident's own stream. hcli_sovereign._log now stamps an
absolute unix time on every entry. The pre-clock entries are reported as
UNCLOCKED rather than back-dated, because inventing a timestamp to complete a
rate is exactly the fabrication this campaign forbids.

    python3 tools/future/claude_interventions.py --build
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/claude_interventions.py"
RECEIPT_NAME = "CLAUDE_INTERVENTIONS.json"

REGISTER_REL = "receipts/future/RECURRING_OPERATIONS.json"
SOVEREIGN_LOG_REL = "receipts/future/_HCLI_SOVEREIGN_log.jsonl"
OBLIGATION = re.compile(r"\((G\d+)")
SINCE = "2026-08-26"          # the S026 60-TPS campaign window


class MetricRefused(RuntimeError):
    """An input stream is missing or carries no clock."""


def _git(*args: str) -> str:
    r = subprocess.run(["git", "--no-optional-locks", *args],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise MetricRefused(f"git {' '.join(args)} failed")
    return r.stdout


def register() -> dict[str, Any]:
    p = REPO / REGISTER_REL
    if not p.is_file():
        raise MetricRefused(
            f"{REGISTER_REL} is not on disk. G114's acceptance says the G106 "
            "register is this metric's INPUT; without it the ratio would be "
            "asserted rather than computed."
        )
    d = json.loads(p.read_text())
    rows = d["register"]
    owned = [r for r in rows if r.get("claude_owned")]
    return {
        "n_operations": len(rows),
        "n_claude_owned": len(owned),
        "n_handed_off": len(rows) - len(owned),
        "fraction_handed_off": round((len(rows) - len(owned)) / len(rows), 4),
        "still_claude_owned": sorted(r["op"] for r in owned),
        "source": REGISTER_REL,
    }


def frontier_moves() -> dict[str, Any]:
    """A landed obligation is a frontier move. git carries the clock."""
    out = []
    for line in _git("log", f"--since={SINCE}", "--format=%H|%at|%s").splitlines():
        h, at, subject = line.split("|", 2)
        m = OBLIGATION.search(subject)
        if m:
            out.append({"sha": h[:9], "unix": int(at), "obligation": m.group(1),
                        "subject": subject})
    if not out:
        raise MetricRefused(f"no obligation-bearing commits since {SINCE}")
    span_h = (max(r["unix"] for r in out) - min(r["unix"] for r in out)) / 3600.0
    return {
        "n": len(out),
        "distinct_obligations": len({r["obligation"] for r in out}),
        "span_hours": round(span_h, 3),
        "per_hour": round(len(out) / span_h, 3) if span_h > 0 else None,
        "first_unix": min(r["unix"] for r in out),
        "last_unix": max(r["unix"] for r in out),
    }


def resident_decisions() -> dict[str, Any]:
    """An iteration whose selected work actually RAN is a resident decision.

    Accepted-but-unlaunched does not count: the resident chose, and nothing
    happened. Counting it would credit the resident for work the harness dropped.
    """
    p = REPO / SOVEREIGN_LOG_REL
    if not p.is_file():
        raise MetricRefused(f"{SOVEREIGN_LOG_REL} is not on disk")
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    its = [r for r in rows if "n" in r]
    executed = [r for r in its
                if any(x.get("ran") for x in (r.get("results") or []))]
    clocked = [r for r in executed if "unix" in r]
    unclocked = len(executed) - len(clocked)
    out = {
        "n_iterations": len(its),
        "n_parsed": sum(1 for r in its if r.get("parsed")),
        "n_executed": len(executed),
        "n_clocked": len(clocked),
        "n_unclocked": unclocked,
        "per_hour": None,
        "per_hour_status": "UNCLOCKED_ENTRIES_PRESENT",
    }
    if clocked and unclocked == 0:
        span = (max(r["unix"] for r in clocked)
                - min(r["unix"] for r in clocked)) / 3600.0
        out["per_hour"] = round(len(clocked) / span, 3) if span > 0 else None
        out["per_hour_status"] = "MEASURED"
    elif unclocked:
        out["why_no_rate"] = (
            f"{unclocked} executed iterations predate the absolute clock in "
            "hcli_sovereign._log and carry only t_s, seconds since their run's "
            "start. Back-dating them to complete a rate would be fabricating a "
            "measurement, so the rate is withheld until the log has run long "
            "enough with a clock."
        )
    return out


def metric() -> dict[str, Any]:
    fm = frontier_moves()
    rd = resident_decisions()
    reg = register()
    # A frontier move Claude landed without a resident decision behind it is an
    # intervention. This is an UPPER BOUND on the resident's credit: it assumes
    # every executed resident experiment fed a landing, which flatters the
    # resident, and the number is still near 1.
    interventions = max(fm["n"] - rd["n_executed"], 0)
    return {
        "frontier_moves": fm["n"],
        "resident_decided": rd["n_executed"],
        "claude_interventions": interventions,
        "interventions_per_frontier_move": round(interventions / fm["n"], 4),
        "interventions_per_hour": (
            round(interventions / fm["span_hours"], 3)
            if fm["span_hours"] > 0 else None),
        "measurement_window_hours": fm["span_hours"],
        "register_fraction_handed_off": reg["fraction_handed_off"],
        "reading": (
            f"{fm['n']} frontier moves landed in {fm['span_hours']:.1f} hours "
            f"and {rd['n_executed']} resident-decided experiments ran in the "
            f"same campaign, so at most {rd['n_executed']} of those moves had a "
            "resident decision behind them. "
            f"{interventions / fm['n']:.1%} of frontier moves are Claude "
            "interventions. The register says "
            f"{reg['fraction_handed_off']:.1%} of recurring OPERATIONS are "
            "handed off, which is a different and more optimistic question: an "
            "operation can have an owner on disk and still be exercised by "
            "Claude on the day."
        ),
        "this_is_an_upper_bound_on_resident_credit": (
            "every executed resident experiment is credited as if it drove a "
            "landing. Several drove none. The true intervention share is higher "
            "than this figure, not lower."
        ),
    }


def is_it_falling() -> dict[str, Any]:
    """One reading is not a trend, and the receipt must not imply otherwise."""
    return {
        "verdict": "NOT_YET_ANSWERABLE",
        "why": (
            "G114 asks for a metric that is MEASURED AND FALLING. This is the "
            "first computation, so there is nothing to compare it against. A "
            "trend needs a second reading from a later window; claiming a "
            "direction from one point would be the assertion this obligation "
            "exists to replace."
        ),
        "what_would_make_it_falling": (
            "resident-decided experiments driving landings - the coding-tool "
            "path from G112 exercised by the resident rather than by Claude - "
            "so resident_decided rises against frontier_moves"
        ),
    }


def build() -> dict[str, Any]:
    return {
        "obligation": "G114",
        "question": (
            "how often does Claude decide something S030 says the resident "
            "should decide?"
        ),
        "metric": metric(),
        "frontier_moves": frontier_moves(),
        "resident_decisions": resident_decisions(),
        "register": register(),
        "is_it_falling": is_it_falling(),
        "producer_fix": (
            "hcli_sovereign._log now stamps an absolute unix time on every "
            "entry. The first 83 entries carry only t_s and are reported as "
            "UNCLOCKED rather than back-dated."
        ),
        "evidence_class": "DERIVED_FROM_RECORDED_EVENTS",
        "inputs": [REGISTER_REL, SOVEREIGN_LOG_REL, "git log"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(REPO / "receipts" / "future" / RECEIPT_NAME,
                            doc, RECORDED_BY))
        return 0
    print(json.dumps({k: doc[k] for k in
                      ("metric", "resident_decisions", "is_it_falling")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
