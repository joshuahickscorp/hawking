"""Is this a good run? Answered from disk, by named criteria, not by vibes.

"The daemon is up" was the only thing anyone could say about a run, and it was
true for 553 minutes while `accepted` stayed 0. Alive is not progress, and a
mission can be alive, busy, warm and structurally incapable of finishing.

Every criterion here is a defect that actually happened on this machine:

* `mission_advanceable`   -- evacuation cancelled the mission, and `cancelled`
                             is blocked, so a graceful stop was permanently
                             fatal.
* `repair_budget_unspent` -- the DAG was not archived with the mission, so a new
                             mission inherited a spent repair budget and every
                             unit failed once, terminally, at depth 0.
* `structured_output_ok`  -- 11 of 11 unit failures were the model failing to
                             emit JSON at all.
* `no_tool_loops`         -- one goal spent five of eleven tool invocations
                             re-issuing a call that had already been rejected.
* `progress`              -- the bar. Everything else can be green while nothing
                             gets done.

UNKNOWN is a real verdict and is never rounded to PASS. A criterion with no
evidence on disk has not been satisfied; it has not been checked.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

PASS = "PASS"
FAIL = "FAIL"
UNKNOWN = "UNKNOWN"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_events(path: Path, limit: int = 20000) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
                if len(rows) >= limit:
                    break
    except OSError:
        return []
    return rows


def _read_log(path: Path, limit: int = 20000) -> List[Dict[str, Any]]:
    return _read_events(path, limit)


def evaluate(workspace: Any) -> Dict[str, Any]:
    """One verdict per criterion, plus the run verdict, from durable state only."""
    root = Path(workspace)
    hcli = root / ".hcli"
    mission = _read_json(hcli / "mission" / "state.json")
    daemon = _read_json(hcli / "resident" / "state.json")
    dag = _read_json(hcli / "dag.json")
    events = _read_events(hcli / "mission" / "events.jsonl")
    log = _read_log(hcli / "mission" / "mission.log")

    criteria: Dict[str, Dict[str, Any]] = {}

    def record(name: str, verdict: str, detail: str) -> None:
        criteria[name] = {"verdict": verdict, "detail": detail}

    # 1. The mission can still be advanced without a human.
    if mission is None:
        record("mission_advanceable", UNKNOWN, "no mission state on disk")
    else:
        from .agentos.resident import BLOCKED_MISSION_PHASES

        phase = str(mission.get("phase") or "")
        if phase in BLOCKED_MISSION_PHASES:
            record("mission_advanceable", FAIL, f"phase={phase} needs a human")
        else:
            record("mission_advanceable", PASS, f"phase={phase}")

    # 2. The repair budget belongs to THIS mission.
    if dag is None:
        record("repair_budget_unspent", UNKNOWN, "no dag.json yet")
    else:
        counts = dag.get("repair_counts") or {}
        units = (mission or {}).get("units") or {}
        # INHERITED is not the same as SPENT, and conflating them made this
        # criterion fire the moment a unit legitimately earned a repair -- an
        # unreachable bar wearing a different hat. A root that this mission
        # actually repaired HAS a repair unit carrying `repairs: <root>`. A
        # count with no such unit came from a previous generation.
        earned = {
            str(unit.get("repairs"))
            for unit in units.values()
            if isinstance(unit, dict) and unit.get("repairs")
        }
        inherited = sorted(root for root in counts if root not in earned)
        if inherited:
            record(
                "repair_budget_unspent",
                FAIL,
                f"{len(inherited)} root(s) carry a repair count with no repair "
                f"unit in this mission ({', '.join(inherited[:4])}); an "
                "inherited budget refuses the first failure at depth 0",
            )
        else:
            record(
                "repair_budget_unspent",
                PASS,
                f"{len(counts)} spent, all earned by this mission",
            )

    # 3. No unit was refused a repair on its FIRST failure.
    depth0 = [
        row
        for row in log
        if row.get("event") == "repair_exhausted" and int(row.get("depth") or 0) == 0
    ]
    if depth0:
        ids = sorted({str(r.get("id")) for r in depth0})
        record(
            "repair_reached_the_unit",
            FAIL,
            f"refused at depth 0: {', '.join(ids[:5])}",
        )
    elif log:
        record("repair_reached_the_unit", PASS, "no depth-0 refusals")
    else:
        record("repair_reached_the_unit", UNKNOWN, "no mission log yet")

    # 4. The model could express itself in the result schema.
    exhausted = [
        row
        for row in events
        if row.get("type") == "model_call_finished"
        and (row.get("data") or {}).get("ok") is False
    ]
    # Scoped to THIS mission. Counting every receipt on disk would hold the
    # criterion against failures that happened before the fix that resolved
    # them, and a bar that can never be cleared is not a bar -- it is noise
    # that trains the reader to ignore the verdict.
    started_at = float((mission or {}).get("started_at") or 0.0)
    receipts = []
    if (hcli / "receipts").is_dir():
        for path in (hcli / "receipts").glob("*.json"):
            try:
                if not started_at or path.stat().st_mtime >= started_at:
                    receipts.append(path)
            except OSError:
                continue
    so_failures = 0
    for path in receipts:
        body = _read_json(path) or {}
        record_ = body.get("structured_output") or {}
        if record_.get("exhausted") is True:
            so_failures += 1
    if not receipts:
        record("structured_output_ok", UNKNOWN, "no receipts from this mission yet")
    elif so_failures:
        record(
            "structured_output_ok",
            FAIL,
            f"{so_failures} of {len(receipts)} receipts exhausted their retries",
        )
    else:
        record("structured_output_ok", PASS, f"0 of {len(receipts)} exhausted")

    # 5. No tool loop: the same call re-issued after it already answered.
    repeats = [r for r in events if r.get("type") == "tool_call_repeated"]
    invoked = [r for r in events if r.get("type") == "tool_invoked"]
    if not invoked:
        record("no_tool_loops", UNKNOWN, "no tool calls yet")
    elif len(repeats) > len(invoked) // 4:
        record(
            "no_tool_loops",
            FAIL,
            f"{len(repeats)} repeated of {len(invoked)} invocations",
        )
    else:
        record("no_tool_loops", PASS, f"{len(repeats)} repeated of {len(invoked)}")

    # 6. The worker is not burning its restart budget.
    if daemon is None:
        record("worker_stable", UNKNOWN, "no daemon state")
    else:
        streak = int(daemon.get("failure_streak") or 0)
        limit = int((daemon.get("config") or {}).get("max_restarts") or 3)
        if streak >= limit:
            record("worker_stable", FAIL, f"failure_streak {streak} of {limit}")
        elif streak:
            record("worker_stable", FAIL, f"failure_streak {streak} of {limit}")
        else:
            record("worker_stable", PASS, "failure_streak 0")

    # 7. THE BAR. Everything above can be green while nothing gets done.
    if mission is None:
        record("progress", UNKNOWN, "no mission state")
    else:
        accepted = int(mission.get("accepted_count") or 0)
        units = mission.get("units") or {}
        completed = [
            uid
            for uid, unit in units.items()
            if isinstance(unit, dict) and unit.get("status") == "completed"
        ]
        if accepted > 0 or completed:
            record(
                "progress",
                PASS,
                f"accepted={accepted}, completed={len(completed)}",
            )
        else:
            record(
                "progress",
                FAIL,
                f"accepted=0, 0 completed of {len(units)} units -- alive is not progress",
            )

    failed = sorted(k for k, v in criteria.items() if v["verdict"] == FAIL)
    unknown = sorted(k for k, v in criteria.items() if v["verdict"] == UNKNOWN)
    if failed:
        verdict = FAIL
    elif unknown:
        verdict = UNKNOWN
    else:
        verdict = PASS

    return {
        "schema": "hcli.cycle_verdict.v1",
        "verdict": verdict,
        "failed": failed,
        "unknown": unknown,
        "criteria": criteria,
        "note": (
            "UNKNOWN is not PASS. A criterion with no evidence on disk has not "
            "been satisfied, it has not been checked."
        ),
    }


def render(verdict: Dict[str, Any]) -> str:
    """One screen, widest signal first."""
    lines = [f"RUN VERDICT: {verdict.get('verdict')}"]
    for name, row in sorted(verdict.get("criteria", {}).items()):
        lines.append(f"  {row['verdict']:<8} {name:<26} {row['detail']}")
    return "\n".join(lines)
