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


#: Named so they can be argued with. A budget buried inside a comparison is a
#: number nobody can find and nobody revises.
DEFAULT_BUDGETS: Dict[str, float] = {
    "tool_p95_ms": 50.0,
    "mean_rounds_per_goal": 4.0,
    "effective_prompt_tps": 100.0,
    "realized_reuse_fraction": 0.5,
}


def evaluate(
    workspace: Any,
    budgets: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """One verdict per criterion, plus the run verdict, from durable state only."""
    budgets = {**DEFAULT_BUDGETS, **(budgets or {})}
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

    # ---------------------------------------------------------------
    # SPEED. Working is not the bar; these subsystems must be FAST.
    # Budgets are named and tunable rather than buried in a comparison,
    # and the MEASURED value is always reported so the number is the
    # signal even when the colour is not.
    # ---------------------------------------------------------------
    measured: Dict[str, Any] = {}
    calls: List[Dict[str, Any]] = []
    for path in receipts:
        body = _read_json(path) or {}
        for call in body.get("model_calls") or []:
            if isinstance(call, dict):
                calls.append(call)

    # Tool latency. These are local function calls; milliseconds or it is broken.
    tool_ms = sorted(
        float((r.get("data") or {}).get("elapsed_s") or 0.0) * 1000.0
        for r in events
        if r.get("type") == "tool_call_finished"
    )
    if not tool_ms:
        record("tools_fast", UNKNOWN, "no tool calls yet")
    else:
        p95 = tool_ms[min(len(tool_ms) - 1, int(len(tool_ms) * 0.95))]
        measured["tool_p95_ms"] = round(p95, 3)
        budget = budgets["tool_p95_ms"]
        record(
            "tools_fast",
            PASS if p95 <= budget else FAIL,
            f"p95 {p95:.1f} ms over {len(tool_ms)} calls (budget {budget} ms)",
        )

    # Round trips per goal. A round is a model call; the tools it drives are
    # milliseconds, so the round count IS the wall clock of a goal.
    per_goal: Dict[str, int] = {}
    for row in events:
        if row.get("type") == "model_call_finished":
            gid = str((row.get("data") or {}).get("goal_id") or "")
            per_goal[gid] = per_goal.get(gid, 0) + 1
    if not per_goal:
        record("few_round_trips", UNKNOWN, "no model calls yet")
    else:
        mean_rounds = sum(per_goal.values()) / len(per_goal)
        measured["mean_rounds_per_goal"] = round(mean_rounds, 2)
        budget = budgets["mean_rounds_per_goal"]
        record(
            "few_round_trips",
            PASS if mean_rounds <= budget else FAIL,
            f"{mean_rounds:.1f} model calls per goal over {len(per_goal)} goal(s) "
            f"(budget {budget})",
        )

    # Effective prefill rate: prompt tokens the caller asked for, per second of
    # wall. KV reuse raises this WITHOUT the kernel getting faster, which is the
    # point -- this is the number a caller experiences.
    rated = [
        (float(c.get("prompt_tokens") or 0), float(c.get("wall_s") or 0.0))
        for c in calls
        if c.get("prompt_tokens") and c.get("wall_s")
    ]
    if not rated:
        record("prefill_fast", UNKNOWN, "no timed model calls yet")
    else:
        tps = sum(t for t, _ in rated) / max(1e-9, sum(w for _, w in rated))
        measured["effective_prompt_tps"] = round(tps, 1)
        budget = budgets["effective_prompt_tps"]
        record(
            "prefill_fast",
            PASS if tps >= budget else FAIL,
            f"{tps:.0f} prompt tok/s over {len(rated)} calls (budget {budget})",
        )

    # KV reuse, from the resident's own count. Never inferred from a clock.
    reused = [
        (float(c.get("prefix_reused_tokens") or 0), float(c.get("prompt_tokens") or 0))
        for c in calls
        if c.get("prefix_reused_tokens") is not None and c.get("prompt_tokens")
    ]
    if not reused:
        record(
            "kv_reuse",
            UNKNOWN,
            "the resident reported no prefix_reused_tokens; unmeasured, NOT zero",
        )
    else:
        fraction = sum(r for r, _ in reused) / max(1e-9, sum(t for _, t in reused))
        measured["realized_reuse_fraction"] = round(fraction, 4)
        budget = budgets["realized_reuse_fraction"]
        record(
            "kv_reuse",
            PASS if fraction >= budget else FAIL,
            f"{fraction:.2f} of prompt tokens reused over {len(reused)} calls "
            f"(budget {budget})",
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
        "measured": measured,
        "budgets": budgets,
        "failed": failed,
        "unknown": unknown,
        "criteria": criteria,
        "note": (
            "UNKNOWN is not PASS. A criterion with no evidence on disk has not "
            "been satisfied, it has not been checked."
        ),
    }


def render(verdict: Dict[str, Any]) -> str:
    """One screen, widest signal first. The MEASURED numbers always show."""
    lines = [f"RUN VERDICT: {verdict.get('verdict')}"]
    for name, row in sorted(verdict.get("criteria", {}).items()):
        lines.append(f"  {row['verdict']:<8} {name:<26} {row['detail']}")
    measured = verdict.get("measured") or {}
    if measured:
        lines.append("  measured: " + "  ".join(
            f"{k}={v}" for k, v in sorted(measured.items())
        ))
    return "\n".join(lines)
