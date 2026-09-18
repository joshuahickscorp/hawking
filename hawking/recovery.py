"""Canonical restart/recovery inspection; it never replays a mutation or spawns work."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .goal_surface import list_goals, load_goal, update_goal
from .hawkingd import daemon_lease_metadata
from .mutation import reconcile_mutation_intent
from .workunit_owner import WorkunitOwner


def _intent_ids(root: Path) -> Iterable[str]:
    for path in sorted((root / ".hawking" / "mutation-intents").glob("*.json")):
        yield path.stem


def recovery_status(workspace: str | Path) -> Dict[str, Any]:
    root = Path(workspace).resolve()
    goals = list_goals(root, limit=128)
    owner = WorkunitOwner(root)
    workers = []
    for goal in goals:
        wid = str(goal.get("workunit_id") or goal.get("goal_id") or "")
        if not wid:
            continue
        try:
            status = owner.status(wid)
        except Exception as exc:
            status = {"workunit_id": wid, "state": "UNKNOWN_OUTCOME", "recovery_error": type(exc).__name__}
        workers.append({"goal_id": goal.get("goal_id"), "workunit": status})
    intents = []
    for request_id in _intent_ids(root):
        try:
            raw = json.loads((root / ".hawking" / "mutation-intents" / f"{request_id}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            raw = {"status": "UNKNOWN_OUTCOME"}
        intents.append({"request_id": request_id, "status": raw.get("status", "UNKNOWN_OUTCOME")})
    return {
        "schema": "hawking.recovery.status.v1",
        "daemon_lease": daemon_lease_metadata(),
        "goals": goals,
        "workers": workers,
        "mutation_intents": intents,
        "claim_boundary": "read-only recovery view; no worker is resumed or mutation replayed",
    }


def reconcile_recovery(workspace: str | Path, *, goal_id: Optional[str] = None) -> Dict[str, Any]:
    """Reconcile observable outcomes only, retaining uncertainty when evidence is absent."""
    root = Path(workspace).resolve()
    goals = [load_goal(root, goal_id)] if goal_id else list_goals(root, limit=128)
    goals = [goal for goal in goals if isinstance(goal, dict)]
    owner = WorkunitOwner(root)
    reconciled_goals = []
    for goal in goals:
        gid = str(goal.get("goal_id") or "")
        wid = str(goal.get("workunit_id") or gid)
        try:
            status = owner.status(wid)
        except Exception as exc:
            status = {"workunit_id": wid, "state": "UNKNOWN_OUTCOME", "recovery_error": type(exc).__name__}
        state = str(status.get("state") or "UNKNOWN_OUTCOME").upper()
        # A dead driver is already converted to BLOCKED by the owner.  Preserve
        # that safe checkpoint in the Goal record, but never promote it to
        # complete and never auto-resume it after a daemon restart.
        if state in {"BLOCKED", "UNKNOWN_OUTCOME"} and str(goal.get("status") or "").upper() == "RUNNING":
            update_goal(root, gid, status="BLOCKED", phase="RECOVERY_REQUIRED", recovery_state=state)
        reconciled_goals.append({"goal_id": gid, "workunit": status, "action": "checkpoint_preserved" if state in {"BLOCKED", "UNKNOWN_OUTCOME"} else "no_change"})
    mutations = [reconcile_mutation_intent(root, request_id) for request_id in _intent_ids(root)]
    return {
        "schema": "hawking.recovery.reconcile.v1",
        "goals": reconciled_goals,
        "mutation_intents": mutations,
        "claim_boundary": "unknown outcomes remain unknown until deterministic evidence proves otherwise; no mutation or worker was replayed",
    }
