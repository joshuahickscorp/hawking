"""Cheap, event-driven supervision for a durable Hawking Goal.

This module is deliberately a watcher, not a scheduler.  It reads compact
canonical Goal/WorkUnit state, caches the governing objective by digest, and
returns a wake decision for the existing supervisor.  Unchanged state never
needs a model call or a prose heartbeat.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from hawking.goal_surface import load_goal, reconcile_goal_frontier
from hawking.persist import atomic_write_json


WATCHER_SCHEMA = "hawking.supervisor_watcher.v1"
_WAKE_EVENTS = frozenset({
    "workunit.accepted",
    "workunit.failed.new_class",
    "frontier.changed",
    "goal.phase_changed",
    "mutation.ready",
    "review.disagreement",
    "test.regression",
    "budget.threshold",
    "resource.changed",
    "dependency.changed",
    "unknown_outcome",
    "authority.changed",
    "goal.blocked",
    "operator.steer",
})
_KNOWN_CHILD_FAILURES = frozenset({
    "PROVIDER_OUTPUT_UNUSABLE",
    "OUTPUT_UNUSABLE",
    "PROVIDER_TOOL_NONCOMPLIANCE",
})


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _budget_band(goal: Mapping[str, Any]) -> str:
    """Make tiny spend changes quiet while retaining meaningful budget changes."""
    raw = goal.get("spent_usd", goal.get("spend_usd", 0))
    try:
        return f"{float(raw or 0):.2f}"
    except (TypeError, ValueError):
        return "unknown"


def _source_revision(root: Path) -> str:
    """Read only the compact repository revision; never fingerprint the tree."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=2, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def objective_digest(path: str | Path) -> str | None:
    """Hash the governing objective without retaining its contents in state."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def batch_worker_results(results: Iterable[Mapping[str, Any]], *, known_failures: Iterable[str] = ()) -> dict[str, Any]:
    """Collapse close child results into one compact supervisor wake packet.

    This is bookkeeping only: it never changes WorkUnit state or starts work.
    A repeated known provider failure is retained in the batch but does not
    request deep review; a new failure class or accepted result does.
    """
    rows = [dict(row) for row in results if isinstance(row, Mapping)]
    classes = {str(row.get("failure_class") or row.get("failure") or "").upper() for row in rows}
    classes.discard("")
    known = {str(item).upper() for item in known_failures} | _KNOWN_CHILD_FAILURES
    new_failures = sorted(classes - known)
    accepted = [str(row.get("workunit_id") or "") for row in rows if str(row.get("state") or "").upper() == "COMPLETE"]
    return {
        "schema": WATCHER_SCHEMA,
        "batch_id": "BATCH-" + _digest([row.get("workunit_id") for row in rows])[:16],
        "count": len(rows),
        "workunit_ids": [str(row.get("workunit_id") or "") for row in rows],
        "accepted_workunits": [item for item in accepted if item],
        "failure_classes": sorted(classes),
        "new_failure_classes": new_failures,
        "known_failure_only": bool(classes) and not new_failures and not accepted,
        "wake_supervisor": bool(accepted or new_failures),
        "tier": "LIGHT_REVIEW" if (accepted or new_failures) else "WATCHER",
    }


def cache_objective(root: str | Path, goal_id: str, objective_path: str | Path) -> dict[str, Any]:
    """Persist the objective binding used by future watcher cycles."""
    workspace = Path(root).expanduser().resolve()
    digest = objective_digest(objective_path)
    path = workspace / ".hawking" / "supervision" / f"{goal_id}.objective.json"
    current = {
        "schema": WATCHER_SCHEMA,
        "goal_id": str(goal_id),
        "objective_path": str(Path(objective_path).expanduser().resolve()),
        "objective_digest": digest,
        "cached_at": time.time(),
    }
    previous: dict[str, Any] = {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            previous = value
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    if previous.get("objective_digest") == digest and previous.get("objective_path") == current["objective_path"]:
        current["cached_at"] = previous.get("cached_at", current["cached_at"])
    atomic_write_json(path, current)
    return current


def compact_state(root: str | Path, goal_id: str) -> dict[str, Any]:
    """Build the bounded state used for a watcher fingerprint."""
    workspace = Path(root).expanduser().resolve()
    goal = load_goal(workspace, goal_id) or {}
    projection = reconcile_goal_frontier(workspace, goal_id, persist=False)
    live = projection.get("goal") or goal
    frontier = projection.get("frontier") or []
    waiting = projection.get("waiting") or []
    return {
        "goal_id": str(goal_id),
        "revision": live.get("revision"),
        "phase": live.get("phase"),
        "status": live.get("status"),
        "active_workunit_ids": projection.get("active_workunit_ids") or [],
        "runnable_frontier": frontier,
        "waiting_dependencies": waiting,
        "accepted_heads": live.get("accepted_heads") or live.get("accepted_workunits") or [],
        "blockers": live.get("blockers") or live.get("blocked_reasons") or [],
        "source_revision": live.get("source_revision") or _source_revision(workspace),
        "budget_band": _budget_band(live),
        "resource_state": live.get("resource_state") or live.get("leases") or {},
    }


def supervisor_checkpoint(root: str | Path, goal_id: str, *, objective_path: str | Path | None = None) -> dict[str, Any]:
    """Return the bounded post-compaction supervisor contract.

    This is a state projection, not a model call and never includes the
    objective body.  An objective digest is included only when the caller has
    an explicit bound path, so unchanged objectives remain cache hits.
    """
    state = compact_state(root, goal_id)
    workspace = Path(root).expanduser().resolve()
    checkpoint = {
        "schema": "hawking.supervisor_checkpoint.v1",
        "root": str(goal_id),
        "programme_graph_revision": state.get("revision"),
        "current_frontier": state.get("runnable_frontier") or [],
        "accepted_heads": state.get("accepted_heads") or [],
        "new_blockers": state.get("blockers") or [],
        "active_effects": state.get("resource_state") or {},
        "budget_state": state.get("budget_band"),
        "source_revision": state.get("source_revision"),
    }
    if objective_path is not None:
        checkpoint["objective_digest"] = objective_digest(objective_path)
    return checkpoint


class SupervisorWatcher:
    """State-fingerprint watcher feeding the existing supervisor tiers."""

    def __init__(self, root: str | Path, goal_id: str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.goal_id = str(goal_id)
        self.state_path = self.root / ".hawking" / "supervision" / f"{self.goal_id}.state.json"

    def snapshot(self) -> dict[str, Any]:
        state = compact_state(self.root, self.goal_id)
        state["fingerprint"] = _digest(state)
        state["schema"] = WATCHER_SCHEMA
        return state

    def decide(self, *, events: Iterable[str] = ()) -> dict[str, Any]:
        current = self.snapshot()
        previous: dict[str, Any] = {}
        try:
            value = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                previous = value
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        event_set = {str(event) for event in events}
        changed = current["fingerprint"] != previous.get("fingerprint")
        urgent = bool(event_set & _WAKE_EVENTS)
        wake = changed or urgent or not previous
        reasons = []
        if not previous:
            reasons.append("initial_state")
        if changed and previous:
            reasons.append("state_fingerprint_changed")
        reasons.extend(sorted(event_set & _WAKE_EVENTS))
        result = {
            "schema": WATCHER_SCHEMA,
            "goal_id": self.goal_id,
            "tier": "LIGHT_REVIEW" if wake else "WATCHER",
            "wake_supervisor": wake,
            "state_changed": changed,
            "fingerprint": current["fingerprint"],
            "previous_fingerprint": previous.get("fingerprint"),
            "reasons": list(dict.fromkeys(reasons)),
            "state": current,
        }
        atomic_write_json(self.state_path, current)
        return result

    def record_delta(self, decision: Mapping[str, Any]) -> bool:
        """Append one durable report row only when the state fingerprint changes."""
        if not decision.get("wake_supervisor"):
            return False
        path = self.root / ".hawking" / "supervision" / f"{self.goal_id}.deltas.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "schema": WATCHER_SCHEMA,
            "at": time.time(),
            "goal_id": self.goal_id,
            "fingerprint": decision.get("fingerprint"),
            "reasons": decision.get("reasons") or [],
            "state": decision.get("state") or {},
        }
        prior_fingerprint = decision.get("previous_fingerprint")
        if prior_fingerprint and prior_fingerprint == row["fingerprint"]:
            return False
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        return True


__all__ = [
    "WATCHER_SCHEMA",
    "SupervisorWatcher",
    "cache_objective",
    "compact_state",
    "supervisor_checkpoint",
    "objective_digest",
    "batch_worker_results",
]
