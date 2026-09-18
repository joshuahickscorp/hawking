"""Read-only H-Web projection and bounded canonical event journal.

The browser is deliberately not a store.  These views are rebuilt from the
existing Session, Goal, WorkunitOwner, share, and capability owners.  The
journal only orders projection changes for reconnect; it never grants
authority or reconstructs domain state after a crash.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

from .persist import atomic_write_json, atomic_write_text

WEB_SCHEMA = "hawking.web.contract.v1"
EVENT_SCHEMA = "hawking.web.event.v1"
EVENT_PAYLOAD_VERSION = 1
MAX_REPLAY_EVENTS = 64
MAX_RETAINED_EVENTS = 512


def _json(value: Any) -> Any:
    """Return a stable, bounded JSON-safe representation for a view digest."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, default=str))
    except (TypeError, ValueError):
        return str(value)


def state_revision(value: Any) -> str:
    payload = json.dumps(_json(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _goal_identity(goal: Mapping[str, Any], owner: Mapping[str, Any]) -> tuple[str, str, str]:
    goal_id = str(goal.get("goal_id") or "")
    workunit_id = str(owner.get("workunit_id") or goal.get("workunit_id") or goal_id)
    worker_id = str(owner.get("worker_id") or "")
    return goal_id, workunit_id, worker_id


def project_budget(goal: Mapping[str, Any], owner: Mapping[str, Any]) -> Dict[str, Any]:
    plan = goal.get("budget_plan") if isinstance(goal.get("budget_plan"), Mapping) else {}
    cap = goal.get("budget_authorized_usd")
    if cap is None:
        cap = plan.get("combined_usd")
    try:
        cap_value = float(cap) if cap is not None else None
    except (TypeError, ValueError):
        cap_value = None
    try:
        spent = max(0.0, float(owner.get("worker_cost_usd") or owner.get("cost_usd") or 0.0))
    except (TypeError, ValueError):
        spent = 0.0
    try:
        reserved = max(0.0, float(owner.get("budget_reserved_usd") or 0.0))
    except (TypeError, ValueError):
        reserved = 0.0
    return {
        "cap_usd": cap_value,
        "spent_usd": spent,
        "reserved_usd": reserved,
        "remaining_usd": max(0.0, cap_value - spent - reserved) if cap_value is not None else None,
        "authority": "hawking.goal.budget",
    }


def project_goal(goal: Mapping[str, Any], owner: Mapping[str, Any], result: Any = None) -> Dict[str, Any]:
    goal_id, workunit_id, worker_id = _goal_identity(goal, owner)
    workunit_ids = [
        str(item).strip() for item in (goal.get("workunit_ids") or [])
        if str(item).strip()
    ] if isinstance(goal.get("workunit_ids"), list) else []
    if workunit_id and workunit_id not in workunit_ids:
        workunit_ids.append(workunit_id)
    artifacts = []
    if isinstance(result, Mapping):
        artifacts.append("result:" + goal_id)
    raw = {
        "schema": "hawking.web.goal.v1",
        "goal_id": goal_id,
        "objective": str(goal.get("objective") or "")[:1600],
        "status": str(goal.get("status") or owner.get("state") or "UNKNOWN"),
        "phase": str(goal.get("phase") or owner.get("current_phase") or "") or None,
        "workunit_ids": workunit_ids[-32:],
        # These are durable scheduler projections, not a second scheduler.
        # Keep the full parallel frontier visible so H-Web does not collapse
        # a multi-lane Goal to the legacy scalar workunit hint.
        "active_workunit_id": str(goal.get("active_workunit_id") or "") or None,
        "active_workunit_ids": [
            str(item).strip() for item in (goal.get("active_workunit_ids") or [])
            if str(item).strip()
        ][-32:] if isinstance(goal.get("active_workunit_ids"), list) else [],
        "runnable_frontier": [
            {
                "workunit_id": str(item.get("workunit_id") or "").strip(),
                "state": str(item.get("state") or "RUNNABLE"),
                "dependencies": [str(dep) for dep in (item.get("dependencies") or [])],
            }
            for item in (goal.get("runnable_frontier") or [])
            if isinstance(item, Mapping) and str(item.get("workunit_id") or "").strip()
        ][-32:] if isinstance(goal.get("runnable_frontier"), list) else [],
        "frontier_waiting_dependencies": [
            {
                "workunit_id": str(item.get("workunit_id") or "").strip(),
                "dependencies": [str(dep) for dep in (item.get("dependencies") or [])],
            }
            for item in (goal.get("frontier_waiting_dependencies") or [])
            if isinstance(item, Mapping) and str(item.get("workunit_id") or "").strip()
        ][-32:] if isinstance(goal.get("frontier_waiting_dependencies"), list) else [],
        "worker_ids": [worker_id] if worker_id else [],
        "budget": project_budget(goal, owner),
        "recent_event_ids": [],
        "result_artifact_ids": artifacts,
    }
    raw["state_revision"] = state_revision(raw)
    return raw


def project_workunit(goal: Mapping[str, Any], owner: Mapping[str, Any]) -> Dict[str, Any]:
    goal_id, workunit_id, worker_id = _goal_identity(goal, owner)
    evidence = owner.get("evidence") if isinstance(owner.get("evidence"), list) else []
    raw = {
        "schema": "hawking.web.workunit.v1",
        "workunit_id": workunit_id,
        "goal_id": goal_id,
        "objective": str(owner.get("objective") or goal.get("objective") or "")[:1600],
        "role": str(owner.get("role") or "worker"),
        "status": str(owner.get("state") or goal.get("status") or "UNKNOWN"),
        "worker_id": worker_id or None,
        "worker_attempt_id": str(owner.get("worker_attempt_id") or "") or None,
        "evidence_ids": [f"evidence:{workunit_id}:{index}" for index, _ in enumerate(evidence[-16:], 1)],
    }
    raw["state_revision"] = state_revision(raw)
    return raw


def project_worker(goal: Mapping[str, Any], owner: Mapping[str, Any]) -> Dict[str, Any]:
    goal_id, workunit_id, worker_id = _goal_identity(goal, owner)
    model = str(owner.get("worker_model") or goal.get("resident_assignment") or "hawking/unknown")
    raw = {
        "schema": "hawking.web.worker.v1",
        "worker_id": worker_id or f"unassigned:{workunit_id}",
        "worker_attempt_id": str(owner.get("worker_attempt_id") or "") or None,
        "goal_id": goal_id,
        "workunit_id": workunit_id,
        "model": model,
        "provider": "openrouter" if "/" in model else None,
        "status": str(owner.get("worker_status") or owner.get("state") or goal.get("status") or "UNKNOWN"),
        "capability_ids": ["hawking.tool-registry"],
        "cost_usd": project_budget(goal, owner)["spent_usd"],
    }
    raw["state_revision"] = state_revision(raw)
    return raw


def project_capabilities(workspace: str | os.PathLike[str] | None = None) -> list[Dict[str, Any]]:
    try:
        from .capabilities import capability_map

        known = capability_map(workspace).get("can", {})
    except Exception:
        known = {}
    return [
        {
            "schema": "hawking.web.capability.v1",
            "capability_id": f"hawking.{name}",
            "schema_version": "v1",
            "effects": ["read"] if bool(value) else [],
            "available": bool(value),
            "health": "READY" if bool(value) else "UNAVAILABLE",
            "expansion_route": None,
        }
        for name, value in sorted(known.items())[:64]
    ]


def project_artifact(goal: Mapping[str, Any], result: Any) -> list[Dict[str, Any]]:
    if not isinstance(result, Mapping):
        return []
    goal_id = str(goal.get("goal_id") or "")
    raw = {
        "schema": "hawking.web.artifact.v1",
        "artifact_id": f"result:{goal_id}",
        "kind": str(result.get("schema") or "hawking.goal.result"),
        "content_sha256": result.get("content_sha256") or result.get("digest"),
        "path": None,
        "redacted": True,
        "provenance_ids": [goal_id] if goal_id else [],
    }
    return [raw]


def project_route(session: Any) -> Dict[str, Any]:
    ui = getattr(session, "ui", {}) if session is not None else {}
    ui = dict(ui) if isinstance(ui, Mapping) else {}
    last = ui.get("last_auto_route") if isinstance(ui.get("last_auto_route"), Mapping) else {}
    selection = str(getattr(session, "model", None) or "hawking/auto")
    raw = {
        "schema": "hawking.web.route.v1",
        "decision_id": "route:" + str(getattr(session, "id", "unknown")),
        "source_scope": str(ui.get("route_scope") or "cloud"),
        "selection": selection,
        "resolved_model": last.get("selected_model") or (selection if selection != "hawking/auto" else None),
        "worker_id": None,
        "rationale": str(last.get("reason") or "Hawking-owned route policy"),
        "policy_revision": str(last.get("policy") or "hawking.auto.v1"),
    }
    raw["state_revision"] = state_revision(raw)
    return raw


def project_share(workspace: str | os.PathLike[str], session_id: str) -> Dict[str, Any]:
    root = Path(workspace) / ".hawking" / "shares"
    count = 0
    try:
        for path in root.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, Mapping) and not data.get("revoked"):
                count += 1
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return {
        "schema": "hawking.web.share.v1",
        "scope": "conversation",
        "target_id": session_id,
        "active_capability_count": count,
        "authority": "hawking.share_bridge",
    }


def project_snapshot(
    workspace: str | os.PathLike[str], session: Any, goal_snapshots: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    goals: list[Dict[str, Any]] = []
    workunits: list[Dict[str, Any]] = []
    workers: list[Dict[str, Any]] = []
    artifacts: list[Dict[str, Any]] = []
    for item in goal_snapshots:
        goal = item.get("goal") if isinstance(item.get("goal"), Mapping) else {}
        owner = item.get("workunit") if isinstance(item.get("workunit"), Mapping) else {}
        result = item.get("result")
        if not goal:
            continue
        goals.append(project_goal(goal, owner, result))
        projected_rows = item.get("workunits") if isinstance(item.get("workunits"), list) else []
        if projected_rows:
            for row in projected_rows:
                if isinstance(row, Mapping):
                    workunits.append(project_workunit(goal, row))
                    workers.append(project_worker(goal, row))
        else:
            workunits.append(project_workunit(goal, owner))
            workers.append(project_worker(goal, owner))
        artifacts.extend(project_artifact(goal, result))
    route = project_route(session)
    messages = getattr(session, "messages", []) if session is not None else []
    messages = messages if isinstance(messages, list) else []
    latest_message_seq = max(
        (int(item.get("seq") or 0) for item in messages if isinstance(item, Mapping)),
        default=0,
    )
    session_view = {
        "schema": "hawking.web.session.v1",
        "session_id": str(getattr(session, "id", "")),
        "conversation_id": str(getattr(session, "id", "")),
        "source_scope": route["source_scope"],
        "selection": route["selection"],
        "auto_enabled": route["selection"] == "hawking/auto",
        "active_goal_ids": [item["goal_id"] for item in goals],
        # Conversation content remains in the Session snapshot, not a Web
        # event payload.  These durable cursors make the event revision change
        # when a streamed user/assistant turn is committed.
        "turn_count": len(messages),
        "latest_message_seq": latest_message_seq,
    }
    session_view["state_revision"] = state_revision(session_view)
    view = {
        "schema": WEB_SCHEMA,
        "version": 1,
        "session": session_view,
        "goals": goals,
        "workunits": workunits,
        "workers": workers,
        "capabilities": project_capabilities(workspace),
        "artifacts": artifacts,
        "route_decision": route,
        "budget": project_budget({}, {}),
        "share": project_share(workspace, session_view["session_id"]),
    }
    view["state_revision"] = state_revision(view)
    return view


class WebEventJournal:
    """Durable ordered notification projection with no domain-state ownership."""

    def __init__(self, workspace: str | os.PathLike[str], session_id: str) -> None:
        self.root = Path(workspace).resolve() / ".hawking" / "web-events"
        safe = "".join(char for char in str(session_id) if char.isalnum() or char in "-_")[:96]
        if not safe:
            raise ValueError("session id is required for event projection")
        self.events_path = self.root / f"{safe}.jsonl"
        self.state_path = self.root / f"{safe}.state.json"
        self.lock_path = self.root / f"{safe}.lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _events(self) -> list[Dict[str, Any]]:
        try:
            rows = [json.loads(line) for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError, json.JSONDecodeError):
            return []
        return [dict(row) for row in rows if isinstance(row, Mapping)]

    def _state(self, rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            data = {}
        sequences = [int(row.get("sequence") or 0) for row in rows]
        return {
            "schema": "hawking.web.event-journal.v1",
            "next_sequence": max(int(data.get("next_sequence") or 1), max(sequences, default=0) + 1),
            "earliest_sequence": int(data.get("earliest_sequence") or min(sequences, default=1)),
        }

    def append(self, *, event_type: str, identity: Mapping[str, Any], state_rev: str,
               payload: Mapping[str, Any], terminal: bool = False) -> Dict[str, Any]:
        """Append exactly once per current canonical object revision."""
        dedupe = state_revision({"type": event_type, "identity": identity, "source_revision": state_rev})
        with self._locked():
            rows = self._events()
            for row in rows:
                if row.get("dedupe_key") == dedupe:
                    return row
            state = self._state(rows)
            sequence = int(state["next_sequence"])
            clean_identity = {str(key): str(value) for key, value in identity.items() if value not in (None, "")}
            event = {
                "schema": EVENT_SCHEMA,
                "event_id": f"WEB-{sequence}-{hashlib.sha256(dedupe.encode()).hexdigest()[:12]}",
                "sequence": sequence,
                "identity": clean_identity,
                "type": str(event_type),
                # Kept beside the general identity object for the typed Rust
                # DTO and existing compact H-Web consumers.
                "session_id": clean_identity.get("session_id", ""),
                "goal_id": clean_identity.get("goal_id") or None,
                "workunit_id": clean_identity.get("workunit_id") or None,
                "worker_id": clean_identity.get("worker_id") or None,
                "timestamp": _iso_now(),
                "payload_version": EVENT_PAYLOAD_VERSION,
                "source_revision": state_rev,
                "terminal": bool(terminal),
                "payload": _json(payload),
                "dedupe_key": dedupe,
            }
            rows.append(event)
            if len(rows) > MAX_RETAINED_EVENTS:
                rows = rows[-MAX_RETAINED_EVENTS:]
                atomic_write_text(self.events_path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            else:
                with self.events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            state["next_sequence"] = sequence + 1
            state["earliest_sequence"] = int(rows[0]["sequence"]) if rows else sequence + 1
            atomic_write_json(self.state_path, state)
            return event

    def sync(self, view: Mapping[str, Any]) -> None:
        session = view.get("session") if isinstance(view.get("session"), Mapping) else {}
        session_id = str(session.get("session_id") or "")
        self.append(event_type="session.updated", identity={"session_id": session_id},
                    state_rev=str(session.get("state_revision") or view.get("state_revision") or ""), payload={"session": session})
        for kind, plural, key in (("goal", "goals", "goal_id"), ("workunit", "workunits", "workunit_id"), ("worker", "workers", "worker_id")):
            for item in view.get(plural, []) if isinstance(view.get(plural), list) else []:
                if not isinstance(item, Mapping):
                    continue
                status = str(item.get("status") or "").upper()
                self.append(event_type=f"{kind}.updated", identity={"session_id": session_id, key: item.get(key), "goal_id": item.get("goal_id")},
                            state_rev=str(item.get("state_revision") or state_revision(item)), payload={kind: item},
                            terminal=status in {"COMPLETE", "FAILED", "CANCELLED", "BLOCKED", "UNKNOWN_OUTCOME"})
        route = view.get("route_decision") if isinstance(view.get("route_decision"), Mapping) else {}
        self.append(event_type="route.updated", identity={"session_id": session_id, "decision_id": route.get("decision_id")},
                    state_rev=str(route.get("state_revision") or state_revision(route)), payload={"route_decision": route})

    def replay(self, after_sequence: int, limit: int = MAX_REPLAY_EVENTS) -> Dict[str, Any]:
        after = max(0, int(after_sequence))
        cap = min(MAX_REPLAY_EVENTS, max(1, int(limit)))
        with self._locked():
            rows = sorted(self._events(), key=lambda row: int(row.get("sequence") or 0))
            state = self._state(rows)
        earliest = int(state["earliest_sequence"])
        eligible = [row for row in rows if int(row.get("sequence") or 0) > after]
        gap = bool(after and after < earliest - 1)
        overflow = len(eligible) > cap
        retained = eligible[-cap:] if overflow else eligible
        through = int(retained[-1].get("sequence") or after) if retained else after
        return {
            "schema": WEB_SCHEMA,
            "from_sequence": after,
            "through_sequence": through,
            "events": retained,
            "gap": gap,
            "snapshot_required": gap or overflow,
            "max_events": cap,
            "earliest_sequence": earliest,
        }
