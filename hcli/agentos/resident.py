"""Resident HCLI daemon and its low-risk qualification harness.

The resident is a control-plane lifecycle, not a second model implementation.
The supervisor is intentionally small and model-free.  It owns one worker
process at a time; the worker may construct :class:`AgentOS`, run a bounded
mission slice, and then exit.  Mission/DAG state remains on disk, so unloading
the worker is ordinary operation rather than data loss.

The module also contains the rules for evidence-derived refill and child
process ownership.  These rules are useful without a live model and are the
ones used by the lightweight qualification tests.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None  # type: ignore[assignment]

try:
    import termios
except ImportError:  # pragma: no cover - Windows has no termios
    termios = None  # type: ignore[assignment]

from hcli.persist import atomic_write_json
from hcli.resources import process_start_token
from hcli.workunit import WorkUnit, AUTHOR_HCLI
from hcli.goal_bank import GoalBank
from hcli.steering import SteeringQueue
from hcli.agentos.event_sink import EventSink, read_events
from hcli.stream_render import render_event


SCHEMA = "hcli.agentos.resident_daemon.v1"
STATE_DIRNAME = "resident"
STATE_FILENAME = "state.json"
KNOWLEDGE_FILENAME = "knowledge.json"
INBOX_FILENAME = "inbox.json"
BODY_FILENAME = "body.json"
MAX_KNOWLEDGE = 256
MAX_CHILD_WORKUNITS = 8
MAX_CHILD_DESCRIPTION = 4000
DEFAULT_INTERVAL_S = 30.0
DEFAULT_EVACUATION_GRACE_S = 10.0
DEFAULT_MAX_RESTARTS = 3
# Keep the model-free daemon's default aligned with ``hcli.machine.MemGate``.
# An omitted resident setting must not turn a packed host's swap budget into an
# implicit unlimited budget just before a native body is admitted.
DEFAULT_SWAP_CEILING_BYTES = 2 * 1024**3
BEHAVIOR_SCHEMA = "hcli.agentos.resident_behavior.v1"
# _watch_unit_summary's cap on named running/failed units -- a module-level
# constant (not a local) so a mutation check can flip it without editing
# source.
WATCH_UNIT_SUMMARY_CAP = 12

# Set by start_resident on the supervisor it deliberately daemonises. Read once
# at startup so a detached supervisor is a recorded fact rather than whatever
# os.getppid() happened to return in the milliseconds before its launcher exited.
DETACHED_ENV = "HCLI_RESIDENT_DETACHED"


def _now() -> float:
    return time.time()


def _safe_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _metric_int(value: Any) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _effective_swap_ceiling(value: Optional[int]) -> int:
    """Resolve the daemon ceiling using the same default as RuntimePool."""
    if value is not None:
        return max(0, int(value))
    raw = os.environ.get("HCLI_SWAP_CEILING_GIB")
    if raw:
        try:
            return max(0, int(float(raw) * 1024**3))
        except (TypeError, ValueError):
            pass
    return DEFAULT_SWAP_CEILING_BYTES


def _pid_matches(pid: Any, token: Any) -> bool:
    number = _safe_int(pid)
    if number is None:
        return False
    try:
        os.kill(number, 0)
    except OSError:
        return False
    if token is None:
        return True
    observed = process_start_token(number)
    return observed is None or str(observed) == str(token)


def _json_safe(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def resident_dir(workspace: str | os.PathLike[str]) -> Path:
    return Path(workspace).expanduser().resolve() / ".hcli" / STATE_DIRNAME


def resident_state_path(workspace: str | os.PathLike[str]) -> Path:
    return resident_dir(workspace) / STATE_FILENAME


def resident_knowledge_path(workspace: str | os.PathLike[str]) -> Path:
    return resident_dir(workspace) / KNOWLEDGE_FILENAME


@dataclass
class ResidentConfig:
    """Persisted policy for one resident lifecycle."""

    workspace: str
    goal: str
    model: Optional[str] = None
    repo_root: Optional[str] = None
    runtime_count: int = 1
    interval_s: float = DEFAULT_INTERVAL_S
    evacuation_grace_s: float = DEFAULT_EVACUATION_GRACE_S
    max_restarts: int = DEFAULT_MAX_RESTARTS
    reserve_bytes: Optional[int] = None
    swap_ceiling_bytes: Optional[int] = None
    auto_restart: bool = True
    exit_when_orphaned: bool = True

    def __post_init__(self) -> None:
        self.workspace = str(Path(self.workspace).expanduser().resolve())
        self.goal = str(self.goal or "").strip()
        if not self.goal:
            raise ValueError("resident goal must not be empty")
        self.runtime_count = max(1, int(self.runtime_count))
        self.interval_s = max(0.1, min(24 * 3600.0, float(self.interval_s)))
        self.evacuation_grace_s = max(0.1, min(300.0, float(self.evacuation_grace_s)))
        self.max_restarts = max(0, min(20, int(self.max_restarts)))
        self.repo_root = (
            str(Path(self.repo_root).expanduser().resolve())
            if self.repo_root
            else None
        )
        if self.reserve_bytes is not None:
            self.reserve_bytes = max(0, int(self.reserve_bytes))
        if self.swap_ceiling_bytes is not None:
            self.swap_ceiling_bytes = max(0, int(self.swap_ceiling_bytes))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workspace": self.workspace,
            "goal": self.goal,
            "model": self.model,
            "repo_root": self.repo_root,
            "runtime_count": self.runtime_count,
            "interval_s": self.interval_s,
            "evacuation_grace_s": self.evacuation_grace_s,
            "max_restarts": self.max_restarts,
            "reserve_bytes": self.reserve_bytes,
            "swap_ceiling_bytes": self.swap_ceiling_bytes,
            "auto_restart": self.auto_restart,
            "exit_when_orphaned": self.exit_when_orphaned,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResidentConfig":
        return cls(
            workspace=str(value.get("workspace") or ""),
            goal=str(value.get("goal") or ""),
            model=(str(value["model"]) if value.get("model") else None),
            repo_root=(str(value["repo_root"]) if value.get("repo_root") else None),
            runtime_count=int(value.get("runtime_count") or 1),
            interval_s=float(value.get("interval_s") or DEFAULT_INTERVAL_S),
            evacuation_grace_s=float(
                value.get("evacuation_grace_s") or DEFAULT_EVACUATION_GRACE_S
            ),
            max_restarts=int(
                value["max_restarts"]
                if value.get("max_restarts") is not None
                else DEFAULT_MAX_RESTARTS
            ),
            reserve_bytes=(
                int(value["reserve_bytes"])
                if value.get("reserve_bytes") is not None
                else None
            ),
            swap_ceiling_bytes=(
                int(value["swap_ceiling_bytes"])
                if value.get("swap_ceiling_bytes") is not None
                else None
            ),
            auto_restart=bool(value.get("auto_restart", True)),
            exit_when_orphaned=bool(value.get("exit_when_orphaned", True)),
        )


class ResidentStore:
    """Atomic resident state and bounded self-knowledge storage."""

    def __init__(self, workspace: str | os.PathLike[str]) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = resident_dir(self.workspace)
        self.state_path = self.root / STATE_FILENAME
        self.knowledge_path = self.root / KNOWLEDGE_FILENAME
        self.inbox_path = self.root / INBOX_FILENAME
        self.body_path = self.root / BODY_FILENAME
        self.lock_path = self.root / ".resident.lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        """Serialize durable read-modify-write transactions across processes."""
        self.root.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+")
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            return None

    def _read_unlocked(self) -> Dict[str, Any]:
        value = self._read_json(self.state_path)
        return dict(value) if isinstance(value, dict) else {}

    def _write_unlocked(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(value)
        payload.setdefault("schema", SCHEMA)
        payload["updated_at"] = _now()
        atomic_write_json(self.state_path, payload)
        return payload

    def read(self) -> Dict[str, Any]:
        with self._locked():
            return self._read_unlocked()

    def write(self, value: Mapping[str, Any]) -> Dict[str, Any]:
        with self._locked():
            return self._write_unlocked(value)

    def update(self, **changes: Any) -> Dict[str, Any]:
        with self._locked():
            current = self._read_unlocked()
            current.update(changes)
            return self._write_unlocked(current)

    def append_knowledge(self, observation: Mapping[str, Any]) -> None:
        with self._locked():
            raw = self._read_json(self.knowledge_path)
            current: List[Any] = raw if isinstance(raw, list) else []
            current.append({"at": _now(), **_json_safe(dict(observation))})
            atomic_write_json(self.knowledge_path, current[-MAX_KNOWLEDGE:])

    def read_inbox(self) -> List[Dict[str, Any]]:
        with self._locked():
            value = self._read_json(self.inbox_path)
            return (
                [item for item in value[-64:] if isinstance(item, dict)]
                if isinstance(value, list)
                else []
            )

    def append_inbox(self, unit: WorkUnit) -> None:
        with self._locked():
            value = self._read_json(self.inbox_path)
            items = [item for item in value[-64:] if isinstance(item, dict)] if isinstance(value, list) else []
            atomic_write_json(self.inbox_path, items + [unit.to_dict()])

    def clear_inbox(self, count: int) -> None:
        with self._locked():
            value = self._read_json(self.inbox_path)
            items = [item for item in value[-64:] if isinstance(item, dict)] if isinstance(value, list) else []
            atomic_write_json(self.inbox_path, items[max(0, int(count)):])

    def body(self) -> Dict[str, Any]:
        with self._locked():
            value = self._read_json(self.body_path)
            return dict(value) if isinstance(value, dict) else {}

    def update_body(self, **changes: Any) -> Dict[str, Any]:
        with self._locked():
            raw = self._read_json(self.body_path)
            value = dict(raw) if isinstance(raw, dict) else {}
            value.update(changes)
            value.setdefault("schema", "hcli.agentos.resident_body.v1")
            value["updated_at"] = _now()
            atomic_write_json(self.body_path, value)
            return value


class ResidentBodyRegistry:
    """Durable physical-body census separate from logical mission state."""

    def __init__(self, workspace: str | os.PathLike[str]) -> None:
        self.store = ResidentStore(workspace)

    def register(self, *, model: Optional[str], runtime_count: int) -> Dict[str, Any]:
        return self.store.update_body(
            body_id=str(self.store.body().get("body_id") or f"body-{uuid.uuid4()}"),
            model=model,
            runtime_count=max(1, int(runtime_count)),
            status="CONFIGURED",
            loaded=False,
        )

    def mark_loaded(self, *, pid: Optional[int] = None) -> Dict[str, Any]:
        return self.store.update_body(
            status="LOADED",
            loaded=True,
            loaded_at=_now(),
            worker_pid=pid,
        )

    def mark_loading(self, *, pid: Optional[int] = None) -> Dict[str, Any]:
        return self.store.update_body(
            status="LOADING",
            loaded=False,
            load_started_at=_now(),
            worker_pid=pid,
        )

    def mark_unloaded(self, *, reason: str = "worker_exit") -> Dict[str, Any]:
        return self.store.update_body(
            status="UNLOADED",
            loaded=False,
            worker_pid=None,
            unloaded_at=_now(),
            unload_reason=reason,
        )


class ResidentAlreadyRunning(RuntimeError):
    """A live supervisor already owns this workspace.

    THE BUG THIS REPLACES: `configure()` returned the CURRENT state unchanged
    when a supervisor was live, and `start_resident` then returned
    `daemon.status()`. Neither wrote the requested config and neither said so,
    so `resident start --goal NEW --interval-s 30 --swap-ceiling 20G` printed a
    healthy-looking JSON status describing the INCUMBENT -- old goal, old
    interval, old ceiling, old mission -- and the operator reasonably believed
    the new goal had launched. Observed live at cycles=818 still resurrecting a
    smoke mission. Refusing loudly is the only honest option; `replace` is the
    explicit way to say "retire the incumbent and take over".
    """

    def __init__(self, state: Mapping[str, Any]) -> None:
        self.existing_pid = state.get("supervisor_pid")
        self.existing_mission = state.get("mission_id")
        config = state.get("config") if isinstance(state.get("config"), Mapping) else {}
        self.existing_goal = str(config.get("goal") or "")
        self.cycles = state.get("cycles")
        super().__init__(
            "RESIDENT_ALREADY_RUNNING\n"
            f"  existing_pid={self.existing_pid}\n"
            f"  existing_mission={self.existing_mission}\n"
            f"  existing_goal={self.existing_goal[:120]!r}\n"
            f"  cycles={self.cycles}\n"
            "  The requested configuration was NOT applied.\n"
            "  Use: hcli resident status | hcli resident stop | "
            "hcli resident replace --goal ..."
        )


def memory_decision(
    snapshot: Mapping[str, Any],
    *,
    reserve_bytes: Optional[int] = None,
    swap_ceiling_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """Classify host pressure without loading a model.

    Missing platform counters are treated as unknown, not as zero.  A
    sufficiently complete probe can still admit on an ``unknown`` pressure
    label, but an incomplete probe fails closed so a missing counter cannot
    grant a heavy model admission on a packed host. The omitted swap ceiling
    uses the same conservative default as RuntimePool (2 GiB), so the
    supervisor and the actual model admission gate cannot disagree silently.
    """
    pressure = str(snapshot.get("pressure") or "unknown").lower()
    total_observed = _metric_int(snapshot.get("total_bytes"))
    free_observed = _metric_int(snapshot.get("free_bytes"))
    swap_observed = _metric_int(snapshot.get("swap_used_bytes"))
    total = total_observed or 0
    free = free_observed or 0
    swap = swap_observed or 0
    reserve = reserve_bytes
    if reserve is None and total:
        reserve = max(12 * 1024**3, int(total * 0.15))
    reasons: List[str] = []
    if pressure == "high":
        reasons.append("host memory pressure is high")
    if reserve is not None and free_observed is not None and free < reserve:
        reasons.append(f"free RAM {free} is below reserve {reserve}")
    effective_swap_ceiling = _effective_swap_ceiling(swap_ceiling_bytes)
    if swap > effective_swap_ceiling:
        reasons.append(
            f"swap {swap} exceeds ceiling {effective_swap_ceiling}"
        )
    if pressure == "unknown" and (
        free_observed is None
        or total_observed is None
        or (total_observed == 0 and free_observed == 0)
    ):
        reasons.append("memory admission is unknown; waiting for a valid probe")
    return {
        "safe": not reasons,
        "reasons": reasons,
        "pressure": pressure,
        "total_bytes": total,
        "free_bytes": free,
        "swap_used_bytes": swap,
        "reserve_bytes": reserve,
        "swap_ceiling_bytes": effective_swap_ceiling,
    }


# A mission that has stopped for good. `no_progress` is included: the mission
# raised NO_PROGRESS against its own threshold, which is a give-up, not a pause.
# Defined in hcli.mission (it is a property of Mission.phase) and re-exported
# here so this module stays the name every existing caller imports it from.
from hcli.mission import TERMINAL_MISSION_PHASES  # noqa: E402

# ...of which these need a human before anything else can run. `completed` does
# not: a completed mission is re-run harmlessly and is how a queued bank goal
# gets promoted.
BLOCKED_MISSION_PHASES = frozenset({"failed", "cancelled", "no_progress"})


def mission_blocked_reason(workspace: Path) -> Optional[str]:
    """Why the durable mission cannot be advanced without a human, or None.

    Unit status is NOT the authority here. ``_mission_has_work`` counted a
    ``failed`` unit as work, so a mission that had already exhausted its own
    repair budget and set ``phase=failed`` looked like available work forever:
    the supervisor dispatched a worker every interval, the worker recovered the
    dead mission, returned its failure, and exited 0 -- so ``failure_streak``
    never rose and ``max_restarts`` never tripped. Observed at cycles=60 and
    climbing, once every 15s, with no path out. The mission's own phase is the
    authority on whether the mission is still runnable.
    """
    from hcli.mission import MissionCorruptError, load_state

    try:
        value = load_state(workspace / ".hcli" / "mission" / "state.json")
    except (FileNotFoundError, MissionCorruptError):
        return None
    phase = str(value.get("phase") or "")
    units = value.get("units") or {}
    # A root whose repair COMPLETED keeps its own `failed` status forever; the
    # repair is the mission's answer to it. Counting those made every mission
    # that ever repaired anything read as needing a human.
    repaired = {
        item.get("repairs")
        for item in units.values()
        if isinstance(item, Mapping)
        and item.get("status") == "completed"
        and item.get("repairs")
    }
    failed_units = [
        uid
        for uid, item in units.items()
        if isinstance(item, Mapping)
        and item.get("status") == "failed"
        and uid not in repaired
    ]
    if phase not in BLOCKED_MISSION_PHASES and not (phase == "completed" and failed_units):
        return None
    if phase == "completed" and failed_units:
        phase = "completed with failed units"
    return (
        f"durable mission {value.get('id')} is {phase} and cannot advance itself; "
        "archive .hcli/mission/state.json or start a new goal"
    )


def _mission_has_work(workspace: Path) -> bool:
    from hcli.mission import MissionCorruptError, load_state

    path = workspace / ".hcli" / "mission" / "state.json"
    try:
        value = load_state(path)
    except FileNotFoundError:
        # Mission genuinely absent: nothing to do.
        return False
    except MissionCorruptError:
        # The file exists but is unreadable/malformed. That is NOT "no
        # work" - agent.recover_mission() will hit this exact same
        # load_state() and raise, and the worker must be spawned so that
        # failure surfaces as a visible worker_failed/error instead of the
        # supervisor silently freezing in IDLE forever with no signal.
        return True
    # The mission's own phase outranks its unit statuses. A terminal mission
    # keeps `failed`/`pending` units on disk forever; reading those as work is
    # what span the supervisor at 15s intervals with nothing to do.
    if str(value.get("phase") or "") in TERMINAL_MISSION_PHASES:
        return False
    units = value.get("units")
    if not isinstance(units, dict):
        return False
    return any(
        isinstance(item, dict)
        and item.get("status") in {"pending", "ready", "running", "interrupted", "failed"}
        for item in units.values()
    )


def _inbox_has_work(workspace: Path) -> bool:
    """Return whether model-free queued work is waiting for the next worker."""
    return bool(ResidentStore(workspace).read_inbox())


def _goal_bank_snapshot(workspace: Path) -> Dict[str, Any]:
    """Return bounded bank state and recover goals from dead worker owners."""
    bank = GoalBank(workspace)
    try:
        recovered = bank.recover_inflight()
        snapshot = bank.snapshot(queued_limit=8, recent_limit=4, display_limit=480)
        if recovered:
            snapshot["recovered"] = recovered
        return snapshot
    except Exception as exc:
        return {
            "available": False,
            "path": str(bank.path),
            "reason": f"{type(exc).__name__}: {exc}",
            "queued_count": 0,
            "running_count": 0,
            "queued": [],
            "running": [],
            "recent": [],
            "next": None,
        }


def _goal_bank_has_work(workspace: Path) -> bool:
    """Return whether a queued future goal should wake the resident worker."""
    snapshot = _goal_bank_snapshot(workspace)
    return bool(snapshot.get("available") and int(snapshot.get("queued_count") or 0) > 0)


def resident_behavior(
    state: Mapping[str, Any],
    memory: Mapping[str, Any],
    *,
    mission_has_work: bool,
    inbox_has_work: bool,
    max_restarts: int,
    auto_restart: bool = True,
) -> Dict[str, Any]:
    """Select the resident's next control action without invoking a model.

    This is the behavioral harness: safety and durable evidence outrank
    throughput, and an idle resident waits for real work rather than inventing
    busywork.  The returned decision is persisted by the supervisor so a
    restart can explain the last control choice.
    """
    if state.get("stop_requested"):
        action = "STOP"
        reason = "stop was requested"
    elif state.get("clean_room_requested"):
        action = "WAIT_FOR_CLEAN_ROOM"
        reason = str(state.get("clean_room_reason") or "protected experiment")
    elif memory.get("safe") is False:
        action = "WAIT_FOR_MEMORY"
        reasons = memory.get("reasons") or ["host memory is unsafe"]
        reason = "; ".join(str(item) for item in reasons)
    elif state.get("worker_live"):
        action = "MONITOR_WORKER"
        reason = "owned worker is executing a bounded mission slice"
    elif int(state.get("failure_streak") or 0) > 0:
        streak = int(state.get("failure_streak") or 0)
        if not auto_restart or streak >= max(0, int(max_restarts)):
            action = "ESCALATE_FAILURE"
            reason = "restart limit reached; new evidence is required"
        else:
            action = "RESTART_WORKER"
            reason = "worker failed; retry is bounded by durable restart policy"
    elif mission_has_work or inbox_has_work:
        action = "DISPATCH_WORK"
        reason = "durable unfinished work is available"
    else:
        action = "WAIT_FOR_WORK"
        reason = "mission is idle; no model-free busywork is authorized"
    return {
        "schema": BEHAVIOR_SCHEMA,
        "action": action,
        "reason": reason,
        "model_load_allowed": action in {"DISPATCH_WORK", "RESTART_WORKER"},
        "evidence_required_for_refill": True,
        "unrelated_process_kill_allowed": False,
        "updated_at": _now(),
    }


def _worker_live(state: Mapping[str, Any]) -> bool:
    return _pid_matches(state.get("worker_pid"), state.get("worker_start_token"))


def _supervisor_live(state: Mapping[str, Any]) -> bool:
    return _pid_matches(state.get("supervisor_pid"), state.get("supervisor_start_token"))


def orphan_exit_reason(
    ppid: int,
    *,
    launch_ppid: Optional[int],
    exit_when_orphaned: bool,
) -> Optional[str]:
    """Return why an orphaned supervisor should stop, or None to keep polling.

    Reparenting to pid 1 is the orphan signal, but it is not on its own
    evidence of abandonment: ``start_resident`` deliberately daemonises the
    supervisor, and such a supervisor also reaches pid 1.  The two are told
    apart by ``launch_ppid``, which is 1 exactly when ``DETACHED_ENV`` said the
    detachment was intentional and otherwise the launcher that owned this
    process.  So this fires only for a supervisor that had a real owner which
    has since exited -- the state PID 96732 was found in: driver gone, pid 1,
    cycles 0, still polling every 5 seconds 40 minutes later.

    On by default.  ``exit_when_orphaned=False`` is the escape hatch for a
    supervisor that should outlive its launcher without being daemonised.
    """
    if not exit_when_orphaned:
        return None
    if ppid != 1:
        return None
    if launch_ppid is None or launch_ppid == 1:
        return None
    return f"launcher pid {launch_ppid} exited; supervisor was reparented to pid 1"


def _owned_signal(pid: Any, token: Any, signum: int) -> bool:
    if not _pid_matches(pid, token):
        return False
    try:
        os.kill(int(pid), signum)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _owned_group_signal(pid: Any, token: Any, signum: int) -> bool:
    """Signal an owned worker session, including model-runtime descendants."""
    if not _pid_matches(pid, token):
        return False
    try:
        os.killpg(int(pid), signum)
        return True
    except (OSError, TypeError, ValueError):
        return _owned_signal(pid, token, signum)


def _child_workunit(parent_id: str, value: Mapping[str, Any]) -> WorkUnit:
    uid = str(value.get("id") or "").strip()
    description = str(value.get("description") or "").strip()
    if not uid or not description:
        raise ValueError("child WorkUnit requires id and description")
    if len(description) > MAX_CHILD_DESCRIPTION:
        raise ValueError("child WorkUnit description is too large")
    if uid == parent_id:
        raise ValueError("child WorkUnit cannot have the parent id")
    dependencies = [str(item) for item in (value.get("dependencies") or [])]
    if parent_id not in dependencies:
        dependencies.insert(0, parent_id)
    # A child may be a TYPED TOOL CALL, not only cognition. Dropping these two
    # fields is what kept the self-build loop open at its last link: the model
    # could ask for `filesystem.search` and the request was silently discarded,
    # so the unit fell back to cognition and the resident never touched the tool
    # surface it can see. Validate here rather than at execution: a malformed
    # proposal must be refused where children are admitted, not become a unit
    # that fails later for a reason the model cannot connect to what it asked.
    tool = value.get("tool")
    if tool is not None and not isinstance(tool, str):
        raise ValueError("child WorkUnit tool must be a string")
    tool = (tool or "").strip() or None
    tool_arguments = value.get("tool_arguments")
    if tool_arguments is not None and not isinstance(tool_arguments, Mapping):
        raise ValueError("child WorkUnit tool_arguments must be an object")
    if tool_arguments is not None and tool is None:
        raise ValueError("child WorkUnit tool_arguments without a tool")
    return WorkUnit(
        id=uid,
        role=str(value.get("role") or "research"),
        description=description,
        dependencies=dependencies,
        verifier=(str(value["verifier"]) if value.get("verifier") else None),
        resource_class=str(value.get("resource_class") or "LIGHT_CONTROL"),
        preferred_backend=(
            str(value["preferred_backend"])
            if value.get("preferred_backend")
            else None
        ),
        provider=(str(value["provider"]) if value.get("provider") else None),
        tool=tool,
        tool_arguments=(dict(tool_arguments) if tool_arguments is not None else None),
        # The resident proposed this unit, so the resident authored it. This is
        # the only site where HCLI itself creates work, which makes it the one
        # place the Claude-removal ratio can be recorded truthfully rather than
        # reconstructed afterwards.
        author=AUTHOR_HCLI,
    )


def admit_evidence_children(mission: Any, evidence: Any) -> List[Dict[str, Any]]:
    """Admit bounded child WorkUnits from verified parent evidence only.

    A model may suggest ``child_workunits`` in its output, but the parent must
    already have passed Mission's verifier.  The suggestion is therefore a
    work proposal, never a completion claim.  Every child remains subject to
    its own verifier and scheduler admission.
    """
    if mission is None:
        return []
    scheduler = getattr(mission, "scheduler", None)
    if scheduler is None:
        return []
    events = evidence if isinstance(evidence, list) else [evidence]
    admitted: List[Dict[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping) or event.get("accepted") is not True:
            continue
        parent_id = str(event.get("unit_id") or "").strip()
        if not parent_id:
            continue
        validation = event.get("validation")
        if not isinstance(validation, Mapping) or validation.get("ok") is not True:
            continue
        candidates = event.get("child_workunits")
        if not isinstance(candidates, list):
            continue
        for raw in candidates[:MAX_CHILD_WORKUNITS]:
            if not isinstance(raw, Mapping):
                continue
            try:
                child = _child_workunit(parent_id, raw)
                outcome = scheduler.submit(child)
            except (TypeError, ValueError, RuntimeError) as exc:
                admitted.append({
                    "parent_id": parent_id,
                    "id": raw.get("id"),
                    "status": "REJECTED",
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                continue
            admitted.append({
                "parent_id": parent_id,
                "id": child.id,
                "status": "ADMITTED" if outcome.kind == "inserted" else "IDEMPOTENT",
                "reason": "verified parent evidence",
            })
    return admitted


@dataclass
class ResidentDaemon:
    """Model-neutral resident control facade.

    This class is intentionally usable in tests without constructing a
    Controller or opening model weights.
    """

    workspace: str | os.PathLike[str]
    store: ResidentStore = field(init=False)

    def __post_init__(self) -> None:
        self.workspace = str(Path(self.workspace).expanduser().resolve())
        self.store = ResidentStore(self.workspace)

    def configure(self, config: ResidentConfig) -> Dict[str, Any]:
        if Path(config.workspace).resolve() != Path(self.workspace).resolve():
            raise ValueError("resident config workspace does not match daemon workspace")
        current = self.store.read()
        if _supervisor_live(current):
            # Silently returning the incumbent here is what made `start` a no-op
            # that looked like a launch. The caller decides whether to refuse or
            # to replace; this layer never pretends the config was applied.
            raise ResidentAlreadyRunning(current)
        if _worker_live(current):
            # A crashed supervisor must not leave its owned model worker
            # orphaned while a new supervisor is taking over.
            _owned_group_signal(
                current.get("worker_pid"),
                current.get("worker_start_token"),
                signal.SIGTERM,
            )
            deadline = _now() + config.evacuation_grace_s
            while _worker_live(self.store.read()) and _now() < deadline:
                time.sleep(min(0.25, max(0.01, deadline - _now())))
            current = self.store.read()
            if _worker_live(current):
                _owned_group_signal(
                    current.get("worker_pid"),
                    current.get("worker_start_token"),
                    signal.SIGKILL,
                )
        old_config = current.get("config")
        mission_path = Path(self.workspace) / ".hcli" / "mission" / "state.json"
        if (
            isinstance(old_config, Mapping)
            and str(old_config.get("goal") or "").strip()
            and str(old_config.get("goal") or "").strip() != config.goal
            and mission_path.is_file()
        ):
            raise RuntimeError(
                "a durable mission already exists; keep its goal or archive it explicitly"
            )
        state = {
            "schema": SCHEMA,
            "resident_id": str(current.get("resident_id") or f"resident-{uuid.uuid4()}"),
            "state": "STARTING",
            "config": config.to_dict(),
            "supervisor_pid": None,
            "supervisor_start_token": None,
            "worker_pid": None,
            "worker_start_token": None,
            "mission_id": current.get("mission_id"),
            "generation": int(current.get("generation") or 0),
            "restart_count": int(current.get("restart_count") or 0),
            # `failure_streak` counts consecutive worker failures since the
            # last human intervention, and it is what the restart budget
            # spends. It only ever reset on a worker that exited 0 -- but a
            # resident that reached the limit is FAILED and never spawns a
            # worker again, so the streak could not come back down and
            # `resident start` returned immediately with state=FAILED forever.
            # Issuing `start` IS the intervention, exactly like the
            # `stop_requested` / `clean_room_requested` flags cleared just
            # below. `restart_count` is the lifetime counter and is preserved.
            "failure_streak": 0,
            "cycles": int(current.get("cycles") or 0),
            "last_event": "configured",
            "stop_requested": False,
            "clean_room_requested": False,
            "clean_room_reason": None,
            "inbox_count": len(self.store.read_inbox()),
            "logical_session": None,
        }
        result = self.store.write(state)
        ResidentBodyRegistry(self.workspace).register(
            model=config.model,
            runtime_count=config.runtime_count,
        )
        return result

    def status(self, *, probe: Optional[Callable[[], Mapping[str, Any]]] = None) -> Dict[str, Any]:
        state = self.store.read()
        if not state:
            return {
                "schema": SCHEMA,
                "workspace": self.workspace,
                "state": "ABSENT",
                "state_path": str(self.store.state_path),
                "goal_bank": _goal_bank_snapshot(Path(self.workspace)),
            }
        config = state.get("config") if isinstance(state.get("config"), dict) else {}
        snapshot: Optional[Mapping[str, Any]] = None
        if probe is not None:
            try:
                snapshot = probe()
            except Exception as exc:
                snapshot = {"error": f"{type(exc).__name__}: {exc}"}
        result = dict(state)
        result["workspace"] = self.workspace
        result["state_path"] = str(self.store.state_path)
        result["knowledge_path"] = str(self.store.knowledge_path)
        result["inbox_count"] = len(self.store.read_inbox())
        result["goal_bank"] = _goal_bank_snapshot(Path(self.workspace))
        result["body"] = self.store.body()
        result["supervisor_live"] = _supervisor_live(state)
        result["worker_live"] = _worker_live(state)
        try:
            from hcli.agentos.background import BackgroundJobStore

            result["children"] = [
                job
                for job in BackgroundJobStore(self.workspace).list()
                if job.get("parent_job_id") == state.get("resident_id")
            ]
        except Exception as exc:
            result["children"] = []
            result["children_error"] = f"{type(exc).__name__}: {exc}"
        if snapshot is not None:
            result["memory"] = memory_decision(
                snapshot,
                reserve_bytes=config.get("reserve_bytes"),
                swap_ceiling_bytes=config.get("swap_ceiling_bytes"),
            )
        return result

    def request_stop(self) -> Dict[str, Any]:
        state = self.store.read()
        if not state:
            return {"state": "ABSENT", "stopped": True}
        self.store.update(stop_requested=True, last_event="stop_requested")
        if _supervisor_live(state):
            _owned_signal(
                state.get("supervisor_pid"),
                state.get("supervisor_start_token"),
                signal.SIGTERM,
            )
        return self.status()

    def request_clean_room(self, reason: str = "protected experiment") -> Dict[str, Any]:
        """Pause model loading and request owned-worker evacuation."""
        text = str(reason or "protected experiment").strip()[:400]
        if not text:
            text = "protected experiment"
        state = self.store.read()
        if not state:
            return {"state": "ABSENT", "clean_room_requested": False}
        self.store.update(
            clean_room_requested=True,
            clean_room_reason=text,
            last_event="clean_room_requested",
        )
        supervisor = self.store.read()
        wake = getattr(signal, "SIGUSR1", None)
        if wake is not None and _supervisor_live(supervisor):
            _owned_signal(
                supervisor.get("supervisor_pid"),
                supervisor.get("supervisor_start_token"),
                wake,
            )
        return self.status()

    def resume_clean_room(self) -> Dict[str, Any]:
        """Release a clean-room pause; the supervisor will re-probe first."""
        state = self.store.read()
        if not state:
            return {"state": "ABSENT", "clean_room_requested": False}
        self.store.update(
            clean_room_requested=False,
            state="STARTING",
            last_event="clean_room_resumed",
        )
        supervisor = self.store.read()
        wake = getattr(signal, "SIGUSR1", None)
        if wake is not None and _supervisor_live(supervisor):
            _owned_signal(
                supervisor.get("supervisor_pid"),
                supervisor.get("supervisor_start_token"),
                wake,
            )
        return self.status()

    def launch_child(
        self,
        argv: Sequence[str],
        *,
        cwd: Optional[str | os.PathLike[str]] = None,
        label: Optional[str] = None,
        resumable: bool = True,
        timeout_s: Optional[float] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        """Launch a durable child under the resident's explicit ownership."""
        state = self.store.read()
        if not state:
            raise RuntimeError("resident is not configured")
        config = state.get("config") if isinstance(state.get("config"), dict) else {}
        from hcli.agentos.background import BackgroundJobStore

        jobs = BackgroundJobStore(self.workspace, allowed_roots=(config.get("repo_root") or self.workspace,))
        parent = str(state.get("resident_id") or "resident-parent")
        result = jobs.start(
            list(argv),
            cwd=cwd,
            label=label or "resident-child",
            resumable=resumable,
            timeout_s=timeout_s,
            env=env,
            parent_job_id=parent,
        )
        children = list(state.get("child_job_ids") or [])
        if result.get("job_id") not in children:
            children.append(result.get("job_id"))
        self.store.update(child_job_ids=children[-64:], last_event="child_started")
        return result

    def refill_from_evidence(self, mission: Any, evidence: Any) -> List[Dict[str, Any]]:
        rows = admit_evidence_children(mission, evidence)
        self.store.append_knowledge({"event": "evidence_refill", "children": rows})
        return rows

    def record_knowledge(self, observation: Mapping[str, Any]) -> None:
        self.store.append_knowledge(observation)

    def enqueue_workunit(self, unit: WorkUnit) -> Dict[str, Any]:
        """Queue one bounded unit for the next worker cycle, model-free."""
        if not isinstance(unit, WorkUnit):
            raise TypeError("resident enqueue requires a WorkUnit")
        self.store.append_inbox(unit)
        self.store.update(
            inbox_count=len(self.store.read_inbox()),
            last_event="workunit_queued",
        )
        state = self.store.read()
        if _supervisor_live(state):
            wake = getattr(signal, "SIGUSR1", None)
            if wake is not None:
                _owned_signal(
                    state.get("supervisor_pid"),
                    state.get("supervisor_start_token"),
                    wake,
                )
        return {
            "status": "QUEUED",
            "workunit": unit.to_dict(),
            "inbox_path": str(self.store.inbox_path),
        }

    def bank_goal(self, goal: str, *, mode: str = "auto") -> Dict[str, Any]:
        """Queue a high-level goal and wake an owned supervisor if present."""
        item = GoalBank(self.workspace).add(goal, mode=mode)
        state = self.store.read()
        if state:
            self.store.update(
                goal_bank=_goal_bank_snapshot(Path(self.workspace)),
                last_event="goal_banked",
            )
            wake = getattr(signal, "SIGUSR1", None)
            if wake is not None and _supervisor_live(state):
                _owned_signal(
                    state.get("supervisor_pid"),
                    state.get("supervisor_start_token"),
                    wake,
                )
        return {
            "status": "QUEUED",
            "goal": item,
            "goal_bank_path": str(GoalBank(self.workspace).path),
        }


class ResidentSupervisor:
    """Tiny parent process that can unload and relaunch one model worker."""

    def __init__(self, state_path: str | os.PathLike[str]) -> None:
        self.state_path = Path(state_path).expanduser().resolve()
        self.workspace = self.state_path.parents[2]
        self.daemon = ResidentDaemon(self.workspace)
        self.store = self.daemon.store
        self._stop = False
        self._wake = threading.Event()
        self._worker_process: Optional[subprocess.Popen[Any]] = None

    def _config(self) -> ResidentConfig:
        state = self.store.read()
        raw = state.get("config")
        if not isinstance(raw, Mapping):
            raise ValueError("resident config is missing")
        return ResidentConfig.from_mapping(raw)

    def _memory(self, config: ResidentConfig) -> Dict[str, Any]:
        try:
            from hcli.machine import host_snapshot

            snapshot = host_snapshot()
        except Exception as exc:
            snapshot = {"pressure": "unknown", "probe_error": str(exc)}
        decision = memory_decision(
            snapshot,
            reserve_bytes=config.reserve_bytes,
            swap_ceiling_bytes=config.swap_ceiling_bytes,
        )
        # Host RAM/swap is the cheap first gate. If it passes and a model was
        # explicitly selected, run the existing MemGate in dry mode as the
        # second gate. This prevents the supervisor from declaring a model
        # load allowed when the real runtime would immediately refuse its UMA
        # working set. Do not inspect the runtime gate while a worker is live:
        # its ownership record belongs to that worker's pool and the
        # supervisor must remain read-only with respect to it.
        if not decision.get("safe") or not config.model:
            return decision
        if self._worker_alive(self.store.read()):
            return decision
        try:
            from hcli.backends import is_remote_endpoint
            from hcli.hawking_native import config_for_model_path, is_hawking_native_path
            from hcli.machine import MemGate, resolve_decode_topology

            model_path = str(config.model)
            model_bytes = 0
            if not is_remote_endpoint(model_path):
                candidate = Path(model_path).expanduser()
                if candidate.is_file():
                    if is_hawking_native_path(str(candidate)):
                        native = config_for_model_path(str(candidate))
                        native.validate()
                        inventory = native.identity().get("artifact_inventory", {})
                        model_bytes = int(inventory.get("artifact_bytes") or 0)
                    else:
                        model_bytes = candidate.stat().st_size
            topology, topology_source = resolve_decode_topology(
                config.repo_root or config.workspace
            )
            gate = MemGate(
                reserve_bytes=config.reserve_bytes,
                swap_ceiling_bytes=config.swap_ceiling_bytes,
                model_bytes=model_bytes,
                topology=topology,
            )
            admission = gate.consider(
                admitted=0,
                extra=1,
                snapshot=dict(snapshot),
                refresh_metal=False,
            )
            runtime_gate = {
                "planned": 1 if admission.allow else 0,
                "allow": admission.allow,
                "refusal_reason": None if admission.allow else admission.reason,
                "gate": admission.gate,
                "topology": topology,
                "topology_source": topology_source,
                "record": {
                    "admitted": admission.allow,
                    "reason": admission.reason,
                    "gate": admission.gate,
                    "details": admission.details,
                },
            }
            decision["runtime_gate"] = runtime_gate
            if not runtime_gate["allow"]:
                decision["safe"] = False
                decision.setdefault("reasons", []).append(
                    "runtime admission refused: "
                    + str(runtime_gate.get("refusal_reason") or "unknown reason")
                )
        except Exception as exc:
            # A failed admission probe is not permission to guess that a
            # heavy model is safe. Keep the resident waiting until the probe
            # becomes valid again.
            decision["runtime_gate"] = {
                "planned": 0,
                "allow": False,
                "refusal_reason": f"{type(exc).__name__}: {exc}",
            }
            decision["safe"] = False
            decision.setdefault("reasons", []).append(
                f"runtime admission probe failed: {type(exc).__name__}: {exc}"
            )
        return decision

    def _spawn_worker(self) -> None:
        state = self.store.read()
        config = self._config()
        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = os.pathsep.join(
            [source_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
        proc = subprocess.Popen(
            daemon_argv("--worker", str(self.state_path)),
            cwd=config.workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        self._worker_process = proc
        self.store.update(
            state="RUNNING",
            worker_pid=proc.pid,
            worker_start_token=process_start_token(proc.pid),
            worker_started_at=_now(),
            worker_returncode=None,
            error=None,
            generation=int(state.get("generation") or 0) + 1,
            last_event="worker_started",
        )

    def _evacuate(
        self,
        reason: str,
        config: ResidentConfig,
        *,
        waiting_state: str = "WAITING_FOR_MEMORY",
    ) -> None:
        state = self.store.read()
        if self._worker_alive(state):
            self.store.update(
                state="EVACUATING",
                last_event="evacuation_requested",
                evacuation_reason=reason,
            )
            _owned_signal(state.get("worker_pid"), state.get("worker_start_token"), signal.SIGTERM)
            deadline = _now() + config.evacuation_grace_s
            while _now() < deadline:
                if not self._worker_alive(self.store.read()):
                    break
                time.sleep(min(0.25, max(0.01, deadline - _now())))
            state = self.store.read()
            if self._worker_alive(state):
                # The PID and start token came from our worker launch record;
                # never broaden this to an arbitrary process sweep.
                _owned_group_signal(
                    state.get("worker_pid"),
                    state.get("worker_start_token"),
                    signal.SIGKILL,
                )
        try:
            ResidentBodyRegistry(self.workspace).mark_unloaded(
                reason="supervisor_evacuated"
            )
        except OSError:
            pass
        self.store.update(
            state=waiting_state,
            worker_pid=None,
            worker_start_token=None,
            last_event=waiting_state.lower(),
            evacuation_reason=reason,
        )

    def _worker_alive(self, state: Mapping[str, Any]) -> bool:
        if self._worker_process is not None:
            return self._worker_process.poll() is None
        return _worker_live(state)

    def _worker_finished(self) -> None:
        if self._worker_process is not None:
            code = self._worker_process.poll()
            if code is None:
                return
            self.store.update(worker_returncode=int(code))
            self._worker_process = None
        state = self.store.read()
        if state.get("worker_pid") is not None and _worker_live(state):
            return
        if state.get("worker_pid") is None:
            return
        code = state.get("worker_returncode")
        failure_streak = int(state.get("failure_streak") or 0)
        if code == 0:
            failure_streak = 0
        else:
            failure_streak += 1
        self.store.update(
            worker_pid=None,
            worker_start_token=None,
            worker_live=False,
            failure_streak=failure_streak,
            restart_count=(int(state.get("restart_count") or 0) + (1 if code != 0 else 0)),
            cycles=(int(state.get("cycles") or 0) + 1),
            last_event="worker_finished" if code == 0 else "worker_failed",
        )

    def run(self) -> int:
        state = self.store.read()
        if not state:
            raise RuntimeError("resident state is missing")
        self.store.update(
            state="RUNNING",
            supervisor_pid=os.getpid(),
            supervisor_start_token=process_start_token(os.getpid()),
            # 1 means "no launcher owns me". A deliberately daemonised
            # supervisor says so via the environment instead of racing its
            # launcher's exit for the value of getppid().
            supervisor_launch_ppid=(
                1 if os.environ.get(DETACHED_ENV) else os.getppid()
            ),
            last_event="supervisor_started",
            stop_requested=False,
            stop_reason=None,
        )

        def stop_handler(_signum: int, _frame: Any) -> None:
            self._stop = True

        previous_term = signal.getsignal(signal.SIGTERM)
        previous_int = signal.getsignal(signal.SIGINT)
        usr1 = getattr(signal, "SIGUSR1", None)
        previous_usr1 = signal.getsignal(usr1) if usr1 is not None else None
        signal.signal(signal.SIGTERM, stop_handler)
        signal.signal(signal.SIGINT, stop_handler)
        if usr1 is not None:
            signal.signal(usr1, lambda _signum, _frame: self._wake.set())
        try:
            while not self._stop:
                state = self.store.read()
                if state.get("stop_requested"):
                    break
                config = self._config()
                orphaned = orphan_exit_reason(
                    os.getppid(),
                    launch_ppid=_safe_int(state.get("supervisor_launch_ppid")),
                    exit_when_orphaned=config.exit_when_orphaned,
                )
                if orphaned:
                    # Same shutdown path as stop_requested; the reason survives
                    # the finally block so a later reader knows why it stopped.
                    self.store.update(
                        last_event="supervisor_orphaned",
                        stop_reason=orphaned,
                    )
                    break
                memory = self._memory(config)
                self.store.update(
                    heartbeat_at=_now(),
                    memory=memory,
                    worker_live=_worker_live(state),
                )
                self._worker_finished()
                state = self.store.read()
                worker_live = self._worker_alive(state)
                mission_path = self.workspace / ".hcli" / "mission" / "state.json"
                mission_pending = _mission_has_work(self.workspace)
                inbox_pending = _inbox_has_work(self.workspace)
                bank_snapshot = _goal_bank_snapshot(self.workspace)
                bank_pending = bool(
                    bank_snapshot.get("available")
                    and int(bank_snapshot.get("queued_count") or 0) > 0
                )
                state = self.store.update(
                    worker_live=worker_live,
                    goal_bank=bank_snapshot,
                )
                decision = resident_behavior(
                    state,
                    memory,
                    mission_has_work=mission_pending or bank_pending,
                    inbox_has_work=inbox_pending,
                    max_restarts=config.max_restarts,
                    auto_restart=config.auto_restart,
                )
                self.store.update(behavior=decision)
                if decision["action"] == "WAIT_FOR_CLEAN_ROOM":
                    reason = str(state.get("clean_room_reason") or "protected experiment")
                    if worker_live:
                        self._evacuate(
                            reason,
                            config,
                            waiting_state="WAITING_FOR_CLEAN_ROOM",
                        )
                    else:
                        self.store.update(
                            state="WAITING_FOR_CLEAN_ROOM",
                            last_event="waiting_for_clean_room",
                            evacuation_reason=reason,
                        )
                elif decision["action"] == "WAIT_FOR_MEMORY":
                    self._evacuate(
                        ", ".join(memory["reasons"]),
                        config,
                        waiting_state="WAITING_FOR_MEMORY",
                    )
                else:
                    # A mission that has given up needs a human, and the worker
                    # cannot tell the supervisor so: it recovers the dead
                    # mission, reports the failure, and exits 0, which leaves
                    # failure_streak at 0 and max_restarts untouched. Stop here
                    # with the reason on the state file instead of dispatching
                    # forever. The goal bank is deliberately NOT drained -- a
                    # failed goal stopping promotion is the documented contract.
                    blocked = mission_blocked_reason(self.workspace)
                    if blocked and not worker_live:
                        self.store.update(
                            state="FAILED",
                            last_event="mission_needs_attention",
                            error=blocked,
                            stop_reason=blocked,
                        )
                        self.daemon.record_knowledge({
                            "event": "mission_needs_attention", "reason": blocked,
                        })
                        break
                    if not worker_live:
                        failure_streak = int(state.get("failure_streak") or 0)
                        if failure_streak > 0 and (
                            not config.auto_restart or failure_streak >= config.max_restarts
                        ):
                            self.store.update(
                                state="FAILED",
                                last_event="restart_limit_reached",
                                error="same worker failure repeated; new evidence required",
                            )
                            break
                        if (
                            mission_pending
                            or bank_pending
                            or inbox_pending
                            or not mission_path.is_file()
                        ):
                            self._spawn_worker()
                        else:
                            self.store.update(state="IDLE", last_event="mission_idle")
                self.daemon.record_knowledge({
                    "event": "heartbeat",
                    "state": self.store.read().get("state"),
                    "memory": memory,
                    "worker_live": self._worker_alive(self.store.read()),
                    "behavior": decision,
                })
                self._wake.wait(config.interval_s)
                self._wake.clear()
        finally:
            state = self.store.read()
            if self._worker_alive(state):
                config = self._config()
                _owned_group_signal(
                    state.get("worker_pid"),
                    state.get("worker_start_token"),
                    signal.SIGTERM,
                )
                deadline = _now() + config.evacuation_grace_s
                while _now() < deadline and self._worker_alive(self.store.read()):
                    time.sleep(min(0.25, max(0.01, deadline - _now())))
                state = self.store.read()
                if self._worker_alive(state):
                    _owned_group_signal(
                        state.get("worker_pid"),
                        state.get("worker_start_token"),
                        signal.SIGKILL,
                    )
            try:
                ResidentBodyRegistry(self.workspace).mark_unloaded(
                    reason="supervisor_stopped"
                )
            except OSError:
                pass
            self.store.update(
                state="STOPPED" if not state.get("error") else state.get("state"),
                supervisor_pid=None,
                supervisor_start_token=None,
                worker_pid=None,
                worker_start_token=None,
                worker_live=False,
                last_event="supervisor_stopped" if not state.get("error") else state.get("last_event"),
            )
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)
            if usr1 is not None and previous_usr1 is not None:
                signal.signal(usr1, previous_usr1)
        return 0


def _session_ledger_evidence(agent: Any) -> Optional[Dict[str, Any]]:
    """Record, never ask: the operator is not watching an unattended mission
    slice, so this is `SessionLedger.snapshot()` (plus why it would have
    prompted, had anyone been there to ask) written into durable resident
    state -- the same numbers `hcli resident watch` already surfaces for
    everything else this worker did. This never lands, pushes, or merges;
    `hcli.landing` already owns that path and has a verifier in front of it
    for good reason, so this module never imports it.
    """
    try:
        from ..session_ledger import SessionLedger

        ledger = SessionLedger(agent.workspace, repo_root=agent.repo_root)
        snap = ledger.snapshot()
        prompt, reason = ledger.should_prompt(snapshot=snap)
        return {"would_prompt": prompt, "reason": reason, **snap}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _worker_main(state_path: str) -> int:
    state_file = Path(state_path).expanduser().resolve()
    workspace = state_file.parents[2]
    daemon = ResidentDaemon(workspace)
    state = daemon.store.read()
    raw_config = state.get("config")
    if not isinstance(raw_config, Mapping):
        raise RuntimeError("resident config is missing")
    config = ResidentConfig.from_mapping(raw_config)
    body = ResidentBodyRegistry(workspace)
    agent: Any = None
    sink: Any = None
    evacuating = False
    body_loaded = False
    runtime_policy_env = {
        "HCLI_MEM_RESERVE_BYTES": (
            str(config.reserve_bytes) if config.reserve_bytes is not None else None
        ),
        "HCLI_SWAP_CEILING_GIB": (
            format(config.swap_ceiling_bytes / (1024**3), ".12g")
            if config.swap_ceiling_bytes is not None
            else None
        ),
    }
    old_runtime_policy_env = {
        name: os.environ.get(name) for name in runtime_policy_env
    }
    for name, value in runtime_policy_env.items():
        if value is None:
            continue
        os.environ[name] = value

    def request_evacuation(_signum: int, _frame: Any) -> None:
        nonlocal evacuating
        evacuating = True
        try:
            if agent is not None:
                agent.checkpoint()
                if getattr(agent, "mission", None) is not None:
                    # NOT cancel(). The supervisor sends SIGTERM to free memory
                    # and expects to resume; cancel() put the mission into
                    # BLOCKED_MISSION_PHASES and the daemon never advanced again.
                    agent.mission.evacuate("resident_self_evacuation")
        except Exception as exc:
            daemon.store.update(last_event="evacuation_checkpoint_error", error=str(exc))

    signal.signal(signal.SIGTERM, request_evacuation)
    signal.signal(signal.SIGINT, request_evacuation)
    daemon.store.update(worker_heartbeat_at=_now(), last_event="worker_started")
    heartbeat_stop = threading.Event()

    def heartbeat() -> None:
        period = min(5.0, max(0.25, config.interval_s))
        while not heartbeat_stop.wait(period):
            daemon.store.update(worker_heartbeat_at=_now(), last_event="worker_heartbeat")

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name="hcli-resident-worker-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()

    try:
        from hcli.agentos import AgentOS

        # Constructed here, ahead of AgentOS/body construction, so a worker
        # that fails before the model ever loads still leaves a durable
        # events.jsonl trail instead of silently vanishing.
        sink = EventSink(config.workspace)

        body.mark_loading(pid=os.getpid())
        agent = AgentOS(
            config.workspace,
            model=config.model,
            runtime_count=config.runtime_count,
            repo_root=config.repo_root,
        )

        def on_runtime_ready(event: Any) -> None:
            nonlocal body_loaded
            if getattr(event, "type", None) != "runtime_ready":
                return
            payload = getattr(event, "data", {})
            runtimes = payload.get("runtimes") if isinstance(payload, Mapping) else None
            if not isinstance(runtimes, list) or not runtimes:
                return
            body.mark_loaded(pid=os.getpid())
            body_loaded = True

        def on_any_event(event: Any) -> None:
            # EventSink.write already swallows every failure and counts it
            # in .dropped -- a second try/except here would only hide a real
            # bug in the sink itself, so none is added.
            sink.write(event)

        bus = getattr(getattr(agent, "controller", None), "bus", None)
        subscribe = getattr(bus, "subscribe", None)
        if callable(subscribe):
            subscribe(on_runtime_ready)
            subscribe(on_any_event)
        mission_path = Path(config.workspace) / ".hcli" / "mission" / "state.json"
        if mission_path.is_file():
            agent.recover_mission()
        else:
            agent.start_mission(config.goal)
        daemon.store.update(
            mission_id=getattr(agent.mission, "id", None),
            logical_session={
                "session_id": getattr(agent.mission, "session_id", None),
                "mission_id": getattr(agent.mission, "id", None),
                "generation": daemon.store.read().get("generation"),
                "status": "ACTIVE",
            },
        )

        queued = daemon.store.read_inbox()
        accepted: List[WorkUnit] = []
        rejected: List[Dict[str, Any]] = []
        for raw in queued:
            try:
                unit = WorkUnit.from_dict(raw)
                if unit.status not in {"pending", "ready"}:
                    raise ValueError(f"queued WorkUnit {unit.id!r} is not pending")
                accepted.append(unit)
            except (KeyError, TypeError, ValueError) as exc:
                rejected.append({
                    "status": "REJECTED",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "item": _json_safe(raw),
                })
        if accepted:
            agent.mission.scheduler.replan(accepted)
        if queued:
            daemon.store.clear_inbox(len(queued))
        result = agent.run()
        evidence = result.get("evidence") if isinstance(result, dict) else None
        refill = daemon.refill_from_evidence(agent.mission, evidence)
        agent.checkpoint()
        daemon.store.update(
            mission_id=getattr(agent.mission, "id", None),
            worker_result=_json_safe(result),
            worker_returncode=0,
            worker_heartbeat_at=_now(),
            last_event="worker_evacuated" if evacuating else "worker_completed",
            work_refilled=refill,
            inbox_rejected=rejected,
            session_ledger=_session_ledger_evidence(agent),
            logical_session={
                **(daemon.store.read().get("logical_session") or {}),
                "status": "EVACUATED" if evacuating else "IDLE",
            },
        )
        return 0
    except Exception as exc:
        failed_session = daemon.store.read().get("logical_session") or {}
        daemon.store.update(
            worker_returncode=1,
            worker_heartbeat_at=_now(),
            last_event="worker_failed",
            error=f"{type(exc).__name__}: {exc}",
            logical_session={**failed_session, "status": "FAILED"},
        )
        return 1
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)
        if sink is not None:
            sink.close()
        controller = getattr(agent, "controller", None) if agent is not None else None
        shutdown = getattr(controller, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
        for name, value in old_runtime_policy_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        try:
            unload_reason = (
                "self_evacuation"
                if evacuating
                else "worker_exit" if body_loaded else "worker_exit_without_model_load"
            )
            body.mark_unloaded(
                reason=unload_reason
            )
        except OSError:
            pass


def _watch_read_mission(mission_path: Path) -> Dict[str, Any]:
    try:
        return json.loads(mission_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _watch_header_lines(state: Mapping[str, Any]) -> List[str]:
    """2-3 sticky lines: resident state, heartbeat age, cycles, pids, memory, goal."""
    config = state.get("config") if isinstance(state.get("config"), Mapping) else {}
    body = state.get("body") if isinstance(state.get("body"), Mapping) else {}
    memory = state.get("memory") if isinstance(state.get("memory"), Mapping) else {}
    hb = state.get("heartbeat_at") or 0
    age = f"{_now() - hb:.0f}s" if hb else "-"
    return [
        f"HCLI RESIDENT  {state.get('state', '?')}   heartbeat {age} ago"
        f"   cycles {state.get('cycles', '?')}",
        f"  supervisor {state.get('supervisor_pid')}  worker {state.get('worker_pid')}"
        f"  body {body.get('status')}"
        f"  memory free={(memory.get('free_bytes') or 0) / 1024 ** 3:.1f}GB",
        f"  goal       {str(config.get('goal') or '')[:100]}",
    ]


def _watch_unit_summary(units: Mapping[str, Any]) -> List[str]:
    """Counts plus only the running/failed units by name -- never all of them.

    ponytail: caps the named list at 12 so even a mass-failure run stays a
    few lines; raise WATCH_UNIT_SUMMARY_CAP if triage needs more at once.
    """
    from collections import Counter

    cap = WATCH_UNIT_SUMMARY_CAP
    counts = Counter(str(u.get("status")) for u in units.values() if isinstance(u, Mapping))
    lines = [f"   {dict(counts)}"]
    active = sorted(
        name for name, u in units.items()
        if isinstance(u, Mapping) and str(u.get("status")) in ("running", "failed")
    )
    for name in active[:cap]:
        status = units[name].get("status")
        mark = "*" if status == "running" else "x"
        lines.append(f"   {mark} {name:<22} {status}")
    if len(active) > cap:
        lines.append(f"   ... {len(active) - cap} more running/failed")
    return lines


def _watch_footer_lines(mission: Mapping[str, Any], bank: Mapping[str, Any]) -> List[str]:
    """Sticky footer: mission id/phase/elapsed, compact units, bank depth, key hint."""
    units = mission.get("units") if isinstance(mission.get("units"), dict) else {}
    elapsed = ""
    if mission.get("started_at"):
        elapsed = f"{(_now() - float(mission['started_at'])) / 60:.0f}m"
    lines = [
        f"MISSION {str(mission.get('id') or '-')[:8]}  {mission.get('phase', '-')}"
        f"  {elapsed}  accepted={mission.get('accepted_count', 0)}",
    ]
    lines.extend(_watch_unit_summary(units))
    queued = int(bank.get("queued_count") or 0)
    lines.append(f"BANK queued={queued} running={bank.get('running_count', 0)}")
    for item in (bank.get("queued") or [])[:3]:
        lines.append(f"   . {str(item.get('goal') or '')[:88]}")
    lines.append(
        "  /bank <goal>  /bank mission <goal>  /quit  /help"
        "  (anything else steers)   leave: Ctrl-C"
    )
    return lines


def _watch_bank(root: Path, goal: str, mode: str) -> List[str]:
    """Queue a future goal straight through GoalBank -- safe to append from
    another process (serialized by GoalBank's own flock), no lifecycle call."""
    goal = goal.strip()
    if not goal:
        return ["! usage: /bank <goal>   or   /bank mission <goal>"]
    try:
        item = GoalBank(root).add(goal, mode=mode)
    except Exception as exc:
        return [f"✗ bank failed: {type(exc).__name__}: {exc}"]
    return [f"▣ banked {item['id']}: {goal[:88]}"]


def _watch_steer(root: Path, mission: Mapping[str, Any], text: str) -> List[str]:
    """Queue a plain-text steer onto the live mission's own SteeringQueue file
    (`.hcli/steering/<session_id>.json`) -- the same file a fresh worker cycle
    loads on `recover_mission()`, and the same construction pattern already
    used cross-process elsewhere in this codebase (see hcli/delegate.py's
    `_steering_queue`). No inbox fallback is needed: this channel reaches the
    worker for real."""
    session_id = str(mission.get("session_id") or mission.get("id") or "").strip()
    if not session_id:
        return ["! no active mission to steer -- try /bank <goal>"]
    try:
        SteeringQueue(str(root), session_id).enqueue(text)
    except Exception as exc:
        return [f"✗ steer failed: {type(exc).__name__}: {exc}"]
    return ["✓ steer queued"]


def _watch_handle_line(root: Path, mission: Mapping[str, Any], line: str) -> Tuple[bool, List[str]]:
    """Dispatch one submitted input line. Returns (quit, status_lines).

    "/bank ..." and "/bank mission ..." queue a goal. "/quit" detaches, same
    as Ctrl-C. Anything else -- no verb needed -- is a steer to the running
    mission, per "auto steer unless banked".
    """
    text = line.strip()
    if not text:
        return False, []
    if text == "/quit":
        return True, []
    if text == "/help":
        return False, [
            "/bank <goal>        queue a future goal",
            "/bank mission <g>   queue a persistent mission goal",
            "/steer <text>       steer the running mission (same as no leading /)",
            "/quit               detach (same as Ctrl-C); resident keeps running",
            "/help               this list",
            "(a line with no leading / is sent to the mission as a steer)",
        ]
    if text.startswith("/bank mission "):
        return False, _watch_bank(root, text[len("/bank mission "):], "mission")
    if text.startswith("/bank "):
        return False, _watch_bank(root, text[len("/bank "):], "auto")
    if text == "/steer" or text.startswith("/steer "):
        # The interactive TUI has a real /steer verb (commands.py _cmd_steer);
        # a user reflexively typing it here must not be swallowed by the
        # unknown-command catch-all below -- strip the verb and steer, same
        # as the bare text would.
        payload = text[len("/steer"):].strip()
        if not payload:
            return False, ["! usage: /steer <text>"]
        return False, _watch_steer(root, mission, payload)
    if text.startswith("/"):
        return False, [f"! unknown command {text.split()[0]!r} -- /help lists commands"]
    return False, _watch_steer(root, mission, text)


def _watch_enter_raw(fd: int) -> Any:
    """ICANON+ECHO off, ISIG left alone -- Ctrl-C still raises KeyboardInterrupt
    the normal way; we just stop the driver from line-buffering or echoing so
    the caller can render its own single input line. Returns the previous
    attributes to restore, or None on a platform/fd with no termios."""
    if termios is None:
        return None
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        return None
    new = termios.tcgetattr(fd)
    new[3] = new[3] & ~(termios.ICANON | termios.ECHO)
    termios.tcsetattr(fd, termios.TCSADRAIN, new)
    return old


def _watch_restore_terminal(fd: int, old: Any) -> None:
    if termios is None or old is None:
        return
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except (termios.error, OSError):
        pass


def _watch_read_key(fd: int, timeout: float) -> str:
    try:
        ready, _, _ = select.select([fd], [], [], max(0.0, timeout))
    except (OSError, ValueError):
        return ""
    if not ready:
        return ""
    try:
        return os.read(fd, 1024).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _watch_repaint(prev_line_count: int, new_transcript: List[str], frame: List[str]) -> int:
    """Erase only the previous header/footer block, print any brand-new
    transcript lines once (they scroll into real terminal history and are
    never touched again), then draw a fresh header/footer block. Returns
    that block's line count, to erase next time -- this is the "only the
    changed regions repaint" fallback, not a scroll region."""
    out = sys.stdout
    if prev_line_count:
        out.write(f"\x1b[{prev_line_count}A\x1b[J")
    for line in new_transcript:
        out.write(line + "\n")
    for line in frame:
        out.write(line + "\n")
    out.flush()
    return len(frame)


def watch_resident(workspace: str | os.PathLike[str], interval_s: float = 2.0) -> int:
    """Live read-only view of a running resident. Opens NO model.

    The interactive TUI builds its own Controller, and executing a goal there
    would open a SECOND 11 GB body beside the resident's. This reads only the
    durable state the supervisor and worker already write -- including the
    per-event stream a worker appends to `.hcli/mission/events.jsonl` -- so it
    is safe to run beside a live daemon and safe to detach from at any moment.

    Passive until touched: it only renders until a key is typed. A submitted
    line starting with "/" is a command (see /help); anything else is sent to
    the live mission as a steer. Ctrl-C, or "/quit", detaches -- the daemon
    keeps running. On a non-interactive stdin this degrades to a plain
    render-and-sleep loop with no input.
    """
    daemon = ResidentDaemon(workspace)
    root = Path(workspace).expanduser().resolve()
    mission_path = root / ".hcli" / "mission" / "state.json"
    events_path = root / ".hcli" / "mission" / "events.jsonl"

    try:
        interactive = bool(sys.stdin.isatty())
    except Exception:
        interactive = False

    fd = None
    old_term = None
    if interactive:
        try:
            fd = sys.stdin.fileno()
            old_term = _watch_enter_raw(fd)
        except Exception:
            fd = None
            old_term = None
        interactive = old_term is not None

    offset = 0
    input_buf = ""
    frame_lines = 0
    last_seq: Optional[int] = None

    try:
        while True:
            state = daemon.store.read() or {}
            mission = _watch_read_mission(mission_path)
            bank = _goal_bank_snapshot(root)
            new_events, offset = read_events(events_path, offset=offset, limit=200)
            transcript: List[str] = []
            # events.jsonl rotates to a single ``.1`` generation past
            # max_bytes (EventSink._rotate_if_needed) and read_events only
            # ever tails the live file -- if this watcher falls behind by
            # more than one rotation window, the events in between are gone
            # for good. EventSink.write's own ``seq`` is strictly increasing
            # for the life of one worker, so a jump in it is a reliable,
            # cheap signal that a gap happened, even though the events
            # themselves cannot be recovered from a bounded single-generation
            # rotation.
            # ponytail: signal-only, does not widen retention; raise
            # EventSink's max_bytes or keep more rotated generations if lost
            # history itself needs recovering, not just flagging.
            if new_events:
                first_seq = new_events[0].get("seq")
                if (
                    last_seq is not None
                    and isinstance(first_seq, int)
                    and first_seq > last_seq + 1
                ):
                    gap = first_seq - last_seq - 1
                    transcript.append(
                        f"! {gap} event(s) lost (resident outpaced the watcher; "
                        "events.jsonl rotated past a stale offset)"
                    )
                last_seq_candidate = new_events[-1].get("seq")
                if isinstance(last_seq_candidate, int):
                    last_seq = last_seq_candidate
            for event in new_events:
                transcript.extend(render_event(event))

            if interactive:
                for ch in _watch_read_key(fd, max(0.05, float(interval_s))):
                    if ch == "\x03":
                        raise KeyboardInterrupt
                    if ch in ("\r", "\n"):
                        quit_now, extra = _watch_handle_line(root, mission, input_buf)
                        transcript.extend(extra)
                        input_buf = ""
                        if quit_now:
                            raise KeyboardInterrupt
                    elif ch in ("\x7f", "\x08"):
                        input_buf = input_buf[:-1]
                    elif ch >= " ":
                        input_buf += ch

            frame = _watch_header_lines(state) + [""] + _watch_footer_lines(mission, bank)
            if interactive:
                frame_lines = _watch_repaint(frame_lines, transcript, frame + [f"> {input_buf}"])
                continue

            for line in transcript:
                print(line)
            for line in frame:
                print(line)
            print()
            time.sleep(max(0.5, float(interval_s)))
    except KeyboardInterrupt:
        print("\n[hcli] detached. The resident is still running.")
        return 0
    finally:
        if fd is not None:
            _watch_restore_terminal(fd, old_term)


def retire_incumbent(daemon: "ResidentDaemon", timeout_s: float = 30.0) -> Dict[str, Any]:
    """Stop a live supervisor and archive the mission it owned. Never deletes.

    `replace` is the only supported way to take a workspace over from a live
    resident. The mission state is MOVED to `.hcli/mission-retired/<stamp>/`,
    not removed, because a terminal mission is still the evidence for why the
    previous run ended.
    """
    state = daemon.store.read()
    report: Dict[str, Any] = {
        "stopped_pid": None,
        "archived_mission": None,
        "previous_mission_id": state.get("mission_id"),
    }
    if _supervisor_live(state):
        report["stopped_pid"] = state.get("supervisor_pid")
        daemon.request_stop()
        deadline = _now() + timeout_s
        while _now() < deadline and _supervisor_live(daemon.store.read()):
            time.sleep(0.25)
        state = daemon.store.read()
        if _supervisor_live(state):
            _owned_signal(
                state.get("supervisor_pid"),
                state.get("supervisor_start_token"),
                signal.SIGKILL,
            )
            deadline = _now() + 5.0
            while _now() < deadline and _supervisor_live(daemon.store.read()):
                time.sleep(0.25)

    mission_dir = Path(daemon.workspace) / ".hcli" / "mission"
    if mission_dir.is_dir() and any(mission_dir.iterdir()):
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        retired = Path(daemon.workspace) / ".hcli" / "mission-retired" / stamp
        retired.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(mission_dir), str(retired))
        report["archived_mission"] = str(retired)
        # The DAG is the OTHER HALF of the same generation, and leaving it
        # behind is not a tidiness problem -- it carries `repair_signatures`
        # and `repair_counts`. A fresh mission rebuilt its repair budget from
        # the previous mission's spent one, so the FIRST failure of a unit
        # matched a signature already in the inherited set and was refused a
        # repair at depth 0:
        #
        #   repair_exhausted G001.work depth=0
        #   "repair cycle: failure signature already seen in lineage G001.work"
        #
        # Every unit got exactly one attempt and no repair, across every
        # restart, forever. Observed with 11 units pre-spent in `.hcli/dag.json`
        # before mission 89e411d8 had run a single one of them.
        dag_path = Path(daemon.workspace) / ".hcli" / "dag.json"
        if dag_path.is_file():
            shutil.move(str(dag_path), str(Path(retired) / "dag.json"))
            report["archived_dag"] = str(Path(retired) / "dag.json")
    return report


def start_resident(
    workspace: str | os.PathLike[str],
    *,
    goal: str,
    model: Optional[str] = None,
    repo_root: Optional[str | os.PathLike[str]] = None,
    runtime_count: int = 1,
    interval_s: float = DEFAULT_INTERVAL_S,
    evacuation_grace_s: float = DEFAULT_EVACUATION_GRACE_S,
    max_restarts: int = DEFAULT_MAX_RESTARTS,
    reserve_bytes: Optional[int] = None,
    swap_ceiling_bytes: Optional[int] = None,
    exit_when_orphaned: bool = True,
    replace: bool = False,
) -> Dict[str, Any]:
    daemon = ResidentDaemon(workspace)
    if replace:
        retire_incumbent(daemon)
    config = ResidentConfig(
        workspace=str(Path(workspace).expanduser().resolve()),
        goal=goal,
        model=model,
        repo_root=str(repo_root) if repo_root else None,
        runtime_count=runtime_count,
        interval_s=interval_s,
        evacuation_grace_s=evacuation_grace_s,
        max_restarts=max_restarts,
        reserve_bytes=reserve_bytes,
        swap_ceiling_bytes=swap_ceiling_bytes,
        exit_when_orphaned=exit_when_orphaned,
    )
    # configure() raises ResidentAlreadyRunning if a live supervisor owns this
    # workspace. That propagates to the CLI, which exits non-zero. `start` never
    # returns the incumbent's status dressed up as a launch.
    daemon.configure(config)
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        [source_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    # Tell the supervisor it was detached ON PURPOSE. `start_new_session=True`
    # does NOT reparent -- it calls setsid(); the child keeps this process as
    # its parent until this process exits, which is milliseconds later. So a
    # supervisor that reads os.getppid() at startup races: sometimes it records
    # a real launcher pid, sometimes 1, for the same intentional daemonisation.
    # An explicit flag turns that race into a fact.
    env[DETACHED_ENV] = "1"
    proc = subprocess.Popen(
        daemon_argv("--supervise", str(daemon.store.state_path)),
        cwd=config.workspace,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    daemon.store.update(
        supervisor_pid=proc.pid,
        supervisor_start_token=process_start_token(proc.pid),
        state="STARTING",
        last_event="supervisor_launch_requested",
    )
    return daemon.status()


def _parse_bytes(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip().lower()
    multiplier = 1
    if text.endswith("g") or text.endswith("gib"):
        multiplier = 1024**3
        text = text.rstrip("ibg")
    elif text.endswith("m") or text.endswith("mib"):
        multiplier = 1024**2
        text = text.rstrip("ibm")
    try:
        return max(0, int(float(text) * multiplier))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid byte size: {value!r}") from exc


def _add_goal_arguments(parser: argparse.ArgumentParser) -> None:
    """`--goal` or `--goal-file`, exactly one.

    A sovereign goal is 5 KB of obligation ledger. Requiring it as a shell
    argument meant the documented launch command (`--goal-file civilization/sovereign-goal.txt`)
    did not exist, and the goal's only durable copy was a JSON field inside the
    daemon's own state file -- which is not a thing an operator can edit or
    review before launching.
    """
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--goal", help="the goal text, inline")
    group.add_argument(
        "--goal-file",
        default=None,
        help="read the goal from a file (use - for stdin)",
    )


def _resolved_goal(args: argparse.Namespace) -> str:
    """The goal text, whichever way it was supplied. Empty is refused upstream."""
    path = getattr(args, "goal_file", None)
    if not path:
        return str(args.goal or "")
    if path == "-":
        return sys.stdin.read()
    return Path(path).expanduser().read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hcli resident",
        description="Run HCLI as a durable, memory-aware resident daemon.",
    )
    sub = parser.add_subparsers(dest="command")
    start = sub.add_parser("start", help="start or attach to the resident supervisor")
    start.add_argument("--workspace", default=os.getcwd())
    start.add_argument("--repo-root", default=None)
    _add_goal_arguments(start)
    start.add_argument("--model", default=None)
    start.add_argument("--runtime-count", type=int, default=1)
    start.add_argument("--interval-s", type=float, default=DEFAULT_INTERVAL_S)
    start.add_argument("--evacuation-grace-s", type=float, default=DEFAULT_EVACUATION_GRACE_S)
    start.add_argument("--max-restarts", type=int, default=DEFAULT_MAX_RESTARTS)
    start.add_argument("--reserve", type=_parse_bytes, default=None)
    start.add_argument("--swap-ceiling", type=_parse_bytes, default=None)
    start.add_argument(
        "--keep-running-when-orphaned",
        dest="exit_when_orphaned",
        action="store_false",
        help="keep polling after the launcher exits (default: stop and say why)",
    )
    replace = sub.add_parser(
        "replace",
        help="retire a live resident (archiving its mission) and start a new goal",
    )
    replace.add_argument("--workspace", default=os.getcwd())
    replace.add_argument("--repo-root", default=None)
    _add_goal_arguments(replace)
    replace.add_argument("--model", default=None)
    replace.add_argument("--runtime-count", type=int, default=1)
    replace.add_argument("--interval-s", type=float, default=DEFAULT_INTERVAL_S)
    replace.add_argument("--evacuation-grace-s", type=float, default=DEFAULT_EVACUATION_GRACE_S)
    replace.add_argument("--max-restarts", type=int, default=DEFAULT_MAX_RESTARTS)
    replace.add_argument("--reserve", type=_parse_bytes, default=None)
    replace.add_argument("--swap-ceiling", type=_parse_bytes, default=None)
    replace.add_argument(
        "--keep-running-when-orphaned",
        dest="exit_when_orphaned",
        action="store_false",
    )
    status = sub.add_parser("status", help="show resident state without opening a model")
    status.add_argument("--workspace", default=os.getcwd())
    verdict = sub.add_parser(
        "verdict",
        help="is this a good run? named criteria, read from disk, opens no model",
    )
    verdict.add_argument("--workspace", default=os.getcwd())
    verdict.add_argument(
        "--json", action="store_true", help="emit the full verdict document"
    )
    watch = sub.add_parser(
        "watch", help="live read-only view of the running resident (opens no model)"
    )
    watch.add_argument("--workspace", default=os.getcwd())
    watch.add_argument("--interval-s", type=float, default=2.0)
    stop = sub.add_parser("stop", help="stop the owned supervisor and worker")
    stop.add_argument("--workspace", default=os.getcwd())
    clean = sub.add_parser(
        "clean-room",
        help="evacuate the owned worker and hold before model loading",
    )
    clean.add_argument("--workspace", default=os.getcwd())
    clean.add_argument("--reason", default="protected experiment")
    resume = sub.add_parser(
        "resume",
        help="release a clean-room pause and re-probe memory",
    )
    resume.add_argument("--workspace", default=os.getcwd())
    queue = sub.add_parser(
        "queue",
        help="queue one WorkUnit without loading a model",
    )
    queue.add_argument("--workspace", default=os.getcwd())
    queue.add_argument("--id", required=True)
    queue.add_argument("--role", default="research")
    queue.add_argument("--description", required=True)
    queue.add_argument("--depends-on", action="append", default=[])
    queue.add_argument("--resource-class", default="LIGHT_CONTROL")
    queue.add_argument("--verifier", default=None)
    queue.add_argument("--preferred-backend", default=None)
    queue.add_argument("--provider", default=None)
    bank = sub.add_parser(
        "bank",
        help="queue a high-level goal for automatic resident promotion",
    )
    bank.add_argument("--workspace", default=os.getcwd())
    bank.add_argument("--mode", choices=("auto", "mission"), default="auto")
    bank.add_argument("goal", nargs="+")
    child = sub.add_parser("child", help="launch one durable child under this resident")
    child.add_argument("--workspace", default=os.getcwd())
    child.add_argument("--cwd", default=None)
    child.add_argument("--label", default=None)
    child.add_argument("--timeout-s", type=float, default=None)
    child.add_argument("--non-resumable", action="store_true")
    child.add_argument("argv", nargs=argparse.REMAINDER, help="argv after --")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command in ("start", "replace"):
        try:
            goal = _resolved_goal(args)
        except OSError as exc:
            # An unreadable goal file must say so on one line, not as a
            # traceback that reads like the daemon crashed on launch.
            print(f"cannot read --goal-file: {exc}", file=sys.stderr)
            return 2
        try:
            result = start_resident(
                args.workspace,
                goal=goal,
                model=args.model,
                repo_root=args.repo_root,
                runtime_count=args.runtime_count,
                interval_s=args.interval_s,
                evacuation_grace_s=args.evacuation_grace_s,
                max_restarts=args.max_restarts,
                reserve_bytes=args.reserve,
                swap_ceiling_bytes=args.swap_ceiling,
                exit_when_orphaned=args.exit_when_orphaned,
                replace=(args.command == "replace"),
            )
        except ResidentAlreadyRunning as exc:
            # Non-zero exit: a caller that greps for a pid must not read this
            # as a launch. The requested config was not applied.
            print(str(exc), file=sys.stderr)
            return 3
    elif args.command == "verdict":
        from hcli.cycle_verdict import evaluate, render

        report = evaluate(args.workspace)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render(report))
        # Exit code carries the verdict so a watcher does not have to parse it.
        # UNKNOWN is 2, not 0: unchecked is not passed.
        return {"PASS": 0, "FAIL": 1}.get(report["verdict"], 2)
    elif args.command == "__never__":
        result = start_resident(
            args.workspace,
            goal=args.goal,
            model=args.model,
            repo_root=args.repo_root,
            runtime_count=args.runtime_count,
            interval_s=args.interval_s,
            evacuation_grace_s=args.evacuation_grace_s,
            max_restarts=args.max_restarts,
            reserve_bytes=args.reserve,
            swap_ceiling_bytes=args.swap_ceiling,
            exit_when_orphaned=args.exit_when_orphaned,
        )
    elif args.command == "watch":
        return watch_resident(args.workspace, interval_s=args.interval_s)
    elif args.command == "status":
        result = ResidentDaemon(args.workspace).status()
    elif args.command == "stop":
        result = ResidentDaemon(args.workspace).request_stop()
    elif args.command == "clean-room":
        result = ResidentDaemon(args.workspace).request_clean_room(args.reason)
    elif args.command == "resume":
        result = ResidentDaemon(args.workspace).resume_clean_room()
    elif args.command == "queue":
        result = ResidentDaemon(args.workspace).enqueue_workunit(
            WorkUnit(
                id=args.id,
                role=args.role,
                description=args.description,
                dependencies=list(args.depends_on),
                resource_class=args.resource_class,
                verifier=args.verifier,
                preferred_backend=args.preferred_backend,
                provider=args.provider,
            )
        )
    elif args.command == "bank":
        result = ResidentDaemon(args.workspace).bank_goal(
            " ".join(args.goal),
            mode=args.mode,
        )
    elif args.command == "child":
        argv_value = list(args.argv)
        if argv_value and argv_value[0] == "--":
            argv_value = argv_value[1:]
        if not argv_value:
            raise ValueError("resident child requires argv after --")
        result = ResidentDaemon(args.workspace).launch_child(
            argv_value,
            cwd=args.cwd,
            label=args.label,
            timeout_s=args.timeout_s,
            resumable=not args.non_resumable,
        )
    else:
        build_parser().print_help()
        return 0
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


DAEMON_PROGRAM = "hawkingd"


def daemon_argv(role: str, state_path: str) -> List[str]:
    """argv that launches this daemon, preferring the `hawkingd` executable.

    The process a human sees in `ps` should be named for the daemon, not for
    one model: a supervisor may come to hold several bodies, in parallel or in
    succession, and `python3 -m hcli.agentos.resident` says nothing about
    either. `hawkingd --supervise <state>` is model-neutral and reads like the
    daemon it is, with `hcli` remaining the client that talks to it.

    Falling back to the module form matters: a source checkout that has not
    been `pip install -e .`-ed has no console script, and a daemon that cannot
    start because its own name is missing would be a poor trade for a nicer
    `ps` line. Ownership is decided by pid plus start token
    (`hcli.resources.process_start_token`), never by argv, so an incumbent
    launched under either name stays owned across this change.
    """
    if role not in ("--supervise", "--worker"):
        raise ValueError(f"unknown daemon role {role!r}")
    program = shutil.which(DAEMON_PROGRAM)
    if program is None:
        # A detached supervisor does not necessarily inherit the PATH that
        # installed the script, so look beside the interpreter running us.
        candidate = Path(sys.executable).with_name(DAEMON_PROGRAM)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            program = str(candidate)
    if program is not None:
        return [program, role, str(state_path)]
    return [sys.executable, "-m", "hcli.hawkingd", role, str(state_path)]


def daemon_main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the `hawkingd` console script.

    Routes the two long-lived roles and otherwise defers to the same parser
    `hcli resident` uses, so both names accept the same verbs.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if "--supervise" in args:
        return ResidentSupervisor(args[args.index("--supervise") + 1]).run()
    if "--worker" in args:
        return _worker_main(args[args.index("--worker") + 1])
    return main(args)


if __name__ == "__main__":
    raise SystemExit(daemon_main(sys.argv[1:]))


__all__ = [
    "ResidentConfig",
    "ResidentBodyRegistry",
    "ResidentDaemon",
    "ResidentStore",
    "ResidentSupervisor",
    "SCHEMA",
    "admit_evidence_children",
    "resident_behavior",
    "build_parser",
    "main",
    "memory_decision",
    "orphan_exit_reason",
    "resident_dir",
    "resident_knowledge_path",
    "resident_state_path",
    "start_resident",
]
