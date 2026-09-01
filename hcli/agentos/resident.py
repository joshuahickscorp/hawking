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
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no fcntl
    fcntl = None  # type: ignore[assignment]

from hcli.persist import atomic_write_json
from hcli.resources import process_start_token
from hcli.workunit import WorkUnit


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
            return current
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
            "failure_streak": int(current.get("failure_streak") or 0),
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
            [
                sys.executable,
                "-m",
                "hcli.agentos.resident",
                "--worker",
                str(self.state_path),
            ],
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
                state = self.store.update(worker_live=worker_live)
                decision = resident_behavior(
                    state,
                    memory,
                    mission_has_work=mission_pending,
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
                    agent.mission.cancel("resident_self_evacuation")
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

        bus = getattr(getattr(agent, "controller", None), "bus", None)
        subscribe = getattr(bus, "subscribe", None)
        if callable(subscribe):
            subscribe(on_runtime_ready)
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
) -> Dict[str, Any]:
    daemon = ResidentDaemon(workspace)
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
    existing = daemon.configure(config)
    if _supervisor_live(existing):
        return daemon.status()
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
        [sys.executable, "-m", "hcli.agentos.resident", "--supervise", str(daemon.store.state_path)],
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hcli resident",
        description="Run HCLI as a durable, memory-aware resident daemon.",
    )
    sub = parser.add_subparsers(dest="command")
    start = sub.add_parser("start", help="start or attach to the resident supervisor")
    start.add_argument("--workspace", default=os.getcwd())
    start.add_argument("--repo-root", default=None)
    start.add_argument("--goal", required=True)
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
    status = sub.add_parser("status", help="show resident state without opening a model")
    status.add_argument("--workspace", default=os.getcwd())
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
    if args.command == "start":
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


if __name__ == "__main__":
    if "--supervise" in sys.argv:
        raise SystemExit(ResidentSupervisor(sys.argv[sys.argv.index("--supervise") + 1]).run())
    if "--worker" in sys.argv:
        raise SystemExit(_worker_main(sys.argv[sys.argv.index("--worker") + 1]))
    raise SystemExit(main(sys.argv[1:]))


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
