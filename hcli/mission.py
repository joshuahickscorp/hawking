"""Persistent mission loop. Owns the process; WorkUnits stay bounded.

The scheduler, DAG store, resource classes, runtime pool and engine are
consumed here. This module is the missing owner of the loop that drives them.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union

from .dag_store import DagStore, atomic_write_json
from .executors import dispatch_workunit
from .goal import (
    GoalCompiler,
    compiled_ir_from_jsonable,
    compiled_ir_to_jsonable,
    compile_worker_context,
)
from .resources import normalize_resource_class, pid_is_alive
from .scheduler import DEFAULT_NO_PROGRESS_THRESHOLD, NO_PROGRESS, Scheduler
from .workunit import IdentityConflict, WorkUnit, identify_ready, transition_status

MISSION_DIRNAME = "mission"
STATE_FILENAME = "state.json"
LOG_FILENAME = "mission.log"
MISSION_VERSION = 1
DEFAULT_HEARTBEAT_S = 30.0
CANCEL_JOIN_TIMEOUT_S = 1.0
POLL_S = 0.1


class MissionCorruptError(ValueError):
    """Raised when mission state exists but is not a valid document."""


class MissionCancelled(Exception):
    """In-flight work saw a cancellation."""


def mission_dir(workspace: Union[str, Path]) -> Path:
    return Path(workspace) / ".hcli" / MISSION_DIRNAME


def mission_state_path(workspace: Union[str, Path]) -> Path:
    return mission_dir(workspace) / STATE_FILENAME


def mission_log_path(workspace: Union[str, Path]) -> Path:
    return mission_dir(workspace) / LOG_FILENAME


def load_state(path: Union[str, Path]) -> Dict[str, Any]:
    dest = Path(path)
    if not dest.is_file():
        raise FileNotFoundError(str(dest))
    try:
        raw = dest.read_text(encoding="utf-8")
    except OSError as exc:
        raise MissionCorruptError(f"mission state unreadable: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MissionCorruptError(
            f"mission state is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise MissionCorruptError("mission state root is not an object")
    return data


def _as_workspace(workspace: Any) -> Path:
    # Path.root is "/" on POSIX — do not treat pathlib.Path as a Workspace.
    from .workspace import Workspace

    if isinstance(workspace, Workspace):
        return Path(os.fspath(workspace.root))
    return Path(os.fspath(workspace))


def _is_grok_unit(wu: WorkUnit) -> bool:
    backend = str(
        getattr(wu, "assigned_backend", None)
        or getattr(wu, "preferred_backend", None)
        or ""
    ).strip().lower()
    if backend == "grok":
        return True
    return str(getattr(wu, "resource_class", "") or "") == "GROK"


def _grok_launch_pid(workspace: Path, task_id: Any) -> Optional[int]:
    if not task_id:
        return None
    try:
        from .grok_bridge import GrokBridge

        return GrokBridge(workspace).launch_pid_for(str(task_id))
    except Exception:
        return None


def _is_live_grok_unit(wu: WorkUnit, workspace: Path) -> bool:
    """True when this unit is a Grok task whose process is still running.

    Adopt and keep waiting in that case. Fail only when the process is
    gone (genuinely interrupted) or the unit is not a Grok task.
    Unobservable liveness on a unit DagStore already left ``running`` is
    kept: failing it would recreate the leak this recovery exists to close.
    """
    if not _is_grok_unit(wu):
        return False
    task_id = str(getattr(wu, "backend_task_id", None) or "").strip()
    if not task_id:
        return False
    try:
        from .grok_bridge import GrokBridge, process_alive

        info = GrokBridge(workspace).status(task_id)
    except Exception:
        return True
    if info.get("process_alive") is True:
        return True
    if info.get("process_alive") is False:
        return False
    state = str(info.get("state") or "").strip().lower()
    if state == "running":
        pid = info.get("launch_pid")
        if pid:
            return process_alive(pid)
        return True
    return False


def _units_by_status(units: Iterable[WorkUnit]) -> Dict[str, int]:
    counts = {
        "pending": 0,
        "ready": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
    }
    for wu in units:
        status = wu.status if wu.status in counts else "pending"
        counts[status] += 1
    return counts


class Mission:
    """Long-lived process. Each WorkUnit is a bounded, independent context."""

    def __init__(
        self,
        workspace: Any,
        *,
        engine: Any = None,
        units: Optional[Union[Dict[str, WorkUnit], Iterable[WorkUnit]]] = None,
        goal: str = "",
        runtime_count: int = 1,
        runtime_pool: Any = None,
        no_progress_threshold: int = DEFAULT_NO_PROGRESS_THRESHOLD,
        fingerprint_fn: Optional[Callable[[], str]] = None,
        heartbeat_s: float = DEFAULT_HEARTBEAT_S,
        quiet: bool = False,
        mission_id: Optional[str] = None,
        session_id: Optional[str] = None,
        scheduler: Optional[Scheduler] = None,
        limits: Any = None,
        repo_root: Optional[Union[str, Path]] = None,
        install_signals: bool = False,
        before_dispatch: Optional[Callable[["Mission"], None]] = None,
        providers: Optional[Dict[str, Any]] = None,
        stop_runtime_pool: bool = True,
        tool_registry: Any = None,
        context_memory: Any = None,
    ) -> None:
        self.workspace = _as_workspace(workspace)
        self.engine = engine
        self.goal = goal or ""
        self.runtime_count = max(1, int(runtime_count))
        self.runtime_pool = runtime_pool
        self.no_progress_threshold = max(1, int(no_progress_threshold))
        self._fingerprint_fn = fingerprint_fn
        self.heartbeat_s = float(heartbeat_s) if heartbeat_s else DEFAULT_HEARTBEAT_S
        self.quiet = bool(quiet)
        self.id = mission_id or str(uuid.uuid4())
        self.retired_dag: Optional[Path] = None
        self.session_id = session_id or self.id
        self.install_signals = bool(install_signals)
        self.before_dispatch = before_dispatch
        self.providers = dict(providers or {})
        # Kept so _run_unit can hand the executor AgentOS's own registry
        # and repo root instead of the executor building its own.
        self.tool_registry = tool_registry
        # A bounded semantic packet may accompany each WorkUnit. It is never
        # the transcript; it lets an overnight worker inherit prior verified
        # facts and operator constraints without replaying the whole session.
        self.context_memory = context_memory if isinstance(context_memory, dict) else None
        self.repo_root = Path(repo_root) if repo_root else None
        # A Mission may be given a pool it owns, or a pool owned by the
        # long-lived Controller/AgentOS facade.  The latter must survive a
        # completed mission so the next durable mission can use the same
        # resident; stopping it here otherwise leaves the controller holding
        # a non-empty-but-dead pool and causes false no-model completions in
        # callers that only inspect a fixed verifier.
        self.stop_runtime_pool = bool(stop_runtime_pool)

        self.phase = "idle"
        self.strategy = "default"
        self.started_at = time.time()
        self.last_checkpoint: float = 0.0
        self.accepted_count = 0
        self.no_progress_warning: Optional[str] = None
        self.cancel_reason: Optional[str] = None
        self.child_pids: set = set()
        self.observed_max_gpu_decode = 0
        self._compiled: Optional[Dict[str, Any]] = None
        self._stop_reason: Optional[str] = None
        self._last_contexts: Dict[str, Dict[str, Any]] = {}
        # Compact, durable evidence is the bridge from one bounded mission
        # slice to the next. It never stores an unbounded model transcript.
        self._evidence: List[Dict[str, Any]] = []
        self._steering = None
        self._signals_installed = False
        self._sigint_count = 0
        self._prev_sigint = None
        self._last_checkpoint_id: Optional[str] = None
        self._last_heartbeat = 0.0
        self._last_observe = 0.0

        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._done: queue.Queue = queue.Queue()
        self._inflight: Dict[str, threading.Thread] = {}
        # Units DagStore adopted as still-running Grok tasks. run() waits on
        # the existing backend_task_id instead of launching a second one.
        self._adopted_unit_ids: set = set()

        if scheduler is not None:
            self.scheduler = scheduler
        else:
            unit_map = self._coerce_units(units)

            def _build_scheduler() -> Scheduler:
                return Scheduler(
                    unit_map,
                    self.runtime_count,
                    workspace=self.workspace,
                    no_progress_threshold=self.no_progress_threshold,
                    repo_root=repo_root,
                    limits=limits,
                )

            try:
                self.scheduler = _build_scheduler()
            except IdentityConflict:
                # A FINISHED mission's graph is still the live dag.json, and
                # GoalCompiler names every mission's units `implement` and
                # `validate`, so the second goal in a workspace collided by id
                # and this constructor raised. That made bank promotion --
                # one new Mission per queued goal -- structurally unable to
                # complete a second goal, and killed the resident worker on
                # `resident start --goal <something new>`. Retire the
                # superseded graph (renamed under .hcli/dag-retired/, never
                # deleted) and build this mission's own. Only reached on a real
                # conflict, so a compatible graph is never disturbed.
                self.retired_dag = DagStore(self.workspace).retire(
                    reason="superseded"
                )
                self.scheduler = _build_scheduler()

        self._maybe_compile()
        self._ensure_steering()

    def _coerce_units(
        self,
        units: Optional[Union[Dict[str, WorkUnit], Iterable[WorkUnit]]],
    ) -> Dict[str, WorkUnit]:
        if units is None:
            self._maybe_compile()
            dag = (self._compiled or {}).get("workunits")
            found = getattr(dag, "units", None)
            if isinstance(found, dict):
                return dict(found)
            return {}
        if isinstance(units, dict):
            return dict(units)
        return {wu.id: wu for wu in units}

    def _maybe_compile(self) -> None:
        if self._compiled is not None:
            return
        text = (self.goal or "").strip()
        if not text:
            return
        try:
            self._compiled = GoalCompiler().compile(text)
        except Exception:
            self._compiled = {
                "invariants": [],
                "acceptance_criteria": [],
                "referenced_files": [],
                "goal_summary": "",
            }

    def _goal_ref_text(self) -> str:
        workspace = getattr(self, "workspace", None)
        if workspace is None:
            return ""
        try:
            return f"{mission_state_path(workspace)}#goal"
        except Exception:
            return ""

    def _ensure_steering(self) -> None:
        try:
            from .steering import SteeringQueue

            self._steering = SteeringQueue(str(self.workspace), self.session_id)
        except Exception:
            self._steering = None

    @property
    def state_path(self) -> Path:
        return mission_state_path(self.workspace)

    @property
    def log_path(self) -> Path:
        return mission_log_path(self.workspace)

    @classmethod
    def from_workspace(cls, workspace: Any, **kwargs: Any) -> "Mission":
        ws = _as_workspace(workspace)
        path = mission_state_path(ws)
        if path.is_file():
            data = load_state(path)
        else:
            data = {}
        runtime_count = int(kwargs.pop("runtime_count", 1) or 1)
        threshold = kwargs.pop(
            "no_progress_threshold",
            data.get("no_progress_threshold", DEFAULT_NO_PROGRESS_THRESHOLD),
        )
        try:
            sched = Scheduler.from_workspace(
                ws,
                runtime_count=runtime_count,
                no_progress_threshold=int(threshold),
            )
        except FileNotFoundError:
            restored = {}
            for uid, payload in (data.get("units") or {}).items():
                if isinstance(payload, dict):
                    restored[str(uid)] = WorkUnit.from_dict(payload)
            sched = Scheduler(
                restored,
                runtime_count,
                workspace=ws,
                no_progress_threshold=int(threshold),
            )
        # DagStore already failed units whose Grok process is gone, and adopted
        # units whose process is still alive. Do not undo that: failing a live
        # adopted unit leaves the expensive child running with nobody tracking
        # it. Anything still `running` that is not a live Grok task is an
        # in-process worker that died with the parent — that is interrupted.
        adopted_ids: List[str] = []
        leftover_failed = False
        for wu in sched.units.values():
            if wu.status != "running":
                continue
            if _is_live_grok_unit(wu, ws):
                adopted_ids.append(wu.id)
                continue
            transition_status(wu, "failed")
            wu.assigned_runtime = None
            leftover_failed = True
        if leftover_failed:
            sched._persist()
        mission = cls(
            ws,
            scheduler=sched,
            units=sched.units,
            mission_id=data.get("id"),
            goal=kwargs.pop("goal", data.get("goal") or ""),
            session_id=kwargs.pop("session_id", data.get("session_id")),
            no_progress_threshold=int(threshold),
            runtime_count=runtime_count,
            **kwargs,
        )
        mission.phase = str(data.get("phase") or "idle")
        mission.strategy = str(data.get("strategy") or "default")
        try:
            mission.started_at = float(data.get("started_at") or mission.started_at)
        except (TypeError, ValueError):
            pass
        try:
            mission.last_checkpoint = float(data.get("last_checkpoint") or 0.0)
        except (TypeError, ValueError):
            mission.last_checkpoint = 0.0
        checkpoint_id = data.get("checkpoint_id")
        mission._last_checkpoint_id = str(checkpoint_id) if checkpoint_id else None
        try:
            mission.accepted_count = int(data.get("accepted_count") or 0)
        except (TypeError, ValueError):
            mission.accepted_count = 0
        warning = data.get("no_progress_warning")
        mission.no_progress_warning = str(warning) if warning else None
        mission.cancel_reason = data.get("cancel_reason")
        for pid in data.get("child_pids") or []:
            try:
                mission.child_pids.add(int(pid))
            except (TypeError, ValueError):
                pass
        if mission.phase == "running":
            mission.phase = "idle"
        ir = compiled_ir_from_jsonable(data.get("compiled"))
        if ir is not None:
            mission._compiled = ir
        else:
            mission._maybe_compile()
        persisted_evidence = data.get("evidence")
        if isinstance(persisted_evidence, list):
            mission._evidence = [
                item for item in persisted_evidence[-64:]
                if isinstance(item, dict)
            ]
        mission._adopted_unit_ids = set(adopted_ids)
        for uid in adopted_ids:
            wu = sched.units.get(uid)
            if wu is None:
                continue
            pid = _grok_launch_pid(ws, getattr(wu, "backend_task_id", None))
            if pid:
                try:
                    mission.child_pids.add(int(pid))
                except (TypeError, ValueError):
                    pass
        return mission

    # A workspace this big cannot be fingerprinted by reading it. The walk
    # below read_bytes()'d EVERY file under the workspace to answer one
    # question -- "did the tree change since the last unit?" -- and
    # `Mission.run()` asks it before the first WorkUnit and again on every
    # heartbeat. On this repo that is tens of gigabytes of model artifacts and
    # activation captures (46,780 files under ONE capture directory), so a
    # mission never reached its first model call: `hcli resident start` sat at
    # body=LOADING with the worker at 70% CPU stat-ing .f32le dumps, and the
    # supervisor eventually evacuated it. git answers the same question from
    # its index in ~0.1s.
    GIT_FINGERPRINT_TIMEOUT_S = 60.0

    def _git_fingerprint(self, root: Path) -> Optional[str]:
        """HEAD plus the size/mtime of every path git reports as changed.

        Sensitive to a second edit of an already-dirty file (which a bare
        `git status` is not) without reading any file's bytes. Returns None
        when this is not a usable git worktree, so the content walk below
        stays the behaviour for a plain directory.
        """
        if not (root / ".git").exists():
            return None
        try:
            status = subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain=v1", "-z"],
                capture_output=True,
                timeout=self.GIT_FINGERPRINT_TIMEOUT_S,
                check=False,
            )
            head = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                capture_output=True,
                timeout=self.GIT_FINGERPRINT_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if status.returncode != 0:
            return None
        digest = hashlib.sha256()
        digest.update(head.stdout.strip() if head.returncode == 0 else b"")
        digest.update(b"\0")
        digest.update(status.stdout)
        for record in status.stdout.split(b"\0"):
            if len(record) < 4:
                continue
            rel = record[3:].decode("utf-8", "replace")
            try:
                info = (root / rel).stat()
            except OSError:
                continue
            digest.update(f"\0{rel}\0{info.st_size}\0{info.st_mtime_ns}".encode("utf-8"))
        return digest.hexdigest()[:20]

    def fingerprint(self) -> str:
        if self._fingerprint_fn is not None:
            return str(self._fingerprint_fn())
        digest = hashlib.sha256()
        root = self.workspace
        skip = {".hcli", ".git", ".haider"}
        if not root.exists():
            return digest.hexdigest()[:20]
        git = self._git_fingerprint(root)
        if git is not None:
            return git
        for dirpath, dirnames, filenames in os.walk(root):
            rel = Path(dirpath).relative_to(root)
            if any(part in skip for part in rel.parts):
                dirnames[:] = []
                continue
            dirnames.sort()
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if not path.is_file():
                    continue
                try:
                    digest.update(str(path.relative_to(root)).encode("utf-8"))
                    digest.update(b"\0")
                    digest.update(path.read_bytes())
                except OSError:
                    continue
        return digest.hexdigest()[:20]

    def status(self) -> Dict[str, Any]:
        units = list(self.scheduler.units.values())
        counts = _units_by_status(units)
        ready_units = identify_ready(self.scheduler.units)
        running_units = [wu for wu in units if wu.status == "running"]
        failed_units = [wu for wu in units if wu.status == "failed"]
        from .agentos.states import mission_state, workunit_state

        decodes = sum(
            1
            for wu in units
            if wu.status == "running"
            and normalize_resource_class(wu.resource_class) == "GPU_DECODE"
        )
        elapsed = max(0.0, time.time() - float(self.started_at or time.time()))
        per_hour = (
            float(self.accepted_count) / (elapsed / 3600.0) if elapsed > 0 else 0.0
        )
        active_runtimes = 0
        pool = self.runtime_pool
        if pool is not None:
            active_runtimes = int(getattr(pool, "admitted_n", 0) or 0)
            if not active_runtimes:
                active_runtimes = sum(
                    1
                    for runtime in getattr(pool, "runtimes", []) or []
                    if getattr(runtime, "active", False)
                )
        return {
            "mission_id": self.id,
            "phase": self.phase,
            "state": mission_state(
                self.phase,
                has_ready=bool(ready_units),
                has_running=bool(running_units),
                has_failed=bool(failed_units),
            ).value,
            "units_by_status": counts,
            "unit_states": {
                wu.id: workunit_state(wu).value
                for wu in units
            },
            "active_runtimes": int(active_runtimes),
            "active_decodes": int(decodes),
            "accepted_units_per_hour": float(per_hour),
            "elapsed_wall": float(elapsed),
            "last_checkpoint": self.last_checkpoint,
            "no_progress_warning": self.no_progress_warning,
        }

    def append_steer(self, text: str, kind: str = "knowledge") -> Any:
        self._ensure_steering()
        if self._steering is None:
            raise RuntimeError("steering is unavailable")
        token = (kind or "knowledge").strip().lower()
        if token not in ("knowledge", "correction", "constraint"):
            token = "knowledge"
        event = self._steering.enqueue(text, kind=token)
        self._log(
            {
                "event": "steer",
                "text": text,
                "id": event.id,
                "kind": token,
            }
        )
        self._term(f"steer: queued kind={token}")
        if token in ("correction", "constraint"):
            self._apply_steer_to_future(event)
        return event

    def _apply_steer_to_future(self, event: Any) -> None:
        """Amend future WorkUnits. Never rewrite completed/verified history."""
        note = f"[steer {getattr(event, 'id', '')} {getattr(event, 'kind', '')}] {getattr(event, 'text', '')}"
        for wu in self.scheduler.units.values():
            if wu.status in ("completed", "running"):
                continue
            extra = f" {note}"
            if extra.strip() not in (wu.description or ""):
                wu.description = (wu.description or "") + extra
        self.scheduler._persist()

    def register_child_pid(self, pid: int) -> None:
        try:
            self.child_pids.add(int(pid))
        except (TypeError, ValueError):
            return
        self.checkpoint()

    def checkpoint(self) -> Path:
        """Write a coherent generation, DAG first, and fail loudly if it cannot.

        Two files hold the same units: the scheduler's dag.json, which
        from_workspace reads FIRST, and mission/state.json, its fallback.
        Checkpointing one and not the other leaves the authoritative store
        stale -- that is how a unit that was `running` with a live
        backend_task_id came back from a restart as `pending` with no task id,
        and therefore why a Grok task got relaunched instead of reconciled.

        Two things an audit of that repair then caught, both fixed here:

        * ORDER. The DAG is written FIRST. If the crash lands between the two
          writes, recovery reads a DAG that is newer than state.json rather
          than older, and a reader that prefers the DAG sees the newer
          generation instead of silently reifying the stale one.
        * FAILURE. A DAG persist that fails is a FAILED checkpoint, not a log
          line. Swallowing it produced exactly the split-brain this method
          exists to prevent.

        Both files carry the same `checkpoint_id`, so a mixed generation is
        detectable rather than merely unlikely.
        """
        path = self.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            inflight = sorted(self._inflight)

        stamp = time.time()
        checkpoint_id = f"{self.id}:{stamp:.6f}"

        # DAG first. A persist failure here aborts the checkpoint, leaving the
        # previous coherent generation in place on BOTH files.
        try:
            self.scheduler._persist({"checkpoint_id": checkpoint_id})
        except Exception as exc:
            self._log(
                {
                    "event": "checkpoint_failed",
                    "mission": self.id,
                    "stage": "dag",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            raise

        payload = {
            "version": MISSION_VERSION,
            "id": self.id,
            "checkpoint_id": checkpoint_id,
            "goal": self.goal,
            "phase": self.phase,
            "strategy": self.strategy,
            "started_at": self.started_at,
            "last_checkpoint": stamp,
            "in_flight": inflight,
            "accepted_count": self.accepted_count,
            "no_progress_warning": self.no_progress_warning,
            "child_pids": sorted(self.child_pids),
            "cancel_reason": self.cancel_reason,
            "session_id": self.session_id,
            "no_progress_threshold": self.no_progress_threshold,
            "units": {
                uid: wu.to_dict() for uid, wu in self.scheduler.units.items()
            },
            "compiled": compiled_ir_to_jsonable(self._compiled),
            "evidence": list(self._evidence[-64:]),
        }
        atomic_write_json(path, payload)
        self.last_checkpoint = stamp
        self._last_checkpoint_id = checkpoint_id
        return path

    def cancel(self, reason: str = "cancelled") -> None:
        self.cancel_reason = reason or "cancelled"
        self._cancel.set()
        engine = self.engine
        if engine is not None:
            stopper = getattr(engine, "cancel", None)
            if callable(stopper):
                try:
                    stopper()
                except Exception:
                    pass
        # Grok launch pids are not in child_pids unless we put them there:
        # the executor records them on the GrokBridge receipt, not here.
        # Kill through the bridge (it knows the pid). _stop_children still
        # runs in _finish as a backstop; calling it here would stop the
        # runtime pool while in-process workers are still joining.
        self._cancel_grok_tasks()
        self._log({"event": "cancel", "reason": self.cancel_reason})

    def _is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def install_signal_handlers(self) -> None:
        if self._signals_installed:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        try:
            self._prev_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._on_sigint)
            self._signals_installed = True
        except Exception:
            return
        try:
            signal.signal(signal.SIGTERM, self._on_sigint)
        except Exception:
            pass

    def _on_sigint(self, signum, frame) -> None:
        self._sigint_count += 1
        if self._sigint_count >= 2:
            os._exit(130)
        self.cancel("SIGINT")

    def main_exit(self) -> int:
        if self.install_signals:
            self.install_signal_handlers()
        self.run()
        if self.phase == "cancelled" or self._cancel.is_set():
            reason = self.cancel_reason or "SIGINT"
            print(f"cancelled: {reason}", file=sys.stderr, flush=True)
            return 130
        if self.phase in ("no_progress", "failed"):
            detail = self.no_progress_warning or self._stop_reason or self.phase
            print(f"{self.phase}: {detail}", file=sys.stderr, flush=True)
            return 1
        return 0

    def run(self) -> Dict[str, Any]:
        if self.engine is None:
            raise RuntimeError("Mission requires an engine")
        if self.install_signals:
            try:
                self.install_signal_handlers()
            except Exception:
                pass
        self.started_at = time.time()
        self.phase = "running"
        self._term("phase: running")
        self._observe()
        self._reattach_adopted()
        self.checkpoint()
        try:
            self._loop()
        except KeyboardInterrupt:
            self.cancel("SIGINT")
        finally:
            self._finish()
        return self._result()

    def _finish(self) -> None:
        if self._cancel.is_set():
            self.phase = "cancelled"
            self._fail_inflight(emit_repair=False)
            self._join_inflight()
            self._stop_children()
            self.checkpoint()
            self._term("phase: cancelled")
            return
        self._join_inflight()
        if self.phase not in ("no_progress", "failed", "cancelled"):
            if self.scheduler.is_done():
                self.phase = "completed"
            elif self._stop_reason:
                self.phase = "failed"
        self._stop_children()
        self.checkpoint()
        self._term(f"phase: {self.phase}")

    def _result(self) -> Dict[str, Any]:
        reason = self._stop_reason or self.cancel_reason
        if self.phase == "no_progress":
            reason = "no_progress"
        failed = [wu.id for wu in self.scheduler.units.values() if wu.status == "failed"]
        state = "INCONCLUSIVE" if failed else ("VERIFIED" if self.phase == "completed" else None)
        return {
            "status": self.phase,
            "state": state,
            "verdict": "ACCEPT" if self.phase == "completed" and not failed else "INCONCLUSIVE" if self.phase == "completed" else "BLOCKED" if self.phase == "failed" else "INCONCLUSIVE",
            "reason": reason,
            "cancelled": self.phase == "cancelled",
            "mission_id": self.id,
            "accepted": self.accepted_count,
            "failed_units": failed,
            "no_progress_warning": self.no_progress_warning,
            "evidence": list(self._evidence[-64:]),
        }

    def _loop(self) -> None:
        idle_spins = 0
        while not self._cancel.is_set() and self._stop_reason is None:
            hook = self.before_dispatch
            if callable(hook):
                try:
                    hook(self)
                except Exception as exc:
                    self._log({"event": "before_dispatch_error", "error": str(exc)})
            self._drain_done(block=False)
            assignments = self.scheduler.dispatch()
            self._note_occupancy()
            for wu, _slot in assignments:
                self._start_unit(wu)
            if assignments:
                self.checkpoint()
                idle_spins = 0

            if self._inflight:
                self._drain_done(block=True)
                idle_spins = 0
                continue

            ready = identify_ready(self.scheduler.units)
            if ready:
                idle_spins += 1
                if idle_spins > 50:
                    self.phase = "failed"
                    self._stop_reason = "blocked"
                    self._term("blocker: ready work could not be admitted")
                    break
                time.sleep(0.01)
                continue

            if self.scheduler.is_done():
                self.phase = "completed"
                break

            self.phase = "failed"
            self._stop_reason = "blocked"
            self._term("blocker: no ready work and mission is not done")
            break

    def _drain_done(self, block: bool) -> None:
        timeout = POLL_S if block else 0.0
        try:
            item = self._done.get(timeout=timeout)
        except queue.Empty:
            if block:
                self._maybe_heartbeat()
            return
        self._integrate(item)
        while True:
            try:
                extra = self._done.get_nowait()
            except queue.Empty:
                break
            self._integrate(extra)
        self.checkpoint()

    def _start_unit(self, wu: WorkUnit) -> None:
        with self._lock:
            if wu.id in self._inflight:
                return
            thread = threading.Thread(
                target=self._worker,
                args=(wu,),
                name=f"hcli-wu-{wu.id}",
                daemon=True,
            )
            self._inflight[wu.id] = thread
        self._log({"event": "dispatch", "id": wu.id, "class": wu.resource_class})
        thread.start()

    def _worker(self, wu: WorkUnit) -> None:
        try:
            if self._cancel.is_set():
                self._done.put({"id": wu.id, "cancelled": True})
                return
            result = self._run_unit(wu)
            self._done.put({"id": wu.id, "result": result})
        except Exception as exc:
            self._done.put({"id": wu.id, "error": exc})

    def _run_unit(self, wu: WorkUnit) -> Dict[str, Any]:
        if wu.id in self._adopted_unit_ids and getattr(wu, "backend_task_id", None):
            return self._resume_adopted_grok(wu)
        context = self._unit_context(wu)
        with self._lock:
            self._last_contexts[wu.id] = context
        if self._cancel.is_set():
            return {"cancelled": True, "validation": {"ok": False, "reason": "cancelled"}}
        raw: Any
        from .executors import (
            WorkUnitExecutor,
            dispatch_workunit,
            select_backend_name,
        )

        backend = select_backend_name(wu)
        wu.assigned_backend = backend
        # The implicit legacy default keeps the strict Engine worker seam.
        # Explicit provider registrations—including the current engine under
        # the generic ``resident`` role—go through WorkUnitExecutor so the
        # provider-neutral adapter records identity and provenance.
        if backend != "qwen" or backend in self.providers:
            executor = WorkUnitExecutor(
                self.workspace,
                engine=self.engine,
                providers=self.providers,
                # AgentOS already owns a registry with the mission's permission
                # set and its tool-receipt path. Hand that one down rather than
                # letting the executor mint a second: two registries can differ
                # on what is permitted, and only AgentOS's persists receipts.
                tool_registry=self.tool_registry,
                repo_root=self.repo_root,
            )
            provider_instance = self.providers.get(backend)
            if provider_instance is not None:
                context["provider_instance"] = provider_instance
            raw = executor.execute(wu, context)
        else:
            # dispatch_workunit binds Engine.execute_workunit when absent and
            # REFUSES to fall through to Engine.execute(prompt). The old
            # fallthrough is what sent the whole root goal to every worker.
            raw = dispatch_workunit(self.engine, wu, context)
        if self._cancel.is_set() or (isinstance(raw, dict) and raw.get("cancelled")):
            return {
                "cancelled": True,
                "raw": raw,
                "validation": {"ok": False, "reason": "cancelled"},
            }
        validation = self._route_validation(wu, raw)
        # Engine telemetry is scoped to one structured execution and is reset
        # for the next WorkUnit.  Carry the completed worker's observation to
        # the Mission log before the next dispatch so a multi-unit mission
        # can report every model call rather than only its final one.
        model_calls = list(getattr(self.engine, "_model_calls", []) or [])
        return {
            "raw": raw,
            "validation": validation,
            "model_calls": model_calls,
            "cancelled": False,
        }

    def _unit_context(self, wu: WorkUnit) -> Dict[str, Any]:
        self._maybe_compile()
        events: List[Any] = []
        if self._steering is not None:
            try:
                events = list(self._steering.all())
            except Exception:
                events = []
        units: Dict[str, WorkUnit] = {}
        sched = getattr(self, "scheduler", None)
        if sched is not None:
            found = getattr(sched, "units", None)
            if isinstance(found, dict):
                units = found
        compiled = self._compiled if isinstance(self._compiled, dict) else {}
        packet = compile_worker_context(
            wu,
            compiled,
            phase=str(getattr(self, "phase", "") or ""),
            units=units,
            steering=events,
            failure_context=wu.failure_context,
            ledger=getattr(self, "ledger", None) or getattr(self, "_ledger", None),
            goal_ref=self._goal_ref_text(),
        )
        return {
            "unit_id": wu.id,
            "role": wu.role,
            "description": wu.description,
            "prompt": packet.prompt,
            "steering": list(packet.steering),
            "failure_context": wu.failure_context,
            "is_cancelled": self._is_cancelled,
            "evidence_paths": list(packet.evidence_paths),
            "phase": packet.phase,
            "invariants": list(packet.invariants),
            "acceptance": list(packet.acceptance),
            "neighborhood": list(packet.neighborhood),
            "compiled": compiled_ir_to_jsonable(compiled),
            "context_memory": self.context_memory,
            "packet": packet,
            "provider": getattr(wu, "provider", None) or getattr(wu, "preferred_backend", None),
        }

    def _route_validation(self, wu: WorkUnit, raw: Any) -> Dict[str, Any]:
        """Acceptance is a deterministic check. Model status is not evidence."""
        if not isinstance(raw, dict):
            return {"ok": False, "reason": "NO_DETERMINISTIC_VALIDATION"}
        if raw.get("cancelled"):
            return {"ok": False, "reason": "cancelled"}
        if raw.get("rolled_back"):
            return {"ok": False, "reason": raw.get("error") or "rolled_back"}
        engine = self.engine
        if hasattr(engine, "validate_workunit"):
            try:
                val = engine.validate_workunit(wu, raw)
            except Exception as exc:
                return {"ok": False, "reason": str(exc)}
            if isinstance(val, dict):
                return val
            return {"ok": bool(val)}
        # The unit's OWN verifier, fixed when the WorkUnit was dispatched, is
        # the authority. Preferring it over `raw["tests"]` matters because
        # `raw` is model output: letting the model nominate the tests that
        # judge its own work means it can name a file it wrote in the very
        # mutation under review.
        verifier = (getattr(wu, "verifier", None) or "").strip()
        if verifier:
            from .executors import WorkUnitExecutor

            cpu = WorkUnitExecutor(self.workspace, engine=self.engine)._run_cpu(wu, {})
            val = cpu.get("validation")
            if isinstance(val, dict):
                val.setdefault("acceptance_source", "workunit_verifier")
                return val
            return {"ok": False, "reason": "NO_DETERMINISTIC_VALIDATION"}

        if isinstance(raw.get("validation"), dict):
            return raw["validation"]

        if hasattr(engine, "_validate"):
            tests = list(raw.get("tests") or [])
            paths = []
            try:
                val = engine._validate(paths, tests)
            except Exception as exc:
                return {"ok": False, "reason": str(exc)}
            if not isinstance(val, dict):
                val = {"ok": bool(val)}
            # Tagged, not silently trusted: this unit had no verifier of its
            # own, so acceptance rests on tests the model chose. That is weaker
            # evidence and must be visible in the receipt rather than
            # indistinguishable from a real gate.
            val.setdefault("acceptance_source", "model_supplied_tests")
            return val
        return {"ok": False, "reason": "NO_DETERMINISTIC_VALIDATION"}

    def _accepted(self, validation: Any, raw: Any) -> bool:
        del raw  # model text, including status: completed, carries zero weight
        if not isinstance(validation, dict):
            return False
        return validation.get("ok") is True

    def _record_evidence(
        self,
        wu: WorkUnit,
        validation: Any,
        raw: Any,
    ) -> None:
        """Persist bounded evidence needed for evidence-derived refill."""
        if not isinstance(raw, Mapping):
            proposed: Any = []
        else:
            proposed = raw.get("child_workunits")
            if proposed is None:
                proposed = raw.get("next_workunits")
        candidates = proposed if isinstance(proposed, list) else []
        allowed = {
            "id",
            "role",
            "description",
            "dependencies",
            "verifier",
            "resource_class",
            "preferred_backend",
            "provider",
        }
        compact_children: List[Dict[str, Any]] = []
        for candidate in candidates[:8]:
            if not isinstance(candidate, Mapping):
                continue
            item = {key: candidate[key] for key in allowed if key in candidate}
            if isinstance(item.get("description"), str):
                item["description"] = item["description"][:4000]
            compact_children.append(item)
        compact_validation = dict(validation) if isinstance(validation, Mapping) else {}
        self._evidence.append(
            {
                "unit_id": wu.id,
                "accepted": compact_validation.get("ok") is True,
                "validation": compact_validation,
                "child_workunits": compact_children,
                "at": time.time(),
            }
        )
        self._evidence = self._evidence[-64:]

    def _integrate(self, item: Dict[str, Any]) -> None:
        uid = item.get("id")
        if not uid:
            return
        with self._lock:
            self._inflight.pop(uid, None)
        wu = self.scheduler.units.get(uid)
        if wu is None:
            return
        if self._cancel.is_set() or item.get("cancelled"):
            self._fail_unit(wu, {"reason": "cancelled"}, emit_repair=False)
            return
        if "error" in item:
            self._fail_unit(wu, {"error": str(item["error"])})
            return
        result = item.get("result") or {}
        observed_model_calls = result.get("model_calls")
        if isinstance(observed_model_calls, list) and observed_model_calls:
            self._log(
                {
                    "event": "model_calls_observed",
                    "id": uid,
                    "calls": observed_model_calls,
                }
            )
        if result.get("cancelled"):
            self._fail_unit(wu, {"reason": "cancelled"}, emit_repair=False)
            return
        validation = result.get("validation")
        self._record_evidence(wu, validation, result.get("raw"))
        if not self._accepted(validation, result.get("raw")):
            context = {
                "validation": validation,
                "error": None,
            }
            raw = result.get("raw")
            if isinstance(raw, dict):
                context["error"] = raw.get("error")
                context["status_claimed"] = raw.get("status")
            self._fail_unit(wu, context)
            return
        fp = self.fingerprint()
        self.accepted_count += 1
        try:
            # Scheduler.complete now refuses to complete a WorkUnit without a
            # passing verifier outcome. `validation` is that outcome — it has
            # already been through `_accepted` above, which is the only path
            # that reaches here. Passing it explicitly keeps the verifier, not
            # the caller, as the thing that authorises completion.
            self.scheduler.complete(uid, fingerprint=fp, verification=validation)
        except NO_PROGRESS as exc:
            self._on_no_progress(exc)
            return
        self._term(f"accepted: {uid}")
        self._log({"event": "accepted", "id": uid, "fingerprint": fp})

    def _fail_unit(
        self,
        wu: WorkUnit,
        context: Optional[Dict[str, Any]],
        emit_repair: bool = True,
    ) -> None:
        if not emit_repair:
            self._fail_without_repair(wu)
            return
        repair = self.scheduler.fail(wu.id, context=context)
        if repair is None and getattr(wu, "repair_exhausted", False):
            # Terminal by policy: the repair budget for this lineage is spent,
            # or the same failure has already been seen in it. Say so plainly
            # rather than emitting another descendant that will fail the same way.
            self._term(f"exhausted: {wu.id} — {wu.repair_reason}")
            self._log(
                {
                    "event": "repair_exhausted",
                    "id": wu.id,
                    "root": wu.repair_root,
                    "depth": wu.repair_depth,
                    "reason": wu.repair_reason,
                }
            )
            return
        self._term(f"blocker: {wu.id} failed")
        self._log(
            {
                "event": "failed",
                "id": wu.id,
                "repair": getattr(repair, "id", None),
            }
        )

    def _fail_without_repair(self, wu: WorkUnit) -> None:
        if wu.status == "running":
            transition_status(wu, "failed")
            wu.assigned_runtime = None
            if normalize_resource_class(wu.resource_class) == "MUTATION":
                try:
                    self.scheduler.mutation_lock.release(wu.id)
                except Exception:
                    pass
            self.scheduler._persist()

    def _fail_inflight(self, emit_repair: bool = False) -> None:
        with self._lock:
            live = list(self._inflight)
        for uid in live:
            wu = self.scheduler.units.get(uid)
            if wu is None:
                continue
            if emit_repair:
                self._fail_unit(wu, {"reason": "cancelled"}, emit_repair=True)
            else:
                self._fail_without_repair(wu)

    def _on_no_progress(self, exc: NO_PROGRESS) -> None:
        self.strategy = "halt_no_progress"
        self.phase = "no_progress"
        self.no_progress_warning = str(exc)
        self._stop_reason = "no_progress"
        self._term(f"no-progress: {exc}")
        self._log(
            {
                "event": "no_progress",
                "fingerprint": exc.fingerprint,
                "count": exc.count,
                "threshold": exc.threshold,
            }
        )
        self.checkpoint()

    def _note_occupancy(self) -> None:
        n = sum(
            1
            for wu in self.scheduler.units.values()
            if wu.status == "running"
            and normalize_resource_class(wu.resource_class) == "GPU_DECODE"
        )
        if n > self.observed_max_gpu_decode:
            self.observed_max_gpu_decode = n

    def _join_inflight(self) -> None:
        with self._lock:
            threads = list(self._inflight.values())
        for thread in threads:
            thread.join(timeout=CANCEL_JOIN_TIMEOUT_S)
        while True:
            try:
                item = self._done.get_nowait()
            except queue.Empty:
                break
            self._integrate(item)

    def _cancel_grok_tasks(self) -> None:
        """Reach Grok processes Mission.cancel otherwise never sees.

        GrokBridge owns the launch pid (receipt). cleanup does not kill.
        """
        try:
            from .grok_bridge import GrokBridge
        except Exception:
            return
        bridge = GrokBridge(self.workspace)
        seen = set()
        for wu in list(self.scheduler.units.values()):
            if wu.status == "completed":
                continue
            tid = getattr(wu, "backend_task_id", None)
            if not tid:
                continue
            tid = str(tid)
            if tid in seen:
                continue
            seen.add(tid)
            try:
                result = bridge.cancel(tid)
            except Exception as exc:
                self._log(
                    {
                        "event": "grok_cancel_error",
                        "task_id": tid,
                        "error": str(exc),
                    }
                )
                continue
            pid = result.get("launch_pid")
            if pid:
                try:
                    self.child_pids.add(int(pid))
                except (TypeError, ValueError):
                    pass
            self._log(
                {
                    "event": "grok_cancel",
                    "task_id": tid,
                    "launch_pid": pid,
                    "process_alive": result.get("process_alive"),
                }
            )

    def _reattach_adopted(self) -> None:
        for uid in list(self._adopted_unit_ids):
            wu = self.scheduler.units.get(uid)
            if wu is None or wu.status != "running":
                continue
            self._start_unit(wu)
            self._log({"event": "adopted", "id": uid, "backend_task_id": wu.backend_task_id})

    def _resume_adopted_grok(self, wu: WorkUnit) -> Dict[str, Any]:
        """Wait on a Grok task that survived parent death. Do not relaunch."""
        from .grok_bridge import GrokBridge, GrokRunError

        task_id = str(getattr(wu, "backend_task_id", "") or "")
        if not task_id:
            return {
                "cancelled": False,
                "validation": {
                    "ok": False,
                    "reason": "adopted grok unit has no backend_task_id",
                },
            }
        bridge = GrokBridge(self.workspace)
        pid = bridge.launch_pid_for(task_id)
        if pid:
            try:
                self.child_pids.add(int(pid))
            except (TypeError, ValueError):
                pass
        timeout = float(os.environ.get("HCLI_GROK_WAIT", "3600"))
        try:
            status = bridge.wait(
                task_id,
                timeout=timeout,
                is_cancelled=self._is_cancelled,
            )
        except GrokRunError as exc:
            return {
                "backend": "grok",
                "backend_task_id": task_id,
                "validation": {"ok": False, "reason": str(exc)},
                "cancelled": False,
            }
        state = str(status.get("state") or "").strip().lower()
        raw = {
            "backend": "grok",
            "backend_task_id": task_id,
            "status": status,
        }
        if self._is_cancelled() or state == "cancelled":
            return {
                "cancelled": True,
                "raw": raw,
                "validation": {"ok": False, "reason": "cancelled"},
            }
        if state in {
            "failed",
            "error",
            "errored",
            "timeout",
            "timed-out",
            "refused",
            "stale-running",
            "unknown",
        }:
            return {
                "raw": raw,
                "validation": {
                    "ok": False,
                    "reason": "GROK_TERMINAL_STATE_NOT_SUCCESSFUL",
                    "grok_state": state,
                },
                "cancelled": False,
            }
        validation = self._route_validation(wu, raw)
        return {"raw": raw, "validation": validation, "cancelled": False}

    def _stop_children(self) -> None:
        pids = set(self.child_pids)
        engine = self.engine
        if engine is not None:
            for pid in getattr(engine, "child_pids", []) or []:
                try:
                    pids.add(int(pid))
                except (TypeError, ValueError):
                    pass
        pool = self.runtime_pool if self.stop_runtime_pool else None
        if pool is not None:
            for runtime in getattr(pool, "runtimes", []) or []:
                pid = getattr(runtime, "pid", None)
                if pid:
                    try:
                        pids.add(int(pid))
                    except (TypeError, ValueError):
                        pass
            stopper = getattr(pool, "stop", None)
            if callable(stopper):
                try:
                    stopper()
                except Exception:
                    pass
        if pids:
            try:
                from .backends import terminate_pid
            except Exception:
                terminate_pid = None
            for pid in pids:
                if terminate_pid is not None:
                    try:
                        terminate_pid(int(pid))
                    except Exception:
                        pass
                if pid_is_alive(int(pid)):
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except OSError:
                        pass
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except OSError:
                        pass
                    try:
                        os.killpg(int(pid), signal.SIGKILL)
                    except OSError:
                        pass
        self.child_pids.clear()

    def _observe(self) -> None:
        fp = self.fingerprint()
        pressure = None
        # host_snapshot can block on memory_pressure; skip in quiet test runs.
        if not self.quiet:
            try:
                from .machine import host_snapshot

                snap = host_snapshot()
                pressure = snap.get("pressure") if isinstance(snap, dict) else None
            except Exception as exc:
                self._log({"event": "observe_machine_error", "error": str(exc)})
        self._log({"event": "observe", "fingerprint": fp, "pressure": pressure})
        self._last_observe = time.time()

    def _maybe_heartbeat(self) -> None:
        now = time.time()
        if now - self._last_heartbeat < self.heartbeat_s:
            return
        self._last_heartbeat = now
        snap = self.status()
        running = 0
        if isinstance(snap.get("units_by_status"), dict):
            running = int(snap["units_by_status"].get("running") or 0)
        self._term(
            f"heartbeat phase={snap['phase']} running={running} "
            f"accepted={self.accepted_count} elapsed={snap['elapsed_wall']:.1f}s"
        )
        if now - self._last_observe >= self.heartbeat_s:
            self._observe()

    def _term(self, msg: str) -> None:
        self._log({"event": "term", "msg": msg})
        if self.quiet:
            return
        print(msg, flush=True)

    def _log(self, payload: Dict[str, Any]) -> None:
        rec = dict(payload)
        rec.setdefault("ts", time.time())
        rec.setdefault("mission_id", self.id)
        for key in ("reasoning", "reasoning_content", "hidden_reasoning", "thinking"):
            rec.pop(key, None)
        path = self.log_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, default=str) + "\n")
        except OSError:
            pass


__all__ = [
    "Mission",
    "MissionCorruptError",
    "MissionCancelled",
    "load_state",
    "mission_dir",
    "mission_state_path",
    "mission_log_path",
]
