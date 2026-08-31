"""DETACHED EXECUTION — HCLI owns the child; the generation loop does not wait.

Long work seals a WorkUnit, launches a real detached OS process, records
pid + start-token identity and the expected receipt, then returns immediately.
HCLI — not the model — owns PID, logs, timeout, retry, cancellation and crash
reason. A restarted supervisor re-adopts a still-running child only when pid
AND start-token match. Pid alone is reusable and is never trusted.

Recovered, not forked: hcli/agentos/background.py (start_new_session, persist,
SIGTERM then SIGKILL, INTERRUPTED rather than inventing an exit code);
hcli/resources.py process_start_token / pid_is_alive / MutationLock.holder_is_live;
hcli/agentos/recovery.py (kill + wait + recover); tools/future/repro_science.py
(killed_subprocess, partial_result: partial stdout is not a result);
tools/future/qualification_pipeline.py (refuse to start GPU / lease / cargo work);
hcli/workunit.py RESUME_POLICY=rerun.

    python3 tools/future/detached.py --selftest
    python3 tools/future/detached.py --build
    python3 -m pytest tools/future/test_detached.py -q

Everything emitted is STATIC_ONLY, bench UNKNOWN, gpu_authority false.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from hcli.persist import atomic_write_json
from hcli.resources import pid_is_alive, process_start_token
from hcli.workunit import DEFAULT_RETRY_BUDGET, RESUME_POLICY, WorkUnit
from tools.future._common import git, seal, sha256_file

RECEIPT = "DETACHED_EXECUTION.json"
SCHEMA = "hawking.future.detached.v1"
SUPERVISION_SCHEMA = "hawking.future.detached.supervision.v1"
RECORDED_BY = "tools/future/detached.py"
VERSION = 1

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)

TERMINAL_CLASSES = (
    "cancelled",
    "completed-with-receipt",
    "completed-without-receipt",
    "crashed",
    "timed_out",
    "unknown",
)

GPU_RESOURCE_CLASSES = frozenset({"GPU_DECODE", "GPU_DIRTY_OK", "GPU_EXCLUSIVE"})
IDENTITY_STATES = ("dead", "match", "reused", "unknown")

# Recovered from hcli.agentos.background: credential-shaped argv is not persisted.
_SECRET_ARGUMENT_RE = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|authorization|password|secret|"
    r"private[_-]?key|bearer|(?:hf|gh|github|openai|anthropic)[_-]?token)\s*[:=]"
)

_CARGO_SUB = frozenset({"b", "bench", "build", "c", "check", "clippy", "r", "run", "t", "test"})
_GPU_ARGV_NEEDLES = (
    "cuda",
    "metal",
    "metallib",
    "mpsgraph",
    "nvidia-smi",
    "protected-accelerator-bench",
    "protected_accelerator_benchmark",
)
_LEASE_ARGV_NEEDLES = (
    "acquire_gpu_lease",
    "fcntl.lock_ex",
    "protected-accelerator-bench.lock",
    "singletonlease",
)
_GPU_DASH_C_NEEDLES = (
    "cuda",
    "metal",
    "mlx.core",
    "torch.cuda",
    "mps",
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. No hardware measurement. "
    "Neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE. "
    "A supervision record is identity and lifecycle, not a bench."
)

LAUNCH_WAIT_S = 1.0
TERMINATE_GRACE_S = 1.0
POLL_S = 0.04


class UnsafeCommandError(RuntimeError):
    """Refused before spawn. The SLEEPING record is on disk; no child exists."""

    def __init__(self, reason: str, record: Optional[dict[str, Any]] = None) -> None:
        self.reason = reason
        self.record = record
        super().__init__(reason)


class DetachedError(RuntimeError):
    """Operational failure that is not an unsafe-command refusal."""


# ---------------------------------------------------------------------------
# WorkUnit coercion (local interface; concurrent-wave resident_api is not imported)
# ---------------------------------------------------------------------------


def _as_mapping(workunit: Any) -> dict[str, Any]:
    if workunit is None:
        raise DetachedError("workunit is required")
    if hasattr(workunit, "to_dict") and callable(workunit.to_dict):
        row = workunit.to_dict()
        extra_cmd = getattr(workunit, "command", None)
        extra_receipt = getattr(workunit, "output_receipt_path", None) or getattr(
            workunit, "expected_receipt", None
        )
        extra_timeout = getattr(workunit, "timeout_s", None)
        if extra_cmd is not None and "command" not in row:
            row["command"] = extra_cmd
        if extra_receipt is not None and "output_receipt_path" not in row:
            row["output_receipt_path"] = extra_receipt
        if extra_timeout is not None and "timeout_s" not in row:
            row["timeout_s"] = extra_timeout
        return dict(row)
    if isinstance(workunit, Mapping):
        return dict(workunit)
    raise DetachedError(f"workunit must be a mapping or WorkUnit, got {type(workunit).__name__}")


def _argv_of(row: Mapping[str, Any]) -> list[str]:
    for key in ("command", "argv", "diagnostic_command"):
        raw = row.get(key)
        if isinstance(raw, (list, tuple)) and raw:
            argv = [str(item) for item in raw]
            if all(argv):
                return argv
            raise DetachedError("argv contains an empty token")
    protected = row.get("protected_command")
    if isinstance(protected, (list, tuple)) and protected:
        return [str(item) for item in protected]
    raise DetachedError("workunit has no command/argv to launch")


def _timeout_of(row: Mapping[str, Any]) -> Optional[float]:
    raw = row.get("timeout_s")
    if raw is None:
        raw = row.get("timeout")
    if raw is None:
        budget = row.get("budget")
        if isinstance(budget, Mapping):
            raw = budget.get("wall_clock_s")
    if raw is None:
        return None
    value = float(raw)
    if value != value:  # NaN
        raise DetachedError("timeout_s is NaN")
    return max(0.05, min(7 * 24 * 3600.0, value))


def _expected_receipt_of(row: Mapping[str, Any], workspace: Path, job_id: str) -> str:
    raw = (
        row.get("output_receipt_path")
        or row.get("expected_receipt")
        or row.get("expected_receipt_path")
    )
    if raw:
        path = Path(str(raw))
        if not path.is_absolute():
            path = workspace / path
        return str(path)
    return str(workspace / "results" / f"{job_id}.json")


def _retry_policy_of(row: Mapping[str, Any]) -> dict[str, Any]:
    raw = row.get("retry_policy") or row.get("retry") or {}
    if not isinstance(raw, Mapping):
        raw = {}
    max_attempts = raw.get("max_attempts", DEFAULT_RETRY_BUDGET)
    try:
        max_attempts = int(max_attempts)
    except (TypeError, ValueError):
        max_attempts = DEFAULT_RETRY_BUDGET
    return {
        "policy": str(raw.get("policy") or RESUME_POLICY),
        "max_attempts": max_attempts,
        "never_resume_mid_execution": True,
        "retry_on": list(raw.get("retry_on") or ["crashed"]),
        "never_retry_on": list(
            raw.get("never_retry_on")
            or [
                "cancelled",
                "completed-with-receipt",
                "completed-without-receipt",
                "timed_out",
                "unknown",
            ]
        ),
        "source": "hcli/workunit.py RESUME_POLICY + DEFAULT_RETRY_BUDGET",
    }


def _job_id(row: Mapping[str, Any]) -> str:
    raw = str(row.get("id") or row.get("job_id") or "").strip()
    token = re.sub(r"[^A-Za-z0-9_-]", "-", raw).strip("-") or "job"
    return f"{token[:80]}-{uuid.uuid4().hex[:8]}"


def _cwd_of(row: Mapping[str, Any], workspace: Path) -> Path:
    raw = row.get("cwd") or row.get("workspace")
    if not raw or raw in {"repo-root", "REPO"}:
        return Path(workspace)
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = workspace / path
    path = path.resolve(strict=False)
    if not path.is_dir():
        raise DetachedError(f"cwd is not a directory: {path}")
    return path


# ---------------------------------------------------------------------------
# Safe-command gate. Recovered from qualification_pipeline refuse_* + this lane.
# ---------------------------------------------------------------------------


def refuse_reason(argv: Sequence[str], *, resource_class: str | None = None) -> Optional[str]:
    """Why this argv/resource_class must not be spawned. None means allowed."""
    rc = str(resource_class or "").strip().upper()
    if rc in GPU_RESOURCE_CLASSES:
        return (
            f"resource_class {rc} requires a GPU lease this sidecar does not hold; "
            "unit SLEEPS until hardware qualifies. A synthetic result is forbidden."
        )
    if not argv:
        return "empty argv"
    if any(_SECRET_ARGUMENT_RE.search(item) for item in argv):
        return "credential-shaped command arguments are not persisted; refuse launch"
    prog = Path(str(argv[0])).name.lower()
    joined = " ".join(str(item) for item in argv)
    low = joined.lower()

    if prog == "cargo" or str(argv[0]).rstrip("/").endswith("/cargo"):
        sub = argv[1].lstrip("-").lower() if len(argv) > 1 else "build"
        if sub in _CARGO_SUB or len(argv) == 1:
            return "cargo build/test/run is forbidden in this lane; refuse launch"
        return "cargo is forbidden in this lane; refuse launch"
    if re.search(r"\bcargo\s+(build|test|check|run|bench|clippy)\b", low):
        return "cargo build/test is forbidden in this lane; refuse launch"

    if prog == "flock" or str(argv[0]).rstrip("/").endswith("/flock"):
        return "flock is a lease seizure; refuse launch"
    for needle in _LEASE_ARGV_NEEDLES:
        if needle in low:
            return f"command takes or touches a GPU/bench lease ({needle}); refuse launch"

    if prog in {"xcrun", "metal", "metallib"} and any(
        tok.lower() in {"metal", "metallib", "-c"} for tok in argv[1:]
    ):
        return "Metal compiler invocation is forbidden in this lane; refuse launch"
    for needle in _GPU_ARGV_NEEDLES:
        if needle in low:
            return f"command touches GPU/accelerator ({needle}); refuse launch"

    if prog in {"python", "python3"} or prog.startswith("python"):
        for i, tok in enumerate(argv):
            if tok in {"-c", "--command"} and i + 1 < len(argv):
                body = str(argv[i + 1]).lower()
                for needle in _GPU_DASH_C_NEEDLES:
                    if needle in body:
                        return f"python -c body touches GPU ({needle}); refuse launch"
    return None


# ---------------------------------------------------------------------------
# Identity. Recovered from hcli.resources.process_start_token: pid is not identity.
# ---------------------------------------------------------------------------


def identity_status(record: Mapping[str, Any]) -> str:
    """match | dead | reused | unknown. unknown is a real outcome; never guess."""
    try:
        pid = int(record.get("pid"))
    except (TypeError, ValueError):
        return "unknown"
    if pid <= 0:
        return "unknown"
    recorded = record.get("start_token")
    if not pid_is_alive(pid):
        return "dead"
    live = process_start_token(pid)
    if not recorded or not live:
        return "unknown"
    if str(recorded) != str(live):
        return "reused"
    return "match"


def identity_proven(record: Mapping[str, Any]) -> bool:
    return identity_status(record) == "match"


def _crash_reason(returncode: Optional[int]) -> Optional[str]:
    if returncode is None or returncode == 0:
        return None
    if returncode < 0:
        try:
            return f"signal:{signal.Signals(-returncode).name}"
        except (ValueError, SystemError):
            return f"signal:{-returncode}"
    if returncode >= 128:
        sig = returncode - 128
        try:
            return f"signal:{signal.Signals(sig).name}+128"
        except (ValueError, SystemError):
            return f"exit:{returncode}"
    return f"exit:{returncode}"


def classify_terminal(
    record: Mapping[str, Any],
    *,
    returncode: Optional[int],
    timed_out: bool = False,
    cancelled: bool = False,
) -> str:
    """Six classes, no extras. unknown is returned rather than guessed."""
    if cancelled:
        return "cancelled"
    if timed_out:
        return "timed_out"
    if returncode is None:
        return "unknown"
    if int(returncode) != 0:
        return "crashed"
    expected = record.get("expected_receipt_path")
    if expected and Path(str(expected)).is_file():
        return "completed-with-receipt"
    return "completed-without-receipt"


def _terminate_session(pid: Optional[int]) -> None:
    """SIGTERM the session, then SIGKILL. Recovered from background.py _terminate."""
    if not isinstance(pid, int) or pid <= 0:
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + TERMINATE_GRACE_S
    while time.monotonic() < deadline and pid_is_alive(pid):
        time.sleep(0.02)
    if pid_is_alive(pid):
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass


def _reap_pid(pid: Optional[int], timeout_s: float = 2.0) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return True
    deadline = time.monotonic() + max(0.05, timeout_s)
    while time.monotonic() < deadline:
        if not pid_is_alive(pid):
            return True
        time.sleep(0.02)
    if pid_is_alive(pid):
        _terminate_session(pid)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if not pid_is_alive(pid):
            return True
        time.sleep(0.02)
    return not pid_is_alive(pid)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


def _empty_resource_lease() -> dict[str, Any]:
    return {
        "held": False,
        "kind": None,
        "refused": True,
        "reason": (
            "this lane never takes a GPU or bench lease; flock of the protected "
            "lock would be a seizure"
        ),
    }


class DetachedSupervisor:
    """Persist and supervise shell-free detached work under a workspace."""

    def __init__(self, workspace: str | os.PathLike[str]) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = self.workspace / "detached"
        self.jobs_root = self.root / "jobs"
        self.logs_root = self.root / "logs"
        self.units_root = self.root / "units"
        self.results_root = self.root / "results"
        for path in (self.jobs_root, self.logs_root, self.units_root, self.results_root):
            path.mkdir(parents=True, exist_ok=True)

    def _path(self, job_id: str) -> Path:
        token = str(job_id or "").strip()
        if not token or any(
            char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for char in token
        ):
            raise DetachedError(f"invalid job id {job_id!r}")
        return self.jobs_root / f"{token}.json"

    def _write(self, record: Mapping[str, Any]) -> None:
        job_id = str(record["job_id"])
        atomic_write_json(self._path(job_id), dict(record))

    def _read(self, job_id: str) -> dict[str, Any]:
        path = self._path(job_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DetachedError(f"supervision record unreadable: {path}") from exc
        if not isinstance(value, dict):
            raise DetachedError(f"supervision record is not an object: {path}")
        return value

    def _try_read(self, job_id: str) -> Optional[dict[str, Any]]:
        try:
            return self._read(job_id)
        except (FileNotFoundError, DetachedError):
            return None

    def launch(self, workunit: Any, *, env: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
        """Seal, spawn a real detached process, persist identity, return immediately."""
        row = _as_mapping(workunit)
        argv = _argv_of(row)
        resource_class = str(row.get("resource_class") or "LIGHT_CONTROL")
        job_id = _job_id(row)
        workdir = _cwd_of(row, self.workspace)
        timeout_s = _timeout_of(row)
        expected = _expected_receipt_of(row, self.workspace, job_id)
        retry = _retry_policy_of(row)
        stdout_log = self.logs_root / f"{job_id}.stdout.log"
        stderr_log = self.logs_root / f"{job_id}.stderr.log"
        unit_path = self.units_root / f"{job_id}.workunit.json"

        sealed_unit = seal({k: v for k, v in row.items() if k != "seal_sha256"})
        atomic_write_json(unit_path, sealed_unit)

        reason = refuse_reason(argv, resource_class=resource_class)
        now = time.time()
        record: dict[str, Any] = {
            "schema": SUPERVISION_SCHEMA,
            "job_id": job_id,
            "workunit_id": row.get("id"),
            "workunit_path": str(unit_path),
            "workunit_seal_sha256": sealed_unit.get("seal_sha256"),
            "argv": list(argv),
            "cwd": str(workdir),
            "state": "STARTING",
            "terminal": None,
            "pid": None,
            "start_token": None,
            "pgid": None,
            "supervisor_pid": None,
            "supervisor_start_token": None,
            "started_at": now,
            "running_at": None,
            "finished_at": None,
            "timeout_s": timeout_s,
            "retry_policy": retry,
            "cancel_requested": False,
            "resource_lease": _empty_resource_lease(),
            "log_paths": {"stdout": str(stdout_log), "stderr": str(stderr_log)},
            "expected_receipt_path": expected,
            "result_location": expected,
            "returncode": None,
            "crash_reason": None,
            "identity_status": "unknown",
            "launch_refused": False,
            "resource_class": resource_class,
            "attempts": int(row.get("attempts") or 0),
        }
        if reason:
            record["state"] = "SLEEPING"
            record["launch_refused"] = True
            record["crash_reason"] = reason
            record["finished_at"] = now
            record["workunit_status"] = "SLEEPING"
            record["blocked_reason"] = reason
            self._write(record)
            raise UnsafeCommandError(reason, record=dict(record))

        Path(expected).parent.mkdir(parents=True, exist_ok=True)
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        self._write(record)

        supervisor_env = dict(os.environ)
        if env is not None:
            if not isinstance(env, Mapping) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in env.items()
            ):
                raise DetachedError("env must be a string mapping")
            supervisor_env.update(env)
        source_root = str(Path(__file__).resolve().parents[2])
        pythonpath = supervisor_env.get("PYTHONPATH", "")
        supervisor_env["PYTHONPATH"] = os.pathsep.join(
            [source_root] + ([pythonpath] if pythonpath else [])
        )
        try:
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--supervise", str(self._path(job_id))],
                cwd=str(self.workspace),
                env=supervisor_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
        except Exception as exc:
            record["state"] = "TERMINAL"
            record["terminal"] = "crashed"
            record["crash_reason"] = f"supervisor could not start: {type(exc).__name__}: {exc}"
            record["finished_at"] = time.time()
            self._write(record)
            raise

        record["supervisor_pid"] = child.pid
        record["supervisor_start_token"] = process_start_token(child.pid)
        record["state"] = "RUNNING"
        self._write(record)

        deadline = time.monotonic() + LAUNCH_WAIT_S
        while time.monotonic() < deadline:
            live = self._try_read(job_id) or record
            if live.get("pid") or live.get("terminal") or live.get("state") == "SLEEPING":
                live["identity_status"] = identity_status(live)
                return live
            if child.poll() is not None and not (self._try_read(job_id) or {}).get("pid"):
                break
            time.sleep(0.01)

        live = self._try_read(job_id) or record
        if live.get("terminal") is None and child.poll() is not None and not live.get("pid"):
            live["state"] = "TERMINAL"
            live["terminal"] = "crashed"
            live["crash_reason"] = live.get("crash_reason") or "supervisor exited before work pid"
            live["finished_at"] = time.time()
            self._write(live)
        live["identity_status"] = identity_status(live)
        return live

    def inspect(self, job_id: str) -> dict[str, Any]:
        record = self._read(job_id)
        return self._refresh(record)

    def list(self) -> list[dict[str, Any]]:
        if not self.jobs_root.is_dir():
            return []
        out: list[dict[str, Any]] = []
        for path in sorted(self.jobs_root.glob("*.json")):
            try:
                out.append(self.inspect(path.stem))
            except (OSError, DetachedError, ValueError):
                continue
        return out

    def adopt(self, job_id: str) -> dict[str, Any]:
        """Re-bind a persisted child after supervisor restart. Pid alone is not enough."""
        record = self._read(job_id)
        if record.get("launch_refused"):
            record["identity_status"] = "unknown"
            return record
        if record.get("terminal"):
            record["identity_status"] = identity_status(record)
            return record
        status = identity_status(record)
        record["identity_status"] = status
        if status == "match":
            record["state"] = "RUNNING"
            record["adopted"] = True
            self._write(record)
            return self._refresh(record)
        if status == "reused":
            record["state"] = "TERMINAL"
            record["terminal"] = "unknown"
            record["crash_reason"] = (
                "pid is alive but start_token does not match; the live process "
                "is not the original child. unknown, not crashed. not signalled."
            )
            record["finished_at"] = record.get("finished_at") or time.time()
            self._write(record)
            return record
        if status == "dead":
            if record.get("returncode") is None and not record.get("terminal"):
                record["state"] = "TERMINAL"
                record["terminal"] = "unknown"
                record["crash_reason"] = (
                    "child pid is dead and no exit code was persisted; "
                    "unknown, not guessed as crashed or completed"
                )
                record["finished_at"] = record.get("finished_at") or time.time()
                self._write(record)
            return record
        # unknown identity while pid may still be alive: do not kill, do not claim match
        record["adopted"] = False
        record["crash_reason"] = record.get("crash_reason") or (
            "cannot prove pid+start_token identity; will not signal this pid"
        )
        self._write(record)
        return record

    def cancel(self, job_id: str) -> dict[str, Any]:
        record = self._read(job_id)
        if record.get("terminal"):
            record["identity_status"] = identity_status(record)
            return record
        if record.get("launch_refused") or record.get("state") == "SLEEPING":
            record["state"] = "SLEEPING"
            record["identity_status"] = "unknown"
            return record
        record["cancel_requested"] = True
        self._write(record)
        status = identity_status(record)
        record["identity_status"] = status
        if status == "match":
            _terminate_session(record.get("pid"))
            _reap_pid(record.get("pid"), timeout_s=2.0)
        elif status == "reused":
            record["state"] = "TERMINAL"
            record["terminal"] = "unknown"
            record["crash_reason"] = (
                "cancel refused: pid reused by another process; original child "
                "was not signalled"
            )
            record["finished_at"] = time.time()
            self._write(record)
            return record
        elif status == "unknown" and pid_is_alive(int(record["pid"]) if record.get("pid") else 0):
            record["crash_reason"] = (
                "cancel refused: identity unproven (missing start_token); "
                "will not signal a pid we cannot prove is ours"
            )
            self._write(record)
            return record
        refreshed = self._wait_terminal(job_id, 3.0)
        if refreshed.get("terminal"):
            return refreshed
        # Supervisor did not persist. If the child is gone, we classified the kill.
        if identity_status(refreshed) == "dead" or not pid_is_alive(int(refreshed.get("pid") or 0)):
            refreshed["state"] = "TERMINAL"
            refreshed["terminal"] = "cancelled"
            refreshed["crash_reason"] = refreshed.get("crash_reason") or "cancelled"
            refreshed["finished_at"] = time.time()
            if refreshed.get("returncode") is None:
                refreshed["returncode"] = -signal.SIGKILL
            self._write(refreshed)
        return refreshed

    def _timeout_due(self, record: Mapping[str, Any]) -> bool:
        timeout_s = record.get("timeout_s")
        started = record.get("started_at")
        if timeout_s is None or started is None:
            return False
        try:
            return (time.time() - float(started)) > float(timeout_s)
        except (TypeError, ValueError):
            return False

    def _refresh(self, record: dict[str, Any]) -> dict[str, Any]:
        if record.get("terminal") or record.get("launch_refused") or record.get("state") == "SLEEPING":
            record["identity_status"] = identity_status(record)
            return record
        status = identity_status(record)
        record["identity_status"] = status
        if record.get("cancel_requested") and status == "match":
            _terminate_session(record.get("pid"))
            _reap_pid(record.get("pid"), timeout_s=1.5)
            record = self._try_read(record["job_id"]) or record
            if not record.get("terminal"):
                record["state"] = "TERMINAL"
                record["terminal"] = "cancelled"
                record["crash_reason"] = "cancelled"
                record["finished_at"] = time.time()
                record["returncode"] = record.get("returncode")
                if record["returncode"] is None:
                    record["returncode"] = -signal.SIGTERM
                self._write(record)
            record["identity_status"] = identity_status(record)
            return record
        if self._timeout_due(record) and status == "match":
            _terminate_session(record.get("pid"))
            _reap_pid(record.get("pid"), timeout_s=1.5)
            record = self._try_read(record["job_id"]) or record
            if not record.get("terminal"):
                record["state"] = "TERMINAL"
                record["terminal"] = "timed_out"
                record["crash_reason"] = "timeout"
                record["finished_at"] = time.time()
                if record.get("returncode") is None:
                    record["returncode"] = -signal.SIGKILL
                self._write(record)
            record["identity_status"] = identity_status(record)
            return record
        if status == "reused" and not record.get("terminal"):
            record["state"] = "TERMINAL"
            record["terminal"] = "unknown"
            record["crash_reason"] = (
                "pid reused; live process is not the original child"
            )
            record["finished_at"] = time.time()
            self._write(record)
            return record
        if status == "dead" and not record.get("terminal"):
            # Supervisor may still be flushing. Brief wait, then unknown.
            time.sleep(0.08)
            again = self._try_read(record["job_id"]) or record
            if again.get("terminal"):
                again["identity_status"] = identity_status(again)
                return again
            again["state"] = "TERMINAL"
            again["terminal"] = "unknown"
            again["crash_reason"] = again.get("crash_reason") or (
                "child is dead; exit code was not persisted; unknown"
            )
            again["finished_at"] = time.time()
            self._write(again)
            again["identity_status"] = identity_status(again)
            return again
        return record

    def _wait_terminal(self, job_id: str, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.05, timeout_s)
        last = self._try_read(job_id) or {"job_id": job_id}
        while time.monotonic() < deadline:
            last = self._try_read(job_id) or last
            if last.get("terminal"):
                last["identity_status"] = identity_status(last)
                return last
            time.sleep(POLL_S)
        return self._refresh(self._read(job_id))

    def wait_terminal(self, job_id: str, timeout_s: float = 8.0) -> dict[str, Any]:
        return self._wait_terminal(job_id, timeout_s)

    def reap_all(self) -> None:
        for record in self.list():
            job_id = record.get("job_id")
            if not job_id:
                continue
            try:
                if not record.get("terminal") and not record.get("launch_refused"):
                    self.cancel(job_id)
            except (OSError, DetachedError):
                pid = record.get("pid")
                if isinstance(pid, int):
                    _terminate_session(pid)


def launch(workunit: Any, *, workspace: str | os.PathLike[str], env: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """Public entry: seal, detach, persist, return while the child is still running."""
    return DetachedSupervisor(workspace).launch(workunit, env=env)


def inspect(job_id: str, *, workspace: str | os.PathLike[str]) -> dict[str, Any]:
    return DetachedSupervisor(workspace).inspect(job_id)


def cancel(job_id: str, *, workspace: str | os.PathLike[str]) -> dict[str, Any]:
    return DetachedSupervisor(workspace).cancel(job_id)


def adopt(job_id: str, *, workspace: str | os.PathLike[str]) -> dict[str, Any]:
    return DetachedSupervisor(workspace).adopt(job_id)


# ---------------------------------------------------------------------------
# Internal supervisor process (owns wait / timeout / crash reason on disk)
# ---------------------------------------------------------------------------


def _wait_record(path: Path, timeout_s: float = 5.0) -> Optional[dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict) and value.get("argv"):
                return value
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        time.sleep(0.02)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def supervise(record_path: str) -> int:
    """Detached supervisor entry. The command is never shell-parsed."""
    path = Path(record_path).expanduser().resolve()
    record = _wait_record(path)
    if record is None:
        return 70
    argv = record.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) and item for item in argv):
        record["terminal"] = "crashed"
        record["crash_reason"] = "supervision record argv is invalid"
        record["state"] = "TERMINAL"
        record["finished_at"] = time.time()
        atomic_write_json(path, record)
        return 70
    reason = refuse_reason(argv, resource_class=str(record.get("resource_class") or ""))
    if reason:
        record["state"] = "SLEEPING"
        record["launch_refused"] = True
        record["crash_reason"] = reason
        record["finished_at"] = time.time()
        atomic_write_json(path, record)
        return 70
    cwd = Path(str(record.get("cwd") or path.parent))
    logs = record.get("log_paths") or {}
    stdout_path = Path(str(logs.get("stdout") or (path.parent.parent / "logs" / f"{record.get('job_id')}.stdout.log")))
    stderr_path = Path(str(logs.get("stderr") or (path.parent.parent / "logs" / f"{record.get('job_id')}.stderr.log")))
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    out_h = stdout_path.open("ab")
    err_h = stderr_path.open("ab")
    timed_out = False
    cancelled = False
    child: Optional[subprocess.Popen[bytes]] = None
    try:
        child = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=os.environ.copy(),
            stdin=subprocess.DEVNULL,
            stdout=out_h,
            stderr=err_h,
            start_new_session=True,
            close_fds=True,
        )
        record["pid"] = child.pid
        record["pgid"] = child.pid
        record["start_token"] = process_start_token(child.pid)
        record["supervisor_pid"] = os.getpid()
        record["supervisor_start_token"] = process_start_token(os.getpid())
        record["state"] = "RUNNING"
        record["running_at"] = time.time()
        atomic_write_json(path, record)
        timeout_s = record.get("timeout_s")
        deadline = (time.monotonic() + float(timeout_s)) if timeout_s is not None else None
        code: Optional[int] = None
        while True:
            code = child.poll()
            if code is not None:
                break
            live = _wait_record(path, timeout_s=0.01) or record
            if live.get("cancel_requested"):
                cancelled = True
                _terminate_session(child.pid)
                try:
                    child.wait(timeout=TERMINATE_GRACE_S + 0.5)
                except subprocess.TimeoutExpired:
                    pass
                code = child.poll()
                break
            if deadline is not None and time.monotonic() >= deadline:
                timed_out = True
                _terminate_session(child.pid)
                try:
                    child.wait(timeout=TERMINATE_GRACE_S + 0.5)
                except subprocess.TimeoutExpired:
                    pass
                code = child.poll()
                break
            time.sleep(POLL_S)
    except Exception as exc:
        record["terminal"] = "crashed"
        record["crash_reason"] = f"{type(exc).__name__}: {exc}"
        record["state"] = "TERMINAL"
        record["finished_at"] = time.time()
        atomic_write_json(path, record)
        return 70
    finally:
        for handle in (out_h, err_h):
            try:
                handle.close()
            except OSError:
                pass

    live = _wait_record(path, timeout_s=0.2) or record
    if live.get("cancel_requested"):
        cancelled = True
    live["returncode"] = code
    live["finished_at"] = time.time()
    live["state"] = "TERMINAL"
    live["terminal"] = classify_terminal(
        live, returncode=code, timed_out=timed_out, cancelled=cancelled
    )
    if live["terminal"] == "crashed":
        live["crash_reason"] = _crash_reason(code)
    elif live["terminal"] == "timed_out":
        live["crash_reason"] = "timeout"
    elif live["terminal"] == "cancelled":
        live["crash_reason"] = "cancelled"
    elif live["terminal"] == "unknown":
        live["crash_reason"] = live.get("crash_reason") or "exit code not observed"
    live["identity_status"] = identity_status(live)
    if live.get("expected_receipt_path") and Path(str(live["expected_receipt_path"])).is_file():
        live["result_location"] = live["expected_receipt_path"]
    atomic_write_json(path, live)
    return 0 if str(live.get("terminal") or "").startswith("completed") else 1


# ---------------------------------------------------------------------------
# Resident-callable WorkUnit (HCLI field set; concurrent-wave APIs not imported)
# ---------------------------------------------------------------------------


def emit_resident_workunit() -> dict[str, Any]:
    """WorkUnit the resident can schedule. Proposal only; does not self-verify."""
    unit = WorkUnit(
        id="future.detached.supervise-safe-work",
        role="science",
        description=(
            "Seal a WorkUnit, launch a real detached OS process, persist "
            "pid+start_token identity plus expected receipt path, and return "
            "scheduler control immediately. HCLI owns timeout, retry, "
            "cancellation and crash reason. GPU/lease/cargo commands SLEEP."
        ),
        resource_class="LIGHT_CONTROL",
        provider="future.detached",
        verifier="future.detached.child_terminal_classified",
        effect_class="READ_ONLY",
        workspace="repo-root",
        classification="STATIC_ONLY",
        status="pending",
    )
    row = unit.to_dict()
    row.update(
        {
            "command": [sys.executable, str(Path(__file__).resolve()), "--selftest"],
            "output_receipt_path": str((REPO / "receipts" / "future" / RECEIPT)),
            "claim_boundary": CLAIM_BOUNDARY,
            "species": "independent_reproduction",
            "requires_quiescence": False,
            "may_promote": False,
            "may_modify_verifier": False,
            "gpu_authority": False,
            "timeout_s": 60.0,
            "retry_policy": _retry_policy_of({}),
        }
    )
    WorkUnit.from_dict(dict(row))
    return row


def resident_callable() -> dict[str, Any]:
    unit = emit_resident_workunit()
    return {
        "callable": True,
        "entry_point": (
            "python3 tools/future/detached.py --selftest | --build | "
            "--launch WORKUNIT.json --workspace DIR | "
            "--inspect JOB_ID --workspace DIR | "
            "--cancel JOB_ID --workspace DIR"
        ),
        "python_api": "tools.future.detached.launch(workunit, workspace=...)",
        "workunit": unit,
        "workunit_id": unit["id"],
        "receipt": f"receipts/future/{RECEIPT}",
        "receipt_schema": SCHEMA,
        "frontier_fed": (
            "A terminal supervision record (and the expected child receipt path) "
            "is disk authority. The next refill inspects terminals: completed "
            "feeds the next unit; crashed may rerun from argv (RESUME_POLICY); "
            "SLEEPING waits for hardware qualification. frontiers.py / wakeup.py "
            "are the integration points and are not imported this wave."
        ),
        "fail_closed": [
            "unsafe command (GPU resource class, GPU argv, flock/lease, cargo build) "
            "raises UnsafeCommandError, persists SLEEPING, spawns nothing",
            "pid alive + mismatched start_token => reused, never signalled, terminal=unknown",
            "pid dead + no persisted exit code => unknown, never guessed crashed/completed",
            "cancel/timeout refuse to signal a pid whose identity is unproven",
            "write_receipt raises HardwareClaimError on a numeric hardware field",
            "partial child output is not completed-with-receipt unless the process exited 0",
        ],
        "how_it_fails_closed": (
            "launch() never starts cargo, Metal, CUDA, flock, or GPU_EXCLUSIVE work; "
            "those units SLEEP. A restarted supervisor re-adopts only on pid+start_token "
            "match. unknown is a first-class terminal class."
        ),
    }


# ---------------------------------------------------------------------------
# Negative-control proofs (real processes, not mocks)
# ---------------------------------------------------------------------------


def _sleep_unit(job: str, seconds: int, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": job,
        "role": "science",
        "description": f"harmless sleep {seconds}s",
        "command": ["/bin/sleep", str(int(seconds))],
        "resource_class": "LIGHT_CONTROL",
        "verifier": "future.detached.child_terminal_classified",
        "classification": "STATIC_ONLY",
    }
    body.update(extra)
    return body


def _python_unit(job: str, dash_c: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": job,
        "role": "science",
        "description": "harmless python3 -c",
        "command": [sys.executable, "-c", dash_c],
        "resource_class": "LIGHT_CONTROL",
        "verifier": "future.detached.child_terminal_classified",
        "classification": "STATIC_ONLY",
    }
    body.update(extra)
    return body


def prove_launch_returns_while_running(workspace: Path) -> dict[str, Any]:
    sup = DetachedSupervisor(workspace)
    t0 = time.monotonic()
    rec = sup.launch(_sleep_unit("launch-not-blocked", 8))
    elapsed = time.monotonic() - t0
    try:
        sup_pid = rec.get("supervisor_pid")
        work_pid = rec.get("pid")
        supervisor_alive = isinstance(sup_pid, int) and pid_is_alive(sup_pid)
        work_alive = isinstance(work_pid, int) and pid_is_alive(work_pid)
        still_running = supervisor_alive or work_alive
        returned_immediately = elapsed < 2.0
        passed = bool(returned_immediately and still_running and rec.get("terminal") is None)
        return {
            "passed": passed,
            "returned_immediately": returned_immediately,
            "supervisor_alive": supervisor_alive,
            "work_alive": work_alive,
            "terminal_at_return": rec.get("terminal"),
            "state_at_return": rec.get("state"),
        }
    finally:
        try:
            sup.cancel(rec["job_id"])
        except Exception:
            _terminate_session(rec.get("pid"))
            _terminate_session(rec.get("supervisor_pid"))


def prove_killed_classified_crashed(workspace: Path) -> dict[str, Any]:
    sup = DetachedSupervisor(workspace)
    rec = sup.launch(_sleep_unit("killed-mid-flight", 20))
    t0 = time.monotonic()
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not rec.get("pid"):
            rec = sup.inspect(rec["job_id"])
            time.sleep(0.02)
        pid = rec.get("pid")
        if not isinstance(pid, int):
            return {"passed": False, "reason": "work pid not recorded"}
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError as exc:
            return {"passed": False, "reason": f"kill failed: {exc}"}
        terminal = sup.wait_terminal(rec["job_id"], timeout_s=4.0)
        hung = (time.monotonic() - t0) > 8.0
        passed = (
            terminal.get("terminal") == "crashed"
            and not hung
            and not pid_is_alive(pid)
        )
        return {
            "passed": passed,
            "terminal": terminal.get("terminal"),
            "crash_reason": terminal.get("crash_reason"),
            "hung": hung,
            "reaped": not pid_is_alive(pid),
        }
    finally:
        try:
            sup.cancel(rec["job_id"])
        except Exception:
            _terminate_session(rec.get("pid"))
            _terminate_session(rec.get("supervisor_pid"))


def prove_stale_pid_not_mistaken(workspace: Path) -> dict[str, Any]:
    """A live pid with the original child's start_token missing/mismatched is not ours.

    Does not assert that the OS reused a pid (that encodes the kernel, not the
    guard). Plants a real second process and a forged record.
    """
    other = subprocess.Popen(
        ["/bin/sleep", "20"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    sup = DetachedSupervisor(workspace)
    rec = sup.launch(_sleep_unit("stale-pid-original", 20))
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not rec.get("start_token"):
            rec = sup.inspect(rec["job_id"])
            time.sleep(0.02)
        other_token = process_start_token(other.pid)
        forged = dict(rec)
        forged["pid"] = other.pid
        # Keep the ORIGINAL start_token so a pid-only checker would claim match.
        forged["job_id"] = rec["job_id"]
        status = identity_status(forged)
        not_match = status != "match"
        # adopt/cancel must not kill the other process
        adopted = dict(forged)
        adopted["job_id"] = rec["job_id"]
        # Persist the forged identity over a copy job to exercise cancel refusal.
        copy_id = f"stale-forged-{uuid.uuid4().hex[:8]}"
        adopted["job_id"] = copy_id
        adopted["terminal"] = None
        adopted["state"] = "RUNNING"
        adopted["cancel_requested"] = False
        adopted["launch_refused"] = False
        atomic_write_json(sup._path(copy_id), adopted)
        cancelled = sup.cancel(copy_id)
        other_still_alive = pid_is_alive(other.pid)
        signalled_wrong_child = not other_still_alive
        passed = (
            not_match
            and other_still_alive
            and cancelled.get("terminal") in {"unknown", None}
            and not signalled_wrong_child
        )
        return {
            "passed": passed,
            "forged_identity_status": status,
            "other_still_alive": other_still_alive,
            "cancel_terminal": cancelled.get("terminal"),
            "cancel_reason": cancelled.get("crash_reason"),
            "other_token_recorded": bool(other_token),
            "original_token_recorded": bool(rec.get("start_token")),
        }
    finally:
        try:
            sup.cancel(rec["job_id"])
        except Exception:
            _terminate_session(rec.get("pid"))
            _terminate_session(rec.get("supervisor_pid"))
        _terminate_session(other.pid)
        try:
            other.wait(timeout=2)
        except subprocess.TimeoutExpired:
            other.kill()


def prove_cancel_reaps(workspace: Path) -> dict[str, Any]:
    sup = DetachedSupervisor(workspace)
    rec = sup.launch(_sleep_unit("cancel-reaps", 20))
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not rec.get("pid"):
            rec = sup.inspect(rec["job_id"])
            time.sleep(0.02)
        cancelled = sup.cancel(rec["job_id"])
        pid = rec.get("pid")
        reaped = not (isinstance(pid, int) and pid_is_alive(pid))
        passed = cancelled.get("terminal") == "cancelled" and reaped
        return {
            "passed": passed,
            "terminal": cancelled.get("terminal"),
            "reaped": reaped,
        }
    finally:
        try:
            sup.cancel(rec["job_id"])
        except Exception:
            _terminate_session(rec.get("pid"))


def prove_timeout_reaps(workspace: Path) -> dict[str, Any]:
    sup = DetachedSupervisor(workspace)
    rec = sup.launch(_sleep_unit("timeout-reaps", 20, timeout_s=0.25))
    try:
        terminal = sup.wait_terminal(rec["job_id"], timeout_s=5.0)
        pid = terminal.get("pid") or rec.get("pid")
        reaped = not (isinstance(pid, int) and pid_is_alive(pid))
        passed = terminal.get("terminal") == "timed_out" and reaped
        return {
            "passed": passed,
            "terminal": terminal.get("terminal"),
            "reaped": reaped,
            "crash_reason": terminal.get("crash_reason"),
        }
    finally:
        try:
            sup.cancel(rec["job_id"])
        except Exception:
            _terminate_session(rec.get("pid"))


def prove_receipt_classes(workspace: Path) -> dict[str, Any]:
    sup = DetachedSupervisor(workspace)
    with_path = workspace / "results" / "with-receipt.json"
    rec_with = sup.launch(
        {
            "id": "completed-with",
            "role": "science",
            "description": "write expected receipt then exit 0",
            "command": [
                sys.executable,
                "-c",
                "import sys; from pathlib import Path; Path(sys.argv[1]).write_text('{\"ok\": true}\\n')",
                str(with_path),
            ],
            "resource_class": "LIGHT_CONTROL",
            "output_receipt_path": str(with_path),
            "verifier": "future.detached.child_terminal_classified",
        }
    )
    rec_without = sup.launch(
        _python_unit("completed-without", "raise SystemExit(0)")
    )
    try:
        with_term = sup.wait_terminal(rec_with["job_id"], timeout_s=5.0)
        without_term = sup.wait_terminal(rec_without["job_id"], timeout_s=5.0)
        passed = (
            with_term.get("terminal") == "completed-with-receipt"
            and without_term.get("terminal") == "completed-without-receipt"
        )
        return {
            "passed": passed,
            "with_receipt": with_term.get("terminal"),
            "without_receipt": without_term.get("terminal"),
            "receipt_present": with_path.is_file(),
        }
    finally:
        for rec in (rec_with, rec_without):
            try:
                sup.cancel(rec["job_id"])
            except Exception:
                _terminate_session(rec.get("pid"))


def prove_unsafe_refused(workspace: Path) -> dict[str, Any]:
    sup = DetachedSupervisor(workspace)
    trials = [
        (["cargo", "build"], "LIGHT_CONTROL"),
        (["/bin/sleep", "1"], "GPU_EXCLUSIVE"),
        (["flock", ".hcli/locks/protected-accelerator-bench.lock", "echo", "x"], "LIGHT_CONTROL"),
        (["xcrun", "metal", "foo.metal"], "LIGHT_CONTROL"),
        ([sys.executable, "-m", "hcli", "agentos", "protected-accelerator-bench"], "LIGHT_CONTROL"),
    ]
    rows: list[dict[str, Any]] = []
    for argv, rc in trials:
        refused = False
        spawned = False
        reason = ""
        try:
            rec = sup.launch(
                {
                    "id": "unsafe-" + Path(argv[0]).name,
                    "command": argv,
                    "resource_class": rc,
                    "role": "science",
                    "description": "must be refused",
                    "verifier": "future.detached.child_terminal_classified",
                }
            )
            spawned = rec.get("pid") is not None
            if rec.get("pid"):
                try:
                    sup.cancel(rec["job_id"])
                except Exception:
                    _terminate_session(rec.get("pid"))
        except UnsafeCommandError as exc:
            refused = True
            reason = exc.reason
            spawned = bool(exc.record and exc.record.get("pid"))
            if exc.record and exc.record.get("state") != "SLEEPING":
                refused = False
        rows.append(
            {
                "argv0": argv[0],
                "resource_class": rc,
                "refused": refused,
                "spawned": spawned,
                "reason": reason,
            }
        )
    passed = all(r["refused"] and not r["spawned"] for r in rows)
    return {"passed": passed, "trials": rows, "n_trials": len(rows)}


def prove_unknown_not_guessed(workspace: Path) -> dict[str, Any]:
    """A dead pid with no returncode is unknown, not crashed or completed."""
    sup = DetachedSupervisor(workspace)
    rec = {
        "schema": SUPERVISION_SCHEMA,
        "job_id": "unknown-dead",
        "argv": ["/bin/sleep", "1"],
        "cwd": str(workspace),
        "state": "RUNNING",
        "terminal": None,
        "pid": 2**30,  # not a live pid on this host
        "start_token": "0.000000",
        "timeout_s": None,
        "cancel_requested": False,
        "launch_refused": False,
        "expected_receipt_path": str(workspace / "results" / "missing.json"),
        "returncode": None,
        "crash_reason": None,
        "started_at": time.time(),
        "log_paths": {"stdout": str(workspace / "detached" / "logs" / "x.stdout.log"),
                      "stderr": str(workspace / "detached" / "logs" / "x.stderr.log")},
        "resource_lease": _empty_resource_lease(),
        "retry_policy": _retry_policy_of({}),
    }
    atomic_write_json(sup._path("unknown-dead"), rec)
    adopted = sup.adopt("unknown-dead")
    passed = adopted.get("terminal") == "unknown"
    return {
        "passed": passed,
        "terminal": adopted.get("terminal"),
        "crash_reason": adopted.get("crash_reason"),
        "identity_status": adopted.get("identity_status"),
    }


def run_all_proofs() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="hawking-detached-") as td:
        workspace = Path(td)
        proofs = {
            "launch_returns_while_child_running": prove_launch_returns_while_running(workspace),
            "killed_child_classified_crashed": prove_killed_classified_crashed(workspace),
            "stale_pid_not_mistaken_for_child": prove_stale_pid_not_mistaken(workspace),
            "cancel_reaps": prove_cancel_reaps(workspace),
            "timeout_reaps": prove_timeout_reaps(workspace),
            "receipt_classes": prove_receipt_classes(workspace),
            "unsafe_refused": prove_unsafe_refused(workspace),
            "unknown_not_guessed": prove_unknown_not_guessed(workspace),
        }
        # Derive pass/fail from the proof dicts; no fixed count of proofs in callers.
        names = sorted(proofs)
        failed = [name for name in names if not proofs[name].get("passed")]
        return {
            "proofs": {k: {ik: iv for ik, iv in proofs[k].items() if ik != "trials"} for k in names},
            "unsafe_trials": proofs["unsafe_refused"].get("trials"),
            "failed": failed,
            "all_passed": not failed,
            "n_proofs": len(names),
            "n_passed": len(names) - len(failed),
        }


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


RECOVERED_IMPLEMENTATION = [
    {
        "path": "hcli/agentos/background.py",
        "what": (
            "BackgroundJobStore.start uses subprocess.Popen(..., start_new_session=True, "
            "close_fds=True) and returns immediately; a --supervise child records the "
            "exit code. cancel SIGTERM then SIGKILL. After owner closure, a dead pid "
            "is INTERRUPTED, not invented COMPLETED/FAILED."
        ),
        "gap": (
            "_pid_alive is pid-only (os.kill(pid, 0)). Pid reuse would look live. "
            "No receipt-presence classification, no GPU/cargo refuse, no WorkUnit seal."
        ),
        "use": "session detach, persist, terminate sequence, INTERRUPTED/unknown honesty",
    },
    {
        "path": "hcli/resources.py",
        "what": (
            "process_start_token (libproc / procfs / ps) plus MutationLock.holder_is_live: "
            "a live pid with a mismatched start token is not the holder."
        ),
        "use": "pid+start_token is the child identity; pid alone is never trusted",
    },
    {
        "path": "hcli/agentos/recovery.py",
        "what": "kill resident+host, wait, recover_mission; fixture proof, not a success stamp",
        "use": "kill + wait + re-adopt; do not claim the child completed",
    },
    {
        "path": "hcli/workunit.py",
        "what": "RESUME_POLICY=rerun; a killed running unit is re-run from the start",
        "use": "retry_policy.policy=rerun; never resume a process mid-token",
    },
    {
        "path": "hcli/persist.py",
        "what": "atomic_write_json via temp + fsync + os.replace",
        "use": "supervision records and sealed workunits",
    },
    {
        "path": "tools/future/repro_science.py",
        "what": (
            "killed_subprocess + partial_result: SIGKILL a child, refuse partial stdout, "
            "resume from checkpoint. unknown/partial is not PASS."
        ),
        "use": "killed child => crashed (when exit observed) or unknown (when not); "
        "partial receipt file + nonzero exit is crashed",
    },
    {
        "path": "tools/future/qualification_pipeline.py",
        "what": "refuse_start_benchmark / refuse_create_lease; flock would be a seizure",
        "use": "unsafe-command gate; GPU units SLEEP rather than synthesising a result",
    },
    {
        "path": "hcli/executors.py",
        "what": "CPU path subprocess.run with timeout; TimeoutExpired is a failure, not success",
        "use": "timeout is a first-class terminal class (timed_out), not completed",
    },
    {
        "path": "hcli/agentos/runtime.py",
        "what": "AgentOS.start_background / cancel_background / recover_mission facade",
        "use": "cited as the resident call surface; this module is the sidecar equivalent",
    },
]

GAPS_CLOSED = [
    "launch() returns while the child is still alive; proven with /bin/sleep, not a mock",
    "supervision record persists pid, start_token, argv, cwd, logs, timeout, retry, "
    "cancel flag, resource_lease=refused, expected receipt, crash reason",
    "re-adopt uses pid+start_token; a live pid with a mismatched token is reused, not ours",
    "six terminal classes including unknown; unknown is never coerced to crashed/completed",
    "cancel and timeout actually signal the session and reap; cancelled child is not left running",
    "cargo build, GPU resource class, flock/lease, Metal/CUDA argv are refused; unit SLEEPS",
    "killed mid-flight classifies crashed (exit observed) and does not hang the caller",
]

NEGATIVE_FINDINGS = [
    "hcli/agentos/background.py still trusts pid alone; this lane did not patch hcli (forbidden)",
    "this process has no Metal GPU and no protected lease; GPU work is SLEEPING, not measured",
    "pid wraparound was not waited out; the reuse guard is proven by a live second process "
    "plus a forged record, which is the actual comparison the supervisor performs",
    "concurrent-wave modules (workgraph, wakeup, sandbox, super_resident, resident_api, "
    "frontiers) are not imported; local WorkUnit+launch is the swap point",
    "a missing start_token with a live pid is unknown identity: we will not kill it, "
    "and we will not claim it is our child",
]


def build() -> Path:
    proofs = run_all_proofs()
    if not proofs["all_passed"]:
        raise DetachedError("negative-control proofs failed: " + ", ".join(proofs["failed"]))
    callable_doc = resident_callable()
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Detached execution so the resident generation loop never sits waiting "
            "on cargo, a compiler, a download, a capability suite or a benchmark. "
            "HCLI owns PID, logs, timeout, retry, cancellation and crash reason."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "module_sha256": sha256_file(Path(__file__)),
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "fpga_note": (
            "FPGA belongs to Accelerator / Physical Compiler / Fusion. "
            "This module does not launch an FPGA or GPU job."
        ),
        "terminal_classes": list(TERMINAL_CLASSES),
        "identity_states": list(IDENTITY_STATES),
        "launch": {
            "returns_immediately": True,
            "spawns_real_os_process": True,
            "start_new_session": True,
            "records": [
                "pid",
                "start_token",
                "argv",
                "cwd",
                "started_at",
                "log_paths",
                "timeout_s",
                "retry_policy",
                "expected_receipt_path",
            ],
        },
        "supervision_record": {
            "schema": SUPERVISION_SCHEMA,
            "owned_by": "HCLI (this sidecar persists it; hcli/agentos/background.py is the recovered owner pattern)",
            "fields": [
                "pid",
                "start_token",
                "argv",
                "cwd",
                "started_at",
                "log_paths",
                "timeout_s",
                "retry_policy",
                "cancel_requested",
                "resource_lease",
                "expected_receipt_path",
                "result_location",
                "crash_reason",
                "terminal",
            ],
            "re_adopt": "pid AND start_token; pid alone is reusable and is not trusted",
            "resource_lease": _empty_resource_lease(),
        },
        "safe_commands": {
            "allowed_in_tests": ["/bin/sleep", "python3 -c (no GPU)"],
            "refused": [
                "cargo build/test/run",
                "GPU_EXCLUSIVE / GPU_DECODE / GPU_DIRTY_OK resource_class",
                "flock / protected-accelerator-bench.lock / SingletonLease",
                "xcrun metal / CUDA / protected-accelerator-bench",
            ],
            "on_refuse": "UnsafeCommandError; persist SLEEPING; spawn nothing; never a synthetic result",
        },
        "retry_policy": _retry_policy_of({}),
        "proofs": proofs["proofs"],
        "proofs_all_passed": proofs["all_passed"],
        "unsafe_trials": proofs["unsafe_trials"],
        "recovered_implementation": RECOVERED_IMPLEMENTATION,
        "gaps_closed": GAPS_CLOSED,
        "negative_findings": NEGATIVE_FINDINGS,
        "resident_callable": callable_doc,
        "integration_points": [
            "hcli.agentos.runtime.AgentOS.start_background / cancel_background — live HCLI surface; do not fork",
            "workgraph.py / wakeup.py / sandbox.py / super_resident.py / resident_api.py / frontiers.py "
            "— concurrent wave; local launch()+WorkUnit is the swap",
            "qualification_pipeline.py SLEEPING until an existing HCLI lease AND quiescence",
        ],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main(argv: Optional[Sequence[str]] = None) -> int:
    values = list(argv if argv is not None else sys.argv[1:])
    if len(values) >= 2 and values[0] == "--supervise":
        return supervise(values[1])
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--workspace")
    ap.add_argument("--launch", metavar="WORKUNIT.json")
    ap.add_argument("--inspect", metavar="JOB_ID")
    ap.add_argument("--cancel", metavar="JOB_ID")
    args = ap.parse_args(values)
    if args.launch or args.inspect or args.cancel:
        if not args.workspace:
            raise SystemExit("--workspace is required with --launch/--inspect/--cancel")
        sup = DetachedSupervisor(args.workspace)
        if args.launch:
            unit = json.loads(Path(args.launch).read_text(encoding="utf-8"))
            rec = sup.launch(unit)
            print(json.dumps(rec, indent=2, sort_keys=True))
            return 0
        if args.inspect:
            rec = sup.inspect(args.inspect)
            print(json.dumps(rec, indent=2, sort_keys=True))
            return 0
        rec = sup.cancel(args.cancel)
        print(json.dumps(rec, indent=2, sort_keys=True))
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLAIM_BOUNDARY",
    "DetachedError",
    "DetachedSupervisor",
    "RECEIPT",
    "SCHEMA",
    "TERMINAL_CLASSES",
    "UnsafeCommandError",
    "adopt",
    "build",
    "cancel",
    "classify_terminal",
    "emit_resident_workunit",
    "identity_status",
    "inspect",
    "launch",
    "main",
    "refuse_reason",
    "selftest",
    "supervise",
]
