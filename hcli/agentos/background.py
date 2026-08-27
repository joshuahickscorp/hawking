"""Durable, bounded background work for AgentOS.

Background work is a control-plane concern, not a model-provider concern.  A
job is an argv vector executed without a shell, with its lifecycle persisted
under ``.hcli/background``.  After sleep or process closure a new AgentOS
instance can inspect the job and explicitly resume an interrupted job from
the beginning when the caller marked it resumable.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hcli.persist import atomic_write_json


SCHEMA = "hcli.agentos.background_job.v1"
_SECRET_ARGUMENT_RE = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|authorization|password|secret|private[_-]?key|bearer|(?:hf|gh|github|openai|anthropic)[_-]?token)\s*[:=]"
)


def _now() -> float:
    return time.time()


def _within(path: Path, roots: Iterable[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root.resolve(strict=False))
            return True
        except ValueError:
            continue
    return False


def _pid_alive(pid: Optional[int]) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@dataclass
class BackgroundJob:
    job_id: str
    argv: List[str]
    cwd: str
    state: str = "PENDING"
    pid: Optional[int] = None
    label: Optional[str] = None
    log_path: Optional[str] = None
    resumable: bool = True
    parent_job_id: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    returncode: Optional[int] = None
    timeout_s: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "job_id": self.job_id,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "state": self.state,
            "pid": self.pid,
            "label": self.label,
            "log_path": self.log_path,
            "resumable": self.resumable,
            "parent_job_id": self.parent_job_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "returncode": self.returncode,
            "timeout_s": self.timeout_s,
            "error": self.error,
            "restart_policy": "explicit rerun from argv; never resume a process mid-token",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BackgroundJob":
        argv = value.get("argv")
        if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
            raise ValueError("background job argv is invalid")
        return cls(
            job_id=str(value.get("job_id") or ""),
            argv=list(argv),
            cwd=str(value.get("cwd") or ""),
            state=str(value.get("state") or "UNKNOWN"),
            pid=(int(value["pid"]) if value.get("pid") is not None else None),
            label=(str(value["label"]) if value.get("label") is not None else None),
            log_path=(str(value["log_path"]) if value.get("log_path") is not None else None),
            resumable=bool(value.get("resumable", True)),
            parent_job_id=(str(value["parent_job_id"]) if value.get("parent_job_id") else None),
            created_at=float(value.get("created_at") or 0.0),
            started_at=(float(value["started_at"]) if value.get("started_at") is not None else None),
            finished_at=(float(value["finished_at"]) if value.get("finished_at") is not None else None),
            returncode=(int(value["returncode"]) if value.get("returncode") is not None else None),
            timeout_s=(float(value["timeout_s"]) if value.get("timeout_s") is not None else None),
            error=(str(value["error"]) if value.get("error") else None),
        )


class BackgroundJobStore:
    """Persist and supervise shell-free background jobs."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        *,
        allowed_roots: Optional[Iterable[str | os.PathLike[str]]] = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.root = self.workspace / ".hcli" / "background"
        self.jobs_root = self.root / "jobs"
        self.logs_root = self.root / "logs"
        roots = [self.workspace]
        roots.extend(Path(item).expanduser().resolve() for item in (allowed_roots or ()))
        self.allowed_roots: Tuple[Path, ...] = tuple(dict.fromkeys(roots))
        self._children: Dict[str, Tuple[subprocess.Popen[str], Any]] = {}

    def _path(self, job_id: str) -> Path:
        token = str(job_id or "").strip()
        if not token or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in token):
            raise ValueError("invalid background job id")
        return self.jobs_root / f"{token}.json"

    def _write(self, job: BackgroundJob) -> None:
        atomic_write_json(self._path(job.job_id), job.to_dict())

    def _read(self, job_id: str) -> BackgroundJob:
        path = self._path(job_id)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"background job receipt is unreadable: {path}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"background job receipt is not an object: {path}")
        job = BackgroundJob.from_dict(value)
        if job.job_id != str(job_id):
            raise ValueError("background job id does not match receipt name")
        return job

    def _resolve_cwd(self, raw: Optional[str | os.PathLike[str]]) -> Path:
        path = Path(raw).expanduser() if raw is not None else self.workspace
        if not path.is_absolute():
            path = self.workspace / path
        path = path.resolve(strict=False)
        if not _within(path, self.allowed_roots):
            raise PermissionError(f"background cwd is outside allowed roots: {path}")
        if not path.is_dir():
            raise NotADirectoryError(path)
        return path

    def _refresh(self, job: BackgroundJob) -> BackgroundJob:
        child_entry = self._children.get(job.job_id)
        if child_entry is not None:
            child, handle = child_entry
            code = child.poll()
            if code is not None:
                job.returncode = int(code)
                job.finished_at = job.finished_at or _now()
                job.state = "COMPLETED" if code == 0 else "FAILED"
                if handle is not None:
                    try:
                        handle.close()
                    except OSError:
                        pass
                self._children.pop(job.job_id, None)
            elif job.timeout_s and job.started_at and _now() - job.started_at > job.timeout_s:
                self._terminate(job)
                job.state = "FAILED"
                job.error = "background job timed out"
                job.finished_at = _now()
        elif job.state == "RUNNING" and not _pid_alive(job.pid):
            # After process closure the parent cannot recover an exit code.
            # Preserve that distinction instead of inventing success/failure.
            job.state = "INTERRUPTED"
            job.finished_at = job.finished_at or _now()
            job.error = job.error or "owner process closed or child exited before status was persisted"
        self._write(job)
        return job

    def start(
        self,
        argv: Sequence[str],
        *,
        cwd: Optional[str | os.PathLike[str]] = None,
        label: Optional[str] = None,
        resumable: bool = True,
        env: Optional[Mapping[str, str]] = None,
        timeout_s: Optional[float] = None,
        parent_job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        command = [str(item) for item in argv]
        if not command or any(not item for item in command):
            raise ValueError("background argv must be a non-empty string sequence")
        if any(_SECRET_ARGUMENT_RE.search(item) for item in command):
            raise PermissionError("credential-shaped command arguments are not persisted; use env")
        workdir = self._resolve_cwd(cwd)
        timeout = None if timeout_s is None else max(0.1, min(7 * 24 * 3600.0, float(timeout_s)))
        job_id = f"job-{uuid.uuid4()}"
        log = self.logs_root / f"{job_id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        child_env = dict(os.environ)
        if env is not None:
            if not isinstance(env, Mapping) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
                raise TypeError("background env must be a string mapping")
            child_env.update(env)
        job = BackgroundJob(
            job_id=job_id,
            argv=command,
            cwd=str(workdir),
            state="STARTING",
            pid=None,
            label=str(label) if label else None,
            log_path=str(log),
            resumable=bool(resumable),
            parent_job_id=str(parent_job_id) if parent_job_id else None,
            created_at=_now(),
            started_at=_now(),
            timeout_s=timeout,
        )
        self._write(job)
        # A detached supervisor owns the child after this API process exits.
        # It records the final exit code in the same receipt, so a later CLI
        # invocation can distinguish COMPLETED from an interrupted process.
        supervisor_env = dict(child_env)
        source_root = str(Path(__file__).resolve().parents[2])
        pythonpath = supervisor_env.get("PYTHONPATH", "")
        supervisor_env["PYTHONPATH"] = os.pathsep.join(
            [source_root] + ([pythonpath] if pythonpath else [])
        )
        try:
            child = subprocess.Popen(
                [
                    os.fspath(os.environ.get("PYTHON", "")) or sys.executable,
                    "-m",
                    "hcli.agentos.background",
                    "--supervise",
                    str(self._path(job_id)),
                ],
                cwd=str(workdir),
                env=supervisor_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
        except Exception:
            job.state = "FAILED"
            job.error = "background supervisor could not start"
            job.finished_at = _now()
            self._write(job)
            raise
        job.state = "RUNNING"
        job.pid = child.pid
        self._children[job_id] = (child, None)
        self._write(job)
        return job.to_dict()

    def inspect(self, job_id: str) -> Dict[str, Any]:
        return self._refresh(self._read(job_id)).to_dict()

    def list(self) -> List[Dict[str, Any]]:
        if not self.jobs_root.is_dir():
            return []
        result: List[Dict[str, Any]] = []
        for path in sorted(self.jobs_root.glob("job-*.json")):
            try:
                result.append(self.inspect(path.stem))
            except (OSError, ValueError):
                continue
        return result

    def resume(self, job_id: str) -> Dict[str, Any]:
        old = self._refresh(self._read(job_id))
        if old.state == "RUNNING":
            return old.to_dict()
        if old.state == "COMPLETED":
            raise RuntimeError("completed background work is not resumable")
        if not old.resumable:
            raise PermissionError("background job was not marked resumable")
        return self.start(
            old.argv,
            cwd=old.cwd,
            label=old.label or f"resume:{old.job_id}",
            resumable=old.resumable,
            timeout_s=old.timeout_s,
            parent_job_id=old.job_id,
        )

    def _terminate(self, job: BackgroundJob) -> None:
        pid = job.pid
        if not isinstance(pid, int) or pid <= 0:
            return
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and _pid_alive(pid):
            time.sleep(0.02)
        if _pid_alive(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass

    def cancel(self, job_id: str) -> Dict[str, Any]:
        job = self._refresh(self._read(job_id))
        if job.state == "RUNNING":
            self._terminate(job)
            child_entry = self._children.pop(job.job_id, None)
            if child_entry is not None:
                child, handle = child_entry
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                if handle is not None:
                    try:
                        handle.close()
                    except OSError:
                        pass
            job.state = "CANCELLED"
            job.finished_at = _now()
            self._write(job)
        return job.to_dict()


def _supervise_job(receipt_path: str) -> int:
    """Detached supervisor entry point; the command is still never shell-parsed."""
    path = Path(receipt_path).expanduser().resolve()
    deadline = time.monotonic() + 5.0
    value: Optional[Dict[str, Any]] = None
    while time.monotonic() < deadline:
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                value = candidate
                if candidate.get("pid") == os.getpid() or candidate.get("state") == "RUNNING":
                    break
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        time.sleep(0.02)
    if value is None:
        return 70
    try:
        job = BackgroundJob.from_dict(value)
        workdir = Path(job.cwd).resolve(strict=False)
        log = Path(job.log_path or (path.parent.parent / "logs" / f"{job.job_id}.log"))
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("ab")
        try:
            child = subprocess.Popen(
                job.argv,
                cwd=str(workdir),
                env=os.environ.copy(),
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
            job.pid = os.getpid()
            job.state = "RUNNING"
            atomic_write_json(path, job.to_dict())
            code = int(child.wait())
        finally:
            try:
                handle.close()
            except OSError:
                pass
        job.returncode = code
        job.finished_at = _now()
        job.state = "COMPLETED" if code == 0 else "FAILED"
        atomic_write_json(path, job.to_dict())
        return max(0, min(255, code))
    except Exception as exc:  # supervisor failures remain visible in the job receipt
        try:
            job.error = f"{type(exc).__name__}: {exc}"
            job.state = "FAILED"
            job.finished_at = _now()
            atomic_write_json(path, job.to_dict())
        except Exception:
            pass
        return 70


def main(argv: Optional[Sequence[str]] = None) -> int:
    values = list(argv if argv is not None else sys.argv[1:])
    if len(values) == 2 and values[0] == "--supervise":
        return _supervise_job(values[1])
    raise SystemExit("background supervisor is an internal entry point")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BackgroundJob", "BackgroundJobStore", "SCHEMA", "main"]
