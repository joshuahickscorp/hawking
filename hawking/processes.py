"""Hawking's Python compatibility skin for the Rust process authority.

Host-wide process classification, footprint collection, owner claims, and
startup reaping live in Hawking's native process authority.  This module keeps
the Python-facing record shape and the small text renderer used by the legacy
REPL while forwarding every observation and signal decision to the Rust HAWKING
binary.  It intentionally has no ``ps`` parser, argv classifier, or kill
policy of its own.
"""
from __future__ import annotations

import atexit
import json
import os
import select
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class NativeProcessError(RuntimeError):
    """The Rust HAWKING process authority could not be invoked."""


class _NativeProcessServer:
    """Persistent adapter for the canonical Rust process inspector.

    The server is created only inside a daemon-owned process when the
    resident environment opts into native Gravity. Standalone compatibility
    calls retain the original one-shot Rust CLI path, while production HAWKING
    avoids paying its multi-second binary startup for every observation.
    """

    def __init__(self, root: Path, binary: Path) -> None:
        self.root = root
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [str(binary), "processes-server", "--workspace", str(root)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def request(self, payload: dict[str, object]) -> Optional[Dict[str, Any]]:
        with self._lock:
            if (
                self._process.poll() is not None
                or self._process.stdin is None
                or self._process.stdout is None
            ):
                return None
            try:
                self._process.stdin.write(json.dumps(payload) + "\n")
                self._process.stdin.flush()
                if not select.select([self._process.stdout], [], [], 30.0)[0]:
                    return None
                response = json.loads(self._process.stdout.readline())
            except (OSError, UnicodeError, ValueError):
                return None
        if not isinstance(response, dict) or response.get("ok") is False:
            return None
        result = response.get("result")
        return result if isinstance(result, dict) else None

    def stop(self) -> None:
        with self._lock:
            if self._process.poll() is None and self._process.stdin is not None:
                try:
                    self._process.stdin.write('{"op":"shutdown"}\n')
                    self._process.stdin.flush()
                except OSError:
                    pass
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except (OSError, subprocess.SubprocessError):
                pass


_NATIVE_PROCESS_SERVERS: Dict[Path, _NativeProcessServer] = {}
_NATIVE_PROCESS_LOCK = threading.Lock()


def _stop_native_process_servers() -> None:
    """Close daemon-owned Rust observers without affecting standalone callers."""
    with _NATIVE_PROCESS_LOCK:
        servers = list(_NATIVE_PROCESS_SERVERS.values())
        _NATIVE_PROCESS_SERVERS.clear()
    for server in servers:
        server.stop()


atexit.register(_stop_native_process_servers)


def _linked_checkout_root(active: Path) -> Optional[Path]:
    """Return the primary checkout that owns a linked worktree's gitdir."""
    marker = active / ".git"
    if marker.is_dir():
        return active
    if not marker.is_file():
        return None
    try:
        line = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not line.startswith("gitdir:"):
        return None
    gitdir = Path(line.split(":", 1)[1].strip()).expanduser()
    if not gitdir.is_absolute():
        gitdir = active / gitdir
    gitdir = gitdir.resolve(strict=False)
    common_git = next((parent for parent in gitdir.parents
                       if parent.name == ".git"), None)
    return common_git.parent if common_git is not None else None


def _native_binary(
    workspace: Optional[str | os.PathLike[str]] = None,
) -> Path:
    """Resolve the *process authority* binary, never the inference CLI.

    Two Rust programs historically used the filename ``hawking``.  The
    inference/runtime CLI owns the current shared build output, while
    ``hawking-process-authority`` owns ``processes`` and the long-lived
    ``processes-server`` JSON-lines protocol.  Selecting the former made the
    compatibility layer send a valid process request to an unrelated command
    surface.  Keep an explicit, separately named authority path first and
    fail closed when it is absent; process facts must not silently fall back to
    a Python ``ps`` parser.
    """
    explicit = (
        os.environ.get("HAWKING_PROCESS_AUTHORITY_BIN")
        or os.environ.get("HAWKING_NATIVE_PROCESS_BIN")
    )
    candidates = [Path(explicit)] if explicit else []
    checkout = Path(__file__).resolve().parents[1]
    active = Path(workspace or Path.cwd()).expanduser().resolve()
    primary = _linked_checkout_root(active)
    candidates.extend(
        [
            # Installed snapshots carry the exact process authority selected
            # at package time.  Do not use ``hawking-rust`` here: that name
            # denotes the inference CLI in current snapshots.
            checkout / "hawking-process-authority",
            # Migration-only fallback for previously built C2 helper output.
            checkout / "hawking-backend",
            # Installed Python snapshots live under ~/.local/share/hawking and do
            # not normally contain Cargo output; older snapshots continue to
            # use the active Hawking workspace, whose target directory is the
            # native owner's canonical build location.
            active / "target" / "release" / "hawking-process-authority",
            active / "target" / "debug" / "hawking-process-authority",
            active / "workspace" / "ops" / "build" / "rust" / "release" / "hawking-process-authority",
            active / "workspace" / "ops" / "build" / "rust" / "debug" / "hawking-process-authority",
            active / "target" / "release" / "hawking-backend",
            active / "target" / "debug" / "hawking-backend",
            active / "workspace" / "ops" / "build" / "rust" / "release" / "hawking-backend",
            active / "workspace" / "ops" / "build" / "rust" / "debug" / "hawking-backend",
            *(([
                primary / "hawking-process-authority",
                primary / "hawking-backend",
                primary / "workspace" / "ops" / "build" / "rust" / "release" / "hawking-process-authority",
                primary / "workspace" / "ops" / "build" / "rust" / "debug" / "hawking-process-authority",
                primary / "target" / "release" / "hawking-process-authority",
                primary / "target" / "debug" / "hawking-process-authority",
                primary / "workspace" / "ops" / "build" / "rust" / "release" / "hawking-backend",
                primary / "workspace" / "ops" / "build" / "rust" / "debug" / "hawking-backend",
                primary / "target" / "release" / "hawking-backend",
                primary / "target" / "debug" / "hawking-backend",
            ]) if primary is not None and primary != active else []),
            checkout / "target" / "release" / "hawking-process-authority",
            checkout / "target" / "debug" / "hawking-process-authority",
            checkout / "workspace" / "ops" / "build" / "rust" / "release" / "hawking-process-authority",
            checkout / "workspace" / "ops" / "build" / "rust" / "debug" / "hawking-process-authority",
            checkout.parent.parent / "workspace" / "ops" / "build" / "rust" / "release" / "hawking-process-authority",
            checkout.parent.parent / "workspace" / "ops" / "build" / "rust" / "debug" / "hawking-process-authority",
            checkout / "target" / "release" / "hawking-backend",
            checkout / "target" / "debug" / "hawking-backend",
            checkout / "workspace" / "ops" / "build" / "rust" / "release" / "hawking-backend",
            checkout / "workspace" / "ops" / "build" / "rust" / "debug" / "hawking-backend",
            checkout.parent.parent / "workspace" / "ops" / "build" / "rust" / "release" / "hawking-backend",
            checkout.parent.parent / "workspace" / "ops" / "build" / "rust" / "debug" / "hawking-backend",
        ]
    )
    named = shutil.which("hawking-process-authority") or shutil.which("hawking-backend")
    if named:
        candidates.append(Path(named))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise NativeProcessError(
        "Rust Hawking process authority is unavailable; build it with "
        "`cargo build -p hawking-process-authority --bin hawking-process-authority` or set "
        "HAWKING_PROCESS_AUTHORITY_BIN"
    )


def _native_result(
    *,
    workspace: Optional[str | os.PathLike[str]] = None,
    no_footprint: bool = False,
    orphaned: bool = False,
    reap: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    root = Path(workspace or Path.cwd()).expanduser().resolve()
    if os.environ.get("HAWKING_NATIVE_PROCESS_SERVER") == "1":
        operation = "reap" if reap else "orphaned" if orphaned else "inspect"
        request: Dict[str, object] = {"op": operation}
        if operation == "inspect":
            request["footprint"] = not no_footprint
        if operation == "reap":
            request["dry_run"] = dry_run
        server: Optional[_NativeProcessServer] = None
        try:
            binary = _native_binary(root)
            with _NATIVE_PROCESS_LOCK:
                server = _NATIVE_PROCESS_SERVERS.get(root)
                if server is None:
                    server = _NativeProcessServer(root, binary)
                    _NATIVE_PROCESS_SERVERS[root] = server
            result = server.request(request)
        except (NativeProcessError, OSError, subprocess.SubprocessError):
            result = None
        if result is not None:
            return result
        if server is not None:
            with _NATIVE_PROCESS_LOCK:
                if _NATIVE_PROCESS_SERVERS.get(root) is server:
                    _NATIVE_PROCESS_SERVERS.pop(root, None)
            server.stop()
    args = [str(_native_binary(root)), "processes", "--workspace", str(root), "--json"]
    if no_footprint:
        args.append("--no-footprint")
    if orphaned:
        args.append("--orphaned")
    if reap:
        args.append("--reap")
    if dry_run:
        args.append("--dry-run")
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise NativeProcessError(f"Rust HAWKING process inspection failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise NativeProcessError(detail or f"Rust HAWKING exited {completed.returncode}")
    try:
        payload = json.loads(completed.stdout)
        result = payload["result"]
    except (TypeError, ValueError, KeyError) as exc:
        raise NativeProcessError("Rust HAWKING returned invalid process JSON") from exc
    if not isinstance(result, dict):
        raise NativeProcessError("Rust HAWKING returned a non-object process result")
    return result


@dataclass
class Process:
    """Stable Python record shape retained for the legacy renderer and callers."""

    pid: int
    ppid: int
    rss_bytes: int
    cpu_percent: float
    elapsed: str
    role: str
    process_class: str
    safe_to_stop: bool
    purpose: str
    command: str
    body: Optional[str] = None
    memory_source: str = "rss"

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Process":
        return cls(
            pid=int(value["pid"]),
            ppid=int(value["ppid"]),
            rss_bytes=int(value["rss_bytes"]),
            cpu_percent=float(value["cpu_percent"]),
            elapsed=str(value["elapsed"]),
            role=str(value["role"]),
            process_class=str(value["class"]),
            safe_to_stop=bool(value["safe_to_stop"]),
            purpose=str(value["purpose"]),
            command=str(value["command"]),
            body=value.get("body"),
            memory_source=str(value.get("memory_source") or "rss"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pid": self.pid,
            "ppid": self.ppid,
            "rss_bytes": self.rss_bytes,
            "rss_gib": round(self.rss_bytes / 1024 ** 3, 3),
            "memory_source": self.memory_source,
            "cpu_percent": self.cpu_percent,
            "elapsed": self.elapsed,
            "role": self.role,
            "class": self.process_class,
            "safe_to_stop": self.safe_to_stop,
            "purpose": self.purpose,
            "command": self.command,
            "body": self.body,
        }


def live_processes(
    *,
    footprint: bool = True,
    workspace: Optional[str | os.PathLike[str]] = None,
) -> List[Process]:
    """Return the Rust authority's classified live host processes."""
    result = _native_result(workspace=workspace, no_footprint=not footprint)
    return [Process.from_dict(row) for row in result.get("processes", [])]


def summary(*, workspace: Optional[str | os.PathLike[str]] = None) -> Dict[str, Any]:
    """Return the Rust authority's process roll-up."""
    return _native_result(workspace=workspace)


def render(
    procs: Optional[List[Process]] = None,
    width: int = 80,
    *,
    workspace: Optional[str | os.PathLike[str]] = None,
) -> str:
    """Render the retained text view; all process facts come from Rust."""
    procs = live_processes(workspace=workspace) if procs is None else procs
    if not procs:
        return "no Hawking processes visible on this host"
    lines = [f"{'ROLE':<21} {'PID':>6} {'MEM':>8} {'CPU':>6} {'AGE':>9}  STOP"]
    for proc in procs:
        gib = proc.rss_bytes / 1024 ** 3
        memory = f"{gib:.2f}G" if gib >= 0.01 else f"{proc.rss_bytes // 1024 ** 2}M"
        lines.append(
            f"{proc.role:<21} {proc.pid:>6} {memory:>8} "
            f"{proc.cpu_percent:>5.1f}% {proc.elapsed:>9}  "
            f"{'yes' if proc.safe_to_stop else 'no'}"
        )
    total = sum(proc.rss_bytes for proc in procs) / 1024 ** 3
    fallback = sum(1 for proc in procs if proc.memory_source != "phys_footprint")
    note = f"  ({fallback} via rss fallback, under-reports)" if fallback else ""
    lines.append(f"{len(procs)} processes, {total:.2f}G footprint total{note}")
    return "\n".join(line[:width] for line in lines)


def orphaned_resident_bodies(
    *,
    workspace: Optional[str | os.PathLike[str]] = None,
) -> List[Process]:
    """Enumerate unclaimed resident bodies; this never sends a signal."""
    result = _native_result(workspace=workspace, no_footprint=True, orphaned=True)
    return [Process.from_dict(row) for row in result.get("orphaned", [])]


def reap_orphaned_bodies(
    *,
    dry_run: bool = False,
    workspace: Optional[str | os.PathLike[str]] = None,
) -> Dict[str, Any]:
    """Invoke the Rust startup-only reaper, preserving the Python result shape."""
    return _native_result(workspace=workspace, reap=True, dry_run=dry_run)


if __name__ == "__main__":
    print(render())
