"""HCLI's Python compatibility skin for the Rust process authority.

Host-wide process classification, footprint collection, owner claims, and
startup reaping live in ``hide-backend::process_inspector``.  This module keeps
the Python-facing record shape and the small text renderer used by the legacy
REPL while forwarding every observation and signal decision to the Rust HCLI
binary.  It intentionally has no ``ps`` parser, argv classifier, or kill
policy of its own.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


class NativeProcessError(RuntimeError):
    """The Rust HCLI process authority could not be invoked."""


def _native_binary() -> Path:
    """Resolve the built Rust HCLI binary without confusing it with Python hcli."""
    explicit = os.environ.get("HCLI_NATIVE_HCLI") or os.environ.get("HCLI_RUST_BIN")
    candidates = [Path(explicit)] if explicit else []
    checkout = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            checkout / "target" / "debug" / "hcli",
            checkout / "target" / "release" / "hcli",
            checkout / "workspace" / "ops" / "build" / "rust" / "debug" / "hcli",
            checkout / "workspace" / "ops" / "build" / "rust" / "release" / "hcli",
            checkout.parent.parent / "workspace" / "ops" / "build" / "rust" / "debug" / "hcli",
            checkout.parent.parent / "workspace" / "ops" / "build" / "rust" / "release" / "hcli",
        ]
    )
    named = shutil.which("hcli-rust")
    if named:
        candidates.append(Path(named))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise NativeProcessError(
        "Rust HCLI binary is unavailable; build it with "
        "`cargo build -p hide-backend --bin hcli` or set HCLI_NATIVE_HCLI"
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
    args = [str(_native_binary()), "processes", "--workspace", str(root), "--json"]
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
        raise NativeProcessError(f"Rust HCLI process inspection failed: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise NativeProcessError(detail or f"Rust HCLI exited {completed.returncode}")
    try:
        payload = json.loads(completed.stdout)
        result = payload["result"]
    except (TypeError, ValueError, KeyError) as exc:
        raise NativeProcessError("Rust HCLI returned invalid process JSON") from exc
    if not isinstance(result, dict):
        raise NativeProcessError("Rust HCLI returned a non-object process result")
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
