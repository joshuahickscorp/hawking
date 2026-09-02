"""What every live Hawking process is, so nobody has to guess from Activity Monitor.

The operator complaint this answers is concrete: the process list shows several
entries called `Python` and one long Rust path, and nothing says which is the
resident, which is a downloader, and which is safe to stop.

Classification is by ARGV, because that is the only thing that actually
distinguishes them - the executable name is `Python` for the supervisor, the
sovereign loop, the ModelLake watcher and every `hf download` child alike.

Nothing here is a guess: RSS, CPU and elapsed come from `ps`, and a process that
cannot be read is reported as unreadable rather than omitted.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# argv pattern -> (role, class, safe_to_stop, what it is)
# Order matters: the first match wins, so the specific patterns precede the
# generic ones. `hf download` must beat a bare `python` match.
_ROLES: List[tuple] = [
    (
        re.compile(r"hcli\.agentos\.resident\b.*--supervise"),
        ("resident-supervisor", "ESSENTIAL_PERSISTENT", False,
         "owns heartbeat, memory admission and restart limits; holds no model"),
    ),
    (
        re.compile(r"hcli_sovereign\.py.*--run"),
        ("sovereign-loop", "ESSENTIAL_PERSISTENT", False,
         "the SUB2 science loop; stopping it ends the running mission"),
    ),
    (
        re.compile(r"modellake_watch\.py"),
        ("modellake-watcher", "ESSENTIAL_PERSISTENT", False,
         "watches acquisitions and seals; parent of the download children"),
    ),
    (
        re.compile(r"\bhf\s+download\b"),
        ("modellake-download", "ESSENTIAL_EPHEMERAL", True,
         "one model acquisition; resumable, so stopping it loses only progress"),
    ),
    (
        # The EXECUTABLE must be a resident binary. `--artifact-root` alone is
        # not enough: gate scripts and probes carry that flag too, and a 3 MB
        # process labelled "resident-body" beside a 1.18 GB one is a label that
        # is worse than no label.
        re.compile(r"(?:^|/)\S*resident\S*\s+.*--(?:artifact-root|resident-identity)\b"),
        ("resident-body", "ESSENTIAL_PERSISTENT", False,
         "the loaded model itself; this is legitimate footprint, not overhead"),
    ),
    (
        re.compile(r"model_bearing_torture\.py"),
        ("torture-harness", "DEBUG_ONLY", True,
         "autonomy trial harness; not part of normal operation"),
    ),
    (
        re.compile(r"WU\.[A-Za-z0-9_.]+\.child\.py|mbt-run-"),
        ("resident-worker", "ESSENTIAL_EPHEMERAL", True,
         "one bounded WorkUnit slice; the mission requeues it"),
    ),
]

_PS_FIELDS = ("pid", "ppid", "rss", "pcpu", "etime", "command")


@dataclass
class Process:
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
            "body": self.body,
        }


def _classify(command: str) -> Optional[tuple]:
    for pattern, meta in _ROLES:
        if pattern.search(command):
            return meta
    return None


def _body_of(command: str) -> Optional[str]:
    """The model identity a process is carrying, if it names one.

    Identity is DATA. It is read off the argv rather than inferred from the
    executable's name, because the executable's name is not the architecture.
    """
    for pattern in (
        r"--resident-identity\s+(\S+)",
        r"--artifact-root\s+\S*?/([^/\s]+)\s*$",
        r"\bhf\s+download\s+(\S+)",
    ):
        match = re.search(pattern, command)
        if match:
            return match.group(1)
    return None


def _footprint_bytes(pid: int) -> Optional[int]:
    """Real memory footprint, the number Activity Monitor's Memory column shows.

    `ps -o rss` is NOT that number and on this platform it is not close. The
    resident body read rss=1.19 GB while its phys_footprint was 12 GB, and two
    `hf download` children read 2.93 and 1.73 GB against 29.84 GB each in
    Activity Monitor -- a ten-to-twentyfold under-report that made a box under
    real memory pressure look idle.

    RSS counts resident pages. It does not count what the compressor is holding
    on the process's behalf, and with 36 GB compressed system-wide that is most
    of the footprint. Reporting RSS here meant the one view built to answer
    "what is eating memory" answered it wrongly.
    """
    try:
        out = subprocess.run(
            ["footprint", "-p", str(pid)],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    match = re.search(r"phys_footprint:\s*([\d.]+)\s*(B|KB|MB|GB|TB)", out.stdout)
    if not match:
        return None
    scale = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3, "TB": 1024 ** 4}
    return int(float(match.group(1)) * scale[match.group(2)])


def live_processes(*, footprint: bool = True) -> List[Process]:
    """Every Hawking process on this host, classified. Empty list if ps fails."""
    try:
        out = subprocess.run(
            ["ps", "-eo", ",".join(_PS_FIELDS)],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []

    found: List[Process] = []
    for line in out.stdout.splitlines()[1:]:
        parts = line.split(None, len(_PS_FIELDS) - 1)
        if len(parts) < len(_PS_FIELDS):
            continue
        pid, ppid, rss, pcpu, etime, command = parts
        # `ps` itself and the grep pipeline that found it are not Hawking.
        if "ps -eo" in command:
            continue
        meta = _classify(command)
        if meta is None:
            continue
        role, klass, safe, purpose = meta
        try:
            pid_i = int(pid)
            measured = _footprint_bytes(pid_i) if footprint else None
            found.append(Process(
                pid=pid_i, ppid=int(ppid),
                # footprint when we can get it, rss only as a labelled fallback
                rss_bytes=measured if measured is not None else int(rss) * 1024,
                memory_source="phys_footprint" if measured is not None else "rss",
                cpu_percent=float(pcpu), elapsed=etime,
                role=role, process_class=klass, safe_to_stop=safe,
                purpose=purpose, command=command, body=_body_of(command),
            ))
        except ValueError:
            continue
    return sorted(found, key=lambda p: (-p.rss_bytes, p.pid))


def summary() -> Dict[str, Any]:
    """Roll-up for /status and for the process audit receipt."""
    procs = live_processes()
    by_class: Dict[str, int] = {}
    for proc in procs:
        by_class[proc.process_class] = by_class.get(proc.process_class, 0) + 1
    return {
        "count": len(procs),
        "total_rss_bytes": sum(p.rss_bytes for p in procs),
        "by_class": by_class,
        "roles": sorted({p.role for p in procs}),
        "processes": [p.to_dict() for p in procs],
    }


def render(procs: Optional[List[Process]] = None, width: int = 80) -> str:
    """One aligned line per process. Role first, because that is the question."""
    procs = live_processes() if procs is None else procs
    if not procs:
        return "no Hawking processes visible on this host"
    lines = [f"{'ROLE':<21} {'PID':>6} {'MEM':>8} {'CPU':>6} {'AGE':>9}  STOP"]
    for proc in procs:
        gib = proc.rss_bytes / 1024 ** 3
        rss = f"{gib:.2f}G" if gib >= 0.01 else f"{proc.rss_bytes // 1024 ** 2}M"
        lines.append(
            f"{proc.role:<21} {proc.pid:>6} {rss:>8} "
            f"{proc.cpu_percent:>5.1f}% {proc.elapsed:>9}  "
            f"{'yes' if proc.safe_to_stop else 'no'}"
        )
    total = sum(p.rss_bytes for p in procs) / 1024 ** 3
    fallback = sum(1 for p in procs if p.memory_source != "phys_footprint")
    note = f"  ({fallback} via rss fallback, under-reports)" if fallback else ""
    lines.append(f"{len(procs)} processes, {total:.2f}G footprint total{note}")
    return "\n".join(line[:width] for line in lines)


def orphaned_resident_bodies() -> List[Process]:
    """Resident bodies with no owner: reparented to pid 1 and claimed by nobody.

    `atexit` cleans up a clean exit and it works - a normal `hcli` invocation
    leaks nothing. But atexit CANNOT run on SIGKILL, and a daemon meets SIGKILL
    routinely: an OOM kill, a crash, a `kill -9`, a power loss. Measured here,
    SIGKILLing the CLI left an 11 GB model body at ppid=1 running forever.

    A body owned by the resident SUPERVISOR is also reparented to pid 1 (it is
    deliberately daemonised), so pid 1 alone is not evidence of abandonment.
    A body is only orphaned if no live resident state file claims its pid.
    """
    claimed = _claimed_worker_pids()
    return [
        proc for proc in live_processes(footprint=False)
        if proc.role == "resident-body"
        and proc.ppid == 1
        and proc.pid not in claimed
    ]


def _claimed_worker_pids() -> set:
    """Pids any live resident state file says it owns. Never reap these."""
    import json

    pids = set()
    for root in (Path.cwd(), Path(__file__).resolve().parents[1]):
        state = root / ".hcli" / "resident" / "state.json"
        try:
            data = json.loads(state.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for key in ("worker_pid", "supervisor_pid", "body_pid"):
            value = data.get(key)
            if isinstance(value, int):
                pids.add(value)
    return pids


def reap_orphaned_bodies(*, dry_run: bool = False) -> Dict[str, Any]:
    """Terminate unowned resident bodies. Safe to call at startup.

    Self-healing on startup rather than trusting the exit path, the same shape
    the ModelLake reconciliation pass uses and for the same reason: the failure
    happens precisely when the process that should have cleaned up is gone.
    """
    import os
    import signal as _signal

    found = orphaned_resident_bodies()
    reaped, failed = [], []
    for proc in found:
        if dry_run:
            continue
        try:
            os.kill(proc.pid, _signal.SIGTERM)
            reaped.append(proc.pid)
        except OSError as exc:
            failed.append({"pid": proc.pid, "error": str(exc)})
    return {
        "found": [p.pid for p in found],
        "bytes_held": sum(p.rss_bytes for p in found),
        "reaped": reaped,
        "failed": failed,
        "dry_run": dry_run,
    }


if __name__ == "__main__":
    print(render())
