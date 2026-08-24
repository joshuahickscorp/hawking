"""MAX-ramp policy helpers. This module is not a scheduler.

It records measured rungs, resolves the Grok admission cap, and counts
**live grok-run processes**. Occupancy is a process check, not a WorkUnit
flag. ``GrokBridge.status()`` is never called from here (it writes receipts).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from .persist import atomic_write_json


GROK_LADDER = (1, 2, 4, 6, 8)
EQUILIBRIUM_NAME = "max-equilibrium.json"
DEFAULT_TASKS_ROOT = Path.home() / ".claude-grok" / "tasks"
_THROTTLE_RE = re.compile(r"429|rate.?limit|too many requests", re.I)
LiveCheck = Callable[[str], bool]


def equilibrium_path(workspace: Union[str, Path]) -> Path:
    return Path(workspace) / ".hcli" / EQUILIBRIUM_NAME


def load_equilibrium(workspace: Union[str, Path]) -> Dict[str, Any]:
    path = equilibrium_path(workspace)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_equilibrium(workspace: Union[str, Path], payload: Dict[str, Any]) -> Path:
    path = equilibrium_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path


def resolve_grok_admitted(workspace: Union[str, Path]) -> tuple:
    raw = os.environ.get("HCLI_GROK_ADMITTED")
    if raw is not None and str(raw).strip() != "":
        try:
            value = int(str(raw).strip())
        except ValueError:
            value = None
        if value is not None and value >= 0:
            return value, "env:HCLI_GROK_ADMITTED"
    payload = load_equilibrium(workspace)
    if payload:
        try:
            value = int(payload.get("grok_admitted"))
        except (TypeError, ValueError):
            value = None
        if value is not None and value >= 0:
            return value, "equilibrium"
    return 1, "fallback"


def next_grok_rung(current: int) -> Optional[int]:
    try:
        cur = int(current)
    except (TypeError, ValueError):
        return GROK_LADDER[0]
    for step in GROK_LADDER:
        if step > cur:
            return step
    return None


def _grok_process_blob() -> Optional[str]:
    """Cmdlines of grok-related processes, or None if we could not observe.

    False-on-doubt: any spawn/timeout/unexpected exit becomes None, and
    callers must not count a task as live.
    """
    try:
        proc = subprocess.run(
            ["pgrep", "-fl", "grok"],
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode not in (0, 1):
        return None
    return proc.stdout or ""


def task_process_is_live(task_id: str, blob: Optional[str] = None) -> bool:
    """True iff a process cmdline contains ``task_id``. False on doubt."""
    if not task_id:
        return False
    if blob is None:
        blob = _grok_process_blob()
    if not blob:
        return False
    return task_id in blob


def _task_started_at_epoch(task_dir: Path) -> Optional[float]:
    path = task_dir / "metadata.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("started_at")
    if isinstance(raw, (int, float)):
        return float(raw)
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _scan_throttle(task_dir: Path) -> Optional[str]:
    for name in (".raw.err", ".raw.out", "stderr", "stdout"):
        path = task_dir / name
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if len(data) > 8192:
            data = data[-8192:]
        text = data.decode("utf-8", "replace")
        match = _THROTTLE_RE.search(text)
        if match:
            start = max(0, match.start() - 20)
            return text[start : start + 200]
    return None


def _wu_grok_counts(mission: Any) -> Dict[str, int]:
    """Secondary WU breakdown. Never used as ``active`` occupancy."""
    counts = {"wu_active": 0, "wu_queued": 0, "wu_done": 0, "wu_failed": 0}
    units = getattr(getattr(mission, "scheduler", None), "units", None) or {}
    values = units.values() if isinstance(units, dict) else units
    for wu in values:
        raw = str(getattr(wu, "resource_class", "") or "")
        role = str(getattr(wu, "role", "") or "")
        tid = getattr(wu, "backend_task_id", None)
        if raw.upper() != "GROK" and role.lower() != "grok" and not tid:
            continue
        status = getattr(wu, "status", "")
        if status == "running":
            counts["wu_active"] += 1
        elif status in ("pending", "ready"):
            counts["wu_queued"] += 1
        elif status == "completed":
            counts["wu_done"] += 1
        elif status == "failed":
            counts["wu_failed"] += 1
    return counts


def grok_pool_snapshot(
    workspace: Union[str, Path],
    mission: Any = None,
    *,
    tasks_root: Optional[Union[str, Path]] = None,
    is_live: Optional[LiveCheck] = None,
) -> Dict[str, Any]:
    """Count live grok-run tasks from ``~/.claude-grok/tasks/*/status``.

    ``running`` is active only if a process cmdline contains the task id.
    A stale running file is failed, not active. WorkUnit flags are a
    second ``wu_*`` breakdown and are never copied onto ``active``.
    """
    admitted, source = resolve_grok_admitted(workspace)
    root = Path(tasks_root) if tasks_root is not None else DEFAULT_TASKS_ROOT
    active = queued = done = failed = stale = 0
    latencies: List[float] = []
    throttle: Optional[str] = None
    now = time.time()
    blob: Optional[str] = None
    if is_live is None:
        blob = _grok_process_blob()

    def live(task_id: str) -> bool:
        if is_live is not None:
            try:
                return bool(is_live(task_id))
            except Exception:
                return False
        return task_process_is_live(task_id, blob)

    if root.is_dir():
        for entry in root.iterdir():
            if not entry.is_dir():
                continue
            st = entry / "status"
            if not st.is_file():
                continue
            try:
                state = st.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0]
            except (OSError, IndexError):
                continue
            state = (state or "").strip().lower()
            if state == "running":
                if live(entry.name):
                    active += 1
                    started = _task_started_at_epoch(entry)
                    if started:
                        latencies.append(max(0.0, now - started))
                    if throttle is None:
                        throttle = _scan_throttle(entry)
                else:
                    failed += 1
                    stale += 1
            elif state == "queued":
                queued += 1
            elif state == "done":
                done += 1
            elif state in ("failed", "error"):
                failed += 1
            if throttle is None and state in ("running", "failed", "error"):
                throttle = _scan_throttle(entry)

    latency_s: Optional[float] = None
    if latencies:
        latencies.sort()
        latency_s = latencies[len(latencies) // 2]

    snap: Dict[str, Any] = {
        "admitted": admitted,
        "admitted_source": source,
        "active": active,
        "queued": queued,
        "done": done,
        "failed": failed,
        "stale": stale,
        "latency_s": latency_s,
        "throttle": throttle,
    }
    if mission is not None:
        snap.update(_wu_grok_counts(mission))
    return snap


def _as_units(units: Any) -> Iterable[Any]:
    if units is None:
        return ()
    if isinstance(units, dict):
        return units.values()
    return units


def record_rung(
    workspace: Union[str, Path],
    *,
    requested: Any,
    admitted: Any,
    actual: Any,
    units: Any,
    elapsed_s: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    extra = extra or {}
    values = list(_as_units(units))
    verified = sum(1 for u in values if getattr(u, "status", None) == "completed")
    failed = sum(1 for u in values if getattr(u, "status", None) == "failed")
    retries = sum(max(0, int(getattr(u, "attempts", 1) or 1) - 1) for u in values)
    lat = sorted(
        (float(u.finished_at) - float(u.running_at))
        for u in values
        if getattr(u, "finished_at", None) and getattr(u, "running_at", None)
    )

    def pct(p: float) -> Optional[float]:
        if not lat:
            return None
        k = min(len(lat) - 1, max(0, int(round(p * (len(lat) - 1)))))
        return lat[k]

    qlat = sorted(
        (float(u.running_at) - float(u.ready_at))
        for u in values
        if getattr(u, "running_at", None) and getattr(u, "ready_at", None)
    )
    try:
        elapsed = float(elapsed_s or 0.0)
    except (TypeError, ValueError):
        elapsed = 0.0
    verified_per_hour = (verified / (elapsed / 3600.0)) if elapsed else 0.0
    payload = load_equilibrium(workspace)
    rungs = list(payload.get("rungs") or [])
    rungs.append(
        {
            "requested": requested,
            "admitted": admitted,
            "actual_active": actual,
            "verified": verified,
            "rejected": extra.get("rejected", 0),
            "failures": failed,
            "retries": retries,
            "queue_latency_s": (qlat[len(qlat) // 2] if qlat else None),
            "completion_latency_p50_s": pct(0.50),
            "completion_latency_p95_s": pct(0.95),
            "throttle": extra.get("throttle"),
            "verifier_wait_s": extra.get("verifier_wait_s"),
            "mutation_wait_s": extra.get("mutation_wait_s"),
            "scheduler_overhead_s": extra.get("scheduler_overhead_s"),
            "verified_units_per_hour": verified_per_hour,
            "elapsed_s": elapsed,
        }
    )
    payload["rungs"] = rungs
    payload["grok_admitted"] = admitted
    payload["grok_active"] = actual
    payload["verified_units_per_hour"] = rungs[-1]["verified_units_per_hour"]
    payload["source"] = "rung-measure"
    return save_equilibrium(workspace, payload)


def run_grok_ramp(workspace: Union[str, Path], run_rung, *, start: int = 1) -> Dict[str, Any]:
    """run_rung(admitted: int) -> dict with verified_units_per_hour and actual_active."""
    current: Optional[int] = int(start)
    best: Optional[Dict[str, Any]] = None
    while current is not None:
        os.environ["HCLI_GROK_ADMITTED"] = str(current)
        stats = run_rung(current)
        extra = dict(stats) if isinstance(stats, dict) else {}
        record_rung(
            workspace,
            requested=current,
            admitted=current,
            actual=stats["actual_active"],
            units=stats["units"],
            elapsed_s=stats["elapsed_s"],
            extra=extra,
        )
        rate = stats["verified_units_per_hour"]
        if best is not None and rate <= best["rate"]:
            payload = load_equilibrium(workspace)
            payload["stop_reason"] = (
                f"rung {current} did not beat {best['admitted']}"
            )
            payload["grok_admitted"] = best["admitted"]
            save_equilibrium(workspace, payload)
            return payload
        best = {"admitted": current, "rate": rate}
        current = next_grok_rung(current)
    return load_equilibrium(workspace)
