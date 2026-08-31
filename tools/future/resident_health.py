"""RESIDENT HEALTH — decide checkpoint-and-restart from a trend, never a spike.

A six-hour trial that restarts on one RSS spike will thrash, and thrashing
destroys the trial. This module takes process telemetry (resident+children
RSS, available memory, swap, tree shape, queue depth/staleness, detached
liveness, context growth, UMA pressure), trends a window, and returns
HEALTHY | DEGRADED | PATHOLOGICAL with the signal that fired.

PATHOLOGICAL requires a TREND: a monotonic climb with no plateau across
the window. One sample cannot produce a verdict. A dead resident is ABSENT
with rss_bytes null, never healthy-with-zero-RSS. An undeclared pid is not
invented from the largest RSS neighbour.

These numbers are SELF_MEASURED_DIRTY process telemetry. They authorize a
restart decision and nothing else. They never rank a model, a kernel, or a
representation, and they are not hardware performance claims.

Does not take a GPU lease, does not flock a bench lock, does not start a
resident, does not 0-fill a missing probe.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import importlib.util
import json
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hcli.resources import pid_is_alive, process_start_token
from tools.future._common import RECEIPTS, REPO, write_receipt
from tools.future.contamination import PRESSURE_NAMES, probe_memory
from tools.future.detached import identity_status

RECEIPT = "RESIDENT_HEALTH.json"
SCHEMA = "hawking.future.resident_health.v1"
SAMPLE_SCHEMA = "hawking.future.resident_health.sample.v1"
TREND_SCHEMA = "hawking.future.resident_health.trend.v1"
RECORDED_BY = "tools/future/resident_health.py"
VERSION = 1

TELEMETRY_CLASS = "SELF_MEASURED_DIRTY"
EVIDENCE_CLASS = "STATIC_ONLY"
VERDICTS = ("HEALTHY", "DEGRADED", "PATHOLOGICAL")
PRESENCE = ("PRESENT", "ABSENT", "UNDECLARED", "UNKNOWN")
DIRECTIONS = (
    "FLAT",
    "UP",
    "DOWN",
    "SPIKE",
    "CLIMB_NO_PLATEAU",
    "UNKNOWN",
)

MIN_TREND_SAMPLES = 2
MIN_PATHOLOGY_SAMPLES = 3
PENDING_STATUSES = frozenset({"pending", "ready"})

AUTHORIZES = "restart_decision_only"
CLAIM_BOUNDARY = (
    "SELF_MEASURED_DIRTY process telemetry. Authorizes a checkpoint-and-restart "
    "decision and nothing else. Never a model/kernel/representation ranking. "
    "Never DIAGNOSTIC_RELATIVE, never PROTECTED_ABSOLUTE, never a hardware field."
)

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

# Signals whose climb-without-plateau is PATHOLOGICAL. available_bytes is
# inverted: shrinking free memory is the climb.
PATHOLOGY_SIGNALS: tuple[tuple[str, bool], ...] = (
    ("rss_bytes", False),
    ("children_n", False),
    ("children_rss_bytes", False),
    ("available_bytes", True),
    ("swap_ins", False),
    ("uma_pressure_level", False),
    ("queue_depth", False),
    ("queue_staleness_s", False),
    ("detached_n_dead", False),
    ("context_bytes", False),
)

QUEUE_RECEIPT = RECEIPTS / "HCLI_FUTURE_WORKUNITS.json"
FRONTIER_RECEIPT = RECEIPTS / "FRONTIER_STATE.json"


class HealthRefuse(ValueError):
    """Fail closed: missing window, missing identity, or a guess that would look like success."""


# ---------------------------------------------------------------------------
# probes — every failure is a recorded reason, never a quiet 0
# ---------------------------------------------------------------------------


def _run(cmd: list[str], timeout: float = 8.0) -> tuple[int | None, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as exc:
        return None, "", f"{type(exc).__name__}: {exc}"


def _rss_bytes_libproc(pid: int) -> int | None:
    """Recovered from contamination._libproc_processes; bytes, not rounded GiB.

    Rounding RSS to 2 GiB decimals turns a live 5 MiB process into 0.00, which
    is exactly the healthy-with-zero-RSS fiction this module exists to refuse.
    """
    import ctypes
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("c") or "/usr/lib/libc.dylib")
    PROC_PIDTASKINFO = 4
    info = ctypes.create_string_buffer(96)
    libc.proc_pidinfo.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    libc.proc_pidinfo.restype = ctypes.c_int
    sz = libc.proc_pidinfo(int(pid), PROC_PIDTASKINFO, 0, info, 96)
    if sz < 16:
        return None
    _virt, rss = struct.unpack_from("<QQ", info.raw, 0)
    return int(rss)


def _rss_bytes_ps(pid: int) -> int | None:
    rc, out, _err = _run(["ps", "-o", "rss=", "-p", str(int(pid))], timeout=5)
    if rc != 0:
        return None
    text = (out or "").strip().split()
    if not text:
        return None
    try:
        kb = int(text[0])
    except ValueError:
        return None
    return kb * 1024


def rss_bytes_of(pid: int) -> int | None:
    """Resident set in bytes. None if unreadable. Does not 0-fill a dead pid."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        got = _rss_bytes_libproc(pid)
    except Exception:
        got = None
    if isinstance(got, int):
        return got
    return _rss_bytes_ps(pid)


def _process_tree(root_pid: int) -> dict[str, Any]:
    """Descendants of root_pid. n=0 is a real empty tree; missing ps is not n=0."""
    rc, out, err = _run(["ps", "-Ao", "pid,ppid,rss,state,comm"], timeout=8)
    if rc != 0 or not (out or "").strip():
        return {
            "status": "FAILED",
            "n": None,
            "rss_bytes_sum": None,
            "tree_depth": None,
            "reason": (err or f"ps rc={rc}").strip() or "ps produced no table",
            "method": None,
        }
    rows: list[dict[str, Any]] = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 4)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            rss_kb = int(parts[2])
        except ValueError:
            continue
        rows.append(
            {
                "pid": pid,
                "ppid": ppid,
                "rss_bytes": rss_kb * 1024,
            }
        )
    by_pid = {r["pid"]: r for r in rows}
    by_parent: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_parent.setdefault(int(row["ppid"]), []).append(row)
    if root_pid not in by_pid:
        return {
            "status": "FAILED",
            "n": None,
            "rss_bytes_sum": None,
            "tree_depth": None,
            "reason": f"pid {root_pid} not in ps table",
            "method": "ps_ppid",
        }
    seen = {root_pid}
    queue = [root_pid]
    descendants: list[dict[str, Any]] = []
    depth = {root_pid: 0}
    while queue:
        cur = queue.pop(0)
        for child in by_parent.get(cur, []):
            cid = int(child["pid"])
            if cid in seen:
                continue
            seen.add(cid)
            descendants.append(child)
            depth[cid] = depth[cur] + 1
            queue.append(cid)
    return {
        "status": "OK",
        "n": len(descendants),
        "rss_bytes_sum": sum(int(d["rss_bytes"]) for d in descendants),
        "tree_depth": max(depth.values()) if depth else 0,
        "reason": None,
        "method": "ps_ppid",
    }


def _observe_memory() -> dict[str, Any]:
    mem = probe_memory()
    pages = mem.get("pages") if isinstance(mem.get("pages"), dict) else {}
    bytes_map = mem.get("bytes") if isinstance(mem.get("bytes"), dict) else {}
    free = bytes_map.get("free")
    inactive = bytes_map.get("inactive")
    available: int | None
    if isinstance(free, int) and isinstance(inactive, int):
        available = free + inactive
    elif isinstance(free, int):
        available = free
    else:
        available = None
    swap_ins = pages.get("Swapins")
    swap_outs = pages.get("Swapouts")
    compressor = bytes_map.get("compressor")
    level = mem.get("pressure_level")
    if not isinstance(level, int):
        level = None
    name = mem.get("pressure_name")
    if not name:
        name = PRESSURE_NAMES.get(level, "UNKNOWN") if level is not None else "UNKNOWN"
    okish = mem.get("status") == "OK"
    have_any = available is not None or level is not None or isinstance(swap_ins, int)
    if not okish and not have_any:
        return {
            "status": "FAILED",
            "available_bytes": None,
            "ram_total_bytes": mem.get("ram_total_bytes") if isinstance(mem.get("ram_total_bytes"), int) else None,
            "swap_ins": None,
            "swap_outs": None,
            "compressor_bytes": None,
            "uma_pressure_level": None,
            "uma_pressure_name": "UNKNOWN",
            "reason": mem.get("reason") or "contamination.probe_memory failed",
            "source": "tools.future.contamination.probe_memory",
        }
    return {
        "status": "OK" if okish and available is not None else ("PARTIAL" if have_any else "FAILED"),
        "available_bytes": available,
        "ram_total_bytes": mem.get("ram_total_bytes") if isinstance(mem.get("ram_total_bytes"), int) else None,
        "swap_ins": swap_ins if isinstance(swap_ins, int) else None,
        "swap_outs": swap_outs if isinstance(swap_outs, int) else None,
        "compressor_bytes": compressor if isinstance(compressor, int) else None,
        "uma_pressure_level": level,
        "uma_pressure_name": str(name),
        "reason": None if okish else (mem.get("reason") or "partial memory probe"),
        "source": "tools.future.contamination.probe_memory",
    }


def _pending_age_s(unit: Mapping[str, Any], now: float) -> float | None:
    for key in ("ready_at", "running_at", "started_at"):
        raw = unit.get(key)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and raw > 0:
            return max(0.0, float(now) - float(raw))
    return None


def _observe_queue(*, now: float) -> dict[str, Any]:
    """Live queue on disk. HEAD is not the live queue; missing is UNOBSERVABLE, not 0."""
    if QUEUE_RECEIPT.is_file():
        try:
            doc = json.loads(QUEUE_RECEIPT.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return {
                "status": "FAILED",
                "depth": None,
                "oldest_pending_age_s": None,
                "source": str(QUEUE_RECEIPT.relative_to(REPO)),
                "reason": f"unreadable: {type(exc).__name__}: {exc}",
            }
        units = doc.get("work_units") if isinstance(doc.get("work_units"), list) else []
        pending = [
            u
            for u in units
            if isinstance(u, dict)
            and str(u.get("status") or "").lower() in PENDING_STATUSES
            and u.get("finished_at") in (None, "", 0)
        ]
        ages = [a for a in (_pending_age_s(u, now) for u in pending) if a is not None]
        return {
            "status": "OK",
            "depth": len(pending),
            "oldest_pending_age_s": max(ages) if ages else None,
            "source": str(QUEUE_RECEIPT.relative_to(REPO)),
            "reason": None
            if ages
            else ("pending units have no ready_at/running_at; staleness unobservable" if pending else None),
        }
    if FRONTIER_RECEIPT.is_file():
        try:
            doc = json.loads(FRONTIER_RECEIPT.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return {
                "status": "FAILED",
                "depth": None,
                "oldest_pending_age_s": None,
                "source": str(FRONTIER_RECEIPT.relative_to(REPO)),
                "reason": f"unreadable: {type(exc).__name__}: {exc}",
            }
        n = doc.get("n_next_work")
        return {
            "status": "PARTIAL" if isinstance(n, int) else "FAILED",
            "depth": n if isinstance(n, int) else None,
            "oldest_pending_age_s": None,
            "source": str(FRONTIER_RECEIPT.relative_to(REPO)),
            "reason": "n_next_work only; staleness unobservable",
        }
    return {
        "status": "UNOBSERVABLE",
        "depth": None,
        "oldest_pending_age_s": None,
        "source": None,
        "reason": "no queue receipt on disk; not defaulted to empty",
    }


def _observe_detached(workspace: Path | None) -> dict[str, Any]:
    """Read existing detached/jobs. Must not mkdir — that would mint an empty queue."""
    if workspace is None:
        return {
            "status": "UNOBSERVABLE",
            "n_jobs": None,
            "n_live": None,
            "n_dead": None,
            "n_unknown": None,
            "reason": "no workspace declared; not defaulted to zero jobs",
        }
    jobs_root = Path(workspace) / "detached" / "jobs"
    if not jobs_root.is_dir():
        return {
            "status": "UNOBSERVABLE",
            "n_jobs": None,
            "n_live": None,
            "n_dead": None,
            "n_unknown": None,
            "reason": f"{jobs_root} is not a directory; not defaulted to zero jobs",
        }
    n_live = n_dead = n_unknown = 0
    n_jobs = 0
    for path in sorted(jobs_root.glob("*.json")):
        try:
            rec = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            n_jobs += 1
            n_unknown += 1
            continue
        if not isinstance(rec, dict):
            n_jobs += 1
            n_unknown += 1
            continue
        n_jobs += 1
        status = identity_status(rec)
        if status == "match":
            n_live += 1
        elif status == "dead":
            n_dead += 1
        else:
            n_unknown += 1
    return {
        "status": "OK",
        "n_jobs": n_jobs,
        "n_live": n_live,
        "n_dead": n_dead,
        "n_unknown": n_unknown,
        "reason": None,
        "source": "tools.future.detached.identity_status",
    }


def _observe_context(context_path: Path | None) -> dict[str, Any]:
    if context_path is None:
        return {
            "status": "UNOBSERVABLE",
            "bytes": None,
            "reason": "no context path declared; will not guess a conversation log",
        }
    path = Path(context_path)
    if not path.is_file():
        return {
            "status": "UNOBSERVABLE",
            "bytes": None,
            "reason": f"{path} is not a file",
        }
    try:
        size = path.stat().st_size
    except OSError as exc:
        return {
            "status": "FAILED",
            "bytes": None,
            "reason": f"stat failed: {type(exc).__name__}: {exc}",
        }
    return {"status": "OK", "bytes": int(size), "reason": None, "path": str(path)}


def restart_supervisor_presence() -> dict[str, Any]:
    """Parallel lane may not have landed. Presence is recorded; the API is not guessed."""
    rel = "tools/future/restart_supervisor.py"
    on_disk = (REPO / rel).is_file()
    spec = importlib.util.find_spec("tools.future.restart_supervisor")
    return {
        "path": rel,
        "on_disk": on_disk,
        "importable": spec is not None,
        "landed": bool(on_disk or spec),
        "reason": None
        if (on_disk or spec)
        else "parallel lane had not landed; this module copes without it",
    }


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------


def _num(value: Any) -> int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def make_sample(
    *,
    presence: str,
    pid: int | None = None,
    start_token: str | None = None,
    identity_status_value: str | None = None,
    rss_bytes: int | None = None,
    children_n: int | None = None,
    children_rss_bytes: int | None = None,
    tree_depth: int | None = None,
    children_status: str = "UNOBSERVABLE",
    available_bytes: int | None = None,
    swap_ins: int | None = None,
    swap_outs: int | None = None,
    uma_pressure_level: int | None = None,
    uma_pressure_name: str | None = None,
    memory_status: str = "UNOBSERVABLE",
    queue_depth: int | None = None,
    oldest_pending_age_s: float | None = None,
    queue_status: str = "UNOBSERVABLE",
    detached_n_jobs: int | None = None,
    detached_n_live: int | None = None,
    detached_n_dead: int | None = None,
    detached_status: str = "UNOBSERVABLE",
    context_bytes: int | None = None,
    context_status: str = "UNOBSERVABLE",
    sampled_at_unix: float | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Construct a sample. Absent/undeclared cannot carry an RSS — that would look like zero-healthy."""
    if presence not in PRESENCE:
        raise HealthRefuse(f"presence {presence!r} is not one of {PRESENCE}")
    if presence in {"ABSENT", "UNDECLARED"} and rss_bytes is not None:
        raise HealthRefuse(
            f"presence={presence} cannot carry rss_bytes={rss_bytes!r}; "
            "that is the healthy-with-zero-RSS fiction"
        )
    if presence in {"ABSENT", "UNDECLARED"}:
        rss_bytes = None
        children_rss_bytes = None
    return {
        "schema": SAMPLE_SCHEMA,
        "evidence_class": TELEMETRY_CLASS,
        "gpu_authority": False,
        "authorizes": AUTHORIZES,
        "sampled_at_unix": sampled_at_unix,
        "resident": {
            "presence": presence,
            "pid": pid,
            "start_token": start_token,
            "identity_status": identity_status_value,
            "rss_bytes": rss_bytes,
            "reason": reason,
        },
        "children": {
            "status": children_status,
            "n": children_n,
            "rss_bytes_sum": children_rss_bytes,
            "tree_depth": tree_depth,
        },
        "memory": {
            "status": memory_status,
            "available_bytes": available_bytes,
            "swap_ins": swap_ins,
            "swap_outs": swap_outs,
            "uma_pressure_level": uma_pressure_level,
            "uma_pressure_name": uma_pressure_name,
        },
        "queue": {
            "status": queue_status,
            "depth": queue_depth,
            "oldest_pending_age_s": oldest_pending_age_s,
        },
        "detached": {
            "status": detached_status,
            "n_jobs": detached_n_jobs,
            "n_live": detached_n_live,
            "n_dead": detached_n_dead,
        },
        "context": {
            "status": context_status,
            "bytes": context_bytes,
        },
    }


def sample(
    *,
    pid: int | None = None,
    start_token: str | None = None,
    workspace: str | os.PathLike[str] | None = None,
    context_path: str | os.PathLike[str] | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """One reading. A missing resident is ABSENT with rss_bytes null, never 0."""
    sampled_at = float(time.time() if now is None else now)
    env_pid = os.environ.get("HAWKING_RESIDENT_PID")
    if pid is None and env_pid and env_pid.strip().lstrip("-").isdigit():
        pid = int(env_pid.strip())

    if pid is None:
        presence = "UNDECLARED"
        ident = None
        rss = None
        token = None
        reason = (
            "no resident pid declared (argument or HAWKING_RESIDENT_PID); "
            "will not invent one from the largest RSS neighbour "
            "(dirty_measure.identify_resident_loaded is a contamination envelope, not identity)"
        )
        tree = {
            "status": "UNOBSERVABLE",
            "n": None,
            "rss_bytes_sum": None,
            "tree_depth": None,
            "reason": "no resident pid; tree would be a guess",
            "method": None,
        }
    else:
        record = {"pid": pid, "start_token": start_token}
        ident = identity_status(record)
        token = process_start_token(pid)
        if ident == "dead" or not pid_is_alive(pid):
            presence = "ABSENT"
            rss = None
            reason = "resident pid is not alive"
            tree = {
                "status": "UNOBSERVABLE",
                "n": None,
                "rss_bytes_sum": None,
                "tree_depth": None,
                "reason": "resident absent; descendants of a dead pid are not the resident",
                "method": None,
            }
        elif ident == "reused":
            presence = "ABSENT"
            rss = None
            reason = "pid is alive but start_token does not match; the live process is not the resident"
            tree = {
                "status": "UNOBSERVABLE",
                "n": None,
                "rss_bytes_sum": None,
                "tree_depth": None,
                "reason": "pid reused; stranger tree is not the resident",
                "method": None,
            }
        else:
            presence = "PRESENT"
            rss = rss_bytes_of(pid)
            reason = None
            if ident == "unknown":
                reason = "start_token unproven; pid is alive"
            if rss is None:
                reason = (reason + "; " if reason else "") + "rss unreadable; left null, not 0"
            tree = _process_tree(pid)

    memory = _observe_memory()
    queue = _observe_queue(now=sampled_at)
    detached = _observe_detached(None if workspace is None else Path(workspace))
    context = _observe_context(None if context_path is None else Path(context_path))

    doc = make_sample(
        presence=presence,
        pid=pid,
        start_token=start_token or token,
        identity_status_value=ident,
        rss_bytes=rss,
        children_n=tree.get("n"),
        children_rss_bytes=tree.get("rss_bytes_sum"),
        tree_depth=tree.get("tree_depth"),
        children_status=str(tree.get("status") or "FAILED"),
        available_bytes=memory.get("available_bytes"),
        swap_ins=memory.get("swap_ins"),
        swap_outs=memory.get("swap_outs"),
        uma_pressure_level=memory.get("uma_pressure_level"),
        uma_pressure_name=memory.get("uma_pressure_name"),
        memory_status=str(memory.get("status") or "FAILED"),
        queue_depth=queue.get("depth"),
        oldest_pending_age_s=queue.get("oldest_pending_age_s"),
        queue_status=str(queue.get("status") or "UNOBSERVABLE"),
        detached_n_jobs=detached.get("n_jobs"),
        detached_n_live=detached.get("n_live"),
        detached_n_dead=detached.get("n_dead"),
        detached_status=str(detached.get("status") or "UNOBSERVABLE"),
        context_bytes=context.get("bytes"),
        context_status=str(context.get("status") or "UNOBSERVABLE"),
        sampled_at_unix=sampled_at,
        reason=reason,
    )
    doc["children"]["reason"] = tree.get("reason")
    doc["children"]["method"] = tree.get("method")
    doc["memory"]["ram_total_bytes"] = memory.get("ram_total_bytes")
    doc["memory"]["compressor_bytes"] = memory.get("compressor_bytes")
    doc["memory"]["reason"] = memory.get("reason")
    doc["memory"]["source"] = memory.get("source")
    doc["queue"]["source"] = queue.get("source")
    doc["queue"]["reason"] = queue.get("reason")
    doc["detached"]["reason"] = detached.get("reason")
    doc["detached"]["n_unknown"] = detached.get("n_unknown")
    doc["context"]["reason"] = context.get("reason")
    if context.get("path"):
        doc["context"]["path"] = context["path"]
    doc["claim_boundary"] = CLAIM_BOUNDARY
    return doc


# ---------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------


def _is_spike(series: Sequence[int | float]) -> bool:
    """One unique maximum, everything else flat. A climb's last point is a max, but the rest is not flat."""
    if len(series) < MIN_PATHOLOGY_SAMPLES:
        return False
    peak = max(series)
    if series.count(peak) != 1:
        return False
    idx = list(series).index(peak)
    rest = list(series)[:idx] + list(series)[idx + 1 :]
    return bool(rest) and all(x == rest[0] for x in rest)


def classify_series(
    values: Sequence[Any],
    *,
    invert: bool = False,
    min_pathology: int = MIN_PATHOLOGY_SAMPLES,
) -> dict[str, Any]:
    """Direction of one numeric window. SPIKE is not CLIMB_NO_PLATEAU."""
    present = [_num(v) for v in values]
    nums = [v for v in present if v is not None]
    n = len(nums)
    if n < MIN_TREND_SAMPLES:
        return {
            "direction": "UNKNOWN",
            "reason": "fewer than 2 numeric points; cannot trend",
            "plateau": False,
            "n_numeric": n,
            "n_window": len(values),
            "pathology_eligible": False,
            "invert": invert,
            "first": None,
            "last": None,
        }
    series: list[int | float] = ([-v for v in nums] if invert else list(nums))
    diffs = [series[i + 1] - series[i] for i in range(len(series) - 1)]
    plateau = any(d == 0 for d in diffs)
    strictly_up = all(d > 0 for d in diffs)
    strictly_down = all(d < 0 for d in diffs)
    if all(x == series[0] for x in series):
        direction = "FLAT"
        why = "all numeric points equal"
    elif _is_spike(series):
        direction = "SPIKE"
        why = "one unique extremum in an otherwise flat series"
    elif strictly_up and not plateau and n >= min_pathology:
        direction = "CLIMB_NO_PLATEAU"
        why = "monotonic climb across the window with no plateau"
    elif strictly_up:
        direction = "UP"
        why = "increasing, but window shorter than pathology or a plateau is present"
    elif strictly_down:
        direction = "DOWN"
        why = "monotonic decrease"
    elif series[-1] > series[0]:
        direction = "UP"
        why = "net increase without a clean monotonic climb"
    elif series[-1] < series[0]:
        direction = "DOWN"
        why = "net decrease without a clean monotonic drop"
    else:
        direction = "FLAT"
        why = "no net movement; noise is not a climb"
    return {
        "direction": direction,
        "reason": why,
        "plateau": plateau,
        "n_numeric": n,
        "n_window": len(values),
        "pathology_eligible": direction == "CLIMB_NO_PLATEAU",
        "invert": invert,
        "first": nums[0],
        "last": nums[-1],
    }


def _pull(sample_doc: Mapping[str, Any], key: str) -> Any:
    resident = sample_doc.get("resident") if isinstance(sample_doc.get("resident"), dict) else {}
    children = sample_doc.get("children") if isinstance(sample_doc.get("children"), dict) else {}
    memory = sample_doc.get("memory") if isinstance(sample_doc.get("memory"), dict) else {}
    queue = sample_doc.get("queue") if isinstance(sample_doc.get("queue"), dict) else {}
    detached = sample_doc.get("detached") if isinstance(sample_doc.get("detached"), dict) else {}
    context = sample_doc.get("context") if isinstance(sample_doc.get("context"), dict) else {}
    table = {
        "rss_bytes": resident.get("rss_bytes"),
        "children_n": children.get("n"),
        "children_rss_bytes": children.get("rss_bytes_sum"),
        "available_bytes": memory.get("available_bytes"),
        "swap_ins": memory.get("swap_ins"),
        "uma_pressure_level": memory.get("uma_pressure_level"),
        "queue_depth": queue.get("depth"),
        "queue_staleness_s": queue.get("oldest_pending_age_s"),
        "detached_n_dead": detached.get("n_dead"),
        "context_bytes": context.get("bytes"),
    }
    return table[key]


def _presence_of(sample_doc: Mapping[str, Any]) -> str:
    resident = sample_doc.get("resident") if isinstance(sample_doc.get("resident"), dict) else {}
    got = str(resident.get("presence") or "UNKNOWN")
    return got if got in PRESENCE else "UNKNOWN"


def trend(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Direction with the evidence. One sample is a moment and is refused."""
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise HealthRefuse("trend() needs a sequence of samples")
    rows = [s for s in samples if isinstance(s, Mapping)]
    if len(rows) != len(list(samples)):
        raise HealthRefuse("trend() refuses a window that contains a non-sample")
    if len(rows) < MIN_TREND_SAMPLES:
        raise HealthRefuse(
            f"one sample cannot produce a trend verdict (n={len(rows)}); "
            "PATHOLOGICAL requires a window, not a moment"
        )
    times = [s.get("sampled_at_unix") for s in rows]
    dated = [_num(t) is not None for t in times]
    if any(dated) and not all(dated):
        raise HealthRefuse("timestamps are partial; will not invent order")
    if all(dated):
        ordered = [_num(t) for t in times]
        for i in range(len(ordered) - 1):
            earlier, later = ordered[i], ordered[i + 1]
            if earlier is not None and later is not None and earlier > later:
                raise HealthRefuse("samples are not in time order; will not reverse a window into a climb")
    signals = {}
    for name, invert in PATHOLOGY_SIGNALS:
        values = [_pull(s, name) for s in rows]
        signals[name] = classify_series(values, invert=invert)
    presences = [_presence_of(s) for s in rows]
    return {
        "schema": TREND_SCHEMA,
        "status": "OK",
        "n_samples": len(rows),
        "min_trend_samples": MIN_TREND_SAMPLES,
        "min_pathology_samples": MIN_PATHOLOGY_SAMPLES,
        "resident_presence": presences,
        "signals": signals,
        "evidence_class": TELEMETRY_CLASS,
        "gpu_authority": False,
        "authorizes": AUTHORIZES,
        "claim_boundary": CLAIM_BOUNDARY,
    }


# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------


def _as_trend(trend_or_samples: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if isinstance(trend_or_samples, Mapping) and trend_or_samples.get("schema") == TREND_SCHEMA:
        return dict(trend_or_samples)
    if isinstance(trend_or_samples, Sequence) and not isinstance(trend_or_samples, (str, bytes)):
        return trend(trend_or_samples)
    raise HealthRefuse("verdict() needs a trend document or a sequence of samples")


def verdict(trend_or_samples: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """HEALTHY | DEGRADED | PATHOLOGICAL with the signal that fired.

    PATHOLOGICAL is a trend, never a single reading. A refused window raises
    rather than coming back HEALTHY.
    """
    tr = _as_trend(trend_or_samples)
    if tr.get("status") != "OK":
        raise HealthRefuse(tr.get("reason") or "trend is not OK")
    presences = list(tr.get("resident_presence") or [])
    signals = tr.get("signals") if isinstance(tr.get("signals"), dict) else {}
    n = int(tr.get("n_samples") or 0)

    pathological: list[str] = []
    degraded: list[str] = []

    if presences and all(p == "ABSENT" for p in presences):
        pathological.append("resident_absent")
    elif "ABSENT" in presences and "PRESENT" in presences:
        degraded.append("resident_flapping")

    for name, row in signals.items():
        if not isinstance(row, dict):
            continue
        direction = row.get("direction")
        if direction == "CLIMB_NO_PLATEAU":
            pathological.append(name)
        elif direction in {"SPIKE", "UP"}:
            degraded.append(name)

    present_throughout = bool(presences) and all(p == "PRESENT" for p in presences)

    if pathological:
        status = "PATHOLOGICAL"
        signal = pathological[0]
        why = (
            f"{signal} produced a monotonic climb with no plateau across {n} samples"
            if signal != "resident_absent"
            else f"resident ABSENT on all {n} samples in the window"
        )
    elif degraded:
        status = "DEGRADED"
        signal = degraded[0]
        why = f"{signal} moved (spike/up/identity) without a climb-without-plateau"
    elif present_throughout:
        status = "HEALTHY"
        signal = None
        why = "resident PRESENT throughout; no climb-without-plateau and no degrading movement"
    else:
        raise HealthRefuse(
            "no observable resident identity in the window; refusing HEALTHY on silence"
        )

    return {
        "verdict": status,
        "signal": signal,
        "triggers": {"pathological": pathological, "degraded": degraded},
        "reason": why,
        "n_samples": n,
        "authorizes_restart": status == "PATHOLOGICAL",
        "evidence_class": TELEMETRY_CLASS,
        "gpu_authority": False,
        "authorizes": AUTHORIZES,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def authorize_restart(verdict_doc: Mapping[str, Any]) -> bool:
    """True only for PATHOLOGICAL. A spike must not restart the resident."""
    return verdict_doc.get("verdict") == "PATHOLOGICAL" and bool(verdict_doc.get("authorizes_restart"))


# ---------------------------------------------------------------------------
# proofs — executed, not declared
# ---------------------------------------------------------------------------


def _series(rss: Iterable[int], *, presence: str = "PRESENT") -> list[dict[str, Any]]:
    out = []
    for i, value in enumerate(rss):
        out.append(
            make_sample(
                presence=presence,
                pid=4242 if presence == "PRESENT" else None,
                rss_bytes=value if presence == "PRESENT" else None,
                sampled_at_unix=1_700_000_000.0 + i,
                children_status="OK",
                children_n=0,
                memory_status="OK",
                available_bytes=8 * 1024 ** 3,
                swap_ins=0,
                uma_pressure_level=0,
                uma_pressure_name="normal",
                queue_status="UNOBSERVABLE",
                detached_status="UNOBSERVABLE",
                context_status="UNOBSERVABLE",
            )
        )
    return out


def prove_synthetic() -> dict[str, Any]:
    """The four mandatory controls, plus HEALTHY reachability. No live processes."""
    results: dict[str, Any] = {}

    one = _series([100])
    try:
        trend(one)
        results["one_sample_refuses"] = {"passed": False, "reason": "trend accepted one sample"}
    except HealthRefuse as exc:
        try:
            verdict(one)
            results["one_sample_refuses"] = {
                "passed": False,
                "reason": f"trend refused ({exc}) but verdict did not",
            }
        except HealthRefuse as exc2:
            results["one_sample_refuses"] = {
                "passed": True,
                "reason": str(exc2),
            }

    spike = _series([100, 100, 800, 100, 100])
    spike_v = verdict(spike)
    results["spike_not_pathological"] = {
        "passed": spike_v["verdict"] != "PATHOLOGICAL" and not authorize_restart(spike_v),
        "verdict": spike_v["verdict"],
        "signal": spike_v["signal"],
    }

    climb = _series([100, 200, 400, 800, 1600])
    climb_v = verdict(climb)
    results["climb_is_pathological"] = {
        "passed": climb_v["verdict"] == "PATHOLOGICAL"
        and climb_v["signal"] == "rss_bytes"
        and authorize_restart(climb_v),
        "verdict": climb_v["verdict"],
        "signal": climb_v["signal"],
    }

    plateau = _series([100, 200, 400, 400, 400])
    plateau_v = verdict(plateau)
    results["plateau_not_pathological"] = {
        "passed": plateau_v["verdict"] != "PATHOLOGICAL",
        "verdict": plateau_v["verdict"],
        "signal": plateau_v["signal"],
    }

    flat = _series([200, 200, 200, 200])
    flat_v = verdict(flat)
    results["flat_is_healthy"] = {
        "passed": flat_v["verdict"] == "HEALTHY" and not authorize_restart(flat_v),
        "verdict": flat_v["verdict"],
        "signal": flat_v["signal"],
    }

    try:
        make_sample(presence="ABSENT", rss_bytes=0)
        results["absent_rejects_zero_rss"] = {
            "passed": False,
            "reason": "make_sample accepted ABSENT with rss_bytes=0",
        }
    except HealthRefuse as exc:
        results["absent_rejects_zero_rss"] = {"passed": True, "reason": str(exc)}

    absent = [
        make_sample(presence="ABSENT", pid=9, sampled_at_unix=1_700_000_000.0 + i)
        for i in range(3)
    ]
    absent_v = verdict(absent)
    results["persistent_absent_is_pathological"] = {
        "passed": absent_v["verdict"] == "PATHOLOGICAL" and absent_v["signal"] == "resident_absent",
        "verdict": absent_v["verdict"],
        "signal": absent_v["signal"],
    }

    results["all_passed"] = all(bool(row.get("passed")) for row in results.values() if isinstance(row, dict))
    results["verdicts_reached"] = sorted(
        {
            spike_v["verdict"],
            climb_v["verdict"],
            plateau_v["verdict"],
            flat_v["verdict"],
            absent_v["verdict"],
        }
    )
    return results


def prove_live_dead_pid() -> dict[str, Any]:
    """A real dead pid must come back ABSENT with rss_bytes null. Spawn failure is recorded, not skipped."""
    try:
        proc = subprocess.Popen(
            ["/bin/sleep", "30"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return {
            "passed": False,
            "coped": True,
            "reason": f"could not spawn /bin/sleep: {type(exc).__name__}: {exc}",
        }
    pid = proc.pid
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception as exc:
        try:
            proc.kill()
        except OSError as kill_exc:
            return {
                "passed": False,
                "coped": True,
                "pid": pid,
                "reason": (
                    f"could not reap child: {type(exc).__name__}: {exc}; "
                    f"kill: {type(kill_exc).__name__}: {kill_exc}"
                ),
            }
        return {
            "passed": False,
            "coped": True,
            "pid": pid,
            "reason": f"could not reap child: {type(exc).__name__}: {exc}",
        }
    dead = sample(pid=pid)
    rss = dead["resident"]["rss_bytes"]
    passed = (
        dead["resident"]["presence"] == "ABSENT"
        and rss is None
        and rss != 0
    )
    return {
        "passed": passed,
        "coped": True,
        "pid": pid,
        "presence": dead["resident"]["presence"],
        "rss_bytes": rss,
        "reason": dead["resident"]["reason"],
    }


# ---------------------------------------------------------------------------
# receipt
# ---------------------------------------------------------------------------


def recovered_implementation() -> list[str]:
    landed = restart_supervisor_presence()
    return [
        "tools/future/contamination.py probe_memory (UMA pressure, vm_stat pages including Swapins/Swapouts, available bytes); libproc RSS idea kept in bytes so small processes do not round to 0.00 GiB",
        "tools/future/detached.py identity_status (match/dead/reused/unknown; pid is not identity)",
        "hcli/resources.py pid_is_alive + process_start_token",
        "tools/future/hardware_doctor.py metal_state — capability probe that is not a measurement (is_a_measurement false; this module's sample is the process-telemetry analogue)",
        "tools/future/dirty_measure.py SELF_MEASURED_DIRTY class and identify_resident_loaded (explicitly NOT used as identity: largest RSS neighbour is not the resident)",
        "tools/future/resident_install.py readiness_probe already names identity.resident_health.pid; this module is that pid's health, not a second install contract",
        "tools/future/hbm_doctor.py ResidentCandidate.missing_inputs stay None, never 0-filled — same fail-closed for rss_bytes",
        (
            f"{landed['path']} LANDED"
            if landed["landed"]
            else f"{landed['path']} NOT LANDED ({landed['reason']})"
        ),
    ]


def gaps_closed() -> list[str]:
    return [
        "sample() / trend() / verdict() as a restart decision with evidence, not a guess",
        "PATHOLOGICAL requires CLIMB_NO_PLATEAU across the window; a single spike cannot authorize restart",
        "one sample refuses a trend verdict (HealthRefuse), never HEALTHY-by-default",
        "dead/reused resident is ABSENT with rss_bytes null, never 0",
        "undeclared pid is UNDECLARED, not invented from the largest RSS neighbour",
        "missing queue/detached/context is UNOBSERVABLE with depth/n_jobs null, not an empty healthy queue",
        "telemetry classed SELF_MEASURED_DIRTY; authorizes restart_decision_only; no hardware field",
    ]


def negative_findings() -> list[str]:
    landed = restart_supervisor_presence()
    findings = [
        "this sidecar did not start a resident and did not take a GPU lease",
        "queue depth is UNOBSERVABLE when receipts/future/HCLI_FUTURE_WORKUNITS.json is not on disk; HEAD is not the live queue",
        "context growth is UNOBSERVABLE unless a context_path is declared; no conversation log is guessed",
        "PID-level GPU working-set attribution still requires a protected lease this sidecar does not hold",
        "a flat high RSS is HEALTHY (stable residency is not a leak); restarting it would thrash",
        "orchestration BINDINGS / frontiers catalog were outside this lane's WRITE list, so this module is not glued into invoke() by this receipt",
    ]
    if not landed["landed"]:
        findings.append(
            "tools/future/restart_supervisor.py had not landed; restart remains a decision this module authorizes, not an action it performs"
        )
    return findings


def build() -> Path:
    synthetic = prove_synthetic()
    if not synthetic.get("all_passed"):
        raise HealthRefuse(f"synthetic negative controls did not all fire: {synthetic}")
    live_dead = prove_live_dead_pid()
    live_undeclared = sample()
    live_self = sample(pid=os.getpid())
    supervisor = restart_supervisor_presence()
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Observe resident process telemetry over time and turn a monotonic "
            "climb (not a spike) into a checkpoint-and-restart decision."
        ),
        "evidence_class": EVIDENCE_CLASS,
        "telemetry_evidence_class": TELEMETRY_CLASS,
        "gpu_authority": False,
        "authorizes": AUTHORIZES,
        "claim_boundary": CLAIM_BOUNDARY,
        "vocabulary": {
            "verdicts": list(VERDICTS),
            "presence": list(PRESENCE),
            "directions": list(DIRECTIONS),
            "eras": list(ERAS),
            "odysseys": list(ODYSSEYS),
            "pathological_rule": (
                "CLIMB_NO_PLATEAU on a pathology signal, or resident ABSENT on every "
                "sample in the window. One reading is a moment."
            ),
            "restart_rule": "authorize_restart is true only for PATHOLOGICAL",
        },
        "executed": {
            "sample_undeclared": True,
            "sample_self_pid": True,
            "trend": True,
            "verdict": True,
            "prove_synthetic": True,
            "prove_live_dead_pid": True,
        },
        "live_undeclared": {
            "presence": live_undeclared["resident"]["presence"],
            "rss_bytes": live_undeclared["resident"]["rss_bytes"],
            "reason": live_undeclared["resident"]["reason"],
            "queue_status": live_undeclared["queue"]["status"],
            "queue_depth": live_undeclared["queue"]["depth"],
            "detached_status": live_undeclared["detached"]["status"],
            "uma_pressure_name": live_undeclared["memory"].get("uma_pressure_name"),
            "memory_status": live_undeclared["memory"]["status"],
        },
        "live_self_process": {
            "note": "sidecar python pid, NOT the trial resident; proves RSS readout works",
            "presence": live_self["resident"]["presence"],
            "rss_bytes_is_int": isinstance(live_self["resident"]["rss_bytes"], int),
            "rss_bytes_nonzero": isinstance(live_self["resident"]["rss_bytes"], int)
            and live_self["resident"]["rss_bytes"] > 0,
            "children_status": live_self["children"]["status"],
        },
        "proofs": {
            "synthetic": synthetic,
            "live_dead_pid": live_dead,
        },
        "restart_supervisor": supervisor,
        "is_a_measurement": False,
        "is_hardware_performance_claim": False,
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "resident_callable": {
            "entry_point": "tools.future.resident_health.sample() / trend() / verdict()",
            "workunit": (
                "one CPU_ANALYSIS unit; sample the declared resident, trend the window, "
                "emit HEALTHY|DEGRADED|PATHOLOGICAL; write receipts/future/RESIDENT_HEALTH.json; "
                "never a GPU lease"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.HCLI_SELF.emit-workunits",
            "fails_closed": (
                "HealthRefuse on n<2; ABSENT rss_bytes is null not 0; missing queue/detached/"
                "context is UNOBSERVABLE not empty; PATHOLOGICAL requires CLIMB_NO_PLATEAU; "
                "authorize_restart is false for a spike"
            ),
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--pid", type=int, default=None)
    a = ap.parse_args()
    if a.sample:
        print(json.dumps(sample(pid=a.pid), indent=1, sort_keys=True))
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
