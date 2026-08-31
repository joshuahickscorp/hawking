"""CONTAMINATION_SCIENCE — machine-state record, A/B stats, promotion gate.

A DIAGNOSTIC_RELATIVE number taken on a busy machine has burned this program
before. This sidecar consolidates the existing quiescence / cleanliness /
benchmark-boundary ideas into one deterministic record and a gate that
mechanically refuses to let that number be used as PROTECTED_ABSOLUTE
promotion evidence.

This module measures MACHINE STATE only. It does not run a benchmark, does
not take a GPU lease, and never writes a throughput or latency field.

    python3 tools/future/contamination.py --snapshot
    python3 tools/future/contamination.py --build
    python3 tools/future/contamination.py --selftest
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import hashlib
import inspect
import json
import os
import random
import re
import statistics
import struct
import subprocess
from collections.abc import Iterable, Mapping
from typing import Any

from tools.future._common import write_receipt, load_json, REPO
from tools.future import status_causality as sc

RECEIPT = "CONTAMINATION_SCIENCE.json"
SCHEMA = "hawking.future.contamination.v1"

FIVE_RECORDED_FIELDS: tuple[str, ...] = getattr(
    sc,
    "FIVE_RECORDED_FIELDS",
    (
        "probe_performed",
        "direct_observation",
        "interpretation",
        "confidence",
        "alternatives",
    ),
)


def _bind_emit() -> None:
    """Consumer-side emit. Sibling owns the routine; this checkout may predate it."""
    if hasattr(sc, "emit"):
        return

    def emit(
        status: str,
        *,
        probe_performed: str = "",
        direct_observation: Any = "",
        interpretation: str = "",
        probe_kind: str = "",
        claim_kind: str | None = None,
        falsifier: str = "",
        source: str = "",
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "status": status,
            "probe_performed": probe_performed,
            "direct_observation": direct_observation,
            "interpretation": interpretation or status,
            "probe_kind": probe_kind,
            "use_catalog": False,
            "source": source or "<emit>",
        }
        if claim_kind:
            row["claim_kind"] = claim_kind
        if falsifier:
            row["falsifier"] = falsifier
        out = sc.challenge(row)
        out["entry"] = "emit"
        return out

    sc.emit = emit  # type: ignore[attr-defined]


_bind_emit()


def records_five_fields(node: Any) -> bool:
    fn = getattr(sc, "records_five_fields", None)
    if callable(fn):
        return bool(fn(node))
    if not isinstance(node, dict):
        return False
    if not all(k in node for k in FIVE_RECORDED_FIELDS):
        return False
    if not str(node.get("probe_performed") or "").strip():
        return False
    if node.get("direct_observation") in (None, "", [], {}):
        return False
    if not str(node.get("interpretation") or "").strip():
        return False
    conf = node.get("confidence")
    if not isinstance(conf, dict):
        return False
    if not {"would_raise", "would_lower", "level", "about"} <= set(conf):
        return False
    alts = node.get("alternatives")
    return isinstance(alts, list) and bool(alts)


def record_contamination_causality(
    result: dict[str, Any],
    *,
    probe_performed: str = "",
    direct_observation: Any = "",
    interpretation: str | None = None,
    probe_kind: str = "",
    claim_kind: str | None = None,
) -> dict[str, Any]:
    """Stamp the five causality fields. Does not change contamination_class.

    An unsupplied observation is UNTESTED, never a restatement of HEAVY/QUIESCENT.
    """
    class_before = result.get("contamination_class")
    status = str(result.get("contamination_class") or "")
    unsupplied = direct_observation in (None, "", [], {})
    rec = sc.emit(
        status,
        probe_performed=str(probe_performed or ""),
        direct_observation="" if unsupplied else direct_observation,
        interpretation=interpretation if interpretation is not None else status,
        probe_kind="" if unsupplied else probe_kind,
        claim_kind=None if unsupplied else claim_kind,
        source="tools/future/contamination.py::build",
    )
    for key in FIVE_RECORDED_FIELDS:
        result[key] = rec[key]
    result["causality_verdict"] = rec["verdict"]
    result["falsifier"] = rec.get("falsifier")
    if rec.get("probe_kind"):
        result["probe_kind"] = rec["probe_kind"]
    if rec.get("claim_kind") is not None:
        result["claim_kind"] = rec["claim_kind"]
    if result.get("contamination_class") != class_before:
        raise RuntimeError("status_causality.emit mutated contamination_class")
    return rec

# Recovered thresholds. Do not silently retune: QUIESCENT is earned, not guessed.
# tools/accelerator/bench.py machine_quiescence
QUIET_CPU_PCT = 20.0
QUIET_RSS_GIB = 2.0
# tools/verify/perfgate.py sample_system
HEAVY_CPU_PCT = 400.0
# quiescence instrument recorded a 17.6 GiB MLX neighbour; 8 GiB is a large resident
HEAVY_RSS_GIB = 8.0
# tools/agentos/machine_state.py clean_box_ok
LIGHT_LOAD_FRACTION = 0.5
# tools/verify/perfgate.py --paired
MIN_PAIRS = 7
# device occupancy (ioreg), not a kernel timing
HEAVY_GPU_UTIL_PCT = 20.0

CONTAMINATION_CLASSES = ("QUIESCENT", "LIGHT", "HEAVY", "UNKNOWN")
MEASUREMENT_CLASSES = ("PROTECTED_ABSOLUTE", "DIAGNOSTIC_RELATIVE", "STATIC_ONLY")

# Dimensionless synthetic pairs for the receipt self-test. Not hardware.
SYNTHETIC_PAIRS = (
    (10.0, 10.4),
    (10.1, 10.3),
    (9.8, 10.5),
    (10.2, 10.2),
    (10.0, 10.6),
    (9.9, 10.4),
    (10.3, 10.7),
)

PRESSURE_NAMES = {0: "normal", 1: "warn", 2: "urgent", 3: "critical"}

# macOS memorystatus.h: 0 normal, 1 warn, 2 urgent, 3 critical.
PRESSURE_HEAVY_MIN = 2


class PromotionRefused(ValueError):
    """This record cannot be used as PROTECTED_ABSOLUTE promotion evidence."""


# ---------------------------------------------------------------------------
# probes — every failure is a recorded reason, never a quiet default
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: float = 8.0) -> tuple[int | None, str, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as exc:  # probe failure is evidence
        return None, "", f"{type(exc).__name__}: {exc}"


def _sysctl(key: str) -> str | None:
    rc, out, err = _run(["sysctl", "-n", key], timeout=3)
    if rc != 0:
        return None
    text = (out or "").strip()
    return text or None


def probe_load() -> dict[str, Any]:
    try:
        load = os.getloadavg()
        ncpu = os.cpu_count()
        return {
            "status": "OK",
            "load_1m": float(load[0]),
            "load_5m": float(load[1]),
            "load_15m": float(load[2]),
            "ncpu": int(ncpu) if ncpu else None,
        }
    except Exception as exc:
        return {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}


def probe_memory() -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "FAILED",
        "reason": "no memory probe succeeded",
        "pressure_level": None,
        "pressure_name": "UNKNOWN",
        "pages": {},
        "bytes": {},
    }
    level_s = _sysctl("kern.memorystatus_vm_pressure_level")
    if level_s is not None and level_s.lstrip("-").isdigit():
        level = int(level_s)
        out["pressure_level"] = level
        out["pressure_name"] = PRESSURE_NAMES.get(level, f"level_{level}")
    page = 16384
    ps = _sysctl("hw.pagesize")
    if ps and ps.isdigit():
        page = int(ps)
    rc, vm, err = _run(["vm_stat"], timeout=3)
    pages: dict[str, int] = {}
    if rc == 0:
        for line in vm.splitlines():
            m = re.match(r"([^:]+):\s+(\d+)", line)
            if m:
                pages[m.group(1).strip()] = int(m.group(2).rstrip("."))
        out["pages"] = {k: pages[k] for k in sorted(pages)}
        out["bytes"] = {
            "free": pages.get("Pages free", 0) * page,
            "active": pages.get("Pages active", 0) * page,
            "inactive": pages.get("Pages inactive", 0) * page,
            "wired": pages.get("Pages wired down", 0) * page,
            "compressor": pages.get("Pages occupied by compressor", 0) * page
            or pages.get("Pages used by compressor", 0) * page,
        }
    if out["pressure_level"] is not None or pages:
        out["status"] = "OK"
        out.pop("reason", None)
    else:
        out["reason"] = err or "vm_stat and memorystatus both failed"
    memsize = _sysctl("hw.memsize")
    out["ram_total_bytes"] = int(memsize) if memsize and memsize.isdigit() else None
    return out


def probe_thermal() -> dict[str, Any]:
    """Thermal is optional. UNKNOWN unless the platform exposes it without sudo."""
    keys = (
        "machdep.xcpm.cpu_thermal_level",
        "machdep.cpu.thermal.level",
        "machdep.thermal.level",
    )
    found: dict[str, str] = {}
    for key in keys:
        val = _sysctl(key)
        if val is not None:
            found[key] = val
    if found:
        return {"status": "OK", "method": "sysctl", "readings": found}
    rc, _out, err = _run(["powermetrics", "--samplers", "thermal", "-n", "1", "-i", "1"], timeout=3)
    if rc == 0:
        return {"status": "OK", "method": "powermetrics", "note": "powermetrics succeeded without sudo"}
    return {
        "status": "UNKNOWN",
        "method": None,
        "reason": (
            "platform does not expose thermal without sudo "
            f"(powermetrics: {(err or 'unavailable').strip()[:80] or 'needs root'}; "
            "no sysctl thermal keys)"
        ),
    }


def probe_gpu_occupancy() -> dict[str, Any]:
    """Device occupancy from IOKit. Occupancy is machine state, not a kernel timing."""
    rc, out, err = _run(["ioreg", "-c", "IOGPU", "-d", "1", "-r"], timeout=10)
    if rc is None or rc != 0 or not out:
        return {
            "status": "FAILED",
            "reason": (err or f"ioreg rc={rc}")[:200],
            "device_utilization_pct": None,
        }
    stats_m = re.search(r'"PerformanceStatistics" = \{([^}]*)\}', out)
    if not stats_m:
        return {"status": "FAILED", "reason": "ioreg IOGPU had no PerformanceStatistics"}
    stats: dict[str, int] = {}
    for key, val in re.findall(r'"([^"]+)"\s*=\s*(-?\d+)', stats_m.group(1)):
        if "Utilization" in key:
            stats[key] = int(val)
    core_m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', out)
    return {
        "status": "OK",
        "method": "ioreg_IOGPU_PerformanceStatistics",
        "no_gpu_lease": True,
        "device_utilization_pct": stats.get("Device Utilization %"),
        "renderer_utilization_pct": stats.get("Renderer Utilization %"),
        "tiler_utilization_pct": stats.get("Tiler Utilization %"),
        "gpu_core_count": int(core_m.group(1)) if core_m else None,
        "note": (
            "device occupancy percent, not a kernel timing and not a throughput claim; "
            "PID-level GPU attribution is unavailable without a protected lease"
        ),
    }


def _parse_ps(stdout: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines()[1:]:
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[0])
            cpu = float(parts[1])
            rss_kb = int(parts[2])
        except ValueError:
            continue
        rows.append(
            {
                "pid": pid,
                "cpu_pct": cpu,
                "rss_gib": round(rss_kb / (1024 * 1024), 2),
                "state": parts[3],
                "name": parts[4].strip(),
            }
        )
    rows.sort(key=lambda r: (-r["rss_gib"], -r["cpu_pct"], r["pid"]))
    return rows


def _libproc_processes() -> list[dict[str, Any]]:
    """macOS libproc fallback when `ps` is blocked. RSS is real; %cpu is not."""
    import ctypes
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("c") or "/usr/lib/libc.dylib")
    PROC_ALL_PIDS = 1
    PROC_PIDTASKINFO = 4
    libc.proc_listpids.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]
    libc.proc_listpids.restype = ctypes.c_int
    libc.proc_pidinfo.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    libc.proc_pidinfo.restype = ctypes.c_int
    libc.proc_name.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    libc.proc_name.restype = ctypes.c_int

    nbytes = libc.proc_listpids(PROC_ALL_PIDS, 0, None, 0)
    if nbytes <= 0:
        return []
    buf = (ctypes.c_int * (nbytes // 4 + 16))()
    got = libc.proc_listpids(PROC_ALL_PIDS, 0, buf, ctypes.sizeof(buf))
    pids = [buf[i] for i in range(max(0, got // 4)) if buf[i]]
    namebuf = ctypes.create_string_buffer(256)
    info = ctypes.create_string_buffer(96)
    rows: list[dict[str, Any]] = []
    for pid in pids:
        nlen = libc.proc_name(pid, namebuf, 256)
        name = namebuf.value.decode("utf-8", "replace") if nlen >= 0 else ""
        sz = libc.proc_pidinfo(pid, PROC_PIDTASKINFO, 0, info, 96)
        if sz < 16:
            continue
        _virt, rss_bytes = struct.unpack_from("<QQ", info.raw, 0)
        rows.append(
            {
                "pid": int(pid),
                "cpu_pct": None,
                "rss_gib": round(rss_bytes / (1024**3), 2),
                "state": None,
                "name": name or f"pid:{pid}",
            }
        )
    rows.sort(key=lambda r: (-r["rss_gib"], r["pid"]))
    return rows


def probe_processes() -> dict[str, Any]:
    """Enumerate every process. There is deliberately no names= parameter.

    The historical defect (receipts/headless/ACCELERATOR_QUIESCENCE_INSTRUMENT.json)
    was a name filter that reported QUIET while fileproviderd and a 17.6 GiB MLX
    run were on the machine. An instrument whose blind spot is configurable is
    one somebody will configure blind.
    """
    rc, out, err = _run(["ps", "-Ao", "pid,pcpu,rss,state,comm"], timeout=8)
    if rc == 0 and out.strip():
        rows = _parse_ps(out)
        return {
            "status": "OK",
            "method": "ps_enumerate",
            "cpu_pct_available": True,
            "no_name_filter": True,
            "n_enumerated": len(rows),
            "all": rows,
        }
    ps_reason = err or f"ps rc={rc}"
    try:
        rows = _libproc_processes()
    except Exception as exc:
        return {
            "status": "FAILED",
            "method": None,
            "cpu_pct_available": False,
            "no_name_filter": True,
            "reason": f"ps failed ({ps_reason}); libproc failed ({type(exc).__name__}: {exc})",
            "n_enumerated": 0,
            "all": [],
        }
    if not rows:
        return {
            "status": "FAILED",
            "method": "libproc_enumerate_rss",
            "cpu_pct_available": False,
            "no_name_filter": True,
            "reason": f"ps failed ({ps_reason}); libproc returned no processes",
            "n_enumerated": 0,
            "all": [],
        }
    return {
        "status": "PARTIAL",
        "method": "libproc_enumerate_rss",
        "cpu_pct_available": False,
        "no_name_filter": True,
        "reason": (
            f"ps failed ({ps_reason}); RSS enumerated via libproc; "
            "%cpu is unavailable so QUIESCENT cannot be earned"
        ),
        "n_enumerated": len(rows),
        "all": rows,
    }


def machine_identity() -> dict[str, Any]:
    fields = {
        "hw.model": _sysctl("hw.model"),
        "machdep.cpu.brand_string": _sysctl("machdep.cpu.brand_string"),
        "hw.memsize": _sysctl("hw.memsize"),
        "hw.ncpu": _sysctl("hw.ncpu"),
        "hw.pagesize": _sysctl("hw.pagesize"),
        "sysctl.proc_translated": _sysctl("sysctl.proc_translated"),
    }
    blob = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return {
        "hash": hashlib.sha256(blob).hexdigest(),
        "fields": fields,
        "note": "identity of the host, not a performance number; serial numbers are not stored",
    }


def _over_threshold(proc: Mapping[str, Any]) -> bool:
    cpu = proc.get("cpu_pct")
    rss = proc.get("rss_gib") or 0.0
    if isinstance(cpu, (int, float)) and cpu >= QUIET_CPU_PCT:
        return True
    if isinstance(rss, (int, float)) and rss >= QUIET_RSS_GIB:
        return True
    return False


def snapshot(*, benchmark_ordinal: int | None = None, probes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Deterministic-shape record of machine state at measurement time.

    `benchmark_ordinal` is the caller's index within a run; None means this
    snapshot is not attached to a benchmark. Everything here is obtainable
    without sudo and without a GPU lease.
    """
    probed = dict(probes) if probes is not None else {
        "processes": probe_processes(),
        "memory": probe_memory(),
        "load": probe_load(),
        "gpu_occupancy": probe_gpu_occupancy(),
        "thermal": probe_thermal(),
    }
    proc_probe = probed.get("processes") or {}
    all_procs = list(proc_probe.get("all") or [])
    contenders = [p for p in all_procs if _over_threshold(p)]
    contenders.sort(key=lambda r: (-(r.get("rss_gib") or 0), -(r.get("cpu_pct") or 0), r.get("pid") or 0))
    gpu_status = proc_probe.get("status") or "FAILED"
    gpu_processes = {
        "status": gpu_status,
        "method": proc_probe.get("method"),
        "cpu_pct_available": bool(proc_probe.get("cpu_pct_available")),
        "no_name_filter": True,
        "attribution": (
            "enumerated by CPU% and/or RSS; PID-level GPU occupancy is not available "
            "without a protected lease. On Apple Silicon RSS is the Metal working-set "
            "dimension (ACCELERATOR_QUIESCENCE_INSTRUMENT)."
        ),
        "thresholds": {"cpu_pct": QUIET_CPU_PCT, "rss_gib": QUIET_RSS_GIB},
        "reason": proc_probe.get("reason"),
        "n_enumerated": proc_probe.get("n_enumerated", 0),
        "n_over_threshold": len(contenders),
        "processes": [
            {"name": p.get("name"), "pid": p.get("pid"), "cpu_pct": p.get("cpu_pct")}
            for p in contenders
        ],
    }
    resident = None
    if contenders:
        top = max(contenders, key=lambda p: p.get("rss_gib") or 0)
        if (top.get("rss_gib") or 0) >= QUIET_RSS_GIB:
            resident = {
                "name": top.get("name"),
                "pid": top.get("pid"),
                "rss_gib": top.get("rss_gib"),
                "cpu_pct": top.get("cpu_pct"),
                "note": "largest RSS neighbour over the quiet threshold; not a model-name guess",
            }
    thermal = probed.get("thermal") or {}
    thermal_state: Any = "UNKNOWN"
    if thermal.get("status") == "OK":
        thermal_state = {k: thermal[k] for k in thermal if k != "status"}
        thermal_state["status"] = "OK"
    return {
        "benchmark_ordinal": benchmark_ordinal,
        "machine_identity": machine_identity() if probes is None else probes.get(
            "machine_identity", machine_identity()
        ),
        "probes": {
            "processes": {k: v for k, v in proc_probe.items() if k != "all"},
            "memory": probed.get("memory"),
            "load": probed.get("load"),
            "gpu_occupancy": probed.get("gpu_occupancy"),
            "thermal": thermal,
        },
        "gpu_processes": gpu_processes,
        "competing_workloads": [
            {
                "name": p.get("name"),
                "pid": p.get("pid"),
                "cpu_pct": p.get("cpu_pct"),
                "rss_gib": p.get("rss_gib"),
            }
            for p in contenders
        ],
        "resident_local_model": resident,
        "memory_pressure": probed.get("memory"),
        "thermal_state": thermal_state,
    }


def _probe_ok(node: Mapping[str, Any] | None, *ok: str) -> bool:
    return isinstance(node, Mapping) and node.get("status") in ok


def classify_contamination(snap: Mapping[str, Any]) -> dict[str, Any]:
    """QUIESCENT / LIGHT / HEAVY / UNKNOWN with the exact evidence that produced it.

    UNKNOWN when a required probe failed — never optimistic QUIESCENT.
    HEAVY evidence still wins if we can prove the machine is dirty from a
    probe that did succeed (a 19 GiB neighbour is HEAVY even if `ps` is blocked).
    """
    probes = snap.get("probes") if isinstance(snap.get("probes"), Mapping) else {}
    proc = probes.get("processes") if isinstance(probes.get("processes"), Mapping) else {}
    load = probes.get("load") if isinstance(probes.get("load"), Mapping) else {}
    mem = probes.get("memory") if isinstance(probes.get("memory"), Mapping) else snap.get("memory_pressure")
    mem = mem if isinstance(mem, Mapping) else {}
    gpu = probes.get("gpu_occupancy") if isinstance(probes.get("gpu_occupancy"), Mapping) else {}

    contenders = list(snap.get("competing_workloads") or [])
    if not contenders and proc.get("all"):
        contenders = [p for p in proc["all"] if _over_threshold(p)]

    evidence: list[dict[str, Any]] = []
    heavy = False
    light = False

    def note(kind: str, probe: str, finding: str, contributes: str) -> None:
        evidence.append(
            {"kind": kind, "probe": probe, "finding": finding, "contributes": contributes}
        )

    cpu_ok = bool(proc.get("cpu_pct_available")) and proc.get("status") == "OK"
    rss_ok = proc.get("status") in {"OK", "PARTIAL"} and int(proc.get("n_enumerated") or 0) > 0
    if proc.get("status") not in {"OK", "PARTIAL"}:
        note("probe_failed", "processes", str(proc.get("reason") or "process enumeration failed"), "blocks QUIESCENT")
    elif not cpu_ok:
        note("probe_partial", "processes", str(proc.get("reason") or "%cpu unavailable"), "blocks QUIESCENT")

    if not _probe_ok(load, "OK"):
        note("probe_failed", "load", str(load.get("reason") or "loadavg failed"), "blocks QUIESCENT")
    if not _probe_ok(mem, "OK"):
        note("probe_failed", "memory", str(mem.get("reason") or "memory probe failed"), "blocks QUIESCENT")

    max_rss = max((float(p.get("rss_gib") or 0) for p in contenders), default=0.0)
    max_cpu = max(
        (float(p["cpu_pct"]) for p in contenders if isinstance(p.get("cpu_pct"), (int, float))),
        default=0.0,
    )
    if rss_ok and max_rss >= HEAVY_RSS_GIB:
        heavy = True
        note("heavy", "rss", f"max_rss_gib={max_rss} >= {HEAVY_RSS_GIB}", "HEAVY")
    if cpu_ok and max_cpu >= HEAVY_CPU_PCT:
        heavy = True
        note("heavy", "cpu", f"max_cpu_pct={max_cpu} >= {HEAVY_CPU_PCT}", "HEAVY")

    ncpu = load.get("ncpu") or 0
    load1 = load.get("load_1m")
    if isinstance(load1, (int, float)) and ncpu:
        if load1 >= float(ncpu):
            heavy = True
            note("heavy", "load", f"load_1m={load1} >= ncpu={ncpu}", "HEAVY")
        elif load1 > LIGHT_LOAD_FRACTION * float(ncpu):
            light = True
            note("light", "load", f"load_1m={load1} > {LIGHT_LOAD_FRACTION}*ncpu={ncpu}", "LIGHT")

    pressure = mem.get("pressure_level")
    if isinstance(pressure, int):
        if pressure >= PRESSURE_HEAVY_MIN:
            heavy = True
            note("heavy", "memory", f"pressure_level={pressure} ({mem.get('pressure_name')})", "HEAVY")
        elif pressure >= 1:
            light = True
            note("light", "memory", f"pressure_level={pressure} ({mem.get('pressure_name')})", "LIGHT")

    util = gpu.get("device_utilization_pct")
    if gpu.get("status") == "OK" and isinstance(util, (int, float)):
        if util >= HEAVY_GPU_UTIL_PCT:
            heavy = True
            note("heavy", "gpu_occupancy", f"device_utilization_pct={util} >= {HEAVY_GPU_UTIL_PCT}", "HEAVY")
        elif util > 0:
            light = True
            note("light", "gpu_occupancy", f"device_utilization_pct={util} > 0", "LIGHT")
    elif gpu.get("status") not in {"OK"}:
        note(
            "probe_optional_failed",
            "gpu_occupancy",
            str(gpu.get("reason") or "gpu occupancy unknown"),
            "does not by itself block QUIESCENT",
        )

    if contenders and not heavy:
        light = True
        note(
            "light",
            "processes",
            f"n_over_threshold={len(contenders)} max_rss_gib={max_rss} max_cpu_pct={max_cpu}",
            "LIGHT",
        )

    required_failed = (not cpu_ok) or (not _probe_ok(load, "OK")) or (not _probe_ok(mem, "OK"))

    if heavy:
        klass = "HEAVY"
        why = "at least one HEAVY probe fired"
    elif light:
        klass = "LIGHT"
        why = "contenders or mild pressure without a HEAVY trigger"
    elif required_failed:
        klass = "UNKNOWN"
        why = "a required probe failed; UNKNOWN, never optimistic QUIESCENT"
    else:
        klass = "QUIESCENT"
        why = "required probes succeeded and no LIGHT/HEAVY evidence"
        note("quiescent", "all_required", why, "QUIESCENT")

    evidence.sort(key=lambda e: (e["kind"], e["probe"], e["finding"]))
    return {
        "contamination_class": klass,
        "contamination_reason": why,
        "contamination_evidence": evidence,
        "required_probes": {
            "processes_cpu_pct": cpu_ok,
            "loadavg": _probe_ok(load, "OK"),
            "memory": _probe_ok(mem, "OK"),
        },
        "rule": (
            "UNKNOWN when a required probe failed — never optimistic QUIESCENT. "
            "HEAVY evidence from a probe that succeeded still classifies HEAVY."
        ),
    }


# ---------------------------------------------------------------------------
# paired / interleaved A/B statistics — never a bare mean
# ---------------------------------------------------------------------------

def _pairs(samples: Iterable[Any]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for item in samples:
        if isinstance(item, Mapping) and "a" in item and "b" in item:
            a, b = item["a"], item["b"]
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            a, b = item[0], item[1]
        else:
            raise ValueError(f"pair must be (a,b) or {{a,b}}, got {type(item).__name__}")
        out.append((float(a), float(b)))
    return out


def _iqr(sorted_vals: list[float]) -> tuple[float, float, float]:
    # Same index rule as tools/accelerator/bench.py time_arm.
    q1 = sorted_vals[len(sorted_vals) // 4]
    q3 = sorted_vals[(3 * len(sorted_vals)) // 4]
    return q1, q3, q3 - q1


def _bootstrap_median_ci(values: list[float], *, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> list[float]:
    rng = random.Random(seed)
    nobs = len(values)
    meds = []
    for _ in range(n):
        sample = [values[rng.randrange(nobs)] for _ in range(nobs)]
        meds.append(statistics.median(sample))
    meds.sort()
    lo = meds[min(n - 1, int((alpha / 2) * n))]
    hi = meds[min(n - 1, int((1 - alpha / 2) * n))]
    return [lo, hi]


def paired_ab_stats(
    samples: Iterable[Any],
    *,
    min_pairs: int = MIN_PAIRS,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Median pairwise B/A ratio, IQR, bootstrap CI, sample count, sufficiency.

    Caller interleaves A/B (ABAB) so both arms share thermal and page-cache
    state; this function consumes already-paired samples. Never reports a mean.
    Ratios are dimensionless. This sidecar does not source the samples.
    """
    raw = _pairs(samples)
    kept: list[float] = []
    dropped = 0
    for a, b in raw:
        if a == 0 or a != a or b != b or a == float("inf") or b == float("inf") or a < 0:
            dropped += 1
            continue
        kept.append(b / a)
    kept.sort()
    n_kept = len(kept)
    if n_kept == 0:
        return {
            "n_pairs": len(raw),
            "n_kept": 0,
            "n_dropped": dropped,
            "min_pairs": min_pairs,
            "median_ratio": None,
            "ratio_q1": None,
            "ratio_q3": None,
            "ratio_iqr": None,
            "bootstrap_ci95": None,
            "sufficient_for_decision": False,
            "reason": "no valid pairs (a must be finite and > 0); never a mean",
            "summary": "median of pairwise B/A ratios; never a mean",
            "pair_protocol": "caller interleaves A/B (ABAB); this function consumes paired samples",
        }
    q1, q3, iqr = _iqr(kept)
    median = statistics.median(kept)
    sufficient = n_kept >= min_pairs
    if sufficient:
        reason = f"n_kept={n_kept} >= min_pairs={min_pairs}; median of pairwise ratios with IQR"
    else:
        reason = f"n_kept={n_kept} < min_pairs={min_pairs}; insufficient for a decision"
    return {
        "n_pairs": len(raw),
        "n_kept": n_kept,
        "n_dropped": dropped,
        "min_pairs": min_pairs,
        "median_ratio": median,
        "ratio_q1": q1,
        "ratio_q3": q3,
        "ratio_iqr": iqr,
        "bootstrap_ci95": _bootstrap_median_ci(kept, seed=bootstrap_seed),
        "sufficient_for_decision": sufficient,
        "reason": reason,
        "summary": "median of pairwise B/A ratios; never a mean",
        "pair_protocol": "caller interleaves A/B (ABAB); this function consumes paired samples",
    }


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------

def assert_promotable(record: Mapping[str, Any]) -> None:
    """RAISE unless this is a well-formed QUIESCENT PROTECTED_ABSOLUTE record.

    A DIAGNOSTIC_RELATIVE measurement offered as PROTECTED_ABSOLUTE promotion
    evidence is refused. LIGHT / HEAVY / UNKNOWN are refused. Insufficient
    paired samples are refused. STATIC_ONLY is refused — this sidecar does
    not produce promotion evidence.
    """
    if not isinstance(record, Mapping):
        raise PromotionRefused("record is not a mapping")
    mclass = record.get("measurement_class")
    offered = record.get("offered_as")
    if mclass == "DIAGNOSTIC_RELATIVE" or offered == "PROTECTED_ABSOLUTE" and mclass != "PROTECTED_ABSOLUTE":
        raise PromotionRefused(
            "DIAGNOSTIC_RELATIVE cannot be used as PROTECTED_ABSOLUTE promotion evidence "
            f"(measurement_class={mclass!r}, offered_as={offered!r})"
        )
    if mclass != "PROTECTED_ABSOLUTE":
        raise PromotionRefused(
            f"measurement_class {mclass!r} is not PROTECTED_ABSOLUTE; "
            "only a protected-lease measurement can promote"
        )
    cclass = record.get("contamination_class")
    if cclass != "QUIESCENT":
        raise PromotionRefused(
            f"contamination_class {cclass!r} is not QUIESCENT; "
            "LIGHT/HEAVY/UNKNOWN cannot promote"
        )
    stats = record.get("ab_stats")
    if not isinstance(stats, Mapping) or not stats.get("sufficient_for_decision"):
        why = stats.get("reason") if isinstance(stats, Mapping) else "ab_stats missing"
        raise PromotionRefused(f"sample count insufficient for a decision ({why})")


def _must_raise(record: Mapping[str, Any], fragment: str) -> dict[str, Any]:
    try:
        assert_promotable(record)
    except PromotionRefused as exc:
        msg = str(exc)
        if fragment not in msg:
            raise AssertionError(f"gate raised but message {msg!r} lacked {fragment!r}") from exc
        return {"fired": True, "message": msg}
    raise AssertionError(f"gate did not refuse {fragment!r}: {dict(record)}")


def selftest() -> dict[str, Any]:
    """Watch the gate fail. A guard nobody has watched fail is not a guard."""
    ok_stats = paired_ab_stats(SYNTHETIC_PAIRS)
    if "mean" in ok_stats or "average" in ok_stats:
        raise AssertionError("paired_ab_stats reported a mean")
    if not ok_stats["sufficient_for_decision"]:
        raise AssertionError("synthetic pairs should be sufficient")
    good = {
        "measurement_class": "PROTECTED_ABSOLUTE",
        "contamination_class": "QUIESCENT",
        "ab_stats": ok_stats,
    }
    assert_promotable(good)
    diagnostic = _must_raise(
        {
            "measurement_class": "DIAGNOSTIC_RELATIVE",
            "offered_as": "PROTECTED_ABSOLUTE",
            "contamination_class": "QUIESCENT",
            "ab_stats": ok_stats,
        },
        "DIAGNOSTIC_RELATIVE",
    )
    heavy = _must_raise(
        {
            "measurement_class": "PROTECTED_ABSOLUTE",
            "contamination_class": "HEAVY",
            "ab_stats": ok_stats,
        },
        "HEAVY",
    )
    unknown = _must_raise(
        {
            "measurement_class": "PROTECTED_ABSOLUTE",
            "contamination_class": "UNKNOWN",
            "ab_stats": ok_stats,
        },
        "UNKNOWN",
    )
    short = _must_raise(
        {
            "measurement_class": "PROTECTED_ABSOLUTE",
            "contamination_class": "QUIESCENT",
            "ab_stats": paired_ab_stats(SYNTHETIC_PAIRS[:3]),
        },
        "insufficient",
    )
    params = set(inspect.signature(probe_processes).parameters)
    if params:
        raise AssertionError(f"probe_processes grew a filter parameter: {params}")
    return {
        "quiescent_protected_passes": True,
        "diagnostic_relative_raises": diagnostic,
        "heavy_raises": heavy,
        "unknown_raises": unknown,
        "insufficient_samples_raises": short,
        "no_name_filter_on_probe_processes": True,
        "paired_ab_stats_has_no_mean": True,
        "synthetic_ab_stats": ok_stats,
    }


def _recovered() -> list[dict[str, Any]]:
    return [
        {
            "path": "tools/odyssey/contamination.py",
            "on_disk_in_this_worktree": (REPO / "tools/odyssey/contamination.py").is_file(),
            "role": (
                "Train/eval overlap barrier (exact hash + Jaccard shingles). Different "
                "sense of 'contamination' than machine-state. Not a promotion gate."
            ),
            "adequate_for_this_lane": False,
        },
        {
            "path": "tools/odyssey/gpu_cleanliness.py",
            "on_disk_in_this_worktree": (REPO / "tools/odyssey/gpu_cleanliness.py").is_file(),
            "role": (
                "G013 pause/resume of PAUSABLE campaign I/O; STANDING daemons are "
                "declared and never paused (no forged speedups). Mechanism, not a "
                "DIAGNOSTIC_RELATIVE/PROTECTED_ABSOLUTE gate."
            ),
            "adequate_for_this_lane": False,
        },
        {
            "path": "tools/accelerator/bench.py",
            "on_disk_in_this_worktree": (REPO / "tools/accelerator/bench.py").is_file(),
            "role": (
                "machine_quiescence enumerates by CPU%/RSS with no names= parameter; "
                "failed ps => quiet=None; IQR reliability gate; bench_block derives "
                "QUIESCED/CONTENDED/UNKNOWN. Codex surface; sidecar reuses the ideas."
            ),
            "adequate_for_this_lane": False,
        },
        {
            "path": "tools/verify/perfgate.py",
            "on_disk_in_this_worktree": (REPO / "tools/verify/perfgate.py").is_file(),
            "role": (
                "sample_system (loadavg, vm_stat, heavy process >400% CPU); paired "
                "ABAB with median + sign test; 'never mean alone'; contamination_note "
                "that absolute numbers on this machine are contaminated."
            ),
            "adequate_for_this_lane": False,
        },
        {
            "path": "tools/odyssey_patient_runner.py",
            "on_disk_in_this_worktree": (REPO / "tools/odyssey_patient_runner.py").is_file(),
            "role": (
                "maybe_machine_note() optionally imports tools.agentos.machine_state; "
                "stamps a contamination flag on SPECIMEN receipts. Inline, not shared."
            ),
            "adequate_for_this_lane": False,
        },
        {
            "path": "tools/agentos/machine_state.py",
            "on_disk_in_this_worktree": (REPO / "tools/agentos/machine_state.py").is_file(),
            "role": "snapshot + clean_box_ok (live lanes, disk, load). No promotion gate.",
            "adequate_for_this_lane": False,
        },
        {
            "path": "hcli/agentos/benchmark_boundary.py",
            "on_disk_in_this_worktree": (REPO / "hcli/agentos/benchmark_boundary.py").is_file(),
            "role": (
                "QUALIFIED_PROTECTED vs DIAGNOSTIC_CONTAMINATED from before/after "
                "machine_quiescence. Vocabulary is not DIAGNOSTIC_RELATIVE / "
                "PROTECTED_ABSOLUTE; default NOT_FOR_PROMOTION."
            ),
            "adequate_for_this_lane": False,
        },
        {
            "path": "hcli/agentos/protected_accelerator_benchmark.py",
            "on_disk_in_this_worktree": (
                REPO / "hcli/agentos/protected_accelerator_benchmark.py"
            ).is_file(),
            "role": "Protected resident window that calls classify_window. Codex/HCLI surface.",
            "adequate_for_this_lane": False,
        },
        {
            "path": "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
            "on_disk_in_this_worktree": (
                REPO / "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"
            ).is_file(),
            "role": (
                "Not in this worktree HEAD. Recovered from git object "
                "a4e11a2c: queue_policy.protected_start_requires_machine_quiescence=true, "
                "diagnostic_results_do_not_promote=true, bench.state=UNKNOWN, "
                "measurement_contract lists physical fields that stay null here."
            ),
            "adequate_for_this_lane": False,
        },
    ]


def build() -> Path:
    snap = snapshot(benchmark_ordinal=None)
    klass = classify_contamination(snap)
    gate = selftest()
    frontier_path = REPO / "receipts" / "future" / "CLAUDE_GLOBAL_FRONTIER.json"
    frontier_f011 = None
    if frontier_path.is_file():
        entries = load_json(frontier_path).get("entries") or []
        frontier_f011 = next((e for e in entries if e.get("id") == "F011"), None)
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Deterministic contamination record and the A/B statistics that go with it, "
            "plus the gate that refuses DIAGNOSTIC_RELATIVE as PROTECTED_ABSOLUTE "
            "promotion evidence."
        ),
        "measurement_class": "STATIC_ONLY",
        "vocabulary": {
            "DIAGNOSTIC_RELATIVE": "contaminated A/B on a busy machine. Guides. Never promotes.",
            "PROTECTED_ABSOLUTE": "measurement taken under a real protected GPU lease. Decides.",
            "STATIC_ONLY": "this sidecar. No GPU. Bench state UNKNOWN. Cannot promote.",
            "contamination_classes": list(CONTAMINATION_CLASSES),
            "eras": ["I Genesis of the Laboratory", "II Compounding Civilization",
                     "III Autonomous Science Civilization", "IV Synthetic Machine Civilization",
                     "V Released Hawking Civilization"],
            "odysseys": ["I WHAT IS TRUE?", "II WHAT DID HAWKING ALREADY LEARN?",
                         "III WHERE IS HAWKING WRONG?"],
        },
        "snapshot": snap,
        "contamination_class": klass["contamination_class"],
        "contamination_reason": klass["contamination_reason"],
        "contamination_evidence": klass["contamination_evidence"],
        "required_probes": klass["required_probes"],
        "classification_rule": klass["rule"],
        "ab_stats_contract": {
            "consumes": "already-paired A/B samples (caller interleaves ABAB)",
            "reports": ["median_ratio", "ratio_iqr", "bootstrap_ci95", "n_kept",
                        "sufficient_for_decision"],
            "never_reports": ["mean", "average"],
            "min_pairs": MIN_PAIRS,
            "synthetic_selftest": gate["synthetic_ab_stats"],
            "note": (
                "synthetic_selftest pairs are dimensionless fixtures, not a benchmark. "
                "This sidecar never sources paired samples from hardware."
            ),
        },
        "gate": {
            "function": "assert_promotable(record) -> None",
            "refuses": [
                "measurement_class == DIAGNOSTIC_RELATIVE (offered as PROTECTED_ABSOLUTE)",
                "contamination_class != QUIESCENT",
                "ab_stats.sufficient_for_decision is not True",
                "measurement_class != PROTECTED_ABSOLUTE (including STATIC_ONLY)",
            ],
            "selftest": {k: v for k, v in gate.items() if k != "synthetic_ab_stats"},
        },
        "thresholds": {
            "quiet_cpu_pct": QUIET_CPU_PCT,
            "quiet_rss_gib": QUIET_RSS_GIB,
            "heavy_cpu_pct": HEAVY_CPU_PCT,
            "heavy_rss_gib": HEAVY_RSS_GIB,
            "light_load_fraction_of_ncpu": LIGHT_LOAD_FRACTION,
            "heavy_gpu_device_utilization_pct": HEAVY_GPU_UTIL_PCT,
            "min_pairs": MIN_PAIRS,
            "sources": {
                "quiet_cpu_pct": "tools/accelerator/bench.py QUIET_CPU_PCT",
                "quiet_rss_gib": "tools/accelerator/bench.py QUIET_RSS_GIB",
                "heavy_cpu_pct": "tools/verify/perfgate.py HEAVY_CPU_PCT",
                "min_pairs": "tools/verify/perfgate.py --paired n>=7 kept",
            },
        },
        "integration": {
            "snapshot": "snapshot(*, benchmark_ordinal: int | None = None) -> dict",
            "classify": "classify_contamination(snapshot) -> dict",
            "paired_ab_stats": "paired_ab_stats(pairs, *, min_pairs=7) -> dict",
            "assert_promotable": "assert_promotable(record) -> None  # raises PromotionRefused",
        },
        "frontier_entry": {
            "id": "F011",
            "title": (frontier_f011 or {}).get("title"),
            "classification": (frontier_f011 or {}).get("classification"),
            "present": frontier_f011 is not None,
        },
        "recovered_implementation": _recovered(),
        "gaps_closed": [
            "One sidecar record with machine identity, GPU-relevant process list, "
            "resident neighbour, thermal (UNKNOWN if unexposed), memory pressure, "
            "competing workloads, and benchmark ordinal.",
            "Four-class contamination taxonomy QUIESCENT/LIGHT/HEAVY/UNKNOWN with "
            "the exact evidence, never optimistic QUIESCENT on a failed required probe.",
            "Paired A/B stats that report median ratio + IQR + bootstrap CI + "
            "sufficient_for_decision, and never a mean.",
            "assert_promotable raises on DIAGNOSTIC_RELATIVE offered for promotion, "
            "on non-QUIESCENT class, and on insufficient samples, and does not raise "
            "on a well-formed QUIESCENT PROTECTED_ABSOLUTE record.",
        ],
        "negative_findings": [
            "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json is not in "
            "this worktree HEAD (sparse checkout / not at this commit). Policy recovered "
            "from git object a4e11a2c.",
            "PID-level GPU process attribution is not available without a protected lease; "
            "gpu_processes is CPU%/RSS enumeration plus optional ioreg device occupancy.",
            "Thermal state is UNKNOWN on this host without sudo (powermetrics is root-only; "
            "no sysctl thermal keys).",
            "tools/odyssey/contamination.py is a train/eval barrier, not machine-state "
            "contamination. Consolidating it into this module would mix two sciences.",
            "This sidecar did not run a benchmark and cannot produce PROTECTED_ABSOLUTE.",
        ],
    }
    probes = (snap.get("probes") or {}) if isinstance(snap.get("probes"), dict) else {}
    probe_names = sorted(str(k) for k in probes)
    evidence = klass.get("contamination_evidence") or []
    record_contamination_causality(
        doc,
        probe_performed=(
            "snapshot() then classify_contamination(snap): loadavg, process "
            "cpu%/rss, memory pressure, optional gpu occupancy; classify by "
            "QUIET_*/HEAVY_* thresholds with UNKNOWN on a failed required probe"
        ),
        direct_observation=(
            f"contamination_class={klass['contamination_class']}; "
            f"required_probes={klass.get('required_probes')}; "
            f"n_evidence={len(evidence)}; probes_run={probe_names}"
        ),
        interpretation=str(klass.get("contamination_reason") or klass["contamination_class"]),
        probe_kind=sc.PROBE_MEASURED_FLAGS,
        claim_kind=sc.CLAIM_FIELD_VALUE,
    )
    return write_receipt(RECEIPT, doc, "tools/future/contamination.py")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", action="store_true", help="take a live snapshot and write the receipt")
    ap.add_argument("--build", action="store_true", help="same as --snapshot")
    ap.add_argument("--selftest", action="store_true", help="run the gate self-test and print JSON")
    a = ap.parse_args()
    if a.selftest:
        print(json.dumps(selftest(), indent=2, sort_keys=True))
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
