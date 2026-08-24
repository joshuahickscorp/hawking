from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

GIB = 1024 ** 3
MIB = 1024 * 1024
DEFAULT_PER_RUNTIME_OVERHEAD_BYTES = int(1.6 * GIB)
DEFAULT_HEADROOM_FRAC = 0.10
DEFAULT_SWAP_CEILING_GIB = 2.0


def _run_cmd(cmd: List[str]) -> Optional[str]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        pass
    return None


def _parse_vm_stat() -> Dict[str, int]:
    out = _run_cmd(["vm_stat"])
    if not out:
        return {}
    result = {}
    for line in out.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            val = val.strip().rstrip(".")
            if val.isdigit():
                result[key.strip()] = int(val)
    return result


def _get_page_size() -> int:
    out = _run_cmd(["sysctl", "-n", "vm.pagesize"])
    if out and out.isdigit():
        return int(out)
    return 16384


def _read_positive_int(value: object) -> Optional[int]:
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if n < 1:
        return None
    return n


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _walk_parents(start: Path):
    start = start.resolve()
    yield start
    yield from start.parents


def default_repo_root() -> Path:
    from .paths import find_repo_root

    return find_repo_root(Path(__file__))


def swap_used_bytes() -> int:
    out = _run_cmd(["sysctl", "-n", "vm.swapusage"]) or ""
    m = re.search(r"used\s*=\s*([\d.]+)([MG])", out)
    if not m:
        return 0
    v = float(m.group(1))
    return int(v * (GIB if m.group(2) == "G" else MIB))


def host_snapshot() -> Dict[str, Any]:
    """Page-level host memory. Free RAM is the SECOND admission gate."""
    page = _get_page_size()
    raw = _parse_vm_stat()
    counts = {k: int(v) * page for k, v in raw.items()}
    total = 0
    mem_out = _run_cmd(["sysctl", "-n", "hw.memsize"])
    if mem_out and mem_out.isdigit():
        total = int(mem_out)
    free = (
        counts.get("Pages free", 0)
        + counts.get("Pages inactive", 0)
        + counts.get("Pages speculative", 0)
    )
    pressure = "unknown"
    pressure_out = _run_cmd(["memory_pressure", "-Q"])
    if pressure_out:
        low = pressure_out.lower()
        if "low" in low:
            pressure = "low"
        elif "normal" in low:
            pressure = "normal"
        elif "high" in low:
            pressure = "high"
    return {
        "total_bytes": total,
        "free_bytes": free,
        "wired_bytes": counts.get("Pages wired down", 0),
        "compressed_bytes": counts.get("Pages occupied by compressor", 0),
        "file_backed_bytes": counts.get("File-backed pages", 0),
        "anonymous_bytes": counts.get("Anonymous pages", 0),
        "swap_used_bytes": swap_used_bytes(),
        "page_size": page,
        "pressure": pressure,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


_METAL_MOD = None
_METAL_CACHE: Optional[Dict[str, Any]] = None


def _metal_budget_module():
    global _METAL_MOD
    if _METAL_MOD is not None:
        return _METAL_MOD
    from .paths import find_repo_root
    path = find_repo_root(Path(__file__)) / "tools" / "headless" / "metal_budget.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("hcli_metal_budget", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _METAL_MOD = mod
    return mod


def metal_device_info(force: bool = False) -> Dict[str, Any]:
    """Re-read the Metal working-set budget. First admission gate on Apple Silicon."""
    global _METAL_CACHE
    if _METAL_CACHE is not None and not force:
        return dict(_METAL_CACHE)
    mod = _metal_budget_module()
    if mod is not None and hasattr(mod, "metal_device"):
        info = dict(mod.metal_device())
        if hasattr(mod, "wired_limit_override"):
            try:
                info["wired_limit"] = mod.wired_limit_override()
            except Exception:
                info["wired_limit"] = None
        _METAL_CACHE = dict(info)
        return info
    total = 0
    mem_out = _run_cmd(["sysctl", "-n", "hw.memsize"])
    if mem_out and mem_out.isdigit():
        total = int(mem_out)
    info = {
        "name": platform.processor() or "unknown",
        "hasUnifiedMemory": True,
        "recommendedMaxWorkingSetSize": int(total * 0.75) if total else 0,
        "maxBufferLength": None,
        "currentAllocatedSize": None,
        "source": "ESTIMATE: 75% of hw.memsize — metal_budget unavailable",
        "wired_limit": None,
    }
    _METAL_CACHE = dict(info)
    return info


@dataclass
class MachineProbe:
    physical_memory_bytes: int = 0
    page_size: int = 16384
    vm_stat: Dict[str, int] = field(default_factory=dict)
    pressure: str = "unknown"
    swap_bytes: int = 0
    free_bytes: int = 0

    def probe(self) -> "MachineProbe":
        snap = host_snapshot()
        self.page_size = int(snap.get("page_size") or _get_page_size())
        self.physical_memory_bytes = int(snap.get("total_bytes") or 0)
        self.vm_stat = _parse_vm_stat()
        self.pressure = str(snap.get("pressure") or "unknown")
        self.swap_bytes = int(snap.get("swap_used_bytes") or 0)
        self.free_bytes = int(snap.get("free_bytes") or 0)
        return self


@dataclass
class AdmissionDecision:
    allow: bool
    reason: str
    gate: str
    details: Dict[str, Any] = field(default_factory=dict)


class MemGate:
    """Two-stage admission: Metal working set first, host RAM/swap second.

    RSS is a closer proxy for GPU cost than the free-RAM delta is. llama.cpp
    mmaps weights so the pages are shared, but each process wraps those pages
    in its own MTLBuffers and Metal charges every process separately.
    """

    def __init__(
        self,
        reserve_bytes: Optional[int] = None,
        swap_ceiling_bytes: Optional[int] = None,
        model_bytes: int = 0,
        per_runtime_overhead_bytes: Optional[int] = None,
        headroom_frac: Optional[float] = None,
        metal_info: Optional[Dict[str, Any]] = None,
        topology: str = "slot",
    ) -> None:
        self.reserve_bytes = reserve_bytes
        self.swap_ceiling_bytes = swap_ceiling_bytes
        self.model_bytes = int(model_bytes or 0)
        env_overhead = os.environ.get("HCLI_PER_RUNTIME_OVERHEAD_GIB")
        if per_runtime_overhead_bytes is not None:
            self.per_runtime_overhead_bytes = int(per_runtime_overhead_bytes)
        elif env_overhead:
            try:
                self.per_runtime_overhead_bytes = int(float(env_overhead) * GIB)
            except ValueError:
                self.per_runtime_overhead_bytes = DEFAULT_PER_RUNTIME_OVERHEAD_BYTES
        else:
            self.per_runtime_overhead_bytes = DEFAULT_PER_RUNTIME_OVERHEAD_BYTES
        env_head = os.environ.get("HCLI_GPU_HEADROOM_FRAC")
        if headroom_frac is not None:
            self.headroom_frac = float(headroom_frac)
        elif env_head:
            try:
                self.headroom_frac = float(env_head)
            except ValueError:
                self.headroom_frac = DEFAULT_HEADROOM_FRAC
        else:
            self.headroom_frac = DEFAULT_HEADROOM_FRAC
        self._metal_info = metal_info
        self.topology = topology if topology in {"slot", "process"} else "slot"
        env_reserve = os.environ.get("HCLI_MEM_RESERVE_BYTES") or os.environ.get(
            "HCLI_RESERVE_GIB"
        )
        if self.reserve_bytes is None and env_reserve:
            try:
                if os.environ.get("HCLI_MEM_RESERVE_BYTES"):
                    self.reserve_bytes = int(os.environ["HCLI_MEM_RESERVE_BYTES"])
                else:
                    self.reserve_bytes = int(float(env_reserve) * GIB)
            except ValueError:
                pass

    def calculate_reserve(self, physical_memory_bytes: int) -> int:
        if self.reserve_bytes is not None:
            return int(self.reserve_bytes)
        return max(12 * 1024**3, int(physical_memory_bytes * 0.15))

    def available_estimate(self, probe: MachineProbe) -> int:
        reserve = self.calculate_reserve(probe.physical_memory_bytes)
        return max(0, probe.physical_memory_bytes - reserve)

    def is_safe(self, probe: MachineProbe) -> bool:
        if probe.pressure == "high":
            return False
        ceiling = self._swap_ceiling()
        if probe.swap_bytes > ceiling:
            return False
        return True

    def _swap_ceiling(self) -> int:
        if self.swap_ceiling_bytes is not None:
            return int(self.swap_ceiling_bytes)
        env = os.environ.get("HCLI_SWAP_CEILING_GIB")
        if env:
            try:
                return int(float(env) * GIB)
            except ValueError:
                pass
        return int(DEFAULT_SWAP_CEILING_GIB * GIB)

    def metal(self, refresh: bool = False) -> Dict[str, Any]:
        if self._metal_info is not None and not refresh:
            return dict(self._metal_info)
        info = metal_device_info(force=refresh)
        if self._metal_info is None:
            self._metal_info = dict(info)
        return info

    def gpu_cost_bytes(self, admitted: int, extra: int = 1) -> int:
        """GPU working-set bytes after admitting ``extra`` more runtimes.

        SLOT: one process holds the weights once; extra slots add KV/compute.
        PROCESS: each runtime charges a full model + overhead against Metal.
        """
        n = max(0, int(admitted) + int(extra))
        if n <= 0:
            return 0
        if self.topology == "process":
            return n * (self.model_bytes + self.per_runtime_overhead_bytes)
        return self.model_bytes + n * self.per_runtime_overhead_bytes

    def consider(
        self,
        admitted: int,
        extra: int = 1,
        snapshot: Optional[Dict[str, Any]] = None,
        refresh_metal: bool = True,
    ) -> AdmissionDecision:
        """Return whether one more runtime (or ``extra`` more) may be admitted.

        Gate order is not negotiable: GPU working set first, host RAM/swap
        second. See receipts/headless/GPU_MEMORY_GATE.json.
        """
        # Injected metal_info is for tests; otherwise re-read so
        # iogpu.wired_limit_mb and foreign allocations can change the budget.
        if self._metal_info is not None:
            metal = dict(self._metal_info)
        else:
            metal = metal_device_info(force=refresh_metal)
        budget = int(metal.get("recommendedMaxWorkingSetSize") or 0)
        usable = int(budget * (1.0 - self.headroom_frac)) if budget else 0
        next_cost = self.gpu_cost_bytes(admitted, extra=extra)
        increment = self.gpu_cost_bytes(admitted, extra=extra) - self.gpu_cost_bytes(
            admitted, extra=0
        )
        details: Dict[str, Any] = {
            "topology": self.topology,
            "admitted": int(admitted),
            "extra": int(extra),
            "model_bytes": self.model_bytes,
            "per_runtime_overhead_bytes": self.per_runtime_overhead_bytes,
            "headroom_frac": self.headroom_frac,
            "recommendedMaxWorkingSetSize": budget,
            "usable_bytes": usable,
            "next_gpu_cost_bytes": next_cost,
            "increment_gpu_bytes": increment,
            "metal_source": metal.get("source"),
            "currentAllocatedSize": metal.get("currentAllocatedSize"),
            "wired_limit": metal.get("wired_limit"),
        }
        if budget <= 0:
            return AdmissionDecision(
                False, "gpu working set budget is unavailable", "gpu", details
            )
        if next_cost > usable:
            return AdmissionDecision(
                False,
                (
                    f"gpu working set: {(admitted + extra)} runtimes would charge "
                    f"{next_cost} bytes against usable {usable} "
                    f"(budget {budget} with {self.headroom_frac:.0%} headroom)"
                ),
                "gpu",
                details,
            )
        current = metal.get("currentAllocatedSize")
        try:
            current_i = int(current) if current is not None else 0
        except (TypeError, ValueError):
            current_i = 0
        if current_i and increment and (current_i + increment) > budget:
            details["currentAllocatedSize"] = current_i
            return AdmissionDecision(
                False,
                (
                    f"gpu currentAllocatedSize {current_i} + increment {increment} "
                    f"exceeds recommendedMaxWorkingSetSize {budget}"
                ),
                "gpu",
                details,
            )

        snap = snapshot if snapshot is not None else host_snapshot()
        total = int(snap.get("total_bytes") or 0)
        free = int(snap.get("free_bytes") or 0)
        swap = int(snap.get("swap_used_bytes") or 0)
        pressure = str(snap.get("pressure") or "unknown")
        reserve = self.calculate_reserve(total)
        ceiling = self._swap_ceiling()
        details.update(
            {
                "free_bytes": free,
                "swap_used_bytes": swap,
                "reserve_bytes": reserve,
                "swap_ceiling_bytes": ceiling,
                "pressure": pressure,
                "total_bytes": total,
            }
        )
        if pressure == "high":
            return AdmissionDecision(
                False, "host memory pressure is high", "host", details
            )
        if free < reserve:
            return AdmissionDecision(
                False,
                f"free RAM {free} bytes is below reserve {reserve} bytes",
                "host",
                details,
            )
        if swap > ceiling:
            return AdmissionDecision(
                False,
                f"swap {swap} bytes exceeds ceiling {ceiling} bytes",
                "host",
                details,
            )
        return AdmissionDecision(True, "ok", "ok", details)


def _resident_from_mapping(data: Dict[str, Any]) -> Optional[int]:
    for key in (
        "resident_runtime_limit",
        "RESIDENT_RUNTIME_LIMIT",
        "resident_limit",
        "bootstrap_workers",
    ):
        if key in data:
            n = _read_positive_int(data.get(key))
            if n is not None:
                return n
    return None


def _active_from_mapping(data: Dict[str, Any]) -> Optional[int]:
    for key in ("active_decode_limit", "ACTIVE_DECODE_LIMIT"):
        if key in data:
            n = _read_positive_int(data.get(key))
            if n is not None:
                return n
    return None


def _env_int(*names: str) -> Optional[Tuple[int, str]]:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            continue
        n = _read_positive_int(str(raw).strip())
        if n is not None:
            return n, f"env:{name}"
    return None


# A genome is a PRIOR, not present truth. The operator-qualified file at
# ~/.config/hcli/machine_genome.json is produced by
# tools/headless/machine_probe.py. Do not hand-edit it to "fix" a limit;
# re-run the probe and promote its receipts/headless/MACHINE_GENOME.json.
GENOME_PRODUCER = "tools/headless/machine_probe.py"
DEFAULT_GENOME_STALENESS_HORIZON_S = 7 * 24 * 3600
GENOME_PROMOTION = (
    "Do not edit machine_genome.json by hand. Re-run "
    "tools/headless/machine_probe.py and promote its "
    "receipts/headless/MACHINE_GENOME.json."
)

# c=1 aggregate tok/s by SLOT count. Prior from
# receipts/headless/QWEN_MAX_EQUILIBRIUM*.json (pools spawned by HCLI,
# alternating paired reps). Not a live measurement of the current spawn.
PUBLISHED_SLOT_C1_TPS_PRIOR: Dict[int, float] = {
    2: 23.727,
    3: 23.691,
    5: 22.489,
}
PUBLISHED_SLOT_C1_TPS_SOURCE = "receipts/headless/QWEN_MAX_EQUILIBRIUM*.json"
PUBLISHED_SLOT_C1_BASELINE_SLOTS = 2


class GenomeStale(Exception):
    """Raised when a caller requires a FRESH genome and the prior is STALE."""

    def __init__(self, freshness: "GenomeFreshness") -> None:
        self.freshness = freshness
        detail = "; ".join(freshness.reasons) or "unspecified"
        super().__init__(f"STALE genome: {detail}")


@dataclass
class GenomeFreshness:
    status: str
    reasons: List[str] = field(default_factory=list)
    path: Optional[str] = None
    generated_at: Optional[str] = None
    horizon_s: int = DEFAULT_GENOME_STALENESS_HORIZON_S
    producer: str = GENOME_PRODUCER
    promotion: str = GENOME_PROMOTION

    @property
    def stale(self) -> bool:
        return self.status == "STALE"

    def raise_if_stale(self) -> None:
        if self.stale:
            raise GenomeStale(self)


def genome_staleness_horizon_s(override: Optional[int] = None) -> int:
    if override is not None:
        try:
            n = int(override)
        except (TypeError, ValueError):
            n = DEFAULT_GENOME_STALENESS_HORIZON_S
        return max(1, n)
    raw = os.environ.get("HCLI_GENOME_STALENESS_HORIZON_S")
    if raw:
        n = _read_positive_int(str(raw).strip())
        if n is not None:
            return n
    return DEFAULT_GENOME_STALENESS_HORIZON_S


def _as_unix(
    value: Optional[Union[str, float, int, datetime]],
) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def live_machine_identity() -> Dict[str, Any]:
    hw_model = _run_cmd(["sysctl", "-n", "hw.model"])
    cpu = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
    ncpu_s = _run_cmd(["sysctl", "-n", "hw.ncpu"])
    mem_s = _run_cmd(["sysctl", "-n", "hw.memsize"])
    ncpu = int(ncpu_s) if ncpu_s and str(ncpu_s).strip().lstrip("-").isdigit() else None
    mem_bytes = int(mem_s) if mem_s and str(mem_s).strip().isdigit() else None
    return {
        "hw_model": hw_model or None,
        "cpu": cpu or None,
        "ncpu": ncpu,
        "mem_bytes": mem_bytes,
    }


def _norm_model_path(path: Optional[str]) -> Optional[str]:
    if not path or not str(path).strip():
        return None
    return os.path.realpath(os.path.expanduser(str(path)))


def _looks_like_genome(data: Dict[str, Any]) -> bool:
    schema = str(data.get("schema") or "")
    if "machine_genome" in schema.lower():
        return True
    if data.get("generated_at") or data.get("measured_by"):
        return True
    if isinstance(data.get("machine"), dict):
        return True
    if isinstance(data.get("runtime_identity"), dict):
        return True
    return False


def assess_genome_freshness(
    data: Optional[Dict[str, Any]],
    *,
    path: Optional[Union[str, Path]] = None,
    model_path: Optional[str] = None,
    model_bytes: Optional[int] = None,
    now: Optional[Union[str, float, int, datetime]] = None,
    horizon_s: Optional[int] = None,
    live_machine: Optional[Dict[str, Any]] = None,
) -> GenomeFreshness:
    """Report FRESH or STALE. Never silently trust a prior.

    Invalidators: machine identity, model identity, measurement timestamp,
    explicit staleness horizon. A FRESH genome can still understate the live
    knee; freshness is identity+time, not a re-measure. Promotion is
    tools/headless/machine_probe.py.
    """
    horizon = genome_staleness_horizon_s(horizon_s)
    path_s = str(path) if path is not None else None
    if not isinstance(data, dict) or not data:
        return GenomeFreshness(
            status="STALE",
            reasons=["genome is empty or unreadable"],
            path=path_s,
            horizon_s=horizon,
        )
    reasons: List[str] = []
    generated_at = data.get("generated_at")
    generated_s = str(generated_at) if generated_at else None

    ts = _as_unix(generated_at)
    now_unix = _as_unix(now) if now is not None else time.time()
    if now_unix is None:
        now_unix = time.time()
    if ts is None:
        reasons.append("missing measurement timestamp (generated_at)")
    else:
        age = now_unix - ts
        if age > horizon:
            reasons.append(
                f"older than staleness horizon: generated_at={generated_s} "
                f"age={int(age)}s horizon={horizon}s"
            )
        elif age < -300:
            reasons.append(
                f"measurement timestamp is in the future: generated_at={generated_s}"
            )

    genome_machine = data.get("machine") if isinstance(data.get("machine"), dict) else None
    live = dict(live_machine) if live_machine is not None else live_machine_identity()
    if not genome_machine:
        reasons.append("missing machine identity")
    else:
        for key, label in (
            ("hw_model", "hw_model"),
            ("mem_bytes", "mem_bytes"),
            ("ncpu", "ncpu"),
            ("cpu", "cpu"),
        ):
            gval = genome_machine.get(key)
            lval = live.get(key) if isinstance(live, dict) else None
            if gval is None or gval == "":
                if key in {"hw_model", "mem_bytes"}:
                    reasons.append(f"missing machine identity field {label}")
                continue
            if lval is None or lval == "":
                reasons.append(
                    f"live machine identity unobservable for {label} "
                    f"(genome {label}={gval!r})"
                )
                continue
            if key in {"mem_bytes", "ncpu"}:
                try:
                    if int(gval) != int(lval):
                        reasons.append(
                            f"machine identity mismatch: genome {label}={gval!r} "
                            f"live {label}={lval!r}"
                        )
                except (TypeError, ValueError):
                    reasons.append(
                        f"machine identity mismatch: genome {label}={gval!r} "
                        f"live {label}={lval!r}"
                    )
            else:
                if str(gval) != str(lval):
                    reasons.append(
                        f"machine identity mismatch: genome {label}={gval!r} "
                        f"live {label}={lval!r}"
                    )

    if model_path:
        ident = (
            data.get("runtime_identity")
            if isinstance(data.get("runtime_identity"), dict)
            else {}
        )
        g_model = _norm_model_path(
            ident.get("model_path") or data.get("model_path")
        )
        live_model = _norm_model_path(model_path)
        if not g_model:
            reasons.append(
                "missing model identity while a live model was supplied"
            )
        elif live_model and g_model != live_model:
            reasons.append(
                f"model identity mismatch: genome model_path={g_model} "
                f"live model_path={live_model}"
            )
        g_bytes = ident.get("model_size_bytes")
        if g_bytes is None:
            g_bytes = data.get("model_size_bytes")
        live_bytes = model_bytes
        if live_bytes is None and live_model and os.path.isfile(live_model):
            try:
                live_bytes = os.path.getsize(live_model)
            except OSError:
                live_bytes = None
        if g_bytes is not None and live_bytes is not None:
            try:
                if int(g_bytes) != int(live_bytes):
                    reasons.append(
                        f"model identity mismatch: genome model_size_bytes={g_bytes!r} "
                        f"live model_size_bytes={live_bytes!r}"
                    )
            except (TypeError, ValueError):
                reasons.append(
                    f"model identity mismatch: genome model_size_bytes={g_bytes!r} "
                    f"live model_size_bytes={live_bytes!r}"
                )

    status = "STALE" if reasons else "FRESH"
    return GenomeFreshness(
        status=status,
        reasons=reasons,
        path=path_s,
        generated_at=generated_s,
        horizon_s=horizon,
    )


def slot_allocation_decision(
    planned_slots: int,
    *,
    topology: str,
    active_decode_limit: int,
    requested_n: int,
    c1_tps_by_slots: Optional[Dict[int, float]] = None,
    c1_tps_source: Optional[str] = None,
) -> Dict[str, Any]:
    """Cost of allocating more sequences than a useful decode width.

    Extra slots above a single deep decode reserve KV and, on this box,
    cost ~5% of single-stream tok/s at 5 slots vs 2. The numeric loss is a
    PRIOR from QWEN_MAX_EQUILIBRIUM, not a live measurement of this spawn.
    Callers who want one deep decode should pass requested_n=1; callers who
    want concurrent decode should pass requested_n at most ACTIVE_DECODE_LIMIT.
    Promotion of a new qualified width is tools/headless/machine_probe.py.
    """
    planned = max(0, int(planned_slots))
    active = max(1, int(active_decode_limit or 1))
    table = {
        int(k): float(v)
        for k, v in (c1_tps_by_slots or PUBLISHED_SLOT_C1_TPS_PRIOR).items()
    }
    source = c1_tps_source or PUBLISHED_SLOT_C1_TPS_SOURCE
    baseline = PUBLISHED_SLOT_C1_BASELINE_SLOTS
    caller_slot = (
        "For a single deep decode, pass requested_n=1. For concurrent decode, "
        "pass requested_n at most ACTIVE_DECODE_LIMIT. Extra slots above the "
        "useful width reserve KV and reduce single-stream tok/s. "
        "Do not raise ACTIVE_DECODE_LIMIT by editing machine_genome.json; "
        "re-run tools/headless/machine_probe.py and promote its "
        "receipts/headless/MACHINE_GENOME.json."
    )
    if topology != "slot":
        return {
            "topology": topology,
            "planned_slots": planned,
            "requested_n": int(requested_n),
            "active_decode_limit": active,
            "oversized": False,
            "oversized_vs_single_stream": False,
            "oversized_vs_active_decode": False,
            "single_stream_cost": {
                "kind": "n/a",
                "relative_loss": None,
                "reason": (
                    "process topology uses 1 slot per process; "
                    "slot-oversizing does not apply"
                ),
            },
            "caller_should": (
                "Process topology does not split one server into N sequences. "
                "For a single deep decode still pass requested_n=1 so MemGate "
                "does not charge extra full-model working sets. "
                + GENOME_PROMOTION
            ),
        }
    oversized_single = planned > 1
    oversized_active = planned > active
    if planned in table and baseline in table:
        planned_tps = float(table[planned])
        baseline_tps = float(table[baseline])
        loss: Optional[float]
        if planned == baseline:
            loss = 0.0
        elif baseline_tps > 0:
            loss = (baseline_tps - planned_tps) / baseline_tps
        else:
            loss = None
        cost: Dict[str, Any] = {
            "kind": "prior",
            "relative_loss": loss,
            "baseline_slots": baseline,
            "planned_slots": planned,
            "baseline_c1_tps": baseline_tps,
            "planned_c1_tps": planned_tps,
            "source": source,
            "reason": None,
        }
    else:
        have = sorted(table)
        cost = {
            "kind": "unknown",
            "relative_loss": None,
            "baseline_slots": baseline,
            "planned_slots": planned,
            "source": source,
            "reason": (
                f"no c=1 tok/s prior for planned_slots={planned} "
                f"(have {have}; source {source})"
            ),
        }
    return {
        "topology": topology,
        "planned_slots": planned,
        "requested_n": int(requested_n),
        "active_decode_limit": active,
        "useful_single_stream_slots": 1,
        "oversized": oversized_single or oversized_active,
        "oversized_vs_single_stream": oversized_single,
        "oversized_vs_active_decode": oversized_active,
        "single_stream_cost": cost,
        "caller_should": caller_slot,
        "idle_slots_vs_active_decode": max(0, planned - active),
    }


@dataclass
class LimitResolution:
    resident_limit: int
    resident_source: str
    active_decode_limit: int
    active_source: str
    genome_reports: List[GenomeFreshness] = field(default_factory=list)


def resolve_runtime_limits(
    repo_root: Optional[Union[str, Path]] = None,
    start_dir: Optional[Union[str, Path]] = None,
    *,
    model_path: Optional[str] = None,
    model_bytes: Optional[int] = None,
    now: Optional[Union[str, float, int, datetime]] = None,
    horizon_s: Optional[int] = None,
    live_machine: Optional[Dict[str, Any]] = None,
) -> LimitResolution:
    """Independent resident / active-decode limits. First hit wins per axis.

    1. env HCLI_RESIDENT_RUNTIME_LIMIT / HCLI_ACTIVE_DECODE_LIMIT
    2. ~/.config/hcli/machine_genome.json  (FRESH genomes only)
    3. <repo>/receipts/headless/MACHINE_GENOME.json  (FRESH if genome-shaped)
    4. <repo>/.haider/bootstrap-director-v6/worker-equilibrium.json
    5. conservative fallback: resident 1, active 1

    A genome is a prior. STALE genomes (wrong machine, wrong model, older
    than the horizon, missing identity/timestamp) are reported on
    ``genome_reports`` and their numbers are not used. Promotion is
    tools/headless/machine_probe.py, not a hand-edit of machine_genome.json.
    """
    resident: Optional[Tuple[int, str]] = _env_int(
        "HCLI_RESIDENT_RUNTIME_LIMIT", "RESIDENT_RUNTIME_LIMIT"
    )
    active: Optional[Tuple[int, str]] = _env_int(
        "HCLI_ACTIVE_DECODE_LIMIT", "ACTIVE_DECODE_LIMIT"
    )

    home_genome = Path.home() / ".config" / "hcli" / "machine_genome.json"
    root = Path(repo_root) if repo_root is not None else default_repo_root()
    start = Path(start_dir) if start_dir is not None else Path.cwd()
    try:
        start = start.resolve()
    except OSError:
        start = Path.cwd()

    eq_path = None
    for directory in _walk_parents(start):
        candidate = (
            directory / ".haider" / "bootstrap-director-v6" / "worker-equilibrium.json"
        )
        if candidate.is_file():
            eq_path = candidate
            break
    if eq_path is None:
        eq_path = (
            root / ".haider" / "bootstrap-director-v6" / "worker-equilibrium.json"
        )

    files: List[Tuple[Path, str]] = [
        (home_genome, str(home_genome)),
        (
            root / "receipts" / "headless" / "MACHINE_GENOME.json",
            "receipts/headless/MACHINE_GENOME.json",
        ),
        (eq_path, "worker-equilibrium.json"),
    ]
    reports: List[GenomeFreshness] = []
    for path, label in files:
        data = _load_json(path)
        if not data:
            continue
        genome_like = path == home_genome or _looks_like_genome(data)
        if genome_like:
            freshness = assess_genome_freshness(
                data,
                path=path,
                model_path=model_path,
                model_bytes=model_bytes,
                now=now,
                horizon_s=horizon_s,
                live_machine=live_machine,
            )
            reports.append(freshness)
            if freshness.stale:
                continue
        if resident is not None and active is not None:
            continue
        if resident is None:
            n = _resident_from_mapping(data)
            if n is not None:
                resident = (n, label)
        if active is None:
            n = _active_from_mapping(data)
            if n is not None:
                active = (n, label)

    if resident is None:
        resident = (1, "fallback")
    if active is None:
        active = (1, "fallback")
    return LimitResolution(
        resident_limit=resident[0],
        resident_source=resident[1],
        active_decode_limit=active[0],
        active_source=active[1],
        genome_reports=reports,
    )


def resolve_decode_topology(
    repo_root: Optional[Union[str, Path]] = None,
) -> Tuple[str, str]:
    """Prefer SLOT when the topology receipt exists; otherwise keep process.

    SLOT is not faster at its best on this box — both topologies top out near
    1.21x a single decoder — but process collapses past its peak while slot
    degrades gracefully. Concurrency is not a throughput lever here.
    """
    env = (os.environ.get("HCLI_DECODE_TOPOLOGY") or "").strip().lower()
    if env in {"slot", "process"}:
        return env, "env:HCLI_DECODE_TOPOLOGY"
    root = Path(repo_root) if repo_root is not None else default_repo_root()
    receipt = root / "receipts" / "headless" / "DECODE_TOPOLOGY.json"
    if receipt.is_file():
        return "slot", "receipts/headless/DECODE_TOPOLOGY.json"
    return "process", "fallback"


@dataclass
class MachineGenome:
    """Compatibility bag over a caller-supplied genome JSON file.

    Not a producer and not an admission authority. Live numbers come from
    ``resolve_runtime_limits``, which reads ``~/.config/hcli/machine_genome.json``
    then ``receipts/headless/MACHINE_GENOME.json`` (FRESH genomes only).
    The producer is ``tools/headless/machine_probe.py``.

    This class must not grow a ``write()`` that bypasses the probe, and it
    must not default onto the canonical config path — a naive save would
    clobber the operator-qualified genome. The default path stays under
    ``HCLI_HOME`` so the two files cannot alias.

    Verified caller: ``tools/headless/hcli_persistence_audit.py`` (crash
    demo of save). Admission must not import this class.
    """

    path: Path
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.path = Path(self.path)
        loaded = _load_json(self.path)
        if loaded is not None:
            self.data = loaded

    def save(self) -> None:
        from .persist import atomic_write_json

        atomic_write_json(self.path, self.data)

    def get_profile(self, key: str) -> Optional[Dict[str, Any]]:
        return self.data.get(key)

    def set_profile(self, key: str, profile: Dict[str, Any]) -> None:
        self.data[key] = profile

    def freshness(self, **kwargs: Any) -> GenomeFreshness:
        """Classify this file the same way ``resolve_runtime_limits`` would."""
        return assess_genome_freshness(self.data, path=self.path, **kwargs)


def get_machine_genome_path() -> Path:
    """Scratch path for the compatibility bag. Not the live genome."""
    base = Path(os.environ.get("HCLI_HOME", Path.home() / ".local" / "share" / "hcli"))
    return base / "machine-genome.json"
