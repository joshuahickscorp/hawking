"""Live host signal collectors for the pressure governor.

Real signal sources used tonight and retained as the production path:

- disk free: ``shutil.disk_usage`` / ``df`` (same family as ``v0_notifier.free_gib``
  and ``reclaim_storage_keep_proto`` free-space reporting)
- memory / swap: ``vm_stat`` + ``sysctl hw.memsize`` + ``sysctl vm.swapusage``
  (same family as ``lab.operators.glm52_grounding.parse_darwin_memory`` and
  ``lab.operators.bounded_cache.available_ram_bytes``)
- GPU utilization: ``ioreg -r -d 1 -c IOAccelerator`` Device Utilization %
  (verbatim technique from ``v0_notifier.gpu_pct``)
- thermal: ``pmset -g therm`` when available; unknown otherwise (fail-open to
  non-thermal, fail-closed only when an explicit thermal-throttle flag is set)
- foreground user activity: ``lsappinfo front`` / ``osascript`` best-effort;
  injectable for tests

All collectors are pure functions of process output where possible so unit tests
can inject fixtures without touching the live machine.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Mapping, Optional


_GIB = 1024**3
_VM_PAGE_SIZE_RE = re.compile(r"page size of (\d+) bytes")
_SWAP_VALUE_RE = re.compile(
    r"(total|used|free)\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGT])",
    re.IGNORECASE,
)
_GPU_UTIL_RE = re.compile(r'"Device Utilization %"=(\d+)')


@dataclass(frozen=True)
class HostSignals:
    """Snapshot of host pressure inputs.

    Missing optional sensors are ``None`` — the governor treats unknown as
    non-escalating unless a hard floor (disk/memory) is already crossed.
    """

    free_disk_bytes: int
    total_disk_bytes: int
    available_ram_bytes: int
    total_ram_bytes: int
    swap_used_bytes: int
    swap_total_bytes: int
    gpu_util_pct: Optional[int] = None
    thermal_throttling: Optional[bool] = None
    foreground_user_active: Optional[bool] = None
    source: str = "synthetic"

    @property
    def free_disk_gib(self) -> float:
        return self.free_disk_bytes / _GIB

    @property
    def available_ram_gib(self) -> float:
        return self.available_ram_bytes / _GIB

    @property
    def swap_used_gib(self) -> float:
        return self.swap_used_bytes / _GIB

    @property
    def ram_pressure_ratio(self) -> float:
        if self.total_ram_bytes <= 0:
            return 1.0
        used = max(0, self.total_ram_bytes - self.available_ram_bytes)
        return min(1.0, used / self.total_ram_bytes)

    def as_dict(self) -> dict:
        return {
            "free_disk_bytes": self.free_disk_bytes,
            "total_disk_bytes": self.total_disk_bytes,
            "available_ram_bytes": self.available_ram_bytes,
            "total_ram_bytes": self.total_ram_bytes,
            "swap_used_bytes": self.swap_used_bytes,
            "swap_total_bytes": self.swap_total_bytes,
            "gpu_util_pct": self.gpu_util_pct,
            "thermal_throttling": self.thermal_throttling,
            "foreground_user_active": self.foreground_user_active,
            "source": self.source,
            "free_disk_gib": round(self.free_disk_gib, 3),
            "available_ram_gib": round(self.available_ram_gib, 3),
            "ram_pressure_ratio": round(self.ram_pressure_ratio, 4),
        }


def free_disk_bytes(path: str = "/") -> tuple[int, int]:
    """Return ``(free, total)`` bytes for ``path`` via ``shutil.disk_usage``."""
    usage = shutil.disk_usage(path)
    return int(usage.free), int(usage.total)


def parse_vm_stat(text: str, *, total_ram_bytes: int) -> tuple[int, int]:
    """Parse Darwin ``vm_stat`` into ``(available_ram_bytes, total_ram_bytes)``.

    Available = (free + inactive + speculative + purgeable) * page_size when
    present; purgeable is optional and added when listed.
    """
    page_match = _VM_PAGE_SIZE_RE.search(text)
    if page_match is None:
        raise ValueError("vm_stat omitted page size")
    page_size = int(page_match.group(1))
    pages: dict[str, int] = {}
    for line in text.splitlines()[1:]:
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        digits = raw.strip().rstrip(".")
        if digits.isdigit():
            pages[name.strip()] = int(digits)
    required = ("Pages free", "Pages inactive", "Pages speculative")
    missing = [n for n in required if n not in pages]
    if missing:
        raise ValueError(f"vm_stat missing fields: {missing}")
    available_pages = sum(pages[n] for n in required)
    available_pages += pages.get("Pages purgeable", 0)
    return available_pages * page_size, total_ram_bytes


def parse_swapusage(text: str) -> tuple[int, int]:
    """Parse ``sysctl -n vm.swapusage`` into ``(used_bytes, total_bytes)``."""
    values: dict[str, float] = {}
    powers = {"K": 1, "M": 2, "G": 3, "T": 4}
    for match in _SWAP_VALUE_RE.finditer(text):
        field = match.group(1).lower()
        number = float(match.group(2))
        unit = match.group(3).upper()
        values[field] = number * (1024 ** powers[unit])
    if "total" not in values or "used" not in values:
        raise ValueError(f"vm.swapusage missing total/used: {text!r}")
    used = min(int(values["used"]), int(values["total"]))
    return used, int(values["total"])


def parse_gpu_util_ioreg(text: str) -> Optional[int]:
    """Parse ``ioreg -r -d 1 -c IOAccelerator`` Device Utilization % (max)."""
    vals = [int(v) for v in _GPU_UTIL_RE.findall(text)]
    return max(vals) if vals else None


def parse_thermal_pmset(text: str) -> Optional[bool]:
    """Best-effort thermal throttle flag from ``pmset -g therm`` output."""
    lower = text.lower()
    if not text.strip():
        return None
    # Explicit throttle / CPU_Speed_Limit below 100 ⇒ thermal pressure.
    if "cpu_speed_limit" in lower:
        m = re.search(r"cpu_speed_limit\s*=\s*(\d+)", lower)
        if m is not None:
            return int(m.group(1)) < 100
    if "thermal level" in lower or "thermallevel" in lower:
        m = re.search(r"thermal\s*level\s*[:=]\s*(\d+)", lower)
        if m is not None:
            return int(m.group(1)) > 0
    if "throttl" in lower:
        return True
    return False


def _run(cmd: list[str], *, timeout: float = 10.0) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        ).stdout
    except Exception:
        return ""


def collect_host_signals(
    *,
    disk_path: str = "/",
    runners: Optional[Mapping[str, Callable[[], str]]] = None,
    disk_probe: Optional[Callable[[str], tuple[int, int]]] = None,
    foreground_probe: Optional[Callable[[], Optional[bool]]] = None,
) -> HostSignals:
    """Collect a live (or injected) host signal snapshot.

    ``runners`` maps logical sensor names to callables returning text:
    ``hw_memsize``, ``vm_stat``, ``swapusage``, ``ioreg_gpu``, ``pmset_therm``.
    Omit to use the live Darwin commands.
    """
    runners = dict(runners or {})
    if disk_probe is None:
        disk_probe = free_disk_bytes
    free_b, total_b = disk_probe(disk_path)

    def _default_hw() -> str:
        return _run(["/usr/sbin/sysctl", "-n", "hw.memsize"]).strip()

    def _default_vm() -> str:
        return _run(["/usr/bin/vm_stat"])

    def _default_swap() -> str:
        return _run(["/usr/sbin/sysctl", "-n", "vm.swapusage"]).strip()

    def _default_gpu() -> str:
        return _run(["/usr/sbin/ioreg", "-r", "-d", "1", "-c", "IOAccelerator"])

    def _default_therm() -> str:
        return _run(["/usr/bin/pmset", "-g", "therm"])

    hw_text = runners.get("hw_memsize", _default_hw)()
    vm_text = runners.get("vm_stat", _default_vm)()
    swap_text = runners.get("swapusage", _default_swap)()
    gpu_text = runners.get("ioreg_gpu", _default_gpu)()
    therm_text = runners.get("pmset_therm", _default_therm)()

    try:
        total_ram = int(str(hw_text).strip())
    except (TypeError, ValueError):
        total_ram = 0
    try:
        available_ram, total_ram = parse_vm_stat(vm_text, total_ram_bytes=total_ram)
    except ValueError:
        available_ram = 0
    try:
        swap_used, swap_total = parse_swapusage(swap_text)
    except ValueError:
        swap_used, swap_total = 0, 0

    gpu = parse_gpu_util_ioreg(gpu_text) if gpu_text else None
    thermal = parse_thermal_pmset(therm_text) if therm_text else None
    fg: Optional[bool] = None
    if foreground_probe is not None:
        fg = foreground_probe()

    source = "injected" if runners or disk_probe is not free_disk_bytes else "darwin:live"
    return HostSignals(
        free_disk_bytes=free_b,
        total_disk_bytes=total_b,
        available_ram_bytes=available_ram,
        total_ram_bytes=total_ram,
        swap_used_bytes=swap_used,
        swap_total_bytes=swap_total,
        gpu_util_pct=gpu,
        thermal_throttling=thermal,
        foreground_user_active=fg,
        source=source,
    )
