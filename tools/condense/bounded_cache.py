#!/usr/bin/env python3.12
"""Pressure-aware LRU cache for decoded experts (and any large reusable tensors).

Design goal (operator directive): use as much memory as the box will give, lean on macOS dynamic
swap as a cushion, and NEVER hard-crash (OOM/jetsam kill). Trade memory for TIME - re-packing an
expert (RVQ kmeans, seconds) is far slower than letting the OS page a cached expert to SSD and read
it back (tens of ms), so a big cache that spills into swap beats a small cache that re-packs.

Policy: grow freely; evict least-recently-used entries ONLY when the machine is genuinely near the
danger line, i.e. available RAM (psutil.available, which already counts reclaimable/inactive pages)
falls below a small floor, OR free disk (the backing store for swap) falls below a reserve. Between
those limits the cache keeps everything resident and lets the kernel swap as it sees fit. A minimum
working set is always retained so forward progress is guaranteed, and a very high hard cap guards
against a pathological loop. Tunable per stage via env vars so bigger future parents (235B, 397B,
685B, 1T) inherit the same never-crash behaviour without code changes.

Env overrides:
  HAWKING_CACHE_FLOOR_GB       (default 4.0)   evict when available RAM drops below this
  HAWKING_CACHE_DISK_RESERVE_GB(default 30.0)  evict when free disk drops below this (protect swap)
  HAWKING_CACHE_MIN_ENTRIES    (default 48)    never evict below this working set
  HAWKING_CACHE_HARD_MAX       (default 200000) absolute backstop
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

try:
    import psutil  # type: ignore
    _HAVE_PSUTIL = True
except Exception:  # pragma: no cover
    _HAVE_PSUTIL = False

_GIB = 2 ** 30


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


def available_ram_bytes() -> int:
    """Reclaimable/available RAM. psutil when present, else a vm_stat parse (macOS)."""
    if _HAVE_PSUTIL:
        return int(psutil.virtual_memory().available)
    try:
        import subprocess
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
        ps = 16384
        def g(tag: str) -> int:
            for line in out.splitlines():
                if tag in line:
                    return int(line.split()[-1].rstrip("."))
            return 0
        return (g("Pages free") + g("Pages inactive") + g("Pages speculative")
                + g("Pages purgeable")) * ps
    except Exception:
        return 8 * _GIB  # conservative fallback


def free_disk_bytes(path: str) -> int:
    try:
        import shutil
        return int(shutil.disk_usage(path).free)
    except Exception:
        return 100 * _GIB


def _entry_bytes(value: Any) -> int:
    """Bytes held by a cache value (a numpy array, or a tuple/list of them)."""
    try:
        if isinstance(value, (tuple, list)):
            return sum(int(getattr(x, "nbytes", 0)) for x in value)
        return int(getattr(value, "nbytes", 0))
    except Exception:
        return 0


class PressureAwareCache:
    """Ordered dict with pressure-triggered LRU eviction. get()/put() only; len() supported."""

    def __init__(self, name: str, *, disk_path: str = ".", floor_gb: float | None = None,
                 disk_reserve_gb: float | None = None, min_entries: int | None = None,
                 hard_max: int | None = None, check_every: int = 4, verbose: bool = True):
        self.name = name
        self.disk_path = disk_path
        self.floor = int((_env_float("HAWKING_CACHE_FLOOR_GB", 4.0) if floor_gb is None else floor_gb) * _GIB)
        self.disk_reserve = int((_env_float("HAWKING_CACHE_DISK_RESERVE_GB", 30.0) if disk_reserve_gb is None else disk_reserve_gb) * _GIB)
        self.min_entries = int(os.environ.get("HAWKING_CACHE_MIN_ENTRIES", min_entries if min_entries is not None else 48))
        self.hard_max = int(os.environ.get("HAWKING_CACHE_HARD_MAX", hard_max if hard_max is not None else 200000))
        # Absolute byte budget: the PRIMARY control. Fill RAM hard (aggressive) but never spill the
        # cache unbounded into swap - which is what killed the campaign (macOS keeps 'available' high
        # by swapping, so an available-floor alone never triggers and the cache balloons into swap).
        self.max_bytes = int(_env_float("HAWKING_CACHE_MAX_GB", 48.0) * _GIB)
        self.check_every = max(1, int(check_every))
        self.verbose = verbose
        self._d: dict[Any, Any] = {}
        self._bytes = 0
        self._ops = 0
        self.evictions = 0
        self.peak_entries = 0
        self._last_log = 0.0

    def get(self, key: Any) -> Any:
        v = self._d.get(key)
        if v is not None:
            self._d[key] = self._d.pop(key)  # mark most-recently-used
        return v

    def put(self, key: Any, value: Any) -> None:
        if key in self._d:
            self._bytes -= _entry_bytes(self._d[key])
        self._d[key] = value
        self._bytes += _entry_bytes(value)
        self.peak_entries = max(self.peak_entries, len(self._d))
        self._ops += 1
        if self._ops % self.check_every == 0 or self._bytes > self.max_bytes or len(self._d) > self.hard_max:
            self._evict_if_pressured()

    def _pressured(self) -> bool:
        return (self._bytes > self.max_bytes
                or available_ram_bytes() < self.floor
                or free_disk_bytes(self.disk_path) < self.disk_reserve
                or len(self._d) > self.hard_max)

    def _evict_if_pressured(self) -> None:
        evicted = 0
        while len(self._d) > self.min_entries and self._pressured():
            k = next(iter(self._d))  # oldest = least recently used
            self._bytes -= _entry_bytes(self._d.pop(k))
            evicted += 1
        if evicted:
            self.evictions += evicted
            now = time.time()
            if self.verbose and (now - self._last_log) > 10.0:
                self._last_log = now
                avail = available_ram_bytes() / _GIB
                sys.stderr.write(f"[cache:{self.name}] evicted {evicted} (total {self.evictions}); "
                                 f"entries={len(self._d)} avail_ram={avail:.1f}GB\n")
                sys.stderr.flush()

    def stats(self) -> dict[str, Any]:
        return {"name": self.name, "entries": len(self._d), "peak_entries": self.peak_entries,
                "evictions": self.evictions, "cache_gb": round(self._bytes / _GIB, 2),
                "max_gb": round(self.max_bytes / _GIB, 1), "floor_gb": round(self.floor / _GIB, 1),
                "disk_reserve_gb": round(self.disk_reserve / _GIB, 1)}

    def __len__(self) -> int:
        return len(self._d)


def _selftest() -> dict[str, Any]:
    c = PressureAwareCache("selftest", floor_gb=1e9, disk_reserve_gb=1e9, min_entries=2, check_every=1)
    # floor set impossibly high -> always pressured -> should hold only min_entries
    for i in range(20):
        c.put(i, i)
    assert len(c) == 2, len(c)
    assert c.get(19) == 19  # newest retained
    c2 = PressureAwareCache("selftest2", floor_gb=0.0, disk_reserve_gb=0.0, min_entries=2)
    for i in range(50):
        c2.put(i, i)
    assert len(c2) == 50, len(c2)  # no pressure -> keep all
    return {"ok": True, "pressured_kept": len(c), "unpressured_kept": len(c2),
            "avail_ram_gb": round(available_ram_bytes() / _GIB, 1)}


if __name__ == "__main__":
    import json
    print(json.dumps(_selftest(), indent=2))
