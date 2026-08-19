#!/usr/bin/env python3
"""Deterministic memory-admission gate for concurrent Odyssey model lanes.

The box is an M3 Ultra with 96 GiB. Ordinary SPECIMEN experiments may run in
parallel while projected swap stays <= SWAP_MAX_GIB (default 30). clean_room is
ONLY for protected TPS timing and needs an exclusive model worker.

    projected_swap = swap_used
                   + max(0, (wired + in_flight + est) - (physical_ram - reserve))

stdlib only. No network, no model. Policy override: ODYSSEY_POLICY.json
`detachment.memory` if present.

    python3 tools/odyssey_memgate.py --self-check
    python3 tools/odyssey_memgate.py
    python3 tools/odyssey_memgate.py --admit 16 --in-flight 16
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "workspace/campaign/odyssey/ODYSSEY_POLICY.json"

# M3 Ultra sealed constants. SWAP_MAX_GIB is the default; policy may override
# the *effective* cap via detachment.memory without rewriting this constant.
PHYSICAL_RAM_GIB = 96.0
RESERVE_GIB = 12.0  # OS / Metal / build headroom
SWAP_MAX_GIB = 30
DEFAULT_EST_GIB = 16.0  # typical 4-bit MoE if the caller does not know
FREE_RAM_RESERVE_GIB = 2.0  # refuse when the box is already page-starved
_CAPACITY_HARD_CAP = 256
_OBS_KEYS = (
    "free_ram_gib",
    "wired_gib",
    "compressor_gib",
    "swap_used_gib",
    "swap_total_gib",
    "cpu_load",
)
_PAGE_RE = re.compile(r"page size of (\d+)", re.I)
_SWAP_RE = re.compile(
    r"(total|used|free)\s*=\s*([\d.]+)\s*([KMGT])i?B?", re.I
)
# vm_stat writes "Pages free: 123." (trailing period); memory_pressure omits it.
_COUNT_RE = re.compile(r'^["\']?([^:"\']+)["\']?\s*:\s*(\d+)\s*\.?\s*$')

_INJECTED: dict[str, float] | None = None

__all__ = [
    "PHYSICAL_RAM_GIB",
    "RESERVE_GIB",
    "SWAP_MAX_GIB",
    "DEFAULT_EST_GIB",
    "FREE_RAM_RESERVE_GIB",
    "observe",
    "admit",
    "capacity",
    "using_snapshot",
    "effective_swap_max_gib",
]


def _run(cmd: list[str], timeout: float = 8.0) -> str:
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return r.stdout or ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _pages_to_gib(pages: float, page_bytes: int) -> float:
    return float(pages) * float(page_bytes) / float(1024 ** 3)


def _swap_to_gib(value: float, unit: str) -> float:
    u = unit.upper()
    if u == "K":
        return value / 1024.0 / 1024.0
    if u == "M":
        return value / 1024.0
    if u == "G":
        return value
    if u == "T":
        return value * 1024.0
    return value / 1024.0  # macOS default is megabytes


def _parse_page_size(*blobs: str) -> int:
    for blob in blobs:
        m = _PAGE_RE.search(blob or "")
        if m:
            n = int(m.group(1))
            if n > 0:
                return n
    raw = _run(["sysctl", "-n", "hw.pagesize"]).strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return 16384


def _parse_page_counts(blob: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in (blob or "").splitlines():
        m = _COUNT_RE.match(line.strip())
        if not m:
            continue
        out[m.group(1).strip()] = int(m.group(2))
    return out


def _parse_swap(blob: str) -> tuple[float, float]:
    found: dict[str, float] = {}
    for kind, num, unit in _SWAP_RE.findall(blob or ""):
        found[kind.lower()] = _swap_to_gib(float(num), unit)
    return found.get("used", 0.0), found.get("total", 0.0)


def _policy_memory() -> Any:
    if not POLICY_PATH.is_file():
        return None
    try:
        data = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    det = data.get("detachment")
    if not isinstance(det, dict):
        return None
    return det.get("memory", None)


def effective_swap_max_gib() -> float:
    """SWAP_MAX_GIB, or detachment.memory from ODYSSEY_POLICY.json if present."""
    mem = _policy_memory()
    if mem is None:
        return float(SWAP_MAX_GIB)
    if isinstance(mem, bool):
        return float(SWAP_MAX_GIB)
    if isinstance(mem, (int, float)):
        return float(mem)
    if isinstance(mem, str):
        try:
            return float(mem.strip())
        except ValueError:
            return float(SWAP_MAX_GIB)
    if isinstance(mem, dict):
        for key in (
            "swap_max_gib",
            "SWAP_MAX_GIB",
            "swap_max",
            "max_swap_gib",
            "max_gib",
        ):
            val = mem.get(key)
            if isinstance(val, bool):
                continue
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                try:
                    return float(val.strip())
                except ValueError:
                    continue
    return float(SWAP_MAX_GIB)


def _normalize_snap(snap: dict[str, Any]) -> dict[str, float]:
    out = {k: 0.0 for k in _OBS_KEYS}
    for k in _OBS_KEYS:
        if k in snap and snap[k] is not None:
            out[k] = float(snap[k])
    return out


@contextmanager
def using_snapshot(snap: dict[str, Any]) -> Iterator[dict[str, float]]:
    """Make observe/admit/capacity read `snap` instead of the live box."""
    global _INJECTED
    prev = _INJECTED
    normalized = _normalize_snap(snap)
    _INJECTED = normalized
    try:
        yield normalized
    finally:
        _INJECTED = prev


def _observe_live() -> dict[str, float]:
    vm = _run(["vm_stat"])
    # Bare invocation prints current stats and exits. Never pass -l/-p/<pages>
    # — those allocate or hold pressure.
    mp = _run(["memory_pressure"])
    sw = _run(["sysctl", "-n", "vm.swapusage"]) or _run(["sysctl", "vm.swapusage"])
    page = _parse_page_size(vm, mp)
    vm_c = _parse_page_counts(vm)
    mp_c = _parse_page_counts(mp)

    def pages(*names: str) -> int:
        for name in names:
            if name in vm_c:
                return vm_c[name]
            if name in mp_c:
                return mp_c[name]
        return 0

    swap_used, swap_total = _parse_swap(sw)
    try:
        load = float(os.getloadavg()[0])
    except (OSError, IndexError):
        load = 0.0
    return {
        "free_ram_gib": _pages_to_gib(pages("Pages free"), page),
        "wired_gib": _pages_to_gib(pages("Pages wired down"), page),
        "compressor_gib": _pages_to_gib(
            pages("Pages occupied by compressor", "Pages used by compressor"),
            page,
        ),
        "swap_used_gib": float(swap_used),
        "swap_total_gib": float(swap_total),
        "cpu_load": load,
    }


def observe() -> dict[str, float]:
    """Live (or injected) macOS memory snapshot. Six keys, all floats."""
    if _INJECTED is not None:
        return dict(_INJECTED)
    return _observe_live()


def _projected_swap_gib(
    snap: dict[str, float], est_gib: float, in_flight_gib: float
) -> float:
    usable = PHYSICAL_RAM_GIB - RESERVE_GIB
    demand = snap["wired_gib"] + in_flight_gib + est_gib
    overflow = max(0.0, demand - usable)
    return snap["swap_used_gib"] + overflow


def _resolve_est(est_gib: float | None) -> float:
    if est_gib is None:
        return float(DEFAULT_EST_GIB)
    return float(est_gib)


def admit(
    est_gib: float | None = None,
    in_flight_gib: float = 0,
    clean_room: bool = False,
) -> dict[str, Any]:
    """GO or REFUSE a model lane of `est_gib` on top of `in_flight_gib`.

    est_gib defaults to DEFAULT_EST_GIB (~16, 4-bit MoE) when unknown.
    clean_room is exclusive: any in-flight model worker is a REFUSE.
    """
    est = _resolve_est(est_gib)
    inflight = float(in_flight_gib or 0.0)
    snap = observe()
    proj = _projected_swap_gib(snap, max(est, 0.0), max(inflight, 0.0))
    swap_max = effective_swap_max_gib()
    free = snap["free_ram_gib"]

    if est < 0 or inflight < 0:
        return {
            "decision": "REFUSE",
            "note": (
                f"REFUSE: est_gib={est:g} and in_flight_gib={inflight:g} "
                "must be >= 0"
            ),
            "projected_swap_gib": proj,
        }
    if clean_room and inflight > 0:
        return {
            "decision": "REFUSE",
            "note": (
                "REFUSE: clean_room requires exclusive timing "
                f"(another model worker is in flight, in_flight_gib={inflight:g}); "
                "clean_room is only for protected TPS timing, not ordinary "
                "SPECIMEN experiments"
            ),
            "projected_swap_gib": proj,
        }
    if proj > swap_max:
        return {
            "decision": "REFUSE",
            "note": (
                f"REFUSE: projected_swap={proj:.3f} GiB exceeds "
                f"SWAP_MAX_GIB={swap_max:g}"
            ),
            "projected_swap_gib": proj,
        }
    if free <= FREE_RAM_RESERVE_GIB:
        return {
            "decision": "REFUSE",
            "note": (
                f"REFUSE: free_ram={free:.3f} GiB is not above the "
                f"{FREE_RAM_RESERVE_GIB:g} GiB reserve"
            ),
            "projected_swap_gib": proj,
        }
    return {
        "decision": "GO",
        "note": (
            f"GO: projected_swap={proj:.3f} GiB <= {swap_max:g} GiB; "
            f"est={est:g} in_flight={inflight:g} "
            f"(physical={PHYSICAL_RAM_GIB:g} reserve={RESERVE_GIB:g})"
        ),
        "projected_swap_gib": proj,
    }


def capacity(est_gib_each: float | None) -> int:
    """How many concurrent lanes of this size fit under the swap bound now."""
    est = _resolve_est(est_gib_each)
    if est <= 0:
        return 0
    snap = observe()
    # Freeze the snapshot so sequential admits do not re-probe the box.
    n = 0
    with using_snapshot(snap):
        while n < _CAPACITY_HARD_CAP:
            verdict = admit(est, in_flight_gib=n * est, clean_room=False)
            if verdict["decision"] != "GO":
                break
            n += 1
    return n


def _self_check() -> int:
    # Canned macOS blobs — trailing period (vm_stat) vs none (memory_pressure).
    canned_vm = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                               1984439.\n"
        "Pages wired down:                          378651.\n"
        "Pages occupied by compressor:               80354.\n"
    )
    canned_mp = (
        "The system has 103079215104 (6291456 pages with a page size of 16384).\n"
        "Pages free: 1984868 \n"
        "Pages wired down: 378598 \n"
        "Pages used by compressor: 80354 \n"
        "System-wide memory free percentage: 92%\n"
    )
    assert _parse_page_size(canned_vm, canned_mp) == 16384
    vc = _parse_page_counts(canned_vm)
    assert vc["Pages free"] == 1984439, vc
    assert vc["Pages wired down"] == 378651, vc
    assert vc["Pages occupied by compressor"] == 80354, vc
    mc = _parse_page_counts(canned_mp)
    assert mc["Pages free"] == 1984868, mc
    used, total = _parse_swap(
        "vm.swapusage: total = 2048.00M  used = 949.31M  free = 1098.69M  (encrypted)"
    )
    assert abs(used - 949.31 / 1024.0) < 1e-9, used
    assert abs(total - 2.0) < 1e-9, total
    g_used, g_total = _parse_swap("total = 32.00G  used = 1.50G  free = 30.50G")
    assert abs(g_used - 1.5) < 1e-9 and abs(g_total - 32.0) < 1e-9

    # Live parser smoke test (no inject). Must not throw; keys must be numeric.
    live = _observe_live()
    for key in _OBS_KEYS:
        assert key in live, f"live observe missing {key}"
        assert isinstance(live[key], (int, float)), (key, live[key])
        assert live[key] >= 0, (key, live[key])
    assert live["swap_used_gib"] <= live["swap_total_gib"] + 1e-6, live

    # Policy reader must not crash; current file has no detachment.memory.
    cap = effective_swap_max_gib()
    assert cap == float(SWAP_MAX_GIB) or isinstance(cap, float), cap
    assert SWAP_MAX_GIB == 30

    low = {
        "free_ram_gib": 48.0,
        "wired_gib": 8.0,
        "compressor_gib": 0.25,
        "swap_used_gib": 0.25,
        "swap_total_gib": 2.0,
        "cpu_load": 1.2,
    }
    with using_snapshot(low) as snap:
        observed = observe()
        for key in _OBS_KEYS:
            assert observed[key] == snap[key], (key, observed, snap)

        # Several 16 GiB lanes GO under low swap.
        go_n = 0
        last_go: dict[str, Any] | None = None
        first_refuse: dict[str, Any] | None = None
        for k in range(0, 16):
            v = admit(16, in_flight_gib=k * 16.0)
            assert set(v) >= {"decision", "note", "projected_swap_gib"}
            expected = _projected_swap_gib(snap, 16.0, k * 16.0)
            assert abs(v["projected_swap_gib"] - expected) < 1e-9, (k, v, expected)
            if v["decision"] == "GO":
                go_n += 1
                last_go = v
                assert v["projected_swap_gib"] <= 30, v
            else:
                first_refuse = v
                break
        assert go_n >= 4, f"expected several 16 GiB GO lanes, got {go_n}"
        assert last_go is not None and last_go["decision"] == "GO"
        assert first_refuse is not None, "never hit the swap bound"
        assert first_refuse["decision"] == "REFUSE"
        assert first_refuse["projected_swap_gib"] > 30, first_refuse
        assert "SWAP_MAX" in first_refuse["note"] or "exceed" in first_refuse["note"]

        # Default est_gib is the 4-bit MoE 16 GiB.
        assert admit()["projected_swap_gib"] == admit(16)["projected_swap_gib"]
        assert admit(None)["decision"] == "GO"

        # clean_room is exclusive when another worker is in flight.
        cr_busy = admit(16, in_flight_gib=16.0, clean_room=True)
        assert cr_busy["decision"] == "REFUSE", cr_busy
        assert "clean_room" in cr_busy["note"]
        cr_free = admit(16, in_flight_gib=0, clean_room=True)
        assert cr_free["decision"] == "GO", cr_free
        # Ordinary SPECIMEN (clean_room=False) still GOes with others in flight.
        spec = admit(16, in_flight_gib=16.0, clean_room=False)
        assert spec["decision"] == "GO", spec

        # Explicit overshoot: a single huge lane.
        huge = admit(200.0, in_flight_gib=0.0)
        assert huge["decision"] == "REFUSE", huge
        assert huge["projected_swap_gib"] > 30, huge

        # Exact bound is GO (<= 30); a hair over is REFUSE.
        # overflow = (wired + in_flight + est) - 84; + swap_used = 30
        # 8 + 0 + est - 84 + 0.25 = 30  =>  est = 105.75
        edge = admit(105.75, in_flight_gib=0.0)
        assert edge["decision"] == "GO", edge
        assert abs(edge["projected_swap_gib"] - 30.0) < 1e-9, edge
        over = admit(105.76, in_flight_gib=0.0)
        assert over["decision"] == "REFUSE", over
        assert over["projected_swap_gib"] > 30, over

        # capacity() matches sequential admit() and is monotone in est size.
        c16 = capacity(16)
        assert c16 == go_n, (c16, go_n)
        sizes = [8.0, 16.0, 24.0, 32.0, 48.0, 64.0, 128.0]
        caps = [capacity(s) for s in sizes]
        for a, b, sa, sb in zip(caps, caps[1:], sizes, sizes[1:]):
            assert a >= b, f"capacity not monotone: {sa}->{a} vs {sb}->{b} ({caps})"
        assert caps[0] > caps[-1], caps
        assert capacity(0) == 0
        assert capacity(-1) == 0

        # Starved free_ram refuses even when the swap formula would pass.
        starved = dict(low)
        starved["free_ram_gib"] = 0.5
        with using_snapshot(starved):
            bad = admit(16, in_flight_gib=0)
            assert bad["decision"] == "REFUSE", bad
            assert "free_ram" in bad["note"]
            assert capacity(16) == 0

    print("odyssey_memgate self-check: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Odyssey memory-admission gate (swap-bounded multi-model)"
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="injected-snapshot assertions; no network, no model",
    )
    p.add_argument(
        "--admit",
        nargs="?",
        const=DEFAULT_EST_GIB,
        type=float,
        metavar="EST_GIB",
        help="admit a lane of EST_GIB (default 16)",
    )
    p.add_argument(
        "--in-flight",
        type=float,
        default=0.0,
        metavar="GIB",
        help="GiB already committed to in-flight model workers",
    )
    p.add_argument(
        "--clean-room",
        action="store_true",
        help="protected TPS timing: refuse if any other model worker is in flight",
    )
    p.add_argument(
        "--capacity",
        nargs="?",
        const=DEFAULT_EST_GIB,
        type=float,
        metavar="EST_GIB",
        help="how many concurrent lanes of EST_GIB fit now",
    )
    args = p.parse_args(argv)

    if args.self_check:
        return _self_check()

    snap = observe()
    payload: dict[str, Any] = {"observe": snap, "swap_max_gib": effective_swap_max_gib()}
    if args.admit is not None:
        payload["admit"] = admit(
            args.admit, in_flight_gib=args.in_flight, clean_room=args.clean_room
        )
    if args.capacity is not None:
        payload["capacity"] = capacity(args.capacity)
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if args.admit is not None and payload["admit"]["decision"] == "REFUSE":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
