#!/usr/bin/env python3
"""MachineState snapshot — the resource-class view the lane scheduler needs.

Per steer S001 (§5F): the ancestor of Machine Genome. Reports exactly the fields
this campaign has been reading by hand ~20x (load / free RAM / free disk / active
heavy lanes) so a resource-aware scheduler (§5C) and clean-box gate (§4) can query
one call instead of re-deriving. stdlib only; read-only; ~10ms; safe to call while
a GPU-heavy clean-TPS lane runs (it perturbs nothing measurable).

`clean_box_ok()` encodes the one rule the campaign learned the hard way: a clean
GPU/TPS measurement must not overlap another GPU_HEAVY or MEMORY_HEAVY lane.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=3
        ).stdout.strip() or None
    except Exception:
        return None


def _load_1m() -> float | None:
    try:
        return os.getloadavg()[0]
    except Exception:
        return None


def _ram_gib() -> tuple[float | None, float | None]:
    total = _sysctl("hw.memsize")
    total_gib = int(total) / 1024**3 if total and total.isdigit() else None
    free_gib = None
    try:
        # macOS: free ~= (free + inactive + speculative) pages * pagesize
        pagesize = int(_sysctl("hw.pagesize") or 16384)
        vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=3).stdout
        pages = {}
        for line in vm.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                v = v.strip().rstrip(".")
                if v.isdigit():
                    pages[k.strip()] = int(v)
        freeish = sum(
            pages.get(k, 0)
            for k in ("Pages free", "Pages inactive", "Pages speculative")
        )
        if freeish:
            free_gib = freeish * pagesize / 1024**3
    except Exception:
        pass
    return total_gib, free_gib


def _active_grok_lanes() -> list[str]:
    """Heavy lanes = worktrees with a LIVE grok process (best-effort).

    A worktree directory is not evidence of a running lane — nothing self-reaps,
    so finished lanes leave their directory behind. Counting directories made
    clean_box_ok permanently False once a few lanes had ever run (36 stale dirs
    observed), which would block every clean measurement forever. Match against
    the running process table instead.
    """
    wt = Path.home() / ".claude-grok" / "worktrees"
    if not wt.is_dir():
        return []
    names = [p.name for p in wt.iterdir() if p.is_dir()]
    if not names:
        return []
    try:
        ps = subprocess.run(
            ["ps", "-axo", "command"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return sorted(names)  # cannot tell: stay conservative
    return sorted(n for n in names if n in ps)


def snapshot() -> dict:
    total_gib, free_gib = _ram_gib()
    du = shutil.disk_usage(str(Path.home()))
    return {
        "type": "machine_state",
        "chip": _sysctl("machdep.cpu.brand_string") or _sysctl("hw.model"),
        "ncpu": os.cpu_count(),
        "ram_total_gib": round(total_gib, 1) if total_gib else None,
        "ram_free_gib": round(free_gib, 1) if free_gib is not None else None,
        "disk_free_gib": round(du.free / 1024**3, 1),
        "load_1m": _load_1m(),
        "active_grok_lanes": _active_grok_lanes(),
    }


def clean_box_ok(snap: dict | None = None, min_free_gib: float = 15.0) -> tuple[bool, str]:
    """A clean GPU/TPS measurement is safe only if no other heavy lane is live and
    disk is above the emergency floor. Conservative: any live worktree counts."""
    s = snap or snapshot()
    lanes = s.get("active_grok_lanes") or []
    if lanes:
        return False, f"{len(lanes)} live lane(s) may contend GPU/memory: {lanes}"
    if (s.get("disk_free_gib") or 0) < min_free_gib:
        return False, f"disk_free {s.get('disk_free_gib')} GiB < floor {min_free_gib}"
    if (s.get("load_1m") or 0) > (s.get("ncpu") or 1) * 0.5:
        return False, f"load_1m {s.get('load_1m')} high vs {s.get('ncpu')} cpu"
    return True, "no live lanes, disk ok, load ok"


if __name__ == "__main__":
    snap = snapshot()
    ok, why = clean_box_ok(snap)
    snap["clean_box_ok"] = ok
    snap["clean_box_reason"] = why
    # self-check: required keys present + types sane
    assert snap["disk_free_gib"] > 0 and snap["ncpu"], "machine_state read failed"
    assert isinstance(snap["active_grok_lanes"], list)
    json.dump(snap, sys.stdout, indent=2)
    sys.stdout.write("\n")
