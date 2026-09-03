"""Host-memory admission for Qwen3.8 children. Refuses before swap.

Measured 2026-08-16 on this box (96 GB unified), four process children of
`qwen38-27b/uniform-q4-v1` at `--max-seq-len 2048`, spawned together:

    sum RSS 35.09 GB = 8.77 GB/child
    machine-wide 40.67 GB = 10.2 GB/child
    free at the load spike 0.37 GB

Artifact pages are not shared. The process-pool is the fallback, with
staggered spawn. The primary design is one process / N sessions.
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import Any, Mapping

SCHEMA = "hawking.qwen38.host_admission.v1"
VERDICT_ADMIT = "admit"
VERDICT_REFUSE = "refuse"

# Decimal GB as written in the 2026-08-16 measurement, not a re-derivation.
MEASURED_FOUR_CHILD_SUM_RSS_BYTES = 35_090_000_000
MEASURED_FOUR_CHILD_MACHINE_BYTES = 40_670_000_000
MEASURED_PROCESS_CHILD_RSS_BYTES = MEASURED_FOUR_CHILD_SUM_RSS_BYTES // 4
MEASURED_PROCESS_CHILD_MACHINE_BYTES = MEASURED_FOUR_CHILD_MACHINE_BYTES // 4
MEASURED_FREE_AT_FOUR_CHILD_LOAD_SPIKE_BYTES = 370_000_000
MEASURED_FREE_ONCE_WARM_BYTES = 5_110_000_000
MEASURED_FREE_BASELINE_BEFORE_BYTES = 45_780_000_000
MEASURED_SINGLE_CHILD_SEQ8192_RSS_BYTES = 15_500_000_000

DEFAULT_RESERVE_BYTES = 4 * 1024 * 1024 * 1024
SESSION_HOST_OVERHEAD_BYTES = 64 * 1024 * 1024
GPU_LOCK = "/tmp/hawking-gpu-lane.lock"


def parse_vm_stat(text: str) -> dict[str, int]:
    match = re.search(r"page size of (\d+)", text)
    if not match:
        raise ValueError("vm_stat missing page size")
    page_size = int(match.group(1))

    def pages(label: str, default: int | None = None) -> int:
        found = re.search(rf"{re.escape(label)}\s+([0-9,]+)\.", text)
        if not found:
            if default is not None:
                return default
            raise ValueError(f"vm_stat missing {label}")
        return int(found.group(1).replace(",", ""))

    free = pages("Pages free:")
    return {
        "page_size_bytes": page_size,
        "pages_free": free,
        "pages_purgeable": pages("Pages purgeable:", 0),
        "pages_speculative": pages("Pages speculative:", 0),
        "free_bytes": free * page_size,
    }


def host_memory_snapshot() -> dict[str, int]:
    raw = subprocess.check_output(["vm_stat"], text=True)
    snap = parse_vm_stat(raw)
    try:
        mem = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        snap["memsize_bytes"] = int(mem.strip())
    except (subprocess.CalledProcessError, ValueError):
        snap["memsize_bytes"] = 96_000_000_000
    return snap


def process_rss_bytes(pid: int) -> int:
    raw = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True)
    return int(raw.strip()) * 1024


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def gpu_lock_held() -> dict[str, Any]:
    pid_path = os.path.join(GPU_LOCK, "pid")
    owner_path = os.path.join(GPU_LOCK, "owner")
    if not os.path.isdir(GPU_LOCK):
        return {"held": False, "owner": None, "pid": None, "owner_alive": False}
    pid = None
    if os.path.isfile(pid_path):
        try:
            pid = int(open(pid_path, encoding="utf-8").read().strip())
        except ValueError:
            pid = None
    owner = None
    if os.path.isfile(owner_path):
        owner = open(owner_path, encoding="utf-8").read().strip() or None
    alive = pid_alive(pid) if pid is not None else False
    return {
        "held": bool(alive),
        "owner": owner,
        "pid": pid,
        "owner_alive": alive,
    }


def process_pool_child_cost_bytes(max_seq_len: int) -> int:
    if max_seq_len <= 2048:
        return MEASURED_PROCESS_CHILD_MACHINE_BYTES
    if max_seq_len >= 8192:
        return MEASURED_SINGLE_CHILD_SEQ8192_RSS_BYTES
    span = 8192 - 2048
    t = max_seq_len - 2048
    lo = MEASURED_PROCESS_CHILD_MACHINE_BYTES
    hi = MEASURED_SINGLE_CHILD_SEQ8192_RSS_BYTES
    return lo + (hi - lo) * t // span


def shared_session_attach_cost_bytes(workspace_bytes: int) -> int:
    return int(workspace_bytes) + SESSION_HOST_OVERHEAD_BYTES


def decide_admission(
    memory: Mapping[str, int],
    *,
    label: str,
    cost_bytes: int,
    kind: str,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
) -> dict[str, Any]:
    free = int(memory["free_bytes"])
    cost = int(cost_bytes)
    if cost > free:
        return {
            "schema": SCHEMA,
            "verdict": VERDICT_REFUSE,
            "reason": f"cost {cost} B exceeds free {free} B; refusing before swap",
            "request": {"label": label, "cost_bytes": cost, "kind": kind},
            "free_before_bytes": free,
            "reserve_bytes": reserve_bytes,
            "free_after_if_admitted_bytes": None,
            "would_breach_reserve": True,
        }
    after = free - cost
    if after < reserve_bytes:
        return {
            "schema": SCHEMA,
            "verdict": VERDICT_REFUSE,
            "reason": (
                f"free after request would be {after} B < reserve "
                f"{reserve_bytes} B; refusing before swap"
            ),
            "request": {"label": label, "cost_bytes": cost, "kind": kind},
            "free_before_bytes": free,
            "reserve_bytes": reserve_bytes,
            "free_after_if_admitted_bytes": after,
            "would_breach_reserve": True,
        }
    return {
        "schema": SCHEMA,
        "verdict": VERDICT_ADMIT,
        "reason": (
            f"free {free} B - cost {cost} B leaves {after} B "
            f"(>= reserve {reserve_bytes} B)"
        ),
        "request": {"label": label, "cost_bytes": cost, "kind": kind},
        "free_before_bytes": free,
        "reserve_bytes": reserve_bytes,
        "free_after_if_admitted_bytes": after,
        "would_breach_reserve": False,
    }


def prove_refuse(memory: Mapping[str, int] | None = None) -> dict[str, Any]:
    snap = dict(memory) if memory is not None else host_memory_snapshot()
    return decide_admission(
        snap,
        label="prove-refuse",
        cost_bytes=int(snap["free_bytes"]) + 1,
        kind="oversub_demo",
    )
