#!/usr/bin/env python3.12
"""Resource scheduler for Odyssey stage apparatus.

Before a stage runs, the scheduler *declares* wall time, RSS, and threads.
It can also admit/deny against sandbox policy ceilings (max concurrent heavy
lanes, memory ceiling). Declaration is mandatory; admission is separate.
"""
from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path
from typing import Any

from tools.odyssey._paths import ODYSSEY

SCHEMA = "hawking.odyssey.resource_scheduler.v1"


def _rss_bytes() -> int:
    # ru_maxrss is bytes on macOS, KiB on Linux. Detect via platform.
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return int(usage)
    return int(usage) * 1024


def declare_resources(
    *,
    stage: str,
    wall_time_budget_s: float,
    rss_budget_bytes: int,
    threads: int,
    heavy: bool = False,
    note: str = "",
) -> dict[str, Any]:
    """Declare the resource envelope before a stage runs. Does not start work."""
    return {
        "schema": SCHEMA,
        "kind": "declaration",
        "stage": stage,
        "declared": {
            "wall_time_budget_s": float(wall_time_budget_s),
            "rss_budget_bytes": int(rss_budget_bytes),
            "threads": int(threads),
            "heavy": bool(heavy),
        },
        "observed_at_declaration": {
            "rss_bytes": _rss_bytes(),
            "wall_time_s": time.time(),
            "pid": os.getpid(),
        },
        "note": note or "resource declaration only; not a measurement of a heavy training step",
        "status": "DECLARED",
    }


def load_policy() -> dict[str, Any]:
    path = ODYSSEY / "sandbox" / "POLICY.json"
    return json.loads(path.read_text())


def admit(
    request: dict[str, Any],
    state: dict[str, Any] | None = None,
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Admit or deny a heavy lane based on sandbox policy ceilings."""
    policy = policy or load_policy()
    resources = policy.get("resources") or {}
    max_heavy = int(resources.get("max_concurrent_heavy_lanes", 1))
    mem_ceiling = int(resources.get("memory_ceiling_bytes", 103_079_215_104))
    state = dict(state or {"active_heavy": 0, "reserved_bytes": 0})
    need = int(request.get("memory_bytes", 0))
    heavy = bool(request.get("heavy", True))
    if heavy and state["active_heavy"] >= max_heavy:
        return {
            "schema": SCHEMA,
            "status": "RUNNABLE",
            "admit": False,
            "reason": "max_concurrent_heavy_lanes reached",
            "state": state,
        }
    if state["reserved_bytes"] + need > mem_ceiling:
        return {
            "schema": SCHEMA,
            "status": "RUNNABLE",
            "admit": False,
            "reason": "memory_ceiling_bytes exceeded",
            "state": state,
        }
    if heavy:
        state["active_heavy"] += 1
    state["reserved_bytes"] += need
    return {
        "schema": SCHEMA,
        "status": "RUNNABLE",
        "admit": True,
        "reason": "within ceilings",
        "state": state,
    }


def declare_and_admit(
    *,
    stage: str,
    wall_time_budget_s: float,
    rss_budget_bytes: int,
    threads: int,
    heavy: bool = False,
    memory_bytes: int = 0,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combined path: declare envelope, then admit against policy."""
    decl = declare_resources(
        stage=stage,
        wall_time_budget_s=wall_time_budget_s,
        rss_budget_bytes=rss_budget_bytes,
        threads=threads,
        heavy=heavy,
    )
    adm = admit(
        {"heavy": heavy, "memory_bytes": memory_bytes or rss_budget_bytes},
        state=state,
    )
    return {
        "schema": SCHEMA,
        "declaration": decl,
        "admission": adm,
        "status": "ADMITTED" if adm["admit"] else "DENIED",
    }
