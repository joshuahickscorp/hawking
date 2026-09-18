#!/usr/bin/env python3
"""Fail closed before retaining a bounded Flash compact bank in UMA.

This is intentionally model- and receipt-bound.  It does not start a model,
claim a resident runtime, or infer unified-memory aliasing.  It only decides
whether the next Flash resident-body experiment has enough observed headroom
for the conservative host-plus-Metal copy budget of its exact compact receipt.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _run(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def _integer_from(text: str, pattern: str) -> int:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"missing system measurement: {pattern}")
    return int(match.group(1))


def assess(
    session: dict[str, Any], *, physical_bytes: int, free_percent: int, swap_used_mb: float,
    headroom_bytes: int,
) -> dict[str, Any]:
    execution = session.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("session omitted execution accounting")
    source_bytes = execution.get("source_payload_bytes_read")
    if not isinstance(source_bytes, int) or source_bytes <= 0:
        raise ValueError("session omitted source payload bytes")
    if execution.get("expert_bank_mode") != "route_union_compact_teacher_bound":
        raise ValueError("only the teacher-bound compact Flash control may enter this preflight")
    state = execution.get("state_memory")
    if not isinstance(state, dict):
        raise ValueError("session omitted persistent state accounting")
    state_bytes = state.get("total_persistent_bytes")
    if not isinstance(state_bytes, int) or state_bytes <= 0:
        raise ValueError("session omitted total persistent state bytes")

    # Source tensor bytes and Metal buffers are separately allocated by the
    # current executor.  Bill both until a future native owner demonstrates
    # shared/aliased storage; UMA alone is not proof of aliasing.
    conservative_model_bytes = source_bytes * 2
    required_bytes = conservative_model_bytes + state_bytes + headroom_bytes
    observed_available_bytes = physical_bytes * free_percent // 100
    admitted = (
        required_bytes <= physical_bytes
        and required_bytes <= observed_available_bytes
        and swap_used_mb <= 128.0
    )
    return {
        "schema": "hawking.flash.compact_residency_preflight.v1",
        "status": "ADMISSIBLE_CONSERVATIVE_COMPACT_RESIDENT_EXPERIMENT" if admitted else "WITHHELD_INSUFFICIENT_HEADROOM",
        "model": session.get("model"),
        "pinned_revision": session.get("pinned_revision"),
        "teacher_bound_only": True,
        "observed_system": {
            "physical_bytes": physical_bytes,
            "memory_free_percent": free_percent,
            "observed_available_bytes_hint": observed_available_bytes,
            "swap_used_mb": swap_used_mb,
        },
        "budget": {
            "compact_source_weight_bytes": source_bytes,
            "conservative_host_plus_metal_weight_bytes": conservative_model_bytes,
            "persistent_state_bytes": state_bytes,
            "reserved_headroom_bytes": headroom_bytes,
            "required_bytes": required_bytes,
        },
        "claim_boundary": "A preflight admission is not a successful resident allocation, source aliasing proof, warm decode measurement, TPS, capability, EBPW, or promotion result.",
        "next": "Allocate only the exact teacher-bound compact bank under one Flash process, record actual process/UMA state after allocation, and fail closed before any second model provider is started.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--headroom-gib", type=float, default=8.0)
    args = parser.parse_args(argv)
    if args.headroom_gib <= 0:
        raise SystemExit("--headroom-gib must be positive")
    session = json.loads(args.session.read_text())
    physical_bytes = int(_run("sysctl", "-n", "hw.memsize").strip())
    pressure = _run("memory_pressure")
    free_percent = _integer_from(pressure, r"memory free percentage:\s*(\d+)%")
    swap = _run("sysctl", "vm.swapusage")
    match = re.search(r"used\s*=\s*([0-9.]+)M", swap)
    if not match:
        raise SystemExit("unable to parse vm.swapusage")
    doc = assess(
        session,
        physical_bytes=physical_bytes,
        free_percent=free_percent,
        swap_used_mb=float(match.group(1)),
        headroom_bytes=int(args.headroom_gib * 1024**3),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"status": doc["status"], "budget": doc["budget"], "out": str(args.out)}, indent=2))
    return 0 if doc["status"].startswith("ADMISSIBLE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
