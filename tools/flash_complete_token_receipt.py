#!/usr/bin/env python3
"""Derive a measured one-token Flash receipt from a completed native chain.

This is intentionally a receipt transformer: it never invents accepted-TPS or
EBPW from a single terminal argmax probe.  The complete-forward and terminal
wall/GPU timings remain integer nanoseconds and the terminal token is recorded
as physically received, not as a multi-token generation acceptance result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("terminal", type=Path)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text())
    terminal = json.loads(args.terminal.read_text())
    profile = json.loads(args.profile.read_text())
    if summary.get("status") != "PASSED" or summary.get("start_layer") != 0 or summary.get("end_layer") != 47:
        raise SystemExit("summary is not a passed 0..47 chain")
    if terminal.get("status") != "PASSED" or terminal.get("complete_token_runtime") != "FIRST_COMPLETE_TOKEN_TERMINAL_PROBE":
        raise SystemExit("terminal is not a passed first-token probe")
    token = (terminal.get("terminal") or {}).get("token_id")
    if not isinstance(token, int):
        raise SystemExit("terminal token id is missing")
    if profile.get("parity_verdict") != "PASS" or profile.get("layer_count") != 48:
        raise SystemExit("profile does not prove all 48 layers")

    forward_wall_ns = integer(summary.get("elapsed_wall_ns"))
    forward_gpu_ns = integer(profile.get("GPU_ns"))
    terminal_wall_ns = integer((terminal.get("execution") or {}).get("wall_ns"))
    terminal_gpu_ns = integer((terminal.get("execution") or {}).get("gpu_ns"))
    source_bytes = integer(profile.get("source_bytes_read"))
    doc = {
        "schema": "hawking.flash.complete_token_measurement.v1",
        "status": "MEASURED_SINGLE_TERMINAL_TOKEN",
        "model": summary.get("root"),
        "source_chain": str(args.summary),
        "terminal_receipt": str(args.terminal),
        "profile_receipt": str(args.profile),
        "representation": "source_bf16_exact_mixed_compact_linear_and_dense_full_attention_expert_banks",
        "execution": {
            "process_boundary": summary.get("process_boundary"),
            "device_resident": summary.get("device_resident") is True,
            "deep_verification": summary.get("deep_verification_enabled") is True,
            "host_activation_roundtrips": integer(profile.get("host_roundtrip_count")),
            "source_bytes_read": source_bytes,
            "forward_wall_ns": forward_wall_ns,
            "forward_gpu_ns": forward_gpu_ns,
            "terminal_wall_ns": terminal_wall_ns,
            "terminal_gpu_ns": terminal_gpu_ns,
            "complete_forward_plus_terminal_wall_ns": forward_wall_ns + terminal_wall_ns,
            "complete_forward_plus_terminal_gpu_ns": forward_gpu_ns + terminal_gpu_ns,
            "timing_unit": "integer nanoseconds",
        },
        "terminal_token": {
            "status": "PHYSICALLY_RECEIVED",
            "token_id": token,
            "sampling": (terminal.get("terminal") or {}).get("sampling"),
            "accepted_generation_tokens": None,
            "classification": "single terminal argmax probe; not multi-token accepted generation",
        },
        "flash_tps": None,
        "accepted_tps": None,
        "complete_system_ebpw": None,
        "promotion_allowed": False,
        "claim_boundary": "Measured complete 48-layer source-BF16 forward plus one native terminal token receipt. This does not establish accepted multi-token generation TPS, EBPW, capability parity, or HCLI residency.",
        "next": "Run a stateful multi-token decode with accepted-token accounting before populating TPS or EBPW.",
    }
    doc["seal_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"status": doc["status"], "token_id": token, "forward_wall_ns": forward_wall_ns, "forward_gpu_ns": forward_gpu_ns, "source_bytes_read": source_bytes}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
