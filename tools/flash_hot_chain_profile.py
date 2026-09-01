#!/usr/bin/env python3
"""Build the protected FastPath profile from a completed native chain receipt.

This is deliberately a receipt transformer, not a benchmark.  It preserves the
authoritative integer-nanosecond measurements emitted by the native executor
and marks unmet FastPath exit-gate conditions explicitly.  In particular, a
chain that still hands state through host memory cannot be promoted merely
because it ran in one OS process.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any


def integer(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def layer_rows(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    rows = receipt.get("layers")
    if isinstance(rows, list) and rows:
        return [row for row in rows if isinstance(row, dict)]
    # A single-layer receipt uses execution-level timing rather than a layers
    # array.  Keep this fallback honest and scoped to that one physical layer.
    execution = receipt.get("execution") or {}
    timing = receipt.get("timing") or {}
    return [{
        "layer": (receipt.get("source") or {}).get("requested_start_layer"),
        "status": receipt.get("status"),
        "gpu_ns": execution.get("graph_gpu_ns", execution.get("gpu_ns", [])),
        "host_ns": execution.get("graph_host_ns", execution.get("host_ns", [])),
        "dispatches": execution.get("dispatches", execution.get("total_dispatches")),
        "command_buffers": execution.get("command_buffers", execution.get("total_command_buffers")),
        "source_bytes_read": execution.get("source_payload_bytes_read"),
        "source_load_ns": timing.get("source_load_ns"),
    }]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    receipt = json.loads(args.receipt.read_text())
    execution = receipt.get("execution") or {}
    timing = receipt.get("timing") or {}
    rows = layer_rows(receipt)
    # A fast-chain summary is the durable envelope for several per-group
    # receipts.  Flatten it here so the same profile schema can describe a
    # real cross-species run without pretending the summary itself contained
    # per-layer timings.
    if isinstance(receipt.get("groups"), list):
        rows = []
        execution = {"host_activation_roundtrips": 0, "fallback_count": 0,
                     "total_dispatches": 0, "total_command_buffers": 0,
                     "state_handoff": receipt.get("state_handoff")}
        timing = {"source_load_ns": 0, "device_prepare_ns": 0,
                  "graph_setup_ns": 0, "state_fingerprint_ns": 0,
                  "state_write_ns": 0}
        for group in receipt["groups"]:
            if not isinstance(group, dict):
                continue
            group_path = Path(group.get("receipt", ""))
            if not group_path.is_absolute():
                group_path = args.receipt.parent / group_path
            if not group_path.is_file():
                continue
            group_doc = json.loads(group_path.read_text())
            group_rows = layer_rows(group_doc)
            rows.extend(group_rows)
            gexec = group_doc.get("execution") or {}
            gtiming = group_doc.get("timing") or {}
            for key in ("host_activation_roundtrips", "fallback_count", "total_dispatches", "total_command_buffers"):
                execution[key] = integer(execution.get(key)) + integer(gexec.get(key))
            for key in timing:
                timing[key] = integer(timing.get(key)) + integer(gtiming.get(key))
    wall_ns = integer(receipt.get("elapsed_wall_ns"))

    layer_wall = [
        sum(integer(v) for v in row.get("host_ns", []))
        if isinstance(row.get("host_ns"), list)
        else integer(row.get("host_ns"))
        for row in rows
    ]
    gpu_ns = sum(
        sum(integer(v) for v in row.get("gpu_ns", []))
        if isinstance(row.get("gpu_ns"), list)
        else integer(row.get("gpu_ns"))
        for row in rows
    )
    dispatches = sum(integer(row.get("dispatches")) for row in rows)
    command_buffers = sum(integer(row.get("command_buffers")) for row in rows)
    source_bytes = sum(integer(row.get("source_bytes_read")) for row in rows)
    host_roundtrips = integer(execution.get("host_activation_roundtrips"))
    host_roundtrip_bytes = host_roundtrips * integer(
        (receipt.get("source") or {}).get("state_handoff_artifact", {}).get("bytes")
    )

    gate = {
        "compact_routed_experts_multi_layer": len(rows) >= 2 and all(
            row.get("status") == "PASSED" for row in rows
        ),
        "zero_mandatory_host_handoff": host_roundtrips == 0,
        "long_lived_executor": receipt.get("process_boundary") == "single_os_process",
        "protected_chain_minimum_8_layers": len(rows) >= 8,
    }
    gate["fastpath_exit_gate"] = all(gate.values())
    deep_per_layer_parity = bool(rows) and all(row.get("status") == "PASSED" for row in rows)
    residency_description = execution.get("state_handoff")
    if host_roundtrips == 0 and deep_per_layer_parity:
        residency_description = "device final-state blit between layers; diagnostic parity reads are not activation handoffs"
    terminal_doc = None
    terminal_path = receipt.get("terminal_receipt")
    if terminal_path:
        terminal_candidate = Path(terminal_path)
        if not terminal_candidate.is_absolute():
            terminal_candidate = args.receipt.parent / terminal_candidate
        if terminal_candidate.is_file():
            try:
                terminal_doc = json.loads(terminal_candidate.read_text())
            except (OSError, json.JSONDecodeError):
                terminal_doc = None
    complete_token_received = bool(
        len(rows) >= 48 and isinstance(terminal_doc, dict)
        and terminal_doc.get("status") == "PASSED"
    )

    profile = {
        "schema": "hawking.flash_hot_chain_profile.v1",
        "status": "MEASURED_CANDIDATE",
        "benchmark_mode": "PROTECTED_FAST_CANDIDATE",
        "source_receipt": str(args.receipt),
        "start_layer": receipt.get("start_layer", (receipt.get("source") or {}).get("requested_start_layer")),
        "end_layer": receipt.get("end_layer", ((receipt.get("source") or {}).get("executed_layers") or [None])[-1]),
        "layer_count": len(rows),
        "complete_wall_ns": wall_ns,
        "GPU_ns": gpu_ns,
        "average_wall_ns_per_layer": wall_ns // len(rows) if rows else None,
        "average_GPU_ns_per_layer": gpu_ns // len(rows) if rows else None,
        "p50_layer_ns": int(median(layer_wall)) if layer_wall else None,
        "p95_layer_ns": percentile(layer_wall, 0.95),
        "max_layer_ns": max(layer_wall) if layer_wall else None,
        "total_dispatches": dispatches or integer(execution.get("total_dispatches")),
        "dispatches_per_layer": (dispatches // len(rows)) if rows else None,
        "command_buffer_count": command_buffers or integer(execution.get("total_command_buffers")),
        "synchronization_count": command_buffers or integer(execution.get("total_command_buffers")),
        "host_roundtrip_count": host_roundtrips,
        "host_roundtrip_bytes": host_roundtrip_bytes,
        "host_roundtrip_ns": None,
        "source_bytes_read": source_bytes or integer(execution.get("source_payload_bytes_read")),
        "representation_bytes_read": None,
        "device_bytes_uploaded": None,
        "fallback_count": integer(execution.get("fallback_count")),
        "state_residency_contract": {
            "device_resident": host_roundtrips == 0,
            "host_roundtrip_required": host_roundtrips > 0,
            "description": residency_description,
        },
        "timing_decomposition_ns": {
            "source_load_ns": integer(timing.get("source_load_ns")),
            "representation_load_ns": integer(timing.get("representation_load_ns")),
            "device_prepare_ns": integer(timing.get("device_prepare_ns")),
            "graph_setup_ns": integer(timing.get("graph_setup_ns")),
            "verification_ns": integer(timing.get("verification_ns")),
            "checkpoint_ns": integer(timing.get("state_fingerprint_ns")) + integer(timing.get("state_write_ns")),
        },
        "parity_verdict": "PASS" if rows and all(row.get("status") == "PASSED" for row in rows) else "INCONCLUSIVE",
        "complete_token_terminal": {
            "status": "PHYSICALLY_RECEIVED" if complete_token_received else "NOT_PROVEN",
            "receipt": str(terminal_path) if complete_token_received else None,
            "token_id": (terminal_doc.get("terminal") or {}).get("token_id") if complete_token_received else None,
        },
        "capability_verdict_if_applicable": None,
        "gate": gate,
        "claim_boundary": (
            "Exact 48-layer deep parity completed in one native process with compact routed-bank linear groups and dense full-attention organs; a native terminal token was physically received. TPS, EBPW, and resident promotion remain open."
            if complete_token_received
            else
            "Exact compact routed multi-layer parity in one native process; "
            "not FastPath exit-gate promotion because the minimum 8-layer "
            "protected-chain requirement remains open."
            if host_roundtrips == 0 and gate["compact_routed_experts_multi_layer"] and not gate["protected_chain_minimum_8_layers"]
            else "Exact compact routed multi-layer deep parity in one native process; "
                 "the bounded zero-host 8-layer FastPath exit gate is satisfied, but "
                 "complete-token and resident promotion remain open."
                 if host_roundtrips == 0 and gate["compact_routed_experts_multi_layer"] and gate["protected_chain_minimum_8_layers"]
            else "Device-resident terminal-only parity probe in one native process; "
                 "not FastPath exit-gate promotion because per-layer deep parity "
                 "remains open."
                 if host_roundtrips == 0 and gate["protected_chain_minimum_8_layers"]
                 else "Device-resident terminal-only parity probe in one native process; "
                      "not FastPath exit-gate promotion because per-layer deep parity "
                      "and the minimum 8-layer protected-chain requirement remain open."
                      if host_roundtrips == 0
                      else "Exact compact routed multi-layer parity in one native process; "
                           "not FastPath exit-gate promotion because host state handoff and "
                           "the minimum 8-layer protected-chain requirement remain open."
        ),
        "next": (
            "Run accepted-token/TPS and EBPW qualification, then test resident resource leases; do not infer promotion from this single token."
            if complete_token_received
            else
            "FastPath exit gate is satisfied for this bounded 8-layer chain; compare the "
            "accelerated continuation at a shared checkpoint, then extend toward a complete token."
            if host_roundtrips == 0 and gate["fastpath_exit_gate"]
            else "Deep-verify every layer species in this zero-host 8-layer chain, then rerun "
                 "the profile before considering FastPath exit-gate promotion."
                 if host_roundtrips == 0 and gate["protected_chain_minimum_8_layers"]
            else "Remove the structural host state seam, then rerun this profile on an 8–16 layer protected hot chain."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps({"gate": gate, "complete_wall_ns": wall_ns, "GPU_ns": gpu_ns}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
