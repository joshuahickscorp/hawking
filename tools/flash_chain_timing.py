#!/usr/bin/env python3
"""Decompose a completed Flash chain without inventing unmeasured buckets.

The layer receipts expose GPU/graph/oracle/input timers, but they do not yet
time every file or receipt operation.  This report keeps those fields separate
and labels the remaining wall interval as unattributed ceremony.  It is the
baseline for the persistent executor and is intentionally not a performance
promotion receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def ns(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("chain", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    summary = json.loads((args.chain / "CHAIN_SUMMARY.json").read_text())
    rows = []
    for entry in summary.get("layers", []):
        receipt_path = args.chain.parent.parent.parent / entry["receipt"]
        if not receipt_path.is_file():
            receipt_path = args.chain / f"layer-{entry['layer']}" / "receipt.json"
        doc = json.loads(receipt_path.read_text())
        execution = doc.get("execution", {})
        gpu = sum(ns(v) for v in execution.get("graph_gpu_ns", []))
        graph_host = sum(ns(v) for v in execution.get("graph_host_ns", []))
        oracle = ns(execution.get("source_cpu_oracle_ns"))
        input_read = ns(execution.get("embedding_read_ns"))
        timing = doc.get("timing", {})
        source_load = ns(timing.get("source_load_ns"))
        device_prepare = ns(timing.get("device_prepare_ns"))
        graph_setup = ns(timing.get("graph_setup_ns"))
        warmup = ns(timing.get("warmup_ns"))
        state_fingerprint = ns(timing.get("state_fingerprint_ns"))
        state_write = ns(timing.get("state_write_ns"))
        elapsed = ns(doc.get("elapsed_wall_ns"))
        known = (input_read + oracle + graph_host + source_load + device_prepare
                 + graph_setup + warmup + state_fingerprint + state_write)
        rows.append({
            "layer": entry.get("layer"),
            "layer_type": entry.get("layer_type"),
            "elapsed_wall_ns": elapsed,
            "source_payload_bytes_read": ns(doc.get("bytes", {}).get("source_payload_bytes_read")),
            "source_weight_bytes": ns(doc.get("bytes", {}).get("source_layer_weight_bytes")),
            "embedding_read_ns": input_read,
            "source_cpu_oracle_ns": oracle,
            "graph_host_ns": graph_host,
            "source_load_ns": source_load,
            "device_prepare_ns": device_prepare,
            "graph_setup_ns": graph_setup,
            "warmup_ns": warmup,
            "state_fingerprint_ns": state_fingerprint,
            "state_write_ns": state_write,
            "gpu_execution_ns": gpu,
            "dispatches": ns(execution.get("dispatches")),
            "command_buffers": sum(ns(v) for v in execution.get("command_buffers_per_rep", [])),
            "unattributed_wall_ns": max(0, elapsed - known),
            "unattributed_includes": [
                "source weight open/read/parse/index",
                "host to GPU weight preparation/upload",
                "pipeline creation",
                "command-buffer wait not separable from graph_host_ns",
                "state serialization/reload",
                "receipt writing",
            ],
        })

    def total(key: str) -> int:
        return sum(ns(row.get(key)) for row in rows)

    total_wall = ns(summary.get("elapsed_wall_ns"))
    report = {
        "schema": "hawking.flash.chain_timing_decomposition.v1",
        "chain": str(args.chain),
        "status": "MEASURED_BASELINE",
        "claim_boundary": "post-hoc decomposition of completed layer receipts; uninstrumented buckets remain explicitly unattributed",
        "process_boundary": summary.get("process_boundary"),
        "state_handoff": summary.get("state_handoff"),
        "layers": rows,
        "totals": {
            "layers": len(rows),
            "elapsed_wall_ns": total_wall,
            "source_payload_bytes_read": total("source_payload_bytes_read"),
            "source_weight_bytes": total("source_weight_bytes"),
            "embedding_read_ns": total("embedding_read_ns"),
            "source_cpu_oracle_ns": total("source_cpu_oracle_ns"),
            "graph_host_ns": total("graph_host_ns"),
            "source_load_ns": total("source_load_ns"),
            "device_prepare_ns": total("device_prepare_ns"),
            "graph_setup_ns": total("graph_setup_ns"),
            "warmup_ns": total("warmup_ns"),
            "state_fingerprint_ns": total("state_fingerprint_ns"),
            "state_write_ns": total("state_write_ns"),
            "gpu_execution_ns": total("gpu_execution_ns"),
            "unattributed_wall_ns": max(
                0,
                total_wall
                - sum(total(key) for key in (
                    "embedding_read_ns", "source_cpu_oracle_ns", "graph_host_ns",
                    "source_load_ns", "device_prepare_ns", "graph_setup_ns",
                    "warmup_ns", "state_fingerprint_ns", "state_write_ns",
                )),
            ),
            "dispatches": total("dispatches"),
            "command_buffers": total("command_buffers"),
        },
        "derived": {
            "elapsed_hours": total_wall / 3_600_000_000_000 if total_wall else None,
            "median_layer_elapsed_ns": median([r["elapsed_wall_ns"] for r in rows]) if rows else None,
            "median_layer_gpu_ns": median([r["gpu_execution_ns"] for r in rows]) if rows else None,
            "source_weight_read_fraction_of_payload": "not inferable from receipt timers",
            "fast_path_requirement": "instrument persistent executor with separate source/index/upload/pipeline/state/receipt timers",
        },
        "next": "Use one cached SourceBf16Index, one MetalContext, reusable scratch, device-resident state, and L0/L1 fingerprints; deep parity only at species checkpoints.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["totals"], indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
