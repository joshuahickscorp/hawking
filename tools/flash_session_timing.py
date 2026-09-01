#!/usr/bin/env python3
"""Extract measured latency buckets from a complete Flash session receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


def integer(value):
    return int(value) if isinstance(value, (int, float)) else 0


def load_receipt(path: Path, root: Path):
    candidate = path if path.is_file() else root / path
    return candidate, json.loads(candidate.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    session = json.loads(args.session.read_text())
    rows = []
    for segment in session.get("segments", []):
        if "steps" in segment:
            steps = segment.get("steps") or []
            wall = sum(integer(s.get("wall_ns")) for s in steps)
            gpu = sum(integer(s.get("gpu_ns")) for s in steps)
            dispatches = sum(integer(s.get("dispatches")) for s in steps)
            source_bytes = integer(segment.get("source_payload_bytes_read"))
            kind = "linear_attention"
            label = f"layers-{segment.get('layers', ['?', '?'])[0]}-{segment.get('layers', ['?', '?'])[1]}"
            receipt_path = None
        else:
            receipt_path = segment.get("receipt")
            if not receipt_path:
                continue
            resolved, nested = load_receipt(Path(receipt_path), root)
            execution = nested.get("execution") or {}
            steps = nested.get("steps") or []
            wall = integer(execution.get("wall_ns"))
            gpu = integer(execution.get("gpu_ns"))
            dispatches = sum(integer(s.get("dispatches")) for s in steps)
            source_bytes = integer(execution.get("source_payload_bytes_read"))
            kind = "full_attention"
            label = f"layer-{segment.get('layer', nested.get('layer', '?'))}"
            receipt_path = str(resolved)
        rows.append({
            "segment": label,
            "kind": kind,
            "wall_ns": wall,
            "gpu_ns": gpu,
            "host_and_unattributed_ns": max(0, wall - gpu),
            "dispatches": dispatches,
            "source_payload_bytes_read": source_bytes,
            "source_and_device_prepare_ns": integer((segment.get("timing") or {}).get("layer_source_and_device_prepare_ns")),
            "execution_wall_ns": integer((segment.get("timing") or {}).get("execution_wall_ns", wall)),
            "receipt": receipt_path,
        })
    total_wall = integer((session.get("execution") or {}).get("elapsed_wall_ns"))
    total_gpu = sum(r["gpu_ns"] for r in rows)
    total_dispatches = sum(r["dispatches"] for r in rows)
    report = {
        "schema": "hawking.flash.complete_session_timing_decomposition.v1",
        "status": "MEASURED_BASELINE",
        "session": str(args.session),
        "session_sha256": hashlib.sha256(args.session.read_bytes()).hexdigest(),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "process_boundary": (session.get("execution") or {}).get("process_boundary"),
        "segments": rows,
        "totals": {
            "elapsed_wall_ns": total_wall,
            "elapsed_hours": total_wall / 3_600_000_000_000 if total_wall else None,
            "measured_gpu_ns": total_gpu,
            "measured_host_and_unattributed_ns": max(0, total_wall - total_gpu),
            "dispatches": total_dispatches,
            "source_payload_bytes_read": sum(r["source_payload_bytes_read"] for r in rows),
            "source_and_device_prepare_ns": sum(r["source_and_device_prepare_ns"] for r in rows),
            "execution_wall_ns": sum(r["execution_wall_ns"] for r in rows),
        },
        "top_runtime_latencies": [
            {"rank": 1, "name": "dense expert-bank source/load ceremony", "status": "dominant_by_source_bytes", "evidence": "per-segment source payload bytes and wall time; exact open/read split not instrumented"},
            {"rank": 2, "name": "host and synchronization ceremony", "status": "measured_residual", "evidence": "segment wall_ns minus reported GPU ns"},
            {"rank": 3, "name": "dispatch/command submission", "status": "measured_count_only", "evidence": "per-step dispatch totals; dispatch ns not separately instrumented"},
        ],
        "claim_boundary": "Post-hoc timing baseline for the exact dense stateful oracle. Source open/read/parse, upload, pipeline, wait, serialization and receipt sub-buckets remain explicitly unattributed where the source receipt did not time them; this is not a TPS or residency claim.",
        "next": "Instrument the persistent route-safe executor with separate source/index/upload/pipeline/wait/state/receipt timers and compare complete accepted-token time.",
        "bench": {"state": "UNKNOWN", "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "recorded_by": "tools/flash_session_timing.py", "machine": "Apple M3 Ultra", "rule": "S032 §3 -- post-hoc baseline; no protected performance claim"},
    }
    report["seal_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
