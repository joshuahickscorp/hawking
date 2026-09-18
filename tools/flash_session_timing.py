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
    ap.add_argument(
        "--require-repeated-acceptance",
        action="store_true",
        help=(
            "refuse an ordinary forward receipt; require the exact two-or-more "
            "stateful reference checks before emitting a repeated-decode census"
        ),
    )
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    session = json.loads(args.session.read_text())
    session_execution = session.get("execution") or {}
    expert_bank_mode = session_execution.get("expert_bank_mode")
    route_safe_compact = expert_bank_mode == "route_union_compact_teacher_bound"
    reference_checks = (session.get("terminal") or {}).get("reference_checks") or []
    accepted_tokens = int(session.get("accepted_generation_tokens") or 0)
    repeated_accepted = (
        session.get("status") == "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE"
        and accepted_tokens >= 2
        and isinstance(reference_checks, list)
        and len(reference_checks) >= 2
        and all(
            isinstance(check, dict)
            and check.get("accepted") is True
            and check.get("expected_token_id") == check.get("predicted_token_id")
            for check in reference_checks
        )
        and session_execution.get("source_reset_or_reprefill") is False
    )
    if args.require_repeated_acceptance and not repeated_accepted:
        raise SystemExit(
            "session does not establish repeated accepted decode with persistent state"
        )
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
            nested_execution = nested.get("execution") or {}
            steps = nested.get("steps") or []
            dispatches = sum(integer(s.get("dispatches")) for s in steps)
            # Stateful attention receipts expose their hot-path timing per
            # token slot.  Their top-level execution object intentionally
            # contains state/accounting rather than a synthetic aggregate.
            wall = sum(integer(s.get("wall_ns")) for s in steps)
            gpu = sum(integer(s.get("gpu_ns")) for s in steps)
            source_bytes = integer(nested_execution.get("source_payload_bytes_read"))
            segment_timing = segment.get("timing") or {}
            nested_timing = nested_execution.get("timing") or {}
            kind = "full_attention"
            label = f"layer-{segment.get('layer', nested.get('layer', '?'))}"
            receipt_path = str(resolved)
        if kind == "linear_attention":
            segment_timing = segment.get("timing") or {}
            nested_timing = {}
        rows.append({
            "segment": label,
            "kind": kind,
            "wall_ns": wall,
            "gpu_ns": gpu,
            "host_and_unattributed_ns": max(0, wall - gpu),
            "dispatches": dispatches,
            "source_payload_bytes_read": source_bytes,
            "source_and_device_prepare_ns": (
                integer(segment_timing.get("layer_source_and_device_prepare_ns"))
                or integer(segment_timing.get("source_load_ns"))
                + integer(segment_timing.get("device_prepare_ns"))
            ),
            "execution_wall_ns": integer(
                segment_timing.get("execution_wall_ns", wall)
            ),
            "source_load_ns": integer(segment_timing.get("source_load_ns") or nested_timing.get("source_load_ns")),
            "oracle_ns": integer(segment_timing.get("oracle_ns") or nested_timing.get("oracle_ns")),
            "device_prepare_ns": integer(segment_timing.get("device_prepare_ns") or nested_timing.get("device_prepare_ns")),
            "graph_prepare_ns": integer(segment_timing.get("graph_prepare_ns")),
            "full_layer_call_wall_ns": integer(segment_timing.get("full_layer_call_wall_ns")),
            "receipt_read_ns": integer(segment_timing.get("receipt_read_ns")),
            "encode_ns": sum(integer(step.get("encode_ns")) for step in steps),
            "command_wait_ns": sum(integer(step.get("command_wait_ns")) for step in steps),
            "state_snapshot_ns": sum(integer(step.get("state_snapshot_ns")) for step in steps),
            "receipt": receipt_path,
        })
    total_wall = integer(session_execution.get("elapsed_wall_ns"))
    total_gpu = sum(r["gpu_ns"] for r in rows)
    total_dispatches = sum(r["dispatches"] for r in rows)
    source_payload_bytes = sum(r["source_payload_bytes_read"] for r in rows)
    token_slots = len(session.get("token_ids") or [])
    prompt_slots = len(session.get("prompt_token_ids") or [])
    repeated_timing = {
        "classification": (
            "CLEAN_SOURCE_BOUND_ONE_PASS_REPEATED_ACCEPTED_DECODE"
            if repeated_accepted
            else "SESSION_TIMING_ONLY"
        ),
        "token_slots_in_single_persistent_session": token_slots or None,
        "prompt_token_slots": prompt_slots or None,
        "accepted_continuation_tokens": accepted_tokens or None,
        "session_elapsed_wall_ns": total_wall or None,
        "source_payload_bytes_per_session_token": (
            source_payload_bytes // token_slots if token_slots else None
        ),
        "source_payload_bytes_per_accepted_continuation": (
            source_payload_bytes // accepted_tokens if accepted_tokens else None
        ),
        "steady_decode_tps": None,
        "why_steady_decode_tps_is_null": (
            ("the teacher-bound compact oracle still performs one-pass source loading "
             "and includes prompt work plus host diagnostic seams; this is a bottleneck "
             "census, not a warmed persistent-runtime decode benchmark"
             if route_safe_compact else
             "the exact oracle still loads dense source families and includes prompt "
             "work plus host diagnostic seams; this one pass is a bottleneck census, "
             "not a warmed persistent-runtime decode benchmark")
        ),
    }
    report = {
        "schema": "hawking.flash.complete_session_timing_decomposition.v1",
        "status": (
            "MEASURED_REPEATED_ACCEPTED_SOURCE_BOUND_ONE_PASS"
            if repeated_accepted
            else "MEASURED_BASELINE"
        ),
        "session": str(args.session),
        "session_sha256": hashlib.sha256(args.session.read_bytes()).hexdigest(),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "process_boundary": session_execution.get("process_boundary"),
        "segments": rows,
        "totals": {
            "elapsed_wall_ns": total_wall,
            "elapsed_hours": total_wall / 3_600_000_000_000 if total_wall else None,
            "measured_gpu_ns": total_gpu,
            "measured_host_and_unattributed_ns": max(0, total_wall - total_gpu),
            "dispatches": total_dispatches,
            "source_payload_bytes_read": source_payload_bytes,
            "source_and_device_prepare_ns": sum(r["source_and_device_prepare_ns"] for r in rows),
            "execution_wall_ns": sum(r["execution_wall_ns"] for r in rows),
            "source_load_ns": sum(r["source_load_ns"] for r in rows),
            "oracle_ns": sum(r["oracle_ns"] for r in rows),
            "device_prepare_ns": sum(r["device_prepare_ns"] for r in rows),
            "graph_prepare_ns": sum(r["graph_prepare_ns"] for r in rows),
            "encode_ns": sum(r["encode_ns"] for r in rows),
            "command_wait_ns": sum(r["command_wait_ns"] for r in rows),
            "state_snapshot_ns": sum(r["state_snapshot_ns"] for r in rows),
            "receipt_read_ns": sum(r["receipt_read_ns"] for r in rows),
        },
        "repeated_decode_timing": repeated_timing,
        "execution_mode": {
            "expert_bank_mode": expert_bank_mode,
            "route_safe_compact_teacher_bound": route_safe_compact,
        },
        "top_runtime_latencies": [
            {"rank": 1,
             "name": ("remaining compact-bank/source-load ceremony"
                      if route_safe_compact else "dense expert-bank source/load ceremony"),
             "status": "dominant_by_source_bytes",
             "evidence": ("per-segment source payload bytes and wall time; the expert bank is a "
                          "teacher-bound compact union, but exact open/read split is not instrumented"
                          if route_safe_compact else
                          "per-segment source payload bytes and wall time; exact open/read split not instrumented")},
            {"rank": 2, "name": "host and synchronization ceremony", "status": "measured_residual", "evidence": "segment wall_ns minus reported GPU ns"},
            {"rank": 3, "name": "dispatch/command submission", "status": "measured_count_only", "evidence": "per-step dispatch totals; dispatch ns not separately instrumented"},
        ],
        "claim_boundary": ("Post-hoc timing census for one teacher-bound compact route-union "
                           "stateful session. It proves neither future-route coverage nor steady "
                           "decode TPS, capability, EBPW, or residency."
                           if route_safe_compact else
                           "Post-hoc timing baseline for the exact dense stateful oracle. When repeated acceptance is required, the receipt proves only one clean source-bound persistent session with independent token checks. Source open/read/parse, upload, pipeline, wait, serialization and receipt sub-buckets remain explicitly unattributed where the source receipt did not time them; this is not steady decode TPS or residency."),
        "next": ("Instrument persistent compact-bank reuse with separate source/index/upload/pipeline/wait/state/receipt timers, then measure a warmed repeated decode loop."
                 if route_safe_compact else
                 "Instrument the persistent route-safe executor with separate source/index/upload/pipeline/wait/state/receipt timers, then eliminate the measured dense source/load and host-seam work before timing a warmed repeated decode loop."),
        "bench": {"state": "UNKNOWN", "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "recorded_by": "tools/flash_session_timing.py", "machine": "Apple M3 Ultra", "rule": "S032 §3 -- post-hoc baseline; no protected performance claim"},
    }
    report["seal_sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
