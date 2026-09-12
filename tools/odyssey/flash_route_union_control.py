#!/usr/bin/env python3
"""Execute the first teacher-bound compact Flash route-union control.

The control is intentionally narrow.  It does not discover routes, extrapolate
cache behavior, or report TPS.  It replays a source-bound accepted sequence
using only unions observed in a complete dense teacher session, and treats a
route or terminal mismatch as the first physical failure.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
)
REPO_ID = "Qwen/Qwen3.8-Flash-Next"
PINNED_REVISION = "34567a4712bc9766c4449e2e98e4468bfa24d915"
TOKENS = [5423, 799, 4581, 3817, 13, 2972, 96125]
PROMPT_LEN = 5


def _load(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"receipt root must be an object: {path}")
    return doc


def _verify_teacher(doc: dict[str, Any]) -> None:
    if doc.get("status") != "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE":
        raise ValueError("route teacher is not a passed repeated accepted decode")
    if doc.get("model") != REPO_ID or doc.get("pinned_revision") != PINNED_REVISION:
        raise ValueError("route teacher source identity mismatch")
    if doc.get("token_ids") != TOKENS:
        raise ValueError("route teacher token sequence mismatch")
    if doc.get("execution", {}).get("source_reset_or_reprefill") is not False:
        raise ValueError("route teacher lacks persistent-session proof")


def _verify_candidate(
    doc: dict[str, Any],
    teacher: dict[str, Any],
    *,
    device_only_resident: bool = False,
    token_major_resident: bool = False,
) -> None:
    _verify_teacher(doc)
    if doc.get("status") != "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE":
        raise ValueError("compact candidate did not preserve repeated acceptance")
    if doc.get("execution", {}).get("expert_bank_mode") != "route_union_compact_teacher_bound":
        raise ValueError("candidate did not execute the teacher-bound compact expert path")
    teacher_bytes = int(teacher.get("execution", {}).get("source_payload_bytes_read", 0))
    candidate_bytes = int(doc.get("execution", {}).get("source_payload_bytes_read", 0))
    if teacher_bytes <= 0 or candidate_bytes <= 0 or candidate_bytes >= teacher_bytes:
        raise ValueError("compact candidate did not reduce source payload reads")
    checks = doc.get("terminal", {}).get("reference_checks")
    if not isinstance(checks, list) or len(checks) != 2:
        raise ValueError("compact candidate omitted independent terminal checks")
    for expected, check in zip(TOKENS[PROMPT_LEN:], checks):
        if not isinstance(check, dict) or check.get("expected_token_id") != expected:
            raise ValueError("compact candidate reference sequence drifted")
        if check.get("predicted_token_id") != expected or check.get("accepted") is not True:
            raise ValueError("compact candidate terminal mismatch")
    if device_only_resident:
        resident = doc.get("execution", {}).get("linear_compact_bank_residency", {})
        expected_linear_status = (
            "ALL_LINEAR_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE"
            if token_major_resident
            else "PARTIAL_PROCESS_LIFETIME_DEVICE_ONLY_LINEAR_BANKS_RETAINED"
        )
        if resident.get("status") != expected_linear_status:
            raise ValueError("device-only resident control did not retain compact linear banks")
        if resident.get("immutable_weight_ownership") != "device_only_after_source_upload":
            raise ValueError("device-only resident control retained an ambiguous weight owner")
        if int(resident.get("retained_linear_layers", 0)) <= 0:
            raise ValueError("device-only resident control retained no linear layers")
        if int(resident.get("retained_linear_source_weight_bytes", -1)) != 0:
            raise ValueError("device-only resident control retained source-side linear weights")
        if int(resident.get("retained_linear_device_weight_bytes", 0)) <= 0:
            raise ValueError("device-only resident control omitted device weight accounting")
        if int(resident.get("observed_process_rss_bytes", 0)) <= 0:
            raise ValueError("device-only resident control omitted observed process RSS")
        attention_segments = [
            segment
            for segment in doc.get("segments", [])
            if isinstance(segment, dict) and segment.get("species") == "full_attention"
        ]
        if not attention_segments:
            raise ValueError("device-only resident control omitted full-attention segments")
        for segment in attention_segments:
            if segment.get("immutable_weight_ownership") != "device_only_after_source_upload":
                raise ValueError("full-attention segment retained an ambiguous weight owner")
            if segment.get("host_source_weights_retained_during_token_loop") is not False:
                raise ValueError("full-attention segment retained source weights during token execution")
        attention_residency = doc.get("execution", {}).get("full_attention_compact_bank_residency", {})
        expected_attention_status = (
            "ALL_FULL_ATTENTION_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE"
            if token_major_resident
            else "PER_LAYER_DEVICE_ONLY_FULL_ATTENTION_BANKS_REUSED_ACROSS_SESSION_TOKENS"
        )
        if attention_residency.get("status") != expected_attention_status:
            raise ValueError("device-only resident control did not account for full-attention bank reuse")
        if int(attention_residency.get("resident_full_attention_layers", 0)) != len(attention_segments):
            raise ValueError("full-attention resident-layer count does not match the session")
        if int(attention_residency.get("expected_full_attention_layers", 0)) != len(attention_segments):
            raise ValueError("full-attention resident control omitted expected layer count")
        if int(attention_residency.get("device_weight_bytes", 0)) <= 0:
            raise ValueError("full-attention resident control omitted device weight accounting")
        if attention_residency.get("host_source_weights_retained_during_token_loop") is not False:
            raise ValueError("full-attention resident control retained source weights")
        if not token_major_resident and attention_residency.get("process_lifetime_all_banks_retained") is not False:
            raise ValueError("full-attention resident control overstates layer-major bank lifetime")
    if token_major_resident:
        if not device_only_resident:
            raise ValueError("token-major resident control requires device-only resident verification")
        token_major = doc.get("execution", {}).get("token_major_resident_banks", {})
        if token_major.get("status") != "ALL_48_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE":
            raise ValueError("token-major resident control did not retain all 48 banks")
        if int(token_major.get("resident_layers", 0)) != 48 or int(token_major.get("expected_layers", 0)) != 48:
            raise ValueError("token-major resident control omitted complete 48-layer accounting")
        if token_major.get("cross_layer_activation_handoff") != "device buffers":
            raise ValueError("token-major resident control did not use device activation handoffs")
        if token_major.get("source_reset_or_reprefill") is not False:
            raise ValueError("token-major resident control lacks persistent-session proof")
        linear = doc.get("execution", {}).get("linear_compact_bank_residency", {})
        if linear.get("status") != "ALL_LINEAR_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE":
            raise ValueError("token-major resident control omitted full linear-bank lifetime")
        attention = doc.get("execution", {}).get("full_attention_compact_bank_residency", {})
        if attention.get("status") != "ALL_FULL_ATTENTION_DEVICE_ONLY_BANKS_RETAINED_ACROSS_ACCEPTED_TOKEN_SEQUENCE":
            raise ValueError("token-major resident control omitted full-attention lifetime")
        if attention.get("process_lifetime_all_banks_retained") is not True:
            raise ValueError("token-major resident control did not retain full-attention banks")
        segments = doc.get("segments")
        if not isinstance(segments, list) or len(segments) != 48:
            raise ValueError("token-major resident control omitted per-layer execution evidence")
        seen_layers: set[int] = set()
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError("token-major resident segment is not an object")
            raw_layer = segment.get("layer")
            if raw_layer is None:
                layers = segment.get("layers")
                raw_layer = layers[0] if isinstance(layers, list) and len(layers) == 2 and layers[0] == layers[1] else None
            if not isinstance(raw_layer, int) or raw_layer < 0 or raw_layer >= 48 or raw_layer in seen_layers:
                raise ValueError("token-major resident layer inventory is incomplete or duplicated")
            seen_layers.add(raw_layer)
            rows = segment.get("steps")
            if not isinstance(rows, list) or len(rows) != len(TOKENS):
                raise ValueError("token-major resident segment omitted repeated-token rows")
            for step, row in enumerate(rows):
                if not isinstance(row, dict) or row.get("step") != step or row.get("token_id") != TOKENS[step]:
                    raise ValueError("token-major resident token sequence drifted")
        if len(seen_layers) != 48:
            raise ValueError("token-major resident control did not cover all layers")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--teacher", type=Path,
        default=ROOT / "receipts/headless/FLASH_DENSE_ROUTE_TRACE_CONTROL.json",
    )
    ap.add_argument(
        "--retain-linear-banks",
        action="store_true",
        help="retain compact linear banks through the complete control process",
    )
    ap.add_argument(
        "--device-only-compact-banks",
        action="store_true",
        help="drop source-side compact linear weights after their Metal upload",
    )
    ap.add_argument(
        "--token-major-resident-banks",
        action="store_true",
        help="retain all 48 exact teacher-bound compact banks across the full token-major session",
    )
    ap.add_argument(
        "--out", type=Path,
        default=ROOT / "receipts/headless/FLASH_ROUTE_UNION_COMPACT_CONTROL.json",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.device_only_compact_banks and not args.retain_linear_banks:
        ap.error("--device-only-compact-banks requires --retain-linear-banks")
    if args.token_major_resident_banks and not (
        args.retain_linear_banks and args.device_only_compact_banks
    ):
        ap.error("--token-major-resident-banks requires --retain-linear-banks --device-only-compact-banks")
    teacher_path = args.teacher.resolve()
    out = args.out.resolve()
    command = [
        "cargo", "run", "--release", "-q", "-p", "hawking-core",
        "--example", "flash_stateful_complete_token_session", "--",
        "--root", str(MODEL_ROOT),
        "--token-ids", ",".join(str(token) for token in TOKENS),
        "--prompt-length", str(PROMPT_LEN),
        "--route-teacher", str(teacher_path),
        "--out", str(out),
    ]
    if args.retain_linear_banks:
        command.append("--retain-linear-banks")
    if args.device_only_compact_banks:
        command.append("--device-only-compact-banks")
    if args.token_major_resident_banks:
        command.append("--token-major-resident-banks")
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN", "command": command}, indent=2))
        return 0
    teacher = _load(teacher_path)
    _verify_teacher(teacher)
    subprocess.run(command, cwd=ROOT, check=True)
    candidate = _load(out)
    _verify_candidate(
        candidate,
        teacher,
        device_only_resident=args.device_only_compact_banks,
        token_major_resident=args.token_major_resident_banks,
    )
    print(json.dumps({
        "status": "PASSED_TEACHER_BOUND_ROUTE_UNION_CONTROL",
        "teacher": str(teacher_path),
        "candidate": str(out),
        "teacher_source_bytes": teacher["execution"]["source_payload_bytes_read"],
        "candidate_source_bytes": candidate["execution"]["source_payload_bytes_read"],
        "device_only_linear_residency": args.device_only_compact_banks,
        "token_major_residency": args.token_major_resident_banks,
        "claim_boundary": "bounded teacher control only; when requested, all 48 compact banks remain device-only through the accepted token sequence with device activation handoffs. Per-layer diagnostics remain enabled, so this is not warmed full-model TPS, EBPW, capability, future-route coverage, or resident promotion.",
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as exc:
        print(json.dumps({"status": "FAILED_PRECISE_DIAGNOSIS", "error": str(exc)}))
        raise SystemExit(2)
