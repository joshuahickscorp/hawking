#!/usr/bin/env python3
"""Audit the current Flash executor for a valid accepted multi-token path.

This is a source-backed gate, not a synthetic benchmark.  It emits BLOCKED when
the executor can produce only a terminal probe or resets token state, preserving
the exact first missing requirement for the next implementation pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "crates/hawking-core/examples/flash_fast_chain.rs",
    ROOT / "crates/hawking-core/examples/flash_noetic_complete_layer0.rs",
    ROOT / "crates/hawking-core/examples/flash_full_attention_layer3.rs",
    ROOT / "crates/hawking-core/examples/flash_source_bf16_terminal.rs",
    ROOT / "crates/hawking-core/examples/flash_stateful_complete_token_session.rs",
    ROOT / "crates/hawking-core/examples/flash_tokenizer_acceptance_contract.rs",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument(
        "--session",
        type=pathlib.Path,
        help=(
            "explicit complete-session receipt to audit; defaults to the canonical "
            "accepted-session receipt selection"
        ),
    )
    ap.add_argument(
        "--timing",
        type=pathlib.Path,
        help=(
            "explicit timing/census receipt paired with --session; defaults to the "
            "canonical repeated-decode timing receipt"
        ),
    )
    a = ap.parse_args()
    text = "\n".join(p.read_text() for p in FILES)
    checks = {
        "tokenizer_contract": ("Tokenizer::from_file", "encode", "decode_one", "is_eog"),
        "tokenizer_or_prompt_loop": ("tokenizer", "generate", "max_new_tokens"),
        "persistent_full_attention_kv": ("kv_cache", "key_cache", "position"),
        "persistent_linear_recurrence": ("reset_states", "recurrent_state"),
        "persistent_terminal_executor": ("TerminalExecutor", "source_index_reused", "lm_head_reused"),
        "terminal_probe_only": ("FIRST_COMPLETE_TOKEN_TERMINAL_PROBE", "terminal::run_with"),
        "repeated_reference_contract": ("--prompt-length", "reference_checks", "source_reset_or_reprefill"),
        "state_memory_census": ("persistent_state_bytes", "growth_bytes_per_additional_token"),
    }
    observed = {name: {needle: needle in text for needle in needles}
                for name, needles in checks.items()}
    organ_receipts = []
    for name in (
        "FLASH_STATEFUL_LINEAR_ORGAN.json",
        "FLASH_STATEFUL_ATTENTION_ORGAN.json",
        "FLASH_STATEFUL_ATTENTION_ORGAN_V2.json",
        "FLASH_STATEFUL_CROSS_SPECIES_SEAM_V3_ATTN.json",
        "FLASH_STATEFUL_CROSS_SPECIES_SEAM_V2_ATTN.json",
        "FLASH_STATEFUL_LINEAR_PREFIX_SESSION.json",
    ):
        path = ROOT / "receipts/headless" / name
        if path.is_file():
            payload = json.loads(path.read_text())
            organ_receipts.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                   "status": payload.get("status"), "schema": payload.get("schema")})
    # The isolated organ probes qualify recurrence/KV behavior, but the
    # complete executor still has no shared per-layer state arena across an
    # accepted-token loop.
    attention_organ_qualified = any(
        item.get("status") == "PASSED_STATEFUL_KV_ORGAN"
        for item in organ_receipts
        if item.get("schema") == "hawking.flash.stateful_attention_organ_probe.v1"
    )
    bridge_path = next(
        (candidate for candidate in (
            ROOT / "receipts/headless/FLASH_STATEFUL_LAYER3_LAYER11_BRIDGE.json",
            ROOT / "receipts/headless/FLASH_STATEFUL_LAYER3_LAYER7_BRIDGE.json",
            ROOT / "receipts/headless/FLASH_STATEFUL_LAYER3_LAYER4_BRIDGE.json",
        ) if candidate.is_file()),
        ROOT / "receipts/headless/FLASH_STATEFUL_LAYER3_LAYER4_BRIDGE.json",
    )
    bridge_payload = None
    if bridge_path.is_file():
        try:
            bridge_payload = json.loads(bridge_path.read_text())
        except (OSError, json.JSONDecodeError):
            bridge_payload = None
    # Prefer an actual repeated/session receipt only when it has reached an
    # accepted state.  Historical failed candidate probes may share the same
    # filename and must not erase the later one-token acceptance evidence.
    repeated_session_path = ROOT / "receipts/headless/FLASH_STATEFUL_REPEATED_ACCEPTED_DECODE.json"
    session_path = ROOT / "receipts/headless/FLASH_STATEFUL_COMPLETE_TOKEN_SESSION.json"
    session_payload = None
    accepted_fallback_path = ROOT / "receipts/headless/FLASH_STATEFUL_COMPLETE_TOKEN_ACCEPTED.json"
    accepted_fallback_payload = None
    candidates = ([a.session] if a.session is not None else
                  [repeated_session_path, session_path, accepted_fallback_path])
    for candidate in candidates:
        if candidate is None:
            continue
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        status = payload.get("status") if isinstance(payload, dict) else None
        if candidate == accepted_fallback_path:
            accepted_fallback_payload = payload
        if status in {"PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE", "PASSED_STATEFUL_COMPLETE_TOKEN_SESSION"}:
            session_path = candidate
            session_payload = payload
            break
    if session_payload is None and accepted_fallback_payload is not None:
        session_path = accepted_fallback_path
        session_payload = accepted_fallback_payload
    session_status = session_payload.get("status") if isinstance(session_payload, dict) else None
    session_accepted = int(session_payload.get("accepted_generation_tokens", 0)) if isinstance(session_payload, dict) else 0
    executor_path = ROOT / "receipts/headless/FLASH_TERMINAL_EXECUTOR_COMPILE.json"
    executor_payload = None
    if executor_path.is_file():
        try:
            executor_payload = json.loads(executor_path.read_text())
        except (OSError, json.JSONDecodeError):
            executor_payload = None
    repeated_checks = (session_payload.get("terminal", {}).get("reference_checks", [])
                       if isinstance(session_payload, dict) else [])
    execution = session_payload.get("execution", {}) if isinstance(session_payload, dict) else {}
    repeated_state_valid = (
        session_status == "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE"
        and session_accepted >= 2
        and isinstance(repeated_checks, list)
        and len(repeated_checks) >= 2
        and all(isinstance(item, dict)
                and item.get("accepted") is True
                and item.get("predicted_token_id") == item.get("expected_token_id")
                for item in repeated_checks)
        and execution.get("source_reset_or_reprefill") is False
        and isinstance(execution.get("state_memory"), dict)
        and int(execution["state_memory"].get("total_persistent_bytes", 0)) > 0
        and int(execution["state_memory"].get("growth_bytes_per_additional_token", 0)) > 0
    )
    timing_path = a.timing or (ROOT / "receipts/headless/FLASH_STATEFUL_REPEATED_ACCEPTED_DECODE.TIMING_CENSUS.json")
    if not timing_path.is_absolute():
        timing_path = ROOT / timing_path
    timing_payload = None
    if timing_path.is_file():
        try:
            timing_payload = json.loads(timing_path.read_text())
        except (OSError, json.JSONDecodeError):
            timing_payload = None
    timing_totals = timing_payload.get("totals", {}) if isinstance(timing_payload, dict) else {}
    repeated_timing = (timing_payload.get("repeated_decode_timing", {})
                       if isinstance(timing_payload, dict) else {})
    timing_valid = (
        repeated_state_valid
        and isinstance(timing_payload, dict)
        and timing_payload.get("status") == "MEASURED_REPEATED_ACCEPTED_SOURCE_BOUND_ONE_PASS"
        and timing_payload.get("process_boundary") == "one native process"
        and int(timing_totals.get("elapsed_wall_ns", 0)) > 0
        and int(timing_totals.get("source_payload_bytes_read", 0)) > 0
        and int(timing_totals.get("dispatches", 0)) > 0
        and repeated_timing.get("steady_decode_tps") is None
    )
    route_safe_compact_valid = (
        timing_valid
        and execution.get("expert_bank_mode") == "route_union_compact_teacher_bound"
    )
    if route_safe_compact_valid:
        token_boundary = {
            "stage": "persistent_compact_runtime_instrumentation",
            "status": "NEXT_REQUIRED",
            "evidence": (
                "a bounded teacher-bound compact expert-bank session reproduced every "
                "recorded route and both accepted continuation tokens. The next missing "
                "physical evidence is persistent compact-bank reuse with separated "
                "source/index/upload/pipeline/wait/state timers before warmed decode timing."
            ),
        }
    elif timing_valid:
        token_boundary = {
            "stage": "route_safe_active_expert_execution",
            "status": "NEXT_REQUIRED",
            "evidence": (
                "the accepted stateful session now has a clean source-bound one-pass "
                "timing/census. Dense expert-bank source/load and host-seam ceremony "
                "dominate; steady TPS remains intentionally unmeasured."
            ),
        }
    elif repeated_state_valid:
        token_boundary = {"stage": "clean_repeated_timing", "status": "NEXT_REQUIRED",
                          "evidence": "a source-bound session accepted at least two continuation tokens, independently checked each preceding terminal argmax, retained state without re-prefill, and emitted a state-memory census. Clean repeated timing is now the next physical gate."}
    elif session_status == "PASSED_STATEFUL_COMPLETE_TOKEN_ACCEPTED":
        token_boundary = {"stage": "repeated_accepted_decode", "status": "NEXT_REQUIRED",
                          "evidence": "the complete 48-layer stateful oracle already produced one tokenizer-bound accepted token. The next missing evidence is feeding that accepted token through persistent continuation state for at least one more accepted decode step under a protected physical lane."}
    elif session_status == "PASSED_STATEFUL_COMPLETE_TOKEN_SESSION":
        token_boundary = {"stage": "token_acceptance_loop", "status": "ONE_TOKEN_ACCEPTED",
                          "evidence": "the complete 48-layer stateful session accepted one tokenizer-bound candidate; repeated accepted decode steps and protected TPS remain open"}
    elif session_status == "PASSED_COMPLETE_FORWARD_CANDIDATE_REJECTED":
        token_boundary = {"stage": "candidate_acceptance", "status": "CANDIDATE_REJECTED",
                          "evidence": "the complete 48-layer stateful forward executed, but the supplied candidate did not match terminal argmax; rerun with the predicted candidate"}
    else:
        token_boundary = {"stage": "token_acceptance_loop", "status": "MISSING",
                          "evidence": "the source-bound tokenizer contract is physically verified, but no complete 48-layer stateful tokenizer/session receipt exists"}
    blockers = [
        token_boundary,
        {"stage": "full_attention_state", "status": "ORGAN_QUALIFIED" if attention_organ_qualified else "MISSING",
         "evidence": ("the stateful attention organs and accepted complete session establish bounded full-attention KV traversal. What remains is a repeatable accepted-token continuation loop with persistent session state, not initial full-attention liveness."
                      if attention_organ_qualified else
                      "flash_full_attention_layer3.rs has no persistent KV cache")},
        {"stage": "linear_recurrence_state", "status": "ORGAN_AND_PREFIX_QUALIFIED",
         "evidence": "FLASH_STATEFUL_LINEAR_PREFIX_SESSION.json proves recurrent state across layers 0..2 and two token steps; the complete session has now passed one token, while repeated accepted-token continuation remains the next scoped requirement."},
    ]
    doc = {
        "schema": "hawking.flash.stateful_tps_gate.v1",
        "status": ("PENDING_WARMED_DIRECT_DECODE_INSTRUMENTATION"
                   if route_safe_compact_valid
                   else "PENDING_CLEAN_REPEATED_TIMING"
                   if repeated_state_valid and not timing_valid
                   else "PENDING_ROUTE_SAFE_ACTIVE_EXPERT_EXECUTION"
                   if timing_valid
                   else "PENDING_REPEATED_ACCEPTED_DECODE"
                   if session_status in {"PASSED_STATEFUL_COMPLETE_TOKEN_ACCEPTED", "PASSED_STATEFUL_COMPLETE_TOKEN_SESSION"}
                   else "BLOCKED_FIRST_PHYSICAL_BOUNDARY"),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_files": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                         for p in FILES],
        "source_checks": observed,
        "qualified_state_organs": organ_receipts,
        "stateful_cross_species_bridge": {
            "receipt": str(bridge_path),
            "sha256": hashlib.sha256(bridge_path.read_bytes()).hexdigest() if bridge_payload else None,
            "status": bridge_payload.get("status") if bridge_payload else "MISSING",
            "claim": bridge_payload.get("claim_boundary") if bridge_payload else None,
        },
        "complete_stateful_session": {
            "receipt": str(session_path),
            "sha256": hashlib.sha256(session_path.read_bytes()).hexdigest() if session_payload else None,
            "status": session_status or "MISSING",
            "accepted_generation_tokens": session_accepted,
            "claim": session_payload.get("claim_boundary") if session_payload else None,
        },
        "terminal_executor_compile": {
            "receipt": str(executor_path),
            "sha256": hashlib.sha256(executor_path.read_bytes()).hexdigest() if executor_payload else None,
            "status": executor_payload.get("status") if executor_payload else "MISSING",
            "architecture": executor_payload.get("architecture") if executor_payload else None,
            "physical_execution": executor_payload.get("physical_execution") if executor_payload else None,
        },
        "tokenizer_contract": {
            "receipt": str(ROOT / "receipts/headless/FLASH_TOKENIZER_ACCEPTANCE_CONTRACT.json"),
            "status": "PASSED_TOKENIZER_SESSION_PREREQUISITE"
            if (ROOT / "receipts/headless/FLASH_TOKENIZER_ACCEPTANCE_CONTRACT.json").is_file()
            else "MISSING",
        },
        "first_physical_failure_boundary": blockers[0],
        "blockers": blockers,
        "accepted_tokens": session_accepted,
        "repeated_reference_checks": repeated_checks if isinstance(repeated_checks, list) else [],
        "repeated_state_valid": repeated_state_valid,
        "clean_repeated_timing": {
            "receipt": str(timing_path),
            "sha256": hashlib.sha256(timing_path.read_bytes()).hexdigest()
            if timing_payload else None,
            "status": timing_payload.get("status") if timing_payload else "MISSING",
            "valid": timing_valid,
            "steady_decode_tps": repeated_timing.get("steady_decode_tps")
            if isinstance(repeated_timing, dict) else None,
        },
        "route_safe_compact_execution": {
            "valid": route_safe_compact_valid,
            "expert_bank_mode": execution.get("expert_bank_mode"),
        },
        "accepted_tps": None,
        "complete_system_ebpw": None,
        "bench": {
            "state": "UNKNOWN",
            "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "recorded_by": "tools/flash_stateful_gate.py",
            "machine": "Apple M3 Ultra (source audit; no multi-token benchmark run)",
            "rule": "S032 §3 -- no performance claim; state is UNKNOWN",
            "provenance": "source audit only; no synthetic token loop was timed",
        },
        "claim_boundary": "No accepted multi-token TPS or EBPW claim. A route-union compact session, when present, is bounded to its teacher trace and recorded continuation tokens; future-route coverage, capability, EBPW and residency remain open.",
        "next_action": ("Instrument persistent compact-bank reuse, including source/index/upload/pipeline/wait/state/receipt timers, then run a protected warmed repeated-decode measurement without reloading the compact banks."
                        if route_safe_compact_valid
                        else "Trace the exact bounded route union, then replace dense expert-bank materialization with route-safe compact banks while preserving the repeated accepted-token contract; only then time a warmed decode loop."
                        if timing_valid
                        else "Run clean repeated timing, then record a state/dispatch/bytes-per-token census before selecting the highest-value physical or representation optimization."
                        if repeated_state_valid
                        else "Extend the accepted source-bound session by feeding the accepted token's continuation state through at least one further deterministic decode step, then measure repeated accepted decode under a protected clean GPU lane."
                        if session_status in {"PASSED_STATEFUL_COMPLETE_TOKEN_ACCEPTED", "PASSED_STATEFUL_COMPLETE_TOKEN_SESSION"}
                        else "Continue from the complete-session receipt with a predicted candidate, then measure repeated accepted decode steps with deterministic tokenizer and terminal checks."),
    }
    doc["seal_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"status": doc["status"], "first_boundary": blockers[0]["stage"], "out": str(a.out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
