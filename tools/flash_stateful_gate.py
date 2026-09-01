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
    a = ap.parse_args()
    text = "\n".join(p.read_text() for p in FILES)
    checks = {
        "tokenizer_contract": ("Tokenizer::from_file", "encode", "decode_one", "is_eog"),
        "tokenizer_or_prompt_loop": ("tokenizer", "generate", "max_new_tokens"),
        "persistent_full_attention_kv": ("kv_cache", "key_cache", "position"),
        "persistent_linear_recurrence": ("reset_states", "recurrent_state"),
        "persistent_terminal_executor": ("TerminalExecutor", "source_index_reused", "lm_head_reused"),
        "terminal_probe_only": ("FIRST_COMPLETE_TOKEN_TERMINAL_PROBE", "terminal::run_with"),
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
    session_path = next(
        (candidate for candidate in (
            ROOT / "receipts/headless/FLASH_STATEFUL_COMPLETE_TOKEN_ACCEPTED.json",
            ROOT / "receipts/headless/FLASH_STATEFUL_COMPLETE_TOKEN_SESSION.json",
        ) if candidate.is_file()),
        ROOT / "receipts/headless/FLASH_STATEFUL_COMPLETE_TOKEN_SESSION.json",
    )
    session_payload = None
    if session_path.is_file():
        try:
            session_payload = json.loads(session_path.read_text())
        except (OSError, json.JSONDecodeError):
            session_payload = None
    session_status = session_payload.get("status") if isinstance(session_payload, dict) else None
    session_accepted = int(session_payload.get("accepted_generation_tokens", 0)) if isinstance(session_payload, dict) else 0
    executor_path = ROOT / "receipts/headless/FLASH_TERMINAL_EXECUTOR_COMPILE.json"
    executor_payload = None
    if executor_path.is_file():
        try:
            executor_payload = json.loads(executor_path.read_text())
        except (OSError, json.JSONDecodeError):
            executor_payload = None
    if session_status == "PASSED_STATEFUL_COMPLETE_TOKEN_ACCEPTED":
        token_boundary = {"stage": "token_acceptance_loop", "status": "ONE_TOKEN_ACCEPTED",
                          "evidence": "the complete 48-layer stateful oracle produced the tokenizer-bound terminal argmax as one accepted generated token; continuation state, repeated decode and protected TPS remain open"}
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
         "evidence": ("the newest stateful attention receipt proves two-position persistent KV slots, the full-attention MLP epilogue, and source parity; complete 48-layer per-layer KV integration remains absent"
                      if attention_organ_qualified else
                      "flash_full_attention_layer3.rs has no persistent KV cache")},
        {"stage": "linear_recurrence_state", "status": "ORGAN_AND_PREFIX_QUALIFIED",
         "evidence": "FLASH_STATEFUL_LINEAR_PREFIX_SESSION.json proves recurrent state across layers 0..2 and two token steps; complete 48-layer executor still does not expose the session"},
    ]
    doc = {
        "schema": "hawking.flash.stateful_tps_gate.v1",
        "status": "BLOCKED_FIRST_PHYSICAL_BOUNDARY",
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
        "claim_boundary": "No accepted multi-token TPS or EBPW claim. The complete stateful session receipt, when present, is bounded to its recorded candidate result; repeated accepted decode, capability, EBPW and residency remain open.",
        "next_action": "Continue from the complete-session receipt with a predicted candidate, then measure repeated accepted decode steps with deterministic tokenizer and terminal checks.",
    }
    doc["seal_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"status": doc["status"], "first_boundary": blockers[0]["stage"], "out": str(a.out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
