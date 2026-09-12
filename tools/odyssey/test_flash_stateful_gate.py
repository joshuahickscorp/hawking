"""Regression coverage for Flash's stateful-decode admission boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tools import flash_stateful_gate as gate


def test_accepted_session_advances_to_repeated_decode_not_initial_liveness(
    tmp_path: Path, monkeypatch
) -> None:
    """One accepted token is a real frontier advance, but not TPS evidence."""
    source = tmp_path / "source.rs"
    source.write_text(
        "Tokenizer::from_file encode decode_one is_eog tokenizer generate "
        "kv_cache key_cache position reset_states recurrent_state "
        "TerminalExecutor source_index_reused lm_head_reused "
        "FIRST_COMPLETE_TOKEN_TERMINAL_PROBE terminal::run_with"
    )
    receipts = tmp_path / "receipts" / "headless"
    receipts.mkdir(parents=True)
    (receipts / "FLASH_STATEFUL_COMPLETE_TOKEN_ACCEPTED.json").write_text(
        json.dumps(
            {
                "status": "PASSED_STATEFUL_COMPLETE_TOKEN_ACCEPTED",
                "accepted_generation_tokens": 1,
                "claim_boundary": "bounded accepted token",
            }
        )
    )
    (receipts / "FLASH_TOKENIZER_ACCEPTANCE_CONTRACT.json").write_text("{}")

    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "FILES", [source])
    out = tmp_path / "gate.json"
    monkeypatch.setattr(sys, "argv", ["flash_stateful_gate.py", "--out", str(out)])

    assert gate.main() == 0
    result = json.loads(out.read_text())
    assert result["status"] == "PENDING_REPEATED_ACCEPTED_DECODE"
    assert result["accepted_tokens"] == 1
    assert result["accepted_tps"] is None
    assert result["first_physical_failure_boundary"]["stage"] == "repeated_accepted_decode"
    assert "persistent continuation state" in result["first_physical_failure_boundary"]["evidence"]
    assert "repeated accepted decode" in result["next_action"]


def test_repeated_accepted_session_advances_only_to_clean_timing(
    tmp_path: Path, monkeypatch
) -> None:
    """A multi-token receipt must prove state, independent checks, and memory."""
    source = tmp_path / "source.rs"
    source.write_text(
        "Tokenizer::from_file encode decode_one is_eog tokenizer generate "
        "kv_cache key_cache position reset_states recurrent_state "
        "TerminalExecutor source_index_reused lm_head_reused "
        "FIRST_COMPLETE_TOKEN_TERMINAL_PROBE terminal::run_with "
        "--prompt-length reference_checks source_reset_or_reprefill "
        "persistent_state_bytes growth_bytes_per_additional_token"
    )
    receipts = tmp_path / "receipts" / "headless"
    receipts.mkdir(parents=True)
    (receipts / "FLASH_STATEFUL_REPEATED_ACCEPTED_DECODE.json").write_text(
        json.dumps(
            {
                "status": "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE",
                "accepted_generation_tokens": 2,
                "terminal": {
                    "reference_checks": [
                        {"expected_token_id": 11, "predicted_token_id": 11, "accepted": True},
                        {"expected_token_id": 12, "predicted_token_id": 12, "accepted": True},
                    ]
                },
                "execution": {
                    "source_reset_or_reprefill": False,
                    "state_memory": {
                        "total_persistent_bytes": 1024,
                        "growth_bytes_per_additional_token": 128,
                    },
                },
            }
        )
    )
    (receipts / "FLASH_TOKENIZER_ACCEPTANCE_CONTRACT.json").write_text("{}")

    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "FILES", [source])
    out = tmp_path / "gate.json"
    monkeypatch.setattr(sys, "argv", ["flash_stateful_gate.py", "--out", str(out)])

    assert gate.main() == 0
    result = json.loads(out.read_text())
    assert result["status"] == "PENDING_CLEAN_REPEATED_TIMING"
    assert result["repeated_state_valid"] is True
    assert result["accepted_tokens"] == 2
    assert result["first_physical_failure_boundary"]["stage"] == "clean_repeated_timing"
    assert result["complete_stateful_session"]["receipt"].endswith(
        "FLASH_STATEFUL_REPEATED_ACCEPTED_DECODE.json"
    )


def test_clean_repeated_census_advances_to_route_safe_active_experts(
    tmp_path: Path, monkeypatch
) -> None:
    """A real one-pass census removes timing as the next missing boundary."""
    source = tmp_path / "source.rs"
    source.write_text(
        "Tokenizer::from_file encode decode_one is_eog tokenizer generate "
        "kv_cache key_cache position reset_states recurrent_state "
        "TerminalExecutor source_index_reused lm_head_reused "
        "FIRST_COMPLETE_TOKEN_TERMINAL_PROBE terminal::run_with "
        "--prompt-length reference_checks source_reset_or_reprefill "
        "persistent_state_bytes growth_bytes_per_additional_token"
    )
    receipts = tmp_path / "receipts" / "headless"
    receipts.mkdir(parents=True)
    (receipts / "FLASH_STATEFUL_REPEATED_ACCEPTED_DECODE.json").write_text(
        json.dumps(
            {
                "status": "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE",
                "accepted_generation_tokens": 2,
                "terminal": {"reference_checks": [
                    {"expected_token_id": 11, "predicted_token_id": 11, "accepted": True},
                    {"expected_token_id": 12, "predicted_token_id": 12, "accepted": True},
                ]},
                "execution": {
                    "source_reset_or_reprefill": False,
                    "state_memory": {
                        "total_persistent_bytes": 1024,
                        "growth_bytes_per_additional_token": 128,
                    },
                },
            }
        )
    )
    (receipts / "FLASH_STATEFUL_REPEATED_ACCEPTED_DECODE.TIMING_CENSUS.json").write_text(
        json.dumps(
            {
                "status": "MEASURED_REPEATED_ACCEPTED_SOURCE_BOUND_ONE_PASS",
                "process_boundary": "one native process",
                "totals": {
                    "elapsed_wall_ns": 10,
                    "source_payload_bytes_read": 20,
                    "dispatches": 3,
                },
                "repeated_decode_timing": {"steady_decode_tps": None},
            }
        )
    )
    (receipts / "FLASH_TOKENIZER_ACCEPTANCE_CONTRACT.json").write_text("{}")

    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "FILES", [source])
    out = tmp_path / "gate.json"
    monkeypatch.setattr(sys, "argv", ["flash_stateful_gate.py", "--out", str(out)])

    assert gate.main() == 0
    result = json.loads(out.read_text())
    assert result["status"] == "PENDING_ROUTE_SAFE_ACTIVE_EXPERT_EXECUTION"
    assert result["clean_repeated_timing"]["valid"] is True
    assert result["first_physical_failure_boundary"]["stage"] == (
        "route_safe_active_expert_execution"
    )
    assert result["clean_repeated_timing"]["steady_decode_tps"] is None


def test_compact_teacher_bound_census_advances_to_persistent_reuse(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.rs"
    source.write_text(
        "Tokenizer::from_file encode decode_one is_eog tokenizer generate "
        "kv_cache key_cache position reset_states recurrent_state "
        "TerminalExecutor source_index_reused lm_head_reused "
        "FIRST_COMPLETE_TOKEN_TERMINAL_PROBE terminal::run_with "
        "--prompt-length reference_checks source_reset_or_reprefill "
        "persistent_state_bytes growth_bytes_per_additional_token"
    )
    receipts = tmp_path / "receipts" / "headless"
    receipts.mkdir(parents=True)
    (receipts / "FLASH_STATEFUL_REPEATED_ACCEPTED_DECODE.json").write_text(json.dumps({
        "status": "PASSED_STATEFUL_REPEATED_ACCEPTED_DECODE",
        "accepted_generation_tokens": 2,
        "terminal": {"reference_checks": [
            {"expected_token_id": 11, "predicted_token_id": 11, "accepted": True},
            {"expected_token_id": 12, "predicted_token_id": 12, "accepted": True},
        ]},
        "execution": {"source_reset_or_reprefill": False,
                      "expert_bank_mode": "route_union_compact_teacher_bound",
                      "state_memory": {"total_persistent_bytes": 1024,
                                       "growth_bytes_per_additional_token": 128}},
    }))
    (receipts / "FLASH_STATEFUL_REPEATED_ACCEPTED_DECODE.TIMING_CENSUS.json").write_text(json.dumps({
        "status": "MEASURED_REPEATED_ACCEPTED_SOURCE_BOUND_ONE_PASS",
        "process_boundary": "one native process",
        "totals": {"elapsed_wall_ns": 10, "source_payload_bytes_read": 20, "dispatches": 3},
        "repeated_decode_timing": {"steady_decode_tps": None},
    }))
    (receipts / "FLASH_TOKENIZER_ACCEPTANCE_CONTRACT.json").write_text("{}")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "FILES", [source])
    out = tmp_path / "gate.json"
    monkeypatch.setattr(sys, "argv", ["flash_stateful_gate.py", "--out", str(out)])
    assert gate.main() == 0
    result = json.loads(out.read_text())
    assert result["status"] == "PENDING_WARMED_DIRECT_DECODE_INSTRUMENTATION"
    assert result["route_safe_compact_execution"]["valid"] is True
    assert result["first_physical_failure_boundary"]["stage"] == "persistent_compact_runtime_instrumentation"
