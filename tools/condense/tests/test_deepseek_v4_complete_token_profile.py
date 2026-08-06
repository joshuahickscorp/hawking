"""Focused contracts for the bounded DeepSeek-V4 diagnostic token profiler."""
from __future__ import annotations

import sys
from threading import Lock
from types import SimpleNamespace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import deepseek_v4_gravity as gravity


class _Tokenizer:
    def encode(self, _prompt: str, *, add_special_tokens: bool) -> SimpleNamespace:
        assert add_special_tokens is False
        return SimpleNamespace(ids=[17])


def _profile(phase: str, ordinal: int, position: int) -> dict:
    profiler = gravity._DiagnosticTokenProfiler(
        phase=phase,
        token_ordinal=ordinal,
        position=position,
        tokenizer_allocation={
            "cpu_duration_ms": 0.1,
            "cpu_wall_elapsed_ms": 0.1,
            "bytes_read_estimate": 4,
            "bytes_written_estimate": 4,
        }
        if phase == "prefill"
        else None,
    )
    with profiler.measure("embedding"):
        pass
    profiler.add_estimate("embedding", bytes_read_estimate=8, bytes_written_estimate=16)
    with profiler.measure("runtime_bookkeeping"):
        pass
    return profiler.finish(forward_wall_elapsed_ms=1.0, forward_cpu_duration_ms=0.8)


def test_complete_token_profile_has_all_required_stages_and_no_other_bucket() -> None:
    profile = _profile("prefill", 0, 0)
    stages = {row["stage"]: row for row in profile["stage_metrics"]}

    assert set(stages) == set(gravity._COMPLETE_TOKEN_PROFILE_STAGES)
    assert stages["embedding"]["gpu_duration_ms"] == 0.0
    assert stages["embedding"]["dispatches"] == 0
    assert stages["embedding"]["occupancy_status"] == "not_available_cpu_numpy_diagnostic"
    assert stages["embedding"]["effective_bandwidth_status"] == "not_available_cpu_numpy_diagnostic"
    assert stages["endpoint_hcli_streaming"]["execution_status"] == "unavailable_in_inprocess_profile"
    assert profile["timing_accounting"]["unexplained_other_wall_elapsed_ms"] == 0.0
    assert profile["timing_accounting"]["status"] == "PASS_ALL_TIME_EXPLICITLY_NAMED"
    stage_wall = sum(row["cpu_wall_elapsed_ms"] for row in stages.values())
    observed_wall = profile["timing_accounting"]["observed_complete_token_wall_elapsed_ms"]
    assert stage_wall <= observed_wall + 0.01


def test_profile_prompt_is_redacted_and_aggregates_real_forward_shaped_records() -> None:
    runtime = object.__new__(gravity.DeepSeekV4DiagnosticRuntime)
    runtime._lock = Lock()
    runtime.tokenizer = _Tokenizer()
    runtime.chat_template_status = "SOURCE_TOKENIZER_CONFIG_HAS_NO_CHAT_TEMPLATE_ROLE_TAG_FALLBACK"
    runtime.eos_id = None
    runtime.position = 0

    def reset() -> None:
        runtime.position = 0

    def forward_token_profiled(
        _token_id: int,
        *,
        phase: str,
        token_ordinal: int,
        tokenizer_allocation: dict | None = None,
    ) -> tuple[np.ndarray, dict, dict, int]:
        profile = gravity._DiagnosticTokenProfiler(
            phase=phase,
            token_ordinal=token_ordinal,
            position=runtime.position,
            tokenizer_allocation=tokenizer_allocation,
        )
        profile.add_manual(
            "router_top6",
            cpu_duration_ms=0.1,
            cpu_wall_elapsed_ms=0.1,
            execution_status="executed",
        )
        record = profile.finish(forward_wall_elapsed_ms=0.5, forward_cpu_duration_ms=0.4)
        runtime.position += 1
        return np.asarray([0.0, 1.0], dtype=np.float32), {"routes": [3, 5, 7, 11, 13, 17]}, record, 1

    runtime.reset = reset
    runtime.forward_token_profiled = forward_token_profiled

    result = gravity.DeepSeekV4DiagnosticRuntime.profile_prompt(
        runtime, "a private prompt", trials=1, decode_forwards=1
    )
    assert result["trials"][0]["prompt_text_disclosed"] is False
    assert "a private prompt" not in repr(result)
    assert len(result["token_profiles"]) == 2
    aggregate = gravity._aggregate_complete_token_profile(result["token_profiles"])
    assert aggregate["real_diagnostic_forward_count"] == 2
    assert set(aggregate["complete_token_wall_elapsed_ms"]) >= {"p50", "p95", "p99"}
    assert aggregate["timing_accounting"]["other_share_percent"] == 0.0


def test_profile_parser_uses_bounded_decode_forward_option() -> None:
    args = gravity._parser().parse_args(
        [
            "profile-complete-token",
            "--artifact-dir",
            "/tmp/artifact.gravity",
            "--prompt",
            "hello",
            "--trials",
            "1",
            "--decode-forwards",
            "0",
            "--out",
            "/tmp/profile.json",
        ]
    )
    assert args.command == "profile-complete-token"
    assert args.decode_forwards == 0
