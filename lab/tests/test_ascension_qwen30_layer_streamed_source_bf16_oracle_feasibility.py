"""CPU-only tests for the Q30 streamed source-BF16 feasibility gate."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.operators import ascension_qwen30_layer_streamed_source_bf16_oracle_feasibility as feasibility
from lab.receipts import seal, verify


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SOURCE_BYTES = 61_066_575_656


def _write(path: Path, value: dict[str, object]) -> Path:
    path.write_text(json.dumps(seal(value), sort_keys=True), encoding="utf-8")
    return path


def _source_contract(path: Path) -> Path:
    return _write(
        path,
        {
            "schema": feasibility.SOURCE_CONTRACT_SCHEMA,
            "status": feasibility.SOURCE_CONTRACT_STATUS,
            "exact_input": {
                "source_template_token_count": feasibility.PREFIX_TOKENS,
                "forced_identical_continuation_token_id": feasibility.FORCED_TOKEN_ID,
                "source_must_execute_the_same_369_token_prefix_then_the_forced_token": True,
                "sampling_or_autoregressive_feedback_is_forbidden": True,
                "source_template_token_ids_u32le_sha256": SHA_A,
            },
            "evidence": {
                "source_config": {"sha256": SHA_B},
                "source_index": {"sha256": SHA_C},
                "source_weight_bytes_exact": SOURCE_BYTES,
            },
            "resource_and_capture_requirements": {
                "source_weights_static_lower_bound_bytes": SOURCE_BYTES,
                "source_model_has_not_been_loaded_by_this_preflight": True,
            },
        },
    )


def _preflight(path: Path, source_contract: Path, *, reclaimable: int, swap: int = 0) -> Path:
    source = json.loads(source_contract.read_text(encoding="utf-8"))
    whole_model_required = SOURCE_BYTES + 8 * 1024**3
    status = (
        feasibility.WHOLE_MODEL_PREFLIGHT_READY_STATUS
        if reclaimable >= whole_model_required and swap == 0
        else feasibility.WHOLE_MODEL_PREFLIGHT_BLOCKED_STATUS
    )
    return _write(
        path,
        {
            "schema": feasibility.WHOLE_MODEL_PREFLIGHT_SCHEMA,
            "status": status,
            "source_bf16_three_way_contract": {"seal_sha256": source["seal_sha256"]},
            "measured_system_snapshot": {
                "physical_memory_bytes": 96 * 1024**3,
                "vm_stat": {"reclaimable_bytes": reclaimable},
                "swap": {"used_bytes": swap},
            },
            "headroom_assessment": {
                "source_weight_bytes": SOURCE_BYTES,
                "minimum_reclaimable_bytes_required_before_source_load": whole_model_required,
            },
        },
    )


def _semantics(path: Path, source_contract: Path, **overrides: object) -> Path:
    source = json.loads(source_contract.read_text(encoding="utf-8"))
    source_binding = {
        "source_config_sha256": source["evidence"]["source_config"]["sha256"],
        "source_index_sha256": source["evidence"]["source_index"]["sha256"],
        "source_template_token_ids_u32le_sha256": source["exact_input"]["source_template_token_ids_u32le_sha256"],
        "source_weight_bytes": SOURCE_BYTES,
        "source_tensor_count": feasibility.SOURCE_TENSOR_COUNT,
    }
    document: dict[str, object] = {
        "schema": feasibility.SEMANTICS_SCHEMA,
        "status": feasibility.SEMANTICS_STATUS,
        "source_binding": source_binding,
        "exact_trace": {
            "prefix_token_count": feasibility.PREFIX_TOKENS,
            "forced_token_id": feasibility.FORCED_TOKEN_ID,
            "prefill_then_forced_cache_order": True,
            "sampling_or_autoregressive_feedback_forbidden": True,
        },
        "exact_semantics": {
            "source_bf16_tensor_row_order_and_offsets_verified": True,
            "range_reader_never_maps_or_caches_a_complete_source_shard": True,
            "source_rmsnorm_rope_attention_router_topk_moe_operator_order_verified": True,
            "source_accumulation_and_expert_combine_order_verified": True,
            "all_48_layers_and_final_norm_head_verified": True,
            "full_f32_final_logits_at_prefix_and_forced_endpoints_verified": True,
        },
        "working_set_policy": dict(feasibility.DEFAULT_POLICY),
    }
    for key, value in overrides.items():
        document[key] = value
    return _write(path, document)


def _inputs(tmp_path: Path, *, reclaimable: int = 44 * 1024**3, swap: int = 0) -> tuple[Path, Path]:
    source = _source_contract(tmp_path / "source.json")
    preflight = _preflight(tmp_path / "preflight.json", source, reclaimable=reclaimable, swap=swap)
    return source, preflight


def test_current_no_attestation_refuses_but_models_the_small_streamed_working_set(tmp_path: Path) -> None:
    source, preflight = _inputs(tmp_path)
    result = feasibility.build_feasibility(
        source_contract_path=source,
        whole_model_preflight_path=preflight,
    )
    assert result["status"] == feasibility.REFUSED_STATUS
    assert result["feasibility"]["oracle_execution_authorized"] is False
    assert result["memory_assessment"]["streamed_memory_arithmetic_fits"] is True
    assert result["memory_assessment"]["whole_model_source_residency_deficit_bytes"] > 0
    assert result["working_set"]["modeled_working_set_before_allocator_reserve_bytes"] < SOURCE_BYTES
    assert result["streaming_semantics_attestation"]["present"] is False
    verify(result, label="streamed feasibility")


def test_derives_all_48_layer_windows_and_terminal_all_row_head(tmp_path: Path) -> None:
    source, preflight = _inputs(tmp_path)
    result = feasibility.build_feasibility(source_contract_path=source, whole_model_preflight_path=preflight)
    windows = result["working_set"]["per_layer_windows"]
    assert len(windows) == feasibility.LAYER_COUNT
    assert windows[0]["source_layer_index"] == 0
    assert windows[-1]["source_layer_index"] == 47
    assert windows[3]["tensors"][3]["full_shape"] == [4096, 2048]
    assert windows[3]["tensors"][8]["full_shape"] == [128, 2048]
    lm_head = result["working_set"]["terminal_windows"][-1]
    assert lm_head["source_tensor"] == "lm_head.weight"
    assert lm_head["full_shape"] == [feasibility.VOCAB_ROWS, feasibility.HIDDEN_SIZE]
    assert lm_head["row_window_shape"] == [128, feasibility.HIDDEN_SIZE]
    assert result["working_set"]["persistent_kv_cache_bytes"] == 36_372_480


def test_future_sealed_semantics_can_prepare_only_the_feasibility_model(tmp_path: Path) -> None:
    source, preflight = _inputs(tmp_path)
    semantics = _semantics(tmp_path / "semantics.json", source)
    result = feasibility.build_feasibility(
        source_contract_path=source,
        whole_model_preflight_path=preflight,
        semantics_attestation_path=semantics,
    )
    assert result["status"] == feasibility.PREPARED_STATUS
    assert result["feasibility"]["safe_streamed_plan_prepared_not_executed"] is True
    assert result["feasibility"]["oracle_execution_authorized"] is False
    assert result["memory_assessment"]["whole_model_residency_block_remains_authoritative"] is True


def test_nonzero_swap_hard_refuses_even_with_exact_semantics(tmp_path: Path) -> None:
    source, preflight = _inputs(tmp_path, swap=1)
    semantics = _semantics(tmp_path / "semantics.json", source)
    result = feasibility.build_feasibility(
        source_contract_path=source,
        whole_model_preflight_path=preflight,
        semantics_attestation_path=semantics,
    )
    assert result["status"] == feasibility.REFUSED_STATUS
    assert "measured swap usage is non-zero" in result["feasibility"]["refusal_reasons"]


def test_insufficient_streamed_margin_hard_refuses(tmp_path: Path) -> None:
    source, preflight = _inputs(tmp_path, reclaimable=64 * 1024**2)
    semantics = _semantics(tmp_path / "semantics.json", source)
    result = feasibility.build_feasibility(
        source_contract_path=source,
        whole_model_preflight_path=preflight,
        semantics_attestation_path=semantics,
    )
    assert result["status"] == feasibility.REFUSED_STATUS
    assert result["memory_assessment"]["streamed_memory_arithmetic_fits"] is False


def test_source_binding_or_trace_mismatch_is_rejected(tmp_path: Path) -> None:
    source, preflight = _inputs(tmp_path)
    bad_semantics = _semantics(
        tmp_path / "semantics.json",
        source,
        exact_trace={
            "prefix_token_count": 368,
            "forced_token_id": feasibility.FORCED_TOKEN_ID,
            "prefill_then_forced_cache_order": True,
            "sampling_or_autoregressive_feedback_forbidden": True,
        },
    )
    with pytest.raises(feasibility.LayerStreamedOracleFeasibilityError, match="prefix length differs"):
        feasibility.build_feasibility(
            source_contract_path=source,
            whole_model_preflight_path=preflight,
            semantics_attestation_path=bad_semantics,
        )


def test_create_new_output_refuses_replay_and_requires_absolute_path(tmp_path: Path) -> None:
    source, preflight = _inputs(tmp_path)
    result = feasibility.build_feasibility(source_contract_path=source, whole_model_preflight_path=preflight)
    output = tmp_path / "result.json"
    feasibility._write_new_json(output, result)
    assert json.loads(output.read_text(encoding="utf-8"))["seal_sha256"] == result["seal_sha256"]
    with pytest.raises(feasibility.LayerStreamedOracleFeasibilityError, match="new file"):
        feasibility._write_new_json(output, result)
    with pytest.raises(feasibility.LayerStreamedOracleFeasibilityError, match="absolute"):
        feasibility._write_new_json(Path("relative.json"), result)
