"""Fixture-only tests for the guarded Q30 streamed-source outer contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lab.operators import ascension_qwen30_guarded_streamed_source_oracle_outer_controller as controller
from lab.operators import ascension_qwen30_quality_repack_raw_final_logit_retention_contract as retention_contract
from lab.operators import ascension_qwen30_quality_repack_source_bf16_memory_lease_preflight as memory_preflight
from lab.operators import ascension_qwen30_quality_repack_source_oracle_three_way_contract as three_way_contract
from lab.receipts import seal, verify


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SOURCE_BYTES = 61_066_575_656
TRACE = {
    "probe_id": "literal_hawking",
    "source_template_token_count": controller.PREFIX_TOKENS,
    "forced_identical_continuation_token_id": controller.FORCED_TOKEN,
    "source_template_token_ids_u32le_sha256": SHA_A,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, document: dict[str, object], *, sealed: bool = False) -> Path:
    value = seal(document) if sealed else document
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _source_contract(path: Path) -> Path:
    return _write(
        path,
        {
            "schema": three_way_contract.SCHEMA,
            "status": three_way_contract.STATUS,
            "exact_input": {
                **TRACE,
                "source_must_execute_the_same_369_token_prefix_then_the_forced_token": True,
                "sampling_or_autoregressive_feedback_is_forbidden": True,
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
        sealed=True,
    )


def _memory_preflight(path: Path, source_path: Path) -> Path:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    required = SOURCE_BYTES + 8 * 1024**3
    return _write(
        path,
        {
            "schema": memory_preflight.SCHEMA,
            "status": memory_preflight.BLOCKED_STATUS,
            "source_bf16_three_way_contract": {
                "path": str(source_path),
                "seal_sha256": source["seal_sha256"],
            },
            "measured_system_snapshot": {
                "physical_memory_bytes": 96 * 1024**3,
                "swap": {"used_bytes": 0},
                "vm_stat": {"reclaimable_bytes": 44 * 1024**3, "swapouts_pages": 0},
            },
            "headroom_assessment": {
                "source_weight_bytes": SOURCE_BYTES,
                "measured_reclaimable_bytes": 44 * 1024**3,
                "measured_swap_used_bytes": 0,
                "minimum_reclaimable_bytes_required_before_source_load": required,
            },
        },
        sealed=True,
    )


def _range_authority(path: Path) -> Path:
    return _write(
        path,
        {
            "authority_content_sha256": SHA_D,
            "authority": {
                "schema": controller.RANGE_AUTHORITY_SCHEMA,
                "status": controller.RANGE_AUTHORITY_STATUS,
                "source": {
                    "model_id": "Qwen3-Coder-30B-A3B-Instruct",
                    "source_revision": "b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
                    "source_tensor_count": 18_867,
                    "source_shard_count": 16,
                    "source_index": {"sha256": SHA_C, "weight_map_tensor_count": 18_867},
                },
                "exact_streamed_oracle_scope": {
                    "source_template_token_count": controller.PREFIX_TOKENS,
                    "forced_identical_continuation_token_id": controller.FORCED_TOKEN,
                    "sampling_or_autoregressive_feedback_forbidden": True,
                    "row_tile_rows": 128,
                },
                "metadata_access_boundary": {
                    "source_model_instantiated": False,
                    "gpu_or_metal_invoked": False,
                    "server_started": False,
                    "hcli_invoked": False,
                    "lease_requested": False,
                    "mmap_or_memory_map_used": False,
                    "tensor_payload_hashes_collected": False,
                    "whole_shard_payload_checksum_collected": False,
                    "source_tensor_payload_bytes_read": 0,
                },
                "tensors": [
                    {
                        "source_dtype": "BF16",
                        "row_window_shape": [128, 2048],
                    }
                ],
                "shards": [{"relative_path": "model-00001-of-00016.safetensors"}],
            },
        },
    )


def _semantics(path: Path, range_path: Path, source_path: Path) -> Path:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    return _write(
        path,
        {
            "schema": controller.SEMANTICS_SCHEMA,
            "status": controller.SEMANTICS_STATUS,
            "execution_boundary": {
                "source_tensor_payload_opened": False,
                "source_safetensors_or_other_weight_path_accepted": False,
                "source_model_instantiated": False,
                "source_inference_executed": False,
                "gpu_or_metal_invoked": False,
                "server_started": False,
                "hcli_invoked": False,
                "lease_requested": False,
                "source_quality_or_coherence_claim_made": False,
                "tps_or_tg_claim_made": False,
            },
            "pinned_source_binding": {
                "source_model_id": "Qwen3-Coder-30B-A3B-Instruct",
                "source_revision": "b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
                "source_index_sha256": SHA_C,
                "geometry": {
                    "layers": 48,
                    "hidden_size": 2048,
                    "vocab_size": controller.VOCAB_ROWS,
                    "attention_heads": 32,
                    "key_value_heads": 4,
                    "head_dim": 128,
                    "experts": 128,
                    "top_k": 8,
                    "moe_intermediate": 768,
                    "source_tensor_count": 18_867,
                    "source_shard_count": 16,
                },
            },
            "consumed_metadata_contracts": {
                "range_authority": {
                    "path": str(range_path),
                    "document_sha256": _sha256(range_path),
                    "authority_content_sha256": SHA_D,
                    "source_payload_read_by_this_attester": False,
                },
                "sealed_replay_contract": {
                    "path": str(source_path),
                    "document_sha256": _sha256(source_path),
                    "seal_sha256": source["seal_sha256"],
                },
            },
            "future_exact_execution_attestation": {
                "schema": controller.FUTURE_EXECUTION_SEMANTICS_SCHEMA,
                "status_only_after_real_separately_leased_source_execution": controller.FUTURE_EXECUTION_SEMANTICS_STATUS,
                "must_retain_and_hash_six_full_f32_endpoint_logit_vectors_before_any_three_way_scoring": True,
            },
        },
    )


def _raw_retention(path: Path, memory_path: Path, current_path: Path) -> Path:
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    current = json.loads(current_path.read_text(encoding="utf-8"))
    return _write(
        path,
        {
            "schema": retention_contract.SCHEMA,
            "status": retention_contract.STATUS,
            "strict_source_bf16_memory_preflight": {
                "path": str(memory_path),
                "seal_sha256": memory["seal_sha256"],
            },
            "replay_binding": {
                "exact_trace": TRACE,
                "candidate_local_comparison": {
                    "path": str(current_path),
                    "seal_sha256": current["seal_sha256"],
                },
            },
            "six_vector_retention_contract": retention_contract.raw_vector_plan(),
            "source_memory_and_eviction_gate": {
                "current_memory_preflight_status": memory_preflight.BLOCKED_STATUS,
                "must_evict_source_weights_and_confirm_release_before_native_capture": True,
                "source_and_native_model_bodies_must_not_be_resident_concurrently": True,
            },
        },
        sealed=True,
    )


def _current_trace(path: Path) -> Path:
    return _write(
        path,
        {
            "schema": three_way_contract.COMPARISON_SCHEMA,
            "status": three_way_contract.COMPARISON_STATUS,
            "binding": TRACE,
            "claim_boundary": {
                "does_not_claim_semantic_coherence_hcli_tps_tg_capability_or_tournament": True,
            },
        },
        sealed=True,
    )


def _fixtures(tmp_path: Path) -> dict[str, Path]:
    source = _source_contract(tmp_path / "source.json")
    memory = _memory_preflight(tmp_path / "memory.json", source)
    range_authority = _range_authority(tmp_path / "range.json")
    semantics = _semantics(tmp_path / "semantics.json", range_authority, source)
    current = _current_trace(tmp_path / "current.json")
    raw = _raw_retention(tmp_path / "raw.json", memory, current)
    return {
        "source_contract_path": source,
        "memory_preflight_path": memory,
        "range_authority_path": range_authority,
        "semantics_path": semantics,
        "raw_retention_path": raw,
        "current_trace_path": current,
    }


def _lease(schema: str, status: str, nonce: str, floor: int) -> dict[str, object]:
    return seal(
        {
            "schema": schema,
            "status": status,
            "one_shot_lifecycle": {
                "fresh_for_this_exact_launch": True,
                "prior_terminal_receipt": None,
                "automatic_retry_allowed": False,
                "new_capture_root": True,
                "existing_output_reuse_forbidden": True,
                "replay_or_relaunch_forbidden": True,
                "exact_launch_nonce": nonce,
            },
            "fresh_pre_child_safety": {
                "observed_immediately_before_child": True,
                "exclusive_clean_window": True,
                "no_source_or_native_model_body_resident_before_child": True,
                "swap_used_bytes": 0,
                "swapouts_pages_delta": 0,
                "reclaimable_bytes": floor + 1,
                "minimum_reclaimable_bytes_required": floor,
            },
        }
    )


def _source_payloads() -> dict[str, object]:
    return {
        endpoint: {
            "path": f"/tmp/source_bf16_{endpoint}_logits.f32le",
            "dtype": "f32le",
            "vocab_rows": controller.VOCAB_ROWS,
            "bytes": controller.F32_VECTOR_BYTES,
            "sha256": SHA_D,
            "all_values_finite": True,
        }
        for endpoint in retention_contract.ENDPOINTS
    }


def test_current_preflight_stays_blocked_and_binds_only_metadata(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    result = controller.build_current_preflight(**paths)
    assert result["status"] == controller.BLOCKED_STATUS
    assert result["future_source_launch_contract"]["actual_streamed_executor_present"] is False
    assert result["derived_current_streamed_feasibility"]["oracle_execution_authorized"] is False
    assert result["claim_boundary"]["no_child_launched"] is True
    assert result["claim_boundary"]["does_not_claim_source_quality_coherence_hcli_tps_tg_or_tournament"] is True
    verify(result, label="guarded streamed-source outer preflight")


def test_future_sequence_requires_fresh_safety_bounded_cache_and_accounting(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    preflight = controller.build_current_preflight(**paths)
    source = json.loads(paths["source_contract_path"].read_text(encoding="utf-8"))
    raw = json.loads(paths["raw_retention_path"].read_text(encoding="utf-8"))
    authority = json.loads(paths["range_authority_path"].read_text(encoding="utf-8"))["authority"]
    floor = preflight["future_source_launch_contract"][
        "minimum_reclaimable_bytes_required_immediately_before_source_child"
    ]
    window = preflight["metadata_only_range_authority"]["maximum_declared_single_bf16_row_window_bytes"]
    source_lease = _lease(controller.SOURCE_LEASE_SCHEMA, controller.SOURCE_LEASE_STATUS, SHA_A, floor)
    source_terminal = seal(
        {
            "schema": controller.SOURCE_TERMINAL_SCHEMA,
            "status": controller.SOURCE_TERMINAL_STATUS,
            "source_lease": {"seal_sha256": source_lease["seal_sha256"]},
            "streamed_execution": {
                "mode": "layer_streamed_bf16_source_teacher",
                "outer_reaped_child_before_terminal_receipt": True,
                "receipt_written_after_payload_fsyncs": True,
            },
            "exact_trace": TRACE,
            "bounded_per_read_cache": {
                "maximum_allowed_window_bytes": window,
                "maximum_observed_window_bytes": window,
                "maximum_cached_bytes": window,
                "maximum_cached_windows": 1,
                "eviction_on_each_read_completion": True,
                "complete_source_shard_mapped_or_cached": False,
                "mmap_or_memory_map_used": False,
            },
            "source_payload_read_accounting": {
                "all_source_payload_reads_accounted": True,
                "source_tensor_payload_reads_executed": True,
                "source_tensor_payload_bytes_read": 4096,
                "source_tensor_payload_read_calls": 1,
                "per_shard": [
                    {
                        "relative_path": "model-00001-of-00016.safetensors",
                        "payload_bytes_read": 4096,
                        "read_calls": 1,
                        "whole_shard_read_as_one_window": False,
                        "whole_shard_cached": False,
                    }
                ],
            },
            "source_payloads": _source_payloads(),
        }
    )
    source_eviction = seal(
        {
            "schema": controller.SOURCE_EVICTION_SCHEMA,
            "status": controller.SOURCE_EVICTION_STATUS,
            "source_teacher_terminal": {"seal_sha256": source_terminal["seal_sha256"]},
            "eviction": {
                "source_weights_evicted": True,
                "source_backend_shutdown": True,
                "source_model_residency_released": True,
                "streamed_reader_cache_cleared": True,
                "source_payloads_durable_and_immutable": True,
                "swap_remained_zero": True,
                "pre_native_lease_process_tree_checked": True,
            },
        }
    )
    native_lease = _lease(controller.NATIVE_LEASE_SCHEMA, controller.NATIVE_LEASE_STATUS, SHA_B, 1)
    native_lease = seal(
        {
            **native_lease,
            "source_eviction": {"seal_sha256": source_eviction["seal_sha256"]},
            "raw_retention_contract": {"seal_sha256": raw["seal_sha256"]},
        }
    )
    result = controller.validate_future_source_then_evict_then_native(
        source_lease=source_lease,
        source_terminal=source_terminal,
        source_eviction=source_eviction,
        native_lease=native_lease,
        authority=authority,
        raw_retention=raw,
        trace=TRACE,
        source_minimum_reclaimable_bytes=floor,
        maximum_window_bytes=window,
    )
    assert result["validated_order"] == ["source_streamed", "source_evicted", "native_lease"]
    assert result["metadata_validation_only_no_child_launched"] is True
    source_lease = seal(
        {
            **source_lease,
            "fresh_pre_child_safety": {
                **source_lease["fresh_pre_child_safety"],  # type: ignore[index]
                "swap_used_bytes": 1,
            },
        }
    )
    with pytest.raises(controller.GuardedStreamedSourceOuterError, match="zero swap"):
        controller.validate_future_source_then_evict_then_native(
            source_lease=source_lease,
            source_terminal=source_terminal,
            source_eviction=source_eviction,
            native_lease=native_lease,
            authority=authority,
            raw_retention=raw,
            trace=TRACE,
            source_minimum_reclaimable_bytes=floor,
            maximum_window_bytes=window,
        )


def test_future_sequence_rejects_replayed_lease_metadata(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    preflight = controller.build_current_preflight(**paths)
    floor = preflight["future_source_launch_contract"][
        "minimum_reclaimable_bytes_required_immediately_before_source_child"
    ]
    lease = _lease(controller.SOURCE_LEASE_SCHEMA, controller.SOURCE_LEASE_STATUS, SHA_A, floor)
    lease["one_shot_lifecycle"]["prior_terminal_receipt"] = {"seal_sha256": SHA_C}  # type: ignore[index]
    with pytest.raises(controller.GuardedStreamedSourceOuterError, match="reuse a terminal receipt"):
        controller.validate_fresh_zero_swap_safety(lease=lease, minimum_reclaimable_bytes=floor, label="fixture")
        controller._one_shot_lifecycle(lease, label="fixture")


def test_output_is_create_new_and_controller_has_no_child_launcher(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    result = controller.build_current_preflight(**paths)
    output = tmp_path / "blocked.json"
    controller._write_new_json(output, result)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == controller.BLOCKED_STATUS
    with pytest.raises(controller.GuardedStreamedSourceOuterError, match="new immutable"):
        controller._write_new_json(output, result)
    assert "subprocess" not in controller.__dict__
