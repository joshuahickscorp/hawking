"""Focused synthetic-only tests for the Q30 production source-teacher outer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.operators import (
    ascension_qwen30_production_source_teacher_outer_lifecycle_preflight as outer,
)
from lab.receipts import seal, verify

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
GIT_REVISION = "f" * 40


def _write_sealed(path: Path, document: dict[str, object]) -> Path:
    path.write_text(json.dumps(seal(document), sort_keys=True), encoding="utf-8")
    return path


def _document(path: Path, document: dict[str, object]) -> outer.Document:
    _write_sealed(path, document)
    return outer._sealed(path, label=path.name)


def _production_antecedents(tmp_path: Path, *, fixture_map: bool = False) -> dict[str, outer.Document]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    flat_map = _document(
        tmp_path / "flat-map.json",
        {
            "schema": outer.FLAT_MAP_SCHEMA,
            "source_revision": GIT_REVISION,
            "source_tensor_count": outer.SOURCE_TENSORS,
            "maximum_window_bytes": outer.MAX_POSITIONED_READ_BYTES,
            "source_index": {"sha256": SHA_B},
            "source_model_id": "Qwen3-Coder-30B-A3B-Instruct",
            "shards": [{} for _ in range(outer.SOURCE_SHARDS)],
            "tensors": [{} for _ in range(outer.SOURCE_TENSORS)],
            "fixture_only": fixture_map,
            "synthetic_fixture_only": False,
            "production_adapter_forbidden": False,
        },
    )
    coverage = _document(
        tmp_path / "coverage.json",
        {
            "schema": outer.HASH_COVERAGE_SCHEMA,
            "status": outer.HASH_COVERAGE_STATUS,
            "fixture_only": False,
            "synthetic_fixture_only": False,
            "production_hash_coverage_earned": True,
            "source_teacher_execution_or_logits": False,
            "source_teacher_runtime_admission_earned": False,
            "operator_or_reader_execution_attestation_emitted": False,
            "flat_runtime_range_map": outer._evidence(flat_map),
            "bounded_positioned_reader": {
                "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
                "maximum_live_raw_bf16_windows": 1,
                "one_shard_handle_at_a_time": True,
                "whole_shard_cache_or_mmap_forbidden": True,
                "cache_zeroed_after_every_visit_and_before_receipt": True,
            },
            "coverage": {
                "source_shards": outer.SOURCE_SHARDS,
                "source_tensors": outer.SOURCE_TENSORS,
                "source_index_sha256": SHA_B,
            },
        },
    )
    child = _document(
        tmp_path / "production-child.json",
        {
            "schema": outer.PRODUCTION_CHILD_SCHEMA,
            "status": outer.PRODUCTION_CHILD_STATUS,
            "fixture_only": False,
            "synthetic_fixture_only": False,
            "production_hash_scan_earned": True,
            "source_handles_closed": True,
            "reader_cache_zeroed": True,
            "receipt_written_last": True,
            "source_teacher_or_logits_executed": False,
            "source_teacher_runtime_admission_earned": False,
            "operator_or_reader_execution_attestation_emitted": False,
            "model_gpu_server_hcli_or_tps_action": False,
            "flat_runtime_range_map": outer._evidence(flat_map),
            "hash_coverage_attestation": outer._evidence(coverage),
            "geometry": {
                "source_shards": outer.SOURCE_SHARDS,
                "source_tensors": outer.SOURCE_TENSORS,
                "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
                "maximum_live_raw_bf16_windows": 1,
            },
        },
    )
    terminal = _document(
        tmp_path / "production-terminal.json",
        {
            "schema": outer.PRODUCTION_OUTER_SCHEMA,
            "status": outer.PRODUCTION_OUTER_STATUS,
            "child_reaped": True,
            "terminal_receipt_written_after_child_capture": True,
            "terminal_receipt_written_last": True,
            "automatic_retry_disabled": True,
            "lease_reuse_prohibited": True,
            "child_timed_out": False,
            "child_exit_code": 0,
            "child_capture": outer._evidence(child),
            "child_capture_seal_sha256": child.seal_sha256,
        },
    )
    release = _document(
        tmp_path / "production-release.json",
        {
            "schema": outer.PRODUCTION_RELEASE_SCHEMA,
            "status": outer.PRODUCTION_RELEASE_STATUS,
            "release_after_outer_terminal": True,
            "one_shot_lease_finalized": True,
            "retry_or_relaunch_forbidden": True,
            "source_teacher_or_logits_authorized": False,
            "native_or_gpu_server_hcli_authorized": False,
            "artifacts_deleted_or_evicted": False,
            "outer_terminal_seal_sha256": terminal.seal_sha256,
            "child_capture_seal_sha256": child.seal_sha256,
        },
    )
    return {
        "production_outer_terminal": terminal,
        "production_child_capture": child,
        "production_flat_map": flat_map,
        "production_hash_coverage": coverage,
        "production_lease_release": release,
    }


def _future_authorities(
    tmp_path: Path,
    antecedents: dict[str, outer.Document],
    *,
    hash_only_runtime: bool = False,
) -> dict[str, outer.Document]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    bridge = _document(
        tmp_path / "post-hash-map-bridge.json",
        {
            "schema": outer.POST_HASH_MAP_BRIDGE_SCHEMA,
            "status": outer.POST_HASH_MAP_BRIDGE_STATUS,
            "prepared": True,
            "fixture_only": False,
            "execution_authorized": False,
            "runtime_admission_earned": False,
            "dual_attestation_runtime_admission_emitted": False,
            "post_hash_map_antecedents": {
                key: outer._evidence(document) for key, document in antecedents.items()
            },
            "validated_production_hash_scan": {
                "non_fixture_production_flat_map": True,
                "source_shards": outer.SOURCE_SHARDS,
                "source_tensors": outer.SOURCE_TENSORS,
                "all_full_shard_and_raw_bf16_hash_coverage_bound": True,
                "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
                "maximum_live_raw_bf16_windows": 1,
                "one_shard_handle_at_a_time": True,
                "reader_cache_zeroed_before_hash_scan_receipt": True,
                "source_handles_closed_before_hash_scan_receipt": True,
                "one_shot_replay_reservation_attempt": 1,
                "outer_child_reaped_successfully": True,
                "lease_finalized_after_outer_terminal": True,
                "source_teacher_execution_or_logits": False,
                "operator_or_reader_execution_attestation_emitted": False,
                "source_teacher_runtime_admission_earned": False,
            },
            "future_source_teacher_provenance_reservation": {
                "reservation_status": "NOT_EXECUTED",
                "post_hash_map_bridge_is_not_runtime_admission": True,
                "post_hash_map_bridge_is_not_dual_attestation_bridge": True,
                "runtime_admission": {
                    "schema": outer.RUNTIME_ADMISSION_SCHEMA,
                    "status": outer.RUNTIME_ADMISSION_STATUS,
                    "must_be_sealed": True,
                    "must_bind_post_hash_map_antecedents": True,
                    "must_bind_flat_map_and_coverage_canonical_hashes": True,
                    "must_bind_both_future_execution_attestation_seals": True,
                    "must_precede_any_source_root_or_payload_open": True,
                    "not_emitted_by_this_bridge": True,
                },
                "dual_attestation_runtime_admission": {
                    "schema": outer.DUAL_BRIDGE_SCHEMA,
                    "status": outer.DUAL_BRIDGE_STATUS,
                    "must_be_sealed": True,
                    "must_bind_this_post_hash_map_bridge_seal_sha256": True,
                    "must_bind_runtime_admission_seal_sha256": True,
                    "must_preserve_existing_source_teacher_child_schema_resolution": True,
                    "not_emitted_by_this_bridge": True,
                },
                "existing_source_teacher_child_compatible_shape": {
                    "schema_resolution": {
                        "runtime_range_map_schema": outer.FLAT_MAP_SCHEMA,
                        "runtime_admission_schema": outer.RUNTIME_ADMISSION_SCHEMA,
                        "runtime_admission_status_only_after_bounded_source_validation": outer.RUNTIME_ADMISSION_STATUS,
                        "operator_accumulation_execution_attestation": {
                            "schema": outer.OPERATOR_ATTESTATION_SCHEMA,
                            "status": outer.OPERATOR_ATTESTATION_STATUS,
                        },
                        "range_reader_exact_semantics_attestation": {
                            "schema": outer.RANGE_READER_ATTESTATION_SCHEMA,
                            "status": outer.RANGE_READER_ATTESTATION_STATUS,
                        },
                        "both_execution_attestations_required_after_source_child": True,
                        "runtime_range_admission_required_before_payload_open": True,
                        "bridge_does_not_authorize_execution": True,
                    },
                    "future_source_worker": {
                        "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
                        "source_layers": outer.SOURCE_LAYERS,
                        "source_forwards": outer.SOURCE_FORWARDS,
                        "source_f32le_vectors": outer.SOURCE_VECTORS,
                        "native_f32le_vectors": outer.NATIVE_VECTORS,
                        "one_bounded_window_only": True,
                        "source_payloads_durable_before_eviction": True,
                        "close_handles_and_clear_cache_before_eviction_receipt": True,
                        "separate_native_four_vector_phase_required": True,
                    },
                },
            },
            "admission_before_open_cycle": {
                "runtime_admission_must_be_earned_before_source_root_open": True,
                "runtime_producer_requires_bounded_source_validation_and_both_execution_attestations": True,
                "existing_source_teacher_child_requires_runtime_admission_before_source_root_open": True,
                "resolved": False,
                "bridge_does_not_relax_or_reorder_any_requirement": True,
            },
            "execution_boundary": {
                "source_root_opened_or_statted": False,
                "source_payload_opened": False,
                "source_model_loaded_or_instantiated": False,
                "whole_source_model_resident": False,
                "source_teacher_or_logits_executed": False,
                "operator_or_reader_execution_attestation_emitted": False,
                "source_teacher_runtime_admission_emitted": False,
                "gpu_native_server_hcli_or_tps_action": False,
                "lease_issued_or_consumed": False,
                "child_process_started": False,
            },
        },
    )
    dual = _document(
        tmp_path / "dual-bridge.json",
        {
            "schema": outer.DUAL_BRIDGE_SCHEMA,
            "status": outer.DUAL_BRIDGE_STATUS,
            "fixture_only": False,
            "post_hash_map_bridge": outer._evidence(bridge),
            "schema_resolution": {
                "runtime_range_map_schema": outer.FLAT_MAP_SCHEMA,
                "runtime_admission_schema": outer.RUNTIME_ADMISSION_SCHEMA,
                "runtime_admission_status_only_after_bounded_source_validation": outer.RUNTIME_ADMISSION_STATUS,
                "both_execution_attestations_required_after_source_child": True,
                "runtime_range_admission_required_before_payload_open": True,
                "bridge_does_not_authorize_execution": True,
            },
            "future_source_worker": {
                "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
                "source_layers": outer.SOURCE_LAYERS,
                "source_forwards": outer.SOURCE_FORWARDS,
                "source_f32le_vectors": outer.SOURCE_VECTORS,
                "native_f32le_vectors": outer.NATIVE_VECTORS,
                "one_bounded_window_only": True,
                "source_payloads_durable_before_eviction": True,
                "close_handles_and_clear_cache_before_eviction_receipt": True,
                "separate_native_four_vector_phase_required": True,
            },
        },
    )
    runtime = _document(
        tmp_path / "runtime-admission.json",
        {
            "schema": outer.RUNTIME_ADMISSION_SCHEMA,
            "status": outer.RUNTIME_ADMISSION_STATUS,
            "fixture_only": False,
            "post_hash_map_bridge": outer._evidence(bridge),
            "source_teacher_execution_admission_not_hash_only": not hash_only_runtime,
            "production_hash_scan_is_not_source_teacher_execution": not hash_only_runtime,
            "flat_runtime_range_map": {
                "schema": outer.FLAT_MAP_SCHEMA,
                "document_sha256": antecedents["production_flat_map"].canonical_document_sha256,
            },
            "bounded_positioned_reader": {
                "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
                "maximum_live_raw_bf16_windows": 1,
                "no_mmap_or_full_shard_cache": True,
                "no_model_residency": True,
                "payload_open_requires_fresh_source_lease": True,
            },
            "dual_bridge_seal_sha256": dual.seal_sha256,
            "execution_boundary": {
                "source_tensor_payload_opened": False,
                "source_model_loaded_or_instantiated": False,
                "gpu_or_metal_invoked": False,
                "server_started_or_contacted": False,
                "hcli_invoked": False,
                "lease_issued_or_consumed": False,
            },
        },
    )
    resource = _document(
        tmp_path / "source-resource.json",
        {
            "schema": outer.SOURCE_RESOURCE_SCHEMA,
            "status": outer.SOURCE_RESOURCE_STATUS,
            "fixture_only": False,
            "source_root_opened_or_statted": False,
            "source_teacher_or_native_child_started": False,
            "lease_issued_or_consumed": False,
            "post_hash_map_bridge": outer._evidence(bridge),
            "dual_attestation_runtime_admission": outer._evidence(dual),
            "runtime_execution_admission": outer._evidence(runtime),
            "fresh_pre_child_safety": {
                "observed_immediately_before_source_teacher_lease": True,
                "exclusive_clean_window": True,
                "no_active_q30_or_q80_capture_child": True,
                "no_source_or_native_model_body_resident": True,
                "swap_used_bytes": 0,
                "swapouts_pages_delta": 0,
                "reclaimable_bytes": 2_000_000,
                "minimum_reclaimable_bytes_required": 1_000_000,
            },
        },
    )
    lease = _document(
        tmp_path / "source-lease.json",
        {
            "schema": outer.SOURCE_LEASE_SCHEMA,
            "status": outer.SOURCE_LEASE_STATUS,
            "fixture_only": False,
            "lease_id": SHA_C,
            "post_hash_map_bridge": outer._evidence(bridge),
            "dual_attestation_runtime_admission": outer._evidence(dual),
            "runtime_execution_admission": outer._evidence(runtime),
            "source_teacher_resource_admission": outer._evidence(resource),
            "one_shot_lifecycle": {
                "fresh_for_this_exact_launch": True,
                "new_capture_root": True,
                "existing_output_reuse_forbidden": True,
                "replay_or_relaunch_forbidden": True,
                "automatic_retry_allowed": False,
                "prior_terminal_receipt": None,
                "exact_launch_nonce": SHA_D,
            },
            "fresh_pre_child_safety": {
                "observed_immediately_before_child": True,
                "exclusive_clean_window": True,
                "no_source_or_native_model_body_resident_before_child": True,
                "swap_used_bytes": 0,
                "swapouts_pages_delta": 0,
                "reclaimable_bytes": 2_000_000,
                "minimum_reclaimable_bytes_required": 1_000_000,
            },
        },
    )
    return {
        "post_hash_map_bridge": bridge,
        "dual_bridge": dual,
        "runtime_execution_admission": runtime,
        "source_teacher_resource": resource,
        "source_teacher_lease": lease,
    }


def _preflight_kwargs(
    antecedents: dict[str, outer.Document], authorities: dict[str, outer.Document] | None = None
) -> dict[str, Path]:
    kwargs = {f"{key}_path": document.path for key, document in antecedents.items()}
    if authorities is not None:
        kwargs.update(
            {
                "post_hash_map_bridge_path": authorities["post_hash_map_bridge"].path,
                "dual_bridge_path": authorities["dual_bridge"].path,
                "runtime_execution_admission_path": authorities["runtime_execution_admission"].path,
                "source_teacher_resource_path": authorities["source_teacher_resource"].path,
                "source_teacher_lease_path": authorities["source_teacher_lease"].path,
            }
        )
    return kwargs


def test_current_antecedents_without_future_teacher_admissions_refuse_before_spawn(
    tmp_path: Path,
) -> None:
    antecedents = _production_antecedents(tmp_path)

    result = outer.build_production_source_teacher_preflight(
        **_preflight_kwargs(antecedents)
    )

    assert result["status"] == outer.REFUSED_STATUS
    assert result["prepared"] is False
    assert result["spawn_permitted"] is False
    assert "sealed_non_fixture_post_hash_map_bridge_absent" in result["blockers"]
    assert "sealed_non_fixture_runtime_execution_admission_absent" in result["blockers"]
    assert "fresh_source_teacher_resource_admission_absent" in result["blockers"]
    assert "fresh_one_shot_source_teacher_lease_absent" in result["blockers"]
    assert all(value is False for value in result["execution_boundary"].values())
    verify(result, label="current production teacher refusal")


def test_future_non_fixture_chain_prepares_only_a_non_authorizing_lifecycle(
    tmp_path: Path,
) -> None:
    antecedents = _production_antecedents(tmp_path)
    authorities = _future_authorities(tmp_path, antecedents)

    result = outer.build_production_source_teacher_preflight(
        **_preflight_kwargs(antecedents, authorities)
    )

    assert result["status"] == outer.PREPARED_STATUS
    assert result["prepared"] is True
    assert result["spawn_permitted"] is False
    assert result["source_teacher_lease_id"] == SHA_C
    assert all(value is False for value in result["execution_boundary"].values())
    assert result["future_lifecycle"]["source_child_capture"]["source_vectors_f32le"] == 2
    assert result["future_lifecycle"]["distinct_native_handoff"]["native_action_not_authorized_by_this_preflight"] is True
    verify(result, label="prepared production teacher lifecycle")


def test_fixture_map_and_hash_only_relabelled_runtime_admission_hard_refuse(
    tmp_path: Path,
) -> None:
    fixture_antecedents = _production_antecedents(tmp_path / "fixture", fixture_map=True)
    fixture_result = outer.build_production_source_teacher_preflight(
        **_preflight_kwargs(fixture_antecedents)
    )
    assert fixture_result["status"] == outer.REFUSED_STATUS
    assert any(
        str(blocker).startswith("earned_production_hash_scan_antecedent_chain_invalid:")
        for blocker in fixture_result["blockers"]
    )

    valid_antecedents = _production_antecedents(tmp_path / "relabel")
    hash_only_authorities = _future_authorities(
        tmp_path / "relabel", valid_antecedents, hash_only_runtime=True
    )
    relabel_result = outer.build_production_source_teacher_preflight(
        **_preflight_kwargs(valid_antecedents, hash_only_authorities)
    )
    assert relabel_result["status"] == outer.REFUSED_STATUS
    assert any(
        str(blocker).startswith("sealed_runtime_execution_admission_invalid:")
        for blocker in relabel_result["blockers"]
    )
    assert relabel_result["spawn_permitted"] is False


def test_fake_lifecycle_requires_receipt_last_eviction_before_distinct_native_lease() -> None:
    replay = {
        "schema": outer.REPLAY_SCHEMA,
        "status": outer.REPLAY_STATUS,
        "create_new_before_source_child": True,
        "one_source_child_maximum": True,
        "automatic_retry_or_relaunch_forbidden": True,
        "attempt": 1,
        "outer_preflight_seal_sha256": SHA_A,
        "source_teacher_lease_id": SHA_C,
    }
    child = seal(
        {
            "schema": outer.SOURCE_CHILD_CAPTURE_SCHEMA,
            "status": outer.SOURCE_CHILD_CAPTURE_STATUS,
            "source_teacher_execution_completed": True,
            "two_source_f32le_vectors_fsynced_before_child_exit": True,
            "source_handles_closed_before_child_exit": True,
            "reader_cache_zeroed_before_child_exit": True,
            "receipt_written_last": True,
            "native_phase_started": False,
            "gpu_server_hcli_or_tps_action": False,
            "geometry": {
                "source_layers": outer.SOURCE_LAYERS,
                "source_forwards": outer.SOURCE_FORWARDS,
                "source_vectors_f32le": outer.SOURCE_VECTORS,
                "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
                "maximum_live_raw_bf16_windows": 1,
            },
            "post_hash_map_bridge_seal_sha256": SHA_B,
            "dual_bridge_seal_sha256": SHA_D,
            "runtime_execution_admission_seal_sha256": SHA_E,
            "source_teacher_resource_seal_sha256": SHA_A,
            "source_teacher_lease_id": SHA_C,
        }
    )
    terminal = seal(
        {
            "schema": outer.SOURCE_TERMINAL_SCHEMA,
            "status": outer.SOURCE_TERMINAL_STATUS,
            "outer_reaped_source_child_before_terminal_receipt": True,
            "terminal_receipt_written_after_child_capture": True,
            "terminal_receipt_written_last": True,
            "native_phase_started": False,
            "source_teacher_lease_id": SHA_C,
            "source_child_capture_seal_sha256": child["seal_sha256"],
        }
    )
    eviction = seal(
        {
            "schema": outer.SOURCE_EVICTION_SCHEMA,
            "status": outer.SOURCE_EVICTION_STATUS,
            "source_weights_evicted": True,
            "source_backend_shutdown": True,
            "source_model_residency_released": True,
            "streamed_reader_cache_cleared": True,
            "source_payloads_durable_and_immutable": True,
            "swap_remained_zero": True,
            "pre_native_lease_process_tree_checked": True,
            "native_phase_started": False,
            "source_terminal_seal_sha256": terminal["seal_sha256"],
        }
    )
    native_lease = {
        "schema": outer.NATIVE_LEASE_SCHEMA,
        "status": outer.NATIVE_LEASE_STATUS,
        "fresh_for_this_exact_native_launch": True,
        "new_capture_root": True,
        "replay_or_relaunch_forbidden": True,
        "source_eviction_verified_before_native_lease": True,
        "lease_id": SHA_B,
        "source_eviction_seal_sha256": eviction["seal_sha256"],
    }

    outer.validate_fake_future_lifecycle(
        replay_reservation=replay,
        source_child_capture=child,
        source_terminal=terminal,
        source_eviction=eviction,
        native_lease=native_lease,
        outer_preflight_seal_sha256=SHA_A,
        post_hash_map_bridge_seal_sha256=SHA_B,
        dual_bridge_seal_sha256=SHA_D,
        runtime_execution_admission_seal_sha256=SHA_E,
        source_teacher_resource_seal_sha256=SHA_A,
        source_teacher_lease_id=SHA_C,
    )

    child["reader_cache_zeroed_before_child_exit"] = False
    with pytest.raises(outer.ProductionSourceTeacherOuterError, match="reader_cache_zeroed"):
        outer.validate_fake_future_lifecycle(
            replay_reservation=replay,
            source_child_capture=child,
            source_terminal=terminal,
            source_eviction=eviction,
            native_lease=native_lease,
            outer_preflight_seal_sha256=SHA_A,
            post_hash_map_bridge_seal_sha256=SHA_B,
            dual_bridge_seal_sha256=SHA_D,
            runtime_execution_admission_seal_sha256=SHA_E,
            source_teacher_resource_seal_sha256=SHA_A,
            source_teacher_lease_id=SHA_C,
        )


def test_preflight_has_no_source_root_or_process_launcher_surface() -> None:
    source = Path(outer.__file__).read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "subprocess." not in source
    destinations = {action.dest for action in outer._parser()._actions}
    assert "source_root" not in destinations
    assert "execute" not in destinations
