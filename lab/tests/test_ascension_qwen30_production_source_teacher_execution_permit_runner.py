"""Synthetic-only tests for the non-circular Q30 execution-permit interface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab.operators import (
    ascension_qwen30_production_source_teacher_execution_permit_runner as permit,
)
from lab.receipts import seal, verify

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
REVISION = "f" * 40


def _write_raw(path: Path, value: dict[str, object]) -> permit.Document:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return permit._read_document(path, label=path.name, sealed=False)


def _write_sealed(path: Path, value: dict[str, object]) -> permit.Document:
    path.write_text(json.dumps(seal(value), sort_keys=True), encoding="utf-8")
    return permit._read_document(path, label=path.name, sealed=True)


def _range_and_semantics(tmp_path: Path) -> tuple[permit.Document, permit.Document]:
    authority = {
        "schema": permit.RANGE_AUTHORITY_SCHEMA,
        "status": permit.RANGE_AUTHORITY_STATUS,
        "source": {
            "model_id": permit.MODEL_ID,
            "source_revision": REVISION,
            "source_shard_count": permit.SOURCE_SHARDS,
            "source_tensor_count": permit.SOURCE_TENSORS,
            "source_index": {"sha256": SHA_A, "format": "huggingface.safetensors.index.json"},
        },
        "metadata_access_boundary": {
            "gpu_or_metal_invoked": False,
            "hcli_invoked": False,
            "lease_requested": False,
            "mmap_or_memory_map_used": False,
            "server_started": False,
            "source_model_instantiated": False,
            "tensor_payload_hashes_collected": False,
            "whole_shard_payload_checksum_collected": False,
            "source_tensor_payload_bytes_read": 0,
        },
        "exact_streamed_oracle_scope": {
            "layers": permit.SOURCE_LAYERS,
            "total_forwards_per_replay_arm": permit.SOURCE_FORWARDS,
        },
    }
    authority_sha = permit._sha256(permit._canonical_json(authority))
    range_document = _write_raw(
        tmp_path / "range-authority.json",
        {"authority": authority, "authority_content_sha256": authority_sha},
    )
    semantics = _write_raw(
        tmp_path / "semantics.json",
        {
            "schema": permit.SEMANTICS_SCHEMA,
            "status": permit.SEMANTICS_STATUS,
            "execution_boundary": {
                "gpu_or_metal_invoked": False,
                "hcli_invoked": False,
                "lease_requested": False,
                "server_started": False,
                "source_inference_executed": False,
                "source_model_instantiated": False,
                "source_quality_or_coherence_claim_made": False,
                "source_safetensors_or_other_weight_path_accepted": False,
                "source_tensor_payload_opened": False,
                "tps_or_tg_claim_made": False,
            },
            "pinned_source_binding": {
                "source_model_id": permit.MODEL_ID,
                "source_revision": REVISION,
                "source_index_sha256": SHA_A,
            },
            "consumed_metadata_contracts": {
                "range_authority": {
                    "document_sha256": range_document.raw_document_sha256,
                    "authority_content_sha256": authority_sha,
                    "source_payload_read_by_this_attester": False,
                }
            },
            "future_exact_execution_attestation": {
                "schema": permit.OPERATOR_ATTESTATION_SCHEMA,
                "status_only_after_real_separately_leased_source_execution": permit.OPERATOR_ATTESTATION_STATUS,
            },
        },
    )
    return range_document, semantics


def _antecedents(tmp_path: Path, *, fixture_map: bool = False) -> dict[str, permit.Document]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    range_document, semantics = _range_and_semantics(tmp_path)
    flat_map = _write_sealed(
        tmp_path / "flat-map.json",
        {
            "schema": permit.FLAT_MAP_SCHEMA,
            "source_model_id": permit.MODEL_ID,
            "source_revision": REVISION,
            "source_tensor_count": permit.SOURCE_TENSORS,
            "maximum_window_bytes": permit.MAX_POSITIONED_READ_BYTES,
            "source_index": {"sha256": SHA_A},
            "shards": [{} for _ in range(permit.SOURCE_SHARDS)],
            "tensors": [{} for _ in range(permit.SOURCE_TENSORS)],
            "fixture_only": fixture_map,
            "synthetic_fixture_only": False,
        },
    )
    coverage = _write_sealed(
        tmp_path / "coverage.json",
        {
            "schema": permit.HASH_COVERAGE_SCHEMA,
            "status": permit.HASH_COVERAGE_STATUS,
            "fixture_only": False,
            "synthetic_fixture_only": False,
            "production_hash_coverage_earned": True,
            "source_teacher_execution_or_logits": False,
            "source_teacher_runtime_admission_earned": False,
            "operator_or_reader_execution_attestation_emitted": False,
            "flat_runtime_range_map": permit._evidence(flat_map),
            "bounded_positioned_reader": {
                "maximum_positioned_read_bytes": permit.MAX_POSITIONED_READ_BYTES,
                "maximum_live_raw_bf16_windows": 1,
                "one_shard_handle_at_a_time": True,
                "whole_shard_cache_or_mmap_forbidden": True,
                "cache_zeroed_after_every_visit_and_before_receipt": True,
            },
            "coverage": {
                "source_shards": permit.SOURCE_SHARDS,
                "source_tensors": permit.SOURCE_TENSORS,
                "source_index_sha256": SHA_A,
            },
        },
    )
    capture = _write_sealed(
        tmp_path / "production-capture.json",
        {
            "schema": permit.PRODUCTION_CAPTURE_SCHEMA,
            "status": permit.PRODUCTION_CAPTURE_STATUS,
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
            "flat_runtime_range_map": permit._evidence(flat_map),
            "hash_coverage_attestation": permit._evidence(coverage),
        },
    )
    terminal = _write_sealed(
        tmp_path / "terminal.json",
        {
            "schema": permit.PRODUCTION_OUTER_SCHEMA,
            "status": permit.PRODUCTION_OUTER_STATUS,
            "child_reaped": True,
            "terminal_receipt_written_after_child_capture": True,
            "terminal_receipt_written_last": True,
            "automatic_retry_disabled": True,
            "lease_reuse_prohibited": True,
            "child_timed_out": False,
            "child_exit_code": 0,
            "child_capture": permit._evidence(capture),
        },
    )
    release = _write_sealed(
        tmp_path / "release.json",
        {
            "schema": permit.PRODUCTION_RELEASE_SCHEMA,
            "status": permit.PRODUCTION_RELEASE_STATUS,
            "release_after_outer_terminal": True,
            "one_shot_lease_finalized": True,
            "retry_or_relaunch_forbidden": True,
            "source_teacher_or_logits_authorized": False,
            "native_or_gpu_server_hcli_authorized": False,
            "artifacts_deleted_or_evicted": False,
            "outer_terminal_seal_sha256": terminal.seal_sha256,
            "child_capture_seal_sha256": capture.seal_sha256,
        },
    )
    bridge = _write_sealed(
        tmp_path / "post-hash-bridge.json",
        {
            "schema": permit.POST_HASH_MAP_BRIDGE_SCHEMA,
            "status": permit.POST_HASH_MAP_BRIDGE_STATUS,
            "prepared": True,
            "execution_authorized": False,
            "runtime_admission_earned": False,
            "dual_attestation_runtime_admission_emitted": False,
            "post_hash_map_antecedents": {
                "production_outer_terminal": permit._evidence(terminal),
                "production_child_capture": permit._evidence(capture),
                "production_flat_map": permit._evidence(flat_map),
                "production_hash_coverage": permit._evidence(coverage),
                "production_lease_release": permit._evidence(release),
            },
            "upstream_authorities": {
                "metadata_range_authority": permit._evidence(range_document),
                "semantics_attester": permit._evidence(semantics),
            },
            "admission_before_open_cycle": {
                "runtime_admission_must_be_earned_before_source_root_open": True,
                "runtime_producer_requires_bounded_source_validation_and_both_execution_attestations": True,
                "existing_source_teacher_child_requires_runtime_admission_before_source_root_open": True,
                "bridge_does_not_relax_or_reorder_any_requirement": True,
                "resolved": False,
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
    return {
        "range_authority": range_document,
        "semantics_attester": semantics,
        "post_hash_map_bridge": bridge,
        "flat_map": flat_map,
        "hash_coverage": coverage,
        "production_capture": capture,
        "production_outer_terminal": terminal,
        "production_release": release,
    }


def _resource(tmp_path: Path, documents: dict[str, permit.Document]) -> permit.Document:
    return _write_sealed(
        tmp_path / "resource.json",
        {
            "schema": permit.RESOURCE_SCHEMA,
            "status": permit.RESOURCE_STATUS,
            "fixture_only": False,
            "source_root_opened_or_statted": False,
            "source_teacher_or_native_child_started": False,
            "execution_permit_lease_issued_or_consumed": False,
            "post_hash_map_bridge": permit._evidence(documents["post_hash_map_bridge"]),
            "production_flat_map": permit._evidence(documents["flat_map"]),
            "production_hash_coverage": permit._evidence(documents["hash_coverage"]),
            "metadata_range_authority": permit._evidence(documents["range_authority"]),
            "semantics_attester": permit._evidence(documents["semantics_attester"]),
            "fresh_pre_execution_safety": {
                "observed_immediately_before_execution_permit_lease": True,
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


def _kwargs(documents: dict[str, permit.Document], resource: permit.Document | None = None) -> dict[str, Path]:
    result = {
        "range_authority_path": documents["range_authority"].path,
        "semantics_attester_path": documents["semantics_attester"].path,
        "post_hash_map_bridge_path": documents["post_hash_map_bridge"].path,
        "flat_map_path": documents["flat_map"].path,
        "hash_coverage_path": documents["hash_coverage"].path,
        "production_capture_path": documents["production_capture"].path,
        "production_outer_terminal_path": documents["production_outer_terminal"].path,
        "production_lease_release_path": documents["production_release"].path,
    }
    if resource is not None:
        result["fresh_resource_admission_path"] = resource.path
    return result


def test_real_run_inputs_are_unavailable_without_a_fresh_resource_admission(tmp_path: Path) -> None:
    documents = _antecedents(tmp_path)

    result = permit.build_execution_permit(**_kwargs(documents))

    assert result["status"] == permit.REFUSED_STATUS
    assert result["prepared"] is False
    assert result["execution_permit_materialized"] is False
    assert "fresh_execution_permit_zero_swap_resource_admission_absent" in result["blockers"]
    assert all(value is False for value in result["execution_boundary"].values())
    verify(result, label="missing resource refusal")


def test_valid_nonfixture_preconditions_materialize_only_a_preexecution_permit(tmp_path: Path) -> None:
    documents = _antecedents(tmp_path)
    resource = _resource(tmp_path, documents)

    result = permit.build_execution_permit(**_kwargs(documents, resource))

    assert result["schema"] == permit.SCHEMA
    assert result["status"] == permit.PREPARED_STATUS
    assert result["execution_permit_materialized"] is True
    assert result["execution_permit_consumed"] is False
    assert result["spawn_permitted"] is False
    assert result["future_runner_interface"]["bounded_source_run"]["source_layer_traversals"] == 48 * 370
    assert result["future_runner_interface"]["bounded_source_run"]["old_final_runtime_admission_is_not_a_pre_execution_input"] is True
    assert all(value is False for value in result["execution_boundary"].values())
    verify(result, label="prepared execution permit")


def test_fixture_map_or_legacy_runtime_relabel_cannot_be_used_as_permit_resource(tmp_path: Path) -> None:
    fixture_documents = _antecedents(tmp_path / "fixture", fixture_map=True)
    fixture_result = permit.build_execution_permit(**_kwargs(fixture_documents))
    assert fixture_result["status"] == permit.REFUSED_STATUS
    assert any(
        str(blocker).startswith("exact_non_fixture_post_hash_map_antecedents_invalid:")
        for blocker in fixture_result["blockers"]
    )

    documents = _antecedents(tmp_path / "relabel")
    legacy_runtime = _write_sealed(
        tmp_path / "relabel" / "legacy-runtime.json",
        {
            "schema": permit.FINAL_RUNTIME_ADMISSION_SCHEMA,
            "status": permit.FINAL_RUNTIME_ADMISSION_STATUS,
            "fixture_only": False,
        },
    )
    relabel_result = permit.build_execution_permit(**_kwargs(documents, legacy_runtime))
    assert relabel_result["status"] == permit.REFUSED_STATUS
    assert any(
        str(blocker).startswith("fresh_execution_permit_zero_swap_resource_admission_invalid:")
        for blocker in relabel_result["blockers"]
    )


def test_fake_48x370_result_can_only_earn_attestations_and_runtime_after_capture() -> None:
    capture = seal(
        {
            "schema": permit.EXECUTION_CAPTURE_SCHEMA,
            "status": permit.EXECUTION_CAPTURE_STATUS,
            "source_teacher_execution_completed": True,
            "two_source_f32le_vectors_fsynced_before_child_exit": True,
            "source_handles_closed_before_child_exit": True,
            "reader_cache_zeroed_before_child_exit": True,
            "receipt_written_last": True,
            "legacy_runtime_admission_used_before_run": False,
            "legacy_dual_bridge_used_before_run": False,
            "native_phase_started": False,
            "gpu_server_hcli_or_tps_action": False,
            "geometry": {
                "source_layers": permit.SOURCE_LAYERS,
                "source_forwards": permit.SOURCE_FORWARDS,
                "source_vectors_f32le": permit.SOURCE_VECTORS,
                "maximum_positioned_read_bytes": permit.MAX_POSITIONED_READ_BYTES,
                "maximum_live_raw_bf16_windows": 1,
            },
            "execution_permit_seal_sha256": SHA_A,
            "execution_resource_seal_sha256": SHA_B,
            "execution_lease_id": SHA_C,
        }
    )
    operator = seal(
        {
            "schema": permit.OPERATOR_ATTESTATION_SCHEMA,
            "status": permit.OPERATOR_ATTESTATION_STATUS,
            "earned_after_real_execution_capture": True,
            "execution_capture_seal_sha256": capture["seal_sha256"],
            "execution_permit_seal_sha256": SHA_A,
        }
    )
    reader = seal(
        {
            "schema": permit.READER_ATTESTATION_SCHEMA,
            "status": permit.READER_ATTESTATION_STATUS,
            "earned_after_real_execution_capture": True,
            "execution_capture_seal_sha256": capture["seal_sha256"],
            "execution_permit_seal_sha256": SHA_A,
        }
    )
    runtime = {
        "schema": permit.FINAL_RUNTIME_ADMISSION_SCHEMA,
        "status": permit.FINAL_RUNTIME_ADMISSION_STATUS,
        "earned_after_execution_capture": True,
        "may_not_authorize_its_own_prior_execution": True,
        "execution_capture_seal_sha256": capture["seal_sha256"],
        "execution_permit_seal_sha256": SHA_A,
        "operator_execution_attestation_seal_sha256": operator["seal_sha256"],
        "reader_execution_attestation_seal_sha256": reader["seal_sha256"],
        "flat_runtime_range_map": {"schema": permit.FLAT_MAP_SCHEMA, "document_sha256": SHA_D},
    }

    permit.validate_fake_post_execution_finalization(
        execution_capture=capture,
        operator_attestation=operator,
        reader_attestation=reader,
        final_runtime_admission=runtime,
        execution_permit_seal_sha256=SHA_A,
        execution_resource_seal_sha256=SHA_B,
        execution_lease_id=SHA_C,
        flat_map_canonical_document_sha256=SHA_D,
    )

    runtime["earned_after_execution_capture"] = False
    with pytest.raises(permit.ExecutionPermitError, match="earned-after-capture"):
        permit.validate_fake_post_execution_finalization(
            execution_capture=capture,
            operator_attestation=operator,
            reader_attestation=reader,
            final_runtime_admission=runtime,
            execution_permit_seal_sha256=SHA_A,
            execution_resource_seal_sha256=SHA_B,
            execution_lease_id=SHA_C,
            flat_map_canonical_document_sha256=SHA_D,
        )


def test_interface_has_no_source_root_or_process_launcher_cli_surface() -> None:
    source = Path(permit.__file__).read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "subprocess." not in source
    destinations = {action.dest for action in permit._parser()._actions}
    assert "source_root" not in destinations
    assert "mode" not in destinations
