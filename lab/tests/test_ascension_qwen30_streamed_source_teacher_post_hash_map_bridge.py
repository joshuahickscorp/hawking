"""Focused CPU/file-only tests for Q30's post-hash-map teacher reservation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from lab.operators import (
    ascension_qwen30_streamed_source_teacher_post_hash_map_bridge as bridge,
)
from lab.receipts import seal, verify


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write(path: Path, value: dict[str, Any], *, sealed: bool) -> tuple[Path, dict[str, Any]]:
    document = seal(value) if sealed else value
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path, document


def _replace_sealed(path: Path, mutate: callable) -> None:
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(json.dumps(seal(document), sort_keys=True), encoding="utf-8")


def _pointer(path: Path, document: dict[str, Any], *, canonical: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "raw_document_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "seal_sha256": document.get("seal_sha256"),
    }
    if canonical:
        result["canonical_document_sha256"] = hashlib.sha256(_canonical(document)).hexdigest()
    return result


def _unsealed_range() -> dict[str, Any]:
    authority = {
        "schema": bridge.RANGE_AUTHORITY_SCHEMA,
        "status": bridge.RANGE_AUTHORITY_STATUS,
        "source": {
            "model_id": bridge.MODEL_ID,
            "source_revision": "a" * 40,
            "source_shard_count": bridge.SOURCE_SHARDS,
            "source_tensor_count": bridge.SOURCE_TENSORS,
            "source_index": {
                "format": "huggingface.safetensors.index.json",
                "relative_path": "model.safetensors.index.json",
                "sha256": _hash("source-index"),
            },
        },
        "metadata_access_boundary": {
            "gpu_or_metal_invoked": False,
            "hcli_invoked": False,
            "lease_requested": False,
            "mmap_or_memory_map_used": False,
            "server_started": False,
            "source_model_instantiated": False,
            "source_tensor_payload_bytes_read": 0,
            "tensor_payload_hashes_collected": False,
            "whole_shard_payload_checksum_collected": False,
        },
        "exact_streamed_oracle_scope": {
            "layers": bridge.SOURCE_LAYERS,
            "total_forwards_per_replay_arm": bridge.SOURCE_FORWARDS,
        },
    }
    return {"authority": authority, "authority_content_sha256": hashlib.sha256(_canonical(authority)).hexdigest()}


def _semantics(range_path: Path, range_document: dict[str, Any]) -> dict[str, Any]:
    authority = range_document["authority"]
    source = authority["source"]
    return {
        "schema": bridge.SEMANTICS_SCHEMA,
        "status": bridge.SEMANTICS_STATUS,
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
            "source_model_id": source["model_id"],
            "source_revision": source["source_revision"],
            "source_index_sha256": source["source_index"]["sha256"],
        },
        "consumed_metadata_contracts": {
            "range_authority": {
                "document_sha256": hashlib.sha256(range_path.read_bytes()).hexdigest(),
                "authority_content_sha256": range_document["authority_content_sha256"],
                "source_payload_read_by_this_attester": False,
            }
        },
        "future_exact_execution_attestation": {
            "schema": bridge.OPERATOR_ATTESTATION_SCHEMA,
            "status_only_after_real_separately_leased_source_execution": bridge.OPERATOR_ATTESTATION_STATUS,
        },
    }


def _runtime_producer(
    range_path: Path,
    range_document: dict[str, Any],
    semantics_path: Path,
) -> dict[str, Any]:
    authority = range_document["authority"]
    source = authority["source"]
    return {
        "schema": bridge.RUNTIME_PRODUCER_SCHEMA,
        "status": bridge.RUNTIME_PRODUCER_STATUS,
        "prepared": True,
        "runtime_admission_earned": False,
        "source_payload_validation_executed": False,
        "sealed_metadata_authority_binding": {
            "metadata_range_authority": {
                "raw_document_sha256": hashlib.sha256(range_path.read_bytes()).hexdigest()
            },
            "authority_content_sha256": range_document["authority_content_sha256"],
            "source_revision": source["source_revision"],
            "source_shard_count": bridge.SOURCE_SHARDS,
            "source_tensor_count": bridge.SOURCE_TENSORS,
            "maximum_declared_bf16_row_window_bytes": bridge.MAX_WINDOW_BYTES,
        },
        "metadata_semantics_binding": {
            "operator_semantics_attester": {
                "raw_document_sha256": hashlib.sha256(semantics_path.read_bytes()).hexdigest()
            },
            "both_execution_attestations_required": True,
            "future_operator_accumulation_execution_attestation": {
                "schema": bridge.OPERATOR_ATTESTATION_SCHEMA,
                "status_only_after_real_separately_leased_source_execution": bridge.OPERATOR_ATTESTATION_STATUS,
            },
            "future_range_reader_exact_semantics_attestation": {
                "schema": bridge.READER_ATTESTATION_SCHEMA,
                "status_only_after_real_separately_leased_source_execution": bridge.READER_ATTESTATION_STATUS,
            },
        },
        "future_flat_runtime_range_map": {
            "schema": bridge.FLAT_MAP_SCHEMA,
            "maximum_window_bytes": bridge.MAX_WINDOW_BYTES,
            "maximum_positioned_read_bytes": bridge.MAX_WINDOW_BYTES,
        },
        "future_runtime_admission_receipt": {
            "schema": bridge.RUNTIME_ADMISSION_SCHEMA,
            "status_only_after_bounded_source_validation": bridge.RUNTIME_ADMISSION_STATUS,
        },
        "execution_boundary": {
            "child_process_started": False,
            "future_source_root_opened_or_statted": False,
            "gpu_metal_mps_or_other_accelerator_invoked": False,
            "hcli_invoked": False,
            "lease_requested_issued_or_consumed": False,
            "server_started_or_contacted": False,
            "source_model_loaded_or_instantiated": False,
            "source_tensor_payload_opened": False,
            "tps_or_tg_measured": False,
            "whole_shard_mapped_or_cached": False,
            "whole_source_model_resident": False,
        },
    }


def _flat_map(range_document: dict[str, Any]) -> dict[str, Any]:
    source = range_document["authority"]["source"]
    shards = [
        {
            "shard_id": f"source-shard-{index:02d}",
            "relative_path": f"model-{index + 1:05d}-of-00016.safetensors",
            "bytes": 10_000_000,
            "sha256": _hash(f"shard-{index}"),
            "safetensors_header_sha256": _hash(f"header-{index}"),
            "safetensors_prefix_sha256": _hash(f"prefix-{index}"),
        }
        for index in range(bridge.SOURCE_SHARDS)
    ]
    tensors = [
        {
            "tensor_name": f"tensor.{index:05d}",
            "shard_id": shards[index % bridge.SOURCE_SHARDS]["shard_id"],
            "dtype": "BF16",
            "shape": [1],
            "data_offset": 4096 + (index // bridge.SOURCE_SHARDS) * 2,
            "data_bytes": 2,
            "raw_bf16_sha256": _hash(f"bf16-{index}"),
        }
        for index in range(bridge.SOURCE_TENSORS)
    ]
    return {
        "schema": bridge.FLAT_MAP_SCHEMA,
        "source_model_id": source["model_id"],
        "source_revision": source["source_revision"],
        "source_tensor_count": bridge.SOURCE_TENSORS,
        "source_index": source["source_index"],
        "maximum_window_bytes": bridge.MAX_WINDOW_BYTES,
        "fixture_only": False,
        "synthetic_fixture_only": False,
        "production_adapter_forbidden": False,
        "shards": shards,
        "tensors": tensors,
    }


def _bundle(tmp_path: Path) -> dict[str, Path]:
    range_path, range_document = _write(tmp_path / "range-authority.json", _unsealed_range(), sealed=False)
    semantics_path, _semantics_document = _write(
        tmp_path / "semantics.json", _semantics(range_path, range_document), sealed=False
    )
    runtime_path, _runtime_document = _write(
        tmp_path / "runtime-producer.json",
        _runtime_producer(range_path, range_document, semantics_path),
        sealed=True,
    )
    map_path, map_document = _write(
        tmp_path / "production-flat-map.json", _flat_map(range_document), sealed=True
    )
    authority_path, authority_document = _write(
        tmp_path / "production-authority.json", {"schema": "authority.v1", "status": "ADMITTED"}, sealed=True
    )
    lease_path, lease_document = _write(
        tmp_path / "issued-lease.json",
        {
            "schema": "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_quiet_lease.v1",
            "status": "GRANTED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_HASH_SCAN_ONE_SHOT",
            "lease_id": _hash("one production lease"),
        },
        sealed=True,
    )
    replay_path, replay_document = _write(
        tmp_path / "scan-replay-reservation.json",
        {
            "schema": bridge.REPLAY_SCHEMA,
            "status": bridge.REPLAY_STATUS,
            "attempt": 1,
            "create_new_before_source_root_open": True,
            "one_child_maximum": True,
            "replay_or_relaunch_forbidden": True,
        },
        sealed=True,
    )
    coverage_path, coverage_document = _write(
        tmp_path / "hash-coverage.json",
        {
            "schema": bridge.HASH_COVERAGE_SCHEMA,
            "status": bridge.HASH_COVERAGE_STATUS,
            "fixture_only": False,
            "synthetic_fixture_only": False,
            "production_adapter_forbidden": False,
            "production_hash_coverage_earned": True,
            "operator_or_reader_execution_attestation_emitted": False,
            "source_teacher_execution_or_logits": False,
            "source_teacher_runtime_admission_earned": False,
            "flat_runtime_range_map": _pointer(map_path, map_document),
            "metadata_range_authority": _pointer(range_path, range_document),
            "production_scan_authority": _pointer(authority_path, authority_document),
            "fresh_bootstrap_lease": _pointer(lease_path, lease_document),
            "replay_reservation": _pointer(replay_path, replay_document),
            "coverage": {
                "source_shards": bridge.SOURCE_SHARDS,
                "source_tensors": bridge.SOURCE_TENSORS,
                "full_shard_sha256_count": bridge.SOURCE_SHARDS,
                "raw_bf16_range_sha256_count": bridge.SOURCE_TENSORS,
                "source_index_sha256": range_document["authority"]["source"]["source_index"]["sha256"],
            },
            "bounded_positioned_reader": {
                "maximum_positioned_read_bytes": bridge.MAX_WINDOW_BYTES,
                "maximum_live_raw_bf16_windows": 1,
                "cache_zeroed_after_every_visit_and_before_receipt": True,
                "one_shard_handle_at_a_time": True,
                "whole_shard_cache_or_mmap_forbidden": True,
                "positioned_read_calls": bridge.SOURCE_TENSORS + bridge.SOURCE_SHARDS,
                "positioned_read_bytes": bridge.SOURCE_TENSORS * 2,
            },
        },
        sealed=True,
    )
    capture_path, capture_document = _write(
        tmp_path / "production-child-capture.json",
        {
            "schema": bridge.CAPTURE_SCHEMA,
            "status": bridge.CAPTURE_STATUS,
            "fixture_only": False,
            "synthetic_fixture_only": False,
            "production_adapter_forbidden": False,
            "production_hash_scan_earned": True,
            "receipt_written_last": True,
            "source_handles_closed": True,
            "reader_cache_zeroed": True,
            "source_teacher_or_logits_executed": False,
            "operator_or_reader_execution_attestation_emitted": False,
            "source_teacher_runtime_admission_earned": False,
            "model_gpu_server_hcli_or_tps_action": False,
            "geometry": {
                "source_shards": bridge.SOURCE_SHARDS,
                "source_tensors": bridge.SOURCE_TENSORS,
                "maximum_positioned_read_bytes": bridge.MAX_WINDOW_BYTES,
                "maximum_live_raw_bf16_windows": 1,
            },
            "flat_runtime_range_map": _pointer(map_path, map_document),
            "hash_coverage_attestation": _pointer(coverage_path, coverage_document),
            "replay_reservation": _pointer(replay_path, replay_document),
            "metadata_range_authority": _pointer(range_path, range_document),
            "independent_non_fixture_semantics_attester": _pointer(semantics_path, _semantics_document),
            "runtime_admission_producer_authority": _pointer(runtime_path, _runtime_document),
            "production_scan_authority": _pointer(authority_path, authority_document),
            "fresh_bootstrap_lease": _pointer(lease_path, lease_document),
        },
        sealed=True,
    )
    terminal_path, terminal_document = _write(
        tmp_path / "outer-terminal.json",
        {
            "schema": bridge.OUTER_TERMINAL_SCHEMA,
            "status": bridge.OUTER_TERMINAL_STATUS,
            "child_reaped": True,
            "terminal_receipt_written_after_child_capture": True,
            "terminal_receipt_written_last": True,
            "automatic_retry_disabled": True,
            "lease_reuse_prohibited": True,
            "child_timed_out": False,
            "child_exit_code": 0,
            "child_signal": None,
            "child_spawn_error": None,
            "child_capture_validation_error": None,
            "child_capture": _pointer(capture_path, capture_document, canonical=False),
            "child_capture_seal_sha256": capture_document["seal_sha256"],
            "issued_lease": _pointer(lease_path, lease_document, canonical=False),
            "production_authority": _pointer(authority_path, authority_document, canonical=False),
        },
        sealed=True,
    )
    release_path, _release_document = _write(
        tmp_path / "lease-release.json",
        {
            "schema": bridge.RELEASE_SCHEMA,
            "status": bridge.RELEASE_STATUS,
            "release_after_outer_terminal": True,
            "one_shot_lease_finalized": True,
            "retry_or_relaunch_forbidden": True,
            "source_teacher_or_logits_authorized": False,
            "native_or_gpu_server_hcli_authorized": False,
            "artifacts_deleted_or_evicted": False,
            "lease_id": lease_document["lease_id"],
            "outer_terminal_seal_sha256": terminal_document["seal_sha256"],
            "child_capture_seal_sha256": capture_document["seal_sha256"],
            "outer_terminal_status": bridge.OUTER_TERMINAL_STATUS,
        },
        sealed=True,
    )
    return {
        "range": range_path,
        "semantics": semantics_path,
        "runtime": runtime_path,
        "map": map_path,
        "coverage": coverage_path,
        "capture": capture_path,
        "terminal": terminal_path,
        "release": release_path,
    }


def _build(paths: dict[str, Path]) -> dict[str, Any]:
    return bridge.build_post_hash_map_bridge(
        range_authority_path=paths["range"],
        semantics_attester_path=paths["semantics"],
        runtime_producer_path=paths["runtime"],
        flat_map_path=paths["map"],
        hash_coverage_path=paths["coverage"],
        production_capture_path=paths["capture"],
        outer_terminal_path=paths["terminal"],
        lease_release_path=paths["release"],
    )


def test_valid_completed_production_scan_yields_only_not_executed_teacher_reservation(tmp_path: Path) -> None:
    result = _build(_bundle(tmp_path))

    assert result["schema"] == bridge.SCHEMA
    assert result["status"] == bridge.STATUS
    assert result["execution_authorized"] is False
    assert result["runtime_admission_earned"] is False
    assert result["dual_attestation_runtime_admission_emitted"] is False
    assert set(result["post_hash_map_antecedents"]) == {
        "production_outer_terminal",
        "production_child_capture",
        "production_flat_map",
        "production_hash_coverage",
        "production_lease_release",
    }
    for pointer in result["post_hash_map_antecedents"].values():
        assert set(pointer) == {
            "path",
            "raw_document_sha256",
            "canonical_document_sha256",
            "seal_sha256",
        }
        assert pointer["canonical_document_sha256"]
    reservation = result["future_source_teacher_provenance_reservation"]
    assert reservation["reservation_status"] == "NOT_EXECUTED"
    assert reservation["runtime_admission"]["schema"] == bridge.RUNTIME_ADMISSION_SCHEMA
    assert reservation["dual_attestation_runtime_admission"]["schema"] == bridge.DUAL_BRIDGE_SCHEMA
    assert result["admission_before_open_cycle"]["resolved"] is False
    assert "admission_before_open_cycle_unresolved" in result["current_blockers"]
    verify(result, label="post-hash-map bridge")


def test_fixture_alias_or_true_flag_refuses_before_following_stale_capture_edges(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    _replace_sealed(
        paths["map"],
        lambda value: value.update({"status": "SYNTHETIC_FIXTURE_RESEALED"}),
    )

    with pytest.raises(bridge.PostHashMapBridgeError, match="fixture-only identity"):
        _build(paths)


def test_missing_canonical_pointer_in_coverage_refuses(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    _replace_sealed(
        paths["coverage"],
        lambda value: value["flat_runtime_range_map"].pop("canonical_document_sha256"),
    )

    with pytest.raises(bridge.PostHashMapBridgeError, match="canonical_document_sha256 is required"):
        _build(paths)


def test_failed_outer_terminal_refuses_even_with_a_sealed_capture(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    _replace_sealed(paths["terminal"], lambda value: value.update({"child_exit_code": 1}))

    with pytest.raises(bridge.PostHashMapBridgeError, match="failed or stale"):
        _build(paths)


def test_replayed_scan_reservation_refuses(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    coverage = json.loads(paths["coverage"].read_text(encoding="utf-8"))
    replay_path = Path(coverage["replay_reservation"]["path"])
    _replace_sealed(replay_path, lambda value: value.update({"attempt": 2}))
    replay_document = json.loads(replay_path.read_text(encoding="utf-8"))
    _replace_sealed(
        paths["coverage"],
        lambda value: value.update({"replay_reservation": _pointer(replay_path, replay_document)}),
    )

    with pytest.raises(bridge.PostHashMapBridgeError, match="stale or replayed"):
        _build(paths)


def test_unclosed_capture_handles_or_cache_refuses(tmp_path: Path) -> None:
    paths = _bundle(tmp_path)
    _replace_sealed(paths["capture"], lambda value: value.update({"reader_cache_zeroed": False}))

    with pytest.raises(bridge.PostHashMapBridgeError, match="reader_cache_zeroed"):
        _build(paths)
