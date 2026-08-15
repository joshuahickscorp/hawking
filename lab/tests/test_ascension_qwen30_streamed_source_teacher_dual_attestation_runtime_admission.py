"""Focused CPU/file-only tests for the Q30 dual-attestation runtime-admission bridge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from lab.operators import (
    ascension_qwen30_streamed_source_teacher_dual_attestation_runtime_admission as bridge,
)
from lab.receipts import seal, verify

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
TOKEN_SHA = "2" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, document: dict[str, Any], *, sealed: bool = False) -> Path:
    value = seal(document) if sealed else document
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _range_authority(path: Path) -> Path:
    authority = {
        "schema": bridge.RANGE_AUTHORITY_SCHEMA,
        "status": bridge.RANGE_AUTHORITY_STATUS,
        "source": {
            "model_id": "Qwen3-Coder-30B-A3B-Instruct",
            "source_revision": "b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
            "source_tensor_count": 18_867,
            "source_shard_count": 16,
            "source_index": {"sha256": SHA_C, "weight_map_tensor_count": 18_867},
        },
        "exact_streamed_oracle_scope": {
            "source_template_token_count": bridge.PREFIX_TOKENS,
            "forced_identical_continuation_token_id": bridge.FORCED_TOKEN_ID,
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
    }
    return _write(
        path,
        {
            "authority_content_sha256": SHA_D,
            "authority": authority,
        },
    )


def _semantics(path: Path) -> Path:
    return _write(
        path,
        {
            "schema": bridge.SEMANTICS_SCHEMA,
            "status": bridge.SEMANTICS_STATUS,
            "execution_boundary": {
                "source_tensor_payload_opened": False,
                "source_model_instantiated": False,
                "source_inference_executed": False,
                "gpu_or_metal_invoked": False,
                "server_started": False,
                "hcli_invoked": False,
                "lease_requested": False,
            },
        },
    )


def _feasibility(path: Path, *, prepared: bool = False) -> Path:
    return _write(
        path,
        {
            "schema": bridge.FEASIBILITY_SCHEMA,
            "status": (
                bridge.FEASIBILITY_PREPARED_STATUS
                if prepared
                else bridge.FEASIBILITY_REFUSED_STATUS
            ),
            "exact_trace": {
                "prefix_token_count": bridge.PREFIX_TOKENS,
                "forced_token_id": bridge.FORCED_TOKEN_ID,
                "source_template_token_ids_u32le_sha256": TOKEN_SHA,
            },
            "memory_assessment": {
                "streamed_memory_arithmetic_fits": prepared,
                "zero_swap_condition_met": prepared,
            },
            "feasibility": {
                "oracle_execution_authorized": False,
                "semantic_equivalence_proven_by_external_sealed_attestation": prepared,
            },
        },
        sealed=True,
    )


def _raw_six_vector(path: Path) -> Path:
    return _write(
        path,
        {
            "schema": bridge.RAW_SIX_VECTOR_SCHEMA,
            "status": bridge.RAW_SIX_VECTOR_STATUS,
            "six_vector_retention_contract": {
                "dtype": "f32le",
                "vocab_rows": bridge.VOCAB_ROWS,
                "bytes_per_vector": bridge.F32_VECTOR_BYTES,
                "required_payload_count": 6,
                "required_total_payload_bytes": bridge.F32_VECTOR_BYTES * 6,
            },
            "source_memory_and_eviction_gate": {
                "source_teacher_capture_is_currently_blocked": True,
            },
        },
        sealed=True,
    )


def _current_trace(path: Path) -> Path:
    return _write(
        path,
        {
            "schema": bridge.CURRENT_TRACE_SCHEMA,
            "status": bridge.CURRENT_TRACE_STATUS,
            "binding": {
                "source_template_token_count": bridge.PREFIX_TOKENS,
                "forced_identical_continuation_token_id": bridge.FORCED_TOKEN_ID,
                "source_template_token_ids_u32le_sha256": TOKEN_SHA,
            },
        },
        sealed=True,
    )


def _fixtures(tmp_path: Path) -> dict[str, Path]:
    return {
        "range_authority_path": _range_authority(tmp_path / "range.json"),
        "semantics_path": _semantics(tmp_path / "semantics.json"),
        "feasibility_path": _feasibility(tmp_path / "feasibility.json"),
        "raw_six_vector_path": _raw_six_vector(tmp_path / "raw.json"),
        "current_trace_path": _current_trace(tmp_path / "current.json"),
    }


def test_valid_upstream_bundle_prepares_non_authorizing_bridge(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    result = bridge.build_dual_attestation_runtime_admission(**paths)
    assert result["schema"] == bridge.SCHEMA
    assert result["status"] == bridge.STATUS
    assert result["execution_authorized"] is False
    assert result["schema_resolution"]["bridge_does_not_authorize_execution"] is True
    assert (
        result["schema_resolution"][
            "a_prepared_bridge_is_non_authorizing_and_cannot_substitute_for_either_execution_attestation"
        ]
        is True
    )
    operator = result["schema_resolution"]["operator_accumulation_execution_attestation"]
    reader = result["schema_resolution"]["range_reader_exact_semantics_attestation"]
    assert operator["schema"] == bridge.OPERATOR_ATTESTATION_SCHEMA
    assert operator["status"] == bridge.OPERATOR_ATTESTATION_STATUS
    assert operator["earned_by_this_bridge"] is False
    assert reader["schema"] == bridge.RANGE_READER_ATTESTATION_SCHEMA
    assert reader["status"] == bridge.RANGE_READER_ATTESTATION_STATUS
    assert reader["earned_by_this_bridge"] is False
    assert result["execution_boundary"]["operator_or_reader_execution_attestation_earned"] is False
    assert result["future_source_worker"]["source_layers"] == bridge.SOURCE_LAYERS
    assert result["future_source_worker"]["source_forwards"] == bridge.SOURCE_FORWARDS
    upstream = result["upstream_metadata"]
    assert upstream["range_authority"]["authority_content_sha256"] == SHA_D
    assert upstream["streamed_feasibility"]["raw_document_sha256"] == _sha256(
        paths["feasibility_path"]
    )
    assert upstream["raw_six_vector_contract"]["seal_sha256"]
    verify(result, label="dual attestation bridge")


def test_bridge_binds_refused_feasibility_without_authorizing(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    result = bridge.build_dual_attestation_runtime_admission(**paths)
    feasibility = json.loads(paths["feasibility_path"].read_text(encoding="utf-8"))
    assert feasibility["status"] == bridge.FEASIBILITY_REFUSED_STATUS
    assert result["execution_authorized"] is False
    assert result["upstream_metadata"]["streamed_feasibility"]["seal_sha256"] == feasibility[
        "seal_sha256"
    ]


def test_tampered_feasibility_is_refused(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    bad = json.loads(paths["feasibility_path"].read_text(encoding="utf-8"))
    bad["seal_sha256"] = "0" * 64
    paths["feasibility_path"].write_text(json.dumps(bad, sort_keys=True), encoding="utf-8")
    with pytest.raises(bridge.DualAttestationBridgeError, match="unsealed or tampered"):
        bridge.build_dual_attestation_runtime_admission(**paths)


def test_unsealed_raw_six_vector_is_refused(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    paths["raw_six_vector_path"] = _write(
        tmp_path / "raw-unsealed.json",
        {
            "schema": bridge.RAW_SIX_VECTOR_SCHEMA,
            "status": bridge.RAW_SIX_VECTOR_STATUS,
            "six_vector_retention_contract": {
                "dtype": "f32le",
                "vocab_rows": bridge.VOCAB_ROWS,
                "bytes_per_vector": bridge.F32_VECTOR_BYTES,
                "required_payload_count": 6,
            },
        },
        sealed=False,
    )
    with pytest.raises(bridge.DualAttestationBridgeError, match="unsealed or tampered"):
        bridge.build_dual_attestation_runtime_admission(**paths)


def test_write_new_refuses_overwrite(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    result = bridge.build_dual_attestation_runtime_admission(**paths)
    out = tmp_path / "dual-bridge.json"
    bridge._write_new(out, result)
    with pytest.raises(bridge.DualAttestationBridgeError, match="new immutable"):
        bridge._write_new(out, result)


def test_cli_main_writes_sealed_bridge(tmp_path: Path) -> None:
    paths = _fixtures(tmp_path)
    out = tmp_path / "dual-bridge.json"
    code = bridge.main(
        [
            "--range-authority",
            str(paths["range_authority_path"]),
            "--semantics",
            str(paths["semantics_path"]),
            "--feasibility",
            str(paths["feasibility_path"]),
            "--raw-six-vector",
            str(paths["raw_six_vector_path"]),
            "--current-trace",
            str(paths["current_trace_path"]),
            "--out",
            str(out),
        ]
    )
    assert code == 0
    document = verify(json.loads(out.read_text(encoding="utf-8")), label="cli dual bridge")
    assert document["status"] == bridge.STATUS
    assert document["execution_authorized"] is False
