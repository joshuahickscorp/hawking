"""CPU/file-only tests for the Q30 production scan authority materializer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_production_scan_authority_materializer as materializer,
)
from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_production_scan_outer_reaper as outer,
)
from lab.receipts import seal, verify


def _canonical_sha(document: object) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _raw_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_raw(path: Path, document: dict[str, object]) -> tuple[Path, dict[str, object]]:
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path, document


def _write_sealed(path: Path, document: dict[str, object]) -> tuple[Path, dict[str, object]]:
    checked = seal(document)
    path.write_text(json.dumps(checked, sort_keys=True), encoding="utf-8")
    return path, checked


def _sealed_pointer(path: Path, document: dict[str, object]) -> dict[str, str]:
    return {
        "raw_document_sha256": _raw_sha(path),
        "seal_sha256": str(document["seal_sha256"]),
    }


def _full_evidence(
    path: Path, document: dict[str, object], *, sealed: bool
) -> dict[str, object]:
    return {
        "path": str(path),
        "raw_document_sha256": _raw_sha(path),
        "canonical_document_sha256": _canonical_sha(document),
        "seal_sha256": str(document["seal_sha256"]) if sealed else None,
    }


def _bootstrap_preflight() -> dict[str, object]:
    return {
        "schema": outer.bootstrap.PREFLIGHT_SCHEMA,
        "status": outer.bootstrap.PREFLIGHT_STATUS,
        "prepared": True,
        "execution_authorized": False,
        "metadata_bindings": {
            "range_authority_document_sha256": "a" * 64,
            "range_authority_content_sha256": "b" * 64,
            "source_index_sha256": "c" * 64,
            "maximum_declared_bf16_window_bytes": 524_288,
        },
        "future_bootstrap_lease": {
            "schema": outer.LEASE_SCHEMA,
            "status": outer.LEASE_STATUS,
            "one_shot": True,
            "separate_from_source_teacher_lease": True,
            "non_inference_only": True,
            "model_server_gpu_hcli_or_tps_allowed": False,
            "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
            "maximum_live_raw_bf16_windows": 1,
        },
        "future_outputs_required_before_source_teacher": {
            "flat_runtime_range_map": {
                "schema": "hawking.ascension.qwen30_source_bf16_range_map.v1",
                "shards": outer.SOURCE_SHARDS,
                "tensors": outer.SOURCE_TENSORS,
            },
            "runtime_admission": {
                "schema": "hawking.ascension.qwen30_streamed_source_teacher_runtime_range_admission.v1",
                "status": "EARNED_QWEN30_STREAMED_SOURCE_TEACHER_RUNTIME_RANGE_ADMISSION_NO_MODEL_RESIDENCY",
            },
        },
        "execution_boundary": {
            "source_root_opened_or_statted": False,
            "source_tensor_payload_opened": False,
            "flat_runtime_range_map_emitted": False,
            "two_attestations_emitted": False,
            "runtime_admission_earned": False,
            "source_teacher_started": False,
            "model_gpu_server_hcli_or_tps_action": False,
            "lease_issued_or_consumed": False,
        },
    }


def _chain(tmp_path: Path) -> dict[str, tuple[Path, dict[str, object]]]:
    preflight_path, preflight = _write_sealed(
        tmp_path / "bootstrap-preflight.json", _bootstrap_preflight()
    )
    bootstrap_binary_path, bootstrap_binary = _write_sealed(
        tmp_path / "bootstrap-binary.json",
        {
            "schema": outer.bootstrap.BINARY_SCHEMA,
            "status": outer.bootstrap.BINARY_STATUS,
            "cpu_only": True,
            "scan_or_runtime_executed": False,
            "binary_sha256": "d" * 64,
            "source_sha256": "e" * 64,
            "bootstrap_preflight": _sealed_pointer(preflight_path, preflight),
        },
    )
    bootstrap_resource_path, bootstrap_resource = _write_sealed(
        tmp_path / "bootstrap-resource.json",
        {
            "schema": outer.bootstrap.RESOURCE_SCHEMA,
            "status": outer.bootstrap.RESOURCE_STATUS,
            "fresh_observation": True,
            "exclusive_clean_window": True,
            "zero_swap": True,
            "zero_swapouts": True,
            "resource_admitted_for_one_future_child": True,
            "source_payload_opened": False,
            "source_model_loaded": False,
            "gpu_server_hcli_or_tps_action": False,
            "lease_issued_or_consumed": False,
            "child_started": False,
            "swap_used_bytes": 0,
            "swapouts_pages_delta": 0,
            "reclaimable_bytes": 10_000,
            "minimum_reclaimable_bytes_required": 1,
            "bootstrap_binary_sha256": "d" * 64,
            "bootstrap_preflight": _sealed_pointer(preflight_path, preflight),
            "resource_window_identity_sha256": "c" * 64,
        },
    )

    executable_path = tmp_path / "production-scan-interface"
    executable_path.write_bytes(b"CPU-only test executable\n")
    executable_path.chmod(0o755)
    source_path = tmp_path / "production-scan-interface.rs"
    source_path.write_text("fn main() {}\n", encoding="utf-8")
    binary_sha = _raw_sha(executable_path)
    source_sha = _raw_sha(source_path)
    command = [
        "cargo",
        "build",
        "-p",
        "hawking-core",
        "--example",
        "ascension_qwen30_streamed_source_range_admission_production_scan_interface",
    ]
    production_binary_path, production_binary = _write_sealed(
        tmp_path / "production-binary.json",
        {
            "schema": outer.PRODUCTION_BINARY_SCHEMA,
            "status": outer.PRODUCTION_BINARY_STATUS,
            "cpu_only": True,
            "production_hash_scan_backend_compiled": True,
            "production_hash_scan_executed": False,
            "source_root_opened_or_statted": False,
            "source_payload_opened": False,
            "source_teacher_or_logits_executed": False,
            "model_gpu_server_hcli_or_tps_action": False,
            "lease_issued_or_consumed": False,
            "binary_sha256": binary_sha,
            "source_sha256": source_sha,
            "executable": {
                "path": str(executable_path),
                "bytes": executable_path.stat().st_size,
                "sha256": binary_sha,
            },
            "source": {
                "path": str(source_path),
                "bytes": source_path.stat().st_size,
                "sha256": source_sha,
            },
            "bootstrap_preflight": _sealed_pointer(preflight_path, preflight),
            "bootstrap_binary": _sealed_pointer(bootstrap_binary_path, bootstrap_binary),
            "bootstrap_resource": _sealed_pointer(bootstrap_resource_path, bootstrap_resource),
            "compiled_command": command,
            "compiled_command_sha256": _canonical_sha(command),
        },
    )
    production_resource_path, production_resource = _write_sealed(
        tmp_path / "production-resource.json",
        {
            "schema": outer.PRODUCTION_RESOURCE_SCHEMA,
            "status": outer.PRODUCTION_RESOURCE_STATUS,
            "prepared": True,
            "fresh_observation": True,
            "observed_after_production_binary_binding": True,
            "exclusive_clean_window": True,
            "zero_swap": True,
            "zero_swapouts": True,
            "no_active_q30_or_q80_capture_child": True,
            "resource_admitted_for_one_future_child": True,
            "source_payload_opened": False,
            "source_model_loaded": False,
            "source_teacher_or_logits_executed": False,
            "native_phase_started": False,
            "gpu_server_hcli_or_tps_action": False,
            "lease_issued_or_consumed": False,
            "child_started": False,
            "swap_used_bytes": 0,
            "swapouts_pages_delta": 0,
            "reclaimable_bytes": 10_000,
            "minimum_reclaimable_bytes_required": 1,
            "production_binary_binding": _sealed_pointer(
                production_binary_path, production_binary
            ),
            "bootstrap_resource_ancestry": _sealed_pointer(
                bootstrap_resource_path, bootstrap_resource
            ),
            "production_resource_window_identity_sha256": "f" * 64,
        },
    )

    revision = "1" * 40
    metadata_authority = {
        "schema": materializer.METADATA_SCHEMA,
        "status": materializer.METADATA_STATUS,
        "source": {
            "source_revision": revision,
            "source_shard_count": outer.SOURCE_SHARDS,
            "source_tensor_count": outer.SOURCE_TENSORS,
        },
    }
    metadata_content_sha = _canonical_sha(metadata_authority)
    metadata_path, metadata = _write_raw(
        tmp_path / "metadata-range.json",
        {
            "authority": metadata_authority,
            "authority_content_sha256": metadata_content_sha,
        },
    )
    semantics_path, semantics = _write_raw(
        tmp_path / "semantics.json",
        {
            "schema": materializer.SEMANTICS_SCHEMA,
            "status": materializer.SEMANTICS_STATUS,
            "pinned_source_binding": {"source_revision": revision},
            "consumed_metadata_contracts": {
                "range_authority": {
                    "document_sha256": _raw_sha(metadata_path),
                    "authority_content_sha256": metadata_content_sha,
                }
            },
        },
    )
    runtime_path, runtime = _write_sealed(
        tmp_path / "runtime-authority.json",
        {
            "schema": materializer.RUNTIME_SCHEMA,
            "status": materializer.RUNTIME_STATUS,
            "prepared": True,
            "runtime_admission_earned": False,
            "source_payload_validation_executed": False,
            "sealed_metadata_authority_binding": {
                "metadata_range_authority": {
                    "raw_document_sha256": _raw_sha(metadata_path)
                },
                "authority_content_sha256": metadata_content_sha,
            },
            "metadata_semantics_binding": {
                "operator_semantics_attester": {
                    "raw_document_sha256": _raw_sha(semantics_path)
                }
            },
        },
    )
    interface_path, interface = _write_sealed(
        tmp_path / "production-interface.json",
        {
            "schema": outer.INTERFACE_SCHEMA,
            "status": outer.INTERFACE_STATUS,
            "prepared": True,
            "execution_authorized": False,
            "strict_non_fixture_boundary": {"before_source_root_access": True},
            "future_bounded_hash_scan": {
                "source_shards": outer.SOURCE_SHARDS,
                "source_tensors": outer.SOURCE_TENSORS,
                "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
            },
            "input_authorities": {
                "metadata_range_authority": _full_evidence(
                    metadata_path, metadata, sealed=False
                ),
                "independent_non_fixture_semantics_attester": _full_evidence(
                    semantics_path, semantics, sealed=False
                ),
                "runtime_admission_producer_authority": _full_evidence(
                    runtime_path, runtime, sealed=True
                ),
                "metadata_authority_content_sha256": metadata_content_sha,
            },
        },
    )
    return {
        "metadata": (metadata_path, metadata),
        "semantics": (semantics_path, semantics),
        "runtime": (runtime_path, runtime),
        "interface": (interface_path, interface),
        "preflight": (preflight_path, preflight),
        "bootstrap_binary": (bootstrap_binary_path, bootstrap_binary),
        "bootstrap_resource": (bootstrap_resource_path, bootstrap_resource),
        "production_binary": (production_binary_path, production_binary),
        "production_resource": (production_resource_path, production_resource),
    }


def _build(chain: dict[str, tuple[Path, dict[str, object]]]) -> dict[str, object]:
    return materializer.build_production_authority(
        metadata_path=chain["metadata"][0],
        semantics_path=chain["semantics"][0],
        runtime_authority_path=chain["runtime"][0],
        interface_path=chain["interface"][0],
        bootstrap_preflight_path=chain["preflight"][0],
        bootstrap_binary_path=chain["bootstrap_binary"][0],
        bootstrap_resource_path=chain["bootstrap_resource"][0],
        production_binary_path=chain["production_binary"][0],
        production_resource_path=chain["production_resource"][0],
        nonce_bytes=b"A" * 32,
    )


def test_valid_nonfixture_chain_materializes_sealed_outer_consumable_authority(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path)
    result = _build(chain)
    assert result["schema"] == outer.PRODUCTION_AUTHORITY_SCHEMA
    assert result["status"] == outer.PRODUCTION_AUTHORITY_STATUS
    assert result["exact_scan_nonce_sha256"] == hashlib.sha256(b"A" * 32).hexdigest()
    assert result["execution_boundary"] == {
        "source_root_argument_or_stat_performed": False,
        "source_payload_opened": False,
        "source_model_loaded": False,
        "source_teacher_or_logits_executed": False,
        "native_phase_started": False,
        "gpu_server_hcli_or_tps_action": False,
        "lease_issued_or_consumed": False,
        "child_started": False,
    }
    verify(result, label="materialized production authority")

    authority_path, _ = _write_sealed(tmp_path / "authority.json", result)
    authority = outer._sealed(authority_path, label="materialized production authority")
    interface = outer._sealed(chain["interface"][0], label="production interface")
    production_binary = outer._sealed(chain["production_binary"][0], label="production binary")
    production_resource = outer._sealed(
        chain["production_resource"][0], label="production resource"
    )
    assert outer._validate_production_authority(
        authority,
        interface=interface,
        production_binary=production_binary,
        production_resource=production_resource,
    ) == result["exact_scan_nonce_sha256"]
    outer_result = outer.build_outer_preflight(
        bootstrap_preflight_path=chain["preflight"][0],
        bootstrap_binary_path=chain["bootstrap_binary"][0],
        bootstrap_resource_path=chain["bootstrap_resource"][0],
        production_binary_path=chain["production_binary"][0],
        production_resource_path=chain["production_resource"][0],
        production_interface_path=chain["interface"][0],
        production_authority_path=authority_path,
    )
    assert outer_result["blockers"] == ["fresh_production_hash_scan_bootstrap_lease_absent"]
    assert outer_result["spawn_permitted"] is False


def test_fixture_marked_raw_metadata_is_refused_before_authority_materializes(
    tmp_path: Path,
) -> None:
    chain = _chain(tmp_path)
    fixture_metadata_path, _ = _write_raw(
        tmp_path / "fixture-evidence.json",
        {
            "fixture_only": True,
            "authority": {
                "schema": materializer.METADATA_SCHEMA,
                "status": materializer.METADATA_STATUS,
            },
        },
    )
    chain["metadata"] = (fixture_metadata_path, chain["metadata"][1])

    with pytest.raises(
        materializer.ProductionAuthorityMaterializationError, match="fixture-only"
    ):
        _build(chain)
