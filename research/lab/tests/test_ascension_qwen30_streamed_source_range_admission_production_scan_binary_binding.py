"""Focused CPU-only tests for Q30 production hash-scan binary binding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_production_scan_binary_binding as binding,
)
from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_production_scan_outer_reaper as outer,
)
from lab.receipts import seal, verify

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _write(path: Path, document: dict[str, object]) -> tuple[Path, dict[str, object]]:
    checked = seal(document)
    path.write_text(json.dumps(checked, sort_keys=True), encoding="utf-8")
    return path, checked


def _raw_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pointer(path: Path, document: dict[str, object]) -> dict[str, str]:
    return {
        "raw_document_sha256": _raw_sha(path),
        "seal_sha256": str(document["seal_sha256"]),
    }


def _ancestry(tmp_path: Path) -> tuple[Path, Path, Path]:
    preflight_path, preflight = _write(
        tmp_path / "bootstrap-preflight.json",
        {
            "schema": outer.bootstrap.PREFLIGHT_SCHEMA,
            "status": outer.bootstrap.PREFLIGHT_STATUS,
            "prepared": True,
            "execution_authorized": False,
            "metadata_bindings": {
                "range_authority_document_sha256": SHA_A,
                "range_authority_content_sha256": SHA_B,
                "source_index_sha256": SHA_C,
                "maximum_declared_bf16_window_bytes": 1024,
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
        },
    )
    binary_path, binary = _write(
        tmp_path / "bootstrap-binary.json",
        {
            "schema": outer.bootstrap.BINARY_SCHEMA,
            "status": outer.bootstrap.BINARY_STATUS,
            "cpu_only": True,
            "scan_or_runtime_executed": False,
            "binary_sha256": SHA_D,
            "source_sha256": SHA_E,
            "bootstrap_preflight": _pointer(preflight_path, preflight),
        },
    )
    resource_path, _ = _write(
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
            "bootstrap_binary_sha256": SHA_D,
            "bootstrap_preflight": _pointer(preflight_path, preflight),
            "resource_window_identity_sha256": SHA_C,
        },
    )
    return preflight_path, binary_path, resource_path


def test_binding_is_sealed_and_consumable_by_production_outer(tmp_path: Path) -> None:
    preflight_path, bootstrap_binary_path, resource_path = _ancestry(tmp_path)
    executable = tmp_path / "production-interface"
    executable.write_bytes(b"synthetic CPU-only production scanner\n")
    executable.chmod(0o755)
    source = tmp_path / "production-interface.rs"
    source.write_text("fn main() {}\n", encoding="utf-8")

    result = binding.build_binary_binding(
        bootstrap_preflight_path=preflight_path,
        bootstrap_binary_path=bootstrap_binary_path,
        bootstrap_resource_path=resource_path,
        executable_path=executable,
        source_path=source,
    )
    assert result["schema"] == outer.PRODUCTION_BINARY_SCHEMA
    assert result["status"] == outer.PRODUCTION_BINARY_STATUS
    assert result["production_hash_scan_executed"] is False
    assert result["execution_boundary"]["source_root_argument_or_stat_performed"] is False
    verify(result, label="production binary binding")

    binding_path, _ = _write(tmp_path / "production-binding.json", result)
    preflight = outer._sealed(preflight_path, label="bootstrap preflight")
    bootstrap_binary = outer._sealed(bootstrap_binary_path, label="bootstrap binary")
    resource = outer._sealed(resource_path, label="bootstrap resource")
    assert outer._validate_production_binary(
        outer._sealed(binding_path, label="production binary"),
        preflight=preflight,
        bootstrap_binary=bootstrap_binary,
        resource=resource,
    ) == result["binary_sha256"]


def test_symlinked_executable_or_unsealed_ancestry_is_refused(tmp_path: Path) -> None:
    preflight_path, bootstrap_binary_path, resource_path = _ancestry(tmp_path)
    target = tmp_path / "target"
    target.write_bytes(b"target")
    link = tmp_path / "link"
    link.symlink_to(target)
    source = tmp_path / "source.rs"
    source.write_text("fn main() {}\n", encoding="utf-8")

    with pytest.raises(binding.ProductionBinaryBindingError, match="regular non-symlink"):
        binding.build_binary_binding(
            bootstrap_preflight_path=preflight_path,
            bootstrap_binary_path=bootstrap_binary_path,
            bootstrap_resource_path=resource_path,
            executable_path=link,
            source_path=source,
        )
