"""Focused CPU-only tests for the Q30 bootstrap binary binding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_bootstrap_binary_binding as binding,
)
from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_bootstrap_outer_preflight as outer,
)
from lab.receipts import seal, verify

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _write_json(path: Path, document: dict[str, object]) -> tuple[Path, dict[str, object]]:
    checked = seal(document)
    path.write_text(json.dumps(checked, sort_keys=True), encoding="utf-8")
    return path, checked


def _raw_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preflight() -> dict[str, object]:
    return {
        "schema": outer.PREFLIGHT_SCHEMA,
        "status": outer.PREFLIGHT_STATUS,
        "prepared": True,
        "execution_authorized": False,
        "metadata_bindings": {
            "range_authority_document_sha256": SHA_A,
            "range_authority_content_sha256": SHA_B,
            "source_index_sha256": SHA_C,
            "maximum_declared_bf16_window_bytes": 512,
        },
        "future_bootstrap_lease": {
            "schema": outer.LEASE_SCHEMA,
            "status": outer.LEASE_STATUS,
            "one_shot": True,
            "separate_from_source_teacher_lease": True,
            "non_inference_only": True,
            "model_server_gpu_hcli_or_tps_allowed": False,
            "maximum_positioned_read_bytes": outer.MAX_WINDOW_BYTES,
            "maximum_live_raw_bf16_windows": 1,
        },
        "future_outputs_required_before_source_teacher": {
            "flat_runtime_range_map": {
                "schema": "hawking.ascension.qwen30_source_bf16_range_map.v1",
                "shards": outer.PRODUCTION_SHARDS,
                "tensors": outer.PRODUCTION_TENSORS,
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


def test_binding_is_sealed_and_consumable_by_the_existing_outer_validator(
    tmp_path: Path,
) -> None:
    preflight_path, preflight = _write_json(tmp_path / "preflight.json", _preflight())
    binary = tmp_path / "bootstrap"
    binary.write_bytes(b"cpu-only bootstrap fixture")
    source = tmp_path / "bootstrap.rs"
    source.write_text("fn main() {}\n", encoding="utf-8")

    result = binding.build_binary_binding(
        preflight_path=preflight_path,
        binary_path=binary,
        source_path=source,
    )
    assert result["schema"] == outer.BINARY_SCHEMA
    assert result["status"] == outer.BINARY_STATUS
    assert result["cpu_only"] is True
    assert result["scan_or_runtime_executed"] is False
    assert result["bootstrap_preflight"] == {
        "raw_document_sha256": _raw_sha(preflight_path),
        "seal_sha256": preflight["seal_sha256"],
    }
    verify(result, label="binary binding")

    binding_path, _ = _write_json(tmp_path / "binding.json", result)
    checked_preflight = outer._sealed(preflight_path, label="preflight")
    checked_binary = outer._sealed(binding_path, label="binding")
    assert outer._validate_binary(checked_binary, preflight=checked_preflight) == result[
        "binary_sha256"
    ]


def test_unsealed_preflight_or_symlinked_binary_is_rejected(tmp_path: Path) -> None:
    preflight = tmp_path / "unsealed-preflight.json"
    preflight.write_text(json.dumps(_preflight()), encoding="utf-8")
    binary_target = tmp_path / "bootstrap-target"
    binary_target.write_bytes(b"fixture")
    binary_link = tmp_path / "bootstrap-link"
    binary_link.symlink_to(binary_target)
    source = tmp_path / "bootstrap.rs"
    source.write_text("fn main() {}\n", encoding="utf-8")

    with pytest.raises(binding.BootstrapBinaryBindingError, match="preflight"):
        binding.build_binary_binding(
            preflight_path=preflight,
            binary_path=binary_link,
            source_path=source,
        )
