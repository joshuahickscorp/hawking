"""Focused read-only tests for Q30 bootstrap resource admission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_bootstrap_outer_preflight as outer,
)
from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_bootstrap_resource_admission_preflight as resource,
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


def _pointer(path: Path, document: dict[str, object]) -> dict[str, object]:
    return {
        "raw_document_sha256": _raw_sha(path),
        "seal_sha256": str(document["seal_sha256"]),
    }


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
            "maximum_declared_bf16_window_bytes": 524_288,
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


def _chain(tmp_path: Path) -> tuple[Path, Path]:
    preflight_path, preflight = _write(tmp_path / "preflight.json", _preflight())
    binary_path, _ = _write(
        tmp_path / "binary.json",
        {
            "schema": outer.BINARY_SCHEMA,
            "status": outer.BINARY_STATUS,
            "cpu_only": True,
            "scan_or_runtime_executed": False,
            "binary_sha256": SHA_D,
            "source_sha256": SHA_E,
            "bootstrap_preflight": _pointer(preflight_path, preflight),
        },
    )
    return preflight_path, binary_path


def _sample(
    *,
    swap_used_bytes: int = 0,
    swapouts_pages: int = 17,
    reclaimable_bytes: int = resource.MINIMUM_RECLAIMABLE_BYTES + 4096,
    q30: tuple[str, ...] = (),
    q80: tuple[str, ...] = (),
) -> resource.HostSample:
    return resource.HostSample(
        backend="synthetic-read-only-test",
        swap_used_bytes=swap_used_bytes,
        swapouts_pages=swapouts_pages,
        reclaimable_bytes=reclaimable_bytes,
        q30_capture_children=q30,
        q80_capture_children=q80,
    )


def _provider(*samples: resource.HostSample):
    values = iter(samples)
    return lambda: next(values)


def test_clean_observations_emit_the_exact_outer_compatible_resource_receipt(
    tmp_path: Path,
) -> None:
    preflight_path, binary_path = _chain(tmp_path)
    result = resource.build_resource_admission(
        preflight_path=preflight_path,
        binary_path=binary_path,
        snapshot_provider=_provider(_sample(), _sample()),
    )
    assert result["schema"] == outer.RESOURCE_SCHEMA
    assert result["status"] == outer.RESOURCE_STATUS
    assert result["prepared"] is True
    assert result["resource_admitted_for_one_future_child"] is True
    assert result["bounded_one_source_hash_scan_resource_profile"] == {
        "exactly_one_future_non_inference_hash_scan_child": True,
        "maximum_concurrent_source_hash_scan_children": 1,
        "maximum_positioned_read_bytes": outer.MAX_WINDOW_BYTES,
        "maximum_live_raw_bf16_windows": 1,
        "maximum_cached_raw_bf16_bytes": outer.MAX_WINDOW_BYTES,
        "maximum_shards": outer.PRODUCTION_SHARDS,
        "maximum_tensors": outer.PRODUCTION_TENSORS,
        "minimum_reclaimable_bytes_required": resource.MINIMUM_RECLAIMABLE_BYTES,
        "source_teacher_or_logits_allowed": False,
        "source_model_residency_allowed": False,
        "model_gpu_server_hcli_or_tps_allowed": False,
        "source_root_statted_or_opened": False,
    }
    verify(result, label="resource admission")

    resource_path, resource_document = _write(tmp_path / "resource.json", result)
    preflight = outer._sealed(preflight_path, label="preflight")
    binary = outer._sealed(binary_path, label="binary")
    binary_sha = outer._validate_binary(binary, preflight=preflight)
    resource_window = outer._validate_resource(
        outer._sealed(resource_path, label="resource"),
        preflight=preflight,
        binary_sha=binary_sha,
    )
    assert resource_window == resource_document["resource_window_identity_sha256"]


def test_nonzero_swap_or_swapout_growth_refuses_without_child_or_lease(
    tmp_path: Path,
) -> None:
    preflight_path, binary_path = _chain(tmp_path)
    result = resource.build_resource_admission(
        preflight_path=preflight_path,
        binary_path=binary_path,
        snapshot_provider=_provider(
            _sample(swap_used_bytes=1, swapouts_pages=4),
            _sample(swap_used_bytes=1, swapouts_pages=5),
        ),
    )
    assert result["status"] == resource.REFUSED_STATUS
    assert "nonzero_swap_used_bytes" in result["blockers"]
    assert "nonzero_swapout_pages_delta" in result["blockers"]
    assert result["execution_boundary"]["source_root_argument_or_stat_performed"] is False
    assert result["execution_boundary"]["capture_child_spawned"] is False
    assert result["execution_boundary"]["lease_issued_or_consumed_or_released"] is False
    verify(result, label="resource refusal")


def test_capture_child_or_safety_floor_failure_refuses_closed(tmp_path: Path) -> None:
    preflight_path, binary_path = _chain(tmp_path)
    result = resource.build_resource_admission(
        preflight_path=preflight_path,
        binary_path=binary_path,
        snapshot_provider=_provider(
            _sample(reclaimable_bytes=1, q80=("f" * 64,)),
            _sample(reclaimable_bytes=1, q80=("f" * 64,)),
        ),
    )
    assert result["status"] == resource.REFUSED_STATUS
    assert "active_q80_capture_child_detected" in result["blockers"]
    assert "reclaimable_bytes_below_explicit_safety_floor" in result["blockers"]
    assert result["source_payload_opened"] is False
    assert result["source_model_loaded"] is False
    assert result["child_started"] is False
