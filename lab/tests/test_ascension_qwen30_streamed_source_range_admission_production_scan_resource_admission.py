"""Focused read-only tests for production-binary-bound Q30 resource admission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_production_scan_outer_reaper as outer,
)
from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_production_scan_resource_admission as resource,
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


def _bootstrap_preflight() -> dict[str, object]:
    return {
        "schema": outer.bootstrap.PREFLIGHT_SCHEMA,
        "status": outer.bootstrap.PREFLIGHT_STATUS,
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


def _chain(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict[str, object]]:
    preflight_path, preflight = _write(tmp_path / "bootstrap-preflight.json", _bootstrap_preflight())
    bootstrap_binary_path, bootstrap_binary = _write(
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
    bootstrap_resource_path, bootstrap_resource = _write(
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
    executable = tmp_path / "production-child"
    executable.write_bytes(b"synthetic production hash scanner\n")
    executable.chmod(0o755)
    source = tmp_path / "production-child.rs"
    source.write_text("// synthetic CPU-only source\n", encoding="utf-8")
    binary_sha = _raw_sha(executable)
    source_sha = _raw_sha(source)
    command = [
        "cargo",
        "build",
        "-p",
        "hawking-core",
        "--example",
        "ascension_qwen30_streamed_source_range_admission_production_scan_interface",
    ]
    command_sha = hashlib.sha256(
        json.dumps(command, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    production_binary_path, production_binary = _write(
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
            "executable": {"path": str(executable), "bytes": executable.stat().st_size, "sha256": binary_sha},
            "source": {"path": str(source), "bytes": source.stat().st_size, "sha256": source_sha},
            "compiled_command": command,
            "compiled_command_sha256": command_sha,
            "bootstrap_preflight": _pointer(preflight_path, preflight),
            "bootstrap_binary": _pointer(bootstrap_binary_path, bootstrap_binary),
            "bootstrap_resource": _pointer(bootstrap_resource_path, bootstrap_resource),
        },
    )
    return (
        preflight_path,
        bootstrap_binary_path,
        bootstrap_resource_path,
        production_binary_path,
        production_binary,
    )


def _sample(
    *,
    swap: int = 0,
    swapouts: int = 10,
    reclaimable: int = resource.MINIMUM_RECLAIMABLE_BYTES + 4096,
    q30: tuple[str, ...] = (),
    q80: tuple[str, ...] = (),
) -> resource.legacy.HostSample:
    return resource.legacy.HostSample(
        backend="synthetic-read-only-test",
        swap_used_bytes=swap,
        swapouts_pages=swapouts,
        reclaimable_bytes=reclaimable,
        q30_capture_children=q30,
        q80_capture_children=q80,
    )


def _provider(*samples: resource.legacy.HostSample):
    values = iter(samples)
    return lambda: next(values)


def test_clean_samples_bind_fresh_production_binary_and_remain_outer_consumable(
    tmp_path: Path,
) -> None:
    preflight, bootstrap_binary, bootstrap_resource, production_binary, production_doc = _chain(tmp_path)
    result = resource.build_resource_admission(
        bootstrap_preflight_path=preflight,
        bootstrap_binary_path=bootstrap_binary,
        bootstrap_resource_path=bootstrap_resource,
        production_binary_path=production_binary,
        snapshot_provider=_provider(_sample(), _sample()),
    )
    assert result["schema"] == outer.PRODUCTION_RESOURCE_SCHEMA
    assert result["status"] == outer.PRODUCTION_RESOURCE_STATUS
    assert result["prepared"] is True
    assert result["observed_after_production_binary_binding"] is True
    assert result["production_binary_binding"] == _pointer(production_binary, production_doc)
    assert result["source_teacher_or_logits_executed"] is False
    verify(result, label="production resource admission")

    resource_path, _ = _write(tmp_path / "production-resource.json", result)
    outer._validate_production_resource(
        outer._sealed(resource_path, label="production resource"),
        production_binary=outer._sealed(production_binary, label="production binary"),
        bootstrap_resource=outer._sealed(bootstrap_resource, label="legacy resource"),
    )


def test_swap_growth_or_capture_refuses_before_any_lease_or_child(tmp_path: Path) -> None:
    preflight, bootstrap_binary, bootstrap_resource, production_binary, _ = _chain(tmp_path)
    result = resource.build_resource_admission(
        bootstrap_preflight_path=preflight,
        bootstrap_binary_path=bootstrap_binary,
        bootstrap_resource_path=bootstrap_resource,
        production_binary_path=production_binary,
        snapshot_provider=_provider(
            _sample(swap=1, swapouts=1, q80=("f" * 64,)),
            _sample(swap=1, swapouts=2, q80=("f" * 64,)),
        ),
    )
    assert result["status"] == resource.REFUSED_STATUS
    assert "nonzero_swap_used_bytes" in result["blockers"]
    assert "nonzero_swapout_pages_delta" in result["blockers"]
    assert "active_q80_capture_child_detected" in result["blockers"]
    assert result["execution_boundary"]["source_root_argument_or_stat_performed"] is False
    assert result["execution_boundary"]["capture_child_spawned"] is False
    assert result["execution_boundary"]["lease_issued_or_consumed_or_released"] is False
    verify(result, label="production resource refusal")
