"""Focused CPU/file-only tests for the Q30 bootstrap outer/reaper grammar."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_bootstrap_outer_preflight as outer,
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


def _complete_chain(tmp_path: Path) -> dict[str, tuple[Path, dict[str, object]]]:
    preflight_path, preflight = _write(tmp_path / "preflight.json", _preflight())
    binary_path, binary = _write(
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
    resource_path, resource = _write(
        tmp_path / "resource.json",
        {
            "schema": outer.RESOURCE_SCHEMA,
            "status": outer.RESOURCE_STATUS,
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
    lease_path, lease = _write(
        tmp_path / "lease.json",
        {
            "schema": outer.LEASE_SCHEMA,
            "status": outer.LEASE_STATUS,
            "fresh_for_this_exact_launch": True,
            "one_shot": True,
            "non_inference_only": True,
            "new_capture_root_required": True,
            "existing_output_reuse_forbidden": True,
            "replay_or_relaunch_forbidden": True,
            "separate_from_source_teacher_lease": True,
            "source_teacher_or_logits_authorized": False,
            "model_gpu_server_hcli_or_tps_authorized": False,
            "lease_consumed_by_this_preflight": False,
            "bootstrap_binary_sha256": SHA_D,
            "resource_window_identity_sha256": SHA_C,
            "bootstrap_preflight": _pointer(preflight_path, preflight),
            "resource_admission": _pointer(resource_path, resource),
            "lease_id": SHA_B,
        },
    )
    return {
        "preflight": (preflight_path, preflight),
        "binary": (binary_path, binary),
        "resource": (resource_path, resource),
        "lease": (lease_path, lease),
    }


def test_missing_future_authorities_refuse_before_spawn(tmp_path: Path) -> None:
    chain = _complete_chain(tmp_path)
    result = outer.build_outer_preflight(preflight_path=chain["preflight"][0])
    assert result["status"] == outer.REFUSED_STATUS
    assert result["spawn_permitted"] is False
    assert result["blockers"] == [
        "bootstrap_binary_binding_absent",
        "fresh_zero_swap_resource_admission_absent",
        "fresh_non_inference_bootstrap_lease_absent",
    ]
    assert result["execution_boundary"]["child_spawned"] is False
    verify(result, label="outer refusal")


def test_all_sealed_future_inputs_prepare_only_a_non_spawn_reservation(
    tmp_path: Path,
) -> None:
    chain = _complete_chain(tmp_path)
    result = outer.build_outer_preflight(
        preflight_path=chain["preflight"][0],
        binary_path=chain["binary"][0],
        resource_path=chain["resource"][0],
        lease_path=chain["lease"][0],
    )
    assert result["status"] == outer.PREPARED_STATUS
    assert result["prepared"] is True
    assert result["spawn_permitted"] is False
    assert result["blockers"] == []
    assert result["execution_boundary"]["replay_reservation_created"] is False
    assert result["future_lifecycle"]["replay_reservation"]["one_child_maximum"] is True
    verify(result, label="outer prepared")


def test_nonzero_swap_resource_receipt_fails_closed(tmp_path: Path) -> None:
    chain = _complete_chain(tmp_path)
    preflight_path, preflight = chain["preflight"]
    binary_path, _ = chain["binary"]
    resource = dict(chain["resource"][1])
    resource["swap_used_bytes"] = 1
    resource_path, resource = _write(tmp_path / "bad-resource.json", resource)
    lease = dict(chain["lease"][1])
    lease["resource_admission"] = _pointer(resource_path, resource)
    lease_path, _ = _write(tmp_path / "bad-resource-lease.json", lease)

    result = outer.build_outer_preflight(
        preflight_path=preflight_path,
        binary_path=binary_path,
        resource_path=resource_path,
        lease_path=lease_path,
    )
    assert result["status"] == outer.REFUSED_STATUS
    assert any("must show zero swap" in item for item in result["blockers"])
    assert "bootstrap_lease_not_evaluated_without_valid_binary_and_resource" in result[
        "blockers"
    ]


def _future_bundle() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    reservation = {
        "schema": outer.REPLAY_SCHEMA,
        "status": outer.REPLAY_STATUS,
        "create_new_before_child": True,
        "one_child_maximum": True,
        "replay_or_relaunch_forbidden": True,
        "attempt": 1,
        "lease_id": SHA_B,
        "preflight_seal_sha256": SHA_A,
    }
    child = {
        "schema": outer.CAPTURE_SCHEMA,
        "status": outer.CAPTURE_STATUS,
        "non_inference_only": True,
        "one_bounded_window": True,
        "flat_runtime_range_map_emitted": True,
        "operator_attestation_emitted": True,
        "range_reader_attestation_emitted": True,
        "receipt_written_last": True,
        "source_teacher_started": False,
        "logits_or_vectors_written": False,
        "source_model_loaded": False,
        "gpu_server_hcli_or_tps_action": False,
        "lease_id": SHA_B,
        "binary_sha256": SHA_D,
    }
    terminal = {
        "schema": outer.OUTER_TERMINAL_SCHEMA,
        "status": outer.OUTER_TERMINAL_STATUS,
        "child_reaped": True,
        "terminal_receipt_written_last": True,
        "automatic_retry_disabled": True,
        "lease_reuse_prohibited": True,
        "child_timed_out": False,
        "child_exit_code": 0,
        "lease_id": SHA_B,
    }
    return reservation, child, terminal


def test_fake_child_and_replay_shape_require_exactly_one_attempt() -> None:
    reservation, child, terminal = _future_bundle()
    outer.validate_fake_child_and_replay(
        reservation=reservation,
        child_capture=child,
        outer_terminal=terminal,
        lease_id=SHA_B,
        preflight_seal_sha256=SHA_A,
        binary_sha256=SHA_D,
    )

    reservation["attempt"] = 2
    with pytest.raises(outer.BootstrapOuterError, match="exactly one"):
        outer.validate_fake_child_and_replay(
            reservation=reservation,
            child_capture=child,
            outer_terminal=terminal,
            lease_id=SHA_B,
            preflight_seal_sha256=SHA_A,
            binary_sha256=SHA_D,
        )

    reservation["attempt"] = 1
    child["source_teacher_started"] = True
    with pytest.raises(outer.BootstrapOuterError, match="source_teacher_started"):
        outer.validate_fake_child_and_replay(
            reservation=reservation,
            child_capture=child,
            outer_terminal=terminal,
            lease_id=SHA_B,
            preflight_seal_sha256=SHA_A,
            binary_sha256=SHA_D,
        )

