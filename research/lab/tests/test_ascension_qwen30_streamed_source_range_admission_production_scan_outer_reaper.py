"""Focused CPU/file-only tests for the Q30 production hash-scan outer/reaper."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

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


def _command_sha(command: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(command, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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


def _complete_chain(tmp_path: Path) -> dict[str, tuple[Path, dict[str, object]]]:
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
    resource_path, resource = _write(
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
    executable_path = tmp_path / "production-hash-scan-child"
    executable_path.write_bytes(b"synthetic compiled production child\n")
    executable_path.chmod(0o755)
    source_path = tmp_path / "production-hash-scan-child.rs"
    source_path.write_text("// synthetic source binding only\n", encoding="utf-8")
    production_binary_sha = _raw_sha(executable_path)
    production_source_sha = _raw_sha(source_path)
    command = [
        "cargo",
        "build",
        "-p",
        "hawking-core",
        "--example",
        "ascension_qwen30_streamed_source_range_admission_production_scan_interface",
    ]
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
            "binary_sha256": production_binary_sha,
            "source_sha256": production_source_sha,
            "executable": {
                "path": str(executable_path),
                "bytes": executable_path.stat().st_size,
                "sha256": production_binary_sha,
            },
            "source": {
                "path": str(source_path),
                "bytes": source_path.stat().st_size,
                "sha256": production_source_sha,
            },
            "bootstrap_preflight": _pointer(preflight_path, preflight),
            "bootstrap_binary": _pointer(bootstrap_binary_path, bootstrap_binary),
            "bootstrap_resource": _pointer(resource_path, resource),
            "compiled_command": command,
            "compiled_command_sha256": _command_sha(command),
        },
    )
    production_resource_path, production_resource = _write(
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
            "production_binary_binding": _pointer(
                production_binary_path, production_binary
            ),
            "bootstrap_resource_ancestry": _pointer(resource_path, resource),
            "production_resource_window_identity_sha256": SHA_E,
        },
    )
    interface_path, interface = _write(
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
        },
    )
    authority_path, authority = _write(
        tmp_path / "production-authority.json",
        {
            "schema": outer.PRODUCTION_AUTHORITY_SCHEMA,
            "status": outer.PRODUCTION_AUTHORITY_STATUS,
            "fresh_for_this_exact_scan": True,
            "one_shot": True,
            "non_inference_hash_scan_only": True,
            "source_root_open_only_after_all_authorities_validate": True,
            "fixture_only": False,
            "synthetic_fixture_only": False,
            "production_adapter_forbidden": False,
            "source_teacher_or_logits_authorized": False,
            "model_gpu_server_hcli_or_tps_authorized": False,
            "lease_consumed": False,
            "immutable_bindings": {
                "interface_authority": _pointer(interface_path, interface),
                "production_binary": _pointer(production_binary_path, production_binary),
                "production_resource_admission": _pointer(
                    production_resource_path, production_resource
                ),
            },
            "geometry": {
                "source_shards": outer.SOURCE_SHARDS,
                "source_tensors": outer.SOURCE_TENSORS,
                "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
                "maximum_live_raw_bf16_windows": 1,
            },
            "exact_scan_nonce_sha256": SHA_A,
        },
    )
    lease_path, lease = _write(
        tmp_path / "production-lease.json",
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
            "production_source_hash_scan_only": True,
            "fixture_only": False,
            "synthetic_fixture_only": False,
            "production_adapter_forbidden": False,
            "source_teacher_or_logits_authorized": False,
            "model_gpu_server_hcli_or_tps_authorized": False,
            "lease_consumed_by_this_preflight": False,
            "production_scan_authority": _pointer(authority_path, authority),
            "production_binary_sha256": production_binary_sha,
            "production_binary_binding": _pointer(production_binary_path, production_binary),
            "bootstrap_resource_ancestry": _pointer(resource_path, resource),
            "production_resource_admission": _pointer(
                production_resource_path, production_resource
            ),
            "resource_window_identity_sha256": SHA_C,
            "fresh_production_resource_observation": {
                "observed_after_production_binary_binding": True,
                "exclusive_clean_window": True,
                "zero_swap": True,
                "zero_swapouts": True,
                "no_active_q30_or_q80_capture_child": True,
                "source_payload_opened": False,
                "source_model_loaded": False,
                "gpu_server_hcli_or_tps_action": False,
                "lease_issued_or_consumed": False,
                "child_started": False,
                "swap_used_bytes": 0,
                "swapouts_pages_delta": 0,
                "reclaimable_bytes": 10_000,
                "minimum_reclaimable_bytes_required": 1,
            },
            "lease_id": SHA_B,
        },
    )
    return {
        "preflight": (preflight_path, preflight),
        "bootstrap_binary": (bootstrap_binary_path, bootstrap_binary),
        "resource": (resource_path, resource),
        "production_binary": (production_binary_path, production_binary),
        "production_resource": (production_resource_path, production_resource),
        "interface": (interface_path, interface),
        "authority": (authority_path, authority),
        "lease": (lease_path, lease),
    }


def _build(chain: dict[str, tuple[Path, dict[str, object]]]) -> dict[str, object]:
    return outer.build_outer_preflight(
        bootstrap_preflight_path=chain["preflight"][0],
        bootstrap_binary_path=chain["bootstrap_binary"][0],
        bootstrap_resource_path=chain["resource"][0],
        production_binary_path=chain["production_binary"][0],
        production_resource_path=chain["production_resource"][0],
        production_interface_path=chain["interface"][0],
        production_authority_path=chain["authority"][0],
        bootstrap_lease_path=chain["lease"][0],
    )


def test_live_ancestry_without_new_production_chain_refuses_without_spawn(tmp_path: Path) -> None:
    chain = _complete_chain(tmp_path)
    result = outer.build_outer_preflight(
        bootstrap_preflight_path=chain["preflight"][0],
        bootstrap_binary_path=chain["bootstrap_binary"][0],
        bootstrap_resource_path=chain["resource"][0],
    )
    assert result["status"] == outer.REFUSED_STATUS
    assert result["spawn_permitted"] is False
    assert result["legacy_bootstrap_resource_is_ancestry_only"] is True
    assert result["blockers"] == [
        "compiled_production_hash_scan_binary_binding_absent",
        "sealed_production_hash_scan_interface_absent",
        "fresh_production_binary_bound_resource_admission_absent",
        "fresh_production_hash_scan_authority_absent",
        "fresh_production_hash_scan_bootstrap_lease_absent",
    ]
    assert result["execution_boundary"]["child_spawned"] is False
    verify(result, label="production outer refusal")


def test_complete_future_chain_prepares_only_without_spawn(tmp_path: Path) -> None:
    result = _build(_complete_chain(tmp_path))
    assert result["status"] == outer.PREPARED_STATUS
    assert result["prepared"] is True
    assert result["spawn_permitted"] is False
    assert result["blockers"] == []
    assert result["future_lifecycle"]["child_capture"]["source_tensors"] == 18_867
    assert result["execution_boundary"]["replay_reservation_created"] is False
    verify(result, label="production outer prepared")


def test_stale_production_binary_bound_resource_admission_refuses_even_with_lease(
    tmp_path: Path,
) -> None:
    chain = _complete_chain(tmp_path)
    production_binary_path, production_binary = chain["production_binary"]
    resource_path, resource = chain["resource"]
    stale_resource = dict(chain["production_resource"][1])
    stale_resource.pop("seal_sha256")
    stale_resource["observed_after_production_binary_binding"] = False
    stale_resource["production_binary_binding"] = _pointer(
        production_binary_path, production_binary
    )
    stale_resource["bootstrap_resource_ancestry"] = _pointer(resource_path, resource)
    stale_resource_path, _ = _write(tmp_path / "stale-production-resource.json", stale_resource)
    result = outer.build_outer_preflight(
        bootstrap_preflight_path=chain["preflight"][0],
        bootstrap_binary_path=chain["bootstrap_binary"][0],
        bootstrap_resource_path=resource_path,
        production_binary_path=production_binary_path,
        production_resource_path=stale_resource_path,
        production_interface_path=chain["interface"][0],
        production_authority_path=chain["authority"][0],
        bootstrap_lease_path=chain["lease"][0],
    )
    assert result["status"] == outer.REFUSED_STATUS
    assert any(
        blocker.startswith("fresh_production_binary_bound_resource_admission_invalid:")
        and "observed_after_production_binary_binding" in blocker
        for blocker in result["blockers"]
    )
    assert any(
        blocker.startswith("fresh_production_hash_scan_authority_invalid:")
        and "production resource admission does not bind" in blocker
        for blocker in result["blockers"]
    )
    assert any(
        blocker.startswith("fresh_production_hash_scan_bootstrap_lease_invalid:")
        and "fresh resource admission does not bind" in blocker
        for blocker in result["blockers"]
    )
    assert result["execution_boundary"]["child_spawned"] is False


def test_legacy_bootstrap_resource_cannot_substitute_for_production_resource(
    tmp_path: Path,
) -> None:
    chain = _complete_chain(tmp_path)
    result = outer.build_outer_preflight(
        bootstrap_preflight_path=chain["preflight"][0],
        bootstrap_binary_path=chain["bootstrap_binary"][0],
        bootstrap_resource_path=chain["resource"][0],
        production_binary_path=chain["production_binary"][0],
        production_resource_path=chain["resource"][0],
        production_interface_path=chain["interface"][0],
        production_authority_path=chain["authority"][0],
        bootstrap_lease_path=chain["lease"][0],
    )
    assert result["status"] == outer.REFUSED_STATUS
    assert any(
        blocker.startswith("fresh_production_binary_bound_resource_admission_invalid:")
        and "schema/status drifted" in blocker
        for blocker in result["blockers"]
    )
    assert result["execution_boundary"]["child_spawned"] is False


def test_fake_child_reap_terminal_and_release_require_one_non_teacher_chain(
    tmp_path: Path,
) -> None:
    chain = _complete_chain(tmp_path)
    preflight = _build(chain)
    production_binary_sha = str(chain["production_binary"][1]["binary_sha256"])
    authority_seal = str(preflight["production_scan_authority"]["seal_sha256"])
    reservation = {
        "schema": outer.REPLAY_SCHEMA,
        "status": outer.REPLAY_STATUS,
        "create_new_before_child": True,
        "one_child_maximum": True,
        "replay_or_relaunch_forbidden": True,
        "attempt": 1,
        "lease_id": SHA_B,
        "outer_preflight_seal_sha256": preflight["seal_sha256"],
    }
    child = seal(
        {
            "schema": outer.CAPTURE_SCHEMA,
            "status": outer.CAPTURE_STATUS,
            "production_hash_scan_earned": True,
            "receipt_written_last": True,
            "source_handles_closed": True,
            "reader_cache_zeroed": True,
            "source_teacher_or_logits_executed": False,
            "operator_or_reader_execution_attestation_emitted": False,
            "source_teacher_runtime_admission_earned": False,
            "model_gpu_server_hcli_or_tps_action": False,
            "geometry": {
                "source_shards": outer.SOURCE_SHARDS,
                "source_tensors": outer.SOURCE_TENSORS,
                "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
                "maximum_live_raw_bf16_windows": 1,
            },
            "production_binary_sha256": production_binary_sha,
            "production_authority_seal_sha256": authority_seal,
            "lease_id": SHA_B,
        }
    )
    terminal = seal(
        {
            "schema": outer.OUTER_TERMINAL_SCHEMA,
            "status": outer.OUTER_TERMINAL_STATUS,
            "child_reaped": True,
            "terminal_receipt_written_after_child_capture": True,
            "automatic_retry_disabled": True,
            "lease_reuse_prohibited": True,
            "child_timed_out": False,
            "child_exit_code": 0,
            "lease_id": SHA_B,
            "production_binary_sha256": production_binary_sha,
            "production_authority_seal_sha256": authority_seal,
            "child_capture_seal_sha256": child["seal_sha256"],
        }
    )
    release = seal(
        {
            "schema": outer.RELEASE_SCHEMA,
            "status": outer.RELEASE_STATUS,
            "release_after_outer_terminal": True,
            "one_shot_lease_finalized": True,
            "retry_or_relaunch_forbidden": True,
            "source_teacher_or_logits_authorized": False,
            "native_or_gpu_server_hcli_authorized": False,
            "artifacts_deleted_or_evicted": False,
            "lease_id": SHA_B,
            "outer_terminal_seal_sha256": terminal["seal_sha256"],
            "child_capture_seal_sha256": child["seal_sha256"],
        }
    )
    outer.validate_fake_child_reap_terminal_release(
        reservation=reservation,
        child_capture=child,
        outer_terminal=terminal,
        release=release,
        outer_preflight_seal_sha256=str(preflight["seal_sha256"]),
        production_binary_sha256=production_binary_sha,
        production_authority_seal_sha256=authority_seal,
        lease_id=SHA_B,
    )

    child["source_teacher_or_logits_executed"] = True
    with pytest.raises(outer.ProductionScanOuterError, match="source_teacher_or_logits_executed"):
        outer.validate_fake_child_reap_terminal_release(
            reservation=reservation,
            child_capture=child,
            outer_terminal=terminal,
            release=release,
            outer_preflight_seal_sha256=str(preflight["seal_sha256"]),
            production_binary_sha256=production_binary_sha,
            production_authority_seal_sha256=authority_seal,
            lease_id=SHA_B,
        )
