"""Focused CPU-only tests for the Q30 receipt-last source-teacher outer grammar."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lab.operators import (
    ascension_qwen30_streamed_source_teacher_outer_runner_preflight as outer,
)
from lab.receipts import seal, verify

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_sealed(path: Path, document: dict[str, object]) -> Path:
    path.write_text(json.dumps(seal(document), sort_keys=True), encoding="utf-8")
    return path


def _bridge_document() -> dict[str, object]:
    return {
        "schema": outer.DUAL_BRIDGE_SCHEMA,
        "status": outer.DUAL_BRIDGE_STATUS,
        "schema_resolution": {
            "runtime_range_map_schema": outer.RUNTIME_RANGE_MAP_SCHEMA,
            "runtime_admission_schema": outer.RUNTIME_ADMISSION_SCHEMA,
            "runtime_admission_status_only_after_bounded_source_validation": outer.RUNTIME_ADMISSION_STATUS,
            "operator_accumulation_execution_attestation": {
                "schema": outer.OPERATOR_ATTESTATION_SCHEMA,
                "status": outer.OPERATOR_ATTESTATION_STATUS,
            },
            "range_reader_exact_semantics_attestation": {
                "schema": outer.RANGE_READER_ATTESTATION_SCHEMA,
                "status": outer.RANGE_READER_ATTESTATION_STATUS,
            },
            "both_execution_attestations_required_after_source_child": True,
            "runtime_range_admission_required_before_payload_open": True,
            "bridge_does_not_authorize_execution": True,
        },
        "future_source_worker": {
            "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
            "source_layers": outer.SOURCE_LAYERS,
            "source_forwards": outer.SOURCE_FORWARDS,
            "source_f32le_vectors": 2,
            "native_f32le_vectors": 4,
            "one_bounded_window_only": True,
            "source_payloads_durable_before_eviction": True,
            "close_handles_and_clear_cache_before_eviction_receipt": True,
            "separate_native_four_vector_phase_required": True,
        },
    }


def _feasibility_document(*, prepared: bool = True) -> dict[str, object]:
    return {
        "schema": outer.FEASIBILITY_SCHEMA,
        "status": (
            outer.FEASIBILITY_PREPARED_STATUS
            if prepared
            else "REFUSED_QWEN30_LAYER_STREAMED_SOURCE_BF16_ORACLE_FEASIBILITY_UNSAFE_OR_UNPROVEN"
        ),
        "exact_trace": {
            "prefix_token_count": outer.PREFIX_TOKENS,
            "forced_token_id": outer.FORCED_TOKEN,
            "source_template_token_ids_u32le_sha256": SHA_A,
        },
        "memory_assessment": {
            "streamed_memory_arithmetic_fits": prepared,
            "zero_swap_condition_met": prepared,
        },
        "feasibility": {
            "semantic_equivalence_proven_by_external_sealed_attestation": prepared,
            "safe_streamed_plan_prepared_not_executed": prepared,
            "oracle_execution_authorized": False,
        },
    }


def _child_document(
    *,
    bridge_path: Path | None = None,
    bridge_document: dict[str, object] | None = None,
    feasibility_path: Path | None = None,
    feasibility_document: dict[str, object] | None = None,
) -> dict[str, object]:
    bridge_pointer: dict[str, object] = {"present": False}
    if bridge_path is not None and bridge_document is not None:
        bridge_pointer = {
            "present": True,
            "evidence": {
                "path": str(bridge_path.resolve()),
                "raw_document_sha256": _sha256(bridge_path),
                "seal_sha256": str(bridge_document["seal_sha256"]),
            },
        }
    feasibility_pointer: dict[str, object] = {"present": False}
    if feasibility_path is not None and feasibility_document is not None:
        prepared = feasibility_document["status"] == outer.FEASIBILITY_PREPARED_STATUS
        feasibility_pointer = {
            "present": True,
            "evidence": {
                "path": str(feasibility_path.resolve()),
                "raw_document_sha256": _sha256(feasibility_path),
                "seal_sha256": str(feasibility_document["seal_sha256"]),
            },
            "status": feasibility_document["status"],
            "semantic_equivalence_proven": prepared,
            "streamed_memory_arithmetic_fits": prepared,
            "zero_swap_condition_met": prepared,
        }
    return {
        "schema": outer.CHILD_PREFLIGHT_SCHEMA,
        "status": outer.CHILD_PREFLIGHT_STATUS,
        "input_bindings": {
            "metadata_range_authority": {
                "maximum_declared_bf16_row_window_bytes": outer.MAX_POSITIONED_READ_BYTES,
            },
            "streamed_feasibility": feasibility_pointer,
            "raw_six_vector_contract": {},
            "current_trace": {},
            "dual_attestation_runtime_admission_bridge": bridge_pointer,
        },
        "trace_binding": {
            "source_template_token_count": outer.PREFIX_TOKENS,
            "forced_identical_continuation_token_id": outer.FORCED_TOKEN,
            "source_template_token_ids_u32le_sha256": SHA_A,
        },
        "future_child_interface": {
            "execution_shape": {
                "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
                "maximum_live_raw_bf16_windows": 1,
                "source_layers": outer.SOURCE_LAYERS,
                "source_forwards": outer.SOURCE_FORWARDS,
                "prefix_tokens": outer.PREFIX_TOKENS,
                "forced_token_id": outer.FORCED_TOKEN,
            },
        },
        "required_dual_schema_resolution": {
            "metadata_range_authority_is_not_the_flat_runtime_range_map": True,
            "future_runtime_range_map_schema": outer.RUNTIME_RANGE_MAP_SCHEMA,
            "future_operator_accumulation_attestation": {
                "schema": outer.OPERATOR_ATTESTATION_SCHEMA,
                "status": outer.OPERATOR_ATTESTATION_STATUS,
            },
            "future_range_reader_exact_semantics_attestation": {
                "schema": outer.RANGE_READER_ATTESTATION_SCHEMA,
                "status": outer.RANGE_READER_ATTESTATION_STATUS,
            },
            "both_execution_attestations_must_bind_the_same_runtime_admission_and_source_payloads": True,
            "a_prepared_bridge_is_non_authorizing_and_cannot_substitute_for_either_execution_attestation": True,
        },
        "execution_authorized": False,
        "execution_boundary": {
            "source_tensor_payload_opened": False,
            "source_model_loaded_or_instantiated": False,
            "whole_source_model_resident": False,
            "gpu_metal_mps_or_other_accelerator_invoked": False,
            "server_started_or_contacted": False,
            "hcli_invoked": False,
            "lease_requested_issued_or_consumed": False,
            "child_process_started": False,
            "source_teacher_or_native_vector_written": False,
            "source_eviction_or_native_phase_performed": False,
        },
    }


def _source_lease(*, replayed: bool = False) -> dict[str, object]:
    return {
        "schema": outer.SOURCE_LEASE_SCHEMA,
        "status": outer.SOURCE_LEASE_STATUS,
        "one_shot_lifecycle": {
            "fresh_for_this_exact_launch": True,
            "prior_terminal_receipt": {"seal_sha256": SHA_C} if replayed else None,
            "automatic_retry_allowed": False,
            "new_capture_root": True,
            "existing_output_reuse_forbidden": True,
            "replay_or_relaunch_forbidden": True,
            "exact_launch_nonce": SHA_B,
        },
        "fresh_pre_child_safety": {
            "observed_immediately_before_child": True,
            "exclusive_clean_window": True,
            "no_source_or_native_model_body_resident_before_child": True,
            "swap_used_bytes": 0,
            "swapouts_pages_delta": 0,
            "reclaimable_bytes": 2_000_000,
            "minimum_reclaimable_bytes_required": 1_000_000,
        },
    }


def test_current_missing_bridge_and_lease_returns_a_sealed_pre_spawn_refusal(
    tmp_path: Path,
) -> None:
    child_path = _write_sealed(tmp_path / "child.json", _child_document())

    result = outer.build_outer_preflight(child_preflight_path=child_path)

    assert result["schema"] == outer.SCHEMA
    assert result["status"] == outer.REFUSED_STATUS
    assert result["spawn_permitted"] is False
    assert result["execution_boundary"] == {
        "source_tensor_payload_opened": False,
        "source_model_loaded_or_instantiated": False,
        "gpu_or_metal_invoked": False,
        "server_or_hcli_started_or_contacted": False,
        "lease_issued_or_consumed": False,
        "source_child_spawned": False,
        "native_child_spawned": False,
        "source_or_native_payload_written": False,
        "source_eviction_or_native_release_performed": False,
    }
    assert any(
        str(item).startswith("streamed_feasibility_or_exact_semantics_not_earned:")
        for item in result["blockers"]
    )
    assert (
        "sealed_dual_attestation_runtime_admission_bridge_absent" in result["blockers"]
    )
    assert "fresh_one_shot_source_lease_absent" in result["blockers"]
    verify(result, label="current outer refusal")


def test_valid_future_inputs_only_prepare_a_non_authorizing_reservation(
    tmp_path: Path,
) -> None:
    bridge_path = _write_sealed(tmp_path / "bridge.json", _bridge_document())
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    feasibility_path = _write_sealed(
        tmp_path / "feasibility.json", _feasibility_document()
    )
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    child_path = _write_sealed(
        tmp_path / "child.json",
        _child_document(
            bridge_path=bridge_path,
            bridge_document=bridge,
            feasibility_path=feasibility_path,
            feasibility_document=feasibility,
        ),
    )
    lease_path = _write_sealed(tmp_path / "lease.json", _source_lease())

    result = outer.build_outer_preflight(
        child_preflight_path=child_path,
        dual_bridge_path=bridge_path,
        source_lease_path=lease_path,
    )

    assert result["status"] == outer.PREPARED_STATUS
    assert result["spawn_permitted"] is False
    assert (
        result["reservation"]["one_source_child_and_one_native_child_maximum"] is True
    )
    assert (
        result["reservation"][
            "this_preflight_did_not_create_a_reservation_or_capture_root"
        ]
        is True
    )
    interface = result["future_lifecycle_interface"]
    assert interface["source_child_command"] == [
        "ascension_qwen30_streamed_source_teacher_child",
        "--source-root",
        "ABSOLUTE_CANONICAL_QWEN30_SOURCE_ROOT",
        "--runtime-admission",
        "ABSOLUTE_SEALED_RUNTIME_ADMISSION_JSON",
        "--dual-attestation-runtime-admission",
        "ABSOLUTE_SEALED_DUAL_BRIDGE_JSON",
        "--source-lease",
        "ABSOLUTE_SEALED_ONE_SHOT_SOURCE_LEASE_JSON",
        "--capture-dir",
        "NEW_ABSOLUTE_SOURCE_CHILD_CAPTURE_DIRECTORY",
    ]
    assert interface["source_child_receipt"]["source_payloads"] == list(
        outer.SOURCE_PAYLOADS
    )
    assert interface["separate_native_four_vector_child"][
        "requires_distinct_native_lease"
    ] == {
        "schema": outer.NATIVE_LEASE_SCHEMA,
        "status": outer.NATIVE_LEASE_STATUS,
    }
    verify(result, label="prepared outer grammar")


def test_refused_feasibility_cannot_be_promoted_by_a_valid_bridge_and_lease(
    tmp_path: Path,
) -> None:
    bridge_path = _write_sealed(tmp_path / "bridge.json", _bridge_document())
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    feasibility_path = _write_sealed(
        tmp_path / "refused-feasibility.json", _feasibility_document(prepared=False)
    )
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    child = _child_document(
        bridge_path=bridge_path,
        bridge_document=bridge,
        feasibility_path=feasibility_path,
        feasibility_document=feasibility,
    )
    child_path = _write_sealed(tmp_path / "child.json", child)
    lease_path = _write_sealed(tmp_path / "lease.json", _source_lease())

    result = outer.build_outer_preflight(
        child_preflight_path=child_path,
        dual_bridge_path=bridge_path,
        source_lease_path=lease_path,
    )

    assert result["status"] == outer.REFUSED_STATUS
    assert any(
        str(item).startswith("streamed_feasibility_or_exact_semantics_not_earned:")
        for item in result["blockers"]
    )
    assert result["spawn_permitted"] is False


def test_bridge_substitution_and_replayed_source_lease_hard_refuse(
    tmp_path: Path,
) -> None:
    expected_bridge_path = _write_sealed(
        tmp_path / "expected-bridge.json", _bridge_document()
    )
    expected_bridge = json.loads(expected_bridge_path.read_text(encoding="utf-8"))
    feasibility_path = _write_sealed(
        tmp_path / "feasibility.json", _feasibility_document()
    )
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    child_path = _write_sealed(
        tmp_path / "child.json",
        _child_document(
            bridge_path=expected_bridge_path,
            bridge_document=expected_bridge,
            feasibility_path=feasibility_path,
            feasibility_document=feasibility,
        ),
    )
    substituted_bridge_path = _write_sealed(
        tmp_path / "substituted-bridge.json", _bridge_document()
    )
    replayed_lease_path = _write_sealed(
        tmp_path / "replayed-lease.json", _source_lease(replayed=True)
    )

    result = outer.build_outer_preflight(
        child_preflight_path=child_path,
        dual_bridge_path=substituted_bridge_path,
        source_lease_path=replayed_lease_path,
    )

    assert result["status"] == outer.REFUSED_STATUS
    assert any(
        str(item).startswith("dual_attestation_runtime_admission_bridge_invalid:")
        for item in result["blockers"]
    )
    assert any(
        str(item).startswith("source_lease_invalid:") for item in result["blockers"]
    )
    assert result["spawn_permitted"] is False
    verify(result, label="substitution/replay refusal")


def test_future_bundle_adds_dual_attestation_and_close_cache_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bridge_path = _write_sealed(tmp_path / "bridge.json", _bridge_document())
    bridge = outer._sealed(bridge_path, label="fixture bridge")
    feasibility_path = _write_sealed(
        tmp_path / "feasibility.json", _feasibility_document()
    )
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    child_path = _write_sealed(
        tmp_path / "child.json",
        _child_document(
            bridge_path=bridge_path,
            bridge_document=bridge.document,
            feasibility_path=feasibility_path,
            feasibility_document=feasibility,
        ),
    )
    child = outer._validate_child_preflight(
        outer._sealed(child_path, label="fixture child")
    )
    source_terminal = {
        "dual_execution_attestations": {
            "dual_bridge": {"seal_sha256": bridge.seal_sha256},
            "runtime_range_admission": {
                "schema": outer.RUNTIME_ADMISSION_SCHEMA,
                "status": outer.RUNTIME_ADMISSION_STATUS,
                "seal_sha256": SHA_A,
            },
            "operator_accumulation": {
                "schema": outer.OPERATOR_ATTESTATION_SCHEMA,
                "status": outer.OPERATOR_ATTESTATION_STATUS,
                "seal_sha256": SHA_B,
            },
            "range_reader_exact_semantics": {
                "schema": outer.RANGE_READER_ATTESTATION_SCHEMA,
                "status": outer.RANGE_READER_ATTESTATION_STATUS,
                "seal_sha256": SHA_C,
            },
        },
        "streamed_execution": {
            "outer_reaped_child_before_terminal_receipt": True,
            "receipt_written_after_payload_fsyncs": True,
            "source_handles_closed_before_child_exit": True,
            "streamed_reader_cache_zeroed_before_child_exit": True,
        },
    }
    source_lease = _source_lease()
    captured: dict[str, object] = {}

    def fake_guarded(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "validated_order": ["source_streamed", "source_evicted", "native_lease"]
        }

    monkeypatch.setattr(
        outer.guarded_outer,
        "validate_future_source_then_evict_then_native",
        fake_guarded,
    )
    result = outer.validate_future_receipt_bundle(
        child=child,
        bridge=bridge,
        source_lease=source_lease,
        source_terminal=source_terminal,
        source_eviction={},
        native_lease={},
        metadata_range_authority={"authority": {}},
        raw_six_vector_contract={},
    )

    assert result["dual_attestation_bound"] is True
    assert result["guarded_source_then_evict_then_native"]["validated_order"] == [
        "source_streamed",
        "source_evicted",
        "native_lease",
    ]
    assert captured["maximum_window_bytes"] == outer.MAX_POSITIONED_READ_BYTES
    source_terminal["streamed_execution"]["source_handles_closed_before_child_exit"] = (
        False
    )
    with pytest.raises(
        outer.SourceTeacherOuterPreflightError, match="source_handles_closed"
    ):
        outer.validate_future_receipt_bundle(
            child=child,
            bridge=bridge,
            source_lease=source_lease,
            source_terminal=source_terminal,
            source_eviction={},
            native_lease={},
            metadata_range_authority={"authority": {}},
            raw_six_vector_contract={},
        )


def test_preflight_module_has_no_process_launcher_or_source_root_cli_surface() -> None:
    source = Path(outer.__file__).read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "subprocess." not in source
    parser = outer._parser()
    destinations = {action.dest for action in parser._actions}
    assert "source_root" not in destinations
    assert "source_command" not in destinations
