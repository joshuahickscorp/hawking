"""Pure sealed-fixture tests for the Q80 one-process memory preflight."""
from __future__ import annotations

from copy import deepcopy

import pytest

from lab.operators import ascension_qwen80_resident_memory_envelope_preflight as preflight
from lab.receipts import seal, verify


GIB = 1024**3


def _reseal(document: dict[str, object]) -> dict[str, object]:
    body = deepcopy(document)
    body.pop("seal_sha256", None)
    return seal(body)


def _artifact() -> dict[str, object]:
    return seal(
        {
            "schema": preflight.ARTIFACT_SCHEMA,
            "status": preflight.ARTIFACT_STATUS,
            "source": {
                "repository": preflight.SOURCE_REPOSITORY,
                "tensor_count": preflight.COMPLETE_TENSOR_COUNT,
            },
            "claim_boundary": {
                "complete_physical_tensor_coverage_is_true": True,
                "not_native_runtime_execution": True,
            },
        }
    )


def _admission(artifact: dict[str, object]) -> dict[str, object]:
    return seal(
        {
            "schema": preflight.ADMISSION_SCHEMA,
            "status": preflight.ADMISSION_STATUS,
            "complete_manifest": {
                "schema": preflight.ARTIFACT_SCHEMA,
                "seal_sha256": artifact["seal_sha256"],
                "status": preflight.ARTIFACT_STATUS,
            },
            "model": {
                "id": preflight.MODEL_ID,
                "key": preflight.MODEL_KEY,
                "repository": preflight.SOURCE_REPOSITORY,
                "revision": preflight.SOURCE_REVISION,
            },
            "native_loader": {
                "tensor_count": preflight.COMPLETE_TENSOR_COUNT,
                "tensor_payload_bytes": preflight.COMPLETE_PAYLOAD_BYTES,
            },
            "claim_boundary": {
                "admission_does_not_implement_or_claim_a_native_qwen_decoder": True,
            },
        }
    )


def _state_layout(
    artifact: dict[str, object], admission: dict[str, object]
) -> dict[str, object]:
    return seal(
        {
            "schema": preflight.STATE_LAYOUT_SCHEMA,
            "status": preflight.STATE_LAYOUT_STATUS,
            "actual_device_allocation_performed": False,
            "actual_device_state_parity_performed": False,
            "max_seq_len": preflight.MAX_SEQUENCE_LENGTH,
            "native_max_seq_len": preflight.MAX_SEQUENCE_LENGTH,
            "schedule_layers": 48,
            "deltanet_layers": preflight.DELTANET_LAYERS,
            "gqa_layers": preflight.GQA_LAYERS,
            "source_identity": {
                "model_id": preflight.MODEL_ID,
                "model_key": preflight.MODEL_KEY,
                "source_repository": preflight.SOURCE_REPOSITORY,
                "source_revision": preflight.SOURCE_REVISION,
                "manifest_seal_sha256": artifact["seal_sha256"],
                "admission_receipt_seal_sha256": admission["seal_sha256"],
            },
            "session_layout": {
                "active_total_bytes": 280_363_008,
                "rollback_total_bytes": 280_363_008,
                "per_session_total_bytes": 560_726_016,
            },
            "contract_checks": {
                "max_seq_len_within_native_bound": True,
                "exact_48_layer_schedule": True,
                "exact_36_deltanet_and_12_gqa_slots": True,
                "exact_offsets_and_capacities": True,
                "active_and_rollback_allocations_disjoint": True,
                "layer_slot_mapping_valid": True,
                "rollback_mirrors_active_layout": True,
                "no_device_allocation_or_runtime_claim": True,
            },
        }
    )


def _activation() -> dict[str, object]:
    return seal(
        {
            "schema": preflight.ACTIVATION_RESULT_SCHEMA,
            "status": preflight.ACTIVATION_REFUSED_STATUS,
            "target_topology": {
                "resident_q80_model_processes": 1,
                "logical_sessions": "many",
                "endpoint": {"host": preflight.QWEN80_HOST, "port": preflight.QWEN80_PORT},
                "qwen30_port_reuse_refused": preflight.QWEN30_PORT,
            },
            "automatic_launch_contract": {
                "processes_to_start": 1,
                "duplicate_model_process_start_prohibited": True,
                "gate_starts_no_process": True,
            },
            "claim_boundary": {
                "gate_started_no_server": True,
                "gate_bound_no_port": True,
                "gate_opened_no_model_artifact": True,
                "gate_executed_no_model_token": True,
                "gate_executed_no_hcli_request": True,
                "gate_measured_no_tps_or_tg": True,
                "eligibility_is_not_server_start_or_hcli_or_tps_evidence": True,
            },
        }
    )


def _topology() -> dict[str, object]:
    return seal(
        {
            "schema": preflight.TOPOLOGY_SCHEMA,
            "model_key": preflight.MODEL_KEY,
            "bind_host": preflight.QWEN80_HOST,
            "bind_port": preflight.QWEN80_PORT,
            "desired_q80_model_processes": 1,
            "existing_q80_model_processes": 0,
            "server_process_starts_per_activation": 1,
            "duplicate_q80_processes_prohibited": True,
            "single_resident_process_many_logical_sessions": True,
            "logical_session_state_isolated": True,
            "bounded_state_allocation": True,
            "state_rollback_allocation_required": True,
            "max_seq_len": preflight.MAX_SEQUENCE_LENGTH,
            "allowed_logical_session_counts": list(preflight.LOGICAL_SESSION_COUNTS),
            "maximum_logical_sessions": preflight.MAX_LOGICAL_SESSIONS,
        }
    )


def _snapshot(*, available: int = 64 * GIB, swap: int = 0) -> dict[str, object]:
    return seal(
        {
            "schema": preflight.HOST_SNAPSHOT_SCHEMA,
            "status": preflight.HOST_SNAPSHOT_STATUS,
            "measured_on_host": True,
            "runtime_or_server_started_by_snapshot": False,
            "resident_q80_model_processes": 0,
            "available_memory_bytes": available,
            "swap_used_bytes": swap,
            "co_resident_reservation_bytes": 16 * GIB,
            "safety_floor_bytes": preflight.MINIMUM_SAFETY_FLOOR_BYTES,
        }
    )


def _evidence(*, available: int = 64 * GIB, swap: int = 0) -> dict[str, object]:
    artifact = _artifact()
    admission = _admission(artifact)
    return {
        "schema": preflight.INPUT_SCHEMA,
        "artifact_manifest": artifact,
        "admission_receipt": admission,
        "device_state_layout_contract": _state_layout(artifact, admission),
        "activation_gate_contract": _activation(),
        "resident_topology_plan": _topology(),
        "host_memory_snapshot": _snapshot(available=available, swap=swap),
    }


def _profile(report: dict[str, object], logical_sessions: int) -> dict[str, object]:
    return next(
        profile
        for profile in report["logical_session_memory_profiles"]  # type: ignore[index]
        if profile["logical_sessions"] == logical_sessions
    )


def test_one_shared_model_body_and_bounded_sessions_have_exact_planned_totals() -> None:
    report = preflight.assess_preflight(_evidence())

    assert report["schema"] == preflight.RESULT_SCHEMA
    assert report["status"] == preflight.STATUS
    assert report["prepared"] is True
    assert report["memory_envelope_healthy"] is False
    assert report["actual_runtime_or_server_launch_performed"] is False
    assert report["all_profiles_static_snapshot_envelope_satisfied"] is True
    assert [
        profile["logical_sessions"] for profile in report["logical_session_memory_profiles"]  # type: ignore[index]
    ] == [1, 2, 4, 8, 16]

    allocations = report["planned_resident_allocations"]  # type: ignore[index]
    assert allocations["resident_weights_bytes"] == 11_207_187_116
    assert allocations["shared_runtime_buffers_bytes"] == 1_281_104
    per_session = allocations["per_logical_session_buffers"]
    assert per_session["deltanet_state_bytes"] == 158_072_832
    assert per_session["gqa_key_cache_bytes"] == 201_326_592
    assert per_session["gqa_value_cache_bytes"] == 201_326_592
    assert per_session["state_and_kv_bytes"] == 560_726_016
    assert per_session["session_control_bytes"] == 32_776
    assert per_session["per_session_total_bytes"] == 560_758_792

    one = _profile(report, 1)
    sixteen = _profile(report, 16)
    assert one["resident_q80_model_processes"] == 1
    assert one["q80_planned_resident_bytes"] == 11_769_227_012
    assert sixteen["resident_q80_model_processes"] == 1
    assert sixteen["q80_planned_resident_bytes"] == 20_180_608_892
    assert sixteen["static_snapshot_envelope_satisfied"] is True
    verify(report, label="result")


def test_swap_or_safety_failure_is_visible_and_cannot_be_promoted() -> None:
    swapped = preflight.assess_preflight(_evidence(swap=4_096))
    assert swapped["all_profiles_zero_swap_satisfied"] is False
    assert swapped["all_profiles_static_snapshot_envelope_satisfied"] is False
    assert _profile(swapped, 16)["zero_swap_satisfied"] is False
    assert swapped["memory_envelope_healthy"] is False

    too_small = preflight.assess_preflight(_evidence(available=24 * GIB))
    assert too_small["all_profiles_safety_floor_satisfied"] is False
    assert too_small["all_profiles_static_snapshot_envelope_satisfied"] is False
    assert _profile(too_small, 16)["safety_floor_satisfied"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("desired_q80_model_processes", 2),
        ("existing_q80_model_processes", 1),
        ("server_process_starts_per_activation", 2),
        ("bind_port", preflight.QWEN30_PORT),
        ("bounded_state_allocation", False),
        ("max_seq_len", preflight.MAX_SEQUENCE_LENGTH + 1),
        ("allowed_logical_session_counts", [1, 2, 4, 8, 32]),
    ],
)
def test_multiple_process_port_reuse_and_unbounded_state_plans_are_refused(
    field: str, value: object
) -> None:
    evidence = _evidence()
    topology = deepcopy(evidence["resident_topology_plan"])
    topology[field] = value
    evidence["resident_topology_plan"] = _reseal(topology)

    with pytest.raises(preflight.ResidentMemoryPreflightError):
        preflight.assess_preflight(evidence)


def test_synthetic_or_unsealed_evidence_is_refused() -> None:
    evidence = _evidence()
    state = deepcopy(evidence["device_state_layout_contract"])
    state["synthetic_input"] = True
    evidence["device_state_layout_contract"] = _reseal(state)
    with pytest.raises(preflight.ResidentMemoryPreflightError, match="synthetic/fixture"):
        preflight.assess_preflight(evidence)

    evidence = _evidence()
    artifact = deepcopy(evidence["artifact_manifest"])
    artifact.pop("seal_sha256")
    evidence["artifact_manifest"] = artifact
    with pytest.raises(preflight.ResidentMemoryPreflightError, match="invalid sealed JSON"):
        preflight.assess_preflight(evidence)


def test_artifact_admission_and_state_bindings_cannot_drift() -> None:
    evidence = _evidence()
    state = deepcopy(evidence["device_state_layout_contract"])
    state["source_identity"]["manifest_seal_sha256"] = "0" * 64
    evidence["device_state_layout_contract"] = _reseal(state)
    with pytest.raises(preflight.ResidentMemoryPreflightError, match="manifest_seal_sha256"):
        preflight.assess_preflight(evidence)

    evidence = _evidence()
    admission = deepcopy(evidence["admission_receipt"])
    admission["complete_manifest"]["seal_sha256"] = "1" * 64
    evidence["admission_receipt"] = _reseal(admission)
    with pytest.raises(preflight.ResidentMemoryPreflightError, match="complete_manifest.seal_sha256"):
        preflight.assess_preflight(evidence)
