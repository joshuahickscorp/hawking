#!/usr/bin/env python3
"""Fail-closed Qwen80 one-resident-process memory-envelope preflight.

This is an evidence-only planner.  It reads only the explicitly supplied,
sealed JSON documents and computes a bounded allocation plan for one resident
Qwen80 model process with 1, 2, 4, 8, or 16 *logical* sessions.  It does not
open the Gravity payload, scan an artifact directory, probe host memory,
allocate Metal memory, acquire a lease, bind a port, start a watcher/server,
or execute a token/HCLI/benchmark.

The shared model weights and decoder scratch are counted once.  The exact
active-plus-rollback DeltaNet/GQA state and tiny caller-owned token control
buffers are counted once per logical session.  Any request for cloned model
processes, Q30-port reuse, unbounded state, or synthetic evidence is refused
before a report is produced.

The resulting receipt is deliberately PREPARED/INCOMPLETE.  It is not a
resident-RSS measurement and cannot satisfy the activation gate's later
EARNED memory-receipt requirement.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.receipts import SealIntegrityError, seal, verify


INPUT_SCHEMA = "hawking.ascension.qwen80_resident_memory_envelope_preflight_input.v1"
RESULT_SCHEMA = "hawking.ascension.qwen80_resident_memory_envelope_receipt.v1"
STATUS = (
    "PREPARED_INCOMPLETE_QWEN80_ONE_RESIDENT_MANY_LOGICAL_SESSIONS_"
    "MEMORY_ENVELOPE_PREFLIGHT_NO_RUNTIME_SERVER_OR_TPS"
)

ARTIFACT_SCHEMA = "hawking.ascension.qwen80_complete_binary_gravity.v1"
ARTIFACT_STATUS = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED"
ADMISSION_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1"
ADMISSION_STATUS = "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
STATE_LAYOUT_SCHEMA = "hawking.ascension.qwen80_device_state_buffer_layout_contract.v1"
STATE_LAYOUT_STATUS = (
    "NOT_READY_NO_DEVICE_ALLOCATION_NO_STATE_PARITY_NO_ROLLBACK_CAPTURE_"
    "QWEN80_PER_SESSION_BUFFER_LAYOUT_CONTRACT"
)
ACTIVATION_RESULT_SCHEMA = "hawking.ascension.qwen80_resident_server_activation_result.v1"
ACTIVATION_REFUSED_STATUS = "REFUSED_QWEN80_ONE_RESIDENT_SERVER_ACTIVATION_NOT_READY_NO_SERVER"
ACTIVATION_ELIGIBLE_STATUS = (
    "ELIGIBLE_QWEN80_ONE_RESIDENT_SERVER_AUTOMATIC_LAUNCH_PRECONDITION_ONLY"
)
TOPOLOGY_SCHEMA = "hawking.ascension.qwen80_resident_memory_topology_plan.v1"
HOST_SNAPSHOT_SCHEMA = "hawking.ascension.qwen80_host_memory_snapshot.v1"
HOST_SNAPSHOT_STATUS = "SEALED_HOST_MEMORY_SNAPSHOT_PRELAUNCH_ONLY"

MODEL_ID = "Qwen3-Coder-Next-80B"
MODEL_KEY = "qwen80"
SOURCE_REPOSITORY = "Qwen/Qwen3-Coder-Next"
SOURCE_REVISION = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb"
COMPLETE_TENSOR_COUNT = 74_391
COMPLETE_PAYLOAD_BYTES = 11_207_187_116

QWEN30_PORT = 18_430
QWEN80_HOST = "127.0.0.1"
QWEN80_PORT = 18_480
LOGICAL_SESSION_COUNTS = (1, 2, 4, 8, 16)
MAX_LOGICAL_SESSIONS = LOGICAL_SESSION_COUNTS[-1]
MAX_SEQUENCE_LENGTH = 4_096
MINIMUM_SAFETY_FLOOR_BYTES = 8 * 1024**3

F32_BYTES = 4
U32_BYTES = 4
HIDDEN_SIZE = 2_048
VOCAB_SIZE = 151_936
EXPERTS = 512
TOP_K = 10
DELTANET_LAYERS = 36
GQA_LAYERS = 12
DELTANET_CONV_CHANNELS = 8_192
DELTANET_CONV_HISTORY = 3
DELTANET_VALUE_HEADS = 32
DELTANET_KEY_DIM = 128
DELTANET_VALUE_DIM = 128
GQA_KV_HEADS = 2
GQA_HEAD_DIM = 256


class ResidentMemoryPreflightError(ValueError):
    """The supplied static evidence cannot safely define a Q80 memory plan."""


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResidentMemoryPreflightError(f"{label} must be an object")
    return dict(value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ResidentMemoryPreflightError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ResidentMemoryPreflightError(f"{label} must be an integer >= {minimum}")
    return value


def _expect(document: Mapping[str, Any], key: str, expected: object, label: str) -> None:
    observed = document.get(key)
    if observed != expected:
        raise ResidentMemoryPreflightError(
            f"{label}.{key}: expected {expected!r}, observed {observed!r}"
        )


def _expect_true(document: Mapping[str, Any], key: str, label: str) -> None:
    if document.get(key) is not True:
        raise ResidentMemoryPreflightError(f"{label}.{key} must be true")


def _expect_false(document: Mapping[str, Any], key: str, label: str) -> None:
    if document.get(key) is not False:
        raise ResidentMemoryPreflightError(f"{label}.{key} must be false")


def _sealed(value: object, label: str) -> dict[str, Any]:
    document = _mapping(value, label)
    try:
        return verify(document, label=label)
    except SealIntegrityError as exc:
        raise ResidentMemoryPreflightError(f"{label}: invalid sealed JSON: {exc}") from exc


def _reject_synthetic(value: object, label: str) -> None:
    """Refuse any document that labels itself fixture/synthetic at any depth."""
    forbidden_true = {
        "synthetic",
        "synthetic_input",
        "synthetic_receipt",
        "fixture_only",
        "fixture_receipt",
        "shadow_model_used",
    }
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in forbidden_true and nested is True:
                raise ResidentMemoryPreflightError(
                    f"{label}.{key}: synthetic/fixture evidence is prohibited"
                )
            _reject_synthetic(nested, f"{label}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_synthetic(nested, f"{label}[{index}]")


def _validate_artifact_manifest(value: object) -> dict[str, Any]:
    document = _sealed(value, "artifact_manifest")
    _reject_synthetic(document, "artifact_manifest")
    _expect(document, "schema", ARTIFACT_SCHEMA, "artifact_manifest")
    _expect(document, "status", ARTIFACT_STATUS, "artifact_manifest")
    source = _mapping(document.get("source"), "artifact_manifest.source")
    _expect(source, "repository", SOURCE_REPOSITORY, "artifact_manifest.source")
    _expect(source, "tensor_count", COMPLETE_TENSOR_COUNT, "artifact_manifest.source")
    claim_boundary = _mapping(
        document.get("claim_boundary"), "artifact_manifest.claim_boundary"
    )
    _expect_true(
        claim_boundary,
        "complete_physical_tensor_coverage_is_true",
        "artifact_manifest.claim_boundary",
    )
    _expect_true(
        claim_boundary,
        "not_native_runtime_execution",
        "artifact_manifest.claim_boundary",
    )
    return document


def _validate_admission(value: object, artifact_manifest: Mapping[str, Any]) -> dict[str, Any]:
    document = _sealed(value, "admission_receipt")
    _reject_synthetic(document, "admission_receipt")
    _expect(document, "schema", ADMISSION_SCHEMA, "admission_receipt")
    _expect(document, "status", ADMISSION_STATUS, "admission_receipt")
    manifest = _mapping(document.get("complete_manifest"), "admission_receipt.complete_manifest")
    _expect(manifest, "schema", ARTIFACT_SCHEMA, "admission_receipt.complete_manifest")
    _expect(
        manifest,
        "seal_sha256",
        artifact_manifest["seal_sha256"],
        "admission_receipt.complete_manifest",
    )
    _expect(
        manifest,
        "status",
        ARTIFACT_STATUS,
        "admission_receipt.complete_manifest",
    )
    model = _mapping(document.get("model"), "admission_receipt.model")
    for key, expected in (
        ("id", MODEL_ID),
        ("key", MODEL_KEY),
        ("repository", SOURCE_REPOSITORY),
        ("revision", SOURCE_REVISION),
    ):
        _expect(model, key, expected, "admission_receipt.model")
    native_loader = _mapping(document.get("native_loader"), "admission_receipt.native_loader")
    _expect(native_loader, "tensor_count", COMPLETE_TENSOR_COUNT, "admission_receipt.native_loader")
    _expect(
        native_loader,
        "tensor_payload_bytes",
        COMPLETE_PAYLOAD_BYTES,
        "admission_receipt.native_loader",
    )
    claim_boundary = _mapping(
        document.get("claim_boundary"), "admission_receipt.claim_boundary"
    )
    _expect_true(
        claim_boundary,
        "admission_does_not_implement_or_claim_a_native_qwen_decoder",
        "admission_receipt.claim_boundary",
    )
    return document


def _validate_state_layout(
    value: object, artifact_manifest: Mapping[str, Any], admission_receipt: Mapping[str, Any]
) -> dict[str, Any]:
    document = _sealed(value, "device_state_layout_contract")
    _reject_synthetic(document, "device_state_layout_contract")
    _expect(document, "schema", STATE_LAYOUT_SCHEMA, "device_state_layout_contract")
    _expect(document, "status", STATE_LAYOUT_STATUS, "device_state_layout_contract")
    _expect(document, "max_seq_len", MAX_SEQUENCE_LENGTH, "device_state_layout_contract")
    _expect(
        document,
        "native_max_seq_len",
        MAX_SEQUENCE_LENGTH,
        "device_state_layout_contract",
    )
    _expect(document, "schedule_layers", 48, "device_state_layout_contract")
    _expect(document, "deltanet_layers", DELTANET_LAYERS, "device_state_layout_contract")
    _expect(document, "gqa_layers", GQA_LAYERS, "device_state_layout_contract")
    _expect_false(
        document,
        "actual_device_allocation_performed",
        "device_state_layout_contract",
    )
    _expect_false(
        document,
        "actual_device_state_parity_performed",
        "device_state_layout_contract",
    )
    identity = _mapping(
        document.get("source_identity"), "device_state_layout_contract.source_identity"
    )
    for key, expected in (
        ("model_id", MODEL_ID),
        ("model_key", MODEL_KEY),
        ("source_repository", SOURCE_REPOSITORY),
        ("source_revision", SOURCE_REVISION),
        ("manifest_seal_sha256", artifact_manifest["seal_sha256"]),
        ("admission_receipt_seal_sha256", admission_receipt["seal_sha256"]),
    ):
        _expect(identity, key, expected, "device_state_layout_contract.source_identity")
    layout = _mapping(
        document.get("session_layout"), "device_state_layout_contract.session_layout"
    )
    expected_totals = {
        "active_total_bytes": 280_363_008,
        "rollback_total_bytes": 280_363_008,
        "per_session_total_bytes": 560_726_016,
    }
    for key, expected in expected_totals.items():
        _expect(layout, key, expected, "device_state_layout_contract.session_layout")
    checks = _mapping(
        document.get("contract_checks"), "device_state_layout_contract.contract_checks"
    )
    for key in (
        "max_seq_len_within_native_bound",
        "exact_48_layer_schedule",
        "exact_36_deltanet_and_12_gqa_slots",
        "exact_offsets_and_capacities",
        "active_and_rollback_allocations_disjoint",
        "layer_slot_mapping_valid",
        "rollback_mirrors_active_layout",
        "no_device_allocation_or_runtime_claim",
    ):
        _expect_true(checks, key, "device_state_layout_contract.contract_checks")
    return document


def _validate_activation_contract(value: object) -> dict[str, Any]:
    document = _sealed(value, "activation_gate_contract")
    _reject_synthetic(document, "activation_gate_contract")
    _expect(document, "schema", ACTIVATION_RESULT_SCHEMA, "activation_gate_contract")
    if document.get("status") not in {
        ACTIVATION_REFUSED_STATUS,
        ACTIVATION_ELIGIBLE_STATUS,
    }:
        raise ResidentMemoryPreflightError(
            "activation_gate_contract.status is not a recognized activation-gate result"
        )
    topology = _mapping(document.get("target_topology"), "activation_gate_contract.target_topology")
    _expect(topology, "resident_q80_model_processes", 1, "activation_gate_contract.target_topology")
    _expect(topology, "logical_sessions", "many", "activation_gate_contract.target_topology")
    endpoint = _mapping(topology.get("endpoint"), "activation_gate_contract.target_topology.endpoint")
    _expect(endpoint, "host", QWEN80_HOST, "activation_gate_contract.target_topology.endpoint")
    _expect(endpoint, "port", QWEN80_PORT, "activation_gate_contract.target_topology.endpoint")
    _expect(
        topology,
        "qwen30_port_reuse_refused",
        QWEN30_PORT,
        "activation_gate_contract.target_topology",
    )
    launch = _mapping(
        document.get("automatic_launch_contract"), "activation_gate_contract.automatic_launch_contract"
    )
    _expect(launch, "processes_to_start", 1, "activation_gate_contract.automatic_launch_contract")
    _expect_true(
        launch,
        "duplicate_model_process_start_prohibited",
        "activation_gate_contract.automatic_launch_contract",
    )
    _expect_true(
        launch,
        "gate_starts_no_process",
        "activation_gate_contract.automatic_launch_contract",
    )
    claim_boundary = _mapping(
        document.get("claim_boundary"), "activation_gate_contract.claim_boundary"
    )
    for key in (
        "gate_started_no_server",
        "gate_bound_no_port",
        "gate_opened_no_model_artifact",
        "gate_executed_no_model_token",
        "gate_executed_no_hcli_request",
        "gate_measured_no_tps_or_tg",
        "eligibility_is_not_server_start_or_hcli_or_tps_evidence",
    ):
        _expect_true(claim_boundary, key, "activation_gate_contract.claim_boundary")
    return document


def _validate_topology(value: object) -> dict[str, Any]:
    document = _sealed(value, "resident_topology_plan")
    _reject_synthetic(document, "resident_topology_plan")
    _expect(document, "schema", TOPOLOGY_SCHEMA, "resident_topology_plan")
    _expect(document, "model_key", MODEL_KEY, "resident_topology_plan")
    _expect(document, "bind_host", QWEN80_HOST, "resident_topology_plan")
    _expect(document, "bind_port", QWEN80_PORT, "resident_topology_plan")
    _expect(document, "desired_q80_model_processes", 1, "resident_topology_plan")
    _expect(document, "existing_q80_model_processes", 0, "resident_topology_plan")
    _expect(document, "server_process_starts_per_activation", 1, "resident_topology_plan")
    _expect_true(document, "duplicate_q80_processes_prohibited", "resident_topology_plan")
    _expect_true(document, "single_resident_process_many_logical_sessions", "resident_topology_plan")
    _expect_true(document, "logical_session_state_isolated", "resident_topology_plan")
    _expect_true(document, "bounded_state_allocation", "resident_topology_plan")
    _expect_true(document, "state_rollback_allocation_required", "resident_topology_plan")
    _expect(document, "max_seq_len", MAX_SEQUENCE_LENGTH, "resident_topology_plan")
    _expect(
        document,
        "allowed_logical_session_counts",
        list(LOGICAL_SESSION_COUNTS),
        "resident_topology_plan",
    )
    _expect(
        document,
        "maximum_logical_sessions",
        MAX_LOGICAL_SESSIONS,
        "resident_topology_plan",
    )
    if document.get("bind_port") == QWEN30_PORT:
        raise ResidentMemoryPreflightError(
            "resident_topology_plan.bind_port must not reuse Q30's port"
        )
    return document


def _validate_host_snapshot(value: object) -> dict[str, Any]:
    document = _sealed(value, "host_memory_snapshot")
    _reject_synthetic(document, "host_memory_snapshot")
    _expect(document, "schema", HOST_SNAPSHOT_SCHEMA, "host_memory_snapshot")
    _expect(document, "status", HOST_SNAPSHOT_STATUS, "host_memory_snapshot")
    _expect_true(document, "measured_on_host", "host_memory_snapshot")
    _expect_false(document, "runtime_or_server_started_by_snapshot", "host_memory_snapshot")
    _expect(document, "resident_q80_model_processes", 0, "host_memory_snapshot")
    _integer(document.get("available_memory_bytes"), "host_memory_snapshot.available_memory_bytes", minimum=1)
    _integer(document.get("swap_used_bytes"), "host_memory_snapshot.swap_used_bytes")
    _integer(
        document.get("co_resident_reservation_bytes"),
        "host_memory_snapshot.co_resident_reservation_bytes",
        minimum=1,
    )
    _integer(
        document.get("safety_floor_bytes"),
        "host_memory_snapshot.safety_floor_bytes",
        minimum=MINIMUM_SAFETY_FLOOR_BYTES,
    )
    return document


def _f32_bytes(elements: int) -> int:
    return elements * F32_BYTES


def _u32_bytes(elements: int) -> int:
    return elements * U32_BYTES


def _shared_runtime_buffers() -> list[dict[str, Any]]:
    """The one-process scratch plan; these allocations never scale by sessions."""
    entries: tuple[tuple[str, int, str], ...] = (
        ("decode_hidden_active", _f32_bytes(HIDDEN_SIZE), "f32[2048]"),
        ("decode_hidden_rollback", _f32_bytes(HIDDEN_SIZE), "f32[2048]"),
        ("post_attention_norm", _f32_bytes(HIDDEN_SIZE), "f32[2048]"),
        ("routed_expert_sum", _f32_bytes(HIDDEN_SIZE), "f32[2048]"),
        ("shared_expert_output", _f32_bytes(HIDDEN_SIZE), "f32[2048]"),
        ("second_residual", _f32_bytes(HIDDEN_SIZE), "f32[2048]"),
        ("final_norm", _f32_bytes(HIDDEN_SIZE), "f32[2048]"),
        ("router_logits", _f32_bytes(EXPERTS), "f32[512]"),
        ("topk_expert_ids", _u32_bytes(TOP_K), "u32[10]"),
        ("topk_weights", _f32_bytes(TOP_K), "f32[10]"),
        ("expert_gate", _f32_bytes(512), "f32[512]"),
        ("expert_up", _f32_bytes(512), "f32[512]"),
        ("expert_activation", _f32_bytes(512), "f32[512]"),
        ("terminal_logits", _f32_bytes(VOCAB_SIZE), "f32[151936]"),
        ("terminal_logits_rollback", _f32_bytes(VOCAB_SIZE), "f32[151936]"),
    )
    return [
        {"allocation_id": allocation_id, "bytes": size, "shape": shape}
        for allocation_id, size, shape in entries
    ]


def _per_session_buffers() -> dict[str, Any]:
    conv_active = _f32_bytes(
        DELTANET_LAYERS * DELTANET_CONV_CHANNELS * DELTANET_CONV_HISTORY
    )
    recurrent_active = _f32_bytes(
        DELTANET_LAYERS
        * DELTANET_VALUE_HEADS
        * DELTANET_KEY_DIM
        * DELTANET_VALUE_DIM
    )
    gqa_cache_active = _f32_bytes(
        GQA_LAYERS * MAX_SEQUENCE_LENGTH * GQA_KV_HEADS * GQA_HEAD_DIM
    )
    buffers = [
        {
            "allocation_id": "deltanet_conv_active_and_rollback",
            "bytes": 2 * conv_active,
            "shape": "2 * f32[36,8192,3]",
        },
        {
            "allocation_id": "deltanet_recurrent_active_and_rollback",
            "bytes": 2 * recurrent_active,
            "shape": "2 * f32[36,32,128,128]",
        },
        {
            "allocation_id": "gqa_key_active_and_rollback",
            "bytes": 2 * gqa_cache_active,
            "shape": "2 * f32[12,4096,2,256]",
        },
        {
            "allocation_id": "gqa_value_active_and_rollback",
            "bytes": 2 * gqa_cache_active,
            "shape": "2 * f32[12,4096,2,256]",
        },
        {
            "allocation_id": "token_history_active_and_rollback",
            "bytes": 2 * _u32_bytes(MAX_SEQUENCE_LENGTH),
            "shape": "2 * u32[4096]",
        },
        {
            "allocation_id": "sampled_feedback_active_and_rollback",
            "bytes": 2 * _u32_bytes(1),
            "shape": "2 * u32[1]",
        },
    ]
    deltanet_state_bytes = 2 * (conv_active + recurrent_active)
    gqa_key_cache_bytes = 2 * gqa_cache_active
    gqa_value_cache_bytes = 2 * gqa_cache_active
    state_and_kv_bytes = deltanet_state_bytes + gqa_key_cache_bytes + gqa_value_cache_bytes
    session_control_bytes = sum(
        buffer["bytes"]
        for buffer in buffers
        if buffer["allocation_id"].startswith(("token_history", "sampled_feedback"))
    )
    return {
        "buffers": buffers,
        "deltanet_state_bytes": deltanet_state_bytes,
        "gqa_key_cache_bytes": gqa_key_cache_bytes,
        "gqa_value_cache_bytes": gqa_value_cache_bytes,
        "state_and_kv_bytes": state_and_kv_bytes,
        "session_control_bytes": session_control_bytes,
        "per_session_total_bytes": state_and_kv_bytes + session_control_bytes,
    }


def _validate_derived_geometry(state_layout: Mapping[str, Any], per_session: Mapping[str, Any]) -> None:
    layout = _mapping(state_layout.get("session_layout"), "device_state_layout_contract.session_layout")
    if per_session["state_and_kv_bytes"] != layout["per_session_total_bytes"]:
        raise ResidentMemoryPreflightError(
            "derived DeltaNet/GQA state does not equal the sealed device-state layout total"
        )


def _session_profiles(
    *,
    weights_bytes: int,
    shared_runtime_bytes: int,
    per_session: Mapping[str, Any],
    host_snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    available = _integer(host_snapshot["available_memory_bytes"], "host_memory_snapshot.available_memory_bytes", minimum=1)
    swap_used = _integer(host_snapshot["swap_used_bytes"], "host_memory_snapshot.swap_used_bytes")
    co_resident = _integer(
        host_snapshot["co_resident_reservation_bytes"],
        "host_memory_snapshot.co_resident_reservation_bytes",
        minimum=1,
    )
    safety_floor = _integer(
        host_snapshot["safety_floor_bytes"],
        "host_memory_snapshot.safety_floor_bytes",
        minimum=MINIMUM_SAFETY_FLOOR_BYTES,
    )
    profiles: list[dict[str, Any]] = []
    for sessions in LOGICAL_SESSION_COUNTS:
        session_total = sessions * _integer(
            per_session["per_session_total_bytes"], "derived.per_session_total_bytes", minimum=1
        )
        resident_total = weights_bytes + shared_runtime_bytes + session_total
        required_available = resident_total + co_resident + safety_floor
        profiles.append(
            {
                "logical_sessions": sessions,
                "resident_q80_model_processes": 1,
                "resident_weights_bytes": weights_bytes,
                "shared_runtime_buffers_bytes": shared_runtime_bytes,
                "deltanet_state_bytes": sessions * _integer(per_session["deltanet_state_bytes"], "derived.deltanet_state_bytes", minimum=1),
                "gqa_key_cache_bytes": sessions * _integer(per_session["gqa_key_cache_bytes"], "derived.gqa_key_cache_bytes", minimum=1),
                "gqa_value_cache_bytes": sessions * _integer(per_session["gqa_value_cache_bytes"], "derived.gqa_value_cache_bytes", minimum=1),
                "session_control_buffers_bytes": sessions * _integer(per_session["session_control_bytes"], "derived.session_control_bytes", minimum=1),
                "q80_planned_resident_bytes": resident_total,
                "co_resident_reservation_bytes": co_resident,
                "safety_floor_bytes": safety_floor,
                "minimum_required_available_bytes": required_available,
                "available_memory_bytes_from_sealed_snapshot": available,
                "swap_used_bytes_from_sealed_snapshot": swap_used,
                "zero_swap_required": True,
                "zero_swap_satisfied": swap_used == 0,
                "safety_floor_satisfied": available >= required_available,
                "static_snapshot_envelope_satisfied": swap_used == 0 and available >= required_available,
            }
        )
    return profiles


def assess_preflight(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate sealed inputs and return a static, never-ready memory plan."""
    document = _mapping(evidence, "evidence")
    _expect(document, "schema", INPUT_SCHEMA, "evidence")
    artifact = _validate_artifact_manifest(document.get("artifact_manifest"))
    admission = _validate_admission(document.get("admission_receipt"), artifact)
    state_layout = _validate_state_layout(
        document.get("device_state_layout_contract"), artifact, admission
    )
    activation = _validate_activation_contract(document.get("activation_gate_contract"))
    topology = _validate_topology(document.get("resident_topology_plan"))
    host_snapshot = _validate_host_snapshot(document.get("host_memory_snapshot"))

    shared_runtime_buffers = _shared_runtime_buffers()
    shared_runtime_bytes = sum(buffer["bytes"] for buffer in shared_runtime_buffers)
    per_session = _per_session_buffers()
    _validate_derived_geometry(state_layout, per_session)
    profiles = _session_profiles(
        weights_bytes=COMPLETE_PAYLOAD_BYTES,
        shared_runtime_bytes=shared_runtime_bytes,
        per_session=per_session,
        host_snapshot=host_snapshot,
    )
    all_profiles_fit = all(profile["static_snapshot_envelope_satisfied"] for profile in profiles)
    all_profiles_zero_swap = all(profile["zero_swap_satisfied"] for profile in profiles)
    all_profiles_safety = all(profile["safety_floor_satisfied"] for profile in profiles)

    report: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": STATUS,
        "prepared": True,
        "complete_decoder_readiness_earned": False,
        "real_gravity_server_launch_precondition_satisfied": False,
        "memory_envelope_healthy": False,
        "actual_resident_q80_rss_measured": False,
        "actual_device_allocation_performed": False,
        "actual_host_memory_probe_performed_by_this_preflight": False,
        "actual_runtime_or_server_launch_performed": False,
        "actual_hcli_or_tps_or_tg_measurement_performed": False,
        "one_q80_process_envelope": True,
        "co_resident_envelope_accounted_for": True,
        "input_document_seals": {
            "artifact_manifest_seal_sha256": artifact["seal_sha256"],
            "admission_receipt_seal_sha256": admission["seal_sha256"],
            "device_state_layout_contract_seal_sha256": state_layout["seal_sha256"],
            "activation_gate_contract_seal_sha256": activation["seal_sha256"],
            "resident_topology_plan_seal_sha256": topology["seal_sha256"],
            "host_memory_snapshot_seal_sha256": host_snapshot["seal_sha256"],
        },
        "source_artifact_binding": {
            "model_id": MODEL_ID,
            "model_key": MODEL_KEY,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "complete_tensor_count": COMPLETE_TENSOR_COUNT,
            "resident_weight_bytes": COMPLETE_PAYLOAD_BYTES,
            "artifact_manifest_seal_sha256": artifact["seal_sha256"],
            "admission_receipt_seal_sha256": admission["seal_sha256"],
        },
        "fixed_one_process_topology": {
            "resident_q80_model_processes": 1,
            "logical_sessions_supported": list(LOGICAL_SESSION_COUNTS),
            "maximum_logical_sessions": MAX_LOGICAL_SESSIONS,
            "endpoint": {"host": QWEN80_HOST, "port": QWEN80_PORT},
            "qwen30_port_reuse_refused": QWEN30_PORT,
            "bounded_max_seq_len": MAX_SEQUENCE_LENGTH,
            "state_rollback_required": True,
        },
        "planned_resident_allocations": {
            "resident_weights_bytes": COMPLETE_PAYLOAD_BYTES,
            "shared_runtime_buffers": shared_runtime_buffers,
            "shared_runtime_buffers_bytes": shared_runtime_bytes,
            "per_logical_session_buffers": per_session,
        },
        "logical_session_memory_profiles": profiles,
        "zero_swap_required_for_every_profile": True,
        "all_profiles_zero_swap_satisfied": all_profiles_zero_swap,
        "safety_floor_required_for_every_profile": True,
        "all_profiles_safety_floor_satisfied": all_profiles_safety,
        "all_profiles_static_snapshot_envelope_satisfied": all_profiles_fit,
        "activation_gate_status_observed": activation["status"],
        "required_before_activation_gate_memory_receipt_can_be_earned": [
            "Measure a real one-process Q80 resident RSS after a truthful complete decoder exists; a static payload plan is not an RSS measurement.",
            "Re-measure available memory and zero swap immediately before the controlled Q80 launch while accounting for the real Q30 resident process and OS reserve.",
            "Prove actual bounded per-session device-state/KV allocation, isolation, rollback, and multi-session behavior under the complete decoder.",
            "Emit the separate EARNED_QWEN80_RESIDENT_MEMORY_ENVELOPE_HEALTHY receipt only after the full decoder, runtime, terminal, HCLI, and rollback gates independently pass.",
        ],
        "claim_boundary": {
            "preflight_opened_no_model_artifact": True,
            "preflight_scanned_no_artifact_directory": True,
            "preflight_probed_no_host_memory": True,
            "preflight_allocated_no_metal_memory": True,
            "preflight_acquired_no_gpu_lease": True,
            "preflight_started_no_watcher_or_server": True,
            "preflight_bound_no_port": True,
            "preflight_executed_no_token_hcli_tps_or_tg": True,
            "static_plan_is_not_a_resident_memory_health_receipt": True,
        },
    }
    return seal(report)


def _regular_absolute_file(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ResidentMemoryPreflightError(f"{label} must be an absolute path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ResidentMemoryPreflightError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ResidentMemoryPreflightError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _read_input(path: Path) -> dict[str, Any]:
    clean = _regular_absolute_file(path, "--input")
    try:
        value = json.loads(clean.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResidentMemoryPreflightError(f"cannot read input JSON {clean}: {exc}") from exc
    return _mapping(value, "input JSON")


def _write_new_report(path: Path, report: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise ResidentMemoryPreflightError("--out must be an absolute path")
    if not path.parent.is_dir():
        raise ResidentMemoryPreflightError("--out parent directory must already exist")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing preflight report: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(report), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing preflight report: {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="absolute sealed evidence JSON")
    parser.add_argument("--out", type=Path, required=True, help="new absolute report path")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        report = assess_preflight(_read_input(arguments.input))
        _write_new_report(arguments.out, report)
    except (ResidentMemoryPreflightError, FileExistsError) as exc:
        print(f"ascension_qwen80_resident_memory_envelope_preflight: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
