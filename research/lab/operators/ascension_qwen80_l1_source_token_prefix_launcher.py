#!/usr/bin/env python3
"""CPU-only authority preflight for a future Qwen80 L0→L1 joint component.

The captured L0 handoff is valuable sealed provenance, but it is *not* a
portable Metal allocation.  This module therefore refuses a design that hands
the terminated L0 process's ``PinnedBuffer`` or state arena to a new process.
It accepts the real L0 outer/inner capture ABI only as baseline evidence and
prepares a future child that must re-encode L0 (23 dispatches) and append the
Layer-1 DeltaNet prefix (9 dispatches) in one new runtime and one TCB.

This module deliberately has no subprocess, lease, Metal, GPU, watcher,
server, benchmark, or output-file code path.  Its only result is a sealed
``PREPARED`` or ``REFUSED`` CPU-only document.  It requires the earned
post-capture assessment and its sealed binding wrapper, but neither record
authorizes execution: a future physical child still needs its own fresh
same-runtime/same-TCB capture authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.receipts import SealIntegrityError, seal, verify


INPUT_SCHEMA = "hawking.ascension.qwen80_l1_source_token_prefix_launcher_input.v4"
INPUT_STATUS = "SUBMITTED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_SLOT1_PREFIX_CPU_ONLY_PREFLIGHT"
RESULT_SCHEMA = "hawking.ascension.qwen80_l1_source_token_prefix_launcher_result.v4"
PREPARED_STATUS = (
    "PREFLIGHTED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_SLOT1_PREFIX_"
    "SAME_RUNTIME_COMPONENT_ONLY_CHILD_NOT_STARTED"
)
REFUSED_STATUS = (
    "REFUSED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_SLOT1_PREFIX_"
    "SAME_RUNTIME_PRECONDITIONS_INCOMPLETE_NO_CHILD_OR_GPU"
)

READINESS_SCHEMA = "hawking.ascension.qwen80_l1_source_token_continuation_readiness_contract.v1"
READINESS_STATUS = (
    "PREPARED_QWEN80_SOURCE_TOKEN_L1_SLOT1_DELTANET_PREFIX_CAPTURE_RESERVED_NOT_EXECUTED"
)
L0_OUTER_SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_outer_capture.v1"
L0_OUTER_STATUS = "CAPTURED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_TERMINAL_PRE_L1_COMPONENT_ONLY"
L0_INNER_SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_capture.v1"
L0_INNER_STATUS = (
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_POST_STATE_ROLLBACK_RETAINED_OUTPUT_"
    "L1_BINDING_NOT_EXECUTED_COMPONENT_ONLY"
)
L0_ASSESSOR_BINDING_SCHEMA = "hawking.ascension.qwen80_l0_state_handoff_post_capture_assessor_binding.v1"
L0_ASSESSOR_BINDING_STATUS = (
    "REQUIRED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_POST_CAPTURE_ASSESSMENT_"
    "BEFORE_L1_JOINT_CAPTURE"
)
L0_ASSESSMENT_SCHEMA = "hawking.ascension.qwen80_l0_state_handoff_post_capture_assessment.v1"
L0_ASSESSMENT_STATUS = "EARNED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_COMPONENT_L1_BINDING_NOT_EXECUTED"
L0_RELEASE_SCHEMA = "hawking.ascension.qwen80_source_token_l0_state_handoff_quiet_metal_lease_release.v1"
L0_RELEASE_STATUS = (
    "RELEASED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_COMPONENT_QUIET_METAL_LEASE_"
    "AFTER_TERMINAL_CAPTURE"
)
MANIFEST_SCHEMA = "hawking.ascension.qwen80_complete_binary_gravity.v1"
ADMISSION_RECEIPT_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1"
ADMISSION_RECEIPT_STATUS = (
    "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
)
SCHEDULE_SCHEMA = "hawking.ascension.qwen80_48_layer_schedule_sealed_wrapper.v1"
SCHEDULE_STATUS = "SEALED_QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY_BOUND_NOT_EXECUTED"
RAW_SCHEDULE_SCHEMA = "hawking.ascension.qwen80_48_layer_payload_schedule_authority.v1"
RAW_SCHEDULE_STATUS = "PREPARED_QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY_NOT_EXECUTED"
JOINT_CHILD_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_child_preflight.v1"
)
JOINT_CHILD_PREFLIGHT_STATUS = (
    "PREPARED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_SLOT1_PREFIX_SAME_RUNTIME_CHILD_NOT_EXECUTED"
)
L0_SOURCE_OUTER_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_all_ten_true_moe_outer_preflight.v1"
)
L0_SOURCE_OUTER_PREFLIGHT_STATUS = (
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_OUTER_READY_FOR_"
    "SOURCE_TOKEN_CHILD_NOT_LEASED_OR_EXECUTED"
)
JOINT_HOST_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_host_preflight.v1"
)
JOINT_HOST_PREFLIGHT_STATUS = (
    "COMPILED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_SLOT1_PREFIX_SAME_RUNTIME_HOST_"
    "CPU_ONLY_NOT_LEASED_OR_EXECUTED"
)
CANONICAL_L0_CAPABILITY_FACTORY = (
    "Qwen80CompleteNativeRuntime::certify_source_token_l0_true_moe_continuation"
)
CANONICAL_L1_PREFIX_ENCODER = (
    "Qwen80CompleteNativeRuntime::encode_source_token_l1_deltanet_prefix_from_canonical_l0_continuation_into"
)
CANONICAL_L0_L1_FINALIZER = (
    "Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder::finalize_after_exact_joint_fence"
)

MODEL_KEY = "qwen80"
SOURCE_TOKEN_ID = 1
L0_LAYER = 0
L1_LAYER = 1
L0_SLOT = 0
L1_SLOT = 1
HIDDEN_ELEMENTS = 2_048
HIDDEN_BYTES = 8_192
L0_PREFIX_DISPATCHES = 9
L0_SUFFIX_DISPATCHES = 14
L0_TOTAL_DISPATCHES = L0_PREFIX_DISPATCHES + L0_SUFFIX_DISPATCHES
L1_PREFIX_DISPATCHES = 9
JOINT_TOTAL_DISPATCHES = L0_TOTAL_DISPATCHES + L1_PREFIX_DISPATCHES
L0_CONV_BYTES = 98_304
L0_RECURRENT_BYTES = 2_097_152
L1_CONV_OFFSET_BYTES = L0_CONV_BYTES
L1_RECURRENT_OFFSET_BYTES = L0_RECURRENT_BYTES
L1_CONV_CAPACITY_BYTES = L1_CONV_OFFSET_BYTES + L0_CONV_BYTES
L1_RECURRENT_CAPACITY_BYTES = L1_RECURRENT_OFFSET_BYTES + L0_RECURRENT_BYTES
# The immutable complete manifest is ~78 MiB.  This remains a bounded,
# caller-supplied CPU file read rather than an artifact scan.
MAX_INPUT_BYTES = 100_000_000

L1_PREFIX: tuple[dict[str, Any], ...] = (
    {"ordinal": 1, "stage": "input_rmsnorm", "kernel": "qwen_next_direct_packed_input_rmsnorm"},
    {"ordinal": 2, "stage": "qkvz_projection", "kernel": "qwen_binary_sign_scale_matvec"},
    {"ordinal": 3, "stage": "ba_projection", "kernel": "qwen_binary_sign_scale_matvec"},
    {"ordinal": 4, "stage": "qkvz_rearrange_conv", "kernel": "qwen_next_qkvz_rearrange_conv_l2"},
    {"ordinal": 5, "stage": "ba_decay_beta", "kernel": "qwen_next_ba_to_decay_beta"},
    {"ordinal": 6, "stage": "deltanet_recurrent", "kernel": "qwen_next_gated_delta_decode_single"},
    {"ordinal": 7, "stage": "deltanet_gated_rmsnorm", "kernel": "qwen_next_deltanet_gated_rmsnorm"},
    {"ordinal": 8, "stage": "out_projection", "kernel": "qwen_binary_sign_scale_matvec"},
    {"ordinal": 9, "stage": "first_residual", "kernel": "qwen_next_add_residual"},
)

L0_TRUE_MOE_KERNEL_TRACE: tuple[dict[str, Any], ...] = tuple(
    {"ordinal": ordinal, "kernel": kernel}
    for ordinal, kernel in enumerate(
        (
            "qwen_next_direct_packed_input_rmsnorm",
            "qwen_binary_sign_scale_matvec",
            "qwen_binary_sign_scale_matvec",
            "qwen_next_qkvz_rearrange_conv_l2",
            "qwen_next_ba_to_decay_beta",
            "qwen_next_gated_delta_decode_single",
            "qwen_next_deltanet_gated_rmsnorm",
            "qwen_binary_sign_scale_matvec",
            "qwen_next_add_residual",
            "qwen80_postnorm_router_top10_rmsnorm",
            "qwen80_postnorm_router_top10_matvec",
            "qwen80_postnorm_router_top10_select",
            "qwen80_all_ten_routed_wave_route_guard",
            "qwen80_all_ten_routed_wave_gate_up",
            "qwen80_all_ten_routed_wave_swiglu",
            "qwen80_all_ten_routed_wave_down_weighted",
            "qwen80_shared_expert_wave_gate_up",
            "qwen80_shared_expert_wave_swiglu",
            "qwen80_shared_expert_wave_down",
            "qwen80_shared_expert_wave_scalar_gate",
            "qwen80_shared_expert_wave_apply_sigmoid_gate",
            "qwen80_moe_wave_aggregate_second_residual_route_sum",
            "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
        ),
        start=1,
    )
)
JOINT_L0_L1_KERNEL_TRACE: tuple[dict[str, Any], ...] = (
    *L0_TRUE_MOE_KERNEL_TRACE,
    *tuple(
        {"ordinal": L0_TOTAL_DISPATCHES + entry["ordinal"], "kernel": entry["kernel"]}
        for entry in L1_PREFIX
    ),
)

FORBIDDEN_ARGUMENTS = frozenset(
    {
        "--execute",
        "--launch",
        "--child",
        "--metal",
        "--gpu",
        "--lease",
        "--server",
        "--watcher",
        "--capture",
        "--out",
    }
)


class SourceTokenL1PrefixLauncherError(ValueError):
    """Raised only for malformed CLI input; authority failures return a refusal."""


@dataclass(frozen=True)
class BoundDocument:
    """A verified document and its canonical sealed identity."""

    document: dict[str, Any]
    document_sha256: str
    document_seal_sha256: str


@dataclass(frozen=True)
class L0BaselineFacts:
    """Captured L0 facts retained strictly as provenance, never live input."""

    session_id: str
    output_f32le_sha256: str
    output_device_buffer_id: str


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(value: object) -> str:
    payload = value if isinstance(value, (bytes, bytearray)) else _canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _object(value: object, label: str, errors: list[str]) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(f"{label} must be an object")
        return None
    return dict(value)


def _require_bool(
    document: Mapping[str, Any], field: str, expected: bool, label: str, errors: list[str]
) -> None:
    if document.get(field) is not expected:
        errors.append(f"{label}.{field} must be {expected!r}")


def _require_int(
    document: Mapping[str, Any], field: str, expected: int, label: str, errors: list[str]
) -> None:
    if document.get(field) != expected:
        errors.append(f"{label}.{field} must be {expected}")


def _require_string(
    document: Mapping[str, Any], field: str, label: str, errors: list[str]
) -> str | None:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        errors.append(f"{label}.{field} must be a non-empty string")
        return None
    return value


def _require_sha(
    document: Mapping[str, Any], field: str, label: str, errors: list[str]
) -> str | None:
    value = document.get(field)
    if not _is_sha256(value):
        errors.append(f"{label}.{field} must be a lowercase SHA-256")
        return None
    return str(value)


def _identity(bound: BoundDocument) -> dict[str, str]:
    return {
        "document_sha256": bound.document_sha256,
        "document_seal_sha256": bound.document_seal_sha256,
    }


def _public_identity(bound: BoundDocument | None) -> dict[str, Any]:
    if bound is None:
        return {"present": False, "document_sha256": None, "document_seal_sha256": None}
    return {"present": True, **_identity(bound)}


def _bound_document(
    input_document: Mapping[str, Any],
    field: str,
    expected_schema: str,
    expected_status: str | None,
    errors: list[str],
) -> BoundDocument | None:
    """Validate one embedded sealed authority without probing arbitrary paths."""
    start = len(errors)
    binding = _object(input_document.get(field), f"input.{field}", errors)
    if binding is None:
        return None
    document = _object(binding.get("document"), f"input.{field}.document", errors)
    if document is None:
        return None
    try:
        verified = verify(document, label=f"input.{field}.document")
    except SealIntegrityError as exc:
        errors.append(f"input.{field}.document has an invalid seal: {exc}")
        return None
    claimed_document_sha = _require_sha(binding, "document_sha256", f"input.{field}", errors)
    claimed_seal = _require_sha(binding, "document_seal_sha256", f"input.{field}", errors)
    actual_document_sha = _sha256(verified)
    actual_seal = verified.get("seal_sha256")
    if claimed_document_sha is not None and claimed_document_sha != actual_document_sha:
        errors.append(f"input.{field}.document_sha256 does not bind its document")
    if claimed_seal is not None and claimed_seal != actual_seal:
        errors.append(f"input.{field}.document_seal_sha256 does not bind its document")
    if verified.get("schema") != expected_schema:
        errors.append(f"input.{field}.document.schema must be {expected_schema!r}")
    if expected_status is not None and verified.get("status") != expected_status:
        errors.append(f"input.{field}.document.status must be {expected_status!r}")
    if len(errors) != start:
        return None
    return BoundDocument(verified, actual_document_sha, str(actual_seal))


def _require_bound_identity(
    reference: object, bound: BoundDocument, label: str, errors: list[str]
) -> None:
    value = _object(reference, label, errors)
    if value is None:
        return
    if value.get("document_sha256") != bound.document_sha256:
        errors.append(f"{label}.document_sha256 does not bind the supplied authority")
    observed_seal = value.get("document_seal_sha256", value.get("seal_sha256"))
    if observed_seal != bound.document_seal_sha256:
        errors.append(f"{label}.document_seal_sha256 does not bind the supplied authority")


def _require_exact_prefix(value: object, label: str, errors: list[str]) -> None:
    if value != list(L1_PREFIX):
        errors.append(f"{label} must be the exact nine-dispatch Layer-1 DeltaNet prefix")


def _validate_manifest(manifest: BoundDocument, errors: list[str]) -> None:
    source = _object(manifest.document.get("source"), "manifest.source", errors)
    if source is not None and source.get("repository") != "Qwen/Qwen3-Coder-Next":
        errors.append("manifest.source.repository must bind Qwen/Qwen3-Coder-Next")


def _validate_admission(
    admission: BoundDocument, manifest: BoundDocument, errors: list[str]
) -> None:
    root = admission.document
    model = _object(root.get("model"), "admission_receipt.model", errors)
    if model is not None:
        if model.get("key") != MODEL_KEY:
            errors.append("admission_receipt.model.key must be qwen80")
        _require_string(model, "revision", "admission_receipt.model", errors)
    complete_manifest = _object(root.get("complete_manifest"), "admission_receipt.complete_manifest", errors)
    if complete_manifest is not None:
        if complete_manifest.get("seal_sha256") != manifest.document_seal_sha256:
            errors.append("admission_receipt.complete_manifest.seal_sha256 must bind manifest")
        if complete_manifest.get("schema") != MANIFEST_SCHEMA:
            errors.append("admission_receipt.complete_manifest.schema must bind manifest schema")


def _validate_schedule(schedule: BoundDocument, errors: list[str]) -> None:
    root = schedule.document
    boundary = _object(root.get("claim_boundary"), "schedule.claim_boundary", errors)
    if boundary is not None:
        _require_bool(boundary, "wrapper_is_read_only", True, "schedule.claim_boundary", errors)
        _require_bool(boundary, "future_joint_l0_to_l1_capture_authorized", False, "schedule.claim_boundary", errors)
    raw = _object(root.get("raw_schedule_authority"), "schedule.raw_schedule_authority", errors)
    if raw is not None:
        if raw.get("schema") != RAW_SCHEDULE_SCHEMA:
            errors.append("schedule.raw_schedule_authority.schema must name the immutable raw schedule")
        if raw.get("status") != RAW_SCHEDULE_STATUS:
            errors.append("schedule.raw_schedule_authority.status must name the non-executed raw schedule")
        _require_bool(raw, "present", True, "schedule.raw_schedule_authority", errors)
        _require_bool(raw, "raw_schedule_is_static_and_unmodified", True, "schedule.raw_schedule_authority", errors)
        _require_sha(raw, "sha256", "schedule.raw_schedule_authority", errors)
        if raw.get("raw_schedule_seal_sha256") is not None:
            errors.append("schedule.raw_schedule_authority.raw_schedule_seal_sha256 must remain null for the immutable raw schedule")
    facts = _object(root.get("schedule_facts"), "schedule.schedule_facts", errors)
    if facts is None:
        return
    _require_bool(facts, "all_48_layers_scheduled", True, "schedule.schedule_facts", errors)
    _require_int(facts, "layer_count", 48, "schedule.schedule_facts", errors)
    _require_int(facts, "delta_net_layer_count", 36, "schedule.schedule_facts", errors)
    _require_int(facts, "gqa_layer_count", 12, "schedule.schedule_facts", errors)
    layer = _object(facts.get("layer_1"), "schedule.schedule_facts.layer_1", errors)
    if layer is not None:
        _require_int(layer, "layer", L1_LAYER, "schedule.schedule_facts.layer_1", errors)
        if layer.get("mixer") != "delta_net":
            errors.append("schedule.schedule_facts.layer_1.mixer must be delta_net")
        _require_int(layer, "state_slot", L1_SLOT, "schedule.schedule_facts.layer_1", errors)
        if layer.get("state_domain") != "delta_net_conv_and_recurrent":
            errors.append("schedule.schedule_facts.layer_1.state_domain must be delta_net_conv_and_recurrent")


def _validate_l0_state_record(
    state: Mapping[str, Any],
    field: str,
    *,
    hash_field: str,
    offset_bytes: int,
    capacity_bytes: int,
    label: str,
    errors: list[str],
) -> None:
    record = _object(state.get(field), f"{label}.{field}", errors)
    if record is None:
        return
    _require_string(record, "allocation_id", f"{label}.{field}", errors)
    _require_int(record, "slot", L0_SLOT, f"{label}.{field}", errors)
    _require_int(record, "offset_bytes", offset_bytes, f"{label}.{field}", errors)
    _require_int(record, "capacity_bytes", capacity_bytes, f"{label}.{field}", errors)
    _require_sha(record, "device_buffer_id", f"{label}.{field}", errors)
    _require_sha(record, hash_field, f"{label}.{field}", errors)


def _validate_l1_binding_state_record(
    state: Mapping[str, Any],
    field: str,
    *,
    offset_bytes: int,
    capacity_bytes: int,
    label: str,
    errors: list[str],
) -> None:
    record = _object(state.get(field), f"{label}.{field}", errors)
    if record is None:
        return
    _require_string(record, "allocation_id", f"{label}.{field}", errors)
    _require_int(record, "slot", L1_SLOT, f"{label}.{field}", errors)
    _require_int(record, "offset_bytes", offset_bytes, f"{label}.{field}", errors)
    _require_int(record, "capacity_bytes", capacity_bytes, f"{label}.{field}", errors)
    identity = _require_sha(record, "device_buffer_id", f"{label}.{field}", errors)
    duplicate = _require_sha(record, "device_buffer_identity_sha256", f"{label}.{field}", errors)
    if identity is not None and duplicate is not None and identity != duplicate:
        errors.append(f"{label}.{field}.device_buffer_identity_sha256 must equal device_buffer_id")


def _validate_l0_inner(
    inner: BoundDocument,
    *,
    manifest: BoundDocument,
    admission: BoundDocument,
    errors: list[str],
) -> L0BaselineFacts | None:
    """Validate the actual captured ABI, whose handoff facts are nested."""
    start = len(errors)
    root = inner.document
    _require_bool(root, "component_only", True, "l0_inner", errors)
    _require_bool(root, "complete_layer_or_token_performed", False, "l0_inner", errors)
    _require_bool(root, "l1_binding_not_executed", True, "l0_inner", errors)
    _require_int(root, "l1_prefix_dispatches", 0, "l0_inner", errors)
    artifact = _object(root.get("artifact_binding"), "l0_inner.artifact_binding", errors)
    if artifact is not None:
        if artifact.get("manifest_seal_sha256") != manifest.document_seal_sha256:
            errors.append("l0_inner.artifact_binding.manifest_seal_sha256 must bind manifest")
        if artifact.get("admission_receipt_seal_sha256") != admission.document_seal_sha256:
            errors.append("l0_inner.artifact_binding.admission_receipt_seal_sha256 must bind admission")
        _require_string(artifact, "source_revision", "l0_inner.artifact_binding", errors)
    graph = _object(root.get("same_command_graph"), "l0_inner.same_command_graph", errors)
    if graph is not None:
        _require_int(graph, "source_token_id", SOURCE_TOKEN_ID, "l0_inner.same_command_graph", errors)
        _require_int(graph, "prefix_dispatches", L0_PREFIX_DISPATCHES, "l0_inner.same_command_graph", errors)
        _require_int(graph, "suffix_dispatches", L0_SUFFIX_DISPATCHES, "l0_inner.same_command_graph", errors)
        _require_int(graph, "total_dispatches", L0_TOTAL_DISPATCHES, "l0_inner.same_command_graph", errors)
        _require_bool(graph, "same_command_graph_retained", True, "l0_inner.same_command_graph", errors)
        _require_bool(graph, "fenced_once_after_prefix_and_suffix", True, "l0_inner.same_command_graph", errors)
    handoff = _object(root.get("l0_state_handoff"), "l0_inner.l0_state_handoff", errors)
    if handoff is None:
        return None
    if handoff.get("schema") != L0_INNER_SCHEMA:
        errors.append(f"l0_inner.l0_state_handoff.schema must be {L0_INNER_SCHEMA!r}")
    if handoff.get("status") != L0_INNER_STATUS:
        errors.append(f"l0_inner.l0_state_handoff.status must be {L0_INNER_STATUS!r}")
    session_id = _require_string(handoff, "session_id", "l0_inner.l0_state_handoff", errors)
    _require_int(handoff, "source_token_id", SOURCE_TOKEN_ID, "l0_inner.l0_state_handoff", errors)
    _require_bool(handoff, "same_command_graph_retained", True, "l0_inner.l0_state_handoff", errors)
    _require_bool(handoff, "l1_binding_not_executed", True, "l0_inner.l0_state_handoff", errors)
    _require_int(handoff, "l1_prefix_dispatches", 0, "l0_inner.l0_state_handoff", errors)
    retained = _object(
        handoff.get("retained_l0_second_residual"),
        "l0_inner.l0_state_handoff.retained_l0_second_residual",
        errors,
    )
    output_hash: str | None = None
    output_buffer: str | None = None
    if retained is not None:
        _require_int(retained, "elements", HIDDEN_ELEMENTS, "l0_inner.l0_state_handoff.retained_l0_second_residual", errors)
        _require_int(retained, "bytes", HIDDEN_BYTES, "l0_inner.l0_state_handoff.retained_l0_second_residual", errors)
        output_hash = _require_sha(retained, "f32le_sha256", "l0_inner.l0_state_handoff.retained_l0_second_residual", errors)
        output_buffer = _require_sha(retained, "device_buffer_id", "l0_inner.l0_state_handoff.retained_l0_second_residual", errors)
        _require_bool(retained, "retained_for_future_layer1_encode", True, "l0_inner.l0_state_handoff.retained_l0_second_residual", errors)
    l0_state = _object(handoff.get("l0_post_state_commit"), "l0_inner.l0_state_handoff.l0_post_state_commit", errors)
    if l0_state is not None:
        _require_int(l0_state, "layer", L0_LAYER, "l0_inner.l0_state_handoff.l0_post_state_commit", errors)
        _require_int(l0_state, "linear_state_slot", L0_SLOT, "l0_inner.l0_state_handoff.l0_post_state_commit", errors)
        _require_bool(l0_state, "checkpoint_before_mutation", True, "l0_inner.l0_state_handoff.l0_post_state_commit", errors)
        _validate_l0_state_record(l0_state, "active_conv", hash_field="post_state_f32le_sha256", offset_bytes=0, capacity_bytes=L0_CONV_BYTES, label="l0_inner.l0_state_handoff.l0_post_state_commit", errors=errors)
        _validate_l0_state_record(l0_state, "active_recurrent", hash_field="post_state_f32le_sha256", offset_bytes=0, capacity_bytes=L0_RECURRENT_BYTES, label="l0_inner.l0_state_handoff.l0_post_state_commit", errors=errors)
        _validate_l0_state_record(l0_state, "rollback_conv", hash_field="checkpoint_f32le_sha256", offset_bytes=0, capacity_bytes=L0_CONV_BYTES, label="l0_inner.l0_state_handoff.l0_post_state_commit", errors=errors)
        _validate_l0_state_record(l0_state, "rollback_recurrent", hash_field="checkpoint_f32le_sha256", offset_bytes=0, capacity_bytes=L0_RECURRENT_BYTES, label="l0_inner.l0_state_handoff.l0_post_state_commit", errors=errors)
    layer1 = _object(handoff.get("layer1_input_binding"), "l0_inner.l0_state_handoff.layer1_input_binding", errors)
    if layer1 is not None:
        if session_id is not None and layer1.get("session_id") != session_id:
            errors.append("l0_inner.l0_state_handoff.layer1_input_binding.session_id must retain captured L0 session")
        _require_int(layer1, "layer", L1_LAYER, "l0_inner.l0_state_handoff.layer1_input_binding", errors)
        _require_int(layer1, "linear_state_slot", L1_SLOT, "l0_inner.l0_state_handoff.layer1_input_binding", errors)
        if output_buffer is not None and layer1.get("input_device_buffer_id") != output_buffer:
            errors.append("l0_inner.l0_state_handoff.layer1_input_binding must name the captured L0 output buffer")
        if output_hash is not None and layer1.get("input_f32le_sha256") != output_hash:
            errors.append("l0_inner.l0_state_handoff.layer1_input_binding must name the captured L0 output hash")
        _require_bool(layer1, "same_command_graph_retained", True, "l0_inner.l0_state_handoff.layer1_input_binding", errors)
        _require_bool(layer1, "l1_binding_executed", False, "l0_inner.l0_state_handoff.layer1_input_binding", errors)
        _validate_l1_binding_state_record(layer1, "active_conv", offset_bytes=L1_CONV_OFFSET_BYTES, capacity_bytes=L1_CONV_CAPACITY_BYTES, label="l0_inner.l0_state_handoff.layer1_input_binding", errors=errors)
        _validate_l1_binding_state_record(layer1, "active_recurrent", offset_bytes=L1_RECURRENT_OFFSET_BYTES, capacity_bytes=L1_RECURRENT_CAPACITY_BYTES, label="l0_inner.l0_state_handoff.layer1_input_binding", errors=errors)
    boundary = _object(handoff.get("claim_boundary"), "l0_inner.l0_state_handoff.claim_boundary", errors)
    if boundary is not None:
        for field in (
            "component_only",
            "layer1_not_encoded",
            "retention_binding_is_not_a_layer1_execution_claim",
            "may_not_satisfy_next_layer_execution_dependency",
        ):
            _require_bool(boundary, field, True, "l0_inner.l0_state_handoff.claim_boundary", errors)
    if len(errors) != start or None in (session_id, output_hash, output_buffer):
        return None
    return L0BaselineFacts(str(session_id), str(output_hash), str(output_buffer))


def _validate_l0_outer(
    outer: BoundDocument,
    *,
    inner: BoundDocument,
    errors: list[str],
) -> None:
    """Validate the actual receipt-last outer ABI from the 08:16:20 capture."""
    root = outer.document
    inner_probe = _object(root.get("inner_probe_capture"), "l0_outer.inner_probe_capture", errors)
    if inner_probe is not None:
        _require_bool(inner_probe, "binding_valid", True, "l0_outer.inner_probe_capture", errors)
        _require_bool(inner_probe, "present", True, "l0_outer.inner_probe_capture", errors)
        if inner_probe.get("schema") != L0_INNER_SCHEMA:
            errors.append("l0_outer.inner_probe_capture.schema must bind L0 inner capture schema")
        if inner_probe.get("status") != L0_INNER_STATUS:
            errors.append("l0_outer.inner_probe_capture.status must bind L0 inner capture status")
        receipt = _object(inner_probe.get("receipt"), "l0_outer.inner_probe_capture.receipt", errors)
        if receipt is not None:
            _require_bool(receipt, "present", True, "l0_outer.inner_probe_capture.receipt", errors)
            _require_sha(receipt, "sha256", "l0_outer.inner_probe_capture.receipt", errors)
            if receipt.get("seal_sha256") != inner.document_seal_sha256:
                errors.append("l0_outer.inner_probe_capture.receipt.seal_sha256 must bind supplied L0 inner capture")
    contract = _object(root.get("source_binding"), "l0_outer.source_binding", errors)
    if contract is not None:
        handoff = _object(contract.get("handoff_contract"), "l0_outer.source_binding.handoff_contract", errors)
        if handoff is not None:
            _require_int(handoff, "source_token_id", SOURCE_TOKEN_ID, "l0_outer.source_binding.handoff_contract", errors)
            _require_int(handoff, "prefix_dispatches", L0_PREFIX_DISPATCHES, "l0_outer.source_binding.handoff_contract", errors)
            _require_int(handoff, "suffix_dispatches", L0_SUFFIX_DISPATCHES, "l0_outer.source_binding.handoff_contract", errors)
            _require_int(handoff, "total_dispatches", L0_TOTAL_DISPATCHES, "l0_outer.source_binding.handoff_contract", errors)
            _require_bool(handoff, "same_tcb_fence_required", True, "l0_outer.source_binding.handoff_contract", errors)
            _require_bool(handoff, "l1_binding_not_executed", True, "l0_outer.source_binding.handoff_contract", errors)
            _require_int(handoff, "l1_prefix_dispatches", 0, "l0_outer.source_binding.handoff_contract", errors)
    one_shot = _object(root.get("one_shot"), "l0_outer.one_shot", errors)
    if one_shot is not None:
        for field in (
            "automatic_retry_disabled",
            "lease_reuse_prohibited_after_terminal",
            "outer_reaped_child",
            "same_capture_dir_never_starts_a_second_child",
            "terminal_receipt_written_last",
        ):
            _require_bool(one_shot, field, True, "l0_outer.one_shot", errors)
    boundary = _object(root.get("claim_boundary"), "l0_outer.claim_boundary", errors)
    if boundary is not None:
        _require_bool(boundary, "l1_binding_not_executed", True, "l0_outer.claim_boundary", errors)
        _require_bool(boundary, "l1_prefix_executed", False, "l0_outer.claim_boundary", errors)
        _require_bool(boundary, "watcher_or_server_transition_not_authorized", True, "l0_outer.claim_boundary", errors)
    child = _object(root.get("child"), "l0_outer.child", errors)
    if child is not None:
        terminal = _object(child.get("terminal"), "l0_outer.child.terminal", errors)
        if terminal is not None:
            _require_int(terminal, "exit_code", 0, "l0_outer.child.terminal", errors)
            _require_bool(terminal, "reaped", True, "l0_outer.child.terminal", errors)
            _require_bool(terminal, "timed_out", False, "l0_outer.child.terminal", errors)


def _validate_readiness(
    readiness: BoundDocument,
    *,
    inner: BoundDocument,
    schedule: BoundDocument,
    errors: list[str],
) -> None:
    root = readiness.document
    _require_bool(root, "prepared", True, "continuation_readiness", errors)
    _require_bool(root, "l1_execution_performed_by_this_contract", False, "continuation_readiness", errors)
    _require_int(root, "l1_prefix_dispatches_executed_by_this_contract", 0, "continuation_readiness", errors)
    _require_bound_identity(root.get("l0_state_handoff_receipt"), inner, "continuation_readiness.l0_state_handoff_receipt", errors)
    _require_bound_identity(root.get("schedule_authority"), schedule, "continuation_readiness.schedule_authority", errors)
    scope = _object(root.get("future_l1_slot1_deltanet_prefix_scope"), "continuation_readiness.future_l1_slot1_deltanet_prefix_scope", errors)
    if scope is not None:
        _require_int(scope, "layer", L1_LAYER, "continuation_readiness.future_l1_slot1_deltanet_prefix_scope", errors)
        if scope.get("mixer") != "delta_net":
            errors.append("continuation_readiness.future_l1_slot1_deltanet_prefix_scope.mixer must be delta_net")
        _require_int(scope, "linear_state_slot", L1_SLOT, "continuation_readiness.future_l1_slot1_deltanet_prefix_scope", errors)
        _require_int(scope, "exact_prefix_dispatch_count", L1_PREFIX_DISPATCHES, "continuation_readiness.future_l1_slot1_deltanet_prefix_scope", errors)
        _require_exact_prefix(scope.get("exact_prefix_dispatches"), "continuation_readiness.future_l1_slot1_deltanet_prefix_scope.exact_prefix_dispatches", errors)
        _require_bool(scope, "no_l1_suffix_or_moe_dispatch_authorized", True, "continuation_readiness.future_l1_slot1_deltanet_prefix_scope", errors)
    boundary = _object(root.get("authority_boundary"), "continuation_readiness.authority_boundary", errors)
    if boundary is not None:
        for field in (
            "new_physical_model_processes_authorized",
            "server_starts_authorized",
            "port_binds_authorized",
            "gpu_leases_authorized",
            "watcher_changes_authorized",
            "tournament_state_mutations_authorized",
        ):
            _require_int(boundary, field, 0, "continuation_readiness.authority_boundary", errors)


def _validate_assessor_binding(
    binding: BoundDocument,
    *,
    l0_outer: BoundDocument,
    l0_inner: BoundDocument,
    assessment: BoundDocument,
    release: BoundDocument,
    manifest: BoundDocument,
    admission: BoundDocument,
    facts: L0BaselineFacts | None,
    errors: list[str],
) -> None:
    """Bind the earned assessor to the exact historical L0 component chain.

    The wrapper is deliberately *not* execution authority.  Its job is to
    make the baseline's immutable identities and non-transfer boundary
    explicit before a separate joint L0+L1 child is even planned.
    """
    root = binding.document
    _require_bound_identity(root.get("l0_outer_terminal"), l0_outer, "l0_post_capture_assessor_binding.l0_outer_terminal", errors)
    _require_bound_identity(root.get("l0_inner_capture"), l0_inner, "l0_post_capture_assessor_binding.l0_inner_capture", errors)
    _require_bound_identity(
        root.get("post_capture_assessment"),
        assessment,
        "l0_post_capture_assessor_binding.post_capture_assessment",
        errors,
    )
    _require_bound_identity(
        root.get("lease_release_receipt"),
        release,
        "l0_post_capture_assessor_binding.lease_release_receipt",
        errors,
    )
    required = _object(root.get("required_assessment"), "l0_post_capture_assessor_binding.required_assessment", errors)
    if required is not None:
        if required.get("schema") != L0_ASSESSMENT_SCHEMA:
            errors.append("l0_post_capture_assessor_binding.required_assessment.schema must name the post-capture assessor")
        if required.get("earned_status") != L0_ASSESSMENT_STATUS:
            errors.append("l0_post_capture_assessor_binding.required_assessment.earned_status must name the earned assessor status")
        if required.get("assessment_document_sha256") != assessment.document_sha256:
            errors.append("l0_post_capture_assessor_binding.required_assessment.document_sha256 must bind supplied assessment")
        if required.get("assessment_document_seal_sha256") != assessment.document_seal_sha256:
            errors.append("l0_post_capture_assessor_binding.required_assessment.document_seal_sha256 must bind supplied assessment")
        _require_bool(required, "must_be_sealed", True, "l0_post_capture_assessor_binding.required_assessment", errors)
        _require_bool(required, "must_bind_actual_release", True, "l0_post_capture_assessor_binding.required_assessment", errors)
        _require_bool(required, "must_bind_l0_outer_and_inner", True, "l0_post_capture_assessor_binding.required_assessment", errors)
        _require_bool(required, "must_remain_l1_not_executed", True, "l0_post_capture_assessor_binding.required_assessment", errors)
    _require_bool(root, "assessment_result_bound", True, "l0_post_capture_assessor_binding", errors)
    _require_bool(root, "assessment_required_before_joint_child_launch", True, "l0_post_capture_assessor_binding", errors)
    _require_bool(root, "baseline_l0_evidence_is_provenance_only", True, "l0_post_capture_assessor_binding", errors)
    _require_bool(root, "cross_process_pinned_buffer_transfer_allowed", False, "l0_post_capture_assessor_binding", errors)
    _require_bool(root, "joint_l0_reencode_required", True, "l0_post_capture_assessor_binding", errors)
    _require_bool(root, "future_l1_requires_fresh_same_runtime_same_tcb_joint_l0_to_l1_capture", True, "l0_post_capture_assessor_binding", errors)
    _require_bool(root, "joint_child_execution_authorized_by_this_wrapper", False, "l0_post_capture_assessor_binding", errors)
    _require_bool(root, "l1_execution_authorized_by_this_wrapper", False, "l0_post_capture_assessor_binding", errors)

    assessment_root = assessment.document
    _require_bool(assessment_root, "earned_l0_state_handoff_component", True, "post_capture_assessment", errors)
    _require_bool(assessment_root, "l0_handoff_is_evidence_baseline_only", True, "post_capture_assessment", errors)
    _require_bool(assessment_root, "l1_binding_not_executed", True, "post_capture_assessment", errors)
    _require_int(assessment_root, "l1_prefix_dispatches", 0, "post_capture_assessment", errors)
    _require_bool(assessment_root, "l1_continuation_prepared", False, "post_capture_assessment", errors)
    _require_bool(assessment_root, "l1_continuation_remains_non_executing", True, "post_capture_assessment", errors)
    _require_bool(assessment_root, "may_not_satisfy_next_layer_execution_dependency", True, "post_capture_assessment", errors)
    _require_bool(
        assessment_root,
        "future_l1_requires_fresh_same_runtime_same_tcb_joint_l0_to_l1_capture",
        True,
        "post_capture_assessment",
        errors,
    )
    _require_bool(
        assessment_root,
        "cross_process_or_prior_capture_pinned_buffer_reuse_authorized",
        False,
        "post_capture_assessment",
        errors,
    )
    _require_bound_identity(
        assessment_root.get("l0_outer_terminal"), l0_outer, "post_capture_assessment.l0_outer_terminal", errors
    )
    _require_bound_identity(
        assessment_root.get("l0_inner_receipt"), l0_inner, "post_capture_assessment.l0_inner_receipt", errors
    )
    _require_bound_identity(
        assessment_root.get("lease_release_receipt"), release, "post_capture_assessment.lease_release_receipt", errors
    )

    release_root = release.document
    coordination = _object(release_root.get("coordination"), "lease_release_receipt.coordination", errors)
    if coordination is not None:
        _require_bool(coordination, "quiet_qwen80_component_lease_released", True, "lease_release_receipt.coordination", errors)
        _require_bool(coordination, "watcher_hold_remains_active", True, "lease_release_receipt.coordination", errors)
        _require_bool(coordination, "automatic_retry_prohibited", True, "lease_release_receipt.coordination", errors)
    release_outer = _object(release_root.get("outer_terminal"), "lease_release_receipt.outer_terminal", errors)
    if release_outer is not None:
        if release_outer.get("seal_sha256") != l0_outer.document_seal_sha256:
            errors.append("lease_release_receipt.outer_terminal.seal_sha256 must bind supplied L0 outer")
        if release_outer.get("status") != L0_OUTER_STATUS:
            errors.append("lease_release_receipt.outer_terminal.status must bind L0 outer terminal status")

    retained = _object(root.get("retained_l0_state_handoff"), "l0_post_capture_assessor_binding.retained_l0_state_handoff", errors)
    if retained is not None:
        _require_int(retained, "source_token_id", SOURCE_TOKEN_ID, "l0_post_capture_assessor_binding.retained_l0_state_handoff", errors)
        _require_bool(retained, "l1_binding_not_executed", True, "l0_post_capture_assessor_binding.retained_l0_state_handoff", errors)
        _require_int(retained, "l1_prefix_dispatches", 0, "l0_post_capture_assessor_binding.retained_l0_state_handoff", errors)
        output = _object(retained.get("retained_l0_second_residual"), "l0_post_capture_assessor_binding.retained_l0_state_handoff.retained_l0_second_residual", errors)
        if output is not None:
            _require_int(output, "elements", HIDDEN_ELEMENTS, "l0_post_capture_assessor_binding.retained_l0_state_handoff.retained_l0_second_residual", errors)
            _require_int(output, "bytes", HIDDEN_BYTES, "l0_post_capture_assessor_binding.retained_l0_state_handoff.retained_l0_second_residual", errors)
            if facts is not None:
                if output.get("f32le_sha256") != facts.output_f32le_sha256:
                    errors.append("l0_post_capture_assessor_binding retained L0 output hash must bind L0 inner")
                if output.get("device_buffer_id") != facts.output_device_buffer_id:
                    errors.append("l0_post_capture_assessor_binding retained L0 buffer id must bind L0 inner")
        reserved = _object(retained.get("reserved_l1_slot"), "l0_post_capture_assessor_binding.retained_l0_state_handoff.reserved_l1_slot", errors)
        if reserved is not None:
            _require_int(reserved, "layer", L1_LAYER, "l0_post_capture_assessor_binding.retained_l0_state_handoff.reserved_l1_slot", errors)
            _require_int(reserved, "linear_state_slot", L1_SLOT, "l0_post_capture_assessor_binding.retained_l0_state_handoff.reserved_l1_slot", errors)
            _require_int(reserved, "active_conv_offset_bytes", L1_CONV_OFFSET_BYTES, "l0_post_capture_assessor_binding.retained_l0_state_handoff.reserved_l1_slot", errors)
            _require_int(reserved, "active_recurrent_offset_bytes", L1_RECURRENT_OFFSET_BYTES, "l0_post_capture_assessor_binding.retained_l0_state_handoff.reserved_l1_slot", errors)
            _require_int(reserved, "active_conv_capacity_bytes", L1_CONV_CAPACITY_BYTES, "l0_post_capture_assessor_binding.retained_l0_state_handoff.reserved_l1_slot", errors)
            _require_int(reserved, "active_recurrent_capacity_bytes", L1_RECURRENT_CAPACITY_BYTES, "l0_post_capture_assessor_binding.retained_l0_state_handoff.reserved_l1_slot", errors)

    future = _object(root.get("future_joint_capture_requirement"), "l0_post_capture_assessor_binding.future_joint_capture_requirement", errors)
    if future is not None:
        _require_int(future, "fresh_l0_reencode_dispatches", L0_TOTAL_DISPATCHES, "l0_post_capture_assessor_binding.future_joint_capture_requirement", errors)
        _require_int(future, "future_l1_slot1_prefix_dispatches", L1_PREFIX_DISPATCHES, "l0_post_capture_assessor_binding.future_joint_capture_requirement", errors)
        _require_int(future, "future_joint_total_dispatches", JOINT_TOTAL_DISPATCHES, "l0_post_capture_assessor_binding.future_joint_capture_requirement", errors)
        _require_bool(future, "historical_pinned_buffer_or_state_import_allowed", False, "l0_post_capture_assessor_binding.future_joint_capture_requirement", errors)
        _require_bool(future, "historical_receipts_are_provenance_only", True, "l0_post_capture_assessor_binding.future_joint_capture_requirement", errors)
        _require_bool(future, "same_runtime_required", True, "l0_post_capture_assessor_binding.future_joint_capture_requirement", errors)
        _require_bool(future, "same_session_required", True, "l0_post_capture_assessor_binding.future_joint_capture_requirement", errors)
        _require_bool(future, "same_tcb_required", True, "l0_post_capture_assessor_binding.future_joint_capture_requirement", errors)

    immutable = _object(root.get("immutable_authority_chain"), "l0_post_capture_assessor_binding.immutable_authority_chain", errors)
    if immutable is not None:
        versioned = _object(immutable.get("versioned_manifest_and_admission"), "l0_post_capture_assessor_binding.immutable_authority_chain.versioned_manifest_and_admission", errors)
        if versioned is not None:
            if versioned.get("model_key") != MODEL_KEY:
                errors.append("l0_post_capture_assessor_binding immutable chain must bind qwen80")
            if versioned.get("manifest_seal_sha256") != manifest.document_seal_sha256:
                errors.append("l0_post_capture_assessor_binding immutable manifest seal must bind supplied manifest")
            if versioned.get("admission_seal_sha256") != admission.document_seal_sha256:
                errors.append("l0_post_capture_assessor_binding immutable admission seal must bind supplied admission")
            manifest_ref = _object(admission.document.get("complete_manifest"), "admission_receipt.complete_manifest", errors)
            if manifest_ref is not None and versioned.get("manifest_file_sha256") != manifest_ref.get("document_sha256"):
                errors.append("l0_post_capture_assessor_binding immutable manifest file SHA must bind admission manifest reference")
            _require_sha(versioned, "admission_file_sha256", "l0_post_capture_assessor_binding.immutable_authority_chain.versioned_manifest_and_admission", errors)


def _upstream_authority_identity(
    *,
    readiness: BoundDocument,
    l0_outer: BoundDocument,
    l0_inner: BoundDocument,
    assessor_binding: BoundDocument,
    assessment: BoundDocument,
    lease_release: BoundDocument,
    manifest: BoundDocument,
    admission_receipt: BoundDocument,
    schedule: BoundDocument,
    child_sha256: str,
) -> str:
    return _sha256(
        {
            "continuation_readiness": _identity(readiness),
            "l0_outer_terminal": _identity(l0_outer),
            "l0_inner_capture": _identity(l0_inner),
            "l0_post_capture_assessor_binding": _identity(assessor_binding),
            "post_capture_assessment": _identity(assessment),
            "lease_release_receipt": _identity(lease_release),
            "manifest": _identity(manifest),
            "admission_receipt": _identity(admission_receipt),
            "schedule": _identity(schedule),
            "future_joint_l0_l1_child_sha256": child_sha256,
        }
    )


def _validate_joint_child_preflight(
    child: BoundDocument,
    *,
    expected_preflight_identity: str,
    errors: list[str],
) -> str | None:
    root = child.document
    child_sha = _require_sha(root, "future_joint_l0_l1_child_sha256", "joint_child_preflight", errors)
    if root.get("preflight_identity_sha256") != expected_preflight_identity:
        errors.append("joint_child_preflight.preflight_identity_sha256 does not bind exact supplied authorities")
    _require_bool(root, "child_started", False, "joint_child_preflight", errors)
    _require_bool(root, "metal_or_gpu_activity_performed", False, "joint_child_preflight", errors)
    _require_bool(root, "component_only", True, "joint_child_preflight", errors)
    _require_bool(root, "same_runtime_required", True, "joint_child_preflight", errors)
    _require_bool(root, "same_session_required", True, "joint_child_preflight", errors)
    _require_bool(root, "same_tcb_required", True, "joint_child_preflight", errors)
    _require_bool(root, "baseline_l0_receipts_provenance_only", True, "joint_child_preflight", errors)
    _require_bool(root, "cross_process_pinned_buffer_transfer_allowed", False, "joint_child_preflight", errors)
    _require_bool(root, "external_l0_buffer_or_state_import_allowed", False, "joint_child_preflight", errors)
    _require_bool(root, "opaque_canonical_l0_continuation_required", True, "joint_child_preflight", errors)
    _require_bool(root, "raw_pinned_buffer_or_dispatch_count_input_allowed", False, "joint_child_preflight", errors)
    _require_bool(root, "opaque_capability_must_bind_runtime_state_arena_identity", True, "joint_child_preflight", errors)
    if root.get("future_joint_host_binary_bound") is not True:
        errors.append("joint_child_preflight does not bind a concrete strict joint L0+L1 host binary")
    if root.get("future_joint_host_binary_role") != "strict_joint_l0_l1_same_runtime_host":
        errors.append("joint_child_preflight.future_joint_host_binary_role must name the concrete strict joint host")
    for prohibited in (
        "session_id",
        "runtime_identity_sha256",
        "input_device_buffer_id",
        "input_f32le_sha256",
        "l0_input_binding",
    ):
        if prohibited in root:
            errors.append(f"joint_child_preflight may not import historical {prohibited}")
    l0 = _object(root.get("l0_reencode"), "joint_child_preflight.l0_reencode", errors)
    if l0 is not None:
        _require_int(l0, "source_token_id", SOURCE_TOKEN_ID, "joint_child_preflight.l0_reencode", errors)
        _require_int(l0, "prefix_dispatches", L0_PREFIX_DISPATCHES, "joint_child_preflight.l0_reencode", errors)
        _require_int(l0, "suffix_dispatches", L0_SUFFIX_DISPATCHES, "joint_child_preflight.l0_reencode", errors)
        _require_int(l0, "total_dispatches", L0_TOTAL_DISPATCHES, "joint_child_preflight.l0_reencode", errors)
        _require_bool(l0, "same_tcb_fence_required", True, "joint_child_preflight.l0_reencode", errors)
    l1 = _object(root.get("l1_prefix"), "joint_child_preflight.l1_prefix", errors)
    if l1 is not None:
        _require_int(l1, "layer", L1_LAYER, "joint_child_preflight.l1_prefix", errors)
        if l1.get("mixer") != "delta_net":
            errors.append("joint_child_preflight.l1_prefix.mixer must be delta_net")
        _require_int(l1, "linear_state_slot", L1_SLOT, "joint_child_preflight.l1_prefix", errors)
        _require_int(l1, "prefix_dispatches", L1_PREFIX_DISPATCHES, "joint_child_preflight.l1_prefix", errors)
        _require_exact_prefix(l1.get("exact_prefix_dispatches"), "joint_child_preflight.l1_prefix.exact_prefix_dispatches", errors)
        _require_bool(l1, "no_l1_suffix_or_moe_dispatch_authorized", True, "joint_child_preflight.l1_prefix", errors)
    graph = _object(root.get("joint_command_graph"), "joint_child_preflight.joint_command_graph", errors)
    if graph is not None:
        _require_int(graph, "l0_dispatches", L0_TOTAL_DISPATCHES, "joint_child_preflight.joint_command_graph", errors)
        _require_int(graph, "l1_prefix_dispatches", L1_PREFIX_DISPATCHES, "joint_child_preflight.joint_command_graph", errors)
        _require_int(graph, "total_dispatches", JOINT_TOTAL_DISPATCHES, "joint_child_preflight.joint_command_graph", errors)
        _require_bool(graph, "same_runtime_same_tcb_required", True, "joint_child_preflight.joint_command_graph", errors)
        _require_bool(graph, "same_session_required", True, "joint_child_preflight.joint_command_graph", errors)
        _require_bool(graph, "single_fence_after_l0_and_l1_prefix_required", True, "joint_child_preflight.joint_command_graph", errors)
        _require_bool(graph, "non_timed_token_command_buffer_required", True, "joint_child_preflight.joint_command_graph", errors)
        if graph.get("tcb_trace_mode") != "off":
            errors.append("joint_child_preflight.joint_command_graph.tcb_trace_mode must be off")
        if graph.get("opaque_capability_factory") != CANONICAL_L0_CAPABILITY_FACTORY:
            errors.append("joint_child_preflight.joint_command_graph.opaque_capability_factory must name the canonical L0 capability factory")
        if graph.get("runtime_api") != CANONICAL_L1_PREFIX_ENCODER:
            errors.append("joint_child_preflight.joint_command_graph.runtime_api must name the opaque-capability L1 prefix encoder")
        if graph.get("consuming_finalizer") != CANONICAL_L0_L1_FINALIZER:
            errors.append("joint_child_preflight.joint_command_graph.consuming_finalizer must own the exact joint fence")
        _require_bool(graph, "structural_kernel_trace_required", True, "joint_child_preflight.joint_command_graph", errors)
        if graph.get("exact_l0_kernel_trace") != list(L0_TRUE_MOE_KERNEL_TRACE):
            errors.append("joint_child_preflight.joint_command_graph.exact_l0_kernel_trace must name canonical L0 23-kernel trace")
        if graph.get("exact_joint_kernel_trace") != list(JOINT_L0_L1_KERNEL_TRACE):
            errors.append("joint_child_preflight.joint_command_graph.exact_joint_kernel_trace must name canonical L0+L1 32-kernel trace")
        _require_bool(graph, "finalizer_must_consume_the_only_command_buffer_before_fence", True, "joint_child_preflight.joint_command_graph", errors)
        l0_readbacks = _object(
            graph.get("fresh_l0_suffix_readbacks_required"),
            "joint_child_preflight.joint_command_graph.fresh_l0_suffix_readbacks_required",
            errors,
        )
        if l0_readbacks is not None:
            for field in (
                "route_guard",
                "postnorm",
                "router_logits",
                "shared_output",
                "routed_sum",
                "second_residual",
            ):
                _require_bool(
                    l0_readbacks,
                    field,
                    True,
                    "joint_child_preflight.joint_command_graph.fresh_l0_suffix_readbacks_required",
                    errors,
                )
            _require_int(
                l0_readbacks,
                "all_ten_weighted_route_witnesses",
                10,
                "joint_child_preflight.joint_command_graph.fresh_l0_suffix_readbacks_required",
                errors,
            )
        _require_bool(
            graph,
            "fresh_l0_and_l1_state_output_rollback_witnesses_required",
            True,
            "joint_child_preflight.joint_command_graph",
            errors,
        )
    boundary = _object(root.get("claim_boundary"), "joint_child_preflight.claim_boundary", errors)
    if boundary is not None:
        for field in (
            "complete_layer_or_token_allowed",
            "decoder_generation_hcli_tps_tg_or_tournament_allowed",
            "l1_suffix_or_moe_allowed",
            "automatic_retry_allowed",
        ):
            _require_bool(boundary, field, False, "joint_child_preflight.claim_boundary", errors)
    return child_sha


def _validate_l0_source_outer_preflight(
    outer: BoundDocument, errors: list[str]
) -> None:
    """Require the CPU-only source-token L0 authority that the joint host reuses.

    This is separate from the historical L0 handoff outer: it identifies the
    fresh source-token L0 23-dispatch constructor, while the historical outer
    remains baseline/provenance only.
    """
    root = outer.document
    route = _object(root.get("source_token_route"), "l0_source_outer_preflight.source_token_route", errors)
    if route is not None:
        _require_int(route, "token_id", SOURCE_TOKEN_ID, "l0_source_outer_preflight.source_token_route", errors)
        _require_int(route, "layer", L0_LAYER, "l0_source_outer_preflight.source_token_route", errors)
        _require_bool(
            route,
            "same_command_graph_required",
            True,
            "l0_source_outer_preflight.source_token_route",
            errors,
        )
        _require_bool(
            route,
            "zero_l0_state_required",
            True,
            "l0_source_outer_preflight.source_token_route",
            errors,
        )
        _require_bool(route, "all_ten_unique", True, "l0_source_outer_preflight.source_token_route", errors)
    boundary = _object(root.get("claim_boundary"), "l0_source_outer_preflight.claim_boundary", errors)
    if boundary is not None:
        _require_bool(boundary, "lease_issued", False, "l0_source_outer_preflight.claim_boundary", errors)
        _require_bool(
            boundary,
            "metal_device_or_dispatch_performed",
            False,
            "l0_source_outer_preflight.claim_boundary",
            errors,
        )
    next_child = _object(root.get("next_child_contract"), "l0_source_outer_preflight.next_child_contract", errors)
    if next_child is not None:
        _require_bool(
            next_child,
            "requires_same_tcb_prefix_lineage",
            True,
            "l0_source_outer_preflight.next_child_contract",
            errors,
        )
        _require_bool(
            next_child,
            "requires_source_token_authority_and_typed_bridge",
            True,
            "l0_source_outer_preflight.next_child_contract",
            errors,
        )


def _validate_joint_host_preflight(
    host: BoundDocument,
    *,
    child: BoundDocument,
    l0_source_outer: BoundDocument,
    child_sha: str,
    errors: list[str],
) -> None:
    """Bind the concrete compiled 23+9 host after the static-plan identity.

    The host is intentionally not part of the static child's upstream hash;
    that would create a circular identity.  It instead cross-binds the
    already-validated static child and exact source-token L0 preflight here.
    """
    root = host.document
    _require_bool(root, "child_started", False, "joint_host_preflight", errors)
    _require_bool(root, "metal_or_gpu_activity_performed", False, "joint_host_preflight", errors)
    _require_bool(root, "lease_issued_or_consumed", False, "joint_host_preflight", errors)
    _require_bool(root, "component_only", True, "joint_host_preflight", errors)
    _require_bool(
        root,
        "same_runtime_same_session_same_tcb_required",
        True,
        "joint_host_preflight",
        errors,
    )

    binary = _object(root.get("host_binary"), "joint_host_preflight.host_binary", errors)
    if binary is not None:
        observed_sha = _require_sha(binary, "sha256", "joint_host_preflight.host_binary", errors)
        if observed_sha is not None and observed_sha != child_sha:
            errors.append("joint_host_preflight.host_binary.sha256 must bind the joint child host SHA")

    static = _object(root.get("joint_static_plan"), "joint_host_preflight.joint_static_plan", errors)
    if static is not None:
        if static.get("document_sha256") != child.document_sha256:
            errors.append("joint_host_preflight.joint_static_plan.document_sha256 must bind joint child preflight")
        if static.get("seal_sha256") != child.document_seal_sha256:
            errors.append("joint_host_preflight.joint_static_plan.seal_sha256 must bind joint child preflight")

    l0_outer = _object(
        root.get("l0_source_outer_preflight"), "joint_host_preflight.l0_source_outer_preflight", errors
    )
    if l0_outer is not None:
        if l0_outer.get("document_sha256") != l0_source_outer.document_sha256:
            errors.append(
                "joint_host_preflight.l0_source_outer_preflight.document_sha256 must bind source-token L0 outer preflight"
            )
        if l0_outer.get("seal_sha256") != l0_source_outer.document_seal_sha256:
            errors.append(
                "joint_host_preflight.l0_source_outer_preflight.seal_sha256 must bind source-token L0 outer preflight"
            )

    graph = _object(root.get("joint_command_graph"), "joint_host_preflight.joint_command_graph", errors)
    if graph is not None:
        _require_int(graph, "source_token_id", SOURCE_TOKEN_ID, "joint_host_preflight.joint_command_graph", errors)
        _require_int(graph, "l0_dispatches", L0_TOTAL_DISPATCHES, "joint_host_preflight.joint_command_graph", errors)
        _require_int(
            graph,
            "l1_prefix_dispatches",
            L1_PREFIX_DISPATCHES,
            "joint_host_preflight.joint_command_graph",
            errors,
        )
        _require_int(graph, "total_dispatches", JOINT_TOTAL_DISPATCHES, "joint_host_preflight.joint_command_graph", errors)
        _require_bool(graph, "single_fence_required", True, "joint_host_preflight.joint_command_graph", errors)
        _require_bool(
            graph,
            "non_timed_token_command_buffer_required",
            True,
            "joint_host_preflight.joint_command_graph",
            errors,
        )
        if graph.get("tcb_trace_mode") != "off":
            errors.append("joint_host_preflight.joint_command_graph.tcb_trace_mode must be off")
        if graph.get("exact_l0_kernel_trace") != list(L0_TRUE_MOE_KERNEL_TRACE):
            errors.append("joint_host_preflight.joint_command_graph.exact_l0_kernel_trace must bind canonical L0 trace")
        if graph.get("exact_joint_kernel_trace") != list(JOINT_L0_L1_KERNEL_TRACE):
            errors.append("joint_host_preflight.joint_command_graph.exact_joint_kernel_trace must bind exact 32-kernel trace")
        if graph.get("opaque_capability_factory") != CANONICAL_L0_CAPABILITY_FACTORY:
            errors.append("joint_host_preflight.joint_command_graph.opaque_capability_factory drifted")
        if graph.get("runtime_api") != CANONICAL_L1_PREFIX_ENCODER:
            errors.append("joint_host_preflight.joint_command_graph.runtime_api drifted")
        if graph.get("consuming_finalizer") != CANONICAL_L0_L1_FINALIZER:
            errors.append("joint_host_preflight.joint_command_graph.consuming_finalizer drifted")

    body = _object(root.get("host_body"), "joint_host_preflight.host_body", errors)
    if body is not None:
        _require_bool(body, "strict_joint_entrypoint_compiled", True, "joint_host_preflight.host_body", errors)
        _require_bool(
            body,
            "future_outer_reaper_and_fresh_lease_required_before_entrypoint_may_run",
            True,
            "joint_host_preflight.host_body",
            errors,
        )
        _require_bool(
            body,
            "historical_l0_receipt_or_pinned_buffer_import_allowed",
            False,
            "joint_host_preflight.host_body",
            errors,
        )
        _require_bool(body, "l1_suffix_or_moe_authorized", False, "joint_host_preflight.host_body", errors)

    boundary = _object(root.get("claim_boundary"), "joint_host_preflight.claim_boundary", errors)
    if boundary is not None:
        _require_bool(
            boundary,
            "complete_layer_or_token_decoder_hcli_tps_tg_or_tournament_allowed",
            False,
            "joint_host_preflight.claim_boundary",
            errors,
        )
        _require_bool(
            boundary,
            "watcher_server_or_runtime_transition_allowed",
            False,
            "joint_host_preflight.claim_boundary",
            errors,
        )
        _require_bool(boundary, "automatic_retry_allowed", False, "joint_host_preflight.claim_boundary", errors)


def _claim_boundary(*, prepared: bool) -> dict[str, Any]:
    return {
        "cpu_only_authority_evaluation": True,
        "prepared": prepared,
        "child_spawned": False,
        "metal_or_gpu_activity_performed": False,
        "lease_issued_or_consumed": False,
        "runtime_server_watcher_or_hcli_started": False,
        "model_or_token_execution_performed": False,
        "tps_or_tg_measured": False,
        "tournament_activity_performed": False,
        "joint_child_execution_authorized_by_this_launcher": False,
        "cross_process_pinned_buffer_transfer_authorized": False,
    }


def preflight(input_document: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate only sealed CPU/file evidence and produce a plan or refusal."""
    errors: list[str] = []
    if input_document is None:
        source: dict[str, Any] = {}
        errors.append("sealed future input is absent")
    elif not isinstance(input_document, Mapping):
        source = {}
        errors.append("input must be an object")
    else:
        source = dict(input_document)
        try:
            verify(source, label="input")
        except SealIntegrityError as exc:
            errors.append(f"input has an invalid seal: {exc}")
        if source.get("schema") != INPUT_SCHEMA:
            errors.append(f"input.schema must be {INPUT_SCHEMA!r}")
        if source.get("status") != INPUT_STATUS:
            errors.append(f"input.status must be {INPUT_STATUS!r}")
        _require_bool(source, "joint_capture_requested", False, "input", errors)
        if source.get("joint_capture_evidence") not in (None, False):
            errors.append("input may not include joint L0/L1 execution evidence")

    readiness = _bound_document(source, "continuation_readiness", READINESS_SCHEMA, READINESS_STATUS, errors)
    l0_outer = _bound_document(source, "l0_outer_terminal", L0_OUTER_SCHEMA, L0_OUTER_STATUS, errors)
    l0_inner = _bound_document(source, "l0_inner_capture", L0_INNER_SCHEMA, L0_INNER_STATUS, errors)
    assessor = _bound_document(source, "l0_post_capture_assessor_binding", L0_ASSESSOR_BINDING_SCHEMA, L0_ASSESSOR_BINDING_STATUS, errors)
    assessment = _bound_document(source, "post_capture_assessment", L0_ASSESSMENT_SCHEMA, L0_ASSESSMENT_STATUS, errors)
    release = _bound_document(source, "lease_release_receipt", L0_RELEASE_SCHEMA, L0_RELEASE_STATUS, errors)
    manifest = _bound_document(source, "manifest", MANIFEST_SCHEMA, None, errors)
    admission = _bound_document(source, "admission_receipt", ADMISSION_RECEIPT_SCHEMA, ADMISSION_RECEIPT_STATUS, errors)
    schedule = _bound_document(source, "schedule", SCHEDULE_SCHEMA, SCHEDULE_STATUS, errors)
    child = _bound_document(source, "joint_l0_l1_child_preflight", JOINT_CHILD_PREFLIGHT_SCHEMA, JOINT_CHILD_PREFLIGHT_STATUS, errors)
    l0_source_outer = _bound_document(
        source,
        "l0_source_outer_preflight",
        L0_SOURCE_OUTER_PREFLIGHT_SCHEMA,
        L0_SOURCE_OUTER_PREFLIGHT_STATUS,
        errors,
    )
    host = _bound_document(
        source,
        "joint_l0_l1_host_preflight",
        JOINT_HOST_PREFLIGHT_SCHEMA,
        JOINT_HOST_PREFLIGHT_STATUS,
        errors,
    )

    facts: L0BaselineFacts | None = None
    if manifest is not None:
        _validate_manifest(manifest, errors)
    if admission is not None and manifest is not None:
        _validate_admission(admission, manifest, errors)
    if schedule is not None:
        _validate_schedule(schedule, errors)
    if l0_source_outer is not None:
        _validate_l0_source_outer_preflight(l0_source_outer, errors)
    if l0_inner is not None and manifest is not None and admission is not None:
        facts = _validate_l0_inner(l0_inner, manifest=manifest, admission=admission, errors=errors)
    if l0_outer is not None and l0_inner is not None:
        _validate_l0_outer(l0_outer, inner=l0_inner, errors=errors)
    if readiness is not None and l0_inner is not None and schedule is not None:
        _validate_readiness(readiness, inner=l0_inner, schedule=schedule, errors=errors)
    if (
        assessor is not None
        and l0_outer is not None
        and l0_inner is not None
        and assessment is not None
        and release is not None
        and manifest is not None
        and admission is not None
    ):
        _validate_assessor_binding(
            assessor,
            l0_outer=l0_outer,
            l0_inner=l0_inner,
            assessment=assessment,
            release=release,
            manifest=manifest,
            admission=admission,
            facts=facts,
            errors=errors,
        )

    supplied_child_sha = source.get("future_joint_l0_l1_child_sha256")
    if not _is_sha256(supplied_child_sha):
        errors.append("input.future_joint_l0_l1_child_sha256 must be an explicit lowercase SHA-256")
        supplied_child_sha = None
    if child is not None and all(
        value is not None
        for value in (
            readiness,
            l0_outer,
            l0_inner,
            assessor,
            assessment,
            release,
            manifest,
            admission,
            schedule,
            supplied_child_sha,
        )
    ):
        upstream_authority_identity = _upstream_authority_identity(
            readiness=readiness,  # type: ignore[arg-type]
            l0_outer=l0_outer,  # type: ignore[arg-type]
            l0_inner=l0_inner,  # type: ignore[arg-type]
            assessor_binding=assessor,  # type: ignore[arg-type]
            assessment=assessment,  # type: ignore[arg-type]
            lease_release=release,  # type: ignore[arg-type]
            manifest=manifest,  # type: ignore[arg-type]
            admission_receipt=admission,  # type: ignore[arg-type]
            schedule=schedule,  # type: ignore[arg-type]
            child_sha256=str(supplied_child_sha),
        )
        child_sha = _validate_joint_child_preflight(
            child, expected_preflight_identity=upstream_authority_identity, errors=errors
        )
        if child_sha is not None and child_sha != supplied_child_sha:
            errors.append("input.future_joint_l0_l1_child_sha256 does not match joint_l0_l1_child_preflight")
        if child_sha is not None and host is not None and l0_source_outer is not None:
            _validate_joint_host_preflight(
                host,
                child=child,
                l0_source_outer=l0_source_outer,
                child_sha=child_sha,
                errors=errors,
            )
    else:
        upstream_authority_identity = None
        child_sha = None

    errors = sorted(set(errors))
    prepared = not errors
    result = {
        "schema": RESULT_SCHEMA,
        "status": PREPARED_STATUS if prepared else REFUSED_STATUS,
        "prepared": prepared,
        "upstream_authority_identity_sha256": upstream_authority_identity if prepared else None,
        "future_joint_l0_l1_child_sha256": child_sha if prepared else None,
        "sealed_inputs": {
            "continuation_readiness": _public_identity(readiness),
            "l0_outer_terminal": _public_identity(l0_outer),
            "l0_inner_capture": _public_identity(l0_inner),
            "l0_post_capture_assessor_binding": _public_identity(assessor),
            "post_capture_assessment": _public_identity(assessment),
            "lease_release_receipt": _public_identity(release),
            "manifest": _public_identity(manifest),
            "admission_receipt": _public_identity(admission),
            "schedule": _public_identity(schedule),
            "joint_l0_l1_child_preflight": _public_identity(child),
            "l0_source_outer_preflight": _public_identity(l0_source_outer),
            "joint_l0_l1_host_preflight": _public_identity(host),
        },
        "baseline_l0_capture": (
            {
                "baseline_evidence_only": True,
                "captured_session_id": facts.session_id,
                "captured_output_f32le_sha256": facts.output_f32le_sha256,
                "captured_output_device_buffer_id": facts.output_device_buffer_id,
                "captured_output_elements": HIDDEN_ELEMENTS,
                "captured_output_bytes": HIDDEN_BYTES,
                "may_not_be_imported_by_future_child": True,
                "may_not_satisfy_same_runtime_or_same_tcb_for_future_child": True,
            }
            if prepared and facts is not None
            else None
        ),
        "required_assessment": {
            "earned_assessment_required_and_sealed": True,
            "assessment_result_bound": True,
            "schema": L0_ASSESSMENT_SCHEMA,
            "earned_status": L0_ASSESSMENT_STATUS,
            "must_bind_baseline_l0_outer_and_inner": True,
            "must_be_earned_before_any_future_joint_child_request": True,
        },
        "authorized_future_component_scope": {
            "source_token_id": SOURCE_TOKEN_ID,
            "l0_reencode_dispatches": L0_TOTAL_DISPATCHES,
            "l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
            "joint_total_dispatches": JOINT_TOTAL_DISPATCHES,
            "same_runtime_required": True,
            "same_tcb_required": True,
            "single_fence_after_l0_and_l1_prefix_required": True,
            "l1_layer": L1_LAYER,
            "l1_linear_state_slot": L1_SLOT,
            "exact_l1_prefix_dispatches": list(L1_PREFIX),
            "no_l1_suffix_or_moe_dispatch_authorized": True,
            "no_complete_layer_or_token_authorized": True,
        },
        "blockers": errors,
        "claim_boundary": _claim_boundary(prepared=prepared),
    }
    return seal(result)


def _read_input(path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        raise SourceTokenL1PrefixLauncherError("--input must be an absolute JSON path")
    raw = path.read_bytes()
    if len(raw) > MAX_INPUT_BYTES:
        raise SourceTokenL1PrefixLauncherError("--input exceeds the bounded JSON size")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceTokenL1PrefixLauncherError(f"--input is not JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SourceTokenL1PrefixLauncherError("--input must contain a JSON object")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    """Emit a sealed CPU-only preflight to stdout; no output/execution mode exists."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        for argument in arguments:
            flag = argument.split("=", 1)[0]
            if flag in FORBIDDEN_ARGUMENTS:
                raise SourceTokenL1PrefixLauncherError(
                    f"{flag} is forbidden: this module has no execution or lease path"
                )
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--input", type=Path, required=False)
        args = parser.parse_args(arguments)
        source = _read_input(args.input) if args.input is not None else None
        print(json.dumps(preflight(source), sort_keys=True))
        return 0
    except (SourceTokenL1PrefixLauncherError, OSError, ValueError) as exc:
        print(f"ascension_qwen80_l1_source_token_prefix_launcher: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
