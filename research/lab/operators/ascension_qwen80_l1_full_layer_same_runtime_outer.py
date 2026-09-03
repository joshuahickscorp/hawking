#!/usr/bin/env python3
"""CPU/file-only outer contract for future Qwen80 L0→L1 full-MoE capture.

The Rust host preflight binds the original sealed Layer-1 route authority,
the exact 46-kernel graph, and the future post-fence receipt ABI.  This
module independently verifies those bindings, seals a receipt-last/replay
outer preflight, and exposes a *fake-child-only* reaper for lifecycle tests.

The recovery canonicalization receipt is retained as audit provenance only.
It can prove why the historical raw inner authority remains a valid input,
but can never replace that raw authority as the route payload source.

It deliberately has no lease issuer, Metal context, device dispatch, catalog
scan, watcher, server, HCLI, TPS, or production execution path.  A later
one-shot runner must only be added after a real host ``--mode metal`` CLI is
independently preflighted against this document.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.receipts import SealIntegrityError, seal, verify


HOST_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_host_preflight.v1"
)
HOST_PREFLIGHT_STATUS = (
    "COMPILED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_COMPLETE_MOE_LAYER_"
    "SAME_RUNTIME_HOST_CPU_ONLY_NOT_LEASED_OR_EXECUTED"
)
OUTER_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_outer_preflight.v1"
)
OUTER_PREFLIGHT_STATUS = (
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_MOE_LAYER_"
    "SAME_RUNTIME_OUTER_CPU_ONLY_NOT_LEASED_OR_EXECUTED"
)
OUTER_REFUSED_STATUS = (
    "REFUSED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_MOE_LAYER_SAME_RUNTIME_OUTER_"
    "PRECONDITIONS_INCOMPLETE_NO_LEASE_OR_EXECUTION"
)
INNER_SCHEMA = "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_capture.v1"
INNER_STATUS = (
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_COMPLETE_MOE_LAYER_"
    "SAME_RUNTIME_COMPONENT_ONLY"
)
OUTER_CAPTURE_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_outer_capture.v1"
)
OUTER_CAPTURE_STATUS = (
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_SAME_RUNTIME_"
    "OUTER_TERMINAL_COMPONENT_ONLY"
)

JOINT_ASSESSMENT_SCHEMA = "hawking.ascension.qwen80_l0_l1_joint_post_capture_assessment.v1"
JOINT_ASSESSMENT_STATUS = (
    "EARNED_QWEN80_SOURCE_TOKEN_L0_L1_COMPONENT_NOT_FULL_LAYER_TOKEN_DECODER"
)
COMPLETION_PREFLIGHT_SCHEMA = "hawking.ascension.qwen80_source_token_l1_moe_completion_preflight.v1"
COMPLETION_PREFLIGHT_STATUS = (
    "PREFLIGHTED_QWEN80_SOURCE_TOKEN_L1_MOE_COMPLETION_COMPONENT_NOT_LEASED_OR_EXECUTED"
)
L0_OUTER_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_all_ten_true_moe_outer_preflight.v1"
)
L0_OUTER_PREFLIGHT_STATUS = (
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_OUTER_READY_FOR_"
    "SOURCE_TOKEN_CHILD_NOT_LEASED_OR_EXECUTED"
)
L1_ROUTE_AUTHORITY_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l1_all_ten_route_authority.v1"
)
L1_ROUTE_AUTHORITY_STATUS = (
    "SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L1_ALL_TEN_ROUTE_AUTHORITY_READY_FOR_"
    "SAME_RUNTIME_MOE_SUFFIX"
)
L1_ROUTE_AUTHORITY_RECOVERY_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l1_router_authority_"
    "recovery_canonicalization.v1"
)
L1_ROUTE_AUTHORITY_RECOVERY_STATUS = "RECOVERED_HISTORICAL_INNER_VALID_OUTER_REMAINS_REFUSED"

L0_DISPATCHES = 23
L1_PREFIX_DISPATCHES = 9
L1_MOE_SUFFIX_DISPATCHES = 14
TOTAL_DISPATCHES = 46
HIDDEN_ELEMENTS = 2048
HIDDEN_BYTES = 8192
MAX_JSON_BYTES = 100_000_000
MAX_STREAM_BYTES = 1_000_000

L0_KERNELS = (
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
)
L1_PREFIX_KERNELS = L0_KERNELS[:9]
L1_MOE_SUFFIX_KERNELS = L0_KERNELS[9:]
EXACT_KERNELS = L0_KERNELS + L1_PREFIX_KERNELS + L1_MOE_SUFFIX_KERNELS


class FullL1OuterError(RuntimeError):
    """The outer refuses an unbound or non-component capture contract."""


@dataclass(frozen=True)
class BoundDocument:
    path: Path
    raw_sha256: str
    bytes: int
    document: dict[str, Any]
    document_sha256: str
    document_seal_sha256: str


@dataclass(frozen=True)
class OuterInputs:
    host_preflight: Path
    host_binary: Path
    joint_assessment: Path
    l1_route_authority: Path
    l1_route_authority_recovery_provenance: Path
    completion_preflight: Path
    l0_source_outer_preflight: Path


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FullL1OuterError(f"{label} must be an object")
    return dict(value)


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FullL1OuterError(f"{label} must be an array")
    return list(value)


def _bool(doc: Mapping[str, Any], field: str, expected: bool, label: str) -> None:
    if doc.get(field) is not expected:
        raise FullL1OuterError(f"{label}.{field} must be {expected}")


def _int(doc: Mapping[str, Any], field: str, expected: int, label: str) -> None:
    if doc.get(field) != expected:
        raise FullL1OuterError(f"{label}.{field} must be {expected}")


def _text(doc: Mapping[str, Any], field: str, expected: str, label: str) -> None:
    if doc.get(field) != expected:
        raise FullL1OuterError(f"{label}.{field} drifted")


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise FullL1OuterError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FullL1OuterError(f"cannot stat {label}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise FullL1OuterError(f"{label} must be a regular non-symlink file")
    if executable and not os.access(path, os.X_OK):
        raise FullL1OuterError(f"{label} must be executable")
    return path.resolve(strict=True)


def _read_bound(path: Path, label: str, schema: str, status: str) -> BoundDocument:
    path = _canonical_regular(path, label)
    raw = path.read_bytes()
    if len(raw) > MAX_JSON_BYTES:
        raise FullL1OuterError(f"{label} exceeds bounded JSON size")
    try:
        document = _mapping(json.loads(raw.decode("utf-8")), label)
        verified = verify(document, label=label)
    except (UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise FullL1OuterError(f"{label} is not a valid sealed JSON document: {exc}") from exc
    if verified.get("schema") != schema or verified.get("status") != status:
        raise FullL1OuterError(f"{label} schema/status drifted")
    seal_sha256 = verified.get("seal_sha256")
    if not _is_sha(seal_sha256):
        raise FullL1OuterError(f"{label}.seal_sha256 must be a lowercase SHA-256")
    return BoundDocument(
        path=path,
        raw_sha256=_sha_bytes(raw),
        bytes=len(raw),
        document=verified,
        document_sha256=str(seal_sha256),
        document_seal_sha256=str(seal_sha256),
    )


def _file_evidence(path: Path, label: str, *, executable: bool = False) -> dict[str, Any]:
    path = _canonical_regular(path, label, executable=executable)
    raw = path.read_bytes()
    return {"path": str(path), "present": True, "bytes": len(raw), "sha256": _sha_bytes(raw)}


def _bound_evidence(bound: BoundDocument) -> dict[str, Any]:
    return {
        "path": str(bound.path),
        "present": True,
        "bytes": bound.bytes,
        "sha256": bound.raw_sha256,
        "document_sha256": bound.document_sha256,
        "document_seal_sha256": bound.document_seal_sha256,
    }


def _require_bound_object(value: object, expected: BoundDocument, label: str) -> None:
    value = _mapping(value, label)
    if (
        value.get("document_sha256") != expected.document_sha256
        or value.get("document_seal_sha256") != expected.document_seal_sha256
    ):
        raise FullL1OuterError(f"{label} document identity drifted")


def _require_full_bound_object(value: object, expected: BoundDocument, label: str) -> None:
    value = _mapping(value, label)
    if (
        value.get("path") != str(expected.path)
        or value.get("present") is not True
        or value.get("bytes") != expected.bytes
        or value.get("sha256") != expected.raw_sha256
    ):
        raise FullL1OuterError(f"{label} raw evidence drifted")
    _require_bound_object(value, expected, label)


def _validate_route_authority(authority: BoundDocument) -> tuple[list[int], list[float]]:
    root = authority.document
    evidence = _mapping(root.get("source_token_router_evidence"), "L1 route authority.router")
    ids = _array(evidence.get("source_stable_route_ids"), "L1 route authority route IDs")
    weights = _array(
        evidence.get("source_stable_normalized_weights"), "L1 route authority route weights"
    )
    if len(ids) != 10 or len(weights) != 10:
        raise FullL1OuterError("L1 route authority must retain exactly ten IDs/weights")
    if len(set(ids)) != 10 or any(not isinstance(value, int) or value < 0 for value in ids):
        raise FullL1OuterError("L1 route authority IDs must be ten unique unsigned values")
    if any(not isinstance(value, (int, float)) or not float(value) >= 0.0 for value in weights):
        raise FullL1OuterError("L1 route authority weights must be finite nonnegative numbers")
    fixed = _array(root.get("fixed_l1_payloads"), "L1 route authority fixed payloads")
    waves = _array(root.get("deterministic_waves"), "L1 route authority waves")
    if len(fixed) != 6 or len(waves) != 10:
        raise FullL1OuterError("L1 route authority must retain six fixed and ten route waves")
    for index, wave_value in enumerate(waves):
        wave = _mapping(wave_value, f"L1 route authority wave {index}")
        if wave.get("wave_index") != index or wave.get("layer") != 1 or wave.get("expert_id") != ids[index]:
            raise FullL1OuterError("L1 route authority wave ordering drifted")
        for kind in ("gate", "up", "down"):
            descriptor = _mapping(wave.get(kind), f"L1 route authority wave {index}.{kind}")
            for field in ("artifact_sha256", "direct_packed_payload_sha256", "header_sha256"):
                if not _is_sha(descriptor.get(field)):
                    raise FullL1OuterError(
                        f"L1 route authority wave {index}.{kind}.{field} is invalid"
                    )
    return [int(value) for value in ids], [float(value) for value in weights]


def _validate_recovery_provenance(
    recovery: BoundDocument, route_authority: BoundDocument
) -> None:
    """Require audit-only recovery evidence to bind the raw execution input.

    The raw historical inner remains the sole payload authority.  This
    prevents a recovery wrapper from being accidentally promoted to a dynamic
    route-authority substitute while retaining the historical refusal science.
    """
    root = recovery.document
    historical_inner = _mapping(
        root.get("historical_inner_authority"), "route-authority recovery historical inner"
    )
    expected = _bound_evidence(route_authority)
    for field in (
        "path",
        "present",
        "bytes",
        "sha256",
        "document_sha256",
        "document_seal_sha256",
    ):
        if historical_inner.get(field) != expected[field]:
            raise FullL1OuterError(
                f"route-authority recovery historical inner.{field} drifted"
            )
    _text(
        historical_inner,
        "schema",
        L1_ROUTE_AUTHORITY_SCHEMA,
        "route-authority recovery historical inner",
    )
    _text(
        historical_inner,
        "status",
        L1_ROUTE_AUTHORITY_STATUS,
        "route-authority recovery historical inner",
    )
    downstream = _mapping(root.get("downstream_authority"), "route-authority recovery downstream")
    _bool(
        downstream,
        "consume_historical_inner_directly",
        True,
        "route-authority recovery downstream",
    )
    _bool(
        downstream,
        "recovery_wrapper_is_not_a_dynamic_route_authority_substitute",
        True,
        "route-authority recovery downstream",
    )
    for field, expected_value in (
        ("authority_path", str(route_authority.path)),
        ("authority_document_sha256", route_authority.document_sha256),
        ("authority_seal_sha256", route_authority.document_seal_sha256),
        ("authority_schema", L1_ROUTE_AUTHORITY_SCHEMA),
        ("authority_status", L1_ROUTE_AUTHORITY_STATUS),
    ):
        if downstream.get(field) != expected_value:
            raise FullL1OuterError(f"route-authority recovery downstream.{field} drifted")
    canonicalization = _mapping(root.get("canonicalization"), "route-authority recovery")
    for field, expected_value in (
        ("historical_inner_validated_against_reaped_identity_chain", True),
        ("historical_outer_remains_refused", True),
        ("historical_outer_status_relabelled", False),
        ("no_new_scan_or_child", True),
        ("static_downstream_contract_valid", True),
        ("downstream_authority_is_historical_inner", True),
    ):
        _bool(canonicalization, field, expected_value, "route-authority recovery")


def _validate_exact_trace(value: object, label: str) -> None:
    trace = _array(value, label)
    observed = []
    for index, entry_value in enumerate(trace):
        entry = _mapping(entry_value, f"{label}[{index}]")
        if entry.get("ordinal") != index or not isinstance(entry.get("kernel"), str):
            raise FullL1OuterError(f"{label}[{index}] ordinal/kernel drifted")
        observed.append(entry["kernel"])
    if tuple(observed) != EXACT_KERNELS:
        raise FullL1OuterError(f"{label} is not the exact fresh 23+9+14 trace")


def _validate_host_preflight(
    host: BoundDocument,
    host_binary: Mapping[str, Any],
    joint: BoundDocument,
    route_authority: BoundDocument,
    completion: BoundDocument,
    l0_outer: BoundDocument,
) -> tuple[list[int], list[float]]:
    root = host.document
    recorded_binary = _mapping(root.get("host_binary"), "host preflight.host_binary")
    if (
        recorded_binary.get("path") != host_binary["path"]
        or recorded_binary.get("present") is not True
        or recorded_binary.get("bytes") != host_binary["bytes"]
        or recorded_binary.get("sha256") != host_binary["sha256"]
    ):
        raise FullL1OuterError("host preflight is not bound to the supplied current binary")
    _require_full_bound_object(
        root.get("joint_assessment"), joint, "host preflight.joint_assessment"
    )
    _require_full_bound_object(
        root.get("completion_preflight"), completion, "host preflight.completion_preflight"
    )
    _require_full_bound_object(
        root.get("l0_source_outer_preflight"), l0_outer, "host preflight.l0_source_outer_preflight"
    )
    ids, weights = _validate_route_authority(route_authority)
    authority = _mapping(root.get("l1_route_payload_authority"), "host preflight route authority")
    _text(authority, "schema", L1_ROUTE_AUTHORITY_SCHEMA, "host preflight route authority")
    _text(authority, "status", L1_ROUTE_AUTHORITY_STATUS, "host preflight route authority")
    binding = _mapping(authority.get("binding"), "host preflight route authority binding")
    if (
        binding.get("path") != str(route_authority.path)
        or binding.get("present") is not True
        or binding.get("bytes") != route_authority.bytes
        or binding.get("sha256") != route_authority.raw_sha256
        or binding.get("document_sha256") != route_authority.document_sha256
        or binding.get("document_seal_sha256") != route_authority.document_seal_sha256
    ):
        raise FullL1OuterError("host preflight route authority binding drifted")
    if authority.get("source_stable_route_ids") != ids or authority.get(
        "source_stable_normalized_weights"
    ) != weights:
        raise FullL1OuterError("host preflight route IDs/weights drifted")
    _int(authority, "distinct_payload_bindings", 36, "host preflight route authority")
    _int(authority, "route_guard_required_value", 1, "host preflight route authority")
    fixed = _array(authority.get("six_fixed_payloads"), "host preflight fixed payloads")
    waves = _array(authority.get("ten_ordered_waves"), "host preflight route waves")
    if len(fixed) != 6 or len(waves) != 10:
        raise FullL1OuterError("host preflight lost fixed or route payload evidence")
    interface = _mapping(root.get("future_same_runtime_host_interface"), "host preflight interface")
    _text(
        interface,
        "consuming_finalizer",
        "Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder::"
        "finalize_after_exact_l1_moe_completion_fence_with_readbacks",
        "host preflight interface",
    )
    for field in (
        "receipt_last_required",
        "fresh_runtime_required",
        "same_runtime_required",
        "same_token_command_buffer_required",
        "single_fence_required",
        "readbacks_after_fence_required",
    ):
        _bool(interface, field, True, "host preflight interface")
    _bool(
        interface,
        "cross_process_pinned_buffer_or_state_import_allowed",
        False,
        "host preflight interface",
    )
    metal_gate = _mapping(root.get("future_metal_entrypoint"), "host preflight Metal gate")
    for field in (
        "explicit_mode_required",
        "default_execution_disabled",
        "requires_new_full_l1_lease",
        "requires_sealed_outer_launch_authority",
        "requires_fresh_outer_and_inner_capture_directories",
        "self_hashes_current_executable",
        "no_device_execution_in_this_cpu_preflight",
    ):
        _bool(metal_gate, field, True, "host preflight Metal gate")
    _bool(metal_gate, "capture_body_wired", True, "host preflight Metal gate")
    graph = _mapping(root.get("future_joint_command_graph"), "host preflight graph")
    _int(graph, "source_token_id", 1, "host preflight graph")
    _int(graph, "l0_reencode_dispatches", L0_DISPATCHES, "host preflight graph")
    _int(graph, "l1_prefix_dispatches", L1_PREFIX_DISPATCHES, "host preflight graph")
    _int(graph, "l1_moe_suffix_dispatches", L1_MOE_SUFFIX_DISPATCHES, "host preflight graph")
    _int(graph, "total_dispatches", TOTAL_DISPATCHES, "host preflight graph")
    _bool(graph, "single_fence_after_all_dispatches_required", True, "host preflight graph")
    _bool(graph, "non_timed_structural_trace_required", True, "host preflight graph")
    _validate_exact_trace(graph.get("exact_kernel_trace"), "host preflight graph.exact_kernel_trace")
    receipt = _mapping(root.get("future_inner_receipt_contract"), "host preflight receipt contract")
    _text(receipt, "schema", INNER_SCHEMA, "host preflight receipt contract")
    _text(receipt, "status", INNER_STATUS, "host preflight receipt contract")
    _text(receipt, "outer_schema", OUTER_CAPTURE_SCHEMA, "host preflight receipt contract")
    _text(receipt, "outer_status", OUTER_CAPTURE_STATUS, "host preflight receipt contract")
    for field in (
        "requires_distinct_cpu_and_device_hashes_with_bounded_numeric_parity",
        "requires_l1_route_guard_all_ten_shared_routed_sum_and_second_residual_readbacks",
        "requires_l0_and_l1_active_rollback_state_witnesses",
    ):
        _bool(receipt, field, True, "host preflight receipt contract")
    boundary = _mapping(root.get("claim_boundary"), "host preflight boundary")
    _bool(
        boundary,
        "cpu_build_preflight_only",
        True,
        "host preflight boundary",
    )
    for field in (
        "catalog_or_payload_scan_performed",
        "metal_context_or_dispatch_performed",
        "lease_issued_or_consumed",
        "watcher_server_hcli_or_runtime_changed",
        "complete_layer_or_token_decoder_claim_earned",
        "tps_tg_or_tournament_claim_earned",
    ):
        _bool(boundary, field, False, "host preflight boundary")
    return ids, weights


def build_outer_preflight(inputs: OuterInputs) -> dict[str, Any]:
    """Validate current CPU/static authority files and seal an outer preflight.

    This has no subprocess or artifact/catalog operation.  The supplied host
    binary is read only for immutable file evidence; it is never executed.
    """
    host = _read_bound(
        inputs.host_preflight, "host preflight", HOST_PREFLIGHT_SCHEMA, HOST_PREFLIGHT_STATUS
    )
    host_binary = _file_evidence(inputs.host_binary, "host binary", executable=True)
    joint = _read_bound(
        inputs.joint_assessment,
        "joint assessment",
        JOINT_ASSESSMENT_SCHEMA,
        JOINT_ASSESSMENT_STATUS,
    )
    authority = _read_bound(
        inputs.l1_route_authority,
        "original L1 route authority",
        L1_ROUTE_AUTHORITY_SCHEMA,
        L1_ROUTE_AUTHORITY_STATUS,
    )
    recovery = _read_bound(
        inputs.l1_route_authority_recovery_provenance,
        "L1 route-authority recovery provenance",
        L1_ROUTE_AUTHORITY_RECOVERY_SCHEMA,
        L1_ROUTE_AUTHORITY_RECOVERY_STATUS,
    )
    completion = _read_bound(
        inputs.completion_preflight,
        "L1 completion preflight",
        COMPLETION_PREFLIGHT_SCHEMA,
        COMPLETION_PREFLIGHT_STATUS,
    )
    l0_outer = _read_bound(
        inputs.l0_source_outer_preflight,
        "L0 source outer preflight",
        L0_OUTER_PREFLIGHT_SCHEMA,
        L0_OUTER_PREFLIGHT_STATUS,
    )
    ids, weights = _validate_host_preflight(
        host, host_binary, joint, authority, completion, l0_outer
    )
    _validate_recovery_provenance(recovery, authority)
    return seal(
        {
            "schema": OUTER_PREFLIGHT_SCHEMA,
            "status": OUTER_PREFLIGHT_STATUS,
            "host_preflight": _bound_evidence(host),
            "host_binary": host_binary,
            "joint_assessment": _bound_evidence(joint),
            "original_l1_route_authority": _bound_evidence(authority),
            "l1_route_authority_recovery_provenance": _bound_evidence(recovery),
            "completion_preflight": _bound_evidence(completion),
            "l0_source_outer_preflight": _bound_evidence(l0_outer),
            "exact_component_scope": {
                "source_token_id": 1,
                "l0_reencode_dispatches": L0_DISPATCHES,
                "l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
                "l1_moe_suffix_dispatches": L1_MOE_SUFFIX_DISPATCHES,
                "total_dispatches": TOTAL_DISPATCHES,
                "one_fence_required": True,
                "non_timed_exact_trace_required": True,
                "kernel_names": list(EXACT_KERNELS),
                "route_ids": ids,
                "normalized_route_weights": weights,
                "six_fixed_payloads_required": True,
                "thirty_route_payloads_required": True,
            },
            "future_inner_receipt_contract": {
                "schema": INNER_SCHEMA,
                "status": INNER_STATUS,
                "outer_schema": OUTER_CAPTURE_SCHEMA,
                "outer_status": OUTER_CAPTURE_STATUS,
                "receipt_last_required": True,
                "fresh_l0_and_l1_state_rollback_readbacks_required": True,
                "separate_cpu_device_hashes_with_bounded_parity_required": True,
                "route_guard_value_required": 1,
            },
            "future_metal_entrypoint": {
                "explicit_mode_required": True,
                "default_execution_disabled": True,
                "requires_new_full_l1_lease": True,
                "requires_sealed_outer_launch_authority": True,
                "requires_fresh_outer_and_inner_capture_directories": True,
                "self_hashes_current_executable": True,
                "no_device_execution_in_this_cpu_preflight": True,
                "capture_body_wired": True,
            },
            "lifecycle": {
                "replay_guard_required": True,
                "one_child_process_required": True,
                "outer_reaped_terminal_required": True,
                "automatic_retry_authorized": False,
                "lease_or_device_execution_authorized_by_this_cpu_preflight": False,
                "real_host_metal_cli_available": True,
                "fake_child_reaper_test_only": True,
            },
            "claim_boundary": {
                "cpu_file_only": True,
                "catalog_or_payload_scan_performed": False,
                "metal_context_or_dispatch_performed": False,
                "lease_issued_or_consumed": False,
                "watcher_server_hcli_or_runtime_changed": False,
                "complete_layer_or_token_decoder_claim_earned": False,
                "tps_tg_or_tournament_claim_earned": False,
            },
        }
    )


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise FullL1OuterError("output must be absolute")
    if path.exists():
        raise FullL1OuterError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _bound_from_document(path: Path, label: str, schema: str, status: str) -> BoundDocument:
    return _read_bound(path, label, schema, status)


def validate_fake_child_inner(inner: Mapping[str, Any], outer: BoundDocument) -> None:
    """Validate enough of the future child receipt for fake reaper coverage.

    The validator is intentionally strict about the production ABI.  It is
    called only by tests until a separate, real host CLI is available.
    """
    try:
        root = verify(_mapping(inner, "fake child inner"), label="fake child inner")
    except SealIntegrityError as exc:
        raise FullL1OuterError(f"fake child inner seal invalid: {exc}") from exc
    if root.get("schema") != INNER_SCHEMA or root.get("status") != INNER_STATUS:
        raise FullL1OuterError("fake child inner schema/status drifted")
    provenance = _mapping(root.get("historical_component_provenance"), "fake child provenance")
    if (
        provenance.get("document_sha256") != outer.document["joint_assessment"]["document_sha256"]
        or provenance.get("document_seal_sha256")
        != outer.document["joint_assessment"]["document_seal_sha256"]
    ):
        raise FullL1OuterError("fake child historical assessment binding drifted")
    execution = _mapping(root.get("fresh_same_runtime_execution"), "fake child execution")
    for field in (
        "fresh_runtime",
        "fresh_session",
        "same_runtime",
        "same_tcb",
        "l0_reencoded_in_this_capture",
        "l1_prefix_and_moe_suffix_in_this_capture",
        "route_guard_enforced_before_l1_moe_suffix",
    ):
        _bool(execution, field, True, "fake child execution")
    for field, expected in (
        ("source_token_id", 1),
        ("l0_dispatches", L0_DISPATCHES),
        ("l1_prefix_dispatches", L1_PREFIX_DISPATCHES),
        ("l1_moe_suffix_dispatches", L1_MOE_SUFFIX_DISPATCHES),
        ("total_dispatches", TOTAL_DISPATCHES),
        ("fence_count", 1),
    ):
        _int(execution, field, expected, "fake child execution")
    for field in ("runtime_identity_sha256", "tcb_identity_sha256"):
        if not _is_sha(execution.get(field)):
            raise FullL1OuterError(f"fake child execution.{field} is invalid")
    trace = _mapping(root.get("structural_kernel_trace"), "fake child trace")
    _bool(trace, "non_timed", True, "fake child trace")
    _bool(trace, "exact_order", True, "fake child trace")
    if _array(trace.get("kernel_names"), "fake child trace kernels") != list(EXACT_KERNELS):
        raise FullL1OuterError("fake child trace drifted from exact 46 kernels")
    fence = _mapping(root.get("single_fence"), "fake child fence")
    for field, expected in (
        ("only_command_buffer_consumed", True),
        ("fence_succeeded", True),
        ("readbacks_after_fence", True),
        ("append_after_fence_possible", False),
    ):
        _bool(fence, field, expected, "fake child fence")
    _int(fence, "fence_count", 1, "fake child fence")
    route = _mapping(root.get("l1_route_payload_authority"), "fake child route authority")
    guard = _mapping(route.get("route_guard"), "fake child route guard")
    _bool(guard, "passed", True, "fake child route guard")
    _int(guard, "value", 1, "fake child route guard")
    if len(_array(route.get("route_payloads"), "fake child route payloads")) != 30:
        raise FullL1OuterError("fake child must retain thirty route payloads")
    readbacks = _mapping(root.get("l1_completion_readbacks"), "fake child readbacks")
    _int(readbacks, "layer", 1, "fake child readbacks")
    _int(readbacks, "slot", 1, "fake child readbacks")
    _int(readbacks, "output_elements", HIDDEN_ELEMENTS, "fake child readbacks")
    _int(readbacks, "output_bytes", HIDDEN_BYTES, "fake child readbacks")
    for field in (
        "input",
        "prefix_first_residual",
        "postnorm",
        "router_logits",
        "shared_output",
        "routed_sum",
        "second_residual_output",
        "active_conv",
        "active_recurrent",
        "rollback_conv",
        "rollback_recurrent",
    ):
        if field not in readbacks:
            raise FullL1OuterError(f"fake child missing L1 readback {field}")
    boundary = _mapping(root.get("claim_boundary"), "fake child boundary")
    _bool(boundary, "complete_l1_component_only", True, "fake child boundary")
    for field in (
        "token_generated",
        "decoder_started",
        "server_or_watcher_started",
        "tps_or_tg_measured",
        "tournament_started",
        "next_layer_executed",
    ):
        _bool(boundary, field, False, "fake child boundary")


def reap_fake_child_for_test(
    *,
    outer_preflight: Path,
    fake_child_command: Sequence[str],
    capture_dir: Path,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Run one disposable fake child and seal an outer terminal for tests.

    This is not reachable from the CLI.  The caller must explicitly pass a
    disposable fake child command; a real host binary is refused by the
    public CLI until a separate execution interface is ready.
    """
    outer = _bound_from_document(
        outer_preflight, "full-L1 outer preflight", OUTER_PREFLIGHT_SCHEMA, OUTER_PREFLIGHT_STATUS
    )
    if not fake_child_command:
        raise FullL1OuterError("fake child command is required")
    if capture_dir.exists():
        raise FullL1OuterError("fake child capture directory must be new")
    capture_dir.mkdir(parents=True, exist_ok=False)
    inner = capture_dir / "inner-receipt.json"
    running = seal(
        {
            "schema": "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_fake_reaper_running.v1",
            "status": "RUNNING_FAKE_CHILD_FOR_CPU_LIFECYCLE_TEST_ONLY",
            "outer_preflight": _bound_evidence(outer),
            "fake_child_only": True,
            "inner_receipt_path": str(inner),
        }
    )
    _write_new(capture_dir / "outer-running.json", running)
    completed = subprocess.run(
        [*fake_child_command, "--out", str(inner)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=False,
        timeout=timeout_seconds,
        check=False,
    )
    if len(completed.stdout) > MAX_STREAM_BYTES or len(completed.stderr) > MAX_STREAM_BYTES:
        raise FullL1OuterError("fake child stream exceeded bounded size")
    (capture_dir / "child.stdout.log").write_bytes(completed.stdout)
    (capture_dir / "child.stderr.log").write_bytes(completed.stderr)
    if completed.returncode != 0:
        raise FullL1OuterError(f"fake child exited {completed.returncode}")
    inner_bound = _read_bound(inner, "fake child inner", INNER_SCHEMA, INNER_STATUS)
    validate_fake_child_inner(inner_bound.document, outer)
    terminal = seal(
        {
            "schema": OUTER_CAPTURE_SCHEMA,
            "status": OUTER_CAPTURE_STATUS,
            "fixture_or_synthetic": False,
            "self_asserted": False,
            "inner_capture": _bound_evidence(inner_bound),
            "child_terminal": {
                "exit_code": completed.returncode,
                "reaped": True,
                "timed_out": False,
                "terminal_receipt_written_last": True,
                "automatic_retry_disabled": True,
                "lease_reuse_prohibited": True,
            },
            "test_only_fake_child": True,
            "claim_boundary": {
                "complete_l1_component_only": True,
                "token_generated": False,
                "decoder_started": False,
                "server_or_watcher_started": False,
                "tps_or_tg_measured": False,
                "tournament_started": False,
            },
        }
    )
    _write_new(capture_dir / "outer-terminal-receipt.json", terminal)
    return terminal


def _parse_args(arguments: Sequence[str]) -> tuple[OuterInputs, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-preflight", required=True, type=Path)
    parser.add_argument("--host-binary", required=True, type=Path)
    parser.add_argument("--joint-assessment", required=True, type=Path)
    parser.add_argument("--l1-route-authority", required=True, type=Path)
    parser.add_argument("--l1-route-authority-recovery-provenance", required=True, type=Path)
    parser.add_argument("--completion-preflight", required=True, type=Path)
    parser.add_argument("--l0-source-outer-preflight", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(arguments)
    return (
        OuterInputs(
            host_preflight=args.host_preflight,
            host_binary=args.host_binary,
            joint_assessment=args.joint_assessment,
            l1_route_authority=args.l1_route_authority,
            l1_route_authority_recovery_provenance=args.l1_route_authority_recovery_provenance,
            completion_preflight=args.completion_preflight,
            l0_source_outer_preflight=args.l0_source_outer_preflight,
        ),
        args.out,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        inputs, out = _parse_args(sys.argv[1:] if argv is None else argv)
        document = build_outer_preflight(inputs)
        _write_new(out, document)
    except (FullL1OuterError, OSError, ValueError) as exc:
        print(f"Qwen80 full-L1 outer preflight refused: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema": document["schema"],
                "status": document["status"],
                "seal_sha256": document["seal_sha256"],
                "child_spawned": False,
                "catalog_or_payload_scan_performed": False,
                "metal_or_gpu_activity_performed": False,
                "lease_issued_or_consumed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
