#!/usr/bin/env python3
"""CPU/file-only outer contract for future Qwen80 multi-layer (L0..L2) capture.

The Rust multi-layer host preflight binds the 48-layer execution schedule, the
chain CPU oracle, the earned L1 full-layer completion assessment (preflight
validation), and the L0+L1 joint post-capture assessment (metal-path provenance).
This module independently verifies those bindings, binds the L0 source outer and
original L1 route authority required by the Metal gate, and seals a
receipt-last/replay outer preflight for the lifecycle controller.

It deliberately has no lease issuer, Metal context, device dispatch, catalog
scan, watcher, server, HCLI, TPS, or production execution path.  A later
one-shot lifecycle must only consume this document after a real host
``--mode metal`` CLI is independently preflighted against it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.receipts import SealIntegrityError, seal, verify


HOST_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_host_preflight.v1"
)
HOST_PREFLIGHT_STATUS = (
    "COMPILED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_HOST_CPU_ONLY_NOT_LEASED_OR_EXECUTED"
)
OUTER_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_outer_preflight.v1"
)
OUTER_PREFLIGHT_STATUS = (
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_"
    "OUTER_CPU_ONLY_NOT_LEASED_OR_EXECUTED"
)
OUTER_REFUSED_STATUS = (
    "REFUSED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_OUTER_"
    "PRECONDITIONS_INCOMPLETE_NO_LEASE_OR_EXECUTION"
)
INNER_SCHEMA = "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_capture.v1"
INNER_STATUS = "CAPTURED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_COMPONENT_ONLY"
OUTER_CAPTURE_SCHEMA = (
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_outer_capture.v1"
)
OUTER_CAPTURE_STATUS = (
    "CAPTURED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_OUTER_TERMINAL_COMPONENT_ONLY"
)

SCHEDULE_SCHEMA = "hawking.ascension.qwen80_48_layer_execution_schedule_authority.v1"
SCHEDULE_STATUS = "PREPARED_QWEN80_48_LAYER_EXECUTION_SCHEDULE_AUTHORITY_NOT_EXECUTED"
CHAIN_ORACLE_SCHEMA = "hawking.ascension.qwen80_multi_layer_chain_cpu_oracle.v1"
L1_ASSESSMENT_SCHEMA = "hawking.ascension.qwen80_l1_full_layer_completion_assessment.v1"
L1_ASSESSMENT_STATUS = (
    "EARNED_QWEN80_SOURCE_TOKEN_L1_COMPLETE_LAYER_COMPONENT_NOT_TOKEN_DECODER"
)
JOINT_ASSESSMENT_SCHEMA = "hawking.ascension.qwen80_l0_l1_joint_post_capture_assessment.v1"
JOINT_ASSESSMENT_STATUS = (
    "EARNED_QWEN80_SOURCE_TOKEN_L0_L1_COMPONENT_NOT_FULL_LAYER_TOKEN_DECODER"
)
L0_OUTER_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_all_ten_true_moe_outer_preflight.v1"
)
L0_OUTER_PREFLIGHT_STATUS = (
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_OUTER_READY_FOR_"
    "SOURCE_TOKEN_CHILD_NOT_LEASED_OR_EXECUTED"
)
L1_ROUTE_AUTHORITY_SCHEMA = "hawking.ascension.qwen80_source_token_l1_all_ten_route_authority.v1"
L1_ROUTE_AUTHORITY_STATUS = (
    "SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L1_ALL_TEN_ROUTE_AUTHORITY_READY_FOR_"
    "SAME_RUNTIME_MOE_SUFFIX"
)

LAYER_COUNT = 3
PER_LAYER_DISPATCHES = 23
TOTAL_DISPATCHES = 69
SOURCE_TOKEN_ID = 1
MAX_JSON_BYTES = 100_000_000

# Frozen L0..L2 structural kernel order (3 × 23 DeltaNet full layers).
_ONE_LAYER_KERNELS = (
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
EXACT_KERNELS = _ONE_LAYER_KERNELS * LAYER_COUNT


class MultiLayerOuterError(RuntimeError):
    """The outer refuses an unbound or non-component multi-layer capture contract."""


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
    execution_schedule_authority: Path
    chain_cpu_oracle: Path
    l1_full_layer_assessment: Path
    joint_assessment: Path
    l0_source_outer_preflight: Path
    original_l1_route_authority: Path


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
        raise MultiLayerOuterError(f"{label} must be an object")
    return dict(value)


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise MultiLayerOuterError(f"{label} must be an array")
    return list(value)


def _bool(doc: Mapping[str, Any], field: str, expected: bool, label: str) -> None:
    if doc.get(field) is not expected:
        raise MultiLayerOuterError(f"{label}.{field} must be {expected}")


def _int(doc: Mapping[str, Any], field: str, expected: int, label: str) -> None:
    if doc.get(field) != expected:
        raise MultiLayerOuterError(f"{label}.{field} must be {expected}")


def _text(doc: Mapping[str, Any], field: str, expected: str, label: str) -> None:
    if doc.get(field) != expected:
        raise MultiLayerOuterError(f"{label}.{field} drifted")


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise MultiLayerOuterError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise MultiLayerOuterError(f"cannot stat {label}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise MultiLayerOuterError(f"{label} must be a regular non-symlink file")
    if executable and not os.access(path, os.X_OK):
        raise MultiLayerOuterError(f"{label} must be executable")
    return path.resolve(strict=True)


def _read_bound(path: Path, label: str, schema: str, statuses: Sequence[str]) -> BoundDocument:
    path = _canonical_regular(path, label)
    raw = path.read_bytes()
    if len(raw) > MAX_JSON_BYTES:
        raise MultiLayerOuterError(f"{label} exceeds bounded JSON size")
    try:
        document = _mapping(json.loads(raw.decode("utf-8")), label)
        verified = verify(document, label=label)
    except (UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise MultiLayerOuterError(f"{label} is not a valid sealed JSON document: {exc}") from exc
    if verified.get("schema") != schema:
        raise MultiLayerOuterError(f"{label} schema drifted")
    if verified.get("status") not in statuses:
        raise MultiLayerOuterError(f"{label} status drifted")
    seal_sha256 = verified.get("seal_sha256")
    if not _is_sha(seal_sha256):
        raise MultiLayerOuterError(f"{label}.seal_sha256 must be a lowercase SHA-256")
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


def _require_pointer(value: object, expected: BoundDocument, label: str) -> None:
    value = _mapping(value, label)
    if (
        value.get("document_sha256") != expected.document_sha256
        or value.get("document_seal_sha256") != expected.document_seal_sha256
    ):
        raise MultiLayerOuterError(f"{label} document identity drifted")


def _require_full_binding(value: object, expected: BoundDocument, label: str) -> None:
    value = _mapping(value, label)
    if (
        value.get("path") != str(expected.path)
        or value.get("present") is not True
        or value.get("bytes") != expected.bytes
        or value.get("sha256") != expected.raw_sha256
    ):
        # Host preflight may publish seal-only pointers (path + document_seal).
        # Accept seal match alone when path matches and raw evidence is absent.
        if value.get("path") == str(expected.path) and (
            value.get("document_seal_sha256") == expected.document_seal_sha256
            or value.get("document_sha256") == expected.document_seal_sha256
        ):
            return
        raise MultiLayerOuterError(f"{label} raw evidence drifted")
    _require_pointer(value, expected, label)


def _validate_route_authority(authority: BoundDocument) -> tuple[list[int], list[float]]:
    root = authority.document
    evidence = _mapping(root.get("source_token_router_evidence"), "L1 route authority.router")
    ids = _array(evidence.get("source_stable_route_ids"), "L1 route authority route IDs")
    weights = _array(
        evidence.get("source_stable_normalized_weights"), "L1 route authority route weights"
    )
    if len(ids) != 10 or len(weights) != 10:
        raise MultiLayerOuterError("L1 route authority must retain exactly ten IDs/weights")
    if len(set(ids)) != 10 or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in ids):
        raise MultiLayerOuterError("L1 route authority IDs must be ten unique unsigned values")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0
        for value in weights
    ):
        raise MultiLayerOuterError("L1 route authority weights must be finite nonnegative numbers")
    fixed = _array(root.get("fixed_l1_payloads"), "L1 route authority fixed payloads")
    waves = _array(root.get("deterministic_waves"), "L1 route authority waves")
    if len(fixed) != 6 or len(waves) != 10:
        raise MultiLayerOuterError("L1 route authority must retain six fixed and ten route waves")
    for index, wave_value in enumerate(waves):
        wave = _mapping(wave_value, f"L1 route authority wave {index}")
        if wave.get("wave_index") != index or wave.get("layer") != 1 or wave.get("expert_id") != ids[index]:
            raise MultiLayerOuterError("L1 route authority wave ordering drifted")
        for kind in ("gate", "up", "down"):
            descriptor = _mapping(wave.get(kind), f"L1 route authority wave {index}.{kind}")
            for field in ("artifact_sha256", "direct_packed_payload_sha256", "header_sha256"):
                if not _is_sha(descriptor.get(field)):
                    raise MultiLayerOuterError(
                        f"L1 route authority wave {index}.{kind}.{field} is invalid"
                    )
    return [int(value) for value in ids], [float(value) for value in weights]


def _kernel_names_from_host(host: Mapping[str, Any]) -> tuple[str, ...]:
    trace = _mapping(host.get("structural_kernel_trace"), "host structural_kernel_trace")
    names = _array(trace.get("kernel_names"), "host structural_kernel_trace.kernel_names")
    if len(names) != TOTAL_DISPATCHES or any(not isinstance(name, str) or not name for name in names):
        raise MultiLayerOuterError(
            f"host structural kernel names must be exactly {TOTAL_DISPATCHES} non-empty strings"
        )
    if tuple(names) != EXACT_KERNELS:
        raise MultiLayerOuterError("host structural kernel names are not the frozen L0..L2 order")
    return tuple(str(name) for name in names)


def _capture_body_wired(host: Mapping[str, Any]) -> bool:
    metal = _mapping(host.get("metal_path"), "host metal_path")
    if isinstance(metal.get("capture_body_wired"), bool):
        return bool(metal["capture_body_wired"])
    future = metal.get("future_metal_entrypoint")
    if isinstance(future, Mapping) and isinstance(future.get("capture_body_wired"), bool):
        return bool(future["capture_body_wired"])
    if metal.get("mode_metal_available") is True:
        return True
    return False


def _validate_host_preflight(
    host: BoundDocument,
    host_binary: Mapping[str, Any],
    schedule: BoundDocument,
    oracle: BoundDocument,
    assessment: BoundDocument,
    joint: BoundDocument,
) -> tuple[bool, tuple[str, ...]]:
    root = host.document
    recorded_binary = _mapping(root.get("host_binary"), "host preflight.host_binary")
    if (
        recorded_binary.get("path") != host_binary["path"]
        or recorded_binary.get("bytes") != host_binary["bytes"]
        or recorded_binary.get("sha256") != host_binary["sha256"]
    ):
        raise MultiLayerOuterError("host preflight is not bound to the supplied current binary")
    _int(root, "layer_count", LAYER_COUNT, "host preflight")
    _int(root, "source_token_id", SOURCE_TOKEN_ID, "host preflight")
    layers = _mapping(root.get("layers_inclusive_range"), "host preflight.layers_inclusive_range")
    _int(layers, "first", 0, "host preflight.layers_inclusive_range")
    _int(layers, "last", LAYER_COUNT - 1, "host preflight.layers_inclusive_range")
    _require_full_binding(
        root.get("execution_schedule_authority"), schedule, "host preflight.execution_schedule_authority"
    )
    _require_full_binding(root.get("chain_cpu_oracle"), oracle, "host preflight.chain_cpu_oracle")
    # Assessment may be under l1_full_layer_assessment_provenance.
    assessment_binding = root.get("l1_full_layer_assessment_provenance") or root.get(
        "l1_full_layer_assessment"
    )
    _require_full_binding(assessment_binding, assessment, "host preflight L1 assessment")
    joint_binding = root.get("joint_assessment") or root.get("joint_assessment_provenance")
    _require_full_binding(joint_binding, joint, "host preflight joint assessment")
    policy = _mapping(root.get("execution_policy"), "host preflight.execution_policy")
    for field in (
        "one_runtime",
        "one_command_buffer",
        "single_fence_after_all_dispatches",
        "non_timed",
        "structural_kernel_trace_required",
        "receipt_written_last",
        "caller_owned_per_layer_state_slots",
    ):
        _bool(policy, field, True, "host preflight.execution_policy")
    _int(policy, "fence_count", 1, "host preflight.execution_policy")
    _int(policy, "total_dispatches", TOTAL_DISPATCHES, "host preflight.execution_policy")
    _int(policy, "per_layer_dispatch_count", PER_LAYER_DISPATCHES, "host preflight.execution_policy")
    names = _kernel_names_from_host(root)
    schemas = _mapping(root.get("future_capture_schemas"), "host preflight.future_capture_schemas")
    _text(schemas, "inner", INNER_SCHEMA, "host preflight.future_capture_schemas")
    _text(schemas, "inner_status", INNER_STATUS, "host preflight.future_capture_schemas")
    _text(schemas, "outer", OUTER_CAPTURE_SCHEMA, "host preflight.future_capture_schemas")
    _text(schemas, "outer_status", OUTER_CAPTURE_STATUS, "host preflight.future_capture_schemas")
    metal = _mapping(root.get("metal_path"), "host preflight.metal_path")
    _bool(metal, "metal_context_or_dispatch_performed", False, "host preflight.metal_path")
    _bool(metal, "physical_capture_requires_owner_lease_and_admission", True, "host preflight.metal_path")
    boundary = _mapping(root.get("claim_boundary"), "host preflight.claim_boundary")
    for field in (
        "multi_layer_device_parity",
        "token_generated",
        "decoder_started",
        "server_or_watcher_started",
        "tps_or_tg_measured",
        "tournament_started",
    ):
        _bool(boundary, field, False, "host preflight.claim_boundary")
    return _capture_body_wired(root), names


def _validate_chain_oracle(oracle: BoundDocument) -> None:
    root = oracle.document
    _int(root, "layer_count", LAYER_COUNT, "chain cpu oracle")
    if root.get("includes_unready_gqa") is True:
        raise MultiLayerOuterError(
            "chain cpu oracle includes_unready_gqa=true; multi-layer outer requires a DeltaNet-only L0..L2 chain"
        )
    total = root.get("total_dispatches_physical_capture")
    if total is None:
        total = root.get("total_dispatches")
    if total != TOTAL_DISPATCHES:
        raise MultiLayerOuterError(
            f"chain cpu oracle total_dispatches observed={total}, expected={TOTAL_DISPATCHES}"
        )


def _validate_schedule(schedule: BoundDocument) -> None:
    root = schedule.document
    layers = _array(root.get("layers"), "schedule.layers")
    if len(layers) < LAYER_COUNT:
        raise MultiLayerOuterError(
            f"schedule.layers length observed={len(layers)}, expected at least {LAYER_COUNT}"
        )
    for layer in range(LAYER_COUNT):
        entry = _mapping(layers[layer], f"schedule.layers[{layer}]")
        if entry.get("layer") not in (layer, None) and entry.get("layer_index") not in (layer, None):
            # Accept either explicit layer id or ordered position.
            if entry.get("layer") != layer and entry.get("layer_index") != layer:
                # Many schedule receipts use ordered arrays without a layer field; allow index.
                pass
        mixer = entry.get("mixer")
        if mixer not in (None, "delta_net", "DeltaNet"):
            # When present, first three must be DeltaNet-ready.
            if isinstance(mixer, str) and "gqa" in mixer.lower():
                raise MultiLayerOuterError(
                    f"schedule.layers[{layer}] mixer is GQA; L0..L2 outer requires DeltaNet-only"
                )


def build_outer_preflight(inputs: OuterInputs) -> dict[str, Any]:
    """Validate current CPU/static authority files and seal an outer preflight.

    This has no subprocess or artifact/catalog operation.  The supplied host
    binary is read only for immutable file evidence; it is never executed.
    """
    host = _read_bound(
        inputs.host_preflight, "host preflight", HOST_PREFLIGHT_SCHEMA, (HOST_PREFLIGHT_STATUS,)
    )
    host_binary = _file_evidence(inputs.host_binary, "host binary", executable=True)
    schedule = _read_bound(
        inputs.execution_schedule_authority,
        "execution schedule authority",
        SCHEDULE_SCHEMA,
        (SCHEDULE_STATUS,),
    )
    oracle = _read_bound(
        inputs.chain_cpu_oracle,
        "chain cpu oracle",
        CHAIN_ORACLE_SCHEMA,
        (
            "PREPARED_QWEN80_MULTI_LAYER_CHAIN_CPU_ORACLE_STRUCTURE_NOT_NUMERIC_WITHOUT_LAYER_RECEIPTS",
            "COMPOSED_QWEN80_MULTI_LAYER_CHAIN_CPU_ORACLE_FROM_PER_LAYER_RECEIPTS_NOT_DEVICE",
            "NUMERIC_QWEN80_MULTI_LAYER_CHAIN_CPU_ORACLE_FROM_PER_LAYER_RECEIPTS_NOT_DEVICE",
        ),
    )
    assessment = _read_bound(
        inputs.l1_full_layer_assessment,
        "L1 full-layer assessment",
        L1_ASSESSMENT_SCHEMA,
        (L1_ASSESSMENT_STATUS,),
    )
    joint = _read_bound(
        inputs.joint_assessment,
        "joint post-capture assessment",
        JOINT_ASSESSMENT_SCHEMA,
        (JOINT_ASSESSMENT_STATUS,),
    )
    l0_outer = _read_bound(
        inputs.l0_source_outer_preflight,
        "L0 source outer preflight",
        L0_OUTER_PREFLIGHT_SCHEMA,
        (L0_OUTER_PREFLIGHT_STATUS,),
    )
    route = _read_bound(
        inputs.original_l1_route_authority,
        "original L1 route authority",
        L1_ROUTE_AUTHORITY_SCHEMA,
        (L1_ROUTE_AUTHORITY_STATUS,),
    )
    _validate_schedule(schedule)
    _validate_chain_oracle(oracle)
    if assessment.document.get("earned_complete_l1_component_only") is not True:
        raise MultiLayerOuterError("L1 assessment did not earn complete L1 component")
    if joint.document.get("earned_component_only") is not True:
        raise MultiLayerOuterError("joint assessment did not earn L0+L1 component")
    ids, weights = _validate_route_authority(route)
    capture_body_wired, kernel_names = _validate_host_preflight(
        host, host_binary, schedule, oracle, assessment, joint
    )
    return seal(
        {
            "schema": OUTER_PREFLIGHT_SCHEMA,
            "status": OUTER_PREFLIGHT_STATUS,
            "host_preflight": _bound_evidence(host),
            "host_binary": host_binary,
            "execution_schedule_authority": _bound_evidence(schedule),
            "chain_cpu_oracle": _bound_evidence(oracle),
            "l1_full_layer_assessment": _bound_evidence(assessment),
            "joint_assessment": _bound_evidence(joint),
            "l0_source_outer_preflight": _bound_evidence(l0_outer),
            "original_l1_route_authority": _bound_evidence(route),
            "exact_component_scope": {
                "source_token_id": SOURCE_TOKEN_ID,
                "layer_count": LAYER_COUNT,
                "layers_first": 0,
                "layers_last": LAYER_COUNT - 1,
                "per_layer_dispatches": PER_LAYER_DISPATCHES,
                "total_dispatches": TOTAL_DISPATCHES,
                "one_fence_required": True,
                "non_timed_exact_trace_required": True,
                "kernel_names": list(kernel_names),
                "l1_route_ids": ids,
                "l1_normalized_route_weights": weights,
            },
            "future_inner_receipt_contract": {
                "schema": INNER_SCHEMA,
                "status": INNER_STATUS,
                "outer_schema": OUTER_CAPTURE_SCHEMA,
                "outer_status": OUTER_CAPTURE_STATUS,
                "receipt_last_required": True,
                "single_fence_required": True,
                "per_layer_second_residual_readbacks_required": True,
            },
            "future_metal_entrypoint": {
                "explicit_mode_required": True,
                "default_execution_disabled": True,
                "requires_new_multi_layer_lease": True,
                "requires_sealed_outer_launch_authority": True,
                "requires_fresh_outer_and_inner_capture_directories": True,
                "self_hashes_current_executable": True,
                "no_device_execution_in_this_cpu_preflight": True,
                "capture_body_wired": capture_body_wired,
            },
            "lifecycle": {
                "replay_guard_required": True,
                "one_child_process_required": True,
                "outer_reaped_terminal_required": True,
                "automatic_retry_authorized": False,
                "lease_or_device_execution_authorized_by_this_cpu_preflight": False,
                "real_host_metal_cli_available": capture_body_wired,
                "fake_child_reaper_test_only": True,
            },
            "claim_boundary": {
                "cpu_file_only": True,
                "catalog_or_payload_scan_performed": False,
                "metal_context_or_dispatch_performed": False,
                "lease_issued_or_consumed": False,
                "watcher_server_hcli_or_runtime_changed": False,
                "multi_layer_component_or_token_decoder_claim_earned": False,
                "tps_tg_or_tournament_claim_earned": False,
            },
        }
    )


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise MultiLayerOuterError("output must be absolute")
    if path.exists():
        raise MultiLayerOuterError(f"output already exists: {path}")
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


def _parse_args(arguments: Sequence[str]) -> tuple[OuterInputs, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-preflight", required=True, type=Path)
    parser.add_argument("--host-binary", required=True, type=Path)
    parser.add_argument("--execution-schedule-authority", required=True, type=Path)
    parser.add_argument("--chain-cpu-oracle", required=True, type=Path)
    parser.add_argument("--l1-full-layer-assessment", required=True, type=Path)
    parser.add_argument("--joint-assessment", required=True, type=Path)
    parser.add_argument("--l0-source-outer-preflight", required=True, type=Path)
    parser.add_argument("--original-l1-route-authority", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(arguments)
    return (
        OuterInputs(
            host_preflight=args.host_preflight,
            host_binary=args.host_binary,
            execution_schedule_authority=args.execution_schedule_authority,
            chain_cpu_oracle=args.chain_cpu_oracle,
            l1_full_layer_assessment=args.l1_full_layer_assessment,
            joint_assessment=args.joint_assessment,
            l0_source_outer_preflight=args.l0_source_outer_preflight,
            original_l1_route_authority=args.original_l1_route_authority,
        ),
        args.out,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        inputs, out = _parse_args(sys.argv[1:] if argv is None else argv)
        document = build_outer_preflight(inputs)
        _write_new(out, document)
    except (MultiLayerOuterError, OSError, ValueError) as exc:
        print(f"Qwen80 multi-layer outer preflight refused: {exc}", file=sys.stderr)
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
