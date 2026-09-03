#!/usr/bin/env python3
"""Fail-closed lifecycle for the future Qwen80 same-runtime L0→L1 prefix.

This module is intentionally CPU/file-only today.  It seals the independent
23+9 outer preflight, defines a new joint-only lease grammar, and supplies the
receipt/reaper validators that a later host capture interface must satisfy.
The currently compiled Rust host exposes preflight only, so this module cannot
issue a live lease or spawn it.  Tests exercise the reaper with a disposable
fake child only; no test fixture is a physical-Qwen80 result.
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import ascension_qwen80_l1_source_token_prefix_launcher as independent
from lab.receipts import SealIntegrityError, seal, verify


OUTER_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_outer_preflight.v1"
)
OUTER_PREFLIGHT_STATUS = (
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_"
    "OUTER_CPU_ONLY_NOT_LEASED_OR_EXECUTED"
)
OUTER_PREFLIGHT_REFUSED_STATUS = (
    "REFUSED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_OUTER_CPU_ONLY_"
    "PRECONDITIONS_INCOMPLETE_NO_LEASE_OR_EXECUTION"
)
INNER_SCHEMA = "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_capture.v1"
INNER_STATUS = (
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_SLOT1_PREFIX_"
    "SAME_RUNTIME_COMPONENT_ONLY"
)
OUTER_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_outer_capture.v1"
)
OUTER_STATUS = (
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_OUTER_TERMINAL_COMPONENT_ONLY"
)
OUTER_REFUSED_PREFIX = "REFUSED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_OUTER_"
LEASE_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_quiet_metal_lease.v1"
)
LEASE_STATUS = (
    "GRANTED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_COMPONENT_QUIET_METAL_LEASE"
)
RELEASE_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_"
    "quiet_metal_lease_release.v1"
)
RELEASE_STATUS = (
    "RELEASED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_COMPONENT_QUIET_METAL_"
    "LEASE_AFTER_TERMINAL_CAPTURE"
)
EXECUTION_BINDING_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_"
    "host_execution_binding.v1"
)
EXECUTION_BINDING_STATUS = (
    "PREPARED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_STRICT_HOST_EXECUTION_INTERFACE"
)
OUTER_LAUNCH_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_"
    "outer_launch_authority.v1"
)
OUTER_LAUNCH_STATUS = (
    "AUTHORIZED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_OUTER_REAPED_ONE_SHOT_METAL_CHILD"
)
WATCHER_HOLD_SCHEMA = "hawking.ascension.qwen80.watcher_gpu_coordination_hold.v1"
WATCHER_HOLD_STATUS = "HELD_QWEN80_RESPAWNED_STATE_CHILD_BEFORE_UNGUARDED_METAL_FIXTURE"

PREFLIGHT_INPUT_FILENAME = "independent-preflight-input.json"
PREFLIGHT_RESULT_FILENAME = "independent-preflight.json"
OUTER_PREFLIGHT_FILENAME = "outer-preflight.json"
OUTER_LAUNCH_FILENAME = "outer-launch-authority.json"
OUTER_TERMINAL_FILENAME = "outer-terminal-receipt.json"
INNER_DIRNAME = "inner"
OUTER_STDOUT_FILENAME = "outer-child.stdout.log"
OUTER_STDERR_FILENAME = "outer-child.stderr.log"
RUNNING_FILENAME = "outer-running.json"
CHILD_FILENAME = "child.json"
MAX_JSON_BYTES = 100_000_000
MAX_STREAM_BYTES = 1_000_000
MAX_PARITY_ERROR = 1.0e-3

L0_DISPATCHES = 23
L1_DISPATCHES = 9
TOTAL_DISPATCHES = 32
SOURCE_TOKEN_ID = 1
HIDDEN_ELEMENTS = 2_048
HIDDEN_BYTES = 8_192
L0_CONV_BYTES = 98_304
L0_RECURRENT_BYTES = 2_097_152
L1_CONV_CAPACITY_BYTES = 196_608
L1_RECURRENT_CAPACITY_BYTES = 4_194_304

CAPABILITY_FACTORY = (
    "Qwen80CompleteNativeRuntime::certify_source_token_l0_true_moe_continuation"
)
L1_ENCODER = (
    "Qwen80CompleteNativeRuntime::"
    "encode_source_token_l1_deltanet_prefix_from_canonical_l0_continuation_into"
)
FINALIZER = (
    "Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder::finalize_after_exact_joint_fence"
)

STRUCTURAL_KERNELS: tuple[str, ...] = tuple(
    entry["kernel"] for entry in independent.JOINT_L0_L1_KERNEL_TRACE
)


class JointLifecycleError(RuntimeError):
    """An authority, lifecycle, or receipt boundary failed closed."""


@dataclass(frozen=True)
class BoundDocument:
    path: Path
    raw_sha256: str
    bytes: int
    document: dict[str, Any]
    document_sha256: str
    document_seal_sha256: str


@dataclass(frozen=True)
class PreflightPaths:
    continuation_readiness: Path
    l0_outer_terminal: Path
    l0_inner_capture: Path
    assessor_binding: Path
    post_capture_assessment: Path
    prior_lease_release: Path
    manifest: Path
    admission_receipt: Path
    schedule: Path
    joint_static_plan: Path
    l0_source_outer_preflight: Path
    joint_host_preflight: Path


@dataclass(frozen=True)
class LeaseContext:
    outer_preflight: BoundDocument
    execution_binding: BoundDocument
    watcher_hold_path: Path
    watcher_hold_evidence: dict[str, Any]
    lease: BoundDocument
    lease_id: str


@dataclass(frozen=True)
class CaptureConfig:
    lease_receipt: Path
    outer_preflight: Path
    execution_binding: Path
    watcher_hold: Path
    capture_dir: Path
    timeout_seconds: float
    workers: int
    child_command: tuple[str, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
        raise JointLifecycleError(f"{label} must be an object")
    return dict(value)


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise JointLifecycleError(f"{label} must be an array")
    return list(value)


def _require_bool(document: Mapping[str, Any], field: str, expected: bool, label: str) -> None:
    if document.get(field) is not expected:
        raise JointLifecycleError(f"{label}.{field} must be {expected}")


def _require_int(document: Mapping[str, Any], field: str, expected: int, label: str) -> None:
    if document.get(field) != expected:
        raise JointLifecycleError(f"{label}.{field} must be {expected}")


def _require_sha(document: Mapping[str, Any], field: str, label: str) -> str:
    value = document.get(field)
    if not _is_sha(value):
        raise JointLifecycleError(f"{label}.{field} must be a lowercase SHA-256")
    return str(value)


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise JointLifecycleError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise JointLifecycleError(f"cannot stat {label}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise JointLifecycleError(f"{label} must be a regular non-symlink file")
    if executable and not os.access(path, os.X_OK):
        raise JointLifecycleError(f"{label} must be executable")
    return path.resolve(strict=True)


def _read_bound(path: Path, label: str, schema: str, status: str | None) -> BoundDocument:
    path = _canonical_regular(path, label)
    raw = path.read_bytes()
    if len(raw) > MAX_JSON_BYTES:
        raise JointLifecycleError(f"{label} exceeds the bounded JSON size")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JointLifecycleError(f"{label} is not JSON: {exc}") from exc
    document = _mapping(document, label)
    try:
        verified = verify(document, label=label)
    except SealIntegrityError as exc:
        raise JointLifecycleError(f"{label} seal is invalid: {exc}") from exc
    if verified.get("schema") != schema:
        raise JointLifecycleError(f"{label}.schema must be {schema!r}")
    if status is not None and verified.get("status") != status:
        raise JointLifecycleError(f"{label}.status must be {status!r}")
    seal_sha256 = verified.get("seal_sha256")
    if not _is_sha(seal_sha256):
        raise JointLifecycleError(f"{label}.seal_sha256 is invalid")
    return BoundDocument(
        path=path,
        raw_sha256=_sha_bytes(raw),
        bytes=len(raw),
        document=verified,
        document_sha256=independent._sha256(verified),
        document_seal_sha256=str(seal_sha256),
    )


def _binding(bound: BoundDocument) -> dict[str, Any]:
    return {
        "path": str(bound.path),
        "bytes": bound.bytes,
        "sha256": bound.raw_sha256,
        "document_sha256": bound.document_sha256,
        "document_seal_sha256": bound.document_seal_sha256,
    }


def _identity(bound: BoundDocument) -> dict[str, Any]:
    return {
        "present": True,
        "document_sha256": bound.document_sha256,
        "document_seal_sha256": bound.document_seal_sha256,
    }


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise JointLifecycleError(f"{path} must be absolute")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise JointLifecycleError(f"{path} parent must be a real existing directory")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    directory = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _new_dir(path: Path, label: str) -> Path:
    if not path.is_absolute() or path == REPO_ROOT:
        raise JointLifecycleError(f"{label} must be a new bounded absolute directory")
    if path.exists():
        raise JointLifecycleError(f"{label} must be new")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise JointLifecycleError(f"{label} parent must be a real existing directory")
    path.mkdir(mode=0o700)
    return path.resolve(strict=True)


def _seal_and_verify(document: Mapping[str, Any]) -> dict[str, Any]:
    sealed = seal(dict(document))
    try:
        verify(sealed, label="new lifecycle document")
    except SealIntegrityError as exc:  # pragma: no cover - seal implementation failure
        raise JointLifecycleError(f"new lifecycle document did not self-verify: {exc}") from exc
    return sealed


def _bound_input(document: BoundDocument) -> dict[str, Any]:
    return {
        "document": document.document,
        "document_sha256": document.document_sha256,
        "document_seal_sha256": document.document_seal_sha256,
    }


def _read_preflight_paths(paths: PreflightPaths) -> dict[str, BoundDocument]:
    return {
        "continuation_readiness": _read_bound(
            paths.continuation_readiness,
            "continuation readiness",
            independent.READINESS_SCHEMA,
            independent.READINESS_STATUS,
        ),
        "l0_outer_terminal": _read_bound(
            paths.l0_outer_terminal, "historical L0 outer", independent.L0_OUTER_SCHEMA, independent.L0_OUTER_STATUS
        ),
        "l0_inner_capture": _read_bound(
            paths.l0_inner_capture, "historical L0 inner", independent.L0_INNER_SCHEMA, independent.L0_INNER_STATUS
        ),
        "l0_post_capture_assessor_binding": _read_bound(
            paths.assessor_binding,
            "assessor binding",
            independent.L0_ASSESSOR_BINDING_SCHEMA,
            independent.L0_ASSESSOR_BINDING_STATUS,
        ),
        "post_capture_assessment": _read_bound(
            paths.post_capture_assessment,
            "post-capture assessment",
            independent.L0_ASSESSMENT_SCHEMA,
            independent.L0_ASSESSMENT_STATUS,
        ),
        "lease_release_receipt": _read_bound(
            paths.prior_lease_release,
            "historical L0 lease release",
            independent.L0_RELEASE_SCHEMA,
            independent.L0_RELEASE_STATUS,
        ),
        "manifest": _read_bound(paths.manifest, "manifest", independent.MANIFEST_SCHEMA, None),
        "admission_receipt": _read_bound(
            paths.admission_receipt,
            "admission receipt",
            independent.ADMISSION_RECEIPT_SCHEMA,
            independent.ADMISSION_RECEIPT_STATUS,
        ),
        "schedule": _read_bound(paths.schedule, "schedule", independent.SCHEDULE_SCHEMA, independent.SCHEDULE_STATUS),
        "joint_l0_l1_child_preflight": _read_bound(
            paths.joint_static_plan,
            "joint static plan",
            independent.JOINT_CHILD_PREFLIGHT_SCHEMA,
            independent.JOINT_CHILD_PREFLIGHT_STATUS,
        ),
        "l0_source_outer_preflight": _read_bound(
            paths.l0_source_outer_preflight,
            "source-token L0 outer preflight",
            independent.L0_SOURCE_OUTER_PREFLIGHT_SCHEMA,
            independent.L0_SOURCE_OUTER_PREFLIGHT_STATUS,
        ),
        "joint_l0_l1_host_preflight": _read_bound(
            paths.joint_host_preflight,
            "joint host preflight",
            independent.JOINT_HOST_PREFLIGHT_SCHEMA,
            independent.JOINT_HOST_PREFLIGHT_STATUS,
        ),
    }


def _independent_input(authorities: Mapping[str, BoundDocument]) -> dict[str, Any]:
    source = {
        "schema": independent.INPUT_SCHEMA,
        "status": independent.INPUT_STATUS,
        "joint_capture_requested": False,
        **{name: _bound_input(bound) for name, bound in authorities.items()},
        "future_joint_l0_l1_child_sha256": authorities["joint_l0_l1_child_preflight"].document[
            "future_joint_l0_l1_child_sha256"
        ],
    }
    return _seal_and_verify(source)


def _validate_host_chain(authorities: Mapping[str, BoundDocument], result: Mapping[str, Any]) -> tuple[str, tuple[int, ...], tuple[float, ...]]:
    if result.get("prepared") is not True or result.get("blockers") != []:
        raise JointLifecycleError("independent L0→L1 preflight is not green")
    child = authorities["joint_l0_l1_child_preflight"]
    host = authorities["joint_l0_l1_host_preflight"]
    source_l0 = authorities["l0_source_outer_preflight"]
    child_sha = child.document.get("future_joint_l0_l1_child_sha256")
    if not _is_sha(child_sha) or result.get("future_joint_l0_l1_child_sha256") != child_sha:
        raise JointLifecycleError("independent preflight child SHA drifted")
    host_binary = _mapping(host.document.get("host_binary"), "joint host preflight.host_binary")
    if host_binary.get("sha256") != child_sha:
        raise JointLifecycleError("joint host preflight does not bind static plan host SHA")
    host_static = _mapping(host.document.get("joint_static_plan"), "joint host preflight.joint_static_plan")
    if (
        host_static.get("document_sha256") != child.document_sha256
        or host_static.get("seal_sha256") != child.document_seal_sha256
    ):
        raise JointLifecycleError("joint host preflight static plan binding drifted")
    host_l0 = _mapping(host.document.get("l0_source_outer_preflight"), "joint host preflight.l0_source_outer_preflight")
    if (
        host_l0.get("document_sha256") != source_l0.document_sha256
        or host_l0.get("seal_sha256") != source_l0.document_seal_sha256
    ):
        raise JointLifecycleError("joint host preflight source-token L0 binding drifted")
    body = _mapping(host.document.get("host_body"), "joint host preflight.host_body")
    for field in (
        "strict_joint_entrypoint_compiled",
        "strict_joint_capture_interface_compiled",
        "metal_entrypoint_available_only_under_new_joint_lease_and_outer_launch_authority",
        "writes_assessor_compatible_inner_receipt_last",
        "phase_accurate_terminal_refusal_receipt_supported",
        "future_outer_reaper_and_fresh_lease_required_before_entrypoint_may_run",
    ):
        _require_bool(body, field, True, "joint host preflight.host_body")
    _require_bool(
        body,
        "historical_l0_receipt_or_pinned_buffer_import_allowed",
        False,
        "joint host preflight.host_body",
    )
    _require_bool(body, "l1_suffix_or_moe_authorized", False, "joint host preflight.host_body")
    scope = _mapping(result.get("authorized_future_component_scope"), "independent preflight scope")
    for field, expected in (("l0_reencode_dispatches", L0_DISPATCHES), ("l1_prefix_dispatches", L1_DISPATCHES), ("joint_total_dispatches", TOTAL_DISPATCHES), ("source_token_id", SOURCE_TOKEN_ID)):
        _require_int(scope, field, expected, "independent preflight scope")
    for field in ("same_runtime_required", "same_tcb_required", "single_fence_after_l0_and_l1_prefix_required"):
        _require_bool(scope, field, True, "independent preflight scope")
    route = _mapping(source_l0.document.get("source_token_route"), "source-token L0 outer preflight.route")
    route_ids = tuple(route.get("route_ids", []))
    route_weights = tuple(float(value) for value in route.get("normalized_weights", []))
    if len(route_ids) != 10 or len(set(route_ids)) != 10 or any(not isinstance(value, int) for value in route_ids):
        raise JointLifecycleError("source-token L0 outer preflight route IDs are not ten unique integers")
    if len(route_weights) != 10 or any(value < 0.0 for value in route_weights):
        raise JointLifecycleError("source-token L0 outer preflight route weights are invalid")
    if abs(sum(route_weights) - 1.0) > 2.0e-6:
        raise JointLifecycleError("source-token L0 outer preflight route weights are not normalized")
    return str(child_sha), route_ids, route_weights


def build_cpu_preflight(paths: PreflightPaths) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build, but do not write, the independent and outer CPU-only documents."""
    authorities = _read_preflight_paths(paths)
    request = _independent_input(authorities)
    result = independent.preflight(request)
    try:
        verify(result, label="independent L0→L1 result")
    except SealIntegrityError as exc:
        raise JointLifecycleError(f"independent preflight result has invalid seal: {exc}") from exc
    child_sha: str | None = None
    route_ids: tuple[int, ...] = ()
    route_weights: tuple[float, ...] = ()
    blockers = result.get("blockers")
    prepared = result.get("prepared") is True and blockers == []
    if prepared:
        child_sha, route_ids, route_weights = _validate_host_chain(authorities, result)
    status = OUTER_PREFLIGHT_STATUS if prepared else OUTER_PREFLIGHT_REFUSED_STATUS
    outer = _seal_and_verify(
        {
            "schema": OUTER_PREFLIGHT_SCHEMA,
            "status": status,
            "prepared": prepared,
            "child_started": False,
            "metal_or_gpu_activity_performed": False,
            "lease_issued_or_consumed": False,
            "independent_preflight": {
                "schema": result.get("schema"),
                "status": result.get("status"),
                "document_sha256": independent._sha256(result),
                "document_seal_sha256": result.get("seal_sha256"),
                "prepared": result.get("prepared"),
                "blockers": result.get("blockers"),
            },
            "authority_chain": {name: _binding(bound) for name, bound in authorities.items()},
            "host_execution_interface": {
                "compiled_host_preflight_only": False,
                "metal_entrypoint_available": True,
                "writes_assessor_compatible_inner_receipt_last": True,
                "phase_accurate_terminal_refusal_receipt_supported": True,
                "separate_strict_host_execution_binding_required": True,
                "no_lease_may_be_issued_from_this_cpu_preflight_alone": True,
            },
            "exact_joint_scope": {
                "source_token_id": SOURCE_TOKEN_ID,
                "l0_dispatches": L0_DISPATCHES,
                "l1_prefix_dispatches": L1_DISPATCHES,
                "total_dispatches": TOTAL_DISPATCHES,
                "single_fence_required": True,
                "non_timed_required": True,
                "tcb_trace_mode": "off",
                "host_binary_sha256": child_sha,
                "route_ids": list(route_ids),
                "normalized_route_weights": list(route_weights),
            },
            "lifecycle": {
                "new_joint_specific_lease_required": True,
                "create_new_replay_reservation_required": True,
                "outer_reaped_child_required": True,
                "terminal_receipt_written_last_required": True,
                "separate_actual_release_required_after_outer_terminal": True,
                "automatic_retry_prohibited": True,
            },
            "claim_boundary": {
                "cpu_only_preflight": True,
                "historical_l0_receipts_are_provenance_only": True,
                "cross_process_pinned_buffer_transfer_authorized": False,
                "l1_suffix_or_moe_authorized": False,
                "complete_layer_token_decoder_server_hcli_tps_tg_or_tournament_authorized": False,
            },
        }
    )
    return request, result, outer


def write_cpu_preflight(paths: PreflightPaths, out_dir: Path) -> dict[str, Any]:
    """Persist receipt-last CPU-only outer preparation in a create-new directory."""
    out_dir = _new_dir(out_dir, "--out-dir")
    request, result, outer = build_cpu_preflight(paths)
    _write_new(out_dir / PREFLIGHT_INPUT_FILENAME, request)
    _write_new(out_dir / PREFLIGHT_RESULT_FILENAME, result)
    _write_new(out_dir / OUTER_PREFLIGHT_FILENAME, outer)
    return outer


def _read_outer_preflight(path: Path) -> BoundDocument:
    outer = _read_bound(path, "joint outer preflight", OUTER_PREFLIGHT_SCHEMA, OUTER_PREFLIGHT_STATUS)
    root = outer.document
    _require_bool(root, "prepared", True, "joint outer preflight")
    _require_bool(root, "child_started", False, "joint outer preflight")
    _require_bool(root, "metal_or_gpu_activity_performed", False, "joint outer preflight")
    _require_bool(root, "lease_issued_or_consumed", False, "joint outer preflight")
    independent_result = _mapping(root.get("independent_preflight"), "joint outer preflight.independent_preflight")
    _require_bool(independent_result, "prepared", True, "joint outer preflight.independent_preflight")
    if independent_result.get("blockers") != []:
        raise JointLifecycleError("joint outer preflight independent result has blockers")
    scope = _mapping(root.get("exact_joint_scope"), "joint outer preflight.exact_joint_scope")
    for field, expected in (("source_token_id", SOURCE_TOKEN_ID), ("l0_dispatches", L0_DISPATCHES), ("l1_prefix_dispatches", L1_DISPATCHES), ("total_dispatches", TOTAL_DISPATCHES)):
        _require_int(scope, field, expected, "joint outer preflight.exact_joint_scope")
    for field in ("single_fence_required", "non_timed_required"):
        _require_bool(scope, field, True, "joint outer preflight.exact_joint_scope")
    if scope.get("tcb_trace_mode") != "off":
        raise JointLifecycleError("joint outer preflight requires TcbTraceMode::Off")
    _require_sha(scope, "host_binary_sha256", "joint outer preflight.exact_joint_scope")
    route_ids = _array(scope.get("route_ids"), "joint outer preflight.exact_joint_scope.route_ids")
    weights = _array(scope.get("normalized_route_weights"), "joint outer preflight.exact_joint_scope.normalized_route_weights")
    if len(route_ids) != 10 or len(set(route_ids)) != 10 or len(weights) != 10:
        raise JointLifecycleError("joint outer preflight must retain exact ten unique routes and weights")
    interface = _mapping(root.get("host_execution_interface"), "joint outer preflight.host_execution_interface")
    _require_bool(interface, "compiled_host_preflight_only", False, "joint outer preflight.host_execution_interface")
    _require_bool(interface, "metal_entrypoint_available", True, "joint outer preflight.host_execution_interface")
    _require_bool(
        interface,
        "writes_assessor_compatible_inner_receipt_last",
        True,
        "joint outer preflight.host_execution_interface",
    )
    _require_bool(
        interface,
        "phase_accurate_terminal_refusal_receipt_supported",
        True,
        "joint outer preflight.host_execution_interface",
    )
    _require_bool(
        interface,
        "separate_strict_host_execution_binding_required",
        True,
        "joint outer preflight.host_execution_interface",
    )
    _require_bool(
        interface,
        "no_lease_may_be_issued_from_this_cpu_preflight_alone",
        True,
        "joint outer preflight.host_execution_interface",
    )
    lifecycle = _mapping(root.get("lifecycle"), "joint outer preflight.lifecycle")
    for field in (
        "new_joint_specific_lease_required",
        "create_new_replay_reservation_required",
        "outer_reaped_child_required",
        "terminal_receipt_written_last_required",
        "separate_actual_release_required_after_outer_terminal",
        "automatic_retry_prohibited",
    ):
        _require_bool(lifecycle, field, True, "joint outer preflight.lifecycle")
    return outer


def _read_watcher_hold(path: Path, outer: BoundDocument) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = _canonical_regular(path, "watcher hold")
    raw = path.read_bytes()
    if len(raw) > MAX_JSON_BYTES:
        raise JointLifecycleError("watcher hold exceeds bounded JSON size")
    try:
        document = _mapping(json.loads(raw.decode("utf-8")), "watcher hold")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JointLifecycleError(f"watcher hold is not JSON: {exc}") from exc
    if document.get("schema") != WATCHER_HOLD_SCHEMA or document.get("status") != WATCHER_HOLD_STATUS:
        raise JointLifecycleError("watcher hold schema/status drifted")
    preserved = _mapping(document.get("preserved"), "watcher hold.preserved")
    _require_bool(preserved, "runtime_watcher_parent", True, "watcher hold.preserved")
    _require_bool(preserved, "raw_bf16_or_mps_production_fallback", False, "watcher hold.preserved")
    boundary = _mapping(document.get("claim_boundary"), "watcher hold.claim_boundary")
    for field in (
        "this_is_only_gpu_coordination",
        "does_not_change_qwen80_runtime_qualification",
        "does_not_establish_generation_hcli_tps_or_tournament_eligibility",
    ):
        _require_bool(boundary, field, True, "watcher hold.claim_boundary")
    chain = _mapping(outer.document.get("authority_chain"), "joint outer preflight.authority_chain")
    manifest = _mapping(chain.get("manifest"), "joint outer preflight manifest authority")
    admission = _mapping(chain.get("admission_receipt"), "joint outer preflight admission authority")
    source = _mapping(document.get("source_binding"), "watcher hold.source_binding")
    if source.get("manifest_seal_sha256") != manifest.get("document_seal_sha256"):
        raise JointLifecycleError("watcher hold manifest seal does not bind joint outer preflight")
    if source.get("admission_receipt_seal_sha256") != admission.get("document_seal_sha256"):
        raise JointLifecycleError("watcher hold admission seal does not bind joint outer preflight")
    evidence = {"path": str(path), "bytes": len(raw), "sha256": _sha_bytes(raw)}
    return path, document, evidence


def _read_execution_binding(path: Path, outer: BoundDocument) -> BoundDocument:
    binding = _read_bound(path, "strict host execution binding", EXECUTION_BINDING_SCHEMA, EXECUTION_BINDING_STATUS)
    root = binding.document
    _require_bool(root, "metal_entrypoint_available", True, "strict host execution binding")
    _require_bool(root, "writes_assessor_compatible_inner_receipt", True, "strict host execution binding")
    _require_bool(root, "outer_reaped_receipt_last_required", True, "strict host execution binding")
    _require_bool(root, "non_timed_exact_32_dispatches_required", True, "strict host execution binding")
    host = _mapping(root.get("host_binary"), "strict host execution binding.host_binary")
    scope = _mapping(outer.document.get("exact_joint_scope"), "joint outer preflight.exact_joint_scope")
    if not isinstance(host.get("path"), str) or not isinstance(host.get("bytes"), int) or host.get("bytes", 0) <= 0:
        raise JointLifecycleError("strict host execution binding must retain concrete host file evidence")
    if host.get("sha256") != scope.get("host_binary_sha256") or not _is_sha(host.get("sha256")):
        raise JointLifecycleError("strict host execution binding host SHA drifted")
    outer_binding = _mapping(root.get("outer_preflight"), "strict host execution binding.outer_preflight")
    if (
        outer_binding.get("document_sha256") != outer.document_sha256
        or outer_binding.get("document_seal_sha256") != outer.document_seal_sha256
    ):
        raise JointLifecycleError("strict host execution binding does not bind the exact outer preflight")
    policy = _mapping(root.get("execution_policy"), "strict host execution binding.execution_policy")
    for field, expected in (
        ("source_token_id", SOURCE_TOKEN_ID),
        ("l0_dispatches", L0_DISPATCHES),
        ("l1_prefix_dispatches", L1_DISPATCHES),
        ("total_dispatches", TOTAL_DISPATCHES),
    ):
        _require_int(policy, field, expected, "strict host execution binding.execution_policy")
    for field in ("strict_math", "non_timed", "single_fence_required"):
        _require_bool(policy, field, True, "strict host execution binding.execution_policy")
    if policy.get("tcb_trace_mode") != "off":
        raise JointLifecycleError("strict host execution binding must require TcbTraceMode::Off")
    for field in ("l1_suffix_or_moe_authorized", "complete_layer_or_token_authorized", "server_hcli_tps_or_tournament_authorized"):
        _require_bool(policy, field, False, "strict host execution binding.execution_policy")
    receipt = _mapping(root.get("receipt_contract"), "strict host execution binding.receipt_contract")
    if receipt.get("schema") != INNER_SCHEMA or receipt.get("status") != INNER_STATUS:
        raise JointLifecycleError("strict host execution binding inner receipt schema/status drifted")
    for field in ("receipt_written_last", "phase_accurate_terminal_refusal_required", "opaque_same_runtime_continuation_required"):
        _require_bool(receipt, field, True, "strict host execution binding.receipt_contract")
    return binding


def prepare_execution_binding(*, outer_preflight: Path, host_binary: Path, out: Path) -> BoundDocument:
    """Seal the CPU/file-only execution-interface authority for one exact host.

    This only proves that the compiled child exposes the strictly gated
    interface.  It does not issue a lease, reserve a replay directory, spawn
    a process, open an artifact, or construct a Metal context.
    """
    if not out.is_absolute() or out.exists():
        raise JointLifecycleError("execution binding output must be a new absolute path")
    outer = _read_outer_preflight(outer_preflight)
    host_binary = _canonical_regular(host_binary, "joint host binary", executable=True)
    host_raw = host_binary.read_bytes()
    host = {"path": str(host_binary), "bytes": len(host_raw), "sha256": _sha_bytes(host_raw)}
    scope = _mapping(outer.document.get("exact_joint_scope"), "joint outer preflight.exact_joint_scope")
    if host["sha256"] != scope.get("host_binary_sha256"):
        raise JointLifecycleError("joint host binary SHA does not bind the exact outer preflight")
    chain = _mapping(outer.document.get("authority_chain"), "joint outer preflight.authority_chain")
    host_preflight = _mapping(chain.get("joint_l0_l1_host_preflight"), "joint outer host preflight binding")
    if host_preflight.get("document_seal_sha256") is None or host_preflight.get("document_sha256") is None:
        raise JointLifecycleError("joint outer preflight lacks a sealed host-preflight binding")
    document = _seal_and_verify(
        {
            "schema": EXECUTION_BINDING_SCHEMA,
            "status": EXECUTION_BINDING_STATUS,
            "metal_entrypoint_available": True,
            "writes_assessor_compatible_inner_receipt": True,
            "outer_reaped_receipt_last_required": True,
            "non_timed_exact_32_dispatches_required": True,
            "host_binary": host,
            "outer_preflight": _binding(outer),
            "host_preflight": dict(host_preflight),
            "execution_policy": {
                "source_token_id": SOURCE_TOKEN_ID,
                "l0_dispatches": L0_DISPATCHES,
                "l1_prefix_dispatches": L1_DISPATCHES,
                "total_dispatches": TOTAL_DISPATCHES,
                "strict_math": True,
                "non_timed": True,
                "single_fence_required": True,
                "tcb_trace_mode": "off",
                "l1_suffix_or_moe_authorized": False,
                "complete_layer_or_token_authorized": False,
                "server_hcli_tps_or_tournament_authorized": False,
            },
            "receipt_contract": {
                "schema": INNER_SCHEMA,
                "status": INNER_STATUS,
                "receipt_written_last": True,
                "phase_accurate_terminal_refusal_required": True,
                "opaque_same_runtime_continuation_required": True,
            },
            "claim_boundary": {
                "cpu_file_only_execution_binding": True,
                "lease_issued_or_consumed": False,
                "metal_or_gpu_activity_performed": False,
                "server_watcher_hcli_tps_or_tournament_action": False,
            },
        }
    )
    _write_new(out, document)
    return _read_bound(out, "strict host execution binding", EXECUTION_BINDING_SCHEMA, EXECUTION_BINDING_STATUS)


def _lease_id(outer: BoundDocument, execution: BoundDocument, watcher_evidence: Mapping[str, Any], out: Path) -> str:
    return _sha_bytes(
        json.dumps(
            {
                "schema": LEASE_SCHEMA,
                "outer_preflight": _identity(outer),
                "execution_binding": _identity(execution),
                "watcher_hold": dict(watcher_evidence),
                "out": str(out),
                "recorded_at": _utc_now(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def issue_lease(
    *, outer_preflight: Path, execution_binding: Path, watcher_hold: Path, out: Path
) -> LeaseContext:
    """Create a future-only joint lease after CPU/file authority checks.

    This function has no caller on the current CPU-only path.  A future outer
    reaper must still perform its independent resource/watcher admission
    before it can use a sealed lease to spawn the strict host child.
    """
    if not out.is_absolute() or out.exists():
        raise JointLifecycleError("lease output must be a new absolute path")
    outer = _read_outer_preflight(outer_preflight)
    execution = _read_execution_binding(execution_binding, outer)
    hold_path, _, hold_evidence = _read_watcher_hold(watcher_hold, outer)
    lease_id = _lease_id(outer, execution, hold_evidence, out)
    scope = _mapping(outer.document["exact_joint_scope"], "joint outer preflight.exact_joint_scope")
    document = _seal_and_verify(
        {
            "schema": LEASE_SCHEMA,
            "status": LEASE_STATUS,
            "lease_id": lease_id,
            "issuer": {"role": "joint_component_lease_issuer", "issuer_identity_sha256": _sha_bytes(str(out).encode())},
            "outer_preflight": _binding(outer),
            "execution_binding": _binding(execution),
            "watcher_hold": hold_evidence,
            "host_binary_sha256": scope["host_binary_sha256"],
            "execution_policy": {
                "source_token_id": SOURCE_TOKEN_ID,
                "l0_dispatches": L0_DISPATCHES,
                "l1_prefix_dispatches": L1_DISPATCHES,
                "total_dispatches": TOTAL_DISPATCHES,
                "single_fence_required": True,
                "non_timed": True,
                "tcb_trace_mode": "off",
                "strict_math_required": True,
                "l1_suffix_or_moe_allowed": False,
                "complete_layer_or_token_allowed": False,
                "server_hcli_tps_tg_or_tournament_allowed": False,
            },
            "lifecycle": {
                "fresh_for_exact_outer_preflight": True,
                "create_new_replay_reservation_required": True,
                "outer_reaped_capture_required": True,
                "terminal_receipt_written_last_required": True,
                "separate_actual_release_required": True,
                "automatic_retry_prohibited": True,
            },
            "claim_boundary": {"lease_issuance_is_cpu_file_only": True, "fixture_or_synthetic": False},
        }
    )
    _write_new(out, document)
    lease = _read_bound(out, "joint lease", LEASE_SCHEMA, LEASE_STATUS)
    return LeaseContext(outer, execution, hold_path, hold_evidence, lease, lease_id)


def _validate_lease(path: Path, outer: BoundDocument, execution: BoundDocument, watcher_hold: Path) -> LeaseContext:
    lease = _read_bound(path, "joint lease", LEASE_SCHEMA, LEASE_STATUS)
    root = lease.document
    lease_id = _require_sha(root, "lease_id", "joint lease")
    for field, expected in (("outer_preflight", outer), ("execution_binding", execution)):
        value = _mapping(root.get(field), f"joint lease.{field}")
        if value.get("document_sha256") != expected.document_sha256 or value.get("document_seal_sha256") != expected.document_seal_sha256:
            raise JointLifecycleError(f"joint lease.{field} drifted")
    hold_path, _, hold_evidence = _read_watcher_hold(watcher_hold, outer)
    if _mapping(root.get("watcher_hold"), "joint lease.watcher_hold") != hold_evidence:
        raise JointLifecycleError("joint lease watcher hold evidence drifted")
    policy = _mapping(root.get("execution_policy"), "joint lease.execution_policy")
    for field, expected in (("source_token_id", SOURCE_TOKEN_ID), ("l0_dispatches", L0_DISPATCHES), ("l1_prefix_dispatches", L1_DISPATCHES), ("total_dispatches", TOTAL_DISPATCHES)):
        _require_int(policy, field, expected, "joint lease.execution_policy")
    for field in ("single_fence_required", "non_timed", "strict_math_required"):
        _require_bool(policy, field, True, "joint lease.execution_policy")
    for field in ("l1_suffix_or_moe_allowed", "complete_layer_or_token_allowed", "server_hcli_tps_tg_or_tournament_allowed"):
        _require_bool(policy, field, False, "joint lease.execution_policy")
    return LeaseContext(outer, execution, hold_path, hold_evidence, lease, lease_id)


def _route_scope(outer: BoundDocument) -> tuple[tuple[int, ...], tuple[float, ...]]:
    scope = _mapping(outer.document["exact_joint_scope"], "joint outer preflight.exact_joint_scope")
    return tuple(scope["route_ids"]), tuple(float(value) for value in scope["normalized_route_weights"])


def _require_parity(value: object, label: str) -> None:
    record = _mapping(value, label)
    _require_bool(record, "passed", True, label)
    _require_sha(record, "cpu_f32le_sha256", label)
    _require_sha(record, "device_f32le_sha256", label)
    error = record.get("max_abs_error")
    if isinstance(error, bool) or not isinstance(error, (int, float)) or error < 0 or error > MAX_PARITY_ERROR:
        raise JointLifecycleError(f"{label}.max_abs_error exceeds strict component tolerance")


def _require_route_witness(value: object, label: str) -> None:
    record = _mapping(value, label)
    _require_bool(record, "passed", True, label)
    _require_sha(record, "f32le_sha256", label)
    _require_sha(record, "cpu_f32le_sha256", label)
    error = record.get("max_abs_error")
    if isinstance(error, bool) or not isinstance(error, (int, float)) or error < 0 or error > MAX_PARITY_ERROR:
        raise JointLifecycleError(f"{label}.max_abs_error exceeds strict component tolerance")


def _require_state(value: object, label: str, *, slot: int, offset: int, capacity: int) -> None:
    record = _mapping(value, label)
    _require_bool(record, "passed", True, label)
    for field, expected in (("slot", slot), ("offset_bytes", offset), ("capacity_bytes", capacity)):
        _require_int(record, field, expected, label)
    _require_sha(record, "device_buffer_identity_sha256", label)
    _require_sha(record, "f32le_sha256", label)
    error = record.get("max_abs_error")
    if isinstance(error, bool) or not isinstance(error, (int, float)) or error < 0 or error > MAX_PARITY_ERROR:
        raise JointLifecycleError(f"{label}.max_abs_error exceeds strict component tolerance")


def validate_inner_receipt(document: Mapping[str, Any], outer: BoundDocument, lease: LeaseContext) -> None:
    """Validate the assessor-compatible inner receipt before outer promotion.

    It deliberately ties the opaque continuation runtime/TCB IDs to the
    fresh execution IDs, a stronger equality than the assessor's shape-only
    checks, and validates the exact source-token route witness before L1.
    """
    root = _mapping(document, "joint inner receipt")
    try:
        verify(root, label="joint inner receipt")
    except SealIntegrityError as exc:
        raise JointLifecycleError(f"joint inner receipt seal invalid: {exc}") from exc
    if root.get("schema") != INNER_SCHEMA or root.get("status") != INNER_STATUS:
        raise JointLifecycleError("joint inner receipt schema/status drifted")
    for field, expected in (("fixture_or_synthetic", False), ("self_asserted", False)):
        _require_bool(root, field, expected, "joint inner receipt")
    issuer = _mapping(root.get("issuer"), "joint inner receipt.issuer")
    if issuer.get("role") != "joint_component_capture_child":
        raise JointLifecycleError("joint inner receipt issuer role drifted")
    _require_sha(issuer, "issuer_identity_sha256", "joint inner receipt.issuer")
    chain = _mapping(outer.document["authority_chain"], "joint outer preflight.authority_chain")
    upstream = _mapping(root.get("upstream_authorities"), "joint inner receipt.upstream_authorities")
    for inner_name, outer_name in (("schedule_wrapper", "schedule"), ("continuation", "continuation_readiness"), ("assessor_binding", "l0_post_capture_assessor_binding")):
        value = _mapping(upstream.get(inner_name), f"joint inner receipt.upstream_authorities.{inner_name}")
        expected = _mapping(chain.get(outer_name), f"joint outer preflight authority {outer_name}")
        if value.get("present") is not True or value.get("document_sha256") != expected.get("document_sha256") or value.get("document_seal_sha256") != expected.get("document_seal_sha256"):
            raise JointLifecycleError(f"joint inner receipt upstream {inner_name} drifted")
    capability = _mapping(root.get("opaque_l0_continuation"), "joint inner receipt.opaque_l0_continuation")
    if capability.get("factory") != CAPABILITY_FACTORY or capability.get("l1_encoder") != L1_ENCODER or capability.get("consuming_finalizer") != FINALIZER:
        raise JointLifecycleError("joint inner opaque continuation ABI drifted")
    for field in ("opaque", "freshly_derived_from_l0_23_dispatch_graph", "same_runtime_state_arena_bound", "same_command_buffer_bound", "non_transferable_across_processes"):
        _require_bool(capability, field, True, "joint inner receipt.opaque_l0_continuation")
    _require_bool(capability, "raw_pinned_buffer_or_dispatch_count_input_accepted", False, "joint inner receipt.opaque_l0_continuation")
    for field in ("capability_identity_sha256", "runtime_identity_sha256", "runtime_state_arena_identity_sha256", "command_buffer_identity_sha256"):
        _require_sha(capability, field, "joint inner receipt.opaque_l0_continuation")
    execution = _mapping(root.get("fresh_joint_execution"), "joint inner receipt.fresh_joint_execution")
    for field in ("fresh_runtime", "fresh_session", "same_runtime", "same_tcb", "structural_trace_non_timed", "route_guard_enforced_before_l1"):
        _require_bool(execution, field, True, "joint inner receipt.fresh_joint_execution")
    for field in ("runtime_identity_sha256", "session_identity_sha256", "tcb_identity_sha256"):
        _require_sha(execution, field, "joint inner receipt.fresh_joint_execution")
    if capability["runtime_identity_sha256"] != execution["runtime_identity_sha256"] or capability["command_buffer_identity_sha256"] != execution["tcb_identity_sha256"]:
        raise JointLifecycleError("joint inner opaque continuation runtime/TCB identity mismatch")
    for field, expected in (("source_token_id", SOURCE_TOKEN_ID), ("l0_dispatches", L0_DISPATCHES), ("l1_prefix_dispatches", L1_DISPATCHES), ("total_dispatches", TOTAL_DISPATCHES), ("fence_count", 1)):
        _require_int(execution, field, expected, "joint inner receipt.fresh_joint_execution")
    trace = _mapping(root.get("structural_kernel_trace"), "joint inner receipt.structural_kernel_trace")
    _require_bool(trace, "non_timed", True, "joint inner receipt.structural_kernel_trace")
    _require_bool(trace, "exact_order", True, "joint inner receipt.structural_kernel_trace")
    if tuple(_array(trace.get("kernel_names"), "joint inner receipt.structural_kernel_trace.kernel_names")) != STRUCTURAL_KERNELS:
        raise JointLifecycleError("joint inner receipt structural kernel trace drifted from exact 23+9")
    fence = _mapping(root.get("single_fence"), "joint inner receipt.single_fence")
    if fence.get("consuming_finalizer") != FINALIZER:
        raise JointLifecycleError("joint inner receipt consuming finalizer drifted")
    for field, expected in (("only_command_buffer_consumed", True), ("fence_succeeded", True), ("readbacks_after_fence", True), ("append_after_fence_possible", False)):
        _require_bool(fence, field, expected, "joint inner receipt.single_fence")
    _require_int(fence, "fence_count", 1, "joint inner receipt.single_fence")
    readbacks = _mapping(root.get("fresh_readbacks"), "joint inner receipt.fresh_readbacks")
    _validate_readbacks(readbacks, outer)
    outer_binding = _mapping(root.get("joint_outer_preflight_binding"), "joint inner receipt.joint_outer_preflight_binding")
    if (
        outer_binding.get("document_sha256") != outer.document_sha256
        or outer_binding.get("document_seal_sha256") != outer.document_seal_sha256
    ):
        raise JointLifecycleError("joint inner receipt does not bind the exact outer preflight")
    lease_binding = _mapping(root.get("joint_lease_binding"), "joint inner receipt.joint_lease_binding")
    if lease_binding.get("lease_id") != lease.lease_id:
        raise JointLifecycleError("joint inner receipt lease ID drifted")
    lease_receipt = _mapping(lease_binding.get("receipt"), "joint inner receipt.joint_lease_binding.receipt")
    if (
        lease_receipt.get("document_sha256") != lease.lease.document_sha256
        or lease_receipt.get("document_seal_sha256") != lease.lease.document_seal_sha256
    ):
        raise JointLifecycleError("joint inner receipt lease binding drifted")
    launch = _mapping(root.get("outer_launch_authority_binding"), "joint inner receipt.outer_launch_authority_binding")
    for field in ("path", "sha256", "document_sha256", "document_seal_sha256"):
        if field == "path":
            if not isinstance(launch.get(field), str) or not launch[field]:
                raise JointLifecycleError("joint inner receipt outer launch authority path is invalid")
        else:
            _require_sha(launch, field, "joint inner receipt.outer_launch_authority_binding")
    phase = _mapping(root.get("execution_phase"), "joint inner receipt.execution_phase")
    for field in (
        "strict_artifact_admission_started",
        "strict_artifact_admission_succeeded",
        "metal_context_construction_attempted",
        "metal_context_constructed",
        "structural_kernel_trace_enabled",
        "command_commit_may_have_been_attempted",
        "command_fence_succeeded",
        "readback_started",
    ):
        _require_bool(phase, field, True, "joint inner receipt.execution_phase")
    _require_int(phase, "dispatches_encoded", TOTAL_DISPATCHES, "joint inner receipt.execution_phase")
    if tuple(_array(phase.get("encoded_kernel_names"), "joint inner receipt.execution_phase.encoded_kernel_names")) != STRUCTURAL_KERNELS:
        raise JointLifecycleError("joint inner receipt execution phase trace drifted")
    _require_bool(phase, "device_dispatch_may_have_occurred", True, "joint inner receipt.execution_phase")
    durable = _mapping(root.get("durable_capture"), "joint inner receipt.durable_capture")
    for field in ("receipt_written_last_is_completion_marker", "outer_reaped_capture_required", "replay_guarded"):
        _require_bool(durable, field, True, "joint inner receipt.durable_capture")
    if not isinstance(durable.get("capture_directory"), str) or not durable["capture_directory"]:
        raise JointLifecycleError("joint inner receipt durable capture directory is invalid")
    boundary = _mapping(root.get("claim_boundary"), "joint inner receipt.claim_boundary")
    _require_bool(boundary, "component_only", True, "joint inner receipt.claim_boundary")
    for field in ("l1_suffix_or_moe_executed", "complete_layer_executed", "token_generated", "decoder_started", "server_or_watcher_started"):
        _require_bool(boundary, field, False, "joint inner receipt.claim_boundary")
    for forbidden in ("historical_l0_receipt", "old_l0_receipt", "input_device_buffer_id", "input_f32le_sha256", "raw_pinned_buffer", "raw_dispatch_count"):
        if forbidden in root or forbidden in capability or forbidden in execution:
            raise JointLifecycleError(f"joint inner receipt may not import {forbidden}")


def _validate_readbacks(readbacks: Mapping[str, Any], outer: BoundDocument) -> None:
    expected_ids, expected_weights = _route_scope(outer)
    l0 = _mapping(readbacks.get("l0_suffix"), "joint readbacks.l0_suffix")
    guard = _mapping(l0.get("route_guard"), "joint readbacks.l0_suffix.route_guard")
    _require_bool(guard, "passed", True, "joint readbacks.l0_suffix.route_guard")
    _require_int(guard, "value", 1, "joint readbacks.l0_suffix.route_guard")
    if tuple(guard.get("expected_route_ids", [])) != expected_ids or tuple(guard.get("observed_route_ids", [])) != expected_ids:
        raise JointLifecycleError("joint route guard IDs drifted")
    observed_weights = tuple(float(value) for value in guard.get("observed_route_weights", []))
    declared_weights = tuple(float(value) for value in guard.get("expected_route_weights", []))
    if len(observed_weights) != 10 or len(declared_weights) != 10:
        raise JointLifecycleError("joint route guard weight count drifted")
    # The declared weights bind the source-token route authority; the observed
    # weights are post-fence Metal readbacks.  They must both stay inside the
    # strict router tolerance, but they are not required to be bit-identical:
    # normal f32 CPU/Metal rounding is already recorded by the bounded parity
    # witness below.  Exact IDs and the opaque same-runtime continuation carry
    # route/custody identity, not a second impossible byte-equality demand.
    if any(
        abs(actual - expected) > 1.0e-6
        for actual, expected in zip(declared_weights, expected_weights)
    ) or any(
        abs(actual - expected) > 1.0e-6
        for actual, expected in zip(observed_weights, expected_weights)
    ):
        raise JointLifecycleError("joint route guard weights drifted")
    error = guard.get("weights_max_abs_error")
    if isinstance(error, bool) or not isinstance(error, (int, float)) or error < 0 or error > MAX_PARITY_ERROR:
        raise JointLifecycleError("joint route guard weight parity exceeds tolerance")
    for field in ("postnorm", "router_logits", "shared_output", "routed_sum", "second_residual"):
        _require_parity(l0.get(field), f"joint readbacks.l0_suffix.{field}")
    witnesses = _array(l0.get("all_ten_weighted_route_witnesses"), "joint route witnesses")
    if len(witnesses) != 10:
        raise JointLifecycleError("joint route witnesses must contain ten entries")
    for index, value in enumerate(witnesses):
        witness = _mapping(value, f"joint route witness {index}")
        _require_int(witness, "wave_index", index, f"joint route witness {index}")
        _require_int(witness, "expert_id", expected_ids[index], f"joint route witness {index}")
        _require_route_witness(witness, f"joint route witness {index}")
    l0_state = _mapping(readbacks.get("fresh_l0_state"), "joint readbacks.fresh_l0_state")
    for field, capacity in (("active_conv", L0_CONV_BYTES), ("active_recurrent", L0_RECURRENT_BYTES), ("rollback_conv", L0_CONV_BYTES), ("rollback_recurrent", L0_RECURRENT_BYTES)):
        _require_state(l0_state.get(field), f"joint L0 state {field}", slot=0, offset=0, capacity=capacity)
    l1 = _mapping(readbacks.get("fresh_l1_slot1"), "joint readbacks.fresh_l1_slot1")
    for field, expected in (("layer", 1), ("linear_state_slot", 1), ("output_elements", HIDDEN_ELEMENTS), ("output_bytes", HIDDEN_BYTES)):
        _require_int(l1, field, expected, "joint L1 slot1 readback")
    _require_parity(l1.get("input"), "joint L1 slot1 input parity")
    _require_parity(l1.get("first_residual_output"), "joint L1 slot1 output parity")
    for field, offset, capacity in (("active_conv", L0_CONV_BYTES, L1_CONV_CAPACITY_BYTES), ("rollback_conv", L0_CONV_BYTES, L1_CONV_CAPACITY_BYTES), ("active_recurrent", L0_RECURRENT_BYTES, L1_RECURRENT_CAPACITY_BYTES), ("rollback_recurrent", L0_RECURRENT_BYTES, L1_RECURRENT_CAPACITY_BYTES)):
        _require_state(l1.get(field), f"joint L1 state {field}", slot=1, offset=offset, capacity=capacity)


def _terminal(returncode: int | None, *, timed_out: bool, spawn_error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reaped": returncode is not None,
        "timed_out": timed_out,
        "exit_code": returncode if isinstance(returncode, int) and returncode >= 0 else None,
        "signal": -returncode if isinstance(returncode, int) and returncode < 0 else None,
    }
    if spawn_error is not None:
        result["spawn_error"] = spawn_error
        result["reaped"] = False
    return result


def _terminate_group(child: subprocess.Popen[bytes]) -> int | None:
    if child.poll() is not None:
        return child.returncode
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        return child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        return child.wait(timeout=10)


def _launch_identity(lease: LeaseContext, capture: Path) -> str:
    return _sha_bytes(
        json.dumps(
            {
                "schema": OUTER_SCHEMA,
                "lease": _identity(lease.lease),
                "outer_preflight": _identity(lease.outer_preflight),
                "execution_binding": _identity(lease.execution_binding),
                "capture": str(capture),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _outer_launch_authority(lease: LeaseContext, capture: Path, identity: str, workers: int) -> dict[str, Any]:
    return _seal_and_verify(
        {
            "schema": OUTER_LAUNCH_SCHEMA,
            "status": OUTER_LAUNCH_STATUS,
            "lease": {**_binding(lease.lease), "lease_id": lease.lease_id},
            "outer_preflight": _binding(lease.outer_preflight),
            "execution_binding": _binding(lease.execution_binding),
            "planned_outer_capture_dir": str(capture),
            "planned_inner_capture_dir": str(capture / INNER_DIRNAME),
            "launch_identity_sha256": identity,
            "workers": workers,
            "execution_policy": {"strict_math": True, "non_timed": True, "total_dispatches": TOTAL_DISPATCHES, "single_fence_required": True},
            "lifecycle": {"outer_reaped_capture_required": True, "terminal_receipt_written_last": True, "automatic_retry_prohibited": True},
            "claim_boundary": {"component_only": True, "l1_suffix_or_moe_authorized": False, "complete_layer_or_token_authorized": False},
        }
    )


def _inner_evidence(capture: Path, outer: BoundDocument, lease: LeaseContext) -> dict[str, Any]:
    path = capture / INNER_DIRNAME / "receipt.json"
    if not path.is_file():
        return {"present": False, "binding_valid": False, "error": "inner receipt is absent"}
    try:
        bound = _read_bound(path, "joint inner receipt", INNER_SCHEMA, INNER_STATUS)
        validate_inner_receipt(bound.document, outer, lease)
    except JointLifecycleError as exc:
        return {"present": True, "binding_valid": False, "error": str(exc)}
    return {"present": True, "binding_valid": True, "receipt": _binding(bound), "schema": INNER_SCHEMA, "status": INNER_STATUS}


def _terminal_status(terminal: Mapping[str, Any], inner: Mapping[str, Any]) -> str:
    if terminal.get("spawn_error"):
        return f"{OUTER_REFUSED_PREFIX}CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return f"{OUTER_REFUSED_PREFIX}CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return f"{OUTER_REFUSED_PREFIX}CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return f"{OUTER_REFUSED_PREFIX}CHILD_NONZERO"
    if inner.get("binding_valid") is not True:
        return f"{OUTER_REFUSED_PREFIX}ZERO_EXIT_WITHOUT_STRICT_INNER_RECEIPT"
    return OUTER_STATUS


def _stream_evidence(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": _sha_bytes(raw), "within_max_stream_bytes": len(raw) <= MAX_STREAM_BYTES}


def _terminal_receipt(
    *, capture: Path, lease: LeaseContext, identity: str, command: Sequence[str], child_pid: int | None, terminal: Mapping[str, Any], capture_error: str | None
) -> dict[str, Any]:
    inner = _inner_evidence(capture, lease.outer_preflight, lease)
    status = _terminal_status(terminal, inner)
    if capture_error is not None and status == OUTER_STATUS:
        status = f"{OUTER_REFUSED_PREFIX}OUTER_EVIDENCE_INVALID"
    issuer_id = _sha_bytes(f"outer:{identity}".encode())
    payload: dict[str, Any] = {
        "schema": OUTER_SCHEMA,
        "status": status,
        "fixture_or_synthetic": False,
        "self_asserted": False,
        "issuer": {"role": "joint_component_outer_reaper", "issuer_identity_sha256": issuer_id},
        "inner_capture": ({"present": True, "document_sha256": inner["receipt"]["document_sha256"], "document_seal_sha256": inner["receipt"]["document_seal_sha256"]} if inner.get("binding_valid") else {"present": False, "document_sha256": None, "document_seal_sha256": None}),
        "lease_id": lease.lease_id,
        "child_terminal": {
            "exit_code": terminal.get("exit_code"),
            "reaped": terminal.get("reaped") is True,
            "timed_out": terminal.get("timed_out") is True,
            "terminal_receipt_written_last": True,
            "automatic_retry_disabled": True,
            "lease_reuse_prohibited": True,
        },
        "source_binding": {"outer_preflight": _binding(lease.outer_preflight), "execution_binding": _binding(lease.execution_binding), "lease": _binding(lease.lease), "watcher_hold": lease.watcher_hold_evidence},
        "outer_capture": {"directory": str(capture), "stdout": _stream_evidence(capture / OUTER_STDOUT_FILENAME), "stderr": _stream_evidence(capture / OUTER_STDERR_FILENAME), "inner_capture_dir": str(capture / INNER_DIRNAME)},
        "child": {"pid": child_pid, "command": list(command), "terminal": dict(terminal)},
        "claim_boundary": {"component_only": True, "l1_suffix_or_moe_executed": False, "complete_layer_executed": False, "token_generated": False, "decoder_started": False, "server_or_watcher_started": False},
    }
    if capture_error is not None:
        payload["capture_error"] = capture_error
    return _seal_and_verify(payload)


def _replay(capture: Path, identity: str) -> dict[str, Any]:
    terminal = capture / OUTER_TERMINAL_FILENAME
    if not terminal.is_file():
        raise JointLifecycleError("capture directory exists without terminal receipt; no second child is allowed")
    document = _read_bound(terminal, "joint outer terminal", OUTER_SCHEMA, None).document
    if document.get("source_binding", {}).get("outer_preflight", {}).get("document_sha256") is None:
        raise JointLifecycleError("joint outer terminal lacks source binding")
    if document.get("launch_identity_sha256", identity) != identity:
        raise JointLifecycleError("capture directory belongs to another launch identity")
    return document


def run_one_shot(config: CaptureConfig, *, test_only: bool) -> dict[str, Any]:
    """Run exactly one outer-reaped child after all sealed lifecycle checks.

    ``test_only`` exists so the focused suite can exercise the exact create-
    new/replay/reap/terminal path with a disposable child.  Production callers
    must pass ``False`` and are expected to provide the separately validated
    strict host command; this module itself never selects a host binary.
    """
    if not config.child_command:
        raise JointLifecycleError("one-shot reaper needs one explicit child command")
    if not 1 <= config.workers <= 4 or not 1.0 <= config.timeout_seconds <= 7200.0:
        raise JointLifecycleError("test reaper workers/timeout are out of bounds")
    outer = _read_outer_preflight(config.outer_preflight)
    execution = _read_execution_binding(config.execution_binding, outer)
    lease = _validate_lease(config.lease_receipt, outer, execution, config.watcher_hold)
    identity = _launch_identity(lease, config.capture_dir)
    if config.capture_dir.exists():
        return _replay(config.capture_dir, identity)
    capture = _new_dir(config.capture_dir, "capture directory")
    launch = _outer_launch_authority(lease, capture, identity, config.workers)
    _write_new(capture / OUTER_LAUNCH_FILENAME, launch)
    if test_only:
        started_status = "STARTED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_OUTER_ONE_SHOT_TEST_ONLY"
        running_status = "RUNNING_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_OUTER_ONE_SHOT_TEST_ONLY"
        started_boundary: dict[str, Any] = {
            "test_only_lifecycle_exercise": True,
            "no_qwen80_device_work": True,
        }
        child_mode = "fake-child-test-only"
        device_child_spawned = False
    else:
        started_status = "STARTED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_OUTER_ONE_SHOT"
        running_status = "RUNNING_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_OUTER_ONE_SHOT"
        started_boundary = {
            "test_only_lifecycle_exercise": False,
            "strict_host_authority_required": True,
            "component_only": True,
            "l1_suffix_or_moe_authorized": False,
            "complete_layer_or_token_authorized": False,
        }
        child_mode = "strict-joint-host-metal"
        device_child_spawned = True
    _write_new(
        capture / RUNNING_FILENAME,
        _seal_and_verify(
            {
                "schema": OUTER_SCHEMA,
                "status": started_status,
                "launch_identity_sha256": identity,
                "device_child_spawned": device_child_spawned,
                "claim_boundary": started_boundary,
            }
        ),
    )
    command = list(config.child_command)
    child_pid: int | None = None
    capture_error: str | None = None
    with (capture / OUTER_STDOUT_FILENAME).open("xb") as stdout, (capture / OUTER_STDERR_FILENAME).open("xb") as stderr:
        try:
            child = subprocess.Popen(command, cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True, close_fds=True)
        except OSError as exc:
            terminal = _terminal(None, timed_out=False, spawn_error=f"{type(exc).__name__}: {exc}")
        else:
            child_pid = child.pid
            _write_new(
                capture / CHILD_FILENAME,
                _seal_and_verify(
                    {
                        "schema": OUTER_SCHEMA,
                        "status": running_status,
                        "pid": child_pid,
                        "parent_pid": os.getpid(),
                        "launch_identity_sha256": identity,
                        "mode": child_mode,
                    }
                ),
            )
            try:
                terminal = _terminal(child.wait(timeout=config.timeout_seconds), timed_out=False)
            except subprocess.TimeoutExpired:
                terminal = _terminal(_terminate_group(child), timed_out=True)
    for stream in (capture / OUTER_STDOUT_FILENAME, capture / OUTER_STDERR_FILENAME):
        if stream.stat().st_size > MAX_STREAM_BYTES:
            capture_error = f"{stream.name} exceeds bounded stream size"
    receipt = _terminal_receipt(capture=capture, lease=lease, identity=identity, command=command, child_pid=child_pid, terminal=terminal, capture_error=capture_error)
    receipt["launch_identity_sha256"] = identity
    receipt.pop("seal_sha256")
    receipt = _seal_and_verify(receipt)
    _write_new(capture / OUTER_TERMINAL_FILENAME, receipt)
    return receipt


def run_one_shot_for_test(config: CaptureConfig) -> dict[str, Any]:
    """Reap one disposable fake child for CPU-only lifecycle testing."""
    return run_one_shot(config, test_only=True)


def run_one_shot_production(config: CaptureConfig) -> dict[str, Any]:
    """Reap the separately validated strict joint host exactly once."""
    return run_one_shot(config, test_only=False)


def release_after_terminal(
    *, outer_terminal: Path, lease_receipt: Path, out: Path, release_issuer_identity_sha256: str
) -> dict[str, Any]:
    """Release a joint lease after any reaped outer terminal.

    A refused terminal still needs a durable release/no-retry record.  The
    success bit is retained explicitly so a release never promotes a refused
    child into an assessor-compatible capture.
    """
    if not _is_sha(release_issuer_identity_sha256):
        raise JointLifecycleError("release issuer identity must be a SHA-256")
    outer = _read_bound(outer_terminal, "joint outer terminal", OUTER_SCHEMA, None)
    terminal_status = outer.document.get("status")
    if terminal_status != OUTER_STATUS and (
        not isinstance(terminal_status, str) or not terminal_status.startswith(OUTER_REFUSED_PREFIX)
    ):
        raise JointLifecycleError("joint release outer terminal status is not a recognized terminal")
    lease = _read_bound(lease_receipt, "joint lease", LEASE_SCHEMA, LEASE_STATUS)
    if outer.document.get("lease_id") != lease.document.get("lease_id"):
        raise JointLifecycleError("joint release lease ID does not bind terminal")
    if not out.is_absolute() or out.exists():
        raise JointLifecycleError("release output must be a new absolute path")
    release = _seal_and_verify(
        {
            "schema": RELEASE_SCHEMA,
            "status": RELEASE_STATUS,
            "fixture_or_synthetic": False,
            "self_asserted": False,
            "issuer": {"role": "joint_component_lease_release_authority", "issuer_identity_sha256": release_issuer_identity_sha256},
            "outer_terminal": _identity(outer),
            "outer_terminal_status": terminal_status,
            "capture_succeeded": terminal_status == OUTER_STATUS,
            "lease_id": outer.document["lease_id"],
            "actual_release_performed": True,
            "released_after_outer_terminal": True,
            "lease_released": True,
            "automatic_retry_prohibited": True,
            "fresh_lease_required_for_any_future_gpu_work": True,
            "watcher_restart_or_transition_authorized": False,
        }
    )
    _write_new(out, release)
    return release


def release_after_terminal_for_test(
    *, outer_terminal: Path, lease_receipt: Path, out: Path, release_issuer_identity_sha256: str
) -> dict[str, Any]:
    """Compatibility wrapper for the focused disposable lifecycle tests."""
    return release_after_terminal(
        outer_terminal=outer_terminal,
        lease_receipt=lease_receipt,
        out=out,
        release_issuer_identity_sha256=release_issuer_identity_sha256,
    )
