#!/usr/bin/env python3
"""Receipt-last, one-shot lifecycle for the Qwen80 46-dispatch L0→L1 host.

This controller deliberately sits *outside* the host and the static full-L1
outer preflight.  A CPU preflight can inspect a frozen host/outer/raw-route
authority chain and a separately sealed resource admission, but it cannot
create a lease, reserve a directory, spawn a child, or touch Metal.

The only production entrypoint is ``--mode execute --execute-one-shot``.  It
requires a freshly sealed zero-swap, >=80%-free, no-competing-child admission;
then it creates one new lease, one launch directory, one replay reservation,
and one outer capture directory.  It reaps exactly one direct host child,
writes an outer terminal receipt last, and writes exactly one release receipt.
Existing/replayed/legacy capture state is never reused.

The focused tests use ``run_one_shot_for_test`` with a disposable fake child.
That helper is intentionally unreachable from the CLI and its terminal status
is explicitly test-only, so it cannot become production component evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.receipts import SealIntegrityError, seal, verify


# Frozen current CPU-only chain.  The values are deliberately explicit: a
# future remint must be consciously supplied as a new pin, never discovered by
# scanning a directory or silently substituted through a recovery wrapper.
CURRENT_HOST_BINARY_SHA256 = "9439c2bad1990892e82bba4aa336d60e4a23fdf1d488773c2b4844dd60e92039"
CURRENT_HOST_PREFLIGHT_SEAL_SHA256 = "b93247c3d4ab3ac99db9531f33be8743b94764a1720b567141529e136191dbb5"
CURRENT_OUTER_PREFLIGHT_SEAL_SHA256 = "dd405da3467371020fc8169696c8aea28106f60a08d851228e2892310d8ee19b"
RAW_L1_ROUTE_AUTHORITY_SEAL_SHA256 = "1be012d736659b4c0d761c6643be590e43dde495a2d05a3ad715928bac642722"

HOST_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_host_preflight.v1"
)
OUTER_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_outer_preflight.v1"
)
ROUTE_AUTHORITY_SCHEMA = "hawking.ascension.qwen80_source_token_l1_all_ten_route_authority.v1"
ROUTE_AUTHORITY_STATUS = (
    "SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L1_ALL_TEN_ROUTE_AUTHORITY_READY_FOR_"
    "SAME_RUNTIME_MOE_SUFFIX"
)

RESOURCE_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_capture_resource_admission.v1"
)
RESOURCE_STATUS = (
    "PREFLIGHTED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_CAPTURE_RESOURCE_"
    "ADMISSION_NOT_LEASED_OR_EXECUTED"
)
RESOURCE_REFUSED_STATUS = "REFUSED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_CAPTURE_RESOURCE_ADMISSION"

PREFLIGHT_SCHEMA = "hawking.ascension.qwen80_source_token_l0_l1_full_layer_capture_lifecycle_preflight.v1"
PREFLIGHT_STATUS = (
    "PREFLIGHTED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_CAPTURE_LIFECYCLE_"
    "NOT_LEASED_OR_EXECUTED"
)
PREFLIGHT_REFUSED_STATUS = (
    "REFUSED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_CAPTURE_LIFECYCLE_"
    "PRECONDITIONS_INCOMPLETE_NO_LEASE_OR_EXECUTION"
)

LEASE_SCHEMA = "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_quiet_metal_lease.v1"
LEASE_STATUS = (
    "GRANTED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_SAME_RUNTIME_COMPONENT_QUIET_METAL_LEASE"
)
OUTER_LAUNCH_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_outer_launch_authority.v1"
)
OUTER_LAUNCH_STATUS = (
    "AUTHORIZED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_MOE_LAYER_SAME_RUNTIME_"
    "OUTER_REAPED_ONE_SHOT_METAL_CHILD"
)
INNER_SCHEMA = "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_capture.v1"
INNER_SUCCESS_STATUS = (
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_COMPLETE_MOE_LAYER_"
    "SAME_RUNTIME_COMPONENT_ONLY"
)
INNER_REFUSED_STATUS = (
    "REFUSED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_COMPLETE_MOE_LAYER_"
    "SAME_RUNTIME_PHASE_ACCURATE_TERMINAL_FAILURE"
)
OUTER_TERMINAL_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_outer_capture.v1"
)
OUTER_TERMINAL_STATUS = (
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_SAME_RUNTIME_"
    "OUTER_TERMINAL_COMPONENT_ONLY"
)
OUTER_TERMINAL_TEST_STATUS = (
    "TERMINAL_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_OUTER_FAKE_CHILD_TEST_ONLY"
)
OUTER_TERMINAL_REFUSED_PREFIX = (
    "REFUSED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_SAME_RUNTIME_OUTER_"
)
RELEASE_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_"
    "quiet_metal_lease_release.v1"
)
RELEASE_STATUS = (
    "RELEASED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_COMPONENT_QUIET_METAL_"
    "LEASE_AFTER_TERMINAL_CAPTURE"
)
REPLAY_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_capture_replay_reservation.v1"
)
REPLAY_STATUS = "RESERVED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_CAPTURE_ONE_SHOT"

LEASE_FILENAME = "full-l1-quiet-metal-lease.json"
LAUNCH_FILENAME = "outer-launch-authority.json"
REPLAY_FILENAME = "replay-reservation.json"
RUNNING_FILENAME = "outer-running.json"
CHILD_FILENAME = "child.json"
TERMINAL_FILENAME = "outer-terminal-receipt.json"
RELEASE_FILENAME = "quiet-metal-lease-release.json"
INNER_DIRNAME = "inner"
INNER_RECEIPT_FILENAME = "receipt.json"
STDOUT_FILENAME = "child.stdout.log"
STDERR_FILENAME = "child.stderr.log"

TOTAL_DISPATCHES = 46
L0_DISPATCHES = 23
L1_PREFIX_DISPATCHES = 9
L1_MOE_SUFFIX_DISPATCHES = 14
MIN_FREE_PERCENT = 80
MAX_RESOURCE_AGE_SECONDS = 300
MAX_TIMEOUT_SECONDS = 7200.0
MAX_JSON_BYTES = 100_000_000
MAX_STREAM_BYTES = 1_000_000


class FullL1LifecycleError(RuntimeError):
    """A full-L1 capture lifecycle prerequisite failed closed."""


@dataclass(frozen=True)
class BoundDocument:
    path: Path
    raw_sha256: str
    bytes: int
    document: dict[str, Any]
    seal_sha256: str


@dataclass(frozen=True)
class AuthorityPins:
    host_binary_sha256: str = CURRENT_HOST_BINARY_SHA256
    host_preflight_seal_sha256: str = CURRENT_HOST_PREFLIGHT_SEAL_SHA256
    outer_preflight_seal_sha256: str = CURRENT_OUTER_PREFLIGHT_SEAL_SHA256
    raw_l1_route_authority_seal_sha256: str = RAW_L1_ROUTE_AUTHORITY_SEAL_SHA256


@dataclass(frozen=True)
class AuthorityContext:
    host_preflight: BoundDocument
    outer_preflight: BoundDocument
    raw_l1_route_authority: BoundDocument
    host_binary: Path
    host_binary_bytes: int
    host_binary_sha256: str
    capture_body_wired: bool
    real_host_metal_cli_available: bool
    kernel_names: tuple[str, ...]


@dataclass(frozen=True)
class ExecuteConfig:
    host_preflight: Path
    outer_preflight: Path
    raw_l1_route_authority: Path
    host_binary: Path
    resource_admission: Path
    launch_dir: Path
    replay_dir: Path
    outer_capture_dir: Path
    workers: int = 1
    timeout_seconds: float = 7200.0
    pins: AuthorityPins = AuthorityPins()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FullL1LifecycleError(f"{label} must be an object")
    return dict(value)


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise FullL1LifecycleError(f"{label} must be an array")
    return list(value)


def _require_bool(document: Mapping[str, Any], field: str, expected: bool, label: str) -> None:
    if document.get(field) is not expected:
        raise FullL1LifecycleError(f"{label}.{field} must be {expected}")


def _require_int(document: Mapping[str, Any], field: str, expected: int, label: str) -> None:
    if document.get(field) != expected:
        raise FullL1LifecycleError(f"{label}.{field} must be {expected}")


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise FullL1LifecycleError(f"{label} must be an absolute path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise FullL1LifecycleError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FullL1LifecycleError(f"{label} must be a regular non-symlink file")
    if executable and not os.access(path, os.X_OK):
        raise FullL1LifecycleError(f"{label} must be executable")
    return path.resolve(strict=True)


def _read_sealed(
    path: Path,
    label: str,
    *,
    schema: str | None = None,
    statuses: Sequence[str] | None = None,
) -> BoundDocument:
    clean = _canonical_regular(path, label)
    raw = clean.read_bytes()
    if len(raw) > MAX_JSON_BYTES:
        raise FullL1LifecycleError(f"{label} exceeds the bounded JSON size")
    try:
        parsed = json.loads(raw.decode("utf-8"))
        verified = verify(_mapping(parsed, label), label=label)
    except (UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise FullL1LifecycleError(f"{label} is not a valid sealed JSON document: {exc}") from exc
    document = _mapping(verified, label)
    if schema is not None and document.get("schema") != schema:
        raise FullL1LifecycleError(f"{label}.schema drifted")
    if statuses is not None and document.get("status") not in statuses:
        raise FullL1LifecycleError(f"{label}.status drifted")
    seal_sha256 = document.get("seal_sha256")
    if not _is_sha256(seal_sha256):
        raise FullL1LifecycleError(f"{label}.seal_sha256 must be a lowercase SHA-256")
    return BoundDocument(
        path=clean,
        raw_sha256=_sha256_bytes(raw),
        bytes=len(raw),
        document=document,
        seal_sha256=str(seal_sha256),
    )


def _evidence(bound: BoundDocument) -> dict[str, Any]:
    return {
        "path": str(bound.path),
        "present": True,
        "bytes": bound.bytes,
        "sha256": bound.raw_sha256,
        "document_sha256": bound.seal_sha256,
        "document_seal_sha256": bound.seal_sha256,
    }


def _matches_evidence(value: object, expected: BoundDocument, label: str) -> None:
    evidence = _mapping(value, label)
    expected_evidence = _evidence(expected)
    for field in (
        "path",
        "present",
        "bytes",
        "sha256",
        "document_sha256",
        "document_seal_sha256",
    ):
        if evidence.get(field) != expected_evidence[field]:
            raise FullL1LifecycleError(f"{label}.{field} drifted")


def _seal(document: Mapping[str, Any]) -> dict[str, Any]:
    result = seal(dict(document))
    try:
        verified = verify(result, label="new full-L1 lifecycle document")
    except SealIntegrityError as exc:  # pragma: no cover - defensive
        raise FullL1LifecycleError(f"new lifecycle document did not self-verify: {exc}") from exc
    return _mapping(verified, "new full-L1 lifecycle document")


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists():
        raise FullL1LifecycleError("output must be a new absolute path")
    try:
        parent_metadata = path.parent.lstat()
    except OSError as exc:
        raise FullL1LifecycleError(f"cannot stat output parent: {exc}") from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise FullL1LifecycleError("output parent must be an existing real directory")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _mkdir_new(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.exists():
        raise FullL1LifecycleError(f"{label} must be a new absolute directory")
    try:
        parent_metadata = path.parent.lstat()
    except OSError as exc:
        raise FullL1LifecycleError(f"cannot stat {label} parent: {exc}") from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise FullL1LifecycleError(f"{label} parent must be an existing real directory")
    try:
        path.mkdir(mode=0o750)
    except OSError as exc:
        raise FullL1LifecycleError(f"cannot create {label}: {exc}") from exc
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FullL1LifecycleError(f"new {label} is not a real directory")
    return path.resolve(strict=True)


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise FullL1LifecycleError(f"{label} must be an RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FullL1LifecycleError(f"{label} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise FullL1LifecycleError(f"{label} must include a timezone")
    return parsed


def _valid_cpu_preflight_status(value: object, prefix: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(prefix)
        and value.endswith("CPU_ONLY_NOT_LEASED_OR_EXECUTED")
    )


def _validate_route_authority(route: BoundDocument) -> None:
    root = route.document
    if root.get("schema") != ROUTE_AUTHORITY_SCHEMA or root.get("status") != ROUTE_AUTHORITY_STATUS:
        raise FullL1LifecycleError("raw L1 route authority schema/status drifted")
    evidence = _mapping(root.get("source_token_router_evidence"), "raw L1 route authority evidence")
    ids = _array(evidence.get("source_stable_route_ids"), "raw L1 route authority IDs")
    weights = _array(evidence.get("source_stable_normalized_weights"), "raw L1 route authority weights")
    if len(ids) != 10 or len(weights) != 10 or len(set(ids)) != 10:
        raise FullL1LifecycleError("raw L1 route authority must retain ten unique routes")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids):
        raise FullL1LifecycleError("raw L1 route authority route IDs drifted")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0 for value in weights):
        raise FullL1LifecycleError("raw L1 route authority weights drifted")
    if abs(sum(float(value) for value in weights) - 1.0) > 2.0e-6:
        raise FullL1LifecycleError("raw L1 route authority weights are not normalized")
    fixed = _array(root.get("fixed_l1_payloads"), "raw L1 route authority fixed payloads")
    waves = _array(root.get("deterministic_waves"), "raw L1 route authority waves")
    if len(fixed) != 6 or len(waves) != 10:
        raise FullL1LifecycleError("raw L1 route authority must retain six fixed and thirty routed payloads")
    for index, value in enumerate(waves):
        wave = _mapping(value, f"raw L1 route authority wave {index}")
        if wave.get("wave_index") != index or wave.get("layer") != 1 or wave.get("expert_id") != ids[index]:
            raise FullL1LifecycleError("raw L1 route authority wave ordering drifted")
        for role in ("gate", "up", "down"):
            descriptor = _mapping(wave.get(role), f"raw L1 route authority wave {index}.{role}")
            for field in ("artifact_sha256", "direct_packed_payload_sha256", "header_sha256"):
                if not _is_sha256(descriptor.get(field)):
                    raise FullL1LifecycleError(
                        f"raw L1 route authority wave {index}.{role}.{field} drifted"
                    )


def _kernel_names(graph: Mapping[str, Any], label: str) -> tuple[str, ...]:
    trace = _array(graph.get("exact_kernel_trace"), f"{label}.exact_kernel_trace")
    if len(trace) != TOTAL_DISPATCHES:
        raise FullL1LifecycleError(f"{label} must retain exactly 46 kernels")
    names: list[str] = []
    for ordinal, value in enumerate(trace):
        entry = _mapping(value, f"{label}.exact_kernel_trace[{ordinal}]")
        if entry.get("ordinal") != ordinal or not isinstance(entry.get("kernel"), str):
            raise FullL1LifecycleError(f"{label} kernel trace ordinal/name drifted")
        names.append(str(entry["kernel"]))
    return tuple(names)


def _validate_host_preflight(
    host: BoundDocument, route: BoundDocument, host_binary: Path, host_binary_bytes: int, host_binary_sha256: str
) -> tuple[bool, tuple[str, ...]]:
    root = host.document
    if root.get("schema") != HOST_PREFLIGHT_SCHEMA or not _valid_cpu_preflight_status(
        root.get("status"), "COMPILED_QWEN80_SOURCE_TOKEN_"
    ):
        raise FullL1LifecycleError("host preflight schema/status is not the current CPU-only host contract")
    binary = _mapping(root.get("host_binary"), "host preflight.host_binary")
    if (
        binary.get("path") != str(host_binary)
        or binary.get("bytes") != host_binary_bytes
        or binary.get("sha256") != host_binary_sha256
    ):
        raise FullL1LifecycleError("host preflight is not bound to the current host binary")
    authority = _mapping(root.get("l1_route_payload_authority"), "host preflight route authority")
    binding = _mapping(authority.get("binding"), "host preflight raw route authority binding")
    expected_route = _evidence(route)
    # The strict host later consumes this same binding through
    # `require_full_binding_matches`, which requires complete raw evidence.
    # Refuse a CPU preflight missing any part of that future host ABI rather
    # than issuing a lease that is guaranteed to fail before inner capture.
    for field in ("path", "present", "bytes", "sha256", "document_sha256", "document_seal_sha256"):
        if binding.get(field) != expected_route[field]:
            raise FullL1LifecycleError(f"host preflight raw route authority.{field} drifted")
    graph = _mapping(root.get("future_joint_command_graph"), "host preflight command graph")
    for field, expected in (
        ("source_token_id", 1),
        ("l0_reencode_dispatches", L0_DISPATCHES),
        ("l1_prefix_dispatches", L1_PREFIX_DISPATCHES),
        ("l1_moe_suffix_dispatches", L1_MOE_SUFFIX_DISPATCHES),
        ("total_dispatches", TOTAL_DISPATCHES),
    ):
        _require_int(graph, field, expected, "host preflight command graph")
    _require_bool(graph, "single_fence_after_all_dispatches_required", True, "host preflight command graph")
    _require_bool(graph, "non_timed_structural_trace_required", True, "host preflight command graph")
    names = _kernel_names(graph, "host preflight command graph")
    gate = _mapping(root.get("future_metal_entrypoint"), "host preflight Metal gate")
    for field in (
        "explicit_mode_required",
        "default_execution_disabled",
        "requires_new_full_l1_lease",
        "requires_sealed_outer_launch_authority",
        "requires_fresh_outer_and_inner_capture_directories",
        "self_hashes_current_executable",
        "no_device_execution_in_this_cpu_preflight",
    ):
        _require_bool(gate, field, True, "host preflight Metal gate")
    if not isinstance(gate.get("capture_body_wired"), bool):
        raise FullL1LifecycleError("host preflight Metal gate.capture_body_wired must be boolean")
    receipt = _mapping(root.get("future_inner_receipt_contract"), "host preflight inner receipt contract")
    if receipt.get("schema") != INNER_SCHEMA or receipt.get("status") != INNER_SUCCESS_STATUS:
        raise FullL1LifecycleError("host preflight inner receipt ABI drifted")
    boundary = _mapping(root.get("claim_boundary"), "host preflight claim boundary")
    for field in (
        "catalog_or_payload_scan_performed",
        "metal_context_or_dispatch_performed",
        "lease_issued_or_consumed",
        "watcher_server_hcli_or_runtime_changed",
        "complete_layer_or_token_decoder_claim_earned",
        "tps_tg_or_tournament_claim_earned",
    ):
        _require_bool(boundary, field, False, "host preflight claim boundary")
    return bool(gate["capture_body_wired"]), names


def _validate_outer_preflight(
    outer: BoundDocument,
    host: BoundDocument,
    route: BoundDocument,
    host_binary: Path,
    host_binary_bytes: int,
    host_binary_sha256: str,
    host_kernel_names: tuple[str, ...],
    host_capture_body_wired: bool,
) -> tuple[bool, bool, tuple[str, ...]]:
    root = outer.document
    if root.get("schema") != OUTER_PREFLIGHT_SCHEMA or not _valid_cpu_preflight_status(
        root.get("status"), "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_"
    ):
        raise FullL1LifecycleError("outer preflight schema/status is not the current CPU-only outer contract")
    _matches_evidence(root.get("host_preflight"), host, "outer preflight.host_preflight")
    outer_binary = _mapping(root.get("host_binary"), "outer preflight.host_binary")
    if (
        outer_binary.get("path") != str(host_binary)
        or outer_binary.get("bytes") != host_binary_bytes
        or outer_binary.get("sha256") != host_binary_sha256
    ):
        raise FullL1LifecycleError("outer preflight host binary drifted")
    _matches_evidence(
        root.get("original_l1_route_authority"), route, "outer preflight original raw L1 route authority"
    )
    scope = _mapping(root.get("exact_component_scope"), "outer preflight exact component scope")
    for field, expected in (
        ("source_token_id", 1),
        ("l0_reencode_dispatches", L0_DISPATCHES),
        ("l1_prefix_dispatches", L1_PREFIX_DISPATCHES),
        ("l1_moe_suffix_dispatches", L1_MOE_SUFFIX_DISPATCHES),
        ("total_dispatches", TOTAL_DISPATCHES),
    ):
        _require_int(scope, field, expected, "outer preflight exact component scope")
    _require_bool(scope, "one_fence_required", True, "outer preflight exact component scope")
    _require_bool(scope, "non_timed_exact_trace_required", True, "outer preflight exact component scope")
    outer_names = tuple(_array(scope.get("kernel_names"), "outer preflight kernel names"))
    if len(outer_names) != TOTAL_DISPATCHES or any(not isinstance(name, str) for name in outer_names):
        raise FullL1LifecycleError("outer preflight kernel names drifted")
    if outer_names != host_kernel_names:
        raise FullL1LifecycleError("host/outer exact 46-kernel traces disagree")
    gate = _mapping(root.get("future_metal_entrypoint"), "outer preflight Metal gate")
    for field in (
        "explicit_mode_required",
        "default_execution_disabled",
        "requires_new_full_l1_lease",
        "requires_sealed_outer_launch_authority",
        "requires_fresh_outer_and_inner_capture_directories",
        "self_hashes_current_executable",
        "no_device_execution_in_this_cpu_preflight",
    ):
        _require_bool(gate, field, True, "outer preflight Metal gate")
    if gate.get("capture_body_wired") is not host_capture_body_wired:
        raise FullL1LifecycleError("host/outer capture-body wiring disagrees")
    lifecycle = _mapping(root.get("lifecycle"), "outer preflight lifecycle")
    for field in (
        "replay_guard_required",
        "one_child_process_required",
        "outer_reaped_terminal_required",
    ):
        _require_bool(lifecycle, field, True, "outer preflight lifecycle")
    _require_bool(lifecycle, "automatic_retry_authorized", False, "outer preflight lifecycle")
    # The in-module outer helper remains intentionally fake-child-only.  This
    # separate controller records that it is the required production wrapper.
    if not isinstance(lifecycle.get("real_host_metal_cli_available"), bool):
        raise FullL1LifecycleError("outer preflight lifecycle.real_host_metal_cli_available must be boolean")
    return bool(gate["capture_body_wired"]), bool(lifecycle["real_host_metal_cli_available"]), outer_names


def load_authority_context(
    *,
    host_preflight: Path,
    outer_preflight: Path,
    raw_l1_route_authority: Path,
    host_binary: Path,
    pins: AuthorityPins = AuthorityPins(),
) -> AuthorityContext:
    """Load exactly one pinned full-L1 authority chain without spawning work."""
    for value, label in (
        (pins.host_binary_sha256, "host binary pin"),
        (pins.host_preflight_seal_sha256, "host preflight pin"),
        (pins.outer_preflight_seal_sha256, "outer preflight pin"),
        (pins.raw_l1_route_authority_seal_sha256, "raw L1 route authority pin"),
    ):
        if not _is_sha256(value):
            raise FullL1LifecycleError(f"{label} must be a lowercase SHA-256")
    clean_host = _canonical_regular(host_binary, "full-L1 host binary", executable=True)
    host_raw = clean_host.read_bytes()
    host_sha = _sha256_bytes(host_raw)
    if host_sha != pins.host_binary_sha256:
        raise FullL1LifecycleError("current host binary SHA does not match the explicit frozen pin")
    host = _read_sealed(host_preflight, "full-L1 host preflight", schema=HOST_PREFLIGHT_SCHEMA)
    outer = _read_sealed(outer_preflight, "full-L1 outer preflight", schema=OUTER_PREFLIGHT_SCHEMA)
    route = _read_sealed(
        raw_l1_route_authority,
        "original raw L1 route authority",
        schema=ROUTE_AUTHORITY_SCHEMA,
        statuses=(ROUTE_AUTHORITY_STATUS,),
    )
    if host.seal_sha256 != pins.host_preflight_seal_sha256:
        raise FullL1LifecycleError("host preflight seal does not match the explicit frozen pin")
    if outer.seal_sha256 != pins.outer_preflight_seal_sha256:
        raise FullL1LifecycleError("outer preflight seal does not match the explicit frozen pin")
    if route.seal_sha256 != pins.raw_l1_route_authority_seal_sha256:
        raise FullL1LifecycleError("raw L1 route authority seal does not match the direct frozen pin")
    _validate_route_authority(route)
    capture_body_wired, host_names = _validate_host_preflight(
        host, route, clean_host, len(host_raw), host_sha
    )
    outer_wired, real_cli, outer_names = _validate_outer_preflight(
        outer,
        host,
        route,
        clean_host,
        len(host_raw),
        host_sha,
        host_names,
        capture_body_wired,
    )
    if outer_wired is not capture_body_wired:
        raise FullL1LifecycleError("host/outer capture body state drifted")
    return AuthorityContext(
        host_preflight=host,
        outer_preflight=outer,
        raw_l1_route_authority=route,
        host_binary=clean_host,
        host_binary_bytes=len(host_raw),
        host_binary_sha256=host_sha,
        capture_body_wired=capture_body_wired,
        real_host_metal_cli_available=real_cli,
        kernel_names=outer_names,
    )


def evaluate_resource_snapshot(snapshot: Mapping[str, Any]) -> list[str]:
    """Return all safety blockers without observing or changing the machine."""
    root = _mapping(snapshot, "resource snapshot")
    blockers: list[str] = []
    free = root.get("memory_free_percent")
    if isinstance(free, bool) or not isinstance(free, int) or free < MIN_FREE_PERCENT:
        blockers.append(f"memory free percentage is below {MIN_FREE_PERCENT}")
    swap = root.get("swap_used_bytes")
    if isinstance(swap, bool) or not isinstance(swap, int) or swap != 0:
        blockers.append("swap must be exactly zero")
    watcher_pids = root.get("q80_watcher_parent_pids")
    if not isinstance(watcher_pids, list) or len(watcher_pids) != 1 or any(
        isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in watcher_pids
    ):
        blockers.append("exactly one held Q80 watcher parent is required")
    if root.get("watcher_hold_active") is not True:
        blockers.append("Q80 watcher hold is not active")
    for field, label in (
        ("q80_full_l1_capture_children", "Q80 full-L1 capture child"),
        ("q30_capture_children", "Q30 capture child"),
    ):
        children = root.get(field)
        if not isinstance(children, list):
            blockers.append(f"{label} observation is invalid")
        elif children:
            blockers.append(f"{label} is already active")
    return blockers


def build_resource_admission(
    *, context: AuthorityContext, snapshot: Mapping[str, Any], recorded_at: str | None = None
) -> dict[str, Any]:
    """Build a sealed admission from a caller-provided read-only snapshot.

    It never samples processes, opens Metal, issues a lease, or creates a
    directory.  A separate coordinator may collect the snapshot immediately
    before the intended future launch.
    """
    blockers = evaluate_resource_snapshot(snapshot)
    return _seal(
        {
            "schema": RESOURCE_SCHEMA,
            "status": RESOURCE_STATUS if not blockers else RESOURCE_REFUSED_STATUS,
            "recorded_at": recorded_at or _utc_now(),
            "prepared": not blockers,
            "blockers": blockers,
            "minimum_memory_free_percent": MIN_FREE_PERCENT,
            "maximum_resource_age_seconds": MAX_RESOURCE_AGE_SECONDS,
            "host_preflight": _evidence(context.host_preflight),
            "outer_preflight": _evidence(context.outer_preflight),
            "original_raw_l1_route_authority": _evidence(context.raw_l1_route_authority),
            "host_binary": {
                "path": str(context.host_binary),
                "present": True,
                "bytes": context.host_binary_bytes,
                "sha256": context.host_binary_sha256,
            },
            "resource_snapshot": dict(snapshot),
            "claim_boundary": {
                "cpu_file_only_resource_admission": True,
                "lease_issued_or_consumed": False,
                "metal_or_gpu_activity_performed": False,
                "server_watcher_hcli_tps_tg_or_tournament_action": False,
                "legacy_lease_or_capture_accepted": False,
            },
        }
    )


def _read_resource_admission(
    *, context: AuthorityContext, path: Path, now: datetime | None = None
) -> BoundDocument:
    resource = _read_sealed(
        path,
        "full-L1 resource admission",
        schema=RESOURCE_SCHEMA,
        statuses=(RESOURCE_STATUS,),
    )
    root = resource.document
    if root.get("prepared") is not True or root.get("blockers") != []:
        raise FullL1LifecycleError("resource admission is not green")
    if root.get("minimum_memory_free_percent") != MIN_FREE_PERCENT:
        raise FullL1LifecycleError("resource admission memory floor drifted")
    if root.get("maximum_resource_age_seconds") != MAX_RESOURCE_AGE_SECONDS:
        raise FullL1LifecycleError("resource admission freshness window drifted")
    observed = _parse_utc(root.get("recorded_at"), "resource admission.recorded_at")
    reference = now or datetime.now(timezone.utc)
    age = (reference - observed).total_seconds()
    if age < 0 or age > MAX_RESOURCE_AGE_SECONDS:
        raise FullL1LifecycleError("resource admission is stale")
    _matches_evidence(root.get("host_preflight"), context.host_preflight, "resource admission.host_preflight")
    _matches_evidence(root.get("outer_preflight"), context.outer_preflight, "resource admission.outer_preflight")
    _matches_evidence(
        root.get("original_raw_l1_route_authority"),
        context.raw_l1_route_authority,
        "resource admission original raw L1 route authority",
    )
    binary = _mapping(root.get("host_binary"), "resource admission.host_binary")
    if (
        binary.get("path") != str(context.host_binary)
        or binary.get("bytes") != context.host_binary_bytes
        or binary.get("sha256") != context.host_binary_sha256
    ):
        raise FullL1LifecycleError("resource admission host binary binding drifted")
    blockers = evaluate_resource_snapshot(_mapping(root.get("resource_snapshot"), "resource admission snapshot"))
    if blockers:
        raise FullL1LifecycleError("resource admission safety snapshot refused: " + "; ".join(blockers))
    boundary = _mapping(root.get("claim_boundary"), "resource admission claim boundary")
    for field in (
        "cpu_file_only_resource_admission",
        "legacy_lease_or_capture_accepted",
        "lease_issued_or_consumed",
        "metal_or_gpu_activity_performed",
        "server_watcher_hcli_tps_tg_or_tournament_action",
    ):
        expected = field in ("cpu_file_only_resource_admission", "legacy_lease_or_capture_accepted")
        if field == "legacy_lease_or_capture_accepted":
            expected = False
        _require_bool(boundary, field, expected, "resource admission claim boundary")
    return resource


def build_lifecycle_preflight(
    *,
    host_preflight: Path,
    outer_preflight: Path,
    raw_l1_route_authority: Path,
    host_binary: Path,
    resource_admission: Path,
    pins: AuthorityPins = AuthorityPins(),
    now: datetime | None = None,
) -> dict[str, Any]:
    """Seal a CPU/file-only future-execution decision without creating state."""
    context: AuthorityContext | None = None
    resource: BoundDocument | None = None
    blockers: list[str] = []
    try:
        context = load_authority_context(
            host_preflight=host_preflight,
            outer_preflight=outer_preflight,
            raw_l1_route_authority=raw_l1_route_authority,
            host_binary=host_binary,
            pins=pins,
        )
        resource = _read_resource_admission(context=context, path=resource_admission, now=now)
        if not context.capture_body_wired:
            blockers.append("current host preflight says capture_body_wired=false")
        if not context.real_host_metal_cli_available:
            blockers.append("current outer preflight says real_host_metal_cli_available=false")
    except FullL1LifecycleError as exc:
        blockers.append(str(exc))
    return _seal(
        {
            "schema": PREFLIGHT_SCHEMA,
            "status": PREFLIGHT_STATUS if not blockers else PREFLIGHT_REFUSED_STATUS,
            "prepared": not blockers,
            "execute_one_shot_required": True,
            "production_lifecycle_wrapper_required": True,
            "lease_issued": False,
            "launch_replay_or_capture_directory_created": False,
            "child_spawned": False,
            "host_preflight": _evidence(context.host_preflight) if context else {"present": False},
            "outer_preflight": _evidence(context.outer_preflight) if context else {"present": False},
            "original_raw_l1_route_authority": _evidence(context.raw_l1_route_authority)
            if context
            else {"present": False},
            "resource_admission": _evidence(resource) if resource else {"present": False},
            "blockers": blockers,
            "claim_boundary": {
                "cpu_file_only_preflight": True,
                "catalog_or_payload_scan_performed": False,
                "metal_context_or_dispatch_performed": False,
                "lease_issued_or_consumed": False,
                "server_watcher_hcli_tps_tg_or_tournament_action": False,
                "capture_component_or_tournament_claim_earned": False,
            },
        }
    )


def _validate_new_layout(config: ExecuteConfig) -> tuple[Path, Path, Path]:
    paths = (config.launch_dir, config.replay_dir, config.outer_capture_dir)
    labels = ("launch directory", "replay directory", "outer capture directory")
    if len({str(path) for path in paths}) != len(paths):
        raise FullL1LifecycleError("launch/replay/outer-capture directories must be distinct")
    resolved_parents: list[Path] = []
    for path, label in zip(paths, labels):
        if not path.is_absolute() or path.exists():
            raise FullL1LifecycleError(f"{label} must be a new absolute directory")
        try:
            parent = path.parent.resolve(strict=True)
        except OSError as exc:
            raise FullL1LifecycleError(f"cannot resolve {label} parent: {exc}") from exc
        if not parent.is_dir() or parent.is_symlink():
            raise FullL1LifecycleError(f"{label} parent must be a real existing directory")
        resolved_parents.append(parent)
    # Do not allow a future caller to hide one lifecycle root inside another.
    absolute = tuple(path.absolute() for path in paths)
    for index, path in enumerate(absolute):
        for other_index, other in enumerate(absolute):
            if index != other_index and other in path.parents:
                raise FullL1LifecycleError("launch/replay/outer-capture directories must not be nested")
    return paths


def _lease_document(context: AuthorityContext, resource: BoundDocument, *, lease_id: str) -> dict[str, Any]:
    return _seal(
        {
            "schema": LEASE_SCHEMA,
            "status": LEASE_STATUS,
            "recorded_at": _utc_now(),
            "lease_id": lease_id,
            "fresh_for_this_exact_launch": True,
            "one_shot": True,
            "new_capture_root_required": True,
            "existing_output_reuse_forbidden": True,
            "replay_or_relaunch_forbidden": True,
            "outer_reaped_terminal_required": True,
            "receipt_last_required": True,
            "automatic_retry_authorized": False,
            "host_binary": {
                "path": str(context.host_binary),
                "present": True,
                "bytes": context.host_binary_bytes,
                "sha256": context.host_binary_sha256,
            },
            "host_preflight": _evidence(context.host_preflight),
            "outer_preflight": _evidence(context.outer_preflight),
            "original_raw_l1_route_authority": _evidence(context.raw_l1_route_authority),
            "fresh_resource_admission": _evidence(resource),
            "execution_policy": {
                "source_token_id": 1,
                "l0_reencode_dispatches": L0_DISPATCHES,
                "l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
                "l1_moe_suffix_dispatches": L1_MOE_SUFFIX_DISPATCHES,
                "total_dispatches": TOTAL_DISPATCHES,
                # These exact names are consumed by the host's file-only
                # Metal gate before it can create its inner capture directory.
                "metal_mode_only": True,
                "non_timed_exact_46_dispatches_required": True,
                "one_fence_required": True,
                "l1_moe_suffix_allowed": True,
                "non_timed_exact_trace_required": True,
                "component_only": True,
                "automatic_retry_allowed": False,
            },
            "claim_boundary": {
                "source_catalog_scan_authorized": False,
                "server_hcli_tps_tg_or_tournament_authorized": False,
                "token_or_decoder_authorized": False,
                "legacy_lease_or_capture_accepted": False,
            },
        }
    )


def _lease_id(context: AuthorityContext, resource: BoundDocument, launch: Path) -> str:
    payload = {
        "schema": LEASE_SCHEMA,
        "host_binary_sha256": context.host_binary_sha256,
        "host_preflight_seal_sha256": context.host_preflight.seal_sha256,
        "outer_preflight_seal_sha256": context.outer_preflight.seal_sha256,
        "route_authority_seal_sha256": context.raw_l1_route_authority.seal_sha256,
        "resource_admission_seal_sha256": resource.seal_sha256,
        "launch_dir": str(launch),
        "random_nonce_sha256": _sha256_bytes(os.urandom(32)),
        "recorded_at": _utc_now(),
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _outer_launch_document(
    *,
    context: AuthorityContext,
    resource: BoundDocument,
    lease: BoundDocument,
    lease_id: str,
    outer_capture_dir: Path,
    inner_capture_dir: Path,
    workers: int,
) -> dict[str, Any]:
    return _seal(
        {
            "schema": OUTER_LAUNCH_SCHEMA,
            "status": OUTER_LAUNCH_STATUS,
            "recorded_at": _utc_now(),
            "lease_id": lease_id,
            "lease": {**_evidence(lease), "lease_id": lease_id},
            # `lease_receipt` is the host ABI; the parallel `lease` object is
            # retained for human-readable outer terminal provenance.
            "lease_receipt": _evidence(lease),
            "host_preflight": _evidence(context.host_preflight),
            "outer_preflight": _evidence(context.outer_preflight),
            "original_raw_l1_route_authority": _evidence(context.raw_l1_route_authority),
            "fresh_resource_admission": _evidence(resource),
            "host_binary": {
                "path": str(context.host_binary),
                "present": True,
                "bytes": context.host_binary_bytes,
                "sha256": context.host_binary_sha256,
            },
            "planned_outer_capture_dir": str(outer_capture_dir),
            "planned_inner_capture_dir": str(inner_capture_dir),
            "workers": workers,
            "execution_policy": {
                "total_dispatches": TOTAL_DISPATCHES,
                "metal_mode_only": True,
                "non_timed_exact_46_dispatches_required": True,
                "one_fence_required": True,
                "l1_moe_suffix_allowed": True,
                "non_timed_exact_trace_required": True,
                "component_only": True,
                "automatic_retry_allowed": False,
                "raw_route_authority_consumed_directly": True,
            },
            "lifecycle": {
                "production_lifecycle_wrapper_required": True,
                "replay_guard_required": True,
                "one_child_process_required": True,
                "outer_reaped_terminal_required": True,
                "terminal_receipt_written_last_required": True,
                "outer_reaped_child_required": True,
                "terminal_receipt_written_last": True,
                "automatic_retry_prohibited": True,
                "replay_or_relaunch_forbidden": True,
            },
            "claim_boundary": {
                "complete_l1_component_only": True,
                "token_decoder_server_hcli_tps_tg_or_tournament_authorized": False,
            },
        }
    )


def _replay_document(
    *, context: AuthorityContext, lease: BoundDocument, lease_id: str, outer_capture_dir: Path
) -> dict[str, Any]:
    return _seal(
        {
            "schema": REPLAY_SCHEMA,
            "status": REPLAY_STATUS,
            "recorded_at": _utc_now(),
            "create_new_before_child": True,
            "one_child_maximum": True,
            "replay_or_relaunch_forbidden": True,
            "attempt": 1,
            "lease_id": lease_id,
            "lease": _evidence(lease),
            "outer_preflight": _evidence(context.outer_preflight),
            "planned_outer_capture_dir": str(outer_capture_dir),
            "claim_boundary": "Reservation only; no child or device execution has occurred.",
        }
    )


def _validate_host_metal_abi(
    *,
    context: AuthorityContext,
    lease: BoundDocument,
    lease_id: str,
    launch: BoundDocument,
    outer_capture_dir: Path,
    inner_capture_dir: Path,
    workers: int,
) -> None:
    """Mirror the host's file-only lease/launch gate before a child exists.

    This is intentionally redundant with the Rust host's own gate.  It turns
    a schema typo into a no-lease prelaunch refusal during focused tests and
    keeps the outer from knowingly spawning a child it already knows will
    reject before its inner capture directory can be created.
    """
    lease_root = lease.document
    if lease_root.get("schema") != LEASE_SCHEMA or lease_root.get("status") != LEASE_STATUS:
        raise FullL1LifecycleError("issued lease schema/status drifted")
    if lease_root.get("lease_id") != lease_id:
        raise FullL1LifecycleError("issued lease ID drifted")
    _matches_evidence(lease_root.get("outer_preflight"), context.outer_preflight, "issued lease.outer_preflight")
    binary = _mapping(lease_root.get("host_binary"), "issued lease.host_binary")
    if (
        binary.get("path") != str(context.host_binary)
        or binary.get("present") is not True
        or binary.get("bytes") != context.host_binary_bytes
        or binary.get("sha256") != context.host_binary_sha256
    ):
        raise FullL1LifecycleError("issued lease host binary drifted")
    lease_policy = _mapping(lease_root.get("execution_policy"), "issued lease.execution_policy")
    for field, expected in (
        ("metal_mode_only", True),
        ("non_timed_exact_46_dispatches_required", True),
        ("one_fence_required", True),
        ("component_only", True),
        ("l1_moe_suffix_allowed", True),
        ("automatic_retry_allowed", False),
    ):
        _require_bool(lease_policy, field, expected, "issued lease.execution_policy")
    launch_root = launch.document
    if launch_root.get("schema") != OUTER_LAUNCH_SCHEMA or launch_root.get("status") != OUTER_LAUNCH_STATUS:
        raise FullL1LifecycleError("issued outer launch schema/status drifted")
    _matches_evidence(
        launch_root.get("outer_preflight"), context.outer_preflight, "issued outer launch.outer_preflight"
    )
    _matches_evidence(launch_root.get("lease_receipt"), lease, "issued outer launch.lease_receipt")
    if launch_root.get("lease_id") != lease_id:
        raise FullL1LifecycleError("issued outer launch lease ID drifted")
    launch_binary = _mapping(launch_root.get("host_binary"), "issued outer launch.host_binary")
    if (
        launch_binary.get("path") != str(context.host_binary)
        or launch_binary.get("present") is not True
        or launch_binary.get("bytes") != context.host_binary_bytes
        or launch_binary.get("sha256") != context.host_binary_sha256
    ):
        raise FullL1LifecycleError("issued outer launch host binary drifted")
    if (
        launch_root.get("planned_outer_capture_dir") != str(outer_capture_dir)
        or launch_root.get("planned_inner_capture_dir") != str(inner_capture_dir)
        or launch_root.get("workers") != workers
    ):
        raise FullL1LifecycleError("issued outer launch capture paths/workers drifted")
    launch_policy = _mapping(launch_root.get("execution_policy"), "issued outer launch.execution_policy")
    for field, expected in (
        ("metal_mode_only", True),
        ("non_timed_exact_46_dispatches_required", True),
        ("one_fence_required", True),
        ("component_only", True),
        ("l1_moe_suffix_allowed", True),
        ("automatic_retry_allowed", False),
    ):
        _require_bool(launch_policy, field, expected, "issued outer launch.execution_policy")
    launch_lifecycle = _mapping(launch_root.get("lifecycle"), "issued outer launch.lifecycle")
    for field in (
        "replay_guard_required",
        "one_child_process_required",
        "outer_reaped_terminal_required",
        "terminal_receipt_written_last_required",
    ):
        _require_bool(launch_lifecycle, field, True, "issued outer launch.lifecycle")


def _terminal(returncode: int | None, *, timed_out: bool, spawn_error: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "reaped": returncode is not None,
        "timed_out": timed_out,
        "returncode": returncode,
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


def _stream_evidence(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
        "within_max_stream_bytes": len(raw) <= MAX_STREAM_BYTES,
    }


def _child_command(
    *,
    context: AuthorityContext,
    lease_path: Path,
    launch_path: Path,
    outer_capture_dir: Path,
    inner_capture_dir: Path,
    workers: int,
    fake_child_command: Sequence[str] | None,
) -> list[str]:
    if fake_child_command is not None:
        if not fake_child_command:
            raise FullL1LifecycleError("fake child command must be non-empty")
        return [
            *fake_child_command,
            "--outer-preflight",
            str(context.outer_preflight.path),
            "--lease-receipt",
            str(lease_path),
            "--outer-launch-authority",
            str(launch_path),
            "--outer-capture-dir",
            str(outer_capture_dir),
            "--capture-dir",
            str(inner_capture_dir),
            "--workers",
            str(workers),
        ]
    return [
        str(context.host_binary),
        "--mode",
        "metal",
        "--outer-preflight",
        str(context.outer_preflight.path),
        "--lease-receipt",
        str(lease_path),
        "--outer-launch-authority",
        str(launch_path),
        "--outer-capture-dir",
        str(outer_capture_dir),
        "--capture-dir",
        str(inner_capture_dir),
        "--workers",
        str(workers),
    ]


def _run_child(
    *, command: Sequence[str], outer_capture_dir: Path, timeout_seconds: float
) -> tuple[dict[str, Any], int | None, dict[str, Any], dict[str, Any]]:
    stdout_path = outer_capture_dir / STDOUT_FILENAME
    stderr_path = outer_capture_dir / STDERR_FILENAME
    child_pid: int | None = None
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            child = subprocess.Popen(
                list(command),
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                close_fds=True,
            )
        except OSError as exc:
            terminal = _terminal(None, timed_out=False, spawn_error=f"{type(exc).__name__}: {exc}")
        else:
            child_pid = child.pid
            try:
                terminal = _terminal(child.wait(timeout=timeout_seconds), timed_out=False)
            except subprocess.TimeoutExpired:
                terminal = _terminal(_terminate_group(child), timed_out=True)
    return terminal, child_pid, _stream_evidence(stdout_path), _stream_evidence(stderr_path)


def _binding_for_path(path: Path, label: str) -> BoundDocument:
    return _read_sealed(path, label)


def _inner_binding_matches(value: object, expected: BoundDocument, label: str) -> None:
    binding = _mapping(value, label)
    # The host owns this ABI.  A lease wrapper may add lease_id next to the
    # usual raw/document/seal evidence, so compare the immutable evidence only.
    expected_evidence = _evidence(expected)
    for field in ("path", "bytes", "sha256", "document_sha256", "document_seal_sha256"):
        if binding.get(field) != expected_evidence[field]:
            raise FullL1LifecycleError(f"{label}.{field} drifted")


def _validate_inner_receipt(
    *,
    path: Path,
    context: AuthorityContext,
    lease: BoundDocument,
    lease_id: str,
    launch: BoundDocument,
    inner_capture_dir: Path,
    test_only: bool,
) -> tuple[BoundDocument, bool]:
    inner = _read_sealed(path, "full-L1 child receipt", schema=INNER_SCHEMA)
    root = inner.document
    status = root.get("status")
    if status not in (INNER_SUCCESS_STATUS, INNER_REFUSED_STATUS):
        raise FullL1LifecycleError("full-L1 child receipt has an unrecognized terminal status")
    fixture = root.get("fixture_or_synthetic")
    if test_only:
        _require_bool(root, "fixture_or_synthetic", True, "fake full-L1 child receipt")
    elif fixture is not False:
        raise FullL1LifecycleError("production full-L1 child receipt may not be synthetic")
    _inner_binding_matches(
        root.get("outer_preflight_binding"), context.outer_preflight, "child outer-preflight binding"
    )
    lease_binding = _mapping(root.get("full_l1_lease_binding"), "child full-L1 lease binding")
    if lease_binding.get("lease_id") != lease_id:
        raise FullL1LifecycleError("child lease ID drifted")
    _inner_binding_matches(lease_binding.get("receipt"), lease, "child lease receipt binding")
    _inner_binding_matches(
        root.get("outer_launch_authority_binding"), launch, "child outer-launch binding"
    )
    durable = _mapping(root.get("durable_capture"), "child durable capture")
    if durable.get("capture_directory") != str(inner_capture_dir):
        raise FullL1LifecycleError("child durable capture directory drifted")
    _require_bool(
        durable, "receipt_written_last_is_completion_marker", True, "child durable capture"
    )
    if status == INNER_REFUSED_STATUS:
        _mapping(root.get("execution_phase"), "phase-accurate child terminal execution phase")
        error = root.get("terminal_error")
        if not isinstance(error, str) or not error:
            raise FullL1LifecycleError("phase-accurate child terminal lacks terminal_error")
        return inner, False
    execution = _mapping(root.get("fresh_same_runtime_execution"), "child same-runtime execution")
    for field in (
        "fresh_runtime",
        "fresh_session",
        "same_runtime",
        "same_tcb",
        "l0_reencoded_in_this_capture",
        "l1_prefix_and_moe_suffix_in_this_capture",
        "route_guard_enforced_before_l1_moe_suffix",
    ):
        _require_bool(execution, field, True, "child same-runtime execution")
    for field, expected in (
        ("source_token_id", 1),
        ("l0_dispatches", L0_DISPATCHES),
        ("l1_prefix_dispatches", L1_PREFIX_DISPATCHES),
        ("l1_moe_suffix_dispatches", L1_MOE_SUFFIX_DISPATCHES),
        ("total_dispatches", TOTAL_DISPATCHES),
        ("fence_count", 1),
    ):
        _require_int(execution, field, expected, "child same-runtime execution")
    trace = _mapping(root.get("structural_kernel_trace"), "child structural kernel trace")
    _require_bool(trace, "non_timed", True, "child structural kernel trace")
    _require_bool(trace, "exact_order", True, "child structural kernel trace")
    names = tuple(_array(trace.get("kernel_names"), "child structural kernel names"))
    if names != context.kernel_names:
        raise FullL1LifecycleError("child structural trace is not the frozen exact 46-kernel trace")
    fence = _mapping(root.get("single_fence"), "child single fence")
    for field, expected in (
        ("only_command_buffer_consumed", True),
        ("fence_succeeded", True),
        ("readbacks_after_fence", True),
        ("append_after_fence_possible", False),
        ("fence_count", 1),
    ):
        if isinstance(expected, bool):
            _require_bool(fence, field, expected, "child single fence")
        else:
            _require_int(fence, field, expected, "child single fence")
    readbacks = _mapping(root.get("l1_completion_readbacks"), "child L1 completion readbacks")
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
            raise FullL1LifecycleError(f"child lacks required L0/L1 readback {field}")
    boundary = _mapping(root.get("claim_boundary"), "child claim boundary")
    _require_bool(boundary, "complete_l1_component_only", True, "child claim boundary")
    for field in (
        "token_generated",
        "decoder_started",
        "server_or_watcher_started",
        "tps_or_tg_measured",
        "tournament_started",
        "next_layer_executed",
    ):
        _require_bool(boundary, field, False, "child claim boundary")
    return inner, True


def _terminal_status(
    *, terminal: Mapping[str, Any], inner_success: bool, inner_error: str | None, test_only: bool
) -> str:
    if test_only:
        return OUTER_TERMINAL_TEST_STATUS
    if terminal.get("spawn_error"):
        return f"{OUTER_TERMINAL_REFUSED_PREFIX}CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return f"{OUTER_TERMINAL_REFUSED_PREFIX}CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return f"{OUTER_TERMINAL_REFUSED_PREFIX}CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return f"{OUTER_TERMINAL_REFUSED_PREFIX}CHILD_NONZERO"
    if inner_error is not None:
        return f"{OUTER_TERMINAL_REFUSED_PREFIX}INVALID_INNER_RECEIPT"
    if not inner_success:
        return f"{OUTER_TERMINAL_REFUSED_PREFIX}PHASE_ACCURATE_CHILD_REFUSAL"
    return OUTER_TERMINAL_STATUS


def _outer_terminal_document(
    *,
    context: AuthorityContext,
    resource: BoundDocument,
    lease: BoundDocument,
    lease_id: str,
    launch: BoundDocument,
    replay: BoundDocument,
    outer_capture_dir: Path,
    command: Sequence[str],
    child_pid: int | None,
    terminal: Mapping[str, Any],
    stdout: Mapping[str, Any],
    stderr: Mapping[str, Any],
    inner: BoundDocument | None,
    inner_success: bool,
    inner_error: str | None,
    test_only: bool,
) -> dict[str, Any]:
    status = _terminal_status(
        terminal=terminal, inner_success=inner_success, inner_error=inner_error, test_only=test_only
    )
    return _seal(
        {
            "schema": OUTER_TERMINAL_SCHEMA,
            "status": status,
            "recorded_at": _utc_now(),
            "test_only_fake_child": test_only,
            # Same assertion the strict host writes into its inner receipt, under
            # the name every consuming assessor reads.  The outer previously
            # published this fact only as ``test_only_fake_child``, so a
            # conforming real capture could not be bound by an assessor that
            # requires ``fixture_or_synthetic`` on both halves of the pair.
            "fixture_or_synthetic": test_only,
            # This terminal is written by the reaping lifecycle from the child's
            # observed exit and its sealed inner receipt, never attested by the
            # child about itself.  Hoist the lease identity to the root as well;
            # it is already carried under "lease", and the assessor reads it here.
            "self_asserted": False,
            "lease_id": lease_id,
            "production_lifecycle_wrapper_required": True,
            "host_preflight": _evidence(context.host_preflight),
            "outer_preflight": _evidence(context.outer_preflight),
            "original_raw_l1_route_authority": _evidence(context.raw_l1_route_authority),
            "fresh_resource_admission": _evidence(resource),
            "lease": {**_evidence(lease), "lease_id": lease_id},
            "outer_launch_authority": _evidence(launch),
            "replay_reservation": _evidence(replay),
            "child_terminal": {
                "reaped": terminal.get("reaped") is True,
                "timed_out": terminal.get("timed_out") is True,
                "exit_code": terminal.get("exit_code"),
                "signal": terminal.get("signal"),
                "terminal_receipt_written_last": True,
                "automatic_retry_disabled": True,
                "lease_reuse_prohibited": True,
            },
            "child": {"pid": child_pid, "command": list(command), "terminal": dict(terminal)},
            "outer_capture": {
                "directory": str(outer_capture_dir),
                "stdout": dict(stdout),
                "stderr": dict(stderr),
                "inner_capture_dir": str(outer_capture_dir / INNER_DIRNAME),
            },
            "inner_capture": _evidence(inner) if inner else {"present": False},
            "inner_capture_success": inner_success,
            "inner_capture_validation_error": inner_error,
            "claim_boundary": {
                "complete_l1_component_only": status == OUTER_TERMINAL_STATUS,
                "token_generated": False,
                "decoder_started": False,
                "server_or_watcher_started": False,
                "tps_or_tg_measured": False,
                "tournament_started": False,
            },
        }
    )


def _release_document(
    *, lease: BoundDocument, lease_id: str, terminal: BoundDocument
) -> dict[str, Any]:
    return _seal(
        {
            "schema": RELEASE_SCHEMA,
            "status": RELEASE_STATUS,
            "recorded_at": _utc_now(),
            "release_after_outer_terminal": True,
            "one_shot_lease_finalized": True,
            "retry_or_relaunch_forbidden": True,
            "exactly_one_release_for_this_lease": True,
            # The same facts above, restated under the names the consuming
            # completion assessor reads.  No new claim is made here: a release
            # is only written after a real reaped outer terminal, this lifecycle
            # finalizes the one-shot lease exactly once, retry/relaunch is
            # forbidden, and a release never authorizes a watcher restart.
            # The producer and the assessor were written against each other's
            # intended shapes and never executed together, so the names drifted
            # (note release_after_ vs released_after_).
            "actual_release_performed": True,
            "released_after_outer_terminal": True,
            "lease_released": True,
            "automatic_retry_prohibited": True,
            "fresh_lease_required_for_any_future_gpu_work": True,
            "watcher_restart_or_transition_authorized": False,
            "lease_id": lease_id,
            "lease": _evidence(lease),
            "outer_terminal": _evidence(terminal),
            "outer_terminal_status": terminal.document.get("status"),
            "capture_succeeded": terminal.document.get("status") == OUTER_TERMINAL_STATUS,
            "claim_boundary": "A release cannot authorize retry, token/decoder, server/HCLI/TPS/TG, or tournament work.",
        }
    )


def _execution_context(config: ExecuteConfig) -> tuple[AuthorityContext, BoundDocument]:
    if not 1 <= config.workers <= 4:
        raise FullL1LifecycleError("workers must be in 1..4")
    if not 1.0 <= config.timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise FullL1LifecycleError("timeout must be within the bounded one-shot policy")
    context = load_authority_context(
        host_preflight=config.host_preflight,
        outer_preflight=config.outer_preflight,
        raw_l1_route_authority=config.raw_l1_route_authority,
        host_binary=config.host_binary,
        pins=config.pins,
    )
    resource = _read_resource_admission(context=context, path=config.resource_admission)
    if not context.capture_body_wired:
        raise FullL1LifecycleError("current host preflight says capture_body_wired=false")
    if not context.real_host_metal_cli_available:
        raise FullL1LifecycleError("current outer preflight says real_host_metal_cli_available=false")
    return context, resource


def _run_one_shot(
    config: ExecuteConfig, *, fake_child_command: Sequence[str] | None = None
) -> dict[str, Any]:
    """Create one new lifecycle and reap one child; never retry or replay."""
    context, resource = _execution_context(config)
    launch_requested, replay_requested, outer_capture_requested = _validate_new_layout(config)
    launch_dir = _mkdir_new(launch_requested, "launch directory")
    replay_dir = _mkdir_new(replay_requested, "replay directory")
    outer_capture_dir = _mkdir_new(outer_capture_requested, "outer capture directory")
    inner_capture_dir = outer_capture_dir / INNER_DIRNAME
    if inner_capture_dir.exists():  # defensive: outer root was required new
        raise FullL1LifecycleError("inner capture directory must be fresh")
    lease_path = launch_dir / LEASE_FILENAME
    lease_id = _lease_id(context, resource, launch_dir)
    _write_new(lease_path, _lease_document(context, resource, lease_id=lease_id))
    lease = _binding_for_path(lease_path, "issued full-L1 lease")
    if lease.document.get("schema") != LEASE_SCHEMA or lease.document.get("status") != LEASE_STATUS:
        raise FullL1LifecycleError("issued full-L1 lease self-validation failed")
    launch_path = launch_dir / LAUNCH_FILENAME
    _write_new(
        launch_path,
        _outer_launch_document(
            context=context,
            resource=resource,
            lease=lease,
            lease_id=lease_id,
            outer_capture_dir=outer_capture_dir,
            inner_capture_dir=inner_capture_dir,
            workers=config.workers,
        ),
    )
    launch = _binding_for_path(launch_path, "issued full-L1 outer launch authority")
    _validate_host_metal_abi(
        context=context,
        lease=lease,
        lease_id=lease_id,
        launch=launch,
        outer_capture_dir=outer_capture_dir,
        inner_capture_dir=inner_capture_dir,
        workers=config.workers,
    )
    replay_path = replay_dir / REPLAY_FILENAME
    _write_new(
        replay_path,
        _replay_document(
            context=context,
            lease=lease,
            lease_id=lease_id,
            outer_capture_dir=outer_capture_dir,
        ),
    )
    replay = _binding_for_path(replay_path, "full-L1 replay reservation")
    command = _child_command(
        context=context,
        lease_path=lease_path,
        launch_path=launch_path,
        outer_capture_dir=outer_capture_dir,
        inner_capture_dir=inner_capture_dir,
        workers=config.workers,
        fake_child_command=fake_child_command,
    )
    _write_new(
        outer_capture_dir / RUNNING_FILENAME,
        _seal(
            {
                "schema": OUTER_TERMINAL_SCHEMA,
                "status": "RUNNING_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_OUTER_ONE_SHOT",
                "recorded_at": _utc_now(),
                "test_only_fake_child": fake_child_command is not None,
                "lease_id": lease_id,
                "outer_launch_authority": _evidence(launch),
                "claim_boundary": "One child maximum; no terminal result is available yet.",
            }
        ),
    )
    terminal, child_pid, stdout, stderr = _run_child(
        command=command, outer_capture_dir=outer_capture_dir, timeout_seconds=config.timeout_seconds
    )
    _write_new(
        outer_capture_dir / CHILD_FILENAME,
        _seal(
            {
                "schema": OUTER_TERMINAL_SCHEMA,
                "status": "REAPED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_OUTER_ONE_SHOT_CHILD",
                "recorded_at": _utc_now(),
                "pid": child_pid,
                "terminal": terminal,
                "lease_id": lease_id,
            }
        ),
    )
    inner: BoundDocument | None = None
    inner_success = False
    inner_error: str | None = None
    child_receipt = inner_capture_dir / INNER_RECEIPT_FILENAME
    if terminal.get("exit_code") == 0 and terminal.get("timed_out") is False:
        try:
            inner, inner_success = _validate_inner_receipt(
                path=child_receipt,
                context=context,
                lease=lease,
                lease_id=lease_id,
                launch=launch,
                inner_capture_dir=inner_capture_dir,
                test_only=fake_child_command is not None,
            )
        except FullL1LifecycleError as exc:
            inner_error = str(exc)
    else:
        inner_error = "child did not exit successfully"
    if stdout.get("within_max_stream_bytes") is not True or stderr.get("within_max_stream_bytes") is not True:
        inner_error = inner_error or "child stdout/stderr exceeded the bounded stream limit"
    terminal_path = outer_capture_dir / TERMINAL_FILENAME
    _write_new(
        terminal_path,
        _outer_terminal_document(
            context=context,
            resource=resource,
            lease=lease,
            lease_id=lease_id,
            launch=launch,
            replay=replay,
            outer_capture_dir=outer_capture_dir,
            command=command,
            child_pid=child_pid,
            terminal=terminal,
            stdout=stdout,
            stderr=stderr,
            inner=inner,
            inner_success=inner_success,
            inner_error=inner_error,
            test_only=fake_child_command is not None,
        ),
    )
    terminal_bound = _binding_for_path(terminal_path, "full-L1 outer terminal")
    release_path = launch_dir / RELEASE_FILENAME
    _write_new(release_path, _release_document(lease=lease, lease_id=lease_id, terminal=terminal_bound))
    release = _binding_for_path(release_path, "full-L1 quiet lease release")
    return {
        "lease": lease.document,
        "lease_path": lease.path,
        "outer_launch_authority": launch.document,
        "outer_launch_authority_path": launch.path,
        "replay_reservation": replay.document,
        "replay_reservation_path": replay.path,
        "outer_terminal": terminal_bound.document,
        "outer_terminal_path": terminal_bound.path,
        "release": release.document,
        "release_path": release.path,
    }


def execute_one_shot(config: ExecuteConfig) -> dict[str, Any]:
    """The production-only API behind the explicit CLI execute flag."""
    return _run_one_shot(config, fake_child_command=None)


def run_one_shot_for_test(
    config: ExecuteConfig, *, fake_child_command: Sequence[str]
) -> dict[str, Any]:
    """Exercise the lifecycle with a disposable fake child only.

    This helper is deliberately not exposed through :func:`main`; its outer
    terminal is tagged ``test_only_fake_child`` and cannot be accepted as a
    physical component capture.
    """
    return _run_one_shot(config, fake_child_command=fake_child_command)


def _pins_from_args(args: argparse.Namespace) -> AuthorityPins:
    return AuthorityPins(
        host_binary_sha256=args.expected_host_binary_sha256,
        host_preflight_seal_sha256=args.expected_host_preflight_seal_sha256,
        outer_preflight_seal_sha256=args.expected_outer_preflight_seal_sha256,
        raw_l1_route_authority_seal_sha256=args.expected_raw_l1_route_authority_seal_sha256,
    )


def _config_from_args(args: argparse.Namespace) -> ExecuteConfig:
    if args.launch_dir is None or args.replay_dir is None or args.outer_capture_dir is None:
        raise FullL1LifecycleError("--mode execute requires new --launch-dir, --replay-dir, and --outer-capture-dir")
    return ExecuteConfig(
        host_preflight=args.host_preflight,
        outer_preflight=args.outer_preflight,
        raw_l1_route_authority=args.raw_l1_route_authority,
        host_binary=args.host_binary,
        resource_admission=args.resource_admission,
        launch_dir=args.launch_dir,
        replay_dir=args.replay_dir,
        outer_capture_dir=args.outer_capture_dir,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        pins=_pins_from_args(args),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "execute"), default="preflight")
    parser.add_argument("--execute-one-shot", action="store_true")
    parser.add_argument("--host-preflight", type=Path, required=True)
    parser.add_argument("--outer-preflight", type=Path, required=True)
    parser.add_argument("--raw-l1-route-authority", type=Path, required=True)
    parser.add_argument("--host-binary", type=Path, required=True)
    parser.add_argument("--resource-admission", type=Path, required=True)
    parser.add_argument("--launch-dir", type=Path)
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--outer-capture-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--expected-host-binary-sha256", default=CURRENT_HOST_BINARY_SHA256)
    parser.add_argument("--expected-host-preflight-seal-sha256", default=CURRENT_HOST_PREFLIGHT_SEAL_SHA256)
    parser.add_argument("--expected-outer-preflight-seal-sha256", default=CURRENT_OUTER_PREFLIGHT_SEAL_SHA256)
    parser.add_argument(
        "--expected-raw-l1-route-authority-seal-sha256", default=RAW_L1_ROUTE_AUTHORITY_SEAL_SHA256
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        return int(exc.code)
    try:
        if args.mode == "preflight":
            if args.execute_one_shot:
                raise FullL1LifecycleError("--execute-one-shot requires --mode execute")
            document = build_lifecycle_preflight(
                host_preflight=args.host_preflight,
                outer_preflight=args.outer_preflight,
                raw_l1_route_authority=args.raw_l1_route_authority,
                host_binary=args.host_binary,
                resource_admission=args.resource_admission,
                pins=_pins_from_args(args),
            )
            _write_new(args.out, document)
        else:
            if not args.execute_one_shot:
                raise FullL1LifecycleError("--mode execute requires the explicit --execute-one-shot flag")
            result = execute_one_shot(_config_from_args(args))
            document = _seal(
                {
                    "schema": PREFLIGHT_SCHEMA,
                    "status": result["outer_terminal"]["status"],
                    "outer_terminal": _evidence(
                        _binding_for_path(result["outer_terminal_path"], "full-L1 outer terminal pointer")
                    ),
                    "release": _evidence(
                        _binding_for_path(result["release_path"], "full-L1 release pointer")
                    ),
                    "claim_boundary": "Pointer to one separately sealed terminal and exactly one release only.",
                }
            )
            _write_new(args.out, document)
    except (FullL1LifecycleError, OSError, ValueError) as exc:
        print(f"Qwen80 full-L1 capture lifecycle refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.out), "status": document["status"], "seal_sha256": document["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
