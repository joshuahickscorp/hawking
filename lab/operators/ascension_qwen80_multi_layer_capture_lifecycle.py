#!/usr/bin/env python3
"""Receipt-last, one-shot lifecycle for the Qwen80 69-dispatch L0..L2 multi-layer host.

This controller deliberately sits *outside* the host and the static multi-layer
outer preflight.  A CPU preflight can inspect a frozen host/outer/schedule/
oracle/assessment authority chain and a separately sealed resource admission,
but it cannot create a lease, reserve a directory, spawn a child, or touch Metal.

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
CURRENT_HOST_BINARY_SHA256 = "0622fa632bee0e9aa8f72aceeeb163a304cefc3557bb695fa7c1f80d59f2613b"
CURRENT_HOST_PREFLIGHT_SEAL_SHA256 = "bf40d5e0fce4259e7f33aa5c5031e6202f1f84d4861c56967b417e7957ae7674"
# Outer preflight seal is filled after the first real outer is sealed against the
# frozen host preflight; tests always override pins with synthetic seals.
CURRENT_OUTER_PREFLIGHT_SEAL_SHA256 = "0" * 64
EXECUTION_SCHEDULE_SEAL_SHA256 = "54084ddfeb117f964d48242b76afe7765eddca39330416c95cbd877cf400106c"
CHAIN_CPU_ORACLE_SEAL_SHA256 = "a217fc80410993fdde888f02e28eb9fae3f20ae88f676b9b4acd6ed60fb4bae2"
L1_FULL_LAYER_ASSESSMENT_SEAL_SHA256 = "47a4f33f0d904873edafc5943ed7a21605386c48428097b14fc167d6f46daea5"
JOINT_ASSESSMENT_SEAL_SHA256 = "d1b2893135287e282987e7d35609db3d44cd6c42846f79518f58f7ed5684829d"

HOST_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_host_preflight.v1"
)
OUTER_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_outer_preflight.v1"
)
SCHEDULE_SCHEMA = "hawking.ascension.qwen80_48_layer_execution_schedule_authority.v1"
SCHEDULE_STATUS = "PREPARED_QWEN80_48_LAYER_EXECUTION_SCHEDULE_AUTHORITY_NOT_EXECUTED"
CHAIN_ORACLE_SCHEMA = "hawking.ascension.qwen80_multi_layer_chain_cpu_oracle.v1"
CHAIN_ORACLE_STRUCTURE_STATUS = (
    "PREPARED_QWEN80_MULTI_LAYER_CHAIN_CPU_ORACLE_STRUCTURE_NOT_NUMERIC_WITHOUT_LAYER_RECEIPTS"
)
CHAIN_ORACLE_COMPOSED_STATUS = (
    "COMPOSED_QWEN80_MULTI_LAYER_CHAIN_CPU_ORACLE_FROM_PER_LAYER_RECEIPTS_NOT_DEVICE"
)
CHAIN_ORACLE_NUMERIC_STATUS = (
    "NUMERIC_QWEN80_MULTI_LAYER_CHAIN_CPU_ORACLE_FROM_PER_LAYER_RECEIPTS_NOT_DEVICE"
)
L1_ASSESSMENT_SCHEMA = "hawking.ascension.qwen80_l1_full_layer_completion_assessment.v1"
L1_ASSESSMENT_STATUS = (
    "EARNED_QWEN80_SOURCE_TOKEN_L1_COMPLETE_LAYER_COMPONENT_NOT_TOKEN_DECODER"
)
JOINT_ASSESSMENT_SCHEMA = "hawking.ascension.qwen80_l0_l1_joint_post_capture_assessment.v1"
JOINT_ASSESSMENT_STATUS = (
    "EARNED_QWEN80_SOURCE_TOKEN_L0_L1_COMPONENT_NOT_FULL_LAYER_TOKEN_DECODER"
)

RESOURCE_SCHEMA = (
    "hawking.ascension.qwen80_source_token_multi_layer_capture_resource_admission.v1"
)
RESOURCE_STATUS = (
    "PREFLIGHTED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_CAPTURE_RESOURCE_"
    "ADMISSION_NOT_LEASED_OR_EXECUTED"
)
RESOURCE_REFUSED_STATUS = "REFUSED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_CAPTURE_RESOURCE_ADMISSION"

PREFLIGHT_SCHEMA = "hawking.ascension.qwen80_source_token_multi_layer_capture_lifecycle_preflight.v1"
PREFLIGHT_STATUS = (
    "PREFLIGHTED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_CAPTURE_LIFECYCLE_"
    "NOT_LEASED_OR_EXECUTED"
)
PREFLIGHT_REFUSED_STATUS = (
    "REFUSED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_CAPTURE_LIFECYCLE_"
    "PRECONDITIONS_INCOMPLETE_NO_LEASE_OR_EXECUTION"
)

LEASE_SCHEMA = "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_quiet_metal_lease.v1"
LEASE_STATUS = (
    "GRANTED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_COMPONENT_QUIET_METAL_LEASE"
)
OUTER_LAUNCH_SCHEMA = (
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_outer_launch_authority.v1"
)
OUTER_LAUNCH_STATUS = (
    "AUTHORIZED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_"
    "OUTER_REAPED_ONE_SHOT_METAL_CHILD"
)
INNER_SCHEMA = "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_capture.v1"
INNER_SUCCESS_STATUS = (
    "CAPTURED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_COMPONENT_ONLY"
)
INNER_REFUSED_STATUS = (
    "REFUSED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_PHASE_ACCURATE_TERMINAL_FAILURE"
)
OUTER_TERMINAL_SCHEMA = (
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_outer_capture.v1"
)
OUTER_TERMINAL_STATUS = (
    "CAPTURED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_"
    "OUTER_TERMINAL_COMPONENT_ONLY"
)
OUTER_TERMINAL_TEST_STATUS = (
    "TERMINAL_QWEN80_SOURCE_TOKEN_MULTI_LAYER_OUTER_FAKE_CHILD_TEST_ONLY"
)
OUTER_TERMINAL_REFUSED_PREFIX = (
    "REFUSED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_OUTER_"
)
RELEASE_SCHEMA = (
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_"
    "quiet_metal_lease_release.v1"
)
RELEASE_STATUS = (
    "RELEASED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_COMPONENT_QUIET_METAL_"
    "LEASE_AFTER_TERMINAL_CAPTURE"
)
REPLAY_SCHEMA = (
    "hawking.ascension.qwen80_source_token_multi_layer_capture_replay_reservation.v1"
)
REPLAY_STATUS = "RESERVED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_CAPTURE_ONE_SHOT"

LEASE_FILENAME = "multi-layer-quiet-metal-lease.json"
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

TOTAL_DISPATCHES = 69
LAYER_COUNT = 3
PER_LAYER_DISPATCHES = 23
MIN_FREE_PERCENT = 80
MAX_RESOURCE_AGE_SECONDS = 300
MAX_TIMEOUT_SECONDS = 7200.0
MAX_JSON_BYTES = 100_000_000
MAX_STREAM_BYTES = 1_000_000


class MultiLayerLifecycleError(RuntimeError):
    """A multi-layer capture lifecycle prerequisite failed closed."""


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
    execution_schedule_seal_sha256: str = EXECUTION_SCHEDULE_SEAL_SHA256
    chain_cpu_oracle_seal_sha256: str = CHAIN_CPU_ORACLE_SEAL_SHA256
    l1_full_layer_assessment_seal_sha256: str = L1_FULL_LAYER_ASSESSMENT_SEAL_SHA256
    joint_assessment_seal_sha256: str = JOINT_ASSESSMENT_SEAL_SHA256


@dataclass(frozen=True)
class AuthorityContext:
    host_preflight: BoundDocument
    outer_preflight: BoundDocument
    execution_schedule_authority: BoundDocument
    chain_cpu_oracle: BoundDocument
    l1_full_layer_assessment: BoundDocument
    joint_assessment: BoundDocument
    host_binary: Path
    host_binary_bytes: int
    host_binary_sha256: str
    capture_body_wired: bool
    real_host_metal_cli_available: bool
    kernel_names: tuple[str, ...]
    layer_count: int


@dataclass(frozen=True)
class ExecuteConfig:
    host_preflight: Path
    outer_preflight: Path
    host_binary: Path
    resource_admission: Path
    launch_dir: Path
    replay_dir: Path
    outer_capture_dir: Path
    workers: int = 1
    timeout_seconds: float = 7200.0
    layer_count: int = LAYER_COUNT
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
        raise MultiLayerLifecycleError(f"{label} must be an object")
    return dict(value)


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise MultiLayerLifecycleError(f"{label} must be an array")
    return list(value)


def _require_bool(document: Mapping[str, Any], field: str, expected: bool, label: str) -> None:
    if document.get(field) is not expected:
        raise MultiLayerLifecycleError(f"{label}.{field} must be {expected}")


def _require_int(document: Mapping[str, Any], field: str, expected: int, label: str) -> None:
    if document.get(field) != expected:
        raise MultiLayerLifecycleError(f"{label}.{field} must be {expected}")


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise MultiLayerLifecycleError(f"{label} must be an absolute path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise MultiLayerLifecycleError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise MultiLayerLifecycleError(f"{label} must be a regular non-symlink file")
    if executable and not os.access(path, os.X_OK):
        raise MultiLayerLifecycleError(f"{label} must be executable")
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
        raise MultiLayerLifecycleError(f"{label} exceeds the bounded JSON size")
    try:
        parsed = json.loads(raw.decode("utf-8"))
        verified = verify(_mapping(parsed, label), label=label)
    except (UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise MultiLayerLifecycleError(f"{label} is not a valid sealed JSON document: {exc}") from exc
    document = _mapping(verified, label)
    if schema is not None and document.get("schema") != schema:
        raise MultiLayerLifecycleError(f"{label}.schema drifted")
    if statuses is not None and document.get("status") not in statuses:
        raise MultiLayerLifecycleError(f"{label}.status drifted")
    seal_sha256 = document.get("seal_sha256")
    if not _is_sha256(seal_sha256):
        raise MultiLayerLifecycleError(f"{label}.seal_sha256 must be a lowercase SHA-256")
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
            raise MultiLayerLifecycleError(f"{label}.{field} drifted")


def _seal(document: Mapping[str, Any]) -> dict[str, Any]:
    result = seal(dict(document))
    try:
        verified = verify(result, label="new multi-layer lifecycle document")
    except SealIntegrityError as exc:  # pragma: no cover - defensive
        raise MultiLayerLifecycleError(f"new lifecycle document did not self-verify: {exc}") from exc
    return _mapping(verified, "new multi-layer lifecycle document")


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists():
        raise MultiLayerLifecycleError("output must be a new absolute path")
    try:
        parent_metadata = path.parent.lstat()
    except OSError as exc:
        raise MultiLayerLifecycleError(f"cannot stat output parent: {exc}") from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise MultiLayerLifecycleError("output parent must be an existing real directory")
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
        raise MultiLayerLifecycleError(f"{label} must be a new absolute directory")
    try:
        parent_metadata = path.parent.lstat()
    except OSError as exc:
        raise MultiLayerLifecycleError(f"cannot stat {label} parent: {exc}") from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise MultiLayerLifecycleError(f"{label} parent must be an existing real directory")
    try:
        path.mkdir(mode=0o750)
    except OSError as exc:
        raise MultiLayerLifecycleError(f"cannot create {label}: {exc}") from exc
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise MultiLayerLifecycleError(f"new {label} is not a real directory")
    return path.resolve(strict=True)


def _parse_utc(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise MultiLayerLifecycleError(f"{label} must be an RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MultiLayerLifecycleError(f"{label} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise MultiLayerLifecycleError(f"{label} must include a timezone")
    return parsed


def _valid_cpu_preflight_status(value: object, prefix: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(prefix)
        and value.endswith("CPU_ONLY_NOT_LEASED_OR_EXECUTED")
    )


def _kernel_names_from_trace(graph: Mapping[str, Any], label: str) -> tuple[str, ...]:
    names = _array(graph.get("kernel_names"), f"{label}.kernel_names")
    if len(names) != TOTAL_DISPATCHES or any(not isinstance(name, str) or not name for name in names):
        raise MultiLayerLifecycleError(
            f"{label} must retain exactly {TOTAL_DISPATCHES} non-empty kernel names"
        )
    return tuple(str(name) for name in names)


def _capture_body_wired_from_host(root: Mapping[str, Any]) -> bool:
    metal = _mapping(root.get("metal_path"), "host preflight.metal_path")
    if isinstance(metal.get("capture_body_wired"), bool):
        return bool(metal["capture_body_wired"])
    future = metal.get("future_metal_entrypoint")
    if isinstance(future, Mapping) and isinstance(future.get("capture_body_wired"), bool):
        return bool(future["capture_body_wired"])
    if metal.get("mode_metal_available") is True:
        return True
    return False


def _validate_host_preflight(
    host: BoundDocument, host_binary: Path, host_binary_bytes: int, host_binary_sha256: str
) -> tuple[bool, tuple[str, ...]]:
    root = host.document
    if root.get("schema") != HOST_PREFLIGHT_SCHEMA or not _valid_cpu_preflight_status(
        root.get("status"), "COMPILED_QWEN80_SOURCE_TOKEN_"
    ):
        raise MultiLayerLifecycleError("host preflight schema/status is not the current CPU-only host contract")
    binary = _mapping(root.get("host_binary"), "host preflight.host_binary")
    if (
        binary.get("path") != str(host_binary)
        or binary.get("bytes") != host_binary_bytes
        or binary.get("sha256") != host_binary_sha256
    ):
        raise MultiLayerLifecycleError("host preflight is not bound to the current host binary")
    _require_int(root, "layer_count", LAYER_COUNT, "host preflight")
    _require_int(root, "source_token_id", 1, "host preflight")
    policy = _mapping(root.get("execution_policy"), "host preflight.execution_policy")
    for field in (
        "one_runtime",
        "one_command_buffer",
        "single_fence_after_all_dispatches",
        "non_timed",
        "structural_kernel_trace_required",
        "receipt_written_last",
    ):
        _require_bool(policy, field, True, "host preflight.execution_policy")
    _require_int(policy, "fence_count", 1, "host preflight.execution_policy")
    _require_int(policy, "total_dispatches", TOTAL_DISPATCHES, "host preflight.execution_policy")
    _require_int(policy, "per_layer_dispatch_count", PER_LAYER_DISPATCHES, "host preflight.execution_policy")
    names = _kernel_names_from_trace(
        _mapping(root.get("structural_kernel_trace"), "host preflight.structural_kernel_trace"),
        "host preflight.structural_kernel_trace",
    )
    metal = _mapping(root.get("metal_path"), "host preflight.metal_path")
    _require_bool(metal, "metal_context_or_dispatch_performed", False, "host preflight.metal_path")
    _require_bool(
        metal,
        "physical_capture_requires_owner_lease_and_admission",
        True,
        "host preflight.metal_path",
    )
    schemas = _mapping(root.get("future_capture_schemas"), "host preflight.future_capture_schemas")
    if schemas.get("inner") != INNER_SCHEMA or schemas.get("inner_status") != INNER_SUCCESS_STATUS:
        raise MultiLayerLifecycleError("host preflight inner receipt ABI drifted")
    if schemas.get("outer") != OUTER_TERMINAL_SCHEMA or schemas.get("outer_status") != OUTER_TERMINAL_STATUS:
        raise MultiLayerLifecycleError("host preflight outer terminal ABI drifted")
    boundary = _mapping(root.get("claim_boundary"), "host preflight claim boundary")
    for field in (
        "multi_layer_device_parity",
        "token_generated",
        "decoder_started",
        "server_or_watcher_started",
        "tps_or_tg_measured",
        "tournament_started",
    ):
        _require_bool(boundary, field, False, "host preflight claim boundary")
    return _capture_body_wired_from_host(root), names


def _validate_outer_preflight(
    outer: BoundDocument,
    host: BoundDocument,
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
        raise MultiLayerLifecycleError("outer preflight schema/status is not the current CPU-only outer contract")
    _matches_evidence(root.get("host_preflight"), host, "outer preflight.host_preflight")
    outer_binary = _mapping(root.get("host_binary"), "outer preflight.host_binary")
    if (
        outer_binary.get("path") != str(host_binary)
        or outer_binary.get("bytes") != host_binary_bytes
        or outer_binary.get("sha256") != host_binary_sha256
    ):
        raise MultiLayerLifecycleError("outer preflight host binary drifted")
    scope = _mapping(root.get("exact_component_scope"), "outer preflight exact component scope")
    for field, expected in (
        ("source_token_id", 1),
        ("layer_count", LAYER_COUNT),
        ("total_dispatches", TOTAL_DISPATCHES),
        ("per_layer_dispatches", PER_LAYER_DISPATCHES),
    ):
        _require_int(scope, field, expected, "outer preflight exact component scope")
    _require_bool(scope, "one_fence_required", True, "outer preflight exact component scope")
    _require_bool(scope, "non_timed_exact_trace_required", True, "outer preflight exact component scope")
    outer_names = tuple(_array(scope.get("kernel_names"), "outer preflight kernel names"))
    if len(outer_names) != TOTAL_DISPATCHES or any(not isinstance(name, str) for name in outer_names):
        raise MultiLayerLifecycleError("outer preflight kernel names drifted")
    if outer_names != host_kernel_names:
        raise MultiLayerLifecycleError(
            f"host/outer exact {TOTAL_DISPATCHES}-kernel traces disagree"
        )
    gate = _mapping(root.get("future_metal_entrypoint"), "outer preflight Metal gate")
    for field in (
        "explicit_mode_required",
        "default_execution_disabled",
        "requires_new_multi_layer_lease",
        "requires_sealed_outer_launch_authority",
        "requires_fresh_outer_and_inner_capture_directories",
        "self_hashes_current_executable",
        "no_device_execution_in_this_cpu_preflight",
    ):
        _require_bool(gate, field, True, "outer preflight Metal gate")
    if gate.get("capture_body_wired") is not host_capture_body_wired:
        raise MultiLayerLifecycleError("host/outer capture-body wiring disagrees")
    lifecycle = _mapping(root.get("lifecycle"), "outer preflight lifecycle")
    for field in (
        "replay_guard_required",
        "one_child_process_required",
        "outer_reaped_terminal_required",
    ):
        _require_bool(lifecycle, field, True, "outer preflight lifecycle")
    _require_bool(lifecycle, "automatic_retry_authorized", False, "outer preflight lifecycle")
    if not isinstance(lifecycle.get("real_host_metal_cli_available"), bool):
        raise MultiLayerLifecycleError(
            "outer preflight lifecycle.real_host_metal_cli_available must be boolean"
        )
    # Bind schedule / oracle / assessment seals from outer to pins (checked in load).
    for field in (
        "execution_schedule_authority",
        "chain_cpu_oracle",
        "l1_full_layer_assessment",
        "joint_assessment",
    ):
        if field not in root:
            raise MultiLayerLifecycleError(f"outer preflight missing {field}")
    return bool(gate["capture_body_wired"]), bool(lifecycle["real_host_metal_cli_available"]), outer_names


def _pointer_seal(value: object, label: str) -> str:
    evidence = _mapping(value, label)
    seal = evidence.get("document_seal_sha256") or evidence.get("document_sha256")
    if not _is_sha256(seal):
        raise MultiLayerLifecycleError(f"{label} seal must be a lowercase SHA-256")
    return str(seal)


def load_authority_context(
    *,
    host_preflight: Path,
    outer_preflight: Path,
    host_binary: Path,
    pins: AuthorityPins = AuthorityPins(),
) -> AuthorityContext:
    """Load exactly one pinned multi-layer authority chain without spawning work."""
    for value, label in (
        (pins.host_binary_sha256, "host binary pin"),
        (pins.host_preflight_seal_sha256, "host preflight pin"),
        (pins.outer_preflight_seal_sha256, "outer preflight pin"),
        (pins.execution_schedule_seal_sha256, "execution schedule pin"),
        (pins.chain_cpu_oracle_seal_sha256, "chain cpu oracle pin"),
        (pins.l1_full_layer_assessment_seal_sha256, "L1 assessment pin"),
        (pins.joint_assessment_seal_sha256, "joint assessment pin"),
    ):
        if not _is_sha256(value):
            raise MultiLayerLifecycleError(f"{label} must be a lowercase SHA-256")
    clean_host = _canonical_regular(host_binary, "multi-layer host binary", executable=True)
    host_raw = clean_host.read_bytes()
    host_sha = _sha256_bytes(host_raw)
    if host_sha != pins.host_binary_sha256:
        raise MultiLayerLifecycleError("current host binary SHA does not match the explicit frozen pin")
    host = _read_sealed(host_preflight, "multi-layer host preflight", schema=HOST_PREFLIGHT_SCHEMA)
    outer = _read_sealed(outer_preflight, "multi-layer outer preflight", schema=OUTER_PREFLIGHT_SCHEMA)
    if host.seal_sha256 != pins.host_preflight_seal_sha256:
        raise MultiLayerLifecycleError("host preflight seal does not match the explicit frozen pin")
    if outer.seal_sha256 != pins.outer_preflight_seal_sha256:
        raise MultiLayerLifecycleError("outer preflight seal does not match the explicit frozen pin")
    # Outer binds schedule/oracle/assessments; pin against those seals.
    schedule_seal = _pointer_seal(
        outer.document.get("execution_schedule_authority"), "outer execution schedule"
    )
    oracle_seal = _pointer_seal(outer.document.get("chain_cpu_oracle"), "outer chain cpu oracle")
    assessment_seal = _pointer_seal(
        outer.document.get("l1_full_layer_assessment"), "outer L1 assessment"
    )
    joint_seal = _pointer_seal(outer.document.get("joint_assessment"), "outer joint assessment")
    if schedule_seal != pins.execution_schedule_seal_sha256:
        raise MultiLayerLifecycleError("execution schedule seal does not match the explicit frozen pin")
    if oracle_seal != pins.chain_cpu_oracle_seal_sha256:
        raise MultiLayerLifecycleError("chain cpu oracle seal does not match the explicit frozen pin")
    if assessment_seal != pins.l1_full_layer_assessment_seal_sha256:
        raise MultiLayerLifecycleError("L1 assessment seal does not match the explicit frozen pin")
    if joint_seal != pins.joint_assessment_seal_sha256:
        raise MultiLayerLifecycleError("joint assessment seal does not match the explicit frozen pin")
    # Load bound documents for evidence in receipts (paths from outer).
    schedule = _read_sealed(
        Path(_mapping(outer.document["execution_schedule_authority"], "schedule")["path"]),
        "execution schedule authority",
        schema=SCHEDULE_SCHEMA,
        statuses=(SCHEDULE_STATUS,),
    )
    oracle = _read_sealed(
        Path(_mapping(outer.document["chain_cpu_oracle"], "oracle")["path"]),
        "chain cpu oracle",
        schema=CHAIN_ORACLE_SCHEMA,
        statuses=(
            CHAIN_ORACLE_STRUCTURE_STATUS,
            CHAIN_ORACLE_COMPOSED_STATUS,
            CHAIN_ORACLE_NUMERIC_STATUS,
        ),
    )
    assessment = _read_sealed(
        Path(_mapping(outer.document["l1_full_layer_assessment"], "assessment")["path"]),
        "L1 full-layer assessment",
        schema=L1_ASSESSMENT_SCHEMA,
        statuses=(L1_ASSESSMENT_STATUS,),
    )
    joint = _read_sealed(
        Path(_mapping(outer.document["joint_assessment"], "joint assessment")["path"]),
        "joint post-capture assessment",
        schema=JOINT_ASSESSMENT_SCHEMA,
        statuses=(JOINT_ASSESSMENT_STATUS,),
    )
    if schedule.seal_sha256 != pins.execution_schedule_seal_sha256:
        raise MultiLayerLifecycleError("loaded schedule seal drifted from pin")
    if oracle.seal_sha256 != pins.chain_cpu_oracle_seal_sha256:
        raise MultiLayerLifecycleError("loaded chain oracle seal drifted from pin")
    if assessment.seal_sha256 != pins.l1_full_layer_assessment_seal_sha256:
        raise MultiLayerLifecycleError("loaded L1 assessment seal drifted from pin")
    if joint.seal_sha256 != pins.joint_assessment_seal_sha256:
        raise MultiLayerLifecycleError("loaded joint assessment seal drifted from pin")
    capture_body_wired, host_names = _validate_host_preflight(
        host, clean_host, len(host_raw), host_sha
    )
    outer_wired, real_cli, outer_names = _validate_outer_preflight(
        outer,
        host,
        clean_host,
        len(host_raw),
        host_sha,
        host_names,
        capture_body_wired,
    )
    if outer_wired is not capture_body_wired:
        raise MultiLayerLifecycleError("host/outer capture body state drifted")
    return AuthorityContext(
        host_preflight=host,
        outer_preflight=outer,
        execution_schedule_authority=schedule,
        chain_cpu_oracle=oracle,
        l1_full_layer_assessment=assessment,
        joint_assessment=joint,
        host_binary=clean_host,
        host_binary_bytes=len(host_raw),
        host_binary_sha256=host_sha,
        capture_body_wired=capture_body_wired,
        real_host_metal_cli_available=real_cli,
        kernel_names=outer_names,
        layer_count=LAYER_COUNT,
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
        ("q80_multi_layer_capture_children", "Q80 multi-layer capture child"),
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
            "execution_schedule_authority": _evidence(context.execution_schedule_authority),
            "chain_cpu_oracle": _evidence(context.chain_cpu_oracle),
            "l1_full_layer_assessment": _evidence(context.l1_full_layer_assessment),
            "joint_assessment": _evidence(context.joint_assessment),
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
        "multi-layer resource admission",
        schema=RESOURCE_SCHEMA,
        statuses=(RESOURCE_STATUS,),
    )
    root = resource.document
    if root.get("prepared") is not True or root.get("blockers") != []:
        raise MultiLayerLifecycleError("resource admission is not green")
    if root.get("minimum_memory_free_percent") != MIN_FREE_PERCENT:
        raise MultiLayerLifecycleError("resource admission memory floor drifted")
    if root.get("maximum_resource_age_seconds") != MAX_RESOURCE_AGE_SECONDS:
        raise MultiLayerLifecycleError("resource admission freshness window drifted")
    observed = _parse_utc(root.get("recorded_at"), "resource admission.recorded_at")
    reference = now or datetime.now(timezone.utc)
    age = (reference - observed).total_seconds()
    if age < 0 or age > MAX_RESOURCE_AGE_SECONDS:
        raise MultiLayerLifecycleError("resource admission is stale")
    _matches_evidence(root.get("host_preflight"), context.host_preflight, "resource admission.host_preflight")
    _matches_evidence(root.get("outer_preflight"), context.outer_preflight, "resource admission.outer_preflight")
    _matches_evidence(
        root.get("execution_schedule_authority"),
        context.execution_schedule_authority,
        "resource admission execution schedule",
    )
    _matches_evidence(
        root.get("chain_cpu_oracle"),
        context.chain_cpu_oracle,
        "resource admission chain cpu oracle",
    )
    _matches_evidence(
        root.get("l1_full_layer_assessment"),
        context.l1_full_layer_assessment,
        "resource admission L1 assessment",
    )
    _matches_evidence(
        root.get("joint_assessment"),
        context.joint_assessment,
        "resource admission joint assessment",
    )
    binary = _mapping(root.get("host_binary"), "resource admission.host_binary")
    if (
        binary.get("path") != str(context.host_binary)
        or binary.get("bytes") != context.host_binary_bytes
        or binary.get("sha256") != context.host_binary_sha256
    ):
        raise MultiLayerLifecycleError("resource admission host binary binding drifted")
    blockers = evaluate_resource_snapshot(_mapping(root.get("resource_snapshot"), "resource admission snapshot"))
    if blockers:
        raise MultiLayerLifecycleError("resource admission safety snapshot refused: " + "; ".join(blockers))
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
            host_binary=host_binary,
            pins=pins,
        )
        resource = _read_resource_admission(context=context, path=resource_admission, now=now)
        if not context.capture_body_wired:
            blockers.append("current host preflight says capture_body_wired=false")
        if not context.real_host_metal_cli_available:
            blockers.append("current outer preflight says real_host_metal_cli_available=false")
    except MultiLayerLifecycleError as exc:
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
            "execution_schedule_authority": _evidence(context.execution_schedule_authority)
            if context
            else {"present": False},
            "chain_cpu_oracle": _evidence(context.chain_cpu_oracle) if context else {"present": False},
            "l1_full_layer_assessment": _evidence(context.l1_full_layer_assessment)
            if context
            else {"present": False},
            "joint_assessment": _evidence(context.joint_assessment) if context else {"present": False},
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
        raise MultiLayerLifecycleError("launch/replay/outer-capture directories must be distinct")
    resolved_parents: list[Path] = []
    for path, label in zip(paths, labels):
        if not path.is_absolute() or path.exists():
            raise MultiLayerLifecycleError(f"{label} must be a new absolute directory")
        try:
            parent = path.parent.resolve(strict=True)
        except OSError as exc:
            raise MultiLayerLifecycleError(f"cannot resolve {label} parent: {exc}") from exc
        if not parent.is_dir() or parent.is_symlink():
            raise MultiLayerLifecycleError(f"{label} parent must be a real existing directory")
        resolved_parents.append(parent)
    # Do not allow a future caller to hide one lifecycle root inside another.
    absolute = tuple(path.absolute() for path in paths)
    for index, path in enumerate(absolute):
        for other_index, other in enumerate(absolute):
            if index != other_index and other in path.parents:
                raise MultiLayerLifecycleError("launch/replay/outer-capture directories must not be nested")
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
            "execution_schedule_authority": _evidence(context.execution_schedule_authority),
            "chain_cpu_oracle": _evidence(context.chain_cpu_oracle),
            "l1_full_layer_assessment": _evidence(context.l1_full_layer_assessment),
            "joint_assessment": _evidence(context.joint_assessment),
            "fresh_resource_admission": _evidence(resource),
            "execution_policy": {
                "source_token_id": 1,
                "layer_count": LAYER_COUNT,
                "per_layer_dispatches": PER_LAYER_DISPATCHES,
                "total_dispatches": TOTAL_DISPATCHES,
                # These exact names are consumed by the host's file-only
                # Metal gate before it can create its inner capture directory.
                "metal_mode_only": True,
                "non_timed_exact_69_dispatches_required": True,
                "one_fence_required": True,
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
        "execution_schedule_seal_sha256": context.execution_schedule_authority.seal_sha256,
        "chain_cpu_oracle_seal_sha256": context.chain_cpu_oracle.seal_sha256,
        "l1_full_layer_assessment_seal_sha256": context.l1_full_layer_assessment.seal_sha256,
        "joint_assessment_seal_sha256": context.joint_assessment.seal_sha256,
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
            "execution_schedule_authority": _evidence(context.execution_schedule_authority),
            "chain_cpu_oracle": _evidence(context.chain_cpu_oracle),
            "l1_full_layer_assessment": _evidence(context.l1_full_layer_assessment),
            "joint_assessment": _evidence(context.joint_assessment),
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
            "layer_count": LAYER_COUNT,
            "execution_policy": {
                "layer_count": LAYER_COUNT,
                "total_dispatches": TOTAL_DISPATCHES,
                "metal_mode_only": True,
                "non_timed_exact_69_dispatches_required": True,
                "one_fence_required": True,
                "non_timed_exact_trace_required": True,
                "component_only": True,
                "automatic_retry_allowed": False,
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
                "multi_layer_component_only": True,
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
        raise MultiLayerLifecycleError("issued lease schema/status drifted")
    if lease_root.get("lease_id") != lease_id:
        raise MultiLayerLifecycleError("issued lease ID drifted")
    _matches_evidence(lease_root.get("outer_preflight"), context.outer_preflight, "issued lease.outer_preflight")
    binary = _mapping(lease_root.get("host_binary"), "issued lease.host_binary")
    if (
        binary.get("path") != str(context.host_binary)
        or binary.get("present") is not True
        or binary.get("bytes") != context.host_binary_bytes
        or binary.get("sha256") != context.host_binary_sha256
    ):
        raise MultiLayerLifecycleError("issued lease host binary drifted")
    lease_policy = _mapping(lease_root.get("execution_policy"), "issued lease.execution_policy")
    for field, expected in (
        ("metal_mode_only", True),
        ("non_timed_exact_69_dispatches_required", True),
        ("one_fence_required", True),
        ("component_only", True),
        ("automatic_retry_allowed", False),
    ):
        _require_bool(lease_policy, field, expected, "issued lease.execution_policy")
    launch_root = launch.document
    if launch_root.get("schema") != OUTER_LAUNCH_SCHEMA or launch_root.get("status") != OUTER_LAUNCH_STATUS:
        raise MultiLayerLifecycleError("issued outer launch schema/status drifted")
    _matches_evidence(
        launch_root.get("outer_preflight"), context.outer_preflight, "issued outer launch.outer_preflight"
    )
    _matches_evidence(launch_root.get("lease_receipt"), lease, "issued outer launch.lease_receipt")
    if launch_root.get("lease_id") != lease_id:
        raise MultiLayerLifecycleError("issued outer launch lease ID drifted")
    launch_binary = _mapping(launch_root.get("host_binary"), "issued outer launch.host_binary")
    if (
        launch_binary.get("path") != str(context.host_binary)
        or launch_binary.get("present") is not True
        or launch_binary.get("bytes") != context.host_binary_bytes
        or launch_binary.get("sha256") != context.host_binary_sha256
    ):
        raise MultiLayerLifecycleError("issued outer launch host binary drifted")
    if (
        launch_root.get("planned_outer_capture_dir") != str(outer_capture_dir)
        or launch_root.get("planned_inner_capture_dir") != str(inner_capture_dir)
        or launch_root.get("workers") != workers
    ):
        raise MultiLayerLifecycleError("issued outer launch capture paths/workers drifted")
    launch_policy = _mapping(launch_root.get("execution_policy"), "issued outer launch.execution_policy")
    for field, expected in (
        ("metal_mode_only", True),
        ("non_timed_exact_69_dispatches_required", True),
        ("one_fence_required", True),
        ("component_only", True),
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
            raise MultiLayerLifecycleError("fake child command must be non-empty")
        return [
            *fake_child_command,
            "--layer-count",
            str(context.layer_count),
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
        "--layer-count",
        str(context.layer_count),
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
            raise MultiLayerLifecycleError(f"{label}.{field} drifted")


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
    inner = _read_sealed(path, "multi-layer child receipt", schema=INNER_SCHEMA)
    root = inner.document
    status = root.get("status")
    if status not in (INNER_SUCCESS_STATUS, INNER_REFUSED_STATUS):
        raise MultiLayerLifecycleError("multi-layer child receipt has an unrecognized terminal status")
    fixture = root.get("fixture_or_synthetic")
    if test_only:
        _require_bool(root, "fixture_or_synthetic", True, "fake multi-layer child receipt")
    elif fixture is not False:
        raise MultiLayerLifecycleError("production multi-layer child receipt may not be synthetic")
    if root.get("lease_id") not in (None, lease_id) and root.get("lease_id") != lease_id:
        # Accept either root lease_id or nested binding.
        pass
    if root.get("lease_id") == lease_id or test_only:
        pass
    # Prefer full_l1-style binding if present; multi-layer may use lease_id root only.
    if "full_l1_lease_binding" in root or "multi_layer_lease_binding" in root:
        lease_binding = _mapping(
            root.get("multi_layer_lease_binding") or root.get("full_l1_lease_binding"),
            "child multi-layer lease binding",
        )
        if lease_binding.get("lease_id") != lease_id:
            raise MultiLayerLifecycleError("child lease ID drifted")
        if "receipt" in lease_binding:
            _inner_binding_matches(lease_binding.get("receipt"), lease, "child lease receipt binding")
    elif root.get("lease_id") != lease_id and not test_only:
        # Fake path may omit nested binding if root lease_id matches.
        if root.get("lease_id") != lease_id:
            raise MultiLayerLifecycleError("child lease ID drifted")
    durable = _mapping(root.get("durable_capture"), "child durable capture")
    if durable.get("capture_directory") != str(inner_capture_dir):
        raise MultiLayerLifecycleError("child durable capture directory drifted")
    _require_bool(
        durable, "receipt_written_last_is_completion_marker", True, "child durable capture"
    )
    if status == INNER_REFUSED_STATUS:
        _mapping(root.get("execution_phase"), "phase-accurate child terminal execution phase")
        error = root.get("terminal_error")
        if not isinstance(error, str) or not error:
            raise MultiLayerLifecycleError("phase-accurate child terminal lacks terminal_error")
        return inner, False
    execution = _mapping(root.get("fresh_same_runtime_execution"), "child same-runtime execution")
    for field in ("fresh_runtime", "same_runtime", "same_tcb", "single_fence_after_all_dispatches"):
        if field in execution:
            _require_bool(execution, field, True, "child same-runtime execution")
    for field, expected in (
        ("layer_count", LAYER_COUNT),
        ("total_dispatches", TOTAL_DISPATCHES),
        ("fence_count", 1),
    ):
        if field in execution:
            _require_int(execution, field, expected, "child same-runtime execution")
    trace = _mapping(root.get("structural_kernel_trace"), "child structural kernel trace")
    _require_bool(trace, "exact_order", True, "child structural kernel trace")
    names = tuple(_array(trace.get("kernel_names"), "child structural kernel names"))
    if names != context.kernel_names:
        raise MultiLayerLifecycleError(
            f"child structural trace is not the frozen exact {TOTAL_DISPATCHES}-kernel trace"
        )
    boundary = _mapping(root.get("claim_boundary"), "child claim boundary")
    _require_bool(boundary, "multi_layer_component_only", True, "child claim boundary")
    for field in (
        "token_generated",
        "decoder_started",
        "tps_or_tg_measured",
        "tournament_started",
    ):
        if field in boundary:
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
            "execution_schedule_authority": _evidence(context.execution_schedule_authority),
            "chain_cpu_oracle": _evidence(context.chain_cpu_oracle),
            "l1_full_layer_assessment": _evidence(context.l1_full_layer_assessment),
            "joint_assessment": _evidence(context.joint_assessment),
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
                "multi_layer_component_only": status == OUTER_TERMINAL_STATUS,
                "token_generated": False,
                "decoder_started": False,
                "server_or_watcher_started": False,
                "tps_or_tg_measured": False,
                "tournament_started": False,
                # Also inside claim_boundary, not only at the document root. The
                # strict child publishes both here, and the completion assessor
                # reads them here; the outer published them at the root only, so
                # a conforming capture still failed independent assessment. Same
                # fact, stated where the consumer looks.
                "test_only_fake_child": test_only,
                "fixture_or_synthetic": test_only,
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
        raise MultiLayerLifecycleError("workers must be in 1..4")
    if not 1.0 <= config.timeout_seconds <= MAX_TIMEOUT_SECONDS:
        raise MultiLayerLifecycleError("timeout must be within the bounded one-shot policy")
    context = load_authority_context(
        host_preflight=config.host_preflight,
        outer_preflight=config.outer_preflight,
        host_binary=config.host_binary,
        pins=config.pins,
    )
    resource = _read_resource_admission(context=context, path=config.resource_admission)
    if not context.capture_body_wired:
        raise MultiLayerLifecycleError("current host preflight says capture_body_wired=false")
    if not context.real_host_metal_cli_available:
        raise MultiLayerLifecycleError("current outer preflight says real_host_metal_cli_available=false")
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
        raise MultiLayerLifecycleError("inner capture directory must be fresh")
    lease_path = launch_dir / LEASE_FILENAME
    lease_id = _lease_id(context, resource, launch_dir)
    _write_new(lease_path, _lease_document(context, resource, lease_id=lease_id))
    lease = _binding_for_path(lease_path, "issued multi-layer lease")
    if lease.document.get("schema") != LEASE_SCHEMA or lease.document.get("status") != LEASE_STATUS:
        raise MultiLayerLifecycleError("issued multi-layer lease self-validation failed")
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
    launch = _binding_for_path(launch_path, "issued multi-layer outer launch authority")
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
    replay = _binding_for_path(replay_path, "multi-layer replay reservation")
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
                "status": "RUNNING_QWEN80_SOURCE_TOKEN_MULTI_LAYER_OUTER_ONE_SHOT",
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
                "status": "REAPED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_OUTER_ONE_SHOT_CHILD",
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
        except MultiLayerLifecycleError as exc:
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
    terminal_bound = _binding_for_path(terminal_path, "multi-layer outer terminal")
    release_path = launch_dir / RELEASE_FILENAME
    _write_new(release_path, _release_document(lease=lease, lease_id=lease_id, terminal=terminal_bound))
    release = _binding_for_path(release_path, "multi-layer quiet lease release")
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
        execution_schedule_seal_sha256=args.expected_execution_schedule_seal_sha256,
        chain_cpu_oracle_seal_sha256=args.expected_chain_cpu_oracle_seal_sha256,
        l1_full_layer_assessment_seal_sha256=args.expected_l1_full_layer_assessment_seal_sha256,
        joint_assessment_seal_sha256=args.expected_joint_assessment_seal_sha256,
    )


def _config_from_args(args: argparse.Namespace) -> ExecuteConfig:
    if args.launch_dir is None or args.replay_dir is None or args.outer_capture_dir is None:
        raise MultiLayerLifecycleError("--mode execute requires new --launch-dir, --replay-dir, and --outer-capture-dir")
    return ExecuteConfig(
        host_preflight=args.host_preflight,
        outer_preflight=args.outer_preflight,
        host_binary=args.host_binary,
        resource_admission=args.resource_admission,
        launch_dir=args.launch_dir,
        replay_dir=args.replay_dir,
        outer_capture_dir=args.outer_capture_dir,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        layer_count=args.layer_count,
        pins=_pins_from_args(args),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "execute"), default="preflight")
    parser.add_argument("--execute-one-shot", action="store_true")
    parser.add_argument("--host-preflight", type=Path, required=True)
    parser.add_argument("--outer-preflight", type=Path, required=True)
    parser.add_argument("--host-binary", type=Path, required=True)
    parser.add_argument("--resource-admission", type=Path, required=True)
    parser.add_argument("--launch-dir", type=Path)
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--outer-capture-dir", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--layer-count", type=int, default=LAYER_COUNT)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--expected-host-binary-sha256", default=CURRENT_HOST_BINARY_SHA256)
    parser.add_argument("--expected-host-preflight-seal-sha256", default=CURRENT_HOST_PREFLIGHT_SEAL_SHA256)
    parser.add_argument("--expected-outer-preflight-seal-sha256", default=CURRENT_OUTER_PREFLIGHT_SEAL_SHA256)
    parser.add_argument(
        "--expected-execution-schedule-seal-sha256", default=EXECUTION_SCHEDULE_SEAL_SHA256
    )
    parser.add_argument(
        "--expected-chain-cpu-oracle-seal-sha256", default=CHAIN_CPU_ORACLE_SEAL_SHA256
    )
    parser.add_argument(
        "--expected-l1-full-layer-assessment-seal-sha256",
        default=L1_FULL_LAYER_ASSESSMENT_SEAL_SHA256,
    )
    parser.add_argument(
        "--expected-joint-assessment-seal-sha256",
        default=JOINT_ASSESSMENT_SEAL_SHA256,
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
                raise MultiLayerLifecycleError("--execute-one-shot requires --mode execute")
            document = build_lifecycle_preflight(
                host_preflight=args.host_preflight,
                outer_preflight=args.outer_preflight,
                host_binary=args.host_binary,
                resource_admission=args.resource_admission,
                pins=_pins_from_args(args),
            )
            _write_new(args.out, document)
        else:
            if not args.execute_one_shot:
                raise MultiLayerLifecycleError("--mode execute requires the explicit --execute-one-shot flag")
            result = execute_one_shot(_config_from_args(args))
            document = _seal(
                {
                    "schema": PREFLIGHT_SCHEMA,
                    "status": result["outer_terminal"]["status"],
                    "outer_terminal": _evidence(
                        _binding_for_path(result["outer_terminal_path"], "multi-layer outer terminal pointer")
                    ),
                    "release": _evidence(
                        _binding_for_path(result["release_path"], "multi-layer release pointer")
                    ),
                    "claim_boundary": "Pointer to one separately sealed terminal and exactly one release only.",
                }
            )
            _write_new(args.out, document)
    except (MultiLayerLifecycleError, OSError, ValueError) as exc:
        print(f"Qwen80 multi-layer capture lifecycle refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.out), "status": document["status"], "seal_sha256": document["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
