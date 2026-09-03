#!/usr/bin/env python3
"""Canonically bind a valid reaped L1 CPU authority without relabeling its outer refusal.

This is deliberately an evidence-only recovery path.  It never opens the
catalog, starts a child, acquires a lease, or creates a Metal context.  Its
only output is a create-new sealed receipt that proves a historical inner
authority survived a now-fixed outer-validator ABI defect.  Downstream Layer-1
completion preflight must consume the historical dynamic authority itself, not
this audit wrapper.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import (
    ascension_qwen80_source_token_l1_router_authority_scan_outer as outer,
)


RECOVERY_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l1_router_authority_"
    "recovery_canonicalization.v1"
)
RECOVERY_STATUS = "RECOVERED_HISTORICAL_INNER_VALID_OUTER_REMAINS_REFUSED"
HISTORICAL_REFUSED_STATUS = (
    "REFUSED_QWEN80_SOURCE_TOKEN_L1_ROUTER_AUTHORITY_SCAN_OUTER_"
    "ZERO_EXIT_WITHOUT_VALID_SEALED_DYNAMIC_AUTHORITY"
)
HISTORICAL_CAPTURE_ERROR = "dynamic L1 route authority.capture_dir drifted"
CHILD_RUNNING_STATUS = (
    "RUNNING_QWEN80_SOURCE_TOKEN_L1_ROUTER_AUTHORITY_SCAN_OUTER_ONE_SHOT_CPU_CHILD"
)


class RecoveryCanonicalizationError(RuntimeError):
    """A reaped historical evidence chain cannot be promoted to canonical input."""


@dataclass(frozen=True)
class RecoveryConfig:
    outer_preflight: Path
    outer_launch_authority: Path
    outer_terminal: Path
    child_record: Path
    inner_authority: Path
    producer_binary: Path
    out: Path


@dataclass(frozen=True)
class ReapedAuthorityChain:
    outer_preflight: outer.BoundDocument
    outer_launch_authority: outer.BoundDocument
    refused_outer_terminal: outer.BoundDocument
    child_record: outer.BoundDocument
    inner_authority: outer.BoundDocument
    producer_preflight: outer.BoundDocument
    source: Mapping[str, outer.BoundDocument]
    producer_binary: Mapping[str, Any]
    recovery_current_pointer: outer.BoundDocument
    capture_dir: Path
    workers: int


def _mapping(value: object, label: str) -> dict[str, Any]:
    try:
        return outer._mapping(value, label)
    except outer.RouterAuthorityScanOuterError as exc:
        raise RecoveryCanonicalizationError(str(exc)) from exc


def _require_bool(value: Mapping[str, Any], field: str, expected: bool, label: str) -> None:
    try:
        outer._require_bool(value, field, expected, label)
    except outer.RouterAuthorityScanOuterError as exc:
        raise RecoveryCanonicalizationError(str(exc)) from exc


def _require_int(value: Mapping[str, Any], field: str, expected: int, label: str) -> None:
    try:
        outer._require_int(value, field, expected, label)
    except outer.RouterAuthorityScanOuterError as exc:
        raise RecoveryCanonicalizationError(str(exc)) from exc


def _require_binding(value: object, expected: outer.BoundDocument, label: str) -> None:
    try:
        outer._require_binding(value, expected, label)
    except outer.RouterAuthorityScanOuterError as exc:
        raise RecoveryCanonicalizationError(str(exc)) from exc


def _require_identity(value: object, expected: outer.BoundDocument, label: str) -> None:
    try:
        outer._require_identity(value, expected, label)
    except outer.RouterAuthorityScanOuterError as exc:
        raise RecoveryCanonicalizationError(str(exc)) from exc


def _require_pointer_evidence(value: object, *, canonical_path: str, label: str) -> dict[str, Any]:
    try:
        return outer._require_pointer_evidence(
            value, canonical_path=canonical_path, label=label
        )
    except outer.RouterAuthorityScanOuterError as exc:
        raise RecoveryCanonicalizationError(str(exc)) from exc


def _validate_versioned(
    value: object,
    *,
    observation_names: tuple[str, ...],
    canonical_path: str,
    manifest: outer.BoundDocument,
    admission_receipt: outer.BoundDocument,
    expected_observations: Mapping[str, object] | None,
    label: str,
) -> dict[str, Any]:
    try:
        return outer._validate_versioned_current_admission(
            value,
            observation_names=observation_names,
            canonical_path=canonical_path,
            manifest=manifest,
            admission_receipt=admission_receipt,
            expected_observations=expected_observations,
            label=label,
        )
    except outer.RouterAuthorityScanOuterError as exc:
        raise RecoveryCanonicalizationError(str(exc)) from exc


def _read_bound(path: Path, label: str, schema: str, status: str | None) -> outer.BoundDocument:
    try:
        return outer._read_bound(path, label, schema, status)
    except outer.RouterAuthorityScanOuterError as exc:
        raise RecoveryCanonicalizationError(str(exc)) from exc


def _same_pointer(
    value: object, expected: Mapping[str, Any], *, canonical_path: str, label: str
) -> None:
    observed = _require_pointer_evidence(value, canonical_path=canonical_path, label=label)
    expected_pointer = _require_pointer_evidence(
        expected, canonical_path=canonical_path, label=f"{label}.expected"
    )
    for field in (
        "path",
        "present",
        "bytes",
        "sha256",
        "raw_sha256",
        "document_sha256",
        "document_seal_sha256",
    ):
        if observed.get(field) != expected_pointer.get(field):
            raise RecoveryCanonicalizationError(f"{label}.{field} drifted")


def _validate_historical_launch(
    launch: outer.BoundDocument,
    *,
    outer_preflight: outer.BoundDocument,
    source: Mapping[str, outer.BoundDocument],
    producer_preflight: outer.BoundDocument,
    producer_binary: Mapping[str, Any],
    capture_dir: Path,
    inner_authority: Path,
) -> int:
    """Validate the original launch with its historical pointer observation.

    It intentionally does not require the mutable pointer bytes to remain the
    same today.  The canonical path is reread independently by the caller;
    only immutable manifest/admission lineage is exact across a permitted
    pointer reseal.
    """
    root = launch.document
    launch_identity = root.get("launch_identity_sha256")
    if not outer._is_sha(launch_identity):
        raise RecoveryCanonicalizationError("historical outer launch identity is invalid")
    _require_identity(
        root.get("outer_preflight"), outer_preflight, "historical outer launch.outer_preflight"
    )
    bindings = _mapping(root.get("source_binding"), "historical outer launch.source_binding")
    for name in ("manifest", "admission_receipt", "joint_assessment", "completion_preflight"):
        _require_binding(bindings.get(name), source[name], f"historical outer launch.source_binding.{name}")
    canonical_pointer_path = str(source["admission_current"].path)
    historical_launch_pointer = _require_pointer_evidence(
        bindings.get("admission_current"),
        canonical_path=canonical_pointer_path,
        label="historical outer launch.source_binding.admission_current",
    )
    if bindings.get("manifest_seal_sha256") != source["manifest"].document_seal_sha256:
        raise RecoveryCanonicalizationError("historical outer launch manifest seal drifted")
    if bindings.get("admission_receipt_seal_sha256") != source["admission_receipt"].document_seal_sha256:
        raise RecoveryCanonicalizationError("historical outer launch admission receipt seal drifted")
    historical_outer_versioned = _mapping(
        outer_preflight.document.get("versioned_current_admission"),
        "historical outer preflight.versioned_current_admission",
    )
    expected_observations = {
        "preflight_observed": _mapping(
            historical_outer_versioned.get("preflight_observed"),
            "historical outer preflight observation",
        ),
        "launch_observed": historical_launch_pointer,
    }
    _validate_versioned(
        root.get("versioned_current_admission"),
        observation_names=("preflight", "launch"),
        canonical_path=canonical_pointer_path,
        manifest=source["manifest"],
        admission_receipt=source["admission_receipt"],
        expected_observations=expected_observations,
        label="historical outer launch.versioned_current_admission",
    )
    _require_binding(
        root.get("producer_preflight"),
        producer_preflight,
        "historical outer launch.producer_preflight",
    )
    observed_binary = _mapping(
        root.get("producer_binary"), "historical outer launch.producer_binary"
    )
    for field in ("path", "present", "bytes", "sha256"):
        if observed_binary.get(field) != producer_binary.get(field):
            raise RecoveryCanonicalizationError(
                f"historical outer launch producer binary {field} drifted"
            )
    if root.get("planned_capture_dir") != str(capture_dir / outer.INNER_DIRNAME):
        raise RecoveryCanonicalizationError("historical outer launch capture directory drifted")
    if root.get("planned_output_authority") != str(inner_authority):
        raise RecoveryCanonicalizationError("historical outer launch output authority drifted")
    workers = root.get("workers")
    try:
        outer._validate_workers(workers)
    except outer.RouterAuthorityScanOuterError as exc:
        raise RecoveryCanonicalizationError(str(exc)) from exc
    policy = _mapping(root.get("execution_policy"), "historical outer launch.execution_policy")
    _require_int(policy, "exact_catalog_admission_scans", 1, "historical outer launch.execution_policy")
    for field, expected in (
        ("cpu_oracle_only", True),
        ("metal_or_gpu_allowed", False),
        ("lease_allowed", False),
        ("watcher_or_server_allowed", False),
        ("automatic_retry_allowed", False),
        ("outer_reaped_required", True),
        ("terminal_receipt_written_last_required", True),
    ):
        _require_bool(policy, field, expected, "historical outer launch.execution_policy")
    replay = _mapping(root.get("replay_guard"), "historical outer launch.replay_guard")
    for field in ("capture_dir_unique", "one_child_maximum"):
        _require_bool(replay, field, True, "historical outer launch.replay_guard")
    return workers


def _validate_child_record(
    child: outer.BoundDocument,
    *,
    launch: outer.BoundDocument,
    terminal: outer.BoundDocument,
    expected_command: Sequence[str],
) -> None:
    root = child.document
    if root.get("launch_identity_sha256") != launch.document.get("launch_identity_sha256"):
        raise RecoveryCanonicalizationError("historical child launch identity drifted")
    pid = root.get("pid")
    parent_pid = root.get("parent_pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise RecoveryCanonicalizationError("historical child PID is invalid")
    if isinstance(parent_pid, bool) or not isinstance(parent_pid, int) or parent_pid <= 0:
        raise RecoveryCanonicalizationError("historical child parent PID is invalid")
    command = root.get("command")
    if command != list(expected_command):
        raise RecoveryCanonicalizationError("historical child command drifted")
    _require_bool(root, "cpu_only_catalog_scan_child", True, "historical child")
    terminal_child = _mapping(
        terminal.document.get("child"), "historical refused outer terminal.child"
    )
    if terminal_child.get("pid") != pid or terminal_child.get("command") != command:
        raise RecoveryCanonicalizationError("historical refused outer terminal child identity drifted")
    child_terminal = _mapping(
        terminal_child.get("terminal"), "historical refused outer terminal.child.terminal"
    )
    for field, expected in (("reaped", True), ("timed_out", False)):
        _require_bool(child_terminal, field, expected, "historical refused outer terminal.child.terminal")
    for field in ("returncode", "exit_code"):
        _require_int(child_terminal, field, 0, "historical refused outer terminal.child.terminal")
    if child_terminal.get("signal") is not None:
        raise RecoveryCanonicalizationError("historical child was signaled")


def _validate_stream(
    value: object, *, label: str, expected_path: Path
) -> dict[str, Any]:
    observed = _mapping(value, label)
    try:
        actual = outer._stream_evidence(expected_path, label)
    except outer.RouterAuthorityScanOuterError as exc:
        raise RecoveryCanonicalizationError(str(exc)) from exc
    for field in ("path", "present", "bytes", "sha256", "within_max_stream_bytes"):
        if observed.get(field) != actual.get(field):
            raise RecoveryCanonicalizationError(f"{label}.{field} drifted")
    if actual.get("present") is not True or actual.get("within_max_stream_bytes") is not True:
        raise RecoveryCanonicalizationError(f"{label} is not retained bounded stream evidence")
    return actual


def _validate_refused_terminal(
    terminal: outer.BoundDocument,
    *,
    outer_preflight: outer.BoundDocument,
    launch: outer.BoundDocument,
    producer_preflight: outer.BoundDocument,
    producer_binary: Mapping[str, Any],
    source: Mapping[str, outer.BoundDocument],
    capture_dir: Path,
    inner_authority: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = terminal.document
    if root.get("launch_identity_sha256") != launch.document.get("launch_identity_sha256"):
        raise RecoveryCanonicalizationError("historical refused outer terminal launch identity drifted")
    if root.get("capture_error") != HISTORICAL_CAPTURE_ERROR:
        raise RecoveryCanonicalizationError("historical refused outer terminal cause is not the ABI defect")
    one_shot = _mapping(root.get("one_shot"), "historical refused outer terminal.one_shot")
    for field in (
        "automatic_retry_disabled",
        "same_capture_dir_never_starts_a_second_child",
        "outer_reaped_child",
        "terminal_receipt_written_last",
    ):
        _require_bool(one_shot, field, True, "historical refused outer terminal.one_shot")
    boundary = _mapping(root.get("claim_boundary"), "historical refused outer terminal.claim_boundary")
    for field in (
        "cpu_router_authority_scan_only",
    ):
        _require_bool(boundary, field, True, "historical refused outer terminal.claim_boundary")
    for field in (
        "metal_device_or_dispatch_performed_by_outer",
        "lease_issued_or_consumed_by_outer",
        "server_watcher_hcli_or_token_execution_performed_by_outer",
        "tps_tg_or_tournament_claim_earned",
    ):
        _require_bool(boundary, field, False, "historical refused outer terminal.claim_boundary")
    source_binding = _mapping(root.get("source_binding"), "historical refused outer terminal.source_binding")
    _require_binding(
        source_binding.get("outer_preflight"), outer_preflight, "historical refused outer terminal.outer_preflight"
    )
    _require_binding(
        source_binding.get("producer_preflight"), producer_preflight, "historical refused outer terminal.producer_preflight"
    )
    _require_binding(
        source_binding.get("outer_launch_authority"), launch, "historical refused outer terminal.outer_launch_authority"
    )
    observed_binary = _mapping(
        source_binding.get("producer_binary"), "historical refused outer terminal.producer_binary"
    )
    for field in ("path", "present", "bytes", "sha256"):
        if observed_binary.get(field) != producer_binary.get(field):
            raise RecoveryCanonicalizationError(
                f"historical refused outer terminal producer binary {field} drifted"
            )
    current_source = _mapping(
        source_binding.get("current_source"), "historical refused outer terminal.current_source"
    )
    for name in ("manifest", "admission_receipt", "joint_assessment", "completion_preflight"):
        _require_binding(
            current_source.get(name), source[name], f"historical refused outer terminal.current_source.{name}"
        )
    launch_versioned = _mapping(
        launch.document.get("versioned_current_admission"),
        "historical outer launch.versioned_current_admission",
    )
    canonical_pointer_path = str(source["admission_current"].path)
    _same_pointer(
        current_source.get("admission_current"),
        _mapping(launch_versioned.get("launch_observed"), "historical outer launch observation"),
        canonical_path=canonical_pointer_path,
        label="historical refused outer terminal.current_source.admission_current",
    )
    _validate_versioned(
        root.get("versioned_current_admission"),
        observation_names=("preflight", "launch", "terminal"),
        canonical_path=canonical_pointer_path,
        manifest=source["manifest"],
        admission_receipt=source["admission_receipt"],
        expected_observations={
            "preflight_observed": _mapping(
                launch_versioned.get("preflight_observed"),
                "historical outer launch preflight observation",
            ),
            "launch_observed": _mapping(
                launch_versioned.get("launch_observed"),
                "historical outer launch observation",
            ),
        },
        label="historical refused outer terminal.versioned_current_admission",
    )
    capture = _mapping(root.get("outer_capture"), "historical refused outer terminal.outer_capture")
    if capture.get("directory") != str(capture_dir):
        raise RecoveryCanonicalizationError("historical refused outer terminal capture directory drifted")
    if capture.get("inner_capture_dir") != str(capture_dir / outer.INNER_DIRNAME):
        raise RecoveryCanonicalizationError("historical refused outer terminal inner directory drifted")
    dynamic = _mapping(capture.get("dynamic_authority"), "historical refused outer terminal.dynamic_authority")
    if dynamic != {"path": str(inner_authority), "present": False}:
        raise RecoveryCanonicalizationError("historical refused outer terminal does not preserve the old validator refusal")
    stdout = _validate_stream(
        capture.get("stdout"),
        label="historical refused outer terminal.stdout",
        expected_path=capture_dir / outer.OUTER_STDOUT_FILENAME,
    )
    stderr = _validate_stream(
        capture.get("stderr"),
        label="historical refused outer terminal.stderr",
        expected_path=capture_dir / outer.OUTER_STDERR_FILENAME,
    )
    return stdout, stderr


def validate_reaped_authority_chain(config: RecoveryConfig) -> ReapedAuthorityChain:
    """Read-only full-chain validation for the known reaped historical capture."""
    try:
        outer_preflight = outer._read_outer_preflight(config.outer_preflight)
        source = outer._source_from_outer(outer_preflight)
    except outer.RouterAuthorityScanOuterError as exc:
        raise RecoveryCanonicalizationError(str(exc)) from exc
    capture_dir = config.outer_terminal.resolve(strict=True).parent
    expected_inner = (capture_dir / outer.INNER_DIRNAME / outer.DYNAMIC_AUTHORITY_FILENAME).resolve(
        strict=True
    )
    if config.inner_authority.resolve(strict=True) != expected_inner:
        raise RecoveryCanonicalizationError("--inner-authority is not the bounded historical inner output")
    if config.outer_launch_authority.resolve(strict=True) != (
        capture_dir / outer.OUTER_LAUNCH_FILENAME
    ).resolve(strict=True):
        raise RecoveryCanonicalizationError("--outer-launch-authority is not in the bounded historical capture")
    if config.child_record.resolve(strict=True) != (capture_dir / outer.CHILD_FILENAME).resolve(
        strict=True
    ):
        raise RecoveryCanonicalizationError("--child-record is not in the bounded historical capture")
    producer_binding = _mapping(
        outer_preflight.document.get("producer_preflight"), "historical outer preflight.producer_preflight"
    )
    producer_preflight = _read_bound(
        Path(str(producer_binding.get("path", ""))),
        "historical producer preflight",
        outer.PRODUCER_PREFLIGHT_SCHEMA,
        outer.PRODUCER_PREFLIGHT_STATUS,
    )
    _require_binding(
        producer_binding,
        producer_preflight,
        "historical outer preflight.producer_preflight",
    )
    try:
        producer_binary = outer._file_evidence(
            config.producer_binary, "historical producer binary", executable=True
        )
        outer._validate_producer_preflight(
            producer_preflight, current=source, producer_binary=producer_binary
        )
    except outer.RouterAuthorityScanOuterError as exc:
        raise RecoveryCanonicalizationError(str(exc)) from exc
    launch = _read_bound(
        config.outer_launch_authority,
        "historical outer launch authority",
        outer.OUTER_LAUNCH_SCHEMA,
        outer.OUTER_LAUNCH_STATUS,
    )
    workers = _validate_historical_launch(
        launch,
        outer_preflight=outer_preflight,
        source=source,
        producer_preflight=producer_preflight,
        producer_binary=producer_binary,
        capture_dir=capture_dir,
        inner_authority=expected_inner,
    )
    inner = _read_bound(
        expected_inner,
        "historical dynamic L1 route authority",
        outer.DYNAMIC_AUTHORITY_SCHEMA,
        outer.DYNAMIC_AUTHORITY_STATUS,
    )
    try:
        validated_inner = outer._validate_dynamic_authority(
            expected_inner,
            source=source,
            producer_preflight=producer_preflight,
            producer_binary=producer_binary,
            launch_authority=launch,
            capture_dir=capture_dir,
            workers=workers,
            versioned_current_admission=_mapping(
                launch.document.get("versioned_current_admission"),
                "historical outer launch.versioned_current_admission",
            ),
        )
    except outer.RouterAuthorityScanOuterError as exc:
        raise RecoveryCanonicalizationError(str(exc)) from exc
    if validated_inner != inner:  # pragma: no cover - defensive identity guard
        raise RecoveryCanonicalizationError("historical dynamic authority identity drifted during validation")
    terminal = _read_bound(
        config.outer_terminal,
        "historical refused outer terminal",
        outer.OUTER_SCHEMA,
        HISTORICAL_REFUSED_STATUS,
    )
    stdout, stderr = _validate_refused_terminal(
        terminal,
        outer_preflight=outer_preflight,
        launch=launch,
        producer_preflight=producer_preflight,
        producer_binary=producer_binary,
        source=source,
        capture_dir=capture_dir,
        inner_authority=expected_inner,
    )
    child = _read_bound(
        config.child_record,
        "historical reaped child record",
        outer.OUTER_SCHEMA,
        CHILD_RUNNING_STATUS,
    )
    expected_command = outer._child_command(
        producer_binary=producer_binary,
        source=source,
        producer_preflight=producer_preflight,
        launch_authority_path=launch.path,
        capture_dir=capture_dir,
        workers=workers,
    )
    _validate_child_record(
        child,
        launch=launch,
        terminal=terminal,
        expected_command=expected_command,
    )
    # A recovery result may record a new pointer reseal, but it must prove the
    # canonical path still binds exactly the same immutable lineage.
    try:
        recovery_current_pointer = outer._read_versioned_admission_current(
            str(source["admission_current"].path),
            manifest=source["manifest"],
            immutable_admission_receipt=source["admission_receipt"],
            label="recovery canonicalization",
        )
    except outer.RouterAuthorityScanOuterError as exc:
        raise RecoveryCanonicalizationError(str(exc)) from exc
    # Keep the streams live in this return value by checking their read-only
    # evidence now, before any wrapper is sealed.
    if stdout["bytes"] < 0 or stderr["bytes"] < 0:  # pragma: no cover - invariant
        raise RecoveryCanonicalizationError("stream byte count is invalid")
    return ReapedAuthorityChain(
        outer_preflight=outer_preflight,
        outer_launch_authority=launch,
        refused_outer_terminal=terminal,
        child_record=child,
        inner_authority=inner,
        producer_preflight=producer_preflight,
        source=source,
        producer_binary=producer_binary,
        recovery_current_pointer=recovery_current_pointer,
        capture_dir=capture_dir,
        workers=workers,
    )


def _identity_with_schema(bound: outer.BoundDocument) -> dict[str, Any]:
    return {
        **outer._binding(bound),
        "seal_sha256": bound.document_seal_sha256,
        "schema": bound.document.get("schema"),
        "status": bound.document.get("status"),
    }


def build_recovery_wrapper(config: RecoveryConfig) -> dict[str, Any]:
    """Return a new, non-promoting receipt after read-only chain validation."""
    chain = validate_reaped_authority_chain(config)
    inner_versioned = _mapping(
        chain.inner_authority.document.get("versioned_current_admission"),
        "historical dynamic L1 route authority.versioned_current_admission",
    )
    terminal = chain.refused_outer_terminal.document
    terminal_child = _mapping(terminal.get("child"), "historical refused outer terminal.child")
    terminal_state = _mapping(
        terminal_child.get("terminal"), "historical refused outer terminal.child.terminal"
    )
    outer_capture = _mapping(terminal.get("outer_capture"), "historical refused outer terminal.outer_capture")
    current = chain.recovery_current_pointer
    document = {
        "schema": RECOVERY_SCHEMA,
        "status": RECOVERY_STATUS,
        "recovery_validated_at": outer._utc_now(),
        "historical_chain": {
            "outer_preflight": _identity_with_schema(chain.outer_preflight),
            "outer_launch_authority": _identity_with_schema(chain.outer_launch_authority),
            "refused_outer_terminal": {
                **_identity_with_schema(chain.refused_outer_terminal),
                "capture_error": terminal.get("capture_error"),
            },
            "child_record": _identity_with_schema(chain.child_record),
            "child_reap": {
                "pid": terminal_child.get("pid"),
                "reaped": terminal_state.get("reaped"),
                "exit_code": terminal_state.get("exit_code"),
                "timed_out": terminal_state.get("timed_out"),
                "stdout": outer_capture.get("stdout"),
                "stderr": outer_capture.get("stderr"),
            },
        },
        "historical_inner_authority": _identity_with_schema(chain.inner_authority),
        "downstream_authority": {
            "consume_historical_inner_directly": True,
            "recovery_wrapper_is_not_a_dynamic_route_authority_substitute": True,
            "authority_path": str(chain.inner_authority.path),
            "authority_schema": outer.DYNAMIC_AUTHORITY_SCHEMA,
            "authority_status": outer.DYNAMIC_AUTHORITY_STATUS,
            "authority_document_sha256": chain.inner_authority.document_sha256,
            "authority_seal_sha256": chain.inner_authority.document_seal_sha256,
        },
        "immutable_source_chain": {
            "manifest": outer._binding(chain.source["manifest"]),
            "admission_receipt": outer._binding(chain.source["admission_receipt"]),
            "joint_assessment": outer._binding(chain.source["joint_assessment"]),
            "completion_preflight": outer._binding(chain.source["completion_preflight"]),
            "recovery_validation_admission_current": outer._binding(current),
        },
        "versioned_current_admission": {
            "canonical_pointer_path": str(current.path),
            "historical_preflight_observed": inner_versioned.get("preflight_observed"),
            "historical_launch_observed": inner_versioned.get("launch_observed"),
            "historical_terminal_observed": inner_versioned.get("terminal_observed"),
            "recovery_validation_observed": outer._binding(current),
            "immutable_manifest": outer._binding(chain.source["manifest"]),
            "immutable_admission_receipt": outer._binding(chain.source["admission_receipt"]),
            "acceptance": outer._versioned_current_acceptance(),
        },
        "canonicalization": {
            "historical_inner_validated_against_reaped_identity_chain": True,
            "static_downstream_contract_valid": True,
            "historical_outer_remains_refused": True,
            "historical_outer_status_relabelled": False,
            "no_new_scan_or_child": True,
            "downstream_authority_is_historical_inner": True,
        },
        "claim_boundary": {
            "cpu_file_only_recovery_validation": True,
            "catalog_or_payload_scan_performed": False,
            "child_started": False,
            "metal_or_gpu_activity_performed": False,
            "lease_issued_or_consumed": False,
            "watcher_or_server_changed": False,
            "model_token_or_tps_claim_earned": False,
            "complete_layer_or_decoder_claim_earned": False,
        },
    }
    try:
        return outer._sealed(document)
    except outer.RouterAuthorityScanOuterError as exc:
        raise RecoveryCanonicalizationError(str(exc)) from exc


def write_recovery_wrapper(config: RecoveryConfig) -> dict[str, Any]:
    if not config.out.is_absolute() or config.out.exists():
        raise RecoveryCanonicalizationError("--out must be a new absolute file")
    document = build_recovery_wrapper(config)
    try:
        outer._write_new(config.out, document)
    except outer.RouterAuthorityScanOuterError as exc:
        raise RecoveryCanonicalizationError(str(exc)) from exc
    return document


def _parse_args(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-preflight", type=Path, required=True)
    parser.add_argument("--outer-launch-authority", type=Path, required=True)
    parser.add_argument("--outer-terminal", type=Path, required=True)
    parser.add_argument("--child-record", type=Path, required=True)
    parser.add_argument("--inner-authority", type=Path, required=True)
    parser.add_argument("--producer-binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if arguments is None else arguments)
    config = RecoveryConfig(
        outer_preflight=args.outer_preflight,
        outer_launch_authority=args.outer_launch_authority,
        outer_terminal=args.outer_terminal,
        child_record=args.child_record,
        inner_authority=args.inner_authority,
        producer_binary=args.producer_binary,
        out=args.out,
    )
    try:
        document = write_recovery_wrapper(config)
    except RecoveryCanonicalizationError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(
        f"recovery_wrapper={config.out} seal={document['seal_sha256']} "
        f"historical_inner={document['historical_inner_authority']['seal_sha256']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
