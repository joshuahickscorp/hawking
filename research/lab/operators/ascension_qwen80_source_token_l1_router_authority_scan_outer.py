#!/usr/bin/env python3
"""Fail-closed outer/replay boundary for a future L1 CPU route-authority scan.

The Rust producer has two deliberately separate CPU modes.  Its ``preflight``
mode only describes a source-bound scan; its future ``cpu-oracle`` mode may
perform one catalog admission scan and write one sealed Layer-1 all-ten route
authority.  This outer validates the former without spawning anything.  A
separate, explicit ``--execute-one-shot`` path reserves a create-new replay
guard, starts exactly one CPU child, reaps it, validates the dynamic authority,
and writes the outer terminal receipt last.

Neither path has a Metal, lease, watcher, server, HCLI, token, benchmark, or
retry capability.  Tests may use a disposable fake CPU child, but this module
does not run a real Qwen80 scan by default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import stat
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import ascension_qwen80_source_token_l0_route_plan as source_identity
from lab.receipts import SealIntegrityError, seal, verify


PRODUCER_BINARY_NAME = "ascension_qwen80_source_token_l1_all_ten_route_authority_cpu"
PRODUCER_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l1_all_ten_route_authority_"
    "producer_preflight.v1"
)
PRODUCER_PREFLIGHT_STATUS = (
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L1_ALL_TEN_ROUTE_AUTHORITY_"
    "CPU_PRODUCER_NOT_EXECUTED"
)
DYNAMIC_AUTHORITY_SCHEMA = "hawking.ascension.qwen80_source_token_l1_all_ten_route_authority.v1"
DYNAMIC_AUTHORITY_STATUS = (
    "SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L1_ALL_TEN_ROUTE_AUTHORITY_"
    "READY_FOR_SAME_RUNTIME_MOE_SUFFIX"
)
JOINT_ASSESSMENT_SCHEMA = "hawking.ascension.qwen80_l0_l1_joint_post_capture_assessment.v1"
JOINT_ASSESSMENT_STATUS = "EARNED_QWEN80_SOURCE_TOKEN_L0_L1_COMPONENT_NOT_FULL_LAYER_TOKEN_DECODER"
COMPLETION_PREFLIGHT_SCHEMA = "hawking.ascension.qwen80_source_token_l1_moe_completion_preflight.v1"
COMPLETION_PREFLIGHT_STATUS = (
    "PREPARED_QWEN80_SOURCE_TOKEN_L1_MOE_COMPLETION_ROUTE_AUTHORITY_REQUIRED_"
    "NOT_LEASED_OR_EXECUTED"
)

OUTER_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l1_router_authority_scan_"
    "outer_preflight.v1"
)
OUTER_PREFLIGHT_STATUS = (
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L1_ROUTER_AUTHORITY_SCAN_"
    "OUTER_CPU_ONLY_NOT_EXECUTED"
)
OUTER_PREFLIGHT_REFUSED_STATUS = (
    "REFUSED_QWEN80_SOURCE_TOKEN_L1_ROUTER_AUTHORITY_SCAN_OUTER_CPU_ONLY_"
    "PRECONDITIONS_INCOMPLETE_NO_CHILD"
)
OUTER_LAUNCH_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l1_all_ten_route_authority_"
    "outer_launch_authority.v1"
)
OUTER_LAUNCH_STATUS = (
    "AUTHORIZED_QWEN80_SOURCE_TOKEN_L1_ALL_TEN_ROUTE_AUTHORITY_CPU_CHILD_"
    "ONE_SHOT"
)
OUTER_SCHEMA = "hawking.ascension.qwen80_source_token_l1_router_authority_scan_outer_capture.v1"
CAPTURED_STATUS = (
    "CAPTURED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L1_ROUTER_AUTHORITY_SCAN_"
    "OUTER_TERMINAL_CPU_ONLY"
)
REFUSED_PREFIX = "REFUSED_QWEN80_SOURCE_TOKEN_L1_ROUTER_AUTHORITY_SCAN_OUTER_"
REPLAY_GUARD_SCHEMA = (
    "hawking.ascension.qwen80_source_token_l1_router_authority_scan_replay_guard.v1"
)
REPLAY_GUARD_STATUS = "RESERVED_ONE_SHOT_CPU_PRODUCER_NOT_EXECUTED"

OUTER_PREFLIGHT_FILENAME = "outer-preflight.json"
OUTER_LAUNCH_FILENAME = "outer-launch-authority.json"
OUTER_TERMINAL_FILENAME = "outer-terminal-receipt.json"
RUNNING_FILENAME = "outer-running.json"
CHILD_FILENAME = "child.json"
INNER_DIRNAME = "inner"
DYNAMIC_AUTHORITY_FILENAME = "l1-source-token-route-authority.json"
OUTER_STDOUT_FILENAME = "outer-child.stdout.log"
OUTER_STDERR_FILENAME = "outer-child.stderr.log"
MAX_JSON_BYTES = 100_000_000
MAX_STREAM_BYTES = 1_000_000
TOP_K = 10
EXPERTS = 512
L1_LAYER = 1
L1_SLOT = 1
L0_DISPATCHES = 23
L1_PREFIX_DISPATCHES = 9
WEIGHT_SUM_TOLERANCE = 2.0e-6


class RouterAuthorityScanOuterError(RuntimeError):
    """An immutable input, replay boundary, or child authority is invalid."""


@dataclass(frozen=True)
class BoundDocument:
    path: Path
    evidence: dict[str, Any]
    document: dict[str, Any]
    document_sha256: str
    document_seal_sha256: str


@dataclass(frozen=True)
class PreflightConfig:
    producer_preflight: Path
    manifest: Path
    admission_current: Path
    joint_assessment: Path
    completion_preflight: Path
    producer_binary: Path
    workers: int = 1


@dataclass(frozen=True)
class CaptureConfig:
    outer_preflight: Path
    producer_binary: Path
    capture_dir: Path
    replay_guard_dir: Path
    workers: int = 1
    timeout_seconds: float = 7200.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _document_sha(value: Mapping[str, Any]) -> str:
    return _sha_bytes(_canonical_bytes(dict(value)))


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RouterAuthorityScanOuterError(f"{label} must be an object")
    return dict(value)


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RouterAuthorityScanOuterError(f"{label} must be an array")
    return list(value)


def _require_bool(value: Mapping[str, Any], field: str, expected: bool, label: str) -> None:
    if value.get(field) is not expected:
        raise RouterAuthorityScanOuterError(f"{label}.{field} must be {expected}")


def _require_int(value: Mapping[str, Any], field: str, expected: int, label: str) -> None:
    if value.get(field) != expected:
        raise RouterAuthorityScanOuterError(f"{label}.{field} must be {expected}")


def _require_sha(value: Mapping[str, Any], field: str, label: str) -> str:
    observed = value.get(field)
    if not _is_sha(observed):
        raise RouterAuthorityScanOuterError(f"{label}.{field} must be a lowercase SHA-256")
    return str(observed)


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    if not path.is_absolute():
        raise RouterAuthorityScanOuterError(f"{label} must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RouterAuthorityScanOuterError(f"cannot stat {label}: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise RouterAuthorityScanOuterError(f"{label} must be a regular non-symlink file")
    if executable and not os.access(path, os.X_OK):
        raise RouterAuthorityScanOuterError(f"{label} must be executable")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise RouterAuthorityScanOuterError(f"cannot canonicalize {label}: {exc}") from exc


def _file_evidence(path: Path, label: str, *, executable: bool = False) -> dict[str, Any]:
    clean = _canonical_regular(path, label, executable=executable)
    raw = clean.read_bytes()
    return {"path": str(clean), "present": True, "bytes": len(raw), "sha256": _sha_bytes(raw)}


def _read_bound(path: Path, label: str, schema: str, status: str | None) -> BoundDocument:
    evidence = _file_evidence(path, label)
    if evidence["bytes"] > MAX_JSON_BYTES:
        raise RouterAuthorityScanOuterError(f"{label} exceeds the bounded JSON size")
    try:
        document = json.loads(Path(str(evidence["path"])).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RouterAuthorityScanOuterError(f"{label} is not JSON: {exc}") from exc
    document = _mapping(document, label)
    try:
        verified = verify(document, label=label)
    except SealIntegrityError as exc:
        raise RouterAuthorityScanOuterError(f"{label} seal is invalid: {exc}") from exc
    if verified.get("schema") != schema:
        raise RouterAuthorityScanOuterError(f"{label}.schema must be {schema!r}")
    if status is not None and verified.get("status") != status:
        raise RouterAuthorityScanOuterError(f"{label}.status must be {status!r}")
    seal_sha256 = verified.get("seal_sha256")
    if not _is_sha(seal_sha256):
        raise RouterAuthorityScanOuterError(f"{label}.seal_sha256 is invalid")
    return BoundDocument(
        path=Path(str(evidence["path"])),
        evidence=evidence,
        document=verified,
        # Rust authority producers deliberately use the unsigned canonical
        # document identity here.  In this seal family that is exactly the
        # seal value; raw on-disk identity remains ``evidence.sha256``.
        document_sha256=str(seal_sha256),
        document_seal_sha256=str(seal_sha256),
    )


def _binding(bound: BoundDocument) -> dict[str, Any]:
    return {
        **bound.evidence,
        # Rust's full cross-language evidence carries this explicit alias;
        # retain it so direct JSON equality cannot erase raw-file history.
        "raw_sha256": bound.evidence["sha256"],
        "document_sha256": bound.document_sha256,
        "document_seal_sha256": bound.document_seal_sha256,
    }


def _identity(bound: BoundDocument) -> dict[str, Any]:
    return {
        "present": True,
        "document_sha256": bound.document_sha256,
        "document_seal_sha256": bound.document_seal_sha256,
    }


def _require_binding(value: object, expected: BoundDocument, label: str) -> None:
    observed = _mapping(value, label)
    for field in (
        "path",
        "present",
        "bytes",
        "sha256",
        "raw_sha256",
        "document_sha256",
        "document_seal_sha256",
    ):
        if observed.get(field) != _binding(expected).get(field):
            raise RouterAuthorityScanOuterError(f"{label}.{field} drifted from exact evidence")


def _require_identity(value: object, expected: BoundDocument, label: str) -> None:
    observed = _mapping(value, label)
    if observed.get("present") is not True:
        raise RouterAuthorityScanOuterError(f"{label}.present must be true")
    if observed.get("document_sha256") != expected.document_sha256:
        raise RouterAuthorityScanOuterError(f"{label}.document_sha256 drifted")
    if observed.get("document_seal_sha256", observed.get("seal_sha256")) != expected.document_seal_sha256:
        raise RouterAuthorityScanOuterError(f"{label}.document_seal_sha256 drifted")


def _require_rust_sealed_identity(
    value: object, expected: BoundDocument, label: str
) -> None:
    """Require Rust's deliberately minimal sealed-identity grammar exactly.

    The Layer-1 CPU producer does not use the Python ``present`` marker for
    its joint-assessment identity.  Requiring precisely the two seal fields
    keeps that cross-language ABI fail-closed without accepting an alias or a
    partial file binding in its place.
    """
    observed = _mapping(value, label)
    if set(observed) != {"document_sha256", "document_seal_sha256"}:
        raise RouterAuthorityScanOuterError(f"{label} must use Rust's exact sealed-identity shape")
    if observed.get("document_sha256") != expected.document_sha256:
        raise RouterAuthorityScanOuterError(f"{label}.document_sha256 drifted")
    if observed.get("document_seal_sha256") != expected.document_seal_sha256:
        raise RouterAuthorityScanOuterError(f"{label}.document_seal_sha256 drifted")


def _versioned_current_acceptance() -> dict[str, bool]:
    """The admission pointer may reseal; its immutable lineage may not drift."""
    return {
        "canonical_pointer_path_required": True,
        "pointer_reseal_allowed_only_when_immutable_authority_is_exact": True,
        "immutable_manifest_raw_sha_and_seal_must_remain_exact": True,
        "immutable_admission_receipt_raw_sha_and_seal_must_remain_exact": True,
        "manifest_or_receipt_substitution_accepted": False,
    }


def _require_bound_evidence(value: object, *, label: str) -> dict[str, Any]:
    """Validate one full document binding while allowing Rust's raw_sha256 alias."""
    observed = _mapping(value, label)
    path = observed.get("path")
    if not isinstance(path, str) or not Path(path).is_absolute() or observed.get("present") is not True:
        raise RouterAuthorityScanOuterError(f"{label} path/presence drifted")
    if (
        isinstance(observed.get("bytes"), bool)
        or not isinstance(observed.get("bytes"), int)
        or observed["bytes"] <= 0
    ):
        raise RouterAuthorityScanOuterError(f"{label}.bytes must be positive")
    for field in ("sha256", "document_sha256", "document_seal_sha256"):
        if not _is_sha(observed.get(field)):
            raise RouterAuthorityScanOuterError(f"{label}.{field} must be a lowercase SHA-256")
    if observed["document_sha256"] != observed["document_seal_sha256"]:
        raise RouterAuthorityScanOuterError(f"{label} document identity/seal drifted")
    if "raw_sha256" in observed and observed["raw_sha256"] != observed["sha256"]:
        raise RouterAuthorityScanOuterError(f"{label}.raw_sha256 drifted")
    return observed


def _require_pointer_evidence(
    value: object, *, canonical_path: str, label: str
) -> dict[str, Any]:
    """Validate a full historical observation without treating it as current."""
    observed = _require_bound_evidence(value, label=label)
    if observed.get("path") != canonical_path:
        raise RouterAuthorityScanOuterError(f"{label} canonical pointer path drifted")
    return observed


def _versioned_current_admission(
    source: Mapping[str, BoundDocument],
    *,
    preflight_observed: Mapping[str, Any] | None = None,
    launch_observed: Mapping[str, Any] | None = None,
    terminal_observed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record named historical observations of the only versioned input."""
    pointer = source["admission_current"]
    current_observed = _binding(pointer)
    result: dict[str, Any] = {
        "canonical_pointer_path": str(pointer.path),
        "preflight_observed": dict(preflight_observed or current_observed),
        "immutable_manifest": _binding(source["manifest"]),
        "immutable_admission_receipt": _binding(source["admission_receipt"]),
        "acceptance": _versioned_current_acceptance(),
    }
    if launch_observed is not None:
        result["launch_observed"] = dict(launch_observed)
    if terminal_observed is not None:
        result["terminal_observed"] = dict(terminal_observed)
    return result


def _validate_versioned_current_admission(
    value: object,
    *,
    observation_names: tuple[str, ...],
    canonical_path: str,
    manifest: BoundDocument | None,
    admission_receipt: BoundDocument | None,
    expected_observations: Mapping[str, object] | None,
    label: str,
) -> dict[str, Any]:
    """Check named observations; raw pointer reseals remain allowed by phase."""
    admission = _mapping(value, label)
    expected_fields = {
        "canonical_pointer_path",
        "immutable_manifest",
        "immutable_admission_receipt",
        "acceptance",
        *{f"{name}_observed" for name in observation_names},
    }
    if set(admission) != expected_fields:
        raise RouterAuthorityScanOuterError(f"{label} has an unrecognized field")
    if admission.get("canonical_pointer_path") != canonical_path:
        raise RouterAuthorityScanOuterError(f"{label}.canonical_pointer_path drifted")
    immutable_manifest = _require_bound_evidence(
        admission.get("immutable_manifest"), label=f"{label}.immutable_manifest"
    )
    immutable_receipt = _require_bound_evidence(
        admission.get("immutable_admission_receipt"),
        label=f"{label}.immutable_admission_receipt",
    )
    if manifest is not None:
        _require_binding(immutable_manifest, manifest, f"{label}.immutable_manifest")
    if admission_receipt is not None:
        _require_binding(
            immutable_receipt,
            admission_receipt,
            f"{label}.immutable_admission_receipt",
        )
    if admission.get("acceptance") != _versioned_current_acceptance():
        raise RouterAuthorityScanOuterError(f"{label}.acceptance drifted")
    for name in observation_names:
        field = f"{name}_observed"
        observed = _require_pointer_evidence(
            admission.get(field),
            canonical_path=canonical_path,
            label=f"{label}.{field}",
        )
        expected_pointer = None if expected_observations is None else expected_observations.get(field)
        if expected_pointer is None:
            continue
        expected = _require_pointer_evidence(
            expected_pointer,
            canonical_path=canonical_path,
            label=f"{label}.expected_{field}",
        )
        for field in (
            "path",
            "present",
            "bytes",
            "sha256",
            "document_sha256",
            "document_seal_sha256",
        ):
            if observed.get(field) != expected.get(field):
                raise RouterAuthorityScanOuterError(
                    f"{label} does not echo its historical pointer {field}"
                )
    return admission


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    document = seal(dict(value))
    try:
        verify(document, label="new router-authority outer document")
    except SealIntegrityError as exc:  # pragma: no cover - defensive only
        raise RouterAuthorityScanOuterError(f"new document did not seal: {exc}") from exc
    return document


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise RouterAuthorityScanOuterError(f"{path} must be absolute")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise RouterAuthorityScanOuterError(f"{path} parent must be a real existing directory")
    payload = _canonical_bytes(dict(document))
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        raise RouterAuthorityScanOuterError(f"cannot create {path}: {exc}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    directory = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _new_dir(path: Path, label: str) -> Path:
    if not path.is_absolute() or path == REPO_ROOT or path == path.parent:
        raise RouterAuthorityScanOuterError(f"{label} must be a bounded new absolute directory")
    if path.exists():
        raise RouterAuthorityScanOuterError(f"{label} must be new")
    parent = path.parent
    if not parent.is_dir() or parent.is_symlink():
        raise RouterAuthorityScanOuterError(f"{label} parent must be a real existing directory")
    try:
        path.mkdir(mode=0o700)
    except OSError as exc:
        raise RouterAuthorityScanOuterError(f"cannot create {label}: {exc}") from exc
    return path.resolve(strict=True)


def _validate_workers(workers: int) -> None:
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4:
        raise RouterAuthorityScanOuterError("--workers must be 1..4")


def _validate_admission_current(
    admission_current: BoundDocument,
    *,
    manifest: BoundDocument,
    immutable_admission_receipt: BoundDocument | None = None,
    label: str,
) -> BoundDocument:
    """Require the mutable pointer to retain the exact immutable lineage."""
    selected_manifest = _mapping(
        admission_current.document.get("complete_manifest"),
        f"{label}.complete_manifest",
    )
    if (
        selected_manifest.get("document_sha256") != manifest.evidence["sha256"]
        or selected_manifest.get("seal_sha256") != manifest.document_seal_sha256
    ):
        raise RouterAuthorityScanOuterError(f"{label} does not bind the supplied manifest")
    selected_receipt = _mapping(
        admission_current.document.get("admission_receipt"),
        f"{label}.admission_receipt",
    )
    receipt_path = selected_receipt.get("path")
    if not isinstance(receipt_path, str) or not receipt_path:
        raise RouterAuthorityScanOuterError(f"{label}.admission_receipt.path must be non-empty")
    receipt = _read_bound(
        Path(receipt_path),
        f"{label} immutable admission receipt",
        source_identity.ADMISSION_RECEIPT_SCHEMA,
        source_identity.ADMISSION_RECEIPT_STATUS,
    )
    if selected_receipt.get("seal_sha256") != receipt.document_seal_sha256:
        raise RouterAuthorityScanOuterError(f"{label} admission receipt seal drifted")
    if immutable_admission_receipt is not None:
        _require_binding(
            _binding(receipt),
            immutable_admission_receipt,
            f"{label} immutable admission receipt",
        )
    return receipt


def _read_versioned_admission_current(
    canonical_path: str,
    *,
    manifest: BoundDocument,
    immutable_admission_receipt: BoundDocument,
    label: str,
) -> BoundDocument:
    if not isinstance(canonical_path, str) or not canonical_path:
        raise RouterAuthorityScanOuterError(f"{label}.canonical_pointer_path is invalid")
    pointer = _read_bound(
        Path(canonical_path),
        f"{label} current admission pointer",
        source_identity.ADMISSION_SCHEMA,
        source_identity.ADMISSION_STATUS,
    )
    if str(pointer.path) != canonical_path:
        raise RouterAuthorityScanOuterError(f"{label}.canonical_pointer_path is not canonical")
    _validate_admission_current(
        pointer,
        manifest=manifest,
        immutable_admission_receipt=immutable_admission_receipt,
        label=f"{label} current admission pointer",
    )
    return pointer


def _read_current_source(config: PreflightConfig) -> dict[str, BoundDocument]:
    """Read and bind the immutable source identity independently of producer output."""
    manifest = _read_bound(config.manifest, "manifest", source_identity.MANIFEST_SCHEMA, None)
    admission_current = _read_bound(
        config.admission_current,
        "admission current",
        source_identity.ADMISSION_SCHEMA,
        source_identity.ADMISSION_STATUS,
    )
    admission_receipt = _validate_admission_current(
        admission_current,
        manifest=manifest,
        label="admission current",
    )
    assessment = _read_bound(
        config.joint_assessment,
        "joint L0/L1 assessment",
        JOINT_ASSESSMENT_SCHEMA,
        JOINT_ASSESSMENT_STATUS,
    )
    completion = _read_bound(
        config.completion_preflight,
        "L1 MoE completion preflight",
        COMPLETION_PREFLIGHT_SCHEMA,
        COMPLETION_PREFLIGHT_STATUS,
    )
    _require_bool(
        completion.document,
        "preflight_ready_for_future_outer_authority_only",
        False,
        "L1 MoE completion preflight",
    )
    antecedent = _mapping(
        completion.document.get("antecedent_l0_l1_component"),
        "L1 MoE completion preflight.antecedent_l0_l1_component",
    )
    _require_identity(antecedent, assessment, "L1 MoE completion preflight antecedent")
    return {
        "manifest": manifest,
        "admission_current": admission_current,
        "admission_receipt": admission_receipt,
        "joint_assessment": assessment,
        "completion_preflight": completion,
    }


def _validate_producer_preflight(
    producer: BoundDocument,
    *,
    current: Mapping[str, BoundDocument],
    producer_binary: dict[str, Any],
) -> None:
    root = producer.document
    source = _mapping(root.get("source_binding"), "producer preflight.source_binding")
    for field in ("manifest", "admission_receipt", "joint_assessment", "completion_preflight"):
        _require_binding(source.get(field), current[field], f"producer preflight.source_binding.{field}")
    canonical_pointer_path = str(current["admission_current"].path)
    historical_pointer = _require_pointer_evidence(
        source.get("admission_current"),
        canonical_path=canonical_pointer_path,
        label="producer preflight.source_binding.admission_current",
    )
    if source.get("manifest_seal_sha256") != current["manifest"].document_seal_sha256:
        raise RouterAuthorityScanOuterError("producer preflight manifest seal drifted")
    if source.get("admission_receipt_seal_sha256") != current["admission_receipt"].document_seal_sha256:
        raise RouterAuthorityScanOuterError("producer preflight admission receipt seal drifted")
    if source.get("admission_current_pointer_seal_sha256") != historical_pointer["document_seal_sha256"]:
        raise RouterAuthorityScanOuterError("producer preflight admission current pointer seal drifted")
    _validate_versioned_current_admission(
        root.get("versioned_current_admission"),
        observation_names=("preflight",),
        canonical_path=canonical_pointer_path,
        manifest=current["manifest"],
        admission_receipt=current["admission_receipt"],
        expected_observations={"preflight_observed": historical_pointer},
        label="producer preflight.versioned_current_admission",
    )
    source_revision = source.get("source_revision")
    if not isinstance(source_revision, str) or not source_revision:
        raise RouterAuthorityScanOuterError("producer preflight source revision is invalid")
    if not _is_sha(source.get("source_audit_seal_sha256")):
        raise RouterAuthorityScanOuterError("producer preflight source audit seal is invalid")

    observed_binary = _mapping(root.get("producer_binary"), "producer preflight.producer_binary")
    for field in ("path", "present", "bytes", "sha256"):
        if observed_binary.get(field) != producer_binary.get(field):
            raise RouterAuthorityScanOuterError(f"producer preflight.producer_binary.{field} drifted")

    contract = _mapping(root.get("dynamic_authority_contract"), "producer preflight.dynamic_authority_contract")
    if contract.get("schema") != DYNAMIC_AUTHORITY_SCHEMA or contract.get("status") != DYNAMIC_AUTHORITY_STATUS:
        raise RouterAuthorityScanOuterError("producer preflight dynamic authority schema/status drifted")
    for field in (
        "all_ten_dynamic_router_ids_and_weights_required",
        "no_fixture_or_cross_process_buffer_substitution",
        "one_current_admitted_cpu_catalog_scan_required",
        "outer_launch_authority_binding_required",
        "planned_output_must_be_new_under_outer_capture_dir",
    ):
        _require_bool(contract, field, True, "producer preflight.dynamic_authority_contract")
    for field, expected in (
        ("source_token_id", 1),
        ("l1_layer", L1_LAYER),
        ("l1_linear_state_slot", L1_SLOT),
        ("l0_reencode_dispatches", L0_DISPATCHES),
        ("l1_prefix_dispatches", L1_PREFIX_DISPATCHES),
        ("l1_moe_suffix_dispatches", 14),
        ("exact_route_payloads_required", 30),
    ):
        _require_int(contract, field, expected, "producer preflight.dynamic_authority_contract")
    fixed = _array(contract.get("exact_fixed_payload_requirements"), "producer preflight.dynamic_authority_contract.exact_fixed_payload_requirements")
    if len(fixed) != 6:
        raise RouterAuthorityScanOuterError("producer preflight must require six fixed L1 payloads")
    boundary = _mapping(root.get("claim_boundary"), "producer preflight.claim_boundary")
    for field in (
        "strict_catalog_admission_scan_performed",
        "admitted_payload_snapshot_opened",
        "child_started",
        "metal_or_gpu_activity_performed",
        "lease_issued_or_consumed",
        "watcher_or_server_changed",
        "model_token_or_tps_claim_earned",
        "complete_layer_or_decoder_claim_earned",
    ):
        _require_bool(boundary, field, False, "producer preflight.claim_boundary")
    _require_bool(boundary, "preflight_only", True, "producer preflight.claim_boundary")


def build_outer_preflight(config: PreflightConfig) -> dict[str, Any]:
    """Validate files only and return a sealed no-subprocess outer preflight."""
    _validate_workers(config.workers)
    current = _read_current_source(config)
    producer_binary = _file_evidence(config.producer_binary, "producer binary", executable=True)
    if Path(str(producer_binary["path"])).name != PRODUCER_BINARY_NAME:
        raise RouterAuthorityScanOuterError(
            f"--producer-binary must be {PRODUCER_BINARY_NAME}"
        )
    producer = _read_bound(
        config.producer_preflight,
        "producer preflight",
        PRODUCER_PREFLIGHT_SCHEMA,
        PRODUCER_PREFLIGHT_STATUS,
    )
    _validate_producer_preflight(producer, current=current, producer_binary=producer_binary)
    producer_versioned_current = _mapping(
        producer.document.get("versioned_current_admission"),
        "producer preflight.versioned_current_admission",
    )
    return _sealed(
        {
            "schema": OUTER_PREFLIGHT_SCHEMA,
            "status": OUTER_PREFLIGHT_STATUS,
            "prepared": True,
            "child_spawned": False,
            "catalog_or_payload_scan_performed": False,
            "metal_or_gpu_activity_performed": False,
            "lease_issued_or_consumed": False,
            "source_binding": {name: _binding(bound) for name, bound in current.items()},
            # Preserve the producer's historical pointer observation verbatim;
            # the canonical pointer itself may have resealed before this outer
            # preflight observes it.
            "versioned_current_admission": _versioned_current_admission(
                current,
                preflight_observed=_mapping(
                    producer_versioned_current.get("preflight_observed"),
                    "producer preflight versioned-current preflight observation",
                ),
            ),
            "producer_preflight": _binding(producer),
            "producer_binary": producer_binary,
            "producer_contract": {
                "schema": PRODUCER_PREFLIGHT_SCHEMA,
                "status": PRODUCER_PREFLIGHT_STATUS,
                "dynamic_authority_schema": DYNAMIC_AUTHORITY_SCHEMA,
                "dynamic_authority_status": DYNAMIC_AUTHORITY_STATUS,
                "producer_mode": "cpu-oracle",
                "producer_binary_self_identity_required": True,
                "one_catalog_admission_scan_at_most": 1,
            },
            "lifecycle": {
                "explicit_execute_one_shot_required": True,
                "create_new_replay_guard_required": True,
                "create_new_capture_dir_required": True,
                "outer_reaps_exactly_one_child": True,
                "terminal_receipt_written_last": True,
                "automatic_retry_prohibited": True,
            },
            "claim_boundary": {
                "cpu_file_only_outer_preflight": True,
                "default_preflight_never_spawns": True,
                "catalog_scan_not_authorized_by_this_preflight_alone": True,
                "metal_gpu_lease_watcher_server_hcli_or_token_authorized": False,
                "tps_tg_or_tournament_claim_earned": False,
            },
        }
    )


def write_outer_preflight(config: PreflightConfig, out: Path) -> dict[str, Any]:
    if not out.is_absolute() or out.exists():
        raise RouterAuthorityScanOuterError("--out must be a new absolute file")
    document = build_outer_preflight(config)
    _write_new(out, document)
    return document


def _read_outer_preflight(path: Path) -> BoundDocument:
    outer = _read_bound(path, "outer preflight", OUTER_PREFLIGHT_SCHEMA, OUTER_PREFLIGHT_STATUS)
    root = outer.document
    for field, expected in (
        ("prepared", True),
        ("child_spawned", False),
        ("catalog_or_payload_scan_performed", False),
        ("metal_or_gpu_activity_performed", False),
        ("lease_issued_or_consumed", False),
    ):
        _require_bool(root, field, expected, "outer preflight")
    lifecycle = _mapping(root.get("lifecycle"), "outer preflight.lifecycle")
    for field in (
        "explicit_execute_one_shot_required",
        "create_new_replay_guard_required",
        "create_new_capture_dir_required",
        "outer_reaps_exactly_one_child",
        "terminal_receipt_written_last",
        "automatic_retry_prohibited",
    ):
        _require_bool(lifecycle, field, True, "outer preflight.lifecycle")
    contract = _mapping(root.get("producer_contract"), "outer preflight.producer_contract")
    if (
        contract.get("schema") != PRODUCER_PREFLIGHT_SCHEMA
        or contract.get("status") != PRODUCER_PREFLIGHT_STATUS
        or contract.get("dynamic_authority_schema") != DYNAMIC_AUTHORITY_SCHEMA
        or contract.get("dynamic_authority_status") != DYNAMIC_AUTHORITY_STATUS
        or contract.get("producer_mode") != "cpu-oracle"
    ):
        raise RouterAuthorityScanOuterError("outer preflight producer contract drifted")
    for field in ("producer_binary_self_identity_required",):
        _require_bool(contract, field, True, "outer preflight.producer_contract")
    _require_int(contract, "one_catalog_admission_scan_at_most", 1, "outer preflight.producer_contract")
    boundary = _mapping(root.get("claim_boundary"), "outer preflight.claim_boundary")
    for field in (
        "cpu_file_only_outer_preflight",
        "default_preflight_never_spawns",
        "catalog_scan_not_authorized_by_this_preflight_alone",
    ):
        _require_bool(boundary, field, True, "outer preflight.claim_boundary")
    for field in (
        "metal_gpu_lease_watcher_server_hcli_or_token_authorized",
        "tps_tg_or_tournament_claim_earned",
    ):
        _require_bool(boundary, field, False, "outer preflight.claim_boundary")
    bindings = _mapping(root.get("source_binding"), "outer preflight.source_binding")
    historical_pointer = _mapping(
        bindings.get("admission_current"),
        "outer preflight.source_binding.admission_current",
    )
    canonical_pointer_path = historical_pointer.get("path")
    if not isinstance(canonical_pointer_path, str) or not Path(canonical_pointer_path).is_absolute():
        raise RouterAuthorityScanOuterError(
            "outer preflight historical admission pointer path is invalid"
        )
    _validate_versioned_current_admission(
        root.get("versioned_current_admission"),
        observation_names=("preflight",),
        canonical_path=canonical_pointer_path,
        manifest=None,
        admission_receipt=None,
        expected_observations=None,
        label="outer preflight.versioned_current_admission",
    )
    return outer


def _source_from_outer(outer: BoundDocument) -> dict[str, BoundDocument]:
    """Re-read immutable sources and revalidate the canonical versioned pointer."""
    root = outer.document
    bindings = _mapping(root.get("source_binding"), "outer preflight.source_binding")
    expected: dict[str, tuple[str, str | None]] = {
        "manifest": (source_identity.MANIFEST_SCHEMA, None),
        "admission_receipt": (
            source_identity.ADMISSION_RECEIPT_SCHEMA,
            source_identity.ADMISSION_RECEIPT_STATUS,
        ),
        "joint_assessment": (JOINT_ASSESSMENT_SCHEMA, JOINT_ASSESSMENT_STATUS),
        "completion_preflight": (COMPLETION_PREFLIGHT_SCHEMA, COMPLETION_PREFLIGHT_STATUS),
    }
    result: dict[str, BoundDocument] = {}
    for name, (schema, status) in expected.items():
        binding = _mapping(bindings.get(name), f"outer preflight.source_binding.{name}")
        path = binding.get("path")
        if not isinstance(path, str) or not path:
            raise RouterAuthorityScanOuterError(f"outer preflight.source_binding.{name}.path is invalid")
        bound = _read_bound(Path(path), name, schema, status)
        _require_binding(binding, bound, f"outer preflight.source_binding.{name}")
        result[name] = bound
    historical_pointer = _mapping(
        bindings.get("admission_current"),
        "outer preflight.source_binding.admission_current",
    )
    canonical_pointer_path = historical_pointer.get("path")
    if not isinstance(canonical_pointer_path, str) or not Path(canonical_pointer_path).is_absolute():
        raise RouterAuthorityScanOuterError(
            "outer preflight historical admission pointer path is invalid"
        )
    _validate_versioned_current_admission(
        root.get("versioned_current_admission"),
        observation_names=("preflight",),
        canonical_path=canonical_pointer_path,
        manifest=result["manifest"],
        admission_receipt=result["admission_receipt"],
        expected_observations=None,
        label="outer preflight.versioned_current_admission",
    )
    result["admission_current"] = _read_versioned_admission_current(
        canonical_pointer_path,
        manifest=result["manifest"],
        immutable_admission_receipt=result["admission_receipt"],
        label="outer launch",
    )
    return result


def _terminal_versioned_current_admission(
    source: Mapping[str, BoundDocument], *, launch_versioned_current: Mapping[str, Any]
) -> dict[str, Any]:
    """Re-read the canonical pointer after reaping, before the terminal receipt."""
    terminal_source = dict(source)
    terminal_source["admission_current"] = _read_versioned_admission_current(
        str(source["admission_current"].path),
        manifest=source["manifest"],
        immutable_admission_receipt=source["admission_receipt"],
        label="outer terminal",
    )
    return _versioned_current_admission(
        terminal_source,
        preflight_observed=_mapping(
            launch_versioned_current.get("preflight_observed"),
            "launch versioned-current preflight observation",
        ),
        launch_observed=_mapping(
            launch_versioned_current.get("launch_observed"),
            "launch versioned-current launch observation",
        ),
        terminal_observed=_binding(terminal_source["admission_current"]),
    )


def _launch_identity(
    outer: BoundDocument,
    producer_binary: Mapping[str, Any],
    capture_dir: Path,
    workers: int,
) -> str:
    return _document_sha(
        {
            "schema": OUTER_SCHEMA,
            "outer_preflight": _identity(outer),
            "producer_binary": dict(producer_binary),
            "capture_dir": str(capture_dir),
            "dynamic_authority_out": str(capture_dir / INNER_DIRNAME / DYNAMIC_AUTHORITY_FILENAME),
            "workers": workers,
        }
    )


def _replay(capture_dir: Path, launch_identity: str) -> dict[str, Any]:
    terminal_path = capture_dir / OUTER_TERMINAL_FILENAME
    if not terminal_path.is_file():
        raise RouterAuthorityScanOuterError(
            "capture directory exists without a terminal receipt; a second child is prohibited"
        )
    terminal = _read_bound(terminal_path, "outer terminal", OUTER_SCHEMA, None)
    if terminal.document.get("launch_identity_sha256") != launch_identity:
        raise RouterAuthorityScanOuterError("capture directory belongs to another launch identity")
    return terminal.document


def _reserve_guard(
    replay_guard_dir: Path,
    *,
    launch_identity: str,
    outer: BoundDocument,
    capture_dir: Path,
) -> Path:
    if not replay_guard_dir.is_absolute():
        raise RouterAuthorityScanOuterError("--replay-guard-dir must be absolute")
    if replay_guard_dir.exists():
        if replay_guard_dir.is_symlink() or not replay_guard_dir.is_dir():
            raise RouterAuthorityScanOuterError("--replay-guard-dir must be a real directory")
    else:
        parent = replay_guard_dir.parent
        if not parent.is_dir() or parent.is_symlink():
            raise RouterAuthorityScanOuterError("--replay-guard-dir parent must be a real directory")
        replay_guard_dir.mkdir(mode=0o700)
    guard = replay_guard_dir / f"{launch_identity}.json"
    _write_new(
        guard,
        _sealed(
            {
                "schema": REPLAY_GUARD_SCHEMA,
                "status": REPLAY_GUARD_STATUS,
                "launch_identity_sha256": launch_identity,
                "outer_preflight": _identity(outer),
                "capture_dir": str(capture_dir),
                "replay_refused": True,
                "child_spawned": False,
                "claim_boundary": {
                    "reservation_only": True,
                    "catalog_or_payload_scan_performed": False,
                    "metal_or_gpu_activity_performed": False,
                    "lease_issued_or_consumed": False,
                },
            }
        ),
    )
    return guard


def _launch_authority(
    *,
    outer: BoundDocument,
    source: Mapping[str, BoundDocument],
    producer_preflight: BoundDocument,
    producer_binary: Mapping[str, Any],
    capture_dir: Path,
    launch_identity: str,
    workers: int,
    versioned_current_admission: Mapping[str, Any],
) -> dict[str, Any]:
    inner = capture_dir / INNER_DIRNAME
    source_binding = {name: _binding(bound) for name, bound in source.items()}
    source_binding["manifest_seal_sha256"] = source["manifest"].document_seal_sha256
    source_binding["admission_receipt_seal_sha256"] = source[
        "admission_receipt"
    ].document_seal_sha256
    return _sealed(
        {
            "schema": OUTER_LAUNCH_SCHEMA,
            "status": OUTER_LAUNCH_STATUS,
            "launch_identity_sha256": launch_identity,
            "outer_preflight": _identity(outer),
            "source_binding": source_binding,
            "versioned_current_admission": dict(versioned_current_admission),
            "producer_preflight": _binding(producer_preflight),
            "producer_binary": dict(producer_binary),
            "planned_capture_dir": str(inner),
            "planned_output_authority": str(inner / DYNAMIC_AUTHORITY_FILENAME),
            "workers": workers,
            "execution_policy": {
                "exact_catalog_admission_scans": 1,
                "cpu_oracle_only": True,
                "metal_or_gpu_allowed": False,
                "lease_allowed": False,
                "watcher_or_server_allowed": False,
                "automatic_retry_allowed": False,
                "outer_reaped_required": True,
                "terminal_receipt_written_last_required": True,
            },
            "replay_guard": {
                "capture_dir_unique": True,
                "one_child_maximum": True,
            },
        }
    )


def _child_command(
    *,
    producer_binary: Mapping[str, Any],
    source: Mapping[str, BoundDocument],
    producer_preflight: BoundDocument,
    launch_authority_path: Path,
    capture_dir: Path,
    workers: int,
) -> list[str]:
    return [
        str(producer_binary["path"]),
        "--mode",
        "cpu-oracle",
        "--manifest",
        str(source["manifest"].path),
        "--admission-current",
        str(source["admission_current"].path),
        "--joint-assessment",
        str(source["joint_assessment"].path),
        "--completion-preflight",
        str(source["completion_preflight"].path),
        "--producer-preflight",
        str(producer_preflight.path),
        "--producer-binary",
        str(producer_binary["path"]),
        "--outer-launch-authority",
        str(launch_authority_path),
        "--capture-dir",
        str(capture_dir / INNER_DIRNAME),
        "--out",
        str(capture_dir / INNER_DIRNAME / DYNAMIC_AUTHORITY_FILENAME),
        "--workers",
        str(workers),
    ]


def _read_launch_authority(
    path: Path,
    *,
    outer: BoundDocument,
    source: Mapping[str, BoundDocument],
    producer_preflight: BoundDocument,
    producer_binary: Mapping[str, Any],
    capture_dir: Path,
    workers: int,
    versioned_current_admission: Mapping[str, Any],
) -> BoundDocument:
    launch = _read_bound(path, "outer launch authority", OUTER_LAUNCH_SCHEMA, OUTER_LAUNCH_STATUS)
    root = launch.document
    _require_identity(root.get("outer_preflight"), outer, "outer launch authority.outer_preflight")
    bindings = _mapping(root.get("source_binding"), "outer launch authority.source_binding")
    for name, bound in source.items():
        if name == "admission_current":
            continue
        _require_binding(bindings.get(name), bound, f"outer launch authority.source_binding.{name}")
    historical_pointer = _require_pointer_evidence(
        bindings.get("admission_current"),
        canonical_path=str(source["admission_current"].path),
        label="outer launch authority.source_binding.admission_current",
    )
    # The source binding keeps historical pointer evidence, but only the
    # canonical path is stable across a permitted reseal.
    expected_observations = {
        "preflight_observed": _mapping(
            versioned_current_admission.get("preflight_observed"),
            "expected outer launch preflight observation",
        ),
        "launch_observed": _binding(source["admission_current"]),
    }
    _validate_versioned_current_admission(
        versioned_current_admission,
        observation_names=("preflight", "launch"),
        canonical_path=str(source["admission_current"].path),
        manifest=source["manifest"],
        admission_receipt=source["admission_receipt"],
        expected_observations=expected_observations,
        label="expected outer launch versioned_current_admission",
    )
    _validate_versioned_current_admission(
        root.get("versioned_current_admission"),
        observation_names=("preflight", "launch"),
        canonical_path=str(source["admission_current"].path),
        manifest=source["manifest"],
        admission_receipt=source["admission_receipt"],
        expected_observations=expected_observations,
        label="outer launch authority.versioned_current_admission",
    )
    if bindings.get("manifest_seal_sha256") != source["manifest"].document_seal_sha256:
        raise RouterAuthorityScanOuterError("outer launch authority manifest seal drifted")
    if bindings.get("admission_receipt_seal_sha256") != source["admission_receipt"].document_seal_sha256:
        raise RouterAuthorityScanOuterError("outer launch authority admission receipt seal drifted")
    _require_binding(root.get("producer_preflight"), producer_preflight, "outer launch authority.producer_preflight")
    binary = _mapping(root.get("producer_binary"), "outer launch authority.producer_binary")
    for field in ("path", "present", "bytes", "sha256"):
        if binary.get(field) != producer_binary.get(field):
            raise RouterAuthorityScanOuterError(f"outer launch authority producer binary {field} drifted")
    if root.get("planned_capture_dir") != str(capture_dir / INNER_DIRNAME):
        raise RouterAuthorityScanOuterError("outer launch authority planned capture directory drifted")
    if root.get("planned_output_authority") != str(capture_dir / INNER_DIRNAME / DYNAMIC_AUTHORITY_FILENAME):
        raise RouterAuthorityScanOuterError("outer launch authority planned output authority drifted")
    _require_int(root, "workers", workers, "outer launch authority")
    policy = _mapping(root.get("execution_policy"), "outer launch authority.execution_policy")
    _require_int(policy, "exact_catalog_admission_scans", 1, "outer launch authority.execution_policy")
    for field, expected in (
        ("cpu_oracle_only", True),
        ("metal_or_gpu_allowed", False),
        ("lease_allowed", False),
        ("watcher_or_server_allowed", False),
        ("automatic_retry_allowed", False),
        ("outer_reaped_required", True),
        ("terminal_receipt_written_last_required", True),
    ):
        _require_bool(policy, field, expected, "outer launch authority.execution_policy")
    replay = _mapping(root.get("replay_guard"), "outer launch authority.replay_guard")
    for field in ("capture_dir_unique", "one_child_maximum"):
        _require_bool(replay, field, True, "outer launch authority.replay_guard")
    return launch


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


def _stream_evidence(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False}
    evidence = _file_evidence(path, label)
    return {**evidence, "within_max_stream_bytes": evidence["bytes"] <= MAX_STREAM_BYTES}


def _require_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RouterAuthorityScanOuterError(f"{label} must be a finite number")
    return float(value)


def _descriptor(value: object, label: str, *, seen_artifacts: set[str], seen_payloads: set[str]) -> None:
    row = _mapping(value, label)
    for field in ("artifact_sha256", "direct_packed_payload_sha256", "header_sha256"):
        _require_sha(row, field, label)
    artifact = str(row["artifact_sha256"])
    payload = str(row["direct_packed_payload_sha256"])
    if artifact in seen_artifacts or payload in seen_payloads:
        raise RouterAuthorityScanOuterError(f"{label} reuses an artifact or packed payload SHA")
    seen_artifacts.add(artifact)
    seen_payloads.add(payload)
    if isinstance(row.get("payload_bytes"), bool) or not isinstance(row.get("payload_bytes"), int) or row["payload_bytes"] <= 0:
        raise RouterAuthorityScanOuterError(f"{label}.payload_bytes must be positive")
    layout = _mapping(row.get("layout"), f"{label}.layout")
    if (
        layout.get("magic") != "HQ30G1B1"
        or layout.get("version") != 1
        or layout.get("group_size") != 128
        or layout.get("scale_dtype") != "float16"
        or layout.get("sign_bit_order") != "little"
    ):
        raise RouterAuthorityScanOuterError(f"{label}.layout drifted")


def _validate_dynamic_authority(
    path: Path,
    *,
    source: Mapping[str, BoundDocument],
    producer_preflight: BoundDocument,
    producer_binary: Mapping[str, Any],
    launch_authority: BoundDocument,
    capture_dir: Path,
    workers: int,
    versioned_current_admission: Mapping[str, Any],
) -> BoundDocument:
    authority = _read_bound(path, "dynamic L1 route authority", DYNAMIC_AUTHORITY_SCHEMA, DYNAMIC_AUTHORITY_STATUS)
    root = authority.document
    _require_bool(root, "fixture_or_synthetic", False, "dynamic L1 route authority")
    _require_bool(root, "metal_or_gpu_activity_performed", False, "dynamic L1 route authority")
    _require_binding(root.get("producer_preflight"), producer_preflight, "dynamic L1 route authority.producer_preflight")
    observed_binary = _mapping(root.get("producer_binary"), "dynamic L1 route authority.producer_binary")
    for field in ("path", "present", "bytes", "sha256"):
        if observed_binary.get(field) != producer_binary.get(field):
            raise RouterAuthorityScanOuterError(f"dynamic L1 route authority.producer_binary.{field} drifted")
    _require_binding(
        root.get("outer_launch_authority_binding"),
        launch_authority,
        "dynamic L1 route authority.outer_launch_authority_binding",
    )
    if root.get("outer_launch_authority_binding", {}).get("path") != str(launch_authority.path):
        raise RouterAuthorityScanOuterError("dynamic L1 route authority outer launch authority path drifted")
    # The Rust producer deliberately nests its one-scan evidence under
    # ``cpu_outer_capture``.  That is the authoritative capture identity;
    # accepting a legacy top-level alias here would let a different output
    # directory be smuggled into an otherwise valid authority.
    cpu_outer_capture = _mapping(
        root.get("cpu_outer_capture"), "dynamic L1 route authority.cpu_outer_capture"
    )
    if cpu_outer_capture.get("capture_dir") != str(capture_dir / INNER_DIRNAME):
        raise RouterAuthorityScanOuterError(
            "dynamic L1 route authority.cpu_outer_capture.capture_dir drifted"
        )
    if cpu_outer_capture.get("output_authority_path") != str(path):
        raise RouterAuthorityScanOuterError(
            "dynamic L1 route authority.cpu_outer_capture.output_authority_path drifted"
        )
    _require_int(
        cpu_outer_capture, "workers", workers, "dynamic L1 route authority.cpu_outer_capture"
    )
    for field, expected in (
        ("one_current_admitted_catalog_scan_performed", True),
        ("raw_bf16_or_safetensors_reopened", False),
        ("outer_terminal_receipt_written_by_parent_last", True),
    ):
        _require_bool(
            cpu_outer_capture,
            field,
            expected,
            "dynamic L1 route authority.cpu_outer_capture",
        )
    if "capture_dir" in root or "workers" in root:
        raise RouterAuthorityScanOuterError(
            "dynamic L1 route authority must use only cpu_outer_capture path/worker evidence"
        )
    expected_observations = {
        "preflight_observed": _mapping(
            versioned_current_admission.get("preflight_observed"),
            "expected dynamic preflight observation",
        ),
        "launch_observed": _mapping(
            versioned_current_admission.get("launch_observed"),
            "expected dynamic launch observation",
        ),
    }
    _validate_versioned_current_admission(
        versioned_current_admission,
        observation_names=("preflight", "launch"),
        canonical_path=str(source["admission_current"].path),
        manifest=source["manifest"],
        admission_receipt=source["admission_receipt"],
        expected_observations=expected_observations,
        label="expected dynamic L1 route authority.versioned_current_admission",
    )
    _validate_versioned_current_admission(
        root.get("versioned_current_admission"),
        observation_names=("preflight", "launch", "terminal"),
        canonical_path=str(source["admission_current"].path),
        manifest=source["manifest"],
        admission_receipt=source["admission_receipt"],
        expected_observations=expected_observations,
        label="dynamic L1 route authority.versioned_current_admission",
    )

    source_binding = _mapping(root.get("source_binding"), "dynamic L1 route authority.source_binding")
    if source_binding.get("manifest_document_sha256") != source["manifest"].evidence["sha256"]:
        raise RouterAuthorityScanOuterError("dynamic L1 route authority manifest document identity drifted")
    if source_binding.get("manifest_seal_sha256") != source["manifest"].document_seal_sha256:
        raise RouterAuthorityScanOuterError("dynamic L1 route authority manifest seal drifted")
    if source_binding.get("admission_receipt_seal_sha256") != source["admission_receipt"].document_seal_sha256:
        raise RouterAuthorityScanOuterError("dynamic L1 route authority admission receipt seal drifted")
    _require_rust_sealed_identity(
        source_binding.get("joint_l0_l1_assessment"),
        source["joint_assessment"],
        "dynamic L1 route authority.source_binding.joint_l0_l1_assessment",
    )
    _require_bool(
        source_binding,
        "prior_joint_assessment_is_provenance_only",
        True,
        "dynamic L1 route authority.source_binding",
    )
    _require_bool(
        source_binding,
        "cross_process_pinned_buffer_import_allowed",
        False,
        "dynamic L1 route authority.source_binding",
    )

    cpu = _mapping(root.get("source_token_l1_cpu_oracle"), "dynamic L1 route authority.cpu_oracle")
    for field, expected in (
        ("source_token_id", 1),
        ("layer", L1_LAYER),
        ("linear_state_slot", L1_SLOT),
        ("fresh_l0_reencode_dispatches", L0_DISPATCHES),
        ("fresh_l1_prefix_dispatches", L1_PREFIX_DISPATCHES),
    ):
        _require_int(cpu, field, expected, "dynamic L1 route authority.cpu_oracle")
    for field in (
        "cpu_oracle_reencodes_l0_then_l1_prefix",
        "zero_initial_l0_state",
        "zero_initial_l1_slot1_state",
    ):
        _require_bool(cpu, field, True, "dynamic L1 route authority.cpu_oracle")
    for field in (
        "source_input_f32le_sha256",
        "l0_second_residual_cpu_f32le_sha256",
        "l1_prefix_input_cpu_f32le_sha256",
        "l1_first_residual_cpu_f32le_sha256",
        "l1_post_attention_normalized_hidden_cpu_f32le_sha256",
        "l1_router_logits_cpu_f32le_sha256",
        "l1_post_conv_state_cpu_f32le_sha256",
        "l1_post_recurrent_state_cpu_f32le_sha256",
    ):
        _require_sha(cpu, field, "dynamic L1 route authority.cpu_oracle")

    route = _mapping(root.get("source_token_router_evidence"), "dynamic L1 route authority.router")
    _require_int(route, "logit_count", EXPERTS, "dynamic L1 route authority.router")
    _require_int(route, "top_k", TOP_K, "dynamic L1 route authority.router")
    for field, expected in (
        ("selection", "source_qwen80_topk_router"),
        ("tie_break", "lowest_expert_id_within_route_tie_epsilon"),
        ("softmax", "subtract_max_exp_f32"),
        ("route_tie_epsilon_source", "HAWKING_DS_ROUTE_TIE_EPS"),
    ):
        if route.get(field) != expected:
            raise RouterAuthorityScanOuterError(f"dynamic L1 route authority.router.{field} drifted")
    _require_bool(route, "selected_probabilities_renormalized", True, "dynamic L1 route authority.router")
    epsilon = _require_finite(route.get("route_tie_epsilon"), "dynamic L1 route authority.router.route_tie_epsilon")
    if epsilon < 0.0:
        raise RouterAuthorityScanOuterError("dynamic L1 route authority router tie epsilon is negative")
    epsilon_bits = route.get("route_tie_epsilon_f32_bits_hex")
    if not isinstance(epsilon_bits, str) or not epsilon_bits.startswith("0x") or len(epsilon_bits) != 10:
        raise RouterAuthorityScanOuterError("dynamic L1 route authority router tie epsilon bits are invalid")
    try:
        exact_epsilon = struct.unpack("!f", int(epsilon_bits[2:], 16).to_bytes(4, "big"))[0]
    except ValueError as exc:
        raise RouterAuthorityScanOuterError("dynamic L1 route authority router tie epsilon bits are invalid") from exc
    if float(exact_epsilon) != epsilon:
        raise RouterAuthorityScanOuterError("dynamic L1 route authority router tie epsilon/bits drifted")
    ids = _array(route.get("source_stable_route_ids"), "dynamic L1 route authority.router IDs")
    weights = _array(route.get("source_stable_normalized_weights"), "dynamic L1 route authority.router weights")
    if len(ids) != TOP_K or len(weights) != TOP_K:
        raise RouterAuthorityScanOuterError("dynamic L1 route authority requires exactly ten IDs and weights")
    if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < EXPERTS for value in ids):
        raise RouterAuthorityScanOuterError("dynamic L1 route authority route IDs are invalid")
    if len(set(ids)) != TOP_K:
        raise RouterAuthorityScanOuterError("dynamic L1 route authority route IDs are not unique")
    numeric_weights = [_require_finite(value, "dynamic L1 route authority route weight") for value in weights]
    if any(value < 0.0 for value in numeric_weights) or abs(sum(numeric_weights) - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise RouterAuthorityScanOuterError("dynamic L1 route authority route weights are not normalized")
    declared_sum = _require_finite(route.get("weights_sum"), "dynamic L1 route authority.router.weights_sum")
    if abs(declared_sum - sum(numeric_weights)) > WEIGHT_SUM_TOLERANCE:
        raise RouterAuthorityScanOuterError("dynamic L1 route authority declared weight sum drifted")

    seen_artifacts: set[str] = set()
    seen_payloads: set[str] = set()
    fixed = _array(root.get("fixed_l1_payloads"), "dynamic L1 route authority.fixed_l1_payloads")
    if len(fixed) != 6:
        raise RouterAuthorityScanOuterError("dynamic L1 route authority must bind six fixed L1 payloads")
    for index, value in enumerate(fixed):
        _descriptor(value, f"dynamic L1 route authority fixed payload {index}", seen_artifacts=seen_artifacts, seen_payloads=seen_payloads)
    waves = _array(root.get("deterministic_waves"), "dynamic L1 route authority.deterministic_waves")
    if len(waves) != TOP_K:
        raise RouterAuthorityScanOuterError("dynamic L1 route authority must bind ten deterministic waves")
    for index, value in enumerate(waves):
        wave = _mapping(value, f"dynamic L1 route authority wave {index}")
        _require_int(wave, "wave_index", index, f"dynamic L1 route authority wave {index}")
        _require_int(wave, "layer", L1_LAYER, f"dynamic L1 route authority wave {index}")
        _require_int(wave, "expert_id", ids[index], f"dynamic L1 route authority wave {index}")
        weight = _require_finite(wave.get("normalized_weight"), f"dynamic L1 route authority wave {index}.normalized_weight")
        if weight != numeric_weights[index]:
            raise RouterAuthorityScanOuterError(f"dynamic L1 route authority wave {index} weight drifted")
        expected_bits = f"0x{struct.unpack('!Q', struct.pack('!d', weight))[0]:016x}"
        if wave.get("normalized_weight_bits_hex") != expected_bits:
            raise RouterAuthorityScanOuterError(f"dynamic L1 route authority wave {index} bits drifted")
        for role in ("gate", "up", "down"):
            _descriptor(wave.get(role), f"dynamic L1 route authority wave {index}.{role}", seen_artifacts=seen_artifacts, seen_payloads=seen_payloads)
    if len(seen_artifacts) != 36 or len(seen_payloads) != 36:
        raise RouterAuthorityScanOuterError("dynamic L1 route authority requires 36 unique artifact/payload identities")
    gate = _mapping(root.get("rawls_real_all_ten_provenance_gate"), "dynamic L1 route authority.provenance_gate")
    for field in (
        "all_ten_source_bindings_complete",
        "execution_receipt_required_for_each_wave",
        "direct_packed_execution_required_for_each_wave",
        "source_bound_input_required_for_each_wave",
        "route_combine_receipt_required_separately",
        "shared_expert_receipt_required_separately",
        "first_and_second_residual_receipts_required_separately",
        "rejects_tensor_substitution",
        "rejects_route_reorder",
        "rejects_duplicate_experts",
        "rejects_missing_tensor_or_weight",
    ):
        _require_bool(gate, field, True, "dynamic L1 route authority.provenance_gate")
    _require_int(gate, "expected_layer", L1_LAYER, "dynamic L1 route authority.provenance_gate")
    for field in (
        "route_execution_performed",
        "route_combine_performed",
        "shared_expert_performed",
        "residual_combine_performed",
        "metal_device_or_dispatch_performed",
        "model_execution_performed",
        "hcli_execution_performed",
        "tps_or_tg_measurement_performed",
        "complete_layer_or_decoder_claim_earned",
    ):
        _require_bool(root, field, False, "dynamic L1 route authority")
    return authority


def _terminal_status(
    terminal: Mapping[str, Any], *, dynamic_valid: bool, capture_error: str | None
) -> str:
    if terminal.get("spawn_error"):
        return f"{REFUSED_PREFIX}CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return f"{REFUSED_PREFIX}CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return f"{REFUSED_PREFIX}CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return f"{REFUSED_PREFIX}CHILD_NONZERO"
    if capture_error is not None or not dynamic_valid:
        return f"{REFUSED_PREFIX}ZERO_EXIT_WITHOUT_VALID_SEALED_DYNAMIC_AUTHORITY"
    return CAPTURED_STATUS


def run_attempt(config: CaptureConfig) -> dict[str, Any]:
    """Explicit future-only CPU child path; no retry and no device action."""
    _validate_workers(config.workers)
    if not 1.0 <= config.timeout_seconds <= 7200.0:
        raise RouterAuthorityScanOuterError("--timeout-seconds must be between 1 and 7200")
    outer = _read_outer_preflight(config.outer_preflight)
    source = _source_from_outer(outer)
    preflight_versioned_current = _mapping(
        outer.document.get("versioned_current_admission"),
        "outer preflight.versioned_current_admission",
    )
    launch_versioned_current = _versioned_current_admission(
        source,
        preflight_observed=_mapping(
            preflight_versioned_current.get("preflight_observed"),
            "outer preflight versioned-current preflight observation",
        ),
        launch_observed=_binding(source["admission_current"]),
    )
    producer_preflight_binding = _mapping(outer.document.get("producer_preflight"), "outer preflight.producer_preflight")
    producer_preflight = _read_bound(
        Path(str(producer_preflight_binding.get("path", ""))),
        "producer preflight",
        PRODUCER_PREFLIGHT_SCHEMA,
        PRODUCER_PREFLIGHT_STATUS,
    )
    _require_binding(producer_preflight_binding, producer_preflight, "outer preflight.producer_preflight")
    producer_binary = _file_evidence(config.producer_binary, "producer binary", executable=True)
    if Path(str(producer_binary["path"])).name != PRODUCER_BINARY_NAME:
        raise RouterAuthorityScanOuterError(
            f"--producer-binary must be {PRODUCER_BINARY_NAME}"
        )
    expected_binary = _mapping(outer.document.get("producer_binary"), "outer preflight.producer_binary")
    for field in ("path", "present", "bytes", "sha256"):
        if expected_binary.get(field) != producer_binary.get(field):
            raise RouterAuthorityScanOuterError(f"producer binary {field} drifted from outer preflight")
    _validate_producer_preflight(producer_preflight, current=source, producer_binary=producer_binary)
    if (
        _mapping(
            outer.document.get("versioned_current_admission"),
            "outer preflight.versioned_current_admission",
        ).get("preflight_observed")
        != _mapping(
            producer_preflight.document.get("versioned_current_admission"),
            "producer preflight.versioned_current_admission",
        ).get("preflight_observed")
    ):
        raise RouterAuthorityScanOuterError(
            "outer preflight did not preserve producer preflight pointer history"
        )

    if not config.capture_dir.is_absolute() or config.capture_dir == REPO_ROOT:
        raise RouterAuthorityScanOuterError("--capture-dir must be a bounded absolute directory")
    launch_identity = _launch_identity(outer, producer_binary, config.capture_dir, config.workers)
    if config.capture_dir.exists():
        return _replay(config.capture_dir, launch_identity)
    _reserve_guard(
        config.replay_guard_dir,
        launch_identity=launch_identity,
        outer=outer,
        capture_dir=config.capture_dir,
    )
    capture_dir = _new_dir(config.capture_dir, "--capture-dir")
    inner = capture_dir / INNER_DIRNAME
    inner.mkdir(mode=0o700)
    authority_path = capture_dir / OUTER_LAUNCH_FILENAME
    launch = _launch_authority(
        outer=outer,
        source=source,
        producer_preflight=producer_preflight,
        producer_binary=producer_binary,
        capture_dir=capture_dir,
        launch_identity=launch_identity,
        workers=config.workers,
        versioned_current_admission=launch_versioned_current,
    )
    _write_new(authority_path, launch)
    launch_bound = _read_launch_authority(
        authority_path,
        outer=outer,
        source=source,
        producer_preflight=producer_preflight,
        producer_binary=producer_binary,
        capture_dir=capture_dir,
        workers=config.workers,
        versioned_current_admission=launch_versioned_current,
    )
    command = _child_command(
        producer_binary=producer_binary,
        source=source,
        producer_preflight=producer_preflight,
        launch_authority_path=authority_path,
        capture_dir=capture_dir,
        workers=config.workers,
    )
    _write_new(
        capture_dir / RUNNING_FILENAME,
        _sealed(
            {
                "schema": OUTER_SCHEMA,
                "status": "STARTED_QWEN80_SOURCE_TOKEN_L1_ROUTER_AUTHORITY_SCAN_OUTER_ONE_SHOT_CPU_CHILD",
                "recorded_at": _utc_now(),
                "launch_identity_sha256": launch_identity,
                "command": command,
                "claim_boundary": {
                    "automatic_retry_disabled": True,
                    "metal_or_gpu_activity_allowed": False,
                    "lease_issued_or_consumed": False,
                },
            }
        ),
    )

    started_at = _utc_now()
    child_pid: int | None = None
    capture_error: str | None = None
    dynamic_valid = False
    returncode: int | None = None
    timed_out = False
    spawn_error: str | None = None
    stdout_path = capture_dir / OUTER_STDOUT_FILENAME
    stderr_path = capture_dir / OUTER_STDERR_FILENAME
    try:
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            try:
                child = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                    close_fds=True,
                )
            except OSError as exc:
                spawn_error = f"{type(exc).__name__}: {exc}"
            else:
                child_pid = child.pid
                try:
                    _write_new(
                        capture_dir / CHILD_FILENAME,
                        _sealed(
                            {
                                "schema": OUTER_SCHEMA,
                                "status": "RUNNING_QWEN80_SOURCE_TOKEN_L1_ROUTER_AUTHORITY_SCAN_OUTER_ONE_SHOT_CPU_CHILD",
                                "recorded_at": _utc_now(),
                                "launch_identity_sha256": launch_identity,
                                "pid": child_pid,
                                "parent_pid": os.getpid(),
                                "command": command,
                                "cpu_only_catalog_scan_child": True,
                            }
                        ),
                    )
                except RouterAuthorityScanOuterError as exc:
                    capture_error = str(exc)
                    returncode = _terminate_group(child)
                else:
                    try:
                        returncode = child.wait(timeout=config.timeout_seconds)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        returncode = _terminate_group(child)
    except OSError as exc:
        capture_error = f"outer stream setup failed: {type(exc).__name__}: {exc}"
        spawn_error = capture_error

    terminal = _terminal(returncode, timed_out=timed_out, spawn_error=spawn_error)
    dynamic_evidence: dict[str, Any] = {
        "path": str(inner / DYNAMIC_AUTHORITY_FILENAME),
        "present": False,
    }
    if terminal.get("exit_code") == 0 and not timed_out and spawn_error is None:
        try:
            dynamic = _validate_dynamic_authority(
                inner / DYNAMIC_AUTHORITY_FILENAME,
                source=source,
                producer_preflight=producer_preflight,
                producer_binary=producer_binary,
                launch_authority=launch_bound,
                capture_dir=capture_dir,
                workers=config.workers,
                versioned_current_admission=launch_versioned_current,
            )
            dynamic_valid = True
            dynamic_evidence = {
                **_binding(dynamic),
                "schema": dynamic.document.get("schema"),
                "status": dynamic.document.get("status"),
            }
        except RouterAuthorityScanOuterError as exc:
            capture_error = str(exc)
    stdout_evidence = _stream_evidence(stdout_path, "outer stdout")
    stderr_evidence = _stream_evidence(stderr_path, "outer stderr")
    if not stdout_evidence.get("within_max_stream_bytes", False) or not stderr_evidence.get("within_max_stream_bytes", False):
        capture_error = capture_error or "outer child stream exceeded the bounded evidence limit"
    terminal_pointer_valid = False
    try:
        terminal_versioned_current: dict[str, Any] = _terminal_versioned_current_admission(
            source, launch_versioned_current=launch_versioned_current
        )
        terminal_pointer_valid = True
    except RouterAuthorityScanOuterError as exc:
        capture_error = capture_error or str(exc)
        terminal_versioned_current = {
            "validation_error": str(exc),
        }
    status = _terminal_status(terminal, dynamic_valid=dynamic_valid, capture_error=capture_error)
    receipt = _sealed(
        {
            "schema": OUTER_SCHEMA,
            "status": status,
            "recorded_at": _utc_now(),
            "launch_identity_sha256": launch_identity,
            "one_shot": {
                "automatic_retry_disabled": True,
                "same_capture_dir_never_starts_a_second_child": True,
                "outer_reaped_child": terminal.get("reaped") is True,
                "terminal_receipt_written_last": True,
            },
            "source_binding": {
                "outer_preflight": _binding(outer),
                "producer_preflight": _binding(producer_preflight),
                "producer_binary": producer_binary,
                "outer_launch_authority": _binding(launch_bound),
                "current_source": {name: _binding(bound) for name, bound in source.items()},
            },
            "versioned_current_admission": terminal_versioned_current,
            "terminal_current_pointer_valid": terminal_pointer_valid,
            "child": {
                "pid": child_pid,
                "started_at": started_at,
                "finished_at": _utc_now(),
                "command": command,
                "terminal": terminal,
            },
            "outer_capture": {
                "directory": str(capture_dir),
                "inner_capture_dir": str(inner),
                "stdout": stdout_evidence,
                "stderr": stderr_evidence,
                "dynamic_authority": dynamic_evidence,
            },
            "claim_boundary": {
                "cpu_router_authority_scan_only": True,
                "metal_device_or_dispatch_performed_by_outer": False,
                "lease_issued_or_consumed_by_outer": False,
                "server_watcher_hcli_or_token_execution_performed_by_outer": False,
                "tps_tg_or_tournament_claim_earned": False,
            },
            **({"capture_error": capture_error} if capture_error is not None else {}),
        }
    )
    # Completion marker: no further writes may occur in capture_dir after this.
    _write_new(capture_dir / OUTER_TERMINAL_FILENAME, receipt)
    return receipt


def _parse_args(arguments: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-one-shot", action="store_true")
    parser.add_argument("--producer-preflight", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--admission-current", type=Path)
    parser.add_argument("--joint-assessment", type=Path)
    parser.add_argument("--completion-preflight", type=Path)
    parser.add_argument("--producer-binary", type=Path, required=True)
    parser.add_argument("--outer-preflight", type=Path)
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--replay-guard-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args(arguments)
    if args.execute_one_shot:
        if args.outer_preflight is None or args.capture_dir is None or args.replay_guard_dir is None:
            parser.error("--execute-one-shot requires --outer-preflight, --capture-dir, and --replay-guard-dir")
        if any(
            value is not None
            for value in (
                args.producer_preflight,
                args.manifest,
                args.admission_current,
                args.joint_assessment,
                args.completion_preflight,
                args.out,
            )
        ):
            parser.error("--execute-one-shot accepts only the sealed --outer-preflight source authority")
    else:
        if args.out is None or any(
            value is None
            for value in (
                args.producer_preflight,
                args.manifest,
                args.admission_current,
                args.joint_assessment,
                args.completion_preflight,
            )
        ):
            parser.error("preflight requires producer/source inputs and --out")
        if any(value is not None for value in (args.outer_preflight, args.capture_dir, args.replay_guard_dir)):
            parser.error("preflight has no capture or replay path; omit outer/capture/replay arguments")
    return args


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if arguments is None else arguments)
    try:
        if args.execute_one_shot:
            receipt = run_attempt(
                CaptureConfig(
                    outer_preflight=args.outer_preflight,
                    producer_binary=args.producer_binary,
                    capture_dir=args.capture_dir,
                    replay_guard_dir=args.replay_guard_dir,
                    workers=args.workers,
                    timeout_seconds=args.timeout_seconds,
                )
            )
            print(json.dumps(receipt, sort_keys=True))
            return 0 if receipt.get("status") == CAPTURED_STATUS else 1
        outer = write_outer_preflight(
            PreflightConfig(
                producer_preflight=args.producer_preflight,
                manifest=args.manifest,
                admission_current=args.admission_current,
                joint_assessment=args.joint_assessment,
                completion_preflight=args.completion_preflight,
                producer_binary=args.producer_binary,
                workers=args.workers,
            ),
            args.out,
        )
        print(json.dumps(outer, sort_keys=True))
        return 0
    except RouterAuthorityScanOuterError as exc:
        print(
            json.dumps(
                {
                    "schema": OUTER_SCHEMA if args.execute_one_shot else OUTER_PREFLIGHT_SCHEMA,
                    "status": REFUSED_PREFIX + "LAUNCHER" if args.execute_one_shot else OUTER_PREFLIGHT_REFUSED_STATUS,
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
