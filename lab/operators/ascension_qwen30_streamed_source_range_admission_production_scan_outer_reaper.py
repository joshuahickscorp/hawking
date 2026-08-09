"""CPU/file-only one-shot outer/reaper preflight for Q30 production hash scans.

This module is deliberately *not* a process runner.  It has no subprocess,
source-root, model, GPU, server, HCLI, or lease-issuer surface.  It binds the
already-prepared bootstrap preflight/binary/resource records to a distinct
compiled production child, then reserves a future create-new/replay/terminal/
release lifecycle.  A prepared result never starts that lifecycle.

The future child is limited to one non-inference 16-shard / 18,867-range
bounded hash scan.  Its production flat map and hash-coverage capture are not
source-teacher execution; operator/reader attestations, runtime admission,
logits, native work, GPU, HCLI, and TPS remain prohibited here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_bootstrap_outer_preflight as bootstrap,
)
from lab.receipts import SealIntegrityError, seal, verify

SCHEMA = "hawking.ascension.qwen30_streamed_source_range_admission_production_scan_outer_reaper_preflight.v1"
PREPARED_STATUS = (
    "PREPARED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_OUTER_REAPER_NOT_SPAWNED"
)
REFUSED_STATUS = (
    "REFUSED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_OUTER_REAPER_PREREQUISITES_ABSENT_OR_INVALID"
)

PRODUCTION_BINARY_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_scan_binary_binding.v1"
)
PRODUCTION_BINARY_STATUS = (
    "COMPILED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_CPU_ONLY_NOT_EXECUTED"
)
PRODUCTION_RESOURCE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_scan_resource_admission.v1"
)
PRODUCTION_RESOURCE_STATUS = (
    "ADMITTED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_ZERO_SWAP_RESOURCE_WINDOW"
)
INTERFACE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_scan_interface.v1"
)
INTERFACE_STATUS = (
    "PREPARED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_INTERFACE_NOT_EXECUTED"
)
PRODUCTION_AUTHORITY_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_scan_authority.v1"
)
PRODUCTION_AUTHORITY_STATUS = (
    "ADMITTED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_ONE_SHOT"
)
LEASE_SCHEMA = bootstrap.LEASE_SCHEMA
LEASE_STATUS = bootstrap.LEASE_STATUS

REPLAY_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_replay_reservation.v1"
)
REPLAY_STATUS = (
    "RESERVED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_ONE_SHOT_NOT_SPAWNED"
)
CAPTURE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_capture.v1"
)
CAPTURE_STATUS = (
    "CAPTURED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_NOT_SOURCE_TEACHER"
)
OUTER_TERMINAL_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_outer_terminal.v1"
)
OUTER_TERMINAL_STATUS = (
    "CAPTURED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_OUTER_TERMINAL_NOT_SOURCE_TEACHER"
)
RELEASE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_quiet_lease_release.v1"
)
RELEASE_STATUS = (
    "RELEASED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_LEASE_AFTER_OUTER_TERMINAL"
)

MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_POSITIONED_READ_BYTES = 1024 * 1024
SOURCE_SHARDS = 16
SOURCE_TENSORS = 18_867


class ProductionScanOuterError(RuntimeError):
    """A record cannot safely reserve the future one-shot lifecycle."""


@dataclass(frozen=True)
class Document:
    path: Path
    document: dict[str, Any]
    raw_document_sha256: str
    seal_sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionScanOuterError(f"{label} must be an object")
    return dict(value)


def _sequence(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProductionScanOuterError(f"{label} must be an array")
    return list(value)


def _text(value: object, *, label: str, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ProductionScanOuterError(f"{label} must be non-empty text")
    if sha256 and (
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProductionScanOuterError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProductionScanOuterError(f"{label} must be an integer >= {minimum}")
    return value


def _require(value: object, *, expected: bool, label: str) -> None:
    if value is not expected:
        raise ProductionScanOuterError(f"{label} must be {expected}")


def _schema_status(
    document: Mapping[str, Any], *, schema: str, status: str, label: str
) -> None:
    if document.get("schema") != schema or document.get("status") != status:
        raise ProductionScanOuterError(f"{label} schema/status drifted")


def _regular_json(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.suffix != ".json":
        raise ProductionScanOuterError(f"{label} must be an absolute .json path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductionScanOuterError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProductionScanOuterError(f"{label} must be a regular non-symlink file")
    if metadata.st_size <= 0 or metadata.st_size > MAX_METADATA_BYTES:
        raise ProductionScanOuterError(f"{label} has invalid metadata size")
    return path.resolve(strict=True)


def _sealed(path: Path, *, label: str) -> Document:
    clean = _regular_json(path, label=label)
    try:
        raw_bytes = clean.read_bytes()
        raw = json.loads(raw_bytes)
        checked = verify(raw, label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise ProductionScanOuterError(f"{label} is absent or invalid: {exc}") from exc
    document = _mapping(checked, label=label)
    return Document(
        path=clean,
        document=document,
        raw_document_sha256=_sha256_bytes(raw_bytes),
        seal_sha256=_text(document.get("seal_sha256"), label=f"{label} seal", sha256=True),
    )


def _evidence(document: Document) -> dict[str, Any]:
    return {
        "path": str(document.path),
        "raw_document_sha256": document.raw_document_sha256,
        "seal_sha256": document.seal_sha256,
    }


def _pointer(value: object, *, expected: Document, label: str) -> None:
    pointer = _mapping(value, label=label)
    if (
        _text(pointer.get("raw_document_sha256"), label=f"{label}.raw_document_sha256", sha256=True)
        != expected.raw_document_sha256
        or _text(pointer.get("seal_sha256"), label=f"{label}.seal_sha256", sha256=True)
        != expected.seal_sha256
    ):
        raise ProductionScanOuterError(f"{label} does not bind the supplied sealed document")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _validate_bound_regular_file(
    value: object, *, label: str, executable: bool
) -> str:
    binding = _mapping(value, label=label)
    path = Path(_text(binding.get("path"), label=f"{label}.path"))
    if not path.is_absolute():
        raise ProductionScanOuterError(f"{label}.path must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductionScanOuterError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProductionScanOuterError(f"{label} must be a regular non-symlink file")
    if executable and not (metadata.st_mode & stat.S_IXUSR):
        raise ProductionScanOuterError(f"{label} must be executable by its owner")
    if _integer(binding.get("bytes"), label=f"{label}.bytes", minimum=1) != metadata.st_size:
        raise ProductionScanOuterError(f"{label} byte count drifted")
    observed = _sha256_bytes(path.read_bytes())
    declared = _text(binding.get("sha256"), label=f"{label}.sha256", sha256=True)
    if observed != declared:
        raise ProductionScanOuterError(f"{label} SHA-256 drifted")
    return declared


def _reject_fixture_identity(value: object, *, label: str) -> None:
    """Reject direct and nested fixture aliases before any future spawn surface."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"schema", "status"} and isinstance(child, str):
                lower = child.lower()
                if "fixture" in lower or "synthetic" in lower:
                    raise ProductionScanOuterError(
                        f"{label}.{key} carries fixture-only identity {child!r}"
                    )
            if key in {
                "fixture_only",
                "synthetic_fixture_only",
                "production_adapter_forbidden",
            } and child is True:
                raise ProductionScanOuterError(
                    f"{label}.{key} marks fixture-only or production-forbidden evidence"
                )
            _reject_fixture_identity(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_fixture_identity(child, label=f"{label}[{index}]")


def _validate_existing_bootstrap_chain(
    *,
    preflight_path: Path,
    binary_path: Path,
    resource_path: Path,
) -> tuple[Document, Document, Document, str, str]:
    """Reuse the established validators without giving this module a runner."""
    preflight = _sealed(preflight_path, label="bootstrap preflight")
    binary = _sealed(binary_path, label="bootstrap binary")
    resource = _sealed(resource_path, label="bootstrap resource")
    # Those validators consume only the sealed mappings and enforce the live
    # preflight/binary/resource chain, zero swap, and one-window geometry.
    bootstrap._validate_preflight(preflight)
    binary_sha = bootstrap._validate_binary(binary, preflight=preflight)
    resource_window_identity = bootstrap._validate_resource(
        resource, preflight=preflight, binary_sha=binary_sha
    )
    return preflight, binary, resource, binary_sha, resource_window_identity


def _validate_production_binary(
    document: Document,
    *,
    preflight: Document,
    bootstrap_binary: Document,
    resource: Document,
) -> str:
    _reject_fixture_identity(document.document, label="production binary")
    root = document.document
    _schema_status(
        root,
        schema=PRODUCTION_BINARY_SCHEMA,
        status=PRODUCTION_BINARY_STATUS,
        label="production binary",
    )
    for field in ("cpu_only", "production_hash_scan_backend_compiled"):
        _require(root.get(field), expected=True, label=f"production binary.{field}")
    for field in (
        "production_hash_scan_executed",
        "source_root_opened_or_statted",
        "source_payload_opened",
        "source_teacher_or_logits_executed",
        "model_gpu_server_hcli_or_tps_action",
        "lease_issued_or_consumed",
    ):
        _require(root.get(field), expected=False, label=f"production binary.{field}")
    binary_sha = _text(root.get("binary_sha256"), label="production binary SHA", sha256=True)
    if _validate_bound_regular_file(
        root.get("executable"), label="production binary executable", executable=True
    ) != binary_sha:
        raise ProductionScanOuterError("production binary executable binding drifted")
    source_sha = _text(root.get("source_sha256"), label="production binary source SHA", sha256=True)
    if _validate_bound_regular_file(
        root.get("source"), label="production binary source", executable=False
    ) != source_sha:
        raise ProductionScanOuterError("production binary source binding drifted")
    _pointer(root.get("bootstrap_preflight"), expected=preflight, label="production binary preflight")
    _pointer(
        root.get("bootstrap_binary"),
        expected=bootstrap_binary,
        label="production binary bootstrap binary",
    )
    _pointer(root.get("bootstrap_resource"), expected=resource, label="production binary resource")
    command = _sequence(root.get("compiled_command"), label="production binary compiled command")
    if command[:2] != ["cargo", "build"] or (
        "ascension_qwen30_streamed_source_range_admission_production_scan_interface"
        not in command
    ):
        raise ProductionScanOuterError("production binary compiled command drifted")
    if _text(
        root.get("compiled_command_sha256"),
        label="production binary compiled command SHA",
        sha256=True,
    ) != _sha256_bytes(_canonical_json(command)):
        raise ProductionScanOuterError("production binary compiled command hash drifted")
    return binary_sha


def _validate_interface(document: Document) -> None:
    _reject_fixture_identity(document.document, label="production interface")
    root = document.document
    _schema_status(root, schema=INTERFACE_SCHEMA, status=INTERFACE_STATUS, label="production interface")
    _require(root.get("prepared"), expected=True, label="production interface.prepared")
    _require(
        root.get("execution_authorized"),
        expected=False,
        label="production interface.execution_authorized",
    )


def _validate_production_resource(
    document: Document, *, production_binary: Document, bootstrap_resource: Document
) -> str:
    _reject_fixture_identity(document.document, label="production resource admission")
    root = document.document
    _schema_status(
        root,
        schema=PRODUCTION_RESOURCE_SCHEMA,
        status=PRODUCTION_RESOURCE_STATUS,
        label="production resource admission",
    )
    for field in (
        "prepared",
        "fresh_observation",
        "observed_after_production_binary_binding",
        "exclusive_clean_window",
        "zero_swap",
        "zero_swapouts",
        "no_active_q30_or_q80_capture_child",
        "resource_admitted_for_one_future_child",
    ):
        _require(root.get(field), expected=True, label=f"production resource.{field}")
    for field in (
        "source_payload_opened",
        "source_model_loaded",
        "source_teacher_or_logits_executed",
        "native_phase_started",
        "gpu_server_hcli_or_tps_action",
        "lease_issued_or_consumed",
        "child_started",
    ):
        _require(root.get(field), expected=False, label=f"production resource.{field}")
    if _integer(root.get("swap_used_bytes"), label="production resource swap") != 0:
        raise ProductionScanOuterError("production resource must show zero swap")
    if _integer(root.get("swapouts_pages_delta"), label="production resource swapouts") != 0:
        raise ProductionScanOuterError("production resource must show zero swapout growth")
    if _integer(root.get("reclaimable_bytes"), label="production resource reclaimable", minimum=1) < _integer(
        root.get("minimum_reclaimable_bytes_required"),
        label="production resource reclaimable floor",
        minimum=1,
    ):
        raise ProductionScanOuterError("production resource reclaimable floor is not met")
    _pointer(
        root.get("production_binary_binding"),
        expected=production_binary,
        label="production resource binary binding",
    )
    _pointer(
        root.get("bootstrap_resource_ancestry"),
        expected=bootstrap_resource,
        label="production resource legacy resource ancestry",
    )
    return _text(
        root.get("production_resource_window_identity_sha256"),
        label="production resource window identity",
        sha256=True,
    )
    bounded = _mapping(root.get("future_bounded_hash_scan"), label="production interface bounded")
    if (
        _integer(bounded.get("source_shards"), label="production interface shards")
        != SOURCE_SHARDS
        or _integer(bounded.get("source_tensors"), label="production interface tensors")
        != SOURCE_TENSORS
        or _integer(
            bounded.get("maximum_positioned_read_bytes"),
            label="production interface maximum read",
        )
        != MAX_POSITIONED_READ_BYTES
    ):
        raise ProductionScanOuterError("production interface geometry drifted")
    boundary = _mapping(root.get("strict_non_fixture_boundary"), label="production interface boundary")
    _require(
        boundary.get("before_source_root_access"),
        expected=True,
        label="production interface fixture rejection order",
    )


def _validate_production_authority(
    document: Document,
    *,
    interface: Document,
    production_binary: Document,
    production_resource: Document,
) -> str:
    _reject_fixture_identity(document.document, label="production scan authority")
    root = document.document
    _schema_status(
        root,
        schema=PRODUCTION_AUTHORITY_SCHEMA,
        status=PRODUCTION_AUTHORITY_STATUS,
        label="production scan authority",
    )
    for field in (
        "fresh_for_this_exact_scan",
        "one_shot",
        "non_inference_hash_scan_only",
        "source_root_open_only_after_all_authorities_validate",
    ):
        _require(root.get(field), expected=True, label=f"production authority.{field}")
    for field in (
        "fixture_only",
        "synthetic_fixture_only",
        "production_adapter_forbidden",
        "source_teacher_or_logits_authorized",
        "model_gpu_server_hcli_or_tps_authorized",
        "lease_consumed",
    ):
        _require(root.get(field), expected=False, label=f"production authority.{field}")
    bindings = _mapping(root.get("immutable_bindings"), label="production authority bindings")
    _pointer(
        bindings.get("interface_authority"),
        expected=interface,
        label="production authority interface",
    )
    _pointer(
        bindings.get("production_binary"),
        expected=production_binary,
        label="production authority production binary",
    )
    _pointer(
        bindings.get("production_resource_admission"),
        expected=production_resource,
        label="production authority production resource admission",
    )
    geometry = _mapping(root.get("geometry"), label="production authority geometry")
    if (
        _integer(geometry.get("source_shards"), label="production authority shards")
        != SOURCE_SHARDS
        or _integer(geometry.get("source_tensors"), label="production authority tensors")
        != SOURCE_TENSORS
        or _integer(
            geometry.get("maximum_positioned_read_bytes"),
            label="production authority maximum read",
        )
        != MAX_POSITIONED_READ_BYTES
        or _integer(
            geometry.get("maximum_live_raw_bf16_windows"),
            label="production authority live windows",
        )
        != 1
    ):
        raise ProductionScanOuterError("production authority geometry drifted")
    _text(root.get("exact_scan_nonce_sha256"), label="production authority nonce", sha256=True)
    return _text(
        root.get("exact_scan_nonce_sha256"), label="production authority nonce", sha256=True
    )


def _validate_fresh_lease(
    document: Document,
    *,
    authority: Document,
    production_binary: Document,
    bootstrap_resource: Document,
    production_resource: Document,
    resource_window_identity: str,
) -> str:
    _reject_fixture_identity(document.document, label="production bootstrap lease")
    root = document.document
    _schema_status(root, schema=LEASE_SCHEMA, status=LEASE_STATUS, label="production bootstrap lease")
    for field in (
        "fresh_for_this_exact_launch",
        "one_shot",
        "non_inference_only",
        "new_capture_root_required",
        "existing_output_reuse_forbidden",
        "replay_or_relaunch_forbidden",
        "separate_from_source_teacher_lease",
        "production_source_hash_scan_only",
    ):
        _require(root.get(field), expected=True, label=f"production lease.{field}")
    for field in (
        "fixture_only",
        "synthetic_fixture_only",
        "production_adapter_forbidden",
        "source_teacher_or_logits_authorized",
        "model_gpu_server_hcli_or_tps_authorized",
        "lease_consumed_by_this_preflight",
    ):
        _require(root.get(field), expected=False, label=f"production lease.{field}")
    _pointer(
        root.get("production_scan_authority"),
        expected=authority,
        label="production lease authority",
    )
    production_binary_sha256 = _text(
        production_binary.document.get("binary_sha256"),
        label="validated production binary SHA",
        sha256=True,
    )
    if _text(root.get("production_binary_sha256"), label="production lease binary SHA", sha256=True) != production_binary_sha256:
        raise ProductionScanOuterError("production lease binary binding drifted")
    _pointer(
        root.get("production_binary_binding"),
        expected=production_binary,
        label="production lease binary binding",
    )
    _pointer(
        root.get("bootstrap_resource_ancestry"),
        expected=bootstrap_resource,
        label="production lease legacy resource ancestry",
    )
    _pointer(
        root.get("production_resource_admission"),
        expected=production_resource,
        label="production lease fresh resource admission",
    )
    if _text(
        root.get("resource_window_identity_sha256"),
        label="production lease resource window",
        sha256=True,
    ) != resource_window_identity:
        raise ProductionScanOuterError("production lease resource-window binding drifted")
    observation = _mapping(
        root.get("fresh_production_resource_observation"),
        label="production lease fresh resource observation",
    )
    for field in (
        "observed_after_production_binary_binding",
        "exclusive_clean_window",
        "zero_swap",
        "zero_swapouts",
        "no_active_q30_or_q80_capture_child",
    ):
        _require(
            observation.get(field),
            expected=True,
            label=f"production lease fresh resource observation.{field}",
        )
    for field in (
        "source_payload_opened",
        "source_model_loaded",
        "gpu_server_hcli_or_tps_action",
        "lease_issued_or_consumed",
        "child_started",
    ):
        _require(
            observation.get(field),
            expected=False,
            label=f"production lease fresh resource observation.{field}",
        )
    if _integer(observation.get("swap_used_bytes"), label="production lease fresh swap") != 0:
        raise ProductionScanOuterError("production lease fresh resource observation must show zero swap")
    if (
        _integer(
            observation.get("swapouts_pages_delta"),
            label="production lease fresh swapouts",
        )
        != 0
    ):
        raise ProductionScanOuterError(
            "production lease fresh resource observation must show zero swapout growth"
        )
    if _integer(
        observation.get("reclaimable_bytes"),
        label="production lease fresh reclaimable",
        minimum=1,
    ) < _integer(
        observation.get("minimum_reclaimable_bytes_required"),
        label="production lease fresh reclaimable floor",
        minimum=1,
    ):
        raise ProductionScanOuterError(
            "production lease fresh resource observation reclaimable floor is not met"
        )
    return _text(root.get("lease_id"), label="production lease ID", sha256=True)


def _future_lifecycle() -> dict[str, Any]:
    return {
        "future_child_command": [
            "ascension_qwen30_streamed_source_range_admission_production_scan_interface",
            "--mode",
            "production-scan",
            "--range-authority",
            "ABSOLUTE_METADATA_RANGE_AUTHORITY_JSON",
            "--semantics-attester",
            "ABSOLUTE_NON_FIXTURE_SEMANTICS_ATTESTER_JSON",
            "--runtime-admission-authority",
            "ABSOLUTE_SEALED_RUNTIME_PRODUCER_AUTHORITY_JSON",
            "--interface-authority",
            "ABSOLUTE_SEALED_PRODUCTION_INTERFACE_JSON",
            "--production-scan-authority",
            "ABSOLUTE_SEALED_FRESH_PRODUCTION_SCAN_AUTHORITY_JSON",
            "--bootstrap-lease",
            "ABSOLUTE_SEALED_FRESH_PRODUCTION_BOOTSTRAP_LEASE_JSON",
            "--source-root",
            "ABSOLUTE_CANONICAL_QWEN30_SOURCE_ROOT",
            "--capture-dir",
            "NEW_ABSOLUTE_CAPTURE_DIRECTORY",
            "--out",
            "NEW_ABSOLUTE_PRODUCTION_CAPTURE_RECEIPT_JSON",
        ],
        "replay_reservation": {
            "schema": REPLAY_SCHEMA,
            "status": REPLAY_STATUS,
            "create_new_before_child": True,
            "one_child_maximum": True,
            "replay_or_relaunch_forbidden": True,
        },
        "child_capture": {
            "schema": CAPTURE_SCHEMA,
            "status": CAPTURE_STATUS,
            "source_shards": SOURCE_SHARDS,
            "source_tensors": SOURCE_TENSORS,
            "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
            "one_raw_window": True,
            "receipt_written_last": True,
            "source_teacher_runtime_attestations_or_logits_forbidden": True,
        },
        "outer_terminal": {
            "schema": OUTER_TERMINAL_SCHEMA,
            "status": OUTER_TERMINAL_STATUS,
            "child_must_be_reaped_before_terminal": True,
            "terminal_receipt_written_after_child_capture": True,
        },
        "lease_release": {
            "schema": RELEASE_SCHEMA,
            "status": RELEASE_STATUS,
            "separate_release_after_outer_terminal": True,
            "release_cannot_authorize_retry_or_teacher": True,
        },
    }


def build_outer_preflight(
    *,
    bootstrap_preflight_path: Path,
    bootstrap_binary_path: Path,
    bootstrap_resource_path: Path,
    production_binary_path: Path | None = None,
    production_resource_path: Path | None = None,
    production_interface_path: Path | None = None,
    production_authority_path: Path | None = None,
    bootstrap_lease_path: Path | None = None,
) -> dict[str, Any]:
    """Return sealed PREPARED/REFUSED metadata without any spawn/lease action."""
    (
        bootstrap_preflight,
        bootstrap_binary,
        bootstrap_resource,
        _bootstrap_binary_sha,
        resource_window_identity,
    ) = _validate_existing_bootstrap_chain(
        preflight_path=bootstrap_preflight_path,
        binary_path=bootstrap_binary_path,
        resource_path=bootstrap_resource_path,
    )
    blockers: list[str] = []
    production_binary: Document | None = None
    production_interface: Document | None = None
    production_resource: Document | None = None
    production_authority: Document | None = None
    production_binary_sha: str | None = None
    scan_nonce: str | None = None
    lease: Document | None = None

    if production_binary_path is None:
        blockers.append("compiled_production_hash_scan_binary_binding_absent")
    else:
        try:
            production_binary = _sealed(production_binary_path, label="production binary")
            production_binary_sha = _validate_production_binary(
                production_binary,
                preflight=bootstrap_preflight,
                bootstrap_binary=bootstrap_binary,
                resource=bootstrap_resource,
            )
        except ProductionScanOuterError as exc:
            blockers.append(f"compiled_production_hash_scan_binary_binding_invalid:{exc}")

    if production_interface_path is None:
        blockers.append("sealed_production_hash_scan_interface_absent")
    else:
        try:
            production_interface = _sealed(production_interface_path, label="production interface")
            _validate_interface(production_interface)
        except ProductionScanOuterError as exc:
            blockers.append(f"sealed_production_hash_scan_interface_invalid:{exc}")

    if production_resource_path is None:
        blockers.append("fresh_production_binary_bound_resource_admission_absent")
    elif production_binary is None:
        blockers.append("production_resource_admission_not_evaluated_without_binary")
    else:
        try:
            production_resource = _sealed(
                production_resource_path, label="production resource admission"
            )
            _validate_production_resource(
                production_resource,
                production_binary=production_binary,
                bootstrap_resource=bootstrap_resource,
            )
        except ProductionScanOuterError as exc:
            blockers.append(f"fresh_production_binary_bound_resource_admission_invalid:{exc}")

    if production_authority_path is None:
        blockers.append("fresh_production_hash_scan_authority_absent")
    elif (
        production_binary is None
        or production_interface is None
        or production_resource is None
    ):
        blockers.append(
            "production_scan_authority_not_evaluated_without_binary_interface_and_resource"
        )
    else:
        try:
            production_authority = _sealed(
                production_authority_path, label="production scan authority"
            )
            scan_nonce = _validate_production_authority(
                production_authority,
                interface=production_interface,
                production_binary=production_binary,
                production_resource=production_resource,
            )
        except ProductionScanOuterError as exc:
            blockers.append(f"fresh_production_hash_scan_authority_invalid:{exc}")

    if bootstrap_lease_path is None:
        blockers.append("fresh_production_hash_scan_bootstrap_lease_absent")
    elif production_authority is None or production_binary_sha is None:
        blockers.append("production_bootstrap_lease_not_evaluated_without_authority_and_binary")
    else:
        try:
            lease = _sealed(bootstrap_lease_path, label="production bootstrap lease")
            _validate_fresh_lease(
                lease,
                authority=production_authority,
                production_binary=production_binary,
                bootstrap_resource=bootstrap_resource,
                production_resource=production_resource,
                resource_window_identity=resource_window_identity,
            )
        except ProductionScanOuterError as exc:
            blockers.append(f"fresh_production_hash_scan_bootstrap_lease_invalid:{exc}")

    prepared = not blockers
    return seal(
        {
            "schema": SCHEMA,
            "status": PREPARED_STATUS if prepared else REFUSED_STATUS,
            "prepared": prepared,
            "spawn_permitted": False,
            "bootstrap_preflight": _evidence(bootstrap_preflight),
            "bootstrap_binary": _evidence(bootstrap_binary),
            "bootstrap_resource": _evidence(bootstrap_resource),
            "legacy_bootstrap_resource_is_ancestry_only": True,
            "production_binary": _evidence(production_binary)
            if production_binary
            else {"present": False},
            "production_interface": _evidence(production_interface)
            if production_interface
            else {"present": False},
            "fresh_production_binary_bound_resource_admission": _evidence(production_resource)
            if production_resource
            else {"present": False},
            "production_scan_authority": _evidence(production_authority)
            if production_authority
            else {"present": False},
            "fresh_bootstrap_lease": _evidence(lease) if lease else {"present": False},
            "expected_scan_nonce_sha256": scan_nonce,
            "future_lifecycle": _future_lifecycle(),
            "blockers": blockers,
            "execution_boundary": {
                "source_root_opened_or_statted": False,
                "source_payload_opened": False,
                "source_teacher_or_logits_executed": False,
                "native_phase_started": False,
                "gpu_server_hcli_or_tps_action": False,
                "lease_issued_or_consumed": False,
                "replay_reservation_created": False,
                "child_spawned": False,
                "child_reaped": False,
                "outer_terminal_written": False,
                "lease_released": False,
            },
            "claim_boundary": "CPU/file-only outer/reaper preflight. It does not create a replay/capture directory, spawn/reap a child, open source payloads, issue/consume/release a lease, earn a map/coverage/capture, start teacher/native/model/GPU/server/HCLI work, or report TPS/TG/tournament evidence.",
        }
    )


def validate_fake_child_reap_terminal_release(
    *,
    reservation: Mapping[str, Any],
    child_capture: Mapping[str, Any],
    outer_terminal: Mapping[str, Any],
    release: Mapping[str, Any],
    outer_preflight_seal_sha256: str,
    production_binary_sha256: str,
    production_authority_seal_sha256: str,
    lease_id: str,
) -> None:
    """Validate future fake child/reaper mappings; never starts a process."""
    _schema_status(reservation, schema=REPLAY_SCHEMA, status=REPLAY_STATUS, label="replay")
    for field in (
        "create_new_before_child",
        "one_child_maximum",
        "replay_or_relaunch_forbidden",
    ):
        _require(reservation.get(field), expected=True, label=f"replay.{field}")
    if _integer(reservation.get("attempt"), label="replay.attempt", minimum=1) != 1:
        raise ProductionScanOuterError("replay attempt must be exactly one")
    if _text(reservation.get("lease_id"), label="replay lease ID", sha256=True) != lease_id:
        raise ProductionScanOuterError("replay lease binding drifted")
    if (
        _text(
            reservation.get("outer_preflight_seal_sha256"),
            label="replay outer preflight seal",
            sha256=True,
        )
        != outer_preflight_seal_sha256
    ):
        raise ProductionScanOuterError("replay outer-preflight binding drifted")

    _schema_status(
        child_capture, schema=CAPTURE_SCHEMA, status=CAPTURE_STATUS, label="fake child capture"
    )
    for field in (
        "production_hash_scan_earned",
        "receipt_written_last",
        "source_handles_closed",
        "reader_cache_zeroed",
    ):
        _require(child_capture.get(field), expected=True, label=f"fake child capture.{field}")
    for field in (
        "source_teacher_or_logits_executed",
        "operator_or_reader_execution_attestation_emitted",
        "source_teacher_runtime_admission_earned",
        "model_gpu_server_hcli_or_tps_action",
    ):
        _require(child_capture.get(field), expected=False, label=f"fake child capture.{field}")
    geometry = _mapping(child_capture.get("geometry"), label="fake child geometry")
    if (
        _integer(geometry.get("source_shards"), label="fake child shards") != SOURCE_SHARDS
        or _integer(geometry.get("source_tensors"), label="fake child tensors")
        != SOURCE_TENSORS
        or _integer(
            geometry.get("maximum_positioned_read_bytes"), label="fake child maximum read"
        )
        != MAX_POSITIONED_READ_BYTES
        or _integer(geometry.get("maximum_live_raw_bf16_windows"), label="fake child windows")
        != 1
    ):
        raise ProductionScanOuterError("fake child geometry drifted")
    if (
        _text(
            child_capture.get("production_binary_sha256"),
            label="fake child binary SHA",
            sha256=True,
        )
        != production_binary_sha256
        or _text(
            child_capture.get("production_authority_seal_sha256"),
            label="fake child authority seal",
            sha256=True,
        )
        != production_authority_seal_sha256
        or _text(child_capture.get("lease_id"), label="fake child lease ID", sha256=True)
        != lease_id
    ):
        raise ProductionScanOuterError("fake child immutable launch binding drifted")

    _schema_status(
        outer_terminal,
        schema=OUTER_TERMINAL_SCHEMA,
        status=OUTER_TERMINAL_STATUS,
        label="fake outer terminal",
    )
    for field in (
        "child_reaped",
        "terminal_receipt_written_after_child_capture",
        "automatic_retry_disabled",
        "lease_reuse_prohibited",
    ):
        _require(outer_terminal.get(field), expected=True, label=f"fake terminal.{field}")
    _require(outer_terminal.get("child_timed_out"), expected=False, label="fake terminal timeout")
    if _integer(outer_terminal.get("child_exit_code"), label="fake terminal exit") != 0:
        raise ProductionScanOuterError("fake terminal child exit must be zero")
    if (
        _text(outer_terminal.get("lease_id"), label="fake terminal lease ID", sha256=True)
        != lease_id
        or _text(
            outer_terminal.get("production_binary_sha256"),
            label="fake terminal binary SHA",
            sha256=True,
        )
        != production_binary_sha256
        or _text(
            outer_terminal.get("production_authority_seal_sha256"),
            label="fake terminal authority seal",
            sha256=True,
        )
        != production_authority_seal_sha256
    ):
        raise ProductionScanOuterError("fake terminal immutable binding drifted")
    child_seal = _text(
        outer_terminal.get("child_capture_seal_sha256"),
        label="fake terminal child capture seal",
        sha256=True,
    )

    _schema_status(release, schema=RELEASE_SCHEMA, status=RELEASE_STATUS, label="fake release")
    for field in (
        "release_after_outer_terminal",
        "one_shot_lease_finalized",
        "retry_or_relaunch_forbidden",
    ):
        _require(release.get(field), expected=True, label=f"fake release.{field}")
    for field in (
        "source_teacher_or_logits_authorized",
        "native_or_gpu_server_hcli_authorized",
        "artifacts_deleted_or_evicted",
    ):
        _require(release.get(field), expected=False, label=f"fake release.{field}")
    if (
        _text(release.get("lease_id"), label="fake release lease ID", sha256=True) != lease_id
        or _text(
            release.get("outer_terminal_seal_sha256"),
            label="fake release terminal seal",
            sha256=True,
        )
        != _text(outer_terminal.get("seal_sha256"), label="fake terminal seal", sha256=True)
        or _text(
            release.get("child_capture_seal_sha256"),
            label="fake release child capture seal",
            sha256=True,
        )
        != child_seal
    ):
        raise ProductionScanOuterError("fake release binding drifted")


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise ProductionScanOuterError("--out must be a new absolute path below an existing parent")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-preflight", type=Path, required=True)
    parser.add_argument("--bootstrap-binary", type=Path, required=True)
    parser.add_argument("--bootstrap-resource", type=Path, required=True)
    parser.add_argument("--production-binary", type=Path)
    parser.add_argument("--production-resource-admission", type=Path)
    parser.add_argument("--production-interface", type=Path)
    parser.add_argument("--production-scan-authority", type=Path)
    parser.add_argument("--bootstrap-lease", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_outer_preflight(
            bootstrap_preflight_path=args.bootstrap_preflight,
            bootstrap_binary_path=args.bootstrap_binary,
            bootstrap_resource_path=args.bootstrap_resource,
            production_binary_path=args.production_binary,
            production_resource_path=args.production_resource_admission,
            production_interface_path=args.production_interface,
            production_authority_path=args.production_scan_authority,
            bootstrap_lease_path=args.bootstrap_lease,
        )
        _write_new(args.out, result)
    except ProductionScanOuterError as exc:
        print(f"Q30 production hash-scan outer/reaper preflight refused: {exc}")
        return 2
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "status": result["status"],
                "seal_sha256": result["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
