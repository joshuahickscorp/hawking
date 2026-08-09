#!/usr/bin/env python3
"""Explicit one-shot lease issuer and receipt-last outer reaper for Q30 hash scans.

Nothing in ``preflight`` creates a directory, lease, child, or source-root
probe.  ``execute`` is deliberately a separate, explicit mode.  It can issue
one fresh lease only after revalidating the sealed no-lease outer preflight,
then runs the exact bound production scanner once, reaps it, and writes a
terminal receipt followed by a final one-shot release receipt.  It never
permits source-teacher, native, GPU, server, HCLI, TPS, TG, or tournament work.

This file is an outer lifecycle controller; it does not implement source
hashing and is not invoked by this change.
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

from lab.operators import (
    ascension_qwen30_streamed_source_range_admission_production_scan_outer_reaper as outer,
)
from lab.receipts import seal

SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_scan_"
    "lease_outer_reaper.v1"
)
PREFLIGHT_STATUS = (
    "PREPARED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_"
    "LEASE_ISSUER_AND_OUTER_REAPER_NOT_EXECUTED"
)
REFUSED_STATUS = (
    "REFUSED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_"
    "LEASE_ISSUER_AND_OUTER_REAPER_PREREQUISITES_ABSENT_OR_INVALID"
)
STARTED_STATUS = (
    "STARTED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_"
    "ONE_SHOT_OUTER_REAPER"
)
TERMINAL_REFUSED_STATUS = (
    "REFUSED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_"
    "OUTER_TERMINAL_CHILD_FAILED_OR_INVALID"
)
REPLAY_FILENAME = "replay-reservation.json"
RUNNING_FILENAME = "outer-running.json"
CHILD_FILENAME = "child.json"
TERMINAL_FILENAME = "outer-terminal-receipt.json"
LEASE_FILENAME = "production-bootstrap-lease.json"
ISSUED_OUTER_FILENAME = "issued-outer-preflight.json"
RELEASE_FILENAME = "quiet-lease-release.json"
CHILD_CAPTURE_DIRNAME = "child-capture"
CHILD_RECEIPT_FILENAME = "receipt.json"
STDOUT_FILENAME = "child.stdout.log"
STDERR_FILENAME = "child.stderr.log"
MAX_CHILD_STREAM_BYTES = 1_000_000


class ProductionLeaseOuterError(RuntimeError):
    """A one-shot Q30 production scan cannot safely be issued or reaped."""


@dataclass(frozen=True)
class LeaseContext:
    outer_preflight: outer.Document
    bootstrap_preflight: outer.Document
    bootstrap_binary: outer.Document
    bootstrap_resource: outer.Document
    production_binary: outer.Document
    production_resource: outer.Document
    production_interface: outer.Document
    production_authority: outer.Document
    bootstrap_resource_window_identity_sha256: str
    production_resource_window_identity_sha256: str
    production_binary_sha256: str


@dataclass(frozen=True)
class ExecuteConfig:
    outer_preflight_path: Path
    bootstrap_preflight_path: Path
    bootstrap_binary_path: Path
    bootstrap_resource_path: Path
    production_binary_path: Path
    production_resource_path: Path
    production_interface_path: Path
    production_authority_path: Path
    launch_dir: Path
    replay_dir: Path
    capture_dir: Path
    source_root: Path | None = None
    timeout_seconds: float = 7200.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionLeaseOuterError(f"{label} must be an object")
    return dict(value)


def _text(value: object, *, label: str, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ProductionLeaseOuterError(f"{label} must be non-empty text")
    if sha256 and (
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProductionLeaseOuterError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProductionLeaseOuterError(f"{label} must be an integer >= {minimum}")
    return value


def _evidence(document: outer.Document) -> dict[str, str]:
    return {
        "path": str(document.path),
        "raw_document_sha256": document.raw_document_sha256,
        "seal_sha256": document.seal_sha256,
    }


def _scanner_sealed_pointer(document: outer.Document) -> dict[str, str]:
    """Match the Rust scanner's exact sealed-input pointer ABI."""
    return {
        **_evidence(document),
        "canonical_document_sha256": _sha256_bytes(_canonical_json(document.document)),
    }


def _pointer(value: object, *, expected: outer.Document, label: str) -> None:
    try:
        outer._pointer(value, expected=expected, label=label)
    except outer.ProductionScanOuterError as exc:
        raise ProductionLeaseOuterError(str(exc)) from exc


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise ProductionLeaseOuterError("output must be a new absolute path below an existing directory")
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


def _mkdir_new(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise ProductionLeaseOuterError(f"{label} must be a new absolute directory below an existing parent")
    try:
        path.mkdir(mode=0o750)
    except OSError as exc:
        raise ProductionLeaseOuterError(f"cannot create {label}: {exc}") from exc
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductionLeaseOuterError(f"cannot stat new {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ProductionLeaseOuterError(f"new {label} is not a regular directory")
    return path.resolve(strict=True)


def _validate_new_layout(config: ExecuteConfig) -> tuple[Path, Path, Path]:
    launch = config.launch_dir
    replay = config.replay_dir
    capture = config.capture_dir
    paths = (launch, replay, capture)
    if len({str(path) for path in paths}) != len(paths):
        raise ProductionLeaseOuterError("launch/replay/capture directories must be distinct")
    for path, label in zip(paths, ("launch directory", "replay directory", "capture directory")):
        if not path.is_absolute() or path.exists() or not path.parent.is_dir():
            raise ProductionLeaseOuterError(f"{label} must be a new absolute path below an existing directory")
    return launch, replay, capture


def _load_context(config: ExecuteConfig) -> LeaseContext:
    """Read and bind all metadata receipts without creating anything."""
    try:
        baseline = outer._sealed(config.outer_preflight_path, label="no-lease outer preflight")
        bootstrap_preflight, bootstrap_binary, bootstrap_resource, _bootstrap_sha, bootstrap_window = (
            outer._validate_existing_bootstrap_chain(
                preflight_path=config.bootstrap_preflight_path,
                binary_path=config.bootstrap_binary_path,
                resource_path=config.bootstrap_resource_path,
            )
        )
        production_binary = outer._sealed(config.production_binary_path, label="production binary")
        production_binary_sha = outer._validate_production_binary(
            production_binary,
            preflight=bootstrap_preflight,
            bootstrap_binary=bootstrap_binary,
            resource=bootstrap_resource,
        )
        production_resource = outer._sealed(
            config.production_resource_path, label="production resource admission"
        )
        production_window = outer._validate_production_resource(
            production_resource,
            production_binary=production_binary,
            bootstrap_resource=bootstrap_resource,
        )
        production_interface = outer._sealed(
            config.production_interface_path, label="production interface"
        )
        outer._validate_interface(production_interface)
        production_authority = outer._sealed(
            config.production_authority_path, label="production authority"
        )
        outer._validate_production_authority(
            production_authority,
            interface=production_interface,
            production_binary=production_binary,
            production_resource=production_resource,
        )
    except outer.ProductionScanOuterError as exc:
        raise ProductionLeaseOuterError(f"current production authority chain is invalid: {exc}") from exc

    _validate_baseline_outer(
        baseline=baseline,
        bootstrap_preflight=bootstrap_preflight,
        bootstrap_binary=bootstrap_binary,
        bootstrap_resource=bootstrap_resource,
        production_binary=production_binary,
        production_resource=production_resource,
        production_interface=production_interface,
        production_authority=production_authority,
    )
    return LeaseContext(
        outer_preflight=baseline,
        bootstrap_preflight=bootstrap_preflight,
        bootstrap_binary=bootstrap_binary,
        bootstrap_resource=bootstrap_resource,
        production_binary=production_binary,
        production_resource=production_resource,
        production_interface=production_interface,
        production_authority=production_authority,
        bootstrap_resource_window_identity_sha256=bootstrap_window,
        production_resource_window_identity_sha256=production_window,
        production_binary_sha256=production_binary_sha,
    )


def _validate_baseline_outer(
    *,
    baseline: outer.Document,
    bootstrap_preflight: outer.Document,
    bootstrap_binary: outer.Document,
    bootstrap_resource: outer.Document,
    production_binary: outer.Document,
    production_resource: outer.Document,
    production_interface: outer.Document,
    production_authority: outer.Document,
) -> None:
    root = baseline.document
    if root.get("schema") != outer.SCHEMA or root.get("status") != outer.REFUSED_STATUS:
        raise ProductionLeaseOuterError("baseline outer preflight must be the sealed no-lease refusal")
    if root.get("prepared") is not False or root.get("spawn_permitted") is not False:
        raise ProductionLeaseOuterError("baseline outer preflight must not be prepared or spawnable")
    if root.get("blockers") != ["fresh_production_hash_scan_bootstrap_lease_absent"]:
        raise ProductionLeaseOuterError("baseline outer preflight must have exactly the fresh-lease blocker")
    _pointer(root.get("bootstrap_preflight"), expected=bootstrap_preflight, label="baseline bootstrap preflight")
    _pointer(root.get("bootstrap_binary"), expected=bootstrap_binary, label="baseline bootstrap binary")
    _pointer(root.get("bootstrap_resource"), expected=bootstrap_resource, label="baseline bootstrap resource")
    _pointer(root.get("production_binary"), expected=production_binary, label="baseline production binary")
    _pointer(
        root.get("fresh_production_binary_bound_resource_admission"),
        expected=production_resource,
        label="baseline production resource",
    )
    _pointer(root.get("production_interface"), expected=production_interface, label="baseline interface")
    _pointer(
        root.get("production_scan_authority"),
        expected=production_authority,
        label="baseline production authority",
    )


def _lease_document(context: LeaseContext, *, lease_id: str) -> dict[str, Any]:
    resource_root = context.production_resource.document
    observation = {
        key: resource_root.get(key)
        for key in (
            "observed_after_production_binary_binding",
            "exclusive_clean_window",
            "zero_swap",
            "zero_swapouts",
            "no_active_q30_or_q80_capture_child",
            "source_payload_opened",
            "source_model_loaded",
            "gpu_server_hcli_or_tps_action",
            "lease_issued_or_consumed",
            "child_started",
            "swap_used_bytes",
            "swapouts_pages_delta",
            "reclaimable_bytes",
            "minimum_reclaimable_bytes_required",
        )
    }
    return seal(
        {
            "schema": outer.LEASE_SCHEMA,
            "status": outer.LEASE_STATUS,
            "recorded_at": _utc_now(),
            "lease_id": lease_id,
            "fresh_for_this_exact_launch": True,
            "one_shot": True,
            "non_inference_only": True,
            "new_capture_root_required": True,
            "existing_output_reuse_forbidden": True,
            "replay_or_relaunch_forbidden": True,
            "separate_from_source_teacher_lease": True,
            "production_source_hash_scan_only": True,
            "fixture_only": False,
            "synthetic_fixture_only": False,
            "production_adapter_forbidden": False,
            "source_teacher_or_logits_authorized": False,
            "model_gpu_server_hcli_or_tps_authorized": False,
            "lease_consumed_by_this_preflight": False,
            "production_scan_authority": _scanner_sealed_pointer(context.production_authority),
            "production_binary_sha256": context.production_binary_sha256,
            "production_binary_binding": _evidence(context.production_binary),
            "bootstrap_resource_ancestry": _evidence(context.bootstrap_resource),
            "production_resource_admission": _evidence(context.production_resource),
            # Existing scanner validation retains the legacy resource-window
            # field for ancestry; the next field binds the fresh production
            # observation that actually gates this lease.
            "resource_window_identity_sha256": context.bootstrap_resource_window_identity_sha256,
            "production_resource_window_identity_sha256": context.production_resource_window_identity_sha256,
            "fresh_production_resource_observation": observation,
            "baseline_outer_preflight": _evidence(context.outer_preflight),
            "claim_boundary": "Fresh one-shot non-inference production hash-scan lease only. It authorizes neither source-teacher semantics/logits nor native/model/GPU/server/HCLI/TPS/TG/tournament work, and is usable only by the receipt-last outer reaper.",
        }
    )


def _validate_issued_lease(path: Path, context: LeaseContext) -> outer.Document:
    try:
        lease = outer._sealed(path, label="issued production bootstrap lease")
        outer._validate_fresh_lease(
            lease,
            authority=context.production_authority,
            production_binary=context.production_binary,
            bootstrap_resource=context.bootstrap_resource,
            production_resource=context.production_resource,
            resource_window_identity=context.bootstrap_resource_window_identity_sha256,
        )
    except outer.ProductionScanOuterError as exc:
        raise ProductionLeaseOuterError(f"issued lease failed self-validation: {exc}") from exc
    if (
        lease.document.get("production_resource_window_identity_sha256")
        != context.production_resource_window_identity_sha256
    ):
        raise ProductionLeaseOuterError("issued lease lost its fresh production resource-window binding")
    pointer = _mapping(
        lease.document.get("production_scan_authority"), label="issued lease production authority"
    )
    expected_canonical = _sha256_bytes(_canonical_json(context.production_authority.document))
    if pointer.get("canonical_document_sha256") != expected_canonical:
        raise ProductionLeaseOuterError("issued lease production authority canonical binding drifted")
    return lease


def _issued_outer_document(context: LeaseContext, *, lease_path: Path) -> dict[str, Any]:
    try:
        result = outer.build_outer_preflight(
            bootstrap_preflight_path=context.bootstrap_preflight.path,
            bootstrap_binary_path=context.bootstrap_binary.path,
            bootstrap_resource_path=context.bootstrap_resource.path,
            production_binary_path=context.production_binary.path,
            production_resource_path=context.production_resource.path,
            production_interface_path=context.production_interface.path,
            production_authority_path=context.production_authority.path,
            bootstrap_lease_path=lease_path,
        )
    except outer.ProductionScanOuterError as exc:
        raise ProductionLeaseOuterError(f"cannot revalidate outer after lease issue: {exc}") from exc
    if result.get("status") != outer.PREPARED_STATUS or result.get("prepared") is not True:
        raise ProductionLeaseOuterError("issued lease did not produce a prepared non-spawning outer")
    if result.get("spawn_permitted") is not False or result.get("blockers") != []:
        raise ProductionLeaseOuterError("prepared outer crossed a spawn or blocker boundary")
    return result


def build_issuer_preflight(config: ExecuteConfig) -> dict[str, Any]:
    """Return a sealed preflight without a lease, source probe, directory, or child."""
    try:
        context = _load_context(config)
        blockers: list[str] = []
    except ProductionLeaseOuterError as exc:
        context = None
        blockers = [str(exc)]
    return seal(
        {
            "schema": SCHEMA,
            "status": PREFLIGHT_STATUS if not blockers else REFUSED_STATUS,
            "prepared": not blockers,
            "execute_mode_required_before_lease_issue": True,
            "lease_issued": False,
            "child_spawned": False,
            "current_no_lease_outer_preflight": _evidence(context.outer_preflight)
            if context
            else {"present": False},
            "production_binary": _evidence(context.production_binary)
            if context
            else {"present": False},
            "production_resource": _evidence(context.production_resource)
            if context
            else {"present": False},
            "production_authority": _evidence(context.production_authority)
            if context
            else {"present": False},
            "future_one_scan_plan": _future_one_scan_plan(),
            "blockers": blockers,
            "execution_boundary": {
                "source_root_opened_or_statted": False,
                "source_payload_opened": False,
                "source_teacher_or_logits_executed": False,
                "native_phase_started": False,
                "gpu_server_hcli_or_tps_action": False,
                "lease_issued_or_consumed": False,
                "launch_replay_or_capture_directory_created": False,
                "child_spawned": False,
                "child_reaped": False,
                "terminal_receipt_written": False,
                "release_receipt_written": False,
            },
            "claim_boundary": "CPU/file-only issuer/outer preflight. It does not create a lease or directory, invoke a child, open a source root/payload, run source teacher/native/model/GPU/server/HCLI/TPS work, or claim a scan result.",
        }
    )


def _future_one_scan_plan() -> dict[str, Any]:
    return {
        "lease": {"schema": outer.LEASE_SCHEMA, "status": outer.LEASE_STATUS, "one_shot": True},
        "resources": {
            "source_shards": outer.SOURCE_SHARDS,
            "source_tensors": outer.SOURCE_TENSORS,
            "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
            "maximum_live_raw_bf16_windows": 1,
            "source_teacher_native_gpu_server_hcli_forbidden": True,
        },
        "directories": {
            "new_launch_directory": True,
            "new_replay_directory": True,
            "new_outer_capture_directory": True,
            "new_child_capture_directory_created_by_bound_scanner": True,
        },
        "exact_child_command": [
            "BOUND_PRODUCTION_SCANNER_BINARY",
            "--mode",
            "production-scan",
            "--range-authority",
            "AUTHORITY_BOUND_METADATA_JSON",
            "--semantics-attester",
            "AUTHORITY_BOUND_SEMANTICS_JSON",
            "--runtime-admission-authority",
            "AUTHORITY_BOUND_RUNTIME_PREFLIGHT_JSON",
            "--interface-authority",
            "SEALED_PRODUCTION_INTERFACE_JSON",
            "--production-scan-authority",
            "SEALED_PRODUCTION_AUTHORITY_JSON",
            "--bootstrap-lease",
            "NEW_ONE_SHOT_LEASE_JSON",
            "--source-root",
            "EXPLICIT_FUTURE_Q30_SOURCE_ROOT",
            "--capture-dir",
            "NEW_CHILD_CAPTURE_DIRECTORY",
            "--out",
            "NEW_CHILD_RECEIPT_JSON",
        ],
        "terminal_order": [
            "create-new replay reservation",
            "issue fresh lease in execute mode",
            "invoke exactly one bound scanner child",
            "reap child",
            "validate child receipt if successful",
            "write outer terminal receipt last",
            "write one-shot release receipt after terminal",
        ],
    }


def _authority_bound_raw_path(context: LeaseContext, *, field: str) -> Path:
    bindings = _mapping(context.production_authority.document.get("immutable_bindings"), label="authority bindings")
    evidence = _mapping(bindings.get(field), label=f"authority {field}")
    raw_path = Path(_text(evidence.get("path"), label=f"authority {field}.path"))
    if not raw_path.is_absolute():
        raise ProductionLeaseOuterError(f"authority {field} path must be absolute")
    return raw_path


def _validate_authority_bound_command_inputs(context: LeaseContext) -> tuple[Path, Path, Path]:
    """Verify metadata identities before an explicit future source-root check."""
    bindings = _mapping(context.production_authority.document.get("immutable_bindings"), label="authority bindings")
    paths = (
        ("metadata_range_authority", False),
        ("independent_semantics_attester", False),
        ("runtime_admission_producer_authority", True),
    )
    resolved: list[Path] = []
    for field, sealed in paths:
        evidence = _mapping(bindings.get(field), label=f"authority {field}")
        path = _authority_bound_raw_path(context, field=field)
        try:
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise OSError("not a regular non-symlink file")
            raw = path.read_bytes()
            parsed = json.loads(raw)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProductionLeaseOuterError(f"cannot validate authority-bound {field}: {exc}") from exc
        if _sha256_bytes(raw) != evidence.get("raw_document_sha256"):
            raise ProductionLeaseOuterError(f"authority-bound {field} raw SHA drifted")
        if _sha256_bytes(_canonical_json(parsed)) != evidence.get("canonical_document_sha256"):
            raise ProductionLeaseOuterError(f"authority-bound {field} canonical SHA drifted")
        if sealed:
            try:
                document = outer._sealed(path, label=f"authority-bound {field}")
            except outer.ProductionScanOuterError as exc:
                raise ProductionLeaseOuterError(str(exc)) from exc
            if document.seal_sha256 != evidence.get("seal_sha256"):
                raise ProductionLeaseOuterError(f"authority-bound {field} seal drifted")
        elif evidence.get("seal_sha256") is not None:
            raise ProductionLeaseOuterError(f"authority-bound {field} unexpectedly claims a seal")
        resolved.append(path.resolve(strict=True))
    return resolved[0], resolved[1], resolved[2]


def _validate_explicit_source_root(path: Path) -> Path:
    if not path.is_absolute():
        raise ProductionLeaseOuterError("--source-root must be absolute in execute mode")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductionLeaseOuterError(f"cannot stat explicit source root: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ProductionLeaseOuterError("explicit source root must be a regular non-symlink directory")
    return path.resolve(strict=True)


def _child_command(
    context: LeaseContext,
    *,
    lease_path: Path,
    child_capture_dir: Path,
    child_receipt_path: Path,
    source_root: Path,
) -> list[str]:
    metadata, semantics, runtime = _validate_authority_bound_command_inputs(context)
    return [
        _text(
            _mapping(context.production_binary.document.get("executable"), label="production binary executable").get("path"),
            label="production executable path",
        ),
        "--mode",
        "production-scan",
        "--range-authority",
        str(metadata),
        "--semantics-attester",
        str(semantics),
        "--runtime-admission-authority",
        str(runtime),
        "--interface-authority",
        str(context.production_interface.path),
        "--production-scan-authority",
        str(context.production_authority.path),
        "--bootstrap-lease",
        str(lease_path),
        "--source-root",
        str(source_root),
        "--capture-dir",
        str(child_capture_dir),
        "--out",
        str(child_receipt_path),
    ]


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


def _file_evidence(path: Path, *, label: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductionLeaseOuterError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProductionLeaseOuterError(f"{label} must be a regular non-symlink file")
    if metadata.st_size > MAX_CHILD_STREAM_BYTES:
        raise ProductionLeaseOuterError(f"{label} exceeds {MAX_CHILD_STREAM_BYTES} bytes")
    return {"path": str(path), "bytes": metadata.st_size, "sha256": _sha256_bytes(path.read_bytes())}


def _validate_child_capture(path: Path, *, context: LeaseContext, lease: outer.Document) -> outer.Document:
    try:
        capture = outer._sealed(path, label="production scanner child receipt")
    except outer.ProductionScanOuterError as exc:
        raise ProductionLeaseOuterError(str(exc)) from exc
    root = capture.document
    if root.get("schema") != outer.CAPTURE_SCHEMA or root.get("status") != outer.CAPTURE_STATUS:
        raise ProductionLeaseOuterError("child receipt schema/status drifted")
    for field in (
        "production_hash_scan_earned",
        "receipt_written_last",
        "source_handles_closed",
        "reader_cache_zeroed",
    ):
        if root.get(field) is not True:
            raise ProductionLeaseOuterError(f"child receipt.{field} must be true")
    for field in (
        "fixture_only",
        "synthetic_fixture_only",
        "production_adapter_forbidden",
        "source_teacher_or_logits_executed",
        "operator_or_reader_execution_attestation_emitted",
        "source_teacher_runtime_admission_earned",
        "model_gpu_server_hcli_or_tps_action",
    ):
        if root.get(field) is not False:
            raise ProductionLeaseOuterError(f"child receipt.{field} must be false")
    geometry = _mapping(root.get("geometry"), label="child receipt geometry")
    if (
        _integer(geometry.get("source_shards"), label="child shards") != outer.SOURCE_SHARDS
        or _integer(geometry.get("source_tensors"), label="child tensors") != outer.SOURCE_TENSORS
        or _integer(geometry.get("maximum_positioned_read_bytes"), label="child maximum read")
        != outer.MAX_POSITIONED_READ_BYTES
        or _integer(geometry.get("maximum_live_raw_bf16_windows"), label="child live windows") != 1
    ):
        raise ProductionLeaseOuterError("child receipt geometry drifted")
    _pointer(
        root.get("interface_authority"), expected=context.production_interface, label="child interface"
    )
    _pointer(
        root.get("production_scan_authority"),
        expected=context.production_authority,
        label="child production authority",
    )
    _pointer(root.get("fresh_bootstrap_lease"), expected=lease, label="child fresh lease")
    return capture


def _outer_terminal_document(
    *,
    context: LeaseContext,
    lease: outer.Document,
    command: list[str],
    terminal: Mapping[str, Any],
    child_pid: int | None,
    child_capture: outer.Document | None,
    child_capture_error: str | None,
    stdout: Mapping[str, Any],
    stderr: Mapping[str, Any],
) -> dict[str, Any]:
    successful = (
        terminal.get("reaped") is True
        and terminal.get("timed_out") is False
        and terminal.get("exit_code") == 0
        and child_capture is not None
        and child_capture_error is None
    )
    return seal(
        {
            "schema": outer.OUTER_TERMINAL_SCHEMA,
            "status": outer.OUTER_TERMINAL_STATUS if successful else TERMINAL_REFUSED_STATUS,
            "recorded_at": _utc_now(),
            "child_reaped": terminal.get("reaped") is True,
            "terminal_receipt_written_after_child_capture": child_capture is not None,
            "terminal_receipt_written_last": True,
            "automatic_retry_disabled": True,
            "lease_reuse_prohibited": True,
            "child_timed_out": terminal.get("timed_out") is True,
            "child_exit_code": terminal.get("exit_code"),
            "child_signal": terminal.get("signal"),
            "child_spawn_error": terminal.get("spawn_error"),
            "child_pid": child_pid,
            "lease_id": lease.document["lease_id"],
            "production_binary_sha256": context.production_binary_sha256,
            "production_authority_seal_sha256": context.production_authority.seal_sha256,
            "child_capture_seal_sha256": child_capture.seal_sha256 if child_capture else None,
            "child_capture": _evidence(child_capture) if child_capture else {"present": False},
            "child_capture_validation_error": child_capture_error,
            "issued_lease": _evidence(lease),
            "production_authority": _evidence(context.production_authority),
            "command": command,
            "stdout": dict(stdout),
            "stderr": dict(stderr),
            "claim_boundary": "Outer terminal for one production hash-scan child only. A success is a hash-map component boundary, never source-teacher semantics/logits, runtime admission, native/model/GPU/server/HCLI/TPS/TG/tournament evidence.",
        }
    )


def _release_document(
    *, lease: outer.Document, terminal: Mapping[str, Any], child_capture: outer.Document | None
) -> dict[str, Any]:
    return seal(
        {
            "schema": outer.RELEASE_SCHEMA,
            "status": outer.RELEASE_STATUS,
            "recorded_at": _utc_now(),
            "release_after_outer_terminal": True,
            "one_shot_lease_finalized": True,
            "retry_or_relaunch_forbidden": True,
            "source_teacher_or_logits_authorized": False,
            "native_or_gpu_server_hcli_authorized": False,
            "artifacts_deleted_or_evicted": False,
            "lease_id": lease.document["lease_id"],
            "outer_terminal_seal_sha256": terminal["seal_sha256"],
            "child_capture_seal_sha256": child_capture.seal_sha256 if child_capture else None,
            "outer_terminal_status": terminal["status"],
            "claim_boundary": "Receipt-last quiet release for one consumed production hash-scan lease only. It cannot authorize retry, source teacher, native/model/GPU/server/HCLI, TPS, TG, or tournament work.",
        }
    )


def _create_replay_reservation(context: LeaseContext, *, replay_dir: Path, lease: outer.Document) -> Path:
    replay = replay_dir / REPLAY_FILENAME
    _write_new(
        replay,
        seal(
            {
                "schema": outer.REPLAY_SCHEMA,
                "status": outer.REPLAY_STATUS,
                "recorded_at": _utc_now(),
                "create_new_before_child": True,
                "one_child_maximum": True,
                "replay_or_relaunch_forbidden": True,
                "attempt": 1,
                "lease_id": lease.document["lease_id"],
                "baseline_outer_preflight": _evidence(context.outer_preflight),
                "production_authority": _evidence(context.production_authority),
                "claim_boundary": "Create-new one-shot replay guard only; no source root, child, or hash scan has run.",
            }
        ),
    )
    return replay


def _run_one_child(
    *,
    command: list[str],
    capture_dir: Path,
    timeout_seconds: float,
) -> tuple[dict[str, Any], int | None, dict[str, Any], dict[str, Any]]:
    stdout_path = capture_dir / STDOUT_FILENAME
    stderr_path = capture_dir / STDERR_FILENAME
    child_pid: int | None = None
    with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
        try:
            child = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
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
    try:
        stdout = _file_evidence(stdout_path, label="child stdout")
        stderr = _file_evidence(stderr_path, label="child stderr")
    except ProductionLeaseOuterError as exc:
        stdout = {"present": False, "error": str(exc)}
        stderr = {"present": False, "error": str(exc)}
    return terminal, child_pid, stdout, stderr


def _reject_forbidden_command(command: Sequence[str]) -> None:
    if not command:
        raise ProductionLeaseOuterError("child command must be non-empty")
    lowered = " ".join(command).lower()
    if any(marker in lowered for marker in ("source-teacher", "native", "gpu", "metal", "hcli")):
        raise ProductionLeaseOuterError("child command crosses a forbidden teacher/native/GPU/HCLI boundary")


def _run_attempt(
    config: ExecuteConfig,
    *,
    enable_real_source_scan: bool = False,
    fake_child_command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Issue exactly one lease and reap exactly one child.

    ``fake_child_command`` exists only for focused unit tests.  It is mutually
    exclusive with real execution and its result is always terminally refused,
    so synthetic data cannot create a production-earned claim.
    """
    fake = fake_child_command is not None
    if fake and enable_real_source_scan:
        raise ProductionLeaseOuterError("fake child and real source execution are mutually exclusive")
    if not fake and not enable_real_source_scan:
        raise ProductionLeaseOuterError("execute mode requires explicit real-source-scan enablement")
    fake_command = list(fake_child_command or ())
    if fake:
        _reject_forbidden_command(fake_command)
    context = _load_context(config)
    launch, replay, capture = _validate_new_layout(config)

    lease_path = launch / LEASE_FILENAME
    issued_outer_path = launch / ISSUED_OUTER_FILENAME
    release_path = launch / RELEASE_FILENAME
    child_capture_dir = capture / CHILD_CAPTURE_DIRNAME
    child_receipt_path = child_capture_dir / CHILD_RECEIPT_FILENAME

    source_root: Path | None = None
    if not fake:
        if config.source_root is None:
            raise ProductionLeaseOuterError("execute mode requires an explicit source root")
        source_root = _validate_explicit_source_root(config.source_root)

    # Validate every authority-bound command input before issuing the lease.
    # If anything has drifted, the controller stops with no launch/replay/
    # capture directory and no lease to finalize.
    if fake:
        command = fake_command
    else:
        assert source_root is not None
        command = _child_command(
            context,
            lease_path=lease_path,
            child_capture_dir=child_capture_dir,
            child_receipt_path=child_receipt_path,
            source_root=source_root,
        )
    _reject_forbidden_command(command)

    launch = _mkdir_new(launch, label="launch directory")
    replay = _mkdir_new(replay, label="replay directory")
    capture = _mkdir_new(capture, label="outer capture directory")

    lease_id = _sha256_bytes(os.urandom(32))
    _write_new(lease_path, _lease_document(context, lease_id=lease_id))
    lease = _validate_issued_lease(lease_path, context)
    issued_outer: dict[str, Any] | None = None
    replay_path: Path | None = None
    terminal_state: dict[str, Any]
    child_pid: int | None = None
    stdout: dict[str, Any] = {"present": False}
    stderr: dict[str, Any] = {"present": False}
    child_capture: outer.Document | None = None
    child_capture_error: str | None = None
    try:
        issued_outer = _issued_outer_document(context, lease_path=lease_path)
        _write_new(issued_outer_path, issued_outer)
        replay_path = _create_replay_reservation(context, replay_dir=replay, lease=lease)
        _write_new(
            capture / RUNNING_FILENAME,
            seal(
                {
                    "schema": SCHEMA,
                    "status": STARTED_STATUS,
                    "recorded_at": _utc_now(),
                    "lease_id": lease.document["lease_id"],
                    "replay_reservation": str(replay_path),
                    "issued_outer_preflight": _evidence(
                        outer._sealed(issued_outer_path, label="issued outer preflight")
                    ),
                    "command": command,
                    "fake_child_test_only": fake,
                    "claim_boundary": "Exactly one one-shot production hash-scan child may be reaped; teacher/native/GPU/server/HCLI work is forbidden.",
                }
            ),
        )
        # Close the metadata/binary TOCTOU window immediately before spawning
        # the sole child.  This remains receipt-only validation; the scanner
        # still performs its own validation before it can read source bytes.
        refreshed_context = _load_context(config)
        if (
            refreshed_context.production_binary.seal_sha256
            != context.production_binary.seal_sha256
            or refreshed_context.production_resource.seal_sha256
            != context.production_resource.seal_sha256
            or refreshed_context.production_authority.seal_sha256
            != context.production_authority.seal_sha256
        ):
            raise ProductionLeaseOuterError("production authority chain changed after lease issue")
        _validate_issued_lease(lease.path, refreshed_context)
        terminal_state, child_pid, stdout, stderr = _run_one_child(
            command=command, capture_dir=capture, timeout_seconds=config.timeout_seconds
        )
        if fake:
            child_capture_error = "test-only fake child is never accepted as production evidence"
        elif terminal_state.get("exit_code") == 0 and terminal_state.get("timed_out") is False:
            try:
                child_capture = _validate_child_capture(
                    child_receipt_path, context=context, lease=lease
                )
            except ProductionLeaseOuterError as exc:
                child_capture_error = str(exc)
        else:
            child_capture_error = "child did not exit successfully"
    except (ProductionLeaseOuterError, OSError) as exc:
        terminal_state = _terminal(
            None, timed_out=False, spawn_error=f"{type(exc).__name__}: {exc}"
        )
        child_capture_error = f"outer failed after lease issue: {exc}"

    terminal = _outer_terminal_document(
        context=context,
        lease=lease,
        command=command,
        terminal=terminal_state,
        child_pid=child_pid,
        child_capture=child_capture,
        child_capture_error=child_capture_error,
        stdout=stdout,
        stderr=stderr,
    )
    terminal_path = capture / TERMINAL_FILENAME
    _write_new(terminal_path, terminal)
    release = _release_document(lease=lease, terminal=terminal, child_capture=child_capture)
    _write_new(release_path, release)
    return {
        "lease": lease.document,
        "lease_path": lease.path,
        "issued_outer_preflight": issued_outer,
        "replay_reservation_path": replay_path,
        "outer_terminal": terminal,
        "outer_terminal_path": terminal_path,
        "release": release,
        "release_path": release_path,
    }


def run_execute(
    config: ExecuteConfig, *, enable_real_source_scan: bool = False
) -> dict[str, Any]:
    """Execute the real scanner path only after an explicit caller opt-in."""
    return _run_attempt(config, enable_real_source_scan=enable_real_source_scan)


def run_fake_child_test(
    config: ExecuteConfig, *, fake_child_command: Sequence[str]
) -> dict[str, Any]:
    """Test-only lifecycle seam; terminally refuses every fake-child result."""
    return _run_attempt(config, fake_child_command=fake_child_command)


def _config_from_args(args: argparse.Namespace) -> ExecuteConfig:
    return ExecuteConfig(
        outer_preflight_path=args.outer_preflight,
        bootstrap_preflight_path=args.bootstrap_preflight,
        bootstrap_binary_path=args.bootstrap_binary,
        bootstrap_resource_path=args.bootstrap_resource,
        production_binary_path=args.production_binary,
        production_resource_path=args.production_resource,
        production_interface_path=args.production_interface,
        production_authority_path=args.production_authority,
        launch_dir=args.launch_dir,
        replay_dir=args.replay_dir,
        capture_dir=args.capture_dir,
        source_root=args.source_root,
        timeout_seconds=args.timeout_seconds,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "execute"), default="preflight")
    parser.add_argument("--outer-preflight", type=Path, required=True)
    parser.add_argument("--bootstrap-preflight", type=Path, required=True)
    parser.add_argument("--bootstrap-binary", type=Path, required=True)
    parser.add_argument("--bootstrap-resource", type=Path, required=True)
    parser.add_argument("--production-binary", type=Path, required=True)
    parser.add_argument("--production-resource", type=Path, required=True)
    parser.add_argument("--production-interface", type=Path, required=True)
    parser.add_argument("--production-authority", type=Path, required=True)
    parser.add_argument("--launch-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--enable-real-production-hash-scan", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _config_from_args(args)
    try:
        if args.mode == "preflight":
            document = build_issuer_preflight(config)
            _write_new(args.out, document)
        else:
            if not args.enable_real_production_hash_scan:
                raise ProductionLeaseOuterError(
                    "--mode execute requires --enable-real-production-hash-scan"
                )
            result = run_execute(config, enable_real_source_scan=True)
            document = seal(
                {
                    "schema": SCHEMA,
                    "status": result["outer_terminal"]["status"],
                    "outer_terminal_path": str(result["outer_terminal_path"]),
                    "outer_terminal_seal_sha256": result["outer_terminal"]["seal_sha256"],
                    "release_path": str(result["release_path"]),
                    "release_seal_sha256": result["release"]["seal_sha256"],
                    "claim_boundary": "Pointer to a separately sealed one-shot outer terminal/release only.",
                }
            )
            _write_new(args.out, document)
    except ProductionLeaseOuterError as exc:
        print(f"Q30 production hash-scan lease/outer refused: {exc}")
        return 2
    print(
        json.dumps(
            {"output": str(args.out.resolve()), "status": document["status"], "seal_sha256": document["seal_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
