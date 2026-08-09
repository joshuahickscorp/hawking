"""One-shot outer launcher for Qwen30 HQ30GR2 sparse gate/up parity.

This module is deliberately an evidence and process-lifecycle boundary.  Its
``cpu-oracle`` mode starts the already-built non-serving child once and seals
the exact F64 output.  Its ``device-parity`` mode can start the same pinned
child only after that CPU terminal record is byte-bound and a fresh,
component-only quiet lease is supplied.  The launcher contains no Metal API,
runtime/server/watcher control, or retry scheduler.

An outer success is only an L0/E0 gate+up+SwiGLU component observation.  It
does not claim an all-layer forward, logits, HCLI/coherence, TPS/TG,
capability, manager, or tournament result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "hawking.ascension.qwen30_hq30gr2_sparse_gate_up_device_parity_outer_launcher.v1"
ACTIVE_FILENAME = "active.json"
CHILD_FILENAME = "child.json"
TERMINAL_FILENAME = "outer-terminal-receipt.json"
OUTER_STDOUT = "outer.stdout.log"
OUTER_STDERR = "outer.stderr.log"
INNER_CAPTURE = "inner"

MODE_CPU = "cpu-oracle"
MODE_DEVICE = "device-parity"
EXPECTED_PROBE_BASENAME = "ascension_qwen30_hq30gr2_sparse_gate_up_device_parity"
EXPECTED_PROBE_BINARY_SHA256 = "42569eecd6d4abaf081ac83e8d98adaa02d62128323bb2ed8c5f1a0471ae82c3"
EXPECTED_INNER_SCHEMA = "hawking.ascension.qwen30_hq30gr2_sparse_gate_up_device_parity.v1"
EXPECTED_CPU_STATUS = "EARNED_HQ30GR2_CPU_FORMAT_ORACLE_NOT_DEVICE_OR_RUNTIME"
EXPECTED_DEVICE_STATUS = "EARNED_HQ30GR2_SPARSE_GATE_UP_CPU_DEVICE_PARITY_NOT_LAYER_OR_RUNTIME"

CANDIDATE_SCHEMA = "hawking.ascension.qwen30_quality_repack_candidate.v1"
CANDIDATE_STATUS = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED"
ADMISSION_CURRENT_SCHEMA = "hawking.ascension.qwen30_quality_repack_native_admission_current_pointer.v1"
ADMISSION_CURRENT_STATUS = "CURRENT_QUALITY_REPACK_NATIVE_ADMISSION_RECEIPT_SELECTED"
ADMISSION_SCHEMA = "hawking.ascension.qwen30_quality_repack_native_admission_receipt.v1"
ADMISSION_STATUS = "EARNED_QUALITY_REPACK_COMPLETE_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
REVALIDATION_SCHEMA = "hawking.ascension.complete_binary_source_revalidation.v1"
REVALIDATION_STATUS = "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED"
SELECTION_SCHEMA = "hawking.ascension.qwen30_quality_repack_selection.v1"
SELECTION_STATUS = "EARNED_SOURCE_BOUND_QUALITY_REPACK_SELECTION_UNQUALIFIED"
SNAPSHOT_SCHEMA = "hawking.ascension.qwen30_quality_repack_source_snapshot.v1"
SNAPSHOT_STATUS = "EARNED_IMMUTABLE_SOURCE_AND_ROLLBACK_BINDING"
CANDIDATE_TERMINAL_SCHEMA = "hawking.ascension.complete_binary_terminal_status.v1"
CANDIDATE_TERMINAL_STATUS = "EARNED_COMPLETE_PHYSICAL_BINARY_CANDIDATE_UNQUALIFIED"

COMPILER_CURRENT_SCHEMA = "hawking.ascension.qwen30_quality_repack_current_hcli_compiler_trace_current.v1"
COMPILER_CURRENT_STATUS = "CURRENT_NEW_DIAGNOSTIC_NOT_HISTORICAL_HCLI_COMPILER_TRACE_SELECTED"
COMPILER_SCHEMA = "hawking.ascension.qwen30_quality_repack_current_hcli_compiler_trace.v1"
COMPILER_STATUS = "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_PRE_EXECUTION_HCLI_COMPILER_TRACE"
ANNOTATED_TRACE_SCHEMA = "hawking.ascension.qwen30_hcli_compiler_pre_execution_trace_annotated.v1"
ANNOTATED_TRACE_STATUS = "NEW_DIAGNOSTIC_NOT_HISTORICAL"
ROUTE_CURRENT_SCHEMA = "hawking.ascension.qwen30_quality_repack_current_hcli_route_capture_current.v1"
ROUTE_CURRENT_STATUS = "CURRENT_NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_CAPTURE_SELECTED"
ROUTE_SCHEMA = "hawking.ascension.qwen30_quality_repack_current_hcli_route_capture.v1"
ROUTE_STATUS = "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_L0_ROUTE_AND_HIDDEN_CAPTURE_UNQUALIFIED"

PINNED_CANDIDATE_MANIFEST_SEAL = "883c59eec0371ebb6d4a9935cdbdc6bcb486c03eebd5312db608a0415a34911f"
PINNED_ADMISSION_RECEIPT_SEAL = "d7645a66d9c682bd4b0b1c0fb7fef86276678f7270862fc92db65e4d4a92c73b"
PINNED_REVALIDATION_SEAL = "ac7208e11c31bbd035bd87fd62a80020b9d1d05970867576f4649f6bebe68123"
PINNED_SELECTION_SEAL = "76fb3bba71a33012a4676cf039a4c75e82457dde85de454dd4ae5c14d91bfccb"
PINNED_SOURCE_SNAPSHOT_SEAL = "d358045c2cb76adedd9f56433e0cf3a9d668ae9216a2516236939dedab6e83ad"
PINNED_COMPILER_TRACE_SEAL = "e698ebc2d405c70a2f6a2df39deaff800efefa9470a8f3efed644855da43a87a"
PINNED_ROUTE_CAPTURE_SEAL = "e60de4072af92f8ecc56b6a9353a2c6ae077fb4ffd6cb8939b9df5f9360feeca"
PREPARATION_CURRENT_SCHEMA = "hawking.ascension.qwen30_hq30gr2_sparse_gate_up_component_prepare_current.v1"
PREPARATION_CURRENT_STATUS = "CURRENT_PREPARED_HQ30GR2_LITERAL_HAWKING_L0_E0_CPU_ORACLE_INPUT_SELECTED"
PREPARATION_SCHEMA = "hawking.ascension.qwen30_hq30gr2_sparse_gate_up_component_prepare.v1"
PREPARATION_STATUS = "PREPARED_HQ30GR2_LITERAL_HAWKING_L0_E0_CPU_ORACLE_INPUT_NOT_RUN"
PINNED_PREPARATION_RECEIPT_SEAL = "10d0e20bb6c447a402ae7b9fd19db579e23b35720084004e53623af1345e441c"

QUIET_LEASE_SCHEMA = "hawking.ascension.qwen30_hq30gr2_sparse_gate_up_device_parity_quiet_lease.v1"
QUIET_LEASE_STATUS = "GRANTED_QWEN30_HQ30GR2_SPARSE_GATE_UP_DEVICE_PARITY_COMPONENT_ONLY_NON_TIMED_LEASE"
QUIET_LEASE_COMPONENT = "qwen30_hq30gr2_sparse_gate_up_device_parity"

TARGET_PROBE = "literal_hawking"
TARGET_POSITION = 337
TARGET_TOKEN_COUNT = 369
INPUT_VALUES = 2048
INPUT_BYTES = INPUT_VALUES * 4
OUTPUT_VALUES = 768
CPU_OUTPUT_BYTES = OUTPUT_VALUES * 8
DEVICE_OUTPUT_BYTES = OUTPUT_VALUES * 4
EXPECTED_SELECTED_ORGANS = (
    "model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.layers.0.mlp.experts.0.up_proj.weight",
)
EXPECTED_KERNEL = "qwen30_quality_repack_sparse_gate_up_swiglu"
EXPECTED_GROUP_SIZE = 128
EXPECTED_RESIDUAL_COUNT = 3933
EXPECTED_VERIFIED_PAYLOADS = 18867
EXPECTED_DIRECT_TENSORS = 18865


class SparseGateUpParityLauncherError(RuntimeError):
    """The isolated parity component cannot safely start."""


@dataclass(frozen=True)
class LaunchConfig:
    probe_bin: Path
    candidate_manifest: Path
    candidate_admission_current: Path
    source_revalidation: Path
    selection_receipt: Path
    source_snapshot: Path
    compiler_trace_current: Path
    route_capture_current: Path
    preparation_current: Path
    capture_dir: Path
    mode: str
    timeout_seconds: float
    cpu_oracle_outer_receipt: Path | None = None
    lease_receipt: Path | None = None


@dataclass(frozen=True)
class CandidateContext:
    manifest: dict[str, Any]
    manifest_seal: str
    admission_current: dict[str, Any]
    admission_current_seal: str
    admission_receipt: dict[str, Any]
    admission_receipt_seal: str
    source_revalidation: dict[str, Any]
    source_revalidation_seal: str
    selection: dict[str, Any]
    selection_seal: str
    source_snapshot: dict[str, Any]
    source_snapshot_seal: str
    candidate_terminal: dict[str, Any]
    candidate_terminal_seal: str
    source_revision: str
    source_audit_seal: str


@dataclass(frozen=True)
class TraceContext:
    compiler_current: dict[str, Any]
    compiler_current_seal: str
    compiler_receipt: dict[str, Any]
    compiler_receipt_seal: str
    route_current: dict[str, Any]
    route_current_seal: str
    route_receipt: dict[str, Any]
    route_receipt_seal: str
    input_f32le: dict[str, Any]


@dataclass(frozen=True)
class PreparationContext:
    current: dict[str, Any]
    current_seal: str
    receipt: dict[str, Any]
    receipt_seal: str
    cpu_command_without_output_dir: tuple[str, ...]


@dataclass(frozen=True)
class CpuOracleContext:
    outer_receipt: dict[str, Any]
    outer_receipt_seal: str
    cpu_activation: dict[str, Any]


@dataclass(frozen=True)
class LaunchContext:
    probe_binary: dict[str, Any]
    candidate: CandidateContext
    trace: TraceContext
    preparation: PreparationContext
    cpu_oracle: CpuOracleContext | None
    lease: dict[str, Any] | None
    lease_seal: str | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SparseGateUpParityLauncherError(f"{label} must be a non-empty string")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SparseGateUpParityLauncherError(f"{label} must be an object")
    return dict(value)


def _require_absolute(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise SparseGateUpParityLauncherError(f"{label} must be absolute: {path}")


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    _require_absolute(path, label)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SparseGateUpParityLauncherError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SparseGateUpParityLauncherError(f"{label} must be a regular non-symlink file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise SparseGateUpParityLauncherError(f"{label} must be executable: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise SparseGateUpParityLauncherError(f"cannot canonicalize {label} {path}: {exc}") from exc


def _canonical_from_document(value: object, label: str) -> Path:
    return _canonical_regular(Path(_text(value, label)), label)


def _file_evidence(path: Path, label: str, *, executable: bool = False) -> dict[str, Any]:
    canonical = _canonical_regular(path, label, executable=executable)
    return {
        "path": str(canonical),
        "present": True,
        "bytes": canonical.stat().st_size,
        "sha256": _file_sha256(canonical),
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SparseGateUpParityLauncherError(f"cannot read JSON {label} at {path}: {exc}") from exc
    return _mapping(value, f"JSON {label} at {path}")


def _sealed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    canonical = _canonical_regular(path, label)
    document = _read_json(canonical, label)
    try:
        verify(document, label=str(canonical))
    except ValueError as exc:
        raise SparseGateUpParityLauncherError(f"{label} is not a valid sealed receipt: {exc}") from exc
    receipt_seal = document.get("seal_sha256")
    if not _is_sha256(receipt_seal):
        raise SparseGateUpParityLauncherError(f"{label} lacks a lowercase SHA-256 seal")
    return document, str(receipt_seal)


def _assert_pinned(value: object, expected: str, label: str) -> None:
    if value != expected:
        raise SparseGateUpParityLauncherError(f"{label} is not the pinned HQ30GR2 evidence")


def _assert_file_reference(reference: object, evidence: Mapping[str, Any], label: str) -> None:
    row = _mapping(reference, label)
    if _canonical_from_document(row.get("path"), f"{label}.path") != Path(str(evidence["path"])):
        raise SparseGateUpParityLauncherError(f"{label} path drifted")
    document_sha = row.get("document_sha256")
    if document_sha is None:
        document_sha = row.get("sha256")
    if document_sha != evidence["sha256"]:
        raise SparseGateUpParityLauncherError(f"{label} document SHA-256 drifted")


def _assert_sealed_reference(
    reference: object, evidence: Mapping[str, Any], receipt_seal: str, label: str
) -> None:
    _assert_file_reference(reference, evidence, label)
    if _mapping(reference, label).get("seal_sha256") != receipt_seal:
        raise SparseGateUpParityLauncherError(f"{label} receipt seal drifted")


def _safe_relative_path(value: object, label: str) -> Path:
    raw = _text(value, label)
    candidate = PurePath(raw)
    if candidate.is_absolute() or ".." in candidate.parts or raw in {"", "."}:
        raise SparseGateUpParityLauncherError(f"{label} must be a contained relative path")
    return Path(candidate)


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Create durable terminal evidence without replacing prior evidence."""

    if path.exists():
        raise SparseGateUpParityLauncherError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as exc:
        raise SparseGateUpParityLauncherError(f"refusing to overwrite {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _bind_candidate(config: LaunchConfig) -> CandidateContext:
    """Require the entire admitted HQ30GR2 source/candidate graph up front."""

    manifest, manifest_seal = _sealed_json(config.candidate_manifest, "--candidate-manifest")
    if manifest.get("schema") != CANDIDATE_SCHEMA or manifest.get("status") != CANDIDATE_STATUS:
        raise SparseGateUpParityLauncherError("--candidate-manifest schema/status drifted")
    _assert_pinned(manifest_seal, PINNED_CANDIDATE_MANIFEST_SEAL, "candidate manifest")
    manifest_evidence = _file_evidence(config.candidate_manifest, "--candidate-manifest")
    representation = _mapping(manifest.get("representation"), "candidate representation")
    if tuple(representation.get("selected_organs", ())) != EXPECTED_SELECTED_ORGANS:
        raise SparseGateUpParityLauncherError("candidate does not select exactly L0/E0 gate/up")
    branch = _mapping(manifest.get("quality_repack_branch"), "candidate quality repack branch")
    if tuple(branch.get("changed_organs", ())) != EXPECTED_SELECTED_ORGANS:
        raise SparseGateUpParityLauncherError("candidate changed-organs boundary drifted")

    revalidation, revalidation_seal = _sealed_json(config.source_revalidation, "--source-revalidation")
    if revalidation.get("schema") != REVALIDATION_SCHEMA or revalidation.get("status") != REVALIDATION_STATUS:
        raise SparseGateUpParityLauncherError("--source-revalidation schema/status drifted")
    _assert_pinned(revalidation_seal, PINNED_REVALIDATION_SEAL, "source revalidation")
    revalidation_evidence = _file_evidence(config.source_revalidation, "--source-revalidation")
    source_revision = _text(revalidation.get("source_revision"), "source revalidation source_revision")
    source_audit_seal = revalidation.get("source_audit_seal_sha256")
    if not _is_sha256(source_audit_seal):
        raise SparseGateUpParityLauncherError("source revalidation source audit seal is absent")
    if manifest.get("source_revalidation_receipt_seal_sha256") != revalidation_seal:
        raise SparseGateUpParityLauncherError("candidate source revalidation seal drifted")
    if _canonical_from_document(
        manifest.get("source_revalidation_receipt_path"), "candidate source revalidation path"
    ) != Path(revalidation_evidence["path"]):
        raise SparseGateUpParityLauncherError("candidate source revalidation path drifted")
    if manifest.get("source_body_audit_seal_sha256") != source_audit_seal:
        raise SparseGateUpParityLauncherError("candidate source-audit seal drifted")

    selection, selection_seal = _sealed_json(config.selection_receipt, "--selection-receipt")
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("status") != SELECTION_STATUS:
        raise SparseGateUpParityLauncherError("--selection-receipt schema/status drifted")
    _assert_pinned(selection_seal, PINNED_SELECTION_SEAL, "selection receipt")
    selection_evidence = _file_evidence(config.selection_receipt, "--selection-receipt")
    _assert_sealed_reference(branch.get("selection_receipt"), selection_evidence, selection_seal, "candidate selection")

    snapshot, snapshot_seal = _sealed_json(config.source_snapshot, "--source-snapshot")
    if snapshot.get("schema") != SNAPSHOT_SCHEMA or snapshot.get("status") != SNAPSHOT_STATUS:
        raise SparseGateUpParityLauncherError("--source-snapshot schema/status drifted")
    _assert_pinned(snapshot_seal, PINNED_SOURCE_SNAPSHOT_SEAL, "source snapshot")
    snapshot_evidence = _file_evidence(config.source_snapshot, "--source-snapshot")
    _assert_sealed_reference(branch.get("source_binding_snapshot"), snapshot_evidence, snapshot_seal, "candidate source snapshot")

    for label, document in (("selection", selection), ("source snapshot", snapshot)):
        binding = _mapping(document.get("binding"), f"{label} binding")
        if tuple(binding.get("selected_organs", ())) != EXPECTED_SELECTED_ORGANS:
            raise SparseGateUpParityLauncherError(f"{label} selected-organs binding drifted")
        revalidation_ref = _mapping(binding.get("immutable_source_revalidation"), f"{label} revalidation")
        _assert_sealed_reference(revalidation_ref, revalidation_evidence, revalidation_seal, f"{label} revalidation")
        if revalidation_ref.get("source_revision") != source_revision:
            raise SparseGateUpParityLauncherError(f"{label} source revision drifted")
        source_audit = _mapping(binding.get("source_audit"), f"{label} source audit")
        if source_audit.get("seal_sha256") != source_audit_seal:
            raise SparseGateUpParityLauncherError(f"{label} source audit seal drifted")

    admission_current, admission_current_seal = _sealed_json(
        config.candidate_admission_current, "--candidate-admission-current"
    )
    if (
        admission_current.get("schema") != ADMISSION_CURRENT_SCHEMA
        or admission_current.get("status") != ADMISSION_CURRENT_STATUS
    ):
        raise SparseGateUpParityLauncherError("--candidate-admission-current schema/status drifted")
    admission_current_evidence = _file_evidence(
        config.candidate_admission_current, "--candidate-admission-current"
    )
    manifest_ref = _mapping(admission_current.get("complete_manifest"), "candidate admission manifest")
    _assert_sealed_reference(manifest_ref, manifest_evidence, manifest_seal, "candidate admission manifest")
    if manifest_ref.get("schema") != CANDIDATE_SCHEMA or manifest_ref.get("status") != CANDIDATE_STATUS:
        raise SparseGateUpParityLauncherError("candidate admission manifest grammar drifted")
    admission_ref = _mapping(admission_current.get("admission_receipt"), "candidate admission receipt")
    admission_path = _canonical_from_document(admission_ref.get("path"), "candidate admission receipt path")
    admission, admission_seal = _sealed_json(admission_path, "candidate admission receipt")
    _assert_pinned(admission_seal, PINNED_ADMISSION_RECEIPT_SEAL, "candidate admission receipt")
    admission_evidence = _file_evidence(admission_path, "candidate admission receipt")
    _assert_sealed_reference(admission_ref, admission_evidence, admission_seal, "candidate admission receipt")
    if admission.get("schema") != ADMISSION_SCHEMA or admission.get("status") != ADMISSION_STATUS:
        raise SparseGateUpParityLauncherError("candidate admission receipt schema/status drifted")
    _assert_sealed_reference(
        admission.get("complete_manifest"), manifest_evidence, manifest_seal, "candidate admission receipt manifest"
    )
    _assert_sealed_reference(admission.get("selection_receipt"), selection_evidence, selection_seal, "candidate admission selection")
    _assert_sealed_reference(
        admission.get("source_binding_snapshot"), snapshot_evidence, snapshot_seal, "candidate admission source snapshot"
    )
    _assert_sealed_reference(
        admission.get("immutable_source_revalidation"),
        revalidation_evidence,
        revalidation_seal,
        "candidate admission revalidation",
    )
    terminal_ref = _mapping(admission.get("terminal"), "candidate admission terminal")
    terminal_path = _canonical_from_document(terminal_ref.get("path"), "candidate admission terminal path")
    terminal, terminal_seal = _sealed_json(terminal_path, "candidate terminal")
    terminal_evidence = _file_evidence(terminal_path, "candidate terminal")
    _assert_sealed_reference(terminal_ref, terminal_evidence, terminal_seal, "candidate admission terminal")
    if terminal.get("schema") != CANDIDATE_TERMINAL_SCHEMA or terminal.get("status") != CANDIDATE_TERMINAL_STATUS:
        raise SparseGateUpParityLauncherError("candidate terminal schema/status drifted")

    # Keep evidence dictionaries self-contained in subsequent receipts.
    manifest_evidence["seal_sha256"] = manifest_seal
    admission_current_evidence["seal_sha256"] = admission_current_seal
    admission_evidence["seal_sha256"] = admission_seal
    revalidation_evidence["seal_sha256"] = revalidation_seal
    selection_evidence["seal_sha256"] = selection_seal
    snapshot_evidence["seal_sha256"] = snapshot_seal
    terminal_evidence["seal_sha256"] = terminal_seal
    return CandidateContext(
        manifest=manifest_evidence,
        manifest_seal=manifest_seal,
        admission_current=admission_current_evidence,
        admission_current_seal=admission_current_seal,
        admission_receipt=admission_evidence,
        admission_receipt_seal=admission_seal,
        source_revalidation=revalidation_evidence,
        source_revalidation_seal=revalidation_seal,
        selection=selection_evidence,
        selection_seal=selection_seal,
        source_snapshot=snapshot_evidence,
        source_snapshot_seal=snapshot_seal,
        candidate_terminal=terminal_evidence,
        candidate_terminal_seal=terminal_seal,
        source_revision=source_revision,
        source_audit_seal=str(source_audit_seal),
    )


def _select_current(
    path: Path,
    *,
    label: str,
    pointer_schema: str,
    pointer_status: str,
    field: str,
    receipt_schema: str,
    receipt_status: str,
    pinned_seal: str,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    pointer, pointer_seal = _sealed_json(path, f"{label} current pointer")
    if pointer.get("schema") != pointer_schema or pointer.get("status") != pointer_status:
        raise SparseGateUpParityLauncherError(f"{label} current pointer schema/status drifted")
    selected = _mapping(pointer.get(field), f"{label} current pointer {field}")
    receipt_path = _canonical_from_document(selected.get("path"), f"{label} selected receipt path")
    receipt, receipt_seal = _sealed_json(receipt_path, f"{label} selected receipt")
    if selected.get("seal_sha256") != receipt_seal:
        raise SparseGateUpParityLauncherError(f"{label} current pointer receipt seal differs from file")
    _assert_pinned(receipt_seal, pinned_seal, label)
    if receipt.get("schema") != receipt_schema or receipt.get("status") != receipt_status:
        raise SparseGateUpParityLauncherError(f"{label} selected receipt schema/status drifted")
    pointer_evidence = _file_evidence(path, f"{label} current pointer")
    pointer_evidence["seal_sha256"] = pointer_seal
    receipt_evidence = _file_evidence(receipt_path, f"{label} selected receipt")
    receipt_evidence["seal_sha256"] = receipt_seal
    return pointer_evidence, pointer_seal, receipt_evidence, receipt_seal


def _bind_trace(config: LaunchConfig, candidate: CandidateContext) -> TraceContext:
    compiler_current, compiler_current_seal, compiler_receipt, compiler_receipt_seal = _select_current(
        config.compiler_trace_current,
        label="compiler trace",
        pointer_schema=COMPILER_CURRENT_SCHEMA,
        pointer_status=COMPILER_CURRENT_STATUS,
        field="compiler_trace_receipt",
        receipt_schema=COMPILER_SCHEMA,
        receipt_status=COMPILER_STATUS,
        pinned_seal=PINNED_COMPILER_TRACE_SEAL,
    )
    compiler_document, _ = _sealed_json(Path(compiler_receipt["path"]), "compiler trace selected receipt")
    binding = _mapping(compiler_document.get("binding"), "compiler trace binding")
    _assert_sealed_reference(binding.get("candidate_manifest"), candidate.manifest, candidate.manifest_seal, "compiler trace candidate")
    admission_ref = _mapping(binding.get("candidate_native_admission"), "compiler trace candidate admission")
    if _canonical_from_document(admission_ref.get("current_pointer_path"), "compiler trace admission pointer") != Path(
        candidate.admission_current["path"]
    ) or admission_ref.get("current_pointer_seal_sha256") != candidate.admission_current_seal:
        raise SparseGateUpParityLauncherError("compiler trace candidate admission binding drifted")
    _assert_sealed_reference(binding.get("selection_receipt"), candidate.selection, candidate.selection_seal, "compiler trace selection")
    snapshot_ref = _mapping(binding.get("source_snapshot"), "compiler trace source snapshot")
    _assert_sealed_reference(snapshot_ref, candidate.source_snapshot, candidate.source_snapshot_seal, "compiler trace source snapshot")
    revalidation_ref = _mapping(snapshot_ref.get("immutable_source_revalidation"), "compiler trace source revalidation")
    _assert_sealed_reference(
        revalidation_ref, candidate.source_revalidation, candidate.source_revalidation_seal, "compiler trace source revalidation"
    )
    if revalidation_ref.get("source_revision") != candidate.source_revision:
        raise SparseGateUpParityLauncherError("compiler trace source revision drifted")
    run_root = Path(_text(binding.get("run_root"), "compiler trace run root"))
    if not run_root.is_absolute() or not run_root.is_dir():
        raise SparseGateUpParityLauncherError("compiler trace run root is absent")
    rows = compiler_document.get("public_probe_compiler_traces")
    if not isinstance(rows, list):
        raise SparseGateUpParityLauncherError("compiler trace lacks public probe rows")
    selected = [dict(row) for row in rows if isinstance(row, Mapping) and row.get("probe_id") == TARGET_PROBE]
    if len(selected) != 1:
        raise SparseGateUpParityLauncherError("compiler trace must contain exactly one literal_hawking row")
    annotated_path = run_root / _safe_relative_path(selected[0].get("annotated_trace_path"), "literal_hawking annotated trace")
    annotated_evidence = _file_evidence(annotated_path, "literal_hawking annotated trace")
    if selected[0].get("annotated_trace_sha256") != annotated_evidence["sha256"]:
        raise SparseGateUpParityLauncherError("literal_hawking annotated trace hash drifted")
    annotated = _read_json(annotated_path, "literal_hawking annotated trace")
    if annotated.get("schema") != ANNOTATED_TRACE_SCHEMA or annotated.get("status") != ANNOTATED_TRACE_STATUS:
        raise SparseGateUpParityLauncherError("literal_hawking annotated trace schema/status drifted")
    compiler = _mapping(annotated.get("compiler_trace"), "literal_hawking compiler trace")
    if (
        compiler.get("status") != ANNOTATED_TRACE_STATUS
        or compiler.get("model_execution_started") is not False
        or compiler.get("capture_timing") != "AFTER_CONTEXT_COMPILATION_BEFORE_PROVIDER_OR_MODEL_EXECUTION"
    ):
        raise SparseGateUpParityLauncherError("literal_hawking trace is not a pre-execution diagnostic")
    prompt = _mapping(
        _mapping(annotated.get("source_tokenizer_annotations"), "literal_hawking annotations").get("source_one_user_native_prompt"),
        "literal_hawking source prompt",
    )
    token_ids = prompt.get("token_ids")
    if not isinstance(token_ids, list) or len(token_ids) != TARGET_TOKEN_COUNT or any(
        not isinstance(value, int) or value < 0 or value > 0xFFFFFFFF for value in token_ids
    ):
        raise SparseGateUpParityLauncherError("literal_hawking source-template IDs drifted")
    token_hash = _sha256_bytes(b"".join(int(value).to_bytes(4, "little") for value in token_ids))
    if prompt.get("token_ids_u32le_sha256") != token_hash or prompt.get("add_special_tokens") is not True:
        raise SparseGateUpParityLauncherError("literal_hawking source-template token hash drifted")

    route_current, route_current_seal, route_receipt, route_receipt_seal = _select_current(
        config.route_capture_current,
        label="L0 route capture",
        pointer_schema=ROUTE_CURRENT_SCHEMA,
        pointer_status=ROUTE_CURRENT_STATUS,
        field="route_capture_receipt",
        receipt_schema=ROUTE_SCHEMA,
        receipt_status=ROUTE_STATUS,
        pinned_seal=PINNED_ROUTE_CAPTURE_SEAL,
    )
    route_document, _ = _sealed_json(Path(route_receipt["path"]), "L0 route capture selected receipt")
    route_binding = _mapping(route_document.get("binding"), "L0 route capture binding")
    compiler_ref = _mapping(route_binding.get("compiler_trace"), "L0 route capture compiler trace")
    if _canonical_from_document(compiler_ref.get("path"), "L0 route capture compiler trace path") != Path(
        compiler_receipt["path"]
    ) or compiler_ref.get("seal_sha256") != compiler_receipt_seal:
        raise SparseGateUpParityLauncherError("L0 route capture compiler trace binding drifted")
    selection_ref = _mapping(route_binding.get("candidate_selection"), "L0 route capture selection")
    if _canonical_from_document(selection_ref.get("path"), "L0 route capture selection path") != Path(
        candidate.selection["path"]
    ) or selection_ref.get("seal_sha256") != candidate.selection_seal:
        raise SparseGateUpParityLauncherError("L0 route capture selection binding drifted")
    summary = route_document.get("probe_summary")
    if not isinstance(summary, list):
        raise SparseGateUpParityLauncherError("L0 route capture lacks probe summary")
    matching = [dict(row) for row in summary if isinstance(row, Mapping) and row.get("probe_id") == TARGET_PROBE]
    if len(matching) != 1:
        raise SparseGateUpParityLauncherError("L0 route capture must contain literal_hawking exactly once")
    selected_summary = matching[0]
    if (
        selected_summary.get("source_template_token_count") != TARGET_TOKEN_COUNT
        or selected_summary.get("route_membership_and_hidden_steps") != TARGET_TOKEN_COUNT
        or selected_summary.get("l0_expert0_selected_positions") != [TARGET_POSITION]
        or selected_summary.get("l0_expert0_selected_position_count") != 1
    ):
        raise SparseGateUpParityLauncherError("L0 route capture literal_hawking E0 selection drifted")
    result_path = _canonical_from_document(route_binding.get("capture_result_path"), "L0 route capture result")
    result_evidence = _file_evidence(result_path, "L0 route capture result")
    if result_evidence["sha256"] != route_binding.get("capture_result_sha256"):
        raise SparseGateUpParityLauncherError("L0 route capture result hash drifted")
    output_root = Path(_text(route_binding.get("capture_output_root"), "L0 route capture output root"))
    if not output_root.is_absolute() or not output_root.is_dir():
        raise SparseGateUpParityLauncherError("L0 route capture output root is absent")
    result = _read_json(result_path, "L0 route capture result")
    if result.get("status") != ROUTE_STATUS or result.get("capture_protocol_revision") != "l0-route-hidden-capture-output-parent-v2":
        raise SparseGateUpParityLauncherError("L0 route capture result protocol drifted")
    probe_rows = result.get("probes")
    if not isinstance(probe_rows, list):
        raise SparseGateUpParityLauncherError("L0 route capture result has no probe rows")
    probe = next((dict(row) for row in probe_rows if isinstance(row, Mapping) and row.get("probe_id") == TARGET_PROBE), None)
    if probe is None or not isinstance(probe.get("steps"), list):
        raise SparseGateUpParityLauncherError("L0 route capture result lacks literal_hawking steps")
    steps = probe["steps"]
    if len(steps) != TARGET_TOKEN_COUNT or not isinstance(steps[TARGET_POSITION], Mapping):
        raise SparseGateUpParityLauncherError("L0 route capture literal_hawking target step is absent")
    step = dict(steps[TARGET_POSITION])
    hidden = _mapping(step.get("router_input_hidden_f32le"), "literal_hawking E0 F32LE input")
    expected_relative = f"hidden/{TARGET_PROBE}/{TARGET_POSITION:06d}.f32le"
    if (
        step.get("position") != TARGET_POSITION
        or not isinstance(step.get("selected_expert_ids"), list)
        or 0 not in step["selected_expert_ids"]
        or hidden.get("relative_path") != expected_relative
        or hidden.get("elements") != INPUT_VALUES
        or hidden.get("bytes") != INPUT_BYTES
        or not _is_sha256(hidden.get("sha256"))
    ):
        raise SparseGateUpParityLauncherError("literal_hawking E0 F32LE route input contract drifted")
    summary_payloads = selected_summary.get("hidden_payloads")
    if not isinstance(summary_payloads, list) or not any(
        isinstance(row, Mapping)
        and row.get("relative_path") == expected_relative
        and row.get("sha256") == hidden.get("sha256")
        for row in summary_payloads
    ):
        raise SparseGateUpParityLauncherError("literal_hawking E0 F32LE summary hash drifted")
    relative = _safe_relative_path(hidden.get("relative_path"), "literal_hawking E0 F32LE relative path")
    input_path = (output_root / relative).resolve()
    try:
        input_path.relative_to(output_root.resolve())
    except ValueError as exc:
        raise SparseGateUpParityLauncherError("literal_hawking E0 F32LE path escapes route output") from exc
    input_evidence = _file_evidence(input_path, "literal_hawking E0 F32LE input")
    if input_evidence["bytes"] != INPUT_BYTES or input_evidence["sha256"] != hidden.get("sha256"):
        raise SparseGateUpParityLauncherError("literal_hawking E0 F32LE bytes/SHA drifted")
    input_evidence.update(
        {
            "probe_id": TARGET_PROBE,
            "position": TARGET_POSITION,
            "elements": INPUT_VALUES,
            "route_capture_result": result_evidence,
        }
    )
    return TraceContext(
        compiler_current=compiler_current,
        compiler_current_seal=compiler_current_seal,
        compiler_receipt=compiler_receipt,
        compiler_receipt_seal=compiler_receipt_seal,
        route_current=route_current,
        route_current_seal=route_current_seal,
        route_receipt=route_receipt,
        route_receipt_seal=route_receipt_seal,
        input_f32le=input_evidence,
    )


def _canonical_cpu_command(context: LaunchContext) -> list[str]:
    """The only child command admitted by the sealed CPU preparation record."""

    candidate = context.candidate
    trace = context.trace
    return [
        str(context.probe_binary["path"]),
        "--mode",
        MODE_CPU,
        "--manifest",
        str(candidate.manifest["path"]),
        "--expected-manifest-seal-sha256",
        candidate.manifest_seal,
        "--expected-source-audit-seal-sha256",
        candidate.source_audit_seal,
        "--expected-source-revision",
        candidate.source_revision,
        "--expected-revalidation-path",
        str(candidate.source_revalidation["path"]),
        "--expected-revalidation-seal-sha256",
        candidate.source_revalidation_seal,
        "--expected-selection-path",
        str(candidate.selection["path"]),
        "--expected-selection-seal-sha256",
        candidate.selection_seal,
        "--expected-source-snapshot-path",
        str(candidate.source_snapshot["path"]),
        "--expected-source-snapshot-seal-sha256",
        candidate.source_snapshot_seal,
        "--expected-terminal-path",
        str(candidate.candidate_terminal["path"]),
        "--expected-terminal-seal-sha256",
        candidate.candidate_terminal_seal,
        "--input-f32le",
        str(trace.input_f32le["path"]),
        "--expected-input-sha256",
        str(trace.input_f32le["sha256"]),
        "--max-seq-len",
        "512",
    ]


def _bind_preparation(config: LaunchConfig, context: LaunchContext) -> PreparationContext:
    current, current_seal, receipt, receipt_seal = _select_current(
        config.preparation_current,
        label="sparse gate/up component preparation",
        pointer_schema=PREPARATION_CURRENT_SCHEMA,
        pointer_status=PREPARATION_CURRENT_STATUS,
        field="component_preparation_receipt",
        receipt_schema=PREPARATION_SCHEMA,
        receipt_status=PREPARATION_STATUS,
        pinned_seal=PINNED_PREPARATION_RECEIPT_SEAL,
    )
    document, _ = _sealed_json(Path(receipt["path"]), "sparse gate/up component preparation receipt")
    binding = _mapping(document.get("binding"), "sparse gate/up component preparation binding")
    candidate = context.candidate
    trace = context.trace
    _assert_file_reference(binding.get("probe_binary"), context.probe_binary, "preparation probe binary")
    if binding.get("candidate_manifest_seal_sha256") != candidate.manifest_seal:
        raise SparseGateUpParityLauncherError("preparation candidate manifest seal drifted")
    for field, evidence, receipt_seal_value in (
        ("candidate_admission_current", candidate.admission_current, candidate.admission_current_seal),
        ("candidate_admission_receipt", candidate.admission_receipt, candidate.admission_receipt_seal),
        ("route_capture_current", trace.route_current, trace.route_current_seal),
        ("route_capture_receipt", trace.route_receipt, trace.route_receipt_seal),
    ):
        _assert_sealed_reference(binding.get(field), evidence, receipt_seal_value, f"preparation {field}")
    _assert_file_reference(binding.get("candidate_manifest"), candidate.manifest, "preparation candidate manifest")
    component_input = _mapping(binding.get("component_input"), "preparation component input")
    input_row = _mapping(component_input.get("device_produced_router_input_f32le"), "preparation E0 input")
    if (
        _canonical_from_document(input_row.get("path"), "preparation E0 input path") != Path(trace.input_f32le["path"])
        or input_row.get("sha256") != trace.input_f32le["sha256"]
        or input_row.get("bytes") != INPUT_BYTES
        or input_row.get("elements") != INPUT_VALUES
        or component_input.get("probe_id") != TARGET_PROBE
        or component_input.get("l0_e0_selected_position") != TARGET_POSITION
        or component_input.get("source_template_token_count") != TARGET_TOKEN_COUNT
    ):
        raise SparseGateUpParityLauncherError("preparation literal_hawking E0 input binding drifted")
    invocation = _mapping(document.get("cpu_oracle_invocation"), "preparation CPU invocation")
    command = invocation.get("command_without_output_dir")
    if not isinstance(command, list) or not all(isinstance(value, str) for value in command):
        raise SparseGateUpParityLauncherError("preparation CPU command is malformed")
    if list(command) != _canonical_cpu_command(context):
        raise SparseGateUpParityLauncherError("preparation CPU command differs from exact current evidence")
    if (
        invocation.get("mode") != MODE_CPU
        or invocation.get("metal_context_or_dispatch_performed") is not False
        or invocation.get("outer_controller_must_create_fresh_output_dir") is not True
    ):
        raise SparseGateUpParityLauncherError("preparation CPU-only boundary drifted")
    boundary = _mapping(document.get("claim_boundary"), "preparation claim boundary")
    if (
        boundary.get("preparation_only") is not True
        or boundary.get("no_metal_context_no_model_forward_no_server_or_watcher_change") is not True
        or boundary.get("does_not_claim_coherence_hcli_tps_tg_capability_or_tournament") is not True
    ):
        raise SparseGateUpParityLauncherError("preparation claim boundary drifted")
    future = _mapping(document.get("future_device_parity_contract"), "preparation future device contract")
    if (
        future.get("mode") != MODE_DEVICE
        or future.get("requires_cpu_oracle_activation_f64le_from_this_exact_binding") is not True
        or future.get("requires_fresh_explicit_quiet_gpu_lease") is not True
        or future.get("automatic_retry_forbidden") is not True
        or future.get("one_device_process_one_lease_one_terminal_receipt_or_refusal") is not True
    ):
        raise SparseGateUpParityLauncherError("preparation future device contract drifted")
    return PreparationContext(
        current=current,
        current_seal=current_seal,
        receipt=receipt,
        receipt_seal=receipt_seal,
        cpu_command_without_output_dir=tuple(command),
    )


def _assert_context_binding(document: Mapping[str, Any], context: LaunchContext, label: str) -> None:
    binding = _mapping(document.get("binding"), f"{label} binding")
    candidate = context.candidate
    trace = context.trace
    if _canonical_from_document(binding.get("candidate_manifest_path"), f"{label} candidate manifest") != Path(candidate.manifest["path"]):
        raise SparseGateUpParityLauncherError(f"{label} candidate manifest path drifted")
    if binding.get("candidate_manifest_seal_sha256") != candidate.manifest_seal:
        raise SparseGateUpParityLauncherError(f"{label} candidate manifest seal drifted")
    if binding.get("source_audit_seal_sha256") != candidate.source_audit_seal:
        raise SparseGateUpParityLauncherError(f"{label} source-audit seal drifted")
    if binding.get("source_revision") != candidate.source_revision:
        raise SparseGateUpParityLauncherError(f"{label} source revision drifted")
    for field, evidence, receipt_seal in (
        ("revalidation", candidate.source_revalidation, candidate.source_revalidation_seal),
        ("selection", candidate.selection, candidate.selection_seal),
        ("source_snapshot", candidate.source_snapshot, candidate.source_snapshot_seal),
        ("terminal", candidate.candidate_terminal, candidate.candidate_terminal_seal),
    ):
        row = _mapping(binding.get(field), f"{label} {field}")
        if _canonical_from_document(row.get("path"), f"{label} {field} path") != Path(evidence["path"]):
            raise SparseGateUpParityLauncherError(f"{label} {field} path drifted")
        if row.get("seal_sha256") != receipt_seal:
            raise SparseGateUpParityLauncherError(f"{label} {field} seal drifted")
    input_row = _mapping(binding.get("input_f32le"), f"{label} input F32LE")
    if (
        _canonical_from_document(input_row.get("path"), f"{label} input F32LE path") != Path(trace.input_f32le["path"])
        or input_row.get("sha256") != trace.input_f32le["sha256"]
        or input_row.get("values") != INPUT_VALUES
    ):
        raise SparseGateUpParityLauncherError(f"{label} literal_hawking E0 F32LE input drifted")
    if binding.get("runtime_executable_sha256") != context.probe_binary["sha256"]:
        raise SparseGateUpParityLauncherError(f"{label} runtime executable SHA drifted")
    topology = _mapping(binding.get("fixed_topology"), f"{label} fixed topology")
    if (
        topology.get("selected_sparse_pair") != "L0/E0 gate+up HQ30GR2"
        or topology.get("kernel") != EXPECTED_KERNEL
        or topology.get("no_direct_fallback_for_sparse_pair") is not True
        or topology.get("no_bf16_or_dense_weight_path") is not True
    ):
        raise SparseGateUpParityLauncherError(f"{label} typed sparse topology drifted")


def _validate_claim_boundary(document: Mapping[str, Any], label: str) -> None:
    boundary = _mapping(document.get("claim_boundary"), f"{label} claim boundary")
    if (
        boundary.get("not_a_complete_layer_or_full_token") is not True
        or boundary.get("no_logits_sampler_generation_hcli_or_server") is not True
        or boundary.get("not_coherence_tps_tg_capability_manager_or_tournament") is not True
    ):
        raise SparseGateUpParityLauncherError(f"{label} claim boundary drifted")


def _validate_typed_catalog(document: Mapping[str, Any], label: str) -> None:
    catalog = _mapping(document.get("typed_catalog"), f"{label} typed catalog")
    if (
        catalog.get("verified_payload_count") != EXPECTED_VERIFIED_PAYLOADS
        or catalog.get("direct_tensor_count") != EXPECTED_DIRECT_TENSORS
        or catalog.get("sparse_residual_tensor_count") != 2
    ):
        raise SparseGateUpParityLauncherError(f"{label} typed payload catalog drifted")
    dispatch = _mapping(catalog.get("sparse_gate_up_dispatch"), f"{label} sparse dispatch")
    if (
        dispatch.get("kernel_name") != EXPECTED_KERNEL
        or dispatch.get("rows") != OUTPUT_VALUES
        or dispatch.get("cols") != INPUT_VALUES
        or dispatch.get("group_size") != EXPECTED_GROUP_SIZE
        or dispatch.get("gate_residual_count") != EXPECTED_RESIDUAL_COUNT
        or dispatch.get("up_residual_count") != EXPECTED_RESIDUAL_COUNT
        or dispatch.get("exact_non_fma_scalar_order_required") is not True
        or dispatch.get("direct_fallback_for_sparse_residual_forbidden") is not True
    ):
        raise SparseGateUpParityLauncherError(f"{label} typed sparse dispatch drifted")


def _validate_inner_result(config: LaunchConfig, context: LaunchContext) -> dict[str, Any]:
    result_path = config.capture_dir / INNER_CAPTURE / "result.json"
    evidence: dict[str, Any] = {"path": str(result_path), "present": result_path.is_file()}
    if not result_path.is_file():
        return evidence
    try:
        result = _read_json(result_path, "inner sparse gate/up parity result")
        result_evidence = _file_evidence(result_path, "inner sparse gate/up parity result")
        evidence.update(result_evidence)
        evidence["schema"] = result.get("schema")
        evidence["status"] = result.get("status")
        evidence["mode"] = result.get("mode")
        if result.get("schema") != EXPECTED_INNER_SCHEMA or result.get("mode") != config.mode:
            raise SparseGateUpParityLauncherError("inner result schema/mode drifted")
        expected_status = EXPECTED_CPU_STATUS if config.mode == MODE_CPU else EXPECTED_DEVICE_STATUS
        if result.get("status") != expected_status:
            raise SparseGateUpParityLauncherError("inner result status is not a successful component result")
        _assert_context_binding(result, context, "inner result")
        _validate_typed_catalog(result, "inner result")
        _validate_claim_boundary(result, "inner result")
        cpu_oracle = _mapping(result.get("cpu_oracle"), "inner CPU oracle")
        if (
            cpu_oracle.get("activation_values") != OUTPUT_VALUES
            or not _is_sha256(cpu_oracle.get("activation_f64le_sha256"))
            or cpu_oracle.get("admission_snapshot_only") is not True
            or cpu_oracle.get("raw_bf16_or_dense_weight_path") is not False
        ):
            raise SparseGateUpParityLauncherError("inner CPU oracle contract drifted")
        if config.mode == MODE_CPU:
            output = _mapping(result.get("cpu_oracle_output"), "inner CPU oracle output")
            output_path = _canonical_from_document(output.get("path"), "inner CPU oracle output path")
            output_evidence = _file_evidence(output_path, "inner CPU oracle output")
            if (
                output_evidence["bytes"] != CPU_OUTPUT_BYTES
                or output.get("values") != OUTPUT_VALUES
                or output.get("sha256") != output_evidence["sha256"]
                or output_evidence["sha256"] != cpu_oracle.get("activation_f64le_sha256")
            ):
                raise SparseGateUpParityLauncherError("inner CPU oracle bytes/SHA drifted")
            evidence["cpu_activation"] = output_evidence | {"values": OUTPUT_VALUES}
        else:
            assert context.cpu_oracle is not None
            protected = _mapping(result.get("protected_cpu_oracle"), "inner protected CPU oracle")
            cpu = context.cpu_oracle.cpu_activation
            if (
                _canonical_from_document(protected.get("path"), "inner protected CPU oracle path") != Path(cpu["path"])
                or protected.get("sha256") != cpu["sha256"]
                or protected.get("recomputed_current_sha256") != cpu["sha256"]
                or cpu_oracle.get("activation_f64le_sha256") != cpu["sha256"]
            ):
                raise SparseGateUpParityLauncherError("inner device CPU-oracle binding drifted")
            parity = _mapping(result.get("device_parity"), "inner device parity")
            if (
                parity.get("passes") is not True
                or parity.get("values_compared") != OUTPUT_VALUES
                or not isinstance(parity.get("max_abs_error"), (int, float))
                or not isinstance(parity.get("max_rel_error"), (int, float))
            ):
                raise SparseGateUpParityLauncherError("inner device parity result drifted")
            device = _mapping(result.get("device_output"), "inner device output")
            device_path = _canonical_from_document(device.get("path"), "inner device output path")
            device_evidence = _file_evidence(device_path, "inner device output")
            if (
                device_evidence["bytes"] != DEVICE_OUTPUT_BYTES
                or device.get("values") != OUTPUT_VALUES
                or device.get("sha256") != device_evidence["sha256"]
            ):
                raise SparseGateUpParityLauncherError("inner device output bytes/SHA drifted")
            execution = _mapping(result.get("device_execution"), "inner device execution")
            if (
                execution.get("metal_context_created") is not True
                or execution.get("kernel") != EXPECTED_KERNEL
                or execution.get("only_selected_l0_e0_gate_up_swiglu_pair_executed") is not True
                or execution.get("all_layers_executed") is not False
                or execution.get("full_token_executed") is not False
            ):
                raise SparseGateUpParityLauncherError("inner device component boundary drifted")
            evidence["device_activation"] = device_evidence | {"values": OUTPUT_VALUES}
    except SparseGateUpParityLauncherError as exc:
        evidence["binding_valid"] = False
        evidence["binding_error"] = str(exc)
    else:
        evidence["binding_valid"] = True
    return evidence


def _bind_cpu_oracle(path: Path, context: LaunchContext) -> CpuOracleContext:
    outer, outer_seal = _sealed_json(path, "--cpu-oracle-outer-receipt")
    if outer.get("schema") != SCHEMA or outer.get("status") != _success_status(MODE_CPU):
        raise SparseGateUpParityLauncherError("--cpu-oracle-outer-receipt is not a successful CPU-oracle outer terminal")
    source = _mapping(outer.get("source_binding"), "CPU oracle outer source binding")
    if source.get("mode") != MODE_CPU or source.get("lease_receipt") is not None or source.get("cpu_oracle_outer_receipt") is not None:
        raise SparseGateUpParityLauncherError("CPU oracle outer receipt has an invalid launch mode")
    _validate_outer_source_binding(source, context, "CPU oracle outer")
    terminal = _mapping(_mapping(outer.get("child"), "CPU oracle outer child").get("terminal"), "CPU oracle outer terminal")
    if terminal.get("reaped") is not True or terminal.get("timed_out") is not False or terminal.get("exit_code") != 0:
        raise SparseGateUpParityLauncherError("CPU oracle outer child did not exit cleanly")
    inner = _mapping(outer.get("inner_probe_capture"), "CPU oracle outer inner capture")
    if inner.get("binding_valid") is not True or inner.get("status") != EXPECTED_CPU_STATUS:
        raise SparseGateUpParityLauncherError("CPU oracle outer lacks a bound CPU result")
    activation = _mapping(inner.get("cpu_activation"), "CPU oracle outer activation")
    activation_path = _canonical_from_document(activation.get("path"), "CPU oracle activation path")
    activation_evidence = _file_evidence(activation_path, "CPU oracle activation")
    if (
        activation_evidence["bytes"] != CPU_OUTPUT_BYTES
        or activation.get("sha256") != activation_evidence["sha256"]
        or activation.get("values") != OUTPUT_VALUES
    ):
        raise SparseGateUpParityLauncherError("CPU oracle outer byte binding drifted")
    outer_evidence = _file_evidence(path, "--cpu-oracle-outer-receipt")
    outer_evidence["seal_sha256"] = outer_seal
    return CpuOracleContext(
        outer_receipt=outer_evidence,
        outer_receipt_seal=outer_seal,
        cpu_activation=activation_evidence | {"values": OUTPUT_VALUES},
    )


def _validate_outer_source_binding(source: Mapping[str, Any], context: LaunchContext, label: str) -> None:
    candidate = context.candidate
    trace = context.trace
    binary = _mapping(source.get("probe_binary"), f"{label} probe binary")
    _assert_file_reference(binary, context.probe_binary, f"{label} probe binary")
    for field, evidence, receipt_seal in (
        ("candidate_manifest", candidate.manifest, candidate.manifest_seal),
        ("candidate_admission_current", candidate.admission_current, candidate.admission_current_seal),
        ("candidate_admission_receipt", candidate.admission_receipt, candidate.admission_receipt_seal),
        ("source_revalidation", candidate.source_revalidation, candidate.source_revalidation_seal),
        ("selection_receipt", candidate.selection, candidate.selection_seal),
        ("source_snapshot", candidate.source_snapshot, candidate.source_snapshot_seal),
        ("candidate_terminal", candidate.candidate_terminal, candidate.candidate_terminal_seal),
        ("compiler_trace_current", trace.compiler_current, trace.compiler_current_seal),
        ("compiler_trace_receipt", trace.compiler_receipt, trace.compiler_receipt_seal),
        ("route_capture_current", trace.route_current, trace.route_current_seal),
        ("route_capture_receipt", trace.route_receipt, trace.route_receipt_seal),
        ("preparation_current", context.preparation.current, context.preparation.current_seal),
        ("preparation_receipt", context.preparation.receipt, context.preparation.receipt_seal),
    ):
        _assert_sealed_reference(source.get(field), evidence, receipt_seal, f"{label} {field}")
    route_input = _mapping(source.get("literal_hawking_e0_input_f32le"), f"{label} E0 input")
    if (
        _canonical_from_document(route_input.get("path"), f"{label} E0 input path") != Path(trace.input_f32le["path"])
        or route_input.get("sha256") != trace.input_f32le["sha256"]
        or route_input.get("bytes") != INPUT_BYTES
        or route_input.get("elements") != INPUT_VALUES
        or route_input.get("position") != TARGET_POSITION
        or route_input.get("probe_id") != TARGET_PROBE
    ):
        raise SparseGateUpParityLauncherError(f"{label} E0 input binding drifted")
    if source.get("source_revision") != candidate.source_revision or source.get("source_audit_seal_sha256") != candidate.source_audit_seal:
        raise SparseGateUpParityLauncherError(f"{label} source revision/audit binding drifted")


def _bind_lease(path: Path, context: LaunchContext) -> tuple[dict[str, Any], str]:
    assert context.cpu_oracle is not None
    lease, lease_seal = _sealed_json(path, "--lease-receipt")
    if lease.get("schema") != QUIET_LEASE_SCHEMA or lease.get("status") != QUIET_LEASE_STATUS:
        raise SparseGateUpParityLauncherError("--lease-receipt schema/status does not authorize the sparse component")
    lifecycle = _mapping(lease.get("one_shot_lifecycle"), "sparse component lease lifecycle")
    if (
        lifecycle.get("fresh_for_this_exact_launch") is not True
        or lifecycle.get("prior_terminal_receipt") is not None
        or lifecycle.get("automatic_retry_allowed") is not False
    ):
        raise SparseGateUpParityLauncherError("--lease-receipt is not a fresh one-shot lease")
    policy = _mapping(lease.get("execution_policy"), "sparse component lease policy")
    if (
        policy.get("component") != QUIET_LEASE_COMPONENT
        or policy.get("quiet_qwen_family_gpu_lease") is not True
        or policy.get("strict_math") is not True
        or policy.get("component_only") is not True
        or policy.get("one_child_process_group_only") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("all_layer_or_full_token_allowed") is not False
        or policy.get("logit_or_generation_allowed") is not False
        or policy.get("hcli_or_server_allowed") is not False
        or policy.get("coherence_claim_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
        or policy.get("capability_claim_allowed") is not False
        or policy.get("tournament_claim_allowed") is not False
    ):
        raise SparseGateUpParityLauncherError("--lease-receipt is not component-only and non-timed")
    artifact = _mapping(lease.get("artifact_binding"), "sparse component lease artifact binding")
    candidate = context.candidate
    for field, evidence, receipt_seal in (
        ("candidate_manifest", candidate.manifest, candidate.manifest_seal),
        ("candidate_admission_current", candidate.admission_current, candidate.admission_current_seal),
        ("candidate_admission_receipt", candidate.admission_receipt, candidate.admission_receipt_seal),
        ("source_revalidation", candidate.source_revalidation, candidate.source_revalidation_seal),
        ("selection_receipt", candidate.selection, candidate.selection_seal),
        ("source_snapshot", candidate.source_snapshot, candidate.source_snapshot_seal),
        ("candidate_terminal", candidate.candidate_terminal, candidate.candidate_terminal_seal),
    ):
        _assert_sealed_reference(artifact.get(field), evidence, receipt_seal, f"lease {field}")
    upstream = _mapping(lease.get("upstream_binding"), "sparse component lease upstream binding")
    trace = context.trace
    for field, evidence, receipt_seal in (
        ("compiler_trace_receipt", trace.compiler_receipt, trace.compiler_receipt_seal),
        ("route_capture_receipt", trace.route_receipt, trace.route_receipt_seal),
        ("preparation_receipt", context.preparation.receipt, context.preparation.receipt_seal),
    ):
        _assert_sealed_reference(upstream.get(field), evidence, receipt_seal, f"lease {field}")
    input_row = _mapping(upstream.get("literal_hawking_e0_input_f32le"), "lease literal_hawking E0 input")
    if (
        _canonical_from_document(input_row.get("path"), "lease literal_hawking E0 input path") != Path(trace.input_f32le["path"])
        or input_row.get("sha256") != trace.input_f32le["sha256"]
        or input_row.get("bytes") != INPUT_BYTES
        or input_row.get("elements") != INPUT_VALUES
    ):
        raise SparseGateUpParityLauncherError("lease literal_hawking E0 input binding drifted")
    cpu = _mapping(lease.get("cpu_oracle_binding"), "sparse component lease CPU oracle binding")
    _assert_sealed_reference(
        cpu.get("outer_terminal_receipt"),
        context.cpu_oracle.outer_receipt,
        context.cpu_oracle.outer_receipt_seal,
        "lease CPU oracle outer terminal",
    )
    activation = _mapping(cpu.get("cpu_activation_f64le"), "lease CPU oracle activation")
    expected = context.cpu_oracle.cpu_activation
    if (
        _canonical_from_document(activation.get("path"), "lease CPU oracle activation path") != Path(expected["path"])
        or activation.get("sha256") != expected["sha256"]
        or activation.get("bytes") != CPU_OUTPUT_BYTES
        or activation.get("values") != OUTPUT_VALUES
    ):
        raise SparseGateUpParityLauncherError("lease CPU oracle byte binding drifted")
    evidence = _file_evidence(path, "--lease-receipt")
    evidence["seal_sha256"] = lease_seal
    return evidence, lease_seal


def _validate_config(config: LaunchConfig) -> LaunchContext:
    probe = _canonical_regular(config.probe_bin, "--probe-bin", executable=True)
    if probe.name != EXPECTED_PROBE_BASENAME:
        raise SparseGateUpParityLauncherError(
            f"--probe-bin must name {EXPECTED_PROBE_BASENAME}, got {probe.name!r}"
        )
    probe_evidence = _file_evidence(probe, "--probe-bin", executable=True)
    if probe_evidence["sha256"] != EXPECTED_PROBE_BINARY_SHA256:
        raise SparseGateUpParityLauncherError("--probe-bin SHA-256 is not the pinned sparse parity executable")
    for path, label in (
        (config.candidate_manifest, "--candidate-manifest"),
        (config.candidate_admission_current, "--candidate-admission-current"),
        (config.source_revalidation, "--source-revalidation"),
        (config.selection_receipt, "--selection-receipt"),
        (config.source_snapshot, "--source-snapshot"),
        (config.compiler_trace_current, "--compiler-trace-current"),
        (config.route_capture_current, "--route-capture-current"),
        (config.preparation_current, "--preparation-current"),
    ):
        _canonical_regular(path, label)
    _require_absolute(config.capture_dir, "--capture-dir")
    if config.mode not in {MODE_CPU, MODE_DEVICE}:
        raise SparseGateUpParityLauncherError(f"unsupported --mode {config.mode!r}")
    if not config.timeout_seconds > 0:
        raise SparseGateUpParityLauncherError("--timeout-seconds must be positive")
    candidate = _bind_candidate(config)
    trace = _bind_trace(config, candidate)
    unprepared = LaunchContext(
        probe_binary=probe_evidence,
        candidate=candidate,
        trace=trace,
        preparation=PreparationContext({}, "", {}, "", ()),
        cpu_oracle=None,
        lease=None,
        lease_seal=None,
    )
    preparation = _bind_preparation(config, unprepared)
    provisional = LaunchContext(
        probe_binary=probe_evidence,
        candidate=candidate,
        trace=trace,
        preparation=preparation,
        cpu_oracle=None,
        lease=None,
        lease_seal=None,
    )
    if config.mode == MODE_CPU:
        if config.cpu_oracle_outer_receipt is not None or config.lease_receipt is not None:
            raise SparseGateUpParityLauncherError("cpu-oracle mode accepts neither --cpu-oracle-outer-receipt nor --lease-receipt")
        return provisional
    if config.cpu_oracle_outer_receipt is None:
        raise SparseGateUpParityLauncherError("device-parity requires --cpu-oracle-outer-receipt")
    if config.lease_receipt is None:
        raise SparseGateUpParityLauncherError("device-parity requires --lease-receipt")
    cpu = _bind_cpu_oracle(config.cpu_oracle_outer_receipt, provisional)
    context = LaunchContext(
        probe_binary=probe_evidence,
        candidate=candidate,
        trace=trace,
        preparation=preparation,
        cpu_oracle=cpu,
        lease=None,
        lease_seal=None,
    )
    lease, lease_seal = _bind_lease(config.lease_receipt, context)
    return LaunchContext(
        probe_binary=probe_evidence,
        candidate=candidate,
        trace=trace,
        preparation=preparation,
        cpu_oracle=cpu,
        lease=lease,
        lease_seal=lease_seal,
    )


def _launch_identity(config: LaunchConfig, context: LaunchContext) -> str:
    payload = {
        "probe_binary": context.probe_binary,
        "candidate": context.candidate.__dict__,
        "trace": context.trace.__dict__,
        "preparation": context.preparation.__dict__,
        "mode": config.mode,
        "timeout_seconds": config.timeout_seconds,
        "cpu_oracle": None if context.cpu_oracle is None else context.cpu_oracle.__dict__,
        "lease": context.lease,
        "lease_seal": context.lease_seal,
    }
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


def _child_command(config: LaunchConfig, context: LaunchContext) -> list[str]:
    command = list(context.preparation.cpu_command_without_output_dir)
    mode_index = command.index("--mode") + 1
    if command[mode_index] != MODE_CPU:
        raise SparseGateUpParityLauncherError("sealed preparation command lost cpu-oracle mode")
    command[mode_index] = config.mode
    if config.mode == MODE_DEVICE:
        assert context.cpu_oracle is not None
        command.extend(
            (
                "--cpu-activation-f64le",
                str(context.cpu_oracle.cpu_activation["path"]),
                "--expected-cpu-activation-sha256",
                str(context.cpu_oracle.cpu_activation["sha256"]),
            )
        )
    command.extend(("--output-dir", str(config.capture_dir / INNER_CAPTURE)))
    return command


def _sync_evidence(path: Path) -> dict[str, Any]:
    with path.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return _file_evidence(path, f"outer stream {path.name}")


def _terminate_process_group(process: subprocess.Popen[bytes]) -> int | None:
    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait(timeout=10)


def _terminal(returncode: int | None, *, timed_out: bool, spawn_error: str | None = None) -> dict[str, Any]:
    terminal: dict[str, Any] = {
        "reaped": returncode is not None,
        "timed_out": timed_out,
        "returncode": returncode,
        "exit_code": returncode if isinstance(returncode, int) and returncode >= 0 else None,
        "signal": -returncode if isinstance(returncode, int) and returncode < 0 else None,
    }
    if spawn_error is not None:
        terminal["spawn_error"] = spawn_error
        terminal["reaped"] = False
    return terminal


def _success_status(mode: str) -> str:
    suffix = "CPU_ORACLE" if mode == MODE_CPU else "DEVICE_PARITY"
    return f"CAPTURED_QWEN30_HQ30GR2_SPARSE_GATE_UP_{suffix}_OUTER_TERMINAL_COMPONENT_ONLY"


def _terminal_status(config: LaunchConfig, terminal: Mapping[str, Any], inner: Mapping[str, Any]) -> str:
    prefix = "REFUSED_QWEN30_HQ30GR2_SPARSE_GATE_UP_OUTER"
    if terminal.get("spawn_error"):
        return f"{prefix}_CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return f"{prefix}_CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return f"{prefix}_CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return f"{prefix}_CHILD_NONZERO"
    expected = EXPECTED_CPU_STATUS if config.mode == MODE_CPU else EXPECTED_DEVICE_STATUS
    if inner.get("binding_valid") is not True or inner.get("status") != expected:
        return f"{prefix}_ZERO_EXIT_WITHOUT_STRICTLY_BOUND_INNER_RESULT"
    return _success_status(config.mode)


def _terminal_success(receipt: Mapping[str, Any]) -> bool:
    return receipt.get("status") in {_success_status(MODE_CPU), _success_status(MODE_DEVICE)}


def _source_binding(config: LaunchConfig, context: LaunchContext) -> dict[str, Any]:
    candidate = context.candidate
    trace = context.trace
    return {
        "probe_binary": context.probe_binary,
        "candidate_manifest": candidate.manifest,
        "candidate_admission_current": candidate.admission_current,
        "candidate_admission_receipt": candidate.admission_receipt,
        "source_revalidation": candidate.source_revalidation,
        "selection_receipt": candidate.selection,
        "source_snapshot": candidate.source_snapshot,
        "candidate_terminal": candidate.candidate_terminal,
        "source_revision": candidate.source_revision,
        "source_audit_seal_sha256": candidate.source_audit_seal,
        "compiler_trace_current": trace.compiler_current,
        "compiler_trace_receipt": trace.compiler_receipt,
        "route_capture_current": trace.route_current,
        "route_capture_receipt": trace.route_receipt,
        "preparation_current": context.preparation.current,
        "preparation_receipt": context.preparation.receipt,
        "literal_hawking_e0_input_f32le": {
            "path": trace.input_f32le["path"],
            "sha256": trace.input_f32le["sha256"],
            "bytes": INPUT_BYTES,
            "elements": INPUT_VALUES,
            "probe_id": TARGET_PROBE,
            "position": TARGET_POSITION,
        },
        "mode": config.mode,
        "cpu_oracle_outer_receipt": None
        if context.cpu_oracle is None
        else context.cpu_oracle.outer_receipt,
        "lease_receipt": context.lease,
    }


def _terminal_receipt(
    config: LaunchConfig,
    context: LaunchContext,
    *,
    identity: str,
    command: Sequence[str],
    child_pid: int | None,
    started_at: str,
    finished_at: str,
    terminal: Mapping[str, Any],
    capture_error: str | None = None,
) -> dict[str, Any]:
    inner = _validate_inner_result(config, context)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": _terminal_status(config, terminal, inner),
        "recorded_at": finished_at,
        "one_shot": {
            "automatic_retry_disabled": True,
            "same_capture_dir_never_starts_a_second_child": True,
            "terminal_receipt_written_last": True,
        },
        "launch_identity_sha256": identity,
        "source_binding": _source_binding(config, context),
        "child": {
            "pid": child_pid,
            "started_at": started_at,
            "finished_at": finished_at,
            "command": list(command),
            "terminal": dict(terminal),
        },
        "outer_capture": {
            "directory": str(config.capture_dir),
            "stdout": _sync_evidence(config.capture_dir / OUTER_STDOUT),
            "stderr": _sync_evidence(config.capture_dir / OUTER_STDERR),
        },
        "inner_probe_capture": inner,
        "claim_boundary": {
            "outer_terminal_capture_only": True,
            "one_literal_hawking_l0_e0_gate_up_swiglu_component_only": True,
            "does_not_execute_or_claim_all_layer_or_full_token": True,
            "does_not_execute_or_claim_logits_generation_hcli_or_coherence": True,
            "does_not_claim_tps_tg_capability_manager_or_tournament": True,
        },
    }
    if config.mode == MODE_CPU and isinstance(inner.get("cpu_activation"), Mapping):
        receipt["cpu_oracle_output"] = dict(inner["cpu_activation"])
    if capture_error is not None:
        receipt["capture_error"] = capture_error
    return seal(receipt)


def _replay_existing(config: LaunchConfig, identity: str) -> dict[str, Any]:
    terminal_path = config.capture_dir / TERMINAL_FILENAME
    if not terminal_path.is_file():
        raise SparseGateUpParityLauncherError(f"capture directory exists without a terminal receipt: {config.capture_dir}")
    receipt = _read_json(terminal_path, "outer terminal receipt")
    try:
        verify(receipt, label=str(terminal_path))
    except ValueError as exc:
        raise SparseGateUpParityLauncherError(f"outer terminal receipt is not sealed: {exc}") from exc
    if receipt.get("schema") != SCHEMA or receipt.get("launch_identity_sha256") != identity:
        raise SparseGateUpParityLauncherError("capture directory belongs to another launch identity")
    return receipt


def run_attempt(config: LaunchConfig) -> dict[str, Any]:
    """Run exactly one child process group, or sealed-replay its terminal record."""

    context = _validate_config(config)
    identity = _launch_identity(config, context)
    if config.capture_dir.exists():
        return _replay_existing(config, identity)
    if not config.capture_dir.parent.is_dir():
        raise SparseGateUpParityLauncherError(f"capture parent does not exist: {config.capture_dir.parent}")
    try:
        config.capture_dir.mkdir(mode=0o750)
    except FileExistsError:
        return _replay_existing(config, identity)
    inner_dir = config.capture_dir / INNER_CAPTURE
    inner_dir.mkdir(mode=0o750)
    command = _child_command(config, context)
    started_at = _utc_now()
    _atomic_json_new(
        config.capture_dir / ACTIVE_FILENAME,
        seal(
            {
                "schema": SCHEMA,
                "status": "STARTED_QWEN30_HQ30GR2_SPARSE_GATE_UP_OUTER_ONE_SHOT",
                "recorded_at": started_at,
                "launch_identity_sha256": identity,
                "mode": config.mode,
                "command": command,
                "claim_boundary": {"automatic_retry_disabled": True, "component_only": True},
            }
        ),
    )
    child_pid: int | None = None
    capture_error: str | None = None
    with (config.capture_dir / OUTER_STDOUT).open("xb") as stdout, (
        config.capture_dir / OUTER_STDERR
    ).open("xb") as stderr:
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
            terminal = _terminal(None, timed_out=False, spawn_error=f"{type(exc).__name__}: {exc}")
        else:
            child_pid = child.pid
            try:
                _atomic_json_new(
                    config.capture_dir / CHILD_FILENAME,
                    seal(
                        {
                            "schema": SCHEMA,
                            "status": "RUNNING_QWEN30_HQ30GR2_SPARSE_GATE_UP_OUTER_ONE_SHOT",
                            "recorded_at": _utc_now(),
                            "launch_identity_sha256": identity,
                            "mode": config.mode,
                            "pid": child_pid,
                            "parent_pid": os.getpid(),
                            "command": command,
                            "inner_capture_dir": str(inner_dir),
                        }
                    ),
                )
            except SparseGateUpParityLauncherError as exc:
                capture_error = str(exc)
                terminal = _terminal(_terminate_process_group(child), timed_out=False)
            else:
                try:
                    terminal = _terminal(child.wait(timeout=config.timeout_seconds), timed_out=False)
                except subprocess.TimeoutExpired:
                    terminal = _terminal(_terminate_process_group(child), timed_out=True)
    receipt = _terminal_receipt(
        config,
        context,
        identity=identity,
        command=command,
        child_pid=child_pid,
        started_at=started_at,
        finished_at=_utc_now(),
        terminal=terminal,
        capture_error=capture_error,
    )
    _atomic_json_new(config.capture_dir / TERMINAL_FILENAME, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-bin", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-admission-current", type=Path, required=True)
    parser.add_argument("--source-revalidation", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--compiler-trace-current", type=Path, required=True)
    parser.add_argument("--route-capture-current", type=Path, required=True)
    parser.add_argument("--preparation-current", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=(MODE_CPU, MODE_DEVICE), required=True)
    parser.add_argument("--cpu-oracle-outer-receipt", type=Path)
    parser.add_argument("--lease-receipt", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    config = LaunchConfig(
        probe_bin=parsed.probe_bin,
        candidate_manifest=parsed.candidate_manifest,
        candidate_admission_current=parsed.candidate_admission_current,
        source_revalidation=parsed.source_revalidation,
        selection_receipt=parsed.selection_receipt,
        source_snapshot=parsed.source_snapshot,
        compiler_trace_current=parsed.compiler_trace_current,
        route_capture_current=parsed.route_capture_current,
        preparation_current=parsed.preparation_current,
        capture_dir=parsed.capture_dir,
        mode=parsed.mode,
        timeout_seconds=parsed.timeout_seconds,
        cpu_oracle_outer_receipt=parsed.cpu_oracle_outer_receipt,
        lease_receipt=parsed.lease_receipt,
    )
    try:
        receipt = run_attempt(config)
    except SparseGateUpParityLauncherError as exc:
        print(json.dumps({"status": "REFUSED_QWEN30_HQ30GR2_SPARSE_GATE_UP_OUTER_LAUNCHER_ERROR", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if _terminal_success(receipt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
