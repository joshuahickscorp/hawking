"""One-shot outer launcher for the HQ30GR2 current-trace diagnostic.

The separately compiled child must consume the exact sealed
``literal_hawking`` source-template trace, execute 369 full native forwards
for each scalar-control/typed-candidate body, then one forced shared
continuation for each body (740 complete forwards total), and leave durable
receipt-last evidence. The outer launcher validates the compiled child,
CPU/disk preflight, component parity, every upstream record, and the fresh
quiet lease before it starts one isolated process group. It never retries a
capture directory.

It is not a production Qwen30 launcher, a server/HCLI adapter, a benchmark,
or a promotion mechanism.  In particular, an outer success remains an
unqualified diagnostic result and cannot establish coherence, HCLI, TPS/TG,
capability, or tournament evidence.
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

SCHEMA = "hawking.ascension.qwen30_hq30gr2_all_layer_current_trace_outer_launcher.v1"
INPUT_SCHEMA = "hawking.ascension.qwen30_hq30gr2_all_layer_current_trace_input_contract.v1"
ACTIVE_FILENAME = "active.json"
CHILD_FILENAME = "child.json"
INPUT_FILENAME = "diagnostic-input-contract.json"
TERMINAL_FILENAME = "outer-terminal-receipt.json"
OUTER_STDOUT = "outer.stdout.log"
OUTER_STDERR = "outer.stderr.log"
INNER_CAPTURE = "inner"

MODE = "metal-diagnostic"
EXPECTED_PROBE_BASENAME = "ascension_qwen30_quality_repack_all_layer_current_trace_diagnostic"
EXPECTED_INNER_SCHEMA = "hawking.ascension.qwen30_quality_repack_all_layer_current_trace_diagnostic.v1"
EXPECTED_INNER_STATUS = (
    "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_TYPED_HQ30GR2_ALL_LAYER_CURRENT_TRACE_UNQUALIFIED"
)

CANDIDATE_SCHEMA = "hawking.ascension.qwen30_quality_repack_candidate.v1"
CANDIDATE_STATUS = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED"
ADMISSION_CURRENT_SCHEMA = "hawking.ascension.qwen30_quality_repack_native_admission_current_pointer.v1"
ADMISSION_CURRENT_STATUS = "CURRENT_QUALITY_REPACK_NATIVE_ADMISSION_RECEIPT_SELECTED"
ADMISSION_RECEIPT_SCHEMA = "hawking.ascension.qwen30_quality_repack_native_admission_receipt.v1"
ADMISSION_RECEIPT_STATUS = (
    "EARNED_QUALITY_REPACK_COMPLETE_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
)
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
PREPARATION_CURRENT_SCHEMA = "hawking.ascension.qwen30_quality_repack_all_layer_current_trace_prepare_current.v1"
PREPARATION_CURRENT_STATUS = "CURRENT_QWEN30_HQ30GR2_ALL_LAYER_CURRENT_TRACE_PREPARATION_SELECTED"
PREPARATION_SCHEMA = "hawking.ascension.qwen30_quality_repack_all_layer_current_trace_prepare.v1"
PREPARATION_STATUS = "PREPARED_CURRENT_TRACE_TYPED_HQ30GR2_ALL_LAYER_DIAGNOSTIC_NOT_RUN"
CONTROL_MANIFEST_SCHEMA = "hawking.ascension.qwen30_complete_binary_gravity.v1"
CONTROL_MANIFEST_STATUS = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED"
CONTROL_RUNTIME_SCHEMA = "hawking.ascension.physical_exact_full_token_runtime.v1"
CONTROL_RUNTIME_STATUS = "PASS_EXACT_NATIVE_FULL_TOKEN_RUNTIME"
CPU_PREFLIGHT_SCHEMA = "hawking.ascension.qwen30_quality_repack_all_layer_current_trace_cpu_preflight.v1"
CPU_PREFLIGHT_STATUS = "EARNED_TYPED_HQ30GR2_ALL_LAYER_CURRENT_TRACE_PREMETAL_BINDING_ONLY"
COMPONENT_CURRENT_SCHEMA = "hawking.ascension.qwen30_hq30gr2_sparse_gate_up_component_parity_current.v1"
COMPONENT_CURRENT_STATUS = "CURRENT_QWEN30_HQ30GR2_SPARSE_GATE_UP_COMPONENT_CPU_DEVICE_PARITY_SELECTED"
COMPONENT_OUTER_SCHEMA = "hawking.ascension.qwen30_hq30gr2_sparse_gate_up_device_parity_outer_launcher.v1"
COMPONENT_OUTER_STATUS = "CAPTURED_QWEN30_HQ30GR2_SPARSE_GATE_UP_DEVICE_PARITY_OUTER_TERMINAL_COMPONENT_ONLY"

# This launcher is intentionally tied to the one prepared HQ30GR2 experiment.
# A future candidate/trace must earn a separately reviewed launcher rather than
# silently inheriting this lease protocol.
PINNED_CANDIDATE_MANIFEST_SEAL = "883c59eec0371ebb6d4a9935cdbdc6bcb486c03eebd5312db608a0415a34911f"
PINNED_ADMISSION_RECEIPT_SEAL = "d7645a66d9c682bd4b0b1c0fb7fef86276678f7270862fc92db65e4d4a92c73b"
PINNED_COMPILER_TRACE_SEAL = "e698ebc2d405c70a2f6a2df39deaff800efefa9470a8f3efed644855da43a87a"
PINNED_ROUTE_CAPTURE_SEAL = "e60de4072af92f8ecc56b6a9353a2c6ae077fb4ffd6cb8939b9df5f9360feeca"
PINNED_PREPARATION_SEAL = "6ecbbca84706f280ef24d7c76e9ae816b482ef65f5d9263c54ca429f3f439487"
PINNED_CPU_PREFLIGHT_SEAL = "948427f3b4d1661a6adddd73b5293da45e709545df00ddc5b73d943fff8350b8"
PINNED_COMPONENT_CURRENT_SEAL = "3f6f02e570c872efa46df6113775482c0618b514f1250156dae34aa9d253693d"
PINNED_EXECUTOR_SHA256 = "ab680e53638b66a2c675d03a59a144c01bc03154420709daa4b3ce11ad1940b9"

QUIET_LEASE_SCHEMA = "hawking.ascension.qwen30_hq30gr2_all_layer_quiet_diagnostic_lease.v1"
QUIET_LEASE_STATUS = "GRANTED_QWEN30_HQ30GR2_ALL_LAYER_CURRENT_TRACE_DIAGNOSTIC_NON_TIMED_LEASE"
QUIET_LEASE_COMPONENT = "qwen30_hq30gr2_all_layer_current_trace_diagnostic"

TARGET_PROBE = "literal_hawking"
TARGET_TOKEN_COUNT = 369
TARGET_POSITION = 337
TARGET_LAYER_COUNT = 48
FORCED_CONTINUATION_FORWARDS_PER_PATH = 1
TOTAL_FULL_TOKEN_FORWARDS_PER_PATH = TARGET_TOKEN_COUNT + FORCED_CONTINUATION_FORWARDS_PER_PATH
TOTAL_FULL_TOKEN_FORWARDS = 2 * TOTAL_FULL_TOKEN_FORWARDS_PER_PATH


class Qwen30AllLayerProbeLauncherError(RuntimeError):
    """The bounded all-layer diagnostic cannot safely continue."""


@dataclass(frozen=True)
class LaunchConfig:
    """Inputs for one future typed HQ30GR2 all-layer device diagnostic."""

    probe_bin: Path
    candidate_manifest: Path
    candidate_admission_current: Path
    compiler_trace_current: Path
    route_capture_current: Path
    preparation_current: Path
    cpu_preflight_receipt: Path
    component_parity_current: Path
    control_manifest: Path
    control_runtime_receipt: Path
    lease_receipt: Path | None
    capture_dir: Path
    workers: int
    timeout_seconds: float


@dataclass(frozen=True)
class TraceContract:
    """The exact pre-execution trace selected for the bounded diagnostic."""

    annotated_trace: dict[str, Any]
    token_ids: list[int]
    token_ids_u32le_sha256: str
    selected_context_span_count: int


@dataclass(frozen=True)
class LaunchContext:
    """Immutable evidence read before the child is allowed to start."""

    probe_binary: dict[str, Any]
    candidate_manifest: dict[str, Any]
    candidate_manifest_seal_sha256: str
    candidate_admission_current: dict[str, Any]
    candidate_admission_pointer_seal_sha256: str
    candidate_admission_receipt: dict[str, Any]
    candidate_admission_receipt_seal_sha256: str
    compiler_trace_current: dict[str, Any]
    compiler_trace_current_seal_sha256: str
    compiler_trace_receipt: dict[str, Any]
    compiler_trace_seal_sha256: str
    route_capture_current: dict[str, Any]
    route_capture_current_seal_sha256: str
    route_capture_receipt: dict[str, Any]
    route_capture_seal_sha256: str
    preparation_current: dict[str, Any]
    preparation_current_seal_sha256: str
    preparation_receipt: dict[str, Any]
    preparation_seal_sha256: str
    cpu_preflight_receipt: dict[str, Any]
    cpu_preflight_seal_sha256: str
    component_parity_current: dict[str, Any]
    component_parity_current_seal_sha256: str
    component_parity_terminal: dict[str, Any]
    component_parity_terminal_seal_sha256: str
    control_manifest: dict[str, Any]
    control_manifest_seal_sha256: str
    control_runtime_receipt: dict[str, Any]
    control_runtime_seal_sha256: str
    trace_contract: TraceContract
    lease_receipt: dict[str, Any]
    lease_seal_sha256: str


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


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    try:
        payload = b"".join(int(token).to_bytes(4, "little", signed=False) for token in token_ids)
    except OverflowError as exc:
        raise Qwen30AllLayerProbeLauncherError("source token IDs are outside unsigned u32") from exc
    return _sha256_bytes(payload)


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
        raise Qwen30AllLayerProbeLauncherError(f"{label} must be a non-empty string")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Qwen30AllLayerProbeLauncherError(f"{label} must be an object")
    return dict(value)


def _require_absolute(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise Qwen30AllLayerProbeLauncherError(f"{label} must be absolute: {path}")


def _canonical_regular(path: Path, label: str, *, executable: bool = False) -> Path:
    _require_absolute(path, label)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise Qwen30AllLayerProbeLauncherError(f"cannot stat {label} {path}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise Qwen30AllLayerProbeLauncherError(f"{label} must be a regular non-symlink file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise Qwen30AllLayerProbeLauncherError(f"{label} must be executable: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as exc:
        raise Qwen30AllLayerProbeLauncherError(f"cannot canonicalize {label} {path}: {exc}") from exc


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
        raise Qwen30AllLayerProbeLauncherError(f"cannot read JSON {label} at {path}: {exc}") from exc
    return _mapping(value, f"JSON {label} at {path}")


def _sealed_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    canonical = _canonical_regular(path, label)
    document = _read_json(canonical, label)
    try:
        verify(document, label=str(canonical))
    except ValueError as exc:
        raise Qwen30AllLayerProbeLauncherError(f"{label} is not a valid sealed receipt: {exc}") from exc
    seal_sha256 = document.get("seal_sha256")
    if not _is_sha256(seal_sha256):
        raise Qwen30AllLayerProbeLauncherError(f"{label} has no lowercase SHA-256 seal")
    return document, str(seal_sha256)


def _assert_file_reference(
    reference: object,
    evidence: Mapping[str, Any],
    label: str,
    *,
    require_document_sha256: bool = True,
) -> None:
    row = _mapping(reference, label)
    if _canonical_from_document(row.get("path"), f"{label}.path") != Path(str(evidence["path"])):
        raise Qwen30AllLayerProbeLauncherError(f"{label} path drifted")
    # Older evidence rows call this the file ``sha256`` while pointer records
    # call it ``document_sha256``.  Both name the exact immutable JSON bytes;
    # do not allow a conflicting explicit document digest to fall back.
    observed_digest = row.get("document_sha256")
    if observed_digest is None:
        observed_digest = row.get("sha256")
    if require_document_sha256 and observed_digest != evidence["sha256"]:
        raise Qwen30AllLayerProbeLauncherError(f"{label} document SHA-256 drifted")


def _assert_seal(value: object, expected: str, label: str) -> None:
    if value != expected:
        raise Qwen30AllLayerProbeLauncherError(f"{label} seal is not the pinned HQ30GR2 evidence")


def _safe_relative_path(value: object, label: str) -> Path:
    raw = _text(value, label)
    candidate = PurePath(raw)
    if candidate.is_absolute() or ".." in candidate.parts or raw in {".", ""}:
        raise Qwen30AllLayerProbeLauncherError(f"{label} must be a contained relative path")
    return Path(candidate)


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    """Create one durable JSON document without replacing prior evidence."""

    if path.exists():
        raise Qwen30AllLayerProbeLauncherError(f"refusing to overwrite {path}")
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
        raise Qwen30AllLayerProbeLauncherError(f"refusing to overwrite {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _select_current(
    pointer_path: Path,
    *,
    label: str,
    pointer_schema: str,
    pointer_status: str,
    field: str,
    receipt_schema: str,
    receipt_status: str,
    pinned_receipt_seal: str,
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any], dict[str, Any], str]:
    """Resolve a sealed current pointer while refusing substituted evidence."""

    pointer, pointer_seal = _sealed_json(pointer_path, f"{label} current pointer")
    if pointer.get("schema") != pointer_schema or pointer.get("status") != pointer_status:
        raise Qwen30AllLayerProbeLauncherError(f"{label} current pointer schema/status drifted")
    selected = _mapping(pointer.get(field), f"{label} current pointer.{field}")
    receipt_path = _canonical_from_document(selected.get("path"), f"{label} selected receipt.path")
    receipt, receipt_seal = _sealed_json(receipt_path, f"{label} selected receipt")
    if selected.get("seal_sha256") != receipt_seal:
        raise Qwen30AllLayerProbeLauncherError(f"{label} pointer receipt seal differs from file")
    _assert_seal(receipt_seal, pinned_receipt_seal, label)
    if receipt.get("schema") != receipt_schema or receipt.get("status") != receipt_status:
        raise Qwen30AllLayerProbeLauncherError(f"{label} selected receipt schema/status drifted")
    return (
        _file_evidence(pointer_path, f"{label} current pointer"),
        pointer,
        pointer_seal,
        _file_evidence(receipt_path, f"{label} selected receipt"),
        receipt,
        receipt_seal,
    )


def _bind_candidate(
    candidate_manifest_path: Path, admission_current_path: Path
) -> tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any], str]:
    candidate, candidate_seal = _sealed_json(candidate_manifest_path, "--candidate-manifest")
    if candidate.get("schema") != CANDIDATE_SCHEMA or candidate.get("status") != CANDIDATE_STATUS:
        raise Qwen30AllLayerProbeLauncherError("--candidate-manifest schema/status drifted")
    _assert_seal(candidate_seal, PINNED_CANDIDATE_MANIFEST_SEAL, "candidate manifest")
    candidate_evidence = _file_evidence(candidate_manifest_path, "--candidate-manifest")

    pointer, pointer_seal = _sealed_json(admission_current_path, "--candidate-admission-current")
    if (
        pointer.get("schema") != ADMISSION_CURRENT_SCHEMA
        or pointer.get("status") != ADMISSION_CURRENT_STATUS
    ):
        raise Qwen30AllLayerProbeLauncherError("--candidate-admission-current schema/status drifted")
    selected_manifest = _mapping(pointer.get("complete_manifest"), "candidate admission complete_manifest")
    _assert_file_reference(selected_manifest, candidate_evidence, "candidate admission manifest")
    if selected_manifest.get("seal_sha256") != candidate_seal:
        raise Qwen30AllLayerProbeLauncherError("candidate admission manifest seal drifted")
    if selected_manifest.get("schema") != CANDIDATE_SCHEMA or selected_manifest.get("status") != CANDIDATE_STATUS:
        raise Qwen30AllLayerProbeLauncherError("candidate admission manifest grammar drifted")
    admission_reference = _mapping(pointer.get("admission_receipt"), "candidate admission receipt reference")
    admission_path = _canonical_from_document(admission_reference.get("path"), "candidate admission receipt.path")
    admission, admission_seal = _sealed_json(admission_path, "candidate admission receipt")
    if admission_reference.get("seal_sha256") != admission_seal:
        raise Qwen30AllLayerProbeLauncherError("candidate admission receipt seal differs from pointer")
    _assert_seal(admission_seal, PINNED_ADMISSION_RECEIPT_SEAL, "candidate admission")
    if admission.get("schema") != ADMISSION_RECEIPT_SCHEMA or admission.get("status") != ADMISSION_RECEIPT_STATUS:
        raise Qwen30AllLayerProbeLauncherError("candidate admission receipt schema/status drifted")
    admission_manifest = _mapping(admission.get("complete_manifest"), "candidate admission receipt complete_manifest")
    _assert_file_reference(admission_manifest, candidate_evidence, "candidate admission receipt manifest")
    if admission_manifest.get("seal_sha256") != candidate_seal:
        raise Qwen30AllLayerProbeLauncherError("candidate admission receipt manifest seal drifted")
    return (
        candidate_evidence,
        candidate_seal,
        _file_evidence(admission_current_path, "--candidate-admission-current"),
        pointer_seal,
        _file_evidence(admission_path, "candidate admission receipt"),
        admission_seal,
    )


def _bind_compiler_trace(
    current_path: Path,
    *,
    candidate_manifest: Mapping[str, Any],
    candidate_manifest_seal: str,
    admission_current: Mapping[str, Any],
    admission_pointer_seal: str,
) -> tuple[dict[str, Any], str, dict[str, Any], str, TraceContract, dict[str, Any]]:
    current_evidence, _, current_seal, receipt_evidence, receipt, receipt_seal = _select_current(
        current_path,
        label="compiler trace",
        pointer_schema=COMPILER_CURRENT_SCHEMA,
        pointer_status=COMPILER_CURRENT_STATUS,
        field="compiler_trace_receipt",
        receipt_schema=COMPILER_SCHEMA,
        receipt_status=COMPILER_STATUS,
        pinned_receipt_seal=PINNED_COMPILER_TRACE_SEAL,
    )
    binding = _mapping(receipt.get("binding"), "compiler trace binding")
    candidate_ref = _mapping(binding.get("candidate_manifest"), "compiler trace candidate manifest")
    _assert_file_reference(candidate_ref, candidate_manifest, "compiler trace candidate manifest")
    if candidate_ref.get("seal_sha256") != candidate_manifest_seal:
        raise Qwen30AllLayerProbeLauncherError("compiler trace candidate manifest seal drifted")
    admission_ref = _mapping(binding.get("candidate_native_admission"), "compiler trace candidate native admission")
    if _canonical_from_document(admission_ref.get("current_pointer_path"), "compiler trace admission pointer") != Path(
        str(admission_current["path"])
    ):
        raise Qwen30AllLayerProbeLauncherError("compiler trace admission pointer path drifted")
    if admission_ref.get("current_pointer_seal_sha256") != admission_pointer_seal:
        raise Qwen30AllLayerProbeLauncherError("compiler trace admission pointer seal drifted")
    source_snapshot = _mapping(binding.get("source_snapshot"), "compiler trace source snapshot")
    source_revalidation = _mapping(
        source_snapshot.get("immutable_source_revalidation"), "compiler trace source revalidation"
    )
    if not _is_sha256(source_revalidation.get("seal_sha256")) or not _text(
        source_revalidation.get("source_revision"), "compiler trace source revision"
    ):
        raise Qwen30AllLayerProbeLauncherError("compiler trace source revalidation is incomplete")
    run_root = Path(_text(binding.get("run_root"), "compiler trace run root"))
    if not run_root.is_absolute() or not run_root.is_dir():
        raise Qwen30AllLayerProbeLauncherError("compiler trace run root is absent or not absolute")
    rows = receipt.get("public_probe_compiler_traces")
    if not isinstance(rows, list):
        raise Qwen30AllLayerProbeLauncherError("compiler trace lacks public probe rows")
    selected_rows = [row for row in rows if isinstance(row, Mapping) and row.get("probe_id") == TARGET_PROBE]
    if len(selected_rows) != 1:
        raise Qwen30AllLayerProbeLauncherError("compiler trace must contain exactly one literal_hawking row")
    selected = dict(selected_rows[0])
    annotated_relative = _safe_relative_path(
        selected.get("annotated_trace_path"), "literal_hawking annotated trace path"
    )
    annotated_path = run_root / annotated_relative
    annotated_evidence = _file_evidence(annotated_path, "literal_hawking annotated trace")
    if selected.get("annotated_trace_sha256") != annotated_evidence["sha256"]:
        raise Qwen30AllLayerProbeLauncherError("literal_hawking annotated trace hash drifted")
    annotated = _read_json(annotated_path, "literal_hawking annotated trace")
    if annotated.get("schema") != ANNOTATED_TRACE_SCHEMA or annotated.get("status") != ANNOTATED_TRACE_STATUS:
        raise Qwen30AllLayerProbeLauncherError("literal_hawking annotated trace schema/status drifted")
    compiler = _mapping(annotated.get("compiler_trace"), "literal_hawking compiler trace")
    if (
        compiler.get("status") != ANNOTATED_TRACE_STATUS
        or compiler.get("model_execution_started") is not False
        or compiler.get("capture_timing") != "AFTER_CONTEXT_COMPILATION_BEFORE_PROVIDER_OR_MODEL_EXECUTION"
    ):
        raise Qwen30AllLayerProbeLauncherError("literal_hawking compiler trace is not pre-execution diagnostic evidence")
    annotations = _mapping(annotated.get("source_tokenizer_annotations"), "literal_hawking tokenizer annotations")
    prompt = _mapping(annotations.get("source_one_user_native_prompt"), "literal_hawking source prompt")
    token_ids = prompt.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != TARGET_TOKEN_COUNT
        or any(not isinstance(token, int) or token < 0 or token > 0xFFFFFFFF for token in token_ids)
    ):
        raise Qwen30AllLayerProbeLauncherError("literal_hawking does not have the exact 369 source-template IDs")
    token_hash = _token_ids_sha256(token_ids)
    if prompt.get("token_ids_u32le_sha256") != token_hash or prompt.get("add_special_tokens") is not True:
        raise Qwen30AllLayerProbeLauncherError("literal_hawking source-template token binding drifted")
    selected_spans = annotations.get("selected_context_spans")
    if not isinstance(selected_spans, list) or not selected_spans:
        raise Qwen30AllLayerProbeLauncherError("literal_hawking has no persisted compiler-selected spans")
    return (
        current_evidence,
        current_seal,
        receipt_evidence,
        receipt_seal,
        TraceContract(
            annotated_trace=annotated_evidence,
            token_ids=list(token_ids),
            token_ids_u32le_sha256=token_hash,
            selected_context_span_count=len(selected_spans),
        ),
        source_revalidation,
    )


def _bind_route_capture(
    current_path: Path,
    *,
    compiler_trace: Mapping[str, Any],
    compiler_trace_seal: str,
    control_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any]]:
    current_evidence, _, current_seal, receipt_evidence, receipt, receipt_seal = _select_current(
        current_path,
        label="L0 route capture",
        pointer_schema=ROUTE_CURRENT_SCHEMA,
        pointer_status=ROUTE_CURRENT_STATUS,
        field="route_capture_receipt",
        receipt_schema=ROUTE_SCHEMA,
        receipt_status=ROUTE_STATUS,
        pinned_receipt_seal=PINNED_ROUTE_CAPTURE_SEAL,
    )
    binding = _mapping(receipt.get("binding"), "L0 route capture binding")
    trace_ref = _mapping(binding.get("compiler_trace"), "L0 route capture compiler trace")
    # The sealed route record predates document-byte bindings for this edge;
    # its immutable receipt seal is checked immediately below.
    _assert_file_reference(
        trace_ref, compiler_trace, "L0 route capture compiler trace", require_document_sha256=False
    )
    if trace_ref.get("seal_sha256") != compiler_trace_seal:
        raise Qwen30AllLayerProbeLauncherError("L0 route capture compiler trace seal drifted")
    control = _mapping(binding.get("baseline_direct_packed_control"), "L0 route capture control binding")
    if _canonical_from_document(control.get("manifest_path"), "L0 route capture control manifest") != Path(
        str(control_manifest["path"])
    ):
        raise Qwen30AllLayerProbeLauncherError("L0 route capture control manifest path drifted")
    if control.get("manifest_seal_sha256") != control_manifest["seal_sha256"]:
        raise Qwen30AllLayerProbeLauncherError("L0 route capture control manifest seal drifted")
    source_revision = _text(control.get("source_revision"), "L0 route capture source revision")
    rows = receipt.get("probe_summary")
    if not isinstance(rows, list):
        raise Qwen30AllLayerProbeLauncherError("L0 route capture has no probe summary")
    selected_rows = [row for row in rows if isinstance(row, Mapping) and row.get("probe_id") == TARGET_PROBE]
    if len(selected_rows) != 1:
        raise Qwen30AllLayerProbeLauncherError("L0 route capture must contain exactly one literal_hawking summary")
    selected = dict(selected_rows[0])
    if (
        selected.get("source_template_token_count") != TARGET_TOKEN_COUNT
        or selected.get("route_membership_and_hidden_steps") != TARGET_TOKEN_COUNT
        or selected.get("l0_expert0_selected_position_count") != 1
        or selected.get("l0_expert0_selected_positions") != [TARGET_POSITION]
    ):
        raise Qwen30AllLayerProbeLauncherError("L0 route capture no longer binds the selected literal_hawking target")
    payloads = selected.get("hidden_payloads")
    if not isinstance(payloads, list):
        raise Qwen30AllLayerProbeLauncherError("L0 route capture lacks literal_hawking hidden payloads")
    expected_relative = f"hidden/{TARGET_PROBE}/{TARGET_POSITION:06d}.f32le"
    hidden = next(
        (row for row in payloads if isinstance(row, Mapping) and row.get("relative_path") == expected_relative),
        None,
    )
    if not isinstance(hidden, Mapping) or not _is_sha256(hidden.get("sha256")):
        raise Qwen30AllLayerProbeLauncherError("L0 route capture selected hidden payload is absent")
    return current_evidence, current_seal, receipt_evidence, receipt_seal, {
        "source_revision": source_revision,
        "selected_hidden_payload": dict(hidden),
    }


def _bind_preparation(
    current_path: Path,
    *,
    candidate_manifest_seal: str,
    candidate_admission_current: Mapping[str, Any],
    candidate_admission_pointer_seal: str,
    route_current: Mapping[str, Any],
    route_current_seal: str,
    route_receipt: Mapping[str, Any],
    route_receipt_seal: str,
    route_target: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    current_evidence, _, current_seal, receipt_evidence, receipt, receipt_seal = _select_current(
        current_path,
        label="all-layer preparation",
        pointer_schema=PREPARATION_CURRENT_SCHEMA,
        pointer_status=PREPARATION_CURRENT_STATUS,
        field="preparation_receipt",
        receipt_schema=PREPARATION_SCHEMA,
        receipt_status=PREPARATION_STATUS,
        pinned_receipt_seal=PINNED_PREPARATION_SEAL,
    )
    binding = _mapping(receipt.get("binding"), "all-layer preparation binding")
    if binding.get("candidate_manifest_seal_sha256") != candidate_manifest_seal:
        raise Qwen30AllLayerProbeLauncherError("all-layer preparation candidate manifest seal drifted")
    admission_ref = _mapping(
        binding.get("candidate_admission_current_pointer"), "all-layer preparation candidate admission"
    )
    if _canonical_from_document(admission_ref.get("path"), "all-layer preparation admission pointer") != Path(
        str(candidate_admission_current["path"])
    ) or admission_ref.get("seal_sha256") != candidate_admission_pointer_seal:
        raise Qwen30AllLayerProbeLauncherError("all-layer preparation candidate admission binding drifted")
    route_current_ref = _mapping(binding.get("route_capture_current_pointer"), "all-layer preparation route pointer")
    if _canonical_from_document(route_current_ref.get("path"), "all-layer preparation route pointer") != Path(
        str(route_current["path"])
    ) or route_current_ref.get("seal_sha256") != route_current_seal:
        raise Qwen30AllLayerProbeLauncherError("all-layer preparation route current binding drifted")
    route_ref = _mapping(binding.get("route_capture_receipt"), "all-layer preparation route receipt")
    # The already sealed preparation points to the immutable route receipt by
    # path + seal, not a duplicate document-byte hash.
    _assert_file_reference(
        route_ref, route_receipt, "all-layer preparation route receipt", require_document_sha256=False
    )
    if route_ref.get("seal_sha256") != route_receipt_seal:
        raise Qwen30AllLayerProbeLauncherError("all-layer preparation route receipt seal drifted")
    planned = _mapping(receipt.get("planned_bounded_input"), "all-layer preparation bounded input")
    expected_hidden = route_target["selected_hidden_payload"]
    if (
        planned.get("probe_id") != TARGET_PROBE
        or planned.get("source_template_token_count") != TARGET_TOKEN_COUNT
        or planned.get("l0_e0_selected_position") != TARGET_POSITION
        or planned.get("l0_e0_router_input_hidden") != expected_hidden
    ):
        raise Qwen30AllLayerProbeLauncherError("all-layer preparation bounded literal_hawking input drifted")
    execution = _mapping(receipt.get("planned_execution_not_run"), "all-layer preparation execution contract")
    if (
        execution.get("one_existing_trace_only") != TARGET_PROBE
        or execution.get("prefill")
        != "exact 369 source-template IDs through all 48 layers for baseline and candidate separately"
        or execution.get("one_bounded_continuation")
        != "derive baseline deterministic argmax after the exact prefix; force that same one token into both paths for one additional 48-layer forward"
    ):
        raise Qwen30AllLayerProbeLauncherError("all-layer preparation prefill/forced-continuation contract drifted")
    forbidden = execution.get("explicitly_not_run_or_claimed")
    if not isinstance(forbidden, list) or not {
        "HCLI endpoint",
        "chat coherence",
        "TPS, TG, capability, manager, or tournament",
    }.issubset(set(forbidden)):
        raise Qwen30AllLayerProbeLauncherError("all-layer preparation claim boundary drifted")
    runtime_contract = _mapping(receipt.get("candidate_runtime_contract"), "all-layer preparation runtime contract")
    if (
        runtime_contract.get("new_runtime_type_required") != "Qwen30QualityRepackNativeDiagnosticRuntime"
        or runtime_contract.get("live_control_runtime_reuse_forbidden")
        != "Qwen30CompleteNativeRuntime only accepts CompleteBinaryArtifact/direct headers"
    ):
        raise Qwen30AllLayerProbeLauncherError("all-layer preparation runtime separation drifted")
    source_readiness = _mapping(binding.get("source_readiness"), "all-layer preparation source readiness")
    live_control = _mapping(source_readiness.get("live_qwen30_runtime_source"), "all-layer live control source")
    typed_diagnostic = _mapping(
        source_readiness.get("typed_candidate_diagnostic_source"), "all-layer typed diagnostic source"
    )
    if (
        live_control.get("candidate_type_absent_from_live_runtime") is not True
        or typed_diagnostic.get("typed_catalog_and_sparse_gate_up_host_dispatch_build_ready") is not True
    ):
        raise Qwen30AllLayerProbeLauncherError("all-layer preparation source/control separation is absent")
    return current_evidence, current_seal, receipt_evidence, receipt_seal


def _bind_component_parity_current(
    current_path: Path,
    *,
    candidate_manifest: Mapping[str, Any],
    candidate_manifest_seal: str,
    candidate_admission_current: Mapping[str, Any],
    candidate_admission_pointer_seal: str,
    candidate_admission_receipt: Mapping[str, Any],
    candidate_admission_receipt_seal: str,
    compiler_trace: Mapping[str, Any],
    compiler_trace_seal: str,
    route_capture: Mapping[str, Any],
    route_capture_seal: str,
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    """Bind the only admitted sparse L0/E0 device-parity component result.

    The component receipt is not a layer/runtime result. It is required here
    solely to prove that the typed candidate executor cannot begin its larger
    all-layer comparison using an unvalidated sparse-device path.
    """

    current, current_seal = _sealed_json(current_path, "--component-parity-current")
    if (
        current.get("schema") != COMPONENT_CURRENT_SCHEMA
        or current.get("status") != COMPONENT_CURRENT_STATUS
    ):
        raise Qwen30AllLayerProbeLauncherError("--component-parity-current schema/status drifted")
    _assert_seal(current_seal, PINNED_COMPONENT_CURRENT_SEAL, "component parity current")

    _assert_file_reference(
        current.get("candidate_manifest"), candidate_manifest, "component parity candidate manifest"
    )
    if current.get("candidate_manifest_seal_sha256") != candidate_manifest_seal:
        raise Qwen30AllLayerProbeLauncherError("component parity candidate manifest seal drifted")
    outer_ref = current.get("component_parity_outer_terminal")
    _assert_file_reference(outer_ref, _file_evidence(_canonical_from_document(
        _mapping(outer_ref, "component parity outer terminal").get("path"),
        "component parity outer terminal.path",
    ), "component parity outer terminal"), "component parity outer terminal")
    outer_path = _canonical_from_document(
        _mapping(outer_ref, "component parity outer terminal").get("path"),
        "component parity outer terminal.path",
    )
    outer, outer_seal = _sealed_json(outer_path, "component parity outer terminal")
    if (
        outer.get("schema") != COMPONENT_OUTER_SCHEMA
        or outer.get("status") != COMPONENT_OUTER_STATUS
        or _mapping(outer_ref, "component parity outer terminal").get("seal_sha256") != outer_seal
    ):
        raise Qwen30AllLayerProbeLauncherError("component parity outer terminal schema/status/seal drifted")
    source = _mapping(outer.get("source_binding"), "component parity outer source binding")
    for label, key, evidence, expected_seal in (
        ("candidate manifest", "candidate_manifest", candidate_manifest, candidate_manifest_seal),
        ("candidate admission current", "candidate_admission_current", candidate_admission_current, candidate_admission_pointer_seal),
        ("candidate admission receipt", "candidate_admission_receipt", candidate_admission_receipt, candidate_admission_receipt_seal),
        ("compiler trace", "compiler_trace_receipt", compiler_trace, compiler_trace_seal),
        ("route capture", "route_capture_receipt", route_capture, route_capture_seal),
    ):
        reference = source.get(key)
        _assert_file_reference(reference, evidence, f"component parity {label}")
        if _mapping(reference, f"component parity {label}").get("seal_sha256") != expected_seal:
            raise Qwen30AllLayerProbeLauncherError(f"component parity {label} seal drifted")
    scope = _mapping(current.get("execution_scope"), "component parity execution scope")
    if (
        scope.get("all_layers_executed") is not False
        or scope.get("full_token_executed") is not False
        or scope.get("literal_hawking_l0_e0_gate_up_swiglu_component_only") is not True
    ):
        raise Qwen30AllLayerProbeLauncherError("component parity execution scope drifted")
    next_use = _mapping(current.get("next_use_contract"), "component parity next-use contract")
    if (
        next_use.get("only_typed_hq30gr2_all_layer_diagnostic_may_consume_this_component_receipt")
        is not True
        or next_use.get("requires_separate_fresh_all_layer_gpu_lease") is not True
    ):
        raise Qwen30AllLayerProbeLauncherError("component parity next-use contract drifted")
    return (
        _file_evidence(current_path, "--component-parity-current"),
        current_seal,
        _file_evidence(outer_path, "component parity outer terminal"),
        outer_seal,
    )


def _bind_cpu_preflight(
    preflight_path: Path,
    *,
    candidate_manifest: Mapping[str, Any],
    candidate_admission_current: Mapping[str, Any],
    candidate_admission_receipt: Mapping[str, Any],
    component_current: Mapping[str, Any],
    component_current_seal: str,
    component_terminal: Mapping[str, Any],
    component_terminal_seal: str,
    compiler_trace_current: Mapping[str, Any],
    compiler_trace: Mapping[str, Any],
    route_capture_current: Mapping[str, Any],
    route_capture: Mapping[str, Any],
    preparation_current: Mapping[str, Any],
    preparation: Mapping[str, Any],
    control_manifest: Mapping[str, Any],
    control_runtime: Mapping[str, Any],
    trace_contract: TraceContract,
) -> tuple[dict[str, Any], str]:
    """Require the exact admitted 18,867-payload CPU/disk preflight.

    This guards the one device attempt against a stale pointer set or a
    component-only result being mistaken for candidate runtime admission.
    """

    preflight, preflight_seal = _sealed_json(preflight_path, "--cpu-preflight-receipt")
    if preflight.get("schema") != CPU_PREFLIGHT_SCHEMA or preflight.get("status") != CPU_PREFLIGHT_STATUS:
        raise Qwen30AllLayerProbeLauncherError("--cpu-preflight-receipt schema/status drifted")
    _assert_seal(preflight_seal, PINNED_CPU_PREFLIGHT_SEAL, "CPU preflight")
    for label, key, evidence in (
        ("candidate manifest", "candidate_manifest", candidate_manifest),
        ("candidate admission current", "candidate_admission_current", candidate_admission_current),
        ("candidate admission receipt", "candidate_admission_receipt", candidate_admission_receipt),
        ("component parity current", "candidate_component_parity_current", component_current),
        ("component parity terminal", "candidate_component_parity_terminal", component_terminal),
        ("compiler trace current", "compiler_trace_current", compiler_trace_current),
        ("compiler trace receipt", "compiler_trace_receipt", compiler_trace),
        ("route capture current", "route_capture_current", route_capture_current),
        ("route capture receipt", "route_capture_receipt", route_capture),
        ("preparation current", "preparation_current", preparation_current),
        ("preparation receipt", "preparation_receipt", preparation),
        ("control manifest", "control_manifest", control_manifest),
        ("control runtime receipt", "control_runtime_receipt", control_runtime),
    ):
        _assert_file_reference(preflight.get(key), evidence, f"CPU preflight {label}")
    if (
        _mapping(preflight.get("candidate_component_parity_current"), "CPU preflight component current").get("seal_sha256")
        != component_current_seal
        or _mapping(preflight.get("candidate_component_parity_terminal"), "CPU preflight component terminal").get("seal_sha256")
        != component_terminal_seal
    ):
        raise Qwen30AllLayerProbeLauncherError("CPU preflight component-parity seal drifted")
    exact_trace = _mapping(preflight.get("exact_source_template_input"), "CPU preflight exact trace")
    if (
        exact_trace.get("probe_id") != TARGET_PROBE
        or exact_trace.get("token_count") != TARGET_TOKEN_COUNT
        or exact_trace.get("token_ids_u32le_sha256") != trace_contract.token_ids_u32le_sha256
        or exact_trace.get("new_diagnostic_not_historical") is not True
    ):
        raise Qwen30AllLayerProbeLauncherError("CPU preflight exact source-template trace drifted")
    _assert_file_reference(
        exact_trace.get("annotated_trace"), trace_contract.annotated_trace, "CPU preflight annotated trace"
    )
    boundary = _mapping(preflight.get("execution_boundary"), "CPU preflight execution boundary")
    if any(
        boundary.get(key) is not False
        for key in (
            "all_layer_forward_performed",
            "endpoint_or_hcli_called",
            "host_fallback_for_future_candidate_execution",
            "metal_context_created",
            "metal_dispatch_performed",
            "raw_bf16_or_dense_weight_path",
            "server_watcher_or_adapter_modified",
            "token_loop_performed",
        )
    ) or boundary.get("future_device_executor_requires_a_new_quiet_lease_and_a_new_outer_capture") is not True:
        raise Qwen30AllLayerProbeLauncherError("CPU preflight execution boundary drifted")
    typed = _mapping(preflight.get("typed_catalog_preflight"), "CPU preflight typed catalog")
    candidate_catalog = _mapping(typed.get("candidate_typed_catalog"), "CPU preflight candidate typed catalog")
    dispatch = _mapping(candidate_catalog.get("sparse_gate_up_dispatch"), "CPU preflight sparse dispatch")
    control_catalog = _mapping(typed.get("control_direct_catalog"), "CPU preflight control direct catalog")
    if (
        candidate_catalog.get("direct_tensor_count") != 18865
        or candidate_catalog.get("immutable_verified_payloads") != 18867
        or candidate_catalog.get("sparse_residual_tensor_count") != 2
        or candidate_catalog.get("l0_e0_gate_up_layout") != "HQ30GR2_SPARSE_RESIDUAL"
        or dispatch.get("direct_fallback_for_sparse_residual_forbidden") is not True
        or dispatch.get("exact_non_fma_scalar_order_required") is not True
        or control_catalog.get("immutable_verified_payloads") != 18867
        or control_catalog.get("l0_e0_gate_layout") != "HQ30G1B1_DIRECT"
    ):
        raise Qwen30AllLayerProbeLauncherError("CPU preflight typed catalog contract drifted")
    claim = _mapping(preflight.get("claim_boundary"), "CPU preflight claim boundary")
    for key in (
        "does_not_claim_capability_or_tournament",
        "does_not_claim_generation_or_coherence",
        "does_not_claim_hcli",
        "does_not_claim_native_runtime",
        "does_not_claim_tps_or_tg",
    ):
        if claim.get(key) is not True:
            raise Qwen30AllLayerProbeLauncherError(f"CPU preflight claim boundary lacks {key}=true")
    return _file_evidence(preflight_path, "--cpu-preflight-receipt"), preflight_seal


def _bind_control(
    manifest_path: Path,
    runtime_receipt_path: Path,
    *,
    route_target: Mapping[str, Any],
    compiler_source_revalidation: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    manifest, manifest_seal = _sealed_json(manifest_path, "--control-manifest")
    if manifest.get("schema") != CONTROL_MANIFEST_SCHEMA or manifest.get("status") != CONTROL_MANIFEST_STATUS:
        raise Qwen30AllLayerProbeLauncherError("--control-manifest schema/status drifted")
    manifest_evidence = _file_evidence(manifest_path, "--control-manifest")
    manifest_evidence["seal_sha256"] = manifest_seal
    runtime, runtime_seal = _sealed_json(runtime_receipt_path, "--control-runtime-receipt")
    if runtime.get("schema") != CONTROL_RUNTIME_SCHEMA or runtime.get("status") != CONTROL_RUNTIME_STATUS:
        raise Qwen30AllLayerProbeLauncherError("--control-runtime-receipt schema/status drifted")
    binding = _mapping(runtime.get("binding"), "control runtime binding")
    if binding.get("complete_manifest_seal_sha256") != manifest_seal:
        raise Qwen30AllLayerProbeLauncherError("control runtime manifest seal drifted")
    if binding.get("source_revalidation_seal_sha256") != compiler_source_revalidation.get("seal_sha256"):
        raise Qwen30AllLayerProbeLauncherError("control runtime source revalidation seal drifted")
    if route_target.get("source_revision") != _text(
        compiler_source_revalidation.get("source_revision"), "compiler source revalidation revision"
    ):
        raise Qwen30AllLayerProbeLauncherError("control route/source revision binding drifted")
    runtime_facts = _mapping(runtime.get("runtime"), "control runtime facts")
    for key in (
        "all_layers_executed",
        "all_weight_tensors_bound",
        "custom_kernel_used",
        "full_token_execution",
        "model_alone",
        "native_exact_decoder",
        "no_fallback",
        "prompt_template_bound",
        "tokenizer_bound",
    ):
        if runtime_facts.get(key) is not True:
            raise Qwen30AllLayerProbeLauncherError(f"control runtime lacks {key}=true")
    return manifest_evidence, manifest_seal, _file_evidence(
        runtime_receipt_path, "--control-runtime-receipt"
    ), runtime_seal


def _bind_lease(
    path: Path,
    *,
    candidate_manifest: Mapping[str, Any],
    candidate_manifest_seal: str,
    candidate_admission_current: Mapping[str, Any],
    candidate_admission_pointer_seal: str,
    candidate_admission_receipt_seal: str,
    compiler_trace: Mapping[str, Any],
    compiler_trace_seal: str,
    route_capture: Mapping[str, Any],
    route_capture_seal: str,
    preparation: Mapping[str, Any],
    preparation_seal: str,
    control_manifest: Mapping[str, Any],
    control_manifest_seal: str,
    control_runtime: Mapping[str, Any],
    control_runtime_seal: str,
    trace_contract: TraceContract,
    cpu_preflight: Mapping[str, Any],
    cpu_preflight_seal: str,
    component_current: Mapping[str, Any],
    component_current_seal: str,
    component_terminal: Mapping[str, Any],
    component_terminal_seal: str,
) -> tuple[dict[str, Any], str]:
    document, lease_seal = _sealed_json(path, "--lease-receipt")
    if document.get("schema") != QUIET_LEASE_SCHEMA or document.get("status") != QUIET_LEASE_STATUS:
        raise Qwen30AllLayerProbeLauncherError("--lease-receipt schema/status does not authorize HQ30GR2")
    _text(document.get("lease_id"), "Q30 diagnostic lease_id")
    _text(document.get("granted_at"), "Q30 diagnostic lease granted_at")
    lifecycle = _mapping(document.get("one_shot_lifecycle"), "Q30 diagnostic lease lifecycle")
    if (
        lifecycle.get("fresh_for_this_exact_launch") is not True
        or lifecycle.get("prior_terminal_receipt") is not None
        or lifecycle.get("automatic_retry_allowed") is not False
    ):
        raise Qwen30AllLayerProbeLauncherError("--lease-receipt is not a fresh one-shot diagnostic lease")
    policy = _mapping(document.get("execution_policy"), "Q30 diagnostic lease policy")
    if (
        policy.get("component") != QUIET_LEASE_COMPONENT
        or policy.get("quiet_qwen_family_gpu_lease") is not True
        or policy.get("strict_math") is not True
        or policy.get("diagnostic_only") is not True
        or policy.get("one_child_process_group_only") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("hcli_or_server_allowed") is not False
        or policy.get("coherence_claim_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
        or policy.get("capability_claim_allowed") is not False
        or policy.get("tournament_claim_allowed") is not False
    ):
        raise Qwen30AllLayerProbeLauncherError("--lease-receipt policy is not a quiet diagnostic-only lease")
    artifact = _mapping(document.get("artifact_binding"), "Q30 diagnostic lease artifact binding")
    _assert_file_reference(artifact.get("candidate_manifest"), candidate_manifest, "lease candidate manifest")
    if artifact.get("candidate_manifest_seal_sha256") != candidate_manifest_seal:
        raise Qwen30AllLayerProbeLauncherError("lease candidate manifest seal drifted")
    if _canonical_from_document(
        artifact.get("candidate_admission_current_path"), "lease candidate admission current"
    ) != Path(str(candidate_admission_current["path"])):
        raise Qwen30AllLayerProbeLauncherError("lease candidate admission pointer path drifted")
    if (
        artifact.get("candidate_admission_pointer_seal_sha256") != candidate_admission_pointer_seal
        or artifact.get("candidate_admission_receipt_seal_sha256") != candidate_admission_receipt_seal
    ):
        raise Qwen30AllLayerProbeLauncherError("lease candidate admission binding drifted")
    _assert_file_reference(artifact.get("control_manifest"), control_manifest, "lease control manifest")
    if artifact.get("control_manifest_seal_sha256") != control_manifest_seal:
        raise Qwen30AllLayerProbeLauncherError("lease control manifest seal drifted")
    _assert_file_reference(artifact.get("control_runtime_receipt"), control_runtime, "lease control runtime")
    if artifact.get("control_runtime_receipt_seal_sha256") != control_runtime_seal:
        raise Qwen30AllLayerProbeLauncherError("lease control runtime seal drifted")
    upstream = _mapping(document.get("upstream_binding"), "Q30 diagnostic lease upstream binding")
    for label, reference, evidence, receipt_seal in (
        ("compiler trace", upstream.get("compiler_trace_receipt"), compiler_trace, compiler_trace_seal),
        ("L0 route capture", upstream.get("route_capture_receipt"), route_capture, route_capture_seal),
        ("all-layer preparation", upstream.get("preparation_receipt"), preparation, preparation_seal),
    ):
        _assert_file_reference(reference, evidence, f"lease {label}")
        if _mapping(reference, f"lease {label}").get("seal_sha256") != receipt_seal:
            raise Qwen30AllLayerProbeLauncherError(f"lease {label} seal drifted")
    readiness = _mapping(document.get("typed_candidate_readiness"), "Q30 diagnostic lease typed readiness")
    for label, key, evidence, receipt_seal in (
        ("CPU preflight", "cpu_preflight_receipt", cpu_preflight, cpu_preflight_seal),
        ("component current", "component_parity_current", component_current, component_current_seal),
        ("component terminal", "component_parity_terminal", component_terminal, component_terminal_seal),
    ):
        reference = readiness.get(key)
        _assert_file_reference(reference, evidence, f"lease {label}")
        if _mapping(reference, f"lease {label}").get("seal_sha256") != receipt_seal:
            raise Qwen30AllLayerProbeLauncherError(f"lease {label} seal drifted")
    trace = _mapping(document.get("trace_contract"), "Q30 diagnostic lease trace contract")
    if (
        trace.get("probe_id") != TARGET_PROBE
        or trace.get("source_template_token_count") != TARGET_TOKEN_COUNT
        or trace.get("source_template_token_ids_u32le_sha256") != trace_contract.token_ids_u32le_sha256
        or trace.get("forced_shared_continuation") is not True
        or trace.get("additional_forwards_per_path") != FORCED_CONTINUATION_FORWARDS_PER_PATH
        or trace.get("complete_native_forwards_per_path") != TOTAL_FULL_TOKEN_FORWARDS_PER_PATH
        or trace.get("complete_native_forwards_total") != TOTAL_FULL_TOKEN_FORWARDS
        or trace.get("complete_native_layers_traversed_total")
        != TOTAL_FULL_TOKEN_FORWARDS * TARGET_LAYER_COUNT
    ):
        raise Qwen30AllLayerProbeLauncherError("lease exact trace/continuation contract drifted")
    return _file_evidence(path, "--lease-receipt"), lease_seal


def _validate_config(config: LaunchConfig) -> LaunchContext:
    probe = _canonical_regular(config.probe_bin, "--probe-bin", executable=True)
    if probe.name != EXPECTED_PROBE_BASENAME:
        raise Qwen30AllLayerProbeLauncherError(
            f"--probe-bin must name {EXPECTED_PROBE_BASENAME}, got {probe.name!r}"
        )
    probe_evidence = _file_evidence(probe, "--probe-bin", executable=True)
    if probe_evidence["sha256"] != PINNED_EXECUTOR_SHA256:
        raise Qwen30AllLayerProbeLauncherError(
            "--probe-bin SHA-256 is not the pinned typed HQ30GR2 all-layer executor"
        )
    for path, label in (
        (config.candidate_manifest, "--candidate-manifest"),
        (config.candidate_admission_current, "--candidate-admission-current"),
        (config.compiler_trace_current, "--compiler-trace-current"),
        (config.route_capture_current, "--route-capture-current"),
        (config.preparation_current, "--preparation-current"),
        (config.cpu_preflight_receipt, "--cpu-preflight-receipt"),
        (config.component_parity_current, "--component-parity-current"),
        (config.control_manifest, "--control-manifest"),
        (config.control_runtime_receipt, "--control-runtime-receipt"),
    ):
        _canonical_regular(path, label)
    _require_absolute(config.capture_dir, "--capture-dir")
    if config.workers != 1:
        raise Qwen30AllLayerProbeLauncherError("--workers must be exactly one for the one-shot diagnostic")
    if not config.timeout_seconds > 0:
        raise Qwen30AllLayerProbeLauncherError("--timeout-seconds must be positive")
    if config.lease_receipt is None:
        raise Qwen30AllLayerProbeLauncherError("the metal diagnostic requires --lease-receipt")
    _canonical_regular(config.lease_receipt, "--lease-receipt")

    (
        candidate_manifest,
        candidate_manifest_seal,
        admission_current,
        admission_pointer_seal,
        admission_receipt,
        admission_receipt_seal,
    ) = _bind_candidate(config.candidate_manifest, config.candidate_admission_current)
    (
        compiler_current,
        compiler_current_seal,
        compiler_receipt,
        compiler_seal,
        trace_contract,
        source_revalidation,
    ) = _bind_compiler_trace(
        config.compiler_trace_current,
        candidate_manifest=candidate_manifest,
        candidate_manifest_seal=candidate_manifest_seal,
        admission_current=admission_current,
        admission_pointer_seal=admission_pointer_seal,
    )
    # Control is bound through the actual L0 route capture before it is bound to
    # the exact full-token control receipt.  This avoids inventing a new
    # candidate/control relationship in the launcher.
    control_manifest, control_manifest_seal = _sealed_json(config.control_manifest, "--control-manifest")
    if control_manifest.get("schema") != CONTROL_MANIFEST_SCHEMA or control_manifest.get("status") != CONTROL_MANIFEST_STATUS:
        raise Qwen30AllLayerProbeLauncherError("--control-manifest schema/status drifted")
    control_manifest_evidence = _file_evidence(config.control_manifest, "--control-manifest")
    control_manifest_evidence["seal_sha256"] = control_manifest_seal
    (
        route_current,
        route_current_seal,
        route_receipt,
        route_seal,
        route_target,
    ) = _bind_route_capture(
        config.route_capture_current,
        compiler_trace=compiler_receipt,
        compiler_trace_seal=compiler_seal,
        control_manifest=control_manifest_evidence,
    )
    (
        control_manifest_checked,
        control_manifest_checked_seal,
        control_runtime,
        control_runtime_seal,
    ) = _bind_control(
        config.control_manifest,
        config.control_runtime_receipt,
        route_target=route_target,
        compiler_source_revalidation=source_revalidation,
    )
    (
        preparation_current,
        preparation_current_seal,
        preparation_receipt,
        preparation_seal,
    ) = _bind_preparation(
        config.preparation_current,
        candidate_manifest_seal=candidate_manifest_seal,
        candidate_admission_current=admission_current,
        candidate_admission_pointer_seal=admission_pointer_seal,
        route_current=route_current,
        route_current_seal=route_current_seal,
        route_receipt=route_receipt,
        route_receipt_seal=route_seal,
        route_target=route_target,
    )
    (
        component_current,
        component_current_seal,
        component_terminal,
        component_terminal_seal,
    ) = _bind_component_parity_current(
        config.component_parity_current,
        candidate_manifest=candidate_manifest,
        candidate_manifest_seal=candidate_manifest_seal,
        candidate_admission_current=admission_current,
        candidate_admission_pointer_seal=admission_pointer_seal,
        candidate_admission_receipt=admission_receipt,
        candidate_admission_receipt_seal=admission_receipt_seal,
        compiler_trace=compiler_receipt,
        compiler_trace_seal=compiler_seal,
        route_capture=route_receipt,
        route_capture_seal=route_seal,
    )
    cpu_preflight, cpu_preflight_seal = _bind_cpu_preflight(
        config.cpu_preflight_receipt,
        candidate_manifest=candidate_manifest,
        candidate_admission_current=admission_current,
        candidate_admission_receipt=admission_receipt,
        component_current=component_current,
        component_current_seal=component_current_seal,
        component_terminal=component_terminal,
        component_terminal_seal=component_terminal_seal,
        compiler_trace_current=compiler_current,
        compiler_trace=compiler_receipt,
        route_capture_current=route_current,
        route_capture=route_receipt,
        preparation_current=preparation_current,
        preparation=preparation_receipt,
        control_manifest=control_manifest_checked,
        control_runtime=control_runtime,
        trace_contract=trace_contract,
    )
    assert config.lease_receipt is not None
    lease, lease_seal = _bind_lease(
        config.lease_receipt,
        candidate_manifest=candidate_manifest,
        candidate_manifest_seal=candidate_manifest_seal,
        candidate_admission_current=admission_current,
        candidate_admission_pointer_seal=admission_pointer_seal,
        candidate_admission_receipt_seal=admission_receipt_seal,
        compiler_trace=compiler_receipt,
        compiler_trace_seal=compiler_seal,
        route_capture=route_receipt,
        route_capture_seal=route_seal,
        preparation=preparation_receipt,
        preparation_seal=preparation_seal,
        control_manifest=control_manifest_checked,
        control_manifest_seal=control_manifest_checked_seal,
        control_runtime=control_runtime,
        control_runtime_seal=control_runtime_seal,
        trace_contract=trace_contract,
        cpu_preflight=cpu_preflight,
        cpu_preflight_seal=cpu_preflight_seal,
        component_current=component_current,
        component_current_seal=component_current_seal,
        component_terminal=component_terminal,
        component_terminal_seal=component_terminal_seal,
    )
    return LaunchContext(
        probe_binary=probe_evidence,
        candidate_manifest=candidate_manifest,
        candidate_manifest_seal_sha256=candidate_manifest_seal,
        candidate_admission_current=admission_current,
        candidate_admission_pointer_seal_sha256=admission_pointer_seal,
        candidate_admission_receipt=admission_receipt,
        candidate_admission_receipt_seal_sha256=admission_receipt_seal,
        compiler_trace_current=compiler_current,
        compiler_trace_current_seal_sha256=compiler_current_seal,
        compiler_trace_receipt=compiler_receipt,
        compiler_trace_seal_sha256=compiler_seal,
        route_capture_current=route_current,
        route_capture_current_seal_sha256=route_current_seal,
        route_capture_receipt=route_receipt,
        route_capture_seal_sha256=route_seal,
        preparation_current=preparation_current,
        preparation_current_seal_sha256=preparation_current_seal,
        preparation_receipt=preparation_receipt,
        preparation_seal_sha256=preparation_seal,
        cpu_preflight_receipt=cpu_preflight,
        cpu_preflight_seal_sha256=cpu_preflight_seal,
        component_parity_current=component_current,
        component_parity_current_seal_sha256=component_current_seal,
        component_parity_terminal=component_terminal,
        component_parity_terminal_seal_sha256=component_terminal_seal,
        control_manifest=control_manifest_checked,
        control_manifest_seal_sha256=control_manifest_checked_seal,
        control_runtime_receipt=control_runtime,
        control_runtime_seal_sha256=control_runtime_seal,
        trace_contract=trace_contract,
        lease_receipt=lease,
        lease_seal_sha256=lease_seal,
    )


def _launch_identity(config: LaunchConfig, context: LaunchContext) -> str:
    payload = {
        "probe_binary": context.probe_binary,
        "candidate_manifest": context.candidate_manifest,
        "candidate_manifest_seal_sha256": context.candidate_manifest_seal_sha256,
        "candidate_admission_current": context.candidate_admission_current,
        "candidate_admission_pointer_seal_sha256": context.candidate_admission_pointer_seal_sha256,
        "candidate_admission_receipt": context.candidate_admission_receipt,
        "candidate_admission_receipt_seal_sha256": context.candidate_admission_receipt_seal_sha256,
        "compiler_trace_current": context.compiler_trace_current,
        "compiler_trace_current_seal_sha256": context.compiler_trace_current_seal_sha256,
        "compiler_trace_receipt": context.compiler_trace_receipt,
        "compiler_trace_seal_sha256": context.compiler_trace_seal_sha256,
        "route_capture_current": context.route_capture_current,
        "route_capture_current_seal_sha256": context.route_capture_current_seal_sha256,
        "route_capture_receipt": context.route_capture_receipt,
        "route_capture_seal_sha256": context.route_capture_seal_sha256,
        "preparation_current": context.preparation_current,
        "preparation_current_seal_sha256": context.preparation_current_seal_sha256,
        "preparation_receipt": context.preparation_receipt,
        "preparation_seal_sha256": context.preparation_seal_sha256,
        "cpu_preflight_receipt": context.cpu_preflight_receipt,
        "cpu_preflight_seal_sha256": context.cpu_preflight_seal_sha256,
        "component_parity_current": context.component_parity_current,
        "component_parity_current_seal_sha256": context.component_parity_current_seal_sha256,
        "component_parity_terminal": context.component_parity_terminal,
        "component_parity_terminal_seal_sha256": context.component_parity_terminal_seal_sha256,
        "control_manifest": context.control_manifest,
        "control_manifest_seal_sha256": context.control_manifest_seal_sha256,
        "control_runtime_receipt": context.control_runtime_receipt,
        "control_runtime_seal_sha256": context.control_runtime_seal_sha256,
        "trace": {
            "annotated_trace": context.trace_contract.annotated_trace,
            "source_template_token_ids_u32le_sha256": context.trace_contract.token_ids_u32le_sha256,
            "source_template_token_count": len(context.trace_contract.token_ids),
            "selected_context_span_count": context.trace_contract.selected_context_span_count,
            "complete_native_forwards_per_path": TOTAL_FULL_TOKEN_FORWARDS_PER_PATH,
            "complete_native_forwards_total": TOTAL_FULL_TOKEN_FORWARDS,
            "layers_traversed_total": TOTAL_FULL_TOKEN_FORWARDS * TARGET_LAYER_COUNT,
        },
        "lease_receipt": context.lease_receipt,
        "lease_seal_sha256": context.lease_seal_sha256,
        "mode": MODE,
        "workers": config.workers,
        "timeout_seconds": config.timeout_seconds,
    }
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def _write_input_contract(
    capture_dir: Path, *, identity: str, context: LaunchContext
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Persist exact trace IDs before the child can open a device context."""

    path = capture_dir / INPUT_FILENAME
    document = seal(
        {
            "schema": INPUT_SCHEMA,
            "status": "PREPARED_EXACT_ONE_LITERAL_HAWKING_ALL_LAYER_DIAGNOSTIC_INPUT",
            "recorded_at": _utc_now(),
            "launch_identity_sha256": identity,
            "source_binding": {
                "candidate_manifest": context.candidate_manifest,
                "candidate_manifest_seal_sha256": context.candidate_manifest_seal_sha256,
                "candidate_admission_current": context.candidate_admission_current,
                "candidate_admission_pointer_seal_sha256": context.candidate_admission_pointer_seal_sha256,
                "candidate_admission_receipt": context.candidate_admission_receipt,
                "candidate_admission_receipt_seal_sha256": context.candidate_admission_receipt_seal_sha256,
                "compiler_trace_receipt": context.compiler_trace_receipt,
                "compiler_trace_seal_sha256": context.compiler_trace_seal_sha256,
                "route_capture_receipt": context.route_capture_receipt,
                "route_capture_seal_sha256": context.route_capture_seal_sha256,
                "preparation_receipt": context.preparation_receipt,
                "preparation_seal_sha256": context.preparation_seal_sha256,
                "cpu_preflight_receipt": context.cpu_preflight_receipt,
                "cpu_preflight_seal_sha256": context.cpu_preflight_seal_sha256,
                "component_parity_current": context.component_parity_current,
                "component_parity_current_seal_sha256": context.component_parity_current_seal_sha256,
                "component_parity_terminal": context.component_parity_terminal,
                "component_parity_terminal_seal_sha256": context.component_parity_terminal_seal_sha256,
                "control_manifest": context.control_manifest,
                "control_manifest_seal_sha256": context.control_manifest_seal_sha256,
                "control_runtime_receipt": context.control_runtime_receipt,
                "control_runtime_seal_sha256": context.control_runtime_seal_sha256,
                "lease_receipt": context.lease_receipt,
                "lease_seal_sha256": context.lease_seal_sha256,
            },
            "exact_trace": {
                "probe_id": TARGET_PROBE,
                "source_template_token_ids": context.trace_contract.token_ids,
                "source_template_token_count": TARGET_TOKEN_COUNT,
                "source_template_token_ids_u32le_sha256": context.trace_contract.token_ids_u32le_sha256,
                "annotated_compiler_trace": context.trace_contract.annotated_trace,
                "selected_context_span_count": context.trace_contract.selected_context_span_count,
                "new_diagnostic_not_historical": True,
            },
            "all_layer_execution_contract": {
                "baseline_and_candidate_exact_prefix_forwards": 1,
                "baseline_exact_prefix_complete_native_forwards": TARGET_TOKEN_COUNT,
                "candidate_exact_prefix_complete_native_forwards": TARGET_TOKEN_COUNT,
                "layers_per_prefix_forward": TARGET_LAYER_COUNT,
                "forced_continuation": {
                    "derive_token_from_baseline_deterministic_argmax_after_exact_prefix": True,
                    "force_identical_token_into_baseline_and_candidate": True,
                    "additional_forwards_per_path": 1,
                    "layers_per_additional_forward": TARGET_LAYER_COUNT,
                },
                "complete_native_forwards_per_path": TOTAL_FULL_TOKEN_FORWARDS_PER_PATH,
                "complete_native_forwards_total": TOTAL_FULL_TOKEN_FORWARDS,
                "complete_native_layers_traversed_total": TOTAL_FULL_TOKEN_FORWARDS * TARGET_LAYER_COUNT,
                "control_and_candidate_model_bodies_must_not_run_concurrently": True,
                "unbounded_generation_or_sampling_loop_forbidden": True,
            },
            "claim_boundary": {
                "typed_hq30gr2_diagnostic_only": True,
                "does_not_call_hcli_or_an_endpoint": True,
                "does_not_claim_hcli": True,
                "does_not_claim_coherence": True,
                "does_not_claim_tps_or_tg": True,
                "does_not_claim_capability": True,
                "does_not_claim_tournament": True,
            },
        }
    )
    _atomic_json_new(path, document)
    return path, document, _file_evidence(path, "diagnostic input contract")


def _child_command(config: LaunchConfig, input_contract: Path, inner_capture: Path) -> list[str]:
    assert config.lease_receipt is not None
    return [
        str(_canonical_regular(config.probe_bin, "--probe-bin", executable=True)),
        "--candidate-manifest",
        str(_canonical_regular(config.candidate_manifest, "--candidate-manifest")),
        "--candidate-admission-current",
        str(_canonical_regular(config.candidate_admission_current, "--candidate-admission-current")),
        "--compiler-trace-current",
        str(_canonical_regular(config.compiler_trace_current, "--compiler-trace-current")),
        "--route-capture-current",
        str(_canonical_regular(config.route_capture_current, "--route-capture-current")),
        "--preparation-current",
        str(_canonical_regular(config.preparation_current, "--preparation-current")),
        "--control-manifest",
        str(_canonical_regular(config.control_manifest, "--control-manifest")),
        "--control-runtime-receipt",
        str(_canonical_regular(config.control_runtime_receipt, "--control-runtime-receipt")),
        "--lease-receipt",
        str(_canonical_regular(config.lease_receipt, "--lease-receipt")),
        "--input-contract",
        str(_canonical_regular(input_contract, "diagnostic input contract")),
        "--capture-dir",
        str(inner_capture),
        "--mode",
        MODE,
        "--workers",
        str(config.workers),
    ]


def _sync_evidence(path: Path) -> dict[str, Any]:
    with path.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return _file_evidence(path, f"outer stream {path.name}")


def _validate_inner_binding(
    receipt: Mapping[str, Any],
    *,
    context: LaunchContext,
    input_contract: Mapping[str, Any],
    input_evidence: Mapping[str, Any],
) -> None:
    if (
        receipt.get("schema") != EXPECTED_INNER_SCHEMA
        or receipt.get("status") != EXPECTED_INNER_STATUS
        or receipt.get("mode") != MODE
        or receipt.get("metal_device_or_dispatch_performed") is not True
        or receipt.get("typed_hq30gr2_diagnostic_only") is not True
    ):
        raise Qwen30AllLayerProbeLauncherError("inner all-layer diagnostic schema/status/scope drifted")
    durable = _mapping(receipt.get("durable_capture"), "inner durable capture")
    if durable.get("receipt_written_last_is_completion_marker") is not True:
        raise Qwen30AllLayerProbeLauncherError("inner receipt does not attest receipt-last capture")
    execution = _mapping(receipt.get("exact_trace_execution"), "inner exact trace execution")
    if (
        execution.get("probe_id") != TARGET_PROBE
        or execution.get("source_template_token_count") != TARGET_TOKEN_COUNT
        or execution.get("source_template_token_ids_u32le_sha256")
        != context.trace_contract.token_ids_u32le_sha256
        or execution.get("baseline_exact_prefix_all_48_layers") is not True
        or execution.get("candidate_exact_prefix_all_48_layers") is not True
        or execution.get("unbounded_generation_or_sampling_loop_performed") is not False
    ):
        raise Qwen30AllLayerProbeLauncherError("inner exact literal_hawking prefix contract drifted")
    continuation = _mapping(execution.get("forced_continuation"), "inner forced continuation")
    if (
        continuation.get("baseline_deterministic_argmax_after_exact_prefix") is not True
        or continuation.get("forced_identical_token_into_baseline_and_candidate") is not True
        or continuation.get("additional_forwards_per_path") != 1
        or continuation.get("baseline_additional_all_48_layers") is not True
        or continuation.get("candidate_additional_all_48_layers") is not True
        or not isinstance(continuation.get("forced_token_id"), int)
        or continuation.get("forced_token_id") < 0
    ):
        raise Qwen30AllLayerProbeLauncherError("inner forced continuation contract drifted")
    witnesses = _mapping(receipt.get("structural_witnesses"), "inner structural witnesses")
    for label, prefix_key, continuation_key in (
        ("scalar control", "control_scalar_path", "control_forced_continuation"),
        ("typed candidate", "candidate_typed_hq30gr2_path", "candidate_forced_continuation"),
    ):
        prefix = _mapping(witnesses.get(prefix_key), f"inner {label} prefix witness")
        if (
            prefix.get("exact_prefix_token_forwards") != TARGET_TOKEN_COUNT
            or prefix.get("all_layer_route_captures") != TARGET_TOKEN_COUNT * TARGET_LAYER_COUNT
            or prefix.get("layers_per_forward") != TARGET_LAYER_COUNT
            or not isinstance(prefix.get("route_trace_sha256"), str)
        ):
            raise Qwen30AllLayerProbeLauncherError(f"inner {label} prefix forward count/witness drifted")
        continuation_witness = _mapping(
            witnesses.get(continuation_key), f"inner {label} forced-continuation witness"
        )
        step = _mapping(continuation_witness.get("step"), f"inner {label} continuation step")
        if (
            continuation_witness.get("additional_forwards") != FORCED_CONTINUATION_FORWARDS_PER_PATH
            or step.get("position") != TARGET_TOKEN_COUNT
            or step.get("all_layers_route_captured") != TARGET_LAYER_COUNT
            or step.get("experts_per_layer") != 8
            or not isinstance(step.get("route_ids_u32le_sha256"), str)
            or step.get("command_buffers", 0) <= 0
            or step.get("metal_dispatches", 0) <= 0
        ):
            raise Qwen30AllLayerProbeLauncherError(f"inner {label} forced-continuation witness drifted")
    typed_sparse = _mapping(
        witnesses.get("typed_l0_e0_sparse_interception"), "inner typed sparse interception witness"
    )
    if (
        typed_sparse.get("selected_residual_organs")
        != [
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.0.up_proj.weight",
        ]
        or not isinstance(typed_sparse.get("device_sparse_gate_up_encodes"), int)
        or typed_sparse.get("device_sparse_gate_up_encodes") <= 0
        or typed_sparse.get("device_sparse_gate_up_encodes")
        != typed_sparse.get("matching_l0_e0_route_selections")
        or typed_sparse.get("direct_fallback_for_sparse_residual_forbidden") is not True
        or typed_sparse.get("scalar_control_topology_for_all_unchanged_organs") is not True
        or witnesses.get("model_bodies_concurrent") is not False
        or witnesses.get("timing_or_rate_values_recorded") is not False
    ):
        raise Qwen30AllLayerProbeLauncherError("inner typed sparse/all-layer scope witness drifted")
    artifact = _mapping(receipt.get("artifact_binding"), "inner artifact binding")
    _assert_file_reference(artifact.get("candidate_manifest"), context.candidate_manifest, "inner candidate manifest")
    if artifact.get("candidate_manifest_seal_sha256") != context.candidate_manifest_seal_sha256:
        raise Qwen30AllLayerProbeLauncherError("inner candidate manifest seal drifted")
    if _canonical_from_document(
        artifact.get("candidate_admission_current_path"), "inner candidate admission pointer"
    ) != Path(str(context.candidate_admission_current["path"])):
        raise Qwen30AllLayerProbeLauncherError("inner candidate admission pointer path drifted")
    if (
        artifact.get("candidate_admission_pointer_seal_sha256")
        != context.candidate_admission_pointer_seal_sha256
        or artifact.get("candidate_admission_receipt_seal_sha256")
        != context.candidate_admission_receipt_seal_sha256
    ):
        raise Qwen30AllLayerProbeLauncherError("inner candidate admission binding drifted")
    _assert_file_reference(artifact.get("control_manifest"), context.control_manifest, "inner control manifest")
    if artifact.get("control_manifest_seal_sha256") != context.control_manifest_seal_sha256:
        raise Qwen30AllLayerProbeLauncherError("inner control manifest seal drifted")
    _assert_file_reference(
        artifact.get("control_runtime_receipt"), context.control_runtime_receipt, "inner control runtime"
    )
    if artifact.get("control_runtime_receipt_seal_sha256") != context.control_runtime_seal_sha256:
        raise Qwen30AllLayerProbeLauncherError("inner control runtime seal drifted")
    upstream = _mapping(receipt.get("upstream_diagnostic_binding"), "inner upstream diagnostic binding")
    for label, reference, evidence, receipt_seal in (
        ("compiler trace", upstream.get("compiler_trace_receipt"), context.compiler_trace_receipt, context.compiler_trace_seal_sha256),
        ("L0 route capture", upstream.get("route_capture_receipt"), context.route_capture_receipt, context.route_capture_seal_sha256),
        ("all-layer preparation", upstream.get("preparation_receipt"), context.preparation_receipt, context.preparation_seal_sha256),
    ):
        _assert_file_reference(reference, evidence, f"inner {label}")
        if _mapping(reference, f"inner {label}").get("seal_sha256") != receipt_seal:
            raise Qwen30AllLayerProbeLauncherError(f"inner {label} seal drifted")
    contract = _mapping(receipt.get("input_contract"), "inner input contract")
    _assert_file_reference(contract, input_evidence, "inner input contract")
    if (
        contract.get("seal_sha256") != input_contract.get("seal_sha256")
        or contract.get("schema") != INPUT_SCHEMA
        or contract.get("status") != input_contract.get("status")
    ):
        raise Qwen30AllLayerProbeLauncherError("inner input contract seal/schema/status drifted")
    policy = _mapping(receipt.get("metal_execution_policy"), "inner Metal execution policy")
    if (
        policy.get("strict_math_required") is not True
        or policy.get("diagnostic_only") is not True
        or policy.get("timing_or_benchmarking_allowed") is not False
        or policy.get("hcli_or_server_allowed") is not False
        or policy.get("tps_or_tg_claim_allowed") is not False
        or policy.get("coherence_claim_allowed") is not False
        or policy.get("capability_claim_allowed") is not False
        or policy.get("tournament_claim_allowed") is not False
    ):
        raise Qwen30AllLayerProbeLauncherError("inner Metal policy drifted")
    lease = _mapping(policy.get("lease_binding"), "inner lease binding")
    _assert_file_reference(lease, context.lease_receipt, "inner lease receipt")
    if (
        lease.get("seal_sha256") != context.lease_seal_sha256
        or lease.get("schema") != QUIET_LEASE_SCHEMA
        or lease.get("status") != QUIET_LEASE_STATUS
    ):
        raise Qwen30AllLayerProbeLauncherError("inner lease binding drifted")
    boundary = _mapping(receipt.get("claim_boundary"), "inner claim boundary")
    for key in (
        "does_not_claim_hcli",
        "does_not_claim_coherence",
        "does_not_claim_tps_or_tg",
        "does_not_claim_capability",
        "does_not_claim_tournament",
    ):
        if boundary.get(key) is not True:
            raise Qwen30AllLayerProbeLauncherError(f"inner claim boundary lacks {key}=true")


def _inner_evidence(
    config: LaunchConfig,
    context: LaunchContext,
    input_contract: Mapping[str, Any],
    input_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    inner_capture = config.capture_dir / INNER_CAPTURE
    receipt_path = inner_capture / "receipt.json"
    evidence: dict[str, Any] = {
        "capture_dir": str(inner_capture),
        "receipt": {"path": str(receipt_path), "present": receipt_path.is_file()},
    }
    if not receipt_path.is_file():
        evidence["invocation"] = {
            "path": str(inner_capture / "invocation.json"),
            "present": (inner_capture / "invocation.json").is_file(),
        }
        return evidence
    try:
        receipt, _ = _sealed_json(receipt_path, "inner HQ30GR2 all-layer receipt")
        evidence["receipt"] = _file_evidence(receipt_path, "inner HQ30GR2 all-layer receipt")
        evidence["schema"] = receipt.get("schema")
        evidence["status"] = receipt.get("status")
        evidence["mode"] = receipt.get("mode")
        evidence["metal_performed"] = receipt.get("metal_device_or_dispatch_performed")
        _validate_inner_binding(
            receipt,
            context=context,
            input_contract=input_contract,
            input_evidence=input_evidence,
        )
    except Qwen30AllLayerProbeLauncherError as exc:
        evidence["binding_valid"] = False
        evidence["binding_error"] = str(exc)
    else:
        evidence["binding_valid"] = True
    return evidence


def _terminate_process_group(process: subprocess.Popen[bytes]) -> int | None:
    """Terminate the isolated session, then reap its direct child deterministically."""

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
        "process_group_isolated": True,
    }
    if spawn_error is not None:
        terminal["spawn_error"] = spawn_error
        terminal["reaped"] = False
    return terminal


def _terminal_status(terminal: Mapping[str, Any], inner: Mapping[str, Any]) -> str:
    if terminal.get("spawn_error"):
        return "REFUSED_QWEN30_HQ30GR2_ALL_LAYER_OUTER_CHILD_SPAWN_ERROR"
    if terminal.get("timed_out"):
        return "REFUSED_QWEN30_HQ30GR2_ALL_LAYER_OUTER_CHILD_TIMEOUT"
    if terminal.get("signal") is not None:
        return "REFUSED_QWEN30_HQ30GR2_ALL_LAYER_OUTER_CHILD_SIGNAL"
    if terminal.get("exit_code") != 0:
        return "REFUSED_QWEN30_HQ30GR2_ALL_LAYER_OUTER_CHILD_NONZERO"
    if inner.get("binding_valid") is not True or inner.get("status") != EXPECTED_INNER_STATUS:
        return "REFUSED_QWEN30_HQ30GR2_ALL_LAYER_OUTER_ZERO_EXIT_WITHOUT_STRICT_INNER_RECEIPT"
    return "CAPTURED_QWEN30_HQ30GR2_ALL_LAYER_CURRENT_TRACE_DIAGNOSTIC_UNQUALIFIED"


def _terminal_success(receipt: Mapping[str, Any]) -> bool:
    return receipt.get("status") == "CAPTURED_QWEN30_HQ30GR2_ALL_LAYER_CURRENT_TRACE_DIAGNOSTIC_UNQUALIFIED"


def _terminal_receipt(
    config: LaunchConfig,
    context: LaunchContext,
    *,
    identity: str,
    input_contract: Mapping[str, Any],
    input_evidence: Mapping[str, Any],
    command: Sequence[str],
    child_pid: int | None,
    started_at: str,
    finished_at: str,
    terminal: Mapping[str, Any],
    capture_error: str | None = None,
) -> dict[str, Any]:
    inner = _inner_evidence(config, context, input_contract, input_evidence)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": _terminal_status(terminal, inner),
        "recorded_at": finished_at,
        "one_shot": {
            "automatic_retry_disabled": True,
            "same_capture_dir_never_starts_a_second_child": True,
            "terminal_receipt_written_last": True,
            "fresh_quiet_diagnostic_lease_consumed_by_one_outer_attempt_only": True,
        },
        "launch_identity_sha256": identity,
        "source_binding": {
            "probe_binary": context.probe_binary,
            "candidate_manifest": context.candidate_manifest,
            "candidate_manifest_seal_sha256": context.candidate_manifest_seal_sha256,
            "candidate_admission_current": context.candidate_admission_current,
            "candidate_admission_pointer_seal_sha256": context.candidate_admission_pointer_seal_sha256,
            "candidate_admission_receipt": context.candidate_admission_receipt,
            "candidate_admission_receipt_seal_sha256": context.candidate_admission_receipt_seal_sha256,
            "compiler_trace_current": context.compiler_trace_current,
            "compiler_trace_current_seal_sha256": context.compiler_trace_current_seal_sha256,
            "compiler_trace_receipt": context.compiler_trace_receipt,
            "compiler_trace_seal_sha256": context.compiler_trace_seal_sha256,
            "route_capture_current": context.route_capture_current,
            "route_capture_current_seal_sha256": context.route_capture_current_seal_sha256,
            "route_capture_receipt": context.route_capture_receipt,
            "route_capture_seal_sha256": context.route_capture_seal_sha256,
            "preparation_current": context.preparation_current,
            "preparation_current_seal_sha256": context.preparation_current_seal_sha256,
            "preparation_receipt": context.preparation_receipt,
            "preparation_seal_sha256": context.preparation_seal_sha256,
            "cpu_preflight_receipt": context.cpu_preflight_receipt,
            "cpu_preflight_seal_sha256": context.cpu_preflight_seal_sha256,
            "component_parity_current": context.component_parity_current,
            "component_parity_current_seal_sha256": context.component_parity_current_seal_sha256,
            "component_parity_terminal": context.component_parity_terminal,
            "component_parity_terminal_seal_sha256": context.component_parity_terminal_seal_sha256,
            "control_manifest": context.control_manifest,
            "control_manifest_seal_sha256": context.control_manifest_seal_sha256,
            "control_runtime_receipt": context.control_runtime_receipt,
            "control_runtime_seal_sha256": context.control_runtime_seal_sha256,
            "lease_receipt": context.lease_receipt,
            "lease_seal_sha256": context.lease_seal_sha256,
            "mode": MODE,
            "workers": config.workers,
        },
        "diagnostic_input_contract": {
            **input_evidence,
            "seal_sha256": input_contract.get("seal_sha256"),
            "source_template_token_count": TARGET_TOKEN_COUNT,
            "source_template_token_ids_u32le_sha256": context.trace_contract.token_ids_u32le_sha256,
            "forced_shared_continuation": True,
            "complete_native_forwards_per_path": TOTAL_FULL_TOKEN_FORWARDS_PER_PATH,
            "complete_native_forwards_total": TOTAL_FULL_TOKEN_FORWARDS,
            "complete_native_layers_traversed_total": TOTAL_FULL_TOKEN_FORWARDS * TARGET_LAYER_COUNT,
        },
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
            "one_typed_hq30gr2_current_trace_diagnostic_only": True,
            "does_not_reload_or_modify_live_qwen30_server_watcher_or_adapter": True,
            "does_not_call_hcli_or_an_endpoint": True,
            "does_not_claim_hcli": True,
            "does_not_claim_coherence": True,
            "does_not_claim_tps_or_tg": True,
            "does_not_claim_capability": True,
            "does_not_claim_tournament": True,
        },
    }
    if capture_error is not None:
        receipt["capture_error"] = capture_error
    return seal(receipt)


def _replay_existing(config: LaunchConfig, identity: str) -> dict[str, Any]:
    terminal_path = config.capture_dir / TERMINAL_FILENAME
    if not terminal_path.is_file():
        raise Qwen30AllLayerProbeLauncherError(
            f"capture directory exists without a terminal receipt: {config.capture_dir}"
        )
    receipt, _ = _sealed_json(terminal_path, "outer terminal receipt")
    if receipt.get("schema") != SCHEMA or receipt.get("launch_identity_sha256") != identity:
        raise Qwen30AllLayerProbeLauncherError("capture directory belongs to another launch identity")
    return receipt


def run_attempt(config: LaunchConfig) -> dict[str, Any]:
    """Run one reaped process group or sealed-replay its terminal record."""

    context = _validate_config(config)
    identity = _launch_identity(config, context)
    if config.capture_dir.exists():
        return _replay_existing(config, identity)
    if not config.capture_dir.parent.is_dir():
        raise Qwen30AllLayerProbeLauncherError(
            f"capture parent does not exist: {config.capture_dir.parent}"
        )
    try:
        config.capture_dir.mkdir(mode=0o750)
    except FileExistsError:
        return _replay_existing(config, identity)
    input_path, input_contract, input_evidence = _write_input_contract(
        config.capture_dir, identity=identity, context=context
    )
    command = _child_command(config, input_path, config.capture_dir / INNER_CAPTURE)
    started_at = _utc_now()
    _atomic_json_new(
        config.capture_dir / ACTIVE_FILENAME,
        seal(
            {
                "schema": SCHEMA,
                "status": "STARTED_QWEN30_HQ30GR2_ALL_LAYER_OUTER_ONE_SHOT",
                "recorded_at": started_at,
                "launch_identity_sha256": identity,
                "command": command,
                "claim_boundary": {
                    "automatic_retry_disabled": True,
                    "fresh_quiet_diagnostic_lease_required": True,
                    "does_not_claim_hcli_tps_coherence_capability_or_tournament": True,
                },
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
                            "status": "RUNNING_QWEN30_HQ30GR2_ALL_LAYER_OUTER_ONE_SHOT",
                            "recorded_at": _utc_now(),
                            "launch_identity_sha256": identity,
                            "pid": child_pid,
                            "parent_pid": os.getpid(),
                            "process_group_id": child_pid,
                            "command": command,
                            "input_contract": str(input_path),
                            "inner_capture_dir": str(config.capture_dir / INNER_CAPTURE),
                        }
                    ),
                )
            except Qwen30AllLayerProbeLauncherError as exc:
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
        input_contract=input_contract,
        input_evidence=input_evidence,
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
    parser.add_argument("--compiler-trace-current", type=Path, required=True)
    parser.add_argument("--route-capture-current", type=Path, required=True)
    parser.add_argument("--preparation-current", type=Path, required=True)
    parser.add_argument("--cpu-preflight-receipt", type=Path, required=True)
    parser.add_argument("--component-parity-current", type=Path, required=True)
    parser.add_argument("--control-manifest", type=Path, required=True)
    parser.add_argument("--control-runtime-receipt", type=Path, required=True)
    parser.add_argument("--lease-receipt", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(argv)
    config = LaunchConfig(
        probe_bin=parsed.probe_bin,
        candidate_manifest=parsed.candidate_manifest,
        candidate_admission_current=parsed.candidate_admission_current,
        compiler_trace_current=parsed.compiler_trace_current,
        route_capture_current=parsed.route_capture_current,
        preparation_current=parsed.preparation_current,
        cpu_preflight_receipt=parsed.cpu_preflight_receipt,
        component_parity_current=parsed.component_parity_current,
        control_manifest=parsed.control_manifest,
        control_runtime_receipt=parsed.control_runtime_receipt,
        lease_receipt=parsed.lease_receipt,
        capture_dir=parsed.capture_dir,
        workers=parsed.workers,
        timeout_seconds=parsed.timeout_seconds,
    )
    try:
        receipt = run_attempt(config)
    except Qwen30AllLayerProbeLauncherError as exc:
        print(json.dumps({"status": "REFUSED_QWEN30_HQ30GR2_ALL_LAYER_OUTER_LAUNCHER_ERROR", "error": str(exc)}))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if _terminal_success(receipt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
