"""Fail-closed terminal handoff for the isolated Qwen30 quality-repack.

This watcher deliberately has a much narrower authority than the shared Qwen
complete-binary admission controller.  It only observes the separately rooted
``gate-up-residual-v1`` candidate and, *after* its immutable terminal receipt
proves a full 18,867-tensor <=1.5 COMPLETE-BPW artifact, invokes the matching
strict native reader.  Its request, immutable receipt history, status, and
mutable current selector all remain under that candidate root.  In particular,
it cannot write a baseline admission/current pointer, start a runtime, or
promote a tournament gate.

An incomplete journal/status is intentionally not enough to create even an
admission request.  The terminal receipt, manifest, source revalidation,
selection receipt, and source snapshot must all agree before native work can
begin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from lab.operators import ascension_qwen_complete_binary_admission as shared
from lab.operators.ascension_qwen30_quality_repack import (
    ARTIFACT_PREFIX,
    BRANCH_ID,
    SCHEMA as QUALITY_MANIFEST_SCHEMA,
    SOURCE_SNAPSHOT_SCHEMA,
    SELECTION_SCHEMA,
)
from lab.receipts import seal


REPO_ROOT = Path(__file__).resolve().parents[2]
QUALITY_ROOT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/quality-candidates/gate-up-residual-v1"
)
BASELINE_ROOT = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen30/complete-gravity"

REQUEST_SCHEMA = "hawking.ascension.qwen30_quality_repack_native_admission_request.v1"
STATUS_SCHEMA = "hawking.ascension.qwen30_quality_repack_native_admission_status.v1"
RECEIPT_SCHEMA = "hawking.ascension.qwen30_quality_repack_native_admission_receipt.v1"
CURRENT_POINTER_SCHEMA = "hawking.ascension.qwen30_quality_repack_native_admission_current_pointer.v1"
NATIVE_RESULT_SCHEMA = "hawking.ascension.qwen30_quality_repack_native_admission_result.v1"
TERMINAL_SCHEMA = "hawking.ascension.complete_binary_terminal_status.v1"
MANIFEST_STATUS = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED"
TERMINAL_STATUS = "EARNED_COMPLETE_PHYSICAL_BINARY_CANDIDATE_UNQUALIFIED"
NATIVE_RESULT_STATUS = "EARNED_QUALITY_REPACK_COMPLETE_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
ADMISSION_RECEIPT_STATUS = NATIVE_RESULT_STATUS
EXPECTED_MODEL_ID = "Qwen3-Coder-30B-A3B-Instruct-quality-gate-up-residual-v1"
EXPECTED_REPOSITORY = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
EXPECTED_REVISION = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120"
EXPECTED_TENSOR_COUNT = 18_867
SELECTED_ORGANS = (
    "model.layers.0.mlp.experts.0.gate_proj.weight",
    "model.layers.0.mlp.experts.0.up_proj.weight",
)


class QualityRepackAdmissionError(RuntimeError):
    """The isolated terminal candidate is not safe to admit."""


NativeRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class QualityAdmissionTarget:
    """All paths are candidate-local except the immutable source control."""

    root: Path
    baseline_revalidation_path: Path
    model_id: str = EXPECTED_MODEL_ID
    repository: str = EXPECTED_REPOSITORY
    revision: str = EXPECTED_REVISION
    expected_tensor_count: int = EXPECTED_TENSOR_COUNT

    @property
    def manifest_path(self) -> Path:
        return self.root / f"{ARTIFACT_PREFIX}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"

    @property
    def terminal_path(self) -> Path:
        return self.root / f"{ARTIFACT_PREFIX}_COMPLETE_GRAVITY_TERMINAL_RECEIPT.json"

    @property
    def selection_path(self) -> Path:
        return self.root / f"{ARTIFACT_PREFIX}_SELECTION_RECEIPT.json"

    @property
    def snapshot_path(self) -> Path:
        return self.root / f"{ARTIFACT_PREFIX}_SOURCE_BINDING_SNAPSHOT.json"

    @property
    def admission_root(self) -> Path:
        return self.root / "complete-admission"

    @property
    def requests_root(self) -> Path:
        return self.admission_root / "requests"

    @property
    def receipts_root(self) -> Path:
        return self.admission_root / "receipts"

    @property
    def status_path(self) -> Path:
        return self.admission_root / f"{ARTIFACT_PREFIX}_NATIVE_ADMISSION_STATUS.json"

    @property
    def current_pointer_path(self) -> Path:
        return self.root / f"{ARTIFACT_PREFIX}_NATIVE_ADMISSION_CURRENT.json"


DEFAULT_TARGET = QualityAdmissionTarget(
    root=QUALITY_ROOT,
    baseline_revalidation_path=BASELINE_ROOT / "QWEN30_CURRENT_SOURCE_SHARD_REVALIDATION.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _error(message: str) -> QualityRepackAdmissionError:
    return QualityRepackAdmissionError(message)


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{label} must be an object")
    return value


def _require_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise _error(f"{label} must be an array")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise _error(f"{label} must be a non-empty string")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise _error(f"{label} must be a lowercase SHA-256")
    return value


def _require_int(value: object, label: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _error(f"{label} must be an integer")
    if positive and value <= 0:
        raise _error(f"{label} must be positive")
    return value


def _require_number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise _error(f"{label} must be a finite number")
    return float(value)


def _same_path(value: object, expected: Path, label: str) -> None:
    observed = Path(_require_string(value, label))
    if not observed.is_absolute():
        raise _error(f"{label} must be absolute")
    try:
        if observed.resolve(strict=True) != expected.resolve(strict=True):
            raise _error(f"{label} does not bind the expected path")
    except OSError as exc:
        raise _error(f"cannot resolve {label}: {exc}") from exc


def _read_sealed(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        document, metadata = shared._read_document(path, label, sealed=True)
    except shared.CompleteBinaryAdmissionError as exc:
        raise _error(str(exc)) from exc
    return document, metadata


def _verify_file_binding(
    binding: object, expected_path: Path, metadata: Mapping[str, Any], label: str, *, require_identity: bool = True
) -> None:
    row = _require_mapping(binding, label)
    _same_path(row.get("path"), expected_path, f"{label}.path")
    if _require_sha256(row.get("document_sha256"), f"{label}.document_sha256") != metadata["document_sha256"]:
        raise _error(f"{label} raw document SHA-256 differs")
    if _require_sha256(row.get("seal_sha256"), f"{label}.seal_sha256") != metadata["seal_sha256"]:
        raise _error(f"{label} seal differs")
    if require_identity and row.get("file_identity") != metadata["file_identity"]:
        raise _error(f"{label} file identity differs")


def _revalidation_binding(target: QualityAdmissionTarget) -> tuple[dict[str, Any], dict[str, Any]]:
    document, metadata = _read_sealed(target.baseline_revalidation_path, "immutable baseline source revalidation")
    if document.get("schema") != "hawking.ascension.complete_binary_source_revalidation.v1":
        raise _error("immutable baseline source revalidation has an unsupported schema")
    if document.get("status") != "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED":
        raise _error("immutable baseline source revalidation is not earned")
    if document.get("source_repository") != target.repository or document.get("source_revision") != target.revision:
        raise _error("immutable baseline source revalidation differs from this Qwen30 source")
    shards = _require_mapping(document.get("shards"), "immutable baseline source revalidation.shards")
    if _require_int(document.get("sealed_shard_count"), "immutable baseline source revalidation.sealed_shard_count", positive=True) != len(shards):
        raise _error("immutable baseline source revalidation shard cardinality differs")
    for name, row in shards.items():
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise _error("immutable baseline source revalidation has unsafe shard name")
        item = _require_mapping(row, f"immutable baseline source revalidation shard {name}")
        expected = _require_sha256(item.get("expected_sha256"), f"immutable revalidation expected SHA {name}")
        if _require_sha256(item.get("observed_sha256"), f"immutable revalidation observed SHA {name}") != expected:
            raise _error(f"immutable baseline source revalidation has a shard hash mismatch: {name}")
        identity = _require_mapping(item.get("file_identity"), f"immutable revalidation identity {name}")
        if _require_int(identity.get("bytes"), f"immutable revalidation identity bytes {name}", positive=True) != _require_int(item.get("expected_bytes"), f"immutable revalidation expected bytes {name}", positive=True):
            raise _error(f"immutable baseline source revalidation has a byte identity mismatch: {name}")
    source_audit_path = Path(_require_string(document.get("source_audit_path"), "immutable revalidation source_audit_path"))
    if not source_audit_path.is_absolute():
        raise _error("immutable revalidation source audit path must be absolute")
    audit, audit_meta = _read_sealed(source_audit_path, "immutable source audit")
    audit_seal = _require_sha256(document.get("source_audit_seal_sha256"), "immutable revalidation source audit seal")
    if audit_meta["seal_sha256"] != audit_seal:
        raise _error("immutable source audit does not match revalidation seal")
    return {
        "path": metadata["path"],
        "document_sha256": metadata["document_sha256"],
        "seal_sha256": metadata["seal_sha256"],
        "file_identity": metadata["file_identity"],
        "repository": target.repository,
        "revision": target.revision,
        "source_audit_path": str(source_audit_path.resolve()),
        "source_audit_document_sha256": audit_meta["document_sha256"],
        "source_audit_seal_sha256": audit_seal,
        "source_model_dir": _require_string(document.get("source_model_dir"), "immutable revalidation source_model_dir"),
        "index_path": _require_string(document.get("index_path"), "immutable revalidation index_path"),
        "index_sha256": _require_sha256(document.get("index_sha256"), "immutable revalidation index_sha256"),
        "sealed_shard_count": len(shards),
        "sealed_shard_hashes_sha256": _require_sha256(document.get("sealed_shard_hashes_sha256"), "immutable revalidation sealed_shard_hashes_sha256"),
        "weight_map_sha256": _require_sha256(document.get("weight_map_sha256"), "immutable revalidation weight_map_sha256"),
    }, audit


def _wait_for_terminal(target: QualityAdmissionTarget) -> tuple[str, dict[str, Any]] | None:
    """Return a non-authorizing wait status; never creates a request here."""

    if not target.manifest_path.exists():
        return "WAITING_FOR_IMMUTABLE_FULL_MANIFEST", {"manifest_path": str(target.manifest_path)}
    if not target.terminal_path.exists():
        return "WAITING_FOR_IMMUTABLE_TERMINAL_QUALITY_CANDIDATE", {
            "manifest_path": str(target.manifest_path),
            "terminal_receipt_path": str(target.terminal_path),
        }
    try:
        terminal, _ = _read_sealed(target.terminal_path, "quality candidate terminal receipt")
    except QualityRepackAdmissionError:
        return "WAITING_FOR_IMMUTABLE_TERMINAL_QUALITY_CANDIDATE", {
            "manifest_path": str(target.manifest_path),
            "terminal_receipt_path": str(target.terminal_path),
            "detail": "terminal receipt is not a usable sealed authority yet",
        }
    if terminal.get("schema") != TERMINAL_SCHEMA or terminal.get("status") != TERMINAL_STATUS:
        return "WAITING_FOR_IMMUTABLE_TERMINAL_QUALITY_CANDIDATE", {
            "manifest_path": str(target.manifest_path),
            "terminal_receipt_path": str(target.terminal_path),
            "observed_status": terminal.get("status"),
        }
    binding = terminal.get("binding")
    candidate = terminal.get("candidate")
    if not isinstance(binding, Mapping) or not isinstance(candidate, Mapping):
        return "WAITING_FOR_IMMUTABLE_TERMINAL_QUALITY_CANDIDATE", {"detail": "terminal receipt has no binding/candidate"}
    progress = binding.get("progress")
    if not isinstance(progress, Mapping):
        return "WAITING_FOR_IMMUTABLE_TERMINAL_QUALITY_CANDIDATE", {"detail": "terminal receipt has no completed cursor"}
    try:
        planned = _require_int(progress.get("planned_tensors"), "terminal progress planned_tensors", positive=True)
        completed = _require_int(progress.get("completed_tensors"), "terminal progress completed_tensors", positive=True)
        cursor = _require_int(progress.get("next_cursor"), "terminal progress next_cursor", positive=True)
    except QualityRepackAdmissionError as exc:
        return "WAITING_FOR_IMMUTABLE_TERMINAL_QUALITY_CANDIDATE", {"detail": str(exc)}
    if planned != target.expected_tensor_count or completed != planned or cursor != planned:
        return "WAITING_FOR_IMMUTABLE_TERMINAL_QUALITY_CANDIDATE", {
            "expected_tensors": target.expected_tensor_count,
            "planned_tensors": planned,
            "completed_tensors": completed,
            "next_cursor": cursor,
        }
    return None


def _validate_terminal_candidate(target: QualityAdmissionTarget) -> dict[str, Any]:
    """Read all small authorities and reject a partial/mixed candidate before native I/O."""

    terminal, terminal_meta = _read_sealed(target.terminal_path, "quality candidate terminal receipt")
    if terminal.get("schema") != TERMINAL_SCHEMA or terminal.get("status") != TERMINAL_STATUS:
        raise _error("quality candidate terminal receipt is not an earned complete candidate")
    binding = _require_mapping(terminal.get("binding"), "quality terminal binding")
    candidate = _require_mapping(terminal.get("candidate"), "quality terminal candidate")
    if binding.get("model_id") != target.model_id or binding.get("artifact_prefix") != ARTIFACT_PREFIX:
        raise _error("quality terminal receipt belongs to a different candidate branch")
    if binding.get("manifest_schema") != QUALITY_MANIFEST_SCHEMA:
        raise _error("quality terminal receipt does not bind the quality manifest schema")
    progress = _require_mapping(binding.get("progress"), "quality terminal progress")
    for key in ("planned_tensors", "completed_tensors", "next_cursor"):
        if _require_int(progress.get(key), f"quality terminal progress.{key}", positive=True) != target.expected_tensor_count:
            raise _error(f"quality terminal receipt did not reach all {target.expected_tensor_count} tensors")
    if progress.get("next_source_shard") is not None or progress.get("next_tensor_name") is not None:
        raise _error("quality terminal receipt has a non-terminal source cursor")

    manifest, manifest_meta = _read_sealed(target.manifest_path, "quality complete manifest")
    if manifest.get("schema") != QUALITY_MANIFEST_SCHEMA or manifest.get("status") != MANIFEST_STATUS:
        raise _error("quality manifest is not the expected unqualified complete candidate")
    _same_path(candidate.get("manifest_path"), target.manifest_path, "quality terminal candidate.manifest_path")
    if _require_sha256(candidate.get("manifest_seal_sha256"), "quality terminal candidate.manifest_seal_sha256") != manifest_meta["seal_sha256"]:
        raise _error("quality terminal candidate manifest seal differs from manifest")
    if _require_sha256(candidate.get("manifest_document_sha256"), "quality terminal candidate.manifest_document_sha256") != manifest_meta["document_sha256"]:
        raise _error("quality terminal candidate manifest raw SHA-256 differs from manifest")
    if candidate.get("manifest_file_identity") != manifest_meta["file_identity"]:
        raise _error("quality terminal candidate manifest identity differs from manifest")

    revalidation, _audit = _revalidation_binding(target)
    if _require_sha256(manifest.get("source_body_audit_seal_sha256"), "quality manifest source body audit seal") != revalidation["source_audit_seal_sha256"]:
        raise _error("quality manifest source audit seal differs from immutable revalidation")
    _same_path(manifest.get("source_revalidation_receipt_path"), target.baseline_revalidation_path, "quality manifest source revalidation path")
    if _require_sha256(manifest.get("source_revalidation_receipt_seal_sha256"), "quality manifest source revalidation seal") != revalidation["seal_sha256"]:
        raise _error("quality manifest source revalidation seal differs from immutable revalidation")
    if binding.get("source_body_audit_seal_sha256") != revalidation["source_audit_seal_sha256"]:
        raise _error("quality terminal audit binding differs from immutable revalidation")
    _same_path(binding.get("source_revalidation_receipt_path"), target.baseline_revalidation_path, "quality terminal revalidation path")
    if _require_sha256(binding.get("source_revalidation_receipt_seal_sha256"), "quality terminal revalidation seal") != revalidation["seal_sha256"]:
        raise _error("quality terminal revalidation seal differs from immutable revalidation")

    source = _require_mapping(manifest.get("source"), "quality manifest.source")
    if source.get("repository") != target.repository or source.get("model_dir") != revalidation["source_model_dir"]:
        raise _error("quality manifest source differs from immutable revalidation")
    if _require_int(source.get("tensor_count"), "quality manifest source.tensor_count", positive=True) != target.expected_tensor_count:
        raise _error("quality manifest tensor count is not the complete Qwen30 count")

    ledger = _require_mapping(manifest.get("complete_physical_bpw_ledger"), "quality manifest ledger")
    elements = _require_int(ledger.get("source_weight_elements"), "quality manifest ledger.source_weight_elements", positive=True)
    payload_bytes = _require_int(ledger.get("tensor_payload_bytes"), "quality manifest ledger.tensor_payload_bytes", positive=True)
    manifest_bytes = _require_int(ledger.get("manifest_bytes_billed"), "quality manifest ledger.manifest_bytes_billed", positive=True)
    total_bytes = _require_int(ledger.get("all_required_weight_artifact_bytes"), "quality manifest ledger.all_required_weight_artifact_bytes", positive=True)
    if manifest_bytes != manifest_meta["file_identity"]["bytes"] or total_bytes != payload_bytes + manifest_bytes:
        raise _error("quality manifest ledger does not bill the exact physical manifest and payload bytes")
    bpw = _require_number(ledger.get("complete_physical_bpw"), "quality manifest ledger.complete_physical_bpw")
    if _require_number(ledger.get("threshold_bpw"), "quality manifest ledger.threshold_bpw") != 1.5:
        raise _error("quality manifest ledger threshold is not exactly 1.5 BPW")
    if not math.isclose(bpw, total_bytes * 8.0 / elements, rel_tol=0.0, abs_tol=1e-12):
        raise _error("quality manifest BPW does not equal its exact ledger")
    if ledger.get("passes_storage_threshold") is not (bpw <= 1.5) or bpw > 1.5:
        raise _error("quality manifest has not earned the <=1.5 COMPLETE-BPW gate")
    for key, value in (("all_required_weight_artifact_bytes", total_bytes), ("complete_physical_bpw", bpw), ("passes_storage_threshold", True)):
        if candidate.get(key) != value:
            raise _error(f"quality terminal candidate.{key} differs from the sealed manifest ledger")

    snapshot, snapshot_meta = _read_sealed(target.snapshot_path, "quality source binding snapshot")
    selection, selection_meta = _read_sealed(target.selection_path, "quality repack selection receipt")
    if snapshot.get("schema") != SOURCE_SNAPSHOT_SCHEMA or snapshot.get("status") != "EARNED_IMMUTABLE_SOURCE_AND_ROLLBACK_BINDING":
        raise _error("quality source snapshot is not an earned immutable binding")
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("status") != "EARNED_SOURCE_BOUND_QUALITY_REPACK_SELECTION_UNQUALIFIED":
        raise _error("quality selection is not the sealed source-bound quality choice")
    branch = _require_mapping(manifest.get("quality_repack_branch"), "quality manifest quality_repack_branch")
    if branch.get("branch_id") != BRANCH_ID:
        raise _error("quality manifest branch id differs")
    _verify_file_binding(branch.get("source_binding_snapshot"), target.snapshot_path, snapshot_meta, "quality manifest snapshot binding")
    _verify_file_binding(branch.get("selection_receipt"), target.selection_path, selection_meta, "quality manifest selection binding")
    _verify_file_binding(selection.get("source_binding_snapshot"), target.snapshot_path, snapshot_meta, "quality selection snapshot binding")
    snapshot_binding = _require_mapping(snapshot.get("binding"), "quality source snapshot.binding")
    selection_binding = _require_mapping(selection.get("binding"), "quality selection.binding")
    if dict(selection_binding) != dict(snapshot_binding) or snapshot_binding.get("branch_id") != BRANCH_ID:
        raise _error("quality source snapshot and selection do not share one immutable branch binding")
    _verify_file_binding(snapshot_binding.get("immutable_source_revalidation"), target.baseline_revalidation_path, {
        "document_sha256": revalidation["document_sha256"], "seal_sha256": revalidation["seal_sha256"], "file_identity": revalidation["file_identity"]
    }, "quality snapshot immutable revalidation", require_identity=False)
    selected_names = list(snapshot_binding.get("selected_organs", []))
    if selected_names != list(SELECTED_ORGANS) or branch.get("changed_organs") != list(SELECTED_ORGANS):
        raise _error("quality candidate changed organs differ from the sealed gate/up policy")

    rows = _require_list(manifest.get("tensors"), "quality manifest tensors")
    if len(rows) != target.expected_tensor_count:
        raise _error("quality manifest does not contain every Qwen30 tensor")
    rows_by_name: dict[str, Mapping[str, Any]] = {}
    seen_payload = 0
    seen_elements = 0
    tensors_root = (target.root / "tensors").resolve()
    for raw_row in rows:
        row = _require_mapping(raw_row, "quality manifest tensor row")
        name = _require_string(row.get("tensor_name"), "quality manifest tensor_name")
        if name in rows_by_name:
            raise _error(f"quality manifest has duplicate tensor {name}")
        artifact = Path(_require_string(row.get("artifact_path"), f"quality tensor {name} artifact_path"))
        if not artifact.is_absolute() or artifact.parent.resolve(strict=False) != tensors_root:
            raise _error(f"quality tensor {name} artifact path leaves the candidate tensors root")
        expected_name = hashlib.sha256(name.encode("utf-8")).hexdigest() + ".hq30g"
        if artifact.name != expected_name:
            raise _error(f"quality tensor {name} artifact path is not deterministic")
        seen_payload += _require_int(row.get("artifact_bytes"), f"quality tensor {name} artifact_bytes", positive=True)
        seen_elements += _require_int(row.get("elements"), f"quality tensor {name} elements", positive=True)
        mutation = _require_mapping(row.get("candidate_mutation"), f"quality tensor {name} candidate_mutation")
        rollback = _require_mapping(mutation.get("baseline_rollback"), f"quality tensor {name} rollback")
        if rollback.get("rollback_action") != "use the separately admitted baseline tensor; this candidate never overwrites it":
            raise _error(f"quality tensor {name} does not preserve baseline rollback isolation")
        rows_by_name[name] = row
    if seen_payload != payload_bytes or seen_elements != elements:
        raise _error("quality manifest ledger totals do not equal its full tensor journal")
    if set(SELECTED_ORGANS) - set(rows_by_name):
        raise _error("quality manifest lacks one of the exactly selected organs")
    for name, row in rows_by_name.items():
        mutation = _require_mapping(row.get("candidate_mutation"), f"quality tensor {name} mutation")
        changed = mutation.get("changed_from_admitted_control")
        if name in SELECTED_ORGANS:
            if changed is not True or _require_mapping(row.get("layout"), f"quality changed tensor {name} layout").get("magic") != "HQ30GR2\x00":
                raise _error(f"quality selected organ {name} lacks its exact sparse-residual layout")
        elif changed is not False:
            raise _error(f"quality non-selected tensor {name} was mutated")
    selected_representation = _require_mapping(selection.get("selected_representation"), "quality selection selected_representation")
    organs = _require_list(selected_representation.get("organs"), "quality selection selected organs")
    if [item.get("tensor_name") if isinstance(item, Mapping) else None for item in organs] != list(SELECTED_ORGANS):
        raise _error("quality selection organ order differs from sealed candidate policy")

    return {
        "terminal": {"path": terminal_meta["path"], "document_sha256": terminal_meta["document_sha256"], "seal_sha256": terminal_meta["seal_sha256"], "file_identity": terminal_meta["file_identity"]},
        "complete_manifest": {"path": manifest_meta["path"], "document_sha256": manifest_meta["document_sha256"], "seal_sha256": manifest_meta["seal_sha256"], "schema": QUALITY_MANIFEST_SCHEMA, "status": MANIFEST_STATUS},
        "immutable_source_revalidation": revalidation,
        "source_binding_snapshot": {"path": snapshot_meta["path"], "document_sha256": snapshot_meta["document_sha256"], "seal_sha256": snapshot_meta["seal_sha256"]},
        "selection_receipt": {"path": selection_meta["path"], "document_sha256": selection_meta["document_sha256"], "seal_sha256": selection_meta["seal_sha256"]},
        "complete_physical_bpw": bpw,
        "tensor_count": len(rows),
        "source_weight_elements": elements,
        "tensor_payload_bytes": payload_bytes,
    }


def _build_request(target: QualityAdmissionTarget) -> dict[str, Any]:
    evidence = _validate_terminal_candidate(target)
    return seal(
        {
            "schema": REQUEST_SCHEMA,
            "status": "SEALED_QUALITY_REPACK_NATIVE_ADMISSION_REQUEST",
            "request_version": 1,
            "model": {"key": "qwen30-quality-repack", "id": target.model_id, "repository": target.repository, "revision": target.revision},
            **evidence,
            "native_admission": {
                "required_api": "hawking_core::model::qwen_complete_binary::admit_qwen30_quality_repack_artifact",
                "strict_native_reader_must_revalidate_every_manifest_tensor_payload_source_chain_and_selected_residual_discriminator": True,
            },
            "isolation": {
                "candidate_root": str(target.root.resolve()),
                "baseline_admission_current_pointer_write_forbidden": True,
                "runtime_server_capability_tps_tg_and_tournament_promotion_forbidden": True,
            },
            "claim_boundary": {
                "request_is_not_admission_or_runtime": True,
                "complete_physical_bpw_is_an_accounting_gate_only": True,
                "native_admission_remains_separate_from_decoder_generation_hcli_and_tps": True,
            },
        }
    )


def _request_path(target: QualityAdmissionTarget, request: Mapping[str, Any]) -> Path:
    manifest = _require_mapping(request.get("complete_manifest"), "quality admission request complete_manifest")
    return target.requests_root / f"{ARTIFACT_PREFIX}_NATIVE_ADMISSION_REQUEST_{_require_sha256(manifest.get('seal_sha256'), 'quality request manifest seal')}.json"


def _receipt_path(target: QualityAdmissionTarget, request: Mapping[str, Any]) -> Path:
    manifest = _require_mapping(request.get("complete_manifest"), "quality admission request complete_manifest")
    return target.receipts_root / f"{ARTIFACT_PREFIX}_NATIVE_ADMISSION_RECEIPT_{_require_sha256(manifest.get('seal_sha256'), 'quality request manifest seal')}.json"


def _loader_digest(path: Path) -> str:
    try:
        return shared._native_loader_digest(path)
    except shared.CompleteBinaryAdmissionError as exc:
        raise _error(str(exc)) from exc


def _default_runner(command: Sequence[str], timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), cwd=REPO_ROOT, check=False, capture_output=True, text=True, timeout=timeout_seconds)


def _invoke_native(
    *, target: QualityAdmissionTarget, request: Mapping[str, Any], native_loader: Path, timeout_seconds: float, runner: NativeRunner
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise _error("native admission timeout must be positive")
    before = _loader_digest(native_loader)
    manifest = _require_mapping(request.get("complete_manifest"), "quality request complete_manifest")
    revalidation = _require_mapping(request.get("immutable_source_revalidation"), "quality request immutable_source_revalidation")
    selection = _require_mapping(request.get("selection_receipt"), "quality request selection_receipt")
    snapshot = _require_mapping(request.get("source_binding_snapshot"), "quality request source_binding_snapshot")
    terminal = _require_mapping(request.get("terminal"), "quality request terminal")
    command = [
        str(native_loader.resolve()), "--manifest", _require_string(manifest.get("path"), "quality request manifest path"),
        "--expected-manifest-seal-sha256", _require_sha256(manifest.get("seal_sha256"), "quality request manifest seal"),
        "--expected-source-audit-seal-sha256", _require_sha256(revalidation.get("source_audit_seal_sha256"), "quality request source audit seal"),
        "--expected-source-revision", target.revision,
        "--expected-revalidation-path", _require_string(revalidation.get("path"), "quality request revalidation path"),
        "--expected-revalidation-seal-sha256", _require_sha256(revalidation.get("seal_sha256"), "quality request revalidation seal"),
        "--expected-selection-path", _require_string(selection.get("path"), "quality request selection path"),
        "--expected-selection-seal-sha256", _require_sha256(selection.get("seal_sha256"), "quality request selection seal"),
        "--expected-source-snapshot-path", _require_string(snapshot.get("path"), "quality request snapshot path"),
        "--expected-source-snapshot-seal-sha256", _require_sha256(snapshot.get("seal_sha256"), "quality request snapshot seal"),
        "--expected-terminal-path", _require_string(terminal.get("path"), "quality request terminal path"),
        "--expected-terminal-seal-sha256", _require_sha256(terminal.get("seal_sha256"), "quality request terminal seal"),
    ]
    try:
        completed = runner(command, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise _error(f"quality native admission timed out after {timeout_seconds:g} seconds") from exc
    except OSError as exc:
        raise _error(f"cannot execute quality native admission: {exc}") from exc
    if _loader_digest(native_loader) != before:
        raise _error("quality native admission executable changed while it ran")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "native admission returned no detail").strip()
        raise _error(f"quality native admission refused candidate (exit={completed.returncode}): {detail[:1000]}")
    try:
        result = shared._parse_json((completed.stdout or "").encode("utf-8"), "quality native admission result")
    except shared.CompleteBinaryAdmissionError as exc:
        raise _error(str(exc)) from exc
    if result.get("schema") != NATIVE_RESULT_SCHEMA or result.get("status") != NATIVE_RESULT_STATUS:
        raise _error("quality native admission did not declare strict success")
    if result.get("model") != "qwen30-quality-repack":
        raise _error("quality native admission returned another model")
    _same_path(result.get("manifest_path"), target.manifest_path, "quality native result manifest_path")
    if _require_sha256(result.get("manifest_seal_sha256"), "quality native result manifest seal") != manifest["seal_sha256"]:
        raise _error("quality native result manifest seal differs from request")
    if _require_sha256(result.get("source_audit_seal_sha256"), "quality native result source audit seal") != revalidation["source_audit_seal_sha256"]:
        raise _error("quality native result audit seal differs from request")
    if result.get("source_revision") != target.revision:
        raise _error("quality native result source revision differs from request")
    if _require_int(result.get("tensor_count"), "quality native result tensor_count", positive=True) != target.expected_tensor_count:
        raise _error("quality native result did not admit every Qwen30 tensor")
    if _require_int(result.get("source_weight_elements"), "quality native result source_weight_elements", positive=True) != request["source_weight_elements"]:
        raise _error("quality native result source-weight element count differs")
    if _require_int(result.get("tensor_payload_bytes"), "quality native result tensor_payload_bytes", positive=True) != request["tensor_payload_bytes"]:
        raise _error("quality native result payload byte count differs")
    if result.get("selected_residual_organs") != list(SELECTED_ORGANS) or result.get("selected_residual_discriminators_verified") is not True:
        raise _error("quality native result did not verify exactly the selected residual discriminators")
    payload_verification = _require_mapping(
        result.get("payload_verification"), "quality native result payload_verification"
    )
    if payload_verification.get("mode") != "bounded_parallel_source_shard_lanes_ordered_reconciliation_v1":
        raise _error("quality native result did not use the bounded ordered payload verifier")
    workers_used = _require_int(
        payload_verification.get("workers_used"), "quality native result payload_verification.workers_used", positive=True
    )
    if workers_used > 4 or _require_int(
        payload_verification.get("workers_cap"), "quality native result payload_verification.workers_cap", positive=True
    ) != 4:
        raise _error("quality native result payload verifier worker bound differs")
    if _require_int(
        payload_verification.get("manifest_rows"), "quality native result payload_verification.manifest_rows", positive=True
    ) != target.expected_tensor_count:
        raise _error("quality native result payload verifier did not scan every manifest row")
    if (
        payload_verification.get("result_order")
        != "manifest_ordinal_ascending_before_catalog_and_receipt"
        or payload_verification.get("candidate_only_read_path") is not True
    ):
        raise _error("quality native result payload verification ordering/isolation differs")
    return {"api": "hawking_core::model::qwen_complete_binary::admit_qwen30_quality_repack_artifact", "executable_path": str(native_loader.resolve()), "executable_sha256": before, **result}


def _receipt_for_success(target: QualityAdmissionTarget, request: Mapping[str, Any], request_path: Path, native: Mapping[str, Any]) -> dict[str, Any]:
    return seal(
        {
            "schema": RECEIPT_SCHEMA,
            "status": ADMISSION_RECEIPT_STATUS,
            "recorded_at": _utc_now(),
            "model": dict(_require_mapping(request.get("model"), "quality request model")),
            "admission_request_path": str(request_path.resolve()),
            "admission_request_seal_sha256": _require_sha256(request.get("seal_sha256"), "quality request seal"),
            "terminal": dict(_require_mapping(request.get("terminal"), "quality request terminal")),
            "complete_manifest": dict(_require_mapping(request.get("complete_manifest"), "quality request manifest")),
            "immutable_source_revalidation": dict(_require_mapping(request.get("immutable_source_revalidation"), "quality request revalidation")),
            "source_binding_snapshot": dict(_require_mapping(request.get("source_binding_snapshot"), "quality request snapshot")),
            "selection_receipt": dict(_require_mapping(request.get("selection_receipt"), "quality request selection")),
            "native_loader": dict(native),
            "isolation": {"candidate_root": str(target.root.resolve()), "baseline_current_pointer_untouched": True, "runtime_server_tournament_promotion_forbidden": True},
            "claim_boundary": {"strict_native_quality_candidate_admission_passed": True, "not_a_native_decoder_generation_hcli_or_tps_result": True, "not_a_capability_tg_agent_os_or_tournament_qualification": True},
        }
    )


def _validate_existing_receipt(target: QualityAdmissionTarget, request: Mapping[str, Any], path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt, meta = _read_sealed(path, "quality native admission receipt")
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("status") != ADMISSION_RECEIPT_STATUS:
        raise _error("existing quality receipt is not a strict native admission receipt")
    if receipt.get("admission_request_seal_sha256") != request.get("seal_sha256"):
        raise _error("existing quality receipt belongs to another request")
    _same_path(receipt.get("admission_request_path"), _request_path(target, request), "existing quality receipt request path")
    for field in ("terminal", "complete_manifest", "immutable_source_revalidation", "source_binding_snapshot", "selection_receipt"):
        if _require_mapping(receipt.get(field), f"existing quality receipt.{field}") != _require_mapping(request.get(field), f"quality request.{field}"):
            raise _error(f"existing quality receipt {field} differs from current exact request")
    native = _require_mapping(receipt.get("native_loader"), "existing quality receipt.native_loader")
    if native.get("api") != "hawking_core::model::qwen_complete_binary::admit_qwen30_quality_repack_artifact":
        raise _error("existing quality receipt does not bind the strict quality native reader")
    return receipt, meta


def _publish_current_pointer(target: QualityAdmissionTarget, request: Mapping[str, Any], receipt_path: Path, receipt: Mapping[str, Any], receipt_meta: Mapping[str, Any], source: str) -> dict[str, Any]:
    """Select the immutable receipt once, without refreshing a live pointer.

    The detached watcher deliberately revisits an already admitted terminal
    candidate.  A timestamped rewrite on every pass makes that otherwise
    immutable selection look like a moving authority to later parity gates.
    Reuse a sealed selector whenever every material binding still agrees; a
    malformed/stale selector is repaired only by the exact candidate-local
    replacement below.
    """

    expected_manifest = dict(_require_mapping(request.get("complete_manifest"), "quality request manifest"))
    expected_request_path = _request_path(target, request)
    expected_request_seal = _require_sha256(request.get("seal_sha256"), "quality request seal")
    expected_isolation = {
        "candidate_root_only": True,
        "baseline_admission_and_current_pointers_unmodified": True,
        "runtime_server_tournament_promotion_forbidden": True,
    }
    if target.current_pointer_path.exists():
        try:
            existing, _existing_meta = _read_sealed(target.current_pointer_path, "existing quality admission current pointer")
            existing_receipt = _require_mapping(existing.get("admission_receipt"), "existing quality admission current receipt")
            if (
                existing.get("schema") == CURRENT_POINTER_SCHEMA
                and existing.get("status") == "CURRENT_QUALITY_REPACK_NATIVE_ADMISSION_RECEIPT_SELECTED"
                and existing.get("candidate_root") == str(target.root.resolve())
                and _require_mapping(existing.get("complete_manifest"), "existing quality admission current manifest") == expected_manifest
                and _require_sha256(existing.get("admission_request_seal_sha256"), "existing quality admission current request seal") == expected_request_seal
                and _require_string(existing_receipt.get("path"), "existing quality admission current receipt path") == str(receipt_path.resolve())
                and _require_sha256(existing_receipt.get("document_sha256"), "existing quality admission current receipt document") == receipt_meta["document_sha256"]
                and _require_sha256(existing_receipt.get("seal_sha256"), "existing quality admission current receipt seal") == receipt["seal_sha256"]
                and _require_mapping(existing.get("isolation"), "existing quality admission current isolation") == expected_isolation
            ):
                _same_path(existing.get("admission_request_path"), expected_request_path, "existing quality admission current request path")
                return existing
        except QualityRepackAdmissionError:
            # A selector is never trusted merely because it exists.  The
            # exact replacement below remains candidate-local and sealed.
            pass
    pointer = seal(
        {
            "schema": CURRENT_POINTER_SCHEMA,
            "status": "CURRENT_QUALITY_REPACK_NATIVE_ADMISSION_RECEIPT_SELECTED",
            "recorded_at": _utc_now(),
            "candidate_root": str(target.root.resolve()),
            "complete_manifest": expected_manifest,
            "admission_request_path": str(expected_request_path.resolve()),
            "admission_request_seal_sha256": expected_request_seal,
            "admission_receipt": {"path": str(receipt_path.resolve()), "document_sha256": receipt_meta["document_sha256"], "seal_sha256": receipt["seal_sha256"], "selection_source": source},
            "isolation": expected_isolation,
        }
    )
    shared._atomic_json(target.current_pointer_path, pointer)
    return pointer


def _publish_status(target: QualityAdmissionTarget, phase: str, **fields: Any) -> dict[str, Any]:
    heartbeat = 1
    if target.status_path.exists():
        try:
            previous, _ = _read_sealed(target.status_path, "prior quality admission status")
            heartbeat = _require_int(previous.get("heartbeat"), "prior quality admission heartbeat", positive=True) + 1
        except QualityRepackAdmissionError:
            heartbeat = 1
    status = seal(
        {
            "schema": STATUS_SCHEMA,
            "recorded_at": _utc_now(),
            "pid": os.getpid(),
            "heartbeat": heartbeat,
            "phase": phase,
            "candidate_root": str(target.root.resolve()),
            "claim_boundary": {"waiting_status_is_not_admission": True, "baseline_admission_current_and_runtime_pointers_are_untouched": True, "not_runtime_capability_hcli_tps_tg_or_tournament": True},
            **fields,
        }
    )
    shared._atomic_json(target.status_path, status)
    return status


def run_once(
    target: QualityAdmissionTarget,
    *,
    native_loader: Path,
    timeout_seconds: float = 7200.0,
    runner: NativeRunner = _default_runner,
) -> dict[str, Any]:
    """Run the one-way terminal handoff, retaining failure as a status only."""

    try:
        waiting = _wait_for_terminal(target)
        if waiting is not None:
            phase, fields = waiting
            return _publish_status(target, phase, **fields)
        request = _build_request(target)
        request_path = _request_path(target, request)
        shared._write_immutable_json(request_path, request, "quality native admission request")
        receipt_path = _receipt_path(target, request)
        if receipt_path.exists():
            receipt, metadata = _validate_existing_receipt(target, request, receipt_path)
            pointer = _publish_current_pointer(target, request, receipt_path, receipt, metadata, "VERSIONED_CURRENT_MANIFEST")
            return _publish_status(target, "EARNED_QUALITY_REPACK_NATIVE_ADMISSION_RECEIPT_REUSED", admission_request_path=str(request_path), admission_receipt_path=str(receipt_path), current_receipt_pointer_path=str(target.current_pointer_path), current_receipt_pointer_seal_sha256=pointer["seal_sha256"])
        _publish_status(target, "STRICT_NATIVE_QUALITY_REPACK_ADMISSION_IN_PROGRESS", admission_request_path=str(request_path), admission_request_seal_sha256=request["seal_sha256"], manifest_path=request["complete_manifest"]["path"], manifest_seal_sha256=request["complete_manifest"]["seal_sha256"])
        native = _invoke_native(target=target, request=request, native_loader=native_loader, timeout_seconds=timeout_seconds, runner=runner)
        if _build_request(target) != request:
            raise _error("terminal manifest/source revalidation/selection changed during strict native admission")
        receipt = _receipt_for_success(target, request, request_path, native)
        shared._write_immutable_json(receipt_path, receipt, "quality native admission receipt")
        verified_receipt, metadata = _validate_existing_receipt(target, request, receipt_path)
        pointer = _publish_current_pointer(target, request, receipt_path, verified_receipt, metadata, "VERSIONED_NEW_NATIVE_SCAN")
        return _publish_status(target, ADMISSION_RECEIPT_STATUS, admission_request_path=str(request_path), admission_request_seal_sha256=request["seal_sha256"], admission_receipt_path=str(receipt_path), admission_receipt_seal_sha256=verified_receipt["seal_sha256"], current_receipt_pointer_path=str(target.current_pointer_path), current_receipt_pointer_seal_sha256=pointer["seal_sha256"], native_loader=native)
    except QualityRepackAdmissionError as exc:
        return _publish_status(target, "BLOCKED_QUALITY_REPACK_NATIVE_ADMISSION_FAIL_CLOSED", detail=str(exc), manifest_path=str(target.manifest_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("once", "watch"), nargs="?", default="once")
    parser.add_argument("--root", type=Path, default=DEFAULT_TARGET.root)
    parser.add_argument("--baseline-revalidation", type=Path, default=DEFAULT_TARGET.baseline_revalidation_path)
    parser.add_argument("--native-loader", type=Path, default=REPO_ROOT / "workspace/ops/build/rust/debug/examples/ascension_qwen30_quality_repack_admission")
    parser.add_argument("--idle-seconds", type=float, default=45.0)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.idle_seconds <= 0 or args.timeout_seconds <= 0:
        raise SystemExit("--idle-seconds and --timeout-seconds must be positive")
    target = QualityAdmissionTarget(root=args.root.expanduser().resolve(), baseline_revalidation_path=args.baseline_revalidation.expanduser().resolve())
    if args.command == "once":
        status = run_once(target, native_loader=args.native_loader, timeout_seconds=args.timeout_seconds)
        print(json.dumps(status, sort_keys=True))
        return 0 if not str(status.get("phase", "")).startswith("BLOCKED") else 2
    stop = False

    def stop_requested(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    previous_int = signal.signal(signal.SIGINT, stop_requested)
    previous_term = signal.signal(signal.SIGTERM, stop_requested)
    try:
        while not stop:
            run_once(target, native_loader=args.native_loader, timeout_seconds=args.timeout_seconds)
            time.sleep(args.idle_seconds)
    finally:
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
