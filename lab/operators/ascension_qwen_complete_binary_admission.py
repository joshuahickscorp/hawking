"""Fail-closed native admission for the physical Qwen complete-binary lanes.

The direct binary packers produce a complete *candidate*, not a runtime.  This
operator is the independent admission boundary between that mutable packer
lane and any later runtime work.  It first seals an immutable request binding
the exact complete manifest to the stable source-content identity and the
current full-shard revalidation receipt.  It then invokes Hawking Core's
strict native reader, which rereads the exact source/manifest chain and every
physical tensor artifact.

Only a fully successful native read receives an immutable admission receipt.
The receipt is deliberately not a decoder, capability, HCLI, TPS, TG, or
tournament qualification claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
REQUEST_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_request.v1"
STATUS_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_status.v1"
RECEIPT_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1"
CURRENT_RECEIPT_POINTER_SCHEMA = (
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1"
)
IDENTITY_SCHEMA = "hawking.ascension.qwen_source_content_identity.v1"
REVALIDATION_SCHEMA = "hawking.ascension.complete_binary_source_revalidation.v1"
NATIVE_RESULT_SCHEMA = "hawking.ascension.qwen_complete_binary_native_admission_result.v1"
PACK_COMPLETE_PHASE = "EARNED_COMPLETE_PHYSICAL_BINARY_CANDIDATE_UNQUALIFIED"
MANIFEST_STATUS = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED"
NATIVE_RESULT_STATUS = "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
ADMISSION_RECEIPT_STATUS = "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
CURRENT_RECEIPT_POINTER_STATUS = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED"


class CompleteBinaryAdmissionError(RuntimeError):
    """An exact admission precondition or postcondition failed."""


@dataclass(frozen=True)
class AdmissionTarget:
    key: str
    prefix: str
    model_id: str
    repository: str
    revision: str
    manifest_schema: str
    complete_root: Path
    identity_path: Path

    @property
    def pack_status_path(self) -> Path:
        return self.complete_root / f"{self.prefix}_COMPLETE_GRAVITY_STATUS.json"

    @property
    def manifest_path(self) -> Path:
        return self.complete_root / f"{self.prefix}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"

    @property
    def revalidation_path(self) -> Path:
        return self.complete_root / f"{self.prefix}_CURRENT_SOURCE_SHARD_REVALIDATION.json"

    @property
    def root(self) -> Path:
        """Private mutable admission workspace.

        Requests and heartbeat status deliberately remain below this directory.
        The immutable public receipt is one level up because the physical
        gatekeeper's fixed contract consumes it directly from
        ``complete-gravity``.
        """

        return self.complete_root / "complete-admission"

    @property
    def requests_root(self) -> Path:
        return self.root / "requests"

    @property
    def receipt_path(self) -> Path:
        """Historical fixed receipt path retained as immutable evidence.

        Older campaigns published one immutable receipt at this fixed location.
        It cannot be overwritten when a later terminal manifest needs its own
        admission.  New current selection therefore uses the versioned receipt
        path plus :attr:`current_receipt_pointer_path` below.
        """

        return self.complete_root / f"{self.prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT.json"

    @property
    def receipts_root(self) -> Path:
        """Append-only immutable receipt history, keyed by manifest seal."""

        return self.root / "receipts"

    @property
    def current_receipt_pointer_path(self) -> Path:
        """Sealed mutable selector for the one terminal artifact in force."""

        return self.complete_root / (
            f"{self.prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_CURRENT.json"
        )

    @property
    def legacy_receipt_path(self) -> Path:
        """Pre-contract private receipt location, retained only for migration.

        A verified receipt found here is hard-linked into ``receipt_path`` and
        then removed from this old location.  We never reconstruct or reseal a
        historical positive result merely to change its address.
        """

        return self.root / f"{self.prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT.json"

    @property
    def status_path(self) -> Path:
        return self.root / f"{self.prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_STATUS.json"


TARGETS: Mapping[str, AdmissionTarget] = {
    "qwen30": AdmissionTarget(
        key="qwen30",
        prefix="QWEN30",
        model_id="Qwen3-Coder-30B-A3B-Instruct",
        repository="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        revision="b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
        manifest_schema="hawking.ascension.qwen30_complete_binary_gravity.v1",
        complete_root=(
            REPO_ROOT
            / "workspace/campaign/records/ascension-sandbox/physical/qwen30/complete-gravity"
        ),
        identity_path=(
            REPO_ROOT
            / "workspace/campaign/records/ascension-sandbox/physical/qwen30/evolution/SOURCE_CONTENT_IDENTITY.json"
        ),
    ),
    "qwen80": AdmissionTarget(
        key="qwen80",
        prefix="QWEN80",
        model_id="Qwen3-Coder-Next-80B",
        repository="Qwen/Qwen3-Coder-Next",
        revision="a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb",
        manifest_schema="hawking.ascension.qwen80_complete_binary_gravity.v1",
        complete_root=(
            REPO_ROOT
            / "workspace/campaign/records/ascension-sandbox/physical/qwen80/complete-gravity"
        ),
        identity_path=(
            REPO_ROOT
            / "workspace/campaign/records/ascension-sandbox/physical/qwen80/evolution/SOURCE_CONTENT_IDENTITY.json"
        ),
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise CompleteBinaryAdmissionError(f"{label} must be a lowercase 64-character SHA-256")
    return str(value)


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CompleteBinaryAdmissionError(f"{label} must be a non-empty string")
    return value


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompleteBinaryAdmissionError(f"{label} must be an object")
    return value


def _require_int(value: object, label: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise CompleteBinaryAdmissionError(f"{label} must be an integer")
    if positive and value <= 0:
        raise CompleteBinaryAdmissionError(f"{label} must be positive")
    return value


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise CompleteBinaryAdmissionError(f"JSON object has duplicate key {key!r}")
        output[key] = value
    return output


def _reject_non_finite(value: str) -> None:
    raise CompleteBinaryAdmissionError(f"JSON contains non-finite constant {value!r}")


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_non_finite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, CompleteBinaryAdmissionError) as exc:
        raise CompleteBinaryAdmissionError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise CompleteBinaryAdmissionError(f"{label} root must be an object")
    return decoded


def _regular_bytes(path: Path, label: str) -> tuple[bytes, dict[str, int]]:
    """Read one regular non-symlink file with a cheap mutation check."""

    try:
        before = os.lstat(path)
    except OSError as exc:
        raise CompleteBinaryAdmissionError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CompleteBinaryAdmissionError(f"{label} must be a regular non-symlink file: {path}")
    try:
        raw = path.read_bytes()
        after = os.lstat(path)
    except OSError as exc:
        raise CompleteBinaryAdmissionError(f"cannot read {label}: {exc}") from exc
    identity = {
        "bytes": int(before.st_size),
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "mtime_ns": int(before.st_mtime_ns),
        "ctime_ns": int(before.st_ctime_ns),
    }
    observed_after = {
        "bytes": int(after.st_size),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "mtime_ns": int(after.st_mtime_ns),
        "ctime_ns": int(after.st_ctime_ns),
    }
    if observed_after != identity or len(raw) != identity["bytes"]:
        raise CompleteBinaryAdmissionError(f"{label} changed while being read: {path}")
    return raw, identity


def _read_document(path: Path, label: str, *, sealed: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, file_identity = _regular_bytes(path, label)
    document = _parse_json(raw, label)
    seal_sha256: str | None = None
    if sealed:
        try:
            verified = verify(document, label=label)
        except SealIntegrityError as exc:
            raise CompleteBinaryAdmissionError(f"{label} seal is invalid: {exc}") from exc
        document = dict(verified)
        seal_sha256 = _require_sha256(document.get("seal_sha256"), f"{label}.seal_sha256")
    return document, {
        "path": str(path.resolve()),
        "document_sha256": _sha256_bytes(raw),
        "file_identity": file_identity,
        "seal_sha256": seal_sha256,
    }


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        node = os.lstat(path)
    except OSError as exc:
        raise CompleteBinaryAdmissionError(f"cannot stat admission directory {path}: {exc}") from exc
    if stat.S_ISLNK(node.st_mode) or not stat.S_ISDIR(node.st_mode):
        raise CompleteBinaryAdmissionError(f"admission directory must be a real directory: {path}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _ensure_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_immutable_json(path: Path, payload: Mapping[str, Any], label: str) -> None:
    """Create an immutable receipt/request, or require byte-equivalent reuse.

    ``link`` gives this write no-replace semantics on the same filesystem.  A
    stale receipt is never silently replaced by a different candidate.
    """

    _ensure_directory(path.parent)
    rendered = json.dumps(
        dict(payload), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ).encode("utf-8") + b"\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing_raw, _ = _regular_bytes(path, label)
            existing = _parse_json(existing_raw, label)
            if existing != dict(payload):
                raise CompleteBinaryAdmissionError(
                    f"refusing to overwrite a different immutable {label}: {path}"
                )
        else:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _same_resolved_path(observed: object, expected: Path, label: str) -> None:
    declared = Path(_require_string(observed, label))
    if not declared.is_absolute():
        raise CompleteBinaryAdmissionError(f"{label} must be an absolute path")
    try:
        actual = declared.resolve(strict=True)
        required = expected.resolve(strict=True)
    except OSError as exc:
        raise CompleteBinaryAdmissionError(f"cannot resolve {label}: {exc}") from exc
    if actual != required:
        raise CompleteBinaryAdmissionError(f"{label} does not bind the expected path")


def _immutable_identity_binding(target: AdmissionTarget) -> dict[str, Any]:
    document, meta = _read_document(target.identity_path, "immutable source content identity", sealed=True)
    if document.get("schema") != IDENTITY_SCHEMA:
        raise CompleteBinaryAdmissionError("immutable source content identity schema is not accepted")
    if document.get("status") != "IMMUTABLE_SOURCE_CONTENT_IDENTITY_BOUND":
        raise CompleteBinaryAdmissionError("immutable source content identity is not bound")
    model = _require_mapping(document.get("model"), "immutable identity.model")
    source = _require_mapping(document.get("source_content"), "immutable identity.source_content")
    for field, expected in (("id", target.model_id), ("repository", target.repository), ("revision", target.revision)):
        if model.get(field) != expected:
            raise CompleteBinaryAdmissionError(f"immutable identity model.{field} differs from target")
    for field, expected in (("repository", target.repository), ("revision", target.revision)):
        if source.get(field) != expected:
            raise CompleteBinaryAdmissionError(f"immutable identity source_content.{field} differs from target")
    source_dir = Path(_require_string(model.get("source_dir"), "immutable identity.model.source_dir"))
    if not source_dir.is_absolute():
        raise CompleteBinaryAdmissionError("immutable identity source directory must be absolute")
    content_identity_sha256 = _require_sha256(
        document.get("content_identity_sha256"), "immutable identity.content_identity_sha256"
    )
    controls = source.get("control_files")
    if not isinstance(controls, list):
        raise CompleteBinaryAdmissionError("immutable identity source_content.control_files must be an array")
    index_rows = [
        row for row in controls
        if isinstance(row, Mapping) and row.get("path") == "model.safetensors.index.json"
    ]
    if len(index_rows) != 1:
        raise CompleteBinaryAdmissionError("immutable identity must bind exactly one safetensors index")
    index_sha256 = _require_sha256(index_rows[0].get("sha256"), "immutable identity index SHA-256")
    return {
        "path": meta["path"],
        "document_sha256": meta["document_sha256"],
        "seal_sha256": meta["seal_sha256"],
        "content_identity_sha256": content_identity_sha256,
        "repository": target.repository,
        "revision": target.revision,
        "source_dir": str(source_dir.resolve()),
        "index_sha256": index_sha256,
        # This historical audit seal is recorded for provenance only.  Current
        # full-shard revalidation is authoritative because audit heartbeats can
        # legitimately reseal the same immutable source content.
        "historical_weight_body_audit_seal_sha256": _require_sha256(
            document.get("weight_body_audit_seal_sha256"),
            "immutable identity historical weight-body audit seal",
        ),
    }


def _current_revalidation_binding(
    target: AdmissionTarget, identity: Mapping[str, Any]
) -> dict[str, Any]:
    document, meta = _read_document(target.revalidation_path, "current source shard revalidation", sealed=True)
    if document.get("schema") != REVALIDATION_SCHEMA:
        raise CompleteBinaryAdmissionError("current source revalidation schema is not accepted")
    if document.get("status") != "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED":
        raise CompleteBinaryAdmissionError("current source revalidation is not earned")
    if document.get("source_repository") != target.repository:
        raise CompleteBinaryAdmissionError("current source revalidation repository differs from target")
    if document.get("source_revision") != target.revision:
        raise CompleteBinaryAdmissionError("current source revalidation revision differs from target")
    source_model_dir = Path(
        _require_string(document.get("source_model_dir"), "current revalidation.source_model_dir")
    )
    if not source_model_dir.is_absolute() or str(source_model_dir.resolve()) != identity["source_dir"]:
        raise CompleteBinaryAdmissionError("current revalidation source model directory differs from immutable identity")
    index_path = Path(_require_string(document.get("index_path"), "current revalidation.index_path"))
    if not index_path.is_absolute() or index_path.resolve() != source_model_dir.resolve() / "model.safetensors.index.json":
        raise CompleteBinaryAdmissionError("current revalidation index path differs from source model directory")
    index_sha256 = _require_sha256(document.get("index_sha256"), "current revalidation.index_sha256")
    if index_sha256 != identity["index_sha256"]:
        raise CompleteBinaryAdmissionError("current revalidation index SHA-256 differs from immutable source identity")
    shards = _require_mapping(document.get("shards"), "current revalidation.shards")
    shard_count = _require_int(document.get("sealed_shard_count"), "current revalidation.sealed_shard_count", positive=True)
    if len(shards) != shard_count:
        raise CompleteBinaryAdmissionError("current revalidation shard count does not match its shard map")
    for shard_name, row in shards.items():
        if not isinstance(shard_name, str) or not shard_name or "/" in shard_name or "\\" in shard_name:
            raise CompleteBinaryAdmissionError("current revalidation has an unsafe shard filename")
        entry = _require_mapping(row, f"current revalidation shard {shard_name}")
        expected_sha = _require_sha256(entry.get("expected_sha256"), f"revalidation expected SHA {shard_name}")
        if _require_sha256(entry.get("observed_sha256"), f"revalidation observed SHA {shard_name}") != expected_sha:
            raise CompleteBinaryAdmissionError(f"current revalidation hash mismatch for {shard_name}")
        _require_int(entry.get("expected_bytes"), f"revalidation expected bytes {shard_name}", positive=True)
        identity_row = _require_mapping(entry.get("file_identity"), f"revalidation file identity {shard_name}")
        if _require_int(identity_row.get("bytes"), f"revalidation file identity bytes {shard_name}", positive=True) != entry.get("expected_bytes"):
            raise CompleteBinaryAdmissionError(f"current revalidation bytes mismatch for {shard_name}")
    return {
        "path": meta["path"],
        "document_sha256": meta["document_sha256"],
        "seal_sha256": meta["seal_sha256"],
        "repository": target.repository,
        "revision": target.revision,
        "source_model_dir": str(source_model_dir.resolve()),
        "index_path": str(index_path.resolve()),
        "index_sha256": index_sha256,
        "source_audit_path": _require_string(
            document.get("source_audit_path"), "current revalidation.source_audit_path"
        ),
        "source_audit_document_sha256": _require_sha256(
            document.get("source_audit_document_sha256"), "current revalidation.source_audit_document_sha256"
        ),
        "source_audit_seal_sha256": _require_sha256(
            document.get("source_audit_seal_sha256"), "current revalidation.source_audit_seal_sha256"
        ),
        "sealed_shard_count": shard_count,
        "sealed_shard_hashes_sha256": _require_sha256(
            document.get("sealed_shard_hashes_sha256"), "current revalidation.sealed_shard_hashes_sha256"
        ),
        "weight_map_sha256": _require_sha256(
            document.get("weight_map_sha256"), "current revalidation.weight_map_sha256"
        ),
    }


def _manifest_binding(
    target: AdmissionTarget, revalidation: Mapping[str, Any]
) -> dict[str, Any]:
    document, meta = _read_document(target.manifest_path, "complete binary manifest", sealed=True)
    if document.get("schema") != target.manifest_schema:
        raise CompleteBinaryAdmissionError("complete binary manifest schema differs from target")
    if document.get("status") != MANIFEST_STATUS:
        raise CompleteBinaryAdmissionError("complete binary manifest is not an unqualified complete candidate")
    if _require_sha256(
        document.get("source_body_audit_seal_sha256"), "complete binary manifest source audit seal"
    ) != revalidation["source_audit_seal_sha256"]:
        raise CompleteBinaryAdmissionError("complete binary manifest audit seal differs from current revalidation")
    _same_resolved_path(
        document.get("source_revalidation_receipt_path"),
        target.revalidation_path,
        "complete binary manifest source revalidation receipt path",
    )
    if _require_sha256(
        document.get("source_revalidation_receipt_seal_sha256"),
        "complete binary manifest source revalidation receipt seal",
    ) != revalidation["seal_sha256"]:
        raise CompleteBinaryAdmissionError("complete binary manifest revalidation seal differs from current revalidation")
    source = _require_mapping(document.get("source"), "complete binary manifest.source")
    if source.get("repository") != target.repository:
        raise CompleteBinaryAdmissionError("complete binary manifest source repository differs from target")
    source_dir = Path(_require_string(source.get("model_dir"), "complete binary manifest source.model_dir"))
    if not source_dir.is_absolute() or str(source_dir.resolve()) != revalidation["source_model_dir"]:
        raise CompleteBinaryAdmissionError("complete binary manifest source directory differs from current revalidation")
    return {
        "path": meta["path"],
        "document_sha256": meta["document_sha256"],
        "seal_sha256": meta["seal_sha256"],
        "schema": target.manifest_schema,
        "status": MANIFEST_STATUS,
    }


def _build_request(target: AdmissionTarget) -> dict[str, Any]:
    identity = _immutable_identity_binding(target)
    revalidation = _current_revalidation_binding(target, identity)
    manifest = _manifest_binding(target, revalidation)
    return seal(
        {
            "schema": REQUEST_SCHEMA,
            "status": "SEALED_EXACT_COMPLETE_BINARY_ADMISSION_REQUEST",
            "request_version": 1,
            "model": {
                "key": target.key,
                "id": target.model_id,
                "repository": target.repository,
                "revision": target.revision,
                "native_core_model": target.key,
            },
            "immutable_source_identity": identity,
            "current_source_revalidation": revalidation,
            "complete_manifest": manifest,
            "native_admission": {
                "required_api": "hawking_core::model::qwen_complete_binary::admit_complete_binary_artifact",
                "rechecks_complete_catalog_payload_hash_layout_and_current_source_identity": True,
            },
            "claim_boundary": {
                "manifest_is_bound_by_exact_seal_and_raw_document_sha256": True,
                "source_content_identity_and_current_full_shard_revalidation_are_both_required": True,
                "raw_bf16_source_is_authority_teacher_not_tournament_participant": True,
                "not_native_decoder_runtime_capability_hcli_tps_tg_or_tournament_qualification": True,
            },
        }
    )


def _request_path(target: AdmissionTarget, request: Mapping[str, Any]) -> Path:
    manifest = _require_mapping(request.get("complete_manifest"), "admission request.complete_manifest")
    return target.requests_root / (
        f"{target.prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_REQUEST_"
        f"{_require_sha256(manifest.get('seal_sha256'), 'admission request manifest seal')}.json"
    )


def _versioned_receipt_path(target: AdmissionTarget, request: Mapping[str, Any]) -> Path:
    """Return the append-only receipt address for this exact manifest.

    The manifest seal is part of the filename as well as the sealed receipt
    body.  A new terminal manifest therefore gets a new immutable record;
    neither a previous valid receipt nor a current pointer is rewritten as if
    it had admitted a different physical artifact.
    """

    manifest = _require_mapping(request.get("complete_manifest"), "admission request.complete_manifest")
    manifest_seal = _require_sha256(
        manifest.get("seal_sha256"), "admission request manifest seal"
    )
    return target.receipts_root / (
        f"{target.prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT_{manifest_seal}.json"
    )


@dataclass(frozen=True)
class ReceiptSelection:
    """One verified immutable receipt selected for the current request."""

    receipt: dict[str, Any]
    path: Path
    metadata: dict[str, Any]
    source: str


def _wait_for_complete_pack(target: AdmissionTarget) -> tuple[str, dict[str, Any]] | None:
    if not os.path.lexists(target.manifest_path):
        return "WAITING_FOR_COMPLETE_BINARY_MANIFEST", {
            "manifest_path": str(target.manifest_path),
            "pack_status_path": str(target.pack_status_path),
        }
    # A manifest object without the packer's terminal status is not a completed
    # candidate.  The native reader remains the authorization mechanism later.
    if not os.path.lexists(target.pack_status_path):
        return "WAITING_FOR_COMPLETE_GRAVITY_STATUS", {
            "manifest_path": str(target.manifest_path),
            "pack_status_path": str(target.pack_status_path),
        }
    try:
        raw, _ = _regular_bytes(target.pack_status_path, "complete gravity status")
        status = _parse_json(raw, "complete gravity status")
    except CompleteBinaryAdmissionError:
        return "WAITING_FOR_COMPLETE_GRAVITY_STATUS", {
            "manifest_path": str(target.manifest_path),
            "pack_status_path": str(target.pack_status_path),
            "detail": "status is missing, mutable, or not usable yet",
        }
    if status.get("schema") != target.manifest_schema or status.get("phase") != PACK_COMPLETE_PHASE:
        return "WAITING_FOR_COMPLETE_PACK", {
            "manifest_path": str(target.manifest_path),
            "pack_status_path": str(target.pack_status_path),
            "observed_phase": status.get("phase"),
        }
    progress = status.get("progress")
    if not isinstance(progress, Mapping):
        return "WAITING_FOR_COMPLETE_PACK", {"detail": "terminal pack status has no progress object"}
    try:
        planned = _require_int(progress.get("planned_tensors"), "complete pack planned tensor count", positive=True)
        completed = _require_int(progress.get("completed_tensors"), "complete pack completed tensor count", positive=True)
    except CompleteBinaryAdmissionError:
        return "WAITING_FOR_COMPLETE_PACK", {"detail": "terminal pack status has invalid progress counts"}
    if planned != completed:
        return "WAITING_FOR_COMPLETE_PACK", {
            "detail": "pack progress is not complete",
            "planned_tensors": planned,
            "completed_tensors": completed,
        }
    if status.get("manifest_path") != str(target.manifest_path):
        return "WAITING_FOR_COMPLETE_PACK", {"detail": "terminal pack status does not bind the expected manifest"}
    return None


def _native_loader_digest(path: Path) -> str:
    raw, _ = _regular_bytes(path, "native complete-binary admission executable")
    try:
        node = os.lstat(path)
    except OSError as exc:
        raise CompleteBinaryAdmissionError(f"cannot stat native admission executable: {exc}") from exc
    if not node.st_mode & stat.S_IXUSR:
        raise CompleteBinaryAdmissionError("native complete-binary admission executable is not owner-executable")
    return _sha256_bytes(raw)


NativeRunner = Callable[[Sequence[str], float], subprocess.CompletedProcess[str]]


def _default_native_runner(command: Sequence[str], timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _invoke_native_loader(
    *,
    target: AdmissionTarget,
    request: Mapping[str, Any],
    native_loader: Path,
    timeout_seconds: float,
    runner: NativeRunner,
) -> dict[str, Any]:
    if timeout_seconds <= 0:
        raise CompleteBinaryAdmissionError("native admission timeout must be positive")
    before_digest = _native_loader_digest(native_loader)
    manifest = _require_mapping(request.get("complete_manifest"), "admission request.complete_manifest")
    revalidation = _require_mapping(
        request.get("current_source_revalidation"), "admission request.current_source_revalidation"
    )
    command = [
        str(native_loader.resolve()),
        "--model",
        target.key,
        "--manifest",
        _require_string(manifest.get("path"), "admission request manifest path"),
        "--expected-manifest-seal-sha256",
        _require_sha256(manifest.get("seal_sha256"), "admission request manifest seal"),
        "--expected-source-audit-seal-sha256",
        _require_sha256(revalidation.get("source_audit_seal_sha256"), "admission request source audit seal"),
        "--expected-source-revision",
        target.revision,
    ]
    try:
        completed = runner(command, timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise CompleteBinaryAdmissionError(
            f"native complete-binary admission timed out after {timeout_seconds:g} seconds"
        ) from exc
    except OSError as exc:
        raise CompleteBinaryAdmissionError(f"cannot execute native complete-binary admission: {exc}") from exc
    if _native_loader_digest(native_loader) != before_digest:
        raise CompleteBinaryAdmissionError("native complete-binary admission executable changed while it ran")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "native admission returned no detail").strip()
        raise CompleteBinaryAdmissionError(
            f"native complete-binary admission refused the candidate (exit={completed.returncode}): {detail[:1000]}"
        )
    result = _parse_json((completed.stdout or "").encode("utf-8"), "native complete-binary admission result")
    if result.get("schema") != NATIVE_RESULT_SCHEMA or result.get("status") != NATIVE_RESULT_STATUS:
        raise CompleteBinaryAdmissionError("native complete-binary admission result does not declare strict success")
    if result.get("model") != target.key:
        raise CompleteBinaryAdmissionError("native complete-binary admission result model differs from target")
    _same_resolved_path(result.get("manifest_path"), target.manifest_path, "native result manifest path")
    if _require_sha256(result.get("manifest_seal_sha256"), "native result manifest seal") != manifest["seal_sha256"]:
        raise CompleteBinaryAdmissionError("native complete-binary admission result manifest seal differs from request")
    if _require_sha256(result.get("source_audit_seal_sha256"), "native result audit seal") != revalidation["source_audit_seal_sha256"]:
        raise CompleteBinaryAdmissionError("native complete-binary admission result audit seal differs from request")
    if result.get("source_revision") != target.revision:
        raise CompleteBinaryAdmissionError("native complete-binary admission result source revision differs from request")
    _same_resolved_path(result.get("source_index_path"), Path(revalidation["index_path"]), "native result source index path")
    tensor_count = _require_int(result.get("tensor_count"), "native result tensor count", positive=True)
    source_weight_elements = _require_int(
        result.get("source_weight_elements"), "native result source weight elements", positive=True
    )
    tensor_payload_bytes = _require_int(
        result.get("tensor_payload_bytes"), "native result tensor payload bytes", positive=True
    )
    return {
        "executable_path": str(native_loader.resolve()),
        "executable_sha256": before_digest,
        "api": "hawking_core::model::qwen_complete_binary::admit_complete_binary_artifact",
        "result_schema": NATIVE_RESULT_SCHEMA,
        "result_status": NATIVE_RESULT_STATUS,
        "tensor_count": tensor_count,
        "source_weight_elements": source_weight_elements,
        "tensor_payload_bytes": tensor_payload_bytes,
    }


def _fsync_directory(path: Path, label: str) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError as exc:
        raise CompleteBinaryAdmissionError(f"cannot open {label} directory for sync: {exc}") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise CompleteBinaryAdmissionError(f"cannot sync {label} directory: {exc}") from exc
    finally:
        os.close(descriptor)


def _validate_existing_receipt(
    *,
    target: AdmissionTarget,
    request: Mapping[str, Any],
    path: Path,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify a prior positive receipt against the exact current request.

    This is intentionally stronger than checking its seal alone: a historical
    receipt may be genuine but bind an older source revalidation or a different
    complete artifact, neither of which can be silently promoted or moved.
    """

    receipt, metadata = _read_document(path, label, sealed=True)
    if receipt.get("schema") != RECEIPT_SCHEMA or receipt.get("status") != ADMISSION_RECEIPT_STATUS:
        raise CompleteBinaryAdmissionError(f"{label} is invalid and will not be overwritten")
    model = _require_mapping(receipt.get("model"), f"{label}.model")
    for field, expected in (
        ("key", target.key),
        ("id", target.model_id),
        ("repository", target.repository),
        ("revision", target.revision),
    ):
        if model.get(field) != expected:
            raise CompleteBinaryAdmissionError(f"{label} model.{field} differs from target")
    if receipt.get("admission_request_seal_sha256") != request.get("seal_sha256"):
        raise CompleteBinaryAdmissionError(
            f"{label} binds a different exact request and will not be overwritten"
        )
    _same_resolved_path(
        receipt.get("admission_request_path"),
        _request_path(target, request),
        f"{label}.admission_request_path",
    )
    for field in ("immutable_source_identity", "current_source_revalidation", "complete_manifest"):
        if _require_mapping(receipt.get(field), f"{label}.{field}") != _require_mapping(
            request.get(field), f"admission request.{field}"
        ):
            raise CompleteBinaryAdmissionError(f"{label} {field} differs from the exact current request")
    native = _require_mapping(receipt.get("native_loader"), f"{label}.native_loader")
    if native.get("api") != "hawking_core::model::qwen_complete_binary::admit_complete_binary_artifact":
        raise CompleteBinaryAdmissionError(f"{label} does not bind the strict native complete-binary API")
    _require_sha256(native.get("executable_sha256"), f"{label}.native_loader.executable_sha256")
    _require_int(native.get("tensor_count"), f"{label}.native_loader.tensor_count", positive=True)
    _require_int(
        native.get("source_weight_elements"),
        f"{label}.native_loader.source_weight_elements",
        positive=True,
    )
    _require_int(
        native.get("tensor_payload_bytes"),
        f"{label}.native_loader.tensor_payload_bytes",
        positive=True,
    )
    boundary = _require_mapping(receipt.get("claim_boundary"), f"{label}.claim_boundary")
    for field in (
        "native_complete_catalog_payload_hash_layout_and_source_chain_admission_passed",
        "admission_does_not_implement_or_claim_a_native_qwen_decoder",
        "admission_does_not_claim_capability_hcli_tps_tg_or_tournament_qualification",
        "raw_bf16_source_remains_authority_teacher_only",
    ):
        if boundary.get(field) is not True:
            raise CompleteBinaryAdmissionError(f"{label}.claim_boundary.{field} must be true")
    return receipt, metadata


def _remove_legacy_receipt_after_publication(
    *, target: AdmissionTarget, expected_document_sha256: str
) -> None:
    """Remove the old directory entry only after revalidating its exact bytes."""

    legacy = target.legacy_receipt_path
    if not os.path.lexists(legacy):
        return
    raw, _ = _regular_bytes(legacy, "legacy complete binary admission receipt")
    if _sha256_bytes(raw) != expected_document_sha256:
        raise CompleteBinaryAdmissionError(
            "legacy admission receipt changed during public-path reconciliation"
        )
    try:
        os.unlink(legacy)
    except OSError as exc:
        raise CompleteBinaryAdmissionError(f"cannot remove reconciled legacy admission receipt: {exc}") from exc
    _fsync_directory(legacy.parent, "legacy complete-admission")


def _publish_verified_legacy_receipt(
    *,
    target: AdmissionTarget,
    request: Mapping[str, Any],
    legacy_receipt: Mapping[str, Any],
    legacy_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish a validated old receipt at the public contract path.

    A hard link preserves the original sealed JSON byte-for-byte.  It avoids a
    second, divergent admission receipt and means migration does not rerun the
    native reader or reseal a past result.
    """

    legacy = target.legacy_receipt_path
    public = target.receipt_path
    _ensure_directory(public.parent)
    try:
        os.link(legacy, public, follow_symlinks=False)
    except FileExistsError:
        # Another worker may have completed the same reconciliation.  Its
        # public receipt must be exactly the one we independently validated.
        pass
    except OSError as exc:
        raise CompleteBinaryAdmissionError(
            f"cannot link verified legacy admission receipt into the public contract path: {exc}"
        ) from exc
    else:
        _fsync_directory(public.parent, "public complete-gravity")

    public_receipt, public_metadata = _validate_existing_receipt(
        target=target,
        request=request,
        path=public,
        label="public complete binary admission receipt",
    )
    if public_metadata["document_sha256"] != legacy_metadata["document_sha256"]:
        raise CompleteBinaryAdmissionError(
            "public and legacy admission receipts differ; refusing divergent receipt reconciliation"
        )
    if public_receipt != dict(legacy_receipt):
        raise CompleteBinaryAdmissionError(
            "public and legacy admission receipts have different sealed contents"
        )
    _remove_legacy_receipt_after_publication(
        target=target,
        expected_document_sha256=str(legacy_metadata["document_sha256"]),
    )
    return public_receipt


def _matching_existing_receipt(
    target: AdmissionTarget, request: Mapping[str, Any]
) -> ReceiptSelection | None:
    """Return only a receipt that binds the exact live terminal request.

    A historical fixed-path receipt can be perfectly valid yet bind an older
    manifest.  That must neither be overwritten nor turn a later physical
    manifest into a permanent ``BLOCKED`` state.  The append-only versioned
    path is authoritative for a replacement manifest; old public/private
    locations are consulted only as backwards-compatible exact matches.
    """

    versioned = _versioned_receipt_path(target, request)
    if os.path.lexists(versioned):
        receipt, metadata = _validate_existing_receipt(
            target=target,
            request=request,
            path=versioned,
            label="versioned complete binary admission receipt",
        )
        return ReceiptSelection(
            receipt=receipt,
            path=versioned,
            metadata=metadata,
            source="VERSIONED_CURRENT_MANIFEST",
        )

    # Fixed paths are immutable historical evidence.  A mismatch is expected
    # after a packer has advanced/resealed a terminal artifact; do not alter or
    # delete that old evidence, and do not allow it to block a fresh strict
    # scan for the new manifest.
    for path, label, source in (
        (target.receipt_path, "historical public complete binary admission receipt", "HISTORICAL_PUBLIC_EXACT_MATCH"),
        (target.legacy_receipt_path, "historical private complete binary admission receipt", "HISTORICAL_PRIVATE_EXACT_MATCH"),
    ):
        if not os.path.lexists(path):
            continue
        try:
            receipt, metadata = _validate_existing_receipt(
                target=target,
                request=request,
                path=path,
                label=label,
            )
        except CompleteBinaryAdmissionError:
            continue
        return ReceiptSelection(
            receipt=receipt,
            path=path,
            metadata=metadata,
            source=source,
        )
    return None


def _publish_current_receipt_pointer(
    *,
    target: AdmissionTarget,
    request: Mapping[str, Any],
    selection: ReceiptSelection,
) -> dict[str, Any]:
    """Publish the current manifest selector after exact receipt validation.

    The pointer is intentionally mutable because the physical *current*
    manifest can move before it is frozen.  Its target and every binding are
    sealed, while the target receipt itself is immutable and append-only.
    """

    receipt, metadata = _validate_existing_receipt(
        target=target,
        request=request,
        path=selection.path,
        label="selected complete binary admission receipt",
    )
    if receipt != selection.receipt:
        raise CompleteBinaryAdmissionError(
            "selected immutable admission receipt changed while publishing current pointer"
        )
    manifest = _require_mapping(request.get("complete_manifest"), "admission request complete manifest")
    request_path = _request_path(target, request)
    pointer = seal(
        {
            "schema": CURRENT_RECEIPT_POINTER_SCHEMA,
            "status": CURRENT_RECEIPT_POINTER_STATUS,
            "pointer_version": 1,
            "recorded_at": _utc_now(),
            "model": {
                "key": target.key,
                "id": target.model_id,
                "repository": target.repository,
                "revision": target.revision,
            },
            "complete_manifest": dict(manifest),
            "admission_request_path": str(request_path.resolve()),
            "admission_request_seal_sha256": _require_sha256(
                request.get("seal_sha256"), "admission request seal"
            ),
            "admission_receipt": {
                "path": str(selection.path.resolve()),
                "document_sha256": _require_sha256(
                    metadata.get("document_sha256"), "selected admission receipt document SHA-256"
                ),
                "seal_sha256": _require_sha256(
                    receipt.get("seal_sha256"), "selected admission receipt seal"
                ),
                "selection_source": selection.source,
            },
            "claim_boundary": {
                "pointer_selects_only_a_receipt_matching_the_current_complete_manifest": True,
                "historical_receipts_are_preserved_not_overwritten_or_resealed": True,
                "pointer_is_storage_artifact_admission_only_not_runtime_or_qualification": True,
            },
        }
    )
    _atomic_json(target.current_receipt_pointer_path, pointer)
    return pointer


def _publish_legacy_public_alias_if_absent(
    *, target: AdmissionTarget, versioned_receipt_path: Path
) -> None:
    """Give untouched first-generation consumers a byte-identical alias.

    This is only a compatibility publication for an otherwise absent fixed
    path.  It never replaces a historical receipt, and hard-linking means the
    alias is the same immutable bytes rather than a second resealed result.
    Current consumers must still follow the sealed current pointer.
    """

    public = target.receipt_path
    if os.path.lexists(public):
        return
    _ensure_directory(public.parent)
    try:
        os.link(versioned_receipt_path, public, follow_symlinks=False)
    except FileExistsError:
        return
    except OSError as exc:
        raise CompleteBinaryAdmissionError(
            f"cannot publish immutable compatibility admission alias: {exc}"
        ) from exc
    _fsync_directory(public.parent, "public complete-gravity")


def _receipt_for_success(
    *, target: AdmissionTarget,
    request: Mapping[str, Any],
    request_path: Path,
    native: Mapping[str, Any],
) -> dict[str, Any]:
    identity = _require_mapping(request.get("immutable_source_identity"), "admission request immutable identity")
    revalidation = _require_mapping(
        request.get("current_source_revalidation"), "admission request current revalidation"
    )
    manifest = _require_mapping(request.get("complete_manifest"), "admission request complete manifest")
    return seal(
        {
            "schema": RECEIPT_SCHEMA,
            "status": ADMISSION_RECEIPT_STATUS,
            "recorded_at": _utc_now(),
            "model": {
                "key": target.key,
                "id": target.model_id,
                "repository": target.repository,
                "revision": target.revision,
            },
            "admission_request_path": str(request_path.resolve()),
            "admission_request_seal_sha256": _require_sha256(
                request.get("seal_sha256"), "admission request seal"
            ),
            "immutable_source_identity": identity,
            "current_source_revalidation": revalidation,
            "complete_manifest": manifest,
            "native_loader": dict(native),
            "claim_boundary": {
                "native_complete_catalog_payload_hash_layout_and_source_chain_admission_passed": True,
                "admission_does_not_implement_or_claim_a_native_qwen_decoder": True,
                "admission_does_not_claim_capability_hcli_tps_tg_or_tournament_qualification": True,
                "raw_bf16_source_remains_authority_teacher_only": True,
            },
        }
    )


def _publish_status(target: AdmissionTarget, phase: str, **fields: Any) -> dict[str, Any]:
    prior: dict[str, Any] = {}
    if target.status_path.exists():
        try:
            prior, _ = _read_document(target.status_path, "prior admission status", sealed=True)
        except CompleteBinaryAdmissionError:
            # A malformed mutable heartbeat cannot be allowed to prevent a
            # correct new heartbeat; it is not an admission authority.
            prior = {}
    payload = seal(
        {
            "schema": STATUS_SCHEMA,
            "recorded_at": _utc_now(),
            "pid": os.getpid(),
            "heartbeat": int(prior.get("heartbeat", 0)) + 1,
            "model": {"key": target.key, "id": target.model_id, "repository": target.repository},
            "phase": phase,
            "claim_boundary": {
                "receipt_is_written_only_after_native_complete_binary_admission": True,
                "waiting_or_blocked_status_is_not_admission": True,
                "not_runtime_capability_hcli_tps_tg_or_tournament_qualification": True,
            },
            **fields,
        }
    )
    _atomic_json(target.status_path, payload)
    return payload


def run_once(
    target: AdmissionTarget,
    *,
    native_loader: Path,
    timeout_seconds: float = 3600.0,
    runner: NativeRunner = _default_native_runner,
) -> dict[str, Any]:
    """Perform one safe admission check without inventing a positive result."""

    try:
        waiting = _wait_for_complete_pack(target)
        if waiting is not None:
            phase, fields = waiting
            return _publish_status(target, phase, **fields)
        request = _build_request(target)
        request_path = _request_path(target, request)
        _write_immutable_json(request_path, request, "complete binary admission request")
        existing = _matching_existing_receipt(target, request)
        if existing is not None:
            pointer = _publish_current_receipt_pointer(
                target=target,
                request=request,
                selection=existing,
            )
            return _publish_status(
                target,
                "EARNED_COMPLETE_BINARY_ADMISSION_RECEIPT_REUSED",
                admission_request_path=str(request_path),
                admission_request_seal_sha256=request["seal_sha256"],
                admission_receipt_path=str(existing.path),
                admission_receipt_seal_sha256=existing.receipt["seal_sha256"],
                current_receipt_pointer_path=str(target.current_receipt_pointer_path),
                current_receipt_pointer_seal_sha256=pointer["seal_sha256"],
                receipt_selection_source=existing.source,
            )
        _publish_status(
            target,
            "NATIVE_COMPLETE_BINARY_ADMISSION_IN_PROGRESS",
            admission_request_path=str(request_path),
            admission_request_seal_sha256=request["seal_sha256"],
            manifest_path=request["complete_manifest"]["path"],
            manifest_seal_sha256=request["complete_manifest"]["seal_sha256"],
        )
        native = _invoke_native_loader(
            target=target,
            request=request,
            native_loader=native_loader,
            timeout_seconds=timeout_seconds,
            runner=runner,
        )
        # Reject a source identity, revalidation, or manifest replacement that
        # raced the native all-artifact pass, even if a replacement was itself
        # syntactically sealed.
        if _build_request(target) != request:
            raise CompleteBinaryAdmissionError(
                "source identity, current revalidation, or exact manifest changed during native admission"
            )
        receipt = _receipt_for_success(
            target=target,
            request=request,
            request_path=request_path,
            native=native,
        )
        receipt_path = _versioned_receipt_path(target, request)
        _write_immutable_json(receipt_path, receipt, "versioned complete binary admission receipt")
        _publish_legacy_public_alias_if_absent(
            target=target,
            versioned_receipt_path=receipt_path,
        )
        selected_receipt, selected_metadata = _validate_existing_receipt(
            target=target,
            request=request,
            path=receipt_path,
            label="new versioned complete binary admission receipt",
        )
        pointer = _publish_current_receipt_pointer(
            target=target,
            request=request,
            selection=ReceiptSelection(
                receipt=selected_receipt,
                path=receipt_path,
                metadata=selected_metadata,
                source="VERSIONED_NEW_NATIVE_SCAN",
            ),
        )
        return _publish_status(
            target,
            "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED",
            admission_request_path=str(request_path),
            admission_request_seal_sha256=request["seal_sha256"],
            admission_receipt_path=str(receipt_path),
            admission_receipt_seal_sha256=receipt["seal_sha256"],
            current_receipt_pointer_path=str(target.current_receipt_pointer_path),
            current_receipt_pointer_seal_sha256=pointer["seal_sha256"],
            receipt_selection_source="VERSIONED_NEW_NATIVE_SCAN",
            native_loader=native,
        )
    except CompleteBinaryAdmissionError as exc:
        return _publish_status(
            target,
            "BLOCKED_COMPLETE_BINARY_ADMISSION_FAIL_CLOSED",
            detail=str(exc),
            manifest_path=str(target.manifest_path),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(TARGETS))
    parser.add_argument("command", choices=("once", "watch"), nargs="?", default="once")
    parser.add_argument(
        "--native-loader",
        type=Path,
        default=(
            REPO_ROOT
            / "workspace/ops/build/rust/debug/examples/ascension_qwen_complete_binary_admission"
        ),
        help="prebuilt strict Hawking Core admission executable",
    )
    parser.add_argument("--idle-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = TARGETS[args.model]
    if args.idle_seconds <= 0:
        raise SystemExit("--idle-seconds must be positive")
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if args.command == "once":
        status = run_once(
            target,
            native_loader=args.native_loader,
            timeout_seconds=args.timeout_seconds,
        )
        print(json.dumps(status, sort_keys=True))
        return 0 if not str(status.get("phase", "")).startswith("BLOCKED") else 2
    while True:
        try:
            run_once(
                target,
                native_loader=args.native_loader,
                timeout_seconds=args.timeout_seconds,
            )
        except BaseException as exc:  # Keep the detached watcher alive after an I/O surprise.
            try:
                _publish_status(target, "BLOCKED_COMPLETE_BINARY_ADMISSION_FAIL_CLOSED", detail=repr(exc))
            except BaseException:
                pass
        time.sleep(args.idle_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
