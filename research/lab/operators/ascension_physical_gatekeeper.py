"""Fail-closed physical Ascension tournament gatekeeper.

This is deliberately separate from the historical V3 controller.  It is a
small, durable gate for the two real Qwen Gravity campaigns.  It consumes only
a fixed set of source-bound, sealed receipts.  It never loads a model, scores
candidates, selects a winner, or activates a sandbox.  Once (and only once)
both fully qualified managers and the frozen protected suite are present, it
may detach the exact protected evaluator runner.

The fixed receipt contract below is intentionally strict.  Existing component
probes, partial packs, worker heartbeats, and unsealed watchdog status files
are useful operational observations, but none can satisfy a tournament gate.
Future runtime and evaluation lanes must emit the named receipts with the
required bindings before this gatekeeper can present the campaign for protected
final review.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import signal
import stat
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from lab.operators import ascension_physical_tournament as physical_tournament
from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PHYSICAL_ROOT = (
    REPO_ROOT / "workspace" / "campaign" / "records" / "ascension-sandbox" / "physical"
)

GATE_SCHEMA = "hawking.ascension.physical_tournament_gate.v1"
WORKFLOW_SCHEMA = "hawking.ascension.physical_tournament_workflow.v1"
SOURCE_IDENTITY_SCHEMA = "hawking.ascension.qwen_source_content_identity.v1"
SOURCE_REVALIDATION_SCHEMA = "hawking.ascension.complete_binary_source_revalidation.v1"
# The packer-side native admission operator owns this public, immutable
# storage-artifact receipt.  It is intentionally *not* a runtime, manager,
# capability, HCLI, TPS, TG, or tournament qualification receipt.
ARTIFACT_ADMISSION_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1"
ARTIFACT_ADMISSION_STATUS = (
    "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
)
ARTIFACT_ADMISSION_CURRENT_POINTER_SCHEMA = (
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1"
)
ARTIFACT_ADMISSION_CURRENT_POINTER_STATUS = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED"
ARTIFACT_ADMISSION_REQUEST_SCHEMA = "hawking.ascension.qwen_complete_binary_gravity_admission_request.v1"
ARTIFACT_ADMISSION_REQUEST_STATUS = "SEALED_EXACT_COMPLETE_BINARY_ADMISSION_REQUEST"
COMPLETE_BINARY_MANIFEST_STATUS = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED"
NATIVE_COMPLETE_BINARY_RESULT_SCHEMA = "hawking.ascension.qwen_complete_binary_native_admission_result.v1"
NATIVE_COMPLETE_BINARY_RESULT_STATUS = (
    "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED"
)
NATIVE_COMPLETE_BINARY_API = "hawking_core::model::qwen_complete_binary::admit_complete_binary_artifact"
RUNTIME_SCHEMA = "hawking.ascension.physical_exact_full_token_runtime.v1"
RUNTIME_SUPERSESSION_SCHEMA = (
    "hawking.ascension.physical_exact_full_token_runtime_supersession.v1"
)
RUNTIME_PASS_STATUS = "PASS_EXACT_NATIVE_FULL_TOKEN_RUNTIME"
RUNTIME_REVOCATION_STATUS_PREFIX = "REVOKED_"
HCLI_SCHEMA = "hawking.ascension.physical_hcli_measurement.v1"
KERNEL_SCHEMA = "hawking.ascension.physical_custom_kernel_operational.v1"
TG10_SCHEMA = "hawking.ascension.qwen_tg_operational_pass.v1"
TG10_STATUS = "PASS"
TG3_SCHEMA = "hawking.ascension.physical_tg3_qualification.v1"
CAPABILITY_SCHEMA = "hawking.ascension.physical_capability_evaluation.v1"
FINAL_REVIEW_SCHEMA = "hawking.ascension.physical_protected_final_review_marker.v1"
MANAGER_OPERATIONS_SCHEMA = "hawking.ascension.physical_final_manager_operations.v1"
MANAGER_OPERATIONS_STATUS = "PASS_FINAL_MANAGER_OPERATIONS"

GATE_FILENAME = "ASCENSION_PHYSICAL_TOURNAMENT_GATE_STATUS.json"
WORKFLOW_FILENAME = "ASCENSION_PHYSICAL_TOURNAMENT_WORKFLOW.json"
OPERATIONAL_ASCENT_FILENAME = "ASCENSION_OPERATIONAL_ASCENT_STATUS.json"
FINAL_REVIEW_FILENAME = "ASCENSION_PHYSICAL_PROTECTED_FINAL_REVIEW_MARKER.json"
LOCK_FILENAME = ".ascension-physical-gatekeeper.lock"

OPERATIONAL_ASCENT_SCHEMA = "hawking.ascension.physical_operational_ascent.v1"
OPERATIONAL_ASCENT_WAITING = "WAITING_FOR_BOTH_VALID_TG10_OPERATIONAL_RECEIPTS"
OPERATIONAL_ASCENT_EARNED = "BOTH_TG10_OPERATIONAL_ASCENT_EARNED_CONTINUING_TO_TG3"

MINIMUM_OPERATIONAL_TPS = 100.0
TG3_TPS = 333.0
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
# Complete manifests are the sole intentionally large JSON documents.  Keep
# their envelope separate from ordinary receipts: raising this bound globally
# would make every untrusted receipt parser accept large inputs.
MEBIBYTE = 1024 * 1024
QWEN30_COMPLETE_MANIFEST_MAX_BYTES = 64 * MEBIBYTE
# The current Qwen80 candidate's sealed physical ledger bills exactly
# 77,842,421 manifest bytes.  Its limit is the next whole MiB only, so this is
# a narrow model-specific envelope (3,946,507 bytes of headroom), not a broad
# relaxation for arbitrary receipts or future unknown manifests.
QWEN80_AUDITED_COMPLETE_MANIFEST_BYTES = 77_842_421
QWEN80_COMPLETE_MANIFEST_MAX_BYTES = 78 * MEBIBYTE


@dataclass(frozen=True)
class ModelSpec:
    """Immutable identity and fixed receipt locations for one contender."""

    key: str
    prefix: str
    model_id: str
    architecture: str
    repository: str
    revision: str
    shard_count: int
    gravity_artifact_id: str


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="qwen30",
        prefix="QWEN30",
        model_id="Qwen3-Coder-30B-A3B-Instruct",
        architecture="Qwen3MoeForCausalLM",
        repository="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        revision="b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
        shard_count=16,
        gravity_artifact_id="Qwen30-Gravity-Manager-Artifact",
    ),
    ModelSpec(
        key="qwen80",
        prefix="QWEN80",
        model_id="Qwen3-Coder-Next-80B",
        architecture="Qwen3NextForCausalLM",
        repository="Qwen/Qwen3-Coder-Next",
        revision="a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb",
        shard_count=40,
        gravity_artifact_id="Qwen80-Gravity-Manager-Artifact",
    ),
)


def _complete_manifest_max_bytes(spec: ModelSpec) -> int:
    """Return the narrow read envelope for this contender's tensor catalog.

    This is deliberately keyed by the immutable contender identity rather than
    being a second generic receipt limit.  Qwen80's current verified catalog
    genuinely exceeds Qwen30's 64 MiB envelope; unknown contenders remain
    fail-closed instead of inheriting Qwen80's allowance.
    """

    if spec.key == "qwen30":
        return QWEN30_COMPLETE_MANIFEST_MAX_BYTES
    if spec.key == "qwen80":
        return QWEN80_COMPLETE_MANIFEST_MAX_BYTES
    raise PhysicalGatekeeperError(
        f"no complete-manifest byte envelope is defined for contender {spec.key!r}"
    )


class PhysicalGatekeeperError(RuntimeError):
    """Raised only for an unsafe gatekeeper invocation or write failure."""


@dataclass
class LoadedReceipt:
    path: Path
    present: bool
    sealed: bool
    document: dict[str, Any] | None
    seal_sha256: str | None
    document_sha256: str | None
    errors: list[str]


@dataclass
class Check:
    requirement: str
    passed: bool
    path: Path
    seal_sha256: str | None
    reasons: list[str]
    details: dict[str, Any]
    document: dict[str, Any] | None = None
    document_sha256: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement,
            "state": "PASS" if self.passed else "BLOCKED",
            "path": str(self.path),
            "seal_sha256": self.seal_sha256,
            "document_sha256": self.document_sha256,
            "reasons": list(self.reasons),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class SourceBinding:
    content_identity_sha256: str
    identity_seal_sha256: str
    revalidation_seal_sha256: str
    source_dir: Path
    weight_shard_count: int
    control_file_count: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest(value: Any) -> str:
    raw = value if isinstance(value, (bytes, bytearray)) else _canonical(value)
    return hashlib.sha256(raw).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    candidate = float(value)
    return candidate if math.isfinite(candidate) else None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _under(root: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def _regular_file_identity(path: Path, *, label: str) -> dict[str, int]:
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise PhysicalGatekeeperError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(observed.st_mode):
        raise PhysicalGatekeeperError(f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(observed.st_mode):
        raise PhysicalGatekeeperError(f"{label} must be a regular file: {path}")
    return {
        "bytes": int(observed.st_size),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "mtime_ns": int(observed.st_mtime_ns),
        "ctime_ns": int(observed.st_ctime_ns),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_json(
    path: Path, *, max_bytes: int = MAX_RECEIPT_BYTES
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    try:
        identity = _regular_file_identity(path, label="JSON receipt")
    except PhysicalGatekeeperError as exc:
        return None, None, [str(exc)]
    if identity["bytes"] > max_bytes:
        return None, None, [f"receipt exceeds {max_bytes} byte safety limit"]
    try:
        raw = path.read_bytes()
        after = _regular_file_identity(path, label="JSON receipt")
        if after != identity or len(raw) != identity["bytes"]:
            return None, None, ["receipt changed while being read"]
        parsed = json.loads(raw.decode("utf-8"))
    except (PhysicalGatekeeperError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, None, [f"cannot read JSON: {type(exc).__name__}: {exc}"]
    if not isinstance(parsed, Mapping):
        return None, None, ["receipt root is not a JSON object"]
    return dict(parsed), _digest(raw), []


def _load_sealed(path: Path, *, max_bytes: int = MAX_RECEIPT_BYTES) -> LoadedReceipt:
    if not path.exists():
        return LoadedReceipt(path, False, False, None, None, None, ["receipt is absent"])
    document, document_sha256, errors = _read_json(path, max_bytes=max_bytes)
    if document is None:
        return LoadedReceipt(path, True, False, None, None, document_sha256, errors)
    try:
        checked = verify(document, label=str(path))
    except SealIntegrityError as exc:
        return LoadedReceipt(path, True, False, None, None, document_sha256, [str(exc)])
    seal_value = checked.get("seal_sha256")
    return LoadedReceipt(
        path,
        True,
        True,
        checked,
        str(seal_value) if _is_sha256(seal_value) else None,
        document_sha256,
        [],
    )


def _load_observation(path: Path) -> dict[str, Any]:
    document, document_sha256, errors = _read_json(path)
    if document is None:
        return {
            "path": str(path),
            "state": "ABSENT_OR_INVALID",
            "sha256": document_sha256,
            "errors": errors,
            "document": None,
        }
    return {
        "path": str(path),
        "state": "OBSERVED_UNSEALED",
        "sha256": document_sha256,
        "errors": [],
        "document": document,
    }


def _paths(root: Path, spec: ModelSpec) -> dict[str, Path]:
    base = root / spec.key
    return {
        "identity": base / "evolution" / "SOURCE_CONTENT_IDENTITY.json",
        "revalidation": base / "complete-gravity" / f"{spec.prefix}_CURRENT_SOURCE_SHARD_REVALIDATION.json",
        "worker": base / "evolution" / f"{spec.prefix}_DUAL_GRAVITY_STATUS.json",
        "pack_status": base / "complete-gravity" / f"{spec.prefix}_COMPLETE_GRAVITY_STATUS.json",
        "runtime_status": base / "complete-runtime" / f"{spec.prefix}_COMPLETE_RUNTIME_STATUS.json",
        "tg3_status": base / "tg3" / f"{spec.prefix}_TG3_ASCENT_STATUS.json",
        "artifact_admission": base / "complete-gravity" / f"{spec.prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT.json",
        "artifact_admission_current": base
        / "complete-gravity"
        / f"{spec.prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_CURRENT.json",
        "runtime": base / "complete-runtime" / f"{spec.prefix}_EXACT_FULL_TOKEN_RUNTIME_RECEIPT.json",
        "runtime_supersession": base
        / "complete-runtime"
        / f"{spec.prefix}_EXACT_FULL_TOKEN_RUNTIME_SUPERSESSION.json",
        "hcli": base / "complete-runtime" / f"{spec.prefix}_MEASURED_HCLI_RECEIPT.json",
        "kernel": root / "kernel" / f"{spec.prefix}_CUSTOM_KERNEL_OPERATIONAL_RECEIPT.json",
        "tg10": base / "tg3" / f"{spec.prefix}_TG10_OPERATIONAL_PASS.json",
        "tg3": base / "tg3" / f"{spec.prefix}_TG3_QUALIFICATION_RECEIPT.json",
        "capability": base / "evaluation" / f"{spec.prefix}_CAPABILITY_EVALUATION_RECEIPT.json",
        "manager_operations": base
        / "agent-os"
        / f"{spec.prefix}_FINAL_MANAGER_OPERATIONS_RECEIPT.json",
        "complete_root": base / "complete-gravity",
    }


def _simple_check(
    requirement: str,
    loaded: LoadedReceipt,
    *,
    details: Mapping[str, Any] | None = None,
    reasons: Sequence[str] = (),
) -> Check:
    all_reasons = list(dict.fromkeys(list(loaded.errors) + list(reasons)))
    return Check(
        requirement=requirement,
        passed=not all_reasons,
        path=loaded.path,
        seal_sha256=loaded.seal_sha256,
        reasons=all_reasons,
        details=dict(details or {}),
        document=loaded.document,
        document_sha256=loaded.document_sha256,
    )


def _validate_source_identity(spec: ModelSpec, path: Path) -> tuple[Check, SourceBinding | None]:
    loaded = _load_sealed(path)
    reasons: list[str] = []
    details: dict[str, Any] = {}
    document = loaded.document or {}
    if loaded.sealed:
        if document.get("schema") != SOURCE_IDENTITY_SCHEMA:
            reasons.append("unexpected source identity schema")
        if document.get("status") != "IMMUTABLE_SOURCE_CONTENT_IDENTITY_BOUND":
            reasons.append("source identity is not immutable-bound")
        model = _mapping(document.get("model"))
        content = _mapping(document.get("source_content"))
        for name, observed, expected in (
            ("model.id", model.get("id"), spec.model_id),
            ("model.architecture", model.get("architecture"), spec.architecture),
            ("model.repository", model.get("repository"), spec.repository),
            ("model.revision", model.get("revision"), spec.revision),
            ("source_content.architecture", content.get("architecture"), spec.architecture),
            ("source_content.repository", content.get("repository"), spec.repository),
            ("source_content.revision", content.get("revision"), spec.revision),
        ):
            if observed != expected:
                reasons.append(f"{name} does not bind {expected}")
        content_id = document.get("content_identity_sha256")
        if not _is_sha256(content_id):
            reasons.append("content_identity_sha256 is not a SHA-256")
        source_dir_value = model.get("source_dir")
        source_dir = Path(str(source_dir_value)).expanduser() if isinstance(source_dir_value, str) else None
        if source_dir is None or not source_dir.is_dir():
            reasons.append("model.source_dir is absent or unavailable")
        weights = content.get("verified_weight_shards")
        controls = content.get("control_files")
        if not isinstance(weights, list) or len(weights) != spec.shard_count:
            reasons.append(f"expected exactly {spec.shard_count} verified source shards")
        if not isinstance(controls, list) or not controls:
            reasons.append("source identity has no verified control files")
        normalized_weights: dict[str, dict[str, Any]] = {}
        normalized_controls: dict[str, dict[str, Any]] = {}
        if isinstance(weights, list):
            for row in weights:
                entry = _mapping(row)
                name = entry.get("path")
                if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
                    reasons.append("source identity contains an unsafe shard path")
                    continue
                if name in normalized_weights:
                    reasons.append(f"duplicate source shard identity: {name}")
                    continue
                if not _is_positive_int(entry.get("bytes")) or not _is_sha256(entry.get("sha256")):
                    reasons.append(f"source shard has invalid bytes or SHA-256: {name}")
                    continue
                normalized_weights[name] = entry
        if isinstance(controls, list):
            for row in controls:
                entry = _mapping(row)
                name = entry.get("path")
                if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
                    reasons.append("source identity contains an unsafe control-file path")
                    continue
                if name in normalized_controls:
                    reasons.append(f"duplicate source control-file identity: {name}")
                    continue
                if not _is_positive_int(entry.get("bytes")) or not _is_sha256(entry.get("sha256")):
                    reasons.append(f"source control file has invalid bytes or SHA-256: {name}")
                    continue
                normalized_controls[name] = entry
        if len(normalized_weights) != spec.shard_count:
            reasons.append(f"source identity does not contain {spec.shard_count} unique valid shards")
        if "model.safetensors.index.json" not in normalized_controls:
            reasons.append("source identity omits model.safetensors.index.json")
        if source_dir is not None and source_dir.is_dir():
            for name, entry in sorted(normalized_controls.items()):
                candidate = source_dir / name
                if not _under(source_dir, candidate):
                    reasons.append(f"control file escapes source directory: {name}")
                    continue
                try:
                    identity = _regular_file_identity(candidate, label=f"source control file {name}")
                    if identity["bytes"] != int(entry["bytes"]):
                        reasons.append(f"control file byte count changed: {name}")
                    elif _sha256_file(candidate) != entry["sha256"]:
                        reasons.append(f"control file SHA-256 changed: {name}")
                except PhysicalGatekeeperError as exc:
                    reasons.append(str(exc))
        details = {
            "model_id": spec.model_id,
            "repository": spec.repository,
            "revision": spec.revision,
            "content_identity_sha256": content_id if _is_sha256(content_id) else None,
            "verified_weight_shard_count": len(normalized_weights),
            "verified_control_file_count": len(normalized_controls),
            "current_control_files_rehashed": bool(normalized_controls),
        }
    check = _simple_check("verified_raw_source_identity", loaded, details=details, reasons=reasons)
    if not check.passed or loaded.document is None or loaded.seal_sha256 is None:
        return check, None
    model = _mapping(loaded.document.get("model"))
    return check, SourceBinding(
        content_identity_sha256=str(loaded.document["content_identity_sha256"]),
        identity_seal_sha256=loaded.seal_sha256,
        revalidation_seal_sha256="",
        source_dir=Path(str(model["source_dir"])).expanduser().resolve(),
        weight_shard_count=details["verified_weight_shard_count"],
        control_file_count=details["verified_control_file_count"],
    )


def _validate_revalidation(
    spec: ModelSpec, path: Path, identity: Check, source: SourceBinding | None
) -> tuple[Check, SourceBinding | None]:
    loaded = _load_sealed(path)
    reasons: list[str] = []
    details: dict[str, Any] = {}
    document = loaded.document or {}
    if source is None or identity.document is None:
        reasons.append("immutable source identity has not passed")
    if loaded.sealed:
        if document.get("schema") != SOURCE_REVALIDATION_SCHEMA:
            reasons.append("unexpected source revalidation schema")
        if document.get("status") != "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED":
            reasons.append("source shards are not currently revalidated")
        if document.get("source_repository") != spec.repository:
            reasons.append("revalidation repository does not match immutable source")
        if document.get("source_revision") != spec.revision:
            reasons.append("revalidation revision does not match immutable source")
        if source is not None:
            model_dir = document.get("source_model_dir")
            if not isinstance(model_dir, str) or Path(model_dir).expanduser().resolve() != source.source_dir:
                reasons.append("revalidation source directory does not match immutable source")
        shards = document.get("shards")
        identity_content = _mapping((identity.document or {}).get("source_content"))
        identity_rows = identity_content.get("verified_weight_shards")
        expected: dict[str, dict[str, Any]] = {}
        if isinstance(identity_rows, list):
            for row in identity_rows:
                item = _mapping(row)
                if isinstance(item.get("path"), str):
                    expected[str(item["path"])] = item
        if not isinstance(shards, Mapping):
            reasons.append("revalidation receipt has no shard map")
            observed_shards: dict[str, Any] = {}
        else:
            observed_shards = {str(name): value for name, value in shards.items()}
        if set(observed_shards) != set(expected):
            reasons.append("revalidation shard set differs from immutable source identity")
        verified_shards = 0
        if source is not None:
            for name, expected_row in sorted(expected.items()):
                observed_row = _mapping(observed_shards.get(name))
                expected_hash = expected_row.get("sha256")
                expected_bytes = expected_row.get("bytes")
                if observed_row.get("expected_sha256") != expected_hash:
                    reasons.append(f"revalidation expected SHA-256 mismatch: {name}")
                    continue
                if observed_row.get("observed_sha256") != expected_hash:
                    reasons.append(f"revalidation observed SHA-256 mismatch: {name}")
                    continue
                if observed_row.get("expected_bytes") != expected_bytes:
                    reasons.append(f"revalidation byte count mismatch: {name}")
                    continue
                identity_row = _mapping(observed_row.get("file_identity"))
                candidate = source.source_dir / name
                try:
                    current = _regular_file_identity(candidate, label=f"source shard {name}")
                except PhysicalGatekeeperError as exc:
                    reasons.append(str(exc))
                    continue
                if any(identity_row.get(field) != current[field] for field in current):
                    reasons.append(f"source shard file identity changed after revalidation: {name}")
                    continue
                verified_shards += 1
        index_control = next(
            (
                _mapping(row)
                for row in (identity_content.get("control_files") or [])
                if isinstance(row, Mapping) and row.get("path") == "model.safetensors.index.json"
            ),
            {},
        )
        index_sha = index_control.get("sha256")
        if not _is_sha256(index_sha):
            reasons.append("immutable identity does not contain an index SHA-256")
        else:
            if document.get("index_sha256") != index_sha:
                reasons.append("revalidation index SHA-256 differs from immutable source")
            sealed_index = document.get("sealed_audit_index_sha256")
            if sealed_index is not None and sealed_index != index_sha:
                reasons.append("revalidation sealed index SHA-256 differs from immutable source")
        if document.get("sealed_shard_count") != spec.shard_count:
            reasons.append("revalidation sealed shard count is incorrect")
        details = {
            "model_id": spec.model_id,
            "revalidated_shard_count": verified_shards,
            "expected_shard_count": spec.shard_count,
            "current_file_identity_checked": bool(source is not None),
            "weight_body_audit_seal_is_not_used_as_a_mutable_identity": True,
        }
    check = _simple_check("current_source_revalidation", loaded, details=details, reasons=reasons)
    if not check.passed or source is None or loaded.seal_sha256 is None:
        return check, None
    return check, SourceBinding(
        content_identity_sha256=source.content_identity_sha256,
        identity_seal_sha256=source.identity_seal_sha256,
        revalidation_seal_sha256=loaded.seal_sha256,
        source_dir=source.source_dir,
        weight_shard_count=source.weight_shard_count,
        control_file_count=source.control_file_count,
    )


def _receipt_header(
    requirement: str,
    path: Path,
    *,
    schema: str,
    status: str,
    spec: ModelSpec,
    source: SourceBinding | None,
    extra_bindings: Mapping[str, str | None] = (),
) -> tuple[LoadedReceipt, list[str], dict[str, Any]]:
    loaded = _load_sealed(path)
    reasons = list(loaded.errors)
    document = loaded.document or {}
    binding = _mapping(document.get("binding"))
    if loaded.sealed:
        if document.get("schema") != schema:
            reasons.append(f"unexpected schema; expected {schema}")
        if document.get("status") != status:
            reasons.append(f"unexpected status; expected {status}")
        if binding.get("model_id") != spec.model_id:
            reasons.append("receipt model binding does not match contender")
        if source is None:
            reasons.append("current source identity/revalidation has not passed")
        else:
            if binding.get("source_content_identity_sha256") != source.content_identity_sha256:
                reasons.append("receipt source content identity binding does not match")
            if binding.get("source_revalidation_seal_sha256") != source.revalidation_seal_sha256:
                reasons.append("receipt source revalidation seal binding does not match")
        for field, expected in dict(extra_bindings).items():
            if not _is_sha256(expected):
                reasons.append(f"upstream gate evidence is unavailable for {field}")
            elif binding.get(field) != expected:
                reasons.append(f"receipt binding does not match {field}")
    details = {
        "model_id": spec.model_id,
        "expected_schema": schema,
        "expected_status": status,
        "source_bound": source is not None,
    }
    return loaded, reasons, details


def _require_true(container: Mapping[str, Any], field: str, reasons: list[str], label: str) -> None:
    if container.get(field) is not True:
        reasons.append(f"{label}.{field} must be true")


def _require_positive(container: Mapping[str, Any], field: str, reasons: list[str], label: str) -> None:
    if not _is_positive_int(container.get(field)):
        reasons.append(f"{label}.{field} must be a positive integer")


def _require_sha(container: Mapping[str, Any], field: str, reasons: list[str], label: str) -> None:
    if not _is_sha256(container.get(field)):
        reasons.append(f"{label}.{field} must be a SHA-256")


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _manifest_schema(spec: ModelSpec) -> str:
    return f"hawking.ascension.{spec.key}_complete_binary_gravity.v1"


def _same_path(
    value: Any, expected: Path, reasons: list[str], label: str
) -> bool:
    """Require an absolute, resolved path to bind one exact local artifact."""

    if not isinstance(value, str) or not value:
        reasons.append(f"{label} must be an absolute path")
        return False
    observed = Path(value).expanduser()
    if not observed.is_absolute():
        reasons.append(f"{label} must be an absolute path")
        return False
    try:
        if observed.resolve() != expected.expanduser().resolve():
            reasons.append(f"{label} does not bind the expected path")
            return False
    except OSError as exc:
        reasons.append(f"cannot resolve {label}: {exc}")
        return False
    return True


def runtime_receipt_supersession_state(
    spec: ModelSpec,
    *,
    runtime_path: Path,
    supersession_path: Path,
    runtime_loaded: LoadedReceipt | None = None,
) -> dict[str, Any]:
    """Resolve the current authority of a canonical runtime receipt.

    A runtime pass is not irrevocable merely because its old receipt remains
    sealed.  A producer may discover an architecture or execution defect after
    publication.  It then preserves the old PASS receipt in the immutable
    ``runtime-receipt-history`` directory, replaces the canonical filename
    with a non-PASS record, and writes this sealed supersession sidecar.

    Every consumer uses this resolver instead of treating a familiar filename
    or an archived historical pass as authority.  A malformed, unreadable, or
    incomplete sidecar is itself fail-closed: silently ignoring a purported
    revocation would recreate the exact stale-evidence failure this contract
    exists to prevent.
    """

    runtime = runtime_loaded or _load_sealed(runtime_path)
    current = runtime.document or {}
    current_binding = _mapping(current.get("binding"))
    current_revoked = _mapping(current.get("revoked_runtime"))
    current_executable = current_binding.get("runtime_executable_sha256")
    if not _is_sha256(current_executable):
        current_executable = current_revoked.get("runtime_executable_sha256")
    current_is_pass = bool(
        runtime.sealed
        and current.get("schema") == RUNTIME_SCHEMA
        and current.get("status") == RUNTIME_PASS_STATUS
    )
    result: dict[str, Any] = {
        "schema": "hawking.ascension.physical_exact_full_token_runtime_authority.v1",
        "model_id": spec.model_id,
        "canonical_runtime_receipt_path": str(runtime_path),
        "canonical_runtime_receipt_seal_sha256": runtime.seal_sha256,
        "canonical_runtime_status": current.get("status"),
        "canonical_runtime_is_pass": current_is_pass,
        "canonical_runtime_executable_sha256": current_executable
        if _is_sha256(current_executable)
        else None,
        "supersession_path": str(supersession_path),
        "supersession_present": supersession_path.exists(),
        "supersession_seal_sha256": None,
        "superseded_runtime_receipt_seal_sha256": None,
        "defective_runtime_executable_sha256": None,
        "historical_pass_archive_path": None,
        "current_runtime_eligible": False,
        "state": "CANONICAL_RUNTIME_NOT_PASS",
        "reasons": [],
    }
    if not supersession_path.exists():
        result["current_runtime_eligible"] = current_is_pass
        result["state"] = (
            "CURRENT_CANONICAL_RUNTIME_PASS_NO_SUPERSESSION"
            if current_is_pass
            else "CANONICAL_RUNTIME_NOT_PASS"
        )
        return result

    supersession = _load_sealed(supersession_path)
    reasons = list(supersession.errors)
    document = supersession.document or {}
    result["supersession_seal_sha256"] = supersession.seal_sha256
    if supersession.sealed:
        expected_top_level = {
            "schema",
            "status",
            "recorded_at",
            "binding",
            "revoked_runtime",
            "historical_pass_archive_path",
            "historical_pass_archive_sha256",
            "defect",
            "invalidates",
            "required_before_reissue",
            "consumer_contract",
            "claim_boundary",
            "seal_sha256",
        }
        observed_top_level = set(document)
        if observed_top_level != expected_top_level:
            missing = sorted(expected_top_level - observed_top_level)
            unexpected = sorted(observed_top_level - expected_top_level)
            if missing:
                reasons.append(f"runtime supersession is missing fields: {', '.join(missing)}")
            if unexpected:
                reasons.append(f"runtime supersession has unexpected fields: {', '.join(unexpected)}")
        if document.get("schema") != RUNTIME_SUPERSESSION_SCHEMA:
            reasons.append("runtime supersession schema is not accepted")
        status = document.get("status")
        if not isinstance(status, str) or not status.startswith(RUNTIME_REVOCATION_STATUS_PREFIX):
            reasons.append("runtime supersession status is not an explicit revocation")
        if not isinstance(document.get("recorded_at"), str) or not document.get("recorded_at"):
            reasons.append("runtime supersession recorded_at is absent")

        binding = _mapping(document.get("binding"))
        expected_binding = {
            "model_id",
            "canonical_runtime_receipt_path",
            "superseded_runtime_receipt_seal_sha256",
            "defective_runtime_executable_sha256",
            "archived_runtime_receipt_path",
            "archived_runtime_receipt_document_sha256",
        }
        if set(binding) != expected_binding:
            reasons.append("runtime supersession binding fields are not the v1 exact contract")
        if binding.get("model_id") != spec.model_id:
            reasons.append("runtime supersession model binding does not match contender")
        _same_path(
            binding.get("canonical_runtime_receipt_path"),
            runtime_path,
            reasons,
            "runtime supersession canonical_runtime_receipt_path",
        )
        superseded_seal = binding.get("superseded_runtime_receipt_seal_sha256")
        defective_executable = binding.get("defective_runtime_executable_sha256")
        archive_value = binding.get("archived_runtime_receipt_path")
        archive_document_sha = binding.get("archived_runtime_receipt_document_sha256")
        for field, value in (
            ("superseded_runtime_receipt_seal_sha256", superseded_seal),
            ("defective_runtime_executable_sha256", defective_executable),
            ("archived_runtime_receipt_document_sha256", archive_document_sha),
        ):
            if not _is_sha256(value):
                reasons.append(f"runtime supersession {field} must be a SHA-256")
        result["superseded_runtime_receipt_seal_sha256"] = (
            superseded_seal if _is_sha256(superseded_seal) else None
        )
        result["defective_runtime_executable_sha256"] = (
            defective_executable if _is_sha256(defective_executable) else None
        )

        history_root = runtime_path.parent / "runtime-receipt-history"
        archive_path: Path | None = None
        if not isinstance(archive_value, str) or not archive_value:
            reasons.append("runtime supersession archived_runtime_receipt_path is absent")
        else:
            archive_path = Path(archive_value).expanduser()
            if not archive_path.is_absolute() or not _under(history_root, archive_path):
                reasons.append("runtime supersession archive path escapes runtime-receipt-history")
            elif document.get("historical_pass_archive_path") != str(archive_path):
                reasons.append("runtime supersession historical archive path does not match binding")
            result["historical_pass_archive_path"] = str(archive_path)
        if document.get("historical_pass_archive_sha256") != archive_document_sha:
            reasons.append("runtime supersession historical archive SHA-256 does not match binding")

        revoked_runtime = _mapping(document.get("revoked_runtime"))
        expected_revoked = {
            "canonical_receipt_path",
            "canonical_receipt_seal_sha256",
            "complete_manifest_seal_sha256",
            "model_id",
            "runtime_executable_sha256",
        }
        if set(revoked_runtime) != expected_revoked:
            reasons.append("runtime supersession revoked_runtime fields are not the v1 exact contract")
        if revoked_runtime.get("model_id") != spec.model_id:
            reasons.append("runtime supersession revoked_runtime model does not match contender")
        _same_path(
            revoked_runtime.get("canonical_receipt_path"),
            runtime_path,
            reasons,
            "runtime supersession revoked_runtime.canonical_receipt_path",
        )
        if revoked_runtime.get("canonical_receipt_seal_sha256") != superseded_seal:
            reasons.append("runtime supersession revoked_runtime seal does not match binding")
        if revoked_runtime.get("runtime_executable_sha256") != defective_executable:
            reasons.append("runtime supersession revoked_runtime executable does not match binding")
        if not _is_sha256(revoked_runtime.get("complete_manifest_seal_sha256")):
            reasons.append("runtime supersession revoked_runtime complete manifest seal is invalid")

        invalidates = _mapping(document.get("invalidates"))
        required_invalidations = {
            "canonical_native_runtime_pass",
            "all_old_full_token_prompt_and_profile_controls_bound_to_runtime_sha",
            "native_http_adapter_and_transport_handoff_bound_to_runtime_sha",
            "any_hcli_tps_tg_capability_or_tournament_consumer_of_that_sha",
        }
        if set(invalidates) != required_invalidations or not all(
            invalidates.get(field) is True for field in required_invalidations
        ):
            reasons.append("runtime supersession invalidation scope is incomplete")
        consumer_contract = _mapping(document.get("consumer_contract"))
        if not all(
            consumer_contract.get(field) is True
            for field in (
                "fail_closed_if_canonical_status_is_not_pass",
                "fail_closed_if_this_supersession_revokes_the_bound_receipt_seal_or_runtime_executable_sha256",
                "historical_archive_is_for_negative_science_only_not_a_gate_authority",
            )
        ):
            reasons.append("runtime supersession consumer fail-closed contract is incomplete")
        if not isinstance(document.get("required_before_reissue"), list) or not document.get(
            "required_before_reissue"
        ) or not all(isinstance(item, str) and item for item in document.get("required_before_reissue", [])):
            reasons.append("runtime supersession required_before_reissue is invalid")
        if not _mapping(document.get("defect")):
            reasons.append("runtime supersession defect record is absent")
        if not _mapping(document.get("claim_boundary")):
            reasons.append("runtime supersession claim boundary is absent")

        if archive_path is not None and _under(history_root, archive_path):
            archive = _load_sealed(archive_path)
            if not archive.sealed:
                reasons.extend(f"runtime supersession historical archive: {reason}" for reason in archive.errors)
            else:
                archived = archive.document or {}
                if archive.seal_sha256 != superseded_seal:
                    reasons.append("runtime supersession historical archive seal does not match revoked PASS")
                if archive.document_sha256 != archive_document_sha:
                    reasons.append("runtime supersession historical archive document SHA-256 does not match binding")
                if archived.get("schema") != RUNTIME_SCHEMA:
                    reasons.append("runtime supersession historical archive is not a runtime receipt")
                if archived.get("status") != RUNTIME_PASS_STATUS:
                    reasons.append("runtime supersession historical archive is not the original PASS receipt")
                archived_binding = _mapping(archived.get("binding"))
                if archived_binding.get("model_id") != spec.model_id:
                    reasons.append("runtime supersession historical archive model binding does not match contender")
                if archived_binding.get("runtime_executable_sha256") != defective_executable:
                    reasons.append("runtime supersession historical archive executable does not match defect")

    if reasons:
        result["state"] = "SUPERSESSION_INVALID_FAIL_CLOSED"
        result["reasons"] = list(dict.fromkeys(reasons))
        return result

    # A new corrected receipt is allowed only if it has a new immutable seal
    # *and* a different executable digest.  This retains the historical defect
    # as negative science without permanently poisoning a correctly rebuilt
    # runtime.
    superseded_seal = result["superseded_runtime_receipt_seal_sha256"]
    defective_executable = result["defective_runtime_executable_sha256"]
    revokes_current = bool(
        runtime.seal_sha256 == superseded_seal
        or current_executable == defective_executable
        or current_revoked.get("canonical_receipt_seal_sha256") == superseded_seal
        or current_revoked.get("runtime_executable_sha256") == defective_executable
    )
    result["current_runtime_revoked"] = revokes_current
    result["current_runtime_eligible"] = bool(current_is_pass and not revokes_current)
    result["state"] = (
        "CURRENT_RUNTIME_REVOKED"
        if revokes_current
        else "SUPERSESSION_PRESERVES_HISTORICAL_REVOCATION_CURRENT_RUNTIME_ELIGIBLE"
        if current_is_pass
        else "CANONICAL_RUNTIME_NOT_PASS_AFTER_SUPERSESSION"
    )
    return result


def _under_complete_root(
    value: Any, root: Path, reasons: list[str], label: str
) -> bool:
    if not isinstance(value, str) or not value:
        reasons.append(f"{label} must be an absolute path")
        return False
    candidate = Path(value).expanduser()
    if not candidate.is_absolute() or not _under(root, candidate):
        reasons.append(f"{label} escapes the complete-gravity root")
        return False
    return True


def _require_bound_sha(
    container: Mapping[str, Any], field: str, expected: str | None, reasons: list[str], label: str
) -> None:
    value = container.get(field)
    if not _is_sha256(value):
        reasons.append(f"{label}.{field} must be a SHA-256")
    elif not _is_sha256(expected):
        reasons.append(f"current evidence is unavailable for {label}.{field}")
    elif value != expected:
        reasons.append(f"{label}.{field} does not match current evidence")


def _require_bound_value(
    container: Mapping[str, Any], field: str, expected: Any, reasons: list[str], label: str
) -> None:
    if container.get(field) != expected:
        reasons.append(f"{label}.{field} does not match current evidence")


def _identity_index_sha(identity: Check) -> str | None:
    content = _mapping(_mapping(identity.document).get("source_content"))
    controls = content.get("control_files")
    if not isinstance(controls, list):
        return None
    matches = [
        _mapping(row).get("sha256")
        for row in controls
        if _mapping(row).get("path") == "model.safetensors.index.json"
    ]
    return str(matches[0]) if len(matches) == 1 and _is_sha256(matches[0]) else None


def _identity_shards(identity: Check) -> dict[str, str]:
    content = _mapping(_mapping(identity.document).get("source_content"))
    result: dict[str, str] = {}
    for row in content.get("verified_weight_shards") or []:
        item = _mapping(row)
        name = item.get("path")
        digest = item.get("sha256")
        if isinstance(name, str) and _is_sha256(digest):
            result[name] = str(digest)
    return result


def _validate_receipt_identity_binding(
    received: Mapping[str, Any],
    *,
    spec: ModelSpec,
    paths: Mapping[str, Path],
    source: SourceBinding,
    identity: Check,
    reasons: list[str],
) -> None:
    expected_index = _identity_index_sha(identity)
    identity_document = _mapping(identity.document)
    historical_audit = identity_document.get("weight_body_audit_seal_sha256")
    _same_path(received.get("path"), paths["identity"], reasons, "immutable_source_identity.path")
    _require_bound_sha(
        received, "document_sha256", identity.document_sha256, reasons, "immutable_source_identity"
    )
    _require_bound_sha(
        received, "seal_sha256", source.identity_seal_sha256, reasons, "immutable_source_identity"
    )
    _require_bound_sha(
        received,
        "content_identity_sha256",
        source.content_identity_sha256,
        reasons,
        "immutable_source_identity",
    )
    _require_bound_value(received, "repository", spec.repository, reasons, "immutable_source_identity")
    _require_bound_value(received, "revision", spec.revision, reasons, "immutable_source_identity")
    _same_path(
        received.get("source_dir"), source.source_dir, reasons, "immutable_source_identity.source_dir"
    )
    _require_bound_sha(received, "index_sha256", expected_index, reasons, "immutable_source_identity")
    _require_bound_sha(
        received,
        "historical_weight_body_audit_seal_sha256",
        historical_audit if _is_sha256(historical_audit) else None,
        reasons,
        "immutable_source_identity",
    )


def _validate_receipt_revalidation_binding(
    received: Mapping[str, Any],
    *,
    spec: ModelSpec,
    paths: Mapping[str, Path],
    source: SourceBinding,
    identity: Check,
    revalidation: Check,
    reasons: list[str],
) -> None:
    current = _mapping(revalidation.document)
    expected_index = _identity_index_sha(identity)
    _same_path(received.get("path"), paths["revalidation"], reasons, "current_source_revalidation.path")
    _require_bound_sha(
        received,
        "document_sha256",
        revalidation.document_sha256,
        reasons,
        "current_source_revalidation",
    )
    _require_bound_sha(
        received,
        "seal_sha256",
        source.revalidation_seal_sha256,
        reasons,
        "current_source_revalidation",
    )
    _require_bound_value(received, "repository", spec.repository, reasons, "current_source_revalidation")
    _require_bound_value(received, "revision", spec.revision, reasons, "current_source_revalidation")
    _same_path(
        received.get("source_model_dir"),
        source.source_dir,
        reasons,
        "current_source_revalidation.source_model_dir",
    )
    _same_path(
        received.get("index_path"),
        source.source_dir / "model.safetensors.index.json",
        reasons,
        "current_source_revalidation.index_path",
    )
    _require_bound_sha(
        received, "index_sha256", expected_index, reasons, "current_source_revalidation"
    )
    _require_bound_value(
        received, "sealed_shard_count", spec.shard_count, reasons, "current_source_revalidation"
    )
    for field in (
        "source_audit_document_sha256",
        "source_audit_seal_sha256",
        "sealed_shard_hashes_sha256",
        "weight_map_sha256",
    ):
        _require_bound_sha(
            received,
            field,
            current.get(field) if _is_sha256(current.get(field)) else None,
            reasons,
            "current_source_revalidation",
        )
    source_audit = current.get("source_audit_path")
    if isinstance(source_audit, str) and source_audit:
        _same_path(
            received.get("source_audit_path"),
            Path(source_audit).expanduser(),
            reasons,
            "current_source_revalidation.source_audit_path",
        )
    else:
        reasons.append("current source revalidation has no source_audit_path")


def _validate_admission_request(
    receipt: Mapping[str, Any],
    *,
    spec: ModelSpec,
    paths: Mapping[str, Path],
    immutable_identity: Mapping[str, Any],
    current_revalidation: Mapping[str, Any],
    complete_manifest: Mapping[str, Any],
    reasons: list[str],
) -> None:
    manifest_seal = complete_manifest.get("seal_sha256")
    if not _is_sha256(manifest_seal):
        reasons.append("complete_manifest.seal_sha256 must be a SHA-256 before request verification")
        return
    request_path = (
        paths["complete_root"]
        / "complete-admission"
        / "requests"
        / f"{spec.prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_REQUEST_{manifest_seal}.json"
    )
    _same_path(receipt.get("admission_request_path"), request_path, reasons, "admission_request_path")
    request_seal = receipt.get("admission_request_seal_sha256")
    if not _is_sha256(request_seal):
        reasons.append("admission_request_seal_sha256 must be a SHA-256")
        return
    request = _load_sealed(request_path)
    if not request.sealed:
        reasons.extend(f"admission request: {reason}" for reason in request.errors)
        return
    if request.seal_sha256 != request_seal:
        reasons.append("admission request seal does not match public admission receipt")
    document = request.document or {}
    if document.get("schema") != ARTIFACT_ADMISSION_REQUEST_SCHEMA:
        reasons.append("admission request schema is not accepted")
    if document.get("status") != ARTIFACT_ADMISSION_REQUEST_STATUS:
        reasons.append("admission request status is not accepted")
    if document.get("request_version") != 1:
        reasons.append("admission request version is not accepted")
    model = _mapping(document.get("model"))
    for field, expected in (
        ("key", spec.key),
        ("id", spec.model_id),
        ("repository", spec.repository),
        ("revision", spec.revision),
        ("native_core_model", spec.key),
    ):
        if model.get(field) != expected:
            reasons.append(f"admission request model.{field} does not bind the contender")
    for field, receipt_binding in (
        ("immutable_source_identity", immutable_identity),
        ("current_source_revalidation", current_revalidation),
        ("complete_manifest", complete_manifest),
    ):
        if _mapping(document.get(field)) != dict(receipt_binding):
            reasons.append(f"admission request {field} does not exactly match public receipt")
    native = _mapping(document.get("native_admission"))
    if native.get("required_api") != NATIVE_COMPLETE_BINARY_API:
        reasons.append("admission request does not require the strict native complete-binary API")
    _require_true(
        native,
        "rechecks_complete_catalog_payload_hash_layout_and_current_source_identity",
        reasons,
        "admission request native_admission",
    )
    boundary = _mapping(document.get("claim_boundary"))
    for field in (
        "manifest_is_bound_by_exact_seal_and_raw_document_sha256",
        "source_content_identity_and_current_full_shard_revalidation_are_both_required",
        "raw_bf16_source_is_authority_teacher_not_tournament_participant",
        "not_native_decoder_runtime_capability_hcli_tps_tg_or_tournament_qualification",
    ):
        _require_true(boundary, field, reasons, "admission request claim_boundary")


def _validate_manifest_catalog_and_ledger(
    manifest: Mapping[str, Any],
    *,
    spec: ModelSpec,
    paths: Mapping[str, Path],
    source: SourceBinding,
    identity: Check,
    current_revalidation: Mapping[str, Any],
    native: Mapping[str, Any],
    reasons: list[str],
) -> dict[str, Any]:
    """Check the sealed manifest's physical ledger without rehashing 4+ GiB.

    The independent native admission result is what rechecked payload bytes,
    hashes, and layouts.  This observer cross-binds its accepted facts to the
    current sealed manifest rather than pretending to repeat that expensive
    native read on every watchdog cycle.
    """

    details: dict[str, Any] = {}
    if manifest.get("schema") != _manifest_schema(spec):
        reasons.append("complete manifest schema is not accepted for this contender")
    if manifest.get("status") != COMPLETE_BINARY_MANIFEST_STATUS:
        reasons.append("complete manifest is not the accepted complete candidate status")
    if manifest.get("source_body_audit_seal_sha256") != current_revalidation.get(
        "source_audit_seal_sha256"
    ):
        reasons.append("complete manifest source audit seal differs from current revalidation")
    _same_path(
        manifest.get("source_revalidation_receipt_path"),
        paths["revalidation"],
        reasons,
        "complete manifest source_revalidation_receipt_path",
    )
    if manifest.get("source_revalidation_receipt_seal_sha256") != source.revalidation_seal_sha256:
        reasons.append("complete manifest revalidation seal differs from current revalidation")

    source_row = _mapping(manifest.get("source"))
    if source_row.get("repository") != spec.repository:
        reasons.append("complete manifest source repository differs from contender")
    _same_path(
        source_row.get("model_dir"), source.source_dir, reasons, "complete manifest source.model_dir"
    )
    native_count = native.get("tensor_count")
    native_elements = native.get("source_weight_elements")
    native_payload = native.get("tensor_payload_bytes")
    if not _is_positive_int(native_count):
        reasons.append("native admission tensor_count must be a positive integer")
    if not _is_positive_int(native_elements):
        reasons.append("native admission source_weight_elements must be a positive integer")
    if not _is_positive_int(native_payload):
        reasons.append("native admission tensor_payload_bytes must be a positive integer")
    if source_row.get("tensor_count") != native_count:
        reasons.append("complete manifest source tensor count differs from native admission")

    ledger = _mapping(manifest.get("complete_physical_bpw_ledger"))
    bpw = _finite_number(ledger.get("complete_physical_bpw"))
    threshold = _finite_number(ledger.get("threshold_bpw"))
    _require_true(ledger, "passes_storage_threshold", reasons, "complete_physical_bpw_ledger")
    for field in ("all_required_weight_artifact_bytes", "source_weight_elements", "tensor_payload_bytes"):
        _require_positive(ledger, field, reasons, "complete_physical_bpw_ledger")
    if not _is_nonnegative_int(ledger.get("manifest_bytes_billed")):
        reasons.append("complete_physical_bpw_ledger.manifest_bytes_billed must be non-negative")
    if bpw is None or bpw <= 0.0:
        reasons.append("complete_physical_bpw_ledger.complete_physical_bpw must be positive and finite")
    elif bpw > 1.5:
        reasons.append("complete physical BPW exceeds the 1.5 BPW gate")
    if threshold is None or not math.isclose(threshold, 1.5, rel_tol=0.0, abs_tol=0.0):
        reasons.append("complete physical ledger threshold must be exactly 1.5 BPW")
    if ledger.get("source_weight_elements") != native_elements:
        reasons.append("complete physical ledger source elements differ from native admission")
    if ledger.get("tensor_payload_bytes") != native_payload:
        reasons.append("complete physical ledger payload bytes differ from native admission")

    tensors = manifest.get("tensors")
    expected_shards = _identity_shards(identity)
    tensor_metadata_invalid = False
    tensor_source_invalid = False
    tensor_path_invalid = False
    tensor_count = 0
    total_elements = 0
    total_payload_bytes = 0
    names: set[str] = set()
    if not isinstance(tensors, list):
        reasons.append("complete manifest tensors must be an array")
    else:
        tensor_count = len(tensors)
        for row in tensors:
            item = _mapping(row)
            name = item.get("tensor_name")
            elements = item.get("elements")
            artifact_bytes = item.get("artifact_bytes")
            shape = item.get("shape")
            if (
                not isinstance(name, str)
                or not name
                or name in names
                or not _is_positive_int(elements)
                or not _is_positive_int(artifact_bytes)
                or not _is_sha256(item.get("artifact_sha256"))
                or item.get("source_dtype") != "BF16"
            ):
                tensor_metadata_invalid = True
            if isinstance(name, str) and name:
                names.add(name)
            if _is_positive_int(elements):
                total_elements += int(elements)
            if _is_positive_int(artifact_bytes):
                total_payload_bytes += int(artifact_bytes)
            if not isinstance(shape, list) or not shape:
                tensor_metadata_invalid = True
            else:
                shape_elements = 1
                for dimension in shape:
                    if not _is_positive_int(dimension):
                        tensor_metadata_invalid = True
                        shape_elements = 0
                        break
                    shape_elements *= int(dimension)
                if _is_positive_int(elements) and shape_elements != int(elements):
                    tensor_metadata_invalid = True
            layout = _mapping(item.get("layout"))
            if (
                not isinstance(layout.get("magic"), str)
                or not layout.get("magic")
                or not _is_positive_int(layout.get("group_size"))
                or not isinstance(layout.get("scale_dtype"), str)
                or not isinstance(layout.get("sign_bit_order"), str)
                or not _is_positive_int(layout.get("version"))
            ):
                tensor_metadata_invalid = True
            if not _under_complete_root(
                item.get("artifact_path"), paths["complete_root"], reasons=[], label="manifest tensor artifact_path"
            ):
                tensor_path_invalid = True
            shard_name = item.get("source_shard")
            if (
                not isinstance(shard_name, str)
                or expected_shards.get(shard_name) != item.get("source_shard_sha256")
            ):
                tensor_source_invalid = True
        if tensor_count != native_count:
            reasons.append("complete manifest tensor catalog count differs from native admission")
    if tensor_metadata_invalid:
        reasons.append("complete manifest has invalid or incomplete tensor metadata")
    if tensor_source_invalid:
        reasons.append("complete manifest tensor source-shard bindings differ from immutable source")
    if tensor_path_invalid:
        reasons.append("complete manifest tensor artifact path escapes complete-gravity root")
    if total_elements != ledger.get("source_weight_elements"):
        reasons.append("complete manifest tensor elements differ from physical ledger")
    if total_payload_bytes != ledger.get("tensor_payload_bytes"):
        reasons.append("complete manifest tensor payload bytes differ from physical ledger")
    billed = ledger.get("manifest_bytes_billed")
    total_artifact_bytes = ledger.get("all_required_weight_artifact_bytes")
    if (
        _is_nonnegative_int(billed)
        and _is_positive_int(ledger.get("tensor_payload_bytes"))
        and _is_positive_int(total_artifact_bytes)
        and int(total_artifact_bytes) != int(ledger["tensor_payload_bytes"]) + int(billed)
    ):
        reasons.append("complete physical ledger billed bytes do not reconcile")
    if (
        bpw is not None
        and _is_positive_int(ledger.get("source_weight_elements"))
        and _is_positive_int(total_artifact_bytes)
    ):
        calculated_bpw = 8.0 * int(total_artifact_bytes) / int(ledger["source_weight_elements"])
        if not math.isclose(bpw, calculated_bpw, rel_tol=1e-12, abs_tol=1e-12):
            reasons.append("complete physical BPW does not reconcile to the sealed ledger")
    boundary = _mapping(manifest.get("claim_boundary"))
    for field in (
        "complete_physical_tensor_coverage_is_true",
        "complete_bpw_pass_does_not_substitute_for_capability",
        "not_native_runtime_execution",
        "not_tg10_tg3_hcli_agent_os_or_manager_qualified",
        "raw_source_remains_authority_teacher_only",
    ):
        _require_true(boundary, field, reasons, "complete manifest claim_boundary")
    details.update(
        {
            "physical_bpw": bpw,
            "physical_bpw_threshold": threshold,
            "complete_source_tensor_count": tensor_count,
            "source_weight_elements": ledger.get("source_weight_elements"),
            "tensor_payload_bytes": ledger.get("tensor_payload_bytes"),
            "all_required_weight_artifact_bytes": total_artifact_bytes,
        }
    )
    return details


def _failed_current_admission_selection(
    pointer: LoadedReceipt, reasons: Sequence[str]
) -> LoadedReceipt:
    """Return a non-admission receipt when a current pointer is unsafe.

    A present current pointer is an explicit selection attempt.  Falling back
    to the historical fixed receipt after that pointer fails would let an old
    admission authorize a newer manifest, so this helper deliberately returns
    no artifact seal/document.
    """

    return LoadedReceipt(
        path=pointer.path,
        present=pointer.present,
        sealed=False,
        document=None,
        seal_sha256=None,
        document_sha256=pointer.document_sha256,
        errors=list(dict.fromkeys(list(pointer.errors) + list(reasons))),
    )


def _select_current_artifact_admission(
    spec: ModelSpec, paths: Mapping[str, Path]
) -> tuple[LoadedReceipt, list[str], dict[str, Any]]:
    """Resolve a current versioned admission receipt without trusting history.

    Before versioned admissions existed, the immutable fixed public receipt is
    a compatibility fallback.  Once a sealed current pointer is present, it
    is authoritative and must bind an exact manifest-keyed receipt; an invalid
    pointer fails closed rather than falling back to the historical path.
    """

    pointer_path = paths["artifact_admission_current"]
    pointer = _load_sealed(pointer_path)
    if not pointer.present:
        return (
            _load_sealed(paths["artifact_admission"]),
            [],
            {
                "admission_selection": "LEGACY_FIXED_RECEIPT_FALLBACK",
                "historical_receipt_path": str(paths["artifact_admission"]),
                "current_pointer_path": str(pointer_path),
            },
        )
    reasons = list(pointer.errors)
    details: dict[str, Any] = {
        "admission_selection": "CURRENT_POINTER",
        "current_pointer_path": str(pointer_path),
        "current_pointer_seal_sha256": pointer.seal_sha256,
    }
    document = pointer.document or {}
    if not pointer.sealed:
        reasons.append("current admission pointer is not a valid sealed receipt")
        return _failed_current_admission_selection(pointer, reasons), reasons, details
    if (
        document.get("schema") != ARTIFACT_ADMISSION_CURRENT_POINTER_SCHEMA
        or document.get("status") != ARTIFACT_ADMISSION_CURRENT_POINTER_STATUS
        or document.get("pointer_version") != 1
    ):
        reasons.append("current admission pointer schema/status/version is not accepted")
    model = _mapping(document.get("model"))
    for field, expected in (
        ("key", spec.key),
        ("id", spec.model_id),
        ("repository", spec.repository),
        ("revision", spec.revision),
    ):
        if model.get(field) != expected:
            reasons.append(f"current admission pointer model.{field} does not bind the contender")
    manifest = _mapping(document.get("complete_manifest"))
    expected_manifest_path = (
        paths["complete_root"] / f"{spec.prefix}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
    )
    if not _same_path(
        manifest.get("path"), expected_manifest_path, reasons, "current pointer complete_manifest.path"
    ):
        pass
    _require_sha(manifest, "document_sha256", reasons, "current pointer complete_manifest")
    _require_sha(manifest, "seal_sha256", reasons, "current pointer complete_manifest")
    manifest_seal = manifest.get("seal_sha256")
    if not _is_sha256(manifest_seal):
        return _failed_current_admission_selection(pointer, reasons), reasons, details
    expected_request_path = (
        paths["complete_root"]
        / "complete-admission"
        / "requests"
        / f"{spec.prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_REQUEST_{manifest_seal}.json"
    )
    _same_path(
        document.get("admission_request_path"),
        expected_request_path,
        reasons,
        "current admission pointer admission_request_path",
    )
    _require_sha(
        document,
        "admission_request_seal_sha256",
        reasons,
        "current admission pointer",
    )
    receipt_binding = _mapping(document.get("admission_receipt"))
    receipt_path_value = receipt_binding.get("path")
    expected_versioned_receipt = (
        paths["complete_root"]
        / "complete-admission"
        / "receipts"
        / f"{spec.prefix}_COMPLETE_BINARY_GRAVITY_ADMISSION_RECEIPT_{manifest_seal}.json"
    )
    receipt_path_matches_versioned = _same_path(
        receipt_path_value,
        expected_versioned_receipt,
        reasons,
        "current admission pointer versioned receipt path",
    )
    # A legacy pointer is permitted only for first-generation intact campaigns
    # and remains subject to the same manifest/request/receipt equality below.
    receipt_path_matches_legacy = False
    if not receipt_path_matches_versioned:
        # `_same_path` has already recorded a reason for the versioned mismatch;
        # remove that compatibility-only diagnostic if the fixed historical
        # path is exactly the selected current receipt.
        expected_reason = "current admission pointer versioned receipt path does not bind the expected path"
        if expected_reason in reasons:
            reasons.remove(expected_reason)
        receipt_path_matches_legacy = _same_path(
            receipt_path_value,
            paths["artifact_admission"],
            reasons,
            "current admission pointer historical receipt path",
        )
    if not (receipt_path_matches_versioned or receipt_path_matches_legacy):
        return _failed_current_admission_selection(pointer, reasons), reasons, details
    _require_sha(receipt_binding, "document_sha256", reasons, "current admission pointer receipt")
    _require_sha(receipt_binding, "seal_sha256", reasons, "current admission pointer receipt")
    if reasons:
        return _failed_current_admission_selection(pointer, reasons), reasons, details
    selected_path = expected_versioned_receipt if receipt_path_matches_versioned else paths["artifact_admission"]
    selected = _load_sealed(selected_path)
    reasons.extend(selected.errors)
    if not selected.sealed:
        return _failed_current_admission_selection(pointer, reasons), reasons, details
    selected_document = selected.document or {}
    if selected.document_sha256 != receipt_binding.get("document_sha256"):
        reasons.append("current admission pointer receipt document SHA-256 does not match selected receipt")
    if selected.seal_sha256 != receipt_binding.get("seal_sha256"):
        reasons.append("current admission pointer receipt seal does not match selected receipt")
    if selected_document.get("admission_request_path") != document.get("admission_request_path"):
        reasons.append("current admission pointer request path does not match selected receipt")
    if selected_document.get("admission_request_seal_sha256") != document.get(
        "admission_request_seal_sha256"
    ):
        reasons.append("current admission pointer request seal does not match selected receipt")
    if _mapping(selected_document.get("complete_manifest")) != manifest:
        reasons.append("current admission pointer complete manifest does not match selected receipt")
    boundary = _mapping(document.get("claim_boundary"))
    for field in (
        "pointer_selects_only_a_receipt_matching_the_current_complete_manifest",
        "historical_receipts_are_preserved_not_overwritten_or_resealed",
        "pointer_is_storage_artifact_admission_only_not_runtime_or_qualification",
    ):
        _require_true(boundary, field, reasons, "current admission pointer claim_boundary")
    if reasons:
        return _failed_current_admission_selection(pointer, reasons), reasons, details
    details.update(
        {
            "selected_admission_receipt_path": str(selected_path),
            "selected_admission_receipt_seal_sha256": selected.seal_sha256,
            "selected_manifest_seal_sha256": manifest_seal,
        }
    )
    return selected, [], details


def _validate_artifact(
    spec: ModelSpec,
    paths: Mapping[str, Path],
    source: SourceBinding | None,
    identity: Check,
    revalidation: Check,
) -> Check:
    """Admit only the real native complete-binary storage receipt.

    Passing this sub-gate means the sealed complete physical artifact is at or
    below 1.5 BPW and the native reader admitted its catalog/payload/layout
    facts.  It deliberately says nothing about runtime, manager readiness,
    HCLI, TPS, TG, capability, or tournament eligibility.
    """

    loaded, selection_reasons, selection_details = _select_current_artifact_admission(spec, paths)
    reasons = list(dict.fromkeys(list(loaded.errors) + selection_reasons))
    details: dict[str, Any] = {
        "model_id": spec.model_id,
        "expected_schema": ARTIFACT_ADMISSION_SCHEMA,
        "expected_status": ARTIFACT_ADMISSION_STATUS,
        "source_bound": source is not None,
        "admission_scope": "COMPLETE_BINARY_STORAGE_ARTIFACT_ONLY",
        "not_a_runtime_capability_or_manager_qualification": True,
    }
    details.update(selection_details)
    document = loaded.document or {}
    if loaded.sealed:
        if document.get("schema") != ARTIFACT_ADMISSION_SCHEMA:
            reasons.append(f"unexpected schema; expected {ARTIFACT_ADMISSION_SCHEMA}")
        if document.get("status") != ARTIFACT_ADMISSION_STATUS:
            reasons.append(f"unexpected status; expected {ARTIFACT_ADMISSION_STATUS}")
        model = _mapping(document.get("model"))
        for field, expected in (
            ("key", spec.key),
            ("id", spec.model_id),
            ("repository", spec.repository),
            ("revision", spec.revision),
        ):
            if model.get(field) != expected:
                reasons.append(f"public admission receipt model.{field} does not bind the contender")
        immutable_identity = _mapping(document.get("immutable_source_identity"))
        current_revalidation = _mapping(document.get("current_source_revalidation"))
        complete_manifest = _mapping(document.get("complete_manifest"))
        native = _mapping(document.get("native_loader"))
        if source is None or not identity.passed or not revalidation.passed:
            reasons.append("current source identity/revalidation has not passed")
        else:
            _validate_receipt_identity_binding(
                immutable_identity,
                spec=spec,
                paths=paths,
                source=source,
                identity=identity,
                reasons=reasons,
            )
            _validate_receipt_revalidation_binding(
                current_revalidation,
                spec=spec,
                paths=paths,
                source=source,
                identity=identity,
                revalidation=revalidation,
                reasons=reasons,
            )
        for field in ("executable_sha256",):
            _require_sha(native, field, reasons, "native_loader")
        executable = native.get("executable_path")
        if not isinstance(executable, str) or not executable or not Path(executable).is_absolute():
            reasons.append("native_loader.executable_path must be an absolute path")
        if native.get("api") != NATIVE_COMPLETE_BINARY_API:
            reasons.append("native_loader.api is not the strict complete-binary API")
        if native.get("result_schema") != NATIVE_COMPLETE_BINARY_RESULT_SCHEMA:
            reasons.append("native_loader.result_schema is not accepted")
        if native.get("result_status") != NATIVE_COMPLETE_BINARY_RESULT_STATUS:
            reasons.append("native_loader.result_status is not accepted")
        for field in ("tensor_count", "source_weight_elements", "tensor_payload_bytes"):
            _require_positive(native, field, reasons, "native_loader")
        boundary = _mapping(document.get("claim_boundary"))
        for field in (
            "native_complete_catalog_payload_hash_layout_and_source_chain_admission_passed",
            "admission_does_not_implement_or_claim_a_native_qwen_decoder",
            "admission_does_not_claim_capability_hcli_tps_tg_or_tournament_qualification",
            "raw_bf16_source_remains_authority_teacher_only",
        ):
            _require_true(boundary, field, reasons, "claim_boundary")

        expected_manifest_path = (
            paths["complete_root"] / f"{spec.prefix}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
        )
        details["complete_manifest_max_bytes"] = _complete_manifest_max_bytes(spec)
        manifest_path_matches = _same_path(
            complete_manifest.get("path"),
            expected_manifest_path,
            reasons,
            "complete_manifest.path",
        )
        if complete_manifest.get("schema") != _manifest_schema(spec):
            reasons.append("complete_manifest.schema is not accepted for this contender")
        if complete_manifest.get("status") != COMPLETE_BINARY_MANIFEST_STATUS:
            reasons.append("complete_manifest.status is not the accepted complete candidate status")
        _require_sha(complete_manifest, "document_sha256", reasons, "complete_manifest")
        _require_sha(complete_manifest, "seal_sha256", reasons, "complete_manifest")
        if manifest_path_matches:
            manifest_loaded = _load_sealed(
                expected_manifest_path, max_bytes=_complete_manifest_max_bytes(spec)
            )
            if not manifest_loaded.sealed:
                reasons.extend(f"complete manifest: {reason}" for reason in manifest_loaded.errors)
            else:
                if manifest_loaded.document_sha256 != complete_manifest.get("document_sha256"):
                    reasons.append("complete manifest raw document SHA-256 does not match admission receipt")
                if manifest_loaded.seal_sha256 != complete_manifest.get("seal_sha256"):
                    reasons.append("complete manifest seal does not match admission receipt")
                if source is not None and identity.passed and revalidation.passed:
                    details.update(
                        _validate_manifest_catalog_and_ledger(
                            manifest_loaded.document or {},
                            spec=spec,
                            paths=paths,
                            source=source,
                            identity=identity,
                            current_revalidation=current_revalidation,
                            native=native,
                            reasons=reasons,
                        )
                    )
                _validate_admission_request(
                    document,
                    spec=spec,
                    paths=paths,
                    immutable_identity=immutable_identity,
                    current_revalidation=current_revalidation,
                    complete_manifest=complete_manifest,
                    reasons=reasons,
                )
    return _simple_check(
        "complete_admitted_artifact_at_most_1_5_bpw", loaded, details=details, reasons=reasons
    )


def _validate_runtime(
    spec: ModelSpec, paths: Mapping[str, Path], source: SourceBinding | None, artifact: Check
) -> Check:
    loaded, reasons, details = _receipt_header(
        "native_exact_full_token_runtime",
        paths["runtime"],
        schema=RUNTIME_SCHEMA,
        status="PASS_EXACT_NATIVE_FULL_TOKEN_RUNTIME",
        spec=spec,
        source=source,
        extra_bindings={"complete_artifact_admission_seal_sha256": artifact.seal_sha256},
    )
    supersession = runtime_receipt_supersession_state(
        spec,
        runtime_path=paths["runtime"],
        supersession_path=paths["runtime_supersession"],
        runtime_loaded=loaded,
    )
    # An absent canonical runtime is already explained by ``_receipt_header``.
    # Surface the supersession state only when a marker exists or when an
    # actual canonical document was explicitly made non-PASS; otherwise every
    # not-yet-built contender would receive a misleading "supersession" note.
    if supersession.get("current_runtime_eligible") is not True and (
        supersession.get("supersession_present") is True or loaded.sealed
    ):
        for reason in supersession.get("reasons") or ():
            reasons.append(f"runtime supersession: {reason}")
        state = supersession.get("state")
        if isinstance(state, str) and state:
            reasons.append(f"runtime supersession state is {state}")
    runtime = _mapping((loaded.document or {}).get("runtime"))
    if not artifact.passed:
        reasons.append("complete admitted artifact gate has not passed")
    if loaded.sealed:
        for field in (
            "native_exact_decoder",
            "full_token_execution",
            "all_layers_executed",
            "all_weight_tensors_bound",
            "tokenizer_bound",
            "prompt_template_bound",
            "model_alone",
            "no_fallback",
            "raw_bf16_teacher_not_runtime_participant",
        ):
            _require_true(runtime, field, reasons, "runtime")
        _require_positive(runtime, "measured_token_count", reasons, "runtime")
        if runtime.get("timing_scope") != "complete_model_token_loop":
            reasons.append("runtime.timing_scope must be complete_model_token_loop")
        details.update(
            {
                "measured_token_count": runtime.get("measured_token_count"),
                "timing_scope": runtime.get("timing_scope"),
                "runtime_supersession": supersession,
            }
        )
    else:
        details["runtime_supersession"] = supersession
    return _simple_check("native_exact_full_token_runtime", loaded, details=details, reasons=reasons)


def _validate_hcli(
    spec: ModelSpec, paths: Mapping[str, Path], source: SourceBinding | None, artifact: Check, runtime: Check
) -> Check:
    loaded, reasons, details = _receipt_header(
        "measured_hcli",
        paths["hcli"],
        schema=HCLI_SCHEMA,
        status="PASS_MEASURED_HCLI",
        spec=spec,
        source=source,
        extra_bindings={
            "complete_artifact_admission_seal_sha256": artifact.seal_sha256,
            "runtime_receipt_seal_sha256": runtime.seal_sha256,
        },
    )
    measurement = _mapping((loaded.document or {}).get("measurement"))
    if not artifact.passed:
        reasons.append("complete admitted artifact gate has not passed")
    if not runtime.passed:
        reasons.append("native exact full-token runtime gate has not passed")
    if loaded.sealed:
        for field in (
            "prompt_dependent_generation",
            "uses_exact_native_runtime",
            "model_alone",
            "no_fallback",
        ):
            _require_true(measurement, field, reasons, "measurement")
        _require_positive(measurement, "measured_request_count", reasons, "measurement")
        _require_positive(measurement, "completed_generated_tokens", reasons, "measurement")
        details.update(
            {
                "measured_request_count": measurement.get("measured_request_count"),
                "completed_generated_tokens": measurement.get("completed_generated_tokens"),
            }
        )
    return _simple_check("measured_hcli", loaded, details=details, reasons=reasons)


def _validate_kernel(
    spec: ModelSpec, paths: Mapping[str, Path], source: SourceBinding | None, artifact: Check, runtime: Check
) -> Check:
    loaded, reasons, details = _receipt_header(
        "custom_kernel_operational_at_least_100_tps",
        paths["kernel"],
        schema=KERNEL_SCHEMA,
        status="PASS_CUSTOM_KERNEL_FULL_MODEL_OPERATIONAL",
        spec=spec,
        source=source,
        extra_bindings={
            "complete_artifact_admission_seal_sha256": artifact.seal_sha256,
            "runtime_receipt_seal_sha256": runtime.seal_sha256,
        },
    )
    measurement = _mapping((loaded.document or {}).get("measurement"))
    if not artifact.passed:
        reasons.append("complete admitted artifact gate has not passed")
    if not runtime.passed:
        reasons.append("native exact full-token runtime gate has not passed")
    if loaded.sealed:
        for field in ("custom_kernel_used", "full_token_execution", "model_alone", "no_fallback"):
            _require_true(measurement, field, reasons, "measurement")
        _require_positive(measurement, "measured_token_count", reasons, "measurement")
        if measurement.get("timing_scope") != "complete_model_token_loop":
            reasons.append("kernel timing_scope must be complete_model_token_loop")
        tps = _finite_number(measurement.get("base_true_tokens_per_second"))
        if tps is None:
            reasons.append("kernel base_true_tokens_per_second must be finite")
        elif tps < MINIMUM_OPERATIONAL_TPS:
            reasons.append(f"kernel base_true_tokens_per_second is below {MINIMUM_OPERATIONAL_TPS}")
        details.update(
            {
                "base_true_tokens_per_second": tps,
                "minimum_operational_tps": MINIMUM_OPERATIONAL_TPS,
                "timing_scope": measurement.get("timing_scope"),
            }
        )
    return _simple_check(
        "custom_kernel_operational_at_least_100_tps", loaded, details=details, reasons=reasons
    )


def _validate_tg10(
    spec: ModelSpec,
    paths: Mapping[str, Path],
    source: SourceBinding | None,
    artifact: Check,
    runtime: Check,
    hcli: Check,
    kernel: Check,
) -> Check:
    """Validate a sealed 100-TPS operational receipt without promoting TG3.

    This is intentionally separate from the final qualification graph.  A
    valid TG10 proves an operational exact-model HCLI path for this contender;
    it does not satisfy the protected TG3, capability, Agent OS, tournament,
    or sandbox gates.
    """

    loaded, reasons, details = _receipt_header(
        "tg10_operational_exact_model_100_tps",
        paths["tg10"],
        schema=TG10_SCHEMA,
        status=TG10_STATUS,
        spec=spec,
        source=source,
        extra_bindings={
            "complete_artifact_admission_seal_sha256": artifact.seal_sha256,
            "runtime_receipt_seal_sha256": runtime.seal_sha256,
            "hcli_receipt_seal_sha256": hcli.seal_sha256,
            "kernel_receipt_seal_sha256": kernel.seal_sha256,
        },
    )
    document = loaded.document or {}
    measurement = _mapping(document.get("measurement"))
    if not artifact.passed:
        reasons.append("complete admitted artifact gate has not passed")
    if not runtime.passed:
        reasons.append("native exact full-token runtime gate has not passed")
    if not hcli.passed:
        reasons.append("measured HCLI gate has not passed")
    if not kernel.passed:
        reasons.append("100 TPS custom-kernel gate has not passed")
    if loaded.sealed:
        if document.get("rung") != "TG10":
            reasons.append("TG10 receipt rung must be TG10")
        threshold = _finite_number(document.get("required_threshold_base_true_tps"))
        if threshold is None or not math.isclose(
            threshold, MINIMUM_OPERATIONAL_TPS, rel_tol=0.0, abs_tol=0.0
        ):
            reasons.append("TG10 required threshold must be exactly 100 BASE_TRUE_TPS")
        for field in (
            "complete_native_model",
            "real_metal",
            "autoregressive_generation",
            "hcli_pass",
        ):
            _require_true(document, field, reasons, "TG10 receipt")
        if document.get("fallback_count") != 0:
            reasons.append("TG10 receipt fallback_count must be exactly 0")
        complete_bpw = _finite_number(document.get("complete_bpw"))
        if complete_bpw is None or complete_bpw <= 0.0 or complete_bpw > 1.5:
            reasons.append("TG10 receipt complete_bpw must be finite and at most 1.5")
        for field in (
            "full_token_execution",
            "model_alone",
            "no_fallback",
            "prompt_dependent_hcli_generation",
        ):
            _require_true(measurement, field, reasons, "TG10 measurement")
        if measurement.get("tg3_completed") is not False:
            reasons.append("TG10 measurement.tg3_completed must be false")
        _require_positive(measurement, "measured_token_count", reasons, "TG10 measurement")
        if measurement.get("timing_scope") != "complete_model_token_loop":
            reasons.append("TG10 measurement.timing_scope must be complete_model_token_loop")
        measurement_tps = _finite_number(measurement.get("base_true_tokens_per_second"))
        median_tps = _finite_number(document.get("median_base_true_tps"))
        sustained_tps = _finite_number(document.get("sustained_base_true_tps"))
        for label, value in (
            ("TG10 measurement base_true_tokens_per_second", measurement_tps),
            ("TG10 median_base_true_tps", median_tps),
            ("TG10 sustained_base_true_tps", sustained_tps),
        ):
            if value is None or value < MINIMUM_OPERATIONAL_TPS:
                reasons.append(f"{label} is below {MINIMUM_OPERATIONAL_TPS}")
        if (
            measurement_tps is not None
            and median_tps is not None
            and not math.isclose(measurement_tps, median_tps, rel_tol=0.0, abs_tol=0.0)
        ):
            reasons.append("TG10 measurement base_true_tokens_per_second differs from median")
        details.update(
            {
                "rung": document.get("rung"),
                "median_base_true_tps": median_tps,
                "sustained_base_true_tps": sustained_tps,
                "timing_scope": measurement.get("timing_scope"),
                "tg3_completed": measurement.get("tg3_completed"),
                "operational_only_not_tg3_or_tournament_qualification": True,
            }
        )
    return _simple_check(
        "tg10_operational_exact_model_100_tps", loaded, details=details, reasons=reasons
    )


def _validate_tg3(
    spec: ModelSpec,
    paths: Mapping[str, Path],
    source: SourceBinding | None,
    artifact: Check,
    runtime: Check,
    hcli: Check,
    kernel: Check,
) -> Check:
    loaded, reasons, details = _receipt_header(
        "tg3_at_least_333_tps",
        paths["tg3"],
        schema=TG3_SCHEMA,
        status="PASS_TG3_FULL_MODEL_QUALIFICATION",
        spec=spec,
        source=source,
        extra_bindings={
            "complete_artifact_admission_seal_sha256": artifact.seal_sha256,
            "runtime_receipt_seal_sha256": runtime.seal_sha256,
            "hcli_receipt_seal_sha256": hcli.seal_sha256,
            "kernel_receipt_seal_sha256": kernel.seal_sha256,
        },
    )
    measurement = _mapping((loaded.document or {}).get("measurement"))
    if not artifact.passed:
        reasons.append("complete admitted artifact gate has not passed")
    if not runtime.passed:
        reasons.append("native exact full-token runtime gate has not passed")
    if not hcli.passed:
        reasons.append("measured HCLI gate has not passed")
    if not kernel.passed:
        reasons.append("100 TPS custom-kernel gate has not passed")
    if loaded.sealed:
        for field in (
            "tg3_completed",
            "full_token_execution",
            "model_alone",
            "no_fallback",
            "prompt_dependent_hcli_generation",
        ):
            _require_true(measurement, field, reasons, "measurement")
        _require_positive(measurement, "measured_token_count", reasons, "measurement")
        if measurement.get("timing_scope") != "complete_model_token_loop":
            reasons.append("TG3 timing_scope must be complete_model_token_loop")
        tps = _finite_number(measurement.get("base_true_tokens_per_second"))
        if tps is None:
            reasons.append("TG3 base_true_tokens_per_second must be finite")
        elif tps < TG3_TPS:
            reasons.append(f"TG3 base_true_tokens_per_second is below {TG3_TPS}")
        details.update(
            {
                "base_true_tokens_per_second": tps,
                "tg3_minimum_tps": TG3_TPS,
                "timing_scope": measurement.get("timing_scope"),
            }
        )
    return _simple_check("tg3_at_least_333_tps", loaded, details=details, reasons=reasons)


def _validate_capability(
    spec: ModelSpec,
    paths: Mapping[str, Path],
    source: SourceBinding | None,
    artifact: Check,
    runtime: Check,
    hcli: Check,
    tg3: Check,
) -> Check:
    loaded, reasons, details = _receipt_header(
        "capability_and_evaluation_receipt",
        paths["capability"],
        schema=CAPABILITY_SCHEMA,
        status="PASS_CAPABILITY_EVALUATION",
        spec=spec,
        source=source,
        extra_bindings={
            "complete_artifact_admission_seal_sha256": artifact.seal_sha256,
            "runtime_receipt_seal_sha256": runtime.seal_sha256,
            "hcli_receipt_seal_sha256": hcli.seal_sha256,
            "tg3_receipt_seal_sha256": tg3.seal_sha256,
        },
    )
    evaluation = _mapping((loaded.document or {}).get("evaluation"))
    if not artifact.passed:
        reasons.append("complete admitted artifact gate has not passed")
    if not runtime.passed:
        reasons.append("native exact full-token runtime gate has not passed")
    if not hcli.passed:
        reasons.append("measured HCLI gate has not passed")
    if not tg3.passed:
        reasons.append("TG3 gate has not passed")
    if loaded.sealed:
        for field in (
            "complete_model_evaluation",
            "prompt_dependent_generation",
            "no_fallback",
            "frozen_hidden_task_catalog",
        ):
            _require_true(evaluation, field, reasons, "evaluation")
        _require_sha(evaluation, "hidden_task_catalog_sha256", reasons, "evaluation")
        _require_positive(evaluation, "attempted_task_count", reasons, "evaluation")
        _require_positive(evaluation, "verified_passed_task_count", reasons, "evaluation")
        attempted = evaluation.get("attempted_task_count")
        passed = evaluation.get("verified_passed_task_count")
        if _is_positive_int(attempted) and _is_positive_int(passed) and int(passed) > int(attempted):
            reasons.append("verified_passed_task_count exceeds attempted_task_count")
        details.update(
            {
                "attempted_task_count": attempted,
                "verified_passed_task_count": passed,
                "hidden_task_catalog_sha256": evaluation.get("hidden_task_catalog_sha256"),
            }
        )
    return _simple_check("capability_and_evaluation_receipt", loaded, details=details, reasons=reasons)


def _validate_suite_preflight(root: Path) -> Check:
    """Validate the independently frozen protected task/environment suite."""

    result = physical_tournament.validate_suite_preflight(root)
    path = Path(result["path"])
    document = result.get("document")
    document_value = dict(document) if isinstance(document, Mapping) else None
    document_sha256: str | None = None
    try:
        _, document_sha256, read_errors = _read_json(path)
    except (OSError, TypeError):
        read_errors = ["cannot read tournament suite preflight"]
    reasons = list(result.get("reasons") or []) + list(read_errors)
    seal_value = result.get("seal_sha256")
    return Check(
        requirement="frozen_protected_tournament_suite_preflight",
        passed=bool(result.get("passed")) and not reasons,
        path=path,
        seal_sha256=str(seal_value) if _is_sha256(seal_value) else None,
        reasons=list(dict.fromkeys(str(reason) for reason in reasons)),
        details=dict(result.get("details") or {}),
        document=document_value,
        document_sha256=document_sha256,
    )


def _require_nonnegative_finite(
    container: Mapping[str, Any], field: str, reasons: list[str], label: str
) -> float | None:
    value = _finite_number(container.get(field))
    if value is None or value < 0.0:
        reasons.append(f"{label}.{field} must be a finite non-negative number")
        return None
    return value


def _validate_manager_operations(
    spec: ModelSpec,
    paths: Mapping[str, Path],
    source: SourceBinding | None,
    artifact: Check,
    runtime: Check,
    hcli: Check,
    kernel: Check,
    tg3: Check,
    capability: Check,
    suite: Check,
) -> Check:
    """Require the final Agent-OS/session/recovery receipt on the exact model.

    This is intentionally later than raw HCLI and capability receipts.  The
    manager must prove the actual final fast artifact under multi-session,
    restart, residency, rollback, storage, and context/KV conditions, tied to
    the frozen tournament environment it will subsequently enter.
    """

    loaded, reasons, details = _receipt_header(
        "final_manager_operations_agent_os_session_restart_residency_rollback_storage",
        paths["manager_operations"],
        schema=MANAGER_OPERATIONS_SCHEMA,
        status=MANAGER_OPERATIONS_STATUS,
        spec=spec,
        source=source,
        extra_bindings={
            "complete_artifact_admission_seal_sha256": artifact.seal_sha256,
            "runtime_receipt_seal_sha256": runtime.seal_sha256,
            "hcli_receipt_seal_sha256": hcli.seal_sha256,
            "kernel_receipt_seal_sha256": kernel.seal_sha256,
            "tg3_receipt_seal_sha256": tg3.seal_sha256,
            "capability_evaluation_receipt_seal_sha256": capability.seal_sha256,
            "tournament_suite_preflight_seal_sha256": suite.seal_sha256,
        },
    )
    if not artifact.passed:
        reasons.append("complete admitted artifact gate has not passed")
    if not runtime.passed:
        reasons.append("native exact full-token runtime gate has not passed")
    if not hcli.passed:
        reasons.append("measured HCLI gate has not passed")
    if not kernel.passed:
        reasons.append("100 TPS custom-kernel gate has not passed")
    if not tg3.passed:
        reasons.append("TG3 gate has not passed")
    if not capability.passed:
        reasons.append("capability/evaluation gate has not passed")
    if not suite.passed:
        reasons.append("frozen protected tournament suite preflight has not passed")
    operations = _mapping((loaded.document or {}).get("operations"))
    if loaded.sealed:
        for field in (
            "uses_exact_native_runtime",
            "uses_admitted_gravity_artifact_only",
            "agent_os_live",
            "context_kv_passed",
            "restart_passed",
            "residency_fit_passed",
            "rollback_passed",
            "storage_rollback_passed",
            "tool_recovery_passed",
            "long_unattended_task_passed",
            "single_model_body_shared_across_sessions",
            "no_fallback",
        ):
            _require_true(operations, field, reasons, "operations")
        if operations.get("fallback_count") != 0:
            reasons.append("operations.fallback_count must be exactly zero")
        endpoint = _mapping(operations.get("hcli_endpoint"))
        if endpoint.get("protocol") != "openai_chat_completions_v1":
            reasons.append("operations.hcli_endpoint.protocol must be openai_chat_completions_v1")
        if endpoint.get("host") != "127.0.0.1":
            reasons.append("operations.hcli_endpoint.host must be loopback 127.0.0.1")
        port = endpoint.get("port")
        if not isinstance(port, int) or isinstance(port, bool) or not 1024 <= port <= 65535:
            reasons.append("operations.hcli_endpoint.port must be a non-privileged integer")
        if endpoint.get("health_path") != "/healthz" or endpoint.get("chat_path") != "/v1/chat/completions":
            reasons.append("operations.hcli_endpoint paths must bind the frozen HCLI protocol")
        if endpoint.get("gravity_artifact_id") != spec.gravity_artifact_id:
            reasons.append("operations.hcli_endpoint does not bind this Gravity artifact")
        if not isinstance(endpoint.get("model"), str) or not endpoint.get("model"):
            reasons.append("operations.hcli_endpoint.model must be non-empty")
        suite_environment = _mapping((suite.document or {}).get("tool_environment"))
        expected_environment_sha = (suite.document or {}).get("tool_environment_sha256")
        if not _is_sha256(expected_environment_sha):
            reasons.append("frozen suite tool environment SHA-256 is unavailable")
        elif operations.get("tool_environment_sha256") != expected_environment_sha:
            reasons.append("operations tool environment does not match the frozen tournament suite")
        if suite_environment.get("endpoint_contract") != {
            "protocol": "openai_chat_completions_v1",
            "loopback_only": True,
            "health_path": "/healthz",
            "chat_path": "/v1/chat/completions",
            "candidate_tool_policy": "no_host_shell_or_hidden_membership_files",
            "candidate_receives_only_current_evaluation_prompt": True,
        }:
            reasons.append("frozen suite endpoint contract is unavailable or drifted")

        sessions = operations.get("session_measurements")
        expected_sessions = {1, 2, 4, 8}
        observed_sessions: set[int] = set()
        if not isinstance(sessions, list):
            reasons.append("operations.session_measurements must be an array")
            sessions = []
        for row in sessions:
            measurement = _mapping(row)
            session_count = measurement.get("logical_sessions")
            if not isinstance(session_count, int) or isinstance(session_count, bool) or session_count <= 0:
                reasons.append("session measurement logical_sessions must be a positive integer")
                continue
            if session_count in observed_sessions:
                reasons.append("session measurements must not duplicate logical_sessions")
            observed_sessions.add(session_count)
            for field in (
                "raw_model_tps",
                "hcli_tps",
                "per_session_p99_ms",
                "verified_tasks_per_hour",
                "kv_state_bytes",
                "context_compile_latency_ms",
                "tool_wait_ms",
                "queue_wait_ms",
            ):
                _require_nonnegative_finite(measurement, field, reasons, "session measurement")
            if _finite_number(measurement.get("raw_model_tps")) in (None, 0.0):
                reasons.append("session measurement raw_model_tps must be positive")
            if _finite_number(measurement.get("hcli_tps")) in (None, 0.0):
                reasons.append("session measurement hcli_tps must be positive")
            if measurement.get("weight_reuse_observed") is not True:
                reasons.append("session measurement must observe cross-session weight reuse")
            if measurement.get("starvation_free") is not True:
                reasons.append("session measurement must be starvation free")
        if not expected_sessions.issubset(observed_sessions):
            reasons.append("session measurements must include 1, 2, 4, and 8 logical sessions")
        details.update(
            {
                "required_sessions": sorted(expected_sessions),
                "observed_sessions": sorted(observed_sessions),
                "hcli_endpoint": dict(endpoint),
                "frozen_suite_bound": suite.passed,
            }
        )
    return _simple_check(
        "final_manager_operations_agent_os_session_restart_residency_rollback_storage",
        loaded,
        details=details,
        reasons=reasons,
    )


def _safe_progress(document: Mapping[str, Any]) -> dict[str, Any]:
    """Extract mutable facts which can demonstrate work, never heartbeats alone."""

    population = _mapping(document.get("population"))
    current = _mapping(document.get("current_experiment"))
    next_item = _mapping(document.get("next_experiment"))
    pack = _mapping(document.get("complete_pack"))
    pack_progress = _mapping(pack.get("progress"))
    integrity = _mapping(document.get("artifact_integrity"))
    return {
        "completed_candidate_count": population.get("completed_candidate_count"),
        "candidate_count": population.get("candidate_count"),
        "current_sequence": current.get("sequence"),
        "current_candidate_id": current.get("candidate_id"),
        "next_sequence": next_item.get("sequence"),
        "next_candidate_id": next_item.get("candidate_id"),
        "last_material_progress_at": document.get("last_material_progress_at"),
        "last_verified_artifact_candidate": integrity.get("candidate_id"),
        "worker_complete_pack_cursor": pack_progress.get("next_cursor"),
        "worker_complete_pack_completed_tensors": pack_progress.get("completed_tensors"),
    }


def _observed_lane_progress(document: Mapping[str, Any] | None) -> dict[str, Any]:
    value = _mapping(document)
    progress = _mapping(value.get("progress"))
    return {
        "phase": value.get("phase"),
        "completed_tensors": progress.get("completed_tensors"),
        "next_cursor": progress.get("next_cursor"),
        "planned_tensors": progress.get("planned_tensors"),
        "artifact_bytes": progress.get("artifact_bytes"),
        "current_artifact": value.get("current_artifact"),
    }


def _pid_alive(value: Any) -> bool:
    if not _is_positive_int(value):
        return False
    try:
        os.kill(int(value), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _liveness(
    spec: ModelSpec,
    paths: Mapping[str, Path],
    previous: Mapping[str, Any] | None,
) -> dict[str, Any]:
    worker = _load_sealed(paths["worker"])
    worker_observation = _load_observation(paths["worker"])
    worker_doc = worker.document or _mapping(worker_observation.get("document"))
    worker_model = _mapping(worker_doc.get("model"))
    worker_reasons: list[str] = []
    if not worker_doc:
        worker_reasons.extend(worker_observation["errors"])
    if worker_doc:
        if worker_doc.get("schema") != "hawking.ascension.dual_gravity_worker.v1":
            worker_reasons.append("unexpected physical worker schema")
        if worker_model.get("id") != spec.model_id or worker_model.get("key") != spec.key:
            worker_reasons.append("worker status does not bind expected model")
    observed_lanes = {
        "complete_pack": _load_observation(paths["pack_status"]),
        "runtime_watchdog": _load_observation(paths["runtime_status"]),
        "tg3_watchdog": _load_observation(paths["tg3_status"]),
    }
    observation_summary = {
        key: {
            "path": value["path"],
            "state": value["state"],
            "sha256": value["sha256"],
            "phase": _mapping(value["document"]).get("phase"),
        }
        for key, value in observed_lanes.items()
    }
    material = {
        "worker": _safe_progress(worker_doc),
        "complete_pack": _observed_lane_progress(observed_lanes["complete_pack"]["document"]),
        "runtime_watchdog": _observed_lane_progress(observed_lanes["runtime_watchdog"]["document"]),
        "tg3_watchdog": _observed_lane_progress(observed_lanes["tg3_watchdog"]["document"]),
    }
    material_fingerprint = _digest(material)
    heartbeats = {
        "worker": worker_doc.get("heartbeat"),
        "complete_pack": _mapping(observed_lanes["complete_pack"]["document"]).get("heartbeat"),
        "runtime_watchdog": _mapping(observed_lanes["runtime_watchdog"]["document"]).get("heartbeat"),
        "tg3_watchdog": _mapping(observed_lanes["tg3_watchdog"]["document"]).get("heartbeat"),
    }
    prior = _mapping(_mapping(previous).get("liveness")).get(spec.key)
    prior_liveness = _mapping(prior)
    prior_fingerprint = prior_liveness.get("material_fingerprint")
    prior_heartbeats = _mapping(prior_liveness.get("observed_heartbeats"))
    pid = worker_doc.get("pid")
    pid_is_alive = _pid_alive(pid)
    heartbeat_advanced = any(
        _is_positive_int(heartbeats.get(name))
        and _is_positive_int(prior_heartbeats.get(name))
        and int(heartbeats[name]) > int(prior_heartbeats[name])
        for name in heartbeats
    )
    material_changed = isinstance(prior_fingerprint, str) and prior_fingerprint != material_fingerprint
    if worker_reasons:
        activity = "NO_USABLE_PHYSICAL_WORKER_STATUS"
    elif not isinstance(prior_fingerprint, str):
        activity = "BASELINE_RECORDED_AWAITING_MATERIAL_DELTA"
    elif material_changed and pid_is_alive:
        activity = "ACTIVE_WITH_MATERIAL_PROGRESS"
    elif material_changed:
        activity = "MATERIAL_PROGRESS_OBSERVED_BUT_WORKER_PID_NOT_LIVE"
    elif heartbeat_advanced:
        activity = "HEARTBEAT_ADVANCED_WITHOUT_MATERIAL_PROGRESS"
    else:
        activity = "NO_NEW_MATERIAL_PROGRESS"
    return {
        "activity": activity,
        "worker_status": {
            "path": str(paths["worker"]),
            "sealed": worker.sealed,
            "trust": "SEALED_OBSERVATION" if worker.sealed else "OBSERVATIONAL_UNSEALED_STATUS",
            "seal_sha256": worker.seal_sha256,
            "declared_status": worker_doc.get("status"),
            "declared_phase": worker_doc.get("phase"),
            "pid": pid if _is_positive_int(pid) else None,
            "declared_ppid": worker_doc.get("ppid") if _is_positive_int(worker_doc.get("ppid")) else None,
            "pid_observed_alive": pid_is_alive,
            "errors": worker_reasons,
            "seal_validation_errors": [] if worker.sealed else list(worker.errors),
        },
        "observational_lanes": observation_summary,
        "material_fingerprint": material_fingerprint,
        "previous_material_fingerprint": prior_fingerprint if isinstance(prior_fingerprint, str) else None,
        "material_changed_since_prior_gate_cycle": material_changed,
        "observed_heartbeats": heartbeats,
        "heartbeat_advanced_since_prior_gate_cycle": heartbeat_advanced,
        "heartbeat_is_not_material_progress": True,
        "qualification_dependency": "none; liveness is operational evidence, not a promotion shortcut",
    }


def _validate_final_review_marker(
    root: Path,
    per_model: Mapping[str, Mapping[str, Check]],
    sources: Mapping[str, SourceBinding | None],
) -> Check:
    path = root / "lifecycle" / FINAL_REVIEW_FILENAME
    loaded = _load_sealed(path)
    reasons = list(loaded.errors)
    document = loaded.document or {}
    details: dict[str, Any] = {
        "expected_schema": FINAL_REVIEW_SCHEMA,
        "marker_is_external_to_gatekeeper": True,
    }
    if loaded.sealed:
        if document.get("schema") != FINAL_REVIEW_SCHEMA:
            reasons.append("unexpected protected final-review marker schema")
        if document.get("status") != "PROTECTED_FINAL_REVIEW_MARKER_PRESENT":
            reasons.append("protected final-review marker is not present")
        if document.get("authority") not in {"protected_controller", "human_operator"}:
            reasons.append("marker authority must be protected_controller or human_operator")
        if document.get("does_not_choose_winner") is not True:
            reasons.append("marker must explicitly prohibit winner selection")
        if document.get("fixed_candidate_order") != [spec.model_id for spec in MODEL_SPECS]:
            reasons.append("marker candidate order differs from fixed physical order")
        reviews = document.get("reviews")
        if not isinstance(reviews, list) or len(reviews) != len(MODEL_SPECS):
            reasons.append("marker must contain exactly one review binding per contender")
            by_model: dict[str, dict[str, Any]] = {}
        else:
            by_model = {}
            for row in reviews:
                review = _mapping(row)
                model_id = review.get("model_id")
                if not isinstance(model_id, str) or model_id in by_model:
                    reasons.append("marker review list has missing or duplicate model_id")
                    continue
                by_model[model_id] = review
        for spec in MODEL_SPECS:
            review = by_model.get(spec.model_id, {})
            checks = per_model.get(spec.key, {})
            source = sources.get(spec.key)
            expected = {
                "source_content_identity_sha256": source.content_identity_sha256 if source else None,
                "source_revalidation_seal_sha256": source.revalidation_seal_sha256 if source else None,
                "complete_artifact_admission_seal_sha256": checks.get("complete_artifact").seal_sha256 if checks.get("complete_artifact") else None,
                "runtime_receipt_seal_sha256": checks.get("runtime").seal_sha256 if checks.get("runtime") else None,
                "hcli_receipt_seal_sha256": checks.get("hcli").seal_sha256 if checks.get("hcli") else None,
                "kernel_receipt_seal_sha256": checks.get("kernel").seal_sha256 if checks.get("kernel") else None,
                "tg3_receipt_seal_sha256": checks.get("tg3").seal_sha256 if checks.get("tg3") else None,
                "capability_evaluation_receipt_seal_sha256": checks.get("capability").seal_sha256 if checks.get("capability") else None,
            }
            if review.get("review_disposition") != "PRESENT_TO_FINAL_REVIEW":
                reasons.append(f"marker review disposition is not present-to-final-review for {spec.model_id}")
            for field, expected_value in expected.items():
                if not _is_sha256(expected_value):
                    reasons.append(f"upstream evidence is not yet available for marker field {field} ({spec.model_id})")
                elif review.get(field) != expected_value:
                    reasons.append(f"marker binding mismatch for {field} ({spec.model_id})")
        details.update(
            {
                "authority_claim": document.get("authority"),
                "review_count": len(reviews) if isinstance(reviews, list) else 0,
                "seal_proves_document_integrity_not_human_identity": True,
            }
        )
    return _simple_check("protected_final_review_marker", loaded, details=details, reasons=reasons)


def _contract(paths: Mapping[str, Path]) -> dict[str, Any]:
    """Publish the durable producer contract alongside each blocked gate."""

    return {
        "complete_artifact_admission": {
            "path": str(paths["artifact_admission"]),
            "schema": ARTIFACT_ADMISSION_SCHEMA,
            "status": ARTIFACT_ADMISSION_STATUS,
            "required_facts": [
                "sealed public native complete-binary admission receipt",
                "sealed immutable source identity and current full-shard revalidation bindings",
                "sealed canonical admission request bound to the same source and manifest",
                "strict native reader result for complete catalog, payload, hash, and layout facts",
                "current sealed complete manifest with every tensor accounted for",
                "complete physical ledger reconciles and complete_physical_bpw <= 1.5",
                "storage-artifact admission only; it is not runtime, manager, capability, HCLI, TPS, TG, or tournament qualification",
            ],
        },
        "native_runtime": {
            "path": str(paths["runtime"]),
            "schema": RUNTIME_SCHEMA,
            "status": "PASS_EXACT_NATIVE_FULL_TOKEN_RUNTIME",
            "required_facts": [
                "native exact decoder",
                "all layers and all weight tensors bound",
                "complete model token loop",
                "no fallback; raw BF16 remains teacher only",
            ],
            "supersession_contract": {
                "path": str(paths["runtime_supersession"]),
                "schema": RUNTIME_SUPERSESSION_SCHEMA,
                "rule": "if present, a sealed revocation blocks the named receipt seal and executable SHA; archived PASS receipts are negative-science history only",
            },
        },
        "measured_hcli": {
            "path": str(paths["hcli"]),
            "schema": HCLI_SCHEMA,
            "status": "PASS_MEASURED_HCLI",
            "required_facts": ["prompt-dependent generation", "measured requests", "no fallback"],
        },
        "custom_kernel_100_tps": {
            "path": str(paths["kernel"]),
            "schema": KERNEL_SCHEMA,
            "status": "PASS_CUSTOM_KERNEL_FULL_MODEL_OPERATIONAL",
            "required_facts": [
                "custom kernel in complete model token loop",
                f"base_true_tokens_per_second >= {MINIMUM_OPERATIONAL_TPS}",
                "model alone; no fallback",
            ],
        },
        "tg3": {
            "path": str(paths["tg3"]),
            "schema": TG3_SCHEMA,
            "status": "PASS_TG3_FULL_MODEL_QUALIFICATION",
            "required_facts": [
                "TG3 complete model token loop",
                f"base_true_tokens_per_second >= {TG3_TPS}",
                "prompt-dependent HCLI; no fallback",
            ],
        },
        "capability_evaluation": {
            "path": str(paths["capability"]),
            "schema": CAPABILITY_SCHEMA,
            "status": "PASS_CAPABILITY_EVALUATION",
            "required_facts": ["frozen hidden catalog", "verified completed tasks", "no fallback"],
        },
        "final_manager_operations": {
            "path": str(paths["manager_operations"]),
            "schema": MANAGER_OPERATIONS_SCHEMA,
            "status": MANAGER_OPERATIONS_STATUS,
            "required_facts": [
                "the exact admitted fast artifact under real HCLI",
                "Agent OS and Context/KV tests at 1, 2, 4, and 8 logical sessions",
                "raw versus HCLI TPS, per-session p99, verified tasks/hour, state/context/tool/queue measurements",
                "single shared model body with observed cross-session weight reuse and no starvation",
                "restart, residency/fit, rollback, storage/rollback, tool recovery, and unattended-task passes",
                "zero fallback and endpoint binding to the frozen protected tournament environment",
            ],
        },
    }


def build_gate_status(root: str | Path, *, previous: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Reconcile physical evidence into a sealed-ready, fail-closed status payload.

    The returned document is unsigned so callers can atomically attach the
    seal after their lock is held.  No blocked condition raises: absence is the
    normal long-running state of this campaign.
    """

    resolved = Path(root).expanduser().resolve()
    per_model_checks: dict[str, dict[str, Check]] = {}
    operational_tg10_checks: dict[str, Check] = {}
    sources: dict[str, SourceBinding | None] = {}
    models: list[dict[str, Any]] = []
    liveness: dict[str, Any] = {}
    suite = _validate_suite_preflight(resolved)
    for spec in MODEL_SPECS:
        model_paths = _paths(resolved, spec)
        identity, initial_source = _validate_source_identity(spec, model_paths["identity"])
        revalidation, source = _validate_revalidation(spec, model_paths["revalidation"], identity, initial_source)
        artifact = _validate_artifact(spec, model_paths, source, identity, revalidation)
        runtime = _validate_runtime(spec, model_paths, source, artifact)
        hcli = _validate_hcli(spec, model_paths, source, artifact, runtime)
        kernel = _validate_kernel(spec, model_paths, source, artifact, runtime)
        tg10 = _validate_tg10(spec, model_paths, source, artifact, runtime, hcli, kernel)
        tg3 = _validate_tg3(spec, model_paths, source, artifact, runtime, hcli, kernel)
        capability = _validate_capability(spec, model_paths, source, artifact, runtime, hcli, tg3)
        manager_operations = _validate_manager_operations(
            spec,
            model_paths,
            source,
            artifact,
            runtime,
            hcli,
            kernel,
            tg3,
            capability,
            suite,
        )
        checks = {
            "source_identity": identity,
            "source_revalidation": revalidation,
            "complete_artifact": artifact,
            "runtime": runtime,
            "hcli": hcli,
            "kernel": kernel,
            "tg3": tg3,
            "capability": capability,
            "manager_operations": manager_operations,
        }
        per_model_checks[spec.key] = checks
        operational_tg10_checks[spec.key] = tg10
        sources[spec.key] = source
        prerequisites_passed = all(item.passed for item in checks.values())
        liveness[spec.key] = _liveness(spec, model_paths, previous)
        models.append(
            {
                "key": spec.key,
                "source_teacher_model_id": spec.model_id,
                "gravity_artifact_id": spec.gravity_artifact_id,
                "raw_bf16_source_is_teacher_authority_not_tournament_participant": True,
                "pre_final_review_qualification": "PASS" if prerequisites_passed else "BLOCKED",
                "requirements": {
                    "verified_raw_source_identity": identity.public(),
                    "current_source_revalidation": revalidation.public(),
                    "complete_admitted_artifact_at_most_1_5_bpw": artifact.public(),
                    "native_exact_full_token_runtime": runtime.public(),
                    "measured_hcli": hcli.public(),
                    "custom_kernel_operational_at_least_100_tps": kernel.public(),
                    "tg10_operational_exact_model_100_tps": tg10.public(),
                    "tg3_at_least_333_tps": tg3.public(),
                    "capability_and_evaluation_receipt": capability.public(),
                    "final_manager_operations_agent_os_session_restart_residency_rollback_storage": manager_operations.public(),
                },
                "future_receipt_contract": _contract(model_paths),
            }
        )
    marker = _validate_final_review_marker(resolved, per_model_checks, sources)
    operational_ascent_ready = all(
        operational_tg10_checks[spec.key].passed for spec in MODEL_SPECS
    )
    operational_ascent_evidence = {
        "fixed_candidate_order": [spec.gravity_artifact_id for spec in MODEL_SPECS],
        "models": {
            spec.key: {
                "source_identity_seal_sha256": per_model_checks[spec.key]["source_identity"].seal_sha256,
                "source_revalidation_seal_sha256": per_model_checks[spec.key]["source_revalidation"].seal_sha256,
                "complete_artifact_admission_seal_sha256": per_model_checks[spec.key]["complete_artifact"].seal_sha256,
                "runtime_receipt_seal_sha256": per_model_checks[spec.key]["runtime"].seal_sha256,
                "hcli_receipt_seal_sha256": per_model_checks[spec.key]["hcli"].seal_sha256,
                "kernel_receipt_seal_sha256": per_model_checks[spec.key]["kernel"].seal_sha256,
                "tg10_receipt_seal_sha256": operational_tg10_checks[spec.key].seal_sha256,
            }
            for spec in MODEL_SPECS
        },
    }
    operational_ascent_fingerprint = (
        _digest(operational_ascent_evidence) if operational_ascent_ready else None
    )
    pre_review_ready = all(
        model["pre_final_review_qualification"] == "PASS" for model in models
    )
    qualifications_complete = pre_review_ready and suite.passed
    qualification_evidence = {
        "fixed_candidate_order": [spec.gravity_artifact_id for spec in MODEL_SPECS],
        "suite_preflight_seal_sha256": suite.seal_sha256,
        "models": {
            spec.key: {
                name: per_model_checks[spec.key][name].seal_sha256
                for name in (
                    "source_identity",
                    "source_revalidation",
                    "complete_artifact",
                    "runtime",
                    "hcli",
                    "kernel",
                    "tg3",
                    "capability",
                    "manager_operations",
                )
            }
            for spec in MODEL_SPECS
        },
    }
    qualification_fingerprint = (
        _digest(qualification_evidence) if qualifications_complete else None
    )
    handoff = physical_tournament.launch_state(
        resolved, qualification_fingerprint=qualification_fingerprint
    )
    handoff_state = str(handoff.get("state") or "NOT_LAUNCHED")
    if handoff_state == physical_tournament.RUNNING:
        phase = "MANAGER_TOURNAMENT_RUNNING"
    elif handoff_state == physical_tournament.COMPLETE:
        phase = "MANAGER_TOURNAMENT_COMPLETE_HUMAN_DECISION_REQUIRED"
    elif handoff_state == physical_tournament.ABORTED:
        phase = "MANAGER_TOURNAMENT_ABORTED_FAIL_CLOSED"
    elif handoff_state == "LAUNCH_FAILED_FAIL_CLOSED":
        phase = "MANAGER_TOURNAMENT_LAUNCH_FAILED_FAIL_CLOSED"
    elif qualifications_complete:
        phase = "QUALIFICATIONS_COMPLETE"
    else:
        phase = "ARMED_WAITING_FOR_QUALIFICATIONS"
    return {
        "schema": GATE_SCHEMA,
        "status": phase,
        "recorded_at": _utc_now(),
        "gatekeeper": {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "root": str(resolved),
            "mode": "OBSERVE_AND_FAIL_CLOSED",
        },
        "fixed_candidate_order": [spec.gravity_artifact_id for spec in MODEL_SPECS],
        "models": models,
        "frozen_protected_tournament_suite_preflight": suite.public(),
        "protected_final_review_marker": marker.public(),
        "pre_final_review_qualification_complete": pre_review_ready,
        "qualifications_complete": qualifications_complete,
        "qualification_evidence": qualification_evidence,
        "qualification_fingerprint": qualification_fingerprint,
        "ready_for_final_review": qualifications_complete,
        "protected_final_review_marker_is_informational_not_a_launch_override": True,
        "operational_ascent": {
            "status": (
                OPERATIONAL_ASCENT_EARNED
                if operational_ascent_ready
                else OPERATIONAL_ASCENT_WAITING
            ),
            "both_valid_tg10_receipts": operational_ascent_ready,
            "evidence": operational_ascent_evidence,
            "evidence_fingerprint": operational_ascent_fingerprint,
            "next_transition": (
                "continue independently toward TG3; protected tournament remains unavailable"
                if operational_ascent_ready
                else "await both exact sealed TG10 operational receipts"
            ),
            "claim_boundary": {
                "does_not_satisfy_tg3": True,
                "does_not_satisfy_capability_or_agent_os": True,
                "does_not_change_protected_final_tournament_qualification": True,
                "does_not_launch_tournament_or_activate_sandbox": True,
            },
        },
        "tournament_execution": {
            "status": handoff_state,
            "winner_selection": "DISABLED",
            "winner": None,
            "launch_receipt_path": str(resolved / "lifecycle" / physical_tournament.LAUNCH_FILENAME),
            "runner_state_path": str(resolved / "lifecycle" / physical_tournament.RUNNER_FILENAME),
            "automatic_handoff": "ARMED_ON_EXACT_QUALIFICATIONS" if qualifications_complete else "WAITING_FOR_EXACT_QUALIFICATIONS",
            "reasons": list(handoff.get("reasons") or []),
        },
        "liveness": liveness,
        "claim_boundary": {
            "raw_bf16_models_are_source_authorities_and_teachers_only": True,
            "component_kernel_timing_is_never_model_tps": True,
            "unsealed_watchdog_status_is_observational_only": True,
            "heartbeat_alone_is_never_material_progress": True,
            "gatekeeper_does_not_run_models_or_benchmarks": True,
            "gatekeeper_may_detach_only_the_exact_protected_runner_after_all_receipts_pass": True,
            "gatekeeper_does_not_choose_winner_or_activate_sandbox": True,
            "gatekeeper_does_not_create_protected_final_review_marker": True,
        },
    }


def build_operational_ascent_status(gate_status: Mapping[str, Any]) -> dict[str, Any]:
    """Seal the non-tournament operational ascent overlay from a fresh gate.

    The overlay intentionally consumes a just-built, sealed physical gate
    instead of launching a second validator/controller.  It records only the
    two independently validated TG10 receipts and their exact upstream
    bindings.  It is not an input to final-review qualification or protected
    tournament launch.
    """

    ascent = _mapping(gate_status.get("operational_ascent"))
    earned = ascent.get("status") == OPERATIONAL_ASCENT_EARNED
    status = OPERATIONAL_ASCENT_EARNED if earned else OPERATIONAL_ASCENT_WAITING
    gate_seal = gate_status.get("seal_sha256")
    if not _is_sha256(gate_seal):
        raise PhysicalGatekeeperError("operational ascent requires a sealed physical gate")
    return seal(
        {
            "schema": OPERATIONAL_ASCENT_SCHEMA,
            "status": status,
            "recorded_at": _utc_now(),
            "physical_gate_status_path": str(
                Path(_mapping(gate_status.get("gatekeeper")).get("root", ""))
                / "lifecycle"
                / GATE_FILENAME
            ),
            "physical_gate_status_seal_sha256": gate_seal,
            "both_valid_tg10_receipts": earned,
            "evidence": _mapping(ascent.get("evidence")),
            "evidence_fingerprint": ascent.get("evidence_fingerprint") if earned else None,
            "next_transition": ascent.get("next_transition"),
            "protected_tournament": {
                "launch_requested": False,
                "qualification_override": False,
                "tg3_remains_required": True,
                "capability_agent_os_and_final_review_remain_required": True,
            },
            "claim_boundary": {
                "operational_ascent_is_not_tg3": True,
                "operational_ascent_is_not_a_capability_or_agent_os_pass": True,
                "operational_ascent_does_not_launch_or_score_the_protected_tournament": True,
                "operational_ascent_does_not_activate_the_sandbox": True,
            },
        }
    )


def build_tournament_workflow(gate_status: Mapping[str, Any]) -> dict[str, Any]:
    """Create the companion workflow without giving it winner authority."""

    phase = str(gate_status.get("status") or "ARMED_WAITING_FOR_QUALIFICATIONS")
    qualified = gate_status.get("qualifications_complete") is True
    execution = _mapping(gate_status.get("tournament_execution"))
    candidates = []
    for model in gate_status.get("models", []):
        row = _mapping(model)
        candidates.append(
            {
                "gravity_artifact_id": row.get("gravity_artifact_id"),
                "source_teacher_model_id": row.get("source_teacher_model_id"),
                "pre_final_review_qualification": row.get("pre_final_review_qualification"),
                "raw_bf16_source_is_not_a_tournament_participant": True,
            }
        )
    return seal(
        {
            "schema": WORKFLOW_SCHEMA,
            "status": "PHYSICAL_TOURNAMENT_WORKFLOW_ARMED_FAIL_CLOSED",
            "recorded_at": _utc_now(),
            "gate_status_schema": gate_status.get("schema"),
            "gate_status_seal_sha256": gate_status.get("seal_sha256"),
            "runtime_phase": phase,
            "fixed_candidate_order": [spec.gravity_artifact_id for spec in MODEL_SPECS],
            "candidates": candidates,
            "required_preconditions": {
                "both_source_identities_and_current_revalidations_verified": True,
                "each_complete_artifact_admitted_at_most_1_5_bpw": True,
                "each_exact_native_full_token_runtime_measured": True,
                "each_hcli_measured_prompt_dependently": True,
                "each_custom_kernel_checkpoint_at_least_100_true_tps": True,
                "each_tg3_checkpoint_at_least_333_true_tps": True,
                "each_capability_evaluation_sealed": True,
                "each_final_manager_operations_receipt_sealed": True,
                "frozen_protected_tournament_suite_preflight_required": True,
            },
            "observed_precondition_satisfaction": {
                "both_candidates_pre_final_review_qualified": gate_status.get(
                    "pre_final_review_qualification_complete"
                ) is True,
                "both_final_manager_operations_receipts": all(
                    _mapping(_mapping(row).get("requirements"))
                    .get("final_manager_operations_agent_os_session_restart_residency_rollback_storage", {})
                    .get("state")
                    == "PASS"
                    for row in gate_status.get("models", [])
                    if isinstance(row, Mapping)
                ),
                "frozen_suite_preflight": _mapping(
                    gate_status.get("frozen_protected_tournament_suite_preflight")
                ).get("state"),
                "qualifications_complete": qualified,
            },
            "authority": {
                "winner_selection": "DISABLED",
                "winner": None,
                "tournament_launch": execution.get("status") or "NOT_LAUNCHED",
                "automatic_launch_after_exact_qualification": True,
                "requires_separate_protected_tournament_execution_receipt": False,
                "requires_separate_protected_winner_receipt": True,
                "raw_bf16_sources_are_teacher_authorities_not_participants": True,
            },
            "claim_boundary": {
                "workflow_is_not_a_tournament_result": True,
                "workflow_does_not_score_or_rank_candidates": True,
                "workflow_does_not_choose_winner": True,
                "workflow_does_not_activate_sandbox": True,
            },
        }
    )


def _previous_status(path: Path) -> dict[str, Any] | None:
    loaded = _load_sealed(path)
    document = loaded.document
    if not loaded.sealed or not isinstance(document, Mapping) or document.get("schema") != GATE_SCHEMA:
        return None
    return dict(document)


def _request_qualified_tournament_handoff(
    root: Path,
    gate: Mapping[str, Any],
    *,
    request_launch: Callable[..., Mapping[str, Any]] = physical_tournament.request_launch,
) -> Mapping[str, Any] | None:
    """Start the protected evaluator exactly once after the sealed gate passes.

    This function receives only public gate facts.  It never loads model data
    and makes no performance or winner decision.  The unique qualification
    fingerprint is the idempotency key: a watcher restart cannot make a second
    launch for the same pair of exact receipts.
    """

    if gate.get("status") != "QUALIFICATIONS_COMPLETE":
        return None
    fingerprint = gate.get("qualification_fingerprint")
    suite = _mapping(gate.get("frozen_protected_tournament_suite_preflight"))
    suite_seal = suite.get("seal_sha256")
    if not _is_sha256(fingerprint) or not _is_sha256(suite_seal):
        raise PhysicalGatekeeperError("qualified gate lacks sealed handoff bindings")
    state = physical_tournament.launch_state(root, qualification_fingerprint=str(fingerprint))
    if state.get("state") != "NOT_LAUNCHED":
        return None
    endpoints: dict[str, Mapping[str, Any]] = {}
    for row in gate.get("models", []):
        model = _mapping(row)
        key = model.get("key")
        requirement = _mapping(_mapping(model.get("requirements")).get(
            "final_manager_operations_agent_os_session_restart_residency_rollback_storage"
        ))
        if key not in {"qwen30", "qwen80"} or requirement.get("state") != "PASS":
            raise PhysicalGatekeeperError("qualified gate lacks a passing final manager operations receipt")
        endpoint = _mapping(_mapping(requirement.get("details")).get("hcli_endpoint"))
        if not endpoint:
            raise PhysicalGatekeeperError("final manager operations receipt lacks HCLI endpoint details")
        endpoints[str(key)] = endpoint
    if set(endpoints) != {"qwen30", "qwen80"}:
        raise PhysicalGatekeeperError("qualified gate has an incomplete endpoint set")
    return request_launch(
        root,
        qualification_fingerprint=str(fingerprint),
        qualification_evidence=_mapping(gate.get("qualification_evidence")),
        suite_seal_sha256=str(suite_seal),
        endpoints=endpoints,
    )


def run_once(
    root: str | Path = DEFAULT_PHYSICAL_ROOT,
    *,
    request_launch: Callable[..., Mapping[str, Any]] = physical_tournament.request_launch,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write one atomic gate + workflow reconciliation cycle."""

    resolved = Path(root).expanduser().resolve()
    lifecycle = resolved / "lifecycle"
    gate_path = lifecycle / GATE_FILENAME
    workflow_path = lifecycle / WORKFLOW_FILENAME
    operational_ascent_path = lifecycle / OPERATIONAL_ASCENT_FILENAME
    with _exclusive_lock(lifecycle / LOCK_FILENAME):
        previous = _previous_status(gate_path)
        gate = seal(build_gate_status(resolved, previous=previous))
        workflow = build_tournament_workflow(gate)
        operational_ascent = build_operational_ascent_status(gate)
        _atomic_json(gate_path, gate)
        _atomic_json(workflow_path, workflow)
        _atomic_json(operational_ascent_path, operational_ascent)
        _request_qualified_tournament_handoff(
            resolved, gate, request_launch=request_launch
        )
    return gate, workflow


def watch(root: str | Path = DEFAULT_PHYSICAL_ROOT, *, idle_seconds: float = 45.0) -> int:
    if idle_seconds <= 0:
        raise PhysicalGatekeeperError("idle_seconds must be positive")
    stopping = False

    def stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    old_term = signal.signal(signal.SIGTERM, stop)
    old_int = signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            run_once(root)
            if not stopping:
                time.sleep(idle_seconds)
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_PHYSICAL_ROOT, help="physical campaign root")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("once", help="reconcile one fail-closed gate cycle")
    watcher = subparsers.add_parser("watch", help="run detached reconciliation cycles")
    watcher.add_argument("--idle-seconds", type=float, default=45.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "watch":
        return watch(args.root, idle_seconds=args.idle_seconds)
    gate, workflow = run_once(args.root)
    print(
        json.dumps(
            {
                "gate_path": str(Path(args.root).expanduser().resolve() / "lifecycle" / GATE_FILENAME),
                "workflow_path": str(Path(args.root).expanduser().resolve() / "lifecycle" / WORKFLOW_FILENAME),
                "status": gate["status"],
                "gate_seal_sha256": gate["seal_sha256"],
                "workflow_seal_sha256": workflow["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ARTIFACT_ADMISSION_SCHEMA",
    "ARTIFACT_ADMISSION_STATUS",
    "CAPABILITY_SCHEMA",
    "DEFAULT_PHYSICAL_ROOT",
    "FINAL_REVIEW_FILENAME",
    "FINAL_REVIEW_SCHEMA",
    "GATE_FILENAME",
    "GATE_SCHEMA",
    "HCLI_SCHEMA",
    "KERNEL_SCHEMA",
    "MANAGER_OPERATIONS_SCHEMA",
    "MANAGER_OPERATIONS_STATUS",
    "MODEL_SPECS",
    "OPERATIONAL_ASCENT_EARNED",
    "OPERATIONAL_ASCENT_FILENAME",
    "OPERATIONAL_ASCENT_SCHEMA",
    "OPERATIONAL_ASCENT_WAITING",
    "RUNTIME_SCHEMA",
    "RUNTIME_SUPERSESSION_SCHEMA",
    "RUNTIME_PASS_STATUS",
    "TG3_SCHEMA",
    "TG10_SCHEMA",
    "TG10_STATUS",
    "WORKFLOW_FILENAME",
    "WORKFLOW_SCHEMA",
    "build_gate_status",
    "build_operational_ascent_status",
    "build_tournament_workflow",
    "main",
    "runtime_receipt_supersession_state",
    "run_once",
    "watch",
]


if __name__ == "__main__":
    raise SystemExit(main())
