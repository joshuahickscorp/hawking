"""Build the first complete, physical Qwen30 Gravity candidate.

This is intentionally a *candidate* compiler, not a manager promotion path.
It reads every tensor from the sealed local Qwen3-Coder-30B source and emits a
deterministic direct-layout binary pack for every tensor: 1-bit signs plus one
FP16 scale per 128 values.  The storage ledger includes headers and the final
manifest, so the ≤1.5 BPW result is a real physical-byte statement.  It does
not imply acceptable capability: this initial family is expected to be a low
fidelity baseline and is barred from runtime, TG, HCLI, and tournament gates
until those properties are measured with a native exact-model implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lab.receipts import seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct"
SOURCE_AUDIT = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen30/QWEN30_SOURCE_BODY_AUDIT_CANDIDATE.json"
ROOT = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen30/complete-gravity"
DEFAULT_REPOSITORY = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_MODEL_ID = "Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_ARTIFACT_PREFIX = "QWEN30"
MAGIC = b"HQ30G1B1"
VERSION = 1
GROUP_SIZE = 128
SCHEMA = "hawking.ascension.qwen30_complete_binary_gravity.v1"
SOURCE_REVALIDATION_SCHEMA = "hawking.ascension.complete_binary_source_revalidation.v1"
PROGRESS_INDEX_SCHEMA = "hawking.ascension.complete_binary_progress_index.v1"
TERMINAL_STATUS_SCHEMA = "hawking.ascension.complete_binary_terminal_status.v1"
COMPLETE_MANIFEST_STATUS = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED"
COMPLETE_CANDIDATE_PHASE = "EARNED_COMPLETE_PHYSICAL_BINARY_CANDIDATE_UNQUALIFIED"


class CompleteGravityError(RuntimeError):
    """An exact source or physical artifact invariant failed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


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
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024**2) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    """Hash a small, deterministic binding record without filesystem I/O."""

    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _file_identity(path: Path, *, label: str) -> dict[str, int]:
    """Return the cheap mutation-detection identity used by a revalidation receipt.

    The receipt contains an expensive SHA-256 of every source shard.  Subsequent
    bounded compiler invocations deliberately use this identity instead of
    rereading tens of gigabytes.  Any identity change causes a fresh full
    revalidation rather than allowing an old receipt to admit changed bytes.
    """

    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise CompleteGravityError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(observed.st_mode):
        raise CompleteGravityError(f"{label} must be a regular source file, not a symlink: {path}")
    if not stat.S_ISREG(observed.st_mode):
        raise CompleteGravityError(f"{label} must be a regular source file: {path}")
    return {
        "bytes": int(observed.st_size),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "mtime_ns": int(observed.st_mtime_ns),
        "ctime_ns": int(observed.st_ctime_ns),
    }


def _artifact_name(tensor_name: str) -> str:
    return hashlib.sha256(tensor_name.encode("utf-8")).hexdigest() + ".hq30g"


def _tensor_count(shape: Sequence[int]) -> int:
    return math.prod(int(dimension) for dimension in shape)


def _payload_bytes(shape: Sequence[int]) -> int:
    elements = _tensor_count(shape)
    groups = (elements + GROUP_SIZE - 1) // GROUP_SIZE
    # fixed header: magic/version/group/rank/reserved/elements + dimensions,
    # then FP16 scales and packed little-endian sign bits.
    # The writer pads each tensor through a whole sign/scale group before
    # packing bits.  Bill those retained tail bits, rather than only the
    # mathematical tensor tail, so the audited byte ledger exactly matches
    # the direct fixed-group layout for non-aligned tensors as well.
    return 32 + 4 * len(shape) + 2 * groups + (groups * GROUP_SIZE) // 8


def _values_from_raw(raw: bytes, dtype: str, shape: Sequence[int]) -> np.ndarray:
    normalized = dtype.upper()
    if normalized in {"BF16", "BFLOAT16"}:
        values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    elif normalized in {"F32", "FLOAT32"}:
        values = np.frombuffer(raw, dtype="<f4")
    elif normalized in {"F16", "FLOAT16"}:
        values = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    else:
        raise CompleteGravityError(f"unsupported safetensors dtype for binary candidate: {dtype}")
    expected = _tensor_count(shape)
    if values.size != expected:
        raise CompleteGravityError(f"tensor byte geometry mismatch: expected {expected} values, received {values.size}")
    return np.asarray(values, dtype=np.float32).reshape(tuple(int(item) for item in shape))


def _pack_binary(values: np.ndarray, shape: Sequence[int]) -> tuple[bytes, dict[str, Any], np.ndarray]:
    flat = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
    total = int(flat.size)
    groups = (total + GROUP_SIZE - 1) // GROUP_SIZE
    padded = np.pad(flat, (0, groups * GROUP_SIZE - total), constant_values=0.0).reshape(groups, GROUP_SIZE)
    if not np.isfinite(padded).all():
        raise CompleteGravityError("source tensor contains a non-finite value; refusing a lossy silent substitution")
    # Mean absolute value is the least-squares scalar for a fixed sign vector.
    scales = np.mean(np.abs(padded), axis=1, dtype=np.float64).astype("<f2")
    signs = np.packbits((padded >= 0.0).reshape(-1).astype(np.uint8), bitorder="little").tobytes()
    dimensions = tuple(int(item) for item in shape)
    header = struct.pack("<8sIIHHQI", MAGIC, VERSION, GROUP_SIZE, len(dimensions), 0, total, 0)
    header += struct.pack("<" + "I" * len(dimensions), *dimensions)
    payload = header + scales.tobytes() + signs
    expected = _payload_bytes(dimensions)
    if len(payload) != expected:
        raise CompleteGravityError(f"packed payload byte mismatch: got {len(payload)}, expected {expected}")
    reconstructed = (np.where(padded >= 0.0, 1.0, -1.0) * scales.astype(np.float32)[:, None]).reshape(-1)[:total]
    original_norm = max(float(np.linalg.norm(flat)), 1e-12)
    recon_norm = max(float(np.linalg.norm(reconstructed)), 1e-12)
    metrics = {
        "relative_l2": float(np.linalg.norm(flat - reconstructed) / original_norm),
        "cosine": float(np.dot(flat, reconstructed) / (original_norm * recon_norm)),
        "rmse": float(np.sqrt(np.mean(np.square(flat - reconstructed)))),
        "finite": True,
    }
    return payload, metrics, reconstructed


class CompleteBinaryGravity:
    def __init__(
        self, *, model_dir: Path, source_audit: Path, root: Path,
        repository: str = DEFAULT_REPOSITORY, model_id: str = DEFAULT_MODEL_ID,
        artifact_prefix: str = DEFAULT_ARTIFACT_PREFIX, schema: str = SCHEMA,
    ) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self.source_audit = source_audit.expanduser().resolve()
        self.root = root.expanduser().resolve()
        self.repository = repository
        self.model_id = model_id
        self.artifact_prefix = artifact_prefix
        self.schema = schema
        self.tensor_dir = self.root / "tensors"
        self.status_path = self.root / f"{artifact_prefix}_COMPLETE_GRAVITY_STATUS.json"
        self.progress_path = self.root / f"{artifact_prefix}_COMPLETE_GRAVITY_PROGRESS.jsonl"
        self.progress_index_path = self.root / f"{artifact_prefix}_COMPLETE_GRAVITY_PROGRESS_INDEX.json"
        self.source_revalidation_path = self.root / f"{artifact_prefix}_CURRENT_SOURCE_SHARD_REVALIDATION.json"
        self.manifest_path = self.root / f"{artifact_prefix}_COMPLETE_BINARY_GRAVITY_CANDIDATE.json"
        self.terminal_receipt_path = self.root / f"{artifact_prefix}_COMPLETE_GRAVITY_TERMINAL_RECEIPT.json"

    def _status_payload(self, phase: str, **fields: Any) -> dict[str, Any]:
        """Construct one mutable heartbeat without accidentally copying a prior seal."""

        prior = _read_json(self.status_path) or {}
        return {
            "schema": self.schema,
            "recorded_at": _utc_now(),
            "pid": os.getpid(),
            "heartbeat": int(prior.get("heartbeat", 0)) + 1,
            "phase": phase,
            "model": self.model_id,
            "representation": "direct_binary_sign_plus_fp16_group_scale",
            "claim_boundary": {
                "every_source_tensor_is_required_for_completion": True,
                "raw_bf16_source_is_authority_teacher_not_tournament_participant": True,
                "physical_bpw_is_ledgered_not_inferred": True,
                "candidate_is_expected_low_fidelity_until_capability_evidence": True,
                "not_native_runtime_or_tg_or_hcli_or_manager_qualification": True,
                "kv_state_and_context_bytes_are_separate_from_weight_bpw": True,
            },
            **fields,
        }

    def _publish(self, phase: str, **fields: Any) -> None:
        _atomic_json(self.status_path, self._status_payload(phase, **fields))

    def _publish_terminal(self, terminal: Mapping[str, Any], *, revalidated_now: bool) -> None:
        """Publish the only completion phase the admission watcher may consume.

        The long-lived status is resealed on every detached invocation, while
        the separate terminal receipt stays immutable.  That avoids a launchd
        KeepAlive restart briefly downgrading a complete candidate to PACKING
        and avoids resealing the complete manifest merely to refresh a
        heartbeat.
        """

        binding = terminal.get("binding")
        candidate = terminal.get("candidate")
        if not isinstance(binding, Mapping) or not isinstance(candidate, Mapping):
            raise CompleteGravityError("terminal receipt has no usable binding/candidate payload")
        progress = binding.get("progress")
        if not isinstance(progress, Mapping):
            raise CompleteGravityError("terminal receipt has no usable progress payload")
        payload = self._status_payload(
            COMPLETE_CANDIDATE_PHASE,
            manifest_path=candidate.get("manifest_path"),
            manifest_seal_sha256=candidate.get("manifest_seal_sha256"),
            terminal_receipt_path=str(self.terminal_receipt_path),
            terminal_receipt_seal_sha256=terminal.get("seal_sha256"),
            terminal_receipt_schema=terminal.get("schema"),
            terminal_status_is_sealed=True,
            source_revalidation={
                "receipt_path": binding.get("source_revalidation_receipt_path"),
                "receipt_seal_sha256": binding.get("source_revalidation_receipt_seal_sha256"),
                "full_shards_revalidated_this_cycle": revalidated_now,
            },
            progress={
                "planned_tensors": progress.get("planned_tensors"),
                "completed_tensors": progress.get("completed_tensors"),
                "artifact_bytes": candidate.get("all_required_weight_artifact_bytes"),
                "complete_physical_bpw": candidate.get("complete_physical_bpw"),
                "progress_index_path": progress.get("progress_index_path"),
                "next_cursor": progress.get("next_cursor"),
                "next_source_shard": progress.get("next_source_shard"),
                "next_tensor_name": progress.get("next_tensor_name"),
            },
        )
        _atomic_json(self.status_path, seal(payload))

    def _admit_source(self) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, Any]]]:
        """Read the sealed audit and normalize its per-shard hash evidence.

        This method only admits the audit's declared source.  ``run`` performs
        the separate, full current-byte revalidation before it writes any new
        tensor candidate.
        """

        source = _read_json(self.source_audit)
        if source is None:
            raise CompleteGravityError(f"missing source audit: {self.source_audit}")
        verified = verify(source, label=str(self.source_audit))
        body = verified.get("source") if isinstance(verified.get("source"), Mapping) else {}
        if body.get("repository") != self.repository:
            raise CompleteGravityError(f"source audit does not bind required repository {self.repository}")
        index_path = self.model_dir / "model.safetensors.index.json"
        _file_identity(index_path, label="safetensors index")
        index = _read_json(index_path)
        weights = index.get("weight_map") if isinstance(index, Mapping) else None
        if not isinstance(weights, Mapping) or not weights:
            raise CompleteGravityError("safetensors index has no weight map")
        weight_map = {str(name): str(shard) for name, shard in weights.items()}
        # Earlier Qwen30 evidence stores shard hashes under source.shards;
        # Qwen80's full acquisition audit stores every inventory row under
        # files.  Normalize both sealed formats before admitting a shared
        # complete-artifact compiler.
        source_shards = body.get("shards") if isinstance(body.get("shards"), Mapping) else {}
        file_rows = verified.get("files") if isinstance(verified.get("files"), list) else []
        shard_evidence: dict[str, dict[str, Any]] = {
            str(key): {
                "sha256": str(value["sha256"]),
                "bytes": value.get("bytes"),
            }
            for key, value in source_shards.items()
            if isinstance(value, Mapping) and isinstance(value.get("sha256"), str)
        }
        for row in file_rows:
            if isinstance(row, Mapping) and isinstance(row.get("path"), str) and isinstance(row.get("sha256"), str):
                shard_evidence[str(row["path"])] = {
                    "sha256": str(row["sha256"]),
                    "bytes": row.get("bytes"),
                }
        for shard in sorted(set(weight_map.values())):
            evidence = shard_evidence.get(shard)
            if evidence is None:
                raise CompleteGravityError(f"source audit is missing hash evidence for {shard}")
            expected_hash = evidence.get("sha256")
            expected_bytes = evidence.get("bytes")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise CompleteGravityError(f"source audit has an invalid SHA-256 for {shard}")
            try:
                int(expected_hash, 16)
            except ValueError as exc:
                raise CompleteGravityError(f"source audit has a non-hex SHA-256 for {shard}") from exc
            if not isinstance(expected_bytes, int) or expected_bytes < 0:
                raise CompleteGravityError(f"source audit is missing byte evidence for {shard}")
            _file_identity(self.model_dir / shard, label=f"source shard {shard}")
        return verified, weight_map, shard_evidence

    @staticmethod
    def _sealed_shard_hashes(shard_evidence: Mapping[str, Mapping[str, Any]], weight_map: Mapping[str, str]) -> dict[str, str]:
        """Return exactly the audit-declared hashes used by the current index."""

        return {
            shard: str(shard_evidence[shard]["sha256"])
            for shard in sorted(set(weight_map.values()))
        }

    def _source_revalidation_binding(
        self,
        *,
        audit: Mapping[str, Any],
        weight_map: Mapping[str, str],
        shard_evidence: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Build the small immutable binding checked on every bounded restart."""

        body = audit.get("source") if isinstance(audit.get("source"), Mapping) else {}
        index_path = self.model_dir / "model.safetensors.index.json"
        sealed_hashes = self._sealed_shard_hashes(shard_evidence, weight_map)
        sealed_index_hash: str | None = None
        file_rows = audit.get("files") if isinstance(audit.get("files"), list) else []
        for row in file_rows:
            if isinstance(row, Mapping) and row.get("path") == index_path.name and isinstance(row.get("sha256"), str):
                sealed_index_hash = str(row["sha256"])
                break
        current_index_hash = _sha256_file(index_path)
        if sealed_index_hash is not None and current_index_hash != sealed_index_hash:
            raise CompleteGravityError(
                "safetensors index SHA-256 does not match the sealed source audit: "
                f"observed={current_index_hash} expected={sealed_index_hash}"
            )
        return {
            "source_audit_path": str(self.source_audit),
            "source_audit_document_sha256": _sha256_file(self.source_audit),
            "source_audit_seal_sha256": str(audit["seal_sha256"]),
            "source_repository": self.repository,
            "source_revision": body.get("revision"),
            "source_model_dir": str(self.model_dir),
            "index_path": str(index_path),
            # The index is small compared with the 30B/80B shard body.  Hash it
            # on restart so a changed map cannot reuse an old source receipt.
            "index_sha256": current_index_hash,
            "sealed_audit_index_sha256": sealed_index_hash,
            "weight_map_sha256": _canonical_sha256(dict(sorted(weight_map.items()))),
            "sealed_shard_hashes_sha256": _canonical_sha256(sealed_hashes),
            "sealed_shard_count": len(sealed_hashes),
        }

    def _receipt_matches_current_source(
        self,
        receipt: Mapping[str, Any],
        *,
        binding: Mapping[str, Any],
        shard_evidence: Mapping[str, Mapping[str, Any]],
        weight_map: Mapping[str, str],
    ) -> bool:
        """Use receipt + cheap file identities to decide whether hashes may be reused."""

        try:
            verified = verify(receipt, label=str(self.source_revalidation_path))
        except Exception as exc:
            raise CompleteGravityError(f"source revalidation receipt is not trustworthy: {exc}") from exc
        if verified.get("schema") != SOURCE_REVALIDATION_SCHEMA:
            return False
        if verified.get("status") != "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED":
            return False
        for key, expected in binding.items():
            if verified.get(key) != expected:
                return False
        expected_hashes = self._sealed_shard_hashes(shard_evidence, weight_map)
        receipt_shards = verified.get("shards")
        if not isinstance(receipt_shards, Mapping) or set(receipt_shards) != set(expected_hashes):
            return False
        for shard, expected_hash in expected_hashes.items():
            row = receipt_shards.get(shard)
            if not isinstance(row, Mapping):
                return False
            if row.get("expected_sha256") != expected_hash or row.get("observed_sha256") != expected_hash:
                return False
            if row.get("expected_bytes") != shard_evidence[shard].get("bytes"):
                return False
            try:
                current_identity = _file_identity(self.model_dir / shard, label=f"source shard {shard}")
            except CompleteGravityError:
                return False
            if row.get("file_identity") != current_identity:
                return False
        return True

    def _revalidate_current_source(
        self,
        *,
        audit: Mapping[str, Any],
        weight_map: Mapping[str, str],
        shard_evidence: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        """Hash every sealed source shard once, then reuse a durable receipt.

        The receipt is only reused while the audit, index, and inexpensive file
        identities all still bind to the exact source bytes that were hashed.
        A changed identity deliberately causes a full all-shard revalidation;
        the compiler never silently samples or reuses a stale hash.
        """

        binding = self._source_revalidation_binding(
            audit=audit,
            weight_map=weight_map,
            shard_evidence=shard_evidence,
        )
        existing = _read_json(self.source_revalidation_path)
        if existing is not None and self._receipt_matches_current_source(
            existing,
            binding=binding,
            shard_evidence=shard_evidence,
            weight_map=weight_map,
        ):
            return existing, False

        sealed_hashes = self._sealed_shard_hashes(shard_evidence, weight_map)
        self._publish(
            "REVALIDATING_CURRENT_SEALED_SOURCE_SHARDS",
            source_revalidation={
                "receipt_path": str(self.source_revalidation_path),
                "audit_seal_sha256": binding["source_audit_seal_sha256"],
                "planned_shards": len(sealed_hashes),
                "completed_shards": 0,
            },
        )
        rows: dict[str, dict[str, Any]] = {}
        observed_bytes = 0
        for ordinal, shard in enumerate(sorted(sealed_hashes), start=1):
            source_path = self.model_dir / shard
            before = _file_identity(source_path, label=f"source shard {shard}")
            expected_bytes = int(shard_evidence[shard]["bytes"])
            if before["bytes"] != expected_bytes:
                raise CompleteGravityError(
                    f"source shard byte mismatch before hashing {shard}: "
                    f"observed={before['bytes']} expected={expected_bytes}"
                )
            observed_hash = _sha256_file(source_path)
            after = _file_identity(source_path, label=f"source shard {shard}")
            if before != after:
                raise CompleteGravityError(f"source shard changed while being revalidated: {shard}")
            if observed_hash != sealed_hashes[shard]:
                raise CompleteGravityError(
                    f"source shard SHA-256 mismatch for {shard}: "
                    f"observed={observed_hash} expected={sealed_hashes[shard]}"
                )
            rows[shard] = {
                "expected_sha256": sealed_hashes[shard],
                "observed_sha256": observed_hash,
                "expected_bytes": expected_bytes,
                "file_identity": after,
            }
            observed_bytes += before["bytes"]
            if ordinal == 1 or ordinal == len(sealed_hashes) or ordinal % 4 == 0:
                self._publish(
                    "REVALIDATING_CURRENT_SEALED_SOURCE_SHARDS",
                    source_revalidation={
                        "receipt_path": str(self.source_revalidation_path),
                        "audit_seal_sha256": binding["source_audit_seal_sha256"],
                        "planned_shards": len(sealed_hashes),
                        "completed_shards": ordinal,
                        "observed_bytes": observed_bytes,
                        "current_shard": shard,
                    },
                )
        receipt = seal(
            {
                "schema": SOURCE_REVALIDATION_SCHEMA,
                "status": "EARNED_CURRENT_SOURCE_SHARDS_REVALIDATED",
                "recorded_at": _utc_now(),
                **binding,
                "shards": rows,
                "observed_total_bytes": observed_bytes,
                "claim_boundary": {
                    "every_index_referenced_source_shard_was_full_sha256_revalidated": True,
                    "receipt_is_reused_only_while_audit_index_and_file_identity_match": True,
                    "revalidation_is_source_integrity_not_model_capability_or_runtime_evidence": True,
                },
            }
        )
        _atomic_json(self.source_revalidation_path, receipt)
        return receipt, True

    @staticmethod
    def _header(path: Path) -> dict[str, Any]:
        with path.open("rb") as handle:
            size_bytes = handle.read(8)
            if len(size_bytes) != 8:
                raise CompleteGravityError(f"invalid safetensors header prefix: {path}")
            size = struct.unpack("<Q", size_bytes)[0]
            raw = handle.read(size)
        decoded = json.loads(raw)
        return dict(decoded) if isinstance(decoded, Mapping) else {}

    @staticmethod
    def _progress_entry(row: Mapping[str, Any]) -> dict[str, Any]:
        """Keep only resume fields in the compact index; JSONL remains authority."""

        name = row.get("tensor_name")
        if not isinstance(name, str) or not name:
            raise CompleteGravityError("progress row has no tensor_name")
        try:
            artifact_bytes = int(row.get("artifact_bytes"))
        except (TypeError, ValueError) as exc:
            raise CompleteGravityError(f"progress row has invalid artifact_bytes for {name}") from exc
        if artifact_bytes < 0:
            raise CompleteGravityError(f"progress row has negative artifact_bytes for {name}")
        return {
            "artifact_path": str(row.get("artifact_path", "")),
            "artifact_bytes": artifact_bytes,
            "artifact_sha256": str(row.get("artifact_sha256", "")),
            "source_shard": str(row.get("source_shard", "")),
            "source_shard_sha256": str(row.get("source_shard_sha256", "")),
        }

    def _progress_binding(self, *, audit_seal: str, revalidation_seal: str) -> dict[str, Any]:
        return {
            "artifact_prefix": self.artifact_prefix,
            "progress_journal_path": str(self.progress_path),
            "source_audit_seal_sha256": audit_seal,
            "source_revalidation_receipt_seal_sha256": revalidation_seal,
        }

    @staticmethod
    def _planned_tensor_order(weight_map: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
        """Return the fixed pack order without opening any source shard header."""

        return tuple(
            (shard, tensor_name)
            for shard in sorted(set(weight_map.values()))
            for tensor_name in sorted(
                name for name, target_shard in weight_map.items() if target_shard == shard
            )
        )

    @staticmethod
    def _progress_row_binds_source(
        row: Mapping[str, Any] | None, *, shard: str, source_hash: str
    ) -> bool:
        return (
            isinstance(row, Mapping)
            and row.get("source_shard") == shard
            and row.get("source_shard_sha256") == source_hash
        )

    @staticmethod
    def _scheduler_record(
        *,
        planned_order: Sequence[tuple[str, str]],
        next_cursor: int,
        source_bound_completed_tensors: int,
    ) -> dict[str, Any]:
        if next_cursor < 0 or next_cursor > len(planned_order):
            raise CompleteGravityError(f"invalid next cursor {next_cursor}")
        if source_bound_completed_tensors < 0 or source_bound_completed_tensors > len(planned_order):
            raise CompleteGravityError(
                f"invalid source-bound completed count {source_bound_completed_tensors}"
            )
        next_shard: str | None = None
        next_tensor: str | None = None
        if next_cursor < len(planned_order):
            next_shard, next_tensor = planned_order[next_cursor]
        return {
            "planned_order_sha256": _canonical_sha256(list(planned_order)),
            "planned_tensor_count": len(planned_order),
            "next_cursor": next_cursor,
            "next_source_shard": next_shard,
            "next_tensor_name": next_tensor,
            "source_bound_completed_tensors": source_bound_completed_tensors,
        }

    def _derive_scheduler(
        self,
        *,
        completed: Mapping[str, Mapping[str, Any]],
        planned_order: Sequence[tuple[str, str]],
        shard_evidence: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Recover a cursor from journal/index rows without opening source headers."""

        next_cursor = 0
        source_bound_completed_tensors = 0
        for position, (shard, tensor_name) in enumerate(planned_order):
            is_complete = self._progress_row_binds_source(
                completed.get(tensor_name),
                shard=shard,
                source_hash=str(shard_evidence[shard]["sha256"]),
            )
            if is_complete:
                source_bound_completed_tensors += 1
                if position == next_cursor:
                    next_cursor += 1
        return self._scheduler_record(
            planned_order=planned_order,
            next_cursor=next_cursor,
            source_bound_completed_tensors=source_bound_completed_tensors,
        )

    def _advance_scheduler(
        self,
        *,
        scheduler: Mapping[str, Any],
        completed: Mapping[str, Mapping[str, Any]],
        planned_order: Sequence[tuple[str, str]],
        shard_evidence: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Move forward from the durable cursor after a journal row is fsync'd."""

        try:
            next_cursor = int(scheduler["next_cursor"])
            completed_count = int(scheduler["source_bound_completed_tensors"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CompleteGravityError("progress scheduler has no usable cursor") from exc
        # A new row is only appended at an unfinished cursor, so it transitions
        # exactly one source-bound tensor.  The loop may additionally skip
        # already-journaled tail rows recovered after a prior interrupted batch.
        completed_count += 1
        while next_cursor < len(planned_order):
            shard, tensor_name = planned_order[next_cursor]
            if not self._progress_row_binds_source(
                completed.get(tensor_name),
                shard=shard,
                source_hash=str(shard_evidence[shard]["sha256"]),
            ):
                break
            next_cursor += 1
        return self._scheduler_record(
            planned_order=planned_order,
            next_cursor=next_cursor,
            source_bound_completed_tensors=completed_count,
        )

    def _scheduler_from_index(
        self,
        *,
        raw_scheduler: Any,
        completed: Mapping[str, Mapping[str, Any]],
        planned_order: Sequence[tuple[str, str]],
        shard_evidence: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        """Return a valid persisted cursor, migrating legacy indexes in-place."""

        expected_digest = _canonical_sha256(list(planned_order))
        if isinstance(raw_scheduler, Mapping):
            try:
                next_cursor = int(raw_scheduler["next_cursor"])
                completed_count = int(raw_scheduler["source_bound_completed_tensors"])
            except (KeyError, TypeError, ValueError):
                next_cursor = -1
                completed_count = -1
            expected = self._scheduler_record(
                planned_order=planned_order,
                next_cursor=next_cursor,
                source_bound_completed_tensors=completed_count,
            ) if 0 <= next_cursor <= len(planned_order) and 0 <= completed_count <= len(planned_order) else None
            if (
                expected is not None
                and raw_scheduler.get("planned_order_sha256") == expected_digest
                and dict(raw_scheduler) == expected
            ):
                return expected, False
        return self._derive_scheduler(
            completed=completed,
            planned_order=planned_order,
            shard_evidence=shard_evidence,
        ), True

    def _read_progress_tail(self, *, offset: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Read only new JSONL rows after a previously sealed byte offset."""

        if not self.progress_path.exists():
            if offset != 0:
                raise CompleteGravityError("progress journal disappeared after its index was written")
            return {}, {
                "device": None,
                "inode": None,
                "indexed_bytes": 0,
                "indexed_rows": 0,
                "last_row_offset": None,
                "last_row_bytes": 0,
                "last_row_sha256": None,
            }
        identity = _file_identity(self.progress_path, label="complete Gravity progress journal")
        if offset < 0 or offset > identity["bytes"]:
            raise CompleteGravityError(
                f"progress index offset {offset} is outside journal length {identity['bytes']}"
            )
        with self.progress_path.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read()
        if payload and not payload.endswith(b"\n"):
            raise CompleteGravityError("progress journal ends in a partial row; refusing to append across it")
        rows: dict[str, dict[str, Any]] = {}
        cursor = offset
        row_count = 0
        last_row_offset: int | None = None
        last_row_bytes = 0
        last_row_sha256: str | None = None
        for line in payload.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                raise CompleteGravityError("progress journal contains a non-terminated row")
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CompleteGravityError(f"progress journal contains invalid JSON at byte {cursor}") from exc
            if not isinstance(decoded, Mapping):
                raise CompleteGravityError(f"progress journal row at byte {cursor} is not an object")
            entry = self._progress_entry(decoded)
            name = str(decoded["tensor_name"])
            rows[name] = entry
            last_row_offset = cursor
            last_row_bytes = len(line)
            last_row_sha256 = hashlib.sha256(line).hexdigest()
            cursor += len(line)
            row_count += 1
        return rows, {
            "device": identity["device"],
            "inode": identity["inode"],
            "indexed_bytes": cursor,
            "indexed_rows": row_count,
            "last_row_offset": last_row_offset,
            "last_row_bytes": last_row_bytes,
            "last_row_sha256": last_row_sha256,
        }

    @staticmethod
    def _same_journal_file(journal: Mapping[str, Any], current: Mapping[str, int]) -> bool:
        return journal.get("device") == current["device"] and journal.get("inode") == current["inode"]

    def _index_tip_matches_journal(self, journal: Mapping[str, Any], current: Mapping[str, int]) -> bool:
        """Validate the indexed last row with one tiny seek, not a whole reparse."""

        indexed_bytes = journal.get("indexed_bytes")
        last_offset = journal.get("last_row_offset")
        last_bytes = journal.get("last_row_bytes")
        last_hash = journal.get("last_row_sha256")
        if not isinstance(indexed_bytes, int) or indexed_bytes < 0 or indexed_bytes > current["bytes"]:
            return False
        if indexed_bytes == 0:
            return last_offset is None and last_bytes == 0 and last_hash is None
        if (
            not isinstance(last_offset, int)
            or not isinstance(last_bytes, int)
            or last_offset < 0
            or last_bytes <= 0
            or last_offset + last_bytes != indexed_bytes
            or not isinstance(last_hash, str)
            or len(last_hash) != 64
        ):
            return False
        with self.progress_path.open("rb") as handle:
            handle.seek(last_offset)
            line = handle.read(last_bytes)
        return len(line) == last_bytes and hashlib.sha256(line).hexdigest() == last_hash

    def _write_progress_index(
        self,
        *,
        completed: Mapping[str, Mapping[str, Any]],
        journal: Mapping[str, Any],
        binding: Mapping[str, Any],
        scheduler: Mapping[str, Any],
    ) -> dict[str, Any]:
        compact = {name: dict(completed[name]) for name in sorted(completed)}
        payload = seal(
            {
                "schema": PROGRESS_INDEX_SCHEMA,
                "recorded_at": _utc_now(),
                **binding,
                "completed": compact,
                "completed_tensor_count": len(compact),
                "indexed_artifact_bytes": sum(int(entry["artifact_bytes"]) for entry in compact.values()),
                "journal": dict(journal),
                "scheduler": dict(scheduler),
                "claim_boundary": {
                    "jsonl_progress_journal_remains_authoritative": True,
                    "index_is_a_restart_accelerator_not_a_completion_claim": True,
                    "journal_tail_is_reconciled_before_reuse": True,
                    "normal_batches_open_source_headers_only_at_or_after_next_cursor": True,
                },
            }
        )
        _atomic_json(self.progress_index_path, payload)
        return payload

    def _rebuild_progress_index(
        self,
        *,
        binding: Mapping[str, Any],
        planned_order: Sequence[tuple[str, str]],
        shard_evidence: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """One-time legacy/corruption recovery path; later starts only read a tail."""

        completed, journal = self._read_progress_tail(offset=0)
        scheduler = self._derive_scheduler(
            completed=completed,
            planned_order=planned_order,
            shard_evidence=shard_evidence,
        )
        self._write_progress_index(
            completed=completed,
            journal=journal,
            binding=binding,
            scheduler=scheduler,
        )
        return completed, journal, scheduler

    def _load_progress_index(
        self,
        *,
        binding: Mapping[str, Any],
        planned_order: Sequence[tuple[str, str]],
        shard_evidence: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """Load the compact index, parsing the JSONL only on first use or a tail."""

        indexed = _read_json(self.progress_index_path)
        if indexed is None:
            return self._rebuild_progress_index(
                binding=binding,
                planned_order=planned_order,
                shard_evidence=shard_evidence,
            )
        try:
            verified = verify(indexed, label=str(self.progress_index_path))
        except Exception as exc:
            raise CompleteGravityError(f"progress index is not trustworthy: {exc}") from exc
        if verified.get("schema") != PROGRESS_INDEX_SCHEMA:
            return self._rebuild_progress_index(
                binding=binding,
                planned_order=planned_order,
                shard_evidence=shard_evidence,
            )
        if any(verified.get(key) != value for key, value in binding.items()):
            return self._rebuild_progress_index(
                binding=binding,
                planned_order=planned_order,
                shard_evidence=shard_evidence,
            )
        raw_completed = verified.get("completed")
        journal = verified.get("journal")
        if not isinstance(raw_completed, Mapping) or not isinstance(journal, Mapping):
            return self._rebuild_progress_index(
                binding=binding,
                planned_order=planned_order,
                shard_evidence=shard_evidence,
            )
        try:
            completed = {
                str(name): self._progress_entry({"tensor_name": str(name), **dict(entry)})
                for name, entry in raw_completed.items()
                if isinstance(name, str) and isinstance(entry, Mapping)
            }
        except (TypeError, CompleteGravityError):
            return self._rebuild_progress_index(
                binding=binding,
                planned_order=planned_order,
                shard_evidence=shard_evidence,
            )
        if len(completed) != len(raw_completed):
            return self._rebuild_progress_index(
                binding=binding,
                planned_order=planned_order,
                shard_evidence=shard_evidence,
            )
        scheduler, scheduler_migrated = self._scheduler_from_index(
            raw_scheduler=verified.get("scheduler"),
            completed=completed,
            planned_order=planned_order,
            shard_evidence=shard_evidence,
        )
        if not self.progress_path.exists():
            if completed:
                return self._rebuild_progress_index(
                    binding=binding,
                    planned_order=planned_order,
                    shard_evidence=shard_evidence,
                )
            empty_journal = {
                "device": None,
                "inode": None,
                "indexed_bytes": 0,
                "indexed_rows": 0,
                "last_row_offset": None,
                "last_row_bytes": 0,
                "last_row_sha256": None,
            }
            if scheduler_migrated:
                self._write_progress_index(
                    completed=completed,
                    journal=empty_journal,
                    binding=binding,
                    scheduler=scheduler,
                )
            return completed, empty_journal, scheduler
        current = _file_identity(self.progress_path, label="complete Gravity progress journal")
        if not self._same_journal_file(journal, current) or not self._index_tip_matches_journal(journal, current):
            return self._rebuild_progress_index(
                binding=binding,
                planned_order=planned_order,
                shard_evidence=shard_evidence,
            )
        tail, observed_tail = self._read_progress_tail(offset=int(journal["indexed_bytes"]))
        if observed_tail["indexed_rows"] == 0:
            if scheduler_migrated:
                self._write_progress_index(
                    completed=completed,
                    journal=journal,
                    binding=binding,
                    scheduler=scheduler,
                )
            return completed, dict(journal), scheduler
        completed.update(tail)
        merged_journal = {
            "device": observed_tail["device"],
            "inode": observed_tail["inode"],
            "indexed_bytes": observed_tail["indexed_bytes"],
            "indexed_rows": int(journal.get("indexed_rows", 0)) + observed_tail["indexed_rows"],
            "last_row_offset": observed_tail["last_row_offset"],
            "last_row_bytes": observed_tail["last_row_bytes"],
            "last_row_sha256": observed_tail["last_row_sha256"],
        }
        scheduler = self._derive_scheduler(
            completed=completed,
            planned_order=planned_order,
            shard_evidence=shard_evidence,
        )
        self._write_progress_index(
            completed=completed,
            journal=merged_journal,
            binding=binding,
            scheduler=scheduler,
        )
        return completed, merged_journal, scheduler

    def _append_progress(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Durably append one complete JSONL row and return its exact byte location."""

        self._progress_entry(row)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        if self.progress_path.exists() and self.progress_path.stat().st_size:
            with self.progress_path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    raise CompleteGravityError("progress journal has a partial final row; refusing unsafe append")
        payload = (json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        with self.progress_path.open("ab") as handle:
            offset = handle.tell()
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(self.progress_path, 0o640)
        identity = _file_identity(self.progress_path, label="complete Gravity progress journal")
        if identity["bytes"] != offset + len(payload):
            raise CompleteGravityError("progress journal changed during append; refusing to advance its index")
        return {
            "offset": offset,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "identity": identity,
        }

    def _advance_progress_index(
        self,
        *,
        completed: dict[str, dict[str, Any]],
        journal: Mapping[str, Any],
        row: Mapping[str, Any],
        appended: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Advance in-memory index state after the journal row is fsync'd."""

        offset = int(appended["offset"])
        if offset != int(journal.get("indexed_bytes", 0)):
            raise CompleteGravityError("progress journal advanced outside this compiler invocation")
        name = str(row["tensor_name"])
        completed[name] = self._progress_entry(row)
        identity = appended["identity"]
        return {
            "device": identity["device"],
            "inode": identity["inode"],
            "indexed_bytes": offset + int(appended["bytes"]),
            "indexed_rows": int(journal.get("indexed_rows", 0)) + 1,
            "last_row_offset": offset,
            "last_row_bytes": int(appended["bytes"]),
            "last_row_sha256": str(appended["sha256"]),
        }

    def _progress_row_is_usable(
        self,
        *,
        tensor_name: str,
        shard: str,
        source_hash: str,
        row: Mapping[str, Any],
    ) -> bool:
        if row.get("source_shard") != shard or row.get("source_shard_sha256") != source_hash:
            return False
        try:
            artifact_bytes = int(row.get("artifact_bytes", -1))
        except (TypeError, ValueError):
            return False
        artifact = Path(str(row.get("artifact_path", "")))
        expected = self.tensor_dir / _artifact_name(tensor_name)
        try:
            if artifact.resolve() != expected.resolve():
                return False
        except OSError:
            return False
        return artifact.is_file() and artifact.stat().st_size == artifact_bytes

    def _full_progress_rows(self) -> dict[str, dict[str, Any]]:
        """Read complete authority rows only when the final manifest needs them."""

        rows: dict[str, dict[str, Any]] = {}
        if not self.progress_path.exists():
            return rows
        with self.progress_path.open("rb") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.endswith(b"\n"):
                    raise CompleteGravityError(f"progress journal row {number} is not newline terminated")
                try:
                    decoded = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CompleteGravityError(f"progress journal row {number} is invalid JSON") from exc
                if not isinstance(decoded, Mapping):
                    raise CompleteGravityError(f"progress journal row {number} is not an object")
                self._progress_entry(decoded)
                rows[str(decoded["tensor_name"])] = dict(decoded)
        return rows

    @staticmethod
    def _is_sha256(value: object) -> bool:
        if not isinstance(value, str) or len(value) != 64:
            return False
        try:
            int(value, 16)
        except ValueError:
            return False
        return True

    def _source_catalog_sha256(
        self,
        *,
        weight_map: Mapping[str, str],
        shard_evidence: Mapping[str, Mapping[str, Any]],
    ) -> str:
        """Bind every required tensor to the sealed source-shard identity."""

        return _canonical_sha256(
            [
                {
                    "tensor_name": tensor_name,
                    "source_shard": weight_map[tensor_name],
                    "source_shard_sha256": str(shard_evidence[weight_map[tensor_name]]["sha256"]),
                }
                for tensor_name in sorted(weight_map)
            ]
        )

    def _complete_progress_catalog_sha256(
        self,
        *,
        progress: Mapping[str, Mapping[str, Any]],
        weight_map: Mapping[str, str],
        shard_evidence: Mapping[str, Mapping[str, Any]],
    ) -> str:
        """Require the compact progress index to cover the exact full source catalog."""

        expected_names = set(weight_map)
        if set(progress) != expected_names:
            raise CompleteGravityError(
                "complete progress catalog does not contain exactly the source tensor names"
            )
        catalog: list[dict[str, Any]] = []
        for tensor_name in sorted(weight_map):
            shard = weight_map[tensor_name]
            source_hash = str(shard_evidence[shard]["sha256"])
            row = progress.get(tensor_name)
            if not isinstance(row, Mapping) or not self._progress_row_binds_source(
                row, shard=shard, source_hash=source_hash
            ):
                raise CompleteGravityError(
                    f"complete progress catalog is not source-bound for {tensor_name}"
                )
            entry = self._progress_entry({"tensor_name": tensor_name, **dict(row)})
            expected_path = str(self.tensor_dir / _artifact_name(tensor_name))
            if (
                entry["artifact_path"] != expected_path
                or entry["artifact_bytes"] <= 0
                or not self._is_sha256(entry["artifact_sha256"])
            ):
                raise CompleteGravityError(
                    f"complete progress catalog has an invalid artifact binding for {tensor_name}"
                )
            catalog.append({"tensor_name": tensor_name, **entry})
        return _canonical_sha256(catalog)

    def _terminal_progress(self, *, planned: int, scheduler: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize the exact all-tensors-complete scheduler state for a terminal seal."""

        try:
            next_cursor = int(scheduler["next_cursor"])
            completed = int(scheduler["source_bound_completed_tensors"])
            planned_order_sha256 = str(scheduler["planned_order_sha256"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CompleteGravityError("terminal scheduler is incomplete") from exc
        if (
            next_cursor != planned
            or completed != planned
            or scheduler.get("next_source_shard") is not None
            or scheduler.get("next_tensor_name") is not None
            or not self._is_sha256(planned_order_sha256)
        ):
            raise CompleteGravityError("terminal scheduler is not an exact completed cursor")
        return {
            "planned_tensors": planned,
            "completed_tensors": completed,
            "next_cursor": next_cursor,
            "next_source_shard": None,
            "next_tensor_name": None,
            "planned_order_sha256": planned_order_sha256,
            "progress_index_path": str(self.progress_index_path),
        }

    def _current_manifest_file_binding(self) -> dict[str, Any]:
        """Hash a complete manifest once while proving it did not change during the read."""

        before = _file_identity(self.manifest_path, label="complete binary manifest")
        document_sha256 = _sha256_file(self.manifest_path)
        after = _file_identity(self.manifest_path, label="complete binary manifest")
        if before != after:
            raise CompleteGravityError("complete binary manifest changed while it was being bound")
        return {
            "manifest_path": str(self.manifest_path),
            "manifest_document_sha256": document_sha256,
            "manifest_file_identity": after,
        }

    def _admit_existing_complete_manifest(
        self,
        *,
        audit: Mapping[str, Any],
        revalidation: Mapping[str, Any],
        weight_map: Mapping[str, str],
        shard_evidence: Mapping[str, Mapping[str, Any]],
        progress: Mapping[str, Mapping[str, Any]],
        source_catalog_sha256: str,
        progress_catalog_sha256: str,
    ) -> dict[str, Any] | None:
        """Reuse only a sealed manifest that still matches source, progress, and catalog.

        This is deliberately stricter than merely finding a JSON file: it
        checks the immutable source bindings, every catalog row, progress-artifact
        correspondence, and the physical ledger before allowing a completed
        candidate to retain its original manifest seal across launchd restarts.
        """

        raw = _read_json(self.manifest_path)
        if raw is None:
            return None
        try:
            manifest = verify(raw, label=str(self.manifest_path))
        except Exception:
            return None
        if (
            manifest.get("schema") != self.schema
            or manifest.get("status") != COMPLETE_MANIFEST_STATUS
            or manifest.get("source_body_audit_seal_sha256") != audit.get("seal_sha256")
            or manifest.get("source_revalidation_receipt_path") != str(self.source_revalidation_path)
            or manifest.get("source_revalidation_receipt_seal_sha256") != revalidation.get("seal_sha256")
        ):
            return None
        source = manifest.get("source")
        if not isinstance(source, Mapping) or (
            source.get("repository") != self.repository
            or source.get("model_dir") != str(self.model_dir)
            or source.get("tensor_count") != len(weight_map)
        ):
            return None
        tensors = manifest.get("tensors")
        if not isinstance(tensors, list) or len(tensors) != len(weight_map):
            return None
        catalog: list[dict[str, Any]] = []
        seen: set[str] = set()
        tensor_payload_bytes = 0
        source_elements = 0
        for row in tensors:
            if not isinstance(row, Mapping):
                return None
            tensor_name = row.get("tensor_name")
            if not isinstance(tensor_name, str) or tensor_name in seen or tensor_name not in weight_map:
                return None
            seen.add(tensor_name)
            shard = weight_map[tensor_name]
            source_hash = str(shard_evidence[shard]["sha256"])
            progress_row = progress.get(tensor_name)
            if not isinstance(progress_row, Mapping) or not self._progress_row_binds_source(
                progress_row, shard=shard, source_hash=source_hash
            ):
                return None
            try:
                row_artifact_bytes = int(row.get("artifact_bytes"))
                row_elements = int(row.get("elements"))
            except (TypeError, ValueError):
                return None
            if (
                row_artifact_bytes <= 0
                or row_elements <= 0
                or row.get("source_shard") != shard
                or row.get("source_shard_sha256") != source_hash
                or row.get("artifact_path") != progress_row.get("artifact_path")
                or row_artifact_bytes != progress_row.get("artifact_bytes")
                or row.get("artifact_sha256") != progress_row.get("artifact_sha256")
                or not self._is_sha256(row.get("artifact_sha256"))
                or not self._progress_row_is_usable(
                    tensor_name=tensor_name,
                    shard=shard,
                    source_hash=source_hash,
                    row=row,
                )
            ):
                return None
            tensor_payload_bytes += row_artifact_bytes
            source_elements += row_elements
            catalog.append(
                {
                    "tensor_name": tensor_name,
                    "source_shard": shard,
                    "source_shard_sha256": source_hash,
                    "artifact_path": str(row["artifact_path"]),
                    "artifact_bytes": row_artifact_bytes,
                    "artifact_sha256": str(row["artifact_sha256"]),
                    "elements": row_elements,
                }
            )
        if seen != set(weight_map):
            return None
        ledger = manifest.get("complete_physical_bpw_ledger")
        if not isinstance(ledger, Mapping):
            return None
        try:
            ledger_elements = int(ledger["source_weight_elements"])
            ledger_payload_bytes = int(ledger["tensor_payload_bytes"])
            ledger_manifest_bytes = int(ledger["manifest_bytes_billed"])
            ledger_total_bytes = int(ledger["all_required_weight_artifact_bytes"])
            ledger_bpw = float(ledger["complete_physical_bpw"])
            threshold_bpw = float(ledger["threshold_bpw"])
        except (KeyError, TypeError, ValueError):
            return None
        if (
            isinstance(ledger.get("source_weight_elements"), bool)
            or isinstance(ledger.get("tensor_payload_bytes"), bool)
            or isinstance(ledger.get("manifest_bytes_billed"), bool)
            or isinstance(ledger.get("all_required_weight_artifact_bytes"), bool)
            or source_elements <= 0
            or tensor_payload_bytes <= 0
            or ledger_elements != source_elements
            or ledger_payload_bytes != tensor_payload_bytes
            or ledger_manifest_bytes < 0
            or ledger_total_bytes != tensor_payload_bytes + ledger_manifest_bytes
            or not math.isfinite(ledger_bpw)
            or not math.isfinite(threshold_bpw)
            or threshold_bpw != 1.5
            or not math.isclose(
                ledger_bpw,
                ledger_total_bytes * 8.0 / source_elements,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or ledger.get("passes_storage_threshold") is not (ledger_bpw <= threshold_bpw)
        ):
            return None
        binding = self._current_manifest_file_binding()
        if ledger_manifest_bytes != binding["manifest_file_identity"]["bytes"]:
            return None
        return {
            **binding,
            "manifest_seal_sha256": manifest["seal_sha256"],
            "manifest_catalog_sha256": _canonical_sha256(catalog),
            "source_catalog_sha256": source_catalog_sha256,
            "progress_catalog_sha256": progress_catalog_sha256,
            "all_required_weight_artifact_bytes": ledger_total_bytes,
            "complete_physical_bpw": ledger_bpw,
            "passes_storage_threshold": ledger_bpw <= threshold_bpw,
        }

    def _write_terminal_receipt(
        self,
        *,
        audit: Mapping[str, Any],
        revalidation: Mapping[str, Any],
        terminal_progress: Mapping[str, Any],
        manifest_binding: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Seal the stable completion binding after the manifest is fully admitted."""

        terminal = seal(
            {
                "schema": TERMINAL_STATUS_SCHEMA,
                "status": COMPLETE_CANDIDATE_PHASE,
                "recorded_at": _utc_now(),
                "binding": {
                    "model_id": self.model_id,
                    "artifact_prefix": self.artifact_prefix,
                    "manifest_schema": self.schema,
                    "source_body_audit_seal_sha256": audit["seal_sha256"],
                    "source_revalidation_receipt_path": str(self.source_revalidation_path),
                    "source_revalidation_receipt_seal_sha256": revalidation["seal_sha256"],
                    "source_catalog_sha256": manifest_binding["source_catalog_sha256"],
                    "progress_catalog_sha256": manifest_binding["progress_catalog_sha256"],
                    "manifest_catalog_sha256": manifest_binding["manifest_catalog_sha256"],
                    "progress": dict(terminal_progress),
                },
                "candidate": {
                    "manifest_path": manifest_binding["manifest_path"],
                    "manifest_seal_sha256": manifest_binding["manifest_seal_sha256"],
                    "manifest_document_sha256": manifest_binding["manifest_document_sha256"],
                    "manifest_file_identity": manifest_binding["manifest_file_identity"],
                    "all_required_weight_artifact_bytes": manifest_binding[
                        "all_required_weight_artifact_bytes"
                    ],
                    "complete_physical_bpw": manifest_binding["complete_physical_bpw"],
                    "passes_storage_threshold": manifest_binding["passes_storage_threshold"],
                },
                "claim_boundary": {
                    "completion_is_bound_to_current_source_revalidation_and_full_catalog": True,
                    "terminal_receipt_reuses_an_immutable_complete_manifest_not_a_new_reseal": True,
                    "candidate_remains_unqualified_for_native_runtime_capability_hcli_tps_tg_and_tournament": True,
                },
            }
        )
        _atomic_json(self.terminal_receipt_path, terminal)
        return terminal

    def _current_terminal_receipt(
        self,
        *,
        audit: Mapping[str, Any],
        revalidation: Mapping[str, Any],
        terminal_progress: Mapping[str, Any],
        source_catalog_sha256: str,
        progress_catalog_sha256: str,
    ) -> dict[str, Any] | None:
        """Return a current terminal seal without reopening/resealing the manifest."""

        raw = _read_json(self.terminal_receipt_path)
        if raw is None:
            return None
        try:
            terminal = verify(raw, label=str(self.terminal_receipt_path))
        except Exception:
            return None
        binding = terminal.get("binding")
        candidate = terminal.get("candidate")
        if not isinstance(binding, Mapping) or not isinstance(candidate, Mapping):
            return None
        if (
            terminal.get("schema") != TERMINAL_STATUS_SCHEMA
            or terminal.get("status") != COMPLETE_CANDIDATE_PHASE
            or binding.get("model_id") != self.model_id
            or binding.get("artifact_prefix") != self.artifact_prefix
            or binding.get("manifest_schema") != self.schema
            or binding.get("source_body_audit_seal_sha256") != audit.get("seal_sha256")
            or binding.get("source_revalidation_receipt_path") != str(self.source_revalidation_path)
            or binding.get("source_revalidation_receipt_seal_sha256") != revalidation.get("seal_sha256")
            or binding.get("source_catalog_sha256") != source_catalog_sha256
            or binding.get("progress_catalog_sha256") != progress_catalog_sha256
            or binding.get("progress") != dict(terminal_progress)
            or candidate.get("manifest_path") != str(self.manifest_path)
            or not self._is_sha256(candidate.get("manifest_seal_sha256"))
            or not self._is_sha256(candidate.get("manifest_document_sha256"))
            or not self._is_sha256(binding.get("manifest_catalog_sha256"))
        ):
            return None
        try:
            current_identity = _file_identity(self.manifest_path, label="complete binary manifest")
        except CompleteGravityError:
            return None
        if candidate.get("manifest_file_identity") != current_identity:
            return None
        try:
            artifact_bytes = int(candidate["all_required_weight_artifact_bytes"])
            complete_bpw = float(candidate["complete_physical_bpw"])
        except (KeyError, TypeError, ValueError):
            return None
        if artifact_bytes <= 0 or not math.isfinite(complete_bpw):
            return None
        return terminal

    def _write_tensor(self, *, tensor_name: str, shard: str, source_hash: str, info: Mapping[str, Any]) -> dict[str, Any]:
        dtype = str(info.get("dtype"))
        shape = [int(item) for item in info.get("shape", [])]
        offsets = info.get("data_offsets")
        if not shape or not isinstance(offsets, list) or len(offsets) != 2:
            raise CompleteGravityError(f"invalid source tensor metadata: {tensor_name}")
        begin, end = (int(item) for item in offsets)
        if begin < 0 or end < begin:
            raise CompleteGravityError(f"invalid source byte range: {tensor_name}")
        source_path = self.model_dir / shard
        with source_path.open("rb") as handle:
            header_bytes = struct.unpack("<Q", handle.read(8))[0]
            handle.seek(8 + header_bytes + begin)
            raw = handle.read(end - begin)
        values = _values_from_raw(raw, dtype, shape)
        payload, quality, _ = _pack_binary(values, shape)
        destination = self.tensor_dir / _artifact_name(tensor_name)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=self.tensor_dir)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o640)
            os.replace(temporary, destination)
            os.chmod(destination, 0o640)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        del values
        return {
            "tensor_name": tensor_name,
            "source_shard": shard,
            "source_shard_sha256": source_hash,
            "source_dtype": dtype,
            "shape": shape,
            "elements": _tensor_count(shape),
            "artifact_path": str(destination),
            "artifact_bytes": len(payload),
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
            "layout": {"magic": MAGIC.decode("ascii"), "version": VERSION, "group_size": GROUP_SIZE, "sign_bit_order": "little", "scale_dtype": "float16"},
            "component_quality": quality,
        }

    # The complete-binary compiler is also used as the audited transport for
    # bounded representation branches.  Keep these manifest hooks deliberately
    # narrow: the baseline implementation returns the historical payload
    # unchanged, while a branch may describe its own layout without teaching
    # the baseline manifest to mislabel bytes it cannot decode.
    def _manifest_representation(self) -> dict[str, Any]:
        return {
            "family": "binary_sign_scale",
            "group_size": GROUP_SIZE,
            "physical_direct_layout": True,
            "training": "none_for_this_baseline; trained low-rank component research remains separate",
        }

    def _manifest_champion_classes(self, *, complete_bpw: float) -> dict[str, Any]:
        return {
            "current_bpw_champion": {
                "candidate": f"{self.model_id}-complete-binary-baseline",
                "complete_physical_bpw": complete_bpw,
                "status": "CANDIDATE_ONLY",
            },
            "current_runtime_champion": {
                "candidate": None,
                "status": "BLOCKED_BY_EXACT_DEPENDENCY",
                "dependency": "QwenMoE full native decoder is unimplemented",
            },
            "current_capability_champion": {
                "candidate": None,
                "status": "BLOCKED_BY_EXACT_DEPENDENCY",
                "dependency": "complete prompt-dependent parity/capability suite not run",
            },
        }

    @staticmethod
    def _manifest_quality_summary(ordered: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "mean_component_cosine": float(np.mean([row["component_quality"]["cosine"] for row in ordered])),
            "mean_component_relative_l2": float(
                np.mean([row["component_quality"]["relative_l2"] for row in ordered])
            ),
            "verdict": "LOW_FIDELITY_BINARY_BASELINE_NOT_ELIGIBLE_FOR_RUNTIME_OR_CAPABILITY_PROMOTION",
        }

    @staticmethod
    def _manifest_claim_boundary() -> dict[str, Any]:
        return {
            "complete_physical_tensor_coverage_is_true": True,
            "complete_bpw_pass_does_not_substitute_for_capability": True,
            "not_a_production_gravity_freeze": True,
            "not_native_runtime_execution": True,
            "not_tg10_tg3_hcli_agent_os_or_manager_qualified": True,
            "raw_source_remains_authority_teacher_only": True,
        }

    def _manifest_extra_fields(
        self,
        *,
        ordered: Sequence[Mapping[str, Any]],
        artifact_bytes: int,
        elements: int,
        complete_bpw: float,
    ) -> dict[str, Any]:
        """Optional additive manifest evidence for a separate candidate branch.

        The baseline intentionally returns no fields so its legacy artifact
        contract and byte ledger remain exactly the same.
        """

        del ordered, artifact_bytes, elements, complete_bpw
        return {}

    def run(self, *, max_tensors: int) -> int:
        if max_tensors <= 0:
            raise CompleteGravityError("max_tensors must be positive")
        audit, weight_map, shard_evidence = self._admit_source()
        revalidation, revalidated_now = self._revalidate_current_source(
            audit=audit,
            weight_map=weight_map,
            shard_evidence=shard_evidence,
        )
        progress_binding = self._progress_binding(
            audit_seal=str(audit["seal_sha256"]),
            revalidation_seal=str(revalidation["seal_sha256"]),
        )
        planned_order = self._planned_tensor_order(weight_map)
        if len(planned_order) != len(weight_map):
            raise CompleteGravityError("source index does not produce one deterministic entry per tensor")
        progress, progress_journal, scheduler = self._load_progress_index(
            binding=progress_binding,
            planned_order=planned_order,
            shard_evidence=shard_evidence,
        )
        self.tensor_dir.mkdir(parents=True, exist_ok=True)
        planned = len(planned_order)
        completed = int(scheduler["source_bound_completed_tensors"])
        if completed == planned and int(scheduler["next_cursor"]) == planned:
            # KeepAlive invokes this bounded compiler again after a successful
            # exit.  A valid terminal receipt must therefore be checked before
            # publishing PACKING; otherwise the admission watcher observes a
            # recurrent false non-terminal window and the manifest gets a new
            # recorded_at/seal on every restart.
            source_catalog_sha256 = self._source_catalog_sha256(
                weight_map=weight_map,
                shard_evidence=shard_evidence,
            )
            progress_catalog_sha256 = self._complete_progress_catalog_sha256(
                progress=progress,
                weight_map=weight_map,
                shard_evidence=shard_evidence,
            )
            terminal_progress = self._terminal_progress(planned=planned, scheduler=scheduler)
            terminal = self._current_terminal_receipt(
                audit=audit,
                revalidation=revalidation,
                terminal_progress=terminal_progress,
                source_catalog_sha256=source_catalog_sha256,
                progress_catalog_sha256=progress_catalog_sha256,
            )
            if terminal is None:
                # Upgrade a complete pre-terminal manifest once.  The helper
                # verifies the full catalog and ledger rather than treating a
                # matching filename as an admission signal.
                manifest_binding = self._admit_existing_complete_manifest(
                    audit=audit,
                    revalidation=revalidation,
                    weight_map=weight_map,
                    shard_evidence=shard_evidence,
                    progress=progress,
                    source_catalog_sha256=source_catalog_sha256,
                    progress_catalog_sha256=progress_catalog_sha256,
                )
                if manifest_binding is not None:
                    terminal = self._write_terminal_receipt(
                        audit=audit,
                        revalidation=revalidation,
                        terminal_progress=terminal_progress,
                        manifest_binding=manifest_binding,
                    )
            if terminal is not None:
                self._publish_terminal(terminal, revalidated_now=revalidated_now)
                return 0
        performed = 0
        self._publish(
            "PACKING_COMPLETE_BINARY_GRAVITY",
            source_revalidation={
                "receipt_path": str(self.source_revalidation_path),
                "receipt_seal_sha256": revalidation["seal_sha256"],
                "full_shards_revalidated_this_cycle": revalidated_now,
            },
            progress={
                "planned_tensors": planned,
                "completed_tensors": completed,
                "batch_limit": max_tensors,
                "progress_index_path": str(self.progress_index_path),
                "next_cursor": scheduler["next_cursor"],
                "next_source_shard": scheduler["next_source_shard"],
                "next_tensor_name": scheduler["next_tensor_name"],
            },
        )
        headers: dict[str, dict[str, Any]] = {}
        for ordinal in range(int(scheduler["next_cursor"]), planned):
            shard, tensor_name = planned_order[ordinal]
            source_hash = str(shard_evidence[shard]["sha256"])
            # This only occurs for an already-journaled tail behind a gap.  It
            # is intentionally checked without opening the source shard.
            if self._progress_row_binds_source(
                progress.get(tensor_name),
                shard=shard,
                source_hash=source_hash,
            ):
                continue
            header = headers.get(shard)
            if header is None:
                header = self._header(self.model_dir / shard)
                headers[shard] = header
            info = header.get(tensor_name)
            if not isinstance(info, Mapping):
                raise CompleteGravityError(f"source header lacks indexed tensor {tensor_name}")
            row = self._write_tensor(
                tensor_name=tensor_name,
                shard=shard,
                source_hash=source_hash,
                info=info,
            )
            appended = self._append_progress(row)
            progress_journal = self._advance_progress_index(
                completed=progress,
                journal=progress_journal,
                row=row,
                appended=appended,
            )
            scheduler = self._advance_scheduler(
                scheduler=scheduler,
                completed=progress,
                planned_order=planned_order,
                shard_evidence=shard_evidence,
            )
            completed = int(scheduler["source_bound_completed_tensors"])
            performed += 1
            if performed % 8 == 0 or performed == 1:
                payload_bytes = sum(int(item["artifact_bytes"]) for item in progress.values())
                self._publish(
                    "PACKING_COMPLETE_BINARY_GRAVITY",
                    current_tensor=tensor_name,
                    progress={
                        "planned_tensors": planned,
                        "completed_tensors": completed,
                        "new_tensors_this_cycle": performed,
                        "artifact_bytes": payload_bytes,
                        "batch_limit": max_tensors,
                        "next_cursor": scheduler["next_cursor"],
                        "next_source_shard": scheduler["next_source_shard"],
                        "next_tensor_name": scheduler["next_tensor_name"],
                    },
                )
            if performed >= max_tensors and int(scheduler["next_cursor"]) < planned:
                self._write_progress_index(
                    completed=progress,
                    journal=progress_journal,
                    binding=progress_binding,
                    scheduler=scheduler,
                )
                self._publish(
                    "PACKING_COMPLETE_BINARY_GRAVITY",
                    source_revalidation={
                        "receipt_path": str(self.source_revalidation_path),
                        "receipt_seal_sha256": revalidation["seal_sha256"],
                        "full_shards_revalidated_this_cycle": revalidated_now,
                    },
                    progress={
                        "planned_tensors": planned,
                        "completed_tensors": completed,
                        "new_tensors_this_cycle": performed,
                        "resume_required": int(scheduler["next_cursor"]) < planned,
                        "progress_index_path": str(self.progress_index_path),
                        "next_cursor": scheduler["next_cursor"],
                        "next_source_shard": scheduler["next_source_shard"],
                        "next_tensor_name": scheduler["next_tensor_name"],
                    },
                )
                return 0
        if int(scheduler["next_cursor"]) != planned:
            raise CompleteGravityError(
                "complete candidate omission: "
                f"cursor={scheduler['next_cursor']}, planned={planned}"
            )
        self._write_progress_index(
            completed=progress,
            journal=progress_journal,
            binding=progress_binding,
            scheduler=scheduler,
        )
        authority_rows = self._full_progress_rows()
        ordered: list[dict[str, Any]] = []
        for tensor_name in sorted(weight_map):
            shard = weight_map[tensor_name]
            row = authority_rows.get(tensor_name)
            if row is None or not self._progress_row_is_usable(
                tensor_name=tensor_name,
                shard=shard,
                source_hash=str(shard_evidence[shard]["sha256"]),
                row=row,
            ):
                raise CompleteGravityError(
                    f"complete manifest cannot admit missing or mismatched tensor progress: {tensor_name}"
                )
            ordered.append(row)
        artifact_bytes = sum(int(row["artifact_bytes"]) for row in ordered)
        elements = sum(int(row["elements"]) for row in ordered)
        # Manifest bytes must be billed too.  The ledger contains its own
        # number, so solve the tiny fixed point against the exact pretty JSON
        # serialization used by _atomic_json rather than estimating it.
        preliminary = {
            "schema": self.schema,
            "status": COMPLETE_MANIFEST_STATUS,
            "recorded_at": _utc_now(),
            "source_body_audit_seal_sha256": audit["seal_sha256"],
            "source_revalidation_receipt_path": str(self.source_revalidation_path),
            "source_revalidation_receipt_seal_sha256": revalidation["seal_sha256"],
            "source": {"repository": self.repository, "model_dir": str(self.model_dir), "tensor_count": planned},
            "representation": self._manifest_representation(),
            "tensors": ordered,
            "claim_boundary": {},
        }
        def build_manifest(*, manifest_bytes_billed: int, ledger_padding: str | None = None) -> dict[str, Any]:
            """Build one candidate whose ledger explicitly bills every JSON byte."""

            total_bytes = artifact_bytes + manifest_bytes_billed
            complete_bpw = total_bytes * 8.0 / elements
            ledger: dict[str, Any] = {
                "source_weight_elements": elements,
                "tensor_payload_bytes": artifact_bytes,
                "manifest_bytes_billed": manifest_bytes_billed,
                "all_required_weight_artifact_bytes": total_bytes,
                "complete_physical_bpw": complete_bpw,
                "threshold_bpw": 1.5,
                "passes_storage_threshold": complete_bpw <= 1.5,
                "explicitly_excluded_separate_state": [
                    "KV_cache_bytes",
                    "Qwen80_recurrent_state_bytes",
                    "Context_OS_cache_bytes",
                    "Agent_OS_memory_bytes",
                ],
            }
            if ledger_padding is not None:
                # This is not virtual accounting: it is literal retained JSON
                # payload that resolves an otherwise possible decimal-length
                # two-cycle in the self-billed manifest fixed point.
                ledger["manifest_ledger_padding_bytes"] = len(ledger_padding)
                ledger["manifest_ledger_padding"] = ledger_padding
            return seal({
                **preliminary,
                "complete_physical_bpw_ledger": ledger,
                "champion_classes": self._manifest_champion_classes(complete_bpw=complete_bpw),
                "quality_summary": self._manifest_quality_summary(ordered),
                "claim_boundary": self._manifest_claim_boundary(),
                **self._manifest_extra_fields(
                    ordered=ordered,
                    artifact_bytes=artifact_bytes,
                    elements=elements,
                    complete_bpw=complete_bpw,
                ),
            })

        def manifest_document_bytes(document: Mapping[str, Any]) -> int:
            return len(
                json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ) + 1

        manifest_bytes_billed = 0
        for _ in range(16):
            candidate = build_manifest(manifest_bytes_billed=manifest_bytes_billed)
            actual_bytes = manifest_document_bytes(candidate)
            if actual_bytes == manifest_bytes_billed:
                break
            manifest_bytes_billed = actual_bytes
        else:
            # Decimal float rendering can make the unpadded equation oscillate
            # by a byte or two for some directory lengths.  Retain explicit
            # ASCII padding so the recorded bill is the exact physical file
            # length rather than accepting a close-but-false ledger.
            target_bytes = max(manifest_bytes_billed, actual_bytes) + 8192
            for _ in range(16):
                padding = ""
                for _ in range(16):
                    candidate = build_manifest(
                        manifest_bytes_billed=target_bytes,
                        ledger_padding=padding,
                    )
                    actual_bytes = manifest_document_bytes(candidate)
                    if actual_bytes == target_bytes:
                        break
                    next_padding = len(padding) + target_bytes - actual_bytes
                    if next_padding < 0:
                        break
                    padding = "0" * next_padding
                if actual_bytes == target_bytes:
                    manifest_bytes_billed = target_bytes
                    break
                target_bytes = max(target_bytes, actual_bytes) + 8192
            else:
                raise CompleteGravityError("complete BPW manifest-byte ledger did not converge")
        _atomic_json(self.manifest_path, candidate)
        source_catalog_sha256 = self._source_catalog_sha256(
            weight_map=weight_map,
            shard_evidence=shard_evidence,
        )
        progress_catalog_sha256 = self._complete_progress_catalog_sha256(
            progress=progress,
            weight_map=weight_map,
            shard_evidence=shard_evidence,
        )
        terminal_progress = self._terminal_progress(planned=planned, scheduler=scheduler)
        manifest_binding = self._admit_existing_complete_manifest(
            audit=audit,
            revalidation=revalidation,
            weight_map=weight_map,
            shard_evidence=shard_evidence,
            progress=progress,
            source_catalog_sha256=source_catalog_sha256,
            progress_catalog_sha256=progress_catalog_sha256,
        )
        if manifest_binding is None:
            raise CompleteGravityError(
                "new complete manifest failed its terminal source/progress/catalog admission"
            )
        terminal = self._write_terminal_receipt(
            audit=audit,
            revalidation=revalidation,
            terminal_progress=terminal_progress,
            manifest_binding=manifest_binding,
        )
        self._publish_terminal(terminal, revalidated_now=revalidated_now)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--source-audit", type=Path, default=SOURCE_AUDIT)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--artifact-prefix", default=DEFAULT_ARTIFACT_PREFIX)
    parser.add_argument("--schema", default=SCHEMA)
    parser.add_argument("--max-tensors", type=int, default=16, help="bounded restart-safe work per process cycle")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return CompleteBinaryGravity(
        model_dir=args.model_dir, source_audit=args.source_audit, root=args.root,
        repository=args.repository, model_id=args.model_id,
        artifact_prefix=args.artifact_prefix, schema=args.schema,
    ).run(max_tensors=args.max_tensors)


if __name__ == "__main__":
    raise SystemExit(main())
