#!/usr/bin/env python3.12
"""Fail-closed content-addressed second-host transport for unbound Doctor work.

The module implements contracts and local primitives only; it opens no socket
and claims no remote host.  Immutable chunks support resume/dedup, signed host
capabilities and coordinator leases bind execution, signed result receipts bind
returned artifacts, and merge/recovery is deterministic.  When no eligible host
is supplied the transport plan is explicitly local-only.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any, Iterable, Mapping


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import doctor_v5_gptoss_mxfp4 as mxfp4
import doctor_v5_streaming_source as streaming
import doctor_v5_aggressive_admission_policy as aggressive_admission


CHUNK_MANIFEST_SCHEMA = "hawking.doctor_v5_transport_chunk_manifest.v1"
SOURCE_VERIFICATION_SCHEMA = "hawking.doctor_v5_transport_source_verification.v1"
HOST_CAPABILITY_SCHEMA = "hawking.doctor_v5_transport_host_capability.v1"
SWAP_BASELINE_AUTHORITY_SCHEMA = "hawking.doctor_v5_swap_baseline_authority.v1"
TRANSPORT_PLAN_SCHEMA = "hawking.doctor_v5_transport_plan.v1"
LEASE_SCHEMA = "hawking.doctor_v5_transport_lease.v1"
RESULT_RECEIPT_SCHEMA = "hawking.doctor_v5_transport_result_receipt.v1"
RESULT_ACCEPTANCE_SCHEMA = "hawking.doctor_v5_transport_result_acceptance.v1"
RECOVERY_SCHEMA = "hawking.doctor_v5_transport_recovery.v1"
RESULT_MERGE_SCHEMA = "hawking.doctor_v5_transport_result_merge.v1"
TRANSPORT_REQUIREMENTS_SCHEMA = "hawking.doctor_v5_transport_requirements.v1"
DEFAULT_REQUIREMENTS = (
    ROOT / "reports/condense/doctor_v5_unbound/distributed_transport/requirements.json"
)
SHA_RE = re.compile(r"[0-9a-f]{64}")
KEY_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")
MAX_JSON_BYTES = 256 * 1024 * 1024
DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024
MAX_CHUNK_BYTES = 64 * 1024 * 1024


class TransportError(RuntimeError):
    """A transport identity, lease, chunk, or result contract failed."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _parse_time(value: Any) -> dt.datetime:
    if not isinstance(value, str):
        raise TransportError("timestamp is not a string")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TransportError(f"invalid timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise TransportError("timestamp must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(doc: dict[str, Any], *fields: str) -> dict[str, Any]:
    return {key: value for key, value in doc.items() if key not in fields}


def _key(key: bytes) -> bytes:
    if not isinstance(key, bytes) or len(key) < 32:
        raise TransportError("HMAC key must contain at least 256 secret bits")
    return key


def _seal(
    doc: dict[str, Any], *, digest_field: str, key_id: str, key: bytes,
) -> dict[str, Any]:
    if not isinstance(key_id, str) or KEY_ID_RE.fullmatch(key_id) is None:
        raise TransportError("signing key identifier is invalid")
    if digest_field in doc or "signature" in doc:
        raise TransportError("document is already sealed")
    doc[digest_field] = _hash_value(doc)
    doc["signature"] = {
        "algorithm": "HMAC-SHA256", "key_id": key_id,
        "digest": hmac.new(_key(key), _canonical(doc), hashlib.sha256).hexdigest(),
    }
    return doc


def _verify_seal(
    doc: Any, *, digest_field: str, keys: Mapping[str, bytes],
) -> list[str]:
    if not isinstance(doc, dict):
        return ["signed document is not an object"]
    errors: list[str] = []
    signature = doc.get("signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != "HMAC-SHA256" \
            or not isinstance(signature.get("key_id"), str) \
            or not isinstance(signature.get("digest"), str) \
            or SHA_RE.fullmatch(signature["digest"]) is None:
        return ["document signature is invalid"]
    payload = _without(doc, "signature")
    digest = payload.get(digest_field)
    if not isinstance(digest, str) or digest != _hash_value(
            _without(payload, digest_field)):
        errors.append(f"document {digest_field} mismatch")
    key = keys.get(signature["key_id"])
    if key is None:
        errors.append("document signing key is not trusted")
    else:
        expected = hmac.new(_key(key), _canonical(payload), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature["digest"]):
            errors.append("document HMAC signature mismatch")
    return errors


def _aggressive_controller_identity() -> dict[str, Any]:
    path = Path(aggressive_admission.__file__).resolve(strict=True)
    raw = path.read_bytes()
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def build_swap_baseline_authority(
    *, host_id: str, instance_nonce: str, sealed_baseline_swap_bytes: int,
    authorized_at: str, expires_at: str, coordinator_key_id: str,
    coordinator_key: bytes,
) -> dict[str, Any]:
    if not isinstance(host_id, str) or not host_id \
            or not isinstance(instance_nonce, str) or not instance_nonce \
            or isinstance(sealed_baseline_swap_bytes, bool) \
            or not isinstance(sealed_baseline_swap_bytes, int) \
            or sealed_baseline_swap_bytes < 0 \
            or _parse_time(authorized_at) >= _parse_time(expires_at):
        raise TransportError("coordinator swap baseline authority is invalid")
    baseline_mb = round(sealed_baseline_swap_bytes / (1024.0 * 1024.0), 3)
    doc: dict[str, Any] = {
        "schema": SWAP_BASELINE_AUTHORITY_SCHEMA,
        "host_id": host_id, "instance_nonce": instance_nonce,
        "authorized_at": authorized_at, "expires_at": expires_at,
        "sealed_baseline_swap_bytes": sealed_baseline_swap_bytes,
        "sealed_baseline_swap_mb": baseline_mb,
        "baseline_can_ratchet": False,
        "controller_artifact": _aggressive_controller_identity(),
        "policy": aggressive_admission.swap_policy(),
        "policy_sha256": _hash_value(aggressive_admission.swap_policy()),
    }
    return _seal(doc, digest_field="baseline_authority_sha256",
                 key_id=coordinator_key_id, key=coordinator_key)


def validate_swap_baseline_authority(
    doc: Any, *, host_id: str, instance_nonce: str,
    coordinator_keys: Mapping[str, bytes], now: str,
) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != SWAP_BASELINE_AUTHORITY_SCHEMA:
        return ["coordinator swap baseline authority schema mismatch"]
    errors = _verify_seal(
        doc, digest_field="baseline_authority_sha256", keys=coordinator_keys
    )
    try:
        authorized, expires, current = (_parse_time(doc.get("authorized_at")),
                                        _parse_time(doc.get("expires_at")),
                                        _parse_time(now))
        if authorized >= expires or current < authorized or current >= expires:
            errors.append("coordinator swap baseline authority is outside its lifetime")
    except TransportError as exc:
        errors.append(str(exc))
    baseline = doc.get("sealed_baseline_swap_bytes")
    if doc.get("host_id") != host_id or doc.get("instance_nonce") != instance_nonce \
            or isinstance(baseline, bool) or not isinstance(baseline, int) or baseline < 0 \
            or doc.get("sealed_baseline_swap_mb") \
            != round(baseline / (1024.0 * 1024.0), 3) \
            or doc.get("baseline_can_ratchet") is not False:
        errors.append("coordinator swap baseline host/value binding differs")
    policy = aggressive_admission.swap_policy()
    if doc.get("controller_artifact") != _aggressive_controller_identity() \
            or doc.get("policy") != policy \
            or doc.get("policy_sha256") != _hash_value(policy):
        errors.append("coordinator swap baseline policy/controller hash differs")
    return errors


def build_resource_attestation(
    *, baseline_authority: dict[str, Any], prior_controller_state: dict[str, Any],
    swap_used_bytes: int, sampled_epoch: float, memory_pressure: str = "normal",
    ac_power: bool = True, thermal_state: str = "nominal",
) -> dict[str, Any]:
    """Advance the reviewed controller from a signed prior state and sample."""
    if isinstance(swap_used_bytes, bool) or not isinstance(swap_used_bytes, int) \
            or swap_used_bytes < 0 or isinstance(sampled_epoch, bool) \
            or not isinstance(sampled_epoch, (int, float)) \
            or not math.isfinite(float(sampled_epoch)) or sampled_epoch < 0:
        raise TransportError("host swap resource sample is invalid")
    baseline_mb = baseline_authority.get("sealed_baseline_swap_mb")
    if isinstance(baseline_mb, bool) or not isinstance(baseline_mb, (int, float)):
        raise TransportError("host resource row lacks a baseline authority")
    pressure_level = {"normal": 1, "warning": 2, "critical": 4}.get(memory_pressure)
    sample = {
        "pressure_level": pressure_level,
        "swap_used_mb": swap_used_bytes / (1024.0 * 1024.0),
        "sampled_epoch": float(sampled_epoch),
    }
    next_state, decision = aggressive_admission.advance_swap_state(
        prior_controller_state,
        {"pressure_level": pressure_level, "swap_used_mb": sample["swap_used_mb"]},
        now_epoch=float(sampled_epoch), sealed_baseline_swap_mb=float(baseline_mb),
    )
    admission: dict[str, Any] = {
        "controller_artifact": _aggressive_controller_identity(),
        "policy": aggressive_admission.swap_policy(),
        "policy_sha256": _hash_value(aggressive_admission.swap_policy()),
        "baseline_authority": baseline_authority,
        "prior_controller_state": prior_controller_state,
        "sample": sample, "next_controller_state": next_state,
        "decision": decision,
    }
    admission["attestation_sha256"] = _hash_value(admission)
    return {
        "memory_pressure": memory_pressure, "swap_used_bytes": swap_used_bytes,
        "ac_power": ac_power, "thermal_state": thermal_state,
        "aggressive_admission": admission,
    }


def _validate_resource_attestation(
    resource: Any, *, host_id: str, instance_nonce: str, observed_at: str,
    baseline_authority_keys: Mapping[str, bytes],
) -> list[str]:
    if not isinstance(resource, dict) or resource.get("memory_pressure") != "normal" \
            or resource.get("ac_power") is not True \
            or resource.get("thermal_state") not in {"nominal", "fair"}:
        return ["host current resource attestation is not execution-admissible"]
    swap = resource.get("swap_used_bytes")
    admission = resource.get("aggressive_admission")
    if isinstance(swap, bool) or not isinstance(swap, int) or swap < 0 \
            or not isinstance(admission, dict) \
            or admission.get("attestation_sha256") != _hash_value(
                _without(admission, "attestation_sha256")):
        return ["host signed aggressive admission attestation is invalid"]
    authority = admission.get("baseline_authority")
    errors = validate_swap_baseline_authority(
        authority, host_id=host_id, instance_nonce=instance_nonce,
        coordinator_keys=baseline_authority_keys, now=observed_at,
    )
    policy = aggressive_admission.swap_policy()
    if admission.get("controller_artifact") != _aggressive_controller_identity() \
            or admission.get("policy") != policy \
            or admission.get("policy_sha256") != _hash_value(policy):
        errors.append("host aggressive admission policy/controller hash differs")
    if not isinstance(authority, dict):
        return errors
    baseline_mb = authority.get("sealed_baseline_swap_mb")
    prior, sample = admission.get("prior_controller_state"), admission.get("sample")
    try:
        observed_epoch = _parse_time(observed_at).timestamp()
    except TransportError as exc:
        errors.append(str(exc))
        return errors
    if isinstance(baseline_mb, bool) or not isinstance(baseline_mb, (int, float)) \
            or not isinstance(sample, dict) \
            or sample.get("pressure_level") != 1 \
            or sample.get("swap_used_mb") != swap / (1024.0 * 1024.0) \
            or sample.get("sampled_epoch") != observed_epoch:
        errors.append("host swap sample differs from the signed capability observation")
        return errors
    if aggressive_admission.validate_swap_state(
            prior, sealed_baseline_swap_mb=float(baseline_mb)):
        errors.append("host prior aggressive-controller state is invalid or re-baselined")
        return errors
    try:
        expected_state, expected_decision = aggressive_admission.advance_swap_state(
            prior, {"pressure_level": 1, "swap_used_mb": sample["swap_used_mb"]},
            now_epoch=float(sample["sampled_epoch"]),
            sealed_baseline_swap_mb=float(baseline_mb),
        )
    except (KeyError, TypeError, ValueError, aggressive_admission.PolicyError) as exc:
        errors.append(f"host aggressive-controller transition is invalid: {exc}")
        return errors
    if admission.get("next_controller_state") != expected_state \
            or admission.get("decision") != expected_decision:
        errors.append("host controller state/decision differs from reviewed transition")
    elif expected_decision.get("allow_launch") is not True \
            or expected_decision.get("mode") not in {"green", "soft_throttle"}:
        errors.append("host signed swap decision does not admit a bounded launch")
    return errors


def build_requirements_packet() -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema": TRANSPORT_REQUIREMENTS_SCHEMA, "created_at": _now(),
        "status": "no-remote-host-claimed-local-fallback-only",
        "authentication": {
            "contract_signature": "HMAC-SHA256 with out-of-band >=256-bit keys",
            "wire_transport": "mutually authenticated TLS required by deployment",
            "secrets_persisted_in_manifest": False,
        },
        "source_transfer": {
            "immutable_content_addressed_chunks": True,
            "default_chunk_bytes": DEFAULT_CHUNK_BYTES,
            "resume_and_dedup": True, "partial_files_never_trusted": True,
            "sender_and_receiver_hash_every_chunk": True,
            "full_source_sha256_verified_after_reassembly": True,
        },
        "execution": {
            "signed_host_capability_required": True,
            "signed_expiring_lease_required": True,
            "tool_hash_equality_required": True,
            "parent_execution_plan_hash_required": True,
            "signed_result_receipt_required": True,
            "conflicting_duplicate_result_fails_closed": True,
            "host_resource_admission": {
                "controller_artifact": _aggressive_controller_identity(),
                "policy_sha256": _hash_value(aggressive_admission.swap_policy()),
                "policy": aggressive_admission.swap_policy(),
                "coordinator_signed_host_instance_baseline_authority_required": True,
                "sealed_non_ratcheting_baseline_required": True,
                "signed_prior_and_next_controller_states_required": True,
                "reviewed_hysteresis_and_cooldowns_required": True,
                "normal_pressure_required": True,
                "signed_admit_decision_required": True,
            },
        },
        "recovery": {
            "verified_chunks_retained_for_dedup": True,
            "unverified_partial_chunks_discarded": True,
            "unfinished_work_released_after_expiry": True,
            "local_only_fallback_always_available": True,
        },
        "remote_host_count": 0, "distributed_execution_enabled": False,
        "runtime_defaults_changed": False, "quality_claims_permitted": False,
    }
    doc["requirements_sha256"] = _hash_value(doc)
    return doc


def build_chunk_manifest(
    *, source_id: str, source_sha256: str, source_bytes: int,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(source_id, str) or not source_id \
            or not isinstance(source_sha256, str) \
            or SHA_RE.fullmatch(source_sha256) is None \
            or isinstance(source_bytes, bool) or not isinstance(source_bytes, int) \
            or source_bytes <= 0:
        raise TransportError("source chunk authority is invalid")
    doc: dict[str, Any] = {
        "schema": CHUNK_MANIFEST_SCHEMA, "created_at": _now(),
        "source": {"source_id": source_id, "sha256": source_sha256,
                   "bytes": source_bytes},
        "chunks": chunks, "chunk_count": len(chunks),
        "transfer": {"resume": True, "dedup_by_sha256": True,
                     "partial_chunk_trusted": False},
    }
    doc["chunk_manifest_sha256"] = _hash_value(doc)
    errors = validate_chunk_manifest(doc)
    if errors:
        raise TransportError("generated chunk manifest invalid: " + "; ".join(errors))
    return doc


def validate_chunk_manifest(doc: Any) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != CHUNK_MANIFEST_SCHEMA:
        return ["chunk-manifest schema mismatch"]
    errors: list[str] = []
    if doc.get("chunk_manifest_sha256") != _hash_value(
            _without(doc, "chunk_manifest_sha256")):
        errors.append("chunk-manifest hash mismatch")
    source = doc.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("source_id"), str) \
            or not isinstance(source.get("sha256"), str) \
            or SHA_RE.fullmatch(source["sha256"]) is None \
            or not isinstance(source.get("bytes"), int) or source["bytes"] <= 0:
        errors.append("chunk-manifest source authority is invalid")
        source = {"bytes": 0}
    chunks = doc.get("chunks")
    if not isinstance(chunks, list) or not chunks \
            or doc.get("chunk_count") != len(chunks):
        errors.append("chunk-manifest rows/count are invalid")
        chunks = []
    cursor = 0
    seen_sha: dict[str, int] = {}
    for index, row in enumerate(chunks):
        byte_range = row.get("absolute_byte_range") if isinstance(row, dict) else None
        if not isinstance(row, dict) or row.get("index") != index \
                or not isinstance(byte_range, list) or len(byte_range) != 2 \
                or byte_range[0] != cursor or byte_range[1] <= byte_range[0] \
                or row.get("bytes") != byte_range[1] - byte_range[0] \
                or row["bytes"] > MAX_CHUNK_BYTES \
                or not isinstance(row.get("sha256"), str) \
                or SHA_RE.fullmatch(row["sha256"]) is None:
            errors.append(f"chunk row is invalid or non-contiguous: {index}")
            continue
        previous_size = seen_sha.setdefault(row["sha256"], row["bytes"])
        if previous_size != row["bytes"]:
            errors.append("same chunk digest is declared with different byte sizes")
        cursor = byte_range[1]
    if cursor != source.get("bytes"):
        errors.append("chunk ranges do not exactly cover the source")
    transfer = doc.get("transfer")
    if not isinstance(transfer, dict) or transfer.get("resume") is not True \
            or transfer.get("dedup_by_sha256") is not True \
            or transfer.get("partial_chunk_trusted") is not False:
        errors.append("chunk transfer safety contract is invalid")
    return errors


def chunk_local_source(
    *, source_id: str, path: Path, expected_sha256: str,
    expected_bytes: int, chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> dict[str, Any]:
    """Hash a local source into chunks without materializing it; caller owns I/O lease."""
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) \
            or not 1 <= chunk_bytes <= MAX_CHUNK_BYTES:
        raise TransportError("chunk size is outside the transport envelope")
    chunks = []
    full_digest = hashlib.sha256()
    with streaming.ImmutableSourceReader(
        path, expected_bytes=expected_bytes, expected_sha256=expected_sha256,
    ) as reader:
        for index, offset in enumerate(range(0, expected_bytes, chunk_bytes)):
            size = min(chunk_bytes, expected_bytes - offset)
            chunk_digest = hashlib.sha256()
            for block in reader.iter_range(
                    offset, size,
                    chunk_bytes=min(chunk_bytes, streaming.DEFAULT_CHUNK_BYTES)):
                chunk_digest.update(block)
                full_digest.update(block)
            chunks.append({"index": index, "absolute_byte_range": [offset, offset + size],
                           "bytes": size, "sha256": chunk_digest.hexdigest()})
    if full_digest.hexdigest() != expected_sha256:
        raise TransportError("single-pass full-source SHA-256 differs from authority")
    return build_chunk_manifest(
        source_id=source_id, source_sha256=expected_sha256,
        source_bytes=expected_bytes, chunks=chunks,
    )


class ContentAddressedChunkStore:
    """Small local CAS primitive used by a future authenticated receiver."""

    def __init__(self, root: Path) -> None:
        raw = Path(root)
        if raw.is_symlink():
            raise TransportError("symlinked chunk store root is forbidden")
        raw.mkdir(parents=True, exist_ok=True)
        self.root = raw.resolve(strict=True)

    def _path(self, digest: str) -> Path:
        if not isinstance(digest, str) or SHA_RE.fullmatch(digest) is None:
            raise TransportError("chunk digest is invalid")
        parent = self.root / digest[:2]
        target = parent / digest
        if parent.is_symlink() or target.is_symlink():
            raise TransportError("symlinked chunk-store component is forbidden")
        return target

    def accept(self, payload: bytes, *, expected_sha256: str) -> dict[str, Any]:
        if not isinstance(payload, bytes) or len(payload) > MAX_CHUNK_BYTES:
            raise TransportError("received chunk is not bounded bytes")
        observed = hashlib.sha256(payload).hexdigest()
        if observed != expected_sha256:
            raise TransportError("received chunk digest differs")
        target = self._path(observed)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.parent.is_symlink() or target.is_symlink():
            raise TransportError("symlinked chunk-store component is forbidden")
        if target.exists():
            current = target.read_bytes()
            if len(current) != len(payload) or hashlib.sha256(current).hexdigest() != observed:
                raise TransportError("existing dedup chunk is corrupt")
            return {"sha256": observed, "bytes": len(payload), "deduplicated": True,
                    "path": str(target)}
        tmp = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        try:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
            fd = os.open(tmp, flags, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
            mxfp4._fsync_dir(target.parent)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        return {"sha256": observed, "bytes": len(payload), "deduplicated": False,
                "path": str(target)}

    def verified_index(self, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
        errors = validate_chunk_manifest(manifest)
        if errors:
            raise TransportError("cannot index invalid chunk manifest")
        result = {}
        for row in manifest["chunks"]:
            path = self._path(row["sha256"])
            if not path.is_file() or path.is_symlink() or path.stat().st_size != row["bytes"]:
                continue
            digest, size = mxfp4._hash_file(path)
            if digest == row["sha256"] and size == row["bytes"]:
                result[digest] = {"sha256": digest, "bytes": size, "path": str(path)}
        return result

    def verify_source(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Re-hash canonical chunk order into the inherited full-source digest."""
        errors = validate_chunk_manifest(manifest)
        if errors:
            raise TransportError("cannot verify invalid chunk manifest")
        digest, total = hashlib.sha256(), 0
        verified = []
        for row in manifest["chunks"]:
            path = self._path(row["sha256"])
            if not path.is_file() or path.is_symlink() or path.stat().st_size != row["bytes"]:
                raise TransportError(f"verified source chunk is absent: {row['index']}")
            chunk_digest = hashlib.sha256()
            with path.open("rb") as handle:
                while True:
                    block = handle.read(8 * 1024 * 1024)
                    if not block:
                        break
                    chunk_digest.update(block)
                    digest.update(block)
                    total += len(block)
            if chunk_digest.hexdigest() != row["sha256"]:
                raise TransportError(f"source chunk digest differs: {row['index']}")
            verified.append(row["sha256"])
        source = manifest["source"]
        if total != source["bytes"] or digest.hexdigest() != source["sha256"]:
            raise TransportError("reassembled source identity differs from authority")
        receipt = {
            "schema": SOURCE_VERIFICATION_SCHEMA, "created_at": _now(),
            "chunk_manifest_sha256": manifest["chunk_manifest_sha256"],
            "source": source, "verified_chunk_sha256": verified,
            "canonical_chunk_order": True, "whole_source_materialized": False,
            "source_deletion_permitted": False,
        }
        receipt["source_verification_sha256"] = _hash_value(receipt)
        return receipt


def build_resume_plan(
    manifest: dict[str, Any], verified_index: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    errors = validate_chunk_manifest(manifest)
    if errors:
        raise TransportError("cannot resume invalid chunk manifest")
    reused, missing = [], []
    for row in manifest["chunks"]:
        local = verified_index.get(row["sha256"])
        if isinstance(local, dict) and local.get("bytes") == row["bytes"]:
            reused.append({"index": row["index"], "sha256": row["sha256"],
                           "bytes": row["bytes"]})
        else:
            missing.append({"index": row["index"], "sha256": row["sha256"],
                            "bytes": row["bytes"],
                            "absolute_byte_range": row["absolute_byte_range"]})
    doc = {
        "schema": "hawking.doctor_v5_transport_resume_plan.v1",
        "created_at": _now(), "chunk_manifest_sha256": manifest[
            "chunk_manifest_sha256"
        ], "reused": reused, "missing": missing,
        "reused_bytes": sum(row["bytes"] for row in reused),
        "missing_bytes": sum(row["bytes"] for row in missing),
        "unverified_partial_chunks_reused": False,
    }
    doc["resume_plan_sha256"] = _hash_value(doc)
    return doc


def build_host_capability(
    *, host_id: str, instance_nonce: str, architecture: str,
    logical_cpu_count: int, memory_bytes: int, process_budget_bytes: int,
    free_disk_bytes: int, tool_artifacts: list[dict[str, Any]],
    resource_state: dict[str, Any],
    transport_certificate_sha256: str, observed_at: str, expires_at: str,
    key_id: str, key: bytes,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema": HOST_CAPABILITY_SCHEMA, "host_id": host_id,
        "instance_nonce": instance_nonce, "observed_at": observed_at,
        "expires_at": expires_at, "architecture": architecture,
        "logical_cpu_count": logical_cpu_count, "memory_bytes": memory_bytes,
        "process_budget_bytes": process_budget_bytes,
        "free_disk_bytes": free_disk_bytes, "tool_artifacts": tool_artifacts,
        "resource_state": resource_state,
        "transport_certificate_sha256": transport_certificate_sha256,
        "capabilities": {
            "immutable_chunk_store": True, "bounded_range_reader": True,
            "signed_lease_enforcement": True, "signed_result_receipts": True,
            "source_deletion_permitted": False,
        },
        "quality_claims_permitted": False,
    }
    return _seal(doc, digest_field="host_capability_sha256", key_id=key_id, key=key)


def validate_host_capability(
    doc: Any, *, keys: Mapping[str, bytes],
    baseline_authority_keys: Mapping[str, bytes] | None = None,
    now: str | None = None,
) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != HOST_CAPABILITY_SCHEMA:
        return ["host-capability schema mismatch"]
    errors = _verify_seal(doc, digest_field="host_capability_sha256", keys=keys)
    try:
        observed, expires = _parse_time(doc.get("observed_at")), _parse_time(
            doc.get("expires_at")
        )
        current = _parse_time(now) if now is not None else dt.datetime.now(dt.timezone.utc)
        if observed >= expires or current >= expires:
            errors.append("host capability is expired or temporally invalid")
    except TransportError as exc:
        errors.append(str(exc))
    if not isinstance(doc.get("host_id"), str) or not doc["host_id"] \
            or not isinstance(doc.get("instance_nonce"), str) or not doc["instance_nonce"] \
            or doc.get("architecture") not in {"arm64", "aarch64"}:
        errors.append("host identity/architecture is invalid")
    for field in ("logical_cpu_count", "memory_bytes", "process_budget_bytes",
                  "free_disk_bytes"):
        value = doc.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"host capability {field} is invalid")
    if isinstance(doc.get("memory_bytes"), int) \
            and isinstance(doc.get("process_budget_bytes"), int) \
            and doc["process_budget_bytes"] > doc["memory_bytes"]:
        errors.append("host process budget exceeds physical memory")
    tools = doc.get("tool_artifacts")
    if not isinstance(tools, list) or not tools:
        errors.append("host tool artifact inventory is missing")
    else:
        roles: set[str] = set()
        for row in tools:
            if not isinstance(row, dict) or not isinstance(row.get("role"), str) \
                    or row["role"] in roles or not isinstance(row.get("sha256"), str) \
                    or SHA_RE.fullmatch(row["sha256"]) is None \
                    or not isinstance(row.get("bytes"), int):
                errors.append("host tool artifact is invalid or duplicated")
                break
            roles.add(row["role"])
    certificate = doc.get("transport_certificate_sha256")
    if not isinstance(certificate, str) or SHA_RE.fullmatch(certificate) is None:
        errors.append("host transport certificate fingerprint is invalid")
    caps = doc.get("capabilities")
    if not isinstance(caps, dict) or any(caps.get(field) is not True for field in (
            "immutable_chunk_store", "bounded_range_reader",
            "signed_lease_enforcement", "signed_result_receipts",
            )) or caps.get("source_deletion_permitted") is not False:
        errors.append("host required capability set is incomplete")
    errors.extend(_validate_resource_attestation(
        doc.get("resource_state"), host_id=doc.get("host_id"),
        instance_nonce=doc.get("instance_nonce"), observed_at=doc.get("observed_at"),
        baseline_authority_keys=(baseline_authority_keys or {}),
    ))
    return errors


def _parent_identity(parent: dict[str, Any]) -> tuple[str, str, list[str]]:
    schema = parent.get("schema")
    if schema == "hawking.doctor_v5_gptoss_reuse_fanout_plan.v1":
        digest, rows, field = parent.get("fanout_plan_sha256"), parent.get("jobs"), "job_id"
    elif schema == "hawking.doctor_v5_higher_tier_admission_plan.v1":
        digest, rows, field = (parent.get("admission_plan_sha256"),
                               parent.get("output_units"), "output_unit_id")
    else:
        raise TransportError(f"unsupported unbound parent plan schema: {schema}")
    if not isinstance(digest, str) or SHA_RE.fullmatch(digest) is None \
            or not isinstance(rows, list):
        raise TransportError("parent execution plan identity is invalid")
    work_ids = [row.get(field) for row in rows if isinstance(row, dict)]
    if any(not isinstance(value, str) or not value for value in work_ids) \
            or len(work_ids) != len(set(work_ids)):
        raise TransportError("parent execution work identifiers are invalid")
    return schema, digest, sorted(work_ids)


def build_transport_plan(
    *, parent: dict[str, Any], chunk_manifests: list[dict[str, Any]],
    host_capabilities: list[dict[str, Any]], trusted_host_keys: Mapping[str, bytes],
    required_tool_artifacts: list[dict[str, Any]], nominal_link_bps: int,
    coordinator_key_id: str, coordinator_key: bytes,
) -> dict[str, Any]:
    parent_schema, parent_sha, work_ids = _parent_identity(parent)
    for manifest in chunk_manifests:
        errors = validate_chunk_manifest(manifest)
        if errors:
            raise TransportError("invalid chunk manifest: " + "; ".join(errors))
    manifest_ids = [row["chunk_manifest_sha256"] for row in chunk_manifests]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise TransportError("duplicate chunk manifest identity")
    required_tools = {row.get("role"): (row.get("sha256"), row.get("bytes"))
                      for row in required_tool_artifacts if isinstance(row, dict)}
    if not required_tools or any(not isinstance(role, str)
                                 or not isinstance(identity[0], str)
                                 or SHA_RE.fullmatch(identity[0]) is None
                                 or not isinstance(identity[1], int)
                                 for role, identity in required_tools.items()):
        raise TransportError("required tool artifact authority is invalid")
    eligible = []
    for capability in host_capabilities:
        errors = validate_host_capability(
            capability, keys=trusted_host_keys,
            baseline_authority_keys={coordinator_key_id: coordinator_key},
        )
        if errors:
            continue
        tools = {row["role"]: (row["sha256"], row["bytes"])
                 for row in capability["tool_artifacts"]}
        if all(tools.get(role) == identity for role, identity in required_tools.items()):
            eligible.append({
                "host_id": capability["host_id"],
                "host_capability_sha256": capability["host_capability_sha256"],
                "host_signing_key_id": capability["signature"]["key_id"],
                "swap_baseline_authority_sha256": capability["resource_state"][
                    "aggressive_admission"
                ]["baseline_authority"]["baseline_authority_sha256"],
                "capability_observed_at": capability["observed_at"],
                "capability_expires_at": capability["expires_at"],
            })
    if isinstance(nominal_link_bps, bool) or not isinstance(nominal_link_bps, int) \
            or nominal_link_bps <= 0:
        raise TransportError("nominal link rate is invalid")
    source_bytes = sum(row["source"]["bytes"] for row in chunk_manifests)
    wire_seconds = source_bytes * 8 / nominal_link_bps
    doc: dict[str, Any] = {
        "schema": TRANSPORT_PLAN_SCHEMA, "created_at": _now(),
        "status": "remote-lease-eligible" if eligible else "local-only-no-eligible-host",
        "parent": {"schema": parent_schema, "sha256": parent_sha,
                   "work_unit_count": len(work_ids),
                   "work_ids_sha256": _hash_value(work_ids)},
        "chunk_manifests": manifest_ids, "source_bytes": source_bytes,
        "required_tool_artifacts": required_tool_artifacts,
        "eligible_hosts": sorted(eligible, key=lambda row: row["host_id"]),
        "network_projection": {
            "nominal_link_bps": nominal_link_bps,
            "wire_floor_seconds": wire_seconds,
            "projected_seconds_at_90pct_efficiency": wire_seconds / 0.90,
            "projected_seconds_at_75pct_efficiency": wire_seconds / 0.75,
            "measured": False, "speed_claim_permitted": False,
        },
        "activation": {
            "remote_execution_enabled": False,
            "explicit_signed_lease_required": True,
            "local_only_fallback": True,
            "runtime_defaults_changed": False,
        },
        "failure_policy": {
            "lease_expiry_requeues_unfinished_work": True,
            "verified_chunks_survive_retry": True,
            "unverified_partial_chunks_are_discarded": True,
            "conflicting_result_hash_fails_closed": True,
        },
        "quality_claims_permitted": False,
    }
    return _seal(doc, digest_field="transport_plan_sha256",
                 key_id=coordinator_key_id, key=coordinator_key)


def validate_transport_plan(
    doc: Any, *, parent: dict[str, Any], coordinator_keys: Mapping[str, bytes],
) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != TRANSPORT_PLAN_SCHEMA:
        return ["transport-plan schema mismatch"]
    errors = _verify_seal(doc, digest_field="transport_plan_sha256",
                          keys=coordinator_keys)
    try:
        parent_schema, parent_sha, work_ids = _parent_identity(parent)
        binding = doc.get("parent")
        if not isinstance(binding, dict) or binding.get("schema") != parent_schema \
                or binding.get("sha256") != parent_sha \
                or binding.get("work_unit_count") != len(work_ids) \
                or binding.get("work_ids_sha256") != _hash_value(work_ids):
            errors.append("transport parent plan binding differs")
    except TransportError as exc:
        errors.append(str(exc))
    eligible = doc.get("eligible_hosts")
    if not isinstance(eligible, list):
        errors.append("transport eligible-host list is invalid")
        eligible = []
    elif any(not isinstance(row, dict)
             or not isinstance(row.get("swap_baseline_authority_sha256"), str)
             or SHA_RE.fullmatch(row["swap_baseline_authority_sha256"]) is None
             for row in eligible):
        errors.append("transport eligible host lacks baseline authority binding")
    expected_status = "remote-lease-eligible" if eligible else "local-only-no-eligible-host"
    if doc.get("status") != expected_status:
        errors.append("transport status does not match eligible hosts")
    activation = doc.get("activation")
    if not isinstance(activation, dict) \
            or activation.get("remote_execution_enabled") is not False \
            or activation.get("explicit_signed_lease_required") is not True \
            or activation.get("local_only_fallback") is not True \
            or activation.get("runtime_defaults_changed") is not False:
        errors.append("transport activation/default boundary is invalid")
    projection = doc.get("network_projection")
    if not isinstance(projection, dict) or projection.get("measured") is not False \
            or projection.get("speed_claim_permitted") is not False:
        errors.append("transport network estimate is not labeled provisional")
    if doc.get("quality_claims_permitted") is not False:
        errors.append("transport plan improperly permits quality claims")
    return errors


def build_lease(
    *, transport_plan: dict[str, Any], parent: dict[str, Any],
    host_capability: dict[str, Any], work_ids: list[str], attempt: int,
    issued_at: str, expires_at: str, coordinator_key_id: str,
    coordinator_key: bytes,
) -> dict[str, Any]:
    _schema, _sha, allowed_work = _parent_identity(parent)
    eligible = {row["host_capability_sha256"]: row
                for row in transport_plan.get("eligible_hosts", [])}
    if host_capability.get("host_capability_sha256") not in eligible:
        raise TransportError("host is not eligible under transport plan")
    if not isinstance(work_ids, list) or not work_ids \
            or len(work_ids) != len(set(work_ids)) or not set(work_ids) <= set(allowed_work):
        raise TransportError("lease work set is invalid")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1 \
            or _parse_time(issued_at) >= _parse_time(expires_at):
        raise TransportError("lease attempt/time window is invalid")
    eligible_host = eligible[host_capability["host_capability_sha256"]]
    capability_authority_sha = host_capability.get("resource_state", {}).get(
        "aggressive_admission", {}
    ).get("baseline_authority", {}).get("baseline_authority_sha256")
    if capability_authority_sha != eligible_host.get("swap_baseline_authority_sha256"):
        raise TransportError("host swap baseline authority differs from transport plan")
    if _parse_time(issued_at) < _parse_time(eligible_host["capability_observed_at"]) \
            or _parse_time(expires_at) > _parse_time(
                eligible_host["capability_expires_at"]
            ):
        raise TransportError("lease window is outside host capability lifetime")
    authority = {
        "transport_plan_sha256": transport_plan["transport_plan_sha256"],
        "host_capability_sha256": host_capability["host_capability_sha256"],
        "host_id": host_capability["host_id"], "work_ids": sorted(work_ids),
        "host_signing_key_id": eligible_host["host_signing_key_id"],
        "swap_baseline_authority_sha256": capability_authority_sha,
        "chunk_manifests": list(transport_plan["chunk_manifests"]),
        "attempt": attempt, "issued_at": issued_at, "expires_at": expires_at,
        "nonce": secrets.token_hex(16),
    }
    doc: dict[str, Any] = {
        "schema": LEASE_SCHEMA, **authority,
        "lease_id": _hash_value(authority),
        "enforcement": {"exclusive_per_work_id": True,
                        "result_after_expiry_accepted_only_if_completed_before_expiry": True,
                        "source_deletion_permitted": False},
    }
    return _seal(doc, digest_field="lease_sha256", key_id=coordinator_key_id,
                 key=coordinator_key)


def validate_lease(
    doc: Any, *, transport_plan: dict[str, Any], parent: dict[str, Any],
    coordinator_keys: Mapping[str, bytes], now: str | None = None,
    require_active: bool = True,
) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != LEASE_SCHEMA:
        return ["lease schema mismatch"]
    errors = _verify_seal(doc, digest_field="lease_sha256", keys=coordinator_keys)
    try:
        _schema, _sha, allowed = _parent_identity(parent)
        issued, expires = _parse_time(doc.get("issued_at")), _parse_time(doc.get("expires_at"))
        current = _parse_time(now) if now is not None else dt.datetime.now(dt.timezone.utc)
        if issued >= expires or (require_active and current >= expires):
            errors.append("lease is expired or temporally invalid")
        work_ids = doc.get("work_ids")
        if not isinstance(work_ids, list) or not work_ids \
                or len(work_ids) != len(set(work_ids)) or not set(work_ids) <= set(allowed):
            errors.append("lease work coverage is invalid")
        authority = {key: doc.get(key) for key in (
            "transport_plan_sha256", "host_capability_sha256", "host_id", "work_ids",
            "host_signing_key_id", "swap_baseline_authority_sha256",
            "chunk_manifests", "attempt", "issued_at", "expires_at", "nonce",
        )}
        if doc.get("lease_id") != _hash_value(authority):
            errors.append("lease identity hash mismatch")
    except TransportError as exc:
        errors.append(str(exc))
    eligible = {row["host_capability_sha256"]: row
                for row in transport_plan.get("eligible_hosts", [])}
    row = eligible.get(doc.get("host_capability_sha256"))
    if doc.get("transport_plan_sha256") != transport_plan.get("transport_plan_sha256") \
            or row is None or row.get("host_id") != doc.get("host_id"):
        errors.append("lease transport/host binding differs")
    elif doc.get("host_signing_key_id") != row.get("host_signing_key_id") \
            or doc.get("swap_baseline_authority_sha256") \
            != row.get("swap_baseline_authority_sha256") \
            or doc.get("chunk_manifests") != transport_plan.get("chunk_manifests"):
        errors.append("lease host key or source chunk binding differs")
    else:
        try:
            if _parse_time(doc["issued_at"]) < _parse_time(row["capability_observed_at"]) \
                    or _parse_time(doc["expires_at"]) > _parse_time(
                        row["capability_expires_at"]
                    ):
                errors.append("lease window exceeds host capability lifetime")
        except TransportError as exc:
            errors.append(str(exc))
    if doc.get("enforcement", {}).get("source_deletion_permitted") is not False:
        errors.append("lease permits source deletion")
    return errors


def validate_nonoverlapping_active_leases(
    leases: Iterable[dict[str, Any]], *, transport_plan: dict[str, Any],
    parent: dict[str, Any], coordinator_keys: Mapping[str, bytes], now: str,
) -> list[str]:
    """Reject two unexpired leases that own the same work identifier."""
    errors: list[str] = []
    owners: dict[str, str] = {}
    current = _parse_time(now)
    for lease in leases:
        lease_errors = validate_lease(
            lease, transport_plan=transport_plan, parent=parent,
            coordinator_keys=coordinator_keys, now=now, require_active=False,
        )
        if lease_errors:
            errors.extend(lease_errors)
            continue
        if current >= _parse_time(lease["expires_at"]):
            continue
        for work_id in lease["work_ids"]:
            previous = owners.setdefault(work_id, lease["lease_id"])
            if previous != lease["lease_id"]:
                errors.append(f"overlapping active lease ownership: {work_id}")
    return errors


def build_result_receipt(
    *, lease: dict[str, Any], results: list[dict[str, Any]],
    source_verifications: list[dict[str, str]], completed_at: str,
    host_key_id: str, host_key: bytes,
) -> dict[str, Any]:
    if not isinstance(results, list) or not results \
            or {row.get("work_id") for row in results if isinstance(row, dict)} \
            != set(lease.get("work_ids", [])) or len(results) != len(lease["work_ids"]):
        raise TransportError("result receipt must cover every leased work id exactly once")
    for row in results:
        if not isinstance(row.get("work_id"), str) \
                or not isinstance(row.get("artifact_sha256"), str) \
                or SHA_RE.fullmatch(row["artifact_sha256"]) is None \
                or not isinstance(row.get("artifact_bytes"), int) \
                or row["artifact_bytes"] < 0 \
                or not isinstance(row.get("artifact_instance_id"), str) \
                or not row["artifact_instance_id"] \
                or not isinstance(row.get("canonical_receipt_sha256"), str) \
                or SHA_RE.fullmatch(row["canonical_receipt_sha256"]) is None:
            raise TransportError("returned work artifact identity is invalid")
    if _parse_time(completed_at) > _parse_time(lease["expires_at"]):
        raise TransportError("results completed after lease expiry")
    if not isinstance(source_verifications, list) \
            or {row.get("chunk_manifest_sha256") for row in source_verifications
                if isinstance(row, dict)} != set(lease.get("chunk_manifests", [])) \
            or len(source_verifications) != len(lease.get("chunk_manifests", [])) \
            or any(not isinstance(row.get("source_verification_sha256"), str)
                   or SHA_RE.fullmatch(row["source_verification_sha256"]) is None
                   for row in source_verifications if isinstance(row, dict)):
        raise TransportError("result source-verification coverage differs")
    doc: dict[str, Any] = {
        "schema": RESULT_RECEIPT_SCHEMA, "status": "complete",
        "lease_id": lease["lease_id"], "lease_sha256": lease["lease_sha256"],
        "transport_plan_sha256": lease["transport_plan_sha256"],
        "host_id": lease["host_id"],
        "host_capability_sha256": lease["host_capability_sha256"],
        "attempt": lease["attempt"], "completed_at": completed_at,
        "source_verifications": sorted(
            source_verifications, key=lambda row: row["chunk_manifest_sha256"]
        ),
        "results": sorted(results, key=lambda row: row["work_id"]),
        "source_files_deleted": False, "quality_claims_permitted": False,
    }
    return _seal(doc, digest_field="result_receipt_sha256",
                 key_id=host_key_id, key=host_key)


def validate_result_receipt(
    doc: Any, *, lease: dict[str, Any], host_keys: Mapping[str, bytes],
) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != RESULT_RECEIPT_SCHEMA:
        return ["result-receipt schema mismatch"]
    errors = _verify_seal(doc, digest_field="result_receipt_sha256", keys=host_keys)
    if doc.get("signature", {}).get("key_id") != lease.get("host_signing_key_id"):
        errors.append("result receipt was not signed by the leased host key")
    if doc.get("lease_id") != lease.get("lease_id") \
            or doc.get("lease_sha256") != lease.get("lease_sha256") \
            or doc.get("transport_plan_sha256") != lease.get("transport_plan_sha256") \
            or doc.get("host_id") != lease.get("host_id") \
            or doc.get("host_capability_sha256") != lease.get("host_capability_sha256"):
        errors.append("result receipt lease/host binding differs")
    try:
        if _parse_time(doc.get("completed_at")) > _parse_time(lease.get("expires_at")):
            errors.append("result receipt completed after lease expiry")
    except TransportError as exc:
        errors.append(str(exc))
    results = doc.get("results")
    if not isinstance(results, list) or len(results) != len(lease.get("work_ids", [])) \
            or {row.get("work_id") for row in results if isinstance(row, dict)} \
            != set(lease.get("work_ids", [])):
        errors.append("result receipt work coverage differs from lease")
    else:
        instances: set[str] = set()
        for row in results:
            instance = row.get("artifact_instance_id")
            if not isinstance(row.get("artifact_sha256"), str) \
                    or SHA_RE.fullmatch(row["artifact_sha256"]) is None \
                    or not isinstance(row.get("canonical_receipt_sha256"), str) \
                    or SHA_RE.fullmatch(row["canonical_receipt_sha256"]) is None \
                    or not isinstance(instance, str) or not instance \
                    or instance in instances:
                errors.append("result artifact identity is invalid or aliased")
                break
            instances.add(instance)
    source_verifications = doc.get("source_verifications")
    if not isinstance(source_verifications, list) \
            or {row.get("chunk_manifest_sha256") for row in source_verifications
                if isinstance(row, dict)} != set(lease.get("chunk_manifests", [])) \
            or len(source_verifications) != len(lease.get("chunk_manifests", [])):
        errors.append("result source-verification coverage differs from lease")
    else:
        for row in source_verifications:
            value = row.get("source_verification_sha256")
            if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
                errors.append("result source verification identity is invalid")
                break
    if doc.get("source_files_deleted") is not False \
            or doc.get("quality_claims_permitted") is not False:
        errors.append("result receipt lifecycle/claim boundary is invalid")
    return errors


def build_result_acceptance(
    *, result_receipt: dict[str, Any], verified_results: list[dict[str, Any]],
    coordinator_key_id: str, coordinator_key: bytes,
) -> dict[str, Any]:
    """Coordinator acknowledgment after independently hashing returned artifacts."""
    expected = {row["work_id"]: row for row in result_receipt.get("results", [])}
    if not isinstance(verified_results, list) or len(verified_results) != len(expected) \
            or {row.get("work_id") for row in verified_results
                if isinstance(row, dict)} != set(expected):
        raise TransportError("result acceptance coverage differs")
    for row in verified_results:
        source = expected[row["work_id"]]
        if any(row.get(field) != source[field] for field in (
                "artifact_sha256", "artifact_bytes", "canonical_receipt_sha256",
                )) or not isinstance(row.get("transfer_verification_sha256"), str) \
                or SHA_RE.fullmatch(row["transfer_verification_sha256"]) is None:
            raise TransportError("coordinator artifact verification differs from host result")
    doc: dict[str, Any] = {
        "schema": RESULT_ACCEPTANCE_SCHEMA, "created_at": _now(),
        "result_receipt_sha256": result_receipt["result_receipt_sha256"],
        "lease_id": result_receipt["lease_id"],
        "verified_results": sorted(verified_results, key=lambda row: row["work_id"]),
        "all_returned_artifacts_rehashed": True,
        "scientific_evidence_accepted": False,
        "quality_claims_permitted": False,
    }
    return _seal(doc, digest_field="result_acceptance_sha256",
                 key_id=coordinator_key_id, key=coordinator_key)


def validate_result_acceptance(
    doc: Any, *, result_receipt: dict[str, Any],
    coordinator_keys: Mapping[str, bytes],
) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != RESULT_ACCEPTANCE_SCHEMA:
        return ["result-acceptance schema mismatch"]
    errors = _verify_seal(doc, digest_field="result_acceptance_sha256",
                          keys=coordinator_keys)
    if doc.get("result_receipt_sha256") != result_receipt.get("result_receipt_sha256") \
            or doc.get("lease_id") != result_receipt.get("lease_id"):
        errors.append("result acceptance binds a different host receipt")
    expected = {row["work_id"]: row for row in result_receipt.get("results", [])}
    verified = doc.get("verified_results")
    if not isinstance(verified, list) or len(verified) != len(expected) \
            or {row.get("work_id") for row in verified if isinstance(row, dict)} \
            != set(expected):
        errors.append("result acceptance work coverage differs")
    else:
        for row in verified:
            source = expected[row["work_id"]]
            if any(row.get(field) != source.get(field) for field in (
                    "artifact_sha256", "artifact_bytes", "canonical_receipt_sha256",
                    )) or not isinstance(row.get("transfer_verification_sha256"), str) \
                    or SHA_RE.fullmatch(row["transfer_verification_sha256"]) is None:
                errors.append("result acceptance artifact verification differs")
                break
    if doc.get("all_returned_artifacts_rehashed") is not True \
            or doc.get("scientific_evidence_accepted") is not False \
            or doc.get("quality_claims_permitted") is not False:
        errors.append("result acceptance claim boundary is invalid")
    return errors


def build_recovery_plan(
    *, lease: dict[str, Any], completed_work_ids: list[str],
    verified_chunk_sha256: list[str], reason: str,
    coordinator_key_id: str, coordinator_key: bytes,
) -> dict[str, Any]:
    leased = lease.get("work_ids", [])
    if not isinstance(completed_work_ids, list) \
            or len(completed_work_ids) != len(set(completed_work_ids)) \
            or not set(completed_work_ids) <= set(leased) \
            or not isinstance(verified_chunk_sha256, list) \
            or any(not isinstance(value, str) or SHA_RE.fullmatch(value) is None
                   for value in verified_chunk_sha256) \
            or not isinstance(reason, str) or not reason:
        raise TransportError("recovery inputs are invalid")
    remaining = sorted(set(leased) - set(completed_work_ids))
    doc: dict[str, Any] = {
        "schema": RECOVERY_SCHEMA, "created_at": _now(),
        "failed_lease_id": lease["lease_id"],
        "failed_lease_sha256": lease["lease_sha256"],
        "previous_attempt": lease["attempt"], "next_attempt": lease["attempt"] + 1,
        "reason": reason, "completed_work_ids": sorted(completed_work_ids),
        "requeue_work_ids": remaining,
        "retain_verified_chunk_sha256": sorted(set(verified_chunk_sha256)),
        "discard_all_unverified_partial_chunks": True,
        "local_only_fallback_permitted": True,
        "source_deletion_permitted": False,
    }
    return _seal(doc, digest_field="recovery_sha256",
                 key_id=coordinator_key_id, key=coordinator_key)


def merge_result_receipts(
    *, target_work_ids: list[str],
    receipt_leases: list[
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
    ],
    host_keys: Mapping[str, bytes], transport_plan: dict[str, Any],
    parent: dict[str, Any], coordinator_keys: Mapping[str, bytes],
    coordinator_key_id: str, coordinator_key: bytes,
) -> dict[str, Any]:
    if not target_work_ids or len(target_work_ids) != len(set(target_work_ids)):
        raise TransportError("result merge target work set is invalid")
    accepted: dict[str, dict[str, Any]] = {}
    receipt_ids: list[str] = []
    acceptance_ids: list[str] = []
    for receipt, lease, acceptance in receipt_leases:
        lease_errors = validate_lease(
            lease, transport_plan=transport_plan, parent=parent,
            coordinator_keys=coordinator_keys, require_active=False,
        )
        if lease_errors:
            raise TransportError("invalid result lease: " + "; ".join(lease_errors))
        errors = validate_result_receipt(receipt, lease=lease, host_keys=host_keys)
        if errors:
            raise TransportError("invalid returned receipt: " + "; ".join(errors))
        acceptance_errors = validate_result_acceptance(
            acceptance, result_receipt=receipt, coordinator_keys=coordinator_keys
        )
        if acceptance_errors:
            raise TransportError(
                "invalid coordinator result acceptance: " + "; ".join(acceptance_errors)
            )
        receipt_ids.append(receipt["result_receipt_sha256"])
        acceptance_ids.append(acceptance["result_acceptance_sha256"])
        for row in receipt["results"]:
            existing = accepted.get(row["work_id"])
            identity = (row["artifact_sha256"], row["artifact_bytes"],
                        row["canonical_receipt_sha256"])
            if existing is not None:
                old = (existing["artifact_sha256"], existing["artifact_bytes"],
                       existing["canonical_receipt_sha256"])
                if old != identity:
                    raise TransportError(
                        f"conflicting duplicate remote result: {row['work_id']}"
                    )
                continue
            accepted[row["work_id"]] = row
    if set(accepted) != set(target_work_ids):
        raise TransportError(
            f"remote result coverage differs: missing={len(set(target_work_ids)-set(accepted))}"
        )
    results = [accepted[work_id] for work_id in sorted(accepted)]
    doc: dict[str, Any] = {
        "schema": RESULT_MERGE_SCHEMA, "created_at": _now(),
        "status": "transport-artifacts-complete-scientific-merge-deferred",
        "target_work_ids_sha256": _hash_value(sorted(target_work_ids)),
        "source_result_receipts": sorted(set(receipt_ids)),
        "coordinator_result_acceptances": sorted(set(acceptance_ids)),
        "result_count": len(results), "results": results,
        "results_sha256": _hash_value(results),
        "scientific_evidence_validated_by_parent_contract": False,
        "quality_claims_permitted": False, "source_deletion_permitted": False,
    }
    return _seal(doc, digest_field="result_merge_sha256",
                 key_id=coordinator_key_id, key=coordinator_key)


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mxfp4._atomic_json(path, doc)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    args = parser.parse_args(argv)
    try:
        doc = build_requirements_packet()
        _write_json(args.requirements, doc)
        print(json.dumps({
            "status": "ok", "output": str(args.requirements.resolve()),
            "requirements_sha256": doc["requirements_sha256"],
            "remote_host_count": 0, "distributed_execution_enabled": False,
            "local_only_fallback": True,
        }, indent=2, sort_keys=True))
        return 0
    except (TransportError, OSError, ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
