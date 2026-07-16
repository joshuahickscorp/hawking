"""Cheap exact fixtures for higher-tier operator-authority tests."""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Iterator

import doctor_v5_higher_tier_authority as authority
import doctor_v5_higher_tier_scaffold as higher
import physical_counter_attestation


VERIFIED = lambda _envelope, _payload: (True, "verified")


def _artifact(path: Path, raw: bytes) -> dict:
    path.write_bytes(raw)
    return higher._artifact(path)


def _binding(value: dict, field: str = "binding_sha256") -> dict:
    value[field] = higher._hash_value(value)
    return value


def make_core_manifest(root: Path, *, remote: bool = True) -> dict:
    logical = 121_000_000_001
    if not remote:
        raise ValueError("valid >120B fixtures use cheap versioned remote-range authority")
    stored_bytes = (logical + 79) // 80  # reviewed tq-0.1bpw, exact ceil(bits/8)
    range_sha = hashlib.sha256(b"declared-versioned-remote-range").hexdigest()
    tensor = {
        "tensor_key": "fixture.weight", "role": "model_parameter",
        "logical_dtype": "BF16", "storage_encoding": "tq-0.1bpw",
        "shape": [logical], "logical_parameters": logical,
        "stored_parameters": logical, "packing_overhead_bytes": 0,
        "stored_bytes": stored_bytes,
        "source_id": "shard-0", "absolute_byte_range": [0, stored_bytes],
        "range_sha256": range_sha,
    }
    tensor_layout = authority.canonical_sha256([tensor])
    range_root = authority.canonical_sha256([{
        "tensor_key": tensor["tensor_key"], "source_id": tensor["source_id"],
        "absolute_byte_range": tensor["absolute_byte_range"],
        "stored_bytes": tensor["stored_bytes"], "range_sha256": range_sha,
    }])
    parameter_doc = authority._stamp({
        "schema": authority.PARAMETER_AUTHORITY_SCHEMA,
        "model": {
            "label": "Fixture-121B", "hf_id_or_source_id": "fixture/121b",
            "family": "fixture-dense", "architecture_kind": "dense",
        },
        "tensors": [tensor], "logical_parameters": logical,
        "stored_parameters": logical, "tensor_count": 1,
        "tensor_layout_sha256": tensor_layout,
        "parameter_ranges_sha256": range_root,
    }, "authority_sha256")
    parameter_path = root / "parameter-authority.json"
    parameter_path.write_text(
        json.dumps(parameter_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    parameter = {
        "artifact": higher._artifact(parameter_path),
        "logical_parameters": logical, "stored_parameters": logical,
        "tensor_count": 1, "tensor_layout_sha256": tensor_layout,
        "parameter_ranges_sha256": range_root, "reviewed": True,
        "authority_sha256": parameter_doc["authority_sha256"],
    }
    architecture = _binding({
        "adapter_id": "fixture-dense-v1",
        "artifact": _artifact(root / "architecture-adapter.py", b"# reviewed adapter\n"),
        "abi_sha256": "7" * 64, "family": "fixture-dense",
        "architecture_kind": "dense", "reviewed": True, "default_off": True,
    })
    tokenizer_artifact = _artifact(root / "tokenizer.json", b"tokenizer\n")
    tokenizer = _binding({
        "artifact": tokenizer_artifact,
        "tokenizer_sha256": tokenizer_artifact["sha256"],
        "chat_template_sha256": "8" * 64,
        "special_tokens_sha256": "9" * 64, "reviewed": True,
    })
    lifecycle = _binding({
        "artifact": _artifact(root / "lifecycle.json", b"lifecycle\n"),
        "source_deletion_permitted": False, "immutable": True,
        "rollback_cas_sha256": "a" * 64,
    })
    transport = _binding({
        "artifact": _artifact(root / "transport.json", b"transport\n"),
        "source_deletion_permitted": False, "immutable": True,
        "rollback_cas_sha256": "b" * 64,
    })
    source: dict = {}
    source_range: dict = {
        "source_id": "shard-0", "absolute_byte_range": [0, stored_bytes],
        "range_role": "tensor-payload", "range_sha256": range_sha,
        "tensor_keys": ["fixture.weight"],
    }
    if remote:
        uri, version = "object://fixture/immutable-shard", "version-0001"
        source_authority = authority._stamp({
            "schema": "hawking.doctor_v5_remote_source_authority.v1",
            "source_id": "shard-0", "uri": uri, "immutable_version": version,
            "bytes": stored_bytes, "sha256": range_sha, "reviewed": True,
        }, "authority_sha256")
        source.clear()
        source.update({
            "source_id": "shard-0", "transport": "object_range", "uri": uri,
            "immutable_version": version, "bytes": stored_bytes,
            "sha256": range_sha, "immutable_content": True,
            "range_reads_supported": True,
            "authority_artifact": _artifact(
                root / "remote-source-authority.json",
                (json.dumps(source_authority, sort_keys=True) + "\n").encode(),
            ),
        })
        range_receipt = authority._stamp({
            "schema": "hawking.doctor_v5_remote_range_receipt.v1",
            "source_id": "shard-0", "uri": uri, "immutable_version": version,
            "absolute_byte_range": [0, stored_bytes],
            "range_sha256": range_sha, "range_bytes": stored_bytes,
            "verified": True,
        }, "receipt_sha256")
        source_range["range_receipt_artifact"] = _artifact(
            root / "remote-range-receipt.json",
            (json.dumps(range_receipt, sort_keys=True) + "\n").encode(),
        )
    manifest = {
        "schema": higher.SOURCE_MANIFEST_SCHEMA,
        "created_at": "2026-07-14T22:00:00+00:00", "status": "sealed",
        "model": {
            "label": "Fixture-121B", "hf_id_or_source_id": "fixture/121b",
            "family": "fixture-dense", "architecture_kind": "dense",
            "logical_parameters": logical,
            "parameter_authority_sha256": parameter_doc["authority_sha256"],
            "parameter_authority": parameter,
        },
        "sources": [source],
        "work_units": [{
            "unit_id": "tensor-batch-0", "kind": "dense_tensor_batch",
            "logical_parameters": logical,
            "estimated_peak_resident_bytes": 1_000_000_000,
            "threads_per_lane": 8, "source_ranges": [source_range],
        }],
        "coverage": {
            "work_unit_count": 1, "logical_parameters": logical,
            "all_model_tensors_assigned_exactly_once": True,
            "tensor_count": 1, "tensor_layout_sha256": tensor_layout,
        },
        "architecture_adapter": architecture, "tokenizer_binding": tokenizer,
        "lifecycle_manifest": lifecycle, "transport_manifest": transport,
        "source_deletion_permitted": False,
    }
    manifest["manifest_sha256"] = higher._hash_value(manifest)
    return manifest


def fake_operator_seal(root: Path, manifest: dict) -> dict:
    issued = time.time_ns()
    attestation = authority.build_manifest_attestation(
        manifest, issued_at_unix_ns=issued, valid_seconds=3600,
    )
    signature = root / "fake-higher-tier.sshsig"
    signature.write_bytes(b"synthetic-signature-for-structural-integration-only")
    envelope = authority._stamp({
        "schema": authority.ENVELOPE_SCHEMA, "attestation": attestation,
        "signer_identity": authority.SIGNER_IDENTITY,
        "signature_namespace": authority.SSHSIG_NAMESPACE,
        "allowed_signers": authority._allowed_signers_identity(),
        "detached_signature": physical_counter_attestation.file_identity(signature),
    }, "envelope_sha256")
    return authority.build_sealed_manifest(
        manifest=manifest, signed_attestation=envelope,
        sealed_at_unix_ns=issued + 1, signature_verifier=VERIFIED,
    )


@contextlib.contextmanager
def ephemeral_trust(root: Path) -> Iterator[Path]:
    key = root / "ephemeral-operator"
    process = subprocess.run(
        [str(authority.SSH_KEYGEN), "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=20, check=False, shell=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"ephemeral ssh-keygen failed: {process.stderr}")
    public = key.with_suffix(".pub").read_text(encoding="utf-8").split()
    allowed = root / "allowed_signers"
    allowed.write_text(
        f"{authority.SIGNER_IDENTITY} {public[0]} {public[1]}\n", encoding="utf-8",
    )
    old = (
        authority.DEFAULT_ALLOWED_SIGNERS,
        authority.PINNED_ALLOWED_SIGNERS_SHA256,
        authority.PINNED_PUBLIC_KEY_BLOB_SHA256,
    )
    authority.DEFAULT_ALLOWED_SIGNERS = allowed
    authority.PINNED_ALLOWED_SIGNERS_SHA256 = hashlib.sha256(allowed.read_bytes()).hexdigest()
    authority.PINNED_PUBLIC_KEY_BLOB_SHA256 = hashlib.sha256(
        f"{public[0]} {public[1]}".encode("ascii")
    ).hexdigest()
    try:
        yield key
    finally:
        (
            authority.DEFAULT_ALLOWED_SIGNERS,
            authority.PINNED_ALLOWED_SIGNERS_SHA256,
            authority.PINNED_PUBLIC_KEY_BLOB_SHA256,
        ) = old
