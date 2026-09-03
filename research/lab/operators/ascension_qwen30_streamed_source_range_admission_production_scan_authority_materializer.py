#!/usr/bin/env python3
"""Materialize a sealed, non-executing Q30 production hash-scan authority.

The authority binds already-reviewed metadata, semantics, runtime-preflight,
scanner-interface, binary, and resource records.  It deliberately has no
source-root argument and no process/lease/GPU surface.  A later distinct lease
is still mandatory before a bounded 16-shard / 18,867-range scan could occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
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

SCHEMA = outer.PRODUCTION_AUTHORITY_SCHEMA
STATUS = outer.PRODUCTION_AUTHORITY_STATUS
METADATA_SCHEMA = "hawking.ascension.qwen30_streamed_oracle_metadata_only_range_map_authority.v1"
METADATA_STATUS = "PREPARED_QWEN30_STREAMED_ORACLE_SOURCE_RANGE_MAP_AUTHORITY_NOT_EXECUTED"
SEMANTICS_SCHEMA = (
    "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_semantics_attester.v1"
)
SEMANTICS_STATUS = (
    "PREPARED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_NOT_EXECUTED"
)
RUNTIME_SCHEMA = "hawking.ascension.qwen30_streamed_source_runtime_range_admission_producer_preflight.v1"
RUNTIME_STATUS = "PREPARED_QWEN30_STREAMED_SOURCE_RUNTIME_RANGE_ADMISSION_PRODUCER_NOT_EXECUTED"
MAX_METADATA_BYTES = 64 * 1024 * 1024


class ProductionAuthorityMaterializationError(RuntimeError):
    """Inputs do not safely bind a prepared one-shot production authority."""


@dataclass(frozen=True)
class RawDocument:
    path: Path
    document: dict[str, Any]
    raw_document_sha256: str
    canonical_document_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionAuthorityMaterializationError(f"{label} must be an object")
    return dict(value)


def _text(value: object, *, label: str, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ProductionAuthorityMaterializationError(f"{label} must be non-empty text")
    if sha256 and (
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProductionAuthorityMaterializationError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProductionAuthorityMaterializationError(f"{label} must be a nonnegative integer")
    return value


def _regular_json(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.suffix != ".json":
        raise ProductionAuthorityMaterializationError(f"{label} must be an absolute .json path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductionAuthorityMaterializationError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProductionAuthorityMaterializationError(f"{label} must be a regular non-symlink file")
    if metadata.st_size <= 0 or metadata.st_size > MAX_METADATA_BYTES:
        raise ProductionAuthorityMaterializationError(f"{label} has invalid metadata size")
    return path.resolve(strict=True)


def _raw_json(path: Path, *, label: str) -> RawDocument:
    clean = _regular_json(path, label=label)
    try:
        raw = clean.read_bytes()
        document = _mapping(json.loads(raw), label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionAuthorityMaterializationError(f"cannot read {label}: {exc}") from exc
    return RawDocument(
        path=clean,
        document=document,
        raw_document_sha256=_sha256_bytes(raw),
        canonical_document_sha256=_sha256_bytes(_canonical_json(document)),
    )


def _reject_fixture_identity(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"schema", "status"} and isinstance(child, str):
                if "fixture" in child.lower() or "synthetic" in child.lower():
                    raise ProductionAuthorityMaterializationError(
                        f"{label}.{key} carries fixture-only identity"
                    )
            if key in {"fixture_only", "synthetic_fixture_only", "production_adapter_forbidden"} and child is True:
                raise ProductionAuthorityMaterializationError(f"{label}.{key} is fixture-only")
            _reject_fixture_identity(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_fixture_identity(child, label=f"{label}[{index}]")


def _evidence_raw(document: RawDocument) -> dict[str, Any]:
    return {
        "path": str(document.path),
        "raw_document_sha256": document.raw_document_sha256,
        "canonical_document_sha256": document.canonical_document_sha256,
        "seal_sha256": None,
    }


def _evidence_sealed(document: outer.Document) -> dict[str, Any]:
    return {
        "path": str(document.path),
        "raw_document_sha256": document.raw_document_sha256,
        "canonical_document_sha256": _sha256_bytes(_canonical_json(document.document)),
        "seal_sha256": document.seal_sha256,
    }


def _require_evidence(
    value: object, *, expected: RawDocument | outer.Document, sealed: bool, label: str
) -> None:
    evidence = _mapping(value, label=label)
    raw = _text(evidence.get("raw_document_sha256"), label=f"{label}.raw", sha256=True)
    canonical = _text(
        evidence.get("canonical_document_sha256"), label=f"{label}.canonical", sha256=True
    )
    expected_canonical = _sha256_bytes(_canonical_json(expected.document))
    if raw != expected.raw_document_sha256 or canonical != expected_canonical:
        raise ProductionAuthorityMaterializationError(f"{label} document identity drifted")
    recorded_seal = evidence.get("seal_sha256")
    if sealed:
        if recorded_seal != expected.seal_sha256:
            raise ProductionAuthorityMaterializationError(f"{label} seal drifted")
    elif recorded_seal is not None:
        raise ProductionAuthorityMaterializationError(f"{label} unexpectedly claims a seal")


def _validate_metadata(document: RawDocument) -> tuple[str, str]:
    _reject_fixture_identity(document.document, label="metadata range authority")
    authority = _mapping(document.document.get("authority"), label="metadata authority")
    if authority.get("schema") != METADATA_SCHEMA or authority.get("status") != METADATA_STATUS:
        raise ProductionAuthorityMaterializationError("metadata authority schema/status drifted")
    content_sha = _text(
        document.document.get("authority_content_sha256"),
        label="metadata authority content hash",
        sha256=True,
    )
    if content_sha != _sha256_bytes(_canonical_json(authority)):
        raise ProductionAuthorityMaterializationError("metadata authority content hash drifted")
    source = _mapping(authority.get("source"), label="metadata source")
    revision = _text(
        source.get("source_revision"), label="metadata source revision", sha256=False
    )
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ProductionAuthorityMaterializationError("metadata source revision must be lowercase SHA-1")
    if _integer(source.get("source_shard_count"), label="metadata shard count") != outer.SOURCE_SHARDS:
        raise ProductionAuthorityMaterializationError("metadata shard count drifted")
    if _integer(source.get("source_tensor_count"), label="metadata tensor count") != outer.SOURCE_TENSORS:
        raise ProductionAuthorityMaterializationError("metadata tensor count drifted")
    return content_sha, revision


def _validate_semantics(
    document: RawDocument, *, metadata: RawDocument, metadata_content_sha256: str, revision: str
) -> None:
    _reject_fixture_identity(document.document, label="independent semantics attester")
    if document.document.get("schema") != SEMANTICS_SCHEMA or document.document.get("status") != SEMANTICS_STATUS:
        raise ProductionAuthorityMaterializationError("semantics attester schema/status drifted")
    source = _mapping(document.document.get("pinned_source_binding"), label="semantics source")
    if source.get("source_revision") != revision:
        raise ProductionAuthorityMaterializationError("semantics source revision drifted")
    consumed = _mapping(document.document.get("consumed_metadata_contracts"), label="semantics contracts")
    range_authority = _mapping(consumed.get("range_authority"), label="semantics range authority")
    if range_authority.get("document_sha256") != metadata.raw_document_sha256:
        raise ProductionAuthorityMaterializationError("semantics metadata raw binding drifted")
    if range_authority.get("authority_content_sha256") != metadata_content_sha256:
        raise ProductionAuthorityMaterializationError("semantics metadata content binding drifted")


def _validate_runtime(
    document: outer.Document, *, metadata: RawDocument, semantics: RawDocument, content_sha: str
) -> None:
    _reject_fixture_identity(document.document, label="runtime producer")
    root = document.document
    if root.get("schema") != RUNTIME_SCHEMA or root.get("status") != RUNTIME_STATUS:
        raise ProductionAuthorityMaterializationError("runtime producer schema/status drifted")
    for field in ("prepared",):
        if root.get(field) is not True:
            raise ProductionAuthorityMaterializationError(f"runtime producer.{field} must be true")
    for field in ("runtime_admission_earned", "source_payload_validation_executed"):
        if root.get(field) is not False:
            raise ProductionAuthorityMaterializationError(f"runtime producer.{field} must be false")
    metadata_binding = _mapping(root.get("sealed_metadata_authority_binding"), label="runtime metadata")
    metadata_pointer = _mapping(metadata_binding.get("metadata_range_authority"), label="runtime metadata pointer")
    if metadata_pointer.get("raw_document_sha256") != metadata.raw_document_sha256:
        raise ProductionAuthorityMaterializationError("runtime metadata pointer drifted")
    if metadata_binding.get("authority_content_sha256") != content_sha:
        raise ProductionAuthorityMaterializationError("runtime metadata content binding drifted")
    semantics_binding = _mapping(root.get("metadata_semantics_binding"), label="runtime semantics")
    semantics_pointer = _mapping(
        semantics_binding.get("operator_semantics_attester"), label="runtime semantics pointer"
    )
    if semantics_pointer.get("raw_document_sha256") != semantics.raw_document_sha256:
        raise ProductionAuthorityMaterializationError("runtime semantics pointer drifted")


def _validate_interface_inputs(
    document: outer.Document,
    *,
    metadata: RawDocument,
    semantics: RawDocument,
    runtime: outer.Document,
    content_sha: str,
) -> None:
    outer._validate_interface(document)
    inputs = _mapping(document.document.get("input_authorities"), label="production interface inputs")
    _require_evidence(
        inputs.get("metadata_range_authority"),
        expected=metadata,
        sealed=False,
        label="interface metadata",
    )
    _require_evidence(
        inputs.get("independent_non_fixture_semantics_attester"),
        expected=semantics,
        sealed=False,
        label="interface semantics",
    )
    _require_evidence(
        inputs.get("runtime_admission_producer_authority"),
        expected=runtime,
        sealed=True,
        label="interface runtime producer",
    )
    if inputs.get("metadata_authority_content_sha256") != content_sha:
        raise ProductionAuthorityMaterializationError("interface metadata content binding drifted")


def _validate_production_chain(
    *,
    bootstrap_preflight_path: Path,
    bootstrap_binary_path: Path,
    bootstrap_resource_path: Path,
    production_binary_path: Path,
    production_resource_path: Path,
) -> tuple[outer.Document, outer.Document]:
    try:
        preflight, bootstrap_binary, bootstrap_resource, _legacy_binary, _legacy_window = (
            outer._validate_existing_bootstrap_chain(
                preflight_path=bootstrap_preflight_path,
                binary_path=bootstrap_binary_path,
                resource_path=bootstrap_resource_path,
            )
        )
        binary = outer._sealed(production_binary_path, label="production binary")
        outer._validate_production_binary(
            binary,
            preflight=preflight,
            bootstrap_binary=bootstrap_binary,
            resource=bootstrap_resource,
        )
        resource = outer._sealed(production_resource_path, label="production resource")
        outer._validate_production_resource(
            resource, production_binary=binary, bootstrap_resource=bootstrap_resource
        )
        return binary, resource
    except outer.ProductionScanOuterError as exc:
        raise ProductionAuthorityMaterializationError(
            f"production binary/resource chain is invalid: {exc}"
        ) from exc


def build_production_authority(
    *,
    metadata_path: Path,
    semantics_path: Path,
    runtime_authority_path: Path,
    interface_path: Path,
    bootstrap_preflight_path: Path,
    bootstrap_binary_path: Path,
    bootstrap_resource_path: Path,
    production_binary_path: Path,
    production_resource_path: Path,
    nonce_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Return a sealed prepared authority, never a lease or a scan result."""
    metadata = _raw_json(metadata_path, label="metadata range authority")
    semantics = _raw_json(semantics_path, label="independent semantics attester")
    try:
        runtime = outer._sealed(runtime_authority_path, label="runtime producer authority")
        interface = outer._sealed(interface_path, label="production interface authority")
    except outer.ProductionScanOuterError as exc:
        raise ProductionAuthorityMaterializationError(str(exc)) from exc

    content_sha, revision = _validate_metadata(metadata)
    _validate_semantics(
        semantics,
        metadata=metadata,
        metadata_content_sha256=content_sha,
        revision=revision,
    )
    _validate_runtime(runtime, metadata=metadata, semantics=semantics, content_sha=content_sha)
    _validate_interface_inputs(
        interface,
        metadata=metadata,
        semantics=semantics,
        runtime=runtime,
        content_sha=content_sha,
    )
    production_binary, production_resource = _validate_production_chain(
        bootstrap_preflight_path=bootstrap_preflight_path,
        bootstrap_binary_path=bootstrap_binary_path,
        bootstrap_resource_path=bootstrap_resource_path,
        production_binary_path=production_binary_path,
        production_resource_path=production_resource_path,
    )
    nonce_input = os.urandom(32) if nonce_bytes is None else nonce_bytes
    if not isinstance(nonce_input, bytes) or len(nonce_input) < 16:
        raise ProductionAuthorityMaterializationError("one-shot authority nonce must contain >=16 bytes")

    return seal(
        {
            "schema": SCHEMA,
            "status": STATUS,
            "recorded_at": _utc_now(),
            "prepared": True,
            "fresh_for_this_exact_scan": True,
            "one_shot": True,
            "non_inference_hash_scan_only": True,
            "source_root_open_only_after_all_authorities_validate": True,
            "fixture_only": False,
            "synthetic_fixture_only": False,
            "production_adapter_forbidden": False,
            "source_teacher_or_logits_authorized": False,
            "model_gpu_server_hcli_or_tps_authorized": False,
            "lease_consumed": False,
            "immutable_bindings": {
                "interface_authority": _evidence_sealed(interface),
                "production_binary": _evidence_sealed(production_binary),
                "production_resource_admission": _evidence_sealed(production_resource),
                "metadata_range_authority": _evidence_raw(metadata),
                "independent_semantics_attester": _evidence_raw(semantics),
                "runtime_admission_producer_authority": _evidence_sealed(runtime),
                "metadata_authority_content_sha256": content_sha,
            },
            "geometry": {
                "source_shards": outer.SOURCE_SHARDS,
                "source_tensors": outer.SOURCE_TENSORS,
                "maximum_positioned_read_bytes": outer.MAX_POSITIONED_READ_BYTES,
                "maximum_live_raw_bf16_windows": 1,
            },
            "exact_scan_nonce_sha256": _sha256_bytes(nonce_input),
            "execution_boundary": {
                "source_root_argument_or_stat_performed": False,
                "source_payload_opened": False,
                "source_model_loaded": False,
                "source_teacher_or_logits_executed": False,
                "native_phase_started": False,
                "gpu_server_hcli_or_tps_action": False,
                "lease_issued_or_consumed": False,
                "child_started": False,
            },
            "claim_boundary": "Prepared non-fixture production hash-scan authority only. It is not a lease and does not open a source root, run a hash scan, earn a map/coverage/capture, attest source-teacher semantics, load a model, or authorize native/GPU/server/HCLI/TPS/TG/tournament work.",
        }
    )


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise ProductionAuthorityMaterializationError(
            "--out must be a new absolute path below an existing parent"
        )
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
    parser.add_argument("--metadata-range", type=Path, required=True)
    parser.add_argument("--semantics-attester", type=Path, required=True)
    parser.add_argument("--runtime-authority", type=Path, required=True)
    parser.add_argument("--interface-authority", type=Path, required=True)
    parser.add_argument("--bootstrap-preflight", type=Path, required=True)
    parser.add_argument("--bootstrap-binary", type=Path, required=True)
    parser.add_argument("--bootstrap-resource", type=Path, required=True)
    parser.add_argument("--production-binary", type=Path, required=True)
    parser.add_argument("--production-resource", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = build_production_authority(
            metadata_path=args.metadata_range,
            semantics_path=args.semantics_attester,
            runtime_authority_path=args.runtime_authority,
            interface_path=args.interface_authority,
            bootstrap_preflight_path=args.bootstrap_preflight,
            bootstrap_binary_path=args.bootstrap_binary,
            bootstrap_resource_path=args.bootstrap_resource,
            production_binary_path=args.production_binary,
            production_resource_path=args.production_resource,
        )
        _write_new(args.out, document)
    except ProductionAuthorityMaterializationError as exc:
        print(f"Q30 production hash-scan authority refused: {exc}")
        return 2
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "status": document["status"],
                "seal_sha256": document["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
