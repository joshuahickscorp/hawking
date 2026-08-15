#!/usr/bin/env python3
"""Validate a completed Q30 production hash scan for a later teacher phase.

This is deliberately a CPU/file-only provenance bridge.  It reads sealed JSON
records produced by the production bounded hash scan and creates one new,
sealed *reservation* record.  It does not inspect a source root, issue a
lease, start a child, load a model, or run teacher/native/GPU/server/HCLI/TPS
work.

The bridge intentionally does not manufacture either of the records which the
existing source-teacher child requires before it can open a source root:

* ``hawking.ascension.qwen30_streamed_source_teacher_runtime_range_admission.v1``
* ``hawking.ascension.qwen30_streamed_source_teacher_dual_attestation_runtime_admission.v1``

Those remain future, separately sealed inputs.  In particular, the bridge
records the current admission-before-open cycle as unresolved rather than
claiming that an immutable post-hash-map result has solved it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lab.receipts import SealIntegrityError, seal, verify


SCHEMA = "hawking.ascension.qwen30_streamed_source_teacher_post_hash_map_bridge.v1"
STATUS = "PREPARED_QWEN30_STREAMED_SOURCE_TEACHER_POST_HASH_MAP_BRIDGE_NOT_EXECUTED"

RANGE_AUTHORITY_SCHEMA = (
    "hawking.ascension.qwen30_streamed_oracle_metadata_only_range_map_authority.v1"
)
RANGE_AUTHORITY_STATUS = (
    "PREPARED_QWEN30_STREAMED_ORACLE_SOURCE_RANGE_MAP_AUTHORITY_NOT_EXECUTED"
)
SEMANTICS_SCHEMA = (
    "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_semantics_attester.v1"
)
SEMANTICS_STATUS = (
    "PREPARED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_NOT_EXECUTED"
)
RUNTIME_PRODUCER_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_runtime_range_admission_producer_preflight.v1"
)
RUNTIME_PRODUCER_STATUS = (
    "PREPARED_QWEN30_STREAMED_SOURCE_RUNTIME_RANGE_ADMISSION_PRODUCER_NOT_EXECUTED"
)

FLAT_MAP_SCHEMA = "hawking.ascension.qwen30_source_bf16_range_map.v1"
HASH_COVERAGE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_production_hash_coverage_attestation.v1"
)
HASH_COVERAGE_STATUS = (
    "EARNED_QWEN30_STREAMED_SOURCE_PRODUCTION_HASH_COVERAGE_ATTESTED_NOT_SOURCE_TEACHER"
)
CAPTURE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_capture.v1"
)
CAPTURE_STATUS = (
    "CAPTURED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_NOT_SOURCE_TEACHER"
)
OUTER_TERMINAL_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_outer_terminal.v1"
)
OUTER_TERMINAL_STATUS = (
    "CAPTURED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_OUTER_TERMINAL_NOT_SOURCE_TEACHER"
)
RELEASE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_quiet_lease_release.v1"
)
RELEASE_STATUS = (
    "RELEASED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_LEASE_AFTER_OUTER_TERMINAL"
)
REPLAY_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_replay_reservation.v1"
)
REPLAY_STATUS = (
    "RESERVED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_ONE_SHOT_NOT_SPAWNED"
)

RUNTIME_ADMISSION_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_teacher_runtime_range_admission.v1"
)
RUNTIME_ADMISSION_STATUS = (
    "EARNED_QWEN30_STREAMED_SOURCE_TEACHER_RUNTIME_RANGE_ADMISSION_NO_MODEL_RESIDENCY"
)
DUAL_BRIDGE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_teacher_dual_attestation_runtime_admission.v1"
)
DUAL_BRIDGE_STATUS = (
    "PREPARED_QWEN30_STREAMED_SOURCE_TEACHER_DUAL_ATTESTATION_RUNTIME_ADMISSION_NOT_EXECUTED"
)
OPERATOR_ATTESTATION_SCHEMA = (
    "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_execution_attestation.v1"
)
OPERATOR_ATTESTATION_STATUS = (
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_ATTESTED"
)
READER_ATTESTATION_SCHEMA = (
    "hawking.ascension.qwen30_layer_streamed_source_bf16_exact_semantics_attestation.v1"
)
READER_ATTESTATION_STATUS = (
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_BF16_EXACT_SEMANTICS_ATTESTED"
)

MODEL_ID = "Qwen3-Coder-30B-A3B-Instruct"
SOURCE_SHARDS = 16
SOURCE_TENSORS = 18_867
MAX_WINDOW_BYTES = 1_048_576
SOURCE_LAYERS = 48
SOURCE_FORWARDS = 370


class PostHashMapBridgeError(ValueError):
    """A supplied post-hash-map record does not earn a teacher reservation."""


@dataclass(frozen=True)
class Document:
    path: Path
    value: dict[str, Any]
    raw_document_sha256: str
    canonical_document_sha256: str
    seal_sha256: str | None

    def evidence(self) -> dict[str, str | None]:
        return {
            "path": str(self.path),
            "raw_document_sha256": self.raw_document_sha256,
            "canonical_document_sha256": self.canonical_document_sha256,
            "seal_sha256": self.seal_sha256,
        }


@dataclass(frozen=True)
class SourceBinding:
    model_id: str
    revision: str
    source_index_sha256: str
    authority_content_sha256: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json(value))


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PostHashMapBridgeError(f"{label} must be a JSON object")
    return value


def _list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PostHashMapBridgeError(f"{label} must be a JSON array")
    return value


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PostHashMapBridgeError(f"{label} must be a non-empty string")
    return value


def _sha(value: object, *, label: str) -> str:
    value = _text(value, label=label)
    if not _is_sha256(value):
        raise PostHashMapBridgeError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PostHashMapBridgeError(f"{label} must be an integer >= {minimum}")
    return value


def _boolean(value: object, *, label: str, expected: bool) -> None:
    if value is not expected:
        raise PostHashMapBridgeError(f"{label} must be {expected}")


def _field(root: Mapping[str, Any], key: str, *, label: str) -> Any:
    if key not in root:
        raise PostHashMapBridgeError(f"{label}.{key} is required")
    return root[key]


def _schema_status(
    root: Mapping[str, Any], *, schema: str, status: str, label: str
) -> None:
    if _text(_field(root, "schema", label=label), label=f"{label}.schema") != schema:
        raise PostHashMapBridgeError(f"{label} schema drifted")
    if _text(_field(root, "status", label=label), label=f"{label}.status") != status:
        raise PostHashMapBridgeError(f"{label} status drifted")


def _reject_fixture_identity(value: object, *, label: str) -> None:
    """Reject fixture/synthetic evidence before accepting any provenance edge."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"schema", "status"} and isinstance(child, str):
                lowered = child.lower()
                if "fixture" in lowered or "synthetic" in lowered:
                    raise PostHashMapBridgeError(
                        f"{label}.{key} carries forbidden fixture-only identity {child!r}"
                    )
            if key in {
                "fixture_only",
                "synthetic_fixture_only",
                "production_adapter_forbidden",
            } and child is True:
                raise PostHashMapBridgeError(
                    f"{label}.{key} marks fixture-only or production-forbidden evidence"
                )
            _reject_fixture_identity(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_fixture_identity(child, label=f"{label}[{index}]")


def _read_document(path: Path, *, label: str, sealed: bool) -> Document:
    if not path.is_absolute() or path.suffix != ".json":
        raise PostHashMapBridgeError(f"{label} must be an absolute JSON path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise PostHashMapBridgeError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise PostHashMapBridgeError(f"{label} must be a regular non-symlink file")
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise PostHashMapBridgeError(f"cannot read {label}: {exc}") from exc
    root = _mapping(value, label=label)
    if not isinstance(value, dict):
        # ``json.loads`` normally produces a dict here, but retain the exact
        # type for an immutable Document object and clear error wording.
        raise PostHashMapBridgeError(f"{label} must be a JSON object")
    _reject_fixture_identity(root, label=label)
    seal_sha256: str | None = None
    if sealed:
        try:
            verified = verify(root, label=label)
        except SealIntegrityError as exc:
            raise PostHashMapBridgeError(str(exc)) from exc
        seal_sha256 = _sha(verified.get("seal_sha256"), label=f"{label}.seal_sha256")
    elif "seal_sha256" in root:
        raise PostHashMapBridgeError(f"{label} is legacy-unsealed and must not carry a seal")
    return Document(
        path=path.resolve(strict=True),
        value=dict(root),
        raw_document_sha256=_sha256_bytes(raw),
        canonical_document_sha256=_sha256_json(root),
        seal_sha256=seal_sha256,
    )


def _pointer(
    value: object,
    *,
    expected: Document,
    label: str,
    canonical_required: bool = True,
) -> None:
    pointer = _mapping(value, label=label)
    observed_path = _text(_field(pointer, "path", label=label), label=f"{label}.path")
    if observed_path != str(expected.path):
        raise PostHashMapBridgeError(f"{label} path does not bind the supplied document")
    if (
        _sha(
            _field(pointer, "raw_document_sha256", label=label),
            label=f"{label}.raw_document_sha256",
        )
        != expected.raw_document_sha256
    ):
        raise PostHashMapBridgeError(f"{label} raw identity drifted")
    if canonical_required:
        if (
            _sha(
                _field(pointer, "canonical_document_sha256", label=label),
                label=f"{label}.canonical_document_sha256",
            )
            != expected.canonical_document_sha256
        ):
            raise PostHashMapBridgeError(f"{label} canonical identity drifted")
    if expected.seal_sha256 is None:
        if pointer.get("seal_sha256") is not None:
            raise PostHashMapBridgeError(f"{label} unexpectedly claims a seal")
    elif (
        _sha(_field(pointer, "seal_sha256", label=label), label=f"{label}.seal_sha256")
        != expected.seal_sha256
    ):
        raise PostHashMapBridgeError(f"{label} seal identity drifted")


def _same_pointer(
    left: object,
    right: object,
    *,
    label: str,
    canonical_required: bool,
) -> None:
    left_mapping = _mapping(left, label=f"{label} left")
    right_mapping = _mapping(right, label=f"{label} right")
    fields = ["path", "raw_document_sha256", "seal_sha256"]
    if canonical_required:
        fields.append("canonical_document_sha256")
    for field in fields:
        if left_mapping.get(field) != right_mapping.get(field):
            raise PostHashMapBridgeError(f"{label}.{field} drifted")


def _validate_range_authority(document: Document) -> SourceBinding:
    root = document.value
    authority = _mapping(_field(root, "authority", label="range authority"), label="range authority")
    _schema_status(
        authority,
        schema=RANGE_AUTHORITY_SCHEMA,
        status=RANGE_AUTHORITY_STATUS,
        label="range authority",
    )
    content_sha = _sha(
        _field(root, "authority_content_sha256", label="range authority"),
        label="range authority.authority_content_sha256",
    )
    if _sha256_json(authority) != content_sha:
        raise PostHashMapBridgeError("range authority content SHA does not bind authority")
    source = _mapping(_field(authority, "source", label="range authority"), label="range source")
    model_id = _text(_field(source, "model_id", label="range source"), label="range source.model_id")
    if model_id != MODEL_ID:
        raise PostHashMapBridgeError("range authority source model drifted")
    revision = _text(
        _field(source, "source_revision", label="range source"), label="range source.source_revision"
    )
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise PostHashMapBridgeError("range authority source revision must be lowercase Git SHA")
    if _integer(
        _field(source, "source_shard_count", label="range source"),
        label="range source.source_shard_count",
    ) != SOURCE_SHARDS or _integer(
        _field(source, "source_tensor_count", label="range source"),
        label="range source.source_tensor_count",
    ) != SOURCE_TENSORS:
        raise PostHashMapBridgeError("range authority source geometry drifted")
    source_index = _mapping(
        _field(source, "source_index", label="range source"), label="range source index"
    )
    source_index_sha = _sha(
        _field(source_index, "sha256", label="range source index"),
        label="range source index.sha256",
    )
    if _text(
        _field(source_index, "format", label="range source index"),
        label="range source index.format",
    ) != "huggingface.safetensors.index.json":
        raise PostHashMapBridgeError("range authority index format drifted")
    boundary = _mapping(
        _field(authority, "metadata_access_boundary", label="range authority"),
        label="range authority metadata boundary",
    )
    for key in [
        "gpu_or_metal_invoked",
        "hcli_invoked",
        "lease_requested",
        "mmap_or_memory_map_used",
        "server_started",
        "source_model_instantiated",
        "tensor_payload_hashes_collected",
        "whole_shard_payload_checksum_collected",
    ]:
        _boolean(
            _field(boundary, key, label="range authority metadata boundary"),
            label=f"range authority metadata boundary.{key}",
            expected=False,
        )
    if _integer(
        _field(boundary, "source_tensor_payload_bytes_read", label="range authority metadata boundary"),
        label="range authority metadata boundary.source_tensor_payload_bytes_read",
    ) != 0:
        raise PostHashMapBridgeError("range authority must not have read source tensor payloads")
    scope = _mapping(
        _field(authority, "exact_streamed_oracle_scope", label="range authority"),
        label="range authority scope",
    )
    if _integer(
        _field(scope, "layers", label="range authority scope"),
        label="range authority scope.layers",
    ) != SOURCE_LAYERS or _integer(
        _field(scope, "total_forwards_per_replay_arm", label="range authority scope"),
        label="range authority scope.total_forwards_per_replay_arm",
    ) != SOURCE_FORWARDS:
        raise PostHashMapBridgeError("range authority source-teacher geometry drifted")
    return SourceBinding(
        model_id=model_id,
        revision=revision,
        source_index_sha256=source_index_sha,
        authority_content_sha256=content_sha,
    )


def _validate_semantics(document: Document, source: SourceBinding, range_document: Document) -> None:
    root = document.value
    _schema_status(root, schema=SEMANTICS_SCHEMA, status=SEMANTICS_STATUS, label="semantics attester")
    boundary = _mapping(
        _field(root, "execution_boundary", label="semantics attester"),
        label="semantics execution boundary",
    )
    for key in [
        "gpu_or_metal_invoked",
        "hcli_invoked",
        "lease_requested",
        "server_started",
        "source_inference_executed",
        "source_model_instantiated",
        "source_quality_or_coherence_claim_made",
        "source_safetensors_or_other_weight_path_accepted",
        "source_tensor_payload_opened",
        "tps_or_tg_claim_made",
    ]:
        _boolean(
            _field(boundary, key, label="semantics execution boundary"),
            label=f"semantics execution boundary.{key}",
            expected=False,
        )
    pinned = _mapping(
        _field(root, "pinned_source_binding", label="semantics attester"),
        label="semantics source binding",
    )
    if _text(
        _field(pinned, "source_model_id", label="semantics source binding"),
        label="semantics source binding.source_model_id",
    ) != source.model_id or _text(
        _field(pinned, "source_revision", label="semantics source binding"),
        label="semantics source binding.source_revision",
    ) != source.revision or _sha(
        _field(pinned, "source_index_sha256", label="semantics source binding"),
        label="semantics source binding.source_index_sha256",
    ) != source.source_index_sha256:
        raise PostHashMapBridgeError("semantics source binding drifted")
    consumed = _mapping(
        _field(root, "consumed_metadata_contracts", label="semantics attester"),
        label="semantics consumed metadata",
    )
    range_pointer = _mapping(
        _field(consumed, "range_authority", label="semantics consumed metadata"),
        label="semantics range authority pointer",
    )
    if _sha(
        _field(range_pointer, "document_sha256", label="semantics range authority pointer"),
        label="semantics range authority pointer.document_sha256",
    ) != range_document.raw_document_sha256 or _sha(
        _field(range_pointer, "authority_content_sha256", label="semantics range authority pointer"),
        label="semantics range authority pointer.authority_content_sha256",
    ) != source.authority_content_sha256:
        raise PostHashMapBridgeError("semantics range-authority binding drifted")
    _boolean(
        _field(range_pointer, "source_payload_read_by_this_attester", label="semantics range authority pointer"),
        label="semantics range authority pointer.source_payload_read_by_this_attester",
        expected=False,
    )
    future = _mapping(
        _field(root, "future_exact_execution_attestation", label="semantics attester"),
        label="semantics future execution attestation",
    )
    if _text(
        _field(future, "schema", label="semantics future execution attestation"),
        label="semantics future execution attestation.schema",
    ) != OPERATOR_ATTESTATION_SCHEMA or _text(
        _field(
            future,
            "status_only_after_real_separately_leased_source_execution",
            label="semantics future execution attestation",
        ),
        label="semantics future execution attestation.status",
    ) != OPERATOR_ATTESTATION_STATUS:
        raise PostHashMapBridgeError("semantics future operator-attestation grammar drifted")


def _validate_runtime_producer(
    document: Document,
    source: SourceBinding,
    range_document: Document,
    semantics_document: Document,
) -> None:
    root = document.value
    _schema_status(
        root,
        schema=RUNTIME_PRODUCER_SCHEMA,
        status=RUNTIME_PRODUCER_STATUS,
        label="runtime-admission producer",
    )
    for key, expected in [
        ("prepared", True),
        ("runtime_admission_earned", False),
        ("source_payload_validation_executed", False),
    ]:
        _boolean(
            _field(root, key, label="runtime-admission producer"),
            label=f"runtime-admission producer.{key}",
            expected=expected,
        )
    metadata = _mapping(
        _field(root, "sealed_metadata_authority_binding", label="runtime-admission producer"),
        label="runtime producer metadata binding",
    )
    pointer = _mapping(
        _field(metadata, "metadata_range_authority", label="runtime producer metadata binding"),
        label="runtime producer range pointer",
    )
    if _sha(
        _field(pointer, "raw_document_sha256", label="runtime producer range pointer"),
        label="runtime producer range pointer.raw_document_sha256",
    ) != range_document.raw_document_sha256 or _sha(
        _field(metadata, "authority_content_sha256", label="runtime producer metadata binding"),
        label="runtime producer metadata binding.authority_content_sha256",
    ) != source.authority_content_sha256:
        raise PostHashMapBridgeError("runtime producer range-authority binding drifted")
    if _text(
        _field(metadata, "source_revision", label="runtime producer metadata binding"),
        label="runtime producer metadata binding.source_revision",
    ) != source.revision or _integer(
        _field(metadata, "source_shard_count", label="runtime producer metadata binding"),
        label="runtime producer metadata binding.source_shard_count",
    ) != SOURCE_SHARDS or _integer(
        _field(metadata, "source_tensor_count", label="runtime producer metadata binding"),
        label="runtime producer metadata binding.source_tensor_count",
    ) != SOURCE_TENSORS:
        raise PostHashMapBridgeError("runtime producer source geometry drifted")
    if _integer(
        _field(metadata, "maximum_declared_bf16_row_window_bytes", label="runtime producer metadata binding"),
        label="runtime producer metadata binding.maximum_declared_bf16_row_window_bytes",
    ) != MAX_WINDOW_BYTES:
        raise PostHashMapBridgeError("runtime producer maximum row window drifted")
    semantics = _mapping(
        _field(root, "metadata_semantics_binding", label="runtime-admission producer"),
        label="runtime producer semantics binding",
    )
    semantics_pointer = _mapping(
        _field(semantics, "operator_semantics_attester", label="runtime producer semantics binding"),
        label="runtime producer semantics pointer",
    )
    if _sha(
        _field(semantics_pointer, "raw_document_sha256", label="runtime producer semantics pointer"),
        label="runtime producer semantics pointer.raw_document_sha256",
    ) != semantics_document.raw_document_sha256:
        raise PostHashMapBridgeError("runtime producer semantics binding drifted")
    _boolean(
        _field(semantics, "both_execution_attestations_required", label="runtime producer semantics binding"),
        label="runtime producer semantics binding.both_execution_attestations_required",
        expected=True,
    )
    for key, schema, status in [
        (
            "future_operator_accumulation_execution_attestation",
            OPERATOR_ATTESTATION_SCHEMA,
            OPERATOR_ATTESTATION_STATUS,
        ),
        (
            "future_range_reader_exact_semantics_attestation",
            READER_ATTESTATION_SCHEMA,
            READER_ATTESTATION_STATUS,
        ),
    ]:
        future = _mapping(
            _field(semantics, key, label="runtime producer semantics binding"),
            label=f"runtime producer semantics binding.{key}",
        )
        if _text(
            _field(future, "schema", label=f"runtime producer semantics binding.{key}"),
            label=f"runtime producer semantics binding.{key}.schema",
        ) != schema or _text(
            _field(
                future,
                "status_only_after_real_separately_leased_source_execution",
                label=f"runtime producer semantics binding.{key}",
            ),
            label=f"runtime producer semantics binding.{key}.status",
        ) != status:
            raise PostHashMapBridgeError(f"runtime producer {key} grammar drifted")
    flat = _mapping(
        _field(root, "future_flat_runtime_range_map", label="runtime-admission producer"),
        label="runtime producer future flat map",
    )
    if _text(
        _field(flat, "schema", label="runtime producer future flat map"),
        label="runtime producer future flat map.schema",
    ) != FLAT_MAP_SCHEMA or _integer(
        _field(flat, "maximum_window_bytes", label="runtime producer future flat map"),
        label="runtime producer future flat map.maximum_window_bytes",
    ) != MAX_WINDOW_BYTES or _integer(
        _field(flat, "maximum_positioned_read_bytes", label="runtime producer future flat map"),
        label="runtime producer future flat map.maximum_positioned_read_bytes",
    ) != MAX_WINDOW_BYTES:
        raise PostHashMapBridgeError("runtime producer flat-map grammar drifted")
    runtime = _mapping(
        _field(root, "future_runtime_admission_receipt", label="runtime-admission producer"),
        label="runtime producer future runtime admission",
    )
    if _text(
        _field(runtime, "schema", label="runtime producer future runtime admission"),
        label="runtime producer future runtime admission.schema",
    ) != RUNTIME_ADMISSION_SCHEMA or _text(
        _field(
            runtime,
            "status_only_after_bounded_source_validation",
            label="runtime producer future runtime admission",
        ),
        label="runtime producer future runtime admission.status",
    ) != RUNTIME_ADMISSION_STATUS:
        raise PostHashMapBridgeError("runtime producer future runtime-admission grammar drifted")
    boundary = _mapping(
        _field(root, "execution_boundary", label="runtime-admission producer"),
        label="runtime producer execution boundary",
    )
    for key in [
        "child_process_started",
        "future_source_root_opened_or_statted",
        "gpu_metal_mps_or_other_accelerator_invoked",
        "hcli_invoked",
        "lease_requested_issued_or_consumed",
        "server_started_or_contacted",
        "source_model_loaded_or_instantiated",
        "source_tensor_payload_opened",
        "tps_or_tg_measured",
        "whole_shard_mapped_or_cached",
        "whole_source_model_resident",
    ]:
        _boolean(
            _field(boundary, key, label="runtime producer execution boundary"),
            label=f"runtime producer execution boundary.{key}",
            expected=False,
        )


def _validate_flat_map(document: Document, source: SourceBinding) -> None:
    root = document.value
    if _text(_field(root, "schema", label="flat map"), label="flat map.schema") != FLAT_MAP_SCHEMA:
        raise PostHashMapBridgeError("flat map schema drifted")
    if _text(_field(root, "source_model_id", label="flat map"), label="flat map.source_model_id") != source.model_id or _text(
        _field(root, "source_revision", label="flat map"), label="flat map.source_revision"
    ) != source.revision:
        raise PostHashMapBridgeError("flat map source identity drifted")
    if _integer(
        _field(root, "source_tensor_count", label="flat map"), label="flat map.source_tensor_count"
    ) != SOURCE_TENSORS or _integer(
        _field(root, "maximum_window_bytes", label="flat map"), label="flat map.maximum_window_bytes"
    ) != MAX_WINDOW_BYTES:
        raise PostHashMapBridgeError("flat map geometry drifted")
    index = _mapping(_field(root, "source_index", label="flat map"), label="flat map source index")
    if _text(
        _field(index, "format", label="flat map source index"), label="flat map source index.format"
    ) != "huggingface.safetensors.index.json" or _sha(
        _field(index, "sha256", label="flat map source index"), label="flat map source index.sha256"
    ) != source.source_index_sha256:
        raise PostHashMapBridgeError("flat map source-index identity drifted")
    shards = _list(_field(root, "shards", label="flat map"), label="flat map.shards")
    if len(shards) != SOURCE_SHARDS:
        raise PostHashMapBridgeError("flat map must carry exactly 16 source shards")
    shard_bytes: dict[str, int] = {}
    for index, item in enumerate(shards):
        shard = _mapping(item, label=f"flat map shard[{index}]")
        shard_id = _text(
            _field(shard, "shard_id", label=f"flat map shard[{index}]"),
            label=f"flat map shard[{index}].shard_id",
        )
        if shard_id in shard_bytes:
            raise PostHashMapBridgeError("flat map shard IDs must be unique")
        relative = _text(
            _field(shard, "relative_path", label=f"flat map shard[{index}]"),
            label=f"flat map shard[{index}].relative_path",
        )
        if Path(relative).is_absolute() or any(part in {"", ".", ".."} for part in Path(relative).parts):
            raise PostHashMapBridgeError("flat map shard relative path is unsafe")
        shard_bytes[shard_id] = _integer(
            _field(shard, "bytes", label=f"flat map shard[{index}]"),
            label=f"flat map shard[{index}].bytes",
            minimum=1,
        )
        for key in ["sha256", "safetensors_header_sha256", "safetensors_prefix_sha256"]:
            _sha(
                _field(shard, key, label=f"flat map shard[{index}]"),
                label=f"flat map shard[{index}].{key}",
            )
    tensors = _list(_field(root, "tensors", label="flat map"), label="flat map.tensors")
    if len(tensors) != SOURCE_TENSORS:
        raise PostHashMapBridgeError("flat map must carry exactly 18,867 tensors")
    tensor_names: set[str] = set()
    ranges: dict[str, list[tuple[int, int]]] = {shard_id: [] for shard_id in shard_bytes}
    for index, item in enumerate(tensors):
        tensor = _mapping(item, label=f"flat map tensor[{index}]")
        name = _text(
            _field(tensor, "tensor_name", label=f"flat map tensor[{index}]"),
            label=f"flat map tensor[{index}].tensor_name",
        )
        if name in tensor_names:
            raise PostHashMapBridgeError("flat map tensor names must be unique")
        tensor_names.add(name)
        shard_id = _text(
            _field(tensor, "shard_id", label=f"flat map tensor[{index}]"),
            label=f"flat map tensor[{index}].shard_id",
        )
        if shard_id not in shard_bytes:
            raise PostHashMapBridgeError("flat map tensor references an unknown shard")
        if _text(
            _field(tensor, "dtype", label=f"flat map tensor[{index}]"),
            label=f"flat map tensor[{index}].dtype",
        ) != "BF16":
            raise PostHashMapBridgeError("flat map contains non-BF16 tensor evidence")
        shape = _list(
            _field(tensor, "shape", label=f"flat map tensor[{index}]"),
            label=f"flat map tensor[{index}].shape",
        )
        if not shape:
            raise PostHashMapBridgeError("flat map tensor shape must be non-empty")
        elements = 1
        for dimension, value in enumerate(shape):
            elements *= _integer(
                value,
                label=f"flat map tensor[{index}].shape[{dimension}]",
                minimum=1,
            )
        offset = _integer(
            _field(tensor, "data_offset", label=f"flat map tensor[{index}]"),
            label=f"flat map tensor[{index}].data_offset",
        )
        data_bytes = _integer(
            _field(tensor, "data_bytes", label=f"flat map tensor[{index}]"),
            label=f"flat map tensor[{index}].data_bytes",
            minimum=1,
        )
        if data_bytes != elements * 2:
            raise PostHashMapBridgeError("flat map BF16 tensor byte size disagrees with shape")
        if offset + data_bytes > shard_bytes[shard_id]:
            raise PostHashMapBridgeError("flat map tensor range exceeds its shard")
        _sha(
            _field(tensor, "raw_bf16_sha256", label=f"flat map tensor[{index}]"),
            label=f"flat map tensor[{index}].raw_bf16_sha256",
        )
        ranges[shard_id].append((offset, offset + data_bytes))
    for shard_id, intervals in ranges.items():
        previous_end = -1
        for start, end in sorted(intervals):
            if start < previous_end:
                raise PostHashMapBridgeError(
                    f"flat map BF16 ranges overlap in source shard {shard_id}"
                )
            previous_end = end


def _read_pointer_document(
    value: object, *, label: str, sealed: bool, expected: Document | None = None
) -> Document:
    pointer = _mapping(value, label=label)
    path = Path(_text(_field(pointer, "path", label=label), label=f"{label}.path"))
    document = _read_document(path, label=f"{label} document", sealed=sealed)
    _pointer(value, expected=expected or document, label=label)
    return document


def _validate_replay(document: Document) -> None:
    root = document.value
    _schema_status(root, schema=REPLAY_SCHEMA, status=REPLAY_STATUS, label="production replay")
    if _integer(_field(root, "attempt", label="production replay"), label="production replay.attempt") != 1:
        raise PostHashMapBridgeError("production replay reservation is stale or replayed")
    for key in [
        "create_new_before_source_root_open",
        "one_child_maximum",
        "replay_or_relaunch_forbidden",
    ]:
        _boolean(
            _field(root, key, label="production replay"),
            label=f"production replay.{key}",
            expected=True,
        )


def _validate_hash_coverage(
    document: Document, *, flat_map: Document, range_document: Document, source: SourceBinding
) -> Document:
    root = document.value
    _schema_status(
        root, schema=HASH_COVERAGE_SCHEMA, status=HASH_COVERAGE_STATUS, label="hash coverage"
    )
    for key, expected in [
        ("production_hash_coverage_earned", True),
        ("operator_or_reader_execution_attestation_emitted", False),
        ("source_teacher_execution_or_logits", False),
        ("source_teacher_runtime_admission_earned", False),
    ]:
        _boolean(
            _field(root, key, label="hash coverage"),
            label=f"hash coverage.{key}",
            expected=expected,
        )
    _pointer(
        _field(root, "flat_runtime_range_map", label="hash coverage"),
        expected=flat_map,
        label="hash coverage flat map",
    )
    _pointer(
        _field(root, "metadata_range_authority", label="hash coverage"),
        expected=range_document,
        label="hash coverage range authority",
    )
    coverage = _mapping(_field(root, "coverage", label="hash coverage"), label="hash coverage coverage")
    if _integer(
        _field(coverage, "source_shards", label="hash coverage coverage"),
        label="hash coverage coverage.source_shards",
    ) != SOURCE_SHARDS or _integer(
        _field(coverage, "source_tensors", label="hash coverage coverage"),
        label="hash coverage coverage.source_tensors",
    ) != SOURCE_TENSORS or _integer(
        _field(coverage, "full_shard_sha256_count", label="hash coverage coverage"),
        label="hash coverage coverage.full_shard_sha256_count",
    ) != SOURCE_SHARDS or _integer(
        _field(coverage, "raw_bf16_range_sha256_count", label="hash coverage coverage"),
        label="hash coverage coverage.raw_bf16_range_sha256_count",
    ) != SOURCE_TENSORS or _sha(
        _field(coverage, "source_index_sha256", label="hash coverage coverage"),
        label="hash coverage coverage.source_index_sha256",
    ) != source.source_index_sha256:
        raise PostHashMapBridgeError("hash coverage geometry or source index drifted")
    reader = _mapping(
        _field(root, "bounded_positioned_reader", label="hash coverage"),
        label="hash coverage positioned reader",
    )
    if _integer(
        _field(reader, "maximum_positioned_read_bytes", label="hash coverage positioned reader"),
        label="hash coverage positioned reader.maximum_positioned_read_bytes",
    ) != MAX_WINDOW_BYTES or _integer(
        _field(reader, "maximum_live_raw_bf16_windows", label="hash coverage positioned reader"),
        label="hash coverage positioned reader.maximum_live_raw_bf16_windows",
    ) != 1:
        raise PostHashMapBridgeError("hash coverage bounded-reader geometry drifted")
    for key in [
        "cache_zeroed_after_every_visit_and_before_receipt",
        "one_shard_handle_at_a_time",
        "whole_shard_cache_or_mmap_forbidden",
    ]:
        _boolean(
            _field(reader, key, label="hash coverage positioned reader"),
            label=f"hash coverage positioned reader.{key}",
            expected=True,
        )
    if _integer(
        _field(reader, "positioned_read_calls", label="hash coverage positioned reader"),
        label="hash coverage positioned reader.positioned_read_calls",
        minimum=1,
    ) < SOURCE_TENSORS or _integer(
        _field(reader, "positioned_read_bytes", label="hash coverage positioned reader"),
        label="hash coverage positioned reader.positioned_read_bytes",
        minimum=1,
    ) < SOURCE_TENSORS * 2:
        raise PostHashMapBridgeError("hash coverage positioned-reader accounting is incomplete")
    replay = _read_pointer_document(
        _field(root, "replay_reservation", label="hash coverage"),
        label="hash coverage replay reservation",
        sealed=True,
    )
    _validate_replay(replay)
    return replay


def _validate_capture(
    document: Document,
    *,
    flat_map: Document,
    hash_coverage: Document,
    replay: Document,
    range_document: Document,
    semantics_document: Document,
    runtime_producer: Document,
) -> None:
    root = document.value
    _schema_status(root, schema=CAPTURE_SCHEMA, status=CAPTURE_STATUS, label="production capture")
    for key, expected in [
        ("production_hash_scan_earned", True),
        ("receipt_written_last", True),
        ("source_handles_closed", True),
        ("reader_cache_zeroed", True),
        ("source_teacher_or_logits_executed", False),
        ("operator_or_reader_execution_attestation_emitted", False),
        ("source_teacher_runtime_admission_earned", False),
        ("model_gpu_server_hcli_or_tps_action", False),
    ]:
        _boolean(
            _field(root, key, label="production capture"),
            label=f"production capture.{key}",
            expected=expected,
        )
    geometry = _mapping(_field(root, "geometry", label="production capture"), label="capture geometry")
    if _integer(
        _field(geometry, "source_shards", label="capture geometry"),
        label="capture geometry.source_shards",
    ) != SOURCE_SHARDS or _integer(
        _field(geometry, "source_tensors", label="capture geometry"),
        label="capture geometry.source_tensors",
    ) != SOURCE_TENSORS or _integer(
        _field(geometry, "maximum_positioned_read_bytes", label="capture geometry"),
        label="capture geometry.maximum_positioned_read_bytes",
    ) != MAX_WINDOW_BYTES or _integer(
        _field(geometry, "maximum_live_raw_bf16_windows", label="capture geometry"),
        label="capture geometry.maximum_live_raw_bf16_windows",
    ) != 1:
        raise PostHashMapBridgeError("production capture geometry drifted")
    for key, expected in [
        ("flat_runtime_range_map", flat_map),
        ("hash_coverage_attestation", hash_coverage),
        ("replay_reservation", replay),
        ("metadata_range_authority", range_document),
        ("independent_non_fixture_semantics_attester", semantics_document),
        ("runtime_admission_producer_authority", runtime_producer),
    ]:
        _pointer(
            _field(root, key, label="production capture"),
            expected=expected,
            label=f"production capture {key}",
        )


def _validate_outer_terminal(
    document: Document, *, capture: Document
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    root = document.value
    _schema_status(
        root, schema=OUTER_TERMINAL_SCHEMA, status=OUTER_TERMINAL_STATUS, label="production outer terminal"
    )
    for key, expected in [
        ("child_reaped", True),
        ("terminal_receipt_written_after_child_capture", True),
        ("terminal_receipt_written_last", True),
        ("automatic_retry_disabled", True),
        ("lease_reuse_prohibited", True),
        ("child_timed_out", False),
    ]:
        _boolean(
            _field(root, key, label="production outer terminal"),
            label=f"production outer terminal.{key}",
            expected=expected,
        )
    if _integer(
        _field(root, "child_exit_code", label="production outer terminal"),
        label="production outer terminal.child_exit_code",
    ) != 0:
        raise PostHashMapBridgeError("production hash scan outer terminal is failed or stale")
    if root.get("child_signal") is not None or root.get("child_spawn_error") is not None or root.get(
        "child_capture_validation_error"
    ) is not None:
        raise PostHashMapBridgeError("production outer terminal contains a child failure")
    # The existing outer-terminal ABI predates canonical-pointer fields.  It
    # carries a raw+seal child pointer; the supplied direct child document is
    # independently canonicalized above, so this preserves compatibility
    # without silently accepting an unbound capture.
    _pointer(
        _field(root, "child_capture", label="production outer terminal"),
        expected=capture,
        label="production outer terminal child capture",
        canonical_required=False,
    )
    if _sha(
        _field(root, "child_capture_seal_sha256", label="production outer terminal"),
        label="production outer terminal.child_capture_seal_sha256",
    ) != capture.seal_sha256:
        raise PostHashMapBridgeError("production outer terminal child capture seal drifted")
    lease = _mapping(_field(root, "issued_lease", label="production outer terminal"), label="outer lease")
    authority = _mapping(
        _field(root, "production_authority", label="production outer terminal"), label="outer authority"
    )
    return lease, authority


def _validate_release_with_capture(
    document: Document,
    *,
    terminal: Document,
    capture: Document,
    terminal_lease: Mapping[str, Any],
) -> None:
    """Validate release with the established capture lease pointer ABI.

    This wrapper is intentionally separate from the generic release checks so
    the special historical outer pointer is explicit and testable.
    """
    root = document.value
    _schema_status(root, schema=RELEASE_SCHEMA, status=RELEASE_STATUS, label="production lease release")
    for key, expected in [
        ("release_after_outer_terminal", True),
        ("one_shot_lease_finalized", True),
        ("retry_or_relaunch_forbidden", True),
        ("source_teacher_or_logits_authorized", False),
        ("native_or_gpu_server_hcli_authorized", False),
        ("artifacts_deleted_or_evicted", False),
    ]:
        _boolean(
            _field(root, key, label="production lease release"),
            label=f"production lease release.{key}",
            expected=expected,
        )
    if _sha(
        _field(root, "outer_terminal_seal_sha256", label="production lease release"),
        label="production lease release.outer_terminal_seal_sha256",
    ) != terminal.seal_sha256 or _sha(
        _field(root, "child_capture_seal_sha256", label="production lease release"),
        label="production lease release.child_capture_seal_sha256",
    ) != capture.seal_sha256:
        raise PostHashMapBridgeError("production lease release provenance drifted")
    if _text(
        _field(root, "outer_terminal_status", label="production lease release"),
        label="production lease release.outer_terminal_status",
    ) != OUTER_TERMINAL_STATUS:
        raise PostHashMapBridgeError("production lease release terminal status drifted")
    issued_lease_path = Path(
        _text(
            _field(terminal_lease, "path", label="outer issued lease"),
            label="outer issued lease.path",
        )
    )
    issued_lease = _read_document(
        issued_lease_path, label="outer issued lease document", sealed=True
    )
    _pointer(
        terminal_lease,
        expected=issued_lease,
        label="outer terminal issued lease",
        canonical_required=False,
    )
    capture_lease = _mapping(
        _field(capture.value, "fresh_bootstrap_lease", label="production capture"),
        label="production capture fresh lease",
    )
    _pointer(
        capture_lease,
        expected=issued_lease,
        label="production capture fresh lease",
    )
    _same_pointer(
        terminal_lease,
        capture_lease,
        label="outer terminal/capture issued lease",
        canonical_required=False,
    )
    if _sha(
        _field(root, "lease_id", label="production lease release"),
        label="production lease release.lease_id",
    ) != _sha(
        _field(issued_lease.value, "lease_id", label="issued lease"),
        label="issued lease.lease_id",
    ):
        raise PostHashMapBridgeError("production lease release lease ID drifted")


def _post_hash_map_bridge_document(
    *,
    range_document: Document,
    semantics_document: Document,
    runtime_producer: Document,
    flat_map: Document,
    hash_coverage: Document,
    capture: Document,
    terminal: Document,
    release: Document,
    source: SourceBinding,
) -> dict[str, Any]:
    return seal(
        {
            "schema": SCHEMA,
            "status": STATUS,
            "prepared": True,
            "execution_authorized": False,
            "runtime_admission_earned": False,
            "dual_attestation_runtime_admission_emitted": False,
            "post_hash_map_antecedents": {
                "production_outer_terminal": terminal.evidence(),
                "production_child_capture": capture.evidence(),
                "production_flat_map": flat_map.evidence(),
                "production_hash_coverage": hash_coverage.evidence(),
                "production_lease_release": release.evidence(),
            },
            "upstream_authorities": {
                "metadata_range_authority": range_document.evidence(),
                "semantics_attester": semantics_document.evidence(),
                "runtime_admission_producer_authority": runtime_producer.evidence(),
            },
            "validated_production_hash_scan": {
                "non_fixture_production_flat_map": True,
                "source_model_id": source.model_id,
                "source_revision": source.revision,
                "source_index_sha256": source.source_index_sha256,
                "source_shards": SOURCE_SHARDS,
                "source_tensors": SOURCE_TENSORS,
                "all_full_shard_and_raw_bf16_hash_coverage_bound": True,
                "maximum_positioned_read_bytes": MAX_WINDOW_BYTES,
                "maximum_live_raw_bf16_windows": 1,
                "one_shard_handle_at_a_time": True,
                "reader_cache_zeroed_before_hash_scan_receipt": True,
                "source_handles_closed_before_hash_scan_receipt": True,
                "one_shot_replay_reservation_attempt": 1,
                "outer_child_reaped_successfully": True,
                "lease_finalized_after_outer_terminal": True,
                "source_teacher_execution_or_logits": False,
                "operator_or_reader_execution_attestation_emitted": False,
                "source_teacher_runtime_admission_earned": False,
            },
            "future_source_teacher_provenance_reservation": {
                "reservation_status": "NOT_EXECUTED",
                "post_hash_map_bridge_is_not_runtime_admission": True,
                "post_hash_map_bridge_is_not_dual_attestation_bridge": True,
                "runtime_admission": {
                    "schema": RUNTIME_ADMISSION_SCHEMA,
                    "status": RUNTIME_ADMISSION_STATUS,
                    "must_be_sealed": True,
                    "must_bind_post_hash_map_antecedents": True,
                    "must_bind_flat_map_and_coverage_canonical_hashes": True,
                    "must_bind_both_future_execution_attestation_seals": True,
                    "must_precede_any_source_root_or_payload_open": True,
                    "not_emitted_by_this_bridge": True,
                },
                "dual_attestation_runtime_admission": {
                    "schema": DUAL_BRIDGE_SCHEMA,
                    "status": DUAL_BRIDGE_STATUS,
                    "must_be_sealed": True,
                    "must_bind_this_post_hash_map_bridge_seal_sha256": True,
                    "must_bind_runtime_admission_seal_sha256": True,
                    "must_preserve_existing_source_teacher_child_schema_resolution": True,
                    "not_emitted_by_this_bridge": True,
                },
                "existing_source_teacher_child_compatible_shape": {
                    "schema_resolution": {
                        "runtime_range_map_schema": FLAT_MAP_SCHEMA,
                        "runtime_admission_schema": RUNTIME_ADMISSION_SCHEMA,
                        "runtime_admission_status_only_after_bounded_source_validation": RUNTIME_ADMISSION_STATUS,
                        "operator_accumulation_execution_attestation": {
                            "schema": OPERATOR_ATTESTATION_SCHEMA,
                            "status": OPERATOR_ATTESTATION_STATUS,
                        },
                        "range_reader_exact_semantics_attestation": {
                            "schema": READER_ATTESTATION_SCHEMA,
                            "status": READER_ATTESTATION_STATUS,
                        },
                        "both_execution_attestations_required_after_source_child": True,
                        "runtime_range_admission_required_before_payload_open": True,
                        "bridge_does_not_authorize_execution": True,
                    },
                    "future_source_worker": {
                        "maximum_positioned_read_bytes": MAX_WINDOW_BYTES,
                        "source_layers": SOURCE_LAYERS,
                        "source_forwards": SOURCE_FORWARDS,
                        "source_f32le_vectors": 2,
                        "native_f32le_vectors": 4,
                        "one_bounded_window_only": True,
                        "source_payloads_durable_before_eviction": True,
                        "close_handles_and_clear_cache_before_eviction_receipt": True,
                        "separate_native_four_vector_phase_required": True,
                    },
                },
                "unbound_later_inputs_required_by_existing_source_teacher_preflight": [
                    "sealed_streamed_feasibility_receipt",
                    "sealed_raw_six_vector_contract",
                    "sealed_current_trace",
                    "fresh_source_teacher_lease",
                ],
            },
            "admission_before_open_cycle": {
                "runtime_admission_must_be_earned_before_source_root_open": True,
                "runtime_producer_requires_bounded_source_validation_and_both_execution_attestations": True,
                "existing_source_teacher_child_requires_runtime_admission_before_source_root_open": True,
                "resolved": False,
                "bridge_does_not_relax_or_reorder_any_requirement": True,
            },
            "current_blockers": [
                "fresh_source_teacher_runtime_admission_absent",
                "sealed_dual_attestation_runtime_admission_bridge_absent",
                "admission_before_open_cycle_unresolved",
                "fresh_source_teacher_lease_absent",
                "no_real_source_teacher_child_or_execution_attestations_exist",
            ],
            "execution_boundary": {
                "source_root_opened_or_statted": False,
                "source_payload_opened": False,
                "source_model_loaded_or_instantiated": False,
                "whole_source_model_resident": False,
                "source_teacher_or_logits_executed": False,
                "operator_or_reader_execution_attestation_emitted": False,
                "source_teacher_runtime_admission_emitted": False,
                "gpu_native_server_hcli_or_tps_action": False,
                "lease_issued_or_consumed": False,
                "child_process_started": False,
            },
            "claim_boundary": "Sealed CPU/file-only post-hash-map provenance reservation only. It validates a completed non-fixture production bounded hash scan and preserves the existing source-teacher ABI, but it does not authorize or execute source teacher/native/model/GPU/server/HCLI/lease/TPS/TG/tournament work, earn a runtime admission, or emit a dual-attestation bridge.",
        }
    )


def build_post_hash_map_bridge(
    *,
    range_authority_path: Path,
    semantics_attester_path: Path,
    runtime_producer_path: Path,
    flat_map_path: Path,
    hash_coverage_path: Path,
    production_capture_path: Path,
    outer_terminal_path: Path,
    lease_release_path: Path,
) -> dict[str, Any]:
    """Validate antecedents and return a sealed, non-authorizing reservation."""
    range_document = _read_document(
        range_authority_path, label="metadata range authority", sealed=False
    )
    source = _validate_range_authority(range_document)
    semantics_document = _read_document(
        semantics_attester_path, label="semantics attester", sealed=False
    )
    _validate_semantics(semantics_document, source, range_document)
    runtime_producer = _read_document(
        runtime_producer_path, label="runtime-admission producer", sealed=True
    )
    _validate_runtime_producer(runtime_producer, source, range_document, semantics_document)
    flat_map = _read_document(flat_map_path, label="production flat map", sealed=True)
    _validate_flat_map(flat_map, source)
    hash_coverage = _read_document(hash_coverage_path, label="production hash coverage", sealed=True)
    replay = _validate_hash_coverage(
        hash_coverage, flat_map=flat_map, range_document=range_document, source=source
    )
    capture = _read_document(production_capture_path, label="production child capture", sealed=True)
    _validate_capture(
        capture,
        flat_map=flat_map,
        hash_coverage=hash_coverage,
        replay=replay,
        range_document=range_document,
        semantics_document=semantics_document,
        runtime_producer=runtime_producer,
    )
    terminal = _read_document(outer_terminal_path, label="production outer terminal", sealed=True)
    terminal_lease, _terminal_authority = _validate_outer_terminal(terminal, capture=capture)
    release = _read_document(lease_release_path, label="production lease release", sealed=True)
    _validate_release_with_capture(
        release, terminal=terminal, capture=capture, terminal_lease=terminal_lease
    )
    return _post_hash_map_bridge_document(
        range_document=range_document,
        semantics_document=semantics_document,
        runtime_producer=runtime_producer,
        flat_map=flat_map,
        hash_coverage=hash_coverage,
        capture=capture,
        terminal=terminal,
        release=release,
        source=source,
    )


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.suffix != ".json" or not path.parent.is_dir() or path.exists():
        raise PostHashMapBridgeError("--out must be a new absolute JSON file below an existing directory")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
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
    parser.add_argument("--range-authority", type=Path, required=True)
    parser.add_argument("--semantics-attester", type=Path, required=True)
    parser.add_argument("--runtime-producer-authority", type=Path, required=True)
    parser.add_argument("--flat-runtime-range-map", type=Path, required=True)
    parser.add_argument("--hash-coverage-attestation", type=Path, required=True)
    parser.add_argument("--production-capture", type=Path, required=True)
    parser.add_argument("--outer-terminal", type=Path, required=True)
    parser.add_argument("--lease-release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_post_hash_map_bridge(
            range_authority_path=args.range_authority,
            semantics_attester_path=args.semantics_attester,
            runtime_producer_path=args.runtime_producer_authority,
            flat_map_path=args.flat_runtime_range_map,
            hash_coverage_path=args.hash_coverage_attestation,
            production_capture_path=args.production_capture,
            outer_terminal_path=args.outer_terminal,
            lease_release_path=args.lease_release,
        )
        _write_new(args.out, result)
    except PostHashMapBridgeError as exc:
        print(json.dumps({"status": "REFUSED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {"output": str(args.out.resolve()), "status": result["status"], "seal_sha256": result["seal_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
