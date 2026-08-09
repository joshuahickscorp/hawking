#!/usr/bin/env python3
"""Non-circular, CPU-only preflight for a future Q30 source-teacher runner.

The legacy source child requires an already-earned runtime admission before it
opens a source root.  That makes the old route circular when those exact
operator/reader attestations can only be earned by the source run itself.

This isolated interface introduces a *production source-teacher execution
permit*.  It is a new pre-execution reservation, never the legacy earned
runtime-admission schema.  It binds the earned non-fixture hash map/coverage,
metadata+semantics, post-hash bridge, and a fresh zero-swap resource admission.
Its future 48x370 source run is the only transition allowed to emit the two
execution attestations and the final legacy runtime admission.

The module reads sealed or metadata JSON only.  It has no source-root CLI,
process launcher, lease issuer, model, GPU, server, or HCLI surface.  The
``validate_fake_post_execution_finalization`` helper is mapping-only test
machinery; it cannot execute a source run.
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
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.receipts import SealIntegrityError, seal, verify

SCHEMA = "hawking.ascension.qwen30_production_source_teacher_execution_permit.v1"
PREPARED_STATUS = (
    "PREPARED_QWEN30_PRODUCTION_SOURCE_TEACHER_EXECUTION_PERMIT_NOT_EXECUTED"
)
REFUSED_STATUS = (
    "REFUSED_QWEN30_PRODUCTION_SOURCE_TEACHER_EXECUTION_PERMIT_PREREQUISITES_ABSENT_OR_INVALID"
)

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

POST_HASH_MAP_BRIDGE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_teacher_post_hash_map_bridge.v1"
)
POST_HASH_MAP_BRIDGE_STATUS = (
    "PREPARED_QWEN30_STREAMED_SOURCE_TEACHER_POST_HASH_MAP_BRIDGE_NOT_EXECUTED"
)
FLAT_MAP_SCHEMA = "hawking.ascension.qwen30_source_bf16_range_map.v1"
HASH_COVERAGE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_production_hash_coverage_attestation.v1"
)
HASH_COVERAGE_STATUS = (
    "EARNED_QWEN30_STREAMED_SOURCE_PRODUCTION_HASH_COVERAGE_ATTESTED_NOT_SOURCE_TEACHER"
)
PRODUCTION_CAPTURE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_capture.v1"
)
PRODUCTION_CAPTURE_STATUS = (
    "CAPTURED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_NOT_SOURCE_TEACHER"
)
PRODUCTION_OUTER_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_outer_terminal.v1"
)
PRODUCTION_OUTER_STATUS = (
    "CAPTURED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_OUTER_TERMINAL_NOT_SOURCE_TEACHER"
)
PRODUCTION_RELEASE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_quiet_lease_release.v1"
)
PRODUCTION_RELEASE_STATUS = (
    "RELEASED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_LEASE_AFTER_OUTER_TERMINAL"
)

RESOURCE_SCHEMA = (
    "hawking.ascension.qwen30_production_source_teacher_execution_permit_resource_admission.v1"
)
RESOURCE_STATUS = (
    "PREPARED_QWEN30_PRODUCTION_SOURCE_TEACHER_EXECUTION_PERMIT_RESOURCE_ADMISSION_NOT_LEASED"
)
EXECUTION_LEASE_SCHEMA = (
    "hawking.ascension.qwen30_production_source_teacher_execution_permit_quiet_lease.v1"
)
EXECUTION_LEASE_STATUS = (
    "GRANTED_QWEN30_PRODUCTION_SOURCE_TEACHER_EXECUTION_PERMIT_ONE_SHOT"
)
EXECUTION_CAPTURE_SCHEMA = (
    "hawking.ascension.qwen30_production_source_teacher_execution_permit_capture.v1"
)
EXECUTION_CAPTURE_STATUS = (
    "CAPTURED_QWEN30_PRODUCTION_SOURCE_TEACHER_48X370_TWO_F32LE_BEFORE_FINAL_ADMISSION"
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
FINAL_RUNTIME_ADMISSION_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_teacher_runtime_range_admission.v1"
)
FINAL_RUNTIME_ADMISSION_STATUS = (
    "EARNED_QWEN30_STREAMED_SOURCE_TEACHER_RUNTIME_RANGE_ADMISSION_NO_MODEL_RESIDENCY"
)

MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_POSITIONED_READ_BYTES = 1024 * 1024
SOURCE_SHARDS = 16
SOURCE_TENSORS = 18_867
SOURCE_LAYERS = 48
SOURCE_FORWARDS = 370
SOURCE_VECTORS = 2
MODEL_ID = "Qwen3-Coder-30B-A3B-Instruct"


class ExecutionPermitError(RuntimeError):
    """An input cannot lawfully reserve a production source-teacher run."""


@dataclass(frozen=True)
class Document:
    path: Path
    value: dict[str, Any]
    raw_document_sha256: str
    canonical_document_sha256: str
    seal_sha256: str | None


@dataclass(frozen=True)
class SourceBinding:
    model_id: str
    revision: str
    source_index_sha256: str
    range_authority_content_sha256: str


@dataclass(frozen=True)
class Antecedents:
    range_authority: Document
    semantics_attester: Document
    post_hash_map_bridge: Document
    flat_map: Document
    hash_coverage: Document
    production_capture: Document
    production_outer_terminal: Document
    production_release: Document
    source: SourceBinding


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExecutionPermitError(f"{label} must be an object")
    return dict(value)


def _sequence(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ExecutionPermitError(f"{label} must be an array")
    return list(value)


def _text(value: object, *, label: str, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ExecutionPermitError(f"{label} must be non-empty text")
    if sha256 and (
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExecutionPermitError(f"{label} must be a lowercase SHA-256")
    return value


def _git_revision(value: object, *, label: str) -> str:
    revision = _text(value, label=label)
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ExecutionPermitError(f"{label} must be a lowercase Git revision")
    return revision


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ExecutionPermitError(f"{label} must be an integer >= {minimum}")
    return value


def _require(value: object, *, expected: bool, label: str) -> None:
    if value is not expected:
        raise ExecutionPermitError(f"{label} must be {expected!r}")


def _schema_status(
    document: Mapping[str, Any], *, schema: str, status: str, label: str
) -> None:
    if document.get("schema") != schema or document.get("status") != status:
        raise ExecutionPermitError(f"{label} schema/status drifted")


def _regular_json(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.suffix != ".json":
        raise ExecutionPermitError(f"{label} must be an absolute .json path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ExecutionPermitError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ExecutionPermitError(f"{label} must be a regular non-symlink file")
    if metadata.st_size <= 0 or metadata.st_size > MAX_METADATA_BYTES:
        raise ExecutionPermitError(f"{label} has invalid metadata size")
    return path.resolve(strict=True)


def _read_document(path: Path, *, label: str, sealed: bool) -> Document:
    clean = _regular_json(path, label=label)
    try:
        raw_bytes = clean.read_bytes()
        raw = json.loads(raw_bytes)
        if sealed:
            checked: Mapping[str, Any] = verify(raw, label=label)
        else:
            checked = _mapping(raw, label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise ExecutionPermitError(f"{label} is absent or invalid: {exc}") from exc
    value = _mapping(checked, label=label)
    seal_value = value.get("seal_sha256")
    if sealed:
        seal_value = _text(seal_value, label=f"{label} seal", sha256=True)
    elif seal_value is not None:
        raise ExecutionPermitError(f"{label} must not carry a synthetic seal")
    return Document(
        path=clean,
        value=value,
        raw_document_sha256=_sha256(raw_bytes),
        canonical_document_sha256=_sha256(_canonical_json(value)),
        seal_sha256=seal_value,
    )


def _evidence(document: Document) -> dict[str, str | None]:
    return {
        "path": str(document.path),
        "raw_document_sha256": document.raw_document_sha256,
        "canonical_document_sha256": document.canonical_document_sha256,
        "seal_sha256": document.seal_sha256,
    }


def _pointer(value: object, *, expected: Document, label: str) -> None:
    pointer = _mapping(value, label=label)
    if _text(pointer.get("path"), label=f"{label}.path") != str(expected.path):
        raise ExecutionPermitError(f"{label} path drifted")
    for field, observed in (
        ("raw_document_sha256", expected.raw_document_sha256),
        ("canonical_document_sha256", expected.canonical_document_sha256),
    ):
        if _text(pointer.get(field), label=f"{label}.{field}", sha256=True) != observed:
            raise ExecutionPermitError(f"{label} {field} drifted")
    if expected.seal_sha256 is None:
        if pointer.get("seal_sha256") is not None:
            raise ExecutionPermitError(f"{label} unexpectedly claims a seal")
    elif _text(pointer.get("seal_sha256"), label=f"{label}.seal_sha256", sha256=True) != expected.seal_sha256:
        raise ExecutionPermitError(f"{label} seal drifted")


def _reject_fixture_identity(value: object, *, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"schema", "status"} and isinstance(child, str):
                if "fixture" in child.lower() or "synthetic" in child.lower():
                    raise ExecutionPermitError(f"{label}.{key} carries fixture identity")
            if key in {"fixture_only", "synthetic_fixture_only", "production_adapter_forbidden"} and child is True:
                raise ExecutionPermitError(f"{label}.{key} marks non-production evidence")
            _reject_fixture_identity(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_fixture_identity(child, label=f"{label}[{index}]")


def _validate_range_authority(document: Document) -> SourceBinding:
    root = document.value
    authority = _mapping(root.get("authority"), label="range authority")
    _schema_status(
        authority,
        schema=RANGE_AUTHORITY_SCHEMA,
        status=RANGE_AUTHORITY_STATUS,
        label="range authority",
    )
    content_sha = _text(root.get("authority_content_sha256"), label="range authority content SHA", sha256=True)
    if _sha256(_canonical_json(authority)) != content_sha:
        raise ExecutionPermitError("range authority content SHA drifted")
    source = _mapping(authority.get("source"), label="range authority source")
    model_id = _text(source.get("model_id"), label="range authority model")
    if model_id != MODEL_ID:
        raise ExecutionPermitError("range authority model identity drifted")
    revision = _git_revision(source.get("source_revision"), label="range authority revision")
    if _integer(source.get("source_shard_count"), label="range authority shard count") != SOURCE_SHARDS:
        raise ExecutionPermitError("range authority shard count drifted")
    if _integer(source.get("source_tensor_count"), label="range authority tensor count") != SOURCE_TENSORS:
        raise ExecutionPermitError("range authority tensor count drifted")
    index = _mapping(source.get("source_index"), label="range authority source index")
    index_sha = _text(index.get("sha256"), label="range authority source index SHA", sha256=True)
    if index.get("format") != "huggingface.safetensors.index.json":
        raise ExecutionPermitError("range authority source index format drifted")
    boundary = _mapping(authority.get("metadata_access_boundary"), label="range authority boundary")
    for field in (
        "gpu_or_metal_invoked",
        "hcli_invoked",
        "lease_requested",
        "mmap_or_memory_map_used",
        "server_started",
        "source_model_instantiated",
        "tensor_payload_hashes_collected",
        "whole_shard_payload_checksum_collected",
    ):
        _require(boundary.get(field), expected=False, label=f"range authority boundary.{field}")
    if _integer(boundary.get("source_tensor_payload_bytes_read"), label="range authority payload bytes") != 0:
        raise ExecutionPermitError("range authority must not have read source payloads")
    scope = _mapping(authority.get("exact_streamed_oracle_scope"), label="range authority scope")
    if _integer(scope.get("layers"), label="range authority layers") != SOURCE_LAYERS:
        raise ExecutionPermitError("range authority layer scope drifted")
    if _integer(scope.get("total_forwards_per_replay_arm"), label="range authority forwards") != SOURCE_FORWARDS:
        raise ExecutionPermitError("range authority forward scope drifted")
    return SourceBinding(
        model_id=model_id,
        revision=revision,
        source_index_sha256=index_sha,
        range_authority_content_sha256=content_sha,
    )


def _validate_semantics(document: Document, *, source: SourceBinding, range_authority: Document) -> None:
    root = document.value
    _schema_status(root, schema=SEMANTICS_SCHEMA, status=SEMANTICS_STATUS, label="semantics attester")
    _reject_fixture_identity(root, label="semantics attester")
    boundary = _mapping(root.get("execution_boundary"), label="semantics boundary")
    for field in (
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
    ):
        _require(boundary.get(field), expected=False, label=f"semantics boundary.{field}")
    pinned = _mapping(root.get("pinned_source_binding"), label="semantics source binding")
    if (
        pinned.get("source_model_id") != source.model_id
        or pinned.get("source_revision") != source.revision
        or _text(pinned.get("source_index_sha256"), label="semantics source index SHA", sha256=True)
        != source.source_index_sha256
    ):
        raise ExecutionPermitError("semantics source binding drifted")
    consumed = _mapping(root.get("consumed_metadata_contracts"), label="semantics consumed metadata")
    range_pointer = _mapping(consumed.get("range_authority"), label="semantics range authority pointer")
    if (
        _text(range_pointer.get("document_sha256"), label="semantics range document SHA", sha256=True)
        != range_authority.raw_document_sha256
        or _text(
            range_pointer.get("authority_content_sha256"),
            label="semantics range authority SHA",
            sha256=True,
        )
        != source.range_authority_content_sha256
    ):
        raise ExecutionPermitError("semantics range authority binding drifted")
    _require(
        range_pointer.get("source_payload_read_by_this_attester"),
        expected=False,
        label="semantics range payload boundary",
    )
    future = _mapping(root.get("future_exact_execution_attestation"), label="semantics future attestation")
    if future.get("schema") != OPERATOR_ATTESTATION_SCHEMA or future.get(
        "status_only_after_real_separately_leased_source_execution"
    ) != OPERATOR_ATTESTATION_STATUS:
        raise ExecutionPermitError("semantics future operator-attestation grammar drifted")


def _validate_flat_map(document: Document, *, source: SourceBinding) -> None:
    root = document.value
    if root.get("schema") != FLAT_MAP_SCHEMA:
        raise ExecutionPermitError("production flat map schema drifted")
    _reject_fixture_identity(root, label="production flat map")
    for field in ("fixture_only", "synthetic_fixture_only"):
        _require(root.get(field), expected=False, label=f"production flat map.{field}")
    if root.get("source_model_id") != source.model_id or root.get("source_revision") != source.revision:
        raise ExecutionPermitError("production flat map source identity drifted")
    if _integer(root.get("source_tensor_count"), label="production flat map tensor count") != SOURCE_TENSORS:
        raise ExecutionPermitError("production flat map tensor count drifted")
    if _integer(root.get("maximum_window_bytes"), label="production flat map window") != MAX_POSITIONED_READ_BYTES:
        raise ExecutionPermitError("production flat map window drifted")
    if len(_sequence(root.get("shards"), label="production flat map shards")) != SOURCE_SHARDS:
        raise ExecutionPermitError("production flat map shard list drifted")
    if len(_sequence(root.get("tensors"), label="production flat map tensors")) != SOURCE_TENSORS:
        raise ExecutionPermitError("production flat map tensor list drifted")
    index = _mapping(root.get("source_index"), label="production flat map source index")
    if _text(index.get("sha256"), label="production flat map source index SHA", sha256=True) != source.source_index_sha256:
        raise ExecutionPermitError("production flat map source index drifted")


def _validate_hash_coverage(document: Document, *, flat_map: Document, source: SourceBinding) -> None:
    root = document.value
    _schema_status(root, schema=HASH_COVERAGE_SCHEMA, status=HASH_COVERAGE_STATUS, label="hash coverage")
    _reject_fixture_identity(root, label="hash coverage")
    for field in (
        "fixture_only",
        "synthetic_fixture_only",
        "source_teacher_execution_or_logits",
        "source_teacher_runtime_admission_earned",
        "operator_or_reader_execution_attestation_emitted",
    ):
        _require(root.get(field), expected=False, label=f"hash coverage.{field}")
    _require(root.get("production_hash_coverage_earned"), expected=True, label="hash coverage earned")
    _pointer(root.get("flat_runtime_range_map"), expected=flat_map, label="hash coverage flat map")
    bounded = _mapping(root.get("bounded_positioned_reader"), label="hash coverage reader")
    if _integer(bounded.get("maximum_positioned_read_bytes"), label="hash coverage max read") != MAX_POSITIONED_READ_BYTES:
        raise ExecutionPermitError("hash coverage max read drifted")
    if _integer(bounded.get("maximum_live_raw_bf16_windows"), label="hash coverage windows") != 1:
        raise ExecutionPermitError("hash coverage window count drifted")
    for field in (
        "one_shard_handle_at_a_time",
        "whole_shard_cache_or_mmap_forbidden",
        "cache_zeroed_after_every_visit_and_before_receipt",
    ):
        _require(bounded.get(field), expected=True, label=f"hash coverage reader.{field}")
    coverage = _mapping(root.get("coverage"), label="hash coverage counts")
    if (
        _integer(coverage.get("source_shards"), label="hash coverage shards") != SOURCE_SHARDS
        or _integer(coverage.get("source_tensors"), label="hash coverage tensors") != SOURCE_TENSORS
        or _text(coverage.get("source_index_sha256"), label="hash coverage index SHA", sha256=True)
        != source.source_index_sha256
    ):
        raise ExecutionPermitError("hash coverage geometry/identity drifted")


def _validate_production_capture(
    document: Document, *, flat_map: Document, hash_coverage: Document
) -> None:
    root = document.value
    _schema_status(root, schema=PRODUCTION_CAPTURE_SCHEMA, status=PRODUCTION_CAPTURE_STATUS, label="production capture")
    _reject_fixture_identity(root, label="production capture")
    for field in (
        "fixture_only",
        "synthetic_fixture_only",
        "source_teacher_or_logits_executed",
        "source_teacher_runtime_admission_earned",
        "operator_or_reader_execution_attestation_emitted",
        "model_gpu_server_hcli_or_tps_action",
    ):
        _require(root.get(field), expected=False, label=f"production capture.{field}")
    for field in ("production_hash_scan_earned", "source_handles_closed", "reader_cache_zeroed", "receipt_written_last"):
        _require(root.get(field), expected=True, label=f"production capture.{field}")
    _pointer(root.get("flat_runtime_range_map"), expected=flat_map, label="production capture flat map")
    _pointer(root.get("hash_coverage_attestation"), expected=hash_coverage, label="production capture coverage")


def _validate_production_outer(document: Document, *, capture: Document) -> None:
    root = document.value
    _schema_status(root, schema=PRODUCTION_OUTER_SCHEMA, status=PRODUCTION_OUTER_STATUS, label="production outer")
    for field in (
        "child_reaped",
        "terminal_receipt_written_after_child_capture",
        "terminal_receipt_written_last",
        "automatic_retry_disabled",
        "lease_reuse_prohibited",
    ):
        _require(root.get(field), expected=True, label=f"production outer.{field}")
    _require(root.get("child_timed_out"), expected=False, label="production outer child timeout")
    if _integer(root.get("child_exit_code"), label="production outer child exit") != 0:
        raise ExecutionPermitError("production outer child exit drifted")
    pointer = _mapping(root.get("child_capture"), label="production outer child capture")
    if (
        _text(pointer.get("path"), label="production outer child path") != str(capture.path)
        or _text(pointer.get("raw_document_sha256"), label="production outer child raw SHA", sha256=True)
        != capture.raw_document_sha256
        or _text(pointer.get("seal_sha256"), label="production outer child seal", sha256=True)
        != capture.seal_sha256
    ):
        raise ExecutionPermitError("production outer child capture binding drifted")


def _validate_production_release(document: Document, *, outer_terminal: Document, capture: Document) -> None:
    root = document.value
    _schema_status(root, schema=PRODUCTION_RELEASE_SCHEMA, status=PRODUCTION_RELEASE_STATUS, label="production release")
    for field in ("release_after_outer_terminal", "one_shot_lease_finalized", "retry_or_relaunch_forbidden"):
        _require(root.get(field), expected=True, label=f"production release.{field}")
    for field in ("source_teacher_or_logits_authorized", "native_or_gpu_server_hcli_authorized", "artifacts_deleted_or_evicted"):
        _require(root.get(field), expected=False, label=f"production release.{field}")
    if (
        _text(root.get("outer_terminal_seal_sha256"), label="production release outer seal", sha256=True)
        != outer_terminal.seal_sha256
        or _text(root.get("child_capture_seal_sha256"), label="production release capture seal", sha256=True)
        != capture.seal_sha256
    ):
        raise ExecutionPermitError("production release antecedent binding drifted")


def _validate_post_hash_map_bridge(document: Document, *, antecedents: Antecedents) -> None:
    root = document.value
    _schema_status(root, schema=POST_HASH_MAP_BRIDGE_SCHEMA, status=POST_HASH_MAP_BRIDGE_STATUS, label="post-hash bridge")
    _reject_fixture_identity(root, label="post-hash bridge")
    for field, expected in (
        ("prepared", True),
        ("execution_authorized", False),
        ("runtime_admission_earned", False),
        ("dual_attestation_runtime_admission_emitted", False),
    ):
        _require(root.get(field), expected=expected, label=f"post-hash bridge.{field}")
    bindings = _mapping(root.get("post_hash_map_antecedents"), label="post-hash bridge antecedents")
    for field, expected in (
        ("production_outer_terminal", antecedents.production_outer_terminal),
        ("production_child_capture", antecedents.production_capture),
        ("production_flat_map", antecedents.flat_map),
        ("production_hash_coverage", antecedents.hash_coverage),
        ("production_lease_release", antecedents.production_release),
    ):
        _pointer(bindings.get(field), expected=expected, label=f"post-hash bridge.{field}")
    upstream = _mapping(root.get("upstream_authorities"), label="post-hash bridge upstream authorities")
    for field, expected in (
        ("metadata_range_authority", antecedents.range_authority),
        ("semantics_attester", antecedents.semantics_attester),
    ):
        _pointer(upstream.get(field), expected=expected, label=f"post-hash bridge upstream.{field}")
    cycle = _mapping(root.get("admission_before_open_cycle"), label="post-hash bridge cycle")
    _require(cycle.get("resolved"), expected=False, label="post-hash bridge cycle.resolved")
    for field in (
        "runtime_admission_must_be_earned_before_source_root_open",
        "runtime_producer_requires_bounded_source_validation_and_both_execution_attestations",
        "existing_source_teacher_child_requires_runtime_admission_before_source_root_open",
        "bridge_does_not_relax_or_reorder_any_requirement",
    ):
        _require(cycle.get(field), expected=True, label=f"post-hash bridge cycle.{field}")
    boundary = _mapping(root.get("execution_boundary"), label="post-hash bridge boundary")
    for field in (
        "source_root_opened_or_statted",
        "source_payload_opened",
        "source_model_loaded_or_instantiated",
        "whole_source_model_resident",
        "source_teacher_or_logits_executed",
        "operator_or_reader_execution_attestation_emitted",
        "source_teacher_runtime_admission_emitted",
        "gpu_native_server_hcli_or_tps_action",
        "lease_issued_or_consumed",
        "child_process_started",
    ):
        _require(boundary.get(field), expected=False, label=f"post-hash bridge boundary.{field}")


def _validate_resource(document: Document, *, antecedents: Antecedents) -> None:
    root = document.value
    _schema_status(root, schema=RESOURCE_SCHEMA, status=RESOURCE_STATUS, label="execution-permit resource")
    _reject_fixture_identity(root, label="execution-permit resource")
    for field in (
        "source_root_opened_or_statted",
        "source_teacher_or_native_child_started",
        "execution_permit_lease_issued_or_consumed",
    ):
        _require(root.get(field), expected=False, label=f"execution-permit resource.{field}")
    for field, expected in (
        ("post_hash_map_bridge", antecedents.post_hash_map_bridge),
        ("production_flat_map", antecedents.flat_map),
        ("production_hash_coverage", antecedents.hash_coverage),
        ("metadata_range_authority", antecedents.range_authority),
        ("semantics_attester", antecedents.semantics_attester),
    ):
        _pointer(root.get(field), expected=expected, label=f"execution-permit resource.{field}")
    safety = _mapping(root.get("fresh_pre_execution_safety"), label="execution-permit resource safety")
    for field in (
        "observed_immediately_before_execution_permit_lease",
        "exclusive_clean_window",
        "no_active_q30_or_q80_capture_child",
        "no_source_or_native_model_body_resident",
    ):
        _require(safety.get(field), expected=True, label=f"execution-permit resource safety.{field}")
    for field in ("swap_used_bytes", "swapouts_pages_delta"):
        if _integer(safety.get(field), label=f"execution-permit resource safety.{field}") != 0:
            raise ExecutionPermitError(f"execution-permit resource safety.{field} must remain zero")
    reclaimable = _integer(safety.get("reclaimable_bytes"), label="execution-permit resource reclaimable")
    minimum = _integer(
        safety.get("minimum_reclaimable_bytes_required"),
        label="execution-permit resource minimum reclaimable",
        minimum=1,
    )
    if reclaimable < minimum:
        raise ExecutionPermitError("execution-permit resource reclaimable floor is not met")


def _antecedent_evidence(antecedents: Antecedents) -> dict[str, dict[str, str | None]]:
    return {
        "metadata_range_authority": _evidence(antecedents.range_authority),
        "semantics_attester": _evidence(antecedents.semantics_attester),
        "post_hash_map_bridge": _evidence(antecedents.post_hash_map_bridge),
        "production_flat_map": _evidence(antecedents.flat_map),
        "production_hash_coverage": _evidence(antecedents.hash_coverage),
        "production_capture": _evidence(antecedents.production_capture),
        "production_outer_terminal": _evidence(antecedents.production_outer_terminal),
        "production_lease_release": _evidence(antecedents.production_release),
    }


def _future_runner_interface() -> dict[str, Any]:
    return {
        "execution_permit": {
            "schema": SCHEMA,
            "status": PREPARED_STATUS,
            "distinct_from_final_runtime_admission_schema": FINAL_RUNTIME_ADMISSION_SCHEMA,
            "cannot_be_supplied_as_legacy_runtime_admission": True,
            "cannot_be_consumed_without_a_fresh_execution_permit_lease": True,
        },
        "future_execution_lease": {
            "schema": EXECUTION_LEASE_SCHEMA,
            "status": EXECUTION_LEASE_STATUS,
            "must_bind_execution_permit_seal_and_fresh_resource_seal": True,
            "one_source_teacher_child_maximum": True,
            "automatic_retry_or_relaunch_forbidden": True,
        },
        "future_runner_command": [
            "ascension_qwen30_production_source_teacher_execution_runner",
            "--mode",
            "execute-permit",
            "--execution-permit",
            "ABSOLUTE_SEALED_EXECUTION_PERMIT_JSON",
            "--execution-permit-lease",
            "ABSOLUTE_SEALED_FRESH_EXECUTION_PERMIT_LEASE_JSON",
            "--source-root",
            "ABSOLUTE_CANONICAL_QWEN30_SOURCE_ROOT",
            "--capture-dir",
            "NEW_ABSOLUTE_SOURCE_TEACHER_CAPTURE_DIRECTORY",
            "--out",
            "NEW_ABSOLUTE_PRE_ATTESTATION_EXECUTION_CAPTURE_JSON",
        ],
        "bounded_source_run": {
            "source_layers": SOURCE_LAYERS,
            "source_forwards": SOURCE_FORWARDS,
            "source_layer_traversals": SOURCE_LAYERS * SOURCE_FORWARDS,
            "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
            "maximum_live_raw_bf16_windows": 1,
            "one_shard_handle_at_a_time": True,
            "whole_shard_cache_or_mmap_forbidden": True,
            "two_durable_f32le_source_vectors": SOURCE_VECTORS,
            "fsync_before_child_exit": True,
            "close_handles_and_zero_cache_before_receipt": True,
            "old_final_runtime_admission_is_not_a_pre_execution_input": True,
            "old_dual_bridge_is_not_a_pre_execution_input": True,
        },
        "post_execution_only_finalization": {
            "execution_capture": {
                "schema": EXECUTION_CAPTURE_SCHEMA,
                "status": EXECUTION_CAPTURE_STATUS,
                "receipt_written_after_vectors_fsync_and_cache_zero": True,
            },
            "operator_execution_attestation": {
                "schema": OPERATOR_ATTESTATION_SCHEMA,
                "status": OPERATOR_ATTESTATION_STATUS,
                "may_only_bind_a_real_execution_capture": True,
            },
            "reader_execution_attestation": {
                "schema": READER_ATTESTATION_SCHEMA,
                "status": READER_ATTESTATION_STATUS,
                "may_only_bind_a_real_execution_capture": True,
            },
            "final_teacher_runtime_admission": {
                "schema": FINAL_RUNTIME_ADMISSION_SCHEMA,
                "status": FINAL_RUNTIME_ADMISSION_STATUS,
                "must_bind_execution_capture_and_both_attestation_seals": True,
                "may_not_authorize_the_run_that_earned_it": True,
            },
        },
    }


def _execution_boundary() -> dict[str, bool]:
    return {
        "source_root_opened_or_statted": False,
        "source_payload_opened": False,
        "execution_permit_lease_issued_or_consumed": False,
        "source_teacher_child_spawned": False,
        "source_teacher_execution_or_logits": False,
        "operator_or_reader_execution_attestation_emitted": False,
        "final_runtime_admission_emitted": False,
        "native_or_model_gpu_server_hcli_action": False,
        "capture_or_terminal_written": False,
    }


def _load_antecedents(
    *,
    range_authority_path: Path,
    semantics_attester_path: Path,
    post_hash_map_bridge_path: Path,
    flat_map_path: Path,
    hash_coverage_path: Path,
    production_capture_path: Path,
    production_outer_terminal_path: Path,
    production_lease_release_path: Path,
) -> Antecedents:
    range_authority = _read_document(range_authority_path, label="range authority", sealed=False)
    source = _validate_range_authority(range_authority)
    semantics = _read_document(semantics_attester_path, label="semantics attester", sealed=False)
    _validate_semantics(semantics, source=source, range_authority=range_authority)
    flat_map = _read_document(flat_map_path, label="production flat map", sealed=True)
    _validate_flat_map(flat_map, source=source)
    coverage = _read_document(hash_coverage_path, label="production hash coverage", sealed=True)
    _validate_hash_coverage(coverage, flat_map=flat_map, source=source)
    capture = _read_document(production_capture_path, label="production capture", sealed=True)
    _validate_production_capture(capture, flat_map=flat_map, hash_coverage=coverage)
    terminal = _read_document(production_outer_terminal_path, label="production outer terminal", sealed=True)
    _validate_production_outer(terminal, capture=capture)
    release = _read_document(production_lease_release_path, label="production release", sealed=True)
    _validate_production_release(release, outer_terminal=terminal, capture=capture)
    bridge = _read_document(post_hash_map_bridge_path, label="post-hash-map bridge", sealed=True)
    antecedents = Antecedents(
        range_authority=range_authority,
        semantics_attester=semantics,
        post_hash_map_bridge=bridge,
        flat_map=flat_map,
        hash_coverage=coverage,
        production_capture=capture,
        production_outer_terminal=terminal,
        production_release=release,
        source=source,
    )
    _validate_post_hash_map_bridge(bridge, antecedents=antecedents)
    return antecedents


def build_execution_permit(
    *,
    range_authority_path: Path | None = None,
    semantics_attester_path: Path | None = None,
    post_hash_map_bridge_path: Path | None = None,
    flat_map_path: Path | None = None,
    hash_coverage_path: Path | None = None,
    production_capture_path: Path | None = None,
    production_outer_terminal_path: Path | None = None,
    production_lease_release_path: Path | None = None,
    fresh_resource_admission_path: Path | None = None,
) -> dict[str, Any]:
    """Return a sealed permit reservation or refusal without running anything."""
    required = (
        range_authority_path,
        semantics_attester_path,
        post_hash_map_bridge_path,
        flat_map_path,
        hash_coverage_path,
        production_capture_path,
        production_outer_terminal_path,
        production_lease_release_path,
    )
    blockers: list[str] = []
    antecedents: Antecedents | None = None
    if any(path is None for path in required):
        blockers.append("exact_non_fixture_post_hash_map_antecedents_absent")
    else:
        assert all(path is not None for path in required)
        try:
            antecedents = _load_antecedents(
                range_authority_path=range_authority_path,
                semantics_attester_path=semantics_attester_path,
                post_hash_map_bridge_path=post_hash_map_bridge_path,
                flat_map_path=flat_map_path,
                hash_coverage_path=hash_coverage_path,
                production_capture_path=production_capture_path,
                production_outer_terminal_path=production_outer_terminal_path,
                production_lease_release_path=production_lease_release_path,
            )
        except ExecutionPermitError as exc:
            blockers.append(f"exact_non_fixture_post_hash_map_antecedents_invalid:{exc}")
    resource: Document | None = None
    if fresh_resource_admission_path is None:
        blockers.append("fresh_execution_permit_zero_swap_resource_admission_absent")
    elif antecedents is None:
        blockers.append("fresh_execution_permit_resource_not_evaluated_without_antecedents")
    else:
        try:
            resource = _read_document(
                fresh_resource_admission_path,
                label="execution-permit resource admission",
                sealed=True,
            )
            _validate_resource(resource, antecedents=antecedents)
        except ExecutionPermitError as exc:
            blockers.append(f"fresh_execution_permit_zero_swap_resource_admission_invalid:{exc}")
    evidence: dict[str, Any] = (
        _antecedent_evidence(antecedents) if antecedents is not None else {"present": False}
    )
    if blockers:
        return seal(
            {
                "schema": SCHEMA,
                "status": REFUSED_STATUS,
                "prepared": False,
                "execution_permit_materialized": False,
                "spawn_permitted": False,
                "antecedents": evidence,
                "blockers": blockers,
                "future_runner_interface": _future_runner_interface(),
                "execution_boundary": _execution_boundary(),
                "claim_boundary": "CPU/file-only refusal. No source root/payload, execution permit lease, child, source teacher, attestation, final runtime admission, native/GPU/server/HCLI/TPS/TG/tournament action occurred.",
            }
        )
    assert antecedents is not None and resource is not None
    return seal(
        {
            "schema": SCHEMA,
            "status": PREPARED_STATUS,
            "prepared": True,
            "execution_permit_materialized": True,
            "execution_permit_consumed": False,
            "spawn_permitted": False,
            "antecedents": _antecedent_evidence(antecedents),
            "fresh_execution_permit_resource_admission": _evidence(resource),
            "future_runner_interface": _future_runner_interface(),
            "execution_boundary": _execution_boundary(),
            "claim_boundary": "Sealed non-circular execution-permit reservation only. It cannot be relabelled as a legacy runtime admission or a source-teacher result. A separate one-shot runner/lease may execute the bounded source pass; only its receipt-last capture can earn the two attestations and final runtime admission.",
        }
    )


def validate_fake_post_execution_finalization(
    *,
    execution_capture: Mapping[str, Any],
    operator_attestation: Mapping[str, Any],
    reader_attestation: Mapping[str, Any],
    final_runtime_admission: Mapping[str, Any],
    execution_permit_seal_sha256: str,
    execution_resource_seal_sha256: str,
    execution_lease_id: str,
    flat_map_canonical_document_sha256: str,
) -> None:
    """Check a fake post-run finalization sequence without source/process access."""
    _schema_status(
        execution_capture,
        schema=EXECUTION_CAPTURE_SCHEMA,
        status=EXECUTION_CAPTURE_STATUS,
        label="execution capture",
    )
    for field in (
        "source_teacher_execution_completed",
        "two_source_f32le_vectors_fsynced_before_child_exit",
        "source_handles_closed_before_child_exit",
        "reader_cache_zeroed_before_child_exit",
        "receipt_written_last",
    ):
        _require(execution_capture.get(field), expected=True, label=f"execution capture.{field}")
    for field in ("legacy_runtime_admission_used_before_run", "legacy_dual_bridge_used_before_run", "native_phase_started", "gpu_server_hcli_or_tps_action"):
        _require(execution_capture.get(field), expected=False, label=f"execution capture.{field}")
    geometry = _mapping(execution_capture.get("geometry"), label="execution capture geometry")
    for field, expected in (
        ("source_layers", SOURCE_LAYERS),
        ("source_forwards", SOURCE_FORWARDS),
        ("source_vectors_f32le", SOURCE_VECTORS),
        ("maximum_positioned_read_bytes", MAX_POSITIONED_READ_BYTES),
        ("maximum_live_raw_bf16_windows", 1),
    ):
        if _integer(geometry.get(field), label=f"execution capture geometry.{field}") != expected:
            raise ExecutionPermitError(f"execution capture geometry.{field} drifted")
    for field, expected in (
        ("execution_permit_seal_sha256", execution_permit_seal_sha256),
        ("execution_resource_seal_sha256", execution_resource_seal_sha256),
        ("execution_lease_id", execution_lease_id),
    ):
        if _text(execution_capture.get(field), label=f"execution capture.{field}", sha256=True) != expected:
            raise ExecutionPermitError(f"execution capture.{field} drifted")
    capture_seal = _text(execution_capture.get("seal_sha256"), label="execution capture seal", sha256=True)

    for document, schema, status, label in (
        (operator_attestation, OPERATOR_ATTESTATION_SCHEMA, OPERATOR_ATTESTATION_STATUS, "operator attestation"),
        (reader_attestation, READER_ATTESTATION_SCHEMA, READER_ATTESTATION_STATUS, "reader attestation"),
    ):
        _schema_status(document, schema=schema, status=status, label=label)
        _require(document.get("earned_after_real_execution_capture"), expected=True, label=f"{label}.earned_after_real_execution_capture")
        if _text(document.get("execution_capture_seal_sha256"), label=f"{label}.capture seal", sha256=True) != capture_seal:
            raise ExecutionPermitError(f"{label} execution capture binding drifted")
        if _text(document.get("execution_permit_seal_sha256"), label=f"{label}.permit seal", sha256=True) != execution_permit_seal_sha256:
            raise ExecutionPermitError(f"{label} execution permit binding drifted")
    operator_seal = _text(operator_attestation.get("seal_sha256"), label="operator attestation seal", sha256=True)
    reader_seal = _text(reader_attestation.get("seal_sha256"), label="reader attestation seal", sha256=True)

    _schema_status(
        final_runtime_admission,
        schema=FINAL_RUNTIME_ADMISSION_SCHEMA,
        status=FINAL_RUNTIME_ADMISSION_STATUS,
        label="final runtime admission",
    )
    _require(
        final_runtime_admission.get("earned_after_execution_capture"),
        expected=True,
        label="final runtime admission earned-after-capture",
    )
    _require(
        final_runtime_admission.get("may_not_authorize_its_own_prior_execution"),
        expected=True,
        label="final runtime admission noncircular boundary",
    )
    for field, expected in (
        ("execution_capture_seal_sha256", capture_seal),
        ("execution_permit_seal_sha256", execution_permit_seal_sha256),
        ("operator_execution_attestation_seal_sha256", operator_seal),
        ("reader_execution_attestation_seal_sha256", reader_seal),
    ):
        if _text(final_runtime_admission.get(field), label=f"final runtime admission.{field}", sha256=True) != expected:
            raise ExecutionPermitError(f"final runtime admission.{field} drifted")
    map_binding = _mapping(final_runtime_admission.get("flat_runtime_range_map"), label="final runtime admission map")
    if (
        map_binding.get("schema") != FLAT_MAP_SCHEMA
        or _text(map_binding.get("document_sha256"), label="final runtime admission map SHA", sha256=True)
        != flat_map_canonical_document_sha256
    ):
        raise ExecutionPermitError("final runtime admission map binding drifted")


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise ExecutionPermitError("--out must be a new absolute JSON path below an existing directory")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range-authority", type=Path)
    parser.add_argument("--semantics-attester", type=Path)
    parser.add_argument("--post-hash-map-bridge", type=Path)
    parser.add_argument("--flat-runtime-range-map", type=Path)
    parser.add_argument("--hash-coverage-attestation", type=Path)
    parser.add_argument("--production-capture", type=Path)
    parser.add_argument("--production-outer-terminal", type=Path)
    parser.add_argument("--production-lease-release", type=Path)
    parser.add_argument("--fresh-resource-admission", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_execution_permit(
            range_authority_path=args.range_authority,
            semantics_attester_path=args.semantics_attester,
            post_hash_map_bridge_path=args.post_hash_map_bridge,
            flat_map_path=args.flat_runtime_range_map,
            hash_coverage_path=args.hash_coverage_attestation,
            production_capture_path=args.production_capture,
            production_outer_terminal_path=args.production_outer_terminal,
            production_lease_release_path=args.production_lease_release,
            fresh_resource_admission_path=args.fresh_resource_admission,
        )
        _write_new(args.out, result)
    except ExecutionPermitError as exc:
        print(f"Q30 execution-permit preflight could not write a sealed result: {exc}")
        return 2
    print(json.dumps({"output": str(args.out.resolve()), "status": result["status"], "seal_sha256": result["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
