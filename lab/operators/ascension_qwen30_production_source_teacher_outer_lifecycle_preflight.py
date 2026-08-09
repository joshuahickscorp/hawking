#!/usr/bin/env python3
"""CPU-only lifecycle preflight for the distinct Q30 production teacher phase.

The already-earned production hash scan is immutable *antecedent* evidence.  It
is deliberately not a teacher admission: this module requires a future sealed
post-hash-map bridge, the existing source-child dual bridge plus runtime
admission, and a fresh source-teacher resource/lease before it can emit a
non-authorizing PREPARED reservation.  It never accepts a fixture map or a
hash-only receipt relabelled as a teacher admission.

No mode in this module opens a source root, starts a child, issues a lease, or
touches model/GPU/server/HCLI/native surfaces.  ``validate_fake_future_lifecycle``
only checks in-memory fake receipt mappings for focused tests and future
controller integration.
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

from lab.operators import (
    ascension_qwen30_guarded_streamed_source_oracle_outer_controller as guarded_outer,
)
from lab.receipts import SealIntegrityError, seal, verify

SCHEMA = "hawking.ascension.qwen30_production_source_teacher_outer_lifecycle_preflight.v1"
PREPARED_STATUS = (
    "PREPARED_QWEN30_PRODUCTION_SOURCE_TEACHER_OUTER_LIFECYCLE_NOT_EXECUTED"
)
REFUSED_STATUS = (
    "REFUSED_QWEN30_PRODUCTION_SOURCE_TEACHER_OUTER_LIFECYCLE_PREREQUISITES_ABSENT_OR_INVALID"
)

PRODUCTION_OUTER_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_outer_terminal.v1"
)
PRODUCTION_OUTER_STATUS = (
    "CAPTURED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_OUTER_TERMINAL_NOT_SOURCE_TEACHER"
)
PRODUCTION_CHILD_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_capture.v1"
)
PRODUCTION_CHILD_STATUS = (
    "CAPTURED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_NOT_SOURCE_TEACHER"
)
FLAT_MAP_SCHEMA = "hawking.ascension.qwen30_source_bf16_range_map.v1"
HASH_COVERAGE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_production_hash_coverage_attestation.v1"
)
HASH_COVERAGE_STATUS = (
    "EARNED_QWEN30_STREAMED_SOURCE_PRODUCTION_HASH_COVERAGE_ATTESTED_NOT_SOURCE_TEACHER"
)
PRODUCTION_RELEASE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_range_admission_production_quiet_lease_release.v1"
)
PRODUCTION_RELEASE_STATUS = (
    "RELEASED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_LEASE_AFTER_OUTER_TERMINAL"
)

# Owned by qwen30_teacher_bridge.  This outer only consumes its sealed output.
POST_HASH_MAP_BRIDGE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_teacher_post_hash_map_bridge.v1"
)
POST_HASH_MAP_BRIDGE_STATUS = (
    "PREPARED_QWEN30_STREAMED_SOURCE_TEACHER_POST_HASH_MAP_BRIDGE_NOT_EXECUTED"
)

# These exact child-ABI identities are deliberately retained.  The post-scan
# bridge reserves them but cannot itself substitute for either one.
DUAL_BRIDGE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_teacher_dual_attestation_runtime_admission.v1"
)
DUAL_BRIDGE_STATUS = (
    "PREPARED_QWEN30_STREAMED_SOURCE_TEACHER_DUAL_ATTESTATION_RUNTIME_ADMISSION_NOT_EXECUTED"
)
RUNTIME_ADMISSION_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_teacher_runtime_range_admission.v1"
)
RUNTIME_ADMISSION_STATUS = (
    "EARNED_QWEN30_STREAMED_SOURCE_TEACHER_RUNTIME_RANGE_ADMISSION_NO_MODEL_RESIDENCY"
)
OPERATOR_ATTESTATION_SCHEMA = (
    "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_execution_attestation.v1"
)
OPERATOR_ATTESTATION_STATUS = (
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_ATTESTED"
)
RANGE_READER_ATTESTATION_SCHEMA = (
    "hawking.ascension.qwen30_layer_streamed_source_bf16_exact_semantics_attestation.v1"
)
RANGE_READER_ATTESTATION_STATUS = (
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_BF16_EXACT_SEMANTICS_ATTESTED"
)

SOURCE_RESOURCE_SCHEMA = (
    "hawking.ascension.qwen30_production_source_teacher_resource_admission.v1"
)
SOURCE_RESOURCE_STATUS = (
    "PREPARED_QWEN30_PRODUCTION_SOURCE_TEACHER_RESOURCE_ADMISSION_ZERO_SWAP_NOT_LEASED"
)
SOURCE_LEASE_SCHEMA = guarded_outer.SOURCE_LEASE_SCHEMA
SOURCE_LEASE_STATUS = guarded_outer.SOURCE_LEASE_STATUS
SOURCE_TERMINAL_SCHEMA = guarded_outer.SOURCE_TERMINAL_SCHEMA
SOURCE_TERMINAL_STATUS = guarded_outer.SOURCE_TERMINAL_STATUS
SOURCE_EVICTION_SCHEMA = guarded_outer.SOURCE_EVICTION_SCHEMA
SOURCE_EVICTION_STATUS = guarded_outer.SOURCE_EVICTION_STATUS
NATIVE_LEASE_SCHEMA = guarded_outer.NATIVE_LEASE_SCHEMA
NATIVE_LEASE_STATUS = guarded_outer.NATIVE_LEASE_STATUS

REPLAY_SCHEMA = (
    "hawking.ascension.qwen30_production_source_teacher_outer_replay_reservation.v1"
)
REPLAY_STATUS = (
    "RESERVED_QWEN30_PRODUCTION_SOURCE_TEACHER_ONE_SHOT_CAPTURE_NOT_SPAWNED"
)
SOURCE_CHILD_CAPTURE_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_teacher_child_execution_evidence.v1"
)
SOURCE_CHILD_CAPTURE_STATUS = (
    "CAPTURED_QWEN30_STREAMED_SOURCE_TEACHER_CHILD_TWO_F32LE_LOGITS_NOT_NATIVE_PHASE"
)

MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_POSITIONED_READ_BYTES = 1024 * 1024
SOURCE_SHARDS = 16
SOURCE_TENSORS = 18_867
SOURCE_LAYERS = 48
SOURCE_FORWARDS = 370
SOURCE_VECTORS = 2
NATIVE_VECTORS = 4


class ProductionSourceTeacherOuterError(RuntimeError):
    """A production source-teacher preflight input is unsafe or incomplete."""


@dataclass(frozen=True)
class Document:
    path: Path
    document: dict[str, Any]
    raw_document_sha256: str
    canonical_document_sha256: str
    seal_sha256: str


@dataclass(frozen=True)
class Antecedents:
    production_outer_terminal: Document
    production_child_capture: Document
    production_flat_map: Document
    production_hash_coverage: Document
    production_lease_release: Document


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProductionSourceTeacherOuterError(f"{label} must be an object")
    return dict(value)


def _sequence(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProductionSourceTeacherOuterError(f"{label} must be an array")
    return list(value)


def _text(value: object, *, label: str, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise ProductionSourceTeacherOuterError(f"{label} must be non-empty text")
    if sha256 and (
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProductionSourceTeacherOuterError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProductionSourceTeacherOuterError(f"{label} must be an integer >= {minimum}")
    return value


def _require(value: object, *, expected: bool, label: str) -> None:
    if value is not expected:
        raise ProductionSourceTeacherOuterError(f"{label} must be {expected!r}")


def _schema_status(
    document: Mapping[str, Any], *, schema: str, status: str, label: str
) -> None:
    if document.get("schema") != schema or document.get("status") != status:
        raise ProductionSourceTeacherOuterError(f"{label} schema/status drifted")


def _regular_json(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.suffix != ".json":
        raise ProductionSourceTeacherOuterError(f"{label} must be an absolute .json path")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ProductionSourceTeacherOuterError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProductionSourceTeacherOuterError(f"{label} must be a regular non-symlink file")
    if metadata.st_size <= 0 or metadata.st_size > MAX_METADATA_BYTES:
        raise ProductionSourceTeacherOuterError(f"{label} has invalid metadata size")
    return path.resolve(strict=True)


def _sealed(path: Path, *, label: str) -> Document:
    clean = _regular_json(path, label=label)
    try:
        raw_bytes = clean.read_bytes()
        raw = json.loads(raw_bytes)
        checked = verify(raw, label=label)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, SealIntegrityError) as exc:
        raise ProductionSourceTeacherOuterError(f"{label} is absent or invalid: {exc}") from exc
    document = _mapping(checked, label=label)
    return Document(
        path=clean,
        document=document,
        raw_document_sha256=_sha256_bytes(raw_bytes),
        canonical_document_sha256=_sha256_bytes(_canonical_json(document)),
        seal_sha256=_text(document.get("seal_sha256"), label=f"{label} seal", sha256=True),
    )


def _evidence(document: Document) -> dict[str, str]:
    return {
        "path": str(document.path),
        "raw_document_sha256": document.raw_document_sha256,
        "canonical_document_sha256": document.canonical_document_sha256,
        "seal_sha256": document.seal_sha256,
    }


def _pointer(
    value: object,
    *,
    expected: Document,
    label: str,
    require_canonical: bool = True,
    require_path: bool = True,
) -> None:
    pointer = _mapping(value, label=label)
    if require_path and _text(pointer.get("path"), label=f"{label}.path") != str(expected.path):
        raise ProductionSourceTeacherOuterError(f"{label} path drifted")
    if (
        _text(pointer.get("raw_document_sha256"), label=f"{label}.raw_document_sha256", sha256=True)
        != expected.raw_document_sha256
        or _text(pointer.get("seal_sha256"), label=f"{label}.seal_sha256", sha256=True)
        != expected.seal_sha256
    ):
        raise ProductionSourceTeacherOuterError(f"{label} does not bind the supplied sealed document")
    if require_canonical and (
        _text(
            pointer.get("canonical_document_sha256"),
            label=f"{label}.canonical_document_sha256",
            sha256=True,
        )
        != expected.canonical_document_sha256
    ):
        raise ProductionSourceTeacherOuterError(f"{label} canonical identity drifted")


def _reject_fixture_identity(value: object, *, label: str) -> None:
    """Reject a fixture flag, fixture schema/status, or production-forbidden marker."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in {"schema", "status"} and isinstance(child, str):
                if "fixture" in child.lower() or "synthetic" in child.lower():
                    raise ProductionSourceTeacherOuterError(
                        f"{label}.{key} carries fixture-only identity {child!r}"
                    )
            if key in {
                "fixture_only",
                "synthetic_fixture_only",
                "production_adapter_forbidden",
            } and child is True:
                raise ProductionSourceTeacherOuterError(
                    f"{label}.{key} marks fixture-only or production-forbidden evidence"
                )
            _reject_fixture_identity(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_fixture_identity(child, label=f"{label}[{index}]")


def _validate_flat_map(document: Document) -> None:
    root = document.document
    if root.get("schema") != FLAT_MAP_SCHEMA:
        raise ProductionSourceTeacherOuterError("production flat map schema drifted")
    _reject_fixture_identity(root, label="production flat map")
    _require(root.get("fixture_only"), expected=False, label="production flat map.fixture_only")
    _require(
        root.get("synthetic_fixture_only"),
        expected=False,
        label="production flat map.synthetic_fixture_only",
    )
    if _integer(root.get("source_tensor_count"), label="production flat map tensor count") != SOURCE_TENSORS:
        raise ProductionSourceTeacherOuterError("production flat map tensor count drifted")
    if _integer(root.get("maximum_window_bytes"), label="production flat map maximum window") != MAX_POSITIONED_READ_BYTES:
        raise ProductionSourceTeacherOuterError("production flat map maximum window drifted")
    if len(_sequence(root.get("shards"), label="production flat map shards")) != SOURCE_SHARDS:
        raise ProductionSourceTeacherOuterError("production flat map shard count drifted")
    if len(_sequence(root.get("tensors"), label="production flat map tensors")) != SOURCE_TENSORS:
        raise ProductionSourceTeacherOuterError("production flat map tensor list drifted")
    # The canonical Q30 source revision is a Git object identity (40 hex),
    # while document/payload identities are SHA-256 (64 hex).  Preserve both
    # accepted immutable encodings instead of misclassifying the live map.
    revision = _text(root.get("source_revision"), label="production flat map source revision")
    if (
        len(revision) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ProductionSourceTeacherOuterError(
            "production flat map source revision must be a lowercase Git/SHA identity"
        )
    source_index = _mapping(root.get("source_index"), label="production flat map source index")
    _text(
        source_index.get("sha256"),
        label="production flat map source index SHA",
        sha256=True,
    )


def _validate_hash_coverage(document: Document, *, flat_map: Document) -> None:
    root = document.document
    _schema_status(
        root,
        schema=HASH_COVERAGE_SCHEMA,
        status=HASH_COVERAGE_STATUS,
        label="production hash coverage",
    )
    _reject_fixture_identity(root, label="production hash coverage")
    for field in (
        "fixture_only",
        "synthetic_fixture_only",
        "source_teacher_execution_or_logits",
        "source_teacher_runtime_admission_earned",
        "operator_or_reader_execution_attestation_emitted",
    ):
        _require(root.get(field), expected=False, label=f"production hash coverage.{field}")
    _require(
        root.get("production_hash_coverage_earned"),
        expected=True,
        label="production hash coverage earned",
    )
    _pointer(
        root.get("flat_runtime_range_map"),
        expected=flat_map,
        label="production hash coverage flat-map binding",
    )
    bounded = _mapping(root.get("bounded_positioned_reader"), label="production hash coverage reader")
    expected_reader = {
        "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
        "maximum_live_raw_bf16_windows": 1,
    }
    for field, expected in expected_reader.items():
        if _integer(bounded.get(field), label=f"production hash coverage reader.{field}") != expected:
            raise ProductionSourceTeacherOuterError(f"production hash coverage reader.{field} drifted")
    for field in (
        "one_shard_handle_at_a_time",
        "whole_shard_cache_or_mmap_forbidden",
        "cache_zeroed_after_every_visit_and_before_receipt",
    ):
        _require(bounded.get(field), expected=True, label=f"production hash coverage reader.{field}")
    coverage = _mapping(root.get("coverage"), label="production hash coverage counts")
    for field, expected in (("source_shards", SOURCE_SHARDS), ("source_tensors", SOURCE_TENSORS)):
        if _integer(coverage.get(field), label=f"production hash coverage.{field}") != expected:
            raise ProductionSourceTeacherOuterError(f"production hash coverage {field} drifted")
    _text(
        coverage.get("source_index_sha256"),
        label="production hash coverage source index SHA",
        sha256=True,
    )


def _validate_production_child(
    document: Document, *, flat_map: Document, hash_coverage: Document
) -> None:
    root = document.document
    _schema_status(
        root,
        schema=PRODUCTION_CHILD_SCHEMA,
        status=PRODUCTION_CHILD_STATUS,
        label="production hash child",
    )
    _reject_fixture_identity(root, label="production hash child")
    for field in (
        "fixture_only",
        "synthetic_fixture_only",
        "source_teacher_or_logits_executed",
        "source_teacher_runtime_admission_earned",
        "operator_or_reader_execution_attestation_emitted",
        "model_gpu_server_hcli_or_tps_action",
    ):
        _require(root.get(field), expected=False, label=f"production hash child.{field}")
    for field in (
        "production_hash_scan_earned",
        "source_handles_closed",
        "reader_cache_zeroed",
        "receipt_written_last",
    ):
        _require(root.get(field), expected=True, label=f"production hash child.{field}")
    _pointer(
        root.get("flat_runtime_range_map"),
        expected=flat_map,
        label="production hash child flat-map binding",
    )
    _pointer(
        root.get("hash_coverage_attestation"),
        expected=hash_coverage,
        label="production hash child coverage binding",
    )
    geometry = _mapping(root.get("geometry"), label="production hash child geometry")
    for field, expected in (
        ("source_shards", SOURCE_SHARDS),
        ("source_tensors", SOURCE_TENSORS),
        ("maximum_positioned_read_bytes", MAX_POSITIONED_READ_BYTES),
        ("maximum_live_raw_bf16_windows", 1),
    ):
        if _integer(geometry.get(field), label=f"production hash child geometry.{field}") != expected:
            raise ProductionSourceTeacherOuterError(f"production hash child geometry.{field} drifted")


def _validate_production_outer(document: Document, *, child: Document) -> None:
    root = document.document
    _schema_status(
        root,
        schema=PRODUCTION_OUTER_SCHEMA,
        status=PRODUCTION_OUTER_STATUS,
        label="production outer terminal",
    )
    for field in (
        "child_reaped",
        "terminal_receipt_written_after_child_capture",
        "terminal_receipt_written_last",
        "automatic_retry_disabled",
        "lease_reuse_prohibited",
    ):
        _require(root.get(field), expected=True, label=f"production outer terminal.{field}")
    _require(root.get("child_timed_out"), expected=False, label="production outer terminal.child_timed_out")
    if _integer(root.get("child_exit_code"), label="production outer terminal child exit") != 0:
        raise ProductionSourceTeacherOuterError("production outer child exit drifted")
    _pointer(
        root.get("child_capture"),
        expected=child,
        label="production outer child binding",
        require_canonical=False,
    )
    if _text(root.get("child_capture_seal_sha256"), label="production outer child seal", sha256=True) != child.seal_sha256:
        raise ProductionSourceTeacherOuterError("production outer child seal drifted")


def _validate_production_release(
    document: Document, *, outer_terminal: Document, child: Document
) -> None:
    root = document.document
    _schema_status(
        root,
        schema=PRODUCTION_RELEASE_SCHEMA,
        status=PRODUCTION_RELEASE_STATUS,
        label="production lease release",
    )
    for field in (
        "release_after_outer_terminal",
        "one_shot_lease_finalized",
        "retry_or_relaunch_forbidden",
    ):
        _require(root.get(field), expected=True, label=f"production lease release.{field}")
    for field in (
        "source_teacher_or_logits_authorized",
        "native_or_gpu_server_hcli_authorized",
        "artifacts_deleted_or_evicted",
    ):
        _require(root.get(field), expected=False, label=f"production lease release.{field}")
    if _text(root.get("outer_terminal_seal_sha256"), label="production release outer seal", sha256=True) != outer_terminal.seal_sha256:
        raise ProductionSourceTeacherOuterError("production release outer-terminal binding drifted")
    if _text(root.get("child_capture_seal_sha256"), label="production release child seal", sha256=True) != child.seal_sha256:
        raise ProductionSourceTeacherOuterError("production release child binding drifted")


def _validate_antecedents(
    *,
    production_outer_terminal_path: Path,
    production_child_capture_path: Path,
    production_flat_map_path: Path,
    production_hash_coverage_path: Path,
    production_lease_release_path: Path,
) -> Antecedents:
    outer_terminal = _sealed(production_outer_terminal_path, label="production outer terminal")
    child = _sealed(production_child_capture_path, label="production hash child")
    flat_map = _sealed(production_flat_map_path, label="production flat map")
    hash_coverage = _sealed(production_hash_coverage_path, label="production hash coverage")
    release = _sealed(production_lease_release_path, label="production lease release")
    _validate_flat_map(flat_map)
    _validate_hash_coverage(hash_coverage, flat_map=flat_map)
    _validate_production_child(child, flat_map=flat_map, hash_coverage=hash_coverage)
    _validate_production_outer(outer_terminal, child=child)
    _validate_production_release(release, outer_terminal=outer_terminal, child=child)
    return Antecedents(
        production_outer_terminal=outer_terminal,
        production_child_capture=child,
        production_flat_map=flat_map,
        production_hash_coverage=hash_coverage,
        production_lease_release=release,
    )


def _antecedent_evidence(antecedents: Antecedents) -> dict[str, dict[str, str]]:
    return {
        "production_outer_terminal": _evidence(antecedents.production_outer_terminal),
        "production_child_capture": _evidence(antecedents.production_child_capture),
        "production_flat_map": _evidence(antecedents.production_flat_map),
        "production_hash_coverage": _evidence(antecedents.production_hash_coverage),
        "production_lease_release": _evidence(antecedents.production_lease_release),
    }


def _validate_post_hash_map_bridge(document: Document, *, antecedents: Antecedents) -> None:
    root = document.document
    _schema_status(
        root,
        schema=POST_HASH_MAP_BRIDGE_SCHEMA,
        status=POST_HASH_MAP_BRIDGE_STATUS,
        label="post-hash-map bridge",
    )
    _reject_fixture_identity(root, label="post-hash-map bridge")
    _require(root.get("prepared"), expected=True, label="post-hash-map bridge.prepared")
    for field, expected in (
        ("execution_authorized", False),
        ("runtime_admission_earned", False),
        ("dual_attestation_runtime_admission_emitted", False),
    ):
        _require(root.get(field), expected=expected, label=f"post-hash-map bridge.{field}")
    bindings = _mapping(
        root.get("post_hash_map_antecedents"), label="post-hash-map bridge antecedents"
    )
    for field, expected in (
        ("production_outer_terminal", antecedents.production_outer_terminal),
        ("production_child_capture", antecedents.production_child_capture),
        ("production_flat_map", antecedents.production_flat_map),
        ("production_hash_coverage", antecedents.production_hash_coverage),
        ("production_lease_release", antecedents.production_lease_release),
    ):
        _pointer(bindings.get(field), expected=expected, label=f"post-hash-map bridge.{field}")
    validated_scan = _mapping(
        root.get("validated_production_hash_scan"),
        label="post-hash-map bridge validated production scan",
    )
    for field in (
        "non_fixture_production_flat_map",
        "all_full_shard_and_raw_bf16_hash_coverage_bound",
        "one_shard_handle_at_a_time",
        "reader_cache_zeroed_before_hash_scan_receipt",
        "source_handles_closed_before_hash_scan_receipt",
        "outer_child_reaped_successfully",
        "lease_finalized_after_outer_terminal",
    ):
        _require(validated_scan.get(field), expected=True, label=f"post-hash-map bridge validated scan.{field}")
    for field in (
        "source_teacher_execution_or_logits",
        "operator_or_reader_execution_attestation_emitted",
        "source_teacher_runtime_admission_earned",
    ):
        _require(validated_scan.get(field), expected=False, label=f"post-hash-map bridge validated scan.{field}")
    for field, expected in (
        ("source_shards", SOURCE_SHARDS),
        ("source_tensors", SOURCE_TENSORS),
        ("maximum_positioned_read_bytes", MAX_POSITIONED_READ_BYTES),
        ("maximum_live_raw_bf16_windows", 1),
        ("one_shot_replay_reservation_attempt", 1),
    ):
        if _integer(validated_scan.get(field), label=f"post-hash-map bridge validated scan.{field}") != expected:
            raise ProductionSourceTeacherOuterError(f"post-hash-map bridge validated scan.{field} drifted")
    reservation = _mapping(
        root.get("future_source_teacher_provenance_reservation"),
        label="post-hash-map bridge future source-teacher reservation",
    )
    if reservation.get("reservation_status") != "NOT_EXECUTED":
        raise ProductionSourceTeacherOuterError(
            "post-hash-map bridge future reservation must remain NOT_EXECUTED"
        )
    for field in (
        "post_hash_map_bridge_is_not_runtime_admission",
        "post_hash_map_bridge_is_not_dual_attestation_bridge",
    ):
        _require(reservation.get(field), expected=True, label=f"post-hash-map bridge reservation.{field}")
    runtime = _mapping(
        reservation.get("runtime_admission"),
        label="post-hash-map bridge reserved runtime admission",
    )
    _schema_status(
        runtime,
        schema=RUNTIME_ADMISSION_SCHEMA,
        status=RUNTIME_ADMISSION_STATUS,
        label="post-hash-map bridge reserved runtime admission",
    )
    for field in (
        "must_be_sealed",
        "must_bind_post_hash_map_antecedents",
        "must_bind_flat_map_and_coverage_canonical_hashes",
        "must_bind_both_future_execution_attestation_seals",
        "must_precede_any_source_root_or_payload_open",
        "not_emitted_by_this_bridge",
    ):
        _require(runtime.get(field), expected=True, label=f"post-hash-map bridge reserved runtime.{field}")
    dual = _mapping(
        reservation.get("dual_attestation_runtime_admission"),
        label="post-hash-map bridge reserved dual bridge",
    )
    _schema_status(
        dual,
        schema=DUAL_BRIDGE_SCHEMA,
        status=DUAL_BRIDGE_STATUS,
        label="post-hash-map bridge reserved dual bridge",
    )
    for field in (
        "must_be_sealed",
        "must_bind_this_post_hash_map_bridge_seal_sha256",
        "must_bind_runtime_admission_seal_sha256",
        "must_preserve_existing_source_teacher_child_schema_resolution",
        "not_emitted_by_this_bridge",
    ):
        _require(dual.get(field), expected=True, label=f"post-hash-map bridge reserved dual.{field}")
    shape = _mapping(
        reservation.get("existing_source_teacher_child_compatible_shape"),
        label="post-hash-map bridge reserved child shape",
    )
    resolution = _mapping(shape.get("schema_resolution"), label="post-hash-map bridge reserved child resolution")
    if (
        resolution.get("runtime_range_map_schema") != FLAT_MAP_SCHEMA
        or resolution.get("runtime_admission_schema") != RUNTIME_ADMISSION_SCHEMA
        or resolution.get("runtime_admission_status_only_after_bounded_source_validation")
        != RUNTIME_ADMISSION_STATUS
    ):
        raise ProductionSourceTeacherOuterError("post-hash-map bridge reserved child resolution drifted")
    for field, schema, status in (
        (
            "operator_accumulation_execution_attestation",
            OPERATOR_ATTESTATION_SCHEMA,
            OPERATOR_ATTESTATION_STATUS,
        ),
        (
            "range_reader_exact_semantics_attestation",
            RANGE_READER_ATTESTATION_SCHEMA,
            RANGE_READER_ATTESTATION_STATUS,
        ),
    ):
        _schema_status(
            _mapping(resolution.get(field), label=f"post-hash-map bridge reserved {field}"),
            schema=schema,
            status=status,
            label=f"post-hash-map bridge reserved {field}",
        )
    for field in (
        "both_execution_attestations_required_after_source_child",
        "runtime_range_admission_required_before_payload_open",
        "bridge_does_not_authorize_execution",
    ):
        _require(resolution.get(field), expected=True, label=f"post-hash-map bridge reserved resolution.{field}")
    worker = _mapping(shape.get("future_source_worker"), label="post-hash-map bridge reserved source worker")
    for field, expected in (
        ("maximum_positioned_read_bytes", MAX_POSITIONED_READ_BYTES),
        ("source_layers", SOURCE_LAYERS),
        ("source_forwards", SOURCE_FORWARDS),
        ("source_f32le_vectors", SOURCE_VECTORS),
        ("native_f32le_vectors", NATIVE_VECTORS),
    ):
        if _integer(worker.get(field), label=f"post-hash-map bridge reserved worker.{field}") != expected:
            raise ProductionSourceTeacherOuterError(f"post-hash-map bridge reserved worker.{field} drifted")
    for field in (
        "one_bounded_window_only",
        "source_payloads_durable_before_eviction",
        "close_handles_and_clear_cache_before_eviction_receipt",
        "separate_native_four_vector_phase_required",
    ):
        _require(worker.get(field), expected=True, label=f"post-hash-map bridge reserved worker.{field}")
    cycle = _mapping(root.get("admission_before_open_cycle"), label="post-hash-map bridge admission-before-open cycle")
    for field in (
        "runtime_admission_must_be_earned_before_source_root_open",
        "runtime_producer_requires_bounded_source_validation_and_both_execution_attestations",
        "existing_source_teacher_child_requires_runtime_admission_before_source_root_open",
        "bridge_does_not_relax_or_reorder_any_requirement",
    ):
        _require(cycle.get(field), expected=True, label=f"post-hash-map bridge cycle.{field}")
    _require(cycle.get("resolved"), expected=False, label="post-hash-map bridge cycle.resolved")
    execution_boundary = _mapping(
        root.get("execution_boundary"), label="post-hash-map bridge execution boundary"
    )
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
        _require(execution_boundary.get(field), expected=False, label=f"post-hash-map bridge boundary.{field}")


def _validate_dual_bridge(document: Document, *, post_hash_map_bridge: Document) -> None:
    root = document.document
    _schema_status(root, schema=DUAL_BRIDGE_SCHEMA, status=DUAL_BRIDGE_STATUS, label="dual bridge")
    _reject_fixture_identity(root, label="dual bridge")
    _pointer(
        root.get("post_hash_map_bridge"),
        expected=post_hash_map_bridge,
        label="dual bridge post-hash-map bridge",
    )
    resolution = _mapping(root.get("schema_resolution"), label="dual bridge schema resolution")
    if resolution.get("runtime_range_map_schema") != FLAT_MAP_SCHEMA:
        raise ProductionSourceTeacherOuterError("dual bridge runtime map schema drifted")
    if resolution.get("runtime_admission_schema") != RUNTIME_ADMISSION_SCHEMA:
        raise ProductionSourceTeacherOuterError("dual bridge runtime admission schema drifted")
    if resolution.get("runtime_admission_status_only_after_bounded_source_validation") != RUNTIME_ADMISSION_STATUS:
        raise ProductionSourceTeacherOuterError("dual bridge runtime admission status drifted")
    for field in (
        "both_execution_attestations_required_after_source_child",
        "runtime_range_admission_required_before_payload_open",
        "bridge_does_not_authorize_execution",
    ):
        _require(resolution.get(field), expected=True, label=f"dual bridge resolution.{field}")
    worker = _mapping(root.get("future_source_worker"), label="dual bridge future source worker")
    for field, expected in (
        ("maximum_positioned_read_bytes", MAX_POSITIONED_READ_BYTES),
        ("source_layers", SOURCE_LAYERS),
        ("source_forwards", SOURCE_FORWARDS),
        ("source_f32le_vectors", SOURCE_VECTORS),
        ("native_f32le_vectors", NATIVE_VECTORS),
    ):
        if _integer(worker.get(field), label=f"dual bridge worker.{field}") != expected:
            raise ProductionSourceTeacherOuterError(f"dual bridge worker.{field} drifted")
    for field in (
        "one_bounded_window_only",
        "source_payloads_durable_before_eviction",
        "close_handles_and_clear_cache_before_eviction_receipt",
        "separate_native_four_vector_phase_required",
    ):
        _require(worker.get(field), expected=True, label=f"dual bridge worker.{field}")


def _validate_runtime_execution_admission(
    document: Document,
    *,
    post_hash_map_bridge: Document,
    dual_bridge: Document,
    flat_map: Document,
) -> None:
    root = document.document
    _schema_status(
        root,
        schema=RUNTIME_ADMISSION_SCHEMA,
        status=RUNTIME_ADMISSION_STATUS,
        label="runtime execution admission",
    )
    _reject_fixture_identity(root, label="runtime execution admission")
    _pointer(
        root.get("post_hash_map_bridge"),
        expected=post_hash_map_bridge,
        label="runtime execution admission post-hash-map bridge",
    )
    _require(
        root.get("source_teacher_execution_admission_not_hash_only"),
        expected=True,
        label="runtime execution admission not-hash-only boundary",
    )
    _require(
        root.get("production_hash_scan_is_not_source_teacher_execution"),
        expected=True,
        label="runtime execution admission hash-scan boundary",
    )
    flat_binding = _mapping(
        root.get("flat_runtime_range_map"), label="runtime execution admission flat map"
    )
    if flat_binding.get("schema") != FLAT_MAP_SCHEMA:
        raise ProductionSourceTeacherOuterError("runtime execution admission flat-map schema drifted")
    if _text(
        flat_binding.get("document_sha256"),
        label="runtime execution admission flat-map document SHA",
        sha256=True,
    ) != flat_map.canonical_document_sha256:
        raise ProductionSourceTeacherOuterError(
            "runtime execution admission does not bind the earned flat-map canonical identity"
        )
    bounded = _mapping(
        root.get("bounded_positioned_reader"), label="runtime execution admission reader"
    )
    for field, expected in (
        ("maximum_positioned_read_bytes", MAX_POSITIONED_READ_BYTES),
        ("maximum_live_raw_bf16_windows", 1),
    ):
        if _integer(bounded.get(field), label=f"runtime execution admission reader.{field}") != expected:
            raise ProductionSourceTeacherOuterError(f"runtime execution admission reader.{field} drifted")
    for field in (
        "no_mmap_or_full_shard_cache",
        "no_model_residency",
        "payload_open_requires_fresh_source_lease",
    ):
        _require(bounded.get(field), expected=True, label=f"runtime execution admission reader.{field}")
    if _text(
        root.get("dual_bridge_seal_sha256"),
        label="runtime execution admission dual bridge seal",
        sha256=True,
    ) != dual_bridge.seal_sha256:
        raise ProductionSourceTeacherOuterError("runtime execution admission dual bridge drifted")
    boundary = _mapping(root.get("execution_boundary"), label="runtime execution admission boundary")
    for field in (
        "source_tensor_payload_opened",
        "source_model_loaded_or_instantiated",
        "gpu_or_metal_invoked",
        "server_started_or_contacted",
        "hcli_invoked",
        "lease_issued_or_consumed",
    ):
        _require(boundary.get(field), expected=False, label=f"runtime execution admission boundary.{field}")


def _validate_source_resource(
    document: Document,
    *,
    post_hash_map_bridge: Document,
    dual_bridge: Document,
    runtime_execution_admission: Document,
) -> None:
    root = document.document
    _schema_status(
        root,
        schema=SOURCE_RESOURCE_SCHEMA,
        status=SOURCE_RESOURCE_STATUS,
        label="source-teacher resource admission",
    )
    _reject_fixture_identity(root, label="source-teacher resource admission")
    for field, expected in (
        ("source_root_opened_or_statted", False),
        ("source_teacher_or_native_child_started", False),
        ("lease_issued_or_consumed", False),
    ):
        _require(root.get(field), expected=expected, label=f"source resource.{field}")
    for field, expected in (
        ("post_hash_map_bridge", post_hash_map_bridge),
        ("dual_attestation_runtime_admission", dual_bridge),
        ("runtime_execution_admission", runtime_execution_admission),
    ):
        _pointer(root.get(field), expected=expected, label=f"source resource.{field}")
    safety = _mapping(root.get("fresh_pre_child_safety"), label="source resource safety")
    for field in (
        "observed_immediately_before_source_teacher_lease",
        "exclusive_clean_window",
        "no_active_q30_or_q80_capture_child",
        "no_source_or_native_model_body_resident",
    ):
        _require(safety.get(field), expected=True, label=f"source resource safety.{field}")
    for field in ("swap_used_bytes", "swapouts_pages_delta"):
        if _integer(safety.get(field), label=f"source resource safety.{field}") != 0:
            raise ProductionSourceTeacherOuterError(f"source resource safety.{field} must remain zero")
    reclaimable = _integer(safety.get("reclaimable_bytes"), label="source resource reclaimable")
    floor = _integer(
        safety.get("minimum_reclaimable_bytes_required"),
        label="source resource minimum reclaimable",
        minimum=1,
    )
    if reclaimable < floor:
        raise ProductionSourceTeacherOuterError("source resource reclaimable floor is not met")


def _validate_source_lease(
    document: Document,
    *,
    post_hash_map_bridge: Document,
    dual_bridge: Document,
    runtime_execution_admission: Document,
    source_resource: Document,
) -> str:
    root = document.document
    _schema_status(root, schema=SOURCE_LEASE_SCHEMA, status=SOURCE_LEASE_STATUS, label="source-teacher lease")
    _reject_fixture_identity(root, label="source-teacher lease")
    for field, expected in (
        ("post_hash_map_bridge", post_hash_map_bridge),
        ("dual_attestation_runtime_admission", dual_bridge),
        ("runtime_execution_admission", runtime_execution_admission),
        ("source_teacher_resource_admission", source_resource),
    ):
        _pointer(root.get(field), expected=expected, label=f"source-teacher lease.{field}")
    lifecycle = _mapping(root.get("one_shot_lifecycle"), label="source-teacher lease lifecycle")
    for field in (
        "fresh_for_this_exact_launch",
        "new_capture_root",
        "existing_output_reuse_forbidden",
        "replay_or_relaunch_forbidden",
    ):
        _require(lifecycle.get(field), expected=True, label=f"source-teacher lease lifecycle.{field}")
    _require(
        lifecycle.get("automatic_retry_allowed"),
        expected=False,
        label="source-teacher lease lifecycle.automatic_retry_allowed",
    )
    if lifecycle.get("prior_terminal_receipt") is not None:
        raise ProductionSourceTeacherOuterError("source-teacher lease must not carry a prior terminal")
    _text(lifecycle.get("exact_launch_nonce"), label="source-teacher lease launch nonce", sha256=True)
    safety = _mapping(root.get("fresh_pre_child_safety"), label="source-teacher lease safety")
    for field in (
        "observed_immediately_before_child",
        "exclusive_clean_window",
        "no_source_or_native_model_body_resident_before_child",
    ):
        _require(safety.get(field), expected=True, label=f"source-teacher lease safety.{field}")
    for field in ("swap_used_bytes", "swapouts_pages_delta"):
        if _integer(safety.get(field), label=f"source-teacher lease safety.{field}") != 0:
            raise ProductionSourceTeacherOuterError(f"source-teacher lease safety.{field} must remain zero")
    reclaimable = _integer(safety.get("reclaimable_bytes"), label="source-teacher lease reclaimable")
    floor = _integer(
        safety.get("minimum_reclaimable_bytes_required"),
        label="source-teacher lease minimum reclaimable",
        minimum=1,
    )
    if reclaimable < floor:
        raise ProductionSourceTeacherOuterError("source-teacher lease reclaimable floor is not met")
    return _text(root.get("lease_id"), label="source-teacher lease ID", sha256=True)


def _future_lifecycle() -> dict[str, Any]:
    return {
        "post_hash_map_bridge": {
            "schema": POST_HASH_MAP_BRIDGE_SCHEMA,
            "status": POST_HASH_MAP_BRIDGE_STATUS,
            "non_authorizing": True,
            "cannot_substitute_for_runtime_execution_admission": True,
        },
        "runtime_execution_inputs": {
            "dual_attestation_runtime_admission": {
                "schema": DUAL_BRIDGE_SCHEMA,
                "status": DUAL_BRIDGE_STATUS,
                "must_bind_post_hash_map_bridge": True,
            },
            "runtime_range_admission": {
                "schema": RUNTIME_ADMISSION_SCHEMA,
                "status": RUNTIME_ADMISSION_STATUS,
                "must_bind_earned_flat_map_canonical_identity": True,
                "must_state_hash_scan_is_not_teacher_execution": True,
            },
            "source_teacher_resource": {
                "schema": SOURCE_RESOURCE_SCHEMA,
                "status": SOURCE_RESOURCE_STATUS,
                "fresh_zero_swap_observation_before_lease": True,
            },
            "source_teacher_lease": {
                "schema": SOURCE_LEASE_SCHEMA,
                "status": SOURCE_LEASE_STATUS,
                "one_shot_and_distinct_from_production_hash_scan_lease": True,
            },
        },
        "replay_reservation": {
            "schema": REPLAY_SCHEMA,
            "status_only_after_create_new_reservation": REPLAY_STATUS,
            "one_source_child_maximum": True,
            "automatic_retry_or_relaunch_forbidden": True,
            "capture_root_must_be_new": True,
        },
        "source_child_command": [
            "ascension_qwen30_streamed_source_teacher_child",
            "--source-root",
            "ABSOLUTE_CANONICAL_QWEN30_SOURCE_ROOT",
            "--runtime-admission",
            "ABSOLUTE_SEALED_RUNTIME_EXECUTION_ADMISSION_JSON",
            "--dual-attestation-runtime-admission",
            "ABSOLUTE_SEALED_DUAL_BRIDGE_JSON",
            "--source-lease",
            "ABSOLUTE_SEALED_FRESH_SOURCE_TEACHER_LEASE_JSON",
            "--capture-dir",
            "NEW_ABSOLUTE_SOURCE_CHILD_CAPTURE_DIRECTORY",
        ],
        "source_child_capture": {
            "schema": SOURCE_CHILD_CAPTURE_SCHEMA,
            "status": SOURCE_CHILD_CAPTURE_STATUS,
            "source_layers": SOURCE_LAYERS,
            "source_forwards": SOURCE_FORWARDS,
            "source_vectors_f32le": SOURCE_VECTORS,
            "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
            "maximum_live_raw_bf16_windows": 1,
            "fsync_two_source_vectors_before_child_exit": True,
            "close_handles_and_zero_cache_before_child_exit": True,
            "must_not_start_native_phase": True,
        },
        "source_terminal_then_eviction": {
            "source_terminal": {"schema": SOURCE_TERMINAL_SCHEMA, "status": SOURCE_TERMINAL_STATUS},
            "source_eviction": {"schema": SOURCE_EVICTION_SCHEMA, "status": SOURCE_EVICTION_STATUS},
            "outer_must_reap_source_child_before_terminal": True,
            "terminal_receipt_last_after_source_capture": True,
            "source_eviction_must_close_handles_clear_cache_and_preserve_vectors": True,
        },
        "distinct_native_handoff": {
            "native_lease": {"schema": NATIVE_LEASE_SCHEMA, "status": NATIVE_LEASE_STATUS},
            "source_eviction_seal_must_bind_native_lease": True,
            "native_lease_id_must_differ_from_source_teacher_lease": True,
            "native_action_not_authorized_by_this_preflight": True,
        },
    }


def _execution_boundary() -> dict[str, bool]:
    return {
        "source_root_opened_or_statted": False,
        "source_payload_opened": False,
        "source_teacher_child_spawned": False,
        "source_teacher_or_logits_executed": False,
        "source_or_native_vectors_written": False,
        "source_eviction_performed": False,
        "native_lease_issued_or_consumed": False,
        "native_child_spawned": False,
        "gpu_server_hcli_or_tps_action": False,
        "replay_reservation_created": False,
        "outer_terminal_or_release_written": False,
    }


def _refusal(*, antecedents: dict[str, Any], blockers: Sequence[str]) -> dict[str, Any]:
    return seal(
        {
            "schema": SCHEMA,
            "status": REFUSED_STATUS,
            "prepared": False,
            "spawn_permitted": False,
            "production_hash_scan_antecedents": antecedents,
            "blockers": list(blockers),
            "future_lifecycle": _future_lifecycle(),
            "execution_boundary": _execution_boundary(),
            "claim_boundary": "CPU/file-only refusal. No source root/payload, model/GPU/server/HCLI, lease, child, eviction, native handoff, TPS, TG, or tournament action occurred.",
        }
    )


def build_production_source_teacher_preflight(
    *,
    production_outer_terminal_path: Path | None = None,
    production_child_capture_path: Path | None = None,
    production_flat_map_path: Path | None = None,
    production_hash_coverage_path: Path | None = None,
    production_lease_release_path: Path | None = None,
    post_hash_map_bridge_path: Path | None = None,
    dual_bridge_path: Path | None = None,
    runtime_execution_admission_path: Path | None = None,
    source_teacher_resource_path: Path | None = None,
    source_teacher_lease_path: Path | None = None,
) -> dict[str, Any]:
    """Build a sealed PREPARED/REFUSED document without a launch surface."""
    antecedent_paths = (
        production_outer_terminal_path,
        production_child_capture_path,
        production_flat_map_path,
        production_hash_coverage_path,
        production_lease_release_path,
    )
    blockers: list[str] = []
    antecedents: Antecedents | None = None
    if any(path is None for path in antecedent_paths):
        blockers.append("earned_production_hash_scan_antecedent_chain_absent")
    else:
        assert all(path is not None for path in antecedent_paths)
        try:
            antecedents = _validate_antecedents(
                production_outer_terminal_path=production_outer_terminal_path,
                production_child_capture_path=production_child_capture_path,
                production_flat_map_path=production_flat_map_path,
                production_hash_coverage_path=production_hash_coverage_path,
                production_lease_release_path=production_lease_release_path,
            )
        except ProductionSourceTeacherOuterError as exc:
            blockers.append(f"earned_production_hash_scan_antecedent_chain_invalid:{exc}")

    bridge: Document | None = None
    if post_hash_map_bridge_path is None:
        blockers.append("sealed_non_fixture_post_hash_map_bridge_absent")
    elif antecedents is None:
        blockers.append("post_hash_map_bridge_not_evaluated_without_earned_antecedents")
    else:
        try:
            bridge = _sealed(post_hash_map_bridge_path, label="post-hash-map bridge")
            _validate_post_hash_map_bridge(bridge, antecedents=antecedents)
        except ProductionSourceTeacherOuterError as exc:
            blockers.append(f"sealed_non_fixture_post_hash_map_bridge_invalid:{exc}")

    dual: Document | None = None
    if dual_bridge_path is None:
        blockers.append("sealed_dual_attestation_runtime_admission_bridge_absent")
    elif bridge is None:
        blockers.append("dual_bridge_not_evaluated_without_post_hash_map_bridge")
    else:
        try:
            dual = _sealed(dual_bridge_path, label="dual-attestation/runtime-admission bridge")
            _validate_dual_bridge(dual, post_hash_map_bridge=bridge)
        except ProductionSourceTeacherOuterError as exc:
            blockers.append(f"sealed_dual_bridge_invalid:{exc}")

    runtime: Document | None = None
    if runtime_execution_admission_path is None:
        blockers.append("sealed_non_fixture_runtime_execution_admission_absent")
    elif bridge is None or dual is None or antecedents is None:
        blockers.append("runtime_execution_admission_not_evaluated_without_bridge_and_antecedents")
    else:
        try:
            runtime = _sealed(runtime_execution_admission_path, label="runtime execution admission")
            _validate_runtime_execution_admission(
                runtime,
                post_hash_map_bridge=bridge,
                dual_bridge=dual,
                flat_map=antecedents.production_flat_map,
            )
        except ProductionSourceTeacherOuterError as exc:
            blockers.append(f"sealed_runtime_execution_admission_invalid:{exc}")

    resource: Document | None = None
    if source_teacher_resource_path is None:
        blockers.append("fresh_source_teacher_resource_admission_absent")
    elif bridge is None or dual is None or runtime is None:
        blockers.append("source_teacher_resource_not_evaluated_without_runtime_execution_admission")
    else:
        try:
            resource = _sealed(source_teacher_resource_path, label="source-teacher resource admission")
            _validate_source_resource(
                resource,
                post_hash_map_bridge=bridge,
                dual_bridge=dual,
                runtime_execution_admission=runtime,
            )
        except ProductionSourceTeacherOuterError as exc:
            blockers.append(f"fresh_source_teacher_resource_admission_invalid:{exc}")

    lease: Document | None = None
    source_lease_id: str | None = None
    if source_teacher_lease_path is None:
        blockers.append("fresh_one_shot_source_teacher_lease_absent")
    elif bridge is None or dual is None or runtime is None or resource is None:
        blockers.append("source_teacher_lease_not_evaluated_without_resource_admission")
    else:
        try:
            lease = _sealed(source_teacher_lease_path, label="source-teacher lease")
            source_lease_id = _validate_source_lease(
                lease,
                post_hash_map_bridge=bridge,
                dual_bridge=dual,
                runtime_execution_admission=runtime,
                source_resource=resource,
            )
        except ProductionSourceTeacherOuterError as exc:
            blockers.append(f"fresh_one_shot_source_teacher_lease_invalid:{exc}")

    antecedent_evidence: dict[str, Any] = (
        _antecedent_evidence(antecedents) if antecedents is not None else {"present": False}
    )
    if blockers:
        return _refusal(antecedents=antecedent_evidence, blockers=blockers)
    assert antecedents is not None and bridge is not None and dual is not None
    assert runtime is not None and resource is not None and lease is not None and source_lease_id is not None
    return seal(
        {
            "schema": SCHEMA,
            "status": PREPARED_STATUS,
            "prepared": True,
            "spawn_permitted": False,
            "production_hash_scan_antecedents": _antecedent_evidence(antecedents),
            "post_hash_map_bridge": _evidence(bridge),
            "dual_attestation_runtime_admission": _evidence(dual),
            "runtime_execution_admission": _evidence(runtime),
            "source_teacher_resource_admission": _evidence(resource),
            "source_teacher_lease": _evidence(lease),
            "source_teacher_lease_id": source_lease_id,
            "future_lifecycle": _future_lifecycle(),
            "reservation": {
                "this_preflight_created_no_replay_or_capture_root": True,
                "this_preflight_issued_or_consumed_no_lease": True,
                "this_preflight_spawned_no_source_or_native_child": True,
                "future_one_source_child_then_close_cache_zero_evict_then_distinct_native_lease": True,
            },
            "execution_boundary": _execution_boundary(),
            "claim_boundary": "Prepared CPU/file-only source-teacher lifecycle grammar. The earned hash scan is immutable antecedent evidence only; a later separately authorized outer still must create a replay guard, run one source child, reap it, write receipt-last terminal/eviction evidence, and obtain a distinct native lease.",
        }
    )


def validate_fake_future_lifecycle(
    *,
    replay_reservation: Mapping[str, Any],
    source_child_capture: Mapping[str, Any],
    source_terminal: Mapping[str, Any],
    source_eviction: Mapping[str, Any],
    native_lease: Mapping[str, Any],
    outer_preflight_seal_sha256: str,
    post_hash_map_bridge_seal_sha256: str,
    dual_bridge_seal_sha256: str,
    runtime_execution_admission_seal_sha256: str,
    source_teacher_resource_seal_sha256: str,
    source_teacher_lease_id: str,
) -> None:
    """Validate the future source→evict→native handoff from fake mappings only."""
    _schema_status(replay_reservation, schema=REPLAY_SCHEMA, status=REPLAY_STATUS, label="replay reservation")
    for field in (
        "create_new_before_source_child",
        "one_source_child_maximum",
        "automatic_retry_or_relaunch_forbidden",
    ):
        _require(replay_reservation.get(field), expected=True, label=f"replay reservation.{field}")
    if _integer(replay_reservation.get("attempt"), label="replay reservation attempt", minimum=1) != 1:
        raise ProductionSourceTeacherOuterError("replay reservation attempt must be exactly one")
    for field, expected in (
        ("outer_preflight_seal_sha256", outer_preflight_seal_sha256),
        ("source_teacher_lease_id", source_teacher_lease_id),
    ):
        if _text(replay_reservation.get(field), label=f"replay reservation.{field}", sha256=True) != expected:
            raise ProductionSourceTeacherOuterError(f"replay reservation.{field} drifted")

    _schema_status(
        source_child_capture,
        schema=SOURCE_CHILD_CAPTURE_SCHEMA,
        status=SOURCE_CHILD_CAPTURE_STATUS,
        label="source child capture",
    )
    for field in (
        "source_teacher_execution_completed",
        "two_source_f32le_vectors_fsynced_before_child_exit",
        "source_handles_closed_before_child_exit",
        "reader_cache_zeroed_before_child_exit",
        "receipt_written_last",
    ):
        _require(source_child_capture.get(field), expected=True, label=f"source child capture.{field}")
    for field in ("native_phase_started", "gpu_server_hcli_or_tps_action"):
        _require(source_child_capture.get(field), expected=False, label=f"source child capture.{field}")
    geometry = _mapping(source_child_capture.get("geometry"), label="source child capture geometry")
    for field, expected in (
        ("source_layers", SOURCE_LAYERS),
        ("source_forwards", SOURCE_FORWARDS),
        ("source_vectors_f32le", SOURCE_VECTORS),
        ("maximum_positioned_read_bytes", MAX_POSITIONED_READ_BYTES),
        ("maximum_live_raw_bf16_windows", 1),
    ):
        if _integer(geometry.get(field), label=f"source child capture geometry.{field}") != expected:
            raise ProductionSourceTeacherOuterError(f"source child capture geometry.{field} drifted")
    for field, expected in (
        ("post_hash_map_bridge_seal_sha256", post_hash_map_bridge_seal_sha256),
        ("dual_bridge_seal_sha256", dual_bridge_seal_sha256),
        ("runtime_execution_admission_seal_sha256", runtime_execution_admission_seal_sha256),
        ("source_teacher_resource_seal_sha256", source_teacher_resource_seal_sha256),
        ("source_teacher_lease_id", source_teacher_lease_id),
    ):
        if _text(source_child_capture.get(field), label=f"source child capture.{field}", sha256=True) != expected:
            raise ProductionSourceTeacherOuterError(f"source child capture.{field} drifted")

    _schema_status(source_terminal, schema=SOURCE_TERMINAL_SCHEMA, status=SOURCE_TERMINAL_STATUS, label="source terminal")
    for field in (
        "outer_reaped_source_child_before_terminal_receipt",
        "terminal_receipt_written_after_child_capture",
        "terminal_receipt_written_last",
    ):
        _require(source_terminal.get(field), expected=True, label=f"source terminal.{field}")
    _require(source_terminal.get("native_phase_started"), expected=False, label="source terminal.native_phase_started")
    if _text(source_terminal.get("source_teacher_lease_id"), label="source terminal lease ID", sha256=True) != source_teacher_lease_id:
        raise ProductionSourceTeacherOuterError("source terminal lease binding drifted")
    child_seal = _text(source_child_capture.get("seal_sha256"), label="source child capture seal", sha256=True)
    if _text(source_terminal.get("source_child_capture_seal_sha256"), label="source terminal child seal", sha256=True) != child_seal:
        raise ProductionSourceTeacherOuterError("source terminal child capture binding drifted")

    _schema_status(source_eviction, schema=SOURCE_EVICTION_SCHEMA, status=SOURCE_EVICTION_STATUS, label="source eviction")
    for field in (
        "source_weights_evicted",
        "source_backend_shutdown",
        "source_model_residency_released",
        "streamed_reader_cache_cleared",
        "source_payloads_durable_and_immutable",
        "swap_remained_zero",
        "pre_native_lease_process_tree_checked",
    ):
        _require(source_eviction.get(field), expected=True, label=f"source eviction.{field}")
    _require(source_eviction.get("native_phase_started"), expected=False, label="source eviction.native_phase_started")
    terminal_seal = _text(source_terminal.get("seal_sha256"), label="source terminal seal", sha256=True)
    if _text(source_eviction.get("source_terminal_seal_sha256"), label="source eviction terminal seal", sha256=True) != terminal_seal:
        raise ProductionSourceTeacherOuterError("source eviction terminal binding drifted")

    _schema_status(native_lease, schema=NATIVE_LEASE_SCHEMA, status=NATIVE_LEASE_STATUS, label="native lease")
    for field in (
        "fresh_for_this_exact_native_launch",
        "new_capture_root",
        "replay_or_relaunch_forbidden",
        "source_eviction_verified_before_native_lease",
    ):
        _require(native_lease.get(field), expected=True, label=f"native lease.{field}")
    native_lease_id = _text(native_lease.get("lease_id"), label="native lease ID", sha256=True)
    if native_lease_id == source_teacher_lease_id:
        raise ProductionSourceTeacherOuterError("native lease must be distinct from source-teacher lease")
    eviction_seal = _text(source_eviction.get("seal_sha256"), label="source eviction seal", sha256=True)
    if _text(native_lease.get("source_eviction_seal_sha256"), label="native lease eviction seal", sha256=True) != eviction_seal:
        raise ProductionSourceTeacherOuterError("native lease eviction binding drifted")


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise ProductionSourceTeacherOuterError("--out must be a new absolute JSON path below an existing directory")
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
    parser.add_argument("--production-outer-terminal", type=Path)
    parser.add_argument("--production-child-capture", type=Path)
    parser.add_argument("--production-flat-map", type=Path)
    parser.add_argument("--production-hash-coverage", type=Path)
    parser.add_argument("--production-lease-release", type=Path)
    parser.add_argument("--post-hash-map-bridge", type=Path)
    parser.add_argument("--dual-attestation-runtime-admission", type=Path)
    parser.add_argument("--runtime-execution-admission", type=Path)
    parser.add_argument("--source-teacher-resource", type=Path)
    parser.add_argument("--source-teacher-lease", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_production_source_teacher_preflight(
            production_outer_terminal_path=args.production_outer_terminal,
            production_child_capture_path=args.production_child_capture,
            production_flat_map_path=args.production_flat_map,
            production_hash_coverage_path=args.production_hash_coverage,
            production_lease_release_path=args.production_lease_release,
            post_hash_map_bridge_path=args.post_hash_map_bridge,
            dual_bridge_path=args.dual_attestation_runtime_admission,
            runtime_execution_admission_path=args.runtime_execution_admission,
            source_teacher_resource_path=args.source_teacher_resource,
            source_teacher_lease_path=args.source_teacher_lease,
        )
        _write_new(args.out, result)
    except ProductionSourceTeacherOuterError as exc:
        print(f"Q30 production source-teacher outer preflight could not write a sealed result: {exc}")
        return 2
    print(json.dumps({"output": str(args.out.resolve()), "status": result["status"], "seal_sha256": result["seal_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
