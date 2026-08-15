"""Receipt-last, CPU-only preflight for a future Q30 streamed source teacher.

This is intentionally not a process runner.  It does not import ``subprocess``
and cannot receive a source root, model command, accelerator setting, server,
HCLI request, or lease issuer.  Its only file inputs are sealed metadata
receipts.  The output is either a sealed refusal before any spawn surface or a
sealed *non-authorizing* reservation grammar for a separately reviewed future
outer runner.

The future lifecycle is rigid:

    one source child -> two durable source F32LE vectors -> reap/close/cache
    clear -> source eviction/release -> distinct native lease -> one native
    four-vector child -> reap -> native outer terminal/release.

The implementation deliberately reuses the existing guarded Q30 outer
validator for future receipt validation.  It does not replace that validator,
weaken its source-then-evict ordering, or launch either child today.
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

from lab.operators import (
    ascension_qwen30_guarded_streamed_source_oracle_outer_controller as guarded_outer,
)
from lab.receipts import SealIntegrityError, seal, verify

SCHEMA = "hawking.ascension.qwen30_streamed_source_teacher_outer_runner_preflight.v1"
PREPARED_STATUS = (
    "PREPARED_QWEN30_STREAMED_SOURCE_TEACHER_OUTER_RECEIPT_LAST_RESERVATION_NOT_SPAWNED"
)
REFUSED_STATUS = (
    "REFUSED_QWEN30_STREAMED_SOURCE_TEACHER_OUTER_PREREQUISITES_ABSENT_OR_INVALID"
)

CHILD_PREFLIGHT_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_teacher_child_preflight.v1"
)
CHILD_PREFLIGHT_STATUS = (
    "PREPARED_QWEN30_STREAMED_SOURCE_TEACHER_CHILD_INTERFACE_NOT_EXECUTED"
)
DUAL_BRIDGE_SCHEMA = "hawking.ascension.qwen30_streamed_source_teacher_dual_attestation_runtime_admission.v1"
DUAL_BRIDGE_STATUS = "PREPARED_QWEN30_STREAMED_SOURCE_TEACHER_DUAL_ATTESTATION_RUNTIME_ADMISSION_NOT_EXECUTED"
RUNTIME_RANGE_MAP_SCHEMA = "hawking.ascension.qwen30_source_bf16_range_map.v1"
RUNTIME_ADMISSION_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_teacher_runtime_range_admission.v1"
)
RUNTIME_ADMISSION_STATUS = (
    "EARNED_QWEN30_STREAMED_SOURCE_TEACHER_RUNTIME_RANGE_ADMISSION_NO_MODEL_RESIDENCY"
)
OPERATOR_ATTESTATION_SCHEMA = "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_execution_attestation.v1"
OPERATOR_ATTESTATION_STATUS = (
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_ATTESTED"
)
RANGE_READER_ATTESTATION_SCHEMA = (
    "hawking.ascension.qwen30_layer_streamed_source_bf16_exact_semantics_attestation.v1"
)
RANGE_READER_ATTESTATION_STATUS = (
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_BF16_EXACT_SEMANTICS_ATTESTED"
)
FEASIBILITY_SCHEMA = "hawking.ascension.qwen30_layer_streamed_source_bf16_final_logit_oracle_feasibility.v1"
FEASIBILITY_PREPARED_STATUS = (
    "PREPARED_QWEN30_LAYER_STREAMED_SOURCE_BF16_ORACLE_FEASIBILITY_NOT_EXECUTED"
)

SOURCE_LEASE_SCHEMA = guarded_outer.SOURCE_LEASE_SCHEMA
SOURCE_LEASE_STATUS = guarded_outer.SOURCE_LEASE_STATUS
SOURCE_TERMINAL_SCHEMA = guarded_outer.SOURCE_TERMINAL_SCHEMA
SOURCE_TERMINAL_STATUS = guarded_outer.SOURCE_TERMINAL_STATUS
SOURCE_EVICTION_SCHEMA = guarded_outer.SOURCE_EVICTION_SCHEMA
SOURCE_EVICTION_STATUS = guarded_outer.SOURCE_EVICTION_STATUS
NATIVE_LEASE_SCHEMA = guarded_outer.NATIVE_LEASE_SCHEMA
NATIVE_LEASE_STATUS = guarded_outer.NATIVE_LEASE_STATUS
NATIVE_CHILD_TERMINAL_SCHEMA = (
    "hawking.ascension.qwen30_hq30gr2_raw_final_logit_retention_capture.v1"
)
NATIVE_CHILD_TERMINAL_STATUS = (
    "EARNED_NEW_DIAGNOSTIC_RAW_FINAL_LOGITS_RETAINED_NOT_THREE_WAY_ORACLE"
)
NATIVE_OUTER_TERMINAL_SCHEMA = "hawking.ascension.qwen30_streamed_source_teacher_native_four_vector_outer_terminal.v1"
NATIVE_OUTER_TERMINAL_STATUS = (
    "CAPTURED_QWEN30_STREAMED_SOURCE_TEACHER_NATIVE_FOUR_VECTOR_OUTER_TERMINAL"
)
NATIVE_RELEASE_SCHEMA = "hawking.ascension.qwen30_streamed_source_teacher_native_four_vector_lease_release.v1"
NATIVE_RELEASE_STATUS = "RELEASED_QWEN30_STREAMED_SOURCE_TEACHER_NATIVE_FOUR_VECTOR_LEASE_AFTER_OUTER_TERMINAL"
REPLAY_RESERVATION_SCHEMA = (
    "hawking.ascension.qwen30_streamed_source_teacher_outer_replay_reservation.v1"
)
REPLAY_RESERVATION_STATUS = (
    "RESERVED_QWEN30_STREAMED_SOURCE_TEACHER_ONE_SHOT_CAPTURE_NOT_SPAWNED"
)

MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_POSITIONED_READ_BYTES = 1024 * 1024
SOURCE_LAYERS = 48
SOURCE_FORWARDS = 370
PREFIX_TOKENS = 369
FORCED_TOKEN = 949
VOCAB_ROWS = 151_936
F32_VECTOR_BYTES = VOCAB_ROWS * 4
SOURCE_PAYLOADS = (
    "source_bf16_exact_prefix_logits.f32le",
    "source_bf16_forced_shared_continuation_logits.f32le",
)
NATIVE_PAYLOADS = (
    "scalar_control_exact_prefix_logits.f32le",
    "scalar_control_forced_shared_continuation_logits.f32le",
    "hq30gr2_candidate_exact_prefix_logits.f32le",
    "hq30gr2_candidate_forced_shared_continuation_logits.f32le",
)


class SourceTeacherOuterPreflightError(RuntimeError):
    """A receipt cannot safely reserve or validate the future lifecycle."""


@dataclass(frozen=True)
class Document:
    path: Path
    document: dict[str, Any]
    sha256: str
    seal_sha256: str


@dataclass(frozen=True)
class ChildPreflight:
    document: Document
    trace: dict[str, Any]
    maximum_window_bytes: int
    bridge_pointer: dict[str, Any]
    feasibility_pointer: dict[str, Any]
    raw_six_vector_pointer: dict[str, Any]
    current_trace_pointer: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _regular_json(path: Path, *, label: str) -> Path:
    if not path.is_absolute() or path.suffix != ".json":
        raise SourceTeacherOuterPreflightError(
            f"{label} must be an absolute .json metadata receipt"
        )
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise SourceTeacherOuterPreflightError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise SourceTeacherOuterPreflightError(
            f"{label} must be a regular non-symlink file"
        )
    if metadata.st_size <= 0 or metadata.st_size > MAX_METADATA_BYTES:
        raise SourceTeacherOuterPreflightError(
            f"{label} must contain 1..={MAX_METADATA_BYTES} metadata bytes"
        )
    return path.resolve(strict=True)


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourceTeacherOuterPreflightError(f"{label} must be an object")
    return dict(value)


def _sequence(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SourceTeacherOuterPreflightError(f"{label} must be an array")
    return list(value)


def _text(value: object, *, label: str, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise SourceTeacherOuterPreflightError(f"{label} must be a non-empty string")
    if sha256 and (
        len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
    ):
        raise SourceTeacherOuterPreflightError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SourceTeacherOuterPreflightError(
            f"{label} must be an integer >= {minimum}"
        )
    return value


def _require_true(value: object, *, label: str) -> None:
    if value is not True:
        raise SourceTeacherOuterPreflightError(f"{label} must be true")


def _schema_status(
    document: Mapping[str, Any], *, schema: str, status: str, label: str
) -> None:
    if document.get("schema") != schema or document.get("status") != status:
        raise SourceTeacherOuterPreflightError(f"{label} schema/status drifted")


def _sealed(path: Path, *, label: str) -> Document:
    clean = _regular_json(path, label=label)
    try:
        raw = json.loads(clean.read_text(encoding="utf-8"))
        checked = verify(raw, label=label)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        SealIntegrityError,
    ) as exc:
        raise SourceTeacherOuterPreflightError(
            f"{label} is absent or invalid: {exc}"
        ) from exc
    if not isinstance(checked, Mapping):
        raise SourceTeacherOuterPreflightError(f"{label} must be a sealed object")
    document = dict(checked)
    return Document(
        path=clean,
        document=document,
        sha256=_sha256_file(clean),
        seal_sha256=_text(
            document.get("seal_sha256"), label=f"{label} seal", sha256=True
        ),
    )


def _metadata(path: Path, *, label: str) -> tuple[dict[str, Any], Path, str]:
    clean = _regular_json(path, label=label)
    try:
        value = json.loads(clean.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceTeacherOuterPreflightError(
            f"{label} is absent or invalid: {exc}"
        ) from exc
    return _mapping(value, label=label), clean, _sha256_file(clean)


def _evidence(document: Document) -> dict[str, Any]:
    return {
        "path": str(document.path),
        "document_sha256": document.sha256,
        "seal_sha256": document.seal_sha256,
    }


def _validate_evidence_pointer(
    pointer: Mapping[str, Any], document: Document, *, label: str
) -> None:
    if (
        Path(_text(pointer.get("path"), label=f"{label}.path")).resolve()
        != document.path
    ):
        raise SourceTeacherOuterPreflightError(f"{label} path drifted")
    if (
        _text(
            pointer.get("raw_document_sha256", pointer.get("document_sha256")),
            label=f"{label} document SHA",
            sha256=True,
        )
        != document.sha256
    ):
        raise SourceTeacherOuterPreflightError(f"{label} document SHA drifted")
    if (
        _text(pointer.get("seal_sha256"), label=f"{label} seal", sha256=True)
        != document.seal_sha256
    ):
        raise SourceTeacherOuterPreflightError(f"{label} seal drifted")


def _trace(value: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if (
        _integer(value.get("source_template_token_count"), label=f"{label} prefix")
        != PREFIX_TOKENS
    ):
        raise SourceTeacherOuterPreflightError(f"{label} prefix drifted")
    if (
        _integer(
            value.get("forced_identical_continuation_token_id"),
            label=f"{label} forced token",
        )
        != FORCED_TOKEN
    ):
        raise SourceTeacherOuterPreflightError(f"{label} forced token drifted")
    return {
        "probe_id": _text(
            value.get("probe_id", "literal_hawking"), label=f"{label} probe"
        ),
        "source_template_token_count": PREFIX_TOKENS,
        "forced_identical_continuation_token_id": FORCED_TOKEN,
        "source_template_token_ids_u32le_sha256": _text(
            value.get("source_template_token_ids_u32le_sha256"),
            label=f"{label} token SHA",
            sha256=True,
        ),
    }


def _validate_child_preflight(document: Document) -> ChildPreflight:
    root = document.document
    _schema_status(
        root,
        schema=CHILD_PREFLIGHT_SCHEMA,
        status=CHILD_PREFLIGHT_STATUS,
        label="source child preflight",
    )
    if root.get("execution_authorized") is not False:
        raise SourceTeacherOuterPreflightError(
            "source child preflight must never authorize execution"
        )
    boundary = _mapping(
        root.get("execution_boundary"),
        label="source child preflight execution boundary",
    )
    for field in (
        "source_tensor_payload_opened",
        "source_model_loaded_or_instantiated",
        "whole_source_model_resident",
        "gpu_metal_mps_or_other_accelerator_invoked",
        "server_started_or_contacted",
        "hcli_invoked",
        "lease_requested_issued_or_consumed",
        "child_process_started",
        "source_teacher_or_native_vector_written",
        "source_eviction_or_native_phase_performed",
    ):
        if boundary.get(field) is not False:
            raise SourceTeacherOuterPreflightError(
                f"source child preflight boundary {field} must remain false"
            )
    bindings = _mapping(
        root.get("input_bindings"), label="source child preflight input bindings"
    )
    range_authority = _mapping(
        bindings.get("metadata_range_authority"), label="child range authority binding"
    )
    maximum_window = _integer(
        range_authority.get("maximum_declared_bf16_row_window_bytes"),
        label="child range-authority maximum window",
        minimum=1,
    )
    if maximum_window > MAX_POSITIONED_READ_BYTES:
        raise SourceTeacherOuterPreflightError(
            "child range authority exceeds the one-MiB positioned-read ceiling"
        )
    trace = _trace(
        _mapping(root.get("trace_binding"), label="source child preflight trace"),
        label="child trace",
    )
    interface = _mapping(
        root.get("future_child_interface"),
        label="source child preflight future interface",
    )
    execution_shape = _mapping(
        interface.get("execution_shape"), label="source child execution shape"
    )
    expected_shape = {
        "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
        "maximum_live_raw_bf16_windows": 1,
        "source_layers": SOURCE_LAYERS,
        "source_forwards": SOURCE_FORWARDS,
        "prefix_tokens": PREFIX_TOKENS,
        "forced_token_id": FORCED_TOKEN,
    }
    for field, expected in expected_shape.items():
        if (
            _integer(
                execution_shape.get(field),
                label=f"source child execution shape {field}",
            )
            != expected
        ):
            raise SourceTeacherOuterPreflightError(
                f"source child execution shape {field} drifted"
            )
    resolution = _mapping(
        root.get("required_dual_schema_resolution"),
        label="source child schema resolution",
    )
    if resolution.get("future_runtime_range_map_schema") != RUNTIME_RANGE_MAP_SCHEMA:
        raise SourceTeacherOuterPreflightError(
            "source child runtime range-map schema drifted"
        )
    for field, expected_schema, expected_status in (
        (
            "future_operator_accumulation_attestation",
            OPERATOR_ATTESTATION_SCHEMA,
            OPERATOR_ATTESTATION_STATUS,
        ),
        (
            "future_range_reader_exact_semantics_attestation",
            RANGE_READER_ATTESTATION_SCHEMA,
            RANGE_READER_ATTESTATION_STATUS,
        ),
    ):
        _schema_status(
            _mapping(resolution.get(field), label=f"source child {field}"),
            schema=expected_schema,
            status=expected_status,
            label=f"source child {field}",
        )
    for field in (
        "metadata_range_authority_is_not_the_flat_runtime_range_map",
        "both_execution_attestations_must_bind_the_same_runtime_admission_and_source_payloads",
        "a_prepared_bridge_is_non_authorizing_and_cannot_substitute_for_either_execution_attestation",
    ):
        _require_true(
            resolution.get(field), label=f"source child schema resolution {field}"
        )
    return ChildPreflight(
        document=document,
        trace=trace,
        maximum_window_bytes=maximum_window,
        bridge_pointer=_mapping(
            bindings.get("dual_attestation_runtime_admission_bridge"),
            label="child dual bridge pointer",
        ),
        feasibility_pointer=_mapping(
            bindings.get("streamed_feasibility"), label="child feasibility pointer"
        ),
        raw_six_vector_pointer=_mapping(
            bindings.get("raw_six_vector_contract"),
            label="child raw-six-vector pointer",
        ),
        current_trace_pointer=_mapping(
            bindings.get("current_trace"), label="child current-trace pointer"
        ),
    )


def _validate_bridge(document: Document, *, child: ChildPreflight) -> None:
    _schema_status(
        document.document,
        schema=DUAL_BRIDGE_SCHEMA,
        status=DUAL_BRIDGE_STATUS,
        label="dual bridge",
    )
    if child.bridge_pointer.get("present") is not True:
        raise SourceTeacherOuterPreflightError(
            "child preflight was not bound to a dual bridge"
        )
    pointer = _mapping(
        child.bridge_pointer.get("evidence"), label="child dual bridge evidence"
    )
    _validate_evidence_pointer(pointer, document, label="child dual bridge evidence")
    resolution = _mapping(
        document.document.get("schema_resolution"),
        label="dual bridge schema resolution",
    )
    if resolution.get("runtime_range_map_schema") != RUNTIME_RANGE_MAP_SCHEMA:
        raise SourceTeacherOuterPreflightError(
            "dual bridge runtime range-map schema drifted"
        )
    if resolution.get("runtime_admission_schema") != RUNTIME_ADMISSION_SCHEMA:
        raise SourceTeacherOuterPreflightError(
            "dual bridge runtime-admission schema drifted"
        )
    if (
        resolution.get("runtime_admission_status_only_after_bounded_source_validation")
        != RUNTIME_ADMISSION_STATUS
    ):
        raise SourceTeacherOuterPreflightError(
            "dual bridge runtime-admission status drifted"
        )
    for field, expected_schema, expected_status in (
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
            _mapping(resolution.get(field), label=f"dual bridge {field}"),
            schema=expected_schema,
            status=expected_status,
            label=f"dual bridge {field}",
        )
    for field in (
        "both_execution_attestations_required_after_source_child",
        "runtime_range_admission_required_before_payload_open",
        "bridge_does_not_authorize_execution",
    ):
        _require_true(resolution.get(field), label=f"dual bridge {field}")
    worker = _mapping(
        document.document.get("future_source_worker"), label="dual bridge future worker"
    )
    for field, expected in (
        ("maximum_positioned_read_bytes", MAX_POSITIONED_READ_BYTES),
        ("source_layers", SOURCE_LAYERS),
        ("source_forwards", SOURCE_FORWARDS),
        ("source_f32le_vectors", 2),
        ("native_f32le_vectors", 4),
    ):
        if (
            _integer(worker.get(field), label=f"dual bridge future worker {field}")
            != expected
        ):
            raise SourceTeacherOuterPreflightError(
                f"dual bridge future worker {field} drifted"
            )
    for field in (
        "one_bounded_window_only",
        "source_payloads_durable_before_eviction",
        "close_handles_and_clear_cache_before_eviction_receipt",
        "separate_native_four_vector_phase_required",
    ):
        _require_true(worker.get(field), label=f"dual bridge future worker {field}")


def _validate_child_feasibility(child: ChildPreflight) -> None:
    """Require the child preflight's already-validated feasibility summary.

    The Rust child contract deliberately remains PREPARED while listing a
    refused feasibility receipt as a current blocker.  The outer must not
    mistake that syntax-only PREPARED state for earned exact source semantics.
    It therefore consumes the child's compact, sealed feasibility result only
    when every safe/semantic field is explicitly true.
    """
    pointer = child.feasibility_pointer
    if pointer.get("present") is not True:
        raise SourceTeacherOuterPreflightError(
            "child preflight lacks sealed streamed feasibility and earned exact semantics"
        )
    if pointer.get("status") != FEASIBILITY_PREPARED_STATUS:
        raise SourceTeacherOuterPreflightError(
            "child feasibility is refused or not prepared"
        )
    for field in (
        "semantic_equivalence_proven",
        "streamed_memory_arithmetic_fits",
        "zero_swap_condition_met",
    ):
        _require_true(pointer.get(field), label=f"child feasibility {field}")
    evidence = _mapping(pointer.get("evidence"), label="child feasibility evidence")
    feasibility = _sealed(
        Path(_text(evidence.get("path"), label="child feasibility evidence path")),
        label="child streamed feasibility",
    )
    _validate_evidence_pointer(
        evidence, feasibility, label="child feasibility evidence"
    )
    _schema_status(
        feasibility.document,
        schema=FEASIBILITY_SCHEMA,
        status=FEASIBILITY_PREPARED_STATUS,
        label="child streamed feasibility",
    )
    exact_trace = _mapping(
        feasibility.document.get("exact_trace"),
        label="child streamed feasibility exact trace",
    )
    if (
        _integer(
            exact_trace.get("prefix_token_count"),
            label="child streamed feasibility prefix",
        )
        != PREFIX_TOKENS
    ):
        raise SourceTeacherOuterPreflightError(
            "child streamed feasibility prefix drifted"
        )
    if (
        _integer(
            exact_trace.get("forced_token_id"),
            label="child streamed feasibility forced token",
        )
        != FORCED_TOKEN
    ):
        raise SourceTeacherOuterPreflightError(
            "child streamed feasibility forced token drifted"
        )
    if (
        _text(
            exact_trace.get("source_template_token_ids_u32le_sha256"),
            label="child streamed feasibility token SHA",
            sha256=True,
        )
        != child.trace["source_template_token_ids_u32le_sha256"]
    ):
        raise SourceTeacherOuterPreflightError(
            "child streamed feasibility trace drifted"
        )
    assessment = _mapping(
        feasibility.document.get("memory_assessment"),
        label="child streamed feasibility memory assessment",
    )
    feasibility_facts = _mapping(
        feasibility.document.get("feasibility"),
        label="child streamed feasibility facts",
    )
    for field, container in (
        ("streamed_memory_arithmetic_fits", assessment),
        ("zero_swap_condition_met", assessment),
        (
            "semantic_equivalence_proven_by_external_sealed_attestation",
            feasibility_facts,
        ),
        ("safe_streamed_plan_prepared_not_executed", feasibility_facts),
    ):
        _require_true(container.get(field), label=f"child streamed feasibility {field}")
    if feasibility_facts.get("oracle_execution_authorized") is not False:
        raise SourceTeacherOuterPreflightError(
            "child streamed feasibility must not authorize execution"
        )


def _validate_source_lease(
    document: Document, *, child: ChildPreflight
) -> dict[str, int]:
    _schema_status(
        document.document,
        schema=SOURCE_LEASE_SCHEMA,
        status=SOURCE_LEASE_STATUS,
        label="source lease",
    )
    safety = _mapping(
        document.document.get("fresh_pre_child_safety"), label="source lease safety"
    )
    floor = _integer(
        safety.get("minimum_reclaimable_bytes_required"),
        label="source lease reclaimable floor",
        minimum=1,
    )
    try:
        # Reuse the guarded controller's replay/relaunch prohibition as well
        # as its zero-swap safety check.  A syntactically valid safety snapshot
        # alone is not a reservation for a fresh source child.
        guarded_outer._one_shot_lifecycle(document.document, label="source lease")
        return guarded_outer.validate_fresh_zero_swap_safety(
            lease=document.document,
            minimum_reclaimable_bytes=floor,
            label="source lease",
        )
    except guarded_outer.GuardedStreamedSourceOuterError as exc:
        raise SourceTeacherOuterPreflightError(
            f"source lease safety is invalid: {exc}"
        ) from exc


def _future_interface() -> dict[str, Any]:
    return {
        "replay_reservation": {
            "schema": REPLAY_RESERVATION_SCHEMA,
            "status_only_after_create_new_reservation": REPLAY_RESERVATION_STATUS,
            "create_new_before_spawn": True,
            "one_source_child_process_group": True,
            "one_native_child_process_group": True,
            "existing_capture_root_without_matching_terminal_must_refuse": True,
            "matching_terminal_must_be_returned_without_respawn": True,
            "automatic_retry_or_relaunch_forbidden": True,
        },
        "source_child_command": [
            "ascension_qwen30_streamed_source_teacher_child",
            "--source-root",
            "ABSOLUTE_CANONICAL_QWEN30_SOURCE_ROOT",
            "--runtime-admission",
            "ABSOLUTE_SEALED_RUNTIME_ADMISSION_JSON",
            "--dual-attestation-runtime-admission",
            "ABSOLUTE_SEALED_DUAL_BRIDGE_JSON",
            "--source-lease",
            "ABSOLUTE_SEALED_ONE_SHOT_SOURCE_LEASE_JSON",
            "--capture-dir",
            "NEW_ABSOLUTE_SOURCE_CHILD_CAPTURE_DIRECTORY",
        ],
        "source_child_receipt": {
            "schema": "hawking.ascension.qwen30_streamed_source_teacher_child_execution_evidence.v1",
            "status_only_after_real_source_execution": "CAPTURED_QWEN30_STREAMED_SOURCE_TEACHER_CHILD_TWO_F32LE_LOGITS_NOT_NATIVE_PHASE",
            "source_payloads": list(SOURCE_PAYLOADS),
            "payload_dtype": "f32le",
            "payload_vocab_rows": VOCAB_ROWS,
            "payload_bytes_each": F32_VECTOR_BYTES,
            "must_fsync_two_payloads_before_child_exit": True,
            "must_close_all_source_handles_and_zero_reader_cache_before_child_exit": True,
            "must_emit_runtime_admission_and_both_execution_attestation_identities": True,
            "must_not_write_source_terminal_or_start_native_phase": True,
        },
        "outer_source_handoff": {
            "outer_must_reap_source_child_before_terminal": True,
            "source_terminal": {
                "schema": SOURCE_TERMINAL_SCHEMA,
                "status": SOURCE_TERMINAL_STATUS,
            },
            "required_source_terminal_fields": [
                "source_lease.seal_sha256",
                "exact_trace",
                "streamed_execution.mode=layer_streamed_bf16_source_teacher",
                "streamed_execution.outer_reaped_child_before_terminal_receipt=true",
                "streamed_execution.receipt_written_after_payload_fsyncs=true",
                "source_payloads",
                "bounded_per_read_cache",
                "source_payload_read_accounting",
                "dual_execution_attestations.runtime_range_admission",
                "dual_execution_attestations.operator_accumulation",
                "dual_execution_attestations.range_reader_exact_semantics",
            ],
            "source_eviction_release": {
                "schema": SOURCE_EVICTION_SCHEMA,
                "status": SOURCE_EVICTION_STATUS,
            },
            "required_eviction_fields": [
                "source_teacher_terminal.seal_sha256",
                "eviction.source_weights_evicted=true",
                "eviction.source_backend_shutdown=true",
                "eviction.source_model_residency_released=true",
                "eviction.streamed_reader_cache_cleared=true",
                "eviction.source_payloads_durable_and_immutable=true",
                "eviction.swap_remained_zero=true",
                "eviction.pre_native_lease_process_tree_checked=true",
            ],
        },
        "separate_native_four_vector_child": {
            "existing_binary": "ascension_qwen30_quality_repack_all_layer_current_trace_diagnostic",
            "mode": "metal-diagnostic-retain-raw-final-logits",
            "requires_distinct_native_lease": {
                "schema": NATIVE_LEASE_SCHEMA,
                "status": NATIVE_LEASE_STATUS,
            },
            "must_bind_source_eviction_seal_before_native_spawn": True,
            "native_child_terminal": {
                "schema": NATIVE_CHILD_TERMINAL_SCHEMA,
                "status": NATIVE_CHILD_TERMINAL_STATUS,
                "payloads": list(NATIVE_PAYLOADS),
            },
            "native_outer_terminal": {
                "schema": NATIVE_OUTER_TERMINAL_SCHEMA,
                "status": NATIVE_OUTER_TERMINAL_STATUS,
            },
            "native_release": {
                "schema": NATIVE_RELEASE_SCHEMA,
                "status": NATIVE_RELEASE_STATUS,
            },
            "outer_must_reap_native_child_before_native_outer_terminal": True,
            "native_release_must_follow_native_outer_terminal": True,
        },
        "six_vector_terminal": {
            "two_source_plus_four_native_payloads": 6,
            "all_six_payloads_fsynced_before_terminal": True,
            "metric_scoring_is_outside_this_outer_runner": True,
        },
    }


def _future_execution_attestations(
    source_terminal: Mapping[str, Any], *, bridge: Document
) -> None:
    dual = _mapping(
        source_terminal.get("dual_execution_attestations"),
        label="source terminal dual execution attestations",
    )
    bridge_ref = _mapping(
        dual.get("dual_bridge"), label="source terminal dual bridge reference"
    )
    if (
        _text(
            bridge_ref.get("seal_sha256"),
            label="source terminal dual bridge seal",
            sha256=True,
        )
        != bridge.seal_sha256
    ):
        raise SourceTeacherOuterPreflightError(
            "source terminal dual bridge seal drifted"
        )
    expected = (
        ("runtime_range_admission", RUNTIME_ADMISSION_SCHEMA, RUNTIME_ADMISSION_STATUS),
        (
            "operator_accumulation",
            OPERATOR_ATTESTATION_SCHEMA,
            OPERATOR_ATTESTATION_STATUS,
        ),
        (
            "range_reader_exact_semantics",
            RANGE_READER_ATTESTATION_SCHEMA,
            RANGE_READER_ATTESTATION_STATUS,
        ),
    )
    for field, schema, status in expected:
        row = _mapping(dual.get(field), label=f"source terminal dual execution {field}")
        _schema_status(
            row,
            schema=schema,
            status=status,
            label=f"source terminal dual execution {field}",
        )
        _text(
            row.get("seal_sha256"),
            label=f"source terminal dual execution {field} seal",
            sha256=True,
        )


def validate_future_receipt_bundle(
    *,
    child: ChildPreflight,
    bridge: Document,
    source_lease: Mapping[str, Any],
    source_terminal: Mapping[str, Any],
    source_eviction: Mapping[str, Any],
    native_lease: Mapping[str, Any],
    metadata_range_authority: Mapping[str, Any],
    raw_six_vector_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate future receipts without spawning or opening payload files.

    The existing guarded validator owns source->evict->native ordering.  This
    wrapper adds the future bridge's dual execution-attestation identity and
    exact outer handoff requirements.  It accepts receipt mappings only, so
    callers cannot use it as a source/model/process launcher.
    """
    authority = _mapping(
        metadata_range_authority.get("authority"), label="metadata range authority"
    )
    source_floor = _integer(
        _mapping(
            source_lease.get("fresh_pre_child_safety"),
            label="future source lease safety",
        ).get("minimum_reclaimable_bytes_required"),
        label="future source lease safety floor",
        minimum=1,
    )
    try:
        guarded = guarded_outer.validate_future_source_then_evict_then_native(
            source_lease=source_lease,
            source_terminal=source_terminal,
            source_eviction=source_eviction,
            native_lease=native_lease,
            authority=authority,
            raw_retention=raw_six_vector_contract,
            trace=child.trace,
            source_minimum_reclaimable_bytes=source_floor,
            maximum_window_bytes=child.maximum_window_bytes,
        )
    except guarded_outer.GuardedStreamedSourceOuterError as exc:
        raise SourceTeacherOuterPreflightError(
            f"guarded source/evict/native validation refused: {exc}"
        ) from exc
    _future_execution_attestations(source_terminal, bridge=bridge)
    execution = _mapping(
        source_terminal.get("streamed_execution"),
        label="future source terminal streamed execution",
    )
    for field in (
        "outer_reaped_child_before_terminal_receipt",
        "receipt_written_after_payload_fsyncs",
        "source_handles_closed_before_child_exit",
        "streamed_reader_cache_zeroed_before_child_exit",
    ):
        _require_true(execution.get(field), label=f"future source terminal {field}")
    return {
        "guarded_source_then_evict_then_native": guarded,
        "dual_attestation_bound": True,
        "metadata_validation_only_no_child_launched": True,
    }


def _refusal(*, child: ChildPreflight, blockers: Sequence[str]) -> dict[str, Any]:
    return seal(
        {
            "schema": SCHEMA,
            "status": REFUSED_STATUS,
            "child_preflight": _evidence(child.document),
            "blockers": list(blockers),
            "future_lifecycle_interface": _future_interface(),
            "spawn_permitted": False,
            "execution_boundary": {
                "source_tensor_payload_opened": False,
                "source_model_loaded_or_instantiated": False,
                "gpu_or_metal_invoked": False,
                "server_or_hcli_started_or_contacted": False,
                "lease_issued_or_consumed": False,
                "source_child_spawned": False,
                "native_child_spawned": False,
                "source_or_native_payload_written": False,
                "source_eviction_or_native_release_performed": False,
            },
            "claim_boundary": "CPU/file-only refusal. No child, source payload, model, GPU, server, HCLI, lease, eviction, native capture, or tournament action occurred.",
        }
    )


def build_outer_preflight(
    *,
    child_preflight_path: Path,
    dual_bridge_path: Path | None = None,
    source_lease_path: Path | None = None,
) -> dict[str, Any]:
    """Return a sealed reservation grammar or pre-spawn refusal.

    This deliberately validates only future *launch* authority.  It never
    creates a replay guard, capture root, process group, lease, or child.
    Those operations belong to a later reviewed physical runner.
    """
    child = _validate_child_preflight(
        _sealed(child_preflight_path, label="source child preflight")
    )
    blockers: list[str] = []
    bridge: Document | None = None
    try:
        _validate_child_feasibility(child)
    except SourceTeacherOuterPreflightError as exc:
        blockers.append(f"streamed_feasibility_or_exact_semantics_not_earned:{exc}")
    if child.bridge_pointer.get("present") is not True:
        blockers.append(
            "child_preflight_lacks_sealed_dual_attestation_runtime_admission_bridge"
        )
    if dual_bridge_path is None:
        blockers.append("sealed_dual_attestation_runtime_admission_bridge_absent")
    else:
        try:
            bridge = _sealed(
                dual_bridge_path, label="dual-attestation/runtime-admission bridge"
            )
            _validate_bridge(bridge, child=child)
        except SourceTeacherOuterPreflightError as exc:
            blockers.append(f"dual_attestation_runtime_admission_bridge_invalid:{exc}")
    source_lease: Document | None = None
    if source_lease_path is None:
        blockers.append("fresh_one_shot_source_lease_absent")
    else:
        try:
            source_lease = _sealed(source_lease_path, label="source lease")
            _validate_source_lease(source_lease, child=child)
        except SourceTeacherOuterPreflightError as exc:
            blockers.append(f"source_lease_invalid:{exc}")
    if blockers:
        return _refusal(child=child, blockers=blockers)
    assert bridge is not None and source_lease is not None
    return seal(
        {
            "schema": SCHEMA,
            "status": PREPARED_STATUS,
            "child_preflight": _evidence(child.document),
            "dual_attestation_runtime_admission_bridge": _evidence(bridge),
            "source_lease": _evidence(source_lease),
            "future_lifecycle_interface": _future_interface(),
            "reservation": {
                "schema": REPLAY_RESERVATION_SCHEMA,
                "status_only_after_a_separate_create_new_physical_reservation": REPLAY_RESERVATION_STATUS,
                "one_source_child_and_one_native_child_maximum": True,
                "this_preflight_did_not_create_a_reservation_or_capture_root": True,
                "this_preflight_did_not_spawn_a_child": True,
                "this_preflight_did_not_issue_or_consume_the_bound_lease": True,
            },
            "spawn_permitted": False,
            "execution_boundary": {
                "source_tensor_payload_opened": False,
                "source_model_loaded_or_instantiated": False,
                "gpu_or_metal_invoked": False,
                "server_or_hcli_started_or_contacted": False,
                "lease_issued_or_consumed": False,
                "source_child_spawned": False,
                "native_child_spawned": False,
                "source_or_native_payload_written": False,
                "source_eviction_or_native_release_performed": False,
            },
            "claim_boundary": "Prepared CPU/file-only outer grammar. A later reviewed runner still needs its own create-new replay reservation, reaper, receipt-last terminal writes, and separately authorized child/lease actions.",
        }
    )


def _write_new_json(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute() or path.exists() or not path.parent.is_dir():
        raise SourceTeacherOuterPreflightError(
            "--out must name a new absolute JSON path below an existing parent"
        )
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False
            )
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
    parser.add_argument("--child-preflight", type=Path, required=True)
    parser.add_argument("--dual-attestation-runtime-admission", type=Path)
    parser.add_argument("--source-lease", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_outer_preflight(
            child_preflight_path=args.child_preflight,
            dual_bridge_path=args.dual_attestation_runtime_admission,
            source_lease_path=args.source_lease,
        )
        _write_new_json(args.out, result)
    except SourceTeacherOuterPreflightError as exc:
        print(
            f"Q30 streamed source-teacher outer preflight could not write a refusal: {exc}"
        )
        return 2
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "status": result["status"],
                "seal_sha256": result["seal_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
