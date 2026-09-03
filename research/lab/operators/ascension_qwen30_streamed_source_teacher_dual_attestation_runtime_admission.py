#!/usr/bin/env python3
"""Prepare the non-authorizing Q30 dual-attestation / runtime-admission bridge.

This is a CPU/file-only metadata operator.  It binds sealed upstream receipts
and records the exact future schema names that a source-teacher child must
resolve after real execution.  It does not:

* open a source root or tensor payload
* earn either execution attestation
* issue or consume a lease
* start a child, GPU, server, or HCLI surface
* authorize source-teacher execution

The consumer shape is the one enforced by
``ascension_qwen30_streamed_source_teacher_child_preflight``:

* schema ``hawking.ascension.qwen30_streamed_source_teacher_dual_attestation_runtime_admission.v1``
* status ``PREPARED_QWEN30_STREAMED_SOURCE_TEACHER_DUAL_ATTESTATION_RUNTIME_ADMISSION_NOT_EXECUTED``
* ``upstream_metadata`` raw/seal pins for range, semantics, feasibility, raw
  six-vector, and current trace
* ``schema_resolution`` naming both future execution attestations without
  claiming they were earned
* ``future_source_worker`` geometry for the 48x370 streamed teacher

A prepared bridge is non-authorizing and cannot substitute for either
execution attestation.
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


SCHEMA = "hawking.ascension.qwen30_streamed_source_teacher_dual_attestation_runtime_admission.v1"
STATUS = "PREPARED_QWEN30_STREAMED_SOURCE_TEACHER_DUAL_ATTESTATION_RUNTIME_ADMISSION_NOT_EXECUTED"

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
FEASIBILITY_SCHEMA = (
    "hawking.ascension.qwen30_layer_streamed_source_bf16_final_logit_oracle_feasibility.v1"
)
FEASIBILITY_PREPARED_STATUS = (
    "PREPARED_QWEN30_LAYER_STREAMED_SOURCE_BF16_ORACLE_FEASIBILITY_NOT_EXECUTED"
)
FEASIBILITY_REFUSED_STATUS = (
    "REFUSED_QWEN30_LAYER_STREAMED_SOURCE_BF16_ORACLE_FEASIBILITY_UNSAFE_OR_UNPROVEN"
)
RAW_SIX_VECTOR_SCHEMA = (
    "hawking.ascension.qwen30_hq30gr2_raw_final_logit_retention_successor.v1"
)
RAW_SIX_VECTOR_STATUS = "PREPARED_RAW_FINAL_LOGIT_RETENTION_SUCCESSOR_NOT_RUN"
CURRENT_TRACE_SCHEMA = (
    "hawking.ascension.qwen30_hq30gr2_all_layer_current_trace_comparison.v1"
)
CURRENT_TRACE_STATUS = (
    "EARNED_CANDIDATE_LOCAL_ALL_LAYER_DIVERGENCE_UNQUALIFIED_NON_PROMOTABLE"
)

RUNTIME_RANGE_MAP_SCHEMA = "hawking.ascension.qwen30_source_bf16_range_map.v1"
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

MAX_METADATA_BYTES = 64 * 1024 * 1024
MAX_POSITIONED_READ_BYTES = 1024 * 1024
SOURCE_LAYERS = 48
SOURCE_FORWARDS = 370
PREFIX_TOKENS = 369
FORCED_TOKEN_ID = 949
VOCAB_ROWS = 151_936
F32_VECTOR_BYTES = VOCAB_ROWS * 4


class DualAttestationBridgeError(ValueError):
    """A dual-attestation bridge input is absent, unsealed, or inconsistent."""


@dataclass(frozen=True)
class BoundDocument:
    path: Path
    value: dict[str, Any]
    raw_document_sha256: str
    seal_sha256: str | None
    bytes: int

    def upstream_pointer(self, *, authority_content_sha256: str | None = None) -> dict[str, Any]:
        pointer: dict[str, Any] = {
            "path": str(self.path),
            "bytes": self.bytes,
            "raw_document_sha256": self.raw_document_sha256,
        }
        if self.seal_sha256 is not None:
            pointer["seal_sha256"] = self.seal_sha256
        if authority_content_sha256 is not None:
            pointer["authority_content_sha256"] = authority_content_sha256
        return pointer


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DualAttestationBridgeError(f"{label} must be an object")
    return dict(value)


def _text(value: object, *, label: str, sha256: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise DualAttestationBridgeError(f"{label} must be a non-empty string")
    if sha256 and not _is_sha256(value):
        raise DualAttestationBridgeError(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DualAttestationBridgeError(f"{label} must be an integer >= {minimum}")
    return value


def _require(value: object, *, expected: object, label: str) -> None:
    if value is not expected:
        raise DualAttestationBridgeError(f"{label} must be {expected!r}")


def _schema_status(
    document: Mapping[str, Any], *, schema: str, status: str | set[str], label: str
) -> None:
    if document.get("schema") != schema:
        raise DualAttestationBridgeError(f"{label} schema drifted")
    allowed = {status} if isinstance(status, str) else set(status)
    if document.get("status") not in allowed:
        raise DualAttestationBridgeError(f"{label} status drifted")


def _regular_json(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        raise DualAttestationBridgeError(f"{label} must be absolute")
    if path.suffix != ".json":
        raise DualAttestationBridgeError(f"{label} must be a .json metadata receipt")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise DualAttestationBridgeError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DualAttestationBridgeError(f"{label} must be a regular non-symlink file")
    if metadata.st_size <= 0 or metadata.st_size > MAX_METADATA_BYTES:
        raise DualAttestationBridgeError(
            f"{label} must contain 1..={MAX_METADATA_BYTES} metadata bytes"
        )
    return path.resolve(strict=True)


def _read_document(path: Path, *, label: str, sealed: bool) -> BoundDocument:
    clean = _regular_json(path, label=label)
    raw = clean.read_bytes()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DualAttestationBridgeError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise DualAttestationBridgeError(f"{label} is not an object")
    document = dict(parsed)
    seal_sha256: str | None = None
    if sealed:
        try:
            verified = verify(document, label=label)
        except SealIntegrityError as exc:
            raise DualAttestationBridgeError(f"{label} is unsealed or tampered: {exc}") from exc
        if not isinstance(verified, Mapping):
            raise DualAttestationBridgeError(f"{label} seal verification did not yield an object")
        document = dict(verified)
        seal_sha256 = _text(document.get("seal_sha256"), label=f"{label}.seal_sha256", sha256=True)
    return BoundDocument(
        path=clean,
        value=document,
        raw_document_sha256=_sha256_bytes(raw),
        seal_sha256=seal_sha256,
        bytes=len(raw),
    )


def _validate_range_authority(document: BoundDocument) -> str:
    root = document.value
    authority_content = _text(
        root.get("authority_content_sha256"),
        label="range authority content SHA",
        sha256=True,
    )
    authority = _mapping(root.get("authority"), label="range authority material")
    _schema_status(
        authority,
        schema=RANGE_AUTHORITY_SCHEMA,
        status=RANGE_AUTHORITY_STATUS,
        label="range authority",
    )
    source = _mapping(authority.get("source"), label="range authority source")
    if _integer(source.get("source_tensor_count"), label="range tensor count", minimum=1) != 18_867:
        raise DualAttestationBridgeError("range authority source tensor count drifted")
    if _integer(source.get("source_shard_count"), label="range shard count", minimum=1) != 16:
        raise DualAttestationBridgeError("range authority source shard count drifted")
    return authority_content


def _validate_semantics(document: BoundDocument) -> None:
    _schema_status(
        document.value,
        schema=SEMANTICS_SCHEMA,
        status=SEMANTICS_STATUS,
        label="semantics attester",
    )
    boundary = _mapping(document.value.get("execution_boundary"), label="semantics execution boundary")
    for key in (
        "source_tensor_payload_opened",
        "source_model_instantiated",
        "source_inference_executed",
        "gpu_or_metal_invoked",
        "server_started",
        "hcli_invoked",
        "lease_requested",
    ):
        if boundary.get(key) is not False:
            raise DualAttestationBridgeError(f"semantics attester {key} must remain false")


def _validate_feasibility(document: BoundDocument) -> None:
    root = document.value
    if document.seal_sha256 is None:
        raise DualAttestationBridgeError("streamed feasibility must be sealed")
    _schema_status(
        root,
        schema=FEASIBILITY_SCHEMA,
        status={FEASIBILITY_PREPARED_STATUS, FEASIBILITY_REFUSED_STATUS},
        label="streamed feasibility",
    )
    feasibility = _mapping(root.get("feasibility"), label="streamed feasibility.feasibility")
    _require(
        feasibility.get("oracle_execution_authorized"),
        expected=False,
        label="streamed feasibility.oracle_execution_authorized",
    )
    exact = _mapping(root.get("exact_trace"), label="streamed feasibility.exact_trace")
    if _integer(exact.get("prefix_token_count"), label="feasibility prefix") != PREFIX_TOKENS:
        raise DualAttestationBridgeError("streamed feasibility prefix drifted")
    if _integer(exact.get("forced_token_id"), label="feasibility forced token") != FORCED_TOKEN_ID:
        raise DualAttestationBridgeError("streamed feasibility forced token drifted")


def _validate_raw_six_vector(document: BoundDocument) -> None:
    root = document.value
    if document.seal_sha256 is None:
        raise DualAttestationBridgeError("raw six-vector contract must be sealed")
    _schema_status(
        root,
        schema=RAW_SIX_VECTOR_SCHEMA,
        status=RAW_SIX_VECTOR_STATUS,
        label="raw six-vector contract",
    )
    plan = _mapping(root.get("six_vector_retention_contract"), label="raw six-vector plan")
    if plan.get("dtype") != "f32le":
        raise DualAttestationBridgeError("raw six-vector dtype drifted")
    if _integer(plan.get("vocab_rows"), label="raw vocab rows", minimum=1) != VOCAB_ROWS:
        raise DualAttestationBridgeError("raw six-vector vocab rows drifted")
    if (
        _integer(plan.get("bytes_per_vector"), label="raw bytes per vector", minimum=1)
        != F32_VECTOR_BYTES
    ):
        raise DualAttestationBridgeError("raw six-vector bytes per vector drifted")
    if _integer(plan.get("required_payload_count"), label="raw payload count", minimum=1) != 6:
        raise DualAttestationBridgeError("raw six-vector payload count drifted")


def _validate_current_trace(document: BoundDocument) -> None:
    root = document.value
    if document.seal_sha256 is None:
        raise DualAttestationBridgeError("current trace must be sealed")
    _schema_status(
        root,
        schema=CURRENT_TRACE_SCHEMA,
        status=CURRENT_TRACE_STATUS,
        label="current trace",
    )


def build_dual_attestation_runtime_admission(
    *,
    range_authority_path: Path,
    semantics_path: Path,
    feasibility_path: Path,
    raw_six_vector_path: Path,
    current_trace_path: Path,
) -> dict[str, Any]:
    """Seal a non-authorizing dual bridge that binds future attestation schemas only."""
    range_document = _read_document(
        range_authority_path, label="metadata range authority", sealed=False
    )
    authority_content = _validate_range_authority(range_document)
    semantics = _read_document(semantics_path, label="semantics attester", sealed=False)
    _validate_semantics(semantics)
    feasibility = _read_document(feasibility_path, label="streamed feasibility", sealed=True)
    _validate_feasibility(feasibility)
    raw = _read_document(raw_six_vector_path, label="raw six-vector contract", sealed=True)
    _validate_raw_six_vector(raw)
    current = _read_document(current_trace_path, label="current trace", sealed=True)
    _validate_current_trace(current)
    return seal(
        {
            "schema": SCHEMA,
            "status": STATUS,
            "prepared": True,
            "execution_authorized": False,
            "upstream_metadata": {
                "range_authority": range_document.upstream_pointer(
                    authority_content_sha256=authority_content
                ),
                "semantics_attester": semantics.upstream_pointer(),
                "streamed_feasibility": feasibility.upstream_pointer(),
                "raw_six_vector_contract": raw.upstream_pointer(),
                "current_trace": current.upstream_pointer(),
            },
            "schema_resolution": {
                "runtime_range_map_schema": RUNTIME_RANGE_MAP_SCHEMA,
                "runtime_admission_schema": RUNTIME_ADMISSION_SCHEMA,
                "runtime_admission_status_only_after_bounded_source_validation": RUNTIME_ADMISSION_STATUS,
                "operator_accumulation_execution_attestation": {
                    "schema": OPERATOR_ATTESTATION_SCHEMA,
                    "status": OPERATOR_ATTESTATION_STATUS,
                    "earned_by_this_bridge": False,
                    "status_only_after_real_source_child_execution": True,
                },
                "range_reader_exact_semantics_attestation": {
                    "schema": RANGE_READER_ATTESTATION_SCHEMA,
                    "status": RANGE_READER_ATTESTATION_STATUS,
                    "earned_by_this_bridge": False,
                    "status_only_after_real_source_child_execution": True,
                },
                "both_execution_attestations_required_after_source_child": True,
                "runtime_range_admission_required_before_payload_open": True,
                "bridge_does_not_authorize_execution": True,
                "a_prepared_bridge_is_non_authorizing_and_cannot_substitute_for_either_execution_attestation": True,
            },
            "future_source_worker": {
                "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
                "source_layers": SOURCE_LAYERS,
                "source_forwards": SOURCE_FORWARDS,
                "source_f32le_vectors": 2,
                "native_f32le_vectors": 4,
                "one_bounded_window_only": True,
                "source_payloads_durable_before_eviction": True,
                "close_handles_and_clear_cache_before_eviction_receipt": True,
                "separate_native_four_vector_phase_required": True,
            },
            "execution_boundary": {
                "source_tensor_payload_opened": False,
                "source_model_loaded_or_instantiated": False,
                "whole_source_model_resident": False,
                "operator_or_reader_execution_attestation_earned": False,
                "runtime_range_admission_earned": False,
                "gpu_metal_mps_or_other_accelerator_invoked": False,
                "server_started_or_contacted": False,
                "hcli_invoked": False,
                "lease_requested_issued_or_consumed": False,
                "child_process_started": False,
                "source_teacher_or_native_vector_written": False,
            },
            "claim_boundary": (
                "Prepared CPU/file-only dual-attestation / runtime-admission bridge only. "
                "It binds upstream metadata and names the two future execution attestation "
                "schemas; it does not earn either attestation, authorize source-teacher "
                "execution, open source payloads, issue a lease, or start a child."
            ),
        }
    )


def _write_new(path: Path, document: Mapping[str, Any]) -> None:
    if not path.is_absolute():
        raise DualAttestationBridgeError("--out must be absolute")
    if path.suffix != ".json":
        raise DualAttestationBridgeError("--out must be a .json path")
    if not path.parent.is_dir():
        raise DualAttestationBridgeError("--out parent directory must already exist")
    if path.exists():
        raise DualAttestationBridgeError("--out must name a new immutable receipt")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    except FileExistsError as exc:
        raise DualAttestationBridgeError("--out must name a new immutable receipt") from exc
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(document), handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--range-authority", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--feasibility", type=Path, required=True)
    parser.add_argument("--raw-six-vector", type=Path, required=True)
    parser.add_argument("--current-trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_dual_attestation_runtime_admission(
            range_authority_path=args.range_authority,
            semantics_path=args.semantics,
            feasibility_path=args.feasibility,
            raw_six_vector_path=args.raw_six_vector,
            current_trace_path=args.current_trace,
        )
        _write_new(args.out, result)
    except DualAttestationBridgeError as exc:
        print(json.dumps({"status": "REFUSED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "output": str(args.out.resolve()),
                "status": result["status"],
                "seal_sha256": result["seal_sha256"],
                "execution_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
