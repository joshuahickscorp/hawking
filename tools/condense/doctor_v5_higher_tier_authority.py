#!/usr/bin/env python3.12
"""Pinned operator authority for exact Doctor V5 models above 120B.

Raw manifests, self-hashes, and ``reviewed: true`` flags are structural input,
never release authority.  This module parses the concrete tensor authority,
closes every tensor/source/range/work-unit binding, then signs a short-lived
canonical attestation in a source-pinned SSHSIG namespace.  Exact 10x4
planning may consume only the resulting sealed envelope.

No command here launches a model, opens a GPU, mutates a runtime default, or
writes a live queue.  Outputs use the retained-dirfd immutable installer shared
with the Appendix operator trust root.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from typing import Any

import appendix_physical_counter_authority as operator_root
import physical_counter_attestation


ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_MANIFEST_SCHEMA = "hawking.doctor_v5_stream_source_manifest.v1"
PARAMETER_AUTHORITY_SCHEMA = "hawking.doctor_v5_higher_tier_parameter_authority.v2"
ATTESTATION_SCHEMA = "hawking.doctor_v5_higher_tier_manifest_attestation.v1"
ENVELOPE_SCHEMA = "hawking.doctor_v5_higher_tier_signed_manifest_envelope.v1"
SEALED_MANIFEST_SCHEMA = "hawking.doctor_v5_higher_tier_operator_sealed_manifest.v1"
SSHSIG_NAMESPACE = "hawking-doctor-v5-higher-tier-manifest-v1"
SIGNER_IDENTITY = operator_root.SIGNER_IDENTITY
DEFAULT_ALLOWED_SIGNERS = operator_root.DEFAULT_ALLOWED_SIGNERS
DEFAULT_PRIVATE_KEY = operator_root.DEFAULT_OPERATOR_PRIVATE_KEY
PINNED_ALLOWED_SIGNERS_SHA256 = operator_root.PINNED_ALLOWED_SIGNERS_SHA256
PINNED_PUBLIC_KEY_BLOB_SHA256 = operator_root.PINNED_PUBLIC_KEY_BLOB_SHA256
SSH_KEYGEN = operator_root.SSH_KEYGEN
DEFAULT_VALIDITY_SECONDS = 24 * 60 * 60
MAX_VALIDITY_SECONDS = 7 * 24 * 60 * 60
MAX_JSON_BYTES = 256 * 1024 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")

PARAMETER_WRAPPER_FIELDS = {
    "artifact", "logical_parameters", "stored_parameters", "tensor_count",
    "tensor_layout_sha256", "parameter_ranges_sha256", "reviewed",
    "authority_sha256",
}
TENSOR_FIELDS = {
    "tensor_key", "role", "logical_dtype", "storage_encoding", "shape",
    "logical_parameters", "stored_parameters", "packing_overhead_bytes",
    "stored_bytes", "source_id", "absolute_byte_range", "range_sha256",
}
LOGICAL_DTYPES = {"F32", "F16", "BF16", "I8", "U8"}
STORAGE_BITS: dict[str, tuple[int, int]] = {
    "fp32": (32, 1), "fp16": (16, 1), "bf16": (16, 1),
    "int8": (8, 1), "uint8": (8, 1), "int4-packed": (4, 1),
    "tq-1bpw": (1, 1), "tq-0.5bpw": (1, 2),
    "tq-0.25bpw": (1, 4), "tq-0.1bpw": (1, 10),
}
SignatureVerifier = Callable[[dict[str, Any], bytes], tuple[bool, str]]


class HigherTierAuthorityError(ValueError):
    """A manifest, operator signature, or immutable output is untrusted."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _stamp(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    stamped = copy.deepcopy(dict(value))
    stamped.pop(field, None)
    stamped[field] = canonical_sha256(stamped)
    return stamped


def _hex(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _canonical_hash_errors(value: Any, field: str, *, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} must be an object"]
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop(field, None)
    if not _hex(claimed):
        return [f"{label}.{field} is invalid"]
    try:
        observed = canonical_sha256(unstamped)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return [f"{label} is not canonical finite JSON"]
    return [] if observed == claimed else [f"{label}.{field} mismatch"]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _safe_bytes(path: pathlib.Path, *, maximum: int = MAX_JSON_BYTES) -> bytes:
    try:
        return operator_root._safe_file_bytes(path, maximum=maximum)
    except FileNotFoundError:
        raise
    except (OSError, operator_root.AuthorityError) as exc:
        raise HigherTierAuthorityError(f"unsafe immutable input {path}: {exc}") from exc


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise HigherTierAuthorityError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise HigherTierAuthorityError(f"{label} JSON root is not an object")
    return value


def _safe_json(path: pathlib.Path) -> dict[str, Any]:
    return _strict_json_bytes(_safe_bytes(path), label=str(path))


def _atomic_bytes(path: pathlib.Path, raw: bytes) -> None:
    try:
        operator_root._atomic_bytes(path, raw, mode=0o444)
    except operator_root.AuthorityError as exc:
        raise HigherTierAuthorityError(str(exc)) from exc


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _atomic_bytes(path, raw)


def _artifact_errors(value: Any, *, label: str, verify_file: bool) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"} \
            or not isinstance(value.get("path"), str) \
            or not pathlib.Path(value.get("path", "")).is_absolute() \
            or not _hex(value.get("sha256")) \
            or isinstance(value.get("bytes"), bool) \
            or not isinstance(value.get("bytes"), int) or value["bytes"] <= 0:
        return [f"{label} artifact identity is invalid"]
    if not verify_file:
        return []
    try:
        raw = _safe_bytes(pathlib.Path(value["path"]))
    except (OSError, HigherTierAuthorityError) as exc:
        return [f"{label} artifact cannot be verified: {exc}"]
    if len(raw) != value["bytes"] or hashlib.sha256(raw).hexdigest() != value["sha256"]:
        return [f"{label} artifact differs from its immutable binding"]
    return []


def _bound_json(
    value: Any, *, label: str, verify_file: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors = _artifact_errors(value, label=label, verify_file=verify_file)
    if errors or not verify_file:
        return None, errors
    try:
        raw = _safe_bytes(pathlib.Path(value["path"]))
        parsed = _strict_json_bytes(raw, label=label)
    except (OSError, HigherTierAuthorityError) as exc:
        return None, [f"{label} cannot be parsed: {exc}"]
    return parsed, []


def _positive_int(value: Any, *, allow_zero: bool = False) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) \
        and (value >= 0 if allow_zero else value > 0)


def _validate_parameter_authority(
    value: Any, *, model: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]], list[str]]:
    expected = {
        "schema", "model", "tensors", "logical_parameters", "stored_parameters",
        "tensor_count", "tensor_layout_sha256", "parameter_ranges_sha256",
        "authority_sha256",
    }
    errors: list[str] = []
    if not isinstance(value, dict) or set(value) != expected:
        return None, {}, ["parameter authority fields are incomplete or unexpected"]
    errors.extend(_canonical_hash_errors(
        value, "authority_sha256", label="parameter_authority",
    ))
    if value.get("schema") != PARAMETER_AUTHORITY_SCHEMA:
        errors.append("parameter authority schema is invalid")
    identity = value.get("model")
    expected_identity = {
        "label": model.get("label"),
        "hf_id_or_source_id": model.get("hf_id_or_source_id"),
        "family": model.get("family"),
        "architecture_kind": model.get("architecture_kind"),
    }
    if identity != expected_identity:
        errors.append("parameter authority model identity differs from source manifest")
    tensors = value.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        errors.append("parameter authority has no exact tensor entries")
        tensors = []
    by_key: dict[str, dict[str, Any]] = {}
    normalized: list[dict[str, Any]] = []
    range_rows: list[dict[str, Any]] = []
    logical_total = stored_total = 0
    for index, row in enumerate(tensors):
        label = f"parameter tensor[{index}]"
        row_error_count = len(errors)
        if not isinstance(row, dict) or set(row) != TENSOR_FIELDS:
            errors.append(f"{label} fields are incomplete or unexpected")
            continue
        key = row.get("tensor_key")
        shape = row.get("shape")
        byte_range = row.get("absolute_byte_range")
        source_id = row.get("source_id")
        if not isinstance(key, str) or not key or key in by_key:
            errors.append(f"{label} key is invalid or duplicated")
            continue
        role = row.get("role")
        logical_dtype = row.get("logical_dtype")
        encoding = row.get("storage_encoding")
        if role not in {"model_parameter", "storage_sidecar"} \
                or logical_dtype not in LOGICAL_DTYPES \
                or encoding not in STORAGE_BITS \
                or not isinstance(shape, list) or not shape \
                or any(not _positive_int(axis) for axis in shape):
            errors.append(f"{label} role/dtype/storage encoding/shape is invalid")
        logical = row.get("logical_parameters")
        stored = row.get("stored_parameters")
        stored_bytes = row.get("stored_bytes")
        overhead = row.get("packing_overhead_bytes")
        if not _positive_int(logical, allow_zero=True) or not _positive_int(stored) \
                or not _positive_int(stored_bytes) \
                or not _positive_int(overhead, allow_zero=True):
            errors.append(f"{label} parameter/byte counts are invalid")
        elif isinstance(shape, list) and shape and all(_positive_int(axis) for axis in shape):
            elements = math.prod(shape)
            expected_logical = elements if role == "model_parameter" else 0
            if logical != expected_logical or stored != elements:
                errors.append(f"{label} logical/stored counts differ from role and shape")
            if encoding in STORAGE_BITS:
                numerator, denominator = STORAGE_BITS[encoding]
                payload_bytes = (stored * numerator + denominator * 8 - 1) \
                    // (denominator * 8)
                if stored_bytes != payload_bytes + overhead:
                    errors.append(f"{label} stored bytes differ from exact encoding contract")
                if overhead > 1_048_576:
                    errors.append(f"{label} packing overhead exceeds the reviewed bound")
        source_size = sources.get(source_id, {}).get("bytes") \
            if isinstance(source_id, str) else None
        if not isinstance(source_id, str) or source_id not in sources \
                or not isinstance(byte_range, list) or len(byte_range) != 2 \
                or any(not _positive_int(item, allow_zero=True) for item in byte_range) \
                or byte_range[1] <= byte_range[0] \
                or not _positive_int(stored_bytes) \
                or byte_range[1] - byte_range[0] != stored_bytes \
                or not _positive_int(source_size) or byte_range[1] > source_size \
                or not _hex(row.get("range_sha256")):
            errors.append(f"{label} source/range binding is invalid")
        by_key[key] = row
        if len(errors) != row_error_count:
            continue
        normalized.append(row)
        if _positive_int(logical):
            logical_total += logical
        if _positive_int(stored):
            stored_total += stored
        range_rows.append({
            "tensor_key": key, "source_id": source_id,
            "absolute_byte_range": byte_range,
            "stored_bytes": stored_bytes, "range_sha256": row.get("range_sha256"),
        })
    normalized.sort(key=lambda row: row["tensor_key"])
    range_rows.sort(key=lambda row: row["tensor_key"])
    ranges_by_source: dict[str, list[tuple[int, int, str]]] = {}
    for row in normalized:
        start, end = row["absolute_byte_range"]
        ranges_by_source.setdefault(row["source_id"], []).append(
            (start, end, row["tensor_key"])
        )
    for source_id, source in sources.items():
        ordered = sorted(ranges_by_source.get(source_id, []))
        source_bytes = source.get("bytes")
        if not ordered or not _positive_int(source_bytes) \
                or ordered[0][0] != 0 or ordered[-1][1] != source_bytes \
                or any(left[1] != right[0] for left, right in zip(ordered, ordered[1:])):
            errors.append(f"parameter tensor ranges do not exactly cover source: {source_id}")
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            errors.append(f"parameter tensor ranges overlap: {source_id}")
    if set(ranges_by_source) != set(sources):
        errors.append("parameter tensors reference missing or uncovered sources")
    try:
        layout_root = canonical_sha256(normalized)
        range_root = canonical_sha256(range_rows)
    except (TypeError, ValueError, OverflowError, RecursionError):
        layout_root = range_root = ""
        errors.append("parameter tensor/range rows are not canonical finite JSON")
    if value.get("logical_parameters") != logical_total \
            or value.get("stored_parameters") != stored_total \
            or value.get("tensor_count") != len(by_key):
        errors.append("parameter authority logical/stored/tensor totals do not recompute")
    if value.get("tensor_layout_sha256") != layout_root:
        errors.append("parameter authority tensor layout root does not recompute")
    if value.get("parameter_ranges_sha256") != range_root:
        errors.append("parameter authority deterministic range root does not recompute")
    summary = {
        "authority_sha256": value.get("authority_sha256"),
        "logical_parameters": logical_total,
        "stored_parameters": stored_total,
        "tensor_count": len(by_key),
        "tensor_layout_sha256": layout_root,
        "parameter_ranges_sha256": range_root,
    }
    return (summary if not errors else None), by_key, errors


def validate_parameter_authority(
    value: Any, *, model: Mapping[str, Any], sources: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]], list[str]]:
    """No-throw public wrapper around concrete tensor semantics."""
    try:
        summary, rows, errors = _validate_parameter_authority(
            value, model=model, sources=sources,
        )
        return (None, {}, errors) if errors else (summary, rows, [])
    except (
        TypeError, ValueError, KeyError, AttributeError, OverflowError,
        RecursionError, OSError,
    ) as exc:
        return None, {}, [f"parameter authority is malformed: {exc}"]


def _binding_artifact_errors(
    value: Any, *, field: str, label: str, verify_files: bool,
) -> list[str]:
    if not isinstance(value, dict) or not _hex(value.get(field)):
        return [f"{label} binding is absent or invalid"]
    errors = _canonical_hash_errors(value, field, label=label)
    errors.extend(_artifact_errors(
        value.get("artifact"), label=label, verify_file=verify_files,
    ))
    return errors


def _inspect_core_manifest(
    manifest: Any, *, verify_files: bool = True,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Recompute every authority binding needed by the operator signature."""
    errors: list[str] = []
    if not isinstance(manifest, dict) or manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
        return None, ["higher-tier core source manifest schema is invalid"]
    if not _hex(manifest.get("manifest_sha256")):
        errors.append("higher-tier core source manifest hash is invalid")
    else:
        try:
            expected_manifest = canonical_sha256({
                key: value for key, value in manifest.items() if key != "manifest_sha256"
            })
        except (TypeError, ValueError, OverflowError, RecursionError):
            expected_manifest = None
        if manifest["manifest_sha256"] != expected_manifest:
            errors.append("higher-tier core source manifest self-hash differs")
    model = manifest.get("model") if isinstance(manifest.get("model"), dict) else {}
    sources_list = manifest.get("sources") if isinstance(manifest.get("sources"), list) else []
    sources: dict[str, Mapping[str, Any]] = {}
    for source in sources_list:
        source_id = source.get("source_id") if isinstance(source, dict) else None
        if isinstance(source_id, str) and source_id and source_id not in sources:
            sources[source_id] = source
    wrapper = model.get("parameter_authority")
    parameter_doc: dict[str, Any] | None = None
    if not isinstance(wrapper, dict) or set(wrapper) != PARAMETER_WRAPPER_FIELDS:
        errors.append("concrete parameter authority wrapper is absent")
        tensor_summary, tensors = None, {}
    else:
        parameter_doc, artifact_errors = _bound_json(
            wrapper.get("artifact"), label="parameter authority", verify_file=verify_files,
        )
        errors.extend(artifact_errors)
        if verify_files and parameter_doc is not None:
            tensor_summary, tensors, parameter_errors = validate_parameter_authority(
                parameter_doc, model=model, sources=sources,
            )
            errors.extend(parameter_errors)
        else:
            tensor_summary, tensors = None, {}
        if tensor_summary is not None:
            for field in (
                "authority_sha256", "logical_parameters", "stored_parameters",
                "tensor_count", "tensor_layout_sha256", "parameter_ranges_sha256",
            ):
                if wrapper.get(field) != tensor_summary[field]:
                    errors.append(f"parameter wrapper {field} differs from concrete authority")
            if wrapper.get("reviewed") is not True \
                    or model.get("logical_parameters") != tensor_summary["logical_parameters"] \
                    or model.get("parameter_authority_sha256") \
                    != tensor_summary["authority_sha256"]:
                errors.append("parameter wrapper/model identity differs from concrete authority")

    architecture = manifest.get("architecture_adapter")
    errors.extend(_binding_artifact_errors(
        architecture, field="binding_sha256", label="architecture adapter",
        verify_files=verify_files,
    ))
    if isinstance(architecture, dict) and (
        architecture.get("reviewed") is not True
        or architecture.get("default_off") is not True
        or architecture.get("family") != model.get("family")
        or architecture.get("architecture_kind") != model.get("architecture_kind")
    ):
        errors.append("architecture adapter semantics differ from exact model")
    tokenizer = manifest.get("tokenizer_binding")
    errors.extend(_binding_artifact_errors(
        tokenizer, field="binding_sha256", label="tokenizer binding",
        verify_files=verify_files,
    ))
    if isinstance(tokenizer, dict) and (
        tokenizer.get("reviewed") is not True
        or tokenizer.get("tokenizer_sha256")
        != (tokenizer.get("artifact") or {}).get("sha256")
        or not _hex(tokenizer.get("chat_template_sha256"))
        or not _hex(tokenizer.get("special_tokens_sha256"))
    ):
        errors.append("tokenizer/chat-template semantics are invalid")
    lifecycle = manifest.get("lifecycle_manifest")
    transport = manifest.get("transport_manifest")
    for value, label in ((lifecycle, "lifecycle manifest"), (transport, "transport manifest")):
        errors.extend(_binding_artifact_errors(
            value, field="binding_sha256", label=label, verify_files=verify_files,
        ))
        if isinstance(value, dict) and (
            value.get("source_deletion_permitted") is not False
            or value.get("immutable") is not True
            or not _hex(value.get("rollback_cas_sha256"))
        ):
            errors.append(f"{label} weakens immutable lifecycle/rollback policy")

    assigned: dict[str, tuple[str, Mapping[str, Any]]] = {}
    units = manifest.get("work_units") if isinstance(manifest.get("work_units"), list) else []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        unit_id = unit.get("unit_id")
        unit_logical = 0
        ranges = unit.get("source_ranges") if isinstance(unit.get("source_ranges"), list) else []
        for source_range in ranges:
            if not isinstance(source_range, dict):
                continue
            keys = source_range.get("tensor_keys") \
                if isinstance(source_range.get("tensor_keys"), list) else []
            if len(keys) != 1:
                errors.append(
                    f"exact work-unit source range must name one tensor: {unit_id}"
                )
            for key in keys:
                tensor = tensors.get(key) if isinstance(key, str) else None
                if tensor is None:
                    errors.append(f"work-unit tensor key is absent from parameter authority: {key}")
                    continue
                if key in assigned:
                    errors.append(f"parameter tensor is assigned more than once: {key}")
                assigned[key] = (str(unit_id), source_range)
                unit_logical += tensor["logical_parameters"]
                if source_range.get("source_id") != tensor["source_id"] \
                        or source_range.get("absolute_byte_range") \
                        != tensor["absolute_byte_range"] \
                        or source_range.get("range_sha256") != tensor["range_sha256"]:
                    errors.append(f"work-unit tensor/source/range differs: {key}")
        if tensors and unit.get("logical_parameters") != unit_logical:
            errors.append(f"work-unit logical count differs from tensor authority: {unit_id}")
    if tensors and set(assigned) != set(tensors):
        errors.append("work-unit tensor coverage does not exactly close parameter authority")
    coverage = manifest.get("coverage") if isinstance(manifest.get("coverage"), dict) else {}
    if tensor_summary is not None and (
        coverage.get("logical_parameters") != tensor_summary["logical_parameters"]
        or coverage.get("tensor_count") != tensor_summary["tensor_count"]
        or coverage.get("tensor_layout_sha256") != tensor_summary["tensor_layout_sha256"]
        or coverage.get("all_model_tensors_assigned_exactly_once") is not True
    ):
        errors.append("manifest coverage root differs from concrete parameter authority")

    remote_source_hashes: dict[str, str] = {}
    remote_range_hashes: dict[str, str] = {}
    for source in sources_list:
        if not isinstance(source, dict) or source.get("transport") not in {
            "https_range", "object_range",
        }:
            continue
        source_id = source.get("source_id")
        authority_doc, authority_errors = _bound_json(
            source.get("authority_artifact"),
            label=f"remote source {source_id} authority", verify_file=verify_files,
        )
        errors.extend(authority_errors)
        if verify_files and isinstance(authority_doc, dict):
            expected = {
                "schema", "source_id", "uri", "immutable_version", "bytes",
                "sha256", "reviewed", "authority_sha256",
            }
            semantic = {key: value for key, value in authority_doc.items()
                        if key != "authority_sha256"}
            if set(authority_doc) != expected \
                    or authority_doc.get("schema") \
                    != "hawking.doctor_v5_remote_source_authority.v1" \
                    or authority_doc.get("source_id") != source_id \
                    or authority_doc.get("uri") != source.get("uri") \
                    or authority_doc.get("immutable_version") \
                    != source.get("immutable_version") \
                    or authority_doc.get("bytes") != source.get("bytes") \
                    or authority_doc.get("sha256") != source.get("sha256") \
                    or authority_doc.get("reviewed") is not True \
                    or authority_doc.get("authority_sha256") \
                    != canonical_sha256(semantic):
                errors.append(f"remote source authority semantics differ: {source_id}")
            else:
                remote_source_hashes[str(source_id)] = authority_doc["authority_sha256"]
    for unit in units:
        if not isinstance(unit, dict):
            continue
        for source_range in unit.get("source_ranges", []):
            if not isinstance(source_range, dict):
                continue
            source = sources.get(source_range.get("source_id"), {})
            if source.get("transport") not in {"https_range", "object_range"}:
                continue
            byte_range = source_range.get("absolute_byte_range")
            range_id = (
                f"{source_range.get('source_id')}:"
                f"{byte_range[0]}-{byte_range[1]}"
                if isinstance(byte_range, list) and len(byte_range) == 2 else "invalid"
            )
            receipt, receipt_errors = _bound_json(
                source_range.get("range_receipt_artifact"),
                label=f"remote range {range_id}", verify_file=verify_files,
            )
            errors.extend(receipt_errors)
            if verify_files and isinstance(receipt, dict):
                semantic = {key: value for key, value in receipt.items()
                            if key != "receipt_sha256"}
                if set(receipt) != {
                    "schema", "source_id", "uri", "immutable_version",
                    "absolute_byte_range", "range_sha256", "range_bytes",
                    "verified", "receipt_sha256",
                } or receipt.get("schema") \
                        != "hawking.doctor_v5_remote_range_receipt.v1" \
                        or receipt.get("source_id") != source_range.get("source_id") \
                        or receipt.get("uri") != source.get("uri") \
                        or receipt.get("immutable_version") \
                        != source.get("immutable_version") \
                        or receipt.get("absolute_byte_range") != byte_range \
                        or receipt.get("range_sha256") \
                        != source_range.get("range_sha256") \
                        or not isinstance(byte_range, list) or len(byte_range) != 2 \
                        or receipt.get("range_bytes") != byte_range[1] - byte_range[0] \
                        or receipt.get("verified") is not True \
                        or receipt.get("receipt_sha256") != canonical_sha256(semantic):
                    errors.append(f"remote range receipt semantics differ: {range_id}")
                else:
                    if range_id in remote_range_hashes:
                        errors.append(f"remote range receipt is reused: {range_id}")
                    remote_range_hashes[range_id] = receipt["receipt_sha256"]

    if errors or tensor_summary is None:
        return None, list(dict.fromkeys(errors))
    valid_sources = [row for row in sources_list if isinstance(row, dict)]
    source_rows = [
        {
            key: source.get(key) for key in (
                "source_id", "transport", "path", "uri", "immutable_version",
                "bytes", "sha256", "immutable_content", "range_reads_supported",
            ) if key in source
        }
        for source in sorted(valid_sources, key=lambda row: row.get("source_id", ""))
    ]
    bindings = {
        "source_manifest_schema": SOURCE_MANIFEST_SCHEMA,
        "source_manifest_sha256": manifest["manifest_sha256"],
        "source_manifest_canonical_sha256": canonical_sha256(manifest),
        "model_identity_sha256": canonical_sha256({
            key: model.get(key) for key in (
                "label", "hf_id_or_source_id", "family", "architecture_kind",
                "logical_parameters", "parameter_authority_sha256",
            )
        }),
        **tensor_summary,
        "architecture_adapter_binding_sha256": architecture["binding_sha256"],
        "tokenizer_binding_sha256": tokenizer["binding_sha256"],
        "lifecycle_binding_sha256": lifecycle["binding_sha256"],
        "transport_binding_sha256": transport["binding_sha256"],
        "remote_source_authority_sha256": remote_source_hashes,
        "remote_range_receipt_sha256": remote_range_hashes,
        "source_identity_root_sha256": canonical_sha256(source_rows),
        "work_unit_root_sha256": canonical_sha256(units),
    }
    return bindings, []


def inspect_core_manifest(
    manifest: Any, *, verify_files: bool = True,
) -> tuple[dict[str, Any] | None, list[str]]:
    """No-throw public validator for an unsealed exact manifest core."""
    try:
        return _inspect_core_manifest(manifest, verify_files=verify_files)
    except (
        TypeError, ValueError, KeyError, AttributeError, OverflowError,
        RecursionError, OSError,
    ) as exc:
        return None, [f"higher-tier core manifest is malformed: {exc}"]


def build_manifest_attestation(
    manifest: Mapping[str, Any], *, issued_at_unix_ns: int | None = None,
    valid_seconds: int = DEFAULT_VALIDITY_SECONDS,
) -> dict[str, Any]:
    bindings, errors = inspect_core_manifest(manifest, verify_files=True)
    if errors or bindings is None:
        raise HigherTierAuthorityError(
            "cannot draft higher-tier manifest authority: " + "; ".join(errors)
        )
    # Reuse the complete structural/range validator without treating its raw
    # self-hash as operator authority.  The private flag is not a release path;
    # only the signature produced below can make this core exact-plan eligible.
    import doctor_v5_higher_tier_scaffold as higher
    structural = higher.validate_source_manifest(
        dict(manifest), require_exact_wiring=True, verify_files=True,
        _operator_authorized_core=True,
    )
    if structural:
        raise HigherTierAuthorityError(
            "cannot sign structurally invalid higher-tier manifest: "
            + "; ".join(structural)
        )
    if isinstance(valid_seconds, bool) or not isinstance(valid_seconds, int) \
            or not 1 <= valid_seconds <= MAX_VALIDITY_SECONDS:
        raise HigherTierAuthorityError(
            f"manifest authority validity must be 1..{MAX_VALIDITY_SECONDS} seconds"
        )
    issued = time.time_ns() if issued_at_unix_ns is None else issued_at_unix_ns
    if not _positive_int(issued):
        raise HigherTierAuthorityError("manifest authority issue time is invalid")
    return _stamp({
        "schema": ATTESTATION_SCHEMA,
        **bindings,
        "issued_at_unix_ns": issued,
        "expires_at_unix_ns": issued + valid_seconds * 1_000_000_000,
        "runtime_defaults_changed": False,
        "execution_permitted": False,
    }, "attestation_sha256")


def validate_manifest_attestation(
    value: Any, *, manifest: Mapping[str, Any], now_unix_ns: int | None = None,
) -> list[str]:
    bindings, binding_errors = inspect_core_manifest(manifest, verify_files=True)
    if not isinstance(value, dict):
        return ["higher-tier manifest attestation is absent"]
    expected_fields = {
        "schema", "source_manifest_schema", "source_manifest_sha256",
        "source_manifest_canonical_sha256", "model_identity_sha256",
        "authority_sha256", "logical_parameters", "stored_parameters", "tensor_count",
        "tensor_layout_sha256", "parameter_ranges_sha256",
        "architecture_adapter_binding_sha256", "tokenizer_binding_sha256",
        "lifecycle_binding_sha256", "transport_binding_sha256",
        "remote_source_authority_sha256", "remote_range_receipt_sha256",
        "source_identity_root_sha256", "work_unit_root_sha256",
        "issued_at_unix_ns", "expires_at_unix_ns", "runtime_defaults_changed",
        "execution_permitted", "attestation_sha256",
    }
    errors = list(binding_errors)
    if set(value) != expected_fields:
        errors.append("higher-tier manifest attestation fields are incomplete or unexpected")
    errors.extend(_canonical_hash_errors(value, "attestation_sha256", label="manifest_attestation"))
    if value.get("schema") != ATTESTATION_SCHEMA:
        errors.append("higher-tier manifest attestation schema is invalid")
    if isinstance(bindings, dict):
        for field, expected in bindings.items():
            if value.get(field) != expected:
                errors.append(f"higher-tier manifest attestation {field} differs")
    issued, expires = value.get("issued_at_unix_ns"), value.get("expires_at_unix_ns")
    now = time.time_ns() if now_unix_ns is None else now_unix_ns
    if not _positive_int(now):
        errors.append("higher-tier manifest verification time is invalid")
    if not _positive_int(issued) or not _positive_int(expires) or expires <= issued \
            or expires - issued > MAX_VALIDITY_SECONDS * 1_000_000_000:
        errors.append("higher-tier manifest authority validity interval is invalid")
    elif _positive_int(now) and now < issued:
        errors.append("higher-tier manifest authority is not yet valid")
    elif _positive_int(now) and now > expires:
        errors.append("higher-tier manifest authority is expired")
    if value.get("runtime_defaults_changed") is not False \
            or value.get("execution_permitted") is not False:
        errors.append("higher-tier manifest attestation weakens default-off policy")
    return list(dict.fromkeys(errors))


def _allowed_signers_identity() -> dict[str, Any]:
    raw = _safe_bytes(DEFAULT_ALLOWED_SIGNERS, maximum=1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != PINNED_ALLOWED_SIGNERS_SHA256:
        raise HigherTierAuthorityError("source-pinned allowed-signers bytes changed")
    try:
        lines = [line.split() for line in raw.decode("utf-8").splitlines() if line.strip()]
    except UnicodeError as exc:
        raise HigherTierAuthorityError("allowed-signers file is not UTF-8") from exc
    if len(lines) != 1 or len(lines[0]) != 3 or lines[0][0] != SIGNER_IDENTITY \
            or lines[0][1] != "ssh-ed25519" \
            or hashlib.sha256(f"{lines[0][1]} {lines[0][2]}".encode("ascii")).hexdigest() \
            != PINNED_PUBLIC_KEY_BLOB_SHA256:
        raise HigherTierAuthorityError("allowed-signers does not contain the source-pinned key")
    return physical_counter_attestation.file_identity(DEFAULT_ALLOWED_SIGNERS)


def _signature_identity_errors(value: Any, *, verify_file: bool) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"} \
            or not isinstance(value.get("path"), str) \
            or not pathlib.Path(value.get("path", "")).is_absolute() \
            or not _hex(value.get("sha256")) \
            or not _positive_int(value.get("size_bytes")):
        return ["higher-tier detached signature identity is invalid"]
    if verify_file:
        try:
            observed = physical_counter_attestation.file_identity(pathlib.Path(value["path"]))
        except (OSError, ValueError) as exc:
            return [f"higher-tier detached signature cannot be verified: {exc}"]
        if observed != value:
            return ["higher-tier detached signature changed after signing"]
    return []


def _sshsig_verify(envelope: dict[str, Any], payload: bytes) -> tuple[bool, str]:
    try:
        allowed_raw = _safe_bytes(DEFAULT_ALLOWED_SIGNERS, maximum=1024 * 1024)
        if hashlib.sha256(allowed_raw).hexdigest() != PINNED_ALLOWED_SIGNERS_SHA256:
            raise HigherTierAuthorityError("allowed-signers trust root changed")
        signature = envelope.get("detached_signature")
        if not isinstance(signature, dict):
            raise HigherTierAuthorityError("detached signature identity is absent")
        signature_raw = _safe_bytes(pathlib.Path(signature["path"]), maximum=1024 * 1024)
        if hashlib.sha256(signature_raw).hexdigest() != signature.get("sha256") \
                or len(signature_raw) != signature.get("size_bytes"):
            raise HigherTierAuthorityError("detached signature changed before verification")
        with tempfile.TemporaryDirectory(prefix="hawking-higher-verify-") as directory:
            root = pathlib.Path(directory)
            root.chmod(0o700)
            allowed = root / "allowed_signers"
            detached = root / "signature.sshsig"
            for path, raw in ((allowed, allowed_raw), (detached, signature_raw)):
                descriptor = os.open(
                    path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0), 0o400,
                )
                try:
                    view = memoryview(raw)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise OSError("short write while preparing SSHSIG verification")
                        view = view[written:]
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            process = subprocess.run(
                [
                    str(SSH_KEYGEN), "-Y", "verify", "-f", str(allowed),
                    "-I", SIGNER_IDENTITY, "-n", SSHSIG_NAMESPACE,
                    "-s", str(detached),
                ],
                cwd=ROOT, input=payload, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, timeout=15, check=False, shell=False,
            )
    except (OSError, KeyError, subprocess.TimeoutExpired, HigherTierAuthorityError) as exc:
        return False, str(exc)
    detail_raw = process.stdout or process.stderr
    detail = detail_raw.decode("utf-8", "replace").strip() \
        if isinstance(detail_raw, bytes) else str(detail_raw).strip()
    return process.returncode == 0, detail[-1000:]


def validate_signed_manifest(
    envelope: Any, *, manifest: Mapping[str, Any], verify_files: bool = True,
    now_unix_ns: int | None = None,
    signature_verifier: SignatureVerifier | None = None,
) -> list[str]:
    expected = {
        "schema", "attestation", "signer_identity", "signature_namespace",
        "allowed_signers", "detached_signature", "envelope_sha256",
    }
    if not isinstance(envelope, dict) or set(envelope) != expected:
        return ["signed higher-tier manifest envelope is malformed"]
    errors = _canonical_hash_errors(envelope, "envelope_sha256", label="manifest_envelope")
    if envelope.get("schema") != ENVELOPE_SCHEMA:
        errors.append("signed higher-tier manifest envelope schema is invalid")
    if envelope.get("signer_identity") != SIGNER_IDENTITY:
        errors.append("signed higher-tier manifest signer is not source-pinned")
    if envelope.get("signature_namespace") != SSHSIG_NAMESPACE:
        errors.append("signed higher-tier manifest namespace is invalid")
    try:
        pinned_allowed = _allowed_signers_identity()
    except HigherTierAuthorityError as exc:
        errors.append(str(exc))
        pinned_allowed = None
    if envelope.get("allowed_signers") != pinned_allowed:
        errors.append("signed higher-tier manifest substituted its trust root")
    errors.extend(_signature_identity_errors(
        envelope.get("detached_signature"), verify_file=verify_files,
    ))
    attestation = envelope.get("attestation")
    errors.extend(validate_manifest_attestation(
        attestation, manifest=manifest, now_unix_ns=now_unix_ns,
    ))
    if not errors:
        verifier = _sshsig_verify if signature_verifier is None else signature_verifier
        try:
            ok, detail = verifier(envelope, canonical_bytes(attestation))
        except Exception as exc:
            ok, detail = False, str(exc)
        if not ok:
            errors.append(
                "higher-tier manifest SSHSIG verification failed"
                + (f": {detail}" if detail else "")
            )
    return list(dict.fromkeys(errors))


def build_sealed_manifest(
    *, manifest: Mapping[str, Any], signed_attestation: Mapping[str, Any],
    sealed_at_unix_ns: int | None = None, verify_files: bool = True,
    signature_verifier: SignatureVerifier | None = None,
) -> dict[str, Any]:
    sealed_at = time.time_ns() if sealed_at_unix_ns is None else sealed_at_unix_ns
    errors = validate_signed_manifest(
        signed_attestation, manifest=manifest, verify_files=verify_files,
        now_unix_ns=sealed_at, signature_verifier=signature_verifier,
    )
    if errors:
        raise HigherTierAuthorityError(
            "cannot seal invalid higher-tier manifest authority: " + "; ".join(errors)
        )
    attestation = signed_attestation["attestation"]
    if not _positive_int(sealed_at) \
            or not attestation["issued_at_unix_ns"] <= sealed_at \
            <= attestation["expires_at_unix_ns"]:
        raise HigherTierAuthorityError("higher-tier manifest seal time is outside validity")
    return _stamp({
        "schema": SEALED_MANIFEST_SCHEMA,
        "core_manifest": copy.deepcopy(dict(manifest)),
        "signed_manifest_attestation": copy.deepcopy(dict(signed_attestation)),
        "sealed_at_unix_ns": sealed_at,
        "runtime_defaults_changed": False,
        "execution_permitted": False,
    }, "sealed_manifest_sha256")


def validate_sealed_manifest(
    value: Any, *, verify_files: bool = True, now_unix_ns: int | None = None,
    signature_verifier: SignatureVerifier | None = None,
) -> list[str]:
    expected = {
        "schema", "core_manifest", "signed_manifest_attestation",
        "sealed_at_unix_ns", "runtime_defaults_changed", "execution_permitted",
        "sealed_manifest_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected:
        return [
            "operator-signed sealed higher-tier manifest is required; "
            "raw/self-hashed/reviewed manifests are not exact-plan authority"
        ]
    errors = _canonical_hash_errors(value, "sealed_manifest_sha256", label="sealed_manifest")
    if value.get("schema") != SEALED_MANIFEST_SCHEMA:
        errors.append("sealed higher-tier manifest schema is invalid")
    if value.get("runtime_defaults_changed") is not False \
            or value.get("execution_permitted") is not False:
        errors.append("sealed higher-tier manifest weakens default-off policy")
    manifest = value.get("core_manifest")
    envelope = value.get("signed_manifest_attestation")
    if not isinstance(manifest, dict) or not isinstance(envelope, dict):
        errors.append("sealed higher-tier manifest lacks exact core/signature")
        return list(dict.fromkeys(errors))
    now = time.time_ns() if now_unix_ns is None else now_unix_ns
    errors.extend(validate_signed_manifest(
        envelope, manifest=manifest, verify_files=verify_files,
        now_unix_ns=now, signature_verifier=signature_verifier,
    ))
    attestation = envelope.get("attestation")
    sealed_at = value.get("sealed_at_unix_ns")
    if not isinstance(attestation, dict) or not _positive_int(sealed_at) \
            or not _positive_int(attestation.get("issued_at_unix_ns")) \
            or not _positive_int(attestation.get("expires_at_unix_ns")) \
            or not attestation["issued_at_unix_ns"] <= sealed_at \
            <= attestation["expires_at_unix_ns"]:
        errors.append("higher-tier operator seal time is outside signed validity")
    return list(dict.fromkeys(errors))


def validate_and_unwrap(
    value: Any, *, verify_files: bool = True, now_unix_ns: int | None = None,
    signature_verifier: SignatureVerifier | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors = validate_sealed_manifest(
        value, verify_files=verify_files, now_unix_ns=now_unix_ns,
        signature_verifier=signature_verifier,
    )
    if errors:
        return None, errors
    return copy.deepcopy(value["core_manifest"]), []


def sign_manifest_attestation(
    attestation: Mapping[str, Any], *, manifest: Mapping[str, Any],
    private_key: pathlib.Path, detached_signature_output: pathlib.Path,
    envelope_output: pathlib.Path, now_unix_ns: int | None = None,
) -> dict[str, Any]:
    errors = validate_manifest_attestation(
        attestation, manifest=manifest, now_unix_ns=now_unix_ns,
    )
    if errors:
        raise HigherTierAuthorityError(
            "operator signer refused invalid higher-tier draft: " + "; ".join(errors)
        )
    private_stat = private_key.stat(follow_symlinks=False)
    resolved = private_key.resolve(strict=True)
    if private_key.is_symlink() or not stat.S_ISREG(private_stat.st_mode) \
            or private_stat.st_mode & 0o077:
        raise HigherTierAuthorityError(
            "operator key must be a non-symlink mode-0600 regular file"
        )
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        pass
    else:
        raise HigherTierAuthorityError("operator key must remain outside the repository")
    process = subprocess.run(
        [str(SSH_KEYGEN), "-y", "-f", str(private_key)], cwd=ROOT,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=15, check=False, shell=False,
    )
    parts = process.stdout.strip().split() if process.returncode == 0 else []
    if len(parts) < 2 or hashlib.sha256(" ".join(parts[:2]).encode("ascii")).hexdigest() \
            != PINNED_PUBLIC_KEY_BLOB_SHA256:
        raise HigherTierAuthorityError("operator key does not match source-pinned signer")
    payload = canonical_bytes(attestation)
    with tempfile.TemporaryDirectory(prefix="hawking-higher-sign-") as directory:
        message = pathlib.Path(directory) / "manifest.canonical.json"
        message.write_bytes(payload)
        signed = subprocess.run(
            [
                str(SSH_KEYGEN), "-Y", "sign", "-f", str(private_key),
                "-n", SSHSIG_NAMESPACE, str(message),
            ],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=30, check=False, shell=False,
        )
        signature = message.with_suffix(message.suffix + ".sig")
        if signed.returncode != 0 or not signature.is_file():
            raise HigherTierAuthorityError(
                f"higher-tier SSHSIG signing failed ({signed.returncode}): "
                f"{(signed.stderr or signed.stdout)[-500:]}"
            )
        raw = signature.read_bytes()
    _atomic_bytes(detached_signature_output, raw)
    envelope = _stamp({
        "schema": ENVELOPE_SCHEMA,
        "attestation": copy.deepcopy(dict(attestation)),
        "signer_identity": SIGNER_IDENTITY,
        "signature_namespace": SSHSIG_NAMESPACE,
        "allowed_signers": _allowed_signers_identity(),
        "detached_signature": physical_counter_attestation.file_identity(
            detached_signature_output,
        ),
    }, "envelope_sha256")
    _atomic_json(envelope_output, envelope)
    return envelope


def status() -> dict[str, Any]:
    try:
        _allowed_signers_identity()
        errors: list[str] = []
    except HigherTierAuthorityError as exc:
        errors = [str(exc)]
    return _stamp({
        "schema": "hawking.doctor_v5_higher_tier_authority_status.v1",
        "physical_execution_capability": False,
        "default_off": True,
        "signature_namespace": SSHSIG_NAMESPACE,
        "source_pinned_trust_root_valid": not errors,
        "errors": errors,
        "raw_manifest_exact_plan_authority": False,
        "runtime_defaults_changed": False,
    }, "status_sha256")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--status", action="store_true")
    actions.add_argument("--draft", type=pathlib.Path, metavar="MANIFEST")
    actions.add_argument("--sign", type=pathlib.Path, metavar="ATTESTATION")
    actions.add_argument("--seal", type=pathlib.Path, metavar="ENVELOPE")
    actions.add_argument("--verify", type=pathlib.Path, metavar="SEALED")
    parser.add_argument("--manifest", type=pathlib.Path)
    parser.add_argument("--private-key", type=pathlib.Path, default=DEFAULT_PRIVATE_KEY)
    parser.add_argument("--signature-output", type=pathlib.Path)
    parser.add_argument("--envelope-output", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--valid-seconds", type=int, default=DEFAULT_VALIDITY_SECONDS)
    args = parser.parse_args(argv)
    try:
        if args.status:
            print(json.dumps(status(), indent=2, sort_keys=True))
            return 0
        if args.draft is not None:
            if args.output is None:
                parser.error("--draft requires --output")
            manifest = _safe_json(args.draft)
            draft = build_manifest_attestation(manifest, valid_seconds=args.valid_seconds)
            _atomic_json(args.output, draft)
            return 0
        if args.sign is not None:
            if args.manifest is None or args.signature_output is None \
                    or args.envelope_output is None:
                parser.error("--sign requires --manifest, --signature-output, --envelope-output")
            sign_manifest_attestation(
                _safe_json(args.sign), manifest=_safe_json(args.manifest),
                private_key=args.private_key,
                detached_signature_output=args.signature_output,
                envelope_output=args.envelope_output,
            )
            return 0
        if args.seal is not None:
            if args.manifest is None or args.output is None:
                parser.error("--seal requires --manifest and --output")
            sealed = build_sealed_manifest(
                manifest=_safe_json(args.manifest),
                signed_attestation=_safe_json(args.seal),
            )
            _atomic_json(args.output, sealed)
            return 0
        if args.verify is not None:
            errors = validate_sealed_manifest(_safe_json(args.verify))
            print(json.dumps({"valid": not errors, "errors": errors}, indent=2, sort_keys=True))
            return 0 if not errors else 1
    except (
        OSError, subprocess.TimeoutExpired, HigherTierAuthorityError,
        TypeError, ValueError, KeyError,
    ) as exc:
        print(f"blocked: {exc}", file=sys.stderr)
        return 75
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
