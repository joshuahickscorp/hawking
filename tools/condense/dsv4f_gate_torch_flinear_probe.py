#!/usr/bin/env python3
"""Bounded DeepSeek-V4 P0 source-CPU Gate evidence producer.

This probe accepts one explicitly bounded, unsealed P0 Gate-input trace and a
sealed full-stream Gravity artifact.  It verifies the trace's provenance,
reconstructs only the source Gate and hash-route table through their
content-addressed chunks, then measures two CPU-only paths:

* upstream-shaped ``torch.nn.functional.linear(x.float(), weight.float())``;
* an independent, row-major, sequential IEEE-754 binary32 reduction with a
  rounded product and rounded addition at every column.

The output retains hashes and aggregate metrics only.  It never serializes raw
weights, activations, logits, route weights, or source parent files.  It is
not Numeric Parity V2.1 evidence, a runtime result, a GPU result, a token
generation result, or a TPS result.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import stat
import struct
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


TRACE_SCHEMA = "hawking.gravity.deepseek_v4.p7_layer0_position0_gate_input_trace.v1"
TRACE_STATUS = "UNSEALED_POST_COMPLETION_BOS_FFN_NORM_GATE_INPUT_TRACE_NON_RECEIPT"
MANIFEST_SCHEMA = "hawking.gravity.deepseek_v4.full_stream.v1"
MANIFEST_STATUS = "FULL_MODEL_STREAMED_SEALED_NOT_RUNTIME_READY"
EVIDENCE_SCHEMA = "hawking.gravity.deepseek_v4.gate_torch_flinear_probe.v1"
GATE_NAME = "layers.0.ffn.gate.weight"
TID2EID_NAME = "layers.0.ffn.gate.tid2eid"
MODEL_PY = "inference/model.py"
HIDDEN_SIZE = 4096
ROUTED_EXPERTS = 256
ACTIVATED_EXPERTS = 6
TOKEN_ID = 0
ROUTE_SCALE = 1.5
BF16_ROW_BYTES = HIDDEN_SIZE * 2
F32_BYTES = struct.Struct("<f")
F32_BITS = struct.Struct("<I")
I64 = struct.Struct("<q")


class ProbeError(ValueError):
    """The requested source-CPU measurement is not safely bound."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProbeError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProbeError(f"{label} must be an array")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProbeError(f"{label} must be a non-empty string")
    return value


def _digest(value: object, label: str) -> str:
    rendered = _string(value, label)
    if not _is_sha256(rendered):
        raise ProbeError(f"{label} must be a lowercase SHA-256 digest")
    return rendered


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ProbeError(f"{label} must be an integer >= {minimum}")
    return value


def _absolute_existing_regular(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ProbeError(f"{label} must be an absolute path")
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ProbeError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProbeError(f"{label} must be a regular non-symlink file")
    return path.resolve()


def _absolute_existing_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise ProbeError(f"{label} must be an absolute path")
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ProbeError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ProbeError(f"{label} must be a directory that is not a symlink")
    return path.resolve()


def _absolute_new_output(path: Path) -> Path:
    if not path.is_absolute():
        raise ProbeError("--out must be an absolute path")
    try:
        os.lstat(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ProbeError(f"cannot inspect --out: {exc}") from exc
    else:
        raise ProbeError(f"refusing to overwrite existing evidence output {path}")
    if not path.name or path.name in {".", ".."}:
        raise ProbeError("--out must name a new JSON file")
    return path


def _regular_child(root: Path, relative: str, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or any(
        component in {"", ".", ".."} for component in candidate.parts
    ):
        raise ProbeError(f"{label} has an unsafe relative path {relative!r}")
    path = root.joinpath(*candidate.parts)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ProbeError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProbeError(f"{label} must be a regular non-symlink file")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ProbeError(f"{label} escapes the artifact root") from exc
    return resolved


def _directory_child(root: Path, relative: str, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts or any(
        component in {"", ".", ".."} for component in candidate.parts
    ):
        raise ProbeError(f"{label} has an unsafe relative path {relative!r}")
    path = root.joinpath(*candidate.parts)
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise ProbeError(f"cannot stat {label}: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ProbeError(f"{label} must be a directory that is not a symlink")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ProbeError(f"{label} escapes the artifact root") from exc
    return resolved


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ProbeError(f"{label} root must be an object")
    return value, raw


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _verify_sealed_json(value: Mapping[str, Any], label: str) -> str:
    recorded = _digest(value.get("seal_sha256"), f"{label}.seal_sha256")
    unsigned = dict(value)
    unsigned.pop("seal_sha256", None)
    observed = _sha256(_canonical_json(unsigned))
    if observed != recorded:
        raise ProbeError(
            f"{label} canonical seal mismatch: recorded={recorded} observed={observed}"
        )
    return recorded


def _trace_field(root: Mapping[str, Any], key: str, label: str) -> Mapping[str, Any]:
    return _mapping(root.get(key), label)


def _validate_trace(trace: Mapping[str, Any], trace_raw: bytes) -> dict[str, Any]:
    if trace.get("schema") != TRACE_SCHEMA or trace.get("status") != TRACE_STATUS:
        raise ProbeError("trace is not the expected P0 v1 Gate-input trace")
    if trace.get("unsealed") is not True or trace.get("receipt_promoted") is not False:
        raise ProbeError("trace must remain explicitly unsealed and non-promoted")
    if trace.get("is_receipt") is not False:
        raise ProbeError("trace must remain explicitly classified as non-receipt")

    binding = _trace_field(trace, "trace_binding", "trace.trace_binding")
    if (
        _integer(binding.get("layer"), "trace_binding.layer") != 0
        or _integer(binding.get("token_id"), "trace_binding.token_id") != TOKEN_ID
        or _integer(binding.get("token_position"), "trace_binding.token_position") != 0
        or binding.get("post_completion_readback_only") is not True
        or binding.get("trace_does_not_feed_graph") is not True
        or binding.get("trace_does_not_modify_graph_counters") is not True
    ):
        raise ProbeError("trace is not an admitted post-completion P0 BOS observation")

    artifact = _trace_field(trace, "artifact", "trace.artifact")
    source = _trace_field(trace, "model_source", "trace.model_source")
    bindings = _trace_field(source, "p6_gate_route_bindings", "trace.p6_gate_route_bindings")
    if _string(bindings.get("gate_weight_name"), "trace gate_weight_name") != GATE_NAME:
        raise ProbeError("trace does not bind the layer-0 Gate tensor")
    if _string(bindings.get("tid2eid_name"), "trace tid2eid_name") != TID2EID_NAME:
        raise ProbeError("trace does not bind the layer-0 tid2eid table")
    expected_ids = _list(
        bindings.get("selected_expert_ids_top_slot_order"), "trace selected expert IDs"
    )
    if len(expected_ids) != ACTIVATED_EXPERTS:
        raise ProbeError("trace must bind exactly six route IDs")
    selected_ids = [
        _integer(value, f"trace selected expert ID {index}")
        for index, value in enumerate(expected_ids)
    ]
    if any(identifier >= ROUTED_EXPERTS for identifier in selected_ids):
        raise ProbeError("trace contains an out-of-range selected expert ID")

    payload = _trace_field(trace, "raw_payload", "trace.raw_payload")
    if (
        _string(payload.get("name"), "trace payload name")
        != "p7_ffn_norm_bf16_bos_gate_input"
        or _list(payload.get("shape"), "trace payload shape") != [HIDDEN_SIZE]
        or _string(payload.get("dtype"), "trace payload dtype") != "BF16"
        or _string(payload.get("byte_order"), "trace payload byte_order") != "little_endian"
        or _integer(payload.get("element_count"), "trace payload element_count") != HIDDEN_SIZE
        or _integer(payload.get("byte_count"), "trace payload byte_count") != BF16_ROW_BYTES
        or _string(payload.get("encoding"), "trace payload encoding")
        != "lowercase_hex_raw_bf16_le"
    ):
        raise ProbeError("trace payload is not exactly one BOS BF16[4096] Gate input")
    payload_hex = _string(payload.get("data"), "trace payload data")
    if len(payload_hex) != BF16_ROW_BYTES * 2 or any(
        character not in "0123456789abcdef" for character in payload_hex
    ):
        raise ProbeError("trace payload must be lowercase hex for exactly 8192 bytes")
    try:
        payload_bytes = bytes.fromhex(payload_hex)
    except ValueError as exc:
        raise ProbeError("trace payload hex decode failed") from exc
    payload_sha256 = _digest(payload.get("sha256"), "trace payload sha256")
    if _sha256(payload_bytes) != payload_sha256:
        raise ProbeError("trace raw payload digest mismatch")

    io_hashes = _trace_field(trace, "input_output_sha256", "trace.input_output_sha256")
    for field in (
        "p7_producer_output_ffn_norm_bf16_le",
        "p6_gate_input_ffn_norm_bf16_le",
    ):
        if _digest(io_hashes.get(field), f"trace.{field}") != payload_sha256:
            raise ProbeError(f"trace {field} does not bind the sole raw Gate input payload")
    gate_output_sha256 = _digest(
        io_hashes.get("p6_gate_output_logits_f32_le"), "trace Gate output SHA-256"
    )

    privacy = _trace_field(trace, "privacy_and_storage_bound", "trace.privacy_and_storage_bound")
    if (
        _integer(privacy.get("raw_payload_count"), "trace raw_payload_count") != 1
        or _integer(privacy.get("raw_payload_actual_bytes"), "trace raw_payload_actual_bytes")
        != BF16_ROW_BYTES
        or _integer(privacy.get("raw_source_weight_payloads"), "trace raw_source_weight_payloads")
        != 0
        or _integer(
            privacy.get("raw_other_activation_payloads"), "trace raw_other_activation_payloads"
        )
        != 0
        or _integer(privacy.get("raw_gate_output_payloads"), "trace raw_gate_output_payloads")
        != 0
    ):
        raise ProbeError("trace privacy/storage bound differs from one-payload P0 contract")

    return {
        "trace_file_sha256": _sha256(trace_raw),
        "artifact": artifact,
        "source": source,
        "bindings": bindings,
        "payload_bytes": payload_bytes,
        "payload_sha256": payload_sha256,
        "gate_output_sha256": gate_output_sha256,
        "selected_ids": selected_ids,
        "trace_executable": _trace_field(trace, "executable", "trace.executable"),
        "trace_binding": binding,
    }


def _validate_manifest(
    artifact_root: Path, trace_binding: Mapping[str, Any]
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    manifest_path = _regular_child(artifact_root, "manifest.json", "artifact manifest")
    manifest, manifest_raw = _read_json(manifest_path, "artifact manifest")
    manifest_seal = _verify_sealed_json(manifest, "artifact manifest")
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("status") != MANIFEST_STATUS:
        raise ProbeError("artifact manifest is not the admitted full DeepSeek-V4 stream")
    storage = _mapping(manifest.get("storage"), "manifest storage")
    if storage.get("source_parent_retained") is not False:
        raise ProbeError("artifact manifest does not preserve source-parent eviction")
    if trace_binding.get("source_parent_retained") is not False:
        raise ProbeError("trace does not preserve source-parent eviction")
    if _digest(trace_binding.get("manifest_seal_sha256"), "trace artifact manifest seal") != manifest_seal:
        raise ProbeError("trace and artifact manifest seals differ")
    manifest_file_sha256 = _sha256(manifest_raw)
    if (
        _digest(trace_binding.get("manifest_file_sha256"), "trace artifact manifest file hash")
        != manifest_file_sha256
    ):
        raise ProbeError("trace and physical artifact manifest bytes differ")
    restart_binding = _digest(
        trace_binding.get("restart_receipt_seal_sha256"), "trace restart receipt seal"
    )
    restart_path = _regular_child(artifact_root, "restart-receipt.json", "artifact restart receipt")
    restart, _ = _read_json(restart_path, "artifact restart receipt")
    restart_seal = _verify_sealed_json(restart, "artifact restart receipt")
    if restart_seal != restart_binding:
        raise ProbeError("trace and artifact restart receipt seals differ")
    if restart.get("source_parent_retained") is not False:
        raise ProbeError("artifact restart receipt does not preserve source-parent eviction")
    return manifest, {
        "path": str(manifest_path),
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_seal_sha256": manifest_seal,
        "restart_receipt_seal_sha256": restart_seal,
    }


def _validate_model_source(
    artifact_root: Path, manifest: Mapping[str, Any], trace_source: Mapping[str, Any]
) -> dict[str, Any]:
    source = _mapping(manifest.get("source"), "manifest.source")
    trace_repository = _string(trace_source.get("repository"), "trace source repository")
    trace_revision = _string(trace_source.get("revision"), "trace source revision")
    if source.get("repository") != trace_repository or source.get("revision") != trace_revision:
        raise ProbeError("trace source identity differs from the sealed artifact manifest")
    metadata_assets = _mapping(source.get("metadata_assets"), "manifest source metadata_assets")
    model_asset = _mapping(metadata_assets.get(MODEL_PY), "manifest inference/model.py asset")
    if _string(model_asset.get("path"), "manifest model.py path") != MODEL_PY:
        raise ProbeError("manifest model.py path is not canonical")
    manifest_model_sha256 = _digest(model_asset.get("sha256"), "manifest model.py SHA-256")
    trace_assets = _mapping(
        trace_source.get("metadata_asset_sha256"), "trace source metadata_asset_sha256"
    )
    if _digest(trace_assets.get(MODEL_PY), "trace model.py SHA-256") != manifest_model_sha256:
        raise ProbeError("trace model.py binding differs from manifest")
    metadata_root = _directory_child(artifact_root, "metadata", "artifact metadata root")
    model_path = _regular_child(metadata_root, MODEL_PY, "artifact model.py")
    observed_model_sha256 = _sha256_file(model_path)
    if observed_model_sha256 != manifest_model_sha256:
        raise ProbeError("artifact model.py bytes differ from its sealed hash")
    return {
        "repository": trace_repository,
        "revision": trace_revision,
        "model_py_path": str(model_path),
        "model_py_sha256": observed_model_sha256,
        "model_py_bytes": model_path.stat().st_size,
    }


def _validate_tensor_descriptor(
    manifest: Mapping[str, Any], name: str, dtype: str, shape: list[int], bytes_: int
) -> Mapping[str, Any]:
    tensors = _mapping(manifest.get("tensors"), "manifest tensors")
    descriptor = _mapping(tensors.get(name), f"manifest tensor {name}")
    if (
        _string(descriptor.get("name"), f"{name} descriptor name") != name
        or _string(descriptor.get("dtype"), f"{name} dtype") != dtype
        or _list(descriptor.get("shape"), f"{name} shape") != shape
        or _integer(descriptor.get("bytes"), f"{name} bytes") != bytes_
    ):
        raise ProbeError(f"{name} has unexpected source descriptor geometry")
    return descriptor


def _read_verified_tensor(
    artifact_root: Path, descriptor: Mapping[str, Any], label: str
) -> tuple[bytes, dict[str, Any]]:
    expected_bytes = _integer(descriptor.get("bytes"), f"{label}.bytes", minimum=1)
    segments = _list(descriptor.get("segments"), f"{label}.segments")
    if not segments:
        raise ProbeError(f"{label} has no source segments")
    ordered = sorted(
        (_mapping(segment, f"{label} segment") for segment in segments),
        key=lambda segment: _integer(segment.get("tensor_start"), f"{label} tensor_start"),
    )
    chunks_root = _directory_child(artifact_root, "chunks", "artifact chunk root")
    raw = bytearray(expected_bytes)
    cursor = 0
    chunk_hashes: list[str] = []
    verified_bytes = 0
    for ordinal, segment in enumerate(ordered):
        start = _integer(segment.get("tensor_start"), f"{label} segment {ordinal} tensor_start")
        end = _integer(segment.get("tensor_end"), f"{label} segment {ordinal} tensor_end")
        size = _integer(segment.get("bytes"), f"{label} segment {ordinal} bytes", minimum=1)
        digest = _digest(segment.get("sha256"), f"{label} segment {ordinal} SHA-256")
        relative = _string(segment.get("chunk_relpath"), f"{label} segment {ordinal} path")
        if (
            start != cursor
            or end <= start
            or end - start != size
            or end > expected_bytes
            or relative != f"chunks/{digest[:2]}/{digest}"
        ):
            raise ProbeError(f"{label} has a non-canonical or non-contiguous chunk segment")
        prefix = _directory_child(
            chunks_root, digest[:2], f"{label} chunk prefix {ordinal}"
        )
        chunk = _regular_child(prefix, digest, f"{label} chunk {ordinal}")
        chunk_bytes = chunk.read_bytes()
        if len(chunk_bytes) != size or _sha256(chunk_bytes) != digest:
            raise ProbeError(f"{label} chunk {ordinal} bytes/hash mismatch")
        raw[start:end] = chunk_bytes
        cursor = end
        verified_bytes += size
        chunk_hashes.append(digest)
    if cursor != expected_bytes:
        raise ProbeError(f"{label} segments do not cover the complete tensor")
    return bytes(raw), {
        "name": _string(descriptor.get("name"), f"{label} name"),
        "dtype": _string(descriptor.get("dtype"), f"{label} dtype"),
        "shape": _list(descriptor.get("shape"), f"{label} shape"),
        "bytes": expected_bytes,
        "verified_chunk_count": len(chunk_hashes),
        "verified_chunk_bytes": verified_bytes,
        "verified_chunk_sha256": chunk_hashes,
        "logical_tensor_sha256": _sha256(bytes(raw)),
    }


def _verify_gate_and_route_bindings(
    artifact_root: Path, manifest: Mapping[str, Any], trace: Mapping[str, Any]
) -> tuple[bytes, list[int], dict[str, Any]]:
    bindings = _mapping(trace["bindings"], "trace gate bindings")
    gate_descriptor = _validate_tensor_descriptor(
        manifest, GATE_NAME, "BF16", [ROUTED_EXPERTS, HIDDEN_SIZE], ROUTED_EXPERTS * BF16_ROW_BYTES
    )
    gate_bytes, gate_binding = _read_verified_tensor(artifact_root, gate_descriptor, "source Gate")
    expected_gate_sha256 = _digest(bindings.get("gate_weight_sha256"), "trace Gate weight SHA-256")
    if gate_binding["logical_tensor_sha256"] != expected_gate_sha256:
        raise ProbeError("reconstructed source Gate bytes differ from trace Gate binding")

    tid_bytes = 129_280 * ACTIVATED_EXPERTS * I64.size
    tid_descriptor = _validate_tensor_descriptor(
        manifest, TID2EID_NAME, "I64", [129_280, ACTIVATED_EXPERTS], tid_bytes
    )
    tid_raw, tid_binding = _read_verified_tensor(artifact_root, tid_descriptor, "source tid2eid")
    expected_tid_sha256 = _digest(bindings.get("tid2eid_sha256"), "trace tid2eid SHA-256")
    if tid_binding["logical_tensor_sha256"] != expected_tid_sha256:
        raise ProbeError("reconstructed source tid2eid bytes differ from trace binding")
    offset = TOKEN_ID * ACTIVATED_EXPERTS * I64.size
    ids = [I64.unpack_from(tid_raw, offset + slot * I64.size)[0] for slot in range(ACTIVATED_EXPERTS)]
    if any(identifier < 0 or identifier >= ROUTED_EXPERTS for identifier in ids):
        raise ProbeError("source tid2eid BOS row contains an out-of-range expert ID")
    source_ids = [int(identifier) for identifier in ids]
    if source_ids != trace["selected_ids"]:
        raise ProbeError("source tid2eid BOS row differs from trace fixed route IDs")
    return gate_bytes, source_ids, {"gate": gate_binding, "tid2eid": tid_binding}


def _bf16_to_f32(bits: int) -> float:
    return F32_BYTES.unpack(F32_BITS.pack(bits << 16))[0]


def _decode_bf16_row(raw: bytes, label: str) -> list[float]:
    if len(raw) != BF16_ROW_BYTES:
        raise ProbeError(f"{label} is not exactly BF16[4096]")
    words = array.array("H")
    words.frombytes(raw)
    if sys.byteorder != "little":
        words.byteswap()
    values = [_bf16_to_f32(int(word)) for word in words]
    if len(values) != HIDDEN_SIZE or not all(math.isfinite(value) for value in values):
        raise ProbeError(f"{label} has invalid BF16 values")
    return values


def _round_f32(value: float) -> float:
    return F32_BYTES.unpack(F32_BYTES.pack(value))[0]


def _sequential_f32_gate(gate_bytes: bytes, input_values: Sequence[float]) -> list[float]:
    """Run a deterministic source-derived serial F32 diagnostic reduction.

    Upstream ``Gate.forward`` declares framework ``F.linear`` only; it does
    not declare this instruction/reduction order.  This loop is therefore an
    independent diagnostic comparator, not an assertion of source-runtime
    arithmetic.
    """

    if len(gate_bytes) != ROUTED_EXPERTS * BF16_ROW_BYTES:
        raise ProbeError("source Gate raw bytes are not BF16[256,4096]")
    if len(input_values) != HIDDEN_SIZE:
        raise ProbeError("sequential Gate input is not 4096 elements")
    weights = array.array("H")
    weights.frombytes(gate_bytes)
    if sys.byteorder != "little":
        weights.byteswap()
    if len(weights) != ROUTED_EXPERTS * HIDDEN_SIZE:
        raise ProbeError("source Gate BF16 word count is invalid")
    output: list[float] = []
    for row in range(ROUTED_EXPERTS):
        accumulator = 0.0
        start = row * HIDDEN_SIZE
        for column, activation in enumerate(input_values):
            weight = _bf16_to_f32(int(weights[start + column]))
            if not math.isfinite(weight):
                raise ProbeError("source Gate contains a non-finite BF16 weight")
            product = _round_f32(activation * weight)
            accumulator = _round_f32(accumulator + product)
        if not math.isfinite(accumulator):
            raise ProbeError("sequential Gate produced a non-finite logit")
        output.append(accumulator)
    return output


def _f32_values_sha256(values: Sequence[float]) -> str:
    return _sha256(b"".join(F32_BYTES.pack(_round_f32(float(value))) for value in values))


def _f32_bits(value: float) -> int:
    return F32_BITS.unpack(F32_BYTES.pack(_round_f32(value)))[0]


def _vector_metrics(reference: Sequence[float], candidate: Sequence[float]) -> dict[str, Any]:
    if len(reference) != len(candidate) or not reference:
        raise ProbeError("metric inputs must be equally sized and non-empty")
    max_abs = 0.0
    max_relative = 0.0
    numerator = 0.0
    denominator = 0.0
    mismatch = 0
    for expected, observed in zip(reference, candidate):
        if not math.isfinite(expected) or not math.isfinite(observed):
            raise ProbeError("metric input contains a non-finite F32 value")
        expected_f32 = _round_f32(float(expected))
        observed_f32 = _round_f32(float(observed))
        if _f32_bits(expected_f32) != _f32_bits(observed_f32):
            mismatch += 1
        absolute = abs(float(observed_f32) - float(expected_f32))
        max_abs = max(max_abs, absolute)
        max_relative = max(max_relative, absolute / max(abs(float(expected_f32)), 1.0e-12))
        numerator += absolute * absolute
        denominator += float(expected_f32) * float(expected_f32)
    relative_l2 = math.sqrt(numerator) / math.sqrt(denominator) if denominator else (0.0 if numerator == 0.0 else math.inf)
    if not math.isfinite(relative_l2):
        raise ProbeError("relative L2 is not finite")
    return {
        "elements": len(reference),
        "f32_bit_exact": mismatch == 0,
        "f32_bit_mismatch_elements": mismatch,
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "max_relative_reference_floor_1e-12": max_relative,
    }


def _torch_measurement(
    gate_bytes: bytes, input_bytes: bytes, selected_ids: Sequence[int]
) -> tuple[list[float], list[float], list[float], list[float], dict[str, Any]]:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise ProbeError(
            "PyTorch is required but not importable; do not install packages from this probe"
        ) from exc

    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise ProbeError(f"cannot enforce one PyTorch inter-op CPU thread: {exc}") from exc
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ProbeError("PyTorch did not accept the required one-thread CPU configuration")

    input_words = torch.frombuffer(bytearray(input_bytes), dtype=torch.uint16).clone()
    gate_words = torch.frombuffer(bytearray(gate_bytes), dtype=torch.uint16).clone()
    if input_words.numel() != HIDDEN_SIZE or gate_words.numel() != ROUTED_EXPERTS * HIDDEN_SIZE:
        raise ProbeError("Torch BF16 storage geometry does not match the source Gate contract")
    input_tensor = input_words.view(torch.bfloat16).to(dtype=torch.float32, device="cpu")
    gate_tensor = gate_words.view(torch.bfloat16).reshape(ROUTED_EXPERTS, HIDDEN_SIZE).to(
        dtype=torch.float32, device="cpu"
    )
    if input_tensor.device.type != "cpu" or gate_tensor.device.type != "cpu":
        raise ProbeError("Torch Gate probe unexpectedly selected a non-CPU device")
    with torch.no_grad():
        torch_logits_tensor = functional.linear(input_tensor.float(), gate_tensor.float()).contiguous()
    torch_logits = [float(value) for value in torch_logits_tensor.tolist()]
    if len(torch_logits) != ROUTED_EXPERTS or not all(math.isfinite(value) for value in torch_logits):
        raise ProbeError("Torch F.linear returned invalid Gate logits")

    sequential_logits = _sequential_f32_gate(gate_bytes, _decode_bf16_row(input_bytes, "trace input"))
    sequential_tensor = torch.tensor(sequential_logits, dtype=torch.float32, device="cpu")
    ids_tensor = torch.tensor(list(selected_ids), dtype=torch.int64, device="cpu")
    with torch.no_grad():
        source_scores = torch.sqrt(functional.softplus(torch_logits_tensor, beta=1, threshold=20))
        sequential_scores = torch.sqrt(functional.softplus(sequential_tensor, beta=1, threshold=20))
        source_selected = source_scores.index_select(0, ids_tensor)
        sequential_selected = sequential_scores.index_select(0, ids_tensor)
        source_sum = source_selected.sum(dtype=torch.float32)
        sequential_sum = sequential_selected.sum(dtype=torch.float32)
        if not bool(torch.isfinite(source_sum)) or not bool(torch.isfinite(sequential_sum)):
            raise ProbeError("source route-score normalization is non-finite")
        if float(source_sum) <= 0.0 or float(sequential_sum) <= 0.0:
            raise ProbeError("source route-score normalization is not positive")
        source_weights_tensor = (source_selected / source_sum) * ROUTE_SCALE
        sequential_weights_tensor = (sequential_selected / sequential_sum) * ROUTE_SCALE
    source_weights = [float(value) for value in source_weights_tensor.tolist()]
    sequential_weights = [float(value) for value in sequential_weights_tensor.tolist()]
    if not all(math.isfinite(value) for value in source_weights + sequential_weights):
        raise ProbeError("source route weights are non-finite")
    runtime = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "torch_git_version": getattr(torch.version, "git_version", None),
        "execution_device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_cuda_invoked": False,
        "torch_f_linear_shape": [ROUTED_EXPERTS],
        "torch_f_linear_contract": "torch.nn.functional.linear(x.float(), weight.float()) on CPU with the bound 1D P0 input",
        "route_score_contract": "torch.sqrt(torch.nn.functional.softplus(logits, beta=1, threshold=20)); fixed tid2eid gather; F32 sum; divide; multiply by route_scale=1.5",
    }
    return torch_logits, sequential_logits, source_weights, sequential_weights, runtime


def _script_binding() -> dict[str, Any]:
    path = _absolute_existing_regular(Path(__file__).resolve(), "probe script")
    return {"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}


def _publish_new_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably publish only a new final pathname, never a partial overwrite."""

    if not path.is_absolute():
        raise ProbeError("evidence output path lost its absolute binding")
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = _absolute_existing_directory(path.parent, "evidence output directory")
    final = parent / path.name
    try:
        os.lstat(final)
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ProbeError(f"cannot inspect final evidence path: {exc}") from exc
    else:
        raise ProbeError(f"refusing to overwrite existing evidence output {final}")
    encoded = _canonical_json(value) + b"\n"
    temporary = parent / f".{path.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, final)
        except OSError as exc:
            raise ProbeError(f"atomic create-new evidence publish failed: {exc}") from exc
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def build_evidence(trace_path: Path, artifact_path: Path) -> dict[str, Any]:
    trace_path = _absolute_existing_regular(trace_path, "--trace")
    artifact_root = _absolute_existing_directory(artifact_path, "--artifact")
    trace_document, trace_raw = _read_json(trace_path, "P0 Gate-input trace")
    trace = _validate_trace(trace_document, trace_raw)
    manifest, artifact = _validate_manifest(artifact_root, _mapping(trace["artifact"], "trace artifact"))
    source = _validate_model_source(artifact_root, manifest, _mapping(trace["source"], "trace source"))
    gate_bytes, source_ids, source_tensors = _verify_gate_and_route_bindings(
        artifact_root, manifest, trace
    )

    torch_logits, sequential_logits, source_weights, sequential_weights, runtime = _torch_measurement(
        gate_bytes, trace["payload_bytes"], source_ids
    )
    torch_logits_sha256 = _f32_values_sha256(torch_logits)
    sequential_logits_sha256 = _f32_values_sha256(sequential_logits)
    metal_gate_sha_matches = sequential_logits_sha256 == trace["gate_output_sha256"]
    torch_vs_sequential = _vector_metrics(torch_logits, sequential_logits)
    route_weight_metrics = _vector_metrics(source_weights, sequential_weights)

    status = (
        "SOURCE_CPU_MEASURED_TORCH_F_LINEAR_AND_SOURCE_DERIVED_SERIAL_F32_METAL_LOGIT_SHA_MATCH_TORCH_VS_SERIAL_NOT_BIT_EXACT_NOT_NUMERIC_PARITY_RUNTIME_OR_TPS"
        if metal_gate_sha_matches
        else "SOURCE_CPU_MEASURED_TORCH_F_LINEAR_AND_SOURCE_DERIVED_SERIAL_F32_METAL_LOGIT_SHA_MISMATCH_TORCH_VS_SERIAL_NOT_BIT_EXACT_NOT_NUMERIC_PARITY_RUNTIME_OR_TPS"
    )
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "status": status,
        "unsealed": True,
        "receipt_promoted": False,
        "measurement_scope": {
            "source_cpu_measurement": True,
            "numeric_parity_v2_1": "not_evaluated",
            "runtime": "not_evaluated",
            "gpu": "not_used",
            "hcli": "not_evaluated",
            "token_generation": "not_evaluated",
            "base_true_tps": "not_evaluated",
            "claim_boundary": "This compares an upstream-shaped CPU Torch Gate matvec with an independent source-derived serial binary32 diagnostic reduction for one bounded P0 BOS input. Upstream Gate.forward specifies framework F.linear, not the serial loop's instruction or reduction order. It is not a Numeric Parity V2.1 admission or runtime/TPS/capability claim.",
        },
        "trace_binding": {
            "path": str(trace_path),
            "file_sha256": trace["trace_file_sha256"],
            "schema": TRACE_SCHEMA,
            "status": TRACE_STATUS,
            "layer": trace["trace_binding"]["layer"],
            "token_id": trace["trace_binding"]["token_id"],
            "token_position": trace["trace_binding"]["token_position"],
            "raw_gate_input_payload_sha256": trace["payload_sha256"],
            "raw_gate_input_payload_geometry": {"dtype": "BF16", "shape": [HIDDEN_SIZE], "bytes": BF16_ROW_BYTES},
            "recorded_metal_gate_logits_f32_le_sha256": trace["gate_output_sha256"],
            "trace_producer_executable_binding": trace["trace_executable"],
        },
        "artifact_binding": artifact,
        "source_binding": source,
        "source_tensor_bindings": source_tensors,
        "route_binding": {
            "selection_method": "source tid2eid[token_id] fixed row; scores remain input-dependent",
            "token_id": TOKEN_ID,
            "source_fixed_ids_top_slot_order": source_ids,
            "trace_fixed_ids_match_source": True,
            "route_scale": ROUTE_SCALE,
        },
        "cpu_runtime_binding": runtime,
        "torch_vs_sequential_gate_logits": {
            "torch_f_linear_logits_f32_le_sha256": torch_logits_sha256,
            "sequential_serial_f32_logits_f32_le_sha256": sequential_logits_sha256,
            "sequential_matches_recorded_metal_gate_sha256": metal_gate_sha_matches,
            "metrics_reference": "Torch F.linear output",
            "metrics_candidate": "independent source-derived row-major sequential binary32 diagnostic reduction",
            "metrics": torch_vs_sequential,
            "sequential_reduction_contract": "For each row and columns 0..4095: decode BF16 to binary32; round binary32(activation * weight); then round binary32(accumulator + product). No FMA, vector reduction, or Torch reduction is used in this source-derived diagnostic path; upstream Gate.forward declares framework F.linear rather than this serial instruction/reduction order.",
        },
        "source_vs_sequential_route_weights": {
            "source_route_weights_f32_le_sha256": _f32_values_sha256(source_weights),
            "sequential_route_weights_f32_le_sha256": _f32_values_sha256(sequential_weights),
            "metrics_reference": "Torch F.linear logits through upstream-shaped sqrt-softplus/fixed-id normalization",
            "metrics_candidate": "sequential binary32 logits through the same CPU source score/normalization operators",
            "metrics": route_weight_metrics,
        },
        "storage_policy": {
            "raw_weight_payloads_in_output": 0,
            "raw_activation_payloads_in_output": 0,
            "raw_logit_payloads_in_output": 0,
            "raw_route_weight_payloads_in_output": 0,
            "retained_values": "only source/artifact/runtime hashes, fixed route IDs, aggregate metrics, and F32 vector hashes",
        },
        "producer": _script_binding(),
    }
    canonical_sha256 = _sha256(_canonical_json(evidence))
    evidence["canonical_sha256"] = canonical_sha256
    return evidence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True, help="absolute P0 Gate-input trace JSON")
    parser.add_argument("--artifact", type=Path, required=True, help="absolute full Gravity artifact directory")
    parser.add_argument("--out", type=Path, required=True, help="absolute new canonical evidence JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        out = _absolute_new_output(args.out)
        evidence = build_evidence(args.trace, args.artifact)
        _publish_new_canonical_json(out, evidence)
    except ProbeError as exc:
        print(f"dsv4f_gate_torch_flinear_probe: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "out": str(out),
                "canonical_sha256": evidence["canonical_sha256"],
                "sequential_matches_recorded_metal_gate_sha256": evidence[
                    "torch_vs_sequential_gate_logits"
                ]["sequential_matches_recorded_metal_gate_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
