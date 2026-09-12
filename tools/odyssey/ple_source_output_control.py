#!/usr/bin/env python3
"""Evaluate one source-bound Qwen4-Exp PLE injection control.

Flash's PLE is an executable organ, not merely a large lookup table: an
EOS-aware hashed lookup is projected into four hyper-connection streams,
gated by the pre-layer hidden state, normalized, and passed through a dilated
depthwise convolution before it is added to the layer input.  This canonical
Gravity adapter closes that *bounded* source formula/input boundary without
loading the full model or pretending that one control is a runnable NR.

The implementation intentionally keeps two concerns separate:

* ``ple_access_trace`` owns the versioned token-to-row and physical-row
  address contract; and
* this module owns exact BF16 support-payload binding plus a direct F32
  evaluation of the documented PLE formula over one sealed pre-PLE state.

Its output is suitable as an input control for later rate--distortion and
runtime work.  It is not independent source-model output parity, complete
Flash execution, a compact representation, or a capability result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:  # ``python -m`` in tests and direct repository-script invocation both work.
    from tools.odyssey.ple_access_trace import (
        DEFAULT_REFERENCE,
        LookupShard,
        PLEContract,
        _is_int,
        _read_safetensors_header,
        bind_source_rows,
        load_contract,
        load_layout,
        locate_combined_row,
        read_source_row,
        sha256_bytes,
        trace_token_segments,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI receipt run.
    from ple_access_trace import (
        DEFAULT_REFERENCE,
        LookupShard,
        PLEContract,
        _is_int,
        _read_safetensors_header,
        bind_source_rows,
        load_contract,
        load_layout,
        locate_combined_row,
        read_source_row,
        sha256_bytes,
        trace_token_segments,
    )


SCHEMA = "hawking.odyssey.ple_source_output_control.v1"
INPUT_RECEIPT_SCHEMA = "hawking.flash_noetic_complete_layer0_source_bf16.v1"
DEFAULT_SPEC = Path(
    "/Volumes/corpdrive/hawking-modellake/specimens/"
    "Qwen--Qwen3.8-Flash-Next@34567a4712bc"
)
DEFAULT_INPUT_STATE = Path(
    "receipts/headless/FLASH_SINGLE_PROCESS_CHAIN_L0_L47/layer-0/state.f32"
)
DEFAULT_INPUT_RECEIPT = Path(
    "receipts/headless/FLASH_SINGLE_PROCESS_CHAIN_L0_L47/layer-0/receipt.json"
)


@dataclass(frozen=True)
class TensorEntry:
    """An exact safetensors payload range used by the PLE formula."""

    name: str
    path: Path
    data_start: int
    payload_offset: int
    dtype: str
    shape: tuple[int, ...]

    @property
    def byte_count(self) -> int:
        count = 1
        for value in self.shape:
            count *= value
        dtype_width = {"BF16": 2, "I64": 8}.get(self.dtype)
        if dtype_width is None:
            raise ValueError(f"unsupported PLE support dtype: {self.dtype}")
        return count * dtype_width


@dataclass(frozen=True)
class PLESupport:
    key_proj: np.ndarray
    value_proj: np.ndarray
    norm_key: np.ndarray
    norm_query: np.ndarray
    norm_conv: np.ndarray
    conv: np.ndarray
    bindings: list[dict[str, Any]]


def _source_revision_from_spec(spec: Path) -> str | None:
    marker = spec.name.rsplit("@", 1)
    return marker[1] if len(marker) == 2 and marker[1] else None


def _payload_bytes(entry: TensorEntry) -> bytes:
    with entry.path.open("rb") as handle:
        handle.seek(entry.data_start + entry.payload_offset)
        raw = handle.read(entry.byte_count)
    if len(raw) != entry.byte_count:
        raise ValueError(f"short source tensor payload: {entry.name}")
    return raw


def bf16_bytes_to_f32(raw: bytes, expected_elements: int | None = None) -> np.ndarray:
    """Decode little-endian IEEE BF16 payload bytes into a copied F32 array."""
    if len(raw) % 2:
        raise ValueError("BF16 payload has an odd byte length")
    words = np.frombuffer(raw, dtype="<u2")
    if expected_elements is not None and words.size != expected_elements:
        raise ValueError(
            f"BF16 element count mismatch: expected={expected_elements} actual={words.size}"
        )
    widened = words.astype("<u4") << 16
    return widened.view("<f4")


def _resolve_tensor_entries(spec: Path, names: Sequence[str]) -> dict[str, TensorEntry]:
    """Resolve only declared PLE support payloads from the source index."""
    index_path = spec / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("source safetensors index has no weight_map")
    headers: dict[Path, tuple[dict[str, Any], int]] = {}
    result: dict[str, TensorEntry] = {}
    for name in names:
        source_file = weight_map.get(name)
        if not isinstance(source_file, str):
            raise ValueError(f"source index has no PLE support tensor: {name}")
        path = spec / source_file
        if path not in headers:
            headers[path] = _read_safetensors_header(path)
        header, data_start = headers[path]
        value = header.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"source header has no PLE support tensor: {name}")
        dtype = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if not (
            isinstance(dtype, str)
            and isinstance(shape, list)
            and shape
            and all(_is_int(item) and item > 0 for item in shape)
            and isinstance(offsets, list)
            and len(offsets) == 2
            and all(_is_int(item) and item >= 0 for item in offsets)
            and int(offsets[1]) >= int(offsets[0])
        ):
            raise ValueError(f"source PLE support tensor has invalid metadata: {name}")
        entry = TensorEntry(
            name=name,
            path=path,
            data_start=data_start,
            payload_offset=int(offsets[0]),
            dtype=dtype,
            shape=tuple(int(item) for item in shape),
        )
        if int(offsets[1]) - int(offsets[0]) != entry.byte_count:
            raise ValueError(f"source PLE support tensor payload size mismatch: {name}")
        result[name] = entry
    return result


def _load_bf16(entry: TensorEntry, expected_shape: tuple[int, ...]) -> tuple[np.ndarray, dict[str, Any]]:
    if entry.dtype != "BF16" or entry.shape != expected_shape:
        raise ValueError(
            f"source PLE tensor geometry mismatch for {entry.name}: "
            f"dtype={entry.dtype} shape={entry.shape} expected=BF16/{expected_shape}"
        )
    raw = _payload_bytes(entry)
    value = bf16_bytes_to_f32(raw, math.prod(expected_shape)).reshape(expected_shape)
    return value, {
        "name": entry.name,
        "path": str(entry.path),
        "dtype": entry.dtype,
        "shape": list(entry.shape),
        "bytes": len(raw),
        "payload_sha256": sha256_bytes(raw),
    }


def _load_i64(entry: TensorEntry, expected_shape: tuple[int, ...]) -> tuple[np.ndarray, dict[str, Any]]:
    if entry.dtype != "I64" or entry.shape != expected_shape:
        raise ValueError(
            f"source PLE tensor geometry mismatch for {entry.name}: "
            f"dtype={entry.dtype} shape={entry.shape} expected=I64/{expected_shape}"
        )
    raw = _payload_bytes(entry)
    value = np.frombuffer(raw, dtype="<i8").copy().reshape(expected_shape)
    return value, {
        "name": entry.name,
        "path": str(entry.path),
        "dtype": entry.dtype,
        "shape": list(entry.shape),
        "bytes": len(raw),
        "payload_sha256": sha256_bytes(raw),
        "values": [int(item) for item in value.reshape(-1)],
    }


def load_source_support(
    spec: Path,
    contract: PLEContract,
    *,
    hidden_size: int,
    hc_count: int,
) -> PLESupport:
    """Load and bind the exact PLE support tensors needed for direct formula use."""
    hc_hidden_size = hidden_size * hc_count
    prefix = f"model.language_model.layers.{contract.source_layer_index}.ple."
    names = {
        "key_proj": prefix + "key_proj.weight",
        "value_proj": prefix + "value_proj.weight",
        "norm_key": prefix + "norm_key.weight",
        "norm_query": prefix + "norm_query.weight",
        "norm_conv": prefix + "norm_conv.weight",
        "conv": prefix + "conv1d.weight",
        "multipliers": prefix + "ple_embedding.layer_multipliers",
        "head_offsets": prefix + "ple_embedding.ngram_heads_offsets",
        "head_sizes": prefix + "ple_embedding.ngram_heads_vocab_sizes",
    }
    entries = _resolve_tensor_entries(spec, list(names.values()))
    key_proj, key_binding = _load_bf16(
        entries[names["key_proj"]], (hc_hidden_size, contract.ple_embed_dim)
    )
    value_proj, value_binding = _load_bf16(
        entries[names["value_proj"]], (hidden_size, contract.ple_embed_dim)
    )
    norm_key, norm_key_binding = _load_bf16(entries[names["norm_key"]], (hc_hidden_size,))
    norm_query, norm_query_binding = _load_bf16(entries[names["norm_query"]], (hc_hidden_size,))
    norm_conv, norm_conv_binding = _load_bf16(entries[names["norm_conv"]], (hc_hidden_size,))
    conv, conv_binding = _load_bf16(entries[names["conv"]], (hc_hidden_size, 1, 4))
    multipliers, multiplier_binding = _load_i64(
        entries[names["multipliers"]], (contract.ngram_size,)
    )
    head_offsets, offset_binding = _load_i64(
        entries[names["head_offsets"]], (contract.ngram_heads,)
    )
    head_sizes, size_binding = _load_i64(
        entries[names["head_sizes"]], (contract.ngram_heads,)
    )
    if tuple(int(value) for value in multipliers) != contract.multipliers:
        raise ValueError("source PLE layer multipliers disagree with the reference hash contract")
    if tuple(int(value) for value in head_offsets) != contract.head_offsets:
        raise ValueError("source PLE head offsets disagree with the reference hash contract")
    if tuple(int(value) for value in head_sizes) != contract.head_vocab_sizes:
        raise ValueError("source PLE head sizes disagree with the reference hash contract")
    return PLESupport(
        key_proj=key_proj,
        value_proj=value_proj,
        norm_key=norm_key,
        norm_query=norm_query,
        norm_conv=norm_conv,
        conv=conv[:, 0, :],
        bindings=[
            key_binding,
            value_binding,
            norm_key_binding,
            norm_query_binding,
            norm_conv_binding,
            conv_binding,
            multiplier_binding,
            offset_binding,
            size_binding,
        ],
    )


def grouped_rms_norm(
    values: np.ndarray,
    weight: np.ndarray,
    *,
    group_size: int,
    eps: float,
) -> np.ndarray:
    """F32 equivalent of ``Qwen4ExpTextRMSNorm.forward`` for a flat last axis."""
    values = np.asarray(values, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if values.shape[-1] != weight.size or values.shape[-1] % group_size:
        raise ValueError("RMSNorm input/weight/group geometry is inconsistent")
    groups = values.reshape(*values.shape[:-1], -1, group_size)
    normalized = groups * np.reciprocal(
        np.sqrt(np.mean(groups * groups, axis=-1, keepdims=True) + np.float32(eps))
    )
    return (normalized.reshape(values.shape) * (np.float32(1.0) + weight)).astype(np.float32)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore"):
        return (np.float32(1.0) / (np.float32(1.0) + np.exp(-values))).astype(np.float32)


def _silu(values: np.ndarray) -> np.ndarray:
    return (values * _sigmoid(values)).astype(np.float32)


def evaluate_ple_sequence(
    hidden_states: np.ndarray,
    embeddings: np.ndarray,
    support: PLESupport,
    *,
    hidden_size: int,
    hc_count: int,
    ngram_size: int,
    eps: float,
    initial_conv_state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Evaluate PLE injection and carry its exact dilated-convolution state.

    This function is deliberately reusable by future stateful direct-runtime
    work.  The CLI currently binds one source pre-layer state, while tests can
    exercise multi-token cache continuity without needing any model payload.
    """
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    hc_hidden_size = hidden_size * hc_count
    if (
        hidden_states.ndim != 2
        or embeddings.ndim != 2
        or hidden_states.shape[0] != embeddings.shape[0]
        or hidden_states.shape[1] != hc_hidden_size
        or embeddings.shape[1] != support.key_proj.shape[1]
    ):
        raise ValueError("PLE hidden-state, embedding, or support geometry is inconsistent")
    key = embeddings @ support.key_proj.T
    value = embeddings @ support.value_proj.T
    key_normed = grouped_rms_norm(key, support.norm_key, group_size=hidden_size, eps=eps).reshape(
        -1, hc_count, hidden_size
    )
    query_normed = grouped_rms_norm(
        hidden_states, support.norm_query, group_size=hidden_size, eps=eps
    ).reshape(-1, hc_count, hidden_size)
    gate = np.sum(key_normed * query_normed, axis=-1, keepdims=True) / np.float32(
        math.sqrt(hidden_size)
    )
    gate = (np.sqrt(np.maximum(np.abs(gate), np.float32(1e-6))) * np.sign(gate)).astype(np.float32)
    gated_value = (_sigmoid(gate) * value[:, None, :]).astype(np.float32)
    gated_value_flat = gated_value.reshape(-1, hc_hidden_size)
    gated_value_normed = grouped_rms_norm(
        gated_value_flat, support.norm_conv, group_size=hidden_size, eps=eps
    )

    state_len = (support.conv.shape[1] - 1) * ngram_size
    if initial_conv_state is None:
        conv_state = np.zeros((state_len, hc_hidden_size), dtype=np.float32)
    else:
        conv_state = np.asarray(initial_conv_state, dtype=np.float32).copy()
        if conv_state.shape != (state_len, hc_hidden_size):
            raise ValueError("PLE convolution state geometry is inconsistent")
    outputs: list[np.ndarray] = []
    # Conv1d is a cross-correlation.  With left padding and dilation=ngram,
    # each output reads offsets 0, ngram, 2*ngram, 3*ngram from this window.
    for normalized_value, raw_value in zip(gated_value_normed, gated_value_flat, strict=True):
        window = np.concatenate((conv_state, normalized_value[None, :]), axis=0)
        convolved = np.zeros(hc_hidden_size, dtype=np.float32)
        for kernel_index in range(support.conv.shape[1]):
            convolved += support.conv[:, kernel_index] * window[kernel_index * ngram_size]
        outputs.append((raw_value + _silu(convolved)).astype(np.float32))
        conv_state = window[-state_len:].copy()
    return (
        np.stack(outputs, axis=0),
        conv_state,
        {
            "key": key,
            "value": value,
            "gate": gate.reshape(-1, hc_count),
            "gated_value": gated_value_flat,
            "gated_value_normed": gated_value_normed,
        },
    )


def load_source_lookup_embeddings(
    shards: Sequence[LookupShard], trace: Sequence[dict[str, Any]], contract: PLEContract
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Read exact referenced lookup rows in head order and form PLE embeddings."""
    cache: dict[int, tuple[np.ndarray, dict[str, Any]]] = {}
    per_token: list[np.ndarray] = []
    ordered_records: list[dict[str, Any]] = []
    for token in trace:
        rows: list[np.ndarray] = []
        accesses = token.get("accesses")
        if not isinstance(accesses, list) or len(accesses) != contract.ngram_heads:
            raise ValueError("PLE trace has an invalid head count")
        for access in accesses:
            combined_row = int(access["combined_row"])
            if combined_row not in cache:
                shard, local_row = locate_combined_row(shards, combined_row)
                raw = read_source_row(shard, local_row)
                cache[combined_row] = (
                    bf16_bytes_to_f32(raw, contract.head_dim),
                    {
                        "combined_row": combined_row,
                        "tensor": shard.tensor,
                        "shard_ordinal": shard.ordinal,
                        "local_row": local_row,
                        "row_bytes": len(raw),
                        "payload_sha256": sha256_bytes(raw),
                    },
                )
            value, record = cache[combined_row]
            rows.append(value)
            ordered_records.append(
                {
                    "session_token_index": int(token["session_token_index"]),
                    "global_head": int(access["global_head"]),
                    **record,
                }
            )
        per_token.append(np.concatenate(rows, axis=0))
    embeddings = np.stack(per_token, axis=0).astype(np.float32)
    source_rows, source_read = bind_source_rows(shards, trace)
    if {record["combined_row"] for record in source_rows} != set(cache):
        raise ValueError("independent PLE source row bindings disagree with direct lookup reads")
    return embeddings, ordered_records, {
        "token_count": len(trace),
        "lookup_events": len(ordered_records),
        "lookup_events_per_token": contract.ngram_heads,
        "embedding_values_per_token": contract.ple_embed_dim,
        "source_row_bytes": contract.head_dim * 2,
        "source_bytes_referenced_across_events": len(ordered_records) * contract.head_dim * 2,
        "unique_source_rows": source_read["unique_rows"],
        "unique_source_bytes": source_read["range_read_bytes"],
        "source_range_read_sha256": source_read["range_read_sha256"],
        "source_row_records": source_rows,
        "ordered_lookup_records": ordered_records,
    }


# Compatibility for the first bounded oracle.  New architecture adapters use
# the public name above rather than duplicating source-row assembly.
_lookup_embeddings = load_source_lookup_embeddings


def _state_summary(path: Path, values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype="<f4")
    raw = values.tobytes(order="C")
    return {
        "path": str(path),
        "dtype": "F32_LE",
        "elements": int(values.size),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "l2": float(np.linalg.norm(values.reshape(-1))),
    }


def _write_state(path: Path, values: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.asarray(values, dtype="<f4").tobytes(order="C"))
    return _state_summary(path, values)


def _load_sealed_layer0_state(
    *,
    spec: Path,
    config_binding: dict[str, Any],
    state_path: Path,
    receipt_path: Path,
    expected_elements: int,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Require the exact sealed layer-0 control before exercising source PLE."""
    document = json.loads(receipt_path.read_text(encoding="utf-8"))
    if document.get("schema") != INPUT_RECEIPT_SCHEMA:
        raise ValueError("pre-PLE state receipt has an unexpected schema")
    source = document.get("source")
    if not isinstance(source, dict):
        raise ValueError("pre-PLE state receipt has no source binding")
    output = source.get("output_state")
    embedding = source.get("embedding")
    if not isinstance(output, dict) or not isinstance(embedding, dict):
        raise ValueError("pre-PLE state receipt lacks output-state or input-token bindings")
    source_revision = _source_revision_from_spec(spec)
    if not source_revision or not str(document.get("pinned_revision", "")).startswith(source_revision):
        raise ValueError("pre-PLE state receipt revision disagrees with the source specimen")
    if source.get("config_sha256") != config_binding.get("sha256"):
        raise ValueError("pre-PLE state receipt config disagrees with the source specimen")
    if (
        output.get("dtype") != "F32_LE"
        or output.get("elements") != expected_elements
        or output.get("bytes") != expected_elements * 4
        or not isinstance(output.get("sha256"), str)
        or not _is_int(embedding.get("token_id"))
    ):
        raise ValueError("pre-PLE state receipt has an invalid output/token contract")
    raw = state_path.read_bytes()
    if len(raw) != expected_elements * 4 or sha256_bytes(raw) != output["sha256"]:
        raise ValueError("pre-PLE input state does not match its sealed receipt")
    return (
        np.frombuffer(raw, dtype="<f4").copy().reshape(1, expected_elements),
        int(embedding["token_id"]),
        {
            "receipt": str(receipt_path),
            "receipt_sha256": sha256_bytes(receipt_path.read_bytes()),
            "schema": document["schema"],
            "qualification": document.get("qualification"),
            "source_revision": document.get("pinned_revision"),
            "input_token_id": int(embedding["token_id"]),
            "state": _state_summary(state_path, np.frombuffer(raw, dtype="<f4")),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--input-state", type=Path, default=DEFAULT_INPUT_STATE)
    parser.add_argument("--input-receipt", type=Path, default=DEFAULT_INPUT_RECEIPT)
    parser.add_argument("--ple-layer-index", type=int, default=0)
    parser.add_argument("--reference-implementation", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--reference-version", default="transformers 5.17.0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-injection", type=Path)
    parser.add_argument("--output-state", type=Path)
    parser.add_argument("--output-conv-state", type=Path)
    args = parser.parse_args()
    started = time.perf_counter_ns()
    spec = args.spec.expanduser().resolve()
    if not spec.is_dir():
        raise FileNotFoundError(spec)
    reference = args.reference_implementation.expanduser().resolve()
    if not reference.is_file():
        raise FileNotFoundError(reference)
    state_path = args.input_state.expanduser().resolve()
    receipt_path = args.input_receipt.expanduser().resolve()
    contract, config_binding = load_contract(spec, args.ple_layer_index)
    config = json.loads((spec / "config.json").read_text(encoding="utf-8"))["text_config"]
    hidden_size = config.get("hidden_size")
    hc_count = config.get("hc_count")
    conv_kernel_size = config.get("ple_conv_kernel_size")
    eps = config.get("rms_norm_eps")
    split_parts = config.get("split_ngram_parts")
    if not all(_is_int(value) and value > 0 for value in (hidden_size, hc_count, conv_kernel_size, split_parts)):
        raise ValueError("Flash source config has invalid PLE hidden/stream/convolution geometry")
    if not isinstance(eps, (int, float)) or not math.isfinite(float(eps)) or float(eps) <= 0:
        raise ValueError("Flash source config has invalid RMSNorm epsilon")
    if conv_kernel_size != 4:
        raise ValueError("this PLE source control requires the documented four-tap PLE convolution")
    hc_hidden_size = int(hidden_size) * int(hc_count)
    hidden_states, token_id, input_binding = _load_sealed_layer0_state(
        spec=spec,
        config_binding=config_binding,
        state_path=state_path,
        receipt_path=receipt_path,
        expected_elements=hc_hidden_size,
    )
    if token_id < 0 or token_id >= contract.vocab_size:
        raise ValueError("sealed pre-PLE source token is outside the PLE vocabulary")
    shards, layout_binding = load_layout(spec, contract, int(split_parts))
    trace = trace_token_segments(contract, [[token_id]])
    embeddings, ordered_lookup_records, active_lookup = load_source_lookup_embeddings(shards, trace, contract)
    support = load_source_support(
        spec,
        contract,
        hidden_size=int(hidden_size),
        hc_count=int(hc_count),
    )
    injection, conv_state, intermediates = evaluate_ple_sequence(
        hidden_states,
        embeddings,
        support,
        hidden_size=int(hidden_size),
        hc_count=int(hc_count),
        ngram_size=contract.ngram_size,
        eps=float(eps),
    )
    post_injection = (hidden_states + injection).astype(np.float32)
    output_state_path = (
        args.output_state.expanduser()
        if args.output_state is not None
        else args.output.with_suffix(".post_ple.f32")
    )
    output_injection_path = (
        args.output_injection.expanduser()
        if args.output_injection is not None
        else args.output.with_suffix(".ple_injection.f32")
    )
    output_conv_state_path = (
        args.output_conv_state.expanduser()
        if args.output_conv_state is not None
        else args.output.with_suffix(".ple_conv_state.f32")
    )
    injection_summary = _write_state(output_injection_path, injection)
    post_summary = _write_state(output_state_path, post_injection)
    conv_summary = _write_state(output_conv_state_path, conv_state)
    support_bytes = sum(int(item["bytes"]) for item in support.bindings)
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "SOURCE_BOUND_PLE_OUTPUT_CONTROL__NO_INDEPENDENT_PLE_OUTPUT_PARITY",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "specimen": {
            "root": str(spec),
            "model": "Qwen/Qwen3.8-Flash-Next",
            "revision": _source_revision_from_spec(spec),
        },
        "reference_contract": {
            "implementation": str(reference),
            "implementation_sha256": sha256_bytes(reference.read_bytes()),
            "implementation_version": args.reference_version,
            "implementation_scope": "Qwen4ExpTextPLELayer.forward + Qwen4ExpTextNGramEmbedding.forward",
            "config": config_binding,
            "formula": {
                "input": "sealed f32 pre-layer hidden state from source-parity layer 0",
                "lookup": "16 EOS-aware source BF16 hashed n-gram rows concatenated to 2560 values",
                "projection": "source BF16 key/value matrices evaluated in F32",
                "normalization": "grouped RMSNorm with source BF16 weights and source epsilon",
                "gate": "signed_sqrt(dot(key_normed, query_normed) / sqrt(hidden_size))",
                "convolution": "four-tap depthwise causal cross-correlation with dilation=ngram_size",
                "output": "PLE injection and post-injection layer-1 input state",
            },
        },
        "input_control": input_binding,
        "checkpoint_layout": layout_binding,
        "source_support": {
            "payload_bytes_materialized": support_bytes,
            "tensors": support.bindings,
            "source_lookup_tensor_full_materializations": 0,
            "source_model_loads": 0,
        },
        "token_sequence": {
            "token_ids": [token_id],
            "binding": "SEALED_LAYER0_BOS_INPUT",
            "initial_ngram_context": [contract.eos_token_id] * contract.context_len,
            "final_ngram_context": [contract.eos_token_id, token_id][-(contract.context_len) :],
            "trace": trace,
        },
        "active_lookup": active_lookup,
        "outputs": {
            "ple_injection": injection_summary,
            "post_injection_state": post_summary,
            "persistent_ple_conv_state": conv_summary,
            "mutable_state_bytes": conv_summary["bytes"] + contract.context_len * 8,
            "intermediate_shapes": {
                name: list(value.shape) for name, value in intermediates.items()
            },
            "post_injection_is_next_layer_input": True,
        },
        "execution": {
            "backend": "numpy_f32_source_formula_control",
            "source_weights": "exact BF16 payloads decoded directly from source ranges",
            "source_lookup_row_range_reads": active_lookup["unique_source_rows"],
            "source_lookup_tensor_full_materializations": 0,
            "source_support_payload_materializations": len(support.bindings),
            "source_support_payload_bytes_materialized": support_bytes,
            "elapsed_ns": time.perf_counter_ns() - started,
        },
        "claim_boundary": (
            "This is one source-bound PLE formula/input control: the sealed layer-0 BOS state, "
            "reference-addressed source lookup rows, and exact source PLE support tensors produce a "
            "direct F32 PLE injection and persistent PLE convolution state. It does not establish "
            "independent HuggingFace/source PLE-output parity, layer-1 parity, an implemented native "
            "PLE runtime, full 48-layer Flash fidelity, complete NR closure/EBPW, TPS, or capability."
        ),
        "promotion_allowed": False,
        "next": (
            "Use the sealed post-PLE state as a bounded layer-1 input control; first compare this "
            "formula path with an independent source PLE oracle or native implementation, then evaluate "
            "non-uniform lookup/substrate repair against actual PLE-output sensitivity. Do not use active "
            "row bytes as a complete-representation claim."
        ),
    }
    document["seal_sha256"] = sha256_bytes(json.dumps(document, sort_keys=True).encode("utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": document["status"],
                "token_id": token_id,
                "lookup_rows": active_lookup["unique_source_rows"],
                "source_support_bytes": support_bytes,
                "out": str(args.output),
                "seal": document["seal_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
