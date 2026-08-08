"""Bounded, source-bound Qwen KV/state codec research.

This lane deliberately sits below a token runtime.  It reads a few verified
local BF16 projection tensors, feeds them deterministic *component* vectors,
and materializes the resulting KV/state-like arrays through several physical
storage codecs.  The artifacts make the state-memory trade-off measurable
without presenting a synthetic prompt probe as model execution.

It is intentionally narrow:

* Qwen30: one real attention K/V projection pair and the exact 48-layer KV
  geometry from the local config/index.
* Qwen80: one real gated-attention K/V projection pair plus one real DeltaNet
  Q/K/V/Z projection and decay parameter.  Its attention cache and recurrent
  DeltaNet state are accounted independently.

No tokenizer, prompt, complete layer graph, Metal runtime, HCLI endpoint, TPS,
or tournament qualification is implemented or claimed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lab.operators.qwen30b_gravity_pack import load_tensor, load_weight_map, read_safetensors_header
from lab.receipts import SealIntegrityError, seal, verify


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "hawking.ascension.qwen_state_kv_physical_research.v1"
COMPONENT_RECEIPT_SCHEMA = "hawking.ascension.qwen_state_kv_component_receipt.v1"
STATUS_SCHEMA = "hawking.ascension.qwen_state_kv_status.v1"
FORMAT_SCHEMA = "hawking.ascension.qwen_state_kv_codec_format.v1"

MAGIC = b"HAWKKV1\0"
FORMAT_VERSION = 1
HEADER = struct.Struct("<8sBBHIIIII")
HEADER_BYTES = HEADER.size
GROUP_SIZE = 64
RESIDUAL_RATIO = 0.01
DEFAULT_SESSION_TOKENS = 8
HASH_CHUNK_BYTES = 8 * 1024 * 1024

DEFAULT_QWEN30_MODEL_DIR = (
    REPO_ROOT
    / "workspace/campaign/records/runs/qwen-30b/Qwen3-Coder-30B-A3B-Instruct"
)
DEFAULT_QWEN80_MODEL_DIR = REPO_ROOT / "workspace/campaign/records/runs/qwen-80b/Qwen3-Coder-Next"
DEFAULT_QWEN30_ROOT = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen30/state-kv"
DEFAULT_QWEN80_ROOT = REPO_ROOT / "workspace/campaign/records/ascension-sandbox/physical/qwen80/state-kv"
DEFAULT_QWEN30_AUDIT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen30/QWEN30_SOURCE_BODY_AUDIT_CANDIDATE.json"
)
DEFAULT_QWEN80_AUDIT = (
    REPO_ROOT
    / "workspace/campaign/records/ascension-sandbox/physical/qwen80-acquisition/QWEN80_SOURCE_BODY_AUDIT_CANDIDATE.json"
)


class StateKVError(RuntimeError):
    """The bounded source-bound state experiment cannot safely proceed."""


@dataclass(frozen=True)
class ModelSpec:
    identifier: str
    architecture: str
    model_dir: Path
    root: Path
    audit_path: Path


@dataclass(frozen=True)
class CodecResult:
    """A fully materialized codec body and its decoded reconstruction."""

    name: str
    code: int
    bits: int | None
    group_size: int
    groups: int
    residual_count: int
    body: bytes
    reconstruction: np.ndarray
    details: Mapping[str, Any]

    @property
    def elements(self) -> int:
        return int(self.reconstruction.size)

    @property
    def serialized_bytes(self) -> int:
        return HEADER_BYTES + len(self.body)


SPECS: Mapping[str, ModelSpec] = {
    "qwen30": ModelSpec(
        identifier="qwen30",
        architecture="Qwen3MoeForCausalLM",
        model_dir=DEFAULT_QWEN30_MODEL_DIR,
        root=DEFAULT_QWEN30_ROOT,
        audit_path=DEFAULT_QWEN30_AUDIT,
    ),
    "qwen80": ModelSpec(
        identifier="qwen80",
        architecture="Qwen3NextForCausalLM",
        model_dir=DEFAULT_QWEN80_MODEL_DIR,
        root=DEFAULT_QWEN80_ROOT,
        audit_path=DEFAULT_QWEN80_AUDIT,
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    normalized = np.ascontiguousarray(values, dtype="<f4")
    return _sha256_bytes(normalized.tobytes())


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
        os.chmod(path, 0o640)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    _atomic_bytes(path, rendered)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _required_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int) or value <= 0:
        raise StateKVError(f"config {key!r} must be a positive integer")
    return value


def _load_config(model_dir: Path) -> dict[str, Any]:
    try:
        value = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateKVError(f"cannot read model config from {model_dir}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise StateKVError("model config root is not an object")
    return dict(value)


def _read_audit(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    if document is None:
        raise StateKVError(f"missing local source audit: {path}")
    try:
        return verify(document, label=str(path))
    except SealIntegrityError as exc:
        raise StateKVError(f"local source audit is not sealed/valid: {exc}") from exc


def _audited_shard_bytes(audit: Mapping[str, Any], shard_name: str) -> int | None:
    source = audit.get("source")
    if isinstance(source, Mapping):
        shards = source.get("shards")
        if isinstance(shards, Mapping):
            entry = shards.get(shard_name)
            if isinstance(entry, Mapping) and isinstance(entry.get("bytes"), int):
                return int(entry["bytes"])
    files = audit.get("files")
    if isinstance(files, Sequence) and not isinstance(files, (str, bytes)):
        for entry in files:
            if (
                isinstance(entry, Mapping)
                and entry.get("path") == shard_name
                and isinstance(entry.get("bytes"), int)
            ):
                return int(entry["bytes"])
    return None


def _raw_tensor_binding(model_dir: Path, weight_map: Mapping[str, str], name: str, values: np.ndarray, audit: Mapping[str, Any]) -> dict[str, Any]:
    """Hash the live source tensor bytes while relying on the sealed full-body audit.

    Rehashing every 3.7--4.0 GB shard would duplicate the acquisition worker's
    completed job.  We instead verify that the shard byte size still matches the
    sealed audit and bind this run to the exact BF16 tensor byte range read now.
    """

    try:
        shard_name = str(weight_map[name])
    except KeyError as exc:
        raise StateKVError(f"source tensor missing from index: {name}") from exc
    shard = model_dir / shard_name
    if not shard.is_file():
        raise StateKVError(f"source shard missing for {name}: {shard}")
    expected_bytes = _audited_shard_bytes(audit, shard_name)
    if expected_bytes is None:
        raise StateKVError(f"sealed source audit does not inventory shard {shard_name}")
    observed_bytes = shard.stat().st_size
    if observed_bytes != expected_bytes:
        raise StateKVError(
            f"source shard size differs from sealed audit for {shard_name}: {observed_bytes} != {expected_bytes}"
        )
    header = read_safetensors_header(shard)
    try:
        tensor_info = header[name]
        start, stop = (int(value) for value in tensor_info["data_offsets"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StateKVError(f"invalid safetensors entry for {name}") from exc
    with shard.open("rb") as handle:
        header_bytes = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(8 + header_bytes + start)
        remaining = stop - start
        digest = hashlib.sha256()
        while remaining:
            chunk = handle.read(min(remaining, HASH_CHUNK_BYTES))
            if not chunk:
                raise StateKVError(f"short source tensor read while hashing {name}")
            digest.update(chunk)
            remaining -= len(chunk)
    return {
        "tensor_name": name,
        "source_shard": shard_name,
        "source_shard_bytes_matched_to_sealed_audit": observed_bytes,
        "raw_tensor_bf16_bytes": stop - start,
        "raw_tensor_bf16_sha256": digest.hexdigest(),
        "decoded_float32_shape": list(values.shape),
        "decoded_float32_sha256": _sha256_array(values),
    }


def deterministic_component_input(tokens: int, width: int, *, label: str) -> np.ndarray:
    """Return a deterministic non-linguistic component input matrix.

    The construction intentionally does not involve tokens, a vocabulary,
    prompt text, or a tokenizer.  Row normalization keeps projection magnitudes
    bounded across both models while preserving a stable source-bound witness.
    """

    if tokens <= 0 or width <= 0:
        raise StateKVError("component input dimensions must be positive")
    label_seed = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little")
    rows = np.arange(tokens, dtype=np.float32)[:, None] + 1.0
    columns = np.arange(width, dtype=np.float32)[None, :] + 1.0
    phase = float((label_seed % 100003) + 1) / 100003.0
    values = np.sin(rows * columns * (0.00037 + phase * 0.00011))
    values += 0.5 * np.cos((rows + phase) * columns * 0.00019)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.ascontiguousarray(values / np.maximum(norms, 1e-12), dtype=np.float32)


def _pack_signed_codes(codes: np.ndarray, *, bits: int, max_code: int) -> bytes:
    unsigned = np.ascontiguousarray(codes.reshape(-1), dtype=np.int16) + max_code
    if np.any(unsigned < 0) or np.any(unsigned >= (1 << bits)):
        raise StateKVError("quantized code cannot fit requested bit width")
    bit_matrix = ((unsigned.astype(np.uint16)[:, None] >> np.arange(bits, dtype=np.uint16)) & 1).astype(np.uint8)
    return np.packbits(bit_matrix.reshape(-1), bitorder="little").tobytes()


def _unpack_signed_codes(payload: bytes, *, elements: int, bits: int, max_code: int) -> np.ndarray:
    needed = elements * bits
    raw_bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder="little")[:needed]
    weights = 1 << np.arange(bits, dtype=np.uint16)
    unsigned = (raw_bits.reshape(elements, bits).astype(np.uint16) * weights).sum(axis=1)
    return unsigned.astype(np.int16) - max_code


def _group_codec(values: np.ndarray, *, name: str, code: int, bits: int, group_size: int = GROUP_SIZE) -> CodecResult:
    reference = np.ascontiguousarray(values, dtype=np.float32)
    flat = reference.reshape(-1)
    elements = int(flat.size)
    groups = math.ceil(elements / group_size)
    padded = np.pad(flat, (0, groups * group_size - elements))
    grouped = padded.reshape(groups, group_size)
    max_code = (1 << (bits - 1)) - 1
    if max_code <= 0:
        raise StateKVError("quantizer bit width cannot represent a signed range")
    scale_f32 = np.max(np.abs(grouped), axis=1) / float(max_code)
    scales = scale_f32.astype("<f2")
    normalizer = np.where(scales.astype(np.float32) > 0.0, scales.astype(np.float32), 1.0)
    codes_padded = np.rint(grouped / normalizer[:, None]).clip(-max_code, max_code).astype(np.int16)
    codes = codes_padded.reshape(-1)[:elements]
    code_bytes = _pack_signed_codes(codes, bits=bits, max_code=max_code)
    decoded_codes = _unpack_signed_codes(code_bytes, elements=elements, bits=bits, max_code=max_code)
    decoded_padded = np.zeros(groups * group_size, dtype=np.float32)
    decoded_padded[:elements] = decoded_codes.astype(np.float32)
    reconstruction = (
        decoded_padded.reshape(groups, group_size) * scales.astype(np.float32)[:, None]
    ).reshape(-1)[:elements].reshape(reference.shape)
    return CodecResult(
        name=name,
        code=code,
        bits=bits,
        group_size=group_size,
        groups=groups,
        residual_count=0,
        body=code_bytes + scales.tobytes(),
        reconstruction=np.ascontiguousarray(reconstruction, dtype=np.float32),
        details={
            "code_bytes": len(code_bytes),
            "scale_dtype": "float16",
            "scale_bytes": int(scales.nbytes),
            "signed_code_range": [-max_code, max_code],
        },
    )


def _fp16_codec(values: np.ndarray) -> CodecResult:
    reference = np.ascontiguousarray(values, dtype=np.float32)
    payload_values = reference.astype("<f2")
    return CodecResult(
        name="fp16_reference",
        code=1,
        bits=16,
        group_size=0,
        groups=0,
        residual_count=0,
        body=payload_values.tobytes(),
        reconstruction=payload_values.astype(np.float32),
        details={"value_dtype": "float16", "value_bytes": int(payload_values.nbytes)},
    )


def _protected_residual_codec(
    values: np.ndarray,
    *,
    group_size: int = GROUP_SIZE,
    residual_ratio: float = RESIDUAL_RATIO,
) -> CodecResult:
    """Q4 base stream with deterministic FP16 corrections for largest errors."""

    if not 0.0 < residual_ratio <= 1.0:
        raise StateKVError("protected residual ratio must be in (0, 1]")
    reference = np.ascontiguousarray(values, dtype=np.float32)
    base = _group_codec(
        reference,
        name="protected_residual_q4_group64_top1pct_fp16",
        code=4,
        bits=4,
        group_size=group_size,
    )
    flat = reference.reshape(-1)
    base_flat = base.reconstruction.reshape(-1)
    residual_count = max(1, math.ceil(flat.size * residual_ratio))
    errors = np.abs(flat - base_flat)
    # lexsort makes equal-error selection stable across NumPy versions.
    indexes = np.lexsort((np.arange(flat.size, dtype=np.int64), -errors))[:residual_count].astype("<u4")
    residuals = (flat[indexes] - base_flat[indexes]).astype("<f2")
    reconstruction = base_flat.copy()
    reconstruction[indexes] += residuals.astype(np.float32)
    return CodecResult(
        name="protected_residual_q4_group64_top1pct_fp16",
        code=4,
        bits=4,
        group_size=group_size,
        groups=base.groups,
        residual_count=residual_count,
        body=base.body + indexes.tobytes() + residuals.tobytes(),
        reconstruction=np.ascontiguousarray(reconstruction.reshape(reference.shape), dtype=np.float32),
        details={
            **dict(base.details),
            "base_representation": "symmetric_group_q4",
            "residual_ratio": residual_ratio,
            "residual_index_dtype": "uint32",
            "residual_value_dtype": "float16",
            "residual_index_bytes": int(indexes.nbytes),
            "residual_value_bytes": int(residuals.nbytes),
            "residual_selection": "largest_absolute_q4_reconstruction_error_stable_tiebreak_by_flat_index",
        },
    )


def codec_suite(values: np.ndarray) -> tuple[CodecResult, ...]:
    """Materialize all state formats required by this bounded lane."""

    return (
        _fp16_codec(values),
        _group_codec(values, name="q8_group64", code=2, bits=8),
        _group_codec(values, name="q4_group64", code=3, bits=4),
        _protected_residual_codec(values),
    )


def serialize_codec(codec: CodecResult) -> bytes:
    header = HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        codec.code,
        codec.group_size,
        codec.elements,
        codec.groups,
        codec.residual_count,
        len(codec.body),
        0,
    )
    return header + codec.body


def _metrics(reference: np.ndarray, reconstruction: np.ndarray) -> dict[str, Any]:
    source = np.ascontiguousarray(reference, dtype=np.float32).reshape(-1)
    restored = np.ascontiguousarray(reconstruction, dtype=np.float32).reshape(-1)
    if source.shape != restored.shape:
        raise StateKVError("codec reconstruction shape does not match component output")
    diff = source - restored
    source_norm = float(np.linalg.norm(source))
    restored_norm = float(np.linalg.norm(restored))
    cosine = 0.0 if source_norm < 1e-12 or restored_norm < 1e-12 else float(np.dot(source, restored) / (source_norm * restored_norm))
    return {
        "finite": bool(np.isfinite(restored).all()),
        "relative_l2": float(np.linalg.norm(diff) / max(source_norm, 1e-12)),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
        "max_abs": float(np.max(np.abs(diff))),
        "cosine": cosine,
    }


def codec_storage_bytes(
    codec_name: str,
    *,
    elements: int,
    group_size: int = GROUP_SIZE,
    residual_ratio: float = RESIDUAL_RATIO,
) -> int:
    """Exact byte count for one independently stored state array in this format."""

    if elements <= 0:
        raise StateKVError("state element count must be positive")
    if codec_name == "fp16_reference":
        return HEADER_BYTES + elements * 2
    if codec_name not in {"q8_group64", "q4_group64", "protected_residual_q4_group64_top1pct_fp16"}:
        raise StateKVError(f"unknown state codec {codec_name!r}")
    bits = 8 if codec_name == "q8_group64" else 4
    groups = math.ceil(elements / group_size)
    code_bytes = math.ceil(elements * bits / 8)
    scale_bytes = groups * 2
    residual_bytes = 0
    if codec_name == "protected_residual_q4_group64_top1pct_fp16":
        residual_bytes = math.ceil(elements * residual_ratio) * (4 + 2)
    return HEADER_BYTES + code_bytes + scale_bytes + residual_bytes


def _codec_names() -> tuple[str, ...]:
    return (
        "fp16_reference",
        "q8_group64",
        "q4_group64",
        "protected_residual_q4_group64_top1pct_fp16",
    )


def growing_kv_ledger(
    *,
    layer_count: int,
    key_value_heads: int,
    head_dim: int,
    session_tokens: int,
) -> dict[str, Any]:
    """Ledger cache arrays per layer, then sums their exact format byte counts."""

    if min(layer_count, key_value_heads, head_dim, session_tokens) <= 0:
        raise StateKVError("KV geometry must be positive")
    values_per_layer_per_token = 2 * key_value_heads * head_dim
    values_per_layer_session = values_per_layer_per_token * session_tokens
    codecs: dict[str, Any] = {}
    for name in _codec_names():
        per_layer = codec_storage_bytes(name, elements=values_per_layer_session)
        per_session = layer_count * per_layer
        codecs[name] = {
            "per_layer_state_array_bytes": per_layer,
            "bytes_per_session": per_session,
            "bytes_per_token_amortized_at_declared_session": per_session / session_tokens,
        }
    return {
        "kind": "growing_attention_kv_cache",
        "session_tokens": session_tokens,
        "layer_count": layer_count,
        "key_value_heads": key_value_heads,
        "head_dim": head_dim,
        "values_per_layer_per_token": values_per_layer_per_token,
        "values_per_layer_session": values_per_layer_session,
        "values_per_session": layer_count * values_per_layer_session,
        "codec_format": FORMAT_SCHEMA,
        "codec_storage_assumption": "each layer state array is independently stored with the fixed 32-byte HAWKKV1 header",
        "codecs": codecs,
    }


def recurrent_state_ledger(
    *,
    layer_count: int,
    heads: int,
    key_dim: int,
    value_dim: int,
    session_tokens: int,
) -> dict[str, Any]:
    """Ledger fixed-size recurrent state, distinct from a cache that grows per token."""

    if min(layer_count, heads, key_dim, value_dim, session_tokens) <= 0:
        raise StateKVError("recurrent state geometry must be positive")
    values_per_layer = heads * key_dim * value_dim
    codecs: dict[str, Any] = {}
    for name in _codec_names():
        per_layer = codec_storage_bytes(name, elements=values_per_layer)
        per_session = layer_count * per_layer
        codecs[name] = {
            "per_layer_state_array_bytes": per_layer,
            "bytes_per_session_resident_state": per_session,
            "bytes_per_token_amortized_at_declared_session": per_session / session_tokens,
            "growth_bytes_per_additional_token": 0,
        }
    return {
        "kind": "fixed_size_deltanet_recurrent_state",
        "session_tokens_for_amortization_only": session_tokens,
        "layer_count": layer_count,
        "heads": heads,
        "key_head_dim": key_dim,
        "value_head_dim": value_dim,
        "values_per_layer": values_per_layer,
        "values_per_session_resident_state": layer_count * values_per_layer,
        "codec_format": FORMAT_SCHEMA,
        "codec_storage_assumption": "each DeltaNet layer state array is independently stored with the fixed 32-byte HAWKKV1 header",
        "codecs": codecs,
    }


def _attention_layers(weight_map: Mapping[str, str]) -> tuple[int, ...]:
    pattern = re.compile(r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$")
    layers = sorted(int(match.group(1)) for name in weight_map for match in [pattern.match(name)] if match)
    if not layers:
        raise StateKVError("no attention key projection tensors found in local index")
    for layer in layers:
        expected = f"model.layers.{layer}.self_attn.v_proj.weight"
        if expected not in weight_map:
            raise StateKVError(f"attention layer {layer} lacks a matching V projection")
    return tuple(layers)


def _linear_attention_layers(weight_map: Mapping[str, str]) -> tuple[int, ...]:
    pattern = re.compile(r"^model\.layers\.(\d+)\.linear_attn\.in_proj_qkvz\.weight$")
    layers = sorted(int(match.group(1)) for name in weight_map for match in [pattern.match(name)] if match)
    if not layers:
        raise StateKVError("no DeltaNet Q/K/V/Z projection tensors found in local index")
    return tuple(layers)


def _source_summary(spec: ModelSpec, audit: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_dir": str(spec.model_dir),
        "config_sha256": _sha256_bytes((spec.model_dir / "config.json").read_bytes()),
        "model_type": config.get("model_type"),
        "architecture": spec.architecture,
        "sealed_source_body_audit": {
            "path": str(spec.audit_path),
            "schema": audit.get("schema"),
            "status": audit.get("status"),
            "seal_sha256": audit.get("seal_sha256"),
        },
    }


def _projection_kv_sample(
    *,
    hidden: np.ndarray,
    key_weights: np.ndarray,
    value_weights: np.ndarray,
    key_value_heads: int,
    head_dim: int,
) -> np.ndarray:
    expected_width = key_value_heads * head_dim
    if key_weights.shape != (expected_width, hidden.shape[1]):
        raise StateKVError(f"K projection shape {key_weights.shape} does not match exact geometry")
    if value_weights.shape != (expected_width, hidden.shape[1]):
        raise StateKVError(f"V projection shape {value_weights.shape} does not match exact geometry")
    keys = hidden @ np.ascontiguousarray(key_weights, dtype=np.float32).T
    values = hidden @ np.ascontiguousarray(value_weights, dtype=np.float32).T
    return np.ascontiguousarray(
        np.stack(
            [
                keys.reshape(hidden.shape[0], key_value_heads, head_dim),
                values.reshape(hidden.shape[0], key_value_heads, head_dim),
            ],
            axis=1,
        ),
        dtype=np.float32,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def _deltanet_state_proxy(
    *,
    hidden: np.ndarray,
    qkvz_weights: np.ndarray,
    a_log: np.ndarray,
    key_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a bounded source-projection state witness, not a DeltaNet runtime.

    The exact state shape is real: [value_heads, key_dim, value_dim].  The
    update is deliberately a transparent proxy: Q/K/V/Z are projected by the
    official source weights and A_log provides per-value-head decay, but this
    omits the native convolution, dt path, normalization, and full model graph.
    """

    q_width = key_heads * key_dim
    k_width = key_heads * key_dim
    v_width = value_heads * value_dim
    z_width = value_heads * value_dim
    expected_width = q_width + k_width + v_width + z_width
    if qkvz_weights.shape != (expected_width, hidden.shape[1]):
        raise StateKVError(f"DeltaNet Q/K/V/Z shape {qkvz_weights.shape} does not match exact config geometry")
    if a_log.reshape(-1).size != value_heads:
        raise StateKVError(f"DeltaNet A_log count {a_log.size} does not match value head count {value_heads}")
    projected = hidden @ np.ascontiguousarray(qkvz_weights, dtype=np.float32).T
    q_part, k_part, v_part, z_part = np.split(projected, [q_width, q_width + k_width, q_width + k_width + v_width], axis=1)
    queries = q_part.reshape(hidden.shape[0], key_heads, key_dim)
    keys = k_part.reshape(hidden.shape[0], key_heads, key_dim)
    values = v_part.reshape(hidden.shape[0], value_heads, value_dim)
    gates = _sigmoid(z_part.reshape(hidden.shape[0], value_heads, value_dim))
    if value_heads % key_heads:
        raise StateKVError("DeltaNet value heads must divide evenly by key heads for this grouped state witness")
    keys_per_value_head = np.repeat(keys, value_heads // key_heads, axis=1)
    query_per_value_head = np.repeat(queries, value_heads // key_heads, axis=1)
    decay = np.exp(-np.exp(np.clip(np.asarray(a_log, dtype=np.float32).reshape(-1), -20.0, 10.0)))
    state = np.zeros((value_heads, key_dim, value_dim), dtype=np.float32)
    for token in range(hidden.shape[0]):
        write = np.einsum("hi,hj->hij", keys_per_value_head[token], values[token] * gates[token], optimize=True)
        # Q is not required to form the state, but source-derived Q participates
        # in a stable bounded write modulation so every Q/K/V/Z partition is read.
        query_modulation = 0.5 + 0.5 * np.tanh(np.mean(query_per_value_head[token], axis=1))
        state = decay[:, None, None] * state + write * query_modulation[:, None, None]
    return np.ascontiguousarray(state, dtype=np.float32), {
        "projected_partition_widths": {"q": q_width, "k": k_width, "v": v_width, "z": z_width},
        "state_shape": [value_heads, key_dim, value_dim],
        "update": "decayed_grouped_outer_product_proxy_from_real_qkvz_projections",
        "proxy_exclusions": ["conv1d", "dt_path", "normalization", "complete_deltanet_layer", "full_model_runtime"],
    }


def _write_component_artifacts(
    *,
    root: Path,
    model: str,
    component: str,
    values: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    artifact_root = root / "artifacts"
    for codec in codec_suite(values):
        payload = serialize_codec(codec)
        path = artifact_root / f"{model}_{component}__{codec.name}.hkv"
        _atomic_bytes(path, payload)
        measured = _metrics(values, codec.reconstruction)
        rows.append(
            {
                "codec": codec.name,
                "format": FORMAT_SCHEMA,
                "bits": codec.bits,
                "group_size": codec.group_size or None,
                "groups": codec.groups or None,
                "residual_count": codec.residual_count or None,
                "physical_artifact": {
                    "path": str(path),
                    "bytes": len(payload),
                    "sha256": _sha256_bytes(payload),
                    "header_bytes": HEADER_BYTES,
                    "body_bytes": len(codec.body),
                },
                "codec_details": dict(codec.details),
                "reconstruction": measured,
            }
        )
    return rows


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    document = seal(payload)
    _atomic_json(path, document)
    return document


def _common_claim_boundary() -> dict[str, bool]:
    return {
        "deterministic_component_vectors_are_not_prompts_or_tokens": True,
        "no_tokenizer_or_prompt_template_executed": True,
        "no_complete_model_layer_graph_executed": True,
        "no_native_full_runtime_or_hcli_endpoint": True,
        "no_model_capability_claim": True,
        "no_tps_measurement_or_100_tps_or_tg3_claim": True,
        "not_a_tournament_candidate_or_selection_action": True,
    }


def _run_qwen30(spec: ModelSpec, *, session_tokens: int) -> dict[str, Any]:
    audit = _read_audit(spec.audit_path)
    config = _load_config(spec.model_dir)
    if config.get("model_type") != "qwen3_moe":
        raise StateKVError(f"qwen30 source config model_type mismatch: {config.get('model_type')!r}")
    hidden_size = _required_int(config, "hidden_size")
    key_value_heads = _required_int(config, "num_key_value_heads")
    head_dim = _required_int(config, "head_dim")
    layer_count = _required_int(config, "num_hidden_layers")
    weight_map = load_weight_map(spec.model_dir)
    attention_layers = _attention_layers(weight_map)
    if len(attention_layers) != layer_count:
        raise StateKVError(f"qwen30 index exposes {len(attention_layers)} attention layers, config requires {layer_count}")
    layer = attention_layers[0]
    key_name = f"model.layers.{layer}.self_attn.k_proj.weight"
    value_name = f"model.layers.{layer}.self_attn.v_proj.weight"
    key_weights = load_tensor(spec.model_dir, weight_map, key_name)
    value_weights = load_tensor(spec.model_dir, weight_map, value_name)
    hidden = deterministic_component_input(session_tokens, hidden_size, label="qwen30-attention-kv")
    state_values = _projection_kv_sample(
        hidden=hidden,
        key_weights=key_weights,
        value_weights=value_weights,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
    )
    source_bindings = [
        _raw_tensor_binding(spec.model_dir, weight_map, key_name, key_weights, audit),
        _raw_tensor_binding(spec.model_dir, weight_map, value_name, value_weights, audit),
    ]
    ledger = growing_kv_ledger(
        layer_count=layer_count,
        key_value_heads=key_value_heads,
        head_dim=head_dim,
        session_tokens=session_tokens,
    )
    artifacts = _write_component_artifacts(root=spec.root, model=spec.identifier, component="attention_kv", values=state_values)
    receipt_path = spec.root / "QWEN30_ATTENTION_KV_STATE_CODEC_RECEIPT.json"
    receipt = _write_receipt(
        receipt_path,
        {
            "schema": COMPONENT_RECEIPT_SCHEMA,
            "recorded_at": _utc_now(),
            "model": {"id": "Qwen3-Coder-30B-A3B-Instruct", "architecture": spec.architecture},
            "source_verification": _source_summary(spec, audit, config),
            "source_projection_bindings": source_bindings,
            "component": {
                "kind": "attention_kv_projection_output",
                "source_layer": layer,
                "source_tensor_names": [key_name, value_name],
                "output_shape": list(state_values.shape),
                "output_float32_sha256": _sha256_array(state_values),
                "output_semantics": "real K and V linear projections of deterministic non-linguistic component vectors",
            },
            "deterministic_component_input": {
                "shape": list(hidden.shape),
                "float32_sha256": _sha256_array(hidden),
                "generator": "normalized_sine_cosine_indexed_by_model_component_label",
            },
            "exact_geometry_ledger": ledger,
            "codec_results": artifacts,
            "claim_boundary": _common_claim_boundary(),
        },
    )
    return {
        "receipt_path": str(receipt_path),
        "receipt_seal_sha256": receipt["seal_sha256"],
        "component_count": 1,
        "artifact_count": len(artifacts),
        "geometry": {"attention_layers": list(attention_layers), "ledger": ledger},
    }


def _run_qwen80(spec: ModelSpec, *, session_tokens: int) -> dict[str, Any]:
    audit = _read_audit(spec.audit_path)
    config = _load_config(spec.model_dir)
    if config.get("model_type") != "qwen3_next":
        raise StateKVError(f"qwen80 source config model_type mismatch: {config.get('model_type')!r}")
    hidden_size = _required_int(config, "hidden_size")
    key_value_heads = _required_int(config, "num_key_value_heads")
    attention_head_dim = _required_int(config, "head_dim")
    total_layers = _required_int(config, "num_hidden_layers")
    linear_key_heads = _required_int(config, "linear_num_key_heads")
    linear_value_heads = _required_int(config, "linear_num_value_heads")
    linear_key_dim = _required_int(config, "linear_key_head_dim")
    linear_value_dim = _required_int(config, "linear_value_head_dim")
    weight_map = load_weight_map(spec.model_dir)
    attention_layers = _attention_layers(weight_map)
    linear_layers = _linear_attention_layers(weight_map)
    if len(attention_layers) + len(linear_layers) != total_layers or set(attention_layers) & set(linear_layers):
        raise StateKVError("qwen80 local index does not describe one attention/DeltaNet operator per layer")
    hidden_attention = deterministic_component_input(session_tokens, hidden_size, label="qwen80-attention-kv")
    attention_layer = attention_layers[0]
    key_name = f"model.layers.{attention_layer}.self_attn.k_proj.weight"
    value_name = f"model.layers.{attention_layer}.self_attn.v_proj.weight"
    key_weights = load_tensor(spec.model_dir, weight_map, key_name)
    value_weights = load_tensor(spec.model_dir, weight_map, value_name)
    attention_values = _projection_kv_sample(
        hidden=hidden_attention,
        key_weights=key_weights,
        value_weights=value_weights,
        key_value_heads=key_value_heads,
        head_dim=attention_head_dim,
    )
    attention_ledger = growing_kv_ledger(
        layer_count=len(attention_layers),
        key_value_heads=key_value_heads,
        head_dim=attention_head_dim,
        session_tokens=session_tokens,
    )
    attention_artifacts = _write_component_artifacts(
        root=spec.root, model=spec.identifier, component="attention_kv", values=attention_values
    )
    attention_receipt_path = spec.root / "QWEN80_ATTENTION_KV_STATE_CODEC_RECEIPT.json"
    attention_receipt = _write_receipt(
        attention_receipt_path,
        {
            "schema": COMPONENT_RECEIPT_SCHEMA,
            "recorded_at": _utc_now(),
            "model": {"id": "Qwen3-Coder-Next-80B", "architecture": spec.architecture},
            "source_verification": _source_summary(spec, audit, config),
            "source_projection_bindings": [
                _raw_tensor_binding(spec.model_dir, weight_map, key_name, key_weights, audit),
                _raw_tensor_binding(spec.model_dir, weight_map, value_name, value_weights, audit),
            ],
            "component": {
                "kind": "gated_attention_kv_projection_output",
                "source_layer": attention_layer,
                "source_tensor_names": [key_name, value_name],
                "attention_layers_derived_from_local_index": list(attention_layers),
                "output_shape": list(attention_values.shape),
                "output_float32_sha256": _sha256_array(attention_values),
                "output_semantics": "real K and V linear projections of deterministic non-linguistic component vectors",
            },
            "deterministic_component_input": {
                "shape": list(hidden_attention.shape),
                "float32_sha256": _sha256_array(hidden_attention),
                "generator": "normalized_sine_cosine_indexed_by_model_component_label",
            },
            "exact_geometry_ledger": attention_ledger,
            "codec_results": attention_artifacts,
            "claim_boundary": _common_claim_boundary(),
        },
    )

    hidden_linear = deterministic_component_input(session_tokens, hidden_size, label="qwen80-deltanet-state")
    linear_layer = linear_layers[0]
    qkvz_name = f"model.layers.{linear_layer}.linear_attn.in_proj_qkvz.weight"
    a_log_name = f"model.layers.{linear_layer}.linear_attn.A_log"
    qkvz_weights = load_tensor(spec.model_dir, weight_map, qkvz_name)
    a_log = load_tensor(spec.model_dir, weight_map, a_log_name)
    recurrent_values, proxy_details = _deltanet_state_proxy(
        hidden=hidden_linear,
        qkvz_weights=qkvz_weights,
        a_log=a_log,
        key_heads=linear_key_heads,
        value_heads=linear_value_heads,
        key_dim=linear_key_dim,
        value_dim=linear_value_dim,
    )
    recurrent_ledger = recurrent_state_ledger(
        layer_count=len(linear_layers),
        heads=linear_value_heads,
        key_dim=linear_key_dim,
        value_dim=linear_value_dim,
        session_tokens=session_tokens,
    )
    recurrent_artifacts = _write_component_artifacts(
        root=spec.root, model=spec.identifier, component="deltanet_recurrent_state", values=recurrent_values
    )
    recurrent_receipt_path = spec.root / "QWEN80_DELTANET_STATE_CODEC_RECEIPT.json"
    recurrent_receipt = _write_receipt(
        recurrent_receipt_path,
        {
            "schema": COMPONENT_RECEIPT_SCHEMA,
            "recorded_at": _utc_now(),
            "model": {"id": "Qwen3-Coder-Next-80B", "architecture": spec.architecture},
            "source_verification": _source_summary(spec, audit, config),
            "source_projection_bindings": [
                _raw_tensor_binding(spec.model_dir, weight_map, qkvz_name, qkvz_weights, audit),
                _raw_tensor_binding(spec.model_dir, weight_map, a_log_name, a_log, audit),
            ],
            "component": {
                "kind": "gated_deltanet_recurrent_state_proxy",
                "source_layer": linear_layer,
                "source_tensor_names": [qkvz_name, a_log_name],
                "linear_attention_layers_derived_from_local_index": list(linear_layers),
                "output_shape": list(recurrent_values.shape),
                "output_float32_sha256": _sha256_array(recurrent_values),
                "output_semantics": "bounded decayed grouped outer-product state witness from real Q/K/V/Z projections and A_log",
                "proxy_details": proxy_details,
            },
            "deterministic_component_input": {
                "shape": list(hidden_linear.shape),
                "float32_sha256": _sha256_array(hidden_linear),
                "generator": "normalized_sine_cosine_indexed_by_model_component_label",
            },
            "exact_geometry_ledger": recurrent_ledger,
            "codec_results": recurrent_artifacts,
            "claim_boundary": _common_claim_boundary(),
        },
    )

    combined: dict[str, Any] = {}
    for codec_name in _codec_names():
        attention_row = attention_ledger["codecs"][codec_name]
        recurrent_row = recurrent_ledger["codecs"][codec_name]
        session_bytes = attention_row["bytes_per_session"] + recurrent_row["bytes_per_session_resident_state"]
        combined[codec_name] = {
            "attention_kv_bytes_per_session": attention_row["bytes_per_session"],
            "deltanet_recurrent_bytes_per_session": recurrent_row["bytes_per_session_resident_state"],
            "combined_bytes_per_session": session_bytes,
            "combined_bytes_per_token_amortized_at_declared_session": session_bytes / session_tokens,
            "deltanet_growth_bytes_per_additional_token": 0,
        }
    return {
        "receipt_path": str(attention_receipt_path),
        "receipt_seal_sha256": attention_receipt["seal_sha256"],
        "deltanet_receipt_path": str(recurrent_receipt_path),
        "deltanet_receipt_seal_sha256": recurrent_receipt["seal_sha256"],
        "component_count": 2,
        "artifact_count": len(attention_artifacts) + len(recurrent_artifacts),
        "geometry": {
            "attention_layers": list(attention_layers),
            "linear_attention_layers": list(linear_layers),
            "attention_kv": attention_ledger,
            "deltanet_recurrent_state": recurrent_ledger,
            "combined": {"session_tokens": session_tokens, "codecs": combined},
        },
    }


def run_model(
    model: str,
    *,
    model_dir: Path | None = None,
    root: Path | None = None,
    audit_path: Path | None = None,
    session_tokens: int = DEFAULT_SESSION_TOKENS,
) -> dict[str, Any]:
    """Run one bounded physical state codec lane and publish its sealed status."""

    if model not in SPECS:
        raise StateKVError(f"unknown model {model!r}; expected one of {sorted(SPECS)}")
    if session_tokens <= 0:
        raise StateKVError("session token count must be positive")
    base = SPECS[model]
    spec = ModelSpec(
        identifier=base.identifier,
        architecture=base.architecture,
        model_dir=(model_dir or base.model_dir).expanduser().resolve(),
        root=(root or base.root).expanduser().resolve(),
        audit_path=(audit_path or base.audit_path).expanduser().resolve(),
    )
    if model == "qwen30":
        outcome = _run_qwen30(spec, session_tokens=session_tokens)
        status_name = "QWEN30_STATE_KV_STATUS.json"
    else:
        outcome = _run_qwen80(spec, session_tokens=session_tokens)
        status_name = "QWEN80_STATE_KV_STATUS.json"
    status_path = spec.root / status_name
    prior = _read_json(status_path) or {}
    status = _write_receipt(
        status_path,
        {
            "schema": STATUS_SCHEMA,
            "recorded_at": _utc_now(),
            "model": model,
            "heartbeat": int(prior.get("heartbeat", 0)) + 1,
            "phase": "COMPLETE_SOURCE_BOUND_COMPONENT_STATE_CODEC_RESEARCH",
            "session_tokens": session_tokens,
            "outcome": outcome,
            "claim_boundary": _common_claim_boundary(),
        },
    )
    return status


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("qwen30", "qwen80", "both"))
    parser.add_argument("--session-tokens", type=int, default=DEFAULT_SESSION_TOKENS)
    parser.add_argument("--model-dir", type=Path, help="override only when running one model")
    parser.add_argument("--root", type=Path, help="override only when running one model")
    parser.add_argument("--audit-path", type=Path, help="override only when running one model")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.model == "both":
        if args.model_dir or args.root or args.audit_path:
            raise StateKVError("model/root/audit overrides require a single model run")
        results = {model: run_model(model, session_tokens=args.session_tokens) for model in ("qwen30", "qwen80")}
    else:
        results = {
            args.model: run_model(
                args.model,
                model_dir=args.model_dir,
                root=args.root,
                audit_path=args.audit_path,
                session_tokens=args.session_tokens,
            )
        }
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    try:
        raise SystemExit(main())
    except StateKVError as exc:
        raise SystemExit(f"state-kv research failed: {exc}") from exc
