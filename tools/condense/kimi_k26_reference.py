#!/usr/bin/env python3.12
"""Bounded-memory Apple/MLX text-core reference for official Kimi K2.6 shards.

The source layout is unusually instrument-friendly: shard 1 is dense decoder
layer 0, shards 2..61 each hold one complete MoE decoder layer, shard 62 holds
embedding/final norm/LM head, and shards 63..64 hold multimodal-only weights.
This reference maps one text shard at a time.  It evaluates routed experts only
after the official router chooses them, and sends native compressed-tensors
INT4 directly to MLX's Metal quantized matmul with the exact signed-offset
affine correction.  It never constructs a full dequantized model.

This is a scientific reference, not a serving runtime.  Parent coherence and
determinism are reported separately from official-runtime parity; parity is not
claimed unless an independent runtime comparison is supplied.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
from typing import Any, Callable

import ml_dtypes
import mlx.core as mx
import numpy as np
import tiktoken


REPO = "moonshotai/Kimi-K2.6"
REVISION = "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"
PREFIX = "language_model.model"
MLX_CACHE_LIMIT_BYTES = 4 * 1024**3
MLX_MEMORY_LIMIT_BYTES = 48 * 1024**3
DTYPES = {
    "BF16": ml_dtypes.bfloat16, "F16": np.float16, "F32": np.float32,
    "I32": np.int32, "I64": np.int64, "U8": np.uint8, "I8": np.int8,
}


class ReferenceError(RuntimeError):
    pass


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    os.replace(temporary, path)
    return digest


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


class TensorShard:
    """Header plus lazy mmap views for one immutable safetensors shard."""

    def __init__(self, path: Path):
        self.path = path.resolve(strict=True)
        with self.path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) != 8:
                raise ReferenceError(f"short safetensors header: {self.path.name}")
            self.header_bytes = struct.unpack("<Q", raw)[0]
            if self.header_bytes <= 0 or self.header_bytes > 512 * 1024**2:
                raise ReferenceError(f"unsafe header size: {self.path.name}")
            self.header = json.loads(handle.read(self.header_bytes))
        self.data_base = 8 + self.header_bytes
        self.accessed: set[str] = set()

    def names(self) -> set[str]:
        return set(self.header) - {"__metadata__"}

    def numpy(self, name: str, *, unsigned_packed: bool = False) -> np.ndarray:
        try:
            info = self.header[name]
        except KeyError as exc:
            raise ReferenceError(f"tensor absent from {self.path.name}: {name}") from exc
        dtype_name = info["dtype"]
        if dtype_name not in DTYPES:
            raise ReferenceError(f"unsupported dtype {dtype_name}: {name}")
        start, end = (int(v) for v in info["data_offsets"])
        dtype = np.dtype(DTYPES[dtype_name])
        expected = math.prod(int(v) for v in info["shape"]) * dtype.itemsize
        if end - start != expected:
            raise ReferenceError(f"tensor byte geometry mismatch: {name}")
        self.accessed.add(name)
        value = np.memmap(self.path, mode="r", dtype=dtype,
                          offset=self.data_base + start, shape=tuple(info["shape"]))
        if unsigned_packed:
            if dtype_name != "I32":
                raise ReferenceError(f"packed tensor is not I32: {name}")
            return value.view(np.uint32)
        return value

    def mlx(self, name: str) -> mx.array:
        return mx.array(self.numpy(name))


class KimiTokenizer:
    def __init__(self, root: Path):
        # ``tiktoken.load.load_tiktoken_bpe`` routes even local paths through the
        # optional blobfile package.  The official asset is the standard two-column
        # base64/rank format, so parse it directly and keep this instrument offline.
        ranks: dict[bytes, int] = {}
        with (root / "tiktoken.model").open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                try:
                    token, rank = raw.split()
                    decoded = base64.b64decode(token, validate=True)
                    value = int(rank)
                except (ValueError, TypeError) as exc:
                    raise ReferenceError(
                        f"malformed official tiktoken row {line_number}"
                    ) from exc
                if decoded in ranks or value != len(ranks):
                    raise ReferenceError(
                        f"non-canonical official tiktoken rank at row {line_number}"
                    )
                ranks[decoded] = value
        config = read_json(root / "tokenizer_config.json")
        decoder = config.get("added_tokens_decoder") or {}
        base = len(ranks)
        special = {
            (decoder.get(str(index)) or {}).get("content", f"<|reserved_token_{index}|>"): index
            for index in range(base, base + 256)
        }
        # Keep the exact official regular-expression grammar from tokenization_kimi.py.
        pattern = "|".join([
            r"[\p{Han}]+",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"\p{N}{1,3}", r" ?[^\s\p{L}\p{N}]+[\r\n]*", r"\s*[\r\n]+",
            r"\s+(?!\S)", r"\s+",
        ])
        self.encoding = tiktoken.Encoding(name="Kimi-K2.6", pat_str=pattern,
                                          mergeable_ranks=ranks, special_tokens=special)
        self.special = special

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text, allowed_special="all")

    def decode(self, ids: list[int]) -> str:
        return self.encoding.decode(ids)

    @staticmethod
    def user_prompt(text: str, *, thinking: bool = False) -> str:
        suffix = "<think>" if thinking else "<think></think>"
        return (f"<|im_user|>user<|im_middle|>{text}<|im_end|>"
                f"<|im_assistant|>assistant<|im_middle|>{suffix}")


def rms_norm(x: mx.array, weight: mx.array, epsilon: float) -> mx.array:
    x32 = x.astype(mx.float32)
    normalized = x32 * mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + epsilon)
    return (normalized.astype(x.dtype) * weight).astype(mx.bfloat16)


def linear(x: mx.array, weight: mx.array) -> mx.array:
    return mx.matmul(x.astype(weight.dtype), mx.swapaxes(weight, -1, -2))


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def dense_mlp(x: mx.array, shard: TensorShard, base: str) -> mx.array:
    gate = shard.mlx(base + ".gate_proj.weight")
    up = shard.mlx(base + ".up_proj.weight")
    down = shard.mlx(base + ".down_proj.weight")
    output = linear(silu(linear(x, gate)) * linear(x, up), down)
    mx.eval(output)
    return output


def quantized_linear(x: mx.array, shard: TensorShard, base: str) -> mx.array:
    packed_view = shard.numpy(base + ".weight_packed", unsigned_packed=True)
    original_shape = [int(v) for v in np.asarray(
        shard.numpy(base + ".weight_shape"), dtype=np.int32).reshape(-1)]
    expected_packed_last = (original_shape[-1] * 4 + 31) // 32
    if (list(packed_view.shape[:-1]) != original_shape[:-1] or
            packed_view.shape[-1] != expected_packed_last):
        raise ReferenceError(f"native INT4 pack geometry mismatch: {base}")
    packed = mx.array(packed_view)
    scales = shard.mlx(base + ".weight_scale")
    # compressed-tensors stores signed q in offset binary (u=q+8).  MLX affine computes
    # u*scale+bias, therefore bias=-8*scale exactly reconstructs q*scale.
    return mx.quantized_matmul(x, packed, scales, -8 * scales,
                               transpose=True, group_size=32, bits=4, mode="affine")


def yarn_get_mscale(scale: float, mscale: float) -> float:
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def yarn_correction_dim(rotations: float, dimension: int, base: float,
                        original_context: int) -> float:
    return (dimension * math.log(original_context / (rotations * 2 * math.pi)) /
            (2 * math.log(base)))


def yarn_cos_sin(sequence: int, dimension: int, *, base: float, factor: float,
                 original_context: int, beta_fast: float, beta_slow: float,
                 mscale: float, mscale_all_dim: float,
                 positions: np.ndarray | None = None) -> tuple[mx.array, mx.array]:
    exponent = np.arange(0, dimension, 2, dtype=np.float32) / dimension
    extra = 1.0 / np.power(base, exponent)
    interpolated = 1.0 / (factor * np.power(base, exponent))
    low = max(math.floor(yarn_correction_dim(beta_fast, dimension, base, original_context)), 0)
    high = min(math.ceil(yarn_correction_dim(beta_slow, dimension, base, original_context)),
               dimension - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(dimension // 2, dtype=np.float32) - low) / (high - low), 0, 1)
    inverse_mask = 1.0 - ramp
    inverse_frequency = interpolated * (1 - inverse_mask) + extra * inverse_mask
    position_values = (np.arange(sequence, dtype=np.float32) if positions is None else
                       np.asarray(positions, dtype=np.float32))
    if position_values.shape != (sequence,):
        raise ReferenceError("RoPE position vector does not match sequence length")
    frequencies = np.outer(position_values, inverse_frequency)
    embedding = np.concatenate((frequencies, frequencies), axis=-1)
    amplitude = yarn_get_mscale(factor, mscale) / yarn_get_mscale(factor, mscale_all_dim)
    return mx.array(np.cos(embedding) * amplitude), mx.array(np.sin(embedding) * amplitude)


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def apply_rope(x: mx.array, cosine: mx.array, sine: mx.array) -> mx.array:
    # x: heads x sequence x rope_dimension; cos/sin: sequence x rope_dimension
    return x * cosine[None, :, :] + rotate_half(x) * sine[None, :, :]


def official_rope_layout(x: mx.array) -> mx.array:
    """Match the official interleaved-pair to half-split permutation before RoPE."""
    heads, sequence, dimension = x.shape
    return mx.reshape(mx.swapaxes(
        mx.reshape(x, (heads, sequence, dimension // 2, 2)), -1, -2
    ), (heads, sequence, dimension))


def attention(x: mx.array, shard: TensorShard, layer: int,
              config: dict[str, Any],
              segment_lengths: list[int] | None = None) -> tuple[mx.array, dict[str, Any]]:
    base = f"{PREFIX}.layers.{layer}.self_attn"
    heads = int(config["num_attention_heads"])
    q_nope_dim = int(config["qk_nope_head_dim"])
    rope_dim = int(config["qk_rope_head_dim"])
    value_dim = int(config["v_head_dim"])
    q_rank = int(config["q_lora_rank"])
    kv_rank = int(config["kv_lora_rank"])
    sequence = x.shape[0]
    q_low = linear(x, shard.mlx(base + ".q_a_proj.weight"))
    q_low = rms_norm(q_low, shard.mlx(base + ".q_a_layernorm.weight"),
                     float(config["rms_norm_eps"]))
    query = linear(q_low, shard.mlx(base + ".q_b_proj.weight"))
    query = mx.reshape(query, (sequence, heads, q_nope_dim + rope_dim))
    query = mx.transpose(query, (1, 0, 2))
    q_nope, q_pe = mx.split(query, [q_nope_dim], axis=-1)

    compressed = linear(x, shard.mlx(base + ".kv_a_proj_with_mqa.weight"))
    kv_low, k_pe = mx.split(compressed, [kv_rank], axis=-1)
    kv_low = rms_norm(kv_low, shard.mlx(base + ".kv_a_layernorm.weight"),
                      float(config["rms_norm_eps"]))
    key_value = linear(kv_low, shard.mlx(base + ".kv_b_proj.weight"))
    key_value = mx.transpose(mx.reshape(
        key_value, (sequence, heads, q_nope_dim + value_dim)), (1, 0, 2))
    k_nope, values = mx.split(key_value, [q_nope_dim], axis=-1)
    k_pe = mx.transpose(mx.reshape(k_pe, (sequence, 1, rope_dim)), (1, 0, 2))

    rope = config["rope_scaling"]
    segment_lengths = segment_lengths or [sequence]
    if sum(segment_lengths) != sequence:
        raise ReferenceError("attention segment lengths do not cover the token sequence")
    positions = np.concatenate([np.arange(length, dtype=np.float32)
                                for length in segment_lengths])
    cosine, sine = yarn_cos_sin(
        sequence, rope_dim, base=float(config["rope_theta"]), factor=float(rope["factor"]),
        original_context=int(rope["original_max_position_embeddings"]),
        beta_fast=float(rope["beta_fast"]), beta_slow=float(rope["beta_slow"]),
        mscale=float(rope.get("mscale", 1.0)),
        mscale_all_dim=float(rope.get("mscale_all_dim", 0.0)),
        positions=positions,
    )
    cosine = cosine.astype(q_pe.dtype)
    sine = sine.astype(q_pe.dtype)
    q_pe = apply_rope(official_rope_layout(q_pe), cosine, sine)
    k_pe = apply_rope(official_rope_layout(k_pe), cosine, sine)
    k_pe = mx.broadcast_to(k_pe, (heads, sequence, rope_dim))
    queries = mx.concatenate((q_nope, q_pe), axis=-1)
    keys = mx.concatenate((k_nope, k_pe), axis=-1)
    scale = (q_nope_dim + rope_dim) ** -0.5
    if rope.get("mscale_all_dim"):
        scale *= yarn_get_mscale(float(rope["factor"]),
                                 float(rope["mscale_all_dim"])) ** 2
    # The official reference explicitly upcasts attention weights for softmax.  The
    # bounded probe envelope is <=64 tokens, so materializing this small score matrix
    # is both more faithful and cheaper than depending on a fused-kernel dtype policy.
    scores = mx.matmul(queries.astype(mx.float32),
                       mx.swapaxes(keys.astype(mx.float32), -1, -2)) * scale
    causal = np.full((sequence, sequence), -np.inf, dtype=np.float32)
    offset = 0
    for length in segment_lengths:
        causal[offset:offset + length, offset:offset + length] = np.triu(
            np.full((length, length), -np.inf, dtype=np.float32), 1
        )
        offset += length
    probabilities = mx.softmax(scores + mx.array(causal)[None, :, :], axis=-1)
    attended = mx.matmul(probabilities.astype(values.dtype), values)
    attended = mx.reshape(mx.transpose(attended, (1, 0, 2)),
                          (sequence, heads * value_dim))
    output = linear(attended, shard.mlx(base + ".o_proj.weight"))
    mx.eval(output)
    return output, {"heads": heads, "sequence": sequence, "query_dim": q_nope_dim + rope_dim,
                    "value_dim": value_dim, "q_rank": q_rank, "kv_rank": kv_rank,
                    "softmax_scale": scale, "segment_lengths": segment_lengths}


def routed_moe(x: mx.array, shard: TensorShard, layer: int,
               config: dict[str, Any]) -> tuple[mx.array, dict[str, Any]]:
    base = f"{PREFIX}.layers.{layer}.mlp"
    router_weight = shard.mlx(base + ".gate.weight")
    correction = shard.mlx(base + ".gate.e_score_correction_bias")
    logits = linear(x.astype(mx.float32), router_weight.astype(mx.float32))
    scores = mx.sigmoid(logits)
    mx.eval(scores)
    scores_np = np.asarray(scores, dtype=np.float32)
    choice = scores_np + np.asarray(correction.astype(mx.float32))[None, :]
    experts = int(config["n_routed_experts"])
    groups = int(config["n_group"])
    top_groups = int(config["topk_group"])
    top_k = int(config["num_experts_per_tok"])
    grouped = choice.reshape(choice.shape[0], groups, experts // groups)
    group_score = np.sort(grouped, axis=-1)[..., -2:].sum(axis=-1)
    allowed_group = np.argpartition(group_score, -top_groups, axis=-1)[:, -top_groups:]
    group_mask = np.zeros_like(group_score, dtype=bool)
    np.put_along_axis(group_mask, allowed_group, True, axis=-1)
    allowed = np.repeat(group_mask, experts // groups, axis=-1)
    filtered = np.where(allowed, choice, 0.0)
    indices = np.argpartition(filtered, -top_k, axis=-1)[:, -top_k:]
    route_weight = np.take_along_axis(scores_np, indices, axis=-1)
    if config["norm_topk_prob"]:
        route_weight /= route_weight.sum(axis=-1, keepdims=True) + 1e-20
    route_weight *= float(config["routed_scaling_factor"])

    combined = mx.zeros_like(x)
    used = sorted(set(int(v) for v in indices.flat))
    sequence = x.shape[0]
    for expert in used:
        tokens, slots = np.where(indices == expert)
        selected = mx.take(x, mx.array(tokens, dtype=mx.int32), axis=0)
        expert_base = base + f".experts.{expert}"
        gate = quantized_linear(selected, shard, expert_base + ".gate_proj")
        up = quantized_linear(selected, shard, expert_base + ".up_proj")
        hidden = silu(gate) * up
        expert_output = quantized_linear(hidden, shard, expert_base + ".down_proj")
        weights = mx.array(route_weight[tokens, slots], dtype=expert_output.dtype)[:, None]
        scatter = np.eye(sequence, dtype=np.float32)[tokens].T
        combined = combined + mx.array(scatter, dtype=expert_output.dtype) @ (expert_output * weights)
        # Force completion here so the lazy graph never retains dozens of expert mmaps at once.
        mx.eval(combined)
    shared = dense_mlp(x, shard, base + ".shared_experts")
    output = combined + shared
    mx.eval(output)
    return output, {"used_experts": used, "used_expert_count": len(used),
                    "route_indices": indices.tolist(),
                    "route_weight_sums": route_weight.sum(axis=-1).tolist()}


def layer_forward(hidden: mx.array, shard: TensorShard, layer: int,
                  config: dict[str, Any],
                  segment_lengths: list[int] | None = None) -> tuple[mx.array, dict[str, Any]]:
    layer_base = f"{PREFIX}.layers.{layer}"
    normalized = rms_norm(hidden, shard.mlx(layer_base + ".input_layernorm.weight"),
                          float(config["rms_norm_eps"]))
    attn, attn_info = attention(normalized, shard, layer, config, segment_lengths)
    hidden = (hidden + attn).astype(mx.bfloat16)
    normalized = rms_norm(hidden, shard.mlx(layer_base + ".post_attention_layernorm.weight"),
                          float(config["rms_norm_eps"]))
    if layer == 0:
        feed = dense_mlp(normalized, shard, layer_base + ".mlp")
        moe_info: dict[str, Any] = {"kind": "dense", "used_expert_count": 0}
    else:
        feed, moe_info = routed_moe(normalized, shard, layer, config)
        moe_info["kind"] = "routed_plus_shared"
    hidden = (hidden + feed).astype(mx.bfloat16)
    mx.eval(hidden)
    unaccessed = shard.names() - shard.accessed
    conditionally_inactive = {
        name for name in unaccessed if ".mlp.experts." in name
    }
    unaccounted = sorted(unaccessed - conditionally_inactive)
    if unaccounted:
        raise ReferenceError(
            f"silent tensor omission in layer {layer}: {unaccounted[:4]}"
        )
    hidden_f32 = np.asarray(hidden.astype(mx.float32))
    segment_hashes = []
    offset = 0
    for length in segment_lengths or [hidden.shape[0]]:
        segment_hashes.append(hashlib.sha256(
            hidden_f32[offset:offset + length].tobytes()).hexdigest())
        offset += length
    return hidden, {"attention": attn_info, "moe": moe_info,
                    "tensor_audit": {
                        "source_tensors": len(shard.names()),
                        "accessed_tensors": len(shard.accessed),
                        "conditionally_inactive_routed_expert_tensors":
                            len(conditionally_inactive),
                        "unaccounted_tensors": [],
                    },
                    "finite": bool(np.isfinite(hidden_f32).all()),
                    "hidden_sha256": hashlib.sha256(hidden_f32.tobytes()).hexdigest(),
                    "segment_hidden_sha256": segment_hashes}


def shard_path(root: Path, number: int) -> Path:
    return root / f"model-{number:05d}-of-000064.safetensors"


def source_identity(root: Path) -> dict[str, Any]:
    config_hash = hashlib.sha256((root / "config.json").read_bytes()).hexdigest()
    index_hash = hashlib.sha256((root / "model.safetensors.index.json").read_bytes()).hexdigest()
    return {"repo": REPO, "revision": REVISION, "snapshot": str(root),
            "config_sha256": config_hash, "index_sha256": index_hash,
            "shards_present": sum(path.is_file() for path in root.glob("model-*.safetensors"))}


def deterministic_signature(result: dict[str, Any]) -> str:
    stable = {
        "source": result["source"], "token_ids": result["token_ids"],
        "top_token_ids": result["top_token_ids"],
        "top_logits_f32_hex": np.asarray(
            result["top_logits"], dtype=np.float32).tobytes().hex(),
        "layer_hidden_sha256": [layer["hidden_sha256"] for layer in result["layers"]],
        "layer_routes": [layer["moe"].get("route_indices", [])
                         for layer in result["layers"]],
    }
    return hashlib.sha256(json.dumps(
        stable, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()


def top_logits_batch(hidden: mx.array, tail: TensorShard, positions: list[int], *,
                     chunk: int = 4096, top_k: int = 20
                     ) -> list[tuple[list[int], list[float], dict[str, float]]]:
    weight = tail.numpy("language_model.lm_head.weight")
    last = mx.take(hidden, mx.array(positions, dtype=mx.int32), axis=0).astype(mx.bfloat16)
    pieces = []
    for start in range(0, weight.shape[0], chunk):
        matrix = mx.array(np.asarray(weight[start:start + chunk]))
        logits = linear(last, matrix).astype(mx.float32)
        mx.eval(logits)
        pieces.append(np.asarray(logits))
    all_logits = np.concatenate(pieces, axis=-1)
    if not np.isfinite(all_logits).all():
        raise ReferenceError("non-finite final logits")
    results = []
    for row in all_logits:
        ids = np.argpartition(row, -top_k)[-top_k:]
        ids = ids[np.argsort(row[ids])[::-1]]
        shifted = row - np.max(row)
        probability = np.exp(shifted)
        probability /= probability.sum()
        entropy = float(-np.sum(probability * np.log(probability + 1e-30)))
        results.append((ids.tolist(), row[ids].astype(float).tolist(), {
            "min": float(row.min()), "max": float(row.max()),
            "mean": float(row.mean()), "std": float(row.std()), "entropy": entropy,
        }))
    return results


def top_logits(hidden: mx.array, tail: TensorShard, *, chunk: int = 4096,
               top_k: int = 20) -> tuple[list[int], list[float], dict[str, float]]:
    return top_logits_batch(hidden, tail, [hidden.shape[0] - 1],
                            chunk=chunk, top_k=top_k)[0]


def forward_batch(root: Path, requests: list[dict[str, Any]], *,
                  progress: Callable[[int, dict[str, Any]], None] | None = None,
                  checkpoint_dir: Path | None = None) -> list[dict[str, Any]]:
    started = time.time()
    config = read_json(root / "config.json")["text_config"]
    tokenizer = KimiTokenizer(root)
    prepared = []
    for request in requests:
        text = str(request["text"])
        chat, thinking = bool(request.get("chat")), bool(request.get("thinking"))
        rendered = tokenizer.user_prompt(text, thinking=thinking) if chat else text
        ids = tokenizer.encode(rendered)
        if not ids or len(ids) > 64:
            raise ReferenceError(f"probe token length outside bounded 1..64 envelope: {len(ids)}")
        prepared.append({**request, "rendered": rendered, "token_ids": ids,
                         "chat": chat, "thinking": thinking})
    segment_lengths = [len(item["token_ids"]) for item in prepared]
    token_ids = [token for item in prepared for token in item["token_ids"]]
    checkpoint = read_json(checkpoint_dir / "checkpoint.json") if checkpoint_dir else {}
    start_layer = 0
    layers: list[dict[str, Any]] = []
    if (checkpoint.get("revision") == REVISION and checkpoint.get("token_ids") == token_ids and
            checkpoint.get("segment_lengths") == segment_lengths and
            isinstance(checkpoint.get("completed_layer"), int) and checkpoint_dir and
            len(checkpoint.get("layers", [])) == int(checkpoint["completed_layer"]) + 1):
        hidden_path = checkpoint_dir / "hidden.npy"
        if (hidden_path.exists() and
                hashlib.sha256(hidden_path.read_bytes()).hexdigest() ==
                checkpoint.get("hidden_npy_sha256")):
            hidden = mx.array(np.load(hidden_path, allow_pickle=False)).astype(mx.bfloat16)
            mx.eval(hidden)
            layers = checkpoint.get("layers", [])
            start_layer = int(checkpoint["completed_layer"]) + 1
    if start_layer == 0:
        tail = TensorShard(shard_path(root, 62))
        embedding = tail.numpy(f"{PREFIX}.embed_tokens.weight")
        hidden = mx.array(np.asarray(embedding[token_ids])).astype(mx.bfloat16)
        mx.eval(hidden)
        del embedding, tail
        gc.collect()
        mx.clear_cache()
    for layer in range(start_layer, int(config["num_hidden_layers"])):
        per_layer_started = time.time()
        shard = TensorShard(shard_path(root, layer + 1))
        expected_prefix = f"{PREFIX}.layers.{layer}."
        if not any(name.startswith(expected_prefix) for name in shard.names()):
            raise ReferenceError(f"layer/shard order mismatch at layer {layer}")
        hidden, info = layer_forward(hidden, shard, layer, config, segment_lengths)
        info.update({"layer": layer, "shard": shard.path.name,
                     "seconds": time.time() - per_layer_started})
        layers.append(info)
        if checkpoint_dir:
            hidden_array = np.asarray(hidden.astype(mx.float32))
            hidden_hash = atomic_npy(checkpoint_dir / "hidden.npy", hidden_array)
            atomic_json(checkpoint_dir / "checkpoint.json", {
                "revision": REVISION, "token_ids": token_ids,
                "segment_lengths": segment_lengths, "completed_layer": layer,
                "hidden_npy_sha256": hidden_hash, "layers": layers, "updated_at": now(),
            })
        if progress:
            progress(layer, info)
        del shard
        gc.collect()
        mx.clear_cache()
    tail = TensorShard(shard_path(root, 62))
    hidden = rms_norm(hidden, tail.mlx(f"{PREFIX}.norm.weight"), float(config["rms_norm_eps"]))
    final_positions = (np.cumsum(segment_lengths) - 1).astype(int).tolist()
    logits = top_logits_batch(hidden, tail, final_positions)
    expected_tail = {
        f"{PREFIX}.embed_tokens.weight", f"{PREFIX}.norm.weight",
        "language_model.lm_head.weight",
    }
    if tail.names() != expected_tail:
        raise ReferenceError(
            f"text boundary shard has unaccounted tensors: {sorted(tail.names() ^ expected_tail)}"
        )
    identity = source_identity(root)
    results = []
    offset = 0
    for segment_index, (item, logit_row) in enumerate(zip(prepared, logits, strict=True)):
        length = segment_lengths[segment_index]
        segment_layers = []
        for shared_layer in layers:
            layer = json.loads(json.dumps(shared_layer))
            layer["hidden_sha256"] = layer.pop("segment_hidden_sha256")[segment_index]
            layer["attention"]["batch_total_sequence"] = layer["attention"]["sequence"]
            layer["attention"]["sequence"] = length
            layer["attention"]["segment_lengths"] = [length]
            route_indices = layer["moe"].get("route_indices", [])[offset:offset + length]
            route_sums = layer["moe"].get("route_weight_sums", [])[offset:offset + length]
            if route_indices:
                layer["moe"]["route_indices"] = route_indices
                layer["moe"]["route_weight_sums"] = route_sums
                layer["moe"]["used_experts"] = sorted(
                    {int(expert) for token in route_indices for expert in token}
                )
                layer["moe"]["used_expert_count"] = len(layer["moe"]["used_experts"])
            segment_layers.append(layer)
        ids, values, stats = logit_row
        result = {
            "schema": "hawking.kimi_k26.reference_forward_probe.v1", "status": "PASS",
            "source": identity, "input_text": item["text"],
            "rendered_prompt": item["rendered"], "chat_protocol": item["chat"],
            "thinking": item["thinking"], "token_ids": item["token_ids"],
            "token_count": length, "finite_logits": True, "top_token_ids": ids,
            "top_token_text": [tokenizer.decode([token]) for token in ids],
            "top_logits": values, "logit_stats": stats, "layers": segment_layers,
            "runtime_seconds": time.time() - started,
            "runtime": "MLX_METAL_NATIVE_INT4_SELECTED_EXPERTS_BATCHED_BLOCK_DIAGONAL",
            "execution_layout": "BATCHED_BLOCK_DIAGONAL",
            "batch_probe_count": len(prepared), "batch_total_tokens": len(token_ids),
            "source_layer_reads_for_batch": 61,
            "mlx_cache_limit_bytes": MLX_CACHE_LIMIT_BYTES,
            "mlx_memory_limit_bytes": MLX_MEMORY_LIMIT_BYTES,
            "complete_model_dequantized": False, "max_source_shards_mapped_at_once": 1,
            "tensor_omission_audit": {
                "layer_unaccounted_tensors": 0, "text_boundary_unaccounted_tensors": 0,
                "unselected_routed_expert_tensors_are_conditionally_inactive": True,
            },
            "vision_executed": False, "official_runtime_parity_claimed": False,
        }
        result["deterministic_signature_sha256"] = deterministic_signature(result)
        result["result_sha256"] = hashlib.sha256(json.dumps(
            result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        results.append(result)
        offset += length
    return results


def forward_prompt(root: Path, text: str, *, progress: Callable[[int, dict[str, Any]], None] | None = None,
                   chat: bool = False, thinking: bool = False,
                   checkpoint_dir: Path | None = None) -> dict[str, Any]:
    return forward_batch(root, [{"text": text, "chat": chat, "thinking": thinking}],
                         progress=progress, checkpoint_dir=checkpoint_dir)[0]


PROBES = [
    {"id": "factual", "text": "The capital of France is", "expected": ["Paris", " Paris"]},
    {"id": "science", "text": "At standard pressure, water freezes at", "expected": ["0", " zero"]},
    {"id": "coding", "text": "def add(a, b):\n    return", "expected": [" a", "a"]},
    {"id": "mathematics", "text": "2 + 2 =", "expected": ["4", " 4"]},
    {"id": "reasoning", "text": "All cats are mammals. Miso is a cat. Therefore Miso is a",
     "expected": [" mammal", "mammal"]},
    {"id": "instruction", "text": "Reply with exactly OK.", "expected": ["OK", " OK"],
     "chat": True, "thinking": False},
    {"id": "tool_thinking_protocol", "text": "Think briefly, then answer: what is 1+1?",
     "expected": ["2", " 2", "\n"], "chat": True, "thinking": True},
    {"id": "rare_token", "text": "The expression ⟨∇ψ, φ⟩ is", "expected": []},
]


def run_suite(root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    by_id: dict[str, dict[str, Any]] = {}
    pending = []
    for probe in PROBES:
        path = output / f"probe_{probe['id']}.json"
        cached = read_json(path)
        if (cached.get("status") == "PASS" and
                cached.get("source", {}).get("revision") == REVISION and
                cached.get("deterministic_signature_sha256") and
                cached.get("execution_layout") == "BATCHED_BLOCK_DIAGONAL"):
            by_id[probe["id"]] = cached
        else:
            pending.append(probe)

    if pending:
        layer_state = output / "active_probe.json"

        def record(layer: int, info: dict[str, Any]) -> None:
            atomic_json(layer_state, {"probe": f"batch_{len(pending)}_probes", "layer": layer,
                                      "layers_total": 61, "updated_at": now(),
                                      "shard": info["shard"],
                                      "hidden_sha256": info["hidden_sha256"]})

        batch_key = hashlib.sha256(json.dumps(
            [probe["id"] for probe in pending], separators=(",", ":")
        ).encode()).hexdigest()[:16]
        batch_results = forward_batch(
            root, pending, progress=record,
            checkpoint_dir=output / f"checkpoint_batch_{batch_key}",
        )
        for probe, result in zip(pending, batch_results, strict=True):
            result["probe_id"] = probe["id"]
            expected = probe["expected"]
            result["expected_fragments"] = expected
            result["coherent_next_token"] = (
                not expected or any(fragment in result["top_token_text"] for fragment in expected)
            )
            atomic_json(output / f"probe_{probe['id']}.json", result)
            by_id[probe["id"]] = result
    results = [by_id[probe["id"]] for probe in PROBES]
    replay_target = next(result for result in results if result.get("probe_id") == "mathematics")
    def replay_record(layer: int, info: dict[str, Any]) -> None:
        atomic_json(output / "active_probe.json", {
            "probe": "batch_replay_8_probes", "layer": layer, "layers_total": 61,
            "updated_at": now(), "shard": info["shard"],
            "hidden_sha256": info["hidden_sha256"],
        })

    replay_results = forward_batch(root, PROBES, progress=replay_record,
                                   checkpoint_dir=output / "checkpoint_batch_replay")
    replay_by_id = dict(zip((probe["id"] for probe in PROBES), replay_results, strict=True))
    replay = replay_by_id["mathematics"]
    replay["probe_id"] = "mathematics_replay"
    replay_match = all(
        by_id[probe["id"]]["deterministic_signature_sha256"] ==
        replay_by_id[probe["id"]]["deterministic_signature_sha256"]
        for probe in PROBES
    )
    atomic_json(output / "probe_mathematics_replay.json", replay)
    finite = all(result.get("finite_logits") for result in results)
    coherent = sum(bool(result.get("coherent_next_token")) for result in results)
    suite = {
        "schema": "hawking.kimi_k26.parent_forward_validation.v1",
        "status": "PASS" if finite and coherent == len(results) and replay_match else "FAIL",
        "sealed_at": now(), "source": source_identity(root), "probe_count": len(results),
        "finite_probe_count": sum(bool(result.get("finite_logits")) for result in results),
        "coherent_probe_count": coherent,
        "deterministic_replay": {
            "status": "PASS" if replay_match else "FAIL",
            "comparison": "all eight block-diagonal probes replayed; exact signature over "
                          "source, tokens, float32 top logits, every layer hidden hash, and "
                          "every router selection",
            "probe_signatures_compared": len(PROBES),
            "first_signature_sha256": replay_target["deterministic_signature_sha256"],
            "replay_signature_sha256": replay["deterministic_signature_sha256"],
        },
        "official_runtime_comparison": "NOT_FEASIBLE_ON_THIS_96_GIB_APPLE_HOST",
        "official_runtime_parity_claimed": False,
        "validation_claim": "COHERENT_TEXT_CORE_NEXT_TOKEN_BEHAVIOR" if finite and coherent == len(results) and replay_match
                            else "INSTRUMENT_FAILURE_OR_INSUFFICIENT_COHERENCE",
        "runtime": "MLX Metal; native packed routed experts; one source shard window",
        "source_layer_read_passes": 2,
        "speed_law": "eight validation probes share one block-diagonal layer-outer pass; "
                     "one independent mathematics pass proves deterministic replay",
        "probes": [{"id": result.get("probe_id"), "token_count": result.get("token_count"),
                    "top_token_text": result.get("top_token_text", [])[:10],
                    "coherent": result.get("coherent_next_token"),
                    "runtime_seconds": result.get("runtime_seconds"),
                    "result_sha256": result.get("result_sha256")} for result in results],
    }
    suite["seal_sha256"] = hashlib.sha256(json.dumps(
        suite, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    atomic_json(output / "KIMI_K26_PARENT_FORWARD_VALIDATION.json", suite)
    return suite


def main() -> int:
    mx.set_cache_limit(MLX_CACHE_LIMIT_BYTES)
    mx.set_memory_limit(MLX_MEMORY_LIMIT_BYTES)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    probe = sub.add_parser("probe")
    probe.add_argument("--source", type=Path, required=True)
    probe.add_argument("--text", required=True)
    probe.add_argument("--chat", action="store_true")
    probe.add_argument("--thinking", action="store_true")
    suite = sub.add_parser("run-suite")
    suite.add_argument("--source", type=Path, required=True)
    suite.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "probe":
            print(json.dumps(forward_prompt(args.source.resolve(strict=True), args.text,
                                            chat=args.chat, thinking=args.thinking), sort_keys=True))
        else:
            print(json.dumps(run_suite(args.source.resolve(strict=True),
                                       args.output.resolve()), sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
