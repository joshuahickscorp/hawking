#!/usr/bin/env python3.12
"""Inspectable NumPy reference forward for GLM-5.2 main and physical MTP.

This is a scientific oracle, not the shipping runtime.  It follows the pinned
Transformers eager implementation for the 78-layer main graph and the pinned
vLLM MTP boundary for the physically stored next-token layer.  Arrays are
decoded to float32 so every intermediate can be inspected and compared.

Important boundaries:
* Transformers parity applies to the main graph only.
* MTP parity is a separate physical-checkpoint/runtime oracle.
* Top-k slot order is not semantically stable when ``sorted=False``; comparison
  helpers canonicalize (index, weight) pairs without changing execution.
"""
from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np


class Glm52ReferenceError(RuntimeError):
    """Fail-closed reference-forward error."""


class TensorSource(Protocol):
    def tensor(self, name: str) -> np.ndarray: ...


@dataclasses.dataclass
class ReferenceCache:
    """Per-layer MLA and DSA keys for prefill/decode parity."""

    keys: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    values: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    indexer_keys: dict[int, np.ndarray] = dataclasses.field(default_factory=dict)
    mtp_iteration_topk: np.ndarray | None = None

    def sequence_length(self) -> int:
        if not self.keys:
            return 0
        lengths = {value.shape[2] for value in self.keys.values()}
        if len(lengths) != 1:
            raise Glm52ReferenceError(f"inconsistent KV cache lengths: {sorted(lengths)}")
        return next(iter(lengths))


def _tensor(source: Mapping[str, np.ndarray] | TensorSource, name: str) -> np.ndarray:
    try:
        value = source[name] if isinstance(source, Mapping) else source.tensor(name)
    except (KeyError, OSError, ValueError) as exc:
        raise Glm52ReferenceError(f"cannot load required tensor {name!r}: {exc}") from exc
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise Glm52ReferenceError(f"tensor is not numeric: {name}: {array.dtype}")
    return array.astype(np.float32, copy=False)


def linear(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """PyTorch ``F.linear(x, weight)`` with strict shape validation."""
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        raise Glm52ReferenceError(
            f"linear shape mismatch: x={x.shape}, weight={weight.shape}"
        )
    return np.matmul(x, weight.T, dtype=np.float32)


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if weight.shape != (x.shape[-1],):
        raise Glm52ReferenceError(f"RMSNorm weight {weight.shape} != ({x.shape[-1]},)")
    variance = np.mean(x * x, axis=-1, keepdims=True, dtype=np.float32)
    return (x / np.sqrt(variance + np.float32(eps), dtype=np.float32)) * weight


def layernorm(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Affine LayerNorm used only by the DSA indexer key projection."""
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    if weight.shape != (x.shape[-1],) or bias.shape != weight.shape:
        raise Glm52ReferenceError("LayerNorm affine shape mismatch")
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    centered = x - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    return centered / np.sqrt(variance + np.float32(eps), dtype=np.float32) * weight + bias


def silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.float32(1.0) + np.exp(-x, dtype=np.float32))


def rope_cos_sin(
    position_ids: np.ndarray,
    rotary_dim: int,
    rope_theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Default GLM RoPE frequencies in forced float32."""
    positions = np.asarray(position_ids, dtype=np.float32)
    if positions.ndim != 2 or rotary_dim <= 0 or rotary_dim % 2:
        raise Glm52ReferenceError("position_ids must be [B,S] and rotary_dim must be positive/even")
    exponent = np.arange(0, rotary_dim, 2, dtype=np.float32) / np.float32(rotary_dim)
    inv_freq = np.float32(1.0) / np.power(np.float32(rope_theta), exponent, dtype=np.float32)
    frequencies = positions[..., None] * inv_freq[None, None, :]
    embedding = np.concatenate([frequencies, frequencies], axis=-1)
    return np.cos(embedding, dtype=np.float32), np.sin(embedding, dtype=np.float32)


def apply_interleaved_rope(
    q: np.ndarray,
    k: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pinned Transformers GLM interleaved layout for tensors ``[B,H,S,D]``.

    The result is concatenated rotated first/second components, *not* scattered
    back to even and odd input locations.  This is an easy but consequential
    GLM-specific trap.
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    cos = np.asarray(cos, dtype=np.float32)
    sin = np.asarray(sin, dtype=np.float32)
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != k.shape[0] or q.shape[2:] != k.shape[2:]:
        raise Glm52ReferenceError(f"invalid RoPE q/k shapes: {q.shape}, {k.shape}")
    if q.shape[-1] % 2 or cos.shape != (q.shape[0], q.shape[2], q.shape[-1]):
        raise Glm52ReferenceError(f"invalid RoPE trig shape: {cos.shape}")
    half_cos = cos[..., : cos.shape[-1] // 2][:, None, :, :]
    half_sin = sin[..., : sin.shape[-1] // 2][:, None, :, :]

    def rotate(value: np.ndarray) -> np.ndarray:
        first, second = value[..., 0::2], value[..., 1::2]
        return np.concatenate(
            [first * half_cos - second * half_sin, second * half_cos + first * half_sin],
            axis=-1,
        )

    return rotate(q), rotate(k)


def _topk_desc(values: np.ndarray, k: int) -> np.ndarray:
    if not 1 <= k <= values.shape[-1]:
        raise Glm52ReferenceError(f"invalid top-k {k} for width {values.shape[-1]}")
    # Stable ordering gives deterministic evidence.  Official kernels use
    # sorted=False; comparisons must canonicalize sets when ties are possible.
    return np.argsort(-values, axis=-1, kind="stable")[..., :k]


def canonical_topk_pairs(indices: np.ndarray, weights: np.ndarray) -> list[list[tuple[int, float]]]:
    idx = np.asarray(indices).reshape(-1, indices.shape[-1])
    val = np.asarray(weights).reshape(-1, weights.shape[-1])
    if idx.shape != val.shape:
        raise Glm52ReferenceError("top-k index/weight shape mismatch")
    return [sorted((int(i), float(w)) for i, w in zip(row_i, row_w)) for row_i, row_w in zip(idx, val)]


def indexer_topk(
    hidden_states: np.ndarray,
    q_resid: np.ndarray,
    *,
    wq_b: np.ndarray,
    wk: np.ndarray,
    k_norm_weight: np.ndarray,
    k_norm_bias: np.ndarray,
    weights_proj: np.ndarray,
    position_ids: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    n_heads: int,
    head_dim: int,
    rotary_dim: int,
    topk: int,
    attention_mask: np.ndarray | None = None,
    past_keys: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic DSA indices, scores and the updated indexer key cache."""
    hidden_states = np.asarray(hidden_states, dtype=np.float32)
    q_resid = np.asarray(q_resid, dtype=np.float32)
    batch, sequence, _ = hidden_states.shape
    q = linear(q_resid, wq_b)
    if q.shape[-1] != n_heads * head_dim:
        raise Glm52ReferenceError("indexer wq_b output does not match heads*dim")
    q = q.reshape(batch, sequence, n_heads, head_dim)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]

    k = layernorm(linear(hidden_states, wk), k_norm_weight, k_norm_bias, 1e-6)
    if k.shape[-1] != head_dim:
        raise Glm52ReferenceError("indexer wk output does not match head_dim")
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot_bhsd = q_rot.transpose(0, 2, 1, 3)
    k_rot_bhsd = k_rot[:, None, :, :]
    q_rot_bhsd, k_rot_bhsd = apply_interleaved_rope(
        q_rot_bhsd, k_rot_bhsd, cos, sin
    )
    q = np.concatenate([q_rot_bhsd.transpose(0, 2, 1, 3), q_pass], axis=-1)
    k = np.concatenate([k_rot_bhsd[:, 0], k_pass], axis=-1)
    all_k = k if past_keys is None else np.concatenate([past_keys, k], axis=1)

    # Match the pinned Transformers matmul layout exactly:
    # [B,S,H,D] x [B,1,D,T] -> [B,S,H,T].  Keeping this layout avoids
    # implementation-dependent einsum contraction order in the score oracle.
    scores = np.matmul(
        q.astype(np.float32, copy=False),
        all_k.transpose(0, 2, 1)[:, None].astype(np.float32, copy=False),
    )
    scores = np.maximum(scores * np.float32(head_dim ** -0.5), np.float32(0.0))
    head_weights = linear(hidden_states, weights_proj).astype(np.float32)
    head_weights *= np.float32(n_heads ** -0.5)
    index_scores = np.matmul(head_weights[:, :, None, :], scores).squeeze(-2)

    if attention_mask is not None:
        mask = np.asarray(attention_mask, dtype=np.float32)
        if mask.shape != index_scores.shape:
            raise Glm52ReferenceError(
                f"indexer mask {mask.shape} != score shape {index_scores.shape}"
            )
        index_scores = index_scores + mask
    else:
        key_positions = np.arange(all_k.shape[1], dtype=np.int64)
        causal = key_positions[None, None, :] > np.asarray(position_ids)[:, :, None]
        index_scores = np.where(causal, -np.inf, index_scores)
    count = min(int(topk), all_k.shape[1])
    indices = _topk_desc(index_scores, count).astype(np.int32)
    return indices, index_scores, all_k


def router_topk(
    hidden_states: np.ndarray,
    weight: np.ndarray,
    correction_bias: np.ndarray,
    *,
    top_k: int,
    num_groups: int,
    topk_groups: int,
    normalize: bool,
    scaling_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GLM noaux_tc router: corrected selection, uncorrected sigmoid weights."""
    hidden = np.asarray(hidden_states, dtype=np.float32)
    original = hidden.shape[:-1]
    flat = hidden.reshape(-1, hidden.shape[-1])
    logits = linear(flat, np.asarray(weight, dtype=np.float32)).astype(np.float32)
    scores = np.float32(1.0) / (np.float32(1.0) + np.exp(-logits, dtype=np.float32))
    bias = np.asarray(correction_bias, dtype=np.float32)
    experts = scores.shape[-1]
    if bias.shape != (experts,) or experts % num_groups:
        raise Glm52ReferenceError("router correction/group shape mismatch")
    corrected = scores + bias
    grouped = corrected.reshape(-1, num_groups, experts // num_groups)
    top_two = np.take_along_axis(grouped, _topk_desc(grouped, min(2, grouped.shape[-1])), axis=-1)
    group_scores = np.sum(top_two, axis=-1, dtype=np.float32)
    selected_groups = _topk_desc(group_scores, topk_groups)
    group_mask = np.zeros_like(group_scores, dtype=bool)
    np.put_along_axis(group_mask, selected_groups, True, axis=-1)
    expert_mask = np.repeat(group_mask, experts // num_groups, axis=-1)
    choice = np.where(expert_mask, corrected, -np.inf)
    indices = _topk_desc(choice, top_k).astype(np.int64)
    weights = np.take_along_axis(scores, indices, axis=-1).astype(np.float32)
    if normalize:
        weights /= np.sum(weights, axis=-1, keepdims=True, dtype=np.float32) + np.float32(1e-20)
    weights *= np.float32(scaling_factor)
    return (
        logits.reshape(*original, experts),
        weights.reshape(*original, top_k),
        indices.reshape(*original, top_k),
    )


def dense_mlp(
    hidden_states: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> np.ndarray:
    return linear(silu(linear(hidden_states, gate)) * linear(hidden_states, up), down)


def routed_moe(
    hidden_states: np.ndarray,
    source: Mapping[str, np.ndarray] | TensorSource,
    prefix: str,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Execute only hit experts from separate official checkpoint tensors."""
    router_logits, topk_weights, topk_indices = router_topk(
        hidden_states,
        _tensor(source, f"{prefix}.gate.weight"),
        _tensor(source, f"{prefix}.gate.e_score_correction_bias"),
        top_k=int(config["num_experts_per_tok"]),
        num_groups=int(config["n_group"]),
        topk_groups=int(config["topk_group"]),
        normalize=bool(config["norm_topk_prob"]),
        scaling_factor=float(config["routed_scaling_factor"]),
    )
    flat = np.asarray(hidden_states, dtype=np.float32).reshape(-1, hidden_states.shape[-1])
    flat_indices = topk_indices.reshape(-1, topk_indices.shape[-1])
    flat_weights = topk_weights.reshape(-1, topk_weights.shape[-1])
    routed = np.zeros_like(flat, dtype=np.float32)
    for expert in sorted(set(int(value) for value in flat_indices.ravel())):
        token_slot = np.argwhere(flat_indices == expert)
        if token_slot.size == 0:
            continue
        tokens, slots = token_slot[:, 0], token_slot[:, 1]
        current = flat[tokens]
        stem = f"{prefix}.experts.{expert}"
        expert_out = dense_mlp(
            current,
            _tensor(source, f"{stem}.gate_proj.weight"),
            _tensor(source, f"{stem}.up_proj.weight"),
            _tensor(source, f"{stem}.down_proj.weight"),
        )
        weighted = expert_out * flat_weights[tokens, slots, None]
        np.add.at(routed, tokens, weighted)
    shared_prefix = f"{prefix}.shared_experts"
    shared = dense_mlp(
        hidden_states,
        _tensor(source, f"{shared_prefix}.gate_proj.weight"),
        _tensor(source, f"{shared_prefix}.up_proj.weight"),
        _tensor(source, f"{shared_prefix}.down_proj.weight"),
    )
    output = routed.reshape(hidden_states.shape) + shared
    return output, {
        "router_logits": router_logits,
        "topk_weights": topk_weights,
        "topk_indices": topk_indices,
        "routed_output": routed.reshape(hidden_states.shape),
        "shared_output": shared,
    }


def _append_cache(
    mapping: dict[int, np.ndarray],
    layer: int,
    current: np.ndarray,
    sequence_axis: int,
) -> np.ndarray:
    previous = mapping.get(layer)
    updated = current if previous is None else np.concatenate([previous, current], axis=sequence_axis)
    mapping[layer] = updated
    return updated


def attention_forward(
    hidden_states: np.ndarray,
    source: Mapping[str, np.ndarray] | TensorSource,
    layer: int,
    config: Mapping[str, Any],
    position_ids: np.ndarray,
    cache: ReferenceCache,
    *,
    indexer_type: str,
    previous_topk: np.ndarray | None,
    additive_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Main MLA+DSA attention for one already-normalized decoder input."""
    prefix = f"model.layers.{layer}.self_attn"
    if indexer_type == "shared" and previous_topk is None:
        raise Glm52ReferenceError("shared IndexShare layer has no previous full-layer indices")
    batch, sequence, _ = hidden_states.shape
    heads = int(config["num_attention_heads"])
    q_rank = int(config["q_lora_rank"])
    kv_rank = int(config["kv_lora_rank"])
    nope = int(config["qk_nope_head_dim"])
    rope = int(config["qk_rope_head_dim"])
    value_dim = int(config["v_head_dim"])
    qk_dim = nope + rope

    q_a = linear(hidden_states, _tensor(source, f"{prefix}.q_a_proj.weight"))
    if q_a.shape[-1] != q_rank:
        raise Glm52ReferenceError("q_a rank mismatch")
    q_resid = rmsnorm(q_a, _tensor(source, f"{prefix}.q_a_layernorm.weight"), float(config["rms_norm_eps"]))
    q = linear(q_resid, _tensor(source, f"{prefix}.q_b_proj.weight"))
    q = q.reshape(batch, sequence, heads, qk_dim).transpose(0, 2, 1, 3)
    q_pass, q_rot = q[..., :nope], q[..., nope:]

    compressed = linear(hidden_states, _tensor(source, f"{prefix}.kv_a_proj_with_mqa.weight"))
    k_latent, k_rot = compressed[..., :kv_rank], compressed[..., kv_rank:]
    k_latent = rmsnorm(
        k_latent,
        _tensor(source, f"{prefix}.kv_a_layernorm.weight"),
        float(config["rms_norm_eps"]),
    )
    kv = linear(k_latent, _tensor(source, f"{prefix}.kv_b_proj.weight"))
    kv = kv.reshape(batch, sequence, heads, nope + value_dim).transpose(0, 2, 1, 3)
    k_pass, values = kv[..., :nope], kv[..., nope:]
    k_rot = k_rot.reshape(batch, 1, sequence, rope)

    cos, sin = rope_cos_sin(
        position_ids,
        rope,
        float(config["rope_parameters"]["rope_theta"]),
    )
    q_rot, k_rot = apply_interleaved_rope(q_rot, k_rot, cos, sin)
    k_rot = np.broadcast_to(k_rot, (*k_pass.shape[:-1], rope))
    queries = np.concatenate([q_pass, q_rot], axis=-1)
    keys_current = np.concatenate([k_pass, k_rot], axis=-1)
    keys = _append_cache(cache.keys, layer, keys_current, 2)
    values_all = _append_cache(cache.values, layer, values, 2)

    if indexer_type == "full":
        idx_prefix = f"{prefix}.indexer"
        index_mask = None
        if additive_mask is not None:
            mask = np.asarray(additive_mask, dtype=np.float32)
            index_mask = mask[:, 0] if mask.ndim == 4 else mask
        topk, index_scores, index_keys = indexer_topk(
            hidden_states,
            q_resid,
            wq_b=_tensor(source, f"{idx_prefix}.wq_b.weight"),
            wk=_tensor(source, f"{idx_prefix}.wk.weight"),
            k_norm_weight=_tensor(source, f"{idx_prefix}.k_norm.weight"),
            k_norm_bias=_tensor(source, f"{idx_prefix}.k_norm.bias"),
            weights_proj=_tensor(source, f"{idx_prefix}.weights_proj.weight"),
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            n_heads=int(config["index_n_heads"]),
            head_dim=int(config["index_head_dim"]),
            rotary_dim=rope,
            topk=int(config["index_topk"]),
            attention_mask=index_mask,
            past_keys=cache.indexer_keys.get(layer),
        )
        cache.indexer_keys[layer] = index_keys
    elif indexer_type == "shared":
        if previous_topk is None:
            raise Glm52ReferenceError("shared IndexShare layer has no previous full-layer indices")
        topk = previous_topk
        index_scores = np.empty((batch, sequence, keys.shape[2]), dtype=np.float32)
        index_scores.fill(np.nan)
    else:
        raise Glm52ReferenceError(f"unknown indexer_type {indexer_type!r}")

    key_count = keys.shape[2]
    allow = np.zeros((batch, sequence, key_count), dtype=bool)
    np.put_along_axis(allow, topk.astype(np.int64), True, axis=-1)
    key_positions = np.arange(key_count, dtype=np.int64)
    allow &= key_positions[None, None, :] <= np.asarray(position_ids)[:, :, None]

    scores = np.matmul(queries, np.swapaxes(keys, 2, 3), dtype=np.float32)
    scores *= np.float32(qk_dim ** -0.5)
    scores = np.where(allow[:, None, :, :], scores, -np.inf)
    if additive_mask is not None:
        scores = scores + np.asarray(additive_mask, dtype=np.float32)
    maximum = np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores - maximum, dtype=np.float32)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True, dtype=np.float32)
    context = np.matmul(probabilities, values_all, dtype=np.float32)
    context = context.transpose(0, 2, 1, 3).reshape(batch, sequence, heads * value_dim)
    output = linear(context, _tensor(source, f"{prefix}.o_proj.weight"))
    return output, topk, {
        "q_resid": q_resid,
        "queries": queries,
        "keys": keys,
        "values": values_all,
        "index_scores": index_scores,
        "topk_indices": topk,
        "attention_probabilities": probabilities,
        "attention_output": output,
    }


def decoder_layer(
    hidden_states: np.ndarray,
    source: Mapping[str, np.ndarray] | TensorSource,
    layer: int,
    config: Mapping[str, Any],
    position_ids: np.ndarray,
    cache: ReferenceCache,
    *,
    mlp_type: str,
    indexer_type: str,
    previous_topk: np.ndarray | None,
    additive_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    prefix = f"model.layers.{layer}"
    before = np.asarray(hidden_states, dtype=np.float32)
    attention_input = rmsnorm(
        before,
        _tensor(source, f"{prefix}.input_layernorm.weight"),
        float(config["rms_norm_eps"]),
    )
    attention_output, topk, attention_trace = attention_forward(
        attention_input,
        source,
        layer,
        config,
        position_ids,
        cache,
        indexer_type=indexer_type,
        previous_topk=previous_topk,
        additive_mask=additive_mask,
    )
    post_attention = before + attention_output
    mlp_input = rmsnorm(
        post_attention,
        _tensor(source, f"{prefix}.post_attention_layernorm.weight"),
        float(config["rms_norm_eps"]),
    )
    if mlp_type == "dense":
        mlp_prefix = f"{prefix}.mlp"
        mlp_output = dense_mlp(
            mlp_input,
            _tensor(source, f"{mlp_prefix}.gate_proj.weight"),
            _tensor(source, f"{mlp_prefix}.up_proj.weight"),
            _tensor(source, f"{mlp_prefix}.down_proj.weight"),
        )
        mlp_trace: dict[str, Any] = {"kind": "dense"}
    elif mlp_type == "sparse":
        mlp_output, mlp_trace = routed_moe(mlp_input, source, f"{prefix}.mlp", config)
        mlp_trace = {"kind": "sparse", **mlp_trace}
    else:
        raise Glm52ReferenceError(f"unknown MLP type {mlp_type!r}")
    output = post_attention + mlp_output
    return output, topk, {
        "layer": layer,
        "input": before,
        "attention_input": attention_input,
        "attention": attention_trace,
        "post_attention": post_attention,
        "mlp_input": mlp_input,
        "mlp": mlp_trace,
        "output": output,
    }


def _embedding_rows(
    source: Mapping[str, np.ndarray] | TensorSource,
    input_ids: np.ndarray,
) -> np.ndarray:
    if not isinstance(source, Mapping) and hasattr(source, "rows"):
        return np.asarray(source.rows("model.embed_tokens.weight", input_ids), dtype=np.float32)
    return _tensor(source, "model.embed_tokens.weight")[np.asarray(input_ids, dtype=np.int64)]


def main_forward(
    input_ids: np.ndarray,
    source: Mapping[str, np.ndarray] | TensorSource,
    config: Mapping[str, Any],
    *,
    cache: ReferenceCache | None = None,
    position_ids: np.ndarray | None = None,
    additive_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, ReferenceCache, dict[str, Any]]:
    """Execute every configured main layer and return real vocabulary logits."""
    ids = np.asarray(input_ids, dtype=np.int64)
    if ids.ndim != 2:
        raise Glm52ReferenceError("input_ids must be [B,S]")
    cache = cache or ReferenceCache()
    if position_ids is None:
        start = cache.sequence_length()
        position_ids = np.broadcast_to(
            np.arange(start, start + ids.shape[1], dtype=np.int64)[None, :], ids.shape
        ).copy()
    hidden = _embedding_rows(source, ids)
    layers: list[dict[str, Any]] = []
    previous_topk: np.ndarray | None = None
    main_layers = int(config["num_hidden_layers"])
    indexer_types = list(config["indexer_types"])
    mlp_types = list(config["mlp_layer_types"])
    if len(indexer_types) != main_layers or len(mlp_types) != main_layers:
        raise Glm52ReferenceError("config main-layer schedules have wrong length")
    for layer in range(main_layers):
        hidden, previous_topk, trace = decoder_layer(
            hidden,
            source,
            layer,
            config,
            position_ids,
            cache,
            mlp_type=mlp_types[layer],
            indexer_type=indexer_types[layer],
            previous_topk=previous_topk,
            additive_mask=additive_mask,
        )
        layers.append(trace)
    pre_final = hidden
    final = rmsnorm(
        pre_final,
        _tensor(source, "model.norm.weight"),
        float(config["rms_norm_eps"]),
    )
    logits = linear(final, _tensor(source, "lm_head.weight"))
    return logits, cache, {
        "position_ids": np.asarray(position_ids),
        "layers": layers,
        "pre_final_hidden": pre_final,
        "final_hidden": final,
        "logits": logits,
        "final_main_topk": previous_topk,
    }


def mtp_forward(
    previous_main_hidden: np.ndarray,
    shifted_input_embeddings: np.ndarray,
    source: Mapping[str, np.ndarray] | TensorSource,
    config: Mapping[str, Any],
    position_ids: np.ndarray,
    *,
    cache: ReferenceCache | None = None,
    speculative_step: int = 0,
    additive_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, ReferenceCache, dict[str, Any]]:
    """Execute the physically stored MTP layer using pinned runtime semantics.

    Step zero computes its own full DSA index.  Later speculative iterations may
    reuse that MTP index when ``index_share_for_mtp_iteration`` is enabled.  It
    never inherits the backbone layer-74/last-full index directly.
    """
    cache = cache or ReferenceCache()
    previous = np.asarray(previous_main_hidden, dtype=np.float32)
    shifted = np.asarray(shifted_input_embeddings, dtype=np.float32).copy()
    positions = np.asarray(position_ids, dtype=np.int64)
    if previous.shape != shifted.shape or positions.shape != previous.shape[:2]:
        raise Glm52ReferenceError("MTP hidden/embed/position shape mismatch")
    shifted = np.where(positions[..., None] == 0, np.float32(0.0), shifted)
    layer = int(config["num_hidden_layers"])
    prefix = f"model.layers.{layer}"
    enorm = rmsnorm(shifted, _tensor(source, f"{prefix}.enorm.weight"), float(config["rms_norm_eps"]))
    hnorm = rmsnorm(previous, _tensor(source, f"{prefix}.hnorm.weight"), float(config["rms_norm_eps"]))
    fused = linear(np.concatenate([enorm, hnorm], axis=-1), _tensor(source, f"{prefix}.eh_proj.weight"))

    reuse = bool(config.get("index_share_for_mtp_iteration", False)) and speculative_step > 0
    if reuse and cache.mtp_iteration_topk is None:
        raise Glm52ReferenceError("MTP iteration requested reuse before step-zero index exists")
    output, topk, layer_trace = decoder_layer(
        fused,
        source,
        layer,
        config,
        positions,
        cache,
        mlp_type="sparse",
        indexer_type="shared" if reuse else "full",
        previous_topk=cache.mtp_iteration_topk if reuse else None,
        additive_mask=additive_mask,
    )
    if not reuse:
        cache.mtp_iteration_topk = topk.copy()
    final = rmsnorm(
        output,
        _tensor(source, f"{prefix}.shared_head.norm.weight"),
        float(config["rms_norm_eps"]),
    )
    logits = linear(final, _tensor(source, "lm_head.weight"))
    return logits, cache, {
        "speculative_step": speculative_step,
        "reused_step_zero_topk": reuse,
        "masked_shifted_embeddings": shifted,
        "enorm": enorm,
        "hnorm": hnorm,
        "fused_input": fused,
        "layer": layer_trace,
        "pre_shared_head": output,
        "shared_head_hidden": final,
        "logits": logits,
        "topk_indices": topk,
    }


def selfcheck() -> dict[str, Any]:
    positions = np.array([[0, 1]], dtype=np.int64)
    cos, sin = rope_cos_sin(positions, 4, 10_000.0)
    q = np.arange(8, dtype=np.float32).reshape(1, 1, 2, 4)
    rotated, _ = apply_interleaved_rope(q, q, cos, sin)
    # Position zero proves the concatenated component layout: [x0,x2,x1,x3].
    if not np.array_equal(rotated[0, 0, 0], np.array([0, 2, 1, 3], dtype=np.float32)):
        raise AssertionError("interleaved RoPE layout selfcheck failed")
    logits, weights, indices = router_topk(
        np.ones((1, 1, 2), dtype=np.float32),
        np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32),
        np.array([0, 0.1, 0, 0], dtype=np.float32),
        top_k=2,
        num_groups=1,
        topk_groups=1,
        normalize=True,
        scaling_factor=2.5,
    )
    if not np.allclose(weights.sum(axis=-1), 2.5):
        raise AssertionError("router scaling selfcheck failed")
    return {
        "status": "PASS",
        "rope_position_zero_layout": rotated[0, 0, 0].tolist(),
        "router_indices": indices.tolist(),
        "router_logits_shape": list(logits.shape),
        "router_weight_sum": float(weights.sum()),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(selfcheck(), indent=2, sort_keys=True))
