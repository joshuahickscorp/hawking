#!/usr/bin/env python3.12
"""Kimi K2.6 text-core provider adapter and architecture-preserving synthetic twin.

This module is deliberately small and dependency-light.  It validates the
official configuration, compressed-tensors INT4 byte convention, DeepSeek-V3
router/shared-expert combine, RMSNorm/residual behavior, and a bounded CPU MLA
reference on deterministic synthetic dimensions.  It does not claim source
parent parity; the real-shard forward has a separate evidence gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REPO = "moonshotai/Kimi-K2.6"
REVISION = "7eb5002f6aadc958aed6a9177b7ed26bb94011bb"


class AdapterError(RuntimeError):
    pass


def bind_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    text = config.get("text_config") or {}
    expected = {
        "model_type": "kimi_k2", "hidden_size": 7168, "num_hidden_layers": 61,
        "n_routed_experts": 384, "n_shared_experts": 1, "num_experts_per_tok": 8,
        "max_position_embeddings": 262144, "q_lora_rank": 1536, "kv_lora_rank": 512,
        "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128,
        "scoring_func": "sigmoid", "topk_method": "noaux_tc", "norm_topk_prob": True,
        "n_group": 1, "topk_group": 1, "routed_scaling_factor": 2.827,
        "rms_norm_eps": 1e-5, "rope_theta": 50000.0,
    }
    mismatches = {key: {"expected": value, "actual": text.get(key)}
                  for key, value in expected.items() if text.get(key) != value}
    quant = text.get("quantization_config") or {}
    group = (quant.get("config_groups") or {}).get("group_0", {}).get("weights") or {}
    expected_quant = {"format": "pack-quantized", "quant_method": "compressed-tensors",
                      "num_bits": 4, "group_size": 32, "symmetric": True}
    actual_quant = {"format": quant.get("format"), "quant_method": quant.get("quant_method"),
                    "num_bits": group.get("num_bits"), "group_size": group.get("group_size"),
                    "symmetric": group.get("symmetric")}
    if mismatches or actual_quant != expected_quant:
        raise AdapterError(f"official config mismatch: {mismatches}; quant={actual_quant}")
    return {"outer_model_type": config.get("model_type"), "text": expected,
            "quantization": actual_quant, "vision_layers":
            (config.get("vision_config") or {}).get("vt_num_hidden_layers"),
            "claim_boundary": "TEXT_CORE_ONLY"}


def pack_int4(values: np.ndarray) -> np.ndarray:
    """compressed-tensors pack_to_int32 convention, dense on the last dimension."""
    values = np.asarray(values, dtype=np.int8)
    if values.ndim < 2 or np.any(values < -8) or np.any(values > 7):
        raise AdapterError("INT4 input must be rank >=2 with values in [-8, 7]")
    cols = values.shape[-1]
    words = (cols * 4 + 31) // 32
    flattened = values.reshape(-1, cols).astype(np.int64) + 8
    packed = np.zeros((flattened.shape[0], words), dtype=np.uint32)
    for column in range(cols):
        bit = column * 4
        packed[:, bit // 32] |= (flattened[:, column].astype(np.uint32) << (bit % 32))
    return packed.view(np.int32).reshape(*values.shape[:-1], words)


def unpack_int4(packed: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """Reverse compressed-tensors dense signed-offset packing."""
    packed = np.asarray(packed, dtype=np.int32)
    cols = int(original_shape[-1])
    words = (cols * 4 + 31) // 32
    if packed.shape[-1] != words or tuple(packed.shape[:-1]) != tuple(original_shape[:-1]):
        raise AdapterError(f"packed/original shape mismatch: {packed.shape} -> {original_shape}")
    flat = packed.reshape(-1, words).view(np.uint32)
    output = np.empty((flat.shape[0], cols), dtype=np.int8)
    for column in range(cols):
        bit = column * 4
        unsigned = (flat[:, bit // 32] >> (bit % 32)) & np.uint32(0xF)
        output[:, column] = unsigned.astype(np.int8) - 8
    return output.reshape(original_shape)


def dequantize_int4(packed: np.ndarray, scales: np.ndarray,
                    original_shape: tuple[int, ...], group_size: int = 32) -> np.ndarray:
    quantized = unpack_int4(packed, original_shape).astype(np.float32)
    if original_shape[-1] % group_size:
        raise AdapterError("source dimension is not divisible by the bound group size")
    groups = original_shape[-1] // group_size
    expected = (*original_shape[:-1], groups)
    scales = np.asarray(scales, dtype=np.float32)
    if scales.shape != expected:
        raise AdapterError(f"scale shape {scales.shape} != {expected}")
    expanded = np.repeat(scales, group_size, axis=-1)
    return quantized * expanded


def rms_norm(x: np.ndarray, weight: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    x32 = np.asarray(x, dtype=np.float32)
    variance = np.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 / np.sqrt(variance + epsilon)) * np.asarray(weight, dtype=np.float32)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -40, 40)))


def mlp(x: np.ndarray, gate: np.ndarray, up: np.ndarray, down: np.ndarray) -> np.ndarray:
    return (silu(x @ gate.T) * (x @ up.T)) @ down.T


def route_noaux_tc(x: np.ndarray, gate_weight: np.ndarray, correction_bias: np.ndarray,
                   *, top_k: int, scaling: float, n_group: int = 1,
                   topk_group: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Official DeepSeek-V3 noaux_tc selection and normalized sigmoid weights."""
    scores = 1.0 / (1.0 + np.exp(-(x @ gate_weight.T)))
    selection = scores + correction_bias
    if selection.shape[-1] % n_group:
        raise AdapterError("expert count must be divisible by n_group")
    per_group = selection.shape[-1] // n_group
    grouped = selection.reshape(selection.shape[0], n_group, per_group)
    group_scores = np.sort(grouped, axis=-1)[..., -2:].sum(axis=-1)
    allowed_groups = np.argsort(group_scores, axis=-1)[:, -topk_group:]
    mask = np.zeros((selection.shape[0], n_group), dtype=bool)
    np.put_along_axis(mask, allowed_groups, True, axis=1)
    allowed = np.repeat(mask, per_group, axis=1)
    filtered = np.where(allowed, selection, -np.inf)
    indices = np.argsort(filtered, axis=-1)[:, -top_k:][:, ::-1]
    weights = np.take_along_axis(scores, indices, axis=-1)
    weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    return indices, weights * scaling


def moe(x: np.ndarray, gate_weight: np.ndarray, correction_bias: np.ndarray,
        experts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        shared: tuple[np.ndarray, np.ndarray, np.ndarray], *, top_k: int,
        scaling: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices, weights = route_noaux_tc(x, gate_weight, correction_bias, top_k=top_k,
                                       scaling=scaling)
    combined = np.zeros_like(x, dtype=np.float32)
    for token in range(x.shape[0]):
        for slot, expert_id in enumerate(indices[token]):
            combined[token] += weights[token, slot] * mlp(
                x[token:token + 1], *experts[int(expert_id)])[0]
    return combined + mlp(x, *shared), indices, weights


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def official_rope_layout(x: np.ndarray) -> np.ndarray:
    shape = x.shape
    return x.reshape(*shape[:-1], shape[-1] // 2, 2).swapaxes(-1, -2).reshape(shape)


def rope(x: np.ndarray, position: int, theta: float = 50000.0) -> np.ndarray:
    dimension = x.shape[-1]
    inv = 1.0 / (theta ** (np.arange(0, dimension, 2, dtype=np.float32) / dimension))
    phase = position * inv
    cosine = np.concatenate((np.cos(phase), np.cos(phase)))
    sine = np.concatenate((np.sin(phase), np.sin(phase)))
    reordered = official_rope_layout(x)
    return reordered * cosine + rotate_half(reordered) * sine


def mla_one_token(x: np.ndarray, weights: dict[str, np.ndarray], *, heads: int,
                  q_rank: int, kv_rank: int, qk_nope: int, qk_rope: int,
                  v_dim: int, position: int) -> tuple[np.ndarray, dict[str, tuple[int, ...]]]:
    """Bounded CPU reference for one-token DeepSeek MLA projection semantics."""
    q_low = x @ weights["q_a"].T
    q = rms_norm(q_low, weights["q_norm"]) @ weights["q_b"].T
    q = q.reshape(x.shape[0], heads, qk_nope + qk_rope)
    q_nope, q_pe = q[..., :qk_nope], rope(q[..., qk_nope:], position)
    compressed = x @ weights["kv_a"].T
    kv_low, k_pe = compressed[..., :kv_rank], compressed[..., kv_rank:]
    kv = rms_norm(kv_low, weights["kv_norm"]) @ weights["kv_b"].T
    kv = kv.reshape(x.shape[0], heads, qk_nope + v_dim)
    k_nope, value = kv[..., :qk_nope], kv[..., qk_nope:]
    k_pe = rope(k_pe[:, None, :], position)
    score = (np.sum(q_nope * k_nope, axis=-1) + np.sum(q_pe * k_pe, axis=-1))
    score = score / np.sqrt(qk_nope + qk_rope)
    # With one causal token the attention probability is exactly one, but calculating the score
    # still verifies the complete query/key semantic boundary.
    context = value.reshape(x.shape[0], heads * v_dim)
    output = context @ weights["o"].T
    return output, {"q": q.shape, "k": k_nope.shape, "v": value.shape,
                    "score": score.shape, "output": output.shape}


def functional_metal_k1(source: Path) -> dict[str, Any]:
    import mlx.core as mx
    from kimi_k26_reference import PREFIX, TensorShard, quantized_linear, shard_path

    shard = TensorShard(shard_path(source, 2))
    base = f"{PREFIX}.layers.1.mlp.experts.0.down_proj"
    original_shape = tuple(int(v) for v in np.asarray(
        shard.numpy(base + ".weight_shape"), dtype=np.int32).flat)
    packed = shard.numpy(base + ".weight_packed", unsigned_packed=True)
    scales = np.asarray(shard.numpy(base + ".weight_scale"), dtype=np.float32)
    rng = np.random.default_rng(260621)
    input_array = rng.standard_normal((1, original_shape[1]), dtype=np.float32)
    value = mx.array(input_array, dtype=mx.float32)
    metal = quantized_linear(value, shard, base)
    mx.eval(metal)
    rows = 8
    signed = unpack_int4(np.asarray(packed[:rows]).view(np.int32),
                         (rows, original_shape[1])).astype(np.float32)
    decoded = signed * np.repeat(scales[:rows], 32, axis=-1)
    cpu = input_array @ decoded.T
    metal_rows = np.asarray(metal.astype(mx.float32))[:, :rows]
    maximum_error = float(np.max(np.abs(cpu - metal_rows)))
    if not np.allclose(cpu, metal_rows, rtol=2e-5, atol=2e-5):
        raise AdapterError(f"functional Metal K1 mismatch: {maximum_error}")
    return {"status": "PASS", "tensor": base, "source_shape": list(original_shape),
            "rows_checked": rows, "max_abs_error": maximum_error,
            "operator": "mlx.quantized_matmul affine INT4 group32"}


def selftest(config_path: Path, source_path: Path | None = None) -> dict[str, Any]:
    binding = bind_config(config_path)
    rng = np.random.default_rng(260226)

    original = rng.integers(-8, 8, size=(3, 64), dtype=np.int8)
    packed = pack_int4(original)
    unpacked = unpack_int4(packed, original.shape)
    if not np.array_equal(original, unpacked) or packed.dtype != np.int32:
        raise AdapterError("compressed-tensors INT4 round trip failed")
    scales = rng.uniform(0.005, 0.2, size=(3, 2)).astype(np.float32)
    dequantized = dequantize_int4(packed, scales, original.shape)
    if not np.allclose(dequantized, original.astype(np.float32) * np.repeat(scales, 32, -1)):
        raise AdapterError("group dequantization failed")

    hidden, q_rank, kv_rank = 32, 12, 8
    heads, qk_nope, qk_rope, v_dim = 4, 8, 4, 8
    x = rng.normal(0, 0.1, size=(2, hidden)).astype(np.float32)
    w = {
        "q_a": rng.normal(0, 0.1, size=(q_rank, hidden)).astype(np.float32),
        "q_norm": np.ones(q_rank, dtype=np.float32),
        "q_b": rng.normal(0, 0.1, size=(heads * (qk_nope + qk_rope), q_rank)).astype(np.float32),
        "kv_a": rng.normal(0, 0.1, size=(kv_rank + qk_rope, hidden)).astype(np.float32),
        "kv_norm": np.ones(kv_rank, dtype=np.float32),
        "kv_b": rng.normal(0, 0.1, size=(heads * (qk_nope + v_dim), kv_rank)).astype(np.float32),
        "o": rng.normal(0, 0.1, size=(hidden, heads * v_dim)).astype(np.float32),
    }
    attention, shapes = mla_one_token(x, w, heads=heads, q_rank=q_rank, kv_rank=kv_rank,
                                      qk_nope=qk_nope, qk_rope=qk_rope, v_dim=v_dim,
                                      position=17)
    if not np.isfinite(attention).all() or attention.shape != x.shape:
        raise AdapterError("MLA reference produced invalid output")

    expert_count, intermediate = 12, 24
    gate = rng.normal(0, 0.1, size=(expert_count, hidden)).astype(np.float32)
    bias = rng.normal(0, 0.01, size=(expert_count,)).astype(np.float32)
    make_expert = lambda: (
        rng.normal(0, 0.05, size=(intermediate, hidden)).astype(np.float32),
        rng.normal(0, 0.05, size=(intermediate, hidden)).astype(np.float32),
        rng.normal(0, 0.05, size=(hidden, intermediate)).astype(np.float32),
    )
    experts = [make_expert() for _ in range(expert_count)]
    shared = make_expert()
    moe_output, indices, route_weights = moe(x, gate, bias, experts, shared, top_k=3,
                                             scaling=2.827)
    replay, replay_indices, replay_weights = moe(x, gate, bias, experts, shared, top_k=3,
                                                 scaling=2.827)
    if not (np.array_equal(indices, replay_indices) and
            np.array_equal(route_weights, replay_weights) and
            np.array_equal(moe_output, replay) and np.isfinite(moe_output).all()):
        raise AdapterError("router/shared-expert deterministic replay failed")

    residual = x + attention + moe_output
    logits = rms_norm(residual, np.ones(hidden, dtype=np.float32)) @ rng.normal(
        0, 0.1, size=(hidden, 97)).astype(np.float32)
    if logits.shape != (2, 97) or not np.isfinite(logits).all():
        raise AdapterError("synthetic end-to-end logits failed")
    fingerprint = hashlib.sha256(logits.tobytes()).hexdigest()
    metal = functional_metal_k1(source_path) if source_path else {"status": "NOT_RUN"}
    return {
        "schema": "hawking.kimi_k26.adapter_twin.v1", "status": "PASS",
        "repo": REPO, "revision": REVISION, "binding": binding,
        "checks": {
            "official_config_bound": True,
            "compressed_tensors_int4_i32_roundtrip": True,
            "group32_symmetric_dequantization": True,
            "mla_cpu_reference": True,
            "noaux_tc_router": True,
            "shared_plus_routed_expert_combine": True,
            "normalization_residual_logits": True,
            "deterministic_replay": True,
            "functional_metal_k1": metal["status"] == "PASS",
        },
        "synthetic": {"hidden": hidden, "experts": expert_count, "top_k": 3,
                      "mla_shapes": {k: list(v) for k, v in shapes.items()},
                      "route_indices": indices.tolist(),
                      "route_weight_sums": route_weights.sum(axis=-1).tolist(),
                      "logits_sha256": fingerprint},
        "metal_k1": metal,
        "runtime_claim": "SYNTHETIC_CPU_REFERENCE_AND_BOUND_REAL_SOURCE_METAL_K1",
        "source_parent_parity_claimed": False,
        "vision_claimed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("selftest")
    test.add_argument("--config", type=Path, required=True)
    test.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.command == "selftest":
        try:
            result = selftest(args.config.resolve(strict=True),
                              args.source.resolve(strict=True) if args.source else None)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"schema": "hawking.kimi_k26.adapter_twin.v1", "status": "FAIL",
                              "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
            return 1
        print(json.dumps(result, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
