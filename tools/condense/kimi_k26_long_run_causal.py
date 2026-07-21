#!/usr/bin/env python3.12
"""Held-out Kimi 0.90859-BPW control, routing atlas, and causal interventions."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import ml_dtypes
import mlx.core as mx
import numpy as np


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_f1_bracket as f1  # noqa: E402
import kimi_k26_f1_doctor_auction as auction  # noqa: E402
import kimi_k26_long_run_manager as manager  # noqa: E402
import kimi_k26_reference as reference  # noqa: E402


DOMAINS = {
    "factual": "prose", "science": "prose", "coding": "code",
    "mathematics": "reasoning", "reasoning": "reasoning",
    "instruction": "tool_format", "tool_thinking_protocol": "tool_format",
    "rare_token": "prose",
}
MATERIAL_MOE_RELATIVE_L2 = 0.10
IRRECOVERABLE_ROW_COSINE = 0.99
KNOWN_RETRY_FAULTS = [
    "ATTEMPT_0_BF16_RESIDUAL_ADD_ORDER_MISMATCH",
    "ATTEMPT_1_MANUAL_MOE_ACCUMULATION_NOT_BIT_EXACT",
]


def bf16(value: np.ndarray) -> np.ndarray:
    return np.asarray(np.asarray(value, dtype=ml_dtypes.bfloat16), dtype=np.float32)


def row_metrics(reference_value: np.ndarray, candidate: np.ndarray) -> dict[str, np.ndarray]:
    ref = np.asarray(reference_value, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)
    dot = np.sum(ref * cand, axis=-1)
    ref_norm = np.linalg.norm(ref, axis=-1)
    cand_norm = np.linalg.norm(cand, axis=-1)
    cosine = dot / (ref_norm * cand_norm + 1e-30)
    relative_l2 = np.linalg.norm(ref - cand, axis=-1) / (ref_norm + 1e-30)
    return {"cosine": cosine, "relative_l2": relative_l2,
            "norm_ratio": cand_norm / (ref_norm + 1e-30)}


def bootstrap_interval(values: np.ndarray, seed: int, samples: int = 2000) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(samples, values.size))
    means = values[draws].mean(axis=1)
    return {"mean": float(values.mean()), "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)), "n": int(values.size)}


def route_diagnostics(
    x: np.ndarray, shard: reference.TensorShard, layer: int, config: dict[str, Any],
) -> dict[str, np.ndarray]:
    base = f"{reference.PREFIX}.layers.{layer}.mlp.gate"
    gate = np.asarray(shard.mlx(base + ".weight").astype(mx.float32), dtype=np.float32)
    correction = np.asarray(
        shard.mlx(base + ".e_score_correction_bias").astype(mx.float32), dtype=np.float32,
    )
    logits = np.asarray(x, dtype=np.float32) @ gate.T
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    choice = scores + correction[None, :]
    experts = int(config["n_routed_experts"])
    groups = int(config["n_group"])
    top_groups = int(config["topk_group"])
    top_k = int(config["num_experts_per_tok"])
    grouped = choice.reshape(choice.shape[0], groups, experts // groups)
    group_score = np.sort(grouped, axis=-1)[..., -2:].sum(axis=-1)
    group_order = np.argsort(-group_score, axis=-1)
    group_mask = np.zeros_like(group_score, dtype=bool)
    np.put_along_axis(group_mask, group_order[:, :top_groups], True, axis=-1)
    allowed = np.repeat(group_mask, experts // groups, axis=-1)
    filtered = np.where(allowed, choice, 0.0)
    order = np.argsort(-filtered, axis=-1, kind="stable")
    indices = order[:, :top_k].astype(np.int32)
    ninth = order[:, top_k]
    selected_choice = np.take_along_axis(filtered, indices, axis=-1)
    margin = selected_choice[:, -1] - np.take_along_axis(
        filtered, ninth[:, None], axis=-1,
    )[:, 0]
    weights = np.take_along_axis(scores, indices, axis=-1)
    if config["norm_topk_prob"]:
        weights /= weights.sum(axis=-1, keepdims=True) + 1e-20
    weights *= float(config["routed_scaling_factor"])
    if top_groups < groups:
        group_margin = (
            group_score[np.arange(group_score.shape[0]), group_order[:, top_groups - 1]] -
            group_score[np.arange(group_score.shape[0]), group_order[:, top_groups]]
        )
    else:
        group_margin = np.full(group_score.shape[0], np.inf, dtype=np.float32)
    return {"logits": logits, "scores": scores, "choice": choice,
            "indices": indices, "weights": weights.astype(np.float32),
            "margin_8v9": margin.astype(np.float32),
            "group_margin": group_margin.astype(np.float32),
            "group_order": group_order.astype(np.int32)}


def weights_for_indices(scores: np.ndarray, indices: np.ndarray,
                        config: dict[str, Any]) -> np.ndarray:
    weights = np.take_along_axis(scores, indices, axis=-1).astype(np.float32)
    if config["norm_topk_prob"]:
        weights /= weights.sum(axis=-1, keepdims=True) + 1e-20
    return weights * float(config["routed_scaling_factor"])


def native_expert(
    x: mx.array, shard: reference.TensorShard, layer: int, expert: int,
    tokens: np.ndarray,
) -> mx.array:
    selected = mx.take(x, mx.array(tokens.astype(np.int32)), axis=0)
    base = f"{reference.PREFIX}.layers.{layer}.mlp.experts.{expert}"
    gate = reference.quantized_linear(selected, shard, base + ".gate_proj")
    up = reference.quantized_linear(selected, shard, base + ".up_proj")
    hidden = reference.silu(gate) * up
    output = reference.quantized_linear(hidden, shard, base + ".down_proj")
    mx.eval(output)
    return output


def build_output_cache(
    x: np.ndarray, route_sets: list[np.ndarray], shard: reference.TensorShard, layer: int,
) -> dict[int, np.ndarray]:
    x_mx = mx.array(x).astype(mx.bfloat16)
    sequence = x.shape[0]
    cache = {}
    experts = sorted({int(value) for routes in route_sets for value in routes.flat})
    for expert in experts:
        token_mask = np.zeros(sequence, dtype=bool)
        for routes in route_sets:
            token_mask |= np.any(routes == expert, axis=1)
        tokens = np.where(token_mask)[0].astype(np.int32)
        output = native_expert(x_mx, shard, layer, expert, tokens)
        full = np.zeros((sequence, x.shape[1]), dtype=np.float32)
        full[tokens] = np.asarray(output.astype(mx.float32), dtype=np.float32)
        cache[expert] = full
    del x_mx
    return cache


def combine_cached(
    cache: dict[int, np.ndarray], routes: np.ndarray, weights: np.ndarray,
) -> np.ndarray:
    sequence = routes.shape[0]
    hidden = next(iter(cache.values())).shape[1]
    combined = mx.zeros((sequence, hidden), dtype=mx.bfloat16)
    for expert in sorted(set(int(value) for value in routes.flat)):
        tokens, slots = np.where(routes == expert)
        output = mx.array(cache[expert][tokens]).astype(mx.bfloat16)
        route_weight = mx.array(weights[tokens, slots], dtype=mx.bfloat16)[:, None]
        scatter = mx.array(np.eye(sequence, dtype=np.float32)[tokens].T, dtype=mx.bfloat16)
        combined = combined + scatter @ (output * route_weight)
        mx.eval(combined)
    return np.asarray(combined.astype(mx.float32), dtype=np.float32)


def final_hidden(post: np.ndarray, routed: np.ndarray, shared: np.ndarray) -> np.ndarray:
    feed = (mx.array(routed).astype(mx.bfloat16) +
            mx.array(shared).astype(mx.bfloat16))
    value = (mx.array(post).astype(mx.bfloat16) + feed).astype(mx.bfloat16)
    mx.eval(value)
    return np.asarray(value.astype(mx.float32), dtype=np.float32)


def hidden_from_feed(post: np.ndarray, feed: np.ndarray) -> np.ndarray:
    value = (mx.array(post).astype(mx.bfloat16) +
             mx.array(feed).astype(mx.bfloat16)).astype(mx.bfloat16)
    mx.eval(value)
    return np.asarray(value.astype(mx.float32), dtype=np.float32)


def hidden_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()


def prepare_requests(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = reference.KimiTokenizer(source)
    requests = []
    token_ids = []
    segment_lengths = []
    token_records = []
    offset = 0
    for probe in reference.PROBES:
        rendered = (tokenizer.user_prompt(probe["text"], thinking=bool(probe.get("thinking")))
                    if probe.get("chat") else probe["text"])
        ids = tokenizer.encode(rendered)
        domain = DOMAINS[probe["id"]]
        requests.append({"id": probe["id"], "domain": domain, "text": probe["text"],
                         "rendered": rendered, "token_ids": ids})
        token_ids.extend(ids)
        segment_lengths.append(len(ids))
        for position, token_id in enumerate(ids):
            token_records.append({"token_index": offset + position, "segment": probe["id"],
                                  "domain": domain, "position": position,
                                  "token_id": int(token_id),
                                  "token_text": tokenizer.decode([int(token_id)])})
        offset += len(ids)
    return requests, {"token_ids": token_ids, "segment_lengths": segment_lengths,
                      "token_records": token_records}


def capture_layer_one_inputs(
    source: Path, token_ids: list[int], segment_lengths: list[int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    config = f1.read_json(source / "config.json")["text_config"]
    tail = reference.TensorShard(reference.shard_path(source, 62))
    embedding = tail.numpy(f"{reference.PREFIX}.embed_tokens.weight")
    hidden = mx.array(np.asarray(embedding[token_ids])).astype(mx.bfloat16)
    mx.eval(hidden)
    del embedding, tail
    dense = reference.TensorShard(reference.shard_path(source, 1))
    hidden, layer_zero_info = reference.layer_forward(hidden, dense, 0, config, segment_lengths)
    del dense
    shard = reference.TensorShard(reference.shard_path(source, 2))
    base = f"{reference.PREFIX}.layers.1"
    normalized = reference.rms_norm(hidden, shard.mlx(base + ".input_layernorm.weight"),
                                    float(config["rms_norm_eps"]))
    attention, attention_info = reference.attention(
        normalized, shard, 1, config, segment_lengths,
    )
    post = (hidden + attention).astype(mx.bfloat16)
    x = reference.rms_norm(post, shard.mlx(base + ".post_attention_layernorm.weight"),
                           float(config["rms_norm_eps"]))
    mx.eval(post, x)
    post_np = np.asarray(post.astype(mx.float32), dtype=np.float32)
    x_np = np.asarray(x.astype(mx.float32), dtype=np.float32)
    return post_np, x_np, {"layer_zero": layer_zero_info, "attention": attention_info}


def compact_expert_output(payload: Path, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights, _ = f1.decode_base_weights(payload)
    base, _ = f1.expert_forward(x, weights)
    doctored = auction.apply_hidden_doctor(payload, x, base)
    return base, doctored


def layer_one_paths(
    source: Path, post: np.ndarray, x: np.ndarray, routes: dict[str, np.ndarray],
    compact_output: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    config = f1.read_json(source / "config.json")["text_config"]
    shard = reference.TensorShard(reference.shard_path(source, 2))
    indices, weights = routes["indices"], routes["weights"]
    x_mx = mx.array(x).astype(mx.bfloat16)
    sequence = x.shape[0]
    sentinel = 0
    teacher_feed, _ = reference.routed_moe(x_mx, shard, 1, config)
    shared = reference.dense_mlp(
        x_mx, shard, f"{reference.PREFIX}.layers.1.mlp.shared_experts",
    )
    teacher_routed = (teacher_feed - shared).astype(mx.bfloat16)
    native_sentinel = native_expert(
        x_mx, shard, 1, sentinel, np.arange(sequence, dtype=np.int32),
    )
    native_sentinel_np = np.asarray(native_sentinel.astype(mx.float32), dtype=np.float32)
    tokens, slots = np.where(indices == sentinel)
    delta = mx.zeros_like(x_mx)
    if tokens.size:
        native_selected = mx.take(native_sentinel, mx.array(tokens.astype(np.int32)), axis=0)
        compact_selected = mx.array(compact_output[tokens]).astype(native_selected.dtype)
        route_weight = mx.array(weights[tokens, slots], dtype=native_selected.dtype)[:, None]
        scatter = mx.array(np.eye(sequence, dtype=np.float32)[tokens].T,
                           dtype=native_selected.dtype)
        delta = scatter @ ((compact_selected - native_selected) * route_weight)
        mx.eval(delta)
    student_routed = teacher_routed + delta
    student_feed = teacher_feed + delta
    teacher_hidden = (mx.array(post).astype(mx.bfloat16) + teacher_feed).astype(mx.bfloat16)
    student_hidden = (mx.array(post).astype(mx.bfloat16) + student_feed).astype(mx.bfloat16)
    mx.eval(teacher_hidden, student_hidden)
    arrays = {
        "teacher_hidden_l1": np.asarray(teacher_hidden.astype(mx.float32), dtype=np.float32),
        "student_hidden_l1": np.asarray(student_hidden.astype(mx.float32), dtype=np.float32),
        "teacher_routed_l1": np.asarray(teacher_routed.astype(mx.float32), dtype=np.float32),
        "student_routed_l1": np.asarray(student_routed.astype(mx.float32), dtype=np.float32),
        "shared_l1": np.asarray(shared.astype(mx.float32), dtype=np.float32),
        "native_sentinel_l1": native_sentinel_np,
    }
    sentinel_mask = indices == sentinel
    details = {"used_experts": sorted(set(int(value) for value in indices.flat)),
               "sentinel_route_slots": int(np.sum(sentinel_mask)),
               "sentinel_routed_tokens": int(np.sum(np.any(sentinel_mask, axis=1))),
               "native_sentinel_vs_compact": f1.quality(native_sentinel_np, compact_output),
               "routed_output": f1.quality(arrays["teacher_routed_l1"],
                                           arrays["student_routed_l1"]),
               "residual_hidden": f1.quality(arrays["teacher_hidden_l1"],
                                             arrays["student_hidden_l1"]),
               "config_top_k": int(config["num_experts_per_tok"])}
    del shard, x_mx, native_sentinel, teacher_routed, student_routed, shared, delta
    gc.collect()
    mx.clear_cache()
    return arrays, details


def pre_moe(
    hidden: np.ndarray, shard: reference.TensorShard, layer: int,
    config: dict[str, Any], segment_lengths: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    hidden_mx = mx.array(hidden).astype(mx.bfloat16)
    base = f"{reference.PREFIX}.layers.{layer}"
    normalized = reference.rms_norm(
        hidden_mx, shard.mlx(base + ".input_layernorm.weight"),
        float(config["rms_norm_eps"]),
    )
    attention, _ = reference.attention(normalized, shard, layer, config, segment_lengths)
    post = (hidden_mx + attention).astype(mx.bfloat16)
    x = reference.rms_norm(
        post, shard.mlx(base + ".post_attention_layernorm.weight"),
        float(config["rms_norm_eps"]),
    )
    mx.eval(post, x)
    return (np.asarray(post.astype(mx.float32), dtype=np.float32),
            np.asarray(x.astype(mx.float32), dtype=np.float32))


def route_pair_metrics(teacher: dict[str, np.ndarray], student: dict[str, np.ndarray]) -> dict[str, Any]:
    teacher_indices, student_indices = teacher["indices"], student["indices"]
    exact = np.array([set(left) == set(right)
                      for left, right in zip(teacher_indices, student_indices, strict=True)])
    jaccard = np.array([len(set(left) & set(right)) / len(set(left) | set(right))
                        for left, right in zip(teacher_indices, student_indices, strict=True)])
    position = np.mean(teacher_indices == student_indices, axis=1)
    rank_concordance = []
    entries = []
    exits = []
    weight_l1 = []
    for teacher_row, student_row, teacher_weight, student_weight in zip(
        teacher_indices, student_indices, teacher["weights"], student["weights"], strict=True,
    ):
        common = list(set(teacher_row) & set(student_row))
        pairs = 0
        concordant = 0
        teacher_position = {int(value): index for index, value in enumerate(teacher_row)}
        student_position = {int(value): index for index, value in enumerate(student_row)}
        for left_index, left in enumerate(common):
            for right in common[left_index + 1:]:
                pairs += 1
                concordant += int((teacher_position[left] - teacher_position[right]) *
                                  (student_position[left] - student_position[right]) > 0)
        rank_concordance.append(concordant / pairs if pairs else 1.0)
        entries.append(sorted(int(value) for value in set(student_row) - set(teacher_row)))
        exits.append(sorted(int(value) for value in set(teacher_row) - set(student_row)))
        teacher_map = {int(value): float(weight) for value, weight in zip(
            teacher_row, teacher_weight, strict=True)}
        student_map = {int(value): float(weight) for value, weight in zip(
            student_row, student_weight, strict=True)}
        union = set(teacher_map) | set(student_map)
        weight_l1.append(sum(abs(teacher_map.get(value, 0) - student_map.get(value, 0))
                             for value in union))
    return {"exact": exact, "jaccard": jaccard, "position_agreement": position,
            "rank_concordance": np.asarray(rank_concordance),
            "entries": entries, "exits": exits, "combine_weight_l1": np.asarray(weight_l1)}


def domain_summary(
    token_records: list[dict[str, Any]], route_metrics: dict[str, Any],
    teacher_hidden: np.ndarray, student_hidden: np.ndarray,
) -> dict[str, Any]:
    rows = row_metrics(teacher_hidden, student_hidden)
    domains = sorted(set(record["domain"] for record in token_records))
    result = {}
    for domain in domains:
        mask = np.array([record["domain"] == domain for record in token_records])
        result[domain] = {
            "tokens": int(np.sum(mask)), "route_set_agreement": float(np.mean(route_metrics["exact"][mask])),
            "route_jaccard": float(np.mean(route_metrics["jaccard"][mask])),
            "rank_concordance": float(np.mean(route_metrics["rank_concordance"][mask])),
            "combine_weight_l1": float(np.mean(route_metrics["combine_weight_l1"][mask])),
            "hidden_cosine": float(np.mean(rows["cosine"][mask])),
            "hidden_relative_l2": float(np.mean(rows["relative_l2"][mask])),
        }
    return result


def propagate_layer_three(
    source: Path, variants: dict[str, np.ndarray], config: dict[str, Any],
    segment_lengths: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    shard = reference.TensorShard(reference.shard_path(source, 4))
    outputs = {}
    infos = {}
    for name, hidden in variants.items():
        output, info = reference.layer_forward(
            mx.array(hidden).astype(mx.bfloat16), shard, 3, config, segment_lengths,
        )
        outputs[name] = np.asarray(output.astype(mx.float32), dtype=np.float32)
        infos[name] = info
        mx.clear_cache()
    teacher_routes = infos["TEACHER"]["moe"]["route_indices"]
    metrics = {}
    for name, output in outputs.items():
        routes = infos[name]["moe"]["route_indices"]
        agreement = np.mean([set(left) == set(right)
                             for left, right in zip(teacher_routes, routes, strict=True)])
        metrics[name] = {"hidden": f1.quality(outputs["TEACHER"], output),
                         "route_set_agreement": float(agreement),
                         "used_expert_count": infos[name]["moe"]["used_expert_count"]}
    del shard
    gc.collect()
    mx.clear_cache()
    return outputs, metrics


def causal_layer_two(
    source: Path, teacher_hidden: np.ndarray, student_hidden: np.ndarray,
    segment_lengths: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    config = f1.read_json(source / "config.json")["text_config"]
    shard = reference.TensorShard(reference.shard_path(source, 3))
    teacher_post, teacher_x = pre_moe(teacher_hidden, shard, 2, config, segment_lengths)
    student_post, student_x = pre_moe(student_hidden, shard, 2, config, segment_lengths)
    teacher_route = route_diagnostics(teacher_x, shard, 2, config)
    student_route = route_diagnostics(student_x, shard, 2, config)
    teacher_cache = build_output_cache(
        teacher_x, [teacher_route["indices"], student_route["indices"]], shard, 2,
    )
    student_cache = build_output_cache(
        student_x, [teacher_route["indices"], student_route["indices"]], shard, 2,
    )
    teacher_shared = np.asarray(reference.dense_mlp(
        mx.array(teacher_x).astype(mx.bfloat16), shard,
        f"{reference.PREFIX}.layers.2.mlp.shared_experts",
    ).astype(mx.float32), dtype=np.float32)
    student_shared = np.asarray(reference.dense_mlp(
        mx.array(student_x).astype(mx.bfloat16), shard,
        f"{reference.PREFIX}.layers.2.mlp.shared_experts",
    ).astype(mx.float32), dtype=np.float32)
    teacher_feed_mx, _ = reference.routed_moe(
        mx.array(teacher_x).astype(mx.bfloat16), shard, 2, config,
    )
    student_feed_mx, _ = reference.routed_moe(
        mx.array(student_x).astype(mx.bfloat16), shard, 2, config,
    )
    teacher_feed = np.asarray(teacher_feed_mx.astype(mx.float32), dtype=np.float32)
    student_feed = np.asarray(student_feed_mx.astype(mx.float32), dtype=np.float32)
    teacher_routed = bf16(teacher_feed - teacher_shared)
    student_routed = bf16(student_feed - student_shared)
    forced_indices_weights = weights_for_indices(
        student_route["scores"], teacher_route["indices"], config,
    )
    forced_indices = combine_cached(student_cache, teacher_route["indices"], forced_indices_weights)
    forced_both = combine_cached(student_cache, teacher_route["indices"], teacher_route["weights"])
    teacher_student_router = combine_cached(
        teacher_cache, student_route["indices"], student_route["weights"],
    )
    manual_teacher_natural = combine_cached(
        teacher_cache, teacher_route["indices"], teacher_route["weights"],
    )
    manual_student_natural = combine_cached(
        student_cache, student_route["indices"], student_route["weights"],
    )

    variants = {
        "TEACHER": hidden_from_feed(teacher_post, teacher_feed),
        "NATURAL_STUDENT": hidden_from_feed(student_post, student_feed),
        "FORCE_TEACHER_INDICES": final_hidden(student_post, forced_indices, student_shared),
        "FORCE_TEACHER_INDICES_AND_WEIGHTS": final_hidden(
            student_post, forced_both, student_shared,
        ),
        "TEACHER_STATE_STUDENT_ROUTER": final_hidden(
            teacher_post, teacher_student_router, teacher_shared,
        ),
        "SUBSTITUTE_TEACHER_WEIGHTED_MOE": hidden_from_feed(student_post, teacher_feed),
        "RESTORE_TEACHER_HIDDEN_BEFORE_LAYER2": None,
        "RESTORE_TEACHER_HIDDEN_AFTER_LAYER2": None,
    }
    variants["RESTORE_TEACHER_HIDDEN_BEFORE_LAYER2"] = variants["TEACHER"].copy()
    variants["RESTORE_TEACHER_HIDDEN_AFTER_LAYER2"] = variants["TEACHER"].copy()
    pair = route_pair_metrics(teacher_route, student_route)
    baseline_error = f1.quality(variants["TEACHER"], variants["NATURAL_STUDENT"])["relative_l2"]
    intervention = {}
    for name, hidden in variants.items():
        metric = f1.quality(variants["TEACHER"], hidden)
        intervention[name] = {"layer2_hidden": metric,
                              "rescue_fraction_relative_l2": float(
                                  1 - metric["relative_l2"] / (baseline_error + 1e-30))}
    arrays = {"teacher_post_l2": teacher_post, "student_post_l2": student_post,
              "teacher_x_l2": teacher_x, "student_x_l2": student_x,
              "teacher_routed_l2": teacher_routed, "student_routed_l2": student_routed,
              **{f"variant_l2_{name}": value for name, value in variants.items()}}
    details = {
        "teacher_route": teacher_route, "student_route": student_route,
        "route_pair": pair, "teacher_routed": teacher_routed, "student_routed": student_routed,
        "intervention": intervention,
        "routed_output": f1.quality(teacher_routed, student_routed),
        "post_attention_residual": f1.quality(teacher_post, student_post),
        "expert_input": f1.quality(teacher_x, student_x),
        "counterfactual_recombination_calibration": {
            "teacher_manual_vs_resident_routed": f1.quality(
                teacher_routed, manual_teacher_natural,
            ),
            "student_manual_vs_resident_routed": f1.quality(
                student_routed, manual_student_natural,
            ),
            "natural_paths_use_bit_exact_resident_forward": True,
            "manual_recombination_used_only_for_interventions": True,
        },
    }
    del shard, teacher_cache, student_cache
    gc.collect()
    mx.clear_cache()
    return arrays, details


def strip_route_arrays(route: dict[str, np.ndarray]) -> dict[str, Any]:
    def summary(values: np.ndarray) -> dict[str, Any]:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return {"active": False, "reason": "GROUP_PRUNING_INACTIVE"}
        return {"active": True, "mean": float(np.mean(finite)),
                "p10": float(np.percentile(finite, 10)), "min": float(np.min(finite))}

    return {"margin_8v9": summary(route["margin_8v9"]),
            "group_margin": summary(route["group_margin"])}


def token_atlas(
    records: list[dict[str, Any]], teacher_route: dict[str, np.ndarray],
    student_route: dict[str, np.ndarray], pair: dict[str, Any],
    teacher_routed: np.ndarray, student_routed: np.ndarray,
    teacher_l2: np.ndarray, student_l2: np.ndarray,
    teacher_l3: np.ndarray, student_l3: np.ndarray,
) -> list[dict[str, Any]]:
    moe_rows = row_metrics(teacher_routed, student_routed)
    residual_rows = row_metrics(teacher_l2, student_l2)
    next_rows = row_metrics(teacher_l3, student_l3)
    result = []
    for index, record in enumerate(records):
        result.append({
            **record,
            "teacher_margin_8v9": float(teacher_route["margin_8v9"][index]),
            "student_margin_8v9": float(student_route["margin_8v9"][index]),
            "route_set_agreement": bool(pair["exact"][index]),
            "route_jaccard": float(pair["jaccard"][index]),
            "rank_position_agreement": float(pair["position_agreement"][index]),
            "rank_concordance": float(pair["rank_concordance"][index]),
            "expert_entries": pair["entries"][index], "expert_exits": pair["exits"][index],
            "combine_weight_l1": float(pair["combine_weight_l1"][index]),
            "weighted_moe_relative_l2": float(moe_rows["relative_l2"][index]),
            "residual_cosine": float(residual_rows["cosine"][index]),
            "residual_relative_l2": float(residual_rows["relative_l2"][index]),
            "next_layer_cosine": float(next_rows["cosine"][index]),
            "next_layer_relative_l2": float(next_rows["relative_l2"][index]),
        })
    return result


def first_index(atlas: list[dict[str, Any]], predicate: Any) -> dict[str, Any] | None:
    return next((row for row in atlas if predicate(row)), None)


def classify_causality(interventions: dict[str, Any]) -> tuple[str, dict[str, float]]:
    index_rescue = interventions["FORCE_TEACHER_INDICES"]["rescue_fraction_relative_l2"]
    weight_rescue = interventions[
        "FORCE_TEACHER_INDICES_AND_WEIGHTS"
    ]["rescue_fraction_relative_l2"]
    moe_rescue = interventions["SUBSTITUTE_TEACHER_WEIGHTED_MOE"][
        "rescue_fraction_relative_l2"
    ]
    if weight_rescue >= 0.50:
        diagnosis = "ROUTE_DRIFT_PRIMARY"
    elif weight_rescue <= 0.20 and moe_rescue >= 0.50:
        diagnosis = "ROUTE_DRIFT_SECONDARY_TO_STATE_OR_EXPERT_OUTPUT"
    else:
        diagnosis = "MIXED_ROUTE_AND_STATE_DAMAGE"
    return diagnosis, {"indices_only_rescue": index_rescue,
                       "indices_plus_weights_rescue": weight_rescue,
                       "teacher_moe_substitution_rescue": moe_rescue}


def recover_cached_paths(
    source: Path, capture_path: Path, segment_lengths: list[int],
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray],
    dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray],
    dict[str, Any], dict[str, Any], dict[str, Any],
]:
    """Recover LR01 analysis without repeating any parent/expert forward."""
    with np.load(capture_path, allow_pickle=False) as capture:
        values = {key: np.asarray(capture[key]) for key in capture.files}
    post_l1 = values["post_l1"]
    x_l1 = values["x_l1"]
    base_output = values["base_expert0_l1"]
    compact_output = values["compact_expert0_l1"]
    layer1_keys = (
        "teacher_hidden_l1", "student_hidden_l1", "teacher_routed_l1",
        "student_routed_l1", "shared_l1", "native_sentinel_l1",
    )
    layer1_arrays = {key: values[key] for key in layer1_keys}
    layer2_arrays = {key: value for key, value in values.items()
                     if key.startswith("variant_l2_") or key in {
                         "teacher_post_l2", "student_post_l2", "teacher_x_l2",
                         "student_x_l2", "teacher_routed_l2", "student_routed_l2",
                     }}
    layer2_variants = {key.removeprefix("variant_l2_"): value
                       for key, value in layer2_arrays.items()
                       if key.startswith("variant_l2_")}
    layer3_outputs = {key.removeprefix("variant_l3_"): value
                      for key, value in values.items() if key.startswith("variant_l3_")}
    config = f1.read_json(source / "config.json")["text_config"]
    shard_l1 = reference.TensorShard(reference.shard_path(source, 2))
    routes_l1 = route_diagnostics(x_l1, shard_l1, 1, config)
    del shard_l1
    sentinel_mask = routes_l1["indices"] == 0
    layer1_details = {
        "used_experts": sorted(set(int(value) for value in routes_l1["indices"].flat)),
        "sentinel_route_slots": int(np.sum(sentinel_mask)),
        "sentinel_routed_tokens": int(np.sum(np.any(sentinel_mask, axis=1))),
        "native_sentinel_vs_compact": f1.quality(
            layer1_arrays["native_sentinel_l1"], compact_output,
        ),
        "routed_output": f1.quality(
            layer1_arrays["teacher_routed_l1"], layer1_arrays["student_routed_l1"],
        ),
        "residual_hidden": f1.quality(
            layer1_arrays["teacher_hidden_l1"], layer1_arrays["student_hidden_l1"],
        ),
        "config_top_k": int(config["num_experts_per_tok"]),
    }
    shard_l2 = reference.TensorShard(reference.shard_path(source, 3))
    teacher_route = route_diagnostics(values["teacher_x_l2"], shard_l2, 2, config)
    student_route = route_diagnostics(values["student_x_l2"], shard_l2, 2, config)
    del shard_l2
    pair = route_pair_metrics(teacher_route, student_route)
    baseline_error = f1.quality(
        layer2_variants["TEACHER"], layer2_variants["NATURAL_STUDENT"],
    )["relative_l2"]
    intervention = {}
    for name, hidden in layer2_variants.items():
        metric = f1.quality(layer2_variants["TEACHER"], hidden)
        intervention[name] = {
            "layer2_hidden": metric,
            "rescue_fraction_relative_l2": float(
                1 - metric["relative_l2"] / (baseline_error + 1e-30)
            ),
        }
    layer2_details = {
        "teacher_route": teacher_route, "student_route": student_route,
        "route_pair": pair, "teacher_routed": values["teacher_routed_l2"],
        "student_routed": values["student_routed_l2"], "intervention": intervention,
        "routed_output": f1.quality(
            values["teacher_routed_l2"], values["student_routed_l2"],
        ),
        "post_attention_residual": f1.quality(
            values["teacher_post_l2"], values["student_post_l2"],
        ),
        "expert_input": f1.quality(values["teacher_x_l2"], values["student_x_l2"]),
        "counterfactual_recombination_calibration": {
            "natural_paths_use_bit_exact_resident_forward": True,
            "manual_recombination_used_only_for_interventions": True,
            "same_kernel_full_batch_layer1_relative_l2_error_floor": 0.001254,
            "provenance": "REJECTED_ATTEMPT_1_RESIDENT_VS_MANUAL_CALIBRATION",
            "interpretation_rule": "DO_NOT_INTERPRET_RESCUE_BELOW_NUMERICAL_ERROR_FLOOR",
        },
    }
    shard_l3 = reference.TensorShard(reference.shard_path(source, 4))
    variant_routes = {}
    for name, hidden in layer2_variants.items():
        _, expert_input = pre_moe(hidden, shard_l3, 3, config, segment_lengths)
        variant_routes[name] = route_diagnostics(expert_input, shard_l3, 3, config)["indices"]
        mx.clear_cache()
    del shard_l3
    teacher_routes_l3 = variant_routes["TEACHER"]
    layer3_metrics = {}
    for name, output in layer3_outputs.items():
        agreement = np.mean([
            set(left) == set(right) for left, right in zip(
                teacher_routes_l3, variant_routes[name], strict=True,
            )
        ])
        layer3_metrics[name] = {
            "hidden": f1.quality(layer3_outputs["TEACHER"], output),
            "route_set_agreement": float(agreement),
            "used_expert_count": len(set(int(value) for value in variant_routes[name].flat)),
        }
    input_info = {
        "recovered_from_raw_checkpoint": True,
        "recovery_scope": "ROUTER_AND_ANALYSIS_ONLY_NO_PARENT_OR_EXPERT_FORWARD_REPEATED",
    }
    return (post_l1, x_l1, base_output, compact_output, layer1_arrays,
            layer1_details, layer2_arrays, layer2_variants, layer3_outputs,
            layer2_details, layer3_metrics, input_info)


def run(
    repo: Path, source: Path, output_dir: Path, seed: int, retry: int,
    resume_capture: bool,
) -> dict[str, Any]:
    started_wall = time.time()
    started_at = f1.now()
    before = manager.resource_snapshot()
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_path = output_dir / "LR01_CONTROL_CAPTURE.npz"
    prior_status = f1.read_json(repo / manager.STATUS_JSON)
    manager.write_status(repo, {
        **prior_status, "status": "RUNNING_EXPERIMENT",
        "active_experiment": "LR01_HELDOUT_090859_CONTROL_CAUSAL_ATLAS",
        "next_experiment": "LR01_IN_PROGRESS", "resources": before,
    })
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_START",
        "experiment_id": "LR01_HELDOUT_090859_CONTROL_CAUSAL_ATLAS",
        "started_at": started_at, "status": "RUNNING", "config": {"seed": seed},
        "hypothesis": "Compact hidden drift crosses native router margins causally.",
    })
    candidate_payload = (manager.RUNTIME / "f1_representation_bracket/doctor_auction" /
                         "P1_DUAL_PATH_RECOVERY_R16X2.k26f1")
    candidate_result = f1.read_json(
        manager.RUNTIME / "f1_representation_bracket/doctor_auction" /
        "P1_DUAL_PATH_RECOVERY_R16X2_RESULT.json",
    )
    if f1.sha256_file(candidate_payload) != candidate_result["payload"]["sha256"]:
        raise f1.F1Error("candidate payload hash mismatch")
    requests, batch = prepare_requests(source)
    token_ids = batch["token_ids"]
    segment_lengths = batch["segment_lengths"]
    config = f1.read_json(source / "config.json")["text_config"]
    if resume_capture:
        if not capture_path.is_file():
            raise f1.F1Error("requested LR01 cache recovery but checkpoint is missing")
        (post_l1, x_l1, base_output, compact_output, layer1_arrays,
         layer1_details, layer2_arrays, layer2_variants, layer3_outputs,
         layer2_details, layer3_metrics, input_info) = recover_cached_paths(
             source, capture_path, segment_lengths,
         )
        shard_l1 = reference.TensorShard(reference.shard_path(source, 2))
        routes_l1 = route_diagnostics(x_l1, shard_l1, 1, config)
        del shard_l1
    else:
        post_l1, x_l1, input_info = capture_layer_one_inputs(
            source, token_ids, segment_lengths,
        )
        shard_l1 = reference.TensorShard(reference.shard_path(source, 2))
        routes_l1 = route_diagnostics(x_l1, shard_l1, 1, config)
        del shard_l1
        base_output, compact_output = compact_expert_output(candidate_payload, x_l1)
        layer1_arrays, layer1_details = layer_one_paths(
            source, post_l1, x_l1, routes_l1, compact_output,
        )
        layer2_arrays, layer2_details = causal_layer_two(
            source, layer1_arrays["teacher_hidden_l1"], layer1_arrays["student_hidden_l1"],
            segment_lengths,
        )
        layer2_variants = {key.removeprefix("variant_l2_"): value
                           for key, value in layer2_arrays.items()
                           if key.startswith("variant_l2_")}
        layer3_outputs, layer3_metrics = propagate_layer_three(
            source, layer2_variants, config, segment_lengths,
        )
    teacher_route_l2 = layer2_details["teacher_route"]
    student_route_l2 = layer2_details["student_route"]
    pair = layer2_details["route_pair"]
    atlas = token_atlas(
        batch["token_records"], teacher_route_l2, student_route_l2, pair,
        layer2_details["teacher_routed"], layer2_details["student_routed"],
        layer2_variants["TEACHER"], layer2_variants["NATURAL_STUDENT"],
        layer3_outputs["TEACHER"], layer3_outputs["NATURAL_STUDENT"],
    )
    exact = pair["exact"].astype(np.float64)
    mismatch = 1 - exact
    route_ci = bootstrap_interval(mismatch, seed + 1)
    hidden_rows_l2 = row_metrics(
        layer2_variants["TEACHER"], layer2_variants["NATURAL_STUDENT"],
    )
    hidden_ci = bootstrap_interval(hidden_rows_l2["cosine"], seed + 2)
    margin_quartiles = np.quantile(teacher_route_l2["margin_8v9"], [0.25, 0.5, 0.75])
    margin_bins = np.digitize(teacher_route_l2["margin_8v9"], margin_quartiles)
    mismatch_by_margin = {str(bin_index): {
        "tokens": int(np.sum(margin_bins == bin_index)),
        "mismatch_rate": float(np.mean(mismatch[margin_bins == bin_index]))
    } for bin_index in range(4)}
    diagnosis, rescue = classify_causality(layer2_details["intervention"])
    first_route = first_index(atlas, lambda row: not row["route_set_agreement"])
    first_moe = first_index(
        atlas, lambda row: row["weighted_moe_relative_l2"] >= MATERIAL_MOE_RELATIVE_L2,
    )
    first_irrecoverable = first_index(
        atlas, lambda row: row["next_layer_cosine"] < IRRECOVERABLE_ROW_COSINE,
    )
    reference_checkpoint = f1.read_json(
        manager.RUNTIME / "reference_run/checkpoint_batch_581f589b3bccdea1/checkpoint.json",
    )
    reference_match = {
        "token_ids": reference_checkpoint.get("token_ids") == token_ids,
        "layer1_hidden_sha256": hidden_hash(layer1_arrays["teacher_hidden_l1"]),
        "expected_layer1_hidden_sha256": reference_checkpoint["layers"][1]["hidden_sha256"],
        "layer2_hidden_sha256": hidden_hash(layer2_variants["TEACHER"]),
        "expected_layer2_hidden_sha256": reference_checkpoint["layers"][2]["hidden_sha256"],
    }
    reference_match["layer1_exact"] = (
        reference_match["layer1_hidden_sha256"] == reference_match["expected_layer1_hidden_sha256"])
    reference_match["layer2_exact"] = (
        reference_match["layer2_hidden_sha256"] == reference_match["expected_layer2_hidden_sha256"])
    instrumentation_valid = all((reference_match["token_ids"], reference_match["layer1_exact"],
                                 reference_match["layer2_exact"]))
    if not instrumentation_valid:
        raise f1.F1Error(f"reference instrumentation mismatch: {reference_match}")

    domain = domain_summary(
        batch["token_records"], pair,
        layer2_variants["TEACHER"], layer2_variants["NATURAL_STUDENT"],
    )
    score_route_tokens = np.any(routes_l1["indices"] == 0, axis=1)
    local_metrics = {
        "all_heldout_context_tokens": f1.quality(
            layer1_arrays["native_sentinel_l1"], compact_output,
        ),
        "actually_routed_tokens": (f1.quality(
            layer1_arrays["native_sentinel_l1"][score_route_tokens],
            compact_output[score_route_tokens],
        ) if np.any(score_route_tokens) else {"status": "NO_SENTINEL_ROUTES"}),
        "routed_token_count": int(np.sum(score_route_tokens)),
    }
    after = manager.resource_snapshot()
    ended_at = f1.now()
    duration = time.time() - started_wall
    arrays_to_save = {
        "post_l1": post_l1, "x_l1": x_l1, "base_expert0_l1": base_output,
        "compact_expert0_l1": compact_output, **layer1_arrays, **layer2_arrays,
        **{f"variant_l3_{name}": value for name, value in layer3_outputs.items()},
        "teacher_route_indices_l2": teacher_route_l2["indices"],
        "student_route_indices_l2": student_route_l2["indices"],
        "teacher_route_weights_l2": teacher_route_l2["weights"],
        "student_route_weights_l2": student_route_l2["weights"],
        "teacher_margin_l2": teacher_route_l2["margin_8v9"],
        "student_margin_l2": student_route_l2["margin_8v9"],
    }
    if not resume_capture:
        temporary = capture_path.with_name(f".{capture_path.name}.{os.getpid()}.tmp")
        with temporary.open("xb") as handle:
            np.savez(handle, **arrays_to_save)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, capture_path)
    capture_sha = f1.sha256_file(capture_path)
    route_agreement = float(np.mean(pair["exact"]))
    artifact = f1.seal({
        "schema": "hawking.kimi_k26.long_run_causal_control.v1", "status": "PASS",
        "experiment_id": "LR01_HELDOUT_090859_CONTROL_CAUSAL_ATLAS",
        "hypothesis": (
            "The 0.90859-BPW compact perturbation crosses small native router margins and route "
            "selection is a causal contributor, not merely a correlate, to downstream damage."
        ),
        "config": {"seed": seed, "split": "EIGHT_QUARANTINED_REFERENCE_PROBES",
                   "layers": [1, 2, 3], "top_k": 8, "candidate_expert": 0,
                   "resumed_from_raw_checkpoint": resume_capture},
        "source": {"repo": f1.REPO, "revision": f1.REVISION},
        "parent_reference_checkpoint_sha256": f1.sha256_file(
            manager.RUNTIME / "reference_run/checkpoint_batch_581f589b3bccdea1/checkpoint.json"
        ),
        "candidate": {"name": "P1_DUAL_PATH_RECOVERY_R16X2",
                      "payload_sha256": candidate_result["payload"]["sha256"],
                      "physical_bytes": candidate_result["payload"]["bytes"],
                      "complete_bpw": candidate_result["physical_budget"]["actual_complete_bpw"],
                      "doctor_bytes": candidate_result["doctor"]["doctor_component_bytes"]},
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resource_observations": {"before": before, "after": after},
        "sample_counts": {"segments": len(requests), "tokens": len(token_ids),
                          "domains": len(set(DOMAINS.values())),
                          "sentinel_routed_tokens": int(np.sum(score_route_tokens)),
                          "layer2_route_mismatches": int(np.sum(mismatch))},
        "reference_instrumentation": reference_match,
        "input_capture": input_info,
        "heldout_control": local_metrics,
        "layer_routing_atlas": {
            "layer1": {"route_set_agreement": 1.0,
                       "reason": "compact perturbation is installed after the layer-1 router",
                       **strip_route_arrays(routes_l1)},
            "layer2": {"route_set_agreement": route_agreement,
                       "route_set_change": 1 - route_agreement,
                       "jaccard_mean": float(np.mean(pair["jaccard"])),
                       "rank_position_agreement": float(np.mean(pair["position_agreement"])),
                       "rank_concordance": float(np.mean(pair["rank_concordance"])),
                       "combine_weight_l1": float(np.mean(pair["combine_weight_l1"])),
                       "teacher_margin": strip_route_arrays(teacher_route_l2),
                       "student_margin": strip_route_arrays(student_route_l2),
                       "mismatch_by_teacher_margin_quartile": mismatch_by_margin,
                       "route_change_ci95": route_ci},
            "layer3": layer3_metrics,
        },
        "moe_residual_f2": {
            "layer1": layer1_details,
            "layer2_post_attention": layer2_details["post_attention_residual"],
            "layer2_expert_input": layer2_details["expert_input"],
            "layer2_weighted_moe": layer2_details["routed_output"],
            "layer2_natural_hidden": layer2_details["intervention"]["NATURAL_STUDENT"],
            "layer2_hidden_cosine_ci95": hidden_ci,
            "layer3_natural": layer3_metrics["NATURAL_STUDENT"],
            "counterfactual_recombination_calibration": layer2_details[
                "counterfactual_recombination_calibration"
            ],
        },
        "intervention_matrix": layer2_details["intervention"],
        "causal_rescue": rescue,
        "primary_causal_diagnosis": diagnosis,
        "first_divergences": {"route": first_route, "material_weighted_moe": first_moe,
                              "irrecoverable_residual": first_irrecoverable},
        "domain_summary": domain, "token_atlas": atlas,
        "capture": {"path": str(capture_path), "bytes": capture_path.stat().st_size,
                    "sha256": capture_sha},
        "faults": (KNOWN_RETRY_FAULTS + [
            "ATTEMPT_2_INACTIVE_GROUP_MARGIN_JSON_SERIALIZATION",
        ])[:retry],
        "retries": retry,
        "causal_interpretation": (
            f"{diagnosis}: forcing teacher indices rescues {rescue['indices_only_rescue']:.3f} "
            f"and forcing indices+weights rescues {rescue['indices_plus_weights_rescue']:.3f} "
            "of layer-2 relative-L2 damage; teacher weighted-MoE substitution gives the causal "
            f"upper-bound rescue {rescue['teacher_moe_substitution_rescue']:.3f}."
        ),
        "decision": "ADVANCE_TO_CAUSALLY_TARGETED_PHYSICAL_BRACKET",
        "next_run_rationale": (
            "Use measured first-divergence margins and intervention rescue to allocate the remaining "
            "0.98-BPW bytes among pre-router and weighted-MoE repair, not generic compression."
        ),
    })
    artifact_path = repo / "KIMI_K26_LONG_RUN_CONTROL_CAUSAL.json"
    f1.atomic_json(artifact_path, artifact)
    f1.atomic_json(manager.RUNTIME / artifact_path.name, artifact)
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_COMPLETE",
        "experiment_id": artifact["experiment_id"], "hypothesis": artifact["hypothesis"],
        "config": artifact["config"], "parent_hash": artifact["parent_reference_checkpoint_sha256"],
        "candidate_hash": artifact["candidate"]["payload_sha256"],
        "physical_bytes": artifact["candidate"]["physical_bytes"],
        "complete_bpw": artifact["candidate"]["complete_bpw"],
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resources": artifact["resource_observations"], "sample_counts": artifact["sample_counts"],
        "metrics": {"route_set_agreement": route_agreement,
                    "layer2_hidden": artifact["moe_residual_f2"]["layer2_natural_hidden"],
                    "layer3": artifact["moe_residual_f2"]["layer3_natural"]},
        "confidence_intervals": {"route_change": route_ci, "hidden_cosine": hidden_ci},
        "faults": artifact["faults"], "retries": retry,
        "causal_interpretation": artifact["causal_interpretation"],
        "decision": artifact["decision"], "next_run_rationale": artifact["next_run_rationale"],
        "evidence_seal_sha256": artifact["seal_sha256"],
    })
    prior = f1.read_json(repo / manager.STATUS_JSON)
    status = manager.write_status(repo, {
        **prior, "status": "MANAGING", "active_experiment": None,
        "experiments_completed": int(prior.get("experiments_completed", 0)) + 1,
        "primary_causal_diagnosis": diagnosis,
        "next_experiment": "LR02_CAUSALLY_TARGETED_PHYSICAL_BRACKET",
        "latest_result": {"experiment_id": artifact["experiment_id"],
                          "route_set_change": 1 - route_agreement,
                          "diagnosis": diagnosis, "rescue": rescue,
                          "evidence_seal_sha256": artifact["seal_sha256"]},
        "resources": after,
    })
    telegram_receipt = manager.telegram(
        repo, "long-run:LR01-complete",
        ("[Kimi K2.6 long run] LR01 causal control complete\n"
         f"tokens/domains: {len(token_ids)} / {len(set(DOMAINS.values()))}\n"
         f"layer2 route change: {(1-route_agreement)*100:.2f}%\n"
         f"diagnosis: {diagnosis}\n"
         f"teacher-index rescue: {rescue['indices_only_rescue']:.3f}\n"
         f"teacher-index+weight rescue: {rescue['indices_plus_weights_rescue']:.3f}\n"
         f"teacher-MoE rescue: {rescue['teacher_moe_substitution_rescue']:.3f}\n"
         f"free disk: {after['free_disk_bytes']/1024**3:.2f} GiB\n"
         "next: causally targeted <=0.98-BPW repair bracket"),
    )
    status = manager.write_status(repo, {
        **status, "latest_result": {**status["latest_result"],
                                    "telegram_delivered": telegram_receipt["delivered"],
                                    "telegram_receipt_seal_sha256": telegram_receipt["seal_sha256"]},
    })
    return {"artifact": artifact, "status": status}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=26072101)
    parser.add_argument("--retry", type=int, default=0)
    parser.add_argument("--resume-capture", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args.repo.resolve(strict=True), args.source.resolve(strict=True),
                     args.output_dir.resolve(), args.seed, args.retry,
                     args.resume_capture)
        print(json.dumps(result["artifact"], sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
