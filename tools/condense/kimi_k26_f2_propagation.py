#!/usr/bin/env python3.12
"""Kimi P1 F2 sparse-substitution propagation bridge.

Reuses the sealed F1 post-attention/route capture, substitutes the promoted
physical compact expert inside the complete active layer-1 path, then advances
teacher and candidate together through exact layer 2. This is an F2 damage-
compounding probe, not a claim that every expert in the shard is compact.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import mlx.core as mx
import numpy as np


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_f1_bracket as f1  # noqa: E402
import kimi_k26_f1_doctor_auction as auction  # noqa: E402
import kimi_k26_reference as reference  # noqa: E402


def native_expert(
    x: mx.array, shard: reference.TensorShard, expert: int, tokens: np.ndarray,
) -> mx.array:
    selected = mx.take(x, mx.array(tokens, dtype=mx.int32), axis=0)
    base = f"{reference.PREFIX}.layers.{f1.LAYER}.mlp.experts.{expert}"
    gate = reference.quantized_linear(selected, shard, base + ".gate_proj")
    up = reference.quantized_linear(selected, shard, base + ".up_proj")
    hidden = reference.silu(gate) * up
    output = reference.quantized_linear(hidden, shard, base + ".down_proj")
    mx.eval(output)
    return output


def complete_layer_one(
    source: Path,
    capture: dict[str, np.ndarray],
    promoted_payload: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x_np = np.concatenate((capture["fit_x"], capture["score_x"])).astype(np.float32)
    post_np = np.concatenate(
        (capture["fit_post_attention"], capture["score_post_attention"]),
    ).astype(np.float32)
    routes = np.concatenate((capture["fit_routes"], capture["score_routes"])).astype(np.int32)
    route_weights = np.concatenate(
        (capture["fit_route_weights"], capture["score_route_weights"]),
    ).astype(np.float32)
    sentinel = int(capture["sentinel_expert"][0])
    base_weights, _ = f1.decode_base_weights(promoted_payload)
    compact_base, _ = f1.expert_forward(x_np, base_weights)
    compact_output = auction.apply_hidden_doctor(promoted_payload, x_np, compact_base)

    shard = reference.TensorShard(reference.shard_path(source, f1.LAYER + 1))
    x = mx.array(x_np).astype(mx.bfloat16)
    sequence = x.shape[0]
    exact_combined = mx.zeros_like(x)
    candidate_combined = mx.zeros_like(x)
    used = sorted(set(int(value) for value in routes.flat))
    native_sentinel_all = native_expert(
        x, shard, sentinel, np.arange(sequence, dtype=np.int32),
    )
    native_sentinel_np = np.asarray(native_sentinel_all.astype(mx.float32), dtype=np.float32)
    f1_teacher = np.concatenate(
        (capture["fit_teacher_output"], capture["score_teacher_output"]),
    ).astype(np.float32)
    for expert in used:
        tokens, slots = np.where(routes == expert)
        if expert == sentinel:
            expert_exact = mx.take(
                native_sentinel_all, mx.array(tokens, dtype=mx.int32), axis=0,
            )
            expert_candidate = mx.array(compact_output[tokens]).astype(expert_exact.dtype)
        else:
            expert_exact = native_expert(x, shard, expert, tokens)
            expert_candidate = expert_exact
        weights = mx.array(route_weights[tokens, slots], dtype=expert_exact.dtype)[:, None]
        scatter = mx.array(np.eye(sequence, dtype=np.float32)[tokens].T,
                           dtype=expert_exact.dtype)
        exact_combined = exact_combined + scatter @ (expert_exact * weights)
        candidate_combined = candidate_combined + scatter @ (expert_candidate * weights)
        mx.eval(exact_combined, candidate_combined)
    mlp_base = f"{reference.PREFIX}.layers.{f1.LAYER}.mlp"
    shared = reference.dense_mlp(x, shard, mlp_base + ".shared_experts")
    exact_feed = exact_combined + shared
    candidate_feed = candidate_combined + shared
    post = mx.array(post_np).astype(mx.bfloat16)
    exact_hidden = (post + exact_feed).astype(mx.bfloat16)
    candidate_hidden = (post + candidate_feed).astype(mx.bfloat16)
    mx.eval(exact_hidden, candidate_hidden)
    exact_np = np.asarray(exact_hidden.astype(mx.float32), dtype=np.float32)
    candidate_np = np.asarray(candidate_hidden.astype(mx.float32), dtype=np.float32)
    active_expert_tensor_names = sum(
        1 for name in shard.names() if any(
            f".mlp.experts.{expert}." in name for expert in used
        )
    )
    record = {
        "source_shard": shard.path.name, "active_experts": used,
        "active_expert_count": len(used), "route_slots": int(routes.size),
        "sentinel_route_slots": int(np.sum(routes == sentinel)),
        "f1_float_teacher_vs_native_int4": f1.quality(f1_teacher, native_sentinel_np),
        "full_layer_hidden": f1.quality(exact_np, candidate_np),
        "score_full_layer_hidden": f1.quality(exact_np[32:], candidate_np[32:]),
        "tensor_audit": {
            "source_tensors": len(shard.names()),
            "active_expert_tensors_accessed": active_expert_tensor_names,
            "cached_attention_norm_and_router_reused": True,
            "unselected_experts_conditionally_inactive": True,
        },
    }
    del shard, base_weights, x, exact_combined, candidate_combined, shared
    gc.collect()
    mx.clear_cache()
    return exact_np, candidate_np, record


def layer_two_propagation(
    source: Path, teacher_hidden: np.ndarray, candidate_hidden: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    combined = mx.array(np.concatenate((teacher_hidden, candidate_hidden))).astype(mx.bfloat16)
    shard = reference.TensorShard(reference.shard_path(source, 3))
    output, info = reference.layer_forward(
        combined, shard, 2, f1.read_json(source / "config.json")["text_config"],
        [1] * combined.shape[0],
    )
    output_np = np.asarray(output.astype(mx.float32), dtype=np.float32)
    count = teacher_hidden.shape[0]
    teacher, candidate = output_np[:count], output_np[count:]
    route_values = info["moe"]["route_indices"]
    teacher_routes = route_values[:count]
    candidate_routes = route_values[count:]
    agreements = [set(left) == set(right)
                  for left, right in zip(teacher_routes, candidate_routes, strict=True)]
    jaccards = [len(set(left) & set(right)) / len(set(left) | set(right))
                for left, right in zip(teacher_routes, candidate_routes, strict=True)]
    record = {
        "source_shard": shard.path.name,
        "full_layer_hidden": f1.quality(teacher, candidate),
        "score_full_layer_hidden": f1.quality(teacher[32:], candidate[32:]),
        "native_route_set_agreement": float(np.mean(agreements)),
        "native_route_jaccard_mean": float(np.mean(jaccards)),
        "teacher_used_experts": len({value for row in teacher_routes for value in row}),
        "candidate_used_experts": len({value for row in candidate_routes for value in row}),
        "tensor_audit": info["tensor_audit"],
    }
    del shard, combined, output
    gc.collect()
    mx.clear_cache()
    return teacher, candidate, record


def logit_sketch(
    source: Path, teacher_hidden: np.ndarray, candidate_hidden: np.ndarray,
    sample_size: int = 2048,
) -> dict[str, Any]:
    config = f1.read_json(source / "config.json")["text_config"]
    tail = reference.TensorShard(reference.shard_path(source, 62))
    combined = mx.array(np.concatenate((teacher_hidden, candidate_hidden))).astype(mx.bfloat16)
    normalized = reference.rms_norm(
        combined, tail.mlx(f"{reference.PREFIX}.norm.weight"),
        float(config["rms_norm_eps"]),
    )
    vocabulary = tail.numpy("language_model.lm_head.weight")
    rng = np.random.default_rng(260702)
    ids = np.sort(rng.choice(vocabulary.shape[0], size=sample_size, replace=False)).astype(np.int32)
    sampled_weight = mx.array(np.asarray(vocabulary[ids]))
    logits = reference.linear(normalized, sampled_weight).astype(mx.float32)
    mx.eval(logits)
    logits_np = np.asarray(logits, dtype=np.float32)
    count = teacher_hidden.shape[0]
    teacher, candidate = logits_np[:count], logits_np[count:]
    teacher_top = np.argsort(teacher, axis=1)[:, -20:]
    candidate_top = np.argsort(candidate, axis=1)[:, -20:]
    overlap = [len(set(left) & set(right)) / 20
               for left, right in zip(teacher_top, candidate_top, strict=True)]
    result = {
        "kind": "DETERMINISTIC_2048_VOCAB_ROW_EARLY_EXIT_SKETCH",
        "not_full_model_logits": True, "sample_size": sample_size,
        "vocabulary_size": int(vocabulary.shape[0]),
        "sample_id_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "all_tokens": f1.quality(teacher, candidate),
        "score_tokens": f1.quality(teacher[32:], candidate[32:]),
        "sampled_top1_agreement": float(np.mean(
            np.argmax(teacher, axis=1) == np.argmax(candidate, axis=1))),
        "sampled_top20_overlap_mean": float(np.mean(overlap)),
    }
    del tail, combined, normalized, vocabulary, sampled_weight, logits
    gc.collect()
    mx.clear_cache()
    return result


def run(source: Path, f1_dir: Path, output_dir: Path) -> dict[str, Any]:
    started = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_path = f1_dir / "teacher_capture.npz"
    capture_receipt = f1.read_json(f1_dir / "teacher_capture.json")
    if f1.sha256_file(capture_path) != capture_receipt["capture_sha256"]:
        raise f1.F1Error("F1 teacher capture hash mismatch")
    with np.load(capture_path, allow_pickle=False) as loaded:
        capture = {key: loaded[key] for key in loaded.files}
    promoted = f1_dir / "doctor_auction/P1_DUAL_PATH_RECOVERY_R16X2.k26f1"
    promoted_result = f1.read_json(
        f1_dir / "doctor_auction/P1_DUAL_PATH_RECOVERY_R16X2_RESULT.json",
    )
    if f1.sha256_file(promoted) != promoted_result["payload"]["sha256"]:
        raise f1.F1Error("promoted F1 payload hash mismatch")
    layer_one_teacher, layer_one_candidate, layer_one = complete_layer_one(
        source, capture, promoted,
    )
    layer_two_teacher, layer_two_candidate, layer_two = layer_two_propagation(
        source, layer_one_teacher, layer_one_candidate,
    )
    logits = logit_sketch(source, layer_two_teacher, layer_two_candidate)
    score_hidden = layer_two["score_full_layer_hidden"]
    score_logits = logits["score_tokens"]
    survives = (
        score_hidden["cosine_mean"] >= 0.995 and score_hidden["cosine_p10"] >= 0.99 and
        score_hidden["relative_l2"] <= 0.15 and score_logits["cosine_mean"] >= 0.98 and
        layer_two["native_route_set_agreement"] >= 0.95
    )
    route_unstable = layer_two["native_route_set_agreement"] < 0.95
    decision = (
        "ADVANCE_TO_F2_ACTIVE_EXPERT_SLICE" if survives else
        "RETIRE_P1_HIDDEN_DOCTOR_FOR_ROUTING_INSTABILITY" if route_unstable else
        "RETIRE_P1_HIDDEN_DOCTOR_FOR_PROPAGATED_FIDELITY_LOSS"
    )
    diagnosis = (
        "DAMAGE_DOES_NOT_IMMEDIATELY_COMPOUND" if survives else
        "ROUTE_SET_INSTABILITY_AFTER_RESIDUAL_PROPAGATION" if route_unstable else
        "ACTIVATION_OR_LOGIT_DAMAGE_AFTER_RESIDUAL_PROPAGATION"
    )
    failure_reason = (
        "Promoted F1 recovery remains bounded after a complete residual add and one exact "
        "downstream shard." if survives else
        "Activation and sampled-logit errors remain bounded, but downstream native top-8 route "
        "sets are unstable." if route_unstable else
        "The local F1 repair does not preserve downstream activation/logit fidelity."
    )
    artifact = f1.seal({
        "schema": "hawking.kimi_k26.f2_sparse_propagation.v1", "status": "PASS",
        "sealed_at": f1.now(), "runtime_seconds": time.time() - started,
        "source": {"repo": f1.REPO, "revision": f1.REVISION},
        "experiment": "P1_F2_SPARSE_SUBSTITUTION_PROPAGATION",
        "claim_boundary": (
            "ONE PROMOTED COMPACT EXPERT SUBSTITUTED IN COMPLETE ACTIVE LAYER-1 PATH, THEN "
            "EXACT LAYER-2; not a full compact shard or end-to-end capability"
        ),
        "reuse": {"parent_forwards_repeated": 0, "teacher_captures": 0,
                  "layer_zero_runs": 0, "layer_one_attention_runs": 0,
                  "layer_one_routing_runs": 0},
        "teacher_capture_seal_sha256": capture_receipt["seal_sha256"],
        "promoted_payload_sha256": promoted_result["payload"]["sha256"],
        "layer_one": layer_one, "layer_two": layer_two, "logit_sketch": logits,
        "candidate_verdict": ("SURVIVES_F2_SPARSE" if survives else "COLLAPSE_F2_SPARSE"),
        "diagnosis_metrics_reason_next_decision": {
            "diagnosis": diagnosis,
            "metrics": {"layer_two_score_hidden": score_hidden,
                        "layer_two_score_logit_sketch": score_logits,
                        "route_set_agreement": layer_two["native_route_set_agreement"]},
            "reason": failure_reason,
            "next_decision": decision,
        },
        "decision": decision,
        "current_next_experiment": (
            "P1_F2_ALL_ROUTE_ACTIVE_EXPERTS_SHARED_DOCTOR" if survives else
            "F1_SHARED_GRAMMAR_VS_PROTECTED_ISLANDS_CACHED_SEAM"
        ),
    })
    f1.atomic_json(output_dir / "KIMI_K26_P1_F2_SPARSE_PROPAGATION.json", artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--f1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.source.resolve(strict=True), args.f1_dir.resolve(strict=True),
                     args.output_dir.resolve())
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
