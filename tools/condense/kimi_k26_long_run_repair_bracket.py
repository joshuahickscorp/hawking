#!/usr/bin/env python3.12
"""Causally targeted <=0.98-BPW Kimi repair bracket on sealed LR01 states."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import struct
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
import kimi_k26_long_run_causal as causal  # noqa: E402
import kimi_k26_long_run_manager as manager  # noqa: E402
import kimi_k26_reference as reference  # noqa: E402


EXPERIMENT_ID = "LR02_CAUSALLY_TARGETED_PHYSICAL_BRACKET"
FIT_SEGMENTS = {"factual", "science", "coding", "mathematics"}
EVAL_SEGMENTS = {"reasoning", "instruction", "tool_thinking_protocol", "rare_token"}
LOGICAL_WEIGHTS = 44_040_192
COMPLETE_CEILING_BPW = 0.98
COMPLETE_CEILING_BYTES = 5_394_923
PRELUDE = b"K26REPAIR1\x00"


def route_from_logits(
    logits: np.ndarray, correction: np.ndarray, config: dict[str, Any],
) -> dict[str, np.ndarray]:
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
    weights = np.take_along_axis(scores, indices, axis=-1)
    if config["norm_topk_prob"]:
        weights /= weights.sum(axis=-1, keepdims=True) + 1e-20
    weights *= float(config["routed_scaling_factor"])
    margin = (
        np.take_along_axis(filtered, indices[:, -1:], axis=-1)[:, 0] -
        np.take_along_axis(filtered, ninth[:, None], axis=-1)[:, 0]
    )
    return {"indices": indices, "weights": weights.astype(np.float32),
            "margin_8v9": margin.astype(np.float32), "scores": scores.astype(np.float32),
            "logits": np.asarray(logits, dtype=np.float32)}


def fit_reduced_rank(
    x: np.ndarray, target: np.ndarray, rank: int, ridge_fraction: float = 1e-4,
) -> dict[str, np.ndarray | float | int]:
    x = np.asarray(x, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    bias = np.mean(target, axis=0, keepdims=True)
    centered = target - bias
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    actual_rank = min(rank, int(np.sum(singular > singular[0] * 1e-6)) if singular.size else 0)
    actual_rank = max(1, actual_rank)
    basis = right[:actual_rank].astype(np.float32)
    coefficients = centered @ basis.T
    gram = x @ x.T
    ridge = float(np.trace(gram) / max(1, x.shape[0]) * ridge_fraction)
    dual = np.linalg.solve(
        gram + np.eye(gram.shape[0], dtype=np.float32) * ridge,
        coefficients,
    )
    projection = (x.T @ dual).astype(np.float32)
    energy = float(np.sum(singular[:actual_rank] ** 2) / (np.sum(singular ** 2) + 1e-30))
    return {"projection": projection, "basis": basis, "bias": bias[0].astype(np.float32),
            "rank": actual_rank, "ridge": ridge, "captured_target_energy": energy}


def quantize_columns(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.max(np.abs(value), axis=0) / 127.0
    scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
    quantized = np.clip(np.rint(value / scale[None, :]), -127, 127).astype(np.int8)
    return quantized, scale


def quantize_rows(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.max(np.abs(value), axis=1) / 127.0
    scale = np.where(scale > 0, scale, 1.0).astype(np.float32)
    quantized = np.clip(np.rint(value / scale[:, None]), -127, 127).astype(np.int8)
    return quantized, scale


def physicalize(model: dict[str, Any], prefix: str) -> tuple[dict[str, Any], list[tuple[str, np.ndarray]]]:
    projection = np.asarray(model["projection"], dtype=np.float32)
    basis = np.asarray(model["basis"], dtype=np.float32)
    bias = np.asarray(model["bias"], dtype=np.float32)
    projection_q, projection_scale = quantize_columns(projection)
    basis_q, basis_scale = quantize_rows(basis)
    arrays = [
        (f"{prefix}.projection_q", projection_q),
        (f"{prefix}.projection_scale", projection_scale),
        (f"{prefix}.basis_q", basis_q),
        (f"{prefix}.basis_scale", basis_scale),
        (f"{prefix}.bias_bf16", np.asarray(bias, dtype=ml_dtypes.bfloat16)),
    ]
    decoded = {
        "projection": projection_q.astype(np.float32) * projection_scale[None, :],
        "basis": basis_q.astype(np.float32) * basis_scale[:, None],
        "bias": np.asarray(np.asarray(bias, dtype=ml_dtypes.bfloat16), dtype=np.float32),
        "rank": int(model["rank"]), "ridge": float(model["ridge"]),
        "captured_target_energy": float(model["captured_target_energy"]),
    }
    return decoded, arrays


def predict(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    return ((np.asarray(x, dtype=np.float32) @ model["projection"]) @ model["basis"] +
            model["bias"][None, :]).astype(np.float32)


def write_payload(
    path: Path, parent_hash: str, architecture: str,
    arrays: list[tuple[str, np.ndarray]], metadata: dict[str, Any],
) -> dict[str, Any]:
    components = []
    offset = 0
    blocks = []
    for name, value in arrays:
        contiguous = np.ascontiguousarray(value)
        block = contiguous.tobytes()
        components.append({"name": name, "dtype": str(contiguous.dtype),
                           "shape": list(contiguous.shape), "offset": offset,
                           "bytes": len(block)})
        blocks.append(block)
        offset += len(block)
    header = json.dumps({"schema": "hawking.kimi_k26.physical_repair.v1",
                         "parent_sha256": parent_hash, "architecture": architecture,
                         "metadata": metadata, "components": components},
                        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(PRELUDE)
        handle.write(struct.pack("<I", len(header)))
        handle.write(header)
        for block in blocks:
            handle.write(block)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": f1.sha256_file(path), "component_count": len(components),
            "header_bytes": len(PRELUDE) + 4 + len(header)}


def custom_routed_moe(
    x: np.ndarray, shard: reference.TensorShard, layer: int,
    routes: dict[str, np.ndarray],
) -> np.ndarray:
    x_mx = mx.array(x).astype(mx.bfloat16)
    indices = routes["indices"]
    weights = routes["weights"]
    combined = mx.zeros_like(x_mx)
    sequence = x.shape[0]
    for expert in sorted(set(int(value) for value in indices.flat)):
        tokens, slots = np.where(indices == expert)
        output = causal.native_expert(x_mx, shard, layer, expert, tokens.astype(np.int32))
        route_weight = mx.array(weights[tokens, slots], dtype=output.dtype)[:, None]
        scatter = mx.array(np.eye(sequence, dtype=np.float32)[tokens].T, dtype=output.dtype)
        combined = combined + scatter @ (output * route_weight)
        mx.eval(combined)
    shared = reference.dense_mlp(
        x_mx, shard, f"{reference.PREFIX}.layers.{layer}.mlp.shared_experts",
    )
    feed = combined + shared
    mx.eval(feed)
    result = np.asarray(feed.astype(mx.float32), dtype=np.float32)
    del x_mx, combined, shared, feed
    gc.collect()
    mx.clear_cache()
    return result


def paired_interval(improvement: np.ndarray, seed: int) -> dict[str, float | int]:
    values = np.asarray(improvement, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(4000, values.size))
    means = values[draws].mean(axis=1)
    return {"mean": float(np.mean(values)), "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)), "n": int(values.size)}


def evaluate(
    name: str, teacher_hidden: np.ndarray, natural_hidden: np.ndarray,
    candidate_hidden: np.ndarray, teacher_route: dict[str, np.ndarray],
    candidate_route: dict[str, np.ndarray], payload: dict[str, Any], seed: int,
    allocation: dict[str, Any],
) -> dict[str, Any]:
    baseline_rows = causal.row_metrics(teacher_hidden, natural_hidden)
    candidate_rows = causal.row_metrics(teacher_hidden, candidate_hidden)
    teacher_indices = teacher_route["indices"]
    candidate_indices = candidate_route["indices"]
    route_exact = np.asarray([
        set(left) == set(right) for left, right in zip(
            teacher_indices, candidate_indices, strict=True,
        )
    ])
    total_bytes = payload["bytes"] + 5_001_815
    return {
        "candidate": name, "physical_payload": payload,
        "physical_allocation": allocation, "parent_bytes": 5_001_815,
        "complete_physical_bytes": total_bytes,
        "complete_bpw": total_bytes * 8 / LOGICAL_WEIGHTS,
        "within_0_98_bpw": total_bytes <= COMPLETE_CEILING_BYTES,
        "heldout_tokens": int(teacher_hidden.shape[0]),
        "hidden": f1.quality(teacher_hidden, candidate_hidden),
        "row_cosine_mean": float(np.mean(candidate_rows["cosine"])),
        "row_relative_l2_mean": float(np.mean(candidate_rows["relative_l2"])),
        "relative_l2_improvement": paired_interval(
            baseline_rows["relative_l2"] - candidate_rows["relative_l2"], seed,
        ),
        "route_set_agreement": float(np.mean(route_exact)),
        "route_set_change": float(1 - np.mean(route_exact)),
        "route_matches": int(np.sum(route_exact)),
    }


def choose_threshold(
    base_margin: np.ndarray, fit_teacher_route: dict[str, np.ndarray],
    candidate_x: np.ndarray, route_fn: Any,
) -> tuple[float, dict[str, Any]]:
    best = None
    for quantile in (0.25, 0.50, 0.75, 1.0):
        threshold = float(np.quantile(base_margin, quantile))
        mask = base_margin <= threshold
        routes = route_fn(np.where(mask[:, None], candidate_x, 0), mask)
        exact = np.mean([
            set(left) == set(right) for left, right in zip(
                fit_teacher_route["indices"], routes["indices"], strict=True,
            )
        ])
        score = float(exact - 0.001 * np.mean(mask))
        record = {"quantile": quantile, "threshold": threshold,
                  "active_fraction": float(np.mean(mask)),
                  "training_route_agreement": float(exact), "selection_score": score}
        if best is None or record["selection_score"] > best[1]["selection_score"]:
            best = (threshold, record)
    assert best is not None
    return best


def run(
    repo: Path, source: Path, output_dir: Path, seed: int, retry: int,
) -> dict[str, Any]:
    started_at = f1.now()
    started_wall = time.time()
    before = manager.resource_snapshot()
    audit = manager.audit(repo, notify=False)
    if audit["status"] != "PASS":
        raise f1.F1Error(f"pre-run guard audit failed: {audit['failures']}")
    status = f1.read_json(repo / manager.STATUS_JSON)
    manager.write_status(repo, {**status, "status": "RUNNING_EXPERIMENT",
                                "active_experiment": EXPERIMENT_ID,
                                "next_experiment": "LR02_IN_PROGRESS", "resources": before})
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_START",
        "experiment_id": EXPERIMENT_ID, "started_at": started_at, "status": "RUNNING",
        "hypothesis": "Upstream state repair buys more held-out F2 survival per bit than route-only repair.",
        "config": {"seed": seed, "fit_segments": sorted(FIT_SEGMENTS),
                   "heldout_segments": sorted(EVAL_SEGMENTS), "complete_bpw_ceiling": 0.98},
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_path = manager.RUNTIME / "long_run/LR01/LR01_CONTROL_CAPTURE.npz"
    control_path = repo / "KIMI_K26_LONG_RUN_CONTROL_CAUSAL.json"
    control = f1.read_json(control_path)
    if f1.sha256_file(capture_path) != control["capture"]["sha256"]:
        raise f1.F1Error("LR01 raw checkpoint hash mismatch")
    if control["seal_sha256"] != f1.seal({k: v for k, v in control.items()
                                           if k != "seal_sha256"})["seal_sha256"]:
        raise f1.F1Error("LR01 evidence seal mismatch")
    with np.load(capture_path, allow_pickle=False) as capture:
        arrays = {key: np.asarray(capture[key]) for key in capture.files}
    _, batch = causal.prepare_requests(source)
    records = batch["token_records"]
    fit_mask = np.asarray([row["segment"] in FIT_SEGMENTS for row in records])
    eval_mask = np.asarray([row["segment"] in EVAL_SEGMENTS for row in records])
    if np.any(fit_mask & eval_mask) or not np.all(fit_mask | eval_mask):
        raise f1.F1Error("invalid LR02 split")
    config = f1.read_json(source / "config.json")["text_config"]
    shard = reference.TensorShard(reference.shard_path(source, 3))
    correction = np.asarray(shard.mlx(
        f"{reference.PREFIX}.layers.2.mlp.gate.e_score_correction_bias"
    ).astype(mx.float32), dtype=np.float32)
    teacher_x = arrays["teacher_x_l2"]
    student_x = arrays["student_x_l2"]
    teacher_hidden = arrays["variant_l2_TEACHER"]
    natural_hidden = arrays["variant_l2_NATURAL_STUDENT"]
    student_post = arrays["student_post_l2"]
    teacher_routed = arrays["teacher_routed_l2"]
    student_routed = arrays["student_routed_l2"]
    teacher_route = causal.route_diagnostics(teacher_x, shard, 2, config)
    student_route = causal.route_diagnostics(student_x, shard, 2, config)
    fit_teacher_route = {key: value[fit_mask] for key, value in teacher_route.items()}
    natural_feed_mx, _ = reference.routed_moe(
        mx.array(student_x).astype(mx.bfloat16), shard, 2, config,
    )
    natural_feed = np.asarray(natural_feed_mx.astype(mx.float32), dtype=np.float32)
    natural_reconstructed = causal.hidden_from_feed(student_post, natural_feed)
    natural_reconstruction = f1.quality(natural_hidden, natural_reconstructed)
    if natural_reconstruction["relative_l2"] != 0.0:
        raise f1.F1Error(f"resident natural reconstruction mismatch: {natural_reconstruction}")
    parent_result = f1.read_json(
        manager.RUNTIME / "f1_representation_bracket/doctor_auction/"
        "P1_DUAL_PATH_RECOVERY_R16X2_RESULT.json"
    )
    parent_hash = parent_result["payload"]["sha256"]
    results = []

    state_model = fit_reduced_rank(
        student_x[fit_mask], teacher_x[fit_mask] - student_x[fit_mask], rank=24,
    )
    state_physical, state_blocks = physicalize(state_model, "state")
    repaired_fit_x = student_x[fit_mask] + predict(state_physical, student_x[fit_mask])
    risk_model = fit_reduced_rank(
        student_x[fit_mask], teacher_x[fit_mask] - student_x[fit_mask], rank=12,
    )
    risk_physical, risk_blocks = physicalize(risk_model, "risk_state")
    risk_fit_x = student_x[fit_mask] + predict(risk_physical, student_x[fit_mask])
    fit_margin = student_route["margin_8v9"][fit_mask]

    def risk_routes(_: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        mixed = np.where(mask[:, None], risk_fit_x, student_x[fit_mask])
        return causal.route_diagnostics(mixed, shard, 2, config)

    risk_threshold, risk_selection = choose_threshold(
        fit_margin, fit_teacher_route, risk_fit_x, risk_routes,
    )
    risk_full_mask = student_route["margin_8v9"] <= risk_threshold
    risk_full_x = student_x.copy()
    risk_full_x[risk_full_mask] += predict(risk_physical, student_x[risk_full_mask])
    risk_feed_mx, _ = reference.routed_moe(
        mx.array(risk_full_x).astype(mx.bfloat16), shard, 2, config,
    )
    risk_feed = np.asarray(risk_feed_mx.astype(mx.float32), dtype=np.float32)
    risk_hidden = causal.hidden_from_feed(student_post, risk_feed)[eval_mask]
    risk_route_full = causal.route_diagnostics(risk_full_x, shard, 2, config)
    risk_route = {key: value[eval_mask] for key, value in risk_route_full.items()}
    risk_payload = write_payload(
        output_dir / "LR02_FIRST_DIVERGENCE_PROTECTION_R12.k26repair", parent_hash,
        "FIRST_DIVERGENCE_LOW_MARGIN_STATE_PROTECTION_R12", risk_blocks,
        {"activation_margin_threshold": risk_threshold, "training_selection": risk_selection},
    )
    results.append(evaluate(
        "FIRST_DIVERGENCE_PROTECTION_R12", teacher_hidden[eval_mask],
        natural_hidden[eval_mask], risk_hidden,
        {key: value[eval_mask] for key, value in teacher_route.items()}, risk_route,
        risk_payload, seed + 1,
        {"parent_base_bytes": parent_result["payload"]["base_component_bytes"],
         "parent_doctor_bytes": parent_result["payload"]["doctor_component_bytes"],
         "first_divergence_state_repair_bytes": risk_payload["bytes"]},
    ))

    repaired_full_x = student_x + predict(state_physical, student_x)
    state_feed_mx, _ = reference.routed_moe(
        mx.array(repaired_full_x).astype(mx.bfloat16), shard, 2, config,
    )
    state_feed = np.asarray(state_feed_mx.astype(mx.float32), dtype=np.float32)
    state_hidden = causal.hidden_from_feed(student_post, state_feed)[eval_mask]
    state_route_full = causal.route_diagnostics(repaired_full_x, shard, 2, config)
    state_route = {key: value[eval_mask] for key, value in state_route_full.items()}
    state_payload = write_payload(
        output_dir / "LR02_PRE_ROUTER_STATE_R24.k26repair", parent_hash,
        "PRE_ROUTER_LOW_RANK_HIDDEN_STATE_REPAIR_R24", state_blocks,
        {"activation": "ALL_TOKENS", "fit_rank": state_physical["rank"]},
    )
    results.append(evaluate(
        "PRE_ROUTER_STATE_R24", teacher_hidden[eval_mask], natural_hidden[eval_mask],
        state_hidden, {key: value[eval_mask] for key, value in teacher_route.items()},
        state_route, state_payload, seed + 2,
        {"parent_base_bytes": parent_result["payload"]["base_component_bytes"],
         "parent_doctor_bytes": parent_result["payload"]["doctor_component_bytes"],
         "pre_router_state_repair_bytes": state_payload["bytes"]},
    ))

    router_model = fit_reduced_rank(
        student_x[fit_mask],
        teacher_route["logits"][fit_mask] - student_route["logits"][fit_mask], rank=24,
    )
    router_physical, router_blocks = physicalize(router_model, "router")
    repaired_fit_logits = (
        student_route["logits"][fit_mask] + predict(router_physical, student_x[fit_mask])
    )

    def router_routes(_: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        logits = np.where(mask[:, None], repaired_fit_logits,
                          student_route["logits"][fit_mask])
        return route_from_logits(logits, correction, config)

    router_threshold, router_selection = choose_threshold(
        fit_margin, fit_teacher_route, repaired_fit_logits, router_routes,
    )
    full_router_mask = student_route["margin_8v9"] <= router_threshold
    repaired_full_logits = student_route["logits"].copy()
    repaired_full_logits[full_router_mask] += predict(
        router_physical, student_x[full_router_mask],
    )
    router_full_route = route_from_logits(repaired_full_logits, correction, config)
    router_feed = custom_routed_moe(student_x, shard, 2, router_full_route)
    router_hidden = causal.hidden_from_feed(student_post, router_feed)[eval_mask]
    router_eval_route = {key: value[eval_mask] for key, value in router_full_route.items()}
    router_payload = write_payload(
        output_dir / "LR02_LOW_MARGIN_ROUTER_R24.k26repair", parent_hash,
        "LOW_MARGIN_CONDITIONAL_ROUTER_LOGIT_REPAIR_R24", router_blocks,
        {"activation_margin_threshold": router_threshold,
         "training_selection": router_selection},
    )
    results.append(evaluate(
        "LOW_MARGIN_ROUTER_R24", teacher_hidden[eval_mask], natural_hidden[eval_mask],
        router_hidden, {key: value[eval_mask] for key, value in teacher_route.items()},
        router_eval_route, router_payload, seed + 3,
        {"parent_base_bytes": parent_result["payload"]["base_component_bytes"],
         "parent_doctor_bytes": parent_result["payload"]["doctor_component_bytes"],
         "conditional_router_repair_bytes": router_payload["bytes"]},
    ))

    moe_model = fit_reduced_rank(
        student_x[fit_mask], teacher_routed[fit_mask] - student_routed[fit_mask], rank=24,
    )
    moe_physical, moe_blocks = physicalize(moe_model, "moe_output")
    moe_delta = predict(moe_physical, student_x)
    moe_hidden_mx = (mx.array(natural_hidden).astype(mx.bfloat16) +
                     mx.array(moe_delta).astype(mx.bfloat16)).astype(mx.bfloat16)
    mx.eval(moe_hidden_mx)
    moe_hidden = np.asarray(moe_hidden_mx.astype(mx.float32), dtype=np.float32)[eval_mask]
    moe_payload = write_payload(
        output_dir / "LR02_WEIGHTED_MOE_OUTPUT_R24.k26repair", parent_hash,
        "WEIGHTED_MOE_OUTPUT_LOW_RANK_REPAIR_R24", moe_blocks,
        {"activation": "ALL_TOKENS", "fit_rank": moe_physical["rank"]},
    )
    results.append(evaluate(
        "WEIGHTED_MOE_OUTPUT_R24", teacher_hidden[eval_mask], natural_hidden[eval_mask],
        moe_hidden, {key: value[eval_mask] for key, value in teacher_route.items()},
        {key: value[eval_mask] for key, value in student_route.items()},
        moe_payload, seed + 4,
        {"parent_base_bytes": parent_result["payload"]["base_component_bytes"],
         "parent_doctor_bytes": parent_result["payload"]["doctor_component_bytes"],
         "weighted_moe_output_repair_bytes": moe_payload["bytes"]},
    ))

    hybrid_state_model = fit_reduced_rank(
        student_x[fit_mask], teacher_x[fit_mask] - student_x[fit_mask], rank=12,
    )
    hybrid_state, hybrid_state_blocks = physicalize(hybrid_state_model, "hybrid_state")
    hybrid_full_x = student_x + predict(hybrid_state, student_x)
    hybrid_full_feed_mx, _ = reference.routed_moe(
        mx.array(hybrid_full_x).astype(mx.bfloat16), shard, 2, config,
    )
    hybrid_full_feed = np.asarray(hybrid_full_feed_mx.astype(mx.float32), dtype=np.float32)
    hybrid_pre_hidden = causal.hidden_from_feed(student_post, hybrid_full_feed)
    hybrid_fit_hidden = hybrid_pre_hidden[fit_mask]
    hybrid_output_model = fit_reduced_rank(
        student_x[fit_mask], teacher_hidden[fit_mask] - hybrid_fit_hidden, rank=12,
    )
    hybrid_output, hybrid_output_blocks = physicalize(hybrid_output_model, "hybrid_output")
    hybrid_delta = predict(hybrid_output, student_x)
    hybrid_hidden_mx = (mx.array(hybrid_pre_hidden).astype(mx.bfloat16) +
                        mx.array(hybrid_delta).astype(mx.bfloat16)).astype(mx.bfloat16)
    mx.eval(hybrid_hidden_mx)
    hybrid_hidden = np.asarray(
        hybrid_hidden_mx.astype(mx.float32), dtype=np.float32,
    )[eval_mask]
    hybrid_route_full = causal.route_diagnostics(hybrid_full_x, shard, 2, config)
    hybrid_route = {key: value[eval_mask] for key, value in hybrid_route_full.items()}
    hybrid_payload = write_payload(
        output_dir / "LR02_HYBRID_R12X2.k26repair", parent_hash,
        "PRE_ROUTER_PLUS_POST_MOE_HYBRID_R12X2",
        hybrid_state_blocks + hybrid_output_blocks,
        {"pre_router_rank": hybrid_state["rank"], "post_moe_rank": hybrid_output["rank"]},
    )
    results.append(evaluate(
        "HYBRID_R12X2", teacher_hidden[eval_mask], natural_hidden[eval_mask],
        hybrid_hidden, {key: value[eval_mask] for key, value in teacher_route.items()},
        hybrid_route, hybrid_payload, seed + 5,
        {"parent_base_bytes": parent_result["payload"]["base_component_bytes"],
         "parent_doctor_bytes": parent_result["payload"]["doctor_component_bytes"],
         "pre_router_repair_bytes": sum(value.nbytes for _, value in hybrid_state_blocks),
         "post_moe_repair_bytes": sum(value.nbytes for _, value in hybrid_output_blocks),
         "repair_header_bytes": hybrid_payload["header_bytes"]},
    ))
    del shard
    gc.collect()
    mx.clear_cache()

    teacher_eval_route = {key: value[eval_mask] for key, value in teacher_route.items()}
    student_eval_route = {key: value[eval_mask] for key, value in student_route.items()}
    baseline_exact = np.mean([
        set(left) == set(right) for left, right in zip(
            teacher_eval_route["indices"], student_eval_route["indices"], strict=True,
        )
    ])
    baseline_rows = causal.row_metrics(teacher_hidden[eval_mask], natural_hidden[eval_mask])
    baseline = {"candidate": "P1_DUAL_PATH_RECOVERY_R16X2_NO_ADDITIONAL_REPAIR",
                "complete_physical_bytes": 5_001_815,
                "complete_bpw": 5_001_815 * 8 / LOGICAL_WEIGHTS,
                "hidden": f1.quality(teacher_hidden[eval_mask], natural_hidden[eval_mask]),
                "row_relative_l2_mean": float(np.mean(baseline_rows["relative_l2"])),
                "route_set_agreement": float(baseline_exact),
                "route_set_change": float(1 - baseline_exact),
                "route_matches": int(round(baseline_exact * int(np.sum(eval_mask))))}
    eligible = [row for row in results if row["within_0_98_bpw"] and
                row["relative_l2_improvement"]["ci95_low"] > 0 and
                row["route_matches"] >= baseline["route_matches"] - 1]
    best = max(eligible, key=lambda row: row["relative_l2_improvement"]["mean"],
               default=None)
    for row in results:
        if not row["within_0_98_bpw"]:
            row["decision"] = "INVALID_OVER_BUDGET"
        elif best is not None and row["candidate"] == best["candidate"]:
            row["decision"] = "PROMOTE_TO_REPLICATION"
        elif row["relative_l2_improvement"]["ci95_high"] <= 0:
            row["decision"] = "RETIRE_HELDOUT_HARM"
        else:
            row["decision"] = "RETIRE_DOMINATED_OR_UNCERTAIN"
    after = manager.resource_snapshot()
    ended_at = f1.now()
    duration = time.time() - started_wall
    promoted = None if best is None else {
        "candidate": best["candidate"], "complete_bpw": best["complete_bpw"],
        "complete_physical_bytes": best["complete_physical_bytes"],
        "repair_payload": best["physical_payload"], "heldout_hidden": best["hidden"],
        "route_set_agreement": best["route_set_agreement"],
        "relative_l2_improvement": best["relative_l2_improvement"],
    }
    artifact = f1.seal({
        "schema": "hawking.kimi_k26.long_run_repair_bracket.v1", "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "Upstream state repair buys more held-out F2 survival per bit than route-only repair.",
        "config": {"seed": seed, "fit_segments": sorted(FIT_SEGMENTS),
                   "heldout_segments": sorted(EVAL_SEGMENTS), "fit_tokens": int(np.sum(fit_mask)),
                   "heldout_tokens": int(np.sum(eval_mask)), "selection_uses_heldout": False,
                   "complete_bpw_ceiling": COMPLETE_CEILING_BPW,
                   "complete_ceiling_bytes": COMPLETE_CEILING_BYTES},
        "parent": {"candidate": "P1_DUAL_PATH_RECOVERY_R16X2",
                   "payload_sha256": parent_hash, "bytes": 5_001_815,
                   "complete_bpw": parent_result["physical_budget"]["actual_complete_bpw"]},
        "evidence_parent": {"lr01_seal_sha256": control["seal_sha256"],
                            "capture_sha256": control["capture"]["sha256"],
                            "guard_audit_seal_sha256": audit["seal_sha256"]},
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resource_observations": {"before": before, "after": after},
        "baseline": baseline, "treatment_frontier": results,
        "causal_upper_bounds_from_lr01": control["intervention_matrix"],
        "promotion": promoted,
        "decision": "PROMOTE_BEST_TO_NEW_SPLIT_REPLICATION" if promoted else
                    "NO_PHYSICAL_ROW_PROMOTED_TESTED_REGION_RETIRED",
        "scientific_interpretation": (
            "The physical bracket adjudicates whether the LR01 causal upper bounds are learnable "
            "from calibration-only low-rank state, route-logit, or MoE-output corrections."
        ),
        "next_run_rationale": (
            "Replicate the promoted row on new contexts and adversarial low-margin tokens."
            if promoted else
            "Falsify the closure with a new-split boundary test before declaring the <=0.98 region closed."
        ),
        "faults": (["ATTEMPT_0_QUANTIZED_KERNEL_BATCH_SHAPE_MISMATCH"] if retry else []),
        "retries": retry,
    })
    artifact_path = repo / "KIMI_K26_LONG_RUN_REPAIR_BRACKET.json"
    f1.atomic_json(artifact_path, artifact)
    f1.atomic_json(manager.RUNTIME / artifact_path.name, artifact)
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_COMPLETE",
        "experiment_id": EXPERIMENT_ID, "hypothesis": artifact["hypothesis"],
        "config": artifact["config"], "parent_hash": parent_hash,
        "candidate_hash": promoted["repair_payload"]["sha256"] if promoted else None,
        "physical_bytes": promoted["complete_physical_bytes"] if promoted else None,
        "complete_bpw": promoted["complete_bpw"] if promoted else None,
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resources": artifact["resource_observations"],
        "sample_counts": {"fit_tokens": int(np.sum(fit_mask)),
                          "heldout_tokens": int(np.sum(eval_mask)), "rows": len(results)},
        "metrics": {"baseline": baseline, "treatment_frontier": results,
                    "promotion": promoted},
        "confidence_intervals": {row["candidate"]: row["relative_l2_improvement"]
                                 for row in results},
        "faults": artifact["faults"], "retries": retry,
        "causal_interpretation": artifact["scientific_interpretation"],
        "decision": artifact["decision"], "next_run_rationale": artifact["next_run_rationale"],
        "evidence_seal_sha256": artifact["seal_sha256"],
    })
    prior = f1.read_json(repo / manager.STATUS_JSON)
    current_best = (promoted["candidate"] if promoted else prior["current_best_candidate"])
    current_bpw = (promoted["complete_bpw"] if promoted else prior["current_best_bpw"])
    status = manager.write_status(repo, {
        **prior, "status": "MANAGING", "active_experiment": None,
        "experiments_completed": int(prior.get("experiments_completed", 0)) + 1,
        "current_best_candidate": current_best, "current_best_bpw": current_bpw,
        "next_experiment": "LR03_NEW_SPLIT_ADVERSARIAL_REPLICATION",
        "latest_result": {"experiment_id": EXPERIMENT_ID, "promotion": promoted,
                          "baseline": baseline, "evidence_seal_sha256": artifact["seal_sha256"]},
        "resources": after,
    })
    receipt = manager.telegram(
        repo, "long-run:LR02-complete",
        ("[Kimi K2.6 long run] LR02 physical repair bracket complete\n"
         f"fit/heldout tokens: {int(np.sum(fit_mask))} / {int(np.sum(eval_mask))}\n"
         f"baseline heldout route change: {(1-baseline_exact)*100:.2f}%\n"
         f"promotion: {promoted['candidate'] if promoted else 'NONE'}\n"
         f"best complete BPW: {promoted['complete_bpw'] if promoted else baseline['complete_bpw']:.6f}\n"
         f"free disk: {after['free_disk_bytes']/1024**3:.2f} GiB\n"
         "next: new-split and adversarial low-margin replication"),
    )
    manager.write_status(repo, {
        **status, "latest_result": {**status["latest_result"],
                                    "telegram_delivered": receipt["delivered"],
                                    "telegram_receipt_seal_sha256": receipt["seal_sha256"]},
    })
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=26072102)
    parser.add_argument("--retry", type=int, default=0)
    args = parser.parse_args()
    try:
        artifact = run(args.repo.resolve(strict=True), args.source.resolve(strict=True),
                       args.output_dir.resolve(), args.seed, args.retry)
        print(json.dumps({"status": artifact["status"], "experiment_id": EXPERIMENT_ID,
                          "promotion": artifact["promotion"],
                          "seal_sha256": artifact["seal_sha256"]}, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
