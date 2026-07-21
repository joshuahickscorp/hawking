#!/usr/bin/env python3.12
"""Auction unused <=0.98-BPW bytes at the compact expert-output origin of drift."""
from __future__ import annotations

import argparse
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
import kimi_k26_long_run_causal as causal  # noqa: E402
import kimi_k26_long_run_manager as manager  # noqa: E402
import kimi_k26_long_run_repair_bracket as bracket  # noqa: E402


EXPERIMENT_ID = "LR10_UPSTREAM_REPRESENTATION_BYTE_AUCTION"
ARCHITECTURES = ("PARENT_OUTPUT_RESIDUAL", "EXPERT_INPUT_RESIDUAL")
RANKS = (2, 4, 8, 12, 16, 24)
RIDGE_FRACTIONS = (1e-4, 1e-2, 1e-1, 1.0, 10.0)
SHRINKAGES = (0.125, 0.25, 0.5, 1.0)
BIAS_OPTIONS = (False, True)


def fit_model(
    feature: np.ndarray, target: np.ndarray, rank: int,
    ridge_fraction: float, bias: bool,
) -> dict[str, Any]:
    feature = np.asarray(feature, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    offset = np.mean(target, axis=0, keepdims=True) if bias else np.zeros(
        (1, target.shape[1]), dtype=np.float32,
    )
    centered = target - offset
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    nonzero = int(np.sum(singular > singular[0] * 1e-6)) if singular.size else 0
    actual_rank = max(1, min(rank, nonzero))
    basis = right[:actual_rank].astype(np.float32)
    coefficients = centered @ basis.T
    gram = feature @ feature.T
    ridge = float(np.trace(gram) / max(1, feature.shape[0]) * ridge_fraction)
    projection = feature.T @ np.linalg.solve(
        gram + np.eye(gram.shape[0], dtype=np.float32) * ridge,
        coefficients,
    )
    energy = float(np.sum(singular[:actual_rank] ** 2) / (np.sum(singular ** 2) + 1e-30))
    return {"projection": projection.astype(np.float32), "basis": basis,
            "bias": offset[0].astype(np.float32), "rank": actual_rank,
            "ridge": ridge, "captured_target_energy": energy}


def apply_output(parent: np.ndarray, delta: np.ndarray, shrinkage: float) -> np.ndarray:
    value = (mx.array(parent).astype(mx.bfloat16) +
             mx.array(delta * shrinkage).astype(mx.bfloat16)).astype(mx.bfloat16)
    mx.eval(value)
    return np.asarray(value.astype(mx.float32), dtype=np.float32)


def run(repo: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    started_at = f1.now()
    started_wall = time.time()
    before = manager.resource_snapshot()
    audit = manager.audit(repo, notify=False)
    if audit["status"] != "PASS":
        raise f1.F1Error(f"pre-run guard audit failed: {audit['failures']}")
    prior_status = f1.read_json(repo / manager.STATUS_JSON)
    manager.write_status(repo, {**prior_status, "status": "RUNNING_EXPERIMENT",
                                "active_experiment": EXPERIMENT_ID,
                                "next_experiment": "LR10_IN_PROGRESS", "resources": before})
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_START",
        "experiment_id": EXPERIMENT_ID, "started_at": started_at, "status": "RUNNING",
        "hypothesis": "Unused bytes buy more capability when allocated where compact expert drift begins.",
        "config": {"seed": seed, "architectures": ARCHITECTURES, "ranks": RANKS,
                   "ridge_fractions": RIDGE_FRACTIONS, "shrinkages": SHRINKAGES,
                   "bias_options": BIAS_OPTIONS, "selection_scope": "F1_FIT_ONLY"},
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_result = f1.read_json(
        manager.RUNTIME / "f1_representation_bracket/doctor_auction/"
        "P1_DUAL_PATH_RECOVERY_R16X2_RESULT.json"
    )
    parent_payload = Path(parent_result["payload"]["path"])
    capture_path = manager.RUNTIME / "f1_representation_bracket/teacher_capture.npz"
    capture_receipt = f1.read_json(
        manager.RUNTIME / "f1_representation_bracket/teacher_capture.json"
    )
    if f1.sha256_file(capture_path) != capture_receipt["capture_sha256"]:
        raise f1.F1Error("sealed F1 teacher capture hash mismatch")
    with np.load(capture_path, allow_pickle=False) as capture:
        fit_x = np.asarray(capture["fit_x"])
        score_x = np.asarray(capture["score_x"])
        fit_teacher = np.asarray(capture["fit_teacher_output"])
        score_teacher = np.asarray(capture["score_teacher_output"])
        score_post = np.asarray(capture["score_post_attention"])
        score_routes = np.asarray(capture["score_routes"])
        score_route_weights = np.asarray(capture["score_route_weights"])
    _, fit_parent = causal.compact_expert_output(parent_payload, fit_x)
    _, score_parent = causal.compact_expert_output(parent_payload, score_x)
    target = fit_teacher - fit_parent
    fold_ids = np.arange(fit_x.shape[0]) % 4
    rng = np.random.default_rng(seed)
    rng.shuffle(fold_ids)
    grid = []
    best = None
    for architecture in ARCHITECTURES:
        fit_feature = fit_parent if architecture == "PARENT_OUTPUT_RESIDUAL" else fit_x
        for rank in RANKS:
            for ridge_fraction in RIDGE_FRACTIONS:
                for bias in BIAS_OPTIONS:
                    predictions = {shrinkage: np.zeros_like(fit_parent)
                                   for shrinkage in SHRINKAGES}
                    for fold in range(4):
                        train = fold_ids != fold
                        heldout = ~train
                        model = fit_model(fit_feature[train], target[train], rank,
                                          ridge_fraction, bias)
                        physical, _ = bracket.physicalize(model, "cv")
                        delta = bracket.predict(physical, fit_feature[heldout])
                        for shrinkage in SHRINKAGES:
                            predictions[shrinkage][heldout] = apply_output(
                                fit_parent[heldout], delta, shrinkage,
                            )
                    baseline_rows = causal.row_metrics(fit_teacher, causal.bf16(fit_parent))[
                        "relative_l2"]
                    for shrinkage, candidate in predictions.items():
                        rows = causal.row_metrics(fit_teacher, candidate)["relative_l2"]
                        improvement = bracket.paired_interval(
                            baseline_rows - rows,
                            seed + rank * 1000 + int(ridge_fraction * 100) +
                            int(shrinkage * 100) + int(bias) +
                            (0 if architecture == ARCHITECTURES[0] else 100_000),
                        )
                        record = {"architecture": architecture, "rank": rank,
                                  "ridge_fraction": ridge_fraction, "bias": bias,
                                  "shrinkage": shrinkage,
                                  "baseline_row_relative_l2_mean": float(
                                      np.mean(baseline_rows)),
                                  "row_relative_l2_mean": float(np.mean(rows)),
                                  "improvement": improvement}
                        grid.append(record)
                        key = (improvement["mean"], -rank, ridge_fraction, -shrinkage)
                        if best is None or key > best[0]:
                            best = (key, record)
    assert best is not None
    selected = best[1]
    final_feature = fit_parent if selected["architecture"] == ARCHITECTURES[0] else fit_x
    score_feature = score_parent if selected["architecture"] == ARCHITECTURES[0] else score_x
    model = fit_model(final_feature, target, selected["rank"],
                      selected["ridge_fraction"], selected["bias"])
    physical, blocks = bracket.physicalize(model, "upstream_residual")
    payload = bracket.write_payload(
        output_dir / "LR10_UPSTREAM_RESIDUAL.k26repair",
        parent_result["payload"]["sha256"],
        f"{selected['architecture']}_R{physical['rank']}", blocks,
        {"rank": physical["rank"], "ridge_fraction": selected["ridge_fraction"],
         "shrinkage": selected["shrinkage"], "bias": selected["bias"],
         "selection": "FOUR_FOLD_F1_FIT_ONLY"},
    )
    complete_bytes = parent_result["payload"]["bytes"] + payload["bytes"]
    complete_bpw = complete_bytes * 8 / bracket.LOGICAL_WEIGHTS
    fit_candidate = apply_output(
        fit_parent, bracket.predict(physical, final_feature), selected["shrinkage"],
    )
    score_candidate = apply_output(
        score_parent, bracket.predict(physical, score_feature), selected["shrinkage"],
    )
    score_parent_bf16 = causal.bf16(score_parent)
    score_base_rows = causal.row_metrics(score_teacher, score_parent_bf16)["relative_l2"]
    score_candidate_rows = causal.row_metrics(score_teacher, score_candidate)["relative_l2"]
    score_improvement = bracket.paired_interval(
        score_base_rows - score_candidate_rows, seed + 500_000,
    )
    sentinel = 0
    sentinel_slots = np.where(score_routes == sentinel)
    routed_tokens = sentinel_slots[0]
    routed_weights = score_route_weights[sentinel_slots][:, None]
    parent_contribution = routed_weights * score_parent_bf16[routed_tokens]
    candidate_contribution = routed_weights * score_candidate[routed_tokens]
    teacher_contribution = routed_weights * score_teacher[routed_tokens]
    parent_residual = score_post[routed_tokens] + parent_contribution
    candidate_residual = score_post[routed_tokens] + candidate_contribution
    teacher_residual = score_post[routed_tokens] + teacher_contribution
    promoted = (
        complete_bytes <= bracket.COMPLETE_CEILING_BYTES and
        score_improvement["ci95_low"] > 0 and
        f1.quality(score_teacher, score_candidate)["cosine_mean"] >
        f1.quality(score_teacher, score_parent_bf16)["cosine_mean"]
    )
    after = manager.resource_snapshot()
    ended_at = f1.now()
    duration = time.time() - started_wall
    artifact = f1.seal({
        "schema": "hawking.kimi_k26.long_run_upstream_auction.v1", "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "Unused bytes buy more capability when allocated where compact expert drift begins.",
        "config": {"seed": seed, "architectures": ARCHITECTURES, "ranks": RANKS,
                   "ridge_fractions": RIDGE_FRACTIONS, "shrinkages": SHRINKAGES,
                   "bias_options": BIAS_OPTIONS, "grid_rows": len(grid),
                   "selection_scope": "F1_FIT_ONLY", "score_used_for_selection": False},
        "parent": {"name": "P1_DUAL_PATH_RECOVERY_R16X2",
                   "payload_sha256": parent_result["payload"]["sha256"],
                   "physical_bytes": parent_result["payload"]["bytes"],
                   "complete_bpw": parent_result["physical_budget"]["actual_complete_bpw"],
                   "reported_score": parent_result["metrics"]["score_doctored"]},
        "selected": selected, "cross_validation_grid": grid,
        "physical_candidate": {"name": "UPSTREAM_RESIDUAL_CV",
                               "payload": payload, "complete_physical_bytes": complete_bytes,
                               "complete_bpw": complete_bpw,
                               "within_0_98_bpw": complete_bytes <= bracket.COMPLETE_CEILING_BYTES,
                               "allocation": {
                                   "parent_base_bytes": parent_result["payload"][
                                       "base_component_bytes"],
                                   "parent_doctor_bytes": parent_result["payload"][
                                       "doctor_component_bytes"],
                                   "parent_header_bytes": parent_result["payload"][
                                       "header_overhead_bytes"],
                                   "upstream_residual_repair_bytes": payload["bytes"]}},
        "f1": {
            "fit": {"parent": f1.quality(fit_teacher, causal.bf16(fit_parent)),
                    "candidate": f1.quality(fit_teacher, fit_candidate)},
            "heldout_score": {"parent": f1.quality(score_teacher, score_parent_bf16),
                              "candidate": f1.quality(score_teacher, score_candidate),
                              "paired_row_relative_l2_improvement": score_improvement,
                              "tokens": int(score_teacher.shape[0])},
            "heldout_routed_subset": {
                "tokens": int(routed_tokens.size),
                "parent_weighted_output": f1.quality(
                    teacher_contribution, parent_contribution),
                "candidate_weighted_output": f1.quality(
                    teacher_contribution, candidate_contribution),
                "parent_modeled_residual": f1.quality(teacher_residual, parent_residual),
                "candidate_modeled_residual": f1.quality(teacher_residual, candidate_residual),
            },
        },
        "evidence_parent": {"teacher_capture_sha256": capture_receipt["capture_sha256"],
                            "parent_result_seal_sha256": parent_result["seal_sha256"],
                            "lr09_seal_sha256": f1.read_json(
                                repo / "KIMI_K26_LONG_RUN_STRATIFIED_RESCUE.json"
                            )["seal_sha256"],
                            "guard_audit_seal_sha256": audit["seal_sha256"]},
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resource_observations": {"before": before, "after": after},
        "decision": ("PROMOTE_TO_HELDOUT_F2" if promoted else
                     "RETIRE_UPSTREAM_RESIDUAL_F1_FAILED"),
        "causal_interpretation": (
            "This candidate spends previously unused bits at the layer-1 expert output before the "
            "first observed router crossing; score tokens remain disjoint from fit and CV selection."
        ),
        "next_run_rationale": (
            "Install the frozen upstream residual in cached LR01 layer-1 states and run held-out F2."
            if promoted else
            "No upstream linear residual survives held-out F1; close this remaining linear family."
        ),
        "faults": [], "retries": 0,
    })
    artifact_path = repo / "KIMI_K26_LONG_RUN_UPSTREAM_AUCTION.json"
    f1.atomic_json(artifact_path, artifact)
    f1.atomic_json(manager.RUNTIME / artifact_path.name, artifact)
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_COMPLETE",
        "experiment_id": EXPERIMENT_ID, "hypothesis": artifact["hypothesis"],
        "config": artifact["config"], "parent_hash": parent_result["payload"]["sha256"],
        "candidate_hash": payload["sha256"], "physical_bytes": complete_bytes,
        "complete_bpw": complete_bpw, "started_at": started_at, "ended_at": ended_at,
        "duration_seconds": duration, "resources": artifact["resource_observations"],
        "sample_counts": {"fit_tokens": 32, "score_tokens": 32,
                          "grid_rows": len(grid)},
        "metrics": {"selected": selected, "f1": artifact["f1"]},
        "confidence_intervals": {"score_improvement": score_improvement},
        "faults": [], "retries": 0, "causal_interpretation": artifact["causal_interpretation"],
        "decision": artifact["decision"], "next_run_rationale": artifact["next_run_rationale"],
        "evidence_seal_sha256": artifact["seal_sha256"],
    })
    prior_status = f1.read_json(repo / manager.STATUS_JSON)
    status = manager.write_status(repo, {
        **prior_status, "status": "MANAGING", "active_experiment": None,
        "experiments_completed": int(prior_status.get("experiments_completed", 0)) + 1,
        "next_experiment": ("LR11_UPSTREAM_RESIDUAL_HELDOUT_F2" if promoted else
                            "LR11_NONLINEAR_BOUNDARY_FALSIFICATION"),
        "latest_result": {"experiment_id": EXPERIMENT_ID, "decision": artifact["decision"],
                          "selected": selected, "complete_bpw": complete_bpw,
                          "score_improvement": score_improvement,
                          "evidence_seal_sha256": artifact["seal_sha256"]},
        "resources": after,
    })
    receipt = manager.telegram(
        repo, "long-run:LR10-complete",
        ("[Kimi K2.6 long run] LR10 upstream byte auction complete\n"
         f"selected: {selected['architecture']} rank {selected['rank']} "
         f"shrink {selected['shrinkage']}\n"
         f"heldout F1 mean rescue: {score_improvement['mean']:.6f}\n"
         f"CI95 low: {score_improvement['ci95_low']:.6f}\n"
         f"complete BPW: {complete_bpw:.6f}\n"
         f"decision: {artifact['decision']}"),
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=26072110)
    args = parser.parse_args()
    try:
        artifact = run(args.repo.resolve(strict=True), args.output_dir.resolve(), args.seed)
        print(json.dumps({"status": artifact["status"], "decision": artifact["decision"],
                          "selected": artifact["selected"],
                          "seal_sha256": artifact["seal_sha256"]}, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
