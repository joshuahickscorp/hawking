#!/usr/bin/env python3.12
"""Calibration-only rank/ridge/shrinkage boundary for post-MoE hidden recovery."""
from __future__ import annotations

import argparse
import gc
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


EXPERIMENT_ID = "LR04_CALIBRATION_ONLY_SHRINKAGE_BOUNDARY"
CALIBRATION_SEGMENTS = ("factual", "science", "coding", "mathematics")
RANKS = (2, 4, 8, 12, 16)
RIDGE_FRACTIONS = (1e-4, 1e-2, 1e-1, 1.0, 10.0)
SHRINKAGES = (0.125, 0.25, 0.5, 1.0)
BIAS_OPTIONS = (False, True)


def fit_model(
    x: np.ndarray, target: np.ndarray, rank: int, ridge_fraction: float, bias: bool,
) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    offset = np.mean(target, axis=0, keepdims=True) if bias else np.zeros((1, target.shape[1]),
                                                                          dtype=np.float32)
    centered = target - offset
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    nonzero = int(np.sum(singular > singular[0] * 1e-6)) if singular.size else 0
    actual_rank = max(1, min(rank, nonzero))
    basis = right[:actual_rank].astype(np.float32)
    coefficients = centered @ basis.T
    gram = x @ x.T
    ridge = float(np.trace(gram) / max(1, x.shape[0]) * ridge_fraction)
    projection = x.T @ np.linalg.solve(
        gram + np.eye(gram.shape[0], dtype=np.float32) * ridge,
        coefficients,
    )
    energy = float(np.sum(singular[:actual_rank] ** 2) / (np.sum(singular ** 2) + 1e-30))
    return {"projection": projection.astype(np.float32), "basis": basis,
            "bias": offset[0].astype(np.float32), "rank": actual_rank,
            "ridge": ridge, "captured_target_energy": energy}


def apply_hidden(natural: np.ndarray, delta: np.ndarray, shrinkage: float) -> np.ndarray:
    result = (mx.array(natural).astype(mx.bfloat16) +
              mx.array(delta * shrinkage).astype(mx.bfloat16)).astype(mx.bfloat16)
    mx.eval(result)
    return np.asarray(result.astype(mx.float32), dtype=np.float32)


def run(repo: Path, source: Path, output_dir: Path, seed: int) -> dict[str, Any]:
    started_at = f1.now()
    started_wall = time.time()
    before = manager.resource_snapshot()
    audit = manager.audit(repo, notify=False)
    if audit["status"] != "PASS":
        raise f1.F1Error(f"pre-run guard audit failed: {audit['failures']}")
    prior_status = f1.read_json(repo / manager.STATUS_JSON)
    manager.write_status(repo, {**prior_status, "status": "RUNNING_EXPERIMENT",
                                "active_experiment": EXPERIMENT_ID,
                                "next_experiment": "LR04_IN_PROGRESS", "resources": before})
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_START",
        "experiment_id": EXPERIMENT_ID, "started_at": started_at, "status": "RUNNING",
        "hypothesis": "Cross-validated shrinkage can prevent high-energy outliers from dominating repair.",
        "config": {"seed": seed, "selection_scope": "LR01_CALIBRATION_ONLY",
                   "ranks": RANKS, "ridge_fractions": RIDGE_FRACTIONS,
                   "shrinkages": SHRINKAGES, "bias_options": BIAS_OPTIONS},
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    lr01 = f1.read_json(repo / "KIMI_K26_LONG_RUN_CONTROL_CAUSAL.json")
    lr03 = f1.read_json(repo / "KIMI_K26_LONG_RUN_REPLICATION.json")
    capture_path = Path(lr01["capture"]["path"])
    if f1.sha256_file(capture_path) != lr01["capture"]["sha256"]:
        raise f1.F1Error("LR01 capture hash mismatch")
    with np.load(capture_path, allow_pickle=False) as capture:
        student_x = np.asarray(capture["student_x_l2"])
        teacher_hidden = np.asarray(capture["variant_l2_TEACHER"])
        natural_hidden = np.asarray(capture["variant_l2_NATURAL_STUDENT"])
    _, batch = causal.prepare_requests(source)
    records = batch["token_records"]
    segment = np.asarray([record["segment"] for record in records])
    calibration_mask = np.isin(segment, CALIBRATION_SEGMENTS)
    if int(np.sum(calibration_mask)) != 25:
        raise f1.F1Error("unexpected LR01 calibration size")
    x = student_x[calibration_mask]
    teacher = teacher_hidden[calibration_mask]
    natural = natural_hidden[calibration_mask]
    calibration_segment = segment[calibration_mask]
    target = teacher - natural
    baseline_rows = causal.row_metrics(teacher, natural)["relative_l2"]
    grid = []
    best = None
    for rank in RANKS:
        for ridge_fraction in RIDGE_FRACTIONS:
            for bias in BIAS_OPTIONS:
                fold_predictions = {shrinkage: np.zeros_like(natural)
                                    for shrinkage in SHRINKAGES}
                for heldout_segment in CALIBRATION_SEGMENTS:
                    train = calibration_segment != heldout_segment
                    heldout = ~train
                    model = fit_model(x[train], target[train], rank, ridge_fraction, bias)
                    physical, _ = bracket.physicalize(model, "cv")
                    delta = bracket.predict(physical, x[heldout])
                    for shrinkage in SHRINKAGES:
                        fold_predictions[shrinkage][heldout] = apply_hidden(
                            natural[heldout], delta, shrinkage,
                        )
                for shrinkage, candidate in fold_predictions.items():
                    candidate_rows = causal.row_metrics(teacher, candidate)["relative_l2"]
                    improvement = baseline_rows - candidate_rows
                    interval = bracket.paired_interval(
                        improvement, seed + rank * 1000 + int(ridge_fraction * 100) +
                        int(shrinkage * 100) + int(bias),
                    )
                    record = {"rank": rank, "ridge_fraction": ridge_fraction,
                              "shrinkage": shrinkage, "bias": bias,
                              "row_relative_l2_mean": float(np.mean(candidate_rows)),
                              "baseline_row_relative_l2_mean": float(np.mean(baseline_rows)),
                              "improvement": interval,
                              "folds": list(CALIBRATION_SEGMENTS)}
                    grid.append(record)
                    key = (interval["mean"], -rank, ridge_fraction, -shrinkage)
                    if best is None or key > best[0]:
                        best = (key, record)
    assert best is not None
    selected = best[1]
    final_model = fit_model(x, target, selected["rank"],
                            selected["ridge_fraction"], selected["bias"])
    final_physical, final_blocks = bracket.physicalize(final_model, "post_moe_hidden")
    parent_hash = lr01["candidate"]["payload_sha256"]
    payload = bracket.write_payload(
        output_dir / "LR04_POST_MOE_HIDDEN_CV.k26repair", parent_hash,
        "CALIBRATION_CV_POST_MOE_HIDDEN_RECOVERY",
        final_blocks,
        {"rank": final_physical["rank"], "ridge_fraction": selected["ridge_fraction"],
         "shrinkage": selected["shrinkage"], "bias": selected["bias"],
         "selection": "LEAVE_ONE_SEGMENT_OUT_CALIBRATION_ONLY"},
    )
    complete_bytes = lr01["candidate"]["physical_bytes"] + payload["bytes"]
    complete_bpw = complete_bytes * 8 / bracket.LOGICAL_WEIGHTS
    final_delta = bracket.predict(final_physical, x)
    calibration_fit_hidden = apply_hidden(natural, final_delta, selected["shrinkage"])
    calibration_fit = f1.quality(teacher, calibration_fit_hidden)
    calibration_fit_rows = causal.row_metrics(teacher, calibration_fit_hidden)["relative_l2"]
    advance = selected["improvement"]["mean"] > 0
    after = manager.resource_snapshot()
    ended_at = f1.now()
    duration = time.time() - started_wall
    artifact = f1.seal({
        "schema": "hawking.kimi_k26.long_run_shrinkage_boundary.v1", "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "Cross-validated shrinkage can prevent high-energy outliers from dominating repair.",
        "config": {"seed": seed, "selection_scope": "LR01_CALIBRATION_ONLY",
                   "calibration_segments": CALIBRATION_SEGMENTS,
                   "calibration_tokens": int(np.sum(calibration_mask)),
                   "heldout_or_replication_used_for_selection": False,
                   "ranks": RANKS, "ridge_fractions": RIDGE_FRACTIONS,
                   "shrinkages": SHRINKAGES, "bias_options": BIAS_OPTIONS,
                   "grid_rows": len(grid)},
        "selected": selected,
        "physical_candidate": {"payload": payload, "parent_sha256": parent_hash,
                               "complete_physical_bytes": complete_bytes,
                               "complete_bpw": complete_bpw,
                               "within_0_98_bpw": complete_bytes <= bracket.COMPLETE_CEILING_BYTES,
                               "allocation": {"parent_bytes": lr01["candidate"]["physical_bytes"],
                                              "post_moe_hidden_repair_bytes": payload["bytes"]}},
        "calibration_fit_not_selection": {
            "hidden": calibration_fit,
            "row_relative_l2_mean": float(np.mean(calibration_fit_rows))},
        "cross_validation_grid": grid,
        "evidence_parent": {"lr01_seal_sha256": lr01["seal_sha256"],
                            "lr03_seal_sha256": lr03["seal_sha256"],
                            "guard_audit_seal_sha256": audit["seal_sha256"]},
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resource_observations": {"before": before, "after": after},
        "decision": ("ADVANCE_SINGLE_PREREGISTERED_ROW_TO_UNTOUCHED_SPLIT" if advance else
                     "RETIRE_SHRINKAGE_FAMILY_CALIBRATION_CV_FAILED"),
        "causal_interpretation": (
            "This boundary tests whether repair failure is merely over-aggressive amplitude. "
            "Only leave-one-segment-out calibration performance may select the next row."
        ),
        "next_run_rationale": (
            "Test exactly the selected installed payload and shrinkage on untouched contexts."
            if advance else
            "No amplitude/rank regularization survives internal calibration; close this repair family."
        ),
        "faults": [], "retries": 0,
    })
    artifact_path = repo / "KIMI_K26_LONG_RUN_SHRINKAGE_BOUNDARY.json"
    f1.atomic_json(artifact_path, artifact)
    f1.atomic_json(manager.RUNTIME / artifact_path.name, artifact)
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_COMPLETE",
        "experiment_id": EXPERIMENT_ID, "hypothesis": artifact["hypothesis"],
        "config": artifact["config"], "parent_hash": parent_hash,
        "candidate_hash": payload["sha256"], "physical_bytes": complete_bytes,
        "complete_bpw": complete_bpw, "started_at": started_at, "ended_at": ended_at,
        "duration_seconds": duration, "resources": artifact["resource_observations"],
        "sample_counts": {"tokens": int(np.sum(calibration_mask)), "grid_rows": len(grid),
                          "folds": len(CALIBRATION_SEGMENTS)},
        "metrics": {"selected": selected, "calibration_fit":
                    artifact["calibration_fit_not_selection"]},
        "confidence_intervals": {"selected_cv_improvement": selected["improvement"]},
        "faults": [], "retries": 0, "causal_interpretation": artifact["causal_interpretation"],
        "decision": artifact["decision"], "next_run_rationale": artifact["next_run_rationale"],
        "evidence_seal_sha256": artifact["seal_sha256"],
    })
    prior_status = f1.read_json(repo / manager.STATUS_JSON)
    status = manager.write_status(repo, {
        **prior_status, "status": "MANAGING", "active_experiment": None,
        "experiments_completed": int(prior_status.get("experiments_completed", 0)) + 1,
        "next_experiment": ("LR05_UNTOUCHED_SHRINKAGE_VALIDATION" if advance else
                            "LR05_BOUNDARY_CLOSURE_REPLICATION"),
        "latest_result": {"experiment_id": EXPERIMENT_ID, "selected": selected,
                          "complete_bpw": complete_bpw, "decision": artifact["decision"],
                          "evidence_seal_sha256": artifact["seal_sha256"]},
        "resources": after,
    })
    receipt = manager.telegram(
        repo, "long-run:LR04-complete",
        ("[Kimi K2.6 long run] LR04 shrinkage boundary complete\n"
         f"CV rows / tokens: {len(grid)} / {int(np.sum(calibration_mask))}\n"
         f"selected rank/ridge/shrink: {selected['rank']} / "
         f"{selected['ridge_fraction']} / {selected['shrinkage']}\n"
         f"CV mean rescue: {selected['improvement']['mean']:.6f}\n"
         f"decision: {artifact['decision']}\n"
         f"free disk: {after['free_disk_bytes']/1024**3:.2f} GiB"),
    )
    manager.write_status(repo, {
        **status, "latest_result": {**status["latest_result"],
                                    "telegram_delivered": receipt["delivered"],
                                    "telegram_receipt_seal_sha256": receipt["seal_sha256"]},
    })
    gc.collect()
    mx.clear_cache()
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=26072104)
    args = parser.parse_args()
    try:
        artifact = run(args.repo.resolve(strict=True), args.source.resolve(strict=True),
                       args.output_dir.resolve(), args.seed)
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
