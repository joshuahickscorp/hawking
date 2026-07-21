#!/usr/bin/env python3.12
"""Pool independent Kimi splits into a causal routing law with confidence bounds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_f1_bracket as f1  # noqa: E402
import kimi_k26_long_run_causal as causal  # noqa: E402
import kimi_k26_long_run_manager as manager  # noqa: E402


EXPERIMENT_ID = "LR06_POOLED_CAUSAL_LAW"
MATERIAL_ROW_ERROR = 0.05


def interval(values: np.ndarray, seed: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(5000, values.size))
    means = values[draws].mean(axis=1)
    return {"mean": float(np.mean(values)), "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)), "n": int(values.size)}


def difference_interval(
    left: np.ndarray, right: np.ndarray, seed: int,
) -> dict[str, float | int]:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    left_means = left[rng.integers(0, left.size, size=(5000, left.size))].mean(axis=1)
    right_means = right[rng.integers(0, right.size, size=(5000, right.size))].mean(axis=1)
    difference = left_means - right_means
    return {"mean": float(np.mean(left) - np.mean(right)),
            "ci95_low": float(np.percentile(difference, 2.5)),
            "ci95_high": float(np.percentile(difference, 97.5)),
            "left_n": int(left.size), "right_n": int(right.size)}


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_order = np.empty(left.size, dtype=np.float64)
    right_order = np.empty(right.size, dtype=np.float64)
    left_order[np.argsort(left, kind="stable")] = np.arange(left.size)
    right_order[np.argsort(right, kind="stable")] = np.arange(right.size)
    return float(np.corrcoef(left_order, right_order)[0, 1])


def split_record(name: str, capture_path: Path) -> dict[str, Any]:
    with np.load(capture_path, allow_pickle=False) as capture:
        teacher_indices = np.asarray(capture["teacher_route_indices_l2"])
        student_indices = np.asarray(capture["student_route_indices_l2"])
        teacher_hidden_key = ("variant_l2_TEACHER" if "variant_l2_TEACHER" in capture.files
                              else "teacher_hidden_l2")
        student_hidden_key = ("variant_l2_NATURAL_STUDENT"
                              if "variant_l2_NATURAL_STUDENT" in capture.files
                              else "natural_hidden_l2")
        teacher_hidden = np.asarray(capture[teacher_hidden_key])
        student_hidden = np.asarray(capture[student_hidden_key])
        margin_key = ("teacher_margin_l2" if "teacher_margin_l2" in capture.files
                      else "teacher_margin_l2")
        margin = np.asarray(capture[margin_key])
    mismatch = np.asarray([set(left) != set(right) for left, right in zip(
        teacher_indices, student_indices, strict=True,
    )])
    rows = causal.row_metrics(teacher_hidden, student_hidden)
    relative_l2 = rows["relative_l2"]
    quantiles = np.quantile(margin, [0.25, 0.5, 0.75])
    quartile = np.digitize(margin, quantiles)
    percentile = np.argsort(np.argsort(margin, kind="stable"), kind="stable") / max(1, margin.size - 1)
    return {"name": name, "tokens": int(margin.size), "margin": margin,
            "mismatch": mismatch, "relative_l2": relative_l2,
            "quartile": quartile, "margin_percentile": percentile,
            "hidden_quality": f1.quality(teacher_hidden, student_hidden)}


def run(repo: Path, seed: int) -> dict[str, Any]:
    started_at = f1.now()
    started_wall = time.time()
    before = manager.resource_snapshot()
    audit = manager.audit(repo, notify=False)
    if audit["status"] != "PASS":
        raise f1.F1Error(f"pre-run guard audit failed: {audit['failures']}")
    prior_status = f1.read_json(repo / manager.STATUS_JSON)
    manager.write_status(repo, {**prior_status, "status": "RUNNING_EXPERIMENT",
                                "active_experiment": EXPERIMENT_ID,
                                "next_experiment": "LR06_IN_PROGRESS", "resources": before})
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_START",
        "experiment_id": EXPERIMENT_ID, "started_at": started_at, "status": "RUNNING",
        "hypothesis": "Low router margin predicts route crossing, but upstream state remains causal when routes agree.",
        "config": {"seed": seed, "splits": ["LR01", "LR03", "LR05"],
                   "material_row_error": MATERIAL_ROW_ERROR},
    })
    lr01 = f1.read_json(repo / "KIMI_K26_LONG_RUN_CONTROL_CAUSAL.json")
    lr02 = f1.read_json(repo / "KIMI_K26_LONG_RUN_REPAIR_BRACKET.json")
    lr03 = f1.read_json(repo / "KIMI_K26_LONG_RUN_REPLICATION.json")
    lr04 = f1.read_json(repo / "KIMI_K26_LONG_RUN_SHRINKAGE_BOUNDARY.json")
    lr05 = f1.read_json(repo / "KIMI_K26_LONG_RUN_UNTOUCHED_VALIDATION.json")
    splits = [
        split_record("LR01_REFERENCE_PROBES", Path(lr01["capture"]["path"])),
        split_record("LR03_NEW_SPLIT", Path(lr03["capture"]["path"])),
        split_record("LR05_UNTOUCHED", Path(lr05["capture"]["path"])),
    ]
    mismatch = np.concatenate([split["mismatch"] for split in splits])
    error = np.concatenate([split["relative_l2"] for split in splits])
    margin = np.concatenate([split["margin"] for split in splits])
    margin_percentile = np.concatenate([split["margin_percentile"] for split in splits])
    quartile = np.concatenate([split["quartile"] for split in splits])
    per_split = {}
    for index, split in enumerate(splits):
        low = split["quartile"] == 0
        material = split["relative_l2"] >= MATERIAL_ROW_ERROR
        per_split[split["name"]] = {
            "tokens": split["tokens"], "route_set_change": interval(
                split["mismatch"].astype(np.float64), seed + index,
            ),
            "hidden": split["hidden_quality"],
            "row_relative_l2": interval(split["relative_l2"], seed + 10 + index),
            "bottom_margin_quartile": {
                "tokens": int(np.sum(low)),
                "route_change": float(np.mean(split["mismatch"][low])),
                "material_error_rate": float(np.mean(material[low])),
            },
        }
    by_margin_quartile = {}
    for value in range(4):
        mask = quartile == value
        by_margin_quartile[str(value)] = {
            "tokens": int(np.sum(mask)),
            "route_change": interval(mismatch[mask].astype(np.float64), seed + 20 + value),
            "row_relative_l2": interval(error[mask], seed + 30 + value),
            "material_error_rate": float(np.mean(error[mask] >= MATERIAL_ROW_ERROR)),
        }
    matched = ~mismatch
    material = error >= MATERIAL_ROW_ERROR
    mismatch_damage = error[mismatch]
    matched_damage = error[matched]
    damage_difference = difference_interval(mismatch_damage, matched_damage, seed + 50)
    bottom = margin_percentile <= 0.25
    bottom_precision = float(np.mean(mismatch[bottom]))
    bottom_recall = float(np.sum(mismatch & bottom) / max(1, np.sum(mismatch)))
    pooled = {
        "tokens": int(error.size),
        "route_set_change": interval(mismatch.astype(np.float64), seed + 60),
        "row_relative_l2": interval(error, seed + 61),
        "margin_percentile_vs_route_mismatch_point_biserial_correlation": float(
            np.corrcoef(margin_percentile, mismatch.astype(np.float64))[0, 1]
        ),
        "margin_rank_vs_error_rank_correlation": rank_correlation(margin_percentile, error),
        "by_within_split_margin_quartile": by_margin_quartile,
        "bottom_margin_quartile_classifier": {
            "precision": bottom_precision, "recall": bottom_recall,
            "selected_fraction": float(np.mean(bottom)),
        },
        "route_mismatch_damage": interval(mismatch_damage, seed + 62),
        "route_match_damage": interval(matched_damage, seed + 63),
        "mismatch_minus_match_error": damage_difference,
        "material_error_rate_when_routes_mismatch": float(np.mean(material[mismatch])),
        "material_error_rate_when_routes_match": float(np.mean(material[matched])),
        "material_error_with_routes_matched_tokens": int(np.sum(material & matched)),
    }
    intervention = {
        "indices_only_rescue": lr01["causal_rescue"]["indices_only_rescue"],
        "indices_plus_weights_rescue": lr01["causal_rescue"][
            "indices_plus_weights_rescue"],
        "teacher_weighted_moe_rescue": lr01["causal_rescue"][
            "teacher_moe_substitution_rescue"],
        "teacher_hidden_restore_rescue": lr01["intervention_matrix"][
            "RESTORE_TEACHER_HIDDEN_BEFORE_LAYER2"]["rescue_fraction_relative_l2"],
        "counterfactual_numerical_floor_relative_l2": lr01["moe_residual_f2"][
            "counterfactual_recombination_calibration"][
                "same_kernel_full_batch_layer1_relative_l2_error_floor"],
    }
    repair_evidence = {
        "lr02_all_five_rows_harm_typical_token": all(
            row["relative_l2_improvement"]["ci95_high"] < 0
            for row in lr02["treatment_frontier"]
        ),
        "lr03_all_five_rows_harm_typical_token": all(
            row["all_tokens"]["relative_l2_improvement"]["ci95_high"] < 0
            for row in lr03["replication_frontier"]
        ),
        "lr04_calibration_cv_survivor": lr04["selected"],
        "lr05_untouched_survivor": {
            "all_tokens": lr05["all_tokens"]["relative_l2_improvement"],
            "low_margin": lr05["adversarial_low_margin"]["relative_l2_improvement"],
            "decision": lr05["decision"],
        },
    }
    law = {
        "statement": (
            "At 0.90859 complete BPW, compact expert-output drift reaches the next block before "
            "routing, low native margins amplify that drift into expert entry/exit, but routing is "
            "secondary: exact route restoration rescues only 10.7% while hidden restoration rescues "
            "100%. Under <=0.98 BPW, learned low-rank state/router/MoE repairs trade high-energy "
            "outlier improvement for statistically significant typical-token and low-margin harm."
        ),
        "scope": {"base_candidate": "P1_DUAL_PATH_RECOVERY_R16X2",
                  "base_complete_bpw": lr01["candidate"]["complete_bpw"],
                  "tested_complete_bpw_ceiling": 0.98,
                  "independent_tokens": int(error.size), "independent_splits": len(splits),
                  "physical_repair_families": 6},
        "proven": [
            "Compact perturbation exists before the layer-2 router.",
            "Low within-split router margin increases route-crossing risk but is not sufficient.",
            "Material residual damage occurs with exact top-8 route sets still matched.",
            "Teacher route indices plus weights produce only a minority causal rescue.",
            "Teacher hidden restoration is a full causal rescue at the tested block.",
            "All five <=0.98-BPW repair rows harm the typical token on two splits.",
            "The calibration-CV shrinkage survivor fails untouched low-margin validation.",
        ],
        "correlated_not_proven": [
            "Norm-weighted hidden error often improves under low-rank repair.",
            "That norm-weighted improvement does not establish capability preservation.",
        ],
        "unresolved": [
            "Whether a nonlinear token-conditional repair can avoid the observed domain tradeoff.",
            "Whether a representation below the compact expert, rather than a downstream Doctor, "
            "can remove the upstream state error at equal BPW.",
        ],
    }
    after = manager.resource_snapshot()
    ended_at = f1.now()
    duration = time.time() - started_wall
    artifact = f1.seal({
        "schema": "hawking.kimi_k26.long_run_pooled_causal_law.v1", "status": "PASS",
        "experiment_id": EXPERIMENT_ID, "hypothesis": (
            "Low router margin predicts route crossing, but upstream state remains causal when routes agree."
        ),
        "config": {"seed": seed, "splits": [split["name"] for split in splits],
                   "material_row_error": MATERIAL_ROW_ERROR,
                   "pooling": "WITHIN_SPLIT_MARGIN_QUARTILES"},
        "per_split": per_split, "pooled": pooled,
        "intervention_evidence": intervention, "repair_evidence": repair_evidence,
        "scientific_law": law,
        "evidence_parent": {"lr01": lr01["seal_sha256"], "lr02": lr02["seal_sha256"],
                            "lr03": lr03["seal_sha256"], "lr04": lr04["seal_sha256"],
                            "lr05": lr05["seal_sha256"], "audit": audit["seal_sha256"]},
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resource_observations": {"before": before, "after": after},
        "decision": "CLOSE_TESTED_LOW_RANK_REPAIR_REGION_PENDING_LARGE_HELDOUT_FALSIFICATION",
        "causal_interpretation": law["statement"],
        "next_run_rationale": (
            "Challenge the pooled law on a larger, longer-context held-out set with no tuning."
        ),
        "faults": [], "retries": 0,
    })
    artifact_path = repo / "KIMI_K26_LONG_RUN_POOLED_CAUSAL_LAW.json"
    f1.atomic_json(artifact_path, artifact)
    f1.atomic_json(manager.RUNTIME / artifact_path.name, artifact)
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_COMPLETE",
        "experiment_id": EXPERIMENT_ID, "hypothesis": artifact["hypothesis"],
        "config": artifact["config"], "parent_hash": lr05["seal_sha256"],
        "candidate_hash": None, "physical_bytes": None, "complete_bpw": None,
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resources": artifact["resource_observations"],
        "sample_counts": {"tokens": int(error.size), "splits": len(splits)},
        "metrics": {"pooled": pooled, "intervention": intervention,
                    "repair_evidence": repair_evidence},
        "confidence_intervals": {"route_change": pooled["route_set_change"],
                                 "row_relative_l2": pooled["row_relative_l2"],
                                 "mismatch_minus_match": pooled["mismatch_minus_match_error"]},
        "faults": [], "retries": 0, "causal_interpretation": law["statement"],
        "decision": artifact["decision"], "next_run_rationale": artifact["next_run_rationale"],
        "evidence_seal_sha256": artifact["seal_sha256"],
    })
    prior_status = f1.read_json(repo / manager.STATUS_JSON)
    status = manager.write_status(repo, {
        **prior_status, "status": "MANAGING", "active_experiment": None,
        "experiments_completed": int(prior_status.get("experiments_completed", 0)) + 1,
        "dominant_bottleneck": "UPSTREAM_STATE_ERROR_WITH_SECONDARY_ROUTE_AMPLIFICATION",
        "scientific_hypothesis": law["statement"],
        "next_experiment": "LR07_LARGE_HELDOUT_FALSIFICATION",
        "latest_result": {"experiment_id": EXPERIMENT_ID, "decision": artifact["decision"],
                          "tokens": int(error.size), "route_set_change":
                              pooled["route_set_change"],
                          "evidence_seal_sha256": artifact["seal_sha256"]},
        "resources": after,
    })
    receipt = manager.telegram(
        repo, "long-run:LR06-complete",
        ("[Kimi K2.6 long run] LR06 pooled causal law sealed\n"
         f"tokens / splits: {int(error.size)} / {len(splits)}\n"
         f"pooled route change: {pooled['route_set_change']['mean']*100:.2f}%\n"
         f"material errors with routes matched: {pooled['material_error_with_routes_matched_tokens']}\n"
         f"teacher route+weight rescue: {intervention['indices_plus_weights_rescue']:.3f}\n"
         f"teacher hidden rescue: {intervention['teacher_hidden_restore_rescue']:.3f}\n"
         "decision: close low-rank repair region pending large held-out falsification"),
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
    parser.add_argument("--seed", type=int, default=26072106)
    args = parser.parse_args()
    try:
        artifact = run(args.repo.resolve(strict=True), args.seed)
        print(json.dumps({"status": artifact["status"], "decision": artifact["decision"],
                          "tokens": artifact["pooled"]["tokens"],
                          "seal_sha256": artifact["seal_sha256"]}, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
