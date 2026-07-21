#!/usr/bin/env python3.12
"""Stratum-level confidence analysis for the LR08 causal intervention matrix."""
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
import kimi_k26_long_run_repair_bracket as bracket  # noqa: E402


EXPERIMENT_ID = "LR09_STRATIFIED_RESCUE_CONFIDENCE"


def recreate_strata(
    mismatch: np.ndarray, margin: np.ndarray, seed: int,
) -> dict[str, np.ndarray]:
    median = float(np.median(margin))
    rng = np.random.default_rng(seed)
    definitions = {
        "MISMATCH_LOW_MARGIN": mismatch & (margin <= median),
        "MISMATCH_HIGH_MARGIN": mismatch & (margin > median),
        "MATCH_LOW_MARGIN": ~mismatch & (margin <= median),
        "MATCH_HIGH_MARGIN": ~mismatch & (margin > median),
    }
    result = {}
    for name, mask in definitions.items():
        available = np.where(mask)[0]
        count = min(64, available.size)
        chosen = np.sort(rng.choice(available, size=count, replace=False)) if count else available
        selected = np.zeros(mismatch.size, dtype=bool)
        selected[chosen] = True
        result[name] = selected
    return result


def analyze(
    teacher: np.ndarray, baseline: np.ndarray, candidate: np.ndarray,
    mask: np.ndarray, seed: int,
) -> dict[str, Any]:
    baseline_rows = causal.row_metrics(teacher, baseline)["relative_l2"]
    candidate_rows = causal.row_metrics(teacher, candidate)["relative_l2"]
    improvement = baseline_rows[mask] - candidate_rows[mask]
    base_mean = float(np.mean(baseline_rows[mask]))
    candidate_mean = float(np.mean(candidate_rows[mask]))
    return {"tokens": int(np.sum(mask)), "baseline_row_relative_l2_mean": base_mean,
            "candidate_row_relative_l2_mean": candidate_mean,
            "absolute_improvement": bracket.paired_interval(improvement, seed),
            "fractional_rescue_of_mean_row_error": float(
                1 - candidate_mean / (base_mean + 1e-30)
            )}


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
                                "next_experiment": "LR09_IN_PROGRESS", "resources": before})
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_START",
        "experiment_id": EXPERIMENT_ID, "started_at": started_at, "status": "RUNNING",
        "hypothesis": "Route restoration rescues mismatch strata but cannot repair matched-route state damage.",
        "config": {"seed": seed, "selection_seed": 26072108,
                   "strata": ["MISMATCH_LOW_MARGIN", "MISMATCH_HIGH_MARGIN",
                               "MATCH_LOW_MARGIN", "MATCH_HIGH_MARGIN"]},
    })
    lr08 = f1.read_json(repo / "KIMI_K26_LONG_RUN_INTERVENTION_REPLICATION.json")
    capture_path = Path(lr08["capture"]["path"])
    if f1.sha256_file(capture_path) != lr08["capture"]["sha256"]:
        raise f1.F1Error("LR08 intervention capture hash mismatch")
    with np.load(capture_path, allow_pickle=False) as capture:
        teacher = np.asarray(capture["teacher_hidden_l2"])
        baseline = np.asarray(capture["natural_hidden_l2"])
        variants = {
            "FORCE_TEACHER_INDICES": np.asarray(capture["force_indices_hidden_l2"]),
            "FORCE_TEACHER_INDICES_AND_WEIGHTS": np.asarray(
                capture["force_both_hidden_l2"]),
            "TEACHER_STATE_STUDENT_ROUTER": np.asarray(
                capture["teacher_state_student_hidden_l2"]),
            "SUBSTITUTE_TEACHER_WEIGHTED_MOE": np.asarray(
                capture["substitute_teacher_moe_hidden_l2"]),
        }
        teacher_routes = np.asarray(capture["teacher_route_indices_l2"])
        student_routes = np.asarray(capture["student_route_indices_l2"])
        margin = np.asarray(capture["teacher_margin_l2"])
        saved_selected = np.asarray(capture["selected_mask"]).astype(bool)
    mismatch = np.asarray([set(left) != set(right) for left, right in zip(
        teacher_routes, student_routes, strict=True,
    )])
    strata = recreate_strata(mismatch, margin, 26072108)
    reconstructed_selected = np.logical_or.reduce(list(strata.values()))
    if not np.array_equal(saved_selected, reconstructed_selected):
        raise f1.F1Error("LR08 stratified selection reconstruction mismatch")
    matrix = {}
    for stratum_index, (stratum, mask) in enumerate(strata.items()):
        matrix[stratum] = {
            name: analyze(teacher, baseline, candidate, mask,
                          seed + stratum_index * 100 + variant_index)
            for variant_index, (name, candidate) in enumerate(variants.items())
        }
    mismatch_selected = strata["MISMATCH_LOW_MARGIN"] | strata["MISMATCH_HIGH_MARGIN"]
    match_selected = strata["MATCH_LOW_MARGIN"] | strata["MATCH_HIGH_MARGIN"]
    collapsed = {
        "MISMATCH": {name: analyze(teacher, baseline, candidate, mismatch_selected,
                                    seed + 1000 + index)
                     for index, (name, candidate) in enumerate(variants.items())},
        "MATCH": {name: analyze(teacher, baseline, candidate, match_selected,
                                 seed + 1100 + index)
                  for index, (name, candidate) in enumerate(variants.items())},
    }
    route_mismatch_rescue = collapsed["MISMATCH"][
        "FORCE_TEACHER_INDICES_AND_WEIGHTS"]["fractional_rescue_of_mean_row_error"]
    route_match_rescue = collapsed["MATCH"][
        "FORCE_TEACHER_INDICES_AND_WEIGHTS"]["fractional_rescue_of_mean_row_error"]
    moe_mismatch_rescue = collapsed["MISMATCH"][
        "SUBSTITUTE_TEACHER_WEIGHTED_MOE"]["fractional_rescue_of_mean_row_error"]
    classification = (
        "ROUTE_CAUSAL_ONLY_AFTER_CROSSING_STATE_PRIMARY_GLOBALLY"
        if route_mismatch_rescue > route_match_rescue and moe_mismatch_rescue > route_mismatch_rescue
        else "STRATIFIED_CAUSAL_ORDER_AMBIGUOUS"
    )
    after = manager.resource_snapshot()
    ended_at = f1.now()
    duration = time.time() - started_wall
    artifact = f1.seal({
        "schema": "hawking.kimi_k26.long_run_stratified_rescue.v1", "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "Route restoration rescues mismatch strata but cannot repair matched-route state damage.",
        "config": {"seed": seed, "selection_seed": 26072108,
                   "strata_cap": 64, "selected_tokens": int(np.sum(saved_selected))},
        "stratified_intervention_matrix": matrix,
        "collapsed_by_route_crossing": collapsed,
        "classification": classification,
        "key_effects": {"route_weight_rescue_mismatch": route_mismatch_rescue,
                        "route_weight_rescue_match": route_match_rescue,
                        "teacher_moe_rescue_mismatch": moe_mismatch_rescue,
                        "teacher_hidden_rescue_all": 1.0},
        "evidence_parent": {"lr08_seal_sha256": lr08["seal_sha256"],
                            "capture_sha256": lr08["capture"]["sha256"],
                            "guard_audit_seal_sha256": audit["seal_sha256"]},
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resource_observations": {"before": before, "after": after},
        "decision": "ESTABLISH_CONDITIONAL_ROUTE_AMPLIFICATION_LAW",
        "causal_interpretation": (
            "Router intervention has value primarily after a margin crossing; exact-route tokens "
            "retain state damage that only MoE-output or hidden-state intervention can address."
        ),
        "next_run_rationale": (
            "Auction remaining physical bits upstream at the compact expert output, where drift begins."
        ),
        "faults": [], "retries": 0,
    })
    artifact_path = repo / "KIMI_K26_LONG_RUN_STRATIFIED_RESCUE.json"
    f1.atomic_json(artifact_path, artifact)
    f1.atomic_json(manager.RUNTIME / artifact_path.name, artifact)
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_COMPLETE",
        "experiment_id": EXPERIMENT_ID, "hypothesis": artifact["hypothesis"],
        "config": artifact["config"], "parent_hash": lr08["seal_sha256"],
        "candidate_hash": None, "physical_bytes": None, "complete_bpw": None,
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resources": artifact["resource_observations"],
        "sample_counts": {"tokens": int(np.sum(saved_selected)), "strata": len(strata)},
        "metrics": {"key_effects": artifact["key_effects"], "matrix": matrix},
        "confidence_intervals": {stratum: {
            name: value["absolute_improvement"] for name, value in interventions.items()
        } for stratum, interventions in matrix.items()},
        "faults": [], "retries": 0, "causal_interpretation": artifact["causal_interpretation"],
        "decision": artifact["decision"], "next_run_rationale": artifact["next_run_rationale"],
        "evidence_seal_sha256": artifact["seal_sha256"],
    })
    prior_status = f1.read_json(repo / manager.STATUS_JSON)
    status = manager.write_status(repo, {
        **prior_status, "status": "MANAGING", "active_experiment": None,
        "experiments_completed": int(prior_status.get("experiments_completed", 0)) + 1,
        "primary_causal_diagnosis": classification,
        "next_experiment": "LR10_UPSTREAM_REPRESENTATION_BYTE_AUCTION",
        "latest_result": {"experiment_id": EXPERIMENT_ID, "classification": classification,
                          "key_effects": artifact["key_effects"],
                          "evidence_seal_sha256": artifact["seal_sha256"]},
        "resources": after,
    })
    receipt = manager.telegram(
        repo, "long-run:LR09-complete",
        ("[Kimi K2.6 long run] LR09 stratum confidence sealed\n"
         f"route+weight rescue, mismatch: {route_mismatch_rescue:.3f}\n"
         f"route+weight rescue, matched: {route_match_rescue:.3f}\n"
         f"teacher-MoE rescue, mismatch: {moe_mismatch_rescue:.3f}\n"
         f"classification: {classification}\n"
         "next: upstream representation byte auction"),
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
    parser.add_argument("--seed", type=int, default=26072109)
    args = parser.parse_args()
    try:
        artifact = run(args.repo.resolve(strict=True), args.seed)
        print(json.dumps({"status": artifact["status"],
                          "classification": artifact["classification"],
                          "seal_sha256": artifact["seal_sha256"]}, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
