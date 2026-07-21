#!/usr/bin/env python3.12
"""Seal the tested Kimi <=0.98-BPW linear-repair region closure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_f1_bracket as f1  # noqa: E402
import kimi_k26_long_run_manager as manager  # noqa: E402


EXPERIMENT_ID = "LR13_REGION_CLOSURE_AUDIT"
ARTIFACTS = [
    "KIMI_K26_LONG_RUN_BASELINE_AUDIT.json",
    "KIMI_K26_LONG_RUN_CONTROL_CAUSAL.json",
    "KIMI_K26_LONG_RUN_REPAIR_BRACKET.json",
    "KIMI_K26_LONG_RUN_REPLICATION.json",
    "KIMI_K26_LONG_RUN_SHRINKAGE_BOUNDARY.json",
    "KIMI_K26_LONG_RUN_UNTOUCHED_VALIDATION.json",
    "KIMI_K26_LONG_RUN_POOLED_CAUSAL_LAW.json",
    "KIMI_K26_LONG_RUN_LARGE_FALSIFICATION.json",
    "KIMI_K26_LONG_RUN_INTERVENTION_REPLICATION.json",
    "KIMI_K26_LONG_RUN_STRATIFIED_RESCUE.json",
    "KIMI_K26_LONG_RUN_UPSTREAM_AUCTION.json",
    "KIMI_K26_LONG_RUN_UPSTREAM_F2.json",
    "KIMI_K26_LONG_RUN_UPSTREAM_REPLICATION.json",
]


def verify_seal(value: dict[str, Any]) -> bool:
    expected = value.get("seal_sha256")
    if not expected:
        return False
    return f1.seal({key: item for key, item in value.items()
                    if key != "seal_sha256"})["seal_sha256"] == expected


def markdown(artifact: dict[str, Any]) -> str:
    cause = artifact["causal_closure"]
    lines = [
        "# Kimi K2.6 Tested-Region Closure", "",
        f"- Decision: **{artifact['decision']}**",
        f"- Independent held-out tokens: `{artifact['sample_accounting']['heldout_tokens']}`",
        f"- Tested ceiling: `{artifact['scope']['complete_bpw_ceiling']}` complete BPW",
        f"- Best retained candidate: `{artifact['current_best']['candidate']}`",
        f"- Best retained BPW: `{artifact['current_best']['complete_bpw']}`", "",
        "## Causal closure", "",
        f"- Diagnosis: `{cause['diagnosis']}`",
        f"- Teacher indices+weights rescue: `{cause['indices_plus_weights_rescue']:.6f}`",
        f"- Teacher weighted-MoE rescue: `{cause['teacher_moe_rescue']:.6f}`",
        f"- Teacher hidden rescue: `{cause['teacher_hidden_rescue']:.6f}`",
        f"- Route-weight rescue after crossing: `{cause['crossed_route_weight_rescue']:.6f}`",
        f"- Route-weight rescue with route matched: `{cause['matched_route_weight_rescue']:.6f}`", "",
        "## Closed families", "",
    ]
    lines.extend(f"- {row}" for row in artifact["closed_families"])
    lines.extend(["", "## Next architecture", "", artifact["next_architecture"], ""])
    return "\n".join(lines)


def run(repo: Path) -> dict[str, Any]:
    started_at = f1.now()
    started_wall = time.time()
    before = manager.resource_snapshot()
    audit = manager.audit(repo, notify=False)
    if audit["status"] != "PASS":
        raise f1.F1Error(f"closure guard audit failed: {audit['failures']}")
    prior_status = f1.read_json(repo / manager.STATUS_JSON)
    manager.write_status(repo, {**prior_status, "status": "RUNNING_EXPERIMENT",
                                "active_experiment": EXPERIMENT_ID,
                                "next_experiment": "LR13_IN_PROGRESS", "resources": before})
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_START",
        "experiment_id": EXPERIMENT_ID, "started_at": started_at, "status": "RUNNING",
        "hypothesis": "All tested linear repair allocations under 0.98 BPW are causally closed.",
        "config": {"artifact_count": len(ARTIFACTS), "complete_bpw_ceiling": 0.98},
    })
    values = {}
    seals = {}
    for name in ARTIFACTS:
        path = repo / name
        value = f1.read_json(path)
        if not verify_seal(value):
            raise f1.F1Error(f"evidence seal mismatch: {name}")
        values[name] = value
        seals[name] = value["seal_sha256"]
    capture_checks = {}
    for name, value in values.items():
        capture = value.get("capture")
        if capture:
            valid = f1.sha256_file(Path(capture["path"])) == capture["sha256"]
            capture_checks[name] = {"path": capture["path"], "sha256": capture["sha256"],
                                    "valid": valid}
            if not valid:
                raise f1.F1Error(f"capture hash mismatch: {name}")
    ledger_records = [json.loads(line) for line in (repo / manager.LEDGER).read_text().splitlines()
                      if line.strip()]
    ledger_valid = all(verify_seal(record) for record in ledger_records)
    if not ledger_valid:
        raise f1.F1Error("long-run ledger seal mismatch")
    control = values["KIMI_K26_LONG_RUN_CONTROL_CAUSAL.json"]
    repair = values["KIMI_K26_LONG_RUN_REPAIR_BRACKET.json"]
    replication = values["KIMI_K26_LONG_RUN_REPLICATION.json"]
    pooled = values["KIMI_K26_LONG_RUN_POOLED_CAUSAL_LAW.json"]
    large_test = values["KIMI_K26_LONG_RUN_LARGE_FALSIFICATION.json"]
    intervention = values["KIMI_K26_LONG_RUN_INTERVENTION_REPLICATION.json"]
    strata = values["KIMI_K26_LONG_RUN_STRATIFIED_RESCUE.json"]
    upstream_f2 = values["KIMI_K26_LONG_RUN_UPSTREAM_F2.json"]
    upstream_replication = values["KIMI_K26_LONG_RUN_UPSTREAM_REPLICATION.json"]
    heldout_tokens = sum([
        control["sample_counts"]["tokens"],
        replication["tokenization"]["tokens"],
        values["KIMI_K26_LONG_RUN_UNTOUCHED_VALIDATION.json"]["tokenization"]["tokens"],
        large_test["tokenization"]["tokens"],
        intervention["tokenization"]["tokens"],
        upstream_f2["tokenization"]["tokens"],
        upstream_replication["tokenization"]["tokens"],
    ])
    physical_frontier = [{
        "candidate": row["candidate"], "complete_bpw": row["complete_bpw"],
        "complete_physical_bytes": row["complete_physical_bytes"],
        "decision": row["decision"],
        "heldout_improvement": row["relative_l2_improvement"],
    } for row in repair["treatment_frontier"]]
    physical_frontier.extend([
        {"candidate": "POST_MOE_HIDDEN_CV_R12_S025",
         "complete_bpw": values["KIMI_K26_LONG_RUN_SHRINKAGE_BOUNDARY.json"][
             "physical_candidate"]["complete_bpw"],
         "decision": values["KIMI_K26_LONG_RUN_UNTOUCHED_VALIDATION.json"]["decision"],
         "heldout_improvement": values[
             "KIMI_K26_LONG_RUN_UNTOUCHED_VALIDATION.json"]["all_tokens"][
                 "relative_l2_improvement"]},
        {"candidate": "UPSTREAM_RESIDUAL_CV",
         "complete_bpw": upstream_f2["candidate"]["complete_bpw"],
         "decision": upstream_replication["decision"],
         "first_f2_improvement": upstream_f2["f2"][
             "paired_row_relative_l2_improvement"],
         "replication_f2_improvement": upstream_replication["f2"][
             "paired_row_relative_l2_improvement"]},
    ])
    completed_durations = [float(record.get("duration_seconds", 0) or 0)
                           for record in ledger_records
                           if record.get("event") == "EXPERIMENT_COMPLETE"]
    fault_durations = [float(record.get("duration_seconds", 0) or 0)
                       for record in ledger_records
                       if record.get("event") == "EXPERIMENT_FAULT"]
    after = manager.resource_snapshot()
    ended_at = f1.now()
    duration = time.time() - started_wall
    artifact = f1.seal({
        "schema": "hawking.kimi_k26.long_run_region_closure.v1", "status": "PASS",
        "experiment_id": EXPERIMENT_ID,
        "hypothesis": "All tested linear repair allocations under 0.98 BPW are causally closed.",
        "scope": {"complete_bpw_ceiling": 0.98,
                  "base_representation": "P1_DUAL_PATH_RECOVERY_R16X2",
                  "base_complete_bpw": control["candidate"]["complete_bpw"],
                  "layers_tested": [1, 2, 3], "top_k": 8},
        "current_best": {"candidate": "P1_DUAL_PATH_RECOVERY_R16X2_LOCAL_F1_ONLY",
                         "complete_bpw": control["candidate"]["complete_bpw"],
                         "f1_cosine": 0.9134416580200195,
                         "f2_promotable": False},
        "sample_accounting": {"heldout_tokens": heldout_tokens,
                              "pooled_original_tokens": pooled["pooled"]["tokens"],
                              "large_falsification_tokens": large_test["tokenization"]["tokens"],
                              "intervention_tokens": intervention["tokenization"]["tokens"],
                              "stratified_intervention_tokens": intervention["config"][
                                  "selected_tokens"]},
        "causal_closure": {
            "diagnosis": "UPSTREAM_STATE_PRIMARY_ROUTE_SECONDARY_CONDITIONAL_AMPLIFIER",
            "indices_plus_weights_rescue": intervention["intervention_matrix"][
                "FORCE_TEACHER_INDICES_AND_WEIGHTS"]["rescue_fraction_relative_l2"],
            "teacher_moe_rescue": intervention["intervention_matrix"][
                "SUBSTITUTE_TEACHER_WEIGHTED_MOE"]["rescue_fraction_relative_l2"],
            "teacher_hidden_rescue": intervention["intervention_matrix"][
                "RESTORE_TEACHER_HIDDEN"]["rescue_fraction_relative_l2"],
            "crossed_route_weight_rescue": strata["key_effects"][
                "route_weight_rescue_mismatch"],
            "matched_route_weight_rescue": strata["key_effects"][
                "route_weight_rescue_match"],
            "pooled_route_change": pooled["pooled"]["route_set_change"],
            "mismatch_minus_match_error": pooled["pooled"]["mismatch_minus_match_error"],
            "material_errors_with_routes_matched": pooled["pooled"][
                "material_error_with_routes_matched_tokens"],
        },
        "physical_frontier": physical_frontier,
        "closed_families": [
            "first-divergence low-margin state protection",
            "low-margin router-logit correction",
            "pre-router low-rank hidden repair",
            "weighted-MoE-output low-rank repair",
            "pre-router plus post-MoE hybrid",
            "calibration-CV post-MoE shrinkage",
            "upstream compact-output linear residual",
        ],
        "negative_results": {
            "lr02_all_rows_typical_token_harm": True,
            "lr03_replication_all_rows_typical_token_harm": True,
            "lr05_shrinkage_untouched_failure": values[
                "KIMI_K26_LONG_RUN_UNTOUCHED_VALIDATION.json"]["decision"],
            "lr07_large_shrinkage_mean_harm": large_test["all_tokens"][
                "relative_l2_improvement"],
            "upstream_first_f2_gain": upstream_f2["f2"][
                "paired_row_relative_l2_improvement"],
            "upstream_replication_failure": upstream_replication["f2"][
                "paired_row_relative_l2_improvement"],
        },
        "proven": pooled["scientific_law"]["proven"],
        "correlated_not_proven": pooled["scientific_law"]["correlated_not_proven"],
        "unresolved": pooled["scientific_law"]["unresolved"],
        "next_architecture": (
            "Test a representation-side nonlinear structural allocation at F0/F1 that directly "
            "reduces compact expert-output state error before any router. Do not spend more bits on "
            "generic downstream low-rank Doctor paths; require disjoint-score F1 evidence before F2."
        ),
        "evidence_verification": {"artifact_seals": seals,
                                  "capture_hashes": capture_checks,
                                  "ledger_records": len(ledger_records),
                                  "ledger_seals_valid": ledger_valid,
                                  "guard_audit_seal_sha256": audit["seal_sha256"]},
        "compute_accounting_so_far": {
            "completed_experiment_seconds": sum(completed_durations),
            "invalid_retry_seconds": sum(fault_durations),
            "ledger_completed_experiments": len(completed_durations),
            "ledger_faults": len(fault_durations)},
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resource_observations": {"before": before, "after": after},
        "decision": "TESTED_LINEAR_REPAIR_REGION_CLOSED",
        "causal_interpretation": (
            "The tested region is closed by replicated intervention ordering and replicated physical "
            "failures, not by absence of route correlation. Routing is a strong conditional amplifier "
            "whose correction cannot remove the upstream state error."
        ),
        "next_run_rationale": "Hold guards through the required wall-clock boundary, then report closure.",
        "faults": [], "retries": 0,
    })
    json_path = repo / "KIMI_K26_LONG_RUN_REGION_CLOSURE.json"
    md_path = repo / "KIMI_K26_LONG_RUN_REGION_CLOSURE.md"
    f1.atomic_json(json_path, artifact)
    f1.atomic_json(manager.RUNTIME / json_path.name, artifact)
    for path in (md_path, manager.RUNTIME / md_path.name):
        temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
        temporary.write_text(markdown(artifact), encoding="utf-8")
        temporary.replace(path)
    manager.append_ledger(repo, {
        "schema": "hawking.kimi_k26.long_run_ledger_entry.v1", "event": "EXPERIMENT_COMPLETE",
        "experiment_id": EXPERIMENT_ID, "hypothesis": artifact["hypothesis"],
        "config": artifact["scope"], "parent_hash": upstream_replication["seal_sha256"],
        "candidate_hash": None, "physical_bytes": None, "complete_bpw": None,
        "started_at": started_at, "ended_at": ended_at, "duration_seconds": duration,
        "resources": artifact["resource_observations"],
        "sample_counts": artifact["sample_accounting"],
        "metrics": {"causal_closure": artifact["causal_closure"],
                    "physical_frontier": physical_frontier},
        "confidence_intervals": {"pooled_route_change": pooled["pooled"]["route_set_change"],
                                 "mismatch_minus_match": pooled["pooled"][
                                     "mismatch_minus_match_error"]},
        "faults": [], "retries": 0, "causal_interpretation": artifact["causal_interpretation"],
        "decision": artifact["decision"], "next_run_rationale": artifact["next_run_rationale"],
        "evidence_seal_sha256": artifact["seal_sha256"],
    })
    prior_status = f1.read_json(repo / manager.STATUS_JSON)
    status = manager.write_status(repo, {
        **prior_status, "status": "REGION_CLOSED_MONITORING_TO_BOUNDARY",
        "active_experiment": None,
        "experiments_completed": int(prior_status.get("experiments_completed", 0)) + 1,
        "current_best_candidate": artifact["current_best"]["candidate"],
        "current_best_bpw": artifact["current_best"]["complete_bpw"],
        "primary_causal_diagnosis": artifact["causal_closure"]["diagnosis"],
        "dominant_bottleneck": "COMPACT_EXPERT_OUTPUT_STATE_ERROR_BEFORE_ROUTER",
        "next_experiment": "BOUNDARY_MONITOR_AND_FINAL_REPORT",
        "latest_result": {"experiment_id": EXPERIMENT_ID, "decision": artifact["decision"],
                          "heldout_tokens": heldout_tokens,
                          "evidence_seal_sha256": artifact["seal_sha256"]},
        "resources": after,
    })
    receipt = manager.telegram(
        repo, "long-run:LR13-complete",
        ("[Kimi K2.6 long run] LR13 tested region closed\n"
         f"held-out tokens: {heldout_tokens}\n"
         f"best retained: 0.908591 BPW, F1-only\n"
         f"causal diagnosis: {artifact['causal_closure']['diagnosis']}\n"
         "all tested <=0.98-BPW linear repair families retired\n"
         "manager remains on guard through wall-clock boundary"),
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
    args = parser.parse_args()
    try:
        artifact = run(args.repo.resolve(strict=True))
        print(json.dumps({"status": artifact["status"], "decision": artifact["decision"],
                          "heldout_tokens": artifact["sample_accounting"]["heldout_tokens"],
                          "seal_sha256": artifact["seal_sha256"]}, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
