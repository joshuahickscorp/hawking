"""Fail-closed protocol for selecting Hawking's final manager.

The Ascension lifecycle already reserves a protected ``MANAGER_TOURNAMENT``
state.  This module makes its intended meaning explicit: it is a comparison of
two independently-qualified Gravity managers as operating intelligences, not a
throughput shoot-out or a self-assessment.  It creates configuration evidence
only; protected execution and the lifecycle controller remain responsible for
admitting scored receipts and selecting a winner.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping

from lab.receipts import seal


SCHEMA = "hawking.ascension.final_manager_tournament_protocol.v1"

EVALUATION_MODES: tuple[str, ...] = (
    "SOLO_MANAGER",
    "MANAGER_AS_ORCHESTRATOR",
)

CANDIDATE_ARTIFACTS: tuple[str, ...] = (
    "Qwen30-Gravity-Manager-Artifact",
    "Qwen80-Gravity-Manager-Artifact",
)

HARD_GATES: tuple[str, ...] = (
    "source_identity",
    "complete_bpw_at_most_1_5",
    "tg3_base_true_tps",
    "fallback_zero",
    "real_metal",
    "coherent_real_hcli",
    "capability_retention",
    "agent_os_production_path",
    "residency",
    "restart_recovery",
    "storage_rollback",
    "security_effect_boundaries",
    "manager_contract",
)

TASK_FAMILIES: tuple[str, ...] = (
    "repository_architecture",
    "kernel_metal_diagnosis",
    "gravity_research",
    "complete_token_performance",
    "bug_repair",
    "tool_use",
    "delegation",
    "helper_management",
    "context_memory",
    "interrupt_recovery",
    "adversarial_evidence",
    "benchmark_honesty",
    "storage_resource_campaign",
    "security_effect_boundaries",
    "release_integration",
)

LONG_HORIZON_CAMPAIGNS: tuple[str, ...] = (
    "six_stage_kernel_optimization",
    "multi_subsystem_hcli_repair",
    "exact_model_family_qualification",
    "storage_pressure_model_acquisition",
    "agent_os_concurrency_optimization",
    "gravity_representation_tournament",
)

FAIRNESS_ENVELOPE: tuple[str, ...] = (
    "gpu_opportunity",
    "cpu_budget",
    "wall_clock_or_experiment_budget",
    "logical_agent_budget",
    "context_budget",
    "tool_access",
    "helper_model_access",
    "retry_policy",
    "storage_budget",
    "hidden_evidence",
    "temperature_sampling_policy",
)

PRIMARY_METRICS: tuple[str, ...] = (
    "verified_tasks_per_hour",
    "first_pass_success",
    "repair_cycles",
    "failed_experiments",
    "tool_errors",
    "worker_rejection_accuracy",
    "human_intervention_count",
    "regressions_introduced",
    "rollback_events",
    "resource_use",
)

PERFORMANCE_METRICS: tuple[str, ...] = (
    "base_true_tps",
    "latency_p50",
    "latency_p95",
    "latency_p99",
    "ttft",
    "hcli_overhead",
    "resident_memory",
    "active_bytes_per_token",
    "state_kv_traffic",
    "energy_resource_efficiency",
    "multi_session_throughput",
    "weight_reuse",
)

MANAGER_INTELLIGENCE_METRICS: tuple[str, ...] = (
    "architecture_comprehension",
    "root_cause_diagnosis",
    "planning_quality",
    "experiment_quality",
    "scientific_reasoning",
    "kernel_reasoning",
    "representation_reasoning",
    "tool_competence",
    "delegation",
    "recovery",
    "context_retention",
    "evidence_discipline",
    "self_correction",
    "release_discipline",
)

ORCHESTRATION_METRICS: tuple[str, ...] = (
    "worker_utilization",
    "duplicate_work",
    "idle_lanes",
    "bad_delegation",
    "integration_failures",
    "merge_conflicts",
    "review_quality",
    "critical_path_scheduling",
    "verified_tasks_per_hour",
)

FAILURE_RECOVERY_INJECTIONS: tuple[str, ...] = (
    "bad_worker_patch",
    "failed_build",
    "false_benchmark",
    "stale_receipt",
    "killed_process",
    "disk_pressure",
    "kv_corruption",
    "failed_metal_parity",
    "incorrect_architectural_assumption",
)

FINAL_REPORT_FIELDS: tuple[str, ...] = (
    "bpw",
    "tg_rung",
    "base_true_tps",
    "p99",
    "memory",
    "verified_tasks_per_hour",
    "solo_capability",
    "orchestrated_capability",
    "repository_architecture_tasks",
    "kernel_tasks",
    "gravity_tasks",
    "tool_reliability",
    "delegation",
    "context",
    "restart",
    "failure_recovery",
    "adversarial_review",
    "resource_efficiency",
    "long_horizon_completion",
    "human_interventions",
    "hard_gate_status",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _required_rows(values: tuple[str, ...]) -> dict[str, dict[str, bool]]:
    return {value: {"required": True} for value in values}


def build_final_manager_tournament_protocol() -> dict[str, Any]:
    """Return the fixed, non-executing final-manager selection protocol."""

    body: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "PREPARED_FINAL_MANAGER_SELECTION_PROTOCOL_NOT_EXECUTED",
            "purpose": (
                "Select the Hawking operating intelligence that produces the most "
                "correct, verified, recoverable engineering progress per unit of "
                "machine resource while preserving the Ascension constitution."
            ),
            "ascent_boundary": {
                "development_race_is_not_final_selection": True,
                "candidate_specific_optimization_allowed_only_before_freeze": True,
                "freeze_requires_both_complete_manager_qualification_floors": True,
                "post_freeze_candidate_specific_change_requires_failed_hard_gate_repair": True,
                "repair_requires_replay_of_all_affected_scored_tests": True,
            },
            "candidates": {
                "fixed_order": list(CANDIDATE_ARTIFACTS),
                "both_must_meet_complete_manager_qualification_floor_before_freeze": True,
            },
            "freeze_contract": {
                "both_candidates_frozen_before_scored_execution": True,
                "candidate_specific_optimization_disabled_during_scored_execution": True,
                "hard_gate_repair_requires_affected_test_replay": True,
            },
            "evaluation_modes": {
                "required": list(EVALUATION_MODES),
                "SOLO_MANAGER": {
                    "measures": "raw_model_manager_capability",
                    "no_helper_advantage": True,
                    "each_candidate_measured": True,
                },
                "MANAGER_AS_ORCHESTRATOR": {
                    "measures": "organization_level_verified_engineering_progress",
                    "same_hawking_agent_os": True,
                    "symmetric_helper_model_infrastructure": True,
                    "no_hidden_information_or_worker_advantage": True,
                    "each_candidate_measured": True,
                },
            },
            "hard_gates": {
                "conjunctive": True,
                "no_score_can_compensate_for_failure": True,
                "each_gate_required_for_each_candidate": True,
                "required": _required_rows(HARD_GATES),
            },
            "protected_task_corpus": {
                "real_hawking_work_only": True,
                "blind_tasks_required": True,
                "hidden_membership_frozen_before_scored_execution": True,
                "required_families": list(TASK_FAMILIES),
            },
            "long_horizon_campaigns": {
                "required": list(LONG_HORIZON_CAMPAIGNS),
                "measure_initial_plan_adaptation_resets_repeated_mistakes_branches_goal_fidelity_and_completion": True,
            },
            "adversarial_review": {
                "required_after_each_candidate_task": True,
                "opposing_candidate_is_read_only_red_team_reviewer": True,
                "reviewer_may_not_modify_candidate_artifact": True,
                "protected_verifier_adjudicates_challenges": True,
                "measure_genuine_defects_and_false_objections": True,
            },
            "fairness_envelope": {
                "record_every_asymmetry": True,
                "equalize": _required_rows(FAIRNESS_ENVELOPE),
            },
            "scorecards": {
                "primary_metric": "verified_tasks_per_hour",
                "primary_metrics": list(PRIMARY_METRICS),
                "performance_metrics": list(PERFORMANCE_METRICS),
                "manager_intelligence_metrics": list(MANAGER_INTELLIGENCE_METRICS),
                "orchestration_metrics": list(ORCHESTRATION_METRICS),
                "failure_recovery_injections": list(FAILURE_RECOVERY_INJECTIONS),
                "pareto_frontier_required_before_selection": True,
                "do_not_collapse_to_scalar_before_complete_evidence_matrix": True,
            },
            "protected_selection": {
                "candidate_self_assessments_are_evidence_only": True,
                "candidates_cannot_self_grade": True,
                "candidates_cannot_change_weights_or_hidden_tests": True,
                "candidates_cannot_promote_self_or_invalidate_opponent": True,
                "only_protected_controller_or_human_may_select_manager": True,
            },
            "winner_freeze_and_alternate": {
                "seal_winner_source_artifact_runtime_kernel_agent_os_context_kv_benchmarks_capability_tournament_and_rollback": True,
                "seal_loser_before_any_evictability": True,
                "cold_store_hash_verify_restore_test_one_command_recovery_required": True,
                "do_not_delete_loser_before_restore_proof": True,
            },
            "final_report": {
                "side_by_side_candidates_required": True,
                "required_fields": list(FINAL_REPORT_FIELDS),
                "must_state_winner_loser_decisive_evidence_tradeoffs_restore_path": True,
            },
            "claim_boundary": {
                "does_not_execute_tournament": True,
                "does_not_score_candidates": True,
                "does_not_choose_winner": True,
                "does_not_activate_a_server_or_sandbox": True,
                "does_not_relax_tg3_or_other_hard_gates": True,
            },
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return seal(
        {
            **body,
            "protocol_identity_sha256": hashlib.sha256(canonical).hexdigest(),
            "recorded_at": _utc_now(),
        }
    )


def _require_exact_keys(
    issues: list[str],
    value: object,
    expected: tuple[str, ...],
    label: str,
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        issues.append(f"tournament requires {label} mapping")
        return None
    observed = tuple(str(item) for item in value.keys())
    if set(observed) != set(expected):
        issues.append(f"tournament {label} keys must exactly cover {', '.join(expected)}")
    return value


def validate_final_manager_tournament_result(document: Mapping[str, Any]) -> list[str]:
    """Return precise blockers for a prospective final tournament receipt.

    This validates the *protected result contract*, not subjective scores.  It
    deliberately does not decide which candidate wins.
    """

    issues: list[str] = []
    if document.get("final_manager_protocol_schema") != SCHEMA:
        issues.append("tournament must bind the final-manager protocol schema")
    protocol_seal = document.get("final_manager_protocol_seal_sha256")
    if not isinstance(protocol_seal, str) or len(protocol_seal) != 64:
        issues.append("tournament must bind a final-manager protocol seal")
    protocol_identity = document.get("final_manager_protocol_identity_sha256")
    if not isinstance(protocol_identity, str) or len(protocol_identity) != 64:
        issues.append("tournament must bind a stable final-manager protocol identity")

    freeze = _require_exact_keys(
        issues, document.get("candidate_freeze"), CANDIDATE_ARTIFACTS, "candidate freeze"
    )
    if freeze is not None:
        for candidate in CANDIDATE_ARTIFACTS:
            row = freeze.get(candidate)
            if not isinstance(row, Mapping):
                issues.append(f"tournament requires freeze evidence for {candidate}")
                continue
            for key in (
                "complete_manager_floor_qualified",
                "frozen_before_scored_execution",
                "candidate_specific_optimization_disabled",
                "affected_tests_replayed_after_any_repair",
            ):
                if row.get(key) is not True:
                    issues.append(f"tournament freeze requires {candidate}/{key}")

    modes = _require_exact_keys(
        issues, document.get("evaluation_modes"), EVALUATION_MODES, "evaluation_modes"
    )
    if modes is not None:
        for mode in EVALUATION_MODES:
            result = modes.get(mode)
            if not isinstance(result, Mapping) or result.get("measured") is not True:
                issues.append(f"tournament requires measured {mode}")
                continue
            candidates = _require_exact_keys(
                issues,
                result.get("candidate_results"),
                CANDIDATE_ARTIFACTS,
                f"{mode} candidate results",
            )
            if candidates is not None:
                for candidate in CANDIDATE_ARTIFACTS:
                    row = candidates.get(candidate)
                    if not isinstance(row, Mapping) or row.get("measured") is not True:
                        issues.append(f"tournament requires measured {mode} result for {candidate}")
        orchestrated = modes.get("MANAGER_AS_ORCHESTRATOR")
        if isinstance(orchestrated, Mapping):
            for key in (
                "same_hawking_agent_os",
                "symmetric_helper_model_infrastructure",
                "identical_resource_envelope",
            ):
                if orchestrated.get(key) is not True:
                    issues.append(f"orchestrated mode requires {key}")

    hard_gates = document.get("hard_gate_results")
    if not isinstance(hard_gates, Mapping):
        issues.append("tournament requires hard_gate_results")
    else:
        if hard_gates.get("conjunctive") is not True or hard_gates.get("no_score_compensation") is not True:
            issues.append("tournament hard gates must remain conjunctive without score compensation")
        by_candidate = _require_exact_keys(
            issues, hard_gates.get("by_candidate"), CANDIDATE_ARTIFACTS, "hard gate candidates"
        )
        if by_candidate is not None:
            for candidate in CANDIDATE_ARTIFACTS:
                results = _require_exact_keys(
                    issues,
                    by_candidate.get(candidate),
                    HARD_GATES,
                    f"hard gate results for {candidate}",
                )
                if results is not None:
                    for gate in HARD_GATES:
                        row = results.get(gate)
                        if not isinstance(row, Mapping) or row.get("passed") is not True:
                            issues.append(f"tournament hard gate failed or missing: {candidate}/{gate}")

    task_corpus = document.get("protected_task_corpus")
    if not isinstance(task_corpus, Mapping):
        issues.append("tournament requires protected_task_corpus")
    else:
        if task_corpus.get("blind_tasks_frozen") is not True:
            issues.append("tournament requires frozen blind tasks")
        if tuple(task_corpus.get("families") or ()) != TASK_FAMILIES:
            issues.append("tournament task corpus families drift from the protected contract")

    campaigns = document.get("long_horizon_campaigns")
    if not isinstance(campaigns, Mapping) or tuple(campaigns.get("completed") or ()) != LONG_HORIZON_CAMPAIGNS:
        issues.append("tournament requires every protected long-horizon campaign")

    fairness = document.get("fairness_envelope")
    if not isinstance(fairness, Mapping):
        issues.append("tournament requires fairness_envelope")
    else:
        equalized = _require_exact_keys(issues, fairness.get("equalized"), FAIRNESS_ENVELOPE, "fairness envelope")
        if equalized is not None:
            for field in FAIRNESS_ENVELOPE:
                if equalized.get(field) is not True:
                    issues.append(f"tournament fairness envelope is not equalized: {field}")
        if fairness.get("asymmetries_recorded") is not True:
            issues.append("tournament must record every resource asymmetry")

    review = document.get("adversarial_review")
    if not isinstance(review, Mapping):
        issues.append("tournament requires adversarial_review")
    else:
        for key in ("both_directions_measured", "protected_verifier_adjudicated", "reviewers_read_only"):
            if review.get(key) is not True:
                issues.append(f"tournament adversarial review requires {key}")

    scorecards = document.get("scorecards")
    if not isinstance(scorecards, Mapping):
        issues.append("tournament requires scorecards")
    else:
        if scorecards.get("primary_metric") != "verified_tasks_per_hour":
            issues.append("tournament primary metric must be verified_tasks_per_hour")
        for label, expected in (
            ("primary_metrics", PRIMARY_METRICS),
            ("performance_metrics", PERFORMANCE_METRICS),
            ("manager_intelligence_metrics", MANAGER_INTELLIGENCE_METRICS),
            ("orchestration_metrics", ORCHESTRATION_METRICS),
            ("failure_recovery_injections", FAILURE_RECOVERY_INJECTIONS),
        ):
            if tuple(scorecards.get(label) or ()) != expected:
                issues.append(f"tournament {label} drift from the protected protocol")
        if scorecards.get("pareto_frontier_complete") is not True:
            issues.append("tournament requires completed Pareto frontier before selection")
        candidate_scorecards = _require_exact_keys(
            issues,
            scorecards.get("by_candidate"),
            CANDIDATE_ARTIFACTS,
            "scorecards by candidate",
        )
        if candidate_scorecards is not None:
            for candidate in CANDIDATE_ARTIFACTS:
                row = candidate_scorecards.get(candidate)
                if not isinstance(row, Mapping) or row.get("all_required_metrics_measured") is not True:
                    issues.append(f"tournament scorecard is incomplete for {candidate}")

    selection = document.get("protected_selection")
    if not isinstance(selection, Mapping):
        issues.append("tournament requires protected_selection")
    else:
        for key in ("no_candidate_self_grading", "no_candidate_rule_change", "protected_authority_selected_winner"):
            if selection.get(key) is not True:
                issues.append(f"tournament protected selection requires {key}")

    report = document.get("final_report")
    if not isinstance(report, Mapping):
        issues.append("tournament requires final_report")
    else:
        if tuple(report.get("fields") or ()) != FINAL_REPORT_FIELDS:
            issues.append("tournament final report fields drift from the protected protocol")
        if report.get("side_by_side_complete") is not True or report.get("winner_loser_tradeoffs_restore_path") is not True:
            issues.append("tournament final report is incomplete")
    return issues


__all__ = [
    "CANDIDATE_ARTIFACTS",
    "EVALUATION_MODES",
    "FAIRNESS_ENVELOPE",
    "FAILURE_RECOVERY_INJECTIONS",
    "FINAL_REPORT_FIELDS",
    "HARD_GATES",
    "LONG_HORIZON_CAMPAIGNS",
    "MANAGER_INTELLIGENCE_METRICS",
    "ORCHESTRATION_METRICS",
    "PERFORMANCE_METRICS",
    "PRIMARY_METRICS",
    "SCHEMA",
    "TASK_FAMILIES",
    "build_final_manager_tournament_protocol",
    "validate_final_manager_tournament_result",
]
