"""Focused fail-closed tests for the final Hawking manager-selection contract."""
from __future__ import annotations

from lab.operators.ascension_manager_tournament_protocol import (
    CANDIDATE_ARTIFACTS,
    EVALUATION_MODES,
    FAIRNESS_ENVELOPE,
    FAILURE_RECOVERY_INJECTIONS,
    FINAL_REPORT_FIELDS,
    HARD_GATES,
    LONG_HORIZON_CAMPAIGNS,
    MANAGER_INTELLIGENCE_METRICS,
    ORCHESTRATION_METRICS,
    PERFORMANCE_METRICS,
    PRIMARY_METRICS,
    SCHEMA,
    TASK_FAMILIES,
    build_final_manager_tournament_protocol,
    validate_final_manager_tournament_result,
)
from lab.receipts import verify


def _complete_result() -> dict[str, object]:
    protocol = build_final_manager_tournament_protocol()
    return {
        "final_manager_protocol_schema": SCHEMA,
        "final_manager_protocol_seal_sha256": protocol["seal_sha256"],
        "final_manager_protocol_identity_sha256": protocol["protocol_identity_sha256"],
        "candidate_freeze": {
            candidate: {
                "complete_manager_floor_qualified": True,
                "frozen_before_scored_execution": True,
                "candidate_specific_optimization_disabled": True,
                "affected_tests_replayed_after_any_repair": True,
            }
            for candidate in CANDIDATE_ARTIFACTS
        },
        "evaluation_modes": {
            "SOLO_MANAGER": {
                "measured": True,
                "candidate_results": {candidate: {"measured": True} for candidate in CANDIDATE_ARTIFACTS},
            },
            "MANAGER_AS_ORCHESTRATOR": {
                "measured": True,
                "same_hawking_agent_os": True,
                "symmetric_helper_model_infrastructure": True,
                "identical_resource_envelope": True,
                "candidate_results": {candidate: {"measured": True} for candidate in CANDIDATE_ARTIFACTS},
            },
        },
        "hard_gate_results": {
            "conjunctive": True,
            "no_score_compensation": True,
            "by_candidate": {
                candidate: {gate: {"passed": True} for gate in HARD_GATES}
                for candidate in CANDIDATE_ARTIFACTS
            },
        },
        "protected_task_corpus": {"blind_tasks_frozen": True, "families": list(TASK_FAMILIES)},
        "long_horizon_campaigns": {"completed": list(LONG_HORIZON_CAMPAIGNS)},
        "fairness_envelope": {
            "equalized": {field: True for field in FAIRNESS_ENVELOPE},
            "asymmetries_recorded": True,
        },
        "adversarial_review": {
            "both_directions_measured": True,
            "protected_verifier_adjudicated": True,
            "reviewers_read_only": True,
        },
        "scorecards": {
            "primary_metric": "verified_tasks_per_hour",
            "primary_metrics": list(PRIMARY_METRICS),
            "performance_metrics": list(PERFORMANCE_METRICS),
            "manager_intelligence_metrics": list(MANAGER_INTELLIGENCE_METRICS),
            "orchestration_metrics": list(ORCHESTRATION_METRICS),
            "failure_recovery_injections": list(FAILURE_RECOVERY_INJECTIONS),
            "pareto_frontier_complete": True,
            "by_candidate": {
                candidate: {"all_required_metrics_measured": True}
                for candidate in CANDIDATE_ARTIFACTS
            },
        },
        "protected_selection": {
            "no_candidate_self_grading": True,
            "no_candidate_rule_change": True,
            "protected_authority_selected_winner": True,
        },
        "final_report": {
            "fields": list(FINAL_REPORT_FIELDS),
            "side_by_side_complete": True,
            "winner_loser_tradeoffs_restore_path": True,
        },
    }


def test_protocol_is_sealed_and_requires_both_evaluation_modes() -> None:
    protocol = build_final_manager_tournament_protocol()
    verify(protocol, label="final manager tournament protocol")
    assert protocol["protocol_identity_sha256"] == build_final_manager_tournament_protocol()["protocol_identity_sha256"]
    assert protocol["evaluation_modes"]["required"] == list(EVALUATION_MODES)
    assert protocol["hard_gates"]["conjunctive"] is True
    assert protocol["protected_selection"]["candidates_cannot_self_grade"] is True


def test_complete_result_contract_is_accepted_without_selecting_a_winner() -> None:
    assert validate_final_manager_tournament_result(_complete_result()) == []


def test_protocol_refuses_score_compensation_for_a_failed_hard_gate() -> None:
    result = _complete_result()
    hard_gates = result["hard_gate_results"]
    assert isinstance(hard_gates, dict)
    by_candidate = hard_gates["by_candidate"]
    assert isinstance(by_candidate, dict)
    results = by_candidate[CANDIDATE_ARTIFACTS[0]]
    assert isinstance(results, dict)
    results["fallback_zero"] = {"passed": False}
    issues = validate_final_manager_tournament_result(result)
    assert any("fallback_zero" in issue for issue in issues)


def test_protocol_refuses_asymmetric_orchestrator_resources_and_blind_task_drift() -> None:
    result = _complete_result()
    modes = result["evaluation_modes"]
    assert isinstance(modes, dict)
    orchestrated = modes["MANAGER_AS_ORCHESTRATOR"]
    assert isinstance(orchestrated, dict)
    orchestrated["identical_resource_envelope"] = False
    corpus = result["protected_task_corpus"]
    assert isinstance(corpus, dict)
    corpus["families"] = list(TASK_FAMILIES[:-1])
    issues = validate_final_manager_tournament_result(result)
    assert any("identical_resource_envelope" in issue for issue in issues)
    assert any("task corpus families" in issue for issue in issues)
