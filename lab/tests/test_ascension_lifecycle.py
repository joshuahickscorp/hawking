"""Fail-closed tests for the full Bible §17 V3 lifecycle controller."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from lab.operators.ascension_foundation_contracts import (
    AGENT_OS_PERFORMANCE_GATES,
    AGENT_ROLES,
    AGENT_SCHEDULER_CAPABILITIES,
    CONTEXT_COMPILER_INPUTS,
    CONTEXT_COMPILER_PROPERTIES,
    ENERGY_EVIDENCE_OUTPUTS,
    GROK_LANE_CLASSES,
    GROK_LANE_CONTRACT_FIELDS,
    GROK_RESOURCE_CLASSES,
    GROK_SCHEDULER_RULES,
    KV_STATE_CAPABILITIES,
    PRESSURE_MODES,
    RESOURCE_TELEMETRY_FIELDS,
    SCHEDULER_SELECTION_OBJECTIVES,
    TASK_SCHEDULER_FIELDS,
)
from lab.operators.ascension_kernel_registry import (
    ARCHITECTURE_FINGERPRINT_FIELDS,
    FAMILY_PLUGINS,
    GRAVITY_COMPONENTS,
    MODEL_PROGRAM_KEY_FIELDS,
    REPRESENTATION_TOURNAMENT_CLASSES,
    SHARED_PRIMITIVES,
)
from lab.operators.ascension_lifecycle import (
    CANONICAL_STATES,
    EXACT_CONTINUATION_OUTPUTS,
    FAMILY_RULES,
    LifecyclePaths,
    MANAGER_CANDIDATE_ORDER,
    MANAGER_KERNEL_OPERATIONAL_TPS_FLOOR,
    MODEL_30B,
    QWEN30_GRAVITY_MANAGER_ARTIFACT,
    TOURNAMENT_CANDIDATE_ORDER,
    STAGE_SPECS,
    TOURNAMENT_DIMENSIONS,
    arm_tournament,
    audit_bible,
    attest_human_adoption,
    evaluate_artifact,
    evaluate_lifecycle,
)
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
    SCHEMA as FINAL_MANAGER_PROTOCOL_SCHEMA,
    TASK_FAMILIES,
    build_final_manager_tournament_protocol,
)
from lab.receipts import seal, verify


def _rule(artifact_id: str):
    for stage in STAGE_SPECS:
        for rule in stage.artifacts:
            if rule.artifact_id == artifact_id:
                return rule
    raise AssertionError(f"unknown artifact {artifact_id}")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base(rule, *, bible_sha256: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": "hawking.ascension.lifecycle_test_evidence.v1",
        "artifact_id": rule.artifact_id,
        "status": "CONTROLLER_CERTIFIED",
        "certified_by": "protected_controller",
        "evidence_basis": "direct_measurement",
    }
    if rule.model_id:
        body.update(
            {
                "model_id": rule.model_id,
                "source_bound": True,
                "official_high_precision_source": True,
                "source_hash": "a" * 64,
            }
        )
    elif rule.source_bound:
        body.update(
            {
                "source_bound": True,
                "official_high_precision_source": True,
                "source_hash": "a" * 64,
            }
        )
    if rule.check == "adoption":
        body.update(
            {
                "adopted_as_sole_canonical_programme": True,
                "bible_sha256": bible_sha256,
            }
        )
    elif rule.check == "authority_freeze":
        body.update(
            {
                "hidden_memberships_frozen": True,
                "deletion_policy_frozen": True,
                "rollback_policy_frozen": True,
            }
        )
    elif rule.check == "build_fabric":
        body.update(
            {
                "isolated_worktree_contracts": True,
                "report_only_model_authority": True,
                "lane_classes": list(GROK_LANE_CLASSES),
                "lane_contract_fields": list(GROK_LANE_CONTRACT_FIELDS),
                "resource_classes": list(GROK_RESOURCE_CLASSES),
                "scheduler_rules": list(GROK_SCHEDULER_RULES),
            }
        )
    elif rule.check == "resource_governor":
        body.update(
            {
                "resource_telemetry": True,
                "pressure_governor": True,
                "storage_ownership": True,
                "telemetry_fields": list(RESOURCE_TELEMETRY_FIELDS),
                "task_scheduler_fields": list(TASK_SCHEDULER_FIELDS),
                "scheduler_selection_objectives": list(SCHEDULER_SELECTION_OBJECTIVES),
                "pressure_modes": list(PRESSURE_MODES),
                "energy_evidence_outputs": list(ENERGY_EVIDENCE_OUTPUTS),
            }
        )
    elif rule.check == "agent_os_foundation":
        body.update(
            {
                "scheduler_live": True,
                "tool_gateway_live": True,
                "memory_live": True,
                "recovery_live": True,
                "agent_roles": list(AGENT_ROLES),
                "scheduler_capabilities": list(AGENT_SCHEDULER_CAPABILITIES),
                "agent_os_performance_gates": list(AGENT_OS_PERFORMANCE_GATES),
                "context_inputs": list(CONTEXT_COMPILER_INPUTS),
                "context_properties": list(CONTEXT_COMPILER_PROPERTIES),
                "kv_state_capabilities": list(KV_STATE_CAPABILITIES),
            }
        )
    elif rule.check == "gravity_foundation":
        body.update(
            {
                "direct_execution_law": True,
                "complete_bpw_accounting": True,
                "negative_science_retrieval": True,
                "gravity_components": list(GRAVITY_COMPONENTS),
                "representation_tournament_classes": list(REPRESENTATION_TOURNAMENT_CLASSES),
            }
        )
    elif rule.check == "metal_foundation":
        body.update(
            {
                "exact_model_codegen": True,
                "family_semantic_binding": True,
                "shared_primitives": list(SHARED_PRIMITIVES),
                "family_plugins": list(FAMILY_PLUGINS),
                "architecture_fingerprint_fields": list(ARCHITECTURE_FINGERPRINT_FIELDS),
                "model_program_key_fields": list(MODEL_PROGRAM_KEY_FIELDS),
            }
        )
    elif rule.check == "manager_source":
        body.update({"source_inventory_frozen": True, "tokenizer_template_frozen": True})
    elif rule.check == "manager_anchor":
        body.update(
            {
                "capability_anchor_frozen": True,
                "capability_anchor_passed": True,
                "frozen_task_catalog_sha256": "b" * 64,
            }
        )
    elif rule.check == "manager_gravity":
        body.update(
            {
                "gravity_manager_artifact_id": QWEN30_GRAVITY_MANAGER_ARTIFACT
                if rule.model_id == MODEL_30B
                else "Qwen80-Gravity-Manager-Artifact",
                "complete_bpw": 1.5,
                "native_direct_execution": True,
                "artifact_loadable": True,
                "no_hidden_dense_shadow": True,
                "rollback_available": True,
            }
        )
    elif rule.check == "manager_tg3":
        body.update(
            {
                "gravity_manager_artifact_id": QWEN30_GRAVITY_MANAGER_ARTIFACT
                if rule.model_id == MODEL_30B
                else "Qwen80-Gravity-Manager-Artifact",
                "base_true_tps": 333.0,
                "fallback_count": 0,
                "real_metal_runtime": True,
                "real_gpu_dispatch": True,
                "gpu_dispatches": 1,
                "stable_p99": True,
                "complete_token_timing": True,
                "batch_1_base_runtime": True,
                "prompt_dependent_coherent_generation": True,
                "same_exact_model": True,
                "same_capability_tier": True,
                "tg3_review_approved": True,
            }
        )
    elif rule.check == "manager_kernel_operational":
        body.update(
            {
                "gravity_manager_artifact_id": QWEN30_GRAVITY_MANAGER_ARTIFACT
                if rule.model_id == MODEL_30B
                else "Qwen80-Gravity-Manager-Artifact",
                "operational_base_true_tps": MANAGER_KERNEL_OPERATIONAL_TPS_FLOOR,
                "fallback_count": 0,
                "exact_model_custom_kernel": True,
                "native_direct_execution": True,
                "real_metal_runtime": True,
                "real_gpu_dispatch": True,
                "gpu_dispatches": 1,
                "hcli_live_path": True,
                "kernel_parity_passed": True,
                "prompt_dependent_coherent_generation": True,
            }
        )
    elif rule.check == "manager_agent":
        body.update(
            {
                "gravity_manager_artifact_id": QWEN30_GRAVITY_MANAGER_ARTIFACT
                if rule.model_id == MODEL_30B
                else "Qwen80-Gravity-Manager-Artifact",
                "hcli_raw_decode_ratio": 0.95,
                "hcli_agent_os_integrated": True,
                "manager_capability_contract_passed": True,
                "residency_safe": True,
                "restart_and_rollback_passed": True,
                "long_unattended_campaign_passed": True,
            }
        )
    elif rule.check == "tournament":
        protocol = build_final_manager_tournament_protocol()
        body.update(
            {
                "candidates": list(TOURNAMENT_CANDIDATE_ORDER),
                "frozen_task_catalog_sha256": "c" * 64,
                "hidden_comparison_tasks_frozen": True,
                "comparison_results": {key: {"measured": True} for key in TOURNAMENT_DIMENSIONS},
                "final_manager_protocol_schema": FINAL_MANAGER_PROTOCOL_SCHEMA,
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
                        "candidate_results": {
                            candidate: {"measured": True} for candidate in CANDIDATE_ARTIFACTS
                        },
                    },
                    "MANAGER_AS_ORCHESTRATOR": {
                        "measured": True,
                        "same_hawking_agent_os": True,
                        "symmetric_helper_model_infrastructure": True,
                        "identical_resource_envelope": True,
                        "candidate_results": {
                            candidate: {"measured": True} for candidate in CANDIDATE_ARTIFACTS
                        },
                    },
                },
                "hard_gate_results": {
                    "conjunctive": True,
                    "no_score_compensation": True,
                    "by_candidate": {
                        candidate: {key: {"passed": True} for key in HARD_GATES}
                        for candidate in CANDIDATE_ARTIFACTS
                    },
                },
                "protected_task_corpus": {
                    "blind_tasks_frozen": True,
                    "families": list(TASK_FAMILIES),
                },
                "long_horizon_campaigns": {"completed": list(LONG_HORIZON_CAMPAIGNS)},
                "fairness_envelope": {
                    "equalized": {key: True for key in FAIRNESS_ENVELOPE},
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
        )
    elif rule.check == "winner":
        body.update(
            {
                "winner_model": QWEN30_GRAVITY_MANAGER_ARTIFACT,
                "designation": "ASCENSION_MANAGER",
                "tournament_seal_sha256": "d" * 64,
            }
        )
    elif rule.check == "alternate_offload":
        body.update(
            {
                "winner_model": QWEN30_GRAVITY_MANAGER_ARTIFACT,
                "alternate_model": TOURNAMENT_CANDIDATE_ORDER[1],
                "alternate_local_body_evicted": True,
                "small_fixtures_retained": True,
                "permanent_second_local_reviewer_required": False,
                "remote_hash": "e" * 64,
                "restore_command": "./restore-qualified-alternate.sh",
            }
        )
    elif rule.check == "sandbox_activation":
        body.update(
            {
                "manager_model": QWEN30_GRAVITY_MANAGER_ARTIFACT,
                "external_manager_gate_passed": True,
                "hcli_agent_os_manager_ready": True,
                "only_winner_active": True,
                "second_local_manager_active": False,
            }
        )
    elif rule.check == "family_launch":
        body.update(
            {
                "family": rule.family,
                "exact_model_qualified": True,
                "capability_passed": True,
                "parity_passed": True,
                "recovery_passed": True,
                "tg3_passed": True,
                "complete_bpw": 1.5,
                "generic_fallback_used": False,
            }
        )
    elif rule.check == "generic_reference":
        body.update({"generic_reference_intake_ready": True, "not_core_family_substitute": True})
    elif rule.check == "matrix":
        body.update({"matrix_complete": True, "direct_evidence_only": True})
    elif rule.check == "product":
        body.update({"product_test_passed": True, "real_product_path": True})
    elif rule.check == "global_audit":
        body.update({"all_advertised_models_qualified": True, "no_launch_exception": True})
    elif rule.check == "external_review":
        body.update({"review_packet_complete": True, "external_review_requested": True})
    elif rule.check == "external_review_acceptance":
        body.update(
            {"external_review_accepted": True, "findings_repaired_or_human_waived": True}
        )
    elif rule.check == "apple_release":
        body.update(
            {
                "all_launch_gates_true": True,
                "external_review_accepted": True,
                "apple_production_package_ready": True,
            }
        )
    elif rule.check == "frontier":
        body.update({"post_release_authorized": True, "tg2_tg1_research_active": True})
    return body


def _write_rule(paths: LifecyclePaths, rule, *, bible_sha256: str) -> Path:
    output = paths.evidence_root / rule.filename
    output.parent.mkdir(parents=True, exist_ok=True)
    if rule.format == "script":
        output.write_text("#!/bin/sh\nset -eu\nexit 0\n", encoding="utf-8")
        output.chmod(0o750)
        return output
    if rule.format == "sqlite":
        connection = sqlite3.connect(output)
        connection.execute("create table if not exists mechanisms (id text primary key)")
        connection.commit()
        connection.close()
        return output
    body = _base(rule, bible_sha256=bible_sha256)
    if rule.format == "jsonl":
        body["ledger_id"] = rule.artifact_id
        output.write_text(json.dumps(seal(body), sort_keys=True) + "\n", encoding="utf-8")
        return output
    output.write_text(json.dumps(seal(body), sort_keys=True), encoding="utf-8")
    return output


def _write_complete_evidence(paths: LifecyclePaths, *, bible: dict[str, Any]) -> None:
    for spec in STAGE_SPECS:
        for rule in spec.artifacts:
            _write_rule(paths, rule, bible_sha256=bible["sha256"])
        if spec.state_id == "V3_SEED_ARCHIVE":
            archive_path = paths.evidence_root / "ASCENSION_V3_SEED_ARCHIVE.json"
            archive = json.loads(archive_path.read_text())
            archive["restore_script_sha256"] = _digest(paths.evidence_root / "ASCENSION_V3_RESTORE.sh")
            archive_path.write_text(json.dumps(seal({k: v for k, v in archive.items() if k != "seal_sha256"})))
        if spec.state_id == "V3_KNOWLEDGE_PLANE":
            transfer_path = paths.evidence_root / "ASCENSION_TRANSFER_MATRIX.json"
            transfer = json.loads(transfer_path.read_text())
            transfer["mechanism_index_sha256"] = _digest(
                paths.evidence_root / "ASCENSION_MECHANISM_INDEX.sqlite"
            )
            transfer_path.write_text(json.dumps(seal({k: v for k, v in transfer.items() if k != "seal_sha256"})))
        if spec.state_id == "MANAGER_TOURNAMENT":
            tournament_path = paths.evidence_root / "ASCENSION_MANAGER_TOURNAMENT.json"
            tournament = json.loads(tournament_path.read_text())
            winner_path = paths.evidence_root / "ASCENSION_MANAGER_WINNER.json"
            winner = json.loads(winner_path.read_text())
            winner["tournament_seal_sha256"] = tournament["seal_sha256"]
            winner_path.write_text(json.dumps(seal({k: v for k, v in winner.items() if k != "seal_sha256"})))


def test_bible_contract_and_bootstrap_outputs_are_exact(tmp_path: Path) -> None:
    bible_path = Path("/Users/scammermike/Downloads/bible.md")
    audit = audit_bible(bible_path)
    assert audit["state_machine_matches"] is True
    assert tuple(audit["observed_state_machine"]) == CANONICAL_STATES

    result = evaluate_lifecycle(tmp_path / "controller", bible_path=bible_path)
    paths = LifecyclePaths.from_root(tmp_path / "controller")
    assert result["first_unmet_state"] == "V3_ADOPT"
    assert result["state_counts"] == {"certified": 0, "blocked": 1, "pending_prerequisites": 27}
    for filename in EXACT_CONTINUATION_OUTPUTS:
        assert (paths.root / filename).is_file(), filename
    assert paths.constitution_path.is_file()
    assert paths.launch_gate_path.is_file()
    assert paths.launch_gate_path.stat().st_mode & 0o111
    state = json.loads(paths.state_path.read_text())
    verify(state, label="state")
    assert state["claim_boundary"]["controller_does_not_auto_select_manager"] is True
    fidelity = json.loads(paths.fidelity_path.read_text())
    verify(fidelity, label="fidelity")
    assert fidelity["fidelity"]["bible_execution_sequence"]["text_matches_bible"] is True
    assert fidelity["fidelity"]["bible_execution_sequence"]["covered"] == 48
    heading_routing = fidelity["fidelity"]["bible_heading_routing"]
    assert heading_routing["headings_found"] == 108
    assert heading_routing["mapped"] == 108
    assert heading_routing["controller_contract_covered"] == 108
    assert heading_routing["unmapped"] == []
    markdown = paths.root / "ASCENSION_V3_FIDELITY.md"
    assert markdown.is_file()
    assert "Live receipt completion: `0/28`" in markdown.read_text(encoding="utf-8")


def test_armed_tournament_is_live_but_cannot_bypass_candidate_qualification(tmp_path: Path) -> None:
    result = arm_tournament(tmp_path / "controller", bible_path="/Users/scammermike/Downloads/bible.md")
    assert result["tournament"]["status"] == "ARMED_BLOCKED_UNQUALIFIED_CANDIDATES"
    assert result["tournament"]["final_manager_protocol_identity_sha256"] == build_final_manager_tournament_protocol()[
        "protocol_identity_sha256"
    ]
    assert result["tournament"]["claim_boundary"]["does_not_run_candidate_models"] is True
    assert result["tournament"]["claim_boundary"]["does_not_select_a_winner"] is True


def test_manager_tournament_refuses_a_missing_orchestration_fairness_gate(tmp_path: Path) -> None:
    bible = audit_bible("/Users/scammermike/Downloads/bible.md")
    rule = _rule("ASCENSION_MANAGER_TOURNAMENT")
    body = _base(rule, bible_sha256=bible["sha256"])
    body["evaluation_modes"]["MANAGER_AS_ORCHESTRATOR"]["identical_resource_envelope"] = False
    path = tmp_path / rule.filename
    path.write_text(json.dumps(seal(body)))
    report, _document = evaluate_artifact(rule, evidence_root=tmp_path, bible=bible)
    assert report["status"] == "REJECTED"
    assert any("identical_resource_envelope" in issue for issue in report["issues"])


def test_explicit_human_adoption_advances_only_the_constitutional_gate(tmp_path: Path) -> None:
    result = attest_human_adoption(
        tmp_path / "controller",
        bible_path="/Users/scammermike/Downloads/bible.md",
        instruction_sha256="f" * 64,
    )
    assert result["first_unmet_state"] == "V3_SEED_ARCHIVE"
    assert result["state_counts"] == {
        "certified": 1,
        "blocked": 1,
        "pending_prerequisites": 26,
    }
    document = json.loads(
        (LifecyclePaths.from_root(tmp_path / "controller").evidence_root / "ASCENSION_V3_ADOPTED.json").read_text()
    )
    verify(document, label="human adoption")
    assert document["certified_by"] == "human_operator"
    assert document["claim_boundary"]["does_not_certify_any_model_measurement"] is True


def test_full_receipt_graph_advances_every_bible_state_and_launch_gate(tmp_path: Path) -> None:
    root = tmp_path / "controller"
    paths = LifecyclePaths.from_root(root)
    initial = evaluate_lifecycle(root, bible_path="/Users/scammermike/Downloads/bible.md")
    bible = json.loads((Path(initial["state_path"])).read_text())["bible"]
    _write_complete_evidence(paths, bible=bible)

    result = arm_tournament(root, bible_path="/Users/scammermike/Downloads/bible.md")
    assert result["state_counts"]["certified"] == len(CANONICAL_STATES)
    assert result["launch_gate"]["status"] == "READY"
    assert result["tournament"]["status"] == "ARMED_COMPLETE_SEALED"
    state = json.loads(paths.state_path.read_text())
    assert all(item["status"] == "CERTIFIED" for item in state["states"])
    assert {item["id"] for item in state["states"]} == set(CANONICAL_STATES)


@pytest.mark.parametrize(
    ("artifact_id", "mutation", "expected"),
    [
        ("QWEN30_MANAGER_GRAVITY", {"complete_bpw": 1.5001}, "complete_bpw"),
        ("QWEN30_MANAGER_TG3", {"base_true_tps": 332.0}, "BASE_TRUE_TPS"),
        ("QWEN30_MANAGER_TG3", {"fallback_count": 1}, "fallback_count"),
        ("QWEN30_MANAGER_TG3", {"real_metal_runtime": False}, "real Metal runtime"),
        ("QWEN30_MANAGER_KERNEL_OPERATIONAL", {"operational_base_true_tps": 99.0}, "100"),
        ("QWEN30_MANAGER_AGENT_OS", {"certified_by": MODEL_30B}, "certifier"),
        ("HCLI_AGENT_OS_V3_STATUS", {"agent_roles": list(AGENT_ROLES[:-1])}, "Agent OS roles"),
        ("METAL_FAMILY_PLUGIN_MATRIX", {"family_plugins": list(FAMILY_PLUGINS[:-1])}, "family plugins"),
        ("QWEN_V3_LAUNCH_READY", {"generic_fallback_used": True}, "generic fallback"),
        (
            "ASCENSION_ALTERNATE_OFFLOAD",
            {"permanent_second_local_reviewer_required": True},
            "hidden permanent second",
        ),
        ("ASCENSION_V3_ADOPTED", {"timeline_based_completion": True}, "timeline"),
    ],
)
def test_no_drift_contract_refuses_all_prohibited_shortcuts(
    tmp_path: Path, artifact_id: str, mutation: dict[str, Any], expected: str
) -> None:
    bible = audit_bible("/Users/scammermike/Downloads/bible.md")
    rule = _rule(artifact_id)
    body = _base(rule, bible_sha256=bible["sha256"])
    body.update(mutation)
    path = tmp_path / rule.filename
    path.write_text(json.dumps(seal(body)))
    report, _document = evaluate_artifact(rule, evidence_root=tmp_path, bible=bible)
    assert report["status"] == "REJECTED"
    assert any(expected.lower() in issue.lower() for issue in report["issues"])


def test_sandbox_cannot_activate_without_a_certified_tournament(tmp_path: Path) -> None:
    paths = LifecyclePaths.from_root(tmp_path / "controller")
    bible = audit_bible("/Users/scammermike/Downloads/bible.md")
    # Seed only the direct sandbox receipt.  It must remain unreachable because
    # the full upstream tournament and both managers are absent.
    sandbox_rule = _rule("ASCENSION_SANDBOX_ACTIVE")
    _write_rule(paths, sandbox_rule, bible_sha256=bible["sha256"])
    result = evaluate_lifecycle(paths.root, bible_path="/Users/scammermike/Downloads/bible.md")
    state = json.loads(Path(result["state_path"]).read_text())
    by_id = {item["id"]: item for item in state["states"]}
    assert by_id["SANDBOX_ACTIVATION"]["status"] == "PENDING_PREREQUISITES"
    assert "MANAGER_TOURNAMENT" in by_id["SANDBOX_ACTIVATION"]["blockers"][0]


def test_family_rules_cover_every_required_launch_family() -> None:
    assert tuple(state for state, _completion, _family in FAMILY_RULES) == CANONICAL_STATES[16:24]
