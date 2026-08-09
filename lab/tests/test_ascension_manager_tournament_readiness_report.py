"""Focused tests for the read-only final-manager readiness snapshot."""
from __future__ import annotations

import json
from pathlib import Path

from lab.operators.ascension_manager_tournament_protocol import (
    build_final_manager_tournament_protocol,
)
from lab.operators.ascension_manager_tournament_readiness_report import (
    AUTHORITY_SPECS,
    CANDIDATES,
    EXPECTED_PROTOCOL_IDENTITY_SHA256,
    ReadinessPaths,
    STATUS_PREPARED,
    STATUS_REFUSED,
    build_readiness_report,
    main,
)
from lab.receipts import seal, verify


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, document: dict, *, sealed: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = seal(document) if sealed else document
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _zero_authority_boundary() -> dict[str, object]:
    return {
        "new_physical_model_processes_authorized": 0,
        "server_starts_authorized": 0,
        "port_binds_authorized": 0,
        "gpu_leases_authorized": 0,
        "tournament_state_mutations_authorized": 0,
        "paired_world_activation_authorized": False,
    }


def _authority_document(schema: str, status: str) -> dict[str, object]:
    return {
        "schema": schema,
        "status": status,
        "prepared": True,
        "authority_boundary": _zero_authority_boundary(),
        "execution_boundary": {
            "model_loaded": False,
            "server_started": False,
            "gpu_used": False,
            "tournament_executed": False,
        },
        "tournament_active": False,
        "winner_selected": False,
        "scored_task_execution_active": False,
        "scorecards_executed_by_this_contract": False,
        "hidden_tasks_created": False,
        "scored_execution_started": False,
        "candidate_or_red_team_hidden_access_granted": False,
    }


def _requirements() -> dict[str, dict[str, str]]:
    return {
        requirement: {"state": "PASS"}
        for requirement in (
            "verified_raw_source_identity",
            "current_source_revalidation",
            "complete_admitted_artifact_at_most_1_5_bpw",
            "native_exact_full_token_runtime",
            "measured_hcli",
            "custom_kernel_operational_at_least_100_tps",
            "capability_and_evaluation_receipt",
            "final_manager_operations_agent_os_session_restart_residency_rollback_storage",
            "tg10_operational_exact_model_100_tps",
            "tg3_at_least_333_tps",
        )
    }


def _paths(tmp_path: Path) -> ReadinessPaths:
    return ReadinessPaths.from_roots(
        repository_root=REPOSITORY_ROOT,
        lifecycle_root=tmp_path / "lifecycle",
        physical_root=tmp_path / "physical",
        authority_root=tmp_path / "authorities",
    )


def _write_ready_fixture(tmp_path: Path) -> ReadinessPaths:
    paths = _paths(tmp_path)
    protocol = build_final_manager_tournament_protocol()
    assert protocol["protocol_identity_sha256"] == EXPECTED_PROTOCOL_IDENTITY_SHA256

    _write_json(
        paths.lifecycle_state,
        {
            "schema": "hawking.ascension.v3_state.v1",
            "states": [
                {"id": "MANAGER_30B_AGENT", "status": "CERTIFIED"},
                {"id": "MANAGER_80B_AGENT", "status": "CERTIFIED"},
                {"id": "MANAGER_TOURNAMENT", "status": "PENDING_PREREQUISITES"},
            ],
            "tournament": {
                "final_manager_protocol_schema": protocol["schema"],
                "final_manager_protocol_identity_sha256": protocol["protocol_identity_sha256"],
            },
        },
    )
    _write_json(
        paths.lifecycle_controller,
        {
            "schema": "hawking.ascension.manager_tournament_controller.v1",
            "candidate_order": [
                "Qwen30-Gravity-Manager-Artifact",
                "Qwen80-Gravity-Manager-Artifact",
            ],
            "final_manager_protocol_schema": protocol["schema"],
            "final_manager_protocol_identity_sha256": protocol["protocol_identity_sha256"],
        },
    )
    _write_json(
        paths.lifecycle_workflow,
        {
            "schema": "hawking.ascension.manager_tournament_workflow.v1",
            "candidates": [
                {
                    "tournament_artifact_id": "Qwen30-Gravity-Manager-Artifact",
                    "agent_qualification_state": "CERTIFIED",
                },
                {
                    "tournament_artifact_id": "Qwen80-Gravity-Manager-Artifact",
                    "agent_qualification_state": "CERTIFIED",
                },
            ],
            "final_manager_selection_protocol": {
                "schema": protocol["schema"],
                "protocol_identity_sha256": protocol["protocol_identity_sha256"],
                "protocol": protocol,
            },
        },
    )
    _write_json(
        paths.physical_gate,
        {
            "schema": "hawking.ascension.physical_tournament_gate.v1",
            "fixed_candidate_order": [
                "Qwen30-Gravity-Manager-Artifact",
                "Qwen80-Gravity-Manager-Artifact",
            ],
            "frozen_protected_tournament_suite_preflight": {"state": "PASS"},
            "models": [
                {
                    "key": "qwen30",
                    "gravity_artifact_id": "Qwen30-Gravity-Manager-Artifact",
                    "pre_final_review_qualification": "PASS",
                    "requirements": _requirements(),
                },
                {
                    "key": "qwen80",
                    "gravity_artifact_id": "Qwen80-Gravity-Manager-Artifact",
                    "pre_final_review_qualification": "PASS",
                    "requirements": _requirements(),
                },
            ],
        },
    )
    _write_json(
        paths.operational_ascent,
        {
            "schema": "hawking.ascension.physical_operational_ascent.v1",
            "status": "BOTH_VALID_TG10_OPERATIONAL_RECEIPTS_BOUND",
            "both_valid_tg10_receipts": True,
            "protected_tournament": {"tg3_remains_required": True},
            "evidence": {
                "models": {
                    "qwen30": {"tg10_receipt_seal_sha256": "a" * 64},
                    "qwen80": {"tg10_receipt_seal_sha256": "b" * 64},
                }
            },
        },
    )
    _write_json(
        paths.tg_status(CANDIDATES[0]),
        {
            "schema": "hawking.ascension.qwen30_bootstrap_lanes.v1",
            "lane": "C_QWEN30_METAL_TG3",
            "phase": "OBSERVATIONAL_ONLY",
        },
        sealed=False,
    )
    _write_json(
        paths.tg_status(CANDIDATES[1]),
        {
            "schema": "hawking.ascension.qwen80_bootstrap_lanes.v1",
            "lane": "QWEN80_METAL_TG3",
            "phase": "OBSERVATIONAL_ONLY",
        },
        sealed=False,
    )
    for authority in AUTHORITY_SPECS:
        _write_json(
            paths.authority_document(authority),
            _authority_document(authority.expected_schema, authority.expected_status),
        )
    return paths


def _reload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_complete_fixture_prepares_but_never_activates_or_selects(tmp_path: Path) -> None:
    paths = _write_ready_fixture(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    report = build_readiness_report(paths, recorded_at="2026-08-09T00:00:00Z")

    assert verify(report) == report
    assert report["status"] == STATUS_PREPARED
    assert report["prepared_not_active"] is True
    assert report["tournament_ready_for_execution"] is False
    assert report["winner_selected"] is False
    assert report["external_report_emitted"] is False
    assert report["missing_prerequisites"] == []
    assert report["claim_boundary"]["read_only"] is True
    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before


def test_current_style_missing_floors_tg10_and_tg3_are_plainly_refused(tmp_path: Path) -> None:
    paths = _write_ready_fixture(tmp_path)
    lifecycle = _reload(paths.lifecycle_state)
    lifecycle["states"][1]["status"] = "PENDING_PREREQUISITES"
    _write_json(paths.lifecycle_state, {key: value for key, value in lifecycle.items() if key != "seal_sha256"})

    gate = _reload(paths.physical_gate)
    gate["models"][0]["pre_final_review_qualification"] = "BLOCKED"
    gate["models"][0]["requirements"]["tg10_operational_exact_model_100_tps"]["state"] = "BLOCKED"
    gate["models"][1]["requirements"]["tg3_at_least_333_tps"]["state"] = "BLOCKED"
    _write_json(paths.physical_gate, {key: value for key, value in gate.items() if key != "seal_sha256"})

    operational = _reload(paths.operational_ascent)
    operational["status"] = "WAITING_FOR_BOTH_VALID_TG10_OPERATIONAL_RECEIPTS"
    operational["both_valid_tg10_receipts"] = False
    operational["evidence"]["models"]["qwen30"]["tg10_receipt_seal_sha256"] = None
    _write_json(paths.operational_ascent, {key: value for key, value in operational.items() if key != "seal_sha256"})

    report = build_readiness_report(paths, recorded_at="2026-08-09T00:00:00Z")

    assert report["status"] == STATUS_REFUSED
    assert "both_complete_manager_floors" in report["missing_prerequisites"]
    assert "both_exact_tg10_operational_receipts" in report["missing_prerequisites"]
    assert "both_tg3_qualifications" in report["missing_prerequisites"]
    assert any("qwen30" in blocker and "manager" in blocker for blocker in report["blockers"])
    assert any("qwen80" in blocker and "TG3" in blocker for blocker in report["blockers"])


def test_protocol_identity_drift_fails_closed_even_if_the_documents_are_resealed(tmp_path: Path) -> None:
    paths = _write_ready_fixture(tmp_path)
    workflow = _reload(paths.lifecycle_workflow)
    embedded = workflow["final_manager_selection_protocol"]["protocol"]
    embedded["protocol_identity_sha256"] = "0" * 64
    workflow["final_manager_selection_protocol"]["protocol_identity_sha256"] = "0" * 64
    workflow["final_manager_selection_protocol"]["protocol"] = seal(
        {key: value for key, value in embedded.items() if key != "seal_sha256"}
    )
    _write_json(paths.lifecycle_workflow, {key: value for key, value in workflow.items() if key != "seal_sha256"})

    report = build_readiness_report(paths, recorded_at="2026-08-09T00:00:00Z")

    assert report["status"] == STATUS_REFUSED
    assert "fixed_final_manager_protocol_identity" in report["missing_prerequisites"]
    assert any("protocol identity mismatch" in blocker for blocker in report["blockers"])


def test_missing_or_unsealed_protected_contract_materialization_fails_closed(tmp_path: Path) -> None:
    paths = _write_ready_fixture(tmp_path)
    corpus = next(spec for spec in AUTHORITY_SPECS if spec.key == "protected_corpus")
    paths.authority_document(corpus).unlink()
    report = build_readiness_report(paths, recorded_at="2026-08-09T00:00:00Z")

    assert report["status"] == STATUS_REFUSED
    assert "prepared_paired_and_protected_contract_identities" in report["missing_prerequisites"]
    assert "protected_corpus authority: missing" in report["blockers"]
    assert report["prepared_authority_source_identities"]["protected_corpus"][
        "prepared_definition_verified"
    ] is True


def test_claimed_readiness_is_rejected_and_cli_only_emits_stdout(tmp_path: Path, capsys) -> None:
    paths = _write_ready_fixture(tmp_path)
    report = build_readiness_report(paths, claimed_ready=True, recorded_at="2026-08-09T00:00:00Z")
    assert report["status"] == STATUS_REFUSED
    assert report["readiness_claim_accepted"] is False
    assert "caller_readiness_claim_rejected" in report["missing_prerequisites"]

    exit_code = main(
        [
            "--repository-root",
            str(paths.repository_root),
            "--lifecycle-root",
            str(paths.lifecycle_root),
            "--physical-root",
            str(paths.physical_root),
            "--authority-root",
            str(paths.authority_root),
        ]
    )
    emitted = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert emitted["schema"] == "hawking.ascension.manager_tournament_readiness_report.v1"
    assert emitted["tournament_ready_for_execution"] is False


def test_missing_authority_source_identity_cannot_be_substituted_by_a_sealed_document(tmp_path: Path) -> None:
    paths = _write_ready_fixture(tmp_path)
    missing_source_root = tmp_path / "empty-source-root"
    missing_source_root.mkdir()
    source_missing_paths = ReadinessPaths.from_roots(
        repository_root=missing_source_root,
        lifecycle_root=paths.lifecycle_root,
        physical_root=paths.physical_root,
        authority_root=paths.authority_root,
    )

    report = build_readiness_report(source_missing_paths, recorded_at="2026-08-09T00:00:00Z")

    assert report["status"] == STATUS_REFUSED
    assert "prepared_paired_and_protected_contract_identities" in report["missing_prerequisites"]
    assert all(
        not authority["prepared_definition_verified"]
        for authority in report["prepared_authority_source_identities"].values()
    )
