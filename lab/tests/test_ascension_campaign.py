"""Tests for the detached, full-Bible Ascension campaign supervisor."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lab.operators.ascension_campaign import CampaignPaths, tick
from lab.operators.ascension_execution_plan import (
    BIBLE_EXECUTION_STEPS,
    audit_execution_sequence,
    extract_bible_execution_sequence,
)
from lab.operators.ascension_kernel_registry import FAMILY_PLUGINS, REPRESENTATION_TOURNAMENT_CLASSES
from lab.operators.ascension_foundation_contracts import AGENT_ROLES, GROK_LANE_CLASSES
from lab.operators.ascension_knowledge_contract import (
    KERNEL_GENOME_FIELDS,
    MECHANISM_INHERITANCE_RULES,
)
from lab.receipts import verify


BIBLE = Path("/Users/scammermike/Downloads/bible.md")


def _write_candidate_summary(root: str | Path) -> dict[str, Any]:
    destination = Path(root) / "source-admission" / "SOURCE_ADMISSION_STATUS.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ALL_METADATA_CAPTURED",
        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "records": [
            {
                "artifact_id": "QWEN30_SOURCE_METADATA_CANDIDATE",
                "status": "CANDIDATE_METADATA_CAPTURED",
                "repository": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                "revision": "a" * 40,
                "no_model_body_downloaded": True,
            },
            {
                "artifact_id": "QWEN80_SOURCE_METADATA_CANDIDATE",
                "status": "CANDIDATE_METADATA_CAPTURED",
                "repository": "Qwen/Qwen3-Coder-Next",
                "revision": "b" * 40,
                "no_model_body_downloaded": True,
            },
        ],
    }
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_bible_execution_map_is_exact_and_covers_all_48_steps() -> None:
    audit = audit_execution_sequence(BIBLE)
    assert audit["matches"] is True
    assert audit["expected_count"] == 48
    assert tuple(step.identifier for step in BIBLE_EXECUTION_STEPS) == tuple(range(48))

    drifted = BIBLE.read_text(encoding="utf-8").replace(
        "Run the protected Manager Tournament.", "Run an unprotected tournament.", 1
    )
    assert len(extract_bible_execution_sequence(drifted)) == 48
    assert extract_bible_execution_sequence(drifted) != tuple(
        step.description for step in BIBLE_EXECUTION_STEPS
    )


def test_tick_wires_each_bible_step_without_claiming_runtime_qualification(tmp_path: Path) -> None:
    calls: list[Path] = []

    def capture(root: str | Path) -> Mapping[str, Any]:
        calls.append(Path(root))
        return _write_candidate_summary(root)

    result = tick(
        tmp_path / "controller",
        bible_path=BIBLE,
        force_metadata_refresh=True,
        capture_sources_fn=capture,
    )

    verify(result, label="campaign supervisor")
    assert result["state"] == "RUNNING"
    assert result["heartbeat"] == 1
    assert result["metadata_refresh"]["performed"] is True
    assert calls == [tmp_path / "controller"]
    assert result["bible_step_coverage"]["all_bible_steps_wired"] is True
    assert result["bible_step_coverage"]["receipt_complete_rows"] == 0
    assert result["claim_boundary"]["no_model_body_downloads"] is True
    assert result["claim_boundary"]["no_automatic_promotion"] is True

    paths = CampaignPaths.from_root(tmp_path / "controller")
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    verify(manifest, label="execution manifest")
    assert len(manifest["steps"]) == 48
    tournament = manifest["steps"][23]
    assert tournament["states"] == ["MANAGER_TOURNAMENT"]
    assert tournament["operating_status"] == "WAITING_FOR_UPSTREAM_CERTIFICATION"
    assert manifest["steps"][46]["operating_status"] == "OUTSIDE_LAUNCH_CRITICAL_PATH"
    assert manifest["source_metadata"]["candidate_only"] is True
    kernel_contract = json.loads(paths.kernel_contract_path.read_text(encoding="utf-8"))
    verify(kernel_contract, label="kernel compiler contract")
    assert kernel_contract["status"] == "CONTROLLER_CONFIGURATION_ONLY"
    assert kernel_contract["required_family_plugins"] == list(FAMILY_PLUGINS)
    assert kernel_contract["representation_tournament_classes"] == list(
        REPRESENTATION_TOURNAMENT_CLASSES
    )
    manager_workflow = json.loads(paths.manager_workflow_path.read_text(encoding="utf-8"))
    verify(manager_workflow, label="dual manager workflow")
    assert manager_workflow["fixed_candidate_order"] == [
        "Qwen30-Gravity-Manager-Artifact",
        "Qwen80-Gravity-Manager-Artifact",
    ]
    assert manager_workflow["raw_bf16_models_are_source_authorities_not_tournament_participants"] is True
    assert manager_workflow["handoff"]["tournament_phase"] == "WAITING_FOR_BOTH_QUALIFIED_MANAGERS"
    assert all(row["source_candidate"]["candidate_only"] is True for row in manager_workflow["managers"])
    family_workflow = json.loads(paths.family_workflow_path.read_text(encoding="utf-8"))
    verify(family_workflow, label="family campaign workflow")
    assert len(family_workflow["families"]) == 8
    assert family_workflow["generic_hf_reference"]["is_not_core_family_substitute"] is True
    assert all(row["metadata_only_support_forbidden"] is True for row in family_workflow["families"])
    tournament_workflow = json.loads(paths.tournament_workflow_path.read_text(encoding="utf-8"))
    verify(tournament_workflow, label="tournament workflow")
    assert tournament_workflow["runtime_phase"] == "NOT_ARMED"
    assert len(tournament_workflow["deterministic_comparison_contract"]["comparison_dimensions"]) == 24
    assert tournament_workflow["alternate_offload_contract"]["must_precede_sandbox_activation"] is True
    final_protocol = tournament_workflow["final_manager_selection_protocol"]
    verify(final_protocol["protocol"], label="final manager tournament protocol")
    assert final_protocol["binding_required_in_scored_tournament_receipt"] is True
    assert final_protocol["protocol_identity_sha256"] == final_protocol["protocol"]["protocol_identity_sha256"]
    assert final_protocol["protocol"]["evaluation_modes"]["required"] == [
        "SOLO_MANAGER",
        "MANAGER_AS_ORCHESTRATOR",
    ]
    assert final_protocol["protocol"]["hard_gates"]["conjunctive"] is True
    release_workflow = json.loads(paths.release_workflow_path.read_text(encoding="utf-8"))
    verify(release_workflow, label="release workflow")
    assert release_workflow["derived_launch_gate"]["status"] == "BLOCKED"
    assert release_workflow["external_review"]["review_packet_is_not_launch_approval"] is True
    assert release_workflow["post_release_frontier"]["may_begin_only_after_apple_release_certified"] is True
    foundation_contracts = json.loads(paths.foundation_contracts_path.read_text(encoding="utf-8"))
    verify(foundation_contracts, label="foundation contracts")
    assert foundation_contracts["grok_build_fabric"]["lane_classes"] == list(GROK_LANE_CLASSES)
    assert foundation_contracts["agent_os"]["roles"] == list(AGENT_ROLES)
    knowledge_contract = json.loads(paths.knowledge_contract_path.read_text(encoding="utf-8"))
    verify(knowledge_contract, label="knowledge plane contract")
    assert knowledge_contract["outputs"]["ASCENSION_KERNEL_GENOME"]["fields"] == list(
        KERNEL_GENOME_FIELDS
    )
    assert knowledge_contract["mechanism_inheritance_rules"] == list(MECHANISM_INHERITANCE_RULES)


def test_fresh_metadata_is_not_refreshed_on_every_watch_tick(tmp_path: Path) -> None:
    calls: list[Path] = []

    def capture(root: str | Path) -> Mapping[str, Any]:
        calls.append(Path(root))
        return _write_candidate_summary(root)

    root = tmp_path / "controller"
    tick(root, bible_path=BIBLE, force_metadata_refresh=True, capture_sources_fn=capture)
    second = tick(root, bible_path=BIBLE, metadata_refresh_seconds=3600, capture_sources_fn=capture)
    assert calls == [root]
    assert second["metadata_refresh"]["performed"] is False
