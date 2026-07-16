#!/usr/bin/env python3.12
"""Contract tests for the condensed Doctor V5 profile and archive surface."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys


CONDENSE = Path(__file__).resolve().parents[1]
ROOT = CONDENSE.parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import doctor_v5_profiles as profiles  # noqa: E402


def test_extreme_layout_and_archive_are_exact() -> None:
    assert profiles.validate() == []
    document = profiles.profile_document()
    assert document["retained_module_count"] == 37
    assert document["retired_module_count"] == 39
    assert len(document["retired_modules_sha256"]) == 64


def test_legacy_record_preserves_discovery_metadata() -> None:
    record = profiles.legacy_record("aggressive-admission-policy")
    assert record["profile"] == "admission"
    assert record["replacement_command"] == "doctor.admission"
    assert "hawking.doctor_v5_aggressive_admission_overlay.v1" in record["schemas"]
    assert "reports/condense/doctor_v5_ultra" in record["artifact_paths"]
    assert record["subcommands"] == ["stage", "status", "validate"]
    assert len(profiles.archive_source(record["legacy_module"])) > 100


def test_replay_is_latest_state_terminal_aware_and_deterministic() -> None:
    events = [
        {"cell_id": "a", "status": "running", "attempt": 1},
        {"cell_id": "b", "status": "resource-stop", "attempt": 2},
        {"cell_id": "a", "status": "complete", "attempt": 1},
        {"cell_id": "c", "status": "negative", "attempt": 1},
    ]
    result = profiles.replay(events)
    assert result["cell_count"] == 3
    assert result["terminal_count"] == 2
    assert result["attempts_total"] == 4
    assert result["status_counts"]["resource-stop"] == 1
    assert result == profiles.replay(events)


def test_post120_handoff_validator_preserves_sealed_artifact_semantics(tmp_path: Path) -> None:
    bindings = {}
    for name, field in profiles.POST120_BINDINGS.items():
        value = {"schema": f"test.{name}.v1", "payload": name}
        value[field] = profiles._strict_sha256(value)
        path = tmp_path / f"{name}.json"
        raw = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
        path.write_bytes(raw)
        bindings[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "schema": value["schema"],
            "semantic_hash_field": field,
            "semantic_sha256": value[field],
        }
    readiness = {
        "exact_120b_work_plan_structurally_valid": True,
        "exact_40_cell_pending_wiring_structurally_valid": True,
        "exact_source_rate_branch_fanout_structurally_valid": True,
        "shared_preprocess_cache_structurally_valid": True,
        "shared_derived_preprocess_consumers_still_unqualified": 24_600,
        "tokenizer_dual_path_gate_passed_but_not_promotion_reviewed": True,
        "higher_tier_generic_manifest_and_admission_contracts_present": True,
        "higher_tier_exact_source_manifests_present": 0,
        "physical_acceleration_facets_qualified": 0,
        "physical_ab_plan_structurally_valid": True,
        "physical_ab_facets_total": 10,
        "physical_ab_facets_qualified": 0,
        "physical_ab_program_adapters_registered": 0,
        "physical_ab_execution_permitted": False,
        "reviewed_120b_live_adapters": 0,
    }
    handoff = {
        "schema": profiles.POST120_HANDOFF_SCHEMA,
        "created_at": "2026-07-16T00:00:00+00:00",
        "status": "sealed-unbound-handoff-not-executable",
        "bindings": bindings,
        "coverage": {
            "gptoss_source_units": 615,
            "gptoss_isolated_jobs": 24_600,
            "gptoss_rates": 10,
            "gptoss_branches": 4,
            "gptoss_pending_cells": 40,
            "named_higher_tier_horizons": 3,
            "named_horizon_cell_templates": 120,
            "aggressive_facets": list(profiles.POST120_FACETS),
            "physical_ab_facets": list(profiles.PHYSICAL_AB_FACETS),
            "physical_thread_profile_cells": 800,
            "physical_block_parallel_cells": 200,
        },
        "readiness": readiness,
        "promotion_gate": profiles._post120_promotion_gate(),
        "claim_boundary": {
            "intermediate_or_structural_evidence_is_final_quality": False,
            "unsupported_or_negative_outcomes_synthesized": False,
            "live_queue_worker_registry_plan_or_runtime_specs_mutated": False,
            "runtime_defaults_changed": False,
            "source_or_parent_deletion_permitted": False,
            "execution_permitted": False,
            "quality_claims_permitted": False,
        },
    }
    handoff["handoff_sha256"] = profiles._strict_sha256(handoff)
    assert profiles.validate_handoff_packet(handoff) == []
    (tmp_path / "gptoss_work_plan.json").write_text("{}\n", encoding="utf-8")
    assert any(
        "gptoss_work_plan" in error
        for error in profiles.validate_handoff_packet(handoff)
    )


def test_unified_dispatcher_exposes_profiles_and_legacy_aliases() -> None:
    selftest = subprocess.run(
        [sys.executable, "-m", "tools.condense", "doctor.profiles", "selftest"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert selftest.returncode == 0, selftest.stderr
    assert "selftest OK" in selftest.stdout

    status = subprocess.run(
        [sys.executable, "-m", "tools.condense", "doctor.acceleration-eta", "status"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert status.returncode == 0, status.stderr
    record = json.loads(status.stdout)
    assert record["status"] == "superseded"
    assert record["replacement_command"] == "doctor.report"
    assert record["mutates_campaign"] is False
