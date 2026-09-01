from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import architecture_atlas as atlas


def test_first_sweep_is_typed_and_covers_the_source_schools():
    document = atlas.build_atlas(repo_root=Path.cwd().parents[1])
    result = atlas.validate_atlas(document)

    assert result["passed"] is True
    assert result["source_school_count"] == len(atlas.SOURCE_SCHOOLS)
    assert result["entry_count"] >= 12
    assert set(atlas.SOURCE_SCHOOLS).issubset(document["source_schools"])
    assert set(atlas.PRIMITIVES).issubset(document["backend_neutral_primitives"])
    coverage = document["source_technique_coverage"]
    assert len(coverage) >= 40
    assert {row["source_school"] for row in coverage} == set(atlas.SOURCE_SCHOOLS)
    assert any(row["source_technique"] == "FlashAttention-style IO-aware attention" for row in coverage)
    assert all(row["behavior_id"] in {entry["behavior_id"] for entry in document["entries"]} for row in coverage)
    assert all(row["behavior_taxonomy"] for row in document["entries"])
    assert all(row["cheapest_falsifier"].strip() for row in document["entries"])


def test_entries_are_ranked_by_information_value_not_nominal_utilization():
    document = atlas.build_atlas()
    scores = [row["expected_value_score"] for row in document["entries"]]

    assert scores == sorted(scores, reverse=True)
    assert document["entries"][0]["behavior_id"] == "move_or_recompute"
    assert all("nominal_utilization" not in row for row in document["entries"])


def test_queue_is_detached_and_has_a_complete_verification_funnel():
    document = atlas.build_atlas()
    experiments = document["experiment_queue"]["experiments"]

    assert len(experiments) >= 8
    assert {row["target"]["model"] for row in experiments} >= {"Qwen27", "Flash"}
    for row in experiments:
        assert row["runner"]["detached"] is True
        assert row["verification_ladder"] == [
            "structural_compile_and_negative_controls",
            "diagnostic_relative_interleaved_ab",
            "protected_absolute_complete_wall_with_capability_gate",
        ]
        assert row["falsifier"]
        assert row["promotion"]["requires_independent_capability"] is True


def test_hwir_and_asic_outputs_are_hypotheses_until_hardware_evidence_exists():
    document = atlas.build_atlas()
    hwir = document["hwir_hypotheses"]
    asic = document["asic_candidate_ledger"]["entries"]

    assert {row["behavior_id"] for row in hwir} == {
        row["behavior_id"] for row in document["entries"]
    }
    assert asic
    assert all(row["status"] == "WATCHLIST" for row in asic)
    assert all(row["asic_candidate"] is False for row in asic)
    assert all(row["label"].startswith("[D]") for row in hwir)


def test_promotion_gate_refuses_diagnostic_or_incomplete_results():
    refused = atlas.promotion_decision(
        benchmark_class="DIAGNOSTIC_RELATIVE",
        evidence_class="HAWKING_MEASURED",
        complete_useful_wall_ns=100,
        capability_verified=True,
        fallback_count=0,
        active_bytes_per_token=1000,
    )
    assert refused["promotion_allowed"] is False
    assert "protected_absolute_evidence_required" in refused["reasons"]

    missing_fallback = atlas.promotion_decision(
        benchmark_class="QUALIFIED_PROTECTED",
        evidence_class="HAWKING_PROTECTED_VERIFIED",
        complete_useful_wall_ns=100,
        capability_verified=True,
        fallback_count=None,
        active_bytes_per_token=1000,
    )
    assert missing_fallback["promotion_allowed"] is False
    assert "zero_fallback_evidence_required" in missing_fallback["reasons"]

    accepted = atlas.promotion_decision(
        benchmark_class="QUALIFIED_PROTECTED",
        evidence_class="HAWKING_PROTECTED_VERIFIED",
        complete_useful_wall_ns=100,
        capability_verified=True,
        fallback_count=0,
        active_bytes_per_token=1000,
    )
    assert accepted["promotion_allowed"] is True
    assert accepted["reasons"] == []


def test_emit_round_trips_through_the_validator(tmp_path: Path):
    path = atlas.emit_atlas(repo_root=tmp_path, output=tmp_path / "atlas.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))

    assert path == tmp_path / "atlas.json"
    assert atlas.validate_atlas(loaded)["passed"] is True
    assert loaded["fingerprint"] == atlas.build_atlas(repo_root=tmp_path)["fingerprint"]


def test_product_names_do_not_leak_into_hawking_primitive_names():
    forbidden = ("CUDA", "TPU", "FPGA", "ANE", "GROQ", "CEREBRAS")
    assert all(not any(word in primitive.upper() for word in forbidden) for primitive in atlas.PRIMITIVES)
