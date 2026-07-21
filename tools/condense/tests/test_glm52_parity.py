#!/usr/bin/env python3.12
"""Acceptance tests for the sealed GLM-5.2 twin/reference parity run."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest


CONDENSE = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = CONDENSE.parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_parity as parity  # noqa: E402
from glm52_common import canonical, verify_sealed  # noqa: E402


@pytest.fixture(scope="module")
def fresh_run() -> tuple[dict, dict]:
    adapter, reference = parity._run()
    verify_sealed(adapter, label="fresh adapter twin")
    verify_sealed(reference, label="fresh reference parity")
    return adapter, reference


def test_fresh_acceptance_run_is_green_without_parent_overclaim(
    fresh_run: tuple[dict, dict],
) -> None:
    adapter, reference = fresh_run
    assert adapter["status"] == (
        "PASS_SYNTHETIC_TWIN_AND_OFFICIAL_HEADER_TOKENIZER_SCHEMA"
    )
    assert reference["status"] == (
        "PASS_SYNTHETIC_MAIN_AND_MTP_SELF_CONSISTENCY_SOURCE_PARENT_PENDING"
    )
    assert adapter["source_parent_parity_claimed"] is False
    assert reference["claim_boundary"]["official_bf16_parent_forward"] == (
        "PENDING_FIRST_ADMITTED_SOURCE_WINDOW"
    )
    assert reference["claim_boundary"]["capability"] == "NOT_CLAIMED"
    environment = reference["runtime"]["environment"]
    assert environment["status"] == "PASS_ISOLATED_FULLY_PINNED"
    assert environment["system_site_packages"] is False
    assert environment["all_direct_imports_within_environment"] is True
    assert environment["locked_versions"]["numpy"] == "2.2.6"
    assert environment["locked_versions"]["torch"] == "2.6.0"
    assert all(
        row["status"] == "PASS"
        for row in reference["per_layer_output_metrics"].values()
    )
    long_probe = reference["long_context_indexer_shape_probe"]
    assert long_probe["tested_key_count"] == 1_048_576
    assert long_probe["official_configured_index_topk"] == 2_048
    assert long_probe["synthetic_fixture_index_topk"] == 2
    assert long_probe["topk_shape"] == [1, 1, 2_048]
    assert long_probe["official_selection_shape_exercised"] is True
    assert long_probe["full_attention_or_model_executed"] is False
    assert long_probe["one_million_context_capability_claimed"] is False


def test_indexer_records_strict_tie_and_raw_score_evidence(
    fresh_run: tuple[dict, dict],
) -> None:
    _adapter, reference = fresh_run
    indexer = reference["indexer"]
    assert indexer["score_parity"]["status"] == "PASS"
    assert indexer["tie_aware_causally_effective_set_agreement"] == 1.0
    assert indexer["causally_effective_set_agreement"] <= 1.0
    assert indexer["tie_equivalent_disagreements"] == len(indexer["tie_evidence"])
    for row in indexer["tie_evidence"]:
        assert row["reference_only"]
        assert len(row["reference_only"]) == len(row["official_only"])
        assert max(row["reference_candidate_scores"]) == min(
            row["reference_candidate_scores"]
        )
        assert max(row["official_candidate_scores"]) == min(
            row["official_candidate_scores"]
        )


def test_cache_mtp_and_generated_artifact_seals(fresh_run: tuple[dict, dict]) -> None:
    adapter, reference = fresh_run
    assert reference["official_prefill_vs_tokenwise_cache"]["status"] == "PASS"
    assert reference["reference_prefill_vs_tokenwise_cache"]["status"] == "PASS"
    assert reference["reference_deterministic_exact_replay"] is True
    assert reference["mtp"]["step_zero_computes_own_index"] is True
    assert reference["mtp"]["step_one_reuses_step_zero_index"] is True
    assert reference["mtp"]["position_zero_shifted_embedding_exactly_zero"] is True
    assert reference["mtp"]["finite_logits"] is True
    assert reference["mtp"]["external_pinned_runtime_executed"] is False

    for name, expected in (
        ("GLM52_ADAPTER_TWIN.json", adapter),
        ("GLM52_REFERENCE_PARITY.json", reference),
    ):
        value = json.loads((REPO_ROOT / name).read_text(encoding="utf-8"))
        verify_sealed(value, label=name)
        assert canonical(value) == canonical(expected)
