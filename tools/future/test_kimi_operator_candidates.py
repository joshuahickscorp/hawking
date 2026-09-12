"""Tests for the Kimi candidate matrix and immutable-base contract."""
from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from tools.future import tabula
from tools.future.kimi_operator_candidates import (
    CandidateContractError,
    apply_candidate,
    catalog,
    refusal_intervention_preflight,
)


def _fixture():
    rng = np.random.default_rng(4)
    return {
        "model.layers.0.mlp.down_proj.weight": rng.normal(size=(5, 4)).astype(np.float32),
        "model.layers.0.mlp.shared_experts.down_proj.weight": rng.normal(size=(5, 4)).astype(np.float32),
        "model.layers.0.mlp.experts.3.down_proj.weight": rng.normal(size=(5, 4)).astype(np.float32),
        "model.layers.1.mlp.down_proj.weight": rng.normal(size=(5, 4)).astype(np.float32),
        "model.embed_tokens.weight": rng.normal(size=(7, 5)).astype(np.float32),
    }


def test_catalog_registers_all_named_methods_without_ranking_them():
    rows = catalog(n_layers=2, target_layers=(0, 1), selected_experts={0: (3,)})
    assert [row.method for row in rows] == ["OPA", "OPB", "OPC", "OPD", "OPE", "OPF", "OPG"]
    assert all(row.source_model == "KIMI_BASE" for row in rows)


def test_every_candidate_is_in_memory_and_does_not_mutate_base():
    rng = np.random.default_rng(8)
    weights = _fixture()
    before = {key: value.copy() for key, value in weights.items()}
    directions = {0: rng.normal(size=5), 1: rng.normal(size=5)}
    input_directions = {0: rng.normal(size=4), 1: rng.normal(size=4)}
    subspaces = {
        0: np.column_stack((directions[0], rng.normal(size=5), rng.normal(size=5), rng.normal(size=5))),
        1: np.column_stack((directions[1], rng.normal(size=5), rng.normal(size=5), rng.normal(size=5))),
    }
    rows = catalog(n_layers=2, target_layers=(0, 1), selected_experts={0: (3,)})
    for candidate in rows:
        output, receipt = apply_candidate(
            weights,
            candidate,
            directions,
            input_directions=input_directions,
            subspaces=subspaces,
            code_commit="test-commit",
        )
        assert output is not weights
        assert receipt["source_model"] == "KIMI_BASE"
        assert receipt["weights_written"] is False
        assert receipt["promotion"] == "NOT_PERFORMED"
        assert receipt["nova_lineage_required_for_promotion"] is True
        assert receipt["nova_lineage_receipt"] is None
        assert receipt["code_commit"] == "test-commit"
        assert receipt["touched_tensors"]
    for key in weights:
        assert np.array_equal(weights[key], before[key])


def test_layer_local_candidate_only_touches_declared_layer():
    weights = _fixture()
    direction = {0: np.ones(5)}
    candidate = catalog(n_layers=2, target_layers=(0,))[4]  # OPE
    output, receipt = apply_candidate(weights, candidate, direction)
    assert receipt["touched_layers"] == [0]
    assert np.array_equal(output["model.layers.1.mlp.down_proj.weight"],
                          weights["model.layers.1.mlp.down_proj.weight"])
    assert not np.array_equal(output["model.layers.0.mlp.down_proj.weight"],
                              weights["model.layers.0.mlp.down_proj.weight"])


def test_expert_candidate_requires_explicit_selected_scope():
    weights = _fixture()
    direction = {0: np.ones(5)}
    candidate = catalog(n_layers=1, target_layers=(0,), selected_experts={})[5]  # OPF
    with pytest.raises(CandidateContractError, match="selected-expert scope"):
        apply_candidate(weights, candidate, direction)


def test_opc_requires_input_direction_and_subspace_uses_tabula_recipe():
    weights = _fixture()
    direction = {0: np.ones(5), 1: np.ones(5)}
    rows = catalog(n_layers=2, target_layers=(0, 1), selected_experts={0: (3,)})
    with pytest.raises(CandidateContractError, match="OPC requires"):
        apply_candidate(weights, rows[2], direction)
    output, receipt = apply_candidate(
        weights,
        rows[3],
        direction,
        subspaces={0: np.eye(5)[:, :4], 1: np.eye(5)[:, :4]},
    )
    touched = [row for row in receipt["touched_tensors"] if row["key"].endswith("mlp.down_proj.weight")]
    assert touched
    assert touched[0]["metrics"]["recipe"]["rank"] == 4
    assert output["model.embed_tokens.weight"].shape == (7, 5)


def test_tabula_rank1_partial_projection_is_reversible():
    W = np.arange(20, dtype=np.float64).reshape(5, 4)
    out, recipe, _ = tabula.project(
        W, np.ones(5), norm_preserve=True, store_component=True, scale=0.25)
    assert recipe is not None
    assert np.allclose(recipe.apply(out), W)


def test_closed_refusal_family_is_refused_and_only_explicit_reproduction_bypasses():
    refused = refusal_intervention_preflight("OPH")
    assert refused["allowed"] is False
    assert refused["status"] == "NEGATIVE_SCIENCE_REFUSED"
    assert refused["scar"]["source_path"] == "receipts/future/SOVEREIGN_NEGATIVE_SCIENCE.json"

    reproduction = refusal_intervention_preflight(
        "OPH", reproduce_scarred_family=True)
    assert reproduction["allowed"] is True
    assert reproduction["status"] == "SCARRED_FAMILY_REPRODUCTION_ONLY"
    assert reproduction["reproduction_only"] is True
    assert "cannot reopen" in reproduction["claim_boundary"]


def test_materially_different_policy_hypothesis_remains_open():
    result = refusal_intervention_preflight(
        "NOVA_POLICY_ADAPTER",
        hypothesis_family="conditional supervised policy adapter",
    )
    assert result["allowed"] is True
    assert result["status"] == "HYPOTHESIS_OPEN"
    assert result["scar"] is None


def test_tabula_metal_probe_does_not_hardcode_the_original_host(monkeypatch):
    def run(argv, **_kwargs):
        if argv[:2] == ["system_profiler", "SPDisplaysDataType"]:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "Chipset Model: Apple M3 Ultra\n"
                    "Metal Support: Metal 4\n"
                ),
            )
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(tabula.subprocess, "run", run)
    environment = tabula.physical_environment()
    blockers = tabula.physical_blockers_today(environment)

    assert environment["metal_capable_gpu"] is True
    assert environment["chipset"] == "Apple M3 Ultra"
    assert environment["metal_compiler"] is False
    assert not any("does not report a Metal-capable GPU" in row for row in blockers)
    assert any("cannot locate the Metal compiler" in row for row in blockers)
