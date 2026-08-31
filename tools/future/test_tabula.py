"""Tests for the Tabula independent-evaluation floor.

Negative controls are the point: rank() must refuse a behavioral-only order,
a transformation that hits its behavioral target while regressing tool use
must score FAILURE, and Tabula must not be able to widen authority.

Never assert that a recovered file is absent. Sparse checkout is not
evidence. Assert the module copes with either state and records the path taken.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from hcli.workunit import WorkUnit, is_ready
from tools.future import tabula as tb
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


def test_build_and_selftest_emit_sealed_receipt():
    out = tb.selftest()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "TABULA_FLOOR.json"
    assert doc["schema"] == "hawking.future.tabula.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["status"] == "BUILT_NOT_PROMOTED"
    assert doc["promoted"] is False
    assert doc["weights_modified"] is False
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    assert doc["resident_callable"]["entry_point"]
    assert doc["resident_callable"]["workunit_emitted"]
    assert doc["resident_callable"]["receipt"] == "receipts/future/TABULA_FLOOR.json"
    assert doc["resident_callable"]["frontier_fed"]["feeds"].endswith("CLAUDE_GLOBAL_FRONTIER.json")
    assert doc["resident_callable"]["fail_closed"]
    assert doc["resident_callable"]["frontier_fed"]["this_lane_writes_frontier"] is False
    _assert_no_hardware_claims(doc)


def test_vocabulary_is_five_eras_three_odysseys():
    assert len(tb.ERAS) == 5
    assert len(tb.ODYSSEYS) == 3
    assert "VI" not in "".join(tb.ERAS)
    assert "IV" not in "".join(tb.ODYSSEYS)
    doc = json.loads(tb.build().read_text())
    assert doc["vocabulary"]["no_era_vi"] is True
    assert doc["vocabulary"]["no_odyssey_iv"] is True
    assert doc["vocabulary"]["zero_refusal_is_never_the_only_score"] is True
    assert doc["vocabulary"]["permission_is_not_personality"] is True


def test_catalog_covers_every_kind_and_is_reproducible():
    a = tb.catalog(seed=0)
    b = tb.catalog(seed=0)
    kinds = {c.kind for c in a}
    assert kinds == set(tb.CONTRACT_KINDS)
    assert [c.identity() for c in a] == [c.identity() for c in b]
    for contract in a:
        assert isinstance(contract.seed, int)
        assert contract.inputs
        assert contract.null
        assert "seed" in contract.null
        body = contract.to_dict()
        assert body["weights_modified"] is False
        assert body["evidence_class"] == "STATIC_ONLY"
        assert body["bench_state"] == "UNKNOWN"
    different = tb.catalog(seed=1)
    assert [c.identity() for c in different] != [c.identity() for c in a]


def test_make_contract_requires_seed_inputs_null():
    with pytest.raises(tb.ExperimentContractError, match="declared inputs"):
        tb.make_contract(
            id="x",
            kind="orthogonal_projection",
            seed=0,
            inputs={},
            null={"seed": 1, "kind": "random"},
            statement="x",
        )
    with pytest.raises(tb.ExperimentContractError, match="declared null"):
        tb.make_contract(
            id="x",
            kind="orthogonal_projection",
            seed=0,
            inputs={"out_dim": 4},
            null={},
            statement="x",
        )
    with pytest.raises(tb.ExperimentContractError, match="null must declare"):
        tb.make_contract(
            id="x",
            kind="orthogonal_projection",
            seed=0,
            inputs={"out_dim": 4},
            null={"kind": "random"},
            statement="x",
        )
    with pytest.raises(tb.ExperimentContractError, match="unknown kind"):
        tb.make_contract(
            id="x",
            kind="abliterate_for_fun",
            seed=0,
            inputs={"out_dim": 4},
            null={"seed": 1},
            statement="x",
        )


def test_geometry_projection_null_layer_norm_and_invert():
    contracts = {c.id: c for c in tb.catalog(seed=0)}
    proofs = {p["contract_id"]: p for p in (tb.run_contract(c) for c in contracts.values())}

    orth = proofs["TAB-ORTH-001"]
    dest = orth["destination_layers"]
    assert dest
    for L in dest:
        residual = orth["metrics_by_layer"][str(L)]["residual_vT_W_out"]
        parent = orth["metrics_by_layer"][str(L)]["residual_vT_W_parent"]
        assert residual < 1e-8
        assert parent > residual
        rec = orth["recovered_direction"][str(L)]
        assert rec["abs_cos_with_v"] > 0.9
        assert rec["abs_cos_with_null"] < rec["abs_cos_with_v"]

    layer = proofs["TAB-LAYER-001"]
    assert layer["off_destination_unchanged"] is True

    normed = proofs["TAB-NORM-001"]
    for L in normed["destination_layers"]:
        met = normed["metrics_by_layer"][str(L)]
        assert met["norm_preserve_error"] < 1e-8

    rev = proofs["TAB-REV-001"]
    assert rev["reversible"] is True
    assert rev["invert_frobenius_error"] is not None
    assert rev["invert_frobenius_error"] < 1e-9
    assert all(item.get("stores") for item in rev["invert"])

    irr = proofs["TAB-IRR-001"]
    assert irr["reversible"] is False
    assert irr["invert_frobenius_error"] is None
    assert irr["authority_required"] == "destructive_mutation"


def test_evaluate_behavior_hit_with_tool_use_regression_is_failure():
    """Negative control: hitting the behavioral target while destroying tool use is FAILURE."""
    vec = tb.ScoreVector(
        behavioral=0.95,
        capability=0.10,
        tool_use=-0.80,
        reasoning=0.05,
        instruction_following=0.04,
    )
    verdict = tb.evaluate(vec)
    assert verdict.outcome == "FAILURE"
    assert verdict.target_hit is True
    assert "tool_use" in verdict.regressions
    assert "destroying" in verdict.reason or "regressing" in verdict.reason

    balanced = tb.ScoreVector(
        behavioral=0.70,
        capability=0.05,
        tool_use=0.02,
        reasoning=0.01,
        instruction_following=0.00,
    )
    ok = tb.evaluate(balanced)
    assert ok.outcome == "PASS"
    assert ok.regressions == ()


def test_evaluate_refuses_refusal_rate_as_the_only_score():
    with pytest.raises(tb.IncompleteScoreVector, match="zero-refusal"):
        tb.scores_from_refusal_rate(0.0)
    with pytest.raises(tb.IncompleteScoreVector, match="missing axes"):
        tb.ScoreVector.from_mapping({"behavioral": 1.0})
    with pytest.raises(tb.IncompleteScoreVector, match="missing axes"):
        tb.evaluate({"behavioral": 1.0, "capability": 0.0})


def test_rank_refuses_behavioral_axis_alone():
    """Negative control: rank() refuses to order on the behavioral axis alone."""
    with pytest.raises(tb.RankRefusal, match="behavioral axis alone"):
        tb.rank(
            [
                {"id": "a", "scores": {"behavioral": 0.9}},
                {"id": "b", "scores": {"behavioral": 0.1}},
            ]
        )
    full_same_nb = [
        {
            "id": "a",
            "scores": {
                "behavioral": 0.9,
                "capability": 0.2,
                "tool_use": 0.2,
                "reasoning": 0.2,
                "instruction_following": 0.2,
            },
        },
        {
            "id": "b",
            "scores": {
                "behavioral": 0.1,
                "capability": 0.2,
                "tool_use": 0.2,
                "reasoning": 0.2,
                "instruction_following": 0.2,
            },
        },
    ]
    with pytest.raises(tb.RankRefusal, match="behavioral axis alone"):
        tb.rank(full_same_nb)
    with pytest.raises(tb.RankRefusal, match="behavioral axis alone"):
        tb.rank(full_same_nb, on=("behavioral",))


def test_rank_orders_pass_before_failure_using_full_vector():
    kill = {
        "id": "kill-tools",
        "scores": {
            "behavioral": 0.95,
            "capability": 0.10,
            "tool_use": -0.80,
            "reasoning": 0.05,
            "instruction_following": 0.04,
        },
    }
    balanced = {
        "id": "balanced",
        "scores": {
            "behavioral": 0.70,
            "capability": 0.05,
            "tool_use": 0.02,
            "reasoning": 0.01,
            "instruction_following": 0.00,
        },
    }
    ranked = tb.rank([kill, balanced])
    assert [r["id"] for r in ranked] == ["balanced", "kill-tools"]
    assert ranked[0]["verdict"]["outcome"] == "PASS"
    assert ranked[1]["verdict"]["outcome"] == "FAILURE"
    assert "tool_use" in ranked[1]["verdict"]["regressions"]


def test_tabula_cannot_widen_its_own_authority():
    """Negative control: the authority lattice — not the model — gates action."""
    floor = tb.TabulaFloor()
    with pytest.raises(tb.AuthorityError, match="cannot widen"):
        floor.widen_authority("external_action")
    with pytest.raises(tb.AuthorityError, match="cannot grant"):
        floor.lattice.grant("external_action")
    with pytest.raises(tb.AuthorityError, match="cannot widen"):
        floor.lattice.widen("fit_weights")
    with pytest.raises(tb.AuthorityError, match="cannot assign"):
        floor.lattice._held = frozenset({"external_action"})
    with pytest.raises(tb.AuthorityError, match="cannot assign"):
        floor.lattice = tb.AuthorityLattice()
    with pytest.raises(tb.AuthorityError, match="forbidden"):
        tb.AuthorityLattice(held={"read_receipts", "external_action"})
    with pytest.raises(tb.AuthorityError, match="unknown authority"):
        tb.AuthorityLattice(held={"read_receipts", "quietly_install_resident"})
    assert not hasattr(tb.TabulaFloor, "promote")
    with pytest.raises(AttributeError):
        floor.promote()  # type: ignore[attr-defined]


def test_permission_is_not_personality():
    lattice = tb.AuthorityLattice()
    assert (
        tb.may_external_action(lattice, model_willingness=True, refusal_rate=0.0) is False
    )
    assert tb.may_external_action(lattice, model_willingness=False, refusal_rate=1.0) is False
    parent = tb.Specimen(
        specimen_id="parent-0",
        personality={"willing_external_action": False, "refusal_rate": 0.4},
        lattice=lattice,
    )
    contract = next(c for c in tb.catalog(seed=0) if c.id == "TAB-REV-001")
    scores = tb.ScoreVector(0.7, 0.05, 0.02, 0.01, 0.0)
    child = tb.transform_specimen(
        parent,
        contract,
        personality_delta={"willing_external_action": True, "refusal_rate": 0.0},
        scores=scores,
    )
    assert child.personality["willing_external_action"] is True
    assert child.personality["refusal_rate"] == 0.0
    assert child.lattice is parent.lattice
    assert tb.may_external_action(child.lattice, model_willingness=True) is False
    assert child.lineage is not None
    assert child.lineage["method"] == "tabula_transformation"
    assert child.lineage["parent_id"] == "parent-0"
    assert child.lineage["reversible"] is True
    with pytest.raises(tb.AuthorityError, match="cannot widen authority"):
        tb.transform_specimen(
            parent,
            contract,
            personality_delta={"willing_external_action": True},
            scores=scores,
            authority_delta={"held": ["external_action"]},
        )
    with pytest.raises(tb.AuthorityError, match="smuggle"):
        tb.transform_specimen(
            parent,
            contract,
            personality_delta={"external_action": True},
            scores=scores,
        )


def test_irreversible_child_requires_higher_authority():
    lattice = tb.AuthorityLattice()
    irr = next(c for c in tb.catalog(seed=0) if c.id == "TAB-IRR-001")
    assert irr.reversible is False
    with pytest.raises(tb.IrreversibleAuthorityError, match="higher authority|destructive_mutation"):
        tb.emit_child(
            parent_id="parent-0",
            contract=irr,
            scores=tb.ScoreVector(0.9, 0.1, 0.1, 0.1, 0.1),
            lattice=lattice,
        )
    rev = next(c for c in tb.catalog(seed=0) if c.id == "TAB-REV-001")
    child = tb.emit_child(
        parent_id="parent-0",
        contract=rev,
        scores=tb.ScoreVector(0.7, 0.05, 0.02, 0.01, 0.0),
        lattice=lattice,
        invert_doc={"method": "unscale_then_add_outer(v, vT_W)"},
    )
    assert child["lineage"]["reversible"] is True
    assert child["weights_modified"] is False
    assert child["lineage"]["evidence_class"] == "STATIC_ONLY"


def test_weights_are_frozen():
    with pytest.raises(tb.WeightsFrozen, match="No weights are modified"):
        tb.apply_to_weights()
    with pytest.raises(tb.WeightsFrozen):
        tb.TabulaFloor().apply_to_weights("any-tensor")


def test_sleeping_fit_unit_is_not_ready_and_round_trips_hcli():
    floor = tb.TabulaFloor()
    contracts = tb.catalog(seed=0)
    capture = tb.teacher_capture_progress()
    units = tb.emit_workunits(contracts=contracts, lattice=floor.lattice, capture=capture)
    ids = [row["id"] for row in units]
    assert "future.tabula.floor" in ids
    assert "future.tabula.independent-eval" in ids
    assert "future.tabula.fit-weights" in ids
    by_id = {row["id"]: row for row in units}
    fit = by_id["future.tabula.fit-weights"]
    assert fit["status"] == "sleeping"
    assert fit["classification"] == "SLEEPING"
    assert fit["sleep_state"] == "SLEEPING"
    assert fit["wake_condition"]
    assert fit["blocked_reason"]
    assert fit["weights_modified"] is False
    assert fit["may_widen_authority"] is False
    mapped = {row["id"]: WorkUnit.from_dict(dict(row)) for row in units}
    assert mapped[fit["id"]].status == "sleeping"
    assert is_ready(mapped[fit["id"]], mapped) is False
    assert tb.sleeping_unit_is_not_ready(units) is True
    # Pending floor unit with no deps is ready; the sleeping one is not.
    assert is_ready(mapped["future.tabula.floor"], mapped) is True
    wake = fit["wake_condition"]["teacher_capture"]
    assert wake["receipt"] == tb.TEACHER_CORPUS_REL
    # Derived, not hard-coded: whatever the receipt says, the rule is equality.
    if wake.get("units"):
        assert ("complete" in wake) and (
            wake["complete"] is (wake["executed_units"] == wake["units"])
        )


def test_recovery_copes_with_sparse_or_full_checkout():
    rows = tb.recover_tabula()
    assert rows
    taken = {row["path_taken"] for row in rows}
    assert taken <= {"worktree", "git:HEAD", "unlocated"}
    for row in rows:
        assert "on_disk" in row
        assert "in_head" in row
        assert row["what"]
        if row["path_taken"] == "worktree":
            assert (tb.REPO / row["path"]).is_file()
            text, taken_now = tb._read_text(row["path"])
            assert text is not None
            assert taken_now == "worktree"
        elif row["path_taken"] == "git:HEAD":
            text, taken_now = tb._read_text(row["path"])
            assert taken_now in {"worktree", "git:HEAD"}
            assert text is not None
        else:
            text, taken_now = tb._read_text(row["path"])
            assert taken_now == "unlocated"
            assert text is None
    doctrine = tb.recovered_doctrine()
    assert doctrine["path_taken"] in {"worktree", "git:HEAD", "unlocated"}
    assert "Behavioral freedom and external authority" in doctrine["doctrine"]
    # If the G1 baseline is locatable, the quoted phrase must actually be in it.
    if doctrine["path_taken"] != "unlocated":
        assert doctrine["quoted_phrase_present"] is True


def test_teacher_capture_progress_records_path_taken():
    progress = tb.teacher_capture_progress()
    assert progress["path_taken"] in {"worktree", "git:HEAD", "unlocated"}
    if progress["path_taken"] == "unlocated":
        assert progress["present"] is False
    else:
        assert progress["present"] is True
        assert "units" in progress
        assert "executed_units" in progress
        assert isinstance(progress["target_row_counts"], list)


def test_negative_controls_are_watched_failing():
    results = tb._prove_negative_controls()
    trials = {row["trial"] for row in results}
    assert "rank_behavioral_only" in trials
    assert "behavior_hit_tool_use_regression" in trials
    assert "widen_authority" in trials
    assert "permission_is_not_personality" in trials
    assert "apply_to_weights" in trials
    assert "irreversible_child" in trials
    assert "promote_absent" in trials
    assert all(row["refused"] is True for row in results)
    by = {row["trial"]: row for row in results}
    assert by["behavior_hit_tool_use_regression"]["outcome"] == "FAILURE"
    assert "tool_use" in by["behavior_hit_tool_use_regression"]["regressions"]


def test_receipt_records_resident_callability_and_sleeping_fit():
    doc = json.loads(tb.build().read_text())
    rc = doc["resident_callable"]
    assert rc["entry_point"] == "python3 tools/future/tabula.py --build"
    assert "future.tabula.floor" in rc["workunit_emitted"]
    assert "future.tabula.fit-weights" in rc["workunit_emitted"]
    assert rc["frontier_fed"]["this_lane_writes_frontier"] is False
    assert "fail_closed" in rc
    assert doc["sleeping_fit"]["status"] == "sleeping"
    assert doc["sleeping_fit"]["ready"] is False
    assert doc["promote_exists"] is False
    kinds = {c["kind"] for c in doc["experiment_contracts"]}
    assert kinds == set(tb.CONTRACT_KINDS)
    assert doc["independent_evaluation"]["demo_failure"]["outcome"] == "FAILURE"
    assert "tool_use" in doc["independent_evaluation"]["demo_failure"]["regressions"]
    ranked_ids = [r["id"] for r in doc["independent_evaluation"]["ranked"]]
    assert ranked_ids[0] == "balanced"


def test_synthetic_project_roundtrip_matches_recipe():
    rng = np.random.default_rng(0)
    v = tb._unit(rng, 16)
    W = tb._matrix(rng, 16, 24)
    W_out, recipe, met = tb.project(W, v, norm_preserve=True, store_component=True)
    assert recipe is not None
    restored = recipe.apply(W_out)
    assert float(np.linalg.norm(restored - W, ord="fro")) < 1e-9
    assert met["norm_preserve_error"] < 1e-9
    W_irr, recipe_irr, _ = tb.project(W, v, norm_preserve=False, store_component=False)
    assert recipe_irr is None
    assert float(np.linalg.norm(v @ W_irr)) < 1e-8
