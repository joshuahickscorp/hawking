"""Causal option substrate: refusals are the guards, live numbers are the test.

A hypothesis without a falsifier, a PARK without a wake, a SCAR without
scope/REOPEN_IF, or a verb in place of a command is not a substrate. The
receipt's tree must be populated from the landed campaign receipts, and the
ALU ingest must be able to say MIXED — strong evidence for B, rule not
satisfied — without collapsing to a pass or a fail.
"""
from __future__ import annotations

import json

import pytest

from tools.future import improvement_metabolism as im
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    _assert_no_hardware_claims,
)


def _ok_kwargs(**overrides):
    base = dict(
        id="t.node",
        title="a real hypothesis",
        prior_confidence=0.4,
        max_possible_gain_ms=1.0,
        cheapest_decisive_experiment=(
            "python3 tools/future/mlp_alu_roofline.py --measure --record"
        ),
        expected_runtime_s=10.0,
        required_resource="CPU",
        falsifier="ARM A stays within 1.12x of production",
    )
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def metab():
    return im.campaign(apply_landed=True)


@pytest.fixture(scope="module")
def receipt_path():
    return im.build()


@pytest.fixture(scope="module")
def doc(receipt_path):
    return json.loads(receipt_path.read_text())


# ---------------------------------------------------------------------------
# Refusals. A guard nobody has watched fail is decoration.
# ---------------------------------------------------------------------------


def test_hypothesis_without_falsifier_is_refused():
    with pytest.raises(im.NotAHypothesis, match="falsifier"):
        im.hypothesis(**_ok_kwargs(falsifier=""))
    with pytest.raises(im.NotAHypothesis, match="falsifier"):
        im.hypothesis(**_ok_kwargs(falsifier=None))
    with pytest.raises(im.NotAHypothesis, match="falsifier"):
        im.hypothesis(**_ok_kwargs(falsifier="   "))


def test_verb_experiment_is_refused():
    for verb in (
        "investigate",
        "investigate the mlp bandwidth",
        "explore",
        "explore cache behaviour",
        "analyze the occupancy",
        "consider a cheaper decode",
        "look into register pressure",
        "research topology",
    ):
        with pytest.raises(im.VerbExperiment):
            im.hypothesis(**_ok_kwargs(cheapest_decisive_experiment=verb))
    with pytest.raises(im.VerbExperiment, match="empty"):
        im.hypothesis(**_ok_kwargs(cheapest_decisive_experiment=""))
    with pytest.raises(im.VerbExperiment):
        im.work_unit(
            id="wu.bad",
            role=im.PROBE,
            hypothesis_id="t.node",
            frontier_id="mlp_execution",
            experiment="investigate",
            required_resource="CPU",
            expected_runtime_s=1.0,
        )


def test_command_and_probe_experiments_are_accepted():
    cmd = im.hypothesis(
        **_ok_kwargs(
            cheapest_decisive_experiment=(
                "python3 tools/future/mlp_alu_roofline.py --measure --record"
            )
        )
    )
    assert cmd.cheapest_decisive_experiment.startswith("python3")
    probe = im.hypothesis(
        **_ok_kwargs(
            id="t.probe",
            cheapest_decisive_experiment=(
                "matched pair ARM A stripped vs production on layer 0, "
                "weight_bytes=83558400, MTLCommandBuffer GPUStartTime/GPUEndTime"
            ),
        )
    )
    assert "weight_bytes=83558400" in probe.cheapest_decisive_experiment


def test_park_without_wake_is_refused():
    with pytest.raises(im.ParkWithoutWake):
        im.terminal(im.PARK)
    with pytest.raises(im.ParkWithoutWake):
        im.terminal(im.PARK, wake_condition="")
    with pytest.raises(im.ParkWithoutWake):
        im.terminal(im.PARK, wake_condition=None)
    node = im.hypothesis(**_ok_kwargs(id="t.park"))
    with pytest.raises(im.ParkWithoutWake):
        im.apply_terminal(node, {"kind": im.PARK, "wake_condition": ""})
    with pytest.raises(im.ParkWithoutWake):
        im.hypothesis(**_ok_kwargs(id="t.parked", status=im.PARKED, wake_condition=""))


def test_park_with_wake_is_accepted():
    term = im.terminal(
        im.PARK,
        wake_condition="protected GPU lease held AND alu_roofline_organs present",
    )
    assert term["kind"] == im.PARK
    assert "GPU lease" in term["wake_condition"]
    node = im.hypothesis(**_ok_kwargs(id="t.park.ok"))
    im.apply_terminal(node, term)
    assert node.status == im.PARKED
    assert node.wake_condition


def test_scar_without_scope_or_reopen_if_is_refused():
    with pytest.raises(im.ScarIncomplete):
        im.terminal(im.SCAR, scope="mlp", reopen_if="")
    with pytest.raises(im.ScarIncomplete):
        im.terminal(im.SCAR, scope="", reopen_if="a different algebra")
    with pytest.raises(im.ScarIncomplete):
        im.terminal(im.SCAR, scope=None, reopen_if=None)
    node = im.hypothesis(**_ok_kwargs(id="t.scar"))
    with pytest.raises(im.ScarIncomplete):
        im.kill_hypothesis(node, reopen_if="")
    with pytest.raises(im.ScarIncomplete):
        im.hypothesis(
            **_ok_kwargs(id="t.killed", status=im.KILLED, reopen_if=None)
        )


def test_scar_with_scope_and_reopen_is_accepted():
    term = im.terminal(
        im.SCAR,
        scope="sealed-3.14 MLP F, r-bottleneck families",
        reopen_if="full-width structured operator that is not an r-bottleneck",
    )
    assert term["kind"] == im.SCAR
    node = im.hypothesis(**_ok_kwargs(id="t.scar.ok"))
    im.apply_terminal(node, term)
    assert node.status == im.KILLED
    assert "r-bottleneck" in (node.reopen_if or "")


def test_keep_and_rollback_are_the_other_two_terminals_and_nothing_else():
    assert im.terminal(im.KEEP)["kind"] == im.KEEP
    assert im.terminal(im.ROLLBACK, restores="prior kernel")["kind"] == im.ROLLBACK
    with pytest.raises(im.UnknownTerminal):
        im.terminal("MAYBE")
    with pytest.raises(im.UnknownTerminal):
        im.terminal("LIMBO")
    assert set(im.TERMINALS) == {im.KEEP, im.ROLLBACK, im.PARK, im.SCAR}


def test_unknown_scientific_role_is_refused():
    with pytest.raises(im.UnknownRole):
        im.work_unit(
            id="wu.nope",
            role="BUSYWORK",
            hypothesis_id="t.node",
            frontier_id="mlp_execution",
            experiment="python3 tools/future/mlp_alu_roofline.py --record",
            required_resource="CPU",
            expected_runtime_s=1.0,
        )


# ---------------------------------------------------------------------------
# Shape: trees nest, frontiers answer why, roles balance.
# ---------------------------------------------------------------------------


def test_nodes_nest_and_refuse_duplicate_ids():
    leaf = im.hypothesis(**_ok_kwargs(id="root.a1", title="A1"))
    root = im.hypothesis(**_ok_kwargs(id="root", title="root", children=(leaf,)))
    assert [n.id for n in root.walk()] == ["root", "root.a1"]
    assert leaf.parent_id == "root"
    assert root.find("root.a1") is leaf
    clone = im.hypothesis(**_ok_kwargs(id="root.a1", title="dup"))
    with pytest.raises(im.NotAHypothesis, match="duplicate"):
        im.attach(root, clone)


def test_role_balance_flags_mutations_without_a_falsifier():
    mutations = [
        im.work_unit(
            id=f"wu.mut.{i}",
            role=im.MUTATION,
            hypothesis_id="mlp.fn.full_width_structured",
            frontier_id="mlp_function_replacement",
            experiment=(
                "python3 tools/future/mlp_nonlinear_program.py --build "
                f"--candidate monarch_{i}"
            ),
            required_resource="CPU",
            expected_runtime_s=60.0,
        )
        for i in range(20)
    ]
    bal = im.role_balance(mutations)
    assert bal.n == 20
    assert bal.counts[im.MUTATION] == 20
    assert bal.counts[im.FALSIFIER] == 0
    assert bal.unbalanced is True
    assert im.FALSIFIER in bal.missing_roles
    assert "no falsifier" in bal.note


def test_frontier_answers_why_it_exists_without_a_human(metab):
    fr = metab.frontiers["mlp_execution"]
    why = fr.why_it_exists()
    assert why["objective"]
    assert "330" in fr.objective or "329.6" in fr.objective or "mlp" in fr.objective.lower()
    assert why["current_best"]
    assert why["target"]
    assert why["remaining_gap"]
    assert why["biggest_unknown"]
    assert why["n_hypotheses"] >= 12  # root + A/B/C/D + nested
    assert "A1" in json.dumps(fr.causal_hypotheses.to_dict())


# ---------------------------------------------------------------------------
# Ingest of the landed ALU receipt: MIXED, not a pass, not a fail.
# ---------------------------------------------------------------------------


def test_alu_ingest_promotes_B_demotes_A_and_records_mixed(metab):
    log = [r for r in metab.ingest_log if r.receipt_name == "MLP_ALU_ROOFLINE.json"]
    assert log, "campaign must ingest the landed ALU receipt"
    result = log[-1]
    assert result.changed is True
    assert result.nothing_changed is False
    assert result.defect is None
    assert result.formal_verdict == "MIXED"
    assert result.nuance
    assert "strong evidence for B" in result.nuance
    assert "rule not satisfied" in result.nuance

    tree = metab.frontiers["mlp_execution"].causal_hypotheses
    assert tree.id == "mlp.why_330"
    assert tree.status == im.MIXED
    assert tree.evidence_for is True
    assert tree.rule_satisfied is False
    assert tree.formal_verdict == "MIXED"

    a = tree.find("mlp.why_330.A")
    b = tree.find("mlp.why_330.B")
    b1 = tree.find("mlp.why_330.B1")
    a1 = tree.find("mlp.why_330.A1")
    a3 = tree.find("mlp.why_330.A3")
    assert a is not None and a.status == im.DEMOTED
    assert a1 is not None and a1.status == im.DEMOTED
    assert a3 is not None and a3.status == im.DEMOTED
    a3_note = (a3.evidence[-1]["note"] if a3.evidence else a3.notes).lower()
    assert "occupancy" in a3_note or "threadgroup" in a3_note
    assert b is not None and b.status == im.MIXED
    assert b.evidence_for is True
    assert b.rule_satisfied is False
    assert b.formal_verdict == im.MIXED
    assert "strong evidence for B, rule not satisfied" in (b.notes or "")
    assert b1 is not None and b1.status == im.PROMOTED

    assert "mlp.why_330.A" in result.demoted
    assert "mlp.why_330.A1" in result.demoted
    assert "mlp.why_330.A2" in result.demoted
    assert "mlp.why_330.A3" in result.demoted
    assert "mlp.why_330.B1" in result.promoted
    assert "mlp.why_330.B" in result.mixed
    assert "mlp.why_330" in result.mixed

    # B2/B3/C/D are not collapsed by a MIXED parent.
    for nid in (
        "mlp.why_330.B2",
        "mlp.why_330.B3",
        "mlp.why_330.C",
        "mlp.why_330.C1",
        "mlp.why_330.C2",
        "mlp.why_330.D",
    ):
        node = tree.find(nid)
        assert node is not None
        assert node.status == im.OPEN, nid


def test_mixed_is_not_alu_bound_and_not_memory_bound(metab):
    cited = metab.cited["mlp_alu"]
    assert cited["verdict"] == "MIXED"
    assert cited["mlp_verdict"] == "MIXED"
    assert cited["arm_a_over_production"] >= 1.5
    assert cited["arm_b_sublinear"] is False
    assert cited["arm_b_linear"] is True
    assert cited["loads_survived"] is True
    why = cited["why_not_forced"]
    assert "half K also halves FMAs" in why or "halves FMAs" in why
    # The substrate must represent the nuance, not collapse it.
    b = metab.frontiers["mlp_execution"].causal_hypotheses.find("mlp.why_330.B")
    assert b.status not in {im.SETTLED, im.KILLED, im.PROMOTED, im.DEMOTED}
    assert b.status == im.MIXED


def test_reingest_of_the_same_receipt_is_an_explicit_defect(metab):
    alu = im.load_receipt("receipts/future/MLP_ALU_ROOFLINE.json")
    again = im.ingest(metab, alu)
    assert again.changed is False
    assert again.nothing_changed is True
    assert again.defect == im.INGEST_CHANGED_NOTHING
    assert again.promoted == []
    assert again.demoted == []


def test_ingest_of_an_unrelated_receipt_is_an_explicit_defect(metab):
    result = im.ingest(metab, {"schema": "hawking.future.not_a_thing.v1", "hello": 1})
    assert result.changed is False
    assert result.defect == im.INGEST_CHANGED_NOTHING


# ---------------------------------------------------------------------------
# Receipt is populated from the LIVE campaign, real numbers.
# ---------------------------------------------------------------------------


def test_build_seals_a_static_only_receipt(receipt_path, doc):
    assert receipt_path.parent == RECEIPTS
    assert receipt_path.name == im.RECEIPT
    assert doc["schema"] == im.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["tenet"] == im.TENET
    assert doc["law"] == im.LAW
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        assert key not in doc or doc[key] in (None, "UNKNOWN")


def test_receipt_tree_is_the_why_330_instance_with_real_alu_numbers(doc):
    tree = doc["mlp_330_tree"]
    assert tree["title"] == "WHY IS THE MLP AT ~330 GB/s?"
    titles = []

    def collect(node):
        titles.append(node["title"])
        for c in node.get("children") or []:
            collect(c)

    collect(tree)
    joined = " ".join(titles)
    assert "A bandwidth ceiling" in joined
    assert "A1 raw DRAM" in joined
    assert "A2 transaction inefficiency" in joined
    assert "A3 cache behaviour" in joined
    assert "B arithmetic ceiling" in joined
    assert "B1 unpack/decode" in joined
    assert "B2 conversion" in joined
    assert "B3 accumulation" in joined
    assert "C dependency ceiling" in joined
    assert "C1 instruction chain" in joined
    assert "C2 register pressure" in joined
    assert "D physical execution topology" in joined

    cited = doc["cited"]["mlp_alu"]
    assert cited["production_gb_s"] == 329.6
    assert cited["arm_a_gb_s"] == 497.4
    assert cited["arm_a_over_production"] == 1.5089
    assert cited["weight_bytes"] == 83_558_400
    assert cited["weight_mb"] == 83.56
    assert cited["production_decode_fma_per_weight_byte"] == 1.3333
    assert cited["target_decode_fma_per_weight_byte_at_497"] == 0.8835
    assert cited["threads_per_threadgroup"] == 128
    assert cited["max_total_threads_per_threadgroup"] == 1024
    assert cited["threadgroups_per_core"] == pytest.approx(145.06666666666666)
    assert cited["occupancy_limited"] is False
    assert cited["loads_survived"] is True
    assert cited["verdict"] == "MIXED"
    assert cited["arm_b_sublinear"] is False

    what = doc["what_the_alu_result_did"]
    assert what["formal_verdict"] == "MIXED"
    assert what["collapsed_to_ALU_BOUND"] is False
    assert what["not_a_pass"] is True
    assert what["not_a_fail"] is True
    assert "mlp.why_330.A" in what["demoted"]
    assert "mlp.why_330.B1" in what["promoted"]
    assert "mlp.why_330.B" in what["mixed"]
    assert "strong evidence for B" in (what["nuance"] or "")


def test_receipt_cites_the_causal_budget_and_path_to_71(doc):
    budget = doc["cited"]["causal_budget"]
    # The RECONSTRUCTION moves whenever the budget's accounting is corrected -
    # 28.722 with no unattributed term, 29.043 with a wrong 0.321, 28.817 with the
    # measured 0.095 remainder. This test's own name says it checks that the
    # receipt CITES the budget, so compare against the budget rather than against
    # whichever value it held the day this was written.
    from tools.future import causal_budget_71 as _cb
    _now = _cb.token_ms()
    assert budget["cited_token_ms"] == pytest.approx(round(_now, 3), abs=1e-3)
    assert budget["cited_tps"] == pytest.approx(round(_cb.tps(_now), 2), abs=1e-2)
    # The ORGAN measurements are pinned on purpose: those are measured constants,
    # and if one moves this test SHOULD fire.
    assert budget["mlp_organ_ms"] == 15.541
    assert budget["mlp_organ_gb_s"] == 344.1
    assert budget["deltanet_organ_ms"] == 8.227
    assert budget["deltanet_organ_gb_s"] == 360.0
    assert budget["mlp_ms_saved_at_497"] == 4.79
    path = doc["cited"]["path_to_71"]
    assert path["best_composed_path"] == "PATH_04"
    assert path["best_composed_cited_tps"] == 42.36
    assert path["still_to_remove_ms"] == 9.52
    assert path["target_token_ms"] == 14.085


def test_deltanet_unexplained_cost_is_arithmetic_over_cited_figures(doc):
    iso = doc["cited"]["deltanet_isolation"]
    assert iso["organ_gb_s"] == 360.0
    assert iso["organ_ms"] == 8.227
    assert iso["organ_gb"] == 2.961659904
    assert iso["isolated_kernel_gb_s"] == 600.9
    expected_cf = round(2.961659904 / 600.9 * 1000.0, 3)
    expected_gap = round(8.227 - expected_cf, 3)
    assert iso["counterfactual_organ_ms_at_isolated_rate"] == expected_cf
    assert iso["unexplained_ms"] == expected_gap
    assert iso["unexplained_ms"] == pytest.approx(3.298, abs=0.01)
    fr = doc["frontiers"]["deltanet_execution"]
    assert "360" in fr["objective"]
    assert "600.9" in fr["objective"]


def test_six_r_bottleneck_families_are_scarred_and_full_width_is_the_reopen(doc):
    fn = doc["cited"]["function_replacement"]
    assert fn["n_nonlinear_families"] == 6
    assert set(fn["families"]) == {
        "FACTORIZE_THE_FACTORS",
        "DICTIONARY_PROGRAM",
        "PRODUCT_DICTIONARY",
        "CONDITIONAL_PROGRAM",
        "GENERATED_BLOCK",
        "NONLINEAR_GENERATOR",
    }
    assert fn["n_survivors_nonlinear"] == 0
    assert fn["n_survivors_shared"] == 0
    assert fn["shared_candidates_measured_negative"] == 17
    assert fn["oracle_pca_r64_held_out_relative_l2"] == pytest.approx(0.895354)
    assert fn["affordable_f16_rank_cap"] == 617
    assert fn["uses_essentially_all_of_W"] == "MEASURED_NEGATIVE"

    fr = doc["frontiers"]["mlp_function_replacement"]
    assert len(fr["scars"]) == 6
    for scar in fr["scars"]:
        assert scar["reopen_if"]
        assert scar["scope"]
        assert scar["verdict"] == "MEASURED_NEGATIVE"
    titles = json.dumps(fr["causal_hypotheses"])
    assert "full-width structured" in titles
    # Killed families carry REOPEN_IF.
    def walk(n):
        yield n
        for c in n.get("children") or []:
            yield from walk(c)

    killed = [n for n in walk(fr["causal_hypotheses"]) if n["status"] == im.KILLED]
    assert len(killed) == 6
    for n in killed:
        assert n["reopen_if"]


def test_odyssey_gate_is_cited_from_the_sealed_receipt(doc):
    gate = doc["cited"]["odyssey_gate"]
    src = im.load_receipt("receipts/future/ODYSSEY_LAUNCH_GATE.json")
    v = src["verdict"]
    assert gate["n_criteria"] == v["n_criteria"] == 16
    assert gate["n_met"] == v["n_met"]
    assert gate["n_unmet"] == v["n_unmet"]
    assert gate["unmet"] == v["unmet"]
    assert gate["verdict"] == v["verdict"]
    fr = doc["frontiers"]["odyssey_gate"]
    assert str(v["n_met"]) in fr["objective"] or str(v["n_criteria"]) in fr["objective"]
    # Every unmet criterion is a nested node.
    ids = []

    def walk(n):
        ids.append(n["id"])
        for c in n.get("children") or []:
            walk(c)

    walk(fr["causal_hypotheses"])
    for cid in v["unmet"]:
        assert f"odyssey.unmet.{cid}" in ids


def test_live_queue_has_all_seven_roles_and_a_falsifier(doc, metab):
    bal = doc["role_balance"]
    assert bal["unbalanced"] is False
    assert bal["counts"][im.FALSIFIER] >= 1
    assert bal["counts"][im.MUTATION] >= 1
    for role in im.ROLES:
        assert bal["counts"][role] >= 1, role
    # PARK on the keep-K falsifier carries a wake condition.
    parked = [w for w in doc["work_units"] if (w.get("terminal") or {}).get("kind") == im.PARK]
    assert parked
    for w in parked:
        assert w["terminal"]["wake_condition"]


def test_public_surface_is_small_and_named():
    names = set(im.__all__)
    # x2 imports these. Do not silently rename.
    for required in (
        "hypothesis",
        "frontier",
        "work_unit",
        "terminal",
        "ingest",
        "role_balance",
        "campaign",
        "build",
        "KEEP",
        "ROLLBACK",
        "PARK",
        "SCAR",
        "PROBE",
        "FALSIFIER",
        "ORACLE",
        "MUTATION",
        "REPLICATION",
        "QUALIFICATION",
        "ADVERSARY",
        "MIXED",
        "Hypothesis",
        "Frontier",
        "WorkUnit",
        "IngestResult",
        "Metabolism",
        "NotAHypothesis",
        "VerbExperiment",
        "ParkWithoutWake",
        "ScarIncomplete",
    ):
        assert required in names, required
    assert "improvement_trial" not in names


def test_campaign_fails_closed_when_a_required_receipt_is_missing(monkeypatch):
    real = im.load_receipt_origin

    def missing(rel):
        if rel.endswith("MLP_ALU_ROOFLINE.json"):
            return None, "unseen_in_this_checkout"
        return real(rel)

    monkeypatch.setattr(im, "load_receipt_origin", missing)
    with pytest.raises(im.MissingReceipt, match="MLP_ALU_ROOFLINE"):
        im.campaign(apply_landed=False)


def test_every_node_in_the_live_tree_has_a_falsifier_and_a_non_verb_experiment(metab):
    for fr in metab.frontiers.values():
        for node in fr.causal_hypotheses.walk():
            assert node.falsifier.strip(), node.id
            im.require_experiment(node.cheapest_decisive_experiment)
            if node.status == im.KILLED:
                assert node.reopen_if
            if node.status == im.PARKED:
                assert node.wake_condition
