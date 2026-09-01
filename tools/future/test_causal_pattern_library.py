"""A rule that cannot fail to fire is not a rule.

All six rules fire against the current repository, which is expected - they were
derived from the evidence that is on disk. The load-bearing tests are the ones
that feed each condition a plausible ALTERNATIVE state and prove it goes quiet.
"""
from __future__ import annotations

import json

import pytest

from tools.future import causal_pattern_library as cpl


def test_every_rule_fires_on_the_current_evidence():
    d = cpl.build()
    assert d["n_rules"] == 6
    assert d["n_fired"] == 6


def test_a_missing_receipt_refuses_rather_than_reporting_not_fired():
    with pytest.raises(cpl.RuleRefused, match="false negative"):
        cpl._receipt("receipts/future/NO_SUCH_RECEIPT.json")


def test_the_reordering_rule_goes_quiet_when_slack_appears(monkeypatch):
    d = cpl._receipt("receipts/future/SINGLE_TOKEN_PARALLEL_SLACK.json")
    monkeypatch.setattr(cpl, "_receipt",
                        lambda rel: {**d, "theoretically_overlapable_ns": 5_000_000})
    fired, obs = cpl._c_no_slack_suppresses_reordering()
    assert fired is False
    assert "5000000" in obs


def test_the_underutilization_rule_goes_quiet_when_an_edge_is_not_a_dependency(monkeypatch):
    d = cpl._receipt("receipts/future/SINGLE_TOKEN_PARALLEL_SLACK.json")
    monkeypatch.setattr(cpl, "_receipt", lambda rel: {**d, "n_true_dependency": 9})
    fired, _ = cpl._c_multistream_underutilization()
    assert fired is False


def test_the_split_rule_goes_quiet_when_one_class_dominates(monkeypatch):
    d = cpl._receipt("receipts/future/OP_CLASS_ABLATION.json")
    dom = {**d, "decomposition": {**d["decomposition"], "classes": {
        "a": {"share_of_arithmetic": 0.80},
        "b": {"share_of_arithmetic": 0.20}}}}
    monkeypatch.setattr(cpl, "_receipt", lambda rel: dom)
    fired, obs = cpl._c_op_class_split_forbids_single_target()
    assert fired is False, "when one class dominates, attacking it IS the lever"
    assert "80.0%" in obs


def test_the_bound_rule_goes_quiet_if_the_school_reaches_the_target(monkeypatch):
    d = cpl._receipt("receipts/future/GAP_LEDGER_60.json")
    reach = {**d, "arithmetic_ceiling": {**d["arithmetic_ceiling"],
                                         "reaches_60": True}}
    monkeypatch.setattr(cpl, "_receipt", lambda rel: reach)
    fired, _ = cpl._c_arithmetic_school_is_bounded()
    assert fired is False


def test_the_projection_rule_goes_quiet_if_the_pattern_had_held(monkeypatch):
    d = cpl._receipt("receipts/future/Q4_BITCAST_AB.json")
    held = {**d, "projection_vs_graph": {**d["projection_vs_graph"],
                                         "the_lower_bound_pattern_did_not_hold": False}}
    monkeypatch.setattr(cpl, "_receipt", lambda rel: held)
    fired, _ = cpl._c_isolated_projection_is_not_a_bound()
    assert fired is False


def test_firing_emits_a_real_workunit_not_a_log_line():
    units = cpl.emitted_workunits()
    assert len(units) == 5
    for u in units:
        assert u["kind"] == "NEXT_WORK"
        assert u["species"] == "causal-rule"
        assert u["evidence"], "a unit with no evidence is a suggestion"
        assert u["hypothesis_family"]


def test_a_suppressing_rule_emits_nothing_and_says_why():
    rows = {r["id"]: r for r in cpl.evaluate()}
    r = rows["RULE.SUPPRESS_REORDERING"]
    assert r["fired"] is True
    assert r["emits_workunit"] is None
    assert "must not also generate work" in r["emits_nothing_because"]


def test_a_rule_cannot_emit_work_the_frontier_layer_would_refuse():
    """Units are built through frontiers._item, so dead schools are unreachable."""
    from tools.future import frontiers as fr
    with pytest.raises(fr.DeadSchoolRefused):
        fr._item(
            id="FT.MODEL_EXECUTION.rule.bad", frontier="MODEL_EXECUTION",
            kind="NEXT_WORK", title="reorder the dispatches",
            detail="x", required_lanes=(), gain=1, species="causal-rule",
            verifier="x", evidence=())


def test_every_rule_carries_its_evidence_and_its_reopen_condition():
    for r in cpl.evaluate():
        assert r["evidence"].startswith("receipts/")
        assert r["reopen"], "a scar without a reopen condition is dogma"
        assert r["bought_by"], "a rule must say what it cost to learn"


def test_the_meta_policy_refuses_to_call_itself_a_law():
    m = cpl.meta_policy()
    assert m["discriminators_that_killed_fastest"]
    assert m["measurements_that_mislead"]
    assert m["kernel_changes_that_usually_fail"]
    assert "not conclusions to cite as evidence" in m["this_is_not_a_law"]
