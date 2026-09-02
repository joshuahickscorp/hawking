"""Typed experiment policy: representation only, resident remains authority."""
from __future__ import annotations

import pytest

from tools.future import complete_ebpw as ce
from tools.future import experiment_policy as ep
from tools.future import science_corpus as sc


def test_option_carries_payoff_info_gain_discriminator_cost_outcome_belief():
    opt = ep.option(
        option_id="opt-1",
        action="run_discriminator",
        hypothesis_id="H-1",
        expected_payoff_row=ep.expected_payoff(
            value=None, evidence_tier="STATIC"
        ),
        information_gain_row=ep.information_gain(
            about="H-1", reduces_uncertainty_in="H-1"
        ),
        discriminator_row=ep.discriminator(
            name="cheapest_falsifier",
            distinguishes=("H-1", "not-H-1"),
            cheapest_falsifier="one measurement that would kill it",
        ),
        cost_row=ep.cost(class_="ACTIVE_BYTES", evidence_tier="COST_MODEL"),
    )
    for key in ep.OPTION_KEYS:
        assert key in opt
    assert opt["learned"] is False
    assert opt["expected_payoff"]["value"] is None
    assert opt["information_gain"]["expected_bits"] is None
    assert opt["discriminator"]["name"] == "cheapest_falsifier"
    assert opt["cost"]["class"] == "ACTIVE_BYTES"
    assert opt["policy_authority"] == ep.POLICY_AUTHORITY
    assert "deterministic/resident policy remains the authority" in opt["policy_authority"]


def test_option_from_real_corpus_hypothesis():
    corpus = sc.load_historical_corpus()
    hyp = next(r for r in corpus["records"] if r["kind"] == "hypothesis")
    opt = ep.option_from_hypothesis(hyp["key_fields"])
    assert opt["option_id"].startswith("hyp:")
    assert opt["hypothesis_id"] == hyp["key_fields"]["id"]
    assert opt["learned"] is False
    assert opt["discriminator"]["name"]
    assert opt["information_gain"]["about"]


def test_deterministic_belief_update_writes_verdict_and_is_not_learned():
    prior = {"hypotheses": {"H-1": {"status": "TESTING"}}}
    out = ep.outcome(
        option_id="hyp:H-1",
        hypothesis_id="H-1",
        status="REFUTED",
        observed="gate never exceeded 1.31x up",
    )
    updated = ep.apply_deterministic_belief_update(prior, out)
    assert updated["learned"] is False
    assert updated["update_rule"] == "deterministic_verdict_write"
    assert "resident" in updated["authority"].lower() or "deterministic" in updated["authority"].lower()
    assert updated["hypotheses"]["H-1"]["status"] == "REFUTED"
    assert updated["hypotheses"]["H-1"]["learned"] is False
    assert updated["prior"]["hypotheses"]["H-1"]["status"] == "TESTING"


def test_learned_belief_update_is_refused():
    with pytest.raises(ep.PolicyRefused, match="learned belief update"):
        ep.apply_deterministic_belief_update(
            {"hypotheses": {}},
            {
                "hypothesis_id": "H-1",
                "status": "REFUTED",
                "learned": True,
            },
        )


def test_refuse_to_select_does_not_pick_an_option():
    options = [
        ep.option(option_id="a", action="test_a"),
        ep.option(option_id="b", action="test_b"),
    ]
    decision = ep.refuse_to_select(options)
    assert decision["selected"] is None
    assert decision["n_options"] == 2
    assert decision["learned"] is False
    assert "resident" in decision["authority"].lower() or "deterministic" in decision["authority"].lower()


def test_complete_ebpw_build_calls_option_from_ebpw_bill():
    """Production call site: complete_ebpw.build invokes option_from_ebpw_bill."""
    doc = ce.build()
    block = doc["science_dataset"]
    assert block["policy_option_id"].startswith("ebpw:")
    assert block["policy_learned"] is False
    assert "deterministic complete_ebpw" in block["cost_authority"]
    assert (
        "tools.future.experiment_policy.option_from_ebpw_bill"
        in block["call_sites"]
    )
    assert ep.POLICY_AUTHORITY == block["policy_authority"]
