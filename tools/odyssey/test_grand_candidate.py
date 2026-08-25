"""G042 pins."""
import json
from pathlib import Path

import pytest

RH = Path(__file__).resolve().parents[2] / "receipts/headless"
R = RH / "GRAND_CANDIDATE.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="G042 receipt not built")


def rec():
    return json.load(open(R))


def test_the_composition_law_is_applied_not_recited():
    """S011 §79: nothing may be stacked before it has survived alone."""
    law = rec()["composition_law_S011_79"]
    assert law["survived_alone"]
    assert law["refused_for_stacking"]
    assert len(law["refused_for_stacking"]) > len(law["survived_alone"])


def test_every_refusal_names_why_it_did_not_survive_alone():
    for w in rec()["composition_law_S011_79"]["refused_for_stacking"]:
        assert any(k in w["why"] for k in
                   ("PROJECTION ONLY", "REFUTED", "UNTESTED")), w


def test_projections_were_not_stacked_into_the_candidate():
    """The MTP head projects 1.48-1.87x. Stacking it would be the whole trap."""
    refused = {w["win"] for w in rec()["composition_law_S011_79"]["refused_for_stacking"]}
    assert "MTP head" in refused
    assert "prefix reuse" in refused


def test_at_least_five_attacks_were_run():
    d = rec()
    assert d["n_attacks"] >= 5
    for a in d["adversary"]:
        assert a["outcome"] and a["question"]


def test_the_adversary_actually_won_something():
    """An adversary that never lands is not an adversary."""
    d = rec()
    assert d["n_attacks_won"] >= 1


def test_the_hardlink_attack_is_recorded_with_its_fix_and_verified_after():
    a = next(x for x in rec()["adversary"] if x["attack"] == "A2_zero_parent_hardlinks")
    assert "353" in a["found"]
    assert a["after"]["files_hardlinked"] == 0
    assert a["after"]["payload_unchanged"] is True


def test_the_closure_attack_is_recorded_and_the_closure_is_now_complete():
    a = next(x for x in rec()["adversary"] if x["attack"] == "A3_closure_completeness")
    assert "0 of 7" in a["found"]
    assert a["after"]["closure_present"] == a["after"]["closure_required"]
    assert a["after"]["missing"] == []


def test_the_candidate_is_actually_self_contained_now():
    c = rec()["candidate"]
    assert c["self_contained"] is True
    root = Path(c["artifact_root"])
    for f in ("tokenizer.json", "chat_template.jinja", "MIX_REPORT.json"):
        assert (root / f).is_file(), f
    assert not any(x.stat().st_nlink > 1 for x in root.rglob("*") if x.is_file())


def test_standalone_execution_was_demonstrated_not_asserted():
    a = next(x for x in rec()["adversary"] if x["attack"] == "A4_standalone_execution")
    assert a["result"]["exit_code"] == 0
    assert a["result"]["reply"].strip() == "Paris"


def test_the_capability_regime_attack_collapsed_the_gap():
    a = next(x for x in rec()["adversary"]
             if x["attack"] == "A5_capability_gap_is_regime_dependent")
    r = a["result"]
    assert r["gap_no_system"] > 0
    assert r["gap_with_system"] == 0
    assert a["outcome"] == "ATTACK WON"


def test_the_regime_finding_was_propagated_back_into_the_pareto_receipt():
    """A finding that stays in its own receipt has not been acted on."""
    p = json.load(open(RH / "PARETO_ARCHIVE.json"))
    c = p["CAPABILITY_COLUMN_IS_REGIME_CONDITIONAL"]
    assert c["with_default_system_prompt"]["gap"] == 0
    assert "no longer separates" in c["consequence"]
    assert c["what_still_separates_them"]


def test_the_verdict_does_not_claim_an_unqualified_pass():
    v = rec()["verdict"]
    assert "three of five attacks landed" in v
