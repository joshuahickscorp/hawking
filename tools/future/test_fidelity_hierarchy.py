"""A refusal is valid only at or below the level of the claim.

The 64x error this prevents: CAPABILITY_INFORMATION_MAP answered a capability
question with a cosine 0.99 bar and licensed 0.28% of the token.
"""
from __future__ import annotations

import pytest

from tools.future import fidelity_hierarchy as fh


def test_the_hierarchy_is_ordered_weakest_evidence_first():
    assert fh.LEVELS[0] == "REPRESENTATION_FIDELITY"
    assert fh.LEVELS[-1] == "HCLI_MISSION_CAPABILITY"
    assert fh.rank("CAPABILITY") > fh.rank("LOCAL_FUNCTIONAL_FIDELITY")
    assert fh.rank("LOCAL_FUNCTIONAL_FIDELITY") > fh.rank("REPRESENTATION_FIDELITY")


def test_an_unknown_level_refuses():
    with pytest.raises(fh.HierarchyRefused, match="is not one of"):
        fh.rank("VIBES")


def test_a_capability_claim_cannot_be_refused_on_local_fidelity():
    """The load-bearing rule."""
    v = fh.may_refuse(claim_level="CAPABILITY",
                      measured_level="LOCAL_FUNCTIONAL_FIDELITY")
    assert v["refusal_is_valid"] is False
    assert "64x error" in v["why"]


def test_a_capability_claim_cannot_be_refused_on_representation_fidelity():
    v = fh.may_refuse(claim_level="CAPABILITY",
                      measured_level="REPRESENTATION_FIDELITY")
    assert v["refusal_is_valid"] is False


def test_a_coding_claim_can_be_refused_on_a_coding_measurement():
    v = fh.may_refuse(claim_level="REPRESENTATION_FIDELITY",
                      measured_level="REPRESENTATION_FIDELITY")
    assert v["refusal_is_valid"] is True


def test_a_stronger_measurement_may_refuse_a_weaker_claim():
    v = fh.may_refuse(claim_level="LOCAL_FUNCTIONAL_FIDELITY",
                      measured_level="CAPABILITY")
    assert v["refusal_is_valid"] is True


def test_every_level_declares_what_it_cannot_establish():
    for lvl in fh.LEVELS:
        assert fh.WHAT_EACH_CANNOT_ESTABLISH[lvl]
        assert fh.WHAT_EACH_MEASURES[lvl]


def test_every_refutation_carries_the_level_it_was_measured_at():
    for r in fh.labelled_refutations():
        assert r["measured_at"] in fh.LEVELS
        assert r["source"]
        assert r["why"]


def test_coding_refutations_still_bind():
    """Entropy and mutual information answered coding claims. They stand."""
    rows = {r["id"]: r for r in fh.labelled_refutations()}
    assert rows["mlp_code_entropy_1p87"]["still_binds"] is True
    assert rows["gate_up_mutual_information"]["still_binds"] is True


def test_the_matched_perturbation_refutation_still_binds():
    """Its claim was about relative sensitivity, which is what was measured."""
    rows = {r["id"]: r for r in fh.labelled_refutations()}
    assert rows["functional_role_gate_dominant"]["still_binds"] is True


def test_the_central_defect_is_named_with_its_numbers():
    d = fh.build()["the_defect_this_prevents"]
    assert "27.7 MB" in d and "1773 MB" in d and "64x" in d
    assert "cosine 0.99" in d


def test_reopenable_is_not_permission_to_delete():
    r = fh.reopenable()
    assert r["n_not_binding_a_capability_claim"] > 0
    assert "nothing here says these approaches WORK" in \
        r["this_is_not_permission_to_delete"]
    assert "capability measurement that was never taken" in \
        r["this_is_not_permission_to_delete"]


def test_the_counts_reconcile():
    r = fh.reopenable()
    assert r["n_still_binding"] + r["n_not_binding_a_capability_claim"] == \
        r["n_refutations"]
