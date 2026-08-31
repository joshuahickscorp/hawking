"""Six families named by S020 §16, and the honest count of which are judged.

G053 asks what recurrent update the model actually needs, judged on state
trajectory and logits. This module is that judge. Two of the six have curves.
The value of the table is that it refuses to let the other four look judged -
especially fused_update_and_consume, whose trajectory is identical BY
CONSTRUCTION, so running it through this judge would return FLAT_ZERO exactly
like the identity control and a reader would see a pass.
"""
from __future__ import annotations

from tools.future import deltanet_multistep as dm


def test_all_six_families_are_present():
    fams = {f["family"] for f in dm.S020_FAMILIES}
    assert fams == {
        "smaller_state",
        "structured_transitions",
        "generated_coefficients",
        "learned_recurrence",
        "conditional_recurrence",
        "fused_update_and_consume",
    }


def test_a_judged_family_names_a_candidate_that_actually_ran():
    ran = {f["candidate_id"] for f in dm.S020_FAMILIES if f["judged"]}
    assert ran == {dm.TRUNCATED_STATE, dm.LOWER_RANK_TRANSITION}
    for f in dm.S020_FAMILIES:
        if f["judged"]:
            assert f["candidate_id"], f"{f['family']} claims judged with no candidate"


def test_an_unjudged_family_must_name_its_blocker_and_its_unblock():
    for f in dm.S020_FAMILIES:
        if f["judged"]:
            continue
        assert f["blocked_by"], f"{f['family']} is unjudged with no reason"
        assert f["unblocks_when"], f"{f['family']} is unjudged with no path forward"


def test_fusion_is_never_judged_by_a_function_judge():
    """Its trajectory is flat by construction; a pass here would be manufactured."""
    fused = next(f for f in dm.S020_FAMILIES if f["family"] == "fused_update_and_consume")
    assert fused["judged"] is False
    assert fused["blocked_by"] == dm.NOT_A_TRAJECTORY_QUESTION
    assert fused["unblocks_when"] == "never through this judge"


def test_the_report_counts_two_of_six():
    r = dm.families_report()
    assert r["n_families"] == 6
    assert r["n_judged_on_trajectory"] == 2
    assert set(r["blocked"]) == {
        "generated_coefficients",
        "learned_recurrence",
        "conditional_recurrence",
        "fused_update_and_consume",
    }
