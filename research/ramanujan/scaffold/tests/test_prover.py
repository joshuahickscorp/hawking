"""The prover is only useful if it refuses. These test the refusals."""
from __future__ import annotations

import json

import pytest

from ramanujan.evidence import PromotionRefused, Tier, promote
from ramanujan.prover import (
    TIER3_REQUIREMENTS,
    Status,
    adjudicate,
    check_no_sorry,
    check_no_undeclared_axioms,
    check_pinned_mathlib,
    machine_check_event,
)

GOOD = {
    "id": "t",
    "proof_lean": "import Mathlib.Tactic.NormNum\n\ntheorem t : (2:Nat)+2 = 4 := by norm_num\n",
    "pins": {},
}


def _cap(**over):
    return {**GOOD, **over}


def test_sorry_is_caught():
    f = check_no_sorry(_cap(proof_lean="theorem t : P := by sorry\n"))
    assert f.status is Status.FAILED and "sorry" in f.detail


def test_admit_is_caught():
    assert check_no_sorry(_cap(proof_lean="theorem t : P := by admit\n")).status is Status.FAILED


def test_sorry_inside_a_comment_is_not_a_hole():
    """A proof discussing sorries is not a proof containing one."""
    src = "-- we removed the sorry here\ntheorem t : (2:Nat)+2 = 4 := by norm_num\n"
    assert check_no_sorry(_cap(proof_lean=src)).status is Status.MET


def test_block_comment_mentioning_sorry_is_not_a_hole():
    src = "/- earlier draft used sorry -/\ntheorem t : (2:Nat)+2=4 := by norm_num\n"
    assert check_no_sorry(_cap(proof_lean=src)).status is Status.MET


def test_declared_axiom_is_caught():
    f = check_no_undeclared_axioms(_cap(proof_lean="axiom cheat : False\ntheorem t : P := cheat.elim\n"))
    assert f.status is Status.FAILED and "cheat" in f.detail


def test_capsule_with_no_pins_fails_rather_than_passing_vacuously():
    assert check_pinned_mathlib(_cap(pins={})).status is Status.FAILED


def test_pins_disagreeing_with_the_lock_are_caught():
    f = check_pinned_mathlib(_cap(pins={"mathlib_commit": "deadbeef"}))
    assert f.status is Status.FAILED and "disagree" in f.detail


def test_adjudication_covers_exactly_the_tier3_set():
    r = adjudicate(_cap(), run_container=False)
    assert {f["requirement"] for f in r["findings"]} == set(TIER3_REQUIREMENTS)


def test_unavailable_never_counts_as_met():
    """Skipping the container must not yield a proven verdict."""
    r = adjudicate(_cap(pins={}), run_container=False)
    assert r["tier3_met"] is False
    assert r["container_hash"] is None
    assert r["verdict"].startswith("REFUSED")


def test_no_event_is_minted_without_all_five():
    with pytest.raises(PermissionError, match="refusing to mint"):
        machine_check_event(_cap(), actor="v", run_container=False)


def test_a_partially_checked_capsule_cannot_reach_tier_3():
    """The end-to-end property: statically-clean is not proven.

    no_sorry, no_undeclared_axioms and pinned_mathlib can all pass on the host,
    and that must still not license Tier 3 without the container.
    """
    with pytest.raises(PermissionError):
        machine_check_event(_cap(), actor="v", run_container=False)


def test_promote_refuses_tier3_without_a_container_hash():
    """Guard the other side of the seam: even a hand-made machine_check event
    with no container hash must be refused by the lattice."""
    from ramanujan.evidence import VerifierEvent

    ev = VerifierEvent(kind="machine_check", actor="v", container_hash=None,
                       independent_of_author=True, detail={})
    with pytest.raises(PromotionRefused, match="clean-container"):
        promote(Tier.FORMALIZED, ev, author="a")


def test_real_capsule_is_adjudicated_not_assumed(tmp_path):
    """The shipped capsule must be judged by the same rules as any other."""
    from ramanujan.prover import ROOT

    p = ROOT / "ramanujan/container/capsules/two_plus_two.capsule.json"
    if not p.is_file():
        pytest.skip("capsule absent")
    r = adjudicate(json.loads(p.read_text()), run_container=False, capsule_path=p)
    assert r["capsule_id"] == "two_plus_two"
    # Static checks should pass on a real capsule; the container ones cannot
    # be decided without running it, and must not be silently treated as MET.
    by = {f["requirement"]: f["status"] for f in r["findings"]}
    assert by["no_sorry"] == "MET"
    assert by["no_undeclared_axioms"] == "MET"
    assert by["pinned_mathlib"] == "MET"
    assert r["tier3_met"] is False
