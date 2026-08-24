"""UNREACHED is not FAILED. Conflating them invents negative results.

The first version of the ladder recorded the 2.25-bpw arm as DIED@coherent_generation
when nothing had ever asked it to generate, and the leader as DIED@capability when no
capability suite exists anywhere in this campaign. Both would have manufactured a
failure out of an absence of evidence.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
P = REPO / "receipts" / "headless" / "COMPOSITION_LADDER.json"


def _d():
    return json.loads(P.read_text())


def _by_id(frag):
    return [c for c in _d()["candidates"] if frag in c["id"]][0]


def test_failed_and_untested_are_distinct_states():
    for c in _d()["candidates"]:
        assert not (c["died_at"] and c.get("unreached_above")), (
            f"{c['id']} is marked both failed and untested"
        )
        assert c["status"] in {"FAILED", "UNTESTED_ABOVE", "PASSED_ALL_TESTED"}


def test_the_2p25_arm_is_untested_not_dead():
    c = _by_id("q2_4level_fitted_g64")
    assert c["died_at"] is None, "it survived the complete token loop; it did not die"
    # N021 carried this arm through native coherent generation. Capability
    # is still UNREACHED — that is untested, not a death.
    assert c["highest_rung_reached"] == "coherent_generation"
    assert c["unreached_above"] == "capability"


def test_the_leader_did_not_fail_capability():
    c = _by_id("PARENT_A")
    assert c["died_at"] is None
    assert c["unreached_above"] == "capability", (
        "no capability suite exists; the rung is unreached, not failed"
    )


def test_a_real_death_is_still_recorded_as_death():
    c = _by_id("ternary_aa_g64")
    assert c["died_at"] == "complete_token", "the argmax flipped; that is a real failure"
    assert c["highest_rung_reached"] == "held_out_activation"


def test_no_candidate_is_described_above_its_evidence():
    for c in _d()["candidates"]:
        assert c["may_be_described_as"] == c["highest_rung_reached"]


def test_capability_is_unreached_everywhere():
    assert "UNREACHED FOR EVERY CANDIDATE" in _d()["capability_rung_status"]


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print(f"ok  {n}")
    print("6/6 passed")
