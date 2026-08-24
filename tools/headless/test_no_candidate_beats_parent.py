"""The honest negative must stay honest: no promotion may be claimed from it.

S006 §33 permits recording NO_CANDIDATE_YET_BEATS_PARENT instead of promoting,
and forbids faking the shift. These assertions exist so a later edit cannot turn
the negative into a promotion without tripping something.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "receipts" / "headless"
P = R / "NO_CANDIDATE_YET_BEATS_PARENT.json"


def _d():
    return json.loads(P.read_text())


def test_verdict_matches_the_evidence():
    """The flat negative can only stand while nothing beats the parent.

    NOETIC_FUSED_SUBBIT reduces dispatches per token, which is the reopen
    condition this receipt itself named, so the verdict must move rather than
    keep asserting a negative the evidence no longer supports.
    """
    d = _d()
    valid = {
        "NO_CANDIDATE_YET_BEATS_PARENT",
        "REOPENED_CANDIDATE_LEADS_BUT_QUALIFICATION_NOT_RUN",
    }
    assert d["verdict"] in valid, d["verdict"]
    leaders = [c for c in d["candidates"] if c.get("beats_parent_on_both_axes")]
    if leaders:
        assert d["verdict"] != "NO_CANDIDATE_YET_BEATS_PARENT", (
            "a candidate leads on both axes; the flat negative is stale"
        )
        assert d.get("reopen_condition_fired") is True


def test_no_candidate_is_marked_promoted():
    body = json.dumps(_d()).lower()
    assert "promoted" not in body or "no promotion is claimed" in body


def test_blocker_carries_measured_numbers_not_adjectives():
    m = _d()["blocker"]["measured"]
    for k in ("cheapest_ebpw", "dearest_ebpw", "cheapest_tok_s", "dearest_tok_s",
              "bytes_reduction_fraction", "throughput_gain_fraction"):
        assert isinstance(m[k], (int, float)), f"{k} must be a measured number"


def test_density_really_did_not_buy_speed():
    # The whole blocker rests on this. If a later run makes bytes buy throughput,
    # the blocker is stale and must be re-derived rather than left standing.
    m = _d()["blocker"]["measured"]
    assert m["bytes_reduction_fraction"] > 0.2, "cheapest must actually be much cheaper"
    assert m["throughput_gain_fraction"] < 0.10, (
        "if cutting >20% of bytes now buys >10% throughput, decode is no longer "
        "dispatch-bound and this blocker must be re-derived"
    )


def test_next_family_has_a_reopen_condition():
    nxt = _d()["next_representation_family"]
    assert nxt["reopen_condition"].strip(), "a negative without a reopen condition is a dead end"
    assert len(nxt["evidence_that_narrows_it"]) >= 3


def test_g014_dependency_is_stated_not_substituted():
    dep = _d()["g014_dependency"].lower()
    assert "substitute" in dep and "old parent" in dep


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print(f"ok  {n}")
    print("6/6 passed")
