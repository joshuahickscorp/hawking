"""G033: multi-axis frontier and a disposition rule that is a rule."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "future"))
from pareto_disposition import dominates, frontier, disposition, AXES

P = json.loads((Path(__file__).resolve().parent.parent
                / "receipts" / "future" / "O003_PARETO_MULTIAXIS.json").read_text())


def test_the_frontier_is_not_ranked_by_capability_alone():
    """S012 §55: dense capability alone is the wrong selector."""
    assert "capability_passes" in AXES
    assert len([a for a in AXES if a != "capability_passes"]) >= 4, AXES


def test_a_failing_arm_can_hold_the_frontier_on_transfer_value():
    """S012 §54: a losing specimen still teaches, and that has to count.

    Recomputed from the points, not read from the stored frontier -- the first
    version asserted against the saved result, so mutating the transfer-value
    axis out of the code left it green.
    """
    live = {p["name"] for p in frontier(P["points"])}
    assert "sharedbasis8" in live, live
    sb = next(p for p in P["points"] if p["name"] == "sharedbasis8")
    assert sb["capability_passes"] is False, "the point of the test is that it FAILED"
    assert sb["transfer_value"] == max(p["transfer_value"] for p in P["points"])
    # and it survives ONLY because of that axis: strip it and it is dominated
    stripped = [{k: v for k, v in p.items() if k != "transfer_value"} for p in P["points"]]
    assert "sharedbasis8" not in {p["name"] for p in frontier(stripped)}, \
        "transfer value is not what holds it -- the test proves nothing"


def test_a_capability_failure_never_dominates_a_pass():
    passer = {"capability_passes": True, "complete_ebpw": 9.0, "representation_complexity": 9,
              "transfer_value": 0}
    failer = {"capability_passes": False, "complete_ebpw": 1.0, "representation_complexity": 0,
              "transfer_value": 9}
    assert not dominates(failer, passer)


def test_a_strictly_worse_arm_is_dominated():
    good = {"capability_passes": True, "complete_ebpw": 3.0, "decode_tok_s": 130.0,
            "representation_complexity": 1, "transfer_value": 2}
    worse = dict(good, complete_ebpw=4.0, decode_tok_s=120.0)
    assert dominates(good, worse) and not dominates(worse, good)
    assert frontier([good, worse]) == [good]


def test_dominated_specimens_are_released_without_executable_synthesis():
    d = disposition({"on_frontier": False, "contributions": ["a transfer Law"]})
    assert d["action"] == "SEAL_AND_RELEASE"
    assert d["earns_executable_synthesis"] is False
    assert d["retained_knowledge"] == ["a transfer Law"]


def test_a_finalist_earns_executable_work():
    d = disposition({"on_frontier": True, "contributions": ["x"]})
    assert d["action"] == "RETAIN" and d["earns_executable_synthesis"] is True


def test_a_specimen_that_taught_nothing_is_flagged_as_owing():
    """Releasing one that contributed nothing means its wall bought nothing."""
    assert disposition({"on_frontier": False, "contributions": []})["owes_findings"] is True
    assert disposition({"on_frontier": False, "contributions": ["y"]})["owes_findings"] is False
