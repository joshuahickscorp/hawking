"""The ranker must reproduce the FPGA result it was generalised from, and must
refuse to recommend measuring anything when nothing can change the decision.
"""
from __future__ import annotations

import pytest

from tools.future import decision_value as D


def _threshold_decision(a: float, b: float):
    """`a` crosses a decision boundary at 10. `b` only scales the magnitude."""
    return {"decision": "ABOVE" if a > 10 else "BELOW", "magnitude": a * b}


def test_the_input_that_crosses_a_boundary_is_measured_first_even_when_it_moves_less():
    """The whole point. b swings the magnitude 100x and decides nothing."""
    r = D.rank_measurements(
        {"a": [1, 5, 9, 11, 20], "b": [1, 10, 100]},
        _threshold_decision,
    )
    order = [m.name for m in r.measurements]
    assert order[0] == "a", order
    by = {m.name: m for m in r.measurements}
    assert by["a"].changes_decision is True
    assert by["b"].changes_decision is False
    assert by["b"].magnitude_swing > by["a"].magnitude_swing, (
        "the test is not exercising the case it claims: b must swing harder"
    )
    assert by["a"].measure_order < by["b"].measure_order


def test_it_says_run_nothing_when_no_input_can_change_the_decision():
    """The most valuable answer a measurement planner can give."""
    r = D.rank_measurements(
        {"a": [1, 2, 3], "b": [1, 2, 3]},
        lambda a, b: {"decision": "FIXED", "magnitude": a * b},
    )
    assert r.any_measurement_changes_the_decision is False
    assert "RUN NOTHING" in r.as_dict()["verdict"]
    assert all(not m.changes_decision for m in r.measurements)


def test_more_reachable_decisions_outranks_fewer():
    def three_way(a: float, b: float):
        d = "LOW" if a < 3 else ("MID" if a < 7 else "HIGH")
        return {"decision": d if b > 0 else "LOW", "magnitude": a}

    def two_way(a: float, b: float):
        return {"decision": "HI" if b > 1 else "LO", "magnitude": a}

    r = D.rank_measurements({"a": [1, 5, 9], "b": [1, 2, 3]}, three_way)
    assert r.measurements[0].name == "a"
    assert len(r.measurements[0].decisions_reachable) == 3

    r2 = D.rank_measurements({"a": [1, 5, 9], "b": [1, 2, 3]}, two_way)
    assert r2.measurements[0].name == "b"


def test_the_baseline_decision_is_taken_at_the_held_medians():
    r = D.rank_measurements({"a": [1, 5, 20], "b": [1, 2, 3]}, _threshold_decision)
    assert r.inputs_held_at == {"a": 5, "b": 2}
    assert r.baseline_decision == "BELOW"


def test_a_decide_that_breaks_the_contract_is_refused():
    """A ranker that silently accepts a bare number ranks noise."""
    with pytest.raises(D.DecisionContract):
        D.rank_measurements({"a": [1, 2]}, lambda a: 5.0)
    with pytest.raises(D.DecisionContract):
        D.rank_measurements({"a": [1, 2]}, lambda a: {"magnitude": 1.0})
    with pytest.raises(D.DecisionContract):
        D.rank_measurements({"a": [1, 2]}, lambda a: {"decision": "X"})


def test_an_unsweepable_input_is_refused():
    with pytest.raises(ValueError):
        D.rank_measurements({"a": [1]}, lambda a: {"decision": "X", "magnitude": 1.0})
    with pytest.raises(ValueError):
        D.rank_measurements({}, lambda: {"decision": "X", "magnitude": 1.0})


def test_zero_magnitude_swing_is_not_reported_as_infinite_information():
    r = D.rank_measurements(
        {"a": [1, 2], "b": [1, 2]},
        lambda a, b: {"decision": "X", "magnitude": 0.0},
    )
    assert all(m.magnitude_swing == 0.0 for m in r.measurements)


def test_it_reproduces_the_fpga_partition_ordering():
    """The case it was generalised from must come out the same way.

    r swings the speedup ~14x and cannot reach APPLE_ONLY; t swings ~2x and can.
    t must be measured first.
    """
    from tools.future import fpga_partition as P

    W, frac = 1.0, 0.002

    def decide(r_fpga: float, t_link: float, setup: float):
        C = P.transport_cost(frac / 2, frac / 2, t_link, setup)
        return {"decision": P.phase(W, r_fpga, C), "magnitude": P.speedup(W, r_fpga, C)}

    ranking = D.rank_measurements(
        {"r_fpga": list(P.R_GRID), "t_link": list(P.T_GRID), "setup": [0.0, 0.1, 0.5, 1.0, 2.0]},
        decide,
    )
    by = {m.name: m for m in ranking.measurements}
    assert "APPLE_ONLY" not in by["r_fpga"].decisions_reachable, (
        "r reached APPLE_ONLY; the break-even no longer excludes it"
    )
    assert "APPLE_ONLY" in by["t_link"].decisions_reachable
    assert by["r_fpga"].magnitude_swing > by["t_link"].magnitude_swing
    assert by["t_link"].measure_order < by["r_fpga"].measure_order, (
        "ranked by swing instead of by decision power"
    )


def test_hold_at_states_the_baseline_instead_of_inventing_one():
    """A grid's median is itself a claim when the grid was invented to span a range.

    Holding the FPGA setup cost at the median of {0, 0.1W, 0.5W, W, 2W} asserts
    that transport already eats half the organ before any payload moves, and that
    made the FPGA rate look 8x weaker than it is (14.19x swing -> 1.81x).
    """
    grid = {"a": [1, 5, 20], "b": [0.0, 0.1, 0.5, 1.0, 2.0]}
    default = D.rank_measurements(grid, _threshold_decision)
    assert default.inputs_held_at["b"] == 0.5

    stated = D.rank_measurements(grid, _threshold_decision, hold_at={"b": 0.0})
    assert stated.inputs_held_at["b"] == 0.0
    by_default = {m.name: m for m in default.measurements}
    by_stated = {m.name: m for m in stated.measurements}
    assert by_default["a"].magnitude_max != by_stated["a"].magnitude_max, (
        "the baseline made no difference; this test proves nothing"
    )


def test_hold_at_refuses_a_name_that_is_not_a_candidate():
    with pytest.raises(ValueError):
        D.rank_measurements({"a": [1, 2]}, lambda a: {"decision": "X", "magnitude": a},
                            hold_at={"typo": 1})
