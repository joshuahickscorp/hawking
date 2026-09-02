"""frontier_scheduler owns frontier switching: WAITING vs STUCK, and the
law that no runnable frontier is ever left idle behind a shrug.

Covers, against fixture data (no file I/O, no subprocess):
  * SUB2 parked with a recorded wake_condition reads as WAITING, exactly
    the state receipts/future/HCLI_MISSION_KERNEL.json is in today.
  * a WAITING state without a wake_condition cannot be constructed at all
    -- sleeping, never dead is a structural guarantee, not a convention.
  * select_next prefers ELIGIBLE over an idle report whenever any frontier
    is runnable (the one failure the brief says must never happen).
  * select_next never preempts a RUNNING frontier for a higher-value
    ELIGIBLE one.
  * PARK_ALL always carries every frontier's wake_condition, never a bare
    idle report -- this is the "specific next wake condition, not a
    shrug" requirement.
  * one broken probe reports UNKNOWN for itself without sinking the
    other frontiers in the same snapshot.
  * the real default probes read the actual repo receipts without raising
    (smoke test against whatever is on disk right now).

Runnable two ways:

    python3 -m pytest hcli/test_frontier_scheduler.py -q
    python3 hcli/test_frontier_scheduler.py
"""
from __future__ import annotations

import pytest

from hcli import frontier_scheduler as fs


# ---------------------------------------------------------------------
# FrontierState invariants
# ---------------------------------------------------------------------

def test_waiting_without_wake_condition_is_refused():
    """A parked frontier is SLEEPING, never dead -- enforced by the
    constructor, not left to callers to remember."""
    with pytest.raises(ValueError):
        fs.FrontierState("X", fs.WAITING, 0.0, None)
    with pytest.raises(ValueError):
        fs.FrontierState("X", fs.WAITING, 0.0, "")


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError):
        fs.FrontierState("X", "MADE_UP_KIND", 0.0, None)


def test_eligible_and_unknown_may_omit_wake_condition():
    fs.FrontierState("X", fs.ELIGIBLE, 1.0, None)
    fs.FrontierState("X", fs.UNKNOWN, 0.0, None)


# ---------------------------------------------------------------------
# SUB2 probe against a fixture shaped exactly like the real kernel
# ---------------------------------------------------------------------

def _sub2_kernel(n_iter=67, stop_n=67, wake="a landed receipt, a new scar, or an operator-set frontier"):
    return {
        "frontier": "SUB2_EBPW",
        "iterations": [{"n": i} for i in range(1, n_iter + 1)],
        "hypotheses": [{"id": "H1"}, {"id": "H2"}],
        "stops": [{"event": "unproductive_stop", "n": stop_n,
                   "reason": "no work accepted for 8 consecutive iterations",
                   "wake_condition": wake}] if stop_n else [],
    }


def test_sub2_parked_reads_as_waiting_with_the_recorded_wake_condition():
    state = fs.probe_sub2(data=_sub2_kernel())
    assert state.kind == fs.WAITING
    assert state.wake_condition == "a landed receipt, a new scar, or an operator-set frontier"


def test_sub2_resumed_past_its_stop_reads_as_eligible():
    """If the kernel shows iterations past the recorded stop's n, the loop
    was woken and continued -- that is no longer a parked frontier."""
    state = fs.probe_sub2(data=_sub2_kernel(n_iter=70, stop_n=67))
    assert state.kind == fs.ELIGIBLE
    assert state.expected_value == 2.0  # two open hypotheses in the fixture


def test_sub2_missing_receipt_is_unknown_not_a_crash():
    state = fs.probe_sub2(path=fs.RECEIPTS / "DOES_NOT_EXIST_12345.json")
    assert state.kind == fs.UNKNOWN


# ---------------------------------------------------------------------
# The heart of it: select_next must never idle while work is runnable
# ---------------------------------------------------------------------

def test_never_reports_idle_while_a_frontier_is_eligible():
    states = [
        fs.FrontierState("SUB2", fs.WAITING, 0.0, "a landed receipt"),
        fs.FrontierState("HCLI_SELF", fs.ELIGIBLE, 42.0, None),
        fs.FrontierState("FORBIDDEN_FRUIT", fs.WAITING, 0.0, "native BlobWriter"),
    ]
    decision = fs.select_next(states)
    assert decision.action == "RUN"
    assert decision.frontier == "HCLI_SELF"


def test_picks_the_highest_expected_value_among_eligible():
    states = [
        fs.FrontierState("A", fs.ELIGIBLE, 3.0, None),
        fs.FrontierState("B", fs.ELIGIBLE, 9.0, None),
        fs.FrontierState("C", fs.ELIGIBLE, 5.0, None),
    ]
    decision = fs.select_next(states)
    assert decision.frontier == "B"


def test_running_frontier_is_left_alone_even_if_something_scores_higher():
    states = [
        fs.FrontierState("SUB2", fs.RUNNING, 1.0, None),
        fs.FrontierState("HCLI_SELF", fs.ELIGIBLE, 1000.0, None),
    ]
    decision = fs.select_next(states)
    assert decision.action == "CONTINUE"
    assert decision.frontier == "SUB2"


def test_park_all_carries_every_wake_condition_never_a_shrug():
    states = [
        fs.FrontierState("SUB2", fs.WAITING, 0.0, "a landed receipt"),
        fs.FrontierState("ODYSSEY", fs.WAITING, 0.0, "a new candidate admitted"),
    ]
    decision = fs.select_next(states)
    assert decision.action == "PARK_ALL"
    assert decision.frontier is None
    assert decision.wake_conditions == {
        "SUB2": "a landed receipt",
        "ODYSSEY": "a new candidate admitted",
    }
    # never a bare idle string -- every non-runnable frontier is named
    assert set(decision.wake_conditions) == {"SUB2", "ODYSSEY"}


def test_park_all_with_only_unknown_probes_still_names_something_not_a_shrug():
    states = [fs.FrontierState("X", fs.UNKNOWN, 0.0, None)]
    decision = fs.select_next(states)
    assert decision.action == "PARK_ALL"
    assert decision.wake_conditions  # non-empty: never a bare shrug


# ---------------------------------------------------------------------
# snapshot() isolates a broken probe
# ---------------------------------------------------------------------

def test_snapshot_isolates_a_broken_probe():
    def _boom():
        raise RuntimeError("this frontier's receipt is corrupt")

    def _fine():
        return fs.FrontierState("OK", fs.ELIGIBLE, 1.0, None)

    result = fs.snapshot({"BROKEN": _boom, "OK": _fine})
    by_name = {s.name: s for s in result}
    assert by_name["BROKEN"].kind == fs.UNKNOWN
    assert by_name["OK"].kind == fs.ELIGIBLE


# ---------------------------------------------------------------------
# real default probes: must not raise against whatever is on disk today
# ---------------------------------------------------------------------

def test_default_probes_read_real_repo_state_without_raising():
    result = fs.snapshot()
    assert len(result) == len(fs.DEFAULT_PROBES)
    for state in result:
        assert state.kind in (fs.RUNNING, fs.ELIGIBLE, fs.WAITING, fs.UNKNOWN)
        if state.kind == fs.WAITING:
            assert state.wake_condition


def test_decide_end_to_end_against_real_repo_state():
    decision = fs.decide()
    assert decision.action in ("CONTINUE", "RUN", "PARK_ALL")
    d = decision.to_dict()
    assert d["snapshot"]  # the auditable record is non-empty
    if decision.action == "PARK_ALL":
        assert d["wake_conditions"], "PARK_ALL must never be a bare shrug"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
