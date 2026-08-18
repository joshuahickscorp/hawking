"""Four-slot scheduler: hard cap, role tags, slot 3 preemptible for qualification."""
from __future__ import annotations

import pytest

from lab.lineage.four_slots import (
    HARD_CAP,
    PREEMPTIBLE_SLOT,
    SLOT_ROLES,
    FourSlotScheduler,
    SlotCapError,
    SlotError,
)


def test_hard_cap_is_four() -> None:
    assert HARD_CAP == 4
    assert set(SLOT_ROLES) == {0, 1, 2, 3}
    sched = FourSlotScheduler()
    sched.occupy(0, "parent")
    sched.occupy(1, "child-a")
    sched.occupy(2, "child-b")
    sched.occupy(3, "temp")
    with pytest.raises(SlotCapError, match="hard cap 4"):
        sched.occupy(4, "fifth")
    with pytest.raises(SlotCapError, match="hard cap 4"):
        sched.occupy(-1, "ghost")


def test_slot_roles_are_fixed() -> None:
    sched = FourSlotScheduler()
    assert sched.slot(0).role == "parent_integrator"
    assert sched.slot(1).role == "child_a_representation"
    assert sched.slot(2).role == "child_b_execution"
    assert sched.slot(3).role == "protected_test_or_temp_experiment"
    with pytest.raises(SlotError, match="parent_integrator"):
        sched.occupy(0, "parent", role="child_a_representation")


def test_slot_3_cannot_be_permanent() -> None:
    sched = FourSlotScheduler()
    with pytest.raises(SlotError, match="must not be permanently occupied"):
        sched.occupy(3, "temp", permanent=True)
    sched.occupy(3, "temp", permanent=False)
    with pytest.raises(SlotError, match="must not be permanently occupied"):
        sched.mark_permanent(3, True)
    assert sched.slot(3).permanent is False


def test_slot_3_preempted_for_qualification() -> None:
    sched = FourSlotScheduler()
    sched.occupy(0, "parent")
    sched.occupy(1, "child-a")
    sched.occupy(2, "child-b")
    sched.occupy(3, "long-running-experiment", purpose="scratch")
    result = sched.request_qualification("child-a")
    assert result["qualification"] is True
    assert result["preempted"]["occupant_id"] == "long-running-experiment"
    assert sched.slot(3).occupant_id == "child-a"
    assert sched.slot(3).permanent is False
    assert sched.slot(3).purpose == "qualification"
    # Home slots stay put.
    assert sched.slot(1).occupant_id == "child-a"
    assert sched.slot(0).occupant_id == "parent"


def test_occupy_full_slot_3_refuses_without_preempt() -> None:
    sched = FourSlotScheduler()
    sched.occupy(3, "temp")
    with pytest.raises(SlotError, match="request_qualification"):
        sched.occupy(3, "other")


def test_only_slot_3_is_preemptible() -> None:
    sched = FourSlotScheduler()
    sched.occupy(0, "parent")
    with pytest.raises(SlotError, match="only slot 3 is preemptible"):
        sched.preempt(0, reason="no")


def test_same_occupant_cannot_hold_two_home_slots() -> None:
    sched = FourSlotScheduler()
    sched.occupy(1, "child-a")
    with pytest.raises(SlotError, match="already holds home slot"):
        sched.occupy(2, "child-a")


def test_release_and_reoccupy() -> None:
    sched = FourSlotScheduler()
    sched.occupy(1, "child-a")
    prev = sched.release(1, reason="done")
    assert prev.occupant_id == "child-a"
    assert sched.slot(1).empty
    sched.occupy(1, "child-a2")
    assert sched.slot(1).occupant_id == "child-a2"
