"""Lineage slots, rollback, checksummed transfer, full reproduction cycle."""
from __future__ import annotations

import pytest

from lab.lineage.identity import GenesisInstance, make_qwen38_genesis
from lab.lineage.promotion import SelfCertificationRefused, evaluate_promotion
from lab.lineage.state import (
    CANDIDATE,
    CURRENT,
    LAST_KNOWN_GOOD,
    SLOT_NAMES,
    LineageError,
    LineageInvariantError,
    LineageState,
)
from lab.lineage.transfer import (
    TRANSFER_PAYLOAD_KEYS,
    TransferChecksumError,
    TransferError,
    accept_transfer,
    pack_state,
)
from lab.receipts import seal, verify
from lab.lineage.testing import (
    armed_lineage,
    make_child,
    passing_evidence,
    science_payload,
)
from lab.lineage.cycle import reproduce
from lab.lineage.four_slots import FourSlotScheduler


def test_three_named_slots_always_present() -> None:
    state = LineageState()
    assert tuple(state.to_dict()["slots"]) == SLOT_NAMES
    assert set(SLOT_NAMES) == {CURRENT, CANDIDATE, LAST_KNOWN_GOOD}
    parent = make_qwen38_genesis()
    snap = state.install(parent)
    assert set(snap["slots"]) == set(SLOT_NAMES)
    assert snap["slots"][CURRENT]["instance_id"] == parent.instance_id
    assert snap["slots"][LAST_KNOWN_GOOD]["instance_id"] == parent.instance_id
    assert snap["slots"][CANDIDATE] is None
    assert snap["valid_count"] >= 1


def test_install_refuses_invalid_genesis() -> None:
    parent = make_qwen38_genesis()
    parent.valid = False
    with pytest.raises(LineageError, match="invalid Genesis"):
        LineageState().install(parent)


def test_never_zero_valid_once_armed() -> None:
    state, parent, _child, _ev, _inv = armed_lineage()
    assert state.valid_count() >= 1
    state._put(CURRENT, None)
    state._put(LAST_KNOWN_GOOD, None)
    state._put(CANDIDATE, None)
    with pytest.raises(LineageInvariantError, match="zero valid Genesis"):
        state._assert_invariant()
    # Recover so later tests in this process aren't poisoned if reused.
    state._put(CURRENT, parent)
    state._put(LAST_KNOWN_GOOD, parent)
    state._assert_invariant()


def test_failed_successor_launch_rolls_back_to_lkg() -> None:
    state, parent, child, _ev, _inv = armed_lineage()
    before = state.current.instance_id
    result = state.launch_successor(lambda _c: False)
    assert result.ok is False
    assert result.rolled_back is True
    verify(result.receipt, label="rollback")
    assert result.receipt["zero_valid_genesis"] is False
    assert state.current is not None
    assert state.current.instance_id == parent.instance_id == before
    assert state.last_known_good is not None and state.last_known_good.valid
    assert state.candidate is not None
    assert state.candidate.valid is False
    assert state.candidate.instance_id == child.instance_id
    assert state.valid_count() >= 1


def test_successor_launch_exception_rolls_back() -> None:
    state, parent, child, _ev, _inv = armed_lineage()

    def boom(_c: GenesisInstance) -> bool:
        raise RuntimeError("exec format error")

    result = state.launch_successor(boom)
    assert result.rolled_back is True
    assert "exec format error" in result.reason
    assert state.current.instance_id == parent.instance_id
    assert state.candidate.valid is False
    assert state.candidate.instance_id == child.instance_id


def test_successful_launch_does_not_move_authority() -> None:
    state, parent, child, _ev, _inv = armed_lineage()
    result = state.launch_successor(lambda _c: True)
    assert result.ok is True
    assert result.rolled_back is False
    assert state.current.instance_id == parent.instance_id
    assert state.candidate.launched is True
    assert state.candidate.instance_id == child.instance_id


def test_launch_refused_without_rollback_artifact() -> None:
    state, _parent, _child, _ev, _inv = armed_lineage()
    state._put(LAST_KNOWN_GOOD, None)
    with pytest.raises(LineageError, match="LAST_KNOWN_GOOD"):
        state.launch_successor(lambda _c: True)
    assert state.current is not None and state.current.valid


def test_state_transfer_checksum_verified_on_far_side() -> None:
    _state, parent, child, _ev, _inv = armed_lineage()
    payload = science_payload()
    package = pack_state(parent, payload, to=child)
    verify(package, label="transfer")
    accepted = accept_transfer(package)
    assert accepted["verified"] is True
    assert accepted["checksum_sha256"] == package["checksum_sha256"]
    assert set(accepted["payload"]) == set(TRANSFER_PAYLOAD_KEYS)
    assert accepted["payload"]["NEXT_BOTTLENECK"].startswith("weight_addressing")


def test_corrupt_checksum_is_refused_and_authority_stays() -> None:
    state, parent, child, ev, inv = armed_lineage()
    package = pack_state(parent, science_payload(), to=child)
    # Reseal after flipping the checksum so the seal is intact and the
    # payload checksum is the thing that goes red.
    tampered = dict(package)
    tampered["checksum_sha256"] = "0" * 64
    tampered = seal(tampered)
    with pytest.raises(TransferChecksumError, match="checksum mismatch"):
        accept_transfer(tampered)
    mutated = dict(package)
    mutated["payload"] = dict(package["payload"])
    mutated["payload"]["NEXT_BOTTLENECK"] = "silently rewritten"
    mutated = seal(mutated)
    with pytest.raises(TransferChecksumError):
        accept_transfer(mutated)
    assert state.current.instance_id == parent.instance_id
    verdict = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "ACCEPT"
    with pytest.raises(TransferChecksumError):
        state.handover(package=mutated, invoker=inv, verdict=verdict)
    assert state.current.instance_id == parent.instance_id
    assert state.current.terminated is False


def test_missing_next_bottleneck_refused() -> None:
    _state, parent, child, _ev, _inv = armed_lineage()
    payload = science_payload()
    payload["NEXT_BOTTLENECK"] = "   "
    with pytest.raises(TransferError, match="NEXT_BOTTLENECK"):
        pack_state(parent, payload, to=child)


def test_handover_moves_science_and_retires_parent() -> None:
    state, parent, child, ev, inv = armed_lineage()
    verdict = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "ACCEPT"
    launched = state.launch_successor(lambda _c: True)
    assert launched.ok
    package = pack_state(parent, science_payload(), to=child)
    receipt = state.handover(package=package, invoker=inv, verdict=verdict)
    verify(receipt, label="handover")
    assert receipt["checksum_verified"] is True
    assert receipt["parent_terminated"] is True
    assert state.current.instance_id == child.instance_id
    assert state.current.live is True
    assert state.current.research_state["NEXT_BOTTLENECK"].startswith("weight_addressing")
    assert state.last_known_good.instance_id == parent.instance_id
    assert state.last_known_good.terminated is True
    assert state.last_known_good.valid is True
    assert state.candidate is None
    assert state.valid_count() >= 1


def test_runtime_handover_marks_activation_pending_until_observed() -> None:
    state, parent, child, ev, inv = armed_lineage()
    verdict = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "ACCEPT"
    package = pack_state(parent, science_payload(), to=child)
    receipt = state.handover(
        package=package,
        invoker=inv,
        verdict=verdict,
        retire_parent=False,
        successor_live=False,
    )
    verify(receipt, label="runtime-handover")
    assert receipt["successor_activation_pending"] is True
    assert state.current is not None and state.current.live is False
    assert state.current.launched is False
    assert state.last_known_good is not None
    assert state.last_known_good.terminated is False
    with pytest.raises(LineageError, match="observed live"):
        state.finalize_parent_retirement()
    observed = state.mark_current_live(instance_id=child.instance_id)
    verify(observed, label="runtime-child-live")
    assert state.current is not None and state.current.live is True
    retired = state.finalize_parent_retirement()
    verify(retired, label="runtime-parent-retired")
    assert state.last_known_good is not None and state.last_known_good.terminated is True


def test_handover_refuses_non_accept() -> None:
    state, parent, child, ev, inv = armed_lineage()
    ev = dict(ev)
    ev["protected_tests"] = [{"name": "coherence_greedy_ids", "status": "FAIL"}]
    verdict = evaluate_promotion(parent=parent, child=child, evidence=ev, invoker=inv, lineage=state)
    assert verdict["verdict"] == "REJECT"
    package = pack_state(parent, science_payload(), to=child)
    with pytest.raises(LineageError, match="ACCEPT"):
        state.handover(package=package, invoker=inv, verdict=verdict)
    assert state.current.instance_id == parent.instance_id


def test_reproduce_happy_path_and_failed_launch() -> None:
    state, parent, child, ev, inv = armed_lineage()
    sched = FourSlotScheduler()
    rolled = reproduce(
        state=state,
        child=child,
        evidence=ev,
        invoker=inv,
        parent_payload=science_payload(),
        launcher=lambda _c: False,
        scheduler=sched,
    )
    verify(rolled, label="cycle-rollback")
    assert rolled["outcome"] == "ROLLBACK"
    assert rolled["authority_moved"] is False
    assert rolled["rolled_back"] is True
    assert state.current.instance_id == parent.instance_id
    assert state.valid_count() >= 1
    # Slot 3 must be free after the failed attempt.
    assert sched.slot(3).empty

    # Fresh candidate of a new generation after the failed one was marked invalid.
    child2 = make_child(parent, instance_id="genesis-child-g1b")
    promoted = reproduce(
        state=state,
        child=child2,
        evidence=passing_evidence(parent, child2),
        invoker=inv,
        parent_payload=science_payload(),
        launcher=lambda _c: True,
        scheduler=sched,
    )
    verify(promoted, label="cycle-promote")
    assert promoted["outcome"] == "PROMOTED"
    assert promoted["authority_moved"] is True
    assert state.current.instance_id == child2.instance_id
    assert state.last_known_good.instance_id == parent.instance_id
    assert state.current.research_state["NEXT_BOTTLENECK"]
    assert sched.slot(3).empty
    assert sched.slot(0).occupant_id == child2.instance_id
    assert sched.slot(1).occupant_id != child2.instance_id


def test_parent_and_child_cannot_invoke_gate_method() -> None:
    _state, parent, child, ev, _inv = armed_lineage()
    with pytest.raises(SelfCertificationRefused, match="may not invoke"):
        parent.invoke_promotion_gate(child=child, evidence=ev)
    with pytest.raises(SelfCertificationRefused, match="may not invoke"):
        child.invoke_promotion_gate(parent=parent, evidence=ev)


def test_unknown_slot_name_refused() -> None:
    state = LineageState()
    with pytest.raises(LineageError, match="unknown lineage slot"):
        state.slot("FOURTH")
