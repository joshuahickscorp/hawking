"""HUMF pins. FRONT H (G050)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
import humf  # noqa: E402
from humf import (DeviceLost, Domain, Humf, HumfError, HumfObject,  # noqa: E402
                  Materialization, MockExternalMemoryProvider, State,
                  TransferTimeout)

N = 1 << 20
NB = N * 4


PAYLOAD = bytes(range(256)) * (NB // 256)


def fabric():
    mp = MockExternalMemoryProvider(capacity_bytes=1 << 30, bandwidth_gb_s=5.0,
                                    latency_s=1e-4)
    doms = {"APPLE_UM": Domain("APPLE_UM", 96 << 30, 589.73, physical=True),
            mp.domain.name: mp.domain}
    # the fabric knows the mock domain is provider-backed, so a transfer into it
    # really moves bytes and can really fail
    h = Humf(doms, providers={mp.domain.name: mp})
    o = HumfObject("W42", "tensor", N, "f32", recompute_cost_s=0.05)
    # a CLEAN copy carries its bytes. Before this the fixture had a copy marked
    # CLEAN that held nothing, which is the fiction the executor's bookkeeping-only
    # transfer allowed.
    o.place(Materialization("APPLE_UM", "dense_f32", "row_major", NB, State.CLEAN,
                            payload=PAYLOAD))
    h.register(o)
    return h, o, mp


def test_illegal_transition_fails_closed():
    m = Materialization("APPLE_UM", "dense_f32", "row_major", NB, State.ABSENT)
    with pytest.raises(HumfError, match="illegal transition"):
        m.transition(State.CLEAN)          # must pass through MATERIALIZING


def test_evicted_cannot_become_clean_directly():
    m = Materialization("D", "r", "l", 8, State.EVICTED)
    with pytest.raises(HumfError):
        m.transition(State.CLEAN)


def test_a_write_makes_every_other_copy_stale():
    h, o, _ = fabric()
    p = h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM")
    h.execute("W42", p, "MOCK_EXTERNAL_VRAM")
    assert sorted(o.valid_copies()) == ["APPLE_UM", "MOCK_EXTERNAL_VRAM"]
    o.mark_written("MOCK_EXTERNAL_VRAM")
    assert o.materializations["APPLE_UM"].state is State.STALE
    assert o.is_dirty() and o.owner == "MOCK_EXTERNAL_VRAM"
    assert o.valid_copies() == []          # a dirty copy is not a valid copy


def test_a_stale_copy_is_never_offered_as_a_transfer_source():
    h, o, _ = fabric()
    h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    o.mark_written("MOCK_EXTERNAL_VRAM")
    plan = h.plan_acquire("W42", "APPLE_UM")
    assert all(opt.get("from") != "APPLE_UM" for opt in plan.options)
    # the only CLEAN-copy source is gone, so recompute is the honest answer
    assert plan.action == "RECOMPUTE"


def test_planner_prefers_the_cheaper_option():
    h, o, _ = fabric()
    plan = h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM")
    assert plan.action == "TRANSFER"       # 3.4ms transfer beats 50ms recompute
    o.recompute_cost_s = 1e-6              # now recompute is far cheaper
    assert h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM").action == "RECOMPUTE"


def test_a_plan_across_a_mock_domain_is_flagged_simulated():
    """The steer forbids treating a simulated transport number as physical
    evidence, so the plan must carry that on its face."""
    h, _, _ = fabric()
    plan = h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM")
    assert plan.cost_provenance == "SIMULATED"
    assert plan.rests_on_simulated_numbers is True


def test_recompute_cost_is_not_laundered_into_measured_transport():
    h, o, _ = fabric()
    o.recompute_cost_s = 1e-9
    plan = h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM")
    assert plan.action == "RECOMPUTE" and plan.cost_provenance == "MEASURED"


def test_impossible_when_no_clean_copy_and_no_recipe():
    h, o, _ = fabric()
    o.recompute_cost_s = None
    o.materializations["APPLE_UM"].transition(State.INVALID)
    plan = h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM")
    assert plan.action == "IMPOSSIBLE"
    with pytest.raises(HumfError):
        h.execute("W42", plan, "MOCK_EXTERNAL_VRAM")


def test_failure_injection_allocate():
    _, _, mp = fabric()
    mp.fail_next = "allocate"
    with pytest.raises(HumfError, match="injected failure: allocate"):
        mp.allocate(NB)
    mp.allocate(NB)                        # injection is one-shot


def test_capacity_is_enforced():
    mp = MockExternalMemoryProvider(capacity_bytes=1024, bandwidth_gb_s=1.0)
    with pytest.raises(HumfError, match="out of capacity"):
        mp.allocate(2048)


def test_mock_domain_is_never_marked_physical():
    mp = MockExternalMemoryProvider(capacity_bytes=1 << 20, bandwidth_gb_s=999.0)
    assert mp.domain.physical is False
    assert mp.domain.provenance == "SIMULATED"


def test_already_resident_costs_nothing():
    h, _, _ = fabric()
    plan = h.plan_acquire("W42", "APPLE_UM")
    assert plan.action == "ALREADY_RESIDENT" and plan.cost_s == 0.0


# ------------------------------------------ failure injection through the executor

def test_a_failed_transfer_leaves_the_destination_INVALID_not_CLEAN():
    """The defect this closes: execute() transitioned TRANSFERRING -> CLEAN on
    bookkeeping alone, never calling the provider. A transfer that cannot fail is
    not a transfer, it is an assumption -- and a copy marked CLEAN that holds
    nothing is exactly the silent corruption this module claims to fail closed on."""
    h, o, mp = fabric()
    mp.fail_next = "copy"
    plan = h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM")
    with pytest.raises(HumfError, match="injected failure"):
        h.execute("W42", plan, "MOCK_EXTERNAL_VRAM")
    dst = o.materializations["MOCK_EXTERNAL_VRAM"]
    assert dst.state is State.INVALID
    assert "MOCK_EXTERNAL_VRAM" not in o.valid_copies()
    assert dst.payload is None


def test_a_failed_outbound_transfer_does_not_damage_the_source():
    """The source was good before and must be good after. Losing the only valid
    copy because a transfer failed would turn a recoverable error into data loss."""
    h, o, mp = fabric()
    mp.fail_next = "copy"
    with pytest.raises(HumfError):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"),
                  "MOCK_EXTERNAL_VRAM")
    src = o.materializations["APPLE_UM"]
    assert src.state is State.CLEAN and src.payload == PAYLOAD
    assert o.valid_copies() == ["APPLE_UM"]


def test_the_failure_is_recorded_in_the_log_as_FAILED():
    h, o, mp = fabric()
    mp.fail_next = "copy"
    with pytest.raises(HumfError):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"),
                  "MOCK_EXTERNAL_VRAM")
    assert h.log and h.log[-1]["outcome"] == "FAILED"


def test_a_successful_transfer_moves_the_bytes_intact():
    """The other half: the same path that can fail must, when it succeeds, deliver
    the actual data. Otherwise the failure test only proves an exception path."""
    h, o, mp = fabric()
    h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    dst = o.materializations["MOCK_EXTERNAL_VRAM"]
    assert dst.state is State.CLEAN
    assert dst.payload == PAYLOAD                       # bit-identical through the provider
    assert sorted(o.valid_copies()) == ["APPLE_UM", "MOCK_EXTERNAL_VRAM"]


def test_recovery_after_a_failed_transfer_is_possible():
    """INVALID must not be a dead end -- a retry has to be able to succeed, or a
    single transport hiccup would permanently strand the object."""
    h, o, mp = fabric()
    mp.fail_next = "copy"
    with pytest.raises(HumfError):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    o.materializations["MOCK_EXTERNAL_VRAM"].transition(State.ABSENT)
    h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    assert o.materializations["MOCK_EXTERNAL_VRAM"].payload == PAYLOAD


def test_a_transfer_with_no_valid_source_payload_is_refused():
    """A domain with no provider must still never conjure bytes from nothing."""
    h, o, mp = fabric()
    o.materializations["APPLE_UM"].payload = None
    with pytest.raises(HumfError, match="no valid source payload"):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    assert o.materializations["MOCK_EXTERNAL_VRAM"].state is State.INVALID


def test_recompute_actually_runs_the_recipe():
    h, o, mp = fabric()
    o.materializations["APPLE_UM"].transition(State.STALE)
    o.recompute = lambda: b"\xAB" * NB
    plan = h.plan_acquire("W42", "APPLE_UM")
    assert plan.action == "RECOMPUTE"
    m = h.execute("W42", plan, "APPLE_UM")
    assert m.state is State.CLEAN and m.payload == b"\xAB" * NB


# ------------------------------------------- the harder failure: a provider that LIES

def test_silent_corruption_is_ACCEPTED_when_verification_is_off():
    """The control that makes the next test mean something. A provider that returns
    the wrong bytes WITHOUT RAISING defeats error handling entirely -- the transfer
    "succeeds", the copy is marked CLEAN, and the data is wrong. This is what the
    failure-injection work explicitly listed as not modelled."""
    from humf import _digest
    mp = MockExternalMemoryProvider(capacity_bytes=1 << 30, bandwidth_gb_s=5.0,
                                    latency_s=1e-4)
    doms = {"APPLE_UM": Domain("APPLE_UM", 96 << 30, 589.73, physical=True),
            mp.domain.name: mp.domain}
    h = Humf(doms, providers={mp.domain.name: mp}, verify_transfers=False)
    o = HumfObject("W42", "tensor", N, "f32", recompute_cost_s=0.05)
    o.place(Materialization("APPLE_UM", "dense_f32", "row_major", NB, State.CLEAN,
                            payload=PAYLOAD))
    h.register(o)
    mp.corrupt_next = True
    h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    dst = o.materializations["MOCK_EXTERNAL_VRAM"]
    assert dst.state is State.CLEAN                     # no error was raised
    assert "MOCK_EXTERNAL_VRAM" in o.valid_copies()     # and it counts as valid
    assert dst.payload != PAYLOAD                       # while holding the wrong bytes
    assert _digest(dst.payload) != _digest(PAYLOAD)


def test_verification_catches_what_error_handling_cannot():
    h, o, mp = fabric()
    mp.corrupt_next = True
    with pytest.raises(HumfError, match="integrity check FAILED"):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"),
                  "MOCK_EXTERNAL_VRAM")
    dst = o.materializations["MOCK_EXTERNAL_VRAM"]
    assert dst.state is State.INVALID
    assert "MOCK_EXTERNAL_VRAM" not in o.valid_copies()
    assert o.materializations["APPLE_UM"].state is State.CLEAN   # source intact


def test_a_clean_transfer_records_a_matching_digest_on_both_copies():
    """Verification must not only reject -- it must also ACCEPT, and leave the
    integrity it checked recorded rather than thrown away."""
    h, o, mp = fabric()
    h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    src = o.materializations["APPLE_UM"]
    dst = o.materializations["MOCK_EXTERNAL_VRAM"]
    assert src.digest is not None and src.digest == dst.digest


def test_integrity_policy_says_verification_is_affordable_where_it_is_needed():
    """The design conclusion, from a MEASURED checksum rate: over Apple unified
    memory a checksum costs many times the transfer and is not affordable -- and
    does not need to be, since there is no transport there to corrupt. Over an
    external link of a few GB/s it costs under a tenth. The domain that needs
    checking is the domain that can afford it."""
    from humf import integrity_policy
    um = integrity_policy(589.73)
    ext = integrity_policy(4.0)
    assert um["affordable"] is False and um["verification_cost_as_multiple_of_transfer"] > 10
    assert ext["affordable"] is True and ext["verification_cost_as_multiple_of_transfer"] < 0.15


def test_a_lying_provider_is_named_as_a_different_threat_from_a_failing_one():
    """crc32 is the right tool for ACCIDENT and the wrong tool for an ADVERSARY.
    Naming it integrity rather than authentication is the honest label, and the
    module says so where someone would otherwise assume more."""
    import humf
    assert "malicious" in humf._digest.__doc__
    assert "authentication" in humf._digest.__doc__


# --------------------------- the three modes the corruption receipt left unmodelled

def test_a_torn_write_is_caught_by_the_same_check_that_catches_a_lie():
    """A transport that delivers a PREFIX and stops leaves plausible bytes and raises
    nothing. It needed no new defence -- a digest over the whole payload cannot miss
    a truncated tail -- and recording that it is the SAME class as silent corruption
    is more useful than inventing a second mechanism."""
    h, o, mp = fabric()
    mp.tear_next = True
    plan = h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM")
    with pytest.raises(HumfError, match="integrity check FAILED"):
        h.execute("W42", plan, "MOCK_EXTERNAL_VRAM")
    assert o.materializations["MOCK_EXTERNAL_VRAM"].state is State.INVALID
    assert "MOCK_EXTERNAL_VRAM" not in o.valid_copies()


def test_without_verification_a_torn_write_is_marked_CLEAN_and_is_wrong():
    """The control. A check that is never watched failing is indistinguishable from
    one that does nothing."""
    mp = MockExternalMemoryProvider(capacity_bytes=1 << 30, bandwidth_gb_s=5.0)
    doms = {"APPLE_UM": Domain("APPLE_UM", 96 << 30, 589.73, physical=True),
            mp.domain.name: mp.domain}
    h = Humf(doms, providers={mp.domain.name: mp}, verify_transfers=False)
    o = HumfObject("W42", "tensor", N, "f32")
    o.place(Materialization("APPLE_UM", "dense_f32", "row_major", NB, State.CLEAN,
                            payload=PAYLOAD))
    h.register(o)
    mp.tear_next = True
    m = h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    assert m.state is State.CLEAN
    assert "MOCK_EXTERNAL_VRAM" in o.valid_copies()
    assert m.payload != PAYLOAD          # counted valid, and wrong


def test_a_hanging_transport_returns_control_and_quarantines_the_domain():
    """A transport that HANGS is worse than one that fails: nothing raises and
    nothing completes, so without a deadline the destination sits in TRANSFERRING
    forever and the except branch never runs."""
    h, o, mp = fabric()
    h.transfer_timeout_s = 0.05
    mp.hang_next_s = 0.6
    with pytest.raises(TransferTimeout):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    assert o.materializations["MOCK_EXTERNAL_VRAM"].state is State.INVALID
    assert "MOCK_EXTERNAL_VRAM" in h.quarantined


def test_a_quarantined_domain_refuses_the_next_transfer_until_released():
    """Quarantine is not decoration: the timed-out call was NOT abandoned, so a
    second transfer would race a worker still inside the provider."""
    h, o, mp = fabric()
    h.transfer_timeout_s = 0.05
    mp.hang_next_s = 0.6
    with pytest.raises(TransferTimeout):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    with pytest.raises(HumfError, match="QUARANTINED"):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    h.release_quarantine("MOCK_EXTERNAL_VRAM")
    m = h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    assert m.state is State.CLEAN and m.payload == PAYLOAD


def test_a_lost_device_invalidates_every_copy_it_held_not_only_the_one_in_flight():
    """The case that separates a lost DEVICE from a failed COPY. Getting it wrong is
    silent: valid_copies() would keep naming copies on a device that is gone."""
    h, o, mp = fabric()
    other = HumfObject("W43", "tensor", N, "f32")
    other.place(Materialization("MOCK_EXTERNAL_VRAM", "dense_f32", "row_major", NB,
                                State.CLEAN, payload=PAYLOAD))
    h.register(other)
    assert other.valid_copies() == ["MOCK_EXTERNAL_VRAM"]
    mp.vanish_next = True
    with pytest.raises(DeviceLost):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    assert other.valid_copies() == []                    # the bystander went too
    assert "MOCK_EXTERNAL_VRAM" in h.quarantined


def test_losing_the_only_live_copy_is_reported_as_DATA_LOSS_not_as_degraded_replication():
    """A DIRTY copy in a lost domain was the ONLY holder of the live state. Folding
    that into a count of invalidated copies would hide the one outcome that cannot
    be recovered from."""
    h, o, mp = fabric()
    live = HumfObject("W44", "tensor", N, "f32")
    live.place(Materialization("MOCK_EXTERNAL_VRAM", "dense_f32", "row_major", NB,
                               State.CLEAN, payload=PAYLOAD))
    h.register(live)
    live.mark_written("MOCK_EXTERNAL_VRAM")              # now DIRTY: the live state
    report = h.device_lost("MOCK_EXTERNAL_VRAM", "cable pulled")
    assert "W44" in report["data_lost"]
    assert "W44" in report["invalidated"]


def test_a_lost_device_does_not_report_data_loss_for_a_replicated_object():
    """The other direction: an object with a good copy elsewhere is degraded, not
    lost, and calling that data loss would be crying wolf."""
    h, o, mp = fabric()
    o.place(Materialization("MOCK_EXTERNAL_VRAM", "dense_f32", "row_major", NB,
                            State.CLEAN, payload=PAYLOAD))
    report = h.device_lost("MOCK_EXTERNAL_VRAM", "bus reset")
    assert "W42" in report["invalidated"]
    assert report["data_lost"] == []
    assert o.valid_copies() == ["APPLE_UM"]


def test_the_source_survives_a_lost_device():
    h, o, mp = fabric()
    mp.vanish_next = True
    with pytest.raises(DeviceLost):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    src = o.materializations["APPLE_UM"]
    assert src.state is State.CLEAN and src.payload == PAYLOAD


# ------------------ partial device loss, and what a round-trip check cannot see

def test_a_partial_loss_probes_each_copy_instead_of_assuming():
    """The full-loss handler is safe because of a BLANKET ASSUMPTION. On a partial
    loss that assumption is wrong in BOTH directions, so the fabric has to ask."""
    h, o, mp = fabric()
    kept = HumfObject("KEPT", "tensor", N, "f32")
    kept.place(Materialization("MOCK_EXTERNAL_VRAM", "dense_f32", "row_major", NB,
                               State.CLEAN, payload=PAYLOAD))
    gone = HumfObject("GONE", "tensor", N, "f32")
    gone.place(Materialization("MOCK_EXTERNAL_VRAM", "dense_f32", "row_major", NB,
                               State.CLEAN, payload=PAYLOAD))
    h.register(kept); h.register(gone)
    mp.store["KEPT"] = PAYLOAD
    mp.store["GONE"] = PAYLOAD
    mp.lose(["GONE"])                       # a reset that cleared ONE allocation
    r = h.device_partially_lost("MOCK_EXTERNAL_VRAM", "bus reset cleared one region")
    assert r["survived"] == ["KEPT"], r
    assert r["lost"] == ["GONE"], r
    assert kept.valid_copies() == ["MOCK_EXTERNAL_VRAM"]
    assert gone.valid_copies() == []


def test_a_blanket_verdict_would_manufacture_data_loss():
    """Treating a partial loss as a full one destroys live state the device still
    holds. Pinned by contrast: the full handler invalidates the survivor, the partial
    handler keeps it."""
    h, o, mp = fabric()
    live = HumfObject("LIVE", "tensor", N, "f32")
    live.place(Materialization("MOCK_EXTERNAL_VRAM", "dense_f32", "row_major", NB,
                               State.CLEAN, payload=PAYLOAD))
    h.register(live)
    mp.store["LIVE"] = PAYLOAD
    live.mark_written("MOCK_EXTERNAL_VRAM")              # DIRTY: the only live state
    r = h.device_partially_lost("MOCK_EXTERNAL_VRAM", "partial reset")
    assert r["survived"] == ["LIVE"]
    assert r["data_lost"] == []                          # nothing manufactured
    # the full-loss handler on the SAME state would have called it lost
    h.release_quarantine("MOCK_EXTERNAL_VRAM")
    full = h.device_lost("MOCK_EXTERNAL_VRAM", "same event read as a full loss")
    assert "LIVE" in full["data_lost"]


def test_a_round_trip_check_validates_the_round_trip_not_the_resident_copy():
    """The hazard the corruption receipt could not see. copy_out and a KERNEL are not
    the same path on a real bridge: one goes back over the transport, the other reads
    device memory directly. A provider whose compute path differs PASSES verification
    and still computes on wrong bytes."""
    h, o, mp = fabric()
    m = h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    assert m.state is State.CLEAN                    # verification passed
    assert m.payload == PAYLOAD                      # the ROUND TRIP is intact
    mp.compute_skew = True
    seen_by_kernel = mp.read_for_compute("W42")
    assert seen_by_kernel != PAYLOAD                 # and the kernel sees something else
    assert m.state is State.CLEAN                    # the fabric still says CLEAN


# ------------- quarantine is about TRUST, not direction; and a probe can time out

def test_a_quarantined_domain_is_not_offered_as_a_transfer_SOURCE():
    """The hole this closes was mine, made two blocks ago: the quarantine check lived
    only in execute() and read want_domain, so the planner would source FROM a domain
    the fabric had just declared untrustworthy without a word."""
    h, o, mp = fabric()
    doms = dict(h.domains); doms["SCRATCH"] = Domain("SCRATCH", 1 << 30, 100.0, physical=True)
    h.domains = doms
    o.place(Materialization("MOCK_EXTERNAL_VRAM", "dense_f32", "row_major", NB,
                            State.CLEAN, payload=PAYLOAD))
    o.materializations["APPLE_UM"].transition(State.EVICTED)   # leave ONE clean copy
    h.quarantined["MOCK_EXTERNAL_VRAM"] = "partial loss, trust unknown"
    plan = h.plan_acquire("W42", "SCRATCH")
    # MY FIRST VERSION OF THIS TEST ASSERTED IMPOSSIBLE AND THE CODE WAS RIGHT, NOT
    # THE TEST: the fixture object HAS a recompute recipe, so refusing the quarantined
    # source correctly falls back to RECOMPUTE rather than stranding the object. What
    # matters is that the untrusted domain is not offered AT ALL.
    assert all(op.get("from") != "MOCK_EXTERNAL_VRAM" for op in plan.options), plan.options
    assert plan.action == "RECOMPUTE", plan.detail


def test_with_no_recipe_a_quarantined_only_object_is_IMPOSSIBLE_and_says_why():
    """When there IS no honest alternative the fabric refuses and names the quarantine,
    rather than handing over bytes it has declared untrustworthy."""
    h, o, mp = fabric()
    doms = dict(h.domains); doms["SCRATCH"] = Domain("SCRATCH", 1 << 30, 100.0, physical=True)
    h.domains = doms
    o.recompute_cost_s = None                      # no way out but the bad one
    o.place(Materialization("MOCK_EXTERNAL_VRAM", "dense_f32", "row_major", NB,
                            State.CLEAN, payload=PAYLOAD))
    o.materializations["APPLE_UM"].transition(State.EVICTED)
    h.quarantined["MOCK_EXTERNAL_VRAM"] = "partial loss"
    plan = h.plan_acquire("W42", "SCRATCH")
    assert plan.action == "IMPOSSIBLE"
    assert "QUARANTINED" in plan.detail


def test_already_resident_in_a_quarantined_domain_is_refused_too():
    """Handing back a copy that happens to live in the quarantined domain would be the
    same trust failure through a different door."""
    h, o, mp = fabric()
    o.place(Materialization("MOCK_EXTERNAL_VRAM", "dense_f32", "row_major", NB,
                            State.CLEAN, payload=PAYLOAD))
    h.quarantined["MOCK_EXTERNAL_VRAM"] = "bus reset"
    plan = h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM")
    assert plan.action == "IMPOSSIBLE" and "QUARANTINED" in plan.detail


def test_releasing_the_quarantine_restores_the_domain_as_a_source():
    """A check that can only ever refuse is as useless as one that only ever accepts."""
    h, o, mp = fabric()
    doms = dict(h.domains); doms["SCRATCH"] = Domain("SCRATCH", 1 << 30, 100.0, physical=True)
    h.domains = doms
    o.place(Materialization("MOCK_EXTERNAL_VRAM", "dense_f32", "row_major", NB,
                            State.CLEAN, payload=PAYLOAD))
    o.materializations["APPLE_UM"].transition(State.EVICTED)
    o.recompute_cost_s = None
    h.quarantined["MOCK_EXTERNAL_VRAM"] = "bus reset"
    assert h.plan_acquire("W42", "SCRATCH").action == "IMPOSSIBLE"
    h.release_quarantine("MOCK_EXTERNAL_VRAM")
    assert h.plan_acquire("W42", "SCRATCH").action == "TRANSFER"


def test_a_probe_that_times_out_is_UNKNOWN_not_lost():
    """A timeout means we could not ask, NOT that the copy is gone. Calling it lost
    would be the manufactured-data-loss error one level up; calling it survived would
    name a phantom. The state is left ALONE."""
    h, o, mp = fabric()
    h.transfer_timeout_s = 0.05
    live = HumfObject("SLOW", "tensor", N, "f32")
    live.place(Materialization("MOCK_EXTERNAL_VRAM", "dense_f32", "row_major", NB,
                               State.CLEAN, payload=PAYLOAD))
    h.register(live)
    mp.store["SLOW"] = PAYLOAD
    mp.hang_next_s = 0.6
    r = h.device_partially_lost("MOCK_EXTERNAL_VRAM", "reset, probe hung")
    assert r["unknown"] == ["SLOW"], r
    assert r["lost"] == [] and r["data_lost"] == []
    m = live.materializations["MOCK_EXTERNAL_VRAM"]
    assert m.state is State.CLEAN and m.payload == PAYLOAD   # untouched
    # and the quarantine is what stops it being used before an operator resolves it
    assert "MOCK_EXTERNAL_VRAM" in h.quarantined


def _unknown_copy(verify=True):
    """A copy in the mock domain whose probe timed out, so its trust is UNKNOWN.

    Built through a REAL transfer so the copy carries whatever digest the transfer
    recorded -- which is the thing resolution has to check against.
    """
    h, o, mp = fabric()
    h.verify_transfers = verify
    p = h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM")
    h.execute("W42", p, "MOCK_EXTERNAL_VRAM")
    h.transfer_timeout_s = 0.05
    mp.hang_next_s = 0.6
    r = h.device_partially_lost("MOCK_EXTERNAL_VRAM", "reset, probe hung")
    assert r["unknown"] == ["W42"], r
    return h, o, mp


def test_releasing_a_quarantine_does_not_resolve_an_unknown_copy():
    """The asymmetry the previous receipt named against itself. Releasing says THE
    LINK IS FINE; it must not also say AND EVERY COPY ON IT IS GOOD."""
    h, o, mp = _unknown_copy()
    m = o.materializations["MOCK_EXTERNAL_VRAM"]
    assert m.trust == "UNKNOWN" and m.state is State.CLEAN   # state untouched
    r = h.release_quarantine("MOCK_EXTERNAL_VRAM")
    assert "MOCK_EXTERNAL_VRAM" not in h.quarantined          # the link is back
    assert r["still_unresolved"] == ["W42"]                   # the copy is not
    assert m.trust == "UNKNOWN"
    # and the refusal is real, not just a report: the copy is still out of service
    o.materializations["APPLE_UM"].transition(State.EVICTED)
    o.recompute_cost_s = None
    p = h.plan_acquire("W42", "APPLE_UM")
    assert p.action == "IMPOSSIBLE" and "UNRESOLVED" in p.detail


def test_resolve_unknown_verifies_against_the_recorded_digest():
    h, o, mp = _unknown_copy()
    h.release_quarantine("MOCK_EXTERNAL_VRAM")
    r = h.resolve_unknown("MOCK_EXTERNAL_VRAM", "W42")
    assert r["verdict"] == "VERIFIED", r
    assert o.materializations["MOCK_EXTERNAL_VRAM"].trust == "TRUSTED"
    # back in service, which is the direction a check that only ever refuses lacks
    o.materializations["APPLE_UM"].transition(State.EVICTED)
    assert h.plan_acquire("W42", "APPLE_UM").action == "TRANSFER"


def test_a_probe_with_nothing_to_compare_against_is_presence_only():
    """PRESENCE IS NOT INTEGRITY. Without a recorded digest the probe proves the copy
    answered and nothing whatever about its bytes, so trust stays UNKNOWN."""
    h, o, mp = _unknown_copy(verify=False)
    assert o.materializations["MOCK_EXTERNAL_VRAM"].digest is None
    r = h.resolve_unknown("MOCK_EXTERNAL_VRAM", "W42")
    assert r["verdict"] == "PRESENT_BUT_UNVERIFIABLE", r
    assert o.materializations["MOCK_EXTERNAL_VRAM"].trust == "UNKNOWN"


def test_resolve_unknown_catches_a_copy_that_was_there_and_wrong():
    """The case a timeout can hide: the copy answers, and the bytes are wrong."""
    h, o, mp = _unknown_copy()
    mp.corrupt_next = True
    r = h.resolve_unknown("MOCK_EXTERNAL_VRAM", "W42")
    assert r["verdict"] == "CORRUPT", r
    m = o.materializations["MOCK_EXTERNAL_VRAM"]
    assert m.state is State.INVALID and m.payload is None
    assert "MOCK_EXTERNAL_VRAM" not in o.valid_copies()


def test_a_probe_that_times_out_again_changes_nothing():
    h, o, mp = _unknown_copy()
    mp.hang_next_s = 0.6
    r = h.resolve_unknown("MOCK_EXTERNAL_VRAM", "W42")
    assert r["verdict"] == "STILL_UNKNOWN", r
    m = o.materializations["MOCK_EXTERNAL_VRAM"]
    assert m.trust == "UNKNOWN" and m.state is State.CLEAN and m.payload == PAYLOAD


def test_accept_unknown_is_recorded_as_an_assertion_not_a_verification():
    h, o, mp = _unknown_copy(verify=False)
    h.accept_unknown("MOCK_EXTERNAL_VRAM", "W42", "operator inspected the bus by hand")
    assert o.materializations["MOCK_EXTERNAL_VRAM"].trust == "ASSERTED"
    entry = [e for e in h.log if e["action"] == "ACCEPT_UNKNOWN"][-1]
    assert entry["evidence"] == "NONE -- operator assertion"
    # AND THE ASYMMETRY RUNS BOTH WAYS, which this caught: accepting the COPY does
    # not clear the LINK. While the domain is still quarantined the copy is not
    # offered at all and the planner falls back to RECOMPUTE; only once the link is
    # released too does the accepted copy return to service. Two axes, both cleared
    # separately, neither standing in for the other.
    o.materializations["APPLE_UM"].transition(State.EVICTED)
    assert h.plan_acquire("W42", "APPLE_UM").action == "RECOMPUTE"
    h.release_quarantine("MOCK_EXTERNAL_VRAM")
    p = h.plan_acquire("W42", "APPLE_UM")
    assert p.action == "TRANSFER" and p.source == "MOCK_EXTERNAL_VRAM"


def test_the_mover_uses_the_source_the_plan_named():
    """A plan whose source is not what executes is not an audit trail. DEMONSTRATED
    against the pre-fix code: a plan reading `from TRUSTED_SRC` moved the bytes out
    of a QUARANTINED domain, because _move re-derived `the first CLEAN copy`."""
    GOOD, BAD = b"G" * NB, b"B" * NB
    doms = {"QUARANTINED_SRC": Domain("QUARANTINED_SRC", 1 << 30, 5.0, physical=True),
            "TRUSTED_SRC": Domain("TRUSTED_SRC", 1 << 30, 5.0, physical=True),
            "SCRATCH": Domain("SCRATCH", 1 << 30, 100.0, physical=True)}
    h = Humf(doms)
    o = HumfObject("W", "tensor", N, "f32")
    # insertion order puts the quarantined domain FIRST, which is all _move looked at
    o.place(Materialization("QUARANTINED_SRC", "dense_f32", "row_major", NB,
                            State.CLEAN, payload=BAD))
    o.place(Materialization("TRUSTED_SRC", "dense_f32", "row_major", NB,
                            State.CLEAN, payload=GOOD))
    h.register(o)
    h.quarantined["QUARANTINED_SRC"] = "bus reset"
    p = h.plan_acquire("W", "SCRATCH")
    assert p.source == "TRUSTED_SRC"
    assert h.execute("W", p, "SCRATCH").payload == GOOD


def test_the_mover_refuses_to_substitute_a_source_the_plan_did_not_name():
    """If the named source went bad between plan and execute, re-plan -- do not pick
    another one silently. A quiet substitution is how the log stops matching reality."""
    h, o, mp = fabric()
    doms = dict(h.domains); doms["SCRATCH"] = Domain("SCRATCH", 1 << 30, 100.0, physical=True)
    h.domains = doms
    p = h.plan_acquire("W42", "SCRATCH")
    assert p.source == "APPLE_UM"
    o.place(Materialization("MOCK_EXTERNAL_VRAM", "dense_f32", "row_major", NB,
                            State.CLEAN, payload=PAYLOAD))
    o.materializations["APPLE_UM"].transition(State.EVICTED)   # the named source dies
    with pytest.raises(HumfError, match="refusing to substitute"):
        h.execute("W42", p, "SCRATCH")


def test_an_unknown_copy_is_not_counted_as_a_survivor_in_data_loss():
    """valid_copies() is about STATE and still names it. trusted_copies() is about
    what may be relied upon, and DATA-LOSS accounting must ask the second: counting
    an UNKNOWN copy as a survivor reports `you still have it` about a copy nobody
    can vouch for."""
    h, o, mp = _unknown_copy()
    assert "MOCK_EXTERNAL_VRAM" in o.valid_copies()
    assert "MOCK_EXTERNAL_VRAM" not in o.trusted_copies()
    o.recompute_cost_s, o.recompute = None, None
    r = h.device_lost("APPLE_UM", "the other domain went away too")
    assert r["data_lost"] == ["W42"], r


def _forge_crc(payload: bytes, free_at: int, want: int) -> bytes:
    """crc32 is affine over GF(2), so repairing a checksum is a 32x32 linear solve
    over 4 chosen bytes -- not a search."""
    import zlib
    crc = lambda b: zlib.crc32(b) & 0xFFFFFFFF
    n = len(payload); k = crc(bytes(n)); piv = {}
    for i in range(32):
        e = bytearray(n); e[free_at + i // 8] = 1 << (i % 8)
        v, tag = crc(bytes(e)) ^ k, 1 << i
        while v:
            b = v.bit_length() - 1
            if b not in piv:
                piv[b] = (v, tag); break
            pv, pt = piv[b]; v ^= pv; tag ^= pt
    t, sol = crc(payload) ^ want, 0
    while t:
        b = t.bit_length() - 1
        pv, pt = piv[b]; t ^= pv; sol ^= pt
    out = bytearray(payload)
    for i in range(32):
        if sol >> i & 1:
            out[free_at + i // 8] ^= 1 << (i % 8)
    return bytes(out)


def test_the_transfer_check_accepts_a_forged_checksum():
    from humf import _digest
    """WATCH THE CHECK FAIL TO FIRE. A payload with a flipped weight byte and four
    repaired padding bytes passes the fabric's integrity check, is marked CLEAN and
    counts as TRUSTED. This is not a flaw in crc32 -- it catches every accidental
    corruption class measured -- it is the difference between error detection and
    identity, and the reason _identity_digest exists."""
    h, o, mp = fabric()
    bad = bytearray(PAYLOAD); bad[8] ^= 0x80
    forged = _forge_crc(bytes(bad), len(PAYLOAD) - 4, _digest(PAYLOAD))
    assert _digest(forged) == _digest(PAYLOAD) and forged != PAYLOAD
    mp.substitute_next = forged
    h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    dst = o.materializations["MOCK_EXTERNAL_VRAM"]
    assert dst.state is State.CLEAN
    assert "MOCK_EXTERNAL_VRAM" in o.trusted_copies()
    assert dst.payload != PAYLOAD                      # while holding wrong bytes
    # and the digest the fabric recorded is the RIGHT one, for the WRONG bytes
    assert dst.digest == _digest(PAYLOAD)


def test_a_source_that_rotted_in_place_propagates_through_a_passing_check():
    from humf import _digest
    """The per-transfer check compares SOURCE to DESTINATION, so a faithful copy of
    corrupt bytes is exactly what it is looking for -- and afterwards the two copies
    AGREE, so every later check confirms the corruption. Sealing the identity at
    registration is what closes it; this pins the hole with the seal removed."""
    h, o, mp = fabric()
    o.content_digest = None                            # the behaviour before sealing
    rotted = bytearray(PAYLOAD); rotted[16] ^= 0x80
    o.materializations["APPLE_UM"].payload = bytes(rotted)
    h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    dst = o.materializations["MOCK_EXTERNAL_VRAM"]
    assert dst.state is State.CLEAN and dst.payload != PAYLOAD
    assert _digest(dst.payload) == _digest(o.materializations["APPLE_UM"].payload)

    h2, o2, mp2 = fabric()                             # same rot, seal intact
    rotted2 = bytearray(PAYLOAD); rotted2[16] ^= 0x80
    o2.materializations["APPLE_UM"].payload = bytes(rotted2)
    with pytest.raises(HumfError, match="no longer matches the identity sealed"):
        h2.execute("W42", h2.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"),
                   "MOCK_EXTERNAL_VRAM")
    # the destination is left INVALID holding NOTHING -- the refusal happens before
    # a single byte is handed to the transport, so the rot never crosses
    dst2 = o2.materializations["MOCK_EXTERNAL_VRAM"]
    assert dst2.state is State.INVALID and dst2.payload is None
    assert "MOCK_EXTERNAL_VRAM" not in o2.valid_copies()
    assert o2.materializations["APPLE_UM"].trust == "UNKNOWN"


def test_audit_finds_rot_nothing_else_would_have_touched():
    h, o, mp = fabric()
    h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    assert h.audit("W42")["diverged"] == []
    scrubbed = bytearray(o.materializations["MOCK_EXTERNAL_VRAM"].payload)
    scrubbed[0:4] = b"\x00\x00\x00\x00"
    o.materializations["MOCK_EXTERNAL_VRAM"].payload = bytes(scrubbed)
    after = h.audit("W42")
    assert after["diverged"] == ["MOCK_EXTERNAL_VRAM"]
    assert o.materializations["MOCK_EXTERNAL_VRAM"].trust == "UNKNOWN"
    assert o.materializations["APPLE_UM"].trust == "TRUSTED"


def test_trust_has_an_age_and_a_write_unseals_rather_than_false_alarming():
    h, o, mp = fabric()
    h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    assert h.stale_verifications(5) == []
    h.epoch += 40
    assert {x["domain"] for x in h.stale_verifications(5)} == {
        "APPLE_UM", "MOCK_EXTERNAL_VRAM"}
    o.mark_written("APPLE_UM")
    assert o.content_digest is None and "written in APPLE_UM" in o.unsealed_because
    assert h.audit("W42")["audited"] is False          # nothing to check against
    o.materializations["APPLE_UM"].transition(State.CLEAN)
    assert h.seal_value("W42", "APPLE_UM") is not None
    assert h.audit("W42")["diverged"] == []


def test_the_identity_recheck_knob_is_the_decay_model_paying_for_itself():
    """Re-hashing the source on EVERY transfer is the safe default and it is not
    free -- blake2b runs ~25x slower than crc32 on this machine. Re-verifying only
    what has gone unlooked-at is what makes it affordable."""
    h, o, mp = fabric()
    h.identity_recheck_age = None                      # registration only
    rotted = bytearray(PAYLOAD); rotted[16] ^= 0x80
    o.materializations["APPLE_UM"].payload = bytes(rotted)
    h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"), "MOCK_EXTERNAL_VRAM")
    assert o.materializations["MOCK_EXTERNAL_VRAM"].state is State.CLEAN  # not caught
    h.identity_recheck_age = 0
    o.materializations.pop("MOCK_EXTERNAL_VRAM")
    with pytest.raises(HumfError, match="no longer matches the identity sealed"):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"),
                  "MOCK_EXTERNAL_VRAM")


# ---------------------------------------------------------------------------
# The device digests its own memory. ACCELERATOR_HUMF_RESIDENT_DIGEST.json.
# ---------------------------------------------------------------------------

def _fabric_with(cls):
    mp = cls(capacity_bytes=1 << 30, bandwidth_gb_s=5.0, latency_s=1e-4)
    doms = {"APPLE_UM": Domain("APPLE_UM", 96 << 30, 589.73, physical=True),
            mp.domain.name: mp.domain}
    h = Humf(doms, providers={mp.domain.name: mp})
    o = HumfObject("W42", "tensor", N, "f32", recompute_cost_s=0.05)
    o.place(Materialization("APPLE_UM", "dense_f32", "row_major", NB, State.CLEAN,
                            payload=PAYLOAD))
    h.register(o)
    return h, o, mp


def test_the_resident_digest_catches_the_skew_the_round_trip_cannot():
    """Three receipts named this gap and none closed it. The device digesting its own
    memory through the path a KERNEL reads is what closes it."""
    h, o, mp = _fabric_with(humf.MockExternalMemoryProvider)
    mp.compute_skew = True                       # skewed BEFORE the transfer
    with pytest.raises(humf.HumfError, match="RESIDENT integrity check FAILED"):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"),
                  "MOCK_EXTERNAL_VRAM")
    assert "MOCK_EXTERNAL_VRAM" not in o.valid_copies()


def test_a_provider_that_digests_its_READBACK_path_still_passes():
    """THE CONTROL THAT GIVES THE CHECK ITS MEANING, and the reason this is a NARROWING
    and not a closure. ReadbackDigestProvider offers the same method, answers, and
    matches the source -- and a kernel still reads different bytes. The fabric cannot
    tell the two apart, so it RECORDS the claimed path rather than trusting it."""
    h, o, bad = _fabric_with(humf.ReadbackDigestProvider)
    bad.compute_skew = True
    m = h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"),
                  "MOCK_EXTERNAL_VRAM")
    assert m.state is State.CLEAN
    assert m.resident_verified is True           # true, and NOT a guarantee
    assert m.resident_digest_path == "readback"  # the boolean is only readable WITH this
    assert bad.read_for_compute("W42") != PAYLOAD


def test_an_honest_transfer_records_the_path_it_was_verified_through():
    """A check that can only ever refuse is as useless as one that only ever accepts."""
    h, o, mp = _fabric_with(humf.MockExternalMemoryProvider)
    m = h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"),
                  "MOCK_EXTERNAL_VRAM")
    assert m.state is State.CLEAN
    assert m.resident_verified is True and m.resident_digest_path == "compute"


def test_a_provider_without_a_device_digest_is_not_treated_as_a_failure():
    """Absence of the capability is not evidence of corruption -- it is absence of
    evidence, and the field says which."""
    class NoDigest(humf.MockExternalMemoryProvider):
        digest_resident = None
    h, o, mp = _fabric_with(NoDigest)
    m = h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"),
                  "MOCK_EXTERNAL_VRAM")
    assert m.state is State.CLEAN
    assert m.resident_verified is False and m.resident_digest_path is None


def test_the_round_trip_check_is_not_replaced_by_the_resident_one():
    """They catch different things: corrupt_next damages what comes BACK over the
    transport, which the resident digest would never see."""
    h, o, mp = _fabric_with(humf.MockExternalMemoryProvider)
    mp.corrupt_next = True
    with pytest.raises(humf.HumfError, match="integrity check FAILED"):
        h.execute("W42", h.plan_acquire("W42", "MOCK_EXTERNAL_VRAM"),
                  "MOCK_EXTERNAL_VRAM")
