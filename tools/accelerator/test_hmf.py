"""HMF pins. G053 -- canonicalization of humf -> hmf, plus HAWKGPU-0."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/accelerator"))
import hmf  # noqa: E402
import humf  # noqa: E402
from hmf import (  # noqa: E402
    Accelerator,
    AcceleratorDomain,
    DeviceIdentity,
    HmfError,
    MachineIdentity,
    build_hawkgpu0,
)
from humf import (  # noqa: E402
    Domain,
    Humf,
    HumfError,
    HumfObject,
    Materialization,
    MemoryClass,
    Ownership,
    State,
)

N = 1 << 10
NB = N * 4
PAYLOAD = bytes(range(256)) * (NB // 256)


# ------------------------------------------------------------- canonicalization

def test_hmf_is_humf_by_identity():
    """The F24 lesson (MEMORY.md 2026-08-23): 'a.Engine is b.Engine' was False
    for one module reachable under two dotted names. Pin the SAME check here,
    for the class that actually matters, and for its two closest neighbours."""
    assert hmf.Humf is humf.Humf
    assert hmf.HumfObject is humf.HumfObject
    assert hmf.Materialization is humf.Materialization
    assert hmf.State is humf.State


def test_humf_keeps_working_unchanged_after_canonicalization():
    """No existing receipt, test or caller breaks: humf.py imported directly,
    with none of hmf.py's vocabulary in play, must behave exactly as before."""
    h = Humf({"APPLE_UM": Domain("APPLE_UM", 1 << 30, 589.73, physical=True)})
    o = HumfObject("W", "tensor", N, "f32")
    o.place(Materialization("APPLE_UM", "dense_f32", "row_major", NB, State.CLEAN,
                            payload=PAYLOAD))
    h.register(o)
    assert o.valid_copies() == ["APPLE_UM"]


def test_hmf_is_not_a_second_implementation():
    """A second copy of the state machine would drift; a re-export cannot. The
    LEGAL table both modules see is the identical dict object."""
    assert hmf.LEGAL is humf.LEGAL


# --------------------------------------------------------- copy-state vocabulary

def test_all_nine_canonical_copy_states_exist():
    canonical = {"ABSENT", "MATERIALIZING", "CLEAN", "DIRTY", "STALE", "UNKNOWN",
                "CORRUPT", "LOST", "EVICTED"}
    assert canonical <= {s.value for s in State}


def test_a_clean_copy_can_legally_become_corrupt_or_lost_or_unknown():
    """Positive control: the new members are not decorative -- they are really
    reachable through the same fail-closed transition() method."""
    m = Materialization("D", "r", "l", 8, State.CLEAN)
    m.transition(State.UNKNOWN)
    assert m.state is State.UNKNOWN
    m2 = Materialization("D", "r", "l", 8, State.CLEAN)
    m2.transition(State.CORRUPT)
    assert m2.state is State.CORRUPT
    m3 = Materialization("D", "r", "l", 8, State.CLEAN)
    m3.transition(State.LOST)
    assert m3.state is State.LOST


def test_lost_is_stricter_than_invalid_no_direct_rematerialize():
    """Negative control paired with the positive above. INVALID may jump
    straight to MATERIALIZING; LOST may not -- it must be cleared to ABSENT
    first, because LOST means no retry path exists yet."""
    invalid = Materialization("D", "r", "l", 8, State.INVALID)
    invalid.transition(State.MATERIALIZING)          # legal for INVALID
    assert invalid.state is State.MATERIALIZING

    lost = Materialization("D", "r", "l", 8, State.LOST)
    with pytest.raises(HumfError, match="illegal transition"):
        lost.transition(State.MATERIALIZING)
    lost.transition(State.ABSENT)                    # the only legal exit
    assert lost.state is State.ABSENT


def test_unknown_state_illegal_transition_fails_closed():
    """Negative control: UNKNOWN's legal set does not include DIRTY."""
    m = Materialization("D", "r", "l", 8, State.UNKNOWN)
    with pytest.raises(HumfError, match="illegal transition"):
        m.transition(State.DIRTY)


def test_no_previously_legal_transition_became_illegal():
    """The additive claim, checked directly: every edge present before G053 is
    still present now (a superset, not a rename)."""
    pre_existing = {
        State.ABSENT:        {State.MATERIALIZING, State.TRANSFERRING},
        State.MATERIALIZING: {State.CLEAN, State.INVALID},
        State.TRANSFERRING:  {State.CLEAN, State.INVALID},
        State.CLEAN:         {State.DIRTY, State.STALE, State.EVICTED,
                              State.INVALID, State.TRANSFERRING},
        State.DIRTY:         {State.CLEAN, State.STALE, State.INVALID},
        State.STALE:         {State.TRANSFERRING, State.MATERIALIZING,
                              State.EVICTED, State.INVALID},
        State.EVICTED:       {State.MATERIALIZING, State.TRANSFERRING,
                              State.ABSENT},
        State.INVALID:       {State.ABSENT, State.MATERIALIZING},
    }
    for state, targets in pre_existing.items():
        assert targets <= hmf.LEGAL[state], (state, targets - hmf.LEGAL[state])


# ------------------------------------------------------------ ownership axis

def test_ownership_is_a_separate_axis_from_copy_state():
    """THE test G053 asks for by name: transitioning ownership must not touch
    state, and transitioning state must not touch ownership."""
    m = Materialization("D", "r", "l", 8, State.CLEAN)
    assert m.ownership is Ownership.APPLE_BIAS

    m.transition(State.DIRTY)                        # copy-state axis moves
    assert m.state is State.DIRTY
    assert m.ownership is Ownership.APPLE_BIAS        # ownership untouched

    m.transition_ownership(Ownership.TRANSIT)         # ownership axis moves
    m.transition_ownership(Ownership.SPARK0_BIAS)
    assert m.ownership is Ownership.SPARK0_BIAS
    assert m.state is State.DIRTY                     # state untouched


def test_ownership_illegal_transition_fails_closed():
    """Negative control: an exclusive bias cannot jump straight to another
    exclusive bias -- it must pass through TRANSIT."""
    m = Materialization("D", "r", "l", 8, State.CLEAN)
    with pytest.raises(HumfError, match="illegal ownership transition"):
        m.transition_ownership(Ownership.SPARK0_BIAS)


def test_ownership_legal_handoff_through_transit():
    """Positive control paired with the negative above."""
    m = Materialization("D", "r", "l", 8, State.CLEAN)
    m.transition_ownership(Ownership.TRANSIT)
    m.transition_ownership(Ownership.SHARED_READ)
    assert m.ownership is Ownership.SHARED_READ


def test_all_five_canonical_ownership_states_exist():
    canonical = {"APPLE_BIAS", "SPARK0_BIAS", "SPARK1_BIAS", "SHARED_READ",
                "TRANSIT"}
    assert canonical == {o.value for o in Ownership}


# ------------------------------------------------------------- memory classes

def _fabric_two_domains():
    h = Humf({"APPLE_UM": Domain("APPLE_UM", 96 << 30, 589.73, physical=True),
             "SCRATCH": Domain("SCRATCH", 1 << 30, 100.0, physical=True)})
    return h


def test_immutable_weights_stay_valid_across_domains_without_restreaming():
    """THE load-bearing case, verbatim from G053: IMMUTABLE_WEIGHTS can go
    SHARED_READ after placement and STAY valid. Transferring it into a second
    domain must not touch the source, and both copies must remain CLEAN --
    execution merely crossing a domain must never force a re-stream."""
    h = _fabric_two_domains()
    o = HumfObject("WEIGHTS", "tensor", N, "f32", memory_class=MemoryClass.IMMUTABLE_WEIGHTS)
    o.place(Materialization("APPLE_UM", "dense_f32", "row_major", NB, State.CLEAN,
                            payload=PAYLOAD))
    h.register(o)

    plan = h.plan_acquire("WEIGHTS", "SCRATCH")
    h.execute("WEIGHTS", plan, "SCRATCH")
    o.materializations["APPLE_UM"].transition_ownership(Ownership.TRANSIT)
    o.materializations["APPLE_UM"].transition_ownership(Ownership.SHARED_READ)
    o.materializations["SCRATCH"].transition_ownership(Ownership.TRANSIT)
    o.materializations["SCRATCH"].transition_ownership(Ownership.SHARED_READ)

    assert sorted(o.valid_copies()) == ["APPLE_UM", "SCRATCH"]
    assert o.materializations["APPLE_UM"].state is State.CLEAN
    assert o.materializations["SCRATCH"].state is State.CLEAN
    assert o.materializations["APPLE_UM"].ownership is Ownership.SHARED_READ
    assert o.materializations["SCRATCH"].ownership is Ownership.SHARED_READ

    # re-acquiring where it is already resident costs nothing -- no re-stream
    assert h.plan_acquire("WEIGHTS", "APPLE_UM").action == "ALREADY_RESIDENT"
    assert h.plan_acquire("WEIGHTS", "SCRATCH").action == "ALREADY_RESIDENT"


def test_immutable_weights_refuse_a_write():
    """The enforcement half of the same case: an IMMUTABLE_WEIGHTS object
    cannot be silently written through. Without this the 'stays valid' claim
    above would be a comment, not a policed invariant."""
    h = _fabric_two_domains()
    o = HumfObject("WEIGHTS", "tensor", N, "f32", memory_class=MemoryClass.IMMUTABLE_WEIGHTS)
    o.place(Materialization("APPLE_UM", "dense_f32", "row_major", NB, State.CLEAN,
                            payload=PAYLOAD))
    h.register(o)
    with pytest.raises(HumfError, match="immutable"):
        o.mark_written("APPLE_UM")
    assert o.materializations["APPLE_UM"].state is State.CLEAN   # untouched


def test_kv_state_is_mutable_and_stales_normally_the_contrast_case():
    """The pinned CONTRAST: a mutable class gets ordinary write/stale
    bookkeeping, unaffected by the immutability guard."""
    h = _fabric_two_domains()
    o = HumfObject("KV", "tensor", N, "f32", memory_class=MemoryClass.KV_STATE)
    o.place(Materialization("APPLE_UM", "dense_f32", "row_major", NB, State.CLEAN,
                            payload=PAYLOAD))
    h.register(o)
    h.execute("KV", h.plan_acquire("KV", "SCRATCH"), "SCRATCH")
    o.mark_written("SCRATCH")                         # must NOT raise
    assert o.materializations["SCRATCH"].state is State.DIRTY
    assert o.materializations["APPLE_UM"].state is State.STALE
    assert o.valid_copies() == []


def test_recurrent_state_is_also_mutable():
    o = HumfObject("REC", "tensor", N, "f32", memory_class=MemoryClass.RECURRENT_STATE)
    o.place(Materialization("APPLE_UM", "dense_f32", "row_major", NB, State.CLEAN,
                            payload=PAYLOAD))
    o.mark_written("APPLE_UM")                         # must NOT raise
    assert o.materializations["APPLE_UM"].state is State.DIRTY


def test_unclassified_objects_are_unaffected_by_the_new_policy():
    """The backward-compatibility case: memory_class defaults to None, and an
    unclassified object -- every object in the pre-G053 test suite -- gets no
    policy enforced at all."""
    o = HumfObject("PLAIN", "tensor", N, "f32")
    assert o.memory_class is None
    o.place(Materialization("APPLE_UM", "dense_f32", "row_major", NB, State.CLEAN,
                            payload=PAYLOAD))
    o.mark_written("APPLE_UM")                         # must NOT raise
    assert o.materializations["APPLE_UM"].state is State.DIRTY


def test_all_nine_memory_classes_are_covered_by_the_policy_table():
    canonical = {"IMMUTABLE_WEIGHTS", "KV_STATE", "RECURRENT_STATE", "ACTIVATIONS",
                "ROUTING", "METADATA", "SCRATCH", "COMPILER_ARTIFACT",
                "EXPERT_CACHE"}
    assert canonical == {c.value for c in MemoryClass}
    assert canonical == {c.value for c in hmf.MEMORY_CLASS_POLICY}


def test_there_is_no_universal_page_policy():
    """Directly checked, not just asserted in prose: the policy table does not
    hand every class the same mutable/shared_read_stable pair."""
    mutability = {c: p["mutable"] for c, p in hmf.MEMORY_CLASS_POLICY.items()}
    assert True in mutability.values() and False in mutability.values()


# ---------------------------------------------------------------- HAWKGPU-0

def test_build_hawkgpu0_has_exactly_apple_domain_0_today():
    accel = build_hawkgpu0()
    assert set(accel.domains) == {"APPLE_DOMAIN_0"}
    d = accel.domains["APPLE_DOMAIN_0"]
    assert d.fabric_domain.name == "APPLE_UM"          # the HUMF/AIR-facing name
    assert d.fabric_domain.physical is True
    assert d.device_identity.physical is True
    assert d.backend == "metal"


def test_apple_domain_0_fabric_domain_feeds_a_real_humf_fabric_untranslated():
    """No mapping layer: what AcceleratorDomain.fabric_domain carries is
    exactly what Humf() and AIR both already expect under the name APPLE_UM."""
    accel = build_hawkgpu0()
    h = Humf(accel.fabric_domains())
    assert "APPLE_UM" in h.domains
    assert h.domains["APPLE_UM"] is accel.domains["APPLE_DOMAIN_0"].fabric_domain


def test_a_second_fictitious_domain_needs_no_architecture_change():
    """The only evidence the abstraction generalizes: register a SECOND,
    fictitious domain (no Spark hardware exists on this machine -- every
    number below is a KNOB) and confirm the same Accelerator answers
    correctly for both, with nothing about Accelerator/AcceleratorDomain
    changed to allow it."""
    accel = build_hawkgpu0()
    accel.register_domain(AcceleratorDomain(
        slot="SPARK_DOMAIN_0",
        fabric_domain=Domain("SPARK_VRAM_0", 64 << 30, 40.0, physical=False,
                             latency_s=5e-5),
        device_identity=DeviceIdentity(
            vendor="SIMULATED", device_class="SPARK_ACCELERATOR", physical=False,
            absent_reason="no Spark hardware exists on this machine; every "
                          "field here is a fictitious placeholder for "
                          "structural testing only"),
        machine_identity=MachineIdentity(
            soc="SIMULATED", machine_class="SPARK_ARM_CLUSTER", physical=False,
            absent_reason="no Spark hardware exists on this machine"),
        backend="MOCK_SPARK_LINK",
        supported_representations=frozenset({"dense_f32", "dense_f16"}),
        topology={"APPLE_DOMAIN_0": 12.0},              # a KNOB, not a measurement
    ))
    assert set(accel.domains) == {"APPLE_DOMAIN_0", "SPARK_DOMAIN_0"}
    fd = accel.fabric_domains()
    assert set(fd) == {"APPLE_UM", "SPARK_VRAM_0"}
    # the generalized structure feeds a real Humf fabric exactly like the
    # one-domain case did, with no special-casing for the second slot
    h = Humf(fd)
    assert set(h.domains) == {"APPLE_UM", "SPARK_VRAM_0"}
    assert accel.domains["SPARK_DOMAIN_0"].fabric_domain.physical is False
    assert accel.domains["SPARK_DOMAIN_0"].fabric_domain.provenance == "SIMULATED"


def test_registering_a_duplicate_slot_is_refused():
    accel = build_hawkgpu0()
    with pytest.raises(HmfError, match="already registered"):
        accel.register_domain(AcceleratorDomain(
            slot="APPLE_DOMAIN_0",
            fabric_domain=Domain("APPLE_UM", 1, 1.0, physical=True),
            device_identity=DeviceIdentity("Apple", "X", physical=True),
            machine_identity=MachineIdentity("Apple", "X", physical=True),
            backend="metal", supported_representations=frozenset()))


def test_per_domain_asymmetry_is_visible_through_the_accelerator():
    """A scheduler that cannot see asymmetry cannot exploit it: bandwidth,
    backend and supported representations differ across domains, and that
    difference is readable directly off Accelerator.domains."""
    accel = build_hawkgpu0()
    accel.register_domain(AcceleratorDomain(
        slot="SPARK_DOMAIN_0",
        fabric_domain=Domain("SPARK_VRAM_0", 64 << 30, 40.0, physical=False),
        device_identity=DeviceIdentity("SIMULATED", "SPARK_ACCELERATOR",
                                       physical=False, absent_reason="no Spark hardware"),
        machine_identity=MachineIdentity("SIMULATED", "SPARK_ARM_CLUSTER",
                                         physical=False, absent_reason="no Spark hardware"),
        backend="MOCK_SPARK_LINK",
        supported_representations=frozenset({"dense_f16"}),
    ))
    apple = accel.domains["APPLE_DOMAIN_0"]
    spark = accel.domains["SPARK_DOMAIN_0"]
    assert apple.fabric_domain.bandwidth_gb_s != spark.fabric_domain.bandwidth_gb_s
    assert apple.backend != spark.backend
    assert apple.supported_representations != spark.supported_representations
    assert apple.device_identity.physical and not spark.device_identity.physical


def test_hawkgpu_hides_topology_from_a_caller_using_only_fabric_domains():
    """The other half of the asymmetry test: a caller that only ever sees
    fabric_domains() gets plain Domain objects with no backend/identity/
    topology attributes at all -- that is the hiding HAWKGPU-0 promises to
    callers above it."""
    accel = build_hawkgpu0()
    fd = accel.fabric_domains()["APPLE_UM"]
    assert not hasattr(fd, "backend")
    assert not hasattr(fd, "device_identity")
    assert not hasattr(fd, "topology")
    assert set(vars(fd)) == {"name", "bytes_capacity", "bandwidth_gb_s",
                             "physical", "latency_s"}


def test_a_fictitious_domain_identity_must_name_why_it_is_absent():
    """Negative control on the ABSENT-with-a-reason discipline: physical=False
    with no reason is refused outright, matching receipt.py's own rule that an
    ABSENT identity is never recorded without saying why."""
    with pytest.raises(HmfError, match="absent_reason"):
        DeviceIdentity("SIMULATED", "SPARK_ACCELERATOR", physical=False)
    with pytest.raises(HmfError, match="absent_reason"):
        MachineIdentity("SIMULATED", "SPARK_ARM_CLUSTER", physical=False)
