"""fusion_isa pins. FRONT G054 (transport-protocol half; humf.py owns the
memory-fabric half and is not touched here)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fusion_isa  # noqa: E402
from fusion_isa import (  # noqa: E402
    OP_SPEC,
    ArityError,
    DuplicateCommandIdError,
    ForwardReferenceError,
    FusionCommand,
    FusionIsaError,
    FusionOp,
    FusionTimeline,
    LockConflictError,
    ReleaseWithoutAcquireError,
    UnknownCommandIdError,
    op_from_packet,
)
from fusion_wire import FusionPacket, decode  # noqa: E402


# ------------------------------------------------------ every op declares its shape

@pytest.mark.parametrize("op", list(FusionOp))
def test_minimal_valid_command_matches_its_ops_declared_side_effects(op):
    """Constructs the smallest legal command for every one of the 14 ops and
    checks side_effects is exactly what OP_SPEC declares for that op -- the
    'each command declares inputs/outputs/dependencies/side effects'
    requirement, exercised over the whole opcode space at once."""
    spec = OP_SPEC[op]
    ins = tuple(f"OBJ{i}" for i in range(spec.min_in))
    outs = tuple(f"OUT{i}" for i in range(spec.min_out))
    reps = tuple(f"R{i}" for i in range(spec.min_replicas))
    cmd = FusionCommand(seq=0, op=op, inputs=ins, outputs=outs, replicas=reps)
    assert cmd.side_effects == spec.side_effects
    assert cmd.inputs == ins and cmd.outputs == outs


def test_side_effects_are_not_settable_per_instance():
    """side_effects is derived from the op, not a field -- constructing a
    command cannot lie about what its op does."""
    cmd = FusionCommand(seq=0, op=FusionOp.FENCE)
    with pytest.raises(AttributeError):
        cmd.side_effects = ("MADE_UP",)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- arity

def test_reduce_requires_at_least_two_inputs():
    with pytest.raises(ArityError, match="inputs"):
        FusionCommand(0, FusionOp.REDUCE, inputs=("A",), outputs=("C",), replicas=("R0",))


def test_reduce_with_two_inputs_is_fine():
    FusionCommand(0, FusionOp.REDUCE, inputs=("A", "B"), outputs=("C",), replicas=("R0",))


def test_fence_op_refuses_any_input():
    with pytest.raises(ArityError):
        FusionCommand(0, FusionOp.FENCE, inputs=("W",))


def test_acquire_read_refuses_zero_replicas():
    with pytest.raises(ArityError, match="replicas"):
        FusionCommand(0, FusionOp.ACQUIRE_READ, inputs=("W",), replicas=())


def test_acquire_read_refuses_two_replicas():
    with pytest.raises(ArityError):
        FusionCommand(0, FusionOp.ACQUIRE_READ, inputs=("W",), replicas=("R0", "R1"))


def test_gather_refuses_zero_inputs():
    with pytest.raises(ArityError):
        FusionCommand(0, FusionOp.GATHER, outputs=("D",), replicas=("R0",))


def test_gather_accepts_many_inputs_and_many_replicas():
    FusionCommand(0, FusionOp.GATHER, inputs=("A", "B", "C"), outputs=("D",),
                 replicas=("R0", "R1", "R2"))


def test_submit_op_accepts_zero_of_everything():
    FusionCommand(0, FusionOp.SUBMIT)  # must not raise; SUBMIT's bounds are (0, None) all round


# --------------------------------------------------------- upward API surface

def test_upward_api_is_exactly_submit_fence_wait_plus_read_only_queries():
    t = FusionTimeline()
    public = {n for n in dir(t) if not n.startswith("_") and callable(getattr(t, n))}
    dispatch_verbs = {"submit", "fence", "wait"}
    read_only_queries = {"object_version", "replica_state", "commands"}
    assert dispatch_verbs <= public
    assert public - dispatch_verbs <= read_only_queries, (
        f"unexpected public method(s) on FusionTimeline: "
        f"{public - dispatch_verbs - read_only_queries}")


def test_no_provider_shaped_method_names_exist():
    """Nothing named metalLaunch, cudaLaunch, or sendPacket may appear in the
    upward API -- those are provider details and no provider is wired here."""
    t = FusionTimeline()
    for name in ("metalLaunch", "metal_launch", "cudaLaunch", "cuda_launch",
                 "sendPacket", "send_packet", "dispatch"):
        assert not hasattr(t, name)
    assert not hasattr(fusion_isa, "metalLaunch")
    assert not hasattr(fusion_isa, "cudaLaunch")
    assert not hasattr(fusion_isa, "sendPacket")


# --------------------------------------------------- timeline structural refusals

def test_duplicate_command_id_refused_not_overwritten():
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.FENCE))
    with pytest.raises(DuplicateCommandIdError, match="0"):
        t.submit(FusionCommand(0, FusionOp.FENCE))
    assert len(t.commands()) == 1   # the duplicate did not silently replace it


def test_forward_reference_refused_not_reordered():
    t = FusionTimeline()
    with pytest.raises(ForwardReferenceError, match="5"):
        t.submit(FusionCommand(1, FusionOp.FENCE, depends_on=(5,)))
    assert t.commands() == ()       # nothing was recorded


def test_dependency_already_submitted_is_accepted():
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.FENCE))
    t.submit(FusionCommand(1, FusionOp.FENCE, depends_on=(0,)))   # must not raise
    assert [c.seq for c in t.commands()] == [0, 1]


# ------------------------------------------------- ACQUIRE_WRITE/RELEASE versioning

def test_acquire_write_then_release_bumps_version_and_stales_other_replicas():
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.MATERIALIZE, outputs=("W",), replicas=("R0",)))
    t.submit(FusionCommand(1, FusionOp.COPY, inputs=("W",), outputs=("W",), replicas=("R1",)))
    assert t.replica_state("W") == {"R0": "CURRENT", "R1": "CURRENT"}
    assert t.object_version("W") == 0

    t.submit(FusionCommand(2, FusionOp.ACQUIRE_WRITE, inputs=("W",), replicas=("R0",)))
    t.submit(FusionCommand(3, FusionOp.RELEASE, inputs=("W",), replicas=("R0",),
                           depends_on=(2,)))

    assert t.object_version("W") == 1
    assert t.replica_state("W") == {"R0": "CURRENT", "R1": "STALE"}


def test_acquire_read_then_release_does_not_bump_version_or_stale_anything():
    """The negative control for the write/release test above -- same starting
    state, a read acquire/release cycle instead, and neither the version nor
    any replica may move."""
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.MATERIALIZE, outputs=("W",), replicas=("R0",)))
    t.submit(FusionCommand(1, FusionOp.COPY, inputs=("W",), outputs=("W",), replicas=("R1",)))

    t.submit(FusionCommand(2, FusionOp.ACQUIRE_READ, inputs=("W",), replicas=("R0",)))
    t.submit(FusionCommand(3, FusionOp.RELEASE, inputs=("W",), replicas=("R0",),
                           depends_on=(2,)))

    assert t.object_version("W") == 0
    assert t.replica_state("W") == {"R0": "CURRENT", "R1": "CURRENT"}


def test_release_without_acquire_is_refused():
    t = FusionTimeline()
    with pytest.raises(ReleaseWithoutAcquireError):
        t.submit(FusionCommand(0, FusionOp.RELEASE, inputs=("W",), replicas=("R0",)))


def test_second_acquire_write_conflicts_with_first():
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.ACQUIRE_WRITE, inputs=("W",), replicas=("R0",)))
    with pytest.raises(LockConflictError):
        t.submit(FusionCommand(1, FusionOp.ACQUIRE_WRITE, inputs=("W",), replicas=("R1",)))


def test_acquire_write_after_release_is_fine():
    """The positive control for the conflict test above: the SAME sequence
    succeeds once the first writer has released."""
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.ACQUIRE_WRITE, inputs=("W",), replicas=("R0",)))
    t.submit(FusionCommand(1, FusionOp.RELEASE, inputs=("W",), replicas=("R0",),
                           depends_on=(0,)))
    t.submit(FusionCommand(2, FusionOp.ACQUIRE_WRITE, inputs=("W",), replicas=("R1",)))


def test_acquire_read_conflicts_with_open_write_lock():
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.ACQUIRE_WRITE, inputs=("W",), replicas=("R0",)))
    with pytest.raises(LockConflictError):
        t.submit(FusionCommand(1, FusionOp.ACQUIRE_READ, inputs=("W",), replicas=("R1",)))


def test_acquire_write_conflicts_with_open_read_lock():
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.ACQUIRE_READ, inputs=("W",), replicas=("R0",)))
    with pytest.raises(LockConflictError):
        t.submit(FusionCommand(1, FusionOp.ACQUIRE_WRITE, inputs=("W",), replicas=("R1",)))


def test_two_concurrent_read_locks_do_not_conflict():
    """The negative control for the two lock-conflict tests above: reads
    don't exclude other reads, only a writer excludes and is excluded."""
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.ACQUIRE_READ, inputs=("W",), replicas=("R0",)))
    t.submit(FusionCommand(1, FusionOp.ACQUIRE_READ, inputs=("W",), replicas=("R1",)))


# --------------------------------------------------------------------------- fence/wait

def test_wait_returns_the_submitted_command():
    t = FusionTimeline()
    cmd = FusionCommand(0, FusionOp.FENCE)
    t.submit(cmd)
    assert t.wait(0) == cmd


def test_wait_on_unknown_seq_is_a_named_refusal():
    t = FusionTimeline()
    with pytest.raises(UnknownCommandIdError, match="99"):
        t.wait(99)


def test_fence_reports_commands_through_a_point():
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.FENCE))
    t.submit(FusionCommand(1, FusionOp.FENCE))
    t.submit(FusionCommand(2, FusionOp.FENCE))
    result = t.fence(upto=1)
    assert result["commands_complete"] == (0, 1)
    assert result["fenced_through"] == 1


def test_fence_with_no_upto_covers_everything_submitted():
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.FENCE))
    t.submit(FusionCommand(1, FusionOp.FENCE))
    result = t.fence()
    assert result["commands_complete"] == (0, 1)


def test_fence_on_unknown_upto_is_a_named_refusal():
    t = FusionTimeline()
    with pytest.raises(UnknownCommandIdError, match="42"):
        t.fence(upto=42)


def test_fence_on_empty_timeline_reports_nothing_fenced():
    t = FusionTimeline()
    result = t.fence()
    assert result == {"fenced_through": None, "commands_complete": ()}


# ------------------------------------------------------------------ other op effects

def test_evict_removes_replica_state():
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.MATERIALIZE, outputs=("W",), replicas=("R0",)))
    assert t.replica_state("W") == {"R0": "CURRENT"}
    t.submit(FusionCommand(1, FusionOp.EVICT, inputs=("W",), replicas=("R0",)))
    assert t.replica_state("W") == {}


def test_evict_of_a_replica_that_was_never_there_is_a_harmless_noop():
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.EVICT, inputs=("W",), replicas=("R9",)))
    assert t.replica_state("W") == {}


def test_invalidate_marks_named_replicas_absent_and_leaves_others_alone():
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.SCATTER, inputs=("W",), outputs=("W",),
                           replicas=("R0", "R1", "R2")))
    assert t.replica_state("W") == {"R0": "CURRENT", "R1": "CURRENT", "R2": "CURRENT"}
    t.submit(FusionCommand(1, FusionOp.INVALIDATE, inputs=("W",), replicas=("R0", "R2")))
    assert t.replica_state("W") == {"R0": "ABSENT", "R1": "CURRENT", "R2": "ABSENT"}


def test_reduce_creates_the_output_replica():
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.REDUCE, inputs=("A", "B"), outputs=("C",),
                           replicas=("R0",)))
    assert t.replica_state("C") == {"R0": "CURRENT"}
    assert t.object_version("C") == 0


def test_gather_creates_the_output_at_every_named_replica():
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.GATHER, inputs=("A", "B", "C"), outputs=("D",),
                           replicas=("R0", "R1")))
    assert t.replica_state("D") == {"R0": "CURRENT", "R1": "CURRENT"}


def test_prefetch_submit_fence_digest_do_not_mutate_bookkeeping():
    """The four record-only ops (see fusion_isa's NOT IMPLEMENTED note): a
    positive test that they run without error, paired with the assertion
    that is actually the point -- nothing about version or replica state
    moved, because none of the four have a real backend behind them here."""
    t = FusionTimeline()
    t.submit(FusionCommand(0, FusionOp.PREFETCH, inputs=("W",), replicas=("R0",)))
    t.submit(FusionCommand(1, FusionOp.SUBMIT, inputs=("W",), outputs=("X",)))
    t.submit(FusionCommand(2, FusionOp.FENCE))
    t.submit(FusionCommand(3, FusionOp.DIGEST, inputs=("W",)))
    assert t.replica_state("W") == {}
    assert t.replica_state("X") == {}
    assert t.object_version("W") == 0
    assert len(t.commands()) == 4   # all four were still recorded


# ----------------------------------------------------------- wire <-> ISA bridge

def test_to_packet_and_op_from_packet_round_trip():
    cmd = FusionCommand(0, FusionOp.COPY, inputs=("W",), outputs=("W",), replicas=("R1",))
    pkt = cmd.to_packet(object_id=42, byte_offset=100, length=200, object_version=3,
                        representation_id=1, flags=0, dependency_epoch=9)
    wire = pkt.encode()
    back = decode(wire)
    assert op_from_packet(back) is FusionOp.COPY
    assert back.object_id == 42
    assert back.length == 200
    assert back.object_version == 3


def test_op_from_packet_refuses_a_command_id_with_no_matching_op():
    pkt = FusionPacket(command_id=250, object_id=0)   # 250 is not one of the 14 ops
    back = decode(pkt.encode())
    with pytest.raises(FusionIsaError, match="250"):
        op_from_packet(back)


def test_op_from_packet_accepts_every_real_op():
    for op in FusionOp:
        pkt = FusionPacket(command_id=int(op), object_id=0)
        back = decode(pkt.encode())
        assert op_from_packet(back) is op
