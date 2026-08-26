"""Fusion ISA -- the logical command protocol half of G054. fusion_wire.py is
the byte encoding; this module is what the bytes MEAN. A different lane owns
the memory-fabric half (tools/accelerator/humf.py: domains, object identity,
coherence state, trust) -- this module does not import or modify it, and does
not reimplement HUMF's coherence engine. What it borrows is HUMF's vocabulary
(an object, a replica/materialization of it, a version, staleness) at the
scale of ONE command timeline, not a whole fabric.

FOURTEEN COMMANDS, on purpose kept small:
    ACQUIRE_READ, ACQUIRE_WRITE, RELEASE, PREFETCH, COPY, MATERIALIZE,
    INVALIDATE, REDUCE, SCATTER, GATHER, SUBMIT, FENCE, DIGEST, EVICT.

Two names collide on purpose and are disambiguated here, not papered over:
  - FusionOp.SUBMIT is an ISA command (dispatch a workload/kernel that
    consumes some objects and produces others). FusionTimeline.submit() is
    the unrelated verb for enqueuing ANY command -- including a SUBMIT
    command -- onto the timeline. Same English word, two grammatical roles,
    the way "copy" is both a verb and one of these ops.
  - FusionOp.FENCE is a barrier MARKER a caller can insert into the recorded
    command sequence. FusionTimeline.fence() is the host-side synchronization
    call. Inserting a FENCE command records a point in the timeline; calling
    .fence() is what a caller actually does to synchronize with it.

THE UPWARD API IS EXACTLY submit() / fence() / wait() on FusionTimeline.
Nothing named metalLaunch, cudaLaunch, or sendPacket exists here or anywhere
below it -- those are provider details, and no provider is wired in this
module (see NOT IMPLEMENTED). FusionTimeline's other public methods
(object_version, replica_state, commands) are read-only queries, not a second
dispatch surface.

NOT IMPLEMENTED, named rather than left silent:
  - No asynchronous execution backend. submit() applies each command's
    bookkeeping side effects SYNCHRONOUSLY, because there is no real backend
    here to defer to -- no external GPU, no Spark, nothing to hand a command
    to and get an interrupt back from later. fence() and wait() therefore do
    not block on anything; they report/look up what already happened. A real
    backend would need both to actually block, and that is the seam a
    provider-backed implementation would fill in.
  - PREFETCH, SUBMIT, FENCE, and DIGEST are recorded in the timeline but do
    NOT mutate replica or version bookkeeping (see FusionTimeline._apply).
    PREFETCH is a hint with nothing behind it to move bytes early. DIGEST
    does not compute a real digest -- there are no real bytes on this wire to
    hash, only a byte RANGE description; see fusion_wire.py's own note that
    packets never carry payload. SUBMIT and FENCE record dispatch/barrier
    points for a backend that does not exist yet.
  - No cross-object consistency modeling beyond locks and the
    version/staleness bookkeeping that ACQUIRE_WRITE/RELEASE and
    INVALIDATE/EVICT perform. COPY, MATERIALIZE, REDUCE, SCATTER, and GATHER
    do not check that their declared source replicas are actually CURRENT --
    that is HUMF's job over a real fabric, not this module's over a command
    list.
  - FusionCommand.to_packet()/from_packet() round-trip only ONE numeric
    object reference per packet, matching fusion_wire's fixed 42-byte,
    single-object layout. A multi-object command (REDUCE, SCATTER, GATHER,
    a multi-input/output SUBMIT) cannot be fully represented by one
    FusionPacket; splitting it across several packets or defining an
    envelope format is not designed here.
  - No bandwidth or latency number appears anywhere in this module. There is
    nothing to measure -- see fusion_wire.py's own note.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import NamedTuple

from fusion_wire import FusionPacket


class FusionOp(IntEnum):
    """Values double as fusion_wire's command_id byte, by convention only --
    fusion_wire itself treats command_id as an opaque u8 and does not import
    this enum."""
    ACQUIRE_READ = 0
    ACQUIRE_WRITE = 1
    RELEASE = 2
    PREFETCH = 3
    COPY = 4
    MATERIALIZE = 5
    INVALIDATE = 6
    REDUCE = 7
    SCATTER = 8
    GATHER = 9
    SUBMIT = 10
    FENCE = 11
    DIGEST = 12
    EVICT = 13


class FusionIsaError(RuntimeError):
    """Base for every error this module raises."""


class ArityError(FusionIsaError):
    """A command's inputs/outputs/replicas do not fit what its op declares."""


class DuplicateCommandIdError(FusionIsaError):
    """A seq was submitted twice. Refused, not overwritten or reordered."""


class ForwardReferenceError(FusionIsaError):
    """depends_on names a seq that has not been submitted yet."""


class LockConflictError(FusionIsaError):
    """An ACQUIRE collides with a lock already held on the same object."""


class ReleaseWithoutAcquireError(FusionIsaError):
    """RELEASE named an (object, replica) with no open lock to release."""


class UnknownCommandIdError(FusionIsaError):
    """fence()/wait() named a seq that was never submitted."""


class OpSpec(NamedTuple):
    min_in: int
    max_in: int | None
    min_out: int
    max_out: int | None
    min_replicas: int
    max_replicas: int | None
    side_effects: tuple[str, ...]


# min/max = None means unbounded. This is the "each command declares its
# object inputs, object outputs, dependencies and side effects" requirement:
# side_effects is intrinsic to the OP, not something a caller can assert for
# an instance -- see FusionCommand.side_effects below.
OP_SPEC: dict[FusionOp, OpSpec] = {
    FusionOp.ACQUIRE_READ:  OpSpec(1, 1, 0, 0, 1, 1, ("SHARED_LOCK",)),
    FusionOp.ACQUIRE_WRITE: OpSpec(1, 1, 0, 0, 1, 1, ("EXCLUSIVE_LOCK",)),
    FusionOp.RELEASE:       OpSpec(1, 1, 0, 0, 1, 1,
                                   ("MAY_BUMP_VERSION", "MAY_STALE_OTHER_REPLICAS")),
    FusionOp.PREFETCH:      OpSpec(1, 1, 0, 0, 1, 1, ("HINT_ONLY",)),
    FusionOp.COPY:          OpSpec(1, 1, 1, 1, 1, 1, ("REPLICA_UPDATED",)),
    FusionOp.MATERIALIZE:   OpSpec(0, 1, 1, 1, 1, 1, ("REPLICA_CREATED",)),
    FusionOp.INVALIDATE:    OpSpec(1, 1, 0, 0, 1, None, ("REPLICA_INVALIDATED",)),
    FusionOp.REDUCE:        OpSpec(2, None, 1, 1, 1, 1, ("OUTPUT_CREATED",)),
    FusionOp.SCATTER:       OpSpec(1, 1, 1, 1, 1, None, ("REPLICAS_UPDATED",)),
    FusionOp.GATHER:        OpSpec(1, None, 1, 1, 1, None, ("OUTPUT_CREATED",)),
    FusionOp.SUBMIT:        OpSpec(0, None, 0, None, 0, None, ("WORKLOAD_DISPATCHED",)),
    FusionOp.FENCE:         OpSpec(0, 0, 0, 0, 0, 0, ("BARRIER",)),
    FusionOp.DIGEST:        OpSpec(1, 1, 0, 0, 0, 1, ("INTEGRITY_CHECK_REQUESTED",)),
    FusionOp.EVICT:         OpSpec(1, 1, 0, 0, 1, 1, ("REPLICA_FREED",)),
}


def _check_arity(field_name: str, values: tuple, lo: int, hi: int | None,
                 cmd: "FusionCommand") -> None:
    n = len(values)
    if n < lo or (hi is not None and n > hi):
        bound = f"[{lo}, {hi if hi is not None else '∞'}]"
        raise ArityError(
            f"command {cmd.seq} ({cmd.op.name}) has {n} {field_name} ({values!r}), "
            f"outside the declared bound {bound} for this op")


@dataclass(frozen=True)
class FusionCommand:
    """One entry in a FusionTimeline. `inputs`/`outputs` are object ids the
    command reads/produces; `replicas` names which replica(s)/location(s) it
    targets; `depends_on` names the seq of every command that must already be
    in the timeline before this one is recorded (see FusionTimeline.submit).

    Validated at construction (fail closed, matching humf.Materialization's
    transition() and air.AirGraph.validate()) -- an invalid command cannot
    exist, not just fail to submit."""
    seq: int
    op: FusionOp
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    replicas: tuple[str, ...] = ()
    depends_on: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        spec = OP_SPEC[self.op]
        _check_arity("inputs", self.inputs, spec.min_in, spec.max_in, self)
        _check_arity("outputs", self.outputs, spec.min_out, spec.max_out, self)
        _check_arity("replicas", self.replicas, spec.min_replicas, spec.max_replicas, self)

    @property
    def side_effects(self) -> tuple[str, ...]:
        """Declared by the OP, not settable per instance -- a command cannot
        claim a side effect its op does not have."""
        return OP_SPEC[self.op].side_effects

    def to_packet(self, object_id: int, *, byte_offset: int = 0, length: int = 0,
                  object_version: int = 0, representation_id: int = 0,
                  flags: int = 0, dependency_epoch: int = 0) -> FusionPacket:
        """Wire-encode a REFERENCE to this command's operation against ONE
        object. See the module docstring: a single FusionPacket cannot carry
        a multi-object command's full input/output list -- the caller picks
        which object_id this packet is about."""
        return FusionPacket(command_id=int(self.op), object_id=object_id,
                            byte_offset=byte_offset, length=length,
                            object_version=object_version,
                            representation_id=representation_id, flags=flags,
                            dependency_epoch=dependency_epoch)


def op_from_packet(pkt: FusionPacket) -> FusionOp:
    """The other half of to_packet(): recover the op from a decoded packet's
    command_id. Raises ValueError (via FusionOp's own enum lookup) if the
    byte is not one of the fourteen known ops -- fusion_wire.decode() does not
    validate command_id against ISA semantics (it is a wire-layer concern
    only), so that check lives here instead."""
    try:
        return FusionOp(pkt.command_id)
    except ValueError as e:
        raise FusionIsaError(
            f"packet names command_id={pkt.command_id}, which is not one of "
            f"the {len(FusionOp)} known Fusion ISA ops") from e


class FusionTimeline:
    """Records FusionCommands in submission order; see module docstring for
    why submit() applies bookkeeping synchronously and what fence()/wait()
    do instead of blocking."""

    def __init__(self) -> None:
        self._commands: dict[int, FusionCommand] = {}
        self._order: list[int] = []
        self._object_version: dict[str, int] = {}
        self._replica_state: dict[str, dict[str, str]] = {}
        self._write_lock: dict[str, tuple[int, str]] = {}
        self._read_locks: dict[str, set[tuple[int, str]]] = {}

    # ------------------------------------------------------------- upward API

    def submit(self, cmd: FusionCommand) -> int:
        """Record one command. REFUSES a duplicate command id and REFUSES a
        forward reference in depends_on rather than silently reordering --
        matching humf.py's illegal-transition refusal and air.AirGraph's
        forward-reference refusal for their own graphs."""
        if cmd.seq in self._commands:
            raise DuplicateCommandIdError(
                f"command id {cmd.seq} was already submitted as "
                f"{self._commands[cmd.seq].op.name}; refusing to silently "
                f"reorder or overwrite it with {cmd.op.name}")
        for dep in cmd.depends_on:
            if dep not in self._commands:
                raise ForwardReferenceError(
                    f"command {cmd.seq} ({cmd.op.name}) depends_on {dep}, which "
                    f"has not been submitted yet; the timeline is recorded in "
                    f"submission order and a forward reference is either a "
                    f"cycle or a typo, refused rather than reordered")
        self._apply(cmd)
        self._commands[cmd.seq] = cmd
        self._order.append(cmd.seq)
        return cmd.seq

    def fence(self, upto: int | None = None) -> dict:
        """Synchronize with everything submitted so far (or through `upto`).
        There is nothing to block on -- see module docstring -- so this
        confirms and reports rather than waits."""
        if upto is not None and upto not in self._commands:
            raise UnknownCommandIdError(
                f"fence(upto={upto}) names a command id that was never submitted")
        seqs = self._order if upto is None else [s for s in self._order if s <= upto]
        return {"fenced_through": seqs[-1] if seqs else None,
                "commands_complete": tuple(seqs)}

    def wait(self, seq: int) -> FusionCommand:
        """Wait for one specific command. There is no asynchrony to wait on --
        see module docstring -- so this is a lookup that fails loudly for an
        unknown id rather than hanging or returning a default."""
        if seq not in self._commands:
            raise UnknownCommandIdError(
                f"wait({seq}) names a command id that was never submitted")
        return self._commands[seq]

    # -------------------------------------------------------- read-only queries

    def object_version(self, obj: str) -> int:
        return self._object_version.get(obj, 0)

    def replica_state(self, obj: str) -> dict[str, str]:
        return dict(self._replica_state.get(obj, {}))

    def commands(self) -> tuple[FusionCommand, ...]:
        return tuple(self._commands[s] for s in self._order)

    # ------------------------------------------------------------------ apply

    def _touch_replica(self, obj: str, replica: str, state: str = "CURRENT") -> None:
        self._replica_state.setdefault(obj, {})[replica] = state

    def _apply(self, cmd: FusionCommand) -> None:
        op = cmd.op
        if op is FusionOp.ACQUIRE_READ:
            (obj,), (replica,) = cmd.inputs, cmd.replicas
            if obj in self._write_lock:
                seq, held_at = self._write_lock[obj]
                raise LockConflictError(
                    f"{obj} is exclusively locked by command {seq} at {held_at}; "
                    f"ACQUIRE_READ at {replica} refused until RELEASE")
            self._read_locks.setdefault(obj, set()).add((cmd.seq, replica))
            self._touch_replica(obj, replica)

        elif op is FusionOp.ACQUIRE_WRITE:
            (obj,), (replica,) = cmd.inputs, cmd.replicas
            if obj in self._write_lock:
                seq, held_at = self._write_lock[obj]
                raise LockConflictError(
                    f"{obj} is already exclusively locked by command {seq} at "
                    f"{held_at}; refusing a second ACQUIRE_WRITE before RELEASE")
            if self._read_locks.get(obj):
                raise LockConflictError(
                    f"{obj} has open read lock(s) {sorted(self._read_locks[obj])}; "
                    f"ACQUIRE_WRITE refused until every reader RELEASEs")
            self._write_lock[obj] = (cmd.seq, replica)
            self._touch_replica(obj, replica)

        elif op is FusionOp.RELEASE:
            (obj,), (replica,) = cmd.inputs, cmd.replicas
            if obj in self._write_lock and self._write_lock[obj][1] == replica:
                del self._write_lock[obj]
                # THE version bump and staleness fan-out this module exists to
                # pin: a write, once released, makes every OTHER replica stale
                # and bumps the object's version. ACQUIRE_READ + RELEASE never
                # reaches this branch.
                self._object_version[obj] = self._object_version.get(obj, 0) + 1
                state = self._replica_state.setdefault(obj, {})
                for r in list(state):
                    if r != replica:
                        state[r] = "STALE"
                state[replica] = "CURRENT"
                return
            matching_read = next(
                (e for e in self._read_locks.get(obj, ()) if e[1] == replica), None)
            if matching_read is not None:
                self._read_locks[obj].discard(matching_read)
                return
            raise ReleaseWithoutAcquireError(
                f"RELEASE of {obj} at {replica} has no matching open ACQUIRE_READ "
                f"or ACQUIRE_WRITE lock to release")

        elif op is FusionOp.COPY:
            (obj,) = cmd.inputs
            (replica,) = cmd.replicas
            self._touch_replica(obj, replica, "CURRENT")

        elif op is FusionOp.MATERIALIZE:
            (obj,) = cmd.outputs
            (replica,) = cmd.replicas
            self._object_version.setdefault(obj, 0)
            self._touch_replica(obj, replica, "CURRENT")

        elif op is FusionOp.INVALIDATE:
            (obj,) = cmd.inputs
            for replica in cmd.replicas:
                self._touch_replica(obj, replica, "ABSENT")

        elif op is FusionOp.REDUCE:
            (obj,) = cmd.outputs
            (replica,) = cmd.replicas
            self._object_version.setdefault(obj, 0)
            self._touch_replica(obj, replica, "CURRENT")

        elif op is FusionOp.SCATTER:
            (obj,) = cmd.outputs
            for replica in cmd.replicas:
                self._touch_replica(obj, replica, "CURRENT")

        elif op is FusionOp.GATHER:
            (obj,) = cmd.outputs
            for replica in cmd.replicas:
                self._touch_replica(obj, replica, "CURRENT")

        elif op is FusionOp.EVICT:
            (obj,) = cmd.inputs
            (replica,) = cmd.replicas
            self._replica_state.get(obj, {}).pop(replica, None)

        elif op in (FusionOp.PREFETCH, FusionOp.SUBMIT, FusionOp.FENCE, FusionOp.DIGEST):
            # Recorded, deliberately inert -- see NOT IMPLEMENTED in the module
            # docstring for why each of these four does not mutate state here.
            return

        else:  # pragma: no cover -- OP_SPEC covers every FusionOp value
            raise FusionIsaError(f"no apply() case for {op.name}")
