"""HMF/HGVAS managed-object semantics that never lie about coherence.

This is a SEMANTIC OVERLAY on the canonical Hawking Memory Fabric
(`tools/accelerator/humf.py`, re-exported as `tools/accelerator/hmf.py`).
It is not a second fabric. HUMF already owns copy-state, ownership,
transfer execution, resident digest, quarantine, and the move-vs-recompute
planner that always picks the cheaper *known* number. This module adds
the object-level contract those files do not enforce:

  * every managed object carries identity, owner, semantic representation,
    physical materialization, location (memory tier + device), device
    visibility set, version, trust, and state in {CLEAN, DIRTY, STALE,
    UNKNOWN}
  * ``is_coherent(obj)`` is tri-state: COHERENT / NOT_COHERENT / UNKNOWN
  * an object whose coherence is UNKNOWN can never be consumed by a path
    that requires coherence (``require_coherent`` / ``consume_for_kernel``)
  * crossing a kernel boundary without synchronisation moves CLEAN to
    UNKNOWN, not CLEAN
  * ``move_vs_recompute`` returns UNDECIDABLE when either cost is UNKNOWN
    -- the layer never guesses that moving is cheaper
  * HGVAS identity is ``{object_id, byte_offset}`` and refuses a native
    pointer (Mac VA / CUDA / Metal buffer pointer) as semantic identity

    python3 tools/future/hmf_objects.py --selftest
    python3 tools/future/hmf_objects.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO


import argparse
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

RECEIPT = "HMF_MANAGED_OBJECTS.json"
SCHEMA = "hawking.future.hmf_objects.v1"

# ---------------------------------------------------------------------------
# Vocabulary. Object-level state is the four-member VIEW of HUMF's richer
# copy-state machine. HUMF remains canonical for copy-state; we do not
# re-declare ABSENT/MATERIALIZING/TRANSFERRING/INVALID/EVICTED/CORRUPT/LOST.
# ---------------------------------------------------------------------------

class ObjectState(str, Enum):
    CLEAN = "CLEAN"
    DIRTY = "DIRTY"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class Coherence(str, Enum):
    """Tri-state. Collapsing this to bool is refused: UNKNOWN is not False."""
    COHERENT = "COHERENT"
    NOT_COHERENT = "NOT_COHERENT"
    UNKNOWN = "UNKNOWN"

    def __bool__(self) -> bool:  # type: ignore[override]
        raise CoherenceBooleanError(
            "is_coherent() is tri-state; refusing to collapse "
            f"{self.value} to a bool. Compare to Coherence.COHERENT or call "
            "require_coherent() so UNKNOWN is handled explicitly."
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, bool):
            raise CoherenceBooleanError(
                f"refusing {self.value} == {other!r}; tri-state coherence "
                "is not a boolean"
            )
        return super().__eq__(other)

    def __hash__(self) -> int:  # type: ignore[override]
        return super().__hash__()


class Trust(str, Enum):
    """Matches HUMF's per-copy trust vocabulary, at object level."""
    TRUSTED = "TRUSTED"
    ASSERTED = "ASSERTED"
    UNKNOWN = "UNKNOWN"


class MemoryTier(str, Enum):
    UMA = "UMA"
    HBM = "HBM"
    ACCELERATOR_SRAM = "ACCELERATOR_SRAM"
    HOST_DRAM = "HOST_DRAM"
    REMOTE_MACHINE = "REMOTE_MACHINE"
    UNKNOWN = "UNKNOWN"


class Decision(str, Enum):
    MOVE = "MOVE"
    RECOMPUTE = "RECOMPUTE"
    TIE = "TIE"
    UNDECIDABLE = "UNDECIDABLE"
    NEITHER = "NEITHER"


class KernelAccess(str, Enum):
    READ = "READ"
    WRITE = "WRITE"
    UNKNOWN = "UNKNOWN"


Cost = float | Literal["UNKNOWN", "UNAVAILABLE"]
Visibility = frozenset[str] | Literal["UNKNOWN"]


# Fail-closed object-level transitions. UNKNOWN is reachable from every
# known state (loss of knowledge is always legal). CLEAN is *not* a
# legal successor of anything on the public transition() path -- it is
# earned by establish_clean / confirmed migrate / a declared-sync WRITE,
# which assign state directly after their own evidence checks. A caller
# who writes obj.transition(CLEAN) from UNKNOWN is trying to skip that.
OBJECT_LEGAL: dict[ObjectState, frozenset[ObjectState]] = {
    ObjectState.CLEAN:   frozenset({ObjectState.DIRTY, ObjectState.STALE, ObjectState.UNKNOWN}),
    ObjectState.DIRTY:   frozenset({ObjectState.STALE, ObjectState.UNKNOWN}),
    ObjectState.STALE:   frozenset({ObjectState.DIRTY, ObjectState.UNKNOWN}),
    ObjectState.UNKNOWN: frozenset({ObjectState.DIRTY, ObjectState.STALE}),
}


# HUMF copy-state -> object-level four-state projection. In-flight is
# UNKNOWN (we do not yet know); known-absent/known-bad is STALE so
# is_coherent returns NOT_COHERENT rather than inventing knowledge.
HUMF_STATE_PROJECTION: dict[str, ObjectState] = {
    "CLEAN": ObjectState.CLEAN,
    "DIRTY": ObjectState.DIRTY,
    "STALE": ObjectState.STALE,
    "UNKNOWN": ObjectState.UNKNOWN,
    "MATERIALIZING": ObjectState.UNKNOWN,
    "TRANSFERRING": ObjectState.UNKNOWN,
    "ABSENT": ObjectState.STALE,
    "EVICTED": ObjectState.STALE,
    "INVALID": ObjectState.STALE,
    "CORRUPT": ObjectState.STALE,
    "LOST": ObjectState.STALE,
}


HUMF_DOMAIN_LOCATION: dict[str, tuple[MemoryTier, str]] = {
    "APPLE_UM": (MemoryTier.UMA, "APPLE_DOMAIN_0"),
}


class ManagedObjectError(RuntimeError):
    """Fail-closed overlay error. Distinct from humf.HumfError on purpose."""


class CoherenceRequiredError(ManagedObjectError):
    """A reader required coherence and did not have it."""


class CoherenceBooleanError(ManagedObjectError):
    """Someone tried to use tri-state coherence as a bool."""


class HgvasError(ManagedObjectError):
    """Semantic identity leaked a native pointer, or was malformed."""


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HgvasRef:
    """One logical address in the HGVAS universe.

    G054 names FusionPtr{object_id, byte_offset} and forbids leaking a Mac
    virtual address, a CUDA pointer or a Metal buffer pointer into semantic
    identity. This overlay enforces that refusal. It is not a page directory,
    not a Fusion ISA, and not a wire format -- those remain G054's job.
    """
    object_id: str
    byte_offset: int = 0

    def __post_init__(self) -> None:
        if self.byte_offset < 0:
            raise HgvasError(f"byte_offset must be >= 0, got {self.byte_offset}")
        if not self.object_id or not isinstance(self.object_id, str):
            raise HgvasError("object_id must be a non-empty string")
        _refuse_native_pointer(self.object_id)


def _refuse_native_pointer(object_id: str) -> None:
    s = object_id.strip()
    low = s.lower()
    if low.startswith("0x") and all(c in "0123456789abcdef" for c in low[2:]) and len(low) > 4:
        raise HgvasError(
            f"HGVAS identity {object_id!r} looks like a native pointer; "
            "FusionPtr is {{object_id, byte_offset}}, never a Mac VA / "
            "CUDA pointer / Metal buffer pointer"
        )
    for needle in ("mtlbuffer", "cuda", "metal_buffer", "deviceptr", "vmaddr"):
        if needle in low:
            raise HgvasError(
                f"HGVAS identity {object_id!r} contains native-pointer residue "
                f"{needle!r}; refusing to let a device address become identity"
            )


@dataclass(frozen=True)
class Location:
    memory_tier: MemoryTier
    device: str

    def to_dict(self) -> dict[str, str]:
        return {"memory_tier": self.memory_tier.value, "device": self.device}


@dataclass(frozen=True)
class PhysicalMaterialization:
    """How the object is laid out where it currently claims to live.

    ``digest`` is the sealed identity of the bytes, or None if nobody has
    digested them. Absence of a digest is absence of evidence, not a hash
    of empty bytes.
    """
    representation: str
    layout: str
    nbytes: int | None
    digest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "layout": self.layout,
            "nbytes": self.nbytes,
            "digest": self.digest,
        }


@dataclass
class ManagedObject:
    """Object-level record the fabric's HumfObject does not carry as a unit.

    Defaults are fail-closed: a newly constructed object is UNKNOWN on every
    axis we do not have evidence for. CLEAN is earned, never assumed.
    """
    identity: HgvasRef
    owner: str | None = None
    semantic_representation: str = "unspecified"
    physical: PhysicalMaterialization = field(
        default_factory=lambda: PhysicalMaterialization("unspecified", "unspecified", None, None)
    )
    location: Location = field(
        default_factory=lambda: Location(MemoryTier.UNKNOWN, "UNKNOWN")
    )
    device_visibility: Visibility = "UNKNOWN"
    version: int = 0
    trust: Trust = Trust.UNKNOWN
    state: ObjectState = ObjectState.UNKNOWN
    evidence: str | None = None
    payload: bytes | None = None
    # Optional binding back to a HUMF identity, so this is an overlay
    # rather than a competing object universe.
    humf_identity: str | None = None
    recompute_recipe: bool = False
    recompute_cost: Cost = "UNKNOWN"
    transfer_cost: Cost = "UNKNOWN"

    def transition(self, to: ObjectState) -> None:
        if to is self.state:
            return
        if to not in OBJECT_LEGAL[self.state]:
            raise ManagedObjectError(
                f"illegal object-state transition {self.state.value} -> {to.value} "
                f"for {self.identity.object_id}"
            )
        self.state = to

    def to_dict(self) -> dict[str, Any]:
        vis: Any
        if self.device_visibility == "UNKNOWN":
            vis = "UNKNOWN"
        else:
            vis = sorted(self.device_visibility)
        return {
            "identity": {
                "object_id": self.identity.object_id,
                "byte_offset": self.identity.byte_offset,
            },
            "owner": self.owner,
            "semantic_representation": self.semantic_representation,
            "physical_materialization": self.physical.to_dict(),
            "location": self.location.to_dict(),
            "device_visibility": vis,
            "version": self.version,
            "trust": self.trust.value,
            "state": self.state.value,
            "evidence": self.evidence,
            "humf_identity": self.humf_identity,
        }


# ---------------------------------------------------------------------------
# Coherence invariant
# ---------------------------------------------------------------------------

def is_coherent(obj: ManagedObject, *, on_device: str | None = None) -> Coherence:
    """THE load-bearing query. Tri-state. Never a bare bool.

    COHERENT only when we *know* the live value is good:
      state CLEAN, trust TRUSTED, visibility known, and (if asked) the
      calling device is in the visibility set.

    UNKNOWN when any required axis is unknown -- including CLEAN copies
    whose trust is UNKNOWN or ASSERTED. ASSERTED is an operator's word,
    not knowledge; reporting it as COHERENT would be the lie this lane
    exists to prevent.

    NOT_COHERENT when we *know* the value is not usable (DIRTY / STALE,
    or a known visibility set that does not include the calling device).
    Known-bad is checked before unknown-trust: a mismatched digest that
    moved the object to STALE is NOT_COHERENT, not UNKNOWN.
    """
    if obj.state in (ObjectState.DIRTY, ObjectState.STALE):
        return Coherence.NOT_COHERENT
    if obj.state is ObjectState.UNKNOWN:
        return Coherence.UNKNOWN
    if obj.trust is Trust.UNKNOWN:
        return Coherence.UNKNOWN
    if obj.device_visibility == "UNKNOWN":
        return Coherence.UNKNOWN
    if obj.location.memory_tier is MemoryTier.UNKNOWN or obj.location.device == "UNKNOWN":
        return Coherence.UNKNOWN
    if on_device is not None and on_device not in obj.device_visibility:
        return Coherence.NOT_COHERENT
    if obj.trust is Trust.ASSERTED:
        return Coherence.UNKNOWN
    if obj.state is ObjectState.CLEAN and obj.trust is Trust.TRUSTED:
        return Coherence.COHERENT
    return Coherence.UNKNOWN


def require_coherent(
    obj: ManagedObject,
    *,
    reader: str,
    on_device: str | None = None,
) -> None:
    """The only legal consume gate. UNKNOWN and NOT_COHERENT both refuse.

    A guard nobody has watched fail is not a guard -- tests must fire this.
    """
    verdict = is_coherent(obj, on_device=on_device)
    if verdict is Coherence.COHERENT:
        return
    raise CoherenceRequiredError(
        f"{reader} requires coherence of {obj.identity.object_id} but "
        f"is_coherent={verdict.value} (state={obj.state.value}, "
        f"trust={obj.trust.value}, visibility="
        f"{'UNKNOWN' if obj.device_visibility == 'UNKNOWN' else sorted(obj.device_visibility)}, "
        f"on_device={on_device!r})"
    )


@dataclass(frozen=True)
class KernelBinding:
    object_id: str
    device: str
    kernel: str
    version: int
    digest: str | None


def consume_for_kernel(
    obj: ManagedObject,
    *,
    kernel: str,
    device: str,
) -> KernelBinding:
    """A kernel argument is a reader that requires coherence."""
    require_coherent(obj, reader=f"kernel:{kernel}", on_device=device)
    return KernelBinding(
        object_id=obj.identity.object_id,
        device=device,
        kernel=kernel,
        version=obj.version,
        digest=obj.physical.digest,
    )


def read_payload(obj: ManagedObject, *, reader: str = "read_payload") -> bytes:
    """Host-side consume path. Same gate as a kernel argument."""
    require_coherent(obj, reader=reader)
    if obj.payload is None:
        raise ManagedObjectError(
            f"{obj.identity.object_id} is COHERENT but holds no payload; "
            "the overlay does not invent bytes"
        )
    return obj.payload


# ---------------------------------------------------------------------------
# Establishing / mutating knowledge
# ---------------------------------------------------------------------------

def establish_clean(
    obj: ManagedObject,
    *,
    location: Location,
    visibility: Iterable[str],
    evidence: str,
    digest: str,
    payload: bytes | None = None,
    owner: str | None = None,
) -> ManagedObject:
    """The only path from UNKNOWN (or STALE/DIRTY) to CLEAN.

    Evidence is required. An empty evidence string is a silent CLEAN, which
    this overlay refuses.
    """
    if not evidence:
        raise ManagedObjectError(
            f"{obj.identity.object_id}: refusing to mark CLEAN without evidence"
        )
    if not digest:
        raise ManagedObjectError(
            f"{obj.identity.object_id}: refusing to mark CLEAN without a digest"
        )
    vis = frozenset(visibility)
    if not vis:
        raise ManagedObjectError(
            f"{obj.identity.object_id}: CLEAN requires a known visibility set"
        )
    if location.memory_tier is MemoryTier.UNKNOWN or location.device == "UNKNOWN":
        raise ManagedObjectError(
            f"{obj.identity.object_id}: CLEAN requires a known location, not UNKNOWN"
        )
    obj.state = ObjectState.CLEAN
    obj.trust = Trust.TRUSTED
    obj.location = location
    obj.device_visibility = vis
    obj.evidence = evidence
    obj.physical = replace(obj.physical, digest=digest)
    obj.payload = payload
    if owner is not None:
        obj.owner = owner
    elif obj.owner is None:
        obj.owner = location.device
    return obj


def mark_written(obj: ManagedObject, *, device: str) -> ManagedObject:
    """A write makes this copy DIRTY. Coherence becomes NOT_COHERENT.

    UNKNOWN -> DIRTY is legal: a write is knowledge ('this device just
    wrote'), even if we did not know the prior bytes.
    """
    obj.transition(ObjectState.DIRTY)
    obj.owner = device
    obj.version += 1
    obj.physical = replace(obj.physical, digest=None)
    obj.evidence = f"written on {device}; unsealed"
    if obj.device_visibility != "UNKNOWN":
        obj.device_visibility = frozenset({device}) | obj.device_visibility
    return obj


def invalidate(obj: ManagedObject, *, reason: str) -> ManagedObject:
    """Explicit invalidation is knowledge of NOT_COHERENT, not of UNKNOWN."""
    obj.transition(ObjectState.STALE)
    obj.evidence = reason
    return obj


def migrate(
    obj: ManagedObject,
    dest: Location,
    *,
    digest: str | None = None,
) -> ManagedObject:
    """Record a location change. Bytes are not moved here -- we have no GPU.

    Without a confirming digest the destination is UNKNOWN: updating
    bookkeeping and then reporting CLEAN is the exact defect HUMF already
    sealed ('a transfer that is only a state change is an assumption').
    With a digest that matches the sealed identity, CLEAN is earned at dest.
    """
    if dest.memory_tier is MemoryTier.UNKNOWN or dest.device == "UNKNOWN":
        raise ManagedObjectError("migrate destination location must be known")
    expected = obj.physical.digest
    if digest is None or expected is None or digest != expected:
        obj.location = dest
        obj.device_visibility = "UNKNOWN"
        obj.trust = Trust.UNKNOWN
        obj.transition(ObjectState.UNKNOWN)
        if digest is None:
            why = "migrate without confirming digest"
        elif expected is None:
            why = "migrate with digest but object had no sealed identity"
        else:
            why = "migrate digest did not match sealed identity"
        obj.evidence = why
        return obj
    obj.location = dest
    obj.device_visibility = frozenset({dest.device})
    obj.owner = dest.device
    obj.trust = Trust.TRUSTED
    obj.state = ObjectState.CLEAN
    obj.evidence = f"migrate confirmed by digest {digest}"
    return obj


@dataclass(frozen=True)
class DeviceDigest:
    status: Literal["VERIFIED", "UNKNOWN", "MISMATCH"]
    digest: str | None
    path: str | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "digest": self.digest,
            "path": self.path,
            "note": self.note,
        }


def device_digest(obj: ManagedObject, provider: Any | None = None) -> DeviceDigest:
    """Ask a device to digest its own memory. Absence of a provider is UNKNOWN.

    A mismatch is knowledge of NOT_COHERENT (object becomes STALE). A
    missing provider, a missing sealed identity, or a provider error is
    UNKNOWN -- we did not learn the bytes were good, and we did not learn
    they were bad.
    """
    fn: Callable[[str], str] | None = None
    if provider is not None:
        fn = getattr(provider, "digest_resident", None)
    if fn is None:
        return DeviceDigest(
            "UNKNOWN", None, None,
            "no provider offering digest_resident; absence of evidence is UNKNOWN, not failure",
        )
    claimed = getattr(provider, "resident_digest_path", "UNDECLARED")
    try:
        seen = fn(obj.identity.object_id)
    except Exception as exc:  # noqa: BLE001 -- provider failure is data, not a crash
        obj.trust = Trust.UNKNOWN
        obj.transition(ObjectState.UNKNOWN)
        obj.evidence = f"device digest raised {type(exc).__name__}"
        return DeviceDigest(
            "UNKNOWN", None, claimed,
            f"provider raised {type(exc).__name__}; state left/moved UNKNOWN because nothing was learned",
        )
    expected = obj.physical.digest
    if expected is None:
        return DeviceDigest(
            "UNKNOWN", seen, claimed,
            "PRESENT_BUT_UNVERIFIABLE: copy answered but no sealed digest exists to compare",
        )
    if seen != expected:
        obj.transition(ObjectState.STALE)
        obj.trust = Trust.UNKNOWN
        obj.evidence = "resident digest mismatched sealed identity"
        return DeviceDigest(
            "MISMATCH", seen, claimed,
            "resident digest != sealed identity; object marked STALE (known not coherent)",
        )
    return DeviceDigest(
        "VERIFIED", seen, claimed,
        "digest matched sealed identity; path is the provider's claim, not independently verified",
    )


# ---------------------------------------------------------------------------
# Kernel-boundary semantics
# ---------------------------------------------------------------------------

def cross_kernel_boundary(
    obj: ManagedObject,
    *,
    synchronized: bool,
    access: KernelAccess = KernelAccess.UNKNOWN,
) -> ManagedObject:
    """What happens to object state when execution crosses a kernel.

    Without a synchronisation the host does not know whether the kernel
    has finished, so the object moves to UNKNOWN -- including when it
    was CLEAN. Leaving it CLEAN would be a coherence lie.

    With a declared synchronisation:
      READ  of CLEAN stays CLEAN (fence makes the prior value visible)
      WRITE commits, version increments, state CLEAN
      UNKNOWN access stays UNKNOWN -- a fence does not reveal what the
      kernel did if nobody said.
    A declared fence is not a digest; WRITE+sync is still CLEAN only
    because the programming model says the write is now host-visible.
    """
    if not synchronized:
        obj.trust = Trust.UNKNOWN
        obj.transition(ObjectState.UNKNOWN)
        obj.evidence = (
            f"kernel boundary crossed without synchronisation (access={access.value})"
        )
        return obj
    if access is KernelAccess.UNKNOWN:
        obj.trust = Trust.UNKNOWN
        obj.transition(ObjectState.UNKNOWN)
        obj.evidence = "kernel boundary synchronized but access was UNKNOWN"
        return obj
    if access is KernelAccess.WRITE:
        obj.version += 1
        obj.physical = replace(obj.physical, digest=None)
        obj.state = ObjectState.CLEAN
        obj.trust = Trust.ASSERTED
        obj.evidence = "kernel WRITE with declared synchronisation; trust ASSERTED not digested"
        return obj
    # READ + sync: prior knowledge stands. UNKNOWN stays UNKNOWN.
    if obj.state is ObjectState.UNKNOWN:
        obj.evidence = "kernel READ with sync; prior state was UNKNOWN and stays UNKNOWN"
        return obj
    obj.evidence = "kernel READ with declared synchronisation; prior state preserved"
    return obj


def apply_kernel_boundary_to_humf(
    humf_obj: Any,
    domain: str,
    *,
    synchronized: bool,
) -> None:
    """Drive HUMF's existing CLEAN->UNKNOWN edge from a kernel boundary.

    HUMF already lists UNKNOWN as a legal successor of CLEAN (G053) but
    nothing in the fabric treats kernel dispatch as that edge. This is
    the overlay's extension point: it mutates a live HumfObject without
    forking the state machine.
    """
    m = humf_obj.materializations.get(domain)
    if m is None:
        raise ManagedObjectError(
            f"{getattr(humf_obj, 'identity', '<unknown>')} has no materialization in {domain}"
        )
    if synchronized:
        return
    state = m.state
    name = getattr(state, "value", str(state))
    # HUMF LEGAL allows CLEAN/DIRTY/STALE -> UNKNOWN. Other copy-states
    # (MATERIALIZING, TRANSFERRING, ...) do not; we do not invent a path.
    if name in {"CLEAN", "DIRTY", "STALE"}:
        m.transition(type(state)("UNKNOWN"))
    m.trust = "UNKNOWN"


# ---------------------------------------------------------------------------
# Move vs recompute -- never guess
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoveRecomputeDecision:
    transfer_cost: Cost
    recompute_cost: Cost
    decision: Decision
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "transfer_cost": self.transfer_cost,
            "recompute_cost": self.recompute_cost,
            "decision": self.decision.value,
            "reason": self.reason,
        }


def _cost_kind(cost: Cost | None) -> str:
    if cost is None:
        return "UNKNOWN"
    if cost in ("UNKNOWN", "UNAVAILABLE"):
        return cost
    if isinstance(cost, (int, float)):
        if cost < 0:
            raise ManagedObjectError(f"cost must be >= 0, got {cost!r}")
        return "NUMBER"
    raise ManagedObjectError(f"unrecognised cost {cost!r}")


def move_vs_recompute(
    transfer_cost: Cost | None,
    recompute_cost: Cost | None,
) -> MoveRecomputeDecision:
    """Return (transfer_cost, recompute_cost) plus a decision.

    Either cost may be UNKNOWN. When either *existing* option has unknown
    cost the decision is UNDECIDABLE -- the layer never guesses that
    moving is cheaper. UNAVAILABLE means the option does not exist (no
    path, no recipe) and is not the same as UNKNOWN.

    Recompute beating transfer is a first-class outcome, not a footnote.
    """
    t_kind = _cost_kind(transfer_cost)
    r_kind = _cost_kind(recompute_cost)
    t_val: Cost = "UNKNOWN" if transfer_cost is None else transfer_cost
    r_val: Cost = "UNKNOWN" if recompute_cost is None else recompute_cost

    if t_kind == "UNAVAILABLE" and r_kind == "UNAVAILABLE":
        return MoveRecomputeDecision(
            t_val, r_val, Decision.NEITHER,
            "no transfer path and no recompute recipe",
        )
    if t_kind == "UNKNOWN" or r_kind == "UNKNOWN":
        return MoveRecomputeDecision(
            t_val, r_val, Decision.UNDECIDABLE,
            "at least one existing option has UNKNOWN cost; refusing to guess "
            "that moving is cheaper (or that recomputing is)",
        )
    if t_kind == "UNAVAILABLE" and r_kind == "NUMBER":
        return MoveRecomputeDecision(
            t_val, r_val, Decision.RECOMPUTE,
            "no transfer path; recompute is the only option",
        )
    if r_kind == "UNAVAILABLE" and t_kind == "NUMBER":
        return MoveRecomputeDecision(
            t_val, r_val, Decision.MOVE,
            "no recompute recipe; transfer is the only option",
        )
    assert t_kind == "NUMBER" and r_kind == "NUMBER"
    assert isinstance(t_val, (int, float)) and isinstance(r_val, (int, float))
    if r_val < t_val:
        return MoveRecomputeDecision(
            t_val, r_val, Decision.RECOMPUTE,
            "recompute is cheaper than transfer; this is a real possibility, not a fallback",
        )
    if t_val < r_val:
        return MoveRecomputeDecision(
            t_val, r_val, Decision.MOVE,
            "transfer is cheaper than recompute",
        )
    return MoveRecomputeDecision(
        t_val, r_val, Decision.TIE,
        "costs are equal; caller may pick either, overlay does not",
    )


def move_vs_recompute_from_humf(
    humf_obj: Any,
    transfer_cost: Cost | None,
) -> MoveRecomputeDecision:
    """Close the HUMF planner gap: a recipe with no measured cost is UNKNOWN.

    HUMF's ``plan_acquire`` only adds RECOMPUTE when ``recompute_cost_s``
    is not None. A live ``recompute`` callable with a missing cost is
    silently dropped, so TRANSFER wins -- which is guessing that moving
    is cheaper than an unmeasured recompute. This adapter refuses that.
    """
    recipe = getattr(humf_obj, "recompute", None) is not None
    cost_s = getattr(humf_obj, "recompute_cost_s", None)
    if cost_s is None and recipe:
        recompute_cost: Cost = "UNKNOWN"
    elif cost_s is None and not recipe:
        recompute_cost = "UNAVAILABLE"
    else:
        recompute_cost = float(cost_s)
    return move_vs_recompute(transfer_cost, recompute_cost)


# ---------------------------------------------------------------------------
# HUMF adapter -- extend, do not fork
# ---------------------------------------------------------------------------

def _load_humf() -> Any | None:
    acc = REPO / "tools" / "accelerator"
    if not (acc / "humf.py").is_file():
        return None
    path = str(acc)
    if path not in _sys.path:
        _sys.path.insert(0, path)
    import humf  # type: ignore  # noqa: E402
    return humf


def project_humf_state(name: str) -> ObjectState:
    try:
        return HUMF_STATE_PROJECTION[name]
    except KeyError as exc:
        raise ManagedObjectError(f"unmapped HUMF copy-state {name!r}") from exc


def from_humf_object(
    humf_obj: Any,
    *,
    prefer_domain: str | None = None,
) -> ManagedObject:
    """Project a live HumfObject onto the overlay without copying the fabric."""
    identity = str(humf_obj.identity)
    mats: Mapping[str, Any] = humf_obj.materializations
    domain = prefer_domain
    if domain is None:
        owner = getattr(humf_obj, "owner", None)
        if owner in mats:
            domain = owner
        else:
            clean = sorted(d for d, m in mats.items() if getattr(m.state, "value", None) == "CLEAN")
            domain = clean[0] if clean else (sorted(mats)[0] if mats else None)
    if domain is None:
        return ManagedObject(
            identity=HgvasRef(identity),
            semantic_representation=str(getattr(humf_obj, "logical_type", "unspecified")),
            humf_identity=identity,
            recompute_recipe=getattr(humf_obj, "recompute", None) is not None,
            recompute_cost=(
                float(humf_obj.recompute_cost_s)
                if getattr(humf_obj, "recompute_cost_s", None) is not None
                else "UNKNOWN"
            ),
        )
    m = mats[domain]
    state_name = getattr(m.state, "value", str(m.state))
    trust_name = getattr(m, "trust", "UNKNOWN")
    try:
        trust = Trust(trust_name)
    except ValueError:
        trust = Trust.UNKNOWN
    tier, device = HUMF_DOMAIN_LOCATION.get(domain, (MemoryTier.UNKNOWN, domain))
    vis: Visibility
    if tier is MemoryTier.UNKNOWN:
        vis = "UNKNOWN"
    else:
        vis = frozenset({device})
    logical = str(getattr(humf_obj, "logical_type", "unspecified"))
    nbytes = getattr(m, "bytes", None)
    digest = getattr(m, "digest", None)
    if isinstance(digest, int):
        digest = f"{digest:08x}"
    content = getattr(humf_obj, "content_digest", None)
    sealed = content if isinstance(content, str) else digest
    return ManagedObject(
        identity=HgvasRef(identity),
        owner=getattr(humf_obj, "owner", None),
        semantic_representation=logical,
        physical=PhysicalMaterialization(
            representation=str(getattr(m, "representation", "unspecified")),
            layout=str(getattr(m, "layout", "unspecified")),
            nbytes=int(nbytes) if nbytes is not None else None,
            digest=sealed,
        ),
        location=Location(tier, device),
        device_visibility=vis,
        version=int(getattr(m, "verified_at", 0) or 0),
        trust=trust,
        state=project_humf_state(state_name),
        evidence=f"projected from HumfObject {identity} domain {domain}",
        payload=getattr(m, "payload", None),
        humf_identity=identity,
        recompute_recipe=getattr(humf_obj, "recompute", None) is not None,
        recompute_cost=(
            float(humf_obj.recompute_cost_s)
            if getattr(humf_obj, "recompute_cost_s", None) is not None
            else ("UNKNOWN" if getattr(humf_obj, "recompute", None) is not None else "UNAVAILABLE")
        ),
    )


def recover_humf_surface() -> dict[str, Any]:
    """Disk-backed description of what already exists. Numbers are names, not measurements."""
    humf = _load_humf()
    acc = REPO / "tools" / "accelerator"
    out: dict[str, Any] = {
        "humf_path": "tools/accelerator/humf.py",
        "hmf_path": "tools/accelerator/hmf.py",
        "humf_present": (acc / "humf.py").is_file(),
        "hmf_present": (acc / "hmf.py").is_file(),
        "imported": humf is not None,
    }
    if humf is None:
        out["import_error"] = "tools/accelerator/humf.py not importable in this worktree"
        return out
    try:
        _sys.path.insert(0, str(acc))
        import hmf as hmf_mod  # type: ignore  # noqa: E402
        out["hmf_is_humf"] = hmf_mod.Humf is humf.Humf
        out["hmf_defines"] = [
            "Accelerator", "AcceleratorDomain", "DeviceIdentity",
            "MachineIdentity", "build_hawkgpu0",
        ]
    except Exception as exc:  # noqa: BLE001
        out["hmf_is_humf"] = None
        out["hmf_import_error"] = type(exc).__name__
    out["humf_states"] = [s.value for s in humf.State]
    out["humf_object_fields"] = list(humf.HumfObject.__dataclass_fields__)
    out["materialization_fields"] = list(humf.Materialization.__dataclass_fields__)
    out["has_is_coherent"] = hasattr(humf, "is_coherent") or hasattr(humf.HumfObject, "is_coherent")
    out["has_kernel_boundary"] = any("kernel" in n.lower() for n in dir(humf))
    out["has_migrate"] = any("migrat" in n.lower() for n in dir(humf))
    out["has_require_coherent"] = hasattr(humf, "require_coherent")
    out["plan_picks_cheapest_known_number"] = True
    out["valid_copies_names_clean_regardless_of_trust"] = True
    out["trusted_copies_excludes_trust_unknown"] = True
    out["trusted_copies_includes_asserted"] = True
    return out


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------

def _selftest_cases() -> dict[str, str]:
    """Run the invariant on in-memory objects. Returns PASS/FAIL per case.

    Failures raise -- a receipt that records FAIL for the load-bearing
    invariant is not a receipt, it is a confession that we sealed anyway.
    """
    results: dict[str, str] = {}

    def record(name: str, ok: bool, detail: str = "") -> None:
        if not ok:
            raise ManagedObjectError(f"selftest {name} failed{': ' + detail if detail else ''}")
        results[name] = "PASS" + (f" ({detail})" if detail else "")

    obj = ManagedObject(identity=HgvasRef("weights.layer0"))
    record("new_object_defaults_unknown", obj.state is ObjectState.UNKNOWN)
    record(
        "new_object_is_not_coherent_claim",
        is_coherent(obj) is Coherence.UNKNOWN,
    )
    refused = False
    try:
        require_coherent(obj, reader="selftest-new")
    except CoherenceRequiredError:
        refused = True
    record("require_coherent_refuses_unknown_default", refused)

    establish_clean(
        obj,
        location=Location(MemoryTier.UMA, "APPLE_DOMAIN_0"),
        visibility={"APPLE_DOMAIN_0"},
        evidence="fixture placement",
        digest="d" * 32,
        payload=b"abcd",
    )
    record("establish_clean_is_coherent", is_coherent(obj) is Coherence.COHERENT)
    consume_for_kernel(obj, kernel="selftest_gemm", device="APPLE_DOMAIN_0")
    record("consume_for_kernel_accepts_coherent", True)

    bool_refused = False
    try:
        bool(is_coherent(obj))
    except CoherenceBooleanError:
        bool_refused = True
    record("boolean_collapse_refused_on_coherent", bool_refused)

    clean = ManagedObject(identity=HgvasRef("kv.seq0"))
    establish_clean(
        clean,
        location=Location(MemoryTier.UMA, "APPLE_DOMAIN_0"),
        visibility={"APPLE_DOMAIN_0"},
        evidence="fixture",
        digest="e" * 32,
        payload=b"kv",
    )
    cross_kernel_boundary(clean, synchronized=False, access=KernelAccess.READ)
    record(
        "kernel_boundary_without_sync_moves_clean_to_unknown",
        clean.state is ObjectState.UNKNOWN,
        f"state={clean.state.value}",
    )
    record(
        "kernel_boundary_without_sync_is_unknown_coherence",
        is_coherent(clean) is Coherence.UNKNOWN,
    )
    k_refused = False
    try:
        consume_for_kernel(clean, kernel="decode", device="APPLE_DOMAIN_0")
    except CoherenceRequiredError:
        k_refused = True
    record("kernel_consume_refuses_unknown_after_unsynced_boundary", k_refused)

    synced = ManagedObject(identity=HgvasRef("act.block1"))
    establish_clean(
        synced,
        location=Location(MemoryTier.UMA, "APPLE_DOMAIN_0"),
        visibility={"APPLE_DOMAIN_0"},
        evidence="fixture",
        digest="f" * 32,
    )
    cross_kernel_boundary(synced, synchronized=True, access=KernelAccess.READ)
    record(
        "kernel_boundary_with_sync_read_stays_clean",
        synced.state is ObjectState.CLEAN,
    )

    d1 = move_vs_recompute("UNKNOWN", 1.0)
    record("unknown_transfer_is_undecidable", d1.decision is Decision.UNDECIDABLE)
    d2 = move_vs_recompute(1.0, "UNKNOWN")
    record("unknown_recompute_is_undecidable", d2.decision is Decision.UNDECIDABLE)
    d3 = move_vs_recompute(5.0, 1.0)
    record("recompute_can_beat_transfer", d3.decision is Decision.RECOMPUTE)
    d4 = move_vs_recompute(1.0, 5.0)
    record("move_wins_when_both_known_and_cheaper", d4.decision is Decision.MOVE)
    d5 = move_vs_recompute("UNAVAILABLE", "UNAVAILABLE")
    record("neither_when_both_unavailable", d5.decision is Decision.NEITHER)

    native_refused = False
    try:
        HgvasRef("0xabc123def")
    except HgvasError:
        native_refused = True
    record("hgvas_refuses_native_pointer", native_refused)

    digest = device_digest(obj, provider=None)
    record("device_digest_without_provider_is_unknown", digest.status == "UNKNOWN")

    moved = ManagedObject(identity=HgvasRef("exp.8"))
    establish_clean(
        moved,
        location=Location(MemoryTier.UMA, "APPLE_DOMAIN_0"),
        visibility={"APPLE_DOMAIN_0"},
        evidence="fixture",
        digest="a" * 32,
    )
    migrate(moved, Location(MemoryTier.HBM, "FPGA_HBM_0"))
    record(
        "unconfirmed_migrate_is_unknown_not_clean",
        moved.state is ObjectState.UNKNOWN and is_coherent(moved) is Coherence.UNKNOWN,
    )

    return results


def build() -> Path:
    recovered = recover_humf_surface()
    selftest = _selftest_cases()

    # hcli/physical_graph.py and hcli/machine.py are not always
    # materialised in a sparse checkout; git-show them conceptually via
    # recovered notes, never claim they were imported if they were not.
    physical_graph = REPO / "hcli" / "physical_graph.py"
    machine = REPO / "hcli" / "machine.py"
    genome = REPO / "tools" / "accelerator" / "machine_genome.py"

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Object-level HMF/HGVAS managed-object semantics that never "
            "report COHERENT when the layer does not know that it is."
        ),
        "invariant": (
            "is_coherent returns COHERENT | NOT_COHERENT | UNKNOWN. "
            "UNKNOWN is never readable as coherent. Boolean collapse is refused. "
            "A kernel boundary without synchronisation moves CLEAN to UNKNOWN."
        ),
        "object_identity_fields": [
            "identity (HgvasRef object_id + byte_offset)",
            "owner",
            "semantic_representation",
            "physical_materialization (representation, layout, nbytes, digest)",
            "location (memory_tier + device)",
            "device_visibility",
            "version",
            "trust (TRUSTED | ASSERTED | UNKNOWN)",
            "state (CLEAN | DIRTY | STALE | UNKNOWN)",
        ],
        "object_states": [s.value for s in ObjectState],
        "coherence_values": [c.value for c in Coherence],
        "trust_values": [t.value for t in Trust],
        "memory_tiers": [t.value for t in MemoryTier],
        "humf_state_projection": {k: v.value for k, v in sorted(HUMF_STATE_PROJECTION.items())},
        "object_legal_transitions": {
            k.value: sorted(x.value for x in vs) for k, vs in OBJECT_LEGAL.items()
        },
        "coherence_rules": {
            "COHERENT": (
                "state CLEAN AND trust TRUSTED AND visibility known AND "
                "location known AND (if on_device set, device in visibility)"
            ),
            "NOT_COHERENT": (
                "state DIRTY or STALE, or a known visibility set that does "
                "not include the calling device"
            ),
            "UNKNOWN": (
                "state UNKNOWN, or trust UNKNOWN/ASSERTED, or visibility "
                "UNKNOWN, or location UNKNOWN. ASSERTED is not knowledge."
            ),
            "boolean_collapse": "refused (CoherenceBooleanError)",
            "consume_paths": [
                "require_coherent",
                "consume_for_kernel",
                "read_payload",
            ],
        },
        "operations": {
            "establish_clean": "only path that earns CLEAN; requires evidence and digest",
            "migrate": (
                "updates claimed location; without a matching digest the "
                "object becomes UNKNOWN, never CLEAN"
            ),
            "invalidate": "explicit STALE -- knowledge of NOT_COHERENT",
            "device_digest": (
                "VERIFIED / UNKNOWN / MISMATCH. No provider => UNKNOWN. "
                "Mismatch => STALE. Provider error => UNKNOWN."
            ),
            "cross_kernel_boundary": (
                "synchronized=False => UNKNOWN from any state including CLEAN. "
                "synchronized READ preserves prior state. synchronized WRITE "
                "commits CLEAN with trust ASSERTED (fence is not a digest). "
                "synchronized UNKNOWN access => UNKNOWN."
            ),
            "mark_written": "DIRTY, version++, digest unsealed",
            "apply_kernel_boundary_to_humf": (
                "drives HUMF Materialization.transition(UNKNOWN) on the "
                "existing CLEAN->UNKNOWN edge; does not fork the fabric"
            ),
        },
        "move_vs_recompute": {
            "returns": ["transfer_cost", "recompute_cost", "decision", "reason"],
            "decisions": [d.value for d in Decision],
            "rule": (
                "If either existing option has cost UNKNOWN, decision is "
                "UNDECIDABLE. UNAVAILABLE (no path / no recipe) is not UNKNOWN. "
                "Recompute beating transfer is a first-class outcome."
            ),
            "humf_gap": (
                "Humf.plan_acquire omits RECOMPUTE when recompute_cost_s is "
                "None even if a recompute callable exists, so TRANSFER wins. "
                "move_vs_recompute_from_humf treats that as UNKNOWN and returns "
                "UNDECIDABLE."
            ),
        },
        "hgvas": {
            "status": "semantic identity only",
            "fusion_ptr": "{object_id, byte_offset}",
            "native_pointers": "refused",
            "not_built": [
                "page directory",
                "bit-level dirty maps",
                "Fusion ISA",
                "logical command timeline",
                "wire format / golden vectors",
                "MockSparkProvider",
            ],
            "note": (
                "G054 names those. Building them here would be a competing "
                "Fusion implementation; this overlay only pins the identity "
                "and coherence contract HGVAS will have to honour."
            ),
        },
        "recovered_implementation": {
            "hmf_humf": recovered,
            "hcli_physical_graph": {
                "path": "hcli/physical_graph.py",
                "materialised": physical_graph.is_file(),
                "role": (
                    "PLAN_ONLY placement/dataflow graph; synchronization is "
                    "runtime_boundary/unresolved. No managed-object coherence."
                ),
            },
            "hcli_machine": {
                "path": "hcli/machine.py",
                "materialised": machine.is_file(),
                "role": (
                    "Host memory admission (vm_stat, Metal working-set). "
                    "Not an object fabric."
                ),
            },
            "machine_genome": {
                "path": "tools/accelerator/machine_genome.py",
                "materialised": genome.is_file(),
                "role": (
                    "SoC identity and contended-bandwidth measurement. "
                    "No object-level coherence API."
                ),
            },
            "air_memory_domains": [
                "APPLE_UM", "MOCK_EXTERNAL_VRAM", "NVIDIA_VRAM_SIDECAR",
                "NVIDIA_VRAM_DIRECT", "SSD_COLD", "HOST_LOGICAL", "ANY",
            ],
            "air_sync_scopes": ["THREADGROUP", "SIMDGROUP", "DEVICE"],
            "note": (
                "HUMF/HMF already implement the copy fabric (twelve recorded "
                "fields, fail-closed transitions, trust decay, resident digest, "
                "HAWKGPU-0). They do not implement tri-state is_coherent, "
                "kernel-boundary UNKNOWN, UNDECIDABLE costs, device visibility, "
                "or HGVAS identity-without-native-pointers. This module closes "
                "those gaps as an overlay rather than a rival fabric."
            ),
        },
        "gaps_closed": [
            "tri-state is_coherent; boolean collapse refused",
            "require_coherent / consume_for_kernel / read_payload refuse UNKNOWN",
            "CLEAN + trust UNKNOWN projects to UNKNOWN coherence (HUMF valid_copies still names it)",
            "ASSERTED trust is UNKNOWN coherence, not COHERENT",
            "kernel boundary without sync: CLEAN -> UNKNOWN",
            "apply_kernel_boundary_to_humf drives the existing HUMF CLEAN->UNKNOWN edge",
            "move_vs_recompute UNDECIDABLE when either cost is UNKNOWN",
            "move_vs_recompute_from_humf treats a recipe with no cost as UNKNOWN, not as 'skip recompute'",
            "migrate without confirming digest is UNKNOWN, never CLEAN",
            "device_digest with no provider is UNKNOWN, never a fabricated hash",
            "HGVAS HgvasRef refuses native pointers",
            "location is memory_tier + device; visibility is a set or UNKNOWN",
            "version increments on write / kernel WRITE, not on invalidate",
        ],
        "negative_findings": [
            "CLAUDE_GLOBAL_FRONTIER.json has no dedicated HMF-objects lane entry (F001-F015)",
            "no HGVAS page directory, Fusion ISA, or FusionPtr type exists on disk (G054 still open)",
            "AIR represents THREADGROUP/SIMDGROUP/DEVICE barriers but DEVICE cannot lower; this overlay does not lower either",
            "HUMF Ownership axis is not wired into plan_acquire/execute (named in hmf.py, left as future work there and here)",
            "no hardware digest, no GPU, no second machine, no HBM device -- every physical cost is UNKNOWN",
            "hcli/physical_graph.py and hcli/machine.py are in git but not always materialised in this sparse checkout",
            "sidecar must not call Humf.execute on a real provider; overlay never moves bytes",
        ],
        "selftest": selftest,
        "integration": {
            "is_coherent": "is_coherent(obj: ManagedObject, *, on_device: str | None = None) -> Coherence",
            "require_coherent": "require_coherent(obj, *, reader: str, on_device: str | None = None) -> None",
            "consume_for_kernel": "consume_for_kernel(obj, *, kernel: str, device: str) -> KernelBinding",
            "cross_kernel_boundary": "cross_kernel_boundary(obj, *, synchronized: bool, access: KernelAccess = UNKNOWN) -> ManagedObject",
            "move_vs_recompute": "move_vs_recompute(transfer_cost, recompute_cost) -> MoveRecomputeDecision",
            "from_humf_object": "from_humf_object(humf_obj, *, prefer_domain: str | None = None) -> ManagedObject",
            "apply_kernel_boundary_to_humf": "apply_kernel_boundary_to_humf(humf_obj, domain: str, *, synchronized: bool) -> None",
        },
        "claim_class": "STATIC_ONLY",
        "gpu_authority": False,
    }
    return write_receipt(RECEIPT, doc, "tools/future/hmf_objects.py")


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
