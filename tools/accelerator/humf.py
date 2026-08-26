"""HUMF — the Hawking Unified Memory Fabric. FRONT H (G050, steer S015).

HUMF DOES NOT LIE. Apple unified memory plus external VRAM is never called
physically unified memory. The physical domains stay distinct and HUMF manages the
illusion at the object level: who owns the current state, which copies are valid,
what a move would cost, and whether recomputing beats transferring.

Every number a MOCK domain produces is stamped SIMULATED. The steer is explicit
that a simulated transport number must never be labelled physical evidence, so the
planner carries the provenance of each cost it used into its decision, and a plan
built on simulated numbers says so in its own output rather than in a footnote.

G053 CANONICALIZATION: `hmf` (tools/accelerator/hmf.py) is now the canonical name
for this module; this file is unchanged in behaviour and remains fully importable
under its own name so no existing receipt, test or caller breaks. hmf.py holds the
genuinely new HAWKGPU-0 device-abstraction layer (Accelerator, per-domain
DeviceIdentity/MachineIdentity, topology) on top of what is defined here: the
State/Ownership/MemoryClass vocabulary additions and the Materialization/HumfObject
fields they need live in THIS file because Materialization and HumfObject are
defined here, and hmf.py re-exports them rather than re-declaring them.
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class State(str, Enum):
    """Steer §14. No accidental hidden coherence assumptions.

    G053 CANONICALIZATION: the canonical copy-state vocabulary names nine states
    -- ABSENT, MATERIALIZING, CLEAN, DIRTY, STALE, UNKNOWN, CORRUPT, LOST, EVICTED.
    Six were already here. UNKNOWN, CORRUPT and LOST are genuinely new members,
    added rather than substituted for the two members the canonical list does not
    mention -- TRANSFERRING and INVALID stay, because every existing transition,
    receipt and test already depends on them and nothing in G053 asks for their
    removal. Read together the set is a superset of the canonical nine, not a
    like-for-like rename.

    UNKNOWN as a STATE is a different axis from Materialization.trust == "UNKNOWN":
    trust says a CLEAN copy's bytes are not vouched for (device_partially_lost's
    probe timeout leaves state alone on purpose -- see that function's docstring);
    State.UNKNOWN says the copy's COHERENCE STATE ITSELF could not be established.
    Existing code paths (device_partially_lost, resolve_unknown, audit) are left
    untouched and keep using trust + INVALID exactly as before; State.UNKNOWN,
    State.CORRUPT and State.LOST are wired into LEGAL below so new callers have a
    real, fail-closed state machine to use rather than a decorative name that is
    never reachable."""
    ABSENT = "ABSENT"
    CLEAN = "CLEAN"
    DIRTY = "DIRTY"
    STALE = "STALE"
    MATERIALIZING = "MATERIALIZING"
    TRANSFERRING = "TRANSFERRING"
    INVALID = "INVALID"
    EVICTED = "EVICTED"
    UNKNOWN = "UNKNOWN"
    CORRUPT = "CORRUPT"
    LOST = "LOST"


# Illegal transitions fail closed (the steer says so of the driver state machine;
# the same discipline belongs here, where a silent bad transition means silent
# data corruption rather than a crash).
#
# UNKNOWN/CORRUPT/LOST are additive: every pre-existing edge is unchanged, so no
# transition legal before G053 became illegal. CORRUPT recovers exactly like
# INVALID (ABSENT or straight to MATERIALIZING) because both mean "known bad,
# retry is fine." LOST is deliberately STRICTER than INVALID -- only -> ABSENT --
# because LOST means the bytes are gone with no retry path; a caller must clear
# the slot before it may be reused, it cannot jump straight back into flight.
LEGAL: dict[State, set[State]] = {
    State.ABSENT:        {State.MATERIALIZING, State.TRANSFERRING},
    State.MATERIALIZING: {State.CLEAN, State.INVALID, State.CORRUPT, State.LOST},
    State.TRANSFERRING:  {State.CLEAN, State.INVALID, State.CORRUPT, State.LOST},
    State.CLEAN:         {State.DIRTY, State.STALE, State.EVICTED, State.INVALID,
                          State.TRANSFERRING, State.UNKNOWN, State.CORRUPT,
                          State.LOST},
    State.DIRTY:         {State.CLEAN, State.STALE, State.INVALID, State.UNKNOWN,
                          State.CORRUPT, State.LOST},
    State.STALE:         {State.TRANSFERRING, State.MATERIALIZING, State.EVICTED,
                          State.INVALID, State.UNKNOWN, State.CORRUPT, State.LOST},
    State.EVICTED:       {State.MATERIALIZING, State.TRANSFERRING, State.ABSENT},
    State.INVALID:       {State.ABSENT, State.MATERIALIZING},
    State.UNKNOWN:       {State.CLEAN, State.STALE, State.CORRUPT, State.LOST,
                          State.ABSENT},
    State.CORRUPT:       {State.ABSENT, State.MATERIALIZING},
    State.LOST:          {State.ABSENT},
}


class Ownership(str, Enum):
    """G053: the SECOND, SEPARATE axis. A copy's Ownership says which domain's
    write is authoritative over it right now (or that it is shared-read, or mid
    handoff) -- it says NOTHING about the copy's coherence (State). A CLEAN copy
    can be APPLE_BIAS or SHARED_READ; a DIRTY copy still has an owner. Collapsing
    the two axes into one would make a write-ownership question and a coherence
    question the same field, which is exactly the bug class State vs trust above
    already exists to avoid repeating.

    Five members, matching the canonical set: APPLE_BIAS and SPARK0_BIAS/
    SPARK1_BIAS name which domain's write is authoritative; SHARED_READ means no
    domain holds exclusive write ownership; TRANSIT is the handoff state a change
    of owner must pass through, the ownership equivalent of TRANSFERRING."""
    APPLE_BIAS = "APPLE_BIAS"
    SPARK0_BIAS = "SPARK0_BIAS"
    SPARK1_BIAS = "SPARK1_BIAS"
    SHARED_READ = "SHARED_READ"
    TRANSIT = "TRANSIT"


# Fail closed exactly like LEGAL above. An exclusive bias may only become shared
# or hand off through TRANSIT -- never jump straight to a different exclusive
# bias -- and TRANSIT is the only state that may resolve into any of the others,
# mirroring how a real handoff has no direct bias-to-bias edge.
OWNERSHIP_LEGAL: dict[Ownership, set[Ownership]] = {
    Ownership.APPLE_BIAS:  {Ownership.TRANSIT, Ownership.SHARED_READ},
    Ownership.SPARK0_BIAS: {Ownership.TRANSIT, Ownership.SHARED_READ},
    Ownership.SPARK1_BIAS: {Ownership.TRANSIT, Ownership.SHARED_READ},
    Ownership.SHARED_READ: {Ownership.TRANSIT},
    Ownership.TRANSIT:     {Ownership.APPLE_BIAS, Ownership.SPARK0_BIAS,
                            Ownership.SPARK1_BIAS, Ownership.SHARED_READ},
}


class MemoryClass(str, Enum):
    """G053: what KIND of object this is, which is what actually decides its
    residency/consistency policy -- see MEMORY_CLASS_POLICY. There is
    deliberately no universal page policy; a class that is never mutated
    (IMMUTABLE_WEIGHTS, COMPILER_ARTIFACT, EXPERT_CACHE) may sit SHARED_READ
    across every domain it has ever been placed in and stay CLEAN there
    indefinitely, while a class that is mutated every step (KV_STATE,
    RECURRENT_STATE) goes through ordinary write/stale bookkeeping exactly as
    today. `memory_class` is optional on HumfObject (default None) so every
    existing caller that never classified an object keeps working unchanged."""
    IMMUTABLE_WEIGHTS = "IMMUTABLE_WEIGHTS"
    KV_STATE = "KV_STATE"
    RECURRENT_STATE = "RECURRENT_STATE"
    ACTIVATIONS = "ACTIVATIONS"
    ROUTING = "ROUTING"
    METADATA = "METADATA"
    SCRATCH = "SCRATCH"
    COMPILER_ARTIFACT = "COMPILER_ARTIFACT"
    EXPERT_CACHE = "EXPERT_CACHE"


# THE LOAD-BEARING FIELD IS `mutable`. False means mark_written() refuses --
# see HumfObject.mark_written -- which is what makes "IMMUTABLE_WEIGHTS can go
# SHARED_READ after placement and stay valid" an enforced invariant rather than
# a comment: nothing can silently write through an immutable object and nothing
# needs to re-stream it merely because execution crossed a domain, since a read
# in a foreign domain was never staling anything to begin with (staleness has
# always come from mark_written, never from mere cross-domain access -- true
# before G053 too, and now it is also POLICED for the classes that must never
# see it fire).
MEMORY_CLASS_POLICY: dict[MemoryClass, dict[str, Any]] = {
    MemoryClass.IMMUTABLE_WEIGHTS: {
        "mutable": False, "shared_read_stable": True,
        "note": "written once at load, then read-only; SHARED_READ across "
                "domains never forces a re-stream because the value cannot "
                "change under it"},
    MemoryClass.COMPILER_ARTIFACT: {
        "mutable": False, "shared_read_stable": True,
        "note": "a compiled kernel/binary is immutable once produced -- same "
                "SHARED_READ-stays-valid treatment as IMMUTABLE_WEIGHTS"},
    MemoryClass.EXPERT_CACHE: {
        "mutable": False, "shared_read_stable": True,
        "note": "a resident expert's weights are immutable once cached; "
                "eviction is a capacity decision, not a correctness one"},
    MemoryClass.KV_STATE: {
        "mutable": True, "shared_read_stable": False,
        "note": "grows/overwrites every decode step; a write anywhere stales "
                "every other copy exactly like any DIRTY object"},
    MemoryClass.RECURRENT_STATE: {
        "mutable": True, "shared_read_stable": False,
        "note": "one authoritative copy per sequence, overwritten every step"},
    MemoryClass.ACTIVATIONS: {
        "mutable": True, "shared_read_stable": False,
        "note": "produced and consumed within one pass; ordinarily short-lived, "
                "no special cross-domain treatment"},
    MemoryClass.ROUTING: {
        "mutable": True, "shared_read_stable": False,
        "note": "recomputed per token/batch; treated as ordinary mutable state"},
    MemoryClass.METADATA: {
        "mutable": True, "shared_read_stable": False,
        "note": "small bookkeeping, mutable, no special policy beyond ordinary "
                "state"},
    MemoryClass.SCRATCH: {
        "mutable": True, "shared_read_stable": False,
        "note": "transient workspace; contents are not expected to outlive one "
                "op"},
}


class DeviceLost(RuntimeError):
    """The domain itself went away mid-transfer -- a pulled cable, a bus reset, a
    driver crash. Distinct from HumfError on purpose: a failed COPY damages ONE
    object, while a lost DEVICE invalidates EVERY copy that domain was holding, and
    treating the second as the first leaves valid_copies() naming copies that no
    longer exist anywhere."""


class TransferTimeout(RuntimeError):
    """The transport did not answer inside its deadline. Detected, not abandoned --
    see Humf._move for exactly what that distinction costs."""


class HumfError(RuntimeError):
    pass


@dataclass
class Domain:
    """A memory domain. `physical` is False for anything mocked, and that flag is
    what stops a simulated bandwidth from being quoted as evidence."""
    name: str
    bytes_capacity: int
    bandwidth_gb_s: float
    physical: bool
    latency_s: float = 0.0

    @property
    def provenance(self) -> str:
        return "MEASURED" if self.physical else "SIMULATED"


@dataclass
class Materialization:
    """One representation of one object in one domain."""
    domain: str
    representation: str
    layout: str
    bytes: int
    state: State = State.ABSENT
    payload: bytes | None = None      # the actual data, when this copy holds any
    digest: int | None = None         # integrity of that data, when it was verified
    # TRUST IS PER COPY, NOT ONLY PER DOMAIN. A quarantine says the LINK is not to
    # be relied upon; it says nothing about whether any individual copy on it is
    # good, and releasing the quarantine used to return both to service together.
    # TRUSTED = usable. UNKNOWN = a probe could not answer for it, so it is out of
    # service until resolved. ASSERTED = an operator said it is fine WITHOUT a
    # check, which is a weaker thing than verified and is recorded as a weaker thing.
    trust: str = "TRUSTED"
    # WHEN that trust was last established, as a fabric event counter. Trust used to
    # be a fact with no age: a copy verified once was TRUSTED forever and nothing
    # ever re-asked. A verification is a statement about A MOMENT, and a copy that
    # has sat resident across a thousand later events is not covered by it.
    verified_at: int = 0
    # Set by the RESIDENT check. `resident_verified` False means the provider offers
    # no device-side digest, NOT that the copy is bad; `resident_digest_path` is the
    # provider's OWN CLAIM about which path it digested and the fabric cannot check it.
    resident_verified: bool = False
    resident_digest_path: str | None = None
    # G053: OWNERSHIP IS A SEPARATE AXIS FROM STATE -- see the Ownership class
    # docstring. Defaults to APPLE_BIAS because today there is exactly one real
    # domain (HAWKGPU-0's APPLE_DOMAIN_0) and it is the only writer there is.
    ownership: Ownership = Ownership.APPLE_BIAS

    def transition(self, to: State) -> None:
        if to not in LEGAL[self.state]:
            raise HumfError(f"illegal transition {self.state.value} -> {to.value} "
                            f"for {self.representation} in {self.domain}")
        self.state = to

    def transition_ownership(self, to: Ownership) -> None:
        """The Ownership-axis twin of transition(). Deliberately a SEPARATE
        method over a SEPARATE table (OWNERSHIP_LEGAL) touching a SEPARATE field
        -- calling this never reads or writes self.state, and transition() never
        reads or writes self.ownership. That separation is the thing G053 asks
        to be pinned by a test, not merely documented."""
        if to not in OWNERSHIP_LEGAL[self.ownership]:
            raise HumfError(
                f"illegal ownership transition {self.ownership.value} -> "
                f"{to.value} for {self.representation} in {self.domain}")
        self.ownership = to


@dataclass
class HumfObject:
    """The twelve recorded fields of steer §32, as one object across domains."""
    identity: str                                   # 1 ObjectIdentity
    logical_type: str                               # 2 logical type
    elements: int
    dtype: str
    materializations: dict[str, Materialization] = field(default_factory=dict)  # 3,7,8,9
    owner: str | None = None                        # 4 current owner of live state
    recompute_cost_s: float | None = None           # 11
    next_use_hint: str | None = None                # 12
    recompute: Callable[[], Any] | None = None
    # THE OBJECT'S IDENTITY, computed by the FABRIC at registration and never by a
    # transport. The per-transfer check compares the source against the DESTINATION,
    # so a source that rotted in place verifies rot against rot and passes -- and
    # afterwards both copies agree, so nothing will ever flag it again. This is the
    # only value in the fabric that a transport cannot influence.
    content_digest: str | None = None
    unsealed_because: str | None = None
    # G053: OPTIONAL classification driving MEMORY_CLASS_POLICY. None (the
    # default) means "unclassified" and every existing caller that never set
    # this keeps working with zero policy enforced -- see mark_written().
    memory_class: MemoryClass | None = None

    # 5 valid copies
    def valid_copies(self) -> list[str]:
        """Copies holding the current value. THIS IS A QUESTION ABOUT STATE, and it
        deliberately still names a copy whose TRUST is UNKNOWN -- such a copy may be
        perfectly current, and folding the two axes together would make field 5 mean
        something other than what §32 says it means. Ask trusted_copies() when the
        question is what may actually be relied upon."""
        return [d for d, m in self.materializations.items() if m.state is State.CLEAN]

    def trusted_copies(self) -> list[str]:
        """Valid AND relied upon. The distinction matters most where it is easiest to
        miss: DATA-LOSS accounting. Counting an UNKNOWN copy as a survivor reports
        'you still have it' about a copy nobody can vouch for, which is the safe-
        looking direction and the wrong one."""
        return [d for d, m in self.materializations.items()
                if m.state is State.CLEAN and m.trust != "UNKNOWN"]

    # 6 dirty state
    def is_dirty(self) -> bool:
        return any(m.state is State.DIRTY for m in self.materializations.values())

    def place(self, m: Materialization) -> None:
        self.materializations[m.domain] = m

    def mark_written(self, domain: str) -> None:
        """A write makes this copy DIRTY and every other copy STALE. Nothing about
        that is implicit.

        G053: if this object is classified into an IMMUTABLE memory_class (per
        MEMORY_CLASS_POLICY), a write is refused outright rather than silently
        accepted -- an immutable object that could still be written through is
        not actually policed, it is only documented, and this fabric does not
        settle for that. Unclassified objects (memory_class is None) are
        unaffected, so no existing caller changes behaviour."""
        if domain not in self.materializations:
            raise HumfError(f"{self.identity} has no materialization in {domain}")
        if self.memory_class is not None and not MEMORY_CLASS_POLICY[self.memory_class]["mutable"]:
            raise HumfError(
                f"{self.identity} is classified {self.memory_class.value}, which "
                f"MEMORY_CLASS_POLICY marks immutable; refusing to write it in "
                f"{domain} rather than silently violating its own policy")
        self.materializations[domain].transition(State.DIRTY)
        # A legitimate write makes the recorded identity WRONG, and an identity check
        # that fires on legitimate writes gets turned off. Unseal instead: origin
        # checking is skipped, and says so, until seal_value() re-establishes it.
        self.content_digest = None
        self.unsealed_because = f"written in {domain}"
        self.owner = domain
        for d, m in self.materializations.items():
            if d != domain and m.state is State.CLEAN:
                m.transition(State.STALE)


@dataclass
class Plan:
    action: str
    cost_s: float
    detail: str
    cost_provenance: str
    options: list[dict[str, Any]]
    # THE PLAN NAMES ITS SOURCE. Without this _move re-derived `the first CLEAN
    # copy` on its own, so a plan reading `via TRANSFER from TRUSTED_SRC` could move
    # bytes out of a QUARANTINED domain -- the planner's refusal was decorative and
    # the log described a transfer that did not happen.
    source: str | None = None

    @property
    def rests_on_simulated_numbers(self) -> bool:
        return self.cost_provenance != "MEASURED"


class Humf:
    def __init__(self, domains: dict[str, Domain],
                 providers: dict[str, Any] | None = None,
                 identity_recheck_age: int | None = 0,
                 verify_transfers: bool = True,
                 transfer_timeout_s: float | None = None):
        self.domains = domains
        self.objects: dict[str, HumfObject] = {}
        self.log: list[dict[str, Any]] = []
        # Domains backed by a provider actually move bytes on execute(). Without
        # this the executor transitioned TRANSFERRING -> CLEAN on bookkeeping alone
        # and marked a copy valid that held nothing -- see the defect recorded in
        # ACCELERATOR_HUMF_FAILURE_INJECTION.json.
        self.providers: dict[str, Any] = providers or {}
        # A provider that FAILS raises and the state machine can react. A provider
        # that LIES returns the wrong bytes and raises nothing, so error handling is
        # no defence at all -- only VERIFICATION is. Enabled across provider-backed
        # domains only; see integrity_policy() for why that is not timidity.
        self.verify_transfers = verify_transfers
        # A monotonic event counter. Trust is stamped with it so a verification has
        # an AGE, which is the difference between "this was checked" and "this is
        # checked". Not a clock: wall time is not what invalidates a copy, activity
        # is, and a counter cannot drift or be adjusted underneath the fabric.
        self.epoch: int = 0
        # HOW OLD a source's verification may be before a transfer re-establishes it.
        # 0 = every transfer (the safe default), None = never (registration only), k =
        # only when the source has not been verified in the last k events. This is the
        # knob the cost forces: the identity digest runs at ~1.39 GB/s against crc32's
        # ~35.5, so on a 5 GB/s bridge an unconditional re-hash costs more than the
        # transfer it protects. Decay is what makes it affordable -- re-verify what
        # has gone unlooked-at, not what was just checked.
        self.identity_recheck_age = identity_recheck_age
        # A transport that HANGS is a third failure mode, and it is worse than one
        # that raises: nothing raises, nothing completes, and the destination sits in
        # TRANSFERRING forever with no path out because the except branch never runs.
        self.transfer_timeout_s = transfer_timeout_s
        # A domain that timed out or vanished is not trustworthy for the NEXT
        # transfer either. Quarantine is what an operator would do with a flaky link,
        # and doing it here is what stops a second transfer racing a first one that
        # is still running inside a worker thread we could not abandon.
        self.quarantined: dict[str, str] = {}

    def device_lost(self, domain: str, reason: str) -> dict[str, Any]:
        """The domain went away. EVERY copy it held is gone, not just the one being
        transferred.

        This is the case that separates a lost DEVICE from a failed COPY, and getting
        it wrong is silent: valid_copies() would keep naming copies in a domain that
        no longer exists, and the planner would keep choosing them as transfer
        sources. The report distinguishes objects that are merely less replicated
        from objects whose LIVE state was in that domain -- a DIRTY copy there was
        the only holder, so losing it is DATA LOSS and must be named as such rather
        than folded into a count of invalidated copies.
        """
        invalidated, data_lost = [], []
        for ident, obj in self.objects.items():
            m = obj.materializations.get(domain)
            if m is None or m.state in (State.ABSENT, State.EVICTED, State.INVALID):
                continue
            was_dirty = m.state is State.DIRTY
            m.transition(State.INVALID)
            m.payload, m.digest = None, None
            invalidated.append(ident)
            if was_dirty or (not obj.trusted_copies() and obj.recompute is None):
                data_lost.append(ident)
        self.quarantined[domain] = reason
        self.log.append({"action": "DEVICE_LOST", "domain": domain, "reason": reason,
                         "invalidated": invalidated, "data_lost": data_lost,
                         "at": time.time()})
        return {"domain": domain, "reason": reason, "invalidated": invalidated,
                "data_lost": data_lost,
                "means": "a lost device invalidates every copy it held; anything whose "
                         "live or only state was there is DATA LOSS, not degraded "
                         "replication"}

    def device_partially_lost(self, domain: str, reason: str) -> dict[str, Any]:
        """SOME allocations are gone and the device is still attached.

        THIS CANNOT REUSE device_lost, and the reason is the whole point. The
        full-loss handler is safe because of a BLANKET ASSUMPTION -- nothing in that
        domain survived -- and on a partial loss that assumption is WRONG IN BOTH
        DIRECTIONS. Assume everything survived and valid_copies() names phantoms.
        Assume everything died and a DIRTY copy that was still there becomes
        MANUFACTURED DATA LOSS: the fabric would destroy live state the device still
        held.

        So the fabric ASKS. Each copy is probed through the provider; the ones that
        answer survive, the ones that do not go INVALID. A partial loss is the case
        where the bookkeeping cannot be derived and has to be MEASURED.
        """
        prov = self.providers.get(domain)
        survived, lost, data_lost, unknown = [], [], [], []
        for ident, obj in self.objects.items():
            m = obj.materializations.get(domain)
            if m is None or m.state in (State.ABSENT, State.EVICTED, State.INVALID):
                continue
            # THE PROBE IS ITSELF A TRANSPORT OPERATION. The previous receipt named
            # this as an honest gap: it was not wrapped in the deadline the transfer
            # path uses, so a device that HANGS on the probe would hang the handler.
            # It is wrapped now -- and a TIMEOUT IS NOT A LOSS. Calling it lost would
            # be the manufactured-data-loss error one level up: the copy may be
            # perfectly fine and we merely could not ask. Calling it survived would
            # name a phantom. So there is a THIRD outcome, UNKNOWN, and the fabric
            # LEAVES THE STATE ALONE -- it must not act on information it does not
            # have. The domain is quarantined either way, which is what stops an
            # UNKNOWN copy being used before an operator resolves it.
            alive = False
            if prov is not None:
                try:
                    self._with_deadline(prov.copy_out, ident)
                    alive = True
                except TransferTimeout:
                    # The STATE is left alone -- CLEAN, payload intact -- because the
                    # copy may be perfectly fine. But its TRUST is now unknown, and
                    # that has to be recorded ON THE COPY: the quarantine protects it
                    # only until someone releases the link, and the link and the copy
                    # are two different questions.
                    m.trust = "UNKNOWN"
                    unknown.append(ident)
                    continue
                except Exception:
                    alive = False
            if alive:
                survived.append(ident)
                continue
            was_dirty = m.state is State.DIRTY
            m.transition(State.INVALID)
            m.payload, m.digest = None, None
            lost.append(ident)
            if was_dirty or (not obj.trusted_copies() and obj.recompute is None):
                data_lost.append(ident)
        self.quarantined[domain] = reason
        self.log.append({"action": "DEVICE_PARTIALLY_LOST", "domain": domain,
                         "reason": reason, "survived": survived, "lost": lost,
                         "unknown": unknown, "data_lost": data_lost,
                         "at": time.time()})
        return {"domain": domain, "reason": reason, "survived": survived,
                "lost": lost, "unknown": unknown, "data_lost": data_lost,
                "means": "each copy was PROBED under a deadline, not assumed. A blanket "
                         "verdict would either name phantoms or manufacture data loss, "
                         "and a probe that TIMED OUT is UNKNOWN -- state untouched, "
                         "counted as neither"}

    def release_quarantine(self, domain: str) -> list[str]:
        """Explicit, never automatic. A link that failed once is trusted again only
        because a person or a policy said so.

        It also clears the wreckage, and it has to: an INVALID copy cannot go
        straight back to TRANSFERRING (the state machine forbids it, correctly --
        INVALID means the contents are unknown, and pretending a transfer is
        resuming would be a lie about what happened). Recovery is INVALID -> ABSENT
        -> TRANSFERRING, and putting that here keeps it ONE explicit operator action
        rather than a sequence a caller can half-perform.

        WHAT IT DOES NOT DO, and this was the asymmetry the previous receipt named
        against itself: releasing the quarantine says THE LINK IS FINE. It does not
        say AND EVERY COPY ON IT IS GOOD. A copy whose probe timed out is UNKNOWN,
        and it stays UNKNOWN through a release -- otherwise an operator clearing a
        flaky bus would silently return copies to service whose trust was never
        re-established. They are reported, not cleared, and resolve_unknown() or
        accept_unknown() is the only way back.
        """
        self.quarantined.pop(domain, None)
        prov = self.providers.get(domain)
        if prov is not None:
            prov.present = True
        cleared, still_unresolved = [], []
        for ident, obj in self.objects.items():
            m = obj.materializations.get(domain)
            if m is None:
                continue
            if m.state is State.INVALID:
                m.transition(State.ABSENT)
                cleared.append(ident)
            elif m.trust == "UNKNOWN":
                still_unresolved.append(ident)
        self.log.append({"action": "RELEASE_QUARANTINE", "domain": domain,
                         "cleared": cleared, "still_unresolved": still_unresolved,
                         "at": time.time()})
        return {"domain": domain, "cleared": cleared,
                "still_unresolved": still_unresolved,
                "means": "the LINK is trusted again. Every copy listed in "
                         "still_unresolved is NOT -- a released quarantine is not a "
                         "certificate for what sits on it"}

    def resolve_unknown(self, domain: str, identity: str) -> dict[str, Any]:
        """Re-probe ONE copy whose trust is UNKNOWN and say what came back.

        RE-ESTABLISHING TRUST NEEDS SOMETHING TO CHECK AGAINST, and that is the whole
        difficulty. Where a digest was recorded when the copy was verified, a probe
        that returns matching bytes is a VERIFICATION. Where none was recorded there
        is nothing to compare to, so the probe establishes PRESENCE ONLY -- the copy
        answered, and that is all anyone can say about it. Reporting the second as if
        it were the first is exactly the kind of quiet promotion this fabric exists
        to refuse, so it stays UNKNOWN and an operator has to say accept_unknown().

        CARRIED FORWARD FROM THE VERIFICATION BLOCK: the probe reads through
        copy_out, the ROUND-TRIP path, not the path a kernel would use. So even
        VERIFIED here means the read-back path agrees, not that the resident bytes a
        compute would see are good.
        """
        obj = self.objects[identity]
        m = obj.materializations.get(domain)
        if m is None or m.trust != "UNKNOWN":
            raise HumfError(f"{identity} has no UNKNOWN copy in {domain} to resolve")
        prov = self.providers.get(domain)
        if prov is None:
            raise HumfError(f"{domain} has no provider to re-probe; only "
                            f"accept_unknown() can resolve a copy nobody can ask about")
        try:
            got = self._with_deadline(prov.copy_out, identity)
        except TransferTimeout:
            verdict, detail = "STILL_UNKNOWN", ("the probe timed out again; nothing "
                                                "changes, because nothing was learned")
        except Exception as e:
            was_dirty = m.state is State.DIRTY
            m.transition(State.INVALID)
            m.payload, m.digest = None, None
            verdict = "LOST"
            detail = (f"the probe answered with a failure, so the copy is gone rather "
                      f"than unknown: {e}")
            if was_dirty or (not obj.trusted_copies() and obj.recompute is None):
                verdict = "LOST_AND_UNRECOVERABLE"
        else:
            if m.digest is None:
                verdict = "PRESENT_BUT_UNVERIFIABLE"
                detail = ("the copy answered, and no digest was ever recorded for it, "
                          "so there is nothing to check the bytes against. PRESENCE "
                          "IS NOT INTEGRITY -- trust stays UNKNOWN and only "
                          "accept_unknown() can move it, as an assertion")
            elif _digest(got) == m.digest:
                m.trust = "TRUSTED"
                verdict = "VERIFIED"
                detail = (f"bytes match the digest recorded when this copy was "
                          f"verified ({m.digest:08x}); trust re-established against "
                          f"the READ-BACK path")
            else:
                m.transition(State.INVALID)
                m.payload, m.digest = None, None
                verdict = "CORRUPT"
                detail = ("the copy was there and it was WRONG -- the probe returned "
                          "bytes that do not match the recorded digest. This is the "
                          "case a timeout could have been hiding")
        self.log.append({"action": "RESOLVE_UNKNOWN", "domain": domain,
                         "object": identity, "verdict": verdict, "at": time.time()})
        return {"domain": domain, "object": identity, "verdict": verdict,
                "detail": detail, "trust": m.trust, "state": m.state.value}

    def accept_unknown(self, domain: str, identity: str, reason: str) -> dict[str, Any]:
        """An operator says the copy is fine. Recorded as an ASSERTION, never as a
        verification, because that is what it is -- nobody checked anything."""
        m = self.objects[identity].materializations.get(domain)
        if m is None or m.trust != "UNKNOWN":
            raise HumfError(f"{identity} has no UNKNOWN copy in {domain} to accept")
        m.trust = "ASSERTED"
        self.log.append({"action": "ACCEPT_UNKNOWN", "domain": domain,
                         "object": identity, "reason": reason,
                         "evidence": "NONE -- operator assertion", "at": time.time()})
        return {"domain": domain, "object": identity, "trust": "ASSERTED",
                "reason": reason,
                "means": "back in service on an operator's word. No probe ran, no "
                         "digest was compared, and the log says so"}

    def register(self, obj: HumfObject) -> None:
        self.objects[obj.identity] = obj
        self.seal_value(obj.identity)

    def seal_value(self, identity: str, domain: str | None = None) -> str | None:
        """Record what this object IS, from a copy the fabric already holds.

        Called at registration and after a write is complete. Cost is once per
        object, not once per transfer, which is why it can afford a cryptographic
        digest where the transfer check cannot -- see integrity_policy()."""
        obj = self.objects[identity]
        m = (obj.materializations.get(domain) if domain else
             next((x for x in obj.materializations.values()
                   if x.state is State.CLEAN and x.payload is not None), None))
        if m is None or m.payload is None:
            return None
        obj.content_digest = _identity_digest(m.payload)
        obj.unsealed_because = None
        self.epoch += 1
        m.verified_at = self.epoch
        return obj.content_digest

    def audit(self, identity: str) -> dict[str, Any]:
        """Re-check every copy against the sealed identity, NOW.

        This is what makes trust something with an age rather than a permanent
        label. Nothing calls it automatically -- a fabric that re-hashed on a timer
        would be paying for a check nobody asked for -- but the query
        stale_verifications() exists so a caller can find out what it is relying on
        that has not been looked at in a long time."""
        obj = self.objects[identity]
        if obj.content_digest is None:
            return {"object": identity, "audited": False,
                    "reason": f"value is UNSEALED ({obj.unsealed_because}); there is "
                              f"nothing to check against until seal_value()"}
        self.epoch += 1
        copies = {}
        for d, m in obj.materializations.items():
            if m.payload is None:
                copies[d] = "NO_BYTES_HELD"
                continue
            if _identity_digest(m.payload) == obj.content_digest:
                m.verified_at = self.epoch
                copies[d] = "MATCHES"
            else:
                m.trust = "UNKNOWN"
                copies[d] = "DIVERGED"
        return {"object": identity, "audited": True, "epoch": self.epoch,
                "copies": copies,
                "diverged": [d for d, v in copies.items() if v == "DIVERGED"]}

    def stale_verifications(self, max_age: int) -> list[dict[str, Any]]:
        """Copies whose trust rests on a check older than max_age fabric events."""
        return [{"object": o.identity, "domain": d, "verified_at": m.verified_at,
                 "age": self.epoch - m.verified_at, "trust": m.trust}
                for o in self.objects.values() for d, m in o.materializations.items()
                if m.state is State.CLEAN and self.epoch - m.verified_at > max_age]

    def _transfer_cost(self, src: str, dst: str, nbytes: int) -> tuple[float, str]:
        a, b = self.domains[src], self.domains[dst]
        bw = min(a.bandwidth_gb_s, b.bandwidth_gb_s) * 1e9
        prov = "MEASURED" if (a.physical and b.physical) else "SIMULATED"
        return nbytes / bw + a.latency_s + b.latency_s, prov

    def plan_acquire(self, identity: str, want_domain: str,
                     want_representation: str | None = None) -> Plan:
        """Steer §34: MOVE, COPY, RECOMPUTE, MATERIALIZE A DIFFERENT REPRESENTATION,
        or LEAVE IT AND MOVE THE COMPUTE. All estimated, cheapest wins."""
        obj = self.objects[identity]
        opts: list[dict[str, Any]] = []

        # QUARANTINE IS ABOUT TRUST, NOT DIRECTION. The check in execute() reads
        # `want_domain in self.quarantined` and is about the DESTINATION -- and for two
        # blocks that was the ONLY place quarantine appeared, so the planner would
        # happily offer a copy LIVING IN a quarantined domain as a transfer SOURCE.
        # DEMONSTRATED AGAINST THE PRE-FIX CODE: after a partial loss the plan came
        # back 'TRANSFER from MOCK_EXTERNAL_VRAM' without a word. That is the
        # silent-wrong-answer class -- reading from a domain the fabric had just
        # declared untrustworthy. A quarantined domain supplies NOTHING, in either
        # direction, until release_quarantine() says otherwise.
        here = obj.materializations.get(want_domain)
        if here and here.state is State.CLEAN and (
                want_representation in (None, here.representation)):
            if want_domain in self.quarantined:
                return Plan("IMPOSSIBLE", float("inf"),
                            f"{identity} is CLEAN in {want_domain} but that domain is "
                            f"QUARANTINED ({self.quarantined[want_domain]}); its "
                            f"contents are not to be relied upon until released",
                            "MEASURED", [])
            if here.trust == "UNKNOWN":
                return Plan("IMPOSSIBLE", float("inf"),
                            f"{identity} is CLEAN in {want_domain} but its trust is "
                            f"UNKNOWN -- a probe could not answer for this copy. "
                            f"Releasing the domain quarantine did not resolve it; "
                            f"resolve_unknown() re-probes, accept_unknown() records "
                            f"an operator's assertion instead",
                            "MEASURED", [])
            return Plan("ALREADY_RESIDENT", 0.0, f"{identity} is CLEAN in {want_domain}",
                        "MEASURED", [])

        for src, m in obj.materializations.items():
            if src == want_domain or m.state is not State.CLEAN:
                continue
            if src in self.quarantined or m.trust == "UNKNOWN":
                continue
            c, prov = self._transfer_cost(src, want_domain, m.bytes)
            opts.append({"action": "TRANSFER", "from": src, "bytes": m.bytes,
                         "representation": m.representation, "cost_s": c,
                         "cost_provenance": prov})

        if obj.recompute_cost_s is not None:
            opts.append({"action": "RECOMPUTE", "from": None,
                         "cost_s": obj.recompute_cost_s,
                         "cost_provenance": "MEASURED"})

        if not opts:
            blocked = [d for d, m in obj.materializations.items()
                       if m.state is State.CLEAN and d in self.quarantined]
            unresolved = [d for d, m in obj.materializations.items()
                          if m.state is State.CLEAN and m.trust == "UNKNOWN"
                          and d not in self.quarantined]
            if blocked:
                why = (f"{identity}'s only CLEAN copies are in QUARANTINED domains "
                       f"{sorted(blocked)}; refusing to source from a domain whose "
                       f"contents the fabric has declared untrustworthy")
            elif unresolved:
                why = (f"{identity}'s only CLEAN copies are UNRESOLVED in "
                       f"{sorted(unresolved)} -- the quarantine was released but the "
                       f"copies' trust was never re-established. A released link is "
                       f"not a certificate for what sits on it")
            else:
                why = f"{identity} has no CLEAN copy and no recompute recipe"
            return Plan("IMPOSSIBLE", float("inf"), why, "MEASURED", [])

        best = min(opts, key=lambda o: o["cost_s"])
        return Plan(best["action"], best["cost_s"],
                    f"{identity} -> {want_domain} via {best['action']}"
                    + (f" from {best['from']}" if best.get("from") else ""),
                    best["cost_provenance"], opts, source=best.get("from"))

    def execute(self, identity: str, plan: Plan, want_domain: str,
                representation: str = "dense_f32", layout: str = "row_major",
                nbytes: int | None = None) -> Materialization:
        obj = self.objects[identity]
        if plan.action == "ALREADY_RESIDENT":
            return obj.materializations[want_domain]
        if plan.action == "IMPOSSIBLE":
            raise HumfError(plan.detail)
        if want_domain in self.quarantined:
            raise HumfError(f"{want_domain} is QUARANTINED ({self.quarantined[want_domain]}); "
                            f"transfers into it are refused until released explicitly")
        dst = obj.materializations.get(want_domain)
        if dst is None:
            src_any = next(iter(obj.materializations.values()), None)
            dst = Materialization(want_domain, representation, layout,
                                  nbytes if nbytes is not None
                                  else (src_any.bytes if src_any else 0))
            obj.place(dst)
        dst.transition(State.TRANSFERRING if plan.action == "TRANSFER"
                       else State.MATERIALIZING)
        # MOVE THE BYTES. A transfer that is only a state change cannot fail, and a
        # transfer that cannot fail is not a transfer -- it is an assumption. If the
        # provider raises, the destination goes INVALID and stays out of
        # valid_copies(); the SOURCE is left untouched, because a failed outbound
        # transfer must not damage the copy that was good.
        try:
            self._move(obj, plan, want_domain, dst)
        except DeviceLost as e:
            dst.transition(State.INVALID)
            dst.payload = None
            self.log.append({"object": identity, "action": plan.action,
                             "domain": want_domain, "outcome": "DEVICE_LOST",
                             "at": time.time()})
            # the whole domain, not just this object
            self.device_lost(want_domain, str(e))
            raise
        except TransferTimeout as e:
            dst.transition(State.INVALID)
            dst.payload = None
            self.quarantined[want_domain] = str(e)
            self.log.append({"object": identity, "action": plan.action,
                             "domain": want_domain, "outcome": "TIMEOUT",
                             "at": time.time()})
            raise
        except Exception:
            dst.transition(State.INVALID)
            self.log.append({"object": identity, "action": plan.action,
                             "domain": want_domain, "outcome": "FAILED",
                             "at": time.time()})
            raise
        dst.transition(State.CLEAN)
        if obj.owner is None:
            obj.owner = want_domain
        self.log.append({"object": identity, "action": plan.action,
                         "domain": want_domain, "cost_s": plan.cost_s,
                         "cost_provenance": plan.cost_provenance,
                         "outcome": "OK", "at": time.time()})
        return dst

    def _move(self, obj: "HumfObject", plan: Plan, want_domain: str,
              dst: Materialization) -> None:
        """Actually put the data where the plan says it goes.

        RECOMPUTE runs the object's recipe. TRANSFER reads a valid source copy and
        hands it to the destination domain's provider, if that domain has one. A
        domain with no provider still only gets bytes it was given -- it never
        gets a payload conjured from nothing.
        """
        if plan.action == "RECOMPUTE":
            if obj.recompute is None:
                raise HumfError(f"{obj.identity} has no recompute recipe")
            dst.payload = obj.recompute()
            return
        # USE THE SOURCE THE PLAN NAMED. This used to re-derive `the first CLEAN
        # copy`, which meant the planner and the mover could disagree: DEMONSTRATED
        # against the pre-fix code, a plan reading `via TRANSFER from TRUSTED_SRC`
        # moved the bytes out of a QUARANTINED domain, so the refusal in
        # plan_acquire was decorative and the log described a transfer that had not
        # happened. A plan whose source is not what executes is not an audit trail.
        def _usable(d, m):
            return (d != want_domain and m.state is State.CLEAN
                    and d not in self.quarantined and m.trust != "UNKNOWN")
        if plan.source is not None:
            src = obj.materializations.get(plan.source)
            if src is None or not _usable(plan.source, src):
                raise HumfError(
                    f"the plan named {plan.source} as the source for {obj.identity} "
                    f"and that copy is no longer usable; refusing to substitute "
                    f"another one silently -- re-plan instead")
        else:
            src = next((m for d, m in obj.materializations.items()
                        if _usable(d, m)), None)
        if src is None or src.payload is None:
            raise HumfError(
                f"transfer of {obj.identity} into {want_domain} has no valid source "
                f"payload. A STALE or empty copy is never a transfer source.")
        prov = self.providers.get(want_domain)
        if prov is None:
            dst.payload = src.payload
            dst.digest = src.digest
            return
        # THE SOURCE IS CHECKED AGAINST THE SEALED IDENTITY BEFORE IT IS SENT. The
        # round-trip check below compares the source to the destination, so a source
        # that rotted in place is compared against a faithful copy of its own rot and
        # PASSES -- and after that both copies agree and nothing will ever flag it.
        # Measured in ACCELERATOR_HUMF_IDENTITY.json: a single flipped weight byte
        # propagated to a second domain through a check that reported success.
        recheck = (obj.content_digest is not None
                   and self.identity_recheck_age is not None
                   and self.epoch - src.verified_at >= self.identity_recheck_age)
        if recheck:
            now = _identity_digest(src.payload)
            if now != obj.content_digest:
                src.trust = "UNKNOWN"
                raise HumfError(
                    f"refusing to transfer {obj.identity} out of {src.domain}: the "
                    f"source no longer matches the identity sealed at registration "
                    f"({now[:16]} vs {obj.content_digest[:16]}). Nothing moved. The "
                    f"per-transfer check would have PASSED this, because it compares "
                    f"the source with the destination and both would have agreed.")
        expect = _digest(src.payload)
        src.digest = expect
        self._with_deadline(prov.copy_in, obj.identity, src.payload)
        got = self._with_deadline(prov.copy_out, obj.identity)
        if self.verify_transfers:
            # WHAT THIS CHECK PROVES AND WHAT IT DOES NOT. It compares the SOURCE
            # bytes against what copy_out RETURNS, so it validates THE ROUND TRIP.
            # It does NOT validate THE RESIDENT COPY, because on a real bridge
            # copy_out and a KERNEL are not the same path -- one comes back over the
            # transport, the other reads device memory directly. A provider whose
            # compute path diverges from its read-back path PASSES this check and
            # still computes on wrong bytes; MockExternalMemoryProvider.compute_skew
            # demonstrates exactly that, with a test showing the copy stays CLEAN.
            # THE FIX IS THE RESIDENT DIGEST BELOW, and it NARROWS this limit rather
            # than removing it -- see _check_resident.
            actual = _digest(got)
            if actual != expect:
                raise HumfError(
                    f"integrity check FAILED transferring {obj.identity} into "
                    f"{want_domain}: expected digest {expect:08x}, got {actual:08x}. "
                    f"The transport returned WITHOUT ERROR and returned the WRONG "
                    f"BYTES. Nothing but verification catches that.")
            self._check_resident(prov, obj, dst, src.payload, want_domain)
            dst.digest = actual
            self.epoch += 1
            dst.verified_at = self.epoch
        dst.payload = got

    def _check_resident(self, prov, obj, dst, source_bytes: bytes,
                        want_domain: str) -> None:
        """Ask the DEVICE to digest its own memory, through the path a KERNEL reads.

        WHY THE ROUND-TRIP CHECK ABOVE CANNOT DO THIS: it compares the source with
        what copy_out RETURNS, and on a real bridge copy_out and a kernel are not the
        same path -- one comes back over the transport, the other reads device memory
        directly. A provider whose compute path diverges from its read-back path
        passes the round trip and still computes on wrong bytes.

        WHAT IT COSTS ON A REAL BRIDGE, and this is the design argument rather than a
        measurement: a round-trip check has to bring the WHOLE PAYLOAD back across the
        link, while a resident digest brings back SIXTEEN BYTES. At a 24 MiB tensor on
        a 5 GB/s bridge that is ~4.8 ms against nothing. The identity block measured
        that a host blake2b re-hash costs MORE THAN THE MOVE at 1.387 GB/s; on the
        device the digest runs at device bandwidth and returns a constant. So moving
        the digest to the device attacks the blind spot AND the affordability problem
        at once -- but the DEVICE-SIDE DIGEST RATE IS UNMEASURED HERE and no speed
        claim attaches to it.

        AND IT IS NOT A CLOSURE. The check is only as strong as WHICH PATH the
        provider's digest reads, and THE FABRIC CANNOT VERIFY THAT CLAIM -- a provider
        that digests its read-back path exposes an identical API and passes on skewed
        memory. `resident_digest_path` is therefore RECORDED, not trusted, and a test
        pins that the dishonest variant still passes. The blind spot moves from
        `copy_out is not a kernel` to `the digest kernel may not be the compute
        kernel`, which is a narrower and STATED device property rather than an
        unconditional hole.
        """
        fn = getattr(prov, "digest_resident", None)
        if fn is None:
            dst.resident_verified = False
            dst.resident_digest_path = None
            return
        claimed = getattr(prov, "resident_digest_path", "UNDECLARED")
        seen = self._with_deadline(fn, obj.identity)
        want = _identity_digest(source_bytes)
        if seen != want:
            raise HumfError(
                f"RESIDENT integrity check FAILED for {obj.identity} in "
                f"{want_domain}: the device digested its own memory through its "
                f"{claimed!r} path and got {seen[:16]}, not {want[:16]}. The round-trip "
                f"check PASSED, so what came back over the transport was right and "
                f"what a kernel would read is not.")
        dst.resident_verified = True
        dst.resident_digest_path = claimed


def _with_deadline_impl(self, fn, *args):
    """Run a provider call under a deadline, and be exact about what that buys.

    WHAT IT DOES: the fabric regains control when the transport does not answer in
    time, the destination goes INVALID instead of sitting in TRANSFERRING forever,
    and the domain is QUARANTINED.

    WHAT IT DOES NOT DO: it cannot ABANDON the call. The worker thread is still
    inside the provider and may still complete, mutating provider state after the
    fabric has given up. That is why quarantine is not optional dressing -- refusing
    the next transfer into that domain is the only thing keeping a second call from
    racing the first. Real abandonment needs the transport to run somewhere it can be
    killed, which is not built here and is not claimed.
    """
    if self.transfer_timeout_s is None:
        return fn(*args)
    import concurrent.futures as _cf
    with _cf.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args)
        try:
            return fut.result(timeout=self.transfer_timeout_s)
        except _cf.TimeoutError:
            ex.shutdown(wait=False)
            raise TransferTimeout(
                f"transport did not answer within {self.transfer_timeout_s}s; the "
                f"call was NOT abandoned, only given up on -- the worker may still "
                f"be running inside the provider, which is why the domain is "
                f"quarantined rather than merely marked failed")


Humf._with_deadline = _with_deadline_impl


def _identity_digest(b: bytes) -> str:
    """WHAT the object IS, as opposed to whether a transfer glitched.

    crc32 is affine over GF(2), so repairing a checksum is a 32x32 linear solve over
    any 4 bytes you are allowed to touch -- 0.035 ms in Python, measured, not a
    search. A payload with a flipped weight byte and four repaired padding bytes
    passed this fabric's own integrity check, was marked CLEAN, and counted as a
    TRUSTED copy. That is not a flaw in crc32: it caught 100% of every accidental
    corruption class measured (single, double and eight-bit flips, 4- and 64-byte
    bursts, 4000 trials each). ERROR DETECTION IS NOT IDENTITY, and the two jobs get
    two functions. This one is paid ONCE PER OBJECT at seal time, not once per
    transfer, so its ~1.37 GB/s costs nothing that matters.
    """
    import hashlib
    return hashlib.blake2b(b, digest_size=16).hexdigest()


def _digest(b: bytes) -> int:
    """CRC32, not a cryptographic hash, and the reason is measured.

    Measured on this machine, across two independent probes: crc32 ~37-38 GB/s,
    sha256 ~2.7-2.9, blake2b ~1.37. crc32 is roughly 13x faster than sha256, which
    puts the affordability crossover -- where verification costs a quarter of the
    transfer -- at about 9 GB/s. Below that a check is cheap; a 32 GB/s link would
    put it back in question, so this is a threshold and not a blanket rule. The threat here is a
    FAULTY transport, not an adversarial one -- a cable, a reset, a driver bug --
    and a checksum is the right tool for accident. It is NOT the right tool for a
    malicious peer, and calling it integrity rather than authentication is the
    honest name.
    """
    return zlib.crc32(b) & 0xFFFFFFFF


def integrity_policy(transport_gb_s: float, checksum_gb_s: float = 38.44) -> dict:
    """What verification costs as a fraction of the transfer it protects.

    The conclusion has a shape worth stating: verification is affordable EXACTLY in
    the regime where an external bridge is used. Over Apple unified memory at
    589.73 GB/s a crc32 pass costs 15x the "transfer" -- and that is fine, because
    there is no transport there to corrupt. Over a Thunderbolt or PCIe link of a
    few GB/s it costs well under a tenth. The domain that NEEDS checking is the
    domain that can AFFORD it.
    """
    overhead = transport_gb_s / checksum_gb_s
    return {"transport_gb_s": transport_gb_s, "checksum_gb_s": checksum_gb_s,
            "verification_cost_as_multiple_of_transfer": round(overhead, 3),
            "affordable": overhead < 0.25,
            "note": "checksum_gb_s is MEASURED on this machine; transport_gb_s is "
                    "whatever the caller supplies and is NOT measured here"}


class MockExternalMemoryProvider:
    """Proves the HUMF state machine, NOT GPU performance. Its bandwidth is a knob,
    not a measurement, and every domain it hands out is physical=False."""

    def __init__(self, *, capacity_bytes: int, bandwidth_gb_s: float,
                 latency_s: float = 0.0):
        self.domain = Domain("MOCK_EXTERNAL_VRAM", capacity_bytes, bandwidth_gb_s,
                             physical=False, latency_s=latency_s)
        self.allocated = 0
        self.fail_next: str | None = None      # failure injection
        self.corrupt_next: bool = False        # the harder mode: LIE, do not fail
        # The HARDEST mode: return chosen bytes. corrupt_next models a FAULT, which is
        # what crc32 is designed against. substitute_next models bytes that were
        # SHAPED -- by a hostile peer or, far more likely on a bridge, by a buggy
        # driver reusing a stale buffer whose checksum happens to agree.
        self.substitute_next: bytes | None = None
        self.tear_next: bool = False           # store only part of what was handed in
        self.hang_next_s: float = 0.0          # a transport that is slow, not broken
        self.vanish_next: bool = False         # the device leaves mid-transfer
        self.present: bool = True
        # A PARTIAL loss: a reset that clears SOME allocations. Harder than a full
        # loss because the fabric cannot tell which copies survived without asking.
        self.lose_keys: set[str] = set()
        # The compute path returning DIFFERENT bytes from the read-back path. This is
        # the hazard a round-trip check cannot see; see read_for_compute.
        self.compute_skew: bool = False
        # Real bytes, so a round trip proves the DATA survived rather than only the
        # bookkeeping agreeing with itself.
        self.store: dict[str, bytes] = {}

    def allocate(self, nbytes: int) -> int:
        if self.fail_next == "allocate":
            self.fail_next = None
            raise HumfError("injected failure: allocate")
        if self.allocated + nbytes > self.domain.bytes_capacity:
            raise HumfError(f"out of capacity: {self.allocated + nbytes} > "
                            f"{self.domain.bytes_capacity}")
        self.allocated += nbytes
        return nbytes

    def free(self, nbytes: int) -> None:
        self.allocated = max(0, self.allocated - nbytes)

    def copy_in(self, key: str, payload: bytes) -> None:
        if self.fail_next == "copy":
            self.fail_next = None
            raise HumfError("injected failure: copy_in")
        self._maybe_hang()
        self._require_present()
        if self.tear_next:
            # A TORN WRITE: the transport delivered a prefix and stopped, leaving the
            # tail as whatever was there before. No exception, plausible-looking bytes.
            self.tear_next = False
            head = bytes(payload[: len(payload) // 2])
            tail = self.store.get(key, b"\x00" * len(payload))[len(payload) // 2:]
            self.store[key] = head + tail[: len(payload) - len(head)].ljust(
                len(payload) - len(head), b"\x00")
            return
        self.store[key] = bytes(payload)

    def _maybe_hang(self) -> None:
        if self.hang_next_s:
            d, self.hang_next_s = self.hang_next_s, 0.0
            time.sleep(d)

    def _require_present(self) -> None:
        if self.vanish_next:
            self.vanish_next = False
            self.present = False
            self.store.clear()          # the device left; so did everything on it
        if not self.present:
            raise DeviceLost(f"{self.domain.name} is no longer attached")

    def lose(self, keys) -> None:
        """A PARTIAL device loss: these allocations are gone, the rest survive. The
        device stays PRESENT, which is exactly what makes it harder than vanishing."""
        for k in keys:
            self.store.pop(k, None)
            self.lose_keys.add(k)

    def read_for_compute(self, key: str) -> bytes:
        """What a KERNEL would see, as distinct from what copy_out returns.

        On a real bridge these are not the same path: copy_out goes back over the
        transport, while a kernel reads device memory directly. compute_skew makes
        them differ so the fabric's round-trip check can be shown for what it is.
        """
        if key not in self.store:
            raise HumfError(f"{key} is not resident in {self.domain.name}")
        b = self.store[key]
        if self.compute_skew:
            b = bytearray(b); b[-1] ^= 0xFF; b = bytes(b)
        return b

    # WHICH PATH THIS PROVIDER DIGESTS. "compute" means digest_resident reads through
    # read_for_compute, the same path a kernel uses, so it CATCHES compute_skew.
    # ReadbackDigestProvider declares "readback" and does not. The fabric RECORDS this
    # and cannot verify it -- see Humf._check_resident.
    resident_digest_path: str = "compute"

    def digest_resident(self, key: str) -> str:
        """The device digests ITS OWN MEMORY, through the path a kernel reads."""
        self._maybe_hang()
        self._require_present()
        return _identity_digest(self.read_for_compute(key))

    def copy_out(self, key: str) -> bytes:
        if self.fail_next == "copy":
            self.fail_next = None
            raise HumfError("injected failure: copy_out")
        self._maybe_hang()
        self._require_present()
        if key not in self.store:
            raise HumfError(f"{key} is not resident in {self.domain.name}")
        if self.corrupt_next:
            self.corrupt_next = False
            b = bytearray(self.store[key])
            b[0] ^= 0xFF                       # one flipped bit pattern, no exception
            return bytes(b)
        if self.substitute_next is not None:
            b, self.substitute_next = self.substitute_next, None
            return b
        return self.store[key]


class ReadbackDigestProvider(MockExternalMemoryProvider):
    """A provider whose device-side digest reads THE READ-BACK PATH, not the compute
    path -- and which is INDISTINGUISHABLE from an honest one through the fabric's API.

    This exists as the control that gives the resident check its meaning. It offers
    digest_resident, it answers, its digest matches the source, and it is still WRONG
    about what a kernel would see. The strength of the resident check is a property of
    the DEVICE's implementation, not of the fabric's protocol, and the only honest move
    is to record the claim rather than trust it.
    """
    resident_digest_path = "readback"

    def digest_resident(self, key: str) -> str:
        self._maybe_hang()
        self._require_present()
        return _identity_digest(super().copy_out(key))
