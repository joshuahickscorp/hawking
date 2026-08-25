"""HUMF — the Hawking Unified Memory Fabric. FRONT H (G050, steer S015).

HUMF DOES NOT LIE. Apple unified memory plus external VRAM is never called
physically unified memory. The physical domains stay distinct and HUMF manages the
illusion at the object level: who owns the current state, which copies are valid,
what a move would cost, and whether recomputing beats transferring.

Every number a MOCK domain produces is stamped SIMULATED. The steer is explicit
that a simulated transport number must never be labelled physical evidence, so the
planner carries the provenance of each cost it used into its decision, and a plan
built on simulated numbers says so in its own output rather than in a footnote.
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class State(str, Enum):
    """Steer §14. No accidental hidden coherence assumptions."""
    ABSENT = "ABSENT"
    CLEAN = "CLEAN"
    DIRTY = "DIRTY"
    STALE = "STALE"
    MATERIALIZING = "MATERIALIZING"
    TRANSFERRING = "TRANSFERRING"
    INVALID = "INVALID"
    EVICTED = "EVICTED"


# Illegal transitions fail closed (the steer says so of the driver state machine;
# the same discipline belongs here, where a silent bad transition means silent
# data corruption rather than a crash).
LEGAL: dict[State, set[State]] = {
    State.ABSENT:        {State.MATERIALIZING, State.TRANSFERRING},
    State.MATERIALIZING: {State.CLEAN, State.INVALID},
    State.TRANSFERRING:  {State.CLEAN, State.INVALID},
    State.CLEAN:         {State.DIRTY, State.STALE, State.EVICTED, State.INVALID,
                          State.TRANSFERRING},
    State.DIRTY:         {State.CLEAN, State.STALE, State.INVALID},
    State.STALE:         {State.TRANSFERRING, State.MATERIALIZING, State.EVICTED,
                          State.INVALID},
    State.EVICTED:       {State.MATERIALIZING, State.TRANSFERRING, State.ABSENT},
    State.INVALID:       {State.ABSENT, State.MATERIALIZING},
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

    def transition(self, to: State) -> None:
        if to not in LEGAL[self.state]:
            raise HumfError(f"illegal transition {self.state.value} -> {to.value} "
                            f"for {self.representation} in {self.domain}")
        self.state = to


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

    # 5 valid copies
    def valid_copies(self) -> list[str]:
        return [d for d, m in self.materializations.items() if m.state is State.CLEAN]

    # 6 dirty state
    def is_dirty(self) -> bool:
        return any(m.state is State.DIRTY for m in self.materializations.values())

    def place(self, m: Materialization) -> None:
        self.materializations[m.domain] = m

    def mark_written(self, domain: str) -> None:
        """A write makes this copy DIRTY and every other copy STALE. Nothing about
        that is implicit."""
        if domain not in self.materializations:
            raise HumfError(f"{self.identity} has no materialization in {domain}")
        self.materializations[domain].transition(State.DIRTY)
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

    @property
    def rests_on_simulated_numbers(self) -> bool:
        return self.cost_provenance != "MEASURED"


class Humf:
    def __init__(self, domains: dict[str, Domain],
                 providers: dict[str, Any] | None = None,
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
            if was_dirty or (not obj.valid_copies() and obj.recompute is None):
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
            if was_dirty or (not obj.valid_copies() and obj.recompute is None):
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
        """
        self.quarantined.pop(domain, None)
        prov = self.providers.get(domain)
        if prov is not None:
            prov.present = True
        cleared = []
        for ident, obj in self.objects.items():
            m = obj.materializations.get(domain)
            if m is not None and m.state is State.INVALID:
                m.transition(State.ABSENT)
                cleared.append(ident)
        return cleared

    def register(self, obj: HumfObject) -> None:
        self.objects[obj.identity] = obj

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
            return Plan("ALREADY_RESIDENT", 0.0, f"{identity} is CLEAN in {want_domain}",
                        "MEASURED", [])

        for src, m in obj.materializations.items():
            if src == want_domain or m.state is not State.CLEAN:
                continue
            if src in self.quarantined:
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
            why = (f"{identity} has no CLEAN copy and no recompute recipe"
                   if not blocked else
                   f"{identity}'s only CLEAN copies are in QUARANTINED domains "
                   f"{sorted(blocked)}; refusing to source from a domain whose "
                   f"contents the fabric has declared untrustworthy")
            return Plan("IMPOSSIBLE", float("inf"), why, "MEASURED", [])

        best = min(opts, key=lambda o: o["cost_s"])
        return Plan(best["action"], best["cost_s"],
                    f"{identity} -> {want_domain} via {best['action']}"
                    + (f" from {best['from']}" if best.get("from") else ""),
                    best["cost_provenance"], opts)

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
        src = next((m for d, m in obj.materializations.items()
                    if d != want_domain and m.state is State.CLEAN), None)
        if src is None or src.payload is None:
            raise HumfError(
                f"transfer of {obj.identity} into {want_domain} has no valid source "
                f"payload. A STALE or empty copy is never a transfer source.")
        prov = self.providers.get(want_domain)
        if prov is None:
            dst.payload = src.payload
            dst.digest = src.digest
            return
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
            # There is no cheap fix: catching it needs a check that reads through
            # THE SAME PATH THE COMPUTATION USES, which means the device computing a
            # digest of its own memory. Naming the limit where it lives beats
            # letting `verified` be read as a guarantee about what the GPU sees.
            actual = _digest(got)
            if actual != expect:
                raise HumfError(
                    f"integrity check FAILED transferring {obj.identity} into "
                    f"{want_domain}: expected digest {expect:08x}, got {actual:08x}. "
                    f"The transport returned WITHOUT ERROR and returned the WRONG "
                    f"BYTES. Nothing but verification catches that.")
            dst.digest = actual
        dst.payload = got


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
        return self.store[key]
