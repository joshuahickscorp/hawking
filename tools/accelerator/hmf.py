"""HMF -- the canonical name for the Hawking Memory Fabric. FRONT H (G053).

CANONICALIZATION. G053 asked for `hmf` to become the canonical module name while
`humf` (tools/accelerator/humf.py) stays importable so nothing that already
depends on it breaks. The cheapest honest way to do that is the way this file
does it: hmf.py imports humf's classes and re-exports the SAME objects, and
humf.py is not touched beyond the additions its own docstring names. There is
no second implementation here -- `hmf.Humf is humf.Humf` is pinned by a test in
test_hmf.py precisely because this campaign has already shipped one module
reachable under two dotted names where that was False (F24, MEMORY.md
2026-08-23). If that test ever fails, this file has drifted into a competing
implementation and that is the bug, not the test.

WHAT IS DEFINED HERE, GENUINELY NEW, NOT A RENAME:
  - Nothing about State, Ownership, MemoryClass or their policy tables. Those
    are re-exported from humf.py, where they had to be added because
    Materialization and HumfObject -- the dataclasses they extend -- are
    defined there. See humf.py's own docstring and the State/Ownership/
    MemoryClass class docstrings for what changed and why.
  - HAWKGPU-0: a logical accelerator (Accelerator) holding one or more device
    domains (AcceleratorDomain). Today exactly one is real: APPLE_DOMAIN_0,
    wrapping this machine's actual Metal/unified-memory domain. Every other
    slot (SPARK_DOMAIN_0, ...) does not exist on this machine; test_hmf.py
    registers a FICTITIOUS one purely to demonstrate the structure needs no
    architectural change to hold a second domain -- see
    test_hmf.py::test_a_second_fictitious_domain_needs_no_architecture_change.
  - DeviceIdentity / MachineIdentity: minimal per-domain identity records,
    following the same ABSENT-with-a-reason discipline receipt.py already
    uses for identities that do not apply to a mocked/fictitious domain.

HARD RULE ON NUMBERS: every bandwidth/latency figure that appears below for a
domain with physical=False is a KNOB, chosen for structural testing, and is
labelled SIMULATED via Domain.provenance exactly like every other mock domain
in humf.py. APPLE_DOMAIN_0's bandwidth (589.73 GB/s) is NOT re-measured here --
it is the same figure already carried as a fixture constant across
test_humf.py and test_air.py for the real APPLE_UM domain on this machine, and
physical=True here matches that existing, established convention. No external
GPU and no Spark hardware exist on this machine; nothing in this file claims
otherwise.

WHY "APPLE_DOMAIN_0" AND NOT "APPLE_UM": these are two different namespaces on
purpose. "APPLE_UM" is the memory-domain NAME that flows unchanged between AIR
(air.py's MEMORY_DOMAINS) and HUMF (Humf.domains) -- a test already pins that
identity with no translation layer, and this file does not touch it.
"APPLE_DOMAIN_0" is HAWKGPU-0's own device-SLOT identifier, one level up: which
physical accelerator, not which memory domain. AcceleratorDomain.fabric_domain
is the seam between the two -- its .name is literally "APPLE_UM", so a caller
who takes Accelerator.fabric_domains() and hands it straight to Humf() gets the
exact same vocabulary AIR already expects, unchanged.

WHAT HAWKGPU-0 HIDES AND WHAT IT DOES NOT. Accelerator.fabric_domains() returns
plain humf.Domain objects -- name, capacity, bandwidth, physical, latency -- and
nothing else; a caller above HAWKGPU-0 (a Humf fabric, a scheduler working only
in HUMF's vocabulary) never sees backend, DeviceIdentity, MachineIdentity or
topology. That is the hiding the module name promises. It is NOT hidden from the
Accelerator itself: Accelerator.domains keeps every AcceleratorDomain with its
full identity, backend and topology intact, precisely so a caller who legitimately
needs to reason about asymmetry (a scheduler deciding where to place work) has
somewhere to look. See test_hmf.py::test_per_domain_asymmetry_is_visible_through_the_accelerator.

NOT IMPLEMENTED, named rather than left silent:
  - No cross-device transport. HAWKGPU-0 hands Domain objects to a Humf fabric;
    it does not itself move bytes, and AcceleratorDomain carries no provider.
    Wiring an AcceleratorDomain to a humf provider (as MockExternalMemoryProvider
    already demonstrates for a single mock domain) is not done here.
  - No topology COST MODEL. AcceleratorDomain.topology is a bare dict of
    knob numbers (peer slot -> a bandwidth figure) with no consumer; nothing
    here computes a route or a cost across more than one hop.
  - No scheduler. HAWKGPU-0 exposes the asymmetry a scheduler would need; it
    does not decide placement itself.
  - Ownership is not wired into Humf.plan_acquire()/execute() at all --
    OWNERSHIP_LEGAL governs Materialization.transition_ownership() in
    isolation. Nothing in the fabric's transfer/recompute planner reads or sets
    ownership yet; that integration is future work, not claimed here.
  - MemoryClass beyond the `mutable` policing in HumfObject.mark_written():
    eviction priority, capacity accounting and residency scheduling per class
    are described in MEMORY_CLASS_POLICY's `note` strings but nothing acts on
    them beyond the immutability guard.
  - SPARK0_BIAS / SPARK1_BIAS ownership and SPARK_DOMAIN_0-style accelerator
    domains have no real backing anywhere: no Spark hardware exists on this
    machine, full stop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# CANONICAL RE-EXPORT, NOT A COPY. Every name below is the SAME object humf.py
# defines -- see the module docstring and test_hmf.py::test_hmf_is_humf_by_identity.
import humf
from humf import (  # noqa: F401
    LEGAL,
    MEMORY_CLASS_POLICY,
    OWNERSHIP_LEGAL,
    Domain,
    DeviceLost,
    Humf,
    HumfError,
    HumfObject,
    Materialization,
    MemoryClass,
    MockExternalMemoryProvider,
    Ownership,
    Plan,
    ReadbackDigestProvider,
    State,
    TransferTimeout,
    integrity_policy,
)


class HmfError(HumfError):
    """This module's own errors are HumfError subclasses so a caller catching
    HumfError -- the fabric's existing contract -- keeps working across the
    canonicalization without knowing hmf.py exists."""


@dataclass(frozen=True)
class DeviceIdentity:
    """Per steer S015 §79's discipline (receipt.py: NO RESULT WITHOUT PHYSICAL
    IDENTITY, ABSENT-with-a-reason where one does not apply): a fictitious
    domain's DeviceIdentity is not omitted and not invented, it is recorded
    ABSENT with a reason."""
    vendor: str
    device_class: str
    physical: bool
    absent_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.physical and not self.absent_reason:
            raise HmfError(
                f"DeviceIdentity({self.vendor}/{self.device_class}) has "
                f"physical=False with no absent_reason; a simulated device's "
                f"identity must say WHY it is not real, not just that it isn't")


@dataclass(frozen=True)
class MachineIdentity:
    """The machine a domain's compute actually runs on. See DeviceIdentity for
    why physical=False requires a reason."""
    soc: str
    machine_class: str
    physical: bool
    absent_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.physical and not self.absent_reason:
            raise HmfError(
                f"MachineIdentity({self.soc}/{self.machine_class}) has "
                f"physical=False with no absent_reason; a simulated machine's "
                f"identity must say WHY it is not real, not just that it isn't")


@dataclass
class AcceleratorDomain:
    """One device domain inside a logical Accelerator (HAWKGPU-0).

    `slot` is the accelerator-relative identifier ("APPLE_DOMAIN_0"), distinct
    from `fabric_domain.name` ("APPLE_UM") -- the HUMF/AIR-facing memory-domain
    name that must stay untranslated. Everything else here is the per-domain
    detail HAWKGPU-0 keeps visible to the Accelerator and hides from callers
    that only ever see fabric_domain: which device this is, which machine it
    runs on, what backend drives it, what representations it can hold, and its
    KNOWN links to other domains (a knob dict, see module docstring)."""
    slot: str
    fabric_domain: Domain
    device_identity: DeviceIdentity
    machine_identity: MachineIdentity
    backend: str
    supported_representations: frozenset[str]
    topology: dict[str, float] = field(default_factory=dict)


class Accelerator:
    """HAWKGPU-0: a logical accelerator holding one or more AcceleratorDomains.

    Structurally this is just a name plus a dict -- deliberately. The point of
    G053's second-domain test is that NOTHING about this class needs to change
    to go from one real domain to two (one real, one fictitious) or more; if it
    did, the abstraction would not have generalized and the test proving that
    is the only evidence that matters (see module docstring)."""

    def __init__(self, name: str = "HAWKGPU-0") -> None:
        self.name = name
        self.domains: dict[str, AcceleratorDomain] = {}

    def register_domain(self, dom: AcceleratorDomain) -> None:
        if dom.slot in self.domains:
            raise HmfError(
                f"domain slot {dom.slot!r} is already registered on {self.name}; "
                f"refusing to silently replace it")
        self.domains[dom.slot] = dom

    def fabric_domains(self) -> dict[str, Domain]:
        """What HAWKGPU-0 hands to a Humf fabric: plain Domain objects keyed by
        their FABRIC name (not the accelerator slot). This is the seam that
        hides backend/identity/topology from callers above HAWKGPU-0 -- a
        Humf built from this dict sees exactly what it would have seen without
        HAWKGPU-0 existing at all, see test_hmf.py::
        test_hawkgpu_hides_topology_from_a_caller_using_only_fabric_domains."""
        return {d.fabric_domain.name: d.fabric_domain for d in self.domains.values()}


def build_hawkgpu0() -> Accelerator:
    """HAWKGPU-0 as it exists TODAY on this machine: one domain, APPLE_DOMAIN_0,
    wrapping the real Apple Silicon Metal/unified-memory domain already used
    throughout humf.py and air.py as "APPLE_UM". supported_representations is
    kept to exactly what this module's tests actually exercise elsewhere
    ("dense_f32") rather than an unfounded claim of broader hardware support.
    """
    accel = Accelerator("HAWKGPU-0")
    accel.register_domain(AcceleratorDomain(
        slot="APPLE_DOMAIN_0",
        fabric_domain=Domain("APPLE_UM", 96 << 30, 589.73, physical=True),
        device_identity=DeviceIdentity(
            vendor="Apple", device_class="APPLE_GPU_UMA", physical=True),
        machine_identity=MachineIdentity(
            soc="Apple Silicon (this machine)", machine_class="APPLE_SILICON_UMA",
            physical=True),
        backend="metal",
        supported_representations=frozenset({"dense_f32"}),
    ))
    return accel
