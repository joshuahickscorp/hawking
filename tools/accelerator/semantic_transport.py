"""Semantic transport -- a typed edge between two execution domains.

This is the device-neutral half of the Fusion Bridge. It does not move bytes.
It does not talk to a driver. It records what a transfer WOULD mean if a
backend later performed it: payload semantics, ownership handoff, the
synchronization the consumer actually needs, and the coherency the PLAN is
allowed to assume -- including the explicit assumption that there is none.

WHY THIS EXISTS. fusion_planner.Link already carries bandwidth, latency and
a physical/simulated flag between named domains. fusion_wire encodes a 42-byte
command header with no payload. fusion_isa records COPY/MATERIALIZE/FENCE on a
timeline. HWIR's atlas already *names* SemanticTransportEdge as a primitive
and lowers it to a dma-transport node inside one FPGA graph. None of those
is a typed INTER-DOMAIN edge that can say "this link is not coherent" and
have a validator refuse a plan that pretends otherwise. That is this module.

COHERENCY LADDER (roadmap §16.2), weakest to strongest:

    NONE                      explicit non-coherence (the missing rung)
    NETWORK_SERVING           0
    DISTRIBUTED_EXECUTION     1
    HGVAS                     2
    SOFTWARE_MANAGED          3
    OWNERSHIP_VERSIONED       4
    KERNEL_BOUNDARY           5
    HARDWARE_UMA              6  true hardware cache-coherent unified memory

Hawking targets 2-5 across heterogeneous devices. Rung 6 is legal ONLY as
the internal coherency of a GPU_UMA domain. It is never a legal assumption
about a link between two distinct domains. A plan that assumes a stronger
rung than the link declares is refused -- that is the load-bearing check.

READBACK VS DEVICE-COMPUTE VISIBILITY (roadmap §16.1):
    PRESENT != VALID != TRUSTED.
    READBACK CORRECTNESS DOES NOT PROVE DEVICE-COMPUTE VISIBILITY.
DomainVisibility stores the two axes independently. Evidence for device
compute that is just "the host read the bytes back" is refused.

COST. Every Cost.label is COST_MODEL. HARDWARE_MEASURED is a forbidden
token: constructing a Cost with that label raises, and no helper here
assigns it. A number taken from fusion_planner.Route is still COST_MODEL;
the planner's own MEASURED/SIMULATED hop flag is preserved only as
`all_hops_physical`, which does not promote the label.

NOT IMPLEMENTED, named rather than left silent:
  - No real transport. No DMA, no PCIe, no interconnect driver.
  - No in-transit compute. The transform names are a vocabulary, not a
    kernel. A backend that actually pack/unpack/quantize on the wire is
    out of scope.
  - No persistent command-buffer runtime (roadmap §15.9). The edge can
    name a token_id; nothing here patches a resident command program.
  - No vendor backend. DomainKind is GPU_UMA / CPU / NPU / FPGA_HBM.
    Which product fills a kind is a backend's fact, not a branch here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum


COST_MODEL = "COST_MODEL"
HARDWARE_MEASURED = "HARDWARE_MEASURED"  # forbidden; never assigned to Cost.label

# In-transit transforms: union of roadmap §15.8 and HWIR TRANSFORMS.
# Names match HWIR where they overlap so a lowerer can pass them through.
IN_TRANSIT_TRANSFORMS = frozenset({
    "identity",
    "unpack",
    "pack",
    "quantize",
    "dequantize",
    "transpose",
    "scatter",
    "gather",
    "codebook_lookup",
    "scales",
    "checksum_digest",
    "reduce",
    "compression",
})


class SemanticTransportError(RuntimeError):
    """Base for every error this module raises."""


class CostLabelError(SemanticTransportError):
    """A Cost was constructed with a label other than COST_MODEL."""


class DomainKind(str, Enum):
    """Device-neutral execution-domain kinds. Backends bind products to these;
    this module never branches on a product name."""
    GPU_UMA = "GPU_UMA"
    CPU = "CPU"
    NPU = "NPU"
    FPGA_HBM = "FPGA_HBM"


class PayloadSemantics(str, Enum):
    """What the bytes MEAN, matching HawkFrame / HWIR frame kinds -- not
    'bytes at address X'."""
    ACTIVATION = "activation"
    PARTIAL_REDUCTION = "partial_reduction"
    COMPACT_REPRESENTATION = "compact_representation_fragment"
    STATE = "state"
    CODEBOOK_ID = "codebook_id"
    SPARSE_RESIDUAL = "sparse_residual"
    WEIGHTS = "weights"
    TOKEN = "token"
    COMMAND = "command"


class OwnershipTransfer(str, Enum):
    """What happens to write-authority when the edge fires. Orthogonal to
    coherency (humf.Ownership vs humf.State). KEEP = source remains owner;
    TRANSFER = exclusive ownership moves to destination; SHARE_READ = both
    sides may read, neither holds exclusive write."""
    KEEP = "keep"
    TRANSFER = "transfer"
    SHARE_READ = "share_read"


class SyncRequirement(str, Enum):
    """What the consumer needs before it may use the payload. NONE is a
    real choice -- a non-coherent streaming link with no barrier -- not a
    missing field."""
    NONE = "none"
    FENCE = "fence"
    ACQUIRE_RELEASE = "acquire_release"
    PRODUCER_CONSUMER = "producer_consumer"


class OrderingGuarantee(IntEnum):
    """Happens-before a link provides, or a plan claims.

    Parallel to CoherencyAssumption: an assumption whose rank exceeds what
    the transport actually guarantees is illegal. NONE is -1 so it is
    strictly weaker than every named protocol.

        NONE                 no happens-before (torn/late/never are legal)
        FENCE                explicit barrier the consumer waits on
        ACQUIRE_RELEASE      ownership-synchronized acquire/release
        PRODUCER_CONSUMER    full handshake; producer will not overwrite
                             until the consumer has acquired
    """
    NONE = -1
    FENCE = 0
    ACQUIRE_RELEASE = 1
    PRODUCER_CONSUMER = 2


_SYNC_TO_ORDERING = {
    SyncRequirement.NONE: OrderingGuarantee.NONE,
    SyncRequirement.FENCE: OrderingGuarantee.FENCE,
    SyncRequirement.ACQUIRE_RELEASE: OrderingGuarantee.ACQUIRE_RELEASE,
    SyncRequirement.PRODUCER_CONSUMER: OrderingGuarantee.PRODUCER_CONSUMER,
}


def ordering_from_sync(sync: SyncRequirement) -> OrderingGuarantee:
    """Map a requested protocol onto the happens-before it provides.

    A software fence/handshake is a protocol, not hardware coherency: it
    is provided even on a link whose coherency is NONE, because the
    consumer actually waits. Requesting NONE provides nothing.
    """
    try:
        return _SYNC_TO_ORDERING[sync]
    except KeyError as exc:
        raise SemanticTransportError(f"unknown SyncRequirement {sync!r}") from exc


class CoherencyAssumption(IntEnum):
    """§16.2 ladder plus an explicit NONE. Integer values are the rank used
    by the overclaim check: an assumption whose rank exceeds the link's
    declared rank is illegal. NONE is  -1 so it is strictly weaker than
    every named ladder rung."""
    NONE = -1
    NETWORK_SERVING = 0
    DISTRIBUTED_EXECUTION = 1
    HGVAS = 2
    SOFTWARE_MANAGED = 3
    OWNERSHIP_VERSIONED = 4
    KERNEL_BOUNDARY = 5
    HARDWARE_UMA = 6


class ComputeVisibilityEvidence(str, Enum):
    """How device-compute visibility was established. READBACK is recorded
    so the validator can refuse it as proof of compute visibility."""
    UNDECLARED = "undeclared"
    READBACK = "readback"
    EXPLICIT_CONTRACT = "explicit_contract"
    DECLARED_ABSENT = "declared_absent"


@dataclass(frozen=True)
class DomainVisibility:
    """Two independent axes. A host that can read bytes back has not thereby
    proven the device can compute on them."""
    readback: bool
    device_compute: bool
    device_compute_evidence: ComputeVisibilityEvidence = ComputeVisibilityEvidence.UNDECLARED

    def to_dict(self) -> dict:
        return {
            "readback": self.readback,
            "device_compute": self.device_compute,
            "device_compute_evidence": self.device_compute_evidence.value,
            "readback_is_not_compute_visibility": True,
        }


@dataclass(frozen=True)
class Cost:
    """A planner estimate. `label` is always COST_MODEL; construction
    refuses any other string, including HARDWARE_MEASURED."""
    time_s: float
    nbytes: int
    bandwidth_gb_s: float
    latency_s: float
    label: str = COST_MODEL
    note: str = ""
    all_hops_physical: bool = False

    def __post_init__(self) -> None:
        if self.label != COST_MODEL:
            raise CostLabelError(
                f"cost label {self.label!r} is refused; only {COST_MODEL!r} is "
                f"legal in this layer (never emit {HARDWARE_MEASURED!r})")
        if self.nbytes < 0:
            raise SemanticTransportError(f"Cost.nbytes must be >= 0, got {self.nbytes}")
        if self.time_s < 0 or self.bandwidth_gb_s < 0 or self.latency_s < 0:
            raise SemanticTransportError("Cost time/bandwidth/latency must be >= 0")

    def to_dict(self) -> dict:
        return {
            "all_hops_physical": self.all_hops_physical,
            "bandwidth_gb_s": self.bandwidth_gb_s,
            "label": self.label,
            "latency_s": self.latency_s,
            "nbytes": self.nbytes,
            "note": self.note,
            "time_s": self.time_s,
        }


def cost_model(*, time_s: float, nbytes: int, bandwidth_gb_s: float,
               latency_s: float, note: str = "",
               all_hops_physical: bool = False) -> Cost:
    """The only public constructor. Always stamps COST_MODEL."""
    return Cost(
        time_s=time_s,
        nbytes=nbytes,
        bandwidth_gb_s=bandwidth_gb_s,
        latency_s=latency_s,
        label=COST_MODEL,
        note=note,
        all_hops_physical=all_hops_physical,
    )


def store_and_forward_cost(*, nbytes: int, bandwidth_gb_s: float,
                           latency_s: float, note: str = "") -> Cost:
    """nbytes/bandwidth + latency, the same single-hop formula
    fusion_planner.Topology.shortest_path and humf._transfer_cost use.
    Declared knobs in, COST_MODEL out."""
    if bandwidth_gb_s <= 0:
        raise SemanticTransportError(
            f"bandwidth_gb_s must be > 0 for a transfer cost, got {bandwidth_gb_s}")
    time_s = nbytes / (bandwidth_gb_s * 1e9) + latency_s
    return cost_model(
        time_s=time_s,
        nbytes=nbytes,
        bandwidth_gb_s=bandwidth_gb_s,
        latency_s=latency_s,
        note=note or "store-and-forward; COST_MODEL knob",
        all_hops_physical=False,
    )


@dataclass(frozen=True)
class ExecutionDomain:
    """One execution/memory domain. `kind` is device-neutral; `physical`
    is False for anything this machine does not actually have. Internal
    coherency is what the domain provides TO ITSELF, not what a link to
    another domain provides."""
    name: str
    kind: DomainKind
    physical: bool
    capacity_bytes: int | None = None
    visibility: DomainVisibility = DomainVisibility(
        readback=False,
        device_compute=False,
        device_compute_evidence=ComputeVisibilityEvidence.DECLARED_ABSENT,
    )
    internal_coherency: CoherencyAssumption = CoherencyAssumption.NONE

    def __post_init__(self) -> None:
        if not self.name:
            raise SemanticTransportError("ExecutionDomain.name must be non-empty")
        if (self.internal_coherency is CoherencyAssumption.HARDWARE_UMA
                and self.kind is not DomainKind.GPU_UMA):
            raise SemanticTransportError(
                f"domain {self.name!r} kind {self.kind.value} cannot declare "
                f"internal coherency HARDWARE_UMA; only GPU_UMA may")

    def to_dict(self) -> dict:
        return {
            "capacity_bytes": self.capacity_bytes,
            "internal_coherency": self.internal_coherency.name,
            "kind": self.kind.value,
            "name": self.name,
            "physical": self.physical,
            "visibility": self.visibility.to_dict(),
        }


@dataclass(frozen=True)
class SemanticTransportEdge:
    """Typed edge between two execution domains.

    `link_coherency` is what the link actually provides (NONE for a
    non-coherent interconnect). `coherency_assumption` is what the PLAN
    claims about that link. The validator refuses assumption > provided.
    Both default independently; a caller who wants a coherent plan over a
    non-coherent link has to say so, and then be refused.

    `ordering_assumption` is the same pattern for happens-before.
    None means honest: the plan assumes exactly the protocol it requested
    (`sync_requirement`). Setting a stronger assumption than the protocol
    provides is constructible and then refused -- a plan that assumes an
    ordering the transport does not guarantee is illegal.
    """
    source: str
    destination: str
    payload_semantics: PayloadSemantics
    ownership_transfer: OwnershipTransfer
    sync_requirement: SyncRequirement
    coherency_assumption: CoherencyAssumption
    link_coherency: CoherencyAssumption
    cost: Cost
    in_transit_transforms: tuple[str, ...] = ()
    object_id: str = ""
    organ_id: str = ""
    token_id: str = ""
    representation_id: str = ""
    ordering_assumption: OrderingGuarantee | None = None

    def __post_init__(self) -> None:
        if not self.source or not self.destination:
            raise SemanticTransportError("source and destination must be non-empty")
        if not isinstance(self.cost, Cost):
            raise SemanticTransportError("cost must be a Cost")
        if self.cost.label != COST_MODEL:
            raise CostLabelError(
                f"edge {self.source}->{self.destination} cost label "
                f"{self.cost.label!r} is not {COST_MODEL!r}")
        unknown = [t for t in self.in_transit_transforms if t not in IN_TRANSIT_TRANSFORMS]
        if unknown:
            raise SemanticTransportError(
                f"unknown in-transit transform(s) {unknown}; "
                f"known: {sorted(IN_TRANSIT_TRANSFORMS)}")

    @property
    def assumes_stronger_than_link(self) -> bool:
        return int(self.coherency_assumption) > int(self.link_coherency)

    @property
    def provided_ordering(self) -> OrderingGuarantee:
        """Happens-before this edge actually guarantees: the protocol it
        requested. NONE requested => nothing guaranteed."""
        return ordering_from_sync(self.sync_requirement)

    @property
    def effective_ordering_assumption(self) -> OrderingGuarantee:
        if self.ordering_assumption is None:
            return self.provided_ordering
        return self.ordering_assumption

    @property
    def assumes_unguaranteed_ordering(self) -> bool:
        return int(self.effective_ordering_assumption) > int(self.provided_ordering)

    @property
    def ownership_handoff_unguaranteed(self) -> bool:
        """Exclusive write-authority moved with no happens-before on a
        non-coherent link. KEEP and SHARE_READ do not move exclusive
        write, so they are not this bug. TRANSFER + FENCE is a software
        ordered handoff and is legal; TRANSFER + silent NONE is not."""
        if self.ownership_transfer is not OwnershipTransfer.TRANSFER:
            return False
        if self.link_coherency is not CoherencyAssumption.NONE:
            return False
        return self.provided_ordering is OrderingGuarantee.NONE

    def to_dict(self) -> dict:
        return {
            "coherency_assumption": self.coherency_assumption.name,
            "cost": self.cost.to_dict(),
            "destination": self.destination,
            "in_transit_transforms": list(self.in_transit_transforms),
            "link_coherency": self.link_coherency.name,
            "object_id": self.object_id,
            "ordering_assumption": self.effective_ordering_assumption.name,
            "organ_id": self.organ_id,
            "ownership_transfer": self.ownership_transfer.value,
            "payload_semantics": self.payload_semantics.value,
            "provided_ordering": self.provided_ordering.name,
            "representation_id": self.representation_id,
            "source": self.source,
            "sync_requirement": self.sync_requirement.value,
            "token_id": self.token_id,
        }


# Atlas / HWIR already own this string. Reusing it is the connection, not a
# second primitive under a new name.
ATLAS_PRIMITIVE = "SemanticTransportEdge"
HWIR_NODE_KIND = "dma-transport"
