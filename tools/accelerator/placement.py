"""Placement -- assign a computation/state/storage node to an execution domain.

Device-neutral: GPU_UMA, CPU, NPU and FPGA_HBM are all expressible as
DomainKind values. This module has no product branch. A backend binds a
product to a kind; a Placement only names the kind's domain.

RELATION TO fusion_planner.Placement. That dataclass is OBJECT-HOME
placement: which domain holds the authoritative copy of a SemanticObject,
and whether immutable organ-granularity copies may be replicated. It does
not carry resource requirements, a validity condition, or a node kind.
THIS module places NODES (a computation, a state machine, a storage
region). fusion_bridge composes both: node placements here, object-home
placements from fusion_planner.place_objects, typed edges from
semantic_transport. SemanticTransportEdge. There is one Placement concept
per job, not two competing implementations of the same job.

VALIDITY is recomputed by the Fusion Bridge validator, not trusted from
the stored flag. A Placement that claims resources_fit on a domain whose
capacity is smaller than resources.bytes is refused.

NOT IMPLEMENTED, named rather than left silent:
  - No capacity accounting beyond a single bytes-vs-capacity_bytes check.
    fusion_planner.place_objects already names this as unbuilt; this
    module does not build a second, more complete one.
  - No floorplanning. FPGA_HBM placements target HWIR node kinds; they
    do not emit a bitstream or a pblock.
  - No runtime binding. A Placement is a plan record.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from semantic_transport import DomainKind


class PlacementError(RuntimeError):
    """Base for every error this module raises."""


class NodeKind(str, Enum):
    COMPUTATION = "computation"
    STATE = "state"
    STORAGE = "storage"


# Workload-profile names owned by tools/odyssey/device_profiles.py
# (INTERACTIVE / MAXX). They are not device domains; they are optional
# hints on a Placement for a later chooser. This module does not interpret
# them beyond admitting the known names.
PROFILE_HINTS = frozenset({"INTERACTIVE", "MAXX"})


@dataclass(frozen=True)
class ResourceRequirements:
    """Generic resource ask. Product-specific budgets (on-chip RAM, DSP
    slices, shader cores, NPU tiles) belong in a backend -- HWIR
    DeviceBudget for FPGA_HBM, MachineGenome for the host GPU_UMA domain
    -- not here."""
    bytes: int = 0
    compute_slots: int = 0
    on_chip_bytes: int = 0

    def __post_init__(self) -> None:
        if self.bytes < 0 or self.compute_slots < 0 or self.on_chip_bytes < 0:
            raise PlacementError("resource requirements must be >= 0")

    def to_dict(self) -> dict:
        return {
            "bytes": self.bytes,
            "compute_slots": self.compute_slots,
            "on_chip_bytes": self.on_chip_bytes,
        }


@dataclass(frozen=True)
class ValidityCondition:
    """Stored snapshot of why this placement was considered legal at
    construction. The Fusion Bridge validator re-checks the live domains
    rather than trusting these flags."""
    domain_registered: bool = True
    resources_fit: bool = True
    kind_admitted: bool = True
    reason: str = ""

    def holds(self) -> bool:
        return self.domain_registered and self.resources_fit and self.kind_admitted

    def to_dict(self) -> dict:
        return {
            "domain_registered": self.domain_registered,
            "holds": self.holds(),
            "kind_admitted": self.kind_admitted,
            "reason": self.reason,
            "resources_fit": self.resources_fit,
        }


def resources_fit(req: ResourceRequirements, capacity_bytes: int | None) -> bool:
    """True when the domain has not declared a capacity, or the ask fits.
    A missing capacity is NOT treated as zero -- that would turn 'undeclared'
    into 'empty', which is the wrong failure direction for a declared-only
    FPGA_HBM domain."""
    if capacity_bytes is None:
        return True
    return req.bytes <= capacity_bytes


@dataclass(frozen=True)
class Placement:
    """One node, one domain. `replicas` is empty unless this STORAGE node
    was lifted from a fusion_planner object-home placement that legally
    replicated an immutable organ; computation and state never replicate
    here (state has a single owner; computation runs in one domain)."""
    node_id: str
    node_kind: NodeKind
    domain: str
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    validity: ValidityCondition = field(default_factory=ValidityCondition)
    profile_hint: str | None = None
    replicas: tuple[str, ...] = ()
    owner_domain: str | None = None

    def __post_init__(self) -> None:
        if not self.node_id:
            raise PlacementError("Placement.node_id must be non-empty")
        if not self.domain:
            raise PlacementError("Placement.domain must be non-empty")
        if self.profile_hint is not None and self.profile_hint not in PROFILE_HINTS:
            raise PlacementError(
                f"profile_hint {self.profile_hint!r} is not one of {sorted(PROFILE_HINTS)}; "
                f"device_profiles.py owns those names and this module does not invent others")
        if self.node_kind is NodeKind.STATE and not (self.owner_domain or self.domain):
            raise PlacementError(
                f"state node {self.node_id!r} has no owner domain")
        if self.node_kind is not NodeKind.STORAGE and self.replicas:
            raise PlacementError(
                f"{self.node_kind.value} node {self.node_id!r} cannot carry replicas; "
                f"replication is an object-home fact for STORAGE, from fusion_planner")

    @property
    def owner(self) -> str:
        """Authoritative domain for this node. State uses owner_domain when
        set; otherwise the placement domain is the owner."""
        return self.owner_domain or self.domain

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "node_id": self.node_id,
            "node_kind": self.node_kind.value,
            "owner_domain": self.owner,
            "profile_hint": self.profile_hint,
            "replicas": list(self.replicas),
            "resources": self.resources.to_dict(),
            "validity": self.validity.to_dict(),
        }


def kind_admits(_kind: DomainKind, _node_kind: NodeKind) -> bool:
    """Every domain kind may host every node kind. Restrictions are a
    backend's fact (an NPU island that cannot hold KV_STATE, an FPGA_HBM
    region that cannot run a particular organ) and do not belong in a
    generic `if kind == ...` here."""
    return True
