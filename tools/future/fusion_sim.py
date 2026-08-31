"""FUSION_SIMULATION — semantic heterogeneous simulation with no fake speed claim.

Models a Hawking execution spread across Apple Silicon, an FPGA card, a future
CUDA/DGX node, an eGPU and additional Macs — before any of those nodes exist.

This module extends concepts already on disk; it does not fork them:

  tools/accelerator/fusion_planner.py  placement, topology, move-vs-recompute,
                                       collectives. Uses KNOB numbers on
                                       non-Apple links and still emits cost_s.
  tools/accelerator/fusion_isa.py      14-command Fusion ISA
  tools/accelerator/fusion_wire.py     42-byte packet, no payload, no timing
  tools/accelerator/humf.py            MemoryClass policy, DeviceLost, quarantine
  tools/accelerator/machine_genome.py  INSTANCE MachineGenome producer
  tools/accelerator/cuda_runtime.py    C2M-T1 Metal stand-in; not a CUDA differential
  tools/accelerator/ccl.py             CUDA capability ledger (no NVIDIA hardware)
  hcli/agentos/fpga_preboard.py        FPGA TARGET_UNSELECTED / board ABSENT
  hcli/machine.py                      MachineGenome compatibility bag

THE HONESTY RULE this file exists to pin, and that fusion_planner.py does not:
every latency/bandwidth/compute input is a MEASURED Hawking record (cited by
receipts/ path) or UNKNOWN. simulate() returns a STRUCTURE plus
timing_decidable. When any critical-path input is UNKNOWN, timing_decidable is
False and no speedup figure is produced — not an estimate, not a range, not a
"roughly". Cost(kind="UNKNOWN", value=...) is unrepresentable.

FPGA is Accelerator / Physical Compiler / Fusion. It is not a civilization and
this module is not an FPGA backend.

    python3 tools/future/fusion_sim.py --selftest
    python3 tools/future/fusion_sim.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import heapq
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from tools.future._common import git

RECEIPT = "FUSION_SIMULATION.json"
SCHEMA = "hawking.future.fusion_sim.v1"

# fusion_wire.py HEADER_SIZE. A dispatch packet is 42 bytes of protocol, not a
# measured interconnect, and not an invented constant.
FUSION_WIRE_HEADER_BYTES = 42

# fusion_isa.py FusionOp names, cited not imported (Codex-owned, not materialized).
FUSION_OPS: tuple[str, ...] = (
    "ACQUIRE_READ", "ACQUIRE_WRITE", "RELEASE", "PREFETCH", "COPY",
    "MATERIALIZE", "INVALIDATE", "REDUCE", "SCATTER", "GATHER",
    "SUBMIT", "FENCE", "DIGEST", "EVICT",
)

# humf.py MEMORY_CLASS_POLICY — the load-bearing field is `mutable`.
IMMUTABLE_CLASSES = frozenset({
    "IMMUTABLE_WEIGHTS", "COMPILER_ARTIFACT", "EXPERT_CACHE",
})
MUTABLE_CLASSES = frozenset({
    "KV_STATE", "RECURRENT_STATE", "ACTIVATIONS", "ROUTING", "METADATA", "SCRATCH",
})

# fusion_planner.py TIE_BREAK_RANK: on an exact cost tie, avoid moving bytes.
TIE_BREAK_RANK = {
    "RECOMPUTE": 0,
    "MOVE_COMPUTE": 1,
    "WAIT": 2,
    "PREFETCH": 3,
    "REPACK": 4,
    "REPLICATE": 5,
    "MOVE_DATA": 6,
}

# Keys that would constitute a heterogeneous speedup figure. simulate() is
# structurally incapable of putting a number in any of these when the critical
# path is undecidable.
SPEEDUP_KEYS = frozenset({
    "speedup",
    "speedup_x",
    "heterogeneous_speedup",
    "relative_speedup",
    "speedup_vs_apple",
    "estimated_speedup",
    "speedup_range",
    "rough_speedup",
    "speedup_lower",
    "speedup_upper",
})

GENOME_RECEIPT = "receipts/headless/MACHINE_GENOME.json"
FPGA_PREBOARD_RECEIPT = "receipts/headless/HCLI_FPGA_PREBOARD.json"
FPGA_ORGAN_MAP = "receipts/headless/FLASH_NEXT_FPGA_ORGAN_MAP.json"
CUDA_CENSUS_RECEIPT = "receipts/headless/CUDA_CAPABILITY_CENSUS.json"
CUDA_LEDGER_RECEIPT = "receipts/headless/CUDA_CAPABILITY_LEDGER.json"
ANE_ATLAS = "receipts/headless/APPLE_ANE_ATLAS.json"
ANE_PROFILE = "receipts/headless/APPLE_ANE_DEVICE_PROFILE.json"


class FusionSimError(RuntimeError):
    """Structural error in a topology, placement, or query."""


class HonestyError(ValueError):
    """Raised when a caller tries to put a number where only UNKNOWN is honest."""


# --------------------------------------------------------------------------- Cost


@dataclass(frozen=True)
class Cost:
    """One latency, bandwidth, or compute input.

    MEASURED requires a numeric value AND a receipts/ citation. UNKNOWN refuses
    to carry a number — that is the unrepresentable state, not a convention.
    """
    kind: str
    value: float | None = None
    unit: str | None = None
    receipt: str | None = None
    field: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "MEASURED":
            if self.value is None or not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
                raise HonestyError("MEASURED cost requires a numeric value")
            if not self.receipt or not str(self.receipt).startswith("receipts/"):
                raise HonestyError(
                    "MEASURED cost must cite a Hawking receipts/ path, not a datasheet "
                    "and not an invented knob"
                )
        elif self.kind == "UNKNOWN":
            if self.value is not None:
                raise HonestyError(
                    "UNKNOWN cost is unrepresentable with a numeric value; "
                    f"got {self.value!r}"
                )
        else:
            raise HonestyError(f"cost kind must be MEASURED or UNKNOWN, not {self.kind!r}")

    @property
    def known(self) -> bool:
        return self.kind == "MEASURED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "unit": self.unit,
            "receipt": self.receipt,
            "field": self.field,
            "reason": self.reason,
        }


def unknown(reason: str) -> Cost:
    return Cost(kind="UNKNOWN", reason=reason)


def measured(value: float, receipt: str, field: str, unit: str | None = None) -> Cost:
    return Cost(kind="MEASURED", value=float(value), receipt=receipt, field=field, unit=unit)


UNKNOWN_COMPUTE = unknown(
    "no PROTECTED_ABSOLUTE compute measurement; sidecar has no GPU lease and "
    "does not invent token_ns"
)
UNKNOWN_INTERCONNECT = unknown(
    "no measured interconnect to an absent node; a knob is not a measurement"
)


# --------------------------------------------------------------------------- topology


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    present: bool
    missing_dependency: str | None
    genome_receipt: str | None = None
    identity: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if self.present and self.missing_dependency is not None:
            raise FusionSimError(f"{self.id}: present node must not name a missing dependency")
        if not self.present and not self.missing_dependency:
            raise FusionSimError(f"{self.id}: absent node must name the exact missing dependency")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "present": self.present,
            "missing_dependency": self.missing_dependency,
            "genome_receipt": self.genome_receipt,
            "identity": dict(self.identity),
            "note": self.note,
        }


@dataclass(frozen=True)
class Link:
    a: str
    b: str
    bandwidth: Cost
    latency: Cost
    note: str = ""

    @property
    def costs_known(self) -> bool:
        return self.bandwidth.known and self.latency.known

    def other(self, name: str) -> str:
        if name == self.a:
            return self.b
        if name == self.b:
            return self.a
        raise FusionSimError(f"{name!r} is not an endpoint of {self.a}-{self.b}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a,
            "b": self.b,
            "bandwidth": self.bandwidth.to_dict(),
            "latency": self.latency.to_dict(),
            "note": self.note,
            "costs_known": self.costs_known,
        }


@dataclass(frozen=True)
class Route:
    path: tuple[str, ...]
    hops: tuple[Link, ...]
    nbytes: int
    timing_decidable: bool
    unknown_inputs: tuple[str, ...]
    total_time_s: float | None
    how_chosen: str
    alpha_s: float | None = None
    beta_s_per_byte: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": list(self.path),
            "hops": [f"{h.a}-{h.b}" for h in self.hops],
            "nbytes": self.nbytes,
            "timing_decidable": self.timing_decidable,
            "unknown_inputs": list(self.unknown_inputs),
            "total_time_s": self.total_time_s,
            "how_chosen": self.how_chosen,
            "alpha_s": self.alpha_s,
            "beta_s_per_byte": self.beta_s_per_byte,
            "evidence_class": "STATIC_ONLY",
        }


def _alpha_beta(hops: Sequence[Link]) -> tuple[float, float]:
    alpha = 0.0
    beta = 0.0
    for h in hops:
        assert h.latency.value is not None and h.bandwidth.value is not None
        alpha += h.latency.value
        beta += 1.0 / (h.bandwidth.value * 1e9)
    return alpha, beta


class Topology:
    """Undirected graph of Nodes and Links. Path existence is structure.
    Path COST is only computed when every hop's bandwidth and latency are MEASURED.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._links: dict[frozenset[str], Link] = {}
        self._adj: dict[str, list[str]] = {}

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes[k] for k in sorted(self._nodes))

    @property
    def links(self) -> tuple[Link, ...]:
        return tuple(self._links[k] for k in sorted(self._links, key=lambda s: tuple(sorted(s))))

    def add_node(self, node: Node) -> None:
        if node.id in self._nodes:
            raise FusionSimError(f"domain {node.id!r} already exists")
        self._nodes[node.id] = node
        self._adj[node.id] = []

    def add_link(self, link: Link) -> None:
        for d in (link.a, link.b):
            if d not in self._nodes:
                raise FusionSimError(f"cannot link {d!r}: not a domain in this topology")
        key = frozenset((link.a, link.b))
        if key in self._links:
            raise FusionSimError(f"a link between {link.a!r} and {link.b!r} already exists")
        self._links[key] = link
        self._adj[link.a].append(link.b)
        self._adj[link.b].append(link.a)

    def node(self, name: str) -> Node:
        if name not in self._nodes:
            raise FusionSimError(f"{name!r} is not a domain in this topology")
        return self._nodes[name]

    def link(self, a: str, b: str) -> Link | None:
        return self._links.get(frozenset((a, b)))

    def neighbors(self, name: str) -> list[str]:
        return list(self._adj.get(name, ()))

    def shortest_path(self, src: str, dst: str, nbytes: int) -> Route:
        for d in (src, dst):
            if d not in self._nodes:
                raise FusionSimError(f"{d!r} is not a domain in this topology")
        if src == dst:
            return Route((src,), (), nbytes, True, (), 0.0, "ALREADY_RESIDENT", 0.0, 0.0)

        measured_route = self._dijkstra_measured(src, dst, nbytes)
        if measured_route is not None:
            return measured_route
        return self._bfs_structural(src, dst, nbytes)

    def _hop_time_s(self, lk: Link, nbytes: int) -> float:
        # Caller has already checked costs_known.
        bw = lk.bandwidth.value
        lat = lk.latency.value
        assert bw is not None and lat is not None
        if bw <= 0:
            raise FusionSimError(f"MEASURED bandwidth on {lk.a}-{lk.b} is not positive")
        # unit of bandwidth.value is GB/s when unit is GB/s; require that.
        unit = lk.bandwidth.unit or "GB/s"
        if unit != "GB/s":
            raise FusionSimError(f"bandwidth unit {unit!r} is not GB/s; refusing to convert")
        return nbytes / (bw * 1e9) + lat

    def _dijkstra_measured(self, src: str, dst: str, nbytes: int) -> Route | None:
        dist: dict[str, float] = {src: 0.0}
        prev: dict[str, tuple[str, Link]] = {}
        heap: list[tuple[float, str]] = [(0.0, src)]
        visited: set[str] = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == dst:
                break
            for v in sorted(self.neighbors(u)):
                lk = self.link(u, v)
                if lk is None or not lk.costs_known:
                    continue
                nd = d + self._hop_time_s(lk, nbytes)
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    prev[v] = (u, lk)
                    heapq.heappush(heap, (nd, v))
        if dst not in dist:
            return None
        hops, path = _unwind(prev, src, dst)
        alpha, beta = _alpha_beta(hops)
        return Route(tuple(path), tuple(hops), nbytes, True, (), dist[dst],
                     "DIJKSTRA_ON_MEASURED_HOPS", alpha, beta)

    def _bfs_structural(self, src: str, dst: str, nbytes: int) -> Route:
        prev: dict[str, tuple[str, Link]] = {}
        q = [src]
        seen = {src}
        while q:
            u = q.pop(0)
            if u == dst:
                break
            for v in sorted(self.neighbors(u)):
                if v in seen:
                    continue
                lk = self.link(u, v)
                if lk is None:
                    continue
                seen.add(v)
                prev[v] = (u, lk)
                q.append(v)
        if dst not in prev and dst != src:
            raise FusionSimError(f"no route from {src!r} to {dst!r} in this topology")
        hops, path = _unwind(prev, src, dst)
        unknowns: list[str] = []
        for h in hops:
            if not h.bandwidth.known:
                unknowns.append(f"{h.a}-{h.b}.bandwidth:{h.bandwidth.reason}")
            if not h.latency.known:
                unknowns.append(f"{h.a}-{h.b}.latency:{h.latency.reason}")
        return Route(tuple(path), tuple(hops), nbytes, False, tuple(unknowns), None,
                     "FEWEST_HOPS_COSTS_UNKNOWN", None, None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "links": [lk.to_dict() for lk in self.links],
        }


def _unwind(prev: dict[str, tuple[str, Link]], src: str, dst: str) -> tuple[list[Link], list[str]]:
    hops: list[Link] = []
    path: list[str] = [dst]
    cur = dst
    while cur != src:
        u, lk = prev[cur]
        hops.append(lk)
        path.append(u)
        cur = u
    path.reverse()
    hops.reverse()
    return hops, path


# --------------------------------------------------------------------------- placement / ownership


class Granularity(str, Enum):
    """Cited from fusion_planner.py. TENSOR is not eagerly replicated."""
    MODEL = "MODEL"
    ORGAN = "ORGAN"
    LAYER_GROUP = "LAYER_GROUP"
    EXPERT_GROUP = "EXPERT_GROUP"
    TENSOR = "TENSOR"


@dataclass(frozen=True)
class SemanticObject:
    identity: str
    memory_class: str
    granularity: Granularity
    nbytes: int
    home_hint: str | None = None
    consumers: tuple[str, ...] = ()
    recompute_legal: bool = False
    checkpointed: bool = False
    representation: str = "metal_f16"


@dataclass(frozen=True)
class Placement:
    identity: str
    home: str
    replicas: tuple[str, ...]
    real_replicas: tuple[str, ...]
    hypothetical_replicas: tuple[str, ...]
    reason: str
    owner: str
    mutable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "home": self.home,
            "replicas": list(self.replicas),
            "real_replicas": list(self.real_replicas),
            "hypothetical_replicas": list(self.hypothetical_replicas),
            "reason": self.reason,
            "owner": self.owner,
            "mutable": self.mutable,
        }


def _mutable(memory_class: str) -> bool:
    if memory_class in MUTABLE_CLASSES:
        return True
    if memory_class in IMMUTABLE_CLASSES:
        return False
    raise FusionSimError(f"unknown memory_class {memory_class!r}")


def place_objects(topo: Topology, objects: Sequence[SemanticObject]) -> dict[str, Placement]:
    """Extend fusion_planner.place_objects: replicas on absent nodes are hypothetical."""
    if not topo.nodes:
        raise FusionSimError("empty topology has nowhere to place anything")
    default_home = topo.nodes[0].id
    out: dict[str, Placement] = {}
    for obj in objects:
        home = obj.home_hint or default_home
        topo.node(home)
        mut = _mutable(obj.memory_class)
        if mut:
            out[obj.identity] = Placement(
                obj.identity, home, (), (), (),
                reason=(f"{obj.memory_class} is mutable per humf.MEMORY_CLASS_POLICY; "
                        f"a single authoritative owner ({home}) is required and "
                        f"replication is never offered"),
                owner=home, mutable=True,
            )
            continue
        remote = tuple(sorted({c for c in obj.consumers if c != home}))
        for c in remote:
            topo.node(c)
        if not remote:
            out[obj.identity] = Placement(
                obj.identity, home, (), (), (),
                reason=f"{obj.memory_class} is immutable but has no remote consumer; "
                       f"single copy at {home}",
                owner=home, mutable=False,
            )
            continue
        if obj.granularity is Granularity.TENSOR:
            out[obj.identity] = Placement(
                obj.identity, home, (), (), (),
                reason=(f"TENSOR-granularity immutable object with remote consumers "
                        f"{list(remote)}; whole-organ placement is preferred over "
                        f"fine-grained ping-pong, so this stays single-copy at {home}"),
                owner=home, mutable=False,
            )
            continue
        real = tuple(c for c in remote if topo.node(c).present)
        hypo = tuple(c for c in remote if not topo.node(c).present)
        out[obj.identity] = Placement(
            obj.identity, home, remote, real, hypo,
            reason=(f"{obj.granularity.value}-granularity {obj.memory_class} is "
                    f"immutable; replicated into every remote consumer {list(remote)}. "
                    f"Replicas on absent nodes {list(hypo)} are HYPOTHETICAL and "
                    f"cannot satisfy recovery."),
            owner=home, mutable=False,
        )
    return out


# --------------------------------------------------------------------------- move vs recompute


@dataclass(frozen=True)
class DependencyQuery:
    identity: str
    home_domain: str
    need_domain: str
    nbytes: int
    memory_class: str
    recompute_cost: Cost = UNKNOWN_COMPUTE
    output_bytes: int = 0
    recompute_legal: bool = False
    in_flight: bool = False
    overlap_legal: bool = False


def plan_dependency(topo: Topology, q: DependencyQuery) -> dict[str, Any]:
    """MOVE_DATA / MOVE_COMPUTE / RECOMPUTE / REPLICATE as STRUCTURE.

    Cheapest-wins only when every option that is offered has a known cost.
    Otherwise the choice is UNDECIDABLE — the thing a naive heterogeneous
    plan gets wrong by plugging in a knob.
    """
    if q.home_domain == q.need_domain:
        return {
            "identity": q.identity,
            "home_domain": q.home_domain,
            "need_domain": q.need_domain,
            "action": "ALREADY_RESIDENT",
            "choice_decidable": True,
            "options": [{
                "action": "ALREADY_RESIDENT",
                "choice_decidable": True,
                "cost_s": 0.0,
                "unknown_inputs": [],
                "detail": "no wire: home == need; structural zero, not a measurement",
            }],
            "reason": f"{q.identity} is already in {q.need_domain}",
            "route": None,
        }

    move_route = topo.shortest_path(q.home_domain, q.need_domain, q.nbytes)
    dispatch_route = topo.shortest_path(q.need_domain, q.home_domain, FUSION_WIRE_HEADER_BYTES)
    result_route = (
        topo.shortest_path(q.home_domain, q.need_domain, q.output_bytes)
        if q.output_bytes else None
    )

    options: list[dict[str, Any]] = []

    def _opt(action: str, decidable: bool, cost_s: float | None, unknowns: tuple[str, ...],
             detail: str) -> None:
        options.append({
            "action": action,
            "choice_decidable": decidable,
            "cost_s": cost_s if decidable else None,
            "unknown_inputs": list(unknowns),
            "detail": detail,
        })

    _opt("MOVE_DATA", move_route.timing_decidable, move_route.total_time_s,
         move_route.unknown_inputs,
         f"transfer {q.nbytes}B {q.home_domain} -> {q.need_domain} via "
         f"{'->'.join(move_route.path)}")

    if not _mutable(q.memory_class):
        _opt("REPLICATE", move_route.timing_decidable, move_route.total_time_s,
             move_route.unknown_inputs,
             f"place a persistent replica of {q.identity} in {q.need_domain} "
             f"(legal: {q.memory_class} is immutable)")

    mc_unknown = list(dispatch_route.unknown_inputs)
    mc_dec = dispatch_route.timing_decidable
    mc_cost = dispatch_route.total_time_s
    if result_route is not None:
        mc_unknown.extend(result_route.unknown_inputs)
        mc_dec = mc_dec and result_route.timing_decidable
        if mc_dec and mc_cost is not None and result_route.total_time_s is not None:
            mc_cost = mc_cost + result_route.total_time_s
        else:
            mc_cost = None
            mc_dec = False
    _opt("MOVE_COMPUTE", mc_dec, mc_cost, tuple(mc_unknown),
         f"run the consumer in {q.home_domain} instead of moving {q.nbytes}B: "
         f"send a {FUSION_WIRE_HEADER_BYTES}B fusion_wire dispatch and return "
         f"{q.output_bytes}B of result")

    if q.recompute_legal:
        _opt("RECOMPUTE", q.recompute_cost.known,
             q.recompute_cost.value if q.recompute_cost.known else None,
             () if q.recompute_cost.known else (q.recompute_cost.reason or "recompute UNKNOWN",),
             f"recompute {q.identity} locally in {q.need_domain} instead of transferring it")

    if q.in_flight:
        _opt("WAIT", False, None, ("in-flight remaining time is UNKNOWN without a live transport",),
             f"{q.identity} is already in flight toward {q.need_domain}; wait vs new transfer "
             f"is UNDECIDABLE without a remaining-time measurement")

    if q.overlap_legal:
        _opt("PREFETCH", False, None, ("overlap window duration is UNKNOWN without hardware",),
             "start the transfer now; how much hides behind other work is not decidable")

    decidable_opts = [o for o in options if o["choice_decidable"] and o["cost_s"] is not None]
    if decidable_opts and len(decidable_opts) == len(options):
        best = min(decidable_opts, key=lambda o: (o["cost_s"], TIE_BREAK_RANK.get(o["action"], 99)))
        action = best["action"]
        choice_decidable = True
        reason = f"cheapest known option is {action}"
    else:
        action = "UNDECIDABLE"
        choice_decidable = False
        reason = (
            "at least one offered option has an UNKNOWN cost on the critical path; "
            "refusing to pick a winner from knobs. This is the move-vs-recompute "
            "trap a naive heterogeneous plan hits."
        )
    return {
        "identity": q.identity,
        "home_domain": q.home_domain,
        "need_domain": q.need_domain,
        "action": action,
        "choice_decidable": choice_decidable,
        "options": options,
        "reason": reason,
        "route": move_route.to_dict(),
    }


# --------------------------------------------------------------------------- collectives


class CollectiveOp(str, Enum):
    ALLREDUCE = "ALLREDUCE"
    BROADCAST = "BROADCAST"
    ALLGATHER = "ALLGATHER"
    REDUCE_SCATTER = "REDUCE_SCATTER"


@dataclass(frozen=True)
class CollectiveSpec:
    op: CollectiveOp
    domains: tuple[str, ...]
    message_bytes: int


def plan_collective(topo: Topology, spec: CollectiveSpec) -> dict[str, Any]:
    """Ring vs tree is a COST choice. With UNKNOWN costs it is UNDECIDABLE.
    At <= 2 participants the algorithms collapse to DIRECT with no cost needed.
    """
    domains = tuple(spec.domains)
    if len(domains) < 2:
        raise FusionSimError("a collective needs at least 2 participants")
    if len(set(domains)) != len(domains):
        raise FusionSimError(f"duplicate participant in {domains!r}")
    for d in domains:
        topo.node(d)
    p = len(domains)
    hop_unknowns: list[str] = []
    hop_known = True
    structural_hops: list[list[str]] = []
    for i in range(p):
        a, b = domains[i], domains[(i + 1) % p]
        route = topo.shortest_path(a, b, spec.message_bytes)
        structural_hops.append(list(route.path))
        if not route.timing_decidable:
            hop_known = False
            hop_unknowns.extend(route.unknown_inputs)
    if p <= 2:
        cost_s = None
        if hop_known:
            cost_s = topo.shortest_path(domains[0], domains[1], spec.message_bytes).total_time_s
        reason = (
            f"only {p} participants; ring and tree collapse to the same direct "
            f"exchange, so choosing between them is not a real distinction"
        )
        if not hop_known:
            reason += " — transfer time itself is still UNKNOWN"
        return {
            "op": spec.op.value,
            "domains": list(domains),
            "message_bytes": spec.message_bytes,
            "algorithm": "DIRECT",
            "algorithm_structural": "DIRECT",
            "choice_decidable": hop_known,
            "unknown_inputs": hop_unknowns,
            "structural_ring_hops": structural_hops,
            "reason": reason,
            "cost_s": cost_s,
        }
    algorithm = "UNDECIDABLE"
    reason = (
        f"p={p}: RING ((p-1) hops, (p-1)/p bandwidth-share) vs TREE "
        f"(ceil(log2 p) hops) is a cost comparison. Critical-path interconnect "
        f"costs are UNKNOWN, so the algorithm is not chosen. fusion_planner.py "
        f"would pick from knobs; this module refuses."
    )
    if hop_known:
        # fusion_planner._group_alpha_beta: worst ring-adjacent hop, independently
        # by latency (alpha) and per-byte time (beta). Path for the pair is the
        # zero-byte (latency) path; nbytes does not invent a second topology.
        worst_alpha = 0.0
        worst_beta = 0.0
        for i in range(p):
            a, b = domains[i], domains[(i + 1) % p]
            route0 = topo.shortest_path(a, b, 0)
            worst_alpha = max(worst_alpha, route0.alpha_s or 0.0)
            worst_beta = max(worst_beta, route0.beta_s_per_byte or 0.0)
        L = max(1, math.ceil(math.log2(p)))
        n = spec.message_bytes
        ring_cost = (p - 1) * worst_alpha + (p - 1) / p * n * worst_beta
        tree_cost = L * worst_alpha + L * n * worst_beta
        if ring_cost <= tree_cost:
            algorithm = "RING"
            reason = (
                f"RING chosen on MEASURED hops: ring_cost={ring_cost} <= "
                f"tree_cost={tree_cost} (fusion_planner alpha/beta pair)"
            )
        else:
            algorithm = "TREE"
            reason = (
                f"TREE chosen on MEASURED hops: tree_cost={tree_cost} < "
                f"ring_cost={ring_cost} (fusion_planner alpha/beta pair)"
            )
        return {
            "op": spec.op.value,
            "domains": list(domains),
            "message_bytes": spec.message_bytes,
            "algorithm": algorithm,
            "algorithm_structural": "RING_OR_TREE",
            "choice_decidable": True,
            "unknown_inputs": [],
            "structural_ring_hops": structural_hops,
            "reason": reason,
            "cost_s": min(ring_cost, tree_cost),
            "evidence_class": "STATIC_ONLY",
            "not_a_protected_measurement": True,
        }
    return {
        "op": spec.op.value,
        "domains": list(domains),
        "message_bytes": spec.message_bytes,
        "algorithm": "UNDECIDABLE",
        "algorithm_structural": "RING_OR_TREE",
        "choice_decidable": False,
        "unknown_inputs": hop_unknowns,
        "structural_ring_hops": structural_hops,
        "reason": reason,
        "cost_s": None,
    }


# --------------------------------------------------------------------------- CUDA seam


LOWERING_CONTRACT: dict[str, Any] = {
    "ir": "fusion_isa.FusionOp",
    "source": "tools/accelerator/fusion_isa.py",
    "wire": "tools/accelerator/fusion_wire.py",
    "ops": list(FUSION_OPS),
    "targets": ["metal", "cuda", "fpga_hwir"],
    "methods": [
        "lower(op, representation) -> kernel_ref",
        "abi_layout(op)",
        "required_objects(op)",
    ],
    "forbidden": [
        "local CUDA performance claim on Apple hardware",
        "tps / token_ns / gpu_ns / bandwidth_gbps in lowering output",
        "building an FPGA backend or an FPGA civilization",
    ],
    "note": (
        "Lowering is a contract over FusionOp, not a timing model. "
        "tools/accelerator/cuda_runtime.py executes a CUDA HOST subset on Metal "
        "and already records is_a_cuda_differential=False because no NVIDIA "
        "hardware exists on this machine."
    ),
}

METAL_CUDA_DIFFERENTIAL_SCHEMA: dict[str, Any] = {
    "schema": "hawking.future.fusion_sim.metal_cuda_differential.v1",
    "required_fields": [
        "workload_id",
        "metal_receipt",
        "cuda_receipt",
        "machine_metal",
        "machine_cuda",
        "bench_state_metal",
        "bench_state_cuda",
        "measurement_class_metal",
        "measurement_class_cuda",
    ],
    "promotion_rule": (
        "both sides must be PROTECTED_ABSOLUTE taken under a real protected GPU "
        "lease on the named machine. DIAGNOSTIC_RELATIVE never promotes. "
        "STATIC_ONLY never promotes."
    ),
    "local_cuda_on_apple": "FORBIDDEN",
    "why_local_cuda_on_apple_is_forbidden": (
        f"{CUDA_CENSUS_RECEIPT} identities.transport.status=ABSENT: "
        "'no NVIDIA hardware exists on this machine'. A CUDA number collected "
        "on Apple Silicon is not CUDA."
    ),
    "hardware_fields_must_be_null_until_both_protected": list(
        sorted({"tps", "token_ns", "gpu_ns", "bandwidth_gbps", "wall_ns", "dispatch_ns"})
    ),
}


def validate_metal_cuda_differential(rec: dict[str, Any]) -> None:
    missing = [f for f in METAL_CUDA_DIFFERENTIAL_SCHEMA["required_fields"] if f not in rec]
    if missing:
        raise FusionSimError(f"differential record missing {missing}")
    cuda_machine = rec.get("machine_cuda") or {}
    cuda_kind = str(cuda_machine.get("kind") or cuda_machine.get("soc") or "")
    if rec.get("local_cuda_on_apple") is True:
        raise HonestyError("local CUDA performance claim on Apple hardware is FORBIDDEN")
    if "Apple" in cuda_kind or cuda_kind in {"APPLE", "APPLE_SILICON"}:
        raise HonestyError(
            "no local CUDA performance claim on Apple hardware: machine_cuda names Apple"
        )
    for key in METAL_CUDA_DIFFERENTIAL_SCHEMA["hardware_fields_must_be_null_until_both_protected"]:
        for side in ("metal", "cuda", ""):
            k = f"{side}_{key}" if side else key
            if k in rec and isinstance(rec[k], (int, float)):
                raise HonestyError(
                    f"{k}={rec[k]!r} is a hardware number; the differential schema "
                    f"refuses numbers until both sides are PROTECTED_ABSOLUTE"
                )
    if rec.get("bench_state_cuda") == "UNKNOWN" and _has_numeric_speedup(rec):
        raise HonestyError("CUDA bench state UNKNOWN cannot carry a speedup figure")


def lower_fusion_op(op: str, target: str) -> dict[str, Any]:
    if op not in FUSION_OPS:
        raise FusionSimError(f"{op!r} is not a FusionOp (see tools/accelerator/fusion_isa.py)")
    if target not in LOWERING_CONTRACT["targets"]:
        raise FusionSimError(f"target {target!r} is not in {LOWERING_CONTRACT['targets']}")
    return {
        "op": op,
        "target": target,
        "kernel_ref": f"{target}:{op.lower()}",
        "abi": "fusion_wire.v1",
        "performance": None,
        "timing_decidable": False,
        "note": "lowering is backend-neutral; it does not time anything",
    }


# --------------------------------------------------------------------------- work / overlap / conversion


@dataclass(frozen=True)
class WorkItem:
    id: str
    kind: str  # COMPUTE | CONVERT
    assigned_node: str
    bytes_in: int
    bytes_out: int
    depends_on: tuple[str, ...]
    memory_class: str
    representation: str
    recompute_legal: bool = False
    compute_cost: Cost = UNKNOWN_COMPUTE
    converts_from: str | None = None
    converts_to: str | None = None


def overlap_structure(work: Sequence[WorkItem]) -> list[dict[str, Any]]:
    """Pairs that MAY overlap: different nodes, no happens-before. No time saved."""
    by_id = {w.id: w for w in work}
    out: list[dict[str, Any]] = []
    ids = [w.id for w in work]
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            wa, wb = by_id[a], by_id[b]
            if wa.assigned_node == wb.assigned_node:
                continue
            if a in wb.depends_on or b in wa.depends_on:
                continue
            out.append({
                "a": a,
                "b": b,
                "may_overlap": True,
                "reason": (
                    f"{a}@{wa.assigned_node} and {b}@{wb.assigned_node} share no "
                    f"happens-before and no node; overlap DURATION is UNKNOWN"
                ),
                "hidden_time_s": None,
            })
    return out


# --------------------------------------------------------------------------- fault / recovery


def vanish_node(
    topo: Topology,
    placements: dict[str, Placement],
    objects: Sequence[SemanticObject],
    work: Sequence[WorkItem],
    collectives: Sequence[CollectiveSpec],
    vanished: str,
    inflight_edges: Sequence[tuple[str, str, str]] = (),
) -> dict[str, Any]:
    """A node vanishing mid-execution must have a defined outcome.

    Extends humf.Humf.device_lost: EVERY copy on the vanished domain is gone.
    Hypothetical replicas on already-absent nodes cannot recover anything.
    """
    node = topo.node(vanished)
    if not node.present:
        return {
            "vanished": vanished,
            "outcome": "ALREADY_ABSENT",
            "objects": [],
            "inflight": [],
            "collectives": [],
            "work": [],
            "reason": (
                f"{vanished} was never present ({node.missing_dependency}); "
                f"vanishing it is a no-op with a defined outcome"
            ),
        }

    obj_by_id = {o.identity: o for o in objects}
    object_outcomes: list[dict[str, Any]] = []
    for ident, p in sorted(placements.items()):
        obj = obj_by_id.get(ident)
        replicas_real = set(p.real_replicas)
        if p.home == vanished:
            replicas_real.discard(vanished)
            if p.mutable:
                if obj and obj.checkpointed:
                    fate = "CHECKPOINT_RESTORE"
                    detail = "mutable owner vanished; a checkpoint exists"
                elif obj and obj.recompute_legal:
                    fate = "RECOMPUTE_ON_SURVIVOR"
                    detail = "mutable owner vanished; object declares recompute_legal"
                else:
                    fate = "DATA_LOSS"
                    detail = (
                        "mutable owner vanished; sole live state was there. "
                        "humf.device_lost names this DATA LOSS, not degraded replication"
                    )
            else:
                survivors = [r for r in p.real_replicas if r != vanished]
                if survivors:
                    fate = "RECOVERED_FROM_REPLICA"
                    detail = f"immutable; real replica remains at {survivors}"
                elif obj and obj.recompute_legal:
                    fate = "RECOMPUTE_ON_SURVIVOR"
                    detail = "immutable with no real replica; recompute_legal"
                else:
                    fate = "DATA_LOSS"
                    detail = (
                        "immutable home vanished; remaining replicas were HYPOTHETICAL "
                        f"(on absent nodes {list(p.hypothetical_replicas)}) and cannot recover"
                    )
        elif vanished in p.real_replicas:
            fate = "REPLICA_INVALIDATED"
            detail = f"a real replica at {vanished} is gone; home {p.home} still holds authority"
        elif vanished in p.hypothetical_replicas:
            fate = "HYPOTHETICAL_REPLICA_DROPPED"
            detail = "dropping a replica that was never physically there"
        else:
            continue
        object_outcomes.append({
            "identity": ident,
            "outcome": fate,
            "detail": detail,
            "mutable": p.mutable,
            "home": p.home,
        })

    inflight_outcomes = []
    for src, dst, ident in inflight_edges:
        if vanished in (src, dst):
            inflight_outcomes.append({
                "identity": ident,
                "src": src,
                "dst": dst,
                "outcome": "ABORT",
                "detail": f"in-flight transport {src}->{dst} involved vanished node {vanished}",
            })

    coll_outcomes = []
    for c in collectives:
        if vanished in c.domains:
            coll_outcomes.append({
                "op": c.op.value,
                "domains": list(c.domains),
                "outcome": "ABORT_COLLECTIVE",
                "detail": (
                    f"{vanished} vanished mid-{c.op.value}; remaining participants FENCE. "
                    f"No partial result is authoritative."
                ),
            })

    work_outcomes = []
    for w in work:
        if w.assigned_node == vanished:
            work_outcomes.append({
                "id": w.id,
                "outcome": "ABORT_WORK",
                "detail": f"work {w.id} was assigned to vanished node {vanished}",
            })

    return {
        "vanished": vanished,
        "outcome": "DEFINED",
        "quarantine": True,
        "quarantine_note": (
            "Cited from humf.Humf.device_lost: the domain is not trustworthy for "
            "the NEXT transfer either. Release is explicit, never automatic."
        ),
        "objects": object_outcomes,
        "inflight": inflight_outcomes,
        "collectives": coll_outcomes,
        "work": work_outcomes,
        "means": (
            "a lost device invalidates every copy it held; anything whose live or "
            "only state was there is DATA_LOSS, not degraded replication"
        ),
    }


# --------------------------------------------------------------------------- simulate


def _numeric_speedup_keys(node: Any, path: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in SPEEDUP_KEYS and isinstance(value, (int, float)):
                found.append(here)
            found.extend(_numeric_speedup_keys(value, here))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found.extend(_numeric_speedup_keys(value, f"{path}[{i}]"))
    return found


def _has_numeric_speedup(node: Any) -> bool:
    return bool(_numeric_speedup_keys(node))


def emit_speedup(hetero_time_s: float | None, baseline_time_s: float | None,
                 *, timing_decidable: bool) -> float:
    """The only door a speedup number can walk through. Tests watch it refuse."""
    if not timing_decidable:
        raise HonestyError(
            "speedup requested but timing_decidable is False; a heterogeneous "
            "speedup from UNKNOWN inputs is unrepresentable"
        )
    if hetero_time_s is None or baseline_time_s is None:
        raise HonestyError("speedup requested without both decidable timings")
    if hetero_time_s <= 0:
        raise HonestyError("speedup requested with non-positive hetero timing")
    return float(baseline_time_s) / float(hetero_time_s)


def simulate(
    topo: Topology,
    objects: Sequence[SemanticObject],
    work: Sequence[WorkItem],
    collectives: Sequence[CollectiveSpec] = (),
    *,
    vanish: str | None = None,
    inflight_edges: Sequence[tuple[str, str, str]] = (),
    baseline_time_s: float | None = None,
) -> dict[str, Any]:
    """Return STRUCTURE + timing_decidable. Never a speedup from UNKNOWN inputs."""
    placements = place_objects(topo, objects)
    transports: list[dict[str, Any]] = []
    conversions: list[dict[str, Any]] = []
    move_plans: list[dict[str, Any]] = []
    unknowns: list[str] = []

    not_executable: list[str] = []
    for n in topo.nodes:
        if not n.present:
            assigned = [w.id for w in work if w.assigned_node == n.id]
            if assigned:
                not_executable.append(
                    f"work {assigned} assigned to absent node {n.id} "
                    f"({n.missing_dependency})"
                )

    for w in work:
        topo.node(w.assigned_node)
        if w.kind == "CONVERT":
            conversions.append({
                "id": w.id,
                "from": w.converts_from,
                "to": w.converts_to,
                "node": w.assigned_node,
                "cost": w.compute_cost.to_dict(),
                "timing_decidable": w.compute_cost.known,
            })
            if not w.compute_cost.known:
                unknowns.append(f"convert.{w.id}:{w.compute_cost.reason}")
            continue
        if not w.compute_cost.known:
            unknowns.append(f"compute.{w.id}:{w.compute_cost.reason}")

    for obj in objects:
        p = placements[obj.identity]
        need_nodes = sorted({c for c in obj.consumers if c != p.home})
        for need in need_nodes:
            recompute_cost = UNKNOWN_COMPUTE
            if obj.recompute_legal:
                dest_work = next(
                    (w for w in work
                     if w.assigned_node == need and w.recompute_legal and w.compute_cost.known),
                    None,
                )
                if dest_work is not None:
                    recompute_cost = dest_work.compute_cost
            q = DependencyQuery(
                identity=obj.identity,
                home_domain=p.home,
                need_domain=need,
                nbytes=obj.nbytes,
                memory_class=obj.memory_class,
                recompute_cost=recompute_cost,
                output_bytes=0,
                recompute_legal=obj.recompute_legal,
            )
            plan = plan_dependency(topo, q)
            move_plans.append(plan)
            route = topo.shortest_path(p.home, need, obj.nbytes)
            transports.append({
                "object": obj.identity,
                "from": p.home,
                "to": need,
                "route": route.to_dict(),
            })
            if not route.timing_decidable:
                unknowns.extend(route.unknown_inputs)
            if not plan["choice_decidable"]:
                for opt in plan["options"]:
                    unknowns.extend(opt.get("unknown_inputs") or [])

    coll_plans = [plan_collective(topo, c) for c in collectives]
    for cp in coll_plans:
        if not cp["choice_decidable"]:
            unknowns.extend(cp.get("unknown_inputs") or [])

    # Deduplicate unknown strings, stable order.
    seen_u: set[str] = set()
    unknown_unique: list[str] = []
    for u in unknowns:
        if u not in seen_u:
            seen_u.add(u)
            unknown_unique.append(u)

    timing_decidable = not unknown_unique
    timing: dict[str, Any] | None = None
    speedup: float | None = None
    if timing_decidable:
        total = 0.0
        for w in work:
            if w.compute_cost.known and w.compute_cost.value is not None:
                total += w.compute_cost.value
        for t in transports:
            ts = t["route"].get("total_time_s")
            if ts is not None:
                total += ts
        for cp in coll_plans:
            if cp.get("cost_s") is not None:
                total += cp["cost_s"]
        timing = {
            "total_time_s": total,
            "provenance": "DERIVED_FROM_MEASURED_INPUTS",
            "evidence_class": "STATIC_ONLY",
            "not_a_protected_measurement": True,
            "not_a_promotion": True,
        }
        if baseline_time_s is not None:
            speedup = emit_speedup(total, baseline_time_s, timing_decidable=True)

    fault = None
    if vanish is not None:
        fault = vanish_node(topo, placements, objects, work, collectives, vanish, inflight_edges)

    residency = []
    for ident, p in sorted(placements.items()):
        residency.append({
            "identity": ident,
            "authoritative": p.home,
            "real_replicas": list(p.real_replicas),
            "hypothetical_replicas": list(p.hypothetical_replicas),
            "present_home": topo.node(p.home).present,
        })

    ownership = [
        {
            "identity": ident,
            "owner": p.owner,
            "mutable": p.mutable,
            "rule": "single owner if mutable; replicas legal only if immutable",
        }
        for ident, p in sorted(placements.items())
    ]

    result: dict[str, Any] = {
        "schema": "hawking.future.fusion_sim.result.v1",
        "topology": topo.to_dict(),
        "placement": {k: v.to_dict() for k, v in sorted(placements.items())},
        "residency": residency,
        "state_ownership": ownership,
        "transport": transports,
        "collectives": coll_plans,
        "overlap": overlap_structure(work),
        "representation_conversion": conversions,
        "move_vs_recompute": move_plans,
        "executable": not not_executable,
        "not_executable_reasons": not_executable,
        "timing_decidable": timing_decidable,
        "unknown_inputs_on_critical_path": unknown_unique,
        "timing": timing,
        "speedup": speedup,
        "fault": fault,
        "honesty": {
            "rule": (
                "when any critical-path input is UNKNOWN, timing_decidable is False "
                "and no speedup number is produced"
            ),
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
            "produces_diagnostic_relative": False,
            "produces_protected_absolute": False,
        },
    }
    if not timing_decidable:
        smugglers = _numeric_speedup_keys(result)
        if smugglers:
            raise HonestyError(f"speedup number leaked into undecidable result at {smugglers}")
        if result["speedup"] is not None:
            raise HonestyError("speedup field must be null when timing is not decidable")
        result["timing"] = None
    return result


# --------------------------------------------------------------------------- default Hawking spread


def _read_repo_json(rel: str) -> dict[str, Any] | None:
    p = REPO / rel
    if p.is_file():
        return load_json(p)
    raw = git("show", f"HEAD:{rel}")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _path_in_git(rel: str) -> bool:
    listed = git("ls-tree", "--name-only", "HEAD", rel)
    return listed.strip() == rel


def hawking_nodes() -> tuple[Node, ...]:
    genome = _read_repo_json(GENOME_RECEIPT) or {}
    identity = {
        "schema": genome.get("schema"),
        "soc": genome.get("soc"),
        "arch": genome.get("arch"),
        "cpu_cores": genome.get("cpu_cores"),
        "gpu_cores": genome.get("gpu_cores"),
        "memory_bytes": genome.get("memory_bytes"),
        "knowledge_level": genome.get("knowledge_level"),
        "os": genome.get("os"),
        "bandwidth_roof_cited": (
            f"{GENOME_RECEIPT}#measured_bandwidth.median_gb_s "
            "(INSTANCE triad; not a heterogeneous interconnect; not token_ns)"
        ),
        "sustained_behaviour": (genome.get("sustained_behaviour") or {}).get("status"),
        "thermal_envelope": (genome.get("thermal_envelope") or {}).get("status"),
    }
    apple = Node(
        id="APPLE",
        kind="APPLE",
        present=True,
        missing_dependency=None,
        genome_receipt=GENOME_RECEIPT,
        identity=identity,
        note=(
            "Present. Real MachineGenome at INSTANCE knowledge_level, produced by "
            "tools/accelerator/machine_genome.py and consumed via hcli/machine.py. "
            "Unified-memory copy deletion is the machine-specific win named in "
            "tools/accelerator/device_ascension.py; it is not a CUDA/FPGA/eGPU number."
        ),
    )
    fpga = Node(
        id="FPGA_U50",
        kind="FPGA_U50",
        present=False,
        missing_dependency=(
            "physical Xilinx Alveo U50 (or any selected FPGA board). "
            f"{FPGA_PREBOARD_RECEIPT} records physical_board.status=ABSENT, "
            "fpga_backend.status=NOT_BUILT, device_genome TARGET_UNSELECTED. "
            f"{FPGA_ORGAN_MAP} is a candidate map with no physical board, bitstream, "
            "or U50 performance claimed. FPGA is Accelerator/Physical Compiler/Fusion; "
            "it is not a civilization and this is not an FPGA backend."
        ),
        note="Absent. Pre-board scaffolding exists; the board does not.",
    )
    cuda = Node(
        id="CUDA_DGX",
        kind="CUDA_DGX",
        present=False,
        missing_dependency=(
            "NVIDIA CUDA device / DGX node. "
            f"{CUDA_CENSUS_RECEIPT} identities.transport.status=ABSENT: "
            "'no NVIDIA hardware exists on this machine'. "
            f"{CUDA_LEDGER_RECEIPT} MULTI_DEVICE.peer_access remains unmeasured. "
            "tools/accelerator/cuda_runtime.py is a Metal stand-in and records "
            "is_a_cuda_differential=False."
        ),
        note="Absent. CUDA Capability Ledger is an inventory, not a device.",
    )
    egpu = Node(
        id="EGPU",
        kind="EGPU",
        present=False,
        missing_dependency=(
            "Thunderbolt/PCIe eGPU enclosure attached to this host. "
            "tools/accelerator/fusion_planner.py models EGPU0 as physical=False and "
            "used a 5 GB/s KNOB; this module replaces the knob with UNKNOWN."
        ),
        note="Absent. fusion_planner's eGPU shape is structural; its bandwidth is not.",
    )
    mac = Node(
        id="MAC_ADDITIONAL",
        kind="MAC",
        present=False,
        missing_dependency=(
            "a second Apple Silicon host with its own INSTANCE MachineGenome plus a "
            "measured interconnect. "
            f"{GENOME_RECEIPT} is INSTANCE-scoped to this machine only; knowledge_level "
            "INSTANCE does not transfer to a sibling Mac."
        ),
        note="Absent. A second Mac is not implied by one MachineGenome.",
    )
    return (apple, fpga, cuda, egpu, mac)


def hawking_topology() -> Topology:
    """Star through APPLE. Direct FPGA-CUDA, eGPU-Spark, Mac-Mac fabrics are not assumed."""
    t = Topology()
    for n in hawking_nodes():
        t.add_node(n)
    # Every absent node attaches through Apple or not at all. Costs are UNKNOWN.
    for other in ("FPGA_U50", "CUDA_DGX", "EGPU", "MAC_ADDITIONAL"):
        t.add_link(Link(
            "APPLE", other,
            bandwidth=UNKNOWN_INTERCONNECT,
            latency=UNKNOWN_INTERCONNECT,
            note=f"APPLE<{other} interconnect; node absent; cost UNKNOWN not a knob",
        ))
    return t


def hawking_objects() -> tuple[SemanticObject, ...]:
    return (
        SemanticObject(
            "weights", "IMMUTABLE_WEIGHTS", Granularity.ORGAN, 1 << 30,
            home_hint="APPLE",
            consumers=("APPLE", "FPGA_U50", "CUDA_DGX", "EGPU", "MAC_ADDITIONAL"),
            recompute_legal=False,
            representation="metal_f16",
        ),
        SemanticObject(
            "kv", "KV_STATE", Granularity.LAYER_GROUP, 1 << 20,
            home_hint="APPLE", consumers=("APPLE",),
            recompute_legal=False, representation="metal_f16",
        ),
        SemanticObject(
            "activations", "ACTIVATIONS", Granularity.TENSOR, 1 << 16,
            home_hint="APPLE",
            consumers=("APPLE", "FPGA_U50", "CUDA_DGX"),
            recompute_legal=True, representation="metal_f16",
        ),
    )


def hawking_work() -> tuple[WorkItem, ...]:
    return (
        WorkItem(
            "decode_attn", "COMPUTE", "APPLE", 1 << 16, 1 << 16, (),
            "KV_STATE", "metal_f16", recompute_legal=False,
        ),
        WorkItem(
            "convert_act_to_hwir", "CONVERT", "APPLE", 1 << 16, 1 << 16,
            ("decode_attn",), "ACTIVATIONS", "fpga_hwir",
            converts_from="metal_f16", converts_to="fpga_hwir",
            compute_cost=unknown("no measured Metal->HWIR conversion cost; no U50"),
        ),
        WorkItem(
            "expert_mlp", "COMPUTE", "FPGA_U50", 1 << 16, 1 << 16,
            ("convert_act_to_hwir",), "ACTIVATIONS", "fpga_hwir",
            recompute_legal=True,
        ),
        WorkItem(
            "lm_head", "COMPUTE", "APPLE", 1 << 16, 1 << 16,
            ("decode_attn", "expert_mlp"), "ACTIVATIONS", "metal_f16",
        ),
    )


def hawking_collectives() -> tuple[CollectiveSpec, ...]:
    return (
        CollectiveSpec(CollectiveOp.ALLREDUCE, ("APPLE", "FPGA_U50", "CUDA_DGX"), 1 << 16),
    )


def simulate_default(*, vanish: str | None = None) -> dict[str, Any]:
    return simulate(
        hawking_topology(), hawking_objects(), hawking_work(), hawking_collectives(),
        vanish=vanish,
        inflight_edges=(("APPLE", "FPGA_U50", "activations"),),
    )


def recovered_implementation() -> dict[str, Any]:
    return {
        "fusion_planner": {
            "path": "tools/accelerator/fusion_planner.py",
            "tests": "tools/accelerator/test_fusion_planner.py",
            "what": (
                "Multi-domain topology, organ-vs-tensor placement, MOVE_DATA/"
                "MOVE_COMPUTE/RECOMPUTE/REPACK/REPLICATE/WAIT/PREFETCH, ring-vs-tree "
                "collectives. SIMULATED hop taints a route. Spark/eGPU shapes exist."
            ),
            "gap": (
                "Non-Apple bandwidth/latency are KNOBS and the planner still emits "
                "cost_s. No FPGA U50 / CUDA-DGX / additional-Mac node model. No "
                "timing_decidable. No speedup refusal. Not imported here (Codex-owned, "
                "not materialized in this sparse checkout)."
            ),
        },
        "fusion_isa": {
            "path": "tools/accelerator/fusion_isa.py",
            "what": "14 FusionOp commands, FusionTimeline submit/fence/wait, no timing.",
        },
        "fusion_wire": {
            "path": "tools/accelerator/fusion_wire.py",
            "what": "42-byte packet, no payload, no bandwidth/latency. HEADER_SIZE reused.",
        },
        "humf": {
            "path": "tools/accelerator/humf.py",
            "what": (
                "MemoryClass policy, two-domain move-vs-recompute, DeviceLost, "
                "device_lost(), vanish_next, quarantine. Fault semantics are extended, "
                "not reimplemented as a live fabric."
            ),
        },
        "machine_genome": {
            "path": "tools/accelerator/machine_genome.py",
            "receipt": GENOME_RECEIPT,
            "hcli": "hcli/machine.py",
            "what": "INSTANCE Apple MachineGenome. Apple node is present because of this.",
        },
        "device_ascension": {
            "path": "tools/accelerator/device_ascension.py",
            "what": "Unified-memory copy deletion as a machine-specific win, not a CUDA number.",
        },
        "cuda": {
            "runtime": "tools/accelerator/cuda_runtime.py",
            "ledger": "tools/accelerator/ccl.py",
            "census_receipt": CUDA_CENSUS_RECEIPT,
            "what": "Metal stand-in + capability inventory. is_a_cuda_differential=False.",
        },
        "fpga": {
            "preboard": "hcli/agentos/fpga_preboard.py",
            "receipt": FPGA_PREBOARD_RECEIPT,
            "organ_map": FPGA_ORGAN_MAP,
            "what": "TARGET_UNSELECTED, physical_board ABSENT, fpga_backend NOT_BUILT.",
        },
        "ane": {
            "asked": [ANE_ATLAS, ANE_PROFILE],
            "found": False,
        },
    }


def gaps_closed() -> list[str]:
    return [
        "Node model for APPLE (present, real MachineGenome) + FPGA_U50 + CUDA_DGX + "
        "EGPU + MAC_ADDITIONAL, each with present:bool and an exact missing dependency "
        "when absent.",
        "Honesty layer fusion_planner lacks: Cost is MEASURED (cited receipts/ path) "
        "or UNKNOWN; UNKNOWN cannot carry a number; simulate() returns structure + "
        "timing_decidable; emit_speedup() refuses when undecidable.",
        "Move-vs-recompute at every transport edge, UNDECIDABLE when either cost is UNKNOWN.",
        "Collectives as structure; algorithm UNDECIDABLE when interconnect costs are UNKNOWN "
        "(DIRECT at p<=2 is structural, not a cost pick).",
        "CUDA seam: backend-neutral FusionOp lowering contract + Metal/CUDA differential "
        "schema that forbids a local CUDA claim on Apple hardware.",
        "Fault/recovery: vanish_node always returns a defined outcome "
        "(DATA_LOSS / RECOVERED_FROM_REPLICA / ABORT / ALREADY_ABSENT / ...), "
        "extending humf.device_lost onto a heterogeneous plan.",
        "Replicas on absent nodes are marked hypothetical and cannot satisfy recovery.",
    ]


def negative_findings() -> list[str]:
    ane_atlas = _path_in_git(ANE_ATLAS)
    ane_profile = _path_in_git(ANE_PROFILE)
    findings = [
        f"{ANE_ATLAS}: {'present in git' if ane_atlas else 'NOT in git'} — asked to read, could not.",
        f"{ANE_PROFILE}: {'present in git' if ane_profile else 'NOT in git'} — asked to read, could not.",
        "No U50-specific device genome (FPGA preboard is TARGET_UNSELECTED, not Alveo U50).",
        "No measured interconnect to FPGA, CUDA/DGX, eGPU, or a second Mac.",
        "No PROTECTED_ABSOLUTE token_ns, so even Apple-local compute is UNKNOWN for execution timing.",
        "tools/accelerator/fusion_planner.py, fusion_isa.py, fusion_wire.py, humf.py, "
        "machine_genome.py, cuda_runtime.py and hcli/machine.py are in git but not "
        "materialized in this sparse checkout; recovered via git show; not imported "
        "(Codex-owned; importing them would also fail).",
        "Cannot run Codex fusion_planner tests from this worktree.",
        "Sidecar produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE.",
    ]
    return findings


def build() -> Any:
    sim = simulate_default()
    fault_apple = simulate_default(vanish="APPLE")
    fault_fpga = simulate_default(vanish="FPGA_U50")
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Semantic heterogeneous simulation of a Hawking execution spread across "
            "Apple Silicon, an FPGA card, a future CUDA/DGX node, an eGPU and "
            "additional Macs — structurally incapable of emitting a heterogeneous "
            "speedup number from unknown inputs."
        ),
        "vocabulary": {
            "eras": ["I Genesis of the Laboratory", "II Compounding Civilization",
                     "III Autonomous Science Civilization", "IV Synthetic Machine Civilization",
                     "V Released Hawking Civilization"],
            "no_era_vi": True,
            "odysseys": ["I WHAT IS TRUE?", "II WHAT DID HAWKING ALREADY LEARN?",
                         "III WHERE IS HAWKING WRONG?"],
            "no_odyssey_iv": True,
            "fpga_is_not_a_civilization": True,
            "disk_state_is_authority": True,
            "this_lane_emits": "STATIC_ONLY / bench.state=UNKNOWN / gpu_authority=false",
        },
        "node_model": [n.to_dict() for n in hawking_nodes()],
        "simulated_dimensions": [
            "placement", "residency", "transport", "collectives", "overlap",
            "state_ownership", "representation_conversion", "fault_recovery",
        ],
        "honesty_rule": {
            "inputs": "MEASURED Hawking record (cite receipts/ path) or UNKNOWN",
            "unknown_cannot_carry_a_number": True,
            "when_critical_path_unknown": "timing_decidable=False and no speedup figure",
            "emit_speedup_refuses_undecidable": True,
            "fusion_planner_gap": (
                "fusion_planner.py stamps SIMULATED provenance but still returns cost_s "
                "from knobs. This module does not compute a time unless every hop is MEASURED."
            ),
        },
        "cuda_seam": {
            "lowering_contract": LOWERING_CONTRACT,
            "metal_cuda_differential_schema": METAL_CUDA_DIFFERENTIAL_SCHEMA,
            "local_cuda_on_apple": "FORBIDDEN",
        },
        "default_simulation": sim,
        "fault_recovery": {
            "vanish_APPLE": fault_apple.get("fault"),
            "vanish_FPGA_U50": fault_fpga.get("fault"),
        },
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(),
        "integration": {
            "simulate": (
                "simulate(topo, objects, work, collectives=(), vanish=None, "
                "inflight_edges=(), baseline_time_s=None) -> dict"
            ),
            "emit_speedup": (
                "emit_speedup(hetero_time_s, baseline_time_s, *, timing_decidable) -> float; "
                "raises HonestyError unless timing_decidable"
            ),
            "hawking_topology": "hawking_topology() -> Topology",
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/fusion_sim.py")


def selftest() -> Any:
    topo = hawking_topology()
    present = {n.id: n.present for n in topo.nodes}
    assert present["APPLE"] is True
    assert present["FPGA_U50"] is False
    assert present["CUDA_DGX"] is False
    assert present["EGPU"] is False
    assert present["MAC_ADDITIONAL"] is False
    for n in topo.nodes:
        assert n.present == (n.missing_dependency is None)
    sim = simulate_default()
    assert sim["timing_decidable"] is False
    assert sim["speedup"] is None
    assert _numeric_speedup_keys(sim) == []
    try:
        emit_speedup(1.0, 2.0, timing_decidable=False)
        raise AssertionError("emit_speedup must refuse undecidable timing")
    except HonestyError:
        pass
    try:
        Cost(kind="UNKNOWN", value=12.0)
        raise AssertionError("UNKNOWN cost must refuse a number")
    except HonestyError:
        pass
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.build:
        out = selftest()
    else:
        out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
