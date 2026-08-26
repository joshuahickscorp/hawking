"""Fusion Planner -- the placement/topology half of G054. FRONT H.

humf.py/hmf.py already own a two-domain move-vs-recompute planner
(Humf.plan_acquire) with a full trust model (State, Ownership, quarantine,
verification). This module does not reimplement any of that -- it is READ
and reused (MemoryClass, MEMORY_CLASS_POLICY, humf.Plan) -- and it does not
touch HUMF's coherence engine at all. What is missing from HUMF, and what
this module adds, is everything HAWKGPU-0's own docstring names as absent:
a multi-domain TOPOLOGY (more than one link, each with its own bandwidth and
latency), a PLACEMENT policy over that topology (which domain a semantic
object's authoritative copy lives in, and whether it may be replicated), an
extended move-vs-recompute-vs-more choice (MOVE_DATA, MOVE_COMPUTE,
RECOMPUTE, REPACK, REPLICATE, WAIT, PREFETCH) for a REMOTE dependency across
that topology, and a COLLECTIVE planner (AllReduce/Broadcast/AllGather/
ReduceScatter) that picks ring vs tree by topology and message size.

THIS PROGRAM'S OWN LAW: no Spark and no eGPU exist on this machine. Every
bandwidth and latency number below a non-APPLE link is a KNOB chosen for
structural testing, never a measurement. Topology.shortest_path() stamps
each Route's cost_provenance "MEASURED" only when every hop on the path is
physical=True; a single simulated hop taints the whole route SIMULATED,
exactly like humf.Humf._transfer_cost already does for a single two-domain
hop. Every Plan and CollectivePlan this module returns carries that
provenance forward untouched -- nothing here launders a SIMULATED number
into a claim a caller could mistake for measured.

TOPOLOGY. Topology is a plain undirected graph: named domains, Links between
pairs of them (bandwidth_gb_s, latency_s, physical). Route cost is STORE-
AND-FORWARD and additive across hops -- nbytes/bw_i + latency_i summed per
hop -- matching humf._transfer_cost's own single-hop formula rather than
inventing a second cost model. shortest_path() is plain Dijkstra with that
per-hop weight, so it (a) generalizes to any number of domains with no
hardcoded ceiling, and (b) naturally refuses to route Spark-to-Spark traffic
through Apple when a direct SPARK_SUPERDOMAIN link is faster, and just as
naturally DOES route through Apple when a direct link is slower than the
two-hop relay -- both directions are pinned in test_fusion_planner.py,
because one direction alone proves nothing about a shortest-path chooser.

PLACEMENT. place_objects() decides, for each SemanticObject, where its
authoritative copy lives and whether it may additionally be REPLICATED.
MEMORY_CLASS_POLICY (imported from humf, not redefined) is the only source
of truth for mutability: any memory_class with `mutable: True` (KV_STATE,
RECURRENT_STATE, and every other mutable class) gets exactly one owner and
is never offered replication -- the option is omitted, not merely
discouraged. An immutable, `shared_read_stable` class (IMMUTABLE_WEIGHTS,
COMPILER_ARTIFACT, EXPERT_CACHE) at ORGAN-or-coarser granularity is
replicated once into every remote consumer domain up front -- whole-organ
placement, matching the task's explicit preference over tensor ping-pong.
The SAME immutable class at TENSOR granularity is deliberately NOT eagerly
replicated -- fine-grained ping-pong is exactly what whole-organ placement
exists to avoid, so a lone tensor's remote reads are left to plan_dependency
per use instead.

THE PLANNER'S CHOICE. plan_dependency() estimates MOVE_DATA, MOVE_COMPUTE,
RECOMPUTE, REPACK, REPLICATE, WAIT and PREFETCH for one remote dependency
and returns cheapest-wins as a humf.Plan (the same dataclass plan_acquire
already returns, not a competing type). REPLICATE is omitted entirely for a
mutable memory_class -- single ownership is enforced by absence from the
option set, the same discipline MEMORY_CLASS_POLICY already enforces for a
write via HumfObject.mark_written(). THE PERMANENT LAW: on an exact cost
tie, the option that avoids moving the object's own bytes wins -- see
TIE_BREAK_RANK. This is why REPLICATE (same cost as MOVE_DATA, since both
require the identical transfer) is preferred over a bare MOVE_DATA whenever
they tie: the data cannot go stale, so paying the transfer once and keeping
it is never worse and serves every future request for free.

COLLECTIVES. plan_collective() covers ALLREDUCE, BROADCAST, ALLGATHER and
REDUCE_SCATTER as PLANNING decisions over a Topology, not as real
implementations -- no bytes move, no kernel launches. All four use ONE
uniform two-term alpha/beta cost pair (RING: (p-1) latency-hops, (p-1)/p
bandwidth-share per hop; TREE: ceil(log2 p) latency-hops, ceil(log2 p)
bandwidth-share per hop -- see the module's own crossover derivation in
collective_crossover_bytes()), which is a deliberate simplification named
here rather than left silent: real libraries (NCCL, MPI implementations)
use per-operation-optimized variants (Rabenseifner's algorithm, recursive
halving/doubling reduce-scatter and allgather, segmented pipelining) that
this module does not model. alpha/beta for a given collective's domain
group are read off the WORST ring-adjacent hop in the topology (a
synchronized collective proceeds at the pace of its slowest link), so an
unconnected pair that must relay through a hub is penalized exactly as a
real collective over that topology would be -- the same shortest-path
machinery the point-to-point planner uses, not a second one. At two or
fewer participants ring and tree are literally the same one-hop exchange,
reported as DIRECT rather than manufacturing a distinction that is not
real.

NOT IMPLEMENTED, named rather than left silent:
  - No wiring into Humf/HumfObject/Materialization at all. This module reads
    humf.MemoryClass, humf.MEMORY_CLASS_POLICY and humf.Plan and returns
    plain dataclasses of its own (Topology, Route, SemanticObject,
    Placement, CollectivePlan); nothing here registers an object with a
    live Humf fabric or calls plan_acquire()/execute(). Bridging a
    Topology's Route into an actual humf.Domain-per-link fabric (so
    Humf.plan_acquire could route across more than the two domains it
    already supports) is not built.
  - No real transport, no bytes move, ever, anywhere in this module --
    matching fusion_wire.py's own "no payload on this wire" discipline.
    Topology, Route, Plan and CollectivePlan are cost ESTIMATES only.
  - No capacity accounting. place_objects() never checks that a domain
    actually has room for what it decides to replicate into; HAWKGPU-0's
    own docstring already names capacity accounting as unbuilt and this
    module does not build it either.
  - No pipelining/segmentation in either the point-to-point Route cost or
    the collective cost model. A multi-hop Route is pure store-and-forward
    (the whole object lands at a hop before the next hop starts); no
    concurrent-hop streaming is modeled.
  - No recursive-halving/doubling variants for AllGather/ReduceScatter, and
    no hybrid algorithm selection beyond the single ring/tree choice -- see
    the COLLECTIVES section above.
  - No online/adaptive replanning. Every Plan and CollectivePlan is a single
    point-in-time estimate from the numbers a caller supplied; nothing here
    watches an in-flight transfer or revises a plan once it starts
    (WAIT/PREFETCH accept a caller-supplied estimate of the OTHER transfer's
    remaining time/overlap; this module does not track that transfer
    itself).
  - No true multi-consumer collective co-scheduling: plan_dependency() plans
    ONE object into ONE need_domain per call. Planning several objects or
    several consumers together for shared-route amortization is not done.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from humf import MEMORY_CLASS_POLICY, MemoryClass, Plan

# Reused, not re-declared: a "move the compute instead" dispatch is modeled as
# sending one fusion_isa command's wire encoding to the remote domain, not an
# invented constant. See fusion_wire.py's own module docstring: 42 bytes,
# fixed, no payload.
from fusion_wire import HEADER_SIZE as _DISPATCH_PACKET_BYTES


class FusionPlannerError(RuntimeError):
    """Base for every error this module raises."""


# --------------------------------------------------------------- topology


@dataclass(frozen=True)
class Link:
    """One edge of a Topology. `physical` is False for anything this machine
    does not actually have -- see the module docstring's law on that."""
    a: str
    b: str
    bandwidth_gb_s: float
    latency_s: float
    physical: bool
    note: str = ""

    @property
    def provenance(self) -> str:
        return "MEASURED" if self.physical else "SIMULATED"


@dataclass(frozen=True)
class Route:
    """One shortest_path() result. `hops` is the ordered sequence of Links
    actually crossed -- `len(hops) == 1` means direct, `> 1` means the path
    was relayed through one or more intermediate domains."""
    path: tuple[str, ...]
    hops: tuple[Link, ...]
    nbytes: int
    total_time_s: float
    cost_provenance: str

    @property
    def alpha_s(self) -> float:
        """Pure per-hop latency, summed over the path -- what total_time_s
        would be at nbytes=0. Exposed directly rather than requiring a
        caller to re-run shortest_path(..., 0) to get it."""
        return sum(h.latency_s for h in self.hops)

    @property
    def beta_s_per_byte(self) -> float:
        """Marginal seconds per additional byte, summed over the path (each
        hop is store-and-forward, so its per-byte cost is additive with
        every other hop's, not bottlenecked to the single slowest one)."""
        return sum(1.0 / (h.bandwidth_gb_s * 1e9) for h in self.hops)


class Topology:
    """An undirected graph of domains and Links between them. No maximum
    node count is enforced or assumed anywhere in this class -- add_domain/
    add_link accept any name, any number of times for different names, and
    shortest_path is plain Dijkstra over whatever has been added."""

    def __init__(self) -> None:
        self.domains: dict[str, bool] = {}          # name -> physical
        self._links: dict[frozenset[str], Link] = {}
        self._adj: dict[str, list[str]] = {}

    def add_domain(self, name: str, *, physical: bool) -> None:
        if name in self.domains:
            raise FusionPlannerError(f"domain {name!r} is already in this topology")
        self.domains[name] = physical
        self._adj[name] = []

    def add_link(self, a: str, b: str, *, bandwidth_gb_s: float, latency_s: float,
                 physical: bool, note: str = "") -> None:
        for d in (a, b):
            if d not in self.domains:
                raise FusionPlannerError(f"cannot link {d!r}: not a domain in this topology")
        key = frozenset((a, b))
        if key in self._links:
            raise FusionPlannerError(f"a link between {a!r} and {b!r} already exists")
        self._links[key] = Link(a, b, bandwidth_gb_s, latency_s, physical, note)
        self._adj[a].append(b)
        self._adj[b].append(a)

    def link(self, a: str, b: str) -> Link | None:
        return self._links.get(frozenset((a, b)))

    def neighbors(self, name: str) -> list[str]:
        return list(self._adj.get(name, ()))

    def shortest_path(self, src: str, dst: str, nbytes: int) -> Route:
        """Dijkstra, edge weight = nbytes/bandwidth + latency (store-and-
        forward, additive per hop -- see module docstring). `nbytes` is
        baked into every edge weight for THIS call, so the path found is
        optimal for that specific transfer size; a different size may
        legitimately prefer a different path (a high-bandwidth/high-latency
        link is worth it for a large transfer and not for a tiny one)."""
        for d in (src, dst):
            if d not in self.domains:
                raise FusionPlannerError(f"{d!r} is not a domain in this topology")
        if src == dst:
            return Route((src,), (), nbytes, 0.0, "MEASURED")
        dist: dict[str, float] = {src: 0.0}
        prev: dict[str, tuple[str, Link]] = {}
        visited: set[str] = set()
        heap: list[tuple[float, str]] = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == dst:
                break
            for v in self.neighbors(u):
                lk = self.link(u, v)
                w = nbytes / (lk.bandwidth_gb_s * 1e9) + lk.latency_s
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v] = nd
                    prev[v] = (u, lk)
                    heapq.heappush(heap, (nd, v))
        if dst not in dist:
            raise FusionPlannerError(f"no route from {src!r} to {dst!r} in this topology")
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
        provenance = "MEASURED" if all(h.physical for h in hops) else "SIMULATED"
        return Route(tuple(path), tuple(hops), nbytes, dist[dst], provenance)


# ---------------------------------------------------------- topology shapes


def topology_apple_alone() -> Topology:
    t = Topology()
    t.add_domain("APPLE", physical=True)
    return t


def topology_apple_plus_sparks(
        n_sparks: int, *,
        apple_spark_bw_gb_s: float = 12.0, apple_spark_latency_s: float = 2.5e-4,
        spark_superdomain_bw_gb_s: float = 100.0,
        spark_superdomain_latency_s: float = 5e-6) -> Topology:
    """Apple plus N Sparks, N >= 1, with no ceiling on N. Every Spark gets a
    link to Apple, and every PAIR of Sparks gets a direct SPARK_SUPERDOMAIN
    link -- a full mesh, so Spark-to-Spark traffic never has to relay
    through Apple by construction when N >= 2. All figures are KNOBS (no
    Spark hardware exists on this machine): apple_spark_bw_gb_s=12.0 matches
    the fixture already used for a fictitious SPARK_DOMAIN_0 in
    test_hmf.py::test_a_second_fictitious_domain_needs_no_architecture_change;
    spark_superdomain_bw_gb_s=100.0 is chosen only to be clearly faster,
    modeling a direct interconnect between co-located accelerators."""
    if n_sparks < 1:
        raise FusionPlannerError("topology_apple_plus_sparks needs n_sparks >= 1")
    t = Topology()
    t.add_domain("APPLE", physical=True)
    sparks = [f"SPARK{i}" for i in range(n_sparks)]
    for name in sparks:
        t.add_domain(name, physical=False)
        t.add_link("APPLE", name, bandwidth_gb_s=apple_spark_bw_gb_s,
                    latency_s=apple_spark_latency_s, physical=False,
                    note="APPLE<->SPARK bridge, SIMULATED")
    for i in range(n_sparks):
        for j in range(i + 1, n_sparks):
            t.add_link(sparks[i], sparks[j], bandwidth_gb_s=spark_superdomain_bw_gb_s,
                        latency_s=spark_superdomain_latency_s, physical=False,
                        note="SPARK_SUPERDOMAIN direct interconnect, SIMULATED")
    return t


def topology_apple_spark_egpu(
        *, n_sparks: int = 1,
        apple_spark_bw_gb_s: float = 12.0, apple_spark_latency_s: float = 2.5e-4,
        spark_superdomain_bw_gb_s: float = 100.0,
        spark_superdomain_latency_s: float = 5e-6,
        egpu_bw_gb_s: float = 5.0, egpu_latency_s: float = 5e-4) -> Topology:
    """Apple + Spark(s) + one eGPU. The eGPU attaches to Apple only (a real
    eGPU enclosure is PCIe/Thunderbolt off the host, not off a Spark) --
    egpu_bw_gb_s=5.0 matches the "a 5 GB/s bridge" figure humf.py's own
    docstrings already use as the representative external-bridge number."""
    t = topology_apple_plus_sparks(
        n_sparks, apple_spark_bw_gb_s=apple_spark_bw_gb_s,
        apple_spark_latency_s=apple_spark_latency_s,
        spark_superdomain_bw_gb_s=spark_superdomain_bw_gb_s,
        spark_superdomain_latency_s=spark_superdomain_latency_s)
    t.add_domain("EGPU0", physical=False)
    t.add_link("APPLE", "EGPU0", bandwidth_gb_s=egpu_bw_gb_s, latency_s=egpu_latency_s,
                physical=False, note="Thunderbolt/PCIe eGPU bridge, SIMULATED")
    return t


# ---------------------------------------------------------------- placement


class Granularity(str, Enum):
    """Coarsest to finest. Whole-organ placement means MODEL/ORGAN/
    LAYER_GROUP/EXPERT_GROUP; TENSOR is the fine-grained case place_objects
    deliberately does NOT eagerly replicate -- see module docstring."""
    MODEL = "MODEL"
    ORGAN = "ORGAN"
    LAYER_GROUP = "LAYER_GROUP"
    EXPERT_GROUP = "EXPERT_GROUP"
    TENSOR = "TENSOR"


@dataclass
class SemanticObject:
    """One thing the planner must place. `home_hint` is where it is produced
    or should live absent any other signal; `consumers` are domains that
    need to read it (not necessarily distinct from home_hint)."""
    identity: str
    memory_class: MemoryClass
    granularity: Granularity
    nbytes: int
    home_hint: str | None = None
    consumers: tuple[str, ...] = ()


@dataclass
class Placement:
    identity: str
    home: str
    replicas: tuple[str, ...]
    reason: str


def place_objects(topo: Topology, objects: Sequence[SemanticObject]) -> dict[str, Placement]:
    """Decide a home (and, where legal and warranted, replicas) for each
    SemanticObject. See module docstring for the mutable/immutable and
    granularity rules; MEMORY_CLASS_POLICY (from humf) is the sole source of
    truth for mutability, not a second table maintained here."""
    if not topo.domains:
        raise FusionPlannerError("empty topology has nowhere to place anything")
    default_home = next(iter(topo.domains))
    out: dict[str, Placement] = {}
    for obj in objects:
        home = obj.home_hint or default_home
        if home not in topo.domains:
            raise FusionPlannerError(
                f"{obj.identity}: home domain {home!r} is not in this topology")
        policy = MEMORY_CLASS_POLICY[obj.memory_class]
        if policy["mutable"]:
            out[obj.identity] = Placement(
                obj.identity, home, (),
                reason=f"{obj.memory_class.value} is mutable per MEMORY_CLASS_POLICY; "
                       f"a single authoritative owner ({home}) is required and "
                       f"replication is never offered for it")
            continue
        remote = tuple(sorted(c for c in obj.consumers if c != home))
        if not remote:
            out[obj.identity] = Placement(
                obj.identity, home, (),
                reason=f"{obj.memory_class.value} is immutable but has no remote "
                       f"consumer; single copy at {home}")
            continue
        if obj.granularity is Granularity.TENSOR:
            out[obj.identity] = Placement(
                obj.identity, home, (),
                reason=f"TENSOR-granularity immutable object with remote consumers "
                       f"{list(remote)}; whole-organ placement is preferred over "
                       f"fine-grained ping-pong, so this stays single-copy at {home} "
                       f"and remote reads go through plan_dependency() per use "
                       f"instead of being eagerly replicated")
            continue
        for c in remote:
            if c not in topo.domains:
                raise FusionPlannerError(
                    f"{obj.identity}: consumer domain {c!r} is not in this topology")
        out[obj.identity] = Placement(
            obj.identity, home, remote,
            reason=f"{obj.granularity.value}-granularity {obj.memory_class.value} is "
                   f"immutable and shared_read_stable; replicated once into every "
                   f"remote consumer {list(remote)} rather than re-streamed per "
                   f"access -- whole-organ placement over tensor ping-pong")
    return out


# --------------------------------------------------------- the planner's choice


TIE_BREAK_RANK = {
    "RECOMPUTE": 0,
    "MOVE_COMPUTE": 1,
    "WAIT": 2,
    "PREFETCH": 3,
    "REPACK": 4,
    "REPLICATE": 5,
    "MOVE_DATA": 6,
}


def _combine_provenance(*provs: str) -> str:
    return "MEASURED" if all(p == "MEASURED" for p in provs) else "SIMULATED"


@dataclass
class DependencyQuery:
    """One remote dependency to resolve: `identity` needs to be usable in
    `need_domain` and currently lives in `home_domain`. Every field below
    `memory_class` is OPTIONAL and turns on exactly one extra option in
    plan_dependency()'s comparison -- a query with none of them set only
    ever gets MOVE_DATA (plus REPLICATE, if the memory class is immutable)."""
    identity: str
    home_domain: str
    need_domain: str
    nbytes: int
    memory_class: MemoryClass
    recompute_cost_s: float | None = None
    repack_bytes: int | None = None
    repack_cost_s: float | None = None
    output_bytes: int = 0
    in_flight_eta_s: float | None = None
    overlap_window_s: float | None = None


def plan_dependency(topo: Topology, q: DependencyQuery) -> Plan:
    """Cheapest of MOVE_DATA / MOVE_COMPUTE / RECOMPUTE / REPACK / REPLICATE
    / WAIT / PREFETCH, returned as a humf.Plan -- the SAME dataclass
    Humf.plan_acquire() already returns, extended with more actions rather
    than replaced by a competing type. On an exact cost tie, TIE_BREAK_RANK
    prefers whichever option avoids moving the object's own bytes -- the
    permanent law: fastest transfer is transfer proven unnecessary."""
    if q.home_domain == q.need_domain:
        return Plan("ALREADY_RESIDENT", 0.0, f"{q.identity} is already in {q.need_domain}",
                    "MEASURED", [], source=q.home_domain)

    opts: list[dict[str, Any]] = []

    move_route = topo.shortest_path(q.home_domain, q.need_domain, q.nbytes)
    opts.append({
        "action": "MOVE_DATA", "cost_s": move_route.total_time_s,
        "cost_provenance": move_route.cost_provenance,
        "detail": f"transfer {q.nbytes}B {q.home_domain} -> {q.need_domain} via "
                  f"{'->'.join(move_route.path)}",
    })

    if not MEMORY_CLASS_POLICY[q.memory_class]["mutable"]:
        # REPLICATE costs exactly what MOVE_DATA costs -- both move the same
        # bytes once. Offered ONLY for an immutable class: for a mutable one
        # a second authoritative copy would be incoherent the instant either
        # side is written, so the option does not exist, matching
        # HumfObject.mark_written()'s own refusal for the same class.
        opts.append({
            "action": "REPLICATE", "cost_s": move_route.total_time_s,
            "cost_provenance": move_route.cost_provenance,
            "detail": f"place a persistent replica of {q.identity} in "
                      f"{q.need_domain} (legal: {q.memory_class.value} is immutable "
                      f"and cannot go stale, so the copy serves every future request "
                      f"at zero further cost)",
        })

    dispatch_route = topo.shortest_path(q.need_domain, q.home_domain, _DISPATCH_PACKET_BYTES)
    if q.output_bytes:
        result_route = topo.shortest_path(q.home_domain, q.need_domain, q.output_bytes)
        mc_cost = dispatch_route.total_time_s + result_route.total_time_s
        mc_prov = _combine_provenance(dispatch_route.cost_provenance, result_route.cost_provenance)
    else:
        mc_cost = dispatch_route.total_time_s
        mc_prov = dispatch_route.cost_provenance
    opts.append({
        "action": "MOVE_COMPUTE", "cost_s": mc_cost, "cost_provenance": mc_prov,
        "detail": f"run the consumer in {q.home_domain} instead of moving {q.nbytes}B: "
                  f"send a {_DISPATCH_PACKET_BYTES}B dispatch packet and return "
                  f"{q.output_bytes}B of result",
    })

    if q.recompute_cost_s is not None:
        opts.append({
            "action": "RECOMPUTE", "cost_s": q.recompute_cost_s,
            "cost_provenance": "MEASURED",
            "detail": f"recompute {q.identity} locally in {q.need_domain} instead of "
                      f"transferring it",
        })

    if q.repack_bytes is not None and q.repack_cost_s is not None:
        repack_route = topo.shortest_path(q.home_domain, q.need_domain, q.repack_bytes)
        opts.append({
            "action": "REPACK",
            "cost_s": q.repack_cost_s + repack_route.total_time_s,
            "cost_provenance": repack_route.cost_provenance,
            "detail": f"repack {q.identity} to {q.repack_bytes}B in {q.home_domain} "
                      f"({q.repack_cost_s}s) then transfer the smaller form",
        })

    if q.in_flight_eta_s is not None:
        opts.append({
            "action": "WAIT", "cost_s": q.in_flight_eta_s, "cost_provenance": "SIMULATED",
            "detail": f"{q.identity} is already in flight toward {q.need_domain}; wait "
                      f"{q.in_flight_eta_s}s for it to land instead of starting a new "
                      f"transfer",
        })

    if q.overlap_window_s is not None:
        hidden = min(q.overlap_window_s, move_route.total_time_s)
        residual = move_route.total_time_s - hidden
        opts.append({
            "action": "PREFETCH", "cost_s": residual,
            "cost_provenance": move_route.cost_provenance,
            "detail": f"start the {q.nbytes}B transfer now; {hidden:.6f}s of its "
                      f"{move_route.total_time_s:.6f}s overlaps other work, leaving "
                      f"{residual:.6f}s on the critical path",
        })

    best = min(opts, key=lambda o: (o["cost_s"], TIE_BREAK_RANK[o["action"]]))
    return Plan(best["action"], best["cost_s"], best["detail"], best["cost_provenance"],
               opts, source=None if best["action"] in ("RECOMPUTE", "MOVE_COMPUTE")
               else q.home_domain)


# ------------------------------------------------------------------ collectives


class CollectiveOp(str, Enum):
    ALLREDUCE = "ALLREDUCE"
    BROADCAST = "BROADCAST"
    ALLGATHER = "ALLGATHER"
    REDUCE_SCATTER = "REDUCE_SCATTER"


def _tree_depth(p: int) -> int:
    return max(1, math.ceil(math.log2(p)))


def collective_crossover_bytes(p: int, alpha_s: float, beta_s_per_byte: float) -> float | None:
    """The message size at which RING and TREE cost the same, for p
    participants at this (alpha, beta). None means one algorithm dominates
    the other for every positive message size at this p -- an honest report,
    not a division by a number that happened to be zero.

    RING: (p-1) latency-hops, (p-1)/p bandwidth-share per hop.
    TREE:  ceil(log2 p) latency-hops, ceil(log2 p) bandwidth-share per hop.
    Solve (p-1)*alpha + (p-1)/p*n*beta == L*alpha + L*n*beta for n."""
    if p <= 2:
        return None
    L = _tree_depth(p)
    d_alpha = (p - 1) - L            # ring's latency-hop excess over tree's
    d_beta = L - (p - 1) / p         # tree's bandwidth-share excess over ring's
    if d_alpha <= 0 or d_beta <= 0 or beta_s_per_byte <= 0:
        return None
    return (d_alpha * alpha_s) / (d_beta * beta_s_per_byte)


@dataclass
class CollectivePlan:
    op: str
    domains: tuple[str, ...]
    message_bytes: int
    algorithm: str
    cost_s: float
    cost_provenance: str
    ring_cost_s: float
    tree_cost_s: float
    crossover_bytes: float | None
    alpha_s: float
    beta_s_per_byte: float
    reason: str


def _group_alpha_beta(topo: Topology, domains: Sequence[str]) -> tuple[float, float, str]:
    """alpha/beta representative of a collective over `domains`: the WORST
    (slowest) ring-adjacent hop, independently by latency and by per-byte
    time. A synchronized collective proceeds at the pace of its slowest
    link, and routing consecutive-in-ring pairs through shortest_path means
    an unconnected pair (must relay through a hub) is penalized exactly the
    way a real collective over this topology would be -- the SAME routing
    machinery plan_dependency() uses, not a second one."""
    p = len(domains)
    worst_alpha = 0.0
    worst_beta = 0.0
    provs: list[str] = []
    for i in range(p):
        route = topo.shortest_path(domains[i], domains[(i + 1) % p], 0)
        worst_alpha = max(worst_alpha, route.alpha_s)
        worst_beta = max(worst_beta, route.beta_s_per_byte)
        provs.append(route.cost_provenance)
    return worst_alpha, worst_beta, _combine_provenance(*provs)


def plan_collective(topo: Topology, op: CollectiveOp, domains: Sequence[str],
                     message_bytes: int) -> CollectivePlan:
    """Ring vs tree, chosen by topology (via alpha/beta) and message size,
    with the crossover reported as a computed quantity -- never asserted by
    fiat. At <= 2 participants ring and tree are the same one-hop exchange;
    reported as DIRECT rather than manufacturing a distinction that is not
    real."""
    domains = list(domains)
    if len(domains) < 2:
        raise FusionPlannerError("a collective needs at least 2 participants")
    if len(set(domains)) != len(domains):
        raise FusionPlannerError(f"duplicate participant in {domains!r}")
    for d in domains:
        if d not in topo.domains:
            raise FusionPlannerError(f"{d!r} is not a domain in this topology")

    alpha, beta, prov = _group_alpha_beta(topo, domains)
    p = len(domains)

    if p <= 2:
        cost = alpha + message_bytes * beta
        return CollectivePlan(
            op.value, tuple(domains), message_bytes, "DIRECT", cost, prov, cost, cost, None,
            alpha, beta,
            reason=f"only {p} participants; ring and tree collapse to the same direct "
                   f"exchange, so choosing between them is not a real distinction here")

    L = _tree_depth(p)
    ring_cost = (p - 1) * alpha + (p - 1) / p * message_bytes * beta
    tree_cost = L * alpha + L * message_bytes * beta
    crossover = collective_crossover_bytes(p, alpha, beta)

    if ring_cost <= tree_cost:
        algorithm, cost = "RING", ring_cost
        why = (f"message_bytes={message_bytes} is at or above the crossover "
               f"({crossover:.0f}B)" if crossover is not None
               else "TREE never wins at this p, alpha and beta")
        reason = (f"RING chosen: {ring_cost:.9f}s <= {tree_cost:.9f}s for p={p}. "
                  f"{why}; ring's (p-1)/p={((p-1)/p):.4f} bandwidth-share per hop "
                  f"beats tree's {L} at this message size.")
    else:
        algorithm, cost = "TREE", tree_cost
        why = (f"message_bytes={message_bytes} is below the crossover "
               f"({crossover:.0f}B)" if crossover is not None
               else "RING never wins at this p, alpha and beta")
        reason = (f"TREE chosen: {tree_cost:.9f}s < {ring_cost:.9f}s for p={p}. "
                  f"{why}; tree's {L} latency-hops beats ring's (p-1)={p - 1} at this "
                  f"message size.")

    return CollectivePlan(op.value, tuple(domains), message_bytes, algorithm, cost, prov,
                           ring_cost, tree_cost, crossover, alpha, beta, reason)
