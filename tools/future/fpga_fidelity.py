"""FPGA multi-fidelity estimator and the four adaptation clocks.

Cheap ladder levels can kill losers before an expensive level runs.
Disagreement between levels is a finding, not noise. HW_EMULATION and
REAL_HARDWARE are UNIMPLEMENTED (no emulation seat, no U50 board) and
raise rather than guess.

The structural graph here is a local stand-in. Swap it for
tools/future/hwir.py when that module lands (frontier F003).

    python3 tools/future/fpga_fidelity.py --selftest
    python3 tools/future/fpga_fidelity.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, git

import argparse
import hashlib
import json
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Mapping, Sequence


RECEIPT = "FPGA_MULTIFIDELITY.json"
SCHEMA = "hawking.future.fpga_fidelity.v1"
COEFFICIENT_TABLE = "v1-assumed-not-measured"

# Model beat for turning an edge frame (bytes) into modelled transfer cycles.
# This is not a clock and is never converted to seconds.
MODEL_BYTES_PER_CYCLE = 64

# Ranking weights over structural resource counts. Not prices, not measurements.
PRESSURE_WEIGHT = {
    "dsp": 100,
    "lut": 1,
    "bram": 8,
    "uram": 16,
    "hbm_channels": 32,
}


class FidelityLevel(str, Enum):
    ANALYTICAL = "ANALYTICAL"
    FUNCTIONAL_SIM = "FUNCTIONAL_SIM"
    CYCLE_MODEL = "CYCLE_MODEL"
    HW_EMULATION = "HW_EMULATION"
    REAL_HARDWARE = "REAL_HARDWARE"


LADDER_ORDER: tuple[FidelityLevel, ...] = (
    FidelityLevel.ANALYTICAL,
    FidelityLevel.FUNCTIONAL_SIM,
    FidelityLevel.CYCLE_MODEL,
    FidelityLevel.HW_EMULATION,
    FidelityLevel.REAL_HARDWARE,
)


class AdaptationClock(IntEnum):
    CLOCK_0_RUNTIME_SCHEDULING = 0
    CLOCK_1_OVERLAY_MICROCODE = 1
    CLOCK_2_CACHED_MODULE_DFX = 2
    CLOCK_3_DEEP_ARCHITECTURE = 3


class ProviderUnavailable(RuntimeError):
    """Raised when a fidelity provider is not available. Never a guess."""

    def __init__(self, level: FidelityLevel, missing_dependency: str) -> None:
        self.level = level if isinstance(level, FidelityLevel) else FidelityLevel(level)
        self.missing_dependency = missing_dependency
        super().__init__(f"{self.level.value} unavailable: {missing_dependency}")


class UnmeasuredConversionError(RuntimeError):
    """Raised if modelled cycles are asked to become wall time."""


class GraphError(ValueError):
    """Ill-formed structural graph."""


# Closed operator catalog. Local stand-in for atlas hwir_hypotheses / hwir.py.
KIND_COEFF: dict[str, dict[str, int]] = {
    "GEMV": {"dsp": 1, "lut": 64, "bram": 1, "uram": 0, "ii": 1, "depth": 8, "hbm": 1},
    "MAC": {"dsp": 1, "lut": 32, "bram": 0, "uram": 0, "ii": 1, "depth": 4, "hbm": 0},
    "REDUCE": {"dsp": 1, "lut": 24, "bram": 0, "uram": 0, "ii": 1, "depth": 6, "hbm": 0},
    "GATHER": {"dsp": 0, "lut": 128, "bram": 2, "uram": 0, "ii": 4, "depth": 12, "hbm": 1},
    "SCATTER": {"dsp": 0, "lut": 128, "bram": 2, "uram": 0, "ii": 4, "depth": 12, "hbm": 1},
    "ROUTER": {"dsp": 0, "lut": 160, "bram": 2, "uram": 0, "ii": 8, "depth": 10, "hbm": 0},
    "STATE": {"dsp": 1, "lut": 80, "bram": 0, "uram": 1, "ii": 2, "depth": 10, "hbm": 0},
    "LOOKUP": {"dsp": 0, "lut": 16, "bram": 0, "uram": 2, "ii": 1, "depth": 6, "hbm": 1},
    "EPILOGUE": {"dsp": 0, "lut": 40, "bram": 0, "uram": 0, "ii": 1, "depth": 3, "hbm": 0},
    "CONTROL": {"dsp": 0, "lut": 96, "bram": 1, "uram": 0, "ii": 1, "depth": 2, "hbm": 0},
    "TRANSPORT": {"dsp": 0, "lut": 20, "bram": 0, "uram": 0, "ii": 1, "depth": 4, "hbm": 1},
}

IRREGULAR_KINDS = frozenset({"GATHER", "SCATTER", "ROUTER"})
RESOURCE_CLASSES = frozenset({"DSP", "LUT", "BRAM", "URAM", "HBM", "CONTROL"})


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    width: int
    tile: int
    banking: int
    resource_class: str

    def __post_init__(self) -> None:
        if not self.id:
            raise GraphError("node id is empty")
        if self.kind not in KIND_COEFF:
            raise GraphError(f"unknown kind {self.kind!r}; local catalog pending hwir.py")
        if self.resource_class not in RESOURCE_CLASSES:
            raise GraphError(f"unknown resource_class {self.resource_class!r}")
        if self.width < 1 or self.tile < 1 or self.banking < 1:
            raise GraphError(f"width/tile/banking must be >= 1 on node {self.id!r}")


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    frame: int

    def __post_init__(self) -> None:
        if not self.src or not self.dst:
            raise GraphError("edge src/dst is empty")
        if self.frame < 0:
            raise GraphError("edge frame must be >= 0 (bytes per issue, structural)")


@dataclass(frozen=True)
class ResourceEnvelope:
    """Caller-declared planning budget. Not a board measurement."""

    dsp: int
    lut: int
    bram: int
    uram: int
    hbm_channels: int
    origin: str = "CALLER_DECLARED_BUDGET_NOT_A_BOARD_MEASUREMENT"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dsp": self.dsp,
            "lut": self.lut,
            "bram": self.bram,
            "uram": self.uram,
            "hbm_channels": self.hbm_channels,
            "origin": self.origin,
        }


@dataclass(frozen=True)
class StructuralGraph:
    """Minimal HWIR-shaped graph. Integration point: tools/future/hwir.py."""

    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    adaptation_clock: int
    graph_id: str = "unnamed"
    issue_count: int = 1
    schedule_id: str = "default"
    overlay_id: str = "default"
    architecture_id: str = "unselected"
    resource_envelope: ResourceEnvelope | None = None
    token_sources: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "edges", tuple(self.edges))
        if self.token_sources is not None:
            object.__setattr__(self, "token_sources", tuple(self.token_sources))
        self.validate()

    def validate(self) -> None:
        if self.adaptation_clock not in {0, 1, 2, 3}:
            raise GraphError(
                f"graph {self.graph_id!r} must declare adaptation_clock in 0..3, "
                f"got {self.adaptation_clock!r}"
            )
        if self.issue_count < 1:
            raise GraphError("issue_count must be >= 1")
        if not self.nodes:
            raise GraphError(f"graph {self.graph_id!r} has no nodes")
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise GraphError(f"graph {self.graph_id!r} has duplicate node ids")
        known = set(ids)
        for e in self.edges:
            if e.src not in known or e.dst not in known:
                raise GraphError(
                    f"edge {e.src}->{e.dst} references a missing node in {self.graph_id!r}"
                )
        if self.token_sources is not None:
            if not self.token_sources:
                raise GraphError(f"graph {self.graph_id!r} token_sources is empty")
            for src in self.token_sources:
                if src not in known:
                    raise GraphError(
                        f"token source {src!r} is not a node in {self.graph_id!r}"
                    )

    def node_map(self) -> dict[str, Node]:
        return {n.id: n for n in self.nodes}

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "adaptation_clock": self.adaptation_clock,
            "issue_count": self.issue_count,
            "schedule_id": self.schedule_id,
            "overlay_id": self.overlay_id,
            "architecture_id": self.architecture_id,
            "token_sources": None if self.token_sources is None else list(self.token_sources),
            "resource_envelope": None if self.resource_envelope is None else self.resource_envelope.to_dict(),
            "nodes": [
                {
                    "id": n.id,
                    "kind": n.kind,
                    "width": n.width,
                    "tile": n.tile,
                    "banking": n.banking,
                    "resource_class": n.resource_class,
                }
                for n in sorted(self.nodes, key=lambda x: x.id)
            ],
            "edges": [
                {"src": e.src, "dst": e.dst, "frame": e.frame}
                for e in sorted(self.edges, key=lambda x: (x.src, x.dst, x.frame))
            ],
            "note": (
                "Local structural stand-in. Swap for tools/future/hwir.py "
                "(nodes: kind, width, tile, banking, resource_class; edges: frame)."
            ),
        }


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj).encode()).hexdigest()


def _ilog2(n: int) -> int:
    return max(0, int(n).bit_length() - 1)


def _ceil_div(n: int, d: int) -> int:
    if d <= 0:
        raise ValueError("divisor must be positive")
    return (int(n) + d - 1) // d if n > 0 else 0


def _node_resources(node: Node) -> dict[str, int]:
    c = KIND_COEFF[node.kind]
    dsp = c["dsp"] * node.width * node.tile
    lut = c["lut"] * node.tile * node.banking * max(1, _ilog2(node.width) + 1)
    bram = c["bram"] * node.tile * node.banking
    uram = c["uram"] * node.tile
    if c["hbm"]:
        hbm = c["hbm"] * node.banking
    elif node.resource_class == "HBM":
        hbm = max(1, node.banking)
    else:
        hbm = 0
    return {
        "dsp": dsp,
        "lut": lut,
        "bram": bram,
        "uram": uram,
        "hbm_channels": hbm,
    }


def _node_ii(node: Node) -> int:
    c = KIND_COEFF[node.kind]
    if node.kind in IRREGULAR_KINDS:
        return c["ii"] * node.banking
    return c["ii"]


def _node_depth(node: Node) -> int:
    return KIND_COEFF[node.kind]["depth"] + _ilog2(node.width)


def _pressure(res: Mapping[str, int]) -> int:
    return (
        res["dsp"] * PRESSURE_WEIGHT["dsp"]
        + res["lut"] * PRESSURE_WEIGHT["lut"]
        + res["bram"] * PRESSURE_WEIGHT["bram"]
        + res["uram"] * PRESSURE_WEIGHT["uram"]
        + res["hbm_channels"] * PRESSURE_WEIGHT["hbm_channels"]
    )


def _rank(pairs: Sequence[tuple[str, int]]) -> list[str]:
    """Higher score first; ties broken by node id."""
    return [nid for nid, _ in sorted(pairs, key=lambda p: (-p[1], p[0]))]


def _predecessors(graph: StructuralGraph) -> dict[str, list[Edge]]:
    pred: dict[str, list[Edge]] = {n.id: [] for n in graph.nodes}
    for e in graph.edges:
        pred[e.dst].append(e)
    for nid in pred:
        pred[nid] = sorted(pred[nid], key=lambda e: (e.src, e.frame))
    return pred


def _successors(graph: StructuralGraph) -> dict[str, list[str]]:
    succ: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
    for e in graph.edges:
        succ[e.src].append(e.dst)
    for nid in succ:
        succ[nid] = sorted(set(succ[nid]))
    return succ


def topo_ids(graph: StructuralGraph) -> list[str] | None:
    """Deterministic Kahn sort. None if the graph is cyclic."""
    succ = _successors(graph)
    indeg = {n.id: 0 for n in graph.nodes}
    for e in graph.edges:
        indeg[e.dst] += 1
    ready = sorted(nid for nid, d in indeg.items() if d == 0)
    out: list[str] = []
    while ready:
        nid = ready.pop(0)
        out.append(nid)
        nxt = []
        for v in succ[nid]:
            indeg[v] -= 1
            if indeg[v] == 0:
                nxt.append(v)
        ready = sorted(ready + nxt)
    if len(out) != len(graph.nodes):
        return None
    return out


def _sources(graph: StructuralGraph) -> list[str]:
    if graph.token_sources is not None:
        return sorted(graph.token_sources)
    incoming = {n.id: 0 for n in graph.nodes}
    for e in graph.edges:
        incoming[e.dst] += 1
    return sorted(nid for nid, d in incoming.items() if d == 0)


def _reachable(graph: StructuralGraph) -> set[str]:
    succ = _successors(graph)
    seen: set[str] = set()
    stack = list(_sources(graph))
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        stack.extend(reversed(succ[nid]))
    return seen


def _module_structure(graph: StructuralGraph) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": n.id,
                "kind": n.kind,
                "width": n.width,
                "tile": n.tile,
                "banking": n.banking,
                "resource_class": n.resource_class,
            }
            for n in sorted(graph.nodes, key=lambda x: x.id)
        ],
        "edges": [
            {"src": e.src, "dst": e.dst, "frame": e.frame}
            for e in sorted(graph.edges, key=lambda x: (x.src, x.dst, x.frame))
        ],
        "architecture_id": graph.architecture_id,
    }


def _fabric_structure(graph: StructuralGraph) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": n.id,
                "kind": n.kind,
                "tile": n.tile,
                "banking": n.banking,
                "resource_class": n.resource_class,
            }
            for n in sorted(graph.nodes, key=lambda x: x.id)
        ],
        "edges": [
            {"src": e.src, "dst": e.dst}
            for e in sorted(graph.edges, key=lambda x: (x.src, x.dst))
        ],
        "architecture_id": graph.architecture_id,
    }


def module_cache_key(graph: StructuralGraph) -> str:
    return _sha(_module_structure(graph))


def _transfer_cycles(edge: Edge) -> int:
    if edge.frame <= 0:
        return 0
    return _ceil_div(edge.frame, MODEL_BYTES_PER_CYCLE)


def _own_cycle_cost(node: Node, issue_count: int) -> int:
    return _node_depth(node) + _node_ii(node) * (issue_count - 1)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class FidelityProvider:
    level: FidelityLevel
    available: bool = True
    missing_dependency: str | None = None
    status: str = "AVAILABLE"

    def identity(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "available": self.available,
            "missing_dependency": self.missing_dependency,
            "status": self.status,
        }

    def estimate(self, graph: StructuralGraph) -> dict[str, Any]:
        graph.validate()
        if not self.available:
            raise ProviderUnavailable(
                self.level,
                self.missing_dependency or "UNIMPLEMENTED",
            )
        return self._run(graph)

    def _run(self, graph: StructuralGraph) -> dict[str, Any]:
        raise NotImplementedError


class AnalyticalProvider(FidelityProvider):
    level = FidelityLevel.ANALYTICAL
    available = True
    missing_dependency = None
    status = "AVAILABLE"

    def _run(self, graph: StructuralGraph) -> dict[str, Any]:
        per_node = []
        totals = {"dsp": 0, "lut": 0, "bram": 0, "uram": 0, "hbm_channels": 0}
        ranking_pairs: list[tuple[str, int]] = []
        for node in sorted(graph.nodes, key=lambda n: n.id):
            res = _node_resources(node)
            for k in totals:
                totals[k] += res[k]
            pressure = _pressure(res)
            ranking_pairs.append((node.id, pressure))
            per_node.append(
                {
                    "id": node.id,
                    "kind": node.kind,
                    "resource_class": node.resource_class,
                    "width": node.width,
                    "tile": node.tile,
                    "banking": node.banking,
                    "dsp": res["dsp"],
                    "lut": res["lut"],
                    "bram": res["bram"],
                    "uram": res["uram"],
                    "hbm_channel_demand": res["hbm_channels"],
                    "ii": _node_ii(node),
                    "pipeline_depth": _node_depth(node),
                    "pressure": pressure,
                }
            )
        bytes_in_flight = sum(e.frame for e in graph.edges) * graph.issue_count
        envelope = graph.resource_envelope
        overflow: list[str] = []
        if envelope is None:
            feasibility = "UNKNOWN"
            feasibility_reason = (
                "no target device selected; FPGADeviceGenome is TARGET_UNSELECTED "
                "and no caller-declared resource_envelope was supplied"
            )
        else:
            env = envelope.to_dict()
            for key, graph_key in (
                ("dsp", "dsp"),
                ("lut", "lut"),
                ("bram", "bram"),
                ("uram", "uram"),
                ("hbm_channels", "hbm_channels"),
            ):
                if totals[graph_key] > int(env[key]):
                    overflow.append(f"{key}:{totals[graph_key]}>{env[key]}")
            feasibility = "INFEASIBLE" if overflow else "FEASIBLE"
            feasibility_reason = (
                "exceeds caller-declared envelope: " + ",".join(overflow)
                if overflow
                else "within caller-declared envelope (not a board measurement)"
            )
        return {
            "level": self.level.value,
            "available": True,
            "status": "AVAILABLE",
            "tag": "STRUCTURAL_COUNTS",
            "evidence_class": "STATIC_ONLY",
            "measurement_state": "STATIC_ONLY",
            "coefficient_table": COEFFICIENT_TABLE,
            "coefficient_honesty": (
                "Counts are shapes times an assumed coefficient table. They are "
                "not a synthesis report, not a place-and-route result, and not "
                "a board measurement. Match tools/accelerator/perf_model.py: "
                "unknowns stay unknown; the model is graded on ranking, not on "
                "invented wall time."
            ),
            "resources": totals,
            "per_node": per_node,
            "ii_by_node": {row["id"]: row["ii"] for row in per_node},
            "pipeline_depth_by_node": {row["id"]: row["pipeline_depth"] for row in per_node},
            "hbm_channel_demand": totals["hbm_channels"],
            "bytes_in_flight": bytes_in_flight,
            "bytes_in_flight_rule": "sum(edge.frame) * issue_count; frame is structural bytes/issue",
            "node_ranking": _rank(ranking_pairs),
            "score": _pressure(totals),
            "score_meaning": "weighted resource pressure; lower is cheaper",
            "feasibility": feasibility,
            "feasibility_reason": feasibility_reason,
            "envelope_overflow": overflow,
            "seconds": None,
        }


class FunctionalSimProvider(FidelityProvider):
    level = FidelityLevel.FUNCTIONAL_SIM
    available = True
    missing_dependency = None
    status = "AVAILABLE"

    def _run(self, graph: StructuralGraph) -> dict[str, Any]:
        errors: list[str] = []
        order = topo_ids(graph)
        if order is None:
            errors.append("cyclic graph; functional token flow is undefined")
        live = _reachable(graph) if order is not None else set()
        all_ids = {n.id for n in graph.nodes}
        dead = sorted(all_ids - live)
        nmap = graph.node_map()
        ranking_pairs: list[tuple[str, int]] = []
        live_volume = 0
        produced: list[tuple[str, int]] = []
        if order is not None:
            for nid in sorted(all_ids):
                node = nmap[nid]
                vol = node.width * node.tile if nid in live else 0
                ranking_pairs.append((nid, vol))
                if nid in live:
                    live_volume += vol
                    produced.append((nid, vol))
        functional_ok = not errors
        digest = _sha({"live": produced, "errors": errors}) if functional_ok else None
        return {
            "level": self.level.value,
            "available": True,
            "status": "AVAILABLE",
            "tag": "FUNCTIONAL_INTERPRETER",
            "evidence_class": "STATIC_ONLY",
            "measurement_state": "STATIC_ONLY",
            "functional_ok": functional_ok,
            "errors": errors,
            "topo_order": order,
            "live_nodes": sorted(live),
            "dead_nodes": dead,
            "op_volume": live_volume,
            "functional_digest": digest,
            "node_ranking": _rank(ranking_pairs) if ranking_pairs else [],
            "score": live_volume if functional_ok else 10**12,
            "score_meaning": "live width*tile volume; ill-formed graphs score as losers",
            "seconds": None,
        }


class CycleModelProvider(FidelityProvider):
    level = FidelityLevel.CYCLE_MODEL
    available = True
    missing_dependency = None
    status = "AVAILABLE"

    def _run(self, graph: StructuralGraph) -> dict[str, Any]:
        order = topo_ids(graph)
        if order is None:
            # A cycle model over a cyclic graph is undefined. Not a guess.
            raise GraphError(
                f"CYCLE_MODEL refuses cyclic graph {graph.graph_id!r}; "
                "no modelled_cycles are emitted"
            )
        nmap = graph.node_map()
        pred = _predecessors(graph)
        start: dict[str, int] = {}
        finish: dict[str, int] = {}
        per_node = []
        ranking_pairs: list[tuple[str, int]] = []
        for nid in order:
            node = nmap[nid]
            ready = 0
            for e in pred[nid]:
                ready = max(ready, finish[e.src] + _transfer_cycles(e))
            own = _own_cycle_cost(node, graph.issue_count)
            start[nid] = ready
            finish[nid] = ready + own
            ranking_pairs.append((nid, own))
            per_node.append(
                {
                    "id": nid,
                    "kind": node.kind,
                    "ii": _node_ii(node),
                    "pipeline_depth": _node_depth(node),
                    "own_cost_cycles": own,
                    "start_cycle": start[nid],
                    "finish_cycle": finish[nid],
                    "tag": "MODELLED_NOT_MEASURED",
                }
            )
        modelled = max(finish.values()) if finish else 0
        return {
            "level": self.level.value,
            "available": True,
            "status": "AVAILABLE",
            "tag": "MODELLED_NOT_MEASURED",
            "evidence_class": "STATIC_ONLY",
            "measurement_state": "STATIC_ONLY",
            "modelled_cycles": modelled,
            "seconds": None,
            "clock_hz": "UNKNOWN",
            "conversion_to_seconds": "REFUSED",
            "conversion_reason": (
                "a cycle count is not a duration without a real clock; this "
                "host has no FPGA and no emulation seat"
            ),
            "model_bytes_per_cycle": MODEL_BYTES_PER_CYCLE,
            "model_bytes_per_cycle_meaning": (
                "assumed fabric beat for frame->cycle conversion; not a measured "
                "clock, not HBM bandwidth, never reported as seconds"
            ),
            "issue_count": graph.issue_count,
            "per_node": per_node,
            "node_ranking": _rank(ranking_pairs),
            "score": modelled,
            "score_meaning": "modelled critical-path cycles; lower is cheaper; MODELLED_NOT_MEASURED",
        }


class HwEmulationProvider(FidelityProvider):
    level = FidelityLevel.HW_EMULATION
    available = False
    missing_dependency = "no emulation seat"
    status = "UNIMPLEMENTED"

    def _run(self, graph: StructuralGraph) -> dict[str, Any]:
        raise ProviderUnavailable(self.level, self.missing_dependency or "no emulation seat")


class RealHardwareProvider(FidelityProvider):
    level = FidelityLevel.REAL_HARDWARE
    available = False
    missing_dependency = "no U50 board"
    status = "UNIMPLEMENTED"

    def _run(self, graph: StructuralGraph) -> dict[str, Any]:
        raise ProviderUnavailable(self.level, self.missing_dependency or "no U50 board")


LADDER: tuple[FidelityProvider, ...] = (
    AnalyticalProvider(),
    FunctionalSimProvider(),
    CycleModelProvider(),
    HwEmulationProvider(),
    RealHardwareProvider(),
)

_PROVIDERS: dict[FidelityLevel, FidelityProvider] = {p.level: p for p in LADDER}


def get_provider(level: FidelityLevel | str) -> FidelityProvider:
    if not isinstance(level, FidelityLevel):
        level = FidelityLevel(level)
    return _PROVIDERS[level]


def estimate(level: FidelityLevel | str, graph: StructuralGraph) -> dict[str, Any]:
    """Run one fidelity provider. Unavailable providers raise, never guess."""
    return get_provider(level).estimate(graph)


def modelled_cycles_to_seconds(cycles: int, clock_hz: Any = None) -> None:
    """Refused. A modelled cycle is not a duration."""
    raise UnmeasuredConversionError(
        "CYCLE_MODEL cycles are MODELLED_NOT_MEASURED and cannot be converted "
        f"to seconds (cycles={cycles!r}, clock_hz={clock_hz!r}); no real clock"
    )


# ---------------------------------------------------------------------------
# Disagreement as evidence
# ---------------------------------------------------------------------------


def _node_meta(graph: StructuralGraph, nid: str | None) -> dict[str, Any] | None:
    if nid is None:
        return None
    n = graph.node_map().get(nid)
    if n is None:
        return None
    return {"id": n.id, "kind": n.kind, "resource_class": n.resource_class}


def _hypothesis(
    level_a: FidelityLevel,
    level_b: FidelityLevel,
    ranking_a: Sequence[str],
    ranking_b: Sequence[str],
    graph: StructuralGraph,
) -> str:
    top_a = ranking_a[0] if ranking_a else None
    top_b = ranking_b[0] if ranking_b else None
    if ranking_a == list(ranking_b):
        return (
            f"{level_a.value} and {level_b.value} agree on node ranking "
            f"{list(ranking_a)}. Agreement is weak evidence: both models can "
            "share the same blind spot."
        )
    ma = _node_meta(graph, top_a)
    mb = _node_meta(graph, top_b)
    pair = {level_a, level_b}
    if pair == {FidelityLevel.ANALYTICAL, FidelityLevel.CYCLE_MODEL}:
        why = (
            "analytical scores static occupancy (DSP/LUT/BRAM/URAM/HBM demand); "
            "the cycle model scores initiation interval and critical-path issue. "
            "A DSP-cheap irregular node can be a cycle-expensive bottleneck."
        )
    elif FidelityLevel.FUNCTIONAL_SIM in pair:
        why = (
            "functional_sim costs only live dataflow (unreachable nodes contribute "
            "zero); the other level still costs static occupancy or issue. Dead "
            "nodes are a finding, not noise."
        )
    else:
        why = "the two levels induce different bottleneck rankings over the same graph."
    return (
        f"{level_a.value} ranks {top_a} ({ma}) as bottleneck; "
        f"{level_b.value} ranks {top_b} ({mb}). {why} "
        "A cheap-level ranking that an expensive level would reverse must not promote."
    )


def compare(
    level_a: FidelityLevel | str,
    level_b: FidelityLevel | str,
    graph: StructuralGraph,
) -> dict[str, Any]:
    """Compare two levels on one graph. Ranking disagreement is a FINDING."""
    la = level_a if isinstance(level_a, FidelityLevel) else FidelityLevel(level_a)
    lb = level_b if isinstance(level_b, FidelityLevel) else FidelityLevel(level_b)
    report_a = estimate(la, graph)
    report_b = estimate(lb, graph)
    ranking_a = list(report_a["node_ranking"])
    ranking_b = list(report_b["node_ranking"])
    agree = ranking_a == ranking_b
    return {
        "graph_id": graph.graph_id,
        "level_a": la.value,
        "level_b": lb.value,
        "agree": agree,
        "kind": "AGREEMENT" if agree else "DISAGREEMENT",
        "evidence_weight": "WEAK" if agree else "FINDING",
        "evidence_class": "STATIC_ONLY",
        "node_ranking_a": ranking_a,
        "node_ranking_b": ranking_b,
        "top_a": ranking_a[0] if ranking_a else None,
        "top_b": ranking_b[0] if ranking_b else None,
        "score_a": report_a["score"],
        "score_b": report_b["score"],
        "tag_a": report_a.get("tag"),
        "tag_b": report_b.get("tag"),
        "hypothesis": _hypothesis(la, lb, ranking_a, ranking_b, graph),
        "promotion_rule": (
            "Agreement does not license promotion. Disagreement is first-class "
            "evidence and blocks treating the cheaper level as decisive."
        ),
    }


def rank_graphs(level: FidelityLevel | str, graphs: Sequence[StructuralGraph]) -> list[str]:
    """Cheapest first at this level. Unavailable levels raise."""
    scored = []
    for g in graphs:
        report = estimate(level, g)
        scored.append((report["score"], g.graph_id))
    scored.sort(key=lambda p: (p[0], p[1]))
    return [gid for _, gid in scored]


def search_ladder(
    graphs: Sequence[StructuralGraph],
    *,
    stop_before_unavailable: bool = True,
) -> dict[str, Any]:
    """Run cheap available levels in order. Hard-kill only infeasibility / ill-formed.

    A worse score is not a kill. Ranking disagreement across survivors is a finding.
    HW_EMULATION and REAL_HARDWARE are never called when stop_before_unavailable.
    """
    remaining = list(graphs)
    killed: list[dict[str, Any]] = []
    ranking_by_level: dict[str, list[str]] = {}
    skipped: list[dict[str, Any]] = []
    reports: dict[str, dict[str, dict[str, Any]]] = {}

    for provider in LADDER:
        if not provider.available:
            skipped.append(provider.identity())
            if stop_before_unavailable:
                continue
            if remaining:
                provider.estimate(remaining[0])
            continue
        still: list[StructuralGraph] = []
        for g in remaining:
            report = provider.estimate(g)
            reports.setdefault(provider.level.value, {})[g.graph_id] = {
                "score": report["score"],
                "node_ranking": report["node_ranking"],
                "tag": report.get("tag"),
                "feasibility": report.get("feasibility"),
                "functional_ok": report.get("functional_ok"),
            }
            reason = None
            if provider.level is FidelityLevel.ANALYTICAL and report.get("feasibility") == "INFEASIBLE":
                reason = "INFEASIBLE_ENVELOPE"
            elif provider.level is FidelityLevel.FUNCTIONAL_SIM and not report.get("functional_ok"):
                reason = "FUNCTIONAL_ILL_FORMED"
            if reason:
                killed.append(
                    {
                        "graph_id": g.graph_id,
                        "killed_at_level": provider.level.value,
                        "reason": reason,
                    }
                )
            else:
                still.append(g)
        remaining = still
        if remaining:
            ranking_by_level[provider.level.value] = rank_graphs(provider.level, remaining)
        else:
            ranking_by_level[provider.level.value] = []

    highest = None
    for level in LADDER_ORDER:
        if get_provider(level).available and level.value in ranking_by_level:
            highest = level.value

    return {
        "survivors": [g.graph_id for g in remaining],
        "killed": killed,
        "ranking_by_level": ranking_by_level,
        "reports": reports,
        "highest_available_level": highest,
        "skipped_unimplemented": skipped,
        "kill_rule": (
            "Hard-kill only envelope overflow (analytical) and ill-formed "
            "functional graphs. Worse-score is a ranking, not a kill, because "
            "a cheap level can rank a later winner as a loser."
        ),
        "evidence_class": "STATIC_ONLY",
    }


# ---------------------------------------------------------------------------
# Four adaptation clocks
# ---------------------------------------------------------------------------


CLOCK_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": 0,
        "name": "CLOCK_0_RUNTIME_SCHEDULING",
        "identity": "runtime_scheduler",
        "compatibility_predicate": "same_module_bitstream",
        "load_switch_cost_class": "CHEAP",
        "live_switch_legal": True,
        "module_cache_key_rule": "not a bitstream; keyed by schedule_id under a frozen module",
        "resource_footprint_rule": "zero extra fabric; uses the already-loaded module",
    },
    {
        "id": 1,
        "name": "CLOCK_1_OVERLAY_MICROCODE",
        "identity": "overlay_microcode",
        "compatibility_predicate": "same_overlay_fabric",
        "load_switch_cost_class": "MODERATE",
        "live_switch_legal": True,
        "module_cache_key_rule": "overlay_id plus fabric geometry (kind/tile/banking/resource_class)",
        "resource_footprint_rule": "instruction/overlay memory only; fabric geometry frozen",
    },
    {
        "id": 2,
        "name": "CLOCK_2_CACHED_MODULE_DFX",
        "identity": "cached_module_dfx",
        "compatibility_predicate": "same_static_shell",
        "load_switch_cost_class": "EXPENSIVE",
        "live_switch_legal": True,
        "module_cache_key_rule": "sha256 of canonical module structure (kind/width/tile/banking/edges)",
        "resource_footprint_rule": "DFX region equals the module's analytical resource counts",
    },
    {
        "id": 3,
        "name": "CLOCK_3_DEEP_ARCHITECTURE",
        "identity": "deep_architecture_evolution",
        "compatibility_predicate": "never_a_live_switch",
        "load_switch_cost_class": "PROHIBITIVE",
        "live_switch_legal": False,
        "module_cache_key_rule": "architecture_id; a new generation, not a loadable module",
        "resource_footprint_rule": "whole-device; UNKNOWN without a selected target",
    },
)


def clock_compatible(parent: StructuralGraph, child: StructuralGraph) -> bool:
    """Apply the child's declared clock as the switch class from parent -> child."""
    clock = child.adaptation_clock
    if clock == 0:
        return _sha(_module_structure(parent)) == _sha(_module_structure(child))
    if clock == 1:
        return _sha(_fabric_structure(parent)) == _sha(_fabric_structure(child))
    if clock == 2:
        return parent.architecture_id == child.architecture_id
    if clock == 3:
        return False
    raise GraphError(f"unknown adaptation_clock {clock}")


def _cache_key_for_clock(clock: int, graph: StructuralGraph) -> str:
    if clock == 0:
        return _sha({"kind": "schedule", "schedule_id": graph.schedule_id, "module": _module_structure(graph)})
    if clock == 1:
        return _sha({"kind": "overlay", "overlay_id": graph.overlay_id, "fabric": _fabric_structure(graph)})
    if clock == 2:
        return module_cache_key(graph)
    if clock == 3:
        return _sha({"kind": "architecture", "architecture_id": graph.architecture_id})
    raise GraphError(f"unknown adaptation_clock {clock}")


def _footprint_for_clock(clock: int, graph: StructuralGraph) -> dict[str, Any]:
    if clock == 0:
        return {
            "dsp": 0,
            "lut": 0,
            "bram": 0,
            "uram": 0,
            "hbm_channels": 0,
            "note": "runtime scheduling adds no fabric",
        }
    if clock == 1:
        return {
            "dsp": 0,
            "lut": 256,
            "bram": 1,
            "uram": 0,
            "hbm_channels": 0,
            "note": "assumed overlay instruction memory; ASSUMED_NOT_MEASURED",
            "tag": "ASSUMED_NOT_MEASURED",
        }
    if clock == 2:
        res = estimate(FidelityLevel.ANALYTICAL, graph)["resources"]
        return {
            **res,
            "note": "DFX region footprint = module analytical counts; STRUCTURAL_COUNTS",
            "tag": "STRUCTURAL_COUNTS",
        }
    return {
        "dsp": "UNKNOWN",
        "lut": "UNKNOWN",
        "bram": "UNKNOWN",
        "uram": "UNKNOWN",
        "hbm_channels": "UNKNOWN",
        "note": "no target device selected; whole-device footprint is UNKNOWN",
        "tag": "UNKNOWN",
    }


def clock_state(graph: StructuralGraph) -> dict[str, Any]:
    graph.validate()
    clock = graph.adaptation_clock
    spec = CLOCK_SPECS[clock]
    return {
        **spec,
        "declared_on_graph": clock,
        "graph_id": graph.graph_id,
        "module_cache_key": _cache_key_for_clock(clock, graph),
        "resource_footprint": _footprint_for_clock(clock, graph),
        "compatible_with_self": clock_compatible(graph, graph) if clock != 3 else False,
        "evidence_class": "STATIC_ONLY",
    }


# ---------------------------------------------------------------------------
# Fixtures used by selftest / receipt
# ---------------------------------------------------------------------------


def gemv_graph(*, graph_id: str = "gemv_heavy", clock: int = 2) -> StructuralGraph:
    return StructuralGraph(
        graph_id=graph_id,
        adaptation_clock=clock,
        issue_count=16,
        architecture_id="unselected-shell",
        nodes=(
            Node(id="gemv0", kind="GEMV", width=64, tile=4, banking=1, resource_class="DSP"),
        ),
        edges=(),
    )


def gather_graph(*, graph_id: str = "gather_heavy", clock: int = 2) -> StructuralGraph:
    return StructuralGraph(
        graph_id=graph_id,
        adaptation_clock=clock,
        issue_count=16,
        architecture_id="unselected-shell",
        nodes=(
            Node(id="gather0", kind="GATHER", width=16, tile=1, banking=8, resource_class="LUT"),
        ),
        edges=(),
    )


def mixed_graph(*, graph_id: str = "mixed_gemv_gather", clock: int = 2) -> StructuralGraph:
    return StructuralGraph(
        graph_id=graph_id,
        adaptation_clock=clock,
        issue_count=16,
        architecture_id="unselected-shell",
        nodes=(
            Node(id="gemv0", kind="GEMV", width=64, tile=4, banking=1, resource_class="DSP"),
            Node(id="gather0", kind="GATHER", width=16, tile=1, banking=8, resource_class="LUT"),
        ),
        edges=(Edge(src="gemv0", dst="gather0", frame=256),),
    )


def dead_node_graph() -> StructuralGraph:
    # token_sources restricts injection so dead0 is an island, not a source.
    return StructuralGraph(
        graph_id="live_plus_dead",
        adaptation_clock=2,
        issue_count=4,
        architecture_id="unselected-shell",
        token_sources=("live0",),
        nodes=(
            Node(id="live0", kind="MAC", width=8, tile=1, banking=1, resource_class="DSP"),
            Node(id="dead0", kind="GEMV", width=64, tile=8, banking=1, resource_class="DSP"),
        ),
        edges=(),
    )


def cyclic_graph() -> StructuralGraph:
    return StructuralGraph(
        graph_id="cyclic",
        adaptation_clock=0,
        issue_count=1,
        nodes=(
            Node(id="a", kind="MAC", width=4, tile=1, banking=1, resource_class="DSP"),
            Node(id="b", kind="MAC", width=4, tile=1, banking=1, resource_class="DSP"),
        ),
        edges=(Edge(src="a", dst="b", frame=8), Edge(src="b", dst="a", frame=8)),
    )


def overflow_graph() -> StructuralGraph:
    return StructuralGraph(
        graph_id="envelope_overflow",
        adaptation_clock=2,
        issue_count=1,
        architecture_id="unselected-shell",
        resource_envelope=ResourceEnvelope(dsp=1, lut=1, bram=0, uram=0, hbm_channels=0),
        nodes=(
            Node(id="fat", kind="GEMV", width=32, tile=4, banking=2, resource_class="DSP"),
        ),
        edges=(),
    )


# ---------------------------------------------------------------------------
# Recovery, receipt, CLI
# ---------------------------------------------------------------------------


def _in_git_head(rel: str) -> bool:
    return bool(git("ls-tree", "--name-only", "HEAD", rel))


def _recovery_row(path: str, role: str, what_found: str) -> dict[str, Any]:
    p = REPO / path
    loaded = None
    if p.exists():
        try:
            loaded = sorted(load_json(p).keys())[:12]
        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
            loaded = ["<unreadable>"]
    return {
        "path": path,
        "role": role,
        "on_disk": p.exists(),
        "in_git_head": _in_git_head(path),
        "what_found": what_found,
        "loaded_keys": loaded,
    }


def recovered_implementation() -> list[dict[str, Any]]:
    return [
        _recovery_row(
            "hcli/agentos/fpga_preboard.py",
            "FPGA pre-board contracts, MockFPGAProvider, organ-map HWIR, link/partition [S] sim",
            "FPGAProvider.execute raises; MockFPGAProvider is SIMULATOR_ONLY; "
            "device genome TARGET_UNSELECTED; physical_board_present False; "
            "HWIR dataclass (nodes/buffers/dependencies/sync/placements); "
            "module_cache schema-only. This is scaffolding, not a fidelity ladder. "
            "Not imported: hcli is Codex-owned.",
        ),
        _recovery_row(
            "receipts/headless/FLASH_NEXT_FPGA_ORGAN_MAP.json",
            "Flash organ map + derived HWIR",
            "7 organs (expert_bank, router_topk_and_gather, routed_plus_shared_expert, "
            "deltanet_persistent_state, ngram_lookup_or_generator, sparse_attention, "
            "mtp_draft_verify_rollback); nodes are organ_operator without width/tile/"
            "banking/resource_class; no DSP/LUT/BRAM/URAM counts.",
        ),
        _recovery_row(
            "receipts/headless/QWEN27_FPGA_ORGAN_MAP.json",
            "Qwen27 organ map + derived HWIR",
            "6 organs (mlp_gate_up_down, gqa_qkv_and_output, "
            "deltanet_state_and_input_projection, norm_add_epilogues, "
            "lm_head_and_sampling, command_buffer_graph); same structural gap.",
        ),
        _recovery_row(
            "receipts/headless/HCLI_FPGA_PREBOARD.json",
            "Preboard receipt",
            "physical_board ABSENT; fpga_backend NOT_BUILT; mock provider simulation-only.",
        ),
        _recovery_row(
            "tools/accelerator/perf_model.py",
            "Existing cost model (honesty template)",
            "Ridge on 4 features; grades the occupancy cliff separately from the flat "
            "region; refuses to let easy points hide the hard one; unknowns stay unknown. "
            "It is a GPU occupancy model, not an FPGA fidelity ladder.",
        ),
        _recovery_row(
            "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json",
            "Claimed home of 15 hwir_hypotheses (frontier F003/F012)",
            "NOT in git HEAD and not on disk in this sparse worktree. Node kinds for "
            "the estimator could not be read from the atlas.",
        ),
        _recovery_row(
            "hcli/physical_graph.py",
            "Provider-neutral PhysicalGraph (F003 prerequisite)",
            "PLAN_ONLY placement/dataflow; computation nodes kind=computation; "
            "no FPGA resource fields. Not imported.",
        ),
        _recovery_row(
            "hcli/agentos/test_fpga_preboard.py",
            "Listed in the lane contract as a recovery target",
            "Path does not exist in git HEAD.",
        ),
        _recovery_row(
            "tools/future/fpga_fidelity.py",
            "This module",
            "Did not exist before this lane; no prior multi-fidelity FPGA estimator.",
        ),
        _recovery_row(
            "tools/future/hwir.py",
            "Sibling-lane HWIR (do not import, do not write)",
            "Frontier F003 still MISSING at recovery time; local StructuralGraph is the stand-in.",
        ),
    ]


def _run_invariants() -> None:
    mixed = mixed_graph()
    gemv = gemv_graph()
    gather = gather_graph()
    dead = dead_node_graph()
    cyclic = cyclic_graph()
    overflow = overflow_graph()

    levels = [p.level for p in LADDER]
    if levels != list(LADDER_ORDER):
        raise AssertionError(f"ladder order drifted: {levels}")
    if LADDER[0].available is not True or LADDER[1].available is not True or LADDER[2].available is not True:
        raise AssertionError("cheap levels must be available")
    if LADDER[3].available or LADDER[4].available:
        raise AssertionError("HW_EMULATION/REAL_HARDWARE must not be available")
    if LADDER[3].missing_dependency != "no emulation seat":
        raise AssertionError(LADDER[3].missing_dependency)
    if LADDER[4].missing_dependency != "no U50 board":
        raise AssertionError(LADDER[4].missing_dependency)

    for bad in (FidelityLevel.HW_EMULATION, FidelityLevel.REAL_HARDWARE):
        raised = False
        try:
            estimate(bad, gemv)
        except ProviderUnavailable as exc:
            raised = True
            if exc.level is not bad:
                raise AssertionError(exc)
            if not exc.missing_dependency:
                raise AssertionError("missing dependency unnamed")
        if not raised:
            raise AssertionError(f"{bad} returned a guess")

    a = estimate(FidelityLevel.ANALYTICAL, mixed)
    if a["tag"] != "STRUCTURAL_COUNTS":
        raise AssertionError(a["tag"])
    for key in ("dsp", "lut", "bram", "uram"):
        if not isinstance(a["resources"][key], int) or a["resources"][key] < 0:
            raise AssertionError(a["resources"])
    if a["feasibility"] != "UNKNOWN":
        raise AssertionError(a["feasibility"])
    if a["seconds"] is not None:
        raise AssertionError("analytical must not emit seconds")

    cyc = estimate(FidelityLevel.CYCLE_MODEL, mixed)
    if cyc["tag"] != "MODELLED_NOT_MEASURED":
        raise AssertionError(cyc["tag"])
    if not isinstance(cyc["modelled_cycles"], int) or cyc["modelled_cycles"] <= 0:
        raise AssertionError(cyc["modelled_cycles"])
    if cyc["seconds"] is not None or cyc["conversion_to_seconds"] != "REFUSED":
        raise AssertionError(cyc)
    try:
        modelled_cycles_to_seconds(cyc["modelled_cycles"])
    except UnmeasuredConversionError:
        pass
    else:
        raise AssertionError("cycle->seconds must refuse")

    d = compare(FidelityLevel.ANALYTICAL, FidelityLevel.CYCLE_MODEL, mixed)
    if d["kind"] != "DISAGREEMENT" or d["evidence_weight"] != "FINDING":
        raise AssertionError(d)
    if d["top_a"] == d["top_b"]:
        raise AssertionError("mixed graph must disagree on the bottleneck")

    w = compare(FidelityLevel.ANALYTICAL, FidelityLevel.CYCLE_MODEL, gemv)
    if w["kind"] != "AGREEMENT" or w["evidence_weight"] != "WEAK":
        raise AssertionError(w)

    func = estimate(FidelityLevel.FUNCTIONAL_SIM, dead)
    if not func["functional_ok"] or "dead0" not in func["dead_nodes"]:
        raise AssertionError(func)
    dead_cmp = compare(FidelityLevel.ANALYTICAL, FidelityLevel.FUNCTIONAL_SIM, dead)
    if dead_cmp["kind"] != "DISAGREEMENT":
        raise AssertionError(dead_cmp)

    func_cyc = estimate(FidelityLevel.FUNCTIONAL_SIM, cyclic)
    if func_cyc["functional_ok"]:
        raise AssertionError("cyclic graph must fail functional_sim")
    try:
        estimate(FidelityLevel.CYCLE_MODEL, cyclic)
    except GraphError:
        pass
    else:
        raise AssertionError("cycle model must refuse cyclic graphs")

    ladder = search_ladder([gemv, gather, overflow, cyclic])
    killed_ids = {k["graph_id"] for k in ladder["killed"]}
    if "envelope_overflow" not in killed_ids or "cyclic" not in killed_ids:
        raise AssertionError(ladder["killed"])
    if "gemv_heavy" not in ladder["survivors"] or "gather_heavy" not in ladder["survivors"]:
        raise AssertionError(ladder["survivors"])

    a_rank = rank_graphs(FidelityLevel.ANALYTICAL, [gemv, gather])
    c_rank = rank_graphs(FidelityLevel.CYCLE_MODEL, [gemv, gather])
    if a_rank == c_rank:
        raise AssertionError("gemv vs gather must reverse between analytical and cycle")

    cs = clock_state(mixed)
    if cs["id"] != 2 or not cs["module_cache_key"] or "dsp" not in cs["resource_footprint"]:
        raise AssertionError(cs)
    parent = gemv_graph(clock=0)
    child_ok = StructuralGraph(
        graph_id="sched",
        adaptation_clock=0,
        issue_count=parent.issue_count,
        architecture_id=parent.architecture_id,
        schedule_id="alt",
        nodes=parent.nodes,
        edges=parent.edges,
    )
    child_bad = gather_graph(clock=0)
    if not clock_compatible(parent, child_ok):
        raise AssertionError("CLOCK 0 same-module switch must be compatible")
    if clock_compatible(parent, child_bad):
        raise AssertionError("CLOCK 0 across different modules must be incompatible")
    arch = gemv_graph(clock=3)
    if clock_compatible(arch, arch):
        raise AssertionError("CLOCK 3 is never a live switch")


def build() -> Any:
    mixed = mixed_graph()
    gemv = gemv_graph()
    gather = gather_graph()
    dead = dead_node_graph()
    cyclic = cyclic_graph()
    overflow = overflow_graph()

    analytical = estimate(FidelityLevel.ANALYTICAL, mixed)
    functional = estimate(FidelityLevel.FUNCTIONAL_SIM, mixed)
    cycle = estimate(FidelityLevel.CYCLE_MODEL, mixed)
    disagreement = compare(FidelityLevel.ANALYTICAL, FidelityLevel.CYCLE_MODEL, mixed)
    agreement = compare(FidelityLevel.ANALYTICAL, FidelityLevel.CYCLE_MODEL, gemv)
    liveness = compare(FidelityLevel.ANALYTICAL, FidelityLevel.FUNCTIONAL_SIM, dead)
    ladder_run = search_ladder([gemv, gather, overflow, cyclic, dead])

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Multi-fidelity FPGA search ladder so cheap levels can kill losers "
            "before an expensive level runs, and disagreement between levels is "
            "first-class evidence. FPGA remains part of Accelerator / Physical "
            "Compiler / Fusion; this is not an FPGA backend."
        ),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "fidelity_ladder": [
            {
                **p.identity(),
                "order": i,
                "cheaper_than": LADDER_ORDER[i + 1].value if i + 1 < len(LADDER_ORDER) else None,
            }
            for i, p in enumerate(LADDER)
        ],
        "unimplemented_levels": [
            {
                "level": "HW_EMULATION",
                "available": False,
                "status": "UNIMPLEMENTED",
                "missing_dependency": "no emulation seat",
                "rule": "calling this provider raises; it never returns a guess",
            },
            {
                "level": "REAL_HARDWARE",
                "available": False,
                "status": "UNIMPLEMENTED",
                "missing_dependency": "no U50 board",
                "rule": "calling this provider raises; it never returns a guess",
            },
        ],
        "structural_graph": {
            "type": "tools.future.fpga_fidelity.StructuralGraph",
            "fields": {
                "nodes": ["id", "kind", "width", "tile", "banking", "resource_class"],
                "edges": ["src", "dst", "frame"],
                "adaptation_clock": "required, 0..3",
            },
            "kinds": sorted(KIND_COEFF),
            "resource_classes": sorted(RESOURCE_CLASSES),
            "integration_point": (
                "Swap StructuralGraph for tools/future/hwir.py when F003 lands. "
                "Do not import HWIR from this module; a sibling lane owns it."
            ),
            "example": mixed.to_dict(),
        },
        "analytical": analytical,
        "functional_sim": functional,
        "cycle_model": cycle,
        "disagreement": {
            "analytical_vs_cycle_on_mixed": disagreement,
            "analytical_vs_cycle_on_single_gemv": agreement,
            "analytical_vs_functional_on_dead_node": liveness,
            "graph_ranking_reversal": {
                "analytical": rank_graphs(FidelityLevel.ANALYTICAL, [gemv, gather]),
                "cycle_model": rank_graphs(FidelityLevel.CYCLE_MODEL, [gemv, gather]),
                "meaning": (
                    "gather_heavy is cheaper analytically (almost no DSP) and "
                    "more expensive in the cycle model (II scales with banking). "
                    "Killing gemv_heavy at the analytical level would discard "
                    "the cycle-model winner."
                ),
            },
        },
        "search_ladder": ladder_run,
        "adaptation_clocks": [dict(spec) for spec in CLOCK_SPECS],
        "clock_state_on_mixed": clock_state(mixed),
        "coefficient_table": {
            "id": COEFFICIENT_TABLE,
            "kinds": KIND_COEFF,
            "pressure_weights": PRESSURE_WEIGHT,
            "honesty": (
                "Assumed, not synthesized, not measured. Used to rank and to "
                "produce structural counts. Never converted to seconds or tps."
            ),
        },
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": [
            "Five-level fidelity ladder with available flags; last two UNIMPLEMENTED.",
            "Analytical estimator: DSP/LUT/BRAM/URAM, II, pipeline depth, HBM channel demand, bytes in flight.",
            "Deterministic cycle model tagged MODELLED_NOT_MEASURED; conversion to seconds refused.",
            "compare(level_a, level_b, graph) records ranking disagreement as a FINDING.",
            "search_ladder hard-kills only envelope overflow and ill-formed graphs.",
            "Four adaptation clocks with identity, compatibility predicate, cost class, cache key, footprint.",
            "Local StructuralGraph stand-in so this lane does not import in-flight hwir.py.",
        ],
        "negative_findings": [
            "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json is not in git HEAD and is not on disk here; the 15 hwir_hypotheses were not readable. Node kinds are a local catalog, not atlas-derived.",
            "hcli/agentos/test_fpga_preboard.py does not exist in git HEAD.",
            "No U50 board and no emulation seat; HW_EMULATION and REAL_HARDWARE cannot run.",
            "No existing tools/future/fpga_fidelity.py (or any multi-fidelity FPGA estimator) to extend; fpga_preboard.py is Codex-owned scaffolding and was not forked.",
            "Organ-map HWIR nodes lack width/tile/banking/resource_class, so organ maps cannot be estimated without a lowering step that hwir.py should own.",
            "Feasibility against a real device envelope is UNKNOWN: FPGADeviceGenome is TARGET_UNSELECTED.",
            "This worktree is a sparse checkout; recovered files were read with git show, not imported.",
        ],
        "claim_boundary_note": (
            "Everything in this receipt is STATIC_ONLY. modelled_cycles is not "
            "gpu_ns, not wall_ns, not token_ns, and not a hardware measurement. "
            "hbm_channel_demand is a structural count, not bandwidth_gbps."
        ),
    }
    return write_receipt(RECEIPT, doc, "tools/future/fpga_fidelity.py")


def selftest() -> Any:
    _run_invariants()
    return build()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    out = selftest() if args.selftest else build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
