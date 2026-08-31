"""STATIC SKELETON — compile-time topology with typed dynamic slots.

Almost all model-execution topology is static. Only five slot kinds are
genuinely dynamic. This module is the representation that makes that
split explicit, plus a validator that REFUSES any skeleton whose
topology (which nodes exist, which edges exist, dispatch counts)
secretly depends on a value outside those five kinds.

That refusal is what makes graph replay, CUDA graphs, TPU compiled
graphs and FPGA spatial pipelines the same idea: capture the static
skeleton once, bind the slots on every token.

This is a sidecar refinement of ``hcli.physical_graph.v1``, not a rival
PhysicalGraph. hcli/ is Codex-owned and is not mutated here.

    python3 tools/future/static_skeleton.py --selftest
    python3 tools/future/static_skeleton.py --build
    python3 -m pytest tools/future/test_static_skeleton.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from tools.future._common import REPO, git, load_json, write_receipt

RECEIPT = "STATIC_SKELETON.json"
SCHEMA = "hawking.future.static_skeleton.v1"

# Exact permitted dynamic-slot set. Topology may bind these; it may not
# depend on any other runtime value (activation magnitudes, nnz, logits,
# host flags, …). Adding a sixth kind is a schema break, not an extension.
SLOT_KINDS: tuple[str, ...] = (
    "EXPERT_ID",
    "TOKEN_POSITION",
    "SAMPLING",
    "REPRESENTATION_FRAGMENT",
    "STATE_VALUE",
)
SLOT_KIND_SET = frozenset(SLOT_KINDS)

# STATIC        — node/edge/dispatch count is a compile-time constant.
# SLOT_INDEXED  — a compile-time BANK exists; runtime binds an allowed slot
#                 as an index into that bank. The bank is the topology.
# VALUE_GATED   — existence is a function of a runtime tensor/value.
#                 Always refused: that is data-dependent topology.
EXISTENCE_MODES: tuple[str, ...] = ("STATIC", "SLOT_INDEXED", "VALUE_GATED")
EXISTENCE_SET = frozenset(EXISTENCE_MODES)

PHYSICAL_GRAPH_SCHEMA = "hcli.physical_graph.v1"
LEDGER_REL = "receipts/headless/DISPATCH_LEDGER.json"
FLASH_ROUTER_SEL_REL = "receipts/headless/FLASH_NOETIC_ROUTER_SELECTION.json"
FLASH_EXPERT_GRAPH_REL = "receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_GRAPH.json"
FPGA_PREBOARD_REL = "receipts/headless/HCLI_FPGA_PREBOARD.json"
ATLAS_REL = "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json"
FLASH_LAYER46_REL = "receipts/headless/FLASH_LAYER46_DISPATCH_LEDGER.json"
FLASH_LAYER30_REL = "receipts/headless/FLASH_LAYER30_CRITICAL_PATH.json"


class SkeletonRefused(ValueError):
    """Topology secretly depends on a value outside the allowed slot set."""


# ---------------------------------------------------------------------------
# IR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Slot:
    """A typed, bounded hole in an otherwise static skeleton."""

    name: str
    kind: str
    lo: int | None = None
    hi: int | None = None
    shape: tuple[int, ...] | None = None
    dtype: str = "u32"
    dispatch_bound: int | None = None
    meaning: str = ""
    bound_basis: str = ""

    def cardinality(self) -> int | None:
        if self.lo is None or self.hi is None:
            return None
        return int(self.hi) - int(self.lo) + 1

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.shape is not None:
            d["shape"] = list(self.shape)
        return d

    @staticmethod
    def from_mapping(m: Mapping[str, Any]) -> "Slot":
        shape = m.get("shape")
        return Slot(
            name=str(m["name"]),
            kind=str(m["kind"]),
            lo=_opt_int(m.get("lo")),
            hi=_opt_int(m.get("hi")),
            shape=tuple(int(x) for x in shape) if shape is not None else None,
            dtype=str(m.get("dtype") or "u32"),
            dispatch_bound=_opt_int(m.get("dispatch_bound")),
            meaning=str(m.get("meaning") or ""),
            bound_basis=str(m.get("bound_basis") or ""),
        )


@dataclass(frozen=True)
class Node:
    """A compile-time graph node with declared (not measured) resources."""

    id: str
    kind: str
    existence: str = "STATIC"
    slot: str | None = None
    gated_on: str | None = None
    binds_slots: tuple[str, ...] = ()
    dispatch_count: int | None = 1
    dispatch_count_from_slot: str | None = None
    dispatch_count_gated_on: str | None = None
    resources: dict[str, Any] = field(default_factory=dict)
    mixer: str | None = None
    operator: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["binds_slots"] = list(self.binds_slots)
        return d

    @staticmethod
    def from_mapping(m: Mapping[str, Any]) -> "Node":
        binds = m.get("binds_slots") or ()
        return Node(
            id=str(m["id"]),
            kind=str(m.get("kind") or "op"),
            existence=str(m.get("existence") or "STATIC"),
            slot=_opt_str(m.get("slot")),
            gated_on=_opt_str(m.get("gated_on")),
            binds_slots=tuple(str(x) for x in binds),
            dispatch_count=_opt_int(m.get("dispatch_count")) if "dispatch_count" in m else 1,
            dispatch_count_from_slot=_opt_str(m.get("dispatch_count_from_slot")),
            dispatch_count_gated_on=_opt_str(m.get("dispatch_count_gated_on")),
            resources=dict(m.get("resources") or {}),
            mixer=_opt_str(m.get("mixer")),
            operator=_opt_str(m.get("operator")),
        )


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    kind: str = "dataflow"
    existence: str = "STATIC"
    slot: str | None = None
    gated_on: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_mapping(m: Mapping[str, Any]) -> "Edge":
        return Edge(
            src=str(m["src"]),
            dst=str(m["dst"]),
            kind=str(m.get("kind") or "dataflow"),
            existence=str(m.get("existence") or "STATIC"),
            slot=_opt_str(m.get("slot")),
            gated_on=_opt_str(m.get("gated_on")),
        )


@dataclass(frozen=True)
class Skeleton:
    name: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    slots: tuple[Slot, ...] = ()
    externals: tuple[str, ...] = ()
    source: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "slots": [s.to_dict() for s in self.slots],
            "externals": list(self.externals),
            "source": dict(self.source),
        }

    @staticmethod
    def from_mapping(m: Mapping[str, Any]) -> "Skeleton":
        return Skeleton(
            name=str(m.get("name") or "unnamed"),
            nodes=tuple(Node.from_mapping(n) for n in (m.get("nodes") or ())),
            edges=tuple(Edge.from_mapping(e) for e in (m.get("edges") or ())),
            slots=tuple(Slot.from_mapping(s) for s in (m.get("slots") or ())),
            externals=tuple(str(x) for x in (m.get("externals") or ())),
            source=dict(m.get("source") or {}),
        )


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "errors": list(self.errors)}


# ---------------------------------------------------------------------------
# Validator — the load-bearing piece of this lane
# ---------------------------------------------------------------------------


def _opt_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _opt_str(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _coerce(skeleton: Skeleton | Mapping[str, Any]) -> Skeleton:
    if isinstance(skeleton, Skeleton):
        return skeleton
    return Skeleton.from_mapping(skeleton)


def _slot_bounded(slot: Slot) -> bool:
    ranged = slot.lo is not None and slot.hi is not None and int(slot.lo) <= int(slot.hi)
    shaped = bool(slot.shape) and all(int(x) > 0 for x in slot.shape or ())
    return bool(ranged or shaped)


def validate(skeleton: Skeleton | Mapping[str, Any]) -> ValidationResult:
    """Refuse any skeleton whose topology depends on a non-slot value.

    Accepted: STATIC topology, and SLOT_INDEXED topology whose index is
    one of the five declared slot kinds.

    Refused: VALUE_GATED node/edge existence; dispatch counts gated on a
    runtime value; unknown slot kinds; unbounded slots; SLOT_INDEXED
    without a declared slot of an allowed kind.
    """
    sk = _coerce(skeleton)
    errors: list[str] = []

    slots: dict[str, Slot] = {}
    for slot in sk.slots:
        if slot.name in slots:
            errors.append(f"duplicate slot name {slot.name!r}")
            continue
        slots[slot.name] = slot
        if slot.kind not in SLOT_KIND_SET:
            errors.append(
                f"slot {slot.name!r} has kind {slot.kind!r}; permitted kinds are "
                f"{list(SLOT_KINDS)}"
            )
        if not _slot_bounded(slot):
            errors.append(
                f"slot {slot.name!r} is not bounded (need lo<=hi or a positive shape)"
            )
        if slot.lo is not None and slot.hi is not None and int(slot.lo) > int(slot.hi):
            errors.append(f"slot {slot.name!r} has lo={slot.lo} > hi={slot.hi}")
        if slot.dispatch_bound is not None and slot.dispatch_bound < 0:
            errors.append(f"slot {slot.name!r} dispatch_bound is negative")

    node_ids: dict[str, Node] = {}
    for node in sk.nodes:
        if node.id in node_ids:
            errors.append(f"duplicate node id {node.id!r}")
            continue
        node_ids[node.id] = node
        errors.extend(_validate_existence("node", node.id, node.existence, node.slot, node.gated_on, slots))
        for bound in node.binds_slots:
            if bound not in slots:
                errors.append(f"node {node.id!r} binds unknown slot {bound!r}")
        errors.extend(_validate_dispatch(node, slots))

    known = set(node_ids) | set(sk.externals)
    for edge in sk.edges:
        label = f"{edge.src}->{edge.dst}"
        if edge.src not in known:
            errors.append(f"edge {label} src {edge.src!r} is neither a node nor an external")
        if edge.dst not in known:
            errors.append(f"edge {label} dst {edge.dst!r} is neither a node nor an external")
        errors.extend(_validate_existence("edge", label, edge.existence, edge.slot, edge.gated_on, slots))

    return ValidationResult(accepted=not errors, errors=tuple(errors))


def _validate_existence(
    kind: str,
    label: str,
    existence: str,
    slot: str | None,
    gated_on: str | None,
    slots: Mapping[str, Slot],
) -> list[str]:
    errors: list[str] = []
    if existence not in EXISTENCE_SET:
        errors.append(f"{kind} {label!r} has unknown existence {existence!r}")
        return errors
    if existence == "VALUE_GATED":
        errors.append(
            f"{kind} {label!r} existence=VALUE_GATED gated_on={gated_on!r}: "
            "topology may not depend on a runtime value outside "
            f"{list(SLOT_KINDS)}. Data-dependent topology cannot be replayed."
        )
        return errors
    if existence == "SLOT_INDEXED":
        if not slot:
            errors.append(f"{kind} {label!r} is SLOT_INDEXED but names no slot")
        elif slot not in slots:
            errors.append(f"{kind} {label!r} is SLOT_INDEXED on undeclared slot {slot!r}")
        else:
            skind = slots[slot].kind
            if skind not in SLOT_KIND_SET:
                errors.append(
                    f"{kind} {label!r} is SLOT_INDEXED on slot {slot!r} of kind {skind!r}"
                )
        if gated_on:
            errors.append(
                f"{kind} {label!r} is SLOT_INDEXED but also gated_on={gated_on!r}"
            )
        return errors
    # STATIC
    if slot:
        errors.append(
            f"{kind} {label!r} is STATIC but names slot {slot!r}; "
            "use existence=SLOT_INDEXED or binds_slots"
        )
    if gated_on:
        errors.append(
            f"{kind} {label!r} is STATIC but gated_on={gated_on!r}; "
            "that is VALUE_GATED topology"
        )
    return errors


def _validate_dispatch(node: Node, slots: Mapping[str, Slot]) -> list[str]:
    errors: list[str] = []
    if node.dispatch_count_gated_on:
        errors.append(
            f"node {node.id!r} dispatch_count_gated_on={node.dispatch_count_gated_on!r}: "
            "a dispatch count that depends on a runtime value is data-dependent "
            f"topology and is refused. Permitted slot kinds: {list(SLOT_KINDS)}"
        )
    if node.dispatch_count_from_slot:
        name = node.dispatch_count_from_slot
        if name not in slots:
            errors.append(
                f"node {node.id!r} dispatch_count_from_slot={name!r} is undeclared"
            )
        else:
            slot = slots[name]
            if slot.kind not in SLOT_KIND_SET:
                errors.append(
                    f"node {node.id!r} dispatch_count_from_slot kind {slot.kind!r} "
                    "is not an allowed slot"
                )
            bound = slot.dispatch_bound if slot.dispatch_bound is not None else slot.cardinality()
            if bound is None:
                errors.append(
                    f"node {node.id!r} dispatch_count_from_slot={name!r} has no "
                    "dispatch_bound or integer cardinality"
                )
    if node.dispatch_count is not None and int(node.dispatch_count) < 0:
        errors.append(f"node {node.id!r} dispatch_count is negative")
    return errors


def require_valid(skeleton: Skeleton | Mapping[str, Any]) -> Skeleton:
    sk = _coerce(skeleton)
    result = validate(sk)
    if not result.accepted:
        raise SkeletonRefused("; ".join(result.errors))
    return sk


# ---------------------------------------------------------------------------
# Topology metrics (compile-time; not timings)
# ---------------------------------------------------------------------------


def static_fraction(skeleton: Skeleton | Mapping[str, Any]) -> dict[str, Any]:
    """What fraction of topology is STATIC vs SLOT_INDEXED vs VALUE_GATED.

    Dispatch counts count as topology: a count that moves with a tensor
    value is as un-replayable as a moving edge set.
    """
    sk = _coerce(skeleton)
    node_exist = _count_existence(n.existence for n in sk.nodes)
    edge_exist = _count_existence(e.existence for e in sk.edges)
    n_dispatch = len(sk.nodes)
    n_dispatch_gated = 0
    n_dispatch_from_slot = 0
    n_dispatch_static = 0
    for node in sk.nodes:
        if node.dispatch_count_gated_on:
            n_dispatch_gated += 1
        elif node.dispatch_count_from_slot:
            n_dispatch_from_slot += 1
        else:
            n_dispatch_static += 1

    total = node_exist["n"] + edge_exist["n"] + n_dispatch
    static = node_exist["STATIC"] + edge_exist["STATIC"] + n_dispatch_static
    slot_ix = node_exist["SLOT_INDEXED"] + edge_exist["SLOT_INDEXED"] + n_dispatch_from_slot
    gated = node_exist["VALUE_GATED"] + edge_exist["VALUE_GATED"] + n_dispatch_gated
    return {
        "nodes": node_exist,
        "edges": edge_exist,
        "dispatch_counts": {
            "n": n_dispatch,
            "static": n_dispatch_static,
            "from_slot_bound": n_dispatch_from_slot,
            "value_gated": n_dispatch_gated,
        },
        "topology_units": total,
        "static_units": static,
        "slot_indexed_units": slot_ix,
        "value_gated_units": gated,
        "topology_static_fraction": (static / total) if total else 1.0,
        "replayable_fraction": ((static + slot_ix) / total) if total else 1.0,
        "reading": (
            "STATIC units are captured once. SLOT_INDEXED units are a compile-time "
            "bank plus an allowed slot bind; they still replay. VALUE_GATED units "
            "cannot be replayed."
        ),
    }


def _count_existence(values: Iterable[str]) -> dict[str, int]:
    counts = {mode: 0 for mode in EXISTENCE_MODES}
    n = 0
    for value in values:
        n += 1
        if value in counts:
            counts[value] += 1
        else:
            counts[value] = counts.get(value, 0) + 1
    return {"n": n, **counts}


def longest_path(skeleton: Skeleton | Mapping[str, Any]) -> dict[str, Any]:
    """Compile-time critical path as a node-count, never a measured ns."""
    sk = _coerce(skeleton)
    ids = [n.id for n in sk.nodes]
    incoming: dict[str, list[str]] = {i: [] for i in ids}
    outgoing: dict[str, list[str]] = {i: [] for i in ids}
    for edge in sk.edges:
        if edge.src in incoming and edge.dst in incoming:
            incoming[edge.dst].append(edge.src)
            outgoing[edge.src].append(edge.dst)
    # Kahn
    indeg = {i: len(incoming[i]) for i in ids}
    q = sorted(i for i in ids if indeg[i] == 0)
    order: list[str] = []
    while q:
        n = q.pop(0)
        order.append(n)
        for nxt in sorted(outgoing[n]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
                q.sort()
    is_dag = len(order) == len(ids)
    if not is_dag:
        return {
            "is_dag": False,
            "path": [],
            "length_nodes": None,
            "unit": "nodes",
            "note": "cycle present; compile-time critical path undefined",
        }
    dist: dict[str, int] = {i: 1 for i in ids}
    pred: dict[str, str | None] = {i: None for i in ids}
    for n in order:
        for nxt in outgoing[n]:
            cand = dist[n] + 1
            if cand > dist[nxt]:
                dist[nxt] = cand
                pred[nxt] = n
    if not dist:
        return {"is_dag": True, "path": [], "length_nodes": 0, "unit": "nodes"}
    end = max(sorted(dist), key=lambda k: (dist[k], k))
    path = [end]
    while pred[path[-1]] is not None:
        path.append(pred[path[-1]] or "")
    path.reverse()
    return {
        "is_dag": True,
        "path": path,
        "length_nodes": dist[end] if ids else 0,
        "unit": "nodes",
        "note": "node-count longest path; not a hardware latency",
    }


# ---------------------------------------------------------------------------
# Constructors: PhysicalGraph wrap, ledger layer, Flash MoE component
# ---------------------------------------------------------------------------


def skeleton_from_physical_graph(
    pg: Mapping[str, Any],
    *,
    name: str | None = None,
) -> Skeleton:
    """Wrap a hcli.physical_graph.v1 computation/dependency list as STATIC.

    Does not import hcli (sidecar boundary; hcli is not on the sparse
    checkout write path). PhysicalGraph already stores a fixed node and
    edge list; this wrapper makes that STATIC-ness explicit and gives the
    validator something to refuse when a caller later adds a gated edge.
    """
    computation = list(pg.get("computation") or [])
    dependencies = list(pg.get("dependencies") or [])
    nodes = []
    for item in computation:
        nid = str(item.get("id") or item.get("organ") or "")
        if not nid:
            continue
        nodes.append(
            Node(
                id=nid,
                kind=str(item.get("kind") or "computation"),
                existence="STATIC",
                dispatch_count=int(item["dispatches_per_sample"]) if item.get("dispatches_per_sample") is not None else 1,
                resources={
                    k: item[k]
                    for k in ("kernel", "tensor_name", "row_count", "row_start", "expert_index")
                    if k in item
                },
            )
        )
    edges = []
    for dep in dependencies:
        src, dst = dep.get("from"), dep.get("to")
        if not src or not dst:
            continue
        edges.append(Edge(src=str(src), dst=str(dst), kind=str(dep.get("kind") or "dataflow")))
    return Skeleton(
        name=name or str(pg.get("model_id") or "physical_graph"),
        nodes=tuple(nodes),
        edges=tuple(edges),
        slots=(),
        externals=(),
        source={
            "parent_schema": pg.get("schema") or PHYSICAL_GRAPH_SCHEMA,
            "semantic_type": pg.get("semantic_type"),
            "qualification": pg.get("qualification"),
            "fingerprint": pg.get("fingerprint"),
            "relationship": (
                "sidecar refinement: PhysicalGraph is PLAN_ONLY placement/"
                "dataflow; this skeleton adds existence modes and slots"
            ),
        },
    )


def skeleton_from_ledger_layer(ledger: Mapping[str, Any], layer: int) -> Skeleton:
    """Build the skeleton of one named layer from DISPATCH_LEDGER.json."""
    rows = [d for d in (ledger.get("dispatches") or []) if d.get("layer") == layer]
    rows = sorted(rows, key=lambda d: int(d.get("index") or 0))
    if not rows:
        raise ValueError(f"DISPATCH_LEDGER has no dispatches for layer {layer}")

    ops = {str(r["operator"]) for r in rows}
    nodes: list[Node] = []
    externals: set[str] = set()
    edges: list[Edge] = []

    seq = ledger.get("seq_len_for_kv_bytes")
    token_hi = int(seq) - 1 if isinstance(seq, int) and seq > 0 else None
    slots: list[Slot] = []
    if token_hi is not None:
        slots.append(
            Slot(
                name="token_position",
                kind="TOKEN_POSITION",
                lo=0,
                hi=token_hi,
                dtype="u32",
                meaning="decode-step index into conv/recurrent state",
                bound_basis=f"{LEDGER_REL}#seq_len_for_kv_bytes={seq}; captured for this compile-time max, not a model-card context",
            )
        )
    # DeltaNet recurrent state is data inside a STATIC node. The ledger's
    # gated_delta class records "48×128×128 rec state"; per layer that is 128×128.
    slots.append(
        Slot(
            name="dn_recurrent_state",
            kind="STATE_VALUE",
            shape=(128, 128),
            dtype="f32",
            meaning="gated-delta recurrent state written/read by linear_attn.recurrence",
            bound_basis=f"{LEDGER_REL} classes.gated_delta fusion_candidacy_why '48×128×128 rec state' / 48 DN layers",
        )
    )

    for row in rows:
        op = str(row["operator"])
        binds: list[str] = []
        if op == "gated_delta":
            binds = ["dn_recurrent_state", "token_position"] if token_hi is not None else ["dn_recurrent_state"]
        elif op == "qkvz_rearrange_conv_l2" and token_hi is not None:
            binds = ["token_position"]
        bytes_ = row.get("bytes") if isinstance(row.get("bytes"), Mapping) else {}
        resources = {
            "kernel": row.get("kernel"),
            "organ": row.get("organ"),
            "declared_from": LEDGER_REL,
            "weight_read_bytes": bytes_.get("weight_read") if bytes_ else None,
            "activation_read_bytes": bytes_.get("activation_read") if bytes_ else None,
            "activation_write_bytes": bytes_.get("activation_write") if bytes_ else None,
            "flops": row.get("flops"),
            "ledger_index": row.get("index"),
            "command_buffer": row.get("command_buffer"),
        }
        nodes.append(
            Node(
                id=op,
                kind="dispatch",
                existence="STATIC",
                binds_slots=tuple(b for b in binds if any(s.name == b for s in slots)),
                dispatch_count=1,
                resources=resources,
                mixer=_opt_str(row.get("mixer")),
                operator=op,
            )
        )
        for dep in row.get("dependencies") or []:
            src = str(dep)
            if src in ops:
                edges.append(Edge(src=src, dst=op, kind="dataflow", existence="STATIC"))
            else:
                externals.add(src)
                edges.append(Edge(src=src, dst=op, kind="dataflow", existence="STATIC"))

    mixer = rows[0].get("mixer")
    return Skeleton(
        name=f"dispatch_ledger.layer_{layer}",
        nodes=tuple(nodes),
        edges=tuple(edges),
        slots=tuple(slots),
        externals=tuple(sorted(externals)),
        source={
            "receipt": LEDGER_REL,
            "schema": ledger.get("schema"),
            "layer": layer,
            "mixer": mixer,
            "n_dispatches_in_layer": len(rows),
            "ledger_graph_formula": (ledger.get("graph") or {}).get("formula"),
            "parent_fused_dispatches": (ledger.get("graph") or {}).get("parent_fused"),
            "note": (
                "Named FLASH_LAYER46_DISPATCH_LEDGER.json is absent from git. "
                "This skeleton is layer "
                f"{layer} of the real sealed {LEDGER_REL} (Qwen38/80 hybrid, "
                "dense MLP, not Flash MoE)."
            ),
        },
    )


def flash_moe_component_skeleton(
    selection: Mapping[str, Any],
    expert_graph: Mapping[str, Any],
) -> Skeleton:
    """Bounded Flash router + expert-bank skeleton from Noetic receipts.

    This is NOT a complete Flash layer: the receipts contain a router
    selection component and a single expert-window body, not a composed
    layer-46 graph. The connecting edge is SLOT_INDEXED on EXPERT_ID.
    """
    router = ((selection.get("config") or {}).get("router") or {})
    n_experts = int(router["num_experts"])
    top_k = int(router["num_experts_per_tok"])
    window = expert_graph.get("component_window") or {}
    row_count = int(window.get("row_count") or 128)

    slots = (
        Slot(
            name="expert_id",
            kind="EXPERT_ID",
            lo=0,
            hi=n_experts - 1,
            dtype="u16",
            dispatch_bound=top_k,
            meaning="Flash routed-expert index; bank is static, bind is the slot",
            bound_basis=f"{FLASH_ROUTER_SEL_REL}#config.router.num_experts={n_experts}, num_experts_per_tok={top_k}",
        ),
        Slot(
            name="expert_row_window",
            kind="REPRESENTATION_FRAGMENT",
            lo=0,
            hi=max(0, row_count - 1),
            dtype="u32",
            meaning="row window into a packed expert body (Flash component_window)",
            bound_basis=f"{FLASH_EXPERT_GRAPH_REL}#component_window.row_count={row_count}",
        ),
    )
    nodes = (
        Node(id="router_body_load", kind="source_independent_component_body", dispatch_count=1),
        Node(id="router_q4_matvec", kind="native_kernel_dispatch", dispatch_count=1,
             resources={"kernel": "qwen_uniform_q4_group64_matvec"}),
        Node(id="router_fp32_softmax", kind="router_probability_normalization", dispatch_count=1),
        Node(id="router_top_k", kind="router_top_k_selection", dispatch_count=1,
             binds_slots=("expert_id",)),
        Node(
            id="expert_bank",
            kind="routed_expert_body",
            existence="SLOT_INDEXED",
            slot="expert_id",
            binds_slots=("expert_row_window",),
            dispatch_count=top_k,
            dispatch_count_from_slot="expert_id",
            resources={
                "kernel": "qwen_uniform_q4_group64_matvec",
                "bank_size": n_experts,
                "component_window": window,
                "declared_from": FLASH_EXPERT_GRAPH_REL,
            },
        ),
        Node(
            id="shared_expert",
            kind="shared_expert_body",
            existence="STATIC",
            dispatch_count=1,
            resources={
                "note": "config.router.shared_expert_sigmoid_is_not_router_selection=true; shared path is not EXPERT_ID-gated"
            },
        ),
    )
    edges = (
        Edge(src="router_body_load", dst="router_q4_matvec"),
        Edge(src="router_q4_matvec", dst="router_fp32_softmax"),
        Edge(src="router_fp32_softmax", dst="router_top_k"),
        Edge(src="router_top_k", dst="expert_bank", existence="SLOT_INDEXED", slot="expert_id"),
        Edge(src="router_top_k", dst="shared_expert", existence="STATIC"),
    )
    return Skeleton(
        name="flash.layer0.router_plus_expert_bank",
        nodes=nodes,
        edges=edges,
        slots=slots,
        source={
            "receipts": [FLASH_ROUTER_SEL_REL, FLASH_EXPERT_GRAPH_REL],
            "complete_layer": False,
            "complete_token_runtime": selection.get("complete_token_runtime"),
            "next_action_on_disk": selection.get("next_action"),
            "note": (
                "Bounded component: router selection is composed; expert dispatch "
                "is a SLOT_INDEXED bank. The receipts themselves say the selection "
                "edge is not yet connected to a protected complete-token graph."
            ),
        },
    )


# ---------------------------------------------------------------------------
# Discriminating fixtures (also used by tests)
# ---------------------------------------------------------------------------


def legal_expert_id_skeleton() -> Skeleton:
    """MoE bank indexed by EXPERT_ID — accepted."""
    return Skeleton(
        name="legal_expert_id",
        slots=(
            Slot(
                name="expert_id",
                kind="EXPERT_ID",
                lo=0,
                hi=511,
                dtype="u16",
                dispatch_bound=10,
                meaning="routed expert index",
            ),
        ),
        nodes=(
            Node(id="router", kind="router_matvec", dispatch_count=1),
            Node(
                id="expert_bank",
                kind="expert_body",
                existence="SLOT_INDEXED",
                slot="expert_id",
                dispatch_count=10,
                dispatch_count_from_slot="expert_id",
            ),
        ),
        edges=(
            Edge(src="router", dst="expert_bank", existence="SLOT_INDEXED", slot="expert_id"),
        ),
    )


def illegal_activation_gated_skeleton() -> Skeleton:
    """Edge set depends on a routed activation VALUE, not an expert id — refused."""
    return Skeleton(
        name="illegal_activation_gated",
        slots=(),
        nodes=(
            Node(id="router", kind="router_matvec", dispatch_count=1),
            Node(id="expert_0", kind="expert_body", dispatch_count=1),
            Node(id="expert_1", kind="expert_body", dispatch_count=1),
        ),
        edges=(
            Edge(
                src="router",
                dst="expert_0",
                existence="VALUE_GATED",
                gated_on="routed_activation_value",
            ),
            Edge(
                src="router",
                dst="expert_1",
                existence="VALUE_GATED",
                gated_on="routed_activation_value",
            ),
        ),
    )


def illegal_activation_gated_dispatch_count() -> Skeleton:
    """Dispatch count = f(activation value) — refused."""
    return Skeleton(
        name="illegal_activation_gated_dispatch_count",
        slots=(),
        nodes=(
            Node(
                id="sparse_mlp",
                kind="mlp",
                dispatch_count=None,
                dispatch_count_gated_on="routed_activation_value",
            ),
        ),
        edges=(),
    )


# ---------------------------------------------------------------------------
# Backend usability — availability flags, no hardware claims
# ---------------------------------------------------------------------------


def _metal_shader_count() -> int:
    root = REPO / "crates" / "hawking-core" / "shaders"
    if not root.is_dir():
        return 0
    return sum(1 for p in sorted(root.glob("*.metal")) if p.is_file())


def _git_path_exists(rel: str) -> bool:
    listed = git("ls-tree", "--name-only", "HEAD", rel)
    return bool(listed.strip())


def _load_optional(rel: str) -> dict[str, Any] | None:
    path = REPO / rel
    if path.is_file():
        try:
            value = load_json(path)
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None
    raw = git("show", f"HEAD:{rel}")
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def backend_usability() -> dict[str, Any]:
    """What each backend could consume, and where a slot becomes indirection.

    Availability is disk/git evidence only. No backend is claimed to be
    physically present unless a board/device flag on disk says so.
    """
    metal_n = _metal_shader_count()
    fpga_receipt = _load_optional(FPGA_PREBOARD_REL)
    fpga_board = False
    if isinstance(fpga_receipt, Mapping):
        genome = fpga_receipt.get("device_genome") or {}
        fpga_board = bool(genome.get("physical_board_present"))

    cuda_runtime_in_git = _git_path_exists("tools/accelerator/cuda_runtime.py")
    air_in_git = _git_path_exists("tools/accelerator/air.py")
    ane_hits = []
    for line in git("ls-tree", "-r", "--name-only", "HEAD", "hcli", "tools").splitlines():
        base = line.rsplit("/", 1)[-1].lower()
        if (
            base.startswith("ane")
            or "_ane_" in base
            or base.endswith("_ane.py")
            or "neural_engine" in base
            or "coreml" in base
            or "mlcompute" in base
        ):
            ane_hits.append(line)

    slot_indirection = {
        "EXPERT_ID": (
            "index into a compile-time expert bank (buffer offset / argument-buffer "
            "id / FPGA mux select). The bank is the captured topology."
        ),
        "TOKEN_POSITION": (
            "index into KV / conv / RoPE tables. Not a new dispatch. Prefill vs "
            "decode are distinct captured skeletons, not a TOKEN_POSITION branch."
        ),
        "SAMPLING": (
            "parameter of the terminal sample node (argmax vs temperature/top-k). "
            "A sampling-mode change that swaps the node kind is a second skeleton, "
            "not a slot bind; backends that cannot parameterize the sample op take "
            "a graph break here."
        ),
        "REPRESENTATION_FRAGMENT": (
            "row_start/row_count (or group window) uniform. The kernel stays; the "
            "window is an offset into a packed body."
        ),
        "STATE_VALUE": (
            "payload of a persistent buffer the STATIC recurrence node already "
            "reads and writes. Values are data, not topology. A backend that "
            "cannot keep the buffer resident has a transport problem, not a "
            "graph-replay problem."
        ),
    }

    return {
        "METAL": {
            "present_on_disk": metal_n > 0,
            "shader_count": metal_n,
            "physical_device_authority": False,
            "consumes": [
                "STATIC nodes as Metal dispatches",
                "STATIC edges as producer/consumer ordering in one command buffer",
                "declared kernel names (already in DISPATCH_LEDGER)",
                "AirGraph one-submission capture (tools/accelerator/air.py) as the Apple-side CUDA-graph analogue",
            ],
            "slot_becomes_indirection": slot_indirection,
            "graph_replay": (
                "AirGraph records a DAG as one submission. SLOT_INDEXED binds become "
                "setBytes/argument-buffer writes before replay. VALUE_GATED topology "
                "cannot be captured."
            ),
            "would_need_hardware_to_claim": [
                "command-buffer replay ns",
                "whether Metal concurrent dispatches match CUDA streams",
            ],
        },
        "FPGA": {
            "present_on_disk": bool(fpga_receipt),
            "physical_board_present": fpga_board,
            "physical_device_authority": False,
            "consumes": [
                "STATIC skeleton as a spatial pipeline (HWIR organ nodes in hcli/agentos/fpga_preboard.py)",
                "declared buffers (resident_weight_shards, activations, persistent_state)",
            ],
            "slot_becomes_indirection": {
                **slot_indirection,
                "EXPERT_ID": (
                    "mux / address generator over an HBM-resident expert bank. "
                    "Runtime bitstream reconfiguration is NOT a slot; it would be "
                    "VALUE_GATED topology and is refused."
                ),
            },
            "graph_replay": (
                "The spatial pipeline IS the captured graph. Slots are wires. "
                "FPGA is part of Accelerator / Physical Compiler / Fusion, not its "
                "own civilization; this lane does not build an FPGA backend."
            ),
            "availability_evidence": FPGA_PREBOARD_REL if fpga_receipt else "absent from this checkout",
            "would_need_hardware_to_claim": ["bitstream timing", "HBM bandwidth", "U50 occupancy"],
        },
        "CUDA": {
            "present_on_disk": cuda_runtime_in_git,
            "cuda_graph_api": (
                "tools/accelerator/cuda_runtime.py lists cudaGraphLaunch as an "
                "UNSUPPORTED host form (Metal translation of a CUDA HOST subset). "
                "No CUDA device is claimed."
            ),
            "physical_device_authority": False,
            "consumes": [
                "STATIC nodes as cudaGraph kernel nodes",
                "STATIC edges as graph dependencies",
            ],
            "slot_becomes_indirection": {
                **slot_indirection,
                "EXPERT_ID": (
                    "cudaKernelNodeSetParams / kernel argument for the expert index; "
                    "the graph stays captured. Instantiating a new graph per expert "
                    "id would be treating a slot as topology and is the thing this "
                    "validator exists to refuse."
                ),
            },
            "graph_replay": "cudaGraphLaunch of the captured skeleton; slot binds via node params",
            "air_analogue_in_git": air_in_git,
            "would_need_hardware_to_claim": ["cuda graph launch ns", "copy-engine overlap"],
        },
        "ANE_GRAPH_SEGMENT": {
            "present_on_disk": False,
            "module": None,
            "path_hits_named_ane": ane_hits[:8],
            "probe": "basename starts with ane, contains _ane_, or names neural_engine/coreml/mlcompute; substring hits like lane/plane are not ANE",
            "physical_device_authority": False,
            "consumes": [
                "STATIC subgraphs that lower to an MLProgram (elementwise, GEMV, norm)",
                "SLOT_INDEXED gathers that ANE can express as gather/select",
            ],
            "slot_becomes_indirection": {
                **slot_indirection,
                "EXPERT_ID": (
                    "a gather over a packed expert tensor if the bank fits the ANE "
                    "weight surface; otherwise a CPU/GPU graph break. ANE cannot "
                    "recompile per token, so VALUE_GATED topology is a hard refusal."
                ),
                "SAMPLING": (
                    "typically a graph break: ANE segments stop before sampling."
                ),
            },
            "graph_replay": (
                "ANE compiles a static graph segment. Slots that are not expressible "
                "as tensor indices become host round-trips between segments."
            ),
            "would_need_hardware_to_claim": ["ANE compile success", "ANE segment latency"],
        },
    }


# ---------------------------------------------------------------------------
# Recovery probes (executed at build; a missing file is a negative finding)
# ---------------------------------------------------------------------------


def recover() -> dict[str, Any]:
    physical_graph_py = git("show", "HEAD:hcli/physical_graph.py")
    has_skeleton_token = "static_skeleton" in physical_graph_py or "dynamic slot" in physical_graph_py.lower()
    atlas = _load_optional(ATLAS_REL)
    layer46_named = _load_optional(FLASH_LAYER46_REL)
    layer30_named = _load_optional(FLASH_LAYER30_REL)
    ledger = _load_optional(LEDGER_REL)
    router_sel = _load_optional(FLASH_ROUTER_SEL_REL)
    expert_graph = _load_optional(FLASH_EXPERT_GRAPH_REL)
    fpga = _load_optional(FPGA_PREBOARD_REL)

    pg_schema = None
    if physical_graph_py:
        for line in physical_graph_py.splitlines():
            if "SCHEMA" in line and "physical_graph" in line:
                pg_schema = line.strip()
                break

    return {
        "hcli/physical_graph.py": {
            "in_git": bool(physical_graph_py),
            "schema_line": pg_schema,
            "mentions_static_skeleton": has_skeleton_token,
            "what_it_is": (
                "Provider-neutral PhysicalGraph planning boundary: computation, "
                "data, representation, memory, residency, state, precision, "
                "dependencies, device_placement, synchronization. qualification="
                "PLAN_ONLY. It does not distinguish STATIC topology from dynamic "
                "control and has no validator against data-dependent edges."
            ),
            "decision": (
                "extend, do not fork: this module wraps PhysicalGraph lists as "
                "STATIC nodes/edges and adds slots + the refusal. hcli/ is not written."
            ),
        },
        ATLAS_REL: {
            "in_git_or_disk": atlas is not None,
            "note": (
                "Frontier F012 probes this path as present on a full checkout. "
                "It is not in git HEAD of this worktree and not on the sparse disk, "
                "so the 'static skeleton + dynamic slots' taxonomy entry could not "
                "be consumed. Slot kinds are taken from this lane's contract."
            ),
        },
        FLASH_LAYER46_REL: {"in_git_or_disk": layer46_named is not None},
        FLASH_LAYER30_REL: {"in_git_or_disk": layer30_named is not None},
        LEDGER_REL: {
            "in_git_or_disk": ledger is not None,
            "schema": (ledger or {}).get("schema"),
            "n_dispatches": len((ledger or {}).get("dispatches") or []),
            "used_as": "the real layer topology source (layer 46 and, as a check, layer 30)",
        },
        FLASH_ROUTER_SEL_REL: {
            "in_git_or_disk": router_sel is not None,
            "num_experts": ((router_sel or {}).get("config") or {}).get("router", {}).get("num_experts") if router_sel else None,
            "num_experts_per_tok": ((router_sel or {}).get("config") or {}).get("router", {}).get("num_experts_per_tok") if router_sel else None,
        },
        FLASH_EXPERT_GRAPH_REL: {
            "in_git_or_disk": expert_graph is not None,
            "component_window": (expert_graph or {}).get("component_window") if expert_graph else None,
        },
        FPGA_PREBOARD_REL: {
            "in_git_or_disk": fpga is not None,
            "physical_board_present": ((fpga or {}).get("device_genome") or {}).get("physical_board_present") if fpga else None,
        },
        "tools/accelerator/air.py": {
            "in_git": _git_path_exists("tools/accelerator/air.py"),
            "role": "AirGraph: Apple-side CUDA-graph analogue (one submission)",
        },
        "tools/accelerator/cuda_runtime.py": {
            "in_git": _git_path_exists("tools/accelerator/cuda_runtime.py"),
            "role": "CUDA HOST subset translated to Metal; cudaGraphLaunch unsupported",
        },
    }


def _layer_demo(ledger: Mapping[str, Any], layer: int) -> dict[str, Any]:
    sk = skeleton_from_ledger_layer(ledger, layer)
    result = validate(sk)
    frac = static_fraction(sk)
    path = longest_path(sk)
    return {
        "layer": layer,
        "mixer": sk.source.get("mixer"),
        "accepted": result.accepted,
        "errors": list(result.errors),
        "n_nodes": len(sk.nodes),
        "n_edges": len(sk.edges),
        "n_slots": len(sk.slots),
        "externals": list(sk.externals),
        "node_ids": [n.id for n in sk.nodes],
        "slot_kinds_used": sorted({s.kind for s in sk.slots}),
        "static_fraction": frac,
        "compile_time_critical_path": path,
        "skeleton": sk.to_dict(),
        "reading": (
            "Every node and edge in this layer is STATIC and every dispatch_count "
            "is 1. The ledger is a dense DeltaNet+MLP hybrid (not Flash MoE): "
            "there is no EXPERT_ID in this layer. Recurrent state and token "
            "position bind as slots on STATIC nodes; they do not change topology. "
            "Graph replay applies to 100% of this layer's topology. That is a "
            "positive bound on the hypothesis for this model, not a Flash MoE claim."
        ),
    }


def _selftest_discrimination() -> dict[str, Any]:
    legal = validate(legal_expert_id_skeleton())
    illegal_edges = validate(illegal_activation_gated_skeleton())
    illegal_count = validate(illegal_activation_gated_dispatch_count())
    if not legal.accepted:
        raise AssertionError(f"legal EXPERT_ID skeleton was refused: {legal.errors}")
    if illegal_edges.accepted:
        raise AssertionError("VALUE_GATED activation-edge skeleton was accepted; the guard did not fire")
    if illegal_count.accepted:
        raise AssertionError("activation-gated dispatch_count skeleton was accepted; the guard did not fire")
    return {
        "legal_expert_id_accepted": True,
        "illegal_activation_gated_edges_refused": True,
        "illegal_activation_gated_edges_errors": list(illegal_edges.errors),
        "illegal_activation_gated_dispatch_count_refused": True,
        "illegal_activation_gated_dispatch_count_errors": list(illegal_count.errors),
        "guard_watched_failing": True,
    }


def build() -> Any:
    recovered = recover()
    ledger = _load_optional(LEDGER_REL)
    router_sel = _load_optional(FLASH_ROUTER_SEL_REL)
    expert_graph = _load_optional(FLASH_EXPERT_GRAPH_REL)

    demonstrations: dict[str, Any] = {}
    negative: list[str] = []

    if recovered[ATLAS_REL]["in_git_or_disk"] is False:
        negative.append(
            f"{ATLAS_REL} is not in git HEAD and not on this sparse disk; "
            "could not read the 'static skeleton + dynamic slots' taxonomy entry"
        )
    if recovered[FLASH_LAYER46_REL]["in_git_or_disk"] is False:
        negative.append(
            f"{FLASH_LAYER46_REL} does not exist; used {LEDGER_REL} layer 46 instead"
        )
    if recovered[FLASH_LAYER30_REL]["in_git_or_disk"] is False:
        negative.append(
            f"{FLASH_LAYER30_REL} does not exist; compile-time critical path is "
            "computed from the layer DAG (node count, not ns)"
        )
    if recovered["hcli/physical_graph.py"]["mentions_static_skeleton"] is False:
        negative.append(
            "hcli/physical_graph.py has no static_skeleton token; the existing "
            "object is PhysicalGraph (PLAN_ONLY), which this module extends rather "
            "than forks"
        )

    if ledger is None:
        negative.append(f"could not load {LEDGER_REL} from disk or git show")
    else:
        for layer in (46, 30):
            try:
                demonstrations[f"dispatch_ledger_layer_{layer}"] = _layer_demo(ledger, layer)
            except ValueError as exc:
                negative.append(str(exc))

    flash_demo: dict[str, Any] | None = None
    if router_sel is not None and expert_graph is not None:
        flash_sk = flash_moe_component_skeleton(router_sel, expert_graph)
        flash_result = validate(flash_sk)
        flash_demo = {
            "accepted": flash_result.accepted,
            "errors": list(flash_result.errors),
            "static_fraction": static_fraction(flash_sk),
            "skeleton": flash_sk.to_dict(),
            "reading": (
                "Flash layer-0 router is STATIC. The expert body is a SLOT_INDEXED "
                "bank of config.router.num_experts with a compile-time dispatch_bound "
                "of num_experts_per_tok. Replayable_fraction is 1.0; topology is not "
                "100% STATIC because the expert edge is a slot bind. The receipts do "
                "not contain a composed complete-token Flash layer, so this is a "
                "bounded component, not a full-layer fraction."
            ),
        }
        if not flash_result.accepted:
            negative.append(f"flash MoE component skeleton refused: {flash_result.errors}")
    else:
        missing = []
        if router_sel is None:
            missing.append(FLASH_ROUTER_SEL_REL)
        if expert_graph is None:
            missing.append(FLASH_EXPERT_GRAPH_REL)
        negative.append("could not load Flash MoE receipts: " + ", ".join(missing))

    discrimination = _selftest_discrimination()
    backends = backend_usability()

    l46 = demonstrations.get("dispatch_ledger_layer_46") or {}
    l46_frac = (l46.get("static_fraction") or {}).get("topology_static_fraction")

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Static physical skeleton with typed dynamic slots, plus a validator "
            "that refuses data-dependent topology so graph replay / CUDA graphs / "
            "TPU compiled graphs / FPGA spatial pipelines are the same idea."
        ),
        "slot_kinds": list(SLOT_KINDS),
        "existence_modes": list(EXISTENCE_MODES),
        "validator_law": {
            "accepts": "STATIC topology; SLOT_INDEXED topology whose index is one of slot_kinds",
            "refuses": (
                "VALUE_GATED node/edge existence; dispatch counts gated on a runtime "
                "value; unknown slot kinds; unbounded slots"
            ),
            "why": "a data-dependent topology cannot be replayed",
        },
        "relationship_to_physical_graph": recovered["hcli/physical_graph.py"],
        "recovered_implementation": recovered,
        "gaps_closed": [
            "typed Slot / Node / Edge / Skeleton IR with exactly five slot kinds",
            "validate() that refuses VALUE_GATED topology and activation-gated dispatch counts",
            "discriminating fixtures: EXPERT_ID SLOT_INDEXED accepted, activation-gated edges refused",
            "backend usability map for METAL / FPGA / CUDA / ANE_GRAPH_SEGMENT with presence flags and slot-to-indirection",
            f"real-layer demonstration from {LEDGER_REL} layer 46 (and layer 30 as the same DeltaNet shape)",
            "PhysicalGraph wrapper that does not fork hcli/physical_graph.py",
        ],
        "negative_findings": negative,
        "backend_usability": backends,
        "discrimination_selftest": discrimination,
        "demonstration": {
            "primary": "dispatch_ledger_layer_46",
            "primary_topology_static_fraction": l46_frac,
            "layers": demonstrations,
            "flash_moe_bounded_component": flash_demo,
        },
        "integration": {
            "validate": "validate(skeleton: Skeleton | Mapping) -> ValidationResult",
            "require_valid": "require_valid(skeleton) -> Skeleton  # raises SkeletonRefused",
            "skeleton_from_physical_graph": "skeleton_from_physical_graph(pg: Mapping) -> Skeleton",
            "skeleton_from_ledger_layer": "skeleton_from_ledger_layer(ledger: Mapping, layer: int) -> Skeleton",
            "static_fraction": "static_fraction(skeleton) -> dict",
        },
    }
    return write_receipt(RECEIPT, doc, "tools/future/static_skeleton.py")


def selftest() -> Any:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("This is a sidecar", 1)[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
