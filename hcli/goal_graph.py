"""GoalGraph: causal edges, dedupe, conflict, and lifecycle over GoalIR nodes.

``hcli.goal_ir.GoalNode`` is one typed, provenance-tagged atom. A directive
compiles to many of them. This module is the graph those atoms live in: it
answers "does A depend on B", "are A and B the same goal said twice", "do A
and B contradict", and "which goals can run right now" -- none of which a
single ``GoalNode`` can answer about itself.

Scope, on purpose:

* PURE GRAPH LOGIC OVER THE IR. No prose parsing (the tokenizer lane reads
  directive text and decides ``GoalType``/``Provenance``/``statement`` --
  this module never looks at raw text, only at already-typed nodes).
* NO SCHEDULER CONTACT. ``hcli.goal.WorkUnitDAG`` / ``hcli.mission`` /
  ``hcli.ledger`` are untouched here; ``ready_frontier`` names which goals
  *could* run concurrently, it does not dispatch them. The adapter lane
  wires that into WorkUnits.

THE EDGES ARE CAUSAL, not just "related to". ``EdgeType`` values read as a
sentence with ``src`` as the subject: ``src ENABLES dst``, ``src BLOCKS
dst``, ``src REQUIRES dst`` (src needs dst first), ``src SUPERSEDES dst``,
``src ALTERNATIVE_TO dst`` (symmetric), ``src SUPPORTS dst``, ``src TESTS
dst``, ``src FALSIFIES dst``. The point of a causal edge over a plain
"related" link is that evidence can prune a whole branch: a FALSIFIES edge
into a HYPOTHESIS is a reason to walk its dependents, not just a footnote.

DEDUPE is deliberately narrow. ``content_signature`` (type + normalized
statement + provenance) is an exact-match signal this module treats as
authoritative merge grounds -- three restatements that hash identically are
the same claim, full stop, so ``add_node`` folds them into one node with
three ``source_refs`` automatically. Anything fuzzier (different wording,
different id, a human or the tokenizer lane recognizing "HCLI should
improve itself" and "Hawking should build Hawking" as one ultragoal) is a
semantic judgment this module refuses to manufacture on its own --
``merge_nodes`` exists for exactly that, but the caller decides, this module
only carries out the merge safely (it still runs every provenance guard in
``GoalNode.__post_init__``, so folding a PASTE-backed claim onto a
MODEL_INFERRED node still raises rather than silently laundering it).

CONTRADICTIONS are surfaced, never silently resolved. The only contradiction
signal this module invents on its own is structural: two ACTIVE nodes at the
same subject (id slug) whose types are flatly incompatible (an
AUTHORITY_GRANT and a PROHIBITION cannot both stand for the same thing).
Classification prefers the least alarming true explanation -- an explicit
SUPERSEDES edge means it is already resolved, an ALTERNATIVE_TO edge means
the coexistence is intentional -- and only falls through to
``UNRESOLVED_CONFLICT`` when neither applies. See ``detect_conflicts``.

TEMPORAL OVERRIDE never erases. ``apply_supersession`` records a SUPERSEDES
edge and transitions the old node to SUPERSEDED; the old node stays in the
graph, readable, forever -- a new node replacing it is not the same as the
old intent having never existed.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Set

from .goal_ir import GoalNode, GoalType, Status, content_signature, transition


def _coerce_edge_type(value: "EdgeType | str") -> "EdgeType":
    if isinstance(value, EdgeType):
        return value
    try:
        return EdgeType(str(value))
    except ValueError:
        valid = ", ".join(m.value for m in EdgeType)
        raise ValueError(f"edge type={value!r} is not one of: {valid}") from None


class EdgeType(str, Enum):
    """Causal edge kinds. Direction reads ``src <VERB> dst`` -- see module doc."""

    ENABLES = "ENABLES"
    BLOCKS = "BLOCKS"
    ALTERNATIVE_TO = "ALTERNATIVE_TO"
    SUPPORTS = "SUPPORTS"
    TESTS = "TESTS"
    FALSIFIES = "FALSIFIES"
    SUPERSEDES = "SUPERSEDES"
    REQUIRES = "REQUIRES"


# Edge types that impose scheduling order (src must resolve before dst can
# proceed, or vice versa -- see ready_frontier). ALTERNATIVE_TO, SUPPORTS,
# TESTS, FALSIFIES and SUPERSEDES describe a relationship but do not order
# two goals against each other for concurrency purposes: racing alternatives
# is exactly the case where two related goals SHOULD run in parallel.
_PRECEDENCE_EDGES = frozenset({EdgeType.REQUIRES, EdgeType.ENABLES, EdgeType.BLOCKS})

# GoalTypes that generate schedulable work in their own right. Everything
# else (SUCCESS_CRITERION, HARD_CONSTRAINT, SUGGESTED_METHOD, ...) is an
# attribute attached to a frontier via a GoalNode field or an edge, not a
# frontier itself -- see classify_frontier.
FRONTIER_TYPES = frozenset({GoalType.ULTRAGOAL, GoalType.OBJECTIVE, GoalType.SUBOBJECTIVE})

# The only contradiction signal this module manufactures unaided: type pairs
# that are structurally incompatible for the same subject, regardless of
# wording. Anything needing statement text to judge is the tokenizer lane's
# semantic call, not this module's.
_OPPOSED_TYPE_PAIRS: FrozenSet[FrozenSet[GoalType]] = frozenset({
    frozenset({GoalType.AUTHORITY_GRANT, GoalType.PROHIBITION}),
    frozenset({GoalType.HARD_CONSTRAINT, GoalType.ANTI_GOAL}),
})


class GoalIdCollisionError(ValueError):
    """Same id, genuinely different content -- this module refuses to guess
    whether that is a restatement or an accidental slug reuse for a new
    goal (``make_stable_id``'s docstring names this exact ambiguity as the
    graph lane's job, not something it can resolve from the id alone).
    Call ``merge_nodes`` if a human/tokenizer judgment says it is the same
    goal, or mint a different slug if it is not.
    """

    def __init__(self, node_id: str, existing_signature: str, new_signature: str) -> None:
        self.node_id = node_id
        self.existing_signature = existing_signature
        self.new_signature = new_signature
        super().__init__(
            f"{node_id}: already present with a different content_signature "
            f"({existing_signature[:12]}... vs {new_signature[:12]}...) -- "
            "same id, different claim; call merge_nodes() if this is a "
            "recognized restatement, otherwise use a different slug"
        )


class ConflictKind(str, Enum):
    SUPERSEDED = "SUPERSEDED"
    SCOPED_EXCEPTION = "SCOPED_EXCEPTION"
    UNRESOLVED_CONFLICT = "UNRESOLVED_CONFLICT"


class Frontier(str, Enum):
    """Where a node sits relative to task generation. Only ACTIVE_FRONTIER
    nodes of a FRONTIER_TYPES type generate options; everything else is
    dormant, done, or not a frontier to begin with."""

    ACTIVE_FRONTIER = "ACTIVE_FRONTIER"
    SLEEPING_FRONTIER = "SLEEPING_FRONTIER"
    PARKED = "PARKED"
    BLOCKED = "BLOCKED"
    COMPLETE = "COMPLETE"
    SUPERSEDED = "SUPERSEDED"
    NOT_A_FRONTIER = "NOT_A_FRONTIER"


_STATUS_TO_FRONTIER: Dict[Status, Frontier] = {
    Status.ACTIVE: Frontier.ACTIVE_FRONTIER,
    Status.SLEEPING: Frontier.SLEEPING_FRONTIER,
    Status.PARKED: Frontier.PARKED,
    Status.BLOCKED: Frontier.BLOCKED,
    Status.COMPLETE: Frontier.COMPLETE,
    Status.SUPERSEDED: Frontier.SUPERSEDED,
}


def classify_frontier(node: GoalNode) -> Frontier:
    if node.type not in FRONTIER_TYPES:
        return Frontier.NOT_A_FRONTIER
    return _STATUS_TO_FRONTIER[node.status]


def _slug(node: GoalNode) -> str:
    """The part of a make_stable_id-built id after its own type prefix, so
    two nodes of different types can be compared on "same subject". Falls
    back to the full id if it was not built with that convention -- a
    conservative miss (never a false match) rather than a guess."""
    prefix = f"{node.type.value}_"
    return node.id[len(prefix):] if node.id.startswith(prefix) else node.id


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    type: EdgeType
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _coerce_edge_type(self.type))
        src = str(self.src).strip()
        dst = str(self.dst).strip()
        if not src or not dst:
            raise ValueError("Edge.src and Edge.dst must not be empty")
        if src == dst:
            raise ValueError(f"an edge cannot connect {src!r} to itself")
        object.__setattr__(self, "src", src)
        object.__setattr__(self, "dst", dst)


@dataclass(frozen=True)
class Conflict:
    a_id: str
    b_id: str
    kind: ConflictKind
    reason: str


def _merge_nodes(existing: GoalNode, incoming: GoalNode) -> GoalNode:
    """Fold *incoming* onto *existing*: union list-like fields and
    source_refs, take the more urgent priority (lower number) and the
    higher confidence, keep existing's id/type/statement/status. Re-runs
    every ``GoalNode.__post_init__`` guard via ``dataclasses.replace`` --
    a provenance-incompatible fold (e.g. a PASTE ref landing on a
    MODEL_INFERRED node) raises rather than being silently absorbed.
    """
    merged_refs = existing.source_refs
    for sref in incoming.source_refs:
        if sref not in merged_refs:
            merged_refs = merged_refs + (sref,)
    return dataclasses.replace(
        existing,
        confidence=max(existing.confidence, incoming.confidence),
        priority=min(existing.priority, incoming.priority),
        dependencies=existing.dependencies + incoming.dependencies,
        blockers=existing.blockers + incoming.blockers,
        success_criteria=existing.success_criteria + incoming.success_criteria,
        failure_criteria=existing.failure_criteria + incoming.failure_criteria,
        evidence_requirements=existing.evidence_requirements + incoming.evidence_requirements,
        resources=existing.resources + incoming.resources,
        related_frontiers=existing.related_frontiers + incoming.related_frontiers,
        source_refs=merged_refs,
    )


class GoalGraph:
    """A mutable collection of GoalNodes plus the causal edges between them."""

    def __init__(self) -> None:
        self.nodes: Dict[str, GoalNode] = {}
        self.edges: Set[Edge] = set()
        # content_signature -> canonical node id, for the exact-match dedupe
        # path in add_node. Not authoritative for anything else -- see
        # goal_ir.content_signature's own docstring.
        self._signature_index: Dict[str, str] = {}

    # -- node lifecycle -------------------------------------------------

    def add_node(self, node: GoalNode) -> GoalNode:
        """Ingest a freshly-derived node, handling exact-signature dedupe.

        Same id + same content_signature -> merge in place (a restatement
        under the id it already has). Same content_signature under a
        DIFFERENT id -> fold into the first-seen node; the new id is never
        materialized, the returned node names the canonical id to use.
        Same id + different signature -> refuse (GoalIdCollisionError):
        that is the "reused slug, actually a different goal" case this
        module cannot resolve on its own.
        """
        sig = content_signature(node)
        if node.id in self.nodes:
            existing = self.nodes[node.id]
            existing_sig = content_signature(existing)
            if existing_sig != sig:
                raise GoalIdCollisionError(node.id, existing_sig, sig)
            merged = _merge_nodes(existing, node)
            self.nodes[node.id] = merged
            self._signature_index[sig] = node.id
            return merged

        canonical_id = self._signature_index.get(sig)
        if canonical_id is not None:
            merged = _merge_nodes(self.nodes[canonical_id], node)
            self.nodes[canonical_id] = merged
            return merged

        self.nodes[node.id] = node
        self._signature_index[sig] = node.id
        return node

    def update_node(self, node: GoalNode) -> GoalNode:
        """Replace an existing node's fields wholesale -- reprioritizing,
        parking, narrowing. Unlike add_node this does no dedupe: it trusts
        the caller already holds the right id (typically
        ``update_node(transition(graph.nodes[id], ...))`` or
        ``update_node(dataclasses.replace(node, priority=0))``).
        """
        if node.id not in self.nodes:
            raise KeyError(
                f"update_node: {node.id!r} is not in the graph; use add_node for a new id"
            )
        self._signature_index.pop(content_signature(self.nodes[node.id]), None)
        self.nodes[node.id] = node
        self._signature_index[content_signature(node)] = node.id
        return node

    def merge_nodes(self, primary_id: str, other_id: str, *, note: str = "") -> GoalNode:
        """Caller-decided merge for two nodes recognized as the same real
        goal despite different wording/id (their content_signature will
        differ -- that is expected, see content_signature's docstring for
        why this module cannot make that call unaided). Folds other_id's
        source_refs onto primary_id (subject to every provenance guard --
        see _merge_nodes) and supersedes other_id via apply_supersession,
        so the duplicate's history stays in the graph rather than vanishing.
        """
        if primary_id == other_id:
            raise ValueError("cannot merge a node into itself")
        merged = _merge_nodes(self.nodes[primary_id], self.nodes[other_id])
        self.update_node(merged)
        self.apply_supersession(primary_id, other_id, note=note or "recognized as the same goal")
        return merged

    # -- edges ------------------------------------------------------------

    def add_edge(self, src: str, dst: str, type: EdgeType, *, note: str = "") -> Edge:
        if src not in self.nodes:
            raise KeyError(f"unknown node id: {src!r}")
        if dst not in self.nodes:
            raise KeyError(f"unknown node id: {dst!r}")
        edge = Edge(src=src, dst=dst, type=type, note=note)
        self.edges.add(edge)
        return edge

    def apply_supersession(self, new_id: str, old_id: str, *, note: str = "") -> GoalNode:
        """Newer explicit intent overrides older: add a SUPERSEDES edge and
        transition old_id to SUPERSEDED. old_id's node is kept in the
        graph -- never erase historical intent. Raises
        InvalidGoalTransitionError (from goal_ir.transition) if old_id
        cannot legally move to SUPERSEDED from its current status (e.g. it
        is already COMPLETE)."""
        if new_id not in self.nodes:
            raise KeyError(f"unknown node id: {new_id!r}")
        updated_old = transition(self.nodes[old_id], Status.SUPERSEDED, superseded_by=new_id)
        self.update_node(updated_old)
        self.add_edge(new_id, old_id, EdgeType.SUPERSEDES, note=note)
        return updated_old

    # -- contradictions -----------------------------------------------------

    def _edge_kind_between(self, a_id: str, b_id: str, edge_type: EdgeType) -> Optional[Edge]:
        for edge in self.edges:
            if edge.type is edge_type and {edge.src, edge.dst} == {a_id, b_id}:
                return edge
        return None

    def detect_conflicts(self) -> List[Conflict]:
        """Structurally-opposed, both-ACTIVE node pairs sharing a subject.
        Surfaced only when both sides are ACTIVE -- a PARKED or SUPERSEDED
        side cannot materially block anything right now. Never silently
        resolved: classification is SUPERSEDED / SCOPED_EXCEPTION only when
        an explicit edge already says so, else UNRESOLVED_CONFLICT.
        """
        # ponytail: O(n^2) pair scan over ACTIVE nodes; fine while
        # bound_active_frontier keeps the active set small. Upgrade to a
        # slug->[node_id] index first if that stops being true.
        conflicts: List[Conflict] = []
        active = [n for n in self.nodes.values() if n.status is Status.ACTIVE]
        for i, a in enumerate(active):
            for b in active[i + 1:]:
                if frozenset({a.type, b.type}) not in _OPPOSED_TYPE_PAIRS:
                    continue
                if _slug(a) != _slug(b):
                    continue
                conflicts.append(self._classify(a.id, b.id))
        return conflicts

    def _classify(self, a_id: str, b_id: str) -> Conflict:
        if self._edge_kind_between(a_id, b_id, EdgeType.SUPERSEDES):
            return Conflict(a_id, b_id, ConflictKind.SUPERSEDED, f"{a_id}/{b_id} already linked by SUPERSEDES")
        if self._edge_kind_between(a_id, b_id, EdgeType.ALTERNATIVE_TO):
            return Conflict(
                a_id, b_id, ConflictKind.SCOPED_EXCEPTION,
                f"{a_id}/{b_id} declared ALTERNATIVE_TO -- coexistence is intentional",
            )
        return Conflict(
            a_id, b_id, ConflictKind.UNRESOLVED_CONFLICT,
            f"{a_id} and {b_id} are both ACTIVE and structurally opposed",
        )

    # -- parallelism / frontier ---------------------------------------------

    def _precedence_predecessors(self, node_id: str) -> FrozenSet[str]:
        preds: Set[str] = set(self.nodes[node_id].dependencies)
        for edge in self.edges:
            if edge.type not in _PRECEDENCE_EDGES:
                continue
            if edge.type is EdgeType.REQUIRES and edge.src == node_id:
                preds.add(edge.dst)  # "src REQUIRES dst": dst must resolve first
            elif edge.type is EdgeType.ENABLES and edge.dst == node_id:
                preds.add(edge.src)  # "src ENABLES dst": src must resolve first
            elif edge.type is EdgeType.BLOCKS and edge.dst == node_id:
                preds.add(edge.src)  # "src BLOCKS dst": src must resolve first
        return frozenset(preds)

    def _predecessor_satisfied(self, pred_id: str) -> bool:
        pred = self.nodes.get(pred_id)
        return pred is None or pred.status in (Status.COMPLETE, Status.SUPERSEDED)

    def ready_frontier(self) -> List[str]:
        """FRONTIER_TYPES nodes that are ACTIVE with every precedence
        predecessor resolved -- the parallelism-extraction output: this
        whole set can be dispatched concurrently right now, because by
        construction none of them is waiting on another one in it. Prose
        order plays no part; only dependencies/edges do.
        """
        ready = [
            node_id for node_id, node in self.nodes.items()
            if node.type in FRONTIER_TYPES
            and node.status is Status.ACTIVE
            and all(self._predecessor_satisfied(p) for p in self._precedence_predecessors(node_id))
        ]
        return sorted(ready, key=lambda nid: (self.nodes[nid].priority, nid))

    def frontier_report(self) -> Dict[str, List[str]]:
        report: Dict[str, List[str]] = {f.value: [] for f in Frontier}
        for node_id, node in self.nodes.items():
            report[classify_frontier(node).value].append(node_id)
        for ids in report.values():
            ids.sort()
        return report

    def bound_active_frontier(self, max_active: int) -> List[GoalNode]:
        """Keep ACTIVE_FRONTIER at or below *max_active* by putting the
        least urgent excess (highest priority number, then id) to SLEEPING
        -- demoted, never deleted or superseded. This is the mechanism
        behind "keep the active set bounded": it protects the system from
        its own ambition without discarding intent.
        """
        if max_active < 0:
            raise ValueError("max_active must be >= 0")
        active_ids = [
            nid for nid, n in self.nodes.items()
            if n.type in FRONTIER_TYPES and n.status is Status.ACTIVE
        ]
        if len(active_ids) <= max_active:
            return []
        active_ids.sort(key=lambda nid: (self.nodes[nid].priority, nid))
        demoted = []
        for nid in active_ids[max_active:]:
            updated = transition(self.nodes[nid], Status.SLEEPING)
            self.update_node(updated)
            demoted.append(updated)
        return demoted


__all__ = [
    "EdgeType",
    "Edge",
    "ConflictKind",
    "Conflict",
    "Frontier",
    "FRONTIER_TYPES",
    "classify_frontier",
    "GoalIdCollisionError",
    "GoalGraph",
]
