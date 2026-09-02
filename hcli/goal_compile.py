"""Goal compile adapter: a GoalIR graph -> the machinery that already exists.

``hcli.goal_ir`` mints typed atoms, ``hcli.goal_tokenizer`` reads directive
prose into them, ``hcli.goal_graph`` gives them a causal graph with dedupe,
conflict detection, and a schedulable frontier. None of those three modules
touches ``hcli.goal.WorkUnitDAG`` / ``hcli.mission`` / ``hcli.scheduler`` on
purpose -- see each one's own docstring. This module is where that contact
happens, and it is the ONLY new code in this lane: no GoalManager2, no
MissionV3, no second scheduler. Everything here either reads a
``GoalGraph`` or produces the exact objects ``WorkUnitDAG``/``Scheduler``/
``Mission`` already know how to consume.

THE PIPELINE, end to end: raw intent -> source preserved (``preserve_source``,
inside ``tokenize``) -> tokenize -> normalize + dedupe (``GoalGraph.add_node``,
inside ``ingest``) -> contradiction analysis (``GoalGraph.detect_conflicts``)
-> frontier extraction (``GoalGraph.ready_frontier``) -> feed the EXISTING
mission/WorkGraph machinery (``schedule`` returns a ``WorkUnitDAG`` whose
``.units`` is exactly what ``Scheduler.replan()`` or ``Mission(units=...)``
already accept). Calling ``ingest`` again with new prose against the same
graph, or ``sync_from_dag`` after real WorkUnits finish, is how the output
stays dynamic and recompilable instead of a one-time static plan.

ANTI-OVERDECOMPOSITION: exactly one node counts as one schedulable atom.
``_UNIT_TYPES`` (OBJECTIVE/SUBOBJECTIVE -- everything ``ready_frontier``
calls frontier-shaped except the root ULTRAGOAL restatement) each become AT
MOST two WorkUnits, an implement and a validate, mirroring the shape
``hcli.goal.GoalCompiler`` already uses for a small obligation set. A
directive with twenty sentences and three real objectives produces three
node-pairs, never twenty.

TWO CHECKS BEFORE ANY IMPLEMENTATION WORK IS GENERATED, both required by
defects this campaign already paid for once:

* GOAL SATISFACTION FROM DISK (``check_disk_satisfaction``) -- when a
  node's own statement yields a real, falsifiable command (the same
  derivation ``GoalCompiler`` uses), run it FIRST. If it already passes,
  the node is marked COMPLETE and no WorkUnit is emitted for it at all --
  prose claiming work remains does not outrank a passing check.
* CAPABILITY REACHABILITY (``load_reachability_snapshot`` /
  ``_capability_notes``) -- reuses ``tools/future/capability_reachability``
  rather than building a second analyzer. That module does a whole-repo AST
  scan (tens of seconds); this module NEVER runs it implicitly. A caller
  computes the snapshot once, out of band, and passes it to ``schedule``.
  A node whose named file is a real, already-callable capability, or one
  that is defined but has no caller outside its own test, gets an advisory
  note on its WorkUnit -- surfaced as text, never silently used to skip or
  rewrite the unit. "X exists but nothing calls it" is a different fix than
  "build X", and only a human/model reading the note can tell which.
"""
from __future__ import annotations

import dataclasses
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .goal import GoalCompiler, WorkUnitDAG
from .goal_graph import Conflict, FRONTIER_TYPES, GoalGraph
from .goal_ir import (
    PRIORITY_MAX,
    GoalNode,
    GoalType,
    Status,
    can_transition,
    transition,
)
from .goal_tokenizer import tokenize
from .paste_cache import PasteCache
from .verifier_pipeline import command_is_admissible

DISK_CHECK_TIMEOUT_S = 20.0

# GoalTypes whose statement is non-negotiable -- these become "invariants"
# in the compiled IR dict, the same key Mission._maybe_compile()/
# compile_worker_context already read off GoalCompiler.compile()'s output.
_INVARIANT_TYPES = frozenset({GoalType.HARD_CONSTRAINT, GoalType.PROHIBITION, GoalType.ANTI_GOAL})

# Frontier-shaped work minus the root restatement. ULTRAGOAL is a
# FRONTIER_TYPES member in goal_graph.py (so ready_frontier() lists it
# whenever it has no unresolved precedence predecessor, which is always,
# since nothing points AT the root) but it is the directive's own heading
# restated, not a piece of work distinct from the objectives under it --
# see _schedulable_ids for the one place that carve-out matters.
_UNIT_TYPES = FRONTIER_TYPES - {GoalType.ULTRAGOAL}

# Reused, not reimplemented: the tokenizer lane already borrows
# GoalCompiler's private file-reference regex for the same reason (see its
# module docstring) -- a parallel verify-command/acceptance deriver here
# would be exactly the second analyzer the module docstring above forbids.
_compiler = GoalCompiler()


# ---------------------------------------------------------------------------
# ingest: tokenize + fold into a graph (normalize/dedupe is GoalGraph's job)
# ---------------------------------------------------------------------------


def ingest(
    text: str,
    graph: Optional[GoalGraph] = None,
    *,
    cache: PasteCache,
    mission: Optional[str] = None,
    parent_ultragoal: Optional[str] = None,
) -> GoalGraph:
    """Tokenize *text* and fold every atom into *graph* (a fresh one if
    ``None``). Calling this again with new prose against the same graph is
    the recompilation entry point: restated atoms merge via
    ``content_signature`` (see ``GoalGraph.add_node``), new atoms are added,
    and anything the new text does not mention is left untouched. It is
    NOT how a hypothesis gets falsified or a node gets marked satisfied by
    real evidence -- that needs ``apply_evidence`` or ``sync_from_dag``,
    because prose alone cannot prove either.
    """
    graph = graph if graph is not None else GoalGraph()
    for node in tokenize(text, cache=cache, mission=mission, parent_ultragoal=parent_ultragoal):
        graph.add_node(node)
    return graph


# ---------------------------------------------------------------------------
# Check 1: goal satisfaction from disk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiskCheck:
    command: str
    passed: bool
    output: str


def check_disk_satisfaction(
    node: GoalNode, *, timeout: float = DISK_CHECK_TIMEOUT_S
) -> Optional[DiskCheck]:
    """Run the same falsifiable command ``GoalCompiler`` would derive for
    this node, if one is derivable, and report whether it ALREADY passes.

    Returns ``None`` when no real check can be derived (most statements --
    see ``GoalCompiler._verify_command``'s own "Honest: no verifier yet").
    ``None`` means "cannot tell", never "satisfied": a missing check is not
    evidence, and this function refuses to treat it as any.
    """
    command = _compiler._verify_command(node.statement, "", list(node.resources))
    if not command:
        return None
    admitted, reason = command_is_admissible(command)
    if not admitted:
        return DiskCheck(command=command, passed=False, output=reason)
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return DiskCheck(command=command, passed=proc.returncode == 0, output=output[-2000:])
    except subprocess.TimeoutExpired:
        return DiskCheck(command=command, passed=False, output=f"TIMEOUT after {timeout}s")


# ---------------------------------------------------------------------------
# Check 2: capability reachability
# ---------------------------------------------------------------------------


def load_reachability_snapshot() -> Dict[str, Any]:
    """Run the real analyzer and return its ``capabilities`` map.

    EXPENSIVE -- a whole-repo AST scan (tens of seconds). ``schedule()``
    never calls this on its own; compute it once, out of band, and pass it
    in as ``reachability_snapshot``. This module reads that analyzer's
    verdicts, it does not build a second one.
    """
    from tools.future.capability_reachability import assemble

    return assemble()["capabilities"]


def _capability_notes(node: GoalNode, snapshot: Mapping[str, Any]) -> List[str]:
    """Advisory strings from real call-site evidence -- never a verdict
    this module invents, and never used to silently skip or alter a unit.
    A resource file matching a DEFINED-but-not-CALLABLE capability is the
    "X exists but has no caller" case: the missing edge may be wiring it
    in, not rebuilding it, and only a reader of the note can judge that.
    """
    notes: List[str] = []
    resources = set(node.resources)
    if not resources:
        return notes
    for name, cap in snapshot.items():
        if not isinstance(cap, Mapping):
            continue
        definition = cap.get("definition")
        path = definition.get("file") if isinstance(definition, Mapping) else None
        if not path or path not in resources:
            continue
        if cap.get("callable"):
            notes.append(
                f"capability already reachable: {name} ({path}) has a real "
                "caller -- confirm this is not already satisfied before "
                "building more"
            )
        elif cap.get("defined"):
            notes.append(
                f"capability exists but is unreachable: {name} ({path}) has "
                "no call site outside its own test"
            )
    return notes


# ---------------------------------------------------------------------------
# frontier -> WorkUnitDAG
# ---------------------------------------------------------------------------


def _schedulable_ids(graph: GoalGraph) -> List[str]:
    """Every OBJECTIVE/SUBOBJECTIVE currently ready, plus any ready
    ULTRAGOAL that has nothing under it (a bare one-line directive that
    mints only a root heading must still produce work). Checked per
    ULTRAGOAL, not gated on whether *some* objective is ready elsewhere in
    the graph -- a multi-directive graph can have one ultragoal with real
    objectives and a second, still-childless one, at the same time.
    """
    ready = graph.ready_frontier()
    ids = [nid for nid in ready if graph.nodes[nid].type in _UNIT_TYPES]
    for nid in ready:
        node = graph.nodes[nid]
        if node.type is not GoalType.ULTRAGOAL:
            continue
        if not any(n.parent_ultragoal == nid for n in graph.nodes.values()):
            ids.append(nid)
    return ids


@dataclass
class CompileResult:
    graph: GoalGraph
    dag: WorkUnitDAG
    compiled_ir: Dict[str, Any]
    satisfied: Tuple[str, ...] = ()
    conflicts: Tuple[Conflict, ...] = ()
    notes: Dict[str, Tuple[str, ...]] = field(default_factory=dict)


def schedule(
    graph: GoalGraph,
    *,
    max_active: Optional[int] = None,
    reachability_snapshot: Optional[Mapping[str, Any]] = None,
    check_disk: bool = True,
    disk_timeout: float = DISK_CHECK_TIMEOUT_S,
) -> CompileResult:
    """Compile *graph*'s CURRENT ready frontier into a fresh ``WorkUnitDAG``.

    Idempotent and safe to call repeatedly as the graph changes: a node
    whose statement/resources have not changed since the last call
    produces byte-identical WorkUnits (see ``workunit.content_identity``),
    so feeding the result to ``Scheduler.replan()`` a second time is a
    no-op for unchanged work, not a duplicate. A node only appears here
    once its precedence predecessors are resolved (``ready_frontier``), so
    cross-node WorkUnit dependencies are never needed within one call --
    a blocked SUBOBJECTIVE simply is not in this batch yet.
    """
    if max_active is not None:
        graph.bound_active_frontier(max_active)
    conflicts = tuple(graph.detect_conflicts())

    dag = WorkUnitDAG()
    satisfied: List[str] = []
    notes: Dict[str, Tuple[str, ...]] = {}
    obligations: List[Dict[str, Any]] = []

    for node_id in _schedulable_ids(graph):
        node = graph.nodes[node_id]

        if check_disk:
            disk = check_disk_satisfaction(node, timeout=disk_timeout)
            if disk is not None and disk.passed:
                graph.update_node(
                    transition(
                        node,
                        Status.COMPLETE,
                        reopen_condition=f"reopen if `{disk.command}` starts failing",
                    )
                )
                satisfied.append(node_id)
                continue

        node_notes: Tuple[str, ...] = ()
        if reachability_snapshot is not None:
            node_notes = tuple(_capability_notes(node, reachability_snapshot))
            if node_notes:
                notes[node_id] = node_notes

        description = node.statement
        if node.resources:
            description += " Relevant files: " + ", ".join(node.resources[:6])
        for extra in node_notes:
            description += f" NOTE: {extra}"

        implement_id = f"{node_id}.implement"
        dag.add_unit(implement_id, description, [], role="implementation")

        verify = _compiler._verify_command(node.statement, "", list(node.resources))
        acceptance = _compiler._falsifiable_acceptance(
            node.statement, "", list(node.resources), []
        )
        dag.add_unit(
            f"{node_id}.validate",
            f"Validate {node_id} against its acceptance criteria: {acceptance}",
            [implement_id],
            role="validation",
            verifier=verify or None,
            resource_class="TEST" if verify else None,
            preferred_backend="cpu" if verify else None,
        )
        obligations.append(
            {
                "id": node_id,
                "text": node.statement,
                "acceptance": acceptance,
                "verify": verify,
                "role": "implementation",
                "kind": "goal",
            }
        )

    compiled_ir = _to_compiled_ir(graph, dag, obligations)
    return CompileResult(
        graph=graph,
        dag=dag,
        compiled_ir=compiled_ir,
        satisfied=tuple(satisfied),
        conflicts=conflicts,
        notes=notes,
    )


def _to_compiled_ir(
    graph: GoalGraph, dag: WorkUnitDAG, obligations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Same shape ``GoalCompiler.compile()`` returns, so a caller can drop
    this straight into ``mission._compiled`` -- see the module docstring
    on why that is the whole integration, not a new Mission constructor.
    """
    active = [n for n in graph.nodes.values() if n.status is Status.ACTIVE]
    ultra = next((n for n in active if n.type is GoalType.ULTRAGOAL), None)
    objective = next((n for n in active if n.type is GoalType.OBJECTIVE), None)
    anchor = ultra or objective
    goal_summary = anchor.statement if anchor else ""

    referenced_files: List[str] = []
    for n in graph.nodes.values():
        for path in n.resources:
            if path not in referenced_files:
                referenced_files.append(path)

    acceptance = [n.statement for n in active if n.type is GoalType.SUCCESS_CRITERION]
    acceptance += [str(ob["acceptance"]) for ob in obligations if ob.get("acceptance")]
    acceptance = list(dict.fromkeys(acceptance))
    if not acceptance:
        acceptance = [
            "an independent check of the requested behavior can fail; "
            "restating the request is not acceptance"
        ]

    return {
        "goal": goal_summary,
        "goal_summary": goal_summary,
        "invariants": [n.statement for n in active if n.type in _INVARIANT_TYPES],
        "acceptance_criteria": acceptance,
        "referenced_files": referenced_files,
        "obligations": obligations,
        "workunits": dag,
    }


def sync_from_dag(graph: GoalGraph, units: Mapping[str, Any]) -> Tuple[str, ...]:
    """Fold WorkUnit completions back into the graph.

    ``schedule()`` turns a ready GoalNode into WorkUnits; this is the other
    half of the loop, turning a finished WorkUnit back into a COMPLETE
    GoalNode so the NEXT ``schedule()`` call sees a wider ready frontier
    (a SUBOBJECTIVE that was waiting on its OBJECTIVE, for instance).
    Neither direction is a second scheduler: both read/write the exact
    ``WorkUnit.status`` field ``Scheduler.complete()`` already gates on a
    passing verifier before setting.
    """
    closed: List[str] = []
    for node_id, node in list(graph.nodes.items()):
        if node.status is not Status.ACTIVE or node.type not in (_UNIT_TYPES | {GoalType.ULTRAGOAL}):
            continue
        validate = units.get(f"{node_id}.validate")
        implement = units.get(f"{node_id}.implement")
        target = validate if validate is not None else implement
        if target is None or getattr(target, "status", None) != "completed":
            continue
        graph.update_node(transition(node, Status.COMPLETE))
        closed.append(node_id)
    return tuple(closed)


# ---------------------------------------------------------------------------
# Recompilation from evidence (not from new prose -- see ingest())
# ---------------------------------------------------------------------------


class EvidenceOutcome(str, Enum):
    CONFIRMED = "CONFIRMED"
    FALSIFIED = "FALSIFIED"
    SATISFIED = "SATISFIED"


def apply_evidence(
    graph: GoalGraph,
    node_id: str,
    outcome: "EvidenceOutcome | str",
    *,
    note: str = "",
) -> GoalNode:
    """Recompile exactly ONE node in light of real evidence.

    "A falsified hypothesis drops priority while the ultragoal survives"
    means precisely that: no parent, dependent, or sibling is touched.
    FALSIFIED demotes priority/confidence and parks the node (when parking
    is legal from its current status; a node already PARKED/SUPERSEDED is
    just demoted further). SATISFIED completes it outright -- for a case
    ``check_disk_satisfaction`` cannot reach on its own, e.g. a human or a
    model reporting a real, already-run check. CONFIRMED restores full
    confidence without changing status: evidence for a live hypothesis
    does not promote it to fact.
    """
    kind = outcome if isinstance(outcome, EvidenceOutcome) else EvidenceOutcome(str(outcome))
    node = graph.nodes[node_id]
    if kind is EvidenceOutcome.SATISFIED:
        updated = transition(node, Status.COMPLETE, reopen_condition=note or None)
    elif kind is EvidenceOutcome.FALSIFIED:
        demoted = dataclasses.replace(
            node,
            priority=min(PRIORITY_MAX, node.priority + 1),
            confidence=min(node.confidence, 0.2),
        )
        if can_transition(demoted.status, Status.PARKED):
            updated = transition(demoted, Status.PARKED, reopen_condition=note or None)
        else:
            updated = demoted
    else:
        updated = dataclasses.replace(node, confidence=1.0)
    return graph.update_node(updated)


# ---------------------------------------------------------------------------
# Read-only inspection surface. Wire this up as a /goals command in
# hcli/command_registry.py -- reported as out_of_scope, not edited here.
# ---------------------------------------------------------------------------


def goals_report(graph: GoalGraph) -> Dict[str, Any]:
    """A read-only snapshot: no mutation, safe to call at any time."""
    return {
        "frontier": graph.frontier_report(),
        "ready": graph.ready_frontier(),
        "conflicts": [
            {"a_id": c.a_id, "b_id": c.b_id, "kind": c.kind.value, "reason": c.reason}
            for c in graph.detect_conflicts()
        ],
        "nodes": {
            nid: {
                "type": n.type.value,
                "status": n.status.value,
                "priority": n.priority,
                "confidence": n.confidence,
                "statement": n.statement,
                "parent_ultragoal": n.parent_ultragoal,
            }
            for nid, n in graph.nodes.items()
        },
    }


if __name__ == "__main__":
    import tempfile

    from .scheduler import Scheduler

    _DEMO = """# Reduce Odyssey wall time

Make Odyssey faster by caching models on SSD.

Do not delete source specimens.

## Acceptance Criteria
- the full suite passes with `pytest hcli/`

Hypothesis: the slowdown is disk contention.
"""
    with tempfile.TemporaryDirectory() as tmp:
        cache = PasteCache(root=tmp)
        graph = ingest(_DEMO, cache=cache)

        result = schedule(graph)
        # Exactly one schedulable node (the OBJECTIVE) -> exactly two units.
        # The ULTRAGOAL is not separately scheduled (see _schedulable_ids)
        # and the HARD_CONSTRAINT/HYPOTHESIS/SUCCESS_CRITERION never become
        # units at all -- anti-overdecomposition holding by construction.
        assert len(result.dag.units) == 2, result.dag.units
        objective_id = next(
            nid for nid, n in graph.nodes.items() if n.type.value == "OBJECTIVE"
        )
        assert f"{objective_id}.implement" in result.dag.units
        assert f"{objective_id}.validate" in result.dag.units
        assert "delete source specimens" in result.compiled_ir["invariants"][0]
        assert any("suite passes" in a for a in result.compiled_ir["acceptance_criteria"])

        # Feed the EXISTING scheduler -- no parallel system. replan() is
        # idempotent for unchanged content, so calling schedule() again
        # (recompilation) and replanning again is always safe.
        sched = Scheduler(dict(result.dag.units), 1, workspace=tmp)
        again = schedule(graph)
        outcomes = sched.replan(again.dag.units)
        assert all(o.kind == "idempotent" for o in outcomes), outcomes

        # Close the loop: mark the validate unit completed the way a real
        # run would (Scheduler.complete() itself gates this on a passing
        # verifier -- sync_from_dag only reads the resulting status, it
        # does not decide whether the unit deserved to reach it).
        sched.units[f"{objective_id}.validate"].status = "completed"
        closed = sync_from_dag(graph, sched.units)
        assert closed == (objective_id,), closed
        assert graph.nodes[objective_id].status.value == "COMPLETE"

        # Falsify the hypothesis: the ultragoal is untouched.
        hyp_id = next(nid for nid, n in graph.nodes.items() if n.type.value == "HYPOTHESIS")
        ultra_id = next(nid for nid, n in graph.nodes.items() if n.type.value == "ULTRAGOAL")
        before_priority = graph.nodes[hyp_id].priority
        apply_evidence(graph, hyp_id, EvidenceOutcome.FALSIFIED, note="measured: not disk-bound")
        assert graph.nodes[hyp_id].priority > before_priority
        assert graph.nodes[hyp_id].status.value == "PARKED"
        assert graph.nodes[ultra_id].status.value == "ACTIVE"

        report = goals_report(graph)
        assert objective_id in report["nodes"]

        print(f"OK: {len(graph.nodes)} nodes, {len(result.dag.units)} units scheduled")
