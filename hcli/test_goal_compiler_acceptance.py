"""Acceptance gate for the goal-compiler lane (goal_ir/goal_tokenizer/goal_graph/goal_compile).

This is the ACCEPTANCE GATE, not a repair crew: it owns no implementation
file in that lane. Every assertion below states what a correct compilation
looks like per the steer that commissioned this lane; where the real
compiler falls short, the test fails and says so -- it does not lower the
bar to match what the code happens to do today (a green suite that dodges
the hard case is worse than a red one that names it).

Three scenarios, verbatim from the steer:

* TEST 1 -- a genesis-scale directive: sections, prohibitions, examples,
  acceptance/failure criteria. The compiler must recover a durable
  ultragoal, active frontiers, parked-but-not-deleted work, constraints,
  evidence doctrine and authority, with NO omission of any hard constraint,
  stop condition, authority boundary or explicit success criterion.
* TEST 2 -- the messy update ("forget X for now; improve Y, especially Z,
  but don't abandon W"): Y must be elevated, X parked (not deleted), Z a
  HYPOTHESIS/measurement target (not a proven fact), W preserved as its own
  atom.
* TEST 3 -- an implementation-heavy instruction ("use A, B and C to get D
  under 24h"): D is the OBJECTIVE, A/B/C are SUGGESTED_METHOD, never
  hardened into requirements.

Plus the structural invariants the steer names explicitly.
"""
from __future__ import annotations

import pytest

from hcli.goal_compile import ingest, schedule
from hcli.goal_ir import GoalNode, GoalType, Provenance, Status, preserve_source
from hcli.paste_cache import PasteCache
from hcli.goal_tokenizer import tokenize


def _cache(tmp_path) -> PasteCache:
    return PasteCache(root=tmp_path)


def _dump(graph) -> str:
    lines = [
        f"  {n.id} [{n.type.value}/{n.status.value}] {n.statement!r}"
        for n in graph.nodes.values()
    ]
    return "\n".join(lines) if lines else "  (no nodes at all)"


# =============================================================================
# TEST 1 -- the genesis / large directive
# =============================================================================

# Constructed, not copied from a real steer: representative of the shape
# (sections, prohibitions, examples, acceptance criteria) the steer asks
# for, but every sentence below is hand-traced against the ACTUAL
# hcli/goal_tokenizer.py classifier tables so this test names the real
# compiler's behavior, not a wish.
GENESIS_DIRECTIVE = """# Reduce Odyssey Wall Time

Make Odyssey wall time fall under a full day by caching finished models on local SSD.

Improve representation fidelity by widening the activation capture window.

## Constraints

- Do not delete any source specimen once it has been captured.
- Never let Claude remain the hot-loop orchestrator for more than one cycle.
- SSH access to the training box is forbidden for any automated agent.

## Authority

- Ask before deleting anything in the receipts directory.
- HCLI is authorized to land a self-verified change without human review.

## Evidence Doctrine

- Every claim must produce a receipt.
- Verify with `pytest hcli/ -q` before calling a mission complete.

## Acceptance Criteria

- the full suite passes with `pytest hcli/ -q`
- Odyssey wall time is measured under 24 hours on real hardware

## Failure Criteria

- any previously-passing test regresses

Stop when the resident has been idle for six consecutive cycles.

Hypothesis: the wall-time floor is disk read contention, not compute.

For example, prefetching the next model while the current one trains is one mitigation.

Not now, but someday consider distributed training across more than one box.

Should the next campaign prioritize latency or throughput?
"""


def test_genesis_directive_recovers_full_doctrine_with_no_omission(tmp_path):
    graph = ingest(GENESIS_DIRECTIVE, cache=_cache(tmp_path))
    dump = _dump(graph)

    by_type = {}
    for n in graph.nodes.values():
        by_type.setdefault(n.type, []).append(n)

    # -- a durable ultragoal ------------------------------------------------
    ultras = by_type.get(GoalType.ULTRAGOAL, [])
    assert len(ultras) == 1, f"expected exactly one ULTRAGOAL, got:\n{dump}"
    assert "odyssey wall time" in ultras[0].statement.lower(), ultras[0]

    # -- active frontiers: both objectives ready to schedule now -----------
    objectives = by_type.get(GoalType.OBJECTIVE, [])
    assert len(objectives) == 2, f"expected two OBJECTIVEs, got:\n{dump}"
    ready = set(graph.ready_frontier())
    for obj in objectives:
        assert obj.id in ready, f"{obj.id} must be an active frontier, ready set was {ready}\n{dump}"

    # methods stay methods -- never folded into the objective statement,
    # and each is a distinct node the objective can supersede independently.
    methods = by_type.get(GoalType.SUGGESTED_METHOD, [])
    assert len(methods) == 2, f"expected two SUGGESTED_METHODs, got:\n{dump}"
    method_text = " ".join(m.statement.lower() for m in methods)
    assert "ssd" in method_text and "activation capture window" in method_text, dump
    for m in methods:
        assert m.dependencies, f"a method must link back to the objective it serves:\n{dump}"

    # -- parked-but-not-deleted work: the "someday" option is retained but
    # never enters the schedulable frontier (FUTURE_OPTION is not a
    # FRONTIER_TYPES member) -- captured, not discarded, not actioned now.
    future = by_type.get(GoalType.FUTURE_OPTION, [])
    assert len(future) == 1, f"expected one FUTURE_OPTION, got:\n{dump}"
    assert future[0].id in graph.nodes, "deferred work must stay addressable, never vanish"
    assert future[0].id not in ready, "deferred work must not silently become active work"

    # -- constraints: hard constraint, anti-goal and prohibition are three
    # DIFFERENT types, not flattened into one bucket -----------------------
    hard = by_type.get(GoalType.HARD_CONSTRAINT, [])
    anti = by_type.get(GoalType.ANTI_GOAL, [])
    prohibition = by_type.get(GoalType.PROHIBITION, [])
    assert len(hard) == 1 and "delete" in hard[0].statement.lower(), dump
    assert len(anti) == 1 and "remain" in anti[0].statement.lower(), dump
    assert len(prohibition) == 1 and "ssh" in prohibition[0].statement.lower(), dump

    # -- evidence doctrine ---------------------------------------------------
    evidence = by_type.get(GoalType.EVIDENCE_REQUIREMENT, [])
    assert len(evidence) == 2, f"expected two EVIDENCE_REQUIREMENTs, got:\n{dump}"

    # -- authority: grant and requirement are opposite verbs, kept distinct -
    auth_required = by_type.get(GoalType.AUTHORITY_REQUIRED, [])
    auth_grant = by_type.get(GoalType.AUTHORITY_GRANT, [])
    assert len(auth_required) == 1 and "ask before" in auth_required[0].statement.lower(), dump
    assert len(auth_grant) == 1 and "authorized" in auth_grant[0].statement.lower(), dump

    # -- OMISSION CHECK: hard constraint, stop condition, authority
    # boundary, explicit success criterion must each be present AND trace
    # back to the literal source sentence that stated them. ----------------
    stop = by_type.get(GoalType.STOP_CONDITION, [])
    success = by_type.get(GoalType.SUCCESS_CRITERION, [])
    failure = by_type.get(GoalType.FAILURE_CRITERION, [])
    assert len(stop) == 1, f"stop condition must not be dropped, got:\n{dump}"
    assert len(success) == 2, f"both acceptance-criteria bullets must survive, got:\n{dump}"
    assert len(failure) == 1, f"failure criterion must not be dropped, got:\n{dump}"

    must_not_be_dropped = [hard[0], stop[0], auth_required[0], success[0], success[1]]
    for node in must_not_be_dropped:
        assert node.source_refs, f"{node.id} has no source_ref -- untraceable:\n{dump}"
        sref = node.source_refs[0]
        span = GENESIS_DIRECTIVE[sref.char_start:sref.char_end]
        # the node's own key words must actually appear in the span it
        # claims to come from -- a real trace, not a coincidence.
        key_word = node.statement.lower().split()[-1].strip(".,;:")
        assert key_word in span.lower(), (
            f"{node.id} claims source span {span!r} but its statement "
            f"{node.statement!r} does not trace to it"
        )

    # -- anti-overdecomposition: twenty-odd goal atoms, but only the two
    # real objectives become schedulable work (implement+validate each) --
    # never one WorkUnit per sentence.
    result = schedule(graph, check_disk=False)
    assert len(graph.nodes) >= 15, "the directive should mint many distinct atoms"
    assert len(result.dag.units) == 4, (
        f"two objectives must produce exactly four WorkUnits (implement+validate "
        f"each), not one per sentence; got {sorted(result.dag.units)}"
    )

    # a genesis-scale directive with unrelated grants/prohibitions must not
    # spuriously flag a conflict just because both types exist somewhere.
    assert graph.detect_conflicts() == [], graph.detect_conflicts()


# =============================================================================
# TEST 2 -- the messy update
# =============================================================================


def test_messy_update_elevates_parks_and_measures_instead_of_asserting(tmp_path):
    """"Forget deep SUB2 for now; improve the environment, especially
    Odyssey HDD bottlenecks, but don't abandon representation work."

    Correct compilation per the steer: environment ELEVATED; SUB2
    PARKED_HIGH_VALUE and NOT deleted; HDD a HYPOTHESIS / target of
    measurement, NOT a proven bottleneck -- that last distinction is the
    whole point, asserted explicitly below.
    """
    text = (
        "Forget deep SUB2 for now; improve the environment, especially "
        "Odyssey HDD bottlenecks, but don't abandon representation work."
    )
    graph = ingest(text, cache=_cache(tmp_path))
    dump = _dump(graph)

    environment_nodes = [
        n for n in graph.nodes.values()
        if n.type in (GoalType.OBJECTIVE, GoalType.SUBOBJECTIVE)
        and "environment" in n.statement.lower()
    ]
    assert len(environment_nodes) == 1, (
        "'improve the environment' must compile to its own elevated OBJECTIVE, "
        f"distinct from the SUB2 park and the HDD hypothesis. Compiled graph:\n{dump}"
    )
    environment = environment_nodes[0]
    assert environment.priority < 2, (
        "'especially' must elevate this objective's priority below the default "
        f"(2); got priority={environment.priority}. Compiled graph:\n{dump}"
    )

    sub2_nodes = [n for n in graph.nodes.values() if "sub2" in n.statement.lower()]
    assert len(sub2_nodes) == 1, f"SUB2 must survive as its own atom, got:\n{dump}"
    sub2 = sub2_nodes[0]
    assert sub2.status is Status.PARKED, (
        f"'forget ... for now' must PARK sub2 (high-value, revisitable), not "
        f"delete it or leave it ACTIVE; got status={sub2.status.value}. Graph:\n{dump}"
    )
    assert sub2.id in graph.nodes, "a parked goal must stay addressable, never removed"

    hdd_nodes = [n for n in graph.nodes.values() if "hdd" in n.statement.lower()]
    assert len(hdd_nodes) == 1, f"the HDD bottleneck must compile to its own atom, got:\n{dump}"
    hdd = hdd_nodes[0]
    assert hdd.type is GoalType.HYPOTHESIS, (
        "HDD is named as something to MEASURE, not a proven bottleneck -- it "
        f"must compile to HYPOTHESIS, not {hdd.type.value}. This is the "
        f"load-bearing distinction the steer names explicitly. Graph:\n{dump}"
    )

    repr_nodes = [
        n for n in graph.nodes.values()
        if "representation work" in n.statement.lower()
        and n.id not in {getattr(x, "id", None) for x in (environment_nodes + sub2_nodes + hdd_nodes)}
    ]
    assert len(repr_nodes) == 1, (
        "'don't abandon representation work' must survive as its own "
        f"protective atom, not fused into the SUB2/HDD/environment text. Graph:\n{dump}"
    )


# =============================================================================
# TEST 3 -- implementation-heavy instruction
# =============================================================================


def test_implementation_heavy_instruction_keeps_methods_suggested(tmp_path):
    """"Use SSD cache, prefetch and model grouping to get Odyssey under 24h."

    Correct compilation: OBJECTIVE "Odyssey under 24h"; SSD/prefetch/
    grouping are SUGGESTED_METHOD, not requirements, unless the user made
    them mandatory (they did not, here) -- HCLI must stay free to find a
    better solution, so none of the three may harden into a HARD_CONSTRAINT.
    """
    text = "Use SSD cache, prefetch and model grouping to get Odyssey under 24h."
    graph = ingest(text, cache=_cache(tmp_path))
    dump = _dump(graph)

    objectives = [
        n for n in graph.nodes.values()
        if n.type in (GoalType.OBJECTIVE, GoalType.SUBOBJECTIVE)
        and "odyssey" in n.statement.lower()
    ]
    assert len(objectives) == 1, (
        "a method-first sentence ('use A, B and C to get <outcome>') must "
        f"still compile to an OBJECTIVE naming the outcome; got:\n{dump}"
    )
    assert "24h" in objectives[0].statement.lower().replace(" ", ""), objectives[0]

    methods = [n for n in graph.nodes.values() if n.type is GoalType.SUGGESTED_METHOD]
    method_text = " ".join(n.statement.lower() for n in methods)
    for keyword in ("ssd", "prefetch", "grouping"):
        assert keyword in method_text, (
            f"{keyword!r} must be captured as a SUGGESTED_METHOD (revisable "
            f"if it fails its verifier), not dropped or hardened. Graph:\n{dump}"
        )
        assert not any(
            n.type is GoalType.HARD_CONSTRAINT and keyword in n.statement.lower()
            for n in graph.nodes.values()
        ), (
            f"{keyword!r} was not declared mandatory by the user -- HCLI must "
            f"stay free to find a better solution. Graph:\n{dump}"
        )
    for m in methods:
        assert m.dependencies and any(dep in {o.id for o in objectives} for dep in m.dependencies), (
            f"each SUGGESTED_METHOD must link back to the objective it serves:\n{dump}"
        )


def test_quoted_counterexample_does_not_become_an_objective(tmp_path):
    text = (
        "Do not interpret this as:\n\n"
        '    "make the resident model as smart as Claude or Grok."\n\n'
        "The resident is one component.\n"
    )
    nodes = tokenize(text, cache=_cache(tmp_path))
    assert not any(
        node.type is GoalType.OBJECTIVE
        and "resident model" in node.statement.lower()
        for node in nodes
    ), nodes
    assert any(
        node.type is GoalType.EXAMPLE
        and "resident model" in node.statement.lower()
        for node in nodes
    ), nodes


# =============================================================================
# Structural invariants
# =============================================================================


def test_explicit_user_provenance_requires_literal_source(tmp_path):
    """EXPLICIT_USER means the human said exactly this. goal_ir enforces the
    guard; this acceptance gate locks onto it so a regression there cannot
    slip past the very suite meant to catch a silently-invented obligation."""
    cache = PasteCache(root=tmp_path)
    sref = preserve_source(cache, "Ship the compiler curriculum by Friday.")

    node = GoalNode(
        id="OBJECTIVE_SHIP_CURRICULUM",
        type=GoalType.OBJECTIVE,
        statement="ship the compiler curriculum",
        provenance=Provenance.EXPLICIT_USER,
        source_refs=(sref,),
    )
    assert node.provenance is Provenance.EXPLICIT_USER

    with pytest.raises(ValueError):
        GoalNode(
            id="OBJECTIVE_INFERRED_AS_EXPLICIT",
            type=GoalType.OBJECTIVE,
            statement="an inferred objective nobody actually said",
            provenance=Provenance.EXPLICIT_USER,
            source_refs=(),
        )


@pytest.mark.parametrize(
    "text",
    [
        GENESIS_DIRECTIVE,
        "Forget deep SUB2 for now; improve the environment, especially "
        "Odyssey HDD bottlenecks, but don't abandon representation work.",
        "Use SSD cache, prefetch and model grouping to get Odyssey under 24h.",
    ],
)
def test_every_node_traces_to_a_source_ref(tmp_path, text):
    graph = ingest(text, cache=_cache(tmp_path))
    # An empty graph would vacuously satisfy "every node has a source_ref" --
    # guard against that so a directive the compiler drops entirely (a real
    # bug, caught explicitly elsewhere) cannot also read as a silent pass
    # here.
    assert graph.nodes, f"expected at least one compiled node for: {text!r}"
    for node in graph.nodes.values():
        assert node.source_refs, f"{node.id} has no source_ref: {node}"


def test_restated_goal_dedupes_to_one_node_with_two_source_refs(tmp_path):
    """Two separate directives (different paste content, e.g. two different
    conversations) both stating the identical sentence must fold to ONE
    node carrying TWO source_refs -- one per real provenance event.

    (A single ingest() of literally the same bytes twice is NOT this case:
    PasteCache.store() correctly returns the SAME paste for identical
    content -- "Identical content returns the existing ref" -- so replaying
    one paste verbatim is one provenance event, not two, and correctly
    yields one ref. Two refs requires two genuinely distinct source texts
    that happen to restate the same goal, which is what this test builds.)
    """
    sentence = "Make the sovereign loop self-correct within three hours."
    cache = _cache(tmp_path)
    graph = ingest(sentence, cache=cache)
    assert len(graph.nodes) == 1, graph.nodes
    node_id = next(iter(graph.nodes))
    assert len(graph.nodes[node_id].source_refs) == 1

    # A second, textually-distinct directive that restates the same goal.
    second_directive = sentence + "\n\n(said again in a follow-up)\n"
    ingest(second_directive, graph, cache=cache)
    assert len(graph.nodes) == 1, (
        f"restating the same goal in a new directive must fold into the "
        f"existing node, not mint a second one: {graph.nodes}"
    )
    assert len(graph.nodes[node_id].source_refs) == 2, (
        "a goal restated across two distinct directives must keep BOTH "
        f"provenance trails, not silently drop the earlier one: {graph.nodes[node_id]}"
    )


def test_one_objective_does_not_explode_into_dozens_of_workunits(tmp_path):
    text = (
        "# Reduce Odyssey Wall Time\n\n"
        "Make Odyssey wall time fall under a full day by caching finished models on local SSD.\n\n"
        "Do not delete source specimens.\n"
        "Prefer the local resident when possible.\n"
        "Never let Claude remain the hot-loop orchestrator.\n\n"
        "Success: the full suite passes with `pytest hcli/ -q`.\n"
        "Failure: any previously-passing test regresses.\n"
        "Hypothesis: the slowdown is disk contention.\n"
        "For example, a warm page cache would mask it.\n"
    )
    graph = ingest(text, cache=_cache(tmp_path))
    result = schedule(graph, check_disk=False)
    assert len(graph.nodes) >= 8, f"expected many distinct atoms, got:\n{_dump(graph)}"
    assert len(result.dag.units) == 2, (
        f"exactly one OBJECTIVE must yield exactly two WorkUnits "
        f"(implement+validate), never one per sentence; got {sorted(result.dag.units)}"
    )


if __name__ == "__main__":
    import subprocess
    import sys

    raise SystemExit(
        subprocess.call([sys.executable, "-m", "pytest", __file__, "-q"])
    )
