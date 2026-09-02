import dataclasses

import pytest

from hcli.goal_graph import (
    Conflict,
    ConflictKind,
    Edge,
    EdgeType,
    Frontier,
    GoalGraph,
    GoalIdCollisionError,
    classify_frontier,
)
from hcli.goal_ir import (
    GoalNode,
    GoalType,
    InvalidGoalTransitionError,
    Provenance,
    Status,
    preserve_source,
    transition,
)
from hcli.paste_cache import PasteCache


def _node(**overrides):
    """The cheapest legal node: MODEL_INFERRED needs no source_ref at all."""
    fields = dict(
        id="OBJECTIVE_REDUCE_WALL_TIME",
        type=GoalType.OBJECTIVE,
        statement="reduce odyssey wall time",
        provenance=Provenance.MODEL_INFERRED,
    )
    fields.update(overrides)
    return GoalNode(**fields)


# -- add_node / dedupe --------------------------------------------------------

def test_add_node_merges_exact_restatement_under_same_id():
    g = GoalGraph()
    g.add_node(_node())
    merged = g.add_node(_node(dependencies=("OBJECTIVE_OTHER",)))
    assert len(g.nodes) == 1
    assert merged.dependencies == ("OBJECTIVE_OTHER",)


def test_add_node_folds_same_signature_different_id_into_canonical(tmp_path):
    """THE LOAD-BEARING DEDUPE CASE: the same goal said three ways is one
    node with three source_refs, never three separate nodes."""
    cache = PasteCache(root=tmp_path)
    g = GoalGraph()
    sref1 = preserve_source(cache, "make odyssey faster pls")
    g.add_node(_node(id="OBJECTIVE_A", provenance=Provenance.EXPLICIT_USER, source_refs=(sref1,)))
    sref2 = preserve_source(cache, "please make odyssey faster")
    merged = g.add_node(
        _node(id="OBJECTIVE_B", provenance=Provenance.EXPLICIT_USER, source_refs=(sref2,))
    )
    assert merged.id == "OBJECTIVE_A"  # first-seen id wins
    assert "OBJECTIVE_B" not in g.nodes
    assert len(g.nodes) == 1
    assert len(merged.source_refs) == 2


def test_add_node_same_id_different_signature_is_refused():
    g = GoalGraph()
    g.add_node(_node())
    with pytest.raises(GoalIdCollisionError):
        g.add_node(_node(statement="a completely different claim"))


# -- update_node ---------------------------------------------------------------

def test_update_node_replaces_fields_in_place():
    g = GoalGraph()
    g.add_node(_node(priority=2))
    current = g.nodes["OBJECTIVE_REDUCE_WALL_TIME"]
    g.update_node(dataclasses.replace(current, priority=0))
    assert g.nodes["OBJECTIVE_REDUCE_WALL_TIME"].priority == 0


def test_update_node_unknown_id_raises():
    g = GoalGraph()
    with pytest.raises(KeyError):
        g.update_node(_node())


# -- edges -----------------------------------------------------------------

def test_add_edge_requires_both_nodes_present():
    g = GoalGraph()
    g.add_node(_node())
    with pytest.raises(KeyError):
        g.add_edge("OBJECTIVE_REDUCE_WALL_TIME", "OBJECTIVE_GHOST", EdgeType.REQUIRES)


def test_add_edge_rejects_self_loop():
    g = GoalGraph()
    g.add_node(_node())
    with pytest.raises(ValueError, match="itself"):
        g.add_edge(
            "OBJECTIVE_REDUCE_WALL_TIME", "OBJECTIVE_REDUCE_WALL_TIME", EdgeType.REQUIRES
        )


# -- supersession / temporal override ----------------------------------------

def test_apply_supersession_transitions_old_and_keeps_it_in_the_graph():
    g = GoalGraph()
    g.add_node(_node(id="OBJECTIVE_OLD"))
    g.add_node(_node(id="OBJECTIVE_NEW", statement="a newer, sharper claim"))
    g.apply_supersession("OBJECTIVE_NEW", "OBJECTIVE_OLD", note="user narrowed scope")
    assert g.nodes["OBJECTIVE_OLD"].status is Status.SUPERSEDED
    assert g.nodes["OBJECTIVE_OLD"].superseded_by == "OBJECTIVE_NEW"
    assert "OBJECTIVE_OLD" in g.nodes  # never erased
    edge = next(iter(g.edges))
    assert edge.type is EdgeType.SUPERSEDES
    assert edge.src == "OBJECTIVE_NEW" and edge.dst == "OBJECTIVE_OLD"


def test_apply_supersession_refuses_to_supersede_completed_work():
    g = GoalGraph()
    g.add_node(_node(id="OBJECTIVE_OLD", status=Status.COMPLETE))
    g.add_node(_node(id="OBJECTIVE_NEW", statement="a newer claim"))
    with pytest.raises(InvalidGoalTransitionError):
        g.apply_supersession("OBJECTIVE_NEW", "OBJECTIVE_OLD")


def test_merge_nodes_combines_evidence_and_supersedes_the_duplicate(tmp_path):
    """The prose example: "HCLI should improve itself" and "Hawking should
    build Hawking" recognized (by the caller) as one ultragoal."""
    cache = PasteCache(root=tmp_path)
    g = GoalGraph()
    sref_a = preserve_source(cache, "HCLI should improve itself")
    g.add_node(
        _node(
            id="ULTRAGOAL_HCLI_SELF_IMPROVEMENT",
            type=GoalType.ULTRAGOAL,
            statement="HCLI improves itself",
            provenance=Provenance.EXPLICIT_USER,
            source_refs=(sref_a,),
        )
    )
    sref_b = preserve_source(cache, "Hawking should build Hawking")
    g.add_node(
        _node(
            id="ULTRAGOAL_HAWKING_BUILDS_HAWKING",
            type=GoalType.ULTRAGOAL,
            statement="Hawking builds Hawking",
            provenance=Provenance.EXPLICIT_USER,
            source_refs=(sref_b,),
        )
    )
    merged = g.merge_nodes("ULTRAGOAL_HCLI_SELF_IMPROVEMENT", "ULTRAGOAL_HAWKING_BUILDS_HAWKING")
    assert len(merged.source_refs) == 2
    old = g.nodes["ULTRAGOAL_HAWKING_BUILDS_HAWKING"]
    assert old.status is Status.SUPERSEDED
    assert old.superseded_by == "ULTRAGOAL_HCLI_SELF_IMPROVEMENT"


def test_merge_nodes_refuses_a_provenance_incompatible_fold(tmp_path):
    cache = PasteCache(root=tmp_path)
    g = GoalGraph()
    g.add_node(_node(id="OBJECTIVE_A", provenance=Provenance.MODEL_INFERRED))
    sref = preserve_source(cache, "actually the user said this")
    g.add_node(
        _node(
            id="OBJECTIVE_B",
            statement="a different claim",
            provenance=Provenance.EXPLICIT_USER,
            source_refs=(sref,),
        )
    )
    with pytest.raises(ValueError, match="MODEL_INFERRED must not carry"):
        g.merge_nodes("OBJECTIVE_A", "OBJECTIVE_B")


# -- conflicts -----------------------------------------------------------------

def _opposed_pair(status_a=Status.ACTIVE, status_b=Status.ACTIVE):
    g = GoalGraph()
    g.add_node(
        _node(
            id="AUTHORITY_GRANT_DELETE_PARTIAL",
            type=GoalType.AUTHORITY_GRANT,
            statement="may delete partial/",
            status=status_a,
        )
    )
    g.add_node(
        _node(
            id="PROHIBITION_DELETE_PARTIAL",
            type=GoalType.PROHIBITION,
            statement="never delete partial/",
            status=status_b,
        )
    )
    return g


def test_detect_conflicts_flags_opposed_types_same_subject():
    g = _opposed_pair()
    conflicts = g.detect_conflicts()
    assert len(conflicts) == 1
    assert conflicts[0].kind is ConflictKind.UNRESOLVED_CONFLICT


def test_detect_conflicts_ignores_non_opposed_types_even_same_subject():
    g = GoalGraph()
    g.add_node(
        _node(id="HARD_CONSTRAINT_DELETE_PARTIAL", type=GoalType.HARD_CONSTRAINT, statement="must not delete")
    )
    g.add_node(
        _node(id="SOFT_PREFERENCE_DELETE_PARTIAL", type=GoalType.SOFT_PREFERENCE, statement="prefer keeping")
    )
    assert g.detect_conflicts() == []


def test_detect_conflicts_classifies_superseded_when_edge_exists():
    g = _opposed_pair()
    g.add_edge("PROHIBITION_DELETE_PARTIAL", "AUTHORITY_GRANT_DELETE_PARTIAL", EdgeType.SUPERSEDES)
    conflicts = g.detect_conflicts()
    assert conflicts[0].kind is ConflictKind.SUPERSEDED


def test_detect_conflicts_classifies_scoped_exception_when_alternative():
    g = _opposed_pair()
    g.add_edge("PROHIBITION_DELETE_PARTIAL", "AUTHORITY_GRANT_DELETE_PARTIAL", EdgeType.ALTERNATIVE_TO)
    conflicts = g.detect_conflicts()
    assert conflicts[0].kind is ConflictKind.SCOPED_EXCEPTION


def test_detect_conflicts_ignores_non_active_nodes():
    g = _opposed_pair(status_a=Status.PARKED)
    assert g.detect_conflicts() == []


# -- parallelism extraction / ready_frontier -----------------------------------

def test_ready_frontier_excludes_non_frontier_types():
    g = GoalGraph()
    g.add_node(
        _node(id="SUCCESS_CRITERION_TESTS_PASS", type=GoalType.SUCCESS_CRITERION, statement="tests pass")
    )
    assert g.ready_frontier() == []


def test_ready_frontier_respects_dependencies_field():
    g = GoalGraph()
    g.add_node(_node(id="OBJECTIVE_A", statement="a", dependencies=("OBJECTIVE_B",)))
    g.add_node(_node(id="OBJECTIVE_B", statement="a prerequisite"))
    assert g.ready_frontier() == ["OBJECTIVE_B"]
    g.update_node(transition(g.nodes["OBJECTIVE_B"], Status.COMPLETE))
    assert g.ready_frontier() == ["OBJECTIVE_A"]


def test_ready_frontier_respects_requires_edge():
    g = GoalGraph()
    g.add_node(_node(id="OBJECTIVE_A", statement="a"))
    g.add_node(_node(id="OBJECTIVE_B", statement="b"))
    g.add_edge("OBJECTIVE_A", "OBJECTIVE_B", EdgeType.REQUIRES)  # A requires B
    assert g.ready_frontier() == ["OBJECTIVE_B"]
    g.update_node(transition(g.nodes["OBJECTIVE_B"], Status.COMPLETE))
    assert g.ready_frontier() == ["OBJECTIVE_A"]


def test_ready_frontier_respects_blocks_edge():
    """THE MUTATION-CHECK TARGET: BLOCKS predecessor wiring in
    _precedence_predecessors. See the mutation-check note in the report."""
    g = GoalGraph()
    g.add_node(_node(id="OBJECTIVE_A", statement="a"))
    g.add_node(_node(id="OBJECTIVE_B", statement="b"))
    g.add_edge("OBJECTIVE_A", "OBJECTIVE_B", EdgeType.BLOCKS)  # A blocks B
    assert g.ready_frontier() == ["OBJECTIVE_A"]
    g.update_node(transition(g.nodes["OBJECTIVE_A"], Status.COMPLETE))
    assert g.ready_frontier() == ["OBJECTIVE_B"]


def test_ready_frontier_alternative_to_does_not_block_parallelism():
    g = GoalGraph()
    g.add_node(_node(id="OBJECTIVE_A", statement="approach one"))
    g.add_node(_node(id="OBJECTIVE_B", statement="approach two"))
    g.add_edge("OBJECTIVE_A", "OBJECTIVE_B", EdgeType.ALTERNATIVE_TO)
    assert g.ready_frontier() == ["OBJECTIVE_A", "OBJECTIVE_B"]


# -- frontier classification ----------------------------------------------------

def test_classify_frontier_maps_type_and_status():
    assert classify_frontier(_node(type=GoalType.SUCCESS_CRITERION)) is Frontier.NOT_A_FRONTIER
    assert classify_frontier(_node(status=Status.ACTIVE)) is Frontier.ACTIVE_FRONTIER
    assert classify_frontier(_node(status=Status.PARKED)) is Frontier.PARKED


def test_frontier_report_buckets_every_node():
    g = GoalGraph()
    g.add_node(_node(id="OBJECTIVE_A", statement="a"))
    g.add_node(_node(id="OBJECTIVE_B", statement="b", status=Status.PARKED))
    report = g.frontier_report()
    assert report["ACTIVE_FRONTIER"] == ["OBJECTIVE_A"]
    assert report["PARKED"] == ["OBJECTIVE_B"]


# -- bounding the active set ------------------------------------------------

def test_bound_active_frontier_demotes_least_urgent_excess():
    g = GoalGraph()
    g.add_node(_node(id="OBJECTIVE_A", statement="a", priority=0))
    g.add_node(_node(id="OBJECTIVE_B", statement="b", priority=1))
    g.add_node(_node(id="OBJECTIVE_C", statement="c", priority=3))
    demoted = g.bound_active_frontier(2)
    assert [n.id for n in demoted] == ["OBJECTIVE_C"]
    assert g.nodes["OBJECTIVE_C"].status is Status.SLEEPING
    assert g.nodes["OBJECTIVE_A"].status is Status.ACTIVE
    assert g.nodes["OBJECTIVE_B"].status is Status.ACTIVE


def test_bound_active_frontier_noop_when_within_bound():
    g = GoalGraph()
    g.add_node(_node())
    assert g.bound_active_frontier(5) == []
    assert g.nodes["OBJECTIVE_REDUCE_WALL_TIME"].status is Status.ACTIVE
