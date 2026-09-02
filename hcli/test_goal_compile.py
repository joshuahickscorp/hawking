from hcli.goal import WorkUnitDAG
from hcli.goal_compile import (
    CompileResult,
    DiskCheck,
    EvidenceOutcome,
    apply_evidence,
    check_disk_satisfaction,
    goals_report,
    ingest,
    load_reachability_snapshot,
    schedule,
    sync_from_dag,
)
from hcli.goal_graph import ConflictKind, GoalGraph
from hcli.goal_ir import GoalNode, GoalType, Provenance, Status
from hcli.paste_cache import PasteCache


def _cache(tmp_path) -> PasteCache:
    return PasteCache(root=tmp_path)


def _node(**overrides):
    """The cheapest legal node: MODEL_INFERRED needs no source_ref at all.
    Same helper shape as test_goal_graph.py's, so a node built for one
    lane's tests reads the same way in the other."""
    fields = dict(
        id="OBJECTIVE_REDUCE_WALL_TIME",
        type=GoalType.OBJECTIVE,
        statement="reduce odyssey wall time",
        provenance=Provenance.MODEL_INFERRED,
    )
    fields.update(overrides)
    return GoalNode(**fields)


def _by_type(graph: GoalGraph, goal_type: GoalType):
    return [n for n in graph.nodes.values() if n.type is goal_type]


def _one(graph: GoalGraph, goal_type: GoalType) -> GoalNode:
    hits = _by_type(graph, goal_type)
    assert len(hits) == 1, f"expected exactly one {goal_type.value}, got {hits}"
    return hits[0]


# -- ingest: tokenize + fold into a graph ------------------------------------

def test_ingest_dedupes_across_repeated_calls(tmp_path):
    text = "Make Odyssey faster by caching models on SSD."
    graph = ingest(text, cache=_cache(tmp_path))
    n_before = len(graph.nodes)
    ingest(text, graph, cache=_cache(tmp_path))
    assert len(graph.nodes) == n_before, "restating the same directive must not double the graph"


# -- schedule: anti-overdecomposition -----------------------------------------

def test_schedule_maps_one_objective_to_exactly_two_units(tmp_path):
    """One conceptual objective, plus a hard constraint, a success
    criterion and a hypothesis around it, must not become five+ WorkUnits
    -- only the OBJECTIVE is schedulable work; everything else is an
    attribute of the goal, not a unit of its own."""
    text = (
        "# Reduce Odyssey wall time\n\n"
        "Make Odyssey faster by caching models on SSD.\n\n"
        "Do not delete source specimens.\n\n"
        "Success: the full suite passes with `pytest hcli/`.\n\n"
        "Hypothesis: the slowdown is disk contention.\n"
    )
    graph = ingest(text, cache=_cache(tmp_path))
    result = schedule(graph)
    assert isinstance(result, CompileResult)
    assert len(result.dag.units) == 2, result.dag.units
    objective = _one(graph, GoalType.OBJECTIVE)
    assert f"{objective.id}.implement" in result.dag.units
    assert f"{objective.id}.validate" in result.dag.units
    assert result.compiled_ir["invariants"] == ["Do not delete source specimens"]


def test_schedule_does_not_double_schedule_the_ultragoal(tmp_path):
    """ULTRAGOAL is a FRONTIER_TYPES member in goal_graph.py (ready_frontier
    lists it), but it restates the directive its own OBJECTIVE already
    covers -- it must not ALSO become a WorkUnit pair."""
    text = "# Reduce Odyssey wall time\n\nMake Odyssey faster.\n"
    graph = ingest(text, cache=_cache(tmp_path))
    ultra = _one(graph, GoalType.ULTRAGOAL)
    result = schedule(graph)
    assert f"{ultra.id}.implement" not in result.dag.units


def test_schedule_falls_back_to_the_bare_ultragoal_when_nothing_is_under_it(tmp_path):
    """A directive that is only a heading, with no sentence the tokenizer
    can classify underneath it, must still produce something schedulable
    -- the ULTRAGOAL itself, not silence."""
    text = "# Ship it\n"
    graph = ingest(text, cache=_cache(tmp_path))
    assert len(graph.nodes) == 1
    ultra = _one(graph, GoalType.ULTRAGOAL)
    result = schedule(graph)
    assert f"{ultra.id}.implement" in result.dag.units
    assert f"{ultra.id}.validate" in result.dag.units


def test_schedule_finds_a_childless_ultragoal_even_when_another_has_objectives(tmp_path):
    """One directive with a real OBJECTIVE must not shadow a second,
    still-childless ULTRAGOAL from an unrelated directive folded into the
    same graph -- each ultragoal's orphan status is its own question."""
    graph = ingest("# Reduce Odyssey wall time\n\nMake Odyssey faster.\n", cache=_cache(tmp_path))
    graph = ingest("# Ship it\n", graph, cache=_cache(tmp_path))
    childless = next(
        n for n in _by_type(graph, GoalType.ULTRAGOAL) if n.statement == "Ship it"
    )
    result = schedule(graph)
    assert f"{childless.id}.implement" in result.dag.units


# -- check 1: goal satisfaction from disk -------------------------------------

def test_check_disk_satisfaction_is_none_when_no_command_is_derivable():
    node = _node()
    assert check_disk_satisfaction(node) is None


def test_check_disk_satisfaction_passing_check_marks_node_complete(tmp_path):
    passing = tmp_path / "test_already_true.py"
    passing.write_text("def test_ok():\n    assert True\n")
    node = _node(id="OBJECTIVE_ALREADY_DONE", resources=(str(passing),))

    disk = check_disk_satisfaction(node)
    assert isinstance(disk, DiskCheck)
    assert disk.passed

    graph = GoalGraph()
    graph.add_node(node)
    result = schedule(graph)
    assert result.satisfied == (node.id,)
    assert f"{node.id}.implement" not in result.dag.units, (
        "a claim that already passes on disk must not generate implementation work"
    )
    assert graph.nodes[node.id].status is Status.COMPLETE
    assert graph.nodes[node.id].reopen_condition


def test_check_disk_satisfaction_failing_check_still_schedules_work(tmp_path):
    failing = tmp_path / "test_still_broken.py"
    failing.write_text("def test_fails():\n    assert False\n")
    node = _node(id="OBJECTIVE_NOT_DONE_YET", resources=(str(failing),))

    disk = check_disk_satisfaction(node)
    assert isinstance(disk, DiskCheck)
    assert not disk.passed

    graph = GoalGraph()
    graph.add_node(node)
    result = schedule(graph)
    assert result.satisfied == ()
    assert f"{node.id}.implement" in result.dag.units
    validate = result.dag.units[f"{node.id}.validate"]
    assert validate.verifier and "test_still_broken.py" in validate.verifier


# -- check 2: capability reachability (fake snapshot -- the real analyzer -----
# -- is a whole-repo scan and is never invoked by the test suite) -------------

def test_capability_notes_flag_a_reachable_capability(tmp_path):
    node = _node(resources=("hcli/tool_registry.py",))
    snapshot = {
        "hcli.tool_registry.default_tool_registry": {
            "defined": True,
            "callable": True,
            "definition": {"file": "hcli/tool_registry.py"},
        }
    }
    graph = GoalGraph()
    graph.add_node(node)
    result = schedule(graph, reachability_snapshot=snapshot)
    assert node.id in result.notes
    assert "already reachable" in result.notes[node.id][0]
    assert "NOTE:" in result.dag.units[f"{node.id}.implement"].description


def test_capability_notes_flag_a_defined_but_unreachable_capability():
    from hcli.goal_compile import _capability_notes

    node = _node(resources=("tools/future/orphan.py",))
    snapshot = {
        "tools.future.orphan.helper": {
            "defined": True,
            "callable": False,
            "definition": {"file": "tools/future/orphan.py"},
        }
    }
    notes = _capability_notes(node, snapshot)
    assert len(notes) == 1
    assert "unreachable" in notes[0]
    assert "not rebuilding it" not in notes[0]  # advisory, not a directive


def test_capability_notes_empty_when_node_names_no_files():
    from hcli.goal_compile import _capability_notes

    node = _node(resources=())
    snapshot = {"anything": {"defined": True, "callable": True}}
    assert _capability_notes(node, snapshot) == []


def test_load_reachability_snapshot_is_never_called_by_schedule(monkeypatch, tmp_path):
    """schedule() must not itself trigger the whole-repo AST scan -- a
    caller opts in by passing a snapshot; omitting it must cost nothing."""
    def _boom():
        raise AssertionError("schedule() must not call load_reachability_snapshot")

    monkeypatch.setattr("hcli.goal_compile.load_reachability_snapshot", _boom)
    graph = ingest("Make Odyssey faster.", cache=_cache(tmp_path))
    schedule(graph)  # must not raise


# -- sync_from_dag: the other half of the recompile loop ----------------------

def test_sync_from_dag_completes_objective_and_unblocks_subobjective():
    objective = _node(id="OBJECTIVE_ROOT", statement="root objective")
    subobjective = _node(
        id="SUBOBJECTIVE_CHILD",
        type=GoalType.SUBOBJECTIVE,
        statement="child piece",
        dependencies=(objective.id,),
    )
    graph = GoalGraph()
    graph.add_node(objective)
    graph.add_node(subobjective)

    first = schedule(graph)
    assert f"{objective.id}.implement" in first.dag.units
    assert f"{subobjective.id}.implement" not in first.dag.units, (
        "the subobjective depends on the objective and must not be ready yet"
    )

    # Still pending -- nothing has actually finished, so nothing closes.
    assert sync_from_dag(graph, first.dag.units) == ()
    assert graph.nodes[objective.id].status is Status.ACTIVE

    first.dag.units[f"{objective.id}.validate"].status = "completed"
    closed = sync_from_dag(graph, first.dag.units)
    assert closed == (objective.id,)
    assert graph.nodes[objective.id].status is Status.COMPLETE

    second = schedule(graph)
    assert f"{subobjective.id}.implement" in second.dag.units, (
        "completing the objective must widen the ready frontier"
    )


def test_schedule_is_idempotent_for_unchanged_content(tmp_path):
    """The whole point of feeding an EXISTING Scheduler: replanning
    unchanged work must be a no-op, never a duplicate or a conflict."""
    from hcli.scheduler import Scheduler

    graph = ingest("Make Odyssey faster.", cache=_cache(tmp_path))
    first = schedule(graph)
    sched = Scheduler(dict(first.dag.units), 1, workspace=tmp_path)
    second = schedule(graph)
    outcomes = sched.replan(second.dag.units)
    assert outcomes, "expected at least one unit"
    assert all(o.kind == "idempotent" for o in outcomes)


# -- apply_evidence: recompiling ONE node from real evidence ------------------

def test_apply_evidence_falsified_demotes_without_touching_the_ultragoal():
    ultra = _node(id="ULTRAGOAL_ROOT", type=GoalType.ULTRAGOAL, statement="root")
    hyp = _node(
        id="HYPOTHESIS_DISK_BOUND",
        type=GoalType.HYPOTHESIS,
        statement="the slowdown is disk contention",
        parent_ultragoal=ultra.id,
    )
    graph = GoalGraph()
    graph.add_node(ultra)
    graph.add_node(hyp)

    updated = apply_evidence(graph, hyp.id, EvidenceOutcome.FALSIFIED, note="measured otherwise")
    assert updated.status is Status.PARKED
    assert updated.priority > hyp.priority
    assert updated.confidence <= 0.2
    assert graph.nodes[ultra.id].status is Status.ACTIVE  # untouched


def test_apply_evidence_falsified_on_a_non_active_node_only_demotes():
    parked = _node(id="HYPOTHESIS_ALREADY_PARKED", type=GoalType.HYPOTHESIS, status=Status.PARKED)
    graph = GoalGraph()
    graph.add_node(parked)
    updated = apply_evidence(graph, parked.id, "FALSIFIED")
    assert updated.status is Status.PARKED  # cannot transition PARKED -> PARKED; just demoted
    assert updated.priority > parked.priority


def test_apply_evidence_satisfied_completes_the_node():
    node = _node()
    graph = GoalGraph()
    graph.add_node(node)
    updated = apply_evidence(graph, node.id, EvidenceOutcome.SATISFIED, note="verified by hand")
    assert updated.status is Status.COMPLETE
    assert updated.reopen_condition == "verified by hand"


def test_apply_evidence_confirmed_restores_full_confidence():
    node = _node(confidence=0.4)
    graph = GoalGraph()
    graph.add_node(node)
    updated = apply_evidence(graph, node.id, EvidenceOutcome.CONFIRMED)
    assert updated.confidence == 1.0
    assert updated.status is Status.ACTIVE


# -- goals_report: read-only inspection surface -------------------------------

def test_goals_report_surfaces_frontier_and_conflicts():
    grant = _node(
        id="AUTHORITY_GRANT_SAME_SUBJECT",
        type=GoalType.AUTHORITY_GRANT,
        statement="claude may edit prod configs",
    )
    ban = _node(
        id="PROHIBITION_SAME_SUBJECT",
        type=GoalType.PROHIBITION,
        statement="claude may not edit prod configs",
    )
    graph = GoalGraph()
    graph.add_node(grant)
    graph.add_node(ban)

    report = goals_report(graph)
    assert "AUTHORITY_GRANT_SAME_SUBJECT" in report["nodes"]
    assert report["conflicts"], "structurally opposed same-subject nodes must surface"
    assert report["conflicts"][0]["kind"] == ConflictKind.UNRESOLVED_CONFLICT.value
    assert isinstance(report["frontier"], dict)
    assert isinstance(report["ready"], list)


# -- compiled_ir shape: the actual integration contract with Mission ---------

def test_compiled_ir_has_the_same_keys_goalcompiler_produces(tmp_path):
    from hcli.goal import GoalCompiler

    text = "Make Odyssey faster by caching models on SSD."
    graph = ingest(text, cache=_cache(tmp_path))
    result = schedule(graph)
    expected_keys = set(GoalCompiler().compile(text).keys())
    assert expected_keys.issubset(result.compiled_ir.keys())
    assert isinstance(result.compiled_ir["workunits"], WorkUnitDAG)


def test_load_reachability_snapshot_reads_the_real_analyzer_shape(monkeypatch):
    """Do not run the real 45s whole-repo scan in the test suite -- assert
    the wrapper reaches the right function and passes its result through
    unmodified, using a stub in the analyzer's place."""
    sentinel = {"some.capability": {"defined": True}}
    monkeypatch.setattr(
        "tools.future.capability_reachability.assemble",
        lambda: {"capabilities": sentinel},
    )
    assert load_reachability_snapshot() == sentinel
