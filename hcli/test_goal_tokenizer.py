import pytest

from hcli.goal_ir import GoalType, Provenance, SourceIntegrityError, verify_source_refs
from hcli.goal_tokenizer import tokenize
from hcli.paste_cache import PasteCache


def _cache(tmp_path) -> PasteCache:
    return PasteCache(root=tmp_path)


def _by_type(nodes, goal_type):
    return [n for n in nodes if n.type is goal_type]


def _one(nodes, goal_type):
    hits = _by_type(nodes, goal_type)
    assert len(hits) == 1, f"expected exactly one {goal_type.value}, got {hits}"
    return hits[0]


# -- THE LOAD-BEARING SEPARATION: outcome vs method --------------------------

def test_objective_method_split_is_the_load_bearing_separation(tmp_path):
    nodes = tokenize(
        "Make Odyssey faster by caching models on SSD.",
        cache=_cache(tmp_path),
    )
    objective = _one(nodes, GoalType.OBJECTIVE)
    method = _one(nodes, GoalType.SUGGESTED_METHOD)

    # The method is not folded into the objective's statement...
    assert "cach" not in objective.statement.lower()
    assert "faster" in objective.statement.lower()
    # ...and the objective is not folded into the method's.
    assert "cach" in method.statement.lower()
    # The method is linked TO the objective it serves, not the reverse --
    # a method can be superseded without touching what it served.
    assert method.dependencies == (objective.id,)
    assert objective.dependencies == ()


def test_a_rejected_method_does_not_touch_the_objective_it_served(tmp_path):
    """The whole point of the split: transition the method away and the
    objective node is untouched (it lives independently in the graph)."""
    from hcli.goal_ir import Status, transition

    nodes = tokenize(
        "Reduce Odyssey wall time by pre-warming the model cache.",
        cache=_cache(tmp_path),
    )
    objective = _one(nodes, GoalType.OBJECTIVE)
    method = _one(nodes, GoalType.SUGGESTED_METHOD)
    superseded = transition(method, Status.SUPERSEDED, superseded_by="SUGGESTED_METHOD_OTHER")
    assert superseded.status is Status.SUPERSEDED
    assert objective.status is Status.ACTIVE  # untouched


# -- HARD_CONSTRAINT vs SOFT_PREFERENCE vs ANTI_GOAL vs PROHIBITION ----------

def test_hard_constraint_vs_soft_preference_worked_examples(tmp_path):
    nodes = tokenize(
        "Do not delete source specimens.\n"
        "Prefer the local resident when possible.",
        cache=_cache(tmp_path),
    )
    hard = _one(nodes, GoalType.HARD_CONSTRAINT)
    soft = _one(nodes, GoalType.SOFT_PREFERENCE)
    assert "delete" in hard.statement.lower()
    assert "resident" in soft.statement.lower()


def test_anti_goal_is_distinguished_from_a_plain_hard_constraint(tmp_path):
    nodes = tokenize(
        "Do not let Claude remain hot-loop orchestrator.\n"
        "Do not optimize utilization instead of token_ns.\n"
        "Do not delete source specimens.\n",
        cache=_cache(tmp_path),
    )
    anti_goals = _by_type(nodes, GoalType.ANTI_GOAL)
    hard = _by_type(nodes, GoalType.HARD_CONSTRAINT)
    assert len(anti_goals) == 2
    assert len(hard) == 1
    assert "delete" in hard[0].statement.lower()


def test_prohibition_uses_its_own_vocabulary_not_negation(tmp_path):
    nodes = tokenize(
        "Editing the ledger schema directly is forbidden.",
        cache=_cache(tmp_path),
    )
    _one(nodes, GoalType.PROHIBITION)


# -- SUCCESS vs FAILURE criteria kept separate, and falsifiable --------------

def test_success_and_failure_criteria_are_separate_types(tmp_path):
    nodes = tokenize(
        "Success: the full suite passes with `pytest hcli/`.\n"
        "Failure: any previously-passing test regresses.\n",
        cache=_cache(tmp_path),
    )
    success = _one(nodes, GoalType.SUCCESS_CRITERION)
    failure = _one(nodes, GoalType.FAILURE_CRITERION)
    assert success.id != failure.id


def test_acceptance_section_bullets_become_success_criteria_without_markers(tmp_path):
    """A bullet needs no marker word of its own once its heading says
    'Acceptance Criteria' -- section context does the classifying."""
    nodes = tokenize(
        "## Acceptance Criteria\n"
        "- the resident boots in under 10 seconds\n"
        "- no regression in decode throughput\n",
        cache=_cache(tmp_path),
    )
    criteria = _by_type(nodes, GoalType.SUCCESS_CRITERION)
    assert len(criteria) == 2


def test_evidence_requirement_from_a_backtick_command_plus_verify_word(tmp_path):
    nodes = tokenize(
        "Verify with `pytest hcli/test_goal_tokenizer.py` before landing.",
        cache=_cache(tmp_path),
    )
    _one(nodes, GoalType.EVIDENCE_REQUIREMENT)


# -- anti-decomposition: narrative prose emits nothing -----------------------

def test_does_not_emit_one_atom_per_sentence(tmp_path):
    narrative = (
        "This module is data and normalization only. "
        "It has no contact with the scheduler. "
        "The graph lane compares many nodes for dedupe. "
        "That is a different concern entirely."
    )
    nodes = tokenize(narrative, cache=_cache(tmp_path))
    assert nodes == []


def test_mixed_text_only_atoms_matching_markers_are_emitted(tmp_path):
    text = (
        "This section explains background context with no obligations in it "
        "at all, just prose describing history.\n"
        "Do not delete source specimens.\n"
        "More unrelated narrative filler goes here for context.\n"
    )
    nodes = tokenize(text, cache=_cache(tmp_path))
    assert len(nodes) == 1
    assert nodes[0].type is GoalType.HARD_CONSTRAINT


# -- provenance: DERIVED with a real, verifiable source_ref; never EXPLICIT_USER

def test_every_node_is_derived_with_a_verifiable_source_ref(tmp_path):
    cache = _cache(tmp_path)
    nodes = tokenize(
        "Make Odyssey faster by caching models on SSD.\n"
        "Do not delete source specimens.\n",
        cache=cache,
    )
    assert nodes
    for node in nodes:
        assert node.provenance is Provenance.DERIVED
        assert node.source_refs
        verify_source_refs(node, paste_cache=cache)  # must not raise


def test_never_emits_explicit_user(tmp_path):
    text = (
        "# Reduce Odyssey wall time\n\n"
        "Make Odyssey faster by caching models on SSD.\n"
        "Do not delete source specimens.\n"
        "Prefer the local resident when possible.\n"
        "Success: the suite passes.\n"
    )
    nodes = tokenize(text, cache=_cache(tmp_path))
    assert nodes
    assert all(n.provenance is not Provenance.EXPLICIT_USER for n in nodes)


def test_tampering_a_paste_is_still_caught_through_the_tokenizer(tmp_path):
    cache = _cache(tmp_path)
    nodes = tokenize("Do not delete source specimens.", cache=cache)
    node = nodes[0]
    paste_id = node.source_refs[0].ref
    (tmp_path / ".hcli" / "pastes" / f"{paste_id}.txt").write_text("tampered bytes")
    with pytest.raises(SourceIntegrityError):
        verify_source_refs(node, paste_cache=cache)


# -- ULTRAGOAL heading becomes root and parents its children -----------------

def test_leading_h1_becomes_ultragoal_and_parents_children(tmp_path):
    text = (
        "# Reduce Odyssey wall time\n\n"
        "Make Odyssey faster by caching models on SSD.\n"
        "Do not delete source specimens.\n"
    )
    nodes = tokenize(text, cache=_cache(tmp_path))
    ultragoal = _one(nodes, GoalType.ULTRAGOAL)
    for node in nodes:
        if node.type is GoalType.ULTRAGOAL:
            continue
        assert node.parent_ultragoal == ultragoal.id


def test_steer_call_with_existing_parent_ultragoal_does_not_mint_a_new_one(tmp_path):
    text = "# Some heading that looks like a root\n\nDo not delete source specimens.\n"
    nodes = tokenize(text, cache=_cache(tmp_path), parent_ultragoal="ULTRAGOAL_EXISTING")
    assert _by_type(nodes, GoalType.ULTRAGOAL) == []
    hard = _one(nodes, GoalType.HARD_CONSTRAINT)
    assert hard.parent_ultragoal == "ULTRAGOAL_EXISTING"


# -- stable, deterministic ids -------------------------------------------------

def test_ids_are_deterministic_across_separate_tokenize_calls(tmp_path_factory):
    text = (
        "# Reduce Odyssey wall time\n\n"
        "Make Odyssey faster by caching models on SSD.\n"
        "Do not delete source specimens.\n"
        "Prefer the local resident when possible.\n"
    )
    nodes_a = tokenize(text, cache=PasteCache(root=tmp_path_factory.mktemp("a")))
    nodes_b = tokenize(text, cache=PasteCache(root=tmp_path_factory.mktemp("b")))
    assert [n.id for n in nodes_a] == [n.id for n in nodes_b]
    assert [n.type for n in nodes_a] == [n.type for n in nodes_b]


def test_colliding_slugs_get_distinct_ids(tmp_path):
    nodes = tokenize(
        "Prefer the fast path when possible.\n"
        "Prefer the fast lane when possible.\n",
        cache=_cache(tmp_path),
    )
    prefs = _by_type(nodes, GoalType.SOFT_PREFERENCE)
    assert len(prefs) == 2
    assert prefs[0].id != prefs[1].id


# -- referenced files carried through as grounding ---------------------------

def test_referenced_file_paths_are_attached_to_the_node(tmp_path):
    nodes = tokenize(
        "Do not touch hcli/goal.py during this change.",
        cache=_cache(tmp_path),
    )
    hard = _one(nodes, GoalType.HARD_CONSTRAINT)
    assert "hcli/goal.py" in hard.resources


# -- misc types smoke coverage ------------------------------------------------

def test_temporal_and_resource_and_priority_and_dependency_markers(tmp_path):
    nodes = tokenize(
        "This needs 40GB of free disk before it can run.\n"
        "Ship the fix by 2026-09-15.\n"
        "This is the top priority for this cycle.\n"
        "Phase 2 depends on Phase 1 landing first.\n",
        cache=_cache(tmp_path),
    )
    assert _by_type(nodes, GoalType.RESOURCE_REQUIREMENT)
    assert _by_type(nodes, GoalType.TEMPORAL_CONSTRAINT)
    assert _by_type(nodes, GoalType.PRIORITY)
    assert _by_type(nodes, GoalType.DEPENDENCY)


def test_open_question_and_empty_input(tmp_path):
    nodes = tokenize("Is the resident actually the bottleneck here?", cache=_cache(tmp_path))
    _one(nodes, GoalType.OPEN_QUESTION)
    assert tokenize("", cache=_cache(tmp_path)) == []
    assert tokenize("   \n\n  ", cache=_cache(tmp_path)) == []
