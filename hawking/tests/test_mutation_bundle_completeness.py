"""Focused tests for the machine-owned Goal mutation-bundle boundary."""

from __future__ import annotations

from pathlib import Path

from hawking.chat_tools import (
    MUTATION_BUNDLE_SCHEMA,
    run_builder_tool,
    validate_mutation_bundle,
)
from hawking.workunit_owner import (
    WorkunitRecord,
    _kept_mutate_snippet_from_trace_entries,
    acceptance_met,
)


SOURCE = {
    "op": "replace",
    "path": "hawking/web.py",
    "old_lines": ["return old"],
    "new_lines": ["return new"],
}
TEST = {
    "op": "append",
    "path": "hawking/test_web_surface.py",
    "new_lines": ["def test_new_regression():", "    assert True"],
}
BUNDLE = {
    "schema": MUTATION_BUNDLE_SCHEMA,
    "require_source_mutation": True,
    "require_regression_test_mutation": True,
    "require_focused_test_invocation": True,
    "allowed_files": ["hawking/web.py", "hawking/test_web_surface.py"],
    "focus_tests": ["hawking/test_web_surface.py"],
    "test_mutation_paths": ["hawking/test_web_surface.py"],
}


class FakeEngine:
    def __init__(self, root: Path):
        self.root = root
        self.calls = []

    def apply_typed_mutation(self, operations, tests):
        self.calls.append((list(operations), list(tests or [])))
        return {
            "status": "accepted",
            "applied": True,
            "rolled_back": False,
            "paths": [str(item.get("path")) for item in operations],
        }


def test_source_and_test_mutations_are_admitted():
    result = validate_mutation_bundle([SOURCE, TEST], ["hawking/test_web_surface.py"],
                                      bundle=BUNDLE)
    assert result["bundle_complete"] is True
    assert result["missing_requirements"] == []
    assert result["source_operations"] == [SOURCE]
    assert result["test_operations"] == [TEST]


def test_source_only_is_rejected_before_apply():
    result = validate_mutation_bundle([SOURCE], ["hawking/test_web_surface.py"],
                                      bundle=BUNDLE)
    assert result["bundle_complete"] is False
    assert "MISSING_REQUIRED_TEST_OPERATION" in result["missing_requirements"]


def test_rejected_bundle_leaves_source_state_untouched():
    engine = FakeEngine(Path.cwd())
    state = {}
    result = run_builder_tool(
        "repo.edit",
        {"operations": [SOURCE], "tests": ["hawking/test_web_surface.py"]},
        engine=engine,
        mutation_bundle=BUNDLE,
        bundle_state=state,
    )
    assert result.ok is False
    assert result.failure_class == "MUTATION_BUNDLE"
    assert engine.calls == []
    assert state["source_operations"] == [SOURCE]


def test_valid_operations_survive_deficiency_cycle():
    state = {}
    first = validate_mutation_bundle([SOURCE], ["hawking/test_web_surface.py"],
                                    bundle=BUNDLE, pending=state)
    state.update({"operations": first["source_operations"],
                  "source_operations": first["source_operations"],
                  "tests": first["tests"]})
    assert state["source_operations"] == [SOURCE]


def test_missing_operation_completes_original_bundle():
    result = validate_mutation_bundle(
        [TEST], [], bundle=BUNDLE,
        pending={"source_operations": [SOURCE],
                 "tests": ["hawking/test_web_surface.py"]},
    )
    assert result["bundle_complete"] is True
    assert result["operations"] == [SOURCE, TEST]


def test_disallowed_repair_cannot_exploit_preserved_source():
    unrelated = {**TEST, "path": "hawking/other_test.py"}
    result = validate_mutation_bundle(
        [unrelated], [], bundle=BUNDLE,
        pending={"source_operations": [SOURCE],
                 "tests": ["hawking/test_web_surface.py"]},
    )
    assert result["bundle_complete"] is False
    assert "DISALLOWED_MUTATION_OPERATION" in result["missing_requirements"]


def test_test_file_same_replace_shape_is_normalized_to_append(tmp_path):
    engine = FakeEngine(tmp_path)
    block = [
        "def test_new_regression():",
        "    assert True",
    ]
    result = run_builder_tool(
        "repo.edit",
        {
            "operations": [
                {"op": "replace", "path": "source.py",
                 "old_lines": ["flag = False"],
                 "new_lines": ["flag = True"]},
                {"op": "replace", "path": "test_source.py",
                 "old_lines": block, "new_lines": block},
            ],
            "tests": ["test_source.py"],
        },
        engine=engine,
        mutation_bundle={
            **BUNDLE,
            "allowed_files": ["source.py", "test_source.py"],
            "focus_files": ["source.py", "test_source.py"],
            "test_mutation_paths": ["test_source.py"],
        },
    )
    assert result.ok is True
    assert result.value["bundle_complete"] is True
    assert result.value["syntax_repairs"] == ["test_replace_to_append"]
    assert engine.calls[0][0][1]["op"] == "append"
    assert engine.calls[0][0][1]["new_lines"] == block


def test_engine_focused_validation_folds_into_goal_seal_evidence():
    folded = _kept_mutate_snippet_from_trace_entries([{
        "tool": "repo.edit",
        "ok": True,
        "applied": True,
        "rolled_back": False,
        "verdict": "accepted",
        "paths": ["hawking/web.py", "hawking/test_web_surface.py"],
        "engine_test_passed": True,
        "engine_test_paths": ["hawking/test_web_surface.py"],
        "engine_test_validation": {"ok": True},
    }])
    assert '"returncode": 0' in folded
    assert 'hawking/test_web_surface.py' in folded


def test_one_complete_bundle_is_sufficient_for_discrete_goal_completion():
    record = WorkunitRecord(
        workunit_id="GOAL-TEST",
        goal_id="GOAL-TEST",
        goal_mode="discrete_goal",
        atomic_subtasks=[{"seal": "GOAL-TEST_SEAL_B1"}],
        evidence=[
            {
                "kind": "worker_completion_contract",
                "complete": True,
                "successful_tools": ["repo.edit", "tests.run", "git.status", "git.diff"],
                "verified_tools": ["tests.run"],
            },
            {"kind": "goal_test_evidence"},
        ],
    )
    assert acceptance_met(record, {}) is True


def test_test_and_diff_without_accepted_repo_edit_cannot_complete_discrete_goal():
    record = WorkunitRecord(
        workunit_id="GOAL-TEST-INCOMPLETE",
        goal_id="GOAL-TEST-INCOMPLETE",
        goal_mode="discrete_goal",
        atomic_subtasks=[{"seal": "GOAL-TEST-INCOMPLETE_SEAL_B1"}],
        evidence=[
            {
                "kind": "worker_completion_contract",
                "complete": False,
                "successful_tools": ["tests.run", "git.status", "git.diff"],
                "verified_tools": ["tests.run"],
                "unmet": ["repo.edit"],
            },
            {"kind": "goal_test_evidence"},
        ],
    )
    assert acceptance_met(record, {}) is False
