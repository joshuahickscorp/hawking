"""A WorkUnit must be able to reach the typed tool surface.

Before this, HCLI had 41 registered tools and NO call site: `ToolRegistry`
appeared zero times in mission.py, executors.py, engine.py and resident.py, and
`resident.py` did not contain the string "tool" at all. The resident could
enumerate `filesystem.write`, `tests.run`, `git.diff` and `shell.exec` and
invoke none of them, which makes self-hosting structurally impossible rather
than merely unimplemented.

These tests drive the real `WorkUnitExecutor` against the real registry. They
fail if the routing is removed, which is the point — a catalogue that nothing
calls passes any test written against the catalogue.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcli.executors import BACKEND_TOOL, WorkUnitExecutor, select_backend_name
from hcli.workunit import WorkUnit

REPO = Path(__file__).resolve().parents[1]


def _unit(**kwargs) -> WorkUnit:
    base = dict(id="wu-tool-1", role="verify", description="call one typed tool")
    base.update(kwargs)
    return WorkUnit(**base)


def _executor(tmp_path) -> WorkUnitExecutor:
    return WorkUnitExecutor(workspace=str(REPO), repo_root=str(REPO))


def test_naming_a_tool_routes_to_the_tool_backend():
    wu = _unit(tool="git.status")
    assert select_backend_name(wu) == BACKEND_TOOL


def test_a_named_tool_outranks_a_stale_provider():
    """A requeued unit can carry an old provider; the tool call is the truth."""
    wu = _unit(tool="git.status", provider="qwen")
    assert select_backend_name(wu) == BACKEND_TOOL


def test_a_workunit_actually_invokes_a_real_tool(tmp_path):
    """End to end through the real registry, not a stub."""
    executor = _executor(tmp_path)
    result = executor.execute(_unit(tool="git.status"))
    assert result["backend"] == BACKEND_TOOL
    assert result["validation"]["ok"] is True, result["validation"]
    assert result["validation"]["tool"] == "git.status"
    assert result["tool_result"]["schema"] == "hcli.agentos.tool.result.v1"
    assert result["validation"]["invocation_id"].startswith("tool-")


def test_the_resident_can_read_its_own_source_through_a_tool(tmp_path):
    """The first link of the self-build loop: search and read its own code."""
    executor = _executor(tmp_path)
    result = executor.execute(
        _unit(tool="filesystem.read", tool_arguments={"path": "hcli/executors.py"})
    )
    assert result["validation"]["ok"] is True, result["validation"]
    assert "WorkUnitExecutor" in json.dumps(result["tool_result"]["value"])


def test_an_unknown_tool_fails_the_unit_rather_than_passing_it(tmp_path):
    executor = _executor(tmp_path)
    result = executor.execute(_unit(tool="no.such.tool"))
    assert result["validation"]["ok"] is False
    assert result["validation"]["failure_class"] == "UNKNOWN_TOOL"


def test_bad_arguments_fail_the_unit(tmp_path):
    """The registry's schema validation is the gate; do not soften its verdict."""
    executor = _executor(tmp_path)
    result = executor.execute(
        _unit(tool="filesystem.read", tool_arguments={"path": 12345})
    )
    assert result["validation"]["ok"] is False
    assert result["validation"]["failure_class"] == "INVALID_ARGUMENTS"


def test_non_mapping_arguments_fail_closed(tmp_path):
    executor = _executor(tmp_path)
    wu = _unit(tool="git.status", tool_arguments=["not", "a", "mapping"])
    result = executor.execute(wu)
    assert result["validation"]["ok"] is False
    assert result["validation"]["reason"] == "TOOL_ARGUMENTS_NOT_A_MAPPING"


def test_a_huge_tool_return_is_bounded_in_the_record(tmp_path):
    """A mission record is evidence, not a log file."""
    from hcli.executors import _redact_tool_value

    assert _redact_tool_value("x" * 10, limit=4000) == "x" * 10
    bounded = _redact_tool_value("y" * 9000, limit=4000)
    assert len(bounded) < 9000
    assert bounded.startswith("y" * 4000)
    assert "+5000 chars" in bounded


def test_tool_backend_without_a_named_tool_keeps_the_old_command_path(tmp_path):
    """Back-compat: `tool` was already an alias for the verifier-command path."""
    executor = _executor(tmp_path)
    wu = _unit(provider="tool", verifier="python3 -c \"import sys; sys.exit(0)\"")
    result = executor.execute(wu)
    # It must NOT take the typed-tool branch, and it must not crash.
    assert "tool_result" not in result




def test_mission_hands_down_agentos_registry_rather_than_minting_one():
    """One registry, not two.

    AgentOS already owns a registry carrying the mission's permission set and
    the path tool receipts are persisted to. If Mission let the executor build
    its own, the two could disagree about what is permitted and only one of
    them would write receipts.
    """
    from hcli.mission import Mission

    sentinel = object()
    mission = Mission("/tmp", tool_registry=sentinel, repo_root=str(REPO))
    assert mission.tool_registry is sentinel

    executor = WorkUnitExecutor("/tmp", tool_registry=sentinel)
    assert executor.tool_registry() is sentinel, "a supplied registry must be used as-is"


def test_executor_still_builds_one_when_none_is_supplied():
    """Standalone use (tests, one-off units) must not require a registry."""
    executor = WorkUnitExecutor(str(REPO), repo_root=str(REPO))
    registry = executor.tool_registry()
    assert registry is not None
    assert executor.tool_registry() is registry, "built once, then cached"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        for name, fn in sorted(globals().items()):
            if not name.startswith("test_"):
                continue
            fn(Path(tmp)) if fn.__code__.co_argcount else fn()
            print(f"ok  {name}")
    print("all green")
