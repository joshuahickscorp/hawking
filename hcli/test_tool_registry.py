"""Reachability, read-only enforcement, and real-data tests for processes.* tools.

G009 (receipts/sovereign/G009_reachability.json) found hcli.processes
REACHABLE FROM PRODUCTION CODE (hcli/runtime.py's startup reaper,
hcli/commands.py's /processes command) but UNREACHABLE FROM THE MODEL:
registry_probes for "process.list", "processes.live", "processes.status" and
"hcli.processes" all returned failure_class "UNKNOWN_TOOL", and
shell.readonly refused `ps` outright. The live goal's first law names
processes as authority and gave the resident no way to look at one.

Every test below goes through ToolRegistry.invoke, never through
hcli.processes directly -- calling the handler or the module function
directly is exactly the shape of probe that hid this gap in the first place
(the registry carried no process capability while hcli.processes.summary()
worked fine when called straight).
"""
from __future__ import annotations

import pytest

from hcli import processes as processes_module
from hcli.tool_registry import READ_ONLY, default_tool_registry

PROCESS_TOOLS = ("processes.list", "processes.summary", "processes.orphaned")


def _registry(tmp_path, **kwargs):
    return default_tool_registry(tmp_path, repo_root=tmp_path, **kwargs)


@pytest.mark.parametrize("name", PROCESS_TOOLS)
def test_processes_tools_are_reachable_through_the_registry(tmp_path, name):
    """G009's exact failure mode must no longer occur: registry.invoke(name)
    must not come back UNKNOWN_TOOL.

    Mutation-checked: with each tool's `registry.register(ToolSpec(...))`
    call commented out in hcli/tool_registry.py, this assertion failed with
    result.ok is False and failure_class == "UNKNOWN_TOOL" (confirmed for
    all three, then restored -- see docs/SCAFFOLD.md for the exact diff).
    """
    registry = _registry(tmp_path)
    spec = registry.get(name)
    assert spec is not None, f"{name} is not registered"
    result = registry.invoke(name, {})
    assert result.ok, f"{name} failed through the registry: {result.error!r}"
    assert result.failure_class is None


@pytest.mark.parametrize("name", PROCESS_TOOLS)
def test_processes_tools_are_read_only(tmp_path, name):
    """mutation == read_only, the schema accepts zero arguments (so no pid or
    signal can ever be smuggled through), and a caller holding only the
    default read_only/research permissions can still invoke it.
    """
    registry = _registry(tmp_path)
    spec = registry.get(name)
    assert spec.mutation == READ_ONLY
    assert spec.input_schema.get("properties") == {}
    assert spec.input_schema.get("additionalProperties") is False

    # A kill-shaped payload is rejected by the schema before the handler runs.
    result = registry.invoke(name, {"pid": 1, "signal": "SIGKILL"})
    assert not result.ok
    assert result.failure_class == "INVALID_ARGUMENTS"

    read_only_registry = _registry(tmp_path, permissions=(READ_ONLY,))
    assert read_only_registry.invoke(name, {}).ok, (
        f"{name} should not require an elevated permission for a plain read"
    )


def test_no_process_kill_capability_is_registered(tmp_path):
    """The fix adds observation only. Killing/reaping must stay reachable
    exclusively through hcli/agentos/resident.py's owned-signal path
    (_owned_signal), which checks a process_start_token -- never through this
    registry, under any name.
    """
    registry = _registry(tmp_path)
    for name in ("processes.kill", "processes.stop", "processes.reap", "processes.terminate", "processes.signal"):
        assert registry.get(name) is None, f"{name} must not be registered"


def test_processes_orphaned_never_calls_reap(tmp_path, monkeypatch):
    """processes.orphaned must reach orphaned_resident_bodies() only. If it
    ever reached reap_orphaned_bodies() (which sends SIGTERM to whatever it
    finds) this fails loudly instead of a test run silently signalling a
    process.
    """
    def _boom(*args, **kwargs):
        raise AssertionError("processes.orphaned must never call reap_orphaned_bodies")

    monkeypatch.setattr(processes_module, "reap_orphaned_bodies", _boom)
    registry = _registry(tmp_path)
    result = registry.invoke("processes.orphaned", {})
    assert result.ok
    assert isinstance(result.value.get("orphaned"), list)


def test_processes_summary_returns_real_process_data(tmp_path):
    """Not a stub: the registry's numbers satisfy the same invariants a fresh
    direct call to hcli.processes.summary() does off real ps/footprint
    output -- total_rss_bytes is the sum of each process's own rss_bytes and
    count equals the number of process rows returned.
    """
    registry = _registry(tmp_path)
    result = registry.invoke("processes.summary", {})
    assert result.ok
    value = result.value
    assert set(value) >= {"count", "total_rss_bytes", "by_class", "roles", "processes"}
    assert value["count"] == len(value["processes"])
    assert value["total_rss_bytes"] == sum(p["rss_bytes"] for p in value["processes"])
    for proc in value["processes"]:
        assert isinstance(proc["pid"], int) and proc["pid"] > 0
        assert isinstance(proc["role"], str) and proc["role"]
        assert isinstance(proc["safe_to_stop"], bool)


def test_processes_list_matches_process_to_dict_shape(tmp_path):
    """processes.list returns real Process.to_dict() rows, not a reshaped
    summary -- cross-checked against a fresh direct hcli.processes call so a
    future field rename in Process.to_dict() cannot drift silently from what
    the tool advertises.
    """
    registry = _registry(tmp_path)
    result = registry.invoke("processes.list", {})
    assert result.ok
    rows = result.value["processes"]
    direct = [p.to_dict() for p in processes_module.live_processes()]
    assert {p["pid"] for p in rows} == {p["pid"] for p in direct}
    if rows:
        assert set(rows[0]) == set(direct[0])
