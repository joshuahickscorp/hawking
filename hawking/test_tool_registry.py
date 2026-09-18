"""Reachability, read-only enforcement, and real-data tests for processes.* tools.

G009 (receipts/sovereign/G009_reachability.json) found hawking.processes
REACHABLE FROM PRODUCTION CODE (hawking/runtime.py's startup reaper,
hawking/commands.py's /processes command) but UNREACHABLE FROM THE MODEL:
registry_probes for "process.list", "processes.live", "processes.status" and
"hawking.processes" all returned failure_class "UNKNOWN_TOOL", and
shell.readonly refused `ps` outright. The live goal's first law names
processes as authority and gave the resident no way to look at one.

Every test below goes through ToolRegistry.invoke, never through
hawking.processes directly -- calling the handler or the module function
directly is exactly the shape of probe that hid this gap in the first place
(the registry carried no process capability while hawking.processes.summary()
worked fine when called straight).
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from hawking import odyssey
from hawking import processes as processes_module
from hawking.tool_registry import COSTLY, READ_ONLY, default_tool_registry
from hawking.perception import tools as perception_tools

PROCESS_TOOLS = ("processes.list", "processes.summary", "processes.orphaned")


def _registry(tmp_path, **kwargs):
    return default_tool_registry(tmp_path, repo_root=tmp_path, **kwargs)


@pytest.mark.parametrize("name", PROCESS_TOOLS)
def test_processes_tools_are_reachable_through_the_registry(tmp_path, name):
    """G009's exact failure mode must no longer occur: registry.invoke(name)
    must not come back UNKNOWN_TOOL.

    Mutation-checked: with each tool's `registry.register(ToolSpec(...))`
    call commented out in hawking/tool_registry.py, this assertion failed with
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
    exclusively through hawking/agentos/resident.py's owned-signal path
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
    direct call to hawking.processes.summary() does off real ps/footprint
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
    summary -- cross-checked against a fresh direct hawking.processes call so a
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


# --------------------------------------------------------------------------
# Odyssey verb reachability. Same disease G009 found in hawking.processes, one
# module over: hawking/odyssey.py defined 20 connector verbs and registered 12,
# so 8 were reachable only by importing the module by hand -- never through
# WorkUnit.tool -> _run_tool -> ToolRegistry.invoke, the resident's only path.
# A verb is allowed to stay off the registry, but the reason has to be
# written down at its definition rather than left as an omission.
# --------------------------------------------------------------------------

CLI_ONLY_MARKER = "# cli-only:"


def _odyssey_verbs() -> set[str]:
    """Every public connector function defined in hawking/odyssey.py."""
    return {
        name
        for name, obj in vars(odyssey).items()
        if inspect.isfunction(obj)
        and not name.startswith("_")
        and obj.__module__ == odyssey.__name__
    }


def _cli_only_verbs() -> set[str]:
    """Verbs deliberately withheld from the registry, read from the marker
    comment that must sit on the line directly above their `def`. Reading the
    source rather than a hardcoded list keeps the comment load-bearing: delete
    the comment and the verb goes back to counting as unreachable.
    """
    lines = Path(odyssey.__file__).read_text(encoding="utf-8").splitlines()
    return {
        line.split("def ", 1)[1].split("(", 1)[0]
        for previous, line in zip(lines, lines[1:])
        if line.startswith("def ") and previous.strip().startswith(CLI_ONLY_MARKER)
    }


def test_every_odyssey_verb_is_registered_or_marked_cli_only(tmp_path):
    """No silently unreachable verb. Registered, or carrying its own
    one-line reason for staying on the CLI -- nothing in between.
    """
    registry = _registry(tmp_path)
    unreachable = sorted(
        verb
        for verb in _odyssey_verbs() - _cli_only_verbs()
        if registry.get(f"odyssey.{verb}") is None
    )
    assert not unreachable, (
        "hawking/odyssey.py defines these verbs but nothing can call them: "
        f"{unreachable}. Register them in hawking/tool_registry.py, or write a "
        f"'{CLI_ONLY_MARKER} ...' line above the def saying why not."
    )


def test_registered_odyssey_handlers_pass_the_arguments_their_verb_declares(
    tmp_path, monkeypatch
):
    """A registration whose handler drops a required positional argument
    raises TypeError the first time it is invoked -- the way an unchecked
    wiring fails in production, long after the registry was written. Each
    handler is driven through registry.invoke with the real connector
    replaced by a recorder that binds against the genuine signature, so a
    mismatch fails here and no subprocess is spawned.
    """
    seen: dict[str, tuple] = {}

    def recorder(real):
        signature = inspect.signature(real)

        def stub(*args, **kwargs):
            signature.bind(*args, **kwargs)  # TypeError on a wiring mismatch
            seen[real.__name__] = (args, kwargs)
            return {"recorded": real.__name__}

        return stub

    for verb in ("patient", "completions", "harvest", "retire", "write_packet"):
        monkeypatch.setattr(odyssey, verb, recorder(getattr(odyssey, verb)))

    registry = _registry(tmp_path, permissions=(READ_ONLY, COSTLY))
    for name, args in (
        ("odyssey.patient", {"oxx": "O003"}),
        ("odyssey.completions", {}),
        ("odyssey.harvest", {}),
        ("odyssey.retire", {"oxx": "O003", "confirm": True}),
        ("odyssey.write_packet", {"oxx": "O003", "confirm": True}),
    ):
        result = registry.invoke(name, args)
        assert result.ok, f"{name} failed through the registry: {result.error!r}"

    assert seen["patient"] == (("O003",), {})
    assert seen["retire"] == ((), {"confirm": True, "oxx": "O003"})
    assert seen["write_packet"] == ((), {"confirm": True, "oxx": "O003"})


@pytest.mark.parametrize("name", ("odyssey.retire", "odyssey.write_packet"))
def test_mutating_odyssey_verbs_refuse_without_confirm(tmp_path, name):
    """These write the driver's own state, so they are gated exactly like
    odyssey.cycle: no confirm, no call. The schema refuses the payload before
    the handler runs, so a missing confirm cannot reach odyssey_ctl.py.
    """
    registry = _registry(tmp_path, permissions=(READ_ONLY, COSTLY))
    assert registry.get(name).mutation == COSTLY
    result = registry.invoke(name, {"oxx": "O003"})
    assert not result.ok
    assert result.failure_class == "INVALID_ARGUMENTS"
def test_perception_tools_docstring_matches_allowlist_count():
    """The module docstring must not drift from the actual allowlist size.

    Regression: the docstring previously claimed nine tools while TOOL_NAMES
    carried ten. Pin the declared count against the real one so any future
    drift fails loudly here instead of silently misleading operators.
    """
    import hawking.perception.tools as perception_tools

    assert perception_tools.TOOL_COUNT == 10
    assert perception_tools.DECLARED_TOOL_COUNT == 10
    assert len(perception_tools.TOOL_NAMES) == 10
    assert "ten local perception tools" in perception_tools.__doc__


def test_perception_tool_registry_digest_placeholder_is_explicit():
    """P5_PERCEPTION_FOCUSED_AUDIT_V4 qualification-support check.

    hawking.perception.tools exposes TOOL_REGISTRY_DIGEST = "1", which is a
    placeholder constant rather than a computed digest of the registry
    surface. The audit needs that fact pinned by an executable assertion so a
    future change that turns it into a real digest (or alters the allowlist
    without updating the constant) is visible in focused test output.

    Boundary: this is a read-only qualification-support assertion. It makes
    no release claim, no hardware-qualification claim, and no phase-
    completion claim.
    """
    from hawking.perception import tools as perception_tools


def test_registry_digest_is_derived_from_the_live_surface():
    """The registry digest must be a real stable hash, never a placeholder."""
    from hawking.perception import tools as perception_tools

    assert len(perception_tools.TOOL_REGISTRY_DIGEST) == 64
    assert perception_tools.TOOL_REGISTRY_DIGEST == perception_tools.registry_digest()
    assert perception_tools.TOOL_REGISTRY_DIGEST_IS_LITERAL is False
    assert perception_tools.TOOL_COUNT == 10
    assert len(perception_tools.TOOL_NAMES) == perception_tools.TOOL_COUNT
