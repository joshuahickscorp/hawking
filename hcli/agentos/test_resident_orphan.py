"""Truth-bound tests for the resident supervisor's orphan exit rule.

An orphaned supervisor polling forever is a low-information busy loop, but
``start_resident`` deliberately detaches its supervisor, so pid 1 alone cannot
be treated as abandonment.  These tests pin both halves: the exit only fires
for a supervisor that a real launcher owned at start, and it stays off unless
the durable config opts in.

The decision is a pure function on purpose, so the rule is checkable without
spawning a real supervisor.
"""
from __future__ import annotations

import os
from pathlib import Path

from hcli.agentos.resident import (
    DETACHED_ENV,
    ResidentConfig,
    ResidentDaemon,
    ResidentSupervisor,
    orphan_exit_reason,
)


def test_owned_supervisor_reparented_to_init_stops_and_says_why():
    reason = orphan_exit_reason(1, launch_ppid=4242, exit_when_orphaned=True)
    assert reason is not None
    assert "4242" in reason


def test_supervisor_whose_launcher_is_alive_keeps_running():
    assert orphan_exit_reason(4242, launch_ppid=4242, exit_when_orphaned=True) is None


def test_supervisor_detached_at_start_is_not_an_orphan():
    # start_resident detaches, so a supervisor that already saw pid 1 at start
    # learns nothing new from pid 1 later and must not be killed as an orphan.
    assert orphan_exit_reason(1, launch_ppid=1, exit_when_orphaned=True) is None
    assert orphan_exit_reason(1, launch_ppid=None, exit_when_orphaned=True) is None


def test_the_escape_hatch_still_disables_the_exit():
    # Inverted from "defaults off": leaving it off meant the mechanism existed
    # with zero callers opting in, so the live defect stayed live. Off is now
    # the escape hatch for a supervisor meant to outlive a non-daemon launcher.
    assert orphan_exit_reason(1, launch_ppid=4242, exit_when_orphaned=False) is None


def test_opt_in_survives_a_durable_config_round_trip():
    config = ResidentConfig(workspace=".", goal="g", exit_when_orphaned=True)
    assert ResidentConfig.from_mapping(config.to_dict()).exit_when_orphaned is True




def _configured_supervisor(tmp: Path, **overrides) -> ResidentSupervisor:
    """A real supervisor over a real state file, with no worker and no model."""
    daemon = ResidentDaemon(tmp)
    config = ResidentConfig(
        workspace=str(tmp),
        goal="orphan integration check",
        interval_s=0.1,
        # These tests assert on ORPHAN handling, never on how long a worker is
        # given to evacuate. The 10.0s default was spent 40 x 0.25s in the
        # grace loop and was 10.11s of a 77.82s suite on its own.
        evacuation_grace_s=overrides.pop("evacuation_grace_s", 0.2),
        **overrides,
    )
    daemon.configure(config)
    return ResidentSupervisor(daemon.store.state_path)


def test_run_actually_consults_the_orphan_rule_and_stops(tmp_path, monkeypatch):
    """The integration, not the helper.

    Five unit tests over ``orphan_exit_reason`` all still pass when the call in
    ``ResidentSupervisor.run()`` is replaced with ``orphaned = None`` -- which
    is the whole defect restored. This drives the real loop instead: owned at
    startup, reparented to pid 1 afterwards, and it must stop on its own.
    """
    supervisor = _configured_supervisor(tmp_path)
    # Owned when run() records launch_ppid, orphaned on every poll after.
    ppids = iter([4242])
    monkeypatch.setattr(os, "getppid", lambda: next(ppids, 1))

    # Bound the loop. Without the orphan check this supervisor polls forever --
    # that IS the defect -- so an unbounded test would HANG on a regression
    # instead of failing. Give up after a handful of polls and fail loudly.
    polls = {"n": 0}

    def held(_config):
        polls["n"] += 1
        if polls["n"] > 5:
            supervisor.store.update(stop_requested=True)
        return {"safe": False, "reasons": ["held"]}

    monkeypatch.setattr(supervisor, "_memory", held)

    assert supervisor.run() == 0, "an orphaned supervisor must exit cleanly"
    assert polls["n"] <= 5, "supervisor kept polling instead of noticing it was orphaned"
    state = supervisor.store.read()
    # last_event is overwritten by the shutdown `finally`; stop_reason is the
    # field that survives it, which is the point of recording the reason there.
    assert "4242" in (state.get("stop_reason") or ""), state.get("stop_reason")
    assert state["state"] == "STOPPED"


def test_a_daemonised_supervisor_is_not_treated_as_orphaned(tmp_path, monkeypatch):
    """DETACHED_ENV set at startup means pid 1 is expected, not abandonment."""
    supervisor = _configured_supervisor(tmp_path)
    monkeypatch.setenv(DETACHED_ENV, "1")
    monkeypatch.setattr(os, "getppid", lambda: 1)
    monkeypatch.setattr(supervisor, "_memory", lambda config: {"safe": False, "reasons": ["held"]})

    # It must NOT exit for being orphaned; stop it the ordinary way instead.
    def stop_after_one_poll(*_a, **_k):
        supervisor.store.update(stop_requested=True)
        return {"safe": False, "reasons": ["held"]}

    monkeypatch.setattr(supervisor, "_memory", stop_after_one_poll)
    assert supervisor.run() == 0
    assert not (supervisor.store.read().get("stop_reason") or ""), "daemonised, not orphaned"


def test_the_exit_is_on_by_default():
    """The mechanism is worthless if nothing turns it on. It was False, and a
    repo-wide grep found zero callers opting in."""
    assert ResidentConfig(workspace=".", goal="g").exit_when_orphaned is True
    assert ResidentConfig.from_mapping({"workspace": ".", "goal": "g"}).exit_when_orphaned is True


# --- the last link of the self-build loop ---------------------------------
# _child_workunit built a unit from the model's proposal but dropped `tool` and
# `tool_arguments`, so a resident that asked for filesystem.search got a
# cognition unit instead and never reached the tool surface it can enumerate.


def test_a_child_workunit_can_carry_a_tool_call():
    from hcli.agentos.resident import _child_workunit

    child = _child_workunit("parent-1", {
        "id": "wu-child-1",
        "description": "read my own executor to find the dispatch seam",
        "tool": "filesystem.read",
        "tool_arguments": {"path": "hcli/executors.py"},
    })
    assert child.tool == "filesystem.read"
    assert child.tool_arguments == {"path": "hcli/executors.py"}
    # And it must actually route there, not merely store the field.
    from hcli.executors import BACKEND_TOOL, select_backend_name
    assert select_backend_name(child) == BACKEND_TOOL


def test_a_child_without_a_tool_is_unchanged():
    from hcli.agentos.resident import _child_workunit

    child = _child_workunit("parent-1", {"id": "wu-c", "description": "think about it"})
    assert child.tool is None
    assert child.tool_arguments is None


def test_a_malformed_tool_proposal_is_refused_where_children_are_admitted():
    from hcli.agentos.resident import _child_workunit

    for bad, why in (
        ({"tool": 42}, "non-string tool"),
        ({"tool": "git.status", "tool_arguments": ["not", "a", "map"]}, "non-object arguments"),
        ({"tool_arguments": {"path": "x"}}, "arguments with no tool"),
    ):
        payload = {"id": "wu-bad", "description": "d", **bad}
        try:
            _child_workunit("parent-1", payload)
        except ValueError:
            continue
        raise AssertionError(f"accepted a child with {why}: {payload}")


if __name__ == "__main__":
    # Only the argument-free checks; the loop tests need pytest fixtures.
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and _fn.__code__.co_argcount == 0:
            _fn()
            print(f"ok  {_name}")
    print("all green (pytest runs the fixture-based loop tests)")
