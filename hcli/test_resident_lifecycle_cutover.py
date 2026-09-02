"""`resident start` must never silently return the incumbent.

LIVE FAILURE: `resident start --goal NEW --interval-s 30 --swap-ceiling 20G`
printed a healthy JSON status and changed nothing. `configure()` returned the
current state when a supervisor was live, and `start_resident` then returned
`daemon.status()`. Neither wrote the requested config; neither said so. The
operator saw goal/interval/ceiling/mission all unchanged and a supervisor at
cycles=818 still resurrecting a terminal smoke mission.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcli.agentos.resident import (
    ResidentAlreadyRunning,
    ResidentConfig,
    ResidentDaemon,
    ResidentSupervisor,
    main,
    mission_blocked_reason,
    retire_incumbent,
    start_resident,
)


def _fake_live(daemon, pid=None):
    """Mark the state as owned by a live supervisor: this process."""
    import os

    from hcli.agentos.resident import process_start_token

    pid = pid or os.getpid()
    daemon.store.update(
        supervisor_pid=pid,
        supervisor_start_token=process_start_token(pid),
        state="RUNNING",
    )


def _configured(tmp_path, goal="goal A", **over):
    daemon = ResidentDaemon(tmp_path)
    daemon.configure(ResidentConfig(workspace=str(tmp_path), goal=goal, **over))
    return daemon


# --- A: the refusal ------------------------------------------------------

def test_configure_refuses_while_a_supervisor_is_live(tmp_path):
    daemon = _configured(tmp_path, "goal A")
    _fake_live(daemon)
    with pytest.raises(ResidentAlreadyRunning) as caught:
        daemon.configure(ResidentConfig(workspace=str(tmp_path), goal="goal B"))
    assert caught.value.existing_goal == "goal A"
    assert "RESIDENT_ALREADY_RUNNING" in str(caught.value)


def test_start_does_not_apply_a_new_config_over_a_live_resident(tmp_path):
    """The exact live failure: new goal/interval/ceiling silently discarded."""
    daemon = _configured(tmp_path, "goal A", interval_s=15.0)
    _fake_live(daemon)
    with pytest.raises(ResidentAlreadyRunning):
        start_resident(
            tmp_path, goal="goal B", interval_s=30.0,
            swap_ceiling_bytes=21474836480, reserve_bytes=12884901888,
        )
    cfg = daemon.store.read()["config"]
    assert cfg["goal"] == "goal A", "the incumbent config was overwritten"
    assert cfg["interval_s"] == 15.0


def test_the_cli_exits_non_zero_on_refusal(tmp_path, capsys):
    daemon = _configured(tmp_path, "goal A")
    _fake_live(daemon)
    code = main(["start", "--workspace", str(tmp_path), "--goal", "goal B"])
    assert code == 3, "a refused start must not exit 0"
    err = capsys.readouterr().err
    assert "RESIDENT_ALREADY_RUNNING" in err
    assert "resident replace" in err, "the refusal must name the remedy"


def test_start_succeeds_when_nothing_owns_the_workspace(tmp_path, monkeypatch):
    spawned = {}

    real_popen = __import__("subprocess").Popen

    def fake_popen(argv, **kw):
        # Only intercept the supervisor spawn; process_start_token and friends
        # still need a real Popen.
        if isinstance(argv, (list, tuple)) and "--supervise" in argv:
            spawned["argv"] = list(argv)

            class _P:
                pid = 4242
            return _P()
        return real_popen(argv, **kw)

    monkeypatch.setattr("hcli.agentos.resident.subprocess.Popen", fake_popen)
    state = start_resident(tmp_path, goal="goal A", interval_s=30.0)
    assert spawned, "no supervisor was spawned"
    assert state["config"]["goal"] == "goal A"
    assert state["config"]["interval_s"] == 30.0


# --- B: replace ----------------------------------------------------------

def test_replace_archives_the_mission_instead_of_deleting_it(tmp_path):
    daemon = _configured(tmp_path, "goal A")
    mission = Path(tmp_path) / ".hcli" / "mission"
    mission.mkdir(parents=True, exist_ok=True)
    (mission / "state.json").write_text(json.dumps({"id": "m-old", "phase": "failed"}))

    report = retire_incumbent(daemon)

    assert not mission.exists(), "mission was left in place"
    archived = Path(report["archived_mission"])
    assert archived.is_dir(), "mission was deleted rather than archived"
    assert json.loads((archived / "state.json").read_text())["id"] == "m-old"


def test_replace_applies_the_new_config(tmp_path, monkeypatch):
    daemon = _configured(tmp_path, "goal A", interval_s=15.0)
    real_popen = __import__("subprocess").Popen

    def fake_popen(argv, **kw):
        if isinstance(argv, (list, tuple)) and "--supervise" in argv:
            return type("P", (), {"pid": 999})()
        return real_popen(argv, **kw)

    monkeypatch.setattr("hcli.agentos.resident.subprocess.Popen", fake_popen)
    state = start_resident(
        tmp_path, goal="goal B", interval_s=30.0,
        swap_ceiling_bytes=21474836480, reserve_bytes=12884901888, replace=True,
    )
    cfg = state["config"]
    assert cfg["goal"] == "goal B"
    assert cfg["interval_s"] == 30.0
    assert cfg["swap_ceiling_bytes"] == 21474836480
    assert cfg["reserve_bytes"] == 12884901888


# --- C: terminal mission must not be resurrected, on the SUPERVISOR path ---

def test_the_supervisor_loop_stops_on_a_terminal_mission(tmp_path):
    """Production path, not a helper: `ResidentSupervisor.run()` must break.

    The incumbent reached ~818 cycles re-dispatching a mission whose own phase
    was `failed`. This drives the real loop.
    """
    daemon = _configured(tmp_path, "goal A", interval_s=0.05)
    mission = Path(tmp_path) / ".hcli" / "mission"
    mission.mkdir(parents=True, exist_ok=True)
    (mission / "state.json").write_text(json.dumps({
        "version": 1, "id": "m-dead", "goal": "g", "phase": "failed",
        "units": {"implement": {"id": "implement", "role": "r",
                                "description": "d", "status": "failed"}},
    }))
    assert mission_blocked_reason(tmp_path)

    sup = ResidentSupervisor(daemon.store.state_path)
    spawns = []
    sup._spawn_worker = lambda *a, **k: spawns.append(1)
    sup._memory = lambda config: {"safe": True, "reasons": []}

    assert sup.run() == 0
    state = daemon.store.read()
    assert not spawns, f"worker was resurrected {len(spawns)}x on a dead mission"
    assert state["state"] == "FAILED"
    assert "m-dead" in str(state.get("error") or state.get("stop_reason") or "")
