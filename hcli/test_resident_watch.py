"""The live viewer must never open a model beside the running resident.

The interactive TUI builds its own Controller; executing a goal there opens a
SECOND 11 GB body next to the resident's. Tonight eight concurrent bodies drove
free RAM to 0.2 GB on this host. The viewer reads durable state only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcli.agentos.resident import ResidentConfig, ResidentDaemon, watch_resident


def _workspace(tmp_path, phase="running"):
    daemon = ResidentDaemon(tmp_path)
    daemon.configure(ResidentConfig(workspace=str(tmp_path), goal="watch me"))
    d = Path(tmp_path) / ".hcli" / "mission"
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps({
        "version": 1, "id": "m-watch", "goal": "watch me", "phase": phase,
        "started_at": 0,
        "units": {"G001": {"id": "G001", "role": "r", "description": "d",
                           "status": "running"}},
    }))
    (d / "mission.log").write_text(
        json.dumps({"ts": 0, "event": "term", "msg": "heartbeat phase=running"}) + "\n"
    )
    return tmp_path


def test_watch_opens_no_runtime_and_no_model(tmp_path, monkeypatch, capsys):
    """Load-bearing: a viewer that can open a body is a second 11 GB process."""
    import subprocess

    real = subprocess.Popen

    def guard(argv, *a, **kw):
        joined = " ".join(map(str, argv)) if isinstance(argv, (list, tuple)) else str(argv)
        assert "ascension" not in joined and "resident" not in joined.replace(
            "hcli.agentos.resident", ""
        ), f"watch spawned a process that looks like a model body: {joined}"
        return real(argv, *a, **kw)

    monkeypatch.setattr(subprocess, "Popen", guard)
    _workspace(tmp_path)

    calls = {"n": 0}

    def stop_after_one(_s):
        calls["n"] += 1
        raise KeyboardInterrupt

    monkeypatch.setattr("hcli.agentos.resident.time.sleep", stop_after_one)
    assert watch_resident(tmp_path, interval_s=0.01) == 0
    assert calls["n"] == 1


def test_watch_renders_the_live_state(tmp_path, monkeypatch, capsys):
    _workspace(tmp_path)
    monkeypatch.setattr(
        "hcli.agentos.resident.time.sleep",
        lambda _s: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    watch_resident(tmp_path, interval_s=0.01)
    out = capsys.readouterr().out
    assert "HCLI RESIDENT" in out
    assert "m-watch" in out, "mission id not rendered"
    assert "G001" in out, "unit not rendered"
    assert "watch me" in out, "goal not rendered"
    assert "BANK" in out


def test_ctrl_c_detaches_without_stopping_anything(tmp_path, monkeypatch, capsys):
    daemon = ResidentDaemon(_workspace(tmp_path))
    before = json.dumps(daemon.store.read(), sort_keys=True)
    monkeypatch.setattr(
        "hcli.agentos.resident.time.sleep",
        lambda _s: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    assert watch_resident(tmp_path, interval_s=0.01) == 0
    assert "still running" in capsys.readouterr().out
    assert json.dumps(daemon.store.read(), sort_keys=True) == before, (
        "watch mutated durable resident state"
    )


def test_watch_survives_a_missing_mission(tmp_path, monkeypatch):
    daemon = ResidentDaemon(tmp_path)
    daemon.configure(ResidentConfig(workspace=str(tmp_path), goal="g"))
    monkeypatch.setattr(
        "hcli.agentos.resident.time.sleep",
        lambda _s: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    assert watch_resident(tmp_path, interval_s=0.01) == 0
