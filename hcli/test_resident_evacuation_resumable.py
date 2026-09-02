"""A graceful stop must leave the mission resumable. SIGTERM is not cancel.

The resident's SIGTERM handler called ``Mission.cancel()``. ``cancelled`` is in
``BLOCKED_MISSION_PHASES``, so ``mission_blocked_reason`` then refused to
advance the mission forever -- correctly, given the phase it was handed. The
consequence was backwards:

  * SIGKILL left the mission ``running``; recovery re-ran the interrupted unit
    and lost nothing (observed twice).
  * SIGTERM -- the supervisor's OWN backpressure verb, sent every time free RAM
    dipped below the reserve -- ended the mission permanently.

The live daemon recorded ``evacuation_reason: "free RAM 11336974336 is below
reserve 12884901888"`` and ``cancel_reason: "resident_self_evacuation"`` on the
same mission: routine memory pressure, not an operator, is what killed it.

The negative control is in this file on purpose: ``cancel()`` must STILL block,
or these tests would pass against a build that simply stopped blocking.
"""
from __future__ import annotations

import json
import signal
import threading
import time

import pytest

from hcli.agentos.resident import (
    BLOCKED_MISSION_PHASES,
    TERMINAL_MISSION_PHASES,
    _mission_has_work,
    mission_blocked_reason,
)
from hcli.mission import Mission
from hcli.workunit import WorkUnit


class _BlockingEngine:
    """Holds one unit inside the model call until the test releases it."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.child_pids: list[int] = []

    def execute_workunit(self, wu, context=None):  # noqa: ANN001 - engine duck type
        self.entered.set()
        self.release.wait(timeout=10.0)
        return {"kind": "answer", "content": "done", "operations": [], "tests": []}

    def cancel(self) -> None:
        self.release.set()


def _mission(tmp_path, engine):
    units = {
        "u1": WorkUnit(
            id="u1",
            role="research",
            description="the unit that is in flight when the supervisor calls",
            status="ready",
        )
    }
    return Mission(
        tmp_path,
        goal="stay resumable across a graceful stop",
        units=units,
        engine=engine,
        install_signals=False,
    )


def _run_until_inflight(mission, engine):
    thread = threading.Thread(target=mission.run, daemon=True)
    thread.start()
    assert engine.entered.wait(timeout=10.0), "engine never received the unit"
    return thread


def _state(tmp_path):
    return json.loads((tmp_path / ".hcli" / "mission" / "state.json").read_text())


def test_evacuation_leaves_the_mission_advanceable(tmp_path):
    engine = _BlockingEngine()
    mission = _mission(tmp_path, engine)
    thread = _run_until_inflight(mission, engine)

    # The unit is NOT released: a real evacuation takes the body away while the
    # model call is still outstanding.
    mission.evacuate("free RAM below reserve")
    thread.join(timeout=20.0)
    engine.release.set()
    assert not thread.is_alive(), "evacuate() did not stop the loop"

    assert mission.phase == "evacuated"
    state = _state(tmp_path)
    assert state["phase"] == "evacuated"
    assert state["evacuation_reason"] == "free RAM below reserve"

    # The load-bearing assertions: the supervisor can dispatch the next worker.
    assert "evacuated" not in BLOCKED_MISSION_PHASES
    assert "evacuated" not in TERMINAL_MISSION_PHASES
    assert mission_blocked_reason(tmp_path) is None
    assert _mission_has_work(tmp_path) is True


def test_the_interrupted_unit_is_re_runnable_not_failed(tmp_path):
    """INTERRUPTED is process death, not a verifier verdict. It keeps its retries."""
    engine = _BlockingEngine()
    mission = _mission(tmp_path, engine)
    thread = _run_until_inflight(mission, engine)

    mission.evacuate("grace")
    thread.join(timeout=20.0)
    engine.release.set()

    unit = _state(tmp_path)["units"]["u1"]
    assert unit["status"] == "interrupted", (
        f"in-flight unit came back {unit['status']!r}; a mission that burns a "
        "unit on every memory dip runs out of work rather than finishing it"
    )
    assert unit.get("classification") == "INTERRUPTED", (
        "process death was recorded as a verifier verdict"
    )

    recovered = Mission.from_workspace(tmp_path, goal="x", engine=_BlockingEngine())
    assert recovered.scheduler.units["u1"].status in ("interrupted", "ready")


def test_cancel_still_blocks(tmp_path):
    """Negative control. Cancellation is an operator verb and must stay terminal."""
    engine = _BlockingEngine()
    mission = _mission(tmp_path, engine)
    thread = _run_until_inflight(mission, engine)

    mission.cancel("an operator said stop")
    engine.release.set()
    thread.join(timeout=20.0)

    assert mission.phase == "cancelled"
    assert "cancelled" in BLOCKED_MISSION_PHASES
    reason = mission_blocked_reason(tmp_path)
    assert reason is not None and "cancelled" in reason


def test_the_resident_sigterm_handler_evacuates_and_does_not_cancel(tmp_path, monkeypatch):
    """The call site, not the definition. This is what the daemon actually runs."""
    from hcli.agentos import resident as resident_mod
    from hcli.agentos.resident import ResidentConfig, ResidentDaemon

    calls: list[str] = []

    class FakeMission:
        id = "m-fake"
        session_id = "s-fake"
        scheduler = type("S", (), {"replan": staticmethod(lambda units: None)})()

        def evacuate(self, reason: str = "evacuated") -> None:
            calls.append(f"evacuate:{reason}")

        def cancel(self, reason: str = "cancelled") -> None:
            calls.append(f"cancel:{reason}")

    class FakeAgentOS:
        def __init__(self, *a, **k) -> None:
            self.mission = FakeMission()
            self.controller = None

        def recover_mission(self):
            return self.mission

        def start_mission(self, goal, **k):
            return self.mission

        def checkpoint(self):
            return None

        def run(self):
            # The supervisor's evacuation verb, delivered exactly as _evacuate
            # sends it, while a mission is live.
            signal.raise_signal(signal.SIGTERM)
            time.sleep(0.05)
            return {"status": "evacuated", "evidence": []}

    import hcli.agentos as agentos_pkg

    monkeypatch.setattr(agentos_pkg, "AgentOS", FakeAgentOS, raising=False)
    monkeypatch.setattr(resident_mod, "EventSink", lambda ws: type(
        "S", (), {"write": lambda self, e: None, "close": lambda self: None}
    )(), raising=True)

    daemon = ResidentDaemon(tmp_path)
    config = ResidentConfig(workspace=str(tmp_path), goal="g", interval_s=1.0)
    daemon.store.update(config=config.to_dict())
    state_path = tmp_path / ".hcli" / "resident" / "state.json"
    assert state_path.is_file()

    previous = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)
    try:
        rc = resident_mod._worker_main(str(state_path))
    finally:
        signal.signal(signal.SIGTERM, previous)
        signal.signal(signal.SIGINT, previous_int)

    assert rc == 0
    assert calls == ["evacuate:resident_self_evacuation"], (
        f"the SIGTERM handler did {calls!r}; cancel() here is what made a "
        "graceful shutdown permanently fatal"
    )


@pytest.mark.parametrize("phase", sorted(BLOCKED_MISSION_PHASES))
def test_blocked_phases_are_unchanged(phase):
    """Guard the fix from the lazy version of itself: do not empty the set."""
    assert phase in {"failed", "cancelled", "no_progress"}
