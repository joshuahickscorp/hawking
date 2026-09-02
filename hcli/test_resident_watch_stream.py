"""Watch v2: the live event stream reaches the worker's sink and the
resident watcher's transcript, and its input line steers/banks correctly.

This does not re-test EventSink or stream_render themselves (see
test_event_sink.py / test_stream_render.py) -- it tests the WIRING: that
_worker_main actually subscribes a sink to every event (not just
runtime_ready), that watch_resident actually tails events.jsonl into visible
lines, that the compact unit summary really stays compact, and that a
submitted input line is routed to bank vs steer correctly.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import hcli.agentos.resident as R
from hcli.agentos.event_sink import EventSink, read_events
from hcli.agentos.resident import (
    GoalBank,
    ResidentConfig,
    ResidentDaemon,
    SteeringQueue,
    _watch_handle_line,
    _watch_unit_summary,
    _worker_main,
    watch_resident,
)


def _configure(tmp_path, goal="watch me"):
    daemon = ResidentDaemon(tmp_path)
    daemon.configure(ResidentConfig(workspace=str(tmp_path), goal=goal))
    return daemon


# --------------------------------------------------------------------------
# C1: the worker's sink receives every event, not just runtime_ready.
# --------------------------------------------------------------------------

class _FakeBus:
    def __init__(self):
        self.subscribers = []

    def subscribe(self, cb):
        self.subscribers.append(cb)

    def fire(self, event):
        for cb in self.subscribers:
            cb(event)


class _FakeController:
    def __init__(self):
        self.bus = _FakeBus()

    def shutdown(self):
        pass


class _FakeMission:
    id = "m-fake"
    session_id = "m-fake"


class _FakeAgent:
    def __init__(self, workspace, model=None, runtime_count=None, repo_root=None):
        self.controller = _FakeController()
        self.mission = None

    def start_mission(self, goal):
        self.mission = _FakeMission()

    def recover_mission(self):
        self.mission = _FakeMission()

    def run(self):
        # Simulate the engine emitting real activity mid-run -- a
        # non-runtime_ready event the old code silently dropped.
        self.controller.bus.fire(SimpleNamespace(
            type="tool_call_started", data={"tool": "sonar_probe"},
        ))
        return {"evidence": None, "status": "ok"}

    def checkpoint(self):
        pass


def test_worker_sink_receives_a_non_runtime_ready_event(tmp_path, monkeypatch):
    daemon = _configure(tmp_path)
    monkeypatch.setattr("hcli.agentos.AgentOS", _FakeAgent)
    monkeypatch.setattr(R.signal, "signal", lambda *a, **k: None)

    rc = _worker_main(str(daemon.store.state_path))
    assert rc == 0

    events_path = Path(tmp_path) / ".hcli" / "mission" / "events.jsonl"
    assert events_path.is_file(), "worker left no durable event trail"
    events, _ = read_events(events_path)
    types = [e.get("type") for e in events]
    assert "tool_call_started" in types, (
        f"non-runtime_ready event never reached the sink; saw {types!r}"
    )


def test_mutation_sink_wiring_actually_gates_the_test(tmp_path, monkeypatch):
    """Mutation check: unsubscribing the sink must make the above fail."""
    daemon = _configure(tmp_path)
    monkeypatch.setattr("hcli.agentos.AgentOS", _FakeAgent)
    monkeypatch.setattr(R.signal, "signal", lambda *a, **k: None)

    # Mutate: the fake bus no longer records the second subscriber, so the
    # worker's sink never sees the mid-run event -- reproduces "only
    # runtime_ready is wired" without touching resident.py itself.
    real_subscribe = _FakeBus.subscribe
    calls = {"n": 0}

    def gutted_subscribe(self, cb):
        calls["n"] += 1
        if calls["n"] == 1:
            real_subscribe(self, cb)  # keep only the first (on_runtime_ready)
        # drop every later subscriber -- simulates the pre-fix worker

    monkeypatch.setattr(_FakeBus, "subscribe", gutted_subscribe)
    rc = _worker_main(str(daemon.store.state_path))
    assert rc == 0

    events_path = Path(tmp_path) / ".hcli" / "mission" / "events.jsonl"
    events, _ = read_events(events_path) if events_path.is_file() else ([], 0)
    types = [e.get("type") for e in events]
    assert "tool_call_started" not in types, (
        "mutation did not actually gut the sink wiring -- test proves nothing"
    )


# --------------------------------------------------------------------------
# C2: read_events tailing drives watch_resident's transcript.
# --------------------------------------------------------------------------

def _workspace(tmp_path, phase="running"):
    daemon = ResidentDaemon(tmp_path)
    daemon.configure(ResidentConfig(workspace=str(tmp_path), goal="watch me"))
    d = Path(tmp_path) / ".hcli" / "mission"
    d.mkdir(parents=True, exist_ok=True)
    (d / "state.json").write_text(json.dumps({
        "version": 1, "id": "m-watch", "session_id": "m-watch", "goal": "watch me",
        "phase": phase, "started_at": 0,
        "units": {"G001": {"id": "G001", "role": "r", "description": "d",
                           "status": "running"}},
    }))
    return tmp_path


def test_watch_tails_new_events_into_the_transcript(tmp_path, monkeypatch, capsys):
    workspace = _workspace(tmp_path)
    sink = EventSink(workspace)
    sink.write({"type": "tool_call_started", "data": {"tool": "sonar_probe"}})

    calls = {"n": 0}

    def fake_sleep(_s):
        calls["n"] += 1
        if calls["n"] == 1:
            # written between the first and second tick -- proves the
            # transcript tails NEW events across ticks, not just at start.
            sink.write({"type": "tool_call_finished",
                        "data": {"tool": "sonar_probe", "ok": True, "elapsed_s": 1.5}})
            return
        raise KeyboardInterrupt

    monkeypatch.setattr("hcli.agentos.resident.time.sleep", fake_sleep)
    assert watch_resident(workspace, interval_s=0.01) == 0
    out = capsys.readouterr().out
    assert "sonar_probe" in out, "first tick's event never reached the transcript"
    assert "ok" in out and "1.5s" in out, "second tick's new event was not tailed"
    assert calls["n"] == 2


def test_mutation_transcript_wiring_actually_gates_the_test(tmp_path, monkeypatch, capsys):
    """Mutation check via monkeypatch: stub render_event to always return []
    (as if watch_resident never called it) and confirm the test above fails."""
    workspace = _workspace(tmp_path)
    sink = EventSink(workspace)
    sink.write({"type": "tool_call_started", "data": {"tool": "sonar_probe"}})

    monkeypatch.setattr("hcli.agentos.resident.render_event", lambda event: [])
    monkeypatch.setattr(
        "hcli.agentos.resident.time.sleep",
        lambda _s: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    assert watch_resident(workspace, interval_s=0.01) == 0
    out = capsys.readouterr().out
    assert "sonar_probe" not in out, "mutation did not actually break tailing"


# --------------------------------------------------------------------------
# C3: falling behind a rotation is signalled, not silently swallowed.
#
# events.jsonl rotates to a single ".1" generation past EventSink's
# max_bytes, and read_events only ever tails the live file -- a watcher that
# falls behind by more than one rotation window loses those events for good.
# EventSink.write's own strictly-increasing "seq" makes that loss detectable
# even though it is not recoverable from a bounded single-generation
# rotation, so watch_resident must at least say so.
# --------------------------------------------------------------------------

def test_watch_signals_a_gap_when_it_falls_behind_a_rotation(tmp_path, monkeypatch, capsys):
    workspace = _workspace(tmp_path)
    sink = EventSink(workspace, max_bytes=2000)
    for i in range(50):
        sink.write({"type": "heartbeat", "data": {"i": i}})

    calls = {"n": 0}

    def fake_sleep(_s):
        calls["n"] += 1
        if calls["n"] == 1:
            # Written between the first and second tick, with no read in
            # between -- forces events.jsonl to rotate past the reader's
            # last offset, exactly the "fell behind a rotation" scenario.
            for i in range(50, 400):
                sink.write({"type": "heartbeat", "data": {"i": i}})
            return
        raise KeyboardInterrupt

    monkeypatch.setattr("hcli.agentos.resident.time.sleep", fake_sleep)
    assert watch_resident(workspace, interval_s=0.01) == 0
    out = capsys.readouterr().out
    assert "event(s) lost" in out, "no gap was signalled after falling behind a rotation"
    assert calls["n"] == 2


def test_mutation_gap_signal_actually_gates_the_test(tmp_path, monkeypatch, capsys):
    """Mutation check: without the seq-discontinuity check, watch_resident
    must go back to silently swallowing the gap."""
    workspace = _workspace(tmp_path)
    sink = EventSink(workspace, max_bytes=2000)
    for i in range(50):
        sink.write({"type": "heartbeat", "data": {"i": i}})

    calls = {"n": 0}

    def fake_sleep(_s):
        calls["n"] += 1
        if calls["n"] == 1:
            for i in range(50, 400):
                sink.write({"type": "heartbeat", "data": {"i": i}})
            return
        raise KeyboardInterrupt

    monkeypatch.setattr("hcli.agentos.resident.time.sleep", fake_sleep)
    # Gut the gap check itself by stripping "seq" from every event before
    # watch_resident sees it -- its discontinuity check can then never fire,
    # reproducing the pre-fix silent swallow.
    real_read_events = R.read_events

    def stripped_seq(path, **kw):
        events, off = real_read_events(path, **kw)
        for e in events:
            e.pop("seq", None)
        return events, off

    monkeypatch.setattr(R, "read_events", stripped_seq)
    assert watch_resident(workspace, interval_s=0.01) == 0
    out = capsys.readouterr().out
    assert "event(s) lost" not in out, "mutation did not actually silence the gap signal"


# --------------------------------------------------------------------------
# Compact unit summary: never all 68 lines.
# --------------------------------------------------------------------------

def test_compact_unit_summary_never_prints_all_units(tmp_path):
    # 30 active (running+failed) units -- well over the 12-unit cap, so this
    # actually exercises the active[:cap] slice and the overflow line, not
    # just the completed-unit filter.
    units = {}
    for i in range(40):
        units[f"G{i:03d}"] = {"status": "completed"}
    for i in range(40, 60):
        units[f"G{i:03d}"] = {"status": "running"}
    for i in range(60, 70):
        units[f"G{i:03d}"] = {"status": "failed"}
    assert len(units) == 70

    lines = _watch_unit_summary(units)
    # counts line + 12 capped names + 1 overflow line, never one per unit.
    assert len(lines) == 14, f"printed {len(lines)} lines for 70 units -- cap not applied"
    joined = "\n".join(lines)
    assert "G040" in joined, "the first active unit (within the cap) should be named"
    assert "G051" in joined, "the 12th active unit (last within the cap) should be named"
    assert "G052" not in joined, "the 13th active unit is past the cap and must be omitted"
    assert "G000" not in joined, "a merely-completed unit should not be named"
    assert "18 more running/failed" in joined, "overflow count must reflect what the cap cut (30 active - 12 shown)"


def test_mutation_unit_summary_cap_actually_gates_the_test(monkeypatch):
    """Mutation check: raising the real cap to swallow every unit must
    reproduce the flood -- proves WATCH_UNIT_SUMMARY_CAP (not incidental
    fixture size) is what keeps the real _watch_unit_summary compact."""
    units = {f"G{i:03d}": {"status": "failed"} for i in range(68)}

    # Sanity: with the real cap, the sibling test's assertion holds.
    assert len(_watch_unit_summary(units)) < 68

    monkeypatch.setattr(R, "WATCH_UNIT_SUMMARY_CAP", 68)
    lines = _watch_unit_summary(units)
    assert not (len(lines) < 68), "mutation did not actually restore the flood"


# --------------------------------------------------------------------------
# Non-slash line -> steer; "/bank ..." line -> GoalBank.
# --------------------------------------------------------------------------

def test_non_slash_line_steers_and_slash_bank_line_banks(tmp_path):
    workspace = _workspace(tmp_path)
    root = Path(workspace)
    mission = {"id": "m-watch", "session_id": "m-watch"}

    quit_now, extra = _watch_handle_line(root, mission, "/bank fix the thing")
    assert quit_now is False
    assert any("banked" in line for line in extra), extra
    bank_snapshot = GoalBank(root).snapshot()
    assert bank_snapshot["queued_count"] == 1
    assert bank_snapshot["queued"][0]["goal"] == "fix the thing"
    assert bank_snapshot["queued"][0]["mode"] == "auto"

    quit_now, extra = _watch_handle_line(root, mission, "/bank mission persist this")
    second_item = GoalBank(root).snapshot()["queued"][1]
    assert second_item["mode"] == "mission"

    quit_now, extra = _watch_handle_line(root, mission, "consider using a smaller batch")
    assert quit_now is False
    assert any("steer queued" in line for line in extra), extra
    # The steer must NOT have touched the goal bank.
    assert GoalBank(root).snapshot()["queued_count"] == 2
    # It must have landed in the mission's own SteeringQueue file.
    events = SteeringQueue(str(root), "m-watch").all()
    assert any(e.text == "consider using a smaller batch" for e in events), events


def test_slash_steer_verb_steers_like_bare_text(tmp_path):
    """The interactive TUI has a real /steer verb; a user reflexively typing
    it here must land the exact same steer bare text would, not an 'unknown
    command' error (finding: /steer used to fall into the generic '/' catch-
    all before it ever reached _watch_steer)."""
    workspace = _workspace(tmp_path)
    root = Path(workspace)
    mission = {"id": "m-watch", "session_id": "m-watch"}

    quit_now, extra = _watch_handle_line(root, mission, "/steer use a smaller batch")
    assert quit_now is False
    assert any("steer queued" in line for line in extra), extra
    assert not any("unknown command" in line for line in extra), extra
    events = SteeringQueue(str(root), "m-watch").all()
    assert any(e.text == "use a smaller batch" for e in events), events

    # No text after the verb: a usage hint, not a silent no-op or a crash.
    quit_now, extra = _watch_handle_line(root, mission, "/steer")
    assert quit_now is False
    assert any("usage" in line for line in extra), extra


def test_mutation_slash_steer_actually_gates_the_test(tmp_path):
    """Mutation check: dropping the /steer branch must reproduce the
    'unknown command' failure the finding described."""
    workspace = _workspace(tmp_path)
    root = Path(workspace)
    mission = {"id": "m-watch", "session_id": "m-watch"}

    def gutted_handle_line(root, mission, line):
        text = line.strip()
        if text.startswith("/bank mission "):
            return False, R._watch_bank(root, text[len("/bank mission "):], "mission")
        if text.startswith("/bank "):
            return False, R._watch_bank(root, text[len("/bank "):], "auto")
        if text.startswith("/"):
            return False, [f"! unknown command {text.split()[0]!r} -- /help lists commands"]
        return False, R._watch_steer(root, mission, text)

    _, extra = gutted_handle_line(root, mission, "/steer use a smaller batch")
    assert any("unknown command" in line for line in extra), (
        "mutation did not actually reproduce the pre-fix rejection", extra,
    )


def test_mutation_steer_vs_bank_dispatch_actually_gates_the_test(tmp_path):
    """Mutation check: route everything through bank (as if the leading-'/'
    check were removed) and confirm a plain steer line wrongly bank-adds."""
    workspace = _workspace(tmp_path)
    root = Path(workspace)

    def gutted_handle_line(root, mission, line):
        return R._watch_bank(root, line, "auto")

    before = GoalBank(root).snapshot()["queued_count"]
    gutted_handle_line(root, {"id": "m-watch"}, "this should have been a steer")
    after = GoalBank(root).snapshot()["queued_count"]
    assert after == before + 1, "mutation did not actually misroute the steer as a bank"


# --------------------------------------------------------------------------
# Non-tty stdin degrades without raising.
# --------------------------------------------------------------------------

def test_non_tty_stdin_degrades_without_raising(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    monkeypatch.setattr(
        "hcli.agentos.resident.time.sleep",
        lambda _s: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    assert watch_resident(workspace, interval_s=0.01) == 0


def test_stdin_isatty_raising_also_degrades_without_raising(tmp_path, monkeypatch):
    class _NoIsatty:
        def isatty(self):
            raise RuntimeError("no controlling terminal")

    monkeypatch.setattr(R.sys, "stdin", _NoIsatty())
    monkeypatch.setattr(
        "hcli.agentos.resident.time.sleep",
        lambda _s: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    workspace = _workspace(tmp_path)
    assert watch_resident(workspace, interval_s=0.01) == 0


# --------------------------------------------------------------------------
# Forbidden substrings stay out of watch_resident's own source.
# --------------------------------------------------------------------------

def test_watch_resident_source_has_no_forbidden_substrings():
    src = inspect.getsource(watch_resident)
    for bad in ("Popen", "subprocess", "start_resident", "request_stop"):
        assert bad not in src, f"watch_resident's source regained {bad!r}"
