"""The frame was measured in code points, so no two rows were the same width.

The top border, the info row and the separator each came out a different length,
multi-line command output got a single trailing `│` on its first line only, and
the box never closed. Every assertion here is about COLUMNS, because that is the
unit the terminal actually draws in -- CJK, emoji and box-drawing characters all
lie about their width under len().
"""
import re
import sys
import threading
import time
import unicodedata

import pytest

from hcli.events import Event, EventBus
from hcli.tui import TUI, display_width, frame_width


class _Recorder:
    """Thread-safe stand-in for stdout so live writes are observable."""

    def __init__(self, tty: bool = False):
        self._chunks = []
        self._lock = threading.Lock()
        self._tty = tty
        self.watching = False
        self.gate = threading.Event()

    def write(self, s):
        s = str(s)
        with self._lock:
            self._chunks.append((time.monotonic(), s))
        if self.watching and s.strip():
            self.gate.set()
        return len(s)

    def flush(self):
        return None

    def isatty(self):
        return self._tty

    def text(self):
        with self._lock:
            return "".join(chunk for _t, chunk in self._chunks)


@pytest.fixture
def tui(monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    return TUI(EventBus(), "/tmp/ws", "qwen3.8", 1)


def _ref_width(line: str) -> int:
    """Column count computed independently of the code under test.

    Measuring the frame with the frame's own width function is a tautology: a
    len()-based renderer pads len()-consistently and every row still "matches".
    """
    visible = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", line)
    return sum(
        0 if unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Me", "Cf")
        else (2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1)
        for ch in visible
    )


def _widths(rendered: str):
    return {_ref_width(line) for line in rendered.splitlines()}


def test_EVERY_row_of_the_frame_is_the_same_width(tui):
    tui.transcript = ["You: /help", "Commands:", "  /help - show this help"]
    assert _widths(tui.render()) == {frame_width()}


def test_the_right_border_is_actually_the_LAST_column(tui):
    """Equal width is not enough if a row is padded past its own border."""
    tui.transcript = ["short", "a much longer transcript entry than the first"]
    for line in tui.render().splitlines():
        assert line[-1] in "┐┤┘│", line


def test_a_WIDE_character_costs_two_columns_not_one(tui):
    # Under len() this row measured 4 short and the border walked left.
    tui.transcript = ["You: 東京都の天気は？", "答え: 晴れ"]
    assert _widths(tui.render()) == {frame_width()}
    assert display_width("東京") == 4
    assert display_width("ab") == 2


def test_an_EMOJI_costs_two_columns(tui):
    tui.transcript = ["✓ done 🚀", "● working 👍"]
    assert _widths(tui.render()) == {frame_width()}
    assert display_width("🚀") == 2


def test_an_OVERLONG_line_is_wrapped_not_allowed_to_blow_the_frame_out(tui):
    tui.transcript = ["x" * 500]
    rendered = tui.render()
    assert _widths(rendered) == {frame_width()}
    # Wrapped, not truncated: the content survives.
    assert sum(line.count("x") for line in rendered.splitlines()) == 500


def test_an_OVERLONG_unbroken_CJK_run_also_stays_inside_the_frame(tui):
    tui.transcript = ["東" * 200]
    assert _widths(tui.render()) == {frame_width()}


def test_MULTILINE_content_becomes_one_padded_row_per_line(tui):
    """A single transcript entry holding /help output used to close only its
    first line, which is how the box came to hang open."""
    tui.transcript = ["Commands:\n  /help - show this help\n  /exit - exit HCLI"]
    rows = tui.render_transcript().splitlines()
    assert len(rows) == 3
    assert _widths(tui.render_transcript()) == {frame_width()}


def test_ANSI_escapes_are_stripped_before_measuring(tui):
    tui.transcript = ["\x1b[31merror\x1b[0m from the runtime"]
    assert _widths(tui.render()) == {frame_width()}
    assert display_width("\x1b[31mred\x1b[0m") == 3


def test_TABS_do_not_smuggle_extra_columns_past_the_border(tui):
    tui.transcript = ["key\tvalue"]
    assert _widths(tui.render()) == {frame_width()}


def test_the_width_FOLLOWS_the_terminal_and_clamps_to_something_readable(monkeypatch):
    monkeypatch.setenv("COLUMNS", "64")
    assert frame_width() == 64
    monkeypatch.setenv("COLUMNS", "500")
    assert frame_width() == 100
    monkeypatch.setenv("COLUMNS", "12")
    assert frame_width() == 40
    monkeypatch.delenv("COLUMNS")
    assert 40 <= frame_width() <= 100


def test_the_frame_stays_square_at_EVERY_admitted_width(monkeypatch):
    bus = EventBus()
    for cols in ("12", "40", "57", "80", "500"):
        monkeypatch.setenv("COLUMNS", cols)
        t = TUI(bus, "/tmp/ws", "qwen3.8 ?B unknown", 3)
        t.transcript = ["You: /status", "● working on 東京\ttask", "x" * 300]
        t.status = "● running a mission with a rather long label attached to it"
        assert _widths(t.render()) == {frame_width()}, cols


def test_an_EMPTY_transcript_still_renders_a_closed_box(tui):
    assert tui.transcript == []
    rendered = tui.render()
    assert "(no activity yet)" in rendered
    assert rendered.splitlines()[-1].startswith("└")
    assert _widths(rendered) == {frame_width()}


def test_runtime_events_leave_a_cold_start_trace_in_the_tui(tui):
    tui._on_event(type("Event", (), {"type": "runtime_loading", "data": {"model": "sealed-3.14"}})())
    tui._on_event(type("Event", (), {"type": "runtime_ready", "data": {"admitted": 1}})())
    tui._on_event(type("Event", (), {"type": "workunit_started", "data": {}})())

    assert "loading resident" in tui.transcript[0]
    assert "resident ready" in tui.transcript[1]
    assert tui.status == "● generating"

def test_live_interval_defaults_to_one_second(tui):
    assert tui._live_interval == 1.0


def test_live_progress_prints_before_on_input_returns(monkeypatch):
    """The measured defect: TUI.run printed only after on_input returned."""
    monkeypatch.setenv("COLUMNS", "80")
    rec = _Recorder(tty=False)
    tui = TUI(
        EventBus(),
        "/tmp/ws",
        "qwen3.8",
        1,
        stream=rec,
        tty=False,
        live_interval=0.2,
    )
    calls = {"n": 0}

    def prompt(msg):
        calls["n"] += 1
        if calls["n"] == 1:
            return "hello"
        raise EOFError

    tui._prompt_fn = prompt
    during = []

    def on_input(text):
        rec.gate.clear()
        rec.watching = True
        ok = rec.gate.wait(timeout=2.0)
        during.append(rec.text())
        time.sleep(0.35)
        rec.watching = False
        assert ok, f"no live output during on_input: {rec.text()!r}"

    assert tui.run(on_input) == 0
    blob = during[0]
    assert "working" in blob or "thinking" in blob or "…" in blob
    # A second tick after the first proves this is a 1 Hz-class ticker,
    # not a single print that happened to race with on_input's entry.
    full = rec.text()
    assert full.count("working") + full.count("thinking") >= 2


def test_tool_call_is_visible_in_the_transcript_as_it_happens(tui):
    tui._on_event(Event("tool_call_started", {"tool": "fs.read"}))
    assert any("fs.read" in entry for entry in tui.transcript)
    tui._on_event(
        Event("tool_call_finished", {"tool": "fs.read", "ok": True, "elapsed_s": 0.4})
    )
    joined = "\n".join(tui.transcript)
    assert "fs.read" in joined
    assert "ok" in joined
    # tool_invoked is still emitted; do not double-print the same outcome.
    before = list(tui.transcript)
    tui._on_event(
        Event("tool_invoked", {"tool": "fs.read", "ok": True, "elapsed_s": 0.4})
    )
    assert tui.transcript == before


def test_legacy_tool_invoked_still_renders_name_and_outcome(tui):
    tui._on_event(Event("tool_invoked", {"tool": "fs.list", "ok": False}))
    joined = "\n".join(tui.transcript)
    assert "fs.list" in joined
    assert "failed" in joined


def test_tool_call_prints_during_the_turn_not_after(monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    rec = _Recorder(tty=False)
    bus = EventBus()
    tui = TUI(bus, "/tmp/ws", "qwen3.8", 1, stream=rec, tty=False, live_interval=0.5)
    calls = {"n": 0}

    def prompt(msg):
        calls["n"] += 1
        if calls["n"] == 1:
            return "look"
        raise EOFError

    tui._prompt_fn = prompt
    seen = []

    def on_input(text):
        bus.emit("tool_call_started", {"tool": "fs.read"})
        bus.emit(
            "tool_call_finished",
            {"tool": "fs.read", "ok": True, "elapsed_s": 0.2},
        )
        seen.append(rec.text())
        assert "fs.read" in seen[0]
        assert "ok" in seen[0]

    assert tui.run(on_input) == 0
    assert "fs.read" in seen[0]


def test_raw_chain_of_thought_is_never_rendered(tui):
    secret = "SECRET_CHAIN_OF_THOUGHT"
    tui._on_event(
        Event(
            "final_response",
            {"content": f"<think>{secret}</think>visible answer"},
        )
    )
    joined = "\n".join(tui.transcript)
    rendered = tui.render()
    assert secret not in joined
    assert secret not in rendered
    assert "visible answer" in joined
    tui._on_event(
        Event(
            "heartbeat",
            {
                "phase": "thinking",
                "elapsed_s": 14,
                "text": secret,
                "content": secret,
                "message": secret,
            },
        )
    )
    assert secret not in tui.status
    assert secret not in tui.render()
    assert "thinking" in tui.status
    assert "14" in tui.status


def test_nontty_live_progress_emits_no_ansi(monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    rec = _Recorder(tty=False)
    tui = TUI(
        EventBus(),
        "/tmp/ws",
        "qwen3.8",
        1,
        stream=rec,
        tty=False,
        live_interval=0.2,
    )
    tui._begin_turn()
    tui._on_event(Event("heartbeat", {"phase": "thinking", "elapsed_s": 3}))
    tui._on_event(Event("tool_call_started", {"tool": "fs.read"}))
    tui._on_event(
        Event("tool_call_finished", {"tool": "fs.read", "ok": True, "elapsed_s": 0.1})
    )
    tui._end_turn()
    blob = rec.text()
    assert "\x1b" not in blob
    assert "\r" not in blob
    assert "thinking" in blob or "working" in blob
    assert "fs.read" in blob


def test_tty_status_updates_in_place_with_carriage_return(monkeypatch):
    monkeypatch.setenv("COLUMNS", "80")
    rec = _Recorder(tty=True)
    tui = TUI(
        EventBus(),
        "/tmp/ws",
        "qwen3.8",
        1,
        stream=rec,
        tty=True,
        live_interval=0.2,
    )
    tui._begin_turn()
    time.sleep(0.05)
    tui._end_turn()
    blob = rec.text()
    assert "\r" in blob
    assert "\x1b" not in blob


def test_goal_compiled_shows_unit_count(tui):
    tui._on_event(Event("goal_compiled", {"workunits": 4}))
    assert any("4 units" in entry for entry in tui.transcript)


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))


# --- a failed turn must not end the session, and a paste must not flood it ---
# LIVE FAILURE: a context-preflight refusal propagated out of the unguarded
# `on_input(text)` in TUI.run(), the REPL exited, and the "> " prompt never
# came back -- there was no way to type /status or /steer to learn why.


class _Bus:
    def __init__(self):
        self.handlers = []

    def subscribe(self, fn):
        self.handlers.append(fn)

    def emit(self, type_, data=None):
        from hcli.events import Event
        for fn in self.handlers:
            fn(Event(type_, data or {}))


def _tui(inputs):
    from hcli.tui import TUI
    t = TUI(event_bus=_Bus(), workspace="/tmp/ws")
    it = iter(inputs)

    def prompt(_msg):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    t._prompt_fn = prompt
    return t


def test_a_raising_turn_keeps_the_session_alive():
    from hcli.engine import EngineError
    seen = []

    def on_input(text):
        seen.append(text)
        if text == "boom":
            raise EngineError("demand 15016 exceeds per-request context 8192")

    t = _tui(["boom", "/status", "/exit"])
    assert t.run(on_input) == 0
    assert seen == ["boom", "/status", "/exit"], (
        "the prompt must come back after a failed turn"
    )
    assert any("EngineError" in row for row in t.transcript), t.transcript


def test_ctrl_c_cancels_the_turn_not_the_session():
    seen = []

    def on_input(text):
        seen.append(text)
        if text == "long":
            raise KeyboardInterrupt

    t = _tui(["long", "/status", "/exit"])
    assert t.run(on_input) == 0
    assert seen == ["long", "/status", "/exit"]
    assert any("cancelled" in row for row in t.transcript), t.transcript


def test_exit_still_ends_the_session():
    t = _tui(["/exit", "never"])
    seen = []
    assert t.run(seen.append) == 0
    assert seen == ["/exit"]


def test_a_large_paste_is_echoed_as_a_receipt_not_the_text():
    from hcli.tui import summarize_paste, _PASTE_ECHO_LIMIT
    small = "just a normal goal"
    assert summarize_paste(small) == small

    body = "HAWKING SOVEREIGN ULTRAGOAL\n" + ("x" * 40_000)
    out = summarize_paste(body)
    assert len(out) < 200, f"receipt is itself huge: {len(out)}"
    assert "pasted text" in out
    assert f"{len(body):,}" in out
    assert "x" * 200 not in out, "the paste body leaked into the frame"


def test_the_paste_receipt_reaches_the_transcript():
    t = _tui([])
    big = "GOAL\n" + ("y" * 30_000)
    t._on_event(_Event("user_message", {"text": big}))
    row = t.transcript[-1]
    assert "pasted text" in row and len(row) < 220, row
    assert len(t.render()) < 20_000, "one paste must not blow up the frame"


class _Event:
    def __init__(self, type_, data):
        self.type = type_
        self.data = data
