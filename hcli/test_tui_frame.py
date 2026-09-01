"""The frame was measured in code points, so no two rows were the same width.

The top border, the info row and the separator each came out a different length,
multi-line command output got a single trailing `│` on its first line only, and
the box never closed. Every assertion here is about COLUMNS, because that is the
unit the terminal actually draws in -- CJK, emoji and box-drawing characters all
lie about their width under len().
"""
import re
import sys
import unicodedata

import pytest

from hcli.events import EventBus
from hcli.tui import TUI, display_width, frame_width


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


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))
