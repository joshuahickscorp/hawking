"""A truncated tool result must say how to get the rest.

Deep code unreachable, in a new costume. fs.read grew start_line/end_line
precisely so a large file could be read past its head, but a whole-file read is
clamped to a fraction of the context window and the notice only reported that
bytes were dropped.

Measured on the live goal: hcli/tool_registry.py is 2,341 lines and 107,527
characters; the clamp at a 5,632-token usable window shows lines 1 to 169; the
target sits at line 582, 413 lines past the cut. The model asked to read the
file to obtain an exact anchor, received the head, and sent "x" as the anchor.

The system prompt names none of start_line, end_line or fs.search, and should
not: every token there is re-prefilled on every call of every goal. The place
to say it is the moment it matters.
"""
from __future__ import annotations

from hcli.engine import Engine


def _clamped(chars: int) -> str:
    return Engine.__new__(Engine)._clamp_observation("y" * chars)


def test_a_truncated_result_names_the_tools_that_reach_past_the_cut():
    out = _clamped(200_000)
    assert "fs.search" in out
    assert "start_line" in out and "end_line" in out


def test_it_says_the_result_is_only_the_head():
    """'truncated' alone reads as 'the tail was boring', not 'you are blind'."""
    out = _clamped(200_000)
    assert "HEAD" in out
    assert "past the cut" in out


def test_a_result_that_fits_is_returned_untouched():
    """The notice must not appear on output that was never cut."""
    small = "def f():\n    return 1\n"
    assert Engine.__new__(Engine)._clamp_observation(small) == small


def test_the_dropped_byte_count_is_still_reported():
    out = _clamped(200_000)
    assert "characters truncated" in out


def test_the_content_itself_still_survives_the_clamp():
    """A remedy that ate the payload would be a worse bug than the one fixed."""
    out = _clamped(200_000)
    assert out.startswith("yyyy")
    assert len(out) > 1200
