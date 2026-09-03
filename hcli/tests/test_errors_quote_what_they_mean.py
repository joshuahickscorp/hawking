"""A rejection must show the bytes it is talking about.

Twice today the same shape: a message named a coordinate the model could not
resolve, the model guessed, and the guess was rejected identically on every
retry.

  * "anchor must occur exactly once ... found 0" -- fixed by quoting the file's
    real bytes at that point.
  * "would not compile: closing parenthesis '}' does not match opening
    parenthesis '(' at line 592 of the RESULTING file" -- a file the model
    never sees. Measured: three attempts on one goal, three identical bracket
    errors, nothing in the message showing the line it meant.

And the receipt that should have made this diagnosable spent its whole 800
character budget on the reply's `content` prose, cutting off at the word
"operations" -- the one part a rejection about an operation needs.
"""
from __future__ import annotations

import json
import pathlib

from hcli.engine import _python_syntax_violation, _rejected_excerpt

TARGET = "hcli/tool_registry.py"
ANCHOR = "    clipped = raw[:limit]\n"


def _op(new_text):
    return json.dumps({"kind": "mutation", "operations": [{
        "op": "replace", "path": TARGET,
        "old_text": ANCHOR, "new_text": new_text,
    }]})


def test_the_anchor_is_unique_so_this_fixture_is_valid():
    """If the anchor stopped matching once, every test below would be vacuous."""
    assert pathlib.Path(TARGET).read_text().count(ANCHOR) == 1


def test_a_syntax_error_quotes_the_line_it_names():
    broken = ANCHOR + '    total = len(raw.decode("utf-8").splitlines()\n'
    message = _python_syntax_violation(_op(broken))
    assert message is not None
    assert "the resulting file reads there:" in message
    assert "splitlines()" in message, message


def test_the_quoted_window_carries_line_numbers_from_the_resulting_file():
    broken = ANCHOR + '    total = len(raw.splitlines()\n'
    message = _python_syntax_violation(_op(broken))
    assert "582:" in message or "583:" in message, message


def test_a_valid_operation_still_says_nothing():
    fine = ANCHOR + "    total_lines = 0\n"
    assert _python_syntax_violation(_op(fine)) is None


def test_the_excerpt_keeps_both_ends_of_a_long_reply():
    """Operations live at the END. A head-only excerpt elides the evidence."""
    text = "HEAD" + "p" * 4000 + '"operations": [{"op": "replace"}]'
    out = _rejected_excerpt(text)
    assert out.startswith("HEAD")
    assert out.endswith('"operations": [{"op": "replace"}]')
    assert "elided" in out


def test_a_short_reply_is_kept_whole():
    assert _rejected_excerpt("small reply") == "small reply"
