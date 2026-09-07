"""The worked example must teach a read the resident can actually use.

Measured 2026-09-05:

    crates/hawking-core/src/model/qwen38_hybrid_decode.rs
        520,266 characters, 141,009 tokens
        the resident's window is 9,728 tokens -- the file is 14.5x ALL of it

    fs.read with no window returns 4,001 characters and "truncated": true,
    i.e. 0.77% of the file, silently.

fs.read has taken start_line/end_line all along and fs.search reports the line a
match is on. The prompt's worked example showed neither -- it showed an unbounded
whole-file read. The campaign's own line-ops test states the reason this matters:
the worked example "is the only part the resident actually imitates". A model that
imitates an unbounded read sees the top of a file and edits what it never saw.
"""
from __future__ import annotations

import re

from hcli.engine import _SYSTEM_PROMPT


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def test_the_worked_tool_example_uses_a_line_window():
    compact = _compact(_SYSTEM_PROMPT)
    start = compact.index('"kind":"tool_use"')
    example = compact[start:start + 700]
    assert '"start_line"' in example and '"end_line"' in example, (
        "the worked tool_use example does not pass a line window; the resident "
        "imitates this example, and an unbounded fs.read is truncated at ~4,000 "
        "characters"
    )


def test_the_worked_tool_example_searches_before_it_reads():
    compact = _compact(_SYSTEM_PROMPT)
    start = compact.index('"kind":"tool_use"')
    example = compact[start:start + 700]
    search_at = example.find('"fs.search"')
    read_at = example.find('"fs.read"')
    assert search_at != -1, "the example never searches; it cannot know which lines to read"
    assert read_at != -1, "the example never reads"
    assert search_at < read_at, "the example reads before it searches"


def test_the_truncation_limit_is_stated_not_merely_implied():
    assert "truncated" in _SYSTEM_PROMPT, (
        "the prompt never tells the resident that an unbounded read is truncated, "
        "so a truncated read looks like a complete one"
    )
