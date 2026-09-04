"""A focused excerpt must contain the code the goal names, not just its header.

The window grew symmetrically from the anchor line, so half the budget went to
code ABOVE the function and the body was only half covered. Measured: a goal
naming _read_file (line 514 of hcli/tool_registry.py) received lines 457-571 --
the def and the first half of the body -- while the whole-file return dict it
was asked to change sits at line 582. The model could not copy an anchor it had
never been shown, so it invented one: "if start is None and end is None:", a
plausible line that is not in the file.

The first fix bounded the body with the next SYMBOL, but _python_symbol_lines
also reports locals, so the next symbol after `def _read_file` was the `path`
assigned on the following line. The body ended before it began.
"""
from __future__ import annotations

import pathlib

from hcli.engine import _focused_excerpt

TARGET = pathlib.Path("hcli/tool_registry.py")
GOAL = (
    "Add a total_lines key to the whole-file return dict of _read_file in "
    "hcli/tool_registry.py so that the test passes."
)


def test_the_excerpt_reaches_the_code_the_goal_names():
    src = TARGET.read_text()
    assert "clipped = raw[:limit]" in src, "fixture drifted; the target is gone"
    excerpt = _focused_excerpt(src, GOAL, 6001, str(TARGET))
    assert "def _read_file" in excerpt
    assert "clipped = raw[:limit]" in excerpt, (
        "the window shows the function header but not the region to change"
    )


def test_a_local_name_does_not_end_the_body(tmp_path):
    """The next TOP-LEVEL def bounds a body, not the next assignment."""
    mod = tmp_path / "m.py"
    body = "\n".join(f"    step_{i} = {i}" for i in range(18))
    mod.write_text(
        "import os\n" + "# filler above the function\n" * 12 + "\n\ndef wanted(arg):\n"
        "    path = arg\n" + body + "\n    return MARKER_AT_THE_END\n\n\n"
        "def other():\n    return 2\n"
    )
    excerpt = _focused_excerpt(mod.read_text(), "change wanted", 500, str(mod))
    assert "def wanted" in excerpt
    assert "MARKER_AT_THE_END" in excerpt, excerpt[:200]


def test_a_file_that_fits_is_returned_whole(tmp_path):
    mod = tmp_path / "small.py"
    mod.write_text("def f():\n    return 1\n")
    assert _focused_excerpt(mod.read_text(), "f", 9999, str(mod)) == mod.read_text()


def test_the_most_specific_name_wins_not_the_earliest(tmp_path):
    """A short common name defined early must not outrank the named target.

    min() on (line, name) picked whichever matching symbol appeared first in
    the file, so a goal naming _list_files and directories_seen anchored on
    `path` instead. Measured on hcli/tool_registry.py: the window came out as
    lines 251-403, containing TOOL_SCHEMA and neither the function the goal
    names nor the dict it asks to change -- and the model edited the return
    dict it had actually been shown.
    """
    mod = tmp_path / "m.py"
    mod.write_text(
        "def path():\n    return 1\n\n\n"
        + "# filler\n" * 40
        + "\ndef the_specific_target():\n    marker = 'FOUND_IT'\n    return marker\n"
    )
    excerpt = _focused_excerpt(
        mod.read_text(), "change the_specific_target and its path", 400, str(mod)
    )
    assert "FOUND_IT" in excerpt, excerpt[:200]


def test_the_real_file_anchors_on_the_named_function():
    src = TARGET.read_text()
    ex = _focused_excerpt(src, "add directories_seen to _list_files", 6027, str(TARGET))
    assert "def _list_files" in ex
    assert "directories_seen" in ex
