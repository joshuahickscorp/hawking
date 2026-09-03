"""fs.search must accept a file, because that is what it gets pointed at.

The truncation notice tells the model, at the moment a large read is cut:
"use fs.search to find the line a symbol is on, then fs.read that file again
with start_line and end_line". fs.search then rejected the very file it had
just been pointed at with NotADirectoryError, so the model went back to reading
the head -- which is exactly where the symbol was not visible.

Measured consequence: the model reconstructed the signature of _read_file from
memory instead of copying it, and emitted
    'def _read_file(context: ToolContext, args: Dict[str, Any] -> Dict[str, Any:'
against a real line reading
    'def _read_file(context: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:'
-- missing '])' and ']', matching zero places, old_text identical to new_text.
Then it gave up and answered instead of mutating, after 8 resident calls and
1160 seconds.
"""
from __future__ import annotations

import pathlib

import pytest

from hcli import tool_registry as tr


class _Ctx:
    def resolve_read_path(self, p):
        return pathlib.Path(p).resolve()


def _search(**args):
    return tr._search_files(_Ctx(), args)


def test_searching_one_file_finds_the_symbol_and_its_line(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text("import os\n\n\ndef wanted(a: int) -> str:\n    return ''\n")
    out = _search(path=str(target), pattern="def wanted")
    assert len(out["matches"]) == 1
    assert out["matches"][0]["line"] == 4


def test_the_match_carries_the_REAL_bytes_of_the_line(tmp_path):
    """The point is to stop the model reconstructing a signature from memory."""
    target = tmp_path / "mod.py"
    target.write_text("def f(a: dict[str, int]) -> dict[str, int]:\n    pass\n")
    out = _search(path=str(target), pattern="def f(")
    assert "dict[str, int]) -> dict[str, int]:" in out["matches"][0]["text"]


def test_naming_a_file_overrides_a_glob_that_would_exclude_it(tmp_path):
    """Otherwise a default glob silently returns nothing for an explicit file."""
    target = tmp_path / "notes.txt"
    target.write_text("needle here\n")
    out = _search(path=str(target), pattern="needle", glob="*.py")
    assert len(out["matches"]) == 1


def test_directory_search_is_unchanged(tmp_path):
    (tmp_path / "a.py").write_text("needle\n")
    (tmp_path / "b.py").write_text("needle\n")
    (tmp_path / "c.txt").write_text("needle\n")
    out = _search(root=str(tmp_path), pattern="needle", glob="*.py")
    assert len(out["matches"]) == 2


def test_a_path_that_is_neither_file_nor_directory_still_refuses(tmp_path):
    with pytest.raises((NotADirectoryError, FileNotFoundError)):
        _search(path=str(tmp_path / "nope"), pattern="x")


def test_pattern_is_still_required(tmp_path):
    target = tmp_path / "m.py"
    target.write_text("x\n")
    with pytest.raises(ValueError):
        _search(path=str(target))
