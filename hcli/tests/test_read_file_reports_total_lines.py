"""A whole-file fs.read must report total_lines, as the windowed read does.

The windowed branch of _read_file returns total_lines so a caller knows how much
file it has NOT seen. The whole-file branch returns bytes and a truncated flag
and nothing about lines, so a model that reads a file, gets the head, and wants
to page through the rest has no idea how far it has to go.

This test is the SPEC. It fails before the change and passes after it.
"""
from __future__ import annotations

import pathlib

from hcli import tool_registry as tr


class _Ctx:
    def resolve_read_path(self, p):
        return pathlib.Path(p).resolve()


def test_a_whole_file_read_reports_total_lines(tmp_path):
    target = tmp_path / "three.py"
    target.write_text("one\ntwo\nthree\n")
    result = tr._read_file(_Ctx(), {"path": str(target)})
    assert result["total_lines"] == 3


def test_a_single_line_file_reports_one(tmp_path):
    target = tmp_path / "one.py"
    target.write_text("only\n")
    result = tr._read_file(_Ctx(), {"path": str(target)})
    assert result["total_lines"] == 1


def test_the_windowed_read_is_unchanged(tmp_path):
    """The branch that already worked must keep working."""
    target = tmp_path / "four.py"
    target.write_text("a\nb\nc\nd\n")
    result = tr._read_file(_Ctx(), {"path": str(target), "start_line": 2, "end_line": 3})
    assert result["total_lines"] == 4
    assert result["content"] == "b\nc\n"
