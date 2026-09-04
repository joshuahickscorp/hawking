"""fs.list must report how many directories it walked.

_list_files already counts them -- `directories_seen` bounds the walk against
_MAX_LIST_DIRECTORIES -- and then discards the number, so a caller that gets a
truncated listing cannot tell whether it hit the directory ceiling or simply ran
out of matches.

This test is the SPEC. It fails before the change and passes after it. The fix
is one line: the value is already computed and in scope at the return.
"""
from __future__ import annotations

import pathlib

from hcli import tool_registry as tr


class _Ctx:
    def resolve_read_path(self, p):
        return pathlib.Path(p).resolve()


def test_a_flat_listing_reports_one_directory(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    out = tr._list_files(_Ctx(), {"path": str(tmp_path)})
    assert out["directories_seen"] == 1


def test_a_recursive_listing_counts_every_directory(tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "one" / "two").mkdir()
    (tmp_path / "one" / "two" / "f.txt").write_text("x")
    out = tr._list_files(_Ctx(), {"path": str(tmp_path), "recursive": True})
    assert out["directories_seen"] == 3


def test_the_existing_keys_are_untouched(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    out = tr._list_files(_Ctx(), {"path": str(tmp_path)})
    for key in ("root", "glob", "files", "directories", "truncated"):
        assert key in out
