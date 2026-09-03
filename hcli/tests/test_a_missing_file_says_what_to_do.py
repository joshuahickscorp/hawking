"""fs.read on a path that does not exist must be correctable, not a traceback.

Measured in one live goal: three identical FileNotFoundError on
hcli/tests/test_tool_registry.py, one 60-190s model call apiece, because the
error said only that the path was absent. Nothing told the model the file was
one it should CREATE rather than keep guessing at, and nothing named what was
actually in the directory. The directory case had carried that guidance for a
while; the file case never did.
"""
from __future__ import annotations

import pathlib

import pytest

from hcli import tool_registry as tr


class _Ctx:
    def resolve_read_path(self, p):
        return pathlib.Path(p).resolve()


def _read(path):
    with pytest.raises(FileNotFoundError) as caught:
        tr._read_file(_Ctx(), {"path": str(path)})
    return str(caught.value)


def test_a_near_miss_is_offered_the_real_name(tmp_path):
    (tmp_path / "test_mlx_backend.py").write_text("x = 1\n")
    message = _read(tmp_path / "test_mlx_backends.py")
    assert "test_mlx_backend.py" in message, message


def test_a_file_that_simply_is_not_there_is_pointed_at_fs_list(tmp_path):
    """A wrong suggestion is worse than none: it sends the model somewhere real
    and wrong. Below the similarity cutoff, say how to look instead."""
    for name in ("alpha.py", "beta.py", "gamma.py"):
        (tmp_path / name).write_text("x = 1\n")
    message = _read(tmp_path / "test_tool_registry.py")
    assert "fs.list" in message
    assert "3 files" in message
    assert "alpha.py" not in message, "a bad guess was offered as a correction"


def test_the_model_is_told_it_may_be_trying_to_create_the_file(tmp_path):
    """The live loop was reading a test file it was supposed to write."""
    message = _read(tmp_path / "test_new_thing.py")
    assert "CREATE" in message
    assert "do not read it first" in message


def test_a_missing_directory_says_the_directory_is_missing_too(tmp_path):
    message = _read(tmp_path / "no_such_dir" / "x.py")
    assert "neither does" in message
    assert "no_such_dir" in message


def test_a_file_that_exists_is_still_read(tmp_path):
    """The guard must not swallow the ordinary path."""
    target = tmp_path / "real.py"
    target.write_text("value = 42\n")
    out = tr._read_file(_Ctx(), {"path": str(target)})
    assert out["content"] == "value = 42\n"
