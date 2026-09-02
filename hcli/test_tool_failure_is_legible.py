"""A tool failure must say what went wrong, in the error and in the event.

Measured on a live run: the model listed `hcli`, called `fs.read` on it, and got
`FileNotFoundError: /Users/.../hcli`. The path was correct; it was a directory.
The model concluded it had the path wrong and spent five retries and eight model
calls -- about six minutes of a ten-minute goal -- hunting a path that was right
all along. An error that misdescribes the situation cannot be recovered from.

The events said only `ok: false`. The cause had to be reproduced by hand
afterwards, which is the same blindness that made the structured-output failures
undiagnosable.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hcli.engine import Engine
from hcli.tool_registry import ToolContext, _read_file


@pytest.fixture()
def ctx(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    return ToolContext(workspace=tmp_path, repo_root=tmp_path)


def test_reading_a_directory_says_it_is_a_directory(ctx, tmp_path):
    with pytest.raises(IsADirectoryError) as exc:
        _read_file(ctx, {"path": "pkg"})
    message = str(exc.value)
    assert "is a directory" in message
    assert "fs.list" in message, (
        "the error must name the tool that WOULD work; without it the model "
        "guesses, and it guessed five times"
    )


def test_a_genuinely_missing_file_still_says_not_found(ctx):
    """Negative control: do not turn every failure into the directory message."""
    with pytest.raises(FileNotFoundError):
        _read_file(ctx, {"path": "pkg/nope.py"})


def test_a_real_file_still_reads(ctx):
    out = _read_file(ctx, {"path": "pkg/mod.py"})
    assert out["content"].startswith("VALUE = 1")
    assert out["bytes"] == len("VALUE = 1\n")


def test_the_failure_reason_is_carried_on_the_event():
    """`ok: false` with no reason is not observability."""
    import inspect

    src = inspect.getsource(Engine._run_tool_calls)
    assert '"error": failure' in src, "tool events dropped the failure reason again"
    assert "TOOL_ERROR_EVENT_CHARS" in src, "the reason must be bounded"


def test_the_event_reason_is_bounded():
    assert 0 < Engine.TOOL_ERROR_EVENT_CHARS <= 2000


def test_fs_list_works_with_no_arguments(ctx):
    """"List the repo" must be expressible.

    The schema demanded `path` while the handler already defaulted to the
    workspace root, so a bare `fs.list` was rejected for a missing property the
    tool did not actually need. Measured live: the model called it with no
    arguments, read the rejection, and burned the round.
    """
    from hcli.tool_registry import _list_files

    out = _list_files(ctx, {})
    # One level, because recursion is opt-in: this fixture's only file lives in
    # `pkg/`, so the root listing names the DIRECTORY, not the file inside it.
    assert [d["path"] for d in out["directories"]] == ["pkg"], (
        "a bare fs.list must list the workspace root"
    )
    assert [f["path"] for f in _list_files(ctx, {"path": "pkg"})["files"]] == ["mod.py"]


def test_fs_list_still_refuses_a_path_outside_the_read_roots(ctx):
    """Negative control: making the argument optional is not making it unsafe."""
    from hcli.tool_registry import _list_files

    with pytest.raises(PermissionError):
        _list_files(ctx, {"path": "/etc"})


def test_fs_list_is_milliseconds_even_on_a_large_tree(tmp_path):
    """A tool that costs half a minute is not one the model can afford.

    Measured live: a bare `fs.list` on this repo took 28.1 s against fs.read's
    6 ms, because the walk recursed by default, kept walking after the result
    caps were full, and stat()ed files it would never return. This repo holds
    model artifacts and capture directories with tens of thousands of files.
    """
    import time

    from hcli.tool_registry import _list_files

    deep = tmp_path
    for level in range(6):
        deep = deep / f"level{level}"
        deep.mkdir()
        for i in range(60):
            (deep / f"f{i}.py").write_text("x", encoding="utf-8")

    ctx = ToolContext(workspace=tmp_path, repo_root=tmp_path)

    started = time.perf_counter()
    out = _list_files(ctx, {})
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert elapsed_ms < 250, f"a bare fs.list took {elapsed_ms:.0f} ms"
    assert out["directories"], "one level must still be listed"


def test_fs_list_does_not_recurse_unless_asked(tmp_path):
    """Recursion is opt-in: 'what is in this directory' is one level."""
    from hcli.tool_registry import _list_files

    (tmp_path / "top.py").write_text("x", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.py").write_text("x", encoding="utf-8")

    ctx = ToolContext(workspace=tmp_path, repo_root=tmp_path)
    shallow = {f["path"] for f in _list_files(ctx, {})["files"]}
    assert shallow == {"top.py"}

    deep = {f["path"] for f in _list_files(ctx, {"recursive": True})["files"]}
    assert "sub/deep.py" in deep, "recursive:true must still recurse"


def test_a_full_result_stops_the_walk(tmp_path):
    """The cap must end the work, not just trim the output."""
    from hcli.tool_registry import _list_files

    for d in range(12):
        sub = tmp_path / f"d{d}"
        sub.mkdir()
        for i in range(40):
            (sub / f"f{i}.py").write_text("x", encoding="utf-8")

    ctx = ToolContext(workspace=tmp_path, repo_root=tmp_path)
    out = _list_files(ctx, {"recursive": True, "max_results": 5})
    assert out["truncated"] is True
    assert len(out["files"]) <= 5
