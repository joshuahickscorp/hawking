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
