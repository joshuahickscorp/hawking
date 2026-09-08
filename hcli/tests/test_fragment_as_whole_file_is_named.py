"""A rejection that does not name the correction costs the whole unit.

Measured on a real autonomy run. The resident had the RIGHT fix for a path
traversal -- `if '..' in digest` -- and lost the WorkUnit three times because
it sent the changed lines as a whole-file operation. The reply it got back was
"applying your operation to hcli/perception/store.py would not compile:
unexpected indent at line 1 ... fix that operation and keep it short", which
diagnoses without saying which way it is wrong. It retried the same shape twice
more and the unit died.

An indent error at LINE 1 of a whole-file op has exactly one cause: a fragment
supplied where the entire file was required. Naming that, and naming
op='replace' with old_lines/new_lines as the thing to use instead, turns a dead
end into a retry that can succeed.
"""
import pytest

from hcli.engine import _python_syntax_violation as _violation


def _reject(ops):
    """The checker takes a whole mutation reply, not a bare op list."""
    return _violation({"kind": "mutation", "operations": ops})


def _op(kind, path, text):
    return {"op": kind, "path": path, "new_text": text}


FRAGMENT = "        if '..' in digest:\n            raise ValueError('no')\n"


def test_a_fragment_sent_as_a_whole_file_names_the_right_operation():
    msg = _reject([_op("replace_file", "hcli/perception/store.py", FRAGMENT)])
    assert msg, "an indented fragment as a whole file must be rejected"
    assert "replaces the ENTIRE file" in msg
    assert "op='replace'" in msg and "old_lines" in msg, (
        f"the rejection does not name the operation to use instead: {msg}")


def test_create_on_an_existing_file_keeps_its_own_better_message():
    """`create` never reaches the fragment hint, and should not.

    An earlier check catches "already exists, so it cannot be created" and says
    something more useful for that case. Precedence is correct; this pins it so
    the new hint is not widened over it later.
    """
    msg = _reject([_op("create", "hcli/perception/store.py", FRAGMENT)])
    assert msg and "cannot be created" in msg
    assert "replaces the ENTIRE file" not in msg


def test_an_ordinary_syntax_error_keeps_the_plain_hint():
    """The specific hint must not swallow every other compile failure."""
    msg = _reject([_op("replace_file", "x.py", "def f(:\n    pass\n")])
    assert msg
    assert "replaces the ENTIRE file" not in msg, (
        "a generic syntax error was misreported as a fragment mistake")
    assert "fix that operation" in msg


def test_a_replace_fragment_is_still_allowed(tmp_path):
    """op=replace splices INTO a file, so an indented body is correct there.

    This is the false rejection the surrounding code already fixed once; the
    new hint must not reintroduce it.
    """
    assert _reject([_op("replace", "hcli/perception/store.py", FRAGMENT)]) is None
