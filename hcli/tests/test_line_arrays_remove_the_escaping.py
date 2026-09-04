"""Operations may send LINES, so there is no newline to escape wrongly.

new_text is one JSON string, so every newline in a patch body is an escape the
model has to get right. Measured on a Gate 1 attempt with the exact target bytes
pre-supplied: the resident produced a test body containing a bytes literal, the
reply came back with "unexpected character after line continuation character",
and it failed three attempts running. The same reply's anchor matched on attempt
two and missed on attempts one and three -- the same fragility, oscillating.

A list of plain lines has nothing to escape, so there is nothing to get wrong.
This is the same principle as fs.read(start_line=): let the harness do the
deterministic mechanics and let the model spend inference on judgment.

ONE resolver serves the applier and the preflight. Two readers disagreeing about
what an operation says is exactly the defect that let a bad anchor reach
_apply_operations with the contract reporting no complaint.
"""
from __future__ import annotations

import json
import pathlib

from hcli.engine import Engine, _operation_text, _python_syntax_violation
from hcli.workspace import Workspace


class _Pool:
    model_path = "sealed-3.14"
    topology = "process"
    requested_n = 1
    admitted_n = 1
    repo_root = "."


def _engine(tmp_path):
    return Engine(Workspace(str(tmp_path)), runtime_provider=lambda: _Pool())


def test_lines_join_with_a_trailing_newline():
    assert _operation_text({"new_lines": ["a", "b"]}, "new_text") == "a\nb\n"


def test_a_string_body_still_works():
    assert _operation_text({"new_text": "x\n"}, "new_text") == "x\n"


def test_an_absent_field_is_absent_not_empty():
    """create must still be able to refuse a body it was never given."""
    assert _operation_text({}, "new_text") is None


def test_an_empty_line_list_is_an_empty_body_not_a_missing_one():
    assert _operation_text({"new_lines": []}, "new_text") == ""


def test_the_APPLIER_creates_a_file_from_lines(tmp_path):
    eng = _engine(tmp_path)
    eng._apply_operations([{
        "op": "create", "path": "made.py",
        "new_lines": ["def f():", "    return 1"],
    }])
    assert (tmp_path / "made.py").read_text() == "def f():\n    return 1\n"


def test_the_APPLIER_replaces_using_an_anchor_given_as_lines(tmp_path):
    target = tmp_path / "mod.py"
    target.write_text("def f():\n    return 0\n")
    eng = _engine(tmp_path)
    eng._apply_operations([{
        "op": "replace", "path": "mod.py",
        "old_lines": ["def f():", "    return 0"],
        "new_lines": ["def f():", "    return 1"],
    }])
    assert target.read_text() == "def f():\n    return 1\n"


def test_the_PREFLIGHT_reads_line_arrays_too(tmp_path, monkeypatch):
    """A preflight blind to new_lines would wave broken patches straight through."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mod.py").write_text("def f():\n    return 0\n")
    payload = json.dumps({"kind": "mutation", "operations": [{
        "op": "replace", "path": "mod.py",
        "old_lines": ["def f():", "    return 0"],
        "new_lines": ["def f():", "    return len(("],
    }]})
    message = _python_syntax_violation(payload)
    assert message is not None, "a broken line-array body passed the preflight"
    assert "would not compile" in message


def test_the_PREFLIGHT_still_catches_a_bad_anchor_given_as_lines(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mod.py").write_text("def f():\n    return 0\n")
    payload = json.dumps({"kind": "mutation", "operations": [{
        "op": "replace", "path": "mod.py",
        "old_lines": ["def nope():"], "new_lines": ["x = 1"],
    }]})
    message = _python_syntax_violation(payload)
    assert message is not None
    assert "does not appear" in message or "matches nothing" in message
