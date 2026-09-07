"""A mutation must not be accepted on evidence that cannot cover it.

Measured 2026-09-05: HCLI mutated `qwen38_batched_prefill_allowed` in
crates/hawking-core/src/model/qwen38_hybrid_decode.rs, deleting the
`reuse == 0 && snapshot_at.is_none()` correctness guard, and the verifier
ACCEPTED it. Its evidence was:

    check: no_checker_available  crates/.../qwen38_hybrid_decode.rs
    check: test hcli/test_engine_tool_loop.py  exit 0

A Python test passes no matter what the Rust says. The mutated file was never
compiled, and the change it made produces "a faster wall and a different answer"
-- the exact outcome the deleted guard exists to prevent.

`no_checker_available` was silence, and silence was being read as a pass.
"""
from __future__ import annotations

from pathlib import Path

import hcli.engine as eng
from hcli.workspace import Workspace


def _kinds(result):
    return [c.get("kind") for c in result["checks"]]


def test_a_rust_path_is_cargo_checked_not_waved_through(tmp_path, monkeypatch):
    """The .rs branch must reach a real checker, and its exit code must bind."""
    seen = {}

    def fake_check(path, root):
        seen["path"] = Path(path).name
        return {"package": "hawking-core", "exit_code": 1,
                "stdout": "", "stderr": "error[E0425]: cannot find value"}

    monkeypatch.setattr(eng, "check_rust_file", fake_check)
    engine = eng.Engine(Workspace(str(tmp_path)))
    result = engine._validate([Path("crates/hawking-core/src/model/x.rs")],
                              tests=["hcli/test_engine_tool_loop.py"])

    assert seen.get("path") == "x.rs", "the .rs path never reached a checker"
    assert "cargo_check" in _kinds(result)
    assert result["ok"] is False, (
        "a Rust file that FAILED cargo check was still accepted; the compiler's "
        "verdict is not advisory"
    )


def test_a_passing_cargo_check_does_not_by_itself_block(tmp_path, monkeypatch):
    """Negative control: the gate must be the exit code, not the suffix."""
    monkeypatch.setattr(
        eng, "check_rust_file",
        lambda path, root: {"package": "hawking-core", "exit_code": 0,
                            "stdout": "", "stderr": ""},
    )
    engine = eng.Engine(Workspace(str(tmp_path)))
    result = engine._validate([Path("crates/hawking-core/src/model/x.rs")],
                              tests=["hcli/test_engine_tool_loop.py"])
    assert "cargo_check" in _kinds(result)
    # ok may still be False for unrelated reasons (the test list is not run here),
    # but the cargo_check record itself must not be the blocker.
    blocking = [c for c in result["checks"]
                if c.get("kind") == "cargo_check" and c.get("exit_code") != 0]
    assert not blocking


def test_uncheckable_source_fails_closed(tmp_path):
    """A .metal file has no checker here, so it cannot be accepted silently."""
    engine = eng.Engine(Workspace(str(tmp_path)))
    result = engine._validate([Path("crates/hawking-core/src/kernels/x.metal")],
                              tests=["hcli/test_engine_tool_loop.py"])
    record = [c for c in result["checks"] if c.get("kind") == "no_checker_available"]
    assert record, "expected a no_checker_available record for .metal"
    assert record[0].get("fatal") is True, (
        "uncheckable SOURCE was recorded as a non-event; that is how a Rust "
        "mutation got accepted on a Python test"
    )
    assert result["ok"] is False


def test_documentation_is_still_allowed_through(tmp_path):
    """Do not turn the fix into a blanket ban: prose has nothing to compile."""
    engine = eng.Engine(Workspace(str(tmp_path)))
    result = engine._validate([Path("receipts/runtime/NOTES.md")],
                              tests=["hcli/test_engine_tool_loop.py"])
    record = [c for c in result["checks"] if c.get("kind") == "no_checker_available"]
    assert record, "expected a no_checker_available record for .md"
    assert not record[0].get("fatal"), (
        "a markdown file was treated as uncheckable SOURCE; that would block "
        "every documentation mutation"
    )
