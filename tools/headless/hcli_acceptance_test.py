#!/usr/bin/env python3
"""Protected HCLI acceptance-integrity checks.

Plain python3 + assert. No pytest fixtures. Must also pass under pytest.

These checks drive the real Engine._validate / Engine._apply_operations
against temp workspaces. They exist to make the acceptance gate able to
fail: a pytest-idiom test file, a refused test, an empty tests list, a
no-op mutation, a green-on-green coincidence, and a non-Python target
must not look like success.

Run:
    python3 tools/headless/hcli_acceptance_test.py
    pytest tools/headless/hcli_acceptance_test.py -q
"""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.engine import Engine  # noqa: E402
from hcli.events import EventBus  # noqa: E402
from hcli.workspace import Workspace  # noqa: E402

WRONG_ADD = "def add(a, b):\n    return a * b - 999\n"
RIGHT_ADD = "def add(a, b):\n    return a + b\n"
TEST_ADD = (
    "from calc import add\n\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
)


def _engine(root: Path) -> Engine:
    return Engine(
        workspace=Workspace(str(root)),
        event_bus=EventBus(),
        runtime_count=1,
        model_name="/missing.gguf",
    )


def _ws_path(engine: Engine, name: str) -> Path:
    """Paths under Engine.root (resolved), so _validate relative_to() matches."""
    return engine.root / name


def check_v1_pytest_idiom_is_executed():
    """A pytest-idiom file must actually run; the wrong add() must not pass."""
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(Path(tmp))
        _ws_path(engine, "calc.py").write_text(WRONG_ADD, encoding="utf-8")
        _ws_path(engine, "test_calc.py").write_text(TEST_ADD, encoding="utf-8")
        validation = engine._validate(
            [_ws_path(engine, "calc.py"), _ws_path(engine, "test_calc.py")],
            ["test_calc.py"],
        )
        assert validation.get("ok") is False, (
            "V1: pytest-idiom test_calc.py against a wrong add() was "
            f"accepted: {validation!r}"
        )
        test_checks = [
            c
            for c in validation.get("checks", [])
            if c.get("kind") == "test"
        ]
        assert test_checks, f"V1: no test check recorded: {validation!r}"
        assert any(
            c.get("admitted") is True and c.get("exit_code") not in (0, None)
            for c in test_checks
        ), f"V1: admitted test did not fail: {test_checks!r}"


def check_v2_refused_test_fails_validation():
    """A test the admission logic refuses must fail validation, with a reason."""
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(Path(tmp))
        _ws_path(engine, "a.py").write_text("x = 1\n", encoding="utf-8")
        validation = engine._validate(
            [_ws_path(engine, "a.py")],
            ["make test"],
        )
        assert validation.get("ok") is False, (
            "V2: refused test left validation.ok True: "
            f"{validation!r}"
        )
        refused = [
            c
            for c in validation.get("checks", [])
            if c.get("kind") == "test" and c.get("admitted") is False
        ]
        assert refused, f"V2: refusal was not recorded: {validation!r}"
        assert any(
            c.get("reason") for c in refused
        ), f"V2: refusal has no reason: {refused!r}"


def check_v3_empty_tests_is_no_evidence():
    """A mutation with tests: [] must not validate as ok."""
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(Path(tmp))
        _ws_path(engine, "a.py").write_text("x = 1\n", encoding="utf-8")
        validation = engine._validate(
            [_ws_path(engine, "a.py")],
            [],
        )
        assert validation.get("ok") is False, (
            "V3: empty tests list validated as ok: "
            f"{validation!r}"
        )
        reason = str(validation.get("reason") or "")
        assert reason == "NO_EVIDENCE", (
            f"V3: expected reason NO_EVIDENCE, got {reason!r} "
            f"from {validation!r}"
        )


def check_v4_noop_mutation():
    """Identical replace is NO_OP_MUTATION; a real change is not."""
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(Path(tmp))
        _ws_path(engine, "a.py").write_text("x = 1\n", encoding="utf-8")
        raised = None
        try:
            engine._apply_operations(
                [
                    {
                        "op": "replace",
                        "path": "a.py",
                        "old_text": "x = 1",
                        "new_text": "x = 1",
                    }
                ]
            )
        except Exception as exc:
            raised = exc
        assert raised is not None, (
            "V4: replace with old_text == new_text was accepted"
        )
        assert "NO_OP_MUTATION" in str(raised), (
            f"V4: expected NO_OP_MUTATION, got {raised!r}"
        )

        _ws_path(engine, "b.py").write_text("x = 1\n", encoding="utf-8")
        try:
            engine._apply_operations(
                [
                    {
                        "op": "replace",
                        "path": "b.py",
                        "old_text": "x = 1",
                        "new_text": "x = 2",
                    }
                ]
            )
        except Exception as exc:
            assert "NO_OP_MUTATION" not in str(exc), (
                f"V4: real change flagged as no-op: {exc!r}"
            )
            raise
        assert "x = 2" in _ws_path(engine, "b.py").read_text(encoding="utf-8")


def check_v5_red_before_green():
    """A repair is red-before-green; a coincidence against an already-green test is not."""
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(Path(tmp))
        _ws_path(engine, "calc.py").write_text(WRONG_ADD, encoding="utf-8")
        _ws_path(engine, "test_calc.py").write_text(TEST_ADD, encoding="utf-8")
        engine._call_model = lambda prompt, evidence, compiled: {
            "kind": "mutation",
            "content": "fix add",
            "operations": [
                {
                    "op": "replace",
                    "path": "calc.py",
                    "old_text": "return a * b - 999",
                    "new_text": "return a + b",
                }
            ],
            "tests": ["test_calc.py"],
        }
        result = engine.execute("fix add so the test passes")
        receipt = json.loads(
            Path(result["receipt"]).read_text(encoding="utf-8")
        )
        validation = receipt.get("validation") or {}
        assert validation.get("red_before_green") is True, (
            "V5: genuine red-to-green repair did not set "
            f"red_before_green True: {validation!r}"
        )
        assert validation.get("ok") is True, (
            f"V5: repair should validate ok: {validation!r}"
        )
        assert result.get("status") == "completed", result

    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(Path(tmp))
        _ws_path(engine, "calc.py").write_text(RIGHT_ADD, encoding="utf-8")
        _ws_path(engine, "test_calc.py").write_text(TEST_ADD, encoding="utf-8")
        engine._call_model = lambda prompt, evidence, compiled: {
            "kind": "mutation",
            "content": "add a comment",
            "operations": [
                {
                    "op": "replace",
                    "path": "calc.py",
                    "old_text": "def add(a, b):",
                    "new_text": "def add(a, b):  # already green",
                }
            ],
            "tests": ["test_calc.py"],
        }
        result = engine.execute("tweak add")
        receipt = json.loads(
            Path(result["receipt"]).read_text(encoding="utf-8")
        )
        validation = receipt.get("validation") or {}
        assert validation.get("red_before_green") is False, (
            "V5: already-green test was reported as a repair: "
            f"{validation!r}"
        )
        assert result.get("status") == "completed", result
        assert validation.get("ok") is True, (
            "V5: red_before_green False must be recorded, not a hard "
            f"rejection: {validation!r}"
        )


def check_v6_no_checker_available():
    """A mutated non-Python file must leave a no_checker_available record."""
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(Path(tmp))
        _ws_path(engine, "notes.txt").write_text("hello\n", encoding="utf-8")
        engine._apply_operations(
            [
                {
                    "op": "replace",
                    "path": "notes.txt",
                    "old_text": "hello",
                    "new_text": "world",
                }
            ]
        )
        validation = engine._validate(
            [_ws_path(engine, "notes.txt")],
            [],
        )
        records = [
            c
            for c in validation.get("checks", [])
            if c.get("kind") == "no_checker_available"
        ]
        assert records, (
            "V6: non-Python mutation left no no_checker_available "
            f"record: {validation!r}"
        )
        assert any(
            str(c.get("path", "")).endswith("notes.txt") for c in records
        ), f"V6: record missing path: {records!r}"


def check_anti_vacuity_of_this_file():
    """This file must not be a vacuous gate, and HCLI must not run tests as scripts.

    Two claims:
    1. `_safe_test_argv` for a pytest-idiom file must invoke pytest, not
       `python <file>`.
    2. This test file itself has a `__main__` block and a pytest entry so
       both `python3` and `pytest` actually execute the checks.
    """
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(Path(tmp))
        _ws_path(engine, "calc.py").write_text(WRONG_ADD, encoding="utf-8")
        _ws_path(engine, "test_calc.py").write_text(TEST_ADD, encoding="utf-8")
        argv = engine._safe_test_argv("test_calc.py")
        assert argv is not None, "V7: pytest-idiom test was not admitted"
        joined = " ".join(argv)
        assert "-m" in argv and "pytest" in argv, (
            "V7: pytest-idiom file was not sent through python -m pytest: "
            f"{argv!r}"
        )
        assert joined.count("pytest") >= 1, argv

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    has_main = False
    has_pytest_entry = False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "test_hcli_acceptance":
            has_pytest_entry = True
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
            ):
                has_main = True
    assert has_main, "V7: this file has no __main__ block; python3 would be vacuous"
    assert has_pytest_entry, (
        "V7: this file has no test_hcli_acceptance; pytest would collect nothing"
    )
    assert len(CHECKS) >= 7, f"V7: expected 7 checks, found {len(CHECKS)}"


CHECKS = [
    ("v1_pytest_idiom_is_executed", check_v1_pytest_idiom_is_executed),
    ("v2_refused_test_fails_validation", check_v2_refused_test_fails_validation),
    ("v3_empty_tests_is_no_evidence", check_v3_empty_tests_is_no_evidence),
    ("v4_noop_mutation", check_v4_noop_mutation),
    ("v5_red_before_green", check_v5_red_before_green),
    ("v6_no_checker_available", check_v6_no_checker_available),
    ("anti_vacuity_of_this_file", check_anti_vacuity_of_this_file),
]


def main() -> int:
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"ok {name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
    return 1 if failed else 0


def test_hcli_acceptance():
    """pytest entry: the same checks as running this file directly."""
    rc = main()
    assert rc == 0


if __name__ == "__main__":
    sys.exit(main())
