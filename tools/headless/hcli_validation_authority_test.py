#!/usr/bin/env python3
"""Protected HCLI validation-authority checks (audit B1–B6).

Drive the real Engine.execute with a stub model_client (no model process).

Run:
    pytest tools/headless/hcli_validation_authority_test.py -q
    python3 tools/headless/hcli_validation_authority_test.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.app import App  # noqa: E402
from hcli.engine import Engine, EngineError  # noqa: E402
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


class _StubController:
    def __init__(self, result):
        self._result = result

    def execute(self, prompt):
        return self._result

    def shutdown(self):
        pass


def _headless_exit(result) -> int:
    app = App.__new__(App)
    app.bus = EventBus()
    app.controller = _StubController(result)
    return app._run_headless("goal")


def check_b1_empty_tests_unverified_keeps_mutation():
    """tests: [] → status unverified, event is not validation_passed, disk kept."""
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(Path(tmp))
        (engine.root / "a.py").write_text("x = 1\n", encoding="utf-8")
        events: list[str] = []
        engine.event_bus.subscribe(lambda ev: events.append(ev.type))
        engine._call_model = lambda prompt, evidence, compiled: {
            "kind": "mutation",
            "content": "change x",
            "operations": [
                {
                    "op": "replace",
                    "path": "a.py",
                    "old_text": "x = 1",
                    "new_text": "x = 2",
                }
            ],
            "tests": [],
        }
        result = engine.execute("change a.py")
        assert result.get("status") == "unverified", result
        assert "validation_passed" not in events, events
        assert "validation_recorded" in events, events
        disk = (engine.root / "a.py").read_text(encoding="utf-8")
        assert disk.startswith("x = 2"), disk


def check_b2_headless_exit_consults_status():
    """App._run_headless is non-zero unless status == completed; 130 on cancel."""
    unverified_rc = _headless_exit(
        {
            "status": "unverified",
            "rolled_back": False,
            "content": "rewrote config.json",
            "error": "",
            "cancelled": False,
        }
    )
    completed_rc = _headless_exit(
        {
            "status": "completed",
            "rolled_back": False,
            "content": "ok",
            "error": "",
            "cancelled": False,
        }
    )
    cancelled_rc = _headless_exit(
        {
            "status": "cancelled",
            "cancelled": True,
            "content": "",
            "error": "Goal cancelled",
        }
    )
    assert unverified_rc != 0, f"B2: unverified exited {unverified_rc}"
    assert completed_rc == 0, f"B2: completed exited {completed_rc}"
    assert cancelled_rc == 130, f"B2: cancelled exited {cancelled_rc}"


def check_b3_all_skipped_is_no_evidence():
    """All-skipped pytest file → NO_EVIDENCE, not a pass."""
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(Path(tmp))
        (engine.root / "test_skip.py").write_text(
            "import pytest\n"
            "pytestmark = pytest.mark.skip(reason='module-level skip')\n"
            "\n"
            "def test_a():\n"
            "    assert False\n"
            "\n"
            "def test_b():\n"
            "    assert False\n",
            encoding="utf-8",
        )
        validation = engine._validate(
            [engine.root / "test_skip.py"],
            ["test_skip.py"],
        )
        assert validation.get("ok") is False, validation
        assert validation.get("reason") == "NO_EVIDENCE", validation
        checks = [
            c for c in validation.get("checks", []) if c.get("kind") == "test"
        ]
        assert checks, validation
        assert checks[0].get("reason") == "NO_EVIDENCE", checks
        assert int(checks[0].get("passed") or 0) == 0, checks
        assert engine._test_record_passed(checks[0]) is False


def check_b4_no_pre_records_is_none_not_true():
    """A pre-pass that produced no test records → red_before_green is None."""
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(Path(tmp))
        post = {
            "ok": True,
            "checks": [
                {
                    "kind": "test",
                    "admitted": True,
                    "runner": "pytest",
                    "exit_code": 0,
                    "collected": 3,
                    "passed": 3,
                }
            ],
        }
        rbg, reason = engine._compute_red_before_green(
            {
                "ok": False,
                "checks": [
                    {"kind": "py_compile", "path": "a.py", "exit_code": 1}
                ],
            },
            post,
        )
        assert rbg is None, (rbg, reason)
        assert rbg is not True
        assert reason == "pre_mutation_tests_did_not_run", reason
        rbg2, reason2 = engine._compute_red_before_green(
            {
                "ok": False,
                "reason": "pre_mutation_exception:TimeoutExpired",
                "checks": [],
            },
            post,
        )
        assert rbg2 is None, (rbg2, reason2)
        assert reason2 == "pre_mutation_tests_did_not_run", reason2


def check_b5_genuine_red_to_green_still_true():
    """A genuine red-to-green repair still yields red_before_green True."""
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(Path(tmp))
        (engine.root / "calc.py").write_text(WRONG_ADD, encoding="utf-8")
        (engine.root / "test_calc.py").write_text(TEST_ADD, encoding="utf-8")
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
        assert validation.get("red_before_green") is True, validation
        assert validation.get("red_before_green_advisory") is True, validation
        assert validation.get("ok") is True, validation
        assert result.get("status") == "completed", result
        assert (
            engine.root / "calc.py"
        ).read_text(encoding="utf-8") == RIGHT_ADD


def check_b6_create_requires_new_text():
    """create with `content` instead of `new_text` raises; no zero-byte file."""
    with tempfile.TemporaryDirectory() as tmp:
        engine = _engine(Path(tmp))
        raised = None
        try:
            engine._apply_operations(
                [
                    {
                        "op": "create",
                        "path": "new.py",
                        "content": "print(1)",
                    }
                ]
            )
        except EngineError as exc:
            raised = exc
        assert raised is not None, "B6: mistyped content key wrote a file"
        assert "new_text" in str(raised)
        assert not (engine.root / "new.py").exists()


CHECKS = [
    ("b1_empty_tests_unverified_keeps_mutation", check_b1_empty_tests_unverified_keeps_mutation),
    ("b2_headless_exit_consults_status", check_b2_headless_exit_consults_status),
    ("b3_all_skipped_is_no_evidence", check_b3_all_skipped_is_no_evidence),
    ("b4_no_pre_records_is_none_not_true", check_b4_no_pre_records_is_none_not_true),
    ("b5_genuine_red_to_green_still_true", check_b5_genuine_red_to_green_still_true),
    ("b6_create_requires_new_text", check_b6_create_requires_new_text),
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


def test_b1_empty_tests_unverified_keeps_mutation():
    check_b1_empty_tests_unverified_keeps_mutation()


def test_b2_headless_exit_consults_status():
    check_b2_headless_exit_consults_status()


def test_b3_all_skipped_is_no_evidence():
    check_b3_all_skipped_is_no_evidence()


def test_b4_no_pre_records_is_none_not_true():
    check_b4_no_pre_records_is_none_not_true()


def test_b5_genuine_red_to_green_still_true():
    check_b5_genuine_red_to_green_still_true()


def test_b6_create_requires_new_text():
    check_b6_create_requires_new_text()


if __name__ == "__main__":
    sys.exit(main())
