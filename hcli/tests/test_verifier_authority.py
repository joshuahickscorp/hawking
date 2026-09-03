"""Verifier authority: no vacuous gates, exact-enum verdicts, refusal is fail.

A pytest-idiom file run as ``python file.py`` used to exit 0 with zero
assertions. ``evaluate_python_test_file`` rejects that. Prefix verdicts
such as ``TRUE-ISH`` used to collapse to TRUE.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.verifier_pipeline import (
    Obligation,
    _normalize_verdict,
    ast_has_tests,
    command_is_admissible,
    evaluate_python_test_file,
    should_run_with_pytest,
    verify,
)


class FakeCaller:
    def __init__(self, handler):
        self.handler = handler

    def __call__(self, prompt: str, *, schema=None):
        return self.handler(prompt, schema)


def _ob() -> Obligation:
    return Obligation(
        id="o1",
        statement="foo holds",
        angles=["read foo.py"],
        consequential=False,
        agent_role="settle",
    )


ZERO_ASSERT_PYTEST = """def test_nothing():
    pass
"""

PYTEST_WITH_MAIN_THAT_PASSES_AS_SCRIPT = '''def test_boom():
    assert 1 == 0

if __name__ == "__main__":
    print("vacuous-pass")
'''

SCRIPT_WITH_ASSERT = """assert 1 == 1
"""

SCRIPT_ASSERT_NEVER_RUN = """def never():
    assert False

print("hello")
"""


class TestExactEnumVerdict(unittest.TestCase):
    def test_only_true_false_unverifiable(self):
        self.assertEqual(_normalize_verdict("TRUE"), "TRUE")
        self.assertEqual(_normalize_verdict("FALSE"), "FALSE")
        self.assertEqual(_normalize_verdict("UNVERIFIABLE"), "UNVERIFIABLE")
        self.assertEqual(_normalize_verdict({"verdict": "TRUE"}), "TRUE")

    def test_prefix_and_alias_are_not_true(self):
        self.assertEqual(_normalize_verdict("TRUE-ISH"), "UNVERIFIABLE")
        self.assertEqual(_normalize_verdict("VERIFIED"), "UNVERIFIABLE")
        self.assertEqual(_normalize_verdict("yes"), "UNVERIFIABLE")
        self.assertEqual(_normalize_verdict({"verdict": "REFUTED"}), "UNVERIFIABLE")
        self.assertEqual(_normalize_verdict("true enough"), "UNVERIFIABLE")


class TestCommandAdmission(unittest.TestCase):
    def test_true_colon_exit_zero_refused(self):
        for cmd in ("true", ":", "exit 0", "/bin/true", "/usr/bin/true"):
            ok, reason = command_is_admissible(cmd)
            self.assertFalse(ok, cmd)
            self.assertTrue(reason)

    def test_systemexit_zero_refused(self):
        ok, reason = command_is_admissible("python3 -c 'raise SystemExit(0)'")
        self.assertFalse(ok)
        self.assertEqual(reason, "VACUOUS_COMMAND")
        ok, reason = command_is_admissible(
            'python3 -c "import sys; sys.exit(0)"'
        )
        self.assertFalse(ok, reason)

    def test_sh_c_true_refused(self):
        ok, reason = command_is_admissible("sh -c true")
        self.assertFalse(ok)
        self.assertEqual(reason, "VACUOUS_COMMAND")

    def test_outside_allowlist_refused(self):
        ok, reason = command_is_admissible("curl https://example.invalid")
        self.assertFalse(ok)
        self.assertEqual(reason, "COMMAND_NOT_ADMITTED")

    def test_real_python_admitted(self):
        ok, reason = command_is_admissible("python3 -c 'print(1)'")
        self.assertTrue(ok, reason)


class TestRefusalErrorTimeoutAreNotPass(unittest.TestCase):
    def test_refused_command_is_false(self):
        def handler(prompt, schema):
            if "MECHANICALLY" in prompt:
                return {"command": "true"}
            return {"verdict": "TRUE"}

        v = verify(_ob(), "ev", FakeCaller(handler), lambda cmd: (0, "ok"))
        self.assertEqual(v.verdict, "FALSE")
        self.assertIn("VACUOUS", v.output)

    def test_errored_command_is_false(self):
        def handler(prompt, schema):
            if "MECHANICALLY" in prompt:
                return {"command": "python3 -c 'print(1)'"}
            return {"verdict": "TRUE"}

        def boom(cmd):
            raise OSError("backend refused")

        v = verify(_ob(), "ev", FakeCaller(handler), boom)
        self.assertEqual(v.verdict, "FALSE")
        self.assertIn("COMMAND_ERROR", v.output)
        self.assertNotEqual(v.verdict, "TRUE")

    def test_timed_out_command_is_false(self):
        def handler(prompt, schema):
            if "MECHANICALLY" in prompt:
                return {"command": "python3 -c 'print(1)'"}
            return {"verdict": "TRUE"}

        def hang(cmd):
            raise subprocess.TimeoutExpired(cmd, 0.1)

        v = verify(_ob(), "ev", FakeCaller(handler), hang)
        self.assertEqual(v.verdict, "FALSE")
        self.assertIn("TIMEOUT", v.output)


class TestVacuousTestFileRejected(unittest.TestCase):
    def test_zero_assertion_pytest_idiom_is_no_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_vacuous.py"
            path.write_text(ZERO_ASSERT_PYTEST, encoding="utf-8")
            as_script = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(
                as_script.returncode,
                0,
                "fixture must still exit 0 when run as a script "
                "(that is the hole we are closing)",
            )
            result = evaluate_python_test_file(path)
            self.assertFalse(result["ok"])
            self.assertEqual(result["reason"], "NO_EVIDENCE")
            self.assertEqual(result["assertion_count"], 0)
            self.assertTrue(result["forced_pytest"])
            self.assertEqual(result["runner"], "pytest")

    def test_pytest_idiom_without_main_is_not_a_script(self):
        self.assertTrue(should_run_with_pytest(ZERO_ASSERT_PYTEST))
        import ast

        tree = ast.parse(ZERO_ASSERT_PYTEST)
        self.assertTrue(ast_has_tests(tree))

    def test_pytest_idiom_with_main_is_forced_pytest_and_fails(self):
        """Used to pass: ``python file.py`` hits the __main__ print and exits 0.

        The file defines a failing test. evaluate must run pytest, not the
        script, so the fixture that used to pass now fails.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_boom.py"
            path.write_text(PYTEST_WITH_MAIN_THAT_PASSES_AS_SCRIPT, encoding="utf-8")
            as_script = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(as_script.returncode, 0, as_script.stdout)
            self.assertIn("vacuous-pass", as_script.stdout)
            self.assertTrue(should_run_with_pytest(path.read_text(encoding="utf-8")))
            result = evaluate_python_test_file(path)
            self.assertEqual(result["runner"], "pytest")
            self.assertTrue(result["forced_pytest"])
            self.assertFalse(result["ok"])
            self.assertIn(result["reason"], {"TEST_FAILED", "NO_EVIDENCE"})

    def test_script_with_unexecuted_assert_is_no_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hello.py"
            path.write_text(SCRIPT_ASSERT_NEVER_RUN, encoding="utf-8")
            result = evaluate_python_test_file(path)
            self.assertFalse(result["ok"])
            self.assertEqual(result["reason"], "NO_EVIDENCE")
            self.assertEqual(result["runner"], "script")
            self.assertEqual(result["assertion_count"], 0)

    def test_script_that_runs_an_assert_can_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.py"
            path.write_text(SCRIPT_WITH_ASSERT, encoding="utf-8")
            result = evaluate_python_test_file(path)
            self.assertTrue(result["ok"], result)
            self.assertGreaterEqual(result["assertion_count"], 1)


if __name__ == "__main__":
    unittest.main()
