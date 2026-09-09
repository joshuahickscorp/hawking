"""Behavioural tests for the validation failure message.

The first version of this file asserted substrings in a window of engine.py
source. All three mutations of the code it claimed to cover PASSED, because the
mutated text still contained the strings being matched. These call the function.
"""
from hcli.engine import validation_failure_message as msg


def test_reason_appears_in_the_message():
    out = msg({"ok": False, "reason": "tests did not pass"})
    assert "tests did not pass" in out, out


def test_returncode_and_stderr_appear():
    out = msg({"ok": False, "returncode": 2, "stderr": "ImportError: no module foo"})
    assert "returncode=2" in out, out
    assert "ImportError: no module foo" in out, out


def test_bare_constant_is_never_the_whole_message_when_detail_exists():
    out = msg({"ok": False, "reason": "boom"})
    assert out != "Deterministic validation failed", "message lost its detail"
    assert out.startswith("Deterministic validation failed:"), out


def test_empty_validation_still_says_something_useful():
    out = msg({})
    assert "validation=" in out, out


def test_non_dict_validation_is_handled():
    assert "validation=False" in msg(False)


def test_empty_and_none_fields_are_skipped():
    out = msg({"reason": "r", "stderr": "", "failures": [], "command": None})
    assert "stderr" not in out and "failures" not in out and "command" not in out, out
    assert "reason=r" in out


def test_huge_stderr_is_truncated():
    out = msg({"stderr": "x" * 5000})
    assert len(out) < 600, f"unbounded output reached the message ({len(out)})"
    assert "x" * 300 in out, "truncated too aggressively to be useful"


def test_failing_checks_are_surfaced():
    out = msg({"ok": False, "checks": [
        {"kind": "test", "cmd": "pytest x", "reason": "TEST_FAILED", "exit_code": 1,
         "stderr": "E   assert 1 == 2"},
        {"kind": "py_compile", "path": "a.py", "exit_code": 0},
    ]})
    assert "TEST_FAILED" in out, out
    assert "assert 1 == 2" in out, out
    assert "py_compile" not in out, "a passing check should not be reported as failing"


def test_fatal_check_is_surfaced_even_without_a_reason():
    out = msg({"ok": False, "checks": [{"kind": "no_checker_available",
                                        "path": "k.metal", "fatal": True}]})
    assert "no_checker_available" in out and "k.metal" in out, out


def test_checks_block_is_bounded():
    out = msg({"ok": False, "checks": [
        {"kind": "test", "reason": "TEST_FAILED", "stderr": "y" * 9000}] * 40})
    assert len(out) < 1200, f"failing_checks was unbounded ({len(out)})"


def test_stderr_keeps_the_TAIL_not_the_head():
    # The outer cap alone would let a head-slice pass, and the head of a
    # traceback is the least informative part of it.
    out = msg({"ok": False, "checks": [
        {"kind": "test", "reason": "TEST_FAILED",
         "stderr": "HEAD" + "." * 4000 + "REAL_ERROR_HERE"}]})
    assert "REAL_ERROR_HERE" in out, "the informative tail of stderr was dropped"
    assert "HEAD" not in out, "kept the head instead of the tail"


def test_rejected_test_command_is_named():
    # NOT_ADMITTED without the command says a form was refused but not which,
    # which is what cost a full diagnostic cycle on the daemon path.
    out = msg({"ok": False, "checks": [
        {"kind": "test", "reason": "NOT_ADMITTED", "admitted": False,
         "requested": 'python -c "import probe; assert probe.VALUE == 7"'}]})
    assert "import probe" in out, f"the refused command is not in the message: {out}"


def test_pytest_failure_detail_reaches_the_message():
    # pytest writes the assertion and the traceback to STDOUT. The message kept
    # a tail of stderr only, so a real failing run reached the worker as
    # "TEST_FAILED exit_code 1" with nothing to diagnose: it re-ran the same
    # command instead of reading the failure, because it had never been shown one.
    stdout = (
        "============================= test session starts ====================\n"
        "collected 1 item\n\n"
        "hcli/tests/test_x.py F                                          [100%]\n\n"
        "=================================== FAILURES =========================\n"
        "______________ TestProjectNameEmpty.test_create_empty_name ___________\n"
        "hcli/tests/test_x.py:6: in test_create_empty_name\n"
        "    with self.assertRaises(ValueError):\n"
        "E   AssertionError: ValueError not raised\n"
        "=========================== short test summary info ==================\n"
        "FAILED hcli/tests/test_x.py::TestProjectNameEmpty::test_create_empty_name\n"
    )
    out = msg({"ok": False, "checks": [
        {"kind": "test", "reason": "TEST_FAILED", "exit_code": 1, "runner": "pytest",
         "requested": "python hcli/tests/test_x.py", "stdout": stdout, "stderr": ""}]})
    assert "AssertionError: ValueError not raised" in out, (
        f"the only line that says what to fix was dropped: {out}"
    )
    assert "test session starts" not in out, (
        "kept the banner instead of the failure"
    )


def test_the_actionable_check_is_reported_before_the_file_hashes():
    # files=[{sha256_before...sha256_after...}] is ~200 characters of hex that
    # nothing can act on, and it was emitted first -- so the downstream cut kept
    # the hashes and dropped the failure.
    out = msg({"ok": False,
               "files": [{"path": "a.py", "sha256_before": "0" * 64,
                          "sha256_after": "1" * 64, "changed": True}],
               "checks": [{"kind": "test", "reason": "TEST_FAILED", "exit_code": 1,
                           "stderr": "E   assert 1 == 2"}]})
    assert out.index("failing_checks") < out.index("files="), out
