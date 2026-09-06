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
