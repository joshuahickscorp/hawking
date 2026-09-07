"""Counting accepted work by the ok flag scores fabrications as successes.

Measured 2026-09-06: five autonomous rounds returned answers claiming a function
was "already present ... the test passes as expected". The name existed nowhere.
Every one was recorded validation {ok: True, kind: read_only}, and a polling line
reading only ok reported five successes.
"""
from hcli.engine import is_accepted_work as acc


def test_the_exact_fabricated_receipt_is_not_accepted_work():
    assert acc({"ok": True, "kind": "read_only", "evidence": "none",
                "accepted_work": False}) is False


def test_a_bare_ok_true_is_not_enough():
    assert acc({"ok": True}) is False, "ok alone must not count as accepted work"


def test_a_verified_mutation_is_accepted_work():
    assert acc({"ok": True, "checks": [
        {"kind": "py_compile", "exit_code": 0},
        {"kind": "test", "exit_code": 0, "admitted": True}]}) is True


def test_a_mutation_whose_test_failed_is_not():
    assert acc({"ok": True, "checks": [{"kind": "test", "exit_code": 1}]}) is False


def test_a_mutation_with_only_a_compile_check_is_not():
    # py_compile passing is not evidence the change is correct.
    assert acc({"ok": True, "checks": [{"kind": "py_compile", "exit_code": 0}]}) is False


def test_ok_false_is_never_accepted():
    assert acc({"ok": False, "checks": [{"kind": "test", "exit_code": 0}]}) is False


def test_non_dict_is_never_accepted():
    assert acc(True) is False and acc(None) is False


def test_read_only_is_refused_even_if_it_somehow_carries_a_passing_test():
    # Defence in depth: if a future change ever attaches checks to an answer,
    # the read_only kind alone must still disqualify it. Without this fixture
    # the clause is masked by the missing-checks path and mutates green.
    assert acc({"ok": True, "kind": "read_only",
                "checks": [{"kind": "test", "exit_code": 0}]}) is False


def test_accepted_work_false_is_refused_even_with_a_passing_test():
    assert acc({"ok": True, "accepted_work": False,
                "checks": [{"kind": "test", "exit_code": 0}]}) is False
