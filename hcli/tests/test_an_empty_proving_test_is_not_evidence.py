"""A named proving test with no tests in it must not be accepted.

Measured 2026-09-05. HCLI named tools/sovereign/test_g002_attribution.py as its
proving test and wrote a file containing ONLY a docstring. The mutation was
ACCEPTED:

    runner: 'script'   exit_code: 0   collected: None   passed: None   reason: None

Its two previous attempts had written REAL tests and were correctly rejected:

    runner: 'pytest'   exit_code: 1   collected: 1   passed: 0   reason: TEST_FAILED

So the empty file was routed to the one runner that cannot notice it is empty.
_admit_test chose the runner from CONTENT (_file_is_pytest_idiom); a file with no
test functions is not "pytest idiom", so it fell through to `python file.py`,
which exits 0 on a docstring.

_pytest_evidence already scores rc 5 / collected 0 as NO_EVIDENCE. The hole was
never the scoring, it was never reaching the scorer.
"""
from __future__ import annotations

from pathlib import Path

from hcli.engine import Engine, _looks_like_a_test_filename
from hcli.workspace import Workspace


def test_a_test_named_file_is_recognised_by_name():
    assert _looks_like_a_test_filename(Path("test_x.py"))
    assert _looks_like_a_test_filename(Path("x_test.py"))
    assert not _looks_like_a_test_filename(Path("helper.py")), (
        "a non-test filename must not be forced onto pytest"
    )


def test_an_empty_test_file_is_run_under_pytest_not_as_a_script(tmp_path):
    """The load-bearing routing decision."""
    (tmp_path / "test_nothing_at_all.py").write_text(
        '"""Proving test."""\n', encoding="utf-8")
    engine = Engine(Workspace(str(tmp_path)))
    # workspace-relative: _admit_test refuses absolute paths by design
    admitted = engine._admit_test("test_nothing_at_all.py")
    assert admitted.get("admitted") is True
    assert admitted.get("runner") == "pytest", (
        "a file that names itself a test fell through to the script runner, which "
        "scores on exit code alone and passes on a docstring"
    )


def test_pytest_scoring_calls_an_empty_run_no_evidence(tmp_path):
    """The scorer half: rc 5 / zero collected is NO_EVIDENCE, not a pass."""
    engine = Engine(Workspace(str(tmp_path)))
    verdict = engine._pytest_evidence("no tests ran in 0.01s", "", 5)
    assert verdict["reason"] == "NO_EVIDENCE"
    assert verdict["passed"] == 0


def test_a_real_passing_run_is_still_evidence(tmp_path):
    """Negative control: the fix must not reject genuine evidence."""
    engine = Engine(Workspace(str(tmp_path)))
    verdict = engine._pytest_evidence("collected 3 items\n\n3 passed in 0.10s", "", 0)
    assert verdict["reason"] is None
    assert verdict["passed"] == 3
    assert verdict["collected"] == 3
