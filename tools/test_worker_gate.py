"""Contract lifecycle step 10, "verify memory release", did not exist:
worker_gate.gate() only ever runs BEFORE a worker loads (grep confirms every
call site is pre-load), and the patient runner relies entirely on process
exit to reclaim wired memory -- nothing observes AFTER release, so a
specimen transition that fails to return memory was invisible.

These tests pin verify_release()/run_with_release_check(): a real process
boundary (spawn -> wait -> re-observe) whose wired-memory delta decides
RELEASED vs LEAK. Mutation check: flip `<=` to `<` in verify_release and
test_boundary_at_tolerance_is_released fails; flip the sign of wired_delta
and test_flags_a_specimen_that_does_not_release fails.
"""
from __future__ import annotations

import sys

from tools import worker_gate


def _obs(wired_gb: float) -> dict:
    # Only wired_gb matters to verify_release; the rest of gate()'s schema
    # is irrelevant here so keep the fixture minimal.
    return {"wired_gb": wired_gb}


def test_flags_a_specimen_that_does_not_release():
    before = _obs(4.63)
    after = _obs(20.22)  # one worker's worth (G131) never came back
    r = worker_gate.verify_release(before, after, tol_gb=2.0)
    assert r["decision"] == "LEAK"
    assert r["released"] is False
    assert r["wired_delta_gb"] == 15.59


def test_clean_release_within_tolerance():
    before = _obs(4.63)
    after = _obs(5.10)  # ordinary drift, worker actually exited
    r = worker_gate.verify_release(before, after, tol_gb=2.0)
    assert r["decision"] == "RELEASED"
    assert r["released"] is True


def test_boundary_at_tolerance_is_released():
    before = _obs(10.0)
    after = _obs(12.0)  # exactly at tol_gb=2.0
    r = worker_gate.verify_release(before, after, tol_gb=2.0)
    assert r["released"] is True, r
    over = worker_gate.verify_release(before, _obs(12.01), tol_gb=2.0)
    assert over["released"] is False, over


def test_run_with_release_check_wraps_a_real_process_boundary():
    # A real subprocess, a real wait(), a real pair of observe() calls --
    # not injected -- exercising the actual mechanism end to end.
    result = worker_gate.run_with_release_check(
        [sys.executable, "-c", "pass"], tol_gb=5.0,
    )
    assert result["returncode"] == 0
    for snap in (result["before"], result["after"]):
        for key in ("total_gb", "wired_gb", "free_gb", "compressed_gb",
                    "swap_used_mb"):
            assert key in snap
    assert result["release"]["decision"] in ("RELEASED", "LEAK")
    assert result["release"]["wired_delta_gb"] == round(
        result["after"]["wired_gb"] - result["before"]["wired_gb"], 2
    )


if __name__ == "__main__":
    test_flags_a_specimen_that_does_not_release()
    test_clean_release_within_tolerance()
    test_boundary_at_tolerance_is_released()
    test_run_with_release_check_wraps_a_real_process_boundary()
    print("test_worker_gate: ok")
