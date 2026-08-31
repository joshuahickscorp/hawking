"""The resume is the dangerous part.

S025 §41: the user already decided pause-measure-resume for G075, so re-asking is
ceremony. §42 keeps the distinction that makes the lease worth having -
contaminated windows still run paired RELATIVE experiments, and quiescence is
spent only on absolutes that will be promoted.

A lease that stops five downloads and dies leaves them stopped. That is worse
than never running, so resume lives in a finally and success is not reported
until every stopped pid is confirmed running again.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from tools.future import protected_bench_lease as lease


def _sleeper():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def test_a_stopped_process_is_seen_as_stopped_and_resumed():
    p = _sleeper()
    try:
        os.kill(p.pid, signal.SIGSTOP)
        time.sleep(0.5)
        assert lease._stopped(p.pid), "SIGSTOP not observed"
        out = lease.release([p.pid])
        assert out["verified"] is True and out["still_stopped"] == []
        time.sleep(0.3)
        assert not lease._stopped(p.pid)
    finally:
        p.kill()
        p.wait(timeout=10)


def test_release_raises_rather_than_reporting_success_on_a_stuck_pid(monkeypatch):
    """Never swallow a failed resume."""
    monkeypatch.setattr(lease, "_alive", lambda pid: True)
    monkeypatch.setattr(lease, "_stopped", lambda pid: True)
    monkeypatch.setattr(lease.os, "kill", lambda *a, **k: None)
    monkeypatch.setattr(lease.time, "sleep", lambda *_: None)
    with pytest.raises(lease.ResumeFailed, match="still stopped"):
        lease.release([424242])


def test_measure_resumes_even_when_the_measurement_raises(monkeypatch):
    seen = {}
    monkeypatch.setattr(lease, "acquire", lambda pattern=None: {
        "pattern": "x", "pids": [1, 2], "stopped": [1, 2], "not_stopped": [],
        "loadavg_after_stop": 0.1, "quiescent": True, "quiescence_bar": 4.0,
    })

    def _release(pids):
        seen["released"] = list(pids)
        return {"resumed": list(pids), "still_stopped": [], "verified": True}

    monkeypatch.setattr(lease, "release", _release)

    def _boom():
        raise RuntimeError("the measurement exploded")

    out = lease.measure(_boom)
    assert seen["released"] == [1, 2], "a failed measurement must still resume"
    assert out["error"].endswith("the measurement exploded")
    assert out["evidence_class"] == "REFUSED", "a failed run is not an absolute"


def test_a_non_quiescent_window_refuses_the_absolute(monkeypatch):
    monkeypatch.setattr(lease, "acquire", lambda pattern=None: {
        "pattern": "x", "pids": [], "stopped": [], "not_stopped": [],
        "loadavg_after_stop": 9.9, "quiescent": False, "quiescence_bar": 4.0,
    })
    monkeypatch.setattr(lease, "release", lambda pids: {
        "resumed": [], "still_stopped": [], "verified": True})
    out = lease.measure(lambda: "a number")
    assert out["measured"] is None
    assert "not protected" in out["error"]
    assert out["evidence_class"] == "REFUSED"


def test_a_contaminated_window_still_permits_paired_ratios():
    """S025 §42. Do not stop all science because ModelLake is active."""
    c = lease.what_a_contaminated_window_may_still_do()
    assert c["allowed"] == lease.RELATIVE
    assert c["reserved_for_protected"] == lease.ABSOLUTE
    assert "601 GB/s" in c["why"] and "1.57x" in c["why"]


def test_the_policy_says_it_does_not_ask_each_time():
    p = lease.build()["policy"]
    assert p["resume_is_in_a_finally"] is True
    assert p["refuses_if_not_quiescent"] is True
    assert "ceremony" in p["does_not_ask_each_time"]


def test_the_lease_does_not_claim_more_than_eligibility():
    cb = lease.build()["claim_boundary"]
    assert "does not make a" in cb and "dirty-source build clean" in cb
