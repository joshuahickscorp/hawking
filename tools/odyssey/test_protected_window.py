"""G013 pins. The bug these exist for: a protected window resumed inside __exit__,
the parent shell was killed on timeout, and six downloaders were left SIGSTOPped."""
import json, os, signal, subprocess, sys, time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from protected_window import LEASE, ProtectedWindow, heal


def _victim():
    return subprocess.Popen(["sleep", "300"])


def _state(pid):
    r = subprocess.run(["ps", "-o", "state=", "-p", str(pid)],
                       capture_output=True, text=True)
    return r.stdout.strip()


def _wait_state(pid, want, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        if _state(pid).startswith(want):
            return True
        time.sleep(0.2)
    return False


@pytest.fixture(autouse=True)
def _clean():
    LEASE.unlink(missing_ok=True)
    yield
    LEASE.unlink(missing_ok=True)


def test_guarantee_1_normal_exit_pauses_then_resumes():
    v = _victim()
    try:
        with ProtectedWindow([v.pid], max_s=60):
            assert _wait_state(v.pid, "T"), "victim was never stopped"
        assert _wait_state(v.pid, "S"), "victim was not resumed on normal exit"
        assert not LEASE.exists()
    finally:
        v.kill(); v.wait()


def test_guarantee_1_resumes_even_when_the_body_raises():
    v = _victim()
    try:
        with pytest.raises(RuntimeError):
            with ProtectedWindow([v.pid], max_s=60):
                assert _wait_state(v.pid, "T")
                raise RuntimeError("boom")
        assert _wait_state(v.pid, "S")
    finally:
        v.kill(); v.wait()


def test_guarantee_2_watchdog_resumes_when_the_parent_is_KILLED():
    """THE REGRESSION. Parent dies mid-window; nothing in it can run.
    Only a detached watchdog can save the victim."""
    v = _victim()
    try:
        # a real child process opens a window and then hangs forever
        code = (
            "import sys,time;sys.path.insert(0,%r);"
            "from protected_window import ProtectedWindow;"
            "w=ProtectedWindow([%d],max_s=6);w.__enter__();"
            "print('in',flush=True);time.sleep(600)"
            % (str(Path(__file__).resolve().parent), v.pid))
        parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
        assert parent.stdout.readline().strip() == b"in"
        assert _wait_state(v.pid, "T"), "victim was never stopped"

        parent.kill()          # SIGKILL: no finally, no __exit__, no atexit
        parent.wait()
        assert _state(v.pid).startswith("T"), "precondition: still stopped after the kill"

        # only the detached watchdog remains to act, at the 6s deadline
        assert _wait_state(v.pid, "S", timeout=45), \
            "victim left STOPPED after the parent was killed -- the original bug"
    finally:
        v.kill(); v.wait()


def test_guarantee_3_heal_recovers_a_lease_whose_owner_is_gone():
    v = _victim()
    try:
        os.kill(v.pid, signal.SIGSTOP)
        assert _wait_state(v.pid, "T")
        LEASE.write_text(json.dumps({"owner_pid": 999999,   # long dead
                                     "pids": [v.pid],
                                     "deadline": time.time() + 9999,
                                     "started": time.time()}))
        r = heal(verbose=False)
        assert r["healed"] is True and v.pid in r["resumed"]
        assert _wait_state(v.pid, "S")
        assert not LEASE.exists()
    finally:
        v.kill(); v.wait()


def test_heal_does_not_steal_a_live_window():
    """Healing must not resume a window somebody is legitimately holding."""
    v = _victim()
    try:
        with ProtectedWindow([v.pid], max_s=60):
            assert _wait_state(v.pid, "T")
            r = heal(verbose=False)
            assert r["healed"] is False
            assert _state(v.pid).startswith("T"), "heal stole a live window"
    finally:
        v.kill(); v.wait()


def test_lease_is_written_before_the_first_sigstop():
    """A crash between the two must leave a record, not an orphaned stop."""
    src = (Path(__file__).resolve().parent / "protected_window.py").read_text()
    body = src[src.index("def __enter__"):src.index("def __exit__")]
    assert body.index("LEASE.write_text") < body.index("signal.SIGSTOP")


def test_entering_a_window_heals_a_stale_one_first():
    v1, v2 = _victim(), _victim()
    try:
        os.kill(v1.pid, signal.SIGSTOP)
        LEASE.write_text(json.dumps({"owner_pid": 999999, "pids": [v1.pid],
                                     "deadline": time.time() + 9999,
                                     "started": time.time()}))
        with ProtectedWindow([v2.pid], max_s=60):
            assert _wait_state(v1.pid, "S"), "stale stop survived a new window"
    finally:
        for v in (v1, v2):
            v.kill(); v.wait()


def test_the_watchdog_resumes_on_OWNER_DEATH_not_only_at_the_deadline(tmp_path,
                                                                      monkeypatch):
    """OBSERVED FOR REAL: a harness timeout killed the owner two seconds into a
    max_s=2400 window and left thirteen of the operator's downloads in state T. The
    watchdog would have waited FORTY MINUTES, because it polled only the clock.

    heal() already computed owner_dead; the watchdog never asked. This pins that it
    does, by writing a lease owned by a PID that cannot exist and requiring the
    watchdog to return promptly rather than sitting out a far-future deadline.
    """
    import json as _json, time as _time
    import protected_window as pw

    lease = tmp_path / "lease.json"
    monkeypatch.setattr(pw, "LEASE", lease)
    dead_pid = 2 ** 22                      # above any real pid on this machine
    assert not pw._alive(dead_pid)
    lease.write_text(_json.dumps({"owner_pid": dead_pid, "pids": [],
                                  "deadline": _time.time() + 3600}))
    t0 = _time.time()
    assert pw._watchdog_body(_time.time() + 3600) == 0
    assert _time.time() - t0 < 5            # NOT the 3600 s deadline
    assert not lease.exists()               # healed and cleared


def test_the_watchdog_still_waits_out_a_LIVE_owner(tmp_path, monkeypatch):
    """The other direction: a live owner mid-window must NOT be healed out from under.
    A check that only ever resumes is as useless as one that never does."""
    import json as _json, time as _time, os as _os
    import protected_window as pw

    lease = tmp_path / "lease.json"
    monkeypatch.setattr(pw, "LEASE", lease)
    lease.write_text(_json.dumps({"owner_pid": _os.getpid(), "pids": [],
                                  "deadline": _time.time() + 3600}))
    pw._watchdog_body(_time.time() + 1.5)   # short deadline so the test ends
    # MY FIRST ASSERTION HERE WAS WRONG AND THE CODE WAS RIGHT: I expected the lease
    # cleared, but heal() honours the LEASE's deadline (+3600), not the watchdog loop's
    # argument, so a live owner mid-window is correctly left alone. That is the property
    # worth pinning -- a window must not be healed out from under its owner.
    assert lease.exists()
    assert pw.heal(verbose=False)["reason"] == "lease still held by a live owner"
