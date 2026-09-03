"""Terminating a process that obeys SIGTERM must return at once.

`_terminate_pids` signals, then polls `process_alive` until a grace deadline.
`process_alive` is `os.kill(pid, 0)`, which SUCCEEDS on a zombie -- and a direct
child that obeys SIGTERM is a zombie about a millisecond after the signal, until
someone waits on it. The reap was called once, immediately after signalling,
before the child had died, so the poll never observed the death and every call
burned the whole grace: 2.0 s per pid, on processes that were already gone.

Not only a test-speed problem. `GrokBridge.cancel`, mission cancellation and
the supervisor's memory-pressure evacuation all route through here, so an
evacuation of four lanes waited eight seconds to notice four dead processes.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from hcli.grok_bridge import _terminate_pids, process_alive

GRACE = 2.0


def _obedient_child() -> subprocess.Popen:
    """A child that dies on SIGTERM and is never reaped by anyone else."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(3600)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(200):
        if process_alive(proc.pid):
            return proc
        time.sleep(0.01)
    return proc


def test_terminate_returns_far_inside_the_grace():
    proc = _obedient_child()
    started = time.monotonic()
    _terminate_pids([proc.pid], grace=GRACE)
    elapsed = time.monotonic() - started

    assert elapsed < GRACE / 2, (
        f"waited {elapsed:.2f}s of a {GRACE}s grace on a process that obeys "
        "SIGTERM -- the zombie is being read as alive"
    )
    assert not process_alive(proc.pid), "the child outlived terminate"


def test_the_child_is_reaped_not_merely_signalled():
    """A zombie left behind is the defect itself, one layer down."""
    proc = _obedient_child()
    _terminate_pids([proc.pid], grace=GRACE)
    try:
        pid, _status = os.waitpid(proc.pid, os.WNOHANG)
    except ChildProcessError:
        return  # already reaped: exactly what is wanted
    assert pid == 0, f"pid {proc.pid} was still an unreaped zombie after terminate"


def test_a_pid_that_refuses_SIGTERM_still_gets_the_full_grace():
    """Negative control: the fix must not turn the grace into no wait at all.

    A process that ignores SIGTERM has to be waited for and then killed. If
    this returned early too, the speed-up would just be a broken deadline.
    """
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
         "print('x', flush=True); time.sleep(3600)"],
        start_new_session=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    proc.stdout.readline()  # the handler is installed once this arrives
    started = time.monotonic()
    _terminate_pids([proc.pid], grace=0.3)
    elapsed = time.monotonic() - started

    assert elapsed >= 0.3, (
        f"returned in {elapsed:.2f}s without honouring a 0.3s grace on a "
        "process that ignores SIGTERM"
    )
    assert not process_alive(proc.pid), "SIGKILL did not follow the grace"


def test_terminating_nothing_is_free():
    started = time.monotonic()
    _terminate_pids([], grace=GRACE)
    _terminate_pids([0, os.getpid()], grace=GRACE)
    assert time.monotonic() - started < 0.05, "a no-op must not wait"
