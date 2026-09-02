"""The recovery gate's fixture child must not outlive its parent.

`_fixture_resident_main` looped on `while True` under a comment calling it "a
long bounded loop". It was not bounded. Every gate run whose parent died or
timed out stranded a child at PPID 1 holding a temp directory; ten were alive on
this host at once, one per suite run, accumulating across sessions.

Two exits now: the parent going away, and a wall-clock ceiling.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from hcli.agentos.recovery import _FIXTURE_MAX_LIFETIME_S


def test_the_lifetime_ceiling_is_real_and_finite():
    assert 0 < _FIXTURE_MAX_LIFETIME_S < 3600


def test_the_fixture_exits_when_orphaned(tmp_path):
    """The load-bearing one. Start it under a parent that dies immediately."""
    # A shell that spawns the fixture and exits, so the fixture is reparented to
    # PID 1 the way a killed pytest run leaves it.
    launcher = (
        # stdout/stderr detached, or `sh` waits on the inherited pipe and this
        # test fails as a subprocess timeout instead of as the assertion below.
        f"{sys.executable} -m hcli.agentos.recovery --fixture-resident {tmp_path} "
        f">/dev/null 2>&1 & echo $!"
    )
    out = subprocess.run(
        ["/bin/sh", "-c", launcher],
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
        timeout=30,
    )
    pid = int(out.stdout.strip())

    deadline = time.time() + 25.0
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return  # exited on its own: the orphan check fired
        time.sleep(0.1)

    try:
        os.kill(pid, 9)
    except OSError:
        pass
    pytest.fail(
        f"fixture pid {pid} was still alive 25s after being orphaned; "
        "this is the leak that accumulated ten strays on this host"
    )
