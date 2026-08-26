#!/usr/bin/env python3
"""Crash-safe protected measurement window (G013 / directive §25).

GPU cleanliness overrides I/O overlap: contaminating transfers are SUSPENDED for the
protected window and resumed immediately after. The transfers are resumable by
construction (huggingface_hub file_download.py:1828 appends to the .incomplete file and
re-requests a byte Range), so suspending costs nothing but the wall time.

WHY THIS IS NOT JUST A finally BLOCK
------------------------------------
The previous implementation resumed inside `__exit__`. That survives an exception in the
with-body and nothing else. It was observed to fail for real: the parent shell was killed
on a timeout mid-window and SIGCONT never ran, leaving six downloader processes SIGSTOPped
indefinitely. A stopped download is strictly worse than an unpaused one, so the resume
cannot depend on the parent surviving.

Three independent guarantees, in order of who is still alive to act:

  1. normal exit          -- __exit__ resumes and clears the lease
  2. parent killed        -- a DETACHED watchdog notices within 1s and resumes
  3. watchdog also lost   -- the next run heals the stale lease on startup

The lease file is the shared state all three read.
"""
import json, os, signal, subprocess, sys, time
from pathlib import Path

LEASE = Path("/tmp/hawking_protected_window.lease")
DEFAULT_MAX_S = 1800


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _stopped(pid):
    """T = stopped. Only a stopped process needs resuming."""
    r = subprocess.run(["ps", "-o", "state=", "-p", str(pid)],
                       capture_output=True, text=True)
    return r.stdout.strip().startswith("T")


def _cont(pids):
    woke = []
    for pid in pids:
        if _alive(pid) and _stopped(pid):
            try:
                os.kill(pid, signal.SIGCONT)
                woke.append(pid)
            except ProcessLookupError:
                pass
    return woke


def heal(verbose=True):
    """Guarantee 3: resume anything a dead run left stopped."""
    if not LEASE.is_file():
        return {"healed": False, "reason": "no lease"}
    try:
        d = json.loads(LEASE.read_text())
    except Exception:
        LEASE.unlink(missing_ok=True)
        return {"healed": False, "reason": "unreadable lease, removed"}
    owner_dead = not _alive(d.get("owner_pid", -1))
    expired = time.time() > d.get("deadline", 0)
    if not (owner_dead or expired):
        return {"healed": False, "reason": "lease still held by a live owner"}
    woke = _cont(d.get("pids", []))
    LEASE.unlink(missing_ok=True)
    if verbose and woke:
        print(f"healed stale protected window: resumed {woke}", file=sys.stderr)
    return {"healed": True, "resumed": woke, "owner_dead": owner_dead,
            "expired": expired}


def _watchdog_body(deadline):
    """Guarantee 2. Runs detached; outlives the parent by construction.

    IT USED TO FIRE AT THE DEADLINE, NOT AT THE DEATH. This module exists because a
    parent was once killed on a timeout mid-window and SIGCONT never ran -- and the
    watchdog it shipped waited out the WHOLE lease before resuming, so with max_s=2400
    a parent killed at t=2s left the downloads stopped for forty minutes. OBSERVED FOR
    REAL AGAIN on 2026-08-25: a two-minute harness timeout killed the owner and left
    THIRTEEN of the operator's download processes in state T, and nothing but a human
    noticing brought them back. A STOPPED DOWNLOAD IS STRICTLY WORSE THAN AN UNPAUSED
    ONE, and that is the whole premise of the module, so the resume must track the
    OWNER'S LIFE and not only the clock.

    heal() ALREADY COMPUTED owner_dead -- the watchdog simply never asked until the
    deadline. Polling it takes resume latency from (deadline - death) to <= 1 s.
    """
    while time.time() < deadline:
        if not LEASE.is_file():
            return 0                      # parent finished cleanly
        try:
            owner = json.loads(LEASE.read_text()).get("owner_pid", -1)
        except Exception:                 # noqa: BLE001 -- unreadable lease: let heal decide
            owner = -1
        if owner != -1 and not _alive(owner):
            heal(verbose=False)           # the owner is gone: resume NOW, not at the deadline
            return 0
        time.sleep(1)
    heal(verbose=False)                   # deadline hit: resume regardless
    return 0


class ProtectedWindow:
    def __init__(self, pids, max_s=DEFAULT_MAX_S):
        self.pids = list(pids)
        self.max_s = max_s
        self.paused = []
        self.watchdog = None

    def __enter__(self):
        heal(verbose=False)               # never inherit someone else's stop
        deadline = time.time() + self.max_s
        # lease is written BEFORE the first SIGSTOP: a crash between the two must
        # leave a recoverable record, never a stopped process nobody knows about
        LEASE.write_text(json.dumps({"owner_pid": os.getpid(), "pids": self.pids,
                                     "deadline": deadline,
                                     "started": time.time()}))
        self.watchdog = subprocess.Popen(
            [sys.executable, __file__, "--watchdog", str(deadline)],
            start_new_session=True,       # detach: survives the parent's death
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for pid in self.pids:
            try:
                os.kill(pid, signal.SIGSTOP)
                self.paused.append(pid)
            except (ProcessLookupError, PermissionError):
                pass
        return self

    def __exit__(self, *exc):
        _cont(self.paused)                # guarantee 1
        LEASE.unlink(missing_ok=True)     # tells the watchdog to stand down
        if self.watchdog:
            try:
                self.watchdog.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.watchdog.kill()
        return False


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--watchdog":
        return _watchdog_body(float(sys.argv[2]))
    if len(sys.argv) > 1 and sys.argv[1] == "--heal":
        print(json.dumps(heal(), indent=1))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
