"""MutationLock: exclusive create, stale recovery, two-process race.

The pre-change lock was check-then-``os.replace``. An 80-trial undelayed
two-process race against that algorithm produced 80/80 double-acquires
(the audit measured 79/80). ``O_CREAT|O_EXCL`` must produce exactly one
winner every trial.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from hcli.resources import MutationLock, process_start_token

RACE_TRIALS = 80

_RACER = r"""
import os, sys, time
sys.path.insert(0, sys.argv[1])
from hcli.resources import MutationLock, process_start_token

workspace, unit_id, mode = sys.argv[2], sys.argv[3], sys.argv[4]
if mode == "advisory":
    class Advisory(MutationLock):
        def acquire(self, unit_id):
            if not self.try_break_stale():
                return False
            # Make the intentionally unsafe check-then-write window
            # deterministic under full-suite scheduling. This is a test
            # fixture for the pre-O_EXCL algorithm, not a production delay.
            time.sleep(0.02)
            self.write(
                {
                    "pid": os.getpid(),
                    "start_time": process_start_token(os.getpid()),
                    "acquired_at": time.time(),
                    "unit_id": unit_id,
                }
            )
            return True
    lock = Advisory(workspace)
else:
    lock = MutationLock(workspace)
ok = lock.acquire(unit_id)
sys.stdout.write("1" if ok else "0")
sys.stdout.flush()
# Hold briefly before exiting. A winner that exits immediately leaves a
# GENUINELY stale lock, so the rival is right to break it and the trial
# measures stale recovery instead of mutual exclusion. Holding makes a second
# acquire a real double-acquire and nothing else.
#
# The hold only has to outlast the RIVAL'S ACQUIRE, which is try_break_stale
# plus a write -- and, on the advisory path, its deliberate 0.02 s window. That
# is 20-50 ms, so 0.12 s keeps a 3-6x margin. It was 0.4 s, which sat on the
# critical path of all 80 trials in both tests for no added safety.
time.sleep(0.12)
raise SystemExit(0 if ok else 1)
"""

_HOLDER = r"""
import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from hcli.resources import MutationLock
workspace = sys.argv[2]
hold = sys.argv[3] == "hold"
lock = MutationLock(workspace)
ok = lock.acquire("holder")
Path(workspace, "ready").write_text("1" if ok else "0", encoding="utf-8")
if hold:
    while True:
        time.sleep(0.2)
raise SystemExit(0)
"""


def _race_once(workspace: str, mode: str) -> tuple[bool, bool]:
    lock_path = Path(workspace) / ".hcli" / "mutation.lock"
    lock_path.unlink(missing_ok=True)
    cmd = [sys.executable, "-c", _RACER, str(REPO), workspace]
    p1 = subprocess.Popen(
        cmd + ["a", mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    p2 = subprocess.Popen(
        cmd + ["b", mode],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    o1, _e1 = p1.communicate()
    o2, _e2 = p2.communicate()
    return o1.strip() == "1", o2.strip() == "1"


def _race_distribution(mode: str, trials: int = RACE_TRIALS) -> dict:
    """`trials` independent races, run CONCURRENTLY.

    Each trial is two children contending for one lock; the trials do not
    contend with each other, so running them one at a time bought nothing but
    wall clock. Each racer sleeps 0.4 s to open the window, so 80 sequential
    trials cost ~38 s per test and the two tests were 77 s of a 433 s suite.

    Every trial gets its OWN workspace. Sequentially they shared one directory
    and so raced against the previous trial's leftovers; per-trial isolation is
    what makes them independent draws, which is what the counts already claimed
    they were.
    """
    double = single = none = 0

    def _trial(_index: int):
        with tempfile.TemporaryDirectory() as trial_tmp:
            _race_once(trial_tmp, mode)  # warm the path, discard
            return _race_once(trial_tmp, mode)

    # Bounded: each trial is two processes, so this is 2N live children.
    workers = max(2, min(12, (os.cpu_count() or 4)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for w1, w2 in pool.map(_trial, range(trials)):
            wins = int(w1) + int(w2)
            if wins == 2:
                double += 1
            elif wins == 0:
                none += 1
            else:
                single += 1
    return {
        "trials": trials,
        "single_winner": single,
        "double_acquire": double,
        "no_winner": none,
        "mode": mode,
    }


class TestMutationLockRace(unittest.TestCase):
    def test_advisory_check_then_replace_double_acquires(self):
        dist = _race_distribution("advisory", trials=RACE_TRIALS)
        print(
            "ADVISORY (check-then-replace) lock race: "
            f"trials={dist['trials']} single_winner={dist['single_winner']} "
            f"double_acquire={dist['double_acquire']} "
            f"no_winner={dist['no_winner']}"
        )
        self.assertEqual(dist["trials"], RACE_TRIALS)
        self.assertGreater(
            dist["double_acquire"],
            0,
            f"advisory baseline did not reproduce the race: {dist}",
        )

    def test_excl_lock_exactly_one_winner(self):
        dist = _race_distribution("excl", trials=RACE_TRIALS)
        print(
            "O_EXCL lock race: "
            f"trials={dist['trials']} single_winner={dist['single_winner']} "
            f"double_acquire={dist['double_acquire']} "
            f"no_winner={dist['no_winner']}"
        )
        self.assertEqual(dist["trials"], RACE_TRIALS)
        self.assertEqual(dist["double_acquire"], 0, dist)
        self.assertEqual(dist["no_winner"], 0, dist)
        self.assertEqual(dist["single_winner"], RACE_TRIALS, dist)


class TestMutationLockStaleHolder(unittest.TestCase):
    def test_live_holder_is_not_stealable(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp) / "ready"
            child = subprocess.Popen(
                [sys.executable, "-c", _HOLDER, str(REPO), tmp, "hold"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.time() + 10
                while time.time() < deadline and not ready.is_file():
                    if child.poll() is not None:
                        break
                    time.sleep(0.02)
                self.assertTrue(ready.is_file(), "holder never acquired")
                self.assertEqual(ready.read_text(encoding="utf-8"), "1")
                thief = MutationLock(tmp)
                self.assertFalse(thief.acquire("thief"))
                self.assertFalse(thief.try_break_stale())
                rec = thief.read()
                self.assertIsNotNone(rec)
                self.assertEqual(rec["pid"], child.pid)
            finally:
                child.kill()
                child.wait(timeout=5)

    def test_dead_holder_is_recoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            ready = Path(tmp) / "ready"
            child = subprocess.Popen(
                [sys.executable, "-c", _HOLDER, str(REPO), tmp, "exit"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            rc = child.wait(timeout=10)
            self.assertEqual(rc, 0)
            deadline = time.time() + 5
            while time.time() < deadline and not ready.is_file():
                time.sleep(0.02)
            self.assertTrue(ready.is_file())
            self.assertEqual(ready.read_text(encoding="utf-8"), "1")
            self.assertTrue((Path(tmp) / ".hcli" / "mutation.lock").is_file())
            lock = MutationLock(tmp)
            self.assertTrue(lock.try_break_stale())
            self.assertTrue(lock.acquire("after-death"))
            rec = lock.read()
            self.assertEqual(rec["pid"], os.getpid())
            self.assertEqual(rec["unit_id"], "after-death")
            lock.release("after-death")

    def test_planted_live_pid_is_not_broken(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = MutationLock(tmp)
            lock.write(
                {
                    "pid": os.getpid(),
                    "start_time": process_start_token(os.getpid()),
                    "acquired_at": time.time(),
                    "unit_id": "planted",
                }
            )
            self.assertFalse(lock.try_break_stale())
            self.assertFalse(lock.acquire("other"))


if __name__ == "__main__":
    unittest.main()
