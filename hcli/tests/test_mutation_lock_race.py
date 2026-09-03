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

#: Trials per distribution.
#:
#: Was 80, chosen when a trial detected the race only sometimes. The persistent
#: racers start their two acquires within microseconds of each other instead of
#: relying on two ~50 ms interpreter startups to overlap, and detection is now
#: DETERMINISTIC: the advisory control reports double_acquire == trials on every
#: run, so a broken lock is caught by the first trial, not the eightieth.
#: Measured over five runs at 30 trials: 30/30 double-acquire on advisory,
#: 30/30 single-winner on O_EXCL, no run deviating.
RACE_TRIALS = 30

_RACER = r"""
# A PERSISTENT racer: one interpreter serves many trials, reading a workspace
# per trial from stdin.
#
# Spawning a fresh interpreter per racer meant 320 of them (80 trials, two
# racers, two tests), and each one imported hcli.resources before it could
# race. That cost 29.5 s of CPU -- 10.5 s of it in the kernel -- inside a file
# whose wall clock was 2.4 s, because twelve workers ran it in parallel. In a
# sharded suite that one file saturated the box and slowed every other shard.
#
# Nothing about the race weakens: still two real processes, still the real
# MutationLock, still 80 independent trials on their own workspaces. Only the
# interpreter startup is amortised.
import os, sys, time
sys.path.insert(0, sys.argv[1])
from hcli.resources import MutationLock, process_start_token

unit_id = sys.argv[2]


class Advisory(MutationLock):
    def acquire(self, unit_id):
        if not self.try_break_stale():
            return False
        # Make the intentionally unsafe check-then-write window deterministic
        # under full-suite scheduling. This is a test fixture for the
        # pre-O_EXCL algorithm, not a production delay.
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


for line in sys.stdin:
    line = line.rstrip("\n")
    if not line:
        break
    workspace, mode = line.split("\t")
    lock = Advisory(workspace) if mode == "advisory" else MutationLock(workspace)
    ok = lock.acquire(unit_id)

# RENDEZVOUS, not a sleep. The winner must still hold the lock while the rival
# makes its attempt -- a winner that exits first leaves a GENUINELY stale lock,
# the rival is right to break it, and the trial measures stale recovery instead
# of mutual exclusion.
#
# A fixed hold is a GUESS at how long the rival needs, and the guess is
# load-dependent: 0.4 s was wasteful on an idle box and 0.12 s was too short
# under the sharded runner, where it failed the O_EXCL test for exactly this
# reason. Waiting for the rival to announce it has finished attempting is the
# exact answer at any load, and it is both faster and harder to break.
    attempted = os.path.join(workspace, "attempted." + unit_id)
    with open(attempted, "w") as handle:
        handle.write("1" if ok else "0")

    rival = "b" if unit_id == "a" else "a"
    rival_path = os.path.join(workspace, "attempted." + rival)
    # Bounded: a rival that died must not hang the trial. Well above any
    # plausible acquire, and reached only when something has already gone wrong.
    deadline = time.time() + 5.0
    while time.time() < deadline and not os.path.exists(rival_path):
        time.sleep(0.002)

    # Reported only after the rendezvous, so the parent cannot start the next
    # trial while this one still holds the lock.
    sys.stdout.write("1" if ok else "0")
    sys.stdout.write("\n")
    sys.stdout.flush()
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


class _RacerPair:
    """Two long-lived racer processes that serve many trials.

    One pair belongs to exactly one worker thread and runs its trials in
    order, so two threads never interleave on the same pipes.
    """

    def __init__(self) -> None:
        cmd = [sys.executable, "-c", _RACER, str(REPO)]
        self.procs = [
            subprocess.Popen(
                cmd + [unit_id],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for unit_id in ("a", "b")
        ]

    def race(self, workspace: str, mode: str) -> tuple[bool, bool]:
        Path(workspace, ".hcli", "mutation.lock").unlink(missing_ok=True)
        line = f"{workspace}\t{mode}\n"
        # Both are already running and imported, so the two acquires start
        # within microseconds of each other. Spawning an interpreter per racer
        # put ~50 ms of startup between them and relied on the startups
        # overlapping; this is a tighter race, not a looser one.
        for proc in self.procs:
            proc.stdin.write(line)
            proc.stdin.flush()
        results = []
        for proc in self.procs:
            reply = proc.stdout.readline()
            if not reply:
                raise RuntimeError(f"racer died mid-trial: {proc.stderr.read()[:400]}")
            results.append(reply.strip() == "1")
        return results[0], results[1]

    def close(self) -> None:
        for proc in self.procs:
            try:
                proc.stdin.close()
            except OSError:
                pass
        for proc in self.procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()



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

    # Bounded: each pair is two live processes, so this is 2N children for the
    # whole distribution rather than 2N per trial.
    workers = max(2, min(4, (os.cpu_count() or 4)))
    chunks = [list(range(i, trials, workers)) for i in range(workers)]
    pairs = [_RacerPair() for _ in chunks if _]

    def _run_chunk(index: int) -> list[tuple[bool, bool]]:
        pair, out = pairs[index], []
        for _ in chunks[index]:
            with tempfile.TemporaryDirectory() as trial_tmp:
                out.append(pair.race(trial_tmp, mode))
        return out

    try:
        with ThreadPoolExecutor(max_workers=len(pairs)) as pool:
            for chunk in pool.map(_run_chunk, range(len(pairs))):
                for w1, w2 in chunk:
                    wins = int(w1) + int(w2)
                    if wins == 2:
                        double += 1
                    elif wins == 0:
                        none += 1
                    else:
                        single += 1
    finally:
        for pair in pairs:
            pair.close()

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
