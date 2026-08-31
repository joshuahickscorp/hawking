"""GpuPark must speak the lane lock's protocol, not a second one.

The lane lock is mkdir-atomic: a DIRECTORY with pid/owner files inside, which is
what tools/gpu_lane_lock.sh and every other holder in this repo uses. GpuPark
used touch() + flock on the same path. Two incompatible protocols on one path,
so whichever ran first broke the other - shell first gave the park [Errno 21] Is
a directory (the 30m run hit exactly this), and park first created the 0-byte
regular file that mkdir can never succeed against, which is the wedge seen at
05:58 and cleared at 06:21.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tools.future import model_bearing_torture as mbt


def test_acquire_creates_a_directory_not_a_file(tmp_path):
    lock = tmp_path / "lane.lock"
    rec = mbt._mkdir_lock_acquire(lock, owner="test")
    try:
        assert lock.is_dir(), "a regular file here is the wedge, not a lock"
        assert (lock / "pid").read_text().strip() == str(os.getpid())
        assert (lock / "owner").read_text().strip() == "test"
        assert rec["protocol"] == "mkdir" and rec["parked"] is False
    finally:
        mbt._mkdir_lock_release(lock)
    assert not lock.exists()


def test_a_wedge_is_reclaimed(tmp_path):
    lock = tmp_path / "lane.lock"
    lock.write_bytes(b"")  # the 0-byte wedge
    mbt._mkdir_lock_acquire(lock, owner="test")
    try:
        assert lock.is_dir()
    finally:
        mbt._mkdir_lock_release(lock)


def test_a_dead_owner_is_reclaimed(tmp_path):
    lock = tmp_path / "lane.lock"
    lock.mkdir()
    (lock / "pid").write_text("999999")   # no such process
    (lock / "owner").write_text("ghost")
    mbt._mkdir_lock_acquire(lock, owner="test")
    try:
        assert (lock / "owner").read_text().strip() == "test"
    finally:
        mbt._mkdir_lock_release(lock)


def test_a_live_holder_is_not_stolen(tmp_path):
    """A LIVE STRANGER, not us. os.getpid() would now read as already-held,
    which is right for our own launcher and wrong for a rival lane - so the
    holder here has to be a real unrelated process."""
    import subprocess

    rival = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        lock = tmp_path / "lane.lock"
        lock.mkdir()
        (lock / "pid").write_text(str(rival.pid))
        (lock / "owner").write_text("someone-else")
        with pytest.raises(TimeoutError):
            mbt._mkdir_lock_acquire(lock, owner="test", timeout_s=0.1)
        assert (lock / "owner").read_text().strip() == "someone-else"
    finally:
        rival.kill()
        rival.wait(timeout=10)


def test_release_never_removes_someone_elses_lock(tmp_path):
    lock = tmp_path / "lane.lock"
    lock.mkdir()
    (lock / "pid").write_text("999999")
    (lock / "owner").write_text("someone-else")
    mbt._mkdir_lock_release(lock)
    assert lock.is_dir(), "released a lock this process never held"


def test_the_production_lane_lock_uses_the_mkdir_protocol():
    assert str(mbt.GPU_LOCK) in mbt.MKDIR_PROTOCOL_LOCKS


def test_a_lock_held_by_our_own_launcher_is_not_contention(tmp_path):
    """The trial runs UNDER gpu_lane_lock.sh, so the wrapper holds the lock.

    Reading that pid and finding it alive - of course it is, it is the parent -
    and then parking for the full deadline is a deadlock on a lock our own
    launcher took on our behalf. It cost a 45-minute run that burned 0.78 s of
    CPU. Worth naming: the old flock code FAILED OPEN here with [Errno 21], so
    making the park correct turned a benign failure into a hang.
    """
    lock = tmp_path / "lane.lock"
    lock.mkdir()
    (lock / "pid").write_text(str(os.getpid()))  # stand in for the wrapper
    (lock / "owner").write_text("gpu_lane_lock.sh")
    rec = mbt._mkdir_lock_acquire(lock, owner="trial", timeout_s=2.0)
    assert rec["parked"] is False
    assert rec["already_held_by_ancestor"] == os.getpid()
    assert rec["release_is_not_ours"] is True
    # and the owner must be untouched - we did not take it, so we do not stamp it
    assert (lock / "owner").read_text().strip() == "gpu_lane_lock.sh"


def test_release_leaves_an_ancestors_lock_alone(tmp_path):
    """The wrapper's EXIT trap owns it; tearing it down here would let that trap
    delete a lock a LATER lane had legitimately taken."""
    lock = tmp_path / "lane.lock"
    lock.mkdir()
    (lock / "pid").write_text("999999")
    (lock / "owner").write_text("gpu_lane_lock.sh")
    mbt._mkdir_lock_release(lock)
    assert lock.is_dir()


def test_ancestors_include_this_process():
    assert os.getpid() in mbt._ancestor_pids()
