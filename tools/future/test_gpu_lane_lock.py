"""The GPU lane lock must reclaim a non-directory at its path.

A 0-byte regular file appeared at /tmp/hawking-gpu-lane.lock twice on this box:
at 05:58 (cleared 06:21) and again at 08:41. mkdir can never succeed against an
existing regular file, and the stale-owner branch tests "$LOCK/pid" - which is
unreachable when $LOCK is a file - so the rm -rf that clears a dead owner could
never fire. Every caller spun to its 5400-second deadline and exited 75, meaning
GPU lanes were either timing out or running UNSERIALISED, and any absolute GB/s
measured in such a window has no serialisation guarantee.

Nothing in this repo creates it as a file; every reader and writer here uses
mkdir / create_dir / [ -d ]. Rather than keep hunting the creator, the lock
reclaims the state.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SH = REPO / "tools" / "future" / "test_gpu_lane_lock_wedge.sh"


def test_wedge_is_reclaimed_live_holder_is_respected():
    r = subprocess.run([str(SH)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "wedge reclaimed" in r.stdout


def test_the_script_still_uses_a_directory_lock():
    """The fix must not quietly turn the lock into a file lock, which would make
    two lanes think they both hold it."""
    src = (REPO / "tools" / "gpu_lane_lock.sh").read_text()
    assert 'mkdir "$LOCK"' in src, "acquisition must stay mkdir-atomic"
    assert 'rm -f "$LOCK"' in src, "the wedge branch must remove the file"
    assert '[ ! -d "$LOCK" ]' in src, "the wedge branch must test for non-directory"


def test_wedge_test_never_touches_the_production_lock():
    """The test that plants a wedge must not plant it where lanes live.

    It rm -rf's its lock path and then creates a 0-byte file there. Pointed at
    /tmp/hawking-gpu-lane.lock that is two hazards at once: it destroys a live
    lane's lock, and an interrupted run leaves behind the exact wedge it exists
    to reclaim. The 08:41 wedge is accounted for by this. Executable lines must
    name only $HAWKING_GPU_LANE_LOCK; the header comment may cite the real path.
    """
    src = (Path(__file__).parent / "test_gpu_lane_lock_wedge.sh").read_text(encoding="utf-8")
    code = [
        ln for ln in src.splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    offenders = [ln for ln in code if "/tmp/hawking-gpu-lane.lock" in ln]
    assert not offenders, f"wedge test drives the production lock: {offenders}"
    assert 'export HAWKING_GPU_LANE_LOCK=' in src


def test_lock_script_honours_the_override():
    """Without the override the wedge test cannot avoid the production path."""
    src = (Path(__file__).parents[2] / "tools" / "gpu_lane_lock.sh").read_text(encoding="utf-8")
    assert "LOCK=${HAWKING_GPU_LANE_LOCK:-/tmp/hawking-gpu-lane.lock}" in src
