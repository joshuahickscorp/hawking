"""The F12 detector must flag work that `diff.patch` structurally cannot see.

`visionmcp/**` is untracked, so `git diff` never captures changes there and the
lane's diff.patch comes back EMPTY for that work -- one lane produced 218 lines
existing only in its worktree. If the watcher misses that, cleanup destroys it.

This test exists because the detector was, at one point, decorative: the logic
was correct but `needs` filtered to lanes whose status file read exactly "done".
A lane killed or crashed before writing a status file has an EMPTY status --
and those are the likeliest to be reaped with unharvested work in them. A
planted worktree-only file in such a lane went unflagged. Watched it fail at 0
flagged, then pass at 1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lane_watch


def _lane(tmp: Path, task_id: str, status: str, worktree_file: str | None):
    (tmp / "tasks" / task_id).mkdir(parents=True, exist_ok=True)
    (tmp / "tasks" / task_id / "status").write_text(status)
    (tmp / "tasks" / task_id / "diff.patch").write_text("")
    (tmp / "tasks" / task_id / "metadata.json").write_text(
        '{"repo": "%s", "started_at": "2026-08-23T00:00:00Z"}' % lane_watch.REPO
    )
    if worktree_file:
        f = tmp / "worktrees" / task_id / worktree_file
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("# only in the worktree\n")


def _flagged(tmp: Path, task_id: str) -> int:
    lane_watch.TASKS = tmp / "tasks"
    lane_watch.WORKTREES = tmp / "worktrees"
    return lane_watch._worktree_extras(task_id, []) or 0


def test_untracked_worktree_file_is_flagged_even_with_empty_status(tmp_path):
    _lane(tmp_path, "crashed-20260823-000000", "", "visionmcp/src/probe.py")
    assert _flagged(tmp_path, "crashed-20260823-000000") == 1, (
        "a lane with no status file still holds real work and must be flagged"
    )


def test_clean_worktree_is_not_flagged(tmp_path):
    _lane(tmp_path, "clean-20260823-000000", "done", None)
    assert _flagged(tmp_path, "clean-20260823-000000") == 0


def test_file_already_in_the_patch_is_not_reflagged(tmp_path):
    _lane(tmp_path, "harvested-20260823-000000", "done", "visionmcp/src/probe.py")
    lane_watch.TASKS = tmp_path / "tasks"
    lane_watch.WORKTREES = tmp_path / "worktrees"
    n = lane_watch._worktree_extras("harvested-20260823-000000", ["visionmcp/src/probe.py"])
    assert (n or 0) == 0, "a file the patch already carries is not worktree-only"


if __name__ == "__main__":
    import tempfile
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory() as d:
                fn(Path(d))
            print(f"ok  {name}")
    print("3/3 passed")
