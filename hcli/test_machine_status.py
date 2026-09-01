"""/status reports the machine, not just this session, and never guesses.

The defect: a fresh CLI session printed ``health=down resident=0`` on a host
that was running a resident supervisor and three ModelLake downloads, because
every field came from the session's own controller.

The rule these tests enforce is that each added value comes from a live
producer -- the resident's own ``state.json``, the ModelLake watcher's lock
file and JSONL -- and that a value which cannot be read prints ``unknown``
rather than a remembered number. In particular a pid that merely exists is
never enough to call a supervisor live.

Runnable two ways:

    python3 -m pytest hcli/test_machine_status.py -q
    python3 hcli/test_machine_status.py
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from hcli.commands import (
    MODELLAKE_TAIL_BYTES,
    RESIDENT_STATE_REL,
    STATUS_LINE_CHARS,
    STATUS_MAX_LINES,
    _last_watcher_sample,
    _looks_like_hawking_root,
    _modellake_status,
    _multiple_hits,
    _pid_liveness,
    _resident_status,
    _status_roots,
    _visible_status_roots,
    format_machine_status,
    format_status,
)

MAX_STATUS_LINE = 80


def fresh() -> Path:
    root = Path(tempfile.mkdtemp(prefix="machine_status_test_"))
    atexit.register(shutil.rmtree, root, True)
    return root


def write_resident(root: Path, **fields) -> Path:
    path = root / ".hcli" / "resident" / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "state": "RUNNING",
        "cycles": 42,
        "last_event": "worker_heartbeat",
        "updated_at": time.time(),
    }
    state.update(fields)
    path.write_text(json.dumps(state), encoding="utf-8")
    return path


def write_watcher(root: Path, *, pid, rows) -> None:
    odyssey = root / "workspace" / "campaign" / "odyssey"
    (odyssey / "downloads").mkdir(parents=True, exist_ok=True)
    if pid is not None:
        (odyssey / ".modellake-watch.lock").write_text(str(pid), encoding="utf-8")
    log = odyssey / "downloads" / "modellake-watch.jsonl"
    log.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


# -- liveness ---------------------------------------------------------------


def test_a_pid_alone_is_never_live():
    """os.getpid() is alive, but not with someone else's start token."""
    assert _pid_liveness(os.getpid(), None) == "live"
    assert _pid_liveness(os.getpid(), "not-the-token-this-process-has") == "dead"


def test_absent_and_impossible_pids_are_not_alive():
    assert _pid_liveness(None, None) == "none"
    assert _pid_liveness(0, None) == "none"
    assert _pid_liveness("nonsense", None) == "none"
    # 2**22 is above every default pid_max; nothing can be running there.
    assert _pid_liveness(2**22, None) == "dead"


# -- resident ---------------------------------------------------------------


def test_no_resident_record_reports_absent_not_healthy():
    root = fresh()
    assert _resident_status([root]) is None
    assert format_machine_status({"resident": None, "modellake": None}) == [
        "Machine resident=absent modellake=absent"
    ]


def test_resident_fields_come_from_the_state_file():
    root = fresh()
    write_resident(root, cycles=7, last_event="mission_idle", state="IDLE")
    snap = _resident_status([root])
    assert snap["state"] == "IDLE"
    assert snap["cycles"] == 7
    assert snap["last_event"] == "mission_idle"
    assert snap["age_s"] < 60


def test_a_running_record_with_a_dead_pid_is_not_reported_live():
    """The exact 'never healthy merely because a PID exists' case."""
    root = fresh()
    write_resident(root, state="RUNNING", supervisor_pid=2**22)
    snap = _resident_status([root])
    assert snap["state"] == "RUNNING", "the record is left as written"
    assert snap["supervisor"] == "dead", "the process table is what decides"
    line = format_machine_status({"resident": snap})[0]
    assert "RUNNING" in line and "supervisor=dead" in line


def test_a_recycled_pid_does_not_inherit_the_record():
    root = fresh()
    write_resident(
        root, supervisor_pid=os.getpid(), supervisor_start_token="a-previous-boot"
    )
    assert _resident_status([root])["supervisor"] == "dead"


def test_a_stale_record_still_shows_its_age():
    root = fresh()
    write_resident(root, updated_at=time.time() - 7200)
    snap = _resident_status([root])
    assert 7000 < snap["age_s"] < 8000
    assert "age=7" in format_machine_status({"resident": snap})[0]


def test_unreadable_state_prints_unknown_not_zero():
    root = fresh()
    write_resident(root, cycles=None, last_event=None, state=None)
    line = format_machine_status({"resident": _resident_status([root])})[0]
    assert "cycles=unknown" in line and "event=unknown" in line
    assert "cycles=0" not in line


def test_a_corrupt_state_file_is_skipped_not_guessed():
    root = fresh()
    path = root / ".hcli" / "resident" / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not json", encoding="utf-8")
    assert _resident_status([root]) is None


# -- modellake --------------------------------------------------------------


def test_modellake_counts_the_watchers_own_last_sample():
    root = fresh()
    write_watcher(
        root,
        pid=os.getpid(),
        rows=[
            {"event": "watcher_sample", "active_jobs": ["old"], "ts": "2020-01-01T00:00:00+00:00"},
            {"event": "network_sample", "rx_bytes": 1},
            {
                "event": "watcher_sample",
                "active_jobs": ["a", "b", "c"],
                "active_remaining_bytes": 3 * 1024**3,
                "ts": "2020-01-02T00:00:00+00:00",
            },
            {"event": "network_sample", "rx_bytes": 2},
        ],
    )
    snap = _modellake_status([root])
    assert snap["jobs"] == 3, "the newest watcher_sample wins, not the newest row"
    assert snap["watcher"] == "live"
    assert snap["sample_age_s"] > 0
    line = format_machine_status({"modellake": snap})[0]
    assert "jobs=3" in line and "remaining=3.0GB" in line


def test_a_dead_watcher_is_not_dressed_up_as_activity():
    root = fresh()
    write_watcher(
        root,
        pid=2**22,
        rows=[{"event": "watcher_sample", "active_jobs": ["a"], "ts": "2020-01-01T00:00:00+00:00"}],
    )
    snap = _modellake_status([root])
    assert snap["watcher"] == "dead"
    assert snap["jobs"] == 1, "its last observation is still reported, with its age"
    assert snap["sample_age_s"] > 86400


def test_a_log_with_no_sample_yet_reports_unknown():
    root = fresh()
    write_watcher(root, pid=None, rows=[{"event": "network_sample", "rx_bytes": 1}])
    snap = _modellake_status([root])
    assert snap["jobs"] is None and snap["remaining_bytes"] is None
    line = format_machine_status({"modellake": snap})[0]
    assert "jobs=unknown" in line and "remaining=unknown" in line
    assert "jobs=0" not in line


def test_only_a_bounded_tail_of_the_log_is_read():
    """The live log is 580 MB. Reading it whole would stall every /status.

    Asserted by what is NOT reachable: a sample buried behind more than
    MODELLAKE_TAIL_BYTES of later rows must come back as None. A full read
    would find it, which is exactly the behaviour being ruled out -- checking
    that the newest sample wins proves nothing here, since a full read
    returns that one too.
    """
    root = fresh()
    filler = [{"event": "network_sample", "rx_bytes": n, "pad": "x" * 512} for n in range(4000)]
    write_watcher(
        root,
        pid=None,
        rows=[{"event": "watcher_sample", "active_jobs": ["buried"], "ts": "2020-01-01T00:00:00+00:00"}]
        + filler,
    )
    log = root / "workspace" / "campaign" / "odyssey" / "downloads" / "modellake-watch.jsonl"
    assert log.stat().st_size > 4 * MODELLAKE_TAIL_BYTES, "the fixture must exceed the tail"
    assert _last_watcher_sample(log) is None


def test_the_newest_sample_inside_the_tail_wins():
    root = fresh()
    filler = [{"event": "network_sample", "rx_bytes": n} for n in range(50)]
    write_watcher(
        root,
        pid=None,
        rows=[{"event": "watcher_sample", "active_jobs": ["older"], "ts": "2020-01-01T00:00:00+00:00"}]
        + filler
        + [{"event": "watcher_sample", "active_jobs": ["recent"], "ts": "2020-01-02T00:00:00+00:00"}]
        + filler,
    )
    log = root / "workspace" / "campaign" / "odyssey" / "downloads" / "modellake-watch.jsonl"
    assert _last_watcher_sample(log)["active_jobs"] == ["recent"]


def test_a_truncated_first_row_in_the_tail_is_skipped():
    root = fresh()
    odyssey = root / "workspace" / "campaign" / "odyssey" / "downloads"
    odyssey.mkdir(parents=True, exist_ok=True)
    log = odyssey / "modellake-watch.jsonl"
    good = json.dumps({"event": "watcher_sample", "active_jobs": ["ok"]})
    log.write_text('{"event": "watcher_sample", "active_jo\n' + good + "\n", encoding="utf-8")
    assert _last_watcher_sample(log)["active_jobs"] == ["ok"]


# -- the rendered screen ----------------------------------------------------


def test_machine_lines_join_the_one_screen_status():
    root = fresh()
    write_resident(root, supervisor_pid=2**22)
    write_watcher(
        root,
        pid=os.getpid(),
        rows=[{"event": "watcher_sample", "active_jobs": ["a"], "ts": "2020-01-01T00:00:00+00:00"}],
    )
    snap = {
        "mission_id": "m-1",
        "phase": "running",
        "goal": "ship it",
        "resident": _resident_status([root]),
        "modellake": _modellake_status([root]),
    }
    text = format_status(snap)
    assert "Resident RUNNING" in text
    assert "ModelLake watcher=live" in text
    # The house limits on /status: one screen, and no wrapped lines.
    lines = text.splitlines()
    assert len(lines) <= 10, lines
    for line in lines:
        assert len(line) <= MAX_STATUS_LINE, (len(line), line)


def test_a_long_event_name_cannot_widen_the_line():
    root = fresh()
    write_resident(
        root,
        state="STARTING",
        cycles=1234567,
        last_event="evacuation_checkpoint_error",
        supervisor_pid=2**22,
        updated_at=time.time() - 999999,
    )
    line = format_machine_status({"resident": _resident_status([root])})[0]
    assert len(line) <= MAX_STATUS_LINE, (len(line), line)


def test_status_roots_fall_back_to_the_repo():
    """A session opened in a scratch directory must still see the machine."""

    class Elsewhere:
        workspace_root = tempfile.gettempdir()

    roots = _status_roots(Elsewhere())
    assert len(roots) == 2 and roots[0] != roots[1]
    assert (roots[1] / "hcli" / "commands.py").is_file(), roots


def test_status_roots_also_try_resolve_workspace():
    """`resolve_workspace()` (HCLI_WORKSPACE / .hcli ancestor walk-up) is an
    existing convention built for exactly this and was never wired in."""
    old = os.environ.pop("HCLI_WORKSPACE", None)
    marked = fresh()
    (marked / ".hcli").mkdir()
    try:
        os.environ["HCLI_WORKSPACE"] = str(marked)

        class NoWorkspace:
            pass

        roots = _status_roots(NoWorkspace())
        # `resolve_workspace()` resolves symlinks (e.g. /tmp -> /private/tmp
        # on macOS); compare on resolved identity, not raw path equality.
        assert marked.resolve() in [r.resolve() for r in roots], roots
    finally:
        if old is None:
            os.environ.pop("HCLI_WORKSPACE", None)
        else:
            os.environ["HCLI_WORKSPACE"] = old


# -- workspace visibility: "quiet" vs "cannot see this host" ----------------


def test_a_real_workspace_with_no_state_yet_is_quiet_not_invisible():
    """A real checkout that just hasn't written resident/modellake state is
    the ordinary 'this host is quiet' case, not 'cannot see this host'."""
    real_root = fresh()
    # A CHECKOUT, not just a directory that happens to hold runtime state.
    # `.hcli` alone was the old marker and it is the exact shape of a stray
    # leftover in the system tmp dir, so it can no longer stand for "real".
    (real_root / ".hcli").mkdir()
    (real_root / "receipts").mkdir()

    class Ctrl:
        workspace_root = str(real_root)

    roots = _visible_status_roots(Ctrl(), repo_root=fresh())
    assert real_root in roots
    snap = {"resident": None, "modellake": None, "workspace_roots_seen": len(roots)}
    assert format_machine_status(snap) == ["Machine resident=absent modellake=absent"]


def test_a_neutral_cwd_with_nothing_real_says_so_not_absent():
    """The actual defect: a stamped-install/scratch cwd must not print the
    same word ('absent') that means 'this host has nothing running'."""
    bare_workspace = fresh()  # no .hcli/receipts/civilization/.git in it
    bare_repo = fresh()  # stands in for a stamped install with no vcs metadata
    old = os.environ.pop("HCLI_WORKSPACE", None)
    try:
        os.environ["HCLI_WORKSPACE"] = str(bare_workspace)

        class Ctrl:
            workspace_root = str(bare_workspace)

        roots = _visible_status_roots(Ctrl(), repo_root=bare_repo)
        assert roots == [], roots
        for max_lines in (2, 1):
            snap = {
                "resident": None,
                "modellake": None,
                "workspace_roots_seen": len(roots),
            }
            line = format_machine_status(snap, max_lines=max_lines)[0]
            assert "absent" not in line, line
            assert "no Hawking workspace" in line, line
            assert len(line) <= STATUS_LINE_CHARS
    finally:
        if old is None:
            os.environ.pop("HCLI_WORKSPACE", None)
        else:
            os.environ["HCLI_WORKSPACE"] = old


def test_a_stamped_install_container_does_not_look_like_a_workspace():
    assert not _looks_like_hawking_root(fresh())


def test_the_repo_checkout_looks_like_a_workspace():
    assert _looks_like_hawking_root(Path(__file__).resolve().parent.parent)


# -- several real candidates: expose the choice, never guess ----------------


def test_two_real_workspaces_with_their_own_resident_are_flagged_not_guessed():
    root_a = fresh()
    root_b = fresh()
    write_resident(root_a, state="RUNNING")
    write_resident(root_b, state="IDLE")
    roots = [root_a, root_b]
    assert _multiple_hits(roots, RESIDENT_STATE_REL)
    # Simulate what enrich_status_snapshot does once it detects the conflict.
    ambiguous = dict(_resident_status(roots), ambiguous=True, root_count=2)
    line = format_machine_status({"resident": ambiguous, "modellake": None})[0]
    assert "ambiguous" in line
    assert "RUNNING" not in line and "IDLE" not in line, "must not silently pick one"
    assert len(line) <= STATUS_LINE_CHARS
    one_line = format_machine_status({"resident": ambiguous}, max_lines=1)[0]
    assert "ambiguous" in one_line
    assert len(one_line) <= STATUS_LINE_CHARS


def test_a_single_real_workspace_is_never_flagged_ambiguous():
    root = fresh()
    write_resident(root)
    assert not _multiple_hits([root], RESIDENT_STATE_REL)


# -- Runtime health line width -----------------------------------------------


def test_runtime_health_line_never_exceeds_the_frame():
    """A long-context model's n_ctx/prompt_tokens used to push this line past
    STATUS_LINE_CHARS and wrap in the frame."""
    for n_ctx, prompt, tps in (
        (131072, 131072, 123.4),
        (1048576, 999999, 71.2),
        (1048576, 1048576, 999.9),
    ):
        snap = {
            "runtime": {
                "health": "ok",
                "resident": 1,
                "active_decode": 1,
                "queued": 0,
                "n_ctx": n_ctx,
                "prompt_tokens": prompt,
                "tps": tps,
            }
        }
        text = format_status(snap)
        runtime_lines = [l for l in text.splitlines() if l.startswith("Runtime ")]
        assert runtime_lines
        for line in runtime_lines:
            assert len(line) <= STATUS_LINE_CHARS, (len(line), line)


def test_runtime_health_down_and_unknown_also_fit():
    for health in ("down", None):
        snap = {"runtime": {"health": health, "queued": 999999999}}
        text = format_status(snap)
        runtime_lines = [l for l in text.splitlines() if l.startswith("Runtime ")]
        assert runtime_lines
        for line in runtime_lines:
            assert len(line) <= STATUS_LINE_CHARS, (len(line), line)


if __name__ == "__main__":
    for name, fn in sorted(dict(globals()).items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all green")


def test_a_warning_carrying_mission_still_fits_one_screen():
    """The case the boundary test above misses.

    That test lands on exactly 10 with no warning and asserts <= 10, so it
    certifies the boundary without ever crossing it. A mission carrying a
    no_progress warning spends an eleventh row, and the machine section is
    what has to give way -- not the cap, and not the warning.
    """
    root = fresh()
    write_resident(root, supervisor_pid=2**22)
    write_watcher(
        root,
        pid=os.getpid(),
        rows=[{"event": "watcher_sample", "active_jobs": ["a"], "ts": "2020-01-01T00:00:00+00:00"}],
    )
    snap = {
        "mission_id": "m-observe",
        "phase": "no_progress",
        "goal": "ship status",
        "resident": _resident_status([root]),
        "modellake": _modellake_status([root]),
        "no_progress_warning": "fingerprint repeated",
    }
    lines = format_status(snap).splitlines()
    assert len(lines) <= STATUS_MAX_LINES, lines
    assert any(line.startswith("no_progress:") for line in lines), "warning must survive"
    machine = [l for l in lines if l.startswith(("Resident ", "ModelLake ", "Machine "))]
    assert len(machine) == 1, machine
    # Collapsing must not silently drop the field that says ModelLake is still
    # competing for this host; a hard line[:80] cut used to eat it.
    assert "modellake=" in machine[0]
    assert max(len(line) for line in lines) <= STATUS_LINE_CHARS


def test_without_a_warning_both_machine_lines_are_kept():
    root = fresh()
    write_resident(root, supervisor_pid=2**22)
    write_watcher(
        root,
        pid=os.getpid(),
        rows=[{"event": "watcher_sample", "active_jobs": ["a"], "ts": "2020-01-01T00:00:00+00:00"}],
    )
    snap = {
        "mission_id": "m-1",
        "phase": "running",
        "goal": "ship it",
        "resident": _resident_status([root]),
        "modellake": _modellake_status([root]),
    }
    lines = format_status(snap).splitlines()
    assert len(lines) <= STATUS_MAX_LINES, lines
    assert sum(1 for l in lines if l.startswith(("Resident ", "ModelLake "))) == 2


# --- the failure path the HCLI_WORKSPACE-based test bypassed ---------------
# `_looks_like_hawking_root` accepted a bare `.git`, and `resolve_workspace()`
# walks up to the filesystem root looking for `.hcli`. A verifier reproduced
# both live: a stray `.hcli` in the system tmp dir and a freshly `git init`-ed
# unrelated repo each read as a visible Hawking workspace. On a developer box
# that is the normal case, so `workspace_roots_seen` came back non-zero with
# zero real roots and /status fell back to the misleading "absent" line.


def test_a_bare_git_repo_is_not_a_hawking_workspace(tmp_path):
    from hcli.commands import _looks_like_hawking_root

    (tmp_path / ".git").mkdir()
    assert _looks_like_hawking_root(tmp_path) is False


def test_a_stray_hcli_directory_alone_is_not_a_hawking_workspace(tmp_path):
    """The exact leftover the verifier found sitting in the system tmp root."""
    from hcli.commands import _looks_like_hawking_root

    (tmp_path / ".hcli").mkdir()
    assert _looks_like_hawking_root(tmp_path) is False


def test_an_empty_scratch_directory_is_not_a_hawking_workspace(tmp_path):
    from hcli.commands import _looks_like_hawking_root

    assert _looks_like_hawking_root(tmp_path) is False


def test_runtime_state_beside_receipts_is_a_hawking_workspace(tmp_path):
    from hcli.commands import _looks_like_hawking_root

    (tmp_path / ".hcli").mkdir()
    (tmp_path / "receipts").mkdir()
    assert _looks_like_hawking_root(tmp_path) is True


def test_the_roadmap_alone_is_decisive(tmp_path):
    """A checkout with no runtime state yet is still unmistakably Hawking."""
    from hcli.commands import _looks_like_hawking_root

    (tmp_path / "civilization").mkdir()
    (tmp_path / "civilization" / "ROADMAP_STATE.json").write_text("{}", encoding="utf-8")
    assert _looks_like_hawking_root(tmp_path) is True


def test_the_live_repo_is_still_recognised():
    from hcli.commands import _looks_like_hawking_root
    from hcli.paths import find_repo_root

    assert _looks_like_hawking_root(find_repo_root(Path(__file__))) is True
