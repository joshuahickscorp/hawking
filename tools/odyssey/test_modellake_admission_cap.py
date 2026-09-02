"""The admission cap must count LIVE JOBS, not set memberships.

A job this watcher launched lives in `children`; the pid scan that fills
`active_tags` finds the very same process again. Summing the two counted every
live download twice, so with MAX_DOWNLOAD_JOBS=2 one transfer saturated the cap
and the queue below it was never reached -- Inkling-Small ran alone for three
hours while three small models sat at 0-4 GB.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

SOURCE = Path(__file__).with_name("modellake_watch.py")


def _admission_cap_expression() -> str:
    """The right-hand side of the `active_count` assignment that guards QUEUE."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "active_count":
                found.append(ast.unparse(node.value))
    assert found, "no `active_count` assignment found; the guard was renamed"
    return found[0]


def test_admission_cap_unions_and_does_not_sum():
    expr = _admission_cap_expression()
    tree = ast.parse(expr, mode="eval").body
    assert not (isinstance(tree, ast.BinOp) and isinstance(tree.op, ast.Add)), (
        f"active_count = {expr!r} SUMS two overlapping sets. A job in both "
        "`children` and `active_tags` is counted twice and the cap halves."
    )
    assert "active_tags" in expr and "children" in expr, (
        f"active_count = {expr!r} no longer considers both live-job sets"
    )


def test_a_single_live_job_leaves_headroom_under_a_cap_of_two():
    """Reproduce the exact production shape: one job, both sets, cap 2."""
    active_tags = {"thinkingmachines--Inkling-Small@8cc5877b44d3"}
    children = {"thinkingmachines--Inkling-Small@8cc5877b44d3": object()}
    expr = _admission_cap_expression()
    active_count = eval(expr, {}, {"active_tags": active_tags, "children": children})
    assert active_count == 1, (
        f"one live download counted as {active_count}; with "
        "MAX_DOWNLOAD_JOBS=2 that breaks out of the QUEUE loop immediately"
    )
    assert active_count < 2, "no admission headroom under a cap of two"


def test_two_distinct_jobs_still_reach_the_cap():
    """The fix must not let the cap be exceeded -- union, not deduplicate away."""
    expr = _admission_cap_expression()
    active_count = eval(
        expr, {}, {"active_tags": {"a"}, "children": {"b": object()}}
    )
    assert active_count == 2, (
        f"two distinct live jobs counted as {active_count}; the cap would be "
        "exceeded and MAX_DOWNLOAD_JOBS would stop bounding concurrency"
    )


def test_gated_repositories_are_flagged_not_retried():
    """Both returned "Access denied. This repository requires approval."."""
    import sys

    sys.path.insert(0, str(SOURCE.parent))
    import modellake_watch as W

    gated = {j["repo"] for j in W.QUEUE if j.get("requires_manual_auth")}
    for repo in ("stabilityai/stable-audio-open-1.0", "facebook/blt-7b"):
        assert repo in gated, (
            f"{repo} is gated upstream but not flagged; it burns an admission "
            "attempt and a launch every rearm cycle"
        )
    assert len(gated) >= 2, "the gated set collapsed; this test would pass vacuously"


def _admission_reservation_source() -> str:
    """The storage-reservation block that guards a QUEUE admission."""
    text = SOURCE.read_text(encoding="utf-8")
    start = text.index("scratch = max(10_000_000_000")
    head = text.rindex("present =", 0, start)
    return text[head:text.index("if projected <", start)]


def test_reservation_accounts_for_bytes_already_on_disk():
    """A nearly-complete model must not be blocked by its own total size.

    Inkling-Small sat at 515.9 of 531.9 GB needing 16 GB, and every admission
    pass emitted admission_blocked_storage: the reservation used the full
    manifest, which exceeds the 519 GB free on the volume. Reserving `expected`
    rather than `expected - present` makes a model unfinishable on any disk
    smaller than its own total size.
    """
    block = _admission_reservation_source()
    assert "present = None" not in block, (
        "the reservation discards what is already on disk; a 16 GB remainder "
        "is reserved as though it were the full manifest"
    )
    assert "durable_bytes(" in block, (
        "the reservation does not measure bytes already present"
    )
    assert "expected - present" in block, (
        f"remaining is not derived from present bytes:\n{block}"
    )


def test_a_nearly_complete_giant_is_admitted_on_a_smaller_disk():
    """Reproduce the exact production arithmetic that deadlocked."""
    import sys

    sys.path.insert(0, str(SOURCE.parent))
    import modellake_watch as W

    expected = 531_900_000_000
    present = 515_900_000_000
    free = 519_000_000_000
    active_remaining = 0

    remaining = max(0, expected - present)
    scratch = max(10_000_000_000, int(remaining * 0.05))
    uncertainty = max(5_000_000_000, int(remaining * 0.02))
    projected = (free - active_remaining - remaining - scratch
                 - uncertainty - W.KNOWN_TEMP_BYTES)
    assert projected >= W.FLOOR_BYTES, (
        f"projected {projected / 1e9:.1f} GB is below the "
        f"{W.FLOOR_BYTES / 1e9:.0f} GB floor; a 16 GB remainder was refused"
    )

    # The old arithmetic, kept as the contrast that makes this test bite.
    old_remaining = expected
    old_projected = (free - active_remaining - old_remaining
                     - max(10_000_000_000, int(old_remaining * 0.05))
                     - max(5_000_000_000, int(old_remaining * 0.02))
                     - W.KNOWN_TEMP_BYTES)
    assert old_projected < W.FLOOR_BYTES, (
        "the pre-fix arithmetic no longer reproduces the deadlock; this test "
        "would pass for the wrong reason"
    )


def test_the_event_log_is_bounded_and_the_tail_survives_a_rotation(tmp_path, monkeypatch):
    """The log reached 612.6 MB, append-only, growing at ~10 Hz.

    Nothing reclaimed it, so on the long horizon this watcher exists for it
    fills the volume it is meant to protect. Rotation alone is not enough:
    _download_history tails the log to learn which jobs were ever started, and
    a roll that hides that would make reconcile() read "started, and now
    nothing on disk" -- its vanished-payload signal -- for every job older
    than the roll.
    """
    import sys

    sys.path.insert(0, str(SOURCE.parent))
    import modellake_watch as W

    log = tmp_path / "watch.jsonl"
    monkeypatch.setattr(W, "LOG", log)
    monkeypatch.setattr(W, "DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(W, "LOG_MAX_BYTES", 4096)

    # Exactly one rotation: enough to roll the live file, not so much that
    # the early row falls off the rotated generation too. Retention across
    # MANY rolls is bounded by design and asserted separately below.
    W.emit("download_started", job="early-job")
    for i in range(25):
        W.emit("network_sample", filler="x" * 200, i=i)
    W.emit("download_started", job="late-job")

    assert log.stat().st_size < 4096 * 4, (
        f"log grew to {log.stat().st_size} bytes against a 4096-byte cap; "
        "rotation did not fire"
    )
    rotated = log.with_suffix(log.suffix + ".1")
    assert rotated.is_file(), "no rotated generation was kept"

    _last_exit, started = W._download_history(max_lines=20_000)
    assert "late-job" in started, "the live log's own history was lost"
    assert "early-job" in started, (
        "a job started before the rotation is no longer remembered; "
        "reconcile() would read it as a vanished payload"
    )


def test_rotation_keeps_only_a_bounded_number_of_generations(tmp_path, monkeypatch):
    """Bounded means bounded: rolling must not just rename the problem."""
    import sys

    sys.path.insert(0, str(SOURCE.parent))
    import modellake_watch as W

    log = tmp_path / "watch.jsonl"
    monkeypatch.setattr(W, "LOG", log)
    monkeypatch.setattr(W, "DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(W, "LOG_MAX_BYTES", 2048)

    for i in range(2000):
        W.emit("network_sample", filler="y" * 200, i=i)

    generations = sorted(tmp_path.glob("watch.jsonl.*"))
    assert len(generations) <= W.LOG_GENERATIONS, (
        f"{len(generations)} generations kept, cap is {W.LOG_GENERATIONS}: "
        f"{[p.name for p in generations]}"
    )
    total = sum(p.stat().st_size for p in list(generations) + [log])
    assert total < 2048 * (W.LOG_GENERATIONS + 2), (
        f"total on-disk log is {total} bytes; growth is still unbounded"
    )


def test_one_generation_outlasts_the_readers_window():
    """Rotation is only safe because a generation dwarfs what readers tail.

    _download_history tails 20_000 lines to learn which jobs were started. If
    one generation held fewer rows than that window, a single roll would
    truncate the reader's view and reconcile() would start inventing vanished
    payloads. This pins the relationship instead of trusting the constant.
    """
    import sys

    sys.path.insert(0, str(SOURCE.parent))
    import modellake_watch as W

    readers_window_lines = 20_000
    generous_row_bytes = 400  # real rows measured well under this
    rows_per_generation = W.LOG_MAX_BYTES / generous_row_bytes
    assert rows_per_generation > readers_window_lines * 4, (
        f"one generation holds ~{rows_per_generation:.0f} rows against a "
        f"{readers_window_lines}-line reader window; a single rotation could "
        "truncate what reconcile() depends on"
    )
