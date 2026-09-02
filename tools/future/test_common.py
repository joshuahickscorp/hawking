"""Guarantees the whole sidecar rests on: sealed receipts, and safe git queries."""
from tools.future import _common

def test_git_queries_never_take_the_index_lock():
    """A stale .git/index.lock blocks every later commit in the repo.

    `git status` refreshes and therefore WRITES the index lock on a tree this
    size, and a git killed while holding it strands the lock -- which has
    happened repeatedly here, each time several minutes old with no process
    holding it. Every caller in this package only reads.
    """
    lock = _common.REPO / ".git" / "index.lock"
    before = lock.exists()
    _common.git("status", "--porcelain")
    assert lock.exists() == before, "a read-only query created the index lock"


def test_git_query_is_bounded_and_fails_soft():
    """A query with no timeout is a query that can hang the resident forever."""
    assert _common.GIT_TIMEOUT_S > 0
    assert _common.git("rev-parse", "HEAD"), "HEAD must resolve"
    # An unknown subcommand must come back empty rather than raising: every
    # caller already treats empty as "not found".
    assert _common.git("not-a-real-subcommand") == ""


def test_write_receipt_is_a_noop_when_only_bookkeeping_moved(tmp_path, monkeypatch):
    """A rerun that found no new evidence must not dirty a clean tree.

    bench.recorded_at is a fresh clock reading on every call, and a producer
    that also stamps "head" gets a different commit every time on a repo that
    lands as many commits as this one does. G151 hand-fixed a batch of
    receipts that had drifted on exactly these two fields with nothing else
    changed ("These carry a process snapshot and a head SHA, so they
    re-stamp whenever their producer runs" - commit 0a844ee1a). Without this
    guard every one of those receipts churns again the next time its
    producer runs, forever.
    """
    monkeypatch.setattr(_common, "RECEIPTS", tmp_path)

    first = _common.write_receipt("R.json", {"schema": "x", "measurement": 42, "head": "aaa"}, "test")
    written_first = first.read_text()

    # Only "head" (and bench.recorded_at, stamped fresh every call) moved.
    second = _common.write_receipt("R.json", {"schema": "x", "measurement": 42, "head": "bbb"}, "test")
    assert second.read_text() == written_first, (
        "no new evidence exists (measurement unchanged); the file must not be rewritten"
    )

    # Real content changed: this must still be written.
    third = _common.write_receipt("R.json", {"schema": "x", "measurement": 43, "head": "bbb"}, "test")
    assert third.read_text() != written_first, "a genuine content change must still land"
