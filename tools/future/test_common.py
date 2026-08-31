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
