"""Sharing the facts cache is safe; sharing the audit cache is not.

The difference is what each reads. The facts dump runs the index binary with
`--git-head --commit <sha>`, which reads the git OBJECT STORE. The audit that
consumes it also reads acceptance receipts off the WORKING TREE and keys on none
of them. So the expensive deterministic half can be shared across processes and
the unsound half must stay per-PID.

Measured before this split: 254 dumps materialised on this machine, 78 distinct.
176 recomputations of a byte-identical 9 MB artifact at ~24 s each.
"""
from __future__ import annotations

import os

from tools.roadmap import index_client
from tools.roadmap.gitfs import SourceView


def test_the_facts_cache_directory_is_stable_across_processes():
    """The whole point. A PID in this path is the defect being fixed."""
    d = index_client.facts_cache_dir()
    assert str(os.getpid()) not in d.name, d
    assert d == index_client.facts_cache_dir()
    assert d.is_dir()


def test_the_audit_artifact_directory_is_still_per_process():
    """The guard on the other cache. It must NOT acquire the same treatment."""
    d = index_client.artifact_session_dir()
    session = os.environ.get("ROADMAP_ARTIFACT_SESSION")
    assert session and session in d.name, (
        "artifact_session_dir stopped being per-session; the audit cache key holds "
        "no digest of the worktree receipts the auditor reads, so sharing it serves "
        "a graph from before the last receipt was written"
    )


def test_a_rebuilt_index_binary_busts_the_facts_key(monkeypatch):
    """The dump depends on the binary, so the key must too.

    Without this the same commit plus a rebuilt binary reuses the old answer.
    """
    view = SourceView()
    before = index_client._facts_key(view, True)
    assert index_client.index_bin_stamp() in {str(p) for p in before}, before
    monkeypatch.setattr(index_client, "index_bin_stamp", lambda: "0:0")
    after = index_client._facts_key(view, True)
    assert before != after, "rebuilding the index binary would reuse a stale dump"


def test_the_key_still_covers_commit_overlay_and_auditor_code():
    view = SourceView()
    key = index_client._facts_key(view, True)
    from tools.roadmap.gitfs import head_commit
    assert head_commit() in {str(p) for p in key}
    assert index_client.code_digest() in {str(p) for p in key}
    assert len(key) == 5, key


def test_the_dump_cannot_see_an_uncommitted_file():
    """The property that makes cross-process sharing sound.

    If the dump read the worktree, a shared cache would serve stale facts the
    moment anyone edited a file without committing.
    """
    probe = index_client.REPO / "tools" / "roadmap" / "_probe_facts_cache.py"
    probe.write_text("# uncommitted probe\n")
    try:
        dump = index_client.load_python_facts(SourceView(), git_head=True)
        assert "tools/roadmap/_probe_facts_cache.py" not in dump["files"], (
            "the facts dump reads the working tree, so the shared cache is unsound"
        )
    finally:
        probe.unlink(missing_ok=True)


def test_the_cache_is_bounded():
    """~9 MB per entry. 254 were materialised while 78 were distinct."""
    assert index_client.FACTS_CACHE_KEEP > 0
    index_client._reap_facts_cache()
    kept = list(index_client.facts_cache_dir().glob("facts-*.json"))
    assert len(kept) <= index_client.FACTS_CACHE_KEEP
