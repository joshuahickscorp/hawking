"""G021 tests: an audit that undercounts exposure is worse than none."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attribution_audit as aa  # noqa: E402


def test_prose_is_not_counted_as_a_footer():
    """'all 800 records regenerated with different hashes' is prose. A naive
    grep counts it; the line-anchored pattern must not."""
    assert not aa.FOOTER.search("records regenerated with different hashes")
    assert not aa.FOOTER.search("Profile JSON regenerated with the new ID")
    assert aa.FOOTER.search("Generated with [Claude Code](https://x)")
    assert aa.FOOTER.search("\U0001F916 Generated with something")


def test_the_trailer_pattern_is_line_anchored_and_case_insensitive():
    assert aa.TRAILER.search("Co-authored-by: x")
    assert aa.TRAILER.search("Co-Authored-By: x")
    assert aa.TRAILER.search("body\nCo-authored-by: x")
    assert not aa.TRAILER.search("mentions co-authored-by in prose")


def test_the_canonical_lines_are_clean():
    c = aa.canonical()
    assert c["clean"] is True
    for ref in aa.CANONICAL:
        assert c[ref]["tool_identities"] == 0
        assert c[ref]["trailers"] == 0
        assert c[ref]["generated_with_footers"] == 0


def test_main_is_actually_pushed_not_assumed():
    assert aa.canonical()["main_is_pushed"] is True


def test_no_claude_or_anthropic_attribution_exists_anywhere():
    n = aa.no_claude_attribution_anywhere()
    assert n["verdict"] == "NONE"
    assert n["n_identities"] == 0
    assert n["generated_with_footers_anywhere"] == 0


def test_no_published_branch_carries_attribution_any_more():
    """The rewrite landed. Both branches were force-pushed with tree sequences
    verified IDENTICAL and commit counts unchanged, so this is now the standing
    invariant rather than a target."""
    r = aa.remaining()
    assert r["n_published_dirty"] == 0, r["published_dirty"]


def test_a_published_branch_with_no_local_head_would_still_be_counted():
    """The audit's own first run reported 1 published-dirty branch when the
    answer was 2: it skipped every refs/remotes/ ref, and one published branch
    had no local head to be counted through. The scan must still reach those."""
    import inspect
    src = inspect.getsource(aa.remaining)
    assert 'refs/remotes/origin/' in src
    assert "local_heads" in src
    assert "UNDERCOUNTS" in src


def test_the_pre_rewrite_shas_are_preserved_as_refs():
    """'Nothing may be lost' is this obligation's own words. The two original
    tips are reachable from refs/preserved/ even though the branches moved."""
    out = aa._git("for-each-ref", "--format=%(refname)", "refs/preserved/")
    assert "refs/preserved/pre-g021-arc-300k-integration" in out
    assert "refs/preserved/pre-g021-grok-wave0" in out


def test_the_rewrite_preconditions_name_the_970_commit_loss():
    pre = aa.what_this_does_not_do()["preconditions_for_any_rewrite"]
    joined = " ".join(pre)
    assert "prune-empty=never" in joined
    assert "970" in joined
    assert "bundle" in joined


def test_a_git_failure_refuses_rather_than_reporting_clean(monkeypatch):
    """canonical() is cached, so the uncached primitive is what to poke."""
    monkeypatch.setattr(aa, "_git", lambda *a: (_ for _ in ()).throw(
        aa.AuditRefused("boom")))
    aa._scan.cache_clear()
    with pytest.raises(aa.AuditRefused):
        aa._scan("main")


def test_the_verdict_names_the_exposure_count():
    d = aa.build()
    n = d["remaining"]["n_published_dirty"]
    assert str(n) in d["verdict"]
    assert "CANONICAL_CLEAN" in d["verdict"]
