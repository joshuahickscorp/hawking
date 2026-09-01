"""The register may only shrink by real handoff, and it must not flatter itself.

The failure guarded here is a register that points at modules nobody wrote, and
a count that reads as progress while every judgement stays with Claude.
"""
from __future__ import annotations

import pytest

from tools.future import recurring_operations as ro


def test_every_named_owner_exists_on_disk():
    for e in ro.entries():
        if e["owner"]:
            assert (ro.REPO / e["owner"]).is_file(), e["op"]


def test_an_owner_that_does_not_exist_refuses(monkeypatch):
    bad = ({"op": "x", "owner": "tools/future/no_such_module.py",
            "rule": None, "note": "y"},)
    monkeypatch.setattr(ro, "OPERATIONS", bad)
    with pytest.raises(ro.RegisterRefused, match="fake completion"):
        ro.entries()


def test_unowned_operations_are_listed_not_dropped():
    c = ro.counts()
    assert c["n_still_claude_owned"] > 0
    assert len(c["still_claude_owned"]) == c["n_still_claude_owned"]
    assert c["n_handed_off"] + c["n_still_claude_owned"] == c["n_operations"]


def test_the_remaining_operations_are_the_judgement_ones():
    """If what remains were mechanical, the handoff count would mean something."""
    left = " ".join(ro.counts()["still_claude_owned"])
    assert "design the next kernel candidate" in left
    assert "decide that an obligation is met" in left
    assert "reconcile two harnesses" in left


def test_the_module_refuses_to_read_its_own_count_as_progress():
    f = ro.failure_mode()
    assert "closer to FAIL than the handed-off count suggests" in \
        f["where_this_campaign_sits"]
    assert "not progress toward autonomy" in f["the_count_is_not_the_metric"]
    assert "faster scaffold" in f["the_count_is_not_the_metric"]


def test_the_criterion_is_stated_as_a_failure_mode():
    f = ro.failure_mode()
    assert "fifty-point steer" in f["FAIL_looks_like"]
    assert "user optionally observes" in f["PASS_looks_like"]


def test_handoff_requires_ownership_not_mention():
    h = ro.build()["how_this_falls_honestly"]
    assert "OWNS it - not when a module mentions it" in h


def test_every_handed_off_entry_says_what_changed():
    for e in ro.entries():
        assert e["note"], e["op"]
