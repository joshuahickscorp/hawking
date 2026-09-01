"""A protected lease that stopped only the workers was not a lease.

The load-invariance check is the load-bearing test here: the campaign has
licensed many claims on "ratios hold under load" and this is the first time that
rule is checked against a protected window at this size.
"""
from __future__ import annotations

import json

import pytest

from tools.future import protected_bitcast_absolute as pa


def _ctrl():
    return json.loads((pa.REPO / pa.CTRL_REL).read_text())


def test_the_pair_is_matched_and_token_identical():
    m = pa.measured()
    assert m["token_identical"] is True
    assert m["dispatches"] == 580
    assert m["fallbacks"] == 0
    assert m["n_tokens"] == 32 and m["reps"] == 9


def test_a_token_divergence_refuses(monkeypatch):
    d = _ctrl()
    d["decode"][pa.LIVE_ARM]["new_token_ids"] = [1, 2]
    monkeypatch.setattr(pa, "_arm",
                        lambda rel, arm: d["decode"][arm] if rel == pa.CTRL_REL
                        else json.loads((pa.REPO / rel).read_text())["decode"][arm])
    with pytest.raises(pa.LeaseRefused, match="a regression, not a win"):
        pa.measured()


def test_a_missing_lease_receipt_refuses(monkeypatch):
    monkeypatch.setattr(pa, "CTRL_REL", "receipts/future/NO_SUCH.json")
    with pytest.raises(pa.LeaseRefused, match="run the lease first"):
        pa.measured()


def test_the_lease_stopped_the_supervisor_not_just_the_workers():
    l = pa.lease()
    assert "SUPERVISOR first" in l["method"]
    assert "respawned them mid-window" in l["why_the_supervisor_first"]
    assert l["states_at_open"].startswith("supervisor T")
    assert l["evidence_class"] == "PROTECTED_ABSOLUTE"


def test_nothing_was_killed_only_stopped():
    l = pa.lease()
    assert "SIGSTOP and SIGCONT only" in l["nothing_was_killed"]
    assert "no partial file was discarded" in l["nothing_was_killed"]


def test_the_load_invariance_rule_is_checked_not_asserted():
    li = pa.load_invariance()
    assert li["agrees"] is True
    assert li["relative_difference"] < 0.02
    assert "had not been checked" in li["reading"], \
        "the point is that the standing rule was previously unverified at this size"


def test_a_disagreeing_protected_measurement_would_show_as_disagreement(monkeypatch):
    monkeypatch.setattr(pa, "UNPROTECTED_COMBINED_MS", 2.0)
    li = pa.load_invariance()
    assert li["agrees"] is False, "the check must be able to fail"


def test_the_two_harnesses_disagree_and_that_is_reported_as_a_range():
    a = pa.against_the_canonical_baseline()
    assert a["the_two_controls_differ_by_ms"] > 0.1
    assert len(a["tps_range"]) == 2 and a["tps_range"][0] < a["tps_range"][1]
    assert "choosing the flattering number" in pa.build()["claim_boundary"]
    assert "still not PROMOTED" in a["open_item"]


def test_sixty_is_not_claimed():
    b = pa.build()
    assert b["still_short_of_60_by_ms"] > 0
    assert b["checkpoints_crossed"] == ["40 TPS"]
    assert b["default_is_unchanged"] is True
