"""The watcher was already writing this; nothing was reading it.

Two failures matter here: presenting a stale sample as current, and inventing a
bandwidth when none was measured. Both are pinned.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from tools.future import modellake_scheduler_view as mv


def test_the_live_view_reads_the_real_watcher_tail():
    lv = mv.live()
    assert lv["sample_ts"]
    assert isinstance(lv["active_jobs"], list)
    assert lv["free_bytes"] and lv["free_bytes"] > 0


def test_a_missing_watch_log_refuses_rather_than_reporting_nothing_downloading(monkeypatch):
    monkeypatch.setattr(mv, "WATCH_LOG", mv.REPO / "no" / "such.jsonl")
    with pytest.raises(mv.LakeViewRefused, match="watcher is not running"):
        mv.live()


def test_a_log_with_no_parseable_event_refuses(monkeypatch):
    monkeypatch.setattr(mv, "_tail", lambda: [{"event": "network_sample"}])
    with pytest.raises(mv.LakeViewRefused, match="no watcher_sample"):
        mv.live()


def test_a_stale_sample_is_flagged_not_presented_as_current():
    lv = mv.live()
    assert lv["stale"] is False, "the watcher is live right now"
    old = mv.live(now=datetime.now(timezone.utc).timestamp() + 10_000)
    assert old["stale"] is True
    assert old["sample_age_s"] > mv.STALE_SECONDS


def test_the_eta_comes_from_the_measured_rate():
    e = mv.eta()
    assert e["seconds"] and e["seconds"] > 0
    assert e["rx_bytes_per_sec_median"] > 0
    assert "not a promise" in e["is_an_estimate_because"]


def test_no_measured_rate_means_no_eta_rather_than_a_nominal_one(monkeypatch):
    real = mv.live
    monkeypatch.setattr(mv, "live",
                        lambda now=None: {**real(now),
                                          "rx_bytes_per_sec_median": None})
    e = mv.eta()
    assert e["seconds"] is None
    assert "unavailable" in e["why"]


def test_the_eta_says_what_it_does_not_cover():
    e = mv.eta()
    assert "queued specimens the watcher has not admitted" in e["does_not_cover"]


def test_active_jobs_are_joined_to_the_registry():
    a = mv.arrivals()
    assert a["n_active"] > 0
    for row in a["active"]:
        assert "known_to_registry" in row
        if row["known_to_registry"]:
            assert row["lifecycle"]


def test_a_job_the_registry_cannot_see_yet_is_not_an_error():
    a = mv.arrivals()
    assert "is NOT an error" in a["unknown_means"]


def test_the_seal_trigger_is_declared_and_admitted_to_be_unwired():
    s = mv.seal_contract()
    assert len(s["must_trigger"]) == 6
    assert s["is_this_wired"] is False
    assert "DECLARED, NOT WIRED" in s["honest_status"]
    assert "fake completion" in s["honest_status"]
    assert s["what_wiring_it_needs"]


def test_the_finding_is_that_the_data_existed_and_nobody_read_it():
    b = mv.build()
    assert "never missing data; it was a missing reader" in \
        b["the_watcher_already_wrote_all_of_this"]
