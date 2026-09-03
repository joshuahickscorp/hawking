"""The watcher was already writing this; nothing was reading it.

Two failures matter here: presenting a stale sample as current, and inventing a
bandwidth when none was measured. Both are pinned.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from tools.future import modellake_scheduler_view as mv


def _install_watch_log(tmp_path, monkeypatch, events):
    """Write a watcher JSONL. The view always drops the first line as a fragment."""
    p = tmp_path / "watch.jsonl"
    p.write_text("{}\n" + "".join(json.dumps(e) + "\n" for e in events))
    monkeypatch.setattr(mv, "WATCH_LOG", p)
    return p


def _skip_if_lake_unmounted():
    if not mv.sr.LAKE.is_dir():
        pytest.skip(f"{mv.sr.LAKE} is not attached")


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


def test_a_stale_sample_is_flagged_not_presented_as_current(tmp_path, monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    _install_watch_log(tmp_path, monkeypatch, [
        {"event": "watcher_sample", "ts": ts, "active_jobs": [], "free_bytes": 1},
    ])
    lv = mv.live()
    assert lv["stale"] is False
    old = mv.live(now=datetime.now(timezone.utc).timestamp() + 10_000)
    assert old["stale"] is True
    assert old["sample_age_s"] > mv.STALE_SECONDS


def test_the_eta_comes_from_the_measured_rate(tmp_path, monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    remaining = 20_000
    rates = [1000, 3000, 2000]
    events = [
        {
            "event": "watcher_sample",
            "ts": ts,
            "active_jobs": ["job-a"],
            "active_remaining_bytes": remaining,
            "free_bytes": 1,
        },
    ]
    for rx in rates:
        events.append({"event": "network_sample", "ts": ts, "rx_bytes_per_sec": rx})
    _install_watch_log(tmp_path, monkeypatch, events)
    e = mv.eta()
    median = 2000
    assert e["rx_bytes_per_sec_median"] == median
    assert e["seconds"] == round(remaining / median)
    assert "not a promise" in e["is_an_estimate_because"]


def test_no_measured_rate_means_no_eta_rather_than_a_nominal_one(monkeypatch):
    real = mv.live
    monkeypatch.setattr(mv, "live",
                        lambda now=None: {**real(now),
                                          "rx_bytes_per_sec_median": None})
    e = mv.eta()
    assert e["seconds"] is None
    assert "unavailable" in e["why"]


def test_the_eta_says_what_it_does_not_cover(tmp_path, monkeypatch):
    ts = datetime.now(timezone.utc).isoformat()
    _install_watch_log(tmp_path, monkeypatch, [
        {
            "event": "watcher_sample",
            "ts": ts,
            "active_jobs": ["job-a"],
            "active_remaining_bytes": 20_000,
            "free_bytes": 1,
        },
        {"event": "network_sample", "ts": ts, "rx_bytes_per_sec": 2000},
    ])
    e = mv.eta()
    assert "queued specimens the watcher has not admitted" in e["does_not_cover"]


def test_active_jobs_are_joined_to_the_registry(tmp_path, monkeypatch):
    known = "known--specimen@abc"
    unknown = "unknown--specimen@def"
    ts = datetime.now(timezone.utc).isoformat()
    _install_watch_log(tmp_path, monkeypatch, [
        {
            "event": "watcher_sample",
            "ts": ts,
            "active_jobs": [known, unknown],
            "active_remaining_bytes": 1,
            "free_bytes": 1,
        },
    ])
    monkeypatch.setattr(mv.sr, "registry", lambda: [{
        "id": known,
        "lifecycle": "DOWNLOADING",
        "architecture": {"model_type": "synth"},
    }])
    a = mv.arrivals()
    by_id = {row["id"]: row for row in a["active"]}
    assert by_id[known]["known_to_registry"] is True
    assert by_id[unknown]["known_to_registry"] is False
    assert by_id[known]["lifecycle"]
    assert a["n_unknown_to_registry"] == 1


def test_a_job_the_registry_cannot_see_yet_is_not_an_error():
    _skip_if_lake_unmounted()
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
    _skip_if_lake_unmounted()
    b = mv.build()
    assert "never missing data; it was a missing reader" in \
        b["the_watcher_already_wrote_all_of_this"]
