#!/usr/bin/env python3.12
"""Tests for the ecosystem status/ETA composer (eco_status)."""
import pathlib
import sys

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import eco_status as stat  # noqa: E402
from eco_common import atomic_write_json, now_iso  # noqa: E402


def _cfg(tmp_path, include_plan=False):
    croot = tmp_path / "doctor_v5_ultra"
    croot.mkdir(parents=True)
    atomic_write_json(croot / "queue_state.json", {
        "plan_sha256": "a" * 64,
        "cells": {"c1": {"status": "complete"}, "c2": {"status": "running"},
                  "c3": {"status": "pending"}},
        "report_checkpoints": {"sub-120B": None, "120B": None},
    })
    return stat.StatusConfig(campaign_root=croot, state_root=tmp_path / "frontier_eco",
                             include_plan=include_plan)


def test_selftest_green():
    assert stat.selftest()["ok"] is True


def test_compose_shows_progress_and_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    st = stat.compose_status(cfg)
    assert "1/3 terminal" in st["text"]
    assert "BLOCKED until signed release" in st["text"]
    assert st["summary"]["progress"]["progress_pct"] == _approx(33.3)


def test_idempotent_send_with_fake_sender(tmp_path):
    cfg = _cfg(tmp_path)
    sent = []

    def fake(text):
        sent.append(text)
        return {"message_id": len(sent), "sent_at": now_iso()}

    r1 = stat.send_status(cfg, sender=fake)
    r2 = stat.send_status(cfg, sender=fake)
    assert r1["sent"] == 1
    assert r2["sent"] == 0        # same text not resent
    assert len(sent) == 1
    r3 = stat.send_status(cfg, sender=fake, force=True)
    assert r3["sent"] == 1
    assert len(sent) == 2


def _approx(v):
    import pytest
    return pytest.approx(v, abs=0.1)
