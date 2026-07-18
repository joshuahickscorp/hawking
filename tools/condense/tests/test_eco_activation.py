#!/usr/bin/env python3.12
"""Tests for the fail-closed activation gate (eco_activation)."""
import pathlib
import sys

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import eco_activation as act  # noqa: E402
from eco_common import atomic_write_json  # noqa: E402

PLAN = "a" * 64


def _cfg(tmp_path):
    croot = tmp_path / "doctor_v5_ultra"
    croot.mkdir(parents=True)
    return act.ActivationConfig(campaign_root=croot, state_root=tmp_path / "frontier_eco",
                                expected_plan_sha256=PLAN)


def _write_queue(cfg, *, terminal, sealed_reports, running=False):
    cells = {"c1": {"status": "complete"},
             "c2": {"status": "running" if running else ("complete" if terminal else "pending")}}
    checkpoints = {"sub-120B": ("b" * 64) if sealed_reports else None,
                   "120B": ("c" * 64) if sealed_reports else None}
    atomic_write_json(cfg.campaign_root / "queue_state.json",
                      {"plan_sha256": PLAN, "cells": cells, "report_checkpoints": checkpoints})


def test_selftest_green():
    assert act.selftest()["ok"] is True


def test_gate_refuses_running_campaign(tmp_path):
    cfg = _cfg(tmp_path)
    _write_queue(cfg, terminal=False, sealed_reports=False, running=True)
    gate = act.supersession_gate(cfg)
    assert not gate["all_pass"]
    assert gate["terminal"] is False
    assert act.activate(cfg, go=True)["activated"] is False


def test_gate_refuses_without_signature(tmp_path):
    cfg = _cfg(tmp_path)
    _write_queue(cfg, terminal=True, sealed_reports=True)
    gate = act.supersession_gate(cfg)
    assert gate["terminal"] and gate["reporter_sealed"]
    assert gate["signed"] is False
    assert not gate["all_pass"]


def test_signed_go_activates_then_rollback(tmp_path):
    cfg = _cfg(tmp_path)
    _write_queue(cfg, terminal=True, sealed_reports=True)
    sig = act.make_signature(PLAN, signed_by="operator", statement="supersede")
    atomic_write_json(cfg.signature_path, sig)
    gate = act.supersession_gate(cfg)
    assert gate["all_pass"], gate["reasons"]
    # without go -> refused
    assert act.activate(cfg, go=False)["activated"] is False
    # with go -> activated
    res = act.activate(cfg, go=True, artifacts={"plan_sha256": "x"})
    assert res["activated"] is True
    assert act.status(cfg)["active"] is True
    # rollback -> default off
    rb = act.rollback(cfg)
    assert rb["rolled_back"] is True


def test_gate_refuses_wrong_generation_even_when_signed(tmp_path):
    # regression: all_pass must be bound to the pinned plan_sha256, not just noted
    cfg = _cfg(tmp_path)
    cells = {"c1": {"status": "complete"}}
    atomic_write_json(cfg.campaign_root / "queue_state.json",
                      {"plan_sha256": "f" * 64,  # a DIFFERENT terminal generation
                       "cells": cells, "report_checkpoints": {"g": "b" * 64}})
    atomic_write_json(cfg.signature_path, act.make_signature(PLAN, signed_by="op", statement="x"))
    gate = act.supersession_gate(cfg)
    assert gate["plan_bound"] is False
    assert gate["all_pass"] is False
    assert act.activate(cfg, go=True)["activated"] is False


def test_empty_dict_checkpoint_not_accepted(tmp_path):
    # regression: an empty-dict checkpoint is not "accepted"
    cfg = _cfg(tmp_path)
    atomic_write_json(cfg.campaign_root / "queue_state.json",
                      {"plan_sha256": PLAN, "cells": {"c1": {"status": "complete"}},
                       "report_checkpoints": {"g1": {}, "g2": "b" * 64}})
    gate = act.supersession_gate(cfg)
    assert gate["reporter_sealed"] is True
    assert gate["checkpoint_accepted"] is False
    # a dict WITH a sha256 self-seal is accepted
    atomic_write_json(cfg.campaign_root / "queue_state.json",
                      {"plan_sha256": PLAN, "cells": {"c1": {"status": "complete"}},
                       "report_checkpoints": {"g1": {"checkpoint_sha256": "a" * 64}}})
    assert act.supersession_gate(cfg)["checkpoint_accepted"] is True


def test_wrong_plan_signature_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    _write_queue(cfg, terminal=True, sealed_reports=True)
    atomic_write_json(cfg.signature_path,
                      act.make_signature("d" * 64, signed_by="x", statement="wrong plan"))
    gate = act.supersession_gate(cfg)
    assert gate["signed"] is False


def test_tampered_signature_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    _write_queue(cfg, terminal=True, sealed_reports=True)
    sig = act.make_signature(PLAN, signed_by="operator", statement="ok")
    sig["statement"] = "changed after signing"  # breaks the seal
    atomic_write_json(cfg.signature_path, sig)
    gate = act.supersession_gate(cfg)
    assert gate["signed"] is False
