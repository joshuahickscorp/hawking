#!/usr/bin/env python3.12
"""Tests for the release watcher + arming artifacts and the calibration precompiler."""
import json
import pathlib
import sys

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import succ_watch as w  # noqa: E402
import succ_calibrate as cal  # noqa: E402
from eco_common import atomic_write_json, repo_root  # noqa: E402


def test_watch_selftest():
    r = w.selftest()
    assert r["ok"] is True
    assert r["tick_waits"] and r["gate_blocked_no_intent"] and r["lease_refuses_concurrent"]
    assert r["intent_template_unsigned"] and r["launchd_written_not_installed"]


def test_calibrate_selftest():
    r = cal.selftest()
    assert r["ok"] is True
    assert r["release_bound"] is True


def test_watch_waits_and_blocks_without_intent(tmp_path):
    croot = tmp_path / "doctor_v5_ultra"
    croot.mkdir(parents=True)
    atomic_write_json(croot / "queue_state.json", {
        "plan_sha256": "a" * 64,
        "cells": {"c1": {"status": "complete"}, "c2": {"status": "running"}},
        "report_checkpoints": {"sub-120B": None, "120B": None}})
    r = w.watch_once(str(croot), successor_root=str(tmp_path / "s"),
                     lease_path=str(tmp_path / "s" / "w.lease"))
    assert r["tick"] is True
    assert r["state"] == "WAIT_OLD_RELEASE"
    assert r["gate"]["all_pass"] is False  # no armed intent


def test_intent_template_is_unsigned_and_bound(tmp_path):
    croot = tmp_path / "doctor_v5_ultra"
    croot.mkdir(parents=True)
    atomic_write_json(croot / "queue_state.json", {
        "plan_sha256": "f" * 64, "cells": {f"c{i}": {"status": "complete"} for i in range(5)},
        "report_checkpoints": {"sub-120B": None, "120B": None}})
    out = tmp_path / "intent.json"
    res = w.write_intent_template(str(croot), successor_repo=str(repo_root()), out_path=str(out))
    assert res["signed"] is False
    assert res["legacy_plan_sha256"] == "f" * 64
    doc = json.loads(out.read_text())
    # a template is NOT a valid intent (no signature/seal); the operator finalizes it
    assert doc["schema"] == w.INTENT_TEMPLATE_SCHEMA
    assert doc["signed"] is False
    assert doc["make_intent_fields"]["legacy_plan_sha256"] == "f" * 64
    assert doc["make_intent_fields"]["expected_terminal_count"] == 5


def test_launchd_plist_written_not_installed(tmp_path):
    out = tmp_path / "w.plist"
    res = w.write_launchd_plist(out_path=str(out), campaign_root=str(tmp_path / "cr"),
                                intent_path=str(tmp_path / "intent.json"))
    assert out.exists()
    assert "launchctl load" in res["install_command"]
    plist = out.read_text()
    assert w.WATCH_LABEL in plist and "watch" in plist


def test_calibration_release_bound_and_deferred(tmp_path):
    # a synthetic ledger with a sealed 72B codec_control whose quality is deferred
    import eco_import
    ledger = {"schema": "hawking.eco.prior_ledger.v1", "campaign_plan_sha256": "c" * 64,
              "ledger_sha256": "d" * 64, "cohort": [],
              "cells": [{"model_label": "72B", "model_family": "qwen2.5-dense", "hf_id": "x",
                         "exact_stored_parameter_count": int(72.7e9), "nominal_params_b": 72.7,
                         "rate_id": "4", "rate_bpw": 4.0, "branch": "codec_control",
                         "status": "complete", "result_sha256": "a" * 64,
                         "cell_identity_sha256": "b" * 64,
                         "physical": {"all_in_model_payload_bpw": 5.08, "target_physical_bpw": 4.0,
                                      "target_met": False},
                         "quality_provisional": {"ppl_relative_delta": None,
                                                 "capability_absolute_delta": None,
                                                 "quality_claims_permitted": False}}]}
    real = eco_import.build_ledger
    eco_import.build_ledger = lambda cfg: ledger
    try:
        prog = cal.build_calibration("72B", admission={"adapter_id": "x", "ready_for_execution": False,
                                                       "execution_capable": True, "blockers": []})
    finally:
        eco_import.build_ledger = real
    assert prog["release_binding"]["executes"] == "post_release_only"
    assert prog["untreated_frontier"][0]["quality_status"] == "deferred_disk_ram_gated"
    assert any(e["kind"] == "deferred_full_model_quality_eval" for e in prog["ordered_experiments"])
