#!/usr/bin/env python3.12
"""Tests for the unified CLI surface (eco_cli)."""
import io
import json
import pathlib
import sys
from contextlib import redirect_stdout

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import eco_cli  # noqa: E402
from eco_common import atomic_write_json, seal_field, hash_value, now_iso  # noqa: E402


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = eco_cli.main(argv)
    return rc, buf.getvalue()


def _mini_campaign(tmp_path):
    root = tmp_path / "doctor_v5_ultra"
    (root / "results" / "m-14b__2bpw__codec-control").mkdir(parents=True)
    (root / "dispositions").mkdir(parents=True)
    cid = "m-14b__2bpw__codec-control"
    idc = hash_value({"cell": cid})
    plan = seal_field({
        "schema": "hawking.doctor_v5_ultra_campaign_plan.v1", "matrix": {"cells": 1},
        "cohort": [{"label": "14B", "family": "qwen2.5-dense", "hf_id": "m/14b",
                    "nominal_params_b": 14.0}],
        "cells": [{"cell_id": cid, "model_label": "14B", "model_family": "qwen2.5-dense",
                   "hf_id": "m/14b", "rate_id": "2", "rate_bpw": 2.0, "branch": "codec_control",
                   "cell_identity_sha256": idc, "exact_stored_parameter_count": 14_000_000_000,
                   "nominal_params_b": 14.0, "dependencies": []}],
    }, "plan_sha256")
    atomic_write_json(root / "campaign_plan.json", plan)
    result = seal_field({
        "schema": "hawking.doctor_v5_adapter_result.v1", "status": "complete",
        "metrics": {"campaign_cell": {"cell_id": cid, "branch": "codec_control",
                                      "model_label": "14B", "rate_id": "2",
                                      "cell_identity_sha256": idc},
                    "physical_accounting": {"all_in_model_payload_bpw": 2.41,
                                            "target_physical_bpw": 2.0, "target_met": False,
                                            "model_payload_bytes": 4_200_000_000,
                                            "packed_2d_tensor_bpw": 2.4,
                                            "lossless_non_2d_passthrough_bytes": 1_000_000,
                                            "full_bundle_bytes": 4_260_000_000},
                    "quality_observation": {"ppl": {"relative_delta": 0.06, "baseline": 12.0},
                                            "capability": {"absolute_delta": -0.02},
                                            "quality_claims_permitted": False,
                                            "status": "provisional_unsealed"},
                    "claims": {"dominance": False, "quality": False, "source_deletion": False,
                               "target_physical_rate_met": False}},
    }, "result_sha256")
    atomic_write_json(root / "results" / cid / "result.json", result)
    atomic_write_json(root / "queue_state.json",
                      {"plan_sha256": plan["plan_sha256"], "cells": {cid: {"status": "complete"}}})
    return root, plan["plan_sha256"]


def test_selftest_all_green():
    rc, out = _run(["selftest"])
    assert rc == 0
    assert json.loads(out)["all_ok"] is True


def test_pipeline_valid():
    rc, out = _run(["pipeline"])
    assert rc == 0
    assert json.loads(out)["valid"] is True


def test_passport_selftest():
    rc, out = _run(["passport", "--selftest"])
    assert rc == 0
    assert json.loads(out)["ok"] is True


def test_admission_no_prior():
    rc, out = _run(["admission"])
    assert rc == 0
    assert "120B" in json.loads(out)["admissible_now"]


def test_import_and_plan_and_materialize_against_mini_campaign(tmp_path, monkeypatch):
    root, plan_sha = _mini_campaign(tmp_path)
    monkeypatch.setattr("eco_common.CAMPAIGN_PLAN_SHA256", plan_sha)
    # the import module read the constant at import time into its ImportConfig default; pass explicitly
    import eco_import
    monkeypatch.setattr(eco_import, "CAMPAIGN_PLAN_SHA256", plan_sha)

    rc, out = _run(["import", "--campaign-root", str(root)])
    assert rc == 0
    assert json.loads(out)["terminal_imported"] == 1

    rc, out = _run(["plan", "--campaign-root", str(root)])
    assert rc == 0
    plan = json.loads(out)
    assert any(p["label"] == "14B" for p in plan["parents"])

    out_dir = tmp_path / "materialized"
    rc, out = _run(["materialize", "--campaign-root", str(root), "--out-dir", str(out_dir)])
    assert rc == 0
    manifest = json.loads(out)
    assert manifest["parents_with_evidence"] == 1
    assert (out_dir / "adaptive_plan.json").exists()
    assert (out_dir / "prior_ledger.json").exists()
    assert (out_dir / "admission_plan.json").exists()


def test_activation_gate_refuses_running(tmp_path, monkeypatch):
    root = tmp_path / "doctor_v5_ultra"
    root.mkdir(parents=True)
    atomic_write_json(root / "queue_state.json", {
        "plan_sha256": "a" * 64,
        "cells": {"c1": {"status": "running"}}, "report_checkpoints": {"g": None}})
    import eco_activation
    monkeypatch.setattr(eco_activation, "CAMPAIGN_PLAN_SHA256", "a" * 64)
    rc, out = _run(["activation", "gate", "--campaign-root", str(root)])
    assert rc == 0
    assert json.loads(out)["all_pass"] is False
