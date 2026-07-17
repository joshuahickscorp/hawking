#!/usr/bin/env python3.12
"""Tests for the immutable campaign import (eco_import)."""
import pathlib
import sys

import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import eco_import as imp  # noqa: E402
from eco_common import EcoError, atomic_write_json, seal_field, hash_value, now_iso  # noqa: E402


def _build_campaign(tmp_path, *, running_cell=True, corrupt_result=False, wrong_plan=False):
    root = tmp_path / "doctor_v5_ultra"
    (root / "results" / "m-14b__2bpw__codec-control").mkdir(parents=True)
    (root / "dispositions").mkdir(parents=True)
    cid_c = "m-14b__2bpw__codec-control"
    cid_d = "m-14b__0p1bpw__doctor-full"
    id_c = hash_value({"cell": cid_c})
    id_d = hash_value({"cell": cid_d})
    plan = {
        "schema": "hawking.doctor_v5_ultra_campaign_plan.v1",
        "matrix": {"cells": 3}, "cohort": [{"label": "14B", "family": "qwen2.5-dense",
                                            "hf_id": "m/14b", "nominal_params_b": 14.0},
                                           {"label": "72B", "family": "qwen2.5-dense",
                                            "hf_id": "m/72b", "nominal_params_b": 72.0}],
        "cells": [
            {"cell_id": cid_c, "model_label": "14B", "model_family": "qwen2.5-dense",
             "hf_id": "m/14b", "rate_id": "2", "rate_bpw": 2.0, "branch": "codec_control",
             "cell_identity_sha256": id_c, "exact_stored_parameter_count": 14_000_000_000,
             "nominal_params_b": 14.0, "dependencies": []},
            {"cell_id": cid_d, "model_label": "14B", "model_family": "qwen2.5-dense",
             "hf_id": "m/14b", "rate_id": "0.1", "rate_bpw": 0.1, "branch": "doctor_full",
             "cell_identity_sha256": id_d, "exact_stored_parameter_count": 14_000_000_000,
             "nominal_params_b": 14.0, "dependencies": ["codec_control"]},
            {"cell_id": "m-72b__2bpw__codec-control", "model_label": "72B",
             "model_family": "qwen2.5-dense", "hf_id": "m/72b", "rate_id": "2", "rate_bpw": 2.0,
             "branch": "codec_control", "cell_identity_sha256": hash_value({"cell": "m-72b"}),
             "exact_stored_parameter_count": 72_000_000_000, "nominal_params_b": 72.0,
             "dependencies": []},
        ],
    }
    plan = seal_field(plan, "plan_sha256")
    atomic_write_json(root / "campaign_plan.json", plan)

    result = {
        "schema": imp.RESULT_SCHEMA, "status": "complete",
        "metrics": {
            "campaign_cell": {"cell_id": cid_c, "branch": "codec_control", "model_label": "14B",
                              "rate_id": "2", "cell_identity_sha256": id_c},
            "physical_accounting": {"all_in_model_payload_bpw": 2.41, "target_physical_bpw": 2.0,
                                    "target_met": False, "model_payload_bytes": 4_200_000_000,
                                    "packed_2d_tensor_bpw": 2.40,
                                    "lossless_non_2d_passthrough_bytes": 1_000_000,
                                    "full_bundle_bytes": 4_260_000_000},
            "quality_observation": {"ppl": {"relative_delta": 0.06, "baseline": 12.0},
                                    "capability": {"absolute_delta": -0.02},
                                    "quality_claims_permitted": False, "status": "provisional_unsealed"},
            "claims": {"dominance": False, "quality": False, "source_deletion": False,
                       "target_physical_rate_met": False},
        },
    }
    result = seal_field(result, "result_sha256")
    if corrupt_result:
        result["result_sha256"] = "0" * 64
    atomic_write_json(root / "results" / cid_c / "result.json", result)

    disp = {"schema": imp.DISPOSITION_SCHEMA, "status": "unsupported", "version": 1,
            "cell_id": cid_d, "cell_identity_sha256": id_d, "plan_sha256": plan["plan_sha256"],
            "reason_code": "empirical-quality-cliff-adaptive-defer", "detail": "toy",
            "quality_claims_permitted": False, "source_deletion_permitted": False,
            "evidence_artifacts": [], "recorded_at": now_iso()}
    disp = seal_field(disp, "disposition_sha256")
    atomic_write_json(root / "dispositions" / f"{cid_d}.json", disp)

    cells = {cid_c: {"status": "complete"}, cid_d: {"status": "unsupported"},
             "m-72b__2bpw__codec-control": {"status": "running" if running_cell else "pending"}}
    plan_sha = plan["plan_sha256"] if not wrong_plan else "0" * 64
    atomic_write_json(root / "queue_state.json", {"plan_sha256": plan_sha, "cells": cells})
    return root, plan["plan_sha256"]


def test_selftest_green():
    assert imp.selftest()["ok"] is True


def test_imports_terminal_skips_running(tmp_path):
    root, plan_sha = _build_campaign(tmp_path, running_cell=True)
    cfg = imp.ImportConfig(campaign_root=root, expected_plan_sha256=plan_sha)
    ledger = imp.build_ledger(cfg)
    assert ledger["terminal_imported"] == 2  # complete + disposition, NOT the running cell
    assert ledger["seal_validated"] == 2
    assert ledger["skipped_nonterminal"].get("running") == 1
    # cohort carried through so 72B is visible downstream
    labels = {m["label"] for m in ledger["cohort"]}
    assert {"14B", "72B"} <= labels


def test_seal_tamper_flagged(tmp_path):
    root, plan_sha = _build_campaign(tmp_path, corrupt_result=True)
    cfg = imp.ImportConfig(campaign_root=root, expected_plan_sha256=plan_sha)
    ledger = imp.build_ledger(cfg)
    graded = {c["cell_id"]: c for c in ledger["cells"]}
    bad = graded["m-14b__2bpw__codec-control"]
    assert bad["evidence_grade"] == "seal_invalid"
    assert "result self-seal invalid" in bad["seal_reasons"]
    # the disposition is still valid
    assert ledger["seal_validated"] == 1


def test_wrong_plan_refused(tmp_path):
    root, _ = _build_campaign(tmp_path, wrong_plan=True)
    cfg = imp.ImportConfig(campaign_root=root, expected_plan_sha256="a" * 64)
    with pytest.raises(EcoError, match="plan_sha256 mismatch"):
        imp.build_ledger(cfg)


def test_missing_status_does_not_break_seal(tmp_path):
    # regression: a queue cell with no 'status' key must not produce a None dict key that
    # crashes the ledger's canonical sort on seal.
    root, plan_sha = _build_campaign(tmp_path, running_cell=True)
    queue = imp.read_json_safe(root / "queue_state.json")
    queue["cells"]["orphan"] = {}  # no status
    atomic_write_json(root / "queue_state.json", queue)
    cfg = imp.ImportConfig(campaign_root=root, expected_plan_sha256=plan_sha)
    ledger = imp.build_ledger(cfg)  # must not raise
    assert "None" in ledger["skipped_nonterminal"]


def test_queue_plan_mismatch_refused(tmp_path):
    root, plan_sha = _build_campaign(tmp_path, wrong_plan=True)
    # expected matches the plan file, but the queue carries a different plan_sha256
    cfg = imp.ImportConfig(campaign_root=root, expected_plan_sha256=plan_sha)
    with pytest.raises(EcoError, match="queue_state plan_sha256"):
        imp.build_ledger(cfg)
