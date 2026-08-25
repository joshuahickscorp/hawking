"""G023 stage-audit pins. A pipeline that grades its own automation is the failure
mode §102 names."""
import json, re
from pathlib import Path

import pytest

RH = Path(__file__).resolve().parents[2] / "receipts/headless"
R = RH / "NOETIC_COMPILER_STAGE_AUDIT.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="stage audit not built")


def rec():
    return json.load(open(R))


def test_the_overstatement_the_audit_found_is_now_resolved():
    """The audit originally found KernelPlanner recorded AUTOMATIC while its receipt
    never named model #2. It was then actually run, so the counts should agree."""
    d = rec()
    assert d["overstated_stages"] == []
    assert d["audited_automatic_on_model_2"] == d["recorded_automatic"]


def test_kernel_planner_now_names_model_2():
    ks = next(s for s in rec()["stages"] if s["stage"] == "KernelPlanner")
    assert ks["names_model_2"] is True
    assert ks["receipt"] == "KERNEL_PLANNER_MODEL2.json"


def test_moe_organ_kernels_are_now_catalogued():
    """The audit's original finding was NONE; registering them is what fixed it."""
    d = rec()["overstatement"]
    assert d["moe_organ_kernels_in_the_library"] != "NONE"
    assert any("moe" in o for o in d["moe_organ_kernels_in_the_library"])


def test_the_stages_that_did_run_declare_model_2_as_their_specimen():
    for s in rec()["stages"]:
        if s["audited_status"] == "AUTOMATIC_ON_MODEL_2":
            assert s["names_model_2"] is True, s["stage"]


def test_no_stage_produced_bytes():
    d = rec()
    assert all(s["produced_bytes"] is False for s in d["stages"])
    assert "PLANNING pipeline" in d["no_stage_produced_bytes"]["finding"]


def test_the_understatement_is_backed_by_files_that_exist():
    d = rec()["understatement"]["what_actually_exists"]
    root = Path(__file__).resolve().parents[2]
    assert (root / d["reader"]).is_file()
    assert d["reader_lines"] > 5000
    assert d["n_moe_expert_kernels"] >= 5
    assert d["moe_shader"]


def test_the_moe_kernels_are_named_not_merely_counted():
    d = rec()["understatement"]["what_actually_exists"]
    assert all(k.startswith("qwen30_expert_table_") for k in d["moe_expert_kernels"])


def test_the_revised_gap_is_smaller_than_both_previous_statements():
    d = rec()["understatement"]
    assert len(d["revised_gap"]) >= 3
    assert "18_867" in " ".join(d["revised_gap"])
    assert "smaller" in d["size"]


def test_the_pipeline_receipt_records_how_the_stage_was_fixed():
    p = json.load(open(RH / "NOETIC_COMPILER_PIPELINE.json"))
    ks = next(s for s in p["stages"] if s["stage"] == "KernelPlanner")
    assert ks["status"] == "AUTOMATIC"
    assert "never named model #2" in ks["why"]
    assert p["n_stages_automatic_on_model_2"] == 6
    assert "no bytes are packed" in p["kernel_planner_note"]


def test_the_obligation_remains_blocked_with_named_unmet_clauses():
    d = rec()
    assert d["still_blocked"] is True
    assert len(d["unmet_acceptance"]) == 3
