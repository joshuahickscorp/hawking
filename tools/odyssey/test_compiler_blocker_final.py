"""G023 blocker-chain pins."""
import json
from pathlib import Path

import pytest

RH = Path(__file__).resolve().parents[2] / "receipts/headless"
R = RH / "NOETIC_COMPILER_BLOCKER_FINAL.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="blocker chain not verified")


def rec():
    return json.load(open(R))


def test_four_of_five_admission_constants_already_match():
    d = rec()
    assert d["n_matching"] == 4 and d["n_constants"] == 5


def test_the_tensor_count_was_computed_not_assumed():
    """18,867 was named as a barrier in every earlier statement, mine included."""
    d = rec()
    c = next(x for x in d["runtime_admission_constants"]
             if x["constant"] == "QWEN30_COMPLETE_TENSOR_COUNT")
    assert c["matches"] is True
    assert "computed from the specimen's own config" in c["note"]
    assert "would not have changed anything" in d["the_tensor_count_is_not_a_barrier"]["consequence"]


def test_the_one_mismatch_is_a_string_not_a_capability():
    d = rec()["the_only_structural_mismatch"]
    assert d["constant"] == "QWEN30_REPOSITORY"
    assert "string equality" in d["kind"]
    assert d["enforced_at_n_sites"] >= 5


def test_the_packer_gap_this_receipt_found_is_now_closed():
    """The receipt records the gap as it stood. It has since been fixed, so the live
    module must now DISAGREE with that snapshot -- the fix landing, not a regression."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "headless"))
    import whole_model_native as w
    d = rec()["THE_ACTUAL_HOLE_IS_A_PACKER"]
    assert d["packer_handles_moe_expert"] is False, "the receipt should record the gap"
    assert "moe_expert" in w.GENOME, "the packer still cannot emit routed experts"
    assert w.organ_role(
        "model.layers.0.mlp.experts.101.gate_proj.weight") == "moe_expert"

def test_the_reframing_is_recorded_as_a_correction():
    d = rec()["THE_ACTUAL_HOLE_IS_A_PACKER"]
    assert "mine included" in d["why_this_reframes_it"]
    assert "nothing to admit" in d["why_this_reframes_it"]


def test_the_unblock_order_puts_the_packer_first():
    steps = rec()["revised_unblock_order"]
    assert len(steps) == 3
    assert "packer" in steps[0].lower()
    assert "QWEN30_REPOSITORY" in steps[1]


def test_the_unblock_reuses_the_representations_the_planner_selected():
    """Not a fresh design: the KernelPlanner already chose these."""
    steps = rec()["revised_unblock_order"]
    assert "conventional_low_bit" in steps[0]
    kp = json.load(open(RH / "KERNEL_PLANNER_MODEL2.json"))
    sel = {r["organ"]: r["selected_representation"] for r in kp["organ_plan"]}
    assert sel["moe_expert"] == "conventional_low_bit"
    assert sel["moe_router"] == "leftover_f32"


def test_it_says_why_it_was_not_attempted():
    assert "half-built packer" in rec()["why_not_attempted_here"]
