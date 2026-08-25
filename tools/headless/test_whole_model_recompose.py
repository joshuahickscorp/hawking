"""N041 baseline self-check: the closure sum is arithmetically correct and the receipt is honest."""
import json, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parents[2] / "receipts" / "headless"
sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_recompose_runs_and_is_generated():
    import whole_model_recompose as w
    assert w.main() == 0
    d = json.loads((R / "WHOLE_MODEL_RECOMPOSE.json").read_text())
    assert d["hand_authored"] is False


def test_complete_ebpw_is_the_weighted_closure():
    d = json.loads((R / "WHOLE_MODEL_RECOMPOSE.json").read_text())
    total = d["parent_parameter_count"]
    # complete EBPW must equal sum(organ_bits)/total_params to floating tolerance
    bits = sum(a["organ_bits"] for a in d["allocation"])
    assert abs(d["current_qwen_complete_ebpw"] - bits / total) < 1e-6


def test_mlp_confirmed_others_provisional():
    d = json.loads((R / "WHOLE_MODEL_RECOMPOSE.json").read_text())
    by = {a["organ"]: a for a in d["allocation"]}
    assert by["mlp"]["floor_status"] == "CONFIRMED"
    assert by["mlp"]["ebpw"] == 2.25
    # the non-MLP organs are either PROVISIONAL (pre-N040) or MEASURED (after N040 lands).
    for name in ("deltanet", "attention_gqa", "embedding", "output"):
        assert by[name]["floor_status"] in ("PROVISIONAL", "MEASURED")


def test_sub3_claim_matches_the_number():
    d = json.loads((R / "WHOLE_MODEL_RECOMPOSE.json").read_text())
    assert d["below_3_0"] == (d["current_qwen_complete_ebpw"] < 3.0)
