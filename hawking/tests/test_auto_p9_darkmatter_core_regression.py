import json
import re
from pathlib import Path

PLAN_PATH = (
    Path(__file__).resolve().parents[2]
    / "research"
    / "hawking-experiments"
    / "frankenstein"
    / "data"
    / "FRANKENSTEIN_PROGRAM_PLAN.json"
)

PINNED_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def _load_plan():
    with PLAN_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _p9_core(plan):
    return plan["fusion_mechanism"]["p9_darkmatter_core"]


def test_p9_darkmatter_core_declares_admission_rule():
    core = _p9_core(_load_plan())
    assert core["unpinned_marker"] == "PIN_AT_DOWNLOAD_TIME"
    assert core["blocked_reason"] == "unpinned_donor_revision"
    assert core["required_donor_roles"] == [
        "deepseek_v4_flash",
        "glm_5_2_stage1_donor",
        "kimi_k3",
        "qwen3_coder_30b_a3b",
    ]


def test_p9_darkmatter_core_blocks_unpinned_donor_revision():
    plan = _load_plan()
    core = _p9_core(plan)
    pins = plan["model_card_pins"]
    unpinned = [
        role
        for role in core["required_donor_roles"]
        if not PINNED_REVISION_RE.match(pins[role]["revision"])
    ]
    assert unpinned == ["qwen3_coder_30b_a3b"], unpinned
    assert core["blocked_reason"] == "unpinned_donor_revision"


def test_p9_darkmatter_core_admits_when_all_donors_pinned():
    plan = _load_plan()
    core = _p9_core(plan)
    pins = plan["model_card_pins"]
    pinned_plan = json.loads(json.dumps(plan))
    for role in core["required_donor_roles"]:
        pinned_plan["model_card_pins"][role]["revision"] = "0" * 40
    unpinned = [
        role
        for role in core["required_donor_roles"]
        if not PINNED_REVISION_RE.match(pinned_plan["model_card_pins"][role]["revision"])
    ]
    assert unpinned == []
    assert pins["deepseek_v4_flash"]["revision"] == "60d8d70770c6776ff598c94bb586a859a38244f1"