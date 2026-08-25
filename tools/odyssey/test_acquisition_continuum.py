"""G041 pins."""
import json
from pathlib import Path

import pytest

RH = Path(__file__).resolve().parents[2] / "receipts/headless"
R = RH / "ODYSSEY_ACQUISITION_CONTINUUM.json"
A = RH / "ARCHITECTURE_RECOGNIZER.json"
pytestmark = pytest.mark.skipif(not R.is_file(), reason="G041 receipt not built")


def rec():
    return json.load(open(R))


def test_queue_is_ahead_of_the_gpu():
    q = rec()["queue"]
    assert q["n_verified_resident"] >= 2
    assert q["n_in_flight"] >= 1
    assert q["n_metadata_resolved"] >= 1
    assert q["depth_satisfied"] is True


def test_capacity_guard_was_exercised_and_can_actually_refuse():
    """A gate that has never returned False is not known to work."""
    c = rec()["capacity"]["guard_exercised"]
    assert c["small_1_gib"]["admitted"] is True
    assert c["impossible_100_tb"]["admitted"] is False
    assert rec()["capacity"]["guard_can_refuse"] is True


def test_protected_headroom_is_refused_separately_from_the_budget():
    """A size can fit the 3.5 TB allocation and still be refused for eating headroom."""
    c = rec()["capacity"]["guard_exercised"]["would_breach_headroom"]
    assert c["admitted"] is False
    assert c["detail"]["headroom_ok"] is False


def test_free_space_is_measured_not_cached():
    c = rec()["capacity"]
    assert c["free_now_gib"] > 0
    assert "never trust a cached number" in c["rule"]


def test_every_resident_specimen_has_an_architecture_census():
    c = rec()["census_before_gpu"]
    assert c["resident_without_census"] == []
    assert c["satisfied"] is True
    assert len(c["censused"]) >= 3


def test_downloads_are_suspended_and_resumed_never_cancelled():
    n = rec()["non_blocking"]
    assert "resumed" in n["mechanism"]
    assert "never cancelled" in n["mechanism"]
    assert "file_download.py" in n["resumability_is_structural"]


# --- the recognizer bugs this obligation surfaced ---------------------------------

def test_vision_fingerprint_is_not_start_anchored():
    """Qwen3-VL nests its tower as model.visual.blocks.N; ^visual missed 327 tensors."""
    src = (Path(__file__).resolve().parent / "arch_recognizer.py").read_text()
    line = next(l for l in src.splitlines() if '"vision_encoder"' in l and "r\"" in l)
    assert "(^|" in line, "vision fingerprint is start-anchored again"


def test_mm_projector_is_tested_before_the_vision_tower():
    """The broad tower pattern swallows model.visual.merger.*; order decides."""
    src = (Path(__file__).resolve().parent / "arch_recognizer.py").read_text()
    assert src.index('("mm_projector"') < src.index('("vision_encoder"')


@pytest.mark.skipif(not A.is_file(), reason="census not built")
def test_the_bug_exposing_specimen_is_in_sample_not_held_out():
    """It found two fingerprint bugs, so scoring it as held-out would inflate the only
    out-of-sample number the receipt has."""
    d = json.load(open(A))
    ins = {s["result"]["repo"] for s in d["specimens"]}
    held = {s["result"]["repo"] for s in d["heldout_specimens"]}
    assert "Qwen/Qwen3-VL-30B-A3B-Instruct" in ins
    assert "Qwen/Qwen3-VL-30B-A3B-Instruct" not in held


@pytest.mark.skipif(not A.is_file(), reason="census not built")
def test_the_vl_specimen_is_fully_recognized_with_nothing_unmatched():
    d = json.load(open(A))
    s = next(x for x in d["specimens"]
             if x["result"]["repo"] == "Qwen/Qwen3-VL-30B-A3B-Instruct")
    assert s["result"]["n_unmatched"] == 0
    organs = {o["organ"] for o in s["result"]["organs"]}
    assert {"vision_encoder", "mm_projector", "moe_expert", "moe_router"} <= organs


@pytest.mark.skipif(not A.is_file(), reason="census not built")
def test_heldout_calibration_still_perfect_after_the_fingerprint_change():
    d = json.load(open(A))
    c = d["calibration_heldout"]
    assert c["precision"] == 1.0 and c["recall"] == 1.0
