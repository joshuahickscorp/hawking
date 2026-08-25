"""G033 regression pins.

The defect these exist for: `complete_ebpw` was computed from hardcoded per-organ
rates and published under a physical name, so adding 1,288,519,664 bytes to an
artifact left the reported density unmoved at 2.596957 while the real physical
figure was 2.980217.
"""
import json
from pathlib import Path

import pytest

RH = Path(__file__).resolve().parents[2] / "receipts/headless"
AUDIT = RH / "PHYSICAL_METRIC_AUDIT.json"
CANARY = RH / "METRIC_MUTATION_CANARIES.json"

pytestmark = pytest.mark.skipif(not AUDIT.is_file() or not CANARY.is_file(),
                                reason="G033 receipts not built")


def _audit():
    return json.load(open(AUDIT))


def _canary():
    return json.load(open(CANARY))


def test_every_metric_is_classified_by_where_its_value_comes_from():
    a = _audit()
    for m in a["metrics"]:
        assert m["classification"].startswith(
            ("DESIGN_EXPECTED_", "ARTIFACT_PHYSICAL_", "RUNTIME_MEASURED_")), m
        assert m["source_of_value"].strip(), m


def test_all_three_classes_are_populated():
    """An audit that finds only one class is not an audit."""
    a = _audit()
    assert a["n_design_expected"] >= 1
    assert a["n_artifact_physical"] >= 3
    assert a["n_runtime_measured"] >= 3


def test_complete_ebpw_is_physical_not_design():
    a = _audit()
    row = next(m for m in a["metrics"] if m["metric"] == "complete_ebpw")
    assert row["classification"] == "ARTIFACT_PHYSICAL_complete_ebpw"
    assert "payload_bytes" in row["source_of_value"]


def test_metrics_that_are_still_frozen_are_named_not_hidden():
    """active_ebpw_per_token is still a design constant. Say so out loud."""
    a = _audit()
    assert "active_ebpw_per_token" in a["still_frozen_and_flagged"]
    row = next(m for m in a["metrics"] if m["metric"] == "active_ebpw_per_token")
    assert row["classification"].startswith("DESIGN_EXPECTED_")


def test_five_canaries_all_executed_and_passed():
    c = _canary()
    names = {r["canary"][0] for r in c["canaries"]}
    assert names == {"A", "B", "C", "D", "E"}
    assert c["n"] == 5 and c["n_passed"] == 5
    assert c["pass"] is True


def test_canaries_A_and_B_are_exact_not_merely_directional():
    """'went up' is not enough. The delta must equal 8*bytes/params exactly."""
    c = _canary()
    for r in c["canaries"]:
        if r["canary"][0] in ("A", "B"):
            assert r["exact"] is True, r


def test_canary_C_proves_a_genome_edit_cannot_move_a_physical_number():
    c = _canary()
    r = next(x for x in c["canaries"] if x["canary"].startswith("C_"))
    assert r["before"] == r["after"]


def test_canary_D_surfaces_the_design_physical_mismatch():
    """This is the exact shape of the original defect."""
    c = _canary()
    r = next(x for x in c["canaries"] if x["canary"].startswith("D_"))
    assert r["physical_after"] > r["physical_before"]
    assert r["design_equals_original"] is True
    assert r["mismatch_surfaced"] is True


def test_canary_E_states_what_it_did_not_do():
    """E verifies the fallback channel exists; it does not induce a fallback."""
    c = _canary()
    r = next(x for x in c["canaries"] if x["canary"].startswith("E_"))
    assert r["not_a_python_literal"] is True
    assert "does not INDUCE" in r["honest_limitation"]


def test_canaries_left_the_clone_at_its_baseline():
    c = _canary()
    assert c["restore_check"]["restored_to_baseline"] is True
    assert c["restore_check"]["baseline_bytes"] == c["restore_check"]["final_bytes"]


def test_receipts_are_machine_generated():
    for r in (_audit(), _canary()):
        assert r["hand_authored"] is False
        assert r["generated_by"] == "tools/odyssey/metric_audit.py"
