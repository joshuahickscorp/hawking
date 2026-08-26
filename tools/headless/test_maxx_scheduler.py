"""MAXX is only worth having if it can refuse, and heartbeats only if they have fired."""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import maxx_scheduler as mx
import maxx_heartbeat as hb

REPO = Path(__file__).resolve().parents[2]
M = REPO / "receipts/headless/MAXX_RESOURCE_PIPELINE.json"
H = REPO / "receipts/headless/LANE_HEARTBEATS.json"


def test_seven_independent_queues():
    d = json.load(open(M))
    assert d["n_queues"] == 7 and len(d["queues"]) == 7
    assert set(d["queues"]) == set(mx.QUEUES)


def test_protected_window_declines_contaminating_work():
    w = mx.ProtectedWindow("test")
    for q in mx.CONTAMINATES_GPU:
        ok, why = mx.admit(q, w)
        assert not ok and "forge" in why, q
    ok, _ = mx.admit("GPU_READY", w)
    assert ok, "the protected measurement's own queue must not be declined"


def test_everything_is_admitted_with_no_window():
    for q in mx.QUEUES:
        ok, _ = mx.admit(q, None)
        assert ok, q


def test_receipt_records_a_real_decline():
    d = json.load(open(M))["protected_window_demo"]
    assert d["n_declined"] >= 1
    assert d["all_admitted_when_no_window"] is True


def test_a_blocked_queue_stalls_nothing():
    d = json.load(open(M))["queue_independence"]
    assert d["blocked_queues"], "no queue is blocked, so independence is untested"
    assert d["still_ready_while_blocked"]
    assert d["a_blocked_queue_stalls_nothing"] is True


def test_objective_counts_only_evidence_that_exists():
    d = json.load(open(M))["objective"]["inputs"]
    assert d["measurable"] is True
    assert d["n_verified"] > 0
    assert d["n_cited_receipts_missing"] == 0, "a VERIFIED obligation cites a missing receipt"
    assert d["evidence_integrity"] == 1.0


def test_all_five_detectors_fired_on_an_injected_fault():
    d = json.load(open(H))
    assert len(d["injected_fault_proofs"]) == 5
    for p in d["injected_fault_proofs"]:
        assert p["detected"] is True, p["class"]
        assert p["control_clean"] is True, p["class"]


def test_a_worktree_of_unknown_dirtiness_is_never_auto_removed():
    d = json.load(open(H))
    for p in d["injected_fault_proofs"]:
        assert not p.get("auto_removed"), p["class"]
    for r in d.get("real_findings", {}).get("stale_worktree", []):
        if r["holds_uncommitted_work"] is not False:
            assert r["safe_to_auto_remove"] is False, r["worktree"]


def test_only_two_uncertainty_classes_may_block():
    d = json.load(open(H))
    assert set(d["only_these_may_block"]) == {"USER_PREFERENCE", "IRREVERSIBLE_OR_EXTERNAL"}
    for k, v in d["uncertainty_classes"].items():
        assert "resolve" in v


def test_heartbeats_carry_every_required_field():
    d = json.load(open(H))
    assert d["n_live_lanes"] > 0
    for lane in d["lanes"]:
        for f in d["heartbeat_fields"]:
            assert f in lane, (lane["lane"], f)
