"""Acceptance tests for Odyssey / HMF / Fusion gates.

Re-invoke each gate's own symbol. A receipt on disk is not the bar: the
numeric/hardware predicates here must fail if a verdict is flipped to ACCEPTED
while the measured condition still fails.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.acceptance.odyssey.run import (
    GATES,
    RECEIPTS_DIR,
    REPO,
    reduction_holds,
    run_all,
)

REQUIRED_RECEIPT_KEYS = (
    "schema",
    "gate",
    "verdict",
    "evidence_tier",
    "criterion",
    "invoked_symbol",
    "command",
    "run",
    "blocker",
    "criterion_altered",
)


@pytest.fixture(scope="module")
def campaign():
    return run_all()


@pytest.fixture(scope="module")
def receipts(campaign):
    out = {}
    for r in campaign["results"]:
        path = Path(r["path"])
        out[r["gate"]] = json.loads(path.read_text(encoding="utf-8"))
    return out


def test_runner_covers_exactly_the_assigned_gates(campaign):
    ids = [r["gate"] for r in campaign["results"]]
    assert ids == list(GATES)


def test_every_gate_wrote_a_receipt_with_the_required_shape(receipts):
    assert set(receipts) == set(GATES)
    for gate, doc in receipts.items():
        missing = [k for k in REQUIRED_RECEIPT_KEYS if k not in doc]
        assert missing == [], f"{gate} missing {missing}"
        assert doc["gate"] == gate
        assert doc["verdict"] in {"ACCEPTED", "BLOCKED"}
        assert doc["criterion_altered"] is False
        assert doc["invoked_symbol"]["kind"] == "call"
        assert doc["command"]
        assert doc["run"]
        quote = (doc.get("criterion") or {}).get("quote") or ""
        assert len(quote) > 40, f"{gate} criterion quote is empty"
        assert doc.get("seal_sha256")


def test_no_acceptance_criterion_was_altered():
    diff = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO),
            "diff",
            "--name-only",
            "HEAD",
            "--",
            "tools/roadmap",
            "tools/audit",
            "tools/theia",
            "hcli",
        ],
        text=True,
    ).strip()
    assert diff == "", f"forbidden path modified: {diff}"


def test_odyssey_i_calls_pick_acquire_candidate_not_just_imports(receipts):
    doc = receipts["ODYSSEY_I_DISCOVERY"]
    assert doc["invoked_symbol"]["symbol"] == "pick_acquire_candidate"
    picker = doc["run"]["pick_acquire_candidate"]
    assert picker["ok"] is True
    assert picker["mutate"] is False
    assert picker["mutated"] is False
    # A real selection from live state (O010 GLM-4.5-Air when this was authored).
    assert picker["n_patients"] >= 1
    assert picker["candidate_oxx"] or picker["n_skipped"] >= 1


def test_odyssey_i_censuses_two_architectures_when_lake_is_mounted(receipts):
    doc = receipts["ODYSSEY_I_DISCOVERY"]
    lake = doc["run"]["lake"]
    if not lake.get("mounted"):
        assert doc["verdict"] == "BLOCKED"
        assert doc["blocker"]["id"] == "MODELLAKE_MOUNTED"
        return
    types = doc["run"]["model_types_censused"]
    assert len(set(types)) >= 2
    census = doc["run"]["census"]
    q = census["qwen3_0.6b"]
    f = census["falcon_h1_7b"]
    assert q["total_params"] > 0 and f["total_params"] > 0
    assert q["model_type"] != f["model_type"]
    assert q["total_bytes"] > 0 and f["total_bytes"] > 0
    assert doc["verdict"] == "ACCEPTED"
    assert doc["blocker"] is None


def test_odyssey_ii_calls_load_qualification_queue(receipts):
    doc = receipts["ODYSSEY_II_TRANSFER"]
    assert doc["invoked_symbol"]["symbol"] == "load_qualification_queue"
    q = doc["run"]["load_qualification_queue"]
    assert q["ok"] is True
    assert q["n_candidates"] >= 1


def test_odyssey_ii_numeric_bar_rejects_negative_savings():
    """Mutation-style: the comparison itself, independent of the receipt."""
    assert reduction_holds(-8) is False
    assert reduction_holds(0) is False
    assert reduction_holds(None) is False
    assert reduction_holds(1) is True
    assert reduction_holds(3) is True


def test_odyssey_ii_is_blocked_because_measured_reduction_is_not_positive(receipts):
    doc = receipts["ODYSSEY_II_TRANSFER"]
    comps = doc["numeric_comparisons"]
    assert comps, "Odyssey II must record a numeric comparison"
    c = comps[0]
    assert c["field"] == "delta.evaluations_avoided"
    assert c["op"] == ">"
    assert c["threshold"] == 0
    measured = c["measured"]
    assert reduction_holds(measured) is False
    assert c["passed"] is False
    assert doc["verdict"] == "BLOCKED"
    assert doc["blocker"]["id"] == "TRANSFER_REDUCTION_NOT_POSITIVE"
    # The checked-in cold-vs-transfer receipt reports COLD winning: -8.
    assert measured == -8


def test_odyssey_iii_calls_scars(receipts):
    doc = receipts["ODYSSEY_III_ADVERSARIAL_META_SCIENCE"]
    assert doc["invoked_symbol"]["symbol"] == "scars"
    scars = doc["run"]["scars"]
    assert scars["n_scars"] >= 1
    assert scars["n_missing_regression_tests"] == 0


def test_odyssey_iii_synthetic_loop_is_not_acceptance(receipts):
    doc = receipts["ODYSSEY_III_ADVERSARIAL_META_SCIENCE"]
    st = doc["run"]["adversary_selftest"]
    assert st["synthetic_result"]["synthetic"] is True
    assert st["cpu_observation"]["physical_arm"] == "not_run"
    assert st["moved_down"] is True  # the loop works
    # Working on a synthetic verdict is not the gate.
    assert doc["verdict"] == "BLOCKED"
    assert doc["blocker"]["id"] == "NO_INDEPENDENT_LAW_REFUTATION"
    fired = st["cpu_observation"]["traps_fired"]
    assert "scale_invariance" in fired
    assert "skip_counted_as_pass" in fired


def test_hmf_probe_absent_so_not_accepted(receipts):
    doc = receipts["HMF_DEVICE_VISIBLE_TRUST"]
    hw = doc["run"]["hardware_probe"]
    assert hw["id"] == "HMF_PRESENT"
    assert hw["present"] is False
    overlay = doc["run"]["overlay"]
    # Overlay contract still holds on UMA — that is not device-visible HMF.
    assert overlay["clean_coherence"] == "COHERENT"
    assert overlay["after_unsynced_kernel_boundary"] == "UNKNOWN"
    assert overlay["boolean_collapse_refused"] is True
    assert doc["verdict"] == "BLOCKED"
    assert doc["blocker"]["id"] == "HMF_PRESENT"
    assert doc["blocker"]["wake_condition"] == "HMF_PRESENT"


def test_fusion_blocked_without_a_second_physical_domain(receipts):
    doc = receipts["FUSION_FIRST_HETEROGENEOUS_EXECUTABLE"]
    probes = doc["run"]["hardware_probes"]
    assert probes["U50_PRESENT"]["present"] is False
    assert probes["EGPU_PRESENT"]["present"] is False
    assert probes["DGX_PRESENT"]["present"] is False
    assert probes["HMF_PRESENT"]["present"] is False
    apple = doc["run"]["fusion_planner_apple_alone"]
    assert apple["n_domains"] == 1
    assert apple["domains"] == ["APPLE"]
    sim = doc["run"]["simulate_default"]
    assert sim["timing_decidable"] is False
    assert sim["speedup"] is None
    present = doc["run"]["fusion_sim_present_nodes"]
    assert present == ["APPLE"]
    assert doc["verdict"] == "BLOCKED"
    assert doc["blocker"]["id"] == "NO_SECOND_PHYSICAL_DOMAIN"
    assert "HMF_PRESENT" in doc["blocker"]["wake_condition"]


def test_manifest_count_matches_verdicts(campaign, receipts):
    man = json.loads(Path(campaign["manifest"]).read_text(encoding="utf-8"))
    accepted = [g for g, d in receipts.items() if d["verdict"] == "ACCEPTED"]
    blocked = [g for g, d in receipts.items() if d["verdict"] == "BLOCKED"]
    assert man["n_accepted"] == len(accepted) == campaign["n_accepted"]
    assert man["n_blocked"] == len(blocked) == campaign["n_blocked"]
    assert man["criterion_altered"] is False
    # Honest count for this hardware: Odyssey I only, if the lake is mounted.
    lake = receipts["ODYSSEY_I_DISCOVERY"]["run"]["lake"]
    if lake.get("mounted"):
        assert accepted == ["ODYSSEY_I_DISCOVERY"]
        assert campaign["n_accepted"] == 1
        assert campaign["n_blocked"] == 4


def test_flipping_hmf_to_accepted_would_fail_the_probe_predicate(receipts):
    """If someone stamps ACCEPTED while HMF_PRESENT is false, this test fails."""
    doc = receipts["HMF_DEVICE_VISIBLE_TRUST"]
    present = doc["run"]["hardware_probe"]["present"]
    if doc["verdict"] == "ACCEPTED":
        assert present is True, "HMF ACCEPTED while probe present=False is a fabricated pass"


def test_flipping_odyssey_ii_to_accepted_would_fail_the_numeric_bar(receipts):
    doc = receipts["ODYSSEY_II_TRANSFER"]
    measured = doc["numeric_comparisons"][0]["measured"]
    if doc["verdict"] == "ACCEPTED":
        assert reduction_holds(measured), (
            f"ODYSSEY_II_TRANSFER ACCEPTED with evaluations_avoided={measured!r}"
        )


def test_receipts_dir_is_the_acceptance_lane_not_future():
    assert RECEIPTS_DIR == REPO / "receipts" / "acceptance"
    diff = subprocess.check_output(
        ["git", "-C", str(REPO), "status", "--porcelain", "--", "receipts/future"],
        text=True,
    ).strip()
    assert diff == "", f"runner touched receipts/future: {diff}"
