import json

import pytest

from tools.future import tournament as tn
from tools.future._common import RECEIPTS, HardwareClaimError, _assert_no_hardware_claims


def _scores(**overrides: float) -> dict[str, float]:
    return tn._demo_scores(**overrides)


def _pass_gates(**flips: bool) -> dict[str, bool]:
    gates = {gid: True for gid in tn.HARD_GATE_IDS}
    gates.update(flips)
    return gates


def test_build_emits_sealed_receipt():
    out = tn.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "TOURNAMENT_READINESS.json"
    assert doc["schema"] == tn.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]
    _assert_no_hardware_claims(doc)


def test_selftest_emits_sealed_receipt():
    out = tn.selftest()
    doc = json.loads(out.read_text())
    assert doc["seal_sha256"]
    assert doc["readiness"]["can_run"] is False


def test_can_run_false_against_real_flash_nx():
    """Negative control: the refusal must fire on the real current receipts."""
    flash_doc, source = tn.load_repo_json(tn.FLASH_NX_REL)
    assert flash_doc is not None, (
        f"Flash NX receipt {tn.FLASH_NX_REL} was not locatable from this sparse "
        "worktree, git HEAD, or the main checkout — cannot prove the metadata-only refusal"
    )
    assert flash_doc["status"] == "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION", flash_doc.get("status")
    assert source

    ok, reasons = tn.can_run()
    assert ok is False
    blob = "\n".join(reasons)
    assert "FLASH_SINGULARITY.NX" in blob
    assert "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION" in blob
    assert any("QWEN27_SINGULARITY.NX" in r for r in reasons)

    inspections = tn.inspect_contenders()
    assert inspections[tn.FLASH_ID]["complete_nx"] is False
    assert inspections[tn.QWEN_ID]["complete_nx"] is False
    assert inspections[tn.FLASH_ID]["status"] == "SEALED_METADATA_ONLY_NOT_FOR_PROMOTION"


def test_winner_refuses_premature_collapse():
    """Negative control: mixed axes must not name a single winner."""
    result = tn.winner([
        {"id": "A", "scores": _scores(accepted_tps=30.0, complete_ebpw=4.0)},
        {"id": "B", "scores": _scores(accepted_tps=10.0, complete_ebpw=2.0)},
    ])
    assert result["unique_winner"] is None, result
    assert result["non_dominated"] == ["A", "B"], result
    assert result["scalar_collapsed"] is False


def test_winner_names_unique_on_genuine_dominance():
    a = {"id": "A", "scores": _scores(accepted_tps=30.0, complete_ebpw=2.0)}
    b = {"id": "B", "scores": _scores(accepted_tps=10.0, complete_ebpw=4.0)}
    result = tn.winner([a, b])
    assert result["unique_winner"] == "A", result
    assert result["non_dominated"] == ["A"]


def test_hard_gate_failure_disqualifies_regardless_of_speed():
    fast_incoherent = {
        "id": "fast",
        "scores": _scores(accepted_tps=100.0),
        "hard_gates": _pass_gates(**{"capability.json-answer": False}),
    }
    slow_capable = {
        "id": "slow",
        "scores": _scores(accepted_tps=1.0),
        "hard_gates": _pass_gates(),
    }
    result = tn.winner([fast_incoherent, slow_capable])
    assert "fast" in result["disqualified"]
    assert "capability.json-answer" in result["disqualified"]["fast"]
    assert result["eligible"] == ["slow"]
    assert result["unique_winner"] == "slow"
    assert result["non_dominated"] == ["slow"]


def test_faster_than_control_is_not_success():
    verdict = tn.interpret_versus_control({"id": "X", "scores": _scores(accepted_tps=100.0)})
    assert verdict["success"] is None
    assert verdict["beating_control_is_not_success"] is True
    assert verdict["role_of_control"] == "CONTROL_NOT_TARGET_NOT_CEILING"
    assert tn.INCUMBENT_CONTROL["success_predicate"] is None


def test_slower_than_control_is_not_failure():
    slow = {"id": "slow", "scores": _scores(accepted_tps=1.0)}
    verdict = tn.interpret_versus_control(slow)
    assert verdict["success"] is None
    assert verdict["falling_short_of_control_is_not_failure"] is True
    # Still eligible for Pareto against an equal peer.
    peer = {"id": "peer", "scores": _scores(accepted_tps=1.0)}
    result = tn.winner([slow, peer])
    assert result["unique_winner"] is None
    assert set(result["non_dominated"]) == {"slow", "peer"}


def test_scalar_score_raises():
    with pytest.raises(tn.ScalarCollapseError):
        tn.scalar_score({"id": "A", "scores": _scores()})


def test_run_refuses_today():
    with pytest.raises(tn.TournamentNotReady) as caught:
        tn.run()
    assert caught.value.reasons
    assert any("FLASH_SINGULARITY.NX" in r for r in caught.value.reasons)


def test_common_profile_identical_for_both():
    profile = tn.common_profile()
    assert profile["identical_for"] == [tn.FLASH_ID, tn.QWEN_ID]
    assert profile["designed_before_either_complete_nx"] is True
    assert profile["no_scalar"] is True
    gate_ids = [g["id"] for g in profile["hard_gates"]]
    assert gate_ids == list(tn.HARD_GATE_IDS)
    axis_names = [a["name"] for a in profile["scored_axes"]]
    assert axis_names == list(tn.AXIS_NAMES)
    assert all(a["value"] is None for a in profile["scored_axes"])
    assert profile["no_anchoring"]["beating_control_is_not_success"] is True


def test_energy_axis_dropped_unless_trustworthy():
    a = {
        "id": "A",
        "energy_trustworthy": False,
        "scores": _scores(),
    }
    b = {
        "id": "B",
        "energy_trustworthy": False,
        "scores": {**_scores(), "energy_joules_per_token": 0.1},
    }
    a["scores"]["energy_joules_per_token"] = 9.9
    result = tn.winner([a, b])
    assert "energy_joules_per_token" not in result["axes"]
    assert result["unique_winner"] is None
    assert result["non_dominated"] == ["A", "B"]

    a2 = {"id": "A", "energy_trustworthy": True, "scores": {**_scores(), "energy_joules_per_token": 0.1}}
    b2 = {"id": "B", "energy_trustworthy": True, "scores": {**_scores(), "energy_joules_per_token": 9.9}}
    trusted = tn.winner([a2, b2])
    assert "energy_joules_per_token" in trusted["axes"]
    assert trusted["unique_winner"] == "A"


def test_missing_axis_prevents_dominance():
    a = {"id": "A", "scores": _scores()}
    b_scores = _scores(accepted_tps=1.0)
    del b_scores["complete_ebpw"]
    b = {"id": "B", "scores": b_scores}
    result = tn.winner([a, b])
    assert result["unique_winner"] is None
    assert set(result["non_dominated"]) == {"A", "B"}


def test_receipt_does_not_treat_control_as_target():
    doc = json.loads(tn.build().read_text())
    control = doc["incumbent_control"]
    assert control["role"] == "CONTROL_NOT_TARGET_NOT_CEILING"
    assert control["beating_control_is_not_success"] is True
    assert control["success_predicate"] is None
    assert doc["readiness"]["can_run"] is False
    assert doc["readiness"]["headline"] == "NO"


def test_hardware_claim_guard_still_armed():
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims({"accepted_tps": 25.0})
