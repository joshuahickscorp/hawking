"""ODYSSEY_BOOTSTRAP_GOLDEN: pin the launch, refuse a fake, prove recovery ran.

A golden point whose launch receipt is missing, or whose gate is not 16/16,
must not seal. A guard nobody has watched fail is not a guard.

Sparse-checkout trap: presence is git-at-the-pin, never "the file is missing
from this worktree". Recovery re-evaluates the sealed 16/16 at that commit;
it does not re-run evaluate_launch_criteria (those evaluators rewrite sibling
receipts).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.future import bootstrap_golden as bg
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, _assert_no_hardware_claims


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _launching_stub(**overrides) -> dict:
    verdict = {
        "allowed": True,
        "verdict": "LAUNCH",
        "n_criteria": 16,
        "n_met": 16,
        "n_unmet": 0,
        "met": list(range(16)),
        "unmet": [],
        "rule": "test",
    }
    doc = {
        "schema": "hawking.future.odyssey_i_launch.v1",
        "phase_transition": "STARTED",
        "verdict": verdict,
        "gpu_authority": False,
    }
    doc.update(overrides)
    if "verdict" in overrides and isinstance(overrides["verdict"], dict):
        merged = dict(verdict)
        merged.update(overrides["verdict"])
        doc["verdict"] = merged
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    doc["seal_sha256"] = _sha(json.dumps(body, sort_keys=True, separators=(",", ":")).encode())
    return doc


def test_refuses_missing_launch_receipt():
    with pytest.raises(bg.GoldenSealError, match="missing"):
        bg.require_launching_receipt(None)


def test_refuses_gate_not_16_16():
    doc = _launching_stub(verdict={"n_met": 15, "n_unmet": 1, "verdict": "REFUSED", "allowed": False, "unmet": ["gravity_callable"]})
    with pytest.raises(bg.GoldenSealError, match="16/16"):
        bg.require_launching_receipt(doc)


def test_refuses_wrong_launch_seal_even_when_16_16():
    """A different 16/16 receipt is a successor candidate, not this golden."""
    doc = _launching_stub()
    assert doc["seal_sha256"] != bg.LAUNCH_SEAL_SHA256
    with pytest.raises(bg.GoldenSealError, match="not the sealed launch"):
        bg.require_launching_receipt(doc)


def test_refuses_tampered_seal():
    real = bg.load_json_at(bg.LAUNCH_COMMIT, bg.LAUNCH_RECEIPT_REL)
    assert real is not None
    tampered = dict(real)
    tampered["seal_sha256"] = "0" * 64
    with pytest.raises(bg.GoldenSealError, match="does not recompute"):
        bg.require_launching_receipt(tampered)


def test_build_refuses_when_launch_receipt_missing(monkeypatch, tmp_path):
    real = bg.load_json_at

    def fake(commit, rel):
        if rel == bg.LAUNCH_RECEIPT_REL:
            return None
        return real(commit, rel)

    monkeypatch.setattr(bg, "load_json_at", fake)
    with pytest.raises(bg.GoldenSealError, match="missing"):
        bg.build(writer=lambda n, d, r: tmp_path / n)


def test_build_refuses_when_gate_is_not_16_16(monkeypatch, tmp_path):
    real = bg.load_json_at

    def fake(commit, rel):
        doc = real(commit, rel)
        if rel == bg.LAUNCH_RECEIPT_REL and isinstance(doc, dict):
            doc = dict(doc)
            doc["verdict"] = {
                **dict(doc.get("verdict") or {}),
                "n_met": 15,
                "n_unmet": 1,
                "verdict": "REFUSED",
                "allowed": False,
                "unmet": ["gravity_callable"],
            }
            return doc
        return doc

    monkeypatch.setattr(bg, "load_json_at", fake)
    with pytest.raises(bg.GoldenSealError, match="16/16"):
        bg.build(writer=lambda n, d, r: tmp_path / n)


def test_real_launch_receipt_at_pin_is_the_launching_one():
    doc = bg.load_json_at(bg.LAUNCH_COMMIT, bg.LAUNCH_RECEIPT_REL)
    accepted = bg.require_launching_receipt(doc)
    assert accepted["seal_sha256"] == bg.LAUNCH_SEAL_SHA256
    assert accepted["verdict"]["n_met"] == 16
    assert accepted["verdict"]["n_unmet"] == 0
    assert accepted["verdict"]["verdict"] == "LAUNCH"
    assert accepted["phase_transition"] == "STARTED"


def test_pin_path_uses_git_not_worktree_absence():
    """resident.py is ABSENT in git at the pin, not because this checkout is sparse."""
    absent = bg.pin_path(bg.LAUNCH_COMMIT, "hcli/agentos/resident.py")
    assert absent["status"] == "ABSENT"
    assert absent["sha256"] is None
    assert "ls-tree" in (absent["reason"] or "")
    present = bg.pin_path(bg.LAUNCH_COMMIT, "hcli/agentos/resident_gate.py")
    assert present["status"] == "PRESENT"
    assert present["sha256"]
    assert present["bytes"] and present["bytes"] > 0
    # Gravity owned files may not be materialized in a sparse worktree and
    # MUST still pin PRESENT from git. Disk existence is not the authority.
    gravity = bg.pin_path(bg.LAUNCH_COMMIT, "tools/odyssey/decoding_gravity.py")
    assert gravity["status"] == "PRESENT"
    assert gravity["sha256"]


@pytest.fixture(scope="module")
def sealed():
    out = bg.build()
    return out


def test_entry_point_seals_receipt(sealed):
    path = Path(sealed["path"])
    assert path.parent == RECEIPTS
    assert path.name == bg.RECEIPT
    doc = json.loads(path.read_text())
    assert doc["schema"] == bg.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert _sha(blob) == doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["measurement_class"] == "STATIC_ONLY"
    assert doc["not_a_development_freeze"] is True
    assert doc["no_era_vi"] is True
    assert doc["no_odyssey_iv"] is True
    _assert_no_hardware_claims(doc)


def test_commit_and_launch_seal_are_pinned_together(sealed):
    doc = sealed["doc"]
    assert doc["commit"]["sha"] == bg.LAUNCH_COMMIT
    assert doc["commit"]["tree"] == bg.LAUNCH_TREE
    assert doc["launch_receipt"]["seal_sha256"] == bg.LAUNCH_SEAL_SHA256
    assert doc["launch_receipt"]["pinned_together"] is True
    assert doc["launch_receipt"]["path"] == bg.LAUNCH_RECEIPT_REL
    assert doc["launch_receipt"]["verdict"]["n_met"] == 16
    assert doc["launch_receipt"]["verdict"]["n_unmet"] == 0
    assert doc["launch_receipt"]["verdict"]["verdict"] == "LAUNCH"
    assert doc["launch_receipt"]["phase_transition"] == "STARTED"
    assert doc["launch_receipt"]["bytes"] == 70623


def test_every_obligation_component_is_a_path_plus_hash_or_absent(sealed):
    doc = sealed["doc"]
    by_id = {c["id"]: c for c in doc["components"]}
    for oid in bg.OBLIGATION_IDS:
        assert oid in by_id, oid
        row = by_id[oid]
        assert row["pins"], oid
        for pin in row["pins"]:
            assert pin["path"], oid
            if pin["status"] == "PRESENT":
                assert isinstance(pin["sha256"], str) and len(pin["sha256"]) == 64, pin
                assert isinstance(pin["bytes"], int) and pin["bytes"] > 0
            else:
                assert pin["sha256"] is None
                assert pin["reason"]
    # Expected-absent row is not omitted.
    absent = by_id["hcli_agentos_resident_py"]
    assert absent["status"] == "ABSENT"
    assert "resident_gate.py" in (absent["reason"] or "")
    assert absent["pins"][0]["path"] == "hcli/agentos/resident.py"


def test_honest_golden_content_includes_what_it_does_not_have(sealed):
    doc = sealed["doc"]
    by_id = {c["id"]: c for c in doc["components"]}
    conc = by_id["concurrency_doctor"]
    assert conc["status"] == "PRESENT"
    assert conc["receipt"]["experiment_state"] == "SLEEPING"
    assert conc["receipt"]["sleeping_by_design"] is True
    assert conc["receipt"]["is_a_measurement"] is False
    safe = by_id["autonomy_run_safe_capabilities"]
    cap = safe["safe_capabilities"]
    assert cap["parsed"] is True
    assert cap["omits_mutation_engine"] is True
    assert "mutation_engine.py" not in cap["names"]
    assert doc["honest_state"]["resident_gate_is_the_live_boundary"] is True
    assert doc["honest_state"]["concurrency_doctor_sleeping_by_design"] is True
    assert doc["honest_state"]["autonomy_run_omits_mutation_engine"] is True


def test_trial_receipts_and_timeline_seals_carry_their_seals(sealed):
    doc = sealed["doc"]
    by_id = {c["id"]: c for c in doc["components"]}
    trials = {r["path"]: r for r in by_id["trial_receipts"]["receipts"]}
    for rel in bg.TRIAL_RECEIPT_RELS:
        assert rel in trials
        row = trials[rel]
        assert row["status"] == "PRESENT"
        assert row["sha256"]
        if rel.endswith("ODYSSEY_I_LAUNCH.json"):
            assert row["seal_sha256"] == bg.LAUNCH_SEAL_SHA256
            assert row["seal_verifies"] is True
            assert row["verdict"]["n_met"] == 16
        else:
            # Every named trial except the unsigned 15m timeline carries a seal
            # on the receipt itself; SUCCESSION_TRIAL's verdict is a status.
            assert row["seal_sha256"]
            assert row["seal_verifies"] is True
    timelines = {r["path"]: r for r in by_id["timeline_seals"]["receipts"]}
    t15 = timelines["receipts/future/AUTONOMY_TIMELINE_15m.json"]
    assert t15["status"] == "PRESENT"
    assert t15["sha256"]
    assert t15["seal_sha256"] is None
    assert "no seal_sha256" in (t15.get("seal_absent_reason") or "")
    assert timelines["receipts/future/AUTONOMY_TIMELINE_30m.json"]["seal_verifies"] is True
    assert timelines["receipts/future/AUTONOMY_TIMELINE_1h.json"]["seal_verifies"] is True


def test_recovery_actually_ran(sealed):
    rec = sealed["recovery"]
    assert rec["ran"] is True
    assert rec["ok"] is True
    assert rec["commit_exists"] is True
    assert rec["tree_matches"] is True
    recon = rec["reconstitution"]
    assert recon["ok"] is True
    assert recon["n_failed"] == 0
    assert recon["n_reconstituted"] == recon["n_present_pins"]
    assert recon["n_reconstituted"] > 20
    launch = rec["launch_receipt"]
    assert launch["ok"] is True
    assert launch["n_met"] == 16
    assert launch["n_unmet"] == 0
    assert launch["seal_is_the_launch_seal"] is True
    assert rec["gate_receipt"]["ok"] is True
    assert rec["gate_receipt"]["n_met"] == 16
    reeval = rec["gate_reeval"]
    assert reeval["ok"] is True
    assert reeval["wrote_receipts"] is False
    assert reeval["n_criteria"] == 16
    assert reeval["n_met"] == 16
    assert reeval["n_unmet"] == 0
    assert reeval["unmet"] == []
    assert reeval["verdict"] == "LAUNCH"
    assert rec["not_proven"], "an unverified recovery point is a note, not a receipt"
    assert any("evaluate_launch_criteria" in x for x in rec["not_proven"])
    assert any("NX weight" in x or "weight files" in x for x in rec["not_proven"])
    assert rec["gravity_owned_present_in_git"] is True


def test_supersession_is_detectable_and_does_not_block(sealed):
    doc = json.loads(Path(sealed["path"]).read_text())
    cmp_now = bg.compare_to_golden(doc)
    assert cmp_now["blocks_descendant"] is False
    assert cmp_now["seal_verifies"] is True
    assert cmp_now["digest_matches"] is True
    assert cmp_now["state"] in {"STILL_THIS_GOLDEN", "HEAD_MOVED"}
    moved = bg.compare_to_golden(doc, current_head="deadbeef" * 5)
    assert moved["state"] == "HEAD_MOVED"
    assert moved["blocks_descendant"] is False
    tampered = dict(doc)
    tampered["seal_sha256"] = "1" * 64
    assert bg.compare_to_golden(tampered)["state"] == "RECEIPT_TAMPERED"
    drifted = dict(doc)
    drifted["golden_digest"] = "2" * 64
    body = {k: v for k, v in drifted.items() if k != "seal_sha256"}
    drifted["seal_sha256"] = _sha(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    )
    assert bg.compare_to_golden(drifted)["state"] == "IDENTITY_DRIFT"
    successor = doc["successor"]
    assert successor["not_a_freeze"] is True
    assert successor["descendants_may_land"] is True
    assert successor["blocking_a_descendant_is_forbidden"] is True
    assert successor["mint_a_successor"]
    assert successor["this_receipt_is_not_rewritten_by_a_successor"] is True
    assert doc["predecessor_golden_digest"] is None


def test_golden_digest_is_stable_against_bench_timestamp(sealed):
    doc = sealed["doc"]
    identity = doc["golden_identity"]
    assert bg.golden_digest(identity) == doc["golden_digest"]
    assert identity["launch_commit"] == bg.LAUNCH_COMMIT
    assert identity["launch_seal_sha256"] == bg.LAUNCH_SEAL_SHA256


def test_build_does_not_rewrite_sibling_receipts(sealed):
    """Write scope is the golden receipt. Evaluators must not have fired."""
    import subprocess

    siblings = (
        "receipts/future/DIRTY_MEASUREMENT.json",
        "receipts/future/EVIDENCE_DAG.json",
        "receipts/future/WORKGRAPH_STATE.json",
        "receipts/future/ODYSSEY_I_LAUNCH.json",
        "receipts/future/ODYSSEY_LAUNCH_GATE.json",
    )
    for rel in siblings:
        head = subprocess.check_output(
            ["git", "--no-optional-locks", "show", f"{bg.LAUNCH_COMMIT}:{rel}"]
        )
        assert Path(rel).read_bytes() == head, rel


def test_hardware_fields_stay_non_numeric(sealed):
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims({"tps": 1.0})
    for key in HARDWARE_FIELDS:
        assert not isinstance(sealed["doc"].get(key), (int, float))
