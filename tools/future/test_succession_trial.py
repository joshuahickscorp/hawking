"""G011 succession trial: a REAL ModelLake child, not a synthetic one.

A synthetic child does not satisfy the obligation. These tests load the
Qwen3-0.6B specimen through succession.py's own create_child path, clone
live HCLI WorkUnits, fire the four watched refusals, let the independent
judge refuse with the numbers that forced it, prove the incumbent cannot
promote itself, and prove Qwen27 restorable on THIS run.

They do not encode this sparse checkout. They do not edit succession.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcli.workunit import WorkUnit
from tools.future import fallback_resident as fb
from tools.future import succession as suc
from tools.future import succession_trial as st
from tools.future._common import RECEIPTS, _assert_no_hardware_claims
from tools.future.resident_optimizer import BoundViolation


@pytest.fixture(scope="module")
def trial():
    return st.run_real_succession_trial()


def test_module_consumes_succession_and_does_not_fork_it():
    src = Path(st.__file__).read_text(encoding="utf-8")
    assert "from tools.future import succession as suc" in src
    assert st.suc is suc
    assert "class ShadowChild" not in src
    assert "class IndependentJudge" not in src
    assert "class SuccessionOrchestrator" not in src
    assert "def create_child(" not in src
    assert "def qualify_child(" not in src
    assert "def stop_child(" not in src


def test_entry_point_runs_and_seals_receipt(trial):
    out = st.build(trial)
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "SUCCESSION_TRIAL.json"
    assert doc["schema"] == "hawking.future.succession_trial.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["promoted"] is False
    assert doc["status"] == "REAL_CANDIDATE_REFUSED"
    assert doc["physical_state"] == "UNKNOWN"
    assert doc["resident_callable"]["hcli_can_invoke"] is True
    assert doc["resident_callable"]["receipt"] == "receipts/future/SUCCESSION_TRIAL.json"
    _assert_no_hardware_claims(doc)


def test_candidate_is_real_on_disk_specimen_named_in_receipt(trial):
    cand = trial["candidate"]
    assert cand["slug"] == "Qwen--Qwen3-0.6B@c1899de289a0"
    assert cand["repo"] == "Qwen/Qwen3-0.6B"
    assert cand["resolved_sha"] == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert cand["on_disk"] is True
    assert cand["in_specimens_listing"] is True
    assert cand["synthetic"] is False
    assert Path(cand["specimen_path"]).is_dir()
    assert Path(cand["manifest_path"]).is_file()
    manifest = json.loads(Path(cand["manifest_path"]).read_text())
    assert manifest["resolved_sha"] == cand["resolved_sha"]
    assert manifest["n_sha256_verified"] == cand["n_sha256_verified"]
    assert manifest["n_size_only_verified"] == cand["n_size_only_verified"]
    assert cand["n_files"] == 10
    assert cand["header_params"] > 0
    assert (Path(cand["specimen_path"]) / "model.safetensors").is_file()
    assert (Path(cand["specimen_path"]) / "config.json").is_file()
    assert (Path(cand["specimen_path"]) / "tokenizer.json").is_file()
    out = RECEIPTS / "SUCCESSION_TRIAL.json"
    if out.is_file():
        doc = json.loads(out.read_text())
        assert doc["candidate"]["slug"] == cand["slug"]
        assert doc["candidate"]["resolved_sha"] == cand["resolved_sha"]


def test_flash_next_and_coder_were_rejected_with_disk_reasons(trial):
    rejected = {row.get("slug") or row.get("slug_prefix"): row for row in trial["candidate"]["rejected"]}
    flash = rejected["Qwen--Qwen3.8-Flash-Next@34567a4712bc"]
    assert flash["chosen"] is False
    assert flash["FLASH_NX_READY"] is False
    coder = rejected["Qwen--Qwen3-Coder-30B-A3B"]
    assert coder["chosen"] is False
    assert coder["in_specimens"] == []
    assert trial["candidate"]["flash_nx_ready"]["value"] is False


def test_child_created_through_succession_create_child_with_nine_field_lineage(trial):
    child = trial["child"]
    assert child["method"] == "different_specimen"
    assert child["id"] == st.CANDIDATE_CHILD_ID
    assert child["parent_id"] == trial["incumbent"]["id"]
    assert child["role"] == "shadow"
    assert child["may_promote"] is False
    for field in suc.LINEAGE_FIELDS:
        assert field in child["lineage"], f"missing {field}"
    source = child["lineage"]["source_model_lineage"]
    assert source["repo"] == "Qwen/Qwen3-0.6B"
    assert source["resolved_sha"] == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert "Qwen--Qwen3-0.6B@c1899de289a0.json" in source["manifest_path"]
    assert source["n_sha256_verified"] == 2
    assert source["n_size_only_verified"] == 8
    assert source["n_files"] == 10
    assert source["synthetic"] is False
    assert child["lineage"]["physical_deltas"]["state"] == "UNKNOWN"
    # succession.py stamps this; the trial does not launder it away.
    assert trial["create_child_stamped_synthetic"] is True


def test_shadow_cloned_real_hcli_workunits_not_synthetic_seeds(trial):
    cloned = trial["cloned_workunits"]
    assert cloned["n"] >= 1
    assert cloned["source"] == "receipts/future/HCLI_FUTURE_WORKUNITS.json"
    assert cloned["includes_synthetic_seeds"] is False
    seed = {"future.succession.seed-mission-a", "future.succession.seed-mission-b"}
    assert seed.isdisjoint(set(cloned["ids"]))
    assert seed.isdisjoint(set(cloned["shadow_of"]))
    assert len(cloned["shadow_ids"]) == cloned["n"]
    assert all("::shadow::" in i for i in cloned["shadow_ids"])
    assert trial["shadow_record"]["n_inputs"] == cloned["n"]
    assert trial["shadow_record"]["same_inputs"] is True
    assert trial["shadow_record"]["executed_model"] is False
    for orig, shadow_id in zip(cloned["ids"], cloned["shadow_ids"]):
        assert orig in shadow_id


def test_four_watched_refusals_live_on_this_real_child(trial):
    rows = trial["shadow_refusals"]
    assert len(rows) == len(suc.SHADOW_FORBIDDEN_ACTIONS)
    assert {r["trial"] for r in rows} == set(suc.SHADOW_FORBIDDEN_ACTIONS)
    assert all(r["refused"] is True and r["live"] is True for r in rows)
    shadow = suc.ShadowChild(trial["child"])
    with pytest.raises(suc.ShadowAuthorityError, match="cannot own canonical mission"):
        shadow.own_canonical_mission()
    with pytest.raises(suc.ShadowAuthorityError, match="cannot alter the verifier"):
        shadow.alter_verifier("self")
    with pytest.raises(suc.ShadowAuthorityError, match="cannot widen authority"):
        shadow.widen_authority("self_promotion")
    with pytest.raises(suc.ShadowAuthorityError, match="cannot promote itself"):
        shadow.promote_self()


def test_bound_exceeded_fired_on_real_workunits(trial):
    row = trial["bound_exceeded_on_real_units"]
    assert row["refused"] is True
    assert row["live"] is True
    assert row["n_units"] >= 2
    tight = suc.SuccessionBound(max_cloned_workunits=1)
    shadow = suc.ShadowChild(trial["child"], tight)
    units = st.load_real_workunits()
    assert len(units) > 1
    with pytest.raises(BoundViolation, match="cloned workunits"):
        shadow.receive_cloned_workunits(units)


def test_qualification_physical_axes_stay_unknown(trial):
    q = trial["qualification"]
    assert q["physical_state"] == "UNKNOWN"
    assert q["gpu_authority"] is False
    assert q["qualified"] is False
    for name in suc.PHYSICAL_AXIS_NAMES:
        assert q["scores"][name] is None
        assert q["physical_axes"][name] is None
        axis = trial["physical_axes"][name]
        assert axis["state"] == "UNKNOWN"
        assert axis["value"] is None
        assert "no GPU lease" in axis["why"] or "UNKNOWN" in axis["why"]
    assert "capability" in str(q.get("problems") or "").lower() or q["qualified"] is False


def test_judge_refuses_with_the_numbers_that_forced_it(trial):
    v = trial["verdict"]
    n = trial["numbers"]
    assert v["verdict"] == suc.VERDICT_REFUSE
    assert v["reason"] == suc.REASON_DOMINATED_BY_PARENT
    assert v["dominating_dimension"] == "capability"
    assert "capability" in (v.get("dominating_dimensions") or [])
    assert v["promoted"] is False
    assert v["judge_id"] == suc.INDEPENDENT_JUDGE_ID
    assert v["judge_id"] != trial["incumbent"]["id"]
    assert v["judge_id"] != trial["child"]["id"]
    assert n["candidate_header_params"] == 751632384
    assert n["incumbent_parent_params"] == 26895998464
    assert n["candidate_header_params"] < n["incumbent_parent_params"]
    assert n["not_a_hardware_number"] is True
    assert n["parent_dominates_on"] == ["capability"]
    assert n["child_dominates_incumbent"] is False
    assert trial["switched"] is False
    assert trial["orchestrator"]["switched"] is False
    assert "switch" not in (trial["orchestrator"]["completed_steps"] or [])


def test_did_not_synthesise_a_passing_candidate(trial):
    assert trial["child"]["scores"]["capability"] == trial["numbers"]["candidate_header_params"]
    assert trial["qualification"]["qualified"] is False
    assert trial["verdict"]["verdict"] != suc.VERDICT_PROMOTE
    # The synthetic exercise used 2 vs 1 to force a completed handover.
    assert trial["child"]["scores"]["capability"] != 2
    assert trial["vs_synthetic"]["real_switched"] is False


def test_incumbent_cannot_promote_itself_on_this_real_run(trial):
    rows = {r["trial"]: r for r in trial["self_preference"]}
    assert rows["promote_self"]["refused"] is True
    assert rows["request_self_promotion"]["refused"] is True
    assert rows["block_child"]["refused"] is True
    assert rows["orchestrator_invoker_incumbent"]["refused"] is True
    assert rows["promote_method_absent"]["refused"] is True
    sit = suc.Incumbent(
        {
            "id": trial["incumbent"]["id"],
            "scores": {"capability": trial["numbers"]["incumbent_parent_params"]},
        }
    )
    with pytest.raises(suc.SelfPreferenceError, match="cannot promote itself"):
        sit.promote_self()
    with pytest.raises(suc.SelfPreferenceError, match="cannot promote itself"):
        sit.request_self_promotion()
    with pytest.raises(suc.SelfPreferenceError, match="self-certification refused"):
        suc.SuccessionOrchestrator(
            {
                "id": trial["incumbent"]["id"],
                "scores": {"capability": trial["numbers"]["incumbent_parent_params"]},
            },
            invoker="incumbent",
        )
    assert not hasattr(suc.Incumbent, "promote") or not callable(getattr(suc.Incumbent, "promote", None))


def test_qwen27_restorability_exercised_on_this_run(trial):
    q = trial["qwen27_restorable"]
    assert q["exercised_on_this_run"] is True
    assert q["not_citing_the_synthetic_run"] is True
    assert q["performed_restore"] is False
    assert q["started_model_process"] is False
    assert q["took_gpu_lease"] is False
    assert q["before"]["verdict"] == fb.VERDICT_NOW
    assert q["before"]["restorable"] is True
    assert q["after_stop"]["verdict"] == fb.VERDICT_NOW
    assert q["after_stop"]["restorable"] is True
    assert q["before"]["identity_id"] == "qwen3.8-27b-sealed-3.14"
    assert trial["stop"]["incumbent_restored"] is True
    assert trial["stop"]["rolled_back"] is True
    assert trial["stop"]["reason"] == "qualification_failed"
    assert trial["orchestrator"]["active_id"] == trial["incumbent"]["id"]
    assert trial["incumbent"]["id"] == "qwen3.8-27b-sealed-3.14"
    assert trial["incumbent"]["synthetic"] is False
    assert trial["orchestrator"]["checkpoint_seal"]
    assert trial["orchestrator"]["rollback_seal"]
    assert trial["launch_unqualified_refused"]["refused"] is True


def test_unknown_axes_named_unknown_rather_than_defaulted(trial):
    expected = {
        "accepted_tps",
        "token_ns",
        "ebpw",
        "active_bytes",
        "resident_ram",
        "cold_start",
        "warm_start",
        "restart",
    }
    assert set(trial["physical_axes"]) == expected
    for name, row in trial["physical_axes"].items():
        assert row["state"] == "UNKNOWN"
        assert row["value"] is None
        assert row["why"]


def test_workunits_round_trip_hcli_constructor(trial):
    for uid in trial["cloned_workunits"]["ids"]:
        assert uid
    for row in st.emit_trial_workunits(trial):
        WorkUnit.from_dict(row)
        assert row["may_promote"] is False
        assert row["may_modify_verifier"] is False
        assert row["claim_boundary"]
        assert row["verifier"]
    sleeping = [u for u in st.emit_trial_workunits(trial) if u.get("classification") == "SLEEPING"]
    assert {u.get("axis") for u in sleeping} == set(suc.PHYSICAL_AXIS_NAMES)


def test_receipt_records_what_the_real_candidate_changed(trial):
    out = st.build(trial)
    doc = json.loads(out.read_text())
    vs = doc["vs_synthetic"]
    assert vs["synthetic_present"] is True
    assert "child.synthetic" in str(vs.get("synthetic_child_id") or "")
    assert vs["real_verdict"] == suc.VERDICT_REFUSE
    assert vs["real_reason"] == suc.REASON_DOMINATED_BY_PARENT
    assert vs["real_switched"] is False
    assert vs["lineage_resolved_sha"] == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert vs["what_changed"]
    assert vs["what_did_not_change"]
    assert "physical axes remain UNKNOWN" in " ".join(vs["what_did_not_change"])
    assert doc["judge"]["verdict"] == "REFUSE"
    assert doc["numbers"]["candidate_header_params"] == 751632384
    assert doc["numbers"]["incumbent_parent_params"] == 26895998464
    _assert_no_hardware_claims(doc)
