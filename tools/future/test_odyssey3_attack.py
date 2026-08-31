"""Odyssey III attack: named campaign law, break or survive, no widen."""
from __future__ import annotations

import json

import pytest

from tools.future import odyssey2_transfer as o2t
from tools.future import odyssey3_adversary as o3
from tools.future import odyssey3_attack as o3a
from tools.future import phase_listeners as pl
from tools.future._common import RECEIPTS, HardwareClaimError, _assert_no_hardware_claims


def test_build_emits_sealed_receipt():
    out = o3a.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "ODYSSEY3_ATTACK.json"
    assert doc["schema"] == o3a.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["odyssey_i_barrier"] is None
    assert doc["phase_iii_waits_for_odyssey_i_complete"] is False
    assert doc["survived_is_a_result"] is True
    assert doc["unmeasured_is_a_result"] is True
    _assert_no_hardware_claims(doc)


def test_headline_attacks_named_l5_and_names_specimen():
    run = o3a.run_attacks()
    assert run["named_law"] == "LAW-L5-PRODUCTION-ROOF-WITH-ACTIVATION-LOAD"
    assert run["verdict"] == o3a.BROKE
    assert run["specimen_used"]
    assert "qwen3.8-27b" in str(run["specimen_used"]) or run["specimen_used"] == o2t.SOURCE_SPECIMEN
    headline = run["headline"]
    assert headline["kind"] == "layer_counterexample"
    assert headline["law_deleted"] is False
    assert headline["scope_after"] == "ORGAN_LOCAL"
    assert o3.is_downgrade(headline["scope_before"], headline["scope_after"])
    assert headline["citations"]["deltanet_arm_a_gb_s"]["copied"] is True
    assert headline["citations"]["deltanet_arm_a_gb_s"]["value"] == 943.2


def test_fidelity_703p5_survives_and_is_a_result():
    row = o3a.attack_l5_fidelity_703p5()
    assert row["named_law"] == "LAW-L5-PRODUCTION-ROOF-WITH-ACTIVATION-LOAD"
    assert row["verdict"] == o3a.SURVIVED
    assert row["strengthened"] is True
    assert row["moved"] is False
    assert row["citations"]["addr_probe_activation_loaded"]["value"] is False
    assert row["citations"]["mlp_arm_a_activation_loaded"]["value"] is True
    assert row["citations"]["clean_gemv_gb_s"]["value"] == 703.5


def test_two_legs_survives_and_cannot_widen():
    row = o3a.attack_l5_two_legs()
    assert row["verdict"] == o3a.SURVIVED
    assert row["n_legs"] == 2
    assert row["cannot_widen_without_replicating_specimen"] is True
    assert row["citations"]["mlp_arm_a_gb_s"]["value"] == 497.4
    assert row["citations"]["lm_head_gb_s"]["value"] == 497.4


def test_l4_survives_directionally_and_names_where_framing_breaks():
    row = o3a.attack_l4_probe_classes()
    assert row["named_law"] == "LAW-L4-PROBE-UNDERSELLS-TOKEN"
    assert row["verdict"] == o3a.SURVIVED
    assert row["moved"] is False
    assert "1.745" in row["where_it_breaks"]
    assert row["citations"]["widen_isolated_ms"]["value"] == 0.7046
    assert row["citations"]["widen_complete_ms"]["value"] == 1.0245
    assert row["citations"]["fold_projection_ms"]["value"] == 1.745


def test_falcon_is_section_66_adversary_and_unmeasured_is_a_result():
    row = o3a.attack_l1_falcon_model_counterexample()
    assert row["named_law"] == "LAW-L1-MLP-ARITHMETIC-SENSITIVE"
    assert row["kind"] == "model_counterexample"
    assert "Falcon" in str(row["specimen_used"])
    assert row["specimen_family"] == "falcon_h1"
    assert row["specimen_whole_tree_verified"] is True
    assert row["verdict"] == o3a.UNMEASURED
    assert row["not_a_null_run"] is True
    assert row["measurement_state"] == o3a.UNMEASURED
    assert "ARM A" in row["experiment_that_would_settle"]
    assert row["execute_attack"]["physical_arm"] == "not_run"
    assert row["moved"] is False


def test_o3_laws_validate_and_each_gets_attacks():
    for law in o3a.campaign_o3_laws():
        o3.validate_law(law)
        plan = o3.emit_for_law(law)
        assert plan["n_attacks"] == len(o3.ATTACK_FAMILIES)
        assert plan["selected_attack_id"]


def test_refuses_widen_without_replicating_specimen():
    """NEGATIVE CONTROL: a survived attack is not a promotion."""
    law = o3a.campaign_o3_law("LAW-L5-PRODUCTION-ROOF-WITH-ACTIVATION-LOAD")
    with pytest.raises(o2t.ReplicatingSpecimenRequired) as ei:
        o3a.widen_scope(
            law,
            "ARCHITECTURE_FAMILY",
            {
                "models": [o2t.SOURCE_MODEL, "tiiuae/Falcon-H1-7B-Instruct"],
                "architecture_families": [o2t.SOURCE_FAMILY, "falcon_h1"],
                "replications": [],
            },
        )
    assert ei.value.reason == "need_replicating_specimen"
    assert law["scope"] == "MODEL_LOCAL"


def test_apply_result_holds_does_not_move_and_refute_does():
    law = o3a.campaign_o3_law("LAW-L5-PRODUCTION-ROOF-WITH-ACTIVATION-LOAD")
    spec = o3a._spec_for(law, "law_scope")
    hold = o3.apply_result(law, spec, {"verdict": "HOLDS", "synthetic": True})
    assert hold["moved"] is False
    assert hold["scope_after"] == law["scope"]
    broke = o3.apply_result(law, spec, {"verdict": "REFUTED", "synthetic": True})
    assert broke["moved"] is True
    assert o3.is_downgrade(broke["scope_before"], broke["scope_after"])
    assert broke["scope_after"] == "ORGAN_LOCAL"
    assert law["scope"] == "MODEL_LOCAL"  # original not mutated


def test_does_not_wait_for_odyssey_i():
    assert o3a.odyssey_i_barrier() is None
    assert pl.LISTEN_RULE
    src = open(o3a.__file__).read()
    assert "phase_iii_waits_for_odyssey_i_complete" in src
    doc = json.loads(o3a.build().read_text())
    assert doc["phase_iii_waits_for_odyssey_i_complete"] is False
    assert doc["odyssey_i_barrier"] is None


def test_does_not_fork_adversary_or_store():
    src = open(o3a.__file__).read()
    assert "from tools.future import odyssey3_adversary as o3" in src
    assert "from tools.future import odyssey2_transfer as o2t" in src
    assert "o3.generate_attacks" in src
    assert "o3.apply_result" in src
    assert "o2t.widen" in src or "o2t.ReplicatingSpecimenRequired" in src


def test_selftest_and_no_hardware_claims():
    loop = o3a.selftest()
    assert loop["ok"] is True
    assert loop["verdict"] == o3a.BROKE
    assert loop["n_survived"] >= 1
    assert loop["n_unmeasured"] >= 1
    assert loop["widen_without_replicating_specimen_raised"]["raised"] is True
    _assert_no_hardware_claims(loop)
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims({"bandwidth_gbps": 497.4})
