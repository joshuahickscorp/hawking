"""Odyssey II transfer: named campaign law, measurement not similarity, no widen."""
from __future__ import annotations

import json

import pytest

from tools.future import odyssey2_law_store as ols
from tools.future import odyssey2_transfer as o2t
from tools.future import phase_listeners as pl
from tools.future._common import RECEIPTS, HardwareClaimError, _assert_no_hardware_claims


def test_build_emits_sealed_receipt():
    out = o2t.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "ODYSSEY2_TRANSFER.json"
    assert doc["schema"] == o2t.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["odyssey_i_barrier"] is None
    assert doc["phase_ii_waits_for_odyssey_i_complete"] is False
    _assert_no_hardware_claims(doc)


def test_consumes_named_campaign_law_and_names_specimen():
    run = o2t.run_transfers()
    assert run["named_law"] == "LAW-L1-MLP-ARITHMETIC-SENSITIVE"
    assert run["verdict"] == o2t.TRANSFER_FAILED
    assert run["specimen_used"]
    assert "Falcon" in str(run["specimen_used"])
    headline = run["headline"]
    assert headline["source_display"] == o2t.SOURCE_DISPLAY
    assert headline["similarity_score"] is None
    assert headline["kind"] == "values"
    assert headline["narrowing"]["deleted"] is False
    assert headline["what_transfer_bought"]["precise_reason_the_law_did_not_carry"]


def test_value_failure_cites_measurement_not_similarity():
    trial = o2t.transfer_values_l1_to_falcon()
    cites = trial["citations"]
    assert cites["arm_a_gb_s"]["copied"] is True
    assert cites["arm_a_gb_s"]["value"] == 497.4
    assert cites["arm_a_gb_s"]["source_receipt"] == o2t.ALU_REL
    assert cites["production_gb_s"]["value"] == 329.6
    assert cites["weight_bytes"]["value"] == 83558400
    assert cites["loads_survived"]["value"] is True
    assert trial["similarity_score"] is None
    assert trial["verdict"] == o2t.TRANSFER_FAILED


def test_method_transfer_is_unmeasured_with_named_experiment():
    trial = o2t.transfer_method_l1_to_falcon()
    assert trial["verdict"] == o2t.UNMEASURED
    assert trial["measurement_state"] == o2t.UNMEASURED
    assert "ARM A" in trial["experiment_that_would_settle"]
    assert "497.4" in trial["why_not_copied"]
    assert "Falcon" in str(trial["target_specimen"])


def test_l2_organ_absent_on_falcon_fails_and_keeps_the_law():
    trial = o2t.transfer_l2_to_falcon()
    assert trial["named_law"] == "LAW-L2-BROADCAST-AUX-NOT-CRITICAL-PATH"
    assert trial["verdict"] == o2t.TRANSFER_FAILED
    assert trial["narrowing"]["deleted"] is False
    assert trial["scope_after"] == "MODEL_LOCAL"
    still = o2t.campaign_law("LAW-L2-BROADCAST-AUX-NOT-CRITICAL-PATH")
    assert still.scope == "MODEL_LOCAL"


def test_campaign_laws_are_ols_laws_model_local():
    laws = o2t.campaign_laws()
    ids = [l.law_id for l in laws]
    assert ids == [
        "LAW-L1-MLP-ARITHMETIC-SENSITIVE",
        "LAW-L2-BROADCAST-AUX-NOT-CRITICAL-PATH",
        "LAW-L3-MLP-FUNCTION-REPLACEMENT-CLOSED",
        "LAW-L4-PROBE-UNDERSELLS-TOKEN",
        "LAW-L5-PRODUCTION-ROOF-WITH-ACTIVATION-LOAD",
    ]
    for law in laws:
        ols.validate_law(law)
        assert law.scope == "MODEL_LOCAL"
        assert law.time_to_first_useful_executable_ns is None
        assert law.source_model == o2t.SOURCE_MODEL


def test_refuses_widen_without_replicating_specimen():
    """NEGATIVE CONTROL: sealed B is not a replication."""
    law = o2t.campaign_law("LAW-L1-MLP-ARITHMETIC-SENSITIVE")
    evidence = {
        "models": [o2t.SOURCE_MODEL, "tiiuae/Falcon-H1-7B-Instruct"],
        "architecture_families": [o2t.SOURCE_FAMILY, "falcon_h1"],
        "evidence_strength": "DIAGNOSTIC_RELATIVE",
        "evidence_refs": [o2t.ALU_REL],
        "replications": [],
    }
    with pytest.raises(o2t.ReplicatingSpecimenRequired) as ei:
        o2t.widen(law, "ARCHITECTURE_FAMILY", evidence)
    assert ei.value.reason == "need_replicating_specimen"
    assert law.scope == "MODEL_LOCAL"
    assert "replicating specimen" in str(ei.value)


def test_unmeasured_replication_does_not_count():
    law = o2t.campaign_law("LAW-L1-MLP-ARITHMETIC-SENSITIVE")
    evidence = {
        "models": [o2t.SOURCE_MODEL, "tiiuae/Falcon-H1-7B-Instruct"],
        "architecture_families": [o2t.SOURCE_FAMILY],
        "replications": [
            {
                "specimen": "tiiuae--Falcon-H1-7B-Instruct@41e72f27effb",
                "verdict": "HOLDS",
                "measurement_state": o2t.UNMEASURED,
                "measurement": None,
                "law_id": law.law_id,
            }
        ],
    }
    with pytest.raises(o2t.ReplicatingSpecimenRequired):
        o2t.widen(law, "ARCHITECTURE_FAMILY", evidence)


def test_same_origin_replication_does_not_count():
    law = o2t.campaign_law("LAW-L1-MLP-ARITHMETIC-SENSITIVE")
    evidence = {
        "replications": [
            {
                "specimen": o2t.SOURCE_MODEL,
                "verdict": "HOLDS",
                "measurement_state": "COPIED_FROM_NAMED_RECEIPT",
                "measurement": {"arm_a_gb_s": 497.4},
                "law_id": law.law_id,
            }
        ]
    }
    assert o2t.replicating_specimens(law, evidence) == []
    with pytest.raises(o2t.ReplicatingSpecimenRequired):
        o2t.widen(law, "ARCHITECTURE_FAMILY", evidence)


def test_similarity_score_is_refused():
    with pytest.raises(o2t.NotAMeasurementError):
        o2t._refuse_similarity({"similarity": 0.99})
    with pytest.raises(o2t.NotAMeasurementError):
        o2t._refuse_similarity({"cosine_to_source": 0.993})


def test_identity_transfer_is_refused():
    row = o2t.identity_transfer_refused()
    assert row["refused"] is True
    assert row["reason_code"] == "not_a_transfer"
    assert o2t.SOURCE_SPECIMEN in str(row["target_specimen"]) or "qwen" in str(row["target_specimen"]).lower()


def test_does_not_wait_for_odyssey_i():
    law = o2t.campaign_law("LAW-L1-MLP-ARITHMETIC-SENSITIVE")
    falcon = o2t.require_sealed("Falcon-H1-7B-Instruct")
    assert o2t.odyssey_i_barrier() is None
    assert o2t.may_transfer(law, falcon) is True
    assert pl.LISTEN_RULE
    assert "once Phase I emits a law" in pl.LISTEN_RULE


def test_sealed_specimens_include_the_five():
    specs = o2t.sealed_specimens()
    assert "qwen27" in specs
    assert "Falcon-H1-7B-Instruct" in specs
    assert "Qwen3-0.6B" in specs
    assert "Mistral-Small-3.1-24B" in specs
    assert "Qwen3.8-Flash-Next" in specs
    assert specs["qwen27"]["source_of_campaign_laws"] is True
    assert specs["Falcon-H1-7B-Instruct"]["whole_tree_verified"] is True
    assert specs["Falcon-H1-7B-Instruct"]["architecture_family"] == "falcon_h1"


def test_selftest_and_no_hardware_claims():
    loop = o2t.selftest()
    assert loop["ok"] is True
    assert loop["verdict"] == o2t.TRANSFER_FAILED
    assert loop["similarity_score_refused"] is True
    assert loop["law_deleted"] is False
    assert loop["widen_without_replicating_specimen_raised"]["raised"] is True
    _assert_no_hardware_claims(loop)
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims({"tps": 12.0})


def test_does_not_fork_the_law_store_lattice():
    src = open(o2t.__file__).read()
    assert "ols.promote" in src or "ols.promote(" in src
    assert "from tools.future import odyssey2_law_store as ols" in src
    assert o2t.widen.__doc__ and "ols.promote" in (o2t.widen.__doc__ or "")
    # The II lattice is the store's, not a second one.
    law = o2t.campaign_law("LAW-L1-MLP-ARITHMETIC-SENSITIVE")
    assert law.scope in ols.SCOPES
