"""Staged capability evaluation: cheaper depths first, SKIP is not a pass.

A guard nobody has watched fail is not a guard. Load-bearing refusals:

  * the module will not choose the component
  * every stage names what it does not establish
  * a SKIPPED stage cannot be counted toward a pass
  * missing stage evidence SKIPS, it does not pass
  * EXPENSIVE_QUALIFICATION refuses politely and is not a pass
  * a FAIL at a cheap stage does not invoke later stages
  * degenerate text that contains a checkable token still fails
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.future import aux_capability_screen as acs
from tools.future import capability_information_map as cim
from tools.future import capability_stages as cs
from tools.future import functional_role_probe as fp
from tools.future import qualification_pipeline as qp
from tools.future import resident_provider as rp
from tools.future import workunit_species as ws
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


def _component(cid: str = "FIXTURE.test") -> dict:
    return {"component": {"id": cid, "kind": "FIXTURE"}}


def _hidden_ok() -> dict:
    a = np.ones(32, dtype=np.float64)
    return {"hidden_a": a, "hidden_b": a + 1e-9}


def _hidden_fail() -> dict:
    a = np.ones(32, dtype=np.float64)
    b = np.zeros(32, dtype=np.float64)
    b[0] = 1.0
    return {"hidden_a": a, "hidden_b": b}


def _logits_ok() -> dict:
    a = np.zeros(32, dtype=np.float64)
    a[0], a[1], a[2] = 5.0, 4.0, 3.0
    return {"logits_a": a, "logits_b": a.copy()}


def _answers_ok() -> dict:
    eos = {"max_new_tokens": 16, "generated_tokens": 2, "new_token_ids": [rp.EOS_IM_END_ID]}
    return {
        "answers": {
            "fact-capital": {"text": "Paris", **eos},
            "fact-choice": {"text": "hbm_doctor.py", **eos},
            "fact-arith": {"text": "323", **eos},
        }
    }


def _unit() -> dict:
    return ws.emit_hcli_workunit(
        id="future.capability_stages.test",
        role="capability_stage_test",
        description="test work request",
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier="future.capability_stages",
        provider="future.capability_stages",
        effect_class="READ_ONLY",
    )


# ---------------------------------------------------------------------------
# Catalog / refusals
# ---------------------------------------------------------------------------


def test_catalog_is_five_stages_cheapest_first_and_names_what_it_does_not_establish():
    rows = cs.catalog()
    assert [r["id"] for r in rows] == list(cs.STAGE_IDS)
    assert len(rows) >= 5
    assert [r["rank"] for r in rows] == sorted(r["rank"] for r in rows)
    assert rows[0]["id"] == cs.LOCAL_FUNCTIONAL_FIDELITY
    assert rows[-1]["id"] == cs.EXPENSIVE_QUALIFICATION
    assert fp.MEASURED_LEVEL == cs.LOCAL_FUNCTIONAL_FIDELITY
    for row in rows:
        assert row["does_not_establish"].strip(), row["id"]
        assert row["measures"].strip(), row["id"]
        assert row["evidence_production_cost"] == "UNMEASURED"
        assert row["evidence_production_reason"].strip(), row["id"]


def test_every_stage_result_declares_what_it_does_not_establish():
    report = cs.evaluate({**_component(), **_hidden_ok(), **_logits_ok(), **_answers_ok(), "emissions": [_unit()]})
    by_id = {r["id"]: r for r in report["stages"]}
    assert set(by_id) == set(cs.STAGE_IDS)
    for sid, spec in zip(cs.STAGE_IDS, cs.catalog()):
        assert by_id[sid]["does_not_establish"] == spec["does_not_establish"]
        assert spec["does_not_establish"]
        assert "not" in spec["does_not_establish"].lower() or "Anything" in spec["does_not_establish"]


def test_module_refuses_to_choose_the_component():
    with pytest.raises(cs.StagesRefuse, match="does not choose"):
        cs.evaluate({})
    with pytest.raises(cs.StagesRefuse, match="does not choose"):
        cs.evaluate({"component": {}})
    with pytest.raises(cs.StagesRefuse, match="does not choose"):
        cs.evaluate({"component": {"id": "  "}})
    with pytest.raises(cs.StagesRefuse, match="does not choose"):
        cs.run_stage(cs.LOCAL_FUNCTIONAL_FIDELITY, {"hidden_a": [1.0], "hidden_b": [1.0]})


def test_unknown_stage_refuses():
    with pytest.raises(cs.StagesRefuse, match="unknown stage"):
        cs.run_stage("CAPABILITY_VIBES", {**_component(), **_hidden_ok()})


# ---------------------------------------------------------------------------
# SKIP is not a pass
# ---------------------------------------------------------------------------


def test_skipped_stage_cannot_be_counted_toward_a_pass():
    """NEGATIVE CONTROL: missing evidence SKIPS, and SKIP is excluded from n_pass."""
    report = cs.evaluate(_component("FIXTURE.none"))
    assert report["n_skipped"] >= 1
    assert report["n_pass"] == 0
    assert report["passed_stage_ids"] == []
    assert report["establishes_capability"] is False
    assert report["overall"] == cs.OVERALL_INCOMPLETE
    for row in report["stages"]:
        assert row["verdict"] in {cs.SKIPPED, cs.NOT_RUN}
        assert cs.counts_as_pass(row) is False
        assert row["counts_as_pass"] is False
    naive = sum(1 for r in report["stages"] if r["verdict"] in {cs.PASS, cs.SKIPPED})
    assert naive > report["n_pass"]
    skip = {"id": cs.EXPENSIVE_QUALIFICATION, "verdict": cs.SKIPPED, "reason": "no GPU"}
    assert cs.counts_as_pass(skip) is False
    with pytest.raises(cs.StagesRefuse, match="no verdict"):
        cs.counts_as_pass({"id": cs.LOGIT_TOKEN})


def test_missing_stage_evidence_skips_rather_than_passing_or_defaulting():
    with pytest.raises(cs.StageInputMissing, match="hidden_a"):
        cs.run_stage(cs.LOCAL_FUNCTIONAL_FIDELITY, _component())
    with pytest.raises(cs.StageInputMissing, match="logits_a"):
        cs.run_stage(cs.LOGIT_TOKEN, _component())
    with pytest.raises(cs.StageInputMissing, match="answers"):
        cs.run_stage(cs.FAST_CAPABILITY, _component())
    with pytest.raises(cs.StageInputMissing, match="emissions"):
        cs.run_stage(cs.HCLI_MISSION_SUBSET, _component())
    report = cs.evaluate({**_component(), **_logits_ok()})
    by_id = {r["id"]: r for r in report["stages"]}
    assert by_id[cs.LOCAL_FUNCTIONAL_FIDELITY]["verdict"] == cs.SKIPPED
    assert by_id[cs.LOGIT_TOKEN]["verdict"] == cs.PASS
    assert cs.counts_as_pass(by_id[cs.LOCAL_FUNCTIONAL_FIDELITY]) is False
    assert report["n_pass"] == 1
    assert cs.LOCAL_FUNCTIONAL_FIDELITY not in report["passed_stage_ids"]


def test_partial_fast_answers_are_a_skip_not_a_pass():
    subject = {
        **_component(),
        "answers": {"fact-capital": "Paris"},
    }
    with pytest.raises(cs.StageInputMissing, match="missing"):
        cs.run_stage(cs.FAST_CAPABILITY, subject)
    report = cs.evaluate(subject, max_stage=cs.FAST_CAPABILITY)
    row = {r["id"]: r for r in report["stages"]}[cs.FAST_CAPABILITY]
    assert row["verdict"] == cs.SKIPPED
    assert cs.counts_as_pass(row) is False


# ---------------------------------------------------------------------------
# Cheap stages wrap existing instruments
# ---------------------------------------------------------------------------


def test_local_functional_fidelity_wraps_hidden_cosine_and_is_not_capability():
    a = np.ones(64)
    b = a + 1e-8
    assert cs.hidden_cosine(a, b) == pytest.approx(cim._cosine(a, b))
    row = cs.run_stage(cs.LOCAL_FUNCTIONAL_FIDELITY, {**_component(), **_hidden_ok()})
    assert row["verdict"] == cs.PASS
    assert row["measurement"]["bar"] == cim.HIDDEN_COSINE_BAR
    assert row["measurement"]["instrument"] == fp.MEASURED_LEVEL
    assert "Capability" in row["does_not_establish"] or "capability" in row["does_not_establish"]
    bad = cs.run_stage(cs.LOCAL_FUNCTIONAL_FIDELITY, {**_component(), **_hidden_fail()})
    assert bad["verdict"] == cs.FAIL
    assert cs.counts_as_pass(bad) is False


def test_logit_token_reports_kl_topk_and_argmax_and_argmax_is_not_the_bar():
    row = cs.run_stage(cs.LOGIT_TOKEN, {**_component(), **_logits_ok()})
    assert row["verdict"] == cs.PASS
    m = row["measurement"]
    assert m["argmax_is_not_parity"] is True
    assert "kl_nats" in m["parity_quantities"]
    assert m["kl_bar"] == acs.LOGIT_KL_BAR
    assert m["top_k_bar"] == acs.TOPK_AGREE_BAR
    assert m["argmax_agreement"] == pytest.approx(1.0)
    drifted = np.zeros(32)
    drifted[7] = 9.0
    fail = cs.run_stage(
        cs.LOGIT_TOKEN,
        {**_component(), "logits_a": _logits_ok()["logits_a"], "logits_b": drifted},
    )
    assert fail["verdict"] == cs.FAIL
    with pytest.raises(acs.ArgmaxAloneParityRefuse):
        acs.report_logit_parity(
            kl_nats=None, top_k_agreement=None, argmax_agreement=1.0
        )


def test_fast_capability_scores_checkable_answers_and_refuses_degenerate_fishing():
    ok = cs.run_stage(cs.FAST_CAPABILITY, {**_component(), **_answers_ok()})
    assert ok["verdict"] == cs.PASS
    assert ok["measurement"]["n_passed"] == 3
    degenerate = {
        **_component(),
        "answers": {
            "fact-capital": {"text": "Paris", "max_new_tokens": 16, "generated_tokens": 2, "new_token_ids": [rp.EOS_IM_END_ID]},
            "fact-choice": {
                "text": rp.MEASURED_DEGENERATE_CHOICE,
                "max_new_tokens": 64,
                "generated_tokens": 20,
            },
            "fact-arith": {"text": "323", "max_new_tokens": 16, "generated_tokens": 1, "new_token_ids": [rp.EOS_IM_END_ID]},
        },
    }
    caught = cs.run_stage(cs.FAST_CAPABILITY, degenerate)
    assert caught["verdict"] == cs.FAIL
    choice = {i["id"]: i for i in caught["measurement"]["items"]}["fact-choice"]
    assert choice["passed"] is False
    assert choice["quality"] == rp.QUALITY_DEGENERATE
    assert "fished" in choice["reason"].lower() or "DEGENERATE" in choice["reason"]
    wrong = {
        **_component(),
        "answers": {
            "fact-capital": "London",
            "fact-choice": "freshness.py",
            "fact-arith": "0",
        },
    }
    assert cs.run_stage(cs.FAST_CAPABILITY, wrong)["verdict"] == cs.FAIL


def test_hcli_mission_subset_validates_structured_work_requests():
    unit = _unit()
    ws.validate_emitted_unit(unit)
    ok = cs.run_stage(cs.HCLI_MISSION_SUBSET, {**_component(), "emissions": [unit]})
    assert ok["verdict"] == cs.PASS
    assert ok["measurement"]["n_valid"] == 1
    bad = cs.run_stage(
        cs.HCLI_MISSION_SUBSET,
        {**_component(), "emissions": [{"id": "nope", "role": "x"}]},
    )
    assert bad["verdict"] == cs.FAIL
    assert cs.counts_as_pass(bad) is False
    as_json = cs.run_stage(
        cs.HCLI_MISSION_SUBSET,
        {**_component(), "emissions": [json.dumps(unit)]},
    )
    assert as_json["verdict"] == cs.PASS


# ---------------------------------------------------------------------------
# Expensive stage / terminate cheaply
# ---------------------------------------------------------------------------


def test_expensive_qualification_refuses_politely_and_is_not_a_pass():
    with pytest.raises(cs.ExpensiveQualificationRefused, match="skip, not a pass"):
        cs.run_stage(cs.EXPENSIVE_QUALIFICATION, _component())
    with pytest.raises(qp.ExecuteRefused):
        qp.execute(explicit_execute=False)
    report = cs.evaluate(_component(), max_stage=cs.EXPENSIVE_QUALIFICATION)
    row = {r["id"]: r for r in report["stages"]}[cs.EXPENSIVE_QUALIFICATION]
    assert row["verdict"] == cs.SKIPPED
    assert cs.counts_as_pass(row) is False
    assert row["counts_as_pass"] is False
    decl = cs.expensive_qualification_declaration()
    assert decl["execute_always_refuses"] is True
    assert decl["gpu_authority"] is False
    assert decl["pipeline_stages"] == list(qp.STAGES)


def test_fail_at_cheapest_stage_does_not_invoke_later_stages():
    called: list[str] = []
    report = cs.evaluate(
        {**_component(), **_hidden_fail(), **_logits_ok(), **_answers_ok(), "emissions": [_unit()]},
        stop_on_fail=True,
        on_stage=lambda sid, _s: called.append(sid),
    )
    assert called == [cs.LOCAL_FUNCTIONAL_FIDELITY]
    by_id = {r["id"]: r for r in report["stages"]}
    assert by_id[cs.LOCAL_FUNCTIONAL_FIDELITY]["verdict"] == cs.FAIL
    for sid in cs.STAGE_IDS[1:]:
        assert by_id[sid]["verdict"] == cs.NOT_RUN
        assert cs.counts_as_pass(by_id[sid]) is False
    assert report["n_pass"] == 0
    assert report["overall"] == cs.OVERALL_FAIL
    assert report["establishes_capability"] is False


def test_max_stage_stops_the_ladder_before_deeper_work():
    called: list[str] = []
    report = cs.evaluate(
        {**_component(), **_hidden_ok(), **_logits_ok()},
        max_stage=cs.LOGIT_TOKEN,
        on_stage=lambda sid, _s: called.append(sid),
    )
    assert called == [cs.LOCAL_FUNCTIONAL_FIDELITY, cs.LOGIT_TOKEN]
    assert [r["id"] for r in report["stages"]] == [
        cs.LOCAL_FUNCTIONAL_FIDELITY,
        cs.LOGIT_TOKEN,
    ]
    assert report["n_pass"] == 2
    assert cs.FAST_CAPABILITY not in report["passed_stage_ids"]
    assert report["establishes_capability"] is False


def test_pass_through_cheap_depth_is_still_not_capability():
    report = cs.evaluate(
        {**_component(), **_hidden_ok(), **_logits_ok(), **_answers_ok(), "emissions": [_unit()]},
        max_stage=cs.HCLI_MISSION_SUBSET,
    )
    assert report["overall"] == cs.OVERALL_PASS_THROUGH_DEPTH
    assert report["n_pass"] == 4
    assert report["n_skipped"] == 0
    assert cs.EXPENSIVE_QUALIFICATION not in {r["id"] for r in report["stages"]}
    assert report["establishes_capability"] is False
    assert "not qualification" in report["why_not_capability"]


# ---------------------------------------------------------------------------
# Receipt / --build
# ---------------------------------------------------------------------------


def test_build_writes_parseable_receipt_with_wall_cost_or_unmeasured():
    path = cs.build()
    assert path.parent == RECEIPTS
    assert path.name == cs.RECEIPT
    doc = json.loads(path.read_text())
    assert doc["schema"] == cs.SCHEMA
    assert doc["seal_sha256"]
    assert doc["chooses_component"] is False
    assert doc["skip_is_not_a_pass"] is True
    assert doc["gpu_authority"] is False
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    _assert_no_hardware_claims(doc)
    costs = doc["per_stage_cost"]
    assert [c["id"] for c in costs] == list(cs.STAGE_IDS)
    for row in costs:
        if row["wall_cost"] == "MEASURED":
            assert isinstance(row["wall_seconds"], (int, float))
            assert row["wall_seconds"] >= 0.0
        else:
            assert row["wall_cost"] == "UNMEASURED"
            assert row["reason"]
            assert row["wall_seconds"] is None
        if row["id"] == cs.EXPENSIVE_QUALIFICATION:
            assert row["evaluation_cost"] == "UNMEASURED"
            assert row["reason"]
        assert row["evidence_production_cost"] == "UNMEASURED"
        assert row["evidence_production_reason"]
    fixture = doc["fixture_run"]
    assert fixture["establishes_capability"] is False
    expensive = {r["id"]: r for r in fixture["stages"]}[cs.EXPENSIVE_QUALIFICATION]
    assert expensive["verdict"] == cs.SKIPPED
    assert cs.counts_as_pass(expensive) is False
    for sid in (
        cs.LOCAL_FUNCTIONAL_FIDELITY,
        cs.LOGIT_TOKEN,
        cs.FAST_CAPABILITY,
        cs.HCLI_MISSION_SUBSET,
    ):
        assert {r["id"]: r for r in fixture["stages"]}[sid]["verdict"] == cs.PASS


def test_evaluate_records_measured_wall_seconds_per_attempted_stage():
    report = cs.evaluate({**_component(), **_hidden_ok()}, max_stage=cs.LOCAL_FUNCTIONAL_FIDELITY)
    row = report["stages"][0]
    assert isinstance(row["wall_seconds"], float)
    assert row["wall_seconds"] >= 0.0
    assert row["cost"]["wall_cost"] == "MEASURED"
    assert row["cost"]["evidence_production_cost"] == "UNMEASURED"


def test_module_parses_and_reuses_existing_modules():
    src = Path(cs.__file__).read_text()
    assert "capability_information_map" in src
    assert "resident_provider" in src
    assert "qualification_pipeline" in src
    assert "workunit_species" in src
    assert "aux_capability_screen" in src
    assert "does_not_establish" in src
    reused = cs.build().__class__  # touch
    _ = reused
    doc = json.loads((RECEIPTS / cs.RECEIPT).read_text())
    assert set(doc["reused_not_rebuilt"]) == set(cs.STAGE_IDS)
