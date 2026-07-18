#!/usr/bin/env python3.12
"""Tests for the adaptive EXTREME planner (eco_planner)."""
import pathlib
import sys

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import eco_planner as pl  # noqa: E402


C = pl.CapabilityContract()


def test_selftest_green():
    assert pl.selftest()["ok"] is True


def test_diagnose_cases():
    assert pl.diagnose(0.03, -0.01, C) == "no_material_damage"
    assert pl.diagnose(0.20, -0.02, C) == "signal_degradation"
    assert pl.diagnose(5.0, -0.5, C) == "computation_collapse"
    assert pl.diagnose(0.05, -0.4, C) == "mixed_failure"       # cap collapse only
    assert pl.diagnose(2.0, -0.01, C) == "mixed_failure"       # ppl collapse only
    assert pl.diagnose(None, None, C) == "undetermined"


def test_doctor_program_from_diagnosis():
    assert pl.doctor_program("signal_degradation")["promote"] == ["doctor_static", "doctor_conditional"]
    assert pl.doctor_program("computation_collapse")["promote"] == []
    # controls always retained
    for d in pl.DIAGNOSES:
        assert pl.doctor_program(d)["controls"] == list(pl.CONTROLS)


def _cell(label, params, rate, branch, status, ppl=None, cap=None, phys=None, reason=None):
    return {"model_label": label, "model_family": "qwen2.5-dense", "hf_id": "toy/x",
            "exact_stored_parameter_count": int(params * 1e9), "nominal_params_b": params,
            "rate_id": str(rate), "rate_bpw": rate, "branch": branch, "status": status,
            "physical": {"all_in_model_payload_bpw": phys, "target_physical_bpw": rate},
            "quality_provisional": {"ppl_relative_delta": ppl, "capability_absolute_delta": cap,
                                    "quality_claims_permitted": False},
            "reason_code": reason}


def _plan(cells, cohort=None):
    ledger = {"schema": "hawking.eco.prior_ledger.v1", "campaign_plan_sha256": "0" * 64,
              "ledger_sha256": "0" * 64, "cohort": cohort or [], "cells": cells}
    return pl.build_plan(ledger)


def test_pass_bracket_and_stop():
    plan = _plan([
        _cell("7B", 7.6, 2.0, "codec_control", "complete", ppl=0.05, cap=-0.02, phys=2.3),
        _cell("7B", 7.6, 1.0, "codec_control", "complete", ppl=1.4, cap=-0.4, phys=1.2),
    ])
    a = plan["parents"][0]
    assert a["extreme_candidate"]["status"] == "PROVISIONAL_PASS"
    assert a["extreme_candidate"]["rate_bpw"] == 2.0
    assert a["event_horizon_bracket"] == {"lowest_pass_bpw": 2.0, "highest_fail_below_bpw": 1.0}
    assert a["stop_condition_met"] is True
    # every lower rate has a reason
    assert all("reason" in r for r in a["lower_rate_reasons"])


def test_codec_control_fail_is_unresolved_pending_doctor():
    # a treatable codec_control failure with NO terminal doctor branch is INCONCLUSIVE,
    # not a resolved FAIL, and produces a boundary probe.
    plan = _plan([
        _cell("7B", 7.6, 2.0, "codec_control", "complete", ppl=0.20, cap=-0.02, phys=2.3),
    ])
    a = plan["parents"][0]
    row = next(f for f in a["rate_frontier"] if f["rate_bpw"] == 2.0)
    assert row["verdict"] == "INCONCLUSIVE"
    assert row["resolved"] is False
    assert row["diagnosis"] == "signal_degradation"
    assert any(bp["rate_bpw"] == 2.0 for bp in a["boundary_probes_needed"])


def test_doctor_tried_but_out_of_contract_is_resolved_fail():
    plan = _plan([
        _cell("7B", 7.6, 2.0, "codec_control", "complete", ppl=0.20, cap=-0.02, phys=2.3),
        _cell("7B", 7.6, 2.0, "doctor_full", "complete", ppl=0.15, cap=-0.03, phys=2.35),
    ])
    a = plan["parents"][0]
    row = next(f for f in a["rate_frontier"] if f["rate_bpw"] == 2.0)
    assert row["verdict"] == "FAIL"
    assert row["resolved"] is True


def test_disposition_is_resolved_fail():
    plan = _plan([
        _cell("14B", 14.8, 0.8, "doctor_full", "unsupported",
              reason="empirical-quality-cliff-adaptive-defer"),
    ])
    a = plan["parents"][0]
    row = next(f for f in a["rate_frontier"] if f["rate_bpw"] == 0.8)
    assert row["verdict"] == "FAIL"
    assert row["fail_kind"] == "campaign_deferral"
    assert row["diagnosis"] == "undetermined"      # NOT relabelled computation_collapse
    assert "not a measured collapse" in row["reason"]


def test_disposition_does_not_set_collapse_boundary():
    # regression: a deferral below a pass must NOT become the proven Event-Horizon boundary;
    # only a MEASURED collapse does.
    plan = _plan([
        _cell("14B", 14.8, 2.0, "codec_control", "complete", ppl=0.05, cap=-0.02, phys=2.3),
        _cell("14B", 14.8, 1.0, "doctor_full", "unsupported",
              reason="empirical-quality-cliff-adaptive-defer"),
    ])
    a = plan["parents"][0]
    assert a["extreme_candidate"]["rate_bpw"] == 2.0
    # 1.0 is a deferral, not a measured collapse -> boundary stays None
    assert a["event_horizon_bracket"] == {"lowest_pass_bpw": 2.0, "highest_fail_below_bpw": None}
    # but with a MEASURED collapse at 1.0 the boundary is asserted
    plan2 = _plan([
        _cell("14B", 14.8, 2.0, "codec_control", "complete", ppl=0.05, cap=-0.02, phys=2.3),
        _cell("14B", 14.8, 1.0, "codec_control", "complete", ppl=5.0, cap=-0.5, phys=1.2),
    ])
    a2 = plan2["parents"][0]
    assert a2["event_horizon_bracket"]["highest_fail_below_bpw"] == 1.0


def test_unproven_when_no_evidence():
    plan = _plan([
        _cell("32B", 32.5, 4.0, "codec_control", "complete", ppl=None, cap=None, phys=4.1),
    ])
    a = plan["parents"][0]
    # 4.0 has physical only -> INCONCLUSIVE; lower rates UNPROVEN
    r4 = next(f for f in a["rate_frontier"] if f["rate_bpw"] == 4.0)
    r2 = next(f for f in a["rate_frontier"] if f["rate_bpw"] == 2.0)
    assert r4["verdict"] == "INCONCLUSIVE"
    assert r2["verdict"] == "UNPROVEN"
    assert a["extreme_candidate"]["status"] == "UNPROVEN"


def test_awaiting_evidence_from_cohort():
    cohort = [{"label": "7B", "family": "qwen2.5-dense", "hf_id": "x", "nominal_params_b": 7.6},
              {"label": "72B", "family": "qwen2.5-dense", "hf_id": "y", "nominal_params_b": 72.0}]
    plan = _plan([
        _cell("7B", 7.6, 2.0, "codec_control", "complete", ppl=0.05, cap=-0.02, phys=2.3),
        _cell("7B", 7.6, 1.0, "codec_control", "complete", ppl=1.4, cap=-0.4, phys=1.2),
    ], cohort=cohort)
    awaiting = {a["model_label"] for a in plan["parents_awaiting_evidence"]}
    assert "72B" in awaiting
    assert "7B" not in awaiting  # has evidence


def test_byte_ceiling_from_device():
    dev = pl.default_device()
    # 7.6B at 78GB budget -> ceiling ~ 82 bpw (very loose; the small models are never byte-bound)
    assert dev.byte_ceiling_bpw(7.6) > 50
    assert dev.byte_ceiling_bpw(405.0) < 2.0  # 405B is byte-bound below 2 bpw resident
