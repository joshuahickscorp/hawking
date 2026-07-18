#!/usr/bin/env python3.12
"""Tests for the empirical layer: harvest, evidence-closed retirement, ETA, engine wiring."""
import pathlib
import sys

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import succ_harvest as hv  # noqa: E402
import succ_retire as rt  # noqa: E402
import succ_engine as eng  # noqa: E402
from eco_common import seal_field  # noqa: E402


def test_harvest_selftest():
    r = hv.selftest()
    assert r["ok"] is True
    assert r["wall_time_from_receipt"] and r["treatment_vs_control"] and r["classification"]


def test_retire_selftest():
    r = rt.selftest()
    assert r["ok"] is True
    assert r["collapse_boundary_7B"] == 2.0 and r["additive_only"] is True


def test_classification_never_conflates_deferral_with_collapse():
    # a scheduling disposition must classify as scheduling_deferral, not collapse
    assert rt.__doc__  # module present
    # a measured collapse (ppl doubled) classifies as computation collapse
    assert hv._classify_complete(5.0, -0.5, {}) == "measured_computation_collapse"
    # a deferred (no ppl) classifies as missing_evaluation, NOT collapse
    assert hv._classify_complete(None, None, {}) == "missing_evaluation"
    # within contract -> no material damage
    assert hv._classify_complete(0.05, -0.02, {}) == "measured_no_material_damage"
    # a disposition reason with adaptive-defer -> scheduling_deferral (not collapse)
    assert hv._classify_disposition("empirical-quality-cliff-adaptive-defer") == "scheduling_deferral"


def test_eta_observations_are_per_segment_not_global():
    ds = {"schema": "hawking.successor.empirical_harvest.v1", "rows": [
        {"status": "complete", "branch": "codec_control", "nominal_target_bpw": 4.0,
         "wall_seconds": 3600, "geometry": {"stored_parameters": 7_600_000_000}},
        {"status": "complete", "branch": "doctor_static", "nominal_target_bpw": 4.0,
         "wall_seconds": 9000, "geometry": {"stored_parameters": 7_600_000_000}},
        {"status": "complete", "branch": "codec_control", "nominal_target_bpw": 2.0,
         "wall_seconds": None, "geometry": {"stored_parameters": 7_600_000_000}},  # no wall -> skipped
    ]}
    obs = hv.eta_observations(ds)
    assert len(obs) == 2  # the None-wall row is skipped
    branches = {o["branch"] for o in obs}
    assert branches == {"codec_control", "doctor_static"}
    assert all(o["stored_params_b"] == 7.6 for o in obs)


def test_engine_skips_retired_experiments():
    # a plan with a 7B boundary probe at 1.0 bpw, and a retirement ledger that retired it
    plan = {"parents": [{
        "binding": {"model_label": "7B"}, "params_b": 7.6,
        "event_horizon_bracket": {"lowest_pass_bpw": None},
        "boundary_probes_needed": [
            {"rate_bpw": 1.0, "current_verdict": "UNPROVEN", "next_feasibility_tier": "F0",
             "doctor_program": {"promote": []}},
            {"rate_bpw": 4.0, "current_verdict": "INCONCLUSIVE",
             "next_feasibility_tier": "F3_full_model_quality", "doctor_program": {"promote": []}}]}]}
    # retirement ledger retiring 7B at 1.0 bpw (below a collapse boundary)
    ledger = {"retired_keys": [f"7B__1.0bpw__{b}" for b in
                              ("codec_control", "doctor_static", "doctor_conditional", "doctor_full")]}
    no_ret = eng.next_experiment(plan)
    with_ret = eng.next_experiment(plan, retirement_ledger=ledger)
    assert no_ret["considered"] == 2
    assert with_ret["considered"] == 1  # the 1.0 bpw probe is skipped
    assert with_ret["skipped_retired"] == 1
    assert with_ret["selected"]["rate_bpw"] == 4.0  # only the non-retired probe remains


def test_retirement_receipts_preserve_evidence_and_reopening():
    r = rt.selftest()
    assert r["preserves_reopening"] is True
