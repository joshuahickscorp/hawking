#!/usr/bin/env python3.12
"""Tests for the 120B+ admission planner (eco_admission)."""
import pathlib
import sys

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import eco_admission as adm  # noqa: E402
from eco_common import sealed  # noqa: E402


def test_selftest_green():
    assert adm.selftest()["ok"] is True


def test_plan_is_sealed_and_covers_parents():
    plan = adm.build_admission_plan()
    assert sealed(plan, "admission_sha256")
    labels = {p["parent"]["label"] for p in plan["parents"]}
    assert {"120B", "235B-A22B", "405B", "671B"} <= labels


def test_gptoss_adapter_built_others_must_build():
    plan = adm.build_admission_plan()
    gptoss = next(p for p in plan["parents"] if p["parent"]["label"] == "120B")
    assert gptoss["adapter"]["status"] == "built"
    assert "llama-dense" in plan["must_build_adapter"]
    assert "deepseek-moe" in plan["must_build_adapter"]


def test_each_parent_has_lifecycle_and_evidence_requirement():
    plan = adm.build_admission_plan()
    for p in plan["parents"]:
        phases = {ph["phase"] for ph in p["streamed_lifecycle"]}
        assert {"procure", "streamed_bake", "seal", "capability_eval", "source_release"} <= phases
        req = p["admission_gate"]["quality_evidence_required"]
        assert "native_load_parity" in req and "F4_replicated_seal" in req


def test_scaling_prior_seeds_candidate_labeled_prior_only():
    prior = {"log10_slope_bpw_per_decade": -0.8,
             "points": [{"model_label": "14B", "params_b": 14.8, "provisional_floor_bpw": 2.0,
                         "log10_params": 10.17}]}
    plan = adm.build_admission_plan(scaling_prior=prior)
    seeded = [p for p in plan["parents"] if p["candidate_basis"] == "scaling_prior_scheduling_only"]
    assert seeded
    for p in seeded:
        assert p["admission_gate"]["candidate_rate_is_prior_only"] is True


def test_admissible_now_is_disk_consistent():
    # regression: admissible_now must not contradict a disk blocker
    plan = adm.build_admission_plan()
    for p in plan["parents"]:
        if p["admissible_now"]:
            assert p["admission_gate"]["disk_feasible_today"] is True
            assert not any("storage budget" in b for b in p["blockers"])


def test_dense_405b_blocked():
    plan = adm.build_admission_plan()
    llama = next(p for p in plan["parents"] if p["parent"]["label"] == "405B")
    assert llama["admissible_now"] is False
    assert any("adapter" in b for b in llama["blockers"])
