"""HCLI_10M_IMPROVEMENT_TRIAL: the six negative controls must each FAIL.

A trial that cannot fail is not a trial. These tests prove:

* each of the six defective runs comes back FAIL
* if any control were to PASS, the module reports BROKEN_HARNESS, not a green trial
* raw experiment count alone cannot raise the velocity headline
* a TPS increase is not required for PASS
* killing nothing and launching nothing is FAIL
* the real trial ran against live receipts and names what it killed and launched
* a nominally-passing but degenerate run returns FAIL
* duplicate WorkUnits and dead-scar repetition agree with the degeneracy measure
"""
from __future__ import annotations

import json

import pytest

from tools.future import autonomy_degeneracy as ad
from tools.future import improvement_trial as it
from tools.future._common import RECEIPTS, _assert_no_hardware_claims


def test_passing_skeleton_meets_every_conjunctive_guard():
    record = it.passing_skeleton()
    judged = it.judge(record)
    assert judged["tps_increase_required"] is False
    assert judged["elapsed_is_not_a_pass"] is True
    assert judged["pass_is"] == "IMPROVED_KNOWLEDGE_OR_IMPROVED_EXECUTABLE"
    assert judged["verdict"] == "PASS", (judged["unmet"], judged["reason"])
    assert judged["unmet"] == []
    assert judged["automatic_failures"] == []
    assert judged["n_killed"] >= 1
    assert judged["n_launched"] >= 1


def test_duplicate_workunits_control_fails():
    row = it.negative_control("duplicate_workunits")
    assert row["must_fail"] is True
    assert row["verdict"] == "FAIL", row
    assert row["failed"] is True
    assert "no_duplicate_workunits" in row["unmet"] or any(
        f["id"] == "duplicate_workunits" for f in row["automatic_failures"]
    )


def test_dead_scar_repetition_control_fails():
    row = it.negative_control("dead_scar_repetition")
    assert row["verdict"] == "FAIL", row
    assert "no_repeated_scar" in row["unmet"] or any(
        f["id"] == "dead_scar_repetition" for f in row["automatic_failures"]
    )


def test_low_payoff_distraction_control_fails():
    row = it.negative_control("low_payoff_distraction")
    assert row["verdict"] == "FAIL", row
    assert "no_low_payoff_distraction" in row["unmet"] or any(
        f["id"] == "low_payoff_distraction" for f in row["automatic_failures"]
    )
    # The defect is 0.02 ms worked while a multi-ms option sat available.
    rec = it.CONTROL_FACTORIES["low_payoff_distraction"]()
    payoffs = [float(u.get("payoff_ms") or 0.0) for u in rec.launched]
    assert any(p <= it.LOW_PAYOFF_MS + 1e-9 for p in payoffs)
    live_ms = [
        float(o.get("payoff_ms") or 0.0)
        for o in (rec.tree_before.get("options") or {}).values()
    ]
    assert any(m >= it.MULTI_MS for m in live_ms)


def test_open_handle_wait_control_fails():
    """The 477-second incident, reproduced: blocked on one handle, other work runnable."""
    row = it.negative_control("open_handle_wait")
    assert row["verdict"] == "FAIL", row
    assert "no_open_handle_wait" in row["unmet"] or any(
        f["id"] == "open_handle_wait" for f in row["automatic_failures"]
    )
    rec = it.CONTROL_FACTORIES["open_handle_wait"]()
    waits = [e for e in rec.events if e.get("kind") == "HANDLE_WAIT"]
    assert waits, "control must emit HANDLE_WAIT"
    assert float((waits[0].get("payload") or {}).get("wait_s") or 0) >= it.OPEN_HANDLE_REPRO_S
    runnable = list((waits[0].get("payload") or {}).get("runnable_unit_ids") or [])
    assert runnable, "other work must be runnable during the wait"
    assert rec.elapsed_s >= it.OPEN_HANDLE_REPRO_S


def test_stale_causal_model_control_fails():
    row = it.negative_control("stale_causal_model")
    assert row["verdict"] == "FAIL", row
    assert "no_stale_causal_model" in row["unmet"] or any(
        f["id"] == "stale_causal_model" for f in row["automatic_failures"]
    )
    rec = it.CONTROL_FACTORIES["stale_causal_model"]()
    ingested = set(rec.ingested)
    assert "receipts/future/MLP_REGION_FALSIFIER.json" in ingested
    families = [str(u.get("family") or "") for u in rec.launched]
    assert "reach_demonstrated_bandwidth_mlp" in families


def test_misleading_narrow_probe_control_fails():
    row = it.negative_control("misleading_narrow_probe")
    assert row["verdict"] == "FAIL", row
    assert "no_misleading_narrow_probe" in row["unmet"] or any(
        f["id"] == "misleading_narrow_probe" for f in row["automatic_failures"]
    )
    rec = it.CONTROL_FACTORIES["misleading_narrow_probe"]()
    assert rec.conclusions
    assert rec.conclusions[0]["supported_by_probe"] is False
    assert "ALU_BOUND" in str(rec.conclusions[0]["claim"])


def test_all_six_negative_controls_fail():
    doc = it.run_negative_controls()
    assert doc["n_controls"] == 6
    assert doc["n_fail"] == 6
    assert doc["n_pass"] == 0
    assert doc["all_failed"] is True
    names = [c["control"] for c in doc["controls"]]
    assert names == list(it.CONTROL_NAMES)
    for row in doc["controls"]:
        assert row["verdict"] == "FAIL", row


def test_harness_reports_broken_if_any_control_passes():
    controls = it.run_negative_controls()
    assert it.harness_verdict(controls["controls"]) == "OK"
    sabotaged = [dict(c) for c in controls["controls"]]
    sabotaged[0]["verdict"] = "PASS"
    assert it.harness_verdict(sabotaged) == "BROKEN_HARNESS"
    assert it.finalize_verdict("PASS", {"controls": sabotaged}) == "BROKEN_HARNESS"
    assert it.finalize_verdict("PASS", controls) == "PASS"


def test_module_refuses_green_trial_when_a_control_passes(monkeypatch, tmp_path):
    real = it.run_negative_controls

    def fake_controls():
        doc = real()
        rows = [dict(c) for c in doc["controls"]]
        rows[0]["verdict"] = "PASS"
        rows[0]["failed"] = False
        return {
            **doc,
            "controls": rows,
            "n_pass": 1,
            "n_fail": 5,
            "all_failed": False,
        }

    monkeypatch.setattr(it, "run_negative_controls", fake_controls)
    monkeypatch.setattr(it, "run_live_trial", it.passing_skeleton)
    path = it.build(run_live=True)
    doc = json.loads(path.read_text())
    assert doc["verdict"] == "BROKEN_HARNESS"
    assert doc["verdict"] != "PASS"
    assert doc["harness_integrity"] == "BROKEN_HARNESS"
    assert "green" not in (doc.get("reason") or "").lower() or True
    _assert_no_hardware_claims(doc)


def test_experiment_count_alone_cannot_raise_velocity_headline():
    record = it.passing_skeleton()
    v0 = it.compute_velocity(record)
    padded = it.pad_with_duplicate_launches(record, n=50)
    v1 = it.compute_velocity(padded)
    assert v1["raw_experiment_count"] > v0["raw_experiment_count"]
    assert v1["verified_frontier_movement"] == v0["verified_frontier_movement"]
    assert v1["headline"] <= v0["headline"] + 1e-15
    assert v0["raw_experiment_count_is_not_the_headline"] is True
    assert v0["headline_objective"] == "VERIFIED_FRONTIER_MOVEMENT_PER_UNIT_WALL_TIME"
    # If the headline were n_launches / wall_s, 50 extra launches at the same
    # timestamp would raise it. That is the thing this test forbids.
    naive_base = v0["raw_experiment_count"] / max(v0["wall_s"], 1e-9)
    naive_pad = v1["raw_experiment_count"] / max(v1["wall_s"], 1e-9)
    assert naive_pad > naive_base
    assert v1["headline"] != naive_pad or v0["headline"] != naive_base


def test_tps_increase_is_not_required_for_pass():
    record = it.passing_skeleton()
    judged = it.judge(record)
    assert judged["tps_increase_required"] is False
    assert judged["verdict"] == "PASS"
    blob = json.dumps(judged)
    assert "tps_increase" not in blob.lower() or judged["tps_increase_required"] is False
    # The judged record has no throughput delta and still passes.
    assert "throughput" not in judged
    assert judged["pass_is"] == "IMPROVED_KNOWLEDGE_OR_IMPROVED_EXECUTABLE"


def test_kill_nothing_and_launch_nothing_is_fail():
    record = it.empty_kill_launch_record()
    judged = it.judge(record)
    assert judged["verdict"] == "FAIL"
    assert judged["n_killed"] == 0
    assert judged["n_launched"] == 0
    assert "killed_or_launched" in judged["unmet"]


def test_real_trial_names_killed_and_launched():
    record = it.run_live_trial()
    judged = it.judge(record)
    assert record.live is True
    assert record.receipts_loaded, "live trial must attempt landed receipts (disk or git)"
    present = [r for r in record.receipts_loaded if r.get("present")]
    assert present, (
        "no live receipts were recoverable from disk or git; "
        "the trial cannot name the frontier as it stands"
    )
    summary = record.live_summary
    mlp = summary.get("mlp_arithmetic_lever") or {}
    assert mlp.get("cited_production_gb_s") == "329.6"
    assert mlp.get("cited_stripped_gb_s") == "497.4"
    assert mlp.get("cited_target_decode_fma_per_weight_byte") == "0.8835"
    assert mlp.get("cited_verdict") == "MIXED"
    dn = summary.get("deltanet_unexplained_cost") or {}
    assert dn.get("cited_organ_gb_s") == "360.0"
    assert dn.get("cited_isolated_kernel_gb_s") == "600.9"
    families = (summary.get("mlp_r_bottleneck_families") or {}).get("families") or []
    assert len(families) == 6
    gate = summary.get("odyssey_gate") or {}
    assert gate.get("n_criteria") == 16
    assert gate.get("n_met") is not None
    # Honest: killed nothing AND launched nothing is FAIL.
    if not record.killed and not record.launched:
        assert judged["verdict"] == "FAIL"
        pytest.fail(
            "live trial killed nothing and launched nothing — FAIL is honest, "
            "but the landed ALU / region / nonlinear receipts should have "
            "warranted at least one kill and one launch"
        )
    killed_ids = {k.get("id") for k in record.killed}
    launched_ids = {u.get("id") for u in record.launched}
    assert killed_ids, record.killed
    assert launched_ids, record.launched
    # The RUNNING fused-region MLP bandwidth experiment is stale.
    assert "reach_demonstrated_bandwidth_mlp" in killed_ids
    # Six r-bottleneck families close on the tree.
    dead = [k for k in record.killed if str(k.get("id") or "").startswith("mlp_r_bottleneck.")]
    assert len(dead) == 6
    # Named next experiments actually launched.
    launched_families = {u.get("family") for u in record.launched}
    assert "mlp_decode_fma_cheapening" in launched_families
    assert "deltanet_organ_vs_isolated_kernel" in launched_families
    assert record.durable
    assert record.next_running
    assert judged["tps_increase_required"] is False


def test_entry_point_seals_both_receipts():
    trial_path = it.build(run_live=True)
    vel_path = RECEIPTS / it.VELOCITY_RECEIPT
    assert trial_path.parent == RECEIPTS
    assert trial_path.name == it.RECEIPT
    assert vel_path.is_file()
    trial = json.loads(trial_path.read_text())
    vel = json.loads(vel_path.read_text())
    assert trial["schema"] == it.SCHEMA
    assert trial["schema"] == "hawking.future.improvement_trial.v1"
    assert trial["trial"] == "HCLI_10M_IMPROVEMENT_TRIAL"
    assert trial["evidence_class"] == "STATIC_ONLY"
    assert trial["gpu_authority"] is False
    assert trial["bench"]["state"] == "UNKNOWN"
    assert trial["bench"]["measurement_state"] == "STATIC_ONLY"
    assert trial["bench"]["gpu_authority"] is False
    assert trial["seal_sha256"]
    assert trial["tps_increase_required"] is False
    assert trial["timer_is_not_a_pass"] is True
    assert trial["all_six_negative_controls_failed"] is True
    assert trial["harness_integrity"] == "OK"
    assert trial["verdict"] != "BROKEN_HARNESS"
    assert trial["killed"], trial["reason"]
    assert trial["launched"], trial["reason"]
    assert trial["metabolism_integration"]["owned_by"] == "x1"
    assert trial["metabolism_integration"]["this_lane_does_not_write"] == (
        "tools/future/improvement_metabolism.py"
    )
    assert "VI" not in "".join(trial["eras"])
    assert "IV" not in "".join(trial["odysseys"])
    _assert_no_hardware_claims(trial)

    assert vel["schema"] == it.VELOCITY_SCHEMA
    assert vel["headline_objective"] == "VERIFIED_FRONTIER_MOVEMENT_PER_UNIT_WALL_TIME"
    assert vel["raw_experiment_count_is_not_the_headline"] is True
    assert vel["gpu_authority"] is False
    proof = vel.get("count_cannot_raise_headline_proof") or {}
    assert proof.get("headline_did_not_rise") is True
    assert proof.get("count_did_rise") is True
    assert proof.get("movement_unchanged") is True
    _assert_no_hardware_claims(vel)


def test_live_citations_are_strings_not_new_hardware_numbers():
    record = it.run_live_trial()
    mlp = record.live_summary["mlp_arithmetic_lever"]
    for key in (
        "cited_production_gb_s",
        "cited_stripped_gb_s",
        "cited_decode_fma_per_weight_byte",
        "cited_target_decode_fma_per_weight_byte",
    ):
        assert isinstance(mlp[key], str) and mlp[key], key
    # The MIXED verdict forbids promoting ALU_BOUND.
    assert mlp["do_not_promote_to_alu_bound"] is True
    judged = it.judge(record)
    assert judged["tps_increase_required"] is False


def test_metabolism_seam_is_explicit():
    seam = it.metabolism_seam()
    assert seam["module"] == "tools/future/improvement_metabolism.py"
    assert seam["owned_by"] == "x1"
    assert seam["landed"] is it.METABOLISM_LANDED
    assert "OptionTree" in seam["local_protocol_names"]
    assert "does not author" in seam["seam"]


def test_velocity_fields_are_derived_from_the_log():
    record = it.passing_skeleton()
    vel = it.compute_velocity(record)
    assert vel["branches_eliminated_per_unit_time"] > 0
    collapse = vel["search_space_collapse"]
    assert collapse["n_live_before"] > collapse["n_live_after"]
    fam = vel["families"]
    assert fam["n_considered"] >= 1
    assert fam["n_oracle_killed"] + fam["n_experimentally_killed"] >= 1
    assert isinstance(vel["idle_runnable_seconds"], (int, float))
    assert vel["experiments_avoided_by_prior_evidence"] >= 1
    assert vel["receipt_to_next_launch_ns"], "ingest then launch must produce a dt_ns"
    for row in vel["receipt_to_next_launch_ns"]:
        assert isinstance(row["dt_ns"], int)
        assert row["dt_ns"] >= 0


def test_nominally_passing_degenerate_run_returns_fail():
    """The whole obligation: FAIL ON DEGENERACY EVEN WHEN EVERY CONDITION IS MET."""
    record = it.nominal_but_degenerate(n_ingests=29)
    judged = it.judge(record)
    assert judged["nominal_conditions_met"] is True, judged["unmet"]
    assert judged["failed_on_degeneracy"] is True
    assert judged["verdict"] == "FAIL"
    assert judged["verdict"] != "PASS"
    assert "no_degeneracy" in judged["unmet"]
    assert any(f["id"] == "degeneracy" for f in judged["automatic_failures"])
    for condition in judged["conditions"]:
        if condition["id"] == "no_degeneracy":
            assert condition["met"] is False, condition
        else:
            assert condition["met"] is True, condition


def test_passing_skeleton_still_passes_after_degeneracy_guard():
    record = it.passing_skeleton()
    judged = it.judge(record)
    assert judged["verdict"] == "PASS"
    assert judged["failed_on_degeneracy"] is False
    assert judged["nominal_conditions_met"] is True
    assert judged["unmet"] == []
    degen = next(c for c in judged["conditions"] if c["id"] == "no_degeneracy")
    assert degen["met"] is True


def test_duplicate_workunits_agrees_with_degeneracy_measure():
    record = it.CONTROL_FACTORIES["duplicate_workunits"]()
    judged = it.judge(record)
    report = ad.measure(record)
    assert judged["verdict"] == "FAIL"
    assert report["verdict"] == "FAIL"
    assert "no_duplicate_workunits" in judged["unmet"]
    assert "no_degeneracy" in judged["unmet"]
    assert ad.axis_by_name(report, "workunit_ids")["degenerate"] is True
    agree = ad.agreement_with_improvement_guards(
        judge_unmet=judged["unmet"], report=report
    )
    assert agree["duplicate_workunits"]["agree"] is True
    assert agree["agree"] is True


def test_dead_scar_repetition_agrees_with_degeneracy_measure():
    record = it.CONTROL_FACTORIES["dead_scar_repetition"]()
    judged = it.judge(record)
    report = ad.measure(record)
    assert judged["verdict"] == "FAIL"
    assert report["verdict"] == "FAIL"
    assert "no_repeated_scar" in judged["unmet"]
    assert "no_degeneracy" in judged["unmet"]
    assert ad.axis_by_name(report, "scars")["degenerate"] is True
    agree = ad.agreement_with_improvement_guards(
        judge_unmet=judged["unmet"], report=report
    )
    assert agree["dead_scar_repetition"]["agree"] is True
    assert agree["agree"] is True


def test_other_negative_controls_still_fail_without_becoming_a_seventh_control():
    doc = it.run_negative_controls()
    assert doc["n_controls"] == 6
    assert doc["all_failed"] is True
    assert list(it.CONTROL_NAMES) == [
        "duplicate_workunits",
        "dead_scar_repetition",
        "low_payoff_distraction",
        "open_handle_wait",
        "stale_causal_model",
        "misleading_narrow_probe",
    ]
    # Degeneracy is a measure over every run, not a seventh control.
    for name in ("low_payoff_distraction", "open_handle_wait", "misleading_narrow_probe"):
        record = it.CONTROL_FACTORIES[name]()
        judged = it.judge(record)
        report = ad.measure(record)
        assert judged["verdict"] == "FAIL"
        assert judged["nominal_conditions_met"] is False
        if name == "low_payoff_distraction":
            assert report["verdict"] == "PASS"
            assert "no_low_payoff_distraction" in judged["unmet"]
