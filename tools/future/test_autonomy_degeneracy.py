"""AUTONOMY EVIDENCE MUST BE NON-DEGENERATE.

These tests replay two receipts already on disk — no fixtures — and prove
the measure FAILs the 1h timeline, PASSes the detached-work trial, reports
a per-axis distinct-vs-repeated table rather than a score, counts
unlabelled/argv[0] units as degenerate, and FAILs a run that meets every
nominal condition while looping one receipt.
"""
from __future__ import annotations

import json
from pathlib import Path

from tools.future import autonomy_degeneracy as ad
from tools.future import improvement_trial as it
from tools.future._common import RECEIPTS, REPO, _assert_no_hardware_claims


TIMELINE_1H = REPO / ad.TIMELINE_1H_REL
DETACHED = REPO / ad.DETACHED_TRIAL_REL


def test_disk_timelines_are_real_receipts_not_fixtures():
    assert TIMELINE_1H.is_file(), TIMELINE_1H
    assert DETACHED.is_file(), DETACHED
    one_h = json.loads(TIMELINE_1H.read_text())
    detached = json.loads(DETACHED.read_text())
    assert one_h.get("trial") == "1h" or one_h.get("duration_s") == 3600
    assert isinstance(one_h.get("events"), list) and len(one_h["events"]) > 100
    assert detached.get("schema") == "hawking.future.detached_trial.v1"
    assert isinstance(detached.get("timeline"), list) and len(detached["timeline"]) > 10
    assert detached.get("fixture") is False


def test_1h_timeline_replayed_from_disk_fails():
    report = ad.measure(TIMELINE_1H)
    assert report["verdict"] == "FAIL", report["reason"]
    assert report["score"] is None
    assert report["table_not_a_score"] is True
    degenerate = set(report["degenerate_axes"])
    for axis in ("rejections", "refills", "ingestion", "launches"):
        assert axis in degenerate, (axis, report["degenerate_axes"], report["reason"])
    rejections = ad.axis_by_name(report, "rejections")
    assert rejections["total"] == 222
    assert rejections["unique"] < rejections["total"]
    assert rejections["unique_ratio"] < ad.REJECTION_MIN_UNIQUE_RATIO
    assert rejections["largest_repeat_run"] > ad.REJECTION_MAX_CONSECUTIVE_RUN
    assert rejections["consecutive_emissions_identical"] is True
    assert rejections["early_cluster"] is True
    refills = ad.axis_by_name(report, "refills")
    assert refills["total"] == 4
    assert refills["consecutive_emissions_identical"] is True
    assert refills["stopped_early"] is True
    ingestion = ad.axis_by_name(report, "ingestion")
    assert ingestion["max_item_count"] > ad.INGEST_MAX_REPEATS
    assert report["specimen_verification_ingests"] >= 29
    assert "SPECIMEN_VERIFICATION" in str(ingestion.get("most_repeated") or "")


def test_detached_timeline_replayed_from_disk_passes():
    report = ad.measure(DETACHED)
    assert report["verdict"] == "PASS", (report["reason"], report["degenerate_axes"])
    assert report["degenerate_axes"] == []
    assert report["n_argv0_labelled"] == 0
    assert report["n_unlabelled"] == 0
    refills = ad.axis_by_name(report, "refills")
    assert refills["total"] == 49
    assert refills["unique"] == 49
    assert refills["consecutive_emissions_identical"] is False
    assert refills["largest_repeat_run"] == 1
    wu = ad.axis_by_name(report, "workunit_ids")
    assert wu["unique"] == wu["total"]
    assert wu["unique"] >= 50
    rejections = ad.axis_by_name(report, "rejections")
    assert rejections["total"] == 0
    assert rejections["degenerate"] is False


def test_per_axis_table_is_not_a_score():
    report = ad.measure(TIMELINE_1H)
    table = ad.axis_table(report)
    assert isinstance(table, list)
    assert len(table) >= len(ad.NAMED_AXES)
    names = [row["axis"] for row in table]
    for axis in ad.NAMED_AXES:
        assert axis in names, axis
    for row in table:
        assert "unique" in row and "total" in row
        assert "largest_repeat_run" in row
        assert "consecutive_emissions_identical" in row
        assert "degenerate" in row
        assert "score" not in row
    assert report.get("score") is None
    # A single number cannot describe a trial healthy on three axes and dead
    # on the fourth. The 1h workunit_ids axis is the healthy one.
    wu = ad.axis_by_name(report, "workunit_ids")
    assert wu["degenerate"] is False
    assert wu["unique"] == wu["total"]
    launches = ad.axis_by_name(report, "launches")
    assert launches["degenerate"] is True


def test_unlabelled_and_argv0_units_are_degenerate_and_counted():
    report = ad.measure(TIMELINE_1H)
    launches = ad.axis_by_name(report, "launches")
    assert launches["n_argv0_labelled"] >= 29
    assert launches["n_unlabelled_or_argv0"] >= 29
    assert report["unlabelled_or_argv0_units"] == launches["n_unlabelled_or_argv0"]
    assert "python3" in (launches.get("argv0_examples") or [])
    assert launches["degenerate"] is True
    healthy = ad.measure(DETACHED)
    assert healthy["n_argv0_labelled"] == 0
    assert healthy["n_unlabelled"] == 0
    assert ad.axis_by_name(healthy, "launches")["degenerate"] is False


def test_nominally_passing_degenerate_run_returns_fail():
    record = it.nominal_but_degenerate(n_ingests=29)
    judged = it.judge(record)
    assert judged["nominal_conditions_met"] is True, judged["unmet"]
    assert judged["failed_on_degeneracy"] is True
    assert judged["verdict"] == "FAIL"
    assert "no_degeneracy" in judged["unmet"]
    for condition in judged["conditions"]:
        if condition["id"] == "no_degeneracy":
            assert condition["met"] is False
        else:
            assert condition["met"] is True, condition
    report = ad.measure(record)
    assert report["verdict"] == "FAIL"
    assert "ingestion" in report["degenerate_axes"]
    assert report["specimen_verification_ingests"] >= 29


def test_passing_skeleton_is_not_degenerate():
    record = it.passing_skeleton()
    judged = it.judge(record)
    report = ad.measure(record)
    assert report["verdict"] == "PASS", report["reason"]
    assert judged["verdict"] == "PASS"
    assert judged["failed_on_degeneracy"] is False
    assert judged["nominal_conditions_met"] is True
    assert "no_degeneracy" not in judged["unmet"]


def test_replay_disk_timelines_are_opposite_verdicts():
    replay = ad.replay_disk_timelines()
    assert replay["fixtures"] is False
    assert replay["autonomy_1h"]["verdict"] == "FAIL"
    assert replay["detached_work_trial"]["verdict"] == "PASS"
    assert replay["opposite_verdicts"] is True


def test_thresholds_are_explicit_and_defended():
    assert ad.INGEST_MAX_REPEATS == 4
    assert ad.LABELLING_ARGV0_OR_UNLABELLED_MAX == 0
    assert ad.REFILL_IDENTICAL_CONSECUTIVE_MAX == 0
    assert ad.DEAD_SCAR_RELAUNCHES_MAX == 0
    assert "ingest_max_repeats_per_receipt" in ad.THRESHOLD_DEFENSE
    assert "29" in ad.THRESHOLD_DEFENSE["ingest_max_repeats_per_receipt"]
    assert "python3" in ad.THRESHOLD_DEFENSE["labelling_argv0_or_unlabelled_max"]
    one_h = ad.measure(TIMELINE_1H)
    detached = ad.measure(DETACHED)
    # The line is where the 1h timeline falls on the wrong side and an honest
    # run does not.
    assert ad.axis_by_name(one_h, "ingestion")["max_item_count"] > ad.INGEST_MAX_REPEATS
    skeleton = ad.measure(it.passing_skeleton())
    assert ad.axis_by_name(skeleton, "ingestion")["max_item_count"] <= ad.INGEST_MAX_REPEATS
    assert detached["verdict"] == "PASS"
    assert one_h["verdict"] == "FAIL"


def test_is_argv0_label_detects_interpreter_basenames_only():
    assert ad.is_argv0_label("python3") is True
    assert ad.is_argv0_label("/opt/homebrew/opt/python@3.14/bin/python3.14") is True
    assert ad.is_argv0_label("python3.14") is True
    assert ad.is_argv0_label("bash") is True
    assert ad.is_argv0_label("WU.DETACHED_TRIAL.ind.0004") is False
    assert ad.is_argv0_label("specimen_verify.py") is False
    assert ad.is_argv0_label("") is False
    assert ad.is_argv0_label(None) is False


def test_duplicate_and_dead_scar_mechanisms_agree():
    dup = it.CONTROL_FACTORIES["duplicate_workunits"]()
    scar = it.CONTROL_FACTORIES["dead_scar_repetition"]()
    dup_j, scar_j = it.judge(dup), it.judge(scar)
    dup_r, scar_r = ad.measure(dup), ad.measure(scar)
    assert dup_j["verdict"] == "FAIL" and dup_r["verdict"] == "FAIL"
    assert scar_j["verdict"] == "FAIL" and scar_r["verdict"] == "FAIL"
    dup_ag = ad.agreement_with_improvement_guards(
        judge_unmet=dup_j["unmet"], report=dup_r
    )
    scar_ag = ad.agreement_with_improvement_guards(
        judge_unmet=scar_j["unmet"], report=scar_r
    )
    assert dup_ag["agree"] is True
    assert scar_ag["agree"] is True
    assert ad.axis_by_name(dup_r, "workunit_ids")["degenerate"] is True
    assert ad.axis_by_name(scar_r, "scars")["degenerate"] is True


def test_entry_point_seals_receipt_with_defended_thresholds():
    path = ad.build()
    assert path.parent == RECEIPTS
    assert path.name == ad.RECEIPT
    doc = json.loads(path.read_text())
    assert doc["schema"] == ad.SCHEMA
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["table_not_a_score"] is True
    assert doc["fails_on_degeneracy_even_when_nominal_conditions_met"] is True
    assert doc["replay"]["autonomy_1h"]["verdict"] == "FAIL"
    assert doc["replay"]["detached_work_trial"]["verdict"] == "PASS"
    assert doc["replay"]["opposite_verdicts"] is True
    assert doc["replay"]["fixtures"] is False
    assert doc["thresholds"]["ingest_max_repeats_per_receipt"] == 4
    assert "ingest_max_repeats_per_receipt" in doc["threshold_defense"]
    assert doc["one_h_measured"]["n_argv0_labelled"] >= 29
    assert doc["one_h_measured"]["specimen_verification_ingests"] >= 29
    assert doc["verdict"] == "PASS"
    assert doc["seal_sha256"]
    _assert_no_hardware_claims(doc)
    named = doc["replay"]["autonomy_1h"]["named_axis_table"]
    axes = {row["axis"] for row in named}
    for axis in ad.NAMED_AXES:
        assert axis in axes


def test_does_not_claim_to_edit_detached_or_autonomy_run():
    src = Path(ad.__file__).read_text(encoding="utf-8")
    assert "tools/future/detached_trial.py" in src
    assert "were not edited" in src or "Does not edit" in src or "does_not_edit" in src
    doc = json.loads(ad.build().read_text())
    assert "tools/future/detached_trial.py" in doc["does_not_edit"]
    assert "tools/future/autonomy_run.py" in doc["does_not_edit"]
