"""Negative controls for the 30-minute power torture.

A composer nobody has watched refuse is a composer that will silently pad a
workload with copies of the 1h trial. A runner that judges itself is the 1h
trial's failure mode. These tests prove the mix still refuses a missing
transition class, the detector still fires FAIL_NO_WAIT_ORCHESTRATION, a
mutation without rollback still does not count, AND that the frozen run is
judged independently from a sealed timeline against a substrate verified
unchanged, with degeneracy scored on that same timeline.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.future import concurrency_doctor as cd
from tools.future import flash_organ_pivot as fop
from tools.future import orchestration as orch
from tools.future import power_torture as pt
from tools.future import status_causality as sc
from tools.future import trial_workload as tw
from tools.future._common import (
    HARDWARE_FIELDS,
    RECEIPTS,
    HardwareClaimError,
    _assert_no_hardware_claims,
)


@pytest.fixture(scope="module")
def book():
    return tw.load_book()


@pytest.fixture(scope="module")
def proofs():
    return pt.drive_proofs()


@pytest.fixture(scope="module")
def written(proofs, tmp_path_factory):
    scope = tmp_path_factory.mktemp("power_torture_run")
    return pt.run_torture(write=True, scope=scope, proofs=proofs)


def test_wait_while_runnable_timeline_is_fail_no_wait_orchestration():
    """NEGATIVE CONTROL: the detector must fire when the loop waited with work ready."""
    verdict = pt.detect_no_wait_orchestration(pt.synthetic_wait_while_runnable_timeline())
    assert verdict["verdict"] == pt.FAIL_NO_WAIT
    assert verdict["fail"] is True
    assert verdict["failures"]
    assert verdict["gpu_authority"] is False
    held = verdict["failures"][0]
    assert "WU.independent.scar_index" in held["runnable_while_waiting"]
    assert held["independent_launched_during_wait"] == []


def test_timestamped_overlap_is_no_wait_ok_not_a_guess():
    verdict = pt.detect_no_wait_orchestration(pt.synthetic_overlap_timeline())
    assert verdict["verdict"] == pt.NO_WAIT_OK
    assert verdict["fail"] is False
    assert verdict["overlaps"]
    row = verdict["overlaps"][0]
    assert row["detached_unit"] == "WU.detached.specimen_verify"
    assert row["independent_unit"] == "WU.independent.status_challenge"
    assert row["independent_started"] < row["independent_progressed"] < row["independent_completed"]
    a0, a1 = row["detached_open"]
    assert a0 <= row["independent_started"] < row["independent_completed"] <= a1


def test_empty_timeline_is_untested_not_a_pass():
    verdict = pt.detect_no_wait_orchestration({"events": []})
    assert verdict["verdict"] == pt.NO_WAIT_UNTESTED
    assert verdict["fail"] is False
    assert "absence is not proof" in verdict["reason"]


def test_absent_timeline_is_refused_not_defaulted():
    with pytest.raises(pt.TortureRefused, match="timeline is required") as excinfo:
        pt.detect_no_wait_orchestration(None)
    assert "timeline" in excinfo.value.missing


def test_mutation_without_rollback_does_not_count(proofs):
    """NEGATIVE CONTROL: an apply with no proven undo is not the MUTATION class."""
    full = pt.credit_mutation(proofs["mutation"])
    assert full["present"] is True
    stripped = dict(proofs["mutation"])
    stripped.pop("rollback", None)
    denied = pt.credit_mutation(stripped)
    assert denied["present"] is False
    assert "rollback" in denied["why"]
    assert pt.credit_mutation(None)["present"] is False
    assert pt.credit_mutation({"proposed": True})["present"] is False


def test_untested_status_challenge_does_not_count(proofs):
    """NEGATIVE CONTROL: a label with no recorded probe is not a challenge."""
    real = pt.credit_status_challenge(proofs["challenge"])
    assert real["present"] is True
    assert real["verdict"] in {sc.SUPPORTED, sc.OVERREACHING}
    assert real["verdict"] != sc.UNTESTED
    untested = sc.challenge("A_LABEL_THAT_HAS_NO_PROBE_ON_DISK")
    assert untested["verdict"] == sc.UNTESTED
    denied = pt.credit_status_challenge(untested)
    assert denied["present"] is False
    assert "UNTESTED" in denied["why"]
    assert pt.credit_status_challenge(None)["present"] is False


def test_catalog_refill_of_the_same_ids_is_not_a_real_refill(proofs):
    catalog = proofs["catalog_refill"]
    real = proofs["real_refill"]
    assert catalog["present"] is False
    assert catalog["n_fresh"] == 0
    assert real["present"] is True
    assert real["n_fresh"] >= 1
    offered = ["FT.A", "FT.B"]
    denied = pt.credit_refill(offered, ["FT.A", "FT.B"], source="replay")
    assert denied["present"] is False


def test_compose_missing_a_transition_class_is_refused_naming_it(book, proofs):
    """NEGATIVE CONTROL: a mix that drops any required class must not look admitted."""
    planned = pt._plan(book, proofs)
    for missing_name in pt.REQUIRED_TRANSITIONS:
        broken = {
            name: dict(row)
            for name, row in proofs["transitions"].items()
        }
        broken[missing_name] = {"present": False, "why": "stripped for negative control"}
        with pytest.raises(pt.TortureRefused, match="missing required transition") as excinfo:
            pt.admit_torture(planned, broken, book=book)
        assert missing_name in excinfo.value.missing


def test_compose_full_mix_is_admitted_and_bound(book, proofs):
    doc = pt.compose(book=book, proofs=proofs)
    assert doc["admitted"] is True
    assert doc["trial_id"] == pt.TRIAL_ID
    assert doc["duration_s"] == pt.DURATION_S
    assert 6 <= doc["n_units"] <= 12
    assert doc["n_replans"] >= 2
    assert doc["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    launches = {str(u.get("launch") or "") for u in doc["units"]}
    assert "detached" in launches
    transitions = {str(u.get("transition") or u.get("mix_role") or "") for u in doc["units"]}
    assert "NO_WAIT" in transitions
    assert "PROTECTED_PARKING" in transitions
    assert "MUTATION" in transitions
    assert "SCAR_PRUNING" in transitions
    for unit in doc["units"]:
        assert unit["module"] in orch.BINDINGS
        assert orch.BINDINGS[unit["module"]][0] == unit["frontier_id"]
        assert tw._item_by_id(book, unit["frontier_id"]) is not None
        assert unit["gpu_authority"] is False
        assert unit["worth_doing_anyway"]
        assert unit.get("description")
    parked = [u for u in doc["units"] if str(u.get("transition") or "") == "PROTECTED_PARKING"]
    assert parked
    assert parked[0]["status"] == "blocked"
    assert parked[0]["classification"] == "SLEEPING"


def test_detector_and_protected_and_concurrency_proofs_are_executed(proofs):
    """EXECUTED capability: naming the modules is not evidence they ran."""
    assert proofs["wait_verdict"]["verdict"] == pt.FAIL_NO_WAIT
    assert proofs["overlap_verdict"]["verdict"] == pt.NO_WAIT_OK
    assert proofs["challenge"]["verdict"] in {sc.SUPPORTED, sc.OVERREACHING}
    assert proofs["challenge"]["status"] == "BLOCKED_NO_METAL_GPU"
    assert proofs["untested_challenge"]["verdict"] == sc.UNTESTED
    assert proofs["untested_challenge"]["credit"]["present"] is False
    assert proofs["protected"]["parked"] is True
    assert proofs["protected"]["verdict"] == "BLOCKED_ON_PROTECTED_WINDOW"
    assert int(proofs["protected"]["n_continued"] or 0) > 0
    assert proofs["scar"]["dead_decision"] == "REFUSED"
    assert proofs["scar"]["scar_id"]
    assert proofs["scar"]["live_decision"] == "ADMITTED"
    assert proofs["mutation_credit"]["present"] is True
    assert proofs["mutation_without_rollback"]["present"] is False
    assert proofs["concurrency"]["verdict_refused"] is True
    assert proofs["concurrency"]["experiment_state"] == "SLEEPING"
    assert proofs["concurrency"]["decide_verdict"] is None
    assert proofs["subagents"]["present"] is True
    assert proofs["subagents"]["disjoint"] is True
    assert proofs["subagents"]["n_states"] == 2
    # Coherence, not a snapshot. This asserted the generic path was NOT callable,
    # which was true when the lane ran and false eight minutes later when the NX
    # packer landed and Odyssey I launched at 16/16. The invariant is that the
    # proof describes a COHERENT state: callable means no failing stage, and not
    # callable means a NAMED one. Never both, never neither.
    nr_nx = proofs["nr_nx"]
    if nr_nx["callable"]:
        assert not nr_nx["first_failing_stage"], (
            "a callable pipeline must not also name a failing stage"
        )
    else:
        assert nr_nx["first_failing_stage"], (
            "a refusal must name the stage that refused"
        )
    stage = proofs["nr_nx"]["first_failing_stage"] or {}
    assert stage.get("status") != "SKIPPED"
    assert proofs["transitions"]["NO_WAIT"]["present"] is True
    assert proofs["transitions"]["MUTATION"]["present"] is True
    assert proofs["transitions"]["STATUS_CAUSALITY"]["present"] is True
    assert proofs["transitions"]["REPLAN"]["present"] is True
    assert proofs["transitions"]["REPLAN"]["n"] >= 2


def test_restatement_invalidation_fires_and_picks_a_replacement(proofs):
    """EXECUTED capability: queued restatement is refused, next organ is not gate_up."""
    assert proofs["transitions"]["FRONTIER_INVALIDATION"]["present"] is True
    assert proofs["ranking"]["next_school"]
    assert proofs["ranking"]["next_school"] != "ROUTED_EXPERTS"
    cand = {
        "id": "test.restatement",
        "family": "shared_input_latent_plus_expert_local_output_readout",
        "organ": fop.EXHAUSTED_ORGAN,
        "surface": fop.EXHAUSTED_SURFACE,
        "school": "ROUTED_EXPERTS",
    }
    ranking = fop.rank_all()
    with pytest.raises(fop.RestatementRefused):
        fop.refuse_if_restatement(cand, ranking["scar"], ranking["killed_families"])


def test_concurrency_verdict_without_observations_refuses():
    """NEGATIVE CONTROL: the doctor must not default to CONCURRENCY_HELPS."""
    with pytest.raises(cd.VerdictRefuse, match="no observations"):
        cd.verdict([])
    decided = cd.decide()
    assert decided["verdict"] is None
    assert decided["experiment_state"] == "SLEEPING"


def test_nr_nx_callable_on_is_staged_refusal_or_real_path():
    row = pt.callable_on()
    assert row["gpu_authority"] is False
    assert row["pipeline_callable"] is False or row["pipeline_callable"] is True
    if row["pipeline_callable"] is False:
        first = row["first_failing_stage"]
        assert first["stage"]
        assert first["status"] in {"REFUSED", "FAILED", "BLOCKED"}
        assert first["status"] != "SKIPPED"
        assert first["why"]
        for stage in row["stages"]:
            assert stage["status"] != "SKIPPED"
    assert pt.run()["pipeline_callable"] == row["pipeline_callable"]


def test_generic_descriptions_are_refused_as_padding(book):
    with pytest.raises(pt.TortureRefused, match="padding") as excinfo:
        pt._cpu_unit(
            "freshness.py",
            description="do work",
            transition="NO_WAIT",
            why_worth_doing="should never be admitted",
            book=book,
        )
    assert "worth_doing_anyway" in excinfo.value.missing


def test_unbound_module_is_refused(book):
    with pytest.raises(pt.TortureRefused, match="BINDINGS"):
        pt._cpu_unit(
            "not_a_real_module.py",
            description="advance a frontier by running a module that does not exist",
            transition="NO_WAIT",
            why_worth_doing="should never be admitted",
            book=book,
        )


def test_build_emits_sealed_static_only_receipt(written):
    out = RECEIPTS / pt.RECEIPT
    timeline_path = RECEIPTS / pt.TIMELINE_RECEIPT
    assert out.is_file()
    assert timeline_path.is_file()
    doc = json.loads(out.read_text())
    timeline = json.loads(timeline_path.read_text())
    assert out.name == "POWER_TORTURE_30M.json"
    assert timeline_path.name == "POWER_TORTURE_TIMELINE.json"
    assert doc["schema"] == pt.SCHEMA
    assert doc["version"] == 2
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["gpu_authority"] is False
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert doc["seal_sha256"] == hashlib.sha256(blob).hexdigest()
    _assert_no_hardware_claims(doc)
    _assert_no_hardware_claims(timeline)
    for key in HARDWARE_FIELDS:
        if key in doc:
            assert not isinstance(doc[key], (int, float))
    assert doc["credit_rules"]["mutation_without_rollback_counts"] is False
    assert doc["credit_rules"]["untested_status_challenge_counts"] is False
    assert doc["credit_rules"]["correct_refusal_counts_as_exercised"] is True
    assert doc["credit_rules"]["skipped_do_not_count_toward_pass"] is True
    assert doc["credit_rules"]["runner_summary_is_not_the_judge"] is True
    assert written["verdict"] in {"PASS", "FAIL"}
    assert doc["pass_means"] == "NEW_POWER_INTEGRATION"
    assert doc["pass_does_not_mean"] == "resident_cognition"
    assert doc["cognition"]["unavailable"] is True
    assert doc["resident_model_attached"] is False


def test_receipt_rejects_a_hardware_number():
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims({"tps": 1})


def test_module_parses_and_has_no_stubs():
    src = Path(pt.__file__).read_text()
    ast.parse(src)
    for needle in ("raise NotImplementedError", "TODO", "\n    pass\n"):
        assert needle not in src
    assert "pytest.skip(" not in src


def test_substrate_hashed_before_and_after_with_equality(written):
    sub = written["substrate"]
    assert sub["before_digest"]
    assert sub["after_digest"]
    assert sub["before_digest"] == sub["after_digest"]
    assert sub["equal"] is True
    assert sub["verdict"] == pt.SUBSTRATE_CLEAN
    assert sub["moved"] == []
    assert sub["n_files"] >= 1
    doc = json.loads((RECEIPTS / pt.RECEIPT).read_text())
    assert doc["substrate"]["equal"] is True
    assert doc["substrate"]["verdict"] == pt.SUBSTRATE_CLEAN


def test_substrate_mismatch_names_the_file():
    before = pt.hash_substrate()
    tampered = copy.deepcopy(before)
    assert tampered["files"]
    tampered["files"][0]["sha256"] = "0" * 64
    tampered["digest"] = "tampered"
    verdict = pt.verify_substrate(tampered, before)
    assert verdict["equal"] is False
    assert verdict["verdict"] == pt.SUBSTRATE_MOVED
    assert verdict["moved"]
    assert verdict["moved"][0]["path"]
    empty = pt.verify_substrate({"files": []}, before)
    assert empty["verdict"] == pt.SUBSTRATE_MOVED


def test_every_power_is_exercised_or_skipped_and_both_sets_named(written):
    catalog = list(pt.POWER_CATALOG)
    named = set(written["exercised"]) | set(written["skipped"]) | set(written["failed"])
    assert named == set(catalog)
    assert written["n_exercised"] == len(written["exercised"])
    assert written["n_skipped"] == len(written["skipped"])
    for name in catalog:
        row = written["powers"][name]
        assert row["status"] in {pt.EXERCISED, pt.EXERCISED_REFUSAL, pt.SKIPPED, pt.FAILED}
        assert row["why"]
        if row["skipped"]:
            assert row["counts_toward_pass"] is False
        if row["correct_refusal"]:
            assert row["status"] == pt.EXERCISED_REFUSAL
            assert row["exercised"] is True
            assert row["skipped"] is False
            assert name in written["exercised"]
            assert name in written["correct_refusals"]


def test_correct_refusal_is_exercised_not_skipped(written):
    conc = written["powers"]["CONCURRENCY"]
    assert conc["status"] == pt.EXERCISED_REFUSAL
    assert conc["exercised"] is True
    assert conc["skipped"] is False
    assert conc["correct_refusal"] is True
    assert "CONCURRENCY" in written["exercised"]
    assert "CONCURRENCY" in written["correct_refusals"]
    assert conc["evidence"]["experiment_state"] == "SLEEPING"
    assert conc["evidence"]["blocked_reason"]
    nr = written["powers"]["GENERIC_NR_NX"]
    assert nr["status"] in {pt.EXERCISED, pt.EXERCISED_REFUSAL}
    assert nr["skipped"] is False
    if nr["correct_refusal"]:
        assert nr["evidence"]["first_failing_stage"]["status"] != "SKIPPED"


def test_timeline_is_append_only_sealed_and_hashed(written):
    path = RECEIPTS / pt.TIMELINE_RECEIPT
    doc = json.loads(path.read_text())
    assert doc["append_only"] is True
    assert doc["sealed"] is True
    assert doc["events_sha256"]
    events = doc["events"]
    assert events
    assert [e["seq"] for e in events] == list(range(len(events)))
    t_s = [float(e["t_s"]) for e in events]
    assert all(t_s[i] <= t_s[i + 1] + 1e-9 for i in range(len(t_s) - 1))
    blob = json.dumps(events, sort_keys=True, separators=(",", ":")).encode()
    assert doc["events_sha256"] == hashlib.sha256(blob).hexdigest()
    tl = pt.SealedTimeline()
    tl.append("WORK_LAUNCHED", {"unit": {"id": "WU.TORTURE.probe"}, "power": "NO_WAIT"})
    tl.seal()
    with pytest.raises(pt.TortureRefused, match="sealed"):
        tl.append("WORK_LAUNCHED", {"unit": {"id": "WU.TORTURE.late"}})


def test_judge_reads_timeline_not_runner_summary(written):
    timeline = json.loads((RECEIPTS / pt.TIMELINE_RECEIPT).read_text())
    bare = {
        "events": timeline["events"],
        "events_sha256": timeline["events_sha256"],
        "sealed": True,
        "append_only": True,
        "runner_summary": {"verdict": "FORGED_PASS", "exercised": []},
        "powers": {"NO_WAIT": {"status": "SKIPPED"}},
        "proofs": {"lie": True},
    }
    judged = pt.judge(bare)
    assert judged["read"] == "timeline.events"
    assert "runner_summary" in judged["ignored"]
    assert judged["n_omitted"] == 0
    assert set(judged["exercised"]) | set(judged["skipped"]) | set(judged["failed"]) == set(
        pt.POWER_CATALOG
    )
    assert judged["n_exercised"] == written["n_exercised"]
    assert set(judged["exercised"]) == set(written["exercised"])
    assert set(judged["skipped"]) == set(written["skipped"])
    assert judged["correct_refusal_counts_as_exercised"] is True
    assert judged["skipped_do_not_count_toward_pass"] is True
    assert "CONCURRENCY" in judged["exercised"]
    assert "CONCURRENCY" in judged["correct_refusals"]
    broken = dict(bare)
    broken["events_sha256"] = "0" * 64
    failed = pt.judge(broken)
    assert failed["verdict"] == "FAIL"
    assert failed["seal_ok"] is False


def test_degeneracy_measure_ran_on_this_timeline(written):
    deg = written["degeneracy"]
    assert deg["instrument"] == "tools.future.autonomy_degeneracy.measure"
    assert deg["axis_table"]
    axes = {row["axis"] for row in deg["axis_table"]}
    for name in ("rejections", "refills", "ingestion", "launches", "workunit_ids", "decisions", "scars"):
        assert name in axes
    for row in deg["axis_table"]:
        assert "unique" in row
        assert "total" in row
        assert "largest_repeat_run" in row
        assert "consecutive_emissions_identical" in row
        assert "degenerate" in row
        assert "reason" in row
    doc = json.loads((RECEIPTS / pt.RECEIPT).read_text())
    assert doc["degeneracy"]["axis_table"]
    assert doc["degeneracy"]["verdict"] in {"PASS", "FAIL"}
    assert deg["n_argv0_labelled"] == 0
    timeline = written["_timeline_doc"]
    measured = pt.measure_degeneracy(timeline)
    assert measured["verdict"] == deg["verdict"]


def test_gpu_lane_lock_was_inspected_not_contended(written):
    lock = written["gpu_lane_lock"]
    assert lock["path"]
    assert lock["contended"] is False
    if lock["present"]:
        assert lock["parked"] is True
        assert lock["waited_for"]


def test_cognition_is_unavailable_and_not_stubbed(written):
    cog = written["cognition"]
    assert cog["unavailable"] is True
    assert cog["state"] == "UNAVAILABLE"
    assert cog.get("asked") in {False, None}


def test_wall_clock_within_thirty_minutes(written):
    assert written["elapsed_s"] < pt.DURATION_S
    assert written["within_budget"] is True
