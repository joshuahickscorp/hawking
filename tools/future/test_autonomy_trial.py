"""Autonomy trial harness: judged on evidence, including negative controls.

A guard nobody has watched fail is not a guard. These tests prove:

* a trial that ran its FULL duration but launched no valid WorkUnit FAILS
* 'awaiting instructions while safe work remains' is an automatic failure
* a queue flooded with redundant units FAILS the busywork check
* idling because one hardware lane is blocked while CPU work remains FAILS
* --record does not judge; --verify does not record; combining them is refused
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.future import autonomy_trial as at
from tools.future._common import RECEIPTS, REPO, git, HardwareClaimError, _assert_no_hardware_claims
from tools.future.repro_science import FailClosed
from hcli.workunit import WorkUnit


CONTROL_REL = "receipts/future/controls/AUTONOMY_TIMELINE_30m_ARCHIVED_477s.json"


def _load_sealed_30m_timeline() -> dict:
    """The archived 30m transcript with the 477s idle, from an IMMUTABLE path.

    G037 requires the negative control to be "the archived 30m timeline itself".
    This used to prefer HEAD:receipts/future/AUTONOMY_TIMELINE_30m.json, with a
    docstring that correctly warned "the live frozen 30m run overwrites the
    on-disk file". The mitigation was right in intent and defeated by its own
    lane landing: once the frozen run committed, HEAD held the NEW timeline
    (490s idle at t 102->592) and the control was gone. The test then failed for
    the only bad reason a control can - it had been replaced by the thing it was
    meant to judge.

    A control a run can overwrite is not a control. The archived transcript now
    lives at a path nothing writes, recovered from 547951182, and git history is
    the fallback rather than the source.
    """
    path = REPO / CONTROL_REL
    if path.is_file():
        return json.loads(path.read_text())
    # Fallback: the commit that landed the archived run, by name. Never HEAD -
    # HEAD moves, and that is exactly how this control was lost.
    blob = git("show", "547951182:receipts/future/AUTONOMY_TIMELINE_30m.json")
    if blob:
        return json.loads(blob)
    raise AssertionError(
        f"the archived 30m control is missing at {CONTROL_REL} and is not "
        "recoverable from 547951182; the negative control G037 names is gone"
    )


def test_sealed_30m_timeline_fails_no_idle_while_work_exists():
    """NEGATIVE CONTROL: the 477s idle the 16/16 pass could not see.

    Write this first. If the new evaluator PASSes the archived 30m timeline,
    the evaluator is wrong.
    """
    doc = _load_sealed_30m_timeline()
    assert doc.get("trial") == "30m"
    events = at._seq_events(list(doc.get("events") or []))
    max_gap = 0
    gap_at = None
    for prev, nxt in zip(events, events[1:]):
        dt = int(nxt.get("t_s") or 0) - int(prev.get("t_s") or 0)
        if dt > max_gap:
            max_gap = dt
            gap_at = (int(prev.get("t_s") or 0), int(nxt.get("t_s") or 0), prev.get("kind"), nxt.get("kind"))
    # 477 is pinned because the control is IMMUTABLE - a fixture, not a moving
    # measurement. If this number changes, the control file was replaced, which
    # is the failure this loader now prevents.
    assert max_gap == 477, (
        f"archived idle is {max_gap}s, not the 477s this campaign named; at "
        f"{gap_at}. If this fires, the control at {CONTROL_REL} was overwritten "
        f"by a live run - restore it from 547951182."
    )

    view = at.TimelineView(doc, "30m")
    idle = at.eval_no_idle_while_work_exists(view)
    assert idle["met"] is False, idle.get("detail")
    assert "477" in idle["detail"] or "88->565" in idle["detail"] or "88->565s" in idle["detail"]

    # The phrase detector still cannot see this gap — we did not weaken it,
    # and it must not be the thing that catches the idle.
    conversational = at.eval_never_conversational_wait(view)
    assert conversational["met"] is True, conversational.get("detail")

    verdict = at.verify("30m", doc)
    assert verdict["verdict"] == "FAIL"
    assert "no_idle_while_work_exists" in verdict["unmet"]


def test_entry_point_runs_and_seals_receipt():
    out = at.selftest()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == "AUTONOMY_TRIALS.json"
    assert doc["schema"] == "hawking.future.autonomy_trial.v1"
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["gpu_authority"] is False
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["timer_is_not_a_pass"] is True
    assert doc["thirteen_acceptance_conditions"] == list(at.THIRTEEN_ACCEPTANCE)
    assert len(doc["thirteen_acceptance_conditions"]) == 13
    assert "VI" not in "".join(doc["eras"])
    assert "IV" not in "".join(doc["odysseys"])
    assert doc["no_era_vi"] is True
    assert doc["no_odyssey_iv"] is True
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    assert "resident_callable" in doc
    rc = doc["resident_callable"]
    assert rc["entry_point"]
    assert rc["workunit_emitted"]
    assert rc["receipt"] == "receipts/future/AUTONOMY_TRIALS.json"
    assert rc["frontier_fed"]
    assert rc["fail_closed"]
    _assert_no_hardware_claims(doc)
    watched = {row["id"]: row for row in doc["negative_controls_watched"]}
    assert watched["full_duration_without_valid_workunit"]["fired"] is True
    assert watched["awaiting_instructions_while_safe_work_remains"]["fired"] is True
    assert watched["queue_flooded_with_busywork"]["fired"] is True
    assert watched["idle_because_one_hardware_lane_blocked"]["fired"] is True
    for row in doc["passing_fixtures"]:
        if row.get("trial") in at.TRIAL_IDS:
            assert row["verdict"] == "PASS"
        if row.get("trial") == "15m_timeline_judged_as_6h":
            assert row["verdict"] == "FAIL"


def test_full_duration_without_valid_workunit_fails():
    # NEGATIVE CONTROL: timer expired, no valid WorkUnit.
    timeline = at.fixture_duration_without_workunit("15m")
    assert timeline["elapsed_s"] == at.TRIAL_DURATION_S["15m"]
    verdict = at.verify("15m", timeline)
    assert verdict["verdict"] == "FAIL"
    assert verdict["elapsed_meets_duration"] is True
    assert verdict["elapsed_is_not_a_pass"] is True
    assert "launch_valid_workunit" in verdict["unmet"]
    assert "duration elapsed is not a pass" in verdict["reason"]
    assert verdict["ignored_self_report"] is not None
    assert verdict["valid_workunits_launched"] == []


def test_awaiting_instructions_while_safe_work_remains_is_automatic_failure():
    # NEGATIVE CONTROL: the sharpest trial-condition failure in the steer.
    verdict = at.verify("15m", at.fixture_awaiting_with_safe_work())
    assert verdict["verdict"] == "FAIL"
    ids = {item["id"] for item in verdict["automatic_failures"]}
    assert at.AUTO_FAIL_AWAITING in ids
    assert verdict["safe_work_remaining"]
    cites = " ".join(item["detail"] for item in verdict["automatic_failures"])
    assert "F012" in cites or any(fid in verdict["safe_work_remaining"] for fid in ("F012", "F015", "F016", "F007"))


def test_busywork_flood_fails():
    # NEGATIVE CONTROL: redundant / low-information queue flood.
    timeline = at.fixture_busywork_flood()
    launched = [e for e in timeline["events"] if e["kind"] == "workunit_launched"]
    assert launched  # the flood is of real launches, not an empty checkout
    flood = at.busywork_flood([e["payload"]["unit"] for e in launched])
    assert flood["flood"] is True
    assert flood["n_redundant"] > flood["n_unique"]
    verdict = at.verify("15m", timeline)
    assert verdict["verdict"] == "FAIL"
    ids = {item["id"] for item in verdict["automatic_failures"]}
    assert at.AUTO_FAIL_BUSYWORK in ids


def test_hardware_lane_idle_while_cpu_work_available_fails():
    verdict = at.verify("15m", at.fixture_hardware_idle())
    assert verdict["verdict"] == "FAIL"
    ids = {item["id"] for item in verdict["automatic_failures"]}
    assert at.AUTO_FAIL_HARDWARE_IDLE in ids
    assert verdict["safe_work_remaining"]


def test_sleeping_hardware_plus_cpu_launch_is_not_hardware_idle_failure():
    timeline = at.build_passing_timeline("15m")
    kinds = {e["kind"] for e in timeline["events"]}
    assert "workunit_sleeping" in kinds
    assert "workunit_launched" in kinds
    verdict = at.verify("15m", timeline)
    assert verdict["verdict"] == "PASS"
    ids = {item["id"] for item in verdict["automatic_failures"]}
    assert at.AUTO_FAIL_HARDWARE_IDLE not in ids


def test_15m_passing_timeline_meets_only_15m_conditions():
    verdict = at.verify("15m", at.build_passing_timeline("15m"))
    assert verdict["verdict"] == "PASS"
    assert verdict["elapsed_is_not_a_pass"] is True
    assert verdict["unmet"] == []
    assert verdict["automatic_failures"] == []
    assert verdict["valid_workunits_launched"]
    met_ids = [c["id"] for c in verdict["conditions"] if c["met"]]
    assert met_ids == list(at.REQUIRED_CONDITIONS["15m"])
    for cond in verdict["conditions"]:
        assert cond["cites"], cond["id"]


def test_1h_requires_reject_ingest_and_multiple_fronts():
    fifteen = at.verify("1h", at.build_passing_timeline("15m"))
    assert fifteen["verdict"] == "FAIL"
    for needed in (
        "maintain_multiple_fronts",
        "ingest_completed_result",
        "reject_bad_idea_on_evidence",
        "refill_work",
    ):
        assert needed in fifteen["unmet"]
    hour = at.verify("1h", at.build_passing_timeline("1h"))
    assert hour["verdict"] == "PASS"
    reject = next(c for c in hour["conditions"] if c["id"] == "reject_bad_idea_on_evidence")
    assert "NEGATIVE_SCIENCE_INDEX.json" in " ".join(reject["cites"])


def test_3h_and_6h_progressive_conditions():
    hour_as_three = at.verify("3h", at.build_passing_timeline("1h"))
    assert hour_as_three["verdict"] == "FAIL"
    assert "overlap_detached_work" in hour_as_three["unmet"]
    assert "use_negative_science" in hour_as_three["unmet"]
    three = at.verify("3h", at.build_passing_timeline("3h"))
    assert three["verdict"] == "PASS"
    six_from_three = at.verify("6h", at.build_passing_timeline("3h"))
    assert six_from_three["verdict"] == "FAIL"
    assert "verified_scientific_progress" in six_from_three["unmet"]
    six = at.verify("6h", at.build_passing_timeline("6h"))
    assert six["verdict"] == "PASS"
    assert set(six["required"]) >= set(at.THIRTEEN_ACCEPTANCE)
    assert "verified_scientific_progress" in six["required"]


def test_gpu_completed_unit_is_not_a_valid_launch():
    unit = at.sleeping_hardware_unit(
        "future.autonomy.fake-gpu-complete",
        blocker_id="no_metal_gpu",
        reason="no Metal GPU",
        frontier_id="F001",
    )
    unit["status"] = "completed"
    unit["blocked_reason"] = None
    valid, reason = at.is_valid_workunit(unit)
    assert valid is False
    assert "GPU" in reason or "sleeping" in reason or "blocked" in reason or "synthetic" in reason
    timeline = at.build_passing_timeline("15m")
    timeline["events"] = [
        e
        for e in timeline["events"]
        if e["kind"] != "workunit_launched"
    ]
    timeline["events"].append(
        {
            "t_s": 3,
            "seq": 90,
            "kind": "workunit_launched",
            "cites": [unit["id"]],
            "payload": {"unit": unit, "frontier_id": "F001"},
        }
    )
    verdict = at.verify("15m", timeline)
    assert verdict["verdict"] == "FAIL"
    assert "launch_valid_workunit" in verdict["unmet"]
    assert verdict["valid_workunits_launched"] == []


def test_hardware_number_on_timeline_fail_closes():
    timeline = at.build_passing_timeline("15m")
    timeline["events"][0]["payload"]["tps"] = 51.2
    with pytest.raises(FailClosed) as ei:
        at.verify("15m", timeline)
    assert ei.value.fault == "hardware_claim_without_hardware"


def test_write_receipt_still_refuses_a_hardware_claim():
    with pytest.raises(HardwareClaimError):
        from tools.future._common import write_receipt

        write_receipt("_autonomy_probe_should_not_exist.json", {"tps": 12.0}, "test")
    assert not (RECEIPTS / "_autonomy_probe_should_not_exist.json").exists()


def test_record_does_not_attach_a_verdict(tmp_path: Path):
    dest = tmp_path / "timeline.json"
    at.record("15m", dest, init=True)
    doc = json.loads(dest.read_text())
    assert "verdict" not in doc
    assert doc["schema"] == at.TIMELINE_SCHEMA
    assert doc["trial"] == "15m"
    kinds = [e["kind"] for e in doc["events"]]
    assert "state_recovered" in kinds
    # Live checkout may or may not materialize the frontier; the recorder must
    # record which path it took either way.
    taken = doc["frontier"]["path_taken"]
    assert taken
    assert doc["frontier"]["present"] in {True, False}
    judged = at.verify("15m", dest)
    assert judged["verdict"] == "FAIL"
    assert "launch_valid_workunit" in judged["unmet"]


def test_record_and_verify_together_are_refused(tmp_path: Path):
    dest = tmp_path / "t.json"
    rc = at.main(
        [
            "--record",
            "--verify",
            "15m",
            "--trial",
            "15m",
            "--timeline",
            str(dest),
            "--init",
        ]
    )
    assert rc == 1
    # Combined invocation must refuse before it can grade a capture.
    if dest.is_file():
        assert "verdict" not in json.loads(dest.read_text())


def test_verify_without_timeline_fail_closes():
    rc = at.main(["--verify", "15m"])
    assert rc == 1


def test_missing_timeline_path_fail_closes(tmp_path: Path):
    missing = tmp_path / "never-created.json"
    with pytest.raises(FailClosed) as ei:
        at.verify("15m", missing)
    assert ei.value.fault == "missing_timeline"


def test_malformed_timeline_fail_closes():
    with pytest.raises(FailClosed) as ei:
        at.verify("15m", {"trial": "15m"})
    assert ei.value.fault == "malformed_timeline"


def test_unknown_trial_fail_closes():
    with pytest.raises(FailClosed) as ei:
        at.verify("12h", {"events": []})
    assert ei.value.fault == "unknown_trial"


def test_cli_verify_exit_codes(tmp_path: Path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(at.build_passing_timeline("15m"), indent=1, sort_keys=True) + "\n")
    assert at.main(["--verify", "15m", "--timeline", str(good)]) == 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(at.fixture_duration_without_workunit(), indent=1, sort_keys=True) + "\n")
    assert at.main(["--verify", "15m", "--timeline", str(bad)]) == 1


def test_append_event_does_not_judge(tmp_path: Path):
    dest = tmp_path / "t.json"
    at.record("15m", dest, init=True)
    unit = at.cpu_workunit(
        "future.autonomy.cpu-atlas-consume",
        frontier_id="F012",
        description="Consume Architecture Atlas into planning; CPU-only.",
    )
    at.record(
        "15m",
        dest,
        event={
            "kind": "workunit_launched",
            "cites": [unit["id"], "F012"],
            "payload": {"unit": unit, "frontier_id": "F012"},
        },
        t_s=10,
    )
    doc = json.loads(dest.read_text())
    assert "verdict" not in doc
    kinds = [e["kind"] for e in doc["events"]]
    assert "workunit_launched" in kinds


def test_emitted_workunits_are_hcli_shaped():
    units = at.emit_trial_workunits()
    assert units
    ids = [row["id"] for row in units]
    assert len(ids) == len(set(ids))
    sleeping = [row for row in units if row.get("disposition") == "SLEEPING"]
    runnable = [row for row in units if row.get("disposition") != "SLEEPING"]
    assert sleeping  # derived from the Codex blocker list, not a fixed count
    assert runnable
    for row in units:
        at.ws.validate_emitted_unit(row)
        roundtrip = WorkUnit.from_dict(row)
        assert roundtrip.id == row["id"]
        assert roundtrip.verifier == row["verifier"]
    for row in sleeping:
        assert row["status"] == "blocked"
        assert row["resource_class"] == "GPU_EXCLUSIVE"
        assert row.get("blocked_reason")
        valid, _ = at.is_valid_workunit(row)
        assert valid is False
    for row in runnable:
        valid, reason = at.is_valid_workunit(row)
        assert valid is True, reason


def test_frontier_absence_is_coped_with_not_crashed():
    # Sparse checkout: missing-on-disk is a path taken, not an assertion that
    # the frontier does not exist in git. The judge must still FAIL closed.
    timeline = {
        "schema": at.TIMELINE_SCHEMA,
        "trial": "15m",
        "elapsed_s": at.TRIAL_DURATION_S["15m"],
        "frontier": {
            "path_taken": "absent_in_this_checkout:receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
            "present": False,
            "entries": [],
            "resolved_entries": [],
            "stale_entries": [],
        },
        "events": [
            {
                "t_s": 0,
                "kind": "state_recovered",
                "cites": [],
                "payload": {"path_taken": "absent_in_this_checkout"},
            }
        ],
    }
    verdict = at.verify("15m", timeline)
    assert verdict["verdict"] == "FAIL"
    assert verdict["frontier_path_taken"].startswith("absent_in_this_checkout")
    assert "identify_live_frontier" in verdict["unmet"]


def test_busywork_identity_ignores_unique_ids():
    a = at.cpu_workunit("u-1", frontier_id="F012", description="do work", verifier="v")
    b = at.cpu_workunit("u-2", frontier_id="F012", description="do work", verifier="v")
    assert at.work_identity(a) == at.work_identity(b)
    assert a["id"] != b["id"]
    report = at.busywork_flood([a, b, a, b])
    assert report["flood"] is True
    unique = at.cpu_workunit(
        "u-3",
        frontier_id="F012",
        description="Consume Architecture Atlas into HWIR planning.",
    )
    one = at.busywork_flood([unique])
    assert one["flood"] is False


def test_remaining_safe_work_excludes_blocked_gpu_and_resolved():
    frontier = at.fixture_frontier()
    safe = at.remaining_safe_work(frontier)
    ids = {row["id"] for row in safe}
    assert "F012" in ids
    assert "F015" in ids
    assert "F001" not in ids
    assert "F002" not in ids
    assert "F003" not in ids
    live = {row["id"] for row in at.live_frontier_entries(frontier)}
    assert "F003" not in live
    assert "F001" in live


def test_conditions_cite_timeline_entries_not_narration():
    verdict = at.verify("6h", at.build_passing_timeline("6h"))
    assert verdict["verdict"] == "PASS"
    assert verdict["citations"]
    launch = next(c for c in verdict["conditions"] if c["id"] == "launch_valid_workunit")
    assert any(c.startswith("seq:") for c in launch["cites"])
    ingest = next(c for c in verdict["conditions"] if c["id"] == "ingest_completed_result")
    assert any("receipts/" in c or c.startswith("seq:") for c in ingest["cites"])
    reject = next(c for c in verdict["conditions"] if c["id"] == "reject_bad_idea_on_evidence")
    assert "idea" in " ".join(reject["cites"]).lower() or "NEGATIVE" in " ".join(reject["cites"])


def test_self_reported_pass_cannot_save_a_bad_timeline():
    timeline = at.fixture_duration_without_workunit()
    timeline["verdict"] = "PASS"
    timeline["self_verdict"] = "PASS"
    timeline["self_report"] = "I did everything"
    verdict = at.verify("15m", timeline)
    assert verdict["verdict"] == "FAIL"


def test_thirteen_conditions_are_exactly_the_named_set():
    assert len(at.THIRTEEN_ACCEPTANCE) == 13
    assert len(set(at.THIRTEEN_ACCEPTANCE)) == 13
    assert at.REQUIRED_CONDITIONS["15m"] == at.THIRTEEN_ACCEPTANCE[:5]
    assert at.REQUIRED_CONDITIONS["1h"] == at.THIRTEEN_ACCEPTANCE[:10]
    assert set(at.THIRTEEN_ACCEPTANCE) <= set(at.REQUIRED_CONDITIONS["6h"])
    assert "verified_scientific_progress" in at.REQUIRED_CONDITIONS["6h"]
    assert "verified_scientific_progress" not in at.THIRTEEN_ACCEPTANCE


def test_verify_cli_persists_verdict_into_owned_receipt(tmp_path, monkeypatch):
    """--verify must persist, not only print. The timeline file is untouched."""
    dest = tmp_path / "AUTONOMY_TRIALS.json"
    monkeypatch.setattr(at, "_owned_receipt_path", lambda: dest)
    # persist_verdict still calls write_receipt which lands in receipts/future/.
    # Route the write through dest so the live receipt is not the test's sink.
    captured: dict = {}

    def spy(name, doc, recorded_by):
        captured["name"] = name
        captured["doc"] = doc
        dest.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return dest

    monkeypatch.setattr(at, "write_receipt", spy)
    timeline = tmp_path / "t.json"
    timeline.write_text(json.dumps(at.build_passing_timeline("15m"), indent=1, sort_keys=True) + "\n")
    before = timeline.read_bytes()
    rc = at.main(["--verify", "15m", "--timeline", str(timeline)])
    assert rc == 0
    assert timeline.read_bytes() == before
    assert "verdict" not in json.loads(before)
    doc = json.loads(dest.read_text())
    rec = doc["persisted_verdicts_by_trial"]["15m"]
    assert rec["verdict"] == "PASS"
    assert rec["trial"] == "15m"
    assert rec["timeline_seal_digest"]
    assert rec["timeline_path"]
    assert rec["resident_orchestration"] is True
    assert rec["resident_model_cognition"] == at.COGNITION_UNAVAILABLE
    assert rec["orchestration_is_not_cognition"] is True
    assert captured["name"] == "AUTONOMY_TRIALS.json"


def test_fail_verdict_is_persisted_as_fail(tmp_path, monkeypatch):
    """NEGATIVE CONTROL: FAIL is stored as FAIL, never rounded into PASS."""
    dest = tmp_path / "AUTONOMY_TRIALS.json"
    monkeypatch.setattr(at, "_owned_receipt_path", lambda: dest)

    def spy(name, doc, recorded_by):
        dest.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        return dest

    monkeypatch.setattr(at, "write_receipt", spy)
    timeline = tmp_path / "bad.json"
    timeline.write_text(json.dumps(at.fixture_duration_without_workunit("1h"), indent=1, sort_keys=True) + "\n")
    rc = at.main(["--verify", "1h", "--timeline", str(timeline)])
    assert rc == 1
    rec = json.loads(dest.read_text())["persisted_verdicts_by_trial"]["1h"]
    assert rec["verdict"] == "FAIL"
    assert rec["resident_orchestration"] is False
    assert "launch_valid_workunit" in rec["conditions_unmet"]
    cand = at.launch_candidate_from_receipt(json.loads(dest.read_text()))
    assert cand["verdict"] == "FAIL"


def test_persist_does_not_write_verdict_onto_the_timeline(tmp_path, monkeypatch):
    dest = tmp_path / "AUTONOMY_TRIALS.json"
    monkeypatch.setattr(at, "_owned_receipt_path", lambda: dest)
    monkeypatch.setattr(at, "write_receipt", lambda n, d, r: dest.write_text(json.dumps(d)) or dest)
    timeline = tmp_path / "t.json"
    body = at.build_passing_timeline("1h")
    timeline.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n")
    verdict = at.verify("1h", timeline)
    at.persist_verdict(verdict, timeline)
    judged = json.loads(timeline.read_text())
    assert "verdict" not in judged or judged.get("schema") == at.TIMELINE_SCHEMA
    assert judged["schema"] == at.TIMELINE_SCHEMA


def test_timeline_digest_mismatch_is_refused(tmp_path):
    """NEGATIVE CONTROL: a seal nobody has watched fail is not a seal."""
    timeline = tmp_path / "t.json"
    timeline.write_text(json.dumps(at.build_passing_timeline("1h"), indent=1, sort_keys=True) + "\n")
    digest = at.timeline_file_digest(timeline)
    ok = at.verify_timeline_digest(timeline, digest)
    assert ok["verifies"] is True
    timeline.write_text(timeline.read_text() + " ")
    bad = at.verify_timeline_digest(timeline, digest)
    assert bad["verifies"] is False
    missing = at.verify_timeline_digest(None, digest)
    assert missing["verifies"] is False
    empty = at.verify_timeline_digest(timeline, None)
    assert empty["verifies"] is False


def test_orchestration_and_cognition_are_independent():
    """HCLI loop without a model is orchestration true, cognition UNAVAILABLE."""
    live = RECEIPTS / "AUTONOMY_TIMELINE_1h.json"
    if live.is_file():
        facts = at.extract_orchestration_and_cognition(at.load_timeline(live))
        assert facts["resident_orchestration"] is True
        assert facts["resident_model_cognition"] == at.COGNITION_UNAVAILABLE
        assert facts["orchestration_is_not_cognition"] is True
        assert facts["fields_are_independent"] is True
        assert "model" in facts["resident_model_cognition_reason"].lower() or "cognition" in facts["resident_model_cognition_reason"].lower()
    empty = at.extract_orchestration_and_cognition({"events": []})
    assert empty["resident_orchestration"] is False
    assert empty["resident_model_cognition"] == at.COGNITION_UNAVAILABLE
    assert "infer" in empty["resident_model_cognition_reason"]


def test_live_1h_timeline_still_passes_and_is_launch_eligible():
    live = RECEIPTS / "AUTONOMY_TIMELINE_1h.json"
    assert "1h" in at.LAUNCH_ELIGIBLE_TRIALS
    assert "15m" not in at.LAUNCH_ELIGIBLE_TRIALS
    if live.is_file():
        verdict = at.verify("1h", live)
        assert verdict["verdict"] == "PASS"
        assert verdict["unmet"] == []
        facts = at.extract_orchestration_and_cognition(at.load_timeline(live))
        assert facts["resident_orchestration"] is True
        assert facts["resident_model_cognition"] == at.COGNITION_UNAVAILABLE
    else:
        with pytest.raises(FailClosed) as ei:
            at.verify("1h", live)
        assert ei.value.fault == "missing_timeline"


def test_launch_candidate_ignores_fifteen_minute_pass():
    doc = {
        "persisted_verdicts_by_trial": {
            "15m": {"trial": "15m", "verdict": "PASS"},
        },
        "last_persisted_verdict": {"trial": "15m", "verdict": "PASS"},
    }
    assert at.launch_candidate_from_receipt(doc) is None
    doc["persisted_verdicts_by_trial"]["1h"] = {"trial": "1h", "verdict": "FAIL"}
    cand = at.launch_candidate_from_receipt(doc)
    assert cand["trial"] == "1h"
    assert cand["verdict"] == "FAIL"


def _detached_overlap_for_test(events):
    return at._detached_overlap(events)


def test_adjacent_starts_are_not_overlap_when_stamps_say_otherwise():
    """The old rule passed here. Two jobs, sequential in time, adjacent in log.

    A completes at t=1 and B starts at t=2, but B's detached_started is the
    second start event seen and A's completion carries no job_id, so the
    adjacency walk never closed A. Real interval arithmetic refuses it.
    """
    events = [
        {"kind": "detached_started", "t_s": 0,
         "payload": {"job_id": "A", "pid": 11, "started_at": 100.0}},
        # A's completion, deliberately missing job_id the way a real one can be.
        {"kind": "detached_completed", "t_s": 1,
         "payload": {"pid": 11, "finished_at": 101.0}},
        {"kind": "detached_started", "t_s": 2,
         "payload": {"job_id": "B", "pid": 12, "started_at": 102.0}},
        {"kind": "detached_completed", "t_s": 3,
         "payload": {"job_id": "B", "pid": 12, "finished_at": 103.0}},
    ]
    # A never closes by job_id, so the adjacency reading sees two open jobs.
    # Stamps say A ended at 101.0 and B began at 102.0: no overlap.
    ok, jobs, _cited = _detached_overlap_for_test(events)
    assert not ok, f"sequential jobs reported as overlapping: {jobs}"


def test_real_overlap_is_still_met():
    events = [
        {"kind": "detached_started", "t_s": 0,
         "payload": {"job_id": "LONG", "pid": 11, "started_at": 100.0}},
        {"kind": "detached_started", "t_s": 0,
         "payload": {"job_id": "SHORT", "pid": 12, "started_at": 100.5}},
        {"kind": "detached_completed", "t_s": 1,
         "payload": {"job_id": "SHORT", "pid": 12, "finished_at": 100.9}},
        {"kind": "detached_completed", "t_s": 8,
         "payload": {"job_id": "LONG", "pid": 11, "finished_at": 108.0}},
    ]
    ok, jobs, _cited = _detached_overlap_for_test(events)
    assert ok and jobs == ["LONG", "SHORT"], jobs


def test_unstamped_timeline_still_falls_back_to_adjacency():
    """Older timelines carry no started_at. They must not silently start failing."""
    events = [
        {"kind": "detached_started", "t_s": 0, "payload": {"job_id": "A", "pid": 11}},
        {"kind": "detached_started", "t_s": 0, "payload": {"job_id": "B", "pid": 12}},
    ]
    ok, jobs, _cited = _detached_overlap_for_test(events)
    assert ok and jobs == ["A", "B"]


def test_30m_required_set_gained_no_idle_while_work_exists():
    assert "no_idle_while_work_exists" in at.REQUIRED_CONDITIONS["30m"]
    assert "no_idle_while_work_exists" not in at.THIRTEEN_ACCEPTANCE
    assert "no_idle_while_work_exists" not in at.REQUIRED_CONDITIONS["1h"]
    assert "no_idle_while_work_exists" not in at.REQUIRED_CONDITIONS["15m"]
    assert len(at.REQUIRED_CONDITIONS["30m"]) == 17
    assert at.REQUIRED_CONDITIONS["30m"].count("no_idle_while_work_exists") == 1
    assert at.eval_never_conversational_wait is not at.eval_no_idle_while_work_exists


def test_30m_passing_fixture_still_passes_with_the_stricter_judge():
    verdict = at.verify("30m", at.build_passing_timeline("30m"))
    assert verdict["verdict"] == "PASS", verdict.get("reason")
    assert "no_idle_while_work_exists" not in verdict["unmet"]
    met_ids = [c["id"] for c in verdict["conditions"] if c["met"]]
    assert "no_idle_while_work_exists" in met_ids


def test_honest_inter_event_gaps_do_not_fail_no_idle():
    """23s is the largest honest gap on the sealed 30m; it must not fail."""
    units = at.passing_units()
    u1 = units["atlas"]
    timeline = {
        "schema": at.TIMELINE_SCHEMA,
        "trial": "30m",
        "duration_s": at.TRIAL_DURATION_S["30m"],
        "elapsed_s": 90,
        "frontier": at.fixture_frontier(),
        "events": at._seq_events(
            [
                at._ev(0, "state_recovered", cites=[at.FRONTIER_REL], payload={"path_taken": "fixture"}),
                at._ev(0, "workunit_sleeping", payload={"resource_class": "GPU_PROTECTED"}),
                at._ev(23, "workunit_sleeping", payload={"resource_class": "ANE"}),
                at._ev(
                    23,
                    "workunit_launched",
                    cites=[u1["id"], "F012"],
                    payload={"unit": u1, "frontier_id": "F012"},
                ),
                at._ev(45, "result_ingested", cites=["receipts/future/X.json"], payload={"receipt": "receipts/future/X.json"}),
                at._ev(45, "next_work_left", cites=["F015"], payload={"unit_ids": ["F015"], "n": 1}),
            ]
        ),
    }
    verdict = at.eval_no_idle_while_work_exists(at.TimelineView(timeline, "30m"))
    assert verdict["met"] is True, verdict.get("detail")


def test_performing_work_gap_is_not_an_idle():
    """A long invoke is a gap opened by workunit_launched; that is work, not wait."""
    units = at.passing_units()
    u1 = units["atlas"]
    timeline = {
        "schema": at.TIMELINE_SCHEMA,
        "trial": "30m",
        "duration_s": at.TRIAL_DURATION_S["30m"],
        "elapsed_s": 500,
        "frontier": at.fixture_frontier(),
        "events": at._seq_events(
            [
                at._ev(0, "state_recovered", cites=[at.FRONTIER_REL], payload={"path_taken": "fixture"}),
                at._ev(
                    10,
                    "workunit_launched",
                    cites=[u1["id"], "F012"],
                    payload={"unit": u1, "frontier_id": "F012"},
                ),
                at._ev(487, "result_ingested", cites=["receipts/future/X.json"], payload={"receipt": "receipts/future/X.json"}),
                at._ev(487, "next_work_left", cites=["F015"], payload={"unit_ids": ["F015"], "n": 1}),
            ]
        ),
    }
    verdict = at.eval_no_idle_while_work_exists(at.TimelineView(timeline, "30m"))
    assert verdict["met"] is True, verdict.get("detail")


def test_justified_idle_gap_passes_no_idle_while_work_exists():
    """The same 477s wait PASSes when the gap opens with a complete idle_justified."""
    units = at.passing_units()
    u1 = units["atlas"]
    timeline = {
        "schema": at.TIMELINE_SCHEMA,
        "trial": "30m",
        "duration_s": at.TRIAL_DURATION_S["30m"],
        "elapsed_s": 565,
        "frontier": at.fixture_frontier(),
        "events": at._seq_events(
            [
                at._ev(0, "state_recovered", cites=[at.FRONTIER_REL], payload={"path_taken": "fixture"}),
                at._ev(
                    3,
                    "workunit_launched",
                    cites=[u1["id"], "F012"],
                    payload={"unit": u1, "frontier_id": "F012"},
                ),
                at._ev(88, "mission_state_written", cites=["mission/state.json"], payload={"path": "mission/state.json", "mission_id": "x", "next_action": "wait"}),
                at._ev(
                    88,
                    at.IDLE_JUSTIFIED_KIND,
                    payload={
                        "why": "queue empty and refill returned no novel work; waiting on open handles",
                        "frontiers_asked": ["F012", "F015"],
                        "returned": [
                            {"frontier_id": "F012", "returned": "already_run"},
                            {"frontier_id": "F015", "returned": "already_held"},
                        ],
                        "waiting_on": [{"job_id": "specimen", "pid": 22827, "unit_id": "WU.TORTURE.NO_WAIT.specimen_verify"}],
                        "n_asked": 2,
                        "n_novel": 0,
                    },
                ),
                at._ev(565, "detached_completed", payload={"job_id": "specimen"}),
                at._ev(565, "next_work_left", cites=["F015", "F016"], payload={"unit_ids": ["F015", "F016"], "n": 2}),
            ]
        ),
    }
    verdict = at.eval_no_idle_while_work_exists(at.TimelineView(timeline, "30m"))
    assert verdict["met"] is True, verdict.get("detail")


def test_idle_justified_with_novel_work_still_fails():
    """A wait that reports n_novel>0 is not a justification; it is the defect confessing."""
    units = at.passing_units()
    u1 = units["atlas"]
    timeline = {
        "schema": at.TIMELINE_SCHEMA,
        "trial": "30m",
        "duration_s": at.TRIAL_DURATION_S["30m"],
        "elapsed_s": 565,
        "frontier": at.fixture_frontier(),
        "events": at._seq_events(
            [
                at._ev(
                    3,
                    "workunit_launched",
                    cites=[u1["id"], "F012"],
                    payload={"unit": u1, "frontier_id": "F012"},
                ),
                at._ev(
                    88,
                    at.IDLE_JUSTIFIED_KIND,
                    payload={
                        "why": "waiting",
                        "frontiers_asked": ["F015"],
                        "returned": [{"frontier_id": "F015", "returned": "novel"}],
                        "waiting_on": [{"job_id": "specimen", "pid": 1}],
                        "n_novel": 1,
                    },
                ),
                at._ev(565, "next_work_left", cites=["F015"], payload={"unit_ids": ["F015"], "n": 1}),
            ]
        ),
    }
    verdict = at.eval_no_idle_while_work_exists(at.TimelineView(timeline, "30m"))
    assert verdict["met"] is False, verdict.get("detail")


def _stage_kinds(timeline: dict, kinds: set[str]) -> dict:
    body = json.loads(json.dumps(timeline))
    for event in body.get("events") or []:
        if event.get("kind") in kinds:
            event["staged"] = True
            payload = event.get("payload")
            if not isinstance(payload, dict):
                payload = {}
                event["payload"] = payload
            payload["staged"] = True
            payload["injected_for_condition"] = True
    return body


def test_staged_event_cannot_satisfy_refill_work():
    """G030: a staged work_refilled cannot close refill_work."""
    good = at.build_passing_timeline("30m")
    assert at.verify("30m", good)["verdict"] == "PASS"
    staged = _stage_kinds(good, {"work_refilled"})
    verdict = at.verify("30m", staged)
    assert verdict["verdict"] == "FAIL"
    assert "refill_work" in verdict["unmet"]
    refill = next(c for c in verdict["conditions"] if c["id"] == "refill_work")
    assert refill["met"] is False
    assert "staged" in refill["detail"]


def test_staged_event_cannot_satisfy_overlap_detached_work():
    good = at.build_passing_timeline("30m")
    staged = _stage_kinds(good, {"detached_started", "detached_completed"})
    verdict = at.verify("30m", staged)
    assert "overlap_detached_work" in verdict["unmet"]
    overlap = next(c for c in verdict["conditions"] if c["id"] == "overlap_detached_work")
    assert overlap["met"] is False
    assert "staged" in overlap["detail"]


def test_staged_event_cannot_satisfy_use_negative_science():
    good = at.build_passing_timeline("30m")
    staged = _stage_kinds(good, {"negative_science_query", "negative_science_refusal"})
    verdict = at.verify("30m", staged)
    assert "use_negative_science" in verdict["unmet"]
    row = next(c for c in verdict["conditions"] if c["id"] == "use_negative_science")
    assert row["met"] is False
    assert "staged" in row["detail"]


def test_staged_event_cannot_satisfy_alter_priority_from_evidence():
    good = at.build_passing_timeline("30m")
    staged = _stage_kinds(good, {"priority_altered"})
    verdict = at.verify("30m", staged)
    assert "alter_priority_from_evidence" in verdict["unmet"]
    row = next(c for c in verdict["conditions"] if c["id"] == "alter_priority_from_evidence")
    assert row["met"] is False
    assert "staged" in row["detail"]


def test_sixteen_thirty_m_is_the_named_set_without_no_idle():
    assert len(at.SIXTEEN_THIRTY_M) == 16
    assert "no_idle_while_work_exists" not in at.SIXTEEN_THIRTY_M
    for cid in at.FOUR_THIRTY_M:
        assert cid in at.SIXTEEN_THIRTY_M
    assert at.REQUIRED_CONDITIONS["30m"][-1] == "no_idle_while_work_exists"


def test_campaign_science_scars_are_reachable_in_the_live_index():
    """6fc77f169: a scar the index cannot see prunes nothing."""
    report = at.campaign_science_scars_reachable()
    assert report["ok"] is True, report
    assert report["missing"] == []
    assert report["n_reachable"] == len(at.CAMPAIGN_SCIENCE_SCARS)
    for name in at.CAMPAIGN_SCIENCE_SCARS:
        row = report["reachable"][name]
        assert row["reachable"] is True, name
        assert row.get("source_path"), name


def test_driver_emits_the_four_at_real_call_sites():
    """The four events must be emitted by autonomy_run, not by this judge."""
    import ast

    src_path = REPO / "tools/future/autonomy_run.py"
    assert src_path.is_file()
    tree = ast.parse(src_path.read_text())
    names: set[str] = set()
    constants: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            constants.add(node.value)
    for fn in (
        "emit_detached_started",
        "emit_priority_altered",
        "emit_negative_science_query",
        "emit_negative_science_refusal",
        "rank_detachable",
        "_try_refill",
        "_kickoff_overlap",
        "_apply_replan",
    ):
        assert fn in names, fn
    for kind in (
        "work_refilled",
        "detached_started",
        "negative_science_query",
        "negative_science_refusal",
        "priority_altered",
    ):
        assert kind in constants, kind
    # Ranking uses `if prio is None`, not the falsy-zero `or 99` default.
    # The documenting comment may still name the scar.
    src = src_path.read_text()
    assert "if prio is None:" in src
    assert "ranked.sort(key=lambda pair: pair[0])" in src


def test_priority_zero_start_requires_a_live_pid_and_started_at():
    fake = {
        "kind": "detached_started",
        "payload": {"job_id": "WU.TORTURE.NO_WAIT.specimen_verify", "capability": "specimen_verify.py"},
    }
    assert at._priority_zero_start(fake) is False
    live = {
        "kind": "detached_started",
        "payload": {
            "job_id": "WU.TORTURE.NO_WAIT.specimen_verify",
            "capability": "specimen_verify.py",
            "pid": 22827,
            "started_at": 1788141745.19,
        },
    }
    assert at._priority_zero_start(live) is True
    staged = {
        "kind": "detached_started",
        "staged": True,
        "payload": {
            "job_id": "WU.TORTURE.NO_WAIT.specimen_verify",
            "capability": "specimen_verify.py",
            "pid": 1,
            "started_at": 1.0,
            "staged": True,
        },
    }
    assert at._priority_zero_start(staged) is False


def test_judge_four_from_sealed_quotes_call_site_and_observation():
    """A constructed real-looking timeline is quoted; a staged one is not met."""
    units = at.passing_units()
    u1, u3 = units["atlas"], units["lpc"]
    timeline = {
        "schema": at.TIMELINE_SCHEMA,
        "trial": "30m",
        "duration_s": 1800,
        "elapsed_s": 1800,
        "frontier": at.fixture_frontier(),
        "events": at._seq_events(
            [
                at._ev(0, "state_recovered", cites=[at.FRONTIER_REL], payload={"path_taken": "fixture"}),
                at._ev(
                    3,
                    "workunit_launched",
                    cites=[u1["id"], "F012"],
                    payload={"unit": u1, "frontier_id": "F012"},
                ),
                at._ev(
                    10,
                    "result_ingested",
                    cites=["receipts/future/CODEX_INGEST_STATE.json", u1["id"]],
                    payload={"receipt": "receipts/future/CODEX_INGEST_STATE.json"},
                ),
                at._ev(
                    12,
                    "work_refilled",
                    cites=[u3["id"], "F007"],
                    payload={
                        "unit_ids": [u3["id"]],
                        "n": 1,
                        "source": "frontiers.refill",
                        "queue_remaining_when_asked": 3,
                    },
                ),
                at._ev(
                    20,
                    "detached_started",
                    payload={
                        "job_id": "WU.TORTURE.NO_WAIT.specimen_verify",
                        "pid": 11,
                        "started_at": 100.0,
                        "capability": "specimen_verify.py",
                        "unit_id": "WU.TORTURE.NO_WAIT.specimen_verify",
                    },
                ),
                at._ev(
                    20,
                    "detached_started",
                    payload={
                        "job_id": "WU.AUTONOMY.detach.census",
                        "pid": 12,
                        "started_at": 100.5,
                        "unit_id": "WU.AUTONOMY.detach.census",
                    },
                ),
                at._ev(
                    21,
                    "detached_overlap_confirmed",
                    payload={
                        "job_ids": [
                            "WU.TORTURE.NO_WAIT.specimen_verify",
                            "WU.AUTONOMY.detach.census",
                        ],
                        "n_live": 2,
                    },
                ),
                at._ev(
                    22,
                    "negative_science_query",
                    cites=[at.NEG_INDEX_REL],
                    payload={"query": {"model": "qwen3.8-27b", "organ": "mlp", "n_families": 63}},
                ),
                at._ev(
                    22,
                    "negative_science_refusal",
                    cites=["receipts/future/MLP_STRUCTURED_OPERATOR.json"],
                    payload={
                        "query": {"hypothesis_family": "MONARCH"},
                        "source_path": "receipts/future/MLP_STRUCTURED_OPERATOR.json",
                        "scar_id": "MONARCH",
                    },
                ),
                at._ev(
                    23,
                    "priority_altered",
                    cites=["receipts/future/CODEX_INGEST_STATE.json"],
                    payload={"before": ["a", "b"], "after": ["b", "a"], "cause": "ingest"},
                ),
                at._ev(
                    40,
                    "detached_completed",
                    payload={
                        "job_id": "WU.AUTONOMY.detach.census",
                        "pid": 12,
                        "finished_at": 108.0,
                    },
                ),
                at._ev(
                    80,
                    "detached_completed",
                    payload={
                        "job_id": "WU.TORTURE.NO_WAIT.specimen_verify",
                        "pid": 11,
                        "finished_at": 180.0,
                    },
                ),
            ]
        ),
    }
    four = at.judge_four_from_sealed(timeline)
    assert four["all_four_met"] is True, {k: v.get("detail") for k, v in four["conditions"].items()}
    assert four["staged_event_cannot_satisfy"] is True
    for cid in at.FOUR_THIRTY_M:
        row = four["conditions"][cid]
        assert row["met"] is True, (cid, row.get("detail"))
        quote = row["quote"]
        assert quote, cid
        assert quote["call_site"], cid
        assert quote["observation"], cid
        assert quote["staged"] is False
        assert "tools/future/autonomy_run.py" in quote["call_site"]
    overlap = four["conditions"]["overlap_detached_work"]
    assert overlap["priority_zero_started"]
    assert overlap["priority_zero_in_overlap"] is True

    staged = _stage_kinds(timeline, {"work_refilled", "detached_started", "negative_science_refusal", "negative_science_query", "priority_altered"})
    four_staged = at.judge_four_from_sealed(staged)
    assert four_staged["all_four_met"] is False
    assert set(four_staged["unmet"]) >= set(at.FOUR_THIRTY_M)


def test_event_is_staged_detects_injected_for_condition():
    assert at.event_is_staged({"kind": "work_refilled", "payload": {"staged": True}}) is True
    assert at.event_is_staged({"kind": "work_refilled", "payload": {"injected_for_condition": "refill_work"}}) is True
    assert at.event_is_staged({"kind": "work_refilled", "payload": {"unit_ids": ["x"]}}) is False


def _live_frozen_30m_doc():
    receipt = RECEIPTS / "AUTONOMY_TRIALS.json"
    run = None
    if receipt.is_file():
        body = json.loads(receipt.read_text())
        raw = body.get("frozen_30m_run")
        if isinstance(raw, dict) and raw.get("timeline_path"):
            run = raw
    path = RECEIPTS / "AUTONOMY_TIMELINE_30m.json"
    if run and run.get("timeline_path"):
        cand = REPO / run["timeline_path"]
        if cand.is_file():
            path = cand
    if not path.is_file():
        pytest.skip("no on-disk 30m timeline")
    doc = json.loads(path.read_text())
    if run is None and int(doc.get("elapsed_s") or 0) < 120:
        pytest.skip("on-disk 30m is not a frozen run")
    return path, doc, run


def test_live_frozen_30m_ran_a_real_30_minutes_if_present():
    path, doc, run = _live_frozen_30m_doc()
    assert doc.get("trial") == "30m"
    assert int(doc.get("elapsed_s") or 0) > 0
    assert "verdict" not in doc or doc.get("schema") == at.TIMELINE_SCHEMA
    four = at.judge_four_from_sealed(doc)
    for cid in at.FOUR_THIRTY_M:
        row = four["conditions"][cid]
        quote = row.get("quote") or {}
        assert quote.get("kind") or row.get("detail")
        if row.get("met"):
            assert quote.get("call_site"), (cid, quote)
            assert quote.get("observation"), (cid, quote)
            assert quote.get("staged") is not True
    overlap = four["conditions"]["overlap_detached_work"]
    if overlap.get("met"):
        assert overlap.get("priority_zero_started"), overlap
        assert overlap.get("priority_zero_in_overlap") is True
    ns = four["conditions"]["use_negative_science"]
    scars = (ns.get("campaign_science_scars") or {})
    assert scars.get("ok") is True, scars
    instruments = at.run_instruments_on_timeline(path)
    assert instruments["exempted"] is False
    assert "degeneracy" in instruments
    assert "no_wait" in instruments
    assert instruments["degeneracy"]["instrument"] == "tools.future.autonomy_degeneracy.measure"
    assert instruments["no_wait"]["instrument"] == "tools.future.no_wait_orchestration.classify"
    if run is not None and int(run.get("elapsed_s") or 0) < at.TRIAL_DURATION_S["30m"]:
        assert "elapsed<30m" in str(run.get("report") or "")


def test_frozen_30m_receipt_if_present_hashes_substrate_and_quotes_four():
    receipt = RECEIPTS / "AUTONOMY_TRIALS.json"
    if not receipt.is_file():
        pytest.skip("no AUTONOMY_TRIALS.json")
    doc = json.loads(receipt.read_text())
    run = doc.get("frozen_30m_run")
    if not isinstance(run, dict) or not run:
        pytest.skip("frozen_30m_run not persisted yet")
    sub = run.get("substrate") or {}
    assert sub.get("equal") is True, sub
    assert sub.get("before_digest")
    assert sub.get("before_digest") == sub.get("after_digest")
    assert run.get("elapsed_s") is not None
    if int(run.get("elapsed_s") or 0) < at.TRIAL_DURATION_S["30m"]:
        assert "elapsed<30m" in str(run.get("report") or "")
    four = run.get("four") or {}
    conds = four.get("conditions") or {}
    for cid in at.FOUR_THIRTY_M:
        row = conds.get(cid) or {}
        quote = row.get("quote") or {}
        assert quote.get("call_site"), (cid, row)
        assert quote.get("observation"), (cid, row)
        assert quote.get("staged") is not True
    overlap = conds.get("overlap_detached_work") or {}
    assert overlap.get("priority_zero_started"), overlap
    assert overlap.get("priority_zero_in_overlap") is True
    inst = run.get("instruments") or {}
    assert inst.get("exempted") is False
    assert "degeneracy" in inst and "no_wait" in inst
    assert run.get("staged_event_used") is False
    assert run.get("judged_from") == "sealed_timeline"


def test_a_snapshot_of_zero_runnable_acquits_and_the_control_still_convicts():
    """The distinction two timelines could not make, now made by evidence.

    A runnability_snapshot at the wait reports a COUNT derived from the frontier
    set, the scar list and the launched set. Zero runnable means the wait was
    justified. The archived 477 s control carries NO snapshot, so it stays
    convicted - which is the whole job of a negative control.

    I first keyed this on the driver's `exhausted` flag and this control caught
    it inside a minute: the archived run emits the identical exhausted True, n 0,
    ids [] and ended with twelve frontiers holding novel work. The flag was false
    there; the snapshot cannot be false the same way.
    """
    live = json.loads((at.REPO / "receipts/future/AUTONOMY_TIMELINE_30m.json").read_text())
    snaps = [e for e in (live.get("events") or [])
             if e.get("kind") == "runnability_snapshot"]
    if not snaps:
        pytest.skip("the live timeline predates the snapshot instrument")
    assert int(snaps[0]["payload"]["n_runnable"]) == 0
    assert at.eval_no_idle_while_work_exists(at.TimelineView(live, "30m"))["met"] is True

    control = _load_sealed_30m_timeline()
    assert not [e for e in (control.get("events") or [])
                if e.get("kind") == "runnability_snapshot"], (
        "the control must stay snapshot-free or it stops being the control"
    )
    assert at.eval_no_idle_while_work_exists(at.TimelineView(control, "30m"))["met"] is False


def test_a_failed_snapshot_is_not_evidence_either_way():
    """An errored snapshot must not acquit; it recorded nothing."""
    doc = {"events": [
        {"kind": "runnability_snapshot", "t_s": 10,
         "payload": {"error": "boom", "n_runnable": 0}},
        {"kind": "next_work_left", "t_s": 10, "payload": {"ids": ["FT.A"], "n": 1}},
        {"kind": "mission_state_written", "t_s": 400, "payload": {}},
    ]}
    got = at._work_remained_across_gap(at.TimelineView(doc, "30m"), 10, 400)
    assert got, "an errored snapshot must not be read as zero runnable"
