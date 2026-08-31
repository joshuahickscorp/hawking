"""Status causality: a detector nobody has watched reject is decoration.

The five historical overreaches must fire. Three well-founded statuses must
not. A status with no recorded probe is UNTESTED, never OVERREACHING. The
routine never names the true world state.
"""
from __future__ import annotations

import ast
import json

from tools.future import autonomy_scars as asc
from tools.future import status_causality as sc
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, _assert_no_hardware_claims


def test_the_five_historical_cases_are_overreaching():
    cases = sc.historical_cases()
    assert len(cases) == 5
    statuses = {c["status"] for c in cases}
    assert statuses >= {
        "BLOCKED_NO_METAL_GPU",
        "MODEL_MISSING",
        "SPECIMEN_NOT_PRESENT",
        "WEIGHTS_NOT_PRESENT",
        "doctor_callable",
    }
    for case in cases:
        row = sc.challenge(case)
        assert row["verdict"] == sc.OVERREACHING, (
            f"{case['id']} verdict={row['verdict']} claim={row.get('claim_kind')} "
            f"probe={row.get('probe_kind')}"
        )
        assert sc.verdict(case) == sc.OVERREACHING
        assert row["verdict"] not in sc.FORBIDDEN_VERDICTS
        assert row["probe_performed"]
        assert row["direct_observation"]
        assert row["interpretation"]
        assert row["alternatives"]
        assert row["falsifier"]
        assert row["confidence"]["level"] == "LOW"
        assert any(
            a["consistent_with_observation"] and not a["consistent_with_claim"]
            for a in row["alternatives"]
        )


def test_scan_detects_the_five_from_their_receipt_shapes():
    excerpts = [c["receipt_excerpt"] for c in sc.historical_cases()]
    rows = sc.scan(excerpts, include_historical=False)
    over = {r["status"] for r in rows if r["verdict"] == sc.OVERREACHING}
    assert "BLOCKED_NO_METAL_GPU" in over
    assert "MODEL_MISSING" in over
    assert "SPECIMEN_NOT_PRESENT" in over
    assert "WEIGHTS_NOT_PRESENT" in over
    assert "doctor_callable" in over
    assert "gravity_callable" in over, "the fifth case is one unreachable bar covering both tools"


def test_at_least_three_well_founded_statuses_are_supported():
    fixtures = sc.supported_fixtures()
    assert len(fixtures) >= 3
    for fix in fixtures:
        row = sc.challenge(fix)
        assert row["verdict"] == sc.SUPPORTED, (
            f"{fix['id']} verdict={row['verdict']} claim={row.get('claim_kind')} "
            f"probe={row.get('probe_kind')} obs={row.get('direct_observation')!r}"
        )
        assert sc.verdict(fix) == sc.SUPPORTED
        assert row["confidence"]["level"] == "HIGH"
        assert not any(
            a["consistent_with_observation"] and not a["consistent_with_claim"]
            for a in row["alternatives"]
        )


def test_scan_of_well_founded_receipts_does_not_cry_wolf():
    excerpts = [f["receipt_excerpt"] for f in sc.supported_fixtures()]
    rows = sc.scan(excerpts, include_historical=False)
    assert rows, "scan produced nothing from the well-founded receipts"
    over = [r for r in rows if r["verdict"] == sc.OVERREACHING]
    assert over == [], f"well-founded receipts flagged overreaching: {over}"
    supported = [r for r in rows if r["verdict"] == sc.SUPPORTED]
    assert len(supported) >= 3


def test_a_status_with_no_recorded_probe_is_untested_never_overreaching():
    bare = sc.challenge(
        {"status": "SOME_NOVEL_BLOCKER", "interpretation": "the GPU is missing"}
    )
    assert bare["verdict"] == sc.UNTESTED
    assert sc.verdict({"status": "SOME_NOVEL_BLOCKER"}) == sc.UNTESTED
    # Even a historically overreaching LABEL, given as a document that recorded
    # no probe, is UNTESTED. Absence of evidence about the probe is not
    # evidence the claim is unjustified.
    rows = sc.scan(
        [{"status": "BLOCKED_NO_METAL_GPU"}],
        include_historical=False,
    )
    assert len(rows) == 1
    assert rows[0]["verdict"] == sc.UNTESTED


def test_the_routine_never_returns_wrong_and_never_asserts_the_world():
    """BLOCKED_NO_METAL_GPU was falsified as a host property. This routine
    still must not say the status is 'wrong': that would adjudicate the world
    rather than the claim-to-probe link."""
    row = sc.challenge("BLOCKED_NO_METAL_GPU")
    assert row["verdict"] == sc.OVERREACHING
    assert "wrong" not in row["verdict"].lower()
    for key in sc.WORLD_STATE_KEYS:
        assert key not in row
    assert row.get("does_not_assert_world_state") is None
    # Alternatives are hypotheticals, labelled as such.
    for alt in row["alternatives"]:
        assert "hypothetical" in alt


def test_the_same_path_probe_supports_a_narrow_label_and_overreaches_a_broad_one():
    """The regex attacker cried wolf by flagging every status-shaped string.
    The same observation must be SUPPORTED under the label the probe actually
    establishes, and OVERREACHING under the causal label that was taken."""
    probe = {
        "probe_kind": sc.PROBE_PATH_EXISTENCE,
        "probe_performed": (
            "Path.exists() on /Users/scammermike/models/qwen3.8-27b-abliterated-bf16"
        ),
        "direct_observation": "exists=False",
    }
    broad = sc.challenge(
        {**probe, "status": "MODEL_MISSING", "interpretation": "the model is missing"}
    )
    narrow = sc.challenge(
        {
            **probe,
            "status": "declared_path_absent",
            "interpretation": "the declared path is absent",
        }
    )
    assert broad["verdict"] == sc.OVERREACHING
    assert narrow["verdict"] == sc.SUPPORTED


def test_the_same_listing_probe_supports_membership_and_overreaches_absence():
    probe = {
        "probe_kind": sc.PROBE_LISTING,
        "probe_performed": "membership in ModelLake specimens/",
        "direct_observation": "present_in_listing=False",
    }
    broad = sc.challenge(
        {
            **probe,
            "status": "SPECIMEN_NOT_PRESENT",
            "interpretation": "the specimen is not present",
        }
    )
    narrow = sc.challenge(
        {
            **probe,
            "status": "not_in_specimens_listing",
            "interpretation": "the name is not in that listing",
        }
    )
    assert broad["verdict"] == sc.OVERREACHING
    assert narrow["verdict"] == sc.SUPPORTED


def test_the_same_process_error_supports_the_failure_and_overreaches_the_host():
    probe = {
        "probe_kind": sc.PROBE_PROCESS_ERROR,
        "probe_performed": "any error at dense_source_bf16_prefix_initialization",
        "direct_observation": (
            "failure.stage=dense_source_bf16_prefix_initialization; "
            "failure.error='metal: no Metal-capable GPU'"
        ),
    }
    broad = sc.challenge(
        {
            **probe,
            "status": "BLOCKED_NO_METAL_GPU",
            "interpretation": "this host has no Metal-capable GPU",
        }
    )
    narrow = sc.challenge(
        {
            **probe,
            "status": "process_failed_at_prefix_initialization",
            "interpretation": "this process failed at prefix initialization",
        }
    )
    assert broad["verdict"] == sc.OVERREACHING
    assert narrow["verdict"] == sc.SUPPORTED


def test_stale_metadata_supports_the_field_and_overreaches_weights_absent():
    probe = {
        "probe_kind": sc.PROBE_METADATA,
        "probe_performed": "read schools.Flash.physical_status",
        "direct_observation": "metadata_only_weights_not_present",
    }
    broad = sc.challenge(
        {
            **probe,
            "status": "WEIGHTS_NOT_PRESENT",
            "interpretation": "Flash weights are not present",
        }
    )
    narrow = sc.challenge(
        {
            **probe,
            "status": "law_store_records_physical_status",
            "interpretation": "the law store currently records this physical_status",
        }
    )
    assert broad["verdict"] == sc.OVERREACHING
    assert narrow["verdict"] == sc.SUPPORTED


def test_a_literal_constant_cannot_establish_a_capability_absence():
    row = sc.challenge(
        {
            "status": "gravity_callable",
            "probe_kind": sc.PROBE_LITERAL,
            "probe_performed": "schedule=False, frontier=False, refill=False as literals",
            "direct_observation": "schedule/frontier/refill are false",
            "interpretation": "Gravity is not resident-callable",
        }
    )
    assert row["verdict"] == sc.OVERREACHING


def test_a_measured_unmet_criterion_is_supported():
    """protected_scheduling still refuses today, but for named measured reasons.
    Flagging that as OVERREACHING would be the cry-wolf failure."""
    row = sc.challenge(
        {
            "status": "protected_scheduling",
            "probe_kind": sc.PROBE_MEASURED_FLAGS,
            "probe_performed": "evaluate contamination_class and gpu_authority",
            "direct_observation": (
                "met=False unmet_flags=['invoke','schedule'] "
                "reason=contamination_class='HEAVY' gpu_authority=False"
            ),
            "interpretation": (
                "protected scheduling cannot start: contamination_class='HEAVY' "
                "(needs QUIESCENT), qualification gpu_authority=False"
            ),
            "claim_kind": sc.CLAIM_MEASURED_UNMET,
        }
    )
    assert row["verdict"] == sc.SUPPORTED


def test_a_partial_hash_cannot_claim_whole_tree_verified():
    row = sc.challenge(
        {
            "status": "WHOLE_TREE_VERIFIED",
            "probe_kind": sc.PROBE_HASH,
            "probe_performed": "recompute published digests",
            "direct_observation": {
                "n_files": 10,
                "verified": 9,
                "mismatched": 0,
                "no_remote_digest": 1,
                "bytes_hashed": 1,
            },
            "interpretation": (
                "this specimen's published digests match the hashes recomputed here"
            ),
            "claim_kind": sc.CLAIM_DIGEST_MATCH,
        }
    )
    assert row["verdict"] == sc.OVERREACHING


def test_challenge_by_status_string_recovers_the_historical_probe():
    row = sc.challenge("BLOCKED_NO_METAL_GPU")
    assert row["verdict"] == sc.OVERREACHING
    assert row["probe_kind"] == sc.PROBE_PROCESS_ERROR
    assert "prefix_initialization" in row["probe_performed"]


def test_the_law_is_the_scar_already_recorded():
    scar = [s for s in asc.scars() if s["id"] == "STATUS_LABEL_LAUNDERED_AS_CAUSAL_CLAIM"][0]
    assert "STATUS LABELS ARE HYPOTHESES" in scar["law"]
    assert sc.LAW.startswith("A STATUS MAY ASSERT ONLY WHAT ITS ACTUAL PROBE ESTABLISHES")
    row = sc.challenge("BLOCKED_NO_METAL_GPU")
    recovered = " ".join(row.get("recovered_from") or [])
    assert scar["id"] in recovered
    # The routine restates the law rather than the world; the scar stays open
    # on the undiagnosed process failure.
    assert "unidentified" in scar["reopen_condition"]


def test_scan_copes_when_a_named_receipt_is_absent():
    """Sparse checkout: a missing receipt is a recorded miss, not a skip
    and not a fabricated status."""
    rows = sc.scan(
        ["receipts/future/DOES_NOT_EXIST_FOR_THIS_TEST.json"],
        include_historical=False,
    )
    assert rows == []


def test_scan_of_the_law_store_flags_stale_physical_status_either_way():
    """If the live receipt is loadable, the detector must fire on it.
    If this checkout cannot read it, the excerpt drawn from that receipt
    still must fire. Never skip."""
    live = sc._load_receipt("receipts/future/ODYSSEY2_LAW_STORE.json")
    if live is None:
        docs: list = [sc.HISTORICAL_CASES[3]["receipt_excerpt"]]
    else:
        docs = [live]
    rows = sc.scan(docs, include_historical=False)
    assert any(
        r["status"] == "WEIGHTS_NOT_PRESENT" and r["verdict"] == sc.OVERREACHING
        for r in rows
    ), rows


def test_scan_of_the_capture_boundary_flags_metal_either_way():
    live = sc._load_receipt(
        "receipts/future/evidence/FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json"
    )
    docs: list = (
        [live] if live is not None else [sc.HISTORICAL_CASES[0]["receipt_excerpt"]]
    )
    rows = sc.scan(docs, include_historical=False)
    assert any(
        r["status"] == "BLOCKED_NO_METAL_GPU" and r["verdict"] == sc.OVERREACHING
        for r in rows
    ), rows


def test_build_emits_a_sealed_static_receipt():
    out = sc.build()
    assert out.parent == RECEIPTS
    assert out.name == sc.RECEIPT
    doc = json.loads(out.read_text())
    assert doc["schema"] == sc.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["n_historical_overreaching"] == 5
    assert doc["n_supported_fixtures"] >= 3
    assert doc["untested_control"]["verdict"] == sc.UNTESTED
    assert doc["does_not_assert_world_state"] is True
    assert doc["resident_callable"]["entry_point"] == "tools.future.status_causality.challenge()"
    assert doc["resident_callable"]["frontier"] == "FT.VERIFICATION.negative-index"
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        assert key not in doc or doc[key] in (None, "UNKNOWN")
    for key in sc.WORLD_STATE_KEYS:
        assert key not in doc
    assert "wrong" not in json.dumps(doc["verdicts_emitted"]).lower()


def test_challenge_output_has_the_contract_fields():
    row = sc.challenge("MODEL_MISSING")
    for field in (
        "probe_performed",
        "direct_observation",
        "interpretation",
        "confidence",
        "alternatives",
        "falsifier",
        "verdict",
    ):
        assert field in row and row[field] != "" and row[field] is not None
    assert set(row["confidence"]) >= {"level", "about", "would_raise", "would_lower"}


def test_module_parses():
    src = (sc.REPO / "tools" / "future" / "status_causality.py").read_text()
    ast.parse(src)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            func = node.exc.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            assert name != "NotImplementedError"
        if isinstance(node, ast.Pass):
            # A pass in a body that is not `except: pass` on a documented path
            # would be a stub. This module must not have any.
            raise AssertionError(f"pass at line {node.lineno}")


def test_emit_time_entry_point_returns_the_five_fields_and_a_verdict():
    """A gate calls emit() as it stamps a status. This is the emit-time path,
    not scan() of a receipt after the fact."""
    row = sc.emit(
        "declared_path_absent",
        probe_performed="Path.exists() on /tmp/does-not-exist-for-this-test",
        direct_observation="exists=False",
        interpretation="the declared path is absent",
        probe_kind=sc.PROBE_PATH_EXISTENCE,
    )
    assert row["entry"] == "emit"
    assert row["verdict"] == sc.SUPPORTED
    for field in sc.FIVE_RECORDED_FIELDS:
        assert field in row and row[field] not in (None, "", [], {})
    assert set(row["confidence"]) >= {"level", "about", "would_raise", "would_lower"}
    assert row["verdict"] in sc.VERDICTS
    assert row["verdict"] not in sc.FORBIDDEN_VERDICTS


def test_emit_does_not_touch_disk_or_the_catalog(monkeypatch):
    """Cheap enough that a gate has no excuse not to call it."""

    def boom(*_a, **_k):
        raise AssertionError("emit touched disk or git")

    monkeypatch.setattr(sc, "_load_receipt", boom)
    monkeypatch.setattr(sc, "git", boom)
    monkeypatch.setattr(sc, "_read_text", boom)
    bare = sc.emit("BLOCKED_NO_METAL_GPU")
    assert bare["verdict"] == sc.UNTESTED, (
        "emit() of a bare historical label must not catalog-lookup; "
        "that would hide an unrecorded probe behind the teaching set"
    )
    assert bare["entry"] == "emit"
    assert sc.challenge("BLOCKED_NO_METAL_GPU")["verdict"] == sc.OVERREACHING


def test_stamp_writes_the_five_fields_onto_the_status_the_gate_emits():
    status_row = {"id": "doctor_callable", "met": False, "reason": "measured unmet"}
    rec = sc.stamp(
        status_row,
        probe_kind=sc.PROBE_MEASURED_FLAGS,
        probe_performed="evaluate launch criterion doctor_callable against disk evidence",
        direct_observation="met=False unmet_flags=['schedule'] reason=measured",
        interpretation="doctor_callable unmet for named measured reasons",
        claim_kind=sc.CLAIM_MEASURED_UNMET,
    )
    assert rec["verdict"] == sc.SUPPORTED
    for field in sc.FIVE_RECORDED_FIELDS:
        assert field in status_row
        assert status_row[field] == rec[field]
    assert status_row["causality_verdict"] == sc.SUPPORTED
    assert sc.records_five_fields(status_row)


def test_shape_regression_fires_on_a_reemitted_overreach():
    """The five historical overreaches had no regression that would fire on
    the next similar label. This is that regression: same SHAPE, new name."""
    fired = []
    for shape in sc.OVERREACH_SHAPES:
        row = sc.emit(
            shape["reemit_status"],
            probe_kind=shape["probe_kind"],
            probe_performed=shape["reemit_probe"],
            direct_observation=shape["reemit_observation"],
            interpretation=shape["reemit_interpretation"],
            claim_kind=shape["claim_kind"],
        )
        assert row["verdict"] == sc.OVERREACHING, (
            f"{shape['id']} re-emitted as {shape['reemit_status']!r} "
            f"verdict={row['verdict']} (must OVERREACH)"
        )
        matched = sc.matching_overreach_shape(row)
        assert matched == shape["id"], (
            f"{shape['id']} matching_overreach_shape={matched!r}"
        )
        fired.append(shape["id"])
    assert len(fired) == 6
    assert "SHAPE.MIXED_LAYER_AS_ALU_BOUND" in fired
    historical = {s["historical_id"] for s in sc.OVERREACH_SHAPES if s.get("historical_id")}
    assert historical == {
        "HC.BLOCKED_NO_METAL_GPU",
        "HC.MODEL_MISSING",
        "HC.SPECIMEN_NOT_PRESENT",
        "HC.WEIGHTS_NOT_PRESENT",
        "HC.DOCTOR_GRAVITY_LITERAL",
    }


def test_mixed_one_layer_roofline_as_alu_bound_agrees_with_improvement_trial():
    """The live example this week: MIXED one-layer probe reported as ALU_BOUND.
    IMPROVEMENT_TRIAL's misleading_narrow_probe control already fails that pair.
    The two must agree."""
    row = sc.emit(
        "ALU_BOUND",
        probe_performed=sc.ALU_BOUND_MIXED_PROBE,
        direct_observation="verdict=MIXED",
        interpretation=sc.ALU_BOUND_ALL_ORGANS_CLAIM,
    )
    assert row["verdict"] == sc.OVERREACHING
    assert row["probe_kind"] == sc.PROBE_ONE_LAYER_ROOFLINE
    assert row["claim_kind"] == sc.CLAIM_GLOBAL_BINDING
    assert sc.matching_overreach_shape(row) == "SHAPE.MIXED_LAYER_AS_ALU_BOUND"
    assert any(
        a["consistent_with_observation"] and not a["consistent_with_claim"]
        for a in row["alternatives"]
    )
    trial = sc.improvement_trial_alu_bound_control()
    assert trial["readable"] is True, trial
    assert trial["agrees"] is True, trial
    assert trial["failed"] is True
    assert trial["verdict"] == "FAIL"
    assert "ALU_BOUND" in trial["detail"]
    assert "MIXED" in trial["detail"]
    assert trial["detail"] == sc.IMPROVEMENT_TRIAL_ALU_BOUND_DETAIL


def test_the_same_one_layer_probe_supports_mixed_and_overreaches_alu_bound():
    probe = {
        "probe_kind": sc.PROBE_ONE_LAYER_ROOFLINE,
        "probe_performed": sc.ALU_BOUND_MIXED_PROBE,
        "direct_observation": "verdict=MIXED",
    }
    broad = sc.emit(
        "ALU_BOUND",
        interpretation=sc.ALU_BOUND_ALL_ORGANS_CLAIM,
        **probe,
    )
    narrow = sc.emit(
        "MIXED",
        interpretation="this layer's roofline verdict is MIXED",
        claim_kind=sc.CLAIM_FIELD_VALUE,
        **probe,
    )
    assert broad["verdict"] == sc.OVERREACHING
    assert narrow["verdict"] == sc.SUPPORTED


def test_consequential_gates_are_explicit_and_the_selection_rule_is_stated():
    gates = sc.consequential_gates()
    names = [g["name"] for g in gates]
    assert names, "an empty gate list is not a coverage claim"
    assert len(names) == len(set(names)), f"duplicate gate names: {names}"
    for required in sc.G007_NAMED_GATES:
        assert required in names, f"G007 named {required} missing from inventory"
    for extra in (
        "native_mission_gate",
        "autonomy_gate",
        "modellake_gate",
        "vmcp_gate",
        "recovery_gate",
        "research_gate",
        "metal_reachability",
        "flash_nx_audit",
        "odyssey2_law_store",
        "contamination",
        "qualification_pipeline",
        "protected_scheduler",
        "flash_meta_teacher_capture_boundary",
    ):
        assert extra in names, (
            f"{extra} is consequential under the stated rule and was dropped; "
            "do not report coverage by narrowing the definition of consequential"
        )
    assert sc.SELECTION_RULE
    assert "named in the G007 obligation" in sc.SELECTION_RULE
    assert "odyssey_launch" in sc.SELECTION_RULE
    assert "integration_gate" in sc.SELECTION_RULE
    cov = sc.coverage()
    assert cov["selection_rule"] == sc.SELECTION_RULE
    assert tuple(g["name"] for g in gates) == tuple(r["name"] for r in cov["gates"])


def test_coverage_reports_gates_by_name_not_a_percentage():
    cov = sc.coverage()
    for banned in ("percent", "percentage", "coverage_pct", "pct"):
        assert banned not in cov
    recording = cov["recording_five_fields"]
    missing = cov["not_recording_five_fields"]
    unread = cov["unreadable"]
    assert isinstance(recording, list)
    assert isinstance(missing, list)
    assert isinstance(unread, list)
    names = {g["name"] for g in sc.consequential_gates()}
    assert set(recording) | set(missing) | set(unread) == names
    assert set(recording).isdisjoint(missing)
    assert set(recording).isdisjoint(unread)
    assert set(missing).isdisjoint(unread)
    # S015 dissolved the Codex write partition. The eight hcli/agentos gates
    # are wired and must stay named on the recording side. The remainder that
    # must stay named is the Rust capture boundary, not a withdrawn ownership
    # glob.
    wired_hcli = {
        "resident_gate",
        "native_gate",
        "native_mission_gate",
        "autonomy_gate",
        "modellake_gate",
        "vmcp_gate",
        "recovery_gate",
        "research_gate",
    }
    for required in wired_hcli:
        assert required in recording, (
            f"{required} not in recording_five_fields={recording}; "
            "S015 withdrew the partition that previously left these named gaps"
        )
        assert required not in missing
    assert "flash_meta_teacher_capture_boundary" in missing, (
        f"flash_meta_teacher_capture_boundary vanished from "
        f"not_recording_five_fields={missing}; it is a Rust emit point and "
        "must stay named rather than dropped or Python-shimmed"
    )
    for wired in sc.G007_NAMED_GATES:
        assert wired in recording or wired in missing or wired in unread, wired
    assert "native_gate" in recording
    # odyssey_launch, integration_gate and specimen_verify were all wired, so
    # each moved from `missing` to `recording`. The invariant is that every
    # launch criterion is accounted for on exactly one side, never that they are
    # all still gaps.
    criteria_recording = cov["odyssey_launch_criteria_recording_five_fields"]
    criteria_missing = cov["odyssey_launch_criteria_not_recording_five_fields"]
    assert set(criteria_recording).isdisjoint(criteria_missing)
    for cid in sc.ODYSSEY_LAUNCH_CRITERIA:
        assert cid in criteria_recording or cid in criteria_missing, (
            f"odyssey_launch criterion {cid} is accounted for on neither side"
        )


def test_build_receipt_names_odyssey_iii_call_and_coverage_lists():
    out = sc.build()
    doc = json.loads(out.read_text())
    iii = doc["odyssey_iii"]
    assert iii["inherits"] is True
    assert iii["calls"] == "tools.future.status_causality.emit"
    assert iii["caller"].startswith("tools.future.odyssey3_adversary")
    args = iii["arguments"]
    assert args["status"] == "law['law_id']"
    assert "evidence_refs" in args["probe_performed"]
    assert args["interpretation"] == "law['statement']"
    assert "law['scope']" in args["probe_kind"]
    assert doc["resident_callable"]["emit_entry_point"] == (
        "tools.future.status_causality.emit()"
    )
    assert isinstance(doc["gates_recording_five_fields"], list)
    assert isinstance(doc["gates_not_recording_five_fields"], list)
    # Wired gates leave the missing list. The Rust capture boundary stays named.
    assert "resident_gate" in doc["gates_recording_five_fields"]
    assert "flash_meta_teacher_capture_boundary" in doc["gates_not_recording_five_fields"]
    assert set(doc["gates_recording_five_fields"]).isdisjoint(
        doc["gates_not_recording_five_fields"]
    )
    assert doc["alu_bound_mixed_agreement"]["agrees"] is True
    assert doc["n_overreach_shapes"] == 6
    assert "percent" not in doc["coverage"]
    _assert_no_hardware_claims(doc)


def test_odyssey_launch_criteria_match_the_module_when_importable():
    try:
        from tools.future import odyssey_launch as ol
    except Exception as exc:
        raise AssertionError(f"odyssey_launch should import in this checkout: {exc}")
    assert tuple(sc.ODYSSEY_LAUNCH_CRITERIA) == ol.CRITERION_IDS
