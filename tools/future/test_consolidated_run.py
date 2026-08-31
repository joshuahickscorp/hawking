"""Consolidated-run descriptor: refuse on an unmet gate, never look like a launch.

Negative controls a validator nobody has watched fail would silently drift:

* current gate (10 of 16 met) — build_run REFUSES and names all six unmet
* a descriptor missing any required field is rejected by its own validator
* a curriculum specimen that is not whole-tree verified cannot appear as a
  source without verification_status attached
* fallback identical to the resident is rejected (identical means no fallback)
* a bare list cannot stand in for EMPTY or UNAVAILABLE
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from tools.future import consolidated_run as cr
from tools.future import odyssey_launch as ol
from tools.future import super_resident as sr
from tools.future._common import HARDWARE_FIELDS, RECEIPTS, HardwareClaimError, write_receipt


CURRENT_UNMET = (
    "resident_autonomy_trial_pass",
    "specimen_curriculum_ready",
    "doctor_callable",
    "gravity_callable",
    "nr_nx_path_callable",
    "protected_scheduling",
)


def _all_met():
    return [{"id": cid, "met": True, "reason": "injected"} for cid in ol.CRITERION_IDS]


def _all_unmet():
    return [{"id": cid, "met": False, "reason": "injected unmet"} for cid in ol.CRITERION_IDS]


def _identity(i: str) -> dict:
    return cr.tagged(cr.PRESENT, value={"id": i, "role": "fixture"}, source="test")


def _specimens(*rows: dict) -> dict:
    return cr.tagged(cr.PRESENT, items=list(rows), source="test")


def _collection(*rows: dict, status: str = cr.PRESENT) -> dict:
    if status == cr.EMPTY:
        return cr.tagged(cr.EMPTY, reason="fixture none exist", source="test")
    if status == cr.UNAVAILABLE:
        return cr.tagged(cr.UNAVAILABLE, reason="fixture could not read", source="test")
    return cr.tagged(cr.PRESENT, items=list(rows), source="test")


def _valid_descriptor() -> dict:
    spec = {
        "role": "very_small_dense_procedural_speed",
        "repo": "Qwen/Qwen3-0.6B",
        "verification_status": "NOT_IN_VERIFICATION_RECEIPT",
        "whole_tree_verified": False,
    }
    tagged_obj = cr.tagged(cr.PRESENT, value={"ok": True}, source="test")
    return {
        "run_id": "odyssey-i-fixture",
        "start_time": "2026-08-30T00:00:00Z",
        "resident_identity": _identity("resident-a"),
        "fallback_identity": _identity("fallback-b"),
        "machine_genome": tagged_obj,
        "source_specimens": _specimens(spec),
        "phase_i_workgraphs": _collection({"specimen": "fixture"}),
        "phase_ii_listener": tagged_obj,
        "phase_iii_listener": tagged_obj,
        "resource_lanes": tagged_obj,
        "evidence_dag_handle": tagged_obj,
        "negative_science_handle": tagged_obj,
        "nr_nx_destinations": _collection({"rel": "receipts/future/FLASH_NR_COMPLETE.json"}),
        "blocked_triggers": _collection({"id": "sleep.no-gpu-authority", "wake_all_of": ["gpu"]}),
        "qualification_backlog": _collection({"candidate_id": "c1", "status": "BLOCKED"}),
        "time_budget": cr.tagged(cr.UNAVAILABLE, reason="no run has started", source="test"),
        "restart_state": tagged_obj,
    }


def test_current_gate_refuses_and_names_all_six_unmet():
    """NEGATIVE CONTROL: today's gate is 10/16. A pass would be fiction."""
    attempt = cr.build_run()
    live = ol.unmet_criteria()
    assert attempt["allowed"] is False
    assert attempt["verdict"] == "REFUSED"
    assert attempt["descriptor"] is None
    assert attempt["unmet"] == live
    assert attempt["n_unmet"] == 6
    assert attempt["n_met"] == 10
    assert attempt["n_criteria"] == 16
    assert tuple(attempt["unmet"]) == CURRENT_UNMET
    for cid in CURRENT_UNMET:
        assert cid in attempt["reason"]


def test_injected_unmet_never_emits_a_descriptor():
    """NEGATIVE CONTROL: any unmet criterion is enough; the first does not hide the rest."""
    attempt = cr.build_run(criteria=_all_unmet(), inventory=cr.assemble_inventory())
    assert attempt["allowed"] is False
    assert attempt["descriptor"] is None
    assert attempt["n_unmet"] == len(ol.CRITERION_IDS)
    assert attempt["unmet"] == list(ol.CRITERION_IDS)


def test_injected_all_met_emits_a_validated_descriptor():
    inv = cr.assemble_inventory()
    attempt = cr.build_run(
        criteria=_all_met(),
        inventory=inv,
        start_time="2026-08-30T12:00:00Z",
        head="deadbeefdead",
    )
    assert attempt["allowed"] is True
    assert attempt["verdict"] == "LAUNCH"
    desc = attempt["descriptor"]
    assert desc is not None
    assert desc["run_id"].startswith("odyssey-i-")
    assert desc["start_time"] == "2026-08-30T12:00:00Z"
    assert cr.validate_descriptor(desc)["status"] == "ACCEPTED"
    # Gate-pass still cannot mint a descriptor whose fallback equals the resident.
    assert cr._identity_key(desc["resident_identity"]) != cr._identity_key(desc["fallback_identity"])


def test_validator_rejects_every_missing_required_field():
    """NEGATIVE CONTROL: each required field, independently, is enough to reject."""
    for field in cr.REQUIRED_FIELDS:
        desc = _valid_descriptor()
        del desc[field]
        result = cr.validate_descriptor(desc)
        assert result["status"] == "REJECTED", field
        assert any(field in r for r in result["reasons"]), (field, result["reasons"])
    with pytest.raises(cr.DescriptorInvalid):
        broken = _valid_descriptor()
        del broken["run_id"]
        cr.accept_descriptor(broken)


def test_bare_list_is_not_empty_and_not_unavailable():
    """NEGATIVE CONTROL: [] cannot mean both 'none exist' and 'could not read'."""
    desc = _valid_descriptor()
    desc["blocked_triggers"] = []
    result = cr.validate_descriptor(desc)
    assert result["status"] == "REJECTED"
    assert any("bare list" in r for r in result["reasons"])

    desc = _valid_descriptor()
    desc["source_specimens"] = []
    result = cr.validate_descriptor(desc)
    assert result["status"] == "REJECTED"

    empty = cr.tagged(cr.EMPTY, reason="looked; none exist", source="test")
    unread = cr.tagged(cr.UNAVAILABLE, reason="could not read the frontier book", source="test")
    assert empty["status"] == cr.EMPTY
    assert unread["status"] == cr.UNAVAILABLE
    assert empty["items"] == []
    assert unread["items"] is None
    assert empty != unread
    with pytest.raises(ValueError):
        cr.tagged(cr.UNAVAILABLE)
    with pytest.raises(ValueError):
        cr.tagged(cr.EMPTY)


def test_unverified_specimen_cannot_appear_without_status():
    """NEGATIVE CONTROL: a curriculum specimen is not a source without verification_status."""
    specimens = cr.assemble_source_specimens()
    assert specimens["status"] == cr.PRESENT
    assert specimens["items"], "curriculum roles exist; an empty items list would be the wrong fact"
    unverified = [s for s in specimens["items"] if s.get("whole_tree_verified") is not True]
    assert unverified, "today at least one curriculum role is not whole-tree verified"
    for row in specimens["items"]:
        assert "verification_status" in row
        assert row["verification_status"]
    # Flash and Qwen3-0.6B are the live unready roles; they must not look verified.
    by_role = {s["role"]: s for s in specimens["items"]}
    small = by_role["very_small_dense_procedural_speed"]
    flash = by_role["flash_heterogeneous_frontier"]
    assert small["whole_tree_verified"] is False
    assert small["verification_status"] != "WHOLE_TREE_VERIFIED"
    assert flash["whole_tree_verified"] is False
    assert flash["verification_status"] != "WHOLE_TREE_VERIFIED"

    desc = _valid_descriptor()
    desc["source_specimens"] = cr.tagged(
        cr.PRESENT,
        items=[{"role": "flash_heterogeneous_frontier", "repo": "Qwen/Qwen3.8-Flash-Next"}],
        source="test",
    )
    result = cr.validate_descriptor(desc)
    assert result["status"] == "REJECTED"
    assert any("verification_status" in r for r in result["reasons"])


def test_identical_fallback_is_rejected():
    """NEGATIVE CONTROL: identical identities mean there is no fallback."""
    desc = _valid_descriptor()
    desc["fallback_identity"] = dict(desc["resident_identity"])
    result = cr.validate_descriptor(desc)
    assert result["status"] == "REJECTED"
    assert any("identical" in r for r in result["reasons"])
    with pytest.raises(cr.DescriptorInvalid):
        cr.accept_descriptor(desc)

    # The live bodies differ: Qwen27 incumbent vs Flash candidate.
    assert sr.QWEN_ID != sr.FLASH_ID
    live_resident = cr.assemble_resident_identity()
    live_fallback = cr.assemble_fallback_identity()
    if live_resident["status"] == cr.PRESENT and live_fallback["status"] == cr.PRESENT:
        assert cr._identity_key(live_resident) != cr._identity_key(live_fallback)


def test_resource_lanes_are_imported_not_restated():
    lanes = cr.assemble_resource_lanes()
    assert lanes["status"] == cr.PRESENT
    val = lanes["value"]
    assert val["authority"] == "tools.future.frontiers"
    assert val["imported"] is True
    assert val["restated"] is False
    from tools.future import frontiers as fr

    assert val["this_host"] == list(fr.THIS_HOST_LANES)
    assert val["hardware_blocked"] == list(fr.HARDWARE_LANES)


def test_tagged_unavailable_carries_a_reason_and_null_items():
    budget = cr.assemble_time_budget()
    assert budget["status"] == cr.UNAVAILABLE
    assert budget["reason"]
    assert budget["items"] is None
    restart = cr.assemble_restart_state()
    assert restart["status"] == cr.PRESENT
    odyssey = restart["value"]["odyssey_run"]
    assert odyssey["status"] == cr.EMPTY
    assert odyssey["items"] == []
    assert odyssey["reason"]


def test_build_writes_refusal_receipt_not_a_launch():
    path = cr.build()
    assert path == RECEIPTS / cr.RECEIPT
    doc = json.loads(path.read_text())
    assert doc["schema"] == cr.SCHEMA
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["measurement_state"] == "STATIC_ONLY"
    assert doc["bench"]["gpu_authority"] is False
    assert doc["descriptor_emitted"] is False
    assert doc["descriptor"] is None
    assert doc["allowed"] is False
    assert doc["verdict"] == "REFUSED"
    assert doc["phase_transition"] == "NOT_STARTED"
    assert doc["disk_inventory_is_not_a_launch"] is True
    assert doc["unmet"] == list(CURRENT_UNMET)
    assert doc["seal_sha256"]
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(blob).hexdigest() == doc["seal_sha256"]
    assert doc["resident_callable"]["entry_point"]
    assert doc["resident_callable"]["fails_closed"]
    assert doc["recovered_implementation"]
    assert doc["gaps_closed"]
    assert doc["negative_findings"]


def test_receipt_has_no_numeric_hardware_fields():
    path = cr.build()
    doc = json.loads(path.read_text())

    def walk(node, p=""):
        if isinstance(node, dict):
            for k, v in node.items():
                here = f"{p}.{k}" if p else k
                if k in HARDWARE_FIELDS and isinstance(v, (int, float)):
                    raise AssertionError(f"hardware field {here}={v!r}")
                walk(v, here)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{p}[{i}]")

    walk(doc)


def test_hardware_claim_still_raises_through_write_receipt():
    with pytest.raises(HardwareClaimError):
        write_receipt(
            "ODYSSEY_CONSOLIDATED_RUN_SHOULD_NOT_LAND.json",
            {"schema": "test", "tps": 12.0},
            "tools/future/test_consolidated_run.py",
        )


def test_module_parses():
    src = Path(__file__).with_name("consolidated_run.py").read_text()
    ast.parse(src)


def test_qualification_backlog_does_not_copy_measurements():
    backlog = cr.assemble_qualification_backlog()
    if backlog["status"] != cr.PRESENT:
        # Sparse miss is not project absence; the tag must still be honest.
        assert backlog["status"] in {cr.EMPTY, cr.UNAVAILABLE}
        assert backlog["reason"]
        return
    for row in backlog["items"]:
        assert "measurements" not in row
        assert "expected_gpu_ns_mechanism" not in row
        assert "expected_dispatch_reduction" not in row
        assert set(row).issubset(set(cr._QUEUE_KEEP))
