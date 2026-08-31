"""Negative controls for the 30-minute power-torture composer.

A composer nobody has watched refuse is a composer that will silently pad a
workload with copies of the 1h trial. These tests prove it can reject a mix
missing a required transition class, fire FAIL_NO_WAIT_ORCHESTRATION on a
synthetic wait-while-runnable timeline, refuse to count a mutation without a
proven rollback, and refuse to count an UNTESTED status challenge.
"""
from __future__ import annotations

import ast
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
    assert proofs["nr_nx"]["callable"] is False or proofs["nr_nx"]["first_failing_stage"]
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


def test_build_emits_sealed_static_only_receipt(book, proofs):
    out = pt.build()
    doc = json.loads(out.read_text())
    assert out.parent == RECEIPTS
    assert out.name == pt.RECEIPT
    assert doc["schema"] == pt.SCHEMA
    assert doc["version"] == 1
    assert doc["seal_sha256"]
    assert doc["evidence_class"] == "STATIC_ONLY"
    assert doc["gpu_authority"] is False
    assert doc["bench"]["state"] == "UNKNOWN"
    assert doc["bench"]["gpu_authority"] is False
    body = {k: v for k, v in doc.items() if k != "seal_sha256"}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert doc["seal_sha256"] == hashlib.sha256(blob).hexdigest()
    _assert_no_hardware_claims(doc)
    for key in HARDWARE_FIELDS:
        if key in doc:
            assert not isinstance(doc[key], (int, float))
    assert doc["detector"]["fail_verdict"] == pt.FAIL_NO_WAIT
    assert doc["detector"]["negative_fired"] is True
    assert doc["credit_rules"]["mutation_without_rollback_counts"] is False
    assert doc["credit_rules"]["untested_status_challenge_counts"] is False
    assert doc["resident_callable"]["entry_point"]
    assert doc["resident_callable"]["receipt"] == f"receipts/future/{pt.RECEIPT}"
    assert "recovered_implementation" in doc
    assert "gaps_closed" in doc
    assert "negative_findings" in doc
    if doc["admitted"]:
        assert 6 <= doc["n_units"] <= 12
        assert doc["n_replans"] >= 2
        assert any(u.get("launch") == "detached" for u in doc["units"])


def test_receipt_rejects_a_hardware_number():
    with pytest.raises(HardwareClaimError):
        _assert_no_hardware_claims({"tps": 1})


def test_module_parses_and_has_no_stubs():
    src = Path(pt.__file__).read_text()
    ast.parse(src)
    for needle in ("raise NotImplementedError", "TODO", "\n    pass\n"):
        assert needle not in src
    assert "pytest.skip(" not in src
