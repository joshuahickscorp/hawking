"""Acceptance tests for the eight AGENTOS gates.

Each test invokes the gate's implementing symbol through the harness and
writes receipts/acceptance/<GATE>.json. A BLOCKED verdict is a legal
outcome: the test still passes when the receipt records the exact blocker
with a real run. Vacuous receipts fail.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.acceptance.agentos import harness

REPO = Path(__file__).resolve().parents[3]
RECEIPT_DIR = REPO / "receipts" / "acceptance"


@pytest.fixture(scope="module")
def all_results():
    return harness.run_all()


def _receipt(gate: str) -> dict:
    path = RECEIPT_DIR / f"{gate}.json"
    assert path.is_file(), f"missing receipt {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("schema") == harness.SCHEMA
    assert data.get("gate") == gate
    assert data.get("criterion_altered") is False
    assert data.get("evidence_tier") == "FUNCTIONAL_SIM"
    assert data.get("verdict") in {"ACCEPTED", "BLOCKED"}
    assert data.get("symbols_invoked"), f"{gate} did not invoke a symbol"
    assert data.get("measured"), f"{gate} has empty measured"
    assert data.get("output"), f"{gate} has empty output"
    assert data.get("hcli_file")
    if data["verdict"] == "BLOCKED":
        assert data.get("blocker"), f"{gate} BLOCKED without a blocker"
    else:
        assert data.get("blocker") in (None, "")
    return data


def test_repair_bounded(all_results):
    data = _receipt("AGENTOS_REPAIR_BOUNDED")
    m = data["measured"]
    assert m["max_repair_depth"] == 3
    assert m["max_repairs_per_root"] == 6
    assert m["unique_deepest"] <= m["max_repair_depth"]
    assert m["unique_n_repairs"] <= m["max_repairs_per_root"]
    assert m["cycle_n_repairs"] <= m["max_repairs_per_root"]
    assert m["exhausted_not_rereadied"] is True
    assert m["negative_control_deepest"] > m["max_repair_depth"]
    assert m["repair_does_not_widen_resource_class"] is True
    assert any("Scheduler" in str(s.get("symbol")) for s in data["symbols_invoked"])
    assert data["verdict"] == "ACCEPTED"


def test_retry_classified(all_results):
    data = _receipt("AGENTOS_RETRY_CLASSIFIED")
    m = data["measured"]
    assert m["scheduler_fail_consults_classify_failure"] is False
    assert m["non_retryable_still_emits_repair"]
    assert "TRANSIENT_BACKEND" in m["failure_kinds_implemented"]
    assert "DISK_HEADROOM_BLOCK" in m["d2_missing"]
    assert data["verdict"] == "BLOCKED"
    assert "classify_failure" in (data.get("blocker") or "")
    # Catalog symbol still invoked.
    assert any("_record_fingerprint" in str(s.get("symbol")) for s in data["symbols_invoked"])


def test_circuit_breaker(all_results):
    data = _receipt("AGENTOS_CIRCUIT_BREAKER")
    m = data["measured"]
    raised = m["no_progress_raised"]
    assert raised is not None
    assert raised["type"] == "NO_PROGRESS"
    assert raised["count"] >= m["no_progress_threshold"]
    assert m["changing_fingerprint_raised"] is None
    assert m["backend_health_open_state"] == "circuit_open"
    assert m["backend_health_allows_when_open"] is False
    assert any(s.get("symbol") == "NO_PROGRESS" for s in data["symbols_invoked"])
    assert data["verdict"] == "ACCEPTED"


def test_cancellation(all_results):
    data = _receipt("AGENTOS_CANCELLATION")
    m = data["measured"]
    assert m["abort_verdict"] == "ABORTED"
    assert m["state_phase"] == "cancelled"
    assert m["state_cancel_reason"] == "acceptance-operator-abort"
    assert m["durable_cancel_file"] is True
    assert m["shared_checkpoint_id"] is True
    assert m["new_generation"] is True
    assert m["coop_phase"] == "cancelled"
    assert m["coop_repairs"] == []
    assert any(s.get("symbol") == "abort" for s in data["symbols_invoked"])
    assert data["verdict"] == "ACCEPTED"


def test_orphan_reconciliation(all_results):
    data = _receipt("AGENTOS_ORPHAN_RECONCILIATION")
    m = data["measured"]
    assert m["live_state"] == "RUNNING"
    assert m["duplicate_ids"] == []
    assert m["after_kill_state"] in {"INTERRUPTED", "FAILED", "CANCELLED"}
    assert m["jobs_after_kill"] == 1
    assert m["dag_statuses"]["live-grok"] == "running"
    assert m["dag_statuses"]["dead-local"] == "interrupted"
    assert m["dag_statuses"]["grok-failed"] == "failed"
    assert m["n_units_after_recover"] == 3
    assert m["no_silent_duplicate"] is True
    assert any("BackgroundJobStore" in str(s.get("symbol")) for s in data["symbols_invoked"])
    assert data["verdict"] == "ACCEPTED"


def test_persistence_single_authority(all_results):
    data = _receipt("AGENTOS_PERSISTENCE_SINGLE_AUTHORITY")
    m = data["measured"]
    assert m["parent_acquired"] is True
    assert m["child_acquired_while_held"] is False
    assert m["acquire_after_release"] is True
    assert m["shared_checkpoint_id"] is True
    assert m["d3_fields_missing"] == []
    assert set(m["d3_fields_audited"]) == set(harness.D3_FIELDS)
    assert m["audit"]["mutation lease"]["canary"] is True
    assert any(s.get("symbol") == "MutationLock" for s in data["symbols_invoked"])
    assert data["verdict"] == "ACCEPTED"


def test_checkpoint_atomicity(all_results):
    data = _receipt("AGENTOS_CHECKPOINT_ATOMICITY")
    m = data["measured"]
    assert m["atomic_write_sigkill"]["ok"] is True
    assert m["atomic_write_sigkill"]["live_has_old"] is True
    assert m["atomic_write_sigkill"]["live_has_new"] is False
    assert m["between_writes"]["coherent"] is True
    assert m["between_writes"]["mixture"] is False
    assert m["between_writes"]["recovered_backend_task_id"] == "task-GEN1"
    assert m["shared_checkpoint_ids"] is True
    assert m["write_program_checkpoint_invoked"] is True
    assert any("write_program_checkpoint" in str(s.get("symbol")) for s in data["symbols_invoked"])
    assert data["verdict"] == "ACCEPTED"


def test_restart_coherence(all_results):
    data = _receipt("AGENTOS_RESTART_COHERENCE")
    m = data["measured"]
    assert m["mission_id_survived"] is True
    assert m["unit_ids_survived"] is True
    assert m["completed_still_completed"] is True
    assert m["stuck_interrupted"] is True
    assert m["backend_task_id_survived"] is True
    assert m["completed_not_replayed"] is True
    assert "done1" not in m["units_executed_after_restart"]
    assert m["later_completed"] is True
    assert m["dead_pid_lock_recoverable"] is True
    assert any("run_recovery_gate" in str(s.get("symbol")) for s in data["symbols_invoked"])
    assert data["verdict"] == "ACCEPTED"


def test_summary_counts_and_criterion_unaltered(all_results):
    summary = all_results["summary"]
    path = RECEIPT_DIR / "AGENTOS_ACCEPTANCE_SUMMARY.json"
    disk = json.loads(path.read_text(encoding="utf-8"))
    assert disk["assigned_count"] == 8
    assert disk["accepted_count"] + disk["blocked_count"] == 8
    assert disk["criterion_altered"] is False
    assert summary["criterion_altered"] is False
    assert set(disk["gates"]) == set(harness.GATES)
    # Honest split: retry classification is the known production-path gap.
    assert disk["verdicts"]["AGENTOS_RETRY_CLASSIFIED"] == "BLOCKED"
    assert disk["accepted_count"] == 7
    assert disk["blocked_count"] == 1


def test_no_negative_control_left_in_source():
    text = Path(harness.__file__).read_text(encoding="utf-8")
    assert "MAX_REPAIR_DEPTH = 12" in text  # the control exists
    # The production modules we import must still carry the real bound.
    from hcli.workunit import MAX_REPAIR_DEPTH, MAX_REPAIRS_PER_ROOT

    assert MAX_REPAIR_DEPTH == 3
    assert MAX_REPAIRS_PER_ROOT == 6
