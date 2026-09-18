"""Focused regression for P4_AUTO_IMPLEMENTATION_CONTRACT_V2.

Exercises the observable behavior added to
``hawking.auto_orchestration.audit_worker_packet_for_execution``: the declared
``task_class`` routing label is normalized (trimmed and lower-cased) before it
is matched against ``AUTO_TASK_CLASSES``, and the normalized label is what the
verdict reports.  This is a production-behavior check, not a marker or import
assertion.
"""
from __future__ import annotations

from hawking.auto_orchestration import (
    AUTO_EXECUTION_AUDIT_SCHEMA,
    audit_worker_packet_for_execution,
)


def _packet(task_class):
    return {
        "workunit_id": "WORKUNIT-20260918-63FBC284",
        "goal_id": "GOAL-A52AB48501",
        "task_class": task_class,
    }


def test_padded_and_uppercased_task_class_is_admissible():
    verdict = audit_worker_packet_for_execution(_packet("  Repo_Engineering "))
    assert verdict["schema"] == AUTO_EXECUTION_AUDIT_SCHEMA
    assert verdict["admissible"] is True
    assert verdict["reasons"] == []
    assert verdict["missing_fields"] == []
    # The reported label is the normalized routing decision, not the raw bytes.
    assert verdict["task_class"] == "repo_engineering"


def test_unknown_task_class_still_rejected_after_normalization():
    verdict = audit_worker_packet_for_execution(_packet("  Not_A_Real_Class "))
    assert verdict["admissible"] is False
    assert verdict["reasons"] == ["task_class_not_accepted"]
    assert verdict["task_class"] == "not_a_real_class"


def test_non_string_task_class_reports_not_string():
    verdict = audit_worker_packet_for_execution(_packet(17))
    assert verdict["admissible"] is False
    assert verdict["reasons"] == ["task_class_not_string"]
    assert verdict["task_class"] is None


def test_missing_fields_are_still_surfaced():
    verdict = audit_worker_packet_for_execution({"task_class": "debugging"})
    assert verdict["admissible"] is False
    assert verdict["reasons"] == ["missing_required_fields"]
    assert verdict["missing_fields"] == ["workunit_id", "goal_id"]
    assert verdict["task_class"] == "debugging"