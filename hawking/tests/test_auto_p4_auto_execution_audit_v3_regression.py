"""Focused regression for P4_AUTO_EXECUTION_AUDIT_V3 worker-packet admission.

This exercises the production audit function directly: a well-formed packet
with an accepted routing label is admissible, while a packet with a missing
required field or an unaccepted task class is rejected with a deterministic
reason list.  No provider, network, or hardware path is touched.
"""
from __future__ import annotations

from hawking.auto_orchestration import (
    AUTO_EXECUTION_AUDIT_REQUIRED_FIELDS,
    AUTO_EXECUTION_AUDIT_SCHEMA,
    AUTO_TASK_CLASSES,
    audit_worker_packet_for_execution,
)


def _packet(**overrides):
    packet = {
        "workunit_id": "WORKUNIT-20260918-043FCC1E",
        "task_class": "routine_implementation",
        "goal_id": "GOAL-A52AB48501",
    }
    packet.update(overrides)
    return packet


def test_accepted_packet_is_admissible():
    verdict = audit_worker_packet_for_execution(_packet())
    assert verdict["schema"] == AUTO_EXECUTION_AUDIT_SCHEMA
    assert verdict["admissible"] is True
    assert verdict["reasons"] == []
    assert verdict["missing_fields"] == []
    assert verdict["task_class"] == "routine_implementation"


def test_legacy_task_class_is_still_accepted():
    assert "repo_engineering" in AUTO_TASK_CLASSES
    verdict = audit_worker_packet_for_execution(_packet(task_class="repo_engineering"))
    assert verdict["admissible"] is True
    assert verdict["reasons"] == []


def test_missing_required_field_is_rejected():
    packet = _packet()
    del packet["goal_id"]
    verdict = audit_worker_packet_for_execution(packet)
    assert verdict["admissible"] is False
    assert verdict["missing_fields"] == ["goal_id"]
    assert "missing_required_fields" in verdict["reasons"]


def test_blank_required_field_is_rejected():
    verdict = audit_worker_packet_for_execution(_packet(workunit_id="   "))
    assert verdict["admissible"] is False
    assert verdict["missing_fields"] == ["workunit_id"]


def test_unaccepted_task_class_is_rejected():
    verdict = audit_worker_packet_for_execution(_packet(task_class="not_a_real_class"))
    assert verdict["admissible"] is False
    assert verdict["reasons"] == ["task_class_not_accepted"]
    assert verdict["task_class"] == "not_a_real_class"


def test_non_mapping_packet_is_rejected():
    verdict = audit_worker_packet_for_execution(None)
    assert verdict["admissible"] is False
    assert verdict["reasons"] == ["packet_not_mapping"]
    assert verdict["missing_fields"] == list(AUTO_EXECUTION_AUDIT_REQUIRED_FIELDS)


def test_audit_does_not_mutate_packet():
    packet = _packet()
    before = dict(packet)
    audit_worker_packet_for_execution(packet)
    assert packet == before