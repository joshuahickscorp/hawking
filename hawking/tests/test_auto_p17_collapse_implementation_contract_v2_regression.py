"""Focused regression for P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2 collapse.

No-release, no-hardware-qualification boundary: this test only exercises the
in-memory collapse surface in ``hawking.goal_surface``. It performs no spool
mutation, no release, and no hardware qualification.
"""
from __future__ import annotations

import pytest

from hawking.goal_surface import (
    P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2,
    collapse_goal_record,
)


def test_collapse_goal_record_marks_qualified_alternate_route():
    collapsed = collapse_goal_record({
        "goal_id": "GOAL-A52AB48501",
        "model": "openrouter:deepseek/deepseek-v4.1-flash",
    })
    assert collapsed["contract"] == P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2
    assert collapsed["goal_id"] == "GOAL-A52AB48501"
    assert collapsed["remote_model"] is True
    assert collapsed["alternate_route_qualified"] is True
    assert collapsed["collapsed"] is True


def test_collapse_goal_record_rejects_bare_transport_prefix():
    collapsed = collapse_goal_record({
        "goal_id": "GOAL-A52AB48501",
        "model": "openrouter:",
    })
    assert collapsed["remote_model"] is False
    assert collapsed["alternate_route_qualified"] is False


def test_collapse_goal_record_rejects_local_artifact_and_missing_model():
    local = collapse_goal_record({
        "goal_id": "GOAL-A52AB48501",
        "model": "/models/weights.safetensors",
    })
    assert local["remote_model"] is False
    assert local["alternate_route_qualified"] is False

    missing = collapse_goal_record({"goal_id": "GOAL-A52AB48501"})
    assert missing["model"] is None
    assert missing["remote_model"] is False
    assert missing["alternate_route_qualified"] is False


def test_collapse_goal_record_requires_mapping():
    with pytest.raises(TypeError):
        collapse_goal_record(["not", "a", "mapping"])