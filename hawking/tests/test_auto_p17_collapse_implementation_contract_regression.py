"""Focused regression for the P17 collapse implementation contract surface.

No-release, no-hardware-qualification boundary: these tests exercise only
in-memory collapse behavior in hawking.goal_surface.
"""
from __future__ import annotations

import pytest

from hawking.goal_surface import (
    GOAL_CONTRACT_SCHEMA,
    GOAL_SCHEMA,
    P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2,
    collapse_goal_record,
    p17_collapse_implementation_contract_v2,
)


def test_contract_surface_reports_collapse_shape():
    contract = p17_collapse_implementation_contract_v2()
    assert contract["schema"] == GOAL_CONTRACT_SCHEMA
    assert contract["contract"] == P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2
    assert contract["collapse"]["goal_schema"] == GOAL_SCHEMA
    assert contract["collapse"]["weight_artifact_suffixes"]


def test_collapse_goal_record_normalizes_remote_model():
    collapsed = collapse_goal_record(
        {"goal_id": "GOAL-A52AB48501", "model": "openrouter:meta/llama-3"}
    )
    assert collapsed["goal_id"] == "GOAL-A52AB48501"
    assert collapsed["model"] == "openrouter:meta/llama-3"
    assert collapsed["remote_model"] is True
    assert collapsed["collapsed"] is True
    assert collapsed["contract"] == P17_COLLAPSE_IMPLEMENTATION_CONTRACT_V2


def test_collapse_goal_record_keeps_local_artifact_local():
    collapsed = collapse_goal_record(
        {"goal_id": "GOAL-LOCAL", "model": "/models/weights.safetensors"}
    )
    assert collapsed["remote_model"] is False


def test_collapse_goal_record_tolerates_missing_fields():
    collapsed = collapse_goal_record({})
    assert collapsed["goal_id"] is None
    assert collapsed["model"] is None
    assert collapsed["remote_model"] is False


def test_collapse_goal_record_rejects_non_mapping():
    with pytest.raises(TypeError):
        collapse_goal_record(["not", "a", "mapping"])