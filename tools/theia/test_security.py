"""H.3 ACTIVE_TEST transition refuses and cannot be forced."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.theia.authority import AUTHORITY_SCHEMA
from tools.theia.security import (
    LAST_LEGAL_STATE,
    POST_ACTIVE_STATES,
    ActiveTestRefused,
    SecurityMachine,
    SecurityState,
)


def _authority(tmp_path: Path) -> Path:
    path = tmp_path / "authority.json"
    path.write_text(
        json.dumps(
            {
                "schema": AUTHORITY_SCHEMA,
                "program_id": "test-program",
                "allowed_targets": ["lab.example.local"],
                "forbidden_test_classes": ["ACTIVE_TEST"],
                "operator": "pytest-operator",
            }
        )
    )
    return path


def test_active_test_is_modeled():
    assert SecurityState.ACTIVE_TEST.value == "ACTIVE_TEST"
    assert SecurityState.ACTIVE_TEST in POST_ACTIVE_STATES


def test_cannot_occupy_active_test():
    with pytest.raises(ActiveTestRefused, match="cannot occupy"):
        SecurityMachine(state=SecurityState.ACTIVE_TEST)


def test_active_test_transition_refuses_and_cannot_be_forced(tmp_path):
    m = SecurityMachine()
    m.pin_from_file(_authority(tmp_path))
    m.walk_to_last_legal()
    assert m.state == LAST_LEGAL_STATE
    with pytest.raises(ActiveTestRefused, match="hard stop"):
        m.advance(SecurityState.ACTIVE_TEST)
    with pytest.raises(ActiveTestRefused, match="cannot be forced"):
        m.force(SecurityState.ACTIVE_TEST)
    assert m.state == LAST_LEGAL_STATE


def test_skip_to_active_test_from_start_refuses():
    m = SecurityMachine()
    with pytest.raises(ActiveTestRefused):
        m.advance(SecurityState.ACTIVE_TEST)


def test_post_active_states_refuse():
    m = SecurityMachine()
    for state in POST_ACTIVE_STATES:
        with pytest.raises(ActiveTestRefused):
            m.advance(state)


def test_execute_refuses_if_state_is_poked_to_active_test():
    m = SecurityMachine()
    object.__setattr__(m, "_state", SecurityState.ACTIVE_TEST)
    with pytest.raises(ActiveTestRefused, match="not an executable action"):
        m.execute()
