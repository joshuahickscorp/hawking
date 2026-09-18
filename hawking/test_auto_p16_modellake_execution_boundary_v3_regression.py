"""Focused regression for the P16 modellake execution boundary (V3).

Exercises the executable boundary check in ``hawking.odyssey``: the registry
must gate every mutating verb on ``confirm``, must not gate read-only verbs,
and must never register ``admit_check`` (which can reclaim disk).
"""
from __future__ import annotations

import pytest

from hawking import odyssey


def _registry() -> dict:
    return {
        name: {"confirm": True} for name in odyssey.MUTATING_VERBS
    } | {
        name: {"confirm": False} for name in odyssey.READ_ONLY_VERBS
    }


def test_mutating_and_read_only_verb_sets_are_disjoint() -> None:
    assert not set(odyssey.MUTATING_VERBS) & set(odyssey.READ_ONLY_VERBS)
    assert "admit_check" not in odyssey.MUTATING_VERBS
    assert "admit_check" not in odyssey.READ_ONLY_VERBS


def test_boundary_accepts_a_conforming_registry() -> None:
    odyssey._assert_execution_boundary(_registry())


def test_boundary_rejects_ungated_mutating_verb() -> None:
    registry = _registry()
    registry["harvest"] = {"confirm": False}
    with pytest.raises(RuntimeError, match="lacks a confirm gate"):
        odyssey._assert_execution_boundary(registry)


def test_boundary_rejects_missing_mutating_verb() -> None:
    registry = _registry()
    del registry["run"]
    with pytest.raises(RuntimeError, match="is not registered"):
        odyssey._assert_execution_boundary(registry)


def test_boundary_rejects_gated_read_only_verb() -> None:
    registry = _registry()
    registry["status"] = {"confirm": True}
    with pytest.raises(RuntimeError, match="must not gate on confirm"):
        odyssey._assert_execution_boundary(registry)


def test_boundary_rejects_registered_admit_check() -> None:
    registry = _registry()
    registry["admit_check"] = {"confirm": True}
    with pytest.raises(RuntimeError, match="admit_check must not be registered"):
        odyssey._assert_execution_boundary(registry)