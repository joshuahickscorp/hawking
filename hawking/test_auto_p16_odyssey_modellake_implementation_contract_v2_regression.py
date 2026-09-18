"""Focused regression for the Odyssey execution-boundary audit.

Exercises ``hawking.odyssey.audit_execution_boundary`` against synthetic
registries so the changed production behavior is observable without touching
persistent Odyssey state, spawning a subprocess, or reclaiming disk.
"""
from __future__ import annotations

from hawking.odyssey import (
    MUTATING_VERBS,
    UNREGISTERED_VERBS,
    audit_execution_boundary,
)


def _gated(*args, confirm=False, **kwargs):
    return None


def _ungated(*args, **kwargs):
    return None


def _full_gated_registry():
    return {name: _gated for name in MUTATING_VERBS}


def test_full_gated_registry_is_ok():
    report = audit_execution_boundary(_full_gated_registry())
    assert report["ok"] is True
    assert report["missing"] == []
    assert report["ungated"] == []
    assert report["leaked"] == []


def test_missing_mutating_verb_is_reported():
    registry = _full_gated_registry()
    registry.pop("harvest")
    report = audit_execution_boundary(registry)
    assert report["ok"] is False
    assert report["missing"] == ["harvest"]


def test_ungated_mutating_verb_is_reported():
    registry = _full_gated_registry()
    registry["run"] = _ungated
    report = audit_execution_boundary(registry)
    assert report["ok"] is False
    assert report["ungated"] == ["run"]


def test_unregistered_verb_leak_is_reported():
    registry = _full_gated_registry()
    registry["admit_check"] = _gated
    report = audit_execution_boundary(registry)
    assert report["ok"] is False
    assert report["leaked"] == ["admit_check"]


def test_audit_does_not_mutate_registry():
    registry = _full_gated_registry()
    before = dict(registry)
    audit_execution_boundary(registry)
    assert registry == before


def test_unregistered_verbs_are_not_mutating_verbs():
    assert UNREGISTERED_VERBS.isdisjoint(MUTATING_VERBS)