"""Focused regression for the P5 perception registry drift report.

Exercises ``registry_drift_report`` in ``hawking.perception.tools``: the
report must be a real, per-component observation of the allowlisted tool
surface, not a constant. The test mutates the live surface (adds a tool
name) and asserts the report names the drift, then restores the surface and
asserts the report collapses back to a clean, non-drifted record.
"""
from __future__ import annotations

from hawking.perception import tools


def test_registry_drift_report_is_clean_when_surface_matches() -> None:
    report = tools.registry_drift_report()
    assert report["drifted"] is False
    assert report["exported_digest"] == tools.TOOL_REGISTRY_DIGEST
    assert report["live_digest"] == tools.TOOL_REGISTRY_DIGEST
    assert report["changed_adapters"] == []
    assert report["changed_adapter_version"] == ""
    assert report["changed_registry_version"] == ""
    assert report["changed_tools"] == []


def test_registry_drift_report_names_changed_tools() -> None:
    original = tools.TOOL_NAMES
    tools.TOOL_NAMES = (*original, "drift.probe")
    try:
        report = tools.registry_drift_report()
        assert report["drifted"] is True
        assert report["live_digest"] != report["exported_digest"]
        assert "drift.probe" in report["changed_tools"]
    finally:
        tools.TOOL_NAMES = original

    restored = tools.registry_drift_report()
    assert restored["drifted"] is False
    assert restored["changed_tools"] == []
