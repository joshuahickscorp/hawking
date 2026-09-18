"""Focused regression for the perception registry drift report.

Exercises ``registry_drift`` in ``hawking.perception.tools``: the report must
be derived from the live allowlisted tool surface, so it agrees with the
import-time digest on an unchanged registry and exposes the disagreeing
observed/expected digests when the surface changes.
"""
from __future__ import annotations

from hawking.perception import tools


def test_registry_drift_matches_unchanged_surface() -> None:
    report = tools.registry_drift()
    assert report["match"] is True
    assert report["observed"] == tools.TOOL_REGISTRY_DIGEST
    assert report["expected"] == tools.TOOL_REGISTRY_DIGEST
    assert report["observed"] == tools.registry_digest()


def test_registry_drift_reports_changed_surface(monkeypatch) -> None:
    monkeypatch.setattr(tools, "TOOL_REGISTRY_VERSION", "2")
    report = tools.registry_drift()
    assert report["match"] is False
    assert report["observed"] != report["expected"]
    assert report["expected"] == tools.TOOL_REGISTRY_DIGEST
    assert report["observed"] == tools.registry_digest()