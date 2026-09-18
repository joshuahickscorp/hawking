"""Focused regression for the P5 perception execution audit V3 tranche.

Exercises the newly added ``registry_drift_components`` behavior in
``hawking.perception.tools``: the function must return an empty list when the
live allowlisted tool surface still matches the digest bound at import time,
and must name the exact changed components when the surface is mutated.
"""
from __future__ import annotations

import hawking.perception.tools as tools


def test_registry_drift_components_empty_when_surface_matches() -> None:
    assert tools.registry_digest_matches_exported() is True
    assert tools.registry_drift_components() == []


def test_registry_drift_components_names_changed_tool(monkeypatch) -> None:
    monkeypatch.setattr(tools, "TOOL_NAMES", (*tools.TOOL_NAMES, "file.extra"))
    components = tools.registry_drift_components()
    assert "tool:file.extra" in components
    assert components == sorted(components)


def test_registry_drift_components_names_changed_adapter_version(monkeypatch) -> None:
    monkeypatch.setattr(tools, "ADAPTER_VERSION", "2")
    components = tools.registry_drift_components()
    assert "adapter_version:2" in components
    assert components == sorted(components)