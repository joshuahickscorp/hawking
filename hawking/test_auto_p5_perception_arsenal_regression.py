"""Focused regression for the P5_PERCEPTION_ARSENAL drift summary surface.

Exercises ``registry_drift_summary`` in ``hawking.perception.tools``: the
summary must report a healthy allowlisted surface as ``status="ok"`` with
``drifted=False`` and must expose the live tool/adapter surface as data. A
simulated drift (a tool name added to the live surface) must flip the summary
to ``status="drifted"`` with ``drifted=True`` and disagreeing digests, so the
observable behavior is exercised rather than merely imported.
"""
from __future__ import annotations

import unittest

from hawking.perception import tools


def test_registry_drift_components_empty_when_no_drift() -> None:
    assert tools.registry_digest_matches_exported() is True
    assert tools.registry_drift_components() == []


def test_registry_drift_components_name_exact_changed_surface(monkeypatch) -> None:
    monkeypatch.setattr(tools, "TOOL_REGISTRY_DIGEST", "0" * 64)
    components = tools.registry_drift_components()
    assert components[:3] == [
        "adapter:file.local",
        "adapter_version:1",
        "registry_version:1",
    ]
    assert components[3:] == sorted(f"tool:{name}" for name in tools.TOOL_NAMES)
    assert len(components) == 3 + len(tools.TOOL_NAMES)


def test_drift_summary_in_sync_matches_live_surface() -> None:
    summary = tools.registry_drift_summary()
    assert summary["status"] == "ok"
    assert summary["components"] == []
    assert summary["exported_digest"] == tools.TOOL_REGISTRY_DIGEST
    assert summary["live_digest"] == tools.registry_digest()
    assert summary["live_digest"] == summary["exported_digest"]


def test_drift_summary_agrees_with_drift_components() -> None:
    summary = tools.registry_drift_summary()
    components = tools.registry_drift_components()
    assert summary["components"] == components
    assert summary["status"] == ("drifted" if components else "ok")


def test_drift_summary_reports_drift_when_surface_changes(monkeypatch) -> None:
    monkeypatch.setattr(tools, "TOOL_REGISTRY_DIGEST", "0" * 64)
    summary = tools.registry_drift_summary()
    assert summary["status"] == "drifted"
    assert summary["components"]
    assert summary["exported_digest"] == "0" * 64
    assert summary["live_digest"] == tools.registry_digest()
    assert summary["live_digest"] != summary["exported_digest"]


class RegistryDriftSummaryTest(unittest.TestCase):
    def test_healthy_surface_reports_ok(self) -> None:
        summary = tools.registry_drift_summary()
        self.assertEqual(summary["status"], "ok")
        self.assertFalse(summary["drifted"])
        self.assertEqual(summary["observed_digest"], summary["expected_digest"])
        self.assertEqual(summary["observed_digest"], tools.TOOL_REGISTRY_DIGEST)
        self.assertEqual(summary["tools"], sorted(tools.TOOL_NAMES))
        self.assertEqual(summary["adapters"], sorted(tools.ADAPTERS))
        self.assertEqual(summary["adapter_version"], tools.ADAPTER_VERSION)
        self.assertEqual(summary["registry_version"], tools.TOOL_REGISTRY_VERSION)

    def test_simulated_drift_flips_summary(self) -> None:
        original = tools.TOOL_NAMES
        try:
            tools.TOOL_NAMES = tuple(original) + ("drift.probe",)
            summary = tools.registry_drift_summary()
        finally:
            tools.TOOL_NAMES = original
        self.assertEqual(summary["status"], "drifted")
        self.assertTrue(summary["drifted"])
        self.assertNotEqual(summary["observed_digest"], summary["expected_digest"])
        self.assertIn("drift.probe", summary["tools"])

    def test_summary_agrees_with_boolean_check(self) -> None:
        summary = tools.registry_drift_summary()
        self.assertEqual(
            summary["drifted"],
            not tools.registry_digest_matches_exported(),
        )


if __name__ == "__main__":
    unittest.main()
