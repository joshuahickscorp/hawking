"""Deterministic coverage for the Phase E native desktop WorldState owner."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from hawking.macos_world import MacOSWorld, MacOSWorldError
from hawking.tool_registry import EXTERNAL_WRITE, READ_ONLY, default_tool_registry


class _FixtureRuntime:
    def __init__(self) -> None:
        self.turn = 0

    def health(self):
        return {"schema": "hawking.macos.helper.v1", "health": "READY", "helper_version": "fixture"}

    def call(self, operation: str):
        assert operation == "observe"
        self.turn += 1
        return {
            "schema": "hawking.macos.helper.v1",
            "platform": "macos",
            "applications": [{"pid": 91, "name": "Fixture", "active": self.turn > 1}],
            "windows": [{"window_id": 12, "owner_pid": 91, "title": "Fixture"}],
            "focused_app": {"pid": 91, "name": "Fixture"},
            "focused_element": {"role": "AXApplication", "title": "Fixture"},
            "accessibility": {"trusted": True},
            "input": {"owner": None, "state": "NO_INPUT_CAPABILITY"},
        }


class _ActionFixtureRuntime:
    def __init__(self) -> None:
        self.value = "idle"
        self.focused = "Fixture"

    def health(self):
        return {"schema": "hawking.macos.helper.v1", "health": "READY", "helper_version": "fixture"}

    def call(self, operation: str, arguments=None):
        if operation == "observe":
            return {
                "schema": "hawking.macos.helper.v1",
                "platform": "macos",
                "applications": [
                    {"pid": 91, "name": "Fixture", "bundle_id": "com.example.fixture", "active": True}
                ],
                "windows": [{"window_id": 12, "owner_pid": 91, "title": "Fixture"}],
                "focused_app": {"pid": 91, "name": self.focused, "bundle_id": "com.example.fixture"},
                "focused_element": {"role": "AXApplication", "title": self.focused},
                "accessibility": {"trusted": True},
                "input": {"owner": None, "state": "NO_UNSCOPED_INPUT"},
            }
        if operation == "find":
            target = dict((arguments or {}).get("target") or {})
            if target.get("role") == "AXButton":
                matches = [{
                    "path": "0.1", "role": "AXButton", "title": "Toggle",
                    "identifier": "hawking.fixture.toggle", "enabled": True,
                }]
            elif target.get("identifier") == "hawking.fixture.state" and target.get("value") == self.value:
                matches = [{
                    "path": "0.0", "role": "AXStaticText", "value": self.value,
                    "identifier": "hawking.fixture.state",
                }]
            else:
                matches = []
            return {"pid": 91, "trusted": True, "matches": matches, "match_count": len(matches)}
        if operation == "press":
            self.value = "done"
            return {"pressed": True, "match_count": 1, "trusted": True}
        raise AssertionError(operation)


def test_macos_world_commits_revisioned_native_observations(tmp_path: Path):
    runtime = _FixtureRuntime()
    world = MacOSWorld(tmp_path, runtime=runtime)

    first = world.invoke("observe", {})
    second = world.invoke("observe", {})
    changed = MacOSWorld(tmp_path, runtime=runtime).invoke(
        "world.changed", {"since_revision": first["world_revision"]}
    )

    assert first["world_revision"] == 1
    assert second["world_revision"] == 2
    assert second["freshness"] == "CURRENT"
    assert second["snapshot"]["input"]["state"] == "NO_INPUT_CAPABILITY"
    assert changed["gap"] is False
    assert changed["events"][-1]["changes"]["applications"]


def test_macos_world_is_read_only_and_registry_has_no_input_escape(tmp_path: Path):
    world = MacOSWorld(tmp_path, runtime=_FixtureRuntime())
    with pytest.raises(MacOSWorldError, match="no 'click' operation"):
        world.invoke("click", {})

    registry = default_tool_registry(tmp_path, permissions={READ_ONLY})
    assert registry.get("macos.health") is not None
    assert registry.get("macos.observe") is not None
    assert registry.get("macos.changed") is not None
    assert registry.get("macos.find") is not None
    assert registry.get("macos.verify") is not None
    assert registry.get("macos.lease.acquire") is not None
    assert registry.get("macos.press") is not None
    assert registry.get("macos.click") is None


def _authority():
    return {
        "schema": "hawking.goal.macos_authority.v1",
        "goal_id": "GOAL-ACTION",
        "workunit_id": "WU-ACTION",
        "worker_id": "hawking-worker-fixture",
        "observe": True,
        "actions": True,
        "applications": ["Fixture"],
        "bundle_ids": [],
        "lease_ttl_s": 30.0,
        "preemptible": True,
    }


def test_semantic_action_requires_lease_and_verifies_revision_delta(tmp_path: Path):
    world = MacOSWorld(tmp_path, runtime=_ActionFixtureRuntime())
    authority = _authority()
    lease = world.invoke(
        "lease.acquire", {"target": {"application": "Fixture"}, "reason": "deterministic fixture"},
        authority=authority,
    )
    assert lease["state"] == "ACTIVE"
    resolved = world.invoke(
        "find", {"application": "Fixture", "target": {"role": "AXButton", "title": "Toggle"}},
        authority=authority,
    )
    assert resolved["verified"] is True
    result = world.invoke(
        "press",
        {
            "lease_id": lease["lease_id"],
            "application": "Fixture",
            "target": {"role": "AXButton", "title": "Toggle"},
            "verify": {
                "role": "AXStaticText",
                "identifier": "hawking.fixture.state",
                "value": "done",
            },
        },
        authority=authority,
    )
    assert result["verified"] is True
    assert result["revision_delta"] >= 1
    assert result["lease_state"] == "RELEASED"
    assert Path(result["receipt"]).is_file()
    action_receipt = json.loads(Path(result["receipt"]).read_text())
    assert action_receipt["operation"] == "press"
    assert action_receipt["outcome"] == "VERIFIED"
    assert action_receipt["details"]["verified"] is True
    assert action_receipt["details"]["revision_delta"] >= 1
    assert list((tmp_path / "receipts" / "future" / "macos").glob("*.json"))

    with pytest.raises(MacOSWorldError, match="active Hawking action lease"):
        world.invoke(
            "press",
            {
                "lease_id": lease["lease_id"],
                "application": "Fixture",
                "target": {"role": "AXButton", "title": "Toggle"},
                "verify": {"role": "AXStaticText", "value": "done"},
            },
            authority=authority,
        )


def test_user_focus_preempts_lease_and_unscoped_actions_are_refused(tmp_path: Path):
    runtime = _ActionFixtureRuntime()
    world = MacOSWorld(tmp_path, runtime=runtime)
    authority = _authority()
    with pytest.raises(MacOSWorldError, match="semantic macOS actions are not admitted"):
        world.invoke("lease.acquire", {"target": {"application": "Fixture"}}, authority={**authority, "actions": False})
    lease = world.invoke("lease.acquire", {"target": {"application": "Fixture"}}, authority=authority)
    runtime.focused = "Other"
    with pytest.raises(MacOSWorldError, match="preempted"):
        world.invoke(
            "press",
            {
                "lease_id": lease["lease_id"], "application": "Fixture",
                "target": {"role": "AXButton", "title": "Toggle"},
                "verify": {"role": "AXStaticText", "value": "done"},
            },
            authority=authority,
        )
    assert world.leases.load()["state"] == "PREEMPTED"

    registry = default_tool_registry(
        tmp_path,
        permissions={READ_ONLY, EXTERNAL_WRITE},
        authority={},
    )
    refused = registry.invoke("macos.lease.acquire", {"target": {"application": "Fixture"}})
    assert refused.ok is False
    assert refused.failure_class in {"PERMISSION_DENIED", "AUTHORITY_REQUIRED", "MacOSWorldError"}


def test_macos_action_receipt_revision_delta_helper_is_durable():
    from hawking.macos_world import action_receipt_revision_delta
    assert action_receipt_revision_delta({"details": {"revision_delta": 1}}) == 1
    assert action_receipt_revision_delta({"details": {"revision_delta": -2}}) == 0
