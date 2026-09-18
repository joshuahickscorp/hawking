"""Regression guards for the Hawking-native H-Web main-chat boundary."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from hawking.chat_tools import run_local_tool
from hawking.serve import (
    HWEB_READ_ONLY_TOOL_NAMES,
    READ_ONLY_RESEARCH_TOOL_NAMES,
    WorkerRequestMode,
    worker_tool_allowlist,
)
from hawking.web_artifacts import admit_bytes


def test_launcher_refuses_a_healthy_daemon_without_current_tool_contract(monkeypatch):
    """A healthy old daemon cannot silently serve a new H-Web tool surface."""
    import hawking.web_launcher as launcher

    health = {
        "owner": {"daemon": "hawkingd", "pid": 42},
        "endpoints": sorted(launcher.REQUIRED_ENDPOINTS),
    }
    models = {"hawking_catalog": {"search_available": True}}
    release = {"current": {
        "release_id": "web-test", "contract": "hawking.web.contract.v1",
    }}

    def responses(url, **_kwargs):
        if url.endswith("/health"):
            return dict(health)
        if url.endswith("/v1/models"):
            return dict(models)
        return dict(release)

    monkeypatch.setattr(launcher, "_json_get", responses)
    with pytest.raises(launcher.WebLaunchError, match="capability projection"):
        launcher._contract_snapshot("http://127.0.0.1:8014")
    health["hweb_capability_projection_revision"] = (
        launcher.HWEB_CAPABILITY_PROJECTION_REVISION
    )
    assert launcher._contract_snapshot("http://127.0.0.1:8014")["pid"] == 42


def test_hweb_read_only_projection_is_richer_without_widening_legacy_workers():
    legacy = worker_tool_allowlist(WorkerRequestMode.READ_ONLY_RESEARCH)
    assert legacy == READ_ONLY_RESEARCH_TOOL_NAMES
    assert {
        "workspace.identity", "source.owner", "git.diff", "runtime.status",
        "artifact.read", "memory.search", "goals.list",
    } <= HWEB_READ_ONLY_TOOL_NAMES
    assert not {
        "repo.edit", "tests.run", "shell.exec", "browser.click",
        "macos.press", "processes.kill",
    } & HWEB_READ_ONLY_TOOL_NAMES


def test_hweb_local_artifact_and_workspace_doors_are_canonical():
    with tempfile.TemporaryDirectory() as tmp:
        ref = admit_bytes(
            tmp,
            filename="native-chat.md",
            payload=b"# Hawking\nunique-main-chat-fact\n",
            source_type="pasted_text",
            session_id="web-test",
        )
        artifact = run_local_tool(
            "artifact.read", {"artifact_id": ref["artifact_id"]}, workspace=tmp
        )
        memory = run_local_tool(
            "memory.search",
            {"query": "unique-main-chat-fact", "session_id": "web-test"},
            workspace=tmp,
        )
        identity = run_local_tool("workspace.identity", {}, workspace=tmp)

    assert artifact.ok, artifact.error
    assert artifact.value["original_preserved"] is True
    assert "unique-main-chat-fact" in artifact.value["text"]
    assert memory.ok, memory.error
    assert memory.value["hits"][0]["artifact_id"] == ref["artifact_id"]
    assert identity.ok, identity.error
    assert identity.value["workspace_id"].startswith("workspace-")


def test_goal_projection_can_be_explicitly_scoped():
    from hawking.goal_surface import create_goal

    with tempfile.TemporaryDirectory() as tmp:
        goal = create_goal(Path(tmp), objective="only this session")
        observed = run_local_tool(
            "goals.list",
            {"limit": 8, "goal_ids": [goal["goal_id"]]},
            workspace=tmp,
        )
        empty = run_local_tool(
            "goals.list", {"limit": 8, "goal_ids": []}, workspace=tmp
        )

    assert observed.ok, observed.error
    assert [row["goal_id"] for row in observed.value["goals"]] == [goal["goal_id"]]
    assert empty.ok, empty.error
    assert empty.value["goals"] == []
