"""Tests for hcli/escalation.py and its two tool_registry.py registrations.

Covers: curated-packet construction, deterministic contract rendering (real
grok-run contract lint, exercised via GROK_DRYRUN -- zero cost, no Grok
session spent, per grok_bridge.py's own documentation), the fail-closed
credential/binary paths, and the proposal-not-fact envelope shape.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from hcli.escalation import (
    MAX_ARTIFACTS,
    MAX_SWARM_LANES,
    EscalationCredentialsError,
    EscalationError,
    SwarmBoundsError,
    curate_packet,
    escalate_to_frontier,
    launch_swarm,
    propose_swarm,
    render_lane_contract,
)
from hcli.grok_bridge import GrokContractError, GrokNotAvailable
from hcli.tool_registry import COSTLY, READ_ONLY, RESEARCH, REVERSIBLE_REPO, REVERSIBLE_RUNTIME, default_tool_registry


# ---------------------------------------------------------------------------
# packet construction
# ---------------------------------------------------------------------------

def test_curate_packet_accepts_named_artifacts_and_a_schema():
    packet = curate_packet(
        "mission kernel text",
        [{"name": "a.py", "content": "print(1)"}],
        {"type": "object", "properties": {"answer": {"type": "string"}}},
    )
    assert packet["mission_kernel"] == "mission kernel text"
    assert packet["artifacts"] == [{"name": "a.py", "content": "print(1)"}]
    assert packet["output_schema"]["type"] == "object"


def test_curate_packet_refuses_to_dump_an_archive_rather_than_truncate():
    too_many = [{"name": f"f{i}.py", "content": "x"} for i in range(MAX_ARTIFACTS + 1)]
    with pytest.raises(ValueError, match="curated"):
        curate_packet("kernel", too_many, {"type": "object"})
    with pytest.raises(ValueError, match="output_schema"):
        curate_packet("kernel", [], None)
    huge = [{"name": "big.py", "content": "x" * 100_000}]
    with pytest.raises(ValueError):
        curate_packet("kernel", huge, {"type": "object"})


# ---------------------------------------------------------------------------
# contract generation (reuses grok_bridge.validate_contract_text, not
# reimplemented) and the real grok-run linter, exercised via GROK_DRYRUN
# ---------------------------------------------------------------------------

def _git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "tracked.py").write_text("print('hi')\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


def test_render_lane_contract_has_write_and_verify_sections():
    text = render_lane_contract("fix the flaky timer", ["tracked.py"], "pytest -q tracked.py")
    assert "WRITE" in text
    assert "VERIFY" in text
    assert "tracked.py" in text


def test_propose_swarm_is_bounded_and_rejects_duplicate_lane_names():
    lanes = [{"name": f"lane{i}", "objective": "x", "write_scope": ["a"], "verify_command": "pytest a"} for i in range(MAX_SWARM_LANES + 1)]
    with pytest.raises(SwarmBoundsError):
        propose_swarm("problem", lanes)
    dupes = [
        {"name": "lane1", "objective": "x", "write_scope": ["a"], "verify_command": "pytest a"},
        {"name": "lane1", "objective": "y", "write_scope": ["b"], "verify_command": "pytest b"},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        propose_swarm("problem", dupes)


def test_propose_swarm_reuses_grok_bridge_contract_validation_not_a_reimplementation():
    # A caller-supplied contract missing VERIFY must be rejected by
    # grok_bridge.validate_contract_text itself, not a local copy of it.
    lanes = [{"name": "lane1", "contract_text": "# just an objective, no sections"}]
    with pytest.raises(GrokContractError):
        propose_swarm("problem", lanes)


@pytest.mark.skipif(not Path("/Users/scammermike/.claude-grok/bin/grok-run").is_file(), reason="grok-run not installed on this box")
def test_launch_swarm_dry_run_passes_the_real_grok_run_linter(tmp_path):
    repo = _git_repo(tmp_path)
    lanes = [{"name": "lane1", "objective": "fix the flaky timer", "write_scope": ["tracked.py"], "verify_command": "pytest -q tracked.py"}]
    result = launch_swarm(repo, "timers are flaky", lanes, mode="audit", dry_run=True)
    assert result["mode"] == "audit"
    assert len(result["lanes"]) == 1
    lane = result["lanes"][0]
    assert lane["dry_run"] is True
    assert lane["task_id"]
    # dry_run means grok-run resolved the command and exited -- no session
    # spent, no worktree created, nothing that costs money.
    assert "grok-run" in lane["command_run"][0]


# ---------------------------------------------------------------------------
# fail-closed paths
# ---------------------------------------------------------------------------

def test_escalate_fails_closed_with_no_credential_and_never_calls_the_network(monkeypatch):
    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("no network call should be attempted without a credential")

    monkeypatch.setattr("hcli.escalation.urllib.request.urlopen", _must_not_be_called)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(EscalationCredentialsError):
        escalate_to_frontier(
            "is this design sound?",
            "mission kernel",
            [],
            {"type": "object"},
        )


def test_launch_swarm_fails_closed_when_grok_run_is_not_available(tmp_path, monkeypatch):
    monkeypatch.setenv("GROK_RUN", str(tmp_path / "no-such-binary"))
    lanes = [{"name": "lane1", "objective": "fix it", "write_scope": ["a"], "verify_command": "pytest a"}]
    with pytest.raises(GrokNotAvailable):
        launch_swarm(tmp_path, "problem", lanes, mode="audit", dry_run=True)


# ---------------------------------------------------------------------------
# proposal-not-fact envelope
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_escalate_wraps_a_successful_call_as_an_unverified_hypothesis(monkeypatch):
    captured = {}

    def _fake_urlopen(request, timeout=None):
        captured["headers"] = dict(request.header_items())
        captured["url"] = request.full_url
        return _FakeResponse({"content": [{"type": "text", "text": "the fix is safe"}]})

    monkeypatch.setattr("hcli.escalation.urllib.request.urlopen", _fake_urlopen)
    result = escalate_to_frontier(
        "is this design sound?",
        "mission kernel",
        [{"name": "diff.patch", "content": "+ line"}],
        {"type": "object", "properties": {"verdict": {"type": "string"}}},
        api_key="test-key-not-real",
    )
    envelope = result["envelope"]
    assert envelope["verdict"] == "UNVERIFIED"
    assert envelope["verified_facts"] == []
    assert envelope["hypotheses"][0]["claim"] == "the fix is safe"
    assert "cloud_frontier:anthropic" in envelope["hypotheses"][0]["source"]
    assert result["provenance"]["credential_values_recorded"] is False
    # the key must reach the wire (that's the whole point) but never come back
    assert captured["headers"].get("X-api-key") == "test-key-not-real"
    assert "test-key-not-real" not in json.dumps(result)


def test_escalate_raises_rather_than_inventing_an_answer_on_a_bad_response(monkeypatch):
    def _fake_urlopen(request, timeout=None):
        return _FakeResponse({"content": []})

    monkeypatch.setattr("hcli.escalation.urllib.request.urlopen", _fake_urlopen)
    with pytest.raises(EscalationError):
        escalate_to_frontier("q", "kernel", [], {"type": "object"}, api_key="k")


# ---------------------------------------------------------------------------
# tool_registry.py wiring
# ---------------------------------------------------------------------------

def _costly_registry(workspace: Path):
    return default_tool_registry(
        workspace,
        repo_root=workspace,
        permissions={READ_ONLY, RESEARCH, REVERSIBLE_REPO, REVERSIBLE_RUNTIME, COSTLY},
    )


def test_frontier_escalate_tool_requires_confirm_and_fails_closed(tmp_path, monkeypatch):
    registry = _costly_registry(tmp_path)
    names = {item["name"] for item in registry.discover()}
    assert "frontier.escalate" in names
    assert "grok.swarm.propose" in names
    assert "grok.swarm.launch" in names

    unconfirmed = registry.invoke("frontier.escalate", {"confirm": False, "question": "q", "mission_kernel": "k"})
    assert unconfirmed.ok is False
    assert unconfirmed.failure_class == "PermissionError"

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = registry.invoke(
        "frontier.escalate",
        {"confirm": True, "question": "q", "mission_kernel": "k"},
    )
    assert result.ok is False
    assert result.failure_class == "EscalationCredentialsError"


def test_grok_swarm_propose_tool_is_read_only_and_renders_contracts(tmp_path):
    registry = default_tool_registry(tmp_path, repo_root=tmp_path)  # no COSTLY needed
    result = registry.invoke(
        "grok.swarm.propose",
        {
            "problem_statement": "flaky timers",
            "lanes": [{"name": "lane1", "objective": "fix it", "write_scope": ["a"], "verify_command": "pytest a"}],
        },
    )
    assert result.ok is True
    assert result.value["lane_count"] == 1


def test_grok_swarm_launch_tool_requires_confirm_and_dry_runs_end_to_end(tmp_path):
    repo = _git_repo(tmp_path)
    registry = _costly_registry(repo)
    unconfirmed = registry.invoke("grok.swarm.launch", {"confirm": False, "problem_statement": "p", "lanes": []})
    assert unconfirmed.ok is False
    assert unconfirmed.failure_class == "PermissionError"

    if not Path("/Users/scammermike/.claude-grok/bin/grok-run").is_file() and not os.environ.get("GROK_RUN"):
        pytest.skip("grok-run not installed on this box")
    result = registry.invoke(
        "grok.swarm.launch",
        {
            "confirm": True,
            "problem_statement": "timers are flaky",
            "lanes": [{"name": "lane1", "objective": "fix the flaky timer", "write_scope": ["tracked.py"], "verify_command": "pytest -q tracked.py"}],
            "mode": "audit",
            "dry_run": True,
        },
    )
    assert result.ok is True
    assert result.value["lanes"][0]["dry_run"] is True
