"""Contracts for the privacy-preserving DeepSeek-V4 HCLI live-suite receipt."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import deepseek_v4_gravity as gravity
from lab.receipts import verify


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _diagnostic_artifact(tmp_path: Path) -> tuple[Path, dict]:
    artifact = tmp_path / "diagnostic.gravity"
    artifact.mkdir()
    manifest = gravity.seal(
        {
            "schema": gravity.ARTIFACT_SCHEMA,
            "status": "DIAGNOSTIC_SEALED_LOADABLE_BY_V4_NUMPY_ADAPTER",
            "diagnostic_scope": {"not_full_model": True, "selected_layer": 4},
            "runtime_adapter": {"id": "deepseek_v4_layer4_diagnostic", "device": "cpu"},
        }
    )
    _write(artifact / "manifest.json", manifest)
    return artifact, manifest


def _endpoint(manifest: dict, *, artifact_seal: str | None = None) -> dict:
    return {
        "schema": "hcli.command.v1",
        "command": "capabilities",
        "result": {
            "backend": {"capabilities": {"agent_kernel": True, "fleet": True}},
            "runtime": {
                "context": {
                    "artifact_seal_sha256": artifact_seal or manifest["seal_sha256"],
                    "arch": "deepseek_v4_layer4_diagnostic",
                    "capability_status": "diagnostic_cpu_only_not_tg_eligible",
                    "ctx_len_effective": 128,
                    "max_output_tokens": 4,
                    "metal_dispatches": 0,
                    "status": "available",
                },
                "health": {"ready": True, "status": "ready"},
            },
        },
    }


def test_hcli_live_suite_hashes_prompts_and_preserves_live_statuses(tmp_path: Path) -> None:
    artifact, manifest = _diagnostic_artifact(tmp_path)
    endpoint = tmp_path / "capabilities.json"
    normal = tmp_path / "normal-turn.json"
    agent = tmp_path / "agent.json"
    out = tmp_path / "live-suite.json"
    _write(endpoint, _endpoint(manifest))
    normal_prompt = "normal prompt must not appear in the aggregate"
    agent_goal = "agent goal must not appear in the aggregate"
    _write(
        normal,
        {
            "schema": "hcli.command.v1",
            "command": "run",
            "status": "completed",
            "result": {
                "prompt": normal_prompt,
                "reason": f"request {normal_prompt} was rejected by the endpoint",
                "turn": {"completion": "also not copied", "output_tokens": 1},
            },
        },
    )
    _write(
        agent,
        {
            "schema": "hide.headless.audit.v1",
            "status": "step_limit",
            "goal": {"text": agent_goal, "blake3": "a" * 64},
            "failure": "policy denied: effectful dispatch refused",
            "runtime": {
                "context_before": {
                    "artifact_seal_sha256": manifest["seal_sha256"],
                    "capability_status": "diagnostic_cpu_only_not_tg_eligible",
                    "status": "available",
                }
            },
        },
    )

    report = gravity.hcli_live_suite_receipt(artifact, endpoint, [normal, agent], out)

    assert verify(report, label="HCLI live suite") == report
    assert report["artifact"]["seal_sha256"] == manifest["seal_sha256"]
    assert report["endpoint"]["runtime_context"]["capability_status"] == "diagnostic_cpu_only_not_tg_eligible"
    assert report["evidence"][0]["statuses"] == [{"location": "status", "value": "completed"}]
    assert report["evidence"][0]["errors"][0]["prompt_material_redacted"] is True
    assert report["evidence"][1]["statuses"] == [{"location": "status", "value": "step_limit"}]
    assert report["evidence"][1]["errors"][0]["text"] == "policy denied: effectful dispatch refused"
    assert report["evidence"][1]["prompt_hashes"][0]["declared_blake3"] == "a" * 64
    encoded = json.dumps(report)
    assert normal_prompt not in encoded
    assert agent_goal not in encoded
    assert "also not copied" not in encoded
    assert json.loads(out.read_text(encoding="utf-8"))["seal_sha256"] == report["seal_sha256"]


def test_hcli_live_suite_rejects_endpoint_artifact_mismatch(tmp_path: Path) -> None:
    artifact, manifest = _diagnostic_artifact(tmp_path)
    endpoint = tmp_path / "capabilities.json"
    evidence = tmp_path / "run.json"
    _write(endpoint, _endpoint(manifest, artifact_seal="0" * 64))
    _write(evidence, {"schema": "hcli.command.v1", "command": "run", "result": {}})

    with pytest.raises(gravity.DeepSeekV4GravityError, match="does not match"):
        gravity.hcli_live_suite_receipt(artifact, endpoint, [evidence], tmp_path / "out.json")


def test_hcli_live_suite_rejects_mismatched_identity_inside_evidence(tmp_path: Path) -> None:
    artifact, manifest = _diagnostic_artifact(tmp_path)
    endpoint = tmp_path / "capabilities.json"
    evidence = tmp_path / "agent.json"
    _write(endpoint, _endpoint(manifest))
    _write(
        evidence,
        {
            "schema": "hide.headless.audit.v1",
            "status": "step_limit",
            "runtime": {"context_before": {"artifact_seal_sha256": "f" * 64}},
        },
    )

    with pytest.raises(gravity.DeepSeekV4GravityError, match="artifact identity mismatch"):
        gravity.hcli_live_suite_receipt(artifact, endpoint, [evidence], tmp_path / "out.json")
