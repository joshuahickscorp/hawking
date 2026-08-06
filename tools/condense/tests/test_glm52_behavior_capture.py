#!/usr/bin/env python3.12
"""Fail-closed tests for GLM-5.2 behavior access + capture harness."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lab.operators import glm52_behavior_access as access  # noqa: E402
from lab.operators import glm52_behavior_capture as capture  # noqa: E402
from lab.operators.glm52_common import verify_sealed  # noqa: E402


def test_architecture_map_seals_against_contract() -> None:
    arch = access.architecture_map()
    verify_sealed(arch)
    assert arch["schema"] == access.SCHEMA_ARCHITECTURE_MAP
    assert arch["geometry"]["main_hidden_layers"] == 78
    assert arch["geometry"]["hidden_size"] == 6144
    assert arch["geometry"]["vocab_size"] == 154880
    assert arch["streaming_schedule"]["W000"]["window_id"] == "W000"
    assert any(p["id"] == "generation_trajectory" for p in arch["transplant_points"])
    assert any(p["id"] == "indexshare_selection" for p in arch["transplant_points"])


def test_path_comparison_chooses_hosted_primary() -> None:
    comparison = access.path_comparison()
    assert comparison["chosen"]["primary"] == "hosted_outputs_only"
    assert comparison["chosen"]["parallel_long_pole"] == "local_streaming_inference"
    assert comparison["paths"]["local_streaming_inference"]["ready_now"] is False
    assert "3" in str(comparison["paths"]["local_streaming_inference"]["unblocks_stages"]) or 3 in comparison[
        "paths"
    ]["local_streaming_inference"]["unblocks_stages"]
    assert 1 in comparison["paths"]["hosted_outputs_only"]["unblocks_stages"]
    # Capsule NPZs must not be silently treated as resident if missing.
    capsules = comparison["capsules"]
    if capsules["metadata_json_count"] > 0 and capsules["resident_npz_count"] == 0:
        assert capsules["payload_status"] == "METADATA_ONLY_PAYLOADS_EVICTED"


def test_feasibility_receipt_seals() -> None:
    receipt = access.build_feasibility_receipt()
    verify_sealed(receipt)
    assert receipt["schema"] == access.SCHEMA_FEASIBILITY
    assert receipt["revision"] == access.IMMUTABLE_REVISION
    assert "hosted_outputs_only" in receipt["chosen_path"]["primary"]
    assert receipt["path_comparison"]["disk_floor_respected"] is True or receipt[
        "path_comparison"
    ]["free_disk_bytes"] < 0
    assert any(
        m["id"] == "M0_behavior_harness" for m in receipt["multilane_plan"]["milestones"]
    )


def test_dry_run_zero_trajectories_and_seals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Strip credentials so dry-run is deterministic.
    for name in (
        "ZHIPU_API_KEY",
        "BIGMODEL_API_KEY",
        "ZAI_API_KEY",
        "OPENROUTER_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    # Write under tmp by patching evidence paths used inside run_dry_run.
    monkeypatch.setattr(capture, "GLM52_EVIDENCE", tmp_path)
    result = capture.run_dry_run(limit=4, write_evidence=True)
    pre = result["preflight"]
    rec = result["receipt"]
    verify_sealed(pre)
    verify_sealed(rec)
    assert pre["status"] == "DRY_RUN_ONLY_NO_CREDENTIAL"
    assert rec["trajectory_count"] == 0
    assert rec["status"] == "DRY_RUN_NO_TRAJECTORIES"
    assert rec["activations_captured"] is False
    assert (tmp_path / "GLM52_BEHAVIOR_CAPTURE_PREFLIGHT.json").is_file()
    assert (tmp_path / "GLM52_BEHAVIOR_CAPTURE_DRY_RUN.json").is_file()


def test_live_fails_closed_without_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "ZHIPU_API_KEY",
        "BIGMODEL_API_KEY",
        "ZAI_API_KEY",
        "OPENROUTER_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(capture.BehaviorCaptureError, match="no hosted GLM-5.2 API credential"):
        capture.capture_live(limit=1)


def test_extract_trajectory_never_invents_topk() -> None:
    record = {
        "record_id": "t1",
        "domain": "math",
        "partition": "test",
        "prompt_sha256": "abc",
        "source": "test",
    }
    provider = {
        "id": "together",
        "base_url": "https://example.invalid",
        "first_party": False,
        "credential_env_hit": "TOGETHER_API_KEY",
    }
    response = {
        "choices": [
            {
                "message": {"content": "42"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": 1},
        "_request_model": "zai-org/GLM-5.2",
    }
    traj = capture._extract_trajectory(
        response, record=record, provider=provider, top_logprobs_requested=20
    )
    verify_sealed(traj)
    assert traj["response"]["text"] == "42"
    assert traj["top_k"]["present"] is False
    assert traj["top_k"]["steps"] is None
    assert traj["activations_captured"] is False
    assert traj["weight_byte_equality_claimed"] is False


def test_extract_trajectory_records_real_topk() -> None:
    record = {
        "record_id": "t2",
        "domain": "math",
        "partition": "test",
        "prompt_sha256": "def",
        "source": "test",
    }
    provider = {
        "id": "zhipu_bigmodel",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "first_party": True,
        "credential_env_hit": "ZHIPU_API_KEY",
    }
    response = {
        "choices": [
            {
                "message": {"content": "ok"},
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "ok",
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": "ok", "logprob": -0.1},
                                {"token": "no", "logprob": -2.0},
                            ],
                        }
                    ]
                },
            }
        ],
        "_request_model": "glm-5.2",
    }
    traj = capture._extract_trajectory(
        response, record=record, provider=provider, top_logprobs_requested=5
    )
    verify_sealed(traj)
    assert traj["top_k"]["present"] is True
    assert traj["top_k"]["steps"] is not None
    assert len(traj["top_k"]["steps"]) == 1
    assert traj["top_k"]["steps"][0]["top_logprobs"][0]["token"] == "ok"


def test_live_capture_with_mocked_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key-not-real")
    monkeypatch.setattr(capture, "default_prompt_records", lambda **_: [
        {
            "record_id": "mock.1",
            "domain": "math",
            "partition": "test",
            "prompt": "1+1?",
            "prompt_sha256": "x",
            "source": "mock",
        }
    ])

    fake_response = {
        "choices": [
            {
                "message": {"content": "2"},
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "2",
                            "logprob": 0.0,
                            "top_logprobs": [{"token": "2", "logprob": 0.0}],
                        }
                    ]
                },
            }
        ],
        "usage": {"completion_tokens": 1, "prompt_tokens": 3},
        "_request_model": "zai-org/GLM-5.2",
        "_request_url": "https://api.together.xyz/v1/chat/completions",
        "_http_status": 200,
    }

    def _fake_request(**kwargs: Any) -> dict[str, Any]:
        return dict(fake_response)

    monkeypatch.setattr(capture, "_chat_completion_request", _fake_request)
    monkeypatch.setattr(capture, "GLM52_EVIDENCE", tmp_path)

    result = capture.capture_live(limit=1, out_dir=tmp_path)
    assert result["receipt"]["trajectory_count"] == 1
    assert result["receipt"]["status"] == "LIVE_CAPTURE_OK"
    bundle = json.loads(Path(result["bundle_path"]).read_text(encoding="utf-8"))
    verify_sealed(bundle)
    assert bundle["trajectories"][0]["response"]["text"] == "2"
    assert bundle["activations_captured"] is False
