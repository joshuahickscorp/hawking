#!/usr/bin/env python3.12
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_runtime_speed_gate as gate  # noqa: E402


def _raw(index: pathlib.Path) -> dict:
    samples = [2.0, 1.0, 3.0, 2.0]
    return {
        "schema": "hawking.gravity.glm_base_tps.v1",
        "scoreboard": "BASE_TRUE_TPS",
        "verify_hash": True,
        "artifact": {
            "index": index.name,
            "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        },
        "measurements": [
            {
                "context_tokens": 8,
                "decode_tokens": len(samples),
                "decode_ms_per_token_all": samples,
                "decode_ms_per_token_median": 2.0,
                "base_true_decode_tps": len(samples) * 1000.0 / sum(samples),
            }
        ],
    }


def test_milestone_ladder_has_no_artificial_ceiling():
    assert gate.achieved_milestones(20.0) == ["TG20"]
    assert gate.achieved_milestones(2.0) == ["TG20", "TG10", "TG5", "TG2"]
    assert gate.achieved_milestones(0.5) == ["TG20", "TG10", "TG5", "TG2", "TG1"]


def test_receipt_binds_exact_index_and_reconciles_tps(tmp_path):
    index = tmp_path / "model.activation_aware.index.json"
    index.write_text(json.dumps({"schema": "fixture"}))
    receipt = gate.validate_measurement(
        _raw(index),
        expected_index=index,
        expected_contexts=[8],
        expected_decode=4,
        target_ms=2.0,
    )
    assert receipt["measurement_verdict"] == "VALID"
    assert receipt["target_verdict"] == "PASS"
    assert receipt["contexts"][0]["achieved_milestones"][-1] == "TG2"


def test_receipt_refuses_wrong_index_or_unreconciled_rate(tmp_path):
    index = tmp_path / "model.activation_aware.index.json"
    index.write_text("{}")
    wrong_hash = _raw(index)
    wrong_hash["artifact"]["index_sha256"] = "0" * 64
    with pytest.raises(gate.RuntimeSpeedError, match="hash"):
        gate.validate_measurement(
            wrong_hash,
            expected_index=index,
            expected_contexts=[8],
            expected_decode=4,
            target_ms=2.0,
        )

    wrong_rate = _raw(index)
    wrong_rate["measurements"][0]["base_true_decode_tps"] = 1.0
    with pytest.raises(gate.RuntimeSpeedError, match="reconcile"):
        gate.validate_measurement(
            wrong_rate,
            expected_index=index,
            expected_contexts=[8],
            expected_decode=4,
            target_ms=2.0,
        )


def test_sustained_command_requests_curve_and_progress(tmp_path):
    command = gate.benchmark_command(
        tmp_path,
        [2048, 8192, 32768],
        80,
        tmp_path / "raw.json",
        verify_hash=True,
        sustained=True,
    )
    assert "--token-curve" in command
    assert "--progress" in command
    assert command.count("--context") == 3
