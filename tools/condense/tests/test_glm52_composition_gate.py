#!/usr/bin/env python3.12
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

CONDENSE = pathlib.Path(__file__).resolve().parents[1]
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

import glm52_composition_gate as gate  # noqa: E402


def _evidence(tmp_path):
    shards = [f"model-{index:05d}-of-00282.aap" for index in range(1, 283)]
    index = {
        "schema": "hawking.activation_aware.model_index.v1",
        "architecture": {"num_hidden_layers": 78},
        "shard_count": 282,
        "shards": shards,
        "coverage": {"verdict": "COMPLETE"},
    }
    index_path = tmp_path / gate.INDEX_NAME
    index_path.write_text(json.dumps(index))
    digest = hashlib.sha256(index_path.read_bytes()).hexdigest()
    assembly = {
        "schema": "hawking.glm52.activation_aware_assembly_receipt.v1",
        "action": "ASSEMBLED",
        "coverage": "COMPLETE",
        "shards_hashed": 282,
        "model_bytes_copied": 0,
        "index_sha256": digest,
    }
    capability = {
        "schema": "hawking.substrate.capability_gate_run.v1",
        "artifact_index_sha256": digest,
        "artifact_verification": True,
        "capability_verdict": "APPROVED",
        "gates": [
            {"gate": "G_math", "status": "PASS"},
            {"gate": "G_live", "status": "PASS"},
        ],
    }
    parity = {
        "schema": "hawking.glm52.runtime_parity_gate.v1",
        "artifact_index_sha256": digest,
        "suite": "capability",
        "verdict": "PASS",
        "cases": [
            {
                "name": name,
                "comparison": {
                    "pass": True,
                    "argmax_exact": True,
                    "ordered_top5_exact": True,
                },
            }
            for name in gate.EXPECTED_CASES
        ],
    }
    return assembly, capability, parity


def test_full_model_receipts_admit_composition(tmp_path):
    assembly, capability, parity = _evidence(tmp_path)
    result = gate.evaluate(
        artifact=tmp_path,
        assembly=assembly,
        capability=capability,
        parity=parity,
    )
    assert result["verdict"] == "PASS_FULL_MODEL_COMPOSITION"
    assert all(result["checks"].values())
    assert result["claims"]["teacher_hidden_state_parity"] is False


def test_any_hash_or_discrete_failure_refuses(tmp_path):
    assembly, capability, parity = _evidence(tmp_path)
    capability["artifact_index_sha256"] = "0" * 64
    parity["cases"][0]["comparison"]["argmax_exact"] = False
    result = gate.evaluate(
        artifact=tmp_path,
        assembly=assembly,
        capability=capability,
        parity=parity,
    )
    assert result["verdict"] == "REFUSED"
    assert not result["checks"]["capability_receipt"]
    assert not result["checks"]["runtime_agreement_receipt"]
