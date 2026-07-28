#!/usr/bin/env python3.12
"""Derive G_cascade from hash-identical whole-model execution receipts.

This gate does not run the model. It admits full-model composition only when:

* assembly proves complete official coverage of all 282 shards / 78 layers;
* G_math and G_live passed with verified artifact reads; and
* Rust/Metal agreed with the numpy artifact oracle on the same three prompts.

It is intentionally named composition rather than reusing the old functional
student cascade, which evaluated a different representation mechanism.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


INDEX_NAME = "model.activation_aware.index.json"
ASSEMBLY_NAME = "ACTIVATION_AWARE_ASSEMBLY_RECEIPT.json"
EXPECTED_LAYERS = 78
EXPECTED_SHARDS = 282
EXPECTED_CASES = ("math", "capital", "python")


class CompositionGateError(RuntimeError):
    """Input evidence cannot support a whole-model composition verdict."""


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CompositionGateError(f"{path}: unreadable JSON: {error}") from error
    if not isinstance(value, dict):
        raise CompositionGateError(f"{path}: expected a JSON object")
    return value


def evaluate(
    *,
    artifact: Path,
    assembly: dict[str, Any],
    capability: dict[str, Any],
    parity: dict[str, Any],
) -> dict[str, Any]:
    artifact = Path(artifact).expanduser().resolve()
    index_path = artifact / INDEX_NAME
    index = _read(index_path)
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()

    architecture = index.get("architecture") or {}
    coverage = index.get("coverage") or {}
    checks = {
        "activation_aware_index_schema": (
            index.get("schema") == "hawking.activation_aware.model_index.v1"
        ),
        "complete_layer_depth": architecture.get("num_hidden_layers") == EXPECTED_LAYERS,
        "complete_shard_set": (
            index.get("shard_count") == EXPECTED_SHARDS
            and len(index.get("shards") or []) == EXPECTED_SHARDS
        ),
        "complete_official_tensor_coverage": coverage.get("verdict") == "COMPLETE",
        "assembly_receipt": (
            assembly.get("schema")
            == "hawking.glm52.activation_aware_assembly_receipt.v1"
            and assembly.get("action") == "ASSEMBLED"
            and assembly.get("coverage") == "COMPLETE"
            and assembly.get("shards_hashed") == EXPECTED_SHARDS
            and assembly.get("model_bytes_copied") == 0
            and assembly.get("index_sha256") == index_sha
        ),
        "capability_receipt": (
            capability.get("schema") == "hawking.substrate.capability_gate_run.v1"
            and capability.get("artifact_index_sha256") == index_sha
            and capability.get("artifact_verification") is True
            and capability.get("capability_verdict") == "APPROVED"
            and {
                row.get("gate"): row.get("status")
                for row in capability.get("gates") or []
            }
            == {"G_math": "PASS", "G_live": "PASS"}
        ),
        "runtime_agreement_receipt": (
            parity.get("schema") == "hawking.glm52.runtime_parity_gate.v1"
            and parity.get("artifact_index_sha256") == index_sha
            and parity.get("suite") == "capability"
            and parity.get("verdict") == "PASS"
            and tuple(row.get("name") for row in parity.get("cases") or [])
            == EXPECTED_CASES
            and all(
                row.get("comparison", {}).get("pass") is True
                and row.get("comparison", {}).get("argmax_exact") is True
                and row.get("comparison", {}).get("ordered_top5_exact") is True
                for row in parity.get("cases") or []
            )
        ),
    }
    passed = all(checks.values())
    return {
        "schema": "hawking.glm52.whole_model_composition_gate.v1",
        "artifact_index": INDEX_NAME,
        "artifact_index_sha256": index_sha,
        "scope": {
            "layers": architecture.get("num_hidden_layers"),
            "shards": index.get("shard_count"),
            "prompt_cases": list(EXPECTED_CASES),
            "whole_model_execution": True,
            "per_layer_proxy": False,
        },
        "checks": checks,
        "verdict": "PASS_FULL_MODEL_COMPOSITION" if passed else "REFUSED",
        "claims": {
            "representation_composes_end_to_end_on_capability_suite": passed,
            "teacher_hidden_state_parity": False,
            "support_halo": False,
            "long_context": False,
            "quality_beyond_capability_suite": False,
            "speed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--assembly", type=Path)
    parser.add_argument("--capability", required=True, type=Path)
    parser.add_argument("--runtime-parity", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    artifact = args.artifact.expanduser().resolve()
    result = evaluate(
        artifact=artifact,
        assembly=_read(args.assembly or artifact / ASSEMBLY_NAME),
        capability=_read(args.capability),
        parity=_read(args.runtime_parity),
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0 if result["verdict"] == "PASS_FULL_MODEL_COMPOSITION" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CompositionGateError, OSError, ValueError) as error:
        print(json.dumps({"verdict": "ERROR", "error": str(error)}, indent=2))
        raise SystemExit(2) from error
