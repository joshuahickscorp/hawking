from __future__ import annotations

import copy
import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tools" / "condense" / "appendix_contract.py"
SPEC = importlib.util.spec_from_file_location("appendix_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _complete_receipt() -> dict:
    sample = {
        "index": 0,
        "phase": "static",
        "wall_ns": 7,
        "accepted_tokens": 0,
        "rejected_tokens": 0,
        "energy_j": None,
        "bytes": {field: 0 for field in MODULE.BYTE_FIELDS},
    }
    return MODULE.stamp_receipt({
        "schema": MODULE.SCHEMA,
        "experiment_schema": "hawking.test.v1",
        "cell_id": "cell-a",
        "status": "complete",
        "bindings": {
            "source_commit": "0123456789abcdef",
            "target_sha256": None,
            "target_not_applicable_reason": "static test",
            "prompt_set_sha256": None,
            "prompt_not_applicable_reason": "no prompts",
            "parent_receipt_sha256": [],
        },
        "exactness": {"required": True, "passed": True, "mismatches": 0},
        "samples": [sample],
        "rollup": MODULE.rollup_samples([sample]),
        "resources": {
            "observed": False,
            "memory_pressure": None,
            "swap_delta_mb": None,
            "thermal_state": None,
            "not_applicable_reason": "static unit test",
        },
        "failure_reasons": [],
    })


def test_deferred_and_complete_receipts_validate() -> None:
    assert MODULE.validate_receipt(MODULE.deferred_template("cell-a", "hawking.test.v1")) == []
    assert MODULE.validate_receipt(_complete_receipt(), known_cell_ids={"cell-a"}) == []


def test_receipt_rejects_hash_rollup_exactness_and_unknown_cell_tampering() -> None:
    receipt = _complete_receipt()
    receipt["samples"][0]["wall_ns"] = 8
    errors = MODULE.validate_receipt(receipt, known_cell_ids={"cell-b"})
    assert "cell_id is not present in the supplied plan" in errors
    assert "receipt_sha256 does not match canonical receipt bytes" in errors

    repaired_hash_only = MODULE.stamp_receipt(receipt)
    errors = MODULE.validate_receipt(repaired_hash_only)
    assert "rollup does not match deterministic sample rollup" in errors

    bad_exactness = _complete_receipt()
    bad_exactness["exactness"] = {"required": True, "passed": False, "mismatches": 1}
    bad_exactness = MODULE.stamp_receipt(bad_exactness)
    assert any("exactness-required" in error for error in MODULE.validate_receipt(bad_exactness))


def test_validator_reports_malformed_samples_without_crashing() -> None:
    receipt = _complete_receipt()
    receipt["samples"] = [{"index": 0}]
    receipt = MODULE.stamp_receipt(receipt)
    errors = MODULE.validate_receipt(receipt)
    assert any("phase" in error for error in errors)
    assert any("wall_ns" in error for error in errors)
