#!/usr/bin/env python3.12
"""Strict, dependency-free receipt contract for Appendix evidence cells."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import pathlib
import re
import sys
from typing import Any


SCHEMA = "hawking.appendix_receipt.v1"
PHASES = {"static", "load", "prefill", "draft", "verify", "decode", "kv", "sync", "io", "quality"}
BYTE_FIELDS = ("weights", "metadata", "codebook", "activations", "scratch", "state", "communication", "io")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{7,64}$")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _nearest_rank(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def rollup_samples(samples: list[dict]) -> dict:
    if not samples:
        return {
            "sample_count": 0,
            "wall_ns": {"p50": None, "p95": None, "p99": None, "total": 0},
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "energy_j": None,
            "bytes": {field: 0 for field in BYTE_FIELDS},
        }
    walls = [int(sample["wall_ns"]) for sample in samples]
    energies = [sample.get("energy_j") for sample in samples]
    energy = None if any(value is None for value in energies) else sum(float(value) for value in energies)
    byte_totals = {field: 0 for field in BYTE_FIELDS}
    for sample in samples:
        for field in BYTE_FIELDS:
            byte_totals[field] += int(sample["bytes"][field])
    return {
        "sample_count": len(samples),
        "wall_ns": {
            "p50": _nearest_rank(walls, 0.50),
            "p95": _nearest_rank(walls, 0.95),
            "p99": _nearest_rank(walls, 0.99),
            "total": sum(walls),
        },
        "accepted_tokens": sum(int(sample["accepted_tokens"]) for sample in samples),
        "rejected_tokens": sum(int(sample["rejected_tokens"]) for sample in samples),
        "energy_j": energy,
        "bytes": byte_totals,
    }


def stamp_receipt(receipt: dict) -> dict:
    stamped = copy.deepcopy(receipt)
    stamped.pop("receipt_sha256", None)
    stamped["receipt_sha256"] = canonical_sha256(stamped)
    return stamped


def deferred_template(cell_id: str, experiment_schema: str) -> dict:
    return stamp_receipt({
        "schema": SCHEMA,
        "experiment_schema": experiment_schema,
        "cell_id": cell_id,
        "status": "deferred",
        "bindings": {
            "source_commit": None,
            "target_sha256": None,
            "target_not_applicable_reason": None,
            "prompt_set_sha256": None,
            "prompt_not_applicable_reason": None,
            "parent_receipt_sha256": [],
        },
        "exactness": {"required": True, "passed": None, "mismatches": None},
        "samples": [],
        "rollup": rollup_samples([]),
        "resources": {
            "observed": False,
            "memory_pressure": None,
            "swap_delta_mb": None,
            "thermal_state": None,
            "not_applicable_reason": "execution not performed",
        },
        "failure_reasons": ["execution not performed"],
    })


def _nonnegative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def validate_receipt(doc: Any, *, known_cell_ids: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    if not isinstance(doc, dict):
        return ["receipt must be an object"]
    if doc.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    if not isinstance(doc.get("experiment_schema"), str) or not doc["experiment_schema"]:
        errors.append("experiment_schema must be a non-empty string")
    cell_id = doc.get("cell_id")
    if not isinstance(cell_id, str) or not cell_id:
        errors.append("cell_id must be a non-empty string")
    elif known_cell_ids is not None and cell_id not in known_cell_ids:
        errors.append("cell_id is not present in the supplied plan")
    status = doc.get("status")
    if status not in {"deferred", "complete", "failed"}:
        errors.append("status must be deferred|complete|failed")

    bindings = doc.get("bindings")
    if not isinstance(bindings, dict):
        errors.append("bindings must be an object")
    else:
        commit = bindings.get("source_commit")
        if status != "deferred" and (not isinstance(commit, str) or not COMMIT.fullmatch(commit)):
            errors.append("bindings.source_commit must be a 7-64 digit lowercase hex commit")
        for name, reason_name in (
            ("target_sha256", "target_not_applicable_reason"),
            ("prompt_set_sha256", "prompt_not_applicable_reason"),
        ):
            value = bindings.get(name)
            reason = bindings.get(reason_name)
            if value is not None and (not isinstance(value, str) or not HEX64.fullmatch(value)):
                errors.append(f"bindings.{name} must be null or lowercase SHA-256")
            if status == "complete" and value is None and (not isinstance(reason, str) or not reason.strip()):
                errors.append(f"complete receipt requires bindings.{name} or {reason_name}")
        parents = bindings.get("parent_receipt_sha256")
        if not isinstance(parents, list) or any(not isinstance(item, str) or not HEX64.fullmatch(item) for item in parents):
            errors.append("bindings.parent_receipt_sha256 must be a list of SHA-256 strings")

    exactness = doc.get("exactness")
    if not isinstance(exactness, dict):
        errors.append("exactness must be an object")
    else:
        required = exactness.get("required")
        passed = exactness.get("passed")
        mismatches = exactness.get("mismatches")
        if not isinstance(required, bool):
            errors.append("exactness.required must be boolean")
        if passed is not None and not isinstance(passed, bool):
            errors.append("exactness.passed must be boolean or null")
        if mismatches is not None and (not isinstance(mismatches, int) or isinstance(mismatches, bool) or mismatches < 0):
            errors.append("exactness.mismatches must be a non-negative integer or null")
        if status == "complete" and required and (passed is not True or mismatches != 0):
            errors.append("exactness-required complete receipt must pass with zero mismatches")

    samples = doc.get("samples")
    if not isinstance(samples, list):
        errors.append("samples must be a list")
        samples = []
    if status == "complete" and not samples:
        errors.append("complete receipt requires at least one sample")
    for index, sample in enumerate(samples):
        prefix = f"samples[{index}]"
        if not isinstance(sample, dict):
            errors.append(f"{prefix} must be an object")
            continue
        if sample.get("index") != index:
            errors.append(f"{prefix}.index must equal {index}")
        if sample.get("phase") not in PHASES:
            errors.append(f"{prefix}.phase is invalid")
        for field in ("wall_ns", "accepted_tokens", "rejected_tokens"):
            if not isinstance(sample.get(field), int) or isinstance(sample.get(field), bool) or sample[field] < 0:
                errors.append(f"{prefix}.{field} must be a non-negative integer")
        energy = sample.get("energy_j")
        if energy is not None and not _nonnegative_number(energy):
            errors.append(f"{prefix}.energy_j must be non-negative or null")
        byte_map = sample.get("bytes")
        if not isinstance(byte_map, dict) or set(byte_map) != set(BYTE_FIELDS):
            errors.append(f"{prefix}.bytes must contain exactly {','.join(BYTE_FIELDS)}")
        elif any(not isinstance(byte_map[field], int) or isinstance(byte_map[field], bool) or byte_map[field] < 0 for field in BYTE_FIELDS):
            errors.append(f"{prefix}.bytes values must be non-negative integers")

    if samples and not errors:
        expected_rollup = rollup_samples(samples)
        if doc.get("rollup") != expected_rollup:
            errors.append("rollup does not match deterministic sample rollup")
    elif not samples and doc.get("rollup") != rollup_samples([]):
        errors.append("empty-sample rollup is invalid")

    resources = doc.get("resources")
    if not isinstance(resources, dict):
        errors.append("resources must be an object")
    else:
        observed = resources.get("observed")
        if not isinstance(observed, bool):
            errors.append("resources.observed must be boolean")
        if resources.get("memory_pressure") not in {None, "normal", "warning", "critical"}:
            errors.append("resources.memory_pressure is invalid")
        swap = resources.get("swap_delta_mb")
        if swap is not None and not _nonnegative_number(swap):
            errors.append("resources.swap_delta_mb must be non-negative or null")
        if resources.get("thermal_state") not in {None, "nominal", "fair", "serious", "critical"}:
            errors.append("resources.thermal_state is invalid")
        reason = resources.get("not_applicable_reason")
        if observed is True:
            if reason is not None:
                errors.append("observed resources cannot carry not_applicable_reason")
            if (
                resources.get("memory_pressure") is None
                or swap is None
                or resources.get("thermal_state") is None
            ):
                errors.append("observed resources require pressure, swap, and thermal fields")
        elif status == "complete" and (not isinstance(reason, str) or not reason.strip()):
            errors.append("unobserved resources require not_applicable_reason")

    failures = doc.get("failure_reasons")
    if not isinstance(failures, list) or any(not isinstance(item, str) or not item for item in failures):
        errors.append("failure_reasons must be a list of non-empty strings")
    elif status in {"deferred", "failed"} and not failures:
        errors.append("deferred/failed receipt requires a reason")
    elif status == "complete" and failures:
        errors.append("complete receipt cannot contain failure reasons")

    claimed_hash = doc.get("receipt_sha256")
    unstamped = copy.deepcopy(doc)
    unstamped.pop("receipt_sha256", None)
    expected_hash = canonical_sha256(unstamped)
    if claimed_hash != expected_hash:
        errors.append("receipt_sha256 does not match canonical receipt bytes")
    return errors


def _load(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _selftest() -> int:
    template = deferred_template("appendix-test", "hawking.test.v1")
    assert validate_receipt(template) == []
    sample = {
        "index": 0,
        "phase": "static",
        "wall_ns": 0,
        "accepted_tokens": 0,
        "rejected_tokens": 0,
        "energy_j": None,
        "bytes": {field: 0 for field in BYTE_FIELDS},
    }
    complete = stamp_receipt({
        **template,
        "status": "complete",
        "bindings": {
            "source_commit": "0123456789abcdef",
            "target_sha256": None,
            "target_not_applicable_reason": "analytic geometry probe",
            "prompt_set_sha256": None,
            "prompt_not_applicable_reason": "no prompts consumed",
            "parent_receipt_sha256": [],
        },
        "exactness": {"required": True, "passed": True, "mismatches": 0},
        "samples": [sample],
        "rollup": rollup_samples([sample]),
        "resources": {
            "observed": False,
            "memory_pressure": None,
            "swap_delta_mb": None,
            "thermal_state": None,
            "not_applicable_reason": "static analytic test",
        },
        "failure_reasons": [],
    })
    assert validate_receipt(complete) == []
    broken = copy.deepcopy(complete)
    broken["rollup"]["sample_count"] = 2
    broken = stamp_receipt(broken)
    assert "rollup does not match deterministic sample rollup" in validate_receipt(broken)
    print("appendix_contract.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--selftest"]:
        return _selftest()
    if len(argv) == 3 and argv[0] == "--template":
        print(json.dumps(deferred_template(argv[1], argv[2]), indent=2, sort_keys=True))
        return 0
    if len(argv) == 2 and argv[0] == "--validate":
        errors = validate_receipt(_load(pathlib.Path(argv[1])))
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
        return 0 if not errors else 1
    print("usage: appendix_contract.py --template CELL_ID SCHEMA | --validate PATH | --selftest", file=sys.stderr)
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
