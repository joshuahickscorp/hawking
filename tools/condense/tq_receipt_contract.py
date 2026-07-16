#!/usr/bin/env python3.12
"""Fail-closed payload contract for post-run TQ device measurements."""

from __future__ import annotations

import json
import math
import pathlib
import sys
from typing import Any

import appendix_contract
import physical_counter_attestation


SCHEMA = "hawking.tq_runtime_device.v1"
RECIPES = {
    "stored": {"metadata": "expanded", "codebook": "stored", "entry_bytes": 84, "kernel": "strand_bitslice_gemv_partials"},
    "compact": {"metadata": "compact", "codebook": "stored", "entry_bytes": 40, "kernel": "strand_bitslice_gemv_partials_compact"},
    "hashed": {"metadata": "expanded", "codebook": "hashed_quantile", "entry_bytes": 84, "kernel": "strand_bitslice_gemv_partials_hashed"},
    "computed": {"metadata": "expanded", "codebook": "computed_acklam", "entry_bytes": 84, "kernel": "strand_bitslice_gemv_partials_computed"},
}
TRAFFIC_FIELDS = ("payload", "metadata", "codebook_staging", "partial_roundtrip")


def requirements() -> dict:
    return {
        "schema": "hawking.tq_runtime_device_contract.v1",
        "outer_schema": appendix_contract.SCHEMA,
        "experiment_schema": SCHEMA,
        "runtime_paths": RECIPES,
        "required_gates": [
            "eligible fused geometry",
            "Metal compile",
            "host/device entry-size identity",
            "exact Q12 and fused-GEMV parity",
            "at least 3 warmups and 10 measured trials",
            "p50/p95/p99, speedup LCB, occupancy, bandwidth, energy, pressure, swap, thermal",
            "no default change from a device microbenchmark alone",
        ],
    }


def _finite(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def validate_payload(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["experiment_payload must be an object"]
    errors: list[str] = []
    runtime = payload.get("runtime_path")
    expected = RECIPES.get(runtime)
    if expected is None:
        errors.append("runtime_path must be stored|compact|hashed|computed")
        expected = {}
    recipe = payload.get("recipe")
    if not isinstance(recipe, dict):
        errors.append("recipe must be an object")
    elif expected and (
        recipe.get("metadata") != expected["metadata"]
        or recipe.get("codebook") != expected["codebook"]
    ):
        errors.append("recipe does not match runtime_path")

    shape = payload.get("shape")
    if not isinstance(shape, dict):
        errors.append("shape must be an object")
        weights = None
    else:
        for field in ("rows", "cols", "blocks"):
            if not isinstance(shape.get(field), int) or isinstance(shape.get(field), bool) or shape[field] <= 0:
                errors.append(f"shape.{field} must be a positive integer")
        k_bits = shape.get("k_bits")
        l_bits = shape.get("l_bits")
        if k_bits not in range(1, 5):
            errors.append("shape.k_bits must be 1..4")
        if l_bits not in range(4, 15) or (isinstance(k_bits, int) and l_bits < k_bits):
            errors.append("shape.l_bits must be max(k,4)..14")
        weights = shape.get("rows", 0) * shape.get("cols", 0) if all(isinstance(shape.get(field), int) for field in ("rows", "cols")) else None
        if weights is not None and shape.get("weights") != weights:
            errors.append("shape.weights must equal rows*cols")

    admission = payload.get("admission")
    if not isinstance(admission, dict) or admission.get("eligible") is not True or admission.get("reason") is not None:
        errors.append("device receipt requires eligible admission with null reason")

    metal = payload.get("metal")
    if not isinstance(metal, dict):
        errors.append("metal must be an object")
    elif expected:
        if metal.get("compiled") is not True:
            errors.append("Metal source did not compile")
        if metal.get("kernel") != expected["kernel"]:
            errors.append("Metal kernel name does not match runtime path")
        if metal.get("host_entry_bytes") != expected["entry_bytes"]:
            errors.append("host entry size is wrong")
        if metal.get("gpu_entry_bytes") != expected["entry_bytes"]:
            errors.append("GPU entry size is wrong")

    parity = payload.get("parity")
    if not isinstance(parity, dict):
        errors.append("parity must be an object")
    else:
        if parity.get("exact_q12") is not True or parity.get("exact_fused_gemv") is not True:
            errors.append("Q12 and fused-GEMV parity must both be exact")
        if parity.get("mismatches") != 0:
            errors.append("parity requires zero mismatches")
        if not isinstance(parity.get("values_compared"), int) or parity["values_compared"] <= 0:
            errors.append("parity.values_compared must be positive")
        if not isinstance(parity.get("cases"), int) or parity["cases"] <= 0:
            errors.append("parity.cases must be positive")

    traffic = payload.get("logical_traffic")
    if not isinstance(traffic, dict):
        errors.append("logical_traffic must be an object")
    else:
        if any(not isinstance(traffic.get(field), int) or isinstance(traffic.get(field), bool) or traffic[field] < 0 for field in TRAFFIC_FIELDS):
            errors.append("logical traffic components must be non-negative integers")
        elif traffic.get("compressed_runtime_total") != sum(traffic[field] for field in TRAFFIC_FIELDS):
            errors.append("compressed_runtime_total does not equal traffic components")
        if weights:
            normalization_weights = weights
            residual = payload.get("residual_probe")
            if isinstance(residual, dict) and residual.get("enabled") is True:
                residual_weights = residual.get("tensor", {}).get("weights")
                if (
                    not isinstance(residual_weights, int)
                    or isinstance(residual_weights, bool)
                    or residual_weights <= 0
                ):
                    errors.append("enabled residual probe lacks positive normalization weights")
                else:
                    normalization_weights += residual_weights
            expected_bpw = (
                traffic.get("compressed_runtime_total", 0) * 8
                / normalization_weights
            )
            if not _finite(traffic.get("compressed_runtime_bpw")) or abs(traffic["compressed_runtime_bpw"] - expected_bpw) > 1e-9:
                errors.append("compressed_runtime_bpw does not match all decoded projection weights")

    benchmark = payload.get("benchmark")
    if not isinstance(benchmark, dict):
        errors.append("benchmark must be an object")
    else:
        if not isinstance(benchmark.get("warmups"), int) or benchmark["warmups"] < 3:
            errors.append("benchmark requires at least three warmups")
        if not isinstance(benchmark.get("trials"), int) or benchmark["trials"] < 10:
            errors.append("benchmark requires at least ten trials")
        p50, p95, p99 = (benchmark.get(field) for field in ("p50_ns", "p95_ns", "p99_ns"))
        if not all(_finite(value, positive=True) for value in (p50, p95, p99)):
            errors.append("benchmark percentiles must be positive")
        elif not (p50 <= p95 <= p99):
            errors.append("benchmark percentiles must be monotone")
        for field in ("speedup_median", "speedup_lcb", "joules_per_invocation"):
            if not _finite(benchmark.get(field), positive=True):
                errors.append(f"benchmark.{field} must be positive")

    counters = payload.get("physical_counters")
    if not isinstance(counters, dict) or counters.get("measured") is not True:
        errors.append("physical counters must be measured")
    else:
        occupancy = counters.get("occupancy_percent")
        if not _finite(occupancy) or occupancy < 0 or occupancy > 100:
            errors.append("occupancy_percent must be in [0,100]")
        if not _finite(counters.get("realized_bandwidth_bytes_per_second"), positive=True):
            errors.append("realized bandwidth must be positive")
        if not _finite(counters.get("gpu_time_ns"), positive=True):
            errors.append("gpu_time_ns must be positive")
        if not _finite(counters.get("physical_bytes_total"), positive=True):
            errors.append("physical_bytes_total must be positive")
        if counters.get("byte_samples_are_logical_accounting") is not True:
            errors.append("device receipt must distinguish logical sample bytes from physical bytes")

    attestation = payload.get("counter_attestation")
    if not isinstance(attestation, dict) or attestation.get("schema") != physical_counter_attestation.SCHEMA:
        errors.append("physical counter attestation is missing")
    else:
        unstamped = dict(attestation)
        claimed = unstamped.pop("attestation_sha256", None)
        if claimed != physical_counter_attestation.canonical_sha256(unstamped):
            errors.append("physical counter attestation self-hash mismatch")

    if payload.get("default_change_requested") is not False:
        errors.append("device microbenchmark cannot request a default change")
    return errors


def validate_receipt(receipt: Any, *, known_cell_ids: set[str] | None = None) -> list[str]:
    errors = appendix_contract.validate_receipt(receipt, known_cell_ids=known_cell_ids)
    if not isinstance(receipt, dict) or receipt.get("status") != "complete":
        return errors
    if receipt.get("experiment_schema") != SCHEMA:
        errors.append(f"experiment_schema must be {SCHEMA}")
    else:
        errors.extend(validate_payload(receipt.get("experiment_payload")))
    if isinstance(receipt, dict) and receipt.get("status") == "complete":
        payload = receipt.get("experiment_payload")
        attestation = payload.get("counter_attestation") if isinstance(payload, dict) else None
        claimed = receipt.get("physical_counter_attestation_sha256")
        if not isinstance(attestation, dict) or claimed != attestation.get("attestation_sha256"):
            errors.append("receipt does not bind the physical counter attestation")
        bundle_sha = receipt.get("physical_counter_bundle_sha256")
        if not isinstance(bundle_sha, str) or len(bundle_sha) != 64 or any(
            character not in "0123456789abcdef" for character in bundle_sha
        ):
            errors.append("receipt does not bind the physical counter bundle")
        sources = receipt.get("physical_counter_sources")
        expected_sources = {
            "occupancy_source", "bandwidth_source", "energy_source",
            "gpu_time_source", "bytes_source",
        }
        if not isinstance(sources, dict) or set(sources) != expected_sources or any(
            not isinstance(value, str) or not value.strip() for value in sources.values()
        ):
            errors.append("receipt physical counter sources are incomplete")
    return errors


def _valid_payload() -> dict:
    weights = 256
    traffic = {"payload": 96, "metadata": 40, "codebook_staging": 64, "partial_roundtrip": 8}
    traffic["compressed_runtime_total"] = sum(traffic.values())
    traffic["compressed_runtime_bpw"] = traffic["compressed_runtime_total"] * 8 / weights
    return {
        "runtime_path": "compact",
        "recipe": {"metadata": "compact", "codebook": "stored"},
        "shape": {"tensor": "unit", "rows": 1, "cols": 256, "weights": weights, "blocks": 1, "k_bits": 3, "l_bits": 9},
        "admission": {"eligible": True, "reason": None},
        "metal": {"compiled": True, "kernel": "strand_bitslice_gemv_partials_compact", "host_entry_bytes": 40, "gpu_entry_bytes": 40},
        "parity": {"exact_q12": True, "exact_fused_gemv": True, "mismatches": 0, "values_compared": 256, "cases": 1},
        "logical_traffic": traffic,
        "benchmark": {"warmups": 3, "trials": 10, "p50_ns": 10, "p95_ns": 11, "p99_ns": 12, "speedup_median": 1.01, "speedup_lcb": 1.0, "joules_per_invocation": 0.001},
        "physical_counters": {"measured": True, "occupancy_percent": 50.0, "realized_bandwidth_bytes_per_second": 1e9, "gpu_time_ns": 9, "physical_bytes_total": 1024, "byte_samples_are_logical_accounting": True},
        "counter_attestation": physical_counter_attestation.stamp({
            "schema": physical_counter_attestation.SCHEMA,
        }),
        "default_change_requested": False,
    }


def _selftest() -> int:
    payload = _valid_payload()
    assert validate_payload(payload) == []
    payload["metal"]["gpu_entry_bytes"] = 84
    assert "GPU entry size is wrong" in validate_payload(payload)
    print("tq_receipt_contract.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    if argv == ["--requirements"]:
        print(json.dumps(requirements(), indent=2, sort_keys=True))
        return 0
    if argv == ["--selftest"]:
        return _selftest()
    if len(argv) == 2 and argv[0] == "--validate":
        with pathlib.Path(argv[1]).open("r", encoding="utf-8") as handle:
            errors = validate_receipt(json.load(handle))
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
        return 0 if not errors else 1
    print("usage: tq_receipt_contract.py --requirements | --validate PATH | --selftest", file=sys.stderr)
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
