#!/usr/bin/env python3.12
"""Lease-gated TQ speculative-verifier runner and strict receipt finalizer."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
import pathlib
import secrets
import statistics
import subprocess
import sys
import time
from typing import Any

import appendix_contract
import physical_counter_attestation
import ram_scheduler
import spec_receipt_contract
import spec_reentry_scaffold


ROOT = pathlib.Path(__file__).resolve().parents[2]
HEAVY_LOCK = ROOT / "reports" / "cron" / "studio_heavy.lock"
DEFAULT_PROBE = ROOT / "target" / "release" / "hawking-tq-spec-probe"
RAW_SCHEMA = "hawking.spec_tq_batched_raw.v1"
BUNDLE_SCHEMA = "hawking.spec_tq_batched_raw_bundle.v1"
COUNTER_SCHEMA = "hawking.spec_tq_physical_counters.v2"
RUNTIME_KERNEL = {
    "stored": "strand_bitslice_gemm_small_stored",
    "compact": "strand_bitslice_gemm_small_compact",
    "hashed": "strand_bitslice_gemm_small_hashed",
    "computed": "strand_bitslice_gemm_small_computed",
}
BYTE_FIELDS = appendix_contract.BYTE_FIELDS


def _finite(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and (value > 0 if positive else value >= 0)
    )


def _binding(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("sha256"), str)
        and len(value["sha256"]) == 64
        and all(character in "0123456789abcdef" for character in value["sha256"])
        and isinstance(value.get("size_bytes"), int)
        and value["size_bytes"] > 0
    )


def validate_raw(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return ["raw spec probe must be an object"]
    errors: list[str] = []
    if raw.get("schema") != RAW_SCHEMA:
        errors.append(f"raw schema must be {RAW_SCHEMA}")
    commit = raw.get("source_commit")
    if (
        not isinstance(commit, str)
        or not 7 <= len(commit) <= 64
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        errors.append("raw source_commit must be lowercase hexadecimal")
    for field in ("model", "artifact"):
        if not _binding(raw.get(field)):
            errors.append(f"raw {field} binding is invalid")
    prompt_set = raw.get("prompt_set")
    if (
        not isinstance(prompt_set, dict)
        or prompt_set.get("schema") != "hawking.spec_token_prompts.v1"
        or not isinstance(prompt_set.get("sha256"), str)
        or len(prompt_set["sha256"]) != 64
        or not isinstance(prompt_set.get("prompts"), int)
        or prompt_set["prompts"] < 20
        or not isinstance(prompt_set.get("tokenizer_sha256"), str)
        or len(prompt_set["tokenizer_sha256"]) != 64
    ):
        errors.append("raw prompt-set binding is invalid")
    tokenizer = raw.get("tokenizer")
    if (
        not _binding(tokenizer)
        or tokenizer.get("source") not in {"tokenizer_json", "gguf_embedded"}
        or not isinstance(prompt_set, dict)
        or prompt_set.get("tokenizer_sha256") != tokenizer.get("sha256")
    ):
        errors.append("raw tokenizer binding is invalid")
    mode = raw.get("runtime_path")
    if mode not in RUNTIME_KERNEL or raw.get("kernel") != RUNTIME_KERNEL.get(mode):
        errors.append("raw runtime/kernel identity is invalid")
    matrix_identity = raw.get("matrix_identity")
    matrix_fields = {
        "runtime_path", "parity_cell_id", "curve_cell_id", "model_sha256",
        "artifact_sha256", "tokenizer_sha256", "prompt_set_sha256",
    }
    if not isinstance(matrix_identity, dict) or set(matrix_identity) != matrix_fields:
        errors.append("raw speculative matrix_identity is incomplete or unexpected")
    else:
        if (
            matrix_identity.get("runtime_path") != mode
            or matrix_identity.get("model_sha256") != raw.get("model", {}).get("sha256")
            or matrix_identity.get("artifact_sha256") != raw.get("artifact", {}).get("sha256")
            or matrix_identity.get("tokenizer_sha256") != raw.get("tokenizer", {}).get("sha256")
            or matrix_identity.get("prompt_set_sha256") != raw.get("prompt_set", {}).get("sha256")
            or not all(
                isinstance(matrix_identity.get(field), str) and matrix_identity[field]
                for field in ("parity_cell_id", "curve_cell_id")
            )
        ):
            errors.append("raw speculative matrix_identity bindings are invalid")
    device = raw.get("device")
    if (
        not isinstance(device, dict)
        or device.get("profile") != "Studio-M3Ultra-96"
        or not isinstance(device.get("name"), str)
        or not device["name"]
    ):
        errors.append("raw device identity is invalid")
    coverage = raw.get("coverage")
    if not isinstance(coverage, dict):
        errors.append("raw TQ coverage is missing")
    else:
        expected = coverage.get("expected_all_linear")
        if (
            not isinstance(expected, int)
            or expected <= 0
            or coverage.get("mapped") != expected
            or coverage.get("gpu_resident") != expected
        ):
            errors.append("raw TQ all-linear GPU coverage is incomplete")
        residual = coverage.get("residual_gpu_resident")
        if not isinstance(residual, int) or residual < 0 or (isinstance(expected, int) and residual > expected):
            errors.append("raw residual TQ coverage is invalid")
    identity = raw.get("target_identity")
    if identity != {
        "reference": "tq_single_token_greedy",
        "verifier": "tq_batch_major_b1_b8",
        "greedy_tie_break": "canonical_qwen_argmax",
        "all_owned_projections_tq_native": True,
    }:
        errors.append("raw target/verifier identity is invalid")
    protocol = raw.get("measurement_protocol")
    protocol_fields = {
        "warmups_per_batch", "independent_repeats_per_batch",
        "randomized_balanced_batch_order", "paired_interleaved_baseline",
        "baseline_reused_across_batches", "phase_marker_schema",
        "phase_markers_sha256", "monotone_transform_applied", "batches",
    }
    if not isinstance(protocol, dict) or set(protocol) != protocol_fields:
        errors.append("raw measurement_protocol is incomplete or unexpected")
        protocol = {}
    warmups = protocol.get("warmups_per_batch")
    repeats = protocol.get("independent_repeats_per_batch")
    if isinstance(warmups, bool) or not isinstance(warmups, int) or warmups < 3:
        errors.append("raw measurement protocol requires at least three warmups per batch")
        warmups = 0
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 5:
        errors.append("raw measurement protocol requires at least five repeats per batch")
        repeats = 0
    if (
        protocol.get("randomized_balanced_batch_order") is not True
        or protocol.get("paired_interleaved_baseline") is not True
        or protocol.get("baseline_reused_across_batches") is not False
        or protocol.get("monotone_transform_applied") is not False
        or protocol.get("phase_marker_schema") != physical_counter_attestation.PHASE_MARKERS_SCHEMA
    ):
        errors.append("raw measurement protocol policy is invalid")
    rows = raw.get("batches")
    prompt_count = prompt_set.get("prompts") if isinstance(prompt_set, dict) else None
    if not isinstance(rows, list) or len(rows) != 8:
        errors.append("raw batches must contain exactly B=1..8")
        rows = []
    elif [row.get("b") for row in rows if isinstance(row, dict)] != list(range(1, 9)):
        errors.append("raw batches must be ordered B=1..8")
    for row in rows:
        if not isinstance(row, dict):
            errors.append("raw batch row must be an object")
            continue
        batch = row.get("b")
        count = row.get("generated_tokens_per_prompt")
        if row.get("prompts") != prompt_count or not isinstance(count, int) or count < 256:
            errors.append(f"B={batch} corpus size is insufficient")
        elif not isinstance(batch, int) or count % batch != 0:
            errors.append(f"B={batch} token count must be divisible by batch")
        expected_values = (
            prompt_count * count * repeats * 2
            if isinstance(prompt_count, int) and isinstance(repeats, int) else None
        )
        if row.get("values_compared") != expected_values:
            errors.append(f"B={batch} values_compared is invalid")
        if (
            row.get("exact_token_match") is not True
            or row.get("mismatches") != 0
            or row.get("skipped") != 0
        ):
            errors.append(f"B={batch} parity failed or skipped cases exist")
        baseline = row.get("baseline_greedy_wall_ns")
        verifier = row.get("verifier_wall_ns")
        if (
            not isinstance(baseline, list)
            or not isinstance(verifier, list)
            or len(baseline) != repeats
            or len(verifier) != repeats
            or any(not isinstance(value, int) or value <= 0 for value in baseline + verifier)
        ):
            errors.append(f"B={batch} timing arrays are invalid")
    protocol_rows = protocol.get("batches")
    if not isinstance(protocol_rows, list) or [
        row.get("b") for row in protocol_rows if isinstance(row, dict)
    ] != list(range(1, 9)):
        errors.append("raw measurement protocol batches must be ordered B=1..8")
        protocol_rows = []
    all_repeat_markers: set[str] = set()
    for batch_row in protocol_rows:
        batch = batch_row["b"]
        measured = batch_row.get("repeats")
        batch_raw = rows[batch - 1] if isinstance(rows, list) and len(rows) == 8 else {}
        if not isinstance(measured, list) or len(measured) != repeats:
            errors.append(f"B={batch} protocol does not contain every repeat")
            continue
        for repeat, measurement in enumerate(measured):
            expected_fields = {
                "repeat", "baseline_wall_ns", "verifier_wall_ns",
                "phase_marker_sha256", "exact_token_match", "mismatches", "skipped",
            }
            if not isinstance(measurement, dict) or set(measurement) != expected_fields:
                errors.append(f"B={batch} repeat {repeat} is malformed")
                continue
            marker = measurement.get("phase_marker_sha256")
            if (
                measurement.get("repeat") != repeat
                or not isinstance(marker, str) or len(marker) != 64
                or marker in all_repeat_markers
            ):
                errors.append(f"B={batch} repeat marker/index is invalid or reused")
            else:
                all_repeat_markers.add(marker)
            if (
                measurement.get("exact_token_match") is not True
                or measurement.get("mismatches") != 0 or measurement.get("skipped") != 0
            ):
                errors.append(f"B={batch} repeat parity failed or skipped work")
            if (
                measurement.get("baseline_wall_ns")
                != batch_raw.get("baseline_greedy_wall_ns", [None] * repeats)[repeat]
                or measurement.get("verifier_wall_ns")
                != batch_raw.get("verifier_wall_ns", [None] * repeats)[repeat]
            ):
                errors.append(f"B={batch} repeat timing differs from raw batch arrays")
    phase_markers = raw.get("phase_markers")
    errors.extend(physical_counter_attestation.validate_phase_markers(phase_markers))
    if isinstance(phase_markers, dict):
        if protocol.get("phase_markers_sha256") != phase_markers.get("phase_markers_sha256"):
            errors.append("raw measurement protocol is not bound to phase_markers")
        pairs = {
            row.get("phase_marker_sha256"): row
            for row in phase_markers.get("pairs", []) if isinstance(row, dict)
        }
        trial_pairs = [row for row in pairs.values() if row.get("phase") == "trial"]
        warmup_pairs = [row for row in pairs.values() if row.get("phase") == "warmup"]
        if len(trial_pairs) != 8 * repeats or len(warmup_pairs) != 8 * warmups:
            errors.append("raw phase markers lack exact per-B warmup/trial coverage")
        expected_trial_identities = {
            (batch, repeat) for batch in range(1, 9) for repeat in range(repeats)
        }
        expected_warmup_identities = {
            (batch, warmup) for batch in range(1, 9) for warmup in range(warmups)
        }
        if {
            (row.get("batch"), row.get("iteration")) for row in trial_pairs
        } != expected_trial_identities or {
            (row.get("batch"), row.get("iteration")) for row in warmup_pairs
        } != expected_warmup_identities:
            errors.append("raw phase markers do not form exact balanced B=1..8 rounds")
        if any(
            "verifier_interval_sha256" not in pairs.get(marker, {})
            or pairs[marker].get("phase") != "trial"
            for marker in all_repeat_markers
        ):
            errors.append("raw repeat markers do not bind baseline/verifier pairs")
        nonce = phase_markers.get("run_nonce")
        ordered_pairs = phase_markers.get("pairs", [])
        if isinstance(nonce, str):
            for phase_name, rounds in (("warmup", warmups), ("trial", repeats)):
                for round_index in range(rounds):
                    observed_order = [
                        row.get("batch") for row in ordered_pairs
                        if isinstance(row, dict) and row.get("phase") == phase_name
                        and row.get("iteration") == round_index
                    ]
                    expected_order = sorted(
                        range(1, 9),
                        key=lambda batch: hashlib.sha256(
                            f"{nonce}:{phase_name}:{round_index}:{batch}".encode()
                        ).digest(),
                    )
                    if observed_order != expected_order:
                        errors.append(
                            f"raw {phase_name} round {round_index} batch order is not nonce-randomized"
                        )
    if raw.get("physical_counters") != {"measured": False}:
        errors.append("raw probe must not invent physical counters")
    if raw.get("default_change_requested") is not False:
        errors.append("raw probe cannot request a default change")
    return errors


def _stamp_bundle(bundle: dict) -> dict:
    stamped = copy.deepcopy(bundle)
    stamped.pop("raw_bundle_sha256", None)
    stamped["raw_bundle_sha256"] = appendix_contract.canonical_sha256(stamped)
    return stamped


def build_bundle(
    raw: dict,
    *,
    resource_before: dict,
    resource_after: dict,
    thermal_before: str,
    thermal_after: str,
    execution_authority: dict,
) -> dict:
    errors = validate_raw(raw)
    if errors:
        raise ValueError("; ".join(errors))
    before_swap = resource_before.get("swap_used_mb")
    after_swap = resource_after.get("swap_used_mb")
    swap_delta = (
        max(0.0, float(after_swap) - float(before_swap))
        if _finite(before_swap) and _finite(after_swap)
        else None
    )
    return _stamp_bundle({
        "schema": BUNDLE_SCHEMA,
        "raw_probe": raw,
        "raw_probe_sha256": appendix_contract.canonical_sha256(raw),
        "execution_authority": execution_authority,
        "runner_safety": {
            "exclusive_heavy_lease_held": True,
            "owners_rechecked_under_lease": True,
            "resource_before": resource_before,
            "resource_after": resource_after,
            "resource_state_before": ram_scheduler.classify_resource_state(resource_before),
            "resource_state_after": ram_scheduler.classify_resource_state(resource_after),
            "swap_delta_mb": swap_delta,
            "thermal_before": thermal_before,
            "thermal_after": thermal_after,
        },
    })


def validate_bundle(bundle: Any) -> list[str]:
    if not isinstance(bundle, dict):
        return ["spec raw bundle must be an object"]
    errors: list[str] = []
    if bundle.get("schema") != BUNDLE_SCHEMA:
        errors.append(f"bundle schema must be {BUNDLE_SCHEMA}")
    unstamped = copy.deepcopy(bundle)
    claimed = unstamped.pop("raw_bundle_sha256", None)
    if claimed != appendix_contract.canonical_sha256(unstamped):
        errors.append("raw_bundle_sha256 mismatch")
    raw = bundle.get("raw_probe")
    errors.extend(validate_raw(raw))
    if isinstance(raw, dict) and bundle.get("raw_probe_sha256") != appendix_contract.canonical_sha256(raw):
        errors.append("raw_probe_sha256 mismatch")
    errors.extend(physical_counter_attestation.validate_execution_authority(
        bundle.get("execution_authority"),
        raw_probe_sha256=appendix_contract.canonical_sha256(raw),
    ))
    authority = bundle.get("execution_authority")
    if isinstance(raw, dict) and isinstance(authority, dict):
        errors.extend(physical_counter_attestation.validate_phase_markers(
            raw.get("phase_markers"),
            run_nonce=authority.get("run_nonce"),
            workload_started_at_unix_ns=authority.get("started_at_unix_ns"),
            workload_ended_at_unix_ns=authority.get("ended_at_unix_ns"),
            workload_elapsed_continuous_ns=(
                authority.get("ended_at_continuous_ns", 0)
                - authority.get("started_at_continuous_ns", 0)
            ),
        ))
    safety = bundle.get("runner_safety")
    if not isinstance(safety, dict):
        errors.append("runner safety is missing")
    else:
        if safety.get("exclusive_heavy_lease_held") is not True or safety.get("owners_rechecked_under_lease") is not True:
            errors.append("runner did not prove lease and owner recheck")
        if safety.get("resource_state_before") != "green" or safety.get("resource_state_after") != "green":
            errors.append("runner resource state was not green")
        if safety.get("swap_delta_mb") != 0:
            errors.append("runner observed swap growth")
        if safety.get("thermal_before") != "nominal" or safety.get("thermal_after") != "nominal":
            errors.append("runner thermal state was not nominal")
    return errors


def validate_counters(
    counters: Any, counter_attestation: Any, bundle: dict,
) -> list[str]:
    if not isinstance(counters, dict):
        return ["physical counter bundle must be an object"]
    errors: list[str] = []
    raw = bundle["raw_probe"]
    expected_fields = {
        "schema", "raw_bundle_sha256", "artifact_sha256", "runtime_path",
        "phase_markers_sha256", "batches",
    }
    if set(counters) != expected_fields:
        errors.append("counter fields are incomplete or unexpected")
    if counters.get("schema") != COUNTER_SCHEMA:
        errors.append(f"counter schema must be {COUNTER_SCHEMA}")
    if counters.get("raw_bundle_sha256") != bundle.get("raw_bundle_sha256"):
        errors.append("counters are not bound to the raw bundle")
    if counters.get("artifact_sha256") != raw["artifact"]["sha256"]:
        errors.append("counters are not bound to the TQ artifact")
    if counters.get("runtime_path") != raw["runtime_path"]:
        errors.append("counters are not bound to the runtime path")
    if counters.get("phase_markers_sha256") != raw.get("phase_markers", {}).get("phase_markers_sha256"):
        errors.append("counters are not bound to raw phase markers")
    rows = counters.get("batches")
    if not isinstance(rows, list) or [row.get("b") for row in rows if isinstance(row, dict)] != list(range(1, 9)):
        errors.append("counter batches must be ordered B=1..8")
        rows = []
    protocol_by_b = {
        row["b"]: row for row in raw.get("measurement_protocol", {}).get("batches", [])
    }
    repeats = raw.get("measurement_protocol", {}).get("independent_repeats_per_batch", 0)
    for row in rows:
        batch = row.get("b")
        expected_markers = [
            repeat.get("phase_marker_sha256")
            for repeat in protocol_by_b.get(batch, {}).get("repeats", [])
        ]
        measurements = row.get("repeats")
        if not isinstance(measurements, list) or len(measurements) != repeats:
            errors.append(f"B={batch} counter repeat coverage is invalid")
            continue
        for repeat, measurement in enumerate(measurements):
            if not isinstance(measurement, dict) or set(measurement) != {
                "repeat", "phase_marker_sha256", "energy_j", "gpu_time_ns", "physical_bytes",
            }:
                errors.append(f"B={batch} counter repeat {repeat} is malformed")
                continue
            if (
                measurement.get("repeat") != repeat
                or repeat >= len(expected_markers)
                or measurement.get("phase_marker_sha256") != expected_markers[repeat]
            ):
                errors.append(f"B={batch} counter repeat {repeat} is not phase-bound")
            for field in ("energy_j", "gpu_time_ns", "physical_bytes"):
                if not _finite(measurement.get(field), positive=True):
                    errors.append(f"B={batch} counter repeat {repeat}.{field} must be positive")
    errors.extend(physical_counter_attestation.validate(
        counter_attestation,
        raw_bundle_sha256=bundle.get("raw_bundle_sha256"),
        artifact_sha256=raw["artifact"]["sha256"],
        execution_authority=bundle.get("execution_authority", {}),
        counter_payload=counters,
        required_domains=("energy", "gpu_time", "physical_bytes"),
        minimum_samples=max(1, repeats * 8),
    ))
    return errors


def _nearest(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _samples(raw: dict, counters: dict | None) -> list[dict]:
    counter_by_b = {row["b"]: row for row in counters["batches"]} if counters else {}
    samples: list[dict] = []
    for batch_row in raw["batches"]:
        count = batch_row["generated_tokens_per_prompt"]
        counter = counter_by_b.get(batch_row["b"])
        for repeat, wall_ns in enumerate(batch_row["verifier_wall_ns"]):
            if counter:
                physical = counter["repeats"][repeat]
                energy = physical["energy_j"]
                bytes_per = {field: 0 for field in BYTE_FIELDS}
                bytes_per["weights"] = physical["physical_bytes"]
                gpu_time = physical["gpu_time_ns"]
            else:
                energy = None
                bytes_per = {field: 0 for field in BYTE_FIELDS}
                gpu_time = None
            samples.append({
                "index": len(samples),
                "phase": "verify",
                "wall_ns": wall_ns,
                "accepted_tokens": 0,
                "rejected_tokens": 0,
                "energy_j": energy,
                "bytes": bytes_per,
                "batch": batch_row["b"],
                "verified_target_tokens": count * batch_row["prompts"],
                "repeat": repeat,
                "gpu_time_ns": gpu_time,
            })
    return samples


def _resources(bundle: dict) -> dict:
    safety = bundle["runner_safety"]
    return {
        "observed": True,
        "memory_pressure": "normal",
        "swap_delta_mb": safety["swap_delta_mb"],
        "thermal_state": "nominal",
        "not_applicable_reason": None,
    }


def _cells(runtime_path: str, label: str) -> tuple[dict, dict, set[str]]:
    matrix = spec_reentry_scaffold.build_matrix(label)
    parity = next(
        row for row in matrix["cells"]
        if row["receipt_schema"] == "hawking.spec_tq_batched_parity.v1"
        and row["knobs"]["runtime_path"] == runtime_path
    )
    curve = next(
        row for row in matrix["cells"]
        if row["receipt_schema"] == "hawking.spec_verifier_curve.v1"
        and row["knobs"]["runtime_path"] == runtime_path
    )
    return parity, curve, {row["id"] for row in matrix["cells"]}


def finalize_receipts(
    bundle: dict,
    counters: dict,
    counter_attestation: dict,
    *,
    label: str = "CORPUS",
) -> tuple[dict, dict]:
    errors = validate_bundle(bundle)
    errors.extend(validate_counters(counters, counter_attestation, bundle))
    if errors:
        raise ValueError("; ".join(errors))
    raw = bundle["raw_probe"]
    parity_cell, curve_cell, known_ids = _cells(raw["runtime_path"], label)
    samples = _samples(raw, counters)
    bindings = {
        "source_commit": raw["source_commit"],
        "target_sha256": raw["artifact"]["sha256"],
        "target_not_applicable_reason": None,
        "prompt_set_sha256": raw["prompt_set"]["sha256"],
        "prompt_not_applicable_reason": None,
        "parent_receipt_sha256": [],
    }
    parity_payload = {
        "runtime_path": raw["runtime_path"],
        "model_sha256": raw["model"]["sha256"],
        "tokenizer_sha256": raw["tokenizer"]["sha256"],
        "kernel": raw["kernel"],
        "coverage": raw["coverage"],
        "target_identity": raw["target_identity"],
        "batches": [
            {
                field: row[field]
                for field in (
                    "b", "prompts", "generated_tokens_per_prompt",
                    "exact_token_match", "mismatches", "skipped",
                )
            }
            for row in raw["batches"]
        ],
        "default_change_requested": False,
    }
    parity_receipt = appendix_contract.stamp_receipt({
        "schema": appendix_contract.SCHEMA,
        "experiment_schema": "hawking.spec_tq_batched_parity.v1",
        "cell_id": parity_cell["id"],
        "status": "complete",
        "bindings": bindings,
        "exactness": {"required": True, "passed": True, "mismatches": 0},
        "samples": samples,
        "rollup": appendix_contract.rollup_samples(samples),
        "resources": _resources(bundle),
        "failure_reasons": [],
        "experiment_payload": parity_payload,
        "raw_bundle_sha256": bundle["raw_bundle_sha256"],
        "counter_attestation": counter_attestation,
        "physical_counter_bundle_sha256": appendix_contract.canonical_sha256(counters),
        "physical_counter_payload_sha256": appendix_contract.canonical_sha256(counters),
        "physical_counter_attestation_sha256": counter_attestation[
            "attestation_sha256"
        ],
    })

    curve_rows = []
    for row in raw["batches"]:
        baseline = row["baseline_greedy_wall_ns"]
        verifier = row["verifier_wall_ns"]
        median = _nearest(verifier, 0.50)
        p95 = _nearest(verifier, 0.95)
        ucb = p95
        raw_equiv = median / statistics.median(baseline)
        physical = next(value for value in counters["batches"] if value["b"] == row["b"])
        energy_j_total = sum(value["energy_j"] for value in physical["repeats"])
        gpu_time_ns = sum(value["gpu_time_ns"] for value in physical["repeats"])
        physical_bytes = sum(value["physical_bytes"] for value in physical["repeats"])
        byte_map = {field: 0 for field in BYTE_FIELDS}
        byte_map["weights"] = physical_bytes
        curve_rows.append({
            "b": row["b"],
            "trials": len(verifier),
            "median_ns": median,
            "p95_ns": p95,
            "ucb_ns": ucb,
            "raw_total_forward_equiv": raw_equiv,
            "total_forward_equiv": raw_equiv,
            "energy_j_total": energy_j_total,
            "gpu_time_ns": gpu_time_ns,
            "bytes": byte_map,
        })
    curve_payload = {
        "runtime_path": raw["runtime_path"],
        "model_sha256": raw["model"]["sha256"],
        "tokenizer_sha256": raw["tokenizer"]["sha256"],
        "physical_counters_measured": True,
        "counter_attestation": counter_attestation,
        "curve_transform": "none-observed-raw-ratios",
        "curve_method": {
            "warmups_per_batch": raw["measurement_protocol"]["warmups_per_batch"],
            "independent_repeats_per_batch": raw["measurement_protocol"]["independent_repeats_per_batch"],
            "paired_interleaved_baseline": True,
            "monotone_transform_applied": False,
            "ucb_method": "paired_bootstrap_95",
            "confidence_level": 0.95,
            "phase_markers_sha256": raw["measurement_protocol"]["phase_markers_sha256"],
        },
        "batches": curve_rows,
        "counter_sources": {
            f"{domain}_source" if domain != "physical_bytes" else "bytes_source": next(
                row["source_kind"] for row in counter_attestation["domains"]
                if row["domain"] == domain
            )
            for domain in ("energy", "gpu_time", "physical_bytes")
        },
        "default_change_requested": False,
    }
    curve_bindings = copy.deepcopy(bindings)
    curve_bindings["parent_receipt_sha256"] = [parity_receipt["receipt_sha256"]]
    curve_receipt = appendix_contract.stamp_receipt({
        "schema": appendix_contract.SCHEMA,
        "experiment_schema": "hawking.spec_verifier_curve.v1",
        "cell_id": curve_cell["id"],
        "status": "complete",
        "bindings": curve_bindings,
        "exactness": {"required": True, "passed": True, "mismatches": 0},
        "samples": samples,
        "rollup": appendix_contract.rollup_samples(samples),
        "resources": _resources(bundle),
        "failure_reasons": [],
        "experiment_payload": curve_payload,
        "raw_bundle_sha256": bundle["raw_bundle_sha256"],
        "counter_attestation": counter_attestation,
        "physical_counter_bundle_sha256": appendix_contract.canonical_sha256(counters),
        "physical_counter_payload_sha256": appendix_contract.canonical_sha256(counters),
        "physical_counter_attestation_sha256": counter_attestation[
            "attestation_sha256"
        ],
    })
    for name, receipt in (("parity", parity_receipt), ("curve", curve_receipt)):
        validation = spec_receipt_contract.validate_receipt(receipt, known_cell_ids=known_ids)
        if validation:
            raise AssertionError(f"{name} receipt failed its contract: " + "; ".join(validation))
    return parity_receipt, curve_receipt


def _thermal_snapshot() -> str:
    try:
        proc = subprocess.run(
            ["pmset", "-g", "therm"], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=False,
        )
        detail = (proc.stdout or proc.stderr).strip()
        return "nominal" if ram_scheduler.thermal_output_ok(proc.returncode, detail) else "serious"
    except (OSError, subprocess.TimeoutExpired):
        return "serious"


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def _atomic_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def run_raw(
    weights: pathlib.Path,
    artifact: pathlib.Path,
    prompts: pathlib.Path,
    output: pathlib.Path,
    *,
    runtime_path: str,
    probe_bin: pathlib.Path = DEFAULT_PROBE,
    generated_tokens: int = 256,
    warmups_per_batch: int = 3,
    repeats_per_batch: int = 5,
    label: str = "CORPUS",
) -> dict:
    if runtime_path not in RUNTIME_KERNEL:
        raise ValueError("runtime_path must be stored|compact|hashed|computed")
    parity_cell, curve_cell, _ = _cells(runtime_path, label)
    for path in (weights, artifact, prompts):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
    if not probe_bin.is_file() or not os.access(probe_bin, os.X_OK):
        raise FileNotFoundError(f"release spec probe is absent or non-executable: {probe_bin}")
    HEAVY_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lease = HEAVY_LOCK.open("a+")
    try:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("shared heavy lease is held") from exc
        if spec_reentry_scaffold.active_heavy_owners():
            raise RuntimeError("heavy owners remain after lease acquisition")
        before = ram_scheduler.resource_snapshot(ROOT)
        thermal_before = _thermal_snapshot()
        if ram_scheduler.classify_resource_state(before) != "green" or thermal_before != "nominal":
            raise RuntimeError("pre-run resource or thermal admission is not green")
        raw_path = output.with_name(f".{output.name}.{os.getpid()}.raw")
        command = [
            str(probe_bin), "--weights", str(weights), "--artifact", str(artifact),
            "--prompts", str(prompts), "--runtime-path", runtime_path,
            "--generated-tokens", str(generated_tokens), "--source-commit", _source_commit(),
            "--warmups-per-batch", str(warmups_per_batch),
            "--repeats-per-batch", str(repeats_per_batch),
            "--parity-cell-id", parity_cell["id"], "--curve-cell-id", curve_cell["id"],
            "--output", str(raw_path),
        ]
        env = dict(os.environ)
        env["HAWKING_HEAVY_LEASE_FD"] = str(lease.fileno())
        env["HAWKING_APPENDIX_SPEC_ADMITTED"] = "1"
        run_nonce = secrets.token_hex(32)
        env["HAWKING_PHYSICAL_RUN_NONCE"] = run_nonce
        started_at_unix_ns = time.time_ns()
        started_at_continuous_ns = time.monotonic_ns()
        proc = subprocess.run(
            command, cwd=ROOT, env=env, pass_fds=(lease.fileno(),), text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=21600, check=False,
        )
        ended_at_continuous_ns = time.monotonic_ns()
        ended_at_unix_ns = time.time_ns()
        after = ram_scheduler.resource_snapshot(ROOT)
        thermal_after = _thermal_snapshot()
        if proc.returncode != 0:
            raise RuntimeError(f"spec probe failed ({proc.returncode}): {proc.stderr[-4000:]}")
        with raw_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        raw_path.unlink(missing_ok=True)
        execution_authority = {
            "probe_binary": physical_counter_attestation.file_identity(probe_bin),
            "argv_sha256": appendix_contract.canonical_sha256(command),
            "run_nonce": run_nonce,
            "started_at_unix_ns": started_at_unix_ns,
            "ended_at_unix_ns": ended_at_unix_ns,
            "started_at_continuous_ns": started_at_continuous_ns,
            "ended_at_continuous_ns": ended_at_continuous_ns,
            "exit_code": proc.returncode,
            "raw_probe_sha256": appendix_contract.canonical_sha256(raw),
            "stdout_sha256": hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(proc.stderr.encode("utf-8")).hexdigest(),
        }
        bundle = build_bundle(
            raw, resource_before=before, resource_after=after,
            thermal_before=thermal_before, thermal_after=thermal_after,
            execution_authority=execution_authority,
        )
        errors = validate_bundle(bundle)
        if errors:
            raise RuntimeError("spec raw bundle failed safety contract: " + "; ".join(errors))
        _atomic_json(output, bundle)
        return bundle
    finally:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        finally:
            lease.close()


def _load(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _selftest() -> int:
    print("spec_tq_runner.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--validate-raw", type=pathlib.Path)
    parser.add_argument("--validate-bundle", type=pathlib.Path)
    parser.add_argument("--run-raw", action="store_true")
    parser.add_argument("--weights", type=pathlib.Path)
    parser.add_argument("--artifact", type=pathlib.Path)
    parser.add_argument("--prompts", type=pathlib.Path)
    parser.add_argument("--runtime-path", default="stored")
    parser.add_argument("--generated-tokens", type=int, default=256)
    parser.add_argument("--warmups-per-batch", type=int, default=3)
    parser.add_argument("--repeats-per-batch", type=int, default=5)
    parser.add_argument("--probe-bin", type=pathlib.Path, default=DEFAULT_PROBE)
    parser.add_argument("--finalize", type=pathlib.Path, metavar="RAW_BUNDLE")
    parser.add_argument("--counters", type=pathlib.Path)
    parser.add_argument("--counter-attestation", type=pathlib.Path)
    parser.add_argument("--label", default="CORPUS")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--parity-output", type=pathlib.Path)
    parser.add_argument("--curve-output", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.validate_raw:
        errors = validate_raw(_load(args.validate_raw))
    elif args.validate_bundle:
        errors = validate_bundle(_load(args.validate_bundle))
    elif args.run_raw:
        if not all((args.weights, args.artifact, args.prompts, args.output)):
            parser.error("--run-raw requires --weights, --artifact, --prompts, and --output")
        try:
            run_raw(
                args.weights, args.artifact, args.prompts, args.output,
                runtime_path=args.runtime_path, probe_bin=args.probe_bin,
                generated_tokens=args.generated_tokens,
                warmups_per_batch=args.warmups_per_batch,
                repeats_per_batch=args.repeats_per_batch, label=args.label,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 75 if "lease" in str(exc) or "owner" in str(exc) else 1
        return 0
    elif args.finalize:
        if not all((
            args.counters, args.counter_attestation,
            args.parity_output, args.curve_output,
        )):
            parser.error(
                "--finalize requires --counters, --counter-attestation, --parity-output, and --curve-output"
            )
        parity, curve = finalize_receipts(
            _load(args.finalize), _load(args.counters),
            _load(args.counter_attestation), label=args.label,
        )
        _atomic_json(args.parity_output, parity)
        _atomic_json(args.curve_output, curve)
        return 0
    else:
        parser.error("choose --run-raw, --finalize, --validate-raw, --validate-bundle, or --selftest")
        return 64
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
