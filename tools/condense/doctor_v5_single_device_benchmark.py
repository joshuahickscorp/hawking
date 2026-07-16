#!/usr/bin/env python3.12
"""Fail-closed benchmark and ETA contract for Doctor V5 single-device sprinting.

Component microbenchmarks are useful for diagnosis, but their speedups overlap.
Only an owner-free, real-artifact, full-stack A/B receipt may change the official
ETA.  This module performs no benchmark itself and is not imported by the live
queue; it validates immutable receipts and computes a conservative projection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
import sys
from typing import Any

import doctor_v5_production_eta as eta_contract


ROOT = Path(__file__).resolve().parents[2]
PAIR_SCHEMA = "hawking.doctor_v5_single_device_benchmark.v1"
PROJECTION_SCHEMA = "hawking.doctor_v5_single_device_projection.v2"
THRESHOLD_SCHEMA = "hawking.doctor_v5_single_device_threshold.v2"
PRODUCTION_AUTHORITY_SCHEMA = "hawking.doctor_v5_benchmark_authority.v1"
MAX_JSON_BYTES = 64 * 1024 * 1024
PRODUCTION_SCOPE = "production-owner-free-real-artifact"
SYNTHETIC_SCOPE = "synthetic-cheap-gate-only"
REQUIRED_COMPONENTS = (
    "qualified-thread-profile",
    "ordered-read-rht-encode-write",
    "cross-shard-finalize-prepare-window",
    "shared-rate-branch-preprocessing",
    "elastic-phase-admission",
    "native-pgo-io-profile",
    "host-sprint-isolation",
    "controlled-swap-shock-absorber",
    "phase-aware-remaining-scratch",
)
OPTIONAL_COMPONENTS = ("metal-rht-preprocessing",)
DOCTOR_WORKLOAD_SEGMENT = "sub-120b-doctor"
STRICT_TARGET_CONTRACT = (
    "projected_sub_120b_upper_bound_days_must_be_strictly_less_than_target_days"
)
STRICT_SPEEDUP_RELATION = (
    "measured_speedup_must_be_strictly_greater_than_threshold"
)
# Structural receipt validation is implemented, but no locally trusted runner
# or hardware-backed attestation root exists yet.  Therefore no caller-supplied
# production receipt is allowed to promote an ETA, even when internally sealed.
TRUSTED_PRODUCTION_PROMOTION_AVAILABLE = False
EXPECTED_TIERS = ("3B", "7B", "14B", "32B", "72B")
EXPECTED_RATES = ("0.1", "0.25", "0.33", "0.5", "0.55", "0.8",
                  "1", "2", "3", "4")
EXPECTED_BRANCHES = ("codec_control", "doctor_conditional", "doctor_full",
                     "doctor_static")


class SprintBenchmarkError(RuntimeError):
    """A benchmark identity, parity, lifecycle, or claim boundary failed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: row for key, row in value.items() if key != field}


def _self_hash_matches(value: Any, field: str) -> bool:
    if not isinstance(value, dict) or not _valid_sha256(value.get(field)):
        return False
    try:
        return value[field] == _hash_value(_without(value, field))
    except (TypeError, ValueError):
        return False


def _read_json(path: Path) -> dict[str, Any]:
    try:
        before = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(before.st_mode) \
                or before.st_size > MAX_JSON_BYTES:
            raise SprintBenchmarkError(f"unsafe JSON artifact: {path}")
        raw = path.read_bytes()
        after = path.lstat()
        identity = lambda info: (info.st_dev, info.st_ino, info.st_size,
                                 info.st_mtime_ns)
        if identity(before) != identity(after) or len(raw) != after.st_size:
            raise SprintBenchmarkError(f"JSON artifact changed while reading: {path}")
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SprintBenchmarkError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SprintBenchmarkError(f"JSON root is not an object: {path}")
    return value


def _finite_positive(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) \
        and math.isfinite(float(value)) and float(value) > 0


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 \
        and all(character in "0123456789abcdef" for character in value)


def _artifact_identity(row: Any) -> bool:
    return isinstance(row, dict) and set(row) == {"sha256", "bytes"} \
        and isinstance(row.get("sha256"), str) and len(row["sha256"]) == 64 \
        and all(char in "0123456789abcdef" for char in row["sha256"]) \
        and not isinstance(row.get("bytes"), bool) \
        and isinstance(row.get("bytes"), int) and row["bytes"] >= 0


def _authority_reference(authority: dict[str, Any]) -> dict[str, Any]:
    payload = _canonical(authority)
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _production_authority_errors(authority: Any, *, repeats: int,
                                 baseline: list[Any], candidate: list[Any],
                                 production_eta_sha256: str | None = None) -> list[str]:
    if not isinstance(authority, dict) \
            or authority.get("schema") != PRODUCTION_AUTHORITY_SCHEMA:
        return ["production benchmark authority schema is absent"]
    errors: list[str] = []
    if not _self_hash_matches(authority, "authority_sha256"):
        errors.append("production benchmark authority hash mismatch")
    if not _valid_sha256(authority.get("production_eta_sha256")) \
            or authority.get("workload_segment") != DOCTOR_WORKLOAD_SEGMENT \
            or authority.get("selection_frozen_before_execution") is not True \
            or authority.get("canonical_baseline_source_bound") is not True \
            or authority.get("every_remaining_workload_class_represented") is not True:
        errors.append("production ETA/workload selection authority is incomplete")
    if production_eta_sha256 is not None \
            and authority.get("production_eta_sha256") != production_eta_sha256:
        errors.append("production benchmark authority targets a different ETA snapshot")
    matrix = authority.get("representative_matrix")
    if not isinstance(matrix, dict) \
            or matrix.get("tiers") != list(EXPECTED_TIERS) \
            or matrix.get("rates") != list(EXPECTED_RATES) \
            or matrix.get("branches") != list(EXPECTED_BRANCHES) \
            or matrix.get("matrix_frozen_before_execution") is not True \
            or not _valid_sha256(matrix.get("remaining_cell_ids_sha256")):
        errors.append("production representative workload matrix is incomplete")
    if isinstance(matrix, dict):
        slices = matrix.get("representative_slices")
        expected_combinations = [
            (tier, rate, branch) for tier in EXPECTED_TIERS
            for rate in EXPECTED_RATES for branch in EXPECTED_BRANCHES
        ]
        slice_fields = {
            "tier", "rate", "branch", "source_cell_identity_sha256",
            "slice_artifact",
        }
        if not isinstance(slices, list) \
                or len(slices) != len(expected_combinations) \
                or any(not isinstance(row, dict) or set(row) != slice_fields
                       for row in slices):
            errors.append("production representative slice matrix is invalid")
        else:
            combinations = [(row.get("tier"), row.get("rate"), row.get("branch"))
                            for row in slices]
            identities = [row.get("source_cell_identity_sha256") for row in slices]
            if combinations != expected_combinations \
                    or any(not _valid_sha256(value) for value in identities) \
                    or len(set(identities)) != len(identities) \
                    or any(not _artifact_identity(row.get("slice_artifact"))
                           for row in slices) \
                    or matrix.get("representative_slices_sha256") \
                    != _hash_value(slices) \
                    or matrix.get("remaining_cell_ids_sha256") \
                    != _hash_value(identities):
                errors.append("production representative slice bindings differ")
    workload = authority.get("workload_manifest")
    base_program = authority.get("baseline_program")
    trial_program = authority.get("candidate_program")
    runner = authority.get("benchmark_runner")
    if any(not _artifact_identity(value) for value in (
            workload, base_program, trial_program, runner)):
        errors.append("production workload/program/runner authority is invalid")
    base_invocations = authority.get("baseline_invocation_sha256s")
    trial_invocations = authority.get("candidate_invocation_sha256s")
    if not isinstance(base_invocations, list) or len(base_invocations) != repeats \
            or not isinstance(trial_invocations, list) \
            or len(trial_invocations) != repeats \
            or any(not _valid_sha256(value)
                   for value in [*base_invocations, *trial_invocations]):
        errors.append("production invocation authority coverage is invalid")
    owner_rows = authority.get("owner_inventory_receipts")
    if not isinstance(owner_rows, list) or len(owner_rows) != repeats * 2:
        errors.append("production per-run owner inventory coverage is incomplete")
        owner_rows = []
    owner_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    owner_fields = {
        "role", "repeat_index", "owner_free", "active_heavy_owner_count",
        "before", "after",
    }
    for row in owner_rows:
        if not isinstance(row, dict):
            errors.append("production owner inventory row is invalid/duplicated")
            continue
        role, repeat_index = row.get("role"), row.get("repeat_index")
        if set(row) != owner_fields \
                or not isinstance(role, str) \
                or role not in {"baseline", "candidate"} \
                or isinstance(repeat_index, bool) \
                or not isinstance(repeat_index, int) \
                or not 0 <= repeat_index < repeats:
            errors.append("production owner inventory row is invalid/duplicated")
            continue
        key = (role, repeat_index)
        if key in owner_by_key \
                or row.get("active_heavy_owner_count") != 0 \
                or row.get("owner_free") is not True \
                or not _artifact_identity(row.get("before")) \
                or not _artifact_identity(row.get("after")):
            errors.append("production owner inventory row is invalid/duplicated")
            continue
        owner_by_key[key] = row
    execution_order = authority.get("execution_order")
    expected_order = {f"baseline:{index}" for index in range(repeats)} | {
        f"candidate:{index}" for index in range(repeats)
    }
    if not isinstance(execution_order, list) \
            or len(execution_order) != repeats * 2 \
            or any(not isinstance(value, str) for value in execution_order) \
            or set(execution_order) != expected_order \
            or execution_order == sorted(execution_order):
        errors.append("production execution order is absent, incomplete, or not interleaved")
    frozen_at = authority.get("frozen_at_epoch_ns")
    if isinstance(frozen_at, bool) or not isinstance(frozen_at, int) or frozen_at <= 0:
        errors.append("production authority freeze time is invalid")
        frozen_at = None
    for index, run in enumerate(baseline):
        if not isinstance(run, dict):
            continue
        owner = owner_by_key.get(("baseline", index))
        invocation_mismatch = (
            isinstance(base_invocations, list)
            and index < len(base_invocations)
            and run.get("invocation_sha256") != base_invocations[index]
        )
        if (run.get("program") != base_program
                or run.get("input_bundle") != workload
                or run.get("benchmark_runner") != runner
                or invocation_mismatch
                or not isinstance(owner, dict)
                or run.get("owner_inventory_sha256") != _hash_value(owner)
                or frozen_at is None
                or not isinstance(run.get("started_at_epoch_ns"), int)
                or run.get("started_at_epoch_ns", 0) <= frozen_at):
            errors.append(f"baseline run {index} differs from frozen production authority")
    for index, run in enumerate(candidate):
        if not isinstance(run, dict):
            continue
        owner = owner_by_key.get(("candidate", index))
        invocation_mismatch = (
            isinstance(trial_invocations, list)
            and index < len(trial_invocations)
            and run.get("invocation_sha256") != trial_invocations[index]
        )
        if (run.get("program") != trial_program
                or run.get("input_bundle") != workload
                or run.get("benchmark_runner") != runner
                or invocation_mismatch
                or not isinstance(owner, dict)
                or run.get("owner_inventory_sha256") != _hash_value(owner)
                or frozen_at is None
                or not isinstance(run.get("started_at_epoch_ns"), int)
                or run.get("started_at_epoch_ns", 0) <= frozen_at):
            errors.append(f"candidate run {index} differs from frozen production authority")
    return errors


def _run_errors(run: Any, *, expected_role: str, expected_repeat: int) -> list[str]:
    if not isinstance(run, dict):
        return [f"{expected_role} run is not an object"]
    errors: list[str] = []
    if run.get("role") != expected_role or run.get("repeat_index") != expected_repeat:
        errors.append(f"{expected_role} run order differs")
    if run.get("status") != "complete" or run.get("exit_code") != 0 \
            or run.get("skipped") is not False:
        errors.append(f"{expected_role} run is incomplete, failed, or skipped")
    if run.get("source_files_deleted") is not False \
            or run.get("runtime_defaults_changed") is not False:
        errors.append(f"{expected_role} run violates lifecycle isolation")
    exercised = run.get("exercised_components")
    if not isinstance(exercised, list) \
            or any(not isinstance(value, str) or not value for value in exercised) \
            or exercised != sorted(set(exercised)):
        errors.append(f"{expected_role} exercised component list is invalid")
    for field in ("program", "benchmark_runner", "input_bundle", "output_bundle",
                  "receipt_bundle"):
        if not _artifact_identity(run.get(field)):
            errors.append(f"{expected_role} {field} identity is invalid")
    for field in ("invocation_sha256", "semantic_contract_sha256"):
        value = run.get(field)
        if not isinstance(value, str) or len(value) != 64 \
                or any(char not in "0123456789abcdef" for char in value):
            errors.append(f"{expected_role} {field} is invalid")
    if "started_at_epoch_ns" in run or "ended_at_epoch_ns" in run:
        started, ended = run.get("started_at_epoch_ns"), run.get("ended_at_epoch_ns")
        if isinstance(started, bool) or not isinstance(started, int) or started <= 0 \
                or isinstance(ended, bool) or not isinstance(ended, int) \
                or ended <= started:
            errors.append(f"{expected_role} production timing identity is invalid")
    for field in ("wall_seconds", "cpu_seconds", "peak_rss_bytes"):
        if not _finite_positive(run.get(field)):
            errors.append(f"{expected_role} {field} is invalid")
    scratch = run.get("scratch_peak_bytes")
    if isinstance(scratch, bool) or not isinstance(scratch, int) or scratch < 0:
        errors.append(f"{expected_role} scratch_peak_bytes is invalid")
    for field in ("disk_free_start_bytes", "disk_free_end_bytes"):
        value = run.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            errors.append(f"{expected_role} {field} is invalid")
    gpu = run.get("gpu_seconds")
    if isinstance(gpu, bool) or not isinstance(gpu, (int, float)) \
            or not math.isfinite(float(gpu)) or gpu < 0:
        errors.append(f"{expected_role} gpu_seconds is invalid")
    for field in ("swap_start_mb", "swap_end_mb"):
        value = run.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or value < 0:
            errors.append(f"{expected_role} {field} is invalid")
    if run.get("memory_pressure_start") != "normal" \
            or run.get("memory_pressure_end") != "normal" \
            or not isinstance(run.get("thermal_start"), str) \
            or run.get("thermal_start") not in {"nominal", "fair"} \
            or not isinstance(run.get("thermal_end"), str) \
            or run.get("thermal_end") not in {"nominal", "fair"}:
        errors.append(f"{expected_role} pressure/thermal envelope is inadmissible")
    return errors


def build_receipt(*, scope: str, components: list[str], baseline_runs: list[dict[str, Any]],
                  candidate_runs: list[dict[str, Any]], environment: dict[str, Any],
                  full_stack_end_to_end: bool,
                  production_authority: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build and validate one paired A/B receipt without executing either arm."""
    document: dict[str, Any] = {
        "schema": PAIR_SCHEMA,
        "scope": scope,
        "components": components,
        "full_stack_end_to_end": full_stack_end_to_end,
        "environment": environment,
        "baseline_runs": baseline_runs,
        "candidate_runs": candidate_runs,
        "source_files_deleted": False,
        "completed_evidence_mutated": False,
        "runtime_defaults_changed": False,
    }
    if production_authority is not None:
        document["production_authority"] = _authority_reference(production_authority)
    errors = validate_receipt(
        document, require_production=False, sealed=False,
        production_authority=production_authority,
    )
    if errors:
        raise SprintBenchmarkError("invalid benchmark receipt: " + "; ".join(errors))
    ratios = [float(base["wall_seconds"]) / float(candidate["wall_seconds"])
              for base, candidate in zip(baseline_runs, candidate_runs, strict=True)]
    summary = {
        "repeat_count": len(ratios),
        "baseline_median_seconds": statistics.median(
            float(row["wall_seconds"]) for row in baseline_runs
        ),
        "candidate_median_seconds": statistics.median(
            float(row["wall_seconds"]) for row in candidate_runs
        ),
        "paired_speedup_median": statistics.median(ratios),
        "paired_speedup_conservative": min(ratios),
        "all_outputs_exact": all(
            base["output_bundle"] == candidate["output_bundle"]
            and base["receipt_bundle"] == candidate["receipt_bundle"]
            for base, candidate in zip(baseline_runs, candidate_runs, strict=True)
        ),
        "component_speedups_multiplied": False,
    }
    document["summary"] = summary
    document["receipt_sha256"] = _hash_value(document)
    final_errors = validate_receipt(
        document, require_production=False,
        production_authority=production_authority,
    )
    if final_errors:
        raise SprintBenchmarkError("generated receipt failed: " + "; ".join(final_errors))
    return document


def validate_receipt(document: Any, *, require_production: bool,
                     sealed: bool = True,
                     production_authority: dict[str, Any] | None = None,
                     production_eta_sha256: str | None = None) -> list[str]:
    if not isinstance(document, dict) or document.get("schema") != PAIR_SCHEMA:
        return ["benchmark receipt schema mismatch"]
    errors: list[str] = []
    if sealed and not _self_hash_matches(document, "receipt_sha256"):
        errors.append("benchmark receipt hash mismatch")
    scope = document.get("scope")
    if not isinstance(scope, str) \
            or scope not in {SYNTHETIC_SCOPE, PRODUCTION_SCOPE}:
        errors.append("benchmark scope is invalid")
    if require_production and scope != PRODUCTION_SCOPE:
        errors.append("official ETA requires a production owner-free receipt")
    if require_production and not TRUSTED_PRODUCTION_PROMOTION_AVAILABLE:
        errors.append(
            "official ETA promotion is disabled until a trusted physical runner "
            "and non-caller-declared campaign attestation are installed"
        )
    components = document.get("components")
    if not isinstance(components, list) or not components \
            or any(not isinstance(value, str) or not value for value in components) \
            or components != sorted(set(components)):
        errors.append("benchmark component list is not unique and canonical")
        components = []
    elif scope == PRODUCTION_SCOPE and not set(components).issubset(
            set(REQUIRED_COMPONENTS) | set(OPTIONAL_COMPONENTS)):
        errors.append("production benchmark declares an unknown sprint component")
    baseline, candidate = document.get("baseline_runs"), document.get("candidate_runs")
    if not isinstance(baseline, list) or not isinstance(candidate, list) \
            or not baseline or len(baseline) != len(candidate):
        errors.append("paired benchmark run counts differ or are empty")
        baseline, candidate = [], []
    minimum = 3 if scope == PRODUCTION_SCOPE else 1
    if len(baseline) < minimum:
        errors.append(f"benchmark scope requires at least {minimum} paired repeats")
    for index, row in enumerate(baseline):
        errors.extend(_run_errors(row, expected_role="baseline", expected_repeat=index))
    for index, row in enumerate(candidate):
        errors.extend(_run_errors(row, expected_role="candidate", expected_repeat=index))
        if isinstance(row, dict) and row.get("exercised_components") != components:
            errors.append("candidate did not exercise the declared component stack")
    for row in baseline:
        if isinstance(row, dict) and row.get("exercised_components") != []:
            errors.append("baseline unexpectedly exercised candidate components")
    for base, trial in zip(baseline, candidate):
        if not isinstance(base, dict) or not isinstance(trial, dict):
            continue
        if base.get("input_bundle") != trial.get("input_bundle") \
                or base.get("semantic_contract_sha256") \
                != trial.get("semantic_contract_sha256"):
            errors.append("paired arms do not share exact input/semantic authority")
        if base.get("output_bundle") != trial.get("output_bundle") \
                or base.get("receipt_bundle") != trial.get("receipt_bundle"):
            errors.append("paired arms are not byte/receipt exact")
    environment = document.get("environment")
    if not isinstance(environment, dict) \
            or not _valid_sha256(environment.get("machine_identity_sha256")) \
            or environment.get("same_machine_both_arms") is not True \
            or environment.get("randomized_interleaved_order") is not True:
        errors.append("paired environment authority is incomplete")
    if scope == PRODUCTION_SCOPE and (
            not isinstance(environment, dict)
            or environment.get("owner_free") is not True
            or environment.get("active_heavy_owner_count") != 0
            or environment.get("real_artifact") is not True
            or environment.get("warmup_complete") is not True
            or environment.get("physical_counters_recorded") is not True
            or environment.get("workload_segment") != DOCTOR_WORKLOAD_SEGMENT
            or document.get("full_stack_end_to_end") is not True):
        errors.append("production benchmark was not owner-free/physical/real-artifact")
    if scope == PRODUCTION_SCOPE:
        authority_ref = document.get("production_authority")
        if not _artifact_identity(authority_ref):
            errors.append("production benchmark lacks a frozen external authority reference")
        if production_authority is None:
            errors.append("production benchmark requires the external frozen authority")
        else:
            try:
                expected_authority_ref = _authority_reference(production_authority)
            except (TypeError, ValueError):
                expected_authority_ref = None
            if authority_ref != expected_authority_ref:
                errors.append("production benchmark authority reference mismatch")
            errors.extend(_production_authority_errors(
                production_authority, repeats=len(baseline), baseline=baseline,
                candidate=candidate,
                production_eta_sha256=production_eta_sha256,
            ))
    if document.get("source_files_deleted") is not False \
            or document.get("completed_evidence_mutated") is not False \
            or document.get("runtime_defaults_changed") is not False:
        errors.append("benchmark lifecycle boundary is weakened")
    if sealed:
        summary = document.get("summary")
        summary_fields = {
            "repeat_count", "baseline_median_seconds", "candidate_median_seconds",
            "paired_speedup_median", "paired_speedup_conservative",
            "all_outputs_exact", "component_speedups_multiplied",
        }
        if not isinstance(summary, dict) or set(summary) != summary_fields \
                or summary.get("all_outputs_exact") is not True \
                or summary.get("component_speedups_multiplied") is not False \
                or not _finite_positive(summary.get("paired_speedup_median")) \
                or not _finite_positive(summary.get("paired_speedup_conservative")):
            errors.append("benchmark summary is invalid or non-exact")
        elif baseline and candidate and all(
                isinstance(row, dict) and _finite_positive(row.get("wall_seconds"))
                for row in [*baseline, *candidate]):
            ratios = [float(base["wall_seconds"]) / float(trial["wall_seconds"])
                      for base, trial in zip(baseline, candidate, strict=True)]
            exact_summary = {
                "repeat_count": len(ratios),
                "baseline_median_seconds": statistics.median(
                    float(row["wall_seconds"]) for row in baseline
                ),
                "candidate_median_seconds": statistics.median(
                    float(row["wall_seconds"]) for row in candidate
                ),
                "paired_speedup_median": statistics.median(ratios),
                "paired_speedup_conservative": min(ratios),
                "all_outputs_exact": all(
                    base.get("output_bundle") == trial.get("output_bundle")
                    and base.get("receipt_bundle") == trial.get("receipt_bundle")
                    and _artifact_identity(base.get("output_bundle"))
                    and _artifact_identity(trial.get("output_bundle"))
                    and _artifact_identity(base.get("receipt_bundle"))
                    and _artifact_identity(trial.get("receipt_bundle"))
                    for base, trial in zip(baseline, candidate, strict=True)
                ),
                "component_speedups_multiplied": False,
            }
            numeric_fields = (
                "baseline_median_seconds", "candidate_median_seconds",
                "paired_speedup_median", "paired_speedup_conservative",
            )
            if summary.get("repeat_count") != exact_summary["repeat_count"] \
                    or summary.get("all_outputs_exact") \
                    is not exact_summary["all_outputs_exact"] \
                    or summary.get("component_speedups_multiplied") is not False \
                    or any(not _finite_positive(summary.get(field))
                           or not math.isclose(float(summary[field]),
                                               float(exact_summary[field]),
                                               rel_tol=0, abs_tol=1e-12)
                           for field in numeric_fields):
                errors.append("benchmark summary differs from paired runs")
    return errors


def build_projection(*, production_eta: dict[str, Any], receipt: dict[str, Any],
                     production_authority: dict[str, Any]) \
        -> dict[str, Any]:
    eta_errors = eta_contract.validate(production_eta, verify_freshness=True)
    if eta_errors:
        raise SprintBenchmarkError(
            "production ETA authority is invalid: " + "; ".join(eta_errors)
        )
    if production_eta.get("eta_blocked") is not False \
            or production_eta.get("status") \
            != "provisional-live-production-calibration":
        raise SprintBenchmarkError(
            "production ETA is blocked; no accelerated completion projection is permitted"
        )
    errors = validate_receipt(
        receipt, require_production=True,
        production_authority=production_authority,
        production_eta_sha256=production_eta["document_sha256"],
    )
    if errors:
        raise SprintBenchmarkError("receipt cannot drive ETA: " + "; ".join(errors))
    if receipt.get("full_stack_end_to_end") is not True \
            or not set(REQUIRED_COMPONENTS).issubset(receipt.get("components", [])):
        raise SprintBenchmarkError("ETA requires one full-stack receipt covering every lever")
    speedup = float(receipt["summary"]["paired_speedup_conservative"])
    if speedup <= 1:
        raise SprintBenchmarkError("full stack has no conservative measured speedup")
    if receipt.get("environment", {}).get("workload_segment") \
            != DOCTOR_WORKLOAD_SEGMENT:
        raise SprintBenchmarkError("receipt is not scoped to the sub-120B Doctor workload")
    base_sub = production_eta["sub_120b"]["seconds_range"]
    sub_days = [float(value) / speedup / 86_400 for value in base_sub]
    document = {
        "schema": PROJECTION_SCHEMA,
        "status": "production-sub-120b-receipt-calibrated",
        "eta_scope": "sub-120b-only",
        "production_eta_sha256": production_eta["document_sha256"],
        "benchmark_receipt_sha256": receipt["receipt_sha256"],
        "conservative_additional_speedup": speedup,
        "sub_120b_days_range": sub_days,
        "through_120b_available": False,
        "through_120b_days_range": None,
        "through_120b_plus_appendix_available": False,
        "through_120b_plus_appendix_days_range": None,
        "sub_120b_under_seven_days": max(sub_days) < 7,
        "strict_under_seven_days_contract": STRICT_TARGET_CONTRACT,
        "threshold_equality_is_sufficient": False,
        "full_campaign_under_seven_days": False,
        "component_speedups_multiplied": False,
        "gptoss_120b_speedup_credit": None,
        "appendix_speedup_credit": None,
        "sub_120b_speedup_transferable_to_gpt_oss_120b": False,
        "sub_120b_speedup_transferable_to_appendix": False,
        "unmeasured_segment_speedup_applied": False,
        "quality_or_rigor_discount_applied": False,
        "runtime_defaults_changed": False,
    }
    document["projection_sha256"] = _hash_value(document)
    return document


def build_threshold(*, production_eta: dict[str, Any], target_days: float = 7.0) \
        -> dict[str, Any]:
    """Describe required measured speed without granting any unmeasured credit."""
    if not _finite_positive(target_days):
        raise SprintBenchmarkError("target days must be finite and positive")
    eta_errors = eta_contract.validate(production_eta, verify_freshness=True)
    if eta_errors:
        raise SprintBenchmarkError(
            "production ETA authority is invalid: " + "; ".join(eta_errors)
        )
    if production_eta.get("eta_blocked") is not False \
            or production_eta.get("status") \
            != "provisional-live-production-calibration":
        blockers = production_eta.get("blockers")
        if not isinstance(blockers, list) or not blockers \
                or any(not isinstance(row, str) or not row for row in blockers):
            raise SprintBenchmarkError("blocked production ETA lacks valid blockers")
        document = {
            "schema": THRESHOLD_SCHEMA,
            "status": "unavailable-production-eta-blocked",
            "available": False,
            "eta_scope": "sub-120b-only",
            "production_eta_sha256": production_eta["document_sha256"],
            "target_days": float(target_days),
            "target_contract": STRICT_TARGET_CONTRACT,
            "strict_inequality": STRICT_SPEEDUP_RELATION,
            "threshold_equality_is_sufficient": False,
            "blockers": sorted(set(blockers)),
            "sub_120b_baseline_days_range": None,
            "sub_120b_required_additional_speedup_by_endpoint": None,
            "sub_120b_required_additional_speedup_for_entire_range": None,
            "unchanged_120b_plus_appendix_increment_days_by_endpoint": None,
            "full_campaign_required_sub_120b_speedup_by_endpoint_if_other_segments_unchanged": None,
            "full_campaign_entire_range_possible_with_sub_120b_speedup_only": False,
            "gpt_oss_120b_threshold_available": False,
            "appendix_threshold_available": False,
            "unmeasured_segment_speedup_applied": False,
            "component_speedups_multiplied": False,
        }
        document["threshold_sha256"] = _hash_value(document)
        return document
    sub_document = production_eta.get("sub_120b")
    seconds = sub_document.get("seconds_range") \
        if isinstance(sub_document, dict) else None
    if not isinstance(seconds, list) or len(seconds) != 2 \
            or any(isinstance(value, bool)
                   or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) or float(value) <= 0
                   for value in seconds):
        raise SprintBenchmarkError("production ETA ranges are invalid")
    base_sub = [float(value) / 86_400 for value in seconds]
    document = {
        "schema": THRESHOLD_SCHEMA,
        "status": "sub-120b-threshold-only",
        "available": True,
        "eta_scope": "sub-120b-only",
        "production_eta_sha256": production_eta["document_sha256"],
        "target_days": float(target_days),
        "target_contract": STRICT_TARGET_CONTRACT,
        "strict_inequality": STRICT_SPEEDUP_RELATION,
        "threshold_equality_is_sufficient": False,
        "sub_120b_baseline_days_range": base_sub,
        "sub_120b_required_additional_speedup_by_endpoint": [
            value / float(target_days) for value in base_sub
        ],
        "sub_120b_required_additional_speedup_for_entire_range": (
            max(base_sub) / float(target_days)
        ),
        "unchanged_120b_plus_appendix_increment_days_by_endpoint": None,
        "full_campaign_required_sub_120b_speedup_by_endpoint_if_other_segments_unchanged": None,
        "full_campaign_entire_range_possible_with_sub_120b_speedup_only": False,
        "gpt_oss_120b_threshold_available": False,
        "appendix_threshold_available": False,
        "unmeasured_segment_speedup_applied": False,
        "component_speedups_multiplied": False,
    }
    document["threshold_sha256"] = _hash_value(document)
    return document


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--require-production", action="store_true")
    validate.add_argument("--authority", type=Path)
    project = sub.add_parser("project")
    project.add_argument("--receipt", type=Path, required=True)
    project.add_argument("--production-eta", type=Path, required=True)
    project.add_argument("--authority", type=Path, required=True)
    threshold = sub.add_parser("threshold")
    threshold.add_argument("--production-eta", type=Path, required=True)
    threshold.add_argument("--target-days", type=float, default=7.0)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            receipt = _read_json(args.receipt)
            authority = _read_json(args.authority) if args.authority else None
            errors = validate_receipt(
                receipt, require_production=args.require_production,
                production_authority=authority,
            )
            print(json.dumps({"ok": not errors, "errors": errors}, indent=2,
                             sort_keys=True))
            return 0 if not errors else 2
        if args.command == "project":
            result = build_projection(
                production_eta=_read_json(args.production_eta),
                receipt=_read_json(args.receipt),
                production_authority=_read_json(args.authority),
            )
        else:
            result = build_threshold(
                production_eta=_read_json(args.production_eta),
                target_days=args.target_days,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, KeyError, TypeError, ValueError, SprintBenchmarkError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
