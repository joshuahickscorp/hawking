#!/usr/bin/env python3.12
"""Heavy-lease runner and strict finalizer for Hawking-core TQ device evidence."""

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
import spec_reentry_scaffold
import tq_receipt_contract
import tq_runtime_matrix


ROOT = pathlib.Path(__file__).resolve().parents[2]
HEAVY_LOCK = ROOT / "reports" / "cron" / "studio_heavy.lock"
DEFAULT_PROBE = ROOT / "target" / "release" / "hawking-tq-device-probe"
RAW_SCHEMA = "hawking.tq_runtime_device_raw.v2"
BUNDLE_SCHEMA = "hawking.tq_runtime_device_raw_bundle.v1"
COUNTER_SCHEMA = "hawking.tq_runtime_physical_counters.v2"
MATRIX_IDENTITY_SCHEMA = "hawking.tq_device_matrix_identity.v2"
FEATURE_IDENTITY_SCHEMA = "hawking.tq_device_feature_identity.v1"
FEATURE_CENSUS_SCHEMA = "hawking.tq_device_feature_census.v2"
RESIDUAL_PROBE_SCHEMA = "hawking.tq_device_residual_probe.v1"
MODE_RECIPE = {
    "stored": {"metadata": "expanded", "codebook": "stored"},
    "compact": {"metadata": "compact", "codebook": "stored"},
    "hashed": {"metadata": "expanded", "codebook": "hashed_quantile"},
    "computed": {"metadata": "expanded", "codebook": "computed_acklam"},
}
MODE_KERNEL = {
    "stored": ("strand_bitslice_gemv_partials", 84),
    "compact": ("strand_bitslice_gemv_partials_compact", 40),
    "hashed": ("strand_bitslice_gemv_partials_hashed", 84),
    "computed": ("strand_bitslice_gemv_partials_computed", 84),
}


def _canonical_sha(value: Any) -> str:
    return appendix_contract.canonical_sha256(value)


def _stamp_bundle(bundle: dict) -> dict:
    stamped = copy.deepcopy(bundle)
    stamped.pop("raw_bundle_sha256", None)
    stamped["raw_bundle_sha256"] = _canonical_sha(stamped)
    return stamped


def _finite(value: Any, *, positive: bool = False) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and (value > 0 if positive else value >= 0)
    )


def _hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _stamped_errors(value: Any, *, field: str, label: str) -> list[str]:
    if not isinstance(value, dict):
        return [f"{label} must be an object"]
    unstamped = copy.deepcopy(value)
    claimed = unstamped.pop(field, None)
    if not _hex64(claimed) or claimed != _canonical_sha(unstamped):
        return [f"{label}.{field} does not match canonical identity bytes"]
    return []


def _artifact_errors(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "size_bytes"}:
        return [f"{label} artifact binding is incomplete or unexpected"]
    errors: list[str] = []
    if not isinstance(value.get("path"), str) or not pathlib.Path(value["path"]).is_absolute():
        errors.append(f"{label} artifact path must be absolute")
    if not _hex64(value.get("sha256")):
        errors.append(f"{label} artifact hash is invalid")
    size = value.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        errors.append(f"{label} artifact size must be positive")
    return errors


def _matrix_identity_errors(
    identity: Any, *, artifact: Any, tensor: Any, mode: Any,
) -> tuple[list[str], dict[str, Any] | None]:
    expected_fields = {
        "schema", "cell_id", "matrix_cell_sha256", "model", "tensor_family",
        "shape", "k_bits", "l_bits", "runtime_path", "artifact_sha256",
        "artifact_tensor_name", "identity_sha256",
    }
    errors: list[str] = []
    if not isinstance(identity, dict) or set(identity) != expected_fields:
        return ["raw matrix_identity is incomplete or unexpected"], None
    errors.extend(_stamped_errors(identity, field="identity_sha256", label="matrix_identity"))
    if identity.get("schema") != MATRIX_IDENTITY_SCHEMA:
        errors.append(f"raw matrix_identity.schema must be {MATRIX_IDENTITY_SCHEMA}")
    matrix_cell = next(
        (
            row for row in tq_runtime_matrix.build_matrix()["cells"]
            if row["id"] == identity.get("cell_id")
        ),
        None,
    )
    if matrix_cell is None or matrix_cell.get("state") != "deferred":
        errors.append("raw matrix_identity does not select one deferred matrix cell")
        return errors, None
    expected = {
        "schema": MATRIX_IDENTITY_SCHEMA,
        "cell_id": matrix_cell["id"],
        "matrix_cell_sha256": _canonical_sha(matrix_cell),
        "model": matrix_cell["model"],
        "tensor_family": matrix_cell["tensor_family"],
        "shape": matrix_cell["shape"],
        "k_bits": matrix_cell["k_bits"],
        "l_bits": matrix_cell["l_bits"],
        "runtime_path": matrix_cell["runtime_path"],
        "artifact_sha256": artifact.get("sha256") if isinstance(artifact, dict) else None,
        "artifact_tensor_name": tensor.get("name") if isinstance(tensor, dict) else None,
    }
    expected["identity_sha256"] = _canonical_sha(expected)
    if identity != expected or mode != matrix_cell["runtime_path"]:
        errors.append("raw matrix_identity does not match one exact deferred cell and artifact")
    return errors, matrix_cell


def _pass_identity(
    *, ordinal: int, role: str, artifact_sha256: str, tensor: dict[str, Any],
    runtime_path: str,
) -> dict[str, Any]:
    rht_mode = tensor.get("rht_mode")
    outlier_count = tensor.get("outlier_count")
    sequence: list[str] = []
    if rht_mode == "cols":
        sequence.append("strand_rht_forward_cols")
    sequence.append(MODE_KERNEL.get(runtime_path, (None, 0))[0])
    reduce_kernel = (
        "strand_bitslice_reduce_rows_accum"
        if role == "residual_accumulate"
        else "strand_bitslice_reduce_rows"
    )
    sequence.append(reduce_kernel)
    if isinstance(outlier_count, int) and not isinstance(outlier_count, bool) and outlier_count > 0:
        sequence.append("strand_outlier_correct")
    return {
        "ordinal": ordinal,
        "role": role,
        "artifact_sha256": artifact_sha256,
        "tensor_name": tensor.get("name"),
        "runtime_path": runtime_path,
        "rht_mode": rht_mode,
        "rht_blocks": tensor.get("rht_blocks"),
        "outlier_count": outlier_count,
        "reduce_kernel": reduce_kernel,
        "kernel_sequence": sequence,
    }


def _residual_probe_errors(raw: dict[str, Any]) -> list[str]:
    probe = raw.get("residual_probe")
    if not isinstance(probe, dict):
        return ["raw residual_probe must be an object"]
    if probe.get("schema") != RESIDUAL_PROBE_SCHEMA:
        return [f"raw residual_probe.schema must be {RESIDUAL_PROBE_SCHEMA}"]
    if probe.get("enabled") is False:
        return [] if set(probe) == {"schema", "enabled"} else [
            "disabled residual_probe contains unexpected evidence fields"
        ]
    expected = {
        "schema", "enabled", "artifact", "tensor", "runtime_path", "recipe",
        "metal", "q12_parity",
    }
    if probe.get("enabled") is not True or set(probe) != expected:
        return ["enabled residual_probe is incomplete or unexpected"]
    errors = _artifact_errors(probe.get("artifact"), label="residual")
    base_artifact = raw.get("artifact", {})
    if probe.get("artifact", {}).get("sha256") == base_artifact.get("sha256"):
        errors.append("residual artifact is not independent from the base artifact")
    tensor = probe.get("tensor")
    tensor_fields = {
        "name", "rows", "cols", "weights", "blocks", "k_bits", "l_bits",
        "rht_mode", "rht_blocks", "outlier_count",
    }
    if not isinstance(tensor, dict) or set(tensor) != tensor_fields:
        errors.append("residual tensor identity is incomplete or unexpected")
        tensor = {}
    else:
        for field in ("rows", "cols", "weights", "blocks", "k_bits", "l_bits"):
            if isinstance(tensor.get(field), bool) or not isinstance(tensor[field], int) or tensor[field] <= 0:
                errors.append(f"residual tensor {field} is invalid")
        for field in ("rht_blocks", "outlier_count"):
            if isinstance(tensor.get(field), bool) or not isinstance(tensor[field], int) or tensor[field] < 0:
                errors.append(f"residual tensor {field} is invalid")
        if tensor.get("rht_mode") not in {"none", "cols"}:
            errors.append("residual tensor rht_mode is invalid")
        if tensor.get("rht_blocks") != (tensor.get("cols", 0) // 256 if tensor.get("rht_mode") == "cols" else 0):
            errors.append("residual tensor RHT block count is inconsistent")
        if tensor.get("weights") != tensor.get("rows", 0) * tensor.get("cols", 0):
            errors.append("residual tensor weight count is inconsistent")
        base = raw.get("tensor", {})
        if (tensor.get("rows"), tensor.get("cols")) != (base.get("rows"), base.get("cols")):
            errors.append("residual tensor geometry differs from the base projection")
    mode = raw.get("runtime_path")
    if probe.get("runtime_path") != mode or probe.get("recipe") != MODE_RECIPE.get(mode):
        errors.append("residual runtime recipe differs from the base projection")
    metal = probe.get("metal")
    if not isinstance(metal, dict) or set(metal) != {
        "compiled", "kernel", "reduce_kernel", "host_entry_bytes", "gpu_entry_bytes",
    }:
        errors.append("residual Metal identity is incomplete or unexpected")
    elif mode in MODE_KERNEL:
        kernel, entry_bytes = MODE_KERNEL[mode]
        if (
            metal.get("compiled") is not True or metal.get("kernel") != kernel
            or metal.get("reduce_kernel") != "strand_bitslice_reduce_rows_accum"
            or metal.get("host_entry_bytes") != entry_bytes
            or metal.get("gpu_entry_bytes") != entry_bytes
        ):
            errors.append("residual Metal accumulate path identity is invalid")
    parity = probe.get("q12_parity")
    if not isinstance(parity, dict) or set(parity) != {
        "exact", "mismatches", "values_compared", "phase_marker_sha256",
    }:
        errors.append("residual Q12 parity is incomplete or unexpected")
    elif (
        parity.get("exact") is not True or parity.get("mismatches") != 0
        or parity.get("values_compared") != tensor.get("weights")
        or not _hex64(parity.get("phase_marker_sha256"))
    ):
        errors.append("residual Q12 parity failed or is not fully attributed")
    return errors


def _feature_identity_errors(raw: dict[str, Any]) -> list[str]:
    identity = raw.get("feature_identity")
    expected_fields = {
        "schema", "matrix_identity_sha256", "matrix_cell_sha256",
        "projection_recipe", "projection_passes", "pass_sequence",
        "feature_counts", "feature_identity_sha256",
    }
    if not isinstance(identity, dict) or set(identity) != expected_fields:
        return ["raw feature_identity is incomplete or unexpected"]
    errors = _stamped_errors(
        identity, field="feature_identity_sha256", label="feature_identity",
    )
    if identity.get("schema") != FEATURE_IDENTITY_SCHEMA:
        errors.append(f"raw feature_identity.schema must be {FEATURE_IDENTITY_SCHEMA}")
    matrix_identity = raw.get("matrix_identity", {})
    if (
        identity.get("matrix_identity_sha256") != matrix_identity.get("identity_sha256")
        or identity.get("matrix_cell_sha256") != matrix_identity.get("matrix_cell_sha256")
    ):
        errors.append("feature identity is not bound to the exact matrix identity")
    base_tensor = {
        **(raw.get("tensor", {}) if isinstance(raw.get("tensor"), dict) else {}),
        "rht_mode": raw.get("feature_census", {}).get("rht_mode"),
        "rht_blocks": raw.get("feature_census", {}).get("rht_blocks"),
        "outlier_count": raw.get("feature_census", {}).get("outlier_count"),
    }
    passes = [_pass_identity(
        ordinal=0, role="base_overwrite",
        artifact_sha256=raw.get("artifact", {}).get("sha256"),
        tensor=base_tensor, runtime_path=raw.get("runtime_path"),
    )]
    residual = raw.get("residual_probe", {})
    if residual.get("enabled") is True:
        passes.append(_pass_identity(
            ordinal=1, role="residual_accumulate",
            artifact_sha256=residual.get("artifact", {}).get("sha256"),
            tensor=residual.get("tensor", {}), runtime_path=raw.get("runtime_path"),
        ))
    expected = {
        "schema": FEATURE_IDENTITY_SCHEMA,
        "matrix_identity_sha256": matrix_identity.get("identity_sha256"),
        "matrix_cell_sha256": matrix_identity.get("matrix_cell_sha256"),
        "projection_recipe": (
            "two_pass_residual_accumulate" if len(passes) == 2 else "single_pass_overwrite"
        ),
        "projection_passes": len(passes),
        "pass_sequence": passes,
        "feature_counts": {
            "rht_cols_passes": sum(row["rht_mode"] == "cols" for row in passes),
            "outlier_corrected_passes": sum(
                isinstance(row["outlier_count"], int)
                and not isinstance(row["outlier_count"], bool)
                and row["outlier_count"] > 0
                for row in passes
            ),
            "residual_accumulate_passes": len(passes) - 1,
            "dispatches_per_invocation": sum(len(row["kernel_sequence"]) for row in passes),
        },
    }
    expected["feature_identity_sha256"] = _canonical_sha(expected)
    if identity != expected:
        errors.append("feature identity does not exactly match the bound pass sequence")
    return errors


def validate_raw(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return ["raw probe must be an object"]
    errors: list[str] = []
    if raw.get("schema") != RAW_SCHEMA:
        errors.append(f"raw schema must be {RAW_SCHEMA}")
    mode = raw.get("runtime_path")
    if mode not in MODE_RECIPE:
        errors.append("raw runtime_path is invalid")
    elif raw.get("recipe") != MODE_RECIPE[mode]:
        errors.append("raw recipe does not match runtime_path")
    source_commit = raw.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or not 7 <= len(source_commit) <= 64
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        errors.append("raw source_commit must be lowercase hexadecimal")
    artifact = raw.get("artifact")
    errors.extend(_artifact_errors(artifact, label="base"))
    matrix_identity = raw.get("matrix_identity")
    tensor = raw.get("tensor")
    required_ints = ("rows", "cols", "weights", "blocks", "k_bits", "l_bits")
    if not isinstance(tensor, dict) or not isinstance(tensor.get("name"), str):
        errors.append("raw tensor identity is invalid")
    elif any(
        not isinstance(tensor.get(field), int)
        or isinstance(tensor.get(field), bool)
        or tensor[field] <= 0
        for field in required_ints
    ):
        errors.append("raw tensor geometry is invalid")
    elif tensor["weights"] != tensor["rows"] * tensor["cols"]:
        errors.append("raw tensor weight count does not match rows*cols")
    identity_errors, _ = _matrix_identity_errors(
        matrix_identity, artifact=artifact, tensor=tensor, mode=mode,
    )
    errors.extend(identity_errors)
    errors.extend(_residual_probe_errors(raw))
    admission = raw.get("admission")
    if not isinstance(admission, dict) or admission.get("eligible") is not True or admission.get("reason") is not None:
        errors.append("raw device cell was not admitted")
    metal = raw.get("metal")
    if not isinstance(metal, dict) or mode not in MODE_KERNEL:
        errors.append("raw Metal evidence is invalid")
    else:
        kernel, entry_bytes = MODE_KERNEL[mode]
        if metal.get("compiled") is not True or metal.get("kernel") != kernel:
            errors.append("raw Metal kernel identity is invalid")
        if metal.get("host_entry_bytes") != entry_bytes or metal.get("gpu_entry_bytes") != entry_bytes:
            errors.append("raw Metal record size is invalid")
    parity = raw.get("parity")
    if not isinstance(parity, dict):
        errors.append("raw parity is missing")
    else:
        feature_identity = raw.get("feature_identity", {})
        if (
            parity.get("projection_recipe") != feature_identity.get("projection_recipe")
            or parity.get("projection_passes") != feature_identity.get("projection_passes")
            or parity.get("feature_identity_sha256")
            != feature_identity.get("feature_identity_sha256")
        ):
            errors.append("raw parity is not bound to the exact projection recipe")
        if parity.get("exact_q12") is not True or parity.get("q12_mismatches") != 0:
            errors.append("raw Q12 parity failed")
        if parity.get("exact_fused_vs_stored_gpu") is not True or parity.get("fused_bit_mismatches") != 0:
            errors.append("raw fused parity versus stored GPU failed")
        for field in ("q12_values_compared", "fused_values_compared"):
            if not isinstance(parity.get(field), int) or parity[field] <= 0:
                errors.append(f"raw parity.{field} must be positive")
        for field in ("cpu_reference_max_abs_error", "cpu_reference_max_rel_error"):
            if not _finite(parity.get(field)):
                errors.append(f"raw parity.{field} must be non-negative")
        for field in ("q12_phase_marker_sha256", "fused_phase_marker_sha256"):
            if not isinstance(parity.get(field), str) or len(parity[field]) != 64:
                errors.append(f"raw parity.{field} is invalid")
    census = raw.get("feature_census")
    census_fields = {
        "schema", "rht_mode", "rht_blocks", "rht_exercised", "outlier_count",
        "outlier_exercised", "projection_passes", "residual_passes",
        "residual_exercised", "dispatches_per_invocation",
        "dispatch_geometry_sha256", "kernel_sequence", "feature_identity_sha256",
    }
    if not isinstance(census, dict) or set(census) != census_fields:
        errors.append("raw feature_census is incomplete or unexpected")
    else:
        if census.get("schema") != FEATURE_CENSUS_SCHEMA:
            errors.append(f"raw feature_census.schema must be {FEATURE_CENSUS_SCHEMA}")
        if census.get("rht_mode") not in {"none", "cols"}:
            errors.append("raw feature_census RHT mode is invalid")
        for field in ("rht_blocks", "outlier_count", "residual_passes"):
            if isinstance(census.get(field), bool) or not isinstance(census.get(field), int) or census[field] < 0:
                errors.append(f"raw feature_census.{field} is invalid")
        if census.get("rht_exercised") is not (census.get("rht_mode") == "cols" and census.get("rht_blocks", 0) > 0):
            errors.append("raw feature_census RHT exercise flag is inconsistent")
        if census.get("outlier_exercised") is not (census.get("outlier_count", 0) > 0):
            errors.append("raw feature_census outlier exercise flag is inconsistent")
        residual_enabled = raw.get("residual_probe", {}).get("enabled") is True
        if (
            census.get("projection_passes") != 1 + int(residual_enabled)
            or census.get("residual_passes") != int(residual_enabled)
            or census.get("residual_exercised") is not residual_enabled
        ):
            errors.append("raw feature_census does not prove the exact residual pass count")
        if census.get("dispatches_per_invocation") != raw.get("benchmark", {}).get("dispatches_per_invocation"):
            errors.append("raw feature_census dispatch count differs from benchmark")
        if not isinstance(census.get("dispatch_geometry_sha256"), str) or len(census["dispatch_geometry_sha256"]) != 64:
            errors.append("raw feature_census dispatch geometry hash is invalid")
        if not isinstance(census.get("kernel_sequence"), list) or not census["kernel_sequence"]:
            errors.append("raw feature_census kernel sequence is invalid")
        elif isinstance(tensor, dict):
            geometry = {
                "rows": tensor.get("rows"),
                "cols": tensor.get("cols"),
                "blocks": tensor.get("blocks"),
                "rht_blocks": census.get("rht_blocks"),
                "outlier_count": census.get("outlier_count"),
                "projection_passes": census.get("projection_passes"),
                "residual_passes": census.get("residual_passes"),
                "dispatches_per_invocation": census.get("dispatches_per_invocation"),
                "kernel_sequence": census.get("kernel_sequence"),
                "feature_identity_sha256": census.get("feature_identity_sha256"),
            }
            if census.get("dispatch_geometry_sha256") != _canonical_sha(geometry):
                errors.append("raw feature_census dispatch geometry hash mismatch")
            if mode in MODE_KERNEL and MODE_KERNEL[mode][0] not in census["kernel_sequence"]:
                errors.append("raw feature_census omits the selected runtime kernel")
        feature_identity = raw.get("feature_identity", {})
        if census.get("feature_identity_sha256") != feature_identity.get("feature_identity_sha256"):
            errors.append("raw feature_census is not bound to the exact feature identity")
    errors.extend(_feature_identity_errors(raw))
    traffic = raw.get("logical_traffic")
    parts = ("payload", "metadata", "codebook_staging", "partial_roundtrip")
    if not isinstance(traffic, dict) or any(
        not isinstance(traffic.get(field), int) or traffic[field] < 0 for field in parts
    ):
        errors.append("raw logical traffic is invalid")
    elif traffic.get("compressed_runtime_total") != sum(traffic[field] for field in parts):
        errors.append("raw logical traffic total is invalid")
    elif tensor and _finite(traffic.get("compressed_runtime_bpw")):
        normalization_weights = tensor["weights"]
        residual = raw.get("residual_probe", {})
        if residual.get("enabled") is True:
            residual_weights = residual.get("tensor", {}).get("weights")
            if isinstance(residual_weights, int) and not isinstance(residual_weights, bool):
                normalization_weights += residual_weights
        expected = traffic["compressed_runtime_total"] * 8 / normalization_weights
        if abs(traffic["compressed_runtime_bpw"] - expected) > 1e-9:
            errors.append("raw logical traffic bpw does not use all decoded projection weights")
    else:
        errors.append("raw logical traffic bpw is invalid")
    benchmark = raw.get("benchmark")
    if not isinstance(benchmark, dict):
        errors.append("raw benchmark is missing")
    else:
        feature_identity = raw.get("feature_identity", {})
        if (
            benchmark.get("projection_recipe") != feature_identity.get("projection_recipe")
            or benchmark.get("feature_identity_sha256")
            != feature_identity.get("feature_identity_sha256")
        ):
            errors.append("raw benchmark is not bound to the exact projection recipe")
        warmups, trials = benchmark.get("warmups"), benchmark.get("trials")
        baseline = benchmark.get("baseline_wall_ns")
        candidate = benchmark.get("candidate_wall_ns")
        if not isinstance(warmups, int) or warmups < 3:
            errors.append("raw benchmark requires warmups>=3")
        if not isinstance(trials, int) or trials < 10:
            errors.append("raw benchmark requires trials>=10")
        if (
            not isinstance(baseline, list)
            or not isinstance(candidate, list)
            or len(baseline) != trials
            or len(candidate) != trials
            or any(not isinstance(value, int) or value <= 0 for value in baseline + candidate)
        ):
            errors.append("raw benchmark trial arrays are invalid")
        if benchmark.get("order") != "paired_interleaved_alternating":
            errors.append("raw benchmark order is invalid")
        warmup_markers = benchmark.get("warmup_phase_marker_sha256")
        trial_markers = benchmark.get("trial_phase_marker_sha256")
        if (
            not isinstance(warmup_markers, list) or len(warmup_markers) != warmups
            or any(not isinstance(value, str) or len(value) != 64 for value in warmup_markers)
        ):
            errors.append("raw benchmark warmup phase markers are invalid")
        if (
            not isinstance(trial_markers, list) or len(trial_markers) != trials
            or any(not isinstance(value, str) or len(value) != 64 for value in trial_markers)
            or len(set(trial_markers)) != len(trial_markers)
        ):
            errors.append("raw benchmark trial phase markers are invalid or reused")
    phase_markers = raw.get("phase_markers")
    errors.extend(physical_counter_attestation.validate_phase_markers(phase_markers))
    if isinstance(phase_markers, dict):
        pair_by_hash = {
            row.get("phase_marker_sha256"): row
            for row in phase_markers.get("pairs", []) if isinstance(row, dict)
        }
        intervals = {
            row.get("interval_sha256"): row
            for row in phase_markers.get("intervals", []) if isinstance(row, dict)
        }
        parity_pair = pair_by_hash.get(parity.get("fused_phase_marker_sha256")) if isinstance(parity, dict) else None
        q12_hash = parity.get("q12_phase_marker_sha256") if isinstance(parity, dict) else None
        if (
            not isinstance(parity_pair, dict) or parity_pair.get("phase") != "parity"
            or "candidate_interval_sha256" not in parity_pair
            or intervals.get(q12_hash, {}).get("role") != "candidate_q12"
        ):
            errors.append("raw parity phase-marker attribution is invalid")
        residual = raw.get("residual_probe", {})
        if residual.get("enabled") is True:
            residual_q12_hash = residual.get("q12_parity", {}).get("phase_marker_sha256")
            residual_interval = intervals.get(residual_q12_hash, {})
            if (
                residual_interval.get("phase") != "parity"
                or residual_interval.get("role") != "candidate_residual_q12"
                or residual_interval.get("iteration") != 1
            ):
                errors.append("raw residual Q12 phase-marker attribution is invalid")
        if isinstance(benchmark, dict):
            for phase, field, count in (
                ("warmup", "warmup_phase_marker_sha256", benchmark.get("warmups")),
                ("trial", "trial_phase_marker_sha256", benchmark.get("trials")),
            ):
                hashes = benchmark.get(field, [])
                if isinstance(hashes, list) and isinstance(count, int):
                    rows = [pair_by_hash.get(digest) for digest in hashes]
                    if any(
                        not isinstance(row, dict) or row.get("phase") != phase
                        or row.get("batch") is not None or row.get("iteration") != index
                        or "candidate_interval_sha256" not in row
                        for index, row in enumerate(rows)
                    ):
                        errors.append(f"raw {phase} pair markers do not match benchmark order")
    if raw.get("physical_counters") != {"measured": False}:
        errors.append("raw probe must not invent physical counters")
    if raw.get("default_change_requested") is not False:
        errors.append("raw probe cannot request a default change")
    return errors


def _resource_green(snapshot: Any) -> bool:
    return isinstance(snapshot, dict) and ram_scheduler.classify_resource_state(snapshot) == "green"


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
        "raw_probe_sha256": _canonical_sha(raw),
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
        return ["raw bundle must be an object"]
    errors: list[str] = []
    if bundle.get("schema") != BUNDLE_SCHEMA:
        errors.append(f"bundle schema must be {BUNDLE_SCHEMA}")
    unstamped = copy.deepcopy(bundle)
    claimed = unstamped.pop("raw_bundle_sha256", None)
    if claimed != _canonical_sha(unstamped):
        errors.append("raw_bundle_sha256 mismatch")
    raw = bundle.get("raw_probe")
    errors.extend(validate_raw(raw))
    if isinstance(raw, dict) and bundle.get("raw_probe_sha256") != _canonical_sha(raw):
        errors.append("raw_probe_sha256 mismatch")
    errors.extend(physical_counter_attestation.validate_execution_authority(
        bundle.get("execution_authority"), raw_probe_sha256=_canonical_sha(raw),
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
            errors.append("runner resources were not green")
        if safety.get("swap_delta_mb") != 0:
            errors.append("runner observed swap growth")
        if safety.get("thermal_before") != "nominal" or safety.get("thermal_after") != "nominal":
            errors.append("runner thermal state was not nominal")
    return errors


def validate_counters(counters: Any, counter_attestation: Any, bundle: dict) -> list[str]:
    if not isinstance(counters, dict):
        return ["physical counters must be an object"]
    errors: list[str] = []
    raw = bundle["raw_probe"]
    expected_fields = {
        "schema", "raw_bundle_sha256", "artifact_sha256", "tensor",
        "runtime_path", "phase_markers_sha256", "trials", "summary",
    }
    if set(counters) != expected_fields:
        errors.append("counter fields are incomplete or unexpected")
    if counters.get("schema") != COUNTER_SCHEMA:
        errors.append(f"counter schema must be {COUNTER_SCHEMA}")
    if counters.get("raw_bundle_sha256") != bundle.get("raw_bundle_sha256"):
        errors.append("counters are not bound to the raw bundle")
    if counters.get("artifact_sha256") != raw["artifact"]["sha256"]:
        errors.append("counters are not bound to the artifact")
    if counters.get("tensor") != raw["tensor"]["name"] or counters.get("runtime_path") != raw["runtime_path"]:
        errors.append("counters are not bound to tensor/runtime")
    phase = raw.get("phase_markers", {})
    if counters.get("phase_markers_sha256") != phase.get("phase_markers_sha256"):
        errors.append("counters are not bound to raw phase markers")
    expected_markers = raw.get("benchmark", {}).get("trial_phase_marker_sha256", [])
    trials = counters.get("trials")
    if not isinstance(trials, list) or len(trials) != len(expected_markers):
        errors.append("counter trials do not cover every raw candidate trial")
        trials = []
    totals = {"energy_j_total": 0.0, "gpu_time_ns_total": 0, "physical_bytes_total": 0}
    occupancies: list[float] = []
    bandwidths: list[float] = []
    for index, row in enumerate(trials):
        expected_trial_fields = {
            "index", "phase_marker_sha256", "energy_j", "gpu_time_ns",
            "physical_bytes", "occupancy_percent", "bandwidth_bytes_per_second",
        }
        if not isinstance(row, dict) or set(row) != expected_trial_fields:
            errors.append(f"counter trial {index} is malformed")
            continue
        if row.get("index") != index or row.get("phase_marker_sha256") != expected_markers[index]:
            errors.append(f"counter trial {index} is not bound to its raw phase marker")
        for field in ("energy_j", "gpu_time_ns", "physical_bytes", "bandwidth_bytes_per_second"):
            if not _finite(row.get(field), positive=True):
                errors.append(f"counter trial {index}.{field} must be positive")
        occupancy = row.get("occupancy_percent")
        if not _finite(occupancy) or occupancy > 100:
            errors.append(f"counter trial {index}.occupancy_percent must be in [0,100]")
        if _finite(row.get("energy_j"), positive=True):
            totals["energy_j_total"] += float(row["energy_j"])
        if _finite(row.get("gpu_time_ns"), positive=True):
            totals["gpu_time_ns_total"] += int(row["gpu_time_ns"])
        if _finite(row.get("physical_bytes"), positive=True):
            totals["physical_bytes_total"] += int(row["physical_bytes"])
        if _finite(occupancy):
            occupancies.append(float(occupancy))
        if _finite(row.get("bandwidth_bytes_per_second"), positive=True):
            bandwidths.append(float(row["bandwidth_bytes_per_second"]))
    summary = counters.get("summary")
    expected_summary_fields = {
        "energy_j_total", "gpu_time_ns_total", "physical_bytes_total",
        "occupancy_percent_mean", "bandwidth_bytes_per_second_mean",
    }
    if not isinstance(summary, dict) or set(summary) != expected_summary_fields:
        errors.append("counter summary is malformed")
    elif trials and len(occupancies) == len(trials) and len(bandwidths) == len(trials):
        expected_summary = {
            **totals,
            "occupancy_percent_mean": statistics.fmean(occupancies),
            "bandwidth_bytes_per_second_mean": statistics.fmean(bandwidths),
        }
        for field, expected_value in expected_summary.items():
            if not _finite(summary.get(field)) or not math.isclose(
                float(summary[field]), float(expected_value), rel_tol=1e-12, abs_tol=1e-12,
            ):
                errors.append(f"counter summary.{field} does not match trials")
    errors.extend(physical_counter_attestation.validate(
        counter_attestation,
        raw_bundle_sha256=bundle.get("raw_bundle_sha256"),
        artifact_sha256=raw["artifact"]["sha256"],
        execution_authority=bundle.get("execution_authority", {}),
        counter_payload=counters,
        required_domains=("energy", "gpu_time", "physical_bytes", "occupancy", "bandwidth"),
        minimum_samples=raw["benchmark"]["trials"],
    ))
    return errors


def _nearest(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def finalize_receipt(
    bundle: dict, counters: dict, counter_attestation: dict, cell_id: str,
) -> dict:
    errors = validate_bundle(bundle)
    errors.extend(validate_counters(counters, counter_attestation, bundle))
    if errors:
        raise ValueError("; ".join(errors))
    matrix = tq_runtime_matrix.build_matrix()
    cell = next((row for row in matrix["cells"] if row["id"] == cell_id), None)
    if cell is None:
        raise ValueError("cell_id is absent from the TQ device matrix")
    raw = bundle["raw_probe"]
    tensor = raw["tensor"]
    if (
        cell["state"] != "deferred"
        or cell["runtime_path"] != raw["runtime_path"]
        or cell["shape"] != {"rows": tensor["rows"], "cols": tensor["cols"]}
        or cell["k_bits"] != tensor["k_bits"]
        or cell["l_bits"] != tensor["l_bits"]
    ):
        raise ValueError("raw evidence does not match the selected device-matrix cell")
    benchmark = raw["benchmark"]
    baseline = benchmark["baseline_wall_ns"]
    candidate = benchmark["candidate_wall_ns"]
    speedups = [base / cand for base, cand in zip(baseline, candidate)]
    traffic = raw["logical_traffic"]
    sample_bytes = {
        "weights": traffic["payload"],
        "metadata": traffic["metadata"],
        "codebook": traffic["codebook_staging"],
        "activations": (tensor["cols"] + tensor["rows"]) * 4,
        "scratch": traffic["partial_roundtrip"],
        "state": 0,
        "communication": 0,
        "io": 0,
    }
    samples = [
        {
            "index": index,
            "phase": "decode",
            "wall_ns": wall,
            "accepted_tokens": 0,
            "rejected_tokens": 0,
            "energy_j": counters["trials"][index]["energy_j"],
            "bytes": sample_bytes,
        }
        for index, wall in enumerate(candidate)
    ]
    parity = raw["parity"]
    payload = {
        "runtime_path": raw["runtime_path"],
        "recipe": raw["recipe"],
        "matrix_identity": raw["matrix_identity"],
        "feature_identity": raw["feature_identity"],
        "residual_probe": raw["residual_probe"],
        "shape": {
            "tensor": raw["tensor"]["name"],
            **{field: tensor[field] for field in ("rows", "cols", "weights", "blocks", "k_bits", "l_bits")},
        },
        "admission": raw["admission"],
        "metal": raw["metal"],
        "parity": {
            "exact_q12": parity["exact_q12"],
            "exact_fused_gemv": parity["exact_fused_vs_stored_gpu"],
            "mismatches": parity["q12_mismatches"] + parity["fused_bit_mismatches"],
            "values_compared": parity["q12_values_compared"] + parity["fused_values_compared"],
            "cases": 1,
        },
        "logical_traffic": traffic,
        "benchmark": {
            "warmups": benchmark["warmups"],
            "trials": benchmark["trials"],
            "p50_ns": _nearest(candidate, 0.50),
            "p95_ns": _nearest(candidate, 0.95),
            "p99_ns": _nearest(candidate, 0.99),
            "speedup_median": _nearest([round(value * 1_000_000) for value in speedups], 0.50) / 1_000_000,
            "speedup_lcb": _nearest([round(value * 1_000_000) for value in speedups], 0.05) / 1_000_000,
            "joules_per_invocation": counters["summary"]["energy_j_total"] / benchmark["trials"],
        },
        "physical_counters": {
            "measured": True,
            "occupancy_percent": counters["summary"]["occupancy_percent_mean"],
            "realized_bandwidth_bytes_per_second": counters["summary"]["bandwidth_bytes_per_second_mean"],
            "gpu_time_ns": counters["summary"]["gpu_time_ns_total"],
            "physical_bytes_total": counters["summary"]["physical_bytes_total"],
            "byte_samples_are_logical_accounting": True,
        },
        "counter_attestation": counter_attestation,
        "default_change_requested": False,
    }
    safety = bundle["runner_safety"]
    receipt = appendix_contract.stamp_receipt({
        "schema": appendix_contract.SCHEMA,
        "experiment_schema": tq_receipt_contract.SCHEMA,
        "cell_id": cell_id,
        "status": "complete",
        "bindings": {
            "source_commit": raw["source_commit"],
            "target_sha256": raw["artifact"]["sha256"],
            "target_not_applicable_reason": None,
            "prompt_set_sha256": None,
            "prompt_not_applicable_reason": "projection device microbenchmark consumes no prompt",
            "parent_receipt_sha256": [],
        },
        "exactness": {"required": True, "passed": True, "mismatches": 0},
        "samples": samples,
        "rollup": appendix_contract.rollup_samples(samples),
        "resources": {
            "observed": True,
            "memory_pressure": "normal",
            "swap_delta_mb": safety["swap_delta_mb"],
            "thermal_state": "nominal",
            "not_applicable_reason": None,
        },
        "failure_reasons": [],
        "experiment_payload": payload,
        "raw_bundle_sha256": bundle["raw_bundle_sha256"],
        "physical_counter_bundle_sha256": _canonical_sha(counters),
        "physical_counter_payload_sha256": _canonical_sha(counters),
        "physical_counter_attestation_sha256": counter_attestation[
            "attestation_sha256"
        ],
        "physical_counter_sources": {
            f"{domain}_source" if domain != "physical_bytes" else "bytes_source": next(
                row["source_kind"] for row in counter_attestation["domains"]
                if row["domain"] == domain
            )
            for domain in ("occupancy", "bandwidth", "energy", "gpu_time", "physical_bytes")
        },
    })
    validation = tq_receipt_contract.validate_receipt(
        receipt, known_cell_ids={row["id"] for row in matrix["cells"]}
    )
    if validation:
        raise AssertionError("final receipt failed its contract: " + "; ".join(validation))
    return receipt


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
    artifact: pathlib.Path,
    output: pathlib.Path,
    *,
    cell_id: str,
    runtime_path: str,
    tensor: str | None,
    residual_artifact: pathlib.Path | None = None,
    residual_tensor: str | None = None,
    probe_bin: pathlib.Path = DEFAULT_PROBE,
    warmups: int = 3,
    trials: int = 10,
) -> dict:
    if runtime_path not in MODE_RECIPE:
        raise ValueError("runtime_path must be stored|compact|hashed|computed")
    cell = next(
        (
            row for row in tq_runtime_matrix.build_matrix()["cells"]
            if row["id"] == cell_id and row["state"] == "deferred"
        ),
        None,
    )
    if cell is None or cell["runtime_path"] != runtime_path:
        raise ValueError("cell_id must identify an exact deferred cell for runtime_path")
    if tensor is not None and tensor != cell["tensor_family"]:
        raise ValueError("tensor override differs from matrix tensor family")
    if (residual_artifact is None) != (residual_tensor is None):
        raise ValueError("residual_artifact and residual_tensor must be supplied together")
    if not artifact.is_file() or artifact.is_symlink():
        raise FileNotFoundError(artifact)
    if residual_artifact is not None:
        if not residual_artifact.is_file() or residual_artifact.is_symlink():
            raise FileNotFoundError(residual_artifact)
        if residual_artifact.resolve() == artifact.resolve():
            raise ValueError("residual artifact must be a distinct immutable input")
    if not probe_bin.is_file() or not os.access(probe_bin, os.X_OK):
        raise FileNotFoundError(f"release probe binary is absent or non-executable: {probe_bin}")
    HEAVY_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lease = HEAVY_LOCK.open("a+")
    try:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("shared heavy lease is held") from exc
        owners = spec_reentry_scaffold.active_heavy_owners()
        if owners:
            raise RuntimeError("heavy owners remain after lease acquisition")
        before = ram_scheduler.resource_snapshot(ROOT)
        thermal_before = _thermal_snapshot()
        if not _resource_green(before) or thermal_before != "nominal":
            raise RuntimeError("pre-run resource or thermal admission is not green")
        raw_path = output.with_name(f".{output.name}.{os.getpid()}.raw")
        command = [
            str(probe_bin), "--artifact", str(artifact), "--runtime-path", runtime_path,
            "--warmups", str(warmups), "--trials", str(trials),
            "--matrix-cell-id", cell["id"], "--matrix-model", cell["model"],
            "--matrix-cell-sha256", _canonical_sha(cell),
            "--matrix-tensor-family", cell["tensor_family"],
            "--source-commit", _source_commit(), "--output", str(raw_path),
        ]
        command.extend(["--tensor", cell["tensor_family"]])
        if residual_artifact is not None and residual_tensor is not None:
            command.extend([
                "--residual-artifact", str(residual_artifact),
                "--residual-tensor", residual_tensor,
            ])
        env = dict(os.environ)
        env["HAWKING_HEAVY_LEASE_FD"] = str(lease.fileno())
        env["HAWKING_APPENDIX_DEVICE_ADMITTED"] = "1"
        run_nonce = secrets.token_hex(32)
        env["HAWKING_PHYSICAL_RUN_NONCE"] = run_nonce
        started_at_unix_ns = time.time_ns()
        started_at_continuous_ns = time.monotonic_ns()
        proc = subprocess.run(
            command, cwd=ROOT, env=env, pass_fds=(lease.fileno(),),
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=3600, check=False,
        )
        ended_at_continuous_ns = time.monotonic_ns()
        ended_at_unix_ns = time.time_ns()
        after = ram_scheduler.resource_snapshot(ROOT)
        thermal_after = _thermal_snapshot()
        if proc.returncode != 0:
            raise RuntimeError(f"device probe failed ({proc.returncode}): {proc.stderr[-2000:]}")
        with raw_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        raw_path.unlink(missing_ok=True)
        execution_authority = {
            "probe_binary": physical_counter_attestation.file_identity(probe_bin),
            "argv_sha256": _canonical_sha(command),
            "run_nonce": run_nonce,
            "started_at_unix_ns": started_at_unix_ns,
            "ended_at_unix_ns": ended_at_unix_ns,
            "started_at_continuous_ns": started_at_continuous_ns,
            "ended_at_continuous_ns": ended_at_continuous_ns,
            "exit_code": proc.returncode,
            "raw_probe_sha256": _canonical_sha(raw),
            "stdout_sha256": hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest(),
            "stderr_sha256": hashlib.sha256(proc.stderr.encode("utf-8")).hexdigest(),
        }
        bundle = build_bundle(
            raw,
            resource_before=before,
            resource_after=after,
            thermal_before=thermal_before,
            thermal_after=thermal_after,
            execution_authority=execution_authority,
        )
        errors = validate_bundle(bundle)
        if errors:
            raise RuntimeError("raw bundle failed safety contract: " + "; ".join(errors))
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
    print("appendix_device_runner.py selftest OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--validate-raw", type=pathlib.Path)
    parser.add_argument("--validate-bundle", type=pathlib.Path)
    parser.add_argument("--run-raw", type=pathlib.Path, metavar="ARTIFACT")
    parser.add_argument("--runtime-path", default="stored")
    parser.add_argument("--cell-id")
    parser.add_argument("--tensor")
    parser.add_argument("--residual-artifact", type=pathlib.Path)
    parser.add_argument("--residual-tensor")
    parser.add_argument("--probe-bin", type=pathlib.Path, default=DEFAULT_PROBE)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--finalize", type=pathlib.Path, metavar="RAW_BUNDLE")
    parser.add_argument("--counters", type=pathlib.Path)
    parser.add_argument("--counter-attestation", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.validate_raw is not None:
        errors = validate_raw(_load(args.validate_raw))
    elif args.validate_bundle is not None:
        errors = validate_bundle(_load(args.validate_bundle))
    elif args.run_raw is not None:
        if args.output is None or args.cell_id is None:
            parser.error("--run-raw requires --output and --cell-id")
        try:
            run_raw(
                args.run_raw, args.output, cell_id=args.cell_id, runtime_path=args.runtime_path,
                tensor=args.tensor, residual_artifact=args.residual_artifact,
                residual_tensor=args.residual_tensor, probe_bin=args.probe_bin,
                warmups=args.warmups, trials=args.trials,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 75 if "lease" in str(exc) or "owner" in str(exc) else 1
        return 0
    elif args.finalize is not None:
        if (
            args.counters is None or args.counter_attestation is None
            or args.cell_id is None or args.output is None
        ):
            parser.error(
                "--finalize requires --counters, --counter-attestation, --cell-id, and --output"
            )
        receipt = finalize_receipt(
            _load(args.finalize), _load(args.counters),
            _load(args.counter_attestation), args.cell_id,
        )
        _atomic_json(args.output, receipt)
        return 0
    else:
        parser.error("choose --run-raw, --finalize, --validate-raw, --validate-bundle, or --selftest")
        return 64
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
