from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from physical_counter_fixtures import attestation, execution_authority, phase_markers
MODULE_PATH = CONDENSE / "appendix_device_runner.py"
SPEC = importlib.util.spec_from_file_location("appendix_device_runner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _cell() -> dict:
    return next(
        cell
        for cell in MODULE.tq_runtime_matrix.build_matrix()["cells"]
        if cell["state"] == "deferred" and cell["runtime_path"] == "compact"
    )


def _raw(cell: dict, *, residual: bool = True) -> dict:
    rows, cols = cell["shape"]["rows"], cell["shape"]["cols"]
    weights = rows * cols
    blocks = weights // 256
    traffic = {
        "payload": weights * cell["k_bits"] // 8,
        "metadata": blocks * 40,
        "codebook_staging": (1 << cell["l_bits"]) * 4 * max(1, blocks // 256),
        "partial_roundtrip": blocks * 8,
    }
    if residual:
        traffic = {field: value * 2 for field, value in traffic.items()}
    total = sum(traffic.values())
    normalization_weights = weights * (2 if residual else 1)
    traffic.update({
        "compressed_runtime_total": total,
        "compressed_runtime_bpw": total * 8 / normalization_weights,
    })
    pairs = [
        {
            "phase": "parity", "batch": None, "iteration": 0,
            "first_role": "baseline", "comparison_role": "candidate",
        },
        *[
            {
                "phase": "warmup", "batch": None, "iteration": index,
                "first_role": "baseline" if index % 2 == 0 else "candidate",
                "comparison_role": "candidate",
            }
            for index in range(3)
        ],
        *[
            {
                "phase": "trial", "batch": None, "iteration": index,
                "first_role": "baseline" if index % 2 == 0 else "candidate",
                "comparison_role": "candidate",
            }
            for index in range(10)
        ],
    ]
    singles = [{
        "phase": "parity", "role": "candidate_q12", "batch": None, "iteration": 0,
    }]
    if residual:
        singles.append({
            "phase": "parity", "role": "candidate_residual_q12",
            "batch": None, "iteration": 1,
        })
    markers = phase_markers(pairs=pairs, singles=singles)
    base_tensor = {
        "name": cell["tensor_family"], "rows": rows, "cols": cols,
        "weights": weights, "blocks": blocks, "k_bits": cell["k_bits"],
        "l_bits": cell["l_bits"], "rht_mode": "none", "rht_blocks": 0,
        "outlier_count": 0,
    }
    residual_tensor = {**base_tensor, "name": f"{cell['tensor_family']}.residual"}
    base_pass = MODULE._pass_identity(
        ordinal=0, role="base_overwrite", artifact_sha256="b" * 64,
        tensor=base_tensor, runtime_path="compact",
    )
    passes = [base_pass]
    if residual:
        passes.append(MODULE._pass_identity(
            ordinal=1, role="residual_accumulate", artifact_sha256="c" * 64,
            tensor=residual_tensor, runtime_path="compact",
        ))
    kernel_sequence = [kernel for row in passes for kernel in row["kernel_sequence"]]
    dispatches = len(kernel_sequence)
    matrix_identity = {
        "schema": MODULE.MATRIX_IDENTITY_SCHEMA,
        "cell_id": cell["id"], "matrix_cell_sha256": MODULE._canonical_sha(cell),
        "model": cell["model"], "tensor_family": cell["tensor_family"],
        "shape": cell["shape"], "k_bits": cell["k_bits"], "l_bits": cell["l_bits"],
        "runtime_path": cell["runtime_path"], "artifact_sha256": "b" * 64,
        "artifact_tensor_name": cell["tensor_family"],
    }
    matrix_identity["identity_sha256"] = MODULE._canonical_sha(matrix_identity)
    feature_identity = {
        "schema": MODULE.FEATURE_IDENTITY_SCHEMA,
        "matrix_identity_sha256": matrix_identity["identity_sha256"],
        "matrix_cell_sha256": matrix_identity["matrix_cell_sha256"],
        "projection_recipe": (
            "two_pass_residual_accumulate" if residual else "single_pass_overwrite"
        ),
        "projection_passes": len(passes),
        "pass_sequence": passes,
        "feature_counts": {
            "rht_cols_passes": 0,
            "outlier_corrected_passes": 0,
            "residual_accumulate_passes": int(residual),
            "dispatches_per_invocation": dispatches,
        },
    }
    feature_identity["feature_identity_sha256"] = MODULE._canonical_sha(feature_identity)
    geometry = {
        "rows": rows, "cols": cols, "blocks": blocks, "rht_blocks": 0,
        "outlier_count": 0, "projection_passes": len(passes),
        "residual_passes": int(residual), "dispatches_per_invocation": dispatches,
        "kernel_sequence": kernel_sequence,
        "feature_identity_sha256": feature_identity["feature_identity_sha256"],
    }
    residual_probe = (
        {
            "schema": MODULE.RESIDUAL_PROBE_SCHEMA, "enabled": True,
            "artifact": {"path": "/artifact/model.res.tq", "sha256": "c" * 64, "size_bytes": 567},
            "tensor": residual_tensor, "runtime_path": "compact",
            "recipe": MODULE.MODE_RECIPE["compact"],
            "metal": {
                "compiled": True, "kernel": MODULE.MODE_KERNEL["compact"][0],
                "reduce_kernel": "strand_bitslice_reduce_rows_accum",
                "host_entry_bytes": 40, "gpu_entry_bytes": 40,
            },
            "q12_parity": {
                "exact": True, "mismatches": 0, "values_compared": weights,
                "phase_marker_sha256": singles[1]["interval_sha256"],
            },
        }
        if residual else {"schema": MODULE.RESIDUAL_PROBE_SCHEMA, "enabled": False}
    )
    return {
        "schema": MODULE.RAW_SCHEMA,
        "source_commit": "a" * 40,
        "artifact": {"path": "/artifact/model.tq", "sha256": "b" * 64, "size_bytes": 1234},
        "matrix_identity": matrix_identity,
        "feature_identity": feature_identity,
        "residual_probe": residual_probe,
        "device": {"name": "Test GPU"},
        "runtime_path": "compact",
        "recipe": MODULE.MODE_RECIPE["compact"],
        "tensor": {
            "name": cell["tensor_family"],
            "rows": rows,
            "cols": cols,
            "weights": weights,
            "blocks": blocks,
            "k_bits": cell["k_bits"],
            "l_bits": cell["l_bits"],
        },
        "admission": {"eligible": True, "reason": None},
        "metal": {
            "compiled": True,
            "kernel": MODULE.MODE_KERNEL["compact"][0],
            "host_entry_bytes": 40,
            "gpu_entry_bytes": 40,
        },
        "parity": {
            "projection_recipe": feature_identity["projection_recipe"],
            "projection_passes": feature_identity["projection_passes"],
            "feature_identity_sha256": feature_identity["feature_identity_sha256"],
            "exact_q12": True,
            "q12_mismatches": 0,
            "q12_values_compared": weights,
            "exact_fused_vs_stored_gpu": True,
            "fused_bit_mismatches": 0,
            "fused_values_compared": rows,
            "cpu_reference_max_abs_error": 1e-6,
            "cpu_reference_max_rel_error": 1e-5,
            "q12_phase_marker_sha256": singles[0]["interval_sha256"],
            "fused_phase_marker_sha256": pairs[0]["phase_marker_sha256"],
        },
        "feature_census": {
            "schema": MODULE.FEATURE_CENSUS_SCHEMA,
            "rht_mode": "none", "rht_blocks": 0, "rht_exercised": False,
            "outlier_count": 0, "outlier_exercised": False,
            "projection_passes": len(passes),
            "residual_passes": int(residual), "residual_exercised": residual,
            "dispatches_per_invocation": dispatches,
            "dispatch_geometry_sha256": MODULE._canonical_sha(geometry),
            "kernel_sequence": kernel_sequence,
            "feature_identity_sha256": feature_identity["feature_identity_sha256"],
        },
        "logical_traffic": traffic,
        "benchmark": {
            "projection_recipe": feature_identity["projection_recipe"],
            "feature_identity_sha256": feature_identity["feature_identity_sha256"],
            "warmups": 3,
            "trials": 10,
            "baseline_wall_ns": [1200 + index for index in range(10)],
            "candidate_wall_ns": [1000 + index for index in range(10)],
            "dispatches_per_invocation": dispatches,
            "order": "paired_interleaved_alternating",
            "warmup_phase_marker_sha256": [
                row["phase_marker_sha256"] for row in pairs if row["phase"] == "warmup"
            ],
            "trial_phase_marker_sha256": [
                row["phase_marker_sha256"] for row in pairs if row["phase"] == "trial"
            ],
        },
        "phase_markers": markers,
        "physical_counters": {"measured": False},
        "default_change_requested": False,
    }


def _snapshot(swap: float = 100.0) -> dict:
    return {"pressure_level": 1, "pressure_name": "normal", "swap_used_mb": swap}


def _bundle(cell: dict, tmp_path: pathlib.Path) -> dict:
    raw = _raw(cell)
    return MODULE.build_bundle(
        raw,
        resource_before=_snapshot(),
        resource_after=_snapshot(),
        thermal_before="nominal",
        thermal_after="nominal",
        execution_authority=execution_authority(tmp_path, raw),
    )


def _counters(bundle: dict, tmp_path: pathlib.Path) -> tuple[dict, dict]:
    raw = bundle["raw_probe"]
    trials = [
        {
            "index": index,
            "phase_marker_sha256": marker,
            "energy_j": 0.002,
            "gpu_time_ns": 90,
            "physical_bytes": 1_000_000,
            "occupancy_percent": 63.0,
            "bandwidth_bytes_per_second": 200e9,
        }
        for index, marker in enumerate(raw["benchmark"]["trial_phase_marker_sha256"])
    ]
    counters = {
        "schema": MODULE.COUNTER_SCHEMA,
        "raw_bundle_sha256": bundle["raw_bundle_sha256"],
        "artifact_sha256": raw["artifact"]["sha256"],
        "tensor": raw["tensor"]["name"],
        "runtime_path": raw["runtime_path"],
        "phase_markers_sha256": raw["phase_markers"]["phase_markers_sha256"],
        "trials": trials,
        "summary": {
            "energy_j_total": 0.002 * len(trials),
            "gpu_time_ns_total": 90 * len(trials),
            "physical_bytes_total": 1_000_000 * len(trials),
            "occupancy_percent_mean": 63.0,
            "bandwidth_bytes_per_second_mean": 200e9,
        },
    }
    counter_attestation = attestation(
        tmp_path, bundle=bundle, counter_payload=counters,
        domains=("energy", "gpu_time", "physical_bytes", "occupancy", "bandwidth"),
        sample_count=raw["benchmark"]["trials"],
    )
    return counters, counter_attestation


def test_raw_bundle_and_final_receipt_are_strict_and_cell_bound(tmp_path: pathlib.Path) -> None:
    cell = _cell()
    raw = _raw(cell)
    assert MODULE.validate_raw(raw) == []
    bundle = _bundle(cell, tmp_path)
    assert MODULE.validate_bundle(bundle) == []
    counters, counter_attestation = _counters(bundle, tmp_path)
    receipt = MODULE.finalize_receipt(
        bundle, counters, counter_attestation, cell["id"],
    )
    assert MODULE.tq_receipt_contract.validate_receipt(receipt) == []
    assert receipt["bindings"]["target_sha256"] == "b" * 64
    assert receipt["experiment_payload"]["default_change_requested"] is False


def test_tampering_swap_growth_and_counter_rebinding_fail(tmp_path: pathlib.Path) -> None:
    cell = _cell()
    bundle = _bundle(cell, tmp_path)
    tampered = copy.deepcopy(bundle)
    tampered["raw_probe"]["parity"]["q12_mismatches"] = 1
    assert MODULE.validate_bundle(tampered)

    growth = MODULE.build_bundle(
        _raw(cell),
        resource_before=_snapshot(100.0),
        resource_after=_snapshot(101.0),
        thermal_before="nominal",
        thermal_after="nominal",
        execution_authority=execution_authority(tmp_path, _raw(cell)),
    )
    assert "runner observed swap growth" in MODULE.validate_bundle(growth)

    counters, counter_attestation = _counters(bundle, tmp_path)
    counters["raw_bundle_sha256"] = "0" * 64
    assert "counters are not bound to the raw bundle" in MODULE.validate_counters(
        counters, counter_attestation, bundle,
    )


def test_cell_shape_or_runtime_mismatch_cannot_finalize(tmp_path: pathlib.Path) -> None:
    cell = _cell()
    bundle = _bundle(cell, tmp_path)
    wrong = next(
        row
        for row in MODULE.tq_runtime_matrix.build_matrix()["cells"]
        if row["state"] == "deferred" and row["id"] != cell["id"]
    )
    try:
        counters, counter_attestation = _counters(bundle, tmp_path)
        MODULE.finalize_receipt(
            bundle, counters, counter_attestation, wrong["id"],
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mismatched matrix cell was accepted")


def test_phase_marker_reordering_and_dual_clock_tampering_fail_closed() -> None:
    raw = _raw(_cell())
    raw["benchmark"]["trial_phase_marker_sha256"].reverse()
    raw["phase_markers"]["intervals"][0]["continuous_ended_ns"] = 1
    errors = MODULE.validate_raw(raw)
    assert any("benchmark order" in error or "pair markers" in error for error in errors)
    assert any("timing is invalid" in error or "interval_sha256 mismatch" in error for error in errors)


def test_single_pass_cannot_self_report_residual_coverage() -> None:
    raw = _raw(_cell(), residual=False)
    assert MODULE.validate_raw(raw) == []
    raw["feature_census"]["projection_passes"] = 2
    raw["feature_census"]["residual_passes"] = 1
    raw["feature_census"]["residual_exercised"] = True
    raw["feature_census"]["kernel_sequence"].append(
        "strand_bitslice_reduce_rows_accum"
    )
    errors = MODULE.validate_raw(raw)
    assert any("exact residual pass count" in error for error in errors)
    assert any("dispatch geometry" in error for error in errors)


def test_residual_artifact_and_accumulate_kernel_are_not_cross_creditable() -> None:
    raw = _raw(_cell())
    raw["residual_probe"]["artifact"]["sha256"] = raw["artifact"]["sha256"]
    raw["residual_probe"]["metal"]["reduce_kernel"] = "strand_bitslice_reduce_rows"
    errors = MODULE.validate_raw(raw)
    assert any("not independent" in error for error in errors)
    assert any("accumulate path" in error for error in errors)
    assert any("pass sequence" in error for error in errors)


def test_matrix_and_feature_identity_hash_tampering_fails_closed() -> None:
    raw = _raw(_cell())
    raw["matrix_identity"]["matrix_cell_sha256"] = "0" * 64
    raw["feature_identity"]["pass_sequence"][1]["artifact_sha256"] = "d" * 64
    errors = MODULE.validate_raw(raw)
    assert any("matrix_identity" in error or "deferred cell" in error for error in errors)
    assert any("feature identity" in error for error in errors)


def test_residual_q12_marker_must_name_the_actual_second_decode_interval() -> None:
    raw = _raw(_cell())
    raw["residual_probe"]["q12_parity"]["phase_marker_sha256"] = raw[
        "parity"
    ]["q12_phase_marker_sha256"]
    errors = MODULE.validate_raw(raw)
    assert any("residual Q12 phase-marker attribution" in error for error in errors)


def test_two_pass_traffic_bpw_normalizes_over_both_decoded_tensors() -> None:
    raw = _raw(_cell())
    assert MODULE.validate_raw(raw) == []
    raw["logical_traffic"]["compressed_runtime_bpw"] *= 2
    errors = MODULE.validate_raw(raw)
    assert any("all decoded projection weights" in error for error in errors)
