from __future__ import annotations

import copy
import importlib.util
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[3]
CONDENSE = ROOT / "tools" / "condense"
sys.path.insert(0, str(CONDENSE))
MODULE_PATH = CONDENSE / "appendix_physical_evidence_gate.py"
SPEC = importlib.util.spec_from_file_location("appendix_physical_evidence_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _stamp(value: dict, field: str) -> dict:
    stamped = copy.deepcopy(value)
    stamped.pop(field, None)
    stamped[field] = MODULE.canonical_sha256(stamped)
    return stamped


def _release_boundary() -> dict:
    return _stamp({
        "schema": MODULE.RELEASE_BOUNDARY_SCHEMA,
        "final_interpretation_ready": True,
        "final_packet_sha256": "a" * 64,
        "observer_state_sha256": "b" * 64,
        "all_recorded_hashes_verified": True,
        "active_heavy_owner_count": 0,
        "owner_snapshot_sha256": "c" * 64,
        "ram_swap_guard_healthy": True,
        "observed_at_unix_ns": 1,
    }, "attestation_sha256")


def _source_manifest() -> dict:
    entries = [
        {"path": path, "sha256": f"{index + 1:064x}", "size_bytes": 1}
        for index, path in enumerate(sorted(MODULE.REQUIRED_SOURCE_PATHS))
    ]
    return _stamp({
        "schema": MODULE.SOURCE_MANIFEST_SCHEMA,
        "source_base_commit": "a" * 40,
        "source_base_commit_role": "repository-base-only-not-byte-authority",
        "scope": "isolated-exact-critical-source-capsule",
        "release_boundary_attestation_sha256": "9" * 64,
        "release_boundary_observation_sha256": "8" * 64,
        "required_paths_sha256": MODULE.canonical_sha256(sorted(MODULE.REQUIRED_SOURCE_PATHS)),
        "entry_count": len(entries),
        "symlink_count": 0,
        "entries": entries,
        "capsule_sha256": MODULE.canonical_sha256(entries),
    }, "manifest_sha256")


def _device_counter(raw: dict) -> dict:
    trials = [
        {
            "index": index,
            "phase_marker_sha256": f"{100 + index:064x}",
            "energy_j": 0.1,
            "gpu_time_ns": 10,
            "physical_bytes": 100,
            "occupancy_percent": 50.0,
            "bandwidth_bytes_per_second": 1000.0,
        }
        for index in range(raw["benchmark"]["trials"])
    ]
    return {
        "schema": MODULE.DEVICE_COUNTER_SCHEMA,
        "raw_bundle_sha256": "d" * 64,
        "artifact_sha256": raw["artifact"]["sha256"],
        "tensor": raw["tensor"]["name"],
        "runtime_path": raw["runtime_path"],
        "phase_markers_sha256": "e" * 64,
        "trials": trials,
        "summary": {
            "energy_j_total": 0.1 * len(trials),
            "gpu_time_ns_total": 10 * len(trials),
            "physical_bytes_total": 100 * len(trials),
            "occupancy_percent_mean": 50.0,
            "bandwidth_bytes_per_second_mean": 1000.0,
        },
    }


def _feature_census() -> dict:
    return {
        "schema": MODULE.appendix_device_runner.FEATURE_CENSUS_SCHEMA,
        "rht_mode": "cols",
        "rht_blocks": 4,
        "rht_exercised": True,
        "outlier_count": 2,
        "outlier_exercised": True,
        "projection_passes": 2,
        "residual_passes": 1,
        "residual_exercised": True,
        "dispatches_per_invocation": 6,
        "dispatch_geometry_sha256": "a" * 64,
        "kernel_sequence": [
            "strand_rht_forward_cols", "tq", "strand_bitslice_reduce_rows",
            "tq", "strand_bitslice_reduce_rows_accum", "outl",
        ],
        "feature_identity_sha256": "b" * 64,
    }


def _spec_protocol(*, transformed: bool = False) -> tuple[dict, dict]:
    batches = []
    curve_rows = []
    for batch in range(1, 9):
        repeats = [
            {
                "repeat": repeat,
                "baseline_wall_ns": 1000,
                "verifier_wall_ns": batch * 1000,
                "phase_marker_sha256": f"{batch * 100 + repeat:064x}",
                "exact_token_match": True,
                "mismatches": 0,
                "skipped": 0,
            }
            for repeat in range(5)
        ]
        batches.append({"b": batch, "repeats": repeats})
        curve_rows.append({
            "b": batch,
            "trials": 5,
            "median_ns": batch * 1000,
            "p95_ns": batch * 1000,
            "ucb_ns": batch * 1000,
            "raw_total_forward_equiv": float(batch),
            "total_forward_equiv": float(batch),
        })
    protocol = {
        "warmups_per_batch": 3,
        "independent_repeats_per_batch": 5,
        "randomized_balanced_batch_order": True,
        "paired_interleaved_baseline": True,
        "baseline_reused_across_batches": False,
        "phase_marker_schema": "hawking.physical_phase_markers.v1",
        "phase_markers_sha256": "f" * 64,
        "monotone_transform_applied": transformed,
        "batches": batches,
    }
    curve = {
        "experiment_payload": {
            "batches": curve_rows,
            "curve_method": {
                "warmups_per_batch": 3,
                "independent_repeats_per_batch": 5,
                "paired_interleaved_baseline": True,
                "monotone_transform_applied": False,
                "ucb_method": "paired_bootstrap_95",
                "confidence_level": 0.95,
                "phase_markers_sha256": "f" * 64,
            },
        }
    }
    return protocol, curve


def test_gate_is_validation_only_and_empty_packet_fails_closed() -> None:
    requirements = MODULE.requirements()
    assert requirements["default_off"] is True
    assert requirements["execution_capability"] is False
    errors = MODULE.validate_gate({})
    assert errors
    assert any("gate_sha256" in error for error in errors)


def test_aggregate_accepts_only_verified_operator_seals_and_unwraps_core() -> None:
    core = {"cell_id": "device-cell", "physical": True}
    sealed = {
        "sealed_evidence_sha256": "a" * 64,
        "request": {"kind": "device"},
        "signed_result_attestation": {"attestation": {"kind": "device"}},
        "core_evidence": core,
    }
    unwrapped, errors = MODULE._unwrap_operator_sealed_evidence(
        sealed, kind="device", verify_counter_files=False,
        validator=lambda _row: [],
    )
    assert errors == []
    assert unwrapped == core

    rejected, errors = MODULE._unwrap_operator_sealed_evidence(
        sealed, kind="device", verify_counter_files=False,
        validator=lambda _row: ["result SSHSIG verification failed"],
    )
    assert rejected is None
    assert errors == ["result SSHSIG verification failed"]

    unsigned, errors = MODULE._unwrap_operator_sealed_evidence(
        core, kind="device", verify_counter_files=False,
        validator=lambda _row: ["sealed counter evidence fields are incomplete"],
    )
    assert unsigned is None
    assert any("sealed counter evidence" in error for error in errors)

    unsigned_packet = MODULE.stamp({
        "schema": MODULE.SCHEMA,
        "release_boundary": {}, "corpus_index": {}, "corpus_verification": {},
        "source_manifest": {}, "release_build": {},
        "cpu_error_policy": MODULE.CPU_ERROR_POLICY, "spec_label": "TEST",
        "device_evidence": [core], "spec_evidence": [],
        "default_mutation_requested": False,
    })
    errors = MODULE.validate_gate(
        unsigned_packet, matrix={"cells": []}, spec_matrix={"cells": []},
        verify_counter_files=False,
    )
    assert any("sealed counter evidence fields" in error for error in errors)


def test_sealed_lists_reject_reuse_across_device_and_spec() -> None:
    # The production validator rejects these intentionally minimal objects;
    # the duplicate-envelope check must still be fail-closed before any core
    # could be credited to another evidence class.
    seal = {
        "sealed_evidence_sha256": "a" * 64,
        "request": {"kind": "device"},
        "signed_result_attestation": {"attestation": {"kind": "device"}},
        "core_evidence": {},
    }
    seen: set[str] = set()
    _items, first_errors = MODULE._unwrap_operator_sealed_list(
        [seal], kind="device", verify_counter_files=False, seen_seals=seen,
    )
    _items, second_errors = MODULE._unwrap_operator_sealed_list(
        [seal], kind="spec", verify_counter_files=False, seen_seals=seen,
    )
    assert first_errors
    assert any("reuses a sealed evidence envelope" in error for error in second_errors)


def test_release_boundary_requires_final_ready_owner_free_and_healthy_guard() -> None:
    boundary = _release_boundary()
    assert MODULE._validate_release_boundary(boundary) == []
    boundary["active_heavy_owner_count"] = 1
    boundary["final_interpretation_ready"] = False
    boundary["ram_swap_guard_healthy"] = False
    boundary = _stamp(boundary, "attestation_sha256")
    errors = MODULE._validate_release_boundary(boundary)
    assert any("final_interpretation_ready" in error for error in errors)
    assert any("owner-free" in error for error in errors)
    assert any("RAM/swap" in error for error in errors)


def test_source_manifest_requires_exact_symlink_free_critical_capsule() -> None:
    manifest = _source_manifest()
    assert MODULE._validate_source_manifest(manifest) == []
    manifest["symlink_count"] = 1
    manifest["entries"].pop()
    manifest["entry_count"] -= 1
    manifest["capsule_sha256"] = MODULE.canonical_sha256(manifest["entries"])
    manifest = _stamp(manifest, "manifest_sha256")
    errors = MODULE._validate_source_manifest(manifest)
    assert any("symlink" in error for error in errors)
    assert any("exactly" in error for error in errors)


def test_device_counter_payload_requires_attributed_unique_trials() -> None:
    raw = {
        "artifact": {"sha256": "a" * 64},
        "tensor": {"name": "tensor"},
        "runtime_path": "stored",
        "phase_markers": {"phase_markers_sha256": "e" * 64},
        "benchmark": {
            "trials": 3,
            "trial_phase_marker_sha256": [f"{100 + index:064x}" for index in range(3)],
        },
    }
    payload = _device_counter(raw)
    assert MODULE._device_counter_errors(payload, raw=raw) == []
    payload["trials"][1]["phase_marker_sha256"] = payload["trials"][0]["phase_marker_sha256"]
    payload["summary"]["physical_bytes_total"] += 1
    errors = MODULE._device_counter_errors(payload, raw=raw)
    assert any("reused" in error for error in errors)
    assert any("physical_bytes_total" in error for error in errors)


def test_feature_census_requires_consistent_rht_outlier_residual_and_dispatch() -> None:
    raw = {
        "feature_census": _feature_census(),
        "residual_probe": {"enabled": True},
        "feature_identity": {"feature_identity_sha256": "b" * 64},
    }
    assert MODULE._feature_census_errors(raw) == []
    raw["feature_census"]["rht_exercised"] = False
    raw["feature_census"]["dispatches_per_invocation"] = 0
    errors = MODULE._feature_census_errors(raw)
    assert any("RHT" in error for error in errors)
    assert any("dispatch count" in error for error in errors)


def test_aggregate_feature_census_cannot_credit_a_declared_only_residual() -> None:
    raw = {
        "feature_census": _feature_census(),
        "residual_probe": {"enabled": False},
        "feature_identity": {"feature_identity_sha256": "b" * 64},
    }
    errors = MODULE._feature_census_errors(raw)
    assert any("exact two-pass probe" in error for error in errors)
    raw["residual_probe"]["enabled"] = True
    raw["feature_census"]["kernel_sequence"].remove(
        "strand_bitslice_reduce_rows_accum"
    )
    errors = MODULE._feature_census_errors(raw)
    assert any("accumulate reduction kernel" in error for error in errors)


def test_residual_artifact_must_be_an_exact_unique_frozen_corpus_member() -> None:
    binding = {"path": "/artifact/residual.tq", "sha256": "d" * 64, "size_bytes": 17}
    index = {
        "entries": [
            {"path": "artifacts/residual.tq", "kind": "artifact", "sha256": "d" * 64, "size": 17},
        ],
    }
    assert MODULE._corpus_artifact_binding_errors(
        binding, corpus_index=index, label="residual",
    ) == []
    index["entries"].append({
        "path": "duplicate/residual.tq", "kind": "artifact", "sha256": "d" * 64, "size": 17,
    })
    errors = MODULE._corpus_artifact_binding_errors(
        binding, corpus_index=index, label="residual",
    )
    assert any("exactly once" in error for error in errors)


def test_stored_parent_requires_identical_normalized_two_pass_projection_work() -> None:
    def raw(runtime: str) -> dict:
        return {
            "runtime_path": runtime,
            "artifact": {"sha256": "a" * 64},
            "tensor": {
                "name": "model.layers.0.mlp.down_proj.weight",
                "rows": 16, "cols": 256, "weights": 4096, "blocks": 16,
                "k_bits": 4, "l_bits": 2,
            },
            "feature_census": {
                "rht_mode": "cols", "rht_blocks": 1, "outlier_count": 3,
                "kernel_sequence": [runtime],
            },
            "feature_identity": {
                "matrix_cell_sha256": runtime * 8,
                "projection_recipe": "two_pass_residual_accumulate",
                "projection_passes": 2,
            },
            "residual_probe": {
                "enabled": True,
                "artifact": {"sha256": "b" * 64},
                "tensor": {
                    "name": "model.layers.0.mlp.down_proj.residual.weight",
                    "rows": 16, "cols": 256, "weights": 4096, "blocks": 16,
                    "k_bits": 3, "l_bits": 2, "rht_mode": "none",
                    "rht_blocks": 0, "outlier_count": 1,
                },
                "runtime_path": runtime,
                "recipe": runtime,
                "metal": {"kernel": runtime},
            },
        }

    stored = raw("stored")
    child = raw("computed")
    assert MODULE._stored_parent_projection_errors(child, stored) == []

    # The runtime/kernel/matrix-cell may differ, but changing the independent
    # residual input makes this a different mathematical projection.
    child["residual_probe"]["artifact"]["sha256"] = "c" * 64
    errors = MODULE._stored_parent_projection_errors(child, stored)
    assert any("same normalized projection work" in error for error in errors)

    child = raw("computed")
    child["tensor"]["k_bits"] = 2
    errors = MODULE._stored_parent_projection_errors(child, stored)
    assert any("quantization" in error for error in errors)


def test_spec_protocol_requires_warmed_independent_untransformed_curve() -> None:
    protocol, curve = _spec_protocol()
    errors, repeats = MODULE._spec_protocol_errors(
        {"measurement_protocol": protocol}, curve,
    )
    assert errors == []
    assert repeats == 5
    protocol["monotone_transform_applied"] = True
    protocol["batches"][0]["repeats"][0]["skipped"] = 1
    errors, _ = MODULE._spec_protocol_errors(
        {"measurement_protocol": protocol}, curve,
    )
    assert any("manufactured monotonicity" in error for error in errors)
    assert any("skipped" in error for error in errors)


def test_spec_protocol_rejects_reused_phase_markers_and_nonmonotone_raw_curve() -> None:
    protocol, curve = _spec_protocol()
    protocol["batches"][1]["repeats"][0]["phase_marker_sha256"] = (
        protocol["batches"][0]["repeats"][0]["phase_marker_sha256"]
    )
    for row in protocol["batches"][3]["repeats"]:
        row["verifier_wall_ns"] = 500
    errors, _ = MODULE._spec_protocol_errors(
        {"measurement_protocol": protocol}, curve,
    )
    assert any("reused" in error for error in errors)
    assert any("not monotone" in error for error in errors)


def test_cpu_error_policy_is_finite_and_bounded() -> None:
    assert MODULE.CPU_ERROR_POLICY["max_abs_error"] <= 1e-4
    assert MODULE.CPU_ERROR_POLICY["max_rel_error"] <= 1e-4
    assert math_is_finite(MODULE.CPU_ERROR_POLICY["max_abs_error"])


def math_is_finite(value: float) -> bool:
    # Local helper keeps this test dependency-free and explicit about NaN/inf.
    return value == value and value not in {float("inf"), float("-inf")}
