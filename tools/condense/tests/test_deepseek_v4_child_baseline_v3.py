"""Focused contract tests for the additive, non-promoting DSV4F v3 freezer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab.operators import deepseek_v4_gravity as gravity
from lab.receipts import seal, verify


FULL_SEAL = "a" * 64
DIAGNOSTIC_SEAL = "b" * 64
LATENT_SEAL = "c" * 64
RESIDENCY_SEAL = "d" * 64


def _write(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return path


def _parent_v2(root: Path) -> Path:
    directory = root / "parent-v2"
    shared_evidence = {
        "full_stream_manifest": {"seal_sha256": FULL_SEAL},
        "diagnostic_manifest": {"seal_sha256": DIAGNOSTIC_SEAL},
        "receipts": {
            "fp8_metal_component_probe": {"seal_sha256": "e" * 64},
            "component_trace": {"seal_sha256": "f" * 64},
        },
    }
    common = {
        "baseline_bundle_id": "1" * 64,
        "evidence_bindings": shared_evidence,
        "claim_boundary": {
            "full_43_layer_runtime": False,
            "source_cpu_parity": False,
            "numeric_parity_v2_1": False,
            "base_true_tps": False,
        },
    }
    payloads = {
        "DSV4F_CHILD_BASELINE.json": {
            "status": "DSV4F_CHILD_BASELINE_FROZEN_FULL_STREAM_RUNTIME_PENDING",
            "frozen_metrics": {},
        },
        "DSV4F_RUNTIME_PROFILE.json": {
            "status": "DSV4F_DIAGNOSTIC_CPU_RUNTIME_PROFILE_FROZEN_FULL_RUNTIME_PENDING",
            "gpu_and_hardware_counter_boundary": {},
        },
        "DSV4F_ROUTE_PROFILE.json": {
            "status": "DSV4F_LAYER4_ROUTE_PROFILE_FROZEN_NOT_FULL_ROUTE_RESIDENCY_PROFILE",
            "unavailable_full_runtime_residency_metrics": {},
        },
        "DSV4F_LATENT_BRIDGE_CONTRACT.json": {
            "status": "DSV4F_FUTURE_BRIDGE_INTERFACES_DECLARED_NO_DONOR_INHERITANCE",
            "available_bounded_evidence": {},
        },
        "DSV4F_TRANSPLANT_POINTS.json": {
            "status": "DSV4F_TRANSPLANT_POINTS_FROZEN_SOURCE_BOUND_NO_WEIGHT_GRAFT",
            "points": [{"name": "selected_expert_ids"}],
        },
        "DSV4F_100TPS_SCOREBOARD.json": {
            "status": "BASE_TRUE_TPS_NOT_REACHED_NOT_ELIGIBLE_ON_CURRENT_RUNTIME",
            "metrics": {"BASE_TRUE_TPS": {"value": None, "status": "WITHHELD"}},
        },
        "DSV4F_KERNEL_REGISTRY.json": {
            "status": "DSV4F_KERNEL_REGISTRY_FROZEN_NO_NATIVE_43_LAYER_KERNEL_REGISTERED",
            "command_topology": {},
        },
        "DSV4F_ROOFLINE.json": {
            "status": "DSV4F_DIAGNOSTIC_LOGICAL_ROOFLINE_FROZEN_FULL_GPU_ROOFLINE_UNAVAILABLE",
            "full_geometry_roofline": {},
        },
    }
    for filename, payload in payloads.items():
        claims = dict(common["claim_boundary"])
        if filename == "DSV4F_CHILD_BASELINE.json":
            claims["component_only_fp8_metal_probe"] = True
        payload.update(common)
        payload["claim_boundary"] = claims
        payload["schema"] = gravity._CHILD_BASELINE_V2_PARENT_SCHEMAS[filename]
        _write(directory / filename, seal(payload))
    return directory


def _fp4() -> dict:
    return seal(
        {
            "schema": "hawking.gravity.deepseek_v4.fp4_e2m1fn_x2_e8m0_metal_component_probe.v1",
            "status": "PASS_REAL_METAL_COMPONENT_PARITY_NOT_FULL_RUNTIME",
            "artifact": {"manifest_seal_sha256": FULL_SEAL},
            "source": {
                "repository": gravity.REPOSITORY,
                "revision": gravity.REVISION,
                "weight": {"name": "layers.0.ffn.experts.0.w1.weight", "dtype": "I8", "shape": [2, 2]},
                "scale": {"name": "layers.0.ffn.experts.0.w1.scale", "dtype": "F8_E8M0", "shape": [1, 1]},
            },
            "scope": {
                "not_a_full_model_load": True,
                "not_a_generation_or_TPS_claim": True,
                "not_a_registered_43_layer_runtime_adapter": True,
                "not_an_MoE_route_or_expert_selection_claim": True,
            },
            "parity": {"status": "PASS", "max_abs_error": 0.0, "max_relative_error": 0.0},
            "metal": {
                "fallback": False,
                "gpu_dispatches": 1,
                "device": "test",
                "kernel": "test.fp4",
                "command_buffers": 1,
                "compute_encoders": 1,
                "timing": {"gpu_duration_us": 1},
            },
        }
    )


def _reader() -> dict:
    return seal(
        {
            "schema": "hawking.gravity.deepseek_v4.full_stream_reader_admission.v1",
            "status": "PASS_FULL_STREAM_READER_ADMISSION_NOT_FORWARD_OR_RUNTIME",
            "artifact": {
                "manifest_seal_sha256": FULL_SEAL,
                "source": {"repository": gravity.REPOSITORY, "revision": gravity.REVISION},
            },
            "execution_boundary": {
                "base_true_tps_measured": False,
                "engine_created": False,
                "hcli_endpoint_started": False,
                "public_cli_serve_admission_changed": False,
                "reader_only": True,
                "forward_tokens": 0,
                "gpu_dispatches": 0,
                "metal_allocations": 0,
            },
            "admission_validity": {
                "manifest_seal_verified": True,
                "all_named_tensor_source_index_bindings_verified": True,
                "all_referenced_chunk_paths_regular_non_symlink_and_exact_tree_verified": True,
                "all_tensor_segment_contiguity_and_source_offset_mappings_verified": True,
                "restart_receipt_and_journal_bindings_verified": True,
                "schema_status_pinned_source_verified": True,
                "native_fp4_pair_count": 1,
                "native_fp8_pair_count": 1,
                "native_scale_pair_count": 2,
                "all_chunk_sha256_bytes_verified": False,
                "pinned_codec_assets": {
                    "inference_kernel_py_sha256": "2" * 64,
                    "inference_model_py_sha256": "3" * 64,
                },
            },
            "validated_native_scale_contracts": [
                {"kind": "native.fp8_e4m3fn_e8m0"},
                {"kind": "native.fp4_e2m1fn_x2_e8m0"},
            ],
            "verified_reads": [{"touched_chunks_sha256_verified_before_return": True}],
        }
    )


def _sweep() -> dict:
    return seal(
        {
            "schema": "hawking.gravity.deepseek_v4.raw_weight_simdgroup_splitk_sweep.v1",
            "status": "PASS_REAL_M3_METAL_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP_NOT_SOURCE_FORWARD_OR_RUNTIME",
            "artifact_binding": {"manifest_seal_sha256": FULL_SEAL},
            "scope": {
                "raw_weight_component_only": True,
                "not_source_forward_parity": True,
                "not_a_full_model_load": True,
                "not_a_full_43_layer_runtime_adapter": True,
                "not_a_token_or_generation": True,
                "not_a_BASE_TRUE_TPS_measurement": True,
                "not_a_runtime_kernel_promotion": True,
                "same_sealed_full_gravity_artifact_before_and_after": True,
                "same_deterministic_input_and_raw_weight_cpu_reference_before_and_after": True,
            },
            "metal": {
                "fallback": False,
                "aggregate_real_gpu_dispatches": 2,
                "aggregate_command_buffers": 2,
                "aggregate_cpu_visible_waits": 2,
                "device": "test",
            },
            "before_after": {
                family: {
                    "p50_outcome": "CANDIDATE_GPU_P50_WIN_NOT_PROMOTED",
                    "same_raw_weight_input_and_cpu_reference": True,
                    "authority_serial_winner_gpu_p50_us": 10,
                    "candidate_parallel_winner_gpu_p50_us": 5,
                    "p50_speedup_authority_divided_by_candidate": 2.0,
                    "promotion": "NOT_PROMOTED",
                }
                for family in ("fp4_routed_expert", "fp8_control")
            },
        }
    )


def _latent() -> dict:
    return seal(
        {
            "schema": "hawking.gravity.deepseek_v4.diagnostic_latent_route_receipt.v1",
            "status": "SEALED_BOUNDED_LAYER4_CPU_DIAGNOSTIC_LATENT_ROUTE_CAPTURE",
            "artifact": {"diagnostic_scope": {"selected_layer": 4, "not_full_model": True}},
            "source_hash_binding": {
                "artifact_seal_sha256": DIAGNOSTIC_SEAL,
                "repository": gravity.REPOSITORY,
                "revision": gravity.REVISION,
                "source_parent_retained": False,
            },
            "claim_boundary": {
                "base_true_tps": False,
                "donor_data_or_distillation": False,
                "full_43_layer_runtime": False,
                "hcli_tool_augmented_throughput": False,
                "metal_dispatch": False,
                "numeric_parity_v2_1": False,
                "source_cpu_parity": False,
                "real_source_derived_layer4_forwards": True,
            },
            "capture": {
                "collection_limits": {
                    "categories": 1,
                    "max_trace_shards_per_category": 1,
                    "raw_completions_retained": False,
                    "raw_hidden_states_retained": False,
                    "raw_prompts_retained": False,
                },
                "membership_partition": {
                    "disjoint_prompt_hashes": True,
                    "disjoint_source_token_sequences": True,
                    "excluded_from": ["fit", "calibration", "public_test", "hidden_test"],
                    "name": "test",
                },
                "route_aggregate": {
                    "total_source_forwards": 1,
                    "distinct_expert_count": 1,
                    "expert_frequency": {"1": 6},
                    "raw_route_sequence_retained": False,
                    "route_set_frequency": {"4" * 64: 1},
                    "route_set_transition_frequency": {},
                },
                "trace_shards": [{}],
            },
        }
    )


def _residency(latent: dict) -> dict:
    return seal(
        {
            "schema": gravity.STATIC_EXPERT_RESIDENCY_SCHEMA,
            "status": "SEALED_STATIC_FULL_STREAM_EXPERT_RESIDENCY_CONTRACT_RUNTIME_PENDING",
            "source_binding": {
                "full_manifest_seal_sha256": FULL_SEAL,
                "repository": gravity.REPOSITORY,
                "revision": gravity.REVISION,
                "metadata": {
                    "config": {"sha256": "4" * 64},
                    "inference_config": {"sha256": "5" * 64},
                },
            },
            "analysis_mode": {
                "base_true_tps": False,
                "runtime_forward_executed": False,
                "training_or_distillation": False,
                "command_buffers": 0,
                "gpu_dispatches": 0,
                "tensor_payload_bytes_read": 0,
                "manifest_and_metadata_only": True,
            },
            "current_runtime_boundary": {
                "full_runtime_adapter": {"id": None, "metal_dispatches": 0},
                "raw_content_addressed_stream_eviction_authorized": False,
            },
            "bounded_layer4_route_observations": {
                "receipt": {
                    "seal_sha256": latent["seal_sha256"],
                    "diagnostic_artifact_seal_sha256": DIAGNOSTIC_SEAL,
                },
                "scope": {"selected_layer": 4, "source_top_k": 6, "total_source_forwards": 1},
                "frequency_ranked_experts": [{"expert_id": 1, "selection_count": 6}],
                "transition_observations": {},
            },
            "static_active_byte_summary": {
                "body_selected_weight_logical_bytes_per_decode_token": 100,
                "always_selected_shared_attention_and_control_logical_bytes": 10,
                "dense_lm_head_logical_source_bytes_per_decode_token_without_residency": 20,
                "physical_active_bytes_per_token": "NOT_MEASURED_NO_NATIVE_RUNTIME",
                "body_selected_weight_logical_bytes_interpretation": "test static bytes",
            },
            "static_packing_and_cache_candidates": {"candidate": "only"},
            "unavailable_until_native_runtime": {"physical_bytes_read": "NOT_MEASURED"},
            "geometry": {"hidden_size": 16},
        }
    )


def _device_copy(residency: dict) -> dict:
    return seal(
        {
            "schema": "hawking.gravity.deepseek_v4.metal_device_copy_roofline.v1",
            "status": "PASS_REAL_M3_METAL_DEVICE_COPY_CEILING_NOT_MODEL_KERNEL_OR_TPS",
            "scope": {
                "base_true_tps_measured": False,
                "deepseek_v4_weights_opened": False,
                "gravity_artifact_opened": False,
                "hcli_endpoint_started": False,
                "deepseek_v4_forward_tokens": 0,
                "gpu_compute_dispatches": 0,
                "gpu_model_kernels": 0,
                "device_copy_only": True,
            },
            "metal": {"fallback": False, "measured_gpu_compute_dispatches": 0, "device": "test"},
            "sizes": [
                {
                    "copy_size_mib": 128,
                    "measured_trials": 5,
                    "measured_topology": {"gpu_compute_dispatches": 0, "accounting_reconciled": True},
                }
            ],
            "static_dsv4f_source_layout_comparator": {
                "full_manifest_seal_sha256": FULL_SEAL,
                "receipt_seal_sha256": residency["seal_sha256"],
                "body_selected_weight_logical_bytes_per_decode_token": 100,
                "physical_active_bytes_per_token": "NOT_MEASURED_NO_NATIVE_RUNTIME",
                "strict_interpretation": "test only",
                "comparator": {
                    "best_median_payload_copy_gib_per_s": 1.0,
                    "best_median_read_plus_write_copy_traffic_gib_per_s": 2.0,
                    "ideal_body_only_ms_per_token_if_every_static_logical_byte_used_best_payload_copy_rate": 3.0,
                    "static_body_requirement_fraction_of_best_payload_copy_ceiling_at_100_tps": 4.0,
                },
            },
        }
    )


def _act_quant() -> dict:
    return seal(
        {
            "schema": "hawking.gravity.deepseek_v4.act_quant_fp8_wq_a_cpu_algorithm_oracle.v1",
            "status": "PASS_SOURCE_DERIVED_CPU_ALGORITHM_ORACLE_NOT_INDEPENDENT_SOURCE_RUNTIME_PARITY",
            "artifact": {
                "manifest_seal_sha256": FULL_SEAL,
                "source": {"repository": gravity.REPOSITORY, "revision": gravity.REVISION},
            },
            "execution_boundary": {
                "base_true_tps_measured": False,
                "full_model_forward": False,
                "full_model_loaded": False,
                "hcli_endpoint_started": False,
                "independently_source_runtime_parity": False,
                "source_runtime_executed": False,
                "command_buffers": 0,
                "cpu_visible_waits": 0,
                "generated_tokens": 0,
                "gpu_dispatches": 0,
                "metal_allocations": 0,
                "not_independently_source_runtime_parity": True,
                "source_derived_algorithm_oracle": True,
            },
            "input": {"captured_from_model_forward": False},
            "act_quant": {
                "source_derived": True,
                "block_size": 128,
                "activation_dtype": "F8_E4M3FN",
                "scale_dtype": "F8_E8M0FNU",
                "scale_format": "ue8m0",
            },
            "source_algorithm_bindings": {
                "official_assets_verified_by_admitted_full_stream_and_exact_anchor": {
                    "inference/kernel.py": "2" * 64,
                    "inference/model.py": "3" * 64,
                    "config.json": "4" * 64,
                    "inference/config.json": "5" * 64,
                }
            },
            "cpu_fp8_gemv": {"operator": "layer0_wq_a", "shape": [2, 2], "output_fp32_le_sha256": "6" * 64},
        }
    )


def _model_linear_extension(oracle_seal: str) -> dict:
    return seal(
        {
            "schema": "hawking.gravity.deepseek_v4.model_linear_fp8_act_quant_metal_component_parity.v1",
            "status": "PASS_REAL_METAL_MODEL_LINEAR_COMPONENT_PARITY_NOT_FULL_RUNTIME",
            "artifact": {"manifest_seal_sha256": FULL_SEAL},
            "source": {"repository": gravity.REPOSITORY, "revision": gravity.REVISION},
            "scope": {
                field: True
                for field in (
                    "model_linear_component_only",
                    "not_attention",
                    "not_base_true_tps_measurement",
                    "not_embedding",
                    "not_full_model_forward",
                    "not_full_model_load",
                    "not_hcli_endpoint",
                    "not_mhc",
                    "not_registered_43_layer_runtime_adapter",
                    "not_router_or_expert_execution",
                    "not_token_execution_or_generation",
                )
            },
            "canonical_cpu_oracle_v2": {
                "receipt_seal_sha256": oracle_seal,
                "direct_cpu_oracle_recomputed_and_matches_sealed_v2": True,
            },
            "input": {"captured_from_model_forward": False},
            "metal": {"fallback": False, "gpu_dispatches": 3},
            "gpu_act_quant": {"fallback": False},
            "gpu_fp8_weighted_projection": {
                "fallback": False,
                "selected_cpu_oracle_parity": {"status": "PASS"},
                "selected_output_bf16_hash_matches_cpu_oracle": True,
                "selected_kernel": "test.model_linear",
            },
        }
    )


def test_freeze_child_baseline_v3_binds_advanced_receipts_without_promotion(tmp_path: Path) -> None:
    parent = _parent_v2(tmp_path)
    before = {path.name: path.read_bytes() for path in parent.iterdir()}
    latent = _latent()
    residency = _residency(latent)
    extension = seal(
        {
            "schema": "hawking.gravity.deepseek_v4.pending_source_linear.v1",
            "status": "PENDING_UNPROMOTED",
            "artifact": {"manifest_seal_sha256": FULL_SEAL},
        }
    )
    paths = {
        "fp4": _write(tmp_path / gravity._CANONICAL_FP4_METAL_COMPONENT_PROBE, _fp4()),
        "reader": _write(tmp_path / gravity._CANONICAL_FULL_STREAM_READER_ADMISSION, _reader()),
        "sweep": _write(
            tmp_path / gravity._CANONICAL_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP_CANONICAL_V1,
            _sweep(),
        ),
        "latent": _write(tmp_path / gravity._CANONICAL_BOUNDED_LATENT_ROUTE_V2, latent),
        "residency": _write(tmp_path / gravity._CANONICAL_STATIC_EXPERT_RESIDENCY_V2, residency),
        "roofline": _write(tmp_path / gravity._CANONICAL_METAL_DEVICE_COPY_ROOFLINE_V1, _device_copy(residency)),
        "oracle": _write(tmp_path / gravity._CANONICAL_ACT_QUANT_WQ_A_CPU_ORACLE_V2, _act_quant()),
        "extension": _write(tmp_path / "pending-source-linear.json", extension),
    }
    out_dir = tmp_path / "child-baseline-v3"
    result = gravity.freeze_child_baseline_v3(
        prior_baseline_v2_dir=parent,
        fp4_metal_component_probe=paths["fp4"],
        full_stream_reader_admission=paths["reader"],
        raw_weight_simdgroup_splitk_sweep=paths["sweep"],
        bounded_latent_route_receipt=paths["latent"],
        static_expert_residency=paths["residency"],
        metal_device_copy_roofline=paths["roofline"],
        act_quant_wq_a_cpu_oracle=paths["oracle"],
        source_forward_extension_receipt=paths["extension"],
        out_dir=out_dir,
    )

    verify(result, label="v3 freeze result")
    assert result["status"] == "DSV4F_CHILD_BASELINE_V3_BUNDLE_SEALED"
    assert {path.name for path in out_dir.iterdir()} == set(gravity._CHILD_BASELINE_FILENAMES)
    assert before == {path.name: path.read_bytes() for path in parent.iterdir()}
    for filename in gravity._CHILD_BASELINE_FILENAMES:
        document = json.loads((out_dir / filename).read_text(encoding="utf-8"))
        verify(document, label=filename)
        assert document["schema"].endswith(".v3")
        assert document["claim_boundary"]["raw_weight_component_parity"] is True
        assert document["claim_boundary"]["source_forward_parity"] is False
        assert document["claim_boundary"]["full_43_layer_runtime"] is False
        assert document["claim_boundary"]["base_true_tps"] is False
    route = json.loads((out_dir / "DSV4F_ROUTE_PROFILE.json").read_text(encoding="utf-8"))
    assert route["v3_bounded_latent_route_capture"]["route_aggregate"]["expert_frequency"] == {"1": 6}
    assert route["v3_static_expert_residency"]["static_active_byte_summary"]["physical_active_bytes_per_token"] == "NOT_MEASURED_NO_NATIVE_RUNTIME"
    roofline = json.loads((out_dir / "DSV4F_ROOFLINE.json").read_text(encoding="utf-8"))
    assert roofline["v3_static_layout_and_device_copy_comparator"]["device_copy_roofline"]["scope"] == "DEVICE_COPY_CEILING_ONLY_NOT_MODEL_KERNEL_OR_TPS"


def test_freeze_child_baseline_v3_parser_exposes_optional_source_forward_extension() -> None:
    args = gravity._parser().parse_args(
        [
            "freeze-child-baseline-v3",
            "--prior-baseline-v2-dir", "/tmp/v2",
            "--fp4-metal-component-probe", "/tmp/fp4-metal-component-probe-receipt.json",
            "--full-stream-reader-admission", "/tmp/full-stream-reader-admission-receipt.json",
            "--raw-weight-simdgroup-splitk-sweep", "/tmp/DSV4F_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP-CANONICAL-v1.json",
            "--bounded-latent-route-receipt", "/tmp/bounded-latent-route-receipt-v2.json",
            "--static-expert-residency", "/tmp/static-expert-residency-receipt-v2.json",
            "--metal-device-copy-roofline", "/tmp/metal-device-copy-roofline-receipt-v1.json",
            "--act-quant-wq-a-cpu-oracle", "/tmp/DSV4F_ACT_QUANT_WQ_A_CPU_ORACLE-v2.json",
            "--source-forward-extension-receipt", "/tmp/prefix.json",
            "--out-dir", "/tmp/v3",
        ]
    )
    assert args.command == "freeze-child-baseline-v3"
    assert args.source_forward_extension_receipt == "/tmp/prefix.json"


def test_freeze_child_baseline_v3_rejects_historical_noncanonical_sweep_path(tmp_path: Path) -> None:
    historical = _write(
        tmp_path / gravity._CANONICAL_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP_V2,
        _sweep(),
    )
    with pytest.raises(gravity.DeepSeekV4GravityError, match="fresh canonical reissue"):
        gravity._child_baseline_v3_simdgroup_sweep(historical, full_seal=FULL_SEAL)


def test_model_linear_extension_binds_canonical_cpu_oracle_but_does_not_promote(tmp_path: Path) -> None:
    oracle_binding = {"seal_sha256": "7" * 64}
    extension = _write(tmp_path / "model-linear.json", _model_linear_extension("7" * 64))
    result = gravity._child_baseline_v3_optional_source_forward_extension(
        extension,
        full_seal=FULL_SEAL,
        act_quant_binding=oracle_binding,
    )
    assert result["status"] == "BOUND_SOURCE_LINEAR_COMPONENT_CHECKPOINT_NOT_FULL_SOURCE_FORWARD"
    assert result["source_linear_component_parity"] is True
    assert result["full_source_forward_parity"] is False
    assert result["changes_any_full_runtime_or_BASE_TRUE_TPS_gate"] is False


def test_freeze_child_baseline_v3_rejects_mismatched_reader_artifact(tmp_path: Path) -> None:
    parent = _parent_v2(tmp_path)
    latent = _latent()
    residency = _residency(latent)
    reader = _reader()
    reader["artifact"]["manifest_seal_sha256"] = "9" * 64
    # Re-seal after the intentional fixture mutation; this proves the freezer
    # checks the cross-receipt identity, not merely an outer seal.
    reader = seal({key: value for key, value in reader.items() if key != "seal_sha256"})
    paths = {
        "fp4": _write(tmp_path / gravity._CANONICAL_FP4_METAL_COMPONENT_PROBE, _fp4()),
        "reader": _write(tmp_path / gravity._CANONICAL_FULL_STREAM_READER_ADMISSION, reader),
        "sweep": _write(
            tmp_path / gravity._CANONICAL_RAW_WEIGHT_SIMDGROUP_SPLITK_SWEEP_CANONICAL_V1,
            _sweep(),
        ),
        "latent": _write(tmp_path / gravity._CANONICAL_BOUNDED_LATENT_ROUTE_V2, latent),
        "residency": _write(tmp_path / gravity._CANONICAL_STATIC_EXPERT_RESIDENCY_V2, residency),
        "roofline": _write(tmp_path / gravity._CANONICAL_METAL_DEVICE_COPY_ROOFLINE_V1, _device_copy(residency)),
        "oracle": _write(tmp_path / gravity._CANONICAL_ACT_QUANT_WQ_A_CPU_ORACLE_V2, _act_quant()),
    }
    with pytest.raises(gravity.DeepSeekV4GravityError, match="full-stream seal"):
        gravity.freeze_child_baseline_v3(
            prior_baseline_v2_dir=parent,
            fp4_metal_component_probe=paths["fp4"],
            full_stream_reader_admission=paths["reader"],
            raw_weight_simdgroup_splitk_sweep=paths["sweep"],
            bounded_latent_route_receipt=paths["latent"],
            static_expert_residency=paths["residency"],
            metal_device_copy_roofline=paths["roofline"],
            act_quant_wq_a_cpu_oracle=paths["oracle"],
            out_dir=tmp_path / "out",
        )
