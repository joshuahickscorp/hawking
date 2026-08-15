"""Regression coverage for the Qwen30 direct-packed complete-token profiler."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from lab.operators.ascension_qwen30_complete_token_profiler import (
    EXPECTED_DISPATCHES,
    FULL_RESULT_SCHEMA,
    FULL_RESULT_STATUS,
    MICROBENCH_RAW_SCHEMA,
    MICROBENCH_RAW_STATUS,
    PAIRED_SCALAR_ORDER_PRODUCTION_EXPECTED_DISPATCHES,
    PAIRED_SCALAR_ORDER_PRODUCTION_GATE_UP_SWIGLU_KERNEL,
    RUNTIME_RECEIPT_SCHEMA,
    RUNTIME_RECEIPT_STATUS,
    Qwen30CompleteTokenProfiler,
    Qwen30CompleteTokenProfilerError,
    _kernel_plan,
)
from lab.receipts import seal, verify


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _binding() -> dict[str, str]:
    return {
        "manifest_path": "/protected/QWEN30_COMPLETE_BINARY_GRAVITY_CANDIDATE.json",
        "manifest_seal_sha256": "a" * 64,
        "source_audit_seal_sha256": "b" * 64,
        "source_revision": "c" * 40,
    }


def _trace(
    binding: dict[str, str],
    *,
    elapsed_us: int = EXPECTED_DISPATCHES,
    gate_up_swiglu_kernel: str = "three_dispatch_direct_packed_gate_up_swiglu_control",
) -> dict[str, object]:
    samples = []
    plan = _kernel_plan(gate_up_swiglu_kernel)
    for ordinal, (kernel, _bucket, _label) in enumerate(plan):
        start = ordinal * 1_000
        samples.append(
            {
                "kernel_name": kernel,
                "wall_us": 1,
                "gpu_us": 1,
                "gpu_start_ns": start,
                "gpu_end_ns": start + 1_000,
            }
        )
    return {
        "schema": FULL_RESULT_SCHEMA,
        "status": FULL_RESULT_STATUS,
        "runtime_binding": {
            "manifest_seal_sha256": binding["manifest_seal_sha256"],
            "source_revision": binding["source_revision"],
            "packed_matvec_kernel": "scalar_one_thread_per_row_control",
            "gate_up_swiglu_kernel": gate_up_swiglu_kernel,
        },
        "execution": {
            "input_token_id": 7,
            "all_48_layers_executed": True,
            "final_norm_lm_head_device_argmax_executed": True,
            "step": {
                "sampled_token_id": 9,
                "elapsed_us_diagnostic_not_tps": elapsed_us,
                "metal_dispatches": len(plan) - 193,
            },
        },
        "profiler": {
            "tcb_trace_mode_requested": "gpu_prod",
            "dispatch_sample_count": len(plan),
            "gpu_timing_sample_count": len(plan),
            "expected_complete_token_dispatch_samples": len(plan),
            "complete_token_gpu_profile_coverage_earned": True,
            "ordered_dispatch_samples": samples,
            "command_buffers_committed": 291,
        },
    }


def _full(binding: dict[str, str]) -> dict[str, object]:
    return {
        "schema": FULL_RESULT_SCHEMA,
        "status": FULL_RESULT_STATUS,
        "runtime_binding": {
            "manifest_seal_sha256": binding["manifest_seal_sha256"],
            "source_revision": binding["source_revision"],
            "packed_matvec_kernel": "scalar_one_thread_per_row_control",
        },
        "execution": {
            "input_token_id": 7,
            "all_48_layers_executed": True,
            "final_norm_lm_head_device_argmax_executed": True,
            "step": {
                "sampled_token_id": 9,
                "elapsed_us_diagnostic_not_tps": 2_900_000,
            },
        },
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_receipt(
    profiler: Qwen30CompleteTokenProfiler,
    binding: dict[str, str],
    trace: dict[str, object],
    full: dict[str, object],
) -> dict[str, object]:
    trace_runtime = trace["runtime_binding"]
    assert isinstance(trace_runtime, dict)
    kernel = trace_runtime["gate_up_swiglu_kernel"]
    return seal(
        {
            "schema": RUNTIME_RECEIPT_SCHEMA,
            "status": RUNTIME_RECEIPT_STATUS,
            "binding": {
                "complete_manifest_seal_sha256": binding["manifest_seal_sha256"],
                "runtime_executable_sha256": "d" * 64,
            },
            "runtime": {
                "custom_kernel_used": True,
                "gate_up_swiglu_kernel": kernel,
            },
            "evidence": {
                "direct_full_token": {
                    "path": str(profiler.full_result),
                    "sha256": _sha256_file(profiler.full_result),
                    "schema": full["schema"],
                    "status": full["status"],
                },
                "complete_gpu_profile": {
                    "path": str(profiler.trace_result),
                    "sha256": _sha256_file(profiler.trace_result),
                    "schema": trace["schema"],
                    "status": trace["status"],
                },
            },
        }
    )


def test_fixed_runtime_plan_accounts_for_vector_decode_and_required_stages() -> None:
    plan = _kernel_plan()
    assert len(plan) == EXPECTED_DISPATCHES == 2454
    assert sum(kernel == "qwen_complete_binary_decode_vector" for kernel, _, _ in plan) == 193
    assert sum(bucket == "expert_gate_up" for _, bucket, _ in plan) == 48 * 8 * 3
    assert sum(bucket == "expert_down" for _, bucket, _ in plan) == 48 * 8
    assert plan[-1][0] == "sample_argmax_f32"


def test_paired_scalar_order_production_plan_is_exact_and_not_scalar_fitted() -> None:
    plan = _kernel_plan(PAIRED_SCALAR_ORDER_PRODUCTION_GATE_UP_SWIGLU_KERNEL)
    assert len(plan) == PAIRED_SCALAR_ORDER_PRODUCTION_EXPECTED_DISPATCHES == 1686
    assert sum(bucket == "expert_gate_up" for _, bucket, _ in plan) == 48 * 8
    assert sum(
        kernel == "qwen_direct_packed_gate_up_swiglu_paired_scalar_order_candidate"
        for kernel, _, _ in plan
    ) == 48 * 8
    assert sum(bucket == "expert_down" for _, bucket, _ in plan) == 48 * 8


def test_profile_accepts_only_production_trace_bound_to_the_exact_paired_plan(
    tmp_path: Path,
) -> None:
    profiler = Qwen30CompleteTokenProfiler(physical_root=tmp_path / "physical")
    binding = _binding()
    trace = _trace(
        binding,
        elapsed_us=PAIRED_SCALAR_ORDER_PRODUCTION_EXPECTED_DISPATCHES,
        gate_up_swiglu_kernel=PAIRED_SCALAR_ORDER_PRODUCTION_GATE_UP_SWIGLU_KERNEL,
    )
    _write(profiler.trace_result, trace)
    _write(profiler.full_result, _full(binding))

    profile = profiler._attribute(trace, _full(binding), binding)
    assert profile["status"] == "EARNED_REAL_COMPLETE_TOKEN_STAGE_PROFILE_DIAGNOSTIC_NOT_TPS"

    malformed = _trace(binding, gate_up_swiglu_kernel=PAIRED_SCALAR_ORDER_PRODUCTION_GATE_UP_SWIGLU_KERNEL)
    malformed["profiler"]["expected_complete_token_dispatch_samples"] = EXPECTED_DISPATCHES
    _write(profiler.trace_result, malformed)
    with pytest.raises(Qwen30CompleteTokenProfilerError, match="exact expected complete-token"):
        profiler._attribute(malformed, _full(binding), binding)


def test_profile_runtime_binding_requires_current_canonical_receipt_and_exact_trace_files(
    tmp_path: Path,
) -> None:
    profiler = Qwen30CompleteTokenProfiler(physical_root=tmp_path / "physical")
    binding = _binding()
    trace = _trace(
        binding,
        elapsed_us=PAIRED_SCALAR_ORDER_PRODUCTION_EXPECTED_DISPATCHES,
        gate_up_swiglu_kernel=PAIRED_SCALAR_ORDER_PRODUCTION_GATE_UP_SWIGLU_KERNEL,
    )
    full = _full(binding)
    full_runtime = full["runtime_binding"]
    assert isinstance(full_runtime, dict)
    full_runtime["gate_up_swiglu_kernel"] = PAIRED_SCALAR_ORDER_PRODUCTION_GATE_UP_SWIGLU_KERNEL
    _write(profiler.trace_result, trace)
    _write(profiler.full_result, full)
    _write(profiler.runtime_receipt, _runtime_receipt(profiler, binding, trace, full))

    profile_binding = profiler._runtime_bound_profile_binding(binding, trace, full)
    assert profile_binding["runtime_executable_sha256"] == "d" * 64
    assert profile_binding["gate_up_swiglu_kernel"] == PAIRED_SCALAR_ORDER_PRODUCTION_GATE_UP_SWIGLU_KERNEL
    assert profile_binding["canonical_runtime_receipt_seal_sha256"]

    receipt = json.loads(profiler.runtime_receipt.read_text(encoding="utf-8"))
    evidence = receipt["evidence"]
    assert isinstance(evidence, dict)
    trace_evidence = evidence["complete_gpu_profile"]
    assert isinstance(trace_evidence, dict)
    trace_evidence["sha256"] = "e" * 64
    _write(profiler.runtime_receipt, seal({key: value for key, value in receipt.items() if key != "seal_sha256"}))
    with pytest.raises(Qwen30CompleteTokenProfilerError, match="does not bind the current artifact"):
        profiler._runtime_bound_profile_binding(binding, trace, full)


def test_profile_maps_real_ordered_trace_and_keeps_hcli_unexecuted(tmp_path: Path) -> None:
    profiler = Qwen30CompleteTokenProfiler(physical_root=tmp_path / "physical")
    binding = _binding()
    trace = _trace(binding)
    full = _full(binding)
    _write(profiler.trace_result, trace)
    _write(profiler.full_result, full)

    profile = profiler._attribute(trace, full, binding)

    checked = verify(profile, label="profile")
    assert checked["status"] == "EARNED_REAL_COMPLETE_TOKEN_STAGE_PROFILE_DIAGNOSTIC_NOT_TPS"
    assert checked["timing"]["complete_token_wall_coverage_percent"] == 100.0
    buckets = {row["bucket"]: row for row in checked["buckets"]}
    assert buckets["embedding"]["dispatches"] == 1
    assert buckets["shared_expert"]["execution_status"] == "NOT_APPLICABLE_QWEN30_SOURCE_HAS_NO_SHARED_EXPERT"
    assert buckets["hcli_overhead"]["execution_status"] == "NOT_EXECUTED_HCLI_ADAPTER_ABSENT"
    assert '"BASE_TRUE_TPS"' not in json.dumps(checked)


def test_profile_refuses_any_kernel_order_substitution(tmp_path: Path) -> None:
    profiler = Qwen30CompleteTokenProfiler(physical_root=tmp_path / "physical")
    binding = _binding()
    trace = _trace(binding)
    samples = trace["profiler"]["ordered_dispatch_samples"]
    assert isinstance(samples, list)
    samples[0]["kernel_name"] = "router_component_probe"
    _write(profiler.trace_result, trace)
    _write(profiler.full_result, _full(binding))

    with pytest.raises(Qwen30CompleteTokenProfilerError, match="sequence mismatch"):
        profiler._attribute(trace, _full(binding), binding)


def test_profile_keeps_overlapping_production_intervals_explicit_and_blocked(tmp_path: Path) -> None:
    """GPU overlap is valid evidence, but it is not permission to serialize it."""

    profiler = Qwen30CompleteTokenProfiler(physical_root=tmp_path / "physical")
    binding = _binding()
    trace = _trace(binding, elapsed_us=3_000)
    samples = trace["profiler"]["ordered_dispatch_samples"]
    assert isinstance(samples, list)
    # Make the Q norm decode overlap the preceding K projection while keeping
    # the exact fixed kernel order intact.
    samples[8]["gpu_start_ns"] = 1_500
    samples[8]["gpu_end_ns"] = 2_500
    _write(profiler.trace_result, trace)
    _write(profiler.full_result, _full(binding))

    profile = verify(profiler._attribute(trace, _full(binding), binding), label="overlap profile")

    assert profile["status"] == "PROFILE_BLOCKED_INSUFFICIENT_REAL_COMPLETE_TOKEN_STAGE_COVERAGE"
    timing = profile["timing"]
    assert timing["production_gpu_multi_stage_overlap_us"] > 0
    assert timing["production_gpu_work_sum_us"] > timing["production_gpu_busy_union_us"]
    buckets = {row["bucket"]: row for row in profile["buckets"]}
    assert buckets["multi_stage_gpu_overlap"]["execution_status"].startswith("OBSERVED_CONCURRENT")


def test_profile_accepts_only_a_non_overlapping_source_bound_host_ledger(tmp_path: Path) -> None:
    profiler = Qwen30CompleteTokenProfiler(physical_root=tmp_path / "physical")
    binding = _binding()
    trace = _trace(binding)
    trace["profiler"].update(
        {
            "host_stage_timer_origin": "complete_token_runtime_start",
            "host_stage_interval_coverage_earned": True,
            "host_stage_intervals": [
                {"bucket": "embedding", "label": "exact embedding host stage", "start_us": 0, "end_us": 1},
                {
                    "bucket": "command_graph_transition_gap",
                    "label": "exact host command graph and wait stage",
                    "start_us": 1,
                    "end_us": EXPECTED_DISPATCHES,
                },
            ],
        }
    )
    _write(profiler.trace_result, trace)
    _write(profiler.full_result, _full(binding))

    profile = verify(profiler._attribute(trace, _full(binding), binding), label="host ledger profile")

    assert profile["status"] == "EARNED_REAL_COMPLETE_TOKEN_STAGE_PROFILE_DIAGNOSTIC_NOT_TPS"
    timing = profile["timing"]
    assert timing["source_bound_host_stage_ledger_available"] is True
    assert timing["source_bound_host_stage_ledger_valid"] is True
    assert timing["source_bound_host_stage_coverage_percent"] == 100.0
    buckets = {row["bucket"]: row for row in profile["buckets"]}
    assert buckets["command_graph_transition_gap"]["source_bound_host_stage_wall_us"] == EXPECTED_DISPATCHES - 1


def test_component_receipt_requires_direct_pack_parity_and_no_tps(tmp_path: Path) -> None:
    profiler = Qwen30CompleteTokenProfiler(physical_root=tmp_path / "physical")
    binding = _binding()
    trace = _trace(binding)
    full = _full(binding)
    _write(profiler.trace_result, trace)
    _write(profiler.full_result, full)
    profile = profiler._attribute(trace, full, binding)
    _write(profiler.profile_path, profile)
    raw = {
        "schema": MICROBENCH_RAW_SCHEMA,
        "status": MICROBENCH_RAW_STATUS,
        "binding": {
            "manifest_seal_sha256": binding["manifest_seal_sha256"],
            "source_audit_seal_sha256": binding["source_audit_seal_sha256"],
            "source_revision": binding["source_revision"],
        },
        "candidate": {
            "id": "qwen30-direct-packed-gate-up-pair-command-topology",
            "baseline_command_topology": {"compute_dispatches": 2},
            "candidate_command_topology": {"compute_dispatches": 1},
            "direct_packed_layout": "HQ30G1B1, group_size=128, FP16 scales plus sign bits",
        },
        "parity": {"baseline_within_tolerance": True, "paired_within_tolerance": True},
        "timing": {
            "baseline_two_dispatch": {"host_wall_us_p50": 20.0},
            "candidate_one_dispatch": {"host_wall_us_p50": 10.0},
            "p50_component_host_wall_delta_us": -10.0,
            "p50_component_host_wall_speedup_ratio": 2.0,
        },
        "claim_boundary": {"not_a_model_or_token_rate": True},
    }
    _write(profiler.microbench_raw, raw)

    receipt = profiler._reconcile_microbench(binding, profile)

    assert receipt is not None
    checked = verify(receipt, label="component receipt")
    assert checked["candidate"]["candidate_command_topology"]["compute_dispatches"] == 1
    assert checked["integration_gate"]["requires_new_all_48_layer_complete_token_profile"] is True
    assert "tokens_per_second" not in json.dumps(checked)
