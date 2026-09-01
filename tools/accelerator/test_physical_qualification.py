from __future__ import annotations

import json
from pathlib import Path

import pytest

import physical_qualification as queue


def _protected_receipt(mutation=None):
    return {
        "schema": "hcli.agentos.protected_accelerator_benchmark.v1",
        "status": "PASSED",
        "benchmark_class": "QUALIFIED_PROTECTED",
        "measurement_verdict": "ACCEPT",
        "qualification": True,
        "protected_window": True,
        "contamination": [],
        "bench": {"state": "QUIESCED"},
        "checks": {
            "all_measurements_ok": True,
            "all_requested_measurements_completed": True,
            "capability_sanity": True,
            "zero_fallbacks": True,
            "one_persistent_pid_when_declared": True,
            "no_connector_restart": True,
            "no_unqualified_promotion": True,
        },
        "fusion_env_overrides": dict(mutation or {}),
        "aggregate": {
            "complete_wall_ns_per_token": {"median": 100.0},
            "gpu_ns_per_token": {"median": 40.0},
            "dispatches_per_token": {"median": 12.0},
        },
        "measurements": [{"fallbacks": 0}],
    }


def test_frontier_contains_concrete_qwen_and_flash_candidates():
    body = queue.build_queue()
    assert queue.validate_queue(body)["passed"] is True
    assert body["schema"] == queue.SCHEMA
    assert body["counts"]["candidates"] >= 12
    assert body["counts"]["by_status"]["STATIC_ONLY"] >= 2
    assert body["counts"]["by_status"]["BLOCKED"] >= 4
    ids = {row["candidate_id"] for row in body["candidates"]}
    assert "qwen27-affine2-splitk4-vec" in ids
    assert "qwen27-q2f-splitk4" in ids
    assert "qwen27-q2f-splitk4-vec" in ids
    assert "qwen27-pipeline-state-elision" in ids
    assert "qwen27-pipeline-cache-reuse" in ids
    assert "qwen27-pipeline-id-resolution" in ids
    assert "qwen27-encoder-label-elision" in ids
    assert "flash-hc-router-topk-fusion" in ids
    assert "flash-p7-mhc-pre-simdgroup" in ids
    assert "flash-device-mhc-state" in ids
    assert "flash-p6-hash-single-command-buffer" in ids
    assert "flash-p6-act-quant-simdgroup" in ids
    assert "flash-p6-routed-fp4-simdgroup" in ids
    assert "flash-p6-shared-fp8-simdgroup" in ids
    assert "flash-fullseq-ordered-encoder" in ids
    assert "flash-pipeline-cache-reuse" in ids
    assert "flash-pipeline-id-resolution" in ids
    assert "flash-encoder-label-elision" in ids
    assert "flash-fullseq-catalog-cache" in ids
    assert "flash-qkv-gqa-rope-fusion" in ids
    assert "flash-routed-fp4-gate-up-swiglu-fused" in ids
    assert "flash-p6-routed-fp4-gate-up-swiglu-fused" in ids
    assert "flash-p6-routed-fp4-gate-up-swiglu-simd" in ids
    assert "flash-p6-routed-fp4-down-bf16-fused" in ids
    assert "flash-p6-learned-reader-reuse" in ids
    assert "flash-p6-learned-expert-cache-reuse" in ids
    assert "flash-p6-batched-down-qat" in ids
    assert "flash-shared-fp8-gate-up-swiglu-fused" in ids
    assert "flash-compact-moe-bf16-vec4" in ids
    assert "flash-meta-sub1-coherent" in ids
    assert body["bench"]["state"] == "UNKNOWN"
    assert body["measurement_contract"]["protected_pass_requires_all_fields"] is True
    for row in body["candidates"]:
        assert set(queue.MEASUREMENT_FIELDS).issubset(row["measurements"])
        assert row["measurements"]["status"] == "NOT_MEASURED"
        assert row["scope_tags"]
        assert set(row["scope_tags"]).issubset(queue.CANDIDATE_SCOPE_TAGS)
        if "GENERIC_CANDIDATE" in row["scope_tags"]:
            assert row["transfer_evidence"]


def test_q2f_geometry_can_be_qualified_independently_from_affine2():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    affine = rows["qwen27-affine2-splitk4"]
    q2f = rows["qwen27-q2f-splitk4"]
    assert affine.exact_mutation["child_fusion_env"]["HAWKING_Q2F_GEO"] == "tpr64"
    assert q2f.exact_mutation["child_fusion_env"]["HAWKING_AFFINE2_GEO"] == "tpr64"
    assert q2f.exact_mutation["child_fusion_env"]["HAWKING_Q2F_GEO"] == "splitk4"


def test_qwen_q4_geometry_candidate_has_a_matched_incumbent_control():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    q4 = rows["qwen27-q4-vecgroup-x64"]
    assert q4.status == "READY_PROTECTED"
    assert q4.exact_mutation["child_fusion_env"] == {
        "HAWKING_QWEN38_FAST": "1",
        "HAWKING_QWEN38_Q4_GEO": "vecgroup_x64",
    }
    assert q4.control_configuration["child_fusion_env"] == {
        "HAWKING_QWEN38_FAST": "1",
        "HAWKING_QWEN38_Q4_GEO": "tpr64",
    }
    assert "affine-Q2" in q4.expected_gpu_ns_mechanism
    assert any(path.endswith("qwen_uniform_q4.metal") for path in q4.source_evidence)
    assert any(path.endswith("matvec-occupancy-230x.json") for path in q4.source_evidence)


def test_qwen_pipeline_state_elision_has_an_explicit_control():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["qwen27-pipeline-state-elision"]
    assert candidate.status == "READY_PROTECTED"
    assert candidate.exact_mutation["child_fusion_env"] == {
        "HAWKING_METAL_PIPELINE_STATE_ELISION": "1",
        "HAWKING_QWEN38_FAST": "1",
    }
    assert candidate.control_configuration["child_fusion_env"] == {
        "HAWKING_METAL_PIPELINE_STATE_ELISION": "0",
        "HAWKING_QWEN38_FAST": "1",
    }
    assert "topology are unchanged" in candidate.expected_dispatch_reduction
    assert any(path.endswith("metal/mod.rs") for path in candidate.source_evidence)


def test_qwen_pipeline_cache_reuse_has_an_explicit_control():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["qwen27-pipeline-cache-reuse"]
    assert candidate.status == "READY_PROTECTED"
    assert candidate.exact_mutation["child_fusion_env"] == {
        "HAWKING_METAL_PIPELINE_CACHE_REUSE": "1",
        "HAWKING_QWEN38_FAST": "1",
    }
    assert candidate.control_configuration["child_fusion_env"] == {
        "HAWKING_METAL_PIPELINE_CACHE_REUSE": "0",
        "HAWKING_QWEN38_FAST": "1",
    }
    assert "topology are unchanged" in candidate.expected_dispatch_reduction
    assert any(path.endswith("qwen38_hybrid_decode.rs") for path in candidate.source_evidence)


def test_qwen_pipeline_id_resolution_has_an_explicit_control():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["qwen27-pipeline-id-resolution"]
    assert candidate.status == "READY_PROTECTED"
    assert candidate.exact_mutation["child_fusion_env"] == {
        "HAWKING_METAL_PIPELINE_CACHE_REUSE": "1",
        "HAWKING_METAL_PIPELINE_ID_RESOLUTION": "1",
        "HAWKING_QWEN38_FAST": "1",
    }
    assert candidate.control_configuration["child_fusion_env"] == {
        "HAWKING_METAL_PIPELINE_CACHE_REUSE": "1",
        "HAWKING_METAL_PIPELINE_ID_RESOLUTION": "0",
        "HAWKING_QWEN38_FAST": "1",
    }
    assert "second" in candidate.expected_eliminated_work
    assert any(path.endswith("metal/mod.rs") for path in candidate.source_evidence)


def test_qwen_encoder_label_elision_has_an_explicit_control():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["qwen27-encoder-label-elision"]
    assert candidate.status == "READY_PROTECTED"
    assert candidate.exact_mutation["child_fusion_env"] == {
        "HAWKING_METAL_ENCODER_LABEL_ELISION": "1",
        "HAWKING_QWEN38_FAST": "1",
    }
    assert candidate.control_configuration["child_fusion_env"] == {
        "HAWKING_METAL_ENCODER_LABEL_ELISION": "0",
        "HAWKING_QWEN38_FAST": "1",
    }
    assert "topology are unchanged" in candidate.expected_dispatch_reduction
    assert any(path.endswith("metal/mod.rs") for path in candidate.source_evidence)


def test_qwen_commit_timing_elision_has_an_explicit_control():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["qwen27-commit-timing-elision"]
    assert candidate.status == "READY_PROTECTED"
    assert candidate.exact_mutation["child_fusion_env"] == {
        "HAWKING_METAL_COMMIT_TIMING_ELISION": "1",
        "HAWKING_QWEN38_FAST": "1",
    }
    assert candidate.control_configuration["child_fusion_env"] == {
        "HAWKING_METAL_COMMIT_TIMING_ELISION": "0",
        "HAWKING_QWEN38_FAST": "1",
    }
    assert "topology are unchanged" in candidate.expected_dispatch_reduction
    assert any(path.endswith("metal/mod.rs") for path in candidate.source_evidence)


def test_qwen_resident_untimed_decode_is_separate_from_measured_qualification():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["qwen27-resident-untimed-decode"]
    assert candidate.status == "STATIC_ONLY"
    assert candidate.exact_mutation["child_fusion_env"] == {
        "HAWKING_METAL_COMMIT_TIMING_ELISION": "1",
        "HAWKING_QWEN38_FAST": "1",
        "HAWKING_QWEN38_SERVE_UNTIMED": "1",
    }
    assert candidate.control_configuration["child_fusion_env"] == {
        "HAWKING_METAL_COMMIT_TIMING_ELISION": "1",
        "HAWKING_QWEN38_FAST": "1",
        "HAWKING_QWEN38_SERVE_UNTIMED": "0",
    }
    assert "omits per-token counters" in candidate.expected_active_byte_change
    assert any(path.endswith("genesis_body/src/main.rs") for path in candidate.source_evidence)


def test_flash_source_bf16_mutation_uses_runtime_on_value_and_records_active_paths():
    candidate = next(
        row for row in queue.frontier_candidates() if row.candidate_id == "flash-source-bf16-simd"
    )
    controls = candidate.exact_mutation["source_oracle_controls"]
    assert controls == {
        "HAWKING_FLASH_BF16_GEO": "1",
        "HAWKING_FLASH_BF16_VEC4": "1",
    }
    assert any(
        path.endswith("gravity_deepseek_v4_native_token_graph.rs")
        for path in candidate.source_evidence
    )
    assert any(
        path.endswith("gravity_deepseek_v4_streamed_native.rs")
        for path in candidate.source_evidence
    )


def test_flash_compact_moe_vec4_is_separate_from_simd_reduction():
    candidate = next(
        row
        for row in queue.frontier_candidates()
        if row.candidate_id == "flash-compact-moe-bf16-vec4"
    )
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_FLASH_MOE_VEC4": "1"
    }
    assert "topology is unchanged" in candidate.expected_dispatch_reduction
    assert "exact-order" in candidate.expected_gpu_ns_mechanism
    assert any(path.endswith("qwen_next.metal") for path in candidate.source_evidence)


def test_flash_qkv_fusion_is_explicitly_queued_and_keeps_diagnostics():
    candidate = next(
        row
        for row in queue.frontier_candidates()
        if row.candidate_id == "flash-qkv-gqa-rope-fusion"
    )
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_FLASH_QKV_GQA_FUSED": "1"
    }
    assert "two launches to one" in candidate.expected_dispatch_reduction
    assert "raw Q/K/V diagnostic buffers" in candidate.expected_intermediate_byte_reduction
    assert any("qwen_next_bf16_qkv_gqa_rope_cache" in path for path in candidate.source_evidence)


def test_flash_fp4_gate_up_fusion_is_explicitly_queued_and_blocked():
    candidate = next(
        row
        for row in queue.frontier_candidates()
        if row.candidate_id == "flash-routed-fp4-gate-up-swiglu-fused"
    )
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED": "1"
    }
    assert "five dispatches to one" in candidate.expected_dispatch_reduction
    assert any(path.endswith("gk_family.metal") for path in candidate.source_evidence)
    assert any(
        path.endswith("gravity_deepseek_v4_native_token_graph.rs")
        for path in candidate.source_evidence
    )


def test_flash_p6_fp4_gate_up_fusion_closes_the_fixed_six_dispatch_budget():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["flash-p6-routed-fp4-gate-up-swiglu-fused"]
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED": "1",
        "HAWKING_DSV4F_P6_FP4_GATE_UP_SWIGLU_SIMD": "0",
    }
    assert candidate.control_configuration["source_oracle_controls"] == {
        "HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED": "0",
        "HAWKING_DSV4F_P6_FP4_GATE_UP_SWIGLU_SIMD": "0",
    }
    assert "38 to 9" in candidate.expected_dispatch_reduction
    assert "60 to 31" in candidate.expected_dispatch_reduction
    assert "explicit read/write residency" in candidate.expected_gpu_ns_mechanism
    assert any(path.endswith("moe.metal") for path in candidate.source_evidence)
    assert any(
        path.endswith("gravity_deepseek_v4_p6_device.rs")
        for path in candidate.source_evidence
    )


def test_flash_p6_fp4_gate_up_fusion_simd_isolated_from_scalar_fusion():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["flash-p6-routed-fp4-gate-up-swiglu-simd"]
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED": "1",
        "HAWKING_DSV4F_P6_FP4_GATE_UP_SWIGLU_SIMD": "1",
    }
    assert candidate.control_configuration["source_oracle_controls"] == {
        "HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED": "1",
        "HAWKING_DSV4F_P6_FP4_GATE_UP_SWIGLU_SIMD": "0",
    }
    assert candidate.dependencies == ("flash-p6-routed-fp4-gate-up-swiglu-fused",)
    assert "eight SIMDgroups" in candidate.expected_gpu_ns_mechanism


def test_flash_p6_routed_down_fusion_closes_the_w2_cast_budget():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["flash-p6-routed-fp4-down-bf16-fused"]
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED": "1"
    }
    assert candidate.control_configuration["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED": "0"
    }
    assert "60 to 49" in candidate.expected_dispatch_reduction
    assert "22 to 11" in candidate.expected_dispatch_reduction
    assert "already-authoritative" in candidate.expected_gpu_ns_mechanism
    assert any(path.endswith("moe.metal") for path in candidate.source_evidence)


def test_flash_p6_batched_down_qat_packs_only_independent_blocks():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["flash-p6-batched-down-qat"]
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_BATCHED_DOWN_QAT": "1"
    }
    assert candidate.control_configuration["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_BATCHED_DOWN_QAT": "0"
    }
    assert "60 to 54" in candidate.expected_dispatch_reduction
    assert "six routed and one shared" in candidate.expected_eliminated_work
    assert "unchanged" in candidate.expected_active_byte_change
    assert any(path.endswith("matmul.metal") for path in candidate.source_evidence)


def test_flash_p6_full_downstream_fusion_removes_the_materialization_tail():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["flash-p6-fused-down-shared-combine"]
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED": "1",
        "HAWKING_DSV4F_P6_FP4_DOWN_SHARED_COMBINE_FUSED": "1",
    }
    assert candidate.control_configuration["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED": "0",
        "HAWKING_DSV4F_P6_FP4_DOWN_SHARED_COMBINE_FUSED": "0",
    }
    assert "fifteen downstream" in candidate.expected_dispatch_reduction
    assert "60 to 46" in candidate.expected_dispatch_reduction
    assert "routed/shared down intermediates" in candidate.expected_eliminated_work
    assert any(path.endswith("moe.metal") for path in candidate.source_evidence)


def test_flash_p6_learned_reader_reuse_is_a_host_ceremony_ab():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["flash-p6-learned-reader-reuse"]
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_LEARNED_READER_REUSE": "1"
    }
    assert candidate.control_configuration["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_LEARNED_READER_REUSE": "0"
    }
    assert "manifest/index admission" in candidate.expected_eliminated_work
    assert "metadata-only reader" in candidate.expected_intermediate_byte_reduction
    assert any(path.endswith("gravity_deepseek_v4.rs") for path in candidate.source_evidence)


def test_flash_p6_learned_expert_cache_reuse_is_separate_from_reader_reuse():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["flash-p6-learned-expert-cache-reuse"]
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_LEARNED_EXPERT_CACHE_REUSE": "1",
        "HAWKING_DSV4F_P6_LEARNED_READER_REUSE": "1",
    }
    assert candidate.control_configuration["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_LEARNED_EXPERT_CACHE_REUSE": "0",
        "HAWKING_DSV4F_P6_LEARNED_READER_REUSE": "1",
    }
    assert candidate.dependencies == ("flash-p6-learned-reader-reuse",)
    assert "source chunk materialization" in candidate.expected_eliminated_work
    assert "exact six-bundle" in candidate.expected_intermediate_byte_reduction


def test_shared_runtime_candidates_are_not_generic_verified():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    for candidate_id in queue.GENERIC_RUNTIME_CANDIDATES:
        candidate = rows[candidate_id]
        assert "BACKEND_FAMILY" in candidate.scope_tags
        assert "GENERIC_CANDIDATE" in candidate.scope_tags
        assert "GENERIC_VERIFIED" not in candidate.scope_tags
        assert candidate.transfer_evidence


def test_flash_shared_fp8_gate_up_fusion_is_explicitly_queued_and_blocked():
    candidate = next(
        row
        for row in queue.frontier_candidates()
        if row.candidate_id == "flash-shared-fp8-gate-up-swiglu-fused"
    )
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_DSV4F_FP8_SHARED_GATE_UP_SWIGLU_FUSED": "1"
    }
    assert "five dispatches to one" in candidate.expected_dispatch_reduction
    assert any(path.endswith("matmul.metal") for path in candidate.source_evidence)
    assert any(
        path.endswith("gravity_deepseek_v4_native_token_graph.rs")
        for path in candidate.source_evidence
    )


def test_flash_meta_representation_is_separate_from_physical_ebpw_and_blocked():
    candidate = next(
        row for row in queue.frontier_candidates() if row.candidate_id == "flash-meta-sub1-coherent"
    )
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_FLASH_META_REPRESENTATION": "teacher_distilled_sub1_v1",
        "HAWKING_FLASH_META_BPW_TARGET": "0.8871807728336929",
        "HAWKING_FLASH_META_ROUTER_GUARD": "exact",
        "HAWKING_FLASH_META_DENSE_REMATERIALIZE": "0",
    }
    assert "prospective meta_bpw" in candidate.expected_active_byte_change
    assert "physical active bytes" in candidate.expected_active_byte_change
    assert "serialized functional artifact" in candidate.blocked_reason
    assert any(path.endswith("FLASH_META_REPRESENTATION_SUB1.json") for path in candidate.source_evidence)
    assert any(path.endswith("flash_meta_coherence_screen.py") for path in candidate.source_evidence)
    assert any(path.endswith("FLASH_META_COHERENCE_SCREEN_L4.json") for path in candidate.source_evidence)
    assert any(path.endswith("FLASH_ORGAN_CENSUS.json") for path in candidate.source_evidence)


def test_ready_candidates_have_argv_only_hcli_workunits():
    body = queue.build_queue(model="Qwen27")
    ready = {
        row["candidate_id"]
        for row in body["candidates"]
        if row["status"] in queue.READY_STATUSES
    }
    assert ready
    assert {row["candidate_id"] for row in body["work_units"]} == ready
    assert all(row["resource_class"] == "GPU_EXCLUSIVE" for row in body["work_units"])
    assert all(row["effect_class"] == "REVERSIBLE" for row in body["work_units"])
    for row in body["work_units"]:
        for command_key in ("diagnostic_command", "protected_command"):
            command = row[command_key]
            assert "-c" not in command
            assert "--shell" not in command
            assert command[0:4] == ["python3", "-m", "hcli", "agentos"]


def test_flash_rows_are_blocked_without_misrepresenting_source_as_nx():
    body = queue.build_queue(model="Flash")
    assert body["counts"]["work_units"] == 0
    assert body["candidates"]
    for row in body["candidates"]:
        assert row["status"] in {"BLOCKED", "STATIC_ONLY"}
        if row["status"] == "BLOCKED":
            assert row["blocked_reason"]
        assert "NX" in row["capability_contract"]
        assert row["baseline_path"].endswith("FLASH_NEXT_NOETIC_EXECUTABLE.json")


def test_flash_pipeline_id_resolution_has_a_matched_control():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["flash-pipeline-id-resolution"]
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_FLASH_PIPELINE_CACHE_REUSE": "1",
        "HAWKING_METAL_PIPELINE_ID_RESOLUTION": "1",
    }
    assert candidate.control_configuration["source_oracle_controls"] == {
        "HAWKING_FLASH_PIPELINE_CACHE_REUSE": "1",
        "HAWKING_METAL_PIPELINE_ID_RESOLUTION": "0",
    }
    assert "source-independent" in candidate.blocked_reason


def test_flash_p7_mhc_pre_simdgroup_has_an_authority_control():
    candidate = next(
        row
        for row in queue.frontier_candidates()
        if row.candidate_id == "flash-p7-mhc-pre-simdgroup"
    )
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_DSV4F_MHC_PRE_SIMD": "1"
    }
    assert candidate.control_configuration["source_oracle_controls"] == {
        "HAWKING_DSV4F_MHC_PRE_SIMD": "0"
    }
    assert "24-SIMDgroup" in candidate.expected_gpu_ns_mechanism
    assert any(path.endswith("deepseek_v4_p7.metal") for path in candidate.source_evidence)
    assert any(path.endswith("gravity_deepseek_v4_p4b_device.rs") for path in candidate.source_evidence)


def test_flash_device_mhc_state_has_a_matched_control_and_material_boundary():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["flash-device-mhc-state"]
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_DSV4F_DEVICE_MHC": "1",
        "HAWKING_DSV4F_MHC_NORM_SIMD": "1",
    }
    assert candidate.control_configuration["source_oracle_controls"] == {
        "HAWKING_DSV4F_DEVICE_MHC": "0",
        "HAWKING_DSV4F_MHC_NORM_SIMD": "0",
    }
    assert "two HIDDEN_SIZE BF16 activation readbacks" in candidate.expected_eliminated_work
    assert "one final report/head readback" in candidate.expected_intermediate_byte_reduction
    assert candidate.dependencies == ("flash-p7-mhc-pre-simdgroup",)
    assert any(path.endswith("gravity_deepseek_v4_native_token_graph.rs") for path in candidate.source_evidence)


def test_flash_p6_single_command_buffer_has_a_historical_control():
    candidate = next(
        row
        for row in queue.frontier_candidates()
        if row.candidate_id == "flash-p6-hash-single-command-buffer"
    )
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_SINGLE_CB": "1"
    }
    assert candidate.control_configuration["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_SINGLE_CB": "0"
    }
    assert "60 dispatches remain unchanged" in candidate.expected_dispatch_reduction
    assert "first P6 command buffer" in candidate.expected_gpu_ns_mechanism
    assert any(path.endswith("gravity_deepseek_v4_p6_device.rs") for path in candidate.source_evidence)


def test_flash_p6_act_quant_simdgroup_has_a_byte_exact_control():
    candidate = next(
        row
        for row in queue.frontier_candidates()
        if row.candidate_id == "flash-p6-act-quant-simdgroup"
    )
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_ACT_QUANT_SIMD": "1"
    }
    assert candidate.control_configuration["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_ACT_QUANT_SIMD": "0"
    }
    assert "SIMD-group" in candidate.expected_gpu_ns_mechanism
    assert any(path.endswith("matmul.metal") for path in candidate.source_evidence)


def test_flash_p6_routed_fp4_simdgroup_has_a_matched_control():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["flash-p6-routed-fp4-simdgroup"]
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_FP4_SIMD": "1",
    }
    assert candidate.control_configuration["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_FP4_SIMD": "0",
    }
    assert "SIMDgroup" in candidate.expected_gpu_ns_mechanism
    assert "18" in candidate.expected_eliminated_work
    assert any(path.endswith("matmul.metal") for path in candidate.source_evidence)


def test_flash_p6_shared_fp8_simdgroup_has_a_matched_control():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["flash-p6-shared-fp8-simdgroup"]
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_FP8_SIMD": "1",
    }
    assert candidate.control_configuration["source_oracle_controls"] == {
        "HAWKING_DSV4F_P6_FP8_SIMD": "0",
    }
    assert "SIMDgroup" in candidate.expected_gpu_ns_mechanism
    assert "three" in candidate.expected_eliminated_work
    assert any(path.endswith("matmul.metal") for path in candidate.source_evidence)


def test_flash_pipeline_cache_reuse_covers_p6_batches():
    rows = {row.candidate_id: row for row in queue.frontier_candidates()}
    candidate = rows["flash-pipeline-cache-reuse"]
    assert candidate.status == "BLOCKED"
    assert candidate.exact_mutation == {
        "source_oracle_controls": {"HAWKING_FLASH_PIPELINE_CACHE_REUSE": "1"}
    }
    assert "four P6 MoE batches" in candidate.expected_gpu_ns_mechanism
    assert any(path.endswith("gravity_deepseek_v4_p6_device.rs") for path in candidate.source_evidence)


def test_validation_rejects_ready_workunit_for_static_candidate():
    body = queue.build_queue()
    static_id = next(
        row["candidate_id"]
        for row in body["candidates"]
        if row["status"] == "STATIC_ONLY"
    )
    tampered = json.loads(json.dumps(body))
    source = next(row for row in tampered["candidates"] if row["candidate_id"] == static_id)
    source["status"] = "READY_DIAGNOSTIC"
    with pytest.raises(queue.PhysicalQueueError, match="every READY candidate"):
        queue.validate_queue(tampered)


def test_candidate_pass_status_requires_evidence():
    candidate = queue.frontier_candidates()[0].to_dict()
    candidate["status"] = "PROTECTED_PASS"
    with pytest.raises(queue.PhysicalQueueError, match="requires evidence"):
        queue.validate_candidate(candidate)


def test_advance_rebuilds_workunits_and_fingerprint():
    body = queue.build_queue()
    candidate_id = "qwen27-affine2-splitk4-vec"
    updated = queue.advance_queue(
        body,
        candidate_id=candidate_id,
        status="READY_DIAGNOSTIC",
    )
    assert queue.validate_queue(updated)["passed"] is True
    row = next(item for item in updated["candidates"] if item["candidate_id"] == candidate_id)
    assert row["status"] == "READY_DIAGNOSTIC"
    assert candidate_id in {item["candidate_id"] for item in updated["work_units"]}
    assert updated["fingerprint"] != body["fingerprint"]


def test_advance_requires_evidence_and_rejects_skipped_rung():
    body = queue.build_queue()
    with pytest.raises(queue.PhysicalQueueError, match="terminal status requires evidence"):
        queue.advance_queue(
            body,
            candidate_id="qwen27-affine2-splitk4",
            status="PROTECTED_REJECT",
        )
    with pytest.raises(queue.PhysicalQueueError, match="cannot advance"):
        queue.advance_queue(
            body,
            candidate_id="qwen27-affine2-splitk4-vec",
            status="PROTECTED_PASS",
            evidence=("receipt.json",),
        )


def test_advance_preserves_rejection_evidence_and_requires_static_review():
    body = queue.build_queue()
    rejected = queue.advance_queue(
        body,
        candidate_id="qwen27-affine2-splitk4-vec",
        status="BLOCKED",
        blocked_reason="Metal compiler unavailable",
    )
    with pytest.raises(queue.PhysicalQueueError, match="cannot advance"):
        queue.advance_queue(
            rejected,
            candidate_id="qwen27-affine2-splitk4-vec",
            status="READY_DIAGNOSTIC",
        )
    static = queue.advance_queue(
        rejected,
        candidate_id="qwen27-affine2-splitk4-vec",
        status="STATIC_ONLY",
    )
    assert queue.validate_queue(static)["passed"] is True


def test_protected_pass_requires_complete_physical_metric_set():
    body = queue.build_queue()
    candidate_id = "qwen27-fast-profile"
    with pytest.raises(queue.PhysicalQueueError, match="protected pass requires recorded metrics"):
        queue.advance_queue(
            body,
            candidate_id=candidate_id,
            status="PROTECTED_PASS",
            evidence=("protected.json",),
        )
    measurements = {name: 1 for name in queue.MEASUREMENT_FIELDS}
    measurements["fallback_count"] = 0
    updated = queue.advance_queue(
        body,
        candidate_id=candidate_id,
        status="PROTECTED_PASS",
        evidence=("protected.json",),
        measurements=measurements,
    )
    assert queue.validate_queue(updated)["passed"] is True
    row = next(item for item in updated["candidates"] if item["candidate_id"] == candidate_id)
    assert row["measurements"]["status"] == "RECORDED"
    assert row["status"] == "PROTECTED_PASS"


def test_protected_receipt_adapter_preserves_missing_physical_metrics():
    receipt = _protected_receipt({"HAWKING_QWEN38_FAST": "1"})
    measurements = queue.measurements_from_receipt(
        receipt,
        expected_mutation={"HAWKING_QWEN38_FAST": "1"},
    )
    assert measurements["status"] == "RECORDED"
    assert measurements["complete_wall_ns_per_accepted_token"] == 100
    assert measurements["gpu_ns_per_token"] == 40
    assert measurements["dispatches_per_token"] == 12
    assert measurements["accepted_tps"] == 10_000_000
    assert measurements["fallback_count"] == 0
    for name in (
        "total_nx_bytes",
        "resident_bytes",
        "actual_read_bytes_per_token",
        "transient_bytes_per_token",
        "sync_ns_per_token",
    ):
        assert measurements[name] is None
        assert name in measurements["absence_reasons"]


def test_protected_receipt_adapter_rejects_wrong_class_or_mutation():
    receipt = _protected_receipt({"HAWKING_QWEN38_FAST": "1"})
    wrong_class = json.loads(json.dumps(receipt))
    wrong_class["benchmark_class"] = "DIAGNOSTIC_CONTAMINATED"
    with pytest.raises(queue.PhysicalQueueError, match="QUALIFIED_PROTECTED"):
        queue.measurements_from_receipt(wrong_class)
    with pytest.raises(queue.PhysicalQueueError, match="does not match"):
        queue.measurements_from_receipt(
            receipt,
            expected_mutation={"HAWKING_QWEN38_FAST": "0"},
        )


def test_emit_advanced_queue_can_import_a_protected_receipt(tmp_path: Path):
    queue_path = tmp_path / "queue.json"
    receipt_path = tmp_path / "protected.json"
    output_path = tmp_path / "advanced.json"
    queue.emit_queue(output=queue_path)
    receipt_path.write_text(
        json.dumps(
            _protected_receipt(
                {
                    "HAWKING_QWEN38_FAST": "1",
                    "HAWKING_AFFINE2_GEO": "splitk4",
                    "HAWKING_Q2F_GEO": "tpr64",
                }
            )
        ),
        encoding="utf-8",
    )
    destination = queue.emit_advanced_queue(
        queue_path=queue_path,
        candidate_id="qwen27-affine2-splitk4",
        status="READY_PROTECTED",
        receipt=receipt_path,
        output=output_path,
    )
    body = json.loads(destination.read_text(encoding="utf-8"))
    row = next(item for item in body["candidates"] if item["candidate_id"] == "qwen27-affine2-splitk4")
    assert row["measurements"]["status"] == "RECORDED"
    assert str(receipt_path.resolve()) in row["evidence"]
