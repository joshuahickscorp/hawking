#!/usr/bin/env python3.12
"""Compile Hawking Condensation Doctor research plans without launching compute.

The Doctor is a correction graph, not a synonym for LoRA.  This module is an
inert policy compiler: it records what may be tried, what evidence is required,
and how every correction byte is charged.  It intentionally does not import the
live Studio runner and cannot launch a bake, training job, download, or eval.

Usage:
  python3.12 tools/condense/condensation_doctor.py catalog
  python3.12 tools/condense/condensation_doctor.py plan \
      --model Kimi-K2.6 --params-b 1100 --active-b 32 --moe \
      --target-bpw 0.50 --target-bpw 0.33
  python3.12 tools/condense/condensation_doctor.py selftest
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


SCHEMA = "hawking.condensation_doctor.plan.v1"
SPEC_SCHEMA = "hawking.healer_spec.v1"
ARTIFACT_SCHEMA = "hawking.healer_artifact.v1"
POLICY_VERSION = "2026-07-12.1"
DEFAULT_OUTPUT = Path("reports/condense/condensation_doctor_plan.json")
PHYSICAL_RAM_GIB = 96.0
PROCESS_BUDGET_GB = 78.0
RUNTIME_WEIGHT_TARGET_GB = 64.0


def _method(
    method_id: str,
    horizon: str,
    status: str,
    operator: str,
    complexity: str,
    bandwidth: str,
    latency: str,
    difficulty: str,
    gpu: str,
    apple: str,
    future_hardware: str,
    quantization: str,
    speculation: str,
    distributed: str,
    future_architectures: str,
    *,
    launchable_now: bool = False,
    note: str = "",
) -> dict:
    return {
        "id": method_id,
        "horizon": horizon,
        "implementation_status": status,
        "launchable_now": launchable_now,
        "correction_operator": operator,
        "theoretical_complexity": complexity,
        "expected_bandwidth_reduction": bandwidth,
        "expected_latency_reduction": latency,
        "implementation_difficulty": difficulty,
        "compatibility": {
            "existing_gpus": gpu,
            "apple_silicon": apple,
            "future_specialized_hardware": future_hardware,
        },
        "interactions": {
            "quantization": quantization,
            "speculative_decoding": speculation,
            "distributed_inference": distributed,
            "future_architectures": future_architectures,
        },
        "note": note,
    }


# `status` is deliberately stricter than the older Doctor registry.  "measured"
# does not imply packed/native deployment, and research entries are never emitted
# as runnable commands.
METHODS = (
    _method(
        "scalar_awq_mixed_controls", "immediate", "measured_or_deployable_control",
        "base_rewrite", "encode O(P); sensitivity scan O(T_calib*P)",
        "2-bit payload is ~8x below BF16 before metadata",
        "potentially bandwidth-bound speedup; must be measured end-to-end", "medium",
        "existing scalar/GPTQ kernels", "MLX packed 2-bit/mixed controls; T-MAC CPU control",
        "direct low-bit MAC/LUT", "anchors every lower-rate claim",
        "smaller targets can draft", "tensor cells shard independently",
        "dense and MoE; expert-aware allocation for MoE", launchable_now=True,
    ),
    _method(
        "activation_weighted_residual_svd", "immediate", "oracle_buildable",
        "sidecar_low_rank", "per tensor O(m*n*r), memory O(m*n+r*(m+n))",
        "for square d, F16 sidecar ~=32r/d bpw before container overhead",
        "extra two narrow GEMVs; fuse or schedule without materializing A@B", "medium",
        "portable GEMV", "CPU/MPS oracle now; native Metal correction ABI required",
        "fused packed-base plus low-rank accumulation", "initializes from W-Q(W)",
        "can repair target or drafter", "per-tensor SVD is shard-parallel",
        "applies to attention, FFN, and experts",
    ),
    _method(
        "fixed_byte_parameter_waterfill", "immediate", "planner_buildable",
        "routing_policy", "multiple-choice knapsack O(N*B) or Lagrangian O(N log N)",
        "spends correction bytes only where marginal capability/byte is highest",
        "may reduce correction traffic versus uniform rank", "medium",
        "portable after kernels exist", "requires variable-rank contract and native apply",
        "natural compiler allocation pass", "joint bits/rank/sparse budget",
        "can allocate more quality to verifier-critical tensors", "tensor/expert decisions parallel",
        "extends to SSM state, MoE experts, and multimodal towers",
    ),
    _method(
        "bias_and_sparse_error_sidecars", "immediate", "oracle_buildable",
        "sidecar_bias_sparse", "bias O(mn) offline/O(m) runtime; sparse select O(P log k)",
        "tiny bias plus budgeted high-value exceptions; actual indices are billed",
        "bias is negligible; irregular sparse gathers may erase savings", "medium",
        "generic sparse kernels", "oracle now; packed correction section and fused Metal apply gated",
        "fused bias/sparse accumulator", "select error/output sensitivity, not |w| alone",
        "compatible", "sparse rows shard; indices require deterministic merge",
        "use row-local or structured exceptions on future accelerators",
    ),
    _method(
        "doctor_objective_ablation", "immediate", "partly_implemented",
        "training_only", "O(T_calib*P*epochs)", "no deployed-byte change for fixed artifact",
        "no inference change", "medium", "training portable", "CPU-bf16 is slow but checkpointable",
        "offline only", "top-k KL, tail mass, layer sketch, CE and capability loss",
        "can optimize target acceptance", "teacher caches can be generated/sharded",
        "larger teachers and multimodal losses are gated",
    ),
    _method(
        "quantized_correction_abi", "immediate", "runtime_gated",
        "sidecar_quantized", "encode O(C), runtime O(C_active)",
        "8/4/2-bit corrections cut F16 correction payload by 50/75/87.5% before metadata",
        "positive only with fused decode", "high", "needs custom kernels",
        "native CPU then Metal parity required", "first-class correction opcodes",
        "all correction bytes count toward physical bpw", "compatible",
        "partition corrections with base shards", "supports routed corrections",
    ),
    _method(
        "rotation_hessian_vector_codebook", "medium", "research_gated",
        "base_rewrite", "Hessian/sketch plus codebook optimization; typically superlinear per block",
        "credible 2-bit quality frontier; LUT/codebook/rotation bytes billed",
        "can win only with native lookup/fused rotation", "high", "CUDA references exist",
        "needs MLX/Metal or T-MAC-style port", "lattice/LUT near-memory decode",
        "SpinQuant/QuIP#/AQLM/VPTQ-style lane", "orthogonal to speculation",
        "block/codebook training parallel with deterministic merge", "MoE per-expert codebooks",
    ),
    _method(
        "streamed_block_qat_kd", "medium", "research_gated",
        "base_rewrite_or_sidecar", "O(T_calib*P*epochs), peak one shard/block plus optimizer",
        "same base bpw for re-pack; sidecars billed separately", "quality lever, not inherent speedup",
        "very high", "GPU training references", "requires exact-resume blockwise CPU/MPS path",
        "block-local training engine", "codec-aware only; no uniform-proxy claim",
        "target and drafter can share teacher caches", "blocks/shards distribute; global eval reduces",
        "required for 32B+ and useful for MoE experts",
    ),
    _method(
        "kv_state_and_speculation_codesign", "medium", "partly_implemented",
        "runtime_state", "attention remains O(L*d); cache traffic scales with cache bpw",
        "2-bit KV is ~8x below FP16 before codebooks; rejected work is charged",
        "hypothesis up to acceptance-limited speedup; never assumed", "high",
        "existing CUDA/CPU methods", "MLX cache controls available; custom low-bit cache is gated",
        "cache-native attention and verifier", "separate from weight floor",
        "adaptive draft length/precision", "cache shards and verifier passes communicate",
        "MLA/SSM/state-space layouts need architecture-specific policies",
    ),
    _method(
        "binary_factor_and_pattern_subbit", "medium", "research_oracle",
        "base_rewrite", "iterative block reconstruction; O(T_calib*P*epochs)",
        "target 1.0/0.8/0.5/0.33 physical bpw, not nominal payload",
        "unknown on Apple until native decoder; lookup/factor overhead dominates", "very_high",
        "CUDA research kernels", "CPU/Metal packed round trip and kernel absent",
        "binary factors or pattern LUTs map well to bitwise/near-memory hardware",
        "NanoQuant/BTC/STBLLM-style lane", "sub-bit drafter only if acceptance pays",
        "block/expert factorization parallel", "especially promising for sparse MoE",
    ),
    _method(
        "progressive_event_driven_healer", "long", "paradigm_research",
        "routed_refinement", "base O(P_active); refinement sum_i p_i*C_i",
        "mean bytes/token B0+sum_i p_i*Bi; installed and p95 bytes also billed",
        "wins only when routing avoids enough refinement", "very_high", "prototype possible",
        "unified memory is useful for hot/cold correction paging", "event queues near memory",
        "0.25-0.5 bpw base plus conditional refinements", "risk can trigger verifier/refinement",
        "cold corrections may be remote; miss tails count", "natural for MoE/early-exit/SSM",
    ),
    _method(
        "retrieval_output_repair", "long", "paradigm_research",
        "external_capability", "retrieval O(log N) approximate plus generation",
        "can replace factual parameter traffic but index/network bytes are charged",
        "may hurt latency; route only on expected information gain", "high", "portable",
        "mmap/local ANN plus prompt-cache reuse", "memory-semantic coprocessor",
        "cannot excuse damaged reasoning circuits", "verifier can request evidence",
        "network and cache-hit distributions are first-class", "works across architectures",
    ),
    _method(
        "native_lowbit_training_and_hardware", "long", "paradigm_research",
        "native_model", "pretraining scale", "1-2 bit weights plus structured sparsity",
        "largest possible win when runtime is co-designed", "extreme", "specialized training stack",
        "BitNet/T-MAC provide controls, not conversion proof", "near-memory LUT/bitwise MAC",
        "changes representation rather than repairing PTQ", "self-speculative exits possible",
        "expert/data parallel training", "architectures should be trained for conditional execution",
    ),
)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _target_row(params_b: float, bpw: float) -> dict:
    nominal_gb = params_b * bpw / 8.0
    return {
        "target_effective_bpw": bpw,
        "nominal_weight_gb_before_overhead": round(nominal_gb, 3),
        "within_78gb_only_if_total_physical_bytes_fit": nominal_gb <= PROCESS_BUDGET_GB,
        "within_64gb_runtime_weight_target_before_overhead": nominal_gb <= RUNTIME_WEIGHT_TARGET_GB,
        "warning": (
            "nominal payload is not a deployable rate; actual container, codebook, scale, "
            "pass-through, correction, alignment and metadata bytes are authoritative"
        ),
    }


def build_plan(model: str, params_b: float, active_b: float | None, moe: bool,
               targets: list[float]) -> dict:
    if not model or not math.isfinite(params_b) or params_b <= 0:
        raise ValueError("model and positive finite --params-b are required")
    if active_b is not None and (not math.isfinite(active_b) or active_b <= 0 or active_b > params_b):
        raise ValueError("--active-b must be positive and <= --params-b")
    if not targets or any(not math.isfinite(v) or v <= 0 for v in targets):
        raise ValueError("at least one positive finite --target-bpw is required")

    identity_template = {
        "schema": SPEC_SCHEMA,
        "policy_version": POLICY_VERSION,
        "model": model,
        "params_b": params_b,
        "active_b": active_b,
        "moe": moe,
        "required_hash_bindings": [
            "parent_revision", "config", "tokenizer", "packed_base", "quantizer_binary",
            "codec_recipe", "calibration_set", "selection_set", "final_eval_set",
            "teacher", "mechanism_source",
        ],
        "required_hyperparameters": [
            "mechanism_id", "mechanism_version", "target_modules_and_shapes",
            "rank_or_sparsity_per_module", "correction_dtype_per_module", "steps", "lr",
            "objective", "seed", "optimizer", "sampler", "memory_class",
            "correction_byte_budget", "runtime_abi",
        ],
        "identity_rule": "canonical JSON SHA-256; any changed field is a different config",
    }
    identity_template["template_sha256"] = _canonical_hash(identity_template)

    return {
        "schema": SCHEMA,
        "policy_version": POLICY_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "mode": "plan_only_no_execution",
        "model": {"label": model, "params_b": params_b, "active_b": active_b, "moe": moe},
        "hardware": {
            "profile": "Studio-M3Ultra-96",
            "physical_ram_gib": PHYSICAL_RAM_GIB,
            "process_budget_gb": PROCESS_BUDGET_GB,
            "runtime_weight_target_gb": RUNTIME_WEIGHT_TARGET_GB,
            "policy": "one heavy job; unlimited wall clock; stall watchdogs and atomic checkpoints",
        },
        "targets": [_target_row(params_b, bpw) for bpw in sorted(set(targets), reverse=True)],
        "objective": {
            "maximize": [
                "capability_per_joule", "capability_per_byte_moved",
                "capability_per_resident_byte", "capability_per_parameter",
                "capability_per_wall_clock_second_at_SLO",
            ],
            "subject_to": [
                "paired capability and multiwindow quality gate", "p95 latency SLO",
                "78GB aggregate process envelope", "zero swap", "native packed execution",
            ],
            "not_an_objective": "raw FLOPS or nominal bits alone",
        },
        "healer_spec_identity": identity_template,
        "correction_graph_abi": {
            "equation": "y=K_base(x,packed_base)+sum_i K_i(x,correction_i,route_i)",
            "base_rewrite": "quality credit requires a newly packed, hashed physical base",
            "sidecar": "ordered correction operators; dense fused shadows are forbidden",
            "artifact_schema": ARTIFACT_SCHEMA,
            "required_artifact_fields": [
                "base_hash", "ordered_operator_ids", "operator_file_hashes",
                "actual_bytes_and_alignment", "tensor_coverage", "routing_and_index_state",
                "apply_semantics", "minimum_runtime_version",
            ],
        },
        "rate_contract": {
            "codec_oracle_bpw": "logical baker accounting; never a deployment claim",
            "physical_model_bpw": (
                "8*(packed base + corrections + routers + LUTs + scales + sparse indices + "
                "pass-through tensors + metadata + alignment bytes)/source parameter count"
            ),
            "dynamic": "also report installed bpw and mean/p95/worst bytes moved per token",
            "authority": "actual file bytes plus decoded tensor ownership; no f16 fallback",
        },
        "methods": list(METHODS),
        "experiment_axes": {
            "base_rate": [4, 3, 2, 1, 0.8, 0.5, 0.33, 0.25],
            "base_representation": [
                "scalar_affine", "STRAND", "mixed_precision", "dense_sparse",
                "residual", "vector_codebook", "binary_factor", "binary_pattern",
            ],
            "transform": ["none", "RHT", "activation_scale", "learned_rotation"],
            "correction": [
                "zero", "bias", "residual_svd", "targeted_rank", "rank_plus_sparse",
                "quantized_sidecar", "codec_rewrite", "block_qat", "teacher_kd",
            ],
            "correction_budget": "waterfill ranks/sparsity/precision under exact physical bytes",
            "selection_controls": [
                "three seeds", "domain-matched and multi-domain calibration",
                "independent frozen selection set", "random allocation", "zero correction",
            ],
            "runtime": [
                "CPU packed", "Metal packed", "KV16/8/4/2", "no speculation",
                "adaptive speculative decoding", "cold/warm cache", "short/long context",
            ],
        },
        "adaptive_campaign": [
            {
                "phase": 0, "name": "identity_and_controls",
                "rule": "capture f16/native parent, zero correction, packed 4/3/2-bit controls",
            },
            {
                "phase": 1, "name": "broad_screen",
                "rule": "run every representation/treatment family on small calibration and retain all negatives",
            },
            {
                "phase": 2, "name": "successive_halving",
                "rule": (
                    "allocate 4x more tokens/steps only to non-dominated candidates; a higher rank "
                    "is admitted only after positive held-out recovery per added serialized byte"
                ),
            },
            {
                "phase": 3, "name": "full_science",
                "rule": "three seeds, calibration ablations, multiwindow>=4, capability/task tripwire",
            },
            {
                "phase": 4, "name": "physical_promotion",
                "rule": "packed round trip -> native parity -> resident execution; oracle rows cannot skip states",
            },
            {
                "phase": 5, "name": "system_efficiency",
                "rule": "same-box bytes/J/tokens/s/p95 plus KV/speculation/distributed accounting",
            },
        ],
        "proof_state_machine": [
            "reconstruction_oracle", "packed_artifact", "native_resident_runtime",
            "capability_efficiency_promoted",
        ],
        "stop_rules": {
            "wall_clock": "none",
            "safety": "checkpoint and pause on non-normal pressure, swap>0, thermal warning, drain, or disk reserve",
            "scientific": (
                "stop a branch only after replicated non-improvement/dominance; never stop the campaign "
                "because one Doctor regresses; the exact zero-correction artifact is the fallback"
            ),
        },
        "launch_blocker": (
            "This artifact is an inert plan. Each method needs an implementation-specific signed "
            "HealerSpec and existing heavy-work lease before any executor may launch it."
        ),
    }


def _selftest() -> int:
    p1 = build_plan("selftest-1.6T", 1600.0, 49.0, True, [0.5, 0.33, 0.25])
    p2 = build_plan("selftest-1.6T", 1600.0, 49.0, True, [0.25, 0.33, 0.5])
    assert p1["healer_spec_identity"]["template_sha256"] == p2["healer_spec_identity"]["template_sha256"]
    rows = {r["target_effective_bpw"]: r for r in p1["targets"]}
    assert rows[0.5]["nominal_weight_gb_before_overhead"] == 100.0
    assert rows[0.33]["nominal_weight_gb_before_overhead"] == 66.0
    assert rows[0.25]["nominal_weight_gb_before_overhead"] == 50.0
    assert not rows[0.5]["within_78gb_only_if_total_physical_bytes_fit"]
    assert rows[0.25]["within_64gb_runtime_weight_target_before_overhead"]
    assert all(not m["launchable_now"] or m["horizon"] == "immediate" for m in METHODS)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "nested" / "plan.json"
        _atomic_json(out, p1)
        loaded = json.loads(out.read_text())
        assert loaded["schema"] == SCHEMA and loaded["mode"] == "plan_only_no_execution"
    print("condensation_doctor.py selftest OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile an inert Condensation Doctor plan")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("catalog")
    plan = sub.add_parser("plan")
    plan.add_argument("--model", required=True)
    plan.add_argument("--params-b", required=True, type=float)
    plan.add_argument("--active-b", type=float)
    plan.add_argument("--moe", action="store_true")
    plan.add_argument("--target-bpw", action="append", required=True, type=float)
    plan.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sub.add_parser("selftest")
    args = parser.parse_args()

    if args.command == "catalog":
        print(json.dumps({"policy_version": POLICY_VERSION, "methods": METHODS}, indent=2))
        return 0
    if args.command == "selftest":
        return _selftest()
    compiled = build_plan(args.model, args.params_b, args.active_b, args.moe, args.target_bpw)
    _atomic_json(args.output, compiled)
    print(f"wrote inert plan: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
