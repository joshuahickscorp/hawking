#!/usr/bin/env python3.12
"""Compile and schedule Doctor-v2's expansive treatment search.

This is a research-space compiler, not a model executor.  It deterministically
expands model/rate/operator axes into explicit experiment identities, preserves
negative controls, and supports multi-fidelity next-candidate selection.  A
candidate becomes runnable only after its full HealerProgram is materialized,
validated, source-bound, and admitted by the existing heavy-work lease.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import healer_abi as abi  # noqa: E402


CAMPAIGN_SCHEMA = "hawking.doctor_frontier_campaign.v2"
SELECTION_SCHEMA = "hawking.doctor_frontier_selection.v2"
CAMPAIGN_VERSION = "2026-07-12.1"
DEFAULT_OUTPUT = Path("reports/condense/doctor_v2_frontier_campaign.json")

LAB_MODELS = (
    {"label": "Qwen2.5-0.5B", "params_b": 0.5, "active_b": None, "moe": False,
     "rates": (4.0, 3.0, 2.0, 1.0, 0.8, 0.55, 0.3, 0.1)},
    {"label": "Qwen2.5-1.5B", "params_b": 1.5, "active_b": None, "moe": False,
     "rates": (4.0, 3.0, 2.0, 1.0, 0.8, 0.55, 0.3, 0.1)},
    {"label": "Qwen2.5-7B", "params_b": 7.0, "active_b": None, "moe": False,
     "rates": (4.0, 3.0, 2.0, 1.0, 0.8, 0.55, 0.3, 0.1)},
    {"label": "Qwen2.5-14B", "params_b": 14.0, "active_b": None, "moe": False,
     "rates": (4.0, 3.0, 2.0, 1.0, 0.8, 0.55, 0.3, 0.1)},
    {"label": "Qwen2.5-32B", "params_b": 32.0, "active_b": None, "moe": False,
     "rates": (4.0, 3.0, 2.0, 1.0, 0.8, 0.55, 0.3, 0.1)},
    {"label": "Qwen2.5-72B", "params_b": 72.0, "active_b": None, "moe": False,
     "rates": (3.0, 2.0, 1.0, 0.8, 0.55, 0.3, 0.1)},
)

FRONTIER_MODELS = (
    {"label": "gpt-oss-120B", "params_b": 117.0, "active_b": 5.1, "moe": True,
     "rates": (3.0, 2.0, 1.0, 0.8, 0.55, 0.3, 0.1)},
    {"label": "DeepSeek-V4-Flash", "params_b": 284.0, "active_b": 13.0, "moe": True,
     "rates": (2.0, 1.34, 1.0, 0.8, 0.55, 0.5, 0.3, 0.1)},
    {"label": "Kimi-K2.6", "params_b": 1100.0, "active_b": 32.0, "moe": True,
     "rates": (0.8, 0.55, 0.5, 0.33, 0.3, 0.25, 0.1)},
    {"label": "DeepSeek-V4-Pro", "params_b": 1600.0, "active_b": 49.0, "moe": True,
     "rates": (0.8, 0.55, 0.5, 0.33, 0.3, 0.25, 0.1)},
)
CAMPAIGN_MODELS = LAB_MODELS + FRONTIER_MODELS


def _entry(category: str, status: str, kind: str, phase: str, source: str,
           complexity: str, apple: str, cuda: str, *, min_bpw: float = 0.0,
           max_bpw: float = 16.0, moe_only: bool = False, dynamic: bool = False,
           executable: bool = False, note: str = "") -> dict[str, Any]:
    return {
        "category": category, "implementation_status": status, "kind": kind,
        "phase": phase, "primary_source": source, "complexity": complexity,
        "apple": apple, "cuda": cuda, "min_bpw": min_bpw, "max_bpw": max_bpw,
        "moe_only": moe_only, "dynamic": dynamic, "executor_wired": executable,
        "note": note,
    }


# The catalog deliberately mixes current Hawking operators, external controls,
# and unimplemented frontier ideas.  The implementation state is part of every
# experiment identity; references are inspiration, never imported evidence.
CATALOG: dict[str, dict[str, Any]] = {
    # Diagnosis: the prescription is allowed to vary by tensor, token, domain,
    # capability syndrome, and expert; weight RMSE alone is never sufficient.
    "weight_error_spectrum": _entry(
        "diagnostic", "prototype", "analyze", "offline", "local:doctor-v2",
        "O(P) statistics plus selected SVD sketches", "stdlib/CPU orchestration", "portable",
    ),
    "activation_hessian_sketch": _entry(
        "diagnostic", "research", "analyze", "offline", "https://arxiv.org/abs/2506.05664",
        "O(T_calib*P) activations plus block sketches", "streamed CPU/MPS design needed", "CUDA research",
    ),
    "cka_entropy_token_risk": _entry(
        "diagnostic", "research", "analyze", "offline", "https://arxiv.org/abs/2606.11244",
        "paired parent/student activation diagnostics", "core token-gated Apple research", "CUDA paper reference",
    ),
    "capability_failure_clusters": _entry(
        "diagnostic", "research", "analyze", "offline", "https://arxiv.org/abs/2501.03035",
        "paired task traces plus clustering/localization", "backend-neutral", "backend-neutral",
    ),
    "expert_affinity_hotness": _entry(
        "diagnostic", "research", "analyze", "offline", "https://arxiv.org/abs/2505.03804",
        "router/expert activation statistics", "future Kimi/V4 adapter required", "MoE research",
        moe_only=True,
    ),
    "causal_capability_islands": _entry(
        "diagnostic", "unimplemented", "analyze", "offline", "local:hawking-hypothesis",
        "activation patching/causal tracing over failure clusters", "streamed paired tracing required",
        "backend-neutral analysis", note="preserve structured circuits, not isolated super-weight fine-tuning",
    ),
    "quant_error_syndrome_model": _entry(
        "diagnostic", "unimplemented", "analyze", "offline", "local:hawking-hypothesis",
        "learn parent/student hidden-logit divergence predictor", "small gate can run on Metal",
        "portable", note="predict which correction stream a token/layer requires",
    ),
    # Base representations.
    "strand_scalar": _entry(
        "base", "measured", "base_codec", "offline", "local:strand-quant",
        "O(P) encode and O(P_active) decode", "oracle/current scalar path", "portable work",
        min_bpw=1.0, max_bpw=4.5, executable=True,
    ),
    "mlx_affine": _entry(
        "base", "prototype", "base_codec", "offline", "https://github.com/ml-explore/mlx-lm",
        "O(P)", "native packed Metal control", "portable affine control",
        min_bpw=2.0, max_bpw=6.0,
    ),
    "hardware_aligned_mixed": _entry(
        "base", "research", "base_codec", "offline", "https://arxiv.org/abs/2602.17698",
        "sensitivity plus global constrained allocation", "compiler/runtime port required",
        "research reference", min_bpw=1.0, max_bpw=4.0,
    ),
    "residual_refinement": _entry(
        "base", "oracle", "base_rewrite", "offline", "https://arxiv.org/abs/2511.21736",
        "O(passes*P)", "existing residual concepts; exact native parity gated", "portable",
        min_bpw=1.0, max_bpw=3.0,
    ),
    "aqlm_additive": _entry(
        "base", "research", "base_codec", "offline", "https://arxiv.org/abs/2401.06118",
        "iterative additive-codebook block optimization", "Metal lookup kernel absent",
        "CUDA reference exists", min_bpw=1.5, max_bpw=3.5,
    ),
    "quip_lattice": _entry(
        "base", "research", "base_codec", "offline", "https://arxiv.org/abs/2402.04396",
        "Hessian/incoherence plus lattice assignment", "Metal lattice kernel absent",
        "CUDA reference exists", min_bpw=1.5, max_bpw=3.5,
    ),
    "unisvq_integer_vq": _entry(
        "base", "research", "base_codec", "offline", "https://arxiv.org/abs/2606.10520",
        "vector-codebook QAT", "integer-lattice Metal port proposed", "CUDA research",
        min_bpw=1.5, max_bpw=3.0,
    ),
    "binary_factor": _entry(
        "base", "research", "base_codec", "offline", "https://arxiv.org/abs/2602.06694",
        "ADMM/block reconstruction over binary low-rank factors", "packed CPU/Metal absent",
        "CUDA reference", min_bpw=0.20, max_bpw=1.2,
    ),
    "binary_pattern": _entry(
        "base", "research", "base_codec", "offline", "https://arxiv.org/abs/2506.12040",
        "learned transform plus binary-pattern clustering", "LUT path promising but absent",
        "research reference", min_bpw=0.4, max_bpw=1.2,
    ),
    "littlebit_factor": _entry(
        "base", "research", "base_codec", "offline", "https://arxiv.org/abs/2506.13771",
        "binarized latent factors plus multiscale compensation", "native decoder absent",
        "research reference", min_bpw=0.1, max_bpw=0.9,
    ),
    "structured_binary": _entry(
        "base", "research", "base_codec", "offline", "https://arxiv.org/abs/2408.01803",
        "layer-wise N:M selection plus binary regions", "Apple structured-bit kernel absent",
        "CUDA kernel reference", min_bpw=0.25, max_bpw=1.0,
    ),
    "qmoe_subbit": _entry(
        "base", "research", "base_codec", "offline", "https://arxiv.org/abs/2310.16795",
        "expert-wise ternary/low-bit compression", "expert pager/kernel absent",
        "CUDA reference", min_bpw=0.5, max_bpw=1.2, moe_only=True,
    ),
    "bwla_binary_ptq": _entry(
        "base", "research", "base_codec", "offline", "https://arxiv.org/abs/2605.00422",
        "orthogonal-Kronecker transform plus proximal-SVD repair", "Apple kernel absent",
        "research reference", min_bpw=0.8, max_bpw=1.2,
    ),
    "lcqat_vector_qat": _entry(
        "base", "research", "base_codec", "offline", "https://arxiv.org/abs/2606.10531",
        "differentiable vector-codebook QAT", "integer/codebook Metal lowering absent",
        "CUDA research", min_bpw=1.5, max_bpw=2.5,
    ),
    "matgptq_sliceable": _entry(
        "base", "research", "base_codec", "offline", "https://arxiv.org/abs/2602.03537",
        "progressive/sliceable matrix quantization", "progressive packed runtime absent",
        "research reference", min_bpw=0.25, max_bpw=4.0,
    ),
    "shared_parameter_grammar": _entry(
        "base", "unimplemented", "base_codec", "offline", "local:hawking-hypothesis",
        "learn shared layer/expert dictionaries or a block generator plus exceptions",
        "requires direct generated-tile consumption", "future tensor-core/near-memory target",
        min_bpw=0.1, max_bpw=2.0,
        note="exploit mutual information across layers and experts rather than quantizing tensors independently",
    ),

    # Exact-function transforms and allocation diagnostics.
    "identity_transform": _entry(
        "transform", "measured", "base_transform", "offline", "local:control",
        "O(1)", "control", "control", executable=True,
    ),
    "rht_incoherence": _entry(
        "transform", "measured", "base_transform", "offline", "local:strand-quant",
        "O(P log d)", "current oracle", "portable", executable=True,
    ),
    "activation_scaling": _entry(
        "transform", "measured", "base_transform", "offline", "https://arxiv.org/abs/2306.00978",
        "O(T_calib*P)", "current act-mean approximation", "portable", executable=True,
    ),
    "learned_rotation": _entry(
        "transform", "research", "base_transform", "offline", "https://arxiv.org/abs/2405.16406",
        "rotation optimization plus transform", "Metal fusion absent", "CUDA reference",
    ),
    "residual_subspace_rotation": _entry(
        "transform", "research", "base_transform", "offline", "https://arxiv.org/abs/2604.11080",
        "layer-wise residual subspace optimization", "offline fusion research", "research",
    ),
    "latent_geometry_rotation": _entry(
        "transform", "research", "base_transform", "offline", "https://arxiv.org/abs/2603.00042",
        "joint iterative rotation of binary latent factors", "folded transform is attractive; implementation absent",
        "research reference",
    ),
    "channel_reorder": _entry(
        "transform", "research", "base_transform", "offline", "https://arxiv.org/abs/2602.17698",
        "bi-directional channel reorder", "compiler/runtime contract required", "research",
    ),

    # Recovery operators.
    "zero_correction": _entry(
        "correction", "measured", "static_correction", "offline", "local:control",
        "O(1)", "control", "control", executable=True,
    ),
    "output_bias": _entry(
        "correction", "oracle", "static_correction", "offline", "local:doctor-v2",
        "O(P) offline/O(rows) runtime", "DBIA packing absent", "portable",
    ),
    "residual_svd": _entry(
        "correction", "oracle", "static_correction", "offline", "https://arxiv.org/abs/2310.08659",
        "O(m*n*r) per tensor", "factor oracle feasible; native apply absent", "portable",
    ),
    "rank_waterfill": _entry(
        "correction", "prototype", "static_correction", "offline", "local:doctor-v2",
        "multiple-choice knapsack", "variable-rank ABI required", "portable compiler",
    ),
    "rank_sparse_error": _entry(
        "correction", "research", "static_correction", "offline", "https://arxiv.org/abs/2306.03078",
        "low-rank fit plus sparse top-k/error allocation", "fused apply absent", "portable research",
    ),
    "module_adaptive_residual": _entry(
        "correction", "research", "static_correction", "offline", "https://arxiv.org/abs/2605.17997",
        "module feedback/PID residual scaling", "new optimizer/runtime contract", "research",
    ),
    "token_gated_error_compensator": _entry(
        "correction", "research", "gated_correction", "decode", "https://arxiv.org/abs/2606.11244",
        "O(gate)+sum p_i*O(C_i)", "central Apple event-driven frontier", "CUDA paper reference",
        dynamic=True,
    ),
    "codec_block_qat": _entry(
        "correction", "runtime_gated", "base_rewrite", "offline", "https://arxiv.org/abs/2407.11062",
        "O(T*P*epochs), block resident", "exact-resume streamed implementation absent", "CUDA reference",
    ),
    "self_distill": _entry(
        "correction", "prototype", "static_correction", "offline", "https://arxiv.org/abs/2402.10631",
        "O(T*P*epochs)", "top-k KL subset exists", "portable", executable=False,
    ),
    "large_teacher_distill": _entry(
        "correction", "unimplemented", "static_correction", "offline", "local:doctor-v2",
        "teacher+student streamed forwards", "streamed teacher cache absent", "distributed friendly",
    ),
    "capability_targeted_tune": _entry(
        "correction", "research", "static_correction", "offline", "https://arxiv.org/abs/2501.03035",
        "task-local fine-tuning", "small-model feasible", "portable",
    ),
    "progressive_refinement_streams": _entry(
        "correction", "unimplemented", "gated_correction", "decode", "local:doctor-v2",
        "B0 + sum p_i*Bi", "core Hawking progressive-code research", "backend neutral",
        dynamic=True,
    ),
    "nested_precision_slices": _entry(
        "correction", "research", "gated_correction", "decode", "https://arxiv.org/abs/2602.03537",
        "ordered nested precision/residual slices", "progressive packed Metal path absent", "research",
        dynamic=True,
    ),
    "entropy_tile_coding": _entry(
        "correction", "research", "static_correction", "load", "https://arxiv.org/abs/2606.15789",
        "O(P) tile-aligned entropy encode/decode", "random-access compute-tile integration absent",
        "research", note="must decode directly into the consumer; a dense shadow does not lower residency",
    ),
    "capability_adapter_bank": _entry(
        "correction", "unimplemented", "gated_correction", "decode", "local:hawking-hypothesis",
        "router plus small domain/capability-specific structured corrections",
        "hot adapters resident; cold adapters mmap/prefetch", "portable routed experts", dynamic=True,
    ),
    "active_teacher_failure_mining": _entry(
        "correction", "unimplemented", "train", "offline", "local:hawking-hypothesis",
        "teacher is queried only on high-uncertainty or divergent traces", "offline Apple feasible at small scale",
        "distributed teacher generation", note="spend teacher compute where it changes the Pareto frontier",
    ),
    "cross_expert_dictionary_repair": _entry(
        "correction", "unimplemented", "static_correction", "offline", "local:hawking-hypothesis",
        "shared low-rank/codebook basis plus expert-specific coefficients", "Kimi/V4 adapter required",
        "expert-parallel friendly", moe_only=True,
    ),

    # State and runtime policies.
    "kv_fp16": _entry(
        "state", "measured", "state_codec", "decode", "local:baseline",
        "O(L*d)", "current baseline", "baseline", executable=True, dynamic=True,
    ),
    "kv_int4": _entry(
        "state", "prototype", "state_codec", "decode", "https://github.com/ml-explore/mlx-lm",
        "O(L*d)", "MLX control/integration needed", "portable", dynamic=True,
    ),
    "kv_int2": _entry(
        "state", "research", "state_codec", "decode", "https://arxiv.org/abs/2402.02750",
        "O(L*d)", "Metal kernel absent", "CUDA reference", dynamic=True,
    ),
    "kv_codebook": _entry(
        "state", "research", "state_codec", "decode", "https://machinelearning.apple.com/research/commutative-vector-quantization",
        "O(L*d) plus codebook lookup", "Apple-origin method; integration absent", "portable research",
        dynamic=True,
    ),
    "spec_none": _entry(
        "runtime", "measured", "runtime_policy", "decode", "local:control",
        "target only", "control", "control", dynamic=True, executable=True,
    ),
    "history_suffix_draft": _entry(
        "runtime", "runtime_gated", "runtime_policy", "decode", "local:spec-readiness",
        "lookup plus verify", "only proposal worth a fresh oracle; parity gated", "portable",
        dynamic=True,
    ),
    "quantized_neural_draft": _entry(
        "runtime", "research", "runtime_policy", "decode", "https://machinelearning.apple.com/research/quantspec",
        "draft+batched target verification", "TQ verifier parity absent", "CUDA portable",
        dynamic=True,
    ),
    "recurrent_draft": _entry(
        "runtime", "research", "runtime_policy", "decode", "https://machinelearning.apple.com/research/recurrent-drafter",
        "recurrent draft+verify", "MLX research control", "portable", dynamic=True,
    ),
    "self_spec_early_exit": _entry(
        "runtime", "research", "runtime_policy", "decode", "https://ai.meta.com/research/publications/layerskip-enabling-early-exit-inference-and-self-speculative-decoding/",
        "partial target+remaining verifier", "architecture training/Metal path absent", "portable",
        dynamic=True,
    ),
    "nested_target_drafter": _entry(
        "runtime", "unimplemented", "runtime_policy", "decode", "local:hawking-hypothesis",
        "0.10-0.25 prefix drafts; ordered enhancement slices verify with shared state",
        "requires TQ-native transactional batched verification", "portable after semantic parity",
        dynamic=True, note="avoid a separately resident drafter when the target representation is nested",
    ),
    "on_demand_weight_synthesis": _entry(
        "runtime", "unimplemented", "runtime_policy", "decode", "local:hawking-hypothesis",
        "generate/cache hot weight tiles from a compact latent grammar", "unified-memory cache experiment",
        "GPU graph/cache research", dynamic=True,
        note="compute-for-bandwidth trade; generated dense shadows are bounded cache, never artifact accounting",
    ),
    "retrieval_repair": _entry(
        "runtime", "research", "retrieval", "prefill", "https://arxiv.org/abs/2112.04426",
        "ANN+prompt processing", "mmap/ANN prototype possible", "portable", dynamic=True,
    ),
    "output_verifier_repair": _entry(
        "runtime", "unimplemented", "verifier", "post_decode", "local:doctor-v2",
        "risk gate+verification+selective regeneration", "new capability lane", "portable",
        dynamic=True,
    ),
    "packed_correction_package": _entry(
        "fixed", "runtime_gated", "package", "load", "local:healer-abi-v2",
        "O(artifact bytes)", "v2 runtime sections are not implemented", "backend-neutral ABI",
    ),
    "paired_capability_efficiency_eval": _entry(
        "fixed", "prototype", "evaluate", "post_decode", "local:doctor-v2",
        "O(eval tokens * active model cost)", "existing evaluators need streamed v2 artifacts", "portable",
        dynamic=True,
    ),
}


BASES = tuple(k for k, v in CATALOG.items() if v["category"] == "base")
DIAGNOSTICS = tuple(k for k, v in CATALOG.items() if v["category"] == "diagnostic")
TRANSFORMS = tuple(k for k, v in CATALOG.items() if v["category"] == "transform")
CORRECTIONS = tuple(k for k, v in CATALOG.items() if v["category"] == "correction")
STATES = tuple(k for k, v in CATALOG.items() if v["category"] == "state")
RUNTIMES = tuple(k for k, v in CATALOG.items() if v["category"] == "runtime")
OBJECTIVES = ("layer_output", "topk_tail_kl", "feature_logit", "capability_composite")
CALIBRATIONS = ("multidomain", "domain_matched", "high_quant_error", "reasoning_hard")
SEEDS = (17, 29, 43)


def _eligible(name: str, model: dict[str, Any], bpw: float) -> bool:
    row = CATALOG[name]
    return row["min_bpw"] <= bpw <= row["max_bpw"] and (not row["moe_only"] or model["moe"])


def _implementation(names: Iterable[str]) -> tuple[str, list[str]]:
    blockers: list[str] = []
    states = []
    for name in names:
        row = CATALOG[name]
        states.append(row["implementation_status"])
        if not row["executor_wired"]:
            blockers.append(f"{name}: executor is not wired")
        if row["implementation_status"] in {"research", "unimplemented", "runtime_gated"}:
            blockers.append(f"{name}: {row['implementation_status']}")
    order = ["measured", "prototype", "oracle", "runtime_gated", "research", "unimplemented"]
    status = max(states, key=lambda value: order.index(value) if value in order else len(order))
    return status, blockers


def _candidate(model: dict[str, Any], bpw: float, diagnostic: str, base: str, transform: str,
               correction: str, state: str, runtime: str, objective: str,
               calibration: str, seed: int) -> dict[str, Any]:
    compact = {
        "model": model["label"], "params_b": model["params_b"], "active_b": model["active_b"],
        "moe": model["moe"], "target_bpw": bpw, "diagnostic": diagnostic,
        "base": base, "transform": transform,
        "correction": correction, "state": state, "runtime": runtime,
        "objective": objective, "calibration": calibration, "seed": seed,
        "campaign_version": CAMPAIGN_VERSION,
    }
    identity = abi.hash_value(compact)
    components = (diagnostic, base, transform, correction, state, runtime)
    status, blockers = _implementation(components)
    dynamic = any(CATALOG[name]["dynamic"] for name in components)
    if model["params_b"] >= 32:
        blockers.append("streamed model/evaluation path required")
    return {
        "experiment_id": f"dr2-{identity[:16]}",
        "identity_sha256": identity,
        **compact,
        "implementation_status": status,
        "launchable": not blockers,
        "blockers": blockers,
        "resource_class": "streamed" if model["params_b"] >= 32 else "resident",
        "dynamic_cost_required": dynamic,
        "fidelity": "F0",
        "status": "pending",
        "proof_state": "planned",
    }


def _space_for(model: dict[str, Any], bpw: float) -> Iterable[dict[str, Any]]:
    dimensions = _dimensions(model, bpw)
    # System axes are intentionally broad.  Invalid or unwired combinations are
    # retained as planned research; the queue never interprets breadth as launch permission.
    for values in itertools.product(*dimensions):
        yield _candidate(model, bpw, *values)


def _dimensions(model: dict[str, Any], bpw: float) -> tuple[tuple[Any, ...], ...]:
    return (
        tuple(name for name in DIAGNOSTICS if _eligible(name, model, bpw)),
        tuple(name for name in BASES if _eligible(name, model, bpw)),
        TRANSFORMS, CORRECTIONS, STATES, RUNTIMES, OBJECTIVES, CALIBRATIONS, SEEDS,
    )


def _candidate_at(model: dict[str, Any], bpw: float, index: int) -> dict[str, Any]:
    dimensions = _dimensions(model, bpw)
    total = math.prod(len(values) for values in dimensions)
    if index < 0 or index >= total:
        raise IndexError(index)
    selected: list[Any] = [None] * len(dimensions)
    cursor = index
    for position in range(len(dimensions) - 1, -1, -1):
        values = dimensions[position]
        cursor, slot = divmod(cursor, len(values))
        selected[position] = values[slot]
    return _candidate(model, bpw, *selected)


def _mandatory_controls(models: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for model in models:
        for bpw in model["rates"]:
            for base in BASES:
                if _eligible(base, model, bpw):
                    out.append(_candidate(
                        model, bpw, "weight_error_spectrum", base,
                        "identity_transform", "zero_correction",
                        "kv_fp16", "spec_none", "layer_output", "multidomain", SEEDS[0],
                    ))
    return out


def compile_campaign(*, models: tuple[dict[str, Any], ...] = CAMPAIGN_MODELS,
                     max_explicit: int = 4096) -> dict[str, Any]:
    if max_explicit < 128:
        raise ValueError("max_explicit must be >=128")
    projected = 0
    spaces: list[tuple[dict[str, Any], float, int]] = []
    for model in models:
        for bpw in model["rates"]:
            count = math.prod(len(values) for values in _dimensions(model, bpw))
            projected += count
            spaces.append((model, bpw, count))

    mandatory = _mandatory_controls(models)
    by_id = {row["experiment_id"]: row for row in mandatory}
    # Sample the huge Cartesian product by mixed-radix unranking.  This creates
    # O(explicit) candidates rather than scanning O(projected) cells.
    cumulative: list[int] = []
    running = 0
    for _model, _bpw, count in spaces:
        running += count
        cumulative.append(running)
    step = max(1, int(projected * 0.6180339887498949)) | 1
    while math.gcd(step, projected) != 1:
        step += 2
    offset = int(abi.hash_value({"campaign": CAMPAIGN_VERSION, "models": models})[:16], 16) % projected
    cursor = 0
    max_attempts = max_explicit * 8
    while len(by_id) < max_explicit and cursor < max_attempts:
        global_index = (offset + cursor * step) % projected
        space_index = bisect.bisect_right(cumulative, global_index)
        start = cumulative[space_index - 1] if space_index else 0
        model, bpw, _count = spaces[space_index]
        row = _candidate_at(model, bpw, global_index - start)
        by_id.setdefault(row["experiment_id"], row)
        cursor += 1
    if len(by_id) < max_explicit:
        raise RuntimeError("deterministic sampler could not fill explicit campaign")

    experiments = sorted(by_id.values(), key=lambda row: row["experiment_id"])
    campaign = {
        "schema": CAMPAIGN_SCHEMA,
        "campaign_version": CAMPAIGN_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "mode": "plan_only_no_execution",
        "models": list(models),
        "operator_catalog": CATALOG,
        "axes": {
            "diagnostics": list(DIAGNOSTICS), "bases": list(BASES), "transforms": list(TRANSFORMS),
            "corrections": list(CORRECTIONS), "states": list(STATES),
            "runtimes": list(RUNTIMES), "objectives": list(OBJECTIVES),
            "calibrations": list(CALIBRATIONS), "seeds": list(SEEDS),
        },
        "search_size": {
            "projected_cartesian_candidates": projected,
            "explicit_candidates": len(experiments),
            "mandatory_controls": len(mandatory),
            "expansion": "materialize additional deterministic cells after evidence-driven promotion",
        },
        "experiments": experiments,
        "fidelity_ladder": [
            {"id": "F0", "work": "weight/statistics and exact-byte feasibility", "budget_multiplier": 1},
            {"id": "F1", "work": "representative tensor/layer output reconstruction", "budget_multiplier": 4},
            {"id": "F2", "work": "representative shard plus activation/logit sketches", "budget_multiplier": 16},
            {"id": "F3", "work": "full streamed model, multiwindow>=4, capability tripwire", "budget_multiplier": 64},
            {"id": "F4", "work": "three seeds, domain and hard-example ablations", "budget_multiplier": 192},
            {"id": "F5", "work": "packed round trip and native CPU/Metal parity", "budget_multiplier": 256},
            {"id": "F6", "work": "same-box KV/speculation/energy/latency; later separate CUDA receipt", "budget_multiplier": 512},
        ],
        "scheduler": {
            "wall_clock_limit": None,
            "one_heavy_lease": True,
            "selection": "Pareto frontier plus expected value of information",
            "mandatory": "zero correction, equal-byte, random allocation, three seeds, negative retention",
            "promotion": "4x budget only after non-dominance or high uncertainty at a decision boundary",
            "metrics": [
                "capability", "ppl", "teacher_kl", "physical_bytes", "resident_bytes",
                "mean_p95_worst_bytes_per_token", "joules_per_accepted_token",
                "cold_warm_latency", "p95_latency", "spec_acceptance", "verifier_equivalent_forwards",
            ],
            "safety": "existing heavy lease; normal pressure; zero swap; AC/thermal/disk gates",
        },
        "regime_policy": {
            "above_2_bits": "PTQ plus last-mile correction remains a valid first hypothesis",
            "at_or_below_2_bits": (
                "treat as representation reconstruction: codebook/factor/QAT/base-rewrite families "
                "must receive coverage; LoRA-only recovery cannot monopolize the search"
            ),
            "sub_1_bit": (
                "binary factors, binary patterns, structured sparsity, progressive slices, and MoE-aware "
                "formats are mandatory families; 0.1 bpw is a destructive stress control, not an assumed win"
            ),
        },
        "transfer_policy": {
            "fingerprint": [
                "architecture", "semantic_tensor_role", "log_params", "log_active_params",
                "depth_width_heads_experts", "activation_energy", "outlier_statistics",
                "Hessian_or_Fisher_sketch", "error_spectrum", "singular_value_decay",
                "expert_hotness", "routing_entropy", "token_error_risk",
            ],
            "normalization": [
                "capability_recovered_per_serialized_byte", "per_moved_byte", "per_train_token",
                "per_active_parameter",
            ],
            "rule": "small-model evidence changes queue priority only; every larger scale needs held-out confirmation",
        },
        "backend_policy": {
            "apple": "primary proof backend; CPU oracle then native Metal parity",
            "cuda": "future independent backend receipt; never merged into Apple headline rows",
            "distributed": "shard/offline parallelism plus explicit communication-byte accounting",
        },
        "velocity_plus_plus_contract": {
            "identity": [
                "target_artifact_sha256", "physical_bpw", "healer_program_sha256", "tokenizer_sha256",
                "backend_and_kernel_sha256", "drafter_family_and_artifact_sha256",
                "verifier_path_sha256", "kv_precision", "cache_namespace_sha256",
                "adaptive_policy_sha256", "workload_sha256", "seed",
            ],
            "current_blockers": [
                "batched Qwen verifier does not execute the exact TQ target used by single-token decode",
                "verifier cost curve is not target/KV/context/backend specific",
                "runtime router does not yet receive complete draft/verify/sync/rollback timing and energy",
                "transactional speculative KV and correction-graph cache namespaces are not proven",
            ],
            "goodput": "expected_committed/(draft_time+verify_time+sync_time+rollback_time)",
            "energy": (
                "(J_draft+J_verify+J_sync+J_KV_writes+J_rejected+J_cache_miss)"
                "/committed_target_tokens"
            ),
            "nested_hypothesis": (
                "use a 0.10-0.25 progressive representation prefix as drafter and enhancement slices "
                "as the same hash-bound target, reusing state instead of assuming two independent models"
            ),
            "promotion": "no live speculative cell until exact TQ B=1..8 parity and conservative utility LCB pass",
        },
        "campaign_sha256": None,
    }
    campaign["campaign_sha256"] = abi.hash_value({k: v for k, v in campaign.items() if k not in {"campaign_sha256", "generated_at"}})
    return campaign


def validate_campaign(campaign: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(campaign, dict):
        return ["campaign must be an object"]
    if campaign.get("schema") != CAMPAIGN_SCHEMA:
        errors.append(f"schema must be {CAMPAIGN_SCHEMA}")
    if campaign.get("campaign_version") != CAMPAIGN_VERSION:
        errors.append(f"campaign_version must be {CAMPAIGN_VERSION}")
    experiments = campaign.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        errors.append("experiments must be a non-empty list")
        experiments = []
    seen: set[str] = set()
    for idx, row in enumerate(experiments):
        if not isinstance(row, dict):
            errors.append(f"experiments[{idx}] is not an object")
            continue
        experiment_id = row.get("experiment_id")
        if not isinstance(experiment_id, str) or experiment_id in seen:
            errors.append(f"experiments[{idx}] id missing/duplicate")
            continue
        seen.add(experiment_id)
        compact_fields = (
            "model", "params_b", "active_b", "moe", "target_bpw", "diagnostic", "base",
            "transform", "correction", "state", "runtime", "objective", "calibration", "seed",
            "campaign_version",
        )
        compact = {field: row.get(field) for field in compact_fields}
        identity = abi.hash_value(compact)
        if row.get("identity_sha256") != identity or experiment_id != f"dr2-{identity[:16]}":
            errors.append(f"{experiment_id}: candidate identity mismatch")
    expected = campaign.get("campaign_sha256")
    payload = {k: v for k, v in campaign.items() if k not in {"campaign_sha256", "generated_at"}}
    if not abi.is_sha256(expected) or expected != abi.hash_value(payload):
        errors.append("campaign_sha256 mismatch")
    return errors


def _node(name: str, node_id: str, deps: list[str]) -> dict[str, Any]:
    row = CATALOG[name]
    dynamic = row["dynamic"]
    support = {
        "apple_cpu": "prototype" if row["implementation_status"] in {"measured", "prototype", "oracle"} else "research",
        "metal": "prototype" if row["implementation_status"] == "measured" else "gated",
        "cuda": "research" if row["cuda"] != "unsupported" else "unsupported",
        "distributed": "research", "future_specialized": "research",
    }
    status = row["implementation_status"]
    if status == "runtime_gated":
        status = "runtime_gated"
    return {
        "id": node_id, "kind": row["kind"], "phase": row["phase"],
        "mechanism": name, "mechanism_version": CAMPAIGN_VERSION,
        "implementation_status": status, "depends_on": deps,
        "parameters": {}, "backend_support": support,
        "cost_contract": {
            "actual_bytes_authoritative": True,
            "dynamic_required_metrics": ["mean", "p95", "worst"] if dynamic else [],
        },
        "executor": {"wired": False, "argv": [], "source_sha256": None},
    }


def materialize(candidate: dict[str, Any]) -> dict[str, Any]:
    chain = []
    deps: list[str] = []
    for idx, name in enumerate((candidate["diagnostic"], candidate["transform"], candidate["base"],
                                candidate["correction"], candidate["state"], candidate["runtime"],
                                "packed_correction_package", "paired_capability_efficiency_eval")):
        node_id = f"n{idx}-{name}"
        chain.append(_node(name, node_id, list(deps)))
        deps = [node_id]
    program = abi.make_planned_program(
        label=candidate["model"], params_b=float(candidate["params_b"]),
        active_b=float(candidate["active_b"]) if candidate.get("active_b") is not None else None,
        physical_bpw=float(candidate["target_bpw"]), operators=chain,
    )
    program["experiment_binding"] = {
        "experiment_id": candidate["experiment_id"],
        "candidate_identity_sha256": candidate["identity_sha256"],
        "objective": candidate["objective"], "calibration": candidate["calibration"],
        "seed": candidate["seed"],
    }
    return abi.stamp_program(program)


def _read_observations(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out = {}
    for line in path.read_text(errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and isinstance(row.get("experiment_id"), str):
            out[row["experiment_id"]] = row
    return out


FIDELITY_IDS = ("F0", "F1", "F2", "F3", "F4", "F5", "F6")
FIDELITY_COST = dict(zip(FIDELITY_IDS, (1, 4, 16, 64, 192, 256, 512)))


def _observation_values(obs: dict[str, Any]) -> tuple[float, float, float, float, float]:
    quality = obs.get("quality") if isinstance(obs.get("quality"), dict) else {}
    vector = quality.get("capability_vector") if isinstance(quality.get("capability_vector"), dict) else {}
    finite_capability = [float(value) for value in vector.values()
                         if isinstance(value, (int, float)) and math.isfinite(float(value))]
    capability = min(finite_capability) if finite_capability else float(obs.get("capability_gain", 0.0) or 0.0)
    costs = obs.get("costs") if isinstance(obs.get("costs"), dict) else obs
    physical = float(costs.get("physical_model_bytes", costs.get("incremental_physical_bytes", math.inf)) or math.inf)
    moved = float(costs.get("p95_bytes_per_token", math.inf) or math.inf)
    joules = float(costs.get("joules_per_accepted_token", math.inf) or math.inf)
    latency = float(costs.get("p95_latency_ms", math.inf) or math.inf)
    return capability, physical, moved, joules, latency


def _pareto_ids(observations: dict[str, dict[str, Any]]) -> set[str]:
    usable = {
        experiment_id: _observation_values(obs)
        for experiment_id, obs in observations.items()
        if obs.get("status") in {"pass", "succeeded"}
    }
    frontier: set[str] = set()
    for experiment_id, values in usable.items():
        dominated = False
        for other_id, other in usable.items():
            if other_id == experiment_id:
                continue
            at_least = other[0] >= values[0] and all(other[i] <= values[i] for i in range(1, 5))
            strict = other[0] > values[0] or any(other[i] < values[i] for i in range(1, 5))
            if at_least and strict:
                dominated = True
                break
        if not dominated:
            frontier.add(experiment_id)
    return frontier


def _next_fidelity(obs: dict[str, Any] | None) -> str | None:
    if obs is None:
        return "F0"
    if obs.get("status") in {"fail", "complete-negative", "complete_negative", "failed_terminal",
                             "blocked", "invalidated", "superseded"}:
        return None
    current = str(obs.get("fidelity", "F0"))
    if current not in FIDELITY_IDS:
        return None
    idx = FIDELITY_IDS.index(current)
    return FIDELITY_IDS[idx + 1] if idx + 1 < len(FIDELITY_IDS) else None


def select_next(campaign: dict[str, Any], observations: dict[str, dict[str, Any]],
                *, allow_unwired: bool = False) -> dict[str, Any]:
    rows = []
    pareto = _pareto_ids(observations)
    coverage: dict[tuple[str, str], int] = {}
    for candidate in campaign.get("experiments", []):
        obs = observations.get(candidate["experiment_id"])
        if not obs:
            continue
        for axis in ("diagnostic", "base", "transform", "correction", "state", "runtime"):
            key = (axis, candidate[axis])
            coverage[key] = coverage.get(key, 0) + 1
    for candidate in campaign.get("experiments", []):
        obs = observations.get(candidate["experiment_id"])
        requested_fidelity = _next_fidelity(obs)
        if requested_fidelity is None:
            continue
        if not allow_unwired and not candidate.get("launchable"):
            continue
        control = candidate["correction"] == "zero_correction"
        rate_bonus = 1.0 / max(float(candidate["target_bpw"]), 0.05)
        quality_doc = obs.get("quality") if obs and isinstance(obs.get("quality"), dict) else {}
        uncertainty = float(quality_doc.get("uncertainty", obs.get("uncertainty", 1.0) if obs else 1.0))
        uncertainty = uncertainty if math.isfinite(uncertainty) and uncertainty >= 0 else 1.0
        feasibility = float(obs.get("feasibility_probability", 0.5)) if obs else 0.5
        transfer = float(obs.get("cross_scale_information_gain", 1.0)) if obs else 1.0
        family_min = min(
            coverage.get((axis, candidate[axis]), 0)
            for axis in ("diagnostic", "base", "transform", "correction", "state", "runtime")
        )
        coverage_bonus = 100.0 / math.sqrt(1.0 + family_min)
        control_bonus = 1000.0 if control and obs is None else 0.0
        pareto_bonus = 250.0 if candidate["experiment_id"] in pareto else 0.0
        fidelity_cost = float(FIDELITY_COST[requested_fidelity])
        voi = uncertainty * max(feasibility, 0.0) * max(transfer, 0.0) / fidelity_cost
        # Stable exploration keeps novel families alive even when proxy metrics
        # initially favor familiar scalar controls.
        exploration = int(candidate["identity_sha256"][-8:], 16) / 0xFFFFFFFF
        score_parts = {
            "control": control_bonus,
            "family_coverage": coverage_bonus,
            "pareto": pareto_bonus,
            "value_of_information": 100.0 * voi,
            "low_rate_boundary": rate_bonus,
            "exploration": exploration,
        }
        score = sum(score_parts.values())
        rows.append((score, candidate["experiment_id"], candidate, requested_fidelity, score_parts))
    rows.sort(key=lambda item: (-item[0], item[1]))
    selected = rows[0][2] if rows else None
    return {
        "schema": SELECTION_SCHEMA,
        "campaign_sha256": campaign.get("campaign_sha256"),
        "selected": selected,
        "requested_fidelity": rows[0][3] if rows else None,
        "score": rows[0][0] if rows else None,
        "score_breakdown": rows[0][4] if rows else None,
        "pareto_observation_ids": sorted(pareto),
        "eligible_count": len(rows),
        "allow_unwired": allow_unwired,
        "note": "selection is advisory; heavy-work admission and a validated executable HealerProgram remain mandatory",
    }


def selftest() -> int:
    tiny = ({"label": "tiny-moe", "params_b": 120.0, "active_b": 5.0, "moe": True,
             "rates": (2.0, 0.5)},)
    campaign = compile_campaign(models=tiny, max_explicit=256)
    assert campaign["schema"] == CAMPAIGN_SCHEMA
    assert campaign["search_size"]["explicit_candidates"] == 256
    assert campaign["search_size"]["projected_cartesian_candidates"] > 100_000
    assert validate_campaign(campaign) == []
    ids = [row["experiment_id"] for row in campaign["experiments"]]
    assert len(ids) == len(set(ids))
    candidate = campaign["experiments"][0]
    program = materialize(candidate)
    assert abi.validate_program(program) == []
    selection = select_next(campaign, {}, allow_unwired=True)
    assert selection["selected"] is not None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "campaign.json"
        abi.atomic_json(path, campaign)
        assert json.loads(path.read_text())["campaign_sha256"] == campaign["campaign_sha256"]
    print("doctor_frontier.py selftest OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    compile_p = sub.add_parser("compile")
    compile_p.add_argument("--max-explicit", type=int, default=4096)
    compile_p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    select_p = sub.add_parser("select")
    select_p.add_argument("--campaign", type=Path, default=DEFAULT_OUTPUT)
    select_p.add_argument("--observations", type=Path)
    select_p.add_argument("--allow-unwired", action="store_true")
    materialize_p = sub.add_parser("materialize")
    materialize_p.add_argument("experiment_id")
    materialize_p.add_argument("--campaign", type=Path, default=DEFAULT_OUTPUT)
    materialize_p.add_argument("--output", type=Path)
    sub.add_parser("selftest")
    args = parser.parse_args()

    if args.command == "selftest":
        return selftest()
    if args.command == "compile":
        campaign = compile_campaign(max_explicit=args.max_explicit)
        abi.atomic_json(args.output, campaign)
        print(json.dumps(campaign["search_size"], indent=2))
        return 0
    campaign = json.loads(args.campaign.read_text())
    problems = validate_campaign(campaign)
    if problems:
        raise SystemExit("invalid campaign: " + "; ".join(problems[:20]))
    if args.command == "select":
        print(json.dumps(select_next(campaign, _read_observations(args.observations),
                                     allow_unwired=args.allow_unwired), indent=2))
        return 0
    candidate = next((row for row in campaign.get("experiments", [])
                      if row.get("experiment_id") == args.experiment_id), None)
    if candidate is None:
        raise SystemExit(f"unknown experiment_id: {args.experiment_id}")
    program = materialize(candidate)
    out = args.output or Path(f"reports/condense/doctor_v2_programs/{args.experiment_id}.json")
    abi.atomic_json(out, program)
    print(f"wrote planned HealerProgram: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
