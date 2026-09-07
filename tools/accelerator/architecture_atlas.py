"""Architecture-behavior atlas and experiment funnel for Hawking Accelerator.

The atlas is deliberately about physical behaviors, not product ports.  A
source school is a teacher; its behavior becomes a Hawking primitive only
after it has a target, a control, metrics, and a falsifier.  This module keeps
that funnel executable without claiming that a source architecture's result
will transfer to Metal.

The canonical JSON artifact is generated from :func:`build_atlas` and can be
validated without a GPU.  Protected measurements are intentionally represented
as queued work: a structural or simulated record can never promote a primitive
into a cross-model compiler law.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
SCHEMA = "hawking.accelerator.architecture_atlas.v1"
QUEUE_SCHEMA = "hawking.accelerator.repatriation_queue.v1"
ASIC_SCHEMA = "hawking.accelerator.asic_candidate_ledger.v1"
PLANNING_BENCH = {
    "state": "UNKNOWN",
    "recorded_at": "2026-08-29T00:00:00Z",
    "recorded_by": "architecture-atlas planning artifact",
    "machine": "planning artifact; no physical benchmark was executed",
    "quiescence": None,
    "rule": "S032 §3 -- if quiescence is unknown the state is UNKNOWN, not quiet",
    "provenance": "This artifact contains hypotheses and schema metadata, not a benchmark result",
}

STATUSES = (
    "DISCOVERED",
    "MAPPED",
    "IMPLEMENTED",
    "DIAGNOSTIC",
    "PHYSICALLY_MEASURED",
    "REJECTED",
    "BLOCKED",
)
EVIDENCE_CLASSES = (
    "DOCUMENTED",
    "OPEN-SOURCE-INSPECTED",
    "DERIVED",
    "SIMULATED",
    "HAWKING_IMPLEMENTED",
    "HAWKING_MEASURED",
    "HAWKING_PROTECTED_VERIFIED",
)
TRANSFER_CONFIDENCE = ("HIGH", "MEDIUM", "LOW")
EFFECT_METRICS = (
    "active_bytes",
    "flops",
    "dispatches",
    "synchronization",
    "host_ceremony",
    "state_movement",
    "token_ns",
    "experiment_turnaround_ns",
)
BEHAVIOR_TAXONOMY = (
    "DATA_STATIONARITY",
    "STREAMING",
    "SPATIAL_EXECUTION",
    "PERSISTENT_EXECUTION",
    "STATIC_SCHEDULING",
    "DYNAMIC_SCHEDULING",
    "LOCAL_MEMORY",
    "GLOBAL_MEMORY",
    "ASYNC_PREFETCH",
    "DOUBLE_BUFFERING",
    "TILING",
    "LAYOUT_TRANSFORMATION",
    "LOW_BIT_ARITHMETIC",
    "SPARSITY_SKIPPING",
    "CONDITIONAL_EXECUTION",
    "ROUTE_AWARE_EXECUTION",
    "COMPUTE_IN_TRANSIT",
    "FUSED_REPRESENTATION_DECODE",
    "STATE_RESIDENCY",
    "CROSS_DEVICE_OVERLAP",
    "GRAPH_REPLAY",
)

# These names are owned by Hawking and are safe to carry across backends.
PRIMITIVES = (
    "PersistentPhysicalRegion",
    "StationaryRepresentation",
    "AsyncPrefetch",
    "DoubleBufferedTile",
    "SpatialPipeline",
    "FusedDecodeCompute",
    "DirectRoutedAccumulate",
    "LocalStateMachine",
    "SemanticTransportEdge",
    "TiledProjection",
    "LayoutTransform",
    "SparseSkip",
    "ConditionalPhysicalProgram",
    "GraphReplay",
    "CollectiveRegion",
    "MoveOrRecompute",
    "MemoryTierIdentity",
)

SOURCE_SCHOOLS = (
    "NVIDIA CUDA",
    "CUTLASS/CUTE",
    "AMD ROCm",
    "TPU/systolic",
    "deterministic dataflow",
    "wafer-scale locality",
    "tile-local dataflow",
    "packetized NoC",
    "reconfigurable dataflow",
    "Apple/other NPU",
    "fabric-first accelerator",
    "CPU matrix engine",
    "FPGA",
    "published inference ASIC",
)


def _source_technique_coverage() -> list[dict[str, str]]:
    """Keep the bounded first sweep auditable without making a prose survey."""

    rows = (
        ("persistent kernels", "NVIDIA CUDA", "persistent_physical_region"),
        ("graph capture / graph replay", "NVIDIA CUDA", "graph_replay"),
        ("asynchronous memory staging", "NVIDIA CUDA", "async_double_buffer"),
        ("double buffering", "NVIDIA CUDA", "async_double_buffer"),
        ("warp/subgroup specialization", "NVIDIA CUDA", "layout_algebra"),
        ("tiled GEMV/GEMM", "NVIDIA CUDA", "layout_algebra"),
        ("fused dequant + compute", "NVIDIA CUDA", "fused_decode_compute"),
        ("low-bit layouts", "NVIDIA CUDA", "layout_algebra"),
        ("FlashAttention-style IO-aware attention", "NVIDIA CUDA", "spatial_local_pipeline"),
        ("paged KV/state", "NVIDIA CUDA", "local_state_machine"),
        ("MoE expert sorting/grouping", "NVIDIA CUDA", "direct_routed_accumulate"),
        ("expert batching", "NVIDIA CUDA", "direct_routed_accumulate"),
        ("fused routing/gating", "NVIDIA CUDA", "direct_routed_accumulate"),
        ("direct expert accumulation", "NVIDIA CUDA", "direct_routed_accumulate"),
        ("quantized expert execution", "NVIDIA CUDA", "fused_decode_compute"),
        ("prefix/cache reuse", "NVIDIA CUDA", "stationary_representation"),
        ("speculative decoding", "NVIDIA CUDA", "static_dynamic_skeleton"),
        ("continuous batching", "NVIDIA CUDA", "static_dynamic_skeleton"),
        ("stream overlap", "NVIDIA CUDA", "async_double_buffer"),
        ("kernel autotuning", "NVIDIA CUDA", "layout_algebra"),
        ("memory swizzling", "NVIDIA CUDA", "layout_algebra"),
        ("register blocking", "NVIDIA CUDA", "layout_algebra"),
        ("shared-memory staging", "NVIDIA CUDA", "async_double_buffer"),
        ("graph-level scheduling", "NVIDIA CUDA", "static_dynamic_skeleton"),
        ("logical/physical/tile/lane layout algebra", "CUTLASS/CUTE", "layout_algebra"),
        ("wave-oriented execution", "AMD ROCm", "spatial_local_pipeline"),
        ("occupancy-vs-register tradeoff", "AMD ROCm", "layout_algebra"),
        ("large-HBM scheduling", "AMD ROCm", "move_or_recompute"),
        ("multi-GPU data movement", "AMD ROCm", "collective_region"),
        ("systolic dataflow", "TPU/systolic", "stationary_representation"),
        ("compile-time operation placement", "TPU/systolic", "static_dynamic_skeleton"),
        ("weight/activation/output stationarity", "TPU/systolic", "stationary_representation"),
        ("explicit collective topology", "TPU/systolic", "collective_region"),
        ("deterministic static schedule", "deterministic dataflow", "static_dynamic_skeleton"),
        ("spatial placement and local ownership", "wafer-scale locality", "spatial_local_pipeline"),
        ("tile-local compute-near-data", "tile-local dataflow", "stationary_representation"),
        ("packetized data movement", "packetized NoC", "semantic_transport"),
        ("runtime parameters without full rebuild", "reconfigurable dataflow", "graph_replay"),
        ("low launch overhead and regular graph islands", "Apple/other NPU", "npu_regular_island"),
        ("communication-aware graph placement", "fabric-first accelerator", "collective_region"),
        ("cache blocking and small-matrix execution", "CPU matrix engine", "layout_algebra"),
        ("NUMA-aware placement and prefetch", "CPU matrix engine", "move_or_recompute"),
        ("spatial pipelines and dataflow execution", "FPGA", "spatial_local_pipeline"),
        ("bit-serial and arbitrary-precision arithmetic", "FPGA", "fused_decode_compute"),
        ("BRAM/URAM/HBM residency", "FPGA", "stationary_representation"),
        ("route-specific hardware and custom transport", "FPGA", "direct_routed_accumulate"),
        ("compression-aware arithmetic and operand reuse", "published inference ASIC", "fused_decode_compute"),
        ("on-chip SRAM hierarchy and deterministic pipelines", "published inference ASIC", "persistent_physical_region"),
    )
    return [
        {
            "source_technique": technique,
            "source_school": school,
            "behavior_id": behavior_id,
            "claim_boundary": "source behavior is a mapped hypothesis, not a Hawking speed claim",
        }
        for technique, school, behavior_id in rows
    ]


class AtlasValidationError(ValueError):
    """The atlas is incomplete, overclaims, or cannot produce an experiment."""


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _effect(direction: str, mechanism: str, *, measurement: str = "measure") -> dict[str, str]:
    return {"direction": direction, "mechanism": mechanism, "measurement": measurement}


def _effects(mechanism: str, *, reduce: Iterable[str] = (), increase: Iterable[str] = ()) -> dict[str, dict[str, str]]:
    lower = set(reduce)
    upper = set(increase)
    unknown = set(EFFECT_METRICS) - lower - upper
    if lower & upper or unknown:
        raise AtlasValidationError(
            f"effect directions must partition metrics; reduce={lower}, increase={upper}, unknown={unknown}"
        )
    return {
        metric: _effect(
            "REDUCE" if metric in lower else "INCREASE",
            mechanism,
        )
        for metric in EFFECT_METRICS
    }


def _entry(
    behavior_id: str,
    *,
    source_architecture_ecosystem: Sequence[str],
    source_behavior: str,
    physical_idea: str,
    vendor_assumptions: str,
    invariant: str,
    primitive: str,
    physical_graph_mapping: Mapping[str, Any],
    models: Sequence[str],
    organs: Sequence[str],
    backends: Sequence[str],
    effects: Mapping[str, Mapping[str, str]],
    difficulty: int,
    confidence: str,
    falsifier: str,
    status: str,
    evidence_class: str,
    evidence: Sequence[str],
    value: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "behavior_id": behavior_id,
        "behavior_taxonomy": [],
        "source_architecture_ecosystem": list(source_architecture_ecosystem),
        "source_behavior": source_behavior,
        "fundamental_physical_idea": physical_idea,
        "vendor_specific_assumptions": vendor_assumptions,
        "architecture_independent_invariant": invariant,
        "hawking_primitive": primitive,
        "physical_graph_mapping": dict(physical_graph_mapping),
        "applicable_models": list(models),
        "applicable_organs": list(organs),
        "applicable_backends": list(backends),
        "expected_effect": dict(effects),
        "implementation_difficulty": difficulty,
        "transfer_confidence": confidence,
        "cheapest_falsifier": falsifier,
        "status": status,
        "evidence_class": evidence_class,
        "source_evidence": list(evidence),
        "ev_inputs": dict(value),
    }


def _raw_entries() -> list[dict[str, Any]]:
    """Return the bounded first sweep, collapsed by physical behavior."""
    return [
        _entry(
            "persistent_physical_region",
            source_architecture_ecosystem=("NVIDIA CUDA", "deterministic dataflow", "FPGA", "Apple/other NPU"),
            source_behavior="persistent kernels, compiled graph execution, and resident pipelines amortize repeated entry",
            physical_idea="keep executable state and reusable bindings alive across token steps",
            vendor_assumptions="cooperative CUDA launch, NPU graph lifetime, or FPGA fabric are not assumed by Hawking",
            invariant="when state and bindings remain valid, repeated host entry is not semantically required",
            primitive="PersistentPhysicalRegion",
            physical_graph_mapping={"execution_policy.process": "long_lived_executor", "residency.weights": "resident", "residency.state": "sequence"},
            models=("Qwen27", "Flash"), organs=("decode", "deltanet", "moe", "attention", "sampling"),
            backends=("metal", "ane", "fpga", "cuda"),
            effects=_effects("amortize launch and binding ceremony", reduce=("host_ceremony", "synchronization", "state_movement", "token_ns", "experiment_turnaround_ns"), increase=("active_bytes", "flops", "dispatches")),
            difficulty=2, confidence="HIGH",
            falsifier="protected complete useful wall time does not improve against the same resident control, with identical output and zero fallback",
            status="IMPLEMENTED", evidence_class="HAWKING_IMPLEMENTED",
            evidence=("hcli/physical_graph.py", "hcli/hawking_native.py", "receipts/headless/ACCELERATOR_DEVICE_RESIDENT.json"),
            value={"token_ns_reduction": 5, "transfer_breadth": 5, "model_applicability": 5, "future_hardware_value": 5, "information_gain": 3},
        ),
        _entry(
            "graph_replay",
            source_architecture_ecosystem=("NVIDIA CUDA", "TPU/systolic", "Apple/other NPU", "FPGA"),
            source_behavior="capture a stable graph and replay it with small parameter/control updates",
            physical_idea="compile the static skeleton once and expose only dynamic slots",
            vendor_assumptions="CUDA graph APIs and NPU executable formats are vendor-specific",
            invariant="a stable dependency graph can be reused if dynamic values do not alter topology",
            primitive="GraphReplay",
            physical_graph_mapping={"execution_policy.pipeline_state": "compile_once_reuse", "execution_policy.dynamic_slots": ["token", "position", "route", "sampling"]},
            models=("Qwen27", "Flash"), organs=("decode", "regular_mlp", "attention", "moe"),
            backends=("metal", "ane", "fpga", "cuda", "cpu"),
            effects=_effects("reuse command topology and reduce rebuilds", reduce=("host_ceremony", "synchronization", "experiment_turnaround_ns", "token_ns"), increase=("active_bytes", "flops", "dispatches", "state_movement")),
            difficulty=4, confidence="HIGH",
            falsifier="replay adds no protected complete-wall improvement or changes a dynamic route/state result",
            status="DIAGNOSTIC", evidence_class="DERIVED",
            evidence=("receipts/headless/ACCELERATOR_GRAPH_SUBMISSION.json", "hcli/physical_graph.py"),
            value={"token_ns_reduction": 5, "transfer_breadth": 5, "model_applicability": 4, "future_hardware_value": 5, "information_gain": 5},
        ),
        _entry(
            "async_double_buffer",
            source_architecture_ecosystem=("NVIDIA CUDA", "AMD ROCm", "FPGA", "reconfigurable dataflow"),
            source_behavior="stage the next tile asynchronously while the current tile computes",
            physical_idea="overlap independent movement and compute with two ownership-safe buffers",
            vendor_assumptions="warp/wave async copies and hardware DMA differ across devices",
            invariant="a transfer can be hidden only inside a real overlap window with explicit producer/consumer ownership",
            primitive="DoubleBufferedTile",
            physical_graph_mapping={"memory": "double_buffered_tiles", "synchronization": "producer_consumer_fences", "execution_policy": "overlap_when_measured"},
            models=("Qwen27", "Flash"), organs=("mlp", "moe", "deltanet", "kv"),
            backends=("metal", "fpga", "cuda", "cpu"),
            effects=_effects("hide movement behind independent work", reduce=("state_movement", "synchronization", "token_ns"), increase=("active_bytes", "flops", "dispatches", "host_ceremony", "experiment_turnaround_ns")),
            difficulty=4, confidence="MEDIUM",
            falsifier="measured overlap window is zero or protected complete wall is not lower after fence costs",
            status="MAPPED", evidence_class="DERIVED",
            evidence=("tools/accelerator/fusion_planner.py", "hcli/agentos/fpga_preboard.py"),
            value={"token_ns_reduction": 4, "transfer_breadth": 4, "model_applicability": 4, "future_hardware_value": 5, "information_gain": 4},
        ),
        _entry(
            "layout_algebra",
            source_architecture_ecosystem=("CUTLASS/CUTE", "AMD ROCm", "TPU/systolic", "FPGA"),
            source_behavior="separate logical tensors from physical packing, tile mapping, lane mapping, and arithmetic mapping",
            physical_idea="choose layout and thread/tile ownership as compiler objects rather than kernel folklore",
            vendor_assumptions="warp, wave, systolic-array, and BRAM banking dimensions are not portable constants",
            invariant="the same logical operation may have materially different movement and reduction costs under different legal layouts",
            primitive="LayoutTransform",
            physical_graph_mapping={"representation": "layout_algebra", "computation": "tile_and_lane_mapping", "precision": "representation_grouping"},
            models=("Qwen27", "Flash"), organs=("mlp", "moe", "attention", "deltanet"),
            backends=("metal", "fpga", "cuda", "cpu", "ane"),
            effects=_effects("reduce strided movement and tails through shape-aware mapping", reduce=("active_bytes", "synchronization", "token_ns", "experiment_turnaround_ns"), increase=("flops", "dispatches", "host_ceremony", "state_movement")),
            difficulty=5, confidence="HIGH",
            falsifier="same-source protected A/B at the chosen organ shows no complete-wall or active-byte benefit",
            status="DIAGNOSTIC", evidence_class="OPEN-SOURCE-INSPECTED",
            evidence=("tools/accelerator/air.py", "tools/accelerator/kernel_forge.py", "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"),
            value={"token_ns_reduction": 5, "transfer_breadth": 5, "model_applicability": 5, "future_hardware_value": 5, "information_gain": 5},
        ),
        _entry(
            "stationary_representation",
            source_architecture_ecosystem=("TPU/systolic", "wafer-scale locality", "tile-local dataflow", "published inference ASIC", "NVIDIA CUDA"),
            source_behavior="keep weights, activations, outputs, or compressed operands stationary near reuse",
            physical_idea="spend residency and local storage to avoid repeatedly moving the dominant operand",
            vendor_assumptions="systolic arrays, SRAM sizes, and wafer-scale placement are not assumed",
            invariant="the best stationary operand is the one whose reuse exceeds the cost of retaining it",
            primitive="StationaryRepresentation",
            physical_graph_mapping={"residency": "stationarity_contract", "memory": "tier_is_executable_identity", "representation": "packed_native"},
            models=("Qwen27", "Flash"), organs=("mlp", "moe", "embedding", "lm_head", "codebook"),
            backends=("metal", "ane", "fpga", "cuda", "cpu"),
            effects=_effects("avoid repeated operand movement", reduce=("active_bytes", "state_movement", "host_ceremony", "token_ns"), increase=("flops", "dispatches", "synchronization", "experiment_turnaround_ns")),
            difficulty=3, confidence="HIGH",
            falsifier="packed resident A/B has no protected complete-wall benefit or violates the no-dense-rematerialization/capability gate",
            status="IMPLEMENTED", evidence_class="HAWKING_IMPLEMENTED",
            evidence=("crates/hawking-core/src/model/qwen38_hybrid_decode.rs", "receipts/headless/ACCELERATOR_DEVICE_RESIDENT.json", "receipts/headless/ACCELERATOR_TOKEN_BYTE_ATLAS_628.json"),
            value={"token_ns_reduction": 5, "transfer_breadth": 5, "model_applicability": 5, "future_hardware_value": 5, "information_gain": 3},
        ),
        _entry(
            "static_dynamic_skeleton",
            source_architecture_ecosystem=("deterministic dataflow", "TPU/systolic", "FPGA", "NVIDIA CUDA"),
            source_behavior="compile a mostly static schedule and leave only bounded route/position/state slots dynamic",
            physical_idea="remove runtime scheduling decisions from the critical path without pretending dynamic MoE is static",
            vendor_assumptions="fully static placement is not valid for arbitrary routes or variable-length requests",
            invariant="static structure and dynamic control can coexist when dynamic choices do not change buffer ownership unsafely",
            primitive="ConditionalPhysicalProgram",
            physical_graph_mapping={"execution_policy": "static_skeleton_plus_dynamic_slots", "dependencies": "precomputed", "synchronization": "precomputed_where_safe"},
            models=("Qwen27", "Flash"), organs=("decode", "deltanet", "moe", "kv", "sampling"),
            backends=("metal", "ane", "fpga", "cuda", "cpu"),
            effects=_effects("precompute safe scheduling and retain dynamic controls", reduce=("host_ceremony", "synchronization", "token_ns", "experiment_turnaround_ns"), increase=("active_bytes", "flops", "dispatches", "state_movement")),
            difficulty=4, confidence="HIGH",
            falsifier="a bounded dynamic slot requires topology rebuild or changes output/capability under a protected replay",
            status="IMPLEMENTED", evidence_class="HAWKING_IMPLEMENTED",
            evidence=("hcli/physical_graph.py", "crates/hawking-core/src/model/qwen38_hybrid_decode.rs", "hcli/agentos/fpga_preboard.py"),
            value={"token_ns_reduction": 4, "transfer_breadth": 5, "model_applicability": 5, "future_hardware_value": 5, "information_gain": 5},
        ),
        _entry(
            "spatial_local_pipeline",
            source_architecture_ecosystem=("wafer-scale locality", "tile-local dataflow", "reconfigurable dataflow", "FPGA"),
            source_behavior="place a producer-consumer chain spatially so intermediates do not round-trip through global memory",
            physical_idea="make locality and pipeline ownership explicit in the graph",
            vendor_assumptions="wafer-scale links, tile SRAM, and FPGA BRAM/URAM are implementation-specific",
            invariant="if an intermediate is not externally observable, its global materialization is optional",
            primitive="SpatialPipeline",
            physical_graph_mapping={"computation": "spatial_regions", "data": "semantic_edges", "memory": "local_intermediates"},
            models=("Qwen27", "Flash"), organs=("mlp", "deltanet", "moe", "attention"),
            backends=("metal", "fpga", "cuda"),
            effects=_effects("fuse producer-consumer regions and keep intermediates local", reduce=("active_bytes", "state_movement", "synchronization", "host_ceremony", "token_ns"), increase=("flops", "dispatches", "experiment_turnaround_ns")),
            difficulty=5, confidence="MEDIUM",
            falsifier="fused region fails numerical parity or protected complete wall increases after local-storage/fence cost",
            status="MAPPED", evidence_class="DERIVED",
            evidence=("receipts/headless/PHYSICAL_GRAPH_COMPILER.json", "hcli/agentos/fpga_preboard.py"),
            value={"token_ns_reduction": 5, "transfer_breadth": 4, "model_applicability": 4, "future_hardware_value": 5, "information_gain": 4},
        ),
        _entry(
            "semantic_transport",
            source_architecture_ecosystem=("packetized NoC", "fabric-first accelerator", "FPGA"),
            source_behavior="move typed activations, state, route metadata, or partial reductions through explicit links",
            physical_idea="make data movement a semantic graph edge instead of an opaque memory copy",
            vendor_assumptions="NoC packet formats and collective fabrics differ; the edge contract is Hawking-owned",
            invariant="a transfer is optimizable only when its payload, owner, ordering, and reduction semantics are explicit",
            primitive="SemanticTransportEdge",
            physical_graph_mapping={"dependencies": "typed_transport_edges", "synchronization": "edge_ownership_and_order", "device_placement": "topology_aware"},
            models=("Qwen27", "Flash"), organs=("moe", "attention", "deltanet", "fpga_partition"),
            backends=("metal", "fpga", "cuda", "cpu"),
            effects=_effects("avoid untyped copies and choose topology-aware movement", reduce=("active_bytes", "state_movement", "synchronization", "token_ns", "experiment_turnaround_ns"), increase=("flops", "dispatches", "host_ceremony")),
            difficulty=4, confidence="MEDIUM",
            falsifier="semantic edge accounting cannot reproduce the reference output or link cost erases the proposed benefit",
            status="IMPLEMENTED", evidence_class="HAWKING_IMPLEMENTED",
            evidence=("tools/accelerator/fusion_planner.py", "hcli/agentos/fpga_preboard.py", "receipts/headless/QWEN27_FPGA_ORGAN_MAP.json"),
            value={"token_ns_reduction": 3, "transfer_breadth": 5, "model_applicability": 5, "future_hardware_value": 5, "information_gain": 5},
        ),
        _entry(
            "direct_routed_accumulate",
            source_architecture_ecosystem=("NVIDIA CUDA", "fabric-first accelerator", "packetized NoC"),
            source_behavior="sort/group routes, execute selected experts, and accumulate weighted outputs without avoidable staging",
            physical_idea="treat routing and accumulation as one physical region around selected payloads",
            vendor_assumptions="large-batch grouped GEMM and datacenter interconnect assumptions do not automatically fit local decode",
            invariant="for few active tokens, route metadata and selected payload locality matter more than nominal expert throughput",
            primitive="DirectRoutedAccumulate",
            physical_graph_mapping={"computation": "route_then_native_expert", "data": "selected_payload_only", "state": "route_metadata_resident"},
            models=("Flash",), organs=("moe", "router", "shared_expert", "expert_cache"),
            backends=("metal", "fpga", "cuda", "cpu"),
            effects=_effects("skip inactive experts and eliminate staging copies", reduce=("active_bytes", "flops", "state_movement", "token_ns"), increase=("dispatches", "synchronization", "host_ceremony", "experiment_turnaround_ns")),
            difficulty=5, confidence="MEDIUM",
            falsifier="selected-route protected complete wall does not beat current Flash control after route/gather/accumulate costs",
            status="DIAGNOSTIC", evidence_class="HAWKING_MEASURED",
            evidence=("receipts/headless/FLASH_NOETIC_ROUTED_EXPERT_COMPONENT_CAMPAIGN.json", "hcli/agentos/flash_executable.py"),
            value={"token_ns_reduction": 5, "transfer_breadth": 3, "model_applicability": 4, "future_hardware_value": 5, "information_gain": 5},
        ),
        _entry(
            "fused_decode_compute",
            source_architecture_ecosystem=("NVIDIA CUDA", "FPGA", "published inference ASIC", "Apple/other NPU"),
            source_behavior="decode compressed operands inside the projection/epilogue instead of materializing a dense intermediate",
            physical_idea="fuse representation decode with arithmetic at the consumer",
            vendor_assumptions="tensor-core instruction shapes and fixed-function units are not portable",
            invariant="a representation-native consumer may remove an intermediate only when its numerical contract is preserved",
            primitive="FusedDecodeCompute",
            physical_graph_mapping={"representation": "native_decode", "computation": "projection_plus_decode", "memory": "no_dense_rematerialization"},
            models=("Qwen27", "Flash"), organs=("mlp", "moe", "lm_head", "codebook"),
            backends=("metal", "fpga", "cuda", "ane"),
            effects=_effects("remove dense rematerialization and its write/read", reduce=("active_bytes", "flops", "state_movement", "token_ns", "experiment_turnaround_ns"), increase=("dispatches", "synchronization", "host_ceremony")),
            difficulty=3, confidence="HIGH",
            falsifier="parity, fallback, or protected complete-wall gate fails against the same-source dense/control path",
            status="PHYSICALLY_MEASURED", evidence_class="HAWKING_MEASURED",
            evidence=("receipts/headless/ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json", "receipts/headless/AFFINE2_NATIVE_MLP.json", "crates/hawking-core/src/model/qwen38_hybrid_decode.rs"),
            value={"token_ns_reduction": 5, "transfer_breadth": 5, "model_applicability": 5, "future_hardware_value": 5, "information_gain": 4},
        ),
        _entry(
            "local_state_machine",
            source_architecture_ecosystem=("FPGA", "deterministic dataflow", "Apple/other NPU", "published inference ASIC"),
            source_behavior="keep recurrent state alive and update it in a persistent local pipeline",
            physical_idea="model decode as a state machine rather than reconstructing stateless operators every step",
            vendor_assumptions="FPGA registers/BRAM and NPU state APIs do not define Metal semantics",
            invariant="mutable state has one authoritative owner and its update ordering is part of the executable identity",
            primitive="LocalStateMachine",
            physical_graph_mapping={"state": "authoritative_resident_owner", "execution_policy": "fixed_state_transitions", "synchronization": "state_update_edges"},
            models=("Qwen27", "Flash"), organs=("deltanet", "kv", "routing", "ngram", "mtp"),
            backends=("metal", "ane", "fpga", "cuda", "cpu"),
            effects=_effects("keep mutable sequence state resident and ordered", reduce=("state_movement", "host_ceremony", "synchronization", "token_ns"), increase=("active_bytes", "flops", "dispatches", "experiment_turnaround_ns")),
            difficulty=3, confidence="HIGH",
            falsifier="device-resident state does not improve protected complete wall or fails checkpoint/bisection parity",
            status="IMPLEMENTED", evidence_class="HAWKING_IMPLEMENTED",
            evidence=("receipts/headless/FLASH_STATEFUL_CROSS_SPECIES_SEAM.json", "hcli/physical_graph.py", "hcli/agentos/fpga_preboard.py"),
            value={"token_ns_reduction": 4, "transfer_breadth": 5, "model_applicability": 5, "future_hardware_value": 5, "information_gain": 5},
        ),
        _entry(
            "sparse_conditional_execution",
            source_architecture_ecosystem=("NVIDIA CUDA", "AMD ROCm", "published inference ASIC", "packetized NoC"),
            source_behavior="skip zero, inactive, or unselected work while preserving ordering and numerical semantics",
            physical_idea="make conditional work explicit in the physical program and pay only for selected operands",
            vendor_assumptions="structured sparsity instructions and sparse hardware support are not assumed",
            invariant="skipping is legal only when the omitted contribution is proven zero or outside the selected computation",
            primitive="SparseSkip",
            physical_graph_mapping={"computation": "conditional_regions", "data": "sparse_indices_and_payloads", "qualification": "parity_required"},
            models=("Qwen27", "Flash"), organs=("sparse_attention", "moe", "residual", "router"),
            backends=("metal", "fpga", "cuda", "cpu"),
            effects=_effects("avoid provably inactive work", reduce=("active_bytes", "flops", "token_ns"), increase=("dispatches", "synchronization", "host_ceremony", "state_movement", "experiment_turnaround_ns")),
            difficulty=4, confidence="MEDIUM",
            falsifier="sparse control does not preserve output or its index/branch overhead exceeds the omitted work in protected complete wall",
            status="DIAGNOSTIC", evidence_class="HAWKING_MEASURED",
            evidence=("receipts/headless/ACCELERATOR_SPARSE.json", "tools/accelerator/air.py"),
            value={"token_ns_reduction": 3, "transfer_breadth": 4, "model_applicability": 4, "future_hardware_value": 4, "information_gain": 4},
        ),
        _entry(
            "move_or_recompute",
            source_architecture_ecosystem=("CPU matrix engine", "tile-local dataflow", "fabric-first accelerator", "FPGA"),
            source_behavior="choose movement, recomputation, replication, prefetch, or local execution from measured costs",
            physical_idea="do not move an operand merely because the source framework would",
            vendor_assumptions="NUMA/cache and accelerator links are different cost surfaces",
            invariant="the cheapest legal way to satisfy a dependency is a physical scheduling decision",
            primitive="MoveOrRecompute",
            physical_graph_mapping={"dependencies": "costed_dependency_queries", "device_placement": "topology_aware", "execution_policy": "measured_complete_wall"},
            models=("Qwen27", "Flash"), organs=("all",),
            backends=("metal", "ane", "fpga", "cuda", "cpu", "remote"),
            effects=_effects("select the lowest complete dependency cost", reduce=("active_bytes", "state_movement", "synchronization", "host_ceremony", "token_ns", "experiment_turnaround_ns"), increase=("flops", "dispatches")),
            difficulty=2, confidence="HIGH",
            falsifier="the costed plan disagrees with a protected end-to-end A/B or chooses a path with missing capability evidence",
            status="IMPLEMENTED", evidence_class="HAWKING_IMPLEMENTED",
            evidence=("tools/accelerator/fusion_planner.py", "hcli/physical_graph.py"),
            value={"token_ns_reduction": 4, "transfer_breadth": 5, "model_applicability": 5, "future_hardware_value": 4, "information_gain": 5},
        ),
        _entry(
            "npu_regular_island",
            source_architecture_ecosystem=("Apple/other NPU", "TPU/systolic", "published inference ASIC"),
            source_behavior="give regular, compiler-friendly islands to a fixed/semi-fixed neural engine",
            physical_idea="choose a backend per organ based on complete latency and boundary cost, not device prestige",
            vendor_assumptions="public Core ML/ML Program/MLComputePlan is the only Apple authority here",
            invariant="a backend is useful only when its transfer, compile, synchronization, and residency costs fit the graph",
            primitive="ConditionalPhysicalProgram",
            physical_graph_mapping={"device_placement": "organ_level_choice", "dependencies": "explicit_transfer_edges", "qualification": "public_api_and_measurement"},
            models=("Qwen27", "Flash"), organs=("normalization", "silu", "regular_mlp", "sampling"),
            backends=("ane", "metal", "cpu"),
            effects=_effects("offload regular islands only when the boundary wins", reduce=("token_ns", "host_ceremony", "experiment_turnaround_ns"), increase=("active_bytes", "flops", "dispatches", "synchronization", "state_movement")),
            difficulty=4, confidence="MEDIUM",
            falsifier="public ANE plan or protected complete-wall measurement fails after all transfer/compile/residency costs",
            status="MAPPED", evidence_class="DERIVED",
            evidence=("receipts/headless/APPLE_ANE_ATLAS.json", "hcli/ane_provider.py"),
            value={"token_ns_reduction": 3, "transfer_breadth": 3, "model_applicability": 3, "future_hardware_value": 3, "information_gain": 5},
        ),
        _entry(
            "collective_region",
            source_architecture_ecosystem=("fabric-first accelerator", "packetized NoC", "AMD ROCm", "NVIDIA CUDA"),
            source_behavior="schedule collectives and communication topology as part of the executable graph",
            physical_idea="choose ring/tree/direct movement from measured alpha/beta and message size",
            vendor_assumptions="NCCL/RCCL and proprietary fabrics are not portable implementations",
            invariant="synchronized work is paced by its slowest required link and cannot hide an unmodeled transfer",
            primitive="CollectiveRegion",
            physical_graph_mapping={"device_placement": "topology_aware", "synchronization": "collective_algorithm", "dependencies": "semantic_transport"},
            models=("Qwen27", "Flash"), organs=("moe", "attention", "multi_device"),
            backends=("cuda", "fpga", "cpu", "metal"),
            effects=_effects("choose topology-aware collective execution", reduce=("state_movement", "synchronization", "token_ns"), increase=("active_bytes", "flops", "dispatches", "host_ceremony", "experiment_turnaround_ns")),
            difficulty=3, confidence="HIGH",
            falsifier="a protected multi-domain or simulated-link A/B does not match the cost ordering after complete transfer accounting",
            status="IMPLEMENTED", evidence_class="HAWKING_IMPLEMENTED",
            evidence=("tools/accelerator/fusion_planner.py", "hcli/agentos/fpga_preboard.py"),
            value={"token_ns_reduction": 2, "transfer_breadth": 5, "model_applicability": 4, "future_hardware_value": 5, "information_gain": 5},
        ),
    ]


def _taxonomy_for(behavior_id: str) -> list[str]:
    values = {
        "persistent_physical_region": ["PERSISTENT_EXECUTION", "STATE_RESIDENCY", "LOCAL_MEMORY"],
        "graph_replay": ["GRAPH_REPLAY", "STATIC_SCHEDULING", "CONDITIONAL_EXECUTION"],
        "async_double_buffer": ["ASYNC_PREFETCH", "DOUBLE_BUFFERING", "STREAMING"],
        "layout_algebra": ["LAYOUT_TRANSFORMATION", "TILING", "LOW_BIT_ARITHMETIC"],
        "stationary_representation": ["DATA_STATIONARITY", "LOCAL_MEMORY", "LOW_BIT_ARITHMETIC"],
        "static_dynamic_skeleton": ["STATIC_SCHEDULING", "DYNAMIC_SCHEDULING", "CONDITIONAL_EXECUTION"],
        "spatial_local_pipeline": ["SPATIAL_EXECUTION", "STREAMING", "COMPUTE_IN_TRANSIT"],
        "semantic_transport": ["GLOBAL_MEMORY", "CROSS_DEVICE_OVERLAP", "COMPUTE_IN_TRANSIT"],
        "direct_routed_accumulate": ["ROUTE_AWARE_EXECUTION", "DYNAMIC_SCHEDULING", "SPARSITY_SKIPPING"],
        "fused_decode_compute": ["FUSED_REPRESENTATION_DECODE", "LOW_BIT_ARITHMETIC", "COMPUTE_IN_TRANSIT"],
        "local_state_machine": ["STATE_RESIDENCY", "PERSISTENT_EXECUTION", "LOCAL_MEMORY"],
        "sparse_conditional_execution": ["SPARSITY_SKIPPING", "CONDITIONAL_EXECUTION", "ROUTE_AWARE_EXECUTION"],
        "move_or_recompute": ["LOCAL_MEMORY", "GLOBAL_MEMORY", "CROSS_DEVICE_OVERLAP"],
        "npu_regular_island": ["CONDITIONAL_EXECUTION", "GRAPH_REPLAY", "DATA_STATIONARITY"],
        "collective_region": ["CROSS_DEVICE_OVERLAP", "GLOBAL_MEMORY", "STATIC_SCHEDULING"],
    }
    return values[behavior_id]


def _experiment(
    experiment_id: str,
    *,
    behavior_id: str,
    model: str,
    backend: str,
    organ: str,
    candidate: str,
    control: str,
    status: str = "READY",
    blocked_reason: str | None = None,
) -> dict[str, Any]:
    requires_physical = backend in {"metal", "ane", "cuda", "cpu"}
    runner: dict[str, Any] = {
        "kind": "hcli_protected_accelerator_bench" if model == "Qwen27" and backend == "metal" else "bounded_native_or_simulation",
        "command": [
            "python3",
            "-m",
            "hcli",
            "agentos",
            "protected-accelerator-bench",
            "--profile",
            "hcli/hawking-native.sealed-3.14.json",
            "--max-new-tokens",
            "32",
        ] if model == "Qwen27" and backend == "metal" else ["python3", "-m", "hcli", "agentos", "fpga-preboard"],
        "detached": True,
        "requires_quiescence": requires_physical,
        "protected_window": requires_physical,
    }
    row = {
        "experiment_id": experiment_id,
        "behavior_id": behavior_id,
        "target": {"model": model, "backend": backend, "organ": organ},
        "hypothesis": f"{candidate} reduces complete useful work for {model} {organ} without changing the accepted output",
        "candidate": candidate,
        "control": control,
        "metrics": [
            "complete_useful_wall_ns",
            "gpu_ns",
            "active_bytes_per_token",
            "dispatches",
            "synchronization_ns",
            "host_ceremony_ns",
            "fallback_count",
            "capability_verified",
        ],
        "verification_ladder": [
            "structural_compile_and_negative_controls",
            "diagnostic_relative_interleaved_ab",
            "protected_absolute_complete_wall_with_capability_gate",
        ],
        "falsifier": "no protected complete-wall improvement with identical oracle/output, zero fallback, and complete metric accounting",
        "runner": runner,
        "status": status,
        "blocked_reason": blocked_reason,
        "promotion": {
            "required_benchmark_class": "QUALIFIED_PROTECTED",
            "required_evidence_class": "HAWKING_PROTECTED_VERIFIED",
            "requires_independent_capability": True,
            "requires_zero_fallback": True,
            "requires_complete_active_bytes_or_explicit_absence": True,
        },
    }
    return row


def _experiments() -> list[dict[str, Any]]:
    return [
        _experiment(
            "qwen27-layout-algebra-mlp",
            behavior_id="layout_algebra", model="Qwen27", backend="metal", organ="mlp",
            candidate="parameterized packed GEMV layout/tile/lane mapping",
            control="sealed resident Qwen27 GeoTpr64Tg128 path",
        ),
        _experiment(
            "qwen27-graph-replay-token-skeleton",
            behavior_id="graph_replay", model="Qwen27", backend="metal", organ="decode",
            candidate="replay static token graph with dynamic token/position slots",
            control="current persistent executor with per-step command encoding",
        ),
        _experiment(
            "qwen27-stationary-packed-weight",
            behavior_id="stationary_representation", model="Qwen27", backend="metal", organ="mlp",
            candidate="keep packed representation resident and expose runtime active bytes separately",
            control="same-source packed path with no active-byte instrumentation",
        ),
        _experiment(
            "qwen27-async-double-buffer",
            behavior_id="async_double_buffer", model="Qwen27", backend="metal", organ="mlp",
            candidate="overlap next packed tile staging with current projection when ownership permits",
            control="current serial projection and command-buffer boundary",
        ),
        _experiment(
            "flash-direct-routed-accumulate",
            behavior_id="direct_routed_accumulate", model="Flash", backend="metal", organ="moe",
            candidate="route-before-payload with selected-expert direct weighted accumulation",
            control="current Flash routed-expert component graph",
            status="BLOCKED",
            blocked_reason="Flash native full-model executable/weights are not available in the current protected lane; retain as detached queue work",
        ),
        _experiment(
            "flash-local-state-machine",
            behavior_id="local_state_machine", model="Flash", backend="metal", organ="deltanet",
            candidate="persistent DeltaNet state machine with checkpoint-bisection verifier",
            control="current Flash stateful seam",
            status="BLOCKED",
            blocked_reason="requires the Flash protected complete-token runtime lane",
        ),
        _experiment(
            "flash-semantic-transport-hwir",
            behavior_id="semantic_transport", model="Flash", backend="fpga", organ="route_and_state",
            candidate="typed route metadata/activation/partial-reduction edges in HWIR",
            control="untyped partition boundary",
            status="READY",
        ),
        _experiment(
            "qwen27-move-or-recompute-boundary",
            behavior_id="move_or_recompute", model="Qwen27", backend="metal", organ="all",
            candidate="costed dependency planner chooses resident/recompute/prefetch by complete boundary cost",
            control="framework-prescribed movement",
        ),
        _experiment(
            "ane-regular-island-probe",
            behavior_id="npu_regular_island", model="Qwen27", backend="ane", organ="normalization",
            candidate="public Core ML/ML Program regular island with explicit transfer accounting",
            control="Metal normalization path",
            status="BLOCKED",
            blocked_reason="public ANE compile/runtime measurement is not available in this process; plan-only until the public path is executable",
        ),
    ]


def _asic_ledger(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    watched = {
        "fused_decode_compute": "compact representation decode + projection",
        "local_state_machine": "persistent recurrent state transition",
        "direct_routed_accumulate": "route-aware selected payload and accumulation",
        "semantic_transport": "typed data movement/reduction edge",
        "stationary_representation": "representation/data stationarity",
    }
    out = []
    for entry in entries:
        bid = str(entry["behavior_id"])
        if bid not in watched:
            continue
        out.append({
            "candidate_id": bid,
            "primitive": entry["hawking_primitive"],
            "candidate_physical_law": watched[bid],
            "source_school_count": len(entry["source_architecture_ecosystem"]),
            "cross_model_survival": "NOT_ESTABLISHED",
            "software_optimization_exhausted": False,
            "fpga_stable": False,
            "cross_generation_hardware_survival": False,
            "status": "WATCHLIST",
            "asic_candidate": False,
            "promotion_gate": [
                "survives Qwen27 and Flash protected measurements",
                "survives multiple representations and machines",
                "software cannot remove the primitive",
                "FPGA simulation and hardware repeatedly prove a stable spatial form",
                "reconfigurability no longer earns its cost",
            ],
            "claim_boundary": "interesting cross-architecture behavior is not an ASIC recommendation",
        })
    return out


def _hwir_hypotheses(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "behavior_id": str(entry["behavior_id"]),
            "primitive": entry["hawking_primitive"],
            "hwir_node_kind": "spatial_region" if entry["hawking_primitive"] in {"SpatialPipeline", "LocalStateMachine", "FusedDecodeCompute"} else "dataflow_region",
            "buffers": ["resident_representation", "token_activation", "partial_reduction", "persistent_state"],
            "semantic_edges": ["activation", "state", "compact_representation", "partial_reduction"],
            "placement_constraint": entry["physical_graph_mapping"],
            "label": "[D] hypothesis; no board or hardware timing claim",
            "status": "CANDIDATE",
        }
        for entry in entries
    ]


def _score(entry: Mapping[str, Any]) -> float:
    value = entry["ev_inputs"]
    numerator = (
        int(value["token_ns_reduction"])
        * int(value["transfer_breadth"])
        * int(value["model_applicability"])
        * int(value["future_hardware_value"])
        * int(value["information_gain"])
    )
    return round(numerator / int(entry["implementation_difficulty"]), 3)


def build_atlas(*, repo_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve() if repo_root else REPO
    entries = _raw_entries()
    for entry in entries:
        entry["behavior_taxonomy"] = _taxonomy_for(str(entry["behavior_id"]))
        entry["expected_value_score"] = _score(entry)
    entries.sort(key=lambda row: (-row["expected_value_score"], row["behavior_id"]))
    experiments = _experiments()
    queue = {
        "schema": QUEUE_SCHEMA,
        "selection_rule": "expected token_ns reduction × transfer breadth × model applicability × future hardware value × information gain / implementation cost",
        "experiments": experiments,
        "claim_boundary": "queue records are hypotheses until a protected complete-token receipt satisfies the promotion contract",
    }
    artifact_inputs = [
        "receipts/headless/CUDA_CAPABILITY_LEDGER.json",
        "receipts/headless/APPLE_ANE_ATLAS.json",
        "receipts/headless/PHYSICAL_GRAPH_COMPILER.json",
        "receipts/headless/QWEN38_ACCELERATOR_TRANSFER_MAP.json",
        "receipts/headless/QWEN27_FPGA_ORGAN_MAP.json",
        "receipts/headless/FLASH_NEXT_FPGA_ORGAN_MAP.json",
    ]
    evidence_inputs = [
        # Keep the generated artifact portable across worktrees. Absolute
        # checkout paths made an otherwise identical atlas fingerprint stale
        # whenever the repository moved or was inspected from a worktree.
        {"path": relative, "present": (root / relative).is_file()}
        for relative in artifact_inputs
    ]
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "bench": dict(PLANNING_BENCH),
        "source_schools": list(SOURCE_SCHOOLS),
        "source_technique_coverage": _source_technique_coverage(),
        "behavior_taxonomy": list(BEHAVIOR_TAXONOMY),
        "backend_neutral_primitives": list(PRIMITIVES),
        "entries": entries,
        "experiment_queue": queue,
        "hwir_hypotheses": _hwir_hypotheses(entries),
        "asic_candidate_ledger": {
            "schema": ASIC_SCHEMA,
            "entries": _asic_ledger(entries),
            "promotion_rule": "no candidate is ASIC-worthy until cross-model, cross-representation, FPGA, and cross-generation evidence survives",
        },
        "evidence_inputs": evidence_inputs,
        "canonicalization_policy": {
            "diagnostic": "may rank a hypothesis, never promote it",
            "protected": "requires complete useful wall, independent capability, zero fallback, and explicit metric scope",
            "cross_model": "a law must re-earn transfer on Qwen27 and Flash before generic status",
            "fpga": "HWIR and link/cycle simulations remain [D]/[S] until hardware receipts exist",
            "asic": "watchlist only until the repeated-survivor gates are all true",
        },
        "claim_boundary": "Architecture schools are textbooks. This artifact records Hawking hypotheses and existing implementation surfaces; it does not convert source claims into Hawking performance claims.",
    }
    body["fingerprint"] = _hash({key: value for key, value in body.items() if key != "fingerprint"})
    return body


def validate_atlas(atlas: Mapping[str, Any]) -> dict[str, Any]:
    if atlas.get("schema") != SCHEMA:
        raise AtlasValidationError(f"schema must be {SCHEMA}")
    for key in ("source_schools", "source_technique_coverage", "behavior_taxonomy", "backend_neutral_primitives", "entries", "experiment_queue", "hwir_hypotheses", "asic_candidate_ledger"):
        if not isinstance(atlas.get(key), (list, dict)):
            raise AtlasValidationError(f"atlas.{key} is missing or has the wrong type")
    schools = set(atlas["source_schools"])
    missing_schools = set(SOURCE_SCHOOLS) - schools
    if missing_schools:
        raise AtlasValidationError(f"source schools missing: {sorted(missing_schools)}")
    taxonomy = set(atlas["behavior_taxonomy"])
    if set(BEHAVIOR_TAXONOMY) - taxonomy:
        raise AtlasValidationError("behavior taxonomy is incomplete")
    primitives = set(atlas["backend_neutral_primitives"])
    if set(PRIMITIVES) - primitives:
        raise AtlasValidationError("backend-neutral primitive vocabulary is incomplete")
    entries = atlas["entries"]
    if not isinstance(entries, list) or len(entries) < 12:
        raise AtlasValidationError("atlas needs a bounded but real first sweep of at least 12 behaviors")
    ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise AtlasValidationError("every entry must be an object")
        bid = str(entry.get("behavior_id") or "")
        if not bid or bid in ids:
            raise AtlasValidationError(f"duplicate or missing behavior_id: {bid!r}")
        ids.add(bid)
        required = (
            "source_architecture_ecosystem", "source_behavior", "fundamental_physical_idea",
            "vendor_specific_assumptions", "architecture_independent_invariant", "hawking_primitive",
            "physical_graph_mapping", "applicable_models", "applicable_organs", "applicable_backends",
            "expected_effect", "implementation_difficulty", "transfer_confidence", "cheapest_falsifier",
            "status", "evidence_class", "source_evidence",
        )
        missing = [key for key in required if key not in entry]
        if missing:
            raise AtlasValidationError(f"{bid} missing {missing}")
        if entry["hawking_primitive"] not in primitives:
            raise AtlasValidationError(f"{bid} uses a non-Hawking primitive")
        if entry["status"] not in STATUSES:
            raise AtlasValidationError(f"{bid} has invalid status {entry['status']!r}")
        if entry["evidence_class"] not in EVIDENCE_CLASSES:
            raise AtlasValidationError(f"{bid} has invalid evidence class {entry['evidence_class']!r}")
        if entry["transfer_confidence"] not in TRANSFER_CONFIDENCE:
            raise AtlasValidationError(f"{bid} has invalid transfer confidence")
        if not isinstance(entry["cheapest_falsifier"], str) or not entry["cheapest_falsifier"].strip():
            raise AtlasValidationError(f"{bid} has no falsifier")
        if not isinstance(entry["implementation_difficulty"], int) or not 1 <= entry["implementation_difficulty"] <= 5:
            raise AtlasValidationError(f"{bid} difficulty must be 1..5")
        effects = entry["expected_effect"]
        if set(effects) != set(EFFECT_METRICS):
            raise AtlasValidationError(f"{bid} expected_effect must cover every metric")
        if set(entry.get("behavior_taxonomy") or ()) - taxonomy:
            raise AtlasValidationError(f"{bid} uses an unknown taxonomy label")
        if not entry.get("source_evidence"):
            raise AtlasValidationError(f"{bid} has no source evidence")
    entry_ids = ids
    coverage = atlas["source_technique_coverage"]
    if not isinstance(coverage, list) or not coverage:
        raise AtlasValidationError("source technique coverage is missing or empty")
    for row in coverage:
        if not isinstance(row, Mapping):
            raise AtlasValidationError("source technique coverage must contain objects")
        if not all(str(row.get(key) or "").strip() for key in ("source_technique", "source_school", "behavior_id")):
            raise AtlasValidationError("source technique coverage rows need technique, school, and behavior")
        if row["source_school"] not in schools:
            raise AtlasValidationError(f"source technique references unknown school {row['source_school']!r}")
        if row["behavior_id"] not in entry_ids:
            raise AtlasValidationError(f"source technique references unknown behavior {row['behavior_id']!r}")
    queue = atlas["experiment_queue"]
    if queue.get("schema") != QUEUE_SCHEMA or not queue.get("experiments"):
        raise AtlasValidationError("experiment queue is missing or empty")
    for exp in queue["experiments"]:
        if exp.get("behavior_id") not in entry_ids:
            raise AtlasValidationError(f"experiment {exp.get('experiment_id')} references unknown behavior")
        for key in ("target", "hypothesis", "candidate", "control", "metrics", "verification_ladder", "falsifier", "runner", "promotion"):
            if key not in exp:
                raise AtlasValidationError(f"experiment {exp.get('experiment_id')} missing {key}")
        if exp["runner"].get("detached") is not True:
            raise AtlasValidationError(f"experiment {exp.get('experiment_id')} must be detached")
        if not exp["falsifier"].strip():
            raise AtlasValidationError(f"experiment {exp.get('experiment_id')} has no falsifier")
    hwir = atlas["hwir_hypotheses"]
    if {row.get("behavior_id") for row in hwir} != entry_ids:
        raise AtlasValidationError("HWIR hypotheses must cover exactly the atlas entries")
    asic = atlas["asic_candidate_ledger"]
    if asic.get("schema") != ASIC_SCHEMA:
        raise AtlasValidationError("ASIC ledger has the wrong schema")
    for row in asic.get("entries") or []:
        if row.get("asic_candidate") is not False or row.get("status") != "WATCHLIST":
            raise AtlasValidationError("ASIC ledger may not promote an unproven candidate")
    expected_hash = _hash({key: value for key, value in atlas.items() if key != "fingerprint"})
    if atlas.get("fingerprint") != expected_hash:
        raise AtlasValidationError("atlas fingerprint does not match canonical body")
    return {
        "schema": "hawking.accelerator.architecture_atlas_validation.v1",
        "passed": True,
        "entry_count": len(entries),
        "experiment_count": len(queue["experiments"]),
        "hwir_hypothesis_count": len(hwir),
        "asic_watchlist_count": len(asic.get("entries") or []),
        "source_school_count": len(schools),
        "claim_boundary": "validation proves schema/funnel invariants, not physical performance",
    }


def promotion_decision(
    *,
    benchmark_class: str | None,
    evidence_class: str | None,
    complete_useful_wall_ns: int | float | None,
    capability_verified: bool | None,
    fallback_count: int | None,
    active_bytes_per_token: int | float | None,
) -> dict[str, Any]:
    """Apply the same conservative gate to a candidate primitive result."""
    reasons: list[str] = []
    if str(benchmark_class or "").upper() not in {"PROTECTED_ABSOLUTE", "QUALIFIED_PROTECTED"}:
        reasons.append("protected_absolute_evidence_required")
    if evidence_class != "HAWKING_PROTECTED_VERIFIED":
        reasons.append("protected_evidence_class_required")
    if complete_useful_wall_ns is None or float(complete_useful_wall_ns) <= 0:
        reasons.append("complete_useful_wall_unmeasured")
    if capability_verified is not True:
        reasons.append("independent_capability_not_verified")
    if fallback_count != 0:
        reasons.append("zero_fallback_evidence_required")
    if active_bytes_per_token is None or float(active_bytes_per_token) <= 0:
        reasons.append("active_bytes_per_token_unmeasured")
    return {
        "promotion_allowed": not reasons,
        "reasons": reasons,
        "authority": "protected complete useful wall + independent capability + zero fallback + explicit active-byte scope",
    }


def emit_atlas(*, repo_root: str | Path | None = None, output: str | Path | None = None) -> Path:
    root = Path(repo_root).expanduser().resolve() if repo_root else REPO
    destination = Path(output).expanduser() if output else root / "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json"
    if not destination.is_absolute():
        destination = root / destination
    atlas = build_atlas(repo_root=root)
    validate_atlas(atlas)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(atlas, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--emit", default=None)
    parser.add_argument("--validate", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.validate:
        path = Path(args.validate).expanduser()
        atlas = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(validate_atlas(atlas), indent=2, sort_keys=True))
        return 0
    path = emit_atlas(repo_root=args.repo_root, output=args.emit)
    print(json.dumps({"status": "PASSED", "path": str(path), "fingerprint": build_atlas(repo_root=args.repo_root)["fingerprint"]}, sort_keys=True))
    return 0


__all__ = [
    "ASIC_SCHEMA",
    "AtlasValidationError",
    "BEHAVIOR_TAXONOMY",
    "EFFECT_METRICS",
    "EVIDENCE_CLASSES",
    "PRIMITIVES",
    "QUEUE_SCHEMA",
    "SCHEMA",
    "SOURCE_SCHOOLS",
    "STATUSES",
    "build_atlas",
    "emit_atlas",
    "main",
    "promotion_decision",
    "validate_atlas",
]


if __name__ == "__main__":
    raise SystemExit(main())
