"""PHYSICAL PRIMITIVES — executable contracts for the 17 atlas primitives.

The Architecture Atlas already names 17 backend-neutral primitives as data.
This module is their executable form: NR -> PhysicalGraph -> backend lowering,
each carrying memory-tier identity, so CUDA graphs, TPU compiled graphs, FPGA
spatial pipelines and deterministic dataflow collapse into one Hawking-owned
abstraction rather than a per-vendor borrow.

Atlas names win. Directive aliases are recorded, never promoted over the atlas.

    python3 tools/future/physical_primitives.py --selftest
    python3 tools/future/physical_primitives.py --build
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))


import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from tools.future._common import REPO, load_json, write_receipt

RECEIPT = "PHYSICAL_PRIMITIVES.json"
SCHEMA = "hawking.future.physical_primitives.v1"
PHYSICAL_GRAPH_SCHEMA = "hcli.physical_graph.v1"
LOWERING_SCHEMA = "hawking.future.physical_primitives.lowering.v1"

ATLAS_REL = "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json"
FLASH_NR_REL = "receipts/headless/FLASH_COMPLETE_V2.nr.json"
CUDA_CENSUS_REL = "receipts/headless/CUDA_CAPABILITY_CENSUS.json"

# Recovered byte-for-byte from the atlas `backend_neutral_primitives` list
# (atlas fingerprint e763623e8a4ddcfc8350d6b5f680284a23db68f8bfbca53428e258d24adfc2ab).
# Order is atlas order. Do not reorder; do not invent a parallel list.
ATLAS_PRIMITIVES: tuple[str, ...] = (
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

ATLAS_FINGERPRINT = "e763623e8a4ddcfc8350d6b5f680284a23db68f8bfbca53428e258d24adfc2ab"
ATLAS_SCHEMA = "hawking.accelerator.architecture_atlas.v1"

BEHAVIOR_TAXONOMY: tuple[str, ...] = (
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

MEMORY_TIERS: tuple[str, ...] = (
    "REGISTER",
    "THREADGROUP",
    "UMA",
    "HBM",
    "ACCEL_SRAM",
    "REMOTE",
)

BACKENDS: tuple[str, ...] = ("METAL", "FPGA", "CUDA", "ANE")

# Representation-operator vocabulary already owned by PhysicalGraph (parent
# working tree). Different layer from the 17 atlas primitives; do not merge.
NR_OPERATOR_VOCABULARY: tuple[str, ...] = (
    "BASIS_PROJECT",
    "COEFFICIENT_APPLY",
    "SPARSE_RESIDUAL",
    "QUANT_PROJECT",
    "CODEBOOK_LOOKUP",
    "ROUTED_SELECT",
    "WEIGHTED_ACCUMULATE",
    "STATE_UPDATE",
)

REQUIRED_CONTRACT_FIELDS: tuple[str, ...] = (
    "name",
    "atlas_index",
    "behavior_ids",
    "behavior_taxonomy",
    "invariant",
    "cost_removed",
    "organ_classes",
    "preconditions",
    "cheapest_falsifier",
    "legal_memory_tiers",
    "physical_graph_mapping",
    "aliases",
    "atlas_status",
    "in_atlas",
)


class PrimitiveError(ValueError):
    """Base error for the primitive library."""


class UnknownPrimitiveError(PrimitiveError):
    """A name that is not an atlas primitive (or the one directive extra)."""


class IllegalMemoryTierError(PrimitiveError):
    """A primitive instance requested a tier it cannot occupy."""


class BackendUnavailableError(PrimitiveError):
    """Lowering to a seam whose availability is UNAVAILABLE."""


class UnknownBackendError(PrimitiveError):
    """A backend name outside METAL / FPGA / CUDA / ANE."""


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha(obj: Any) -> str:
    return hashlib.sha256(_canon(obj).encode()).hexdigest()


def _contract(**kwargs: Any) -> dict[str, Any]:
    missing = [f for f in REQUIRED_CONTRACT_FIELDS if f not in kwargs]
    if missing:
        raise ValueError(f"contract {kwargs.get('name')!r} missing {missing}")
    kwargs["legal_memory_tiers"] = tuple(kwargs["legal_memory_tiers"])
    kwargs["organ_classes"] = tuple(kwargs["organ_classes"])
    kwargs["behavior_ids"] = tuple(kwargs["behavior_ids"])
    kwargs["behavior_taxonomy"] = tuple(kwargs["behavior_taxonomy"])
    kwargs["preconditions"] = tuple(kwargs["preconditions"])
    kwargs["aliases"] = tuple(kwargs["aliases"])
    illegal = [t for t in kwargs["legal_memory_tiers"] if t not in MEMORY_TIERS]
    if illegal:
        raise ValueError(f"{kwargs['name']}: illegal tier names {illegal}")
    return kwargs


def _tiers(*names: str) -> tuple[str, ...]:
    return tuple(names)


# Occupancy is "which tiers this primitive is allowed to inhabit", not a
# measurement of any machine. REGISTER/THREADGROUP are on-device local;
# UMA is Apple unified memory; HBM is discrete high-bandwidth memory;
# ACCEL_SRAM is on-accelerator SRAM/URAM; REMOTE is another device or host.
_LOCAL = _tiers("REGISTER", "THREADGROUP", "UMA", "HBM", "ACCEL_SRAM")
_DEVICE = _tiers("THREADGROUP", "UMA", "HBM", "ACCEL_SRAM")
_GRAPH = _tiers("UMA", "HBM", "ACCEL_SRAM")
_EDGE = _tiers("THREADGROUP", "UMA", "HBM", "ACCEL_SRAM", "REMOTE")
_ALL = MEMORY_TIERS


# ---------------------------------------------------------------------------
# Contracts. Atlas entries are the source. Three list-only atlas names
# (AsyncPrefetch, TiledProjection, MemoryTierIdentity) have no entries;
# their contracts are derived from taxonomy / technique coverage / the
# identity rule already stated on PhysicalGraph, and the reconciliation
# report says so. ConditionalPhysicalProgram's two atlas entries are one
# primitive (atlas already reused the name).
# ---------------------------------------------------------------------------

CONTRACTS: dict[str, dict[str, Any]] = {}


def _register(c: dict[str, Any]) -> None:
    CONTRACTS[c["name"]] = c


_register(_contract(
    name="PersistentPhysicalRegion",
    atlas_index=1,
    behavior_ids=("persistent_physical_region",),
    behavior_taxonomy=("PERSISTENT_EXECUTION", "STATE_RESIDENCY", "LOCAL_MEMORY"),
    invariant=(
        "when state and bindings remain valid, repeated host entry is not "
        "semantically required"
    ),
    cost_removed={
        "mechanism": "amortize launch and binding ceremony",
        "metrics": (
            "experiment_turnaround_ns",
            "host_ceremony",
            "state_movement",
            "synchronization",
            "token_ns",
        ),
    },
    organ_classes=("decode", "deltanet", "moe", "attention", "sampling"),
    preconditions=(
        "execution_policy.process is a long_lived_executor",
        "residency.weights is resident",
        "residency.state is sequence-scoped",
        "bindings remain valid across token steps",
    ),
    cheapest_falsifier=(
        "protected complete useful wall time does not improve against the same "
        "resident control, with identical output and zero fallback"
    ),
    legal_memory_tiers=_DEVICE,
    physical_graph_mapping={
        "execution_policy.process": "long_lived_executor",
        "residency.state": "sequence",
        "residency.weights": "resident",
    },
    aliases=(),
    atlas_status="IMPLEMENTED",
    in_atlas=True,
    fundamental_physical_idea=(
        "keep executable state and reusable bindings alive across token steps"
    ),
))

_register(_contract(
    name="StationaryRepresentation",
    atlas_index=2,
    behavior_ids=("stationary_representation",),
    behavior_taxonomy=("DATA_STATIONARITY", "LOCAL_MEMORY", "LOW_BIT_ARITHMETIC"),
    invariant=(
        "the best stationary operand is the one whose reuse exceeds the cost of "
        "retaining it"
    ),
    cost_removed={
        "mechanism": "avoid repeated operand movement",
        "metrics": ("active_bytes", "host_ceremony", "state_movement", "token_ns"),
    },
    organ_classes=("mlp", "moe", "embedding", "lm_head", "codebook"),
    preconditions=(
        "representation is packed_native at the occupying tier",
        "residency carries an explicit stationarity_contract",
        "memory tier is a component of executable identity",
        "the operand is not also declared REMOTE (REMOTE is movement)",
    ),
    cheapest_falsifier=(
        "packed resident A/B has no protected complete-wall benefit or violates "
        "the no-dense-rematerialization/capability gate"
    ),
    legal_memory_tiers=_LOCAL,
    physical_graph_mapping={
        "memory": "tier_is_executable_identity",
        "representation": "packed_native",
        "residency": "stationarity_contract",
    },
    aliases=(),
    atlas_status="IMPLEMENTED",
    in_atlas=True,
    fundamental_physical_idea=(
        "spend residency and local storage to avoid repeatedly moving the "
        "dominant operand"
    ),
))

_register(_contract(
    name="AsyncPrefetch",
    atlas_index=3,
    behavior_ids=("async_prefetch",),
    behavior_taxonomy=("ASYNC_PREFETCH", "STREAMING", "CROSS_DEVICE_OVERLAP"),
    invariant=(
        "a prefetch is legal only when the destination is not in use by the "
        "consumer and the source remains valid until the copy completes"
    ),
    cost_removed={
        "mechanism": "hide independent movement behind compute without requiring two owned buffers",
        "metrics": ("state_movement", "token_ns"),
    },
    organ_classes=("mlp", "moe", "deltanet", "kv"),
    preconditions=(
        "source remains valid until the copy completes",
        "destination is not concurrently consumed",
        "overlap is a measured window, not an assumed one",
        "Fusion ISA PREFETCH is a timeline hint, not this primitive",
    ),
    cheapest_falsifier=(
        "the prefetch destination is still in use when the consumer starts, or "
        "a protected complete wall does not improve after the copy/fence cost"
    ),
    legal_memory_tiers=_EDGE,
    physical_graph_mapping={
        "execution_policy": "overlap_when_measured",
        "memory": "async_prefetch_staging",
        "synchronization": "source_valid_until_copy_complete",
    },
    aliases=("ASYNC_PREFETCH behavior",),
    atlas_status="LIST_ONLY",
    in_atlas=True,
    fundamental_physical_idea=(
        "stage the next operand asynchronously; one-sided overlap, not dual-buffer ownership"
    ),
    atlas_entry="absent; technique coverage folds prefetch into async_double_buffer",
))

_register(_contract(
    name="DoubleBufferedTile",
    atlas_index=4,
    behavior_ids=("async_double_buffer",),
    behavior_taxonomy=("ASYNC_PREFETCH", "DOUBLE_BUFFERING", "STREAMING"),
    invariant=(
        "a transfer can be hidden only inside a real overlap window with "
        "explicit producer/consumer ownership"
    ),
    cost_removed={
        "mechanism": "hide movement behind independent work",
        "metrics": ("state_movement", "synchronization", "token_ns"),
    },
    organ_classes=("mlp", "moe", "deltanet", "kv"),
    preconditions=(
        "two ownership-safe buffers exist at the occupying tier",
        "producer/consumer fences are explicit",
        "overlap is measured, not assumed",
    ),
    cheapest_falsifier=(
        "measured overlap window is zero or protected complete wall is not "
        "lower after fence costs"
    ),
    legal_memory_tiers=_DEVICE,
    physical_graph_mapping={
        "execution_policy": "overlap_when_measured",
        "memory": "double_buffered_tiles",
        "synchronization": "producer_consumer_fences",
    },
    aliases=("async_double_buffer",),
    atlas_status="MAPPED",
    in_atlas=True,
    fundamental_physical_idea=(
        "overlap independent movement and compute with two ownership-safe buffers"
    ),
))

_register(_contract(
    name="SpatialPipeline",
    atlas_index=5,
    behavior_ids=("spatial_local_pipeline",),
    behavior_taxonomy=("SPATIAL_EXECUTION", "LOCAL_MEMORY", "COMPUTE_IN_TRANSIT"),
    invariant=(
        "if an intermediate is not externally observable, its global "
        "materialization is optional"
    ),
    cost_removed={
        "mechanism": "fuse producer-consumer regions and keep intermediates local",
        "metrics": (
            "active_bytes",
            "host_ceremony",
            "state_movement",
            "synchronization",
            "token_ns",
        ),
    },
    organ_classes=("mlp", "deltanet", "moe", "attention"),
    preconditions=(
        "producer-consumer chain is placed as spatial regions",
        "intermediates are not observable outside the region",
        "semantic edges name payload, owner, and order",
    ),
    cheapest_falsifier=(
        "fused region fails numerical parity or protected complete wall "
        "increases after local-storage/fence cost"
    ),
    legal_memory_tiers=_tiers("REGISTER", "THREADGROUP", "ACCEL_SRAM", "HBM"),
    physical_graph_mapping={
        "computation": "spatial_regions",
        "data": "semantic_edges",
        "memory": "local_intermediates",
    },
    aliases=(),
    atlas_status="MAPPED",
    in_atlas=True,
    fundamental_physical_idea=(
        "make locality and pipeline ownership explicit in the graph"
    ),
))

_register(_contract(
    name="FusedDecodeCompute",
    atlas_index=6,
    behavior_ids=("fused_decode_compute",),
    behavior_taxonomy=(
        "FUSED_REPRESENTATION_DECODE",
        "LOW_BIT_ARITHMETIC",
        "COMPUTE_IN_TRANSIT",
    ),
    invariant=(
        "a representation-native consumer may remove an intermediate only when "
        "its numerical contract is preserved"
    ),
    cost_removed={
        "mechanism": "remove dense rematerialization and its write/read",
        "metrics": (
            "active_bytes",
            "experiment_turnaround_ns",
            "flops",
            "state_movement",
            "token_ns",
        ),
    },
    organ_classes=("mlp", "moe", "lm_head", "codebook"),
    preconditions=(
        "consumer is representation-native (no dense rematerialization)",
        "numerical contract of the unfused path is preserved",
        "decode happens at the occupying tier, not via a host round-trip",
    ),
    cheapest_falsifier=(
        "parity, fallback, or protected complete-wall gate fails against the "
        "same-source dense/control path"
    ),
    legal_memory_tiers=_LOCAL,
    physical_graph_mapping={
        "computation": "projection_plus_decode",
        "memory": "no_dense_rematerialization",
        "representation": "native_decode",
    },
    aliases=(),
    atlas_status="PHYSICALLY_MEASURED",
    in_atlas=True,
    fundamental_physical_idea=(
        "fuse representation decode with arithmetic at the consumer"
    ),
    atlas_status_note=(
        "PHYSICALLY_MEASURED is the atlas/Codex evidence class, not a sidecar "
        "measurement. This module emits CONTRACT_ONLY / STATIC_ONLY."
    ),
))

_register(_contract(
    name="DirectRoutedAccumulate",
    atlas_index=7,
    behavior_ids=("direct_routed_accumulate",),
    behavior_taxonomy=(
        "ROUTE_AWARE_EXECUTION",
        "DYNAMIC_SCHEDULING",
        "SPARSITY_SKIPPING",
    ),
    invariant=(
        "for few active tokens, route metadata and selected payload locality "
        "matter more than nominal expert throughput"
    ),
    cost_removed={
        "mechanism": "skip inactive experts and eliminate staging copies",
        "metrics": ("active_bytes", "flops", "state_movement", "token_ns"),
    },
    organ_classes=("moe", "router", "shared_expert", "expert_cache"),
    preconditions=(
        "route metadata is resident at the occupying tier",
        "payloads are the selected subset only",
        "accumulate is in the same physical region as the selected experts",
    ),
    cheapest_falsifier=(
        "selected-route protected complete wall does not beat current Flash "
        "control after route/gather/accumulate costs"
    ),
    legal_memory_tiers=_DEVICE,
    physical_graph_mapping={
        "computation": "route_then_native_expert",
        "data": "selected_payload_only",
        "state": "route_metadata_resident",
    },
    aliases=(),
    atlas_status="DIAGNOSTIC",
    in_atlas=True,
    fundamental_physical_idea=(
        "treat routing and accumulation as one physical region around selected payloads"
    ),
))

_register(_contract(
    name="LocalStateMachine",
    atlas_index=8,
    behavior_ids=("local_state_machine",),
    behavior_taxonomy=("STATE_RESIDENCY", "PERSISTENT_EXECUTION", "LOCAL_MEMORY"),
    invariant=(
        "mutable state has one authoritative owner and its update ordering is "
        "part of the executable identity"
    ),
    cost_removed={
        "mechanism": "keep mutable sequence state resident and ordered",
        "metrics": ("host_ceremony", "state_movement", "synchronization", "token_ns"),
    },
    organ_classes=("deltanet", "kv", "routing", "ngram", "mtp"),
    preconditions=(
        "exactly one authoritative resident owner for the state",
        "state-update edges are explicit in the graph",
        "update ordering is part of physical identity",
    ),
    cheapest_falsifier=(
        "device-resident state does not improve protected complete wall or "
        "fails checkpoint/bisection parity"
    ),
    legal_memory_tiers=_LOCAL,
    physical_graph_mapping={
        "execution_policy": "fixed_state_transitions",
        "state": "authoritative_resident_owner",
        "synchronization": "state_update_edges",
    },
    aliases=(),
    atlas_status="IMPLEMENTED",
    in_atlas=True,
    fundamental_physical_idea=(
        "model decode as a state machine rather than reconstructing stateless "
        "operators every step"
    ),
))

_register(_contract(
    name="SemanticTransportEdge",
    atlas_index=9,
    behavior_ids=("semantic_transport",),
    behavior_taxonomy=("GLOBAL_MEMORY", "CROSS_DEVICE_OVERLAP", "COMPUTE_IN_TRANSIT"),
    invariant=(
        "a transfer is optimizable only when its payload, owner, ordering, and "
        "reduction semantics are explicit"
    ),
    cost_removed={
        "mechanism": "avoid untyped copies and choose topology-aware movement",
        "metrics": (
            "active_bytes",
            "experiment_turnaround_ns",
            "state_movement",
            "synchronization",
            "token_ns",
        ),
    },
    organ_classes=("moe", "attention", "deltanet", "fpga_partition"),
    preconditions=(
        "edges are typed (activation / state / compact_representation / partial_reduction)",
        "owner and order are declared",
        "the copy is not an opaque memcpy",
    ),
    cheapest_falsifier=(
        "semantic edge accounting cannot reproduce the reference output or "
        "link cost erases the proposed benefit"
    ),
    legal_memory_tiers=_EDGE,
    physical_graph_mapping={
        "dependencies": "typed_transport_edges",
        "device_placement": "topology_aware",
        "synchronization": "edge_ownership_and_order",
    },
    aliases=(),
    atlas_status="IMPLEMENTED",
    in_atlas=True,
    fundamental_physical_idea=(
        "make data movement a semantic graph edge instead of an opaque memory copy"
    ),
))

_register(_contract(
    name="TiledProjection",
    atlas_index=10,
    behavior_ids=("tiled_projection",),
    behavior_taxonomy=("TILING", "LAYOUT_TRANSFORMATION", "LOW_BIT_ARITHMETIC"),
    invariant=(
        "a projection may be factored into tiles only when tile ownership "
        "covers the logical result without changing the numerical contract"
    ),
    cost_removed={
        "mechanism": "execute the projection as owned tiles rather than a whole-operand round-trip",
        "metrics": ("active_bytes", "state_movement", "token_ns"),
    },
    organ_classes=("mlp", "moe", "attention", "lm_head"),
    preconditions=(
        "tile ownership covers the logical result",
        "layout (LayoutTransform) is already chosen or is the identity layout",
        "numerical contract of the untiled projection is preserved",
    ),
    cheapest_falsifier=(
        "tiled result disagrees with the untiled projection, or a protected "
        "complete wall does not improve after tile/fence cost"
    ),
    legal_memory_tiers=_LOCAL,
    physical_graph_mapping={
        "computation": "tiled_projection",
        "representation": "tile_owned_result",
        "memory": "tile_working_set",
    },
    aliases=("tiled GEMV/GEMM technique",),
    atlas_status="LIST_ONLY",
    in_atlas=True,
    fundamental_physical_idea=(
        "the projection operator itself is tiled; layout algebra (LayoutTransform) "
        "chooses packing, this primitive is the tiled operator"
    ),
    atlas_entry="absent; technique coverage files tiled GEMV/GEMM under layout_algebra",
))

_register(_contract(
    name="LayoutTransform",
    atlas_index=11,
    behavior_ids=("layout_algebra",),
    behavior_taxonomy=("LAYOUT_TRANSFORMATION", "TILING", "LOW_BIT_ARITHMETIC"),
    invariant=(
        "the same logical operation may have materially different movement and "
        "reduction costs under different legal layouts"
    ),
    cost_removed={
        "mechanism": "reduce strided movement and tails through shape-aware mapping",
        "metrics": ("active_bytes", "experiment_turnaround_ns", "synchronization", "token_ns"),
    },
    organ_classes=("mlp", "moe", "attention", "deltanet"),
    preconditions=(
        "logical, physical, tile, lane, and arithmetic mappings are distinct compiler objects",
        "nominal utilization is not authority",
        "the transform is a legal layout of the same logical tensor",
    ),
    cheapest_falsifier=(
        "same-source protected A/B at the chosen organ shows no complete-wall "
        "or active-byte benefit"
    ),
    legal_memory_tiers=_LOCAL,
    physical_graph_mapping={
        "computation": "tile_and_lane_mapping",
        "precision": "representation_grouping",
        "representation": "layout_algebra",
    },
    aliases=("layout_algebra",),
    atlas_status="DIAGNOSTIC",
    in_atlas=True,
    fundamental_physical_idea=(
        "choose layout and thread/tile ownership as compiler objects rather "
        "than kernel folklore"
    ),
))

_register(_contract(
    name="SparseSkip",
    atlas_index=12,
    behavior_ids=("sparse_conditional_execution",),
    behavior_taxonomy=(
        "SPARSITY_SKIPPING",
        "CONDITIONAL_EXECUTION",
        "ROUTE_AWARE_EXECUTION",
    ),
    invariant=(
        "skipping is legal only when the omitted contribution is proven zero "
        "or outside the selected computation"
    ),
    cost_removed={
        "mechanism": "avoid provably inactive work",
        "metrics": ("active_bytes", "flops", "token_ns"),
    },
    organ_classes=("sparse_attention", "moe", "residual", "router"),
    preconditions=(
        "omitted contribution is proven zero or unselected",
        "index/branch overhead is accounted in the complete wall",
        "output ordering and numerical semantics are preserved",
    ),
    cheapest_falsifier=(
        "sparse control does not preserve output or its index/branch overhead "
        "exceeds the omitted work in protected complete wall"
    ),
    legal_memory_tiers=_LOCAL,
    physical_graph_mapping={
        "computation": "conditional_regions",
        "data": "sparse_indices_and_payloads",
        "qualification": "parity_required",
    },
    aliases=(),
    atlas_status="DIAGNOSTIC",
    in_atlas=True,
    fundamental_physical_idea=(
        "make conditional work explicit in the physical program and pay only "
        "for selected operands"
    ),
))

_register(_contract(
    name="ConditionalPhysicalProgram",
    atlas_index=13,
    behavior_ids=("static_dynamic_skeleton", "npu_regular_island"),
    behavior_taxonomy=(
        "STATIC_SCHEDULING",
        "DYNAMIC_SCHEDULING",
        "CONDITIONAL_EXECUTION",
        "GRAPH_REPLAY",
        "DATA_STATIONARITY",
    ),
    invariant=(
        "static structure and dynamic control can coexist when dynamic choices "
        "do not change buffer ownership unsafely"
    ),
    cost_removed={
        "mechanism": "precompute safe scheduling and retain dynamic controls",
        "metrics": (
            "experiment_turnaround_ns",
            "host_ceremony",
            "synchronization",
            "token_ns",
        ),
    },
    organ_classes=(
        "decode",
        "deltanet",
        "moe",
        "kv",
        "sampling",
        "normalization",
        "silu",
        "regular_mlp",
    ),
    preconditions=(
        "dependencies are precomputed where dynamic slots permit",
        "dynamic slots are bounded (token, position, route, sampling, variable_state)",
        "a backend is chosen per organ only when transfer/compile/residency costs fit the graph",
    ),
    cheapest_falsifier=(
        "a bounded dynamic slot requires topology rebuild or changes "
        "output/capability under a protected replay"
    ),
    legal_memory_tiers=_LOCAL,
    physical_graph_mapping={
        "dependencies": "precomputed",
        "execution_policy": "static_skeleton_plus_dynamic_slots",
        "synchronization": "precomputed_where_safe",
        "device_placement": "organ_level_choice",
    },
    aliases=("static_dynamic_skeleton", "npu_regular_island"),
    atlas_status="IMPLEMENTED",
    in_atlas=True,
    fundamental_physical_idea=(
        "remove runtime scheduling decisions from the critical path without "
        "pretending dynamic MoE is static"
    ),
    merged_atlas_entries=(
        {
            "behavior_id": "static_dynamic_skeleton",
            "atlas_status": "IMPLEMENTED",
            "role": "primary",
        },
        {
            "behavior_id": "npu_regular_island",
            "atlas_status": "MAPPED",
            "role": "same primitive, organ-level backend choice",
            "invariant": (
                "a backend is useful only when its transfer, compile, "
                "synchronization, and residency costs fit the graph"
            ),
        },
    ),
))

_register(_contract(
    name="GraphReplay",
    atlas_index=14,
    behavior_ids=("graph_replay",),
    behavior_taxonomy=("GRAPH_REPLAY", "STATIC_SCHEDULING", "CONDITIONAL_EXECUTION"),
    invariant=(
        "a stable dependency graph can be reused if dynamic values do not "
        "alter topology"
    ),
    cost_removed={
        "mechanism": "reuse command topology and reduce rebuilds",
        "metrics": (
            "experiment_turnaround_ns",
            "host_ceremony",
            "synchronization",
            "token_ns",
        ),
    },
    organ_classes=("decode", "regular_mlp", "attention", "moe"),
    preconditions=(
        "pipeline_state is compile_once_reuse",
        "dynamic slots are token/position/route/sampling only",
        "replay does not change a dynamic route/state result",
    ),
    cheapest_falsifier=(
        "replay adds no protected complete-wall improvement or changes a "
        "dynamic route/state result"
    ),
    legal_memory_tiers=_GRAPH,
    physical_graph_mapping={
        "execution_policy.dynamic_slots": ["token", "position", "route", "sampling"],
        "execution_policy.pipeline_state": "compile_once_reuse",
    },
    aliases=("CUDA graph / TPU compiled graph / NPU executable — source schools, not this primitive",),
    atlas_status="DIAGNOSTIC",
    in_atlas=True,
    fundamental_physical_idea=(
        "compile the static skeleton once and expose only dynamic slots"
    ),
))

_register(_contract(
    name="CollectiveRegion",
    atlas_index=15,
    behavior_ids=("collective_region",),
    behavior_taxonomy=("CROSS_DEVICE_OVERLAP", "GLOBAL_MEMORY", "STATIC_SCHEDULING"),
    invariant=(
        "synchronized work is paced by its slowest required link and cannot "
        "hide an unmodeled transfer"
    ),
    cost_removed={
        "mechanism": "choose topology-aware collective execution",
        "metrics": ("state_movement", "synchronization", "token_ns"),
    },
    organ_classes=("moe", "attention", "multi_device"),
    preconditions=(
        "the collective algorithm is an explicit graph node",
        "every required link is modeled",
        "NCCL/RCCL are source implementations, not this primitive",
    ),
    cheapest_falsifier=(
        "a protected multi-domain or simulated-link A/B does not match the "
        "cost ordering after complete transfer accounting"
    ),
    legal_memory_tiers=_EDGE,
    physical_graph_mapping={
        "dependencies": "semantic_transport",
        "device_placement": "topology_aware",
        "synchronization": "collective_algorithm",
    },
    aliases=(),
    atlas_status="IMPLEMENTED",
    in_atlas=True,
    fundamental_physical_idea=(
        "choose ring/tree/direct movement from measured alpha/beta and message size"
    ),
))

_register(_contract(
    name="MoveOrRecompute",
    atlas_index=16,
    behavior_ids=("move_or_recompute",),
    behavior_taxonomy=("LOCAL_MEMORY", "GLOBAL_MEMORY", "CROSS_DEVICE_OVERLAP"),
    invariant=(
        "the cheapest legal way to satisfy a dependency is a physical "
        "scheduling decision"
    ),
    cost_removed={
        "mechanism": "select the lowest complete dependency cost",
        "metrics": (
            "active_bytes",
            "experiment_turnaround_ns",
            "host_ceremony",
            "state_movement",
            "synchronization",
            "token_ns",
        ),
    },
    organ_classes=("all",),
    preconditions=(
        "dependency queries are costed (not assumed from the source framework)",
        "every candidate path has capability evidence or is refused",
        "device placement is topology-aware",
    ),
    cheapest_falsifier=(
        "the costed plan disagrees with a protected end-to-end A/B or chooses "
        "a path with missing capability evidence"
    ),
    legal_memory_tiers=_ALL,
    physical_graph_mapping={
        "dependencies": "costed_dependency_queries",
        "device_placement": "topology_aware",
        "execution_policy": "measured_complete_wall",
    },
    aliases=(),
    atlas_status="IMPLEMENTED",
    in_atlas=True,
    fundamental_physical_idea=(
        "do not move an operand merely because the source framework would"
    ),
))

_register(_contract(
    name="MemoryTierIdentity",
    atlas_index=17,
    behavior_ids=("memory_tier_identity",),
    behavior_taxonomy=("LOCAL_MEMORY", "GLOBAL_MEMORY", "DATA_STATIONARITY"),
    invariant=(
        "memory tier is a component of executable identity; the same semantic "
        "program at two tiers is two physical executables"
    ),
    cost_removed={
        "mechanism": (
            "prevent a UMA plan from being silently treated as an HBM plan "
            "(or REGISTER as THREADGROUP, etc.)"
        ),
        "metrics": ("state_movement",),
    },
    organ_classes=("all",),
    preconditions=(
        "every primitive instance declares exactly one occupying tier",
        "the identity function includes that tier",
        "PhysicalGraph.execution_policy.memory_tier_is_executable_identity is true",
    ),
    cheapest_falsifier=(
        "two instances that differ only in memory tier hash to the same "
        "physical identity (this library's negative control)"
    ),
    legal_memory_tiers=_ALL,
    physical_graph_mapping={
        "memory": "tier_is_executable_identity",
        "execution_policy.memory_tier_is_executable_identity": True,
    },
    aliases=("execution_policy.memory_tier_is_executable_identity",),
    atlas_status="LIST_ONLY",
    in_atlas=True,
    fundamental_physical_idea=(
        "the occupying tier is not a hint; it is part of what the executable is"
    ),
    atlas_entry=(
        "absent as an atlas entry; PhysicalGraph already states "
        "memory_tier_is_executable_identity. This primitive IS that function."
    ),
))

# Directive listed VerificationRegion. Atlas does not. Implemented as a
# sidecar contract that maps onto PhysicalGraph.execution_policy.verification
# so we do not mint an 18th atlas primitive.
DIRECTIVE_ONLY = "VerificationRegion"

_register(_contract(
    name="VerificationRegion",
    atlas_index=None,
    behavior_ids=("verification_region",),
    behavior_taxonomy=("CONDITIONAL_EXECUTION", "STATIC_SCHEDULING"),
    invariant=(
        "a verification region may observe but must not change the accepted "
        "output of the region it checks"
    ),
    cost_removed={
        "mechanism": "observe in place; refuse a second semantic program that 'verifies' by rewriting",
        "metrics": ("state_movement",),
    },
    organ_classes=("all",),
    preconditions=(
        "PhysicalGraph.execution_policy.verification is populated",
        "the region is observational (L0/L1/L2/L3 as already named by PhysicalGraph)",
        "divergence handling is checkpoint bisection, not a silent rewrite",
    ),
    cheapest_falsifier=(
        "inserting the verification region changes the accepted output, or a "
        "diverging run cannot be bisected to a local probe"
    ),
    legal_memory_tiers=_LOCAL,
    physical_graph_mapping={
        "execution_policy.verification": {
            "fast": "L0_device_state_plus_L1_fingerprint_checkpoint",
            "protected": "L0_plus_L1_plus_L2_sampled_probes",
            "debug": "L0_plus_L1_plus_L2_plus_L3_full_state",
            "divergence": "checkpoint_bisection_then_local_deep_probe",
        }
    },
    aliases=("execution_policy.verification",),
    atlas_status="NOT_IN_ATLAS",
    in_atlas=False,
    fundamental_physical_idea=(
        "verification is a PhysicalGraph execution-policy region, not a new atlas primitive"
    ),
))

assert tuple(c for c in CONTRACTS if CONTRACTS[c]["in_atlas"]) == ATLAS_PRIMITIVES
assert DIRECTIVE_ONLY in CONTRACTS


# ---------------------------------------------------------------------------
# Backend seams. A seam is a declared interface plus an availability flag.
# METAL and FPGA are PLANNED. CUDA and ANE are UNAVAILABLE with a named
# missing dependency. Lowering to an unavailable backend raises.
# FPGA is part of Accelerator / Physical Compiler / Fusion — not a
# civilization and not "an FPGA backend".
# ---------------------------------------------------------------------------

SEAMS: dict[str, dict[str, Any]] = {
    "METAL": {
        "backend": "METAL",
        "availability": "PLANNED",
        "interface": {
            "entry": "lower_physical_graph_to_backend(graph, 'METAL')",
            "emits": "PLAN_ONLY Metal command-buffer / persistent-pipeline plan",
            "does_not_emit": "kernels, timing, or a protected measurement",
            "owned_by": "Hawking PhysicalGraph",
        },
        "missing_dependency": None,
        "claim_boundary": (
            "PLANNED is not a measurement. No Metal timing, tps, or joule "
            "figure is asserted by this sidecar."
        ),
    },
    "FPGA": {
        "backend": "FPGA",
        "availability": "PLANNED",
        "interface": {
            "entry": "lower_physical_graph_to_backend(graph, 'FPGA')",
            "emits": "PLAN_ONLY spatial-pipeline / dataflow-region plan",
            "does_not_emit": "a bitstream, a board timing, or an FPGA civilization",
            "owned_by": "Hawking Physical Compiler / Fusion (FPGA is an organ of Accelerator, not its own civilization)",
        },
        "missing_dependency": None,
        "not_a_civilization": (
            "FPGA is part of Accelerator / Physical Compiler / Fusion. This "
            "seam is a declared interface plus an availability flag. It is "
            "not an FPGA backend."
        ),
        "claim_boundary": (
            "PLANNED is not a measurement. HWIR and link/cycle simulations "
            "remain diagnostic until hardware receipts exist."
        ),
    },
    "CUDA": {
        "backend": "CUDA",
        "availability": "UNAVAILABLE",
        "interface": {
            "entry": "lower_physical_graph_to_backend(graph, 'CUDA')",
            "emits": "never; raises BackendUnavailableError",
            "owned_by": "Hawking PhysicalGraph (CUDA graphs are a source school, not this seam)",
        },
        "missing_dependency": (
            "no NVIDIA CUDA device on this Apple host. "
            "receipts/headless/CUDA_CAPABILITY_CENSUS.json identities.device.api "
            "is 'Metal via MLX'; identities.transport.status is ABSENT "
            "('no NVIDIA hardware exists on this machine'). CUDA graphs are an "
            "atlas source school, not a Hawking execution backend. Local CUDA "
            "performance claims are forbidden on Apple hardware. "
            "tools/accelerator/cuda_runtime.py is Codex-owned inventory, not this seam."
        ),
        "claim_boundary": (
            "UNAVAILABLE. This sidecar will not return a CUDA plan, estimate, "
            "or performance number."
        ),
    },
    "ANE": {
        "backend": "ANE",
        "availability": "UNAVAILABLE",
        "interface": {
            "entry": "lower_physical_graph_to_backend(graph, 'ANE')",
            "emits": "never; raises BackendUnavailableError",
            "owned_by": "Hawking PhysicalGraph (public Core ML / MLComputePlan is the only Apple authority, and it is not a runnable seam here)",
        },
        "missing_dependency": (
            "ANE lowering requires compiled MLProgram assets and "
            "MLComputePlan/runtime evidence. APPLE_ANE_ATLAS.json (parent "
            "working tree, untracked in this snapshot) status is "
            "ATLAS_SCAFFOLD_COMPILE_BOUNDARY; xcodebuild unavailable "
            "(CommandLineTools only). Graph manifests do not prove "
            "compilation, placement, latency, energy, or Flash parity."
        ),
        "claim_boundary": (
            "UNAVAILABLE. This sidecar will not return an ANE plan or latency."
        ),
    },
}


def normalize_backend(backend: str) -> str:
    name = str(backend).strip().upper()
    if name not in SEAMS:
        raise UnknownBackendError(
            f"backend {backend!r} is not a declared seam; declared: {list(SEAMS)}"
        )
    return name


def normalize_tier(tier: str) -> str:
    name = str(tier).strip().upper()
    if name not in MEMORY_TIERS:
        raise IllegalMemoryTierError(
            f"memory tier {tier!r} is not one of {list(MEMORY_TIERS)}"
        )
    return name


def contract(name: str) -> dict[str, Any]:
    if name not in CONTRACTS:
        raise UnknownPrimitiveError(
            f"{name!r} is not an atlas primitive or the directive VerificationRegion"
        )
    return CONTRACTS[name]


def physical_identity(
    *,
    semantic_program_id: str,
    primitive: str,
    memory_tier: str,
    backend: str,
    organ_class: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Identity of one physical executable.

    Memory tier is a required component. Two instances of the same semantic
    program at UMA and at HBM MUST hash differently. Backend is also a
    component: a METAL UMA instance is not a CUDA UMA instance.
    """
    spec = contract(primitive)
    tier = normalize_tier(memory_tier)
    back = normalize_backend(backend)
    if tier not in spec["legal_memory_tiers"]:
        raise IllegalMemoryTierError(
            f"{primitive} cannot occupy {tier}; legal={list(spec['legal_memory_tiers'])}"
        )
    body = {
        "schema": "hawking.future.physical_identity.v1",
        "semantic_program_id": semantic_program_id,
        "primitive": primitive,
        "memory_tier": tier,
        "backend": back,
        "organ_class": organ_class,
        "extra": dict(extra) if extra else {},
    }
    return _sha(body)


@dataclass(frozen=True)
class PrimitiveInstance:
    primitive: str
    memory_tier: str
    backend: str
    semantic_program_id: str
    organ_class: str | None
    identity: str
    extra: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        spec = contract(self.primitive)
        return {
            "primitive": self.primitive,
            "memory_tier": self.memory_tier,
            "backend": self.backend,
            "semantic_program_id": self.semantic_program_id,
            "organ_class": self.organ_class,
            "identity": self.identity,
            "in_atlas": spec["in_atlas"],
            "invariant": spec["invariant"],
            "physical_graph_mapping": spec["physical_graph_mapping"],
            "extra": self.extra,
        }


def instantiate(
    primitive: str,
    *,
    memory_tier: str,
    semantic_program_id: str,
    backend: str = "METAL",
    organ_class: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> PrimitiveInstance:
    spec = contract(primitive)
    tier = normalize_tier(memory_tier)
    back = normalize_backend(backend)
    if organ_class is not None:
        allowed = spec["organ_classes"]
        if "all" not in allowed and organ_class not in allowed:
            raise PrimitiveError(
                f"{primitive} does not apply to organ class {organ_class!r}; "
                f"applies to {list(allowed)}"
            )
    ident = physical_identity(
        semantic_program_id=semantic_program_id,
        primitive=primitive,
        memory_tier=tier,
        backend=back,
        organ_class=organ_class,
        extra=extra,
    )
    return PrimitiveInstance(
        primitive=primitive,
        memory_tier=tier,
        backend=back,
        semantic_program_id=semantic_program_id,
        organ_class=organ_class,
        identity=ident,
        extra=dict(extra) if extra else {},
    )


# NR organ families recovered from FLASH_COMPLETE_V2.nr.json representation.parts.
# Mapping is a planning hypothesis: which atlas primitives each family needs.
# It is not a claim that Flash currently executes any of them.
FAMILY_PRIMITIVES: dict[str, tuple[str, ...]] = {
    "embedding_lm_head": ("StationaryRepresentation", "TiledProjection"),
    "ngram_embedding": ("StationaryRepresentation", "SparseSkip"),
    "norm": ("ConditionalPhysicalProgram",),
    "linear_attention_hyperconnection": ("LocalStateMachine", "PersistentPhysicalRegion"),
    "full_attention": ("LocalStateMachine", "PersistentPhysicalRegion"),
    "mlp_hyperconnection": (
        "StationaryRepresentation",
        "TiledProjection",
        "FusedDecodeCompute",
    ),
    "shared_expert": (
        "StationaryRepresentation",
        "TiledProjection",
        "FusedDecodeCompute",
    ),
    "routed_experts": (
        "DirectRoutedAccumulate",
        "SparseSkip",
        "StationaryRepresentation",
    ),
    "other": ("MoveOrRecompute",),
}

GRAPH_PRIMITIVES: tuple[str, ...] = (
    "GraphReplay",
    "SemanticTransportEdge",
    "MoveOrRecompute",
    "MemoryTierIdentity",
    "ConditionalPhysicalProgram",
    "VerificationRegion",
)

# Fixture used when FLASH_COMPLETE_V2.nr.json is not materialized in this
# worktree. Family names and schema are recovered from that receipt; numeric
# payloads (parameter counts, lookup ns, cosine) are deliberately omitted.
FLASH_NR_FIXTURE: dict[str, Any] = {
    "schema": "hawking.flash.complete_nr.v2",
    "artifact_kind": "NR",
    "status": "COMPLETE_HETEROGENEOUS_CANDIDATE_NOT_FOR_PROMOTION",
    "semantic_provenance": {"parent_model": "Qwen/Qwen3.8-Flash-Next"},
    "representation": {
        "scope": "complete 48-layer Flash model (family names recovered; payloads omitted)",
        "parts": [{"family": family, "runtime_required": True} for family in FAMILY_PRIMITIVES],
    },
    "promotion": {"allowed": False},
    "claim_boundary": (
        "Fixture recovered from FLASH_COMPLETE_V2.nr.json. It is not a compact "
        "complete representation, not a capability claim, and not a machine execution."
    ),
}


def load_nr(path: str | Path | None = None) -> dict[str, Any]:
    """Load a real NR if present; otherwise the recovered Flash family fixture."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    candidates.append(REPO / FLASH_NR_REL)
    for p in candidates:
        if p.is_file():
            doc = load_json(p)
            doc["_loaded_from"] = str(p.relative_to(REPO) if p.is_relative_to(REPO) else p)
            return doc
    fixture = json.loads(_canon(FLASH_NR_FIXTURE))
    fixture["_loaded_from"] = "embedded FLASH_NR_FIXTURE (FLASH_COMPLETE_V2.nr.json not in this worktree)"
    return fixture


def load_atlas() -> dict[str, Any] | None:
    p = REPO / ATLAS_REL
    if not p.is_file():
        return None
    return load_json(p)


def _nr_families(nr: Mapping[str, Any]) -> list[str]:
    parts = (nr.get("representation") or {}).get("parts") or []
    families: list[str] = []
    for part in parts:
        if isinstance(part, Mapping) and part.get("family"):
            families.append(str(part["family"]))
    return families


def lower_nr_to_physical_graph(
    nr: Mapping[str, Any] | None = None,
    *,
    memory_tier: str = "UMA",
    backend: str = "METAL",
    semantic_program_id: str | None = None,
) -> dict[str, Any]:
    """Lower an NR into a PhysicalGraph-shaped PLAN_ONLY document.

    Does not import hcli (not in this sparse checkout). Emits the field
    names PhysicalGraph.to_dict() already uses so a downstream module can
    feed this to compile_physical_graph / apply_architecture_atlas.
    """
    if nr is None:
        nr = load_nr()
    tier = normalize_tier(memory_tier)
    back = normalize_backend(backend)
    provenance = nr.get("semantic_provenance") if isinstance(nr.get("semantic_provenance"), Mapping) else {}
    model_id = str(
        semantic_program_id
        or provenance.get("parent_model")
        or nr.get("model_id")
        or "unknown"
    )
    families = _nr_families(nr) or list(FAMILY_PRIMITIVES)
    instances: list[PrimitiveInstance] = []
    computation: list[dict[str, Any]] = []
    data: list[dict[str, Any]] = []
    for family in families:
        prims = FAMILY_PRIMITIVES.get(family, ("MoveOrRecompute",))
        organ = family
        for prim in prims:
            spec = contract(prim)
            # organ_class check: family names are not always atlas organ
            # classes. Bind organ_class only when it is in the contract.
            bound = organ if (organ in spec["organ_classes"] or "all" in spec["organ_classes"]) else None
            inst = instantiate(
                prim,
                memory_tier=tier,
                semantic_program_id=model_id,
                backend=back,
                organ_class=bound,
                extra={"nr_family": family},
            )
            instances.append(inst)
            computation.append({
                "id": f"{family}:{prim}",
                "kind": "computation",
                "primitive": prim,
                "nr_family": family,
                "memory_tier": tier,
                "identity": inst.identity,
                "present": True,
            })
            data.append({
                "id": f"{family}:{prim}",
                "kind": "tensor_group",
                "bytes": None,
                "active_bytes_per_token": None,
                "source": "NR family lowering; size unresolved (STATIC_ONLY)",
                "memory_tier": tier,
            })
    for prim in GRAPH_PRIMITIVES:
        inst = instantiate(
            prim,
            memory_tier=tier if tier in contract(prim)["legal_memory_tiers"] else contract(prim)["legal_memory_tiers"][0],
            semantic_program_id=model_id,
            backend=back,
            extra={"scope": "whole_graph"},
        )
        instances.append(inst)
        computation.append({
            "id": f"graph:{prim}",
            "kind": "computation",
            "primitive": prim,
            "nr_family": None,
            "memory_tier": inst.memory_tier,
            "identity": inst.identity,
            "present": True,
        })

    identities = sorted({i.identity for i in instances})
    graph = {
        "schema": PHYSICAL_GRAPH_SCHEMA,
        "semantic_type": "PhysicalGraphPlan",
        "compiler_stage": "PhysicalPrimitivesLowering",
        "model_id": model_id,
        "qualification": "PLAN_ONLY",
        "computation": computation,
        "data": data,
        "representation": {
            "nr_schema": nr.get("schema"),
            "nr_status": nr.get("status"),
            "nr_loaded_from": nr.get("_loaded_from"),
            "native_representation_verified": False,
            "memory_tier_is_executable_identity": True,
            "atlas_primitives": list(ATLAS_PRIMITIVES),
            "nr_operator_vocabulary_not_merged": list(NR_OPERATOR_VOCABULARY),
        },
        "memory": [
            {
                "tier": tier,
                "role": "executable identity component",
                "status": "planned",
            }
        ],
        "residency": {"weights": "unresolved", "state": "unresolved", "page_cache": "unresolved"},
        "state": {"kv_cache": "unresolved", "recurrent_state": "unresolved"},
        "precision": {"weight": "unresolved", "activation": "unresolved", "accumulator": "unresolved"},
        "dependencies": [
            {"from": family, "to": "next", "kind": "dataflow"} for family in families
        ],
        "device_placement": {
            "candidates": ["METAL", "FPGA"],
            "unavailable": ["CUDA", "ANE"],
            "selected": None,
            "requested_backend": back,
            "primitive_realizations": {
                inst.primitive: contract(inst.primitive)["physical_graph_mapping"]
                for inst in instances
            },
        },
        "synchronization": [{"kind": "runtime_boundary", "status": "unresolved"}],
        "evidence": [
            {
                "kind": "physical_primitives_lowering",
                "claim": "hypothesis projection only; not physical performance evidence",
                "atlas_fingerprint": ATLAS_FINGERPRINT,
            }
        ],
        "execution_policy": {
            "process": "long_lived_executor",
            "pipeline_state": "compile_once_reuse",
            "memory_tier_is_executable_identity": True,
            "dynamic_slots": ["token", "position", "route", "sampling", "variable_state"],
            "verification": contract("VerificationRegion")["physical_graph_mapping"][
                "execution_policy.verification"
            ],
            "promotion_metric": "measured_complete_useful_work",
            "device_count_is_not_speed_authority": True,
        },
        "primitive_instances": [inst.to_dict() for inst in instances],
        "instance_identities": identities,
        "nr_families": families,
        "memory_tier": tier,
        "backend_requested": back,
    }
    graph["fingerprint"] = _sha(
        {k: v for k, v in graph.items() if k != "fingerprint"}
    )
    return graph


def lower_physical_graph_to_backend(
    graph: Mapping[str, Any],
    backend: str,
) -> dict[str, Any]:
    """Lower a PhysicalGraph-shaped plan onto one backend seam.

    UNAVAILABLE seams raise. PLANNED seams return a PLAN_ONLY document.
    Never a hardware number.
    """
    back = normalize_backend(backend)
    seam = SEAMS[back]
    if seam["availability"] == "UNAVAILABLE":
        raise BackendUnavailableError(
            f"lowering to {back} is refused: {seam['missing_dependency']}"
        )
    instances = list(graph.get("primitive_instances") or [])
    realizations = []
    for inst in instances:
        if not isinstance(inst, Mapping):
            continue
        realizations.append({
            "primitive": inst.get("primitive"),
            "memory_tier": inst.get("memory_tier"),
            "identity": inst.get("identity"),
            "physical_graph_mapping": inst.get("physical_graph_mapping"),
            "seam": back,
            "availability": seam["availability"],
        })
    plan = {
        "schema": LOWERING_SCHEMA,
        "backend": back,
        "availability": seam["availability"],
        "qualification": "PLAN_ONLY",
        "measurement_state": "STATIC_ONLY",
        "gpu_authority": False,
        "physical_graph_fingerprint": graph.get("fingerprint"),
        "physical_graph_schema": graph.get("schema"),
        "interface": seam["interface"],
        "claim_boundary": seam["claim_boundary"],
        "primitive_realizations": realizations,
        "missing_dependency": seam.get("missing_dependency"),
    }
    if back == "FPGA":
        plan["not_a_civilization"] = seam["not_a_civilization"]
    plan["fingerprint"] = _sha({k: v for k, v in plan.items() if k != "fingerprint"})
    return plan


def lower_nr_to_backend(
    nr: Mapping[str, Any] | None = None,
    *,
    backend: str,
    memory_tier: str = "UMA",
) -> dict[str, Any]:
    """NR -> PhysicalGraph -> backend. CUDA/ANE raise at the seam."""
    graph = lower_nr_to_physical_graph(nr, memory_tier=memory_tier, backend=backend)
    return lower_physical_graph_to_backend(graph, backend)


def reconciliation() -> dict[str, Any]:
    """For each of the 17 atlas primitives: implemented / merged / left out."""
    rows: dict[str, Any] = {}
    for name in ATLAS_PRIMITIVES:
        spec = CONTRACTS[name]
        disposition = "IMPLEMENTED"
        reason = "atlas list name given an executable contract"
        if spec["atlas_status"] == "LIST_ONLY":
            reason = spec.get("atlas_entry") or (
                "atlas list name; no atlas entry. Contract derived from taxonomy "
                "and PhysicalGraph, not invented as a parallel primitive."
            )
        if name == "ConditionalPhysicalProgram":
            disposition = "IMPLEMENTED"
            reason = (
                "atlas used this name for two entries (static_dynamic_skeleton "
                "IMPLEMENTED and npu_regular_island MAPPED). Merged into one "
                "contract with two behavior_ids so the name is not duplicated."
            )
        if name == "LayoutTransform":
            reason = (
                "atlas name kept. The directive omitted it. Not merged into "
                "TiledProjection: layout algebra chooses packing; tiled "
                "projection is the tiled operator."
            )
        if name == "AsyncPrefetch":
            reason = (
                "atlas list name kept. No atlas entry (technique coverage folds "
                "prefetch into async_double_buffer). Not merged into "
                "DoubleBufferedTile: prefetch is one-sided overlap; "
                "double-buffer is two-buffer ownership."
            )
        if name == "TiledProjection":
            reason = (
                "atlas list name kept. No atlas entry (technique coverage files "
                "tiled GEMV/GEMM under layout_algebra). Not merged into "
                "LayoutTransform."
            )
        if name == "MemoryTierIdentity":
            reason = (
                "atlas list name kept and also applied as a field on every "
                "other primitive. The primitive IS the identity function. "
                "PhysicalGraph already states memory_tier_is_executable_identity; "
                "this module does not fork a second identity scheme."
            )
        rows[name] = {
            "disposition": disposition,
            "atlas_status": spec["atlas_status"],
            "in_atlas": True,
            "behavior_ids": list(spec["behavior_ids"]),
            "aliases": list(spec["aliases"]),
            "reason": reason,
        }
    return {
        "atlas_count": len(ATLAS_PRIMITIVES),
        "implemented": [n for n, r in rows.items() if r["disposition"] == "IMPLEMENTED"],
        "merged_into_another": ["npu_regular_island -> ConditionalPhysicalProgram"],
        "left_out": [],
        "per_primitive": rows,
        "directive_only": {
            DIRECTIVE_ONLY: {
                "disposition": "IMPLEMENTED_OUTSIDE_ATLAS_SEVENTEEN",
                "in_atlas": False,
                "reason": (
                    "directive listed VerificationRegion; atlas does not. "
                    "Implemented as a lowering of PhysicalGraph.execution_policy."
                    "verification rather than minting an 18th atlas primitive."
                ),
                "aliases": list(CONTRACTS[DIRECTIVE_ONLY]["aliases"]),
            }
        },
        "not_merged_with": {
            "PhysicalGraph.NR_PRIMITIVES": (
                "BASIS_PROJECT / COEFFICIENT_APPLY / ... are representation "
                "operators for Gravity/NR lowering, a different vocabulary "
                "from the 17 architecture-atlas primitives."
            ),
            "fusion_isa.FusionOp.PREFETCH": (
                "a timeline hint in the Fusion ISA, not AsyncPrefetch."
            ),
        },
    }


def recovered_implementation() -> dict[str, Any]:
    atlas_on_disk = (REPO / ATLAS_REL).is_file()
    flash_on_disk = (REPO / FLASH_NR_REL).is_file()
    cuda_census_on_disk = (REPO / CUDA_CENSUS_REL).is_file()
    return {
        "atlas": {
            "path": ATLAS_REL,
            "schema": ATLAS_SCHEMA,
            "fingerprint": ATLAS_FINGERPRINT,
            "present_in_this_worktree": atlas_on_disk,
            "note": (
                "Untracked in the parent working tree (~91 KB, 17 primitives, "
                "21-behavior taxonomy, 15 entries, 15 HWIR hypotheses). Not "
                "in this sparse snapshot's git tree. Primitive names and "
                "entry contracts were recovered from that file and embedded."
            ),
            "backend_neutral_primitives": list(ATLAS_PRIMITIVES),
            "behavior_taxonomy": list(BEHAVIOR_TAXONOMY),
            "n_entries": 15,
            "entries_missing_for": [
                "AsyncPrefetch",
                "TiledProjection",
                "MemoryTierIdentity",
            ],
            "duplicate_entry_name": (
                "ConditionalPhysicalProgram used twice "
                "(static_dynamic_skeleton, npu_regular_island)"
            ),
        },
        "physical_graph": {
            "path": "hcli/physical_graph.py",
            "present_in_this_worktree": (REPO / "hcli" / "physical_graph.py").is_file(),
            "committed_schema": PHYSICAL_GRAPH_SCHEMA,
            "committed_role": (
                "provider-neutral PhysicalGraph planning boundary; "
                "compile_physical_graph from ArchitectureRecognizer organs"
            ),
            "parent_working_tree_additions": (
                "apply_architecture_atlas, score_physical_candidates, "
                "execution_policy.memory_tier_is_executable_identity, "
                "NR_PRIMITIVES representation-operator vocabulary, "
                "verification L0-L3 policy. Not imported here (hcli is "
                "Codex-owned and not in this sparse checkout)."
            ),
        },
        "physical_graph_compiler": {
            "path": "tools/odyssey/physical_graph_compiler.py",
            "present_in_this_worktree": False,
            "role": "source graph -> OrganGraph -> PhysicalOperatorGraph collapses",
        },
        "noetic_compiler": {
            "path": "tools/odyssey/noetic_compiler.py",
            "present_in_this_worktree": False,
            "role": (
                "Foreign Model -> ArchitectureRecognizer -> OrganGraph -> "
                "Doctor -> RepresentationPlanner -> PhysicalGraphCompiler -> "
                "KernelPlanner -> DeviceCompiler -> NoeticExecutable"
            ),
        },
        "flash_nr": {
            "path": FLASH_NR_REL,
            "present_in_this_worktree": flash_on_disk,
            "schema": "hawking.flash.complete_nr.v2",
            "status": "COMPLETE_HETEROGENEOUS_CANDIDATE_NOT_FOR_PROMOTION",
            "families": list(FAMILY_PRIMITIVES),
            "note": (
                "Untracked in the parent working tree. Family names recovered "
                "and embedded as FLASH_NR_FIXTURE; numeric payloads omitted."
            ),
        },
        "cuda_census": {
            "path": CUDA_CENSUS_REL,
            "present_in_this_worktree": cuda_census_on_disk,
            "used_for": "named missing dependency of the CUDA seam",
        },
        "fusion_isa": {
            "path": "tools/accelerator/fusion_isa.py",
            "present_in_this_worktree": False,
            "role": "14 Fusion commands including PREFETCH; different layer, not forked",
        },
        "existing_sidecar": {
            "physical_primitives_py": False,
            "note": "no tools/future/physical_primitives.py existed; this module is new",
        },
    }


def gaps_closed() -> list[str]:
    return [
        "executable contracts for all 17 atlas primitives (invariant, cost removed, organ classes, preconditions, cheapest falsifier, legal memory tiers)",
        "memory-tier identity function: same semantic program at UMA vs HBM hashes differently",
        "backend lowering seams METAL/FPGA PLANNED, CUDA/ANE UNAVAILABLE with named missing dependencies; unavailable lowering raises",
        "NR family -> PhysicalGraph-shaped PLAN_ONLY document (no hcli import)",
        "reconciliation of atlas 17 vs directive list (LayoutTransform kept; VerificationRegion not added to the 17)",
        "ConditionalPhysicalProgram's two atlas entries merged into one contract",
    ]


def negative_findings(atlas: Mapping[str, Any] | None) -> list[str]:
    findings = [
        "ACCELERATOR_ARCHITECTURE_ATLAS.json is not in this worktree (untracked in the parent); names/contracts were recovered and embedded rather than live-read at runtime"
        if atlas is None
        else "atlas present in this worktree and live-reconciled",
        "FLASH_COMPLETE_V2.nr.json is not in this worktree; family names recovered into FLASH_NR_FIXTURE, numeric payloads omitted",
        "hcli/physical_graph.py is not in this sparse checkout; lowering emits a compatible shape but cannot call compile_physical_graph",
        "tools/odyssey/physical_graph_compiler.py and noetic_compiler.py are not in this sparse checkout (read via git show)",
        "APPLE_ANE_ATLAS.json is not in this worktree; ANE missing-dependency text recovered from the parent working tree",
        "this sidecar has no GPU, no FPGA board, no ANE MLComputePlan; METAL/FPGA seams are PLANNED, never measured",
        "atlas entries for AsyncPrefetch, TiledProjection, MemoryTierIdentity do not exist; contracts for those three are derived",
        "VerificationRegion is not an atlas primitive",
        "cannot prove a Metal/FPGA plan is faster than anything; STATIC_ONLY only",
    ]
    return findings


def _public_contract(spec: dict[str, Any]) -> dict[str, Any]:
    """Receipt form: tuples become lists; no hardware numbers."""
    out = {}
    for k, v in spec.items():
        if isinstance(v, tuple):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def build() -> Path:
    atlas = load_atlas()
    atlas_match: dict[str, Any] = {"present": atlas is not None}
    if atlas is not None:
        listed = tuple(atlas.get("backend_neutral_primitives") or ())
        atlas_match["names_match"] = listed == ATLAS_PRIMITIVES
        atlas_match["fingerprint_match"] = atlas.get("fingerprint") == ATLAS_FINGERPRINT
        if listed != ATLAS_PRIMITIVES:
            raise PrimitiveError(
                "embedded ATLAS_PRIMITIVES drifted from disk atlas: "
                f"embedded={list(ATLAS_PRIMITIVES)} disk={list(listed)}"
            )

    graph = lower_nr_to_physical_graph(memory_tier="UMA", backend="METAL")
    metal = lower_physical_graph_to_backend(graph, "METAL")
    fpga = lower_physical_graph_to_backend(graph, "FPGA")

    uma = instantiate(
        "StationaryRepresentation",
        memory_tier="UMA",
        semantic_program_id="identity-negative-control",
        backend="METAL",
        organ_class="mlp",
    )
    hbm = instantiate(
        "StationaryRepresentation",
        memory_tier="HBM",
        semantic_program_id="identity-negative-control",
        backend="METAL",
        organ_class="mlp",
    )
    cuda_raised = False
    cuda_error = None
    try:
        lower_physical_graph_to_backend(graph, "CUDA")
    except BackendUnavailableError as exc:
        cuda_raised = True
        cuda_error = str(exc)

    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Executable contracts for the 17 atlas backend-neutral primitives, "
            "with memory-tier identity, connecting NR -> PhysicalGraph -> "
            "backend lowering. CUDA graphs / TPU compiled graphs / FPGA "
            "spatial pipelines / deterministic dataflow collapse into one "
            "Hawking-owned abstraction."
        ),
        "atlas_source": {
            "schema": ATLAS_SCHEMA,
            "fingerprint": ATLAS_FINGERPRINT,
            "path": ATLAS_REL,
            "live_disk": atlas_match,
        },
        "primitives": [_public_contract(CONTRACTS[n]) for n in ATLAS_PRIMITIVES],
        "directive_only_contracts": [_public_contract(CONTRACTS[DIRECTIVE_ONLY])],
        "memory_tiers": list(MEMORY_TIERS),
        "identity_rule": {
            "function": "physical_identity(semantic_program_id, primitive, memory_tier, backend, organ_class=..., extra=...)",
            "memory_tier_is_component": True,
            "negative_control": {
                "semantic_program_id": "identity-negative-control",
                "primitive": "StationaryRepresentation",
                "uma_identity": uma.identity,
                "hbm_identity": hbm.identity,
                "identities_differ": uma.identity != hbm.identity,
            },
        },
        "backend_seams": {name: dict(SEAMS[name]) for name in BACKENDS},
        "nr_to_physical_graph": {
            "entry": "lower_nr_to_physical_graph(nr, memory_tier=..., backend=...)",
            "example_fingerprint": graph["fingerprint"],
            "example_model_id": graph["model_id"],
            "example_nr_families": graph["nr_families"],
            "example_memory_tier": graph["memory_tier"],
            "n_instances": len(graph["primitive_instances"]),
            "qualification": graph["qualification"],
        },
        "backend_lowering": {
            "entry": "lower_physical_graph_to_backend(graph, backend)",
            "metal": {
                "availability": metal["availability"],
                "qualification": metal["qualification"],
                "fingerprint": metal["fingerprint"],
            },
            "fpga": {
                "availability": fpga["availability"],
                "qualification": fpga["qualification"],
                "fingerprint": fpga["fingerprint"],
                "not_a_civilization": fpga["not_a_civilization"],
            },
            "cuda_raises": cuda_raised,
            "cuda_error_prefix": (cuda_error or "")[:240],
            "ane_raises": True,
        },
        "reconciliation": reconciliation(),
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(atlas),
        "vocabulary": {
            "eras": ["I Genesis of the Laboratory", "II Compounding Civilization",
                     "III Autonomous Science Civilization", "IV Synthetic Machine Civilization",
                     "V Released Hawking Civilization"],
            "odysseys": ["I WHAT IS TRUE?", "II WHAT DID HAWKING ALREADY LEARN?",
                         "III WHERE IS HAWKING WRONG?"],
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "fpga_is_not_a_civilization": True,
            "disk_state_is_authority": True,
        },
    }
    if uma.identity == hbm.identity:
        raise PrimitiveError("identity negative control failed: UMA == HBM")
    if not cuda_raised:
        raise PrimitiveError("CUDA seam did not raise")
    return write_receipt(RECEIPT, doc, "tools/future/physical_primitives.py")


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    out = selftest() if (a.selftest or not a.build) else build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
