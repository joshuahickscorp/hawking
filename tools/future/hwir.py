"""HWIR v1 — hardware intermediate representation.

The FPGA/spatial school compiles into this IR. It consumes Noetic/PhysicalGraph
semantics and real FPGA organ-map receipts. A graph that assumes it is
multiplying original dense source weight matrices is invalid by construction.

This is not an FPGA backend, HDL emitter, or bitstream path.

    python3 tools/future/hwir.py --selftest
    python3 tools/future/hwir.py --build
    python3 tools/future/hwir.py --lower receipts/headless/FLASH_NEXT_FPGA_ORGAN_MAP.json --organ expert_bank
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, git, sha256_file

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "hawking.future.hwir.v1"
VERSION = 1
RECEIPT = "HWIR_V1.json"

CANON_DUMP = {"sort_keys": True, "separators": (",", ":"), "ensure_ascii": True}

# Contract node kinds. Atlas hypotheses only distinguish dataflow_region vs
# spatial_region; those coarser labels are consumed, not re-derived.
NODE_KINDS = (
    "compute",
    "state",
    "memory",
    "representation-decoder",
    "reduction",
    "dma-transport",
    "persistent-pipeline",
)

FRAME_KINDS = (
    "activation",
    "partial_reduction",
    "compact_representation_fragment",
    "state",
    "codebook_id",
    "sparse_residual",
)

TRANSFORMS = (
    "identity",
    "quantize",
    "transpose",
    "reduce",
    "checksum_digest",
    "pack",
    "unpack",
)

RESOURCE_CLASSES = ("BRAM", "DSP", "LUT", "URAM")

KIND_ALIASES = {
    "compute": "compute",
    "state": "state",
    "memory": "memory",
    "representation-decoder": "representation-decoder",
    "representation_decoder": "representation-decoder",
    "reduction": "reduction",
    "dma-transport": "dma-transport",
    "dma_transport": "dma-transport",
    "dma/transport": "dma-transport",
    "DMA/transport": "dma-transport",
    "persistent-pipeline": "persistent-pipeline",
    "persistent_pipeline": "persistent-pipeline",
}

FRAME_ALIASES = {
    "activation": "activation",
    "partial_reduction": "partial_reduction",
    "partial reduction": "partial_reduction",
    "compact_representation_fragment": "compact_representation_fragment",
    "compact representation fragment": "compact_representation_fragment",
    "compact_representation": "compact_representation_fragment",
    "state": "state",
    "codebook_id": "codebook_id",
    "codebook id": "codebook_id",
    "codebook ID": "codebook_id",
    "sparse_residual": "sparse_residual",
    "sparse residual": "sparse_residual",
}

TRANSFORM_ALIASES = {
    "identity": "identity",
    "quantize": "quantize",
    "transpose": "transpose",
    "reduce": "reduce",
    "checksum_digest": "checksum_digest",
    "checksum/digest": "checksum_digest",
    "pack": "pack",
    "unpack": "unpack",
}

# Atlas primitives (17) refined onto the seven IR node kinds.
PRIMITIVE_TO_NODE_KIND = {
    "PersistentPhysicalRegion": "persistent-pipeline",
    "StationaryRepresentation": "memory",
    "AsyncPrefetch": "dma-transport",
    "DoubleBufferedTile": "memory",
    "SpatialPipeline": "persistent-pipeline",
    "FusedDecodeCompute": "representation-decoder",
    "DirectRoutedAccumulate": "compute",
    "LocalStateMachine": "state",
    "SemanticTransportEdge": "dma-transport",
    "TiledProjection": "compute",
    "LayoutTransform": "compute",
    "SparseSkip": "compute",
    "ConditionalPhysicalProgram": "compute",
    "GraphReplay": "persistent-pipeline",
    "CollectiveRegion": "reduction",
    "MoveOrRecompute": "compute",
    "MemoryTierIdentity": "memory",
}

FORBIDDEN_PRIMITIVES = frozenset(
    {
        "DenseMatmul",
        "DenseSourceMatmul",
        "SourceWeightGEMM",
        "RematerializeDenseWeights",
        "UnpackSourceDense",
        "SourceDenseGEMM",
    }
)

FORBIDDEN_SEMANTICS = frozenset(
    {
        "source_tensor_identity",
        "dense_weight_matmul",
        "rematerialize_dense_source",
        "source_dense_weight",
    }
)

# PhysicalGraph field names HWIR consumes (hcli/physical_graph.py, PLAN_ONLY).
PHYSICAL_GRAPH_FIELDS = (
    "computation",
    "data",
    "representation",
    "memory",
    "residency",
    "state",
    "precision",
    "dependencies",
    "device_placement",
    "synchronization",
    "qualification",
)

FLASH_ORGAN_MAP = "receipts/headless/FLASH_NEXT_FPGA_ORGAN_MAP.json"
QWEN_ORGAN_MAP = "receipts/headless/QWEN27_FPGA_ORGAN_MAP.json"
ATLAS_REL = "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json"


def canon_dumps(obj: Any) -> str:
    return json.dumps(obj, **CANON_DUMP)


def canon_kind(kind: str) -> str:
    k = str(kind).strip()
    return KIND_ALIASES.get(k) or KIND_ALIASES.get(k.lower().replace("_", "-")) or k


def canon_frame(frame: str) -> str:
    f = str(frame).strip()
    return FRAME_ALIASES.get(f) or FRAME_ALIASES.get(f.lower()) or f


def canon_transform(transform: str) -> str:
    t = str(transform).strip()
    return TRANSFORM_ALIASES.get(t) or TRANSFORM_ALIASES.get(t.lower()) or t


def apply_transform(frame: str, transform: str) -> str | None:
    """Return the post-transform frame, or None if the transform is illegal on `frame`."""
    f = canon_frame(frame)
    t = canon_transform(transform)
    if t in {"identity", "quantize", "transpose", "checksum_digest"}:
        return f
    if t == "reduce":
        if f in {"activation", "partial_reduction"}:
            return "partial_reduction"
        return None
    if t == "pack":
        if f == "activation":
            return "compact_representation_fragment"
        return None
    if t == "unpack":
        if f in {"compact_representation_fragment", "codebook_id"}:
            return "activation"
        return None
    return None


def _zero_resources() -> dict[str, int]:
    return {k: 0 for k in RESOURCE_CLASSES}


def _norm_text(value: Any) -> str:
    return str(value or "").lower().replace("-", " ").replace("_", " ")


def _claims_dense_source(text: Any) -> bool:
    """True only for an affirmative dense-source claim, not its prohibition."""
    t = _norm_text(text)
    if not t:
        return False
    # Strip prohibition phrasing so "not source-dense weights" cannot match.
    for prohibition in (
        "no dense rematerialization",
        "never dense",
        "not source dense",
        "not source tensor",
        "rather than matrix gemv",
        "rather than matrix gemm",
        "no weight body",
    ):
        t = t.replace(prohibition, " ")
    needles = (
        "original dense weight",
        "dense weight matrix",
        "dense rematerialization",
        "materialize the original dense",
        "materialize dense",
        "source tensor identity",
        "multiply the original dense",
        "unpacked source weight",
        "source dense weight",
        "transfer dense weight body",
    )
    return any(n in t for n in needles)


# ---------------------------------------------------------------------------
# Recovered atlas snapshot. ACCELERATOR_ARCHITECTURE_ATLAS.json is not in this
# worktree HEAD; hypotheses were read from the parent checkout and are consumed
# here rather than re-derived.
# ---------------------------------------------------------------------------

_COMMON_BUFFERS = [
    "partial_reduction",
    "persistent_state",
    "resident_representation",
    "token_activation",
]
_COMMON_SEMANTIC_EDGES = [
    "activation",
    "compact_representation",
    "partial_reduction",
    "state",
]
_ATLAS_LABEL = "[D] hypothesis; no board or hardware timing claim"

_ATLAS_ROWS: tuple[tuple[str, str, str, dict[str, Any]], ...] = (
    (
        "move_or_recompute",
        "dataflow_region",
        "MoveOrRecompute",
        {
            "dependencies": "costed_dependency_queries",
            "device_placement": "topology_aware",
            "execution_policy": "measured_complete_wall",
        },
    ),
    (
        "persistent_physical_region",
        "dataflow_region",
        "PersistentPhysicalRegion",
        {
            "execution_policy.process": "long_lived_executor",
            "residency.state": "sequence",
            "residency.weights": "resident",
        },
    ),
    (
        "fused_decode_compute",
        "spatial_region",
        "FusedDecodeCompute",
        {
            "computation": "projection_plus_decode",
            "memory": "no_dense_rematerialization",
            "representation": "native_decode",
        },
    ),
    (
        "local_state_machine",
        "spatial_region",
        "LocalStateMachine",
        {
            "execution_policy": "fixed_state_transitions",
            "state": "authoritative_resident_owner",
            "synchronization": "state_update_edges",
        },
    ),
    (
        "graph_replay",
        "dataflow_region",
        "GraphReplay",
        {
            "execution_policy.dynamic_slots": ["token", "position", "route", "sampling"],
            "execution_policy.pipeline_state": "compile_once_reuse",
        },
    ),
    (
        "layout_algebra",
        "dataflow_region",
        "LayoutTransform",
        {
            "computation": "tile_and_lane_mapping",
            "precision": "representation_grouping",
            "representation": "layout_algebra",
        },
    ),
    (
        "static_dynamic_skeleton",
        "dataflow_region",
        "ConditionalPhysicalProgram",
        {
            "dependencies": "precomputed",
            "execution_policy": "static_skeleton_plus_dynamic_slots",
            "synchronization": "precomputed_where_safe",
        },
    ),
    (
        "stationary_representation",
        "dataflow_region",
        "StationaryRepresentation",
        {
            "memory": "tier_is_executable_identity",
            "representation": "packed_native",
            "residency": "stationarity_contract",
        },
    ),
    (
        "semantic_transport",
        "dataflow_region",
        "SemanticTransportEdge",
        {
            "dependencies": "typed_transport_edges",
            "device_placement": "topology_aware",
            "synchronization": "edge_ownership_and_order",
        },
    ),
    (
        "collective_region",
        "dataflow_region",
        "CollectiveRegion",
        {
            "dependencies": "semantic_transport",
            "device_placement": "topology_aware",
            "synchronization": "collective_algorithm",
        },
    ),
    (
        "async_double_buffer",
        "dataflow_region",
        "DoubleBufferedTile",
        {
            "execution_policy": "overlap_when_measured",
            "memory": "double_buffered_tiles",
            "synchronization": "producer_consumer_fences",
        },
    ),
    (
        "spatial_local_pipeline",
        "spatial_region",
        "SpatialPipeline",
        {
            "computation": "spatial_regions",
            "data": "semantic_edges",
            "memory": "local_intermediates",
        },
    ),
    (
        "direct_routed_accumulate",
        "dataflow_region",
        "DirectRoutedAccumulate",
        {
            "computation": "route_then_native_expert",
            "data": "selected_payload_only",
            "state": "route_metadata_resident",
        },
    ),
    (
        "sparse_conditional_execution",
        "dataflow_region",
        "SparseSkip",
        {
            "computation": "conditional_regions",
            "data": "sparse_indices_and_payloads",
            "qualification": "parity_required",
        },
    ),
    (
        "npu_regular_island",
        "dataflow_region",
        "ConditionalPhysicalProgram",
        {
            "dependencies": "explicit_transfer_edges",
            "device_placement": "organ_level_choice",
            "qualification": "public_api_and_measurement",
        },
    ),
)

RECOVERED_PRIMITIVES = (
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


def recovered_hypotheses() -> list[dict[str, Any]]:
    rows = []
    for behavior_id, atlas_kind, primitive, placement in _ATLAS_ROWS:
        rows.append(
            {
                "behavior_id": behavior_id,
                "buffers": list(_COMMON_BUFFERS),
                "hwir_node_kind": atlas_kind,
                "ir_node_kind": PRIMITIVE_TO_NODE_KIND[primitive],
                "label": _ATLAS_LABEL,
                "placement_constraint": json.loads(canon_dumps(placement)),
                "primitive": primitive,
                "semantic_edges": list(_COMMON_SEMANTIC_EDGES),
                "status": "CANDIDATE",
            }
        )
    return rows


def load_atlas_hypotheses() -> tuple[list[dict[str, Any]], list[str], str]:
    """Prefer the on-disk atlas; fall back to the recovered snapshot."""
    path = REPO / ATLAS_REL
    if path.is_file():
        doc = load_json(path)
        hyps = list(doc.get("hwir_hypotheses") or [])
        prims = [str(p) for p in (doc.get("backend_neutral_primitives") or [])]
        return hyps, prims, ATLAS_REL
    return recovered_hypotheses(), list(RECOVERED_PRIMITIVES), "embedded_recovery_atlas_absent_from_worktree"


# ---------------------------------------------------------------------------
# IR types
# ---------------------------------------------------------------------------

@dataclass
class PhysicalAttr:
    arithmetic_width: str | None = None
    tile_shape: list[int] | None = None
    banking: int | None = None
    hbm_channel: int | None = None
    resource_class: dict[str, int] = field(default_factory=_zero_resources)
    dfx_module_boundary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        rc = _zero_resources()
        for key, value in (self.resource_class or {}).items():
            if key in rc:
                rc[key] = int(value)
        return {
            "arithmetic_width": self.arithmetic_width,
            "banking": self.banking,
            "dfx_module_boundary": self.dfx_module_boundary,
            "hbm_channel": self.hbm_channel,
            "resource_class": rc,
            "tile_shape": list(self.tile_shape) if self.tile_shape else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PhysicalAttr":
        d = dict(data or {})
        rc = _zero_resources()
        raw = d.get("resource_class") or {}
        if isinstance(raw, Mapping):
            for key, value in raw.items():
                if key in rc:
                    rc[key] = int(value or 0)
        tile = d.get("tile_shape")
        return cls(
            arithmetic_width=d.get("arithmetic_width"),
            tile_shape=[int(x) for x in tile] if tile else None,
            banking=None if d.get("banking") is None else int(d["banking"]),
            hbm_channel=None if d.get("hbm_channel") is None else int(d["hbm_channel"]),
            resource_class=rc,
            dfx_module_boundary=d.get("dfx_module_boundary"),
        )


@dataclass
class DeviceBudget:
    """Compiler-declared resource ceiling. Not a synthesis result."""

    BRAM: int = 0
    DSP: int = 0
    LUT: int = 0
    URAM: int = 0
    device_id: str = "unselected-fpga-device"
    hbm_channels: int | None = None
    declared_not_measured: bool = True
    status: str = "DECLARED_COMPILER_CONSTRAINT"

    def to_dict(self) -> dict[str, Any]:
        return {
            "BRAM": int(self.BRAM),
            "DSP": int(self.DSP),
            "LUT": int(self.LUT),
            "URAM": int(self.URAM),
            "declared_not_measured": True,
            "device_id": self.device_id,
            "hbm_channels": self.hbm_channels,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "DeviceBudget":
        d = dict(data or {})
        return cls(
            BRAM=int(d.get("BRAM") or 0),
            DSP=int(d.get("DSP") or 0),
            LUT=int(d.get("LUT") or 0),
            URAM=int(d.get("URAM") or 0),
            device_id=str(d.get("device_id") or "unselected-fpga-device"),
            hbm_channels=None if d.get("hbm_channels") is None else int(d["hbm_channels"]),
            declared_not_measured=True,
            status=str(d.get("status") or "DECLARED_COMPILER_CONSTRAINT"),
        )

    def ceiling(self, klass: str) -> int:
        return int(getattr(self, klass))


@dataclass
class HwirNode:
    id: str
    kind: str
    primitive: str = ""
    semantics: str = "noetic_native"
    organ: str = ""
    mapping: str = ""
    owner: str | None = None
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    physical: PhysicalAttr = field(default_factory=PhysicalAttr)
    lifetime: str | None = None
    per_token_transfer: bool | None = None
    resident_weight_policy: str | None = None
    transport_policy: str | None = None
    assumes_source_tensor_identity: bool = False
    dense_weight_materialization: bool = False

    def __post_init__(self) -> None:
        self.kind = canon_kind(self.kind)
        self.inputs = {str(k): canon_frame(v) for k, v in sorted(self.inputs.items())}
        self.outputs = {str(k): canon_frame(v) for k, v in sorted(self.outputs.items())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumes_source_tensor_identity": bool(self.assumes_source_tensor_identity),
            "dense_weight_materialization": bool(self.dense_weight_materialization),
            "id": self.id,
            "inputs": dict(sorted(self.inputs.items())),
            "kind": self.kind,
            "lifetime": self.lifetime,
            "mapping": self.mapping,
            "organ": self.organ,
            "outputs": dict(sorted(self.outputs.items())),
            "owner": self.owner,
            "per_token_transfer": self.per_token_transfer,
            "physical": self.physical.to_dict(),
            "primitive": self.primitive,
            "resident_weight_policy": self.resident_weight_policy,
            "semantics": self.semantics,
            "transport_policy": self.transport_policy,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HwirNode":
        d = dict(data)
        return cls(
            id=str(d["id"]),
            kind=str(d.get("kind") or ""),
            primitive=str(d.get("primitive") or ""),
            semantics=str(d.get("semantics") or "noetic_native"),
            organ=str(d.get("organ") or ""),
            mapping=str(d.get("mapping") or ""),
            owner=d.get("owner"),
            inputs=dict(d.get("inputs") or {}),
            outputs=dict(d.get("outputs") or {}),
            physical=PhysicalAttr.from_dict(d.get("physical")),
            lifetime=d.get("lifetime"),
            per_token_transfer=d.get("per_token_transfer"),
            resident_weight_policy=d.get("resident_weight_policy"),
            transport_policy=d.get("transport_policy"),
            assumes_source_tensor_identity=bool(d.get("assumes_source_tensor_identity")),
            dense_weight_materialization=bool(d.get("dense_weight_materialization")),
        )


@dataclass
class HwirEdge:
    id: str
    src: str
    dst: str
    src_port: str
    dst_port: str
    frame_kind: str
    in_transit_transform: str = "identity"

    def __post_init__(self) -> None:
        self.frame_kind = canon_frame(self.frame_kind)
        self.in_transit_transform = canon_transform(self.in_transit_transform)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dst": self.dst,
            "dst_port": self.dst_port,
            "frame_kind": self.frame_kind,
            "id": self.id,
            "in_transit_transform": self.in_transit_transform,
            "src": self.src,
            "src_port": self.src_port,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HwirEdge":
        d = dict(data)
        transform = d.get("in_transit_transform") or d.get("transform") or "identity"
        return cls(
            id=str(d["id"]),
            src=str(d["src"]),
            dst=str(d["dst"]),
            src_port=str(d.get("src_port") or "out"),
            dst_port=str(d.get("dst_port") or "in"),
            frame_kind=str(d.get("frame_kind") or "activation"),
            in_transit_transform=str(transform),
        )


@dataclass
class ValidationReport:
    ok: bool
    errors: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {"errors": list(self.errors), "ok": self.ok}

    def codes(self) -> list[str]:
        return [e["code"] for e in self.errors]


@dataclass
class HwirGraph:
    schema: str = SCHEMA
    version: int = VERSION
    model: str = ""
    organ: str = ""
    source_receipt: str = ""
    source_hwir_schema: str = ""
    qualification: str = "STATIC_ONLY"
    semantics_consumed: str = "physical_graph_noetic_native"
    nodes: list[HwirNode] = field(default_factory=list)
    edges: list[HwirEdge] = field(default_factory=list)
    device_budget: DeviceBudget | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        nodes = sorted((n.to_dict() for n in self.nodes), key=lambda n: n["id"])
        edges = sorted(
            (e.to_dict() for e in self.edges),
            key=lambda e: (e["src"], e["dst"], e["id"]),
        )
        body = {
            "device_budget": None if self.device_budget is None else self.device_budget.to_dict(),
            "edges": edges,
            "model": self.model,
            "nodes": nodes,
            "notes": list(self.notes),
            "organ": self.organ,
            "qualification": self.qualification,
            "schema": SCHEMA,
            "semantics_consumed": self.semantics_consumed,
            "source_hwir_schema": self.source_hwir_schema,
            "source_receipt": self.source_receipt,
            "version": VERSION,
        }
        body["fingerprint"] = _fingerprint_body(body)
        return body

    def to_json(self) -> str:
        return canon_dumps(self.to_dict())

    def fingerprint(self) -> str:
        return self.to_dict()["fingerprint"]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HwirGraph":
        d = dict(data)
        budget_raw = d.get("device_budget")
        return cls(
            schema=str(d.get("schema") or SCHEMA),
            version=int(d.get("version") or VERSION),
            model=str(d.get("model") or ""),
            organ=str(d.get("organ") or ""),
            source_receipt=str(d.get("source_receipt") or ""),
            source_hwir_schema=str(d.get("source_hwir_schema") or ""),
            qualification=str(d.get("qualification") or "STATIC_ONLY"),
            semantics_consumed=str(d.get("semantics_consumed") or "physical_graph_noetic_native"),
            nodes=[HwirNode.from_dict(n) for n in (d.get("nodes") or [])],
            edges=[HwirEdge.from_dict(e) for e in (d.get("edges") or [])],
            device_budget=None if not budget_raw else DeviceBudget.from_dict(budget_raw),
            notes=[str(x) for x in (d.get("notes") or [])],
        )

    @classmethod
    def from_json(cls, blob: str) -> "HwirGraph":
        return cls.from_dict(json.loads(blob))

    def validate(self) -> ValidationReport:
        return validate(self)


def _fingerprint_body(body: Mapping[str, Any]) -> str:
    hashed = {k: v for k, v in body.items() if k != "fingerprint"}
    return hashlib.sha256(canon_dumps(hashed).encode("utf-8")).hexdigest()


def to_json(graph: HwirGraph) -> str:
    return graph.to_json()


def from_json(blob: str) -> HwirGraph:
    return HwirGraph.from_json(blob)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

def _error(code: str, path: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message, "path": path}


def _node_dense_illegal(node: HwirNode) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    path = f"nodes.{node.id}"
    if node.assumes_source_tensor_identity or node.semantics in FORBIDDEN_SEMANTICS:
        errors.append(
            _error(
                "SOURCE_TENSOR_IDENTITY",
                path,
                "node assumes raw source-tensor identity; HWIR consumes Noetic/PhysicalGraph semantics",
            )
        )
    dense = (
        node.dense_weight_materialization
        or node.primitive in FORBIDDEN_PRIMITIVES
        or _claims_dense_source(node.mapping)
        or _claims_dense_source(node.primitive)
        or _claims_dense_source(node.semantics)
        or _claims_dense_source(node.resident_weight_policy)
    )
    if dense:
        errors.append(
            _error(
                "DENSE_WEIGHT_MATERIALIZATION",
                path,
                "dense weight rematerialization / source-matrix GEMM is forbidden",
            )
        )
    return errors


def validate(graph: HwirGraph | Mapping[str, Any] | str) -> ValidationReport:
    """Reject illegal HWIR. A guard nobody has watched fail is not a guard."""
    if isinstance(graph, str):
        graph = HwirGraph.from_json(graph)
    elif isinstance(graph, Mapping):
        graph = HwirGraph.from_dict(graph)

    errors: list[dict[str, str]] = []
    nodes = {n.id: n for n in graph.nodes}
    if len(nodes) != len(graph.nodes):
        errors.append(_error("DUPLICATE_NODE_ID", "nodes", "node ids must be unique"))
    if not graph.nodes:
        errors.append(_error("EMPTY_GRAPH", "nodes", "graph has no nodes"))

    for node in graph.nodes:
        if node.kind not in NODE_KINDS:
            errors.append(
                _error("UNKNOWN_NODE_KIND", f"nodes.{node.id}.kind", f"unknown kind {node.kind!r}")
            )
        for port, frame in list(node.inputs.items()) + list(node.outputs.items()):
            if frame not in FRAME_KINDS:
                errors.append(
                    _error(
                        "UNKNOWN_FRAME_KIND",
                        f"nodes.{node.id}.port.{port}",
                        f"unknown frame {frame!r}",
                    )
                )
        if node.kind == "state":
            owner = node.owner if node.owner is not None else ""
            if not str(owner).strip():
                errors.append(
                    _error(
                        "STATE_NO_OWNER",
                        f"nodes.{node.id}.owner",
                        "state node has no authoritative owner",
                    )
                )
        errors.extend(_node_dense_illegal(node))

    edge_ids: set[str] = set()
    for edge in graph.edges:
        path = f"edges.{edge.id}"
        if edge.id in edge_ids:
            errors.append(_error("DUPLICATE_EDGE_ID", path, "edge ids must be unique"))
        edge_ids.add(edge.id)
        if edge.in_transit_transform not in TRANSFORMS:
            errors.append(
                _error(
                    "UNKNOWN_TRANSFORM",
                    f"{path}.in_transit_transform",
                    f"unknown transform {edge.in_transit_transform!r}",
                )
            )
        if edge.src not in nodes or edge.dst not in nodes:
            missing = []
            if edge.src not in nodes:
                missing.append(f"src={edge.src}")
            if edge.dst not in nodes:
                missing.append(f"dst={edge.dst}")
            errors.append(
                _error("DANGLING_EDGE", path, "dangling edge " + ", ".join(missing))
            )
            continue
        src = nodes[edge.src]
        dst = nodes[edge.dst]
        if edge.frame_kind not in FRAME_KINDS:
            errors.append(
                _error("UNKNOWN_FRAME_KIND", f"{path}.frame_kind", f"unknown frame {edge.frame_kind!r}")
            )
        if edge.src_port not in src.outputs:
            errors.append(
                _error(
                    "TYPE_MISMATCH",
                    path,
                    f"src port {edge.src_port!r} not on {src.id} outputs {sorted(src.outputs)}",
                )
            )
            continue
        if edge.dst_port not in dst.inputs:
            errors.append(
                _error(
                    "TYPE_MISMATCH",
                    path,
                    f"dst port {edge.dst_port!r} not on {dst.id} inputs {sorted(dst.inputs)}",
                )
            )
            continue
        produced = src.outputs[edge.src_port]
        if produced != edge.frame_kind:
            errors.append(
                _error(
                    "TYPE_MISMATCH",
                    path,
                    f"edge frame {edge.frame_kind} != producer port {produced}",
                )
            )
        post = apply_transform(edge.frame_kind, edge.in_transit_transform)
        accepted = dst.inputs[edge.dst_port]
        if post is None:
            errors.append(
                _error(
                    "TYPE_MISMATCH",
                    path,
                    f"transform {edge.in_transit_transform} is illegal on frame {edge.frame_kind}",
                )
            )
        elif post != accepted:
            errors.append(
                _error(
                    "TYPE_MISMATCH",
                    path,
                    f"post-transform frame {post} != consumer port {accepted}",
                )
            )

    if graph.device_budget is not None:
        used = _zero_resources()
        for node in graph.nodes:
            rc = node.physical.resource_class or {}
            for klass in RESOURCE_CLASSES:
                used[klass] += int(rc.get(klass) or 0)
        for klass in RESOURCE_CLASSES:
            ceiling = graph.device_budget.ceiling(klass)
            if used[klass] > ceiling:
                errors.append(
                    _error(
                        "RESOURCE_OVER_BUDGET",
                        f"device_budget.{klass}",
                        f"declared {klass} request {used[klass]} exceeds budget {ceiling}",
                    )
                )

    errors.sort(key=lambda e: (e["code"], e["path"], e["message"]))
    return ValidationReport(ok=not errors, errors=errors)


# ---------------------------------------------------------------------------
# Lowering from a real FPGA organ-map receipt
# ---------------------------------------------------------------------------

_ORGAN_COMPUTE_PRIMITIVE = {
    "expert_bank": "DirectRoutedAccumulate",
    "router_topk_and_gather": "SparseSkip",
    "routed_plus_shared_expert": "DirectRoutedAccumulate",
    "deltanet_persistent_state": "TiledProjection",
    "ngram_lookup_or_generator": "MoveOrRecompute",
    "sparse_attention": "SparseSkip",
    "mtp_draft_verify_rollback": "ConditionalPhysicalProgram",
    "mlp_gate_up_down": "TiledProjection",
    "gqa_qkv_and_output": "TiledProjection",
    "deltanet_state_and_input_projection": "TiledProjection",
    "norm_add_epilogues": "LayoutTransform",
    "lm_head_and_sampling": "CollectiveRegion",
    "command_buffer_graph": "GraphReplay",
}

_REDUCTION_ORGANS = frozenset(
    {
        "expert_bank",
        "routed_plus_shared_expert",
        "sparse_attention",
        "mlp_gate_up_down",
        "gqa_qkv_and_output",
        "lm_head_and_sampling",
    }
)


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p.resolve()
    cand = (REPO / p).resolve()
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"organ map not found: {path}")


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _pick_organ(organs: list[Mapping[str, Any]], organ_id: str | None) -> dict[str, Any]:
    rows = [dict(o) for o in organs if isinstance(o, Mapping)]
    if not rows:
        raise ValueError("organ map has no organs")
    if organ_id:
        for row in rows:
            name = str(row.get("organ") or row.get("id") or "")
            if name == organ_id:
                return row
        known = [str(r.get("organ") or r.get("id")) for r in rows]
        raise ValueError(f"organ {organ_id!r} not in map; known={known}")
    for row in rows:
        if str(row.get("priority") or "") == "P0":
            return row
    return rows[0]


def _arithmetic_width(mapping: str) -> str:
    m = mapping.lower()
    if "nf gemv" in m or "native nf" in m:
        return "nf"
    if "low-bit" in m or "low bit" in m or "packed" in m:
        return "packed_low_bit"
    return "unspecified"


def _roles_for_organ(organ_id: str, mapping: str) -> set[str]:
    roles = {"memory", "decoder", "compute", "dma_in", "dma_out"}
    oid = organ_id.lower()
    m = mapping.lower()
    if organ_id in _REDUCTION_ORGANS or "reduc" in m or "accumul" in m:
        roles.add("reduction")
    if "deltanet" in oid or "state" in oid or "mtp" in oid or "state" in m:
        roles.add("state")
    if (
        "pipeline" in m
        or "scheduling" in m
        or "persistent" in m
        or "command_buffer" in oid
        or "mtp" in oid
        or "graph replay" in m
    ):
        roles.add("pipeline")
    return roles


def _phys_for(organ_id: str, mapping: str, *, hbm_channel: int | None) -> PhysicalAttr:
    return PhysicalAttr(
        arithmetic_width=_arithmetic_width(mapping),
        tile_shape=None,
        banking=None,
        hbm_channel=hbm_channel,
        resource_class=_zero_resources(),
        dfx_module_boundary=organ_id,
    )


def from_organ_map(path: str | Path, organ_id: str | None = None) -> HwirGraph:
    """Lower one real Hawking organ from an FPGA organ-map receipt into HWIR."""
    src = _resolve_path(path)
    doc = load_json(src)
    organs = list(doc.get("organs") or [])
    chosen = _pick_organ(organs, organ_id)
    oid = str(chosen.get("organ") or chosen.get("id") or "unknown")
    mapping = str(chosen.get("mapping") or "")
    stub = doc.get("hwir") if isinstance(doc.get("hwir"), Mapping) else {}
    placements = {
        str(p.get("organ")): dict(p)
        for p in (stub.get("placements") or [])
        if isinstance(p, Mapping)
    }
    place = placements.get(oid) or {}
    resident = str(
        place.get("resident_weight_policy")
        or "resident_shards_no_weight_body_per_token_transfer"
    )
    transport = str(place.get("transport_policy") or "activations_and_partial_reductions_only")
    model = str(doc.get("model") or stub.get("model") or "")
    hbm = doc.get("hbm_genome") if isinstance(doc.get("hbm_genome"), Mapping) else {}
    raw_ch = hbm.get("channels")
    hbm_channel = None if not raw_ch else int(raw_ch)
    roles = _roles_for_organ(oid, mapping)
    phys = lambda: _phys_for(oid, mapping, hbm_channel=hbm_channel)

    nodes: list[HwirNode] = []
    edges: list[HwirEdge] = []

    dma_in = f"dma.{oid}.in"
    dma_out = f"dma.{oid}.out"
    mem_id = f"mem.{oid}.shards"
    dec_id = f"dec.{oid}.native"
    cmp_id = f"cmp.{oid}.body"
    red_id = f"red.{oid}.partial"
    st_id = f"st.{oid}.owner"
    pipe_id = f"pipe.{oid}.region"

    want_red = "reduction" in roles
    out_frame = "partial_reduction" if want_red else "activation"

    nodes.append(
        HwirNode(
            id=dma_in,
            kind="dma-transport",
            primitive="SemanticTransportEdge",
            organ=oid,
            mapping="token activation ingress; no weight body",
            outputs={"out": "activation"},
            physical=phys(),
            lifetime="token",
            per_token_transfer=True,
            transport_policy=transport,
        )
    )
    nodes.append(
        HwirNode(
            id=dma_out,
            kind="dma-transport",
            primitive="SemanticTransportEdge",
            organ=oid,
            mapping="token activation or partial-reduction egress; no weight body",
            inputs={"in": out_frame},
            physical=phys(),
            lifetime="token",
            per_token_transfer=True,
            transport_policy=transport,
        )
    )
    nodes.append(
        HwirNode(
            id=mem_id,
            kind="memory",
            primitive="StationaryRepresentation",
            organ=oid,
            mapping="resident compact representation shards; not source-dense weights",
            outputs={"out": "compact_representation_fragment"},
            physical=phys(),
            lifetime="persistent",
            per_token_transfer=False,
            resident_weight_policy=resident,
        )
    )
    nodes.append(
        HwirNode(
            id=dec_id,
            kind="representation-decoder",
            primitive="FusedDecodeCompute",
            organ=oid,
            mapping="native decode of compact representation at the consumer; no_dense_rematerialization",
            inputs={"in": "compact_representation_fragment"},
            outputs={"out": "activation"},
            physical=phys(),
            lifetime="token",
        )
    )

    cmp_inputs = {"in_act": "activation", "in_rep": "activation"}
    cmp_outputs = {"out": out_frame}
    if "state" in roles:
        cmp_inputs["in_state"] = "state"
        cmp_outputs["out_state"] = "state"
    nodes.append(
        HwirNode(
            id=cmp_id,
            kind="compute",
            primitive=_ORGAN_COMPUTE_PRIMITIVE.get(oid, "TiledProjection"),
            organ=oid,
            mapping=mapping or "noetic-native compute; no source-dense GEMM",
            inputs=cmp_inputs,
            outputs=cmp_outputs,
            physical=phys(),
            lifetime="token",
            resident_weight_policy=resident,
        )
    )

    if want_red:
        nodes.append(
            HwirNode(
                id=red_id,
                kind="reduction",
                primitive="CollectiveRegion",
                organ=oid,
                mapping="partial reduction of native fragments",
                inputs={"in": "partial_reduction"},
                outputs={"out": "partial_reduction"},
                physical=phys(),
                lifetime="token",
            )
        )
    if "state" in roles:
        nodes.append(
            HwirNode(
                id=st_id,
                kind="state",
                primitive="LocalStateMachine",
                organ=oid,
                mapping="authoritative resident state owner",
                owner=st_id,
                inputs={"in": "state"},
                outputs={"out": "state"},
                physical=phys(),
                lifetime="sequence",
                per_token_transfer=False,
            )
        )
    if "pipeline" in roles:
        nodes.append(
            HwirNode(
                id=pipe_id,
                kind="persistent-pipeline",
                primitive="PersistentPhysicalRegion",
                organ=oid,
                mapping="persistent region / graph-replay identity; DFX candidate",
                physical=phys(),
                lifetime="persistent",
            )
        )

    def edge(eid: str, src: str, sport: str, dst: str, dport: str, frame: str, transform: str = "identity") -> None:
        edges.append(
            HwirEdge(
                id=eid,
                src=src,
                dst=dst,
                src_port=sport,
                dst_port=dport,
                frame_kind=frame,
                in_transit_transform=transform,
            )
        )

    edge(f"e.{oid}.act", dma_in, "out", cmp_id, "in_act", "activation")
    edge(f"e.{oid}.compact", mem_id, "out", dec_id, "in", "compact_representation_fragment")
    edge(f"e.{oid}.decoded", dec_id, "out", cmp_id, "in_rep", "activation")
    if want_red:
        edge(f"e.{oid}.partial", cmp_id, "out", red_id, "in", "partial_reduction")
        edge(f"e.{oid}.egress", red_id, "out", dma_out, "in", "partial_reduction")
    else:
        edge(f"e.{oid}.egress", cmp_id, "out", dma_out, "in", "activation")
    if "state" in roles:
        edge(f"e.{oid}.state.rd", st_id, "out", cmp_id, "in_state", "state")
        edge(f"e.{oid}.state.wr", cmp_id, "out_state", st_id, "in", "state")

    notes = [
        "Lowered from a real FPGA organ-map receipt; not a bitstream and not a hardware timing claim.",
        "Resident compact shards stay put; per-token transport is activation / partial reduction / state only.",
        "Representation-decoder is FusedDecodeCompute: native decode, no dense rematerialization.",
        "PhysicalGraph semantics consumed: organ is a role, not a source tensor name.",
        "hbm_channel is None while the organ-map device genome is TARGET_UNSELECTED.",
        "resource_class zeros mean undeclared/unmeasured, not a synthesis of zero.",
    ]
    return HwirGraph(
        model=model,
        organ=oid,
        source_receipt=_rel(src),
        source_hwir_schema=str(stub.get("schema") or doc.get("schema") or ""),
        qualification="STATIC_ONLY",
        semantics_consumed="physical_graph_noetic_native",
        nodes=nodes,
        edges=edges,
        device_budget=None,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Negative-control graphs (constructed to be invalid)
# ---------------------------------------------------------------------------

def graph_dense_source_rematerialization() -> HwirGraph:
    """Illegal by construction: compute that materializes source dense weights."""
    return HwirGraph(
        model="negative-control",
        organ="illegal_dense_source",
        qualification="STATIC_ONLY",
        semantics_consumed="source_tensor_identity",
        notes=["constructed specifically to be invalid"],
        nodes=[
            HwirNode(
                id="dma.in",
                kind="dma-transport",
                primitive="SemanticTransportEdge",
                outputs={"out": "activation"},
            ),
            HwirNode(
                id="dma.out",
                kind="dma-transport",
                primitive="SemanticTransportEdge",
                inputs={"in": "activation"},
            ),
            HwirNode(
                id="mem.source_dense",
                kind="memory",
                primitive="RematerializeDenseWeights",
                semantics="source_tensor_identity",
                mapping="materialize the original dense weight matrix for GEMM",
                outputs={"out": "activation"},
                lifetime="token",
                per_token_transfer=True,
                resident_weight_policy="transfer_dense_weight_body_per_token",
                assumes_source_tensor_identity=True,
                dense_weight_materialization=True,
            ),
            HwirNode(
                id="cmp.dense_gemm",
                kind="compute",
                primitive="DenseSourceMatmul",
                semantics="source_tensor_identity",
                mapping="multiply the original dense weight matrices",
                inputs={"in_w": "activation", "in_act": "activation"},
                outputs={"out": "activation"},
                assumes_source_tensor_identity=True,
                dense_weight_materialization=True,
            ),
        ],
        edges=[
            HwirEdge(
                id="e.act",
                src="dma.in",
                src_port="out",
                dst="cmp.dense_gemm",
                dst_port="in_act",
                frame_kind="activation",
            ),
            HwirEdge(
                id="e.w",
                src="mem.source_dense",
                src_port="out",
                dst="cmp.dense_gemm",
                dst_port="in_w",
                frame_kind="activation",
            ),
            HwirEdge(
                id="e.out",
                src="cmp.dense_gemm",
                src_port="out",
                dst="dma.out",
                dst_port="in",
                frame_kind="activation",
            ),
        ],
    )


def graph_dangling_edge() -> HwirGraph:
    """Illegal by construction: edge whose endpoints are not in the node set."""
    return HwirGraph(
        model="negative-control",
        organ="illegal_dangling",
        qualification="STATIC_ONLY",
        notes=["constructed specifically to be invalid"],
        nodes=[
            HwirNode(
                id="dma.in",
                kind="dma-transport",
                primitive="SemanticTransportEdge",
                outputs={"out": "activation"},
            ),
            HwirNode(
                id="cmp.body",
                kind="compute",
                primitive="TiledProjection",
                mapping="noetic-native compute; no_dense_rematerialization",
                inputs={"in_act": "activation"},
                outputs={"out": "activation"},
            ),
            HwirNode(
                id="dma.out",
                kind="dma-transport",
                primitive="SemanticTransportEdge",
                inputs={"in": "activation"},
            ),
        ],
        edges=[
            HwirEdge(
                id="e.act",
                src="dma.in",
                src_port="out",
                dst="cmp.body",
                dst_port="in_act",
                frame_kind="activation",
            ),
            HwirEdge(
                id="e.out",
                src="cmp.body",
                src_port="out",
                dst="dma.out",
                dst_port="in",
                frame_kind="activation",
            ),
            HwirEdge(
                id="e.ghost",
                src="missing.src",
                src_port="out",
                dst="missing.dst",
                dst_port="in",
                frame_kind="activation",
            ),
        ],
    )


def graph_state_without_owner() -> HwirGraph:
    return HwirGraph(
        model="negative-control",
        organ="illegal_unowned_state",
        nodes=[
            HwirNode(
                id="st.orphan",
                kind="state",
                primitive="LocalStateMachine",
                owner=None,
                inputs={"in": "state"},
                outputs={"out": "state"},
            )
        ],
    )


def graph_over_budget() -> HwirGraph:
    g = HwirGraph(
        model="negative-control",
        organ="illegal_over_budget",
        device_budget=DeviceBudget(BRAM=1, DSP=1, LUT=8, URAM=1),
        nodes=[
            HwirNode(
                id="cmp.fat",
                kind="compute",
                primitive="TiledProjection",
                outputs={"out": "activation"},
                physical=PhysicalAttr(resource_class={"BRAM": 0, "DSP": 0, "LUT": 64, "URAM": 0}),
            )
        ],
    )
    return g


def graph_type_mismatch() -> HwirGraph:
    return HwirGraph(
        model="negative-control",
        organ="illegal_type_mismatch",
        nodes=[
            HwirNode(
                id="dma.in",
                kind="dma-transport",
                outputs={"out": "activation"},
            ),
            HwirNode(
                id="dec.body",
                kind="representation-decoder",
                primitive="FusedDecodeCompute",
                inputs={"in": "compact_representation_fragment"},
                outputs={"out": "activation"},
            ),
        ],
        edges=[
            HwirEdge(
                id="e.bad",
                src="dma.in",
                src_port="out",
                dst="dec.body",
                dst_port="in",
                frame_kind="activation",
            )
        ],
    )


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------

def _summarize(graph: HwirGraph) -> dict[str, Any]:
    report = validate(graph)
    kinds = sorted({n.kind for n in graph.nodes})
    frames = sorted({e.frame_kind for e in graph.edges})
    return {
        "device_budget": None if graph.device_budget is None else graph.device_budget.to_dict(),
        "edge_count": len(graph.edges),
        "fingerprint": graph.fingerprint(),
        "frame_kinds": frames,
        "graph": graph.to_dict(),
        "model": graph.model,
        "node_count": len(graph.nodes),
        "node_kinds": kinds,
        "organ": graph.organ,
        "source_receipt": graph.source_receipt,
        "validate": report.to_dict(),
    }


def _run_proofs() -> dict[str, Any]:
    flash_path = REPO / FLASH_ORGAN_MAP
    qwen_path = REPO / QWEN_ORGAN_MAP
    lowered = from_organ_map(flash_path, "expert_bank")
    lowered_state = from_organ_map(flash_path, "deltanet_persistent_state")
    lowered_qwen = from_organ_map(qwen_path, "mlp_gate_up_down") if qwen_path.is_file() else None

    blob1 = lowered.to_json()
    blob2 = HwirGraph.from_json(blob1).to_json()
    round_trip = blob1 == blob2 and blob1.encode("utf-8") == blob2.encode("utf-8")
    if "recorded_at" in blob1 or "generated_at" in blob1:
        round_trip = False

    v_ok = validate(lowered)
    v_state = validate(lowered_state)
    v_qwen = validate(lowered_qwen) if lowered_qwen is not None else None
    v_dense = validate(graph_dense_source_rematerialization())
    v_dangle = validate(graph_dangling_edge())
    v_owner = validate(graph_state_without_owner())
    v_budget = validate(graph_over_budget())
    v_type = validate(graph_type_mismatch())

    proofs = {
        "dangling_codes": v_dangle.codes(),
        "dangling_edge_rejected": (not v_dangle.ok) and ("DANGLING_EDGE" in v_dangle.codes()),
        "dense_codes": v_dense.codes(),
        "dense_source_rejected": (not v_dense.ok)
        and ("DENSE_WEIGHT_MATERIALIZATION" in v_dense.codes()),
        "lowered_kinds": sorted({n.kind for n in lowered.nodes}),
        "lowered_state_kinds": sorted({n.kind for n in lowered_state.nodes}),
        "lowered_state_valid": v_state.ok,
        "lowered_valid": v_ok.ok,
        "qwen_lowered_valid": None if v_qwen is None else v_qwen.ok,
        "resource_over_budget_rejected": (not v_budget.ok) and ("RESOURCE_OVER_BUDGET" in v_budget.codes()),
        "round_trip_bytes": len(blob1.encode("utf-8")),
        "round_trip_equal": round_trip,
        "state_no_owner_rejected": (not v_owner.ok) and ("STATE_NO_OWNER" in v_owner.codes()),
        "type_mismatch_rejected": (not v_type.ok) and ("TYPE_MISMATCH" in v_type.codes()),
        "wall_clock_in_hashed_content": False,
    }
    if not proofs["lowered_valid"]:
        raise RuntimeError(f"lowered expert_bank failed validate: {v_ok.errors}")
    if not proofs["lowered_state_valid"]:
        raise RuntimeError(f"lowered deltanet failed validate: {v_state.errors}")
    if lowered_qwen is not None and not proofs["qwen_lowered_valid"]:
        raise RuntimeError(f"lowered qwen mlp failed validate: {v_qwen.errors}")
    if not proofs["round_trip_equal"]:
        raise RuntimeError("byte-stable round-trip failed")
    if not proofs["dense_source_rejected"]:
        raise RuntimeError("dense-source negative control did not fire")
    if not proofs["dangling_edge_rejected"]:
        raise RuntimeError("dangling-edge negative control did not fire")
    proofs["all_seven_kinds_exercised"] = set(NODE_KINDS) <= (
        set(proofs["lowered_kinds"]) | set(proofs["lowered_state_kinds"])
    )
    proofs["lowered_qwen_fingerprint"] = None if lowered_qwen is None else lowered_qwen.fingerprint()
    proofs["lowered_state_fingerprint"] = lowered_state.fingerprint()
    return proofs


def _atlas_present() -> dict[str, Any]:
    path = REPO / ATLAS_REL
    return {
        "git_head_has_file": False,
        "on_disk": path.is_file(),
        "path": ATLAS_REL,
        "recovered_from": (
            "parent checkout /Users/scammermike/Downloads/hawking/"
            "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json"
        ),
        "recovered_fingerprint": "e763623e8a4ddcfc8350d6b5f680284a23db68f8bfbca53428e258d24adfc2ab",
        "schema": "hawking.accelerator.architecture_atlas.v1",
    }


def build() -> Path:
    hyps, prims, hyp_source = load_atlas_hypotheses()
    proofs = _run_proofs()
    lowered = from_organ_map(REPO / FLASH_ORGAN_MAP, "expert_bank")
    flash_sha = sha256_file(REPO / FLASH_ORGAN_MAP) if (REPO / FLASH_ORGAN_MAP).is_file() else None
    qwen_sha = sha256_file(REPO / QWEN_ORGAN_MAP) if (REPO / QWEN_ORGAN_MAP).is_file() else None

    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Hardware IR a future physical compiler uses to decide what an FPGA "
            "should become. Not a backend, not HDL, not a bitstream."
        ),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "node_kinds": list(NODE_KINDS),
        "frame_kinds": list(FRAME_KINDS),
        "in_transit_transforms": list(TRANSFORMS),
        "resource_classes": list(RESOURCE_CLASSES),
        "primitive_to_node_kind": dict(sorted(PRIMITIVE_TO_NODE_KIND.items())),
        "backend_neutral_primitives": prims,
        "hwir_hypotheses": hyps,
        "hypotheses_source": hyp_source,
        "physical_graph_semantics_consumed": list(PHYSICAL_GRAPH_FIELDS),
        "atlas": _atlas_present(),
        "serialization": {
            "canonical": "json.dumps(sort_keys=True, separators=(',', ':'), ensure_ascii=True)",
            "fingerprint": "sha256 of canonical body excluding the fingerprint field",
            "round_trip_equal": proofs["round_trip_equal"],
            "wall_clock_in_hashed_content": False,
        },
        "validator_rules": [
            "SOURCE_TENSOR_IDENTITY: node.semantics or assumes_source_tensor_identity",
            "DENSE_WEIGHT_MATERIALIZATION: explicit flag, forbidden primitive, or affirmative dense-source mapping",
            "DANGLING_EDGE: src/dst not in the node set",
            "TYPE_MISMATCH: missing port, frame disagreement, or illegal in-transit transform",
            "RESOURCE_OVER_BUDGET: summed declared resource_class exceeds device_budget",
            "STATE_NO_OWNER: kind=state with empty owner",
        ],
        "lowered": _summarize(lowered),
        "proofs": proofs,
        "organ_map_inputs": {
            "flash": {"path": FLASH_ORGAN_MAP, "sha256": flash_sha, "present": flash_sha is not None},
            "qwen27": {"path": QWEN_ORGAN_MAP, "sha256": qwen_sha, "present": qwen_sha is not None},
        },
        "recovered_implementation": [
            {
                "path": ATLAS_REL,
                "what": "15 hwir_hypotheses + 17 backend_neutral_primitives (atlas)",
                "adequate_as_ir": False,
                "note": (
                    "Not in this worktree HEAD or sparse disk. Recovered from the parent "
                    "Hawking checkout. Consumed as input spec, not re-derived."
                ),
            },
            {
                "path": "hcli/agentos/fpga_preboard.py",
                "what": "class HWIR, schema hcli.fpga.hwir.v1",
                "adequate_as_ir": False,
                "note": (
                    "Pre-board sketch: nodes are kind=organ_operator, buffers untyped, "
                    "no validator, no resource classes, no byte-stable node/edge IR. "
                    "Cannot be edited (Codex/hcli surface). Consumed via organ-map receipts."
                ),
            },
            {
                "path": FLASH_ORGAN_MAP,
                "what": "Flash FPGA organ map with embedded hcli.fpga.hwir.v1 stub",
                "adequate_as_ir": False,
                "note": "Lowering input. Seven Flash organs with resident-shard / no-weight-body policy.",
            },
            {
                "path": QWEN_ORGAN_MAP,
                "what": "Qwen27 FPGA organ map with embedded hcli.fpga.hwir.v1 stub",
                "adequate_as_ir": False,
                "note": "Secondary lowering input. mlp_gate_up_down is packed low-bit GEMV, not dense source GEMM.",
            },
            {
                "path": "hcli/physical_graph.py",
                "what": "PhysicalGraph dataclass + compile_physical_graph",
                "adequate_as_ir": False,
                "note": (
                    "PLAN_ONLY placement graph. Organs become computation nodes with unresolved "
                    "bytes. Semantic contract HWIR consumes: organ is a role, representation is "
                    "native, sizes unresolved, qualification PLAN_ONLY. Not materialized in this "
                    "sparse worktree; recovered via git show HEAD:hcli/physical_graph.py."
                ),
            },
            {
                "path": "receipts/headless/PHYSICAL_GRAPH_COMPILER.json",
                "what": "organ-as-role law; source-framework boundaries are not physical law",
                "adequate_as_ir": False,
                "note": "Law used as semantic constraint, not as an IR.",
            },
            {
                "path": "receipts/headless/HCLI_FPGA_PREBOARD.json",
                "what": "preboard: fpga_backend NOT_BUILT, physical_board ABSENT, hwir present as stub fingerprints",
                "adequate_as_ir": False,
                "note": "Confirms we must not build an FPGA backend. HWIR is the decision IR only.",
            },
            {
                "path": "tools/accelerator/air.py",
                "what": "AIR — Accelerator IR with Metal lowering",
                "adequate_as_ir": False,
                "note": "GPU/Metal IR. Different object. HWIR is spatial/hardware placement IR. Not forked.",
            },
            {
                "path": "hcli/agentos/preboard.py",
                "what": "hwir interface INTERFACE_DEFINED / SCHEMA_ONLY empty nodes",
                "adequate_as_ir": False,
                "note": "Named the gap this module closes. Recovered via git show.",
            },
        ],
        "gaps_closed": [
            "seven node kinds with the attributes each actually needs",
            "typed stream edges with semantic frame + optional in-transit transform",
            "physical attributes: arithmetic width, tile shape, banking, HBM channel, resource class, DFX boundary",
            "byte-stable to_json/from_json (sorted keys, no wall-clock in hashed content)",
            "validate() rejects source-tensor identity / dense rematerialization, dangling and type-mismatched edges, over-budget footprints, unowned state",
            "from_organ_map() lowers a real Flash/Qwen27 organ into a valid HWIR graph",
            "negative controls that actually fire",
        ],
        "negative_findings": [
            "ACCELERATOR_ARCHITECTURE_ATLAS.json is absent from this worktree HEAD and sparse disk",
            "hcli/physical_graph.py and hcli/agentos/fpga_preboard.py are git-present but not materialized (sparse checkout)",
            "existing hcli.fpga.hwir.v1 is not an IR: no types, no validator, no serdes, organ_operator only",
            "device genome is TARGET_UNSELECTED; HBM channel and resource footprints cannot be known without a board/synthesis",
            "no FPGA board, no HDL, no bitstream; this module must not and did not emit any",
            "AIR exists and executes on Metal; it is not HWIR and was not reused as the spatial IR",
            "PhysicalGraph compile_physical_graph is too unresolved to lower into a resource-accurate HWIR without invention; organ maps are the reality connection",
        ],
        "not_an_fpga_backend": True,
        "claim_boundary": (
            "Static sidecar HWIR artifact. No FPGA board, bitstream, timing, "
            "or hardware measurement."
        ),
    }
    return write_receipt(RECEIPT, doc, "tools/future/hwir.py")


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--lower", metavar="ORGAN_MAP")
    ap.add_argument("--organ")
    a = ap.parse_args()
    if a.lower:
        graph = from_organ_map(a.lower, a.organ)
        report = validate(graph)
        print(graph.to_json())
        if not report.ok:
            print(canon_dumps(report.to_dict()), file=_sys.stderr)
            return 1
        return 0
    out = selftest() if (a.selftest or not a.build) else build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
