"""HWIR v1 — hardware intermediate representation plus the pre-board stack.

The FPGA/spatial school compiles into this IR. It consumes Noetic/PhysicalGraph
semantics and real FPGA organ-map receipts. A graph that assumes it is
multiplying original dense source weight matrices is invalid by construction.

This is not a vendor backend or bitstream path. Lowering targets emit
PREHARDWARE source artifacts only; they do not synthesize. There is no U50
board on this host. Every number this module emits is PREHARDWARE: STATIC,
FUNCTIONAL_SIM, COST_MODEL, or CYCLE_APPROX. None of them is HARDWARE_MEASURED.

What already lived here (v1 IR): seven node kinds, typed stream edges, physical
attributes, byte-stable serdes, a validator that actually refuses, and lowering
from Flash/Qwen27 FPGA organ maps.

What the pre-board stack adds, by connecting existing implementations rather
than rewriting them:

* evidence-backed atlas primitives via tools.future.physical_primitives.instantiate
* FUNCTIONAL_SIM of a qGEMV kernel by calling tools.future.fpga_engines.qgemv
* CYCLE_APPROX critical-path modelled cycles (not a clock, not seconds)
* COST_MODEL HBM traffic and host<->device transfer (bytes and modelled cycles)
* STATIC LUT/DSP/BRAM/URAM estimator that refuses an over-budget engine
* a synthetic U50-class device profile and a row-split partitioner
* sealed COST_MODEL / CYCLE_APPROX / STATIC predictions with a falsifier
* a scoring path keyed to wake_condition U50_PRESENT that names the
  implicated coefficient on FALSIFIED; a synthetic-arrival rehearsal
  that is never an arrival
* selectable U50-family variants (U50 / U50C / U50DD / U50LV) with
  per-field provenance; UNPINNED where public literature does not pin a number
* CarrierEnvelope: a host-side bound that DOWNGRADES a DeviceProfile (PCIe,
  power, thermal/airflow, mechanical). The real comma-device carrier is UNPINNED.
* a pluggable LoweringTarget interface with two equal-citizen source emitters
  (HLS-style C/C++ and Rust-hosted HDL/IR). Toolchain choice is not encoded.

    python3 tools/future/hwir.py --selftest
    python3 tools/future/hwir.py --build
    python3 tools/future/hwir.py --lower receipts/headless/FLASH_NEXT_FPGA_ORGAN_MAP.json --organ expert_bank
    python3 tools/future/hwir.py --qgemv
    python3 tools/future/hwir.py --seal-predictions
    python3 tools/future/hwir.py --rehearse
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO, git, sha256_file

import abc
import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "hawking.future.hwir.v1"
VERSION = 1
RECEIPT = "HWIR_V1.json"
PREDICTION_SCHEMA = "hawking.future.hwir.sealed_prediction.v1"
PREDICTION_SET_SCHEMA = "hawking.future.hwir.sealed_prediction_set.v1"
SCORE_SCHEMA = "hawking.future.hwir.prediction_score.v1"
REHEARSAL_SCHEMA = "hawking.future.hwir.synthetic_arrival_rehearsal.v1"
PREDICTION_RECEIPT = "HWIR_SEALED_PREDICTIONS.json"
REHEARSAL_RECEIPT = "HWIR_SYNTHETIC_ARRIVAL_REHEARSAL.json"
CAPABILITY_GRAPH_REL = "civilization/CAPABILITY_GRAPH.json"
WAKE_U50_PRESENT = "U50_PRESENT"

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

# Closed evidence set for every pre-board output. HARDWARE_MEASURED is illegal
# here: there is no FPGA, no U50, no synthesis, no board census.
EVIDENCE_TIERS = ("STATIC", "FUNCTIONAL_SIM", "COST_MODEL", "CYCLE_APPROX")
ILLEGAL_EVIDENCE_TIERS = frozenset({"HARDWARE_MEASURED"})
PREHARDWARE = "PREHARDWARE"
UNPINNED = "UNPINNED"

# Document-class labels for STATIC vendor-literature fields. These are not
# measurements. A field is either pinned with one of these, or explicitly
# UNPINNED with a reason. Silent defaults are illegal on family variants.
DOC_DS965 = "AMD_DATASHEET_DS965"
DOC_UG1371 = "AMD_USER_GUIDE_UG1371"
DOC_UG1120 = "AMD_PLATFORMS_USER_GUIDE_UG1120"
DOC_BRIEF = "AMD_PRODUCT_BRIEF_U50"
DOC_PSG = "AMD_PRODUCT_SELECTION_GUIDE"
DOC_COST_MODEL = "DECLARED_COST_MODEL_COEFFICIENT"
DOC_CARRIER_DOWNGRADE = "CARRIER_DOWNGRADE"
DOC_EXAMPLE = "EXAMPLE_DECLARED_ENVELOPE"
DOC_UNPINNED = "UNPINNED"

REAL_CARRIER_NOTE = (
    "The operator reports the inbound card will hang off a comma device. "
    "That carrier is UNPINNED. Example envelopes are labeled examples, not "
    "that carrier. Pin CarrierEnvelope in one constructor call when numbers "
    "are known. Do not invent the comma-device specifications."
)

# Mutation target for the carrier-binding refusal test. Set False to ignore
# carrier limits; the brochure-kernel-refused-under-constrained-carrier test
# must then FAIL. Never leave this False in source.
CARRIER_ENVELOPE_BINDING = True

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


def _ceil_div(n: int, d: int) -> int:
    if d <= 0:
        raise ValueError("divisor must be positive")
    n = int(n)
    return (n + d - 1) // d if n > 0 else 0


class IllegalEvidenceTier(ValueError):
    """A code path tried to emit a tier this pre-board stack is not allowed to claim."""


class ResourceOverBudget(ValueError):
    """STATIC resource ESTIMATE exceeds a declared compiler budget. Not a synth result."""

    def __init__(
        self,
        used: Mapping[str, int],
        budget: Mapping[str, int],
        overflow: Mapping[str, tuple[int, int]],
        device_id: str,
    ) -> None:
        self.used = {k: int(v) for k, v in used.items()}
        self.budget = {k: int(v) for k, v in budget.items()}
        self.overflow = {k: (int(a), int(b)) for k, (a, b) in overflow.items()}
        self.device_id = device_id
        parts = [f"{k}:{a}>{b}" for k, (a, b) in sorted(self.overflow.items())]
        super().__init__(
            f"STATIC resource ESTIMATE exceeds declared budget on {device_id}: "
            + ",".join(parts)
        )


class UnmeasuredConversionError(RuntimeError):
    """Raised if modelled cycles are asked to become wall time. There is no clock."""


def emit_evidence(tier: str, body: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Stamp a report with a legal evidence tier. The only legal producer of the field."""
    t = str(tier)
    if t in ILLEGAL_EVIDENCE_TIERS or t == "HARDWARE_MEASURED":
        raise IllegalEvidenceTier(
            f"refusing to emit evidence_tier={t!r}; no U50 board, no code path "
            "may label an output HARDWARE_MEASURED"
        )
    if t not in EVIDENCE_TIERS:
        raise IllegalEvidenceTier(
            f"evidence_tier={t!r} is not one of {list(EVIDENCE_TIERS)}"
        )
    out = dict(body or {})
    out["evidence_tier"] = t
    out["prehardware"] = True
    out["qualification"] = PREHARDWARE
    out["hardware_measured"] = False
    return out


def assert_no_hardware_measured(node: Any, path: str = "$") -> None:
    """Walk a report. HARDWARE_MEASURED as a produced tier/claim is illegal."""
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}"
            if key in {"evidence_tier", "measurement_state", "evidence_class", "tier"}:
                if value == "HARDWARE_MEASURED":
                    raise IllegalEvidenceTier(f"{here}={value!r}")
            if key == "HARDWARE_MEASURED" and value not in {False, None, 0}:
                raise IllegalEvidenceTier(f"{here}={value!r}")
            if key == "hardware_measured" and value not in {False, None, 0}:
                raise IllegalEvidenceTier(f"{here}={value!r} (must be false/absent)")
            assert_no_hardware_measured(value, here)
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            assert_no_hardware_measured(value, f"{path}[{i}]")


def collect_evidence_tiers(node: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(node, dict):
        if "evidence_tier" in node and node["evidence_tier"] is not None:
            found.add(str(node["evidence_tier"]))
        for value in node.values():
            found |= collect_evidence_tiers(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            found |= collect_evidence_tiers(value)
    return found


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

    def as_map(self) -> dict[str, int]:
        return {k: int(getattr(self, k)) for k in RESOURCE_CLASSES}


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
    evidence_tier: str = "STATIC"
    memory_tier: str | None = None
    backed_identity: str | None = None

    def __post_init__(self) -> None:
        self.kind = canon_kind(self.kind)
        self.inputs = {str(k): canon_frame(v) for k, v in sorted(self.inputs.items())}
        self.outputs = {str(k): canon_frame(v) for k, v in sorted(self.outputs.items())}
        tier = str(self.evidence_tier or "STATIC")
        if tier in ILLEGAL_EVIDENCE_TIERS or tier == "HARDWARE_MEASURED":
            raise IllegalEvidenceTier(
                f"node {self.id!r} cannot carry evidence_tier={tier!r}"
            )
        if tier not in EVIDENCE_TIERS:
            raise IllegalEvidenceTier(
                f"node {self.id!r} evidence_tier={tier!r} is not one of {list(EVIDENCE_TIERS)}"
            )
        self.evidence_tier = tier

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumes_source_tensor_identity": bool(self.assumes_source_tensor_identity),
            "backed_identity": self.backed_identity,
            "dense_weight_materialization": bool(self.dense_weight_materialization),
            "evidence_tier": self.evidence_tier,
            "id": self.id,
            "inputs": dict(sorted(self.inputs.items())),
            "kind": self.kind,
            "lifetime": self.lifetime,
            "mapping": self.mapping,
            "memory_tier": self.memory_tier,
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
            evidence_tier=str(d.get("evidence_tier") or "STATIC"),
            memory_tier=d.get("memory_tier"),
            backed_identity=d.get("backed_identity"),
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
    kernel: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        nodes = sorted((n.to_dict() for n in self.nodes), key=lambda n: n["id"])
        edges = sorted(
            (e.to_dict() for e in self.edges),
            key=lambda e: (e["src"], e["dst"], e["id"]),
        )
        body = {
            "device_budget": None if self.device_budget is None else self.device_budget.to_dict(),
            "edges": edges,
            "kernel": None if self.kernel is None else json.loads(canon_dumps(self.kernel)),
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
            kernel=None if not d.get("kernel") else dict(d.get("kernel") or {}),
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


def _as_repo_rel(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        return str(p).replace("\\", "/")
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def _git_json(rel: str) -> tuple[dict[str, Any], str] | None:
    """Load a JSON blob from git HEAD. Sparse checkouts leave receipts off disk."""
    blob = git("show", f"HEAD:{rel}")
    if not blob:
        return None
    try:
        doc = json.loads(blob)
    except json.JSONDecodeError:
        return None
    if not isinstance(doc, dict):
        return None
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return doc, digest


def _load_organ_doc(path: str | Path) -> tuple[dict[str, Any], str, str | None]:
    """Disk first, then git show HEAD:<rel>. Returns (doc, repo-rel, sha256)."""
    rel = _as_repo_rel(path)
    disk = Path(path)
    if not disk.is_file():
        disk = REPO / rel
    if disk.is_file():
        return load_json(disk), rel, sha256_file(disk)
    git_doc = _git_json(rel)
    if git_doc is not None:
        return git_doc[0], rel, git_doc[1]
    raise FileNotFoundError(f"organ map not found on disk or in git HEAD: {path}")


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
    doc, src_rel, _digest = _load_organ_doc(path)
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
        source_receipt=src_rel,
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
# Pre-board stack. PREHARDWARE. No board, no bitstream, no synthesis.
#
# Found and connected, not rewritten:
#   tools.future.fpga_engines.qgemv          — bit-exact FUNCTIONAL_SIM golden
#   tools.future.physical_primitives.instantiate — atlas primitive identity
#   DeviceBudget + validate() RESOURCE_OVER_BUDGET — already refused over-budget
#   fpga_fidelity.StructuralGraph            — sibling stand-in; not imported
#   hcli.agentos.fpga_preboard.HWIR          — schema-only; cannot edit
# ---------------------------------------------------------------------------

# Assumed coefficient table for STATIC resource estimates. Not a synth report,
# not a place-and-route result, not a board census. K.1 arithmetic tournament
# default: DSP MAC. Graded on ranking/refusal, not on invented Fmax.
LUT_PER_MAC_LANE = 64
LUT_DECODE_BASE = 512
LUT_DMA_BASE = 256
LUT_REDUCE_BASE = 128
BRAM_BLOCK_BITS = 36 * 1024  # UltraScale+ RAMB36 class size; STATIC vendor literature
URAM_BLOCK_BITS = 288 * 1024  # UltraRAM class size; STATIC vendor literature

# COST_MODEL fabric beats. Not clocks. Never converted to seconds.
# On-chip beat matches tools/future/fpga_fidelity.py MODEL_BYTES_PER_CYCLE = 64.
FABRIC_BYTES_PER_MODELLED_CYCLE = 64
# HBM beat is a declared planning coefficient, not a measured HBM2 rate.
HBM_BYTES_PER_MODELLED_CYCLE = 1024
# Host<->device beat is a USB4/Thunderbolt ~40 Gb/s CLASS prior from
# H-ROADMAP.md §15.1, turned into bytes/cycle against a notional fabric beat.
# It is a COST_MODEL prior. It is not a measurement of any cable.
# U50-family profiles that pin PCIe gen/lanes derive this beat from a declared
# payload-class mapping (Gen3 x16 = this unit). That mapping is also a
# COST_MODEL coefficient, not a slot measurement and not GB/s.
HOST_DEVICE_BYTES_PER_MODELLED_CYCLE = 16
HOST_DEVICE_QUEUE_CYCLES = 32
# Declared PCIe payload-class units relative to Gen3 x1. Gen4 is 2x Gen3
# (16 GT/s vs 8 GT/s, same 128b/130b encoding in vendor PCIe literature).
# Planning only: not a measured payload rate.
PCIE_GEN_PAYLOAD_UNIT = {3: 1, 4: 2}
PCIE_BEAT_REFERENCE_GEN = 3
PCIE_BEAT_REFERENCE_LANES = 16

# Wake-gated FPGA ledger rows. Recovered from civilization/CAPABILITY_GRAPH.json
# (hardware_blocker / wake_condition = U50_PRESENT). Twelve gates. Not a board.
U50_WAKE_GATES: tuple[str, ...] = (
    "U50_PURCHASE_ACCEPTANCE",
    "U50_SAFE_COOLING",
    "U50_DEVICE_PROFILE",
    "U50_DMA_HBM",
    "U50_FIRST_NATIVE_ENGINE",
    "U50_MIXED_APPLE_FPGA_GRAPH",
    "U50_34_TO_40",
    "U50_40_TO_50",
    "U50_50_TO_60",
    "U50_60_TO_70",
    "U50_70_TO_80",
    "U50_80_TO_90",
)

# Sealed prediction ids. Stable names so scoring and the rehearsal key on them.
PRED_HBM_BEAT = "u50.coeff.hbm_bytes_per_modelled_cycle"
PRED_FABRIC_BEAT = "u50.coeff.fabric_bytes_per_modelled_cycle"
PRED_HOST_BEAT = "u50.coeff.host_device_bytes_per_modelled_cycle"
PRED_HOST_QUEUE = "u50.coeff.host_device_queue_cycles"
PRED_CANON_HBM_BYTES = "qgemv.canonical.hbm_per_token_bytes"
PRED_CANON_HBM_CYCLES = "qgemv.canonical.hbm_modelled_cycles"
PRED_CANON_XFER_CYCLES = "qgemv.canonical.host_device_modelled_cycles_total"
PRED_CANON_CRIT_CYCLES = "qgemv.canonical.critical_path_modelled_cycles"
PRED_CANON_DSP = "qgemv.canonical.dsp_estimate"
PRED_CANON_LUT = "qgemv.canonical.lut_estimate"
PRED_PLAN_HBM_BYTES = "qgemv.planning.hbm_per_token_bytes"
PRED_PLAN_HBM_CYCLES = "qgemv.planning.hbm_modelled_cycles"

# Synthetic-rehearsal divergence. Fabricated. Not an arrival, not a board.
# HBM beat 1024 -> 256 (outside 50% relative). Fabric beat 64 -> 8.
# Host beat is the CONFIRMED control (left at the declared 16).
SYNTHETIC_DIVERGENT_HBM_BEAT = 256
SYNTHETIC_DIVERGENT_FABRIC_BEAT = 8

TOLERANCE_KINDS = ("relative", "absolute")
SCORE_VERDICTS = ("CONFIRMED", "FALSIFIED", "UNPINNED", "REFUSED")

# §15.5 Apple/FPGA split is "only a measured-bandwidth prior". We store it as
# a COST_MODEL prior and never call it a measurement.
APPLE_FPGA_PRIOR_NUM = 65
APPLE_FPGA_PRIOR_DEN = 100


@dataclass(frozen=True)
class DeviceProfile:
    """Synthetic planning envelope. Declared, not measured, not a board census.

    U50-class LUT/DSP/BRAM/URAM/HBM counts are vendor-literature envelopes
    (AMD Alveo U50 / VU35P product brief class). They are STATIC planning
    numbers. They are not HARDWARE_MEASURED and they are not a local census.

    Family variants (see u50_family_profile) add PCIe, power, cooling, and
    form-factor fields with per-field provenance. A CarrierEnvelope can
    DOWNGRADE this envelope; the result is a COST_MODEL overlay, not a
    measurement of any slot.
    """

    device_id: str
    origin: str
    LUT: int
    DSP: int
    BRAM: int
    URAM: int
    hbm_channels: int
    hbm_capacity_bytes: int
    mac_lanes_default: int = 8
    pipeline_depth: int = 8
    initiation_interval: int = 1
    fabric_bytes_per_modelled_cycle: int = FABRIC_BYTES_PER_MODELLED_CYCLE
    hbm_bytes_per_modelled_cycle: int = HBM_BYTES_PER_MODELLED_CYCLE
    host_device_bytes_per_modelled_cycle: int = HOST_DEVICE_BYTES_PER_MODELLED_CYCLE
    declared_not_measured: bool = True
    vendor_literature: str = (
        "AMD Alveo U50 / XCU50 / VU35P class envelope from public product "
        "literature. SYNTHETIC planning profile. Not a local board census."
    )
    variant_id: str | None = None
    sku: str | None = None
    fpga_part: str | None = None
    pcie_generation: int | None = None
    pcie_lanes: int | None = None
    power_envelope_w: int | None = None
    cooling: str | None = None
    form_factor: str | None = None
    airflow_requirement: str | None = None
    field_provenance: Mapping[str, Any] = field(default_factory=dict)
    constrained_by_carrier: str | None = None
    brochure_device_id: str | None = None
    thermal_mismatch: bool = False
    mechanically_inadmissible: bool = False

    def to_dict(self) -> dict[str, Any]:
        tier = "COST_MODEL" if self.constrained_by_carrier else "STATIC"

        def _opt_int(value: int | None) -> Any:
            return UNPINNED if value is None else int(value)

        def _opt_str(value: str | None) -> Any:
            return UNPINNED if value is None else value

        return emit_evidence(
            tier,
            {
                "BRAM": int(self.BRAM),
                "DSP": int(self.DSP),
                "LUT": int(self.LUT),
                "URAM": int(self.URAM),
                "airflow_requirement": _opt_str(self.airflow_requirement),
                "brochure_device_id": self.brochure_device_id,
                "constrained_by_carrier": self.constrained_by_carrier,
                "cooling": _opt_str(self.cooling),
                "declared_not_measured": True,
                "device_id": self.device_id,
                "fabric_bytes_per_modelled_cycle": int(self.fabric_bytes_per_modelled_cycle),
                "field_provenance": {
                    str(k): dict(v) for k, v in dict(self.field_provenance).items()
                },
                "form_factor": _opt_str(self.form_factor),
                "fpga_part": _opt_str(self.fpga_part),
                "hbm_bytes_per_modelled_cycle": int(self.hbm_bytes_per_modelled_cycle),
                "hbm_capacity_bytes": int(self.hbm_capacity_bytes),
                "hbm_channels": int(self.hbm_channels),
                "host_device_bytes_per_modelled_cycle": int(
                    self.host_device_bytes_per_modelled_cycle
                ),
                "initiation_interval": int(self.initiation_interval),
                "kind": "SYNTHETIC_DEVICE_PROFILE",
                "mac_lanes_default": int(self.mac_lanes_default),
                "mechanically_inadmissible": bool(self.mechanically_inadmissible),
                "origin": self.origin,
                "pcie_generation": _opt_int(self.pcie_generation),
                "pcie_lanes": _opt_int(self.pcie_lanes),
                "pipeline_depth": int(self.pipeline_depth),
                "power_envelope_w": _opt_int(self.power_envelope_w),
                "real_carrier": UNPINNED,
                "real_carrier_note": REAL_CARRIER_NOTE,
                "sku": _opt_str(self.sku),
                "thermal_mismatch": bool(self.thermal_mismatch),
                "variant_id": self.variant_id,
                "vendor_literature": self.vendor_literature,
            },
        )

    def budget(self) -> DeviceBudget:
        return DeviceBudget(
            BRAM=int(self.BRAM),
            DSP=int(self.DSP),
            LUT=int(self.LUT),
            URAM=int(self.URAM),
            device_id=self.device_id,
            hbm_channels=int(self.hbm_channels),
            declared_not_measured=True,
            status="DECLARED_COMPILER_CONSTRAINT",
        )

    def resource_map(self) -> dict[str, int]:
        return {k: int(getattr(self, k)) for k in RESOURCE_CLASSES}


def synthetic_u50_class() -> DeviceProfile:
    """Textbook U50-class envelope from §15. Not a board we have."""
    return DeviceProfile(
        device_id="synthetic-u50-class",
        origin="SYNTHETIC_U50_CLASS_DECLARED_NOT_A_BOARD",
        LUT=872_000,
        DSP=9_024,
        BRAM=2_016,
        URAM=960,
        hbm_channels=32,
        hbm_capacity_bytes=8 * 1024 ** 3,
    )


def synthetic_device(
    *,
    lut: int,
    dsp: int,
    bram: int,
    uram: int,
    hbm_channels: int = 1,
    hbm_capacity_bytes: int = 64 * 1024,
    device_id: str = "synthetic-tiny-declared-not-a-board",
    pipeline_depth: int = 8,
    initiation_interval: int = 1,
) -> DeviceProfile:
    """Caller-declared planning envelope. Used by overflow tests. Not a board."""
    return DeviceProfile(
        device_id=device_id,
        origin="SYNTHETIC_DECLARED_NOT_A_BOARD",
        LUT=int(lut),
        DSP=int(dsp),
        BRAM=int(bram),
        URAM=int(uram),
        hbm_channels=int(hbm_channels),
        hbm_capacity_bytes=int(hbm_capacity_bytes),
        pipeline_depth=int(pipeline_depth),
        initiation_interval=int(initiation_interval),
        vendor_literature="caller-declared test/planning envelope; not vendor literature",
    )


# ---------------------------------------------------------------------------
# U50-family variants + CarrierEnvelope
#
# The generic synthetic_u50_class() envelope is unchanged (LUT 872000 / DSP
# 9024 / BRAM 2016 / URAM 960 / 32 HBM channels / 8 GiB). That mixed class
# figure is NOT rewritten. Family SKUs below are sourced per-field from
# public vendor literature, or explicitly UNPINNED.
# ---------------------------------------------------------------------------

U50_FAMILY_VARIANT_IDS = ("u50", "u50c", "u50dd", "u50lv")

# Fields that must be either sourced-with-provenance or explicitly UNPINNED
# on every family variant. No silent defaults.
U50_VARIANT_REQUIRED_FIELDS = (
    "LUT",
    "DSP",
    "BRAM",
    "URAM",
    "hbm_channels",
    "hbm_capacity_bytes",
    "pcie_generation",
    "pcie_lanes",
    "power_envelope_w",
    "cooling",
    "form_factor",
    "sku",
    "fpga_part",
)


def sourced_field(
    value: Any,
    document_class: str,
    citation: str,
    note: str = "",
    *,
    evidence_tier: str = "STATIC",
) -> dict[str, Any]:
    """Pin a field to a public document class. STATIC vendor literature, not a measurement."""
    if value is None or value == UNPINNED:
        raise ValueError("sourced_field requires a pinned value; use unpinned_field otherwise")
    if not document_class or document_class == UNPINNED:
        raise ValueError("sourced_field requires a document class")
    if not citation:
        raise ValueError("sourced_field requires a citation")
    if evidence_tier not in EVIDENCE_TIERS:
        raise IllegalEvidenceTier(f"provenance evidence_tier={evidence_tier!r}")
    if evidence_tier == "HARDWARE_MEASURED":
        raise IllegalEvidenceTier("provenance must not claim HARDWARE_MEASURED")
    return {
        "citation": citation,
        "document_class": document_class,
        "evidence_tier": evidence_tier,
        "hardware_measured": False,
        "note": note,
        "pinned": True,
        "value": value,
        "vendor_literature_not_measurement": True,
    }


def unpinned_field(reason: str, *, document_class: str = DOC_UNPINNED) -> dict[str, Any]:
    """Explicit gap. A profile that names three unknown fields beats one that invents them."""
    if not reason:
        raise ValueError("unpinned_field requires a reason")
    return {
        "citation": "",
        "document_class": document_class,
        "evidence_tier": "STATIC",
        "hardware_measured": False,
        "note": reason,
        "pinned": False,
        "value": UNPINNED,
        "vendor_literature_not_measurement": True,
    }


def assert_variant_provenance(profile: DeviceProfile) -> None:
    """Every required family field is sourced-with-provenance or explicitly UNPINNED."""
    prov = dict(profile.field_provenance)
    missing = [n for n in U50_VARIANT_REQUIRED_FIELDS if n not in prov]
    if missing:
        raise ValueError(f"{profile.device_id}: missing provenance for {missing}")
    for name in U50_VARIANT_REQUIRED_FIELDS:
        meta = dict(prov[name])
        pinned = bool(meta.get("pinned"))
        value = meta.get("value")
        if pinned:
            if value is None or value == UNPINNED:
                raise ValueError(f"{profile.device_id}.{name}: pinned field has no value")
            if not meta.get("document_class") or meta.get("document_class") == UNPINNED:
                raise ValueError(f"{profile.device_id}.{name}: pinned field needs a document class")
            if not meta.get("citation"):
                raise ValueError(f"{profile.device_id}.{name}: pinned field needs a citation")
        else:
            if value != UNPINNED:
                raise ValueError(
                    f"{profile.device_id}.{name}: unpinned field must be UNPINNED, not {value!r}"
                )
            if not meta.get("note"):
                raise ValueError(f"{profile.device_id}.{name}: unpinned field needs a reason")
        if meta.get("evidence_tier") == "HARDWARE_MEASURED":
            raise IllegalEvidenceTier(f"{profile.device_id}.{name} claimed HARDWARE_MEASURED")
        if meta.get("hardware_measured") not in {False, None, 0}:
            raise IllegalEvidenceTier(f"{profile.device_id}.{name} hardware_measured must be false")


def pcie_payload_beat(generation: int, lanes: int) -> int:
    """COST_MODEL host<->device bytes/cycle from a declared PCIe payload class.

    Gen3 x16 maps onto HOST_DEVICE_BYTES_PER_MODELLED_CYCLE so we do not
    introduce a new invented rate. Smaller gen/lanes produce a strictly
    smaller beat. Not GB/s, not a slot measurement, not HARDWARE_MEASURED.
    """
    gen = int(generation)
    width = int(lanes)
    unit = PCIE_GEN_PAYLOAD_UNIT.get(gen)
    if unit is None:
        raise ValueError(
            f"no declared PCIe payload-class unit for generation={gen}; "
            "planning mapping covers Gen3 and Gen4 only (DS965 PCIe table)"
        )
    if width < 1:
        raise ValueError("pcie lanes must be >= 1")
    return max(
        1,
        HOST_DEVICE_BYTES_PER_MODELLED_CYCLE
        * unit
        * width
        // (PCIE_GEN_PAYLOAD_UNIT[PCIE_BEAT_REFERENCE_GEN] * PCIE_BEAT_REFERENCE_LANES),
    )


def _pinned_value(fields: Mapping[str, Mapping[str, Any]], name: str) -> Any:
    meta = fields[name]
    if not meta.get("pinned") or meta.get("value") == UNPINNED:
        return None
    return meta["value"]


def _profile_from_fields(
    *,
    device_id: str,
    origin: str,
    variant_id: str,
    fields: Mapping[str, Mapping[str, Any]],
    vendor_literature: str,
    mac_lanes_default: int = 8,
    pipeline_depth: int = 8,
    initiation_interval: int = 1,
) -> DeviceProfile:
    lut = _pinned_value(fields, "LUT")
    dsp = _pinned_value(fields, "DSP")
    bram = _pinned_value(fields, "BRAM")
    uram = _pinned_value(fields, "URAM")
    hbm_ch = _pinned_value(fields, "hbm_channels")
    hbm_cap = _pinned_value(fields, "hbm_capacity_bytes")
    gen = _pinned_value(fields, "pcie_generation")
    lanes = _pinned_value(fields, "pcie_lanes")
    if gen is not None and lanes is not None:
        beat = pcie_payload_beat(int(gen), int(lanes))
    else:
        beat = HOST_DEVICE_BYTES_PER_MODELLED_CYCLE
    profile = DeviceProfile(
        device_id=device_id,
        origin=origin,
        LUT=0 if lut is None else int(lut),
        DSP=0 if dsp is None else int(dsp),
        BRAM=0 if bram is None else int(bram),
        URAM=0 if uram is None else int(uram),
        hbm_channels=0 if hbm_ch is None else int(hbm_ch),
        hbm_capacity_bytes=0 if hbm_cap is None else int(hbm_cap),
        mac_lanes_default=int(mac_lanes_default),
        pipeline_depth=int(pipeline_depth),
        initiation_interval=int(initiation_interval),
        host_device_bytes_per_modelled_cycle=int(beat),
        vendor_literature=vendor_literature,
        variant_id=variant_id,
        sku=_pinned_value(fields, "sku"),
        fpga_part=_pinned_value(fields, "fpga_part"),
        pcie_generation=None if gen is None else int(gen),
        pcie_lanes=None if lanes is None else int(lanes),
        power_envelope_w=(
            None
            if _pinned_value(fields, "power_envelope_w") is None
            else int(_pinned_value(fields, "power_envelope_w"))
        ),
        cooling=_pinned_value(fields, "cooling"),
        form_factor=_pinned_value(fields, "form_factor"),
        airflow_requirement=_pinned_value(fields, "airflow_requirement")
        if "airflow_requirement" in fields
        else None,
        field_provenance={str(k): dict(v) for k, v in fields.items()},
        brochure_device_id=device_id,
    )
    assert_variant_provenance(profile)
    assert_no_hardware_measured(profile.to_dict())
    return profile


def _u50_common_hbm_capacity() -> dict[str, Any]:
    return sourced_field(
        8 * 1024 ** 3,
        DOC_DS965,
        "DS965 Table 1 HBM2 total capacity 8 GB",
        note=(
            "STATIC vendor-literature capacity. Planning uses 8 * 1024**3 bytes "
            "to match the existing synthetic_u50_class convention. Not a "
            "measured occupancy. DS965 peak/nominal GB/s figures are vendor "
            "literature and are NOT copied into a planning rate."
        ),
    )


def _u50_common_resources_ds965(column: str) -> dict[str, dict[str, Any]]:
    """LUT/DSP/BRAM/URAM shared by U50 production, U50LV, and U50DD ES in DS965."""
    cite = f"DS965 Table 1, {column}"
    return {
        "LUT": sourced_field(872_000, DOC_DS965, cite + " Look-up tables (LUTs) 872K"),
        "DSP": sourced_field(5_952, DOC_DS965, cite + " DSP slices 5,952"),
        "BRAM": sourced_field(
            1_344,
            DOC_DS965,
            cite + " 36 Kb block RAM 1344 (47.3 Mb)",
        ),
        "URAM": sourced_field(
            640,
            DOC_DS965,
            cite + " 288 Kb UltraRAM 640 (180.0 Mb)",
        ),
        "hbm_capacity_bytes": _u50_common_hbm_capacity(),
        "hbm_channels": sourced_field(
            32,
            DOC_UG1371,
            "UG1371 HBM Memory / Card Features: 32 AXI interfaces; 32 channels of 256 MB",
            note="STATIC vendor-literature channel count. Not a local census.",
        ),
        "power_envelope_w": sourced_field(
            75,
            DOC_DS965,
            cite + " Total electrical card load 75W",
            note=(
                "STATIC datasheet total electrical card load. Not a measured "
                "board wattage. DS965 also notes a 10W HBM rail limit from the "
                "PCIe 3.3V rail; that rail is recorded as literature, not used "
                "as a fabricated HBM GB/s derate."
            ),
        ),
        "cooling": sourced_field(
            "passive",
            DOC_DS965,
            cite + " Thermal cooling solution Passive",
        ),
        "form_factor": sourced_field(
            "half_height_half_length_single_slot",
            DOC_DS965,
            cite + " Form factor Half height, half length; single slot low profile",
        ),
        "airflow_requirement": sourced_field(
            "forced_server",
            DOC_DS965,
            "DS965 Summary: passively-cooled card designed for server deployment",
            note=(
                "Passive cards in this family expect server forced airflow. "
                "A carrier with airflow_class='none' is a thermal mismatch; "
                "the brochure 75W envelope is then not the planning envelope."
            ),
        ),
    }


def _u50_production_fields() -> dict[str, dict[str, Any]]:
    fields = _u50_common_resources_ds965("U50 Production (A-U50-P00G-PQ-G)")
    fields["sku"] = sourced_field(
        "A-U50-P00G-PQ-G",
        DOC_DS965,
        "DS965 Table 1 Product SKU A-U50-P00G-PQ-G",
    )
    fields["fpga_part"] = sourced_field(
        "XCU50-FSVH2104-2-E",
        DOC_UG1371,
        "UG1371 UltraScale+ Device: XCU50-FSVH2104-2-E for U50",
    )
    fields["pcie_generation"] = sourced_field(
        3,
        DOC_DS965,
        "DS965 Table 1 PCIe interface Gen3 x16, Gen4 x8, CCIX",
        note=(
            "Planning pins Gen3 as the Table 1 generation used with x16. "
            "Gen4 x8 is the same declared payload class (pcie_payload_beat). "
            "Not a measured link training result."
        ),
    )
    fields["pcie_lanes"] = sourced_field(
        16,
        DOC_DS965,
        "DS965 Table 1 PCIe interface Gen3 x16, Gen4 x8, CCIX",
        note="Planning uses the Gen3 x16 listing. Dual Gen4 x8 is the same payload class.",
    )
    return fields


def _u50lv_fields() -> dict[str, dict[str, Any]]:
    fields = _u50_common_resources_ds965("U50 LV Production (A-U50-P00G-LV-G)")
    fields["sku"] = sourced_field(
        "A-U50-P00G-LV-G",
        DOC_DS965,
        "DS965 Table 1 Product SKU A-U50-P00G-LV-G",
    )
    fields["fpga_part"] = sourced_field(
        "XCU50-FSVH2104-2LV-E",
        DOC_UG1371,
        "UG1371 UltraScale+ Device: XCU50-FSVH2104-2LV-E for U50LV (VLOW 0.72V)",
    )
    fields["pcie_generation"] = sourced_field(
        3,
        DOC_DS965,
        "DS965 Table 1 U50 LV PCIe interface Gen3 x16 (no Gen4 in that column)",
        note="U50LV is VLOW; DS965 does not list Gen4 for this SKU.",
    )
    # Honest gap: Table 1 says Gen3 x16; a DS965 PCIe note and UG1120 Vitis
    # platform say Gen3 x4 at VLOW. Do not pick one by interpolation.
    fields["pcie_lanes"] = unpinned_field(
        "DS965 Table 1 lists Gen3 x16 for U50 LV; DS965 PCIe section notes "
        "the U50 LV card only supports PCIe Gen3 x4 with VCCINT set to VLOW; "
        "UG1120 Vitis platform for U50LV is Gen3 x4 XDMA. Both figures are "
        "public vendor literature. Lanes are UNPINNED rather than interpolated."
    )
    return fields


def _u50dd_fields() -> dict[str, dict[str, Any]]:
    # Older DS965 tables (v1.2 era) list a U50DD ES3 column alongside U50
    # production. Later DS965 (v1.8) dropped that column. Numbers below are
    # from the ES3 column, not copied from production by interpolation.
    cite = "DS965 (v1.2-era Table 1) U50DD ES3 column (A-U50DD-P00G-ES3-G)"
    fields = {
        "LUT": sourced_field(872_000, DOC_DS965, cite + " Look-up tables (LUTs) 872K"),
        "DSP": sourced_field(5_952, DOC_DS965, cite + " DSP slices 5,952"),
        "BRAM": sourced_field(1_344, DOC_DS965, cite + " 36 Kb block RAM 1344"),
        "URAM": sourced_field(640, DOC_DS965, cite + " 288 Kb UltraRAM 640"),
        "hbm_capacity_bytes": sourced_field(
            8 * 1024 ** 3,
            DOC_DS965,
            cite + " HBM2 total capacity 8 GB",
            note="STATIC vendor-literature capacity. Not a measured occupancy.",
        ),
        "hbm_channels": sourced_field(
            32,
            DOC_UG1371,
            "UG1371 Card Features (covers ES3 SKU in Table 1): 32 channels of 256 MB",
        ),
        "power_envelope_w": sourced_field(75, DOC_DS965, cite + " Total electrical card load 75W"),
        "cooling": sourced_field("passive", DOC_DS965, cite + " Thermal cooling solution Passive"),
        "form_factor": sourced_field(
            "half_height_half_length_single_slot",
            DOC_DS965,
            cite + " Form factor Half height, half length",
        ),
        "airflow_requirement": sourced_field(
            "forced_server",
            DOC_DS965,
            "DS965: passively-cooled card; ES3 listed in the same family table",
        ),
        "sku": sourced_field(
            "A-U50DD-P00G-ES3-G",
            DOC_DS965,
            cite + " Product SKU; product selection guide: engineering sample, not volume production",
        ),
        "fpga_part": sourced_field(
            "XCU50",
            DOC_UG1371,
            "UG1371 Card Features: UltraScale+ XCU50 FPGA (ES3 listed in the SKU table)",
            note="Speed grade for the ES3 SKU is UNPINNED; not copied from production U50.",
        ),
        "pcie_generation": sourced_field(
            3,
            DOC_DS965,
            cite + " PCIe interface Gen3 x16, Gen4 x8, CCIX",
            note="Planning pins Gen3 with x16; Gen4 x8 is the same payload class.",
        ),
        "pcie_lanes": sourced_field(
            16,
            DOC_DS965,
            cite + " PCIe interface Gen3 x16, Gen4 x8, CCIX",
        ),
    }
    return fields


def _u50c_fields() -> dict[str, dict[str, Any]]:
    reason = (
        "Public AMD/Xilinx U50-family literature sourced for this module "
        "(DS965 Table 1, UG1371, Alveo U50 product brief, Alveo product "
        "selection guide) names U50, U50LV, and U50DD (ES SKU "
        "A-U50DD-P00G-ES3-G). No distinct U50C SKU, resource table, HBM "
        "capacity, PCIe listing, power, cooling, or form factor was found. "
        "Not interpolated from U50. Not copied from Alveo U55C (different "
        "card, 16 GB HBM in UG1120)."
    )
    return {name: unpinned_field(reason) for name in U50_VARIANT_REQUIRED_FIELDS} | {
        "airflow_requirement": unpinned_field(reason),
    }


def u50_family_profile(variant: str) -> DeviceProfile:
    """Selectable U50-family DeviceProfile. STATIC vendor literature, not a board."""
    key = str(variant).strip().lower().replace("_", "").replace("-", "")
    aliases = {
        "u50": "u50",
        "alveou50": "u50",
        "au50p00gpqg": "u50",
        "u50c": "u50c",
        "alveou50c": "u50c",
        "u50dd": "u50dd",
        "alveou50dd": "u50dd",
        "au50ddp00ges3g": "u50dd",
        "u50lv": "u50lv",
        "alveou50lv": "u50lv",
        "au50p00glvg": "u50lv",
    }
    vid = aliases.get(key)
    if vid is None:
        raise ValueError(
            f"unknown U50-family variant {variant!r}; "
            f"known: {list(U50_FAMILY_VARIANT_IDS)}"
        )
    if vid == "u50":
        return _profile_from_fields(
            device_id="alveo-u50",
            origin="U50_FAMILY_VARIANT_DECLARED_NOT_A_BOARD",
            variant_id="u50",
            fields=_u50_production_fields(),
            vendor_literature=(
                "AMD Alveo U50 production SKU A-U50-P00G-PQ-G from public "
                "DS965 / UG1371. STATIC vendor literature. Not a local board."
            ),
        )
    if vid == "u50lv":
        return _profile_from_fields(
            device_id="alveo-u50lv",
            origin="U50_FAMILY_VARIANT_DECLARED_NOT_A_BOARD",
            variant_id="u50lv",
            fields=_u50lv_fields(),
            vendor_literature=(
                "AMD Alveo U50LV production SKU A-U50-P00G-LV-G from public "
                "DS965 / UG1371. Identical to U50 except VCCINT VLOW. "
                "STATIC vendor literature. Not a local board."
            ),
        )
    if vid == "u50dd":
        return _profile_from_fields(
            device_id="alveo-u50dd",
            origin="U50_FAMILY_VARIANT_DECLARED_NOT_A_BOARD",
            variant_id="u50dd",
            fields=_u50dd_fields(),
            vendor_literature=(
                "AMD Alveo U50DD engineering sample SKU A-U50DD-P00G-ES3-G "
                "from public DS965 ES3 column / UG1371. Not qualified for "
                "volume deployment. STATIC vendor literature. Not a local board."
            ),
        )
    return _profile_from_fields(
        device_id="alveo-u50c",
        origin="U50_FAMILY_VARIANT_DECLARED_NOT_A_BOARD",
        variant_id="u50c",
        fields=_u50c_fields(),
        vendor_literature=(
            "No public AMD U50C SKU table was sourced. Profile exists so the "
            "name is selectable; every required field is UNPINNED. Not U55C."
        ),
    )


def list_u50_family_profiles() -> dict[str, DeviceProfile]:
    return {vid: u50_family_profile(vid) for vid in U50_FAMILY_VARIANT_IDS}


def select_device_profile(name: str) -> DeviceProfile:
    """Select the generic class envelope or a U50-family variant."""
    key = str(name).strip().lower().replace("_", "-")
    if key in {"synthetic-u50-class", "u50-class", "synthetic_u50_class".replace("_", "-")}:
        return synthetic_u50_class()
    return u50_family_profile(name)


@dataclass(frozen=True)
class CarrierEnvelope:
    """Host-side bound on a card. Declared planning envelope, not a board census.

    A card is bounded by what its host can actually give it: PCIe generation
    and lane count, sustained power delivery, thermal/airflow class, and any
    mechanical limit. constrain() DOWNGRADES a DeviceProfile; the planner
    must see the reduced envelope, not the brochure one.

    The inbound comma-device carrier is UNPINNED. Labeled example envelopes
    exist so the real one can be pinned in a single constructor call.
    """

    carrier_id: str
    origin: str
    pcie_generation: int | None
    pcie_lanes: int | None
    sustained_power_w: int | None
    airflow_class: str | None
    mechanical_limit: str | None
    note: str
    example: bool = False
    field_provenance: Mapping[str, Any] = field(default_factory=dict)
    # A bridge/completer can cap payload BELOW what its lane width implies.
    # Chestnut measures ~the same at Gen3 x4 as at Gen3 x2, so the ceiling is a
    # property of the ASM2464PD completer, not of the lane count -- it cannot be
    # derived from pcie_generation/pcie_lanes and must be stated.
    observed_payload_beat: int | None = None
    observed_payload_bytes_per_s: int | None = None

    def to_dict(self) -> dict[str, Any]:
        def _opt_int(value: int | None) -> Any:
            return UNPINNED if value is None else int(value)

        def _opt_str(value: str | None) -> Any:
            return UNPINNED if value is None else value

        return emit_evidence(
            "STATIC",
            {
                "airflow_class": _opt_str(self.airflow_class),
                "carrier_id": self.carrier_id,
                "example": bool(self.example),
                "field_provenance": {
                    str(k): dict(v) for k, v in dict(self.field_provenance).items()
                },
                "kind": "CARRIER_ENVELOPE",
                "mechanical_limit": _opt_str(self.mechanical_limit),
                "note": self.note,
                "origin": self.origin,
                "pcie_generation": _opt_int(self.pcie_generation),
                "pcie_lanes": _opt_int(self.pcie_lanes),
                "real_carrier": UNPINNED,
                "real_carrier_note": REAL_CARRIER_NOTE,
                "observed_payload_beat": _opt_int(self.observed_payload_beat),
                "observed_payload_bytes_per_s": _opt_int(self.observed_payload_bytes_per_s),
                "sustained_power_w": _opt_int(self.sustained_power_w),
            },
        )

    def constrain(self, device: DeviceProfile) -> DeviceProfile:
        return constrain_device_profile(device, self)


def _min_optional(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return left if left <= right else right


def _scale_int(value: int, numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("power derate denominator must be > 0")
    if numerator < 0:
        raise ValueError("power derate numerator must be >= 0")
    return (int(value) * int(numerator)) // int(denominator)


def _downgrade_provenance(
    original: Mapping[str, Any],
    *,
    value: Any,
    citation: str,
    note: str,
    evidence_tier: str,
) -> dict[str, Any]:
    base = dict(original) if original else {}
    brochure = base.get("value", UNPINNED)
    return {
        "brochure_value": brochure,
        "citation": citation,
        "document_class": DOC_CARRIER_DOWNGRADE,
        "evidence_tier": evidence_tier,
        "hardware_measured": False,
        "note": note,
        "pinned": value != UNPINNED and value is not None,
        "value": UNPINNED if value is None else value,
        "vendor_literature_not_measurement": True,
    }


# ---------------------------------------------------------------------------
# comma "Tiny Chestnut" — the REAL inbound carrier, pinned 2026-09-02.
#
# EVIDENCE PROVENANCE, and this distinction is load-bearing: these figures are
# THIRD_PARTY_REPORTED (tinygrad's open ASM2464PD firmware source/README and
# comma product pages). They are NOT AMD vendor literature and they are NOT a
# Hawking measurement. Hardware Doctor overwrites them on U50_PRESENT; until
# then nothing here may be emitted as HARDWARE_MEASURED.
#
# The decisive number: U50 HBM ~316 GB/s against a ~1.68 GB/s bridge is ~188x.
# The FPGA drains local HBM ~188x faster than the bridge can refill it from the
# Mac, which is why roadmap J.7 residency is not a preference but the only
# viable thesis behind this carrier.
# ---------------------------------------------------------------------------

CHESTNUT_THIRD_PARTY = "THIRD_PARTY_REPORTED"
CHESTNUT_BRIDGE = "ASMedia ASM2464PD"
CHESTNUT_SOURCE = (
    "tinygrad/asm2464pd-firmware source + README, and comma product pages. "
    "Third-party reported; not vendor literature, not a Hawking measurement."
)


def _chestnut_provenance(pinned: bool, note: str) -> dict[str, Any]:
    return {
        "pinned": bool(pinned),
        "tier": CHESTNUT_THIRD_PARTY,
        "source": CHESTNUT_SOURCE,
        "note": note,
    }


def chestnut_theoretical() -> CarrierEnvelope:
    """USB4 headline class. NOT achievable payload -- the ceiling before overhead."""
    return CarrierEnvelope(
        carrier_id="chestnut-theoretical",
        origin="CHESTNUT_THEORETICAL_THIRD_PARTY_REPORTED",
        pcie_generation=4,
        pcie_lanes=4,
        sustained_power_w=75,
        airflow_class="none",
        mechanical_limit=None,
        note=(
            "USB4 40 Gb/s host class, ~5.0 GB/s RAW before protocol overhead. "
            "Usable payload TBD. Do not plan against this number."
        ),
        example=False,
        field_provenance={
            "pcie": _chestnut_provenance(True, "ASM2464PD is Gen4-class, up to x4 in principle"),
            "payload": _chestnut_provenance(False, "usable payload UNPINNED at this tier"),
        },
    )


def chestnut_current_firmware() -> CarrierEnvelope:
    """What the carrier actually delivers today. This is the planning default."""
    return CarrierEnvelope(
        carrier_id="chestnut-current-fw",
        origin="CHESTNUT_CURRENT_FW_THIRD_PARTY_REPORTED",
        pcie_generation=3,
        pcie_lanes=2,
        sustained_power_w=75,
        airflow_class="none",
        mechanical_limit=None,
        note=(
            "tinygrad firmware deliberately trains PCIe Gen3 x2. Observed payload "
            "~1.66-1.68 GB/s, and ~the same at x4 because the ASM2464PD completer "
            "caps there -- so the ceiling is a COMPLETER property, not a lane count. "
            "12V/3.3V rails and PERST# are firmware-controlled; INA231 telemetry. "
            "Carrier supplies power and mechanical support but NO airflow."
        ),
        example=False,
        observed_payload_beat=2,
        observed_payload_bytes_per_s=1_680_000_000,
        field_provenance={
            "pcie": _chestnut_provenance(True, "firmware trains Gen3 x2"),
            "observed_payload": _chestnut_provenance(
                True, "~1.68 GB/s completer ceiling; same at x4 as at x2"
            ),
            "airflow": _chestnut_provenance(True, "carrier provides none; external airflow required"),
        },
    )


def chestnut_hawking_optimized() -> CarrierEnvelope:
    """Reserved for Hawking's OWN measurements. Deliberately UNPINNED."""
    return CarrierEnvelope(
        carrier_id="chestnut-hawking-optimized",
        origin="CHESTNUT_HAWKING_OPTIMIZED_UNPINNED",
        pcie_generation=None,
        pcie_lanes=None,
        sustained_power_w=75,
        airflow_class="none",
        mechanical_limit=None,
        note=(
            "Downstream width, payload bandwidth, latency and queue depth are "
            "BENCHMARK INPUTS to be written by Hardware Doctor on U50_PRESENT. "
            "Gen3 x2 is a firmware choice, not a physical limit: the gap between "
            "~1.68 GB/s observed and ~5.0 GB/s raw is open research. "
            "UNTIL MEASURED this mode inherits the CURRENT-FIRMWARE ceiling. An "
            "unpinned carrier must not license more optimism than the pinned one: "
            "planning as if an unmeasured bridge were a full server slot is exactly "
            "the fail-open default this envelope exists to prevent."
        ),
        example=False,
        observed_payload_beat=2,
        observed_payload_bytes_per_s=1_680_000_000,
        field_provenance={
            "everything": _chestnut_provenance(False, "awaiting Hawking measurement"),
            "ceiling_until_measured": _chestnut_provenance(
                True,
                "inherits chestnut-current-fw ~1.68 GB/s; raised only by a real measurement",
            ),
        },
    )


CHESTNUT_MODES = {
    "theoretical": chestnut_theoretical,
    "current-fw": chestnut_current_firmware,
    "hawking-optimized": chestnut_hawking_optimized,
}


def residency_ratio(local_bytes_consumed: int, carrier_bytes_transferred: int) -> float:
    """R = FPGA-local bytes consumed / carrier bytes transferred.

    The first-class admissibility metric behind a ~188x bandwidth cliff.
    R ~ 500 (4 MB sent to enable 2 GB of local consumption) is attractive.
    R ~ 1.5 (1 GB sent for 1.5 GB of local reads) means the architecture is
    doomed behind this carrier -- the accelerator has become expensive storage,
    which is exactly the failure roadmap J.7 names.
    """
    moved = int(carrier_bytes_transferred)
    if moved <= 0:
        raise ValueError("carrier_bytes_transferred must be > 0 to form a ratio")
    return float(local_bytes_consumed) / float(moved)


def constrain_device_profile(
    device: DeviceProfile,
    carrier: CarrierEnvelope,
) -> DeviceProfile:
    """DOWNGRADE a DeviceProfile to what the carrier can actually give it.

    Load-bearing: host<->device beat falls when PCIe gen/lanes fall;
    LUT/DSP/BRAM/URAM ceilings scale with delivered watts vs brochure watts
    (COST_MODEL linear derate, not a thermal simulation). A carrier must
    not raise any axis above the brochure envelope.

    Mutation target: CARRIER_ENVELOPE_BINDING. When False this returns the
    brochure profile unchanged and the refusal test must fail.
    """
    if not CARRIER_ENVELOPE_BINDING:
        return device

    brochure_id = device.brochure_device_id or device.device_id
    orig_prov = dict(device.field_provenance)

    eff_gen = _min_optional(device.pcie_generation, carrier.pcie_generation)
    eff_lanes = _min_optional(device.pcie_lanes, carrier.pcie_lanes)
    eff_power = _min_optional(device.power_envelope_w, carrier.sustained_power_w)

    thermal_mismatch = bool(
        device.cooling == "passive"
        and carrier.airflow_class == "none"
    )
    mechanically_inadmissible = bool(
        carrier.mechanical_limit == "incompatible"
        or (
            device.form_factor == "full_height_full_length"
            and carrier.mechanical_limit == "low_profile_ok"
        )
    )

    new_beat = int(device.host_device_bytes_per_modelled_cycle)
    if eff_gen is not None and eff_lanes is not None and eff_gen in PCIE_GEN_PAYLOAD_UNIT:
        derived = pcie_payload_beat(eff_gen, eff_lanes)
        # Never upgrade the brochure beat.
        new_beat = derived if derived <= new_beat else new_beat
    # An explicitly observed completer ceiling OUTRANKS the lane derivation,
    # because a bridge can cap below what its width implies. Cap only.
    if carrier.observed_payload_beat is not None:
        obs = int(carrier.observed_payload_beat)
        new_beat = obs if obs <= new_beat else new_beat

    lut, dsp, bram, uram = int(device.LUT), int(device.DSP), int(device.BRAM), int(device.URAM)
    power_derated = False
    if mechanically_inadmissible:
        lut = dsp = bram = uram = 0
        eff_power = 0 if eff_power is not None else 0
    elif (
        device.power_envelope_w is not None
        and device.power_envelope_w > 0
        and eff_power is not None
        and eff_power < device.power_envelope_w
    ):
        lut = _scale_int(device.LUT, eff_power, device.power_envelope_w)
        dsp = _scale_int(device.DSP, eff_power, device.power_envelope_w)
        bram = _scale_int(device.BRAM, eff_power, device.power_envelope_w)
        uram = _scale_int(device.URAM, eff_power, device.power_envelope_w)
        power_derated = True
    elif thermal_mismatch and carrier.sustained_power_w is None:
        # Passive card, no forced airflow, carrier watts UNPINNED: refuse to
        # claim the brochure envelope. Do not invent an unforced-airflow wattage.
        lut = dsp = bram = uram = 0
        eff_power = None

    new_prov = dict(orig_prov)
    if power_derated or mechanically_inadmissible or (thermal_mismatch and lut == 0):
        derate_note = (
            "COST_MODEL linear derate of brochure resources by "
            "min(carrier.sustained_power_w, device.power_envelope_w) / "
            "device.power_envelope_w. Not a synthesis result, not a thermal "
            "CFD, not HARDWARE_MEASURED."
        )
        if mechanically_inadmissible:
            derate_note = (
                "Mechanical limit refuses the brochure envelope. Resources "
                "zeroed rather than inventing a partial-fit. Not a measurement."
            )
        for name, val in (("LUT", lut), ("DSP", dsp), ("BRAM", bram), ("URAM", uram)):
            new_prov[name] = _downgrade_provenance(
                orig_prov.get(name, {}),
                value=val,
                citation="CarrierEnvelope.constrain COST_MODEL derate of STATIC brochure resources",
                note=derate_note,
                evidence_tier="COST_MODEL",
            )
        new_prov["power_envelope_w"] = _downgrade_provenance(
            orig_prov.get("power_envelope_w", {}),
            value=UNPINNED if eff_power is None else eff_power,
            citation="min(device.power_envelope_w, carrier.sustained_power_w)",
            note="Carrier power bound. Declared, not measured.",
            evidence_tier="COST_MODEL",
        )
    if eff_gen is not None or eff_lanes is not None:
        new_prov["pcie_generation"] = _downgrade_provenance(
            orig_prov.get("pcie_generation", {}),
            value=UNPINNED if eff_gen is None else eff_gen,
            citation="min(device.pcie_generation, carrier.pcie_generation)",
            note="Carrier PCIe generation bound. Not a trained link.",
            evidence_tier="COST_MODEL",
        )
        new_prov["pcie_lanes"] = _downgrade_provenance(
            orig_prov.get("pcie_lanes", {}),
            value=UNPINNED if eff_lanes is None else eff_lanes,
            citation="min(device.pcie_lanes, carrier.pcie_lanes)",
            note="Carrier PCIe lane bound. Not a trained link.",
            evidence_tier="COST_MODEL",
        )

    return replace(
        device,
        device_id=f"{device.device_id}@{carrier.carrier_id}",
        origin="CARRIER_CONSTRAINED_DEVICE_PROFILE",
        LUT=int(lut),
        DSP=int(dsp),
        BRAM=int(bram),
        URAM=int(uram),
        host_device_bytes_per_modelled_cycle=int(new_beat),
        pcie_generation=eff_gen,
        pcie_lanes=eff_lanes,
        power_envelope_w=eff_power,
        field_provenance=new_prov,
        constrained_by_carrier=carrier.carrier_id,
        brochure_device_id=brochure_id,
        thermal_mismatch=thermal_mismatch,
        mechanically_inadmissible=mechanically_inadmissible,
        vendor_literature=(
            device.vendor_literature
            + " Constrained by CarrierEnvelope "
            + carrier.carrier_id
            + ". COST_MODEL overlay on STATIC brochure figures. Not a slot census. "
            + REAL_CARRIER_NOTE
        ),
    )


def _example_carrier_provenance(
    *,
    gen: int,
    lanes: int,
    watts: int,
    airflow: str,
    mechanical: str,
) -> dict[str, dict[str, Any]]:
    note = (
        "LABELED EXAMPLE envelope, declared so a real carrier can be pinned "
        "later. Not a host census, not the comma-device carrier, not HARDWARE_MEASURED."
    )
    cite = "declared example CarrierEnvelope in tools.future.hwir"
    return {
        "pcie_generation": sourced_field(gen, DOC_EXAMPLE, cite, note=note),
        "pcie_lanes": sourced_field(lanes, DOC_EXAMPLE, cite, note=note),
        "sustained_power_w": sourced_field(watts, DOC_EXAMPLE, cite, note=note),
        "airflow_class": sourced_field(airflow, DOC_EXAMPLE, cite, note=note),
        "mechanical_limit": sourced_field(mechanical, DOC_EXAMPLE, cite, note=note),
    }


def example_full_airflow_server_slot() -> CarrierEnvelope:
    """Labeled EXAMPLE: a full-airflow 75W Gen4 x16 server slot.

    Not the comma-device carrier. Declared so constrain() has an unconstrained
    comparison point. STATIC example, not a measurement of any chassis.
    """
    return CarrierEnvelope(
        carrier_id="example-full-airflow-server-slot",
        origin="EXAMPLE_ENVELOPE_NOT_THE_REAL_CARRIER",
        pcie_generation=4,
        pcie_lanes=16,
        sustained_power_w=75,
        airflow_class="forced_server",
        mechanical_limit="full_height_full_length_ok",
        note=(
            "LABELED EXAMPLE. Full-airflow server slot matching a 75W PCIe "
            "card's own datasheet load. NOT the inbound comma-device carrier. "
            + REAL_CARRIER_NOTE
        ),
        example=True,
        field_provenance=_example_carrier_provenance(
            gen=4,
            lanes=16,
            watts=75,
            airflow="forced_server",
            mechanical="full_height_full_length_ok",
        ),
    )


def example_constrained_low_power_slot() -> CarrierEnvelope:
    """Labeled EXAMPLE: a constrained low-power few-lane slot.

    25W sustained, Gen3 x4, no forced airflow. NOT the comma-device carrier.
    """
    return CarrierEnvelope(
        carrier_id="example-constrained-low-power-slot",
        origin="EXAMPLE_ENVELOPE_NOT_THE_REAL_CARRIER",
        pcie_generation=3,
        pcie_lanes=4,
        sustained_power_w=25,
        airflow_class="none",
        mechanical_limit="low_profile_ok",
        note=(
            "LABELED EXAMPLE. Constrained low-power / few-lane slot. NOT the "
            "inbound comma-device carrier. "
            + REAL_CARRIER_NOTE
        ),
        example=True,
        field_provenance=_example_carrier_provenance(
            gen=3,
            lanes=4,
            watts=25,
            airflow="none",
            mechanical="low_profile_ok",
        ),
    )


def unpinned_real_carrier() -> CarrierEnvelope:
    """The inbound comma-device carrier. Every axis UNPINNED on purpose."""
    reason = REAL_CARRIER_NOTE
    fields = {
        name: unpinned_field(reason)
        for name in (
            "pcie_generation",
            "pcie_lanes",
            "sustained_power_w",
            "airflow_class",
            "mechanical_limit",
        )
    }
    return CarrierEnvelope(
        carrier_id="comma-device-real-carrier",
        origin="REAL_CARRIER_UNPINNED",
        pcie_generation=None,
        pcie_lanes=None,
        sustained_power_w=None,
        airflow_class=None,
        mechanical_limit=None,
        note=REAL_CARRIER_NOTE,
        example=False,
        field_provenance=fields,
    )


def select_carrier_envelope(name: str) -> CarrierEnvelope:
    key = str(name).strip().lower().replace("_", "-")
    aliases = {
        "full": example_full_airflow_server_slot,
        "full-airflow-server": example_full_airflow_server_slot,
        "example-full-airflow-server-slot": example_full_airflow_server_slot,
        "constrained": example_constrained_low_power_slot,
        "low-power": example_constrained_low_power_slot,
        "constrained-low-power": example_constrained_low_power_slot,
        "example-constrained-low-power-slot": example_constrained_low_power_slot,
        "unpinned": unpinned_real_carrier,
        "real": unpinned_real_carrier,
        "comma": unpinned_real_carrier,
        "comma-device": unpinned_real_carrier,
        "comma-device-real-carrier": unpinned_real_carrier,
    }
    factory = aliases.get(key)
    if factory is None:
        raise ValueError(
            f"unknown carrier envelope {name!r}; known: full, constrained, unpinned"
        )
    return factory()


def admissible_plan(
    kernel: "QGemvKernel",
    device: DeviceProfile,
    carrier: CarrierEnvelope | None = None,
) -> dict[str, Any]:
    """COST_MODEL comparison surface: constrained vs brochure envelopes.

    Returns the reduced envelope, host<->device beat, and whether the kernel
    is admitted. Not a board run.
    """
    profile = constrain_device_profile(device, carrier) if carrier is not None else device
    overflow: dict[str, Any] | None = None
    fit: dict[str, Any] | None = None
    ok = True
    try:
        fit = fit_kernel_to_device(kernel, profile)
    except ResourceOverBudget as exc:
        ok = False
        overflow = {
            "budget": dict(exc.budget),
            "device_id": exc.device_id,
            "overflow": {k: {"used": a, "budget": b} for k, (a, b) in exc.overflow.items()},
            "used": dict(exc.used),
        }
    xfer = None
    part = None
    if ok:
        xfer = model_host_device_transfer(kernel, profile)
        part = partition_qgemv(kernel, profile)
    return emit_evidence(
        "COST_MODEL",
        {
            "carrier_id": None if carrier is None else carrier.carrier_id,
            "constrained_by_carrier": profile.constrained_by_carrier,
            "device_id": profile.device_id,
            "fpga_rows": None if part is None else part["fpga_rows"],
            "host_device_bytes_per_modelled_cycle": int(
                profile.host_device_bytes_per_modelled_cycle
            ),
            "kind": "ADMISSIBLE_PLAN",
            "mechanically_inadmissible": bool(profile.mechanically_inadmissible),
            "modelled_cycles_total": None if xfer is None else xfer["modelled_cycles_total"],
            "note": (
                "COST_MODEL / STATIC planning comparison. Not a board run, "
                "not HARDWARE_MEASURED. Real comma-device carrier is UNPINNED."
            ),
            "ok": ok,
            "overflow": overflow,
            "pcie_generation": UNPINNED
            if profile.pcie_generation is None
            else int(profile.pcie_generation),
            "pcie_lanes": UNPINNED if profile.pcie_lanes is None else int(profile.pcie_lanes),
            "power_envelope_w": UNPINNED
            if profile.power_envelope_w is None
            else int(profile.power_envelope_w),
            "real_carrier": UNPINNED,
            "refused": not ok,
            "resource_budget": profile.resource_map(),
            "thermal_mismatch": bool(profile.thermal_mismatch),
        },
    )


@dataclass(frozen=True)
class QGemvKernel:
    """qGEMV-class kernel. Canonical first FPGA engine in §15.14 / APPENDIX K.2 L1."""

    M: int
    K: int
    weight_bits: int = 4
    group_size: int = 64
    mac_lanes: int = 8
    tile_m: int = 4
    organ: str = "qgemv"
    model: str = "qgemv-class"
    engine_ladder_level: str = "L1"
    arithmetic: str = "DSP_MAC"

    def __post_init__(self) -> None:
        if self.M < 1 or self.K < 1:
            raise ValueError("qGEMV M and K must be >= 1")
        if self.weight_bits < 1 or self.weight_bits > 8:
            raise ValueError("weight_bits must be in 1..8")
        if self.group_size < 1 or self.K % self.group_size != 0:
            raise ValueError(f"K={self.K} must be a positive multiple of group_size={self.group_size}")
        if self.mac_lanes < 1 or self.tile_m < 1:
            raise ValueError("mac_lanes and tile_m must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return emit_evidence(
            "STATIC",
            {
                "K": int(self.K),
                "M": int(self.M),
                "arithmetic": self.arithmetic,
                "engine_ladder_level": self.engine_ladder_level,
                "engine_ladder_meaning": (
                    "APPENDIX K.2 L1 = correct low-bit. Not L5 HBM-channel-"
                    "specialized, not a board engine."
                ),
                "group_size": int(self.group_size),
                "mac_lanes": int(self.mac_lanes),
                "model": self.model,
                "organ": self.organ,
                "tile_m": int(self.tile_m),
                "weight_bits": int(self.weight_bits),
            },
        )

    def weight_bytes(self) -> int:
        return _ceil_div(self.M * self.K * self.weight_bits, 8)

    def scale_bytes(self) -> int:
        return self.M * (self.K // self.group_size) * 4

    def activation_in_bytes(self) -> int:
        return self.K * 4

    def activation_out_bytes(self) -> int:
        return self.M * 4


def canonical_qgemv_kernel() -> QGemvKernel:
    """Matches tools.future.fpga_engines.VECTORS qgemv_hand. Exact in float32."""
    return QGemvKernel(
        M=2,
        K=4,
        weight_bits=4,
        group_size=4,
        mac_lanes=2,
        tile_m=2,
        organ="qgemv",
        model="qgemv-class",
        engine_ladder_level="L1",
        arithmetic="DSP_MAC",
    )


def planning_qgemv_kernel() -> QGemvKernel:
    """Declared planning-scale qGEMV for inbound-board COST_MODEL predictions.

    Not a measured workload and not a board trace. Sized so per-token HBM
    bytes exceed one HBM beat, so a wrong hbm_bytes_per_modelled_cycle can
    actually move modelled_cycles. The canonical L1 golden (M=2, K=4) cannot:
    24 bytes ceil-divides to 1 cycle for every beat >= 24.
    """
    return QGemvKernel(
        M=1024,
        K=4096,
        weight_bits=4,
        group_size=64,
        mac_lanes=8,
        tile_m=4,
        organ="qgemv",
        model="qgemv-class",
        engine_ladder_level="L1",
        arithmetic="DSP_MAC",
    )


def canonical_qgemv_operands() -> dict[str, Any]:
    """The qgemv_hand vector. Independent of the engine's own expected table."""
    return {
        "codes": [[1, -1, 2, 0], [3, 1, -2, 1]],
        "scales": [[2.0], [0.5]],
        "x": [1.0, 2.0, 3.0, 4.0],
        "expected": [10.0, 1.5],
        "source": "tools.future.fpga_engines.VECTORS[qgemv_hand]",
    }


def overflow_probe_kernel() -> QGemvKernel:
    """Engine wide enough that a tiny declared DSP budget must refuse it."""
    return QGemvKernel(
        M=32,
        K=256,
        weight_bits=4,
        group_size=64,
        mac_lanes=64,
        tile_m=16,
        organ="qgemv",
        model="qgemv-class",
        engine_ladder_level="L1",
        arithmetic="DSP_MAC",
    )


def brochure_fit_kernel() -> QGemvKernel:
    """Fits a sourced U50 brochure (5952 DSP / 75W); refused under a 25W derate.

    STATIC estimate uses mac_lanes * tile_m DSP. 64*64 = 4096, which is below
    DS965 U50 DSP 5952 and above 5952 * 25/75 = 1984. COST_MODEL derate of
    the brochure, not a synthesis result.
    """
    return QGemvKernel(
        M=32,
        K=256,
        weight_bits=4,
        group_size=64,
        mac_lanes=64,
        tile_m=64,
        organ="qgemv",
        model="qgemv-class",
        engine_ladder_level="L1",
        arithmetic="DSP_MAC",
    )


def _back_primitive(
    name: str,
    *,
    memory_tier: str,
    organ_class: str,
    extra: Mapping[str, Any] | None = None,
) -> Any:
    """CALL SITE: tools.future.physical_primitives.instantiate (not a mere import)."""
    from tools.future.physical_primitives import instantiate

    return instantiate(
        name,
        memory_tier=memory_tier,
        semantic_program_id="hwir.qgemv",
        backend="FPGA",
        organ_class=organ_class,
        extra=dict(extra or {}) | {"prehardware": True, "evidence_tier": "STATIC"},
    )


def estimate_qgemv_resources(kernel: QGemvKernel) -> dict[str, Any]:
    """STATIC LUT/DSP/BRAM/URAM ESTIMATE for a qGEMV engine. Not synthesis."""
    dsp = int(kernel.mac_lanes) * int(kernel.tile_m)
    lut = (
        dsp * LUT_PER_MAC_LANE
        + LUT_DECODE_BASE
        + LUT_DMA_BASE * 2
        + LUT_REDUCE_BASE
        + int(kernel.mac_lanes) * 8
    )
    tile_bits = (
        int(kernel.tile_m) * int(kernel.K) * int(kernel.weight_bits)
        + int(kernel.tile_m) * (int(kernel.K) // int(kernel.group_size)) * 32
    )
    bram = max(1, _ceil_div(tile_bits, BRAM_BLOCK_BITS))
    uram = 0
    if tile_bits > 4 * BRAM_BLOCK_BITS:
        uram = max(1, _ceil_div(tile_bits, URAM_BLOCK_BITS))
        bram = min(bram, 4)
    used = {"BRAM": int(bram), "DSP": int(dsp), "LUT": int(lut), "URAM": int(uram)}
    return emit_evidence(
        "STATIC",
        {
            "arithmetic": kernel.arithmetic,
            "coefficient_table": "v1-assumed-not-synthesized",
            "kind": "RESOURCE_ESTIMATE",
            "note": (
                "ESTIMATE from an assumed coefficient table. Not a synthesis "
                "report, not place-and-route, not a board measurement."
            ),
            "tile_bits": int(tile_bits),
            "used": used,
        },
    )


def resource_overflow(used: Mapping[str, int], budget: Mapping[str, int]) -> dict[str, tuple[int, int]]:
    overflow: dict[str, tuple[int, int]] = {}
    for klass in RESOURCE_CLASSES:
        have = int(used.get(klass) or 0)
        ceiling = int(budget.get(klass) or 0)
        if have > ceiling:
            overflow[klass] = (have, ceiling)
    return overflow


def fit_kernel_to_device(kernel: QGemvKernel, device: DeviceProfile) -> dict[str, Any]:
    """Refuse a kernel whose STATIC resource ESTIMATE exceeds the declared budget.

    CALL SITE of estimate_qgemv_resources. This is the estimator's refusal path;
    validate() RESOURCE_OVER_BUDGET remains the IR-level guard for filled graphs.
    """
    estimate = estimate_qgemv_resources(kernel)
    used = dict(estimate["used"])
    budget = device.resource_map()
    overflow = resource_overflow(used, budget)
    report = emit_evidence(
        "STATIC",
        {
            "budget": budget,
            "device_id": device.device_id,
            "kind": "RESOURCE_FIT",
            "ok": not overflow,
            "overflow": {k: {"used": a, "budget": b} for k, (a, b) in overflow.items()},
            "used": used,
        },
    )
    report["estimate"] = estimate
    if overflow:
        raise ResourceOverBudget(used, budget, overflow, device.device_id)
    return report


def _node_resources_from_estimate(kernel: QGemvKernel, used: Mapping[str, int]) -> dict[str, dict[str, int]]:
    """Split the engine estimate across IR nodes so validate() sees the same sum."""
    dsp = int(used["DSP"])
    lut_mac = dsp * LUT_PER_MAC_LANE + int(kernel.mac_lanes) * 8
    zero = _zero_resources()

    def pack(**kwargs: int) -> dict[str, int]:
        out = dict(zero)
        for k, v in kwargs.items():
            out[k] = int(v)
        return out

    return {
        "compute": pack(DSP=dsp, LUT=lut_mac),
        "decoder": pack(LUT=LUT_DECODE_BASE),
        "dma_in": pack(LUT=LUT_DMA_BASE),
        "dma_out": pack(LUT=LUT_DMA_BASE),
        "reduction": pack(LUT=LUT_REDUCE_BASE),
        "memory": pack(BRAM=int(used["BRAM"]), URAM=int(used["URAM"])),
    }


def from_qgemv(
    kernel: QGemvKernel | None = None,
    device: DeviceProfile | None = None,
    *,
    bind_budget: bool = True,
) -> HwirGraph:
    """Lower a qGEMV-class kernel into HWIR with evidence-backed primitives.

    CALL SITES:
      physical_primitives.instantiate  (via _back_primitive)
      estimate_qgemv_resources
    """
    kernel = kernel or canonical_qgemv_kernel()
    device = device or synthetic_u50_class()
    estimate = estimate_qgemv_resources(kernel)
    used = dict(estimate["used"])
    split = _node_resources_from_estimate(kernel, used)

    def backed(
        name: str,
        *,
        memory_tier: str,
        organ_class: str,
    ) -> tuple[str, str | None]:
        inst = _back_primitive(name, memory_tier=memory_tier, organ_class=organ_class)
        return inst.identity, inst.memory_tier

    id_mem, tier_mem = backed("StationaryRepresentation", memory_tier="HBM", organ_class="mlp")
    id_dec, tier_dec = backed("FusedDecodeCompute", memory_tier="ACCEL_SRAM", organ_class="mlp")
    id_cmp, tier_cmp = backed("TiledProjection", memory_tier="ACCEL_SRAM", organ_class="mlp")
    id_red, tier_red = backed("CollectiveRegion", memory_tier="ACCEL_SRAM", organ_class="multi_device")
    id_dma, tier_dma = backed("SemanticTransportEdge", memory_tier="REMOTE", organ_class="fpga_partition")

    dma_in = "dma.qgemv.in"
    dma_out = "dma.qgemv.out"
    mem_id = "mem.qgemv.packed"
    dec_id = "dec.qgemv.native"
    cmp_id = "cmp.qgemv.body"
    red_id = "red.qgemv.partial"

    def phys(rc: Mapping[str, int], *, hbm_channel: int | None = 0) -> PhysicalAttr:
        return PhysicalAttr(
            arithmetic_width="packed_low_bit",
            tile_shape=[int(kernel.tile_m), int(kernel.K)],
            banking=int(kernel.mac_lanes),
            hbm_channel=hbm_channel,
            resource_class=dict(rc),
            dfx_module_boundary="qgemv",
        )

    nodes = [
        HwirNode(
            id=dma_in,
            kind="dma-transport",
            primitive="SemanticTransportEdge",
            organ=kernel.organ,
            mapping="token activation ingress; no weight body",
            outputs={"out": "activation"},
            physical=phys(split["dma_in"], hbm_channel=None),
            lifetime="token",
            per_token_transfer=True,
            transport_policy="activations_and_partial_reductions_only",
            evidence_tier="STATIC",
            memory_tier=tier_dma,
            backed_identity=id_dma,
        ),
        HwirNode(
            id=dma_out,
            kind="dma-transport",
            primitive="SemanticTransportEdge",
            organ=kernel.organ,
            mapping="partial-reduction egress; no weight body",
            inputs={"in": "partial_reduction"},
            physical=phys(split["dma_out"], hbm_channel=None),
            lifetime="token",
            per_token_transfer=True,
            transport_policy="activations_and_partial_reductions_only",
            evidence_tier="STATIC",
            memory_tier=tier_dma,
            backed_identity=id_dma,
        ),
        HwirNode(
            id=mem_id,
            kind="memory",
            primitive="StationaryRepresentation",
            organ=kernel.organ,
            mapping="resident packed-low-bit shards; not source-dense weights",
            outputs={"out": "compact_representation_fragment"},
            physical=phys(split["memory"], hbm_channel=0),
            lifetime="persistent",
            per_token_transfer=False,
            resident_weight_policy="resident_shards_no_weight_body_per_token_transfer",
            evidence_tier="STATIC",
            memory_tier=tier_mem,
            backed_identity=id_mem,
        ),
        HwirNode(
            id=dec_id,
            kind="representation-decoder",
            primitive="FusedDecodeCompute",
            organ=kernel.organ,
            mapping="native decode of packed codes at the consumer; no_dense_rematerialization",
            inputs={"in": "compact_representation_fragment"},
            outputs={"out": "activation"},
            physical=phys(split["decoder"]),
            lifetime="token",
            evidence_tier="STATIC",
            memory_tier=tier_dec,
            backed_identity=id_dec,
        ),
        HwirNode(
            id=cmp_id,
            kind="compute",
            primitive="TiledProjection",
            organ=kernel.organ,
            mapping="low-bit qGEMV tiled projection; DSP MAC; no source-dense GEMM",
            inputs={"in_act": "activation", "in_rep": "activation"},
            outputs={"out": "partial_reduction"},
            physical=phys(split["compute"]),
            lifetime="token",
            resident_weight_policy="resident_shards_no_weight_body_per_token_transfer",
            evidence_tier="STATIC",
            memory_tier=tier_cmp,
            backed_identity=id_cmp,
        ),
        HwirNode(
            id=red_id,
            kind="reduction",
            primitive="CollectiveRegion",
            organ=kernel.organ,
            mapping="partial reduction of native qGEMV fragments",
            inputs={"in": "partial_reduction"},
            outputs={"out": "partial_reduction"},
            physical=phys(split["reduction"]),
            lifetime="token",
            evidence_tier="STATIC",
            memory_tier=tier_red,
            backed_identity=id_red,
        ),
    ]
    edges = [
        HwirEdge(
            id="e.qgemv.act",
            src=dma_in,
            src_port="out",
            dst=cmp_id,
            dst_port="in_act",
            frame_kind="activation",
        ),
        HwirEdge(
            id="e.qgemv.compact",
            src=mem_id,
            src_port="out",
            dst=dec_id,
            dst_port="in",
            frame_kind="compact_representation_fragment",
        ),
        HwirEdge(
            id="e.qgemv.decoded",
            src=dec_id,
            src_port="out",
            dst=cmp_id,
            dst_port="in_rep",
            frame_kind="activation",
        ),
        HwirEdge(
            id="e.qgemv.partial",
            src=cmp_id,
            src_port="out",
            dst=red_id,
            dst_port="in",
            frame_kind="partial_reduction",
        ),
        HwirEdge(
            id="e.qgemv.egress",
            src=red_id,
            src_port="out",
            dst=dma_out,
            dst_port="in",
            frame_kind="partial_reduction",
        ),
    ]
    notes = [
        "Lowered qGEMV-class kernel. PREHARDWARE. Not a bitstream and not a hardware timing claim.",
        "Resident packed shards stay put; per-token transport is activation / partial reduction only.",
        "Representation-decoder is FusedDecodeCompute: native decode, no dense rematerialization.",
        "Primitive identities come from physical_primitives.instantiate (FPGA backend).",
        "resource_class values are STATIC estimates from an assumed coefficient table, not synthesis.",
        "Device profile is synthetic/declared; there is no U50 board on this host.",
    ]
    return HwirGraph(
        model=kernel.model,
        organ=kernel.organ,
        source_receipt="tools.future.hwir.from_qgemv",
        source_hwir_schema=SCHEMA,
        qualification="STATIC_ONLY",
        semantics_consumed="physical_graph_noetic_native",
        nodes=nodes,
        edges=edges,
        device_budget=device.budget() if bind_budget else None,
        notes=notes,
        kernel=kernel.to_dict(),
    )


def simulate_qgemv_functional(
    kernel: QGemvKernel | None = None,
    operands: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """FUNCTIONAL_SIM of a qGEMV kernel.

    CALL SITE: tools.future.fpga_engines.qgemv — the bit-exact golden. This
    function does not reimplement the MAC loop.
    """
    from tools.future.fpga_engines import qgemv as qgemv_engine

    kernel = kernel or canonical_qgemv_kernel()
    ops = dict(operands or canonical_qgemv_operands())
    y = qgemv_engine(
        ops["codes"],
        ops["scales"],
        ops["x"],
        weight_bits=kernel.weight_bits,
        group_size=kernel.group_size,
    )
    y_list = [float(v) for v in list(y)]
    expected = ops.get("expected")
    match = None
    if expected is not None:
        exp = [float(v) for v in list(expected)]
        match = len(exp) == len(y_list) and all(a == b for a, b in zip(y_list, exp))
    return emit_evidence(
        "FUNCTIONAL_SIM",
        {
            "engine": "tools.future.fpga_engines.qgemv",
            "engine_symbol": "qgemv",
            "expected": None if expected is None else [float(v) for v in list(expected)],
            "kernel": kernel.to_dict(),
            "matches_expected": match,
            "note": (
                "Host-CPU functional interpreter of the qGEMV numerical contract. "
                "Not a cycle count, not a board run, not HARDWARE_MEASURED."
            ),
            "ok": True,
            "y": y_list,
        },
    )


def simulate_functional(
    graph: HwirGraph,
    operands: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """FUNCTIONAL_SIM dispatch. qGEMV-class graphs call fpga_engines.qgemv."""
    report = validate(graph)
    if not report.ok:
        return emit_evidence(
            "FUNCTIONAL_SIM",
            {
                "engine": None,
                "engine_symbol": None,
                "ok": False,
                "validate": report.to_dict(),
                "y": None,
            },
        )
    kernel_doc = graph.kernel or {}
    if graph.organ == "qgemv" or str(kernel_doc.get("organ") or "") == "qgemv":
        kernel = QGemvKernel(
            M=int(kernel_doc.get("M") or canonical_qgemv_kernel().M),
            K=int(kernel_doc.get("K") or canonical_qgemv_kernel().K),
            weight_bits=int(kernel_doc.get("weight_bits") or 4),
            group_size=int(kernel_doc.get("group_size") or 4),
            mac_lanes=int(kernel_doc.get("mac_lanes") or 2),
            tile_m=int(kernel_doc.get("tile_m") or 2),
            organ=str(kernel_doc.get("organ") or graph.organ or "qgemv"),
            model=str(kernel_doc.get("model") or graph.model or "qgemv-class"),
        )
        sim = simulate_qgemv_functional(kernel, operands)
        sim["graph_fingerprint"] = graph.fingerprint()
        sim["validate"] = report.to_dict()
        return sim
    return emit_evidence(
        "FUNCTIONAL_SIM",
        {
            "engine": None,
            "engine_symbol": None,
            "graph_fingerprint": graph.fingerprint(),
            "note": "no functional engine wired for this organ; qGEMV is the canonical first",
            "ok": False,
            "organ": graph.organ,
            "validate": report.to_dict(),
            "y": None,
        },
    )


def model_hbm_traffic(
    kernel: QGemvKernel,
    device: DeviceProfile | None = None,
    *,
    weights_resident: bool = True,
    fpga_rows: int | None = None,
) -> dict[str, Any]:
    """COST_MODEL of HBM bytes moved. Not a measured HBM2 rate."""
    device = device or synthetic_u50_class()
    rows = int(kernel.M if fpga_rows is None else fpga_rows)
    shard = replace(kernel, M=max(0, rows))
    weight_bytes = shard.weight_bytes()
    scale_bytes = shard.scale_bytes()
    x_bytes = shard.activation_in_bytes()
    y_bytes = shard.activation_out_bytes()
    # Doctrine §15.4: resident weights do not move per token.
    per_token = x_bytes + y_bytes
    if not weights_resident:
        per_token += weight_bytes + scale_bytes
    channels = min(int(device.hbm_channels), max(1, int(kernel.mac_lanes) // 8 or 1))
    modelled_cycles = _ceil_div(per_token, int(device.hbm_bytes_per_modelled_cycle))
    return emit_evidence(
        "COST_MODEL",
        {
            "assumed_bytes_per_modelled_cycle": int(device.hbm_bytes_per_modelled_cycle),
            "channels_touched": int(channels),
            "kind": "HBM_TRAFFIC_MODEL",
            "note": (
                "COST_MODEL. Bytes and modelled cycles from a declared HBM beat. "
                "Not a measured HBM2 bandwidth, not HARDWARE_MEASURED."
            ),
            "per_token_bytes": int(per_token),
            "resident_weight_bytes": int(weight_bytes + scale_bytes) if weights_resident else 0,
            "scale_bytes": int(scale_bytes),
            "weight_bytes": int(weight_bytes),
            "weights_resident": bool(weights_resident),
            "x_bytes": int(x_bytes),
            "y_bytes": int(y_bytes),
            "modelled_cycles": int(modelled_cycles),
        },
    )


def model_host_device_transfer(
    kernel: QGemvKernel,
    device: DeviceProfile | None = None,
    *,
    fpga_rows: int | None = None,
    weights_resident: bool = True,
) -> dict[str, Any]:
    """COST_MODEL of host<->device bytes. USB4-class prior, not a cable measurement.

    Deliberately does not emit bandwidth_gbps: that field is a hardware-claim
    key in tools.future._common and a number here would be a fabricated rate.
    """
    device = device or synthetic_u50_class()
    rows = int(kernel.M if fpga_rows is None else fpga_rows)
    rows = max(0, min(rows, kernel.M))
    h2c = kernel.activation_in_bytes() if rows else 0
    c2h = rows * 4  # partial reduction, float32
    if not weights_resident and rows:
        shard = replace(kernel, M=rows)
        h2c += shard.weight_bytes() + shard.scale_bytes()
    beat = int(device.host_device_bytes_per_modelled_cycle)
    cycles_h2c = 0 if h2c == 0 else HOST_DEVICE_QUEUE_CYCLES + _ceil_div(h2c, beat)
    cycles_c2h = 0 if c2h == 0 else HOST_DEVICE_QUEUE_CYCLES + _ceil_div(c2h, beat)
    pcie_gen = device.pcie_generation
    pcie_lanes = device.pcie_lanes
    if pcie_gen is not None and pcie_lanes is not None:
        transport_class = f"PCIE_GEN{int(pcie_gen)}_X{int(pcie_lanes)}_CLASS"
        transport_source = (
            "COST_MODEL PCIe payload-class beat (Gen3 lane unit=1, Gen4=2) "
            "scaled so Gen3 x16 equals HOST_DEVICE_BYTES_PER_MODELLED_CYCLE. "
            "Not a trained link, not a slot measurement."
        )
        note = (
            "COST_MODEL. Host<->device modelled cycles from a declared PCIe "
            "payload-class beat on the (possibly carrier-downgraded) device "
            "profile. Not a measurement of any cable or slot, not "
            "HARDWARE_MEASURED. bandwidth_gbps is intentionally absent."
        )
    else:
        transport_class = "USB4_40G_CLASS"
        transport_source = "H-ROADMAP.md §15.1 initial ~40 Gb/s class; COST_MODEL prior"
        note = (
            "COST_MODEL. USB4/Thunderbolt ~40 Gb/s CLASS prior from "
            "H-ROADMAP.md §15.1, expressed as a declared bytes/cycle beat. "
            "Not a measurement of any cable, not HARDWARE_MEASURED. "
            "bandwidth_gbps is intentionally absent."
        )
    return emit_evidence(
        "COST_MODEL",
        {
            "assumed_bytes_per_modelled_cycle": beat,
            "bytes_c2h": int(c2h),
            "bytes_h2c": int(h2c),
            "fpga_rows": int(rows),
            "kind": "HOST_DEVICE_TRANSFER_MODEL",
            "modelled_cycles_c2h": int(cycles_c2h),
            "modelled_cycles_h2c": int(cycles_h2c),
            "modelled_cycles_total": int(cycles_h2c + cycles_c2h),
            "note": note,
            "pcie_generation": UNPINNED if pcie_gen is None else int(pcie_gen),
            "pcie_lanes": UNPINNED if pcie_lanes is None else int(pcie_lanes),
            "queue_cycles_assumed": HOST_DEVICE_QUEUE_CYCLES,
            "transport_class": transport_class,
            "transport_class_source": transport_source,
            "weights_resident": bool(weights_resident),
        },
    )


def approximate_cycles(
    kernel: QGemvKernel,
    device: DeviceProfile | None = None,
    hbm: Mapping[str, Any] | None = None,
    xfer: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """CYCLE_APPROX critical path. Modelled cycles, not a clock, not seconds."""
    device = device or synthetic_u50_class()
    hbm = dict(hbm or model_hbm_traffic(kernel, device))
    xfer = dict(xfer or model_host_device_transfer(kernel, device))
    issue_width = max(1, int(kernel.mac_lanes) * int(kernel.tile_m))
    macs = int(kernel.M) * int(kernel.K)
    issue_beats = _ceil_div(macs, issue_width)
    compute = int(device.pipeline_depth) + int(device.initiation_interval) * max(0, issue_beats - 1)
    hbm_cycles = int(hbm.get("modelled_cycles") or 0)
    h2c = int(xfer.get("modelled_cycles_h2c") or 0)
    c2h = int(xfer.get("modelled_cycles_c2h") or 0)
    # Conservative overlap: host H2C, then max(compute, HBM), then C2H.
    modelled = h2c + max(compute, hbm_cycles) + c2h
    return emit_evidence(
        "CYCLE_APPROX",
        {
            "clock_hz": "UNKNOWN",
            "compute_modelled_cycles": int(compute),
            "conversion_reason": (
                "a cycle approximation is not a duration without a real clock; "
                "this host has no FPGA and no emulation seat"
            ),
            "conversion_to_seconds": "REFUSED",
            "hbm_modelled_cycles": int(hbm_cycles),
            "host_c2h_modelled_cycles": int(c2h),
            "host_h2c_modelled_cycles": int(h2c),
            "initiation_interval": int(device.initiation_interval),
            "issue_beats": int(issue_beats),
            "issue_width": int(issue_width),
            "kind": "CYCLE_APPROXIMATION",
            "modelled_cycles": int(modelled),
            "note": (
                "CYCLE_APPROX. Modelled critical-path cycles from declared II, "
                "depth, and COST_MODEL beats. Not a measurement. Not seconds."
            ),
            "pipeline_depth": int(device.pipeline_depth),
            "seconds": None,
        },
    )


def cycles_to_seconds(cycles: int, clock_hz: Any = None) -> None:
    """Refused. A cycle approximation is not a duration."""
    raise UnmeasuredConversionError(
        "CYCLE_APPROX cycles cannot be converted to seconds "
        f"(cycles={cycles!r}, clock_hz={clock_hz!r}); no real clock, no U50 board"
    )


def floorplan_hints(kernel: QGemvKernel, device: DeviceProfile | None = None) -> dict[str, Any]:
    """STATIC floorplan hints (APPENDIX K.4). Not place-and-route."""
    device = device or synthetic_u50_class()
    return emit_evidence(
        "STATIC",
        {
            "bram_uram_locality": "tile_buffer_beside_mac_array",
            "device_id": device.device_id,
            "dfx_region": "qgemv_module",
            "dsp_locality": "column_aligned_mac_array",
            "engine_placement": "near_hbm_controller",
            "hbm_controller_proximity": True,
            "kind": "FLOORPLAN_HINTS",
            "module_boundary": "qgemv",
            "note": (
                "STATIC hints from APPENDIX K.4. Not a floorplan, not P&R, "
                "not a congestion map, not HARDWARE_MEASURED."
            ),
            "stream_topology": "dma_in -> decode+mac -> reduce -> dma_out",
        },
    )


def partition_qgemv(
    kernel: QGemvKernel,
    device: DeviceProfile | None = None,
    *,
    weights_resident: bool = True,
) -> dict[str, Any]:
    """Row-split partitioner. FPGA shard must fit declared HBM capacity.

    CALL SITES: estimate_qgemv_resources / fit_kernel_to_device (engine must
    fit LUT/DSP/BRAM/URAM), model_host_device_transfer, model_hbm_traffic.

    LUT/DSP are properties of the engine, not of M. The split is on M so the
    resident packed shard fits hbm_capacity_bytes. §15.5 65/35 is recorded as
    a COST_MODEL prior and is not applied as physics.
    """
    device = device or synthetic_u50_class()
    engine = fit_kernel_to_device(kernel, device)
    bytes_per_row = 0
    if kernel.M > 0:
        bytes_per_row = _ceil_div(kernel.weight_bytes() + kernel.scale_bytes(), kernel.M)
    cap = int(device.hbm_capacity_bytes)
    max_rows = kernel.M
    if weights_resident and bytes_per_row > 0:
        max_rows = min(kernel.M, cap // bytes_per_row)
    fpga_rows = int(max_rows)
    host_rows = int(kernel.M - fpga_rows)
    xfer = model_host_device_transfer(
        kernel, device, fpga_rows=fpga_rows, weights_resident=weights_resident
    )
    hbm = model_hbm_traffic(
        kernel, device, weights_resident=weights_resident, fpga_rows=fpga_rows
    )
    return emit_evidence(
        "COST_MODEL",
        {
            "apple_fpga_prior": {
                "apple": APPLE_FPGA_PRIOR_NUM,
                "den": APPLE_FPGA_PRIOR_DEN,
                "fpga": APPLE_FPGA_PRIOR_DEN - APPLE_FPGA_PRIOR_NUM,
                "note": (
                    "H-ROADMAP.md §15.5 initial 65/35 Apple/FPGA is a COST_MODEL "
                    "prior, not a measurement, and is not applied as physics."
                ),
            },
            "axis": "within_organ_tensor_parallel_rows",
            "bytes_per_resident_row": int(bytes_per_row),
            "device_assignment": "APPLE_UMA_PLUS_FPGA_HBM_HYPOTHESIS",
            "device_id": device.device_id,
            "engine_fit": {k: engine[k] for k in ("ok", "used", "budget", "device_id", "evidence_tier")},
            "fpga_rows": fpga_rows,
            "hbm": hbm,
            "hbm_capacity_bytes": cap,
            "host_rows": host_rows,
            "kind": "PARTITION",
            "note": (
                "COST_MODEL partition. Resident packed shards stay on the FPGA "
                "shard; per-token transport is activation / partial reduction. "
                "Not a board placement."
            ),
            "resident_weight_policy": "resident_shards_no_weight_body_per_token_transfer",
            "transfer": xfer,
            "transport_policy": "activations_and_partial_reductions_only",
            "weights_resident": bool(weights_resident),
        },
    )


def run_qgemv_preboard(
    kernel: QGemvKernel | None = None,
    device: DeviceProfile | None = None,
    operands: Mapping[str, Any] | None = None,
    carrier: CarrierEnvelope | None = None,
) -> dict[str, Any]:
    """Lower, estimate, simulate, cost, approximate, partition. All PREHARDWARE.

    If a CarrierEnvelope is supplied, the device is DOWNGRADED first. The
    planner sees the reduced envelope, not the brochure one.
    """
    kernel = kernel or canonical_qgemv_kernel()
    device = device or synthetic_u50_class()
    if carrier is not None:
        device = constrain_device_profile(device, carrier)
    operands = dict(operands or canonical_qgemv_operands())
    graph = from_qgemv(kernel, device)
    report = validate(graph)
    resources = estimate_qgemv_resources(kernel)
    overflow = resource_overflow(resources["used"], device.resource_map())
    resource_fit: dict[str, Any]
    if overflow:
        resource_fit = emit_evidence(
            "STATIC",
            {
                "budget": device.resource_map(),
                "device_id": device.device_id,
                "kind": "RESOURCE_FIT",
                "ok": False,
                "overflow": {k: {"used": a, "budget": b} for k, (a, b) in overflow.items()},
                "used": resources["used"],
            },
        )
        partition = emit_evidence(
            "COST_MODEL",
            {
                "kind": "PARTITION",
                "ok": False,
                "reason": "engine resource ESTIMATE exceeds declared device budget",
                "refused": True,
            },
        )
        cycles = emit_evidence(
            "CYCLE_APPROX",
            {
                "kind": "CYCLE_APPROXIMATION",
                "modelled_cycles": None,
                "note": "not produced; engine does not fit the declared budget",
                "ok": False,
                "seconds": None,
            },
        )
        hbm = emit_evidence("COST_MODEL", {"kind": "HBM_TRAFFIC_MODEL", "ok": False, "refused": True})
        xfer = emit_evidence(
            "COST_MODEL", {"kind": "HOST_DEVICE_TRANSFER_MODEL", "ok": False, "refused": True}
        )
        functional = emit_evidence(
            "FUNCTIONAL_SIM",
            {
                "engine": None,
                "ok": False,
                "reason": "engine resource ESTIMATE exceeds declared device budget",
            },
        )
    else:
        resource_fit = fit_kernel_to_device(kernel, device)
        functional = simulate_functional(graph, operands)
        hbm = model_hbm_traffic(kernel, device)
        xfer = model_host_device_transfer(kernel, device)
        cycles = approximate_cycles(kernel, device, hbm, xfer)
        partition = partition_qgemv(kernel, device)
    hints = floorplan_hints(kernel, device)
    doc = emit_evidence(
        "STATIC",
        {
            "carrier_envelope": None if carrier is None else carrier.to_dict(),
            "claim_boundary": (
                "PREHARDWARE qGEMV pre-board stack. No FPGA board, no bitstream, "
                "no synthesis, no U50, no HARDWARE_MEASURED number. The resource "
                "figure is an ESTIMATE. The cycle figure is an APPROXIMATION, "
                "not a measurement. The real comma-device carrier is UNPINNED."
            ),
            "cycle_approx": cycles,
            "device_profile": device.to_dict(),
            "floorplan_hints": hints,
            "functional_sim": functional,
            "graph_fingerprint": graph.fingerprint(),
            "hbm_traffic": hbm,
            "host_device_transfer": xfer,
            "kernel": kernel.to_dict(),
            "kind": "QGEMV_PREBOARD",
            "organ_map_lowering": "from_qgemv",
            "partition": partition,
            "real_carrier": UNPINNED,
            "real_carrier_note": REAL_CARRIER_NOTE,
            "resource_estimate": resources,
            "resource_fit": resource_fit,
            "validate": report.to_dict(),
        },
    )
    assert_no_hardware_measured(doc)
    illegal = collect_evidence_tiers(doc) - set(EVIDENCE_TIERS)
    if illegal:
        raise IllegalEvidenceTier(f"illegal evidence tiers in preboard report: {sorted(illegal)}")
    return doc


# === GENERIC LOWERING LAYER BEGIN ===
# Vendor-neutral HWIR lowering-target interface. Style families live behind
# this boundary. Toolchain choice is not encoded. Rust owns Hawking semantics;
# host/runtime APIs stay behind a thin boundary and must not leak through
# the compiler. Device-specific names belong in backends, not here.

# Closed catalog the interface can name. Atlas primitives are the 17 IR
# contracts; platform primitives are the pieces that typically still need
# hand-written HDL. An emitter that cannot cover a name must say so.
PLATFORM_PRIMITIVES: tuple[str, ...] = (
    "clock_generator",
    "dfx_region_wrapper",
    "hbm_memory_controller",
    "host_link_phy",
    "interrupt_doorbell",
    "io_pinout_constraints",
    "vendor_dsp_primitive",
)

HARDWARE_PRIMITIVE_CATALOG: tuple[str, ...] = tuple(RECOVERED_PRIMITIVES) + PLATFORM_PRIMITIVES

# Explicitly unset. Ranking a target would encode a toolchain decision.
PREFERRED_LOWERING_TARGET: str | None = None

_HWIR_EMITTED_RE = re.compile(r"HWIR_EMITTED:([A-Za-z0-9_]+)")
_HWIR_HOLE_RE = re.compile(r"HWIR_HOLE:([A-Za-z0-9_]+)")


class UnknownLoweringTarget(KeyError):
    """No registered lowering target with that id."""


def lowering_emitted_primitives(result: Mapping[str, Any] | str) -> set[str]:
    """Primitives for which a lowering result contains an implementation body."""
    found: set[str] = set()
    if isinstance(result, str):
        found.update(_HWIR_EMITTED_RE.findall(result))
        return found
    for art in result.get("artifacts") or []:
        found.update(_HWIR_EMITTED_RE.findall(str(art.get("body") or "")))
    return found


def lowering_hole_primitives(result: Mapping[str, Any] | str) -> set[str]:
    """Primitives a lowering result names as a hole (not expressed)."""
    found: set[str] = set()
    if isinstance(result, str):
        found.update(_HWIR_HOLE_RE.findall(result))
        return found
    for art in result.get("artifacts") or []:
        found.update(_HWIR_HOLE_RE.findall(str(art.get("body") or "")))
    for name in result.get("cannot_express") or []:
        found.add(str(name))
    return found


def _c_ident(node_id: str) -> str:
    chars = [(ch if ch.isalnum() else "_") for ch in str(node_id)]
    ident = "".join(chars).strip("_") or "node"
    if ident[0].isdigit():
        ident = "n_" + ident
    return ident


def _kernel_params(graph: HwirGraph) -> dict[str, Any]:
    raw = dict(graph.kernel or {})

    def _int(name: str) -> int:
        value = raw.get(name)
        return 0 if value is None else int(value)

    return {
        "K": _int("K"),
        "M": _int("M"),
        "arithmetic": str(raw.get("arithmetic") or ""),
        "group_size": _int("group_size"),
        "mac_lanes": _int("mac_lanes"),
        "organ": str(raw.get("organ") or graph.organ or "organ"),
        "tile_m": _int("tile_m"),
        "weight_bits": _int("weight_bits"),
    }


def _topo_ids(graph: HwirGraph) -> list[str]:
    ids = [n.id for n in sorted(graph.nodes, key=lambda n: n.id)]
    incoming = {i: 0 for i in ids}
    adj = {i: [] for i in ids}
    for edge in graph.edges:
        if edge.src in adj and edge.dst in incoming:
            adj[edge.src].append(edge.dst)
            incoming[edge.dst] += 1
    ready = sorted(i for i, count in incoming.items() if count == 0)
    ordered: list[str] = []
    while ready:
        node_id = ready.pop(0)
        ordered.append(node_id)
        for dest in sorted(adj[node_id]):
            incoming[dest] -= 1
            if incoming[dest] == 0:
                ready.append(dest)
                ready.sort()
    for node_id in ids:
        if node_id not in ordered:
            ordered.append(node_id)
    return ordered


def _source_artifact(
    *,
    filename: str,
    language: str,
    body: str,
    target_id: str,
) -> dict[str, Any]:
    return emit_evidence(
        "STATIC",
        {
            "body": body,
            "filename": filename,
            "kind": "SOURCE_ARTIFACT",
            "language": language,
            "note": (
                "PREHARDWARE source artifact. Compiler product only. "
                "Not synthesis, not a bitstream, not a correctness claim, "
                "not HARDWARE_MEASURED."
            ),
            "target_id": target_id,
        },
    )


def _artifact_filename(graph: HwirGraph, target_id: str, ext: str) -> str:
    organ = _c_ident(graph.organ or "graph")
    return f"{organ}_{target_id}.{ext}"


class LoweringTarget(abc.ABC):
    """Pluggable HWIR lowering target. Equal citizen. Not a ranked backend.

    Generic methods must not branch on a device vendor. A target declares
    what it emits, what it cannot express, and which primitives still need
    hand-written HDL. 'Minimal HDL' is only a real claim when those holes
    are named.
    """

    TARGET_ID: str = ""
    FAMILY: str = ""
    EMITS: tuple[str, ...] = ()

    def target_id(self) -> str:
        return str(self.TARGET_ID)

    def family(self) -> str:
        return str(self.FAMILY)

    def emits(self) -> tuple[str, ...]:
        return tuple(self.EMITS)

    @abc.abstractmethod
    def cannot_express(self) -> tuple[str, ...]:
        """Primitives this target cannot express. Must be non-empty and honest."""

    @abc.abstractmethod
    def handwritten_hdl(self) -> tuple[str, ...]:
        """Primitives that still require hand-written HDL under this target."""

    @abc.abstractmethod
    def supported_primitives(self) -> tuple[str, ...]:
        """Primitives this target claims to emit an implementation body for."""

    @abc.abstractmethod
    def emit_artifacts(self, graph: HwirGraph) -> list[dict[str, Any]]:
        """Return SOURCE_ARTIFACT dicts. Source text only; no synthesis."""

    def lower(self, graph: HwirGraph) -> dict[str, Any]:
        return _finalize_lowering(graph, self, self.emit_artifacts(graph))

    def manifest(self) -> dict[str, Any]:
        return emit_evidence(
            "STATIC",
            {
                "cannot_express": list(self.cannot_express()),
                "emits": list(self.emits()),
                "family": self.family(),
                "handwritten_hdl": list(self.handwritten_hdl()),
                "kind": "LOWERING_TARGET_MANIFEST",
                "note": (
                    "Equal-citizen lowering target. PREHARDWARE source artifacts "
                    "only. Toolchain choice is not encoded."
                ),
                "preferred": False,
                "supported_primitives": list(self.supported_primitives()),
                "target_id": self.target_id(),
                "toolchain_choice": None,
            },
        )


_LOWERING_REGISTRY: dict[str, LoweringTarget] = {}


def register_lowering_target(target: LoweringTarget) -> LoweringTarget:
    """Register a lowering target. Ids are equal citizens; none is preferred."""
    if not isinstance(target, LoweringTarget):
        raise TypeError("target must implement LoweringTarget")
    tid = str(target.target_id() or "").strip()
    if not tid:
        raise ValueError("lowering target id is empty")
    prev = _LOWERING_REGISTRY.get(tid)
    if prev is not None and type(prev) is not type(target):
        raise ValueError(f"duplicate lowering target id {tid!r}")
    _LOWERING_REGISTRY[tid] = target
    return target


def list_lowering_targets() -> tuple[str, ...]:
    """Registered target ids, lexicographic. Order is not preference."""
    return tuple(sorted(_LOWERING_REGISTRY))


def get_lowering_target(target_id: str) -> LoweringTarget:
    tid = str(target_id or "").strip()
    target = _LOWERING_REGISTRY.get(tid)
    if target is None:
        known = list(list_lowering_targets())
        raise UnknownLoweringTarget(
            f"unknown lowering target {tid!r}; registered: {known}"
        )
    return target


def lowering_target_manifests() -> dict[str, dict[str, Any]]:
    return {tid: get_lowering_target(tid).manifest() for tid in list_lowering_targets()}


def _finalize_lowering(
    graph: HwirGraph,
    target: LoweringTarget,
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    report = validate(graph)
    cannot = list(target.cannot_express())
    if not cannot:
        raise ValueError(
            f"{target.target_id()} cannot_express is empty; "
            "unsupported primitives must be named"
        )
    hdl = list(target.handwritten_hdl())
    if not hdl:
        raise ValueError(
            f"{target.target_id()} handwritten_hdl is empty; "
            "minimal-HDL is only a real claim when the holes are named"
        )
    stamped = [_source_artifact(
        filename=str(art.get("filename") or "unnamed"),
        language=str(art.get("language") or "text"),
        body=str(art.get("body") or ""),
        target_id=target.target_id(),
    ) if not (isinstance(art, Mapping) and art.get("kind") == "SOURCE_ARTIFACT")
        else dict(art) for art in artifacts]
    for art in stamped:
        if art.get("kind") != "SOURCE_ARTIFACT":
            raise ValueError(f"{target.target_id()} emitted a non-source artifact")
        if "PREHARDWARE" not in str(art.get("body") or ""):
            raise ValueError(
                f"{target.target_id()} artifact {art.get('filename')!r} "
                "is not labeled PREHARDWARE"
            )
    body = {
        "artifacts": stamped,
        "cannot_express": cannot,
        "emits": list(target.emits()),
        "family": target.family(),
        "graph_fingerprint": graph.fingerprint(),
        "handwritten_hdl": hdl,
        "kind": "HWIR_LOWERING",
        "note": (
            "PREHARDWARE source artifacts. Compiler product only. "
            "Not synthesis, not a bitstream, not a correctness claim, "
            "not HARDWARE_MEASURED."
        ),
        "organ": graph.organ,
        "preferred": False,
        "supported_primitives": list(target.supported_primitives()),
        "target_id": target.target_id(),
        "toolchain_choice": None,
        "validate": report.to_dict(),
    }
    doc = emit_evidence("STATIC", body)
    assert_no_hardware_measured(doc)
    illegal = collect_evidence_tiers(doc) - set(EVIDENCE_TIERS)
    if illegal:
        raise IllegalEvidenceTier(
            f"illegal evidence tiers in lowering result: {sorted(illegal)}"
        )
    if PREFERRED_LOWERING_TARGET is not None:
        raise RuntimeError(
            "PREFERRED_LOWERING_TARGET must stay unset; "
            "the compiler does not pick a toolchain"
        )
    return doc


def lower_hwir(graph: HwirGraph, target_id: str) -> dict[str, Any]:
    """Lower `graph` through a registered target. Dispatcher is target-id keyed,
    never vendor keyed.
    """
    target = get_lowering_target(target_id)
    return target.lower(graph)


def lower_hwir_all(graph: HwirGraph) -> dict[str, dict[str, Any]]:
    """Lower the same graph through every registered target. No ranking."""
    return {tid: lower_hwir(graph, tid) for tid in list_lowering_targets()}


def lower_qgemv_targets(
    kernel: QGemvKernel | None = None,
    device: DeviceProfile | None = None,
) -> dict[str, Any]:
    """Lower the canonical (or supplied) qGEMV HWIR graph through every target."""
    graph = from_qgemv(kernel, device)
    results = lower_hwir_all(graph)
    doc = emit_evidence(
        "STATIC",
        {
            "graph_fingerprint": graph.fingerprint(),
            "kind": "QGEMV_LOWERING_TARGETS",
            "preferred": None,
            "target_ids": list(results),
            "targets": results,
            "toolchain_choice": None,
        },
    )
    assert_no_hardware_measured(doc)
    return doc


# === GENERIC LOWERING LAYER END ===


def _hole_comment(primitive: str, node_id: str, dialect: str) -> str:
    return (
        f"/* HWIR_HOLE:{primitive} node={node_id} "
        f"not expressed by {dialect}; hand-written HDL required */"
    )


def _hls_body_tiled_projection(node: HwirNode, kp: Mapping[str, Any]) -> str:
    ident = _c_ident(node.id)
    tile = int(kp.get("tile_m") or 0)
    lanes = int(kp.get("mac_lanes") or 0)
    return (
        f"/* HWIR_EMITTED:{node.primitive} node={node.id} */\n"
        f"static void tiled_projection_{ident}("
        "const float in_act[], const float in_rep[], float out_partial[]) {\n"
        f"    /* pipeline ii=1; unroll mac_lanes={lanes}; SOURCE ARTIFACT ONLY */\n"
        "    for (int m0 = 0; m0 < M; m0 += (TILE_M > 0 ? TILE_M : 1)) {\n"
        "        for (int k = 0; k < K; ++k) {\n"
        f"            for (int tm = 0; tm < {max(tile, 1)} && (m0 + tm) < M; ++tm) {{\n"
        "                out_partial[m0 + tm] += in_act[k] * in_rep[(m0 + tm) * (K > 0 ? K : 1) + k];\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "}\n"
    )


def _hls_body_fused_decode(node: HwirNode, kp: Mapping[str, Any]) -> str:
    ident = _c_ident(node.id)
    bits = int(kp.get("weight_bits") or 0)
    return (
        f"/* HWIR_EMITTED:{node.primitive} node={node.id} */\n"
        f"static void fused_decode_{ident}(const uint8_t packed[], float decoded[]) {{\n"
        "    /* native decode at the consumer; no dense rematerialization */\n"
        f"    /* packed codes at {bits} bits; FUNCTIONAL_SIM is the qGEMV engine, not this text */\n"
        "    (void)packed; (void)decoded;\n"
        "}\n"
    )


def _hls_body_stationary(node: HwirNode, kp: Mapping[str, Any]) -> str:
    ident = _c_ident(node.id)
    return (
        f"/* HWIR_EMITTED:{node.primitive} node={node.id} */\n"
        f"/* resident packed shards; per_token_transfer is false; "
        "memory-controller PHY is not in this source */\n"
        f"static const uint8_t stationary_{ident}[] = {{0}};\n"
    )


def _hls_body_transport(node: HwirNode, kp: Mapping[str, Any]) -> str:
    ident = _c_ident(node.id)
    return (
        f"/* HWIR_EMITTED:{node.primitive} node={node.id} */\n"
        f"static void semantic_transport_{ident}(const float src[], float dst[], int n) {{\n"
        "    /* typed stream copy; host-link PHY is not in this source */\n"
        "    for (int i = 0; i < n; ++i) dst[i] = src[i];\n"
        "}\n"
    )


def _hls_body_collective(node: HwirNode, kp: Mapping[str, Any]) -> str:
    ident = _c_ident(node.id)
    return (
        f"/* HWIR_EMITTED:{node.primitive} node={node.id} */\n"
        f"static void collective_{ident}(const float in[], float out[], int n) {{\n"
        "    for (int i = 0; i < n; ++i) out[i] = in[i];\n"
        "}\n"
    )


# Implementation bodies. Independent of HLS_STYLE_SUPPORTED_PRIMITIVES so a
# claim that is not backed by a body is detectable.
_HLS_BODIES = {
    "CollectiveRegion": _hls_body_collective,
    "FusedDecodeCompute": _hls_body_fused_decode,
    "SemanticTransportEdge": _hls_body_transport,
    "StationaryRepresentation": _hls_body_stationary,
    "TiledProjection": _hls_body_tiled_projection,
}

# Claimed support. Mutation check: add a primitive with no body here; the
# unsupported-primitive honesty test must then FAIL. Never leave a lie.
HLS_STYLE_SUPPORTED_PRIMITIVES: tuple[str, ...] = (
    "CollectiveRegion",
    "FusedDecodeCompute",
    "SemanticTransportEdge",
    "StationaryRepresentation",
    "TiledProjection",
)

HLS_STYLE_CANNOT_EXPRESS: tuple[str, ...] = (
    "AsyncPrefetch",
    "ConditionalPhysicalProgram",
    "DirectRoutedAccumulate",
    "DoubleBufferedTile",
    "GraphReplay",
    "LayoutTransform",
    "LocalStateMachine",
    "MemoryTierIdentity",
    "MoveOrRecompute",
    "PersistentPhysicalRegion",
    "SparseSkip",
    "SpatialPipeline",
    "clock_generator",
    "dfx_region_wrapper",
    "explicit_pipeline_registers",
    "explicit_ready_valid_handshake",
    "hbm_memory_controller",
    "host_link_phy",
    "interrupt_doorbell",
    "io_pinout_constraints",
    "vendor_dsp_primitive",
)

HLS_STYLE_HANDWRITTEN_HDL: tuple[str, ...] = (
    "clock_generator",
    "dfx_region_wrapper",
    "hbm_memory_controller",
    "host_link_phy",
    "interrupt_doorbell",
    "io_pinout_constraints",
    "vendor_dsp_primitive",
)


class HlsStyleEmitter(LoweringTarget):
    """High-level-synthesis-style C/C++ kernel source. Not a vendor dialect."""

    TARGET_ID = "hls_style"
    FAMILY = "hls"
    EMITS = ("c_kernel_source",)

    def cannot_express(self) -> tuple[str, ...]:
        return HLS_STYLE_CANNOT_EXPRESS

    def handwritten_hdl(self) -> tuple[str, ...]:
        return HLS_STYLE_HANDWRITTEN_HDL

    def supported_primitives(self) -> tuple[str, ...]:
        return HLS_STYLE_SUPPORTED_PRIMITIVES

    def emit_artifacts(self, graph: HwirGraph) -> list[dict[str, Any]]:
        kp = _kernel_params(graph)
        chunks = [
            "/* PREHARDWARE SOURCE ARTIFACT */",
            "/* evidence_tier: STATIC */",
            "/* qualification: PREHARDWARE */",
            "/* hardware_measured: false */",
            "/* Not synthesis. Not a bitstream. Not a timing claim. */",
            f"/* HWIR graph fingerprint: {graph.fingerprint()} */",
            f"/* lowering_target: {self.target_id()} */",
            "#include <stdint.h>",
            f"#define M {int(kp['M'])}",
            f"#define K {int(kp['K'])}",
            f"#define TILE_M {int(kp['tile_m'])}",
            f"#define MAC_LANES {int(kp['mac_lanes'])}",
            f"#define WEIGHT_BITS {int(kp['weight_bits'])}",
            f"#define GROUP_SIZE {int(kp['group_size'])}",
            "",
        ]
        emitted_fns: list[str] = []
        for node in sorted(graph.nodes, key=lambda n: n.id):
            prim = node.primitive or node.kind
            body_fn = _HLS_BODIES.get(prim)
            if body_fn is None:
                chunks.append(_hole_comment(prim, node.id, self.target_id()))
            else:
                chunks.append(body_fn(node, kp))
                emitted_fns.append(f"{prim}:{_c_ident(node.id)}")
        chunks.append("/* edges */")
        for edge in sorted(graph.edges, key=lambda e: e.id):
            chunks.append(
                f"/* edge {edge.id}: {edge.src}.{edge.src_port} -> "
                f"{edge.dst}.{edge.dst_port} frame={edge.frame_kind} */"
            )
        chunks.append("/* topo kernel (shape-faithful skeleton, not a golden) */")
        chunks.append("void hwir_kernel(const float act_in[], float partial_out[]) {")
        chunks.append(f"    /* topo: {' -> '.join(_topo_ids(graph))} */")
        chunks.append("    (void)act_in; (void)partial_out;")
        chunks.append("}")
        chunks.append("/* cannot_express roster */")
        for name in self.cannot_express():
            chunks.append(_hole_comment(name, "-", self.target_id()))
        chunks.append("/* handwritten_hdl: " + ", ".join(self.handwritten_hdl()) + " */")
        body = "\n".join(chunks) + "\n"
        return [
            _source_artifact(
                filename=_artifact_filename(graph, self.target_id(), "cpp"),
                language="c++",
                body=body,
                target_id=self.target_id(),
            )
        ]


def _rust_body_tiled_projection(node: HwirNode, kp: Mapping[str, Any]) -> str:
    ident = _c_ident(node.id)
    return (
        f"/// HWIR_EMITTED:{node.primitive} node={node.id}\n"
        f"pub struct TiledProjection_{ident} {{\n"
        f"    pub tile_m: u32,\n"
        f"    pub k: u32,\n"
        f"    pub mac_lanes: u32,\n"
        f"    pub acc: i32, // explicit pipeline register\n"
        "}\n"
        f"impl TiledProjection_{ident} {{\n"
        "    pub fn tick(&mut self, act: i32, decoded: i32) -> i32 {\n"
        "        // structural MAC; SOURCE ARTIFACT ONLY\n"
        f"        let _lanes = {int(kp.get('mac_lanes') or 0)}u32;\n"
        "        self.acc = self.acc.wrapping_add(act.wrapping_mul(decoded));\n"
        "        self.acc\n"
        "    }\n"
        "}\n"
    )


def _rust_body_fused_decode(node: HwirNode, kp: Mapping[str, Any]) -> str:
    ident = _c_ident(node.id)
    bits = int(kp.get("weight_bits") or 0)
    return (
        f"/// HWIR_EMITTED:{node.primitive} node={node.id}\n"
        f"pub struct FusedDecode_{ident};\n"
        f"impl FusedDecode_{ident} {{\n"
        "    pub fn tick(&self, packed: u8) -> i32 {\n"
        f"        // native {bits}-bit decode; no dense rematerialization\n"
        "        let _ = packed;\n"
        "        0\n"
        "    }\n"
        "}\n"
    )


def _rust_body_stationary(node: HwirNode, kp: Mapping[str, Any]) -> str:
    ident = _c_ident(node.id)
    return (
        f"/// HWIR_EMITTED:{node.primitive} node={node.id}\n"
        f"pub struct Stationary_{ident} {{\n"
        "    pub resident: bool,\n"
        "}\n"
        f"impl Stationary_{ident} {{\n"
        "    pub const fn new() -> Self { Self { resident: true } }\n"
        "}\n"
    )


def _rust_body_transport(node: HwirNode, kp: Mapping[str, Any]) -> str:
    ident = _c_ident(node.id)
    return (
        f"/// HWIR_EMITTED:{node.primitive} node={node.id}\n"
        f"pub struct SemanticTransport_{ident};\n"
        f"impl SemanticTransport_{ident} {{\n"
        "    pub fn tick(&self, src: ReadyValid<i32>) -> ReadyValid<i32> {\n"
        "        // typed stream; host-link PHY is not in this source\n"
        "        src\n"
        "    }\n"
        "}\n"
    )


def _rust_body_collective(node: HwirNode, kp: Mapping[str, Any]) -> str:
    ident = _c_ident(node.id)
    return (
        f"/// HWIR_EMITTED:{node.primitive} node={node.id}\n"
        f"pub struct Collective_{ident} {{ pub acc: i32 }}\n"
        f"impl Collective_{ident} {{\n"
        "    pub fn tick(&mut self, lane: i32) -> i32 {\n"
        "        self.acc = self.acc.wrapping_add(lane);\n"
        "        self.acc\n"
        "    }\n"
        "}\n"
    )


_RUST_BODIES = {
    "CollectiveRegion": _rust_body_collective,
    "FusedDecodeCompute": _rust_body_fused_decode,
    "SemanticTransportEdge": _rust_body_transport,
    "StationaryRepresentation": _rust_body_stationary,
    "TiledProjection": _rust_body_tiled_projection,
}

RUST_HDL_SUPPORTED_PRIMITIVES: tuple[str, ...] = (
    "CollectiveRegion",
    "FusedDecodeCompute",
    "SemanticTransportEdge",
    "StationaryRepresentation",
    "TiledProjection",
)

RUST_HDL_CANNOT_EXPRESS: tuple[str, ...] = (
    "AsyncPrefetch",
    "ConditionalPhysicalProgram",
    "DirectRoutedAccumulate",
    "DoubleBufferedTile",
    "GraphReplay",
    "LayoutTransform",
    "LocalStateMachine",
    "MemoryTierIdentity",
    "MoveOrRecompute",
    "PersistentPhysicalRegion",
    "SparseSkip",
    "SpatialPipeline",
    "clock_generator",
    "dfx_region_wrapper",
    "hbm_memory_controller",
    "host_link_phy",
    "inferred_loop_pipelining",
    "interrupt_doorbell",
    "io_pinout_constraints",
    "vendor_dsp_primitive",
)

RUST_HDL_HANDWRITTEN_HDL: tuple[str, ...] = (
    "clock_generator",
    "dfx_region_wrapper",
    "hbm_memory_controller",
    "host_link_phy",
    "interrupt_doorbell",
    "io_pinout_constraints",
    "vendor_dsp_primitive",
)


class RustHdlEmitter(LoweringTarget):
    """Rust-hosted HDL/IR-style source. Structural modules, not a toolchain."""

    TARGET_ID = "rust_hdl_style"
    FAMILY = "rust_hdl"
    EMITS = ("rust_hdl_ir_source",)

    def cannot_express(self) -> tuple[str, ...]:
        return RUST_HDL_CANNOT_EXPRESS

    def handwritten_hdl(self) -> tuple[str, ...]:
        return RUST_HDL_HANDWRITTEN_HDL

    def supported_primitives(self) -> tuple[str, ...]:
        return RUST_HDL_SUPPORTED_PRIMITIVES

    def emit_artifacts(self, graph: HwirGraph) -> list[dict[str, Any]]:
        kp = _kernel_params(graph)
        chunks = [
            "//! PREHARDWARE SOURCE ARTIFACT",
            "//! evidence_tier: STATIC",
            "//! qualification: PREHARDWARE",
            "//! hardware_measured: false",
            "//! Not synthesis. Not a bitstream. Not a timing claim.",
            f"//! HWIR graph fingerprint: {graph.fingerprint()}",
            f"//! lowering_target: {self.target_id()}",
            "#![allow(non_camel_case_types, dead_code)]",
            "",
            f"pub const M: u32 = {int(kp['M'])};",
            f"pub const K: u32 = {int(kp['K'])};",
            f"pub const TILE_M: u32 = {int(kp['tile_m'])};",
            f"pub const MAC_LANES: u32 = {int(kp['mac_lanes'])};",
            f"pub const WEIGHT_BITS: u32 = {int(kp['weight_bits'])};",
            f"pub const GROUP_SIZE: u32 = {int(kp['group_size'])};",
            "",
            "/// explicit ready/valid handshake (expressed in this dialect)",
            "#[derive(Clone, Copy)]",
            "pub struct ReadyValid<T> { pub valid: bool, pub ready: bool, pub data: T }",
            "",
        ]
        for node in sorted(graph.nodes, key=lambda n: n.id):
            prim = node.primitive or node.kind
            body_fn = _RUST_BODIES.get(prim)
            if body_fn is None:
                chunks.append(
                    f"// {_hole_comment(prim, node.id, self.target_id())}"
                )
            else:
                chunks.append(body_fn(node, kp))
        chunks.append("pub struct Edge { pub id: &'static str, pub src: &'static str, pub dst: &'static str }")
        chunks.append("pub const EDGES: &[Edge] = &[")
        for edge in sorted(graph.edges, key=lambda e: e.id):
            chunks.append(
                f'    Edge {{ id: "{edge.id}", src: "{edge.src}", dst: "{edge.dst}" }},'
            )
        chunks.append("];")
        chunks.append(f"pub const TOPO: &[&str] = &[{', '.join(repr(i) for i in _topo_ids(graph))}];")
        chunks.append("// cannot_express roster")
        for name in self.cannot_express():
            chunks.append(f"// {_hole_comment(name, '-', self.target_id())}")
        chunks.append("// handwritten_hdl: " + ", ".join(self.handwritten_hdl()))
        body = "\n".join(chunks) + "\n"
        return [
            _source_artifact(
                filename=_artifact_filename(graph, self.target_id(), "rs"),
                language="rust",
                body=body,
                target_id=self.target_id(),
            )
        ]


# Backend registration. Lexicographic listing is not preference.
register_lowering_target(HlsStyleEmitter())
register_lowering_target(RustHdlEmitter())


# ---------------------------------------------------------------------------
# Sealed predictions. PREHARDWARE. Scored when U50_PRESENT; rehearsed without.
#
# An unsealed prediction can be rationalised after the board arrives. Sealing
# before arrival is the scientific act. There is no FPGA on this host; the
# rehearsal below is a test of the scoring machinery, never an arrival.
# ---------------------------------------------------------------------------


class PredictionRefused(ValueError):
    """A prediction was refused at creation. Not a hardware result."""


class TamperedPrediction(ValueError):
    """Sealed prediction content does not match its content hash."""


class ScoringRefused(ValueError):
    """Scoring cannot run: no U50, a mislabeled rehearsal, or a bad observation."""


_PREDICTION_SEAL_EXCLUDE = frozenset({"content_sha256"})

SYNTHETIC_REHEARSAL_LABEL = (
    "SYNTHETIC REHEARSAL — NOT AN ARRIVAL — NOT A BOARD MEASUREMENT — "
    "NOT HARDWARE_MEASURED. A test of the scoring machinery, not evidence "
    "about any FPGA."
)


def _require_falsifier(value: Any) -> str:
    if value is None:
        raise PredictionRefused(
            "prediction refused: falsification_condition is required"
        )
    if not isinstance(value, str):
        raise PredictionRefused(
            "prediction refused: falsification_condition must be a non-empty string"
        )
    text = value.strip()
    if not text:
        raise PredictionRefused(
            "prediction refused: falsification_condition is empty"
        )
    return text


def _require_number(value: Any, what: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionRefused(f"prediction refused: {what} must be a number")
    return value


def _prediction_material(pred: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in pred.items() if k not in _PREDICTION_SEAL_EXCLUDE}


def prediction_content_sha256(pred: Mapping[str, Any]) -> str:
    """Content hash of a prediction. Everything except content_sha256 itself."""
    material = json.loads(canon_dumps(_prediction_material(pred)))
    return hashlib.sha256(canon_dumps(material).encode("utf-8")).hexdigest()


def _seal_is_valid(expected: str, stored: Any) -> bool:
    """True iff the recomputed hash matches the stored seal.

    MUTATION_CHECK: replace the body with `return True` (treat any stored
    seal as valid). test_tampered_sealed_prediction_is_rejected must then
    FAIL. Restore after the check.
    """
    return stored == expected


def verify_prediction_seal(pred: Mapping[str, Any]) -> str:
    """Recompute the content hash. Raise TamperedPrediction on mismatch."""
    stored = pred.get("content_sha256")
    if not stored:
        raise TamperedPrediction("sealed prediction has no content_sha256")
    expected = prediction_content_sha256(pred)
    if not _seal_is_valid(expected, stored):
        raise TamperedPrediction(
            f"sealed prediction {pred.get('id')!r} content does not match "
            f"content_sha256 (stored={stored!r} recomputed={expected!r})"
        )
    return expected


def declared_planning_coefficients(
    device: DeviceProfile | None = None,
) -> dict[str, Any]:
    """Snapshot of DECLARED planning coefficients. Not measurements."""
    device = device or synthetic_u50_class()
    return {
        "hbm_bytes_per_modelled_cycle": {
            "document_class": (
                "COST_MODEL declared planning coefficient. Not vendor "
                "datasheet HBM2 GB/s, not a board census, not HARDWARE_MEASURED."
            ),
            "evidence_tier": "COST_MODEL",
            "note": "Declared HBM beat used by model_hbm_traffic. Not a measured HBM2 rate.",
            "origin": "DECLARED_PLANNING_COEFFICIENT",
            "units": "bytes/modelled_cycle",
            "value": int(device.hbm_bytes_per_modelled_cycle),
        },
        "fabric_bytes_per_modelled_cycle": {
            "document_class": (
                "COST_MODEL declared planning coefficient matching "
                "tools/future/fpga_fidelity.py MODEL_BYTES_PER_CYCLE. "
                "Not a measurement. Honest gap: CYCLE_APPROX does not yet "
                "fold this beat into critical-path modelled cycles."
            ),
            "evidence_tier": "COST_MODEL",
            "note": "Declared on-chip fabric beat. Not a measured interconnect rate.",
            "origin": "DECLARED_PLANNING_COEFFICIENT",
            "units": "bytes/modelled_cycle",
            "value": int(device.fabric_bytes_per_modelled_cycle),
        },
        "host_device_bytes_per_modelled_cycle": {
            "document_class": (
                "H-ROADMAP.md §15.1 USB4/Thunderbolt ~40 Gb/s CLASS prior, "
                "expressed as a declared bytes/cycle beat. COST_MODEL. "
                "Not a cable measurement."
            ),
            "evidence_tier": "COST_MODEL",
            "note": "Declared host<->device beat used by model_host_device_transfer.",
            "origin": "DECLARED_PLANNING_COEFFICIENT",
            "units": "bytes/modelled_cycle",
            "value": int(device.host_device_bytes_per_modelled_cycle),
        },
        "host_device_queue_cycles": {
            "document_class": (
                "COST_MODEL declared queue/setup overhead in modelled cycles. "
                "Not a measured doorbell or descriptor latency."
            ),
            "evidence_tier": "COST_MODEL",
            "note": "Declared HOST_DEVICE_QUEUE_CYCLES prior.",
            "origin": "DECLARED_PLANNING_COEFFICIENT",
            "units": "modelled_cycles",
            "value": int(HOST_DEVICE_QUEUE_CYCLES),
        },
        "lut_per_mac_lane": {
            "document_class": (
                "STATIC assumed coefficient table v1-assumed-not-synthesized. "
                "Not a synthesis report, not place-and-route."
            ),
            "evidence_tier": "STATIC",
            "note": "LUT_PER_MAC_LANE assumed coefficient.",
            "origin": "ASSUMED_COEFFICIENT_TABLE",
            "units": "LUT",
            "value": int(LUT_PER_MAC_LANE),
        },
        "pipeline_depth": {
            "document_class": "STATIC declared pipeline depth on the synthetic device profile.",
            "evidence_tier": "STATIC",
            "note": "Declared DeviceProfile.pipeline_depth. Not a measured II/depth.",
            "origin": "DECLARED_PLANNING_COEFFICIENT",
            "units": "modelled_cycles",
            "value": int(device.pipeline_depth),
        },
        "initiation_interval": {
            "document_class": "STATIC declared initiation interval on the synthetic device profile.",
            "evidence_tier": "STATIC",
            "note": "Declared DeviceProfile.initiation_interval. Not a measured II.",
            "origin": "DECLARED_PLANNING_COEFFICIENT",
            "units": "modelled_cycles",
            "value": int(device.initiation_interval),
        },
        "resident_per_token_byte_identity": {
            "document_class": (
                "COST_MODEL identity. weights_resident=True => per_token_bytes "
                "= activation_in_bytes + activation_out_bytes. Doctrine §15.4: "
                "resident weights do not move per token. Not a beat coefficient."
            ),
            "evidence_tier": "COST_MODEL",
            "note": "Per-token HBM traffic identity used by model_hbm_traffic.",
            "origin": "COST_MODEL_IDENTITY",
            "units": "bytes",
            "value": "activation_in_bytes + activation_out_bytes",
        },
        "dsp_equals_mac_lanes_times_tile": {
            "document_class": (
                "STATIC identity: DSP = mac_lanes * tile_m in estimate_qgemv_resources. "
                "Not a synthesis result."
            ),
            "evidence_tier": "STATIC",
            "note": "DSP MAC mapping used by the STATIC resource estimator.",
            "origin": "ASSUMED_COEFFICIENT_TABLE",
            "units": "DSP",
            "value": "mac_lanes * tile_m",
        },
    }


def _subset_coefficients(
    names: Sequence[str],
    table: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    table = dict(table or declared_planning_coefficients())
    missing = [n for n in names if n not in table]
    if missing:
        raise PredictionRefused(
            f"prediction refused: unknown model coefficient(s) {missing}"
        )
    return {n: json.loads(canon_dumps(table[n])) for n in names}


def _normalize_tolerance(tolerance: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(tolerance, Mapping):
        raise PredictionRefused("prediction refused: tolerance is required")
    kind = str(tolerance.get("kind") or "").strip()
    if kind not in TOLERANCE_KINDS:
        raise PredictionRefused(
            f"prediction refused: tolerance.kind must be one of {list(TOLERANCE_KINDS)}"
        )
    raw = tolerance.get("value")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise PredictionRefused("prediction refused: tolerance.value must be a number")
    if float(raw) < 0:
        raise PredictionRefused("prediction refused: tolerance.value must be >= 0")
    return {"kind": kind, "value": float(raw)}


def seal_prediction(
    *,
    id: str,
    plan: str,
    quantity: str,
    predicted_value: Any,
    units: str,
    model_coefficients: Mapping[str, Any],
    depends_on: Sequence[str],
    tolerance: Mapping[str, Any],
    falsification_condition: Any,
    implicated_coefficient: str,
    evidence_tier: str,
    kernel: Mapping[str, Any] | None = None,
    notes: Sequence[str] | None = None,
    wake_condition: str = WAKE_U50_PRESENT,
) -> dict[str, Any]:
    """Create a sealed prediction. Refused without a falsification condition.

    The content hash covers every scientific field. Re-sealing after an edit
    produces a different hash; that is a new prediction, not a quiet re-aim.
    """
    falsifier = _require_falsifier(falsification_condition)
    if not str(id).strip():
        raise PredictionRefused("prediction refused: id is required")
    if not str(plan).strip():
        raise PredictionRefused("prediction refused: plan is required")
    if not str(quantity).strip():
        raise PredictionRefused("prediction refused: quantity is required")
    if not str(units).strip():
        raise PredictionRefused("prediction refused: units are required")
    if not str(implicated_coefficient).strip():
        raise PredictionRefused("prediction refused: implicated_coefficient is required")
    if evidence_tier in ILLEGAL_EVIDENCE_TIERS or evidence_tier == "HARDWARE_MEASURED":
        raise IllegalEvidenceTier(
            f"prediction {id!r} cannot carry evidence_tier={evidence_tier!r}"
        )
    if evidence_tier not in EVIDENCE_TIERS:
        raise IllegalEvidenceTier(
            f"prediction {id!r} evidence_tier={evidence_tier!r} is not one of {list(EVIDENCE_TIERS)}"
        )
    if str(wake_condition) != WAKE_U50_PRESENT:
        raise PredictionRefused(
            f"prediction refused: wake_condition must be {WAKE_U50_PRESENT!r}"
        )
    predicted = _require_number(predicted_value, "predicted_value")
    tol = _normalize_tolerance(tolerance)
    deps = [str(x) for x in depends_on]
    if not deps:
        raise PredictionRefused("prediction refused: depends_on must name at least one coefficient")
    if implicated_coefficient not in deps:
        raise PredictionRefused(
            "prediction refused: implicated_coefficient must be listed in depends_on"
        )
    coeff_table = dict(model_coefficients)
    missing_deps = [n for n in deps if n not in coeff_table]
    if missing_deps:
        raise PredictionRefused(
            f"prediction refused: depends_on not present in model_coefficients: {missing_deps}"
        )
    body = {
        "depends_on": list(deps),
        "evidence_tier": str(evidence_tier),
        "falsification_condition": falsifier,
        "hardware_measured": False,
        "id": str(id),
        "implicated_coefficient": str(implicated_coefficient),
        "kernel": None if kernel is None else json.loads(canon_dumps(dict(kernel))),
        "model_coefficients": json.loads(canon_dumps(coeff_table)),
        "notes": [str(x) for x in (notes or ())],
        "plan": str(plan),
        "predicted_value": predicted,
        "prehardware": True,
        "qualification": PREHARDWARE,
        "quantity": str(quantity),
        "schema": PREDICTION_SCHEMA,
        "tolerance": tol,
        "units": str(units),
        "wake_condition": WAKE_U50_PRESENT,
    }
    body = json.loads(canon_dumps(body))
    body["content_sha256"] = prediction_content_sha256(body)
    assert_no_hardware_measured(body)
    return body


def prediction_set_digest(predictions: Sequence[Mapping[str, Any]]) -> str:
    pairs = sorted((str(p["id"]), str(p["content_sha256"])) for p in predictions)
    return hashlib.sha256(canon_dumps(pairs).encode("utf-8")).hexdigest()


def load_u50_wake_condition() -> dict[str, Any]:
    """U50_PRESENT from the capability graph. Absent graph => present=False.

    Never invents present=True. There is no FPGA on this host.
    """
    present = False
    source = "default_false_no_board"
    probe: dict[str, Any] | None = None
    gates_carrying: list[str] = []
    graph_sha: str | None = None
    disk = REPO / CAPABILITY_GRAPH_REL
    doc: dict[str, Any] | None = None
    if disk.is_file():
        try:
            doc = load_json(disk)
            graph_sha = sha256_file(disk)
            source = CAPABILITY_GRAPH_REL
        except (OSError, ValueError, TypeError):
            doc = None
    if doc is None:
        git_doc = _git_json(CAPABILITY_GRAPH_REL)
        if git_doc is not None:
            doc, graph_sha = git_doc
            source = f"git HEAD:{CAPABILITY_GRAPH_REL}"
    if isinstance(doc, dict):
        probes = doc.get("hardware_probes") if isinstance(doc.get("hardware_probes"), Mapping) else {}
        raw = probes.get(WAKE_U50_PRESENT) if isinstance(probes, Mapping) else None
        if isinstance(raw, Mapping):
            probe = {
                "description": raw.get("description"),
                "evidence": raw.get("evidence"),
                "evidence_tier": raw.get("evidence_tier"),
                "id": raw.get("id") or WAKE_U50_PRESENT,
                "present": bool(raw.get("present")),
            }
            present = bool(raw.get("present"))
        gates = doc.get("gates") if isinstance(doc.get("gates"), Mapping) else {}
        for gid in U50_WAKE_GATES:
            row = gates.get(gid) if isinstance(gates, Mapping) else None
            if not isinstance(row, Mapping):
                continue
            if (
                row.get("wake_condition") == WAKE_U50_PRESENT
                or row.get("hardware_blocker") == WAKE_U50_PRESENT
            ):
                gates_carrying.append(gid)
    if not gates_carrying:
        gates_carrying = list(U50_WAKE_GATES)
        if source == "default_false_no_board":
            source = "hardcoded_U50_WAKE_GATES_graph_absent"
    return emit_evidence(
        "STATIC",
        {
            "capability_graph": CAPABILITY_GRAPH_REL,
            "capability_graph_sha256": graph_sha,
            "gates_carrying_wake": list(gates_carrying),
            "kind": "WAKE_CONDITION",
            "note": (
                "U50_PRESENT is the wake condition for the inbound Alveo U50. "
                "present=false unless the capability graph says otherwise. "
                "This host has no FPGA; do not treat this block as a census."
            ),
            "present": bool(present),
            "probe": probe,
            "source": source,
            "wake_condition": WAKE_U50_PRESENT,
        },
    )


def inbound_board_predictions(
    device: DeviceProfile | None = None,
) -> list[dict[str, Any]]:
    """Starting sealed set about the inbound board. Cost-model numbers, not measurements."""
    device = device or synthetic_u50_class()
    canonical = canonical_qgemv_kernel()
    planning = planning_qgemv_kernel()
    coeffs = declared_planning_coefficients(device)
    hbm_c = model_hbm_traffic(canonical, device)
    xfer_c = model_host_device_transfer(canonical, device)
    cyc_c = approximate_cycles(canonical, device, hbm_c, xfer_c)
    est_c = estimate_qgemv_resources(canonical)
    hbm_p = model_hbm_traffic(planning, device)
    plan = "inbound-u50-qgemv"
    canon_kernel = canonical.to_dict()
    plan_kernel = planning.to_dict()

    def coeff_pred(
        *,
        pid: str,
        quantity: str,
        value: int | float,
        units: str,
        implicated: str,
        falsifier: str,
        evidence_tier: str,
        kernel: Mapping[str, Any] | None,
        notes: Sequence[str],
        rel: float = 0.50,
        abs_tol: float | None = None,
        extra_depends: Sequence[str] = (),
    ) -> dict[str, Any]:
        depends = (implicated, *extra_depends)
        tol = (
            {"kind": "absolute", "value": float(abs_tol)}
            if abs_tol is not None
            else {"kind": "relative", "value": float(rel)}
        )
        return seal_prediction(
            id=pid,
            plan=plan,
            quantity=quantity,
            predicted_value=value,
            units=units,
            model_coefficients=_subset_coefficients(depends, coeffs),
            depends_on=depends,
            tolerance=tol,
            falsification_condition=falsifier,
            implicated_coefficient=implicated,
            evidence_tier=evidence_tier,
            kernel=kernel,
            notes=notes,
        )

    rows = [
        coeff_pred(
            pid=PRED_HBM_BEAT,
            quantity="hbm_bytes_per_modelled_cycle",
            value=int(device.hbm_bytes_per_modelled_cycle),
            units="bytes/modelled_cycle",
            implicated="hbm_bytes_per_modelled_cycle",
            evidence_tier="COST_MODEL",
            kernel=canon_kernel,
            notes=[
                "Declared planning coefficient on the synthetic U50-class profile.",
                "Not a measured HBM2 rate and not a vendor-datasheet GB/s figure.",
            ],
            falsifier=(
                "An observation of HBM bytes per modelled cycle on the inbound "
                f"U50 that differs from {int(device.hbm_bytes_per_modelled_cycle)} "
                "by more than 50% relative. That observation falsifies the "
                "declared planning coefficient hbm_bytes_per_modelled_cycle."
            ),
        ),
        coeff_pred(
            pid=PRED_FABRIC_BEAT,
            quantity="fabric_bytes_per_modelled_cycle",
            value=int(device.fabric_bytes_per_modelled_cycle),
            units="bytes/modelled_cycle",
            implicated="fabric_bytes_per_modelled_cycle",
            evidence_tier="COST_MODEL",
            kernel=canon_kernel,
            notes=[
                "Declared on-chip fabric beat. COST_MODEL prior.",
                "Honest gap: approximate_cycles does not fold this beat into the critical path.",
            ],
            falsifier=(
                "An observation of on-chip fabric bytes per modelled cycle on the "
                f"inbound U50 that differs from {int(device.fabric_bytes_per_modelled_cycle)} "
                "by more than 50% relative. That observation falsifies the "
                "declared planning coefficient fabric_bytes_per_modelled_cycle."
            ),
        ),
        coeff_pred(
            pid=PRED_HOST_BEAT,
            quantity="host_device_bytes_per_modelled_cycle",
            value=int(device.host_device_bytes_per_modelled_cycle),
            units="bytes/modelled_cycle",
            implicated="host_device_bytes_per_modelled_cycle",
            evidence_tier="COST_MODEL",
            kernel=canon_kernel,
            notes=[
                "USB4/Thunderbolt ~40 Gb/s CLASS prior from H-ROADMAP.md §15.1.",
                "Not a measurement of any cable.",
            ],
            falsifier=(
                "An observation of host<->device bytes per modelled cycle on the "
                f"inbound U50 that differs from {int(device.host_device_bytes_per_modelled_cycle)} "
                "by more than 50% relative. That observation falsifies the "
                "declared planning coefficient host_device_bytes_per_modelled_cycle."
            ),
        ),
        coeff_pred(
            pid=PRED_HOST_QUEUE,
            quantity="host_device_queue_cycles",
            value=int(HOST_DEVICE_QUEUE_CYCLES),
            units="modelled_cycles",
            implicated="host_device_queue_cycles",
            evidence_tier="COST_MODEL",
            kernel=canon_kernel,
            notes=["Declared queue/setup overhead. Not a measured doorbell latency."],
            falsifier=(
                "An observation of host<->device queue/setup overhead in modelled "
                f"cycles that differs from {int(HOST_DEVICE_QUEUE_CYCLES)} by more "
                "than 50% relative. That observation falsifies the declared "
                "planning coefficient host_device_queue_cycles."
            ),
        ),
        coeff_pred(
            pid=PRED_CANON_HBM_BYTES,
            quantity="hbm_per_token_bytes",
            value=int(hbm_c["per_token_bytes"]),
            units="bytes",
            implicated="resident_per_token_byte_identity",
            extra_depends=(),
            abs_tol=0.0,
            evidence_tier="COST_MODEL",
            kernel=canon_kernel,
            notes=[
                "Canonical qGEMV L1 golden (M=2, K=4), weights resident: x_bytes + y_bytes.",
                "Geometry identity. A mismatch falsifies resident_per_token_byte_identity.",
            ],
            falsifier=(
                "An observation of per-token HBM bytes for the canonical qGEMV "
                f"(weights resident) that is not exactly {int(hbm_c['per_token_bytes'])} "
                "bytes. That observation falsifies the model's per-token traffic "
                "identity (resident weights do not move; only activations do)."
            ),
        ),
        coeff_pred(
            pid=PRED_CANON_HBM_CYCLES,
            quantity="hbm_modelled_cycles",
            value=int(hbm_c["modelled_cycles"]),
            units="modelled_cycles",
            implicated="hbm_bytes_per_modelled_cycle",
            evidence_tier="COST_MODEL",
            kernel=canon_kernel,
            notes=[
                "Canonical L1 kernel moves 24 bytes; ceil(24/beat) is 1 for every "
                "beat >= 24. This quantity cannot falsify the HBM beat at this size.",
            ],
            falsifier=(
                "An observation of HBM modelled cycles for the canonical qGEMV "
                f"per-token traffic that differs from {int(hbm_c['modelled_cycles'])} "
                "by more than 50% relative. That observation falsifies "
                "hbm_bytes_per_modelled_cycle (or unmodeled HBM latency)."
            ),
        ),
        coeff_pred(
            pid=PRED_CANON_XFER_CYCLES,
            quantity="host_device_modelled_cycles_total",
            value=int(xfer_c["modelled_cycles_total"]),
            units="modelled_cycles",
            implicated="host_device_bytes_per_modelled_cycle",
            extra_depends=("host_device_queue_cycles",),
            evidence_tier="COST_MODEL",
            kernel=canon_kernel,
            notes=[
                "Canonical L1 host<->device modelled cycles. Queue (32) dominates "
                "the 16+8 byte payload, so a wrong host beat barely moves this number.",
            ],
            falsifier=(
                "An observation of host<->device modelled cycles (H2C+C2H including "
                f"queue) for the canonical qGEMV that differs from {int(xfer_c['modelled_cycles_total'])} "
                "by more than 50% relative. That observation falsifies "
                "host_device_bytes_per_modelled_cycle (and/or host_device_queue_cycles)."
            ),
        ),
        coeff_pred(
            pid=PRED_CANON_CRIT_CYCLES,
            quantity="critical_path_modelled_cycles",
            value=int(cyc_c["modelled_cycles"]),
            units="modelled_cycles",
            implicated="host_device_bytes_per_modelled_cycle",
            extra_depends=(
                "host_device_queue_cycles",
                "hbm_bytes_per_modelled_cycle",
                "pipeline_depth",
                "initiation_interval",
            ),
            evidence_tier="CYCLE_APPROX",
            kernel=canon_kernel,
            notes=[
                "CYCLE_APPROX critical path: H2C + max(compute, HBM) + C2H.",
                "Canonical L1 is host-transfer dominated. Not seconds. clock_hz UNKNOWN.",
            ],
            falsifier=(
                "An observation of critical-path modelled cycles for the canonical "
                f"qGEMV that differs from {int(cyc_c['modelled_cycles'])} by more than "
                "50% relative. That observation falsifies the host-device beat/queue "
                "and/or the declared pipeline depth / initiation interval."
            ),
        ),
        coeff_pred(
            pid=PRED_CANON_DSP,
            quantity="dsp_estimate",
            value=int(est_c["used"]["DSP"]),
            units="DSP",
            implicated="dsp_equals_mac_lanes_times_tile",
            abs_tol=0.0,
            evidence_tier="STATIC",
            kernel=canon_kernel,
            notes=[
                "STATIC identity: DSP = mac_lanes * tile_m. Not a synthesis result.",
            ],
            falsifier=(
                "An observation (synthesis report or board census of the qGEMV engine, "
                f"not the whole device) of DSP count not exactly {int(est_c['used']['DSP'])} "
                "for the canonical kernel. That observation falsifies the declared "
                "DSP = mac_lanes * tile_m mapping used by the STATIC estimator."
            ),
        ),
        coeff_pred(
            pid=PRED_CANON_LUT,
            quantity="lut_estimate",
            value=int(est_c["used"]["LUT"]),
            units="LUT",
            implicated="lut_per_mac_lane",
            rel=1.0,
            evidence_tier="STATIC",
            kernel=canon_kernel,
            notes=[
                "STATIC ESTIMATE from the assumed coefficient table. Factor-of-two tolerance.",
                "Not a synthesis report and not a board utilisation figure.",
            ],
            falsifier=(
                "An observation of LUT count for the canonical qGEMV engine that "
                f"differs from {int(est_c['used']['LUT'])} by more than 100% relative "
                "(outside a factor of two). That observation falsifies lut_per_mac_lane "
                "and/or the assumed decode/DMA/reduce LUT bases."
            ),
        ),
        coeff_pred(
            pid=PRED_PLAN_HBM_BYTES,
            quantity="hbm_per_token_bytes",
            value=int(hbm_p["per_token_bytes"]),
            units="bytes",
            implicated="resident_per_token_byte_identity",
            abs_tol=0.0,
            evidence_tier="COST_MODEL",
            kernel=plan_kernel,
            notes=[
                "Planning-scale qGEMV (M=1024, K=4096), weights resident: x_bytes + y_bytes.",
                "Declared dimensions so HBM bytes exceed one beat. Not a measured workload.",
            ],
            falsifier=(
                "An observation of per-token HBM bytes for the planning-scale qGEMV "
                f"(weights resident) that is not exactly {int(hbm_p['per_token_bytes'])} "
                "bytes. That observation falsifies the model's per-token traffic identity."
            ),
        ),
        coeff_pred(
            pid=PRED_PLAN_HBM_CYCLES,
            quantity="hbm_modelled_cycles",
            value=int(hbm_p["modelled_cycles"]),
            units="modelled_cycles",
            implicated="hbm_bytes_per_modelled_cycle",
            evidence_tier="COST_MODEL",
            kernel=plan_kernel,
            notes=[
                "Planning-scale qGEMV HBM modelled cycles. Sensitive to the HBM beat "
                "(unlike the canonical L1 24-byte kernel).",
            ],
            falsifier=(
                "An observation of HBM modelled cycles for the planning-scale qGEMV "
                f"per-token traffic that differs from {int(hbm_p['modelled_cycles'])} "
                "by more than 50% relative. That observation falsifies "
                "hbm_bytes_per_modelled_cycle."
            ),
        ),
    ]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise PredictionRefused("prediction refused: duplicate prediction id in inbound set")
    for row in rows:
        verify_prediction_seal(row)
        assert_no_hardware_measured(row)
    return rows


def _within_tolerance(
    predicted: int | float,
    observed: int | float,
    tolerance: Mapping[str, Any],
) -> bool:
    kind = str(tolerance.get("kind") or "")
    raw = tolerance.get("value")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ScoringRefused("tolerance.value must be a number")
    t = float(raw)
    if t < 0:
        raise ScoringRefused("tolerance.value must be >= 0")
    p = float(predicted)
    o = float(observed)
    if kind == "absolute":
        return abs(o - p) <= t
    if kind == "relative":
        scale = abs(p)
        if scale == 0.0:
            return abs(o) <= t
        return abs(o - p) <= t * scale
    raise ScoringRefused(f"unknown tolerance kind {kind!r}")


def _observation_value(obs: Mapping[str, Any] | None) -> tuple[int | float | None, str | None, str | None]:
    """Return (value, units, unpin_reason). predicted_value in obs is ignored (anti-re-aim)."""
    if obs is None:
        return None, None, "no observation supplied"
    if not isinstance(obs, Mapping):
        return None, None, "observation is not a mapping"
    if "value" not in obs:
        return None, None, "observation has no value"
    raw = obs.get("value")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None, None, "observation value is not a number"
    units = obs.get("units")
    return raw, None if units is None else str(units), None


def grade_prediction(
    pred: Mapping[str, Any],
    observation: Mapping[str, Any] | None,
    *,
    synthetic_rehearsal: bool,
) -> dict[str, Any]:
    """Grade one sealed prediction. Verifies the seal first. Never re-aims."""
    digest = verify_prediction_seal(pred)
    observed, obs_units, unpin = _observation_value(observation)
    predicted = pred["predicted_value"]
    units = str(pred["units"])
    verdict = "UNPINNED"
    implicated: str | None = None
    delta = None
    within = None
    reason = unpin or ""
    if unpin is None:
        if obs_units is not None and obs_units != units:
            verdict = "REFUSED"
            reason = (
                f"units mismatch: prediction {units!r} vs observation {obs_units!r}; "
                "refusing to convert"
            )
        else:
            within = _within_tolerance(predicted, observed, pred["tolerance"])
            delta = float(observed) - float(predicted)
            if within:
                verdict = "CONFIRMED"
                implicated = None
                reason = "observation within sealed tolerance"
            else:
                verdict = "FALSIFIED"
                implicated = str(pred["implicated_coefficient"])
                reason = (
                    "observation outside sealed tolerance; implicated_coefficient="
                    f"{implicated}"
                )
    body: dict[str, Any] = {
        "delta": delta,
        "depends_on": list(pred.get("depends_on") or []),
        "hardware_measured": False,
        "implicated_coefficient": implicated,
        "kind": "PREDICTION_SCORE",
        "not_an_arrival": bool(synthetic_rehearsal),
        "observed_units": obs_units if obs_units is not None else units,
        "observed_value": observed,
        "predicted_value": predicted,
        "prediction_content_sha256": digest,
        "prediction_id": pred["id"],
        "quantity": pred["quantity"],
        "reason": reason,
        "schema": SCORE_SCHEMA,
        "synthetic_rehearsal": bool(synthetic_rehearsal),
        "tolerance": json.loads(canon_dumps(pred["tolerance"])),
        "units": units,
        "verdict": verdict,
        "wake_condition": WAKE_U50_PRESENT,
        "within_tolerance": within,
    }
    if synthetic_rehearsal:
        body["label"] = SYNTHETIC_REHEARSAL_LABEL
        body["not_a_board_measurement"] = True
    report = emit_evidence("STATIC", body)
    assert_no_hardware_measured(report)
    if report["verdict"] not in SCORE_VERDICTS:
        raise ScoringRefused(f"illegal score verdict {report['verdict']!r}")
    return report


def score_prediction_set(
    predictions: Sequence[Mapping[str, Any]],
    observations: Mapping[str, Mapping[str, Any]] | None,
    *,
    synthetic_rehearsal: bool,
) -> dict[str, Any]:
    """Score a sealed set. Real path is keyed to U50_PRESENT; rehearsal is not an arrival.

    Re-scoring cannot quietly re-aim: each prediction's seal is verified, and
    the predicted_value used is the sealed one. Observation['predicted_value']
    is ignored if present.
    """
    wake = load_u50_wake_condition()
    present = bool(wake.get("present"))
    if synthetic_rehearsal:
        if present:
            raise ScoringRefused(
                "synthetic rehearsal cannot run under U50_PRESENT=true; "
                "that would look like an arrival"
            )
        kind = "SYNTHETIC_ARRIVAL_REHEARSAL"
    else:
        if not present:
            raise ScoringRefused(
                "real scoring is keyed to wake_condition U50_PRESENT; "
                "no U50 is present on this host"
            )
        kind = "U50_PRESENT_SCORE"
    obs_map = dict(observations or {})
    scores = []
    for pred in predictions:
        pid = str(pred.get("id") or "")
        scores.append(
            grade_prediction(
                pred,
                obs_map.get(pid),
                synthetic_rehearsal=synthetic_rehearsal,
            )
        )
    falsified = [s for s in scores if s["verdict"] == "FALSIFIED"]
    confirmed = [s for s in scores if s["verdict"] == "CONFIRMED"]
    unpinned = [s for s in scores if s["verdict"] == "UNPINNED"]
    refused = [s for s in scores if s["verdict"] == "REFUSED"]
    implicated = sorted(
        {s["implicated_coefficient"] for s in falsified if s.get("implicated_coefficient")}
    )
    body: dict[str, Any] = {
        "confirmed_ids": [s["prediction_id"] for s in confirmed],
        "falsified_ids": [s["prediction_id"] for s in falsified],
        "hardware_measured": False,
        "implicated_coefficients": implicated,
        "kind": kind,
        "not_an_arrival": bool(synthetic_rehearsal),
        "prediction_set_sha256": prediction_set_digest(predictions),
        "refused_ids": [s["prediction_id"] for s in refused],
        "scores": scores,
        "synthetic_rehearsal": bool(synthetic_rehearsal),
        "unpinned_ids": [s["prediction_id"] for s in unpinned],
        "u50": wake,
        "wake_condition": WAKE_U50_PRESENT,
    }
    if synthetic_rehearsal:
        body["label"] = SYNTHETIC_REHEARSAL_LABEL
        body["not_a_board_measurement"] = True
        body["not_a_board_result"] = True
    report = emit_evidence("STATIC", body)
    assert_no_hardware_measured(report)
    return report


def synthetic_rehearsal_observations(
    device: DeviceProfile | None = None,
) -> dict[str, dict[str, Any]]:
    """Fabricated observation set. SYNTHETIC REHEARSAL. Not an arrival.

    Diverges HBM and fabric beats on purpose. Leaves the host beat as the
    CONFIRMED control. Derived planning-scale HBM cycles are produced by
    re-running the cost model with the divergent HBM beat so the falsifier
    is consistent with the coefficient error, not a hand-picked number.
    """
    device = device or synthetic_u50_class()
    canonical = canonical_qgemv_kernel()
    planning = planning_qgemv_kernel()
    hbm_c = model_hbm_traffic(canonical, device)
    xfer_c = model_host_device_transfer(canonical, device)
    cyc_c = approximate_cycles(canonical, device, hbm_c, xfer_c)
    est_c = estimate_qgemv_resources(canonical)
    hbm_p = model_hbm_traffic(planning, device)
    divergent = replace(
        device, hbm_bytes_per_modelled_cycle=int(SYNTHETIC_DIVERGENT_HBM_BEAT)
    )
    hbm_p_div = model_hbm_traffic(planning, divergent)

    def obs(value: int | float, units: str) -> dict[str, Any]:
        return {
            "label": SYNTHETIC_REHEARSAL_LABEL,
            "not_a_board_measurement": True,
            "not_an_arrival": True,
            "note": (
                "SYNTHETIC REHEARSAL observation. Fabricated to test scoring. "
                "Not an FPGA result, not HARDWARE_MEASURED."
            ),
            "synthetic_rehearsal": True,
            "units": units,
            "value": value,
        }

    return {
        PRED_HBM_BEAT: obs(int(SYNTHETIC_DIVERGENT_HBM_BEAT), "bytes/modelled_cycle"),
        PRED_FABRIC_BEAT: obs(int(SYNTHETIC_DIVERGENT_FABRIC_BEAT), "bytes/modelled_cycle"),
        PRED_HOST_BEAT: obs(
            int(device.host_device_bytes_per_modelled_cycle), "bytes/modelled_cycle"
        ),
        PRED_HOST_QUEUE: obs(int(HOST_DEVICE_QUEUE_CYCLES), "modelled_cycles"),
        PRED_CANON_HBM_BYTES: obs(int(hbm_c["per_token_bytes"]), "bytes"),
        PRED_CANON_HBM_CYCLES: obs(int(hbm_c["modelled_cycles"]), "modelled_cycles"),
        PRED_CANON_XFER_CYCLES: obs(int(xfer_c["modelled_cycles_total"]), "modelled_cycles"),
        PRED_CANON_CRIT_CYCLES: obs(int(cyc_c["modelled_cycles"]), "modelled_cycles"),
        PRED_CANON_DSP: obs(int(est_c["used"]["DSP"]), "DSP"),
        PRED_CANON_LUT: obs(int(est_c["used"]["LUT"]), "LUT"),
        PRED_PLAN_HBM_BYTES: obs(int(hbm_p["per_token_bytes"]), "bytes"),
        PRED_PLAN_HBM_CYCLES: obs(int(hbm_p_div["modelled_cycles"]), "modelled_cycles"),
    }


def write_sealed_predictions_receipt(
    predictions: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    preds = list(predictions or inbound_board_predictions())
    for pred in preds:
        verify_prediction_seal(pred)
        assert_no_hardware_measured(pred)
    wake = load_u50_wake_condition()
    doc = {
        "schema": PREDICTION_SET_SCHEMA,
        "kind": "SEALED_PREDICTION_SET",
        "purpose": (
            "Pre-registered predictions about the inbound U50, sealed before "
            "arrival so they cannot be rationalised after the fact. Every "
            "number is PREHARDWARE (COST_MODEL / CYCLE_APPROX / STATIC). "
            "None is HARDWARE_MEASURED. There is no FPGA on this host."
        ),
        "wake_condition": WAKE_U50_PRESENT,
        "u50": wake,
        "plan": "inbound-u50-qgemv",
        "predictions": preds,
        "prediction_ids": [p["id"] for p in preds],
        "prediction_set_sha256": prediction_set_digest(preds),
        "gates_keyed_to_wake": list(U50_WAKE_GATES),
        "claim_boundary": (
            "PREHARDWARE sealed predictions. Not a board result, not a "
            "synthesis report, not HARDWARE_MEASURED. Scoring of real "
            "observations is keyed to wake_condition U50_PRESENT."
        ),
    }
    assert_no_hardware_measured(doc)
    return write_receipt(PREDICTION_RECEIPT, doc, "tools/future/hwir.py")


def run_synthetic_arrival_rehearsal(*, write: bool = True) -> dict[str, Any]:
    """Seal, fabricate divergent observations, score. NOT AN ARRIVAL."""
    preds = inbound_board_predictions()
    observations = synthetic_rehearsal_observations()
    scored = score_prediction_set(
        preds, observations, synthetic_rehearsal=True
    )
    doc = {
        "schema": REHEARSAL_SCHEMA,
        "kind": "SYNTHETIC_ARRIVAL_REHEARSAL",
        "label": SYNTHETIC_REHEARSAL_LABEL,
        "not_an_arrival": True,
        "not_a_board_measurement": True,
        "not_a_board_result": True,
        "synthetic_rehearsal": True,
        "hardware_measured": False,
        "wake_condition": WAKE_U50_PRESENT,
        "u50_present": False,
        "purpose": (
            "Prove the scoring path without a board: seal predictions, replay "
            "a fabricated measurement set that deliberately diverges from the "
            "model, and show FALSIFIED rows name the responsible coefficient. "
            "This is a test of the scoring machinery, not evidence about any FPGA."
        ),
        "claim_boundary": SYNTHETIC_REHEARSAL_LABEL,
        "predictions": preds,
        "observations": observations,
        "score": scored,
        "confirmed_ids": list(scored["confirmed_ids"]),
        "falsified_ids": list(scored["falsified_ids"]),
        "implicated_coefficients": list(scored["implicated_coefficients"]),
        "prediction_set_sha256": scored["prediction_set_sha256"],
    }
    assert_no_hardware_measured(doc)
    illegal = collect_evidence_tiers(doc) - set(EVIDENCE_TIERS)
    if illegal:
        raise IllegalEvidenceTier(f"illegal evidence tiers in rehearsal: {sorted(illegal)}")
    if write:
        write_receipt(REHEARSAL_RECEIPT, doc, "tools/future/hwir.py")
    return doc


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

    q_kernel = canonical_qgemv_kernel()
    q_device = synthetic_u50_class()
    q_graph = from_qgemv(q_kernel, q_device)
    q_val = validate(q_graph)
    q_sim = simulate_functional(q_graph, canonical_qgemv_operands())
    q_fit = fit_kernel_to_device(q_kernel, q_device)
    q_pre = run_qgemv_preboard(q_kernel, q_device)
    assert_no_hardware_measured(q_pre)
    q_tiers = collect_evidence_tiers(q_pre)
    overflow_ok = False
    try:
        fit_kernel_to_device(
            overflow_probe_kernel(),
            synthetic_device(lut=4096, dsp=8, bram=2, uram=0, hbm_channels=1),
        )
    except ResourceOverBudget:
        overflow_ok = True
    proofs["qgemv_functional_matches"] = bool(q_sim.get("matches_expected"))
    proofs["qgemv_functional_symbol"] = q_sim.get("engine_symbol")
    proofs["qgemv_graph_valid"] = q_val.ok
    proofs["qgemv_preboard_tiers"] = sorted(q_tiers)
    proofs["qgemv_resource_fit"] = bool(q_fit.get("ok"))
    proofs["qgemv_overflow_refused"] = overflow_ok
    proofs["no_hardware_measured_emitted"] = True
    if not q_val.ok:
        raise RuntimeError(f"qGEMV graph failed validate: {q_val.errors}")
    if not proofs["qgemv_functional_matches"]:
        raise RuntimeError("qGEMV functional sim did not match qgemv_hand expected")
    if q_sim.get("engine_symbol") != "qgemv":
        raise RuntimeError("qGEMV functional sim did not call fpga_engines.qgemv")
    if not overflow_ok:
        raise RuntimeError("overflow probe was not refused")
    if q_tiers - set(EVIDENCE_TIERS):
        raise RuntimeError(f"illegal evidence tiers: {sorted(q_tiers)}")
    if "HARDWARE_MEASURED" in q_tiers:
        raise RuntimeError("HARDWARE_MEASURED leaked into the preboard report")

    preds = inbound_board_predictions()
    for pred in preds:
        verify_prediction_seal(pred)
        if not str(pred.get("falsification_condition") or "").strip():
            raise RuntimeError(f"sealed prediction {pred['id']} has empty falsifier")
    refused_empty = False
    try:
        seal_prediction(
            id="negative.empty_falsifier",
            plan="inbound-u50-qgemv",
            quantity="hbm_bytes_per_modelled_cycle",
            predicted_value=1024,
            units="bytes/modelled_cycle",
            model_coefficients=_subset_coefficients(["hbm_bytes_per_modelled_cycle"]),
            depends_on=["hbm_bytes_per_modelled_cycle"],
            tolerance={"kind": "relative", "value": 0.5},
            falsification_condition="",
            implicated_coefficient="hbm_bytes_per_modelled_cycle",
            evidence_tier="COST_MODEL",
        )
    except PredictionRefused:
        refused_empty = True
    tampered = dict(preds[0])
    tampered["predicted_value"] = float(preds[0]["predicted_value"]) * 2 + 1
    tamper_caught = False
    try:
        verify_prediction_seal(tampered)
    except TamperedPrediction:
        tamper_caught = True
    rehearsal = run_synthetic_arrival_rehearsal(write=False)
    assert_no_hardware_measured(rehearsal)
    if rehearsal.get("kind") != "SYNTHETIC_ARRIVAL_REHEARSAL":
        raise RuntimeError("rehearsal is not labeled SYNTHETIC_ARRIVAL_REHEARSAL")
    if not rehearsal.get("not_an_arrival"):
        raise RuntimeError("rehearsal must declare not_an_arrival")
    if "HARDWARE_MEASURED" in collect_evidence_tiers(rehearsal):
        raise RuntimeError("HARDWARE_MEASURED leaked into the rehearsal")
    named = set(rehearsal.get("implicated_coefficients") or [])
    if "hbm_bytes_per_modelled_cycle" not in named:
        raise RuntimeError("rehearsal did not name hbm_bytes_per_modelled_cycle")
    if "fabric_bytes_per_modelled_cycle" not in named:
        raise RuntimeError("rehearsal did not name fabric_bytes_per_modelled_cycle")
    if PRED_HBM_BEAT not in rehearsal["falsified_ids"]:
        raise RuntimeError("divergent HBM beat was not FALSIFIED")
    if PRED_FABRIC_BEAT not in rehearsal["falsified_ids"]:
        raise RuntimeError("divergent fabric beat was not FALSIFIED")
    if PRED_HOST_BEAT not in rehearsal["confirmed_ids"]:
        raise RuntimeError("host beat control was not CONFIRMED")
    if PRED_PLAN_HBM_CYCLES not in rehearsal["falsified_ids"]:
        raise RuntimeError("planning-scale HBM cycles were not FALSIFIED")
    real_score_refused = False
    try:
        score_prediction_set(preds, synthetic_rehearsal_observations(), synthetic_rehearsal=False)
    except ScoringRefused:
        real_score_refused = True
    proofs["prediction_count"] = len(preds)
    proofs["prediction_set_sha256"] = prediction_set_digest(preds)
    proofs["prediction_without_falsifier_refused"] = refused_empty
    proofs["tampered_prediction_rejected"] = tamper_caught
    proofs["synthetic_rehearsal_not_an_arrival"] = bool(rehearsal.get("not_an_arrival"))
    proofs["synthetic_rehearsal_implicated"] = sorted(named)
    proofs["synthetic_rehearsal_falsified_ids"] = list(rehearsal["falsified_ids"])
    proofs["real_scoring_refused_without_u50"] = real_score_refused
    if not refused_empty:
        raise RuntimeError("empty falsification_condition was not refused")
    if not tamper_caught:
        raise RuntimeError("tampered sealed prediction was not rejected")
    if not real_score_refused:
        raise RuntimeError("real scoring ran without U50_PRESENT")
    u50 = u50_family_profile("u50")
    full_slot = example_full_airflow_server_slot()
    low_slot = example_constrained_low_power_slot()
    real_slot = unpinned_real_carrier()
    brochure_k = brochure_fit_kernel()
    plan_full = admissible_plan(brochure_k, u50, full_slot)
    plan_low = admissible_plan(brochure_k, u50, low_slot)
    proofs["u50_family_ids"] = list(U50_FAMILY_VARIANT_IDS)
    proofs["u50_brochure_fits_full_carrier"] = bool(plan_full.get("ok"))
    proofs["u50_brochure_refused_constrained_carrier"] = bool(plan_low.get("refused"))
    proofs["carrier_downgrade_reduces_host_device_beat"] = int(
        constrain_device_profile(u50, low_slot).host_device_bytes_per_modelled_cycle
    ) < int(constrain_device_profile(u50, full_slot).host_device_bytes_per_modelled_cycle)
    proofs["real_carrier_unpinned"] = (
        real_slot.origin == "REAL_CARRIER_UNPINNED"
        and all(
            (not dict(v).get("pinned"))
            for v in dict(real_slot.field_provenance).values()
        )
    )
    if not proofs["u50_brochure_fits_full_carrier"]:
        raise RuntimeError("brochure-width kernel did not fit the full example carrier")
    if not proofs["u50_brochure_refused_constrained_carrier"]:
        raise RuntimeError("constrained carrier did not refuse the brochure-width kernel")
    if not proofs["carrier_downgrade_reduces_host_device_beat"]:
        raise RuntimeError("constrained carrier did not reduce host<->device beat")
    if not proofs["real_carrier_unpinned"]:
        raise RuntimeError("real comma-device carrier must stay UNPINNED")
    for vid in U50_FAMILY_VARIANT_IDS:
        assert_variant_provenance(u50_family_profile(vid))
        assert_no_hardware_measured(u50_family_profile(vid).to_dict())

    target_ids = list_lowering_targets()
    if len(target_ids) < 2:
        raise RuntimeError(f"need at least two lowering targets, got {target_ids}")
    if PREFERRED_LOWERING_TARGET is not None:
        raise RuntimeError("PREFERRED_LOWERING_TARGET must stay unset")
    q_lower = lower_hwir_all(q_graph)
    fps = {doc["graph_fingerprint"] for doc in q_lower.values()}
    if fps != {q_graph.fingerprint()}:
        raise RuntimeError("lowering targets did not see the same qGEMV graph")
    key_sets = [frozenset(doc) for doc in q_lower.values()]
    if len(set(key_sets)) != 1:
        raise RuntimeError("lowering targets did not share an interface result schema")
    for tid, doc in q_lower.items():
        assert_no_hardware_measured(doc)
        if not doc.get("cannot_express"):
            raise RuntimeError(f"{tid} declared empty cannot_express")
        if not doc.get("handwritten_hdl"):
            raise RuntimeError(f"{tid} declared empty handwritten_hdl")
        if doc.get("preferred"):
            raise RuntimeError(f"{tid} marked preferred; toolchain choice is not encoded")
        if "HARDWARE_MEASURED" in collect_evidence_tiers(doc):
            raise RuntimeError(f"{tid} leaked HARDWARE_MEASURED")
        if not doc.get("artifacts"):
            raise RuntimeError(f"{tid} emitted no source artifacts")
        for art in doc["artifacts"]:
            if art.get("kind") != "SOURCE_ARTIFACT":
                raise RuntimeError(f"{tid} emitted a non-source artifact")
            if "PREHARDWARE" not in str(art.get("body") or ""):
                raise RuntimeError(f"{tid} artifact missing PREHARDWARE label")
    proofs["lowering_target_ids"] = list(target_ids)
    proofs["lowering_same_graph"] = True
    proofs["lowering_interface_identical"] = True
    proofs["lowering_cannot_express_nonempty"] = True
    proofs["lowering_no_preferred_target"] = True
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
    try:
        _flash_doc, _flash_rel, flash_sha = _load_organ_doc(FLASH_ORGAN_MAP)
        del _flash_doc, _flash_rel
    except FileNotFoundError:
        flash_sha = None
    try:
        _qwen_doc, _qwen_rel, qwen_sha = _load_organ_doc(QWEN_ORGAN_MAP)
        del _qwen_doc, _qwen_rel
    except FileNotFoundError:
        qwen_sha = None
    qgemv_preboard = run_qgemv_preboard()
    sealed_path = write_sealed_predictions_receipt()
    rehearsal_doc = run_synthetic_arrival_rehearsal(write=True)
    sealed_doc = load_json(sealed_path)

    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Hardware IR a future physical compiler uses to decide what an FPGA "
            "should become. Lowering targets emit PREHARDWARE source artifacts. "
            "Not a vendor backend, not a bitstream."
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
            {
                "path": "tools/future/fpga_engines.py",
                "what": "qgemv bit-exact functional golden (sequential left-to-right float32)",
                "adequate_as_ir": False,
                "note": (
                    "CALL SITE: simulate_qgemv_functional invokes fpga_engines.qgemv. "
                    "Not forked, not reimplemented. FUNCTIONAL_SIM only."
                ),
            },
            {
                "path": "tools/future/physical_primitives.py",
                "what": "atlas primitive contracts + instantiate/physical_identity",
                "adequate_as_ir": False,
                "note": (
                    "CALL SITE: from_qgemv backs each node via instantiate(..., backend='FPGA'). "
                    "An import of the module is not a call site; instantiate is."
                ),
            },
            {
                "path": "tools/future/fpga_fidelity.py",
                "what": "multi-fidelity ladder over a local StructuralGraph stand-in",
                "adequate_as_ir": False,
                "note": (
                    "Sibling. Explicitly asked not to import HWIR. Different graph type. "
                    "Not imported here; HWIR owns the qGEMV preboard stack on HwirGraph."
                ),
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
            "from_qgemv() lowers a qGEMV-class kernel with evidence-backed atlas primitives (physical_primitives.instantiate)",
            "FUNCTIONAL_SIM calls tools.future.fpga_engines.qgemv (not a reimplementation)",
            "CYCLE_APPROX modelled cycles refuse conversion to seconds",
            "COST_MODEL HBM traffic + host<->device transfer; bandwidth_gbps is not emitted",
            "STATIC resource estimator refuses an engine that exceeds a declared device budget",
            "synthetic U50-class device profile (declared, not a board census) and row-split partitioner",
            "U50-family variants (U50/U50C/U50DD/U50LV) selectable with per-field provenance or explicit UNPINNED",
            "CarrierEnvelope DOWNGRADES a DeviceProfile (PCIe beat, power-derated resources); constrained carrier refuses a brochure-fit kernel",
            "no code path emits HARDWARE_MEASURED",
            "pluggable LoweringTarget interface with equal-citizen HLS-style and Rust-HDL-style source emitters",
            "each lowering target names the primitives it cannot express and the HDL still required",
            "sealed predictions: content-hashed, refused without a falsification condition, keyed to U50_PRESENT",
            "synthetic-arrival rehearsal grades divergent predictions FALSIFIED and names the implicated coefficient; not an arrival",
            "tampered sealed predictions are detected and rejected",
        ],
        "negative_findings": [
            "ACCELERATOR_ARCHITECTURE_ATLAS.json is absent from this worktree HEAD and sparse disk",
            "hcli/physical_graph.py and hcli/agentos/fpga_preboard.py are git-present but not materialized (sparse checkout)",
            "existing hcli.fpga.hwir.v1 is not an IR: no types, no validator, no serdes, organ_operator only",
            "device genome is TARGET_UNSELECTED; HBM channel and resource footprints cannot be known without a board/synthesis",
            "no FPGA board, no bitstream; lowering targets emit PREHARDWARE source artifacts only",
            "AIR exists and executes on Metal; it is not HWIR and was not reused as the spatial IR",
            "PhysicalGraph compile_physical_graph is too unresolved to lower into a resource-accurate HWIR without invention; organ maps are the reality connection",
            "fpga_fidelity.StructuralGraph is a sibling stand-in and was not imported; HWIR owns the qGEMV preboard stack",
            "hcli.agentos.fpga_preboard.HWIR remains schema-only (Codex/hcli surface; cannot edit)",
            "no U50 board; every preboard number is PREHARDWARE",
            "real comma-device carrier is UNPINNED; example envelopes are labeled examples, not that carrier",
            "U50C has no sourced public SKU table; every required field is UNPINNED rather than interpolated",
            "U50LV PCIe lane width is UNPINNED (DS965 Table 1 Gen3 x16 vs VLOW Gen3 x4 note / UG1120 Gen3 x4 XDMA)",
        ],
        "not_an_fpga_backend": True,
        "lowering_targets": {
            "interface": "LoweringTarget",
            "preferred": PREFERRED_LOWERING_TARGET,
            "qgemv": {
                tid: {
                    "artifact_filenames": [
                        a.get("filename") for a in (doc.get("artifacts") or [])
                    ],
                    "cannot_express": list(doc.get("cannot_express") or []),
                    "emits": list(doc.get("emits") or []),
                    "family": doc.get("family"),
                    "graph_fingerprint": doc.get("graph_fingerprint"),
                    "handwritten_hdl": list(doc.get("handwritten_hdl") or []),
                    "supported_primitives": list(doc.get("supported_primitives") or []),
                    "target_id": tid,
                }
                for tid, doc in lower_hwir_all(
                    from_qgemv(canonical_qgemv_kernel(), synthetic_u50_class())
                ).items()
            },
            "registered": lowering_target_manifests(),
            "toolchain_choice": None,
        },
        "claim_boundary": (
            "PREHARDWARE sidecar HWIR artifact. No FPGA board, bitstream, "
            "timing, or HARDWARE_MEASURED number. Resource figures are "
            "ESTIMATES. Cycle figures are APPROXIMATIONS, not measurements."
        ),
        "preboard": qgemv_preboard,
        "sealed_predictions": {
            "receipt": PREDICTION_RECEIPT,
            "kind": "SEALED_PREDICTION_SET",
            "count": len(sealed_doc.get("predictions") or []),
            "ids": list(sealed_doc.get("prediction_ids") or []),
            "prediction_set_sha256": sealed_doc.get("prediction_set_sha256"),
            "wake_condition": WAKE_U50_PRESENT,
            "u50_present": bool((sealed_doc.get("u50") or {}).get("present")),
        },
        "synthetic_arrival_rehearsal": {
            "receipt": REHEARSAL_RECEIPT,
            "kind": "SYNTHETIC_ARRIVAL_REHEARSAL",
            "not_an_arrival": True,
            "not_a_board_result": True,
            "label": SYNTHETIC_REHEARSAL_LABEL,
            "falsified_ids": list(rehearsal_doc.get("falsified_ids") or []),
            "confirmed_ids": list(rehearsal_doc.get("confirmed_ids") or []),
            "implicated_coefficients": list(rehearsal_doc.get("implicated_coefficients") or []),
            "prediction_set_sha256": rehearsal_doc.get("prediction_set_sha256"),
        },
        "u50_family": {
            "generic_class_device_id": "synthetic-u50-class",
            "generic_class_unchanged_envelope": {
                "BRAM": 2016,
                "DSP": 9024,
                "LUT": 872000,
                "URAM": 960,
                "hbm_capacity_bytes": 8 * 1024 ** 3,
                "hbm_channels": 32,
                "note": (
                    "The mixed class envelope is not rewritten. Family SKUs "
                    "are separate selectable profiles."
                ),
            },
            "real_carrier": UNPINNED,
            "real_carrier_note": REAL_CARRIER_NOTE,
            "variants": {
                vid: u50_family_profile(vid).to_dict() for vid in U50_FAMILY_VARIANT_IDS
            },
            "example_carriers": {
                "constrained_low_power": example_constrained_low_power_slot().to_dict(),
                "full_airflow_server": example_full_airflow_server_slot().to_dict(),
                "unpinned_real": unpinned_real_carrier().to_dict(),
            },
        },
        "evidence_tiers_legal": list(EVIDENCE_TIERS),
        "evidence_tiers_illegal": sorted(ILLEGAL_EVIDENCE_TIERS),
    }
    assert_no_hardware_measured(doc)
    return write_receipt(RECEIPT, doc, "tools/future/hwir.py")


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--lower", metavar="ORGAN_MAP")
    ap.add_argument("--organ")
    ap.add_argument("--qgemv", action="store_true", help="run the PREHARDWARE qGEMV pre-board stack")
    ap.add_argument(
        "--emit",
        metavar="TARGET",
        help="lower canonical qGEMV through a registered target (source artifacts only)",
    )
    ap.add_argument(
        "--emit-all",
        action="store_true",
        help="lower canonical qGEMV through every registered target",
    )
    ap.add_argument(
        "--seal-predictions",
        action="store_true",
        help="seal the inbound-board prediction set (PREHARDWARE, keyed to U50_PRESENT)",
    )
    ap.add_argument(
        "--rehearse",
        action="store_true",
        help="synthetic arrival rehearsal; NOT an arrival, NOT a board measurement",
    )
    ap.add_argument(
        "--device",
        default="synthetic-u50-class",
        help="synthetic-u50-class | u50 | u50c | u50dd | u50lv",
    )
    ap.add_argument(
        "--carrier",
        default=None,
        help="full | constrained | unpinned  (unpinned is the real comma-device carrier)",
    )
    a = ap.parse_args()
    if a.emit_all:
        doc = lower_qgemv_targets()
        print(canon_dumps({
            "graph_fingerprint": doc.get("graph_fingerprint"),
            "kind": doc.get("kind"),
            "target_ids": doc.get("target_ids"),
            "cannot_express": {
                tid: (doc.get("targets") or {}).get(tid, {}).get("cannot_express")
                for tid in (doc.get("target_ids") or [])
            },
        }))
        return 0
    if a.emit:
        graph = from_qgemv()
        doc = lower_hwir(graph, a.emit)
        print(canon_dumps(doc))
        return 0
    if a.seal_predictions:
        out = write_sealed_predictions_receipt()
        print(out)
        return 0
    if a.rehearse:
        doc = run_synthetic_arrival_rehearsal(write=True)
        print(canon_dumps({
            "kind": doc.get("kind"),
            "not_an_arrival": doc.get("not_an_arrival"),
            "falsified_ids": doc.get("falsified_ids"),
            "implicated_coefficients": doc.get("implicated_coefficients"),
            "receipt": REHEARSAL_RECEIPT,
        }))
        return 0
    if a.qgemv:
        device = select_device_profile(a.device)
        carrier = None if a.carrier is None else select_carrier_envelope(a.carrier)
        doc = run_qgemv_preboard(device=device, carrier=carrier)
        print(canon_dumps(doc))
        return 0 if doc.get("functional_sim", {}).get("ok") else 1
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
