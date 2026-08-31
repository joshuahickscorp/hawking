"""P6_PRIMITIVE_PROJECTION — map Codex P6/P7 physical candidates onto Hawking.

Codex owns the physical Accelerator frontier. This sidecar does not measure
anything: every P6/P7 row is BLOCKED and unmeasured. The job is compounding —
each candidate becomes a Hawking primitive instance, an HWIR hypothesis when
spatially meaningful, an FPGA spatial form, a transfer-scope ceiling, an
Odyssey III counterexample, and a software lesson for Metal today.

    python3 tools/future/p6_projection.py --build
    python3 tools/future/p6_projection.py --selftest
    python3 -m pytest tools/future/test_p6_projection.py -q

No Era VI. No Odyssey IV. FPGA is Accelerator / Physical Compiler / Fusion,
not a civilization and not "an FPGA backend".
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import (
    HARDWARE_FIELDS,
    HardwareClaimError,
    REPO,
    _assert_no_hardware_claims,
    git,
    load_json,
    write_receipt,
)

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from tools.future import candidate_planner as cp
from tools.future import fpga_engines as fe
from tools.future import hwir
from tools.future import odyssey2_law_store as ols
from tools.future import odyssey3_adversary as o3
from tools.future import physical_primitives as pp

RECEIPT = "P6_PRIMITIVE_PROJECTION.json"
SCHEMA = "hawking.future.p6_projection.v1"
VERSION = 1
RECORDED_BY = "tools.future.p6_projection.py"

ERAS = (
    "I Genesis of the Laboratory",
    "II Compounding Civilization",
    "III Autonomous Science Civilization",
    "IV Synthetic Machine Civilization",
    "V Released Hawking Civilization",
)
ODYSSEYS = (
    "I WHAT IS TRUE?",
    "II WHAT DID HAWKING ALREADY LEARN?",
    "III WHERE IS HAWKING WRONG?",
)

PINNED_QUEUE = REPO / "receipts" / "future" / "evidence" / "ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"
LIVE_QUEUE = REPO / "receipts" / "headless" / "ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"
PINNED_ATLAS = REPO / "receipts" / "future" / "evidence" / "ACCELERATOR_ARCHITECTURE_ATLAS.json"
LIVE_ATLAS = REPO / "receipts" / "headless" / "ACCELERATOR_ARCHITECTURE_ATLAS.json"
PINNED_FRONT = REPO / "receipts" / "future" / "evidence" / "ACCELERATOR_FRONT_G_P6.json"
LIVE_FRONT = REPO / "receipts" / "headless" / "ACCELERATOR_FRONT_G_P6.json"
EVIDENCE_MANIFEST = REPO / "receipts" / "future" / "EVIDENCE_SNAPSHOT.json"

# Odyssey II lattice. GENERIC_VERIFIED is on the lattice and is refused here:
# these candidates are BLOCKED and unmeasured.
TRANSFER_SCOPES = ols.SCOPES
LEGAL_EMITTED_SCOPES = tuple(s for s in TRANSFER_SCOPES if s != "GENERIC_VERIFIED")

UNMAPPED = "UNMAPPED"

# Extra keys a projection must not carry as numbers. HARDWARE_FIELDS already
# cover tps/token_ns/gpu_ns/... ; these are the ones write_receipt would miss.
MEASURED_EFFECT_FIELDS = HARDWARE_FIELDS | frozenset(
    {
        "speedup",
        "latency_ms",
        "throughput",
        "dispatch_count",
        "cuda_era_ms",
        "unified_ms",
        "measured_bandwidth_gb_s",
        "accepted_tps",
        "gpu_ns_per_token",
        "complete_wall_ns_per_accepted_token",
    }
)

CODEX_EXPECTATION_FIELDS = (
    "expected_eliminated_work",
    "expected_dispatch_reduction",
    "expected_gpu_ns_mechanism",
    "expected_active_byte_change",
    "expected_intermediate_byte_reduction",
)


class ProjectionClaimError(ValueError):
    """Emitter refusal: GENERIC_VERIFIED, a measured-effect field, or a bad primitive."""


class QueueUnavailableError(FileNotFoundError):
    """Neither the live queue nor the pinned snapshot is readable."""


# ---------------------------------------------------------------------------
# Path recovery. Never assert a file is absent; record which path was taken.
# ---------------------------------------------------------------------------


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _worktree_hits(rel: str) -> list[Path]:
    """This worktree first, then other git worktrees. Missing is not a result."""
    hits: list[Path] = []
    seen: set[str] = set()
    roots: list[Path] = [REPO]
    blob = git("worktree", "list", "--porcelain")
    for line in blob.splitlines():
        if line.startswith("worktree "):
            roots.append(Path(line.split(" ", 1)[1]))
    for root in roots:
        path = root / rel
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            hits.append(path)
    return hits


def _classify_source(path: Path) -> str:
    text = str(path)
    if "/receipts/future/evidence/" in text.replace("\\", "/"):
        return "pinned_snapshot"
    if "/receipts/headless/" in text.replace("\\", "/"):
        return "live_headless"
    return "live_headless"


def resolve_input(rel_live: str, pinned: Path) -> dict[str, Any]:
    """Prefer live Codex state when present; else the pinned snapshot.

    A sparse worktree may have neither, one, or both. The caller records the
    path taken. Tests must not assert that live is missing.
    """
    live_hits = _worktree_hits(rel_live)
    if live_hits:
        path = live_hits[0]
        return {
            "present": True,
            "path": str(path),
            "rel": _rel(path) if path.exists() else rel_live,
            "evidence_source": _classify_source(path),
            "consulted": [_rel(p) if p.exists() else str(p) for p in live_hits[:4]]
            + ([_rel(pinned)] if pinned.is_file() else []),
        }
    if pinned.is_file():
        return {
            "present": True,
            "path": str(pinned),
            "rel": _rel(pinned),
            "evidence_source": "pinned_snapshot",
            "consulted": [_rel(pinned)],
        }
    return {
        "present": False,
        "path": None,
        "rel": rel_live,
        "evidence_source": None,
        "consulted": [rel_live, _rel(pinned)],
    }


def load_qualification_queue(path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read Codex's queue. Prefer live; fall back to the pinned snapshot."""
    if path is not None:
        doc = cp.load_queue(path)
        src = {
            "present": True,
            "path": str(path),
            "rel": _rel(path),
            "evidence_source": _classify_source(path),
            "consulted": [_rel(path)],
        }
        return doc, src
    resolved = resolve_input(
        "receipts/headless/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
        PINNED_QUEUE,
    )
    if not resolved["present"]:
        try:
            found = cp.find_queue_path()
        except cp.QueueNotFoundError as exc:
            raise QueueUnavailableError(
                "neither live receipts/headless queue nor pinned evidence snapshot "
                "is readable; projection copes with either, not with neither"
            ) from exc
        doc = cp.load_queue(found)
        resolved = {
            "present": True,
            "path": str(found),
            "rel": _rel(found),
            "evidence_source": _classify_source(found),
            "consulted": [_rel(found)],
        }
        return doc, resolved
    return cp.load_queue(Path(resolved["path"])), resolved


def load_front_g_p6() -> tuple[dict[str, Any] | None, dict[str, Any]]:
    resolved = resolve_input(
        "receipts/headless/ACCELERATOR_FRONT_G_P6.json",
        PINNED_FRONT,
    )
    if not resolved["present"]:
        return None, resolved
    return load_json(resolved["path"]), resolved


def load_atlas() -> tuple[dict[str, Any] | None, dict[str, Any]]:
    # Prefer the pinned snapshot for a stable primitive list.
    if PINNED_ATLAS.is_file():
        return load_json(PINNED_ATLAS), {
            "present": True,
            "path": str(PINNED_ATLAS),
            "rel": _rel(PINNED_ATLAS),
            "evidence_source": "pinned_snapshot",
            "consulted": [_rel(PINNED_ATLAS)],
        }
    resolved = resolve_input(
        "receipts/headless/ACCELERATOR_ARCHITECTURE_ATLAS.json",
        PINNED_ATLAS,
    )
    if not resolved["present"]:
        return None, resolved
    return load_json(resolved["path"]), resolved


# ---------------------------------------------------------------------------
# Candidate selection. Derive from the queue; do not hard-code a count.
# ---------------------------------------------------------------------------


def is_p6_p7(candidate_id: str) -> bool:
    """Hyphen-token p6 / p7. Does not match flash-pipeline-* host-ceremony rows."""
    tokens = str(candidate_id).lower().split("-")
    return "p6" in tokens or "p7" in tokens


def p6_p7_rows(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(c) for c in (queue.get("candidates") or []) if isinstance(c, Mapping)]
    selected = [r for r in rows if is_p6_p7(str(r.get("candidate_id") or ""))]
    selected.sort(key=lambda r: str(r.get("candidate_id") or ""))
    return selected


def _mutation_blob(row: Mapping[str, Any]) -> str:
    env = cp.mutation_env(row)
    return " ".join(f"{k}={v}" for k, v in sorted(env.items()))


def _text(row: Mapping[str, Any], *keys: str) -> str:
    parts = [str(row.get("candidate_id") or "")]
    parts.append(_mutation_blob(row))
    for key in keys:
        parts.append(str(row.get(key) or ""))
    return " ".join(parts).lower()


def _has(blob: str, *needles: str) -> bool:
    return any(n.lower() in blob for n in needles)


# ---------------------------------------------------------------------------
# Primitive mapping. Atlas names win. UNMAPPED is a legal honest result.
# ---------------------------------------------------------------------------


def _prims(*names: str) -> list[str]:
    for name in names:
        if name != UNMAPPED and name not in pp.ATLAS_PRIMITIVES:
            raise ProjectionClaimError(
                f"refused to mint primitive {name!r}; atlas seventeen are {pp.ATLAS_PRIMITIVES}"
            )
    return list(names)


def classify_physical_idea(row: Mapping[str, Any]) -> dict[str, Any]:
    """Map one queue row onto atlas primitives and spatial/scope class.

    Named overlays cover the live P6/P7 ids. A keyword fallback keeps a later
    queue row from silently dropping out. Nothing here is a measurement.
    """
    cid = str(row.get("candidate_id") or "")
    blob = _text(
        row,
        "affected_physical_region",
        "expected_eliminated_work",
        "expected_gpu_ns_mechanism",
        "expected_dispatch_reduction",
        "blocked_reason",
    )
    overlay = _OVERLAYS.get(cid)
    if overlay is not None:
        spec = dict(overlay)
        spec["candidate_id"] = cid
        spec["overlay"] = True
        return spec
    return _heuristic(cid, blob)


def _heuristic(cid: str, blob: str) -> dict[str, Any]:
    spatial = True
    form = "streaming_producer_consumer"
    scope = "MODEL_LOCAL"
    engines: list[str] = []
    organ = "routed_plus_shared_expert"
    attack = "causal_control"
    sketch = "fused_routed"
    prims: list[str] = []
    justification = ""

    command_buffer = _has(
        blob,
        "command-buffer",
        "command buffer",
        "single_cb",
        "p6_single_cb",
        "commit/wait",
        "commit/wait",
    )
    reader = _has(blob, "reader_reuse", "reader reuse", "admission") and not _has(
        blob, "expert_cache", "expert cache", "expert source cache"
    )
    cache = _has(blob, "expert_cache", "expert cache", "source cache")
    simd = _has(blob, "simdgroup", "simd-group", "simd group", "_simd", "simd=")
    fused_swiglu = _has(blob, "swiglu") and _has(blob, "fused")
    fused_down = _has(blob, "down") and _has(blob, "fused") and _has(blob, "fp4")
    prefix = _has(blob, "prefix") and _has(blob, "concurrent")
    batched = _has(blob, "batched") and _has(blob, "qat", "quant")
    stack = _has(blob, "epilogue-stack", "epilogue stack", "composed latency")
    mhc = _has(blob, "mhc")

    if command_buffer and not fused_swiglu and not fused_down and not stack:
        prims = _prims("GraphReplay")
        spatial = False
        form = "not_spatially_meaningful"
        scope = "BACKEND_FAMILY"
        attack = "compiler_prior"
        sketch = "none"
        engines = []
        organ = "command_buffer_graph"
        justification = (
            "exact_mutation is a Metal command-buffer topology flag; "
            "expected_eliminated_work is a CPU-visible commit/wait. GraphReplay "
            "covers command-topology reuse. There is no spatial analogue."
        )
    elif reader and not cache:
        prims = _prims("PersistentPhysicalRegion")
        spatial = False
        form = "not_spatially_meaningful"
        scope = "MODEL_LOCAL"
        attack = "measurement_trap"
        sketch = "none"
        organ = "expert_bank"
        justification = (
            "exact_mutation reuses a sealed reader / metadata map. "
            "expected_dispatch_reduction is 0 and the device topology is unchanged. "
            "PersistentPhysicalRegion covers binding/admission residency; no FPGA "
            "spatial region corresponds to host manifest admission."
        )
    elif cache:
        prims = _prims("StationaryRepresentation", "PersistentPhysicalRegion")
        form = "stationary_operand"
        scope = "MACHINE_LOCAL"
        attack = "law_scope"
        sketch = "cache"
        engines = ["dictionary_arithmetic", "codebook_arithmetic"]
        organ = "expert_bank"
        justification = (
            "exact_mutation retains overlapping learned-route expert source bundles. "
            "expected_eliminated_work is repeated source-chunk materialization. "
            "StationaryRepresentation + PersistentPhysicalRegion. FRONT_G_P6 "
            "constrains copy-elision to unified-memory machines."
        )
    elif stack:
        prims = _prims("SpatialPipeline", "GraphReplay")
        form = "persistent_state_machine"
        scope = "MODEL_LOCAL"
        attack = "goodhart"
        sketch = "stack"
        engines = ["qgemv", "routed_expert_accumulate", "semantic_transport_transform"]
        justification = (
            "exact_mutation composes existing guarded P6 flags. "
            "expected_gpu_ns_mechanism says no new arithmetic: SpatialPipeline + GraphReplay."
        )
    elif batched:
        prims = _prims("SpatialPipeline", "GraphReplay")
        form = "explicit_banking"
        scope = "MODEL_LOCAL"
        attack = "causal_control"
        sketch = "batched_qat"
        engines = ["semantic_transport_transform"]
        justification = (
            "exact_mutation packs seven independent QAT launches into one fixed-seven "
            "indirect dispatch. SpatialPipeline (one engine, many banks) + GraphReplay "
            "(command topology). The seven-count is Flash P6 topology."
        )
    elif prefix:
        prims = _prims("SpatialPipeline")
        form = "streaming_producer_consumer"
        scope = "BACKEND_FAMILY"
        attack = "causal_control"
        sketch = "prefix"
        engines = ["semantic_transport_transform", "reductions"]
        justification = (
            "expected_gpu_ns_mechanism places independent Gate and QAT in one concurrent "
            "encoder because they share an input and write disjoint outputs. "
            "SpatialPipeline / transfer-compute overlap. Encoder grouping is Metal."
        )
    elif fused_swiglu:
        prims = _prims("FusedDecodeCompute", "DirectRoutedAccumulate")
        form = "streaming_producer_consumer"
        scope = "BACKEND_FAMILY" if simd else "MODEL_LOCAL"
        attack = "compiler_prior" if simd else "negative_transfer"
        sketch = "simd_fused" if simd else "fused_routed"
        engines = ["qgemv", "routed_expert_accumulate"]
        justification = (
            "exact_mutation fuses routed FP4 gate/up/SwiGLU. expected_eliminated_work "
            "drops separate W1/W3/cast/SwiGLU launches. FusedDecodeCompute + "
            "DirectRoutedAccumulate. Simdgroup occupancy, when set, is LayoutTransform."
        )
        if simd:
            prims = _prims("LayoutTransform", "FusedDecodeCompute", "DirectRoutedAccumulate")
    elif fused_down:
        prims = _prims("FusedDecodeCompute", "DirectRoutedAccumulate")
        form = "streaming_producer_consumer"
        scope = "BACKEND_FAMILY" if simd else "MODEL_LOCAL"
        attack = "compiler_prior" if simd else "negative_transfer"
        sketch = "simd_fused" if simd else "fused_routed"
        engines = ["qgemv", "routed_expert_accumulate"]
        justification = (
            "exact_mutation fuses routed FP4 W2/down with the BF16 boundary. "
            "FusedDecodeCompute + DirectRoutedAccumulate."
        )
        if simd:
            prims = _prims("LayoutTransform", "FusedDecodeCompute", "DirectRoutedAccumulate")
    elif mhc:
        prims = _prims("LayoutTransform")
        form = "explicit_banking"
        scope = "BACKEND_FAMILY"
        attack = "compiler_prior"
        sketch = "mhc"
        engines = ["reductions"]
        organ = "sparse_attention"
        justification = (
            "exact_mutation assigns 24 SIMDgroups to mHC-pre rows. Dispatch count "
            "is unchanged: LayoutTransform (lane/tile ownership), not a new operator."
        )
    elif simd:
        prims = _prims("LayoutTransform")
        if _has(blob, "quant"):
            prims = _prims("LayoutTransform", "FusedDecodeCompute")
            engines = ["semantic_transport_transform"]
            sketch = "simd_quant"
        else:
            engines = ["qgemv"]
            sketch = "simd_layout"
        form = "explicit_banking"
        scope = "BACKEND_FAMILY"
        attack = "compiler_prior"
        justification = (
            "expected_gpu_ns_mechanism is SIMDgroup / block ownership with packed "
            "loads. expected_dispatch_reduction is 0. LayoutTransform."
        )
    else:
        prims = [UNMAPPED]
        spatial = False
        form = "not_spatially_meaningful"
        scope = "MODEL_LOCAL"
        attack = "causal_control"
        sketch = "none"
        organ = ""
        justification = (
            "no atlas primitive fits the exact_mutation / expected_gpu_ns_mechanism "
            "text; UNMAPPED rather than a forced projection"
        )

    return {
        "candidate_id": cid,
        "overlay": False,
        "primitives": prims,
        "spatially_meaningful": spatial,
        "fpga_form": form,
        "scope": scope,
        "engines": engines,
        "organ": organ,
        "attack_family": attack,
        "sketch": sketch,
        "justification": justification,
    }


# Overlays justified from the live queue rows' own fields. Keys are ids, not a
# required set: a queue without one of them still projects the rest.
_OVERLAYS: dict[str, dict[str, Any]] = {
    "flash-p6-act-quant-simdgroup": {
        "primitives": _prims("LayoutTransform", "FusedDecodeCompute"),
        "spatially_meaningful": True,
        "fpga_form": "explicit_banking",
        "scope": "BACKEND_FAMILY",
        "engines": ["semantic_transport_transform"],
        "organ": "routed_plus_shared_expert",
        "attack_family": "compiler_prior",
        "sketch": "simd_quant",
        "justification": (
            "exact_mutation HAWKING_DSV4F_P6_ACT_QUANT_SIMD=1. "
            "expected_gpu_ns_mechanism: one SIMD-group owns each 128-wide block "
            "with packed BF16/uchar4 loads and the same finite-table RNE encoding. "
            "expected_dispatch_reduction: 0 (wider block ownership, same dispatch). "
            "LayoutTransform (lane/block map) + FusedDecodeCompute (BF16->E4M3FN "
            "at the consumer). Simdgroup width is Metal, so BACKEND_FAMILY."
        ),
    },
    "flash-p6-batched-down-qat": {
        "primitives": _prims("SpatialPipeline", "GraphReplay"),
        "spatially_meaningful": True,
        "fpga_form": "explicit_banking",
        "scope": "MODEL_LOCAL",
        "engines": ["semantic_transport_transform"],
        "organ": "routed_plus_shared_expert",
        "attack_family": "causal_control",
        "sketch": "batched_qat",
        "justification": (
            "exact_mutation HAWKING_DSV4F_P6_BATCHED_DOWN_QAT=1. "
            "expected_eliminated_work: six routed and one shared independent "
            "BF16-to-E4M3FN/E8M0 launch boundaries. expected_gpu_ns_mechanism: "
            "one logical 128-value block per thread reads one of seven scratch "
            "tensors through a fixed indirect record. SpatialPipeline (one engine, "
            "seven banks) + GraphReplay (one dispatch topology). The seven-count "
            "is Flash P6; default DOWN to MODEL_LOCAL."
        ),
    },
    "flash-p6-fused-down-shared-combine": {
        "primitives": _prims("FusedDecodeCompute", "DirectRoutedAccumulate"),
        "spatially_meaningful": True,
        "fpga_form": "streaming_producer_consumer",
        "scope": "MODEL_LOCAL",
        "engines": ["qgemv", "routed_expert_accumulate"],
        "organ": "routed_plus_shared_expert",
        "attack_family": "negative_transfer",
        "sketch": "fused_routed",
        "justification": (
            "exact_mutation HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED=1 plus "
            "HAWKING_DSV4F_P6_FP4_DOWN_SHARED_COMBINE_FUSED=1. "
            "expected_eliminated_work: six routed FP4 W2, one shared FP8 W2, seven "
            "FP32-to-BF16 casts, one final combine. expected_gpu_ns_mechanism: one "
            "256-thread row launch reads six resident indirect FP4 records, the "
            "shared FP8 row, then the fixed-six numeric-order BF16 combine. "
            "FusedDecodeCompute + DirectRoutedAccumulate. Flash fixed-six+shared "
            "topology; default DOWN to MODEL_LOCAL. Live-queue id not in the pinned snapshot."
        ),
    },
    "flash-p6-fused-epilogue-stack": {
        "primitives": _prims("SpatialPipeline", "GraphReplay"),
        "spatially_meaningful": True,
        "fpga_form": "persistent_state_machine",
        "scope": "MODEL_LOCAL",
        "engines": ["qgemv", "routed_expert_accumulate", "semantic_transport_transform"],
        "organ": "routed_plus_shared_expert",
        "attack_family": "goodhart",
        "sketch": "stack",
        "justification": (
            "exact_mutation composes existing fused-epilogue, batched-QAT, prefix, "
            "and single-CB flags. expected_gpu_ns_mechanism: compose guarded "
            "primitives with explicit dependency waves; no new arithmetic. "
            "SpatialPipeline + GraphReplay. Interaction, not a summed isolated win."
        ),
    },
    "flash-p6-hash-single-command-buffer": {
        "primitives": _prims("GraphReplay"),
        "spatially_meaningful": False,
        "fpga_form": "not_spatially_meaningful",
        "scope": "BACKEND_FAMILY",
        "engines": [],
        "organ": "command_buffer_graph",
        "attack_family": "compiler_prior",
        "sketch": "none",
        "justification": (
            "exact_mutation HAWKING_DSV4F_P6_SINGLE_CB=1. "
            "expected_eliminated_work: one CPU-visible commit/wait between the "
            "up/SwiGLU wave and the down/combine wave. expected_dispatch_reduction: "
            "command buffers 2->1; 60 dispatches unchanged. GraphReplay covers "
            "command-topology reuse. A forced HWIR spatial graph would invent a "
            "region the mutation does not have; spatially_meaningful is false."
        ),
    },
    "flash-p6-learned-expert-cache-reuse": {
        "primitives": _prims("StationaryRepresentation", "PersistentPhysicalRegion"),
        "spatially_meaningful": True,
        "fpga_form": "stationary_operand",
        "scope": "MACHINE_LOCAL",
        "engines": ["dictionary_arithmetic", "codebook_arithmetic"],
        "organ": "expert_bank",
        "attack_family": "law_scope",
        "sketch": "cache",
        "justification": (
            "exact_mutation HAWKING_DSV4F_P6_LEARNED_EXPERT_CACHE_REUSE=1 (with "
            "reader reuse). expected_eliminated_work: repeated source-chunk "
            "materialization for overlapping learned-route expert bundles. "
            "expected_dispatch_reduction: 0. StationaryRepresentation of packed "
            "FP4/FP8 source + PersistentPhysicalRegion. FRONT_G_P6 claim_boundary: "
            "skipping host<->device copies is unified-memory-specific, so the "
            "transfer ceiling is MACHINE_LOCAL not GENERIC."
        ),
    },
    "flash-p6-learned-reader-reuse": {
        "primitives": _prims("PersistentPhysicalRegion"),
        "spatially_meaningful": False,
        "fpga_form": "not_spatially_meaningful",
        "scope": "MODEL_LOCAL",
        "engines": [],
        "organ": "expert_bank",
        "attack_family": "measurement_trap",
        "sketch": "none",
        "justification": (
            "exact_mutation HAWKING_DSV4F_P6_LEARNED_READER_REUSE=1. "
            "expected_eliminated_work: repeated sealed-reader manifest/index "
            "admission. expected_dispatch_reduction: 0; device topology unchanged. "
            "PersistentPhysicalRegion covers binding residency. Host manifest "
            "admission has no spatial FPGA analogue; spatially_meaningful is false."
        ),
    },
    "flash-p6-prefix-concurrent-wave": {
        "primitives": _prims("SpatialPipeline"),
        "spatially_meaningful": True,
        "fpga_form": "streaming_producer_consumer",
        "scope": "BACKEND_FAMILY",
        "engines": ["semantic_transport_transform", "reductions"],
        "organ": "routed_plus_shared_expert",
        "attack_family": "causal_control",
        "sketch": "prefix",
        "justification": (
            "exact_mutation HAWKING_DSV4F_P6_PREFIX_CONCURRENT=1. "
            "expected_eliminated_work: one compute-encoder boundary between "
            "independent Gate reduction and activation quantization. "
            "expected_gpu_ns_mechanism: both only read the shared input and write "
            "disjoint outputs. SpatialPipeline (overlap). Encoder grouping is Metal."
        ),
    },
    "flash-p6-routed-fp4-down-bf16-fused": {
        "primitives": _prims("FusedDecodeCompute", "DirectRoutedAccumulate"),
        "spatially_meaningful": True,
        "fpga_form": "streaming_producer_consumer",
        "scope": "MODEL_LOCAL",
        "engines": ["qgemv", "routed_expert_accumulate"],
        "organ": "routed_plus_shared_expert",
        "attack_family": "negative_transfer",
        "sketch": "fused_routed",
        "justification": (
            "exact_mutation HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED=1. "
            "expected_eliminated_work: six routed W2 launches and six FP32-to-BF16 "
            "cast launches. expected_gpu_ns_mechanism: one fixed-six indirect launch "
            "consumes authoritative per-expert QAT buffers and writes BF16 W2. "
            "FusedDecodeCompute + DirectRoutedAccumulate. Fixed-six is Flash P6."
        ),
    },
    "flash-p6-routed-fp4-down-bf16-simd": {
        "primitives": _prims("LayoutTransform", "FusedDecodeCompute", "DirectRoutedAccumulate"),
        "spatially_meaningful": True,
        "fpga_form": "explicit_banking",
        "scope": "BACKEND_FAMILY",
        "engines": ["qgemv", "routed_expert_accumulate", "reductions"],
        "organ": "routed_plus_shared_expert",
        "attack_family": "compiler_prior",
        "sketch": "simd_fused",
        "justification": (
            "exact_mutation HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED=1 plus "
            "HAWKING_DSV4F_P6_FP4_DOWN_BF16_SIMD=1. expected_dispatch_reduction: "
            "0 relative to scalar fusion. expected_gpu_ns_mechanism: eight "
            "SIMDgroups per 256-threadgroup split each 32-value FP4 block. "
            "LayoutTransform on top of the fused decode/accumulate. Metal simdgroup."
        ),
    },
    "flash-p6-routed-fp4-gate-up-swiglu-fused": {
        "primitives": _prims("FusedDecodeCompute", "DirectRoutedAccumulate"),
        "spatially_meaningful": True,
        "fpga_form": "streaming_producer_consumer",
        "scope": "MODEL_LOCAL",
        "engines": ["qgemv", "routed_expert_accumulate"],
        "organ": "routed_plus_shared_expert",
        "attack_family": "negative_transfer",
        "sketch": "fused_routed",
        "justification": (
            "exact_mutation HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED=1 with SIMD=0. "
            "expected_eliminated_work: six W1, six W3, twelve FP32-to-BF16 casts, "
            "six SwiGLU launches. expected_gpu_ns_mechanism: one fixed-six "
            "indirect-address launch, paired FP4 reductions, clamp/SwiGLU, device "
            "route weighting. FusedDecodeCompute + DirectRoutedAccumulate."
        ),
    },
    "flash-p6-routed-fp4-gate-up-swiglu-simd": {
        "primitives": _prims("LayoutTransform", "FusedDecodeCompute", "DirectRoutedAccumulate"),
        "spatially_meaningful": True,
        "fpga_form": "explicit_banking",
        "scope": "BACKEND_FAMILY",
        "engines": ["qgemv", "routed_expert_accumulate", "reductions"],
        "organ": "routed_plus_shared_expert",
        "attack_family": "compiler_prior",
        "sketch": "simd_fused",
        "justification": (
            "exact_mutation fused flag plus HAWKING_DSV4F_P6_FP4_GATE_UP_SWIGLU_SIMD=1. "
            "expected_eliminated_work is the same epilogue as scalar fusion; only "
            "row-reduction geometry changes. LayoutTransform on the fused pair. "
            "Simdgroup occupancy is BACKEND_FAMILY."
        ),
    },
    "flash-p6-routed-fp4-simdgroup": {
        "primitives": _prims("LayoutTransform", "TiledProjection"),
        "spatially_meaningful": True,
        "fpga_form": "explicit_banking",
        "scope": "BACKEND_FAMILY",
        "engines": ["qgemv", "reductions"],
        "organ": "routed_plus_shared_expert",
        "attack_family": "compiler_prior",
        "sketch": "simd_layout",
        "justification": (
            "exact_mutation HAWKING_DSV4F_P6_FP4_SIMD=1. expected_eliminated_work: "
            "one serial thread per routed-expert output row. expected_dispatch_reduction: "
            "0 (18 matvec dispatches remain). expected_gpu_ns_mechanism: 64-lane-x-4-row "
            "threadgroup, packed uchar4 loads, SIMDgroup split-K, deterministic row "
            "reduction. LayoutTransform + TiledProjection. Not a new primitive."
        ),
    },
    "flash-p6-shared-fp8-simdgroup": {
        "primitives": _prims("LayoutTransform", "TiledProjection"),
        "spatially_meaningful": True,
        "fpga_form": "explicit_banking",
        "scope": "BACKEND_FAMILY",
        "engines": ["qgemv", "reductions"],
        "organ": "routed_plus_shared_expert",
        "attack_family": "compiler_prior",
        "sketch": "simd_layout",
        "justification": (
            "exact_mutation HAWKING_DSV4F_P6_FP8_SIMD=1. expected_gpu_ns_mechanism: "
            "256-threadgroup, eight SIMDgroups on 128-wide blocks, block partials in "
            "source block order. expected_dispatch_reduction: 0. LayoutTransform + "
            "TiledProjection on shared-expert FP8 matvec."
        ),
    },
    "flash-p7-mhc-pre-simdgroup": {
        "primitives": _prims("LayoutTransform"),
        "spatially_meaningful": True,
        "fpga_form": "explicit_banking",
        "scope": "BACKEND_FAMILY",
        "engines": ["reductions"],
        "organ": "sparse_attention",
        "attack_family": "compiler_prior",
        "sketch": "mhc",
        "justification": (
            "exact_mutation HAWKING_DSV4F_MHC_PRE_SIMD=1. expected_eliminated_work: "
            "one-thread serialization across 24 mHC rows. expected_dispatch_reduction: "
            "0. expected_gpu_ns_mechanism: one 24-SIMDgroup threadgroup, stage RMS "
            "partials, leave Sinkhorn/control source-ordered. LayoutTransform. A "
            "simd_sum reordering is a different float32 number than fpga_engines.reductions "
            "sequential golden (associativity waiver required)."
        ),
    },
}


# ---------------------------------------------------------------------------
# HWIR sketches. validate() is the authority; a failing sketch is a bug.
# ---------------------------------------------------------------------------


def _node(**kwargs: Any) -> hwir.HwirNode:
    kwargs.setdefault("semantics", "noetic_native")
    kwargs.setdefault("assumes_source_tensor_identity", False)
    kwargs.setdefault("dense_weight_materialization", False)
    return hwir.HwirNode(**kwargs)


def _edge(
    eid: str,
    src: str,
    dst: str,
    src_port: str,
    dst_port: str,
    frame: str,
    transform: str = "identity",
) -> hwir.HwirEdge:
    return hwir.HwirEdge(
        id=eid,
        src=src,
        dst=dst,
        src_port=src_port,
        dst_port=dst_port,
        frame_kind=frame,
        in_transit_transform=transform,
    )


def _weight_mem(nid: str = "w") -> hwir.HwirNode:
    return _node(
        id=nid,
        kind="memory",
        primitive="StationaryRepresentation",
        mapping="packed_native stationarity_contract; no dense rematerialization",
        outputs={"packed": "compact_representation_fragment"},
        per_token_transfer=False,
        resident_weight_policy="no_weight_body packed_native",
        lifetime="resident",
    )


def _act_mem(nid: str = "x") -> hwir.HwirNode:
    return _node(
        id=nid,
        kind="memory",
        primitive="StationaryRepresentation",
        mapping="token activation at occupying tier",
        outputs={"x": "activation"},
        per_token_transfer=True,
        lifetime="token",
    )


def hwir_sketch(spec: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    cited = ["exact_mutation", "affected_physical_region", "expected_gpu_ns_mechanism"]
    if not spec.get("spatially_meaningful"):
        return {
            "spatially_meaningful": False,
            "reason": spec["justification"],
            "cited_from": cited + ["expected_eliminated_work"],
            "graph": None,
            "validate": None,
        }
    kind = str(spec.get("sketch") or "fused_routed")
    cid = str(row.get("candidate_id") or spec.get("candidate_id") or "candidate")
    organ = str(spec.get("organ") or "routed_plus_shared_expert")
    nodes, edges, notes = _sketch_parts(kind, spec)
    graph = hwir.HwirGraph(
        model=str(row.get("model") or "Flash"),
        organ=organ,
        source_receipt="ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE",
        qualification="STATIC_ONLY",
        nodes=nodes,
        edges=edges,
        notes=list(notes) + [f"hypothesis for {cid}; STATIC_ONLY; not a board placement"],
    )
    report = hwir.validate(graph)
    if not report.ok:
        raise ProjectionClaimError(
            f"HWIR sketch for {cid} failed validate: {report.errors}"
        )
    return {
        "spatially_meaningful": True,
        "cited_from": cited,
        "organ": organ,
        "ir_node_kinds": sorted({n.kind for n in graph.nodes}),
        "validate": report.to_dict(),
        "graph": graph.to_dict(),
    }


def _sketch_parts(
    kind: str, spec: Mapping[str, Any]
) -> tuple[list[hwir.HwirNode], list[hwir.HwirEdge], list[str]]:
    prims = [p for p in spec.get("primitives") or [] if p != UNMAPPED]
    notes = [spec.get("justification") or ""]
    if kind == "fused_routed":
        dec = _node(
            id="decode",
            kind="representation-decoder",
            primitive="FusedDecodeCompute",
            mapping="projection_plus_decode native_decode; no dense rematerialization",
            inputs={
                "packed": "compact_representation_fragment",
                "x": "activation",
            },
            outputs={"y": "activation"},
        )
        acc = _node(
            id="accum",
            kind="compute",
            primitive="DirectRoutedAccumulate",
            mapping="route_then_native_expert; selected_payload_only",
            inputs={"y": "activation"},
            outputs={"out": "partial_reduction"},
        )
        return (
            [_weight_mem(), _act_mem(), dec, acc],
            [
                _edge("e_w", "w", "decode", "packed", "packed", "compact_representation_fragment"),
                _edge("e_x", "x", "decode", "x", "x", "activation"),
                _edge("e_y", "decode", "accum", "y", "y", "activation"),
            ],
            notes,
        )
    if kind in {"simd_fused", "simd_layout", "simd_quant"}:
        tile = [32] if "fused" in kind else [128]
        lay = _node(
            id="layout",
            kind="compute",
            primitive="LayoutTransform",
            mapping="tile_and_lane_mapping; simdgroup block ownership",
            inputs={"packed": "compact_representation_fragment"},
            outputs={"tiled": "compact_representation_fragment"},
            physical=hwir.PhysicalAttr(tile_shape=tile, banking=8 if "fused" in kind else 4),
        )
        nodes: list[hwir.HwirNode] = [_weight_mem(), _act_mem(), lay]
        edges: list[hwir.HwirEdge] = [
            _edge("e_w", "w", "layout", "packed", "packed", "compact_representation_fragment"),
        ]
        if kind == "simd_quant":
            dec = _node(
                id="quant",
                kind="representation-decoder",
                primitive="FusedDecodeCompute",
                mapping="BF16 activation to packed E4M3FN at the consumer; RNE table",
                inputs={
                    "tiled": "compact_representation_fragment",
                    "x": "activation",
                },
                outputs={"q": "compact_representation_fragment"},
            )
            nodes.append(dec)
            edges.extend(
                [
                    _edge("e_t", "layout", "quant", "tiled", "tiled", "compact_representation_fragment"),
                    _edge("e_x", "x", "quant", "x", "x", "activation"),
                ]
            )
        elif kind == "simd_fused":
            dec = _node(
                id="decode",
                kind="representation-decoder",
                primitive="FusedDecodeCompute",
                mapping="projection_plus_decode native_decode",
                inputs={
                    "tiled": "compact_representation_fragment",
                    "x": "activation",
                },
                outputs={"y": "activation"},
            )
            acc = _node(
                id="accum",
                kind="compute",
                primitive="DirectRoutedAccumulate",
                mapping="route_then_native_expert",
                inputs={"y": "activation"},
                outputs={"out": "partial_reduction"},
            )
            nodes.extend([dec, acc])
            edges.extend(
                [
                    _edge("e_t", "layout", "decode", "tiled", "tiled", "compact_representation_fragment"),
                    _edge("e_x", "x", "decode", "x", "x", "activation"),
                    _edge("e_y", "decode", "accum", "y", "y", "activation"),
                ]
            )
        else:
            proj = _node(
                id="proj",
                kind="compute",
                primitive="TiledProjection" if "TiledProjection" in prims else "LayoutTransform",
                mapping="tiled projection of packed low-bit codes; no dense weight body",
                inputs={
                    "tiled": "compact_representation_fragment",
                    "x": "activation",
                },
                outputs={"y": "partial_reduction"},
            )
            nodes.append(proj)
            edges.extend(
                [
                    _edge("e_t", "layout", "proj", "tiled", "tiled", "compact_representation_fragment"),
                    _edge("e_x", "x", "proj", "x", "x", "activation"),
                ]
            )
        return nodes, edges, notes
    if kind == "prefix":
        gate = _node(
            id="gate",
            kind="compute",
            primitive="TiledProjection",
            mapping="Gate reduction; reads shared input, disjoint output vs QAT",
            inputs={"x": "activation"},
            outputs={"logits": "activation"},
        )
        qat = _node(
            id="qat",
            kind="representation-decoder",
            primitive="FusedDecodeCompute",
            mapping="activation quantization; reads shared input, disjoint output vs Gate",
            inputs={"x": "activation"},
            outputs={"q": "compact_representation_fragment"},
        )
        pipe = _node(
            id="pipe",
            kind="persistent-pipeline",
            primitive="SpatialPipeline",
            mapping="spatial_regions; local_intermediates; producer-consumer overlap",
            inputs={"logits": "activation", "q": "compact_representation_fragment"},
            outputs={"y": "activation"},
        )
        return (
            [_act_mem(), gate, qat, pipe],
            [
                _edge("e_g", "x", "gate", "x", "x", "activation"),
                _edge("e_q", "x", "qat", "x", "x", "activation"),
                _edge("e_pg", "gate", "pipe", "logits", "logits", "activation"),
                _edge("e_pq", "qat", "pipe", "q", "q", "compact_representation_fragment"),
            ],
            notes,
        )
    if kind == "batched_qat":
        qat = _node(
            id="qat",
            kind="representation-decoder",
            primitive="FusedDecodeCompute",
            mapping="fixed-seven indirect QAT; same arithmetic, packed pointer binding",
            inputs={"x": "activation"},
            outputs={"q": "compact_representation_fragment"},
            physical=hwir.PhysicalAttr(banking=7, tile_shape=[128]),
        )
        pipe = _node(
            id="pipe",
            kind="persistent-pipeline",
            primitive="SpatialPipeline",
            mapping="one engine, seven banks; GraphReplay of the dispatch topology",
            inputs={"q": "compact_representation_fragment"},
            outputs={"y": "compact_representation_fragment"},
        )
        replay = _node(
            id="replay",
            kind="persistent-pipeline",
            primitive="GraphReplay",
            mapping="compile_once_reuse of the fixed-seven indirect record",
            inputs={"y": "compact_representation_fragment"},
            outputs={"out": "compact_representation_fragment"},
            lifetime="resident",
        )
        return (
            [_act_mem(), qat, pipe, replay],
            [
                _edge("e_x", "x", "qat", "x", "x", "activation"),
                _edge("e_q", "qat", "pipe", "q", "q", "compact_representation_fragment"),
                _edge("e_r", "pipe", "replay", "y", "y", "compact_representation_fragment"),
            ],
            notes,
        )
    if kind == "stack":
        src = _act_mem()
        pipe = _node(
            id="pipe",
            kind="persistent-pipeline",
            primitive="SpatialPipeline",
            mapping="composed fused epilogues; explicit dependency waves",
            inputs={"x": "activation"},
            outputs={"y": "activation"},
        )
        replay = _node(
            id="replay",
            kind="persistent-pipeline",
            primitive="GraphReplay",
            mapping="compile_once_reuse of the composed P6 graph; measure interaction",
            inputs={"y": "activation"},
            outputs={"z": "activation"},
        )
        return (
            [src, pipe, replay],
            [
                _edge("e_x", "x", "pipe", "x", "x", "activation"),
                _edge("e_y", "pipe", "replay", "y", "y", "activation"),
            ],
            notes,
        )
    if kind == "cache":
        bank = _node(
            id="bank",
            kind="memory",
            primitive="StationaryRepresentation",
            mapping="packed FP4/FP8 expert source bundles; no decoded dense weights",
            outputs={"packed": "compact_representation_fragment"},
            per_token_transfer=False,
            resident_weight_policy="no_weight_body packed_native six_bundle_hot_capacity",
            lifetime="resident",
        )
        region = _node(
            id="region",
            kind="persistent-pipeline",
            primitive="PersistentPhysicalRegion",
            mapping="long_lived_executor; bindings remain valid across overlapping routes",
            inputs={"packed": "compact_representation_fragment"},
            outputs={"held": "compact_representation_fragment"},
            lifetime="sequence",
        )
        return (
            [bank, region],
            [_edge("e_b", "bank", "region", "packed", "packed", "compact_representation_fragment")],
            notes,
        )
    if kind == "mhc":
        residual = _act_mem("residual")
        lay = _node(
            id="layout",
            kind="compute",
            primitive="LayoutTransform",
            mapping="24-SIMDgroup occupancy of mHC-pre rows; RMS partials staged",
            inputs={"x": "activation"},
            outputs={"partial": "partial_reduction"},
            physical=hwir.PhysicalAttr(tile_shape=[24], banking=24),
        )
        return (
            [residual, lay],
            [_edge("e_x", "residual", "layout", "x", "x", "activation")],
            notes,
        )
    # Fallback spatial sketch: stationary packed operand into a native decode.
    dec = _node(
        id="decode",
        kind="representation-decoder",
        primitive=prims[0] if prims and prims[0] in pp.ATLAS_PRIMITIVES else "FusedDecodeCompute",
        mapping="native packed consumer",
        inputs={"packed": "compact_representation_fragment", "x": "activation"},
        outputs={"y": "activation"},
    )
    return (
        [_weight_mem(), _act_mem(), dec],
        [
            _edge("e_w", "w", "decode", "packed", "packed", "compact_representation_fragment"),
            _edge("e_x", "x", "decode", "x", "x", "activation"),
        ],
        notes,
    )


# ---------------------------------------------------------------------------
# FPGA form, transfer scope, Odyssey III, software lesson
# ---------------------------------------------------------------------------


def fpga_realization(spec: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    engines = [e for e in spec.get("engines") or [] if e in fe.ENGINE_FNS]
    unknown = [e for e in spec.get("engines") or [] if e not in fe.ENGINE_FNS]
    if unknown:
        raise ProjectionClaimError(f"fpga engine refs not in fpga_engines.ENGINE_FNS: {unknown}")
    organ = str(spec.get("organ") or "")
    organ_engines = (fe.ORGAN_ENGINE_MAP.get("flash-next") or {}).get(organ) or []
    form = str(spec.get("fpga_form") or "not_spatially_meaningful")
    return {
        "form": form,
        "description": spec["justification"],
        "engine_refs": engines,
        "engine_refs_are_functional_goldens": True,
        "not_an_fpga_backend": True,
        "emits_hdl": False,
        "organ": organ or None,
        "organ_engine_map_refs": [e for e in engines if e in organ_engines],
        "associativity_note": (
            "fpga_engines sequential_left_to_right is the golden. A Metal simd_sum "
            "or spatial tree is a different float32 number without an associativity waiver."
        ),
        "cited_from": ["expected_gpu_ns_mechanism", "affected_physical_region", "exact_mutation"],
        "missing_functional_reference": (
            "no SwiGLU/silu golden in fpga_engines; fused gate/up/SwiGLU cites qgemv "
            "for the projections and routed_expert_accumulate for the combine"
            if "swiglu" in _text(row, "affected_physical_region", "expected_eliminated_work")
            else None
        ),
    }


def _front_g_constraint(front: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not front:
        return None
    boundary = str(front.get("claim_boundary") or "")
    identities = front.get("identities") or {}
    experiment = identities.get("experiment") if isinstance(identities, Mapping) else {}
    return {
        "experiment_class": front.get("experiment_class"),
        "knowledge_level": front.get("knowledge_level"),
        "proof": (experiment or {}).get("proof") if isinstance(experiment, Mapping) else None,
        "claim_boundary_excerpt": boundary[:800],
        "constrains": (
            "Unified-memory copy elision is machine-specific. Discrete GPUs still "
            "pay host-device copies. That ceiling applies to any P6 idea whose "
            "mechanism is 'do not copy / keep resident' (expert-cache reuse)."
        ),
        # Deliberately omit result.speedups / measured_bandwidth_gb_s.
        "numbers_omitted": "FRONT_G_P6 result timings are Codex measurements; sidecar does not restate them",
    }


def transfer_scope(
    spec: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    front_constraint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    scope = str(spec.get("scope") or "MODEL_LOCAL")
    if scope == "GENERIC_VERIFIED":
        raise ProjectionClaimError("blocked unmeasured candidates cannot emit GENERIC_VERIFIED")
    if scope not in LEGAL_EMITTED_SCOPES:
        raise ProjectionClaimError(f"scope {scope!r} is not on the Odyssey II lattice")
    queue_tags = [str(t) for t in (row.get("scope_tags") or [])]
    reason = (
        f"Default DOWN from GENERIC_CANDIDATE: status={row.get('status')!r} is "
        f"BLOCKED/unmeasured, so GENERIC_VERIFIED is illegal. "
        f"Queue scope_tags={queue_tags}. Assigned ceiling {scope}: {spec['justification']}"
    )
    if scope == "MACHINE_LOCAL" and front_constraint:
        reason += " FRONT_G_P6 claim_boundary constrains copy-elision to this memory topology."
    return {
        "scope": scope,
        "lattice": list(TRANSFER_SCOPES),
        "legal_emitted_scopes": list(LEGAL_EMITTED_SCOPES),
        "defaulted_down_from": "GENERIC_CANDIDATE",
        "queue_scope_tags": queue_tags,
        "reason": reason,
        "front_g_p6_constraint": front_constraint if scope == "MACHINE_LOCAL" else None,
        "cited_from": ["scope_tags", "exact_mutation", "expected_gpu_ns_mechanism", "status"],
    }


def odyssey_iii_counterexample(
    spec: Mapping[str, Any],
    row: Mapping[str, Any],
    scope: str,
) -> dict[str, Any]:
    family = str(spec.get("attack_family") or "causal_control")
    if family not in o3.ATTACK_FAMILIES:
        raise ProjectionClaimError(f"attack family {family!r} not in {o3.ATTACK_FAMILIES}")
    cid = str(row.get("candidate_id") or "")
    experiments = {
        "compiler_prior": (
            f"Replay {cid} on a backend whose SIMD/command-buffer geometry is not "
            f"Metal simdgroup/CB (or force the scalar/fused-off control). If the "
            f"concept were BACKEND_FAMILY it would still hold; collapse refutes transfer."
        ),
        "causal_control": (
            f"Hold representation and dispatch topology fixed; disable only the "
            f"{cid} flag. If complete-token wall is unchanged the mechanism is not causal."
        ),
        "negative_transfer": (
            f"Apply the fused/fixed-K idea of {cid} to another MoE that is not "
            f"Flash's fixed-six/fixed-seven hash graph. Failure keeps the idea MODEL_LOCAL."
        ),
        "law_scope": (
            f"Run the resident-source/cache claim of {cid} on a discrete-memory GPU "
            f"where host-device copies are not optional (FRONT_G_P6 claim_boundary). "
            f"If the win vanishes, MACHINE_LOCAL is the ceiling, not a family law."
        ),
        "goodhart": (
            f"Measure the composed stack {cid} as one A/B rather than summing the "
            f"isolated expected_dispatch_reduction strings. If the composed graph "
            f"does not dominate the control, isolated 'wins' were not transferable inward."
        ),
        "measurement_trap": (
            f"A drop in host ceremony or dispatch count for {cid} is not a complete-"
            f"token result (ACCELERATOR_DISPATCH_IS_NOT_THE_COST). If only the "
            f"proxy moves, the transfer claim is a trap."
        ),
    }
    falsifiers = {
        "compiler_prior": "same mutation, different backend geometry, concept does not hold",
        "causal_control": "flag off, complete-token behaviour unchanged",
        "negative_transfer": "non-Flash MoE does not accept the fused topology",
        "law_scope": "discrete-memory machine still pays the copy; UMA elision does not transfer",
        "goodhart": "composed stack does not beat control after interaction",
        "measurement_trap": "proxy (ceremony/dispatch) moves, complete-token wall does not",
    }
    # Odyssey III's own ladder differs from Odyssey II. We name the Odyssey II
    # ceiling we would drop to; REFUTED is the honest floor for a failed transfer.
    down = {
        "GENERIC_CANDIDATE": "MACHINE_LOCAL",
        "MACHINE_LOCAL": "BACKEND_FAMILY",
        "BACKEND_FAMILY": "ARCHITECTURE_FAMILY",
        "ARCHITECTURE_FAMILY": "MODEL_LOCAL",
        "MODEL_LOCAL": "REFUTED",
    }.get(scope, "REFUTED")
    return {
        "family": family,
        "cost_units": o3.FAMILY_COST[family],
        "cost_units_are": "dimensionless Odyssey III planning cost; not a wall-clock",
        "experiment": experiments.get(family, experiments["causal_control"]),
        "falsifier": falsifiers.get(family, falsifiers["causal_control"]),
        "target_scope_if_refuted": down,
        "cited_from": ["expected_gpu_ns_mechanism", "exact_mutation", "blocked_reason"],
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "not_a_measurement": True,
    }


def software_lesson_now(spec: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    cid = str(row.get("candidate_id") or "")
    lessons = {
        "LayoutTransform": (
            "Treat simdgroup width, block ownership and packed-load layout as "
            "compiler objects (LayoutTransform), not kernel folklore. Keep the "
            "sequential reduction order as the numerical contract; simd_sum needs "
            "an associativity waiver against fpga_engines.reductions."
        ),
        "FusedDecodeCompute": (
            "Fuse representation decode with the consumer and do not write the "
            "dense/FP32 staging the fused path no longer needs. Authority scratch "
            "may stay allocated for A/B; it should not become a traffic path."
        ),
        "DirectRoutedAccumulate": (
            "Keep route metadata with the selected payloads in one physical region. "
            "Do not gather inactive experts and do not introduce a second combine."
        ),
        "SpatialPipeline": (
            "Independent producers that share an input and write disjoint outputs "
            "belong in one spatial region / one encoder. Overlap is a measured "
            "window, never an assumed one."
        ),
        "GraphReplay": (
            "Do not insert a host commit/wait between device-resident waves whose "
            "hazards are already explicit. Collapsing command-buffer ceremony is "
            "GraphReplay, not a new kernel."
        ),
        "StationaryRepresentation": (
            "Keep the packed source (not a decoded dense body) as the stationary "
            "operand. FRONT_G_P6: on unified memory the copy can be absent; on a "
            "discrete GPU it cannot. Do not write software that assumes UMA."
        ),
        "PersistentPhysicalRegion": (
            "Reuse bindings and admission state across tokens/routes while they "
            "remain valid. Reconstructing a sealed reader every route change is "
            "host ceremony the Metal path can already drop."
        ),
        "TiledProjection": (
            "Own the projection as tiles that cover the logical result. Packed "
            "uchar4 loads are a layout of the same operator, not a different math."
        ),
    }
    chosen = []
    for name in spec.get("primitives") or []:
        if name in lessons:
            chosen.append({"primitive": name, "lesson": lessons[name]})
    if spec.get("primitives") == [UNMAPPED]:
        chosen.append(
            {
                "primitive": UNMAPPED,
                "lesson": (
                    f"{cid} has no atlas mapping; do not change Metal on the strength of it."
                ),
            }
        )
    return {
        "lessons": chosen,
        "cited_from": ["exact_mutation", "expected_gpu_ns_mechanism", "expected_eliminated_work"],
        "loop": "FPGA-teaches-Metal: spatial ownership, then current software",
    }


def codex_expectations(row: Mapping[str, Any]) -> dict[str, Any]:
    """Carry Codex expected_* strings as expectations, never as results."""
    out = {
        "label": "expectation_not_result",
        "status": str(row.get("status") or "UNKNOWN"),
        "blocked_reason": row.get("blocked_reason"),
        "cited_from": list(CODEX_EXPECTATION_FIELDS) + ["status", "blocked_reason"],
    }
    for key in CODEX_EXPECTATION_FIELDS:
        val = row.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            # A numeric expected_* would look like a measurement. Keep the label,
            # stringify, and refuse to promote it.
            out[key] = {
                "expectation": str(val),
                "numeric_value_omitted": True,
                "label": "expectation_not_result",
            }
        else:
            out[key] = val
    measurements = row.get("measurements") if isinstance(row.get("measurements"), Mapping) else {}
    out["codex_measurement_status"] = measurements.get("status") or "NOT_COPIED"
    return out


# ---------------------------------------------------------------------------
# Emitter guards. A guard nobody has watched fail is not a guard.
# ---------------------------------------------------------------------------


def _illegal_measured(node: Any, path: str = "") -> str | None:
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in MEASURED_EFFECT_FIELDS and isinstance(value, (int, float)) and not isinstance(value, bool):
                return f"{here}={value!r}"
            if key in {"scope", "transfer_scope"} and value == "GENERIC_VERIFIED":
                return f"{here}=GENERIC_VERIFIED"
            if key == "bench" and isinstance(value, dict) and value.get("state") not in (None, "UNKNOWN"):
                return f"{here}.state={value.get('state')!r}"
            hit = _illegal_measured(value, here)
            if hit:
                return hit
    elif isinstance(node, list):
        for i, value in enumerate(node):
            hit = _illegal_measured(value, f"{path}[{i}]")
            if hit:
                return hit
    return None


def assert_projection_legal(doc: Mapping[str, Any]) -> None:
    """Refuse GENERIC_VERIFIED and any numeric measured-effect field."""
    hit = _illegal_measured(doc)
    if hit:
        raise ProjectionClaimError(
            f"emitter refused illegal hardware/scope claim at {hit}. "
            "P6 projections are STATIC_ONLY, bench UNKNOWN, never GENERIC_VERIFIED"
        )
    for rec in doc.get("projections") or []:
        if not isinstance(rec, Mapping):
            continue
        scope = (rec.get("transfer_scope") or {}).get("scope")
        if scope == "GENERIC_VERIFIED":
            raise ProjectionClaimError(
                f"{rec.get('candidate_id')}: GENERIC_VERIFIED refused (BLOCKED, unmeasured)"
            )
        if scope is not None and scope not in LEGAL_EMITTED_SCOPES:
            raise ProjectionClaimError(f"{rec.get('candidate_id')}: illegal scope {scope!r}")
        prims = rec.get("hawking_primitive") or {}
        names = prims.get("names") if isinstance(prims, Mapping) else prims
        for name in names or []:
            if name != UNMAPPED and name not in pp.ATLAS_PRIMITIVES:
                raise ProjectionClaimError(
                    f"{rec.get('candidate_id')}: {name!r} is not an atlas primitive"
                )
    _assert_no_hardware_claims(doc)


def project_candidate(
    row: Mapping[str, Any],
    *,
    front_constraint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = classify_physical_idea(row)
    cid = str(row.get("candidate_id") or spec["candidate_id"])
    prim_names = list(spec["primitives"])
    hwir_hyp = hwir_sketch(spec, row)
    fpga = fpga_realization(spec, row)
    scope = transfer_scope(spec, row, front_constraint=front_constraint)
    attack = odyssey_iii_counterexample(spec, row, scope["scope"])
    lesson = software_lesson_now(spec, row)
    rec = {
        "candidate_id": cid,
        "model": row.get("model"),
        "status": row.get("status"),
        "blocked_reason": row.get("blocked_reason"),
        "affected_physical_region": row.get("affected_physical_region"),
        "exact_mutation": row.get("exact_mutation"),
        "hawking_primitive": {
            "names": prim_names,
            "unmapped": prim_names == [UNMAPPED],
            "justification": spec["justification"],
            "cited_from": ["exact_mutation", "expected_gpu_ns_mechanism", "expected_eliminated_work"],
            "atlas": list(pp.ATLAS_PRIMITIVES),
            "new_primitive_proposed": None,
        },
        "hwir_hypothesis": hwir_hyp,
        "fpga_realization": fpga,
        "transfer_scope": scope,
        "odyssey_iii_counterexample": attack,
        "software_lesson_now": lesson,
        "codex_expectations": codex_expectations(row),
        "qualification": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
    }
    assert_projection_legal({"projections": [rec]})
    return rec


def project_queue(
    queue: Mapping[str, Any],
    *,
    front_constraint: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [project_candidate(row, front_constraint=front_constraint) for row in p6_p7_rows(queue)]


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def _recovered() -> dict[str, Any]:
    return {
        "existed_before_this_module": [
            "tools/future/physical_primitives.py",
            "tools/future/hwir.py",
            "tools/future/fpga_engines.py",
            "tools/future/odyssey2_law_store.py",
            "tools/future/odyssey3_adversary.py",
            "tools/future/candidate_planner.py",
            "receipts/future/PHYSICAL_PRIMITIVES.json",
            "receipts/future/HWIR_V1.json",
            "receipts/future/FPGA_ENGINE_SCHOOL.json",
            "receipts/future/ODYSSEY2_LAW_STORE.json",
            "receipts/future/ODYSSEY3_ADVERSARY.json",
            "receipts/future/evidence/ACCELERATOR_ARCHITECTURE_ATLAS.json",
            "receipts/future/evidence/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
            "receipts/future/FUTURE_SUBSTRATE_HANDOFF.json",
        ],
        "p6_projection_module_existed": False,
        "note": (
            "The 41-system sidecar already owned primitives, HWIR, FPGA goldens, "
            "and both Odysseys. This module maps Codex P6/P7 candidates onto them. "
            "It does not fork a parallel primitive list or an FPGA backend."
        ),
    }


def build(path: Path | None = None) -> Path:
    queue, queue_src = load_qualification_queue(path)
    front, front_src = load_front_g_p6()
    atlas, atlas_src = load_atlas()
    front_constraint = _front_g_constraint(front)
    projections = project_queue(queue, front_constraint=front_constraint)

    spatial = [p for p in projections if p["hwir_hypothesis"]["spatially_meaningful"]]
    not_spatial = [p for p in projections if not p["hwir_hypothesis"]["spatially_meaningful"]]
    unmapped = [p for p in projections if p["hawking_primitive"]["unmapped"]]
    scopes: dict[str, int] = {}
    for p in projections:
        s = p["transfer_scope"]["scope"]
        scopes[s] = scopes.get(s, 0) + 1

    atlas_prims = list(pp.ATLAS_PRIMITIVES)
    if atlas and isinstance(atlas.get("backend_neutral_primitives"), list):
        atlas_prims = [str(x) for x in atlas["backend_neutral_primitives"]]

    evidence_inputs = {
        "qualification_queue": {
            "evidence_source": queue_src.get("evidence_source"),
            "path": queue_src.get("rel") or queue_src.get("path"),
            "present": queue_src.get("present"),
            "consulted": queue_src.get("consulted"),
            "n_candidates_in_queue": len(queue.get("candidates") or []),
            "n_p6_p7": len(projections),
        },
        "front_g_p6": {
            "evidence_source": front_src.get("evidence_source"),
            "path": front_src.get("rel") or front_src.get("path"),
            "present": front_src.get("present"),
            "consulted": front_src.get("consulted"),
        },
        "architecture_atlas": {
            "evidence_source": atlas_src.get("evidence_source"),
            "path": atlas_src.get("rel") or atlas_src.get("path"),
            "present": atlas_src.get("present"),
            "consulted": atlas_src.get("consulted"),
        },
        "evidence_snapshot_manifest": {
            "path": "receipts/future/EVIDENCE_SNAPSHOT.json",
            "present": EVIDENCE_MANIFEST.is_file(),
            "evidence_source": "pinned_snapshot" if EVIDENCE_MANIFEST.is_file() else None,
        },
    }
    # Receipt-level source: live if the queue (the P6/P7 input) came from live.
    evidence_source = queue_src.get("evidence_source") or "pinned_snapshot"

    pinned_ids: list[str] = []
    if PINNED_QUEUE.is_file():
        pinned_doc = load_json(PINNED_QUEUE)
        pinned_ids = [
            str(c.get("candidate_id"))
            for c in (pinned_doc.get("candidates") or [])
            if is_p6_p7(str(c.get("candidate_id") or ""))
        ]
    live_ids = [p["candidate_id"] for p in projections]
    only_live = sorted(set(live_ids) - set(pinned_ids)) if pinned_ids else []
    only_pinned = sorted(set(pinned_ids) - set(live_ids)) if pinned_ids else []

    negative = [
        "No protected GPU lease: every hardware quantity remains UNKNOWN.",
        "No FPGA board, bitstream, or cycle-accurate model; FPGA forms are hypotheses.",
        "fpga_engines has no SwiGLU/silu golden; fused gate/up/SwiGLU cites qgemv + routed_expert_accumulate only.",
        "Nothing here is GENERIC_VERIFIED; candidates are BLOCKED and unmeasured.",
        "Sidecar produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE.",
    ]
    if not front_src.get("present"):
        negative.append(
            "ACCELERATOR_FRONT_G_P6.json was not readable from this worktree or the "
            "pinned snapshot; MACHINE_LOCAL ceiling still cites the recovered claim_boundary text when present."
        )
    if only_live:
        negative.append(
            "Pinned evidence snapshot is missing P6/P7 ids present in the live queue: "
            + ", ".join(only_live)
        )
    if only_pinned:
        negative.append(
            "Live queue is missing P6/P7 ids present in the pinned snapshot: "
            + ", ".join(only_pinned)
        )
    if not PINNED_FRONT.is_file():
        negative.append(
            "FRONT_G_P6 is not in receipts/future/evidence/ (snapshot did not capture it)."
        )

    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Project each Codex P6/P7 physical-qualification candidate onto a "
            "Hawking atlas primitive, an HWIR hypothesis where spatially meaningful, "
            "an FPGA spatial form, a transfer-scope ceiling, an Odyssey III "
            "counterexample, and a software lesson. STATIC_ONLY. No hardware claim."
        ),
        "eras": list(ERAS),
        "odysseys": list(ODYSSEYS),
        "not_an_fpga_backend": True,
        "gpu_authority": False,
        "atlas_primitives": atlas_prims,
        "attack_families": list(o3.ATTACK_FAMILIES),
        "transfer_scope_lattice": list(TRANSFER_SCOPES),
        "projections": projections,
        "counts": {
            "p6_p7_candidates": len(projections),
            "queue_candidates": len(queue.get("candidates") or []),
            "spatially_meaningful": len(spatial),
            "spatially_not_meaningful": len(not_spatial),
            "unmapped": len(unmapped),
            "by_scope": dict(sorted(scopes.items())),
            "by_status": _count([str(p.get("status") or "") for p in projections]),
        },
        "spatially_not_meaningful_ids": [p["candidate_id"] for p in not_spatial],
        "unmapped_ids": [p["candidate_id"] for p in unmapped],
        "front_g_p6": front_constraint,
        "evidence_source": evidence_source,
        "evidence_inputs": evidence_inputs,
        "recovered_implementation": _recovered(),
        "gaps_closed": [
            "P6/P7 queue rows now have a primitive mapping justified from exact_mutation.",
            "Spatially meaningful rows have an HWIR sketch that passes hwir.validate.",
            "Command-buffer / host-admission rows are recorded spatially_meaningful=false.",
            "Transfer-scope ceiling defaults DOWN; GENERIC_VERIFIED is refused by the emitter.",
            "Odyssey III counterexample uses attack-family vocabulary and planning cost_units.",
            "FPGA form cites fpga_engines functional references; no HDL, no backend.",
        ],
        "negative_findings": negative,
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    }
    assert_projection_legal(doc)
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def _count(values: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser(description="Project Codex P6/P7 candidates onto Hawking primitives.")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--queue", type=str, default=None, help="optional queue JSON path")
    args = ap.parse_args()
    out = build(Path(args.queue) if args.queue else None)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
