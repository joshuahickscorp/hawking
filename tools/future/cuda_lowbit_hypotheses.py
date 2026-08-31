"""CUDA low-bit decode literature -> executable Hawking hypotheses.

CUDA is a textbook now and a physical oracle later. This sidecar extracts
physical invariants from publicly-known low-bit decode / MoE kernel design
(NVIDIA as a source school) and emits them as Hawking hypotheses: named
primitive, target organ, cheapest falsifier, cost CLASS not number, backend
candidate. It does not copy vendors, and it makes no local CUDA performance
claim on Apple hardware.

The Architecture Atlas already represents NVIDIA CUDA and CUTLASS/CUTE.
This module enumerates that coverage first and emits only genuine additional
physical ideas. Restatements of mapped techniques are refused.

    python3 tools/future/cuda_lowbit_hypotheses.py --build
    python3 tools/future/cuda_lowbit_hypotheses.py --selftest
    python3 -m pytest tools/future/test_cuda_lowbit_hypotheses.py -q
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import (
    HARDWARE_FIELDS,
    REPO,
    HardwareClaimError,
    _assert_no_hardware_claims,
    git,
    load_json,
    write_receipt,
)
from tools.future.negative_index import canon_family, query, refuse_if_dead
from tools.future.physical_primitives import ATLAS_PRIMITIVES, CONTRACTS

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


RECEIPT = "CUDA_LOWBIT_HYPOTHESES.json"
SCHEMA = "hawking.future.cuda_lowbit.v1"
VERSION = 1
RECORDED_BY = "tools/future/cuda_lowbit_hypotheses.py"
EXPERIMENT_SPEC_SCHEMA = "hawking.accelerator.experiment_spec.v1"
GENERATOR = RECORDED_BY

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

CUDA_SCHOOLS = ("NVIDIA CUDA", "CUTLASS/CUTE")
KNOWLEDGE_SOURCES = ("KNOWN_FROM_REPO", "MODEL_KNOWLEDGE")
BACKENDS = ("METAL", "FPGA", "CUDA", "ANE")

# Cost is a CLASS. A number here is a campaign-level failure.
COST_CLASSES = (
    "ACTIVE_BYTES",
    "HOST_CEREMONY",
    "STATE_MOVEMENT",
    "SYNCHRONIZATION",
    "REDUCTION_TRAFFIC",
    "LAUNCH_CEREMONY",
    "DENSE_INTERMEDIATE",
    "RANDOM_ACCESS_TAX",
    "LAYOUT_TRANSFORM_BYTES",
    "UNPACK_ISSUE",
)
DIRECTIONS = ("REDUCE", "INCREASE", "TRADE")
COST_MEASUREMENTS = ("unmeasured", "UNKNOWN")

REQUIRED_HYPOTHESIS_FIELDS = (
    "id",
    "physical_invariant",
    "hawking_primitive",
    "target_organ",
    "cheapest_falsifier",
    "expected_removed_cost",
    "backend_candidate",
    "transfer_scope",
    "knowledge_source",
    "hypothesis_family",
    "behavior_id",
    "candidate",
)

# Extra keys a hypothesis must not carry as numeric claims. HARDWARE_FIELDS
# are already banned by write_receipt; these catch speedup/latency folklore.
NUMERIC_CLAIM_KEYS = frozenset(
    {
        "speedup",
        "speedup_x",
        "latency",
        "latency_ns",
        "latency_us",
        "latency_ms",
        "x_faster",
        "tokens_per_second",
        "tflops",
        "occupancy_pct",
        "ns_per_token",
        "us_per_token",
        "ms_per_token",
        "gbps",
        "bandwidth",
        "joules",
        "watts",
        "percent_faster",
        "gain",
        "speedup_factor",
        "wall_ms",
        "kernel_ms",
    }
) | HARDWARE_FIELDS

# Families Odyssey II already killed. Catalog items must not use these slugs.
# refuse_if_dead is still the authority; this list is the AVOID surface.
AVOID_FAMILIES = (
    "megakernel",
    "learned_codebook",
    "residual_codebook",
    "binary_quantization",
    "cross_expert_structure",
    "raw_weight_pq_vq",
    "large_expert_cache",
    "low_bpw_materialize_w_expand_to_q4_float_generic_gemv",
    "assume_lower_storage_bpw_is_faster_per_token_density_is_velocity",
    "gemv_kernel_micro_opt_axis_on_qwen3_8",
    "gather_vs_sequential_as_the_q80_qwen3_8_bandwidth_explanation",
    "q5_0_simd_shuffle",
    "mla_phase_4_simdgroup_attn",
    "expert_wave",
    "persistent_8_gib_expert_arena_previous_token_route_prefetch",
)

PINNED_DIR = "receipts/future/evidence"
LIVE_DIR = "receipts/headless"
ATLAS_NAME = "ACCELERATOR_ARCHITECTURE_ATLAS.json"
QUEUE_NAME = "ACCELERATOR_REPATRIATION_QUEUE.json"

# Recovered CUDA-to-Metal inventory (Codex-owned; read-only).
CUDA_INVENTORY = (
    "tools/accelerator/c2m.py",
    "tools/accelerator/c2m_idiom.py",
    "tools/accelerator/cuda_runtime.py",
    "tools/accelerator/gemm.py",
)

_SLUG_RE = re.compile(r"[^a-z0-9]+")
# Numeric performance claims in prose. Structural counts (M=1, N>2, warp of 32)
# do not match: they lack faster/speedup/tps/ns/gbps/tflops.
_SPEED_CLAIM_RE = re.compile(
    r"(?:"
    r"\d+(?:\.\d+)?\s*[x×]\s*(?:faster|speedup|speed)\b"
    r"|\d+(?:\.\d+)?\s*%\s*(?:faster|speedup|slower)\b"
    r"|\b(?:tps|tokens?\s*/\s*s|tokens_per_second)\b\s*[:=]?\s*\d"
    r"|\b\d+(?:\.\d+)?\s*(?:ns|us|µs|ms|gbps|tflops)\b"
    r")",
    re.I,
)
_PROSE_SCAN_FIELDS = (
    "physical_invariant",
    "cheapest_falsifier",
    "candidate",
    "transfer_scope",
    "why_not_a_restatement",
    "control",
)


class HypothesisError(ValueError):
    """Base error for CUDA-lowbit hypothesis admission."""


class MissingFieldError(HypothesisError):
    """admit() refused a record that omitted a required field."""


class UnknownPrimitiveError(HypothesisError):
    """admit() refused a primitive outside the atlas seventeen."""


class NumericClaimError(HardwareClaimError, HypothesisError):
    """Hypothesis carried a numeric speedup, latency, or hardware measurement."""


class ScarRefusal(HypothesisError):
    """Hypothesis matches a recorded negative-science scar."""


class OverlapRefusal(HypothesisError):
    """Hypothesis restates a CUDA/CUTLASS technique the atlas already mapped."""


# ---------------------------------------------------------------------------
# IO — prefer the pinned snapshot; cope with live headless either way.
# ---------------------------------------------------------------------------


def repo_text(rel: str) -> tuple[str | None, str]:
    """Disk first, then git show HEAD:<rel>. Missing here is not repo-absence."""
    path = REPO / rel
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8"), "disk"
    except OSError:
        pass
    blob = git("show", f"HEAD:{rel}")
    if blob:
        return blob, "git"
    return None, "unresolved"


def load_evidence(name: str) -> dict[str, Any]:
    """Load a named evidence file. Prefer pinned snapshot.

    Returns a record with `doc`, `evidence_source` (`pinned_snapshot` or
    `live_headless`), `path_used`, and resolution of both candidate locations.
    Does not treat an unresolved live path as proof the file does not exist
    in another worktree.
    """
    pinned_rel = f"{PINNED_DIR}/{name}"
    live_rel = f"{LIVE_DIR}/{name}"
    pinned_text, pinned_via = repo_text(pinned_rel)
    live_text, live_via = repo_text(live_rel)
    record: dict[str, Any] = {
        "name": name,
        "pinned": {"path": pinned_rel, "resolved_via": pinned_via},
        "live_headless": {"path": live_rel, "resolved_via": live_via},
    }
    if pinned_text is not None:
        record["doc"] = json.loads(pinned_text)
        record["evidence_source"] = "pinned_snapshot"
        record["path_used"] = pinned_rel
        return record
    if live_text is not None:
        record["doc"] = json.loads(live_text)
        record["evidence_source"] = "live_headless"
        record["path_used"] = live_rel
        return record
    raise FileNotFoundError(
        f"{name} unresolved at {pinned_rel} and {live_rel} (disk and HEAD)"
    )


def _slug(text: str) -> str:
    return _SLUG_RE.sub("_", (text or "").strip().lower()).strip("_")


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (dict, list, tuple)) and not value:
        return False
    return True


# ---------------------------------------------------------------------------
# Gap analysis (atlas coverage first; emit only what is missing)
# ---------------------------------------------------------------------------


def _entry_primitive(entries: Iterable[Mapping[str, Any]], behavior_id: str) -> str | None:
    for e in entries:
        if e.get("behavior_id") == behavior_id:
            p = e.get("hawking_primitive")
            return str(p) if p else None
    return None


def coverage_slugs(atlas: Mapping[str, Any]) -> dict[str, set[str]]:
    """Slugs of CUDA/CUTLASS techniques and atlas behavior ids already mapped."""
    techniques: set[str] = set()
    behaviors: set[str] = set()
    for row in atlas.get("source_technique_coverage") or []:
        if row.get("source_school") not in CUDA_SCHOOLS:
            continue
        techniques.add(_slug(str(row.get("source_technique") or "")))
        behaviors.add(_slug(str(row.get("behavior_id") or "")))
    for e in atlas.get("entries") or []:
        schools = e.get("source_architecture_ecosystem") or []
        if any(s in CUDA_SCHOOLS for s in schools):
            behaviors.add(_slug(str(e.get("behavior_id") or "")))
    techniques.discard("")
    behaviors.discard("")
    return {"techniques": techniques, "behaviors": behaviors}


def analyze_gaps(atlas: Mapping[str, Any]) -> dict[str, Any]:
    """Enumerate CUDA/CUTLASS coverage, then name the holes — not a second atlas."""
    schools = list(atlas.get("source_schools") or [])
    coverage = list(atlas.get("source_technique_coverage") or [])
    entries = list(atlas.get("entries") or [])
    cuda_cutlass = [
        {
            "source_school": r.get("source_school"),
            "source_technique": r.get("source_technique"),
            "behavior_id": r.get("behavior_id"),
            "mapped_primitive": _entry_primitive(entries, str(r.get("behavior_id") or "")),
        }
        for r in coverage
        if r.get("source_school") in CUDA_SCHOOLS
    ]
    cuda_cutlass.sort(
        key=lambda r: (
            str(r["source_school"]),
            str(r["behavior_id"]),
            str(r["source_technique"]),
        )
    )
    entry_rows = []
    for e in entries:
        schools_e = list(e.get("source_architecture_ecosystem") or [])
        if not any(s in CUDA_SCHOOLS for s in schools_e):
            continue
        entry_rows.append(
            {
                "behavior_id": e.get("behavior_id"),
                "hawking_primitive": e.get("hawking_primitive"),
                "status": e.get("status"),
                "architecture_independent_invariant": e.get(
                    "architecture_independent_invariant"
                ),
                "source_architecture_ecosystem": schools_e,
            }
        )
    entry_rows.sort(key=lambda r: str(r["behavior_id"]))
    slugs = coverage_slugs(atlas)
    already_covered = [
        {
            "source_school": r["source_school"],
            "source_technique": r["source_technique"],
            "behavior_id": r["behavior_id"],
            "mapped_primitive": r["mapped_primitive"],
            "disposition": "RESTATED_IN_ATLAS_DO_NOT_EMIT",
        }
        for r in cuda_cutlass
    ]
    genuine_gaps = [
        {
            "id": h["id"],
            "hypothesis_family": h["hypothesis_family"],
            "why_not_a_restatement": h["why_not_a_restatement"],
        }
        for h in CATALOG
    ]
    return {
        "source_schools": schools,
        "cuda_and_cutlass_already_in_schools": all(s in schools for s in CUDA_SCHOOLS),
        "n_source_schools": len(schools),
        "n_source_technique_coverage": len(coverage),
        "n_cuda_cutlass_technique_rows": len(cuda_cutlass),
        "n_atlas_entries": len(entries),
        "n_atlas_entries_with_cuda_or_cutlass": len(entry_rows),
        "cuda_cutlass_coverage": cuda_cutlass,
        "atlas_entries_with_cuda_or_cutlass": entry_rows,
        "already_covered": already_covered,
        "genuine_gaps": genuine_gaps,
        "covered_technique_slugs": sorted(slugs["techniques"]),
        "covered_behavior_slugs": sorted(slugs["behaviors"]),
        "overlap_rule": (
            "A hypothesis whose family slugs to an already-mapped CUDA/CUTLASS "
            "source_technique or atlas behavior_id is an OverlapRefusal. "
            "Persistent kernels, fused dequant+compute, double buffering, "
            "warp/subgroup specialization, MoE sort/group/accumulate, and "
            "CUTLASS layout algebra are already mapped; this module does not "
            "restate them."
        ),
    }


# ---------------------------------------------------------------------------
# Numeric / scar / overlap gates
# ---------------------------------------------------------------------------


def _reject_numeric_claims(node: Any, path: str = "") -> None:
    """Raise NumericClaimError on a hardware number or speedup/latency claim."""
    if isinstance(node, dict):
        for key, value in node.items():
            here = f"{path}.{key}" if path else key
            if key in NUMERIC_CLAIM_KEYS and isinstance(value, (int, float)):
                raise NumericClaimError(
                    f"{here} = {value!r}: sidecar has no GPU authority; "
                    "speedup/latency/hardware fields must be null/UNKNOWN/class"
                )
            if key in NUMERIC_CLAIM_KEYS and isinstance(value, str) and re.search(r"\d", value):
                raise NumericClaimError(
                    f"{here} = {value!r}: numeric performance claim in a banned key"
                )
            if key == "expected_removed_cost" and isinstance(value, (int, float)):
                raise NumericClaimError(
                    f"{here} = {value!r}: expected_removed_cost is a CLASS, not a number"
                )
            _reject_numeric_claims(value, here)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _reject_numeric_claims(value, f"{path}[{i}]")
    elif isinstance(node, str) and path.rsplit(".", 1)[-1] in _PROSE_SCAN_FIELDS:
        if _SPEED_CLAIM_RE.search(node):
            raise NumericClaimError(
                f"{path} contains a numeric speedup/latency claim: {node!r}"
            )


def _cost_class(value: Any) -> dict[str, str]:
    if isinstance(value, (int, float)):
        raise NumericClaimError(
            f"expected_removed_cost = {value!r}: class required, number forbidden"
        )
    if not isinstance(value, dict):
        raise MissingFieldError(
            "expected_removed_cost must be {class, direction, measurement, mechanism}"
        )
    klass = value.get("class")
    direction = value.get("direction")
    measurement = value.get("measurement")
    mechanism = value.get("mechanism")
    if klass not in COST_CLASSES:
        raise HypothesisError(f"expected_removed_cost.class {klass!r} not in COST_CLASSES")
    if direction not in DIRECTIONS:
        raise HypothesisError(
            f"expected_removed_cost.direction {direction!r} not in DIRECTIONS"
        )
    if measurement not in COST_MEASUREMENTS:
        raise HypothesisError(
            f"expected_removed_cost.measurement {measurement!r} must be unmeasured/UNKNOWN"
        )
    if not _present(mechanism):
        raise MissingFieldError("expected_removed_cost.mechanism is required")
    for banned in ("value", "amount", "delta", "factor", "ns"):
        if isinstance(value.get(banned), (int, float)):
            raise NumericClaimError(
                f"expected_removed_cost.{banned} is a fabricated number; use class"
            )
    return {
        "class": str(klass),
        "direction": str(direction),
        "measurement": str(measurement),
        "mechanism": str(mechanism),
    }


def _overlap_hit(family: str, slugs: Mapping[str, set[str]]) -> str | None:
    s = _slug(family)
    if not s:
        return None
    if s in slugs.get("techniques", ()):
        return f"family {family!r} slugs to mapped CUDA/CUTLASS source_technique"
    if s in slugs.get("behaviors", ()):
        return f"family {family!r} slugs to mapped atlas behavior_id"
    return None


def _scar_gate(raw: Mapping[str, Any]) -> None:
    proposal = {
        "hypothesis_family": raw.get("hypothesis_family"),
        "technique": raw.get("hypothesis_family"),
        "model": raw.get("model"),
        "organ": raw.get("organ") or raw.get("target_organ"),
        "representation": raw.get("representation"),
        "machine": raw.get("machine"),
    }
    refusal = refuse_if_dead(proposal)
    if refusal:
        raise ScarRefusal(
            "known-dead hypothesis; rediscovery is not free. "
            f"scar_id={refusal.get('scar_id')} "
            f"source_path={refusal.get('source_path')} "
            f"family={refusal.get('hypothesis_family')} "
            f"reopen={refusal.get('reopen_condition')}"
        )


# ---------------------------------------------------------------------------
# Catalog — genuine additional physical ideas only
# ---------------------------------------------------------------------------


def _cost(klass: str, direction: str, mechanism: str) -> dict[str, str]:
    return {
        "class": klass,
        "direction": direction,
        "measurement": "unmeasured",
        "mechanism": mechanism,
    }


CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "H-CLB-001",
        "hypothesis_family": "dequant_site_tier",
        "behavior_id": "fused_decode_compute",
        "hawking_primitive": "MemoryTierIdentity",
        "compared_primitives": ("FusedDecodeCompute",),
        "target_organ": "mlp",
        "physical_invariant": (
            "Unpacked values occupy more bytes than packed ones. The legal "
            "dequant site is the highest-bandwidth, lowest-capacity memory tier "
            "that still feeds the arithmetic unit without writing a dense "
            "intermediate to a slower tier. Unpack-at-REGISTER and "
            "unpack-at-THREADGROUP are two executables of the same semantic "
            "program (MemoryTierIdentity). Global dense rematerialization is "
            "already forbidden by FusedDecodeCompute; this law names the LOCAL "
            "tier of the unpack, not the fusion itself."
        ),
        "cheapest_falsifier": (
            "STATIC: unpack-at-REGISTER and unpack-at-THREADGROUP instances that "
            "share a semantic program id must not hash to the same physical "
            "identity. If they do, MemoryTierIdentity is already false and this "
            "site-choice is undefined. DIAGNOSTIC (non-promoting): if billed "
            "unpacked working-set bytes do not differ by site after fence cost, "
            "the site-choice law is false. Promotion still requires a protected "
            "complete-wall A/B under a GPU lease this sidecar does not have."
        ),
        "expected_removed_cost": _cost(
            "DENSE_INTERMEDIATE",
            "REDUCE",
            "avoid writing unpacked values to a slower tier than the consumer",
        ),
        "backend_candidate": "METAL",
        "source_schools": ("NVIDIA CUDA",),
        "transfer_scope": (
            "CUDA register vs shared-memory dequant site -> Metal thread vs "
            "threadgroup; FPGA ACCEL_SRAM is an Accelerator analogue, not a "
            "backend this module builds. Vendor cp.async/TMA is not the law."
        ),
        "knowledge_source": "MODEL_KNOWLEDGE",
        "why_not_a_restatement": (
            "Atlas maps 'fused dequant + compute' and 'shared-memory staging' "
            "as separate techniques. It does not state the unpack-site as a "
            "MemoryTierIdentity choice between REGISTER and THREADGROUP."
        ),
        "candidate": (
            "unpack packed weights at REGISTER vs THREADGROUP; never write a "
            "dense intermediate to a slower tier"
        ),
        "control": (
            "fused dequant that materializes unpacked values into a slower tier "
            "than the arithmetic unit (the dead expand-to-generic-GEMV path is "
            "a different, already-refuted family)"
        ),
    },
    {
        "id": "H-CLB-002",
        "hypothesis_family": "decode_gemv_stationarity_choice",
        "behavior_id": "stationary_representation",
        "hawking_primitive": "StationaryRepresentation",
        "compared_primitives": (),
        "target_organ": "mlp",
        "physical_invariant": (
            "For a packed projection Y = X W, the cheaper stationary operand is "
            "the one whose bytes times reuse exceed the other's. At decode "
            "(M near 1) W dominates and must be stationary while X streams. At "
            "prefill (M large) X can dominate and the stationarity contract "
            "flips. Stationarity is a function of live M, not a kernel-style "
            "constant and not 'keep packed weights resident' as a slogan."
        ),
        "cheapest_falsifier": (
            "STATIC: a plan that declares W stationary at decode and X "
            "stationary at prefill must carry two distinct stationarity "
            "contracts (two PhysicalGraph identities). If one contract is "
            "reused across M, the choice law is already false. DIAGNOSTIC: an "
            "activation-stationary decode GEMV that bills more weight movement "
            "than the weight-stationary twin falsifies the M-near-1 side. "
            "Promotion requires a protected complete-wall A/B this sidecar "
            "cannot take."
        ),
        "expected_removed_cost": _cost(
            "ACTIVE_BYTES",
            "REDUCE",
            "stop moving the dominant operand every token; stream the other",
        ),
        "backend_candidate": "METAL",
        "source_schools": ("NVIDIA CUDA", "TPU/systolic"),
        "transfer_scope": (
            "TPU weight/activation/output stationarity as kinds; CUDA decode "
            "GEMV as the missing M-dependent choice. Metal keeps packed W "
            "resident at decode; FPGA BRAM/URAM residency is the same law at "
            "ACCEL_SRAM, not an FPGA backend."
        ),
        "knowledge_source": "MODEL_KNOWLEDGE",
        "why_not_a_restatement": (
            "CUDA school maps stationarity only to prefix/cache reuse. The "
            "existing repatriation spec qwen27-stationary-packed-weight keeps "
            "packed W resident. Neither states stationarity as a function of "
            "live M (decode vs prefill)."
        ),
        "candidate": (
            "choose the stationary operand of a packed GEMV from live M; "
            "weight-stationary at decode, reconsider at prefill"
        ),
        "control": "packed-weight residency with a single stationarity contract for all M",
    },
    {
        "id": "H-CLB-003",
        "hypothesis_family": "occupancy_unit_cooperative_simd",
        "behavior_id": "layout_algebra",
        "hawking_primitive": "LayoutTransform",
        "compared_primitives": ("TiledProjection",),
        "target_organ": "mlp",
        "physical_invariant": (
            "Occupancy and reduction identity are defined at the cooperative "
            "SIMD width (CUDA warp / Metal simdgroup), not at the scalar "
            "thread. A kernel written at thread granularity pays a shuffle or "
            "barrier tax to reconstruct the width the arithmetic unit already "
            "is. The occupancy unit is a LayoutTransform object, not a vendor "
            "intrinsic. This is not a megakernel and not a simd-shuffle codec."
        ),
        "cheapest_falsifier": (
            "STATIC: a PhysicalGraph that names thread as the occupancy unit "
            "and also names a warp/simdgroup MMA/simdgroup_matrix consumer "
            "without a LayoutTransform between them is invalid. DIAGNOSTIC: if "
            "a thread-granularity kernel and a simdgroup-width kernel of the "
            "same numerical contract bill identical shuffle/sync edges, the "
            "occupancy-unit law is false. C2M refusing __syncwarp/__shfl is a "
            "frontend gap, not a physical counterexample. Promotion requires "
            "a protected complete-wall A/B this sidecar cannot take."
        ),
        "expected_removed_cost": _cost(
            "SYNCHRONIZATION",
            "REDUCE",
            "remove shuffle/barrier tax paid to reconstruct the unit the ALU already is",
        ),
        "backend_candidate": "METAL",
        "source_schools": ("NVIDIA CUDA",),
        "transfer_scope": (
            "CUDA warp / MMA fragment as occupancy unit -> Metal simdgroup / "
            "simdgroup_matrix. tools/accelerator/c2m.py currently refuses "
            "__syncwarp and __shfl by name; that is recovered inventory, not a "
            "claimed Metal timing."
        ),
        "knowledge_source": "KNOWN_FROM_REPO",
        "why_not_a_restatement": (
            "Atlas lists 'warp/subgroup specialization' under layout_algebra. "
            "It does not state that cooperative SIMD width is the occupancy "
            "unit of executable identity, nor the Metal simdgroup analogue as "
            "the same LayoutTransform law. Dead levers q5_0_simd_shuffle and "
            "mla_phase_4_simdgroup_attn are codec/attention specifics; this is "
            "not those families."
        ),
        "candidate": (
            "name cooperative SIMD width as the occupancy unit of the packed "
            "GEMV LayoutTransform (Metal simdgroup analogue of a CUDA warp)"
        ),
        "control": "thread-granularity kernel that reconstructs SIMD width with shuffle/barrier",
    },
    {
        "id": "H-CLB-004",
        "hypothesis_family": "n_stage_software_pipeline",
        "behavior_id": "async_double_buffer",
        "hawking_primitive": "AsyncPrefetch",
        "compared_primitives": ("DoubleBufferedTile", "SpatialPipeline"),
        "target_organ": "mlp",
        "physical_invariant": (
            "The number of in-flight tiles N is set by the ratio of movement "
            "latency to compute latency, occupancy-capped. N=2 is double "
            "buffering (already mapped). N>2 is legal only when the extra "
            "buffers' occupancy cost is less than the uncovered movement. "
            "Overlap is a measured window with explicit producer/consumer "
            "ownership, never an assumed one. Vendor cp.async/TMA is a "
            "mechanism, not the law. The Metal analogue is explicit "
            "multi-buffer threadgroup staging plus barriers, not a TMA copy."
        ),
        "cheapest_falsifier": (
            "STATIC: a plan with N>2 that does not name N distinct ownership-"
            "safe buffers, or that treats overlap as assumed rather than a "
            "window, is invalid under AsyncPrefetch/DoubleBufferedTile "
            "preconditions. DIAGNOSTIC: if billed overlap is zero after fence "
            "cost, extra stages cannot hide movement and the N>2 law is false. "
            "Promotion requires a protected complete-wall A/B this sidecar "
            "cannot take."
        ),
        "expected_removed_cost": _cost(
            "STATE_MOVEMENT",
            "TRADE",
            "hide movement behind compute; extra stages cost occupancy",
        ),
        "backend_candidate": "METAL",
        "source_schools": ("NVIDIA CUDA", "CUTLASS/CUTE"),
        "transfer_scope": (
            "CUDA multi-stage cp.async / software pipeline -> Metal explicit "
            "multi-buffer threadgroup staging with barriers. Apple has no TMA "
            "in this inventory (cuda_runtime.py refuses cudaMemcpyAsync and "
            "streams). FPGA spatial pipeline depth is the same N-choice at "
            "ACCEL_SRAM, not an FPGA backend."
        ),
        "knowledge_source": "MODEL_KNOWLEDGE",
        "why_not_a_restatement": (
            "Atlas maps asynchronous staging, double buffering, stream overlap "
            "and shared-memory staging, all as the N=2 special case. It does "
            "not treat pipeline depth N as a compiler object derived from the "
            "movement/compute ratio."
        ),
        "candidate": (
            "choose in-flight tile count N from movement/compute ratio; N=2 is "
            "double buffering; N>2 only when occupancy cost < uncovered movement"
        ),
        "control": "N=2 double-buffered tile with overlap assumed rather than owned",
    },
    {
        "id": "H-CLB-005",
        "hypothesis_family": "persist_vs_replay_decode_loop",
        "behavior_id": "persistent_physical_region",
        "hawking_primitive": "PersistentPhysicalRegion",
        "compared_primitives": ("GraphReplay", "MoveOrRecompute"),
        "target_organ": "decode",
        "physical_invariant": (
            "A decode step whose topology is stable may persist the executable "
            "(amortize launch and binding) or replay a captured graph "
            "(amortize topology rebuild). These are two organizations of the "
            "same static skeleton, not stacked by default. Persist wins when "
            "per-token host entry dominates; replay wins when occupancy wants "
            "a fresh grid and topology rebuild is the remaining tax. Stacking "
            "them without a measured remaining tax is ceremony. This is not a "
            "megakernel: the layer graph stays a graph."
        ),
        "cheapest_falsifier": (
            "STATIC: a plan that both persists the token loop and replays a "
            "per-token graph without naming which remaining tax each removes "
            "is invalid (double-counted ceremony). DIAGNOSTIC: if persist and "
            "replay bill the same host-ceremony edges, they are not two "
            "organizations and the choice law is false. Promotion requires a "
            "protected complete-wall A/B this sidecar cannot take."
        ),
        "expected_removed_cost": _cost(
            "LAUNCH_CEREMONY",
            "REDUCE",
            "pay either persist or replay for the remaining host tax, not both",
        ),
        "backend_candidate": "METAL",
        "source_schools": ("NVIDIA CUDA",),
        "transfer_scope": (
            "CUDA persistent kernel vs CUDA graph replay as decode-loop "
            "organizations -> Metal long-lived encoder/pipeline vs captured "
            "command-buffer replay. cuda_runtime.py refuses cudaGraphLaunch; "
            "that is a T1 frontend gap, not a physical counterexample."
        ),
        "knowledge_source": "KNOWN_FROM_REPO",
        "why_not_a_restatement": (
            "Atlas already maps persistent kernels and graph capture/replay as "
            "separate entries. It does not state the XOR choice for the decode "
            "token loop, nor forbid stacking them without a remaining tax. "
            "Family megakernel (full-layer MoE megakernel) is a recorded scar "
            "and is not this hypothesis."
        ),
        "candidate": (
            "choose persist XOR replay for a stable decode token loop; do not "
            "stack without a named remaining tax"
        ),
        "control": "per-token host relaunch of a stable topology with neither persist nor replay",
    },
    {
        "id": "H-CLB-006",
        "hypothesis_family": "split_reduction_dimension",
        "behavior_id": "collective_region",
        "hawking_primitive": "TiledProjection",
        "compared_primitives": ("CollectiveRegion", "SemanticTransportEdge"),
        "target_organ": "mlp",
        "physical_invariant": (
            "Partitioning the reduction dimension (K of a decode GEMV, or the "
            "KV sequence of decode attention) creates occupancy at the cost of "
            "a subsequent partial-sum reduction. The split is legal iff the "
            "recovered occupancy exceeds the reduction traffic plus fence. An "
            "unmodeled reduction is a hidden CollectiveRegion. Decode "
            "attention split-KV is the same law at organ attention, not a "
            "separate FlashAttention restatement."
        ),
        "cheapest_falsifier": (
            "STATIC: a split-K / split-KV plan whose partial-sum edge is "
            "missing from the PhysicalGraph is invalid (unmodeled "
            "CollectiveRegion). DIAGNOSTIC: if billed reduction traffic is "
            "zero while K (or KV) was partitioned, the law is already false. "
            "Promotion requires a protected complete-wall A/B this sidecar "
            "cannot take."
        ),
        "expected_removed_cost": _cost(
            "REDUCTION_TRAFFIC",
            "TRADE",
            "buy occupancy by partitioning K; pay a modeled partial-sum reduction",
        ),
        "backend_candidate": "METAL",
        "source_schools": ("NVIDIA CUDA",),
        "transfer_scope": (
            "CUDA split-K GEMM / FlashDecoding split-KV -> Metal tiled "
            "projection plus an explicit reduction edge. No CUDA occupancy "
            "number is claimed. FPGA reduction trees are the same law on "
            "Accelerator, not an FPGA backend."
        ),
        "knowledge_source": "MODEL_KNOWLEDGE",
        "why_not_a_restatement": (
            "Split-K and its reduction cost do not appear in CUDA/CUTLASS "
            "source_technique_coverage. FlashAttention-style IO-aware "
            "attention is mapped as SpatialPipeline (prefill tiling). Decode "
            "split of the reduction dimension is not that technique."
        ),
        "candidate": (
            "partition K (or decode-KV) across tiles for occupancy; account "
            "the partial-sum reduction as a CollectiveRegion / semantic edge"
        ),
        "control": "unsplit reduction with no partial-sum edge and no occupancy claim",
    },
    {
        "id": "H-CLB-007",
        "hypothesis_family": "selection_density_sparse_vs_dense",
        "behavior_id": "sparse_conditional_execution",
        "hawking_primitive": "SparseSkip",
        "compared_primitives": ("DirectRoutedAccumulate",),
        "target_organ": "moe",
        "physical_invariant": (
            "Gather/scatter of selected tokens is cheaper than dense-with-mask "
            "iff selection density is low enough that the gather's random-"
            "access tax plus the sparse kernel stays below the dense kernel "
            "that would compute the omitted work. At high selection density, "
            "dense-with-mask dominates because gather is a random-access tax "
            "on bytes the dense path would have streamed sequentially. Atlas "
            "DirectRoutedAccumulate states the few-active-tokens side; this "
            "law names the density threshold that flips the choice."
        ),
        "cheapest_falsifier": (
            "STATIC: a MoE plan that gathers selected tokens and also claims "
            "dense-with-mask occupancy without naming selection density is "
            "invalid (two organizations, no predicate). DIAGNOSTIC: if a "
            "high-density route still prefers gather after billing the "
            "random-access tax, or a low-density route prefers dense after "
            "billing omitted work, the threshold law is false. This is not "
            "'gather vs sequential explains Q80 bandwidth' (a recorded "
            "model-specific scar). Promotion requires a protected complete-"
            "wall A/B this sidecar cannot take."
        ),
        "expected_removed_cost": _cost(
            "RANDOM_ACCESS_TAX",
            "TRADE",
            "pay gather only while selection density keeps it cheaper than masked dense",
        ),
        "backend_candidate": "METAL",
        "source_schools": ("NVIDIA CUDA",),
        "transfer_scope": (
            "CUDA grouped-GEMM gather/scatter vs masked dense MoE -> Metal "
            "DirectRoutedAccumulate vs SparseSkip-as-dense-mask. Model is "
            "unspecified so MODEL_SPECIFIC Q80 gather scars do not attach."
        ),
        "knowledge_source": "MODEL_KNOWLEDGE",
        "why_not_a_restatement": (
            "Atlas DirectRoutedAccumulate already says few active tokens make "
            "route metadata and selected-payload locality dominate. SparseSkip "
            "says skip is legal when omitted work is proven zero. Neither "
            "states the density threshold that prefers dense-with-mask."
        ),
        "candidate": (
            "choose MoE gather/scatter vs dense-with-mask from selection "
            "density; gather is not unconditionally cheaper"
        ),
        "control": "unconditional gather/scatter of selected experts regardless of density",
    },
    {
        "id": "H-CLB-008",
        "hypothesis_family": "lut_working_set_tier",
        "behavior_id": "stationary_representation",
        "hawking_primitive": "StationaryRepresentation",
        "compared_primitives": ("MemoryTierIdentity", "FusedDecodeCompute"),
        "target_organ": "codebook",
        "physical_invariant": (
            "A codebook, scale table, or LUT of an already-chosen "
            "representation is a StationaryRepresentation whose working set "
            "must fit in a named memory tier. Spilling it to a slower tier "
            "converts a broadcast or lookup into a gather. The LUT is not "
            "the weight body; it is a distinct resident operand with its own "
            "stationarity contract. This is not a learned-codebook or "
            "residual-codebook codec (both recorded dead)."
        ),
        "cheapest_falsifier": (
            "STATIC: a plan that names a codebook organ without a memory-tier "
            "identity for the LUT working set is invalid. DIAGNOSTIC: if a "
            "LUT that does not fit its declared tier still bills as a "
            "broadcast (not a gather), the spill-to-gather law is false. "
            "Promotion requires a protected complete-wall A/B this sidecar "
            "cannot take."
        ),
        "expected_removed_cost": _cost(
            "RANDOM_ACCESS_TAX",
            "REDUCE",
            "keep the LUT working set in a tier that preserves broadcast/lookup",
        ),
        "backend_candidate": "METAL",
        "source_schools": ("NVIDIA CUDA",),
        "transfer_scope": (
            "CUDA constant-memory / shared-memory LUT residency -> Metal "
            "threadgroup or resident buffer for scale tables of a packed "
            "representation already in the graph (NR CODEBOOK_LOOKUP). FPGA "
            "BRAM/URAM codebook residency is the same law, not an FPGA backend."
        ),
        "knowledge_source": "KNOWN_FROM_REPO",
        "why_not_a_restatement": (
            "Atlas organs already include codebook on fused_decode_compute and "
            "stationary_representation. NR vocabulary already has "
            "CODEBOOK_LOOKUP. Neither states that spilling the LUT working "
            "set to a slower tier converts lookup into gather."
        ),
        "candidate": (
            "give the LUT/scale table its own stationarity contract and a "
            "named memory tier; spill converts lookup to gather"
        ),
        "control": "LUT treated as metadata with no tier identity and no working-set bound",
    },
    {
        "id": "H-CLB-009",
        "hypothesis_family": "arithmetic_unit_fragment_constraint",
        "behavior_id": "layout_algebra",
        "hawking_primitive": "LayoutTransform",
        "compared_primitives": (),
        "target_organ": "mlp",
        "physical_invariant": (
            "An arithmetic unit consumes operands only in a named fragment "
            "layout (MMA fragment, simdgroup matrix, tensor-core K-major vs "
            "MN-major). A packing that is compact in storage but illegal at "
            "the unit FORCES a LayoutTransform; those transform bytes are "
            "part of the kernel cost. Compact-in-memory is not compact-at-"
            "the-unit. Atlas layout_algebra already says legal layouts have "
            "different costs; this law names the illegal-at-unit constraint "
            "that can erase a packing win."
        ),
        "cheapest_falsifier": (
            "STATIC: a packed layout that is not a legal fragment of the named "
            "arithmetic unit and that has no LayoutTransform edge before the "
            "consumer is an invalid graph. DIAGNOSTIC: if billed "
            "LayoutTransform bytes for an illegal packing are zero, the "
            "constraint was not enforced and the law is false. Promotion "
            "requires a protected complete-wall A/B this sidecar cannot take."
        ),
        "expected_removed_cost": _cost(
            "LAYOUT_TRANSFORM_BYTES",
            "TRADE",
            "packing wins only after the forced fragment transform is billed",
        ),
        "backend_candidate": "METAL",
        "source_schools": ("CUTLASS/CUTE", "NVIDIA CUDA"),
        "transfer_scope": (
            "CUTLASS logical/physical/tile/lane algebra plus tensor-core "
            "fragment shapes -> Metal simdgroup matrix operand layout. "
            "Fragment shapes themselves are MODEL_KNOWLEDGE; the atlas has "
            "the legal-layout cost law, not the illegal-at-unit force."
        ),
        "knowledge_source": "MODEL_KNOWLEDGE",
        "why_not_a_restatement": (
            "CUTLASS coverage is 'logical/physical/tile/lane layout algebra'; "
            "CUDA coverage includes low-bit layouts, swizzling, register "
            "blocking. All of those are cost differences among legal layouts. "
            "None states that an illegal fragment forces a transform whose "
            "bytes can erase the packing win."
        ),
        "candidate": (
            "treat arithmetic-unit fragment layout as a hard LayoutTransform "
            "constraint; compact packing that cannot be consumed is not compact"
        ),
        "control": "packed storage layout consumed directly by an MMA/simdgroup unit without a named transform",
    },
)


def _experiment_id(h: Mapping[str, Any]) -> str:
    fam = _slug(str(h.get("hypothesis_family") or h.get("id") or "unnamed"))
    return f"cuda-lowbit-{fam}"


def as_repatriation_spec(
    h: Mapping[str, Any], atlas_fingerprint: str
) -> dict[str, Any]:
    """Shape the existing repatriation queue vocabulary can ingest.

    Status is HYPOTHESIS, not READY: this sidecar has no GPU lease and must
    not look like a queued protected run. command is not executable here.
    """
    eid = _experiment_id(h)
    cost = h["expected_removed_cost"]
    effect_key = {
        "ACTIVE_BYTES": "active_bytes",
        "HOST_CEREMONY": "host_ceremony",
        "STATE_MOVEMENT": "state_movement",
        "SYNCHRONIZATION": "synchronization",
        "REDUCTION_TRAFFIC": "state_movement",
        "LAUNCH_CEREMONY": "host_ceremony",
        "DENSE_INTERMEDIATE": "active_bytes",
        "RANDOM_ACCESS_TAX": "active_bytes",
        "LAYOUT_TRANSFORM_BYTES": "active_bytes",
        "UNPACK_ISSUE": "host_ceremony",
    }[cost["class"]]
    expected_effect = {
        effect_key: {
            "direction": cost["direction"],
            "measurement": "unmeasured",
            "mechanism": cost["mechanism"],
        }
    }
    return {
        "schema": EXPERIMENT_SPEC_SCHEMA,
        "experiment_id": eid,
        "atlas_fingerprint": atlas_fingerprint,
        "backend": str(h["backend_candidate"]).lower(),
        "behavior_id": h["behavior_id"],
        "benchmark_mode": "complete_useful_wall_ns_authority",
        "candidate": h["candidate"],
        "claim_boundary": (
            "Executable hypothesis specification for repatriation ingest. "
            "Not a performance result, capability result, or source-product "
            "claim. STATIC_ONLY; no CUDA device on this Apple host."
        ),
        "command": [
            "PLAN_ONLY",
            "sidecar-has-no-gpu-authority",
            "not-an-executable-protected-accelerator-bench",
        ],
        "control": h.get("control") or "atlas-mapped control of the parent behavior",
        "expected_effect": expected_effect,
        "falsifier": h["cheapest_falsifier"],
        "kernel_lowering": h["hawking_primitive"],
        "metrics": [
            "complete_useful_wall",
            "active_bytes_per_token",
            "dispatches",
            "synchronization",
            "host_ceremony",
            "fallback_count",
            "capability_verified",
        ],
        "model_identity": "unspecified-decode-lowbit",
        "nr_identity": f"NR_CUDA_LOWBIT:{h['hypothesis_family']}",
        "nx_identity": "UNKNOWN",
        "organ": h["target_organ"],
        "organ_range": {
            "organ": h["target_organ"],
            "range": "candidate-defined",
            "selection": "candidate-defined; no implicit full-model claim",
        },
        "output_receipt_path": (
            f"receipts/headless/ACCELERATOR_REPATRIATION/{eid}.json"
        ),
        "promotion": {
            "required_benchmark_class": "QUALIFIED_PROTECTED",
            "required_evidence_class": "HAWKING_PROTECTED_VERIFIED",
            "requires_complete_active_bytes_or_explicit_absence": True,
            "requires_independent_capability": True,
            "requires_zero_fallback": True,
            "sidecar_cannot_promote": True,
        },
        "runner": {
            "detached": False,
            "lease": None,
            "protected_window": False,
            "requires_quiescence": True,
            "resumable": False,
            "shell": False,
            "executable_here": False,
            "reason": "STATIC_ONLY sidecar; no GPU authority",
        },
        "source_evidence": [
            f"{PINNED_DIR}/{ATLAS_NAME}",
            "tools/future/physical_primitives.py",
            "tools/future/negative_index.py",
        ],
        "state_session_inputs": {
            "blocked_reason": "HYPOTHESIS_PLAN_ONLY_NO_GPU",
            "memory_tier_is_executable_identity": True,
            "session": "not_started",
        },
        "status": "HYPOTHESIS",
        "target": {
            "backend": str(h["backend_candidate"]).lower(),
            "model_identity": "unspecified-decode-lowbit",
            "nr_identity": f"NR_CUDA_LOWBIT:{h['hypothesis_family']}",
            "nx_identity": "UNKNOWN",
            "organ": h["target_organ"],
            "organ_range": {
                "organ": h["target_organ"],
                "range": "candidate-defined",
                "selection": "candidate-defined; no implicit full-model claim",
            },
        },
        "verification_mode": "structural_then_diagnostic_then_protected",
        "hypothesis_id": h["id"],
        "knowledge_source": h["knowledge_source"],
        "measurement_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }


def admit(
    raw: Mapping[str, Any],
    *,
    atlas: Mapping[str, Any] | None = None,
    slugs: Mapping[str, set[str]] | None = None,
) -> dict[str, Any]:
    """Validate and freeze a hypothesis. Raises on numeric claims, scars, overlap."""
    missing = [f for f in REQUIRED_HYPOTHESIS_FIELDS if not _present(raw.get(f))]
    if missing:
        raise MissingFieldError(f"hypothesis {raw.get('id')!r} missing {missing}")

    primitive = str(raw["hawking_primitive"])
    if primitive not in ATLAS_PRIMITIVES:
        raise UnknownPrimitiveError(
            f"{raw.get('id')}: hawking_primitive {primitive!r} is not an atlas primitive"
        )
    if str(raw["knowledge_source"]) not in KNOWLEDGE_SOURCES:
        raise HypothesisError(
            f"{raw.get('id')}: knowledge_source must be KNOWN_FROM_REPO or MODEL_KNOWLEDGE"
        )
    if str(raw["backend_candidate"]) not in BACKENDS:
        raise HypothesisError(
            f"{raw.get('id')}: backend_candidate {raw['backend_candidate']!r} not in {BACKENDS}"
        )

    _reject_numeric_claims(dict(raw))
    _assert_no_hardware_claims(dict(raw))

    family = str(raw["hypothesis_family"])
    # Negative index is the authority: cite the scar when it fires.
    _scar_gate(raw)
    # AVOID catches families Odyssey II already killed even when the index
    # has only MODEL_SPECIFIC scars and the proposal omitted a model.
    if canon_family(family) in AVOID_FAMILIES or _slug(family) in AVOID_FAMILIES:
        raise ScarRefusal(
            f"{raw.get('id')}: family {family!r} is on the AVOID surface "
            "(recorded negative-science family; refuse before emit)"
        )

    if slugs is None and atlas is not None:
        slugs = coverage_slugs(atlas)
    if slugs is not None:
        hit = _overlap_hit(family, slugs)
        if hit:
            raise OverlapRefusal(f"{raw.get('id')}: {hit}")

    cost = _cost_class(raw["expected_removed_cost"])
    compared = tuple(raw.get("compared_primitives") or ())
    illegal_compared = [p for p in compared if p not in ATLAS_PRIMITIVES]
    if illegal_compared:
        raise UnknownPrimitiveError(
            f"{raw.get('id')}: compared_primitives not in atlas: {illegal_compared}"
        )

    out = {
        "id": str(raw["id"]),
        "hypothesis_family": family,
        "behavior_id": str(raw["behavior_id"]),
        "hawking_primitive": primitive,
        "compared_primitives": list(compared),
        "target_organ": str(raw["target_organ"]),
        "physical_invariant": str(raw["physical_invariant"]),
        "cheapest_falsifier": str(raw["cheapest_falsifier"]),
        "expected_removed_cost": cost,
        "backend_candidate": str(raw["backend_candidate"]),
        "source_schools": list(raw.get("source_schools") or CUDA_SCHOOLS[:1]),
        "transfer_scope": str(raw["transfer_scope"]),
        "knowledge_source": str(raw["knowledge_source"]),
        "why_not_a_restatement": str(raw.get("why_not_a_restatement") or ""),
        "candidate": str(raw["candidate"]),
        "control": str(raw.get("control") or ""),
        "status": "HYPOTHESIS",
        "measurement_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "gpu_authority": False,
        "evidence_class": "STATIC_ONLY",
        "era": "II Compounding Civilization",
        "odyssey": "I WHAT IS TRUE?",
        "fpga_is": (
            "Accelerator / Physical Compiler / Fusion — never its own "
            "civilization; this module does not build an FPGA backend"
        ),
        "cuda_is": (
            "a source school and a future physical oracle, not an execution "
            "backend on this Apple host"
        ),
    }
    if raw.get("model"):
        out["model"] = raw["model"]
    return out


def emit_catalog(atlas: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Admit catalog items. Return (hypotheses, restatement_refusals)."""
    slugs = coverage_slugs(atlas)
    admitted: list[dict[str, Any]] = []
    for raw in CATALOG:
        admitted.append(admit(raw, atlas=atlas, slugs=slugs))
    admitted.sort(key=lambda h: h["id"])
    probes = (
        {
            "id": "PROBE-restatement-persistent-kernels",
            "hypothesis_family": "persistent_kernels",
            "behavior_id": "persistent_physical_region",
            "hawking_primitive": "PersistentPhysicalRegion",
            "target_organ": "decode",
            "physical_invariant": "restatement probe of atlas persistent kernels",
            "cheapest_falsifier": "this probe must be refused as overlap",
            "expected_removed_cost": _cost("LAUNCH_CEREMONY", "REDUCE", "probe"),
            "backend_candidate": "METAL",
            "transfer_scope": "probe",
            "knowledge_source": "KNOWN_FROM_REPO",
            "candidate": "restatement of persistent kernels",
        },
        {
            "id": "PROBE-restatement-fused-dequant",
            "hypothesis_family": "fused_dequant_compute",
            "behavior_id": "fused_decode_compute",
            "hawking_primitive": "FusedDecodeCompute",
            "target_organ": "mlp",
            "physical_invariant": "restatement probe of atlas fused dequant + compute",
            "cheapest_falsifier": "this probe must be refused as overlap",
            "expected_removed_cost": _cost("DENSE_INTERMEDIATE", "REDUCE", "probe"),
            "backend_candidate": "METAL",
            "transfer_scope": "probe",
            "knowledge_source": "KNOWN_FROM_REPO",
            "candidate": "restatement of fused dequant + compute",
        },
    )
    refusals: list[dict[str, Any]] = []
    for probe in probes:
        try:
            admit(probe, atlas=atlas, slugs=slugs)
        except OverlapRefusal as e:
            refusals.append(
                {
                    "id": probe["id"],
                    "hypothesis_family": probe["hypothesis_family"],
                    "refused": True,
                    "reason": str(e),
                    "disposition": "OVERLAP_REFUSAL",
                }
            )
        else:
            refusals.append(
                {
                    "id": probe["id"],
                    "hypothesis_family": probe["hypothesis_family"],
                    "refused": False,
                    "reason": "probe was admitted; overlap guard did not fire",
                    "disposition": "GUARD_FAILURE",
                }
            )
    refusals.sort(key=lambda r: r["id"])
    return admitted, refusals


def query_avoid_families() -> list[dict[str, Any]]:
    """Ask the negative index about families we must not rediscover."""
    rows: list[dict[str, Any]] = []
    for family in AVOID_FAMILIES:
        hits = query(hypothesis_family=family)
        refuse_hits = [h for h in hits if h.get("refuse_eligible")]
        sample = refuse_hits[0] if refuse_hits else (hits[0] if hits else None)
        rows.append(
            {
                "family": family,
                "n_query_hits": len(hits),
                "n_refuse_eligible": len(refuse_hits),
                "sample_scar_id": (sample or {}).get("scar_id"),
                "sample_source_path": (sample or {}).get("source_path"),
                "sample_level": (sample or {}).get("level"),
            }
        )
    rows.sort(key=lambda r: r["family"])
    return rows


def recovered_implementation(atlas_rec: Mapping[str, Any], queue_rec: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        (
            atlas_rec["path_used"],
            "Architecture Atlas: CUDA and CUTLASS already represented; this module maps the holes",
        ),
        (
            queue_rec["path_used"],
            "Repatriation queue vocabulary this module emits into (status HYPOTHESIS, not READY)",
        ),
        (
            "tools/future/physical_primitives.py",
            "Seventeen atlas primitives; hypotheses map onto these names only",
        ),
        (
            "tools/future/negative_index.py",
            "Scar query / refuse_if_dead; catalog families must not match recorded scars",
        ),
        (
            "tools/accelerator/c2m.py",
            "C2M-T0 CUDA-to-Metal frontend; refuses __syncwarp, __shfl, __shared__, loops",
        ),
        (
            "tools/accelerator/c2m_idiom.py",
            "C2M-T2 idiom recognition (tiled GEMM, block reduce); not a C compiler",
        ),
        (
            "tools/accelerator/cuda_runtime.py",
            "C2M-T1 host runtime on Metal; refuses cudaGraphLaunch, streams, cudaMemcpyAsync",
        ),
        (
            "tools/accelerator/gemm.py",
            "Tiled Metal GEMM exercising shared memory and barriers (Codex-owned; read-only)",
        ),
        (
            "receipts/future/PHYSICAL_PRIMITIVES.json",
            "CUDA lowering seam UNAVAILABLE on this Apple host; local CUDA claims forbidden",
        ),
        (
            "receipts/future/NEGATIVE_SCIENCE_INDEX.json",
            "Sibling scar index; consulted, not forked",
        ),
    ]
    out: list[dict[str, Any]] = []
    for path, what in rows:
        _text, via = repo_text(path)
        out.append({"path": path, "resolved_via": via, "what": what})
    out.sort(key=lambda r: r["path"])
    return out


def gaps_closed(n_hypotheses: int, n_overlap_refusals: int) -> list[str]:
    return [
        (
            f"{n_hypotheses} CUDA-lowbit hypotheses that are not restatements of "
            "atlas CUDA/CUTLASS techniques (dequant site tier, M-dependent "
            "stationarity, occupancy-unit SIMD, N-stage pipeline, persist-XOR-"
            "replay, split reduction, selection-density sparse-vs-dense, LUT "
            "working-set tier, arithmetic-unit fragment constraint)"
        ),
        "admit() refuses numeric speedup/latency/hardware claims (NumericClaimError / HardwareClaimError)",
        "admit() refuses hypotheses matching refuse_if_dead scars (ScarRefusal)",
        "admit() refuses families that slug to mapped CUDA/CUTLASS techniques (OverlapRefusal)",
        f"restatement probes refused: {n_overlap_refusals} (persistent kernels, fused dequant+compute)",
        "each admitted hypothesis is also a repatriation-queue-shaped spec with status HYPOTHESIS, not READY",
        "evidence loaded from pinned snapshot when present, else live headless; path recorded per input",
        "no CUDA performance number, no claimed speedup, no benchmark; bench UNKNOWN, STATIC_ONLY",
    ]


def negative_findings(
    atlas_rec: Mapping[str, Any],
    queue_rec: Mapping[str, Any],
    avoid_rows: Iterable[Mapping[str, Any]],
) -> list[str]:
    findings = [
        "No NVIDIA CUDA device on this Apple host; CUDA is a source school, not an execution backend here.",
        "This lane produces neither DIAGNOSTIC_RELATIVE nor PROTECTED_ABSOLUTE; every hardware field is unmeasured.",
        "Tensor-core fragment shapes (mma.sync / wgmma K-major vs MN-major) are not encoded in the atlas; labelled MODEL_KNOWLEDGE.",
        "Metal async-copy / TMA analogue is not a named ISA in tools/accelerator (cuda_runtime refuses cudaMemcpyAsync); labelled as explicit multi-buffer staging.",
        "Split-K and N-stage pipeline depth are absent from CUDA/CUTLASS source_technique_coverage.",
        "FPGA is part of Accelerator / Physical Compiler / Fusion; this module does not build an FPGA backend.",
        "C2M is T0/T1/T2 subset recognition, not CUDA support; warp shuffle, shared memory, graphs, streams are refused by name.",
        "Pinned snapshot is preferred; live receipts/headless may exist in another worktree and is not treated as absence.",
    ]
    if atlas_rec.get("evidence_source") != "pinned_snapshot":
        findings.append(
            f"Atlas was not served from the pinned snapshot (used {atlas_rec.get('evidence_source')} at {atlas_rec.get('path_used')})"
        )
    if queue_rec.get("evidence_source") != "pinned_snapshot":
        findings.append(
            f"Repatriation queue was not served from the pinned snapshot (used {queue_rec.get('evidence_source')} at {queue_rec.get('path_used')})"
        )
    for row in avoid_rows:
        if row.get("n_refuse_eligible"):
            findings.append(
                f"Odyssey II scar live for family {row['family']}: "
                f"n_refuse_eligible={row['n_refuse_eligible']} "
                f"sample={row.get('sample_scar_id')}"
            )
        else:
            findings.append(
                f"Looked up AVOID family {row['family']}: no refuse-eligible hit "
                f"(n_query_hits={row.get('n_query_hits')}); still refused by AVOID list"
            )
    census_text, census_via = repo_text("receipts/headless/CUDA_CAPABILITY_CENSUS.json")
    findings.append(
        "CUDA_CAPABILITY_CENSUS.json resolution="
        + census_via
        + (
            "; cited by physical_primitives as Metal-via-MLX / transport ABSENT"
            if census_text
            else "; unresolved in this checkout (not treated as repo-absence)"
        )
    )
    return findings


def build() -> Path:
    atlas_rec = load_evidence(ATLAS_NAME)
    queue_rec = load_evidence(QUEUE_NAME)
    atlas = atlas_rec["doc"]
    queue = queue_rec["doc"]
    gap = analyze_gaps(atlas)
    hypotheses, restatement_refusals = emit_catalog(atlas)
    if any(r.get("disposition") == "GUARD_FAILURE" for r in restatement_refusals):
        raise HypothesisError("overlap guard failed to refuse a restatement probe")
    atlas_fp = str(atlas.get("fingerprint") or queue.get("atlas_fingerprint") or "")
    specs = [as_repatriation_spec(h, atlas_fp) for h in hypotheses]
    specs.sort(key=lambda s: s["experiment_id"])
    avoid_rows = query_avoid_families()
    emitted_families = {h["hypothesis_family"] for h in hypotheses}
    dead_collision = sorted(emitted_families & set(AVOID_FAMILIES))
    if dead_collision:
        raise ScarRefusal(f"catalog emitted AVOID families: {dead_collision}")

    evidence_source = {
        ATLAS_NAME: atlas_rec["evidence_source"],
        QUEUE_NAME: queue_rec["evidence_source"],
    }
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Convert publicly known CUDA/CUTLASS low-bit decode and MoE kernel "
            "design into Hawking hypotheses. Extract laws; do not copy vendors. "
            "No local CUDA performance claim on Apple hardware."
        ),
        "vocabulary": {
            "eras": list(ERAS),
            "odysseys": list(ODYSSEYS),
            "fpga_is": (
                "Accelerator / Physical Compiler / Fusion — never its own "
                "civilization; this module does not build an FPGA backend"
            ),
            "cuda_is": (
                "textbook now, physical oracle later; source school, not an "
                "execution backend on this Apple host"
            ),
            "evidence": {
                "DIAGNOSTIC_RELATIVE": "contaminated A/B; guides; never promotes; this lane does not produce it",
                "PROTECTED_ABSOLUTE": "protected GPU lease; decides; this lane does not produce it",
                "STATIC_ONLY": "the only evidence class this lane may emit",
            },
        },
        "claim_boundary": (
            "Static sidecar artifact. No hardware measurement. No CUDA device "
            "on this Apple host. These are physical-law hypotheses, not vendor "
            "copies and not benchmarks. Neither DIAGNOSTIC_RELATIVE nor "
            "PROTECTED_ABSOLUTE."
        ),
        "evidence_source": evidence_source,
        "evidence_resolution": {
            ATLAS_NAME: {
                "used": atlas_rec["path_used"],
                "source": atlas_rec["evidence_source"],
                "pinned": atlas_rec["pinned"],
                "live_headless": atlas_rec["live_headless"],
            },
            QUEUE_NAME: {
                "used": queue_rec["path_used"],
                "source": queue_rec["evidence_source"],
                "pinned": queue_rec["pinned"],
                "live_headless": queue_rec["live_headless"],
            },
        },
        "atlas_fingerprint": atlas_fp,
        "atlas_schema": atlas.get("schema"),
        "repatriation_queue_schema": queue.get("schema"),
        "experiment_spec_schema": EXPERIMENT_SPEC_SCHEMA,
        "gap_analysis": gap,
        "hypotheses": hypotheses,
        "n_hypotheses": len(hypotheses),
        "repatriation_specs": specs,
        "n_repatriation_specs": len(specs),
        "restatement_refusals": restatement_refusals,
        "avoid_families": avoid_rows,
        "required_fields": list(REQUIRED_HYPOTHESIS_FIELDS),
        "cost_classes": list(COST_CLASSES),
        "knowledge_sources": list(KNOWLEDGE_SOURCES),
        "atlas_primitives": list(ATLAS_PRIMITIVES),
        "primitive_contracts_recovered": {
            name: {
                "invariant": CONTRACTS[name]["invariant"],
                "behavior_ids": list(CONTRACTS[name]["behavior_ids"]),
            }
            for name in ATLAS_PRIMITIVES
        },
        "recovered_implementation": recovered_implementation(atlas_rec, queue_rec),
        "gaps_closed": gaps_closed(len(hypotheses), len(restatement_refusals)),
        "negative_findings": negative_findings(atlas_rec, queue_rec, avoid_rows),
        "integration": {
            "admit": (
                "admit(raw, *, atlas=None, slugs=None) -> dict  "
                "# raises MissingFieldError | UnknownPrimitiveError | "
                "NumericClaimError | ScarRefusal | OverlapRefusal"
            ),
            "as_repatriation_spec": "as_repatriation_spec(h, atlas_fingerprint) -> dict",
            "analyze_gaps": "analyze_gaps(atlas) -> dict",
            "load_evidence": "load_evidence(name) -> {doc, evidence_source, path_used, ...}",
            "build": f"build() -> Path  # receipts/future/{RECEIPT}",
        },
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args(argv)
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
