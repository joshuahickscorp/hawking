"""FLASH_ORGAN_PIVOT — leave the exhausted gate_up surface; rank remaining organs by expected information gain per cost.

The rival-codec screen spent 1024 real teacher rows on
layer_4.routed_experts.gate_up_proj and killed every tested family there
(0 contract passes, 0 families beating per-expert Q4). That is a scoped
scar: one organ, one surface, one split. It is not a license to sweep
more ranks of those families on that surface, and it is not a license to
treat other expert tensors, other layers, or other Flash organs as dense.

This module ranks where Flash still has untested, possibly-low independent
information. Source bytes come from the organ census (or a cited overlay).
Mechanisms come from the landed schools. A nearby restatement of a killed
family on the exhausted surface is REFUSED with the family and the receipt
named. An organ with no census bytes is UNRANKABLE, not guessed.

Refuses: hardware measurement, GPU lease, EBPW/capability claims, inventing
census bytes, rounding a partial ranking into a pass, generalising the scar.

Cannot establish: mutual information of any organ, a physical EBPW number,
capability survival, that down_proj is compressible, or that n-gram generation
works. Expected IG/cost is a structural prior, not a measurement.
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from typing import Any, Mapping, Sequence

from tools.future._common import RECEIPTS, write_receipt
from tools.future import expert_bank_school as ebs
from tools.future import flash_schools as fs
from tools.future.meta_funnel import load_receipt
from tools.future import ngram_school as ngs
from tools.future import rival_codec_screen as rcs
from tools.future import router_science as rs


RECEIPT = "FLASH_ORGAN_PIVOT.json"
SCHEMA = "hawking.future.flash_organ_pivot.v1"
VERSION = 1
RECORDED_BY = "tools/future/flash_organ_pivot.py"

RIVAL_REL = "receipts/future/RIVAL_CODEC_SCREEN.json"
REPLAN_REL = "receipts/future/FLASH_META_REPLAN.json"
SCHOOLS_REL = "receipts/future/FLASH_ORGAN_SCHOOLS.json"
CENSUS_RELS = (
    "receipts/future/evidence/FLASH_ORGAN_CENSUS.json",
    "receipts/headless/FLASH_ORGAN_CENSUS.json",
)
ROUTER_MAP_RELS = (
    "receipts/future/evidence/FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json",
    "receipts/headless/FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json",
)
MIB = 1024 * 1024

EXHAUSTED_ORGAN = rcs.REFERENCE_ORGAN  # layer_4.routed_experts.gate_up_proj
EXHAUSTED_TENSOR = rcs.TENSOR_GATE_UP
EXHAUSTED_SURFACE = "model.language_model.layers.4.mlp_input"
KILLED_BY = RIVAL_REL

# Screen family_ids recovered from RIVAL_CODEC_SCREEN. Used only when the
# receipt is unreachable so restatement refusal still fires. Receipt wins.
BUILTIN_KILLED_FAMILY_IDS: tuple[str, ...] = (
    "per_expert_q4_control",
    "shared_input_latent_plus_expert_local_output_readout",
    "per_expert_q3_control",
    "per_expert_q2_control",
    "common_left_subspace_plus_expert_local_core",
    "common_right_subspace_plus_expert_local_core",
    "clustered_subspaces_route_conditioned",
    "dictionary_plus_per_expert_sparse_residual",
    "expert_local_small_core_plus_shared_decoder",
    "sparse_residual_on_cheap_backbone",
)

# Long phrases only. Short tokens like "shared" or "expert" must not match.
KILLED_ALIASES: dict[str, tuple[str, ...]] = {
    "shared_input_latent_plus_expert_local_output_readout": (
        "shared_input_latent_plus_expert_local_output_readout",
        "shared input latent plus expert local output readout",
        "shared input latent",
        "expert-local readout",
        "expert local output readout",
        "latent+readout",
        "latent plus readout",
        "expert-local latent code + shared tile decoder",
        "STORE-SHARED-INPUT-TRANSFORM",
        "STORE-SHARED-OUTPUT-LATENT",
        "shared_input_transforms",
        "shared_output_latent_spaces",
        "expert_bank_shared_input_transforms",
        "expert_bank_shared_output_latent_spaces",
    ),
    "common_left_subspace_plus_expert_local_core": (
        "common_left_subspace_plus_expert_local_core",
        "common left subspace",
        "common_left_subspaces",
        "STORE-COMMON-LEFT-SUBSPACE",
        "expert_bank_common_left_subspaces",
        "y_e = U @ (C_e @ x)",
    ),
    "common_right_subspace_plus_expert_local_core": (
        "common_right_subspace_plus_expert_local_core",
        "common right subspace",
        "common_right_subspaces",
        "STORE-COMMON-RIGHT-SUBSPACE",
        "expert_bank_common_right_subspaces",
        "y_e = C_e @ (V @ x)",
    ),
    "clustered_subspaces_route_conditioned": (
        "clustered_subspaces_route_conditioned",
        "clustered subspaces",
        "clustered_subspaces",
        "STORE-CLUSTERED-SUBSPACES",
        "expert_bank_clustered_subspaces",
        "V_{cluster(e)}",
        "STORE-ROUTE-CONDITIONED-ARCHETYPES",
        "route_conditioned_archetypes",
        "expert_bank_route_conditioned_archetypes",
    ),
    "dictionary_plus_per_expert_sparse_residual": (
        "dictionary_plus_per_expert_sparse_residual",
        "dictionary_families",
        "STORE-DICTIONARY-FAMILIES",
        "expert_bank_dictionary_families",
        "dictionary plus per-expert sparse residual",
        "shared codebook across routed experts",
        "dictionary of experts",
    ),
    "expert_local_small_core_plus_shared_decoder": (
        "expert_local_small_core_plus_shared_decoder",
        "expert_specific_small_cores",
        "STORE-EXPERT-SMALL-CORE",
        "expert_bank_expert_specific_small_cores",
        "small-core sandwich",
        "U @ C_e @ (V @ x)",
        "STORE-TENSOR-DECOMPOSITION",
        "tensor_decomposition",
        "expert_bank_tensor_decomposition",
    ),
    "sparse_residual_on_cheap_backbone": (
        "sparse_residual_on_cheap_backbone",
        "residual on architecture shared expert",
        "architecture shared expert residual",
        "W_shared @ x + U_e",
        "STORE-CONDITIONAL-RESIDUAL",
        "conditional_residuals",
        "expert_bank_conditional_residuals",
    ),
    "per_expert_q4_control": (
        "per_expert_q4_control",
        "more ranks of q4",
        "q4 rank sweep",
    ),
    "per_expert_q3_control": (
        "per_expert_q3_control",
        "more ranks of q3",
        "q3 rank sweep",
        "per_expert_q3",
    ),
    "per_expert_q2_control": (
        "per_expert_q2_control",
        "more ranks of q2",
        "q2 rank sweep",
        "per_expert_q2",
    ),
}

# Unlabelled nearby restatements of the low-rank sharing families already
# killed on gate_up. Matched only on the exhausted surface.
NEARBY_PHRASES: tuple[tuple[str, str], ...] = (
    ("low-rank sharing", "shared_input_latent_plus_expert_local_output_readout"),
    ("more ranks of", "shared_input_latent_plus_expert_local_output_readout"),
    ("rank sweep", "shared_input_latent_plus_expert_local_output_readout"),
    ("another rank of", "shared_input_latent_plus_expert_local_output_readout"),
    ("shared subspace", "common_left_subspace_plus_expert_local_core"),
    ("product codebook on routed", "dictionary_plus_per_expert_sparse_residual"),
    ("hierarchical dictionary on routed", "dictionary_plus_per_expert_sparse_residual"),
)

# Structural priors from flash_schools.elimination_answers + the scoped scar.
# Higher weight = more expected collapse of independent information, not a
# measured mutual-information number. Function organs have weight but still
# require census bytes to rank.
INDEPENDENCE: dict[str, tuple[str, int, str]] = {
    "NGRAM": (
        "GENERATABLE_BULK",
        8,
        "table is ~28% of specimen; ngram_school exists to kill independent storage of 320M rows",
    ),
    "POSITIONAL_STRUCTURE": (
        "GENERATABLE_BULK",
        8,
        "RoPE is a closed-form program; textbook ZERO_STORAGE",
    ),
    "ROUTED_EXPERTS": (
        "MIXED_DENSE_AND_UNTESTED",
        5,
        "gate_up empirically dense on the stated scope; W2/down and other layers untested; family is ~68% of specimen",
    ),
    "EMBEDDING": (
        "GENERATABLE_TAIL",
        5,
        "lookup organ; rare-row generation is live; shares embedding_lm_head parent budget with LM_HEAD, unsplit",
    ),
    "SHARED_EXPERTS": (
        "UNTESTED_WEIGHT_ORGAN",
        4,
        "protected island until generator parity; residual-as-backbone on gate_up is dead, compressing the shared expert itself is not that scar",
    ),
    "LM_HEAD": (
        "PROTECTED_TERMINAL",
        3,
        "terminal-logit island; crush refused; tying with embed is live; shares parent budget, unsplit",
    ),
    "HC_HYPERCONNECTION": (
        "PROTECTED_COEFFICIENT_ISLAND",
        3,
        "small exact mix coefficients; fusion kills standalone kernels, not the information",
    ),
    "DELTANET_RECURRENT_STATE": (
        "PROTECTED_STATE",
        3,
        "state is the information (CLOSED); linear-attention weight compression is open",
    ),
    "FULL_ATTENTION": (
        "PROTECTED_ISLAND",
        3,
        "KV-sensitive; QKV factor is untested and is not a routed-expert restatement",
    ),
    "ROUTER": (
        "PROTECTED_CONTROL",
        2,
        "control plane; independent storage required; overlay bytes only if census family is empty",
    ),
    "NORMALIZATION": (
        "PROTECTED_EXACT",
        1,
        "tiny exact RMSNorm; do not generate",
    ),
    "KV_STATE": (
        "FUNCTION_NO_CENSUS",
        0,
        "runtime state, not a census weight family",
    ),
    "DECODING": (
        "FUNCTION_NO_CENSUS",
        0,
        "function organ, not a census weight family",
    ),
    "MTP_SPECULATION": (
        "FUNCTION_NO_CENSUS",
        0,
        "function organ; expert tensors live inside routed_experts",
    ),
}

COST_WEIGHT: dict[str, int] = {
    "CHEAP_ANALYTICAL": 1,
    "CHEAP_SYNTHETIC": 2,
    "NEW_SURFACE_TEACHER": 8,
}

# Next surface to spend cost on. ROUTED leaves gate_up.
SCHOOL_PIVOT: dict[str, dict[str, str]] = {
    "NGRAM": {
        "organ": "ngram_embedding",
        "surface": "n-gram table lookup (frequency-tiered generator + hot islands)",
        "mechanism": "product_codebooks / hierarchical_codebooks / generated_lookup / literal_exception_islands (ngram_school)",
        "cheapest_falsifier": "Five-axis Pareto in ngram_school, then a tiny held-out retrieval; kill if dominated on every axis or retrieval fails.",
        "why_not_restatement": "Different organ. Dictionary-on-gate_up is dead; a product codebook of the 320M-row table is not that family.",
        "cost_class": "CHEAP_ANALYTICAL",
        "extends": "tools/future/ngram_school.py",
    },
    "ROUTED_EXPERTS": {
        "organ": "routed_experts.down_proj",
        "surface": "routed-output / W2 down_proj (NOT layer_4.routed_experts.gate_up_proj)",
        "mechanism": "cross-layer expert prediction; capability islands; one-sided factors on down_proj only",
        "cheapest_falsifier": "Same coherence contract on a captured routed-output / down_proj surface of one layer. Kill if it fails the screen's own inequalities. Do not reopen by lowering the gate.",
        "why_not_restatement": "Scoped scar is gate_up_proj on layer 4. down_proj is a different tensor; the receipt explicitly refuses to generalise.",
        "cost_class": "NEW_SURFACE_TEACHER",
        "extends": "tools/future/expert_bank_school.py#STORE-CROSS-LAYER-EXPERT-PREDICTION",
    },
    "EMBEDDING": {
        "organ": "embedding_lm_head",
        "surface": "input embedding rows (lookup, not GEMV)",
        "mechanism": "hot-row literal + rare-row generator (flash_schools EMBEDDING:RARE-GENERATOR)",
        "cheapest_falsifier": "Kill if rare-token held-out retrieval fails.",
        "why_not_restatement": "Lookup organ, not routed gate_up. Vocab generation is not a shared expert basis.",
        "cost_class": "CHEAP_SYNTHETIC",
        "extends": "tools/future/flash_schools.py#EMBEDDING:RARE-GENERATOR",
        "parent_budget_shared_with": "LM_HEAD",
    },
    "LM_HEAD": {
        "organ": "embedding_lm_head",
        "surface": "terminal logits / argmax neighborhood",
        "mechanism": "tie to embed with adapter, or PREMIUM island on the argmax neighborhood",
        "cheapest_falsifier": "Kill if terminal logits disagree with untied source, or argmax flips (decode_civilization exact-target).",
        "why_not_restatement": "Vocab tying, not routed-expert sharing. Terminal island is a protected control surface.",
        "cost_class": "CHEAP_SYNTHETIC",
        "extends": "tools/future/flash_schools.py#LM_HEAD:TIE-EMBED",
        "parent_budget_shared_with": "EMBEDDING",
    },
    "POSITIONAL_STRUCTURE": {
        "organ": "other",
        "surface": "RoPE / position tables",
        "mechanism": "closed-form rotary program (flash_schools POSITIONAL_STRUCTURE:ROPE-FORMULA)",
        "cheapest_falsifier": "Kill if generated rotary disagrees with source tables on a position grid.",
        "why_not_restatement": "Formula, not a learned codebook and not a routed-expert factor.",
        "cost_class": "CHEAP_ANALYTICAL",
        "extends": "tools/future/flash_schools.py#POSITIONAL_STRUCTURE:ROPE-FORMULA",
    },
    "HC_HYPERCONNECTION": {
        "organ": "mlp_hyperconnection",
        "surface": "HC mix coefficients / fused read-write",
        "mechanism": "generated coefficients stay exact; fuse standalone HC kernels (ZERO_EXECUTION of dispatches)",
        "cheapest_falsifier": "Kill if fused order differs from source HC read precondition, or if mix coefficients themselves are dropped.",
        "why_not_restatement": "Exact coefficient island, not a shared expert subspace. Fusion is physical, not low-rank sharing.",
        "cost_class": "CHEAP_SYNTHETIC",
        "extends": "tools/future/flash_schools.py#HC_HYPERCONNECTION:FUSED-READ-WRITE",
    },
    "SHARED_EXPERTS": {
        "organ": "shared_expert",
        "surface": "shared-expert SwiGLU (the organ itself, not as a backbone for routed gate_up)",
        "mechanism": "factorized fused SwiGLU; fuse epilogue with routed accumulation (moe_physical_school)",
        "cheapest_falsifier": "One-layer shared-expert function screen vs source BF16. Kill if the factor rank that saves bytes fails SwiGLU.",
        "why_not_restatement": "sparse_residual_on_cheap_backbone used the shared expert as a backbone for routed gate_up. Compressing the shared expert itself is a different organ.",
        "cost_class": "CHEAP_SYNTHETIC",
        "extends": "tools/future/moe_physical_school.py",
    },
    "DELTANET_RECURRENT_STATE": {
        "organ": "linear_attention_hyperconnection",
        "surface": "DeltaNet transition + live recurrent state",
        "mechanism": "keep the transition; search a smaller state encoding; fuse in-proj/conv/update/out-proj",
        "cheapest_falsifier": "Kill if recurrent_state_semantics leave source-approved (coherence_contract). State traces, not a weight MSE.",
        "why_not_restatement": "RecurrentStateProgram is the native family. A static low-rank of gate_up is a different object.",
        "cost_class": "NEW_SURFACE_TEACHER",
        "extends": "tools/future/flash_schools.py#DELTANET_RECURRENT_STATE:TRANSITION",
    },
    "FULL_ATTENTION": {
        "organ": "full_attention",
        "surface": "QKV projections + SDPA window",
        "mechanism": "fused QKV/RoPE/SDPA; optional one-sided Q/K/V factor",
        "cheapest_falsifier": "Kill if fused path disagrees with source attention on a short teacher window, or QKV factor fails at the rank that saves bytes.",
        "why_not_restatement": "Attention organ, not routed gate_up. QKV factor is not a shared expert basis.",
        "cost_class": "NEW_SURFACE_TEACHER",
        "extends": "tools/future/flash_schools.py#FULL_ATTENTION:FUSED-QKV-SDPA",
    },
    "ROUTER": {
        "organ": "router",
        "surface": "router logits / top-k membership and order (L3-L4 seam already mapped)",
        "mechanism": "CONTROL_FLOW_PREMIUM gate + margin-gated residual (router_science)",
        "cheapest_falsifier": "Kill if compact+residual still changes top-K membership on the existing L3-L4 map. Do not rerun the seam as if it were new.",
        "why_not_restatement": "Control plane, not payload. Crushing the gate to the bulk Q4 of routed experts is the uniform-subbit scar, not this ranking.",
        "cost_class": "CHEAP_ANALYTICAL",
        "extends": "tools/future/router_science.py",
    },
    "NORMALIZATION": {
        "organ": "norm",
        "surface": "RMSNorm weights (exact island)",
        "mechanism": "fuse into the consumer; keep exact coefficients",
        "cheapest_falsifier": "Kill if fused norm is not source-exact.",
        "why_not_restatement": "Tiny exact island. A residual on a wrong norm is still a wrong norm; this is not a routed dictionary.",
        "cost_class": "CHEAP_SYNTHETIC",
        "extends": "tools/future/flash_schools.py#NORMALIZATION:FUSED-INTO-CONSUMER",
    },
    "KV_STATE": {
        "organ": "kv",
        "surface": "runtime KV cache (not a census weight family)",
        "mechanism": "quantized cache / recompute / sinking-token dictionary (decode_civilization)",
        "cheapest_falsifier": "Kill on recall, not on bytes. Absent long-context probe = not ranked.",
        "why_not_restatement": "Runtime state, not gate_up weights.",
        "cost_class": "NEW_SURFACE_TEACHER",
        "extends": "tools/future/decode_civilization.py#kv_compression",
    },
    "DECODING": {
        "organ": "whole_model",
        "surface": "accept/reject / exact-target predicate",
        "mechanism": "decoding program; WHAT is checked may not move (decode_civilization)",
        "cheapest_falsifier": "VerificationCorrectnessError if WHAT changes.",
        "why_not_restatement": "Function organ, not a weight codec.",
        "cost_class": "CHEAP_ANALYTICAL",
        "extends": "tools/future/decode_civilization.py",
    },
    "MTP_SPECULATION": {
        "organ": "whole_model",
        "surface": "draft / verify / rollback function (MTP expert bytes stay in routed_experts)",
        "mechanism": "ZeroProgram (no speculation) is the control every draft must beat on accepted-token cost",
        "cheapest_falsifier": "Kill if accepted complete-token cost exceeds the no-speculation control. Do not cite draft TPS.",
        "why_not_restatement": "Function school. MTP expert tensors are indexed inside routed_experts; this school does not re-quantize them.",
        "cost_class": "NEW_SURFACE_TEACHER",
        "extends": "tools/future/flash_schools.py#MTP_SPECULATION:NO-SPECULATION",
    },
}

FUNCTION_ORGANS: frozenset[str] = frozenset(
    sid for sid, fams in fs.SCHOOL_CENSUS_FAMILIES.items() if not fams
)

CLAIM_BOUNDARY = (
    "Static sidecar artifact. Expected information-gain ranking off an exhausted "
    "teacher surface. No hardware measurement. No physical EBPW. No capability "
    "result. A rank is a structural prior over census bytes and school mechanisms, "
    "not evidence about any representation."
)


class RestatementRefused(ValueError):
    """A candidate restates a family already killed on the exhausted surface."""

    def __init__(self, candidate_id: str, killed_family: str, killed_by: str, reason: str):
        self.candidate_id = candidate_id
        self.killed_family = killed_family
        self.killed_by = killed_by
        self.reason = reason
        super().__init__(
            f"REFUSED {candidate_id}: restatement of killed family "
            f"{killed_family} ({killed_by}): {reason}"
        )


class UnrankableOrgan(ValueError):
    """An organ with no census bytes is not given a guessed rank."""

    def __init__(self, school: str, reason: str):
        self.school = school
        self.reason = reason
        super().__init__(f"UNRANKABLE {school}: {reason}")


class ScarUnavailable(ValueError):
    """The exhausted-surface receipt is absent; the scar is not invented as a pass."""


# ---------------------------------------------------------------------------
# Loaders. Disk then HEAD. Absence is not proof the object is gone.
# ---------------------------------------------------------------------------


def load_named(*rels: str) -> tuple[dict[str, Any] | None, str | None]:
    for rel in rels:
        doc = load_receipt(rel)
        if isinstance(doc, dict):
            return doc, rel
    return None, None


def _int_bytes(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer() and value > 0:
        return int(value)
    return None


def load_inventory() -> dict[str, Any]:
    """Census bytes from the organ census, else the sealed schools inventory.

    Unreachability is not absence. A missing file here is not a guess of zero.
    Zero is returned only when a reachable census/inventory lists the family
    at 0 or omits it after the census itself was reachable.
    """
    census, census_rel = load_named(*CENSUS_RELS)
    router_map, router_rel = load_named(*ROUTER_MAP_RELS)
    schools_doc, schools_rel = load_named(SCHOOLS_REL)

    families: list[dict[str, Any]] = []
    by_family: dict[str, dict[str, Any]] = {}
    specimen_bytes = None
    census_source = "unavailable"
    largest_tensors: list[dict[str, Any]] = []

    if census:
        census_source = f"census:{census_rel}"
        specimen_bytes = census.get("source_parameter_bytes_indexed")
        largest_tensors = [t for t in (census.get("largest_tensors") or []) if isinstance(t, dict)]
        for row in census.get("family_summary") or []:
            if not isinstance(row, dict) or not row.get("family"):
                continue
            item = {
                "family": row["family"],
                "tensor_count": row.get("tensor_count"),
                "bytes": int(row["bytes"]) if isinstance(row.get("bytes"), (int, float)) else 0,
                "fraction": row.get("fraction"),
            }
            families.append(item)
            by_family[item["family"]] = item
        families.sort(key=lambda r: r["family"])
    elif schools_doc and isinstance((schools_doc.get("inventory") or {}).get("families"), list):
        inv = schools_doc["inventory"]
        census_source = f"sidecar_inventory:{schools_rel}"
        specimen_bytes = inv.get("specimen_bytes")
        for row in inv.get("families") or []:
            if not isinstance(row, dict) or not row.get("family"):
                continue
            item = {
                "family": row["family"],
                "tensor_count": row.get("tensor_count"),
                "bytes": int(row["bytes"]) if isinstance(row.get("bytes"), (int, float)) else 0,
                "fraction": row.get("fraction"),
            }
            families.append(item)
            by_family[item["family"]] = item
        families.sort(key=lambda r: r["family"])

    router_bytes = None
    router_source = "unavailable"
    if router_map:
        tb = _int_bytes((router_map.get("router_source") or {}).get("tensor_bytes"))
        if tb is not None:
            router_bytes = tb
            router_source = f"router_map:{router_rel}"
    if router_bytes is None and schools_doc:
        tb = _int_bytes((schools_doc.get("inventory") or {}).get("router_tensor_bytes"))
        if tb is not None:
            router_bytes = tb
            router_source = f"sidecar_inventory:{schools_rel}"
    if router_bytes is None:
        tb = _int_bytes((rs.RECOVERED_MAP.get("router_source") or {}).get("tensor_bytes"))
        if tb is not None:
            router_bytes = tb
            router_source = "tools/future/router_science.py#RECOVERED_MAP"

    return {
        "census_source": census_source,
        "census_rel": census_rel,
        "router_source": router_source,
        "router_rel": router_rel,
        "schools_rel": schools_rel,
        "specimen_bytes": specimen_bytes,
        "families": families,
        "by_family": by_family,
        "router_tensor_bytes": router_bytes,
        "largest_tensors": largest_tensors,
        "n_census_families": len(families),
        "reachable": census_source != "unavailable",
    }


def census_bytes_for(school: str, inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Bytes this school is allowed to rank on. Zero/absent is UNRANKABLE."""
    if school not in fs.SCHOOL_CATALOG:
        raise fs.UnknownSchoolError(school)
    declared = list(fs.SCHOOL_CENSUS_FAMILIES.get(school) or ())
    by_family = inventory.get("by_family") or {}
    reachable = bool(inventory.get("reachable"))
    census_total = 0
    present = []
    for fam in declared:
        row = by_family.get(fam) or {}
        b = row.get("bytes")
        if isinstance(b, int):
            present.append(fam)
            census_total += b

    overlay = inventory.get("router_tensor_bytes") if school == "ROUTER" else None
    overlay_ok = isinstance(overlay, int) and overlay > 0

    if census_total > 0:
        return {
            "school": school,
            "bytes": census_total,
            "provenance": "census_family_summary",
            "declared_families": declared,
            "present_families": present,
            "rankable": True,
            "reason": f"census family bytes={census_total} families={present}",
        }
    if overlay_ok:
        return {
            "school": school,
            "bytes": int(overlay),
            "provenance": str(inventory.get("router_source") or "overlay"),
            "declared_families": declared,
            "present_families": present,
            "rankable": True,
            "reason": (
                f"census family empty; cited overlay bytes={overlay} "
                f"from {inventory.get('router_source')}"
            ),
        }
    if not reachable and declared:
        return {
            "school": school,
            "bytes": None,
            "provenance": "census_unreachable",
            "declared_families": declared,
            "present_families": [],
            "rankable": False,
            "reason": (
                "FLASH_ORGAN_CENSUS unreachable in this checkout; unreachability "
                "is not a zero. Organ is UNRANKABLE until bytes are cited, not guessed."
            ),
        }
    return {
        "school": school,
        "bytes": None,
        "provenance": "no_census_bytes",
        "declared_families": declared,
        "present_families": present,
        "rankable": False,
        "reason": (
            f"{school} has no census family bytes"
            + (" (function organ; empty declaration is not a missing weight family)" if not declared else "")
            + "; refusing a guessed rank"
        ),
    }


# ---------------------------------------------------------------------------
# Scoped scar. Read from the rival screen; never generalised.
# ---------------------------------------------------------------------------


def _best_heldout(fam: Mapping[str, Any]) -> float | None:
    errors: list[float] = []
    for row in fam.get("rows") or []:
        if not isinstance(row, dict) or not row.get("scored"):
            continue
        err = row.get("heldout_relative_fro_error")
        if isinstance(err, (int, float)) and not isinstance(err, bool):
            errors.append(float(err))
    return min(errors) if errors else None


def killed_families_from_screen(screen: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(screen, Mapping):
        return [
            {
                "family": name,
                "any_pass": False,
                "n_beats_q4": 0,
                "n_passed_contract": 0,
                "best_heldout_relative_fro_error": None,
                "killed_by": KILLED_BY,
                "authority": "recovered_builtin_not_rescored",
                "algebra": None,
                "is_comparator": name == "per_expert_q4_control",
            }
            for name in BUILTIN_KILLED_FAMILY_IDS
        ]
    out: list[dict[str, Any]] = []
    for fam in screen.get("families") or []:
        if not isinstance(fam, dict) or not fam.get("family"):
            continue
        name = str(fam["family"])
        rows = [r for r in (fam.get("rows") or []) if isinstance(r, dict)]
        algebra = next((r.get("algebra") for r in rows if r.get("algebra")), None)
        out.append(
            {
                "family": name,
                "any_pass": bool(fam.get("any_pass")),
                "n_beats_q4": fam.get("n_beats_q4"),
                "n_passed_contract": fam.get("n_passed_contract"),
                "best_heldout_relative_fro_error": _best_heldout(fam),
                "killed_by": KILLED_BY,
                "authority": "read_from_receipt",
                "algebra": algebra,
                "is_comparator": name == "per_expert_q4_control",
                "wins_the_screen": bool(fam.get("wins_the_screen")),
            }
        )
    return out


def scoped_scar(screen: Mapping[str, Any] | None, *, screen_rel: str | None) -> dict[str, Any]:
    """The scar is scoped. Recording it does not generalise it."""
    if not isinstance(screen, Mapping):
        return {
            "status": "REFUSED_UNAVAILABLE",
            "reason": (
                f"{KILLED_BY} is not reachable; refusing to invent the scar "
                "as a pass or to generalise a density finding that was not re-read"
            ),
            "organ": EXHAUSTED_ORGAN,
            "tensor": EXHAUSTED_TENSOR,
            "surface": EXHAUSTED_SURFACE,
            "split": None,
            "refuses_to_generalise": True,
            "generalises_to_other_expert_tensors": False,
            "generalises_to_other_layers": False,
            "generalises_to_other_flash_organs": False,
            "generalises_to_other_mechanisms": False,
            "any_family_passed_contract": None,
            "any_family_beats_q4": None,
            "q4_is_best_tested_local_comparator": None,
            "screen_rel": screen_rel,
            "killed_by": KILLED_BY,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }

    specimen = screen.get("specimen") or {}
    split = {
        "fit_rows": screen.get("n_fit"),
        "heldout_rows": screen.get("n_heldout"),
        "teacher_rows": (screen.get("state") or {}).get("rows"),
    }
    any_pass = bool(screen.get("any_family_passed_contract"))
    families = killed_families_from_screen(screen)
    any_beats = any(int(f.get("n_beats_q4") or 0) > 0 for f in families)
    q4 = next((f for f in families if f["family"] == "per_expert_q4_control"), None)
    residual = next((f for f in families if f["family"] == "sparse_residual_on_cheap_backbone"), None)
    q4_err = screen.get("q4_error")
    residual_err = residual.get("best_heldout_relative_fro_error") if residual else None
    worse_than_zero = (
        isinstance(residual_err, (int, float)) and float(residual_err) > 1.0
    )
    return {
        "status": "SCOPED_SCAR",
        "reason": (
            "every tested family failed the screen's own coherence contract on "
            f"{EXHAUSTED_ORGAN}; 0 passes, 0 beating Q4; this does not generalise"
        ),
        "organ": EXHAUSTED_ORGAN,
        "tensor": specimen.get("tensor") or EXHAUSTED_TENSOR,
        "surface": EXHAUSTED_SURFACE,
        "split": split,
        "weight_shape": specimen.get("weight_shape"),
        "n_experts": specimen.get("n_experts"),
        "refuses_to_generalise": True,
        "generalises_to_other_expert_tensors": False,
        "generalises_to_other_layers": False,
        "generalises_to_other_flash_organs": False,
        "generalises_to_other_mechanisms": False,
        "any_family_passed_contract": any_pass,
        "any_family_beats_q4": any_beats,
        "q4_is_best_tested_local_comparator": True,
        "q4_heldout_relative_fro_error": q4_err if q4_err is not None else (q4 or {}).get("best_heldout_relative_fro_error"),
        "residual_on_shared_expert_worse_than_predicting_zero": worse_than_zero,
        "residual_heldout_relative_fro_error": residual_err,
        "harness_ok": bool((screen.get("harness") or {}).get("ok")),
        "promotion_allowed": bool(screen.get("promotion_allowed")),
        "screen_rel": screen_rel or KILLED_BY,
        "killed_by": KILLED_BY,
        "killed_family_ids": [f["family"] for f in families],
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "not_a_capability_result": True,
        "not_physical_ebpw": True,
    }


def on_exhausted_surface(candidate: Mapping[str, Any], scar: Mapping[str, Any]) -> bool:
    """True only for the stated organ/surface/split, not for down_proj or other layers."""
    organ = str(candidate.get("organ") or "")
    surface = str(candidate.get("surface") or "")
    tensor = str(candidate.get("tensor") or "")
    ident = str(candidate.get("id") or "")
    blob = " ".join((organ, surface, tensor, ident)).lower()
    if "down_proj" in blob or "w2" in blob.replace("w2/", "w2 "):
        return False
    scar_organ = str(scar.get("organ") or EXHAUSTED_ORGAN)
    scar_tensor = str(scar.get("tensor") or EXHAUSTED_TENSOR)
    if organ == scar_organ or surface == scar_organ or tensor == scar_tensor:
        return True
    layer4 = (
        "layer_4" in blob
        or "layers.4" in blob
        or "layer-4" in blob
        or "l4." in blob
        or " l4 " in f" {blob} "
    )
    return ("gate_up" in blob) and layer4


def _blob(candidate: Mapping[str, Any]) -> str:
    parts = [
        str(candidate.get("id") or ""),
        str(candidate.get("family") or ""),
        str(candidate.get("hypothesis_family") or ""),
        str(candidate.get("kind") or ""),
        str(candidate.get("mechanism") or ""),
        str(candidate.get("program_family") or ""),
        str(candidate.get("algebra") or ""),
    ]
    return " ".join(parts).lower()


def match_killed_family(candidate: Mapping[str, Any], killed: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Name the killed family this candidate restates, or None.

    Matching is phrase/id identity, not a guess that every codec is dead.
    """
    blob = _blob(candidate)
    family = str(candidate.get("family") or "").lower()
    ident = str(candidate.get("id") or "").lower()
    hypo = str(candidate.get("hypothesis_family") or "").lower()
    kind = str(candidate.get("kind") or "").lower()
    killed_names = {str(k.get("family") or "") for k in killed}

    for killed_family, aliases in KILLED_ALIASES.items():
        if killed_family not in killed_names and killed_names:
            # Screen present but this name was not on it: do not invent a kill.
            # Builtin path has all names.
            continue
        for alias in aliases:
            a = alias.lower()
            if family == a or ident == a or hypo == a or kind == a:
                return next((dict(k) for k in killed if k.get("family") == killed_family), {"family": killed_family})
            if len(a) >= 12 and a in blob:
                return next((dict(k) for k in killed if k.get("family") == killed_family), {"family": killed_family})

    for phrase, killed_family in NEARBY_PHRASES:
        if phrase in blob and (killed_family in killed_names or not killed_names):
            return next((dict(k) for k in killed if k.get("family") == killed_family), {"family": killed_family})
    return None


def restatement_verdict(
    candidate: Mapping[str, Any],
    scar: Mapping[str, Any],
    killed: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Refusal record if this is a nearby restatement on the exhausted surface."""
    exhausted = on_exhausted_surface(candidate, scar)
    family = str(candidate.get("family") or "")
    unspecified = not (
        candidate.get("organ") or candidate.get("surface") or candidate.get("tensor")
    )
    matched = match_killed_family(candidate, killed)
    default_exhausted = (
        unspecified
        and candidate.get("school") == "ROUTED_EXPERTS"
        and matched is not None
    )
    if not exhausted and not default_exhausted:
        return None
    if matched is None:
        return None
    killed_family = str(matched.get("family") or family)
    killed_by = str(matched.get("killed_by") or scar.get("killed_by") or KILLED_BY)
    reason = (
        f"nearby restatement of {killed_family} on {scar.get('organ') or EXHAUSTED_ORGAN} "
        f"(split={scar.get('split')}); more ranks of a killed family on the exhausted "
        "surface are refused. The scar does not generalise, and this candidate did not leave the surface."
    )
    return {
        "status": "REFUSED_RESTATEMENT",
        "killed_family": killed_family,
        "killed_by": killed_by,
        "reason": reason,
        "organ": scar.get("organ") or EXHAUSTED_ORGAN,
        "surface": scar.get("surface") or EXHAUSTED_SURFACE,
        "split": scar.get("split"),
        "refuses_to_generalise": True,
        "defaulted_unspecified_routed_to_exhausted": bool(default_exhausted and not exhausted),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


def refuse_if_restatement(
    candidate: Mapping[str, Any],
    scar: Mapping[str, Any],
    killed: Sequence[Mapping[str, Any]],
) -> None:
    verdict = restatement_verdict(candidate, scar, killed)
    if verdict is None:
        return
    raise RestatementRefused(
        str(candidate.get("id") or "<no-id>"),
        str(verdict["killed_family"]),
        str(verdict["killed_by"]),
        str(verdict["reason"]),
    )


def require_census_bytes(byte_row: Mapping[str, Any]) -> int:
    if not byte_row.get("rankable") or not isinstance(byte_row.get("bytes"), int) or byte_row["bytes"] <= 0:
        raise UnrankableOrgan(str(byte_row.get("school") or "<no-school>"), str(byte_row.get("reason") or "no census bytes"))
    return int(byte_row["bytes"])


def expected_ig_per_cost_milli(source_bytes: int, independence_weight: int, cost_weight: int) -> int:
    """Structural score. Not EBPW, not capability, not a measurement.

    Uses MiB so a 376 kB island cannot impersonate a 100 GB organ.
    """
    if source_bytes <= 0 or independence_weight < 0 or cost_weight <= 0:
        raise ValueError("ig/cost refuses non-positive cost or negative independence")
    return (source_bytes // MIB) * independence_weight // cost_weight


# ---------------------------------------------------------------------------
# Ranking.
# ---------------------------------------------------------------------------


def rank_school(
    school: str,
    *,
    inventory: Mapping[str, Any],
    scar: Mapping[str, Any],
    killed: Sequence[Mapping[str, Any]],
    source_bytes_override: int | None = None,
) -> dict[str, Any]:
    if school not in fs.SCHOOL_CATALOG:
        raise fs.UnknownSchoolError(school)
    pivot = SCHOOL_PIVOT[school]
    indep_class, indep_weight, indep_why = INDEPENDENCE[school]
    cost_class = pivot["cost_class"]
    cost_weight = COST_WEIGHT[cost_class]
    byte_row = census_bytes_for(school, inventory)
    if source_bytes_override is not None:
        byte_row = dict(byte_row)
        byte_row["bytes"] = source_bytes_override if source_bytes_override > 0 else None
        byte_row["rankable"] = source_bytes_override > 0
        byte_row["reason"] = (
            f"override bytes={source_bytes_override}" if source_bytes_override > 0
            else "override: no census bytes; refusing a guessed rank"
        )
        byte_row["provenance"] = "test_override" if source_bytes_override else "no_census_bytes"

    candidate = {
        "id": f"{school}:PIVOT",
        "school": school,
        "family": pivot["mechanism"],
        "mechanism": pivot["mechanism"],
        "organ": pivot["organ"],
        "surface": pivot["surface"],
        "hypothesis_family": school.lower(),
    }
    restatement = restatement_verdict(candidate, scar, killed)
    if restatement is not None:
        return {
            "school": school,
            "organ_slug": fs.SCHOOL_ORGAN_SLUG[school],
            "status": "REFUSED_RESTATEMENT",
            "rank": None,
            "expected_ig_per_cost_milli": None,
            "source_bytes": byte_row.get("bytes"),
            "bytes_provenance": byte_row.get("provenance"),
            **pivot,
            "independence_class": indep_class,
            "independence_weight": indep_weight,
            "independence_why": indep_why,
            "cost_weight": cost_weight,
            "restatement": restatement,
            "not_ebpw": True,
            "not_capability": True,
            "not_a_measurement": True,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }

    if not byte_row.get("rankable"):
        return {
            "school": school,
            "organ_slug": fs.SCHOOL_ORGAN_SLUG[school],
            "status": "UNRANKABLE",
            "rank": None,
            "expected_ig_per_cost_milli": None,
            "source_bytes": None,
            "bytes_provenance": byte_row.get("provenance"),
            "unrankable_reason": byte_row.get("reason"),
            **pivot,
            "independence_class": indep_class,
            "independence_weight": indep_weight,
            "independence_why": indep_why,
            "cost_weight": cost_weight,
            "function_organ": school in FUNCTION_ORGANS,
            "not_ebpw": True,
            "not_capability": True,
            "not_a_measurement": True,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }

    source_bytes = int(byte_row["bytes"])
    ig = expected_ig_per_cost_milli(source_bytes, indep_weight, cost_weight)
    parent = pivot.get("parent_budget_shared_with")
    return {
        "school": school,
        "organ_slug": fs.SCHOOL_ORGAN_SLUG[school],
        "status": "RANKED",
        "rank": None,  # filled after sort
        "expected_ig_per_cost_milli": ig,
        "source_bytes": source_bytes,
        "bytes_provenance": byte_row.get("provenance"),
        "source_mib": source_bytes // MIB,
        **{k: v for k, v in pivot.items()},
        "independence_class": indep_class,
        "independence_weight": indep_weight,
        "independence_why": indep_why,
        "cost_weight": cost_weight,
        "parent_budget_shared_with": parent,
        "scores_not_additive_with": [parent] if parent else [],
        "census_families": list(fs.SCHOOL_CENSUS_FAMILIES.get(school) or ()),
        "function_organ": school in FUNCTION_ORGANS,
        "not_ebpw": True,
        "not_capability": True,
        "not_a_measurement": True,
        "status_is_not_a_causal_claim": True,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


def classify_candidate(
    candidate: Mapping[str, Any],
    *,
    scar: Mapping[str, Any],
    killed: Sequence[Mapping[str, Any]],
    source_bytes: int | None,
) -> dict[str, Any]:
    """One mechanism row. Restatement / unrankable / ranked. Never a default pass."""
    ident = str(candidate.get("id") or "<no-id>")
    restatement = restatement_verdict(candidate, scar, killed)
    if restatement is not None:
        return {
            "id": ident,
            "status": "REFUSED_RESTATEMENT",
            "killed_family": restatement["killed_family"],
            "killed_by": restatement["killed_by"],
            "reason": restatement["reason"],
            "organ": candidate.get("organ"),
            "surface": candidate.get("surface"),
            "school": candidate.get("school"),
            "mechanism": candidate.get("mechanism"),
            "expected_ig_per_cost_milli": None,
            "not_ebpw": True,
            "not_capability": True,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }
    if not isinstance(source_bytes, int) or source_bytes <= 0:
        return {
            "id": ident,
            "status": "UNRANKABLE",
            "reason": "organ has no census bytes; refusing a guessed rank",
            "organ": candidate.get("organ"),
            "surface": candidate.get("surface"),
            "school": candidate.get("school"),
            "mechanism": candidate.get("mechanism"),
            "expected_ig_per_cost_milli": None,
            "not_ebpw": True,
            "not_capability": True,
            "evidence_class": "STATIC_ONLY",
            "gpu_authority": False,
        }
    return {
        "id": ident,
        "status": "RANKED",
        "school": candidate.get("school"),
        "organ": candidate.get("organ"),
        "surface": candidate.get("surface"),
        "mechanism": candidate.get("mechanism"),
        "family": candidate.get("family"),
        "cheapest_falsifier": candidate.get("cheapest_falsifier"),
        "why_not_restatement": candidate.get("why_not_restatement"),
        "source_bytes": source_bytes,
        "not_ebpw": True,
        "not_capability": True,
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
    }


def restatement_probes(scar: Mapping[str, Any]) -> list[dict[str, Any]]:
    organ = str(scar.get("organ") or EXHAUSTED_ORGAN)
    tensor = str(scar.get("tensor") or EXHAUSTED_TENSOR)
    return [
        {
            "id": "PROBE-GATEUP-SHARED-LATENT-RANK128",
            "school": "ROUTED_EXPERTS",
            "family": "shared_input_latent_plus_expert_local_output_readout",
            "mechanism": "more ranks of shared input latent plus expert local output readout (rank 128)",
            "organ": organ,
            "tensor": tensor,
            "cheapest_falsifier": "already falsified on this surface; do not spend another rank sweep",
        },
        {
            "id": "PROBE-GATEUP-COMMON-LEFT",
            "school": "ROUTED_EXPERTS",
            "family": "common_left_subspace_plus_expert_local_core",
            "mechanism": "common left subspace plus expert-local core on gate_up",
            "organ": organ,
            "tensor": tensor,
        },
        {
            "id": "PROBE-GATEUP-COMMON-RIGHT",
            "school": "ROUTED_EXPERTS",
            "family": "common_right_subspace_plus_expert_local_core",
            "mechanism": "common right subspace plus expert-local core on gate_up",
            "organ": organ,
            "tensor": tensor,
        },
        {
            "id": "PROBE-GATEUP-CLUSTERED",
            "school": "ROUTED_EXPERTS",
            "family": "clustered_subspaces_route_conditioned",
            "mechanism": "clustered subspaces route-conditioned on gate_up",
            "organ": organ,
            "tensor": tensor,
        },
        {
            "id": "PROBE-GATEUP-DICTIONARY",
            "school": "ROUTED_EXPERTS",
            "family": "dictionary_plus_per_expert_sparse_residual",
            "mechanism": "dictionary plus per-expert sparse residual on gate_up",
            "organ": organ,
            "tensor": tensor,
        },
        {
            "id": "PROBE-GATEUP-SANDWICH",
            "school": "ROUTED_EXPERTS",
            "family": "expert_local_small_core_plus_shared_decoder",
            "mechanism": "small-core sandwich / shared decoder on gate_up",
            "organ": organ,
            "tensor": tensor,
        },
        {
            "id": "PROBE-GATEUP-SHARED-EXPERT-RESIDUAL",
            "school": "ROUTED_EXPERTS",
            "family": "sparse_residual_on_cheap_backbone",
            "mechanism": "residual on architecture shared expert as backbone for routed gate_up",
            "organ": organ,
            "tensor": tensor,
        },
        {
            "id": "PROBE-GATEUP-LOW-RANK-SHARING",
            "school": "ROUTED_EXPERTS",
            "family": "unlabelled_low_rank_share",
            "mechanism": "low-rank sharing under a new label on gate_up",
            "organ": organ,
            "tensor": tensor,
        },
        {
            "id": "PROBE-GATEUP-Q3-SWEEP",
            "school": "ROUTED_EXPERTS",
            "family": "per_expert_q3_control",
            "mechanism": "more ranks of q3 on the exhausted gate_up surface",
            "organ": organ,
            "tensor": tensor,
        },
        {
            "id": "PROBE-DOWN-COMMON-LEFT",
            "school": "ROUTED_EXPERTS",
            "family": "common_left_subspace_plus_expert_local_core",
            "mechanism": "common left subspace on down_proj (different tensor; scoped scar does not generalise)",
            "organ": "layer_4.routed_experts.down_proj",
            "surface": "routed_experts.down_proj",
            "why_not_restatement": "down_proj is not gate_up_proj; scar refuses to generalise",
        },
        {
            "id": "PROBE-NGRAM-PRODUCT-CODEBOOKS",
            "school": "NGRAM",
            "family": "product_codebooks",
            "mechanism": "product codebooks of the n-gram table",
            "organ": "ngram_embedding",
            "surface": "n-gram table lookup",
            "why_not_restatement": "different organ than routed gate_up",
            "cheapest_falsifier": "ngram_school five-axis Pareto; kill if dominated on every axis",
        },
    ]


def live_mechanism_rows(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Mechanisms the task named that are not restatements when placed off gate_up."""
    ngram_bytes = census_bytes_for("NGRAM", inventory).get("bytes")
    routed_bytes = census_bytes_for("ROUTED_EXPERTS", inventory).get("bytes")
    hc_bytes = census_bytes_for("HC_HYPERCONNECTION", inventory).get("bytes")
    return [
        {
            "id": "MECH-NGRAM-PRODUCT-CODEBOOKS",
            "school": "NGRAM",
            "family": "product_codebooks",
            "mechanism": "product codebooks of the n-gram table (ngram_school)",
            "organ": "ngram_embedding",
            "surface": "n-gram table lookup",
            "cheapest_falsifier": "Five-axis Pareto; kill if dominated on executable_bytes, active_lookup, ops, decode, sensitivity.",
            "why_not_restatement": "n-gram organ, not routed gate_up dictionary-of-experts",
            "source_bytes": ngram_bytes,
            "extends": "tools/future/ngram_school.py#product_codebooks",
        },
        {
            "id": "MECH-NGRAM-HIERARCHICAL",
            "school": "NGRAM",
            "family": "hierarchical_codebooks",
            "mechanism": "hierarchical codebooks of the n-gram table",
            "organ": "ngram_embedding",
            "surface": "n-gram table lookup",
            "cheapest_falsifier": "Kill if the walk depth that saves bytes fails retrieval.",
            "why_not_restatement": "n-gram organ; not a routed-expert dictionary on gate_up",
            "source_bytes": ngram_bytes,
            "extends": "tools/future/ngram_school.py#hierarchical_codebooks",
        },
        {
            "id": "MECH-NGRAM-GENERATED-LOOKUP",
            "school": "NGRAM",
            "family": "generated_lookup",
            "mechanism": "generated lookup from token embeds + MLP; frequency-tiered generator + hot islands",
            "organ": "ngram_embedding",
            "surface": "n-gram table lookup",
            "cheapest_falsifier": "Kill if generated rows fail held-out n-gram retrieval.",
            "why_not_restatement": "functional replacement of a table, not a low-rank of gate_up",
            "source_bytes": ngram_bytes,
            "extends": "tools/future/ngram_school.py#generated_lookup",
        },
        {
            "id": "MECH-NGRAM-LITERAL-ISLANDS",
            "school": "NGRAM",
            "family": "literal_exception_islands",
            "mechanism": "sparse literal exception islands on Zipf heavy-hitters",
            "organ": "ngram_embedding",
            "surface": "n-gram table lookup",
            "cheapest_falsifier": "Kill if the island set required to pass retrieval exceeds independent Q4 bytes.",
            "why_not_restatement": "Zipf islands on a lookup table, not residual-on-shared-expert for routed gate_up",
            "source_bytes": ngram_bytes,
            "extends": "tools/future/ngram_school.py#literal_exception_islands",
        },
        {
            "id": "MECH-ROUTED-W2-DOWN",
            "school": "ROUTED_EXPERTS",
            "family": "cross_layer_expert_prediction",
            "mechanism": "cross-layer expert prediction on down_proj / other layers (expert_bank STORE-CROSS-LAYER-EXPERT-PREDICTION)",
            "organ": "routed_experts.down_proj",
            "surface": "routed-output / W2 down_proj",
            "cheapest_falsifier": "Kill if predicting W from other layers fails the coherence contract on a captured down_proj / routed-output surface.",
            "why_not_restatement": "different tensor and a different mechanism than intra-layer low-rank sharing on gate_up",
            "source_bytes": routed_bytes,
            "extends": "tools/future/expert_bank_school.py#STORE-CROSS-LAYER-EXPERT-PREDICTION",
            "surface_bytes_itemized": False,
            "surface_bytes_note": (
                "census largest_tensors are gate_up_proj; down_proj is not itemized. "
                "Parent family bytes are cited; per-tensor down_proj bytes are not guessed."
            ),
        },
        {
            "id": "MECH-ROUTE-CONDITIONED-ON-DOWN",
            "school": "ROUTED_EXPERTS",
            "family": "route_conditioned_archetypes",
            "mechanism": "route-conditioned archetypes on down_proj (not clustered subspaces on gate_up)",
            "organ": "routed_experts.down_proj",
            "surface": "routed-output / W2 down_proj",
            "cheapest_falsifier": "Kill if the K that beats the orthogonal null is ~E, or if the method reconstructs a held-out expert (dead merge).",
            "why_not_restatement": "clustered_subspaces_route_conditioned is dead on gate_up; this placement is a different tensor. Same-surface rerun is refused separately.",
            "source_bytes": routed_bytes,
            "extends": "tools/future/expert_bank_school.py#STORE-ROUTE-CONDITIONED-ARCHETYPES",
        },
        {
            "id": "MECH-HC-GENERATED-COEFFICIENTS",
            "school": "HC_HYPERCONNECTION",
            "family": "generated_coefficients",
            "mechanism": "generated / fused HC mix coefficients; kernels merge, coefficients stay",
            "organ": "mlp_hyperconnection",
            "surface": "HC mix coefficients",
            "cheapest_falsifier": "Kill if coefficients are dropped or fused order disagrees with source HC read.",
            "why_not_restatement": "exact mix island, not a routed expert factor",
            "source_bytes": hc_bytes,
            "extends": "tools/future/flash_schools.py#HC_HYPERCONNECTION:ZERO-STANDALONE-KERNELS",
        },
        {
            "id": "MECH-FUNCTIONAL-REPLACEMENT-NGRAM",
            "school": "NGRAM",
            "family": "generated_lookup",
            "mechanism": "replace the 320M-row table with a generator (function, not tensor packing)",
            "organ": "ngram_embedding",
            "surface": "n-gram table lookup",
            "cheapest_falsifier": "Kill if the generator fails held-out retrieval; packing Q4 of the table is a control, not this candidate.",
            "why_not_restatement": "functional replacement of a lookup organ, not latent+readout on gate_up",
            "source_bytes": ngram_bytes,
            "extends": "tools/future/flash_schools.py#NGRAM",
        },
    ]


def rank_all(
    *,
    inventory: Mapping[str, Any] | None = None,
    screen: Mapping[str, Any] | None = None,
    screen_rel: str | None = None,
) -> dict[str, Any]:
    inv = dict(inventory) if inventory is not None else load_inventory()
    scar = scoped_scar(screen, screen_rel=screen_rel)
    killed = killed_families_from_screen(screen)
    rows = [rank_school(sid, inventory=inv, scar=scar, killed=killed) for sid in fs.SCHOOL_CATALOG]
    ranked = [r for r in rows if r["status"] == "RANKED"]
    ranked.sort(
        key=lambda r: (
            -(r["expected_ig_per_cost_milli"] or 0),
            -(r["source_bytes"] or 0),
            r["school"],
        )
    )
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i
    by_school = {r["school"]: r["rank"] for r in ranked}
    for r in rows:
        if r["status"] == "RANKED":
            r["rank"] = by_school[r["school"]]

    probes = restatement_probes(scar)
    probe_verdicts = [
        classify_candidate(p, scar=scar, killed=killed, source_bytes=1)
        for p in probes
    ]
    refused_probes = [p for p in probe_verdicts if p["status"] == "REFUSED_RESTATEMENT"]
    live_probes = [p for p in probe_verdicts if p["status"] != "REFUSED_RESTATEMENT"]

    mechanisms = []
    for mech in live_mechanism_rows(inv):
        mechanisms.append(
            classify_candidate(
                mech,
                scar=scar,
                killed=killed,
                source_bytes=mech.get("source_bytes") if isinstance(mech.get("source_bytes"), int) else None,
            )
            | {k: mech[k] for k in ("extends", "why_not_restatement", "cheapest_falsifier") if k in mech}
        )

    return {
        "scar": scar,
        "killed_families": killed,
        "schools": rows,
        "ranked": ranked,
        "unrankable": [r for r in rows if r["status"] == "UNRANKABLE"],
        "refused_schools": [r for r in rows if r["status"] == "REFUSED_RESTATEMENT"],
        "restatement_probes": probe_verdicts,
        "n_restatement_probes_refused": len(refused_probes),
        "n_restatement_probes_live": len(live_probes),
        "mechanisms": mechanisms,
        "inventory": {
            "census_source": inv.get("census_source"),
            "router_source": inv.get("router_source"),
            "specimen_bytes": inv.get("specimen_bytes"),
            "n_census_families": inv.get("n_census_families"),
            "families": inv.get("families"),
            "reachable": inv.get("reachable"),
        },
    }


def prove_negative_controls(
    scar: Mapping[str, Any],
    killed: Sequence[Mapping[str, Any]],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Watched-fail proofs. A validator nobody has watched reject is not a validator."""
    restatement_probe = {
        "id": "PROOF-MORE-RANKS-SHARED-LATENT",
        "school": "ROUTED_EXPERTS",
        "family": "shared_input_latent_plus_expert_local_output_readout",
        "mechanism": "more ranks of shared input latent plus expert-local readout",
        "organ": scar.get("organ") or EXHAUSTED_ORGAN,
        "tensor": scar.get("tensor") or EXHAUSTED_TENSOR,
    }
    restatement_fired = False
    restatement_error = None
    restatement_family = None
    restatement_receipt = None
    try:
        refuse_if_restatement(restatement_probe, scar, killed)
    except RestatementRefused as exc:
        restatement_fired = True
        restatement_error = str(exc)
        restatement_family = exc.killed_family
        restatement_receipt = exc.killed_by

    unrank_fired = False
    unrank_error = None
    empty = {
        "census_source": "unavailable",
        "reachable": True,  # reachable empty listing: organ is present, bytes are not
        "by_family": {},
        "families": [],
        "router_tensor_bytes": None,
    }
    try:
        require_census_bytes(census_bytes_for("NGRAM", empty))
    except UnrankableOrgan as exc:
        unrank_fired = True
        unrank_error = str(exc)

    down = classify_candidate(
        {
            "id": "PROOF-DOWN-COMMON-LEFT-LIVE",
            "school": "ROUTED_EXPERTS",
            "family": "common_left_subspace_plus_expert_local_core",
            "mechanism": "common left on down_proj",
            "organ": "layer_4.routed_experts.down_proj",
            "surface": "down_proj",
        },
        scar=scar,
        killed=killed,
        source_bytes=1,
    )
    ngram = classify_candidate(
        {
            "id": "PROOF-NGRAM-PRODUCT-LIVE",
            "school": "NGRAM",
            "family": "product_codebooks",
            "mechanism": "product codebooks of the n-gram table",
            "organ": "ngram_embedding",
        },
        scar=scar,
        killed=killed,
        source_bytes=1,
    )
    return {
        "restatement_refused": restatement_fired,
        "restatement_error": restatement_error,
        "restatement_killed_family": restatement_family,
        "restatement_killed_by": restatement_receipt,
        "unrankable_without_bytes": unrank_fired,
        "unrankable_error": unrank_error,
        "down_proj_common_left_not_refused": down.get("status") != "REFUSED_RESTATEMENT",
        "down_proj_status": down.get("status"),
        "ngram_product_codebooks_not_refused": ngram.get("status") != "REFUSED_RESTATEMENT",
        "ngram_status": ngram.get("status"),
        "watched_fail": bool(restatement_fired and unrank_fired),
        "inventory_cited": bool(inventory.get("census_source")),
    }


def next_workunits(ranking: Mapping[str, Any]) -> list[dict[str, Any]]:
    ranked = list(ranking.get("ranked") or [])
    units: list[dict[str, Any]] = []
    for row in ranked[:4]:
        units.append(
            {
                "id": f"future.flash_organ_pivot.{row['school']}",
                "species": "CPU_ANALYSIS",
                "school": row["school"],
                "surface": row.get("surface"),
                "mechanism": row.get("mechanism"),
                "cheapest_falsifier": row.get("cheapest_falsifier"),
                "resource_lane": "CPU_ANALYSIS",
                "gpu_authority": False,
                "not_a_hardware_lease": True,
                "do_not": "more ranks of killed families on layer_4.routed_experts.gate_up_proj",
            }
        )
    units.append(
        {
            "id": "future.flash_organ_pivot.do_not_resweep_gate_up",
            "species": "CPU_ANALYSIS",
            "action": "REFUSE",
            "reason": (
                "do not spend real teacher rows on more ranks of killed families "
                f"on {EXHAUSTED_ORGAN}; scar is scoped and the ranking exists to leave it"
            ),
            "killed_by": KILLED_BY,
            "gpu_authority": False,
        }
    )
    return units


def recovered_implementation() -> list[dict[str, str]]:
    return [
        {"path": "tools/future/rival_codec_screen.py", "role": "exhausted surface: 10 families, 0 passes, 0 beat Q4, harness that reproduced the committed screen"},
        {"path": "receipts/future/RIVAL_CODEC_SCREEN.json", "role": "the scar this pivot leaves; not re-run"},
        {"path": "tools/future/flash_meta_replan.py", "role": "keep the gate; next rows go to unmeasured surfaces; scoped falsification"},
        {"path": "receipts/future/FLASH_META_REPLAN.json", "role": "UNTOUCHED families and do-not-spend-ranks instruction"},
        {"path": "tools/future/flash_schools.py", "role": "fourteen organ schools, census family map, elimination answers, school_source_bytes"},
        {"path": "tools/future/flash_organ_workgraphs.py", "role": "function organs vs empty census family; teacher-fit sleep"},
        {"path": "tools/future/expert_bank_school.py", "role": "named mechanisms; dead raw-global / trivial-basis / unchanged-archetype already refused"},
        {"path": "tools/future/ngram_school.py", "role": "product/hierarchical/generated/literal-island families on the n-gram organ"},
        {"path": "tools/future/router_science.py", "role": "control-plane overlay bytes and margin-gated residual"},
        {"path": "tools/future/moe_physical_school.py", "role": "physical fusion of shared+routed, not a storage restatement"},
        {"path": "tools/future/meta_funnel.py", "role": "load_receipt disk-then-HEAD so a sparse checkout is not treated as absence"},
        {"path": "receipts/future/evidence/FLASH_ORGAN_CENSUS.json", "role": "family_summary source bytes (git HEAD if not materialized)"},
        {"path": "receipts/future/FLASH_ORGAN_SCHOOLS.json", "role": "sealed sidecar inventory fallback when the census file is not on disk"},
    ]


def gaps_closed() -> list[str]:
    return [
        "No sidecar module turned the rival-codec exhaustion into a pivot ranking that leaves that surface.",
        "Restatement refusal that names the killed family and the receipt that killed it, and actually fires.",
        "Organs with no census bytes are UNRANKABLE rather than guessed.",
        "Scoped scar records organ, surface, split and explicitly refuses to generalise.",
        "Ranking is expected information gain per cost, not EBPW and not capability.",
    ]


def negative_findings(ranking: Mapping[str, Any]) -> list[str]:
    inv = ranking.get("inventory") or {}
    findings = [
        "Did not re-fit any codec and did not consume another teacher row.",
        "Did not itemize down_proj bytes; census largest_tensors are gate_up_proj. Parent family bytes are cited; a 50/50 split is refused.",
        "Independence weights are structural priors from elimination_answers plus the scoped scar, not measured mutual information.",
        "EMBEDDING and LM_HEAD share the embedding_lm_head parent budget unsplit; their scores are not additive.",
        "Function organs with no census bytes are UNRANKABLE; that is not a claim they carry zero information.",
        "orchestration.BINDINGS is outside this lane's WRITE list; the frontier named below is declared, not wired.",
        "Did not take a hardware measurement, acquire a GPU lease, or claim physical EBPW / capability.",
    ]
    if not inv.get("reachable"):
        findings.append(
            f"Census source={inv.get('census_source')}; ranking used whatever cited overlay/inventory was reachable and left the rest UNRANKABLE."
        )
    return findings


def build() -> Any:
    screen, screen_rel = load_named(RIVAL_REL)
    replan, replan_rel = load_named(REPLAN_REL)
    inventory = load_inventory()
    ranking = rank_all(inventory=inventory, screen=screen, screen_rel=screen_rel)
    scar = ranking["scar"]
    killed = ranking["killed_families"]
    proof = prove_negative_controls(scar, killed, inventory)

    ranked = ranking["ranked"]
    top = ranked[0] if ranked else None
    claims = {
        "ranking_is_expected_information_gain_per_cost": True,
        "ranking_is_ebpw": False,
        "ranking_is_capability": False,
        "ranking_is_hardware": False,
        "any_contract_pass_on_exhausted_surface": scar.get("any_family_passed_contract"),
        "any_family_beats_q4_on_exhausted_surface": scar.get("any_family_beats_q4"),
        "q4_is_best_tested_local_comparator": scar.get("q4_is_best_tested_local_comparator"),
        "scar_is_scoped": bool(scar.get("refuses_to_generalise")),
        "scar_organ": scar.get("organ"),
        "scar_surface": scar.get("surface"),
        "scar_split": scar.get("split"),
        "n_ranked": len(ranked),
        "n_unrankable": len(ranking["unrankable"]),
        "n_restatements_refused": ranking["n_restatement_probes_refused"],
        "top_school": (top or {}).get("school"),
        "status_is_not_a_causal_claim": True,
    }

    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Stop spending real teacher rows on more ranks of killed families on "
            f"{EXHAUSTED_ORGAN}. Rank remaining Flash organ schools by expected "
            "information gain per unit of cost, refuse restatements, and keep the scar scoped."
        ),
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "status": "STATIC_PIVOT_RANKING",
        "status_is_not_a_causal_claim": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "cited_inputs": {
            "rival_codec_screen": {"path": screen_rel or RIVAL_REL, "present": screen is not None},
            "flash_meta_replan": {"path": replan_rel or REPLAN_REL, "present": replan is not None},
            "census": {"path": inventory.get("census_rel") or CENSUS_RELS[0], "present": bool(inventory.get("reachable"))},
            "router_overlay": {"source": inventory.get("router_source"), "bytes": inventory.get("router_tensor_bytes")},
        },
        "scoped_scar": scar,
        "killed_families": killed,
        "principle": (
            "If routed gate_up weights are information-dense, that is not a failure: "
            "spend more bits there. Complete-model executable information is global; "
            "the model does not need every organ near 1 bpw. Rank for total executable "
            "information subject to capability, not for uniformity. This receipt does "
            "not claim an EBPW or capability number."
        ),
        "ranking_rule": (
            "expected_ig_per_cost_milli = (census_bytes // 1MiB) * independence_weight "
            "// cost_weight. Independence is a structural prior. Cost is cheapest-falsifier "
            "class. Scalar EBPW ranking is refused."
        ),
        "ranked_schools": ranked,
        "unrankable_schools": ranking["unrankable"],
        "refused_schools": ranking["refused_schools"],
        "all_schools": ranking["schools"],
        "mechanisms": ranking["mechanisms"],
        "restatement_probes": ranking["restatement_probes"],
        "negative_control": proof,
        "claims": claims,
        "inventory": ranking["inventory"],
        "replan_next_capture": (replan or {}).get("next_capture") if isinstance(replan, dict) else None,
        "untouched_families_from_replan": (replan or {}).get("untouched_families") if isinstance(replan, dict) else None,
        "next_workunits": next_workunits(ranking),
        "recovered_implementation": recovered_implementation(),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(ranking),
        "resident_callable": {
            "entry_point": "tools.future.flash_organ_pivot.rank_all()",
            "workunit": (
                "one CPU_ANALYSIS unit; rank Flash organ schools by expected "
                "information gain per cost off the exhausted gate_up surface; no GPU"
            ),
            "receipt": f"receipts/future/{RECEIPT}",
            "frontier": "FT.MODEL_REPRESENTATION.meta-gates-3-9",
            "fails_closed": (
                "restatement of a killed family on the exhausted surface raises "
                "RestatementRefused naming the family and receipts/future/RIVAL_CODEC_SCREEN.json; "
                "an organ with no census bytes is UNRANKABLE; absent screen records "
                "REFUSED_UNAVAILABLE rather than a generalised scar; ranking fields "
                "are tagged not_ebpw / not_capability / not_a_measurement"
            ),
            "discoverable": True,
        },
        "ebs_kinds_consulted": list(ebs.REQUIRED_STORAGE_KINDS),
        "ngram_families_consulted": [c.get("id") or c.get("family") for c in ngs.candidates()],
        "school_catalog": list(fs.SCHOOL_CATALOG),
    }
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.parse_args()
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
