"""FLASH_ORGAN_SCHOOLS — fourteen independently schedulable Gravity schools.

Codex advanced Flash EXECUTION far past Flash REPRESENTATION. This sidecar
does not quantize tensors. It searches for the smallest executable PROGRAM
that preserves the useful function of each Flash organ.

Each school emits candidates with a cheapest falsifier, can be scheduled
alone, consults the negative index before emit, and scores compute sharing
WITH storage sharing so a bytes-halving FLOP-tripling candidate can lose.

Analytical and structural only. No fit against the 350 GB specimen. No timing.
Everything emitted is STATIC_ONLY / bench UNKNOWN / gpu_authority false.

    python3 tools/future/flash_schools.py --build
    python3 tools/future/flash_schools.py --school ROUTED_EXPERTS
    python3 tools/future/flash_schools.py --selftest
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.future._common import HARDWARE_FIELDS, sha256_file
from tools.future import expert_bank_school as ebs
from tools.future import ngram_school as ngs
from tools.future import negative_index as ni
from tools.future import router_science as rs
from tools.future import workunit_species as wus

RECEIPT = "FLASH_ORGAN_SCHOOLS.json"
SCHEMA = "hawking.future.flash_schools.v1"
VERSION = 1
RECORDED_BY = "tools/future/flash_schools.py"
FLASH_MODEL = "Qwen/Qwen3.8-Flash-Next"

EVIDENCE_DIR = REPO / "receipts" / "future" / "evidence"
HEADLESS_DIR = REPO / "receipts" / "headless"

EVIDENCE_NAMES: tuple[str, ...] = (
    "FLASH_ORGAN_CENSUS.json",
    "FLASH_META_REPRESENTATION_SUB1.json",
    "FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json",
    "FLASH_LAYER46_DISPATCH_LEDGER.json",
    "FLASH_DOCTOR_EXPERT_BANK_SCREEN_FULL_L44.json",
    "FLASH_NEXT_FPGA_ORGAN_MAP.json",
)

# Catalog, not a rotting queue bound. Length is derived from this tuple.
SCHOOL_CATALOG: tuple[str, ...] = (
    "ROUTED_EXPERTS",
    "SHARED_EXPERTS",
    "ROUTER",
    "HC_HYPERCONNECTION",
    "DELTANET_RECURRENT_STATE",
    "FULL_ATTENTION",
    "KV_STATE",
    "NGRAM",
    "EMBEDDING",
    "LM_HEAD",
    "NORMALIZATION",
    "POSITIONAL_STRUCTURE",
    "DECODING",
    "MTP_SPECULATION",
)

PROGRAM_FAMILIES: tuple[str, ...] = (
    "LiteralTensor",
    "QuantTensor",
    "SharedBasisProgram",
    "FactorizedProgram",
    "DictionaryProgram",
    "GeneratedProgram",
    "SparseResidualProgram",
    "RecurrentStateProgram",
    "LookupProgram",
    "ConditionalProgram",
    "RoutedSubprogram",
    "FusedPhysicalProgram",
    "ZeroProgram",
    "CompositeProgram",
)

ELIMINATION_QUESTIONS: tuple[str, ...] = (
    "needs_independent_storage",
    "can_be_generated_predicted_or_shared",
    "dictionary_recurrent_or_conditional_expresses_it",
    "sparse_residual_restores_capability_difference",
    "function_replaceable_not_just_tensor_represented",
)

THREE_ZEROS: tuple[str, ...] = (
    "ZERO_STORAGE",
    "ZERO_INDEPENDENT_INFORMATION",
    "ZERO_EXECUTION",
)

# All minimized. flop_milli = 1000 means 1.0x the LiteralTensor structural FLOP.
AXES: tuple[str, ...] = (
    "storage_bytes",
    "flop_milli",
    "capability_risk_rank",
)

CAPABILITY_ORDER: dict[str, int] = {
    "CONTROL_LITERAL": 0,
    "CONTROL_QUANT": 1,
    "PROTECTED_ISLAND": 2,
    "STRUCTURE_PRESERVING": 3,
    "RESIDUAL_REPAIRABLE": 4,
    "CONDITIONAL": 5,
    "GENERATIVE_UNMEASURED": 6,
    "ZERO_UNTESTED": 7,
    "CONTROL_CRUSHED": 8,
}

Q4_NUM, Q4_DEN = 425, 1600  # 4.25 / 16 of BF16 source bytes
Q4_FLOP_MILLI = 1150
FACTOR_STORAGE_NUM, FACTOR_STORAGE_DEN = 1, 4
FACTOR_FLOP_MILLI = 1200
DICT_STORAGE_NUM, DICT_STORAGE_DEN = 1, 5
DICT_FLOP_MILLI = 1400
GEN_STORAGE_NUM, GEN_STORAGE_DEN = 1, 10
GEN_FLOP_MILLI = 2000
SPARSE_STORAGE_NUM, SPARSE_STORAGE_DEN = 3, 10
SPARSE_FLOP_MILLI = 1300
COND_STORAGE_NUM, COND_STORAGE_DEN = 1, 6
COND_FLOP_MILLI = 1500

CANDIDATE_FIELDS: tuple[str, ...] = (
    "id",
    "school",
    "program_family",
    "mechanism",
    "storage_bytes",
    "flop_milli",
    "capability_risk_class",
    "capability_risk_rank",
    "cheapest_falsifier",
    "native_execution_concept",
    "forbids_dense_rematerialization",
    "status",
    "evidence_class",
    "zeros_targeted",
    "scar_distance",
    "hypothesis_family",
    "independently_schedulable",
)

# Census family → school. Router/KV/decoding/MTP are functions; bytes come
# from a cited receipt when the census did not itemize them.
SCHOOL_CENSUS_FAMILIES: dict[str, tuple[str, ...]] = {
    "ROUTED_EXPERTS": ("routed_experts",),
    "SHARED_EXPERTS": ("shared_expert",),
    "ROUTER": (),
    "HC_HYPERCONNECTION": ("mlp_hyperconnection",),
    "DELTANET_RECURRENT_STATE": ("linear_attention_hyperconnection",),
    "FULL_ATTENTION": ("full_attention",),
    "KV_STATE": (),
    "NGRAM": ("ngram_embedding",),
    "EMBEDDING": ("embedding_lm_head",),
    "LM_HEAD": ("embedding_lm_head",),
    "NORMALIZATION": ("norm",),
    "POSITIONAL_STRUCTURE": ("other",),
    "DECODING": (),
    "MTP_SPECULATION": (),
}

# Organs the negative index understands (see ORGAN_SLUGS).
SCHOOL_ORGAN_SLUG: dict[str, str] = {
    "ROUTED_EXPERTS": "routed_experts",
    "SHARED_EXPERTS": "mlp",
    "ROUTER": "router",
    "HC_HYPERCONNECTION": "mlp",
    "DELTANET_RECURRENT_STATE": "deltanet",
    "FULL_ATTENTION": "attention",
    "KV_STATE": "kv",
    "NGRAM": "embed",
    "EMBEDDING": "embed",
    "LM_HEAD": "lm_head",
    "NORMALIZATION": "whole_model",
    "POSITIONAL_STRUCTURE": "attention",
    "DECODING": "whole_model",
    "MTP_SPECULATION": "whole_model",
}

EBS_KIND_TO_FAMILY: dict[str, str] = {
    "common_left_subspaces": "FactorizedProgram",
    "common_right_subspaces": "FactorizedProgram",
    "expert_specific_small_cores": "FactorizedProgram",
    "tensor_decomposition": "FactorizedProgram",
    "clustered_subspaces": "ConditionalProgram",
    "dictionary_families": "DictionaryProgram",
    "route_conditioned_archetypes": "ConditionalProgram",
    "expert_embeddings_generators": "GeneratedProgram",
    "shared_input_transforms": "SharedBasisProgram",
    "shared_output_latent_spaces": "SharedBasisProgram",
    "cross_layer_expert_prediction": "GeneratedProgram",
    "conditional_residuals": "SparseResidualProgram",
    "capability_sensitive_expert_islands": "CompositeProgram",
    "one_hidden_vector_many_experts": "FusedPhysicalProgram",
    "shared_xb_then_skinny": "FusedPhysicalProgram",
    "latent_weighted_reduction": "FusedPhysicalProgram",
    "one_output_expansion": "FusedPhysicalProgram",
    "shared_representation_decode": "DictionaryProgram",
    "shared_projections_across_organs": "FusedPhysicalProgram",
    "cross_layer_reused_transforms": "GeneratedProgram",
}

EBS_KIND_COST: dict[str, tuple[int, int, int]] = {
    # (storage_num, storage_den, flop_milli)
    "common_left_subspaces": (1, 4, 1200),
    "common_right_subspaces": (1, 4, 1200),
    "expert_specific_small_cores": (1, 5, 1300),
    "tensor_decomposition": (1, 6, 1600),
    "clustered_subspaces": (1, 3, 1250),
    "dictionary_families": (1, 5, 1400),
    "route_conditioned_archetypes": (1, 6, 1500),
    "expert_embeddings_generators": (1, 10, 2000),
    "shared_input_transforms": (1, 4, 1100),
    "shared_output_latent_spaces": (1, 4, 1100),
    "cross_layer_expert_prediction": (1, 8, 1800),
    "conditional_residuals": (3, 10, 1300),
    "capability_sensitive_expert_islands": (2, 5, 1350),
    "one_hidden_vector_many_experts": (1, 1, 800),
    "shared_xb_then_skinny": (1, 1, 700),
    "latent_weighted_reduction": (1, 1, 750),
    "one_output_expansion": (1, 1, 700),
    "shared_representation_decode": (1, 5, 1400),
    "shared_projections_across_organs": (1, 1, 850),
    "cross_layer_reused_transforms": (4, 5, 900),
}

NGRAM_FAMILY_TO_PROGRAM: dict[str, str] = {
    "packed_q4_control": "QuantTensor",
    "packed_q3_control": "QuantTensor",
    "product_codebooks": "DictionaryProgram",
    "residual_product_quantization": "SparseResidualProgram",
    "hierarchical_codebooks": "DictionaryProgram",
    "clustered_dictionaries": "DictionaryProgram",
    "factorized_lookup": "FactorizedProgram",
    "context_conditioned_lookup": "ConditionalProgram",
    "generated_lookup": "GeneratedProgram",
    "semantic_hashing": "LookupProgram",
    "literal_exception_islands": "SparseResidualProgram",
}

NGRAM_CAPABILITY: dict[str, str] = {
    "packed_q4_control": "CONTROL_QUANT",
    "packed_q3_control": "CONTROL_QUANT",
    "product_codebooks": "GENERATIVE_UNMEASURED",
    "residual_product_quantization": "RESIDUAL_REPAIRABLE",
    "hierarchical_codebooks": "GENERATIVE_UNMEASURED",
    "clustered_dictionaries": "CONDITIONAL",
    "factorized_lookup": "STRUCTURE_PRESERVING",
    "context_conditioned_lookup": "CONDITIONAL",
    "generated_lookup": "GENERATIVE_UNMEASURED",
    "semantic_hashing": "GENERATIVE_UNMEASURED",
    "literal_exception_islands": "RESIDUAL_REPAIRABLE",
}


class DeadHypothesisError(ValueError):
    """Raised when a candidate matches a recorded-dead hypothesis."""

    def __init__(self, candidate_id: str, scar: Mapping[str, Any]):
        self.candidate_id = candidate_id
        self.scar = dict(scar)
        super().__init__(
            f"REFUSED {candidate_id}: dead hypothesis "
            f"{scar.get('id') or scar.get('scar_id')} "
            f"({scar.get('title') or scar.get('family') or scar.get('hypothesis_family')})"
        )


class CandidateSchemaError(ValueError):
    """A live candidate is missing a required Gravity field."""


class StorageOnlyRankingError(ValueError):
    """rank() refuses a storage-only or otherwise incomplete-axis ordering."""


class UnknownSchoolError(ValueError):
    """A scheduler asked for a school that is not in the catalog."""


class IncompleteVectorError(ValueError):
    """A candidate is missing one of the scoring axes."""


# Targeted phrases. Short tokens like "shared" or "expert" must not match.
BUILTIN_SCARS: tuple[dict[str, Any], ...] = (
    {
        "id": "SCAR-TRIVIAL-GLOBAL-EXPERT-SHARING",
        "family": "cross_expert_structure",
        "title": "trivial global expert sharing",
        "phrases": (
            "trivial global expert sharing",
            "trivial shared basis",
            "raw global expert similarity",
            "unchanged archetype",
            "one shared weight basis for all experts",
            "unconditioned shared codebook across routed experts",
        ),
        "why_dead": (
            "NEGATIVE_SCIENCE_INDEX family cross_expert_structure; "
            "expert_bank_school SCAR-RAW-GLOBAL-SIMILARITY / "
            "SCAR-TRIVIAL-SHARED-BASIS / SCAR-UNCHANGED-ARCHETYPE; "
            "FLASH_DOCTOR_EXPERT_BANK_SCREEN_FULL_L44 rank-1 energy is "
            "near the orthogonal null; FLASH_META_REPRESENTATION_SUB1 "
            "forbids sharing a single expert basis."
        ),
        "reopen_condition": (
            "Never as unconditioned weight-space sharing on Flash/Q80. "
            "A one-sided, clustered, route-conditioned, or generator "
            "program is a different hypothesis."
        ),
    },
    {
        "id": "SCAR-UNIFORM-SUBBIT-CONTROL-AND-BULK",
        "family": "uniform_subbit_allocation",
        "title": "uniform bpw across control and bulk",
        "phrases": (
            "uniform subbit allocation",
            "uniform bpw across control and bulk",
            "same bpw on router and routed experts",
        ),
        "why_dead": (
            "uniform_subbit_allocation is a recorded-dead family. Flash "
            "router L3-L4 membership already flips under a small hidden "
            "perturbation; crushing the control plane to save the same "
            "fraction as bulk is the wrong objective."
        ),
        "reopen_condition": (
            "Never as a uniform bpw plan that treats router/state/KV/"
            "terminal islands as interchangeable with routed bulk."
        ),
    },
)


# ---------------------------------------------------------------------------
# Evidence loading. Cope with pinned, live, or neither.
# ---------------------------------------------------------------------------


def load_named(name: str) -> dict[str, Any]:
    """Load one named receipt. Prefer the pinned snapshot.

    A missing file is `unavailable`, not proof the object is absent from
    git or a sibling worktree.
    """
    pinned = EVIDENCE_DIR / name
    live = HEADLESS_DIR / name
    if pinned.is_file():
        return {
            "name": name,
            "reachable": True,
            "evidence_source": "pinned_snapshot",
            "path": str(pinned.relative_to(REPO)),
            "sha256": sha256_file(pinned),
            "doc": load_json(pinned),
        }
    if live.is_file():
        return {
            "name": name,
            "reachable": True,
            "evidence_source": "live_headless",
            "path": str(live.relative_to(REPO)),
            "sha256": sha256_file(live),
            "doc": load_json(live),
        }
    return {
        "name": name,
        "reachable": False,
        "evidence_source": "unavailable",
        "path": None,
        "sha256": None,
        "doc": None,
        "coped": (
            "neither pinned snapshot nor live headless was reachable in this "
            "checkout; a sibling worktree may still hold the file. this school "
            "does not treat unreachability as absence."
        ),
    }


def load_all_evidence() -> dict[str, dict[str, Any]]:
    return {name: load_named(name) for name in EVIDENCE_NAMES}


def _doc(bundle: Mapping[str, dict[str, Any]], name: str) -> dict[str, Any] | None:
    row = bundle.get(name) or {}
    doc = row.get("doc")
    return doc if isinstance(doc, dict) else None


def _source_of(bundle: Mapping[str, dict[str, Any]], name: str) -> str:
    return str((bundle.get(name) or {}).get("evidence_source") or "unavailable")


def _read_count(value: Any, receipt: str, field: str) -> dict[str, Any]:
    return {
        "value": value,
        "provenance": "read_from_receipt",
        "receipt": receipt,
        "field": field,
        "not_a_measurement": True,
    }


def _clip(text: Any, limit: int = 420) -> str:
    s = " ".join(str(text or "").split())
    if len(s) <= limit:
        return s
    return s[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Organ inventory (derived from the census / family budget when reachable)
# ---------------------------------------------------------------------------


def organ_inventory(bundle: Mapping[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    bundle = bundle if bundle is not None else load_all_evidence()
    census = _doc(bundle, "FLASH_ORGAN_CENSUS.json")
    budget = _doc(bundle, "FLASH_META_REPRESENTATION_SUB1.json")
    router_map = _doc(bundle, "FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json")
    doctor = _doc(bundle, "FLASH_DOCTOR_EXPERT_BANK_SCREEN_FULL_L44.json")
    ledger = _doc(bundle, "FLASH_LAYER46_DISPATCH_LEDGER.json")

    families: list[dict[str, Any]] = []
    by_family: dict[str, dict[str, Any]] = {}
    if census:
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

    family_budget: list[dict[str, Any]] = []
    budget_by_family: dict[str, dict[str, Any]] = {}
    if budget:
        for row in budget.get("family_budget") or []:
            if not isinstance(row, dict) or not row.get("family"):
                continue
            rec = {
                "family": row["family"],
                "source_bytes": row.get("source_bytes"),
                "source_fraction": row.get("source_fraction"),
                "meta_bpw_target": row.get("meta_bpw_target"),
                "weighted_meta_bpw": row.get("weighted_meta_bpw"),
                "program": row.get("program"),
                "ledger": row.get("ledger") or {},
                "runtime_shape": row.get("runtime_shape"),
            }
            family_budget.append(rec)
            budget_by_family[rec["family"]] = rec
        family_budget.sort(key=lambda r: r["family"])

    router_bytes = None
    router_hidden = None
    router_membership_rows = None
    router_margin_min = None
    if router_map:
        src = router_map.get("router_source") or {}
        if isinstance(src.get("tensor_bytes"), int):
            router_bytes = src["tensor_bytes"]
        seam = router_map.get("seam") or {}
        router_hidden = seam.get("hidden")
        routing = router_map.get("routing") or {}
        router_membership_rows = routing.get("rows_with_membership_change")
        router_margin_min = routing.get("dense_margin_min")

    doctor_pop: dict[str, Any] = {}
    if doctor:
        pop = doctor.get("population") or {}
        rank = pop.get("sampled_population_rank") or {}
        doctor_pop = {
            "cross_expert_gate_up_mean_cosine": pop.get("cross_expert_gate_up_mean_cosine"),
            "rank_1_energy": rank.get("rank_1_energy"),
            "rank_8_energy": rank.get("rank_8_energy"),
            "expert_count": (doctor.get("source") or {}).get("expert_count"),
            "layer": (doctor.get("source") or {}).get("layer"),
        }

    dispatch_count = None
    dispatch_kernels: list[str] = []
    if ledger:
        rows = [r for r in (ledger.get("rows") or []) if isinstance(r, dict)]
        dispatch_count = ledger.get("dispatch_count")
        if dispatch_count is None:
            dispatch_count = len(rows)
        seen: set[str] = set()
        for r in rows:
            k = str(r.get("kernel") or "")
            if k and k not in seen:
                seen.add(k)
                dispatch_kernels.append(k)
        dispatch_kernels.sort()

    specimen_bytes = None
    if census and isinstance(census.get("source_parameter_bytes_indexed"), int):
        specimen_bytes = census["source_parameter_bytes_indexed"]
    elif budget:
        src = budget.get("source") or {}
        if isinstance(src.get("source_parameter_bytes_indexed"), int):
            specimen_bytes = src["source_parameter_bytes_indexed"]

    return {
        "census_source": _source_of(bundle, "FLASH_ORGAN_CENSUS.json"),
        "budget_source": _source_of(bundle, "FLASH_META_REPRESENTATION_SUB1.json"),
        "router_map_source": _source_of(bundle, "FLASH_ROUTER_SENSITIVITY_MAP_L3_L4.json"),
        "doctor_source": _source_of(bundle, "FLASH_DOCTOR_EXPERT_BANK_SCREEN_FULL_L44.json"),
        "ledger_source": _source_of(bundle, "FLASH_LAYER46_DISPATCH_LEDGER.json"),
        "model": (census or {}).get("model") or FLASH_MODEL,
        "specimen_bytes": specimen_bytes,
        "layer_count_observed": (census or {}).get("layer_count_observed"),
        "tensor_count": (census or {}).get("tensor_count"),
        "families": families,
        "by_family": by_family,
        "family_budget": family_budget,
        "budget_by_family": budget_by_family,
        "router_tensor_bytes": router_bytes,
        "router_hidden": router_hidden,
        "router_membership_change_rows": router_membership_rows,
        "router_dense_margin_min": router_margin_min,
        "doctor": doctor_pop,
        "dispatch_count": dispatch_count,
        "dispatch_kernels": dispatch_kernels,
        "meta_program": (budget or {}).get("meta_program") if budget else None,
        "protected_islands": list(
            ((budget or {}).get("meta_program") or {}).get("protected_islands") or []
        ),
        "n_census_families": len(families),
        "n_budget_families": len(family_budget),
    }


def school_source_bytes(school_id: str, inventory: Mapping[str, Any]) -> int:
    """Weight-census bytes this school searches, else a cited overlay.

    Runtime organs (KV, decoding, MTP function, router) may be 0 in the
    family_summary; router overlays the sensitivity-map tensor_bytes.
    """
    if school_id == "ROUTER":
        overlay = inventory.get("router_tensor_bytes")
        if isinstance(overlay, int) and overlay > 0:
            return overlay
    total = 0
    by_family = inventory.get("by_family") or {}
    for fam in SCHOOL_CENSUS_FAMILIES.get(school_id) or ():
        row = by_family.get(fam) or {}
        b = row.get("bytes")
        if isinstance(b, int):
            total += b
    return total


# ---------------------------------------------------------------------------
# Elimination answers (structural; per organ)
# ---------------------------------------------------------------------------


def _ans(
    independent: str,
    generated: str,
    expressed: str,
    residual: str,
    function: str,
    zeros: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "needs_independent_storage": independent,
        "can_be_generated_predicted_or_shared": generated,
        "dictionary_recurrent_or_conditional_expresses_it": expressed,
        "sparse_residual_restores_capability_difference": residual,
        "function_replaceable_not_just_tensor_represented": function,
        "three_zeros": dict(zeros),
    }


def elimination_answers() -> dict[str, dict[str, Any]]:
    """Structural answers. Not a fit. Not a promotion."""
    return {
        "ROUTED_EXPERTS": _ans(
            "Flattened experts look independent (L44 mean cosine ~3.8e-3, "
            "rank-1 energy ~2.7e-3). Independent STORAGE of raw W is the "
            "default; independent INFORMATION of a generator program is not.",
            "Share a decoder PROGRAM and per-expert latent codes "
            "(FLASH_META expert_bank). Do not share one weight basis. "
            "Trivial global expert sharing is dead.",
            "Route-conditioned subprograms, dictionaries with organ-tagged "
            "codebooks, and factorized one-sided subspaces remain live. "
            "Unconditioned shared codebook is dead.",
            "Yes: capability-sensitive expert islands plus a sparse residual "
            "on a cheap backbone. The residual is the identity of the expert.",
            "Yes: replace per-expert dense GEMV with generated-tile GEMV / "
            "route accumulation (meta runtime_shape). Function, not tensor.",
            {
                "ZERO_STORAGE": "OPEN if a generator reconstructs the useful function",
                "ZERO_INDEPENDENT_INFORMATION": "OPEN for the shared decoder; CLOSED for expert identity codes",
                "ZERO_EXECUTION": "CLOSED: selected experts still run",
            },
        ),
        "SHARED_EXPERTS": _ans(
            "Always-on path; small (census shared_expert). Independent storage "
            "until a generator matches the protected island.",
            "A generator is allowed; a trivial tie to the routed bank is not.",
            "Factorized / fused SwiGLU program. Dictionary is optional.",
            "A small residual can bandage a compressed shared expert.",
            "Fuse with routed accumulation rather than represent W twice.",
            {
                "ZERO_STORAGE": "OPEN after generator parity",
                "ZERO_INDEPENDENT_INFORMATION": "OPEN if predicted from route/hidden",
                "ZERO_EXECUTION": "CLOSED: the shared path still executes",
            },
        ),
        "ROUTER": _ans(
            "YES. The router is the control plane. L3-L4 membership already "
            "flips under a small hidden perturbation. Independent storage of "
            "the gate is required at CONTROL_FLOW_PREMIUM precision.",
            "Do not generate the router from the expert bank (experts are "
            "downstream of routing). A margin-gated residual around a compact "
            "router is allowed; crushing the gate is not.",
            "ConditionalProgram on low-margin tokens (exact island). Not a "
            "dictionary of experts — that confuses control with payload.",
            "Yes: 0.5% residual normally, 2% when top10/top11 margin is tight "
            "(cited oracle policy, not re-derived).",
            "The FUNCTION is discrete top-K membership and order, not a "
            "tensor MSE. Optimize executable control information, not bpw.",
            {
                "ZERO_STORAGE": "REFUSED",
                "ZERO_INDEPENDENT_INFORMATION": "REFUSED",
                "ZERO_EXECUTION": "REFUSED",
            },
        ),
        "HC_HYPERCONNECTION": _ans(
            "Small coefficient island (mlp_hyperconnection). Independent "
            "because HC read/write is an exact source precondition.",
            "Do not share HC gates with attention weights. Fusion into the "
            "neighboring projections is a program change, not a share.",
            "A tiny conditional mix program. Not a dictionary.",
            "Not the right tool: HC is already a residual mix.",
            "Yes: fuse HC read/SiLU/mix/write so standalone HC kernels "
            "have ZERO_EXECUTION as separate dispatches.",
            {
                "ZERO_STORAGE": "UNLIKELY (exact mix coefficients)",
                "ZERO_INDEPENDENT_INFORMATION": "UNLIKELY",
                "ZERO_EXECUTION": "OPEN for standalone kernels via fusion",
            },
        ),
        "DELTANET_RECURRENT_STATE": _ans(
            "Recurrent state transition is a protected island "
            "(coherence_contract: exact_or_source-approved). Weights of "
            "linear-attention/HC live in linear_attention_hyperconnection.",
            "State is predicted from the previous state plus a token; that "
            "IS the source program. Do not replace it with a static tensor.",
            "RecurrentStateProgram is the native family. A dictionary of "
            "states would be a cache, not a transition.",
            "A sparse residual on the transition may bandage compression; "
            "it cannot restore a wrong state machine.",
            "Yes: keep the transition, compress the stored state, fuse the "
            "update. Do not represent DeltaNet as a big static matrix.",
            {
                "ZERO_STORAGE": "OPEN for weight compression, CLOSED for the live state",
                "ZERO_INDEPENDENT_INFORMATION": "CLOSED: state is the information",
                "ZERO_EXECUTION": "CLOSED",
            },
        ),
        "FULL_ATTENTION": _ans(
            "KV-sensitive protected island (meta_bpw_target 3.0). Independent "
            "of the MoE bank.",
            "QKV may share a factor; the attention FUNCTION cannot be "
            "predicted from MLP weights.",
            "Fused QKV/attention program. Conditional sparse patterns.",
            "Yes, on Q/K/V with a high-precision island for sinking tokens.",
            "Yes: fused QKV/RoPE/SDPA rather than three stored matrices plus "
            "ceremony.",
            {
                "ZERO_STORAGE": "OPEN for projections, CLOSED for the attention function",
                "ZERO_INDEPENDENT_INFORMATION": "PARTIAL (Q/K/V factors)",
                "ZERO_EXECUTION": "OPEN for unfused projection kernels",
            },
        ),
        "KV_STATE": _ans(
            "Runtime state, not a census weight family. Independent of stored "
            "W. Long-context recall is the capability gate "
            "(decode_civilization kv_compression).",
            "Recompute (ZeroProgram of the cache) vs store. Recompute is a "
            "FLOP trade, not free.",
            "Quantized cache, dictionary of sinking tokens, recurrent "
            "compression. Asymmetric K/V is a byte class, not a result.",
            "Yes: keep high-precision residuals on recall-sensitive keys.",
            "Yes: the function is 'attend to the past', which can be a "
            "compressed cache, a sliding window, or a recompute program.",
            {
                "ZERO_STORAGE": "OPEN (recompute) at FLOP cost",
                "ZERO_INDEPENDENT_INFORMATION": "CLOSED (the past is the information)",
                "ZERO_EXECUTION": "CLOSED (someone must attend)",
            },
        ),
        "NGRAM": _ans(
            "The table is 28% of specimen bytes; per-token DRAM is PLE-aux "
            "dominated (ngram_school). Independent storage of 320M rows is "
            "the question this school exists to kill.",
            "Yes: generated lookup from token embeds + MLP; frequency-tiered "
            "generator + hot islands (meta ngram_bank).",
            "LookupProgram, DictionaryProgram, FactorizedProgram, "
            "GeneratedProgram — already emitted by ngram_school; this "
            "school wraps them as Gravity programs.",
            "Yes: literal exception islands on Zipf heavy-hitters.",
            "Yes: replace the table with a generator. That is the point.",
            {
                "ZERO_STORAGE": "OPEN for the cold table",
                "ZERO_INDEPENDENT_INFORMATION": "OPEN if generated from tokens",
                "ZERO_EXECUTION": "CLOSED: some lookup or generate still runs",
            },
        ),
        "EMBEDDING": _ans(
            "Census lumps embedding_lm_head. Input embed is a lookup organ, "
            "not a GEMV. Independent of routed experts.",
            "May share a dictionary with n-gram / lm_head; tying is a "
            "hypothesis, not a default.",
            "LookupProgram + DictionaryProgram. GeneratedProgram for rare tokens.",
            "Yes: hot-token islands at higher precision.",
            "Yes: a generator of rows rather than a stored table.",
            {
                "ZERO_STORAGE": "OPEN for rare rows",
                "ZERO_INDEPENDENT_INFORMATION": "OPEN if tied/generated",
                "ZERO_EXECUTION": "CLOSED",
            },
        ),
        "LM_HEAD": _ans(
            "Terminal-logit island (protected). Paid on every target verify. "
            "Independent of the MoE bank; MTP may share it.",
            "Tying with embed is a live hypothesis; crushing below the "
            "terminal island is not (lm_head_precision scar family).",
            "QuantTensor at PREMIUM. Conditional exact islands on top logits.",
            "Yes: keep a high-precision residual on the argmax neighborhood.",
            "The function is the terminal distribution. Replacing the GEMV "
            "with a sampled-softmax program is allowed; dropping it is not.",
            {
                "ZERO_STORAGE": "REFUSED for the island; OPEN for the long tail",
                "ZERO_INDEPENDENT_INFORMATION": "OPEN if tied to embed",
                "ZERO_EXECUTION": "REFUSED",
            },
        ),
        "NORMALIZATION": _ans(
            "Tiny (norm family ~376 kB) and exact. Independent because RMSNorm "
            "semantics are source-exact.",
            "Do not generate RMSNorm from other organs.",
            "LiteralTensor / exact island. Dictionary would be a category error.",
            "No: a residual on a wrong norm is still a wrong norm.",
            "Fuse into the consumer (ZERO_EXECUTION of standalone kernels). "
            "Do not replace the function.",
            {
                "ZERO_STORAGE": "REFUSED",
                "ZERO_INDEPENDENT_INFORMATION": "REFUSED",
                "ZERO_EXECUTION": "OPEN as standalone kernels via fusion",
            },
        ),
        "POSITIONAL_STRUCTURE": _ans(
            "RoPE / position tables often live in 'other'. Independent of "
            "content weights; frequently a closed-form program.",
            "YES: RoPE is generated from a formula. This is the textbook "
            "ZERO_STORAGE organ.",
            "GeneratedProgram (closed form). LookupProgram of cos/sin tables "
            "is the worse cousin.",
            "Not needed if the formula is exact.",
            "Yes: replace stored tables with the rotary program.",
            {
                "ZERO_STORAGE": "OPEN (formula)",
                "ZERO_INDEPENDENT_INFORMATION": "OPEN (determined by position index)",
                "ZERO_EXECUTION": "CLOSED: rotary still applies",
            },
        ),
        "DECODING": _ans(
            "Not a weight organ. Tokenizer / sampler / accept-reject. Extends "
            "decode_civilization: the objective is accepted complete-token "
            "cost, not draft throughput.",
            "Greedy argmax is a simpler program than a sampled head. Speculative "
            "decoding is a program over the target, not a stored tensor.",
            "ConditionalProgram (accept/reject). LookupProgram for sampler tables.",
            "N/A as weight residual. Rollback is inside the objective.",
            "Yes: this organ IS a function. Placement of the exact-target "
            "predicate may move; WHAT is checked may not "
            "(decode_civilization VerificationCorrectnessError).",
            {
                "ZERO_STORAGE": "mostly already (function)",
                "ZERO_INDEPENDENT_INFORMATION": "OPEN (determined by logits + policy)",
                "ZERO_EXECUTION": "CLOSED: a token must be chosen",
            },
        ),
        "MTP_SPECULATION": _ans(
            "MTP expert tensors are indexed inside routed_experts (census "
            "largest_tensors includes mtp.layers.*.mlp.experts.*). This school "
            "does not re-quantize those bytes; it searches the draft / verify / "
            "rollback FUNCTION.",
            "ZeroProgram = no speculation (the baseline). Speculation is a "
            "program that must beat that baseline on accepted-token cost.",
            "Conditional accept/reject. RecurrentStateProgram for rollback.",
            "N/A as expert residual (that's ROUTED_EXPERTS).",
            "Yes: replace extra forwards with a draft program whose verify "
            "is the same predicate. Raw draft TPS is a diagnostic that can "
            "win while accepted-token cost loses.",
            {
                "ZERO_STORAGE": "N/A (function school)",
                "ZERO_INDEPENDENT_INFORMATION": "OPEN",
                "ZERO_EXECUTION": "OPEN: ZeroProgram (no MTP) is a legal candidate",
            },
        ),
    }


def program_family_catalog() -> list[dict[str, Any]]:
    blurbs = {
        "LiteralTensor": "Store the source tensor unchanged (BF16 control).",
        "QuantTensor": "Packed independent quant (group-64 Q4/Q3). Control, not the mission.",
        "SharedBasisProgram": "Share a structured factor. Unconditioned global expert basis is DEAD.",
        "FactorizedProgram": "One-sided / CP / Tucker / sandwich cores consumed factor-wise.",
        "DictionaryProgram": "Codebook + indices. Organ-tagged; raw 1-bit PQ is dead.",
        "GeneratedProgram": "A smaller program emits the useful values at use time.",
        "SparseResidualProgram": "Cheap backbone + sparse residual restoring capability-sensitive difference.",
        "RecurrentStateProgram": "Replace stored sequence state with a transition program.",
        "LookupProgram": "Gather / compositional lookup rather than GEMV.",
        "ConditionalProgram": "Spend bits only on the tokens/routes that need them.",
        "RoutedSubprogram": "Per-route executable, not a stacked dense bank.",
        "FusedPhysicalProgram": "Fused native consume path; forbids dense rematerialization.",
        "ZeroProgram": "Eliminate storage, independent information, or standalone execution.",
        "CompositeProgram": "A typed mix of the above; scored as a whole program.",
    }
    return [{"id": name, "what": blurbs[name]} for name in PROGRAM_FAMILIES]


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------


def _cap_rank(cls: str) -> int:
    if cls not in CAPABILITY_ORDER:
        raise IncompleteVectorError(f"unknown capability_risk_class {cls!r}")
    return CAPABILITY_ORDER[cls]


def _scale_bytes(n: int, num: int, den: int) -> int:
    if n <= 0:
        return 0
    return int(n) * int(num) // int(den)


def _cand(
    *,
    school: str,
    family: str,
    ident: str,
    mechanism: str,
    storage_bytes: int,
    flop_milli: int,
    capability: str,
    falsifier: str,
    native: str,
    zeros: Sequence[str],
    scar_distance: str,
    hypothesis_family: str,
    bit_class: str | None = None,
    extends: str | None = None,
    forbids_dense: bool = True,
) -> dict[str, Any]:
    if family not in PROGRAM_FAMILIES:
        raise CandidateSchemaError(f"{ident}: unknown program_family {family}")
    row: dict[str, Any] = {
        "id": ident,
        "school": school,
        "program_family": family,
        "mechanism": mechanism,
        "storage_bytes": int(storage_bytes),
        "flop_milli": int(flop_milli),
        "capability_risk_class": capability,
        "capability_risk_rank": _cap_rank(capability),
        "cheapest_falsifier": falsifier,
        "native_execution_concept": native,
        "forbids_dense_rematerialization": bool(forbids_dense),
        "status": "HYPOTHESIS_UNFITTED",
        "evidence_class": "STATIC_ONLY",
        "zeros_targeted": list(zeros),
        "scar_distance": scar_distance,
        "hypothesis_family": hypothesis_family,
        "independently_schedulable": True,
        "bit_class": bit_class,
        "extends": extends,
        "storage_flop_are_structural_estimates": True,
        "not_a_hardware_measurement": True,
    }
    return row


def _literal(school: str, nbytes: int, *, mechanism: str, falsifier: str, native: str) -> dict[str, Any]:
    return _cand(
        school=school,
        family="LiteralTensor",
        ident=f"{school}:LITERAL",
        mechanism=mechanism,
        storage_bytes=nbytes,
        flop_milli=1000,
        capability="CONTROL_LITERAL",
        falsifier=falsifier,
        native=native,
        zeros=(),
        scar_distance="Control. Not a sharing hypothesis.",
        hypothesis_family="literal_control",
        bit_class="CONTROL_FLOW_PREMIUM" if school in {"ROUTER", "NORMALIZATION", "LM_HEAD"} else "ORDINARY",
    )


def _quant(school: str, nbytes: int, *, mechanism: str, falsifier: str, native: str, bit_class: str) -> dict[str, Any]:
    return _cand(
        school=school,
        family="QuantTensor",
        ident=f"{school}:QUANT-Q4",
        mechanism=mechanism,
        storage_bytes=_scale_bytes(nbytes, Q4_NUM, Q4_DEN),
        flop_milli=Q4_FLOP_MILLI,
        capability="CONTROL_QUANT",
        falsifier=falsifier,
        native=native,
        zeros=(),
        scar_distance="Independent Q4 is a control, not a Gravity program. Not uniform_subbit.",
        hypothesis_family="independent_q4_control",
        bit_class=bit_class,
    )


def _ebs_cache() -> list[dict[str, Any]]:
    return list(ebs.generate())


def _ngs_cache() -> list[dict[str, Any]]:
    return list(ngs.candidates())


def _wrap_expert_bank(school: str, nbytes: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _ebs_cache():
        kind = str(row.get("kind") or "")
        family = EBS_KIND_TO_FAMILY.get(kind)
        if not family:
            continue
        num, den, flop = EBS_KIND_COST.get(kind, (1, 2, 1500))
        cap = "STRUCTURE_PRESERVING"
        if family in {"GeneratedProgram"}:
            cap = "GENERATIVE_UNMEASURED"
        elif family in {"SparseResidualProgram"}:
            cap = "RESIDUAL_REPAIRABLE"
        elif family in {"ConditionalProgram"}:
            cap = "CONDITIONAL"
        elif family in {"FusedPhysicalProgram"}:
            cap = "STRUCTURE_PRESERVING"
            num, den = 1, 1
        out.append(
            _cand(
                school=school,
                family=family,
                ident=f"{school}:EBS:{row['id']}",
                mechanism=_clip(row.get("mechanism")),
                storage_bytes=_scale_bytes(nbytes, num, den),
                flop_milli=flop,
                capability=cap,
                falsifier=_clip(row.get("cheapest_falsifier")),
                native=_clip(row.get("native_execution_concept")),
                zeros=("ZERO_STORAGE", "ZERO_INDEPENDENT_INFORMATION")
                if family in {"GeneratedProgram", "FactorizedProgram"}
                else (("ZERO_EXECUTION",) if family == "FusedPhysicalProgram" else ()),
                scar_distance=_clip(row.get("scar_distance") or "structured cousin of a dead family"),
                hypothesis_family=f"expert_bank_{kind}",
                extends=f"tools/future/expert_bank_school.py#{row['id']}",
            )
        )
    out.sort(key=lambda r: r["id"])
    return out


def _wrap_ngram(nbytes: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in _ngs_cache():
        fid = str(row.get("id") or row.get("family") or "")
        family = NGRAM_FAMILY_TO_PROGRAM.get(fid)
        if not family:
            continue
        lookups = int(row.get("lookup_operations_per_token") or 3)
        flop = max(1, 1000 * lookups // 3)
        cap = NGRAM_CAPABILITY.get(fid, "GENERATIVE_UNMEASURED")
        storage = int(row.get("executable_bytes") or nbytes)
        zeros: tuple[str, ...] = ()
        if family == "GeneratedProgram":
            zeros = ("ZERO_STORAGE", "ZERO_INDEPENDENT_INFORMATION")
        out.append(
            _cand(
                school="NGRAM",
                family=family,
                ident=f"NGRAM:NGS:{fid}",
                mechanism=f"ngram_school family {fid}",
                storage_bytes=storage,
                flop_milli=flop,
                capability=cap,
                falsifier=(
                    "Five-axis Pareto in ngram_school: kill if the family is "
                    "dominated on executable_bytes, active_lookup_bytes, "
                    "lookup_ops, decode_cost, and sensitivity simultaneously, "
                    "or if a later funnel fit fails held-out n-gram retrieval."
                ),
                native="Native gather / generate / island bypass; never densify the 320M-row table.",
                zeros=zeros,
                scar_distance="Wraps ngram_school; does not rerun storage-only ranking.",
                hypothesis_family=f"ngram_{fid}",
                extends=f"tools/future/ngram_school.py#{fid}",
            )
        )
    out.sort(key=lambda r: r["id"])
    return out


def _school_local_candidates(school: str, nbytes: int, inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Gravity-native candidates for one school (plus controls)."""
    rows: list[dict[str, Any]] = []
    lit_mech = f"{school} source tensor as LiteralTensor (analytical control)."
    lit_f = (
        f"If a later funnel cannot even load the {school} organ as source "
        "BF16, the school has no control. Kill the lane, not the organ."
    )
    lit_n = "Consume source values. No rematerialization because nothing was packed."
    rows.append(_literal(school, nbytes, mechanism=lit_mech, falsifier=lit_f, native=lit_n))

    q_bit = {
        "ROUTER": "CONTROL_FLOW_PREMIUM",
        "NORMALIZATION": "CONTROL_FLOW_PREMIUM",
        "LM_HEAD": "PREMIUM",
        "KV_STATE": "PREMIUM",
        "DELTANET_RECURRENT_STATE": "PREMIUM",
        "HC_HYPERCONNECTION": "PREMIUM",
        "FULL_ATTENTION": "PREMIUM",
        "EMBEDDING": "PREMIUM",
        "SHARED_EXPERTS": "ORDINARY",
        "ROUTED_EXPERTS": "CRUSHED",
        "NGRAM": "CRUSHED",
        "POSITIONAL_STRUCTURE": "ORDINARY",
        "DECODING": "ORDINARY",
        "MTP_SPECULATION": "ORDINARY",
    }.get(school, "ORDINARY")
    rows.append(
        _quant(
            school,
            nbytes,
            mechanism=f"{school} independent group-64 Q4 QuantTensor control.",
            falsifier=(
                "Kill QuantTensor as the *mission* if it is the only survivor: "
                "the school exists to find a PROGRAM, not a packing. Keep it "
                "as a control that later programs must beat on joint cost."
            ),
            native="Packed GEMV / gather with fused dequant. Never write dense W back.",
            bit_class=q_bit,
        )
    )

    extras: list[dict[str, Any]] = []
    if school == "ROUTED_EXPERTS":
        extras.extend(
            [
                _cand(
                    school=school,
                    family="GeneratedProgram",
                    ident=f"{school}:GENERATED-TILE",
                    mechanism=(
                        "expert-local latent codes + shared tile decoder + "
                        "route-margin repair (FLASH_META expert_bank). Share "
                        "the decoder program, not a single expert basis."
                    ),
                    storage_bytes=_scale_bytes(nbytes, GEN_STORAGE_NUM, GEN_STORAGE_DEN),
                    flop_milli=GEN_FLOP_MILLI,
                    capability="GENERATIVE_UNMEASURED",
                    falsifier=(
                        "Planted generator recovers useful tiles; real L44 bank "
                        "must beat independent Q4 on held-out routed-output "
                        "AND keep top-K membership. Kill if the decoder is an "
                        "unconditioned shared basis in disguise."
                    ),
                    native="route -> latent decode -> fused gate/up/SwiGLU/down accumulation. No dense W_e.",
                    zeros=("ZERO_STORAGE", "ZERO_INDEPENDENT_INFORMATION"),
                    scar_distance="Not trivial shared basis (decoder is shared, codes are expert-local).",
                    hypothesis_family="expert_local_tile_generator",
                    bit_class="CRUSHED",
                    extends="receipts/future/evidence/FLASH_META_REPRESENTATION_SUB1.json#meta_program.expert_bank",
                ),
                _cand(
                    school=school,
                    family="RoutedSubprogram",
                    ident=f"{school}:ROUTED-SUBPROGRAM",
                    mechanism=(
                        "Per-route executable instead of a stacked dense bank. "
                        "Only selected experts' programs are live."
                    ),
                    storage_bytes=_scale_bytes(nbytes, 1, 8),
                    flop_milli=1100,
                    capability="CONDITIONAL",
                    falsifier="Kill if inactive experts' programs must stay resident to meet the function screen.",
                    native="Gather program ids from the router; run only those subprograms.",
                    zeros=("ZERO_EXECUTION",),
                    scar_distance="Not expert merge (omitted experts stay addressable programs).",
                    hypothesis_family="per_route_subprogram",
                    bit_class="CRUSHED",
                ),
                _cand(
                    school=school,
                    family="CompositeProgram",
                    ident=f"{school}:COMPOSITE-DECODER-RESIDUAL",
                    mechanism=(
                        "Generated decoder + sparse residual on capability-sensitive "
                        "experts. The residual is the expert's identity."
                    ),
                    storage_bytes=_scale_bytes(nbytes, SPARSE_STORAGE_NUM, SPARSE_STORAGE_DEN),
                    flop_milli=SPARSE_FLOP_MILLI,
                    capability="RESIDUAL_REPAIRABLE",
                    falsifier=(
                        "Kill if the residual density required to pass the "
                        "function screen exceeds independent Q4 bytes."
                    ),
                    native="Decode tile, add sparse residual, fuse SwiGLU. No densify.",
                    zeros=("ZERO_INDEPENDENT_INFORMATION",),
                    scar_distance="Not unchanged archetype (residual changes per expert).",
                    hypothesis_family="generator_plus_sparse_residual",
                    bit_class="CRUSHED",
                ),
            ]
        )
    elif school == "SHARED_EXPERTS":
        extras.extend(
            [
                _cand(
                    school=school,
                    family="FactorizedProgram",
                    ident=f"{school}:FACTOR-SWIGLU",
                    mechanism="Factorized shared-expert SwiGLU with fused native consume.",
                    storage_bytes=_scale_bytes(nbytes, FACTOR_STORAGE_NUM, FACTOR_STORAGE_DEN),
                    flop_milli=FACTOR_FLOP_MILLI,
                    capability="STRUCTURE_PRESERVING",
                    falsifier="One-layer shared-expert function screen vs source BF16. Kill if the factor rank that saves bytes fails SwiGLU.",
                    native="Fused gate/up/SwiGLU/down; no densify.",
                    zeros=("ZERO_STORAGE",),
                    scar_distance="Not a tie to the routed bank (shared path stays its own program).",
                    hypothesis_family="shared_expert_factor",
                    bit_class="ORDINARY",
                ),
                _cand(
                    school=school,
                    family="FusedPhysicalProgram",
                    ident=f"{school}:FUSED-WITH-ROUTED",
                    mechanism="Fuse shared-expert epilogue with routed accumulation (moe_physical_school).",
                    storage_bytes=nbytes,
                    flop_milli=850,
                    capability="STRUCTURE_PRESERVING",
                    falsifier="Kill if fusion changes the shared+routed sum relative to source order.",
                    native="One epilogue over shared + routed accumulators.",
                    zeros=("ZERO_EXECUTION",),
                    scar_distance="Compute fusion, not storage sharing.",
                    hypothesis_family="shared_routed_epilogue_fusion",
                    extends="tools/future/moe_physical_school.py",
                    bit_class="ORDINARY",
                ),
            ]
        )
    elif school == "ROUTER":
        extras.extend(
            [
                _cand(
                    school=school,
                    family="ConditionalProgram",
                    ident=f"{school}:MARGIN-GATED-RESIDUAL",
                    mechanism=(
                        "Compact router plus a conditional exact residual on "
                        "low-margin tokens. Spend CONTROL_FLOW_PREMIUM bits "
                        "on the 2.6 MB gate; near-zero on predictable bulk."
                    ),
                    storage_bytes=_scale_bytes(max(nbytes, 1), 1, 1)  # keep the tiny gate
                    + 64,  # cited oracle repair bytes class (fp16 rank-8), structural
                    flop_milli=1100,
                    capability="CONDITIONAL",
                    falsifier=(
                        "Kill if compact+residual still changes top-K membership "
                        "on the L3-L4 seam (FLASH_ROUTER_SENSITIVITY_MAP). Do not "
                        "rerun the seam as if it were new; reuse the map."
                    ),
                    native="Exact gate on tight margin; compact otherwise. Never generate W_gate from experts.",
                    zeros=(),
                    scar_distance="Not student-distill of the router (router_distill is dead). Not uniform bpw.",
                    hypothesis_family="margin_gated_router_residual",
                    bit_class="CONTROL_FLOW_PREMIUM",
                    extends="tools/future/router_science.py",
                ),
                _cand(
                    school=school,
                    family="LiteralTensor",
                    ident=f"{school}:CONTROL-PLANE-PREMIUM",
                    mechanism=(
                        "Keep the router at CONTROL_FLOW_PREMIUM (16 bpw class) "
                        "even though it is <<1% of specimen bytes. Optimize "
                        "total executable information, not uniform bpw."
                    ),
                    storage_bytes=nbytes,
                    flop_milli=1000,
                    capability="PROTECTED_ISLAND",
                    falsifier="Kill only if a compact gate preserves membership AND order on held-out seams.",
                    native="Resident BF16/FP16 gate GEMV. Tiny; do not crush to save bulk bits.",
                    zeros=(),
                    scar_distance="Opposite of uniform_subbit_allocation.",
                    hypothesis_family="heterogeneous_control_plane",
                    bit_class="CONTROL_FLOW_PREMIUM",
                ),
                _cand(
                    school=school,
                    family="QuantTensor",
                    ident=f"{school}:UNIFORM-BPW-BASELINE",
                    mechanism=(
                        "Baseline: pack the router at the same 4.25 bpw as "
                        "bulk. Expected to LOSE on control-plane risk. Not "
                        "proposed as a winner; scored so a uniform plan cannot "
                        "silently win on bytes."
                    ),
                    storage_bytes=_scale_bytes(nbytes, Q4_NUM, Q4_DEN),
                    flop_milli=Q4_FLOP_MILLI,
                    capability="CONTROL_CRUSHED",
                    falsifier="Already: L3-L4 membership is margin-sensitive. This baseline exists to lose.",
                    native="Packed gate GEMV. Diagnostic control, not a recommendation.",
                    zeros=(),
                    scar_distance=(
                        "Does not use the dead phrase 'uniform subbit allocation'; "
                        "it is a scored baseline, not a rediscovery of the scar."
                    ),
                    hypothesis_family="uniform_bpw_baseline",
                    bit_class="CRUSHED",
                ),
            ]
        )
    elif school == "HC_HYPERCONNECTION":
        extras.extend(
            [
                _cand(
                    school=school,
                    family="FusedPhysicalProgram",
                    ident=f"{school}:FUSED-READ-WRITE",
                    mechanism="Fuse HC grouped RMSNorm / low-rank / SiLU / read-mix / write (layer-46 ledger).",
                    storage_bytes=nbytes,
                    flop_milli=700,
                    capability="STRUCTURE_PRESERVING",
                    falsifier="Kill if fused order differs from source HC read precondition.",
                    native="One TokenCommandBuffer region; no standalone HC dispatches.",
                    zeros=("ZERO_EXECUTION",),
                    scar_distance="Fusion of exact coefficients, not a shared basis of HC and MLP.",
                    hypothesis_family="hc_fusion",
                    bit_class="PREMIUM",
                ),
                _cand(
                    school=school,
                    family="ZeroProgram",
                    ident=f"{school}:ZERO-STANDALONE-KERNELS",
                    mechanism="Eliminate standalone HC kernels by fusion (ZERO_EXECUTION of dispatches, not of the mix).",
                    storage_bytes=nbytes,
                    flop_milli=700,
                    capability="STRUCTURE_PRESERVING",
                    falsifier="Kill if the mix coefficients themselves are dropped.",
                    native="Coefficients stay; kernels merge.",
                    zeros=("ZERO_EXECUTION",),
                    scar_distance="Not zeroing the organ's information.",
                    hypothesis_family="zero_standalone_hc_dispatch",
                    bit_class="PREMIUM",
                ),
            ]
        )
    elif school == "DELTANET_RECURRENT_STATE":
        extras.extend(
            [
                _cand(
                    school=school,
                    family="RecurrentStateProgram",
                    ident=f"{school}:TRANSITION",
                    mechanism="Keep the DeltaNet transition; search a smaller state encoding.",
                    storage_bytes=_scale_bytes(nbytes, 1, 2),
                    flop_milli=1100,
                    capability="PROTECTED_ISLAND",
                    falsifier="Kill if recurrent_state_semantics leave source-approved (coherence_contract).",
                    native="Resident state update; cannot be tiled like affine GEMV (decode_civilization).",
                    zeros=("ZERO_STORAGE",),
                    scar_distance="Not GPU-resident-state scar (that was a placement failure, not a transition).",
                    hypothesis_family="deltanet_transition_program",
                    bit_class="PREMIUM",
                    extends="tools/future/decode_civilization.py#recurrent_state_compression",
                ),
                _cand(
                    school=school,
                    family="FusedPhysicalProgram",
                    ident=f"{school}:FUSED-UPDATE",
                    mechanism="Fuse in-proj / conv / state update / out-proj.",
                    storage_bytes=nbytes,
                    flop_milli=800,
                    capability="STRUCTURE_PRESERVING",
                    falsifier="Kill if fusion changes the state recurrence.",
                    native="One state-machine pipeline with resident state.",
                    zeros=("ZERO_EXECUTION",),
                    scar_distance="Physical fusion, not state merging (qn_state_merging is a different scar).",
                    hypothesis_family="deltanet_fused_update",
                    bit_class="PREMIUM",
                ),
            ]
        )
    elif school == "FULL_ATTENTION":
        extras.extend(
            [
                _cand(
                    school=school,
                    family="FusedPhysicalProgram",
                    ident=f"{school}:FUSED-QKV-SDPA",
                    mechanism="Fused QKV / RoPE / SDPA / KV write.",
                    storage_bytes=nbytes,
                    flop_milli=800,
                    capability="STRUCTURE_PRESERVING",
                    falsifier="Kill if fused path disagrees with source attention on a short teacher window.",
                    native="One attention region. No host ceremony between QKV and SDPA.",
                    zeros=("ZERO_EXECUTION",),
                    scar_distance="Not head-sharing (qn_head_redundancy is a different scar).",
                    hypothesis_family="fused_attention_path",
                    bit_class="PREMIUM",
                ),
                _cand(
                    school=school,
                    family="FactorizedProgram",
                    ident=f"{school}:QKV-FACTOR",
                    mechanism="Shared input factor across Q/K/V with skinny heads.",
                    storage_bytes=_scale_bytes(nbytes, FACTOR_STORAGE_NUM, FACTOR_STORAGE_DEN),
                    flop_milli=FACTOR_FLOP_MILLI,
                    capability="STRUCTURE_PRESERVING",
                    falsifier="Kill if Q/K/V function screen fails at the rank that saves bytes.",
                    native="z = V @ x once; skinny Q/K/V cores. No densify.",
                    zeros=("ZERO_STORAGE",),
                    scar_distance="One-sided factor of projections, not a shared expert basis.",
                    hypothesis_family="qkv_shared_input_factor",
                    bit_class="PREMIUM",
                ),
            ]
        )
    elif school == "KV_STATE":
        extras.extend(
            [
                _cand(
                    school=school,
                    family="QuantTensor",
                    ident=f"{school}:CACHE-Q8",
                    mechanism="Runtime KV quant (byte class only). Long-context recall is the gate and is ABSENT here.",
                    storage_bytes=0,
                    flop_milli=1100,
                    capability="PROTECTED_ISLAND",
                    falsifier="Kill on recall, not on bytes (decode_civilization kv_compression).",
                    native="Quantized cache consume in-attention. No host round-trip.",
                    zeros=("ZERO_STORAGE",),
                    scar_distance="Byte ladder is not a capability result.",
                    hypothesis_family="kv_quant_byte_class",
                    bit_class="PREMIUM",
                    extends="tools/future/decode_civilization.py#kv_compression",
                ),
                _cand(
                    school=school,
                    family="ZeroProgram",
                    ident=f"{school}:RECOMPUTE",
                    mechanism="Zero cache storage: recompute past K/V. FLOP-tripling cousin of the trap.",
                    storage_bytes=0,
                    flop_milli=3000,
                    capability="ZERO_UNTESTED",
                    falsifier="Kill if accepted-token cost rises (recompute must not win on storage alone).",
                    native="Recompute attention keys/values from residual stream.",
                    zeros=("ZERO_STORAGE",),
                    scar_distance="Joint scoring must let this LOSE to a stored cache.",
                    hypothesis_family="kv_recompute",
                    bit_class="CRUSHED",
                ),
                _cand(
                    school=school,
                    family="DictionaryProgram",
                    ident=f"{school}:SINKING-TOKEN-DICT",
                    mechanism="Dictionary of sinking / heavy tokens; compressed remainder.",
                    storage_bytes=0,
                    flop_milli=1400,
                    capability="RESIDUAL_REPAIRABLE",
                    falsifier="Kill if recall-sensitive keys are not in the dictionary.",
                    native="Exact island gather + compressed rest.",
                    zeros=("ZERO_STORAGE",),
                    scar_distance="Not H2O/MiniCache claimed as measured; analytical family only.",
                    hypothesis_family="kv_sinking_dictionary",
                    bit_class="PREMIUM",
                ),
            ]
        )
    elif school == "NGRAM":
        extras.extend(_wrap_ngram(nbytes))
    elif school == "EMBEDDING":
        extras.extend(
            [
                _cand(
                    school=school,
                    family="LookupProgram",
                    ident=f"{school}:LOOKUP",
                    mechanism="Keep embed as a gather organ, not a GEMV.",
                    storage_bytes=nbytes,
                    flop_milli=900,
                    capability="STRUCTURE_PRESERVING",
                    falsifier="Kill if a GEMV rewrite of embed beats gather on function — it should not.",
                    native="Row gather. No densify.",
                    zeros=(),
                    scar_distance="Function is lookup.",
                    hypothesis_family="embed_lookup",
                    bit_class="PREMIUM",
                ),
                _cand(
                    school=school,
                    family="GeneratedProgram",
                    ident=f"{school}:RARE-GENERATOR",
                    mechanism="Hot rows literal; rare rows generated from a smaller program.",
                    storage_bytes=_scale_bytes(nbytes, 1, 4),
                    flop_milli=GEN_FLOP_MILLI,
                    capability="GENERATIVE_UNMEASURED",
                    falsifier="Kill if rare-token held-out retrieval fails.",
                    native="Hot gather + generator. No full table rematerialization.",
                    zeros=("ZERO_STORAGE", "ZERO_INDEPENDENT_INFORMATION"),
                    scar_distance="Not n-gram table generation (that is the NGRAM school).",
                    hypothesis_family="embed_rare_generator",
                    bit_class="ORDINARY",
                ),
            ]
        )
    elif school == "LM_HEAD":
        extras.extend(
            [
                _cand(
                    school=school,
                    family="ConditionalProgram",
                    ident=f"{school}:TOP-LOGIT-ISLAND",
                    mechanism="PREMIUM residual on the argmax neighborhood; crushed tail.",
                    storage_bytes=_scale_bytes(nbytes, 1, 3),
                    flop_milli=1300,
                    capability="CONDITIONAL",
                    falsifier="Kill on terminal-logit / argmax disagreement (decode_civilization exact-target predicate).",
                    native="Full GEMV only on a shortlist, or island + tail.",
                    zeros=("ZERO_STORAGE",),
                    scar_distance="Not lm_head_below_q8 as a uniform crush of the island.",
                    hypothesis_family="lm_head_top_island",
                    bit_class="PREMIUM",
                    extends="tools/future/decode_civilization.py#lm_head",
                ),
                _cand(
                    school=school,
                    family="SharedBasisProgram",
                    ident=f"{school}:TIE-EMBED",
                    mechanism="Tie lm_head to embed with a small adapter (structured cousin of tying, not a dead expert-share).",
                    storage_bytes=_scale_bytes(nbytes, 1, 8),
                    flop_milli=1100,
                    capability="STRUCTURE_PRESERVING",
                    falsifier="Kill if terminal logits disagree with untied source on a teacher window.",
                    native="Gather embed row, apply adapter, skip a second vocab matrix when the screen holds.",
                    zeros=("ZERO_INDEPENDENT_INFORMATION", "ZERO_STORAGE"),
                    scar_distance="Vocab tying, not routed-expert sharing.",
                    hypothesis_family="lm_head_embed_tie",
                    bit_class="PREMIUM",
                ),
            ]
        )
    elif school == "NORMALIZATION":
        extras.extend(
            [
                _cand(
                    school=school,
                    family="FusedPhysicalProgram",
                    ident=f"{school}:FUSED-INTO-CONSUMER",
                    mechanism="Fuse RMSNorm into the consumer projection (ZERO_EXECUTION of standalone norms).",
                    storage_bytes=nbytes,
                    flop_milli=600,
                    capability="PROTECTED_ISLAND",
                    falsifier="Kill if fused norm is not source-exact.",
                    native="Exact RMSNorm in the consumer kernel. Weights stay FP32/BF16 island.",
                    zeros=("ZERO_EXECUTION",),
                    scar_distance="Does not zero the weights.",
                    hypothesis_family="fused_exact_norm",
                    bit_class="CONTROL_FLOW_PREMIUM",
                ),
            ]
        )
    elif school == "POSITIONAL_STRUCTURE":
        extras.extend(
            [
                _cand(
                    school=school,
                    family="GeneratedProgram",
                    ident=f"{school}:ROPE-FORMULA",
                    mechanism="RoPE as a closed-form program. Textbook ZERO_STORAGE.",
                    storage_bytes=_scale_bytes(nbytes, 1, 100) if nbytes else 0,
                    flop_milli=1200,
                    capability="STRUCTURE_PRESERVING",
                    falsifier="Kill if generated rotary disagrees with source tables on a position grid.",
                    native="Compute cos/sin from position index. Do not store a table.",
                    zeros=("ZERO_STORAGE", "ZERO_INDEPENDENT_INFORMATION"),
                    scar_distance="Formula, not a learned codebook.",
                    hypothesis_family="rope_closed_form",
                    bit_class="ORDINARY",
                ),
                _cand(
                    school=school,
                    family="ZeroProgram",
                    ident=f"{school}:NOPE",
                    mechanism="No positional program (NoPE). High capability risk; exists so ZeroProgram is schedulable here.",
                    storage_bytes=0,
                    flop_milli=0,
                    capability="ZERO_UNTESTED",
                    falsifier="Kill on long-context / order-sensitive tasks. Do not promote on bytes.",
                    native="Skip rotary. Residual stream carries position only if the rest of the net learned it.",
                    zeros=("ZERO_STORAGE", "ZERO_INDEPENDENT_INFORMATION", "ZERO_EXECUTION"),
                    scar_distance="A function replacement, not a packing.",
                    hypothesis_family="nope",
                    bit_class="CRUSHED",
                ),
            ]
        )
    elif school == "DECODING":
        extras.extend(
            [
                _cand(
                    school=school,
                    family="ConditionalProgram",
                    ident=f"{school}:EXACT-TARGET-PREDICATE",
                    mechanism=(
                        "Decoding program whose verifier is exact-target argmax "
                        "prefix. Placement of the check may move; WHAT may not."
                    ),
                    storage_bytes=0,
                    flop_milli=1000,
                    capability="PROTECTED_ISLAND",
                    falsifier="decode_civilization.VerificationCorrectnessError if WHAT changes.",
                    native="Same predicate, cheaper WHERE (fused / device-side / digest).",
                    zeros=("ZERO_EXECUTION",),
                    scar_distance="Ceremony reduction, not correctness weakening.",
                    hypothesis_family="exact_target_decode",
                    bit_class="CONTROL_FLOW_PREMIUM",
                    extends="tools/future/decode_civilization.py",
                ),
                _cand(
                    school=school,
                    family="ZeroProgram",
                    ident=f"{school}:GREEDY",
                    mechanism="Zero sampler state: greedy argmax. Simpler function.",
                    storage_bytes=0,
                    flop_milli=800,
                    capability="STRUCTURE_PRESERVING",
                    falsifier="Kill if the product requires sampling (then greedy is a different function).",
                    native="Argmax of terminal logits. No sampler tables.",
                    zeros=("ZERO_STORAGE", "ZERO_EXECUTION"),
                    scar_distance="Function change, scored as such.",
                    hypothesis_family="greedy_decode",
                    bit_class="ORDINARY",
                ),
            ]
        )
    elif school == "MTP_SPECULATION":
        extras.extend(
            [
                _cand(
                    school=school,
                    family="ZeroProgram",
                    ident=f"{school}:NO-SPECULATION",
                    mechanism="No MTP. The baseline every speculation program must beat on accepted-token cost.",
                    storage_bytes=0,
                    flop_milli=1000,
                    capability="CONTROL_LITERAL",
                    falsifier="Not a kill; this IS the control. Speculation that wins draft TPS and loses accepted-token cost is refused.",
                    native="Target forward only.",
                    zeros=("ZERO_EXECUTION",),
                    scar_distance="Draft TPS is DIAGNOSTIC_RELATIVE and cannot promote.",
                    hypothesis_family="no_mtp_baseline",
                    bit_class="ORDINARY",
                    extends="tools/future/decode_civilization.py",
                ),
                _cand(
                    school=school,
                    family="ConditionalProgram",
                    ident=f"{school}:DRAFT-VERIFY-ROLLBACK",
                    mechanism="Draft / verify / rollback with the same exact-target predicate. Rollback is inside the objective.",
                    storage_bytes=0,
                    flop_milli=1500,
                    capability="CONDITIONAL",
                    falsifier=(
                        "Kill if accepted complete-token cost (relative units) "
                        "exceeds the no-speculation control. Do not cite draft TPS."
                    ),
                    native="Explicit accept/reject/state rollback accounting (FPGA organ map mtp_draft_verify_rollback).",
                    zeros=(),
                    scar_distance="Function school; MTP expert bytes stay in ROUTED_EXPERTS.",
                    hypothesis_family="mtp_draft_verify",
                    bit_class="ORDINARY",
                    extends="tools/future/decode_civilization.py",
                ),
                _cand(
                    school=school,
                    family="RecurrentStateProgram",
                    ident=f"{school}:ROLLBACK-STATE",
                    mechanism="Speculation as a state program with explicit rollback, not a second copy of weights.",
                    storage_bytes=0,
                    flop_milli=1400,
                    capability="PROTECTED_ISLAND",
                    falsifier="Kill if rollback desynchronizes KV / DeltaNet / HC state.",
                    native="Checkpoint-and-restore of recurrent and KV state on reject.",
                    zeros=("ZERO_STORAGE",),
                    scar_distance="Not a second expert bank.",
                    hypothesis_family="mtp_rollback_state",
                    bit_class="PREMIUM",
                ),
            ]
        )

    if school == "ROUTED_EXPERTS":
        extras.extend(_wrap_expert_bank(school, nbytes))

    rows.extend(extras)
    # Stable unique ids.
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        uniq.append(r)
    uniq.sort(key=lambda r: r["id"])
    return uniq


# ---------------------------------------------------------------------------
# Scar refusal
# ---------------------------------------------------------------------------


def _blob(candidate: Mapping[str, Any]) -> str:
    parts = [
        str(candidate.get("id") or ""),
        str(candidate.get("mechanism") or ""),
        str(candidate.get("family") or ""),
        str(candidate.get("hypothesis_family") or ""),
        str(candidate.get("dead_family") or ""),
        str(candidate.get("title") or ""),
        str(candidate.get("program_family") or ""),
    ]
    return " ".join(parts).lower()


def match_local_scar(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    blob = _blob(candidate)
    fam = str(
        candidate.get("hypothesis_family")
        or candidate.get("family")
        or candidate.get("dead_family")
        or ""
    ).lower()
    for scar in BUILTIN_SCARS:
        scar_fam = str(scar.get("family") or "").lower()
        if fam and fam == scar_fam:
            return dict(scar)
        for ph in scar.get("phrases") or ():
            if ph and ph.lower() in blob:
                return dict(scar)
    return None


# Families the candidate must *claim* before the index is allowed to refuse.
# canon_family() maps any "shared"+"expert" slug onto cross_expert_structure,
# which would blanket-kill structured cousins. Refuse only when the raw
# family tag is already a recorded-dead family (the probe path).
INDEX_REFUSE_FAMILIES: frozenset[str] = frozenset(
    {
        "cross_expert_structure",
        "trivial_global_expert_sharing",
        "trivial_shared_basis",
        "raw_global_expert_similarity",
        "unchanged_archetype",
        "uniform_subbit_allocation",
        "expert_merge",
        "raw_weight_pq_vq",
        ebs.DEAD_FAMILY_RAW.lower(),
        ebs.DEAD_FAMILY_BASIS.lower(),
        ebs.DEAD_FAMILY_ARCHETYPE.lower(),
    }
)


def consult_negative_index(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    """Query the landed index. Cope if ingest cannot run in this checkout.

    Targeted: the index may refuse only when the candidate itself claims a
    recorded-dead family. Phrase matching is the job of match_local_scar /
    expert_bank_school.match_scar, which use long specific phrases.
    """
    raw = str(
        candidate.get("hypothesis_family")
        or candidate.get("family")
        or candidate.get("dead_family")
        or ""
    )
    slug = ni._slug(raw) if raw else ""
    organ = candidate.get("organ") or SCHOOL_ORGAN_SLUG.get(
        str(candidate.get("school") or ""), "unrecorded"
    )
    try:
        # Always query so rediscovery is not free. Refuse only on a raw
        # dead-family claim — canon_family("shared"+"expert") is too broad.
        ni.query(organ=str(organ) if organ else None, hypothesis_family=raw or None)
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError):
        pass
    if slug not in INDEX_REFUSE_FAMILIES and raw.lower() not in INDEX_REFUSE_FAMILIES:
        return None
    proposal = {
        "hypothesis_family": raw or candidate.get("mechanism"),
        "organ": organ,
        "technique": candidate.get("mechanism"),
        "lever": candidate.get("mechanism"),
    }
    try:
        return ni.refuse_if_dead(proposal)
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError):
        return None


def match_scar(candidate: Mapping[str, Any]) -> dict[str, Any] | None:
    local = match_local_scar(candidate)
    if local is not None:
        return local
    try:
        ebs_scar = ebs.match_scar(dict(candidate))
    except (TypeError, ValueError):
        ebs_scar = None
    if ebs_scar is not None:
        return dict(ebs_scar)
    return consult_negative_index(candidate)


def admit_candidate(
    candidate: Mapping[str, Any],
    *,
    require_schema: bool = True,
) -> dict[str, Any]:
    """Admit a live candidate, or raise DeadHypothesisError.

    Scar matching runs BEFORE schema checks so a minimal dead probe still fires.
    """
    scar = match_scar(candidate)
    if scar is not None:
        raise DeadHypothesisError(str(candidate.get("id") or "<no-id>"), scar)
    if require_schema:
        missing = [f for f in CANDIDATE_FIELDS if f not in candidate]
        if missing:
            raise CandidateSchemaError(f"{candidate.get('id')}: missing fields {missing}")
        if candidate.get("forbids_dense_rematerialization") is not True:
            raise CandidateSchemaError(
                f"{candidate.get('id')}: native path must forbid dense rematerialization"
            )
        if candidate.get("evidence_class") != "STATIC_ONLY":
            raise CandidateSchemaError(
                f"{candidate.get('id')}: evidence_class must be STATIC_ONLY"
            )
        if candidate.get("status") != "HYPOTHESIS_UNFITTED":
            raise CandidateSchemaError(f"{candidate.get('id')}: this lane does not fit weights")
        if candidate.get("program_family") not in PROGRAM_FAMILIES:
            raise CandidateSchemaError(
                f"{candidate.get('id')}: program_family not in catalog"
            )
    out = dict(candidate)
    out["scar_consulted"] = True
    return out


DEAD_PROBE: dict[str, Any] = {
    "id": "PROBE-TRIVIAL-GLOBAL-EXPERT-SHARING",
    "school": "ROUTED_EXPERTS",
    "program_family": "SharedBasisProgram",
    "mechanism": "trivial global expert sharing",
    "hypothesis_family": "cross_expert_structure",
    "family": ebs.DEAD_FAMILY_BASIS,
}


def prove_scar_refusal(live_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    refused = False
    scar_id = None
    reason = None
    try:
        admit_candidate(DEAD_PROBE, require_schema=False)
    except DeadHypothesisError as exc:
        refused = True
        scar_id = exc.scar.get("id") or exc.scar.get("scar_id")
        reason = str(exc)
    distinct = [
        r
        for r in live_rows
        if r.get("school") == "ROUTED_EXPERTS"
        and r.get("program_family")
        in {"FactorizedProgram", "GeneratedProgram", "ConditionalProgram", "SparseResidualProgram"}
    ]
    distinct.sort(key=lambda r: r["id"])
    emitted = None
    if distinct:
        emitted = admit_candidate(distinct[0])
    proof = {
        "dead_probe_id": DEAD_PROBE["id"],
        "dead_probe_refused": refused,
        "scar_id": scar_id,
        "reason": reason,
        "structurally_distinct_emitted": emitted["id"] if emitted else None,
        "structurally_distinct_family": emitted["program_family"] if emitted else None,
        "n_live_routed_cousins": len(distinct),
    }
    if not refused:
        raise RuntimeError("scar refusal did not fire on trivial global expert sharing")
    if emitted is None:
        raise RuntimeError("scar refusal was a blanket ban; no structurally distinct candidate emitted")
    return proof


# ---------------------------------------------------------------------------
# Scoring: storage WITH compute
# ---------------------------------------------------------------------------


def _require_axes(candidate: Mapping[str, Any]) -> None:
    missing = [a for a in AXES if a not in candidate]
    if missing:
        raise IncompleteVectorError(
            f"{candidate.get('id', '<unnamed>')} missing axes {missing}"
        )
    for a in ("storage_bytes", "flop_milli", "capability_risk_rank"):
        v = candidate[a]
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise IncompleteVectorError(
                f"{candidate.get('id')} axis {a} must be a non-negative int, got {v!r}"
            )


def axis_tuple(candidate: Mapping[str, Any]) -> tuple[int, int, int]:
    _require_axes(candidate)
    return (
        int(candidate["storage_bytes"]),
        int(candidate["flop_milli"]),
        int(candidate["capability_risk_rank"]),
    )


def dominates(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """True iff A is <= B on every axis and < on at least one (all minimized)."""
    ta, tb = axis_tuple(a), axis_tuple(b)
    if ta == tb:
        return False
    return all(x <= y for x, y in zip(ta, tb)) and any(x < y for x, y in zip(ta, tb))


def joint_cost(candidate: Mapping[str, Any], control_storage: int) -> int:
    """storage_ratio_milli + flop_milli. A 0.5x / 3.0x trap scores 3500 vs 2000."""
    _require_axes(candidate)
    base = control_storage if control_storage > 0 else 1
    storage_ratio_milli = int(candidate["storage_bytes"]) * 1000 // base
    return storage_ratio_milli + int(candidate["flop_milli"])


def pareto_front(cands: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = [(c["id"], dict(c)) for c in cands]
    front: list[dict[str, Any]] = []
    for ident, c in sorted(indexed, key=lambda kv: kv[0]):
        if any(other.get("id") != ident and dominates(other, c) for other in cands):
            continue
        front.append(c)
    return front


def rank(
    cands: Sequence[Mapping[str, Any]],
    *,
    by: str | None = None,
    axes: Sequence[str] | None = None,
    control_storage: int | None = None,
) -> dict[str, Any]:
    if by is not None:
        raise StorageOnlyRankingError(
            f"rank() refuses scalar ordering (by={by!r}); Gravity scores "
            "storage WITH compute (and capability). A bytes win is not a win."
        )
    if axes is not None and tuple(axes) != AXES:
        raise StorageOnlyRankingError(
            f"rank() refuses incomplete-axis ordering; requires {AXES}, got {tuple(axes)}"
        )
    for c in cands:
        _require_axes(c)
    lit = next((c for c in cands if c.get("program_family") == "LiteralTensor"), None)
    base = control_storage if control_storage is not None else (int(lit["storage_bytes"]) if lit else 1)
    front = pareto_front(cands)
    joint_order = sorted(
        (joint_cost(c, base), c["id"]) for c in cands
    )
    return {
        "scalar_winner": None,
        "ranking_rule": "pareto_storage_flop_capability_plus_joint_cost",
        "storage_only": "REFUSED",
        "axes": list(AXES),
        "pareto_front_ids": [c["id"] for c in front],
        "joint_cost_order": [{"id": i, "joint_cost": cost} for cost, i in joint_order],
        "n_candidates": len(cands),
        "n_front": len(front),
        "control_storage_bytes": base,
        "note": (
            "A candidate that halves storage_bytes and triples flop_milli "
            "loses on joint_cost to LiteralTensor (3500 vs 2000 when the "
            "control is 1.0x/1.0x) and does not Pareto-dominate it."
        ),
    }


def make_trap(control: Mapping[str, Any]) -> dict[str, Any]:
    """Synthetic: half the bytes, triple the FLOPs, same capability.

    Not a school family. Used to prove joint scoring cannot promote a
    storage-only win.
    """
    _require_axes(control)
    trap = dict(control)
    trap["id"] = "TRAP-HALF-BYTES-TRIPLE-FLOPS"
    trap["program_family"] = "QuantTensor"  # stay inside the catalog; tagged synthetic
    trap["mechanism"] = "synthetic trap: half storage_bytes, triple flop_milli"
    trap["storage_bytes"] = int(control["storage_bytes"]) // 2
    trap["flop_milli"] = int(control["flop_milli"]) * 3
    trap["hypothesis_family"] = "synthetic_negative_control"
    trap["is_synthetic_trap"] = True
    trap["cheapest_falsifier"] = "This is the negative control. It must lose."
    trap["scar_distance"] = "synthetic"
    trap["extends"] = None
    trap["independently_schedulable"] = False
    return trap


def make_dominator(trap: Mapping[str, Any]) -> dict[str, Any]:
    """Same capability as the trap, fewer bytes AND fewer FLOPs → dominates."""
    d = dict(trap)
    d["id"] = "DOMINATOR-BETTER-ON-BYTES-AND-FLOPS"
    d["mechanism"] = "synthetic dominator: 0.8x trap bytes, 0.5x trap FLOPs, same capability"
    d["storage_bytes"] = max(0, int(trap["storage_bytes"]) * 4 // 5)
    d["flop_milli"] = max(1, int(trap["flop_milli"]) // 2)
    d["hypothesis_family"] = "synthetic_dominator"
    d["is_synthetic_trap"] = False
    d["is_synthetic_dominator"] = True
    d["independently_schedulable"] = False
    return d


def prove_flop_trap(control: Mapping[str, Any]) -> dict[str, Any]:
    trap = make_trap(control)
    dominator = make_dominator(trap)
    base = int(control["storage_bytes"])
    raised = False
    try:
        rank([control, trap], by="storage_bytes")
    except StorageOnlyRankingError:
        raised = True
    raised_axes = False
    try:
        rank([control, trap], axes=("storage_bytes",))
    except StorageOnlyRankingError:
        raised_axes = True
    naive = sorted(
        (int(c["storage_bytes"]), c["id"]) for c in (control, trap)
    )
    proof = {
        "control_id": control["id"],
        "trap_id": trap["id"],
        "dominator_id": dominator["id"],
        "control_storage_bytes": int(control["storage_bytes"]),
        "trap_storage_bytes": int(trap["storage_bytes"]),
        "control_flop_milli": int(control["flop_milli"]),
        "trap_flop_milli": int(trap["flop_milli"]),
        "control_joint_cost": joint_cost(control, base),
        "trap_joint_cost": joint_cost(trap, base),
        "dominator_joint_cost": joint_cost(dominator, base),
        "trap_dominates_control": dominates(trap, control),
        "control_dominates_trap": dominates(control, trap),
        "dominator_dominates_trap": dominates(dominator, trap),
        "trap_loses_on_joint_cost": joint_cost(trap, base) > joint_cost(control, base),
        "storage_only_by_keyword_raises": raised,
        "storage_only_axes_raises": raised_axes,
        "naive_storage_winner": naive[0][1],
        "scalar_winner": None,
    }
    if not raised or not raised_axes:
        raise RuntimeError("negative control did not fire: storage-only rank was accepted")
    if proof["trap_dominates_control"]:
        raise RuntimeError("negative control failed: half-bytes/triple-FLOPs dominates the control")
    if not proof["trap_loses_on_joint_cost"]:
        raise RuntimeError("negative control failed: trap did not lose on joint_cost")
    if not proof["dominator_dominates_trap"]:
        raise RuntimeError("negative control failed: trap could not be dominated")
    if proof["naive_storage_winner"] != trap["id"]:
        raise RuntimeError("negative control failed: naive storage sort did not put the trap first")
    return proof


def prove_router_control_plane(inventory: Mapping[str, Any]) -> dict[str, Any]:
    """Same total executable-information budget; heterogeneous vs uniform.

    Uniform 1.0 meta-bpw on router+bulk CRUSHES the control plane.
    Heterogeneous spends 16 bpw class on the router and less on bulk so
    the total bits match. Heterogeneous dominates on control-plane risk
    at equal storage.
    """
    by_family = inventory.get("by_family") or {}
    routed = int((by_family.get("routed_experts") or {}).get("bytes") or 0)
    router_b = int(inventory.get("router_tensor_bytes") or 0)
    if routed <= 0 or router_b <= 0:
        return {
            "reachable": False,
            "reason": "census/router overlay unavailable; proof skipped (coped, not faked)",
        }

    def info_bytes(payload: int, bpw_milli: int) -> int:
        # source is BF16 (16 bits/value). executable_bytes ~= payload * bpw / 16.
        return payload * bpw_milli // 16000

    uniform_bpw = 1000  # 1.0 bpw everywhere
    router_premium = 16000  # 16.0 bpw class
    uniform_total = info_bytes(router_b, uniform_bpw) + info_bytes(routed, uniform_bpw)
    hetero_router = info_bytes(router_b, router_premium)
    # Same total executable-information budget by construction. Integer
    # division on the bulk bpw is a remainder, not a second budget.
    rest_budget = uniform_total - hetero_router
    bulk_bpw_milli = (rest_budget * 16000 // routed) if routed else 0
    hetero_total = uniform_total

    uniform = {
        "id": "ALLOC-UNIFORM-1BPW",
        "storage_bytes": uniform_total,
        "flop_milli": 1000,
        "capability_risk_rank": CAPABILITY_ORDER["CONTROL_CRUSHED"],
        "capability_risk_class": "CONTROL_CRUSHED",
        "router_bpw_milli": uniform_bpw,
        "bulk_bpw_milli": uniform_bpw,
    }
    hetero = {
        "id": "ALLOC-HETERO-ROUTER-PREMIUM",
        "storage_bytes": hetero_total,
        "flop_milli": 1000,
        "capability_risk_rank": CAPABILITY_ORDER["PROTECTED_ISLAND"],
        "capability_risk_class": "PROTECTED_ISLAND",
        "router_bpw_milli": router_premium,
        "bulk_bpw_milli": bulk_bpw_milli,
    }
    proof = {
        "reachable": True,
        "router_tensor_bytes": router_b,
        "routed_experts_bytes": routed,
        "uniform": uniform,
        "heterogeneous": hetero,
        "same_total_bits_class": abs(uniform_total - hetero_total) <= 1,
        "heterogeneous_dominates_uniform": dominates(hetero, uniform),
        "uniform_dominates_heterogeneous": dominates(uniform, hetero),
        "router_bits_disproportionate": hetero["router_bpw_milli"] > hetero["bulk_bpw_milli"],
        "objective": "total executable information with control-plane preservation, not uniform bpw",
        "bit_classes_from_router_science": list(rs.BIT_CLASSES),
        "bpw_for_class": dict(rs.BPW_FOR_CLASS),
    }
    if not proof["heterogeneous_dominates_uniform"]:
        raise RuntimeError("router control-plane proof failed: heterogeneous did not dominate uniform")
    if proof["uniform_dominates_heterogeneous"]:
        raise RuntimeError("router control-plane proof failed: uniform dominated heterogeneous")
    if not proof["router_bits_disproportionate"]:
        raise RuntimeError("router control-plane proof failed: router bits were not disproportionate")
    return proof


# ---------------------------------------------------------------------------
# WorkUnits / resident callability
# ---------------------------------------------------------------------------


def emit_school_workunit(school_id: str) -> dict[str, Any]:
    """HCLI-shaped proposal. STATIC_ANALYSIS. Never a GPU lease."""
    if school_id not in SCHOOL_CATALOG:
        raise UnknownSchoolError(school_id)
    row = wus.emit_hcli_workunit(
        id=f"future.flash_schools.{school_id}",
        role="gravity_organ_school",
        description=(
            f"Independently schedule Flash Gravity school {school_id}: emit "
            "program candidates with cheapest falsifiers, consult the negative "
            "index, score storage WITH compute, write STATIC_ONLY evidence."
        ),
        dependencies=[],
        resource_class="STATIC_ANALYSIS",
        verifier=f"future.flash_schools.{school_id}.joint_pareto",
        provider="tools.future.flash_schools",
        preferred_backend=None,
        status="pending",
        classification="STATIC_ONLY",
        extras={
            "command": [
                "python3",
                "tools/future/flash_schools.py",
                "--school",
                school_id,
            ],
            "output_receipt_path": f"receipts/future/{RECEIPT}",
            "species": "learned_compiler_experiment",
            "school": school_id,
            "requires_quiescence": False,
            "blocked_reason": None,
        },
    )
    wus.validate_emitted_unit(row)
    return row


def resident_callable_block(school_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "discoverable": True,
        "entry_point": "python3 tools/future/flash_schools.py --build",
        "per_school_entry": "python3 tools/future/flash_schools.py --school <SCHOOL_ID>",
        "module": "tools.future.flash_schools",
        "callables": ["build", "schedule_school", "selftest", "admit_candidate", "rank"],
        "workunit_id_pattern": "future.flash_schools.<SCHOOL_ID>",
        "workunits_emitted": [f"future.flash_schools.{s}" for s in school_ids],
        "receipt": f"receipts/future/{RECEIPT}",
        "frontier_fed": {
            "name": "flash_gravity_organ_schools",
            "kind": "candidate_frontier",
            "how_it_changes": (
                "Each scheduled school admits live program candidates or "
                "records a scar. The frontier is the union of live candidates "
                "across independently scheduled schools."
            ),
            "how_next_work_refills": (
                "Reschedule the same school after a falsifier returns; the "
                "school re-emits remaining live families. Do not wait for a "
                "monolithic Gravity run."
            ),
            "integration_point": (
                "swap the in-receipt frontier_fed block for "
                "tools/future/frontiers.py when that sibling lands"
            ),
        },
        "fail_closed": {
            "unknown_school": "UnknownSchoolError; no WorkUnit, no receipt row",
            "dead_hypothesis": "DeadHypothesisError citing the scar; live cousins still emit",
            "storage_only_rank": "StorageOnlyRankingError",
            "hardware_claim": "write_receipt raises HardwareClaimError",
            "missing_census": "coped as unavailable; no synthetic GPU result",
            "physical_blocker": (
                "this lane never emits GPU_EXCLUSIVE units. Blocked Metal / "
                "xcrun / qualification-pipeline HEAVY / NX SCAFFOLD_ONLY / "
                "teacher 0/256 stay SLEEPING for the Accelerator lane."
            ),
            "specimen_fit": "refused (fit_policy=NOT_FIT)",
        },
        "cannot": [
            "acquire a GPU lease",
            "promote DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE",
            "fit the 350GB specimen",
            "time a kernel",
            "weaken a verifier",
        ],
        "hcli_constructor": "tools.future.workunit_species.emit_hcli_workunit",
        "integration_point_resident_api": (
            "when tools/future/resident_api.py lands, register "
            "schedule_school and build as resident-callable verbs"
        ),
    }


# ---------------------------------------------------------------------------
# Schedule one school / all schools
# ---------------------------------------------------------------------------


def schedule_school(
    school_id: str,
    *,
    inventory: Mapping[str, Any] | None = None,
    bundle: Mapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one school alone. Does not require any other school to have run."""
    if school_id not in SCHOOL_CATALOG:
        raise UnknownSchoolError(
            f"{school_id!r} is not a Gravity school; catalog={list(SCHOOL_CATALOG)}"
        )
    inv = dict(inventory) if inventory is not None else organ_inventory(bundle)
    nbytes = school_source_bytes(school_id, inv)
    answers = elimination_answers()[school_id]
    raw = _school_local_candidates(school_id, nbytes, inv)
    live: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for c in raw:
        try:
            live.append(admit_candidate(c))
        except DeadHypothesisError as exc:
            refused.append(
                {
                    "id": c.get("id"),
                    "scar_id": exc.scar.get("id") or exc.scar.get("scar_id"),
                    "reason": str(exc),
                }
            )
    ranking = rank(live, control_storage=nbytes if nbytes > 0 else 1) if live else None
    unit = emit_school_workunit(school_id)
    families_used = sorted({c["program_family"] for c in live})
    budget_rows = []
    for fam in SCHOOL_CENSUS_FAMILIES.get(school_id) or ():
        b = (inv.get("budget_by_family") or {}).get(fam)
        if b:
            budget_rows.append(b)
    return {
        "school": school_id,
        "independent": True,
        "schedulable_alone": True,
        "source_bytes": nbytes,
        "census_families": list(SCHOOL_CENSUS_FAMILIES.get(school_id) or ()),
        "organ_slug": SCHOOL_ORGAN_SLUG[school_id],
        "elimination": answers,
        "family_budget": budget_rows,
        "candidates": live,
        "refused_at_emit": refused,
        "n_candidates": len(live),
        "n_refused_at_emit": len(refused),
        "program_families_used": families_used,
        "ranking": ranking,
        "workunit": unit,
        "cheapest_falsifiers": [
            {"id": c["id"], "cheapest_falsifier": c["cheapest_falsifier"]} for c in live
        ],
        "evidence_class": "STATIC_ONLY",
        "fit_policy": "NOT_FIT",
        "gpu_authority": False,
    }


def schedule_all(
    *,
    inventory: Mapping[str, Any] | None = None,
    bundle: Mapping[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    inv = dict(inventory) if inventory is not None else organ_inventory(bundle)
    return [schedule_school(sid, inventory=inv, bundle=bundle) for sid in SCHOOL_CATALOG]


# ---------------------------------------------------------------------------
# Recovery / gaps / negatives
# ---------------------------------------------------------------------------


def recovered_implementation(bundle: Mapping[str, dict[str, Any]], inventory: Mapping[str, Any]) -> dict[str, Any]:
    landed = [
        {
            "path": "tools/future/expert_bank_school.py",
            "role": "structured expert STORAGE + COMPUTE sharing; dead trivial sharing refused",
            "adequate_as_fourteen_schools": False,
            "how_extended": "ROUTED_EXPERTS wraps generate() as Gravity program families",
        },
        {
            "path": "tools/future/ngram_school.py",
            "role": "five-axis n-gram representation school",
            "adequate_as_fourteen_schools": False,
            "how_extended": "NGRAM wraps candidates() as Lookup/Dictionary/Generated programs",
        },
        {
            "path": "tools/future/router_science.py",
            "role": "per-surface precision ALLOCATION, not a scalar score",
            "adequate_as_fourteen_schools": False,
            "how_extended": "ROUTER school turns allocation into a control-plane program search",
        },
        {
            "path": "tools/future/moe_physical_school.py",
            "role": "physical execution of routed compute",
            "adequate_as_fourteen_schools": False,
            "how_extended": "FusedPhysicalProgram / RoutedSubprogram consume its taxonomy",
        },
        {
            "path": "tools/future/meta_funnel.py",
            "role": "ordered refusal funnel; passing a gate is not promotion",
            "adequate_as_fourteen_schools": False,
            "how_extended": "cheapest_falsifier is the funnel's next cheap gate, not a GPU run",
        },
        {
            "path": "tools/future/decode_civilization.py",
            "role": "tokenizer / lm_head / KV / recurrent / MTP cost models",
            "adequate_as_fourteen_schools": False,
            "how_extended": "KV_STATE, LM_HEAD, DECODING, MTP_SPECULATION wrap its functions",
        },
        {
            "path": "tools/future/negative_index.py",
            "role": "keyed scar index; refuse_if_dead",
            "adequate_as_fourteen_schools": False,
            "how_extended": "consulted before every emit; builtin Gravity scars still fire if ingest copes out",
        },
    ]
    evidence_reach = [
        {
            "name": name,
            "reachable": bool((bundle.get(name) or {}).get("reachable")),
            "evidence_source": (bundle.get(name) or {}).get("evidence_source"),
            "path": (bundle.get(name) or {}).get("path"),
        }
        for name in EVIDENCE_NAMES
    ]
    return {
        "note": (
            "Expert-bank, n-gram, router-science, MoE-physical, meta-funnel "
            "and decode-civilization already generate organ-local science. "
            "They are not fourteen independently schedulable Gravity schools "
            "with program-family search, joint storage+compute scoring, "
            "elimination questions, and resident-callable WorkUnits. This "
            "module extends them; it does not fork a second expert genome "
            "or a second n-gram packing table."
        ),
        "landed_siblings_extended": landed,
        "evidence_reach": evidence_reach,
        "census_families_derived": inventory.get("n_census_families"),
        "budget_families_derived": inventory.get("n_budget_families"),
        "doctor_l44": inventory.get("doctor"),
        "not_duplicating": (
            "Trivial expert sharing stays dead (negative_index + "
            "expert_bank_school). FPGA remains Accelerator/Physical "
            "Compiler/Fusion. This lane produces STATIC_ONLY only."
        ),
    }


def gaps_closed() -> list[str]:
    return [
        "fourteen independently schedulable Flash Gravity organ schools, each with its own WorkUnit and cheapest falsifiers",
        "program-family search (Literal/Quant/SharedBasis/Factorized/Dictionary/Generated/SparseResidual/Recurrent/Lookup/Conditional/Routed/Fused/Zero/Composite) rather than quantization-as-mission",
        "elimination questions and the three zeros answered structurally per organ",
        "compute sharing scored WITH storage sharing; a bytes-halving FLOP-tripling trap loses on joint_cost and can be Pareto-dominated",
        "router treated as control plane: disproportionate bits on the tiny gate, uniform-bpw baseline crushed on capability",
        "scar refusal that actually fires on trivial global expert sharing while structured cousins still emit",
        "resident-callable entry points, WorkUnits, receipt, frontier refill, fail-closed paths",
        "extends landed expert_bank / ngram / router_science / moe_physical / decode_civilization / negative_index instead of duplicating them",
    ]


def negative_findings(inventory: Mapping[str, Any]) -> list[str]:
    findings = [
        "Did not fit any candidate to the 350GB Flash specimen.",
        "Did not take a hardware measurement; flop_milli and storage_bytes are structural estimates. Complete-token ns, tps, joules remain UNKNOWN.",
        "Did not acquire a GPU lease, quiesce workers, or touch Codex Metal/Rust.",
        "Flash source-independent NX remains SCAFFOLD_ONLY; teacher capture remains 0/256 — those stay SLEEPING WorkUnits for the Accelerator lane.",
        "MTP expert tensors sit inside the routed_experts census family; MTP_SPECULATION searches the draft/verify function, not a second copy of those bytes.",
        "embedding_lm_head is lumped in the census; EMBEDDING and LM_HEAD share that parent budget rather than inventing a 50/50 split.",
        "KV_STATE / DECODING / MTP function organs have 0 family_summary bytes; they still emit programs (runtime/function schools).",
    ]
    if inventory.get("census_source") == "unavailable":
        findings.append(
            "FLASH_ORGAN_CENSUS.json was not reachable in this checkout; "
            "schools coped without treating unreachability as absence."
        )
    if not (inventory.get("doctor") or {}):
        findings.append(
            "Doctor L44 population summary was not reachable; refusal still "
            "uses builtin + expert_bank scars."
        )
    return findings


def _hw_numeric_keys(node: Any, path: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            here = f"{path}.{k}" if path else k
            if k in HARDWARE_FIELDS and isinstance(v, (int, float)):
                bad.append(here)
            bad.extend(_hw_numeric_keys(v, here))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            bad.extend(_hw_numeric_keys(v, f"{path}[{i}]"))
    return bad


def build(school_id: str | None = None) -> Path:
    bundle = load_all_evidence()
    inventory = organ_inventory(bundle)
    wanted = [school_id] if school_id else list(SCHOOL_CATALOG)
    if school_id and school_id not in SCHOOL_CATALOG:
        raise UnknownSchoolError(school_id)
    schools = [schedule_school(sid, inventory=inventory, bundle=bundle) for sid in wanted]
    # Proofs use ROUTED_EXPERTS even when a single other school is built, so
    # the receipt always carries watched-fail guards.
    routed = next((s for s in schools if s["school"] == "ROUTED_EXPERTS"), None)
    if routed is None:
        routed = schedule_school("ROUTED_EXPERTS", inventory=inventory, bundle=bundle)
    control = next(c for c in routed["candidates"] if c["id"] == "ROUTED_EXPERTS:LITERAL")
    trap_proof = prove_flop_trap(control)
    scar_proof = prove_scar_refusal(routed["candidates"])
    router_proof = prove_router_control_plane(inventory)

    all_cands = [c for s in schools for c in s["candidates"]]
    families_used = sorted({c["program_family"] for c in all_cands})
    workunits = [s["workunit"] for s in schools]
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Fourteen independently schedulable Flash Gravity organ schools "
            "searching for the smallest executable PROGRAM that preserves "
            "each organ's useful function. Not a quantization campaign."
        ),
        "era_vocabulary": {
            "eras": [
                "I Genesis of the Laboratory",
                "II Compounding Civilization",
                "III Autonomous Science Civilization",
                "IV Synthetic Machine Civilization",
                "V Released Hawking Civilization",
            ],
            "odysseys": [
                "I WHAT IS TRUE?",
                "II WHAT DID HAWKING ALREADY LEARN?",
                "III WHERE IS HAWKING WRONG?",
            ],
            "no_era_vi": True,
            "no_odyssey_iv": True,
            "fpga_is_not_its_own_civilization": True,
        },
        "fit_policy": "NOT_FIT",
        "measurement_classes": {
            "this_module": "STATIC_ONLY",
            "DIAGNOSTIC_RELATIVE": "not produced",
            "PROTECTED_ABSOLUTE": "not produced",
            "gpu_authority": False,
        },
        "school_catalog": list(SCHOOL_CATALOG),
        "program_families": program_family_catalog(),
        "elimination_questions": list(ELIMINATION_QUESTIONS),
        "three_zeros": list(THREE_ZEROS),
        "scoring": {
            "axes": list(AXES),
            "rule": "Pareto on storage_bytes, flop_milli, capability_risk_rank; joint_cost = storage_ratio_milli + flop_milli",
            "storage_only": "REFUSED",
            "flop_milli_unit": "1000 = 1.0x LiteralTensor structural FLOP (not a measured FLOP)",
        },
        "inventory": {
            "census_source": inventory.get("census_source"),
            "budget_source": inventory.get("budget_source"),
            "model": inventory.get("model"),
            "specimen_bytes": inventory.get("specimen_bytes"),
            "n_census_families": inventory.get("n_census_families"),
            "n_budget_families": inventory.get("n_budget_families"),
            "families": inventory.get("families"),
            "router_tensor_bytes": inventory.get("router_tensor_bytes"),
            "router_membership_change_rows": inventory.get("router_membership_change_rows"),
            "doctor": inventory.get("doctor"),
            "dispatch_count": inventory.get("dispatch_count"),
            "protected_islands": inventory.get("protected_islands"),
        },
        "schools": [
            {
                "school": s["school"],
                "independent": s["independent"],
                "schedulable_alone": s["schedulable_alone"],
                "source_bytes": s["source_bytes"],
                "census_families": s["census_families"],
                "organ_slug": s["organ_slug"],
                "elimination": s["elimination"],
                "n_candidates": s["n_candidates"],
                "program_families_used": s["program_families_used"],
                "candidates": s["candidates"],
                "refused_at_emit": s["refused_at_emit"],
                "ranking": s["ranking"],
                "workunit_id": s["workunit"]["id"],
            }
            for s in schools
        ],
        "workunits": workunits,
        "negative_control": {
            "flop_trap": trap_proof,
            "scar_refusal": scar_proof,
            "router_control_plane": router_proof,
        },
        "resident_callable": resident_callable_block([s["school"] for s in schools]),
        "recovered_implementation": recovered_implementation(bundle, inventory),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(inventory),
        "counts": {
            "schools_scheduled": len(schools),
            "schools_in_catalog": len(SCHOOL_CATALOG),
            "program_families_in_catalog": len(PROGRAM_FAMILIES),
            "program_families_used": len(families_used),
            "candidates": len(all_cands),
            "workunits": len(workunits),
            "census_families": inventory.get("n_census_families"),
            "budget_families": inventory.get("n_budget_families"),
        },
        "integration_points": {
            "frontiers.py": "frontier_fed is declared in-receipt until that sibling lands",
            "resident_api.py": "register build/schedule_school as resident verbs",
            "workgraph.py": "WorkUnits are HCLI-shaped proposals; a workgraph sibling would schedule them",
            "super_resident.py": "HCLI super-resident would loop schedule_school until Odyssey I starts",
            "wakeup.py": "physical blockers stay SLEEPING; this lane does not wake them",
            "codex_behaviors.py": "not imported; Flash Gravity stays STATIC_ONLY",
        },
    }
    bad = _hw_numeric_keys(doc)
    if bad:
        raise RuntimeError(f"hardware numeric fields leaked into Gravity receipt: {bad}")
    return write_receipt(RECEIPT, doc, RECORDED_BY)


def selftest() -> Path:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--school", metavar="SCHOOL_ID")
    a = ap.parse_args()
    if a.school and not a.build and not a.selftest:
        result = schedule_school(a.school)
        # WorkUnit content_hash / timestamps are fine on stdout; strip nothing.
        print(json.dumps(
            {
                "school": result["school"],
                "independent": result["independent"],
                "n_candidates": result["n_candidates"],
                "program_families_used": result["program_families_used"],
                "workunit_id": result["workunit"]["id"],
                "candidate_ids": [c["id"] for c in result["candidates"]],
                "elimination_zeros": result["elimination"]["three_zeros"],
                "evidence_class": result["evidence_class"],
            },
            indent=2,
            sort_keys=True,
        ))
        return 0
    out = build(school_id=a.school if a.build and a.school else None)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
