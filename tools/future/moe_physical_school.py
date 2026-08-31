"""MOE_PHYSICAL_SCHOOL — physical execution of routed expert compute.

Flash is a large MoE. Most of its bytes and most of its blocked candidates
live on the routed-expert path. Expert-bank school already generates
STORAGE and COMPUTE sharing candidates; static skeleton already names
EXPERT_ID as a permitted dynamic slot. This module is the missing
physical-execution school: how routed compute is scheduled, which
repeats exist only because experts were treated as independent, which
of those are physical invariants of MoE, and which are Metal accidents.

Analytical and structural only. Dispatch counts read out of an existing
ledger are labelled read-from-receipt. No timing, no throughput, no
hardware measurement.

    python3 tools/future/moe_physical_school.py --build
    python3 tools/future/moe_physical_school.py --selftest
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from typing import Any, Mapping

from tools.future._common import REPO, load_json, sha256_file, write_receipt
from tools.future import expert_bank_school as ebs
from tools.future import static_skeleton as sk

RECEIPT = "MOE_PHYSICAL_SCHOOL.json"
SCHEMA = "hawking.future.moe_physical.v1"
VERSION = 1

EVIDENCE_DIR = REPO / "receipts" / "future" / "evidence"
HEADLESS_DIR = REPO / "receipts" / "headless"
INDEX_RECEIPT = REPO / "receipts" / "future" / "NEGATIVE_SCIENCE_INDEX.json"

# Curated evidence this school is licensed to read. Prefer the pinned
# snapshot; fall back to live headless only when the snapshot is missing
# and record which path was taken. Never treat a missing file as proof
# the object does not exist elsewhere.
EVIDENCE_NAMES: tuple[str, ...] = (
    "FLASH_ORGAN_CENSUS.json",
    "FLASH_META_REPRESENTATION_SUB1.json",
    "FLASH_LAYER46_DISPATCH_LEDGER.json",
    "FLASH_LAYER10_CRITICAL_PATH.json",
    "FLASH_LAYER30_CRITICAL_PATH.json",
    "ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
    "ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json",
    "FLASH_NOETIC_ROUTER_SELECTION.json",
    "FLASH_DOCTOR_EXPERT_BANK_SCREEN_FULL_L44.json",
)

# Contract-named Flash routed/MoE candidate id prefixes. Match is prefix
# or exact; the queue length is derived, never hard-coded.
ROUTED_ID_PREFIXES: tuple[str, ...] = (
    "flash-p6-",
    "flash-routed-",
    "flash-compact-moe-",
    "flash-hc-",
    "flash-router-topk-fusion",
)

CLASSIFICATIONS: tuple[str, ...] = ("PHYSICAL_INVARIANT", "METAL_ACCIDENT")
TRANSFER_SCOPES: tuple[str, ...] = (
    "GENERAL_PHYSICAL",
    "MOE_FAMILY",
    "FLASH_MODEL",
    "METAL_BACKEND",
    "THIS_GRAPH",
)
INVARIANT_SCOPES = frozenset({"GENERAL_PHYSICAL", "MOE_FAMILY", "FLASH_MODEL"})
ACCIDENT_SCOPES = frozenset({"METAL_BACKEND", "THIS_GRAPH"})

OBSERVATION_FIELDS: tuple[str, ...] = (
    "id",
    "claim",
    "classification",
    "transfer_scope",
    "cheapest_falsifier",
    "distinguish_experiment",
    "evidence",
    "physical_consequence",
)
CONSEQUENCE_KEYS: tuple[str, ...] = (
    "active_bytes",
    "dispatch_count_class",
    "synchronization",
    "memory_tier_residency",
)

# Negative-index families that are MoE sharing / routed-execution scars.
# Queried first so this school does not re-propose recorded-dead sharing.
MOE_NEGATIVE_FAMILIES: tuple[str, ...] = (
    "cross_expert_structure",
    "expert_merge",
    "shared_basis",
    "expert_wave",
    "large_expert_cache",
    "megakernel",
    "router_distill",
)

WORK_CLASSES: tuple[str, ...] = (
    "ARITHMETIC",
    "CONTROL",
    "COMBINE",
    "SHARED_ARITHMETIC",
    "NOT_ROUTED_PATH",
)


class ClaimRefused(ValueError):
    """An observation that claims an invariant without a falsifier, or a
    classification/scope pair that cannot be told from a Metal accident.
    """


class FormSchemaError(ValueError):
    """An execution form is missing a required physical-consequence field."""


# ---------------------------------------------------------------------------
# Evidence loading. Cope with pinned, live, or neither.
# ---------------------------------------------------------------------------


def load_named(name: str) -> dict[str, Any]:
    """Load one named receipt. Prefer the pinned snapshot.

    Records which path was taken. A missing file is `unavailable`, not
    proof the object is absent from the primary worktree or from git.
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


# ---------------------------------------------------------------------------
# Grounding extractors (structural; numbers are read-from-receipt).
# ---------------------------------------------------------------------------


def _read_count(value: Any, receipt: str, field: str) -> dict[str, Any]:
    """Wrap a ledger/census integer so it cannot be mistaken for a measurement."""
    return {
        "value": value,
        "provenance": "read_from_receipt",
        "receipt": receipt,
        "field": field,
        "not_a_measurement": True,
    }


def organ_census_grounding(doc: dict[str, Any] | None, source: str) -> dict[str, Any]:
    if not doc:
        return {"reachable": False, "evidence_source": source}
    families = []
    routed = None
    for row in doc.get("family_summary") or []:
        if not isinstance(row, dict):
            continue
        item = {
            "family": row.get("family"),
            "tensor_count": _read_count(row.get("tensor_count"), "FLASH_ORGAN_CENSUS.json", "family_summary.tensor_count"),
            "bytes": _read_count(row.get("bytes"), "FLASH_ORGAN_CENSUS.json", "family_summary.bytes"),
            "fraction": _read_count(row.get("fraction"), "FLASH_ORGAN_CENSUS.json", "family_summary.fraction"),
        }
        families.append(item)
        if row.get("family") == "routed_experts":
            routed = item
    largest = []
    for t in doc.get("largest_tensors") or []:
        if isinstance(t, dict) and t.get("family") == "routed_experts":
            largest.append(
                {
                    "name": t.get("name"),
                    "shape": t.get("shape"),
                    "dtype": t.get("dtype"),
                    "layer": t.get("layer"),
                    "bytes": _read_count(t.get("bytes"), "FLASH_ORGAN_CENSUS.json", "largest_tensors.bytes"),
                }
            )
            if len(largest) >= 3:
                break
    return {
        "reachable": True,
        "evidence_source": source,
        "schema": doc.get("schema"),
        "status": doc.get("status"),
        "model": doc.get("model"),
        "tensor_count": _read_count(doc.get("tensor_count"), "FLASH_ORGAN_CENSUS.json", "tensor_count"),
        "layer_count_observed": _read_count(doc.get("layer_count_observed"), "FLASH_ORGAN_CENSUS.json", "layer_count_observed"),
        "source_parameter_bytes_indexed": _read_count(
            doc.get("source_parameter_bytes_indexed"),
            "FLASH_ORGAN_CENSUS.json",
            "source_parameter_bytes_indexed",
        ),
        "routed_experts": routed,
        "family_summary": families,
        "largest_routed_tensors_head": largest,
        "doctor_priority_head": list(doc.get("doctor_priority") or [])[:3],
    }


def family_budget_grounding(doc: dict[str, Any] | None, source: str) -> dict[str, Any]:
    if not doc:
        return {"reachable": False, "evidence_source": source}
    routed = None
    for row in doc.get("family_budget") or []:
        if isinstance(row, dict) and row.get("family") == "routed_experts":
            routed = {
                "family": "routed_experts",
                "source_bytes": _read_count(row.get("source_bytes"), "FLASH_META_REPRESENTATION_SUB1.json", "family_budget.source_bytes"),
                "source_fraction": _read_count(row.get("source_fraction"), "FLASH_META_REPRESENTATION_SUB1.json", "family_budget.source_fraction"),
                "meta_bpw_target": row.get("meta_bpw_target"),
                "program": row.get("program"),
                "ledger": row.get("ledger"),
                "runtime_shape": row.get("runtime_shape"),
            }
            break
    acc = doc.get("accelerator_contract") or {}
    bank = (doc.get("meta_program") or {}).get("expert_bank") or {}
    return {
        "reachable": True,
        "evidence_source": source,
        "schema": doc.get("schema"),
        "routed_experts": routed,
        "accelerator_contract": {
            "dense_rematerialization": acc.get("dense_rematerialization"),
            "resident_shared_program": acc.get("resident_shared_program"),
            "route_before_payload": acc.get("route_before_payload"),
            "fused_boundaries": list(acc.get("fused_boundaries") or []),
            "measured_now": acc.get("measured_now"),
        },
        "expert_bank_program": {
            "kind": bank.get("kind"),
            "why_not_global_sharing": bank.get("why_not_global_sharing"),
            "per_expert_code": bank.get("per_expert_code"),
            "direct_consumer": bank.get("direct_consumer"),
        },
    }


def is_routed_path_row(row: Mapping[str, Any]) -> bool:
    """True for producers of the routed/shared-expert organ.

    Consuming ``moe_output`` in a HyperConnection write is not the routed
    path. Matching the bare substring ``moe`` would pull that consumer in.
    """
    kernel = str(row.get("kernel") or "").lower()
    output = str(row.get("output") or "").lower()
    if (
        kernel.startswith("moe_")
        or "expert" in kernel
        or "topk_gate" in kernel
        or "shared_expert" in kernel
    ):
        return True
    out_tokens = (
        "router_logits",
        "route_ids",
        "routed_",
        "shared_gate",
        "shared_up",
        "shared_down",
        "shared_activation",
        "shared_output",
        "shared_scalar",
        "shared_gated",
        "moe_output",
    )
    return any(t in output for t in out_tokens)


def work_class_for_row(row: Mapping[str, Any]) -> str:
    if not is_routed_path_row(row):
        return "NOT_ROUTED_PATH"
    kernel = str(row.get("kernel") or "")
    output = str(row.get("output") or "")
    if kernel == "moe_topk_gate" or output in {"route_ids + route_weights", "router_logits"}:
        return "CONTROL"
    if kernel in {"qwen_next_moe_weighted_sum", "qwen_next_moe_add_shared"}:
        return "COMBINE"
    if "shared" in kernel or output.startswith("shared_") or "shared_" in str(row.get("input") or ""):
        return "SHARED_ARITHMETIC"
    return "ARITHMETIC"


def ledger_grounding(doc: dict[str, Any] | None, source: str) -> dict[str, Any]:
    if not doc:
        return {"reachable": False, "evidence_source": source, "rows": []}
    rows_out = []
    for i, row in enumerate(doc.get("rows") or []):
        if not isinstance(row, dict):
            continue
        rows_out.append(
            {
                "index": i,
                "kernel": row.get("kernel"),
                "input": row.get("input"),
                "output": row.get("output"),
                "why_it_exists": row.get("why_it_exists"),
                "fusion_candidate": row.get("fusion_candidate"),
                "barrier_or_dependency": row.get("barrier_or_dependency"),
                "classification_in_ledger": row.get("classification"),
                "on_routed_path": is_routed_path_row(row),
                "work_class": work_class_for_row(row),
                # gpu_ns / host_encode_us in the source ledger are None in
                # this snapshot (trace_mode off). This school does not
                # restate them even when a later snapshot fills them.
            }
        )
    barriers = sorted({str(r.get("barrier_or_dependency") or "") for r in rows_out})
    return {
        "reachable": True,
        "evidence_source": source,
        "schema": doc.get("schema"),
        "layer": doc.get("layer"),
        "status": doc.get("status"),
        "trace_mode": doc.get("trace_mode"),
        "integrated_graph": doc.get("integrated_graph"),
        "promotion_allowed": doc.get("promotion_allowed"),
        "dispatch_count": _read_count(doc.get("dispatch_count"), "FLASH_LAYER46_DISPATCH_LEDGER.json", "dispatch_count"),
        "row_count": _read_count(len(rows_out), "FLASH_LAYER46_DISPATCH_LEDGER.json", "len(rows)"),
        "barriers_observed": barriers,
        "rows": rows_out,
    }


def critical_path_owners(doc: dict[str, Any] | None, source: str, layer_name: str) -> dict[str, Any]:
    if not doc:
        return {"reachable": False, "evidence_source": source, "layer_name": layer_name}
    owners = []
    for o in doc.get("owners") or []:
        if not isinstance(o, dict):
            continue
        owners.append(
            {
                "owner": o.get("owner"),
                "kernels": o.get("kernels"),
                # ns is a hardware quantity in the source; this school records
                # only whether the field is populated, never the number.
                "ns_populated": o.get("ns") is not None,
                "measurement_note": o.get("measurement"),
            }
        )
    return {
        "reachable": True,
        "evidence_source": source,
        "layer_name": layer_name,
        "layer": doc.get("layer"),
        "schema": doc.get("schema"),
        "status": doc.get("status"),
        "promotion_allowed": doc.get("promotion_allowed"),
        "owners": owners,
        "ceremony_owners": [
            o["owner"]
            for o in owners
            if o.get("owner") in {"synchronization", "command submission"}
        ],
        "routed_owners": [
            o["owner"]
            for o in owners
            if o.get("owner") in {"routing", "selected experts", "shared expert"}
        ],
    }


def router_config_grounding(doc: dict[str, Any] | None, source: str) -> dict[str, Any]:
    if not doc:
        return {"reachable": False, "evidence_source": source}
    router = ((doc.get("config") or {}).get("router") or {})
    return {
        "reachable": True,
        "evidence_source": source,
        "schema": doc.get("schema"),
        "num_experts": _read_count(router.get("num_experts"), "FLASH_NOETIC_ROUTER_SELECTION.json", "config.router.num_experts"),
        "num_experts_per_tok": _read_count(router.get("num_experts_per_tok"), "FLASH_NOETIC_ROUTER_SELECTION.json", "config.router.num_experts_per_tok"),
        "selection": router.get("selection"),
        "router_logits": router.get("router_logits"),
        "router_probability": router.get("router_probability"),
        "norm_topk_prob": router.get("norm_topk_prob"),
        "shared_expert_sigmoid_is_not_router_selection": router.get(
            "shared_expert_sigmoid_is_not_router_selection"
        ),
    }


def doctor_screen_grounding(doc: dict[str, Any] | None, source: str) -> dict[str, Any]:
    if not doc:
        return {"reachable": False, "evidence_source": source}
    pop = doc.get("population") or {}
    src = doc.get("source") or {}
    return {
        "reachable": True,
        "evidence_source": source,
        "schema": doc.get("schema"),
        "status": doc.get("status"),
        "layer": src.get("layer"),
        "expert_count": _read_count(src.get("expert_count"), "FLASH_DOCTOR_EXPERT_BANK_SCREEN_FULL_L44.json", "source.expert_count"),
        "experts_sampled_count": _read_count(
            len(src.get("experts_sampled") or []),
            "FLASH_DOCTOR_EXPERT_BANK_SCREEN_FULL_L44.json",
            "len(source.experts_sampled)",
        ),
        # Structure metrics, not hardware. Cited so repeated-work is not
        # mistaken for "experts are similar".
        "cross_expert_gate_up_mean_cosine": pop.get("cross_expert_gate_up_mean_cosine"),
        "cross_expert_gate_up_min_cosine": pop.get("cross_expert_gate_up_min_cosine"),
        "rank_1_energy": ((pop.get("sampled_population_rank") or {}).get("rank_1_energy")),
        "early_rejection": (doc.get("doctor_funnel") or {}).get("early_rejection"),
    }


def is_routed_candidate_id(candidate_id: str) -> bool:
    cid = str(candidate_id or "")
    return any(cid == p or cid.startswith(p) for p in ROUTED_ID_PREFIXES)


def select_routed_candidates(queue: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Derive the routed/MoE subset. Queue length is not an invariant."""
    if not queue:
        return []
    rows = []
    for c in queue.get("candidates") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("candidate_id") or "")
        if not is_routed_candidate_id(cid):
            continue
        rows.append(
            {
                "candidate_id": cid,
                "status": c.get("status"),
                "exact_mutation": c.get("exact_mutation"),
                "expected_eliminated_work": c.get("expected_eliminated_work"),
                "expected_dispatch_reduction": c.get("expected_dispatch_reduction"),
                "expected_active_byte_change": c.get("expected_active_byte_change"),
                "expected_intermediate_byte_reduction": c.get("expected_intermediate_byte_reduction"),
                "expected_gpu_ns_mechanism": c.get("expected_gpu_ns_mechanism"),
                "affected_physical_region": c.get("affected_physical_region"),
                "scope_tags": list(c.get("scope_tags") or []),
                "blocked_reason": c.get("blocked_reason"),
            }
        )
    rows.sort(key=lambda r: str(r["candidate_id"]))
    return rows


def dispatch_cost_law(doc: dict[str, Any] | None, source: str) -> dict[str, Any]:
    """Cite the qualitative law. Do not restate measured microseconds."""
    if not doc:
        return {"reachable": False, "evidence_source": source}
    finding = doc.get("FINDING_3_A_DISPATCH_IS_NOT_A_UNIT_OF_COST") or {}
    return {
        "reachable": True,
        "evidence_source": source,
        "receipt": "ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json",
        "law": "a dispatch is not a unit of cost",
        "headline_finding_present": bool(finding),
        "instance_note": (
            "the source receipt is a Qwen3.8 Metal measurement on one prompt "
            "length. this school cites the qualitative law only; magnitudes "
            "are not restated and are not Flash authority."
        ),
        "so_this_school": (
            "ceremony vs work is not a dispatch-count ranking. a fusion "
            "candidate is a hypothesis about eliminated arithmetic or "
            "movement, not about subtracting launches."
        ),
    }


# ---------------------------------------------------------------------------
# Negative-science query (receipt, not a live ingest).
# ---------------------------------------------------------------------------


def consult_moe_negatives() -> dict[str, Any]:
    """Query the landed index for MoE sharing / routed-execution scars.

    Uses the sealed sidecar receipt so this school does not re-ingest the
    corpus. Copes if the index has not landed in this checkout.
    """
    if not INDEX_RECEIPT.is_file():
        return {
            "index_reachable": False,
            "evidence_source": "unavailable",
            "coped": (
                "NEGATIVE_SCIENCE_INDEX.json not reachable here; builtin "
                "expert-bank scars still cover raw global similarity, trivial "
                "shared basis, and unchanged archetypes."
            ),
            "families_queried": list(MOE_NEGATIVE_FAMILIES),
            "hit_count": 0,
            "refuse_eligible_count": 0,
            "hits": [],
        }
    doc = load_json(INDEX_RECEIPT)
    scars = doc.get("scars") or []
    want = set(MOE_NEGATIVE_FAMILIES)
    hits = []
    for s in scars:
        if not isinstance(s, dict):
            continue
        fam = str(s.get("hypothesis_family") or "")
        organ = str(s.get("organ") or "")
        if fam not in want:
            continue
        hits.append(
            {
                "scar_id": s.get("scar_id"),
                "hypothesis_family": fam,
                "organ": organ,
                "model": s.get("model"),
                "verdict": s.get("verdict"),
                "level": s.get("level"),
                "refuse_eligible": bool(s.get("refuse_eligible")),
            }
        )
    hits.sort(key=lambda r: (str(r.get("hypothesis_family")), str(r.get("scar_id"))))
    n_refuse = sum(1 for h in hits if h.get("refuse_eligible"))
    return {
        "index_reachable": True,
        "evidence_source": "sidecar_receipt",
        "path": "receipts/future/NEGATIVE_SCIENCE_INDEX.json",
        "families_queried": list(MOE_NEGATIVE_FAMILIES),
        "hit_count": len(hits),
        "refuse_eligible_count": n_refuse,
        "sample_scar_ids": [h["scar_id"] for h in hits[:12]],
        "implication": (
            "Flash L44 gate_up mean cosine is ~0.0038 (doctor screen). "
            "Experts are not similar, so repeated work is structural "
            "(same x, same SwiGLU shape, independent W_e), not a license "
            "to merge experts or to reopen a trivial shared basis. "
            "Megakernel-as-TPS-lever is recorded-dead; fused routed-organ "
            "dispatch is a different object from a full-layer megakernel."
        ),
    }


# ---------------------------------------------------------------------------
# Execution taxonomy
# ---------------------------------------------------------------------------


def _consequence(
    active_bytes: str,
    dispatch_count_class: str,
    synchronization: str,
    memory_tier_residency: str,
) -> dict[str, str]:
    return {
        "active_bytes": active_bytes,
        "dispatch_count_class": dispatch_count_class,
        "synchronization": synchronization,
        "memory_tier_residency": memory_tier_residency,
    }


def execution_taxonomy() -> list[dict[str, Any]]:
    """The nine forms the lane contract named, with physical consequences."""
    forms: list[dict[str, Any]] = [
        {
            "id": "gather_scatter",
            "axis": "geometry",
            "title": "gather/scatter of selected expert payloads",
            "description": (
                "Route ids index a compile-time bank. The kernel gathers k of E "
                "weight tiles against one x and scatters weighted outputs into "
                "the residual. Unselected experts are not read."
            ),
            "physical_consequence": _consequence(
                "O_K_EXPERT_PAYLOADS_PLUS_SHARED",
                "STATIC_ONE_OR_BOUNDED_TOPK",
                "ONE_ENCODER_ORDERED",
                "SELECTED_TILES_ON_DEMAND_BANK_RESIDENT",
            ),
            "flash_layer46": "OBSERVED",
            "replayable_with_expert_id_only": True,
            "skeleton_kind": "SLOT_INDEXED",
            "notes": (
                "layer-46 kernels qwen_next_bf16_expert_gate_up_swiglu and "
                "qwen_next_bf16_expert_down take route_ids; they do not "
                "iterate the 512-expert bank."
            ),
        },
        {
            "id": "dense_with_mask",
            "axis": "geometry",
            "title": "dense-with-mask over the whole expert bank",
            "description": (
                "Every expert is applied; unselected outputs are multiplied by "
                "zero (or a mask). Topology is STATIC (all E always exist). "
                "Active bytes are O(E), not O(k)."
            ),
            "physical_consequence": _consequence(
                "O_E_EXPERT_PAYLOADS",
                "STATIC_ONE_OR_ONE_PER_EXPERT_IN_BANK",
                "ONE_ENCODER_ORDERED",
                "WHOLE_BANK_TOUCHED_PER_TOKEN",
            ),
            "flash_layer46": "NOT_OBSERVED",
            "replayable_with_expert_id_only": True,
            "skeleton_kind": "STATIC",
            "notes": (
                "replayable because the node set does not depend on the route; "
                "EXPERT_ID is unused or is only a mask tensor. Census bytes "
                "make this form byte-catastrophic on Flash decode."
            ),
        },
        {
            "id": "sorted_by_expert",
            "axis": "geometry",
            "title": "sort tokens by expert, then grouped GEMM",
            "description": (
                "Batched/prefill form: permute tokens so each expert's rows "
                "are contiguous, run grouped GEMM, unpermute. Decode of one "
                "token has nothing to sort."
            ),
            "physical_consequence": _consequence(
                "O_K_EXPERT_PAYLOADS_TIMES_TOKENS_IN_GROUP",
                "BOUNDED_BY_UNIQUE_EXPERTS_IN_BATCH",
                "SORT_THEN_GROUPED_THEN_UNSORT",
                "GROUPED_TILES_STREAMED",
            ),
            "flash_layer46": "NOT_OBSERVED",
            "replayable_with_expert_id_only": True,
            "skeleton_kind": "SLOT_INDEXED",
            "notes": (
                "the sort is an operator, not topology. dispatch_bound is a "
                "compile-time max unique experts, not a value-gated count."
            ),
        },
        {
            "id": "per_expert_dispatch",
            "axis": "launch",
            "title": "one launch family per selected expert",
            "description": (
                "Each selected expert is its own dispatch (or its own W1/W3/W2 "
                "triple). Bound by top-k / hash-k if the slot is declared."
            ),
            "physical_consequence": _consequence(
                "O_K_EXPERT_PAYLOADS_PLUS_SHARED",
                "ONE_PER_SELECTED_EXPERT",
                "PER_EXPERT_OR_CONCURRENT_WAVES",
                "PER_EXPERT_BUFFERS_PLUS_COMBINE",
            ),
            "flash_layer46": "NOT_OBSERVED",
            "p6_hash_graph": "OBSERVED",
            "replayable_with_expert_id_only": True,
            "skeleton_kind": "SLOT_INDEXED",
            "notes": (
                "replayable iff dispatch_count_from_slot=expert_id and "
                "dispatch_bound=k. Unbounded per-score launches are "
                "VALUE_GATED and refused. P6 candidates describe six W1, "
                "six W3, six SwiGLU, six casts as the historical hash graph."
            ),
        },
        {
            "id": "fused_routed_dispatch",
            "axis": "launch",
            "title": "one fused routed dispatch over route_ids",
            "description": (
                "A single (or small static) kernel consumes the route_id "
                "buffer, gathers selected tiles, applies gate/up/SwiGLU/down, "
                "and writes the combine contract."
            ),
            "physical_consequence": _consequence(
                "O_K_EXPERT_PAYLOADS_PLUS_SHARED",
                "STATIC_ONE",
                "ONE_ENCODER_ORDERED",
                "ROUTE_IDS_PLUS_SELECTED_TILES_RESIDENT",
            ),
            "flash_layer46": "OBSERVED_PARTIAL",
            "replayable_with_expert_id_only": True,
            "skeleton_kind": "SLOT_INDEXED_INSIDE_STATIC_KERNEL",
            "notes": (
                "layer-46 already fuses selected experts into two kernels "
                "(gate_up_swiglu, down), not ten. remaining splits are "
                "gate_up vs down, router vs top-k, and the six-kernel shared "
                "expert. that remainder is a fusion candidate, not a new form."
            ),
        },
        {
            "id": "route_before_payload",
            "axis": "order",
            "title": "route, then fetch/apply only selected payloads",
            "description": (
                "Router logits and top-k run first. Expert bodies read "
                "route_ids. Unselected tiles stay untouched."
            ),
            "physical_consequence": _consequence(
                "O_K_EXPERT_PAYLOADS_PLUS_SHARED",
                "STATIC_ROUTER_PLUS_STATIC_OR_SLOT_BODY",
                "ROUTE_THEN_BODY_DEPENDENCY",
                "ROUTE_METADATA_THEN_SELECTED_TILES",
            ),
            "flash_layer46": "OBSERVED",
            "replayable_with_expert_id_only": True,
            "skeleton_kind": "STATIC_THEN_SLOT_INDEXED",
            "notes": (
                "FLASH_META accelerator_contract.route_before_payload is true. "
                "runtime_shape is route -> latent decode -> fused "
                "gate/up/SwiGLU/down accumulation."
            ),
        },
        {
            "id": "payload_then_select",
            "axis": "order",
            "title": "apply payloads first, then select",
            "description": (
                "Expert bodies run before (or without) a route decision, then "
                "a mask/select keeps k of E. Two skeletonizations: STATIC all-E "
                "(replayable, pays O(E)) or VALUE_GATED skip (not replayable)."
            ),
            "physical_consequence": _consequence(
                "O_E_EXPERT_PAYLOADS_UNLESS_GATED",
                "STATIC_ALL_E_OR_DATA_DEPENDENT_UNBOUNDED",
                "BODY_THEN_SELECT",
                "WHOLE_BANK_OR_GATED_SUBGRAPH",
            ),
            "flash_layer46": "NOT_OBSERVED",
            "replayable_with_expert_id_only": "STATIC_ALL_E_ONLY",
            "skeleton_kind": "STATIC_OR_VALUE_GATED",
            "notes": (
                "the gated skeletonization is illegal under static_skeleton. "
                "the static all-E skeletonization is dense-with-mask by another "
                "name. meta contract forbids dense rematerialization."
            ),
        },
        {
            "id": "static_skeleton_expert_id",
            "axis": "skeleton",
            "title": "compile-time bank, EXPERT_ID as the only routed hole",
            "description": (
                "Nodes, edges, and dispatch bounds are captured once. Runtime "
                "binds a bounded EXPERT_ID slot (and maybe a representation "
                "window). Graph replay / CUDA graphs / FPGA spatial pipelines "
                "are the same idea."
            ),
            "physical_consequence": _consequence(
                "O_K_OF_CAPTURED_BANK",
                "CAPTURED_BOUND",
                "REPLAY_WITH_SETBYTES",
                "BANK_RESIDENT_SLOT_IS_AN_INDEX",
            ),
            "flash_layer46": "OBSERVED",
            "replayable_with_expert_id_only": True,
            "skeleton_kind": "SLOT_INDEXED",
            "notes": (
                "static_skeleton.flash_moe_component_skeleton already encodes "
                "this: router STATIC, expert_bank SLOT_INDEXED on expert_id, "
                "shared_expert STATIC. layer-46 agrees: all 35 rows exist "
                "regardless of which ids were selected."
            ),
        },
        {
            "id": "data_dependent_topology",
            "axis": "skeleton",
            "title": "node/edge/dispatch existence depends on a runtime value",
            "description": (
                "Which kernels exist, which edges exist, or how many dispatches "
                "fire is a function of activation magnitude, nnz, or an "
                "unbounded score threshold. Cannot be replayed."
            ),
            "physical_consequence": _consequence(
                "UNKNOWN_AT_CAPTURE",
                "DATA_DEPENDENT_UNBOUNDED",
                "HOST_RESHAPE_OR_INDIRECT_GRAPH_BREAK",
                "CANNOT_DECLARE_RESIDENCY_AHEAD_OF_ROUTE",
            ),
            "flash_layer46": "NOT_OBSERVED",
            "replayable_with_expert_id_only": False,
            "skeleton_kind": "VALUE_GATED",
            "notes": (
                "static_skeleton.validate refuses VALUE_GATED existence and "
                "dispatch_count_gated_on a runtime value. this is the form "
                "the school exists to keep off the routed path."
            ),
        },
    ]
    for form in forms:
        _require_form(form)
    return forms


def _require_form(form: Mapping[str, Any]) -> None:
    if not form.get("id") or not form.get("description"):
        raise FormSchemaError(f"form missing id/description: {form.get('id')}")
    pc = form.get("physical_consequence") or {}
    missing = [k for k in CONSEQUENCE_KEYS if k not in pc]
    if missing:
        raise FormSchemaError(f"{form.get('id')} missing consequence {missing}")


# ---------------------------------------------------------------------------
# Skeleton cross-reference
# ---------------------------------------------------------------------------


def skeleton_for_form(form_id: str) -> sk.Skeleton:
    """A minimal skeleton realizing the form, for the validator."""
    expert = sk.Slot(
        name="expert_id",
        kind="EXPERT_ID",
        lo=0,
        hi=511,
        dtype="u16",
        dispatch_bound=10,
        meaning="routed expert index",
    )
    if form_id == "gather_scatter":
        return sk.Skeleton(
            name="gather_scatter",
            slots=(expert,),
            nodes=(
                sk.Node(id="router", kind="router_matvec", dispatch_count=1),
                sk.Node(
                    id="expert_body",
                    kind="fused_routed_body",
                    existence="SLOT_INDEXED",
                    slot="expert_id",
                    dispatch_count=1,
                    binds_slots=("expert_id",),
                ),
                sk.Node(id="combine", kind="weighted_sum", dispatch_count=1),
            ),
            edges=(
                sk.Edge(src="router", dst="expert_body", existence="SLOT_INDEXED", slot="expert_id"),
                sk.Edge(src="expert_body", dst="combine"),
            ),
        )
    if form_id == "dense_with_mask":
        return sk.Skeleton(
            name="dense_with_mask",
            slots=(expert,),
            nodes=(
                sk.Node(id="router", kind="router_matvec", dispatch_count=1),
                sk.Node(id="all_experts", kind="dense_bank", existence="STATIC", dispatch_count=1),
                sk.Node(
                    id="mask",
                    kind="route_mask",
                    existence="STATIC",
                    dispatch_count=1,
                    binds_slots=("expert_id",),
                ),
            ),
            edges=(
                sk.Edge(src="router", dst="mask"),
                sk.Edge(src="all_experts", dst="mask"),
            ),
        )
    if form_id == "sorted_by_expert":
        return sk.Skeleton(
            name="sorted_by_expert",
            slots=(expert,),
            nodes=(
                sk.Node(id="sort", kind="sort_by_expert", dispatch_count=1, binds_slots=("expert_id",)),
                sk.Node(
                    id="grouped",
                    kind="grouped_gemm",
                    existence="SLOT_INDEXED",
                    slot="expert_id",
                    dispatch_count=1,
                ),
                sk.Node(id="unsort", kind="unpermute", dispatch_count=1),
            ),
            edges=(
                sk.Edge(src="sort", dst="grouped", existence="SLOT_INDEXED", slot="expert_id"),
                sk.Edge(src="grouped", dst="unsort"),
            ),
        )
    if form_id == "per_expert_dispatch":
        return sk.Skeleton(
            name="per_expert_dispatch",
            slots=(expert,),
            nodes=(
                sk.Node(id="router", kind="router_matvec", dispatch_count=1),
                sk.Node(
                    id="expert_bank",
                    kind="expert_body",
                    existence="SLOT_INDEXED",
                    slot="expert_id",
                    dispatch_count=10,
                    dispatch_count_from_slot="expert_id",
                ),
            ),
            edges=(
                sk.Edge(src="router", dst="expert_bank", existence="SLOT_INDEXED", slot="expert_id"),
            ),
        )
    if form_id == "fused_routed_dispatch":
        return sk.Skeleton(
            name="fused_routed_dispatch",
            slots=(expert,),
            nodes=(
                sk.Node(id="router", kind="router_matvec", dispatch_count=1),
                sk.Node(
                    id="fused_body",
                    kind="gate_up_swiglu_down",
                    existence="SLOT_INDEXED",
                    slot="expert_id",
                    dispatch_count=1,
                    binds_slots=("expert_id",),
                ),
            ),
            edges=(
                sk.Edge(src="router", dst="fused_body", existence="SLOT_INDEXED", slot="expert_id"),
            ),
        )
    if form_id == "route_before_payload":
        return sk.legal_expert_id_skeleton()
    if form_id == "payload_then_select_static":
        return sk.Skeleton(
            name="payload_then_select_static",
            slots=(expert,),
            nodes=(
                sk.Node(id="all_experts", kind="dense_bank", existence="STATIC", dispatch_count=1),
                sk.Node(
                    id="select",
                    kind="route_select",
                    existence="STATIC",
                    dispatch_count=1,
                    binds_slots=("expert_id",),
                ),
            ),
            edges=(sk.Edge(src="all_experts", dst="select"),),
        )
    if form_id == "payload_then_select_gated":
        return sk.illegal_activation_gated_skeleton()
    if form_id == "static_skeleton_expert_id":
        return sk.legal_expert_id_skeleton()
    if form_id == "data_dependent_topology":
        return sk.illegal_activation_gated_skeleton()
    raise KeyError(f"unknown form {form_id}")


REPLAYABLE_FORMS: tuple[str, ...] = (
    "gather_scatter",
    "dense_with_mask",
    "sorted_by_expert",
    "per_expert_dispatch",
    "fused_routed_dispatch",
    "route_before_payload",
    "payload_then_select_static",
    "static_skeleton_expert_id",
)
NOT_REPLAYABLE_FORMS: tuple[str, ...] = (
    "payload_then_select_gated",
    "data_dependent_topology",
)


def skeleton_cross_reference() -> dict[str, Any]:
    accepted = []
    refused = []
    for form_id in REPLAYABLE_FORMS:
        result = sk.validate(skeleton_for_form(form_id))
        accepted.append(
            {
                "form_id": form_id,
                "accepted": result.accepted,
                "errors": list(result.errors),
            }
        )
        if not result.accepted:
            raise ClaimRefused(f"replayable form {form_id} was refused: {result.errors}")
    for form_id in NOT_REPLAYABLE_FORMS:
        result = sk.validate(skeleton_for_form(form_id))
        refused.append(
            {
                "form_id": form_id,
                "accepted": result.accepted,
                "errors": list(result.errors),
            }
        )
        if result.accepted:
            raise ClaimRefused(f"VALUE_GATED form {form_id} was accepted; guard did not fire")
    return {
        "permitted_dynamic_slots": list(sk.SLOT_KINDS),
        "expert_id_is_permitted": "EXPERT_ID" in sk.SLOT_KINDS,
        "replayable_with_expert_id_only": accepted,
        "require_data_dependent_topology": refused,
        "flash_moe_component": (
            "tools/future/static_skeleton.py::flash_moe_component_skeleton "
            "already models Flash router STATIC + expert bank SLOT_INDEXED "
            "on EXPERT_ID + shared expert STATIC. this school does not fork it."
        ),
        "reading": (
            "gather/scatter, fused routed dispatch, per-expert dispatch with "
            "a declared top-k bound, sorted-by-expert, dense-with-mask, and "
            "route-before-payload all replay. payload-then-select replays only "
            "as STATIC all-E. VALUE_GATED topology does not replay, and the "
            "validator refuses it."
        ),
    }


# ---------------------------------------------------------------------------
# Repeated-work census (grounded, not theoretical)
# ---------------------------------------------------------------------------


def _layer46_kernel_set(ledger: Mapping[str, Any]) -> list[str]:
    return [str(r.get("kernel")) for r in ledger.get("rows") or [] if r.get("on_routed_path")]


def repeated_work_census(
    census: Mapping[str, Any],
    budget: Mapping[str, Any],
    ledger: Mapping[str, Any],
    doctor: Mapping[str, Any],
    router: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """What is repeated ONLY because source matrices were independent.

    Grounded in the census, the layer-46 ledger, the doctor screen, and
    the already-landed expert-bank COMPUTE kinds. Does not restate those
    kinds as new candidates.
    """
    kernels = _layer46_kernel_set(ledger)
    cosine = doctor.get("cross_expert_gate_up_mean_cosine")
    items = [
        {
            "id": "RW-INDEPENDENT-W-E",
            "repeated": (
                "512 independent expert matrices of shape [512, 1280, 2560] "
                "BF16 gate_up (and the paired down) per MoE layer, stored as "
                "separate source tensors."
            ),
            "why_only_because_independent": (
                "source packs one W_e per expert. the doctor screen mean "
                "cosine is not a sharing license; the repeat is the storage "
                "of orthogonal operators with a shared SwiGLU *shape*."
            ),
            "already_named_by": "tools/future/expert_bank_school.py STORAGE axis",
            "layer46_status": (
                "not a dispatch repeat: layer-46 gathers selected tiles inside "
                "two kernels. the independence that remains is the payload."
            ),
            "grounding": {
                "census_family": "routed_experts",
                "cosine_mean": cosine,
                "shape_from_largest_tensor": (census.get("largest_routed_tensors_head") or [{}])[0].get("shape"),
            },
        },
        {
            "id": "RW-SAME-X-MANY-EXPERTS",
            "repeated": (
                "the same hidden vector x is dotted with k independent "
                "gate/up rows."
            ),
            "why_only_because_independent": (
                "source stores one matrix per expert, so a naive runtime "
                "honors that with one launch family per expert even though "
                "decode x is identical."
            ),
            "already_named_by": "COMPUTE-ONE-X-MANY-EXPERTS",
            "layer46_status": (
                "DISPATCH FUSED: qwen_next_bf16_expert_gate_up_swiglu takes "
                "route_ids + mlp_input + expert_gate_up as one kernel. "
                "remaining repeat is k independent weight streams, not k "
                "launches."
            ),
            "grounding": {"kernels_on_routed_path": kernels},
        },
        {
            "id": "RW-GATE-AND-UP-TWO-MATRICES",
            "repeated": (
                "gate_proj @ x and up_proj @ x as two full projections, even "
                "when fused in one kernel."
            ),
            "why_only_because_independent": (
                "source has two independent input matrices per expert "
                "(or a concatenated gate_up that is still two operators). "
                "fusion multiplies both by x; it does not hoist a shared "
                "factor of x."
            ),
            "already_named_by": "COMPUTE-SHARED-PROJECTIONS-ACROSS-ORGANS",
            "layer46_status": "FUSED MULTIPLY, UNSHARED FACTORS. Not in the graph as a shared V.",
            "grounding": {"kernel": "qwen_next_bf16_expert_gate_up_swiglu"},
        },
        {
            "id": "RW-DOWN-THEN-COMBINE",
            "repeated": (
                "k independent down_proj expansions to d_model, then a "
                "weighted sum in d_model."
            ),
            "why_only_because_independent": (
                "source down_proj is per-expert to full hidden, so the "
                "combiner is written after expansion. linearity would allow "
                "reduce-then-expand IF a shared U existed; it does not in "
                "this graph."
            ),
            "already_named_by": "COMPUTE-LATENT-WEIGHTED-REDUCTION / COMPUTE-ONE-OUTPUT-EXPANSION",
            "layer46_status": (
                "qwen_next_bf16_expert_down writes routed_outputs; "
                "qwen_next_moe_weighted_sum combines at full width. no "
                "latent combine node."
            ),
            "grounding": {
                "kernels": ["qwen_next_bf16_expert_down", "qwen_next_moe_weighted_sum"]
            },
        },
        {
            "id": "RW-SHARED-EXPERT-NOT-A-REPEAT-OF-THE-BANK",
            "repeated": (
                "the shared expert is a single always-on SwiGLU, not 512 "
                "copies. it is a different organ (census family shared_expert, "
                "fraction ~0.13%)."
            ),
            "why_only_because_independent": (
                "this is the control: a native shared operator already exists "
                "and is NOT routed by EXPERT_ID. treating it as another "
                "routed expert would invent a repeat the source does not have."
            ),
            "already_named_by": "static_skeleton shared_expert STATIC; router config shared_expert_sigmoid_is_not_router_selection",
            "layer46_status": (
                "six kernels (gate, up, silu_mul, down, scalar, sigmoid) plus "
                "qwen_next_moe_add_shared. always present in the ledger, "
                "independent of route_ids."
            ),
            "grounding": {
                "router_flag": router.get("shared_expert_sigmoid_is_not_router_selection"),
                "census_family": "shared_expert",
            },
        },
        {
            "id": "RW-P6-PER-EXPERT-LAUNCHES",
            "repeated": (
                "P6 hash graph historically launches six W1, six W3, six "
                "SwiGLU, six FP32-to-BF16 casts, six W2 — selected experts "
                "as independent DISPATCHES, not only independent payloads."
            ),
            "why_only_because_independent": (
                "the runtime honors source independence with one launch "
                "family per selected expert. layer-46 BF16 source-oracle "
                "already refused that interpretation for the same algebra."
            ),
            "already_named_by": (
                "queue candidates flash-p6-routed-fp4-gate-up-swiglu-fused "
                "and flash-p6-routed-fp4-down-bf16-fused"
            ),
            "layer46_status": "NOT the layer-46 source-oracle graph. P6 / hash-route specific.",
            "grounding": {"queue_prefix": "flash-p6-routed-fp4-"},
        },
        {
            "id": "RW-PER-EXPERT-UNPACK",
            "repeated": (
                "per-expert dequant / unpack / scale of packed FP4/FP8 "
                "payloads before or during each expert matvec."
            ),
            "why_only_because_independent": (
                "source payloads were packed independently, so decode is "
                "written as an expert-local prologue rather than a bank-level "
                "codebook in registers."
            ),
            "already_named_by": "COMPUTE-SHARED-REPRESENTATION-DECODE",
            "layer46_status": (
                "source-oracle layer-46 is BF16 (critical-path owner "
                "'representation conversion' = none; source BF16 remains "
                "resident). unpack repeat is a P6 packed path, not layer-46."
            ),
            "grounding": {"layer46_dtype": "BF16"},
        },
        {
            "id": "RW-CROSS-LAYER-PROJECT-X-AGAIN",
            "repeated": (
                "independent V_L @ x_L at every layer, including when x_L is "
                "a residual update of x_{L-1}."
            ),
            "why_only_because_independent": (
                "source layers are separate tensors with no residual-product "
                "identity in the graph."
            ),
            "already_named_by": "COMPUTE-CROSS-LAYER-REUSED-TRANSFORM",
            "layer46_status": "NOT VISIBLE in a single-layer ledger. named so it is not mistaken for a layer-46 dispatch split.",
            "grounding": {"ledger_layer": ledger.get("layer")},
        },
    ]
    items.append(
        {
            "id": "RW-WHAT-IS-NOT-REPEATED",
            "repeated": None,
            "why_only_because_independent": None,
            "already_named_by": "FLASH_META expert_bank.why_not_global_sharing",
            "layer46_status": (
                "do not read independent W_e as 'the same computation ran 512 "
                "times'. cosine ~0 and rank-1 energy ~0.003 say the operators "
                "are not copies. the repeated object is the *schedule shape* "
                "(SwiGLU against one x, then linear combine), not the values."
            ),
            "grounding": {
                "meta_program": (budget.get("expert_bank_program") or {}).get("why_not_global_sharing"),
                "cosine_mean": cosine,
                "rank_1_energy": doctor.get("rank_1_energy"),
            },
        }
    )
    return items


def cited_compute_sharing(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Pointer into expert_bank COMPUTE kinds with layer-46 execution status."""
    status_by_kind = {
        "one_hidden_vector_many_experts": "DISPATCH_FUSED_PAYLOAD_INDEPENDENT",
        "shared_xb_then_skinny": "NOT_IN_GRAPH",
        "latent_weighted_reduction": "NOT_IN_GRAPH",
        "one_output_expansion": "NOT_IN_GRAPH",
        "shared_representation_decode": "NOT_IN_LAYER46_BF16_P6_PACKED_PATH",
        "shared_projections_across_organs": "FUSED_MULTIPLY_UNSHARED_FACTORS",
        "cross_layer_reused_transforms": "NOT_IN_SINGLE_LAYER_LEDGER",
    }
    by_kind = {c["kind"]: c for c in ebs.COMPUTE_CANDIDATES}
    rows = []
    for kind in ebs.REQUIRED_COMPUTE_KINDS:
        c = by_kind[kind]
        rows.append(
            {
                "id": c["id"],
                "kind": c["kind"],
                "repeated_computation": c["repeated_computation"],
                "why_currently_repeated": c["why_currently_repeated"],
                "source": "tools/future/expert_bank_school.py",
                "layer46_execution_status": status_by_kind.get(c["kind"], "UNMAPPED"),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Ceremony vs work
# ---------------------------------------------------------------------------


def ceremony_vs_work(
    ledger: Mapping[str, Any],
    path10: Mapping[str, Any],
    path30: Mapping[str, Any],
    cost_law: Mapping[str, Any],
) -> dict[str, Any]:
    routed = [r for r in ledger.get("rows") or [] if r.get("on_routed_path")]
    by_class: dict[str, list[dict[str, Any]]] = {k: [] for k in WORK_CLASSES}
    for r in ledger.get("rows") or []:
        by_class[r["work_class"]].append(
            {
                "index": r["index"],
                "kernel": r["kernel"],
                "output": r["output"],
                "fusion_candidate": r["fusion_candidate"],
            }
        )
    fusion_splits = [
        {
            "index": r["index"],
            "kernel": r["kernel"],
            "fusion_candidate": r["fusion_candidate"],
        }
        for r in routed
    ]
    return {
        "law_cited": cost_law.get("law"),
        "law_implication": cost_law.get("so_this_school"),
        "ledger_layer": ledger.get("layer"),
        "ledger_dispatch_count": ledger.get("dispatch_count"),
        "barriers_observed": ledger.get("barriers_observed"),
        "routed_path_row_count": _read_count(
            len(routed),
            "FLASH_LAYER46_DISPATCH_LEDGER.json",
            "count(rows on routed path)",
        ),
        "arithmetic": {
            "meaning": (
                "genuine expert / shared-expert FLOPs on activations and weights "
                "that produce the MoE output."
            ),
            "rows": by_class["ARITHMETIC"] + by_class["SHARED_ARITHMETIC"],
        },
        "control": {
            "meaning": (
                "route decision: router logits and device-resident top-k. this "
                "is necessary control-flow arithmetic, not scheduling overhead. "
                "it produces the EXPERT_ID slot bind."
            ),
            "rows": by_class["CONTROL"],
        },
        "combine": {
            "meaning": (
                "linear weighted sum of routed outputs and the routed-plus-shared "
                "add. algebra, not ceremony."
            ),
            "rows": by_class["COMBINE"],
        },
        "scheduling_overhead": {
            "meaning": (
                "what is still split only to meet the current Metal encoder / "
                "kernel ABI, as named by each row's fusion_candidate, plus the "
                "command-buffer model. not a measured cost: a dispatch is not "
                "a unit of cost."
            ),
            "token_command_buffer": (
                "every layer-46 row is 'ordered within one TokenCommandBuffer'. "
                "the source-oracle graph is already a single CB; P6 hash-route "
                "historically used two CBs with a host wait."
            ),
            "fusion_candidates_still_open_on_routed_path": fusion_splits,
            "critical_path_ceremony_owners": {
                "layer10": path10.get("ceremony_owners") or [],
                "layer30": path30.get("ceremony_owners") or [],
                "note": (
                    "owners 'synchronization' (TokenCommandBuffer commit_and_wait) "
                    "and 'command submission' (Metal command buffer). per-owner "
                    "GPU ns is unpopulated while trace mode is off; this school "
                    "does not fill it."
                ),
            },
            "not_in_the_ledger_but_named_by_the_queue": [
                "P6 dual command-buffer host commit/wait",
                "P6 learned-route sealed-reader admission and source-cache",
                "P6 FP32 staging + BF16 cast launches",
                "scalar vs vec4 / simdgroup occupancy geometry",
            ],
        },
        "not_routed_path": {
            "meaning": "attention / DeltaNet / HyperConnection rows in the same layer ledger.",
            "row_count": _read_count(
                len(by_class["NOT_ROUTED_PATH"]),
                "FLASH_LAYER46_DISPATCH_LEDGER.json",
                "count(rows not on routed path)",
            ),
        },
    }


# ---------------------------------------------------------------------------
# Invariant vs accident (load-bearing)
# ---------------------------------------------------------------------------


def _obs(
    oid: str,
    claim: str,
    classification: str,
    transfer_scope: str,
    cheapest_falsifier: str,
    distinguish_experiment: str,
    evidence: str,
    physical_consequence: Mapping[str, str],
) -> dict[str, Any]:
    return admit_observation(
        {
            "id": oid,
            "claim": claim,
            "classification": classification,
            "transfer_scope": transfer_scope,
            "cheapest_falsifier": cheapest_falsifier,
            "distinguish_experiment": distinguish_experiment,
            "evidence": evidence,
            "physical_consequence": dict(physical_consequence),
        }
    )


def admit_observation(obs: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse an invariant with no falsifier, and refuse confused scopes.

    A guard nobody has watched fail is not a guard. Tests plant a
    PHYSICAL_INVARIANT with an empty falsifier and require ClaimRefused.
    Classification-specific refusals fire before the generic missing-field
    check so the watched-fail message is the one the receipt records.
    """
    classification = str(obs.get("classification") or "")
    scope = str(obs.get("transfer_scope") or "")
    if classification == "PHYSICAL_INVARIANT" and not str(obs.get("cheapest_falsifier") or "").strip():
        raise ClaimRefused(
            f"{obs.get('id')}: PHYSICAL_INVARIANT with no falsifier is refused"
        )
    if classification == "METAL_ACCIDENT" and not str(obs.get("distinguish_experiment") or "").strip():
        raise ClaimRefused(
            f"{obs.get('id')}: METAL_ACCIDENT with no distinguishing experiment "
            "cannot be told from an invariant"
        )
    if classification == "METAL_ACCIDENT" and scope not in ACCIDENT_SCOPES:
        raise ClaimRefused(
            f"{obs.get('id')}: METAL_ACCIDENT cannot have invariant scope {scope}"
        )
    if classification == "PHYSICAL_INVARIANT" and scope not in INVARIANT_SCOPES:
        raise ClaimRefused(
            f"{obs.get('id')}: PHYSICAL_INVARIANT cannot have accident scope {scope}"
        )
    missing = [f for f in OBSERVATION_FIELDS if f not in obs or obs.get(f) in (None, "", [], {})]
    if missing:
        raise ClaimRefused(f"{obs.get('id')}: missing {missing}")
    if classification not in CLASSIFICATIONS:
        raise ClaimRefused(f"{obs.get('id')}: unknown classification {classification}")
    if scope not in TRANSFER_SCOPES:
        raise ClaimRefused(f"{obs.get('id')}: unknown transfer_scope {scope}")
    if not str(obs.get("distinguish_experiment") or "").strip():
        raise ClaimRefused(
            f"{obs.get('id')}: every observation needs a distinguishing experiment"
        )
    pc = obs.get("physical_consequence") or {}
    miss_pc = [k for k in CONSEQUENCE_KEYS if k not in pc]
    if miss_pc:
        raise ClaimRefused(f"{obs.get('id')}: physical_consequence missing {miss_pc}")
    row = {k: obs[k] for k in OBSERVATION_FIELDS}
    row["evidence_class"] = "STATIC_ONLY"
    return row


def observations() -> list[dict[str, Any]]:
    """Load-bearing invariant vs accident judgements."""
    sparse = _consequence(
        "O_K_EXPERT_PAYLOADS_PLUS_SHARED",
        "STATIC_ONE_OR_BOUNDED_TOPK",
        "ROUTE_THEN_BODY",
        "SELECTED_TILES_NOT_WHOLE_BANK",
    )
    rows = [
        _obs(
            "INV-SPARSE-ACTIVE-BYTES",
            "Sparse MoE that gathers selected tiles bills O(k) expert-weight "
            "bytes per token, not O(E). Dense-with-mask is a different form "
            "and pays O(E).",
            "PHYSICAL_INVARIANT",
            "MOE_FAMILY",
            "A gather-of-k executor whose active expert-weight traffic matches "
            "an all-E dense-mask executor on the same bank (byte identity, not "
            "timing) refutes sparsity of the gather form.",
            "Re-implement gather vs dense-mask on CUDA grouped GEMM and on an "
            "FPGA spatial pipeline. If both still pay O(k) vs O(E) in bytes, "
            "the claim is not a Metal accident. If only Metal's argument "
            "buffers skip unselected tiles, demote to METAL_BACKEND.",
            "FLASH_ORGAN_CENSUS routed_experts fraction; layer-46 kernels take "
            "route_ids rather than the 512-expert bank.",
            sparse,
        ),
        _obs(
            "INV-SHARED-ACTIVATION-DECODE",
            "Decode applies the same hidden vector x to every selected expert. "
            "That identity is why compute-sharing of x is even speakable. "
            "Prefill (distinct x per row) is a different skeleton.",
            "PHYSICAL_INVARIANT",
            "MOE_FAMILY",
            "Exhibit a decode token whose selected experts consume different "
            "activations in the source graph.",
            "If a CUDA/FPGA decode path must re-load a distinct x per expert "
            "for ABI reasons while the math still uses one x, that reload is "
            "a backend accident; the identity remains.",
            "layer-46 input mlp_input is shared across router, expert_gate_up, "
            "shared expert, and scalar. COMPUTE-ONE-X-MANY-EXPERTS.",
            sparse,
        ),
        _obs(
            "INV-LINEAR-WEIGHTED-COMBINE",
            "Flash's routed combine is a linear weighted sum "
            "(qwen_next_moe_weighted_sum). When a shared expansion U exists, "
            "sum π_e (U u_e) = U (sum π_e u_e) is free algebra. It is not "
            "free when U is per-expert, which is the layer-46 graph.",
            "PHYSICAL_INVARIANT",
            "MOE_FAMILY",
            "If the combine kernel is not linear in the expert outputs (a "
            "non-linear mixer, a learned combiner, a top-1 hard select with "
            "no weights), the identity does not apply. Kill the latent-reduce "
            "candidate on that parent.",
            "Linearity is algebra, not Metal. A CUDA or FPGA combiner that "
            "implements the same weighted sum has the same identity. A "
            "backend that expands then sums because it lacks a latent U is "
            "an implementation accident of missing U, not of Metal.",
            "layer-46 qwen_next_moe_weighted_sum; COMPUTE-LATENT-WEIGHTED-REDUCTION.",
            _consequence(
                "COMBINE_AT_D_MODEL_TODAY",
                "STATIC_ONE",
                "AFTER_EXPERT_BODY",
                "ROUTED_OUTPUTS_THEN_SUM",
            ),
        ),
        _obs(
            "INV-BANK-STATIC-EXPERT-ID",
            "The expert bank's node set is compile-time. Which expert runs is "
            "an EXPERT_ID slot bind, not a graph rewrite. Shared expert is "
            "STATIC and not EXPERT_ID-gated.",
            "PHYSICAL_INVARIANT",
            "MOE_FAMILY",
            "A graph whose node or edge set changes with the route (VALUE_GATED "
            "existence) refutes this. static_skeleton.validate is the cheap "
            "check; a runtime that adds launches beyond dispatch_bound is the "
            "physical check.",
            "If Metal indirect command buffers hide a value-gated node set "
            "while CUDA graphs cannot capture it, the VALUE_GATED form is "
            "still illegal for replay — that is general, not Metal. What "
            "would be a Metal accident is a host-side graph rebuild that "
            "looks like topology change but is only encoder recording.",
            "static_skeleton.SLOT_KINDS; flash_moe_component_skeleton; "
            "layer-46 all 35 rows exist independent of route_ids.",
            _consequence(
                "O_K_OF_CAPTURED_BANK",
                "CAPTURED_BOUND",
                "REPLAY_WITH_SLOT_BIND",
                "BANK_RESIDENT",
            ),
        ),
        _obs(
            "INV-ROUTE-BEFORE-PAYLOAD",
            "A sparse consume of k of E expert payloads requires a route "
            "decision first (or it is dense-with-mask). Flash's meta contract "
            "states route_before_payload: true.",
            "PHYSICAL_INVARIANT",
            "MOE_FAMILY",
            "A kernel that never reads a route id / mask and still touches "
            "only k experts' bytes — that would require an oracle, and "
            "refutes 'route is the thing that selects bytes'.",
            "Payload-then-select that still reads all E is the dense form, "
            "portable. A Metal-only prefetch of the whole bank 'just in "
            "case' is a backend accident sitting on top of the invariant.",
            "FLASH_META accelerator_contract.route_before_payload; layer-46 "
            "order router gemv -> moe_topk_gate -> expert kernels.",
            sparse,
        ),
        _obs(
            "INV-SHARED-EXPERT-ALWAYS-ON",
            "Flash's shared expert is an always-on organ. "
            "shared_expert_sigmoid_is_not_router_selection is true. It is "
            "not an 513th routed expert.",
            "PHYSICAL_INVARIANT",
            "FLASH_MODEL",
            "A configuration where the shared-expert sigmoid is wired into "
            "router top-k, or where omitting the shared expert changes "
            "selected ids, refutes the flag on that parent.",
            "Other MoE parents have no shared expert; that is model family, "
            "not Metal. A Metal implementation that launched the shared "
            "expert with the routed wave would be a backend accident; the "
            "source flag would still say it is not router selection.",
            "FLASH_NOETIC_ROUTER_SELECTION config.router; layer-46 shared "
            "kernels have no route_ids in their inputs.",
            _consequence(
                "O_ONE_SHARED_EXPERT_PLUS_ROUTED_K",
                "STATIC_ALWAYS_ON",
                "PARALLEL_TO_ROUTED_THEN_ADD",
                "SHARED_WEIGHTS_RESIDENT",
            ),
        ),
        _obs(
            "INV-TOPK-IS-A-BOUND",
            "num_experts_per_tok is a compile-time bound (10 on Flash). "
            "Adaptive unbounded k (score threshold with no max) is "
            "VALUE_GATED topology unless a max-k slot bound is declared.",
            "PHYSICAL_INVARIANT",
            "FLASH_MODEL",
            "A Flash config or runtime whose selected-expert count exceeds "
            "num_experts_per_tok, or a parent with unbounded variable-k and "
            "no dispatch_bound, refutes 'k is a captured bound' on that graph.",
            "Metal indirect-command-buffer draw counts that vary with k "
            "inside a declared bound are still SLOT_INDEXED. A host that "
            "rebuilds the encoder when k changes, even though k <= bound, "
            "is a Metal/recording accident.",
            "FLASH_NOETIC_ROUTER_SELECTION config.router.num_experts_per_tok; "
            "static_skeleton Slot.dispatch_bound.",
            _consequence(
                "O_K_LEQ_BOUND",
                "BOUNDED_BY_TOPK",
                "CAPTURED",
                "BANK_INDEXED",
            ),
        ),
        _obs(
            "INV-DISPATCH-COUNT-IS-NOT-COST",
            "A dispatch is not a unit of cost. Two graphs with the same "
            "launch count can differ in arithmetic, movement, and occupancy. "
            "This school therefore refuses to rank ceremony by subtracting "
            "launches.",
            "PHYSICAL_INVARIANT",
            "GENERAL_PHYSICAL",
            "Two captured graphs, equal dispatch counts, identical bytes and "
            "arithmetic, whose protected complete-token always matches within "
            "noise on every backend, would make dispatch count a sufficient "
            "cost unit and refute the law.",
            "The cited receipt is Metal Qwen3.8. Re-run the equal-dispatch "
            "unequal-cost pair under CUDA graphs and (when a board exists) "
            "an FPGA spatial pipeline. If only Metal command-buffer "
            "accounting shows the split, demote this observation to "
            "METAL_ACCIDENT. If the split survives, it stays GENERAL_PHYSICAL.",
            "ACCELERATOR_DISPATCH_IS_NOT_THE_COST FINDING_3 qualitative law "
            "(magnitudes not restated).",
            _consequence(
                "NOT_PREDICTED_BY_LAUNCH_COUNT",
                "NOT_A_UNIT_OF_COST",
                "NOT_A_UNIT_OF_COST",
                "BYTES_AND_ARITHMETIC_DOMINATE",
            ),
        ),
        _obs(
            "ACC-LAYER46-KERNEL-SPLITS",
            "The layer-46 routed path is still many kernels (router gemv, "
            "top-k, expert gate_up_swiglu, expert down, weighted sum, six "
            "shared-expert kernels, add_shared) inside one TokenCommandBuffer. "
            "Every one of those rows carries a fusion_candidate. The splits "
            "are the current Metal encoder/kernel ABI, not MoE algebra.",
            "METAL_ACCIDENT",
            "THIS_GRAPH",
            "A fused routed organ (gate/up/SwiGLU/down + combine, and a fused "
            "shared-expert body) that fails NumericParity against the split "
            "graph, or a protected complete-token that does not move after "
            "the split is removed, means the split was carrying real "
            "association / occupancy work, not just ABI.",
            "Lower the same algebraic sequence (route, gather k, SwiGLU, "
            "down, weighted sum, shared SwiGLU, add) as one CUDA kernel / "
            "one FPGA region. If those backends still require the same "
            "splits for hazards, promote toward MOE_FAMILY. If they fuse "
            "and match parity, the splits stay a Metal/this-graph accident.",
            "FLASH_LAYER46_DISPATCH_LEDGER fusion_candidate column on routed "
            "rows; queue flash-router-topk-fusion, flash-routed-fp4-gate-up-"
            "swiglu-fused, flash-compact-moe-epilogue.",
            _consequence(
                "UNCHANGED_IF_FUSED",
                "STATIC_MANY_TODAY_STATIC_FEW_IF_FUSED",
                "ONE_TOKENCOMMANDBUFFER_ALREADY",
                "STAGED_INTERMEDIATES_NAMED_BY_FUSION_CANDIDATES",
            ),
        ),
        _obs(
            "ACC-P6-DUAL-COMMANDBUFFER",
            "P6 hash-route historically used two command buffers with a "
            "CPU-visible commit/wait between up/SwiGLU and down/combine. "
            "Layer-46 source-oracle already uses one TokenCommandBuffer. "
            "The dual-CB wait is a Metal recording accident of that path.",
            "METAL_ACCIDENT",
            "METAL_BACKEND",
            "flash-p6-hash-single-command-buffer: a single CB that fails "
            "Metal hazards or NumericParity keeps the wait. A single CB "
            "that matches parity with 60 dispatches unchanged (the candidate's "
            "own expected_dispatch_reduction) confirms the wait was ceremony.",
            "CUDA graphs capture both waves as one replay with events, not "
            "a host wait. An FPGA pipeline has no command buffer. If those "
            "backends still need a host-visible barrier between up and down, "
            "promote. Otherwise it stays Metal.",
            "queue flash-p6-hash-single-command-buffer expected_eliminated_work.",
            _consequence(
                "UNCHANGED",
                "DISPATCHES_UNCHANGED_CB_COUNT_IS_CEREMONY",
                "HOST_COMMIT_WAIT_VS_SINGLE_CB",
                "BUFFERS_ALREADY_RESIDENT",
            ),
        ),
        _obs(
            "ACC-P6-PER-EXPERT-LAUNCHES",
            "P6 hash-route launches selected experts as independent W1/W3/"
            "SwiGLU/cast/W2 waves (historical 60-dispatch hash graph). The "
            "same algebra on layer-46 BF16 is two kernels. Per-expert "
            "launches are an implementation of independence, not a physical "
            "requirement of MoE.",
            "METAL_ACCIDENT",
            "THIS_GRAPH",
            "Fused-six indirect launch (flash-p6-routed-fp4-gate-up-swiglu-fused, "
            "flash-p6-routed-fp4-down-bf16-fused) fails exact-order FP4 "
            "reduction parity.",
            "CUDA grouped GEMM and FPGA spatial multiplex of six experts "
            "with EXPERT_ID slots. If those still need six launches for "
            "the same packed payload, promote toward MOE_FAMILY occupancy. "
            "If one grouped launch matches parity, it stays this-graph.",
            "queue flash-p6-routed-fp4-* expected_eliminated_work.",
            _consequence(
                "UNCHANGED_SIX_PAYLOADS",
                "ONE_PER_SELECTED_TODAY_STATIC_ONE_IF_FUSED",
                "CONCURRENT_WAVES_OR_ONE_INDIRECT",
                "PER_EXPERT_QAT_BUFFERS_AS_CONTRACT",
            ),
        ),
        _obs(
            "ACC-ROUTER-TOPK-KERNEL-SPLIT",
            "Router logits (gemv_native_bf16_seq) and moe_topk_gate are two "
            "kernels. Top-k is an epilogue of the gemv. The split is encoder "
            "ABI, not a MoE invariant.",
            "METAL_ACCIDENT",
            "METAL_BACKEND",
            "flash-router-topk-fusion / flash-hc-router-topk-fusion: fused "
            "select that breaks the exact tie / top-k contract (norm_topk_prob, "
            "num_experts_per_tok=10) is a real reason to keep the split.",
            "Any backend that can fuse a gemv epilogue (CUDA, FPGA reduction "
            "tree) can host the same top-k. If fused top-k is illegal on "
            "those backends for the same tie contract, promote. Else Metal.",
            "layer-46 rows router_logits then route_ids; queue flash-router-topk-fusion.",
            _consequence(
                "ROUTE_METADATA_ONLY",
                "STATIC_TWO_TODAY_STATIC_ONE_IF_FUSED",
                "ROUTE_DEPENDENCY_KEPT",
                "LOGITS_STAGING_REMOVABLE",
            ),
        ),
        _obs(
            "ACC-P6-FP32-STAGING-CASTS",
            "P6 routed FP4 historically writes FP32 then launches FP32-to-BF16 "
            "casts as separate dispatches. Accumulation dtype is a kernel "
            "choice; Metal does not require a second launch for the cast.",
            "METAL_ACCIDENT",
            "THIS_GRAPH",
            "Fused BF16 write that fails the numeric contract of the FP32 "
            "staging path keeps the cast as arithmetic, not ceremony.",
            "CUDA and FPGA fused kernels routinely accumulate in a wider "
            "register type and write BF16 without a cast dispatch. If they "
            "cannot match the exact association of the FP32 staging, the "
            "cast was carrying association (closer to invariant of THAT "
            "reduction). If they match, it was this-graph ABI.",
            "queue flash-p6-routed-fp4-down-bf16-fused and "
            "flash-routed-fp4-gate-up-swiglu-fused expected_eliminated_work.",
            _consequence(
                "UNCHANGED_WEIGHTS_FEWER_TRANSIENTS",
                "CAST_LAUNCHES_REMOVABLE",
                "IN_KERNEL_ROUNDTRIP",
                "FP32_STAGING_BYPASSABLE",
            ),
        ),
        _obs(
            "ACC-HOST-LEARNED-ROUTE-ADMISSION",
            "P6 learned-route reader reuse and expert source-cache eliminate "
            "host admission / chunk materialization, not device topology. "
            "A resident bank with an EXPERT_ID bind has no such ceremony.",
            "METAL_ACCIDENT",
            "METAL_BACKEND",
            "The candidates already name the falsifier: route overlap and "
            "complete-token wall time (later lane, protected). If overlap "
            "does not change complete-token, the ceremony was idle.",
            "A CUDA persistent module or FPGA-resident bank never pays "
            "sealed-reader manifest admission per changed route. If a "
            "discrete-GPU path still needs a host cache because PCIe is "
            "the wall, that is a different backend accident (movement), "
            "not MoE algebra.",
            "queue flash-p6-learned-reader-reuse, flash-p6-learned-expert-cache-reuse.",
            _consequence(
                "UNCHANGED_DEVICE_BYTES",
                "DEVICE_TOPOLOGY_UNCHANGED",
                "HOST_ADMISSION_ELIDABLE",
                "BOUNDED_SIX_BUNDLE_CACHE",
            ),
        ),
        _obs(
            "ACC-SIMDGROUP-OCCUPANCY",
            "flash-p6-*-simd candidates change SIMDgroup / vec4 occupancy, "
            "not topology, not bytes, not dispatch count. Occupancy geometry "
            "is Metal-specific.",
            "METAL_ACCIDENT",
            "METAL_BACKEND",
            "Independent A/B of simd vs scalar fusion (already named on the "
            "candidates) with NumericParity. If occupancy does not move "
            "protected complete-token, the geometry change is idle.",
            "CUDA warps and FPGA lanes are different occupancy objects. A "
            "win that does not transfer is METAL_BACKEND by construction. "
            "A win that transfers as 'split the reduction across a warp/"
            "wavefront/lane group' would be GENERAL_PHYSICAL occupancy, "
            "and the distinguish experiment is that transfer.",
            "queue flash-p6-routed-fp4-*-simd, flash-compact-moe-bf16-vec4, "
            "flash-p6-act-quant-simdgroup.",
            _consequence(
                "UNCHANGED",
                "UNCHANGED",
                "UNCHANGED",
                "UNCHANGED_ABI",
            ),
        ),
        _obs(
            "ACC-NOT-A-LAYER-MEGAKERNEL",
            "Fusing the routed organ is not a full-layer MoE megakernel. "
            "The negative index records full-layer megakernel-as-TPS-lever "
            "as dead (including a fused expert-wave collapse). This school "
            "does not reopen that scar by renaming kernel splits.",
            "METAL_ACCIDENT",
            "THIS_GRAPH",
            "A 'fuse the whole layer-46 TokenCommandBuffer into one kernel' "
            "proposal that claims TPS without a new parent and without "
            "addressing the recorded megakernel scar. refuse_if_dead on "
            "hypothesis_family=megakernel.",
            "If a CUDA megakernel of the whole Flash layer ever beats the "
            "fused-organ graph on protected complete-token AND capability, "
            "the scar's transfer scope was MODEL_SPECIFIC and can reopen "
            "on that parent. Until then, whole-layer fusion stays dead; "
            "organ fusion stays a different, still-legal object.",
            "negative_index family megakernel; layer-46 still has attention/"
            "DeltaNet/HC rows that are not the routed organ.",
            _consequence(
                "N_A",
                "NOT_THE_LEVER",
                "NOT_THE_LEVER",
                "NOT_THE_LEVER",
            ),
        ),
    ]
    rows.sort(key=lambda r: r["id"])
    return rows


DEAD_PROBES: tuple[dict[str, Any], ...] = (
    {
        "id": "PROBE-INVARIANT-WITHOUT-FALSIFIER",
        "claim": "routed MoE is always faster fused",
        "classification": "PHYSICAL_INVARIANT",
        "transfer_scope": "MOE_FAMILY",
        "cheapest_falsifier": "",
        "distinguish_experiment": "run it on CUDA",
        "evidence": "none",
        "physical_consequence": _consequence("x", "y", "z", "w"),
    },
    {
        "id": "PROBE-ACCIDENT-WITHOUT-DISTINGUISH",
        "claim": "35 layer-46 launches are a Metal accident",
        "classification": "METAL_ACCIDENT",
        "transfer_scope": "METAL_BACKEND",
        "cheapest_falsifier": "fused parity fails",
        "distinguish_experiment": "",
        "evidence": "ledger",
        "physical_consequence": _consequence("x", "y", "z", "w"),
    },
    {
        "id": "PROBE-ACCIDENT-WITH-INVARIANT-SCOPE",
        "claim": "Metal command buffers are a law of MoE",
        "classification": "METAL_ACCIDENT",
        "transfer_scope": "GENERAL_PHYSICAL",
        "cheapest_falsifier": "a non-Metal backend still needs them",
        "distinguish_experiment": "CUDA graphs",
        "evidence": "none",
        "physical_consequence": _consequence("x", "y", "z", "w"),
    },
)


def refusal_controls() -> list[dict[str, Any]]:
    rows = []
    for probe in DEAD_PROBES:
        try:
            admit_observation(probe)
        except ClaimRefused as exc:
            rows.append(
                {
                    "probe_id": probe["id"],
                    "refused": True,
                    "reason": str(exc),
                }
            )
            continue
        rows.append(
            {
                "probe_id": probe["id"],
                "refused": False,
                "error": "GUARD_FAILED_TO_FIRE",
            }
        )
    return rows


def assert_guards_fire() -> list[dict[str, Any]]:
    rows = refusal_controls()
    failed = [r for r in rows if not r.get("refused")]
    if failed:
        raise RuntimeError(f"observation refusal did not fire: {failed}")
    return rows


def observed_flash_form(ledger: Mapping[str, Any], budget: Mapping[str, Any]) -> dict[str, Any]:
    acc = budget.get("accelerator_contract") or {}
    routed_kernels = _layer46_kernel_set(ledger)
    return {
        "geometry": "gather_scatter",
        "launch": "fused_routed_dispatch",
        "launch_qualifier": (
            "two static expert kernels (gate_up_swiglu, down) plus a static "
            "shared-expert path; not per-expert launches, not dense-with-mask."
        ),
        "order": "route_before_payload",
        "skeleton": "static_skeleton_expert_id",
        "shared_expert": "static_always_on",
        "meta_route_before_payload": acc.get("route_before_payload"),
        "meta_dense_rematerialization": acc.get("dense_rematerialization"),
        "meta_runtime_shape": (budget.get("routed_experts") or {}).get("runtime_shape"),
        "routed_path_kernels": routed_kernels,
        "p6_hash_graph_is_a_different_observed_form": {
            "launch": "per_expert_dispatch",
            "synchronization": "dual_command_buffer_host_wait",
            "source": "qualification queue flash-p6-* exact_mutation / expected_eliminated_work",
        },
    }


# ---------------------------------------------------------------------------
# Recovery / gaps / negatives
# ---------------------------------------------------------------------------


def recovered_implementation(evidence: Mapping[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "path": "tools/future/expert_bank_school.py",
            "role": (
                "STORAGE and COMPUTE sharing candidates, including "
                "repeated_computation / why_currently_repeated. scar refusal "
                "for raw global similarity, trivial shared basis, unchanged archetypes."
            ),
            "adequate": "for sharing hypotheses, not for physical execution taxonomy",
            "gap_this_school_closes": (
                "does not classify gather/scatter vs dense-mask vs sorted, "
                "does not partition layer-46 ceremony vs arithmetic, does not "
                "judge invariant vs Metal accident."
            ),
        },
        {
            "path": "tools/future/static_skeleton.py",
            "role": (
                "EXPERT_ID is one of five permitted dynamic slots; "
                "flash_moe_component_skeleton; VALUE_GATED refusal."
            ),
            "adequate": "for replayability of SLOT_INDEXED banks",
            "gap_this_school_closes": (
                "does not walk the layer-46 routed path or the P6 candidate "
                "queue as an execution school."
            ),
        },
        {
            "path": "tools/future/router_science.py",
            "role": "route-margin precision allocation (CONTROL_FLOW_PREMIUM bits).",
            "adequate": "for which bits preserve routing, not for how routed compute is scheduled",
            "gap_this_school_closes": "execution order, launch topology, ceremony.",
        },
        {
            "path": "tools/future/negative_index.py",
            "role": "keyed scars; MoE sharing negatives are numerous.",
            "adequate": "as the query path this school calls before speaking of sharing",
            "gap_this_school_closes": "none duplicated; queried, not forked.",
        },
        {
            "path": "tools/future/physical_primitives.py",
            "role": "PersistentPhysicalRegion / StationaryRepresentation host_ceremony metrics.",
            "adequate": "as primitive contracts; not a MoE schedule.",
            "gap_this_school_closes": "binds those primitives to the routed path.",
        },
        {
            "path": "receipts/future/evidence/FLASH_LAYER46_DISPATCH_LEDGER.json",
            "role": "real dispatch topology for one Flash layer (35 rows, trace off).",
            "adequate": "as the source-oracle graph this school reads",
            "evidence_source": _source_of(evidence, "FLASH_LAYER46_DISPATCH_LEDGER.json"),
        },
        {
            "path": "receipts/future/evidence/FLASH_ORGAN_CENSUS.json",
            "role": "routed_experts is the dominant byte family.",
            "adequate": "as the byte ground",
            "evidence_source": _source_of(evidence, "FLASH_ORGAN_CENSUS.json"),
        },
        {
            "path": "receipts/future/evidence/FLASH_META_REPRESENTATION_SUB1.json",
            "role": "family_budget + accelerator_contract.route_before_payload.",
            "adequate": "as the intended runtime_shape",
            "evidence_source": _source_of(evidence, "FLASH_META_REPRESENTATION_SUB1.json"),
        },
        {
            "path": "receipts/future/evidence/ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
            "role": "flash-p6-*, flash-routed-*, flash-compact-moe-*, flash-hc-*, flash-router-topk-fusion.",
            "adequate": "as the live candidate list (count derived, not pinned)",
            "evidence_source": _source_of(evidence, "ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json"),
        },
        {
            "path": "receipts/future/evidence/ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json",
            "role": "qualitative law: a dispatch is not a unit of cost.",
            "adequate": "as a law citation; magnitudes are Qwen3.8 Metal and are not restated",
            "evidence_source": _source_of(evidence, "ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json"),
        },
    ]


def gaps_closed() -> list[str]:
    return [
        "execution taxonomy for routed expert compute (geometry, launch, order, skeleton) with physical consequences",
        "repeated-work census grounded in the organ census, layer-46 ledger, and doctor cosine, including the fact that layer-46 BF16 already fused selected-expert launches",
        "ceremony vs work partition of the routed path from the real ledger, citing that a dispatch is not a unit of cost",
        "invariant vs Metal accident with transfer scope and a distinguishing experiment on every observation",
        "cheapest falsifier on every claim; refusal of PHYSICAL_INVARIANT with no falsifier (watched-fail probes)",
        "cross-reference to static_skeleton: which forms replay with only EXPERT_ID and which are VALUE_GATED",
        "routed/MoE qualification-queue subset derived from prefixes, not a hard-coded candidate count",
    ]


def negative_findings(evidence: Mapping[str, dict[str, Any]], negatives: Mapping[str, Any]) -> list[str]:
    findings = [
        "Did not take a hardware measurement. gpu_ns, token_ns, tps, joules, bandwidth remain UNKNOWN. per-owner critical-path ns is unpopulated (trace mode off) and was not filled in.",
        "Did not run the distinguishing CUDA / FPGA experiments that would promote or demote METAL_ACCIDENT observations. They are named, not executed. FPGA is Accelerator, not its own civilization; this school does not build an FPGA backend.",
        "Did not fit weights or reopen expert-bank STORAGE candidates. Doctor L44 cosine is cited as a structure metric already on disk.",
        "Did not treat dispatch-count reductions in the qualification queue as measurements; they are candidate hypotheses labelled expected_*.",
        "Did not reopen megakernel-as-TPS-lever (negative index). Organ fusion is a different object.",
        "Did not produce DIAGNOSTIC_RELATIVE or PROTECTED_ABSOLUTE evidence. Everything here is STATIC_ONLY.",
    ]
    unreachable = [
        name
        for name, row in evidence.items()
        if not row.get("reachable")
    ]
    if unreachable:
        findings.append(
            "Evidence files not reachable in this checkout (coped; not treated "
            "as absent from the campaign): " + ", ".join(sorted(unreachable))
        )
    if not negatives.get("index_reachable"):
        findings.append(str(negatives.get("coped") or "negative index unreachable"))
    live_used = [
        name
        for name, row in evidence.items()
        if row.get("evidence_source") == "live_headless"
    ]
    if live_used:
        findings.append(
            "Fell back to live headless for: " + ", ".join(sorted(live_used))
        )
    return findings


def evidence_source_block(evidence: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    per = {
        name: {
            "evidence_source": row.get("evidence_source"),
            "reachable": row.get("reachable"),
            "path": row.get("path"),
            "sha256": row.get("sha256"),
        }
        for name, row in evidence.items()
    }
    sources = {row.get("evidence_source") for row in evidence.values()}
    if sources <= {"pinned_snapshot"}:
        overall = "pinned_snapshot"
    elif sources <= {"pinned_snapshot", "unavailable"}:
        overall = "pinned_snapshot"
    elif "live_headless" in sources and "pinned_snapshot" not in sources:
        overall = "live_headless"
    elif "live_headless" in sources:
        overall = "mixed_pinned_and_live"
    else:
        overall = "unavailable"
    return {"overall": overall, "per_input": per}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_document() -> dict[str, Any]:
    evidence = load_all_evidence()
    census = organ_census_grounding(
        _doc(evidence, "FLASH_ORGAN_CENSUS.json"),
        _source_of(evidence, "FLASH_ORGAN_CENSUS.json"),
    )
    budget = family_budget_grounding(
        _doc(evidence, "FLASH_META_REPRESENTATION_SUB1.json"),
        _source_of(evidence, "FLASH_META_REPRESENTATION_SUB1.json"),
    )
    ledger = ledger_grounding(
        _doc(evidence, "FLASH_LAYER46_DISPATCH_LEDGER.json"),
        _source_of(evidence, "FLASH_LAYER46_DISPATCH_LEDGER.json"),
    )
    path10 = critical_path_owners(
        _doc(evidence, "FLASH_LAYER10_CRITICAL_PATH.json"),
        _source_of(evidence, "FLASH_LAYER10_CRITICAL_PATH.json"),
        "FLASH_LAYER10_CRITICAL_PATH.json",
    )
    path30 = critical_path_owners(
        _doc(evidence, "FLASH_LAYER30_CRITICAL_PATH.json"),
        _source_of(evidence, "FLASH_LAYER30_CRITICAL_PATH.json"),
        "FLASH_LAYER30_CRITICAL_PATH.json",
    )
    router = router_config_grounding(
        _doc(evidence, "FLASH_NOETIC_ROUTER_SELECTION.json"),
        _source_of(evidence, "FLASH_NOETIC_ROUTER_SELECTION.json"),
    )
    doctor = doctor_screen_grounding(
        _doc(evidence, "FLASH_DOCTOR_EXPERT_BANK_SCREEN_FULL_L44.json"),
        _source_of(evidence, "FLASH_DOCTOR_EXPERT_BANK_SCREEN_FULL_L44.json"),
    )
    cost_law = dispatch_cost_law(
        _doc(evidence, "ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json"),
        _source_of(evidence, "ACCELERATOR_DISPATCH_IS_NOT_THE_COST.json"),
    )
    queue_doc = _doc(evidence, "ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json")
    routed_cands = select_routed_candidates(queue_doc)
    negatives = consult_moe_negatives()
    controls = assert_guards_fire()
    obs = observations()
    n_inv = sum(1 for o in obs if o["classification"] == "PHYSICAL_INVARIANT")
    n_acc = sum(1 for o in obs if o["classification"] == "METAL_ACCIDENT")
    if n_acc < 1:
        raise RuntimeError("school emitted no METAL_ACCIDENT observation")
    if n_inv < 1:
        raise RuntimeError("school emitted no PHYSICAL_INVARIANT observation")
    xref = skeleton_cross_reference()
    taxonomy = execution_taxonomy()
    repeated = repeated_work_census(census, budget, ledger, doctor, router)
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "purpose": (
            "Physical-execution school for Flash routed expert compute: "
            "taxonomy, repeated-work census, ceremony vs work, invariant vs "
            "Metal accident, falsifiers, and EXPERT_ID replayability. "
            "Extends expert_bank_school and static_skeleton; does not fork them."
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
        "measurement_classes": {
            "this_module": "STATIC_ONLY",
            "DIAGNOSTIC_RELATIVE": "not produced",
            "PROTECTED_ABSOLUTE": "not produced",
            "bench_state": "UNKNOWN",
            "gpu_authority": False,
        },
        "evidence_source": evidence_source_block(evidence),
        "grounding": {
            "organ_census": census,
            "family_budget": budget,
            "layer46_ledger": {
                k: ledger[k]
                for k in ledger
                if k != "rows"
            },
            "layer46_routed_rows": [r for r in ledger.get("rows") or [] if r.get("on_routed_path")],
            "critical_path_layer10": path10,
            "critical_path_layer30": path30,
            "router_config": router,
            "doctor_screen": doctor,
            "dispatch_cost_law": cost_law,
        },
        "execution_taxonomy": taxonomy,
        "observed_flash_form": observed_flash_form(ledger, budget),
        "repeated_work_census": repeated,
        "cited_compute_sharing": cited_compute_sharing(ledger),
        "ceremony_vs_work": ceremony_vs_work(ledger, path10, path30, cost_law),
        "observations": obs,
        "skeleton_cross_reference": xref,
        "routed_candidates": {
            "prefixes": list(ROUTED_ID_PREFIXES),
            "count": len(routed_cands),
            "count_is_derived": True,
            "queue_total_candidates": _read_count(
                len((queue_doc or {}).get("candidates") or []),
                "ACCELERATOR_PHYSICAL_QUALIFICATION_QUEUE.json",
                "len(candidates)",
            ),
            "rows": routed_cands,
        },
        "moe_sharing_negatives": negatives,
        "refusal_controls": controls,
        "counts": {
            "taxonomy_forms": len(taxonomy),
            "observations": len(obs),
            "physical_invariants": n_inv,
            "metal_accidents": n_acc,
            "repeated_work_items": len(repeated),
            "routed_candidates": len(routed_cands),
            "refusal_controls_fired": sum(1 for r in controls if r.get("refused")),
            "compute_kinds_cited": len(ebs.REQUIRED_COMPUTE_KINDS),
        },
        "recovered_implementation": recovered_implementation(evidence),
        "gaps_closed": gaps_closed(),
        "negative_findings": negative_findings(evidence, negatives),
    }


def build() -> Any:
    doc = build_document()
    return write_receipt(RECEIPT, doc, "tools/future/moe_physical_school.py")


def selftest() -> Any:
    return build()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
