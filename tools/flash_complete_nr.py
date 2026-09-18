#!/usr/bin/env python3
"""Compose the current complete-family Flash NR accounting frontier.

The inventory covers every Flash organ family, but no complete source-independent
NR exists yet. Exact source-BF16 remains an external control, and compact options
remain scoped candidates until their complete storage, direct execution, and
accepted-token effects are qualified. An NR is eligible to run directly through
Hawking only after its declared closure and direct runtime are qualified; an NX
is only an optional later machine specialization, so this file contains no
machine binding.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import time
from pathlib import Path

from nr_container import seal_payload_sha256_v1, validate
from foundry.query_gravity_methods import validate_execution_frontier


PPM_BITS_PER_BYTE = 8_000_000
DOMINANT_DENSITY_FAMILIES = ("routed_experts", "ngram_embedding")
DEFAULT_TARGET_EBPW = ("1.0", "0.75", "0.5", "0.25", "0.1", "0.05", "0.025", "0.01")
LOWRANK_ACTIVATION_SCREEN_SCHEMA = "hawking.odyssey.stacked_expert_lowrank_sparse_repair.v1"
SHARDED_LOOKUP_PQ_SCREEN_SCHEMA = "hawking.odyssey.sharded_lookup_pq_screen.v1"
PLE_ACCESS_TRACE_SCHEMA = "hawking.odyssey.ple_access_trace.v1"
PLE_SOURCE_OUTPUT_CONTROL_SCHEMA = "hawking.odyssey.ple_source_output_control.v1"
PLE_REFERENCE_FORMULA_ORACLE_SCHEMA = "hawking.odyssey.ple_reference_formula_oracle.v1"
PLE_LOOKUP_PQ_OUTPUT_SCREEN_SCHEMA = "hawking.odyssey.ple_lookup_pq_output_screen.v1"
PLE_ADDRESS_GENERATOR_SCREEN_SCHEMA = "hawking.odyssey.ple_address_generator_screen.v1"
PLE_CONTEXT_RESPONSE_SCREEN_SCHEMA = "hawking.odyssey.ple_context_response_screen.v1"
EXPERT_VECTOR_CODE_SCREEN_SCHEMA = "hawking.odyssey.expert_vector_code_screen.v1"
EXPERT_ATOM_SHARING_SCREEN_SCHEMA = "hawking.odyssey.expert_atom_sharing_screen.v1"
HC_CLOSURE_SCREEN_SCHEMA = "hawking.odyssey.hyperconnection_closure_screen.v1"
ROUTER_SOURCE_OUTPUT_SCREEN_SCHEMA = "hawking.odyssey.flash_router_source_output_screen.v1"
PLE_SHARD_LOCAL_PQ_OUTPUT_SCREEN_STATUS = "SOURCE_PLE_OUTPUT_SHARD_LOCAL_PQ_RATE_DISTORTION_ONLY"
PLE_ADDITIVE_PQ_OUTPUT_SCREEN_STATUS = "SOURCE_PLE_OUTPUT_ADDITIVE_PQ_RATE_DISTORTION_ONLY"
PLE_CONTEXT_RESPONSE_SCREEN_STATUS = "SOURCE_PLE_CONTEXT_RESPONSE_TANGENT_ONLY"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_ebpw_ppm(raw: str) -> tuple[str, int]:
    """Parse an EBPW target exactly enough to avoid a flattering byte budget."""
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise SystemExit(f"invalid --target-ebpw {raw!r}") from exc
    if not value.is_finite() or value <= 0:
        raise SystemExit(f"--target-ebpw must be positive and finite, got {raw!r}")
    ppm = value * Decimal(1_000_000)
    if ppm != ppm.to_integral_value():
        raise SystemExit(
            f"--target-ebpw {raw!r} cannot be represented in integer EBPW ppm; "
            "refusing to round the target"
        )
    return format(value.normalize(), "f"), int(ppm)


def target_budget(
    *,
    parameter_count: int,
    family_source_bytes: dict[str, int],
    raw_target: str,
) -> dict:
    """Return a whole-body rate budget, not a candidate representation claim.

    This is the Flash adapter for the same integer accounting law implemented
    in `NoeticRepresentationManifest::rate_budget`: EBPW is charged against
    original logical weights, and a fractional final byte is rounded *up*.
    """
    target_text, target_ppm = target_ebpw_ppm(raw_target)
    scaled = parameter_count * target_ppm
    target_bytes = (scaled + PPM_BITS_PER_BYTE - 1) // PPM_BITS_PER_BYTE
    dominant = {
        name: family_source_bytes[name]
        for name in DOMINANT_DENSITY_FAMILIES
    }
    dominant_source_bytes = sum(dominant.values())
    tail_source_bytes = sum(
        value for name, value in family_source_bytes.items()
        if name not in DOMINANT_DENSITY_FAMILIES
    )
    remaining_after_tail = target_bytes - tail_source_bytes
    dominant_weights = dominant_source_bytes // 2
    tail_ebpw_ppm = tail_source_bytes * PPM_BITS_PER_BYTE // parameter_count
    result = {
        "target_complete_ebpw": target_text,
        "target_complete_ebpw_ppm": target_ppm,
        "target_persistent_bytes_ceiling": target_bytes,
        "source_control_bytes_over_budget": max(0, sum(family_source_bytes.values()) - target_bytes),
        "dominant_families": list(DOMINANT_DENSITY_FAMILIES),
        "dominant_source_bytes": dominant_source_bytes,
        "tail_source_bytes_if_left_bf16": tail_source_bytes,
        "tail_source_global_ebpw_ppm_if_left_bf16": tail_ebpw_ppm,
        "tail_alone_exceeds_target": tail_source_bytes > target_bytes,
        "remaining_for_dominant_bytes_if_tail_left_bf16": max(0, remaining_after_tail),
        "dominant_local_ebpw_ppm_if_tail_left_bf16": (
            max(0, remaining_after_tail) * PPM_BITS_PER_BYTE // dominant_weights
            if dominant_weights else None
        ),
    }
    if tail_source_bytes > target_bytes:
        result["finding"] = (
            "The non-routed/non-ngram source tail alone exceeds this whole-body "
            "budget. A target candidate must transform that tail as well; improving "
            "only routed experts and n-gram embeddings cannot reach the target."
        )
    else:
        result["finding"] = (
            "Keeping the non-routed/non-ngram tail at BF16 is arithmetically "
            "possible but leaves the dominant routed-expert and n-gram families "
            "only the stated residual budget. This is a pressure signal, not a "
            "claim that such a local rate preserves capability."
        )
    return result


def optional_receipt(path: Path, expected_schema: str, label: str) -> dict | None:
    """Load a compatible optional discriminator or refuse a misleading binding."""
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{label} receipt is unreadable: {path}") from exc
    if not isinstance(document, dict) or document.get("schema") != expected_schema:
        raise SystemExit(f"{label} receipt has unexpected schema: {path}")
    return document


def _best_lowrank_activation_rows(document: dict | None) -> list[dict]:
    """Condense a source-activation screen into target-rate decisions."""
    if not document or document.get("status") != "SOURCE_ACTIVATION_RATE_DISTORTION_ONLY":
        return []
    rows = [
        row
        for row in document.get("rows", [])
        if isinstance(row, dict)
        and isinstance(row.get("source_activation_output"), dict)
        and isinstance(row.get("complete_scoped_ebpw"), (int, float))
        and isinstance(row["source_activation_output"].get("relative_l2"), (int, float))
    ]
    result = []
    for target in (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
        candidates = [row for row in rows if float(row["complete_scoped_ebpw"]) <= target]
        best = min(
            candidates,
            key=lambda row: float(row["source_activation_output"]["relative_l2"]),
            default=None,
        )
        result.append(
            {
                "target_scoped_ebpw": target,
                "best_candidate": (
                    {
                        "rank": best.get("rank"),
                        "residual_fraction": best.get("residual_fraction"),
                        "complete_scoped_ebpw": best.get("complete_scoped_ebpw"),
                        "weight_relative_l2": best.get("relative_l2"),
                        "source_activation_relative_l2": best["source_activation_output"].get("relative_l2"),
                        "source_activation_cosine": best["source_activation_output"].get("cosine"),
                    }
                    if best is not None
                    else None
                ),
                "status": "SOURCE_ACTIVATION_CONTROL_ONLY__NOT_A_REPRESENTATION_CLAIM",
            }
        )
    return result


def _ngram_pq_summary(document: dict | None) -> list[dict]:
    """Keep only the automatically reusable target decisions from a PQ screen."""
    if not document or document.get("status") != "SAMPLED_STATIC_RATE_DISTORTION_ONLY":
        return []
    summaries = document.get("target_summary")
    if not isinstance(summaries, list):
        return []
    return [item for item in summaries if isinstance(item, dict)]


def _ple_access_trace_summary(document: dict | None) -> dict | None:
    """Expose an address trace without confusing it for PLE-output evidence."""
    if (
        not document
        or document.get("status")
        != "SOURCE_ALGORITHM_BOUND_TOKEN_ACCESS_TRACE__NOT_PLE_OUTPUT_PARITY"
    ):
        return None
    active = document.get("active_lookup")
    sequence = document.get("sequence")
    if not isinstance(active, dict) or not isinstance(sequence, dict):
        return None
    fields = (
        "token_count",
        "lookup_events",
        "lookup_events_per_token",
        "unique_source_rows",
        "unique_source_bytes",
        "source_bytes_referenced_across_events",
    )
    if any(not isinstance(active.get(field), int) for field in fields):
        return None
    return {
        "status": document["status"],
        "sequence_binding": sequence.get("binding"),
        **{field: active[field] for field in fields},
        "boundary": "ADDRESS_AND_SOURCE_ROW_BINDING_ONLY__NOT_PLE_OUTPUT_OR_CAPABILITY",
    }


def _ple_source_output_control_summary(document: dict | None) -> dict | None:
    """Carry a sealed PLE formula control without laundering it into parity."""
    if (
        not document
        or document.get("schema") != PLE_SOURCE_OUTPUT_CONTROL_SCHEMA
        or document.get("status")
        != "SOURCE_BOUND_PLE_OUTPUT_CONTROL__NO_INDEPENDENT_PLE_OUTPUT_PARITY"
    ):
        return None
    input_control = document.get("input_control")
    active = document.get("active_lookup")
    outputs = document.get("outputs")
    post = outputs.get("post_injection_state") if isinstance(outputs, dict) else None
    if not isinstance(input_control, dict) or not isinstance(active, dict) or not isinstance(post, dict):
        return None
    fields = (
        "token_count",
        "lookup_events",
        "lookup_events_per_token",
        "unique_source_rows",
        "unique_source_bytes",
    )
    if any(not isinstance(active.get(field), int) for field in fields):
        return None
    return {
        "status": document["status"],
        "input_qualification": input_control.get("qualification"),
        "input_state_sha256": (input_control.get("state") or {}).get("sha256"),
        **{field: active[field] for field in fields},
        "post_injection_state_sha256": post.get("sha256"),
        "mutable_state_bytes": outputs.get("mutable_state_bytes"),
        "boundary": "SOURCE_FORMULA_INPUT_OUTPUT_CONTROL_ONLY__NOT_INDEPENDENT_PLE_PARITY_OR_CAPABILITY",
    }


def _ple_reference_formula_parity_summary(document: dict | None) -> dict | None:
    """Expose a bounded reference-framework formula pass without broadening scope."""
    if (
        not document
        or document.get("schema") != PLE_REFERENCE_FORMULA_ORACLE_SCHEMA
        or document.get("status") != "REFERENCE_FRAMEWORK_PLE_FORMULA_PARITY_PASS"
    ):
        return None
    metrics = document.get("metrics")
    source_control = document.get("source_control")
    if not isinstance(metrics, dict) or not isinstance(source_control, dict):
        return None
    fields = ("max_abs", "rmse", "relative_l2", "cosine", "finite")
    if not all(field in metrics for field in fields):
        return None
    return {
        "status": document["status"],
        "source_control_seal": source_control.get("seal_sha256"),
        **{field: metrics[field] for field in fields},
        "boundary": "ONE_SEALED_PLE_CONTROL_REFERENCE_FORMULA_PARITY_ONLY__NOT_FULL_CHECKPOINT_OR_CAPABILITY",
    }


def _ple_pq_output_screen_summary(
    document: dict | None, *, expected_status: str, boundary: str
) -> dict | None:
    """Carry reusable bounded decision rows without conflating PQ scopes."""
    if (
        not document
        or document.get("schema") != PLE_LOOKUP_PQ_OUTPUT_SCREEN_SCHEMA
        or document.get("status") != expected_status
    ):
        return None
    rows = document.get("target_summary")
    if not isinstance(rows, list):
        return None
    decisions = []
    for row in rows:
        best = row.get("best_candidate") if isinstance(row, dict) else None
        if not isinstance(best, dict):
            continue
        decisions.append(
            {
                "target_projected_table_ebpw": row.get("target_projected_table_ebpw"),
                "codebook_scope": best.get("codebook_scope"),
                "quantizer_family": best.get("quantizer_family"),
                "residual_stages": best.get("residual_stages"),
                "subdimension": best.get("subdimension"),
                "cardinality": best.get("cardinality"),
                "repair_count_per_row": best.get("repair_count_per_row"),
                "projected_complete_table_ebpw": best.get("projected_complete_table_ebpw"),
                "ple_output_relative_l2": best.get("ple_output_relative_l2"),
                "ple_output_cosine": best.get("ple_output_cosine"),
            }
        )
    return {
        "status": document["status"],
        "target_summary": decisions,
        "source_control_seal": ((document.get("source_control") or {}).get("seal_sha256")),
        "reference_parity_seal": ((document.get("reference_parity") or {}).get("seal_sha256")),
        "boundary": boundary,
    }


def _ple_lookup_pq_output_screen_summary(document: dict | None) -> dict | None:
    """Carry the global-PQ plus fixed-repair rate/output curve."""
    return _ple_pq_output_screen_summary(
        document,
        expected_status="SOURCE_PLE_OUTPUT_PQ_PLUS_SPARSE_REPAIR_RATE_DISTORTION_ONLY",
        boundary="ONE_SOURCE_PLE_OUTPUT_CONTROL_WITH_PROJECTED_GLOBAL_TABLE_ACCOUNTING__NOT_COMPLETE_PLE_OR_NR",
    )


def _ple_shard_local_pq_output_screen_summary(document: dict | None) -> dict | None:
    """Carry the physical-shard-local PQ rate/output curve separately."""
    return _ple_pq_output_screen_summary(
        document,
        expected_status=PLE_SHARD_LOCAL_PQ_OUTPUT_SCREEN_STATUS,
        boundary="ONE_SOURCE_PLE_OUTPUT_CONTROL_WITH_PROJECTED_SHARD_LOCAL_TABLE_ACCOUNTING__NOT_COMPLETE_PLE_OR_NR",
    )


def _ple_additive_pq_output_screen_summary(document: dict | None) -> dict | None:
    """Carry the greedy additive-PQ curve separately from source repair/PQ scopes."""
    return _ple_pq_output_screen_summary(
        document,
        expected_status=PLE_ADDITIVE_PQ_OUTPUT_SCREEN_STATUS,
        boundary="ONE_SOURCE_PLE_OUTPUT_CONTROL_WITH_PROJECTED_ADDITIVE_TABLE_ACCOUNTING__NOT_COMPLETE_PLE_OR_NR",
    )


def _ple_address_generator_screen_summary(
    document: dict | None, *, expected_family: str | None = None
) -> dict | None:
    """Carry the rate/output curve for one declared address-function family."""
    if (
        not document
        or document.get("schema") != PLE_ADDRESS_GENERATOR_SCREEN_SCHEMA
        or document.get("status") != "SOURCE_PLE_OUTPUT_GENERATOR_RATE_DISTORTION_ONLY"
    ):
        return None
    if expected_family is not None:
        representation = document.get("representation")
        if not isinstance(representation, dict) or representation.get("requested_family") != expected_family:
            return None
    rows = document.get("target_summary")
    if not isinstance(rows, list):
        return None
    decisions = []
    for row in rows:
        best = row.get("best_candidate") if isinstance(row, dict) else None
        if not isinstance(best, dict):
            continue
        decisions.append(
            {
                "target_projected_table_ebpw": row.get("target_projected_table_ebpw"),
                "family": best.get("family"),
                "configuration": best.get("configuration"),
                "feature_count": best.get("feature_count"),
                "projected_complete_table_ebpw": best.get("projected_complete_table_ebpw"),
                "heldout_row_relative_l2": best.get("heldout_row_relative_l2"),
                "ple_output_relative_l2": best.get("ple_output_relative_l2"),
                "ple_output_cosine": best.get("ple_output_cosine"),
            }
        )
    return {
        "status": document["status"],
        "target_summary": decisions,
        "source_control_seal": ((document.get("source_control") or {}).get("seal_sha256")),
        "reference_parity_seal": ((document.get("reference_parity") or {}).get("seal_sha256")),
        "boundary": "ONE_SOURCE_PLE_OUTPUT_ADDRESS_GENERATOR_CONTROL__NOT_COMPLETE_PLE_OR_NR",
    }


def _ple_additive_hash_function_screen_summary(document: dict | None) -> dict | None:
    """Keep the fixed additive-hash function prior separate from Fourier."""
    return _ple_address_generator_screen_summary(
        document,
        expected_family="additive-hash",
    )


def _expert_vector_code_screen_summary(document: dict | None) -> dict | None:
    """Carry a source-bound expert-family vector-code discriminator."""
    if (
        not document
        or document.get("schema") != EXPERT_VECTOR_CODE_SCREEN_SCHEMA
        or document.get("status") != "SOURCE_BOUND_EXPERT_VECTOR_CODE_FANOUT_COMPLETE"
    ):
        return None
    source = document.get("source")
    candidates = document.get("candidates")
    if not isinstance(source, dict) or not isinstance(candidates, list) or not candidates:
        return None
    summaries = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        accounting = candidate.get("accounting")
        heldout = candidate.get("heldout_weight_metrics")
        output = (candidate.get("source_activation_output") or {}).get("route_weighted_sum")
        variant = candidate.get("variant")
        if not all(isinstance(value, dict) for value in (accounting, heldout, output, variant)):
            continue
        summaries.append(
            {
                "variant": variant,
                "complete_scoped_ebpw": accounting.get("complete_scoped_ebpw"),
                "complete_bytes": accounting.get("complete_bytes"),
                "heldout_global_relative_l2": heldout.get("global_relative_l2"),
                "heldout_mean_row_cosine": heldout.get("mean_row_cosine"),
                "route_weighted_relative_l2": output.get("relative_l2"),
                "route_weighted_cosine": output.get("cosine"),
            }
        )
    if not summaries:
        return None
    best_density = min(summaries, key=lambda row: float(row["complete_scoped_ebpw"]))
    best_output = min(summaries, key=lambda row: float(row["route_weighted_relative_l2"]))
    return {
        "status": document["status"],
        "specimen": source.get("specimen"),
        "tensor": source.get("tensor"),
        "expert_ids": source.get("expert_ids"),
        "source_index_sha256": source.get("index_sha256"),
        "source_activation_sha256": (source.get("activation_control") or {}).get("activation_sha256"),
        "candidates": summaries,
        "best_density": best_density,
        "best_route_output": best_output,
        "boundary": "ONE_SOURCE_ACTIVATION_SELECTED_EXPERT_FAMILY_SCREEN__NOT_COMPLETE_NR_OR_CAPABILITY",
    }


def _expert_atom_sharing_screen_summary(document: dict | None) -> dict | None:
    """Carry the E02 lawful nonlinear atom-sharing discriminator."""
    if (
        not document
        or document.get("schema") != EXPERT_ATOM_SHARING_SCREEN_SCHEMA
        or document.get("status") != "SOURCE_BOUND_NONLINEAR_ATOM_SHARING_SCREEN_COMPLETE"
    ):
        return None
    source = document.get("source")
    candidate = document.get("candidate")
    if not isinstance(source, dict) or not isinstance(candidate, dict):
        return None
    accounting = candidate.get("accounting")
    outcome = candidate.get("outcome")
    source_output = candidate.get("source_activation_output")
    matching = candidate.get("matching")
    if not all(isinstance(value, dict) for value in (accounting, outcome, source_output, matching)):
        return None
    per_expert = source_output.get("per_expert")
    if not isinstance(per_expert, list) or len(per_expert) != 2:
        return None
    second = per_expert[1]
    if not isinstance(second, dict):
        return None
    selected = second.get("selected_triple_output")
    full = second.get("full_expert_output")
    if not isinstance(selected, dict) or not isinstance(full, dict):
        return None
    matched = selected.get("source")
    shuffled = selected.get("shuffled_null")
    matched_full = full.get("source")
    shuffled_full = full.get("shuffled_null")
    return {
        "status": document["status"],
        "specimen": source.get("specimen"),
        "gate_up_tensor": source.get("gate_up_tensor"),
        "down_tensor": source.get("down_tensor"),
        "expert_ids": source.get("activation_control", {}).get("selected_expert_ids"),
        "selected_neuron_count": candidate.get("variant", {}).get("selected_neuron_count"),
        "outcome": outcome,
        "complete_scoped_ebpw": accounting.get("complete_scoped_ebpw"),
        "complete_bytes": accounting.get("complete_bytes"),
        "byte_advantage_percent": accounting.get("byte_advantage_percent"),
        "matched_b_selected_relative_l2": matched.get("relative_l2") if isinstance(matched, dict) else None,
        "shuffled_b_selected_relative_l2": shuffled.get("relative_l2") if isinstance(shuffled, dict) else None,
        "matched_b_full_relative_l2": matched_full.get("relative_l2") if isinstance(matched_full, dict) else None,
        "shuffled_b_full_relative_l2": shuffled_full.get("relative_l2") if isinstance(shuffled_full, dict) else None,
        "assignment_similarity_mean": matching.get("assignment_similarity_mean"),
        "assignment_similarity_min": matching.get("assignment_similarity_min"),
        "boundary": "TWO_EXPERT_SELECTED_TRIPLES_ONLY__SOURCE_OUTPUT_DISCRIMINATOR__NOT_COMPLETE_NR_OR_CAPABILITY",
    }


def _hc_closure_screen_summary(document: dict | None) -> dict | None:
    """Carry the symbolic HyperConnection closure falsifier."""
    if (
        not document
        or document.get("schema") != HC_CLOSURE_SCREEN_SCHEMA
        or document.get("status") not in {
            "SYMBOLIC_FULL_JOINT_RANK__QUOTIENT_REJECTED",
            "SYMBOLIC_NONTRIVIAL_QUOTIENT_SURVIVES",
        }
    ):
        return None
    geometry = document.get("geometry")
    rank = document.get("rank_solution")
    if not isinstance(geometry, dict) or not isinstance(rank, dict):
        return None
    return {
        "status": document["status"],
        "streams": geometry.get("source_streams_m"),
        "joint_observable_rank": rank.get("joint_observable_rank"),
        "joint_observable_nullity": rank.get("joint_observable_nullity"),
        "read_mix_rank": rank.get("read_mix_rank"),
        "read_mix_determinant": rank.get("read_mix_determinant"),
        "nontrivial_r_less_than_m_possible": rank.get("nontrivial_r_less_than_m_possible"),
        "falsifier": rank.get("falsifier"),
        "boundary": "SYMBOLIC_HC_CONSUMER_CLOSURE_ONLY__NO_NUMERIC_SOURCE_STATE_OR_EBPW",
    }


def _router_source_output_screen_summary(document: dict | None) -> dict | None:
    """Carry the fixed source-bound E03 router gate without promoting it."""
    if (
        not document
        or document.get("schema") != ROUTER_SOURCE_OUTPUT_SCREEN_SCHEMA
        or document.get("status") not in {
            "FIXED_INT4_ROUTER_ROUTE_GATE_REJECTED",
            "FIXED_INT4_ROUTER_SOURCE_GATE_SURVIVES__REPLICATION_REQUIRED",
        }
    ):
        return None
    source = document.get("source")
    candidate = document.get("candidate")
    matched_null = document.get("matched_null")
    accounting = document.get("accounting")
    gate = document.get("gate")
    if not all(isinstance(value, dict) for value in (source, candidate, matched_null, accounting, gate)):
        return None
    source_output = candidate.get("source_output")
    null_output = matched_null.get("source_output")
    if not isinstance(source_output, dict) or not isinstance(null_output, dict):
        return None
    logits = source_output.get("logits")
    null_logits = null_output.get("logits")
    if not isinstance(logits, dict) or not isinstance(null_logits, dict):
        return None
    return {
        "status": document["status"],
        "router_tensor": source.get("router_tensor"),
        "router_shape": source.get("router_shape"),
        "source_activation_sha256": source.get("activation_sha256"),
        "source_route_ids": source_output.get("source_ids"),
        "candidate_route_ids": source_output.get("candidate_ids"),
        "candidate_top10_symmetric_difference": source_output.get("candidate_top10_symmetric_difference"),
        "candidate_top10_order_positions_changed": source_output.get("candidate_top10_order_positions_changed"),
        "source_logit_relative_l2": logits.get("relative_l2"),
        "source_logit_rmse": logits.get("rmse"),
        "source_logit_max_abs_error": logits.get("max_abs_error"),
        "candidate_route_weight_max_abs_error": (source_output.get("route_weight_error") or {}).get("max_abs_error"),
        "source_top10_top11_margin": source_output.get("source_top10_top11_margin"),
        "candidate_top10_top11_margin": source_output.get("candidate_top10_top11_margin"),
        "matched_null_top10_symmetric_difference": null_output.get("null_top10_symmetric_difference"),
        "matched_null_logit_relative_l2": null_logits.get("relative_l2"),
        "complete_scoped_ebpw": accounting.get("complete_scoped_ebpw"),
        "complete_candidate_bytes": accounting.get("complete_candidate_bytes"),
        "source_control_recomputed_ids_match_bridge": gate.get("source_control_recomputed_ids_match_bridge"),
        "fixed_candidate_route_identity_pass": gate.get("fixed_candidate_route_identity_pass"),
        "tail_non_router_organs": gate.get("tail_non_router_organs"),
        "boundary": "ONE_FIXED_SOURCE_BOUND_LAYER0_ROUTER_GATE__NOT_TAIL_COMPLETE_NR_OR_CAPABILITY",
    }


def _ple_context_response_screen_summary(document: dict | None) -> dict | None:
    """Carry a source-formula geometry prior without calling it a candidate."""
    if (
        not document
        or document.get("schema") != PLE_CONTEXT_RESPONSE_SCREEN_SCHEMA
        or document.get("status") != PLE_CONTEXT_RESPONSE_SCREEN_STATUS
    ):
        return None
    seal = document.get("seal_sha256")
    unsealed = dict(document)
    unsealed.pop("seal_sha256", None)
    if not isinstance(seal, str) or seal != hashlib.sha256(
        json.dumps(unsealed, sort_keys=True).encode()
    ).hexdigest():
        return None
    check = document.get("source_control_batch_check")
    bank = document.get("context_bank")
    responses = document.get("address_context_response")
    inputs = document.get("address_inputs")
    if not all(isinstance(value, dict) for value in (check, bank, responses, inputs)):
        return None
    cross = responses.get("cross_address_delta_spectrum")
    historical = inputs.get("historical_stateful_address_sequence")
    if not isinstance(cross, dict) or not isinstance(historical, dict):
        return None
    return {
        "status": document["status"],
        "source_control_batch_passed": check.get("passed"),
        "synthetic_context_bank": {
            "direction_count": bank.get("direction_count"),
            "context_count": bank.get("context_count"),
            "relative_rms": bank.get("relative_rms"),
            "source_trajectory_distribution": bank.get("source_trajectory_distribution"),
        },
        "cross_address_delta_spectrum": {
            "ranks_at_energy": cross.get("ranks_at_energy"),
            "numerical_rank": cross.get("numerical_rank"),
        },
        "historical_session_integrity": (
            (historical.get("historical_session_provenance") or {}).get("integrity_boundary")
        ),
        "boundary": (
            "SYNTHETIC_LOCAL_SOURCE_FORMULA_TANGENT_ONLY__NOT_REAL_STATE_BANK_OR_REPRESENTATION"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, default=Path("receipts/headless/FLASH_ORGAN_CENSUS.json"))
    ap.add_argument("--doctor", type=Path, default=Path("receipts/headless/FLASH_GRAVITY_DOCTOR_CYCLE.nr.json"))
    ap.add_argument("--bank-screen", type=Path, default=Path("receipts/headless/FLASH_DOCTOR_EXPERT_BANK_SCREEN.json"))
    ap.add_argument("--ngram-screen", type=Path, default=Path("receipts/headless/FLASH_DOCTOR_NGRAM_SCREEN.json"))
    ap.add_argument("--ngram-lookup-oracle", type=Path, default=Path("receipts/headless/FLASH_NGRAM_LOOKUP_ORACLE.json"))
    ap.add_argument(
        "--routed-lowrank-activation-screen",
        type=Path,
        default=Path("receipts/headless/FLASH_SOURCE_ROUTE_E411_GATEUP_LOWRANK_ACTIVATION_REPAIR.json"),
        help="optional source-activation low-rank/sparse rate-distortion screen",
    )
    ap.add_argument(
        "--ngram-pq-screen",
        type=Path,
        default=Path("receipts/headless/FLASH_NGRAM_SHARDED_LOOKUP_PQ_SCREEN.json"),
        help="optional held-out sharded-lookup PQ rate-distortion screen",
    )
    ap.add_argument(
        "--ngram-access-trace",
        type=Path,
        default=Path("receipts/headless/FLASH_PLE_ACCEPTED_TOKEN_ACCESS_TRACE.json"),
        help="optional source-algorithm PLE token-to-row address trace",
    )
    ap.add_argument(
        "--ple-source-output-control",
        type=Path,
        default=Path("receipts/headless/FLASH_PLE_LAYER1_BOS_SOURCE_OUTPUT_CONTROL.json"),
        help="optional source-bound PLE formula/input output control",
    )
    ap.add_argument(
        "--ple-reference-formula-parity",
        type=Path,
        default=Path("receipts/headless/FLASH_PLE_LAYER1_BOS_REFERENCE_FORMULA_PARITY.json"),
        help="optional bounded reference-framework PLE formula parity receipt",
    )
    ap.add_argument(
        "--ple-lookup-pq-output-screen",
        type=Path,
        default=Path("receipts/headless/FLASH_PLE_LOOKUP_PQ_SOURCE_OUTPUT_REPAIR_SCREEN.json"),
        help="optional source-control PQ plus sparse-repair output screen",
    )
    ap.add_argument(
        "--ple-shard-local-pq-output-screen",
        type=Path,
        default=Path("receipts/headless/FLASH_PLE_SHARD_LOCAL_PQ_SOURCE_OUTPUT_SCREEN.json"),
        help="optional source-control physical-shard-local PQ output screen",
    )
    ap.add_argument(
        "--ple-additive-pq-output-screen",
        type=Path,
        default=Path("receipts/headless/FLASH_PLE_ADDITIVE_PQ_SOURCE_OUTPUT_SCREEN.json"),
        help="optional source-control greedy additive residual-PQ output screen",
    )
    ap.add_argument(
        "--ple-address-generator-screen",
        type=Path,
        default=Path("receipts/headless/FLASH_PLE_ADDRESS_GENERATOR_SOURCE_OUTPUT_SCREEN.json"),
        help="optional source-control address-generator output screen",
    )
    ap.add_argument(
        "--ple-additive-hash-function-screen",
        type=Path,
        default=Path("receipts/headless/FLASH_PLE_ADDITIVE_HASH_FUNCTION_SOURCE_OUTPUT_SCREEN.json"),
        help="optional source-control discontinuous additive-hash function screen",
    )
    ap.add_argument(
        "--ple-context-response-screen",
        type=Path,
        default=Path("receipts/headless/FLASH_PLE_CONTEXT_RESPONSE_SCREEN.json"),
        help="optional source-formula address/context tangent screen",
    )
    ap.add_argument(
        "--expert-vector-code-screen",
        type=Path,
        default=Path("receipts/headless/FLASH_EXPERT_VECTOR_CODE_SOURCE_OUTPUT_SCREEN.json"),
        help="optional source-bound shared expert vector-code screen",
    )
    ap.add_argument(
        "--expert-shared-basis-screen",
        type=Path,
        default=Path("receipts/headless/FLASH_EXPERT_SHARED_BASIS_SOURCE_OUTPUT_SCREEN.json"),
        help="optional source-bound shared expert input-basis screen",
    )
    ap.add_argument(
        "--expert-atom-sharing-screen",
        type=Path,
        default=Path("receipts/headless/FLASH_EXPERT_LAWFUL_ATOM_SHARING_SCREEN.json"),
        help="optional source-bound lawful nonlinear gated-MoE atom-sharing screen",
    )
    ap.add_argument(
        "--hc-closure-screen",
        type=Path,
        default=Path("receipts/headless/FLASH_HYPERCONNECTION_SYMBOLIC_CLOSURE_SCREEN.json"),
        help="optional symbolic HyperConnection stream-closure screen",
    )
    ap.add_argument(
        "--router-source-output-screen",
        type=Path,
        default=Path("receipts/headless/FLASH_ROUTER_SOURCE_OUTPUT_SCREEN.json"),
        help="optional fixed source-bound router source-output screen",
    )
    ap.add_argument(
        "--gravity-registry",
        type=Path,
        default=Path("tools/foundry/GRAVITY_METHOD_REGISTRY.json"),
        help="canonical registry whose E02/E06/E13 frontier is independently verified",
    )
    ap.add_argument(
        "--target-ebpw",
        action="append",
        help=(
            "whole-body EBPW target to budget; repeatable "
            "(default: 1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01)"
        ),
    )
    ap.add_argument(
        "--generation",
        default="accounting-baseline-v1",
        help="explicit NR generation label",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("receipts/headless/FLASH_NR_ACCOUNTING_BASELINE_20260911.nr.json"),
        help="new accounting receipt; historical FLASH_COMPLETE_V*.nr.json is never the default target",
    )
    a = ap.parse_args()
    current_execution_frontier = validate_execution_frontier(a.gravity_registry)
    census = json.loads(a.census.read_text())
    doctor = json.loads(a.doctor.read_text())
    bank_screen = json.loads(a.bank_screen.read_text()) if a.bank_screen.is_file() else None
    ngram_screen = json.loads(a.ngram_screen.read_text()) if a.ngram_screen.is_file() else None
    ngram_lookup = json.loads(a.ngram_lookup_oracle.read_text()) if a.ngram_lookup_oracle.is_file() else None
    lowrank_activation_screen = optional_receipt(
        a.routed_lowrank_activation_screen,
        LOWRANK_ACTIVATION_SCREEN_SCHEMA,
        "routed low-rank activation",
    )
    ngram_pq_screen = optional_receipt(
        a.ngram_pq_screen,
        SHARDED_LOOKUP_PQ_SCREEN_SCHEMA,
        "n-gram PQ",
    )
    ngram_access_trace = optional_receipt(
        a.ngram_access_trace,
        PLE_ACCESS_TRACE_SCHEMA,
        "n-gram access trace",
    )
    ple_source_output_control = optional_receipt(
        a.ple_source_output_control,
        PLE_SOURCE_OUTPUT_CONTROL_SCHEMA,
        "PLE source output control",
    )
    ple_reference_formula_parity = optional_receipt(
        a.ple_reference_formula_parity,
        PLE_REFERENCE_FORMULA_ORACLE_SCHEMA,
        "PLE reference formula parity",
    )
    ple_lookup_pq_output_screen = optional_receipt(
        a.ple_lookup_pq_output_screen,
        PLE_LOOKUP_PQ_OUTPUT_SCREEN_SCHEMA,
        "PLE lookup PQ output screen",
    )
    ple_shard_local_pq_output_screen = optional_receipt(
        a.ple_shard_local_pq_output_screen,
        PLE_LOOKUP_PQ_OUTPUT_SCREEN_SCHEMA,
        "PLE shard-local PQ output screen",
    )
    ple_additive_pq_output_screen = optional_receipt(
        a.ple_additive_pq_output_screen,
        PLE_LOOKUP_PQ_OUTPUT_SCREEN_SCHEMA,
        "PLE additive PQ output screen",
    )
    ple_address_generator_screen = optional_receipt(
        a.ple_address_generator_screen,
        PLE_ADDRESS_GENERATOR_SCREEN_SCHEMA,
        "PLE address generator screen",
    )
    ple_additive_hash_function_screen = optional_receipt(
        a.ple_additive_hash_function_screen,
        PLE_ADDRESS_GENERATOR_SCREEN_SCHEMA,
        "PLE additive-hash function screen",
    )
    ple_context_response_screen = optional_receipt(
        a.ple_context_response_screen,
        PLE_CONTEXT_RESPONSE_SCREEN_SCHEMA,
        "PLE context response screen",
    )
    expert_vector_code_screen = optional_receipt(
        a.expert_vector_code_screen,
        EXPERT_VECTOR_CODE_SCREEN_SCHEMA,
        "expert vector-code screen",
    )
    expert_shared_basis_screen = optional_receipt(
        a.expert_shared_basis_screen,
        EXPERT_VECTOR_CODE_SCREEN_SCHEMA,
        "expert shared-basis screen",
    )
    expert_atom_sharing_screen = optional_receipt(
        a.expert_atom_sharing_screen,
        EXPERT_ATOM_SHARING_SCREEN_SCHEMA,
        "expert lawful nonlinear atom-sharing screen",
    )
    hc_closure_screen = optional_receipt(
        a.hc_closure_screen,
        HC_CLOSURE_SCREEN_SCHEMA,
        "HyperConnection closure screen",
    )
    router_source_output_screen = optional_receipt(
        a.router_source_output_screen,
        ROUTER_SOURCE_OUTPUT_SCREEN_SCHEMA,
        "router source-output screen",
    )
    lowrank_activation_targets = _best_lowrank_activation_rows(lowrank_activation_screen)
    ngram_pq_targets = _ngram_pq_summary(ngram_pq_screen)
    ngram_access_summary = _ple_access_trace_summary(ngram_access_trace)
    ple_source_output_summary = _ple_source_output_control_summary(ple_source_output_control)
    ple_reference_formula_summary = _ple_reference_formula_parity_summary(ple_reference_formula_parity)
    ple_lookup_pq_output_summary = _ple_lookup_pq_output_screen_summary(ple_lookup_pq_output_screen)
    ple_shard_local_pq_output_summary = _ple_shard_local_pq_output_screen_summary(
        ple_shard_local_pq_output_screen
    )
    ple_additive_pq_output_summary = _ple_additive_pq_output_screen_summary(
        ple_additive_pq_output_screen
    )
    ple_address_generator_summary = _ple_address_generator_screen_summary(ple_address_generator_screen)
    ple_additive_hash_function_summary = _ple_additive_hash_function_screen_summary(
        ple_additive_hash_function_screen
    )
    ple_context_response_summary = _ple_context_response_screen_summary(ple_context_response_screen)
    expert_vector_code_summary = _expert_vector_code_screen_summary(expert_vector_code_screen)
    expert_shared_basis_summary = _expert_vector_code_screen_summary(expert_shared_basis_screen)
    expert_atom_sharing_summary = _expert_atom_sharing_screen_summary(expert_atom_sharing_screen)
    hc_closure_summary = _hc_closure_screen_summary(hc_closure_screen)
    router_source_output_summary = _router_source_output_screen_summary(router_source_output_screen)
    if census.get("schema") != "hawking.flash.organ_census.v1":
        raise SystemExit("unexpected census schema")
    families = {row["family"]: row for row in census.get("family_summary", [])}
    ngram_q4 = next((row for row in (ngram_screen or {}).get("quantization_screen", []) if row.get("candidate") == "uniform_q4_g32"), None)
    parts = [
        {"family": "embedding_lm_head", "representation": "source_bf16_exact", "runtime_required": True, "qualification": "complete terminal control exists"},
        {"family": "ngram_embedding", "representation": "factorized_lookup_candidate", "runtime_required": True, "qualification": "128-shard global row layout, source-algorithm addresses, and one sealed source-bound PLE formula/input control have passed reference-framework parity on one BOS control; broader PLE output coverage and native compact lookup remain open"},
        {"family": "norm", "representation": "source_bf16_exact", "runtime_required": True, "qualification": "source graph contract"},
        {"family": "linear_attention_hyperconnection", "representation": "source_bf16_exact", "runtime_required": True, "qualification": "48-layer source parity and stateful organ/prefix evidence"},
        {"family": "full_attention", "representation": "source_bf16_exact", "runtime_required": True, "qualification": "all full-attention source organs and KV organ evidence"},
        {"family": "mlp_hyperconnection", "representation": "source_bf16_exact", "runtime_required": True, "qualification": "source graph contract"},
        {"family": "shared_expert", "representation": "source_bf16_exact", "runtime_required": True, "qualification": "source graph contract"},
        {"family": "routed_experts", "representation": "route_conditioned_compact_candidate", "runtime_required": True, "qualification": "compact routed-bank exact parity on verified layers; dynamic complete-bank storage not yet qualified"},
        {"family": "other", "representation": "source_bf16_exact", "runtime_required": True, "qualification": "source graph contract"},
    ]
    total = census["source_parameter_bytes_indexed"]
    if total % 2:
        raise SystemExit("source BF16 byte census is odd; logical-weight denominator is undefined")
    parameter_count = total // 2
    family_source_bytes = {name: int(row.get("bytes", 0)) for name, row in families.items()}
    expected_families = {part["family"] for part in parts}
    if expected_families != set(family_source_bytes):
        raise SystemExit(
            "Flash source family census does not match the complete NR family list: "
            f"expected={sorted(expected_families)} observed={sorted(family_source_bytes)}"
        )
    for part in parts:
        # This bills the exact source control separately.  It is evidence for
        # the denominator and current storage burden, not a claim that those
        # bytes have been packaged inside the new NR artifact.
        part["source_control_bytes"] = family_source_bytes[part["family"]]
    budgets = [
        target_budget(
            parameter_count=parameter_count,
            family_source_bytes=family_source_bytes,
            raw_target=raw,
        )
        for raw in (a.target_ebpw or DEFAULT_TARGET_EBPW)
    ]
    doc = {
        "nr_version": "1.0.0",
        "nr_kind": "hawking.nos.noetic_representation",
        "artifact_kind": "NR",
        "schema": f"hawking.flash.complete_nr.{a.generation}",
        "generation": a.generation,
        "status": "OPEN_COMPLETE_FAMILY_ACCOUNTING_BASELINE__NOT_FOR_PROMOTION",
        "current_execution_frontier": current_execution_frontier,
        "research_target_ladder": {
            "engineering_gate_complete_ebpw": "0.5",
            "deep_frontier_complete_ebpw": "0.25",
            "grand_challenge_complete_ebpw": "0.1",
            "open_horizon_complete_ebpw": ["0.05", "0.025", "0.01"],
            "law": "These are complete-NR research pressures. No target is earned until declared closure, direct execution, physical behavior, and capability gates pass.",
        },
        "semantic_provenance": {
            "parent_model": census["model"],
            "parent_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
            "parameter_count": parameter_count,
            "source_parameter_bytes_indexed": total,
            "source_index_sha256": census["source_index_sha256"],
            "gravity_registry_sha256": sha(a.gravity_registry),
            "e02_replication_sha256": sha(Path(current_execution_frontier["e02_replication"])),
            "e06_prefilter_sha256": sha(Path(current_execution_frontier["e06_prefilter"])),
            "e13_vocabulary_frontier_sha256": sha(
                Path(current_execution_frontier["e13_preflight"])
            ),
            "e13_fixed_corpus_refusal_sha256": sha(
                Path(current_execution_frontier["e13_refusal"])
            ),
            "e13_first_fixed_corpus_refusal_sha256": sha(
                Path(current_execution_frontier["e13_first_refusal"])
            ),
            "e13_lexical_universe_sha256": sha(
                Path(current_execution_frontier["e13_lexical_universe"])
            ),
            "census_sha256": sha(a.census),
            "doctor_nr_sha256": sha(a.doctor),
            "doctor_bank_screen_sha256": sha(a.bank_screen) if bank_screen else None,
            "doctor_ngram_screen_sha256": sha(a.ngram_screen) if ngram_screen else None,
            "ngram_lookup_oracle_sha256": sha(a.ngram_lookup_oracle) if ngram_lookup else None,
            "routed_lowrank_activation_screen_sha256": sha(a.routed_lowrank_activation_screen) if lowrank_activation_screen else None,
            "ngram_pq_screen_sha256": sha(a.ngram_pq_screen) if ngram_pq_screen else None,
            "ngram_access_trace_sha256": sha(a.ngram_access_trace) if ngram_access_trace else None,
            "ple_source_output_control_sha256": sha(a.ple_source_output_control) if ple_source_output_control else None,
            "ple_reference_formula_parity_sha256": sha(a.ple_reference_formula_parity) if ple_reference_formula_parity else None,
            "ple_lookup_pq_output_screen_sha256": sha(a.ple_lookup_pq_output_screen) if ple_lookup_pq_output_screen else None,
            "ple_shard_local_pq_output_screen_sha256": sha(a.ple_shard_local_pq_output_screen) if ple_shard_local_pq_output_screen else None,
            "ple_additive_pq_output_screen_sha256": sha(a.ple_additive_pq_output_screen) if ple_additive_pq_output_screen else None,
            "ple_address_generator_screen_sha256": sha(a.ple_address_generator_screen) if ple_address_generator_screen else None,
            "ple_additive_hash_function_screen_sha256": sha(a.ple_additive_hash_function_screen) if ple_additive_hash_function_screen else None,
            "ple_context_response_screen_sha256": sha(a.ple_context_response_screen) if ple_context_response_screen else None,
            "expert_vector_code_screen_sha256": sha(a.expert_vector_code_screen) if expert_vector_code_screen else None,
            "expert_shared_basis_screen_sha256": sha(a.expert_shared_basis_screen) if expert_shared_basis_screen else None,
            "expert_atom_sharing_screen_sha256": sha(a.expert_atom_sharing_screen) if expert_atom_sharing_screen else None,
            "hc_closure_screen_sha256": sha(a.hc_closure_screen) if hc_closure_screen else None,
            "router_source_output_screen_sha256": sha(a.router_source_output_screen) if router_source_output_screen else None,
        },
        "representation": {
            "scope": "complete 48-layer Flash model",
            "parts": parts,
            "family_source_bytes": family_source_bytes,
            "accounting": {
                "canonical_rust_contract": "crates/hawking-core/src/gravity_manifest.rs:NoeticRepresentationManifest::totals + rate_budget",
                "flash_adapter": "tools/flash_complete_nr.py",
                "source_control": {
                    "status": "EXACT_SOURCE_BF16_STORAGE_CONTROL__NOT_NR_CLOSURE",
                    "persistent_bytes": total,
                    "logical_weights": parameter_count,
                    "complete_ebpw": 16.0,
                    "complete_ebpw_ppm": 16_000_000,
                    "source_dependency": "pinned BF16 source shards remain outside this NR",
                },
                "nr_candidate": {
                    "status": "OPEN__NO_COMPLETE_EBPW_CLAIM",
                    "closure_status": "OPEN",
                    "complete_persistent_bytes": None,
                    "complete_ebpw": None,
                    "active_bytes_per_token": None,
                    "mutable_state_bytes": None,
                    "scratch_bytes": None,
                    "direct_execution": "PARTIAL_COMPONENT_ONLY",
                    "source_independent": False,
                    "external_dependencies": [
                        "pinned BF16 source-family bodies",
                        "unqualified dynamic routed-expert bank representation",
                        "unqualified n-gram representation and lookup path",
                        "unqualified broader PLE output coverage and native compact PLE runtime",
                        "unmeasured complete direct runtime state",
                        "unmeasured complete direct scratch working set",
                    ],
                    "reason": "The document identifies every organ family but does not yet contain a closed direct body for all of them.",
                },
                "whole_body_target_budgets": budgets,
            },
            "candidate_variants": [
                {"name": "external_source_bf16_control", "source_control_ebpw": 16.0, "nr_complete_ebpw": None, "runnable_nr": False, "capability_status": "source-control-only"},
                {"name": "route_conditioned_compact_experts_v0", "complete_bits_per_weight": None, "runtime_ready": False, "capability_status": "not-yet-qualified", "open": "dynamic expert-bank representation and accepted-token accounting", "bank_screen": "cross-expert sharing hypothesis is weak in sampled real weights; pursue active-route storage, not unconditional shared basis"},
                {"name": "ngram_factorized_lookup_v0", "complete_bits_per_weight": None, "runtime_ready": False, "capability_status": "not-yet-qualified", "open": "broader PLE output coverage, lookup representation fidelity, and native compact lookup", "lookup_oracle_status": (ngram_lookup or {}).get("status"), "access_trace": ngram_access_summary, "source_output_control": ple_source_output_summary, "reference_formula_parity": ple_reference_formula_summary},
                {"name": "ngram_uniform_q4_g32_v0", "complete_bits_per_weight": None, "runtime_ready": False, "capability_status": "stage-a-only", "nominal_bpw": (ngram_q4 or {}).get("nominal_bpw"), "sample_cosine": (ngram_q4 or {}).get("sample_cosine"), "open": "activation/output sensitivity and native compact lookup"},
                {"name": "ngram_packed_q4_g32_lookup_v1", "complete_bits_per_weight": 4.0 + 32.0 / 32.0, "runtime_ready": False, "capability_status": "row-oracle-only", "sample_rows": ((ngram_lookup or {}).get("source") or {}).get("sample_rows"), "mean_lookup_ns": ((ngram_lookup or {}).get("bench") or {}).get("lookup_ns_mean"), "open": "native lookup kernel, activation sensitivity, collision semantics, and complete-token impact"},
                {"name": "ngram_packed_q3_g32_lookup_v1", "complete_bits_per_weight": 3.0 + 32.0 / 32.0, "runtime_ready": False, "capability_status": "row-oracle-only", "sample_rows": ((ngram_lookup or {}).get("source") or {}).get("sample_rows"), "mean_lookup_ns": ((ngram_lookup or {}).get("bench") or {}).get("lookup_ns_mean"), "open": "native lookup kernel, activation sensitivity, collision semantics, and complete-token impact"},
                {
                    "name": "routed_expert_lowrank_sparse_activation_repair_v1",
                    "scope": "one exact layer-0 source-selected gate/up expert control",
                    "runtime_ready": False,
                    "capability_status": "SOURCE_ACTIVATION_NEGATIVE__NOT_ESCALATED",
                    "target_summary": lowrank_activation_targets,
                    "open": "A new composition needs multiple source/held-out activations or learned repair; do not escalate pure BF16 low-rank plus sparse correction from this control.",
                },
                {
                    "name": "ngram_shared_product_quantization_v1",
                    "scope": "128-shard deterministic train/held-out row screen",
                    "runtime_ready": False,
                    "capability_status": "HELDOUT_STATIC_NEGATIVE__NOT_ESCALATED",
                    "target_summary": ngram_pq_targets,
                    "open": "The source-algorithm address trace and one sealed reference-checked PLE formula/input control are bound. Test selective repair against that output control before considering learned/generative table representations; do not materialize uniform global PQ from this screen.",
                },
                {
                    "name": "ngram_shared_pq_fixed_sparse_repair_output_v1",
                    "scope": "one sealed source PLE output control with projected full-table code and exact-BF16 residual accounting",
                    "runtime_ready": False,
                    "capability_status": "SOURCE_FUNCTION_CONTROL_NEGATIVE__NOT_ESCALATED",
                    "target_summary": ple_lookup_pq_output_summary,
                    "open": "At the measured sub-1 projected table points, fixed per-row top-error BF16 repair leaves high PLE-output distortion. Do not retry this same uniform-PQ plus local-error residual rule; test a materially different function representation, such as a generated/state-conditioned lookup or learned repair, before broader PLE controls.",
                },
                {
                    "name": "ngram_physical_shard_local_pq_output_v1",
                    "scope": "one sealed source PLE output control with one separately billed BF16 PQ codebook family per physical checkpoint shard",
                    "runtime_ready": False,
                    "capability_status": "SOURCE_FUNCTION_CONTROL_NEGATIVE__NOT_ESCALATED",
                    "target_summary": ple_shard_local_pq_output_summary,
                    "open": "Physical shard-local BF16 PQ does not improve the useful sub-one PLE-output curve over the global-plus-fixed-repair control. Do not retry this fixed local-codebook hierarchy; a next family must change the function or allocation through learned/context-conditioned/hierarchical residuals or a learned PLE replacement.",
                },
                {
                    "name": "ngram_greedy_additive_residual_pq_output_v1",
                    "scope": "one sealed source PLE output control with globally billed BF16 residual-PQ codebook stages and packed stage codes",
                    "runtime_ready": False,
                    "capability_status": "SOURCE_FUNCTION_CONTROL_NEGATIVE__NOT_ESCALATED",
                    "target_summary": ple_additive_pq_output_summary,
                    "open": "Greedy additive residual PQ does not improve the useful <=0.5 PLE-output curve over the existing global-plus-fixed-repair result. Do not sweep stage/card variants as a substitute for a new hypothesis; move to an address/context/state contribution function, learned allocation, or learned PLE replacement.",
                },
                {
                    "name": "ngram_address_conditioned_fourier_generator_v1",
                    "scope": "one sealed source PLE output control with a complete projected BF16 address-generator program",
                    "runtime_ready": False,
                    "capability_status": "SOURCE_FUNCTION_CONTROL_NEGATIVE__NOT_ESCALATED",
                    "target_summary": ple_address_generator_summary,
                    "open": "A tiny address-conditioned Fourier generator has no useful held-out or PLE-output fidelity on the measured source control. Do not retry an address-only smooth-coordinate generator; a next function family must use materially different state/context information, hierarchy, or learned/Nova repair.",
                },
                {
                    "name": "ngram_address_conditioned_additive_hash_factor_v1",
                    "scope": "one sealed source PLE output control with a complete projected BF16 discontinuous additive-hash factor program",
                    "runtime_ready": False,
                    "capability_status": "SOURCE_FUNCTION_CONTROL_NEGATIVE__NOT_ESCALATED",
                    "target_summary": ple_additive_hash_function_summary,
                    "open": "The fixed three-term additive-hash factor program has no useful held-out or PLE-output fidelity despite a tiny projected table rate. Do not tune bucket count, seed, or ridge as a substitute for a new hypothesis; move to real context/history-conditioned, learned-topology, sparse-exception, or Nova-trained PLE functions.",
                },
                {
                    "name": "ngram_address_context_response_tangent_v1",
                    "scope": "one sealed source PLE state with deterministic synthetic local context perturbations across real hashed addresses",
                    "runtime_ready": False,
                    "capability_status": "SOURCE_FUNCTION_GEOMETRY_PRIOR__NOT_A_REPRESENTATION_OR_CAPABILITY_RESULT",
                    "geometry_summary": ple_context_response_summary,
                    "open": "The source formula has a compact local response spectrum over the declared synthetic plane, but this is not a real source-state distribution. Before any learned (address, context, persistent PLE state) contribution model, collect a separately sealed multi-context source-state bank and use disjoint context/sequence controls.",
                },
                {
                    "name": "routed_expert_shared_vector_code_v1",
                    "scope": "ten source-selected layer-0 gate/up experts under one sealed source activation and route control",
                    "runtime_ready": False,
                    "capability_status": "SOURCE_OUTPUT_SCREEN_NEGATIVE__NOT_ESCALATED",
                    "target_summary": expert_vector_code_summary,
                    "open": "The tested shared BF16 product-vector-code family reaches 0.034197 scoped EBPW at its densest point but remains highly distortive on held-out weights and route-weighted source output. Do not tune this fixed shared-PQ family as a substitute for a materially different expert function representation; require learned/shared generator, output-aware residual, or another source-bound hypothesis.",
                },
                {
                    "name": "routed_expert_shared_input_basis_v1",
                    "scope": "ten source-selected layer-0 gate/up experts under one sealed source activation and route control",
                    "runtime_ready": False,
                    "capability_status": "SOURCE_OUTPUT_SCREEN_NEGATIVE__NOT_ESCALATED",
                    "target_summary": expert_shared_basis_summary,
                    "open": "The tested shared-input-basis factor formula reaches 0.046072 scoped EBPW at rank 4 and 0.256072 at rank 32, but remains highly distortive on held-out rows and route-weighted source output. Do not increase rank or retune randomized-basis settings as a substitute for a materially different expert-family function hypothesis.",
                },
                {
                    "name": "routed_expert_lawful_nonlinear_atom_sharing_v1",
                    "scope": "four fixed disjoint adjacent pairs from the sealed layer-0 top-eight route",
                    "runtime_ready": False,
                    "capability_status": current_execution_frontier["e02_state"],
                    "target_summary": {
                        "receipt": current_execution_frontier["e02_replication"],
                        "seal_sha256": current_execution_frontier[
                            "e02_replication_seal_sha256"
                        ],
                        "original_calibration_pair_reproduced": current_execution_frontier[
                            "e02_original_calibration_pair_reproduced"
                        ],
                        "independent_replication_pair_count": current_execution_frontier[
                            "e02_independent_replication_pair_count"
                        ],
                        "independent_survivor_count": current_execution_frontier[
                            "e02_independent_survivor_count"
                        ],
                        "replicated": current_execution_frontier["e02_replicated"],
                    },
                    "open": "The original pair reproduced, but the preregistered top-eight bank did not replicate across all three independent pairs. Close this fixed pairwise family at its measured scope. Broader learned or output-aware lawful sharing remains a distinct hypothesis, not a continuation of this fixed setting.",
                },
                {
                    "name": "hyperconnection_simultaneous_invariant_quotient_v1",
                    "scope": "Flash HyperConnection stream axis and declared read-mix/combine consumers",
                    "runtime_ready": False,
                    "capability_status": (
                        "SYMBOLIC_FULL_RANK__QUOTIENT_REJECTED"
                        if (hc_closure_summary or {}).get("joint_observable_nullity") == 0
                        else "SYMBOLIC_SURVIVOR__NUMERIC_SOURCE_STATE_REQUIRED"
                    ),
                    "target_summary": hc_closure_summary,
                    "open": "The declared consumer set is already full-rank at m=4, so no nontrivial exact HC stream quotient is available. Reopen only with a materially different consumer closure or a measured source-approved approximation; do not fit a smaller state by assumption.",
                },
                {
                    "name": "flash_router_rowwise_int4_source_output_v1",
                    "scope": "one exact layer-0 router and one sealed source MLP input",
                    "runtime_ready": False,
                    "capability_status": (
                        "SOURCE_ROUTER_ROUTE_GATE_NEGATIVE__NOT_ESCALATED"
                        if (router_source_output_summary or {}).get("status")
                        == "FIXED_INT4_ROUTER_ROUTE_GATE_REJECTED"
                        else "SOURCE_OUTPUT_DISCRIMINATOR_SURVIVES__REPLICATION_REQUIRED"
                    ),
                    "target_summary": router_source_output_summary,
                    "open": "The fixed rowwise INT4 router candidate is rejected when source route identity changes. Keep the router high fidelity; reopen only with a materially different representation or disjoint source-input bank, and gate the other tail organs independently.",
                },
            ],
            "portable_identity": {
                "census": "receipts/headless/FLASH_ORGAN_CENSUS.json",
                "doctor": "receipts/headless/FLASH_GRAVITY_DOCTOR_CYCLE.nr.json",
                "routed_lowrank_activation_screen": str(a.routed_lowrank_activation_screen) if lowrank_activation_screen else None,
                "ngram_pq_screen": str(a.ngram_pq_screen) if ngram_pq_screen else None,
                "ngram_access_trace": str(a.ngram_access_trace) if ngram_access_trace else None,
                "ple_source_output_control": str(a.ple_source_output_control) if ple_source_output_control else None,
                "ple_reference_formula_parity": str(a.ple_reference_formula_parity) if ple_reference_formula_parity else None,
                "ple_lookup_pq_output_screen": str(a.ple_lookup_pq_output_screen) if ple_lookup_pq_output_screen else None,
                "ple_shard_local_pq_output_screen": str(a.ple_shard_local_pq_output_screen) if ple_shard_local_pq_output_screen else None,
                "ple_additive_pq_output_screen": str(a.ple_additive_pq_output_screen) if ple_additive_pq_output_screen else None,
                "ple_address_generator_screen": str(a.ple_address_generator_screen) if ple_address_generator_screen else None,
                "ple_additive_hash_function_screen": str(a.ple_additive_hash_function_screen) if ple_additive_hash_function_screen else None,
                "ple_context_response_screen": str(a.ple_context_response_screen) if ple_context_response_screen else None,
                "expert_vector_code_screen": str(a.expert_vector_code_screen) if expert_vector_code_screen else None,
                "expert_shared_basis_screen": str(a.expert_shared_basis_screen) if expert_shared_basis_screen else None,
                "expert_atom_sharing_screen": str(a.expert_atom_sharing_screen) if expert_atom_sharing_screen else None,
                "hc_closure_screen": str(a.hc_closure_screen) if hc_closure_screen else None,
                "router_source_output_screen": str(a.router_source_output_screen) if router_source_output_screen else None,
            },
        },
        "kernel_requirements": [
            {"requires": "source_bf16_gemv_family", "applies_to": "exact fallback organs"},
            {"requires": "grouped_absmax_decoder", "applies_to": "future compact routed-expert candidates only"},
            {"requires": "route_conditioned_expert_selection", "applies_to": "routed expert family"},
            {"requires": "gated_delta_recurrence", "applies_to": "linear-attention state"},
            {"requires": "causal_attention_kv_state", "applies_to": "full-attention state"},
        ],
        "promotion": {"allowed": False, "blockers": ["closed direct runtime", "complete accepted-token session", "complete candidate byte ledger", "capability suite"]},
        "evidence": {"state": "STATIC_ACCOUNTING_ONLY", "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "recorded_by": "tools/flash_complete_nr.py representation composition", "rule": "S032 §3 -- NR composition carries no physical timing claim"},
        "negative_science": [
            {"receipt": str(a.bank_screen), "finding": ((bank_screen or {}).get("population") or {}).get("cross_expert_gate_up_mean_cosine"), "law": "do not assume routed experts share a low-dimensional basis; sampled gate-up expert cosine was near zero"},
            {"receipt": str(a.ngram_screen), "finding": ((ngram_screen or {}).get("population") or {}).get("mean_pairwise_row_cosine"), "law": "do not assume n-gram shards share a low-dimensional basis; shard-row similarity was near zero"},
            {
                "receipt": str(a.routed_lowrank_activation_screen),
                "finding": lowrank_activation_targets,
                "law": "On the sealed source-selected E411 gate/up control, pure BF16 low-rank plus sparse correction remained high-error at sub-1 scoped rates; do not escalate this substrate without a materially different repair hypothesis.",
            },
            {
                "receipt": str(a.ngram_pq_screen),
                "finding": ngram_pq_targets,
                "law": "Shared product quantization can meet sub-bit projected n-gram table budgets arithmetically, but its held-out row distortion is high at the tested rates; source addresses plus one reference-checked PLE formula/input control now permit bounded selective-repair controls, but not broad PLE parity or capability claims.",
            },
            {
                "receipt": str(a.ple_lookup_pq_output_screen) if ple_lookup_pq_output_screen else None,
                "finding": ple_lookup_pq_output_summary,
                "law": "On the sealed Flash BOS PLE formula control, globally billed shared PQ plus fixed per-row exact-BF16 top-error repair remains highly distortive below 1.0 projected table EBPW (best <=0.5 point: 0.850550 relative-L2; best <=1.0 point: 0.727215). Do not repeat this uniform base plus local-error residual rule as a PLE escalation; it is not a claim that a state-conditioned/generated or learned PLE function cannot work.",
            },
            {
                "receipt": str(a.ple_shard_local_pq_output_screen) if ple_shard_local_pq_output_screen else None,
                "finding": ple_shard_local_pq_output_summary,
                "law": "On the sealed Flash BOS PLE formula control, independently fitted BF16 PQ codebooks for each of the 128 physical PLE shards do not yield a useful sub-one output curve: the best <=0.5 point is 0.375062 projected table EBPW with 0.859731 relative-L2, worse than the existing global-plus-fixed-repair <=0.5 point at 0.850550. Do not retry this fixed shard-local PQ hierarchy; this does not reject a learned, context-conditioned, generated, or Nova-trained PLE function.",
            },
            {
                "receipt": str(a.ple_additive_pq_output_screen) if ple_additive_pq_output_screen else None,
                "finding": ple_additive_pq_output_summary,
                "law": "On the sealed Flash BOS PLE formula control, two-stage greedily decoded BF16 additive residual PQ has no useful sub-one output curve: the best <=0.5 point is 0.400011 projected table EBPW with 0.861647 relative-L2, while the 0.500011 point remains 0.814410. Do not turn this into another stage/card sweep; this does not reject jointly learned, context-conditioned, generated, or Nova-trained PLE replacement functions.",
            },
            {
                "receipt": str(a.ple_address_generator_screen) if ple_address_generator_screen else None,
                "finding": ple_address_generator_summary,
                "law": "On the sealed Flash BOS PLE formula control, an address-only BF16 Fourier row generator is effectively uncorrelated with held-out table rows (best reported held-out relative-L2 1.035783) and leaves 0.953283 PLE-output relative-L2 despite a 0.000007130 projected table EBPW. Do not retry smooth address-only coordinate generators; this is not a claim against context-conditioned, hierarchical, or learned PLE functions.",
            },
            {
                "receipt": str(a.ple_additive_hash_function_screen) if ple_additive_hash_function_screen else None,
                "finding": ple_additive_hash_function_summary,
                "law": "On the sealed Flash BOS PLE formula control, a fixed three-term discontinuous additive-hash BF16 factor generator has no useful held-out or PLE-output fidelity (best tested output point: 0.979713 relative-L2 at 0.000010330 projected table EBPW). Do not retune bucket count, mix seed, or ridge as a substitute for a materially new context/history-conditioned, learned-topology, sparse-exception, or Nova-trained PLE function.",
            },
            {
                "receipt": str(a.expert_vector_code_screen) if expert_vector_code_screen else None,
                "finding": expert_vector_code_summary,
                "law": "On the sealed Flash layer-0 source route, shared BF16 product-vector codes are arithmetically tiny but highly distortive: the densest tested selected-expert family is 0.034197 scoped EBPW with 0.848695 route-weighted source-output relative-L2, while the best route-output point is 0.119822 scoped EBPW with 0.812955 error. Close this fixed shared-PQ expert family for the tested control; this does not reject learned generators, output-aware residuals, or other functional expert representations.",
            },
            {
                "receipt": str(a.expert_shared_basis_screen) if expert_shared_basis_screen else None,
                "finding": expert_shared_basis_summary,
                "law": "On the sealed Flash layer-0 source route, a shared BF16 input-basis factor formula is also highly distortive: rank 4 reaches 0.046072 scoped EBPW with 0.986917 route-weighted source-output relative-L2, while rank 32 reaches 0.256072 with 0.903899 error. Close this fixed shared-basis form for the tested control; this does not reject lawful nonlinear atom sharing, route-aggregate representations, learned generators, or output-aware repair.",
            },
            {
                "receipt": current_execution_frontier["e02_replication"],
                "finding": {
                    "state": current_execution_frontier["e02_state"],
                    "original_calibration_pair_reproduced": current_execution_frontier[
                        "e02_original_calibration_pair_reproduced"
                    ],
                    "independent_replication_pair_count": current_execution_frontier[
                        "e02_independent_replication_pair_count"
                    ],
                    "independent_survivor_count": current_execution_frontier[
                        "e02_independent_survivor_count"
                    ],
                    "replicated": current_execution_frontier["e02_replicated"],
                },
                "law": "The original E02 pair reproduced, but only two of three preregistered independent top-eight pairs beat the unchanged matched null. Close this fixed pairwise replication family at its measured scope; retain the original as a bounded non-generalized signal, not a usable representation.",
            },
            {
                "receipt": current_execution_frontier["e06_prefilter"],
                "finding": {
                    "state": current_execution_frontier["e06_state"],
                    "conditional_saving_percent": current_execution_frontier[
                        "e06_conditional_saving_percent"
                    ],
                    "conditional_net_saving_bits_after_auxiliary_cost": current_execution_frontier[
                        "e06_conditional_net_saving_bits_after_auxiliary_cost"
                    ],
                    "candidate_for_separate_review": current_execution_frontier[
                        "e06_candidate_for_separate_review"
                    ],
                },
                "law": "The canonical E06 row-norm code screen found only 3.158489863% conditional saving and loses 74645.5625 projected bits after auxiliary cost. It does not earn source-output escalation; do not substitute marginal occupancy, omit predictor cost, or drop the shuffled null.",
            },
        ],
        "frontier_evidence": [
            {
                "receipt": str(a.expert_atom_sharing_screen) if expert_atom_sharing_screen else None,
                "finding": expert_atom_sharing_summary,
                "law": "The original two-expert E02 calibration produced a bounded source-output signal, but the later sealed top-eight replication failed its all-pairs rule. Preserve this receipt as calibration evidence only; the fixed pairwise family is closed at the replicated scope.",
            },
            {
                "receipt": str(a.hc_closure_screen) if hc_closure_screen else None,
                "finding": hc_closure_summary,
                "law": "The canonical Flash HyperConnection read-mix and combine consumers have full joint stream-axis rank 4/4; the direct residual-stream consumer alone blocks a nontrivial exact quotient. Close E04 for this declared formula set without model loading, and do not claim an HC state reduction absent a materially different consumer or a source-approved approximation.",
            },
            {
                "receipt": str(a.router_source_output_screen) if router_source_output_screen else None,
                "finding": router_source_output_summary,
                "law": "On the sealed Flash layer-0 source MLP input, source route recomputation matches all ten authoritative IDs, but the one fixed rowwise INT4 router candidate changes 8 of 10 route identities (16-ID symmetric difference) at 4.4078125 scoped EBPW and 0.346322 router-logit relative-L2. Close this fixed router family for the tested control; do not attribute the route failure to expert compression, and keep non-router tail gates withheld until valid source-output controls exist.",
            },
            {
                "receipt": current_execution_frontier["e13_refusal"],
                "supporting_preflight": current_execution_frontier["e13_preflight"],
                "finding": {
                    "state": current_execution_frontier["e13_state"],
                    "tensor_row_count": current_execution_frontier["e13_tensor_row_count"],
                    "tokenizer_addressable_count": current_execution_frontier[
                        "e13_tokenizer_addressable_count"
                    ],
                    "lexical_eligible_count": current_execution_frontier[
                        "e13_lexical_eligible_count"
                    ],
                    "tensor_only_tail_count": current_execution_frontier[
                        "e13_tensor_only_tail_count"
                    ],
                    "row_extraction_performed": current_execution_frontier[
                        "e13_row_extraction_performed"
                    ],
                    "tensor_payload_bytes_read": current_execution_frontier[
                        "e13_tensor_payload_bytes_read"
                    ],
                    "first_underpowered_stratum": current_execution_frontier[
                        "e13_first_underpowered_stratum"
                    ],
                    "first_underpowered_eligible_rows": current_execution_frontier[
                        "e13_first_underpowered_eligible_rows"
                    ],
                    "first_underpowered_required_rows": current_execution_frontier[
                        "e13_first_underpowered_required_rows"
                    ],
                    "han_short_structural_rows": current_execution_frontier[
                        "e13_han_short_structural_rows"
                    ],
                },
                "law": "E13 has exact tokenizer/tensor ownership and a corpus-free structural census. All script/byte cells are geometrically possible (Han-short 5426/512), but two predeclared absolute-frequency controls failed different count==1 cells: Wikinews rare-Han-short 166/512 and broader Wikimedia rare-Latin-short 330/512. This is now a frequency-design review boundary, not permission to tune thresholds or acquire a third corpus; extraction and tail sampling remain withheld.",
            },
        ],
        "claim_boundary": "This is a complete source-family inventory plus an open-NR accounting baseline. The exact 16.0 EBPW figure belongs only to external source BF16 storage; this receipt claims no complete NR EBPW, closed standalone runnable NR, accepted-token TPS, or capability preservation.",
        "next": "First admit the captured external PLE-inclusive teacher through the machine-owner gate, then localize the layer-4/token-0 divergence before another full replay. E02 and E06 are closed negative at their declared scopes. E13 is structurally possible but its two predeclared absolute-frequency corpora failed different rare cells; do not tune thresholds or acquire a third corpus, and revisit the frequency design only if its expected information exceeds source-bound work. No complete source-independent candidate exists, so do not freeze, report EBPW/TPS, or build an executor yet. Preserve exact fallbacks while testing only materially new source-bound hypotheses. Package any future complete survivor through the strict Rust NR admission owner; NX remains optional.",
    }
    ok, bad = validate(doc)
    if not ok:
        raise SystemExit("NR validation failed: " + "; ".join(bad))
    doc["seal_sha256"] = seal_payload_sha256_v1(doc)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({
        "status": doc["status"],
        "families": len(parts),
        "source_control_ebpw": doc["representation"]["accounting"]["source_control"]["complete_ebpw"],
        "nr_complete_ebpw": doc["representation"]["accounting"]["nr_candidate"]["complete_ebpw"],
        "target_ebpw_budgets": [row["target_complete_ebpw"] for row in budgets],
        "out": str(a.out),
        "seal": doc["seal_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
