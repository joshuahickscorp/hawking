#!/usr/bin/env python3
"""Compose the first complete, portable Flash NR candidate.

The candidate covers every Flash organ family.  Exact source-BF16 remains the
fallback control; the routed-expert compact option is included only as a
route-conditioned candidate because its complete dynamic-bank storage and
accepted-token impact are not yet qualified.  An NR is eligible to run directly
through Hawking once its declared closure and direct runtime are qualified; an
NX is only an optional later machine specialization, so this file contains no
machine binding.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
import time
from pathlib import Path

from nr_container import validate


PPM_BITS_PER_BYTE = 8_000_000
DOMINANT_DENSITY_FAMILIES = ("routed_experts", "ngram_embedding")
DEFAULT_TARGET_EBPW = ("1.0", "0.75", "0.5", "0.25", "0.1")
LOWRANK_ACTIVATION_SCREEN_SCHEMA = "hawking.odyssey.stacked_expert_lowrank_sparse_repair.v1"
SHARDED_LOOKUP_PQ_SCREEN_SCHEMA = "hawking.odyssey.sharded_lookup_pq_screen.v1"
PLE_ACCESS_TRACE_SCHEMA = "hawking.odyssey.ple_access_trace.v1"


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
    for target in (0.1, 0.25, 0.5, 0.75, 1.0):
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
        "--target-ebpw",
        action="append",
        help="whole-body EBPW target to budget; repeatable (default: 1.0, 0.75, 0.5, 0.25, 0.1)",
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
    lowrank_activation_targets = _best_lowrank_activation_rows(lowrank_activation_screen)
    ngram_pq_targets = _ngram_pq_summary(ngram_pq_screen)
    ngram_access_summary = _ple_access_trace_summary(ngram_access_trace)
    if census.get("schema") != "hawking.flash.organ_census.v1":
        raise SystemExit("unexpected census schema")
    families = {row["family"]: row for row in census.get("family_summary", [])}
    ngram_q4 = next((row for row in (ngram_screen or {}).get("quantization_screen", []) if row.get("candidate") == "uniform_q4_g32"), None)
    parts = [
        {"family": "embedding_lm_head", "representation": "source_bf16_exact", "runtime_required": True, "qualification": "complete terminal control exists"},
        {"family": "ngram_embedding", "representation": "factorized_lookup_candidate", "runtime_required": True, "qualification": "128-shard global row layout and source-algorithm accepted-token addresses are bound; static global-PQ rate screen is negative, while PLE output fidelity and native compact lookup remain open"},
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
        "semantic_provenance": {
            "parent_model": census["model"],
            "parent_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
            "parameter_count": parameter_count,
            "source_parameter_bytes_indexed": total,
            "source_index_sha256": census["source_index_sha256"],
            "census_sha256": sha(a.census),
            "doctor_nr_sha256": sha(a.doctor),
            "doctor_bank_screen_sha256": sha(a.bank_screen) if bank_screen else None,
            "doctor_ngram_screen_sha256": sha(a.ngram_screen) if ngram_screen else None,
            "ngram_lookup_oracle_sha256": sha(a.ngram_lookup_oracle) if ngram_lookup else None,
            "routed_lowrank_activation_screen_sha256": sha(a.routed_lowrank_activation_screen) if lowrank_activation_screen else None,
            "ngram_pq_screen_sha256": sha(a.ngram_pq_screen) if ngram_pq_screen else None,
            "ngram_access_trace_sha256": sha(a.ngram_access_trace) if ngram_access_trace else None,
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
                        "unmeasured source PLE output/hidden-state control for selective repair",
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
                {"name": "ngram_factorized_lookup_v0", "complete_bits_per_weight": None, "runtime_ready": False, "capability_status": "not-yet-qualified", "open": "source PLE output/activation control, lookup representation fidelity, and native compact lookup", "lookup_oracle_status": (ngram_lookup or {}).get("status"), "access_trace": ngram_access_summary},
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
                    "open": "The source-algorithm address trace is now bound, but a source PLE output/hidden-state control is still required before testing selective hot-row repair or a learned/generative table representation; do not materialize uniform global PQ from this screen.",
                },
            ],
            "portable_identity": {
                "census": "receipts/headless/FLASH_ORGAN_CENSUS.json",
                "doctor": "receipts/headless/FLASH_GRAVITY_DOCTOR_CYCLE.nr.json",
                "routed_lowrank_activation_screen": str(a.routed_lowrank_activation_screen) if lowrank_activation_screen else None,
                "ngram_pq_screen": str(a.ngram_pq_screen) if ngram_pq_screen else None,
                "ngram_access_trace": str(a.ngram_access_trace) if ngram_access_trace else None,
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
                "law": "Shared product quantization can meet sub-bit projected n-gram table budgets arithmetically, but its held-out row distortion is high at the tested rates; the bound token-to-row trace permits future selective-repair controls, but not a PLE-output or capability claim.",
            },
        ],
        "claim_boundary": "This is a complete source-family inventory plus an open-NR accounting baseline. The exact 16.0 EBPW figure belongs only to external source BF16 storage; this receipt claims no complete NR EBPW, closed standalone runnable NR, accepted-token TPS, or capability preservation.",
        "next": "Use the source-bound routed-MoE bridge and direct packed substrate as controls. The n-gram address trace is now closed; obtain a source PLE output/hidden-state control before testing selective hot-row repair or a learned/generative lookup representation. Replace only Pareto survivors, close every dependency before reporting complete EBPW, and build NX only when a machine specialization earns it.",
    }
    ok, bad = validate(doc)
    if not ok:
        raise SystemExit("NR validation failed: " + "; ".join(bad))
    doc["seal_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
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
