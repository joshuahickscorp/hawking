#!/usr/bin/env python3
"""Compose the first complete, portable Flash NR candidate.

The candidate covers every Flash organ family.  Exact source-BF16 remains the
fallback control; the routed-expert compact option is included only as a
route-conditioned candidate because its complete dynamic-bank storage and
accepted-token impact are not yet qualified.  This file intentionally contains
no machine bindings; those belong in NX.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from nr_container import validate


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, default=Path("receipts/headless/FLASH_ORGAN_CENSUS.json"))
    ap.add_argument("--doctor", type=Path, default=Path("receipts/headless/FLASH_GRAVITY_DOCTOR_CYCLE.nr.json"))
    ap.add_argument("--bank-screen", type=Path, default=Path("receipts/headless/FLASH_DOCTOR_EXPERT_BANK_SCREEN.json"))
    ap.add_argument("--ngram-screen", type=Path, default=Path("receipts/headless/FLASH_DOCTOR_NGRAM_SCREEN.json"))
    ap.add_argument("--ngram-lookup-oracle", type=Path, default=Path("receipts/headless/FLASH_NGRAM_LOOKUP_ORACLE.json"))
    ap.add_argument("--generation", default="v0", help="explicit NR generation label, e.g. v1")
    ap.add_argument("--out", type=Path, default=Path("receipts/headless/FLASH_COMPLETE_V0.nr.json"))
    a = ap.parse_args()
    census = json.loads(a.census.read_text())
    doctor = json.loads(a.doctor.read_text())
    bank_screen = json.loads(a.bank_screen.read_text()) if a.bank_screen.is_file() else None
    ngram_screen = json.loads(a.ngram_screen.read_text()) if a.ngram_screen.is_file() else None
    ngram_lookup = json.loads(a.ngram_lookup_oracle.read_text()) if a.ngram_lookup_oracle.is_file() else None
    if census.get("schema") != "hawking.flash.organ_census.v1":
        raise SystemExit("unexpected census schema")
    families = {row["family"]: row for row in census.get("family_summary", [])}
    ngram_q4 = next((row for row in (ngram_screen or {}).get("quantization_screen", []) if row.get("candidate") == "uniform_q4_g32"), None)
    parts = [
        {"family": "embedding_lm_head", "representation": "source_bf16_exact", "runtime_required": True, "qualification": "complete terminal control exists"},
        {"family": "ngram_embedding", "representation": "factorized_lookup_candidate", "runtime_required": True, "qualification": "large 128-shard n-gram table inventoried; lookup/factorization fidelity and native path remain open"},
        {"family": "norm", "representation": "source_bf16_exact", "runtime_required": True, "qualification": "source graph contract"},
        {"family": "linear_attention_hyperconnection", "representation": "source_bf16_exact", "runtime_required": True, "qualification": "48-layer source parity and stateful organ/prefix evidence"},
        {"family": "full_attention", "representation": "source_bf16_exact", "runtime_required": True, "qualification": "all full-attention source organs and KV organ evidence"},
        {"family": "mlp_hyperconnection", "representation": "source_bf16_exact", "runtime_required": True, "qualification": "source graph contract"},
        {"family": "shared_expert", "representation": "source_bf16_exact", "runtime_required": True, "qualification": "source graph contract"},
        {"family": "routed_experts", "representation": "route_conditioned_compact_candidate", "runtime_required": True, "qualification": "compact routed-bank exact parity on verified layers; dynamic complete-bank storage not yet qualified"},
        {"family": "other", "representation": "source_bf16_exact", "runtime_required": True, "qualification": "source graph contract"},
    ]
    total = census["source_parameter_bytes_indexed"]
    doc = {
        "nr_version": "1.0.0",
        "nr_kind": "hawking.nos.noetic_representation",
        "artifact_kind": "NR",
        "schema": f"hawking.flash.complete_nr.{a.generation}",
        "generation": a.generation,
        "status": "COMPLETE_HETEROGENEOUS_CANDIDATE_NOT_FOR_PROMOTION",
        "semantic_provenance": {
            "parent_model": census["model"],
            "parent_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
            "parameter_count": total // 2,
            "source_parameter_bytes_indexed": total,
            "source_index_sha256": census["source_index_sha256"],
            "census_sha256": sha(a.census),
            "doctor_nr_sha256": sha(a.doctor),
            "doctor_bank_screen_sha256": sha(a.bank_screen) if bank_screen else None,
            "doctor_ngram_screen_sha256": sha(a.ngram_screen) if ngram_screen else None,
            "ngram_lookup_oracle_sha256": sha(a.ngram_lookup_oracle) if ngram_lookup else None,
        },
        "representation": {
            "scope": "complete 48-layer Flash model",
            "complete_bits_per_weight": 16.0,
            "parts": parts,
            "family_source_bytes": {name: row.get("bytes", 0) for name, row in families.items()},
            "candidate_variants": [
                {"name": "exact_control", "complete_bits_per_weight": 16.0, "runtime_ready": True, "capability_status": "source-control-only"},
                {"name": "route_conditioned_compact_experts_v0", "complete_bits_per_weight": None, "runtime_ready": False, "capability_status": "not-yet-qualified", "open": "dynamic expert-bank representation and accepted-token accounting", "bank_screen": "cross-expert sharing hypothesis is weak in sampled real weights; pursue active-route storage, not unconditional shared basis"},
                {"name": "ngram_factorized_lookup_v0", "complete_bits_per_weight": None, "runtime_ready": False, "capability_status": "not-yet-qualified", "open": "n-gram table information floor, collision/lookup semantics, and native compact lookup", "lookup_oracle_status": (ngram_lookup or {}).get("status")},
                {"name": "ngram_uniform_q4_g32_v0", "complete_bits_per_weight": None, "runtime_ready": False, "capability_status": "stage-a-only", "nominal_bpw": (ngram_q4 or {}).get("nominal_bpw"), "sample_cosine": (ngram_q4 or {}).get("sample_cosine"), "open": "activation/output sensitivity and native compact lookup"},
                {"name": "ngram_packed_q4_g32_lookup_v1", "complete_bits_per_weight": 4.0 + 32.0 / 32.0, "runtime_ready": False, "capability_status": "row-oracle-only", "sample_rows": ((ngram_lookup or {}).get("source") or {}).get("sample_rows"), "mean_lookup_ns": ((ngram_lookup or {}).get("bench") or {}).get("lookup_ns_mean"), "open": "native lookup kernel, activation sensitivity, collision semantics, and complete-token impact"},
                {"name": "ngram_packed_q3_g32_lookup_v1", "complete_bits_per_weight": 3.0 + 32.0 / 32.0, "runtime_ready": False, "capability_status": "row-oracle-only", "sample_rows": ((ngram_lookup or {}).get("source") or {}).get("sample_rows"), "mean_lookup_ns": ((ngram_lookup or {}).get("bench") or {}).get("lookup_ns_mean"), "open": "native lookup kernel, activation sensitivity, collision semantics, and complete-token impact"},
            ],
            "portable_identity": {
                "census": "receipts/headless/FLASH_ORGAN_CENSUS.json",
                "doctor": "receipts/headless/FLASH_GRAVITY_DOCTOR_CYCLE.nr.json",
            },
        },
        "kernel_requirements": [
            {"requires": "source_bf16_gemv_family", "applies_to": "exact fallback organs"},
            {"requires": "grouped_absmax_decoder", "applies_to": "future compact routed-expert candidates only"},
            {"requires": "route_conditioned_expert_selection", "applies_to": "routed expert family"},
            {"requires": "gated_delta_recurrence", "applies_to": "linear-attention state"},
            {"requires": "causal_attention_kv_state", "applies_to": "full-attention state"},
        ],
        "promotion": {"allowed": False, "blockers": ["complete accepted-token session", "complete candidate byte ledger", "capability suite", "machine-bound NX compilation"]},
        "bench": {"state": "UNKNOWN", "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "recorded_by": "tools/flash_complete_nr.py representation composition", "machine": "Apple host; no execution benchmark", "rule": "S032 §3 -- NR composition carries no physical timing claim"},
        "negative_science": [
            {"receipt": str(a.bank_screen), "finding": ((bank_screen or {}).get("population") or {}).get("cross_expert_gate_up_mean_cosine"), "law": "do not assume routed experts share a low-dimensional basis; sampled gate-up expert cosine was near zero"},
            {"receipt": str(a.ngram_screen), "finding": ((ngram_screen or {}).get("population") or {}).get("mean_pairwise_row_cosine"), "law": "do not assume n-gram shards share a low-dimensional basis; shard-row similarity was near zero"},
        ],
        "claim_boundary": "Complete heterogeneous portable NR candidate and exact control composition. It does not claim a compact complete representation, accepted-token TPS, EBPW, capability preservation, or machine execution.",
        "next": "compile exact-control NX first, then replace only Pareto-surviving organs and measure complete accepted-token impact",
    }
    ok, bad = validate(doc)
    if not ok:
        raise SystemExit("NR validation failed: " + "; ".join(bad))
    doc["seal_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({"status": doc["status"], "families": len(parts), "complete_bits_per_weight": 16.0, "out": str(a.out), "seal": doc["seal_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
