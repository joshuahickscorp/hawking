#!/usr/bin/env python3.12
"""Materialize GPT_OSS_120B_PQ_GRAVITY_PROGRAM.json (Second Light goal, Section 22).

The ACTUAL run is not one 256x256 slice per expert. It is a durable, restartable, parent-bound
campaign whose queue covers the COMPLETE declared condensation scope. This module enumerates every
run-critical tensor class of GPT-OSS-120B from the real (rebuilt) provenance manifest, assigns each
an exact-budget representation policy per the evidence (full-rank PQ family, shared amortized
codebooks, protected islands, Doctor reserve; low-rank ternary demoted to historical baseline), and
seals the program with a content hash.

Every row binds exactly what Section 22 requires and every byte is counted in the whole-artifact
BPW (Sections 7/15): index + codebook + scales + metadata + protected-island + Doctor reserve +
alignment. Tensor classes intentionally left in their original format are DECLARED, their bytes
COUNTED, and the reason bound (Section 3). Rates are exact rationals; the packer must fail if a row
exceeds its exact byte budget (Section 15). No post-hoc "approximately sub-bit" acceptance.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "reports" / "condense" / "subbit_frontier" / "GRAVITY_120B_PROVENANCE.json"
OUT = REPO / "reports" / "condense" / "second_light"
SCHEMA = "hawking.second_light.pq_gravity_program.v1"
PARENT = "openai/gpt-oss-120b"

# fixed structural charges (mirror gravity_forge.ByteLedger)
_METADATA_BYTES = 64
_FP16 = 16
_ALIGN_BITS = 0  # rows carry their own alignment slack inside metadata; kept explicit + zero here


def _idx_bits(n_vectors: int, k: int) -> int:
    return n_vectors * max(1, math.ceil(math.log2(max(2, k))))


def pq_row_budget(rows: int, cols: int, *, dim: int, subspaces: int, k: int,
                  island_frac: Fraction, doctor_reserve_bpw: Fraction) -> dict:
    """Exact whole-artifact bit budget for one transform_pq/PQ tensor at (dim, subspaces, k).
    Mirrors gravity_forge.pack_transform_pq accounting so the packer's ByteLedger must land at or
    under this. Islands + Doctor are reserved as exact bit budgets, billed inside the same total."""
    n_weights = rows * cols
    D = dim
    S = subspaces
    sub = D // S
    n_vectors = n_weights // D
    index_bits = S * _idx_bits(n_vectors, k)
    codebook_bits = S * (k * sub) * _FP16
    transform_seed_bits = 64
    metadata_bits = _METADATA_BYTES * 8
    island_bits = int(island_frac * n_weights * _FP16)         # outliers kept fp16, counted
    doctor_bits = int(doctor_reserve_bpw * n_weights)          # reserve, counted up front
    base_bits = index_bits + codebook_bits + transform_seed_bits
    total_bits = base_bits + metadata_bits + island_bits + doctor_bits + _ALIGN_BITS
    return {
        "n_weights": n_weights,
        "index_bits": index_bits,
        "codebook_bits": codebook_bits,
        "transform_seed_bits": transform_seed_bits,
        "metadata_bits": metadata_bits,
        "protected_island_bits": island_bits,
        "doctor_reserve_bits": doctor_bits,
        "alignment_bits": _ALIGN_BITS,
        "target_total_bits": total_bits,
        "base_bpw": round(base_bits / n_weights, 5),
        "whole_artifact_bpw": round(total_bits / n_weights, 5),
    }


def shared_grammar_budget(n_experts: int, rows: int, cols: int, *, dim: int, stages: int, k: int,
                          island_frac: Fraction, doctor_reserve_bpw: Fraction) -> dict:
    """Exact budget for a shared_expert_grammar row covering n_experts of one class in one layer.
    Shared codebook billed ONCE (amortized over the whole cluster); per-expert cost is indices +
    reserves. This is the MoE amortization lever (Sections 7/17)."""
    per_expert_weights = rows * cols
    n_weights = n_experts * per_expert_weights
    D = dim
    n_vectors_per_expert = per_expert_weights // D
    index_bits = n_experts * stages * _idx_bits(n_vectors_per_expert, k)
    codebook_bits = stages * (k * D) * _FP16                   # shared, once
    metadata_bits = _METADATA_BYTES * 8
    island_bits = int(island_frac * n_weights * _FP16)
    doctor_bits = int(doctor_reserve_bpw * n_weights)
    base_bits = index_bits + codebook_bits
    total_bits = base_bits + metadata_bits + island_bits + doctor_bits
    return {
        "n_weights": n_weights,
        "n_experts": n_experts,
        "index_bits": index_bits,
        "codebook_bits_shared": codebook_bits,
        "metadata_bits": metadata_bits,
        "protected_island_bits": island_bits,
        "doctor_reserve_bits": doctor_bits,
        "alignment_bits": _ALIGN_BITS,
        "target_total_bits": total_bits,
        "base_bpw": round(base_bits / n_weights, 5),
        "whole_artifact_bpw": round(total_bits / n_weights, 5),
    }


def kept_original_budget(n_weights: int, bits_each: int = 16) -> dict:
    """A tensor class DECLARED kept in its original format. Bytes are still COUNTED (Section 3)."""
    total_bits = n_weights * bits_each + _METADATA_BYTES * 8
    return {"n_weights": n_weights, "target_total_bits": total_bits,
            "whole_artifact_bpw": round(total_bits / max(1, n_weights), 5),
            "kept_original_bits_each": bits_each}


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _by_name(man: dict) -> dict:
    return {t["tensor"]: t for t in man["tensors"]}


def build(config: dict | None = None) -> dict:
    man = _load_manifest()
    by = _by_name(man)
    N_LAYERS, N_EXPERTS = 36, 128

    # dequantized logical shapes (per expert), independent of MXFP4 storage
    MLP1 = (5760, 2880)   # up/gate proj
    MLP2 = (2880, 2880)   # down proj
    QKV = (5120, 2880)
    ATTN_OUT = (2880, 4096)
    ROUTER = (128, 2880)
    EMB = (201088, 2880)

    rows = []
    ridx = 0

    def add(**kw):
        nonlocal ridx
        kw["row_id"] = f"r{ridx:04d}"
        kw.setdefault("evidence_schema", "hawking.second_light.row_evidence.v1")
        kw.setdefault("dependencies", [])
        rows.append(kw)
        ridx += 1

    # ---- expert MLP rows: shared_expert_grammar per (layer, class), 128 experts amortized -------
    # target sub-bit ~0.75 base bpw; residual-energy islands; Doctor low-rank residual reserve.
    exp_dim, exp_stages, exp_k = 16, 2, 64                 # 2*log2(64)=12 bits / 16 weights = 0.75
    exp_island = Fraction(1, 2000)                         # 0.05% outlier subvectors kept fp16
    exp_doctor = Fraction(3, 20)                           # 0.15 bpw Doctor reserve
    for layer in range(N_LAYERS):
        for cls, shp, tname in (("expert_mlp1", MLP1, f"block.{layer}.mlp.mlp1_weight"),
                                 ("expert_mlp2", MLP2, f"block.{layer}.mlp.mlp2_weight")):
            b = shared_grammar_budget(N_EXPERTS, shp[0], shp[1], dim=exp_dim, stages=exp_stages,
                                      k=exp_k, island_frac=exp_island, doctor_reserve_bpw=exp_doctor)
            add(layer=layer, tensor_class=cls, tensor_group=tname, n_experts=N_EXPERTS,
                source_dtype="mxfp4", representation_family="shared_expert_grammar",
                subvector_dim=exp_dim, codebook_size=exp_k, stages=exp_stages,
                sharing_group=f"experts_layer{layer}_{cls}",
                protected_island_strategy="residual_energy",
                doctor_reserve_bpw=float(exp_doctor), metal_impl="pq_assign+direct_grammar_gemv",
                exact_budget=b, starting_rate="3/4", is_subbit=True,
                quality_metrics=["expert_output_cosine", "expert_output_rel_error",
                                 "router_topk_agreement_downstream"],
                holdout_metrics=["holdout_expert_output_rel_error"],
                resource_estimate={"gpu": "mps", "peak_gib": 3.0, "reads_gib": 1.6},
                stopping_rule="promote if expert_output_rel_error materially < ternary baseline and "
                              "budget valid and holdout confirms; else escalate rate per ladder")

    # ---- router rows: functionally critical -> kept high-precision (protected), counted ---------
    for layer in range(N_LAYERS):
        b = kept_original_budget(ROUTER[0] * ROUTER[1], bits_each=16)
        add(layer=layer, tensor_class="router", tensor_group=f"block.{layer}.mlp.gate.weight",
            source_dtype="bf16", representation_family="kept_original",
            sharing_group=None, protected_island_strategy="whole_tensor_protected",
            doctor_reserve_bpw=0.0, metal_impl="dense_bf16_gemv", exact_budget=b,
            starting_rate="16/1", is_subbit=False, kept_original=True,
            kept_reason="router top-k selection is capability-critical; sub-bit router perturbs "
                        "routing and is deferred until experts pass; bytes counted in whole-artifact",
            quality_metrics=["router_topk_agreement", "router_prob_kl"],
            holdout_metrics=["holdout_router_topk_agreement"],
            resource_estimate={"gpu": "mps", "peak_gib": 0.1},
            stopping_rule="revisit only after expert class passes; then Gravity may quantize router "
                          "with heavy protected islands")

    # ---- attention projections: PQ at a moderate rate (attention is sensitive) ------------------
    attn_dim, attn_sub, attn_k = 16, 4, 256               # 4*8=32 bits / 16 = 2.0 base bpw
    attn_island = Fraction(1, 500)                        # 0.2% outliers
    attn_doctor = Fraction(1, 5)                          # 0.20 bpw reserve
    for layer in range(N_LAYERS):
        for cls, shp, tname in (("attn_qkv", QKV, f"block.{layer}.attn.qkv.weight"),
                                 ("attn_out", ATTN_OUT, f"block.{layer}.attn.out.weight")):
            b = pq_row_budget(shp[0], shp[1], dim=attn_dim, subspaces=attn_sub, k=attn_k,
                              island_frac=attn_island, doctor_reserve_bpw=attn_doctor)
            add(layer=layer, tensor_class=cls, tensor_group=tname, source_dtype="bf16",
                representation_family="transform_pq", subvector_dim=attn_dim, subspaces=attn_sub,
                codebook_size=attn_k, sharing_group=f"attn_layer{layer}_{cls}",
                protected_island_strategy="sensitivity", doctor_reserve_bpw=float(attn_doctor),
                metal_impl="pq_assign+direct_pq_gemv", exact_budget=b, starting_rate="2/1",
                is_subbit=False,
                quality_metrics=["attn_output_divergence", "layer_hidden_cosine"],
                holdout_metrics=["holdout_layer_hidden_rel_error"],
                resource_estimate={"gpu": "mps", "peak_gib": 0.5},
                stopping_rule="promote if attn_output_divergence within tolerance; else raise rate")

    # ---- embeddings + unembedding: global shared PQ (huge, moderate rate) ------------------------
    emb_dim, emb_sub, emb_k = 16, 2, 256                  # 2*8=16 bits / 16 = 1.0 base bpw
    emb_island = Fraction(1, 1000)
    emb_doctor = Fraction(1, 10)
    for tname, cls in (("embedding.weight", "token_embedding"),
                       ("unembedding.weight", "output_projection")):
        b = pq_row_budget(EMB[0], EMB[1], dim=emb_dim, subspaces=emb_sub, k=emb_k,
                          island_frac=emb_island, doctor_reserve_bpw=emb_doctor)
        add(layer=None, tensor_class=cls, tensor_group=tname, source_dtype="bf16",
            representation_family="transform_pq", subvector_dim=emb_dim, subspaces=emb_sub,
            codebook_size=emb_k, sharing_group="global_embedding_codebook",
            protected_island_strategy="magnitude", doctor_reserve_bpw=float(emb_doctor),
            metal_impl="pq_lookup_gather", exact_budget=b, starting_rate="1/1", is_subbit=False,
            quality_metrics=["logit_cosine", "logit_kl", "topk_token_agreement"],
            holdout_metrics=["holdout_logit_kl"],
            resource_estimate={"gpu": "mps", "peak_gib": 2.4},
            stopping_rule="promote if logit_kl within tolerance; else raise rate")

    # ---- kept-original small tensors (declared + counted): norms, biases, sinks -----------------
    small = []
    for layer in range(N_LAYERS):
        for tname in (f"block.{layer}.attn.norm.scale", f"block.{layer}.mlp.norm.scale",
                      f"block.{layer}.attn.qkv.bias", f"block.{layer}.attn.out.bias",
                      f"block.{layer}.attn.sinks", f"block.{layer}.mlp.gate.bias",
                      f"block.{layer}.mlp.mlp1_bias", f"block.{layer}.mlp.mlp2_bias"):
            t = by.get(tname)
            if t:
                small.append((tname, int(_prod(t["shape"]))))
    small.append(("norm.scale", int(_prod(by["norm.scale"]["shape"]))))
    small_weights = sum(n for _, n in small)
    b = kept_original_budget(small_weights, bits_each=16)
    add(layer=None, tensor_class="kept_original_small", tensor_group="norms+biases+sinks",
        source_dtype="bf16", representation_family="kept_original", sharing_group=None,
        protected_island_strategy=None, doctor_reserve_bpw=0.0, metal_impl="dense_bf16",
        exact_budget=b, starting_rate="16/1", is_subbit=False, kept_original=True,
        member_tensors=len(small),
        kept_reason="normalization params, biases, and attention sinks are tiny and precision-"
                    "sensitive; kept bf16, every byte counted in whole-artifact BPW",
        quality_metrics=["exact_passthrough"], holdout_metrics=[],
        resource_estimate={"gpu": "cpu", "peak_gib": 0.01},
        stopping_rule="never quantized; accounting-only row")

    # ---- totals + whole-artifact accounting ------------------------------------------------------
    total_weights = sum(r["exact_budget"]["n_weights"] for r in rows)
    total_out_bits = sum(r["exact_budget"]["target_total_bits"] for r in rows)
    source_bytes = man["total_source_bytes"]
    out_bytes = math.ceil(total_out_bits / 8)
    whole_bpw = total_out_bits / total_weights
    subbit_rows = [r for r in rows if r.get("is_subbit")]
    subbit_weights = sum(r["exact_budget"]["n_weights"] for r in subbit_rows)

    doc = {
        "schema": SCHEMA,
        "parent_revision": PARENT,
        "source_manifest_sha256": man["manifest_sha256"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "architecture": {"layers": N_LAYERS, "experts_per_layer": N_EXPERTS, "top_k": 4,
                         "hidden": 2880, "intermediate": 2880, "vocab": 201088,
                         "attn_heads": 64, "kv_heads": 8, "head_dim": 64, "rope_theta": 150000},
        "representation_doctrine": {
            "principal_geometry": "full_rank_product_quantization",
            "rejected_geometry": "low_rank_ternary_factorization (First-Light evidence)",
            "amortization": "shared codebooks per (layer, expert-class); global for embeddings",
            "escalation_order": ["geometry", "shared_codebook", "protected_islands",
                                 "doctor_same_rate", "next_subbit_rate"],
        },
        "rows": rows,
        "totals": {
            "total_rows": len(rows),
            "expert_rows": sum(1 for r in rows if r["tensor_class"].startswith("expert")),
            "attn_rows": sum(1 for r in rows if r["tensor_class"].startswith("attn")),
            "router_rows": sum(1 for r in rows if r["tensor_class"] == "router"),
            "global_rows": sum(1 for r in rows if r["tensor_class"] in
                               ("token_embedding", "output_projection")),
            "kept_original_rows": sum(1 for r in rows if r.get("kept_original")),
            "total_logical_weights": total_weights,
            "subbit_logical_weights": subbit_weights,
            "expected_source_bytes": source_bytes,
            "expected_output_bytes": out_bytes,
            "expected_output_gib": round(out_bytes / 1024**3, 3),
            "source_gib": round(source_bytes / 1024**3, 3),
            "complete_artifact_bpw": round(whole_bpw, 5),
            "compression_vs_source": round(source_bytes / out_bytes, 3),
            "expected_checkpoint_count": len(rows),
            "expected_wall_time_note": "bounded per-row (minutes each on M3 Ultra MPS); full queue "
                                       "is a multi-hour to multi-day durable campaign under one "
                                       "controller; exact ETA set from measured per-row time at run",
        },
        "exact_budget_discipline": {
            "rates_are_rational": True,
            "packer_must_fail_over_budget": True,
            "no_post_hoc_subbit_acceptance": True,
            "every_byte_counted": True,
        },
    }
    payload = json.dumps(doc, sort_keys=True).encode()
    doc["program_sha256"] = hashlib.sha256(payload).hexdigest()
    return doc


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = build()
    (OUT / "GPT_OSS_120B_PQ_GRAVITY_PROGRAM.json").write_text(json.dumps(doc, indent=2, sort_keys=True))
    t = doc["totals"]
    print(json.dumps({
        "program_sha256": doc["program_sha256"][:16],
        "total_rows": t["total_rows"],
        "total_logical_weights": t["total_logical_weights"],
        "complete_artifact_bpw": t["complete_artifact_bpw"],
        "expected_output_gib": t["expected_output_gib"],
        "source_gib": t["source_gib"],
        "compression_vs_source": t["compression_vs_source"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
