#!/usr/bin/env python3.12
"""Materialize G2_COMPLETE_LAYER_PROGRAM.json (Hawking Full Frontier goal, Part 7 - Gate G2).

G2 is the complete-layer gate: run a COMPLETE GPT-OSS layer (layer 0) through each candidate
representation and measure functional quality on REAL residual-stream activations. This module writes
the sealed candidate x complete-layer-0 program the durable G2 controller consumes.

Rows (the candidate set):
  * expert_mlp1: pq_protected_islands, pq_doctor_lowrank, product_quant, source_native_control
  * expert_mlp2: pq_protected_islands, naive_rvq,        product_quant, source_native_control
  * router:      source_native_control
  * attn:        source_native_control

Each row binds: row_id, tensor_class, layer 0, representation_family, family_params, a codebook-aware
exact_budget{target_total_bits, n_weights} (per the gravity_frontier_program budget discipline), the
functional_metrics list, the calibration/validation input counts, and a stopping_rule. The
source_native controls are reference boundaries (kept native, billed at the true native rate) and are
excluded from winner selection.

program_sha256 is sealed over the program CONTENT EXCLUDING the volatile generated_at timestamp (the
goal flagged the timestamp-hash bug): a regenerated program hashes IDENTICALLY. The exclusion set and
the canonical hash come from the controller (single source of truth), so the controller validates the
seal exactly as this builder writes it.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from fractions import Fraction
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gravity_frontier_g2_controller import (  # noqa: E402
    BF16_BPW, CONTROL_FAMILY, HIDDEN, MLP1_ROWS, MLP2_ROWS, MXFP4_BPW, N_EXPERTS, PROGRAM_SCHEMA,
    TOP_K, program_body_hash,
)

REPO = Path(_HERE).resolve().parents[1]
OUT_DIR = REPO / "reports" / "condense" / "general_frontier" / "GENERAL_FRONTIER_PROGRAMS"
OUT_FILE = OUT_DIR / "G2_COMPLETE_LAYER_PROGRAM.json"
PARENT = "openai/gpt-oss-120b"
LAYER = 0

# per-expert dequantized matrix weight counts (the exact per-matrix budget cost).
_N_WEIGHTS = {"expert_mlp1": MLP1_ROWS * HIDDEN, "expert_mlp2": MLP2_ROWS * HIDDEN,
              "router": N_EXPERTS * HIDDEN, "attn": 5120 * HIDDEN}

# Codebook allowance (matches gravity_frontier_program): per-matrix geometries carry a per-matrix
# codebook, so the whole-artifact budget at an index rate is index + codebook. 0.01 bpw is generous
# vs the ~0.002 bpw a k<=256 codebook actually costs.
_CODEBOOK_ALLOWANCE = Fraction(1, 100)
_METADATA_BYTES = 64

FUNCTIONAL_METRICS = [
    "router_topk_agreement", "expert_output_cosine", "expert_output_rel_error",
    "weighted_combine_divergence", "layer_hidden_state_cosine", "complete_layer_bpw",
]


def _target_bits(n_weights: int, rate: Fraction, *, island_frac: Fraction = Fraction(0),
                 doctor_bpw: Fraction = Fraction(0)) -> int:
    """Exact codebook-aware per-matrix budget in bits, mirroring gravity_frontier_program._target_bits.
    The controller FAILS_OVER_BUDGET any geometry whose physical bits exceed this."""
    base = rate * n_weights
    codebook = _CODEBOOK_ALLOWANCE * n_weights
    island = island_frac * n_weights * 16
    doctor = doctor_bpw * n_weights
    return int(math.ceil(base + codebook + island + doctor)) + _METADATA_BYTES * 8


def _native_budget_bits(n_weights: int, native_bpw: float) -> int:
    """Budget for a source_native control: its true native storage rate plus metadata, so a kept-native
    organ passes exactly (it is a reference boundary, not a compression candidate)."""
    return int(math.ceil(native_bpw * n_weights)) + _METADATA_BYTES * 8


# candidate set per tensor class. sub-bit geometries run at the 3/4 stress rate; controls are native.
_R34 = Fraction(3, 4)
_ISLAND_FRAC = Fraction(1, 100)
_DOCTOR_BPW = Fraction(3, 20)

_CANDIDATES: dict[str, list[dict]] = {
    "expert_mlp1": [
        {"family": "pq_protected_islands",
         "params": {"dim": 16, "subspaces": 2, "k": 64, "strategy": "residual_energy",
                    "budget_frac": 0.01},
         "note": "PQ + residual-energy protected islands (heavy-tail reserve)"},
        {"family": "pq_doctor_lowrank",
         "params": {"dim": 16, "subspaces": 2, "k": 64, "doctor": "residual_codebook",
                    "doctor_bpw": 0.15},
         "note": "PQ + Doctor residual codebook inside the same rate"},
        {"family": "product_quant", "params": {"dim": 16, "subspaces": 2, "k": 64},
         "note": "plain product quantization (no Hadamard rotation)"},
        {"family": CONTROL_FAMILY, "params": {},
         "note": "source-native reference boundary (kept native, excluded from selection)"},
    ],
    "expert_mlp2": [
        {"family": "pq_protected_islands",
         "params": {"dim": 16, "subspaces": 2, "k": 64, "strategy": "residual_energy",
                    "budget_frac": 0.01},
         "note": "PQ + residual-energy protected islands (heavy-tail reserve)"},
        {"family": "naive_rvq", "params": {"dim": 16, "k": 64, "stages": 2},
         "note": "residual vector quantization (different representation)"},
        {"family": "product_quant", "params": {"dim": 16, "subspaces": 2, "k": 64},
         "note": "plain product quantization (no Hadamard rotation)"},
        {"family": CONTROL_FAMILY, "params": {},
         "note": "source-native reference boundary (kept native, excluded from selection)"},
    ],
    "router": [
        {"family": CONTROL_FAMILY, "params": {},
         "note": "router gate kept source-native (routing must not drift; a valid control result)"},
    ],
    "attn": [
        {"family": CONTROL_FAMILY, "params": {},
         "note": "attention organ kept source-native (reference boundary)"},
    ],
}

_NATIVE_BPW = {"expert_mlp1": MXFP4_BPW, "expert_mlp2": MXFP4_BPW,
               "router": BF16_BPW, "attn": BF16_BPW}
_TENSOR_GROUP_FMT = {
    "expert_mlp1": "block.{b}.mlp.mlp1_weight", "expert_mlp2": "block.{b}.mlp.mlp2_weight",
    "router": "block.{b}.mlp.gate.weight", "attn": "block.{b}.attn.qkv.weight",
}

_STOPPING_RULE = ("promote the representation with the HIGHEST layer_hidden_state_cosine within the "
                  "exact per-matrix budget per tensor_class; source_native controls are reference "
                  "boundaries and are excluded from selection")


def build() -> dict:
    rows = []
    ridx = 0
    for tensor_class in ("expert_mlp1", "expert_mlp2", "router", "attn"):
        n_weights = _N_WEIGHTS[tensor_class]
        for cand in _CANDIDATES[tensor_class]:
            family = cand["family"]
            params = cand["params"]
            if family == CONTROL_FAMILY:
                rate_label = "native"
                budget = _native_budget_bits(n_weights, _NATIVE_BPW[tensor_class])
            else:
                rate_label = "3/4"
                island = _ISLAND_FRAC if "islands" in family else Fraction(0)
                doctor = _DOCTOR_BPW if "doctor" in family else Fraction(0)
                budget = _target_bits(n_weights, _R34, island_frac=island, doctor_bpw=doctor)
            rows.append({
                "row_id": f"g2_{ridx:03d}",
                "candidate": True,
                "source_revision": PARENT,
                "tensor_class": tensor_class,
                "layer": LAYER,
                "tensor_group_fmt": _TENSOR_GROUP_FMT[tensor_class],
                "representation_family": family,
                "family_params": params,
                "is_control": family == CONTROL_FAMILY,
                "exact_rate": rate_label,
                "exact_budget": {"n_weights_per_matrix": n_weights, "target_total_bits": budget,
                                 "rate": rate_label, "native_bpw": _NATIVE_BPW[tensor_class]},
                "functional_metrics": FUNCTIONAL_METRICS,
                "calibration_inputs": {"kind": "real_block0_moe_inputs", "n_sequences": 3,
                                       "tokens_per_seq": 6, "seed_domain": "calibration"},
                "validation_inputs": {"kind": "real_block0_moe_inputs", "n_sequences": 2,
                                      "tokens_per_seq": 6, "seed_domain": "validation_disjoint"},
                "top_k": TOP_K,
                "cpu_impl": "gravity_frontier_g2 numpy reference forward",
                "metal_impl": "gravity_forge MPS (_kmeans/_assign) for the packer fit",
                "stopping_rule": _STOPPING_RULE,
                "note": cand["note"],
            })
            ridx += 1

    doc = {
        "schema": PROGRAM_SCHEMA,
        "gate": "G2_complete_layer",
        "parent_revision": PARENT,
        "layer": LAYER,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_question": ("Which full-rank sub-bit representation best preserves a COMPLETE "
                                "GPT-OSS layer-0's function on real residual-stream activations?"),
        "search_doctrine": {
            "gate": "complete_layer",
            "ranking_signal": "layer_hidden_state_cosine (highest within budget)",
            "controls_excluded_from_selection": True,
            "class_isolation": "each candidate replaces ONLY its tensor class; the router stays native",
        },
        "functional_metrics": FUNCTIONAL_METRICS,
        "activation_provenance": {
            "path": "gptoss_block.block0_moe_inputs (verified reference forward)",
            "token_source": "synthetic Harmony-ish id sequences pushed through the REAL embedding + "
                            "attention + mlp-norm path",
            "capability_parity": False,
            "honesty": "real-activation FUNCTIONAL PROXY; not real Harmony text, no holdout corpus; "
                       "authorizes no Escape Receipt / Event Horizon seal",
        },
        "rows": rows,
        "totals": {
            "total_rows": len(rows),
            "tensor_classes": ["expert_mlp1", "expert_mlp2", "router", "attn"],
            "candidates_per_expert_class": 4,
            "expected_checkpoint_count": len(rows),
        },
        "exact_budget_discipline": {"codebook_aware": True, "rates_are_rational": True,
                                    "packer_fails_over_budget": True,
                                    "controls_billed_at_native_rate": True},
        "hash_discipline": {"excluded_from_hash": ["program_sha256", "generated_at"],
                            "reason": "stable program identity across regeneration (timestamp-hash bug fix)"},
    }
    doc["program_sha256"] = program_body_hash(doc)
    return doc


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc = build()
    OUT_FILE.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    t = doc["totals"]
    print(json.dumps({"program_sha256": doc["program_sha256"][:16],
                      "total_rows": t["total_rows"],
                      "tensor_classes": t["tensor_classes"],
                      "out": str(OUT_FILE)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
