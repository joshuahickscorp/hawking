#!/usr/bin/env python3.12
"""Materialize G3_CROSS_LAYER_PROGRAM.json (Hawking Full Frontier goal, Part 8 - Gate G3).

G2 measured the complete-layer functional quality of each candidate representation on ONE layer
(layer 0). G3 is the next fidelity: CROSS-LAYER TRANSFER. It runs the SAME complete-layer measurement
over an EARLY (0), a MIDDLE (18) and a LATE (35) GPT-OSS-120B layer, so the Frontier can answer the
real question the negative science leaves open: does the geometry that best preserves layer 0 still
best preserve the residual stream deeper in the network, or does the winning representation change
with depth?

Rows are the G2 geometry candidate set (the winners + their in-class rivals + the source-native
reference boundary) crossed with the three probe layers:

  expert_mlp1 : pq_protected_islands, pq_doctor_lowrank, product_quant, source_native_control
  expert_mlp2 : pq_protected_islands, naive_rvq,        product_quant, source_native_control
              x layers [0 (early), 18 (mid), 35 (late)]  =  8 rows/layer * 3 layers = 24 rows.

Each row binds: row_id, layer, layer_role, tensor_class, representation_family, family_params, a
codebook-aware exact_budget{target_total_bits, n_weights}, the functional_metrics list, the
EXECUTION GENERATION ("M": the verified parity-preserving m2_shared_lookup_linear grammar is an
AVAILABLE provider; Gen-F direct-compact is the wall-safe default), the base_execution_provider
descriptor, and a stopping_rule. The source_native controls are reference boundaries (kept native,
billed at the true native rate) and are excluded from winner selection and the transfer verdict.

EXECUTION-vs-REPRESENTATION honesty: Generation M promoted an EXECUTION grammar (lookup-linear) that
is EXACT-parity to the Gen-F direct-compact path, NOT a new representation. So the functional
divergence a candidate produces is identical whether it is executed via direct-compact or the Gen-M
lookup-linear provider; binding execution_generation="M" changes the mechanical cost, never the
measured quality. The representation science stays Gravity-NEGATIVE (sub-bit); capability_parity is
False. G3 authorizes no Escape Receipt and no Event Horizon seal.

program_sha256 is sealed over the program CONTENT EXCLUDING the volatile generated_at timestamp (the
same stable-identity discipline as G2, via the frozen controller's program_body_hash). A regenerated
program hashes IDENTICALLY.
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

# Reuse the FROZEN G2 controller's geometry constants + the stable program hash (single source of
# truth: the G3 controller validates the seal exactly as this builder writes it).
from gravity_frontier_g2_controller import (  # noqa: E402
    BF16_BPW, CONTROL_FAMILY, HIDDEN, MLP1_ROWS, MLP2_ROWS, MXFP4_BPW, N_EXPERTS, program_body_hash,
)
from gravity_frontier_g3_controller import (  # noqa: E402
    BASE_EXECUTION_PROVIDER, EXECUTION_GENERATION, FUNCTIONAL_METRICS, LAYERS, LAYER_ROLES,
    PROGRAM_SCHEMA, TOP_K,
)

REPO = Path(_HERE).resolve().parents[1]
OUT_DIR = (REPO / "reports" / "condense" / "general_frontier"
           / "GENERAL_FRONTIER_PROGRAMS")
OUT_FILE = OUT_DIR / "G3_CROSS_LAYER_PROGRAM.json"
PARENT = "openai/gpt-oss-120b"

# per-matrix dequantized weight counts (the exact per-matrix budget cost), identical across blocks.
_N_WEIGHTS = {"expert_mlp1": MLP1_ROWS * HIDDEN, "expert_mlp2": MLP2_ROWS * HIDDEN}
_NATIVE_BPW = {"expert_mlp1": MXFP4_BPW, "expert_mlp2": MXFP4_BPW}
_TENSOR_GROUP_FMT = {"expert_mlp1": "block.{b}.mlp.mlp1_weight",
                     "expert_mlp2": "block.{b}.mlp.mlp2_weight"}

# Exact codebook-aware budget discipline, byte-identical to G2 (gravity_frontier_g2_program).
_CODEBOOK_ALLOWANCE = Fraction(1, 100)
_METADATA_BYTES = 64
_R34 = Fraction(3, 4)
_ISLAND_FRAC = Fraction(1, 100)
_DOCTOR_BPW = Fraction(3, 20)


def _target_bits(n_weights: int, rate: Fraction, *, island_frac: Fraction = Fraction(0),
                 doctor_bpw: Fraction = Fraction(0)) -> int:
    base = rate * n_weights
    codebook = _CODEBOOK_ALLOWANCE * n_weights
    island = island_frac * n_weights * 16
    doctor = doctor_bpw * n_weights
    return int(math.ceil(base + codebook + island + doctor)) + _METADATA_BYTES * 8


def _native_budget_bits(n_weights: int, native_bpw: float) -> int:
    return int(math.ceil(native_bpw * n_weights)) + _METADATA_BYTES * 8


# candidate set per tensor class (the G2 winners + their in-class rivals + the native control).
_CANDIDATES: dict[str, list[dict]] = {
    "expert_mlp1": [
        {"family": "pq_protected_islands",
         "params": {"dim": 16, "subspaces": 2, "k": 64, "strategy": "residual_energy",
                    "budget_frac": 0.01},
         "note": "PQ + residual-energy protected islands (heavy-tail reserve)"},
        {"family": "pq_doctor_lowrank",
         "params": {"dim": 16, "subspaces": 2, "k": 64, "doctor": "residual_codebook",
                    "doctor_bpw": 0.15},
         "note": "PQ + Doctor residual codebook inside the same rate (G2 layer-0 mlp1 winner)"},
        {"family": "product_quant", "params": {"dim": 16, "subspaces": 2, "k": 64},
         "note": "plain product quantization (no Hadamard rotation)"},
        {"family": CONTROL_FAMILY, "params": {},
         "note": "source-native reference boundary (kept native, excluded from selection)"},
    ],
    "expert_mlp2": [
        {"family": "pq_protected_islands",
         "params": {"dim": 16, "subspaces": 2, "k": 64, "strategy": "residual_energy",
                    "budget_frac": 0.01},
         "note": "PQ + residual-energy protected islands (G2 layer-0 mlp2 winner)"},
        {"family": "naive_rvq", "params": {"dim": 16, "k": 64, "stages": 2},
         "note": "residual vector quantization (different representation)"},
        {"family": "product_quant", "params": {"dim": 16, "subspaces": 2, "k": 64},
         "note": "plain product quantization (no Hadamard rotation)"},
        {"family": CONTROL_FAMILY, "params": {},
         "note": "source-native reference boundary (kept native, excluded from selection)"},
    ],
}

_STOPPING_RULE = ("per (tensor_class, layer) promote the representation with the HIGHEST "
                  "layer_hidden_state_cosine within the exact per-matrix budget; source_native "
                  "controls are reference boundaries and are excluded; then record whether the "
                  "layer-0 winner still wins at the mid (18) and late (35) layers (cross-layer "
                  "transfer)")


def build() -> dict:
    rows = []
    ridx = 0
    for layer, role in zip(LAYERS, LAYER_ROLES):
        for tensor_class in ("expert_mlp1", "expert_mlp2"):
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
                    "row_id": f"g3_{ridx:03d}",
                    "candidate": True,
                    "source_revision": PARENT,
                    "tensor_class": tensor_class,
                    "layer": layer,
                    "layer_role": role,
                    "tensor_group_fmt": _TENSOR_GROUP_FMT[tensor_class],
                    "representation_family": family,
                    "family_params": params,
                    "is_control": family == CONTROL_FAMILY,
                    "exact_rate": rate_label,
                    "exact_budget": {"n_weights_per_matrix": n_weights,
                                     "target_total_bits": budget, "rate": rate_label,
                                     "native_bpw": _NATIVE_BPW[tensor_class]},
                    "functional_metrics": FUNCTIONAL_METRICS,
                    "execution_generation": EXECUTION_GENERATION,
                    "base_execution_provider": BASE_EXECUTION_PROVIDER,
                    "calibration_inputs": {"kind": "real_block_n_moe_inputs", "n_sequences": 3,
                                           "tokens_per_seq": 6, "seed_domain": "calibration",
                                           "layer": layer},
                    "validation_inputs": {"kind": "real_block_n_moe_inputs", "n_sequences": 2,
                                          "tokens_per_seq": 6, "seed_domain": "validation_disjoint",
                                          "layer": layer},
                    "top_k": TOP_K,
                    "cpu_impl": "gravity_frontier_g3 numpy reference forward (generalized block N)",
                    "metal_impl": "gravity_forge MPS (_kmeans/_assign) for the packer fit",
                    "stopping_rule": _STOPPING_RULE,
                    "note": cand["note"],
                })
                ridx += 1

    doc = {
        "schema": PROGRAM_SCHEMA,
        "gate": "G3_cross_layer_transfer",
        "parent_revision": PARENT,
        "layers": list(LAYERS),
        "layer_roles": dict(zip((str(x) for x in LAYERS), LAYER_ROLES)),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_question": ("Does the full-rank sub-bit representation that best preserves a "
                                "COMPLETE GPT-OSS layer 0 still best preserve the residual stream at "
                                "a middle (18) and a late (35) layer? i.e. does the winning geometry "
                                "TRANSFER across depth?"),
        "execution_generation": EXECUTION_GENERATION,
        "base_execution_provider": BASE_EXECUTION_PROVIDER,
        "search_doctrine": {
            "gate": "cross_layer_transfer",
            "ranking_signal": "layer_hidden_state_cosine (highest within budget per (class, layer))",
            "controls_excluded_from_selection": True,
            "class_isolation": "each candidate replaces ONLY its tensor class; the router stays native",
            "transfer_verdict": ("compare the per-layer winner family to the layer-0 winner family "
                                 "per tensor_class; report transfers_to_mid / transfers_to_late"),
            "execution_note": ("execution_generation=M is quality-NEUTRAL (m2_shared_lookup_linear "
                               "is EXACT parity to Gen-F direct-compact); it changes mechanical cost, "
                               "never the measured divergence"),
        },
        "functional_metrics": FUNCTIONAL_METRICS,
        "activation_provenance": {
            "path": "gravity_frontier_g3_controller.block_n_moe_inputs (generalized reference forward)",
            "token_source": ("synthetic Harmony-ish id sequences (SAME tokens at every layer) pushed "
                             "through the REAL embedding + block-N attention + block-N mlp-norm path"),
            "capability_parity": False,
            "honesty": ("generalized-block APPROXIMATION: block-N attention is a from-config forward "
                        "(RoPE theta 150000, GQA 64/8, sinks, sliding-window inactive at seq<128), "
                        "exactly as the block-0 forward is; it is valid ONLY for the RELATIVE "
                        "orig-vs-packed divergence, where the shared residual + shared reference "
                        "SwiGLU largely cancel the approximation. Not real Harmony text, no holdout "
                        "corpus; authorizes no Escape Receipt / Event Horizon seal."),
        },
        "rows": rows,
        "totals": {
            "total_rows": len(rows),
            "layers": list(LAYERS),
            "tensor_classes": ["expert_mlp1", "expert_mlp2"],
            "candidates_per_expert_class": 4,
            "rows_per_layer": 8,
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
                      "total_rows": t["total_rows"], "layers": t["layers"],
                      "rows_per_layer": t["rows_per_layer"], "out": str(OUT_FILE)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
