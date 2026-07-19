#!/usr/bin/env python3.12
"""Materialize GPT_OSS_120B_GRAVITY_FRONTIER_PROGRAM.json (Gravity Frontier goal, Section 14).

Second Light built ONE PQ geometry into a 0.770-BPW artifact with negative function. The Frontier
SEARCHES representation GEOMETRY before rate (goal Section 9 priority order): change representation
-> change sharing -> change subvector geometry -> change island allocation -> Doctor -> residual/
additive stages -> only then raise rate. Each queue row is a geometry TRIAL competing on FUNCTIONAL
output divergence (real reference forward, routed experts exercised), evaluated on a representative
expert sample across early/middle/late layers. Trials at the same rate compete; the winner is the
lowest functional divergence within its exact byte budget. Rate rises only after geometry is
exhausted at a rate. Exact integer budgets; the packer fails if the budget is exceeded.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FR = REPO / "reports" / "condense" / "gravity_frontier"
SCHEMA = "hawking.gpt_oss_120b.gravity_frontier_program.v1"
PARENT = "openai/gpt-oss-120b"
SAMPLE_LAYERS = [0, 18, 35]              # early / middle / late

# per-expert dequantized matrix shapes
MLP1 = (5760, 2880)
MLP2 = (2880, 2880)


# Per-matrix geometries carry a per-matrix codebook (not amortized across experts the way a shared
# grammar is), so the WHOLE-artifact rate at a given index rate is index + codebook. The exact budget
# therefore includes a codebook allowance (0.01 bpw, generous vs the ~0.002 bpw a k<=256 codebook
# actually costs). The rate LABEL names the index/base rate; the whole budget is honestly slightly
# above it. A geometry whose components exceed even this allowance is correctly FAILED_OVER_BUDGET.
_CODEBOOK_ALLOWANCE = Fraction(1, 100)


def _target_bits(n_weights: int, rate: Fraction, island_frac: Fraction = Fraction(0),
                 doctor_bpw: Fraction = Fraction(0)) -> int:
    base = rate * n_weights
    codebook = _CODEBOOK_ALLOWANCE * n_weights
    island = island_frac * n_weights * 16
    doctor = doctor_bpw * n_weights
    return int(math.ceil(base + codebook + island + doctor)) + 64 * 8   # + metadata


def build() -> dict:
    R34 = Fraction(3, 4)     # 0.75 bpw stress rate
    R1 = Fraction(1, 1)      # 1.0 bpw escalation

    # geometry candidates (per-expert-matrix packers measured by functional output divergence).
    # priority_rank encodes Section 9: 1=representation, 2=sharing, 3=subvector geom, 4=islands,
    # 5=Doctor, 6=residual/additive; rate rises only after a rate's geometries are exhausted.
    candidates = [
        {"family": "transform_pq", "params": {"dim": 16, "subspaces": 2, "k": 64},
         "priority_rank": 1, "note": "Second Light baseline geometry (rotated PQ)"},
        {"family": "product_quant", "params": {"dim": 16, "subspaces": 2, "k": 64},
         "priority_rank": 1, "note": "plain PQ, no Hadamard rotation (different representation)"},
        {"family": "naive_rvq", "params": {"dim": 16, "k": 64, "stages": 2},
         "priority_rank": 1, "note": "residual vector quantization (different representation)"},
        {"family": "transform_pq", "params": {"dim": 32, "subspaces": 4, "k": 64},
         "priority_rank": 3, "note": "alternative subvector geometry (dim 32, 4 subspaces)"},
        {"family": "transform_pq", "params": {"dim": 8, "subspaces": 1, "k": 16},
         "priority_rank": 3, "note": "alternative subvector geometry (dim 8, single subspace)"},
        {"family": "pq_protected_islands", "params": {"dim": 16, "subspaces": 2, "k": 64,
         "strategy": "residual_energy", "budget_frac": 0.01},
         "priority_rank": 4, "note": "PQ + residual-energy protected islands (heavy-tail reserve)"},
        {"family": "pq_doctor_lowrank", "params": {"dim": 16, "subspaces": 2, "k": 64,
         "doctor": "residual_codebook", "doctor_bpw": 0.15},
         "priority_rank": 5, "note": "PQ + Doctor residual codebook inside the same rate"},
        {"family": "repairability_shaped", "params": {"base_dim": 16, "base_k": 16, "corr_rank": 4,
         "sparse_rows": 8}, "priority_rank": 6, "note": "cheap base + billed low-rank + sparse Doctor"},
    ]

    rows = []
    ridx = 0

    def add_trials(rate, rate_label):
        nonlocal ridx
        for cand in candidates:
            for cls, shp, tname_fmt in (("expert_mlp1", MLP1, "block.{b}.mlp.mlp1_weight"),
                                        ("expert_mlp2", MLP2, "block.{b}.mlp.mlp2_weight")):
                nW = shp[0] * shp[1]
                isl = Fraction(1, 100) if "islands" in cand["family"] else Fraction(0)
                doc = Fraction(3, 20) if "doctor" in cand["family"] else Fraction(0)
                budget = _target_bits(nW, rate, isl, doc)
                rows.append({
                    "row_id": f"t{ridx:04d}",
                    "trial": True,
                    "source_revision": PARENT,
                    "tensor_class": cls,
                    "sample_layers": SAMPLE_LAYERS,
                    "tensor_group_fmt": tname_fmt,
                    "representation_family": cand["family"],
                    "family_params": cand["params"],
                    "sharing_group": f"{cls}_{cand['family']}",
                    "protected_island_strategy": cand["params"].get("strategy"),
                    "exact_rate": rate_label,
                    "doctor_reserve_bpw": float(doc),
                    "cpu_impl": "gravity_forge numpy",
                    "metal_impl": "gravity_forge MPS (_kmeans/_assign) / streaming",
                    "exact_budget": {"n_weights_per_matrix": nW, "target_total_bits": budget,
                                     "rate": rate_label},
                    "functional_metric": "output_divergence (reference MoE forward, routed experts)",
                    "calibration_metrics": ["weight_relative_error", "functional_output_divergence"],
                    "holdout_metrics": ["holdout_functional_output_divergence"],
                    "resource_estimate": {"gpu": "mps", "peak_gib": 1.5},
                    "dependencies": [],
                    "priority_rank": cand["priority_rank"],
                    "stopping_rule": "promote the lowest functional divergence within budget; escalate "
                                     "rate only after all geometries at this rate are tried",
                    "evidence_schema": "hawking.gravity_frontier.trial_evidence.v1",
                    "note": cand["note"],
                })
                ridx += 1

    add_trials(R34, "3/4")     # geometry search at the sub-bit stress rate
    add_trials(R1, "1/1")      # escalation rate (only reached if 3/4 geometry exhausted + failing)

    doc = {
        "schema": SCHEMA,
        "parent_revision": PARENT,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scientific_question": "Which full-rank geometry preserves function below one bit?",
        "search_doctrine": {
            "geometry_before_rate": True,
            "priority_order": ["representation", "sharing", "subvector_geometry",
                               "protected_islands", "doctor", "residual_additive", "raise_rate"],
            "ranking_signal": "functional_output_divergence (real reference forward)",
            "baseline_to_beat": 0.688,
        },
        "sample_layers": SAMPLE_LAYERS,
        "rows": rows,
        "totals": {
            "total_trial_rows": len(rows),
            "geometries": len(candidates),
            "rates": ["3/4", "1/1"],
            "sample_layers": len(SAMPLE_LAYERS),
            "expected_checkpoint_count": len(rows),
            "expected_wall_time_note": "each trial packs the sampled experts + runs the reference "
                                       "forward on routed experts (~seconds-minutes); bounded, durable",
        },
        "exact_budget_discipline": {"rates_are_rational": True, "packer_fails_over_budget": True,
                                    "no_post_hoc_rate_acceptance": True},
    }
    payload = json.dumps(doc, sort_keys=True).encode()
    doc["program_sha256"] = hashlib.sha256(payload).hexdigest()
    return doc


def main() -> int:
    FR.mkdir(parents=True, exist_ok=True)
    doc = build()
    (FR / "GPT_OSS_120B_GRAVITY_FRONTIER_PROGRAM.json").write_text(json.dumps(doc, indent=2, sort_keys=True))
    t = doc["totals"]
    print(json.dumps({"program_sha256": doc["program_sha256"][:16],
                      "total_trial_rows": t["total_trial_rows"], "geometries": t["geometries"],
                      "rates": t["rates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
