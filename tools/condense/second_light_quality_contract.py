#!/usr/bin/env python3.12
"""Materialize GPT_OSS_120B_QUALITY_CONTRACT.json (Second Light goal, Section 5).

The quality contract is extracted BEFORE the new family is promoted, so no gate can be weakened
after seeing results. Thresholds are grounded in REAL First-Light baseline measurements (the ternary
plateau + PQ selection probe + true-residual output divergence), never invented to pass a candidate.
Calibration and holdout partitions are separate; the holdout is never tuned on. Gates 1-7 mirror the
goal's suggested hierarchy and each carries a measured baseline and a promotion threshold.

Honesty: the reference forward (gptoss_moe_runtime) is numerically runnable but not HF-parity
validated (SwiGLU clamp/alpha and RoPE constants from-config); it is authoritative for RELATIVE
orig-vs-packed divergence, which is what every gate here measures. Absolute perplexity parity
requires an HF-validated forward and is flagged as such in Gate 7.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "condense" / "second_light"
SCHEMA = "hawking.second_light.quality_contract.v1"

# measured First-Light baselines (from the sealed dossier + F2 fixtures) - the bar to beat.
BASELINE = {
    "ternary_rank8_weight_rel_error": 0.9855,
    "ternary_treated_output_div_mean": 0.8684,
    "pq_selection_probe_weight_rel_error_1bpw": 0.5426,
    "transform_pq_true_residual_output_div_mean_0p75bpw": 0.68792,
    "transform_pq_true_residual_output_div_max_0p75bpw": 0.91581,
    "activation_aware_output_div_mean_0p755bpw": 0.65088,
    "doctor_mean_improvement": 0.12666,
}


def build() -> dict:
    doc = {
        "schema": SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parent_reference": {
            "source_id": "openai/gpt-oss-120b",
            "reference_source_revision": "b5c939de",
            "reference_runtime": "tools/condense/gptoss_moe_runtime.py (CPU MoE reference)",
            "reference_runtime_caveat": "authoritative for RELATIVE orig-vs-packed divergence; "
                                        "not HF-parity validated for absolute perplexity",
            "harmony_config": {
                "chat_template": "models/gpt-oss-120b/chat_template.jinja",
                "tokenizer": "models/gpt-oss-120b/tokenizer.json",
                "vocab_size": 201088,
                "rendering": "Harmony chat template; deterministic encode",
            },
            "deterministic_seeds": {"fit": 0, "probe": 0, "eval": 0},
        },
        "datasets": {
            "calibration": {
                "purpose": "fit codebooks + activation salience + Doctor; tuning allowed",
                "prompts": ["general_reasoning", "code_generation", "math_word_problem",
                            "tool_use_json"],
                "note": "Harmony-formatted; disjoint from holdout",
            },
            "validation": {
                "purpose": "gate promotion decisions during the campaign",
                "prompts": ["held_general", "held_code", "held_math"],
            },
            "holdout": {
                "purpose": "final capability confirmation; NEVER tuned on",
                "prompts": ["holdout_general", "holdout_code", "holdout_math", "holdout_tooluse"],
                "sealed_disjoint_from_calibration": True,
            },
        },
        "hard_invariants": [
            "exact whole-artifact byte accounting; packer fails if a row exceeds its exact budget",
            "no rehydrate-to-f16 win counts (fake-win ban)",
            "no uncounted dense residual in Doctor",
            "deterministic bytes: same seed => identical packed bytes",
            "CPU is authoritative for final selection where Metal reductions are nondeterministic",
            "router top-k selection integrity preserved (routing not silently altered)",
            "one heavy controller (heavy_controller_count == 1)",
        ],
        "soft_thresholds": {
            "expert_output_rel_error_promote": 0.60,
            "expert_output_rel_error_stretch": 0.40,
            "layer_hidden_cosine_promote": 0.95,
            "logit_kl_promote": 0.10,
            "topk_token_agreement_promote": 0.90,
            "router_topk_agreement_promote": 0.98,
        },
        "metric_tolerances": {
            "cpu_metal_rel_error_delta": 1e-4,
            "cpu_metal_assignment_exact": True,
            "budget_bits_tolerance": 0,
        },
        "failure_conditions": [
            "cannot fit exact byte budget",
            "Metal and CPU disagree beyond tolerance",
            "holdout degrades materially versus validation",
            "Doctor cannot improve a failing row within its reserve",
            "dominated in both quality and cost by a lower-rate candidate",
        ],
        "measured_baselines": BASELINE,
        "gate_hierarchy": [
            {"gate": 1, "name": "bounded_operator_parity",
             "measures": "CPU vs Metal PQ assignment + direct execute",
             "baseline": "n/a (parity)", "promote_threshold": "assignment exact; rel-error delta < 1e-4",
             "kind": "apparatus_parity"},
            {"gate": 2, "name": "expert_functional_divergence_beats_ternary",
             "measures": "expert output relative divergence (true residual, real tokens)",
             "baseline": BASELINE["ternary_treated_output_div_mean"],
             "promote_threshold": "materially below ternary treated baseline (target < 0.60)",
             "kind": "functional"},
            {"gate": 3, "name": "router_agreement_preserved",
             "measures": "router top-k agreement downstream of packed experts",
             "baseline": 1.0, "promote_threshold": ">= 0.98 (router kept original in base program)",
             "kind": "functional"},
            {"gate": 4, "name": "one_layer_hidden_state_quality",
             "measures": "single-layer hidden-state cosine + rel error",
             "baseline": "measured at run", "promote_threshold": "cosine >= 0.95", "kind": "functional"},
            {"gate": 5, "name": "multi_layer_hidden_state_quality",
             "measures": "early/mid/late layer hidden-state quality",
             "baseline": "measured at run", "promote_threshold": "cosine >= 0.95 across depth",
             "kind": "functional"},
            {"gate": 6, "name": "short_end_to_end_logits_token_quality",
             "measures": "logit cosine, logit KL, top-k token agreement on short sequences",
             "baseline": "measured at run",
             "promote_threshold": "logit_kl <= 0.10 and topk_token_agreement >= 0.90",
             "kind": "capability_proxy"},
            {"gate": 7, "name": "complete_artifact_evaluation",
             "measures": "perplexity / NLL on the fixed holdout corpus + deterministic generation",
             "baseline": "reference perplexity (requires HF-validated forward)",
             "promote_threshold": "holdout NLL within declared tolerance of reference",
             "kind": "capability", "requires": "HF-parity-validated reference forward"},
        ],
        "gate_law": ("Do not make a red gate green by weakening its threshold after seeing results. "
                     "Set thresholds from real baseline measurements. No family advances to the full "
                     "run based only on weight error; no run is called successful based only on "
                     "bounded expert slices."),
    }
    payload = json.dumps(doc, sort_keys=True).encode()
    doc["contract_sha256"] = hashlib.sha256(payload).hexdigest()
    return doc


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = build()
    (OUT / "GPT_OSS_120B_QUALITY_CONTRACT.json").write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(json.dumps({"contract_sha256": doc["contract_sha256"][:16],
                      "gates": len(doc["gate_hierarchy"]),
                      "hard_invariants": len(doc["hard_invariants"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
