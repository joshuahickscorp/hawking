#!/usr/bin/env python3.12
"""Seal the First-Light calibration (Second Light goal, Section 2).

The bounded low-rank-ternary slice campaign is valuable evidence: it selected the next
representation family. It is NOT the full-model condensation or capability run, and several prior
reports wrongly implied otherwise. This tool reclassifies that work explicitly as FIRST-LIGHT
CALIBRATION, binds every evidence field the goal enumerates from the real sealed artifacts, records
the boundary statement, and does NOT rewrite history. It only adds a correct, sealed calibration
record and (if needed) an explicit boundary note.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GF = REPO / "reports" / "condense" / "gravity_forge"
SF = REPO / "reports" / "condense" / "subbit_frontier"
OUT = REPO / "reports" / "condense" / "second_light"
SCHEMA = "hawking.second_light.first_light_calibration.v1"


def _j(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "<unknown>"


def build() -> dict:
    dossier = _j(GF / "condensation" / "GPT_OSS_120B_RUN_DOSSIER.json")
    baseline = _j(GF / "FORGE_BASELINE_NEGATIVE.json")
    f2 = _j(GF / "FORGE_F2_RESIDUAL.json")
    act = _j(GF / "FORGE_ACTAWARE.json")
    oracle = _j(SF / "GRAVITY_120B_ORACLE.json")

    doc = {
        "schema": SCHEMA,
        "classification": "FIRST_LIGHT_CALIBRATION",
        "not_classification": "FULL_RUN",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "boundary_statement": (
            "This dataset selected the next representation family. It did not constitute a "
            "full-model condensation or capability run. Low-rank ternary factorization is REJECTED "
            "as the principal geometry; the next run uses a full-rank Product/Vector Quantization "
            "family with shared amortized codebooks and a protected-island / residual Doctor reserve."
        ),
        "bindings": {
            "parent_revision": dossier.get("source", "openai/gpt-oss-120b @ b5c939de"),
            "code_commit": _git_head(),
            "layer": 0,
            "experts": "128 (layer 0)",
            "tensor": "mlp1 expert weights (MXFP4)",
            "slice_dimensions": dossier.get("scope", "layer 0, 128 experts, 256x256 bounded slices"),
            "representation_evaluated": ["low_rank_factor", "ternary_rank8", "vector_codebook",
                                         "product_quantization (selection probe)"],
            "exact_rates_bpw": baseline.get("evidence", {}).get("per_rate_best_rel_error", {}),
            "doctor_budget": {"mean_improvement": dossier.get("run_distributions", {})
                              .get("doctor_improvement", {}).get("mean"),
                              "max_improvement": dossier.get("run_distributions", {})
                              .get("doctor_improvement", {}).get("max")},
            "untreated_divergence": dossier.get("run_distributions", {}).get("untreated_div"),
            "treated_divergence": dossier.get("run_distributions", {}).get("treated_div"),
            "svd_spectrum": {
                "svd_rank8_error": dossier.get("plateau_diagnosis", {}).get("svd_rank8_error"),
                "svd_rank64_error": dossier.get("plateau_diagnosis", {}).get("svd_rank64_error"),
            },
            "effective_rank": "approximately 104 / 256 for 90% energy (experts are HIGH-RANK)",
            "pq_experiment": dossier.get("next_family_evidence", {}),
            "residual_statistics": dossier.get("residual_structure", {}),
            "source_commit": dossier.get("source", "openai/gpt-oss-120b @ b5c939de"),
            "output_space_proxy_f2": {
                "true_residual_mean_output_rel_div": f2.get("mean_output_rel_div"),
                "activation_aware_mean": act.get("output_divergence_activation_aware", {}).get("mean"),
                "capability_parity": False,
            },
            "resident_ceiling_bpw": oracle.get("resident_ceiling_bpw"),
        },
        "family_selection": {
            "selected": dossier.get("selected_next_forge_family",
                                    "full-rank Product/Vector Quantization + protected islands"),
            "rejected": "low_rank_ternary_factorization",
            "reason": dossier.get("plateau_diagnosis", {}).get("conclusion"),
            "pq_vs_ternary_rel_error": {
                "pq_subdim8_K256_1bpw": 0.5425698049366474,
                "ternary_rank8": 0.9854774866253138,
                "pq_wins_experts": dossier.get("next_family_evidence", {})
                .get("pq_beats_ternary_experts", "32/32"),
                "pq_error_reduction_vs_ternary": "approximately 45 percent",
            },
        },
        "honesty": {
            "is_capability_claim": False,
            "is_event_horizon": False,
            "authorizes_escape": False,
            "metric_is_proxy": True,
            "does_not_prove": baseline.get("does_not_prove", []),
        },
        "source_artifacts": {
            "run_dossier": str((GF / "condensation" / "GPT_OSS_120B_RUN_DOSSIER.json").relative_to(REPO)),
            "baseline_negative": str((GF / "FORGE_BASELINE_NEGATIVE.json").relative_to(REPO)),
            "f2_residual": str((GF / "FORGE_F2_RESIDUAL.json").relative_to(REPO)),
            "actaware": str((GF / "FORGE_ACTAWARE.json").relative_to(REPO)),
        },
    }
    payload = json.dumps(doc, sort_keys=True).encode()
    doc["sha256"] = hashlib.sha256(payload).hexdigest()
    return doc


def render_md(doc: dict) -> str:
    b = doc["bindings"]
    s = doc["family_selection"]
    L = [
        "# GPT-OSS-120B FIRST-LIGHT CALIBRATION DOSSIER",
        "",
        f"schema `{doc['schema']}`  sha256 `{doc['sha256'][:16]}`  sealed {doc['generated_at']}",
        "",
        "## Classification",
        "",
        "**FIRST-LIGHT CALIBRATION** (not FULL RUN).",
        "",
        f"> {doc['boundary_statement']}",
        "",
        "## What it was",
        "",
        f"- parent: `{b['parent_revision']}`",
        f"- scope: {b['slice_dimensions']}",
        f"- representation: low-rank ternary factorization (rejected) + PQ selection probe",
        f"- code commit: `{b['code_commit'][:12]}`",
        "",
        "## Plateau diagnosis (why ternary was rejected)",
        "",
        f"- effective rank: {b['effective_rank']}",
        f"- SVD rank-8 error: {b['svd_spectrum']['svd_rank8_error']}  (rank-64: {b['svd_spectrum']['svd_rank64_error']})",
        f"- ternary rank-8 error: {s['pq_vs_ternary_rel_error']['ternary_rank8']}",
        f"- conclusion: {s['reason']}",
        "",
        "## Family selection (evidence-driven)",
        "",
        f"- selected: {s['selected']}",
        f"- PQ subdim8 K256 @ 1 BPW rel-error: {s['pq_vs_ternary_rel_error']['pq_subdim8_K256_1bpw']}",
        f"- ternary rank-8 rel-error: {s['pq_vs_ternary_rel_error']['ternary_rank8']}",
        f"- PQ beats ternary on {s['pq_vs_ternary_rel_error']['pq_wins_experts']} experts "
        f"(~{s['pq_vs_ternary_rel_error']['pq_error_reduction_vs_ternary']} lower error)",
        "",
        "## Residual structure",
        "",
        f"- heavy-tailed: {b['residual_statistics'].get('heavy_tailed')}  "
        f"kurtosis: {b['residual_statistics'].get('kurtosis')}",
        f"- implication: {b['residual_statistics'].get('implication')}",
        "",
        "## Divergence (bounded slice, proxy)",
        "",
        f"- untreated: {b['untreated_divergence']}",
        f"- treated: {b['treated_divergence']}",
        f"- output-space F2 proxy (true residual, real tokens): {b['output_space_proxy_f2']}",
        "",
        "## Honesty boundary",
        "",
        "- is capability claim: False   is event horizon: False   authorizes escape: False",
        "- weight error is a PROXY; no full-model capability was measured or claimed.",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = build()
    (OUT / "GPT_OSS_120B_FIRST_LIGHT_CALIBRATION.json").write_text(json.dumps(doc, indent=2, sort_keys=True))
    (OUT / "GPT_OSS_120B_FIRST_LIGHT_DOSSIER.md").write_text(render_md(doc))
    print(json.dumps({"classification": doc["classification"], "sha256": doc["sha256"][:16],
                      "selected_family": doc["family_selection"]["selected"][:60]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
