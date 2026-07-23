#!/usr/bin/env python3.12
"""Seal the frozen pre-campaign Gravity maturity audit for GLM-5.2.

The scores describe the evidence available at the campaign-entry boundary, not
the state at report-generation time.  This makes the later PRE/POST comparison
resistant to retrospective score inflation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from glm52_common import REPO_ROOT, atomic_json, atomic_text, seal, sha256_file, utc_now


ENTRY_COMMIT = "8d634af9702aa965728f057661b5d1fad1883f45"
ENTRY_AT = "2026-07-21T20:29:51.121903Z"

AXES = (
    "source_authority",
    "source_precision",
    "logical_weight_accounting",
    "physical_artifact_accounting",
    "adapter_fidelity",
    "teacher_forward_fidelity",
    "streaming_completeness",
    "resume_recovery",
    "data_integrity",
    "causal_diagnosis",
    "doctor_breadth",
    "native_studentization",
    "rate_exploration",
    "full_model_artifact",
    "capability_evaluation",
    "direct_runtime",
    "metal_execution",
    "speed_efficiency",
    "resource_utilization",
    "scientific_transfer",
    "reproducibility",
)

SCORES: dict[str, tuple[int, ...]] = {
    "GPT_OSS_120B": (5, 3, 5, 5, 4, 4, 4, 4, 3, 4, 4, 0, 4, 4, 5, 4, 2, 4, 4, 5, 4),
    "QWEN3_235B": (5, 5, 5, 4, 4, 4, 4, 3, 4, 5, 4, 0, 5, 4, 5, 4, 2, 4, 4, 5, 4),
    "KIMI_K26": (5, 3, 4, 2, 4, 4, 4, 4, 3, 3, 3, 2, 3, 1, 3, 3, 3, 4, 4, 4, 3),
    "GLM52_PRE": (1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 2, 1),
}

EVIDENCE: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "GPT_OSS_120B": [
        (
            "reports/condense/gravity_forge/condensation/GPT_OSS_120B_SOURCE_RECEIPT.json",
            ("source_authority", "immutable_revision", "tensor_formats", "tensor_count", "index_consistent"),
        ),
        (
            "reports/condense/second_light/GPT_OSS_120B_SECOND_LIGHT_BASELINE.json",
            ("result.logical_weight_count", "result.complete_physical_bits", "result.realized_whole_artifact_bpw"),
        ),
        (
            "reports/condense/general_frontier/GPT_OSS_120B_G4_RESULT.json",
            ("forward", "rows", "packed_control_rvq_1bpw", "verdict"),
        ),
        (
            "reports/condense/second_light/evidence/CRASH_RESUME_PROOF.json",
            ("status", "resume"),
        ),
        (
            "reports/condense/second_light/evidence/PQ_CPU_METAL_PARITY.json",
            ("status", "parity"),
        ),
    ],
    "QWEN3_235B": [
        (
            "reports/condense/general_frontier/QWEN3_235B_SOURCE_ADMISSION.json",
            ("immutable_revision", "config_summary.torch_dtype", "n_weight_shards", "weight_bytes"),
        ),
        (
            "reports/condense/storage_stripdown/MODEL_RELEASE_qwen3-235b-a22b.json",
            ("payload_count", "payload_bytes", "revision", "scientific_conclusion"),
        ),
        (
            "reports/condense/storage_stripdown/qwen_final_evidence/QWEN_GRAVITY_STATE.json",
            ("status", "final", "ladder", "capability_passes", "promote_thresholds"),
        ),
        (
            "reports/condense/storage_stripdown/qwen_final_evidence/QWEN235B_VULTURE_HARVEST.json",
            ("parent.original_weight_count", "A_bpw_potency", "B_decision_speed", "honesty"),
        ),
        (
            "reports/condense/storage_stripdown/qwen_final_evidence/parent_logits/MANIFEST.json",
            ("status", "files"),
        ),
    ],
    "KIMI_K26": [
        (
            "KIMI_K26_SOURCE_VERIFICATION.json",
            ("weight_shards", "source_bytes", "index_tensor_count", "failures"),
        ),
        (
            "KIMI_K26_LOGICAL_WEIGHT_LEDGER.json",
            ("all_logical_original_weights", "tensor_count", "denominator_rule"),
        ),
        (
            "KIMI_K26_ADAPTER_TWIN.json",
            ("checks", "metal_k1", "source_parent_parity_claimed", "binding.quantization"),
        ),
        (
            "KIMI_K26_REFERENCE_FORWARD.json",
            ("all_61_text_layers_in_official_order", "real_logits", "deterministic_replay"),
        ),
        (
            "KIMI_K26_GRAVITY_FINAL.json",
            ("best_deployable_candidate", "fidelity", "causal_diagnosis", "doctor_versus_native"),
        ),
        (
            "KIMI_K26_CORPUS_INTEGRITY.json",
            ("corpus",),
        ),
    ],
    "GLM52_PRE": [
        (
            "GLM52_HANDOFF_PRECHECK.json",
            ("status", "admission_decision", "rollback_exception"),
        ),
        (
            "KIMI_K26_SOURCE_RELEASE_FOR_GLM52.json",
            ("status", "source_release", "recovered_bytes"),
        ),
    ],
}


def _evidence_rows() -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    missing: list[str] = []
    for model, entries in EVIDENCE.items():
        rows[model] = []
        for relative, fields in entries:
            path = REPO_ROOT / relative
            if not path.is_file():
                missing.append(relative)
                continue
            rows[model].append({
                "path": relative,
                "sha256": sha256_file(path),
                "fields": list(fields),
                "bytes": path.stat().st_size,
            })
    if missing:
        raise FileNotFoundError("maturity evidence missing: " + ", ".join(missing))
    return rows


def build() -> dict[str, Any]:
    score_rows: dict[str, Any] = {}
    for model, values in SCORES.items():
        if len(values) != len(AXES) or any(value < 0 or value > 5 for value in values):
            raise ValueError(f"invalid score vector for {model}")
        total = sum(values)
        score_rows[model] = {
            "axes": dict(zip(AXES, values, strict=True)),
            "total": total,
            "maximum": len(AXES) * 5,
            "mean": total / len(AXES),
        }
    return seal({
        "schema": "hawking.gravity_completeness_audit.glm52_pre.v1",
        "status": "PASS_FROZEN_PRE_CAMPAIGN_BASELINE",
        "generated_at": utc_now(),
        "snapshot": {
            "at_utc": ENTRY_AT,
            "repository_head": ENTRY_COMMIT,
            "branch": "campaign/glm52-bf16-xet-gravity",
            "boundary": "Immediately after verified Kimi handoff/release and before GLM source admission, header audit, adapter, or runtime work.",
            "later_glm_artifacts_excluded_from_scores": True,
        },
        "scoring": {
            "range": [0, 5],
            "axis_count": len(AXES),
            "maximum_total": len(AXES) * 5,
            "score_5": "Reproducible capability-fidelity evidence on the axis; a reproducible negative can earn 5 for evaluation maturity.",
            "score_5_does_not_mean_capability_pass": True,
            "plans_and_unexecuted_code_maximum": 1,
        },
        "axes": list(AXES),
        "scores": score_rows,
        "evidence": _evidence_rows(),
        "honest_status": {
            "operationally_proven_before_glm": [
                "giant official-source admission and traversal on prior parents",
                "exact complete-BPW accounting",
                "bounded source readers and durable controllers",
                "crash/resume or checkpoint durability",
                "Apple-local resource control",
                "component-level Metal execution",
            ],
            "scientifically_proven_before_glm": [
                "reproducible negative full-model capability evidence on GPT-OSS and Qwen",
                "cross-parent causal negative science",
                "Doctor and representation families can be compared without free bytes",
                "Kimi upstream-state error is primary and routing is a secondary conditional amplifier",
            ],
            "not_proven_at_entry": [
                "capability-preserving complete model below 1 BPW",
                "capability-preserving complete model at 0.50 BPW",
                "universal cross-parent compression law",
                "native functional studentization beyond Kimi one-block contextual F1",
                "direct high-performance native runtime for every giant architecture",
                "any GLM-5.2 scientific result",
            ],
        },
        "model_boundaries": {
            "GPT_OSS_120B": "Official mixed MXFP4-expert/BF16-control parent; complete 0.76976-BPW physical baseline and real full-forward negative, not a BF16-teacher success.",
            "QWEN3_235B": "Official BF16 parent and real 94-layer parent-versus-packed negative; Python reference runtime and no retained standalone native complete payload.",
            "KIMI_K26": "Official packed-INT4 parent; strongest retained evidence is local F1/F2, with no complete compact capability artifact.",
            "GLM52_PRE": "Handoff and planning only at the frozen boundary; no teacher forward, corpus, artifact, Metal execution, or capability result.",
        },
    })


def markdown(audit: dict[str, Any]) -> str:
    labels = list(SCORES)
    lines = [
        "# Gravity completeness audit — GLM-5.2 pre-campaign",
        "",
        f"Frozen boundary: `{audit['snapshot']['at_utc']}` at `{ENTRY_COMMIT}`.",
        "Later GLM work is deliberately excluded, so the post-campaign delta cannot rewrite the baseline.",
        "",
        "A 5 denotes reproducible evaluation maturity on an axis; it does not imply a capability pass.",
        "",
        "| Axis | " + " | ".join(labels) + " |",
        "|---|" + "---:|" * len(labels),
    ]
    for axis in AXES:
        lines.append(
            "| " + axis.replace("_", " ") + " | "
            + " | ".join(str(audit["scores"][label]["axes"][axis]) for label in labels)
            + " |"
        )
    lines.append(
        "| **Total / 105** | "
        + " | ".join(f"**{audit['scores'][label]['total']}**" for label in labels)
        + " |"
    )
    lines.extend(["", "## Honest boundary", ""])
    for label in labels:
        lines.append(f"- **{label}:** {audit['model_boundaries'][label]}")
    lines.extend(["", f"Seal: `{audit['seal_sha256']}`.", ""])
    return "\n".join(lines)


def main() -> int:
    audit = build()
    atomic_json(REPO_ROOT / "GRAVITY_COMPLETENESS_AUDIT_GLM52_PRE.json", audit)
    atomic_text(REPO_ROOT / "GRAVITY_COMPLETENESS_AUDIT_GLM52_PRE.md", markdown(audit))
    print(json.dumps({
        "status": audit["status"],
        "totals": {model: row["total"] for model, row in audit["scores"].items()},
        "seal_sha256": audit["seal_sha256"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
