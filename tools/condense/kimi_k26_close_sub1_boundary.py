#!/usr/bin/env python3.12
"""Seal and publish the empirical Kimi K2.6 sub-1-BPW representation law."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import kimi_k26_f1_bracket as f1  # noqa: E402


def verify_seal(value: dict[str, Any], label: str) -> None:
    expected = f1.seal({key: item for key, item in value.items() if key != "seal_sha256"})[
        "seal_sha256"
    ]
    if value.get("seal_sha256") != expected:
        raise f1.F1Error(f"evidence seal mismatch: {label}")


def read_evidence(path: Path) -> dict[str, Any]:
    value = f1.read_json(path)
    verify_seal(value, path.name)
    if value.get("status") != "PASS":
        raise f1.F1Error(f"non-PASS evidence cannot close boundary: {path}")
    return value


def run(runtime: Path, publish: Path) -> dict[str, Any]:
    f1_root = runtime / "f1_representation_bracket"
    bracket = read_evidence(f1_root / "KIMI_K26_F1_REPRESENTATION_BRACKET.json")
    doctor = read_evidence(f1_root / "doctor_auction/KIMI_K26_F1_DOCTOR_AUCTION.json")
    propagation = read_evidence(runtime / "f2_propagation/KIMI_K26_P1_F2_SPARSE_PROPAGATION.json")
    alternatives = read_evidence(
        runtime / "f1_grammar_islands/KIMI_K26_P1_F1_GRAMMAR_ISLANDS_BRACKET.json",
    )
    functional = read_evidence(
        runtime / "f1_functional_codebooks/KIMI_K26_P1_F1_FUNCTIONAL_CODEBOOKS.json",
    )
    promoted = read_evidence(
        f1_root / "doctor_auction/P1_DUAL_PATH_RECOVERY_R16X2_RESULT.json",
    )
    if propagation["promoted_payload_sha256"] != promoted["payload"]["sha256"]:
        raise f1.F1Error("F2 propagation does not bind the promoted physical F1 payload")

    law_text = (
        "Under the tested Kimi K2.6 layer-1 routed-expert constraints, no complete physical "
        "representation at or below 0.98 BPW is promotable beyond F1. Independent residual-weight "
        "PQ, salience-protected islands, shared additive grammar, and fixed-index functional "
        "codebook training fail routed expert-output fidelity. Sample-limited functional "
        "hidden-state recovery can pass local F1 at 0.908591 BPW, but its residual perturbation "
        "changes the next exact layer's native top-8 route set on 21.875% of held-out tokens, "
        "preventing F2 promotion."
    )
    evidence_chain = [
        {
            "mechanism": "INDEPENDENT_PQ_PLUS_WEIGHT_RESIDUAL_DOCTOR",
            "actual_bpw": bracket["candidate_results"]["P1"]["actual_bpw"],
            "verdict": bracket["candidate_results"]["P1"]["verdict"],
            "score_cosine": bracket["current_best_capability"]["cosine_mean"],
            "reason": "Gate/up errors compound through multiplicative hidden state and down projection.",
            "seal_sha256": bracket["seal_sha256"],
        },
        {
            "mechanism": "FUNCTIONAL_HIDDEN_STATE_RECOVERY",
            "actual_bpw": promoted["physical_budget"]["actual_complete_bpw"],
            "f1_verdict": promoted["candidate_verdict"],
            "f1_score_cosine": promoted["metrics"]["score_doctored"]["cosine_mean"],
            "f2_verdict": propagation["candidate_verdict"],
            "downstream_route_set_agreement": propagation["layer_two"][
                "native_route_set_agreement"
            ],
            "reason": "Local repair crosses F1 but is not causally stable at the next native router.",
            "seal_sha256": propagation["seal_sha256"],
        },
        {
            "mechanism": "PROTECTED_ISLANDS",
            **alternatives["family_results"]["PROTECTED_ISLANDS"],
            "reason": "Salience concentration cannot restore enough expert-output direction.",
        },
        {
            "mechanism": "SHARED_ADDITIVE_GRAMMAR",
            **alternatives["family_results"]["SHARED_ADDITIVE_GRAMMAR"],
            "reason": "Cross-expert codebook amortization saves bits but loses local expert structure.",
        },
        {
            "mechanism": "FUNCTIONALLY_TUNED_FIXED_INDEX_CODEBOOKS",
            "actual_bpw": functional["physical_budget"]["actual_complete_bpw"],
            "verdict": functional["candidate_verdict"],
            "fit_score_cosine": functional["metrics"]["fit_expert_output"]["cosine_mean"],
            "holdout_score_cosine": functional["metrics"]["score_expert_output"]["cosine_mean"],
            "reason": "Functional optimization improves fit but does not generalize across disjoint tokens.",
            "seal_sha256": functional["seal_sha256"],
        },
    ]
    law = f1.seal({
        "schema": "hawking.kimi_k26.scientific_law.sub1_bpw.v1", "status": "PASS",
        "sealed_at": f1.now(), "outcome": "EMPIRICAL_BOUNDARY_ESTABLISHED",
        "source": {"repo": f1.REPO, "revision": f1.REVISION},
        "law": law_text,
        "scope": {
            "rate_ceiling_complete_physical_bpw": 0.98,
            "model_component": "TEXT_CORE_LAYER_1_ROUTED_EXPERTS",
            "fit_tokens": 32, "held_out_score_tokens": 32,
            "downstream_propagation": "ONE_COMPLETE_EXACT_NATIVE_LAYER",
            "families_tested": [
                "independent_product_quantization", "weight_residual_doctor",
                "protected_islands", "shared_additive_grammar",
                "functional_fixed_index_codebooks", "functional_hidden_state_recovery",
            ],
            "claim_exclusions": [
                "not a proof above 0.98 BPW", "not a multimodal claim",
                "not a full-model impossibility theorem", "not a deployment artifact",
            ],
        },
        "causal_explanation": {
            "local": (
                "Independent gate/up cosine near 0.925 falls to 0.886 after multiplicative SiLU "
                "interaction and 0.838 after down projection."
            ),
            "residual": (
                "The first residual add masks much of the norm error but does not guarantee causal "
                "equivalence at the next discontinuous top-k router."
            ),
            "downstream": (
                "The F1-surviving hidden Doctor retains 0.99498 mean hidden cosine after the next "
                "exact layer, yet exact route-set agreement is only 0.78125."
            ),
            "generalization": (
                "Direct codebook functional fitting reaches 0.96144 fit cosine but only 0.86804 "
                "on disjoint score tokens."
            ),
        },
        "evidence_chain": evidence_chain,
        "scientific_decision": "CLOSE_TESTED_SUB_1_BPW_REGION",
        "reopen_condition": (
            "A materially nonlocal router-co-designed representation, a larger disjoint Doctor "
            "training capture, or a complete physical rate above 0.98 BPW."
        ),
    })
    status = f1.seal({
        "schema": "hawking.kimi_k26.scientific_status.v2", "status": "BOUNDARY_ESTABLISHED",
        "updated_at": f1.now(), "source": {"repo": f1.REPO, "revision": f1.REVISION},
        "current_best_candidate": "NONE_PROMOTABLE_AT_OR_BELOW_0.98_BPW",
        "current_best_bpw": None,
        "current_best_capability": {"evidence_level": "NO_F2_SURVIVOR"},
        "best_local_candidate": {
            "candidate": "P1_DUAL_PATH_RECOVERY_R16X2",
            "actual_complete_bpw": promoted["physical_budget"]["actual_complete_bpw"],
            "f1_score": promoted["metrics"]["score_doctored"],
            "terminal_verdict": propagation["candidate_verdict"],
        },
        "current_doctor_allocation": {
            "architecture": "DUAL_PATH_RECOVERY_R16X2",
            "bytes": promoted["doctor"]["doctor_component_bytes"],
            "status": "RETIRED_AFTER_F2_ROUTER_INSTABILITY",
        },
        "current_failure_mode": "ROUTE_SET_INSTABILITY_AFTER_RESIDUAL_PROPAGATION",
        "current_dominant_bottleneck": "DISCONTINUOUS_NATIVE_TOPK_ROUTER_SENSITIVITY",
        "current_scientific_hypothesis": law_text,
        "current_next_experiment": "NONE_WITHIN_CLOSED_TESTED_CONSTRAINTS",
        "reopen_condition": law["reopen_condition"],
        "scientific_law_seal_sha256": law["seal_sha256"],
    })
    artifacts = {
        "KIMI_K26_F1_REPRESENTATION_BRACKET.json": bracket,
        "KIMI_K26_F1_DOCTOR_AUCTION.json": doctor,
        "KIMI_K26_P1_F2_SPARSE_PROPAGATION.json": propagation,
        "KIMI_K26_P1_F1_GRAMMAR_ISLANDS_BRACKET.json": alternatives,
        "KIMI_K26_P1_F1_FUNCTIONAL_CODEBOOKS.json": functional,
        "KIMI_K26_SCIENTIFIC_LAW.json": law,
        "KIMI_K26_SCIENTIFIC_STATUS.json": status,
    }
    for name, value in artifacts.items():
        f1.atomic_json(publish / name, value)
    for name in ("KIMI_K26_SCIENTIFIC_LAW.json", "KIMI_K26_SCIENTIFIC_STATUS.json"):
        f1.atomic_json(runtime / name, artifacts[name])
    manifest = f1.seal({
        "schema": "hawking.kimi_k26.science_publication_manifest.v1", "status": "PASS",
        "sealed_at": f1.now(),
        "artifacts": {name: {
            "bytes": (publish / name).stat().st_size,
            "sha256": hashlib.sha256((publish / name).read_bytes()).hexdigest(),
            "content_seal_sha256": value["seal_sha256"],
        } for name, value in artifacts.items()},
    })
    f1.atomic_json(publish / "KIMI_K26_SCIENCE_PUBLICATION_MANIFEST.json", manifest)
    return law


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--publish", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.runtime.resolve(strict=True), args.publish.resolve(strict=True))
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
