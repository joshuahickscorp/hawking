#!/usr/bin/env python3.12
"""Frozen promotion gate for functional-transfer Proto-Frankenstein.

Promotion requires (owner steer §10–11):
  - held-out math gains
  - measured recovery of GLM-vs-DSV4F gap (>=70% initial target band)
  - coding/tool/agent/long-context non-regression
  - stable routing
  - exact provenance
  - Gravity byte/TPS/p99 accounting
  - independent challenge

Until benchmark corpus + trained transfer + live scores exist, this gate
returns PENDING honestly — never fabricates ACCEPT.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from lab.operators.frankenstein_gates import (
    REQUIRES_BENCHMARK_CORPUS,
    REQUIRES_DSV4F_FORWARD,
    REQUIRES_TRAINING_LOOP,
    gate_record,
    inventory_built_vs_gated,
)
from lab.receipts import seal


PROMOTION_GATE_SCHEMA = "hawking.frankenstein.functional_transfer_promotion_gate.v1"

# Frozen targets (owner-set before runs, not after).
MATH_GAP_RECOVERY_MIN = 0.70  # recover >=70% of GLM-vs-DSV4F math gap
SECONDARY_TOLERANCE = 0.02
INDEPENDENT_CHALLENGE_REQUIRED = True

SECONDARY_AXES: tuple[str, ...] = (
    "coding_and_repository_work",
    "tool_use",
    "agentic_planning",
    "long_context_reasoning",
    "repair_and_critique",
    "hcli_protocols",
    "general_knowledge_conversation",
    "runtime_tps",
    "routing_stability",
    "context_behavior",
)

REJECT_IMITATION_WITHOUT_PROOF = (
    "REJECT any checkpoint that improves imitation but fails "
    "proof/computation/repair/transfer/hidden eval"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def frozen_targets() -> dict[str, Any]:
    return {
        "math_gap_recovery_min": MATH_GAP_RECOVERY_MIN,
        "secondary_tolerance_absolute": SECONDARY_TOLERANCE,
        "secondary_axes": list(SECONDARY_AXES),
        "independent_challenge_required": INDEPENDENT_CHALLENGE_REQUIRED,
        "reject_imitation_without_proof": REJECT_IMITATION_WITHOUT_PROOF,
        "require_exact_provenance": True,
        "require_gravity_accounting": True,
        "require_stable_routing": True,
        "require_held_out_math_gains": True,
        "require_hidden_eval": True,
        "secondary_search_gains_measured_separately": True,
        "note": (
            "Targets frozen prior to evaluation.  Secondary search/retrieval/tool/"
            "formal-proof gains are measured separately from raw-model gains."
        ),
    }


def evaluate_promotion(
    *,
    scores: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    gravity_accounting: Mapping[str, Any] | None = None,
    routing: Mapping[str, Any] | None = None,
    independent_challenge: Mapping[str, Any] | None = None,
    ablation_verdict: str | None = None,
) -> dict[str, Any]:
    """Run the promotion gate.

    Without real sealed scores / corpus, returns PENDING (not ACCEPT, not a fake REJECT
    on missing data — missing evidence is PENDING; explicit fail scores are REJECT).
    """

    targets = frozen_targets()
    checks: list[dict[str, Any]] = []

    def _check(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    # Infrastructure readiness
    if scores is None:
        _check(
            "held_out_math_scores",
            "PENDING",
            f"{REQUIRES_BENCHMARK_CORPUS}: no sealed held-out score bundle provided",
        )
        _check(
            "math_gap_recovery",
            "PENDING",
            "cannot compute gap recovery without base/proto/glm reference scores",
        )
        _check(
            "secondary_non_regression",
            "PENDING",
            "no secondary score map provided",
        )
        _check(
            "hidden_eval",
            "PENDING",
            "hidden-test membership eval not provided",
        )
        _check(
            "proof_computation_repair",
            "PENDING",
            "verifier/proof outcomes not provided",
        )
    else:
        # Real score path (when later supplied) — still validate structure.
        math = scores.get("math") or {}
        gap = math.get("gap_recovery")
        if gap is None:
            _check("math_gap_recovery", "PENDING", "scores.math.gap_recovery missing")
        else:
            ok = float(gap) >= MATH_GAP_RECOVERY_MIN
            _check(
                "math_gap_recovery",
                "PASS" if ok else "FAIL",
                f"gap_recovery={gap} min={MATH_GAP_RECOVERY_MIN}",
            )
        secondary = scores.get("secondary") or {}
        regs = []
        for axis in SECONDARY_AXES:
            row = secondary.get(axis)
            if not isinstance(row, Mapping):
                regs.append(axis)
                continue
            if row.get("gate") == "FAIL":
                regs.append(axis)
        if any(
            isinstance(secondary.get(a), Mapping) and secondary[a].get("gate") == "FAIL"
            for a in SECONDARY_AXES
        ):
            _check(
                "secondary_non_regression",
                "FAIL",
                f"regressions={regs}",
            )
        elif not secondary:
            _check("secondary_non_regression", "PENDING", "empty secondary map")
        else:
            _check("secondary_non_regression", "PASS", "all provided secondaries held")

        if scores.get("hidden_eval") is None:
            _check("hidden_eval", "PENDING", "hidden_eval missing")
        else:
            he = scores["hidden_eval"]
            ok = bool(he.get("pass")) if isinstance(he, Mapping) else bool(he)
            _check("hidden_eval", "PASS" if ok else "FAIL", str(he))

        proof = scores.get("proof_computation_repair")
        if proof is None:
            _check("proof_computation_repair", "PENDING", "not provided")
        else:
            ok = bool(proof.get("pass")) if isinstance(proof, Mapping) else bool(proof)
            _check(
                "proof_computation_repair",
                "PASS" if ok else "FAIL",
                "imitation-only improvements without proof/repair fail here"
                if not ok
                else "ok",
            )

    if provenance is None or not provenance.get("complete"):
        _check(
            "exact_provenance",
            "PENDING" if provenance is None else "FAIL",
            "exact hash-bound provenance required",
        )
    else:
        _check("exact_provenance", "PASS", "provenance.complete=true")

    if gravity_accounting is None:
        _check(
            "gravity_byte_tps_p99",
            "PENDING",
            "Gravity byte/TPS/p99 accounting not bound",
        )
    else:
        needed = ("parameter_bytes", "tps", "p99")
        missing = [k for k in needed if k not in gravity_accounting]
        if missing:
            _check(
                "gravity_byte_tps_p99",
                "PENDING",
                f"missing keys {missing}",
            )
        else:
            _check("gravity_byte_tps_p99", "PASS", "accounting present")

    if routing is None:
        _check("routing_stability", "PENDING", "routing stats not provided")
    else:
        ok = bool(routing.get("stable"))
        _check("routing_stability", "PASS" if ok else "FAIL", str(routing))

    if independent_challenge is None:
        _check(
            "independent_challenge",
            "PENDING",
            "independent challenge not submitted",
        )
    else:
        ok = bool(independent_challenge.get("pass"))
        _check("independent_challenge", "PASS" if ok else "FAIL", str(independent_challenge))

    if ablation_verdict == "REJECT":
        _check("ablation_ag", "FAIL", "ablation reject rule fired")
    elif ablation_verdict is None:
        _check("ablation_ag", "PENDING", "A–G ablation not yet run with live scores")
    else:
        _check("ablation_ag", "PASS" if ablation_verdict == "ACCEPT" else "FAIL", ablation_verdict)

    statuses = {c["status"] for c in checks}
    if "FAIL" in statuses:
        overall = "REJECT"
        reason = "one or more hard promotion checks failed"
    elif statuses == {"PASS"}:
        overall = "ACCEPT"
        reason = "all frozen promotion checks passed"
    else:
        overall = "PENDING"
        reason = (
            "promotion evidence incomplete; gate returns PENDING honestly "
            f"(gates: {REQUIRES_BENCHMARK_CORPUS}, {REQUIRES_TRAINING_LOOP}, "
            f"{REQUIRES_DSV4F_FORWARD}). Never fabricates ACCEPT."
        )

    document = {
        "schema": PROMOTION_GATE_SCHEMA,
        "recorded_at": _utc_now(),
        "verdict": overall,
        "reason": reason,
        "targets": targets,
        "checks": checks,
        "infra_gates": {
            REQUIRES_BENCHMARK_CORPUS: gate_record(REQUIRES_BENCHMARK_CORPUS),
            REQUIRES_TRAINING_LOOP: gate_record(REQUIRES_TRAINING_LOOP),
            REQUIRES_DSV4F_FORWARD: gate_record(REQUIRES_DSV4F_FORWARD),
        },
        "reject_rule": REJECT_IMITATION_WITHOUT_PROOF,
        "fabricated_scores": False,
        "claim_boundary": {
            "proto_frankenstein_complete": overall == "ACCEPT",
            "pending_is_not_accept": True,
            "linear_init_cannot_promote": True,
        },
        "inventory": inventory_built_vs_gated(),
    }
    return seal(document)


def secondary_non_regression_suite_framework() -> dict[str, Any]:
    """Framework descriptor for secondary capability non-regression (no live scores)."""

    return seal(
        {
            "schema": "hawking.frankenstein.secondary_non_regression_suite.v1",
            "recorded_at": _utc_now(),
            "status": "FRAMEWORK_ONLY",
            "axes": list(SECONDARY_AXES),
            "tolerance_absolute": SECONDARY_TOLERANCE,
            "predicate": "candidate[axis] >= base[axis] - tolerance for all axes",
            "bench_scope": "REQUIRES_BENCHMARK_CORPUS",
            "fabricated_scores": False,
            "note": (
                "Suite harness structure only.  Live evaluation fails closed "
                "until a frozen corpus and forward scores exist."
            ),
        }
    )
