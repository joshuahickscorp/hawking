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
    REQUIRES_TRAINED_MODULES,
    REQUIRES_TRAINING_LOOP,
    gate_record,
    inventory_built_vs_gated,
)
from lab.receipts import seal


PROMOTION_GATE_SCHEMA = "hawking.frankenstein.functional_transfer_promotion_gate.v1"
V0_PROMOTION_GATE_SCHEMA = "hawking.frankenstein.v0_promotion_gate.v1"

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

REJECT_CONTAMINATION_OR_MEMORIZATION = (
    "REJECT gains from train/eval contamination or teacher-answer memorization. "
    "Hidden-test membership must stay disjoint; exact teacher answer replay is not transfer."
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
        "reject_contamination_or_memorization": REJECT_CONTAMINATION_OR_MEMORIZATION,
        "require_exact_provenance": True,
        "require_gravity_accounting": True,
        "require_stable_routing": True,
        "require_held_out_math_gains": True,
        "require_hidden_eval": True,
        "require_trained_modules": True,
        "require_retention_gate": True,
        "require_reversible_loadable": True,
        "require_kimi_bridge_intact": True,
        "require_complete_beats_linear_init": True,
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
    retention_verdict: str | None = None,
    trained_modules: Mapping[str, Any] | bool | None = None,
    contamination: Mapping[str, Any] | None = None,
    teacher_memorization: Mapping[str, Any] | None = None,
    complete_beats_linear: bool | None = None,
    reversible_loadable: bool | None = None,
    kimi_bridge_intact: bool | None = None,
) -> dict[str, Any]:
    """Run the promotion gate.

    Without real sealed scores / corpus, returns PENDING (not ACCEPT, not a fake REJECT
    on missing data — missing evidence is PENDING; explicit fail scores are REJECT).

    V0 extensions (optional kwargs) wire retention, trained-module presence,
    contamination / teacher-memorization rejects, complete-beats-linear, reversible
    loadability, and Kimi-bridge integrity without duplicating the latent_v0 path.
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

    # --- V0 extensions (optional; PENDING when omitted) ---
    if retention_verdict is None:
        _check("retention_gate", "PENDING", "retention / secondary non-regression not submitted")
    elif str(retention_verdict).upper() in {"PASS", "ACCEPT"}:
        _check("retention_gate", "PASS", retention_verdict)
    else:
        _check("retention_gate", "FAIL", f"retention_verdict={retention_verdict}")

    if trained_modules is None:
        _check(
            "trained_modules",
            "PENDING",
            f"{REQUIRES_TRAINED_MODULES}: no trained-module admission submitted",
        )
    elif trained_modules is True or (
        isinstance(trained_modules, Mapping) and trained_modules.get("trained") is True
        and trained_modules.get("complete") is True
    ):
        _check("trained_modules", "PASS", "trained modules admitted")
    elif isinstance(trained_modules, Mapping) and trained_modules.get("trained") is True:
        _check(
            "trained_modules",
            "FAIL",
            "trained=true but module pack incomplete (missing required V0 sites)",
        )
    else:
        _check(
            "trained_modules",
            "FAIL",
            f"{REQUIRES_TRAINED_MODULES}: refuse empty/untrained/scaffold packs",
        )

    if contamination is None:
        _check("contamination_barrier", "PENDING", "contamination audit not submitted")
    else:
        clean = contamination.get("pass") is True or contamination.get("clean") is True
        if contamination.get("contamination_detected") is True:
            clean = False
        _check(
            "contamination_barrier",
            "PASS" if clean else "FAIL",
            str(contamination),
        )

    if teacher_memorization is None:
        _check(
            "teacher_memorization",
            "PENDING",
            "teacher-answer memorization audit not submitted",
        )
    else:
        mem = bool(teacher_memorization.get("memorization_detected"))
        ok = teacher_memorization.get("pass") is True or (
            not mem and teacher_memorization.get("pass") is not False
        )
        if mem:
            ok = False
        _check(
            "teacher_memorization",
            "PASS" if ok else "FAIL",
            str(teacher_memorization),
        )

    if complete_beats_linear is None:
        _check(
            "complete_beats_linear_init",
            "PENDING",
            "complete V0 vs linear-init comparison not submitted",
        )
    else:
        _check(
            "complete_beats_linear_init",
            "PASS" if complete_beats_linear else "FAIL",
            f"complete_beats_linear={complete_beats_linear}",
        )

    if reversible_loadable is None:
        _check("reversible_loadable", "PENDING", "loadable/reversible proof not submitted")
    else:
        _check(
            "reversible_loadable",
            "PASS" if reversible_loadable else "FAIL",
            f"reversible_loadable={reversible_loadable}",
        )

    if kimi_bridge_intact is None:
        _check("kimi_bridge_intact", "PENDING", "KIMI_STRATEGIC_BRIDGE integrity not submitted")
    else:
        _check(
            "kimi_bridge_intact",
            "PASS" if kimi_bridge_intact else "FAIL",
            f"kimi_bridge_intact={kimi_bridge_intact}",
        )

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
            f"{REQUIRES_DSV4F_FORWARD}, {REQUIRES_TRAINED_MODULES}). "
            "Never fabricates ACCEPT."
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
            REQUIRES_TRAINED_MODULES: gate_record(REQUIRES_TRAINED_MODULES),
        },
        "reject_rule": REJECT_IMITATION_WITHOUT_PROOF,
        "reject_contamination_or_memorization": REJECT_CONTAMINATION_OR_MEMORIZATION,
        "fabricated_scores": False,
        "claim_boundary": {
            "proto_frankenstein_complete": overall == "ACCEPT",
            "pending_is_not_accept": True,
            "linear_init_cannot_promote": True,
            "empty_untrained_modules_cannot_seal": True,
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
