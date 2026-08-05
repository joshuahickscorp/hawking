#!/usr/bin/env python3.12
"""Shared frankenstein / functional-transfer gates and role labels.

Honest status vocabulary for the functional-transfer program.  Linear
GLM→DSV4F projection is infrastructure + initialization, not inheritance.
Never seal PROTO_FRANKENSTEIN_COMPLETE from projected weights alone.
"""
from __future__ import annotations

from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Role / seal labels (mapping output)
# ---------------------------------------------------------------------------

# The linear mapping run's ONLY admissible completion label.
LINEAR_SUBSPACE_INITIALIZATION = "LINEAR_SUBSPACE_INITIALIZATION"

# Forbidden: never declare proto complete from weight projection alone.
FORBIDDEN_PROTO_COMPLETE_LABELS: frozenset[str] = frozenset(
    {
        "PROTO_FRANKENSTEIN_COMPLETE",
        "PROTO_COMPLETE",
        "PROTO_FRANKENSTEIN_INHERITANCE_COMPLETE",
        "MATH_INHERITANCE_COMPLETE",
        "FUNCTIONAL_TRANSFER_COMPLETE",
    }
)

# Historical / structural descriptors still used by the training-free lane.
LINEAR_INIT_ROLE = LINEAR_SUBSPACE_INITIALIZATION
LINEAR_INIT_CAPABILITY_STATUS = "UNVALIDATED_WEIGHT_ONLY_DERIVED"
LINEAR_INIT_HONEST_STATUS = (
    "Linear subspace projection sealed as initialization only. "
    "Not inheritance. Not PROTO_FRANKENSTEIN. Capability unvalidated."
)

# ---------------------------------------------------------------------------
# Runtime gates (fail closed — never fabricate)
# ---------------------------------------------------------------------------

REQUIRES_GLM_RUNTIME = "REQUIRES_GLM_RUNTIME"
REQUIRES_TRAINING_LOOP = "REQUIRES_TRAINING_LOOP"
REQUIRES_VERIFIER = "REQUIRES_VERIFIER"
REQUIRES_BENCHMARK_CORPUS = "REQUIRES_BENCHMARK_CORPUS"
REQUIRES_DSV4F_FORWARD = "DEEPSEEK_FORWARD_PENDING"  # existing forward gate alias

# Stages that need each gate (owner steer numbering).
STAGE_GATES: dict[str, tuple[str, ...]] = {
    "1_baseline_freeze": (),  # partially measurable now
    "2_paired_evidence": (REQUIRES_GLM_RUNTIME, REQUIRES_BENCHMARK_CORPUS),
    "3_tokenizer_alignment": (),  # pure code; capture needs GLM
    "3_capture_side": (REQUIRES_GLM_RUNTIME,),
    "4_layer_cartography": (REQUIRES_GLM_RUNTIME,),  # synthetic OK without
    "5_nonlinear_bridges": (REQUIRES_TRAINING_LOOP,),
    "6_distill_adapters": (REQUIRES_TRAINING_LOOP,),
    "7_route_policy": (REQUIRES_TRAINING_LOOP,),
    "8_verified_expert_iteration": (REQUIRES_VERIFIER, REQUIRES_TRAINING_LOOP),
    "9_ablation_ag": (),  # harness exists; live scores need corpus/forward
    "10_promotion": (
        REQUIRES_BENCHMARK_CORPUS,
        REQUIRES_DSV4F_FORWARD,
        REQUIRES_TRAINING_LOOP,
    ),
    "11_reject_imitation": (),  # policy only
    "12_secondary_search_gains": (REQUIRES_VERIFIER, REQUIRES_BENCHMARK_CORPUS),
}

GATE_MISSING_INFRA: dict[str, str] = {
    REQUIRES_GLM_RUNTIME: (
        "No local GLM-5.2 runtime (≈1.5 TB donor not served). "
        "Cannot capture GLM trajectories, activations, or logits."
    ),
    REQUIRES_TRAINING_LOOP: (
        "No DSV4F training loop (forward-only runtime; no backward/optimizer). "
        "Cannot fit nonlinear bridges or adapters."
    ),
    REQUIRES_VERIFIER: (
        "No verifier / Lean / tool loop wired. "
        "Cannot run verified expert iteration (attempt→verify→critique→repair)."
    ),
    REQUIRES_BENCHMARK_CORPUS: (
        "L0/L1 problem corpus + disjoint memberships may be frozen under "
        "evidence/models/frankenstein/corpus/, but live held-out eval still needs "
        "student/teacher forward measurement scores (not corpus assembly alone)."
    ),
    REQUIRES_DSV4F_FORWARD: (
        "Student forward measurement gated or partial; full capability benches "
        "require a registered DeepSeek forward callable."
    ),
}


class GateClosedError(RuntimeError):
    """A required runtime capability is absent; stage fails closed."""

    def __init__(self, gate: str, *, stage: str | None = None, detail: str | None = None):
        self.gate = gate
        self.stage = stage
        self.detail = detail or GATE_MISSING_INFRA.get(gate, "infrastructure absent")
        prefix = f"[{stage}] " if stage else ""
        super().__init__(f"{prefix}{gate}: {self.detail}")


def assert_not_proto_complete(label: str | None) -> None:
    """Refuse any label that pretends linear init is PROTO complete."""

    if label is None:
        return
    token = str(label).strip().upper()
    if token in FORBIDDEN_PROTO_COMPLETE_LABELS or token.endswith("_COMPLETE") and (
        "PROTO" in token or "INHERITANCE" in token
    ):
        if token in FORBIDDEN_PROTO_COMPLETE_LABELS or (
            "PROTO" in token and "COMPLETE" in token
        ):
            raise ValueError(
                f"refusing label {label!r}: linear mapping is "
                f"{LINEAR_SUBSPACE_INITIALIZATION}, never PROTO complete"
            )


def fail_closed(
    gate: str,
    *,
    stage: str,
    operation: str,
) -> dict[str, Any]:
    """Return a structured fail-closed record (no fabricated outputs)."""

    return {
        "status": "FAIL_CLOSED",
        "gate": gate,
        "stage": stage,
        "operation": operation,
        "executed": False,
        "fabricated": False,
        "missing_infra": GATE_MISSING_INFRA.get(gate, "unknown"),
        "note": (
            f"{operation} refused: {gate}. Framework/interface exists; "
            "runtime work is not simulated."
        ),
    }


def gate_record(gate: str, *, open_: bool = False, detail: str | None = None) -> dict[str, Any]:
    return {
        "gate": gate,
        "state": "OPEN" if open_ else "CLOSED",
        "missing_infra": None if open_ else GATE_MISSING_INFRA.get(gate),
        "detail": detail,
    }


def linear_init_claim_boundary() -> dict[str, Any]:
    return {
        "role": LINEAR_SUBSPACE_INITIALIZATION,
        "is_inheritance": False,
        "is_proto_frankenstein": False,
        "proto_complete_declared": False,
        "forbidden_labels": sorted(FORBIDDEN_PROTO_COMPLETE_LABELS),
        "use_for": [
            "layer_cartography_init",
            "bridge_weight_initialization",
            "structural_apply_artifact",
        ],
        "not_sufficient_for": [
            "math_capability_claim",
            "PROTO_FRANKENSTEIN_COMPLETE",
            "promotion",
            "held_out_math_gain",
        ],
        "trained": False,
        "capability_claim": False,
    }


def inventory_built_vs_gated() -> dict[str, Any]:
    """Declarative inventory used by FUNCTIONAL_TRANSFER_PROGRAM seal."""

    return {
        "built_now": [
            "LINEAR_SUBSPACE_INITIALIZATION label + claim boundary",
            "BASE_DSV4F baseline freeze descriptor (measurable fields)",
            "paired evidence trace schema + membership manager (incl. retention)",
            "PROTO_FRANKENSTEIN_V0 L0/L1 real-problem corpus assembler",
            "tokenizer-independent span/byte/token→span/semantic-anchor aligner",
            "layer correspondence cartography (CKA/CCA/Procrustes/intervention) on synthetic",
            "GLM_DSV4F_LAYER_CORRESPONDENCE + PHASE_ALIGNMENT emitters (PENDING activations)",
            "reversible nonlinear bridge + adapter module architectures",
            "A–G ablation harness + reject rule extension",
            "frozen promotion gate (returns PENDING honestly)",
            "secondary non-regression suite framework",
            "Gravity byte/TPS accounting hooks on adapters",
        ],
        "runtime_gated": {
            REQUIRES_GLM_RUNTIME: [
                "stage2 paired GLM trajectory/activation capture",
                "stage3 capture-side of alignment",
                "stage4 real GLM×DSV4F correspondence matrix",
            ],
            REQUIRES_TRAINING_LOOP: [
                "stage5 fit nonlinear bridges",
                "stage6 distill method/decomp/formal/repair adapters + value head",
                "stage7 method-conditioned route-bias residual training",
            ],
            REQUIRES_VERIFIER: [
                "stage8 verified expert iteration loop",
            ],
            REQUIRES_BENCHMARK_CORPUS: [
                "held-out math suite evaluation",
                "promotion gate live scores",
                "disjoint membership eval over real problems",
            ],
        },
        "missing_infra": dict(GATE_MISSING_INFRA),
    }


def is_linear_init_label(value: Mapping[str, Any] | str | None) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value == LINEAR_SUBSPACE_INITIALIZATION
    role = value.get("role") or value.get("status") or value.get("inheritance_status")
    return role == LINEAR_SUBSPACE_INITIALIZATION
