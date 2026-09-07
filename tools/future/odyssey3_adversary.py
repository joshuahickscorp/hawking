"""ODYSSEY III — WHERE IS HAWKING WRONG?

Generate executable attack specs against Odyssey II law records, rank them by
cost versus probability of refutation, and close the loop

    LAW -> TRANSFER HYPOTHESIS -> ADVERSARIAL TARGET -> EXPERIMENT SPEC
        -> RESULT -> LAW SCOPE UPDATE

A refutation that does not move scope DOWN is a bug in the loop. A law that
emits no attack is refused, not silently published. Everything here is
STATIC_ONLY / bench UNKNOWN: these are specs, not measurements.

    python3 tools/future/odyssey3_adversary.py --build
    python3 tools/future/odyssey3_adversary.py --selftest
    python3 tools/future/odyssey3_adversary.py --replay-attack <attack_id>
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

from tools.future._common import write_receipt, load_json, REPO

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

RECEIPT = "ODYSSEY3_ADVERSARY.json"
SCHEMA = "hawking.future.odyssey3_adversary.v1"
LAW_STORE = REPO / "receipts" / "future" / "ODYSSEY2_LAW_STORE.json"

# Exact Odyssey II field set. Extra keys on a record are ignored; missing keys refuse.
LAW_FIELDS = (
    "law_id",
    "statement",
    "source_model",
    "source_device",
    "architecture_family",
    "organ_class",
    "backend",
    "evidence_strength",
    "evidence_refs",
    "scope",
    "transfer_candidates",
    "transfer_confidence",
    "counterexample_requirement",
)

# Most general first. Rank is index; DOWN means a strictly smaller index.
SCOPE_LADDER = (
    "GENERIC_VERIFIED",
    "FAMILY_VERIFIED",
    "MODEL_LOCAL",
    "ORGAN_LOCAL",
    "DEVICE_LOCAL",
    "MACHINE_LOCAL",
    "REFUTED",
)

ATTACK_FAMILIES = (
    "negative_transfer",
    "blind_holdout",
    "measurement_trap",
    "contamination_trap",
    "goodhart",
    "compiler_prior",
    "representation_overfit",
    "law_scope",
    "causal_control",
)

# Planning cost in dimensionless units. Not a wall-clock, not a GPU number.
FAMILY_COST = {
    "measurement_trap": 1,
    "causal_control": 3,
    "blind_holdout": 4,
    "contamination_trap": 5,
    "law_scope": 6,
    "compiler_prior": 8,
    "representation_overfit": 10,
    "goodhart": 12,
    "negative_transfer": 15,
}

ATTACK_SPEC_FIELDS = (
    "attack_id",
    "family",
    "law_id",
    "command",
    "inputs",
    "expected_if_law_holds",
    "expected_if_law_false",
    "falsifier",
    "cost_units",
    "p_refutation",
    "selection_score",
    "target_scope_if_refuted",
    "transfer_hypothesis",
    "adversarial_target",
    "evidence_class",
    "bench_state",
)

REFUTING_VERDICTS = frozenset({"REFUTED", "TRAP_TRIGGERED", "HARNESS_SELF_MEASURE"})

# Historical failure shapes this generator is built to catch (recovered, not invented).
PAST_FAILURE_SHAPES = (
    "cosine_scale_invariance",   # 0.01*W cosine=1.0 while rel error ~0.99
    "skip_counted_as_pass",      # grades resting on tests that silently SKIPPED
    "receipt_reread_as_run",     # suite read a checked-in receipt instead of running the tool
    "gaussian_proxy_inversion",  # synthetic X inverted a ranking that real X reversed
    "layout_vs_kernel_reuse",    # architecture similarity ranked a kernel that storage forbids
)


class LawSchemaError(ValueError):
    """Law record does not carry the Odyssey II field set, or its scope is unusable."""


class NoAttackError(ValueError):
    """Emitter refusal: a law with no generated attack must not be published."""


class ScopeUnmovedError(ValueError):
    """A refuting result that does not move scope DOWN is a bug in the loop."""


def scope_rank(scope: str) -> int:
    if scope not in SCOPE_LADDER:
        raise LawSchemaError(f"unknown scope {scope!r}; not in {SCOPE_LADDER}")
    return SCOPE_LADDER.index(scope)


def is_downgrade(before: str, after: str) -> bool:
    return scope_rank(after) > scope_rank(before)


def one_step_down(scope: str) -> str:
    i = scope_rank(scope)
    return SCOPE_LADDER[min(i + 1, len(SCOPE_LADDER) - 1)]


def _strict_down(current: str, desired: str) -> str:
    """Named floor if it is strictly below current, else one step down. Never equal."""
    if desired not in SCOPE_LADDER:
        desired = one_step_down(current)
    if scope_rank(desired) > scope_rank(current):
        return desired
    nxt = one_step_down(current)
    if nxt == current:
        # Already REFUTED — caller should have refused the law.
        return "REFUTED"
    return nxt


def validate_law(law: Any) -> dict[str, Any]:
    if not isinstance(law, dict):
        raise LawSchemaError("law is not an object")
    missing = [f for f in LAW_FIELDS if f not in law]
    if missing:
        raise LawSchemaError(f"{law.get('law_id', '<no id>')}: missing fields {missing}")
    if law["scope"] not in SCOPE_LADDER:
        raise LawSchemaError(f"{law['law_id']}: unknown scope {law['scope']!r}")
    if law["scope"] == "REFUTED":
        raise LawSchemaError(
            f"{law['law_id']}: already REFUTED; Odyssey III does not attack a dead law"
        )
    if not isinstance(law["transfer_candidates"], list):
        raise LawSchemaError(f"{law['law_id']}: transfer_candidates must be a list")
    if not isinstance(law["evidence_refs"], list):
        raise LawSchemaError(f"{law['law_id']}: evidence_refs must be a list")
    try:
        conf = float(law["transfer_confidence"])
    except (TypeError, ValueError) as e:
        raise LawSchemaError(f"{law['law_id']}: transfer_confidence is not a number") from e
    if not 0.0 <= conf <= 1.0:
        raise LawSchemaError(f"{law['law_id']}: transfer_confidence {conf} not in [0, 1]")
    if not str(law["law_id"]).strip() or not str(law["statement"]).strip():
        raise LawSchemaError("law_id and statement must be non-empty")
    return law


def _blob(law: dict[str, Any]) -> str:
    return " ".join(
        str(law.get(k, ""))
        for k in (
            "statement",
            "organ_class",
            "backend",
            "architecture_family",
            "counterexample_requirement",
            "evidence_strength",
        )
    ).lower()


def _has_any(blob: str, words: tuple[str, ...]) -> bool:
    """Whole-token match so 'scored' does not count as 'score'."""
    for w in words:
        if re.search(r"(?<![a-z0-9])" + re.escape(w) + r"(?![a-z0-9])", blob):
            return True
    return False


def p_refutation(family: str, law: dict[str, Any]) -> float:
    """Deterministic prior that the attack will refute the law as currently scoped.

    Not a measurement. Keywords recover the shapes this project has already
    watched fail (cosine-as-gate, contaminated windows, layout-vs-kernel).
    """
    blob = _blob(law)
    scope = law["scope"]
    conf = float(law["transfer_confidence"])
    metric = _has_any(blob, ("cosine", "adequacy", "bpw", "rel_fro", "metric", "certificate"))
    timing = _has_any(blob, ("wall", "dispatch", "scheduling", "telemetry", "latency", "token time"))
    fitted = _has_any(
        blob, ("fit", "fitted", "seed", "seeded", "held-out", "holdout", "held_out", "grid", "search")
    )
    layout = _has_any(blob, ("layout", "storage", "packing", "kernel", "representation", "gemv", "codec"))
    compiler = _has_any(blob, ("compiler", "default", "flag", "metallib"))
    causal = _has_any(blob, ("because", "cause", "causal", "mechanism", "therefore", "so the"))
    overclaimed = scope in {"GENERIC_VERIFIED", "FAMILY_VERIFIED"}

    if family == "measurement_trap":
        p = 0.85 if metric else 0.12
    elif family == "goodhart":
        p = 0.80 if metric else 0.10
    elif family == "blind_holdout":
        p = 0.70 if fitted else 0.35
    elif family == "contamination_trap":
        p = 0.80 if timing else (0.55 if overclaimed else 0.25)
    elif family == "compiler_prior":
        p = 0.70 if (compiler or layout) else 0.30
    elif family == "representation_overfit":
        p = 0.75 if layout else 0.40
    elif family == "negative_transfer":
        p = 0.25 + 0.60 * conf
        if overclaimed:
            p = min(0.95, p + 0.15)
        if scope == "MODEL_LOCAL" and conf < 0.3:
            p = 0.20
    elif family == "law_scope":
        p = { "GENERIC_VERIFIED": 0.80, "FAMILY_VERIFIED": 0.55, "MODEL_LOCAL": 0.35
            }.get(scope, 0.25)
    elif family == "causal_control":
        p = 0.70 if causal else (0.50 if layout or metric else 0.35)
    else:
        raise ValueError(f"unknown family {family!r}")
    return round(min(0.95, max(0.08, p)), 3)


def _target_scope(family: str, law: dict[str, Any]) -> str:
    current = law["scope"]
    if family == "contamination_trap":
        return _strict_down(current, "MACHINE_LOCAL")
    if family == "measurement_trap" or family == "goodhart" or family == "causal_control":
        return _strict_down(current, "REFUTED")
    if family == "compiler_prior":
        return _strict_down(current, "DEVICE_LOCAL")
    if family == "representation_overfit":
        return _strict_down(current, "ORGAN_LOCAL")
    if family == "law_scope":
        return one_step_down(current)
    if family == "negative_transfer":
        if current == "GENERIC_VERIFIED":
            return "FAMILY_VERIFIED"
        if current == "FAMILY_VERIFIED":
            return "MODEL_LOCAL"
        return _strict_down(current, "REFUTED")
    if family == "blind_holdout":
        return _strict_down(current, "REFUTED" if current == "MODEL_LOCAL" else one_step_down(current))
    return one_step_down(current)


def _hostile_target(law: dict[str, Any]) -> dict[str, str]:
    """A place the transfer hypothesis is most likely to be wrong."""
    cands = [str(c) for c in law["transfer_candidates"]]
    if cands:
        # Last candidate is treated as the least like the source (deterministic).
        name = sorted(cands)[-1]
        kind = "transfer_candidate"
    else:
        name = f"NOT_{law['source_model']}"
        kind = "synthesized_out_of_model"
    return {
        "name": name,
        "kind": kind,
        "source_model": str(law["source_model"]),
        "source_family": str(law["architecture_family"]),
        "source_organ": str(law["organ_class"]),
        "source_device": str(law["source_device"]),
        "source_backend": str(law["backend"]),
    }


def _command(attack_id: str) -> list[str]:
    return [
        "python3",
        "tools/future/odyssey3_adversary.py",
        "--replay-attack",
        attack_id,
    ]


def _spec(
    law: dict[str, Any],
    family: str,
    *,
    inputs: dict[str, Any],
    expected_if_law_holds: dict[str, Any],
    expected_if_law_false: dict[str, Any],
    falsifier: str,
    transfer_hypothesis: str,
    adversarial_target: str,
) -> dict[str, Any]:
    attack_id = f"{law['law_id']}::{family}"
    cost = FAMILY_COST[family]
    p = p_refutation(family, law)
    score = round(cost / p, 6)
    target = _target_scope(family, law)
    if not is_downgrade(law["scope"], target):
        raise ScopeUnmovedError(
            f"{attack_id}: target_scope_if_refuted {target!r} is not below {law['scope']!r}"
        )
    return {
        "attack_id": attack_id,
        "family": family,
        "law_id": law["law_id"],
        "command": _command(attack_id),
        "inputs": inputs,
        "expected_if_law_holds": expected_if_law_holds,
        "expected_if_law_false": expected_if_law_false,
        "falsifier": falsifier,
        "cost_units": cost,
        "p_refutation": p,
        "selection_score": score,
        "target_scope_if_refuted": target,
        "transfer_hypothesis": transfer_hypothesis,
        "adversarial_target": adversarial_target,
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }


def _gen_negative_transfer(law: dict[str, Any]) -> dict[str, Any]:
    hostile = _hostile_target(law)
    return _spec(
        law,
        "negative_transfer",
        inputs={
            "apply_on": hostile,
            "predicted_effect": law["statement"],
            "out_of_evidenced_scope": True,
            "do_not_run_on_gpu": True,
        },
        expected_if_law_holds={
            "effect_present_on_hostile_target": True,
            "scope_survives": True,
        },
        expected_if_law_false={
            "effect_present_on_hostile_target": False,
            "scope_survives": False,
        },
        falsifier=(
            "effect_present_on_hostile_target is False: the transfer hypothesis "
            "does not survive the hostile target"
        ),
        transfer_hypothesis=(
            f"{law['law_id']} transfers from {law['source_model']} "
            f"({law['architecture_family']}/{law['organ_class']}) to {hostile['name']}"
        ),
        adversarial_target=hostile["name"],
    )


def _gen_blind_holdout(law: dict[str, Any]) -> dict[str, Any]:
    return _spec(
        law,
        "blind_holdout",
        inputs={
            "fit_fraction": 0.5,
            "holdout_fraction": 0.5,
            "split_seed": 0,
            "score_on": "holdout_only",
            "forbid_fit_on_holdout": True,
            "organ_class": law["organ_class"],
        },
        expected_if_law_holds={
            "holdout_agrees_with_fit": True,
            "underdetermined_fit_refused": True,
        },
        expected_if_law_false={
            "holdout_agrees_with_fit": False,
            "underdetermined_fit_refused": False,
        },
        falsifier=(
            "holdout_agrees_with_fit is False, or the fit was underdetermined "
            "(n_fit < organ dim) and still certified itself"
        ),
        transfer_hypothesis=(
            f"{law['law_id']} was not an in-sample tautology; it survives a withheld slice"
        ),
        adversarial_target=f"{law['organ_class']}#holdout_slice",
    )


def _gen_measurement_trap(law: dict[str, Any]) -> dict[str, Any]:
    return _spec(
        law,
        "measurement_trap",
        inputs={
            "controls": [
                "scale_invariance",
                "skip_counted_as_pass",
                "receipt_reread_as_run",
            ],
            "scale_factor": 0.01,
            "vector_dim": 256,
            "vector_seed": 0,
            "skip_suite": [
                {"name": "control_a", "status": "SKIPPED"},
                {"name": "control_b", "status": "SKIPPED"},
            ],
            "past_failure_shapes": list(PAST_FAILURE_SHAPES[:3]),
        },
        expected_if_law_holds={
            "scale_invariance_trap_fired": False,
            "skip_as_pass_trap_fired": False,
            "receipt_reread_trap_fired": False,
            "harness_self_measure": False,
        },
        expected_if_law_false={
            "harness_self_measure": True,
            "note": "a control with the mechanism absent produced an 'effect'",
        },
        falsifier=(
            "any control fires: scale-invariant cosine certifies 0.01*W, "
            "SKIPPED tests count as passes, or a checked-in receipt is treated as a run"
        ),
        transfer_hypothesis=(
            f"the metric named by {law['law_id']} measures the organ, not the harness"
        ),
        adversarial_target="harness_self_measure_controls",
    )


def _gen_contamination_trap(law: dict[str, Any]) -> dict[str, Any]:
    return _spec(
        law,
        "contamination_trap",
        inputs={
            "quiet_arm": "PROTECTED_ABSOLUTE_required_not_available",
            "dirty_arm": "DIAGNOSTIC_RELATIVE_busy_machine",
            "same_command": True,
            "sidecar_cannot_open_protected_lease": True,
            "if_only_quiet_holds": "MACHINE_LOCAL",
        },
        expected_if_law_holds={
            "effect_present_on_dirty_arm": True,
            "effect_agrees_across_arms": True,
        },
        expected_if_law_false={
            "effect_present_on_dirty_arm": False,
            "effect_agrees_across_arms": False,
            "scope_becomes": "MACHINE_LOCAL",
        },
        falsifier=(
            "the effect vanishes or reverses under a deliberately contaminated machine "
            "state: the law is MACHINE_LOCAL, not a physical law"
        ),
        transfer_hypothesis=(
            f"{law['law_id']} is a property of the organ, not of a quiet window"
        ),
        adversarial_target="contaminated_machine_state",
    )


def _gen_goodhart(law: dict[str, Any]) -> dict[str, Any]:
    return _spec(
        law,
        "goodhart",
        inputs={
            "optimize": "the_named_metric_directly",
            "hold_capability_probe": True,
            "capability_probe": "generation_coherence_or_rel_fro_not_the_metric",
            "known_shape": (
                "mixed-2p0 mean component cosine 0.907 is native-INCOHERENT; "
                "0.01*W cosine 1.0 destroys magnitude"
            ),
        },
        expected_if_law_holds={
            "metric_improved": True,
            "capability_collapsed": False,
        },
        expected_if_law_false={
            "metric_improved": True,
            "capability_collapsed": True,
        },
        falsifier=(
            "the named metric improved while the underlying capability collapsed "
            "(cosine-as-gate, BPW-as-velocity, skip-as-pass)"
        ),
        transfer_hypothesis=(
            f"optimising the metric of {law['law_id']} still preserves capability"
        ),
        adversarial_target="metric_direct_optimisation",
    )


def _gen_compiler_prior(law: dict[str, Any]) -> dict[str, Any]:
    return _spec(
        law,
        "compiler_prior",
        inputs={
            "arm_a": "law_change_with_incumbent_compiler_defaults",
            "arm_b": "no_law_change_with_the_same_default_flip",
            "isolate": "unrelated_compiler_default",
            "known_shape": (
                "Qwen3-VL stores fused 3-D gate_up transposed vs per-expert 2-D; "
                "kernel identity is a storage claim, not an architecture-config claim"
            ),
        },
        expected_if_law_holds={
            "effect_present_when_default_reverted": True,
            "effect_absent_on_default_flip_alone": True,
        },
        expected_if_law_false={
            "effect_present_when_default_reverted": False,
            "effect_absent_on_default_flip_alone": False,
        },
        falsifier=(
            "reverting the compiler default removes the effect while the 'law' change "
            "is held fixed, or flipping the default alone reproduces the effect"
        ),
        transfer_hypothesis=(
            f"the effect named by {law['law_id']} is caused by the law, not a compiler default"
        ),
        adversarial_target="unrelated_compiler_default",
    )


def _gen_representation_overfit(law: dict[str, Any]) -> dict[str, Any]:
    return _spec(
        law,
        "representation_overfit",
        inputs={
            "same_organ": law["organ_class"],
            "same_model": law["source_model"],
            "alternate_representation": "repack_or_alternate_layout_of_the_same_organ",
            "known_shape": (
                "Gaussian-proxy ranking inverted on real X; VL fused layout "
                "broke kernel identity the config-similarity ranking promised"
            ),
        },
        expected_if_law_holds={
            "effect_survives_repack": True,
        },
        expected_if_law_false={
            "effect_survives_repack": False,
        },
        falsifier=(
            "the law fails under a different representation of the same organ "
            "(layout, packing, or real vs synthetic X)"
        ),
        transfer_hypothesis=(
            f"{law['law_id']} is a property of the organ, not of one packing of it"
        ),
        adversarial_target=f"{law['organ_class']}#alternate_representation",
    )


def _gen_law_scope(law: dict[str, Any]) -> dict[str, Any]:
    nxt = one_step_down(law["scope"])
    return _spec(
        law,
        "law_scope",
        inputs={
            "current_scope": law["scope"],
            "one_step_narrower": nxt,
            "cheapest_discriminating_axis": {
                "GENERIC_VERIFIED": "a second architecture_family",
                "FAMILY_VERIFIED": "a second source_model in-family",
                "MODEL_LOCAL": "a second organ_class on the same model",
                "ORGAN_LOCAL": "a second source_device",
                "DEVICE_LOCAL": "a contaminated machine state",
                "MACHINE_LOCAL": "any independent rerun",
            }.get(law["scope"], "independent rerun"),
        },
        expected_if_law_holds={
            "still_holds_one_step_narrower": True,
            "scope_survives": True,
        },
        expected_if_law_false={
            "still_holds_one_step_narrower": False,
            "scope_becomes": nxt,
        },
        falsifier=(
            f"the cheapest experiment that would force a scope DOWNGRADE to {nxt} "
            "reports the effect absent"
        ),
        transfer_hypothesis=(
            f"{law['law_id']} at {law['scope']} still holds after a one-step scope probe"
        ),
        adversarial_target=f"scope_probe:{law['scope']}->{nxt}",
    )


def _gen_causal_control(law: dict[str, Any]) -> dict[str, Any]:
    return _spec(
        law,
        "causal_control",
        inputs={
            "arm_a": "mechanism_enabled",
            "arm_b": "same_change_mechanism_disabled",
            "require_effect_to_vanish_on_arm_b": True,
            "known_shape": (
                "shared-operator 'breakthrough' vanished under leakage/aggregation/"
                "wrong-input audit; the leaked number is not a reopen"
            ),
        },
        expected_if_law_holds={
            "effect_on_enabled": True,
            "effect_on_disabled": False,
        },
        expected_if_law_false={
            "effect_on_enabled": True,
            "effect_on_disabled": True,
        },
        falsifier=(
            "the effect persists with the claimed mechanism disabled: the law is "
            "not causal"
        ),
        transfer_hypothesis=(
            f"the mechanism named by {law['law_id']} is necessary for the effect"
        ),
        adversarial_target="mechanism_disabled_control",
    )


GENERATORS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "negative_transfer": _gen_negative_transfer,
    "blind_holdout": _gen_blind_holdout,
    "measurement_trap": _gen_measurement_trap,
    "contamination_trap": _gen_contamination_trap,
    "goodhart": _gen_goodhart,
    "compiler_prior": _gen_compiler_prior,
    "representation_overfit": _gen_representation_overfit,
    "law_scope": _gen_law_scope,
    "causal_control": _gen_causal_control,
}


def generate_attacks(law: dict[str, Any]) -> list[dict[str, Any]]:
    validate_law(law)
    attacks = [GENERATORS[family](law) for family in ATTACK_FAMILIES]
    for spec in attacks:
        missing = [f for f in ATTACK_SPEC_FIELDS if f not in spec]
        if missing:
            raise RuntimeError(f"{spec.get('attack_id')}: spec missing {missing}")
    return attacks


def rank_attacks(attacks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cheapest expected-cost-per-refutation first; ties by cost then family name."""
    return sorted(
        attacks,
        key=lambda a: (float(a["selection_score"]), int(a["cost_units"]), a["family"]),
    )


def emit_for_law(law: dict[str, Any]) -> dict[str, Any]:
    """Ranked attack plan for one law. Refuses rather than emit an empty plan."""
    validate_law(law)
    attacks = generate_attacks(law)
    if not attacks:
        raise NoAttackError(
            f"{law['law_id']}: emitter refused — no attack generated; "
            "a law that cannot be attacked cannot be published"
        )
    ranked = rank_attacks(attacks)
    return {
        "law_id": law["law_id"],
        "statement": law["statement"],
        "scope": law["scope"],
        "source_model": law["source_model"],
        "architecture_family": law["architecture_family"],
        "organ_class": law["organ_class"],
        "n_attacks": len(ranked),
        "selected_attack_id": ranked[0]["attack_id"],
        "selected_family": ranked[0]["family"],
        "selected_cost_units": ranked[0]["cost_units"],
        "selected_p_refutation": ranked[0]["p_refutation"],
        "ranked_attack_ids": [a["attack_id"] for a in ranked],
        "ranked_attacks": ranked,
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }


def apply_result(
    law: dict[str, Any],
    attack: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Consume an experiment result and return a scope update.

    Refuting verdicts MUST move scope DOWN. A no-op refutation raises
    ScopeUnmovedError — that is the loop bug the tests watch fire.
    """
    validate_law(law)
    before = law["scope"]
    verdict = str(result.get("verdict") or "")
    if verdict in REFUTING_VERDICTS:
        after = attack["target_scope_if_refuted"]
        if not is_downgrade(before, after):
            raise ScopeUnmovedError(
                f"{law['law_id']} via {attack.get('attack_id')}: refuting verdict "
                f"{verdict!r} would leave scope at {before!r} (target {after!r}). "
                "A refutation that does not change scope is a bug in the loop."
            )
        moved = True
        direction = "DOWN"
    elif verdict in {"HOLDS", "INCONCLUSIVE"}:
        after = before
        moved = False
        direction = "NONE"
    else:
        raise ValueError(
            f"unknown result verdict {verdict!r}; "
            "expected HOLDS, REFUTED, INCONCLUSIVE, TRAP_TRIGGERED, or HARNESS_SELF_MEASURE"
        )
    return {
        "law_id": law["law_id"],
        "attack_id": attack["attack_id"],
        "family": attack["family"],
        "verdict": verdict,
        "scope_before": before,
        "scope_after": after,
        "moved": moved,
        "direction": direction,
        "reason": result.get("reason") or result.get("falsifier_observed") or verdict,
        "synthetic": bool(result.get("synthetic", False)),
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
        "law_after": {**{k: law[k] for k in LAW_FIELDS}, "scope": after},
    }


# ---------------------------------------------------------------------------
# CPU-only trap execution (synthetic vectors / protocol controls, not a model)
# ---------------------------------------------------------------------------


def _cosine_and_rel(a, b) -> tuple[float, float]:
    import numpy as np

    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    cosine = float(np.dot(a, b) / denom) if denom else 0.0
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(a) or 1.0))
    return cosine, rel


def run_scale_invariance_trap(dim: int = 256, seed: int = 0, scale: float = 0.01) -> dict[str, Any]:
    """Replay the cosine-as-gate failure on a synthetic vector. CPU only."""
    import numpy as np

    rng = np.random.RandomState(seed)
    w = rng.randn(dim).astype(np.float64)
    s = scale * w
    cosine, rel = _cosine_and_rel(w, s)
    # Cosine is scale-invariant: this must be ~1.0 while magnitude dies.
    trap_fired = cosine > 0.99 and rel > 0.9
    return {
        "control": "scale_invariance",
        "dim": dim,
        "seed": seed,
        "scale": scale,
        "cosine": cosine,
        "rel_fro": rel,
        "trap_fired": trap_fired,
        "note": "synthetic vector; not a model organ; STATIC_ONLY",
    }


def run_skip_as_pass_trap(suite: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    suite = suite or [
        {"name": "control_a", "status": "SKIPPED"},
        {"name": "control_b", "status": "SKIPPED"},
    ]
    naive = all(t.get("status") in {"PASSED", "SKIPPED"} for t in suite) and bool(suite)
    honest = all(t.get("status") == "PASSED" for t in suite) and bool(suite)
    return {
        "control": "skip_counted_as_pass",
        "n_tests": len(suite),
        "n_skipped": sum(1 for t in suite if t.get("status") == "SKIPPED"),
        "naive_pass": naive,
        "honest_pass": honest,
        "trap_fired": bool(naive and not honest),
        "note": "SKIPPED is not PASSED; a grade that cannot fail is not a grade",
    }


def run_receipt_reread_trap(*, ran_generator: bool, read_checked_in_receipt: bool) -> dict[str, Any]:
    trap_fired = read_checked_in_receipt and not ran_generator
    return {
        "control": "receipt_reread_as_run",
        "ran_generator": ran_generator,
        "read_checked_in_receipt": read_checked_in_receipt,
        "trap_fired": trap_fired,
        "note": "a test that only loads a checked-in receipt has not run the tool",
    }


def execute_attack(attack: dict[str, Any], law: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the CPU-only portion of a spec. Physical arms stay UNKNOWN / not_run."""
    family = attack["family"]
    if family == "measurement_trap":
        inp = attack["inputs"]
        scale = run_scale_invariance_trap(
            dim=int(inp.get("vector_dim", 256)),
            seed=int(inp.get("vector_seed", 0)),
            scale=float(inp.get("scale_factor", 0.01)),
        )
        skip = run_skip_as_pass_trap(inp.get("skip_suite"))
        reread = run_receipt_reread_trap(ran_generator=True, read_checked_in_receipt=False)
        fired = [c["control"] for c in (scale, skip, reread) if c["trap_fired"]]
        # scale + skip fire on the built-in controls; reread does not because we ran.
        verdict = "TRAP_TRIGGERED" if fired else "HOLDS"
        return {
            "verdict": verdict,
            "family": family,
            "attack_id": attack["attack_id"],
            "controls": [scale, skip, reread],
            "traps_fired": fired,
            "synthetic": True,
            "physical_arm": "not_run",
            "evidence_class": "STATIC_ONLY",
            "bench_state": "UNKNOWN",
            "reason": (
                f"CPU controls fired: {fired}" if fired
                else "CPU controls did not fire"
            ),
        }
    return {
        "verdict": "INCONCLUSIVE",
        "family": family,
        "attack_id": attack["attack_id"],
        "synthetic": True,
        "physical_arm": "not_run",
        "reason": (
            f"{family} requires a physical experiment this sidecar cannot run "
            "(no GPU lease; STATIC_ONLY)"
        ),
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }


# ---------------------------------------------------------------------------
# Law loading: Odyssey II store if present, else inline fixture. Never import
# the sibling module. Absence is not an error.
# ---------------------------------------------------------------------------


def fixture_laws() -> list[dict[str, Any]]:
    """Inline laws recovered from existing receipts, used when the store is absent.

    These are historically-held claims, not new measurements. They exist so
    Odyssey III can attack something real while the sibling lane writes the store.
    """
    return [
        {
            "law_id": "LAW-COSINE-ADEQUACY-GATE",
            "statement": (
                "Mean component cosine >= 0.86 on held-out activations is an adequacy "
                "certificate for a packed organ: the organ is functionally preserved."
            ),
            "source_model": "glm-5.2 / qwen3-80b / qwen3.8-27b",
            "source_device": "APPLE_GPU_0",
            "architecture_family": "moe_and_dense_qwen_glm",
            "organ_class": "packed_organ.held_out_activation",
            "backend": "metal",
            "evidence_strength": "historically_used_as_gate",
            "evidence_refs": [
                "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
                "receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
            ],
            "scope": "GENERIC_VERIFIED",
            "transfer_candidates": [
                "qwen3-30b-a3b",
                "qwen3-vl-30b-a3b",
                "any_packed_organ",
            ],
            "transfer_confidence": 0.85,
            "counterexample_requirement": (
                "a packing whose cosine clears 0.86 while magnitude, generation, or "
                "rel_fro shows the organ is destroyed (0.01*W is the canonical trap)"
            ),
        },
        {
            "law_id": "LAW-Q4-REPRESENTATION-FIDELITY-TRANSFER",
            "statement": (
                "ws_rtn_q4_g64 weight cosine remains >= 0.993 on moe expert gate_proj "
                "across Qwen3-30B-A3B family specimens, so the representation transfers."
            ),
            "source_model": "Qwen3-30B-A3B",
            "source_device": "APPLE_GPU_0",
            "architecture_family": "qwen3_moe",
            "organ_class": "moe_expert.gate_proj",
            "backend": "metal",
            "evidence_strength": "receipt_backed_one_projection",
            "evidence_refs": [
                "receipts/headless/ACCELERATOR_TRANSFER_VERIFIED.json",
            ],
            "scope": "FAMILY_VERIFIED",
            "transfer_candidates": [
                "Qwen3-VL-30B-A3B-Instruct",
                "Kimi-VL-A3B-Instruct",
            ],
            "transfer_confidence": 0.70,
            "counterexample_requirement": (
                "a same-organ packing whose cosine or kernel identity falls outside "
                "the claimed band once storage layout is the variable"
            ),
        },
        {
            "law_id": "LAW-AFFINE-SEEDED-MATCHED-BITS",
            "statement": (
                "An affine-seeded Odyssey transfer, fitted as a METHOD on Qwen MLP, "
                "beats uniform round-to-nearest on moe_expert at matched bits per weight "
                "on every scored tier."
            ),
            "source_model": "Qwen3-30B-A3B",
            "source_device": "APPLE_GPU_0",
            "architecture_family": "qwen3_moe",
            "organ_class": "moe_expert",
            "backend": "metal",
            "evidence_strength": "receipt_backed_matched_bits_not_search_cost",
            "evidence_refs": [
                "receipts/headless/ODYSSEY_TRANSFER_PROVEN.json",
            ],
            "scope": "MODEL_LOCAL",
            "transfer_candidates": ["Flash dense parent", "Qwen3-VL-30B-A3B-Instruct"],
            "transfer_confidence": 0.40,
            "counterexample_requirement": (
                "a held-out expert or a different organ on the same specimen where "
                "the seeded family is not better at matched bits"
            ),
        },
        {
            "law_id": "LAW-PACKED-GEMV-DIRECT-TRANSFER",
            "statement": (
                "Packed low-bit GEMV is a DIRECT_TRANSFER primitive from the Qwen3.8 "
                "accelerator parent; command-buffer scheduling is provider-neutral."
            ),
            "source_model": "qwen3.8-27b-abliterated",
            "source_device": "APPLE_GPU_0",
            "architecture_family": "qwen3_dense_hybrid",
            "organ_class": "packed_lowbit_gemv",
            "backend": "metal",
            "evidence_strength": "transfer_map_hypothesis",
            "evidence_refs": [
                "receipts/headless/QWEN38_ACCELERATOR_TRANSFER_MAP.json",
            ],
            "scope": "FAMILY_VERIFIED",
            "transfer_candidates": ["Flash", "Qwen27"],
            "transfer_confidence": 0.55,
            "counterexample_requirement": (
                "a Flash or Qwen27 packing of the same primitive whose kernel identity "
                "or complete-token path does not reuse, or whose 'speedup' is a compiler default"
            ),
        },
    ]


def _extract_law_records(doc: Any) -> list[Any]:
    if isinstance(doc, list):
        return doc
    if not isinstance(doc, dict):
        return []
    for key in ("laws", "entries", "records"):
        v = doc.get(key)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            inner = v.get("laws")
            if isinstance(inner, list):
                return inner
    nested = doc.get("store")
    if isinstance(nested, dict) and isinstance(nested.get("laws"), list):
        return nested["laws"]
    keyed = [v for v in doc.values() if isinstance(v, dict) and "law_id" in v]
    return keyed


def load_laws() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load Odyssey II laws. Store wins when it yields at least one valid law."""
    meta: dict[str, Any] = {
        "store_path": "receipts/future/ODYSSEY2_LAW_STORE.json",
        "store_present": LAW_STORE.is_file(),
        "source": "inline_fixture",
        "n_store_records": 0,
        "n_store_valid": 0,
        "n_store_rejected": 0,
        "rejected_store_ids": [],
    }
    if LAW_STORE.is_file():
        doc = load_json(LAW_STORE)
        records = _extract_law_records(doc)
        meta["n_store_records"] = len(records)
        valid: list[dict[str, Any]] = []
        rejected: list[str] = []
        for rec in records:
            try:
                valid.append(validate_law(rec))
            except LawSchemaError as e:
                rejected.append(str(e))
        meta["n_store_valid"] = len(valid)
        meta["n_store_rejected"] = len(rejected)
        meta["rejected_store_ids"] = sorted(rejected)[:20]
        if valid:
            valid.sort(key=lambda l: l["law_id"])
            meta["source"] = "receipts/future/ODYSSEY2_LAW_STORE.json"
            return valid, meta
        meta["source"] = "inline_fixture_store_present_but_no_valid_laws"
    laws = fixture_laws()
    for law in laws:
        validate_law(law)
    laws.sort(key=lambda l: l["law_id"])
    return laws, meta


def _recovered_implementation() -> list[dict[str, Any]]:
    rows = [
        {
            "path": "tools/headless/adversarial_sweep.py",
            "role": "ten gate attacks (self-certified PASS, dead evidence, vacuous check, skip-shaped)",
            "gap": "attacks PASS gates, not Odyssey II law records; no scope loop",
        },
        {
            "path": "receipts/headless/FRONTIER_ADVERSARY.json",
            "role": "archived Noetic frontier adversary findings",
            "gap": "historical receipt; no live runner remains in this sparse checkout",
        },
        {
            "path": "receipts/headless/NOETIC_GATE_ADVERSARY.json",
            "role": "archived re-verification of newly closed headless gates",
            "gap": "historical receipt; the external VisionMCP runner is retired",
        },
        {
            "path": "tools/headless/negative_science.py",
            "role": "failure store, nine fields, three levels, promotion refusal",
            "gap": "query-before-experiment, not attack generation against a live law",
        },
        {
            "path": "receipts/headless/ODYSSEY_ADVERSARIAL_SWEEP.json",
            "role": "G031 sweep receipt (27 gates, 1 weakened on stale_cache)",
            "gap": "evidence of a different adversary; not a law store",
        },
        {
            "path": "receipts/headless/NOETIC_NEGATIVE_SCIENCE.json",
            "role": "watched-fail memory: cosine scale-invariance, Gaussian proxy, skip-as-gate",
            "gap": "consumed as scar memory for attack families; not extended in place",
        },
        {
            "path": "receipts/ascent-2026-08-16/NEGATIVE_SCIENCE_REGISTER.json",
            "role": "38 entries REFUTED/DEAUTHORISED/CATEGORY_ERROR with retry_when",
            "gap": "ascent-campaign register, not an Odyssey III loop",
        },
        {
            "path": "tools/odyssey/known_failures.py",
            "role": "sealed negatives with cheap reproductions",
            "gap": "asserts a receipt still reads as negative; does not attack a law",
        },
        {
            "path": "tools/odyssey/contamination.py",
            "role": "train/eval contamination barrier (shingles/jaccard)",
            "gap": "corpus overlap, not DIAGNOSTIC_RELATIVE vs PROTECTED_ABSOLUTE",
        },
        {
            "path": "receipts/headless/ACCELERATOR_TRANSFER_VERIFIED.json",
            "role": "kernel vs representation transfer; layout trap on Qwen3-VL",
            "gap": "seed for LAW-Q4-REPRESENTATION-FIDELITY-TRANSFER",
        },
        {
            "path": "receipts/headless/ODYSSEY_TRANSFER_PROVEN.json",
            "role": "cold vs transfer; matched-bits 3/3; honest_note that COLD won landing",
            "gap": "seed for LAW-AFFINE-SEEDED-MATCHED-BITS",
        },
        {
            "path": "receipts/future/CLAUDE_GLOBAL_FRONTIER.json",
            "role": "F010 (no scoped law store), F009 (negative science unindexed), F015 (no ingest)",
            "gap": "this module is the Odyssey III sidecar F010/F015 asked for, not a fork of them",
        },
    ]
    for row in rows:
        row["on_disk"] = (REPO / row["path"]).is_file()
    return rows


def _find_attack(attack_id: str, plans: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    for plan in plans:
        for spec in plan["ranked_attacks"]:
            if spec["attack_id"] == attack_id:
                law = next(l for l in load_laws()[0] if l["law_id"] == plan["law_id"])
                return spec, law
    raise KeyError(attack_id)


def selftest() -> dict[str, Any]:
    """Closed loop on LAW-COSINE-ADEQUACY-GATE with a synthetic refuting result.

    Scope MUST move DOWN. Raises ScopeUnmovedError / NoAttackError if the
    loop is a no-op. The CPU measurement trap is run as an observation; the
    result fed to apply_result is synthetic, as the contract requires.
    """
    laws = {l["law_id"]: l for l in fixture_laws()}
    law = laws["LAW-COSINE-ADEQUACY-GATE"]
    plan = emit_for_law(law)
    selected = plan["ranked_attacks"][0]
    cpu = execute_attack(selected, law)
    synthetic = {
        "verdict": "REFUTED",
        "synthetic": True,
        "reason": (
            "synthetic refuting result for closed-loop proof: cosine is scale-invariant, "
            "so 0.01*W scores ~1.0 while rel_fro ~0.99; mixed-2p0 cosine 0.907 was "
            "native-INCOHERENT. The metric is not an adequacy certificate."
        ),
        "cpu_observation": cpu,
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }
    update = apply_result(law, selected, synthetic)
    if not update["moved"] or update["direction"] != "DOWN":
        raise ScopeUnmovedError(
            f"closed loop failed to downgrade {law['law_id']}: {update}"
        )
    if not is_downgrade(update["scope_before"], update["scope_after"]):
        raise ScopeUnmovedError(
            f"closed loop claimed DOWN but ranks are not ordered: {update}"
        )
    hold = apply_result(law, selected, {
        "verdict": "HOLDS",
        "synthetic": True,
        "reason": "negative control: a hold must not move scope",
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    })
    if hold["moved"] or hold["scope_after"] != law["scope"]:
        raise ScopeUnmovedError(f"HOLDS result moved scope: {hold}")
    return {
        "loop": [
            "LAW",
            "TRANSFER_HYPOTHESIS",
            "ADVERSARIAL_TARGET",
            "EXPERIMENT_SPEC",
            "RESULT",
            "LAW_SCOPE_UPDATE",
        ],
        "law_id": law["law_id"],
        "statement": law["statement"],
        "selected_attack_id": selected["attack_id"],
        "selected_family": selected["family"],
        "transfer_hypothesis": selected["transfer_hypothesis"],
        "adversarial_target": selected["adversarial_target"],
        "experiment_spec_command": selected["command"],
        "synthetic_result": {
            "verdict": synthetic["verdict"],
            "reason": synthetic["reason"],
            "synthetic": True,
        },
        "cpu_observation": cpu,
        "scope_update": update,
        "hold_negative_control": hold,
        "scope_before": update["scope_before"],
        "scope_after": update["scope_after"],
        "moved_down": True,
        "ranked_attack_ids": plan["ranked_attack_ids"],
        "evidence_class": "STATIC_ONLY",
        "bench_state": "UNKNOWN",
    }


def build() -> Path:
    laws, source_meta = load_laws()
    if not laws:
        raise NoAttackError("no laws to attack; emitter refused")
    plans = [emit_for_law(law) for law in laws]
    for plan in plans:
        if plan["n_attacks"] < 1:
            raise NoAttackError(f"{plan['law_id']}: emitter refused — zero attacks")
    loop = selftest()
    recovered = _recovered_implementation()
    on_disk = sorted(r["path"] for r in recovered if r["on_disk"])
    missing_on_disk = sorted(r["path"] for r in recovered if not r["on_disk"])
    doc = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Odyssey III adversary: for any Odyssey II law, emit ranked executable "
            "attack specs and close LAW -> TRANSFER HYPOTHESIS -> ADVERSARIAL TARGET "
            "-> EXPERIMENT SPEC -> RESULT -> LAW SCOPE UPDATE. Sidecar has no GPU; "
            "every spec is STATIC_ONLY / UNKNOWN."
        ),
        "odysseys": ["I WHAT IS TRUE?", "II WHAT DID HAWKING ALREADY LEARN?",
                     "III WHERE IS HAWKING WRONG?"],
        "eras": [
            "I Genesis of the Laboratory",
            "II Compounding Civilization",
            "III Autonomous Science Civilization",
            "IV Synthetic Machine Civilization",
            "V Released Hawking Civilization",
        ],
        "law_source": source_meta,
        "n_laws": len(plans),
        "n_attacks": sum(p["n_attacks"] for p in plans),
        "attack_families": list(ATTACK_FAMILIES),
        "scope_ladder_general_to_refuted": list(SCOPE_LADDER),
        "selection_rule": (
            "selection_score = cost_units / p_refutation; emit lowest score first "
            "(cheapest expected cost per refutation). Every law gets at least one "
            "attack or the emitter refuses."
        ),
        "past_failure_shapes_encoded": list(PAST_FAILURE_SHAPES),
        "laws_attacked": plans,
        "closed_loop": loop,
        "recovered_implementation": recovered,
        "gaps_closed": [
            "law-record input with the exact Odyssey II field set, store-or-fixture",
            "nine attack-family generators each emitting an executable spec",
            "ranking by cost_units / p_refutation; cheapest capable first",
            "apply_result(law, attack, result) -> scope_update with mandatory DOWN",
            "emitter refusal when a law generates zero attacks (watched failing in tests)",
            "CPU replay of the cosine scale-invariance and SKIPPED-as-pass traps",
            "closed loop demonstrated on LAW-COSINE-ADEQUACY-GATE with synthetic REFUTED",
        ],
        "negative_findings": [
            (
                "receipts/future/ODYSSEY2_LAW_STORE.json "
                + ("present" if source_meta["store_present"] else "absent")
                + "; sibling module was not imported"
            ),
            (
                "sparse checkout: recovered adversary/negative-science paths not on disk: "
                + (", ".join(missing_on_disk) if missing_on_disk else "(all recovered paths materialized)")
            ),
            "no GPU lease: physical arms of every spec are not_run / UNKNOWN",
            "did not execute tools/headless/adversarial_sweep.py (it writes Codex receipts)",
            "did not execute retired frontier/gate adversary runners; their sealed receipts remain evidence",
            "evidence_refs on fixture laws are citations; this checkout may not materialize them",
            "independence limitation: the same operator wrote the fixture laws and the attacks",
            "on_disk recovered paths: " + (", ".join(on_disk) if on_disk else "(none)"),
        ],
        "claim_boundary_reminder": (
            "DIAGNOSTIC_RELATIVE guides, never promotes. PROTECTED_ABSOLUTE decides. "
            "This module produces neither."
        ),
    }
    return write_receipt(RECEIPT, doc, "tools/future/odyssey3_adversary.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--replay-attack", metavar="ATTACK_ID")
    a = ap.parse_args()
    if a.selftest:
        loop = selftest()
        print(json.dumps({
            "ok": True,
            "law_id": loop["law_id"],
            "scope_before": loop["scope_before"],
            "scope_after": loop["scope_after"],
            "selected_family": loop["selected_family"],
        }, indent=1, sort_keys=True))
        return 0
    if a.replay_attack:
        laws, _ = load_laws()
        plans = [emit_for_law(law) for law in laws]
        spec, law = _find_attack(a.replay_attack, plans)
        print(json.dumps(execute_attack(spec, law), indent=1, sort_keys=True))
        return 0
    print(build())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
