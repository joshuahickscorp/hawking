#!/usr/bin/env python3.12
"""Hawking Gravity: sealed evidence objects that authorize escape from the sub-bit region.

A parent may rise above one physical BPW only when one of these tamper-detectable,
self-sealed receipts justifies it:

  - hawking.gravity.escape_receipt.v1            (section 5): the general justification.
  - hawking.gravity.structural_incompatibility.v1 (section 4B): a representation family
      cannot represent a required tensor class / architecture component.
  - hawking.gravity.physical_impossibility.v1     (section 4C): no complete artifact in the
      target region can satisfy the mandatory byte budget.

The Escape Receipt must ANSWER: why is Hawking justified in spending more than one physical
bit per weight on this parent? The validator rejects the four non-justifications explicitly
named in the doctrine:
  * a missing experiment,
  * a scheduler deferral,
  * a single failed scalar codec,
  * one weak Doctor treatment.

Every receipt is sealed byte-identically to the campaign reporter (eco_common.seal_field),
so mutating any field breaks the seal (tamper-detectable).
"""
from __future__ import annotations

import os
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from eco_common import EcoError, seal_field, sealed, now_iso  # noqa: E402
import succ_gravity_policy as gp  # noqa: E402

ESCAPE_RECEIPT_SCHEMA = "hawking.gravity.escape_receipt.v1"
STRUCTURAL_INCOMPAT_SCHEMA = "hawking.gravity.structural_incompatibility.v1"
PHYSICAL_IMPOSSIBILITY_SCHEMA = "hawking.gravity.physical_impossibility.v1"

_RECEIPT_SEAL = "receipt_sha256"


class ReceiptError(EcoError):
    """Fail-closed error while building or validating a Gravity receipt."""


# ── Escape Receipt (section 5) ─────────────────────────────────────────────────────────
_ESCAPE_REQUIRED = (
    "parent_identity", "parameter_count", "capability_contract",
    "attempted_subbit_rates", "attempted_representation_families", "evidence_level_by_family",
    "diagnosis_by_rate", "doctor_treatments_attempted", "doctor_treatments_unavailable",
    "physical_byte_budgets", "causal_treatment_reachability", "lower_rate_quality_outcomes",
    "structural_incompatibility_receipts", "physical_impossibility_receipts", "uncertainty",
    "recommended_next_higher_rate", "reopening_criteria", "reviewer_identity",
)


def escape_receipt(**fields: Any) -> dict[str, Any]:
    """Build a sealed Escape Receipt binding every field the doctrine requires (section 5).

    Missing required fields fail closed. `attempted_subbit_rates` and
    `recommended_next_higher_rate` are stored as exact rational identities.
    """
    missing = [k for k in _ESCAPE_REQUIRED if k not in fields]
    if missing:
        raise ReceiptError(f"escape receipt missing required fields: {missing}")
    receipt = {"schema": ESCAPE_RECEIPT_SCHEMA, "created_at": now_iso()}
    receipt.update({k: fields[k] for k in _ESCAPE_REQUIRED})
    # normalize rate identities if callers passed Fractions/strings
    receipt["attempted_subbit_rates"] = [_as_identity(r) for r in fields["attempted_subbit_rates"]]
    if fields.get("recommended_next_higher_rate") is not None:
        receipt["recommended_next_higher_rate"] = _as_identity(fields["recommended_next_higher_rate"])
    return seal_field(receipt, _RECEIPT_SEAL)


def _as_identity(r: Any) -> dict[str, Any]:
    if isinstance(r, dict) and "num" in r and "den" in r:
        return r
    return gp.rate_identity(gp.parse_rate(r) if isinstance(r, str) else r)


def escape_receipt_valid(receipt: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate an Escape Receipt: seal intact, all fields present, and the justification is
    real. Rejects the four named non-justifications (section 5)."""
    reasons: list[str] = []
    if not isinstance(receipt, dict) or receipt.get("schema") != ESCAPE_RECEIPT_SCHEMA:
        return False, ["not an escape_receipt.v1"]
    if not sealed(receipt, _RECEIPT_SEAL):
        reasons.append("seal invalid (tampered or unsealed)")
    for k in _ESCAPE_REQUIRED:
        if k not in receipt:
            reasons.append(f"missing field {k}")

    families = receipt.get("attempted_representation_families", []) or []
    rates = receipt.get("attempted_subbit_rates", []) or []
    diagnoses = receipt.get("diagnosis_by_rate", {}) or {}
    treatments = receipt.get("doctor_treatments_attempted", []) or []
    struct = receipt.get("structural_incompatibility_receipts", []) or []
    phys = receipt.get("physical_impossibility_receipts", []) or []

    # A missing experiment is not justification.
    if not rates:
        reasons.append("no sub-bit rate was attempted (a missing experiment is not justification)")

    # A scheduler deferral is not justification: if every diagnosis is a deferral, reject.
    diag_values = [str(v).lower() for v in diagnoses.values()]
    if diag_values and all(("defer" in v or "deferred" in v) for v in diag_values):
        reasons.append("every rate diagnosis is a scheduler deferral (not justification)")

    # A failed scalar codec is not justification: a single scalar-class family alone is thin.
    material = gp.count_material_families([f for f in families if f in gp.REPRESENTATION_FAMILIES])
    scalar_only = all(gp.REPRESENTATION_FAMILIES.get(f, {}).get("class") == "scalar_trellis"
                      for f in families) if families else True
    have_receipt = bool(struct) or bool(phys)
    if scalar_only and not have_receipt:
        reasons.append("only a scalar codec was tried and no structural/physical receipt "
                       "(a failed scalar codec is not justification)")
    # Otherwise Route-A style justification needs >=2 material families unless a receipt covers it.
    if material < 2 and not have_receipt:
        reasons.append("fewer than 2 materially different families and no structural/physical "
                       "receipt (coverage not established)")

    # One weak Doctor treatment is not justification: with reachable treatments left untried, reject.
    reachable_untried = [t for t in gp.REACHABLE_TREATMENTS
                         if gp.treatment_reachable(t) and t not in treatments]
    if len(treatments) <= 1 and reachable_untried and not have_receipt:
        reasons.append("only one Doctor treatment attempted while reachable treatments remain "
                       "untried (one weak treatment is not justification)")

    return (not reasons), reasons


# ── structural-incompatibility receipt (section 4B) ────────────────────────────────────
_STRUCT_REQUIRED = (
    "parent_identity", "exact_revision", "component", "attempted_representation",
    "incompatibility", "evidence_level", "alternatives_considered", "reopening_condition",
)


def structural_incompatibility_receipt(**fields: Any) -> dict[str, Any]:
    missing = [k for k in _STRUCT_REQUIRED if k not in fields]
    if missing:
        raise ReceiptError(f"structural-incompatibility receipt missing: {missing}")
    receipt = {"schema": STRUCTURAL_INCOMPAT_SCHEMA, "created_at": now_iso()}
    receipt.update({k: fields[k] for k in _STRUCT_REQUIRED})
    return seal_field(receipt, _RECEIPT_SEAL)


# ── physical-impossibility receipt (section 4C) ────────────────────────────────────────
# All mandatory byte categories must be present; a nominal base-rate estimate is insufficient.
_PHYS_REQUIRED = (
    "parent_identity", "exact_revision", "target_region", "min_whole_bpw", "envelope_bpw",
    "byte_budget",  # dict with every mandatory category below
)
_PHYS_BYTE_CATEGORIES = (
    "base_representation", "doctor_correction", "lexical_tensors", "routers", "shared_experts",
    "pass_through_tensors", "codebooks", "exception_tables", "indices", "metadata",
    "alignment", "packaging", "mandatory_runtime_tables",
)


def physical_impossibility_receipt(**fields: Any) -> dict[str, Any]:
    missing = [k for k in _PHYS_REQUIRED if k not in fields]
    if missing:
        raise ReceiptError(f"physical-impossibility receipt missing: {missing}")
    budget = fields["byte_budget"]
    if not isinstance(budget, dict):
        raise ReceiptError("byte_budget must be a dict of all mandatory categories")
    missing_cats = [c for c in _PHYS_BYTE_CATEGORIES if c not in budget]
    if missing_cats:
        raise ReceiptError(f"byte_budget missing mandatory categories (nominal base rate is "
                           f"insufficient): {missing_cats}")
    receipt = {"schema": PHYSICAL_IMPOSSIBILITY_SCHEMA, "created_at": now_iso()}
    receipt.update({k: fields[k] for k in _PHYS_REQUIRED})
    return seal_field(receipt, _RECEIPT_SEAL)


# ── generic verification (tamper-detectable) ───────────────────────────────────────────
def verify_receipt(receipt: dict[str, Any]) -> tuple[bool, list[str]]:
    """Dispatch by schema; check seal + required fields. Any field mutation breaks the seal."""
    if not isinstance(receipt, dict):
        return False, ["not a receipt object"]
    schema = receipt.get("schema")
    if schema == ESCAPE_RECEIPT_SCHEMA:
        return escape_receipt_valid(receipt)
    reasons: list[str] = []
    if schema == STRUCTURAL_INCOMPAT_SCHEMA:
        required = _STRUCT_REQUIRED
    elif schema == PHYSICAL_IMPOSSIBILITY_SCHEMA:
        required = _PHYS_REQUIRED
    else:
        return False, [f"unknown receipt schema {schema!r}"]
    if not sealed(receipt, _RECEIPT_SEAL):
        reasons.append("seal invalid (tampered or unsealed)")
    for k in required:
        if k not in receipt:
            reasons.append(f"missing field {k}")
    if schema == PHYSICAL_IMPOSSIBILITY_SCHEMA:
        budget = receipt.get("byte_budget", {}) or {}
        for c in _PHYS_BYTE_CATEGORIES:
            if c not in budget:
                reasons.append(f"byte_budget missing {c}")
    return (not reasons), reasons


def _fully_justified_example() -> dict[str, Any]:
    """A minimally complete, valid escape receipt (used by selftest and validation)."""
    return escape_receipt(
        parent_identity={"label": "72B", "hf_or_source_id": "Qwen/Qwen2.5-72B-Instruct",
                         "exact_revision": "pinned"},
        parameter_count=72_706_203_648,
        capability_contract={"metric": "exec_grounded_thesis_gate", "min_pass_rate": 0.90},
        attempted_subbit_rates=["1/4", "1/3", "1/2", "11/20", "4/5"],
        attempted_representation_families=["scalar_trellis_tqv2", "binary_latent_factors",
                                           "additive_codebooks"],
        evidence_level_by_family={"scalar_trellis_tqv2": "F2", "binary_latent_factors": "F2",
                                  "additive_codebooks": "F1"},
        diagnosis_by_rate={"1/4": "computation_collapse", "1/3": "mixed_failure",
                           "1/2": "signal_degraded_treated_insufficient", "4/5": "signal_degraded"},
        doctor_treatments_attempted=["doctor_static", "doctor_conditional", "doctor_full"],
        doctor_treatments_unavailable={"lora_kd": "no executable adapter",
                                       "strand_hessian": "no executable adapter"},
        physical_byte_budgets={"1/2": {"base": 0.5, "doctor": 0.4, "passthrough": 0.2055,
                                       "overhead": 0.0213}},
        causal_treatment_reachability={"missing_causal_subspace": True,
                                       "reachable_within_reserve": False},
        lower_rate_quality_outcomes={"1/2": {"capability": 0.61, "target": 0.90}},
        structural_incompatibility_receipts=[],
        physical_impossibility_receipts=[],
        uncertainty={"boundary_rungs": 1, "confidence": "medium"},
        recommended_next_higher_rate="5/4",
        reopening_criteria=["binary-pattern codebook packer lands",
                            "measured doctor reserve exceeds 0.10 bpw at scale"],
        reviewer_identity={"role": "operator", "signed": False},
    )


def selftest() -> dict[str, Any]:
    good = _fully_justified_example()
    ok, why = escape_receipt_valid(good)
    assert ok, why

    # tamper: mutating any field breaks the seal
    tampered = dict(good); tampered["parameter_count"] = 1
    ok_t, _ = escape_receipt_valid(tampered)
    assert not ok_t

    # a missing experiment is not justification
    thin = dict(good)
    thin["attempted_subbit_rates"] = []
    thin = seal_field({k: v for k, v in thin.items() if k != _RECEIPT_SEAL}, _RECEIPT_SEAL)
    ok_thin, why_thin = escape_receipt_valid(thin)
    assert not ok_thin and any("missing experiment" in r for r in why_thin)

    # a scheduler deferral is not justification
    deferred = dict(good)
    deferred["diagnosis_by_rate"] = {"1/2": "deferred", "1/3": "deferred"}
    deferred = seal_field({k: v for k, v in deferred.items() if k != _RECEIPT_SEAL}, _RECEIPT_SEAL)
    ok_def, why_def = escape_receipt_valid(deferred)
    assert not ok_def and any("deferral" in r for r in why_def)

    # a single failed scalar codec is not justification
    scalar = dict(good)
    scalar["attempted_representation_families"] = ["scalar_trellis_tqv2"]
    scalar["evidence_level_by_family"] = {"scalar_trellis_tqv2": "F2"}
    scalar = seal_field({k: v for k, v in scalar.items() if k != _RECEIPT_SEAL}, _RECEIPT_SEAL)
    ok_s, why_s = escape_receipt_valid(scalar)
    assert not ok_s and any("scalar codec" in r for r in why_s)

    # structural + physical receipts
    sr = structural_incompatibility_receipt(
        parent_identity={"label": "1T"}, exact_revision="pinned",
        component="vision_tower.attn", attempted_representation="binary_pattern_codebooks",
        incompatibility="multimodal cross-attn tensors lack a codebook-representable factorization",
        evidence_level="F2", alternatives_considered=["additive_codebooks", "vtq_vector_trellis"],
        reopening_condition="a vision-aware codebook packer lands")
    ok_sr, _ = verify_receipt(sr); assert ok_sr
    sr_bad = dict(sr); sr_bad["component"] = "x"
    ok_srb, _ = verify_receipt(sr_bad); assert not ok_srb

    pr = physical_impossibility_receipt(
        parent_identity={"label": "72B"}, exact_revision="pinned",
        target_region="< 0.20 bpw whole", min_whole_bpw=0.24, envelope_bpw=0.20,
        byte_budget={c: 0.01 for c in _PHYS_BYTE_CATEGORIES})
    ok_pr, _ = verify_receipt(pr); assert ok_pr
    try:
        physical_impossibility_receipt(parent_identity={}, exact_revision="x",
                                       target_region="x", min_whole_bpw=1, envelope_bpw=1,
                                       byte_budget={"base_representation": 0.1})
        raise AssertionError("nominal base-rate estimate must be rejected")
    except ReceiptError:
        pass
    return {"ok": True, "escape_valid": ok, "tamper_detected": not ok_t}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2, sort_keys=True))
