#!/usr/bin/env python3.12
"""Hawking Gravity: the sub-bit-first execution LAW (policy + invariants).

Gravity is an ENFORCED policy inside the existing single-owner successor pipeline,
not a second controller and not a philosophy document. The doctrine:

  Hawking must begin beneath one physical bit per weight whenever physically and
  architecturally meaningful. It may rise above one BPW only when measured evidence
  proves the lower representation cannot be restored within the complete physical
  byte and capability contract.

  The gravitational pull is toward the smallest complete artifact.
  The Event Horizon is the lowest point at which verified capability survives.
  Escape above sub-bit requires a sealed Escape Receipt.

This module owns the *policy surface*: exact-rational rates, parent-specific
gravitational starting points (priors from the sealed Sub-Bit Readiness Diagnostic),
mandatory sub-bit coverage (Routes A/B/C/D), the total-rate conservation law, the
representation-family escalation order, the EXTREME finalization gate, and the
default-off activation gate. It is additive and default-off: nothing here launches
model work; execution enters only through the existing successor admission/queue/
lease/transition boundary, and only when every gate below is satisfied.

Scientific ground (do not silently drift from it):
  reports/condense/subbit_frontier/SUBBIT_READINESS_PACKET.json
    seal sha256 f6c6b2d8cd046827add88202681fae9fbc30383831947ad1e4caef4490d38bda
  reports/condense/subbit_frontier/subbit_inverted_search_sim.py  (11/11 policy proof)
"""
from __future__ import annotations

import os
from fractions import Fraction
from typing import Any, Iterable

# eco_common is the successor controller's shared foundation (all succ_* import it).
import sys as _sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
from eco_common import EcoError, seal_field, sealed, now_iso  # noqa: E402

GRAVITY_POLICY_SCHEMA = "hawking.gravity.policy.v1"
GRAVITY_POLICY_VERSION = "2026-07-17.1"


class GravityError(EcoError):
    """Fail-closed error in the Gravity policy layer."""


# ── the hard invariant (section 3): enforced by code, not only documentation ───────────
INVARIANT: dict[str, Any] = {
    "default_search_direction": "upward from a parent-specific sub-bit stress point",
    "default_target_region": "physical whole-artifact BPW < 1.0",
    "fallback_direction": "upward only",
    "quality_contract": "unchanged",
    "physical_accounting": "complete whole artifact",
    "heavy_controller_count": 1,
    "escape_from_subbit": "requires a sealed Escape Receipt",
}

ONE_BPW = Fraction(1, 1)


# ── exact-rational rate ladder (section 6): numerator/denominator identity, no floats ──
# Superset of the diagnostic sim's LADDER; adds 1/5 (0.20) per the Gravity rate table.
RATE_LADDER: list[Fraction] = [
    Fraction(1, 10), Fraction(1, 5), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2),
    Fraction(11, 20), Fraction(4, 5), Fraction(1, 1), Fraction(5, 4), Fraction(3, 2),
    Fraction(2, 1), Fraction(3, 1), Fraction(4, 1),
]
assert RATE_LADDER == sorted(RATE_LADDER), "rate ladder must be strictly ascending"
assert len(RATE_LADDER) == len(set(RATE_LADDER)), "rate ladder must be unique"
assert Fraction(1, 5) in RATE_LADDER, "1/5 (0.20) is a required Gravity rate"


def rate_identity(q: Fraction) -> dict[str, Any]:
    """Bind numerator AND denominator as the scientific identity. `value` is a
    convenience float and is NEVER the identity (section 6)."""
    q = Fraction(q)
    return {"num": q.numerator, "den": q.denominator,
            "label": f"{q.numerator}/{q.denominator}", "value": float(q)}


def rate_from_identity(d: dict[str, Any]) -> Fraction:
    return Fraction(int(d["num"]), int(d["den"]))


def parse_rate(text: str) -> Fraction:
    """Parse an exact rational rate. Accepts 'n/d' or an integer string. A rounded
    decimal like '0.33' is REJECTED as an identity (it is not exact); pass '1/3'."""
    s = str(text).strip()
    if "/" in s:
        n, d = s.split("/", 1)
        return Fraction(int(n), int(d))
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return Fraction(int(s), 1)
    raise GravityError(f"rate identity must be exact rational 'n/d' or integer, not {text!r}")


def is_subbit(q: Fraction) -> bool:
    return Fraction(q) < ONE_BPW


def ladder_index(q: Fraction) -> int:
    q = Fraction(q)
    if q not in RATE_LADDER:
        raise GravityError(f"rate {q} is not on the exact-rational ladder")
    return RATE_LADDER.index(q)


def next_higher(q: Fraction) -> Fraction | None:
    i = ladder_index(q)
    return RATE_LADDER[i + 1] if i + 1 < len(RATE_LADDER) else None


def next_lower(q: Fraction) -> Fraction | None:
    i = ladder_index(q)
    return RATE_LADDER[i - 1] if i > 0 else None


# ── parent-specific gravitational starting points (section 7) ──────────────────────────
# PRIORS, not immutable constants: bound from the sealed readiness packet's
# `parent_specific_starting_rates`. compute_stress_start() recomputes feasibility
# against the live envelope and treats these as anchors.
ENVELOPE_GIB = {"headline": 64, "conservative": 58, "aggressive": 80}
DOCTOR_RESERVE_PRIOR = Fraction(1, 10)   # packet headline 0.1 bpw (band 0.05..0.20), unmeasured at scale

PARENT_PRIORS: dict[str, dict[str, Any]] = {
    "72B": {
        "n_params": 72_706_203_648, "resident_ceiling_bpw": 7.5613, "fixed_overhead_bpw": 0.0213,
        "passthru_floor_bpw": 0.2055, "base_target_bpw": 7.44,
        "subbit_stress_start": Fraction(4, 5), "subbit_is": "collapse_probe_not_resident_forced",
        "first_likely_viable_bpw": 1.44, "representation_families": ["scalar_trellis_tqv2"],
        "risk": "high", "architecture_family": "qwen2.5-dense",
        "hf_or_source_id": "Qwen/Qwen2.5-72B-Instruct",
    },
    "120B": {
        "n_params": 116_829_156_672, "resident_ceiling_bpw": 4.7056, "fixed_overhead_bpw": 0.0208,
        "passthru_floor_bpw": 0.0591, "base_target_bpw": 4.58,
        "subbit_stress_start": Fraction(4, 5), "subbit_is": "collapse_probe_not_resident_forced",
        "first_likely_viable_bpw": 1.2, "representation_families": ["scalar_trellis_tqv2", "mxfp4_native_control"],
        "risk": "high", "architecture_family": "gpt-oss-moe",
        "hf_or_source_id": "openai/gpt-oss-120b",
    },
    "685B": {
        "n_params": 685_000_000_000, "resident_ceiling_bpw": 0.8026, "fixed_overhead_bpw": 0.0201,
        "passthru_floor_bpw": 0.0161, "base_target_bpw": 0.68,
        "subbit_stress_start": Fraction(11, 20), "subbit_is": "resident_forced",
        "first_likely_viable_bpw": "0.70-0.80", "representation_families": ["scalar_trellis_tqv2", "vtq_vector_trellis"],
        "risk": "very_high_unproven", "architecture_family": "deepseek-moe",
        "hf_or_source_id": "deepseek-ai/DeepSeek-V3.2", "frontier_row_id": "deepseek-v3.2-685b",
    },
    "1T": {
        "n_params": 1_000_000_000_000, "resident_ceiling_bpw": 0.5498, "fixed_overhead_bpw": 0.0201,
        "passthru_floor_bpw": 0.0138, "base_target_bpw": 0.43,
        "subbit_stress_start": Fraction(1, 3), "subbit_is": "resident_forced",
        "first_likely_viable_bpw": "0.45-0.55", "representation_families": ["scalar_trellis_tqv2", "vtq_vector_trellis"],
        "risk": "very_high_unproven", "architecture_family": "kimi-moe",
        "hf_or_source_id": "moonshotai/Kimi-K2.6", "frontier_row_id": "kimi-k2.6-1t",
    },
    "1.6T": {
        "n_params": 1_600_000_000_000, "resident_ceiling_bpw": 0.3436, "fixed_overhead_bpw": 0.0201,
        "passthru_floor_bpw": 0.0097, "base_target_bpw": 0.22,
        "subbit_stress_start": Fraction(1, 4), "subbit_is": "resident_forced_stream_only",
        "first_likely_viable_bpw": "0.25-0.33", "representation_families": ["scalar_trellis_tqv2", "vtq_vector_trellis"],
        "risk": "very_high_unproven", "architecture_family": "deepseek-v4-moe",
        "hf_or_source_id": "deepseek-ai/DeepSeek-V4-Pro", "frontier_row_id": "deepseek-v4-pro-1.6t",
        "full_install_forbidden": True,
    },
}
# frontier row_id -> canonical prior label, so giant queue rows resolve to a prior.
_ROWID_ALIAS = {v["frontier_row_id"]: k for k, v in PARENT_PRIORS.items() if "frontier_row_id" in v}


def prior_for(parent_label: str) -> dict[str, Any]:
    key = parent_label if parent_label in PARENT_PRIORS else _ROWID_ALIAS.get(parent_label, parent_label)
    if key not in PARENT_PRIORS:
        raise GravityError(f"no gravitational prior for parent {parent_label!r}")
    return {**PARENT_PRIORS[key], "prior_label": key}


def resident_ceiling_bpw(n_params: int, envelope_gib: float = ENVELOPE_GIB["headline"]) -> Fraction:
    """8 * M_safe / N  as an exact rational (whole-artifact bpw that fits resident)."""
    m_safe = int(round(envelope_gib * (1024 ** 3)))
    return Fraction(8 * m_safe, int(n_params))


def compute_stress_start(parent_label: str, *, envelope_gib: float = ENVELOPE_GIB["headline"]) -> dict[str, Any]:
    """Section 7: derive the gravitational stress-start from the exact parameter count,
    the safe resident envelope, fixed overhead, doctor reserve, and the packet prior.

    The prior is an ANCHOR. For resident-forced parents the returned rate is the highest
    ladder rate at or below BOTH the prior and the recomputed resident ceiling; for
    collapse-probe parents the prior stands (it maps the dense collapse floor, not fit).
    """
    p = prior_for(parent_label)
    anchor = Fraction(p["subbit_stress_start"])
    ceiling = resident_ceiling_bpw(p["n_params"], envelope_gib)
    resident_forced = str(p["subbit_is"]).startswith("resident_forced")
    # candidate ladder rates at/below the anchor
    below_anchor = [r for r in RATE_LADDER if r <= anchor]
    chosen = below_anchor[-1] if below_anchor else RATE_LADDER[0]
    if resident_forced:
        # never start above what the whole artifact can even fit resident
        fit = [r for r in RATE_LADDER if r <= min(anchor, ceiling)]
        chosen = fit[-1] if fit else RATE_LADDER[0]
    return {
        "parent_label": parent_label, "prior_label": p["prior_label"],
        "anchor_rate": rate_identity(anchor), "chosen_stress_rate": rate_identity(chosen),
        "resident_ceiling_recomputed": rate_identity(_nearest_ladder_at_or_below(ceiling) or RATE_LADDER[0]),
        "resident_ceiling_exact": float(ceiling), "resident_forced": resident_forced,
        "subbit_is": p["subbit_is"], "envelope_gib": envelope_gib,
        "note": "prior is an anchor; recomputed against live envelope (section 7)",
    }


def _nearest_ladder_at_or_below(q: Fraction) -> Fraction | None:
    below = [r for r in RATE_LADDER if r <= q]
    return below[-1] if below else None


# ── representation families (section 4 Route A): materially different CLASSES ───────────
# Two families are "materially different" iff their `class` differs. A scalar quantizer at
# a different group size is the SAME class and does NOT count as a second family.
REPRESENTATION_FAMILIES: dict[str, dict[str, Any]] = {
    "scalar_trellis_tqv2":          {"class": "scalar_trellis",        "impl": "executable",         "deployable": True,  "floor_eff_bpw": 1.34},
    "vtq_vector_trellis":           {"class": "vector_codebook",       "impl": "oracle_only",        "deployable": False},
    "binary_latent_factors":        {"class": "binary_factor",         "impl": "designed_not_wired", "deployable": False},
    "multi_envelope_binary":        {"class": "binary_factor_menv",    "impl": "designed_not_wired", "deployable": False},
    "additive_codebooks":           {"class": "additive_codebook",     "impl": "designed_not_wired", "deployable": False},
    "vector_codebooks":             {"class": "vector_codebook",       "impl": "designed_not_wired", "deployable": False},
    "binary_pattern_codebooks":     {"class": "binary_pattern_codebook","impl": "designed_not_wired","deployable": False},
    "progressive_semantic_slices":  {"class": "progressive_slice",     "impl": "designed_not_wired", "deployable": False},
    "expert_genome":                {"class": "expert_genome",         "impl": "designed_not_wired", "deployable": False},
    "shared_parameter_grammar":     {"class": "parameter_grammar",     "impl": "designed_not_wired", "deployable": False},
    "subbit_mixed_precision_islands":{"class": "mixed_precision_islands","impl": "designed_not_wired","deployable": False},
    "structured_sparse_exceptions": {"class": "sparse_exception",      "impl": "designed_not_wired", "deployable": False},
    "repairability_shaped":         {"class": "repairability_shaped",  "impl": "designed_not_wired", "deployable": False},
}


def family_class(name: str) -> str:
    entry = REPRESENTATION_FAMILIES.get(name)
    if entry is None:
        raise GravityError(f"unknown representation family {name!r}")
    return entry["class"]


def materially_different(a: str, b: str) -> bool:
    """True iff a and b are different representation CLASSES (section 4)."""
    return family_class(a) != family_class(b)


def count_material_families(families: Iterable[str]) -> int:
    return len({family_class(f) for f in families})


# ── reachable Doctor treatments (section 12): only EXECUTABLE adapters are selectable ───
# Mirrors succ_doctor.IMPL_STATES / UNWIRED_TREATMENT_HOOKS and the readiness packet.
REACHABLE_TREATMENTS: dict[str, dict[str, Any]] = {
    "condense_control":   {"adapter_id": "doctor-v5-strand-ladder-qwen25-dense", "impl": "executable"},
    "doctor_static":      {"adapter_id": "doctor-v5-static-repair",              "impl": "executable"},
    "doctor_conditional": {"adapter_id": "doctor-v5-conditional-repair",         "impl": "executable_negative"},
    "doctor_full":        {"adapter_id": "doctor-v5-full-treatment",             "impl": "executable"},
}
# Treatments that exist only as registry entries or unwired hooks: NEVER selectable.
UNWIRED_TREATMENTS: tuple[str, ...] = (
    "lora_kd", "blockwise_qat", "strand_hessian",
    "gptoss_static", "gptoss_conditional", "gptoss_full", "gptoss_codec_control",
)
# Treatment preference when a degraded rung needs repair (strongest reachable first).
_TREATMENT_PREFERENCE = ("doctor_full", "doctor_conditional", "doctor_static", "condense_control")


def treatment_reachable(name: str, *, architecture_family: str = "qwen2.5-dense") -> bool:
    """Executable-adapter check. gpt-oss / giant families have no executable Doctor module
    yet, so their treatments are unreachable (registry-only), exactly like the packet."""
    if name in UNWIRED_TREATMENTS:
        return False
    if name not in REACHABLE_TREATMENTS:
        return False
    if architecture_family != "qwen2.5-dense":
        # only the qwen2.5-dense executor is wired today (packet: executable_qwen25_dense_only)
        return False
    return True


def select_treatment(reachable: Iterable[str], *, degraded: bool,
                     architecture_family: str = "qwen2.5-dense") -> str | None:
    """Never selects an unsupported treatment (section 12 / invariant test)."""
    if not degraded:
        return None
    reachable_set = {t for t in reachable if treatment_reachable(t, architecture_family=architecture_family)}
    for t in _TREATMENT_PREFERENCE:
        if t in reachable_set:
            return t
    return None


# ── total-rate conservation law (section 13) ───────────────────────────────────────────
_BYTE_COMPONENTS = (
    "base", "doctor", "passthrough", "metadata", "packaging",
    "codebooks", "exception_tables", "indices", "alignment", "runtime_tables",
    "lexical", "routers", "shared_experts",
)


def whole_artifact_bpw(components: dict[str, float]) -> float:
    """Sum of ALL mandatory physical components. Doctor (treatment) bytes are counted,
    never granted an unbilled reserve (section 13). Unknown keys are rejected."""
    total = 0.0
    for k, v in components.items():
        if k not in _BYTE_COMPONENTS:
            raise GravityError(f"unknown physical component {k!r} (conservation law is complete)")
        total += float(v)
    return total


def is_subbit_artifact(whole_bpw: float) -> bool:
    """A whole-artifact bpw < 1.0 is sub-bit. 0.50 base + 0.40 doctor + 0.15 overhead = 1.05
    is NOT sub-bit (section 13)."""
    return float(whole_bpw) < 1.0


# ── mandatory sub-bit coverage (section 4): Routes A / B / C / D ────────────────────────
def subbit_coverage_status(parent_state: dict[str, Any]) -> dict[str, Any]:
    """Return whether a parent has satisfied mandatory sub-bit coverage, and by which route.

    Route A: >=2 materially different representation families reached F1 or F2 evidence.
    Route B: a valid sealed structural-incompatibility receipt.
    Route C: a valid sealed physical-impossibility receipt.
    Route D: a calibrated proxy with a known false-negative bound PLUS a targeted
             parent-specific confirmation (scale interpolation alone is forbidden).
    """
    tested = parent_state.get("representation_families_tested", []) or []
    # Route A: material families that reached at least F1
    f1_families = [t["family"] for t in tested
                   if str(t.get("evidence_level", "")).upper() in ("F1", "F2", "F3", "F4")
                   and t.get("family") in REPRESENTATION_FAMILIES]
    if count_material_families(f1_families) >= 2:
        return {"satisfied": True, "route": "A",
                "detail": f"{count_material_families(f1_families)} materially different families reached F1+"}
    # Route B / C: sealed receipts
    receipts = parent_state.get("receipts", {}) or {}
    if receipts.get("structural_incompatibility"):
        return {"satisfied": True, "route": "B", "detail": "sealed structural-incompatibility receipt"}
    if receipts.get("physical_impossibility"):
        return {"satisfied": True, "route": "C", "detail": "sealed physical-impossibility receipt"}
    # Route D: calibrated proxy + targeted confirmation
    proxy = parent_state.get("proxy_confirmation") or {}
    if proxy.get("calibrated_false_negative_bound") is not None and proxy.get("targeted_confirmation"):
        return {"satisfied": True, "route": "D",
                "detail": "calibrated proxy with known FN bound plus targeted confirmation"}
    return {"satisfied": False, "route": None,
            "detail": "no route satisfied: need 2 material families at F1+, or a sealed "
                      "structural/physical receipt, or calibrated-proxy+confirmation"}


# ── EXTREME finalization gate (section 3 / 5): enforced by code ─────────────────────────
def can_finalize_extreme(*, whole_bpw: float, coverage: dict[str, Any],
                         escape_receipt: dict[str, Any] | None,
                         escape_receipt_valid: bool) -> tuple[bool, list[str]]:
    """No parent may be finalized with an EXTREME result above one physical BPW unless the
    mandatory sub-bit-coverage condition is satisfied AND a valid sealed Escape Receipt is
    bound. Sub-bit (or exactly 1.0) EXTREME results are not blocked by Gravity here; the
    quality contract is enforced separately and is unchanged.
    """
    reasons: list[str] = []
    if float(whole_bpw) <= 1.0:
        return True, reasons  # already inside the gravitational region; Gravity does not block
    # above 1.0 BPW: escape is required and must be justified
    if not coverage.get("satisfied"):
        reasons.append("above 1.0 BPW without mandatory sub-bit coverage (Route A/B/C/D)")
    if escape_receipt is None:
        reasons.append("above 1.0 BPW without a sealed Escape Receipt")
    elif not escape_receipt_valid:
        reasons.append("Escape Receipt present but invalid (seal or justification fails)")
    return (not reasons), reasons


# ── representation-family-before-BPW escalation (section 14) ────────────────────────────
ESCALATION_ORDER: tuple[str, ...] = (
    "same_rate_different_representation",
    "same_rate_different_byte_allocation",
    "same_rate_different_doctor_treatment",
    "slightly_higher_rate",
)


def next_escalation_step(current_step: str | None) -> str | None:
    if current_step is None:
        return ESCALATION_ORDER[0]
    if current_step not in ESCALATION_ORDER:
        raise GravityError(f"unknown escalation step {current_step!r}")
    i = ESCALATION_ORDER.index(current_step)
    return ESCALATION_ORDER[i + 1] if i + 1 < len(ESCALATION_ORDER) else None


def rate_closed(parent_state: dict[str, Any], rate: Fraction) -> tuple[bool, list[str]]:
    """A rate is closed only when representation coverage is complete, treatment
    reachability tested, physical accounting complete, causal failure understood, and
    uncertainty low enough (section 14). Otherwise the rate stays open for escalation."""
    rate = Fraction(rate)
    reasons: list[str] = []
    per_rate = (parent_state.get("per_rate") or {}).get(str(rate), {})
    if count_material_families(per_rate.get("families_tried", [])) < 2 and not per_rate.get("coverage_receipt"):
        reasons.append("representation-family coverage incomplete at this rate")
    if not per_rate.get("treatment_reachability_tested"):
        reasons.append("treatment reachability not tested")
    if not per_rate.get("physical_accounting_complete"):
        reasons.append("physical accounting incomplete")
    if not per_rate.get("causal_failure_understood"):
        reasons.append("causal failure not understood")
    if float(per_rate.get("uncertainty", 1.0)) > 0.25:
        reasons.append("uncertainty too high to close")
    return (not reasons), reasons


# ── default-off activation gate (section 2) ────────────────────────────────────────────
_ACTIVATION_GATES = (
    "schemas_validated", "release_boundary_signed", "successor_authorized",
    "resource_admission_passed", "source_program_executable",
)


def default_policy() -> dict[str, Any]:
    """The default Gravity policy: enabled False, every activation gate False. This is the
    content of GRAVITY_POLICY.json until the campaign reaches its signed release boundary."""
    policy = {
        "schema": GRAVITY_POLICY_SCHEMA,
        "policy_version": GRAVITY_POLICY_VERSION,
        "enabled": False,
        "created_at": now_iso(),
        "activation_gates": {g: False for g in _ACTIVATION_GATES},
        "invariant": INVARIANT,
        "env_flag": "HAWKING_GRAVITY_ENABLED",
    }
    return seal_field(policy, "policy_sha256")


def gravity_enabled(*, policy: dict[str, Any] | None = None, env: dict[str, str] | None = None) -> bool:
    """Gravity is active ONLY when the env flag is set AND the policy is enabled AND every
    activation gate holds. Default OFF (section 2). Any missing gate returns False."""
    env = env if env is not None else dict(os.environ)
    if str(env.get("HAWKING_GRAVITY_ENABLED", "")).lower() not in ("1", "true", "yes", "on"):
        return False
    if policy is None:
        return False
    if not policy.get("enabled"):
        return False
    gates = policy.get("activation_gates", {}) or {}
    return all(bool(gates.get(g)) for g in _ACTIVATION_GATES)


# ── policy manifest (the GRAVITY_POLICY.json deliverable, section 20) ───────────────────
def build_policy_manifest() -> dict[str, Any]:
    """A sealed, self-describing policy manifest. Default-off; priors, not constants."""
    manifest = {
        "schema": GRAVITY_POLICY_SCHEMA,
        "policy_version": GRAVITY_POLICY_VERSION,
        "generated_at": now_iso(),
        "doctrine": ("Gravity pulls every model toward the smallest complete physical "
                     "representation; Doctor fights to preserve the model during the fall; "
                     "Event Horizon marks the lowest point where capability survives; escape "
                     "above sub-bit is permitted only with a sealed Escape Receipt."),
        "invariant": INVARIANT,
        "enabled": False,
        "activation_gates": {g: False for g in _ACTIVATION_GATES},
        "env_flag": "HAWKING_GRAVITY_ENABLED",
        "rate_ladder": [rate_identity(r) for r in RATE_LADDER],
        "doctor_reserve_prior": rate_identity(DOCTOR_RESERVE_PRIOR),
        "envelope_gib": ENVELOPE_GIB,
        "parent_stress_starts": {
            label: {
                "stress_start": rate_identity(Fraction(p["subbit_stress_start"])),
                "subbit_is": p["subbit_is"], "n_params": p["n_params"],
                "resident_ceiling_bpw": p["resident_ceiling_bpw"],
                "first_likely_viable_bpw": p["first_likely_viable_bpw"],
                "representation_families": p["representation_families"], "risk": p["risk"],
            }
            for label, p in PARENT_PRIORS.items()
        },
        "representation_families": REPRESENTATION_FAMILIES,
        "reachable_treatments": REACHABLE_TREATMENTS,
        "unwired_treatments": list(UNWIRED_TREATMENTS),
        "coverage_routes": {
            "A": ">=2 materially different representation families at F1 or F2",
            "B": "sealed structural-incompatibility receipt",
            "C": "sealed physical-impossibility receipt",
            "D": "calibrated proxy with known FN bound plus targeted confirmation",
        },
        "conservation_law": {
            "components": list(_BYTE_COMPONENTS),
            "rule": "whole = sum(all components incl. doctor); sub-bit iff whole < 1.0",
        },
        "escalation_order": list(ESCALATION_ORDER),
        "scientific_ground": {
            "packet": "reports/condense/subbit_frontier/SUBBIT_READINESS_PACKET.json",
            "packet_seal_sha256": "f6c6b2d8cd046827add88202681fae9fbc30383831947ad1e4caef4490d38bda",
            "policy_proof": "reports/condense/subbit_frontier/subbit_inverted_search_sim.py",
        },
    }
    return seal_field(manifest, "policy_sha256")


def selftest() -> dict[str, Any]:
    # ladder identity is exact
    assert Fraction(1, 10) + Fraction(1, 5) == Fraction(3, 10)
    assert parse_rate("1/2") == Fraction(1, 2)
    try:
        parse_rate("0.33"); raise AssertionError("rounded decimal must be rejected as identity")
    except GravityError:
        pass
    # stress starts resolve and are sub-bit for every parent
    for label in PARENT_PRIORS:
        ss = compute_stress_start(label)
        assert is_subbit(rate_from_identity(ss["chosen_stress_rate"])), label
    # conservation: the worked non-example is not sub-bit
    whole = whole_artifact_bpw({"base": 0.50, "doctor": 0.40, "packaging": 0.15})
    assert abs(whole - 1.05) < 1e-9 and not is_subbit_artifact(whole)
    # EXTREME gate refuses >1 BPW without coverage+receipt
    ok, why = can_finalize_extreme(whole_bpw=1.5, coverage={"satisfied": False},
                                   escape_receipt=None, escape_receipt_valid=False)
    assert not ok and why
    ok2, _ = can_finalize_extreme(whole_bpw=0.8, coverage={"satisfied": False},
                                  escape_receipt=None, escape_receipt_valid=False)
    assert ok2  # sub-bit finalization not blocked by Gravity
    # default-off
    assert gravity_enabled(policy=default_policy(), env={"HAWKING_GRAVITY_ENABLED": "1"}) is False
    # materially different classes
    assert not materially_different("scalar_trellis_tqv2", "scalar_trellis_tqv2")
    assert materially_different("binary_latent_factors", "additive_codebooks")
    # unsupported treatment never selected
    assert select_treatment(["lora_kd", "strand_hessian"], degraded=True) is None
    assert select_treatment(["doctor_static"], degraded=True) == "doctor_static"
    m = build_policy_manifest()
    assert sealed(m, "policy_sha256")
    return {"ok": True, "ladder_len": len(RATE_LADDER), "parents": list(PARENT_PRIORS),
            "policy_version": GRAVITY_POLICY_VERSION}


if __name__ == "__main__":
    import json
    print(json.dumps(selftest(), indent=2, sort_keys=True))
