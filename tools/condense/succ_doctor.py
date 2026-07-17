#!/usr/bin/env python3.12
"""Typed Doctor: mechanism registry, joint base/healing allocator, causal-proof frame.

Master goal 8. The Doctor is modeled here as a typed, evidence-driven pathology /
treatment / proof system, NOT a LoRA wrapper. It answers three questions with
mechanisms whose wiring state is honest:

  1. What is wrong (diagnosis, section 8.3)? The vocabulary and thresholds are reused
     from `eco_planner.diagnose` so a diagnosis means the same thing here as in the
     adaptive planner.
  2. What can act on it (mechanism registry, section 8.4)? Every mechanism is a typed
     row binding its operator kind, provenance, and -- crucially -- its
     implementation_state. Only an `executable` mechanism is selectable for execution.
     The treatment hooks lora_kd / blockwise_qat / strand_hessian are marked
     `blocked_missing_adapter` because the qwen2.5-dense executor reports them
     unsupported (only method=none is wired), so they CANNOT be selected for execution.
  3. How are bytes spent and how is a claim proven (joint allocator section 8.7,
     repairability curve section 8.8, causal control set section 8.10)?

This module is additive, default-off scaffolding. It launches no compute, imports no
gitignored campaign runtime, adopts no live pid, and writes only under the successor
namespace `reports/condense/event_horizon_successor/`. Its selftest is fully offline.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
from itertools import product
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError, seal_field, sealed, now_iso, atomic_write_json, repo_root,
)
# Reuse the exact diagnosis semantics from the adaptive planner (section 8.3).
from eco_planner import diagnose, DIAGNOSES, CapabilityContract  # noqa: E402

# -- schema registry ---------------------------------------------------------------------
SCHEMA_MECHANISM = "hawking.successor.mechanism.v1"
SCHEMA_MECHANISM_REGISTRY = "hawking.successor.mechanism_registry.v1"
SCHEMA_ALLOCATION = "hawking.successor.byte_allocation.v1"
SCHEMA_REPAIRABILITY = "hawking.successor.repairability.v1"
SCHEMA_CONTROL_SET = "hawking.successor.control_set.v1"

# -- typed vocabularies ------------------------------------------------------------------
# implementation_state (section 8.4). selectable-for-execution == executable, enforced.
IMPL_STATES: tuple[str, ...] = ("executable", "designed_not_wired", "blocked_missing_adapter")

OPERATOR_KINDS: tuple[str, ...] = (
    "control", "codec", "treatment", "gate", "allocator", "firewall", "proof", "ledger",
)

# claim lanes a mechanism may contribute evidence to (Pass B seals physical only).
CLAIM_LANES: tuple[str, ...] = ("physical_bytes", "provisional_quality", "sealed_quality")

# proof_ceiling: the strongest claim a mechanism can currently substantiate, given state.
PROOF_CEILINGS: tuple[str, ...] = (
    "none_unwired", "design_only", "physical_bytes_sealed", "provisional_quality",
    "sealed_quality",
)

# the failure routes (diagnoses) a mechanism is designed to treat; drawn from DIAGNOSES.
FAILURE_ROUTES: tuple[str, ...] = tuple(DIAGNOSES)

# The three treatment hooks the dense executor reports unsupported (only method=none is
# wired). They are pinned blocked_missing_adapter and MUST NOT be selectable.
UNWIRED_TREATMENT_HOOKS: tuple[str, ...] = ("lora_kd", "blockwise_qat", "strand_hessian")

# The dense qwen2.5 executors that are actually wired. Only method=none and the codec-time
# precision island run through them without a training adapter.
DENSE_NONE_EXECUTOR = "qwen2.5_dense_none_v5"
DENSE_CODEC_EXECUTOR = "qwen2.5_dense_codec_v5"


class DoctorError(EcoError):
    """Fail-closed error in the typed Doctor."""


# -- the causal control set (section 8.10) ----------------------------------------------
# A treatment gets credit only for the effect it produces BEYOND every control below. The
# order is the canonical evaluation order (cheapest / most-null first).
_CONTROL_SET: tuple[dict[str, str], ...] = (
    {"id": "treatment_on_off",
     "purpose": "same recipe, treatment enabled vs disabled; the raw treatment delta"},
    {"id": "zero_treatment",
     "purpose": "method=none baseline (the codec with no healing); the null mechanism"},
    {"id": "equal_byte_random",
     "purpose": "spend the treatment's bytes on random init; separates bytes from method"},
    {"id": "stronger_equal_byte_base",
     "purpose": "spend the same bytes on a higher-bit base instead of healing"},
    {"id": "bf16_parent_same_data",
     "purpose": "the bf16 parent on the identical eval data; the loss-free ceiling"},
    {"id": "parent_vs_stronger_teacher",
     "purpose": "parent teacher vs a stronger teacher; isolates KD-signal source"},
    {"id": "forgetting_monitor",
     "purpose": "off-target regression probe; a treatment must not trade capabilities"},
    {"id": "exact_byte_accounting",
     "purpose": "every byte the treatment adds is counted against its own budget"},
    {"id": "multi_seed",
     "purpose": "repeat across seeds; the effect must survive seed variance"},
    {"id": "sealed_split",
     "purpose": "train/heal and eval on a sealed split; no leakage into the claim"},
    {"id": "packed_parity",
     "purpose": "the packed artifact reproduces the healed logits bit-for-bit"},
)
CONTROL_IDS: tuple[str, ...] = tuple(c["id"] for c in _CONTROL_SET)


def causal_control_set(treatment: str | None = None) -> dict[str, Any]:
    """The required control list for a treatment (section 8.10).

    Returns the FULL causal control set. A treatment is credited only for the effect it
    produces beyond every one of these controls. `treatment` is recorded for provenance
    and used to flag which controls are load-bearing for that treatment (teacher controls
    matter for KD-style hooks), but none are dropped: a missing control is a missing proof.
    """
    kd_like = bool(treatment) and (
        "kd" in treatment or "teacher" in treatment or "distill" in treatment
    )
    controls = []
    for c in _CONTROL_SET:
        load_bearing = True
        if c["id"] == "parent_vs_stronger_teacher":
            load_bearing = kd_like
        controls.append({**c, "load_bearing_for_treatment": load_bearing})
    out = {
        "schema": SCHEMA_CONTROL_SET,
        "treatment": treatment,
        "controls": controls,
        "control_ids": list(CONTROL_IDS),
        "credit_rule": "a treatment gets credit only for effect beyond every control",
        "generated_at": now_iso(),
    }
    return seal_field(out, "control_set_sha256")


# -- the typed mechanism (section 8.4) --------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Mechanism:
    """One typed row of the mechanism registry.

    A mechanism is a candidate operator the Doctor may bind. Its `implementation_state`
    is the honesty gate: only an `executable` mechanism is selectable for execution.
    """
    mechanism_id: str
    version: str
    operator_kind: str
    provenance: str
    implementation_state: str
    source_sha256: str | None                       # source hash, or None if unwired
    supported_architectures: tuple[str, ...]
    supported_claim_lanes: tuple[str, ...]
    supported_failure_routes: tuple[str, ...]
    min_physical_bpw: float
    max_physical_bpw: float
    inference_byte_model: str                        # how inference bytes are accounted
    resident_byte_model: str                         # how resident weight bytes are accounted
    moved_byte_model: str                            # how per-token moved bytes are accounted
    training_compute_model: str                      # FLOP / pass model, or "none"
    executor_adapter_id: str | None                  # wired executor, or None
    native_runtime_requirements: tuple[str, ...]
    contraindications: tuple[str, ...]
    required_controls: tuple[str, ...]
    proof_ceiling: str

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["schema"] = SCHEMA_MECHANISM
        d["selectable_for_execution"] = self.selectable_for_execution()
        return d

    def selectable_for_execution(self) -> bool:
        """A mechanism is selectable for execution IFF it is executable (section 8.4:
        'An unwired mechanism cannot be selected for execution')."""
        return self.implementation_state == "executable"


def _proof_ceiling_for(state: str) -> str:
    """The strongest claim a mechanism in `state` may substantiate. Executable mechanisms
    seal physical bytes (Pass B forbids sealing quality); everything else cannot seal."""
    if state == "executable":
        return "physical_bytes_sealed"
    if state == "blocked_missing_adapter":
        return "none_unwired"
    return "design_only"  # designed_not_wired


def _mech(**kw: Any) -> Mechanism:
    """Row constructor with byte-accounting and proof-ceiling defaults derived from state,
    so the registry cannot silently claim more proof than its wiring allows."""
    state = kw["implementation_state"]
    kw.setdefault("source_sha256", None)
    kw.setdefault("inference_byte_model", "packed_all_in_payload")
    kw.setdefault("resident_byte_model", "weights_resident_at_bpw")
    kw.setdefault("moved_byte_model", "weights_streamed_per_token")
    kw.setdefault("training_compute_model", "none")
    kw.setdefault("executor_adapter_id", None)
    kw.setdefault("native_runtime_requirements", ())
    kw.setdefault("contraindications", ())
    kw.setdefault("required_controls", CONTROL_IDS)
    kw.setdefault("proof_ceiling", _proof_ceiling_for(state))
    return Mechanism(**kw)


def _build_registry() -> tuple[Mechanism, ...]:
    """The typed mechanism registry (section 8.4 + the 13 Hawking-native lineages 8.9).

    Honesty pins:
      - executable  : zero_treatment, protected_precision_island (codec-time, no adapter).
      - blocked_missing_adapter : lora_kd, blockwise_qat, strand_hessian, low_rank_residual,
        hawking_correction_field -- all need a training/fit adapter the dense executor does
        not wire, so none may be selected for execution.
      - designed_not_wired : every other Hawking-native lineage; specified, not yet wired.
    """
    dense = ("qwen2.5-dense",)
    moe = ("qwen2.5-moe",)
    both = ("qwen2.5-dense", "qwen2.5-moe")
    rows: list[Mechanism] = []

    # ---- simple controls -----------------------------------------------------------------
    rows.append(_mech(
        mechanism_id="zero_treatment", version="v1", operator_kind="control",
        provenance="doctor-native/control", implementation_state="executable",
        supported_architectures=both, supported_claim_lanes=("physical_bytes",),
        supported_failure_routes=(), min_physical_bpw=0.1, max_physical_bpw=8.0,
        training_compute_model="none", executor_adapter_id=DENSE_NONE_EXECUTOR,
        contraindications=(), required_controls=("exact_byte_accounting", "packed_parity"),
    ))
    rows.append(_mech(
        mechanism_id="protected_precision_island", version="v1", operator_kind="codec",
        provenance="doctor-native/codec", implementation_state="executable",
        supported_architectures=both, supported_claim_lanes=("physical_bytes",),
        supported_failure_routes=("signal_degradation",), min_physical_bpw=0.5,
        max_physical_bpw=8.0, training_compute_model="none",
        executor_adapter_id=DENSE_CODEC_EXECUTOR,
        native_runtime_requirements=("mixed_precision_pack",),
        contraindications=("island_exceeds_byte_ceiling",),
        required_controls=("stronger_equal_byte_base", "exact_byte_accounting", "packed_parity"),
    ))
    rows.append(_mech(
        mechanism_id="bias_correction", version="v1", operator_kind="treatment",
        provenance="doctor-native/control", implementation_state="designed_not_wired",
        supported_architectures=dense, supported_claim_lanes=("provisional_quality",),
        supported_failure_routes=("signal_degradation",), min_physical_bpw=1.0,
        max_physical_bpw=6.0, training_compute_model="single_calibration_pass",
        native_runtime_requirements=("calibration_activation_capture",),
        contraindications=("no_calibration_corpus",),
    ))
    rows.append(_mech(
        mechanism_id="low_rank_residual", version="v1", operator_kind="treatment",
        provenance="doctor-native/treatment", implementation_state="blocked_missing_adapter",
        supported_architectures=dense, supported_claim_lanes=("provisional_quality",),
        supported_failure_routes=("signal_degradation", "mixed_failure"),
        min_physical_bpw=1.0, max_physical_bpw=6.0,
        training_compute_model="low_rank_fit_over_residual",
        native_runtime_requirements=("qwen2.5_dense_train_adapter",),
        contraindications=("dense_executor_reports_method_unsupported",),
    ))

    # ---- treatment execution hooks (unsupported by the dense executor) --------------------
    rows.append(_mech(
        mechanism_id="lora_kd", version="v1", operator_kind="treatment",
        provenance="doctor-v5/treatment", implementation_state="blocked_missing_adapter",
        supported_architectures=dense, supported_claim_lanes=("provisional_quality",),
        supported_failure_routes=("signal_degradation", "mixed_failure"),
        min_physical_bpw=1.0, max_physical_bpw=6.0,
        training_compute_model="lora_kd_teacher_forced_forward_per_step",
        native_runtime_requirements=("qwen2.5_dense_train_adapter", "teacher_logits_stream"),
        contraindications=("dense_executor_reports_method_unsupported",),
    ))
    rows.append(_mech(
        mechanism_id="blockwise_qat", version="v1", operator_kind="treatment",
        provenance="doctor-v5/treatment", implementation_state="blocked_missing_adapter",
        supported_architectures=dense, supported_claim_lanes=("provisional_quality",),
        supported_failure_routes=("signal_degradation", "mixed_failure", "computation_collapse"),
        min_physical_bpw=0.5, max_physical_bpw=4.0,
        training_compute_model="blockwise_qat_forward_backward_per_block",
        native_runtime_requirements=("qwen2.5_dense_train_adapter", "straight_through_estimator"),
        contraindications=("dense_executor_reports_method_unsupported",),
    ))
    rows.append(_mech(
        mechanism_id="strand_hessian", version="v1", operator_kind="treatment",
        provenance="strand/treatment", implementation_state="blocked_missing_adapter",
        supported_architectures=dense, supported_claim_lanes=("provisional_quality",),
        supported_failure_routes=("signal_degradation", "mixed_failure"),
        min_physical_bpw=0.8, max_physical_bpw=4.0,
        training_compute_model="strand_hessian_second_order_per_layer",
        native_runtime_requirements=("qwen2.5_dense_train_adapter", "layer_hessian_estimate"),
        contraindications=("dense_executor_reports_method_unsupported",),
    ))

    # ---- the 13 Hawking-native lineages (section 8.9) ------------------------------------
    rows.append(_mech(
        mechanism_id="event_horizon_waterfilling", version="v0", operator_kind="allocator",
        provenance="hawking-native/8.9", implementation_state="designed_not_wired",
        supported_architectures=both, supported_claim_lanes=("physical_bytes", "provisional_quality"),
        supported_failure_routes=("signal_degradation",), min_physical_bpw=0.5, max_physical_bpw=6.0,
        training_compute_model="none",
        native_runtime_requirements=("per_group_sensitivity", "joint_byte_allocator"),
        contraindications=("sensitivity_estimate_missing",),
    ))
    rows.append(_mech(
        mechanism_id="repairability_shaped_condensation", version="v0", operator_kind="codec",
        provenance="hawking-native/8.9", implementation_state="designed_not_wired",
        supported_architectures=both, supported_claim_lanes=("physical_bytes",),
        supported_failure_routes=("signal_degradation",), min_physical_bpw=0.5, max_physical_bpw=6.0,
        native_runtime_requirements=("repairability_curve", "joint_base_heal_budget"),
        contraindications=("heal_budget_unbounded",),
    ))
    rows.append(_mech(
        mechanism_id="causal_organ_preservation", version="v0", operator_kind="codec",
        provenance="hawking-native/8.9", implementation_state="designed_not_wired",
        supported_architectures=both, supported_claim_lanes=("physical_bytes", "provisional_quality"),
        supported_failure_routes=("mixed_failure",), min_physical_bpw=0.8, max_physical_bpw=6.0,
        native_runtime_requirements=("capability_attribution_map",),
        contraindications=("organ_map_unavailable",),
    ))
    rows.append(_mech(
        mechanism_id="synaptic_exception_lattice", version="v0", operator_kind="codec",
        provenance="hawking-native/8.9", implementation_state="designed_not_wired",
        supported_architectures=dense, supported_claim_lanes=("physical_bytes",),
        supported_failure_routes=("signal_degradation",), min_physical_bpw=0.5, max_physical_bpw=4.0,
        native_runtime_requirements=("outlier_exception_index",),
        contraindications=("exception_index_exceeds_byte_ceiling",),
    ))
    rows.append(_mech(
        mechanism_id="hawking_correction_field", version="v0", operator_kind="treatment",
        provenance="hawking-native/8.9", implementation_state="blocked_missing_adapter",
        supported_architectures=dense, supported_claim_lanes=("provisional_quality",),
        supported_failure_routes=("signal_degradation", "mixed_failure"),
        min_physical_bpw=0.8, max_physical_bpw=4.0,
        training_compute_model="correction_field_fit_per_layer",
        native_runtime_requirements=("qwen2.5_dense_train_adapter", "quant_error_field_estimate"),
        contraindications=("dense_executor_reports_method_unsupported",),
    ))
    rows.append(_mech(
        mechanism_id="capability_immune_bank", version="v0", operator_kind="gate",
        provenance="hawking-native/8.9", implementation_state="designed_not_wired",
        supported_architectures=both, supported_claim_lanes=("provisional_quality",),
        supported_failure_routes=("mixed_failure",), min_physical_bpw=0.5, max_physical_bpw=6.0,
        native_runtime_requirements=("capability_regression_bank", "forgetting_monitor"),
        contraindications=("regression_bank_unpopulated",),
    ))
    rows.append(_mech(
        mechanism_id="quant_error_syndrome_gate", version="v0", operator_kind="gate",
        provenance="hawking-native/8.9", implementation_state="designed_not_wired",
        supported_architectures=both, supported_claim_lanes=("physical_bytes",),
        supported_failure_routes=("signal_degradation", "computation_collapse"),
        min_physical_bpw=0.5, max_physical_bpw=6.0,
        native_runtime_requirements=("per_block_error_syndrome",),
        contraindications=("syndrome_threshold_uncalibrated",),
    ))
    rows.append(_mech(
        mechanism_id="progressive_eh_slices", version="v0", operator_kind="codec",
        provenance="hawking-native/8.9", implementation_state="designed_not_wired",
        supported_architectures=both, supported_claim_lanes=("physical_bytes",),
        supported_failure_routes=("signal_degradation",), min_physical_bpw=0.25, max_physical_bpw=4.0,
        native_runtime_requirements=("progressive_slice_schedule",),
        contraindications=("slice_boundary_unbracketed",),
    ))
    rows.append(_mech(
        mechanism_id="expert_genome_codec", version="v0", operator_kind="codec",
        provenance="hawking-native/8.9", implementation_state="designed_not_wired",
        supported_architectures=moe, supported_claim_lanes=("physical_bytes",),
        supported_failure_routes=("signal_degradation",), min_physical_bpw=0.5, max_physical_bpw=4.0,
        native_runtime_requirements=("expert_routing_stats", "shared_expert_genome"),
        contraindications=("architecture_is_dense",),
    ))
    rows.append(_mech(
        mechanism_id="lexical_ark", version="v0", operator_kind="codec",
        provenance="hawking-native/8.9", implementation_state="designed_not_wired",
        supported_architectures=both, supported_claim_lanes=("physical_bytes", "provisional_quality"),
        supported_failure_routes=("mixed_failure",), min_physical_bpw=0.5, max_physical_bpw=6.0,
        native_runtime_requirements=("token_frequency_prior", "protected_lexeme_set"),
        contraindications=("vocab_prior_unavailable",),
    ))
    rows.append(_mech(
        mechanism_id="error_propagation_firewall", version="v0", operator_kind="firewall",
        provenance="hawking-native/8.9", implementation_state="designed_not_wired",
        supported_architectures=both, supported_claim_lanes=("physical_bytes",),
        supported_failure_routes=("computation_collapse", "mixed_failure"),
        min_physical_bpw=0.5, max_physical_bpw=6.0,
        native_runtime_requirements=("layerwise_error_budget",),
        contraindications=("error_budget_unset",),
    ))
    rows.append(_mech(
        mechanism_id="treatment_transplant_test", version="v0", operator_kind="proof",
        provenance="hawking-native/8.9", implementation_state="designed_not_wired",
        supported_architectures=both, supported_claim_lanes=("sealed_quality",),
        supported_failure_routes=FAILURE_ROUTES, min_physical_bpw=0.1, max_physical_bpw=8.0,
        native_runtime_requirements=("donor_recipient_swap_harness", "sealed_split"),
        contraindications=("no_donor_recipient_pair",),
    ))
    rows.append(_mech(
        mechanism_id="doctor_autopsy_ledger", version="v0", operator_kind="ledger",
        provenance="hawking-native/8.9", implementation_state="designed_not_wired",
        supported_architectures=both, supported_claim_lanes=("sealed_quality",),
        supported_failure_routes=FAILURE_ROUTES, min_physical_bpw=0.1, max_physical_bpw=8.0,
        native_runtime_requirements=("per_treatment_effect_ledger",),
        contraindications=("effect_ledger_unsealed",),
    ))
    return tuple(rows)


REGISTRY: tuple[Mechanism, ...] = _build_registry()


def registry() -> tuple[Mechanism, ...]:
    return REGISTRY


def selectable_mechanisms(reg: tuple[Mechanism, ...] | None = None) -> list[Mechanism]:
    reg = REGISTRY if reg is None else reg
    return [m for m in reg if m.selectable_for_execution()]


def registry_validate(reg: tuple[Mechanism, ...] | None = None) -> dict[str, Any]:
    """Enforce the registry invariants (section 8.4). Returns a sealed report.

    Hard invariant: selectable-for-execution == executable. A mechanism that is not
    `executable` MUST NOT be selectable, and the three treatment hooks
    (lora_kd / blockwise_qat / strand_hessian) MUST be blocked_missing_adapter and
    therefore not selectable.
    """
    reg = REGISTRY if reg is None else reg
    reasons: list[str] = []
    seen: set[str] = set()
    selectable: list[str] = []
    for m in reg:
        mid = m.mechanism_id
        if mid in seen:
            reasons.append(f"duplicate mechanism_id {mid}")
        seen.add(mid)
        if m.implementation_state not in IMPL_STATES:
            reasons.append(f"{mid}: bad implementation_state {m.implementation_state}")
        if m.operator_kind not in OPERATOR_KINDS:
            reasons.append(f"{mid}: bad operator_kind {m.operator_kind}")
        if m.proof_ceiling not in PROOF_CEILINGS:
            reasons.append(f"{mid}: bad proof_ceiling {m.proof_ceiling}")
        for lane in m.supported_claim_lanes:
            if lane not in CLAIM_LANES:
                reasons.append(f"{mid}: bad claim lane {lane}")
        for route in m.supported_failure_routes:
            if route not in FAILURE_ROUTES:
                reasons.append(f"{mid}: bad failure route {route}")
        for cid in m.required_controls:
            if cid not in CONTROL_IDS:
                reasons.append(f"{mid}: unknown required control {cid}")
        if not (0.0 < m.min_physical_bpw <= m.max_physical_bpw):
            reasons.append(f"{mid}: bad bpw range {m.min_physical_bpw}..{m.max_physical_bpw}")
        # the core honesty invariant: selectable == executable
        is_sel = m.selectable_for_execution()
        if is_sel != (m.implementation_state == "executable"):
            reasons.append(f"{mid}: selectable/executable mismatch")
        if is_sel:
            selectable.append(mid)
            if m.executor_adapter_id is None:
                reasons.append(f"{mid}: executable but no executor_adapter_id")
            if m.proof_ceiling == "none_unwired":
                reasons.append(f"{mid}: executable but proof_ceiling none_unwired")
        else:
            if m.executor_adapter_id is not None:
                reasons.append(f"{mid}: non-executable but has executor_adapter_id")
            if m.proof_ceiling == "physical_bytes_sealed":
                reasons.append(f"{mid}: non-executable but claims physical_bytes_sealed")
        # proof_ceiling must match the state's honest ceiling
        if m.proof_ceiling != _proof_ceiling_for(m.implementation_state):
            reasons.append(f"{mid}: proof_ceiling {m.proof_ceiling} inconsistent with state")

    # the three hooks must be present, blocked, and not selectable
    by_id = {m.mechanism_id: m for m in reg}
    for hook in UNWIRED_TREATMENT_HOOKS:
        m = by_id.get(hook)
        if m is None:
            reasons.append(f"missing required hook row {hook}")
            continue
        if m.implementation_state != "blocked_missing_adapter":
            reasons.append(f"{hook}: must be blocked_missing_adapter, is {m.implementation_state}")
        if m.selectable_for_execution():
            reasons.append(f"{hook}: is selectable for execution (must not be)")

    report = {
        "schema": SCHEMA_MECHANISM_REGISTRY,
        "ok": not reasons,
        "reasons": reasons,
        "mechanism_count": len(reg),
        "selectable_for_execution": sorted(selectable),
        "blocked_hooks_not_selectable": all(
            (by_id.get(h) is not None and not by_id[h].selectable_for_execution())
            for h in UNWIRED_TREATMENT_HOOKS
        ),
        "generated_at": now_iso(),
    }
    return seal_field(report, "registry_sha256")


def select_for_execution(diagnosis: str, architecture: str, claim_lane: str = "physical_bytes",
                         reg: tuple[Mechanism, ...] | None = None) -> list[Mechanism]:
    """Return the executable mechanisms that can act on `diagnosis` for `architecture`.

    Fails closed: it can NEVER return an unwired mechanism. A caller cannot select a
    blocked_missing_adapter or designed_not_wired mechanism for execution through here.
    """
    reg = REGISTRY if reg is None else reg
    if diagnosis not in FAILURE_ROUTES:
        raise DoctorError(f"unknown diagnosis {diagnosis}")
    out = []
    for m in reg:
        if not m.selectable_for_execution():
            continue
        if architecture not in m.supported_architectures:
            continue
        if claim_lane not in m.supported_claim_lanes:
            continue
        # the null control (zero_treatment) treats no route but is always admissible.
        if m.supported_failure_routes and diagnosis not in m.supported_failure_routes:
            if m.operator_kind != "control":
                continue
        out.append(m)
    # defensive post-check: nothing unwired slipped through.
    for m in out:
        if not m.selectable_for_execution():
            raise DoctorError(f"invariant breach: unwired {m.mechanism_id} selected")
    return out


# -- joint base/healing byte allocator (section 8.7) ------------------------------------
def allocate_bytes(groups: list[dict[str, Any]], budget_bytes: int, *,
                   quantum_bytes: int = 1) -> dict[str, Any]:
    """Solve a multiple-choice knapsack by DP over quantized byte units (section 8.7).

    Each semantic group `g` offers a set of mutually exclusive actions; the Doctor must
    pick exactly one action per group. Each action carries a byte cost `byte_cost` and a
    predicted protected-capability contribution `protected_quality`.

    Objective: maximize the predicted MIN protected-capability quality across groups
    (protect the weakest capability) subject to total bytes <= budget_bytes. This is a
    real max-min multiple-choice knapsack, solved exactly by a forward DP over byte units:
      dp[u] = the best achievable min-quality using u byte units across processed groups.
    The min-quality-subject-to-budget objective couples the groups (you cannot optimize a
    group in isolation), so the DP is doing real joint work.

    Returns the chosen action per group, the total bytes, the achieved min-quality, and a
    regret note (gap to the unconstrained max-min, i.e. what an infinite budget would buy).
    """
    if quantum_bytes <= 0:
        raise DoctorError("quantum_bytes must be positive")
    if budget_bytes < 0:
        raise DoctorError("budget_bytes must be non-negative")
    if not groups:
        raise DoctorError("no groups to allocate")

    # normalize + validate
    norm: list[dict[str, Any]] = []
    for g in groups:
        actions = g.get("actions") or []
        if not actions:
            raise DoctorError(f"group {g.get('group')} has no actions")
        na = []
        for a in actions:
            cost = a.get("byte_cost")
            q = a.get("protected_quality")
            if not isinstance(cost, int) or cost < 0:
                raise DoctorError(f"action {a.get('action')} bad byte_cost")
            if not isinstance(q, (int, float)):
                raise DoctorError(f"action {a.get('action')} bad protected_quality")
            na.append({"action": a.get("action"), "byte_cost": cost,
                       "protected_quality": float(q),
                       "cost_units": math.ceil(cost / quantum_bytes)})
        norm.append({"group": g.get("group"), "actions": na})

    budget_units = budget_bytes // quantum_bytes
    G = len(norm)
    NEG = float("-inf")
    # dp[u] = best min-quality reachable using u byte-units (None if unreachable).
    dp: list[float | None] = [None] * (budget_units + 1)
    dp[0] = float("inf")  # min over the empty set; first group overwrites it
    parents: list[list[tuple[int, int] | None]] = []
    for g in norm:
        ndp: list[float | None] = [None] * (budget_units + 1)
        par: list[tuple[int, int] | None] = [None] * (budget_units + 1)
        for u in range(budget_units + 1):
            base = dp[u]
            if base is None:
                continue
            for ai, a in enumerate(g["actions"]):
                nu = u + a["cost_units"]
                if nu > budget_units:
                    continue
                cand = base if base < a["protected_quality"] else a["protected_quality"]
                if ndp[nu] is None or cand > ndp[nu]:
                    ndp[nu] = cand
                    par[nu] = (u, ai)
        dp = ndp
        parents.append(par)

    # best reachable state within budget (prefer higher min-quality, then fewer bytes)
    best_u: int | None = None
    best_q: float = NEG
    for u in range(budget_units + 1):
        if dp[u] is None:
            continue
        if dp[u] > best_q or (dp[u] == best_q and (best_u is None or u < best_u)):
            best_q = dp[u]
            best_u = u

    if best_u is None:
        # even the cheapest action per group does not fit the budget
        cheapest = sum(min(a["cost_units"] for a in g["actions"]) for g in norm)
        return {
            "schema": SCHEMA_ALLOCATION, "feasible": False,
            "budget_bytes": budget_bytes, "quantum_bytes": quantum_bytes,
            "min_feasible_units": cheapest,
            "solver_note": "infeasible: cheapest one-per-group assignment exceeds the budget",
            "generated_at": now_iso(),
        }

    # reconstruct the per-group choices
    choices: list[int] = [0] * G
    u = best_u
    for gi in range(G - 1, -1, -1):
        pu, ai = parents[gi][u]  # type: ignore[misc]
        choices[gi] = ai
        u = pu

    chosen = []
    total_bytes = 0
    for gi, g in enumerate(norm):
        a = g["actions"][choices[gi]]
        total_bytes += a["byte_cost"]
        chosen.append({"group": g["group"], "action": a["action"],
                       "byte_cost": a["byte_cost"], "protected_quality": a["protected_quality"]})

    # unconstrained max-min (infinite budget): each group picks its highest-quality action.
    ideal_min_q = min(max(a["protected_quality"] for a in g["actions"]) for g in norm)
    regret = round(ideal_min_q - best_q, 12)

    return {
        "schema": SCHEMA_ALLOCATION,
        "feasible": True,
        "objective": "max_min_protected_quality_subject_to_byte_budget",
        "budget_bytes": budget_bytes,
        "quantum_bytes": quantum_bytes,
        "chosen": chosen,
        "total_bytes": total_bytes,
        "total_units": best_u,
        "achieved_min_quality": best_q,
        "unconstrained_min_quality": ideal_min_q,
        "regret_vs_infinite_budget": regret,
        "solver_note": ("exact max-min multiple-choice knapsack via forward DP over "
                        f"{budget_units + 1} byte-unit states; regret is the min-quality "
                        "the budget could not buy versus an infinite budget"),
        "generated_at": now_iso(),
    }


def _brute_force_allocate(groups: list[dict[str, Any]], budget_bytes: int,
                          quantum_bytes: int = 1) -> tuple[float | None, int | None]:
    """Reference max-min solver by exhaustive enumeration (selftest oracle)."""
    budget_units = budget_bytes // quantum_bytes
    best_q: float | None = None
    best_units: int | None = None
    option_lists = [
        [(math.ceil(a["byte_cost"] / quantum_bytes), float(a["protected_quality"]))
         for a in g["actions"]]
        for g in groups
    ]
    for combo in product(*option_lists):
        units = sum(c for c, _ in combo)
        if units > budget_units:
            continue
        mq = min(q for _, q in combo)
        if best_q is None or mq > best_q or (mq == best_q and units < (best_units or 0)):
            best_q = mq
            best_units = units
    return best_q, best_units


# -- repairability curve H_P(r) (section 8.8) -------------------------------------------
def repairability_curve(points: list[dict[str, Any]], *, fixed_bytes: float = 0.0) -> dict[str, Any]:
    """Total-cost repairability curve (section 8.8).

    Each point pins a base representation and its minimum healing cost at that base:
      base_bytes  = r          (bytes spent on the condensed base)
      min_heal_bytes = H_P(r)  (minimum bytes to heal that base back into contract)
    The total is T_P(r) = r + H_P(r) + fixed. Lowering the base `r` is only a win while it
    lowers the total; below some base the healing cost H_P(r) explodes and the total rises
    again. This routine returns the curve sorted by ascending base and flags every crossover
    where lowering the base RAISED the total, plus the minimum-total base.
    """
    if not points:
        raise DoctorError("no repairability points")
    curve = []
    for p in points:
        base = p.get("base_bytes")
        heal = p.get("min_heal_bytes")
        if not isinstance(base, (int, float)) or not isinstance(heal, (int, float)):
            raise DoctorError(f"bad repairability point {p}")
        total = float(base) + float(heal) + float(fixed_bytes)
        curve.append({"base_bytes": float(base), "min_heal_bytes": float(heal),
                      "total_bytes": total})
    curve.sort(key=lambda c: c["base_bytes"])

    # crossovers: adjacent ascending bases where the LOWER base has the HIGHER total.
    non_monotone = []
    for i in range(len(curve) - 1):
        lo, hi = curve[i], curve[i + 1]
        if lo["total_bytes"] > hi["total_bytes"]:
            non_monotone.append({
                "lower_base_bytes": lo["base_bytes"],
                "higher_base_bytes": hi["base_bytes"],
                "lower_base_total": lo["total_bytes"],
                "higher_base_total": hi["total_bytes"],
                "total_increase_from_lowering_base": round(lo["total_bytes"] - hi["total_bytes"], 12),
            })
    min_point = min(curve, key=lambda c: c["total_bytes"])
    out = {
        "schema": SCHEMA_REPAIRABILITY,
        "fixed_bytes": float(fixed_bytes),
        "curve": curve,
        "is_monotone_nonincreasing_toward_lower_base": not non_monotone,
        "non_monotone_crossovers": non_monotone,
        "min_total_point": min_point,
        "note": ("lowering the base helps only until healing cost dominates; below the "
                 "min-total base a lower base raises the total (the repairability floor)"),
        "generated_at": now_iso(),
    }
    return seal_field(out, "repairability_sha256")


# -- successor namespace (non-interference) ---------------------------------------------
def doctor_state_root() -> str:
    """Successor-only artifact namespace. Never under the campaign (doctor_v5_ultra)."""
    return str(repo_root() / "reports" / "condense" / "event_horizon_successor" / "doctor")


@dataclasses.dataclass(frozen=True)
class DoctorConfig:
    state_root: str
    default_quantum_bytes: int = 1
    dense_none_executor: str = DENSE_NONE_EXECUTOR
    dense_codec_executor: str = DENSE_CODEC_EXECUTOR
    contract: CapabilityContract = dataclasses.field(default_factory=CapabilityContract)

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["contract"] = self.contract.as_dict()
        return d


def default_config() -> DoctorConfig:
    return DoctorConfig(state_root=doctor_state_root())


def write_registry_snapshot(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Seal + atomically write a registry snapshot artifact (idiom demonstration)."""
    report = registry_validate()
    snapshot = {
        "schema": SCHEMA_MECHANISM_REGISTRY,
        "validation": report,
        "mechanisms": [m.as_dict() for m in REGISTRY],
        "control_set": causal_control_set(None),
        "generated_at": now_iso(),
    }
    snapshot = seal_field(snapshot, "snapshot_sha256")
    atomic_write_json(path, snapshot)
    return snapshot


# -- offline selftest -------------------------------------------------------------------
def selftest() -> dict[str, Any]:
    import tempfile
    from pathlib import Path

    # 1. registry validates and the three hooks are NOT selectable for execution.
    report = registry_validate()
    if not report["ok"]:
        raise DoctorError(f"registry invalid: {report['reasons']}")
    if not sealed(report, "registry_sha256"):
        raise DoctorError("registry report not sealed")
    if not report["blocked_hooks_not_selectable"]:
        raise DoctorError("blocked hooks are selectable")
    sel_ids = set(report["selectable_for_execution"])
    for hook in UNWIRED_TREATMENT_HOOKS:
        if hook in sel_ids:
            raise DoctorError(f"{hook} is selectable but must not be")
    if sel_ids != {"zero_treatment", "protected_precision_island"}:
        raise DoctorError(f"unexpected selectable set: {sel_ids}")
    # select_for_execution can never surface an unwired mechanism
    chosen = select_for_execution("signal_degradation", "qwen2.5-dense")
    if any(not m.selectable_for_execution() for m in chosen):
        raise DoctorError("select_for_execution surfaced an unwired mechanism")
    hook_refused = False
    try:
        select_for_execution("not_a_diagnosis", "qwen2.5-dense")
    except DoctorError:
        hook_refused = True
    if not hook_refused:
        raise DoctorError("unknown diagnosis not refused")

    # 2. allocator matches a brute-force optimum on a 3-group toy instance.
    groups = [
        {"group": "attention", "actions": [
            {"action": "island_2b", "byte_cost": 2, "protected_quality": 0.90},
            {"action": "flat_1b", "byte_cost": 1, "protected_quality": 0.50}]},
        {"group": "mlp", "actions": [
            {"action": "island_3b", "byte_cost": 3, "protected_quality": 0.80},
            {"action": "flat_1b", "byte_cost": 1, "protected_quality": 0.40}]},
        {"group": "embeddings", "actions": [
            {"action": "island_2b", "byte_cost": 2, "protected_quality": 0.95},
            {"action": "flat_1b", "byte_cost": 1, "protected_quality": 0.60}]},
    ]
    budget = 5
    alloc = allocate_bytes(groups, budget, quantum_bytes=1)
    if not alloc["feasible"]:
        raise DoctorError("toy instance should be feasible")
    bf_q, _bf_units = _brute_force_allocate(groups, budget, 1)
    if alloc["achieved_min_quality"] != bf_q:
        raise DoctorError(f"allocator {alloc['achieved_min_quality']} != brute force {bf_q}")
    if alloc["total_bytes"] > budget:
        raise DoctorError("allocator exceeded budget")
    # exhaustively confirm optimality across a sweep of budgets
    for b in range(0, 9):
        a = allocate_bytes(groups, b, quantum_bytes=1)
        bq, _ = _brute_force_allocate(groups, b, 1)
        if a["feasible"] != (bq is not None):
            raise DoctorError(f"feasibility mismatch at budget {b}")
        if a["feasible"] and a["achieved_min_quality"] != bq:
            raise DoctorError(f"allocator suboptimal at budget {b}: {a['achieved_min_quality']} != {bq}")

    # 3. repairability finds a non-monotone total (lowering base raises total).
    rep = repairability_curve([
        {"base_bytes": 100, "min_heal_bytes": 5},    # total 105
        {"base_bytes": 80, "min_heal_bytes": 10},    # total 90  (the floor)
        {"base_bytes": 60, "min_heal_bytes": 60},    # total 120 (lower base, higher total)
        {"base_bytes": 40, "min_heal_bytes": 100},   # total 140
    ])
    if rep["is_monotone_nonincreasing_toward_lower_base"]:
        raise DoctorError("repairability should be non-monotone")
    if not rep["non_monotone_crossovers"]:
        raise DoctorError("no crossover found")
    if rep["min_total_point"]["base_bytes"] != 80.0:
        raise DoctorError(f"min-total base wrong: {rep['min_total_point']}")

    # 4. causal_control_set returns the full control list.
    cs = causal_control_set("lora_kd")
    if not sealed(cs, "control_set_sha256"):
        raise DoctorError("control set not sealed")
    if [c["id"] for c in cs["controls"]] != list(CONTROL_IDS):
        raise DoctorError("control set incomplete")
    # KD-like treatment marks the teacher control load-bearing
    teacher = next(c for c in cs["controls"] if c["id"] == "parent_vs_stronger_teacher")
    if not teacher["load_bearing_for_treatment"]:
        raise DoctorError("teacher control should be load-bearing for lora_kd")

    # 5. diagnosis semantics are the planner's (section 8.3) and a sealed snapshot writes.
    c = CapabilityContract()
    if diagnose(0.02, -0.01, c) != "no_material_damage":
        raise DoctorError("diagnosis semantics drifted from eco_planner")
    if diagnose(2.0, -0.5, c) != "computation_collapse":
        raise DoctorError("collapse diagnosis wrong")
    with tempfile.TemporaryDirectory() as d:
        snap = write_registry_snapshot(Path(d) / "registry_snapshot.json")
        if not sealed(snap, "snapshot_sha256"):
            raise DoctorError("snapshot not sealed")

    return {
        "ok": True,
        "registry_valid": True,
        "mechanism_count": report["mechanism_count"],
        "selectable_for_execution": sorted(sel_ids),
        "blocked_hooks_not_selectable": True,
        "allocator_matches_brute_force": True,
        "allocator_min_quality": alloc["achieved_min_quality"],
        "repairability_non_monotone": True,
        "repairability_floor_base_bytes": rep["min_total_point"]["base_bytes"],
        "control_set_complete": True,
        "registry_sha256": report["registry_sha256"],
    }


if __name__ == "__main__":
    print(json.dumps(selftest(), indent=2, sort_keys=True))
