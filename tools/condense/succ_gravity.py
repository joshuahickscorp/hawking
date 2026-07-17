#!/usr/bin/env python3.12
"""Hawking Gravity engine: inverted sub-bit search, acquisition, state, source-binding.

This is the executable heart of the Gravity law, integrated into the existing successor
controller (it reuses eco_common sealing, the succ_events hash-chained log, the succ_queue /
succ_frontier rows, succ_engine.materialize_program, and succ_telegram). It launches
NOTHING heavy: every model action enters through the existing admission/queue/lease/
transition boundary and is refused while the campaign holds the heavy lease or while Gravity
is default-off.

What lives here (sections 8, 9, 10, 11, 14, 16, 21):
  - the exact-rational inverted search (ascend on collapse, stay on survivable signal,
    descend on pass, refine, stop with pass + evidenced lower boundary) with
    representation-family-before-BPW escalation and MIXED organ-aware handling;
  - the gravitational acquisition score A_Gravity(x) that changes scheduling PRIORITY only,
    never quality or evidence verdicts;
  - the persistent, hash-chained, resumable Gravity parent-state FSM;
  - additive Gravity augmentation of the existing queue / frontier rows;
  - the source-bound real-parent sub-bit program and its higher-rate fallback, both
    launch-gated so neither can execute outside the admission boundary.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, Callable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import EcoError, seal_field, sealed, hash_value, now_iso  # noqa: E402
import succ_gravity_policy as gp  # noqa: E402
import succ_gravity_receipts as gr  # noqa: E402

GRAVITY_PARENT_STATE_SCHEMA = "hawking.gravity.parent_state.v1"
GRAVITY_PROGRAM_SCHEMA = "hawking.gravity.program.v1"
GRAVITY_STATE_DOC_SCHEMA = "hawking.gravity.state.v1"
GRAVITY_VALIDATION_SCHEMA = "hawking.gravity.validation.v1"


class GravityEngineError(EcoError):
    """Fail-closed engine error."""


# ── signal / outcome vocabulary (sections 10, 11) ──────────────────────────────────────
class Signal(Enum):
    OK = "signal_ok"                    # survives without treatment
    DEGRADED = "signal_degraded"        # survives, treatment may restore
    MIXED = "mixed_failure"             # partial collapse; some organs dead, some alive
    COLLAPSE = "computation_collapse"   # NaN / diverged / cannot compute
    UNKNOWN = "signal_unknown"          # real signal requires a real run (not oracle-decidable)


class Outcome(Enum):
    PASS = "pass"
    FAIL_DEGRADED = "fail_degraded"              # signal there, contract not met
    FAIL_MIXED = "fail_mixed"                    # partial collapse not repaired
    FAIL_COLLAPSE = "fail_collapse"              # ascend trigger
    DEFERRED = "deferred"                        # missing wiring; NOT collapse
    INFEASIBLE_RESIDENT = "infeasible_resident"  # too big to fit resident
    UNDETERMINED = "undetermined"                # needs the cheapest discriminating probe


# ── parent physical model (port of the sealed sim's Parent, hardened) ──────────────────
@dataclass(frozen=True)
class Parent:
    name: str
    n_params: int
    embed_head_params: int
    resident_env_bytes: int
    passthru_keep_bpw: Fraction
    doctor_reserve: Fraction
    fixed_overhead: Fraction
    representable_at: frozenset            # ladder rates a real packer can byte-pack today
    reachable_treatments: frozenset        # treatments with EXECUTABLE adapters
    representation_families: tuple = ("scalar_trellis_tqv2",)
    families_representable: frozenset = frozenset()  # families with a wired packer (empty => deployable ones)
    architecture_family: str = "qwen2.5-dense"
    # signal floors: KNOWN only for fixture parents. None => real-run required (UNKNOWN).
    collapse_floor: Optional[Fraction] = None
    mixed_floor: Optional[Fraction] = None
    degradation_floor: Optional[Fraction] = None
    mixed_repairable: bool = False         # can an organ-protecting treatment reach a partial collapse

    @property
    def resident_ceiling(self) -> Fraction:
        return Fraction(8 * self.resident_env_bytes, self.n_params)

    @property
    def passthru_floor(self) -> Fraction:
        return Fraction(self.embed_head_params, self.n_params) * self.passthru_keep_bpw

    def whole_bpw(self, body: Fraction, *, treated: bool) -> Fraction:
        """Whole-artifact bpw: body + treatment + overhead, floored by pass-through.
        Treatment reserve is ALWAYS counted; it never gets unbilled bytes (section 13)."""
        treat = self.doctor_reserve if treated else Fraction(0)
        body_component = Fraction(body) + treat + self.fixed_overhead
        return max(body_component, self.passthru_floor + treat + self.fixed_overhead)


def parent_from_prior(label: str, *, envelope_gib: float = gp.ENVELOPE_GIB["headline"]) -> Parent:
    """Build a real Parent (feasibility + accounting only) from the sealed diagnostic prior.
    Signal floors are None: a real parent's signal is a real-run verdict, never invented."""
    p = gp.prior_for(label)
    n = int(p["n_params"])
    # lexical (embed+head) share implied by the packet's passthru_floor_bpw at 6-bit keep.
    keep = Fraction(6)
    embed_head = int(round((float(p["passthru_floor_bpw"]) / float(keep)) * n))
    # A REAL parent can only byte-pack with a deployable family, and only at/above its packer
    # floor. Today that is scalar_trellis_tqv2 at >= 1.34 eff-bpw, so every sub-bit rate DEFERS
    # (the honest "no sub-1-bit deployable packer" state, never a false collapse floor).
    deployable = [f for f in p["representation_families"]
                  if gp.REPRESENTATION_FAMILIES.get(f, {}).get("deployable")]
    floor = min((float(gp.REPRESENTATION_FAMILIES[f].get("floor_eff_bpw", 1e9)) for f in deployable),
                default=1e9)
    representable = frozenset(r for r in gp.RATE_LADDER if float(r) >= floor)
    reachable = frozenset({t for t in gp.REACHABLE_TREATMENTS
                           if gp.treatment_reachable(t, architecture_family=p["architecture_family"])})
    return Parent(
        name=label, n_params=n, embed_head_params=embed_head,
        resident_env_bytes=int(round(envelope_gib * (1024 ** 3))),
        passthru_keep_bpw=keep,
        doctor_reserve=gp.DOCTOR_RESERVE_PRIOR,
        fixed_overhead=Fraction(str(p["fixed_overhead_bpw"])).limit_denominator(10000),
        representable_at=representable, reachable_treatments=reachable,
        representation_families=tuple(p["representation_families"]),
        families_representable=frozenset(deployable),
        architecture_family=p["architecture_family"],
    )


@dataclass
class ExperimentResult:
    parent: str
    rate: Fraction
    family: Optional[str]
    treatment: Optional[str]
    signal: Optional[Signal]
    outcome: Outcome
    body_bpw: Fraction
    whole_bpw: Optional[Fraction]
    note: str = ""


class HeavyLock:
    """Single-heavy-controller guard. Mirrors reports/cron/studio_heavy.lock. Gravity never
    acquires it while another owner (the live 72B campaign) holds it, so it can never become
    a second heavy controller (section 2, invariant test)."""

    def __init__(self, held_by: Optional[str]):
        self.held_by = held_by

    def held(self) -> bool:
        return self.held_by is not None

    def free_for(self, lane_id: str) -> bool:
        return self.held_by is None or self.held_by == lane_id


# ── the inverted (ascending) search scheduler ──────────────────────────────────────────
class InvertedSearch:
    """Deterministic oracle-class scheduler. It runs only light probes and refuses to seize
    the heavy lock. Representation-family escalation precedes BPW escalation (section 14)."""

    def __init__(self, heavy_lock: HeavyLock, *, lane_id: str = "hawking-gravity"):
        self.heavy_lock = heavy_lock
        self.lane_id = lane_id
        self._seen: set = set()          # dedup: (parent, rate, family, treatment)
        self._results: dict = {}         # cache: (parent, rate, family) -> result
        self.log: list[ExperimentResult] = []

    # -- oracles ----------------------------------------------------------------------
    def _f0(self, p: Parent, rate: Fraction, family: str) -> Optional[Outcome]:
        if family not in p.representation_families:
            return Outcome.DEFERRED
        representable_fams = p.families_representable or frozenset(
            f for f in p.representation_families
            if gp.REPRESENTATION_FAMILIES.get(f, {}).get("deployable"))
        if family not in representable_fams:
            return Outcome.DEFERRED  # no wired packer for this family -> deferral, not collapse
        if rate not in p.representable_at:
            return Outcome.DEFERRED
        if p.whole_bpw(rate, treated=False) > p.resident_ceiling:
            return Outcome.INFEASIBLE_RESIDENT
        return None

    def _signal(self, p: Parent, rate: Fraction) -> Signal:
        if p.collapse_floor is None:
            return Signal.UNKNOWN  # real parent: a real run decides, never invented
        if rate < p.collapse_floor:
            return Signal.COLLAPSE
        if p.mixed_floor is not None and rate < p.mixed_floor:
            return Signal.MIXED
        if p.degradation_floor is not None and rate < p.degradation_floor:
            return Signal.DEGRADED
        return Signal.OK

    def probe(self, p: Parent, rate: Fraction, family: str, contract_max_whole: Fraction) -> ExperimentResult:
        rate = Fraction(rate)
        ck = (p.name, rate, family)
        if ck in self._results:
            return self._results[ck]

        f0 = self._f0(p, rate, family)
        if f0 is not None:
            return self._finish(ExperimentResult(p.name, rate, family, None, None, f0, rate, None,
                                                  note="f0 gate"))

        sig = self._signal(p, rate)
        if sig is Signal.UNKNOWN:
            return self._finish(ExperimentResult(p.name, rate, family, None, sig, Outcome.UNDETERMINED,
                                                  rate, None, note="real run required"))
        if sig is Signal.COLLAPSE:
            return self._finish(ExperimentResult(p.name, rate, family, None, sig,
                                                  Outcome.FAIL_COLLAPSE, rate, None, note="computation collapse"))

        degraded = sig in (Signal.DEGRADED, Signal.MIXED)
        treatment = gp.select_treatment(p.reachable_treatments, degraded=degraded,
                                         architecture_family=p.architecture_family)
        treated = treatment is not None
        whole = p.whole_bpw(rate, treated=treated)

        if sig is Signal.MIXED:
            # organ-aware: a global correction is NOT applied blindly to a partial collapse.
            # A MIXED rung passes only when an organ-protecting treatment can actually reach the
            # collapsed subspace (p.mixed_repairable). Otherwise it fails and the search must try
            # a different representation before ascending (section 10, 14).
            if p.mixed_repairable and treated and whole <= contract_max_whole:
                outcome, note = Outcome.PASS, f"mixed repaired ({treatment}, organ-protected)"
            else:
                outcome = Outcome.FAIL_MIXED
                note = "partial collapse; organ-protecting treatment could not close the contract"
            return self._finish(ExperimentResult(p.name, rate, family, treatment, sig, outcome,
                                                  rate, whole, note=note))

        if whole <= contract_max_whole and (not degraded or treated):
            outcome, note = Outcome.PASS, ("treated" if treated else "clean")
        else:
            outcome = Outcome.FAIL_DEGRADED
            note = "contract not met" if whole > contract_max_whole else "no reachable treatment"
        return self._finish(ExperimentResult(p.name, rate, family, treatment, sig, outcome,
                                              rate, whole, note=note))

    def _finish(self, r: ExperimentResult) -> ExperimentResult:
        key = (r.parent, r.rate, r.family, r.treatment)
        if key in self._seen:
            raise GravityEngineError(f"duplicate experiment {key}")
        self._seen.add(key)
        self._results[(r.parent, r.rate, r.family)] = r
        self.log.append(r)
        return r

    def _families_for(self, p: Parent) -> list[str]:
        # deployable/representable families first, then oracle families (materially different)
        fams = list(p.representation_families)
        fams.sort(key=lambda f: (not gp.REPRESENTATION_FAMILIES.get(f, {}).get("deployable"), f))
        return fams

    def evaluate_rate(self, p: Parent, rate: Fraction, contract_max_whole: Fraction) -> ExperimentResult:
        """Probe a rate, escalating representation family BEFORE giving up (section 14).
        Returns the best result at this rate (a PASS if any family passes; otherwise the
        most informative failure). A collapse across the FIRST family still tries an
        alternate family before the caller ascends."""
        best: Optional[ExperimentResult] = None
        for fam in self._families_for(p):
            res = self.probe(p, rate, fam, contract_max_whole)
            if res.outcome is Outcome.PASS:
                return res
            if best is None:
                best = res
            else:
                # prefer a signal-bearing failure over a deferral/infeasible for information
                rank = {Outcome.FAIL_DEGRADED: 3, Outcome.FAIL_MIXED: 3, Outcome.FAIL_COLLAPSE: 2,
                        Outcome.UNDETERMINED: 1, Outcome.INFEASIBLE_RESIDENT: 0, Outcome.DEFERRED: 0}
                if rank.get(res.outcome, 0) > rank.get(best.outcome, 0):
                    best = res
        assert best is not None
        return best

    def search(self, p: Parent, start: Fraction, contract_max_whole: Fraction) -> dict[str, Any]:
        """Inverted search. Ascend on collapse/exhausted-fail, descend on pass, refine,
        stop with a pass AND an evidenced lower boundary directly below it."""
        start = Fraction(start)
        idx = gp.ladder_index(start)
        best_pass: Optional[Fraction] = None
        lower_boundary: Optional[Fraction] = None
        trajectory: list[dict[str, Any]] = []

        i = idx
        while 0 <= i < len(gp.RATE_LADDER):
            rate = gp.RATE_LADDER[i]
            res = self.evaluate_rate(p, rate, contract_max_whole)
            trajectory.append({"rate": gp.rate_identity(rate)["label"], "family": res.family,
                               "outcome": res.outcome.value, "note": res.note})

            if res.outcome is Outcome.PASS:
                best_pass = rate
                if i == 0:
                    break
                i -= 1
                continue
            if res.outcome in (Outcome.FAIL_COLLAPSE, Outcome.FAIL_DEGRADED, Outcome.FAIL_MIXED):
                if best_pass is not None:
                    lower_boundary = rate
                    break
                i += 1
                continue
            if res.outcome is Outcome.INFEASIBLE_RESIDENT:
                i += 1
                continue
            if res.outcome in (Outcome.DEFERRED, Outcome.UNDETERMINED):
                i += 1  # missing wiring or needs a real run: skip up, never conclude a false floor
                continue

        return {
            "parent": p.name,
            "event_horizon": gp.rate_identity(best_pass) if best_pass else None,
            "passing_rate": gp.rate_identity(best_pass)["label"] if best_pass else None,
            "lower_boundary": gp.rate_identity(lower_boundary)["label"] if lower_boundary else None,
            "evidenced_floor": best_pass is not None and lower_boundary is not None,
            "trajectory": trajectory,
        }


def better_artifact(a: ExperimentResult, b: ExperimentResult) -> ExperimentResult:
    """At equal quality (both PASS), the SMALLER complete WHOLE-artifact bpw wins. A lower
    nominal BODY rate with heavy healing does NOT automatically win (section 13)."""
    if a.outcome is not Outcome.PASS or b.outcome is not Outcome.PASS:
        raise GravityEngineError("better_artifact compares two PASS results")
    return a if a.whole_bpw <= b.whole_bpw else b


# ── gravitational acquisition (section 9): changes PRIORITY only, never quality ────────
_GRAVITY_WEIGHTS = {"info": 0.5, "boundary": 0.7, "transfer": 0.2, "cost": 0.3,
                    "wall": 0.3, "disk": 0.2, "risk": 0.4, "gravity": 1.0}


def gravity_bonus(candidate: dict[str, Any], parent_state: dict[str, Any]) -> float:
    """G(x): the Gravity scheduling bonus (section 9). Positive pull toward serious sub-bit
    probes; negative for physically impossible, duplicated, or frontier-inert probes.
    This adjusts ORDERING only."""
    rate = gp.parse_rate(candidate["rate"]) if isinstance(candidate.get("rate"), str) \
        else Fraction(candidate["rate"])
    g = 0.0
    if gp.is_subbit(rate):
        g += 1.0
    cov = gp.subbit_coverage_status(parent_state)
    if not cov["satisfied"]:
        g += 0.8
    fam = candidate.get("family")
    failed = {t.get("family") for t in parent_state.get("families_failed_causally", [])}
    if fam and all(fam and gp.materially_different(fam, f) for f in failed if f in gp.REPRESENTATION_FAMILIES):
        g += 0.5  # a materially different family from every prior causal failure
    if candidate.get("near_boundary"):
        g += 0.4
    if candidate.get("can_change_extreme"):
        g += 0.6
    if candidate.get("distinguishes_degradation_from_collapse"):
        g += 0.3
    # penalties
    if fam in failed:
        g -= 1.0
    if candidate.get("physically_impossible"):
        g -= 1.5
    if candidate.get("duplicates_evidence"):
        g -= 1.0
    if candidate.get("below_sealed_lower_bound"):
        g -= 1.0
    if candidate.get("cannot_alter_frontier"):
        g -= 0.8
    return g


def gravity_acquisition(candidate: dict[str, Any], parent_state: dict[str, Any], *,
                        p_physical: float, p_signal: float, p_doctor_reachable: float,
                        p_changes_extreme: float, hv_gain: float, info_gain: float,
                        cost_wall: float, cost_disk: float, cost_risk: float,
                        weights: dict[str, float] | None = None) -> float:
    """A_Gravity(x) = P_phys * P_signal * P_doctor * P_changes_extreme * H + I + G
    - (lambda_wall*C_wall + lambda_disk*C_disk + lambda_risk*C_risk).
    Deterministic and recorded (section 9)."""
    w = {**_GRAVITY_WEIGHTS, **(weights or {})}
    core = p_physical * p_signal * p_doctor_reachable * p_changes_extreme * hv_gain
    g = gravity_bonus(candidate, parent_state)
    return (core + w["info"] * info_gain + w["gravity"] * g
            - w["wall"] * cost_wall - w["disk"] * cost_disk - w["risk"] * cost_risk)


def rank_candidates(candidates: list[dict[str, Any]], states: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-rank candidates by A_Gravity (priority only). Representation-family escalation
    precedes BPW escalation: a lower-rate different-representation probe outranks a
    higher-rate probe when the lower rate is still open (section 14)."""
    scored = []
    for c in candidates:
        st = states.get(c.get("model_label") or c.get("parent") or "", {}) or {}
        score = gravity_acquisition(
            c, st,
            p_physical=float(c.get("p_physical", 0.9)), p_signal=float(c.get("p_signal", 0.7)),
            p_doctor_reachable=float(c.get("p_doctor_reachable", 0.8)),
            p_changes_extreme=float(c.get("p_changes_extreme", 0.6)),
            hv_gain=float(c.get("hv_gain", 1.0)), info_gain=float(c.get("info_gain", 0.5)),
            cost_wall=float(c.get("cost_wall", 0.0)), cost_disk=float(c.get("cost_disk", 0.0)),
            cost_risk=float(c.get("cost_risk", 0.0)))
        rate = gp.parse_rate(c["rate"]) if isinstance(c.get("rate"), str) else Fraction(c.get("rate", 1))
        scored.append((score, gp.is_subbit(rate), -float(rate), c))
    # sort: higher score first; among ties prefer sub-bit and then lower rate
    scored.sort(key=lambda t: (-t[0], not t[1], t[2]))
    return [{**c, "gravity_acquisition": round(s, 6)} for s, _sb, _r, c in scored]


# ── Gravity parent-state FSM (section 8): hash-chained, resumable, idempotent ──────────
GRAVITY_STATES: tuple[str, ...] = (
    "GRAVITY_UNINITIALIZED", "GRAVITY_DIAGNOSTIC", "GRAVITY_F0", "GRAVITY_F1", "GRAVITY_F2",
    "GRAVITY_DIAGNOSE", "GRAVITY_RESCUE", "GRAVITY_DESCEND", "GRAVITY_ASCEND", "GRAVITY_BOUNDARY",
    "GRAVITY_F3", "GRAVITY_F4", "GRAVITY_SEALED",
)
GRAVITY_TERMINALS: tuple[str, ...] = (
    "GRAVITY_PHYSICAL_IMPOSSIBILITY", "GRAVITY_STRUCTURAL_INCOMPATIBILITY",
    "GRAVITY_ESCAPE_AUTHORIZED", "GRAVITY_UNPROVEN", "GRAVITY_INVALID",
)
# Allowed transitions: the linear progression plus jumps to diagnose/rescue/descend/ascend
# loops and to terminals. Enforced so no state is entered illegally.
_ALLOWED: dict[str, frozenset] = {
    "GRAVITY_UNINITIALIZED": frozenset({"GRAVITY_DIAGNOSTIC", "GRAVITY_INVALID"}),
    "GRAVITY_DIAGNOSTIC": frozenset({"GRAVITY_F0", "GRAVITY_INVALID", "GRAVITY_PHYSICAL_IMPOSSIBILITY"}),
    "GRAVITY_F0": frozenset({"GRAVITY_F1", "GRAVITY_PHYSICAL_IMPOSSIBILITY", "GRAVITY_INVALID"}),
    "GRAVITY_F1": frozenset({"GRAVITY_F2", "GRAVITY_DIAGNOSE", "GRAVITY_STRUCTURAL_INCOMPATIBILITY", "GRAVITY_INVALID"}),
    "GRAVITY_F2": frozenset({"GRAVITY_DIAGNOSE", "GRAVITY_INVALID"}),
    "GRAVITY_DIAGNOSE": frozenset({"GRAVITY_RESCUE", "GRAVITY_DESCEND", "GRAVITY_ASCEND", "GRAVITY_INVALID"}),
    "GRAVITY_RESCUE": frozenset({"GRAVITY_DESCEND", "GRAVITY_ASCEND", "GRAVITY_BOUNDARY", "GRAVITY_DIAGNOSE", "GRAVITY_INVALID"}),
    "GRAVITY_DESCEND": frozenset({"GRAVITY_F0", "GRAVITY_BOUNDARY", "GRAVITY_DIAGNOSE", "GRAVITY_INVALID"}),
    "GRAVITY_ASCEND": frozenset({"GRAVITY_F0", "GRAVITY_BOUNDARY", "GRAVITY_ESCAPE_AUTHORIZED", "GRAVITY_UNPROVEN", "GRAVITY_INVALID"}),
    "GRAVITY_BOUNDARY": frozenset({"GRAVITY_F3", "GRAVITY_DESCEND", "GRAVITY_ASCEND", "GRAVITY_INVALID"}),
    "GRAVITY_F3": frozenset({"GRAVITY_F4", "GRAVITY_BOUNDARY", "GRAVITY_INVALID"}),
    "GRAVITY_F4": frozenset({"GRAVITY_SEALED", "GRAVITY_ESCAPE_AUTHORIZED", "GRAVITY_UNPROVEN", "GRAVITY_INVALID"}),
    "GRAVITY_SEALED": frozenset(),
}
for _t in GRAVITY_TERMINALS:
    _ALLOWED.setdefault(_t, frozenset())


def new_parent_state(parent_label: str, *, initial_stress_rate: Fraction | None = None,
                     policy_version: str = gp.GRAVITY_POLICY_VERSION) -> dict[str, Any]:
    """Build a fresh, sealed Gravity parent state (section 8)."""
    ss = initial_stress_rate or gp.parse_rate(
        gp.compute_stress_start(parent_label)["chosen_stress_rate"]["label"])
    state = {
        "schema": GRAVITY_PARENT_STATE_SCHEMA,
        "parent_identity": {"label": parent_label,
                            "hf_or_source_id": gp.prior_for(parent_label).get("hf_or_source_id")},
        "gravity_enabled": False,
        "gravity_policy_version": policy_version,
        "initial_stress_rate": gp.rate_identity(ss),
        "current_rate": gp.rate_identity(ss),
        "current_representation_family": gp.prior_for(parent_label)["representation_families"][0],
        "current_evidence_level": None,
        "current_diagnosis": None,
        "current_doctor_program": None,
        "lowest_tested_rate": None,
        "lowest_measured_fail": None,
        "highest_measured_fail": None,
        "lowest_measured_pass": None,
        "highest_unproven": None,
        "first_passing_rate": None,
        "escape_receipt": None,
        "subbit_coverage_status": {"satisfied": False, "route": None},
        "representation_families_tested": [],
        "families_failed_causally": [],
        "doctor_treatments_attempted": [],
        "physical_budget": None,
        "resource_budget": None,
        "per_rate": {},
        "next_action": "diagnose",
        "fallback_rate": None,
        "stop_condition": None,
        "reopening_conditions": [],
        "fsm_state": "GRAVITY_UNINITIALIZED",
        "transitions": [],
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    return seal_field(state, "state_sha256")


def transition_allowed(from_state: str, to_state: str) -> bool:
    return to_state in _ALLOWED.get(from_state, frozenset())


def advance_state(state: dict[str, Any], to_state: str, *, event: dict[str, Any] | None = None,
                  experiment_id: str | None = None) -> dict[str, Any]:
    """Append a hash-chained FSM transition. Append-only, resumable, idempotent, and protected
    from duplicate launch (section 8). An identical (from,to,experiment_id) transition already
    at the tail is a no-op; a NEW launch reusing a consumed experiment_id is refused."""
    if not sealed(state, "state_sha256"):
        raise GravityEngineError("cannot advance an unsealed/tampered Gravity state")
    frm = state["fsm_state"]
    if to_state not in GRAVITY_STATES and to_state not in GRAVITY_TERMINALS:
        raise GravityEngineError(f"unknown Gravity state {to_state!r}")
    transitions = list(state.get("transitions", []))

    # idempotency: the same transition already applied at the tail -> no-op replay. Matches on
    # (to_state, experiment_id) because after applying, the state's fsm_state is already to_state.
    if transitions:
        tail = transitions[-1]
        if tail["to"] == to_state and tail.get("experiment_id") == experiment_id:
            return state
    # duplicate-launch protection: a consumed experiment_id cannot launch again
    if experiment_id is not None:
        for t in transitions:
            if t.get("experiment_id") == experiment_id and t.get("launched"):
                raise GravityEngineError(f"duplicate heavy launch refused for experiment {experiment_id}")
    if not transition_allowed(frm, to_state):
        raise GravityEngineError(f"illegal Gravity transition {frm} -> {to_state}")

    prev_link = transitions[-1]["entry_sha256"] if transitions else "genesis"
    entry = {
        "seq": len(transitions), "from": frm, "to": to_state, "at": now_iso(),
        "event": event or {}, "experiment_id": experiment_id,
        "launched": bool(experiment_id), "prev_link_sha256": prev_link,
    }
    entry["entry_sha256"] = hash_value({k: v for k, v in entry.items() if k != "entry_sha256"})
    transitions.append(entry)

    body = {k: v for k, v in state.items() if k != "state_sha256"}
    body["fsm_state"] = to_state
    body["transitions"] = transitions
    body["updated_at"] = now_iso()
    return seal_field(body, "state_sha256")


def parent_identity_key(state: dict[str, Any]) -> tuple[Any, ...]:
    """The immutable Gravity identity of a parent: label + policy version + initial stress rate."""
    pid = state.get("parent_identity", {}) or {}
    ss = state.get("initial_stress_rate", {}) or {}
    return (pid.get("label"), state.get("gravity_policy_version"), ss.get("label"))


def assert_no_identity_conflict(states: list[dict[str, Any]]) -> bool:
    """Section 17: the same parent may not carry conflicting Gravity identities. Two states with
    the same label but a different policy version or stress rate are a conflict (fail closed)."""
    by_label: dict[str, tuple] = {}
    for s in states:
        label = (s.get("parent_identity", {}) or {}).get("label")
        ident = parent_identity_key(s)
        if label in by_label and by_label[label] != ident:
            raise GravityEngineError(f"conflicting Gravity identities for parent {label!r}: "
                                     f"{by_label[label]} vs {ident}")
        by_label[label] = ident
    return True


# Only replicated F4 evidence may finalize the Event Horizon; F0-F2 cannot masquerade (section 19).
FINALIZABLE_EVIDENCE = "F4"


def can_finalize_event_horizon(state: dict[str, Any]) -> tuple[bool, list[str]]:
    lvl = str(state.get("current_evidence_level") or "").upper()
    if lvl != FINALIZABLE_EVIDENCE:
        return False, [f"evidence level {lvl or 'none'} cannot finalize the Event Horizon "
                       f"(requires replicated F4; F0-F2 cannot masquerade)"]
    return True, []


def verify_state_chain(state: dict[str, Any]) -> tuple[bool, list[str]]:
    """Verify the state seal AND the hash-chain of its transitions (crash/resume integrity)."""
    reasons: list[str] = []
    if not sealed(state, "state_sha256"):
        reasons.append("state self-seal invalid")
    prev = "genesis"
    for i, t in enumerate(state.get("transitions", [])):
        if t.get("seq") != i:
            reasons.append(f"transition {i} out of order")
        if t.get("prev_link_sha256") != prev:
            reasons.append(f"transition {i} chain break")
        recomputed = hash_value({k: v for k, v in t.items() if k != "entry_sha256"})
        if t.get("entry_sha256") != recomputed:
            reasons.append(f"transition {i} entry hash mismatch (tampered)")
        prev = t.get("entry_sha256")
    return (not reasons), reasons


# ── additive Gravity augmentation of existing rows (section 16) ─────────────────────────
def _gravity_row_fields(state: dict[str, Any]) -> dict[str, Any]:
    cov = state.get("subbit_coverage_status", {}) or {}
    esc = state.get("escape_receipt")
    return {
        "gravity_policy": state.get("gravity_policy_version"),
        "gravity_state": state.get("fsm_state"),
        "gravity_start_rate": state.get("initial_stress_rate"),
        "gravity_current_rate": state.get("current_rate"),
        "gravity_coverage": {"satisfied": cov.get("satisfied", False), "route": cov.get("route")},
        "gravity_escape_status": ("granted" if esc else "none"),
        "gravity_next_probe": state.get("next_action"),
    }


def augment_row(row: dict[str, Any], state: dict[str, Any], *, seal_name: str = "row_sha256") -> dict[str, Any]:
    """Attach Gravity fields to an existing queue/frontier row and re-seal it. Additive: it
    never duplicates a parent row, it augments the one that exists (section 16)."""
    body = {k: v for k, v in row.items() if k != seal_name}
    body["gravity"] = _gravity_row_fields(state)
    return seal_field(body, seal_name)


# ── source-bound program + launch gate (section 21) ────────────────────────────────────
def materialize_gravity_program(parent_label: str, *, rate: Fraction, kind: str,
                                source_manifest_sha256: str | None,
                                doctor_program: dict[str, Any] | None = None,
                                adapter_id: str | None = None) -> dict[str, Any]:
    """Compile a source-bound Gravity program descriptor. kind is 'subbit_stress' or
    'higher_rate_fallback'. The program declares its launch gate; it cannot execute until
    every gate holds (default-off, admission, heavy-lease free, release boundary)."""
    if kind not in ("subbit_stress", "higher_rate_fallback"):
        raise GravityEngineError(f"unknown program kind {kind!r}")
    p = gp.prior_for(parent_label)
    rate = Fraction(rate)
    program = {
        "schema": GRAVITY_PROGRAM_SCHEMA,
        "kind": kind,
        "parent_label": parent_label,
        "hf_or_source_id": p.get("hf_or_source_id"),
        "architecture_family": p.get("architecture_family"),
        "rate": gp.rate_identity(rate),
        "is_subbit": gp.is_subbit(rate),
        "representation_family": p["representation_families"][0],
        "doctor_program": doctor_program or {"promote": [], "note": "diagnose-first"},
        "adapter_id": adapter_id or (gp.REACHABLE_TREATMENTS["condense_control"]["adapter_id"]
                                     if p.get("architecture_family") == "qwen2.5-dense" else None),
        "source_manifest_sha256": source_manifest_sha256,
        "required_controls": ["zero_treatment", "equal_byte_codec", "smaller_higher_bit"],
        "lane": "hawking-gravity-oracle-first",
        "launch_gate": {
            "gravity_enabled": False, "resource_admission_passed": False,
            "heavy_lease_free": False, "release_boundary_signed": False,
            "source_program_executable": source_manifest_sha256 is not None,
        },
        "created_at": now_iso(),
    }
    return seal_field(program, "program_sha256")


def program_launchable(program: dict[str, Any], *, policy: dict[str, Any] | None,
                       heavy_lock: HeavyLock, env: dict[str, str] | None = None,
                       admission_passed: bool = False) -> tuple[bool, list[str]]:
    """A Gravity program may launch ONLY when Gravity is enabled, admission passed, the heavy
    lease is free, and the release boundary is signed. Default-off => never launchable now."""
    reasons: list[str] = []
    if not sealed(program, "program_sha256"):
        reasons.append("program seal invalid")
    if not gp.gravity_enabled(policy=policy, env=env):
        reasons.append("gravity default-off (not enabled)")
    if not admission_passed:
        reasons.append("resource admission not passed")
    if not heavy_lock.free_for("hawking-gravity"):
        reasons.append(f"heavy lease held by {heavy_lock.held_by} (would be a second heavy controller)")
    if program.get("source_manifest_sha256") is None:
        reasons.append("program is not source-bound")
    return (not reasons), reasons


def materialize_live_parent_programs(parent_label: str, *,
                                     source_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Section 21: the source-bound sub-bit stress program for the current highest-value
    eligible parent PLUS its higher-rate fallback. Neither launches (both launch-gated)."""
    p = gp.prior_for(parent_label)
    stress = gp.parse_rate(gp.compute_stress_start(parent_label)["chosen_stress_rate"]["label"])
    # fallback: the smallest ladder rate above 1.0 at/above first_likely_viable
    flv = p.get("first_likely_viable_bpw")
    flv_val = float(str(flv).split("-")[0]) if isinstance(flv, str) else float(flv)
    fallback_candidates = [r for r in gp.RATE_LADDER if r > gp.ONE_BPW and float(r) >= flv_val]
    fallback = fallback_candidates[0] if fallback_candidates else gp.next_higher(gp.ONE_BPW)
    return {
        "parent_label": parent_label,
        "subbit_stress": materialize_gravity_program(
            parent_label, rate=stress, kind="subbit_stress",
            source_manifest_sha256=source_manifest_sha256),
        "higher_rate_fallback": materialize_gravity_program(
            parent_label, rate=fallback, kind="higher_rate_fallback",
            source_manifest_sha256=source_manifest_sha256),
    }


# ── deliverable generators (section 20) ────────────────────────────────────────────────
def build_state_doc(parent_labels: list[str], *, live_parent: str,
                    source_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Content of GRAVITY_STATE.json: one Gravity parent-state per parent + the live parent's
    materialized source-bound programs. Default-off; nothing launchable."""
    states = {label: new_parent_state(label) for label in parent_labels}
    doc = {
        "schema": GRAVITY_STATE_DOC_SCHEMA,
        "policy_version": gp.GRAVITY_POLICY_VERSION,
        "generated_at": now_iso(),
        "gravity_enabled": False,
        "live_parent": live_parent,
        "parent_states": states,
        "source_bound_programs": materialize_live_parent_programs(
            live_parent, source_manifest_sha256=source_manifest_sha256),
        "note": "default-off; programs are launch-gated and cannot execute outside admission",
    }
    return seal_field(doc, "state_doc_sha256")


def build_validation_doc() -> dict[str, Any]:
    """Content of GRAVITY_VALIDATION.json: module selftests + invariant assertions + seals."""
    checks = {
        "policy_selftest": gp.selftest(),
        "receipts_selftest": gr.selftest(),
        "engine_selftest": selftest(),
    }
    # invariant witnesses
    p = parent_from_prior("72B")
    ok_gate, why_gate = gp.can_finalize_extreme(
        whole_bpw=1.5, coverage={"satisfied": False}, escape_receipt=None, escape_receipt_valid=False)
    doc = {
        "schema": GRAVITY_VALIDATION_SCHEMA,
        "policy_version": gp.GRAVITY_POLICY_VERSION,
        "generated_at": now_iso(),
        "module_selftests": {k: {"ok": v.get("ok", False)} for k, v in checks.items()},
        "invariants": {
            "exact_rational_rates": gp.RATE_LADDER == sorted(gp.RATE_LADDER),
            "extreme_gate_refuses_escape_without_receipt": (not ok_gate) and bool(why_gate),
            "whole_artifact_bpw_authoritative": not gp.is_subbit_artifact(
                gp.whole_artifact_bpw({"base": 0.5, "doctor": 0.4, "packaging": 0.15})),
            "default_off": gp.gravity_enabled(policy=gp.default_policy(),
                                              env={"HAWKING_GRAVITY_ENABLED": "1"}) is False,
            "resident_ceiling_72b": float(p.resident_ceiling),
        },
        "all_selftests_ok": all(v.get("ok") for v in checks.values()),
    }
    return seal_field(doc, "validation_sha256")


# ── offline selftest (exercises the search-policy invariants end-to-end) ───────────────
def _sim_giant_moe() -> Parent:
    GiB = 1024 ** 3
    return Parent(
        name="685B(sim)", n_params=685_000_000_000, embed_head_params=1_840_000_000,
        resident_env_bytes=64 * GiB, passthru_keep_bpw=Fraction(6),
        doctor_reserve=Fraction(3, 100), fixed_overhead=Fraction(1, 50),
        representable_at=frozenset({Fraction(1, 10), Fraction(1, 5), Fraction(1, 4), Fraction(1, 3),
                                    Fraction(1, 2), Fraction(11, 20), Fraction(4, 5)}),
        reachable_treatments=frozenset({"doctor_static", "doctor_conditional"}),
        representation_families=("scalar_trellis_tqv2", "additive_codebooks"),
        families_representable=frozenset({"scalar_trellis_tqv2", "additive_codebooks"}),
        collapse_floor=Fraction(1, 4), mixed_floor=Fraction(3, 10), degradation_floor=Fraction(1, 2),
        mixed_repairable=True,
    )


def _sim_synthetic() -> Parent:
    """Section 19 synthetic outcomes: 0.25 collapse, 0.33 mixed, 0.50 degraded+doctor pass.
    Expected Event Horizon = 0.50, not 1.00."""
    GiB = 1024 ** 3
    return Parent(
        name="synthetic", n_params=685_000_000_000, embed_head_params=1_840_000_000,
        resident_env_bytes=64 * GiB, passthru_keep_bpw=Fraction(6),
        doctor_reserve=Fraction(3, 100), fixed_overhead=Fraction(1, 50),
        representable_at=frozenset({Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(11, 20), Fraction(4, 5)}),
        reachable_treatments=frozenset({"doctor_static", "doctor_full"}),
        representation_families=("scalar_trellis_tqv2", "additive_codebooks"),
        families_representable=frozenset({"scalar_trellis_tqv2", "additive_codebooks"}),
        collapse_floor=Fraction(3, 10), mixed_floor=Fraction(2, 5), degradation_floor=Fraction(11, 20),
        mixed_repairable=False,
    )


def selftest() -> dict[str, Any]:
    # synthetic Event Horizon must be 0.50
    p = _sim_synthetic()
    s = InvertedSearch(HeavyLock("doctor_v5_disk25_successor"))
    out = s.search(p, start=Fraction(1, 4), contract_max_whole=Fraction(1))
    assert out["passing_rate"] == "1/2", out
    assert out["evidenced_floor"] and out["lower_boundary"] == "1/3", out
    # never seized the heavy lock
    assert s.heavy_lock.held_by == "doctor_v5_disk25_successor"

    # giant realizes sub-bit
    g = InvertedSearch(HeavyLock(None))
    og = g.search(_sim_giant_moe(), start=Fraction(11, 20), contract_max_whole=Fraction(1))
    assert og["passing_rate"] is not None and og["evidenced_floor"]

    # FSM: hash-chained, resumable, tamper-detectable
    stt = new_parent_state("72B")
    stt = advance_state(stt, "GRAVITY_DIAGNOSTIC")
    stt = advance_state(stt, "GRAVITY_F0", experiment_id="exp-72b-0.8-f0")
    ok_chain, _ = verify_state_chain(stt); assert ok_chain
    # idempotent
    stt2 = advance_state(stt, "GRAVITY_F0", experiment_id="exp-72b-0.8-f0")
    assert stt2 == stt
    # duplicate launch refused
    stt3 = advance_state(stt, "GRAVITY_F1")
    dup_refused = False
    try:
        advance_state(stt3, "GRAVITY_F0", experiment_id="exp-72b-0.8-f0")
    except GravityEngineError:
        dup_refused = True
    assert dup_refused
    # illegal transition refused
    illegal = False
    try:
        advance_state(new_parent_state("72B"), "GRAVITY_SEALED")
    except GravityEngineError:
        illegal = True
    assert illegal

    # programs are not launchable while default-off / lock held
    progs = materialize_live_parent_programs("72B", source_manifest_sha256="a" * 64)
    ok_launch, why = program_launchable(progs["subbit_stress"], policy=gp.default_policy(),
                                        heavy_lock=HeavyLock("doctor_v5_disk25_successor"),
                                        env={"HAWKING_GRAVITY_ENABLED": "1"}, admission_passed=True)
    assert not ok_launch and why

    # row augmentation is additive + re-sealed
    import succ_queue as q
    row = q.make_row(parent_label="72B", current_status="planned", architecture_family="qwen2.5-dense")
    aug = augment_row(row, stt)
    assert sealed(aug, "row_sha256") and "gravity" in aug and "gravity" not in row

    return {"ok": True, "event_horizon_synthetic": out["passing_rate"],
            "giant_pass": og["passing_rate"], "fsm_chain_ok": ok_chain}


# ── CLI (section 22) ───────────────────────────────────────────────────────────────────
def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="succ_gravity", description="Hawking Gravity engine (sub-bit-first).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("policy")
    ins = sub.add_parser("inspect"); ins.add_argument("--parent", default="72B")
    en = sub.add_parser("explain-next"); en.add_argument("--parent", default="72B")
    mat = sub.add_parser("materialize"); mat.add_argument("--parent", default="72B")
    mat.add_argument("--source-manifest-sha256", default=None)
    sub.add_parser("validate")
    st = sub.add_parser("state"); st.add_argument("--live-parent", default="72B")
    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        print(json.dumps(selftest(), indent=2, sort_keys=True)); return 0
    if args.cmd == "policy":
        print(json.dumps(gp.build_policy_manifest(), indent=2, sort_keys=True)); return 0
    if args.cmd == "inspect":
        ss = gp.compute_stress_start(args.parent)
        print(json.dumps({"parent": args.parent, "stress_start": ss,
                          "state": new_parent_state(args.parent)}, indent=2, sort_keys=True, default=str))
        return 0
    if args.cmd == "explain-next":
        p = gp.prior_for(args.parent)
        ss = gp.parse_rate(gp.compute_stress_start(args.parent)["chosen_stress_rate"]["label"])
        cand = [{"model_label": args.parent, "rate": gp.rate_identity(ss)["label"],
                 "family": p["representation_families"][0], "near_boundary": True,
                 "can_change_extreme": True, "distinguishes_degradation_from_collapse": True}]
        ranked = rank_candidates(cand, {args.parent: new_parent_state(args.parent)})
        print(json.dumps({"parent": args.parent, "next": ranked[0],
                          "direction": "upward from sub-bit stress point"},
                         indent=2, sort_keys=True, default=str))
        return 0
    if args.cmd == "materialize":
        print(json.dumps(materialize_live_parent_programs(
            args.parent, source_manifest_sha256=args.source_manifest_sha256),
            indent=2, sort_keys=True, default=str))
        return 0
    if args.cmd == "validate":
        print(json.dumps(build_validation_doc(), indent=2, sort_keys=True, default=str)); return 0
    if args.cmd == "state":
        print(json.dumps(build_state_doc(list(gp.PARENT_PRIORS), live_parent=args.live_parent),
                         indent=2, sort_keys=True, default=str))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
