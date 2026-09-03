"""Adaptive, constraint-first experiment-wave selection for Gravity campaigns.

The controller deliberately selects *waves*, not a fixed rung ladder.  A wave
can combine representation, kernel, scheduler, and state interventions, while
remaining pinned to one model and the current storage authority.  It is a
planning gate only: no returned plan is evidence of a speed or capability win.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "hawking.frontier.dynamic_wave.v1"
COST_DIMENSIONS = ("bytes", "operations", "depth", "synchronization", "state")
MEASUREMENT_MODES = ("clean", "shared_load_paired")
MIN_MATERIAL_REDUCTION = 0.01


class PlanningRefusal(ValueError):
    """The controller cannot lawfully schedule the requested experiment."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _number(value: Any, name: str, *, minimum: float = 0.0, maximum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PlanningRefusal(f"{name} must be a number")
    result = float(value)
    if result < minimum or (maximum is not None and result > maximum):
        bound = f">= {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise PlanningRefusal(f"{name} must be {bound}")
    return result


def _costs(value: Mapping[str, Any], name: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise PlanningRefusal(f"{name} must be a mapping")
    unknown = sorted(set(value) - set(COST_DIMENSIONS))
    if unknown:
        raise PlanningRefusal(f"{name} has unknown dimensions: {unknown}")
    return {dimension: _number(value.get(dimension, 0.0), f"{name}.{dimension}") for dimension in COST_DIMENSIONS}


def _reduction(value: Mapping[str, Any], name: str) -> dict[str, float]:
    result = _costs(value, name)
    for dimension, reduction in result.items():
        if reduction > 1.0:
            raise PlanningRefusal(f"{name}.{dimension} must not exceed 1")
    return result


@dataclass(frozen=True)
class Candidate:
    """A preregistrable intervention, expressed in physical cost reduction terms."""

    id: str
    families: tuple[str, ...]
    reduces: Mapping[str, float]
    artifact_bytes: int
    temporary_bytes: int
    falsifier: str
    blocked_by: tuple[str, ...] = ()
    reopens: tuple[str, ...] = ()
    incompatible_with: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Candidate":
        if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str) or not raw["id"].strip():
            raise PlanningRefusal("candidate needs a non-empty id")
        families = raw.get("families")
        if not isinstance(families, (list, tuple)) or not families or not all(isinstance(f, str) and f.strip() for f in families):
            raise PlanningRefusal(f"candidate {raw['id']!r} needs one or more method families")
        artifact = raw.get("artifact_bytes", 0)
        temporary = raw.get("temporary_bytes", 0)
        if not isinstance(artifact, int) or isinstance(artifact, bool) or artifact < 0:
            raise PlanningRefusal(f"candidate {raw['id']!r}.artifact_bytes must be a non-negative integer")
        if not isinstance(temporary, int) or isinstance(temporary, bool) or temporary < 0:
            raise PlanningRefusal(f"candidate {raw['id']!r}.temporary_bytes must be a non-negative integer")
        falsifier = raw.get("falsifier")
        if not isinstance(falsifier, str) or not falsifier.strip():
            raise PlanningRefusal(f"candidate {raw['id']!r} needs a concrete falsifier")
        return cls(
            id=raw["id"],
            families=tuple(sorted(set(families))),
            reduces=_reduction(raw.get("reduces", {}), f"candidate {raw['id']!r}.reduces"),
            artifact_bytes=artifact,
            temporary_bytes=temporary,
            falsifier=falsifier,
            blocked_by=tuple(sorted(set(raw.get("blocked_by", ())))),
            reopens=tuple(sorted(set(raw.get("reopens", ())))),
            incompatible_with=tuple(sorted(set(raw.get("incompatible_with", ())))),
        )


def validate_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the non-negotiable model, storage, roofline, and lease fences."""
    if not isinstance(state, Mapping):
        raise PlanningRefusal("state must be a mapping")
    model_id = state.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise PlanningRefusal("state.model_id is required")
    if state.get("model_selection_frozen") is not True:
        raise PlanningRefusal("one-mountain model selection is not frozen")
    storage = state.get("storage")
    if not isinstance(storage, Mapping):
        raise PlanningRefusal("state.storage is required")
    lane = storage.get("available_model_lane_bytes")
    if not isinstance(lane, int) or isinstance(lane, bool) or lane < 0:
        raise PlanningRefusal("storage.available_model_lane_bytes must be a non-negative integer")
    roofline = _costs(state.get("roofline", {}), "state.roofline")
    if not any(roofline.values()):
        raise PlanningRefusal("roofline has no measured remaining physical deficit")
    measurement_mode = state.get("measurement_mode", "clean")
    if measurement_mode not in MEASUREMENT_MODES:
        raise PlanningRefusal(f"state.measurement_mode must be one of {MEASUREMENT_MODES}")
    if measurement_mode == "clean":
        if state.get("clean_lease") is not True:
            raise PlanningRefusal("a clean lease is required before scheduling a performance wave")
        measurement = {"mode": "clean", "absolute_tps_eligible": True}
    else:
        # Shared-load runs are legitimate differential experiments when a clean
        # lane is unavailable, but their absolute TPS is deliberately unusable.
        # Alternating control/candidate runs is the minimum defence against the
        # natural drift of other workloads over the course of a long decode.
        if state.get("shared_load_invariant") is not True:
            raise PlanningRefusal("shared-load mode needs an explicit same-contemporaneous-load invariant")
        pairs = state.get("paired_control_runs")
        if not isinstance(pairs, int) or isinstance(pairs, bool) or pairs < 2:
            raise PlanningRefusal("shared-load mode needs at least two interleaved control/candidate pairs")
        if state.get("absolute_tps_claim") is not False:
            raise PlanningRefusal("shared-load mode must explicitly forbid an absolute TPS claim")
        measurement = {
            "mode": "shared_load_paired",
            "absolute_tps_eligible": False,
            "paired_control_runs": pairs,
            "required_order": "alternate source/candidate; preserve every pair including failures",
        }
    negatives = state.get("sealed_negatives", {})
    if not isinstance(negatives, Mapping):
        raise PlanningRefusal("state.sealed_negatives must be a mapping")
    return {
        "model_id": model_id,
        "model_selection_frozen": True,
        "storage": {"available_model_lane_bytes": lane, "authority": storage.get("authority")},
        "roofline": roofline,
        "measurement": measurement,
        "sealed_negatives": dict(negatives),
        "context": state.get("context", {}),
    }


def _is_reopened(candidate: Candidate, negative_id: str, negatives: Mapping[str, Any]) -> bool:
    row = negatives.get(negative_id)
    if not isinstance(row, Mapping):
        return negative_id in candidate.reopens
    return bool(row.get("reopen_condition_met")) and negative_id in candidate.reopens


def _admission_reasons(checked: Mapping[str, Any], candidate: Candidate) -> list[str]:
    """Return every reason a candidate is ineligible from already-checked state."""
    reasons: list[str] = []
    if max(candidate.reduces.values()) < MIN_MATERIAL_REDUCTION:
        reasons.append("no material physical-cost reduction declared")
    if candidate.artifact_bytes + candidate.temporary_bytes > checked["storage"]["available_model_lane_bytes"]:
        reasons.append("artifact plus temporary bytes exceed the authoritative model lane")
    for negative_id in candidate.blocked_by:
        if not _is_reopened(candidate, negative_id, checked["sealed_negatives"]):
            reasons.append(f"sealed negative {negative_id!r} has no satisfied reopen condition")
    return reasons


def admission_reasons(state: Mapping[str, Any], candidate: Candidate) -> list[str]:
    """Return every reason a candidate is ineligible; empty means admissible."""
    return _admission_reasons(validate_state(state), candidate)


def _pair_compatible(left: Candidate, right: Candidate) -> bool:
    left_families, right_families = set(left.families), set(right.families)
    return not (
        right.id in left.incompatible_with
        or left.id in right.incompatible_with
        or left_families.intersection(right.incompatible_with)
        or right_families.intersection(left.incompatible_with)
    )


def _combined_reduction(candidates: Iterable[Candidate]) -> dict[str, float]:
    result = {dimension: 0.0 for dimension in COST_DIMENSIONS}
    for candidate in candidates:
        for dimension in COST_DIMENSIONS:
            result[dimension] = 1.0 - (1.0 - result[dimension]) * (1.0 - candidate.reduces[dimension])
    return result


def _score(roofline: Mapping[str, float], candidates: tuple[Candidate, ...]) -> tuple[float, int, tuple[str, ...]]:
    reduction = _combined_reduction(candidates)
    physical_gain = sum(roofline[dimension] * reduction[dimension] for dimension in COST_DIMENSIONS)
    families = {family for candidate in candidates for family in candidate.families}
    # A small diversity bonus breaks ties in favour of genuinely different methods,
    # not endless variants of one presumed mechanism.
    return (round(physical_gain + 0.001 * len(families), 12), len(families), tuple(candidate.id for candidate in candidates))


def plan_wave(state: Mapping[str, Any], candidates: Iterable[Mapping[str, Any] | Candidate], *, max_candidates: int = 3) -> dict[str, Any]:
    """Choose the strongest admissible, pairwise-compatible experiment wave.

    Candidate effects are composed multiplicatively: two reductions of 20% and
    30% on the same cost leave 56%, not an impossible negative cost.  The return
    contains rejected candidates too, so an experiment cannot disappear merely
    because a controller did not choose it.
    """
    checked = validate_state(state)
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or not 1 <= max_candidates <= 8:
        raise PlanningRefusal("max_candidates must be an integer from 1 through 8")
    parsed = [item if isinstance(item, Candidate) else Candidate.from_mapping(item) for item in candidates]
    ids = [candidate.id for candidate in parsed]
    if len(ids) != len(set(ids)):
        raise PlanningRefusal("candidate ids must be unique")
    admitted, rejected = [], []
    for candidate in parsed:
        reasons = _admission_reasons(checked, candidate)
        (rejected if reasons else admitted).append((candidate, reasons))
    options: list[tuple[Candidate, ...]] = []
    for width in range(1, min(max_candidates, len(admitted)) + 1):
        for indexed in combinations([candidate for candidate, _ in admitted], width):
            if all(_pair_compatible(left, right) for left, right in combinations(indexed, 2)):
                options.append(indexed)
    if not options:
        raise PlanningRefusal("no candidate is admissible under the frozen model, storage, negative-result, and lease fences")
    selected = max(options, key=lambda option: _score(checked["roofline"], option))
    reduction = _combined_reduction(selected)
    wave = {
        "schema": SCHEMA,
        "status": "PREREGISTRATION_REQUIRED_NOT_EVIDENCE",
        "model_id": checked["model_id"],
        "model_selection_frozen": True,
        "storage_authority": checked["storage"]["authority"],
        "available_model_lane_bytes": checked["storage"]["available_model_lane_bytes"],
        "roofline_deficit": checked["roofline"],
        "measurement": checked["measurement"],
        "selected": [
            {
                "id": candidate.id,
                "families": list(candidate.families),
                "reduces": dict(candidate.reduces),
                "artifact_bytes": candidate.artifact_bytes,
                "temporary_bytes": candidate.temporary_bytes,
                "falsifier": candidate.falsifier,
                "challenger_requirement": "independent complete-token wall-clock, parity, capability, and storage-accounting check",
            }
            for candidate in selected
        ],
        "combined_predicted_reduction": reduction,
        "combined_predicted_remaining_fraction": {dimension: 1.0 - reduction[dimension] for dimension in COST_DIMENSIONS},
        "rejected": [{"id": candidate.id, "reasons": reasons} for candidate, reasons in rejected],
        "score": _score(checked["roofline"], selected)[0],
        "promotion_rule": (
            "Promote only after a clean complete-token before/after wall-clock win, G0-G9 capability gate, fallback=0, and independent challenger verification."
            if checked["measurement"]["absolute_tps_eligible"]
            else "This shared-load wave may only establish an interleaved differential result. Re-run on a clean lease before any absolute TPS, named rung, or production promotion."
        ),
        "retirement_rule": "Retire a failed candidate and preserve its falsifier, physical vector, and reopen condition; do not silently retune it.",
    }
    wave["plan_sha256"] = _digest(wave)
    return wave


def write_wave(path: str | Path, wave: Mapping[str, Any]) -> Path:
    """Seal a generated plan without treating it as an execution receipt."""
    if wave.get("schema") != SCHEMA or not isinstance(wave.get("plan_sha256"), str):
        raise PlanningRefusal("only a generated dynamic wave may be written")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(wave), indent=2, sort_keys=True) + "\n")
    return destination
