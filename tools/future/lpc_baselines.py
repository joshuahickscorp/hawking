"""LPC_BASELINES — honest baselines with calibrated uncertainty and ABSTENTION.

A learned compiler that cannot say "I do not know" will confidently mispredict
outside its support. These baselines refuse to extrapolate:

  * nearest measured neighbour over the LPC row space, with an explicit metric
  * a transparent hand-written rule cost model
  * held-out splits by architecture, organ, and device
  * an authority helper: PROTECTED_ABSOLUTE always outranks a model

Predict returns (value, uncertainty) or ABSTAINS. Abstention is a first-class
outcome, not an error. Null inputs are never treated as zero.

    python3 tools/future/lpc_baselines.py --build
    python3 tools/future/lpc_baselines.py --selftest
"""
from __future__ import annotations

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from tools.future._common import git, write_receipt
from tools.future.experiment_receipt import attach, input_ref
from tools.future.lpc_dataset import (
    CONTAMINATION_CLASSES,
    NUMERIC_FIELDS,
    REQUIRED_FIELDS,
    as_numeric,
    row_template,
    validate_row,
)

RECEIPT = "LPC_BASELINES.json"
SCHEMA = "hawking.future.lpc_baselines.v1"

IDENTITY_AXES: tuple[str, ...] = tuple(
    name for name in REQUIRED_FIELDS if name not in NUMERIC_FIELDS
)

# A total categorical mismatch scores 1.0; this radius is below that, so a
# query that shares no identity with the training support must abstain.
SUPPORT_RADIUS = 0.60
MIN_COMPARED_AXES = 4
UNCERTAINTY_FLOOR = 1e-9

# Transparent rule: cost_units = w_a * active_bytes + w_d * dispatches + w_s * synchronization.
# Coefficients are declared model parameters, not measured roofs, and are not
# nanoseconds. Any null input ABSTAINS (null is not 0).
RULE_WEIGHTS: dict[str, float] = {
    "active_bytes": 1.0,
    "dispatches": 1000.0,
    "synchronization": 100.0,
}

SPLIT_AXES: dict[str, str] = {
    "architecture": "model",
    "organ": "organ_fingerprint",
    "device": "machine_genome",
}

ABSTAIN = "ABSTAIN"
PREDICTED = "PREDICTED"


class AuthorityError(ValueError):
    """A caller tried to treat non-protected evidence as protected authority."""


@dataclass(frozen=True)
class Prediction:
    status: Literal["PREDICTED", "ABSTAIN"]
    value: float | None = None
    uncertainty: float | None = None
    reason: str | None = None
    neighbour_id: str | None = None
    distance: float | None = None
    method: str | None = None
    compared_axes: int = 0

    def as_tuple(self) -> tuple[float, float] | Literal["ABSTAIN"]:
        if self.status == ABSTAIN:
            return ABSTAIN
        assert self.value is not None and self.uncertainty is not None
        return (self.value, self.uncertainty)


@dataclass(frozen=True)
class Split:
    axis: str
    field: str
    holdout_key: str
    train_ids: tuple[str, ...]
    holdout_ids: tuple[str, ...]


def _canonical(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return value or None
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text or None


def _row_id(row: Mapping[str, Any], fallback: int) -> str:
    value = row.get("row_id")
    return str(value) if value else f"row:{fallback:04d}"


def row_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
    """Explicit metric on the LPC row space.

    d = Hamming(identity axes that both rows bind)
      + 0.5 * mean relative L1(numeric axes that both rows measure)
      + 0.25 * (1 - coverage)

    Null is not a category and is not the number 0. Axes that either side
    left unbound are skipped, then penalised through the coverage term so
    sparse rows are not treated as close. Returns None when nothing is
    comparable.
    """
    cat_compared = 0
    cat_mismatch = 0
    for axis in IDENTITY_AXES:
        a = _canonical(left.get(axis))
        b = _canonical(right.get(axis))
        if a is None or b is None:
            continue
        cat_compared += 1
        if a != b:
            cat_mismatch += 1
    num_compared = 0
    num_l1 = 0.0
    for axis in NUMERIC_FIELDS:
        a = as_numeric(left, axis)
        b = as_numeric(right, axis)
        if a is None or b is None:
            continue
        scale = max(abs(a), abs(b), 1.0)
        num_l1 += abs(float(a) - float(b)) / scale
        num_compared += 1
    compared = cat_compared + num_compared
    if compared == 0:
        return None
    cat_term = (cat_mismatch / cat_compared) if cat_compared else 0.0
    num_term = (num_l1 / num_compared) if num_compared else 0.0
    coverage = compared / float(len(IDENTITY_AXES) + len(NUMERIC_FIELDS))
    return cat_term + 0.5 * num_term + 0.25 * (1.0 - coverage)


def _compatible_neighbour(query: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    """Diagnostic evidence must not support a protected query."""
    q = query.get("contamination_class")
    r = row.get("contamination_class")
    if q == "PROTECTED_ABSOLUTE" and r == "DIAGNOSTIC_RELATIVE":
        return False
    return True


def nearest_measured_neighbour(
    query: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    label_field: str = "latency",
) -> dict[str, Any] | None:
    """Closest row that actually measured `label_field`. Null labels are skipped."""
    best: dict[str, Any] | None = None
    for i, row in enumerate(rows):
        if as_numeric(row, label_field) is None:
            continue
        if not _compatible_neighbour(query, row):
            continue
        distance = row_distance(query, row)
        if distance is None:
            continue
        candidate = {
            "row": row,
            "row_id": _row_id(row, i),
            "distance": distance,
            "label": float(as_numeric(row, label_field)),  # type: ignore[arg-type]
            "compared_axes": _compared_axes(query, row),
        }
        if best is None or candidate["distance"] < best["distance"]:
            best = candidate
        elif candidate["distance"] == best["distance"] and candidate["row_id"] < best["row_id"]:
            best = candidate
    return best


def _compared_axes(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    n = 0
    for axis in IDENTITY_AXES:
        if _canonical(left.get(axis)) is not None and _canonical(right.get(axis)) is not None:
            n += 1
    for axis in NUMERIC_FIELDS:
        if as_numeric(left, axis) is not None and as_numeric(right, axis) is not None:
            n += 1
    return n


def _uncertainty(value: float, distance: float) -> float:
    return max(UNCERTAINTY_FLOOR, abs(value) * (distance + 0.05))


def _abstain(reason: str, **extra: Any) -> Prediction:
    return Prediction(status=ABSTAIN, value=None, uncertainty=None, reason=reason, **extra)


def predict_nearest(
    query: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    label_field: str = "latency",
) -> Prediction:
    neighbour = nearest_measured_neighbour(query, rows, label_field=label_field)
    if neighbour is None:
        return _abstain("no_measured_neighbour", method="nearest")
    if neighbour["compared_axes"] < MIN_COMPARED_AXES:
        return _abstain(
            "too_few_compared_axes",
            neighbour_id=neighbour["row_id"],
            distance=neighbour["distance"],
            method="nearest",
            compared_axes=neighbour["compared_axes"],
        )
    if neighbour["distance"] > SUPPORT_RADIUS:
        return _abstain(
            "outside_support",
            neighbour_id=neighbour["row_id"],
            distance=neighbour["distance"],
            method="nearest",
            compared_axes=neighbour["compared_axes"],
        )
    value = neighbour["label"]
    return Prediction(
        status=PREDICTED,
        value=value,
        uncertainty=_uncertainty(value, neighbour["distance"]),
        reason=None,
        neighbour_id=neighbour["row_id"],
        distance=neighbour["distance"],
        method="nearest",
        compared_axes=neighbour["compared_axes"],
    )


def rule_cost(row: Mapping[str, Any]) -> Prediction:
    """Hand-written linear cost. Any null input ABSTAINS; null is not 0."""
    missing = [
        name
        for name in ("active_bytes", "dispatches", "synchronization")
        if as_numeric(row, name) is None
    ]
    if missing:
        return _abstain(
            "rule_cost_null_inputs:" + ",".join(missing),
            method="rule",
        )
    value = (
        RULE_WEIGHTS["active_bytes"] * float(as_numeric(row, "active_bytes"))  # type: ignore[arg-type]
        + RULE_WEIGHTS["dispatches"] * float(as_numeric(row, "dispatches"))  # type: ignore[arg-type]
        + RULE_WEIGHTS["synchronization"] * float(as_numeric(row, "synchronization"))  # type: ignore[arg-type]
    )
    # Uncalibrated coefficients: report 100% relative uncertainty.
    return Prediction(
        status=PREDICTED,
        value=value,
        uncertainty=max(UNCERTAINTY_FLOOR, abs(value)),
        method="rule",
        reason="uncalibrated_rule_coefficients",
        compared_axes=3,
    )


def predict(
    query: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]] | None = None,
    *,
    method: str = "nearest",
    label_field: str = "latency",
) -> Prediction:
    if method == "rule":
        return rule_cost(query)
    if method == "nearest":
        if rows is None:
            return _abstain("no_training_rows", method="nearest")
        return predict_nearest(query, rows, label_field=label_field)
    raise ValueError(f"unknown method {method!r}")


def held_out_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    axis: str,
) -> list[Split]:
    """Leave-one-value-out splits. Nulls on the axis never form a group and never leak."""
    if axis not in SPLIT_AXES:
        raise ValueError(f"unknown split axis {axis!r}; expected {sorted(SPLIT_AXES)}")
    field = SPLIT_AXES[axis]
    groups: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for i, row in enumerate(rows):
        key = _canonical(row.get(field))
        if key is None:
            continue
        groups.setdefault(key, []).append((_row_id(row, i), row))
    splits: list[Split] = []
    for hold_key in sorted(groups):
        holdout_ids = tuple(sorted(rid for rid, _ in groups[hold_key]))
        train_ids = tuple(
            sorted(
                rid
                for other, items in groups.items()
                if other != hold_key
                for rid, _ in items
            )
        )
        splits.append(
            Split(
                axis=axis,
                field=field,
                holdout_key=hold_key,
                train_ids=train_ids,
                holdout_ids=holdout_ids,
            )
        )
    return splits


def split_has_no_leak(split: Split) -> bool:
    return set(split.train_ids).isdisjoint(split.holdout_ids)


def resolve_authority(
    model_prediction: Prediction | Mapping[str, Any],
    protected_measurement: Mapping[str, Any],
) -> dict[str, Any]:
    """PROTECTED_ABSOLUTE measurement always wins. The model cannot override it.

    A null protected value does not license the model to fill the hole.
    A DIAGNOSTIC_RELATIVE or STATIC_ONLY payload is refused: this helper
    is not a laundering path.
    """
    if not isinstance(protected_measurement, Mapping):
        raise AuthorityError("protected_measurement must be an object")
    klass = protected_measurement.get("contamination_class")
    if klass != "PROTECTED_ABSOLUTE":
        raise AuthorityError(
            f"measurement contamination_class={klass!r} is not PROTECTED_ABSOLUTE; "
            "this helper will not promote diagnostic or static evidence"
        )
    value = protected_measurement.get("value")
    if value is None:
        value = as_numeric(protected_measurement, "latency")
    if value is None:
        raise AuthorityError(
            "PROTECTED_ABSOLUTE measurement has a null value; disk state is still "
            "unknown and the model is not allowed to fill it"
        )
    if isinstance(model_prediction, Prediction):
        model_status = model_prediction.status
        model_value = model_prediction.value
        model_uncertainty = model_prediction.uncertainty
    elif isinstance(model_prediction, Mapping):
        model_status = str(model_prediction.get("status") or PREDICTED)
        model_value = model_prediction.get("value")
        model_uncertainty = model_prediction.get("uncertainty")
    else:
        raise AuthorityError("model_prediction must be a Prediction or mapping")
    return {
        "value": value,
        "source": "PROTECTED_ABSOLUTE",
        "model_prediction_ignored": True,
        "model_status": model_status,
        "model_value": model_value,
        "model_uncertainty": model_uncertainty,
        "rule": "protected deterministic evidence decides; models propose",
    }


def describe() -> dict[str, Any]:
    """Structural description sealed into receipts. No hardware numbers."""
    return {
        "nearest_measured_neighbour": {
            "metric": (
                "d = Hamming(bound identity axes) + 0.5 * mean relative L1"
                "(shared numeric axes) + 0.25 * (1 - coverage). "
                "Null is skipped, never treated as 0."
            ),
            "identity_axes": list(IDENTITY_AXES),
            "numeric_axes": list(NUMERIC_FIELDS),
            "support_radius": SUPPORT_RADIUS,
            "min_compared_axes": MIN_COMPARED_AXES,
            "outside_support": ABSTAIN,
        },
        "rule_cost_model": {
            "formula": (
                "cost_units = w_active_bytes * active_bytes + w_dispatches * "
                "dispatches + w_synchronization * synchronization"
            ),
            "weights": dict(RULE_WEIGHTS),
            "units": "declared cost units, not nanoseconds, not a measurement",
            "null_input": ABSTAIN,
        },
        "uncertainty": {
            "nearest": "max(floor, |value| * (distance + 0.05)); never 0 for k=1",
            "rule": "100% relative; coefficients are uncalibrated",
        },
        "abstention": (
            "First-class outcome. Fired when there is no measured neighbour, "
            "compared axes < min, distance > support radius, or a rule input is null."
        ),
        "held_out_splits": dict(SPLIT_AXES),
        "authority_rule": (
            "resolve_authority(model, protected) always returns the "
            "PROTECTED_ABSOLUTE measurement. A confident disagreeing model loses. "
            "A null protected value does not license the model to fill it. "
            "DIAGNOSTIC_RELATIVE is refused."
        ),
        "contamination_classes": list(CONTAMINATION_CLASSES),
    }


def _complete_fixture(**overrides: Any) -> dict[str, Any]:
    base = dict(
        model="qwen3.8-27b-sealed-3.14",
        organ_fingerprint="mlp",
        representation="native-packed",
        machine_genome="fixture-m3-ultra",
        physical_graph_identity="fixture-graph-1",
        backend="metal",
        layout="row-major",
        tile="tg64",
        grouping="gqa",
        fusion="none",
        persistent_resources="resident-weights",
        active_bytes=1_000_000,
        resident_bytes=20_000_000,
        dispatches=100,
        synchronization=50,
        latency=1_000_000.0,
        complete_token_effect=True,
        contamination_class="PROTECTED_ABSOLUTE",
        capability=True,
        source="FIXTURE_SYNTHETIC",
        absence_reasons={},
    )
    base.update(overrides)
    return row_template(reasons_for_missing=None, **base)


def selftest() -> None:
    a = _complete_fixture(row_id="a", organ_fingerprint="mlp", latency=10.0)
    b = _complete_fixture(row_id="b", organ_fingerprint="attention", latency=20.0)
    assert validate_row(a)["complete"] is True
    assert row_distance(a, a) is not None and row_distance(a, a) < 0.05
    assert abs((row_distance(a, b) or 0) - (row_distance(b, a) or 0)) < 1e-12

    near = predict_nearest(a, [a, b])
    assert near.status == PREDICTED and near.value == 10.0

    far = _complete_fixture(
        row_id="far",
        model="not-a-hawking-model",
        organ_fingerprint="vacuum-chamber",
        representation="imaginary-nr",
        machine_genome="other-soc",
        physical_graph_identity="no-such-graph",
        backend="not-a-backend",
        layout="scrambled",
        tile="tg1",
        grouping="none",
        fusion="everything",
        persistent_resources="none",
        active_bytes=1e12,
        resident_bytes=1e12,
        dispatches=1e9,
        synchronization=1e9,
        latency=None,
        absence_reasons={"latency": "UNMEASURED"},
        contamination_class="PROTECTED_ABSOLUTE",
    )
    d = row_distance(a, far)
    assert d is not None and d > SUPPORT_RADIUS
    refusal = predict_nearest(far, [a, b])
    assert refusal.status == ABSTAIN, refusal
    assert refusal.value is None
    assert refusal.reason == "outside_support"

    null_active = _complete_fixture(active_bytes=None, absence_reasons={"active_bytes": "UNMEASURED"})
    rule_refused = rule_cost(null_active)
    assert rule_refused.status == ABSTAIN
    assert rule_refused.value is None
    assert "active_bytes" in (rule_refused.reason or "")

    measured = {"contamination_class": "PROTECTED_ABSOLUTE", "value": 9.0}
    confident = Prediction(status=PREDICTED, value=1.0, uncertainty=UNCERTAINTY_FLOOR, method="nearest")
    decided = resolve_authority(confident, measured)
    assert decided["value"] == 9.0
    assert decided["source"] == "PROTECTED_ABSOLUTE"
    assert decided["model_prediction_ignored"] is True

    splits = held_out_splits([a, b], axis="organ")
    assert splits and all(split_has_no_leak(s) for s in splits)


def build() -> Any:
    selftest()
    doc: dict[str, Any] = {
        "schema": SCHEMA,
        "version": 1,
        "purpose": (
            "Honest LPC baselines: nearest measured neighbour, rule cost model, "
            "uncertainty, ABSTENTION, held-out splits, and the authority rule "
            "that a protected measurement always outranks a model."
        ),
        "head": git("rev-parse", "HEAD"),
        "baselines": describe(),
        "selftest": "passed",
        "claim_boundary": (
            "Static sidecar artifact. No hardware measurement. Rule weights are "
            "declared model parameters in cost units, not measured nanoseconds."
        ),
    }
    doc = attach(
        doc,
        producer="tools/future/lpc_baselines.py",
        location="receipts/future/LPC_BASELINES.json",
        inputs=[input_ref("rule_weights", RULE_WEIGHTS)],
        claim=(
            "Nearest-neighbour and rule-cost LPC baselines abstain outside "
            "support; a protected measurement always outranks a model."
        ),
        verdict="ACCEPT",
        evidence_tier="STATIC",
        scope="synthetic fixture rows; no hardware measurement",
        facts=[
            {"claim": "selftest passed", "source": "lpc_baselines.selftest"},
            {"claim": "null numeric inputs abstain rather than become zero"},
        ],
        hypotheses=[],
        negative_controls=[
            {
                "id": "null_active_bytes_is_not_zero",
                "what": "rule_cost abstains when active_bytes is null",
            },
            {
                "id": "protected_absolute_outranks_model",
                "what": "a confident disagreeing model loses to PROTECTED_ABSOLUTE",
            },
        ],
        failures=[],
        resource_usage={"gpu_authority": False},
        qualification="STATIC_ONLY; rule weights are declared cost units, not ns",
        contamination=[],
        uncertainty=[
            "nearest: max(floor, |value| * (distance + 0.05)); never 0 for k=1",
            "rule: 100% relative; coefficients are uncalibrated",
        ],
        falsifier=(
            "a null input treated as zero, a query outside support that does "
            "not abstain, or a model prediction outranking PROTECTED_ABSOLUTE"
        ),
        next_actions=[],
        receipts=["receipts/future/LPC_BASELINES.json"],
        tests={"selftest": "passed"},
    )
    return write_receipt(RECEIPT, doc, "tools/future/lpc_baselines.py")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest and not a.build:
        selftest()
        print("selftest ok")
        return 0
    out = build()
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
