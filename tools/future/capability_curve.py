#!/usr/bin/env python3
"""Where is the cliff, found without a human typing 10/20/30/40?

perturbation_workunit runs ONE curve point. This module is the search on
top of it: coarse scan, detect the interval where behaviour changes,
binary-refine that interval to a stated resolution, stop at a point budget.

The caller names component, layer, axis and perturbation type. This module
does not. A cliff located between two samples is an INTERVAL, not a point.
A flat curve is reported as "no cliff found in [lo, hi]", which is a result,
not a failure. Each point records the level it was measured at. Measured
points are cached so a re-run does not repay.

    python3 tools/future/capability_curve.py --build
    python3 -m pytest tools/future/test_capability_curve.py -q
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_receipt  # noqa: E402

RECORDED_BY = "tools/future/capability_curve.py"
RECEIPT_NAME = "CAPABILITY_CURVE.json"
SCHEMA = "hawking.future.capability_curve.v1"
VERSION = 1

IDENTITY_FIELDS = ("component", "layer", "axis", "perturbation_type")
VALUE_KEYS = ("value", "metric", "damage", "score", "cosine", "y")
WORKUNIT_MODULE = "tools.future.perturbation_workunit"
WORKUNIT_RUNNERS = ("run_one", "measure_one", "measure", "point")

# Search structure, not a scientific target. The coarse grid is derived
# from the caller's [lo, hi] by linspace; it is never a typed percentage list.
DEFAULT_N_COARSE = 5
DEFAULT_CLIFF_FRACTION = 0.5
LEVEL_DIGITS = 12

# --build self-proofs use an obviously non-organ identity so the receipt
# cannot be misread as a live sweep the resident did not ask for.
SYNTHETIC_IDENTITY = {
    "component": "SYNTHETIC",
    "layer": "SYNTHETIC",
    "axis": "SYNTHETIC",
    "perturbation_type": "SYNTHETIC",
}

Measure = Callable[[Mapping[str, Any]], Any]


class CurveRefused(RuntimeError):
    """A required input is missing, so the search will not invent one."""


def _require_identity(
    component: Any,
    layer: Any,
    axis: Any,
    perturbation_type: Any,
) -> dict[str, Any]:
    raw = {
        "component": component,
        "layer": layer,
        "axis": axis,
        "perturbation_type": perturbation_type,
    }
    out: dict[str, Any] = {}
    for name in IDENTITY_FIELDS:
        v = raw[name]
        if v is None or (isinstance(v, str) and not v.strip()):
            raise CurveRefused(
                f"{name} is missing; this module supplies the search, not the "
                "target. The caller names the component, layer, axis and "
                "perturbation type"
            )
        out[name] = v
    return out


def _canon_level(level: float) -> float:
    return round(float(level), LEVEL_DIGITS)


def cache_key(identity: Mapping[str, Any], level: float) -> str:
    payload = {
        "component": identity["component"],
        "layer": identity["layer"],
        "axis": identity["axis"],
        "perturbation_type": identity["perturbation_type"],
        "level": _canon_level(level),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def linspace(lo: float, hi: float, n: int) -> list[float]:
    """Inclusive grid derived from the range. Not a typed 10/20/30/40 list."""
    if n < 2:
        raise CurveRefused(
            f"n_coarse={n} cannot scan an interval; need at least 2 points"
        )
    span = hi - lo
    return [lo + span * i / (n - 1) for i in range(n)]


def fmt_interval(lo: float, hi: float) -> str:
    return f"[{lo:g}, {hi:g}]"


def extract_value(raw: Any) -> tuple[float, dict[str, Any]]:
    """Pull a finite curve value out of a measure return. Missing → refuse."""
    if raw is None:
        raise CurveRefused(
            "measure returned nothing; a missing value is not a curve point"
        )
    extra: dict[str, Any] = {}
    if isinstance(raw, Mapping):
        value = None
        for k in VALUE_KEYS:
            if k in raw and raw[k] is not None:
                value = raw[k]
                break
        if value is None:
            raise CurveRefused(
                "measure returned a dict with no value; refusing to default one"
            )
        if "measured_at_level" in raw:
            extra["measured_at_level"] = raw["measured_at_level"]
        raw_value = value
    elif isinstance(raw, bool):
        raise CurveRefused("measure returned a boolean; that is not a curve value")
    elif isinstance(raw, (int, float)):
        raw_value = raw
    else:
        raise CurveRefused(
            f"measure returned {type(raw).__name__}, not a number or a dict"
        )
    try:
        v = float(raw_value)
    except (TypeError, ValueError) as e:
        raise CurveRefused("measure returned a value that is not numeric") from e
    if not math.isfinite(v):
        raise CurveRefused("measure returned a non-finite value")
    return v, extra


def bind_workunit() -> Measure:
    """Adapter onto perturbation_workunit's one-point runner.

    Missing module, or a module with no one-point entry, is a refusal - this
    search will not invent a GPU call. The caller that wants a live point
    injects the callable this returns, or any other measure=.
    """
    try:
        pw = importlib.import_module(WORKUNIT_MODULE)
    except ImportError as e:
        raise CurveRefused(
            "tools/future/perturbation_workunit.py is not importable; "
            "inject measure= so this search does not invent a GPU call"
        ) from e
    for name in WORKUNIT_RUNNERS:
        fn = getattr(pw, name, None)
        if not callable(fn):
            continue

        def _measure(spec: Mapping[str, Any], _fn: Callable[..., Any] = fn) -> Any:
            try:
                return _fn(spec)
            except TypeError:
                return _fn(
                    component=spec["component"],
                    layer=spec["layer"],
                    axis=spec["axis"],
                    perturbation_type=spec["perturbation_type"],
                    level=spec["level"],
                )

        return _measure
    raise CurveRefused(
        "perturbation_workunit is importable but exposes no one-point runner "
        f"(tried {WORKUNIT_RUNNERS}); inject measure="
    )


def _call_measure(measure: Measure, spec: Mapping[str, Any]) -> Any:
    return measure(spec)


def _no_cliff_result(
    *,
    identity: Mapping[str, Any],
    lo: float,
    hi: float,
    points: list[dict[str, Any]],
    n_measured: int,
    n_cache_hits: int,
    budget: int,
    resolution: float,
    n_coarse: int,
    coarse_levels: list[float],
    why: str,
) -> dict[str, Any]:
    message = f"no cliff found in {fmt_interval(lo, hi)}"
    return {
        "cliff_found": False,
        "bracket": None,
        "search_range": [lo, hi],
        "message": message,
        "why": why,
        "points": points,
        "n_measured": n_measured,
        "n_cache_hits": n_cache_hits,
        "budget": budget,
        "budget_honoured": n_measured <= budget,
        "resolution": resolution,
        "resolution_met": False,
        "n_coarse": n_coarse,
        "coarse_levels": coarse_levels,
        "detected_interval": None,
        **dict(identity),
    }


def sweep(
    *,
    component: Any,
    layer: Any,
    axis: Any,
    perturbation_type: Any,
    lo: Any,
    hi: Any,
    resolution: Any,
    budget: Any,
    measure: Measure | None = None,
    n_coarse: int = DEFAULT_N_COARSE,
    cliff_fraction: float = DEFAULT_CLIFF_FRACTION,
    cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Adaptive capability curve: coarse scan, detect, binary-refine, stop.

    Returns a bracketing interval when a cliff is found, or an explicit
    no-cliff result over the whole range. Never a single number. Never a
    chosen organ - identity is required and not defaulted.
    """
    identity = _require_identity(component, layer, axis, perturbation_type)
    if measure is None:
        raise CurveRefused(
            "measure is missing; inject a callable or bind_workunit(). "
            "This module will not default to a GPU call"
        )
    if lo is None or hi is None:
        raise CurveRefused(
            "search range [lo, hi] is missing; refusing to invent one"
        )
    try:
        lo_f = float(lo)
        hi_f = float(hi)
    except (TypeError, ValueError) as e:
        raise CurveRefused("search range is not numeric") from e
    if not (math.isfinite(lo_f) and math.isfinite(hi_f)):
        raise CurveRefused("search range is not finite")
    if lo_f >= hi_f:
        raise CurveRefused(
            f"search range {fmt_interval(lo_f, hi_f)} is empty; lo must be < hi"
        )
    if resolution is None:
        raise CurveRefused(
            "resolution is missing; a bound that is not stated is not a bound"
        )
    try:
        res = float(resolution)
    except (TypeError, ValueError) as e:
        raise CurveRefused("resolution is not numeric") from e
    if not math.isfinite(res) or res <= 0:
        raise CurveRefused("resolution must be a positive number of level units")
    if budget is None:
        raise CurveRefused(
            "point budget is missing; refinement is bounded by a budget, "
            "not run forever"
        )
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise CurveRefused("point budget is not an integer")
    bud = int(budget)
    if bud < 2:
        raise CurveRefused(
            f"point budget {bud} cannot measure two ends of an interval"
        )
    try:
        n_c = int(n_coarse)
    except (TypeError, ValueError) as e:
        raise CurveRefused("n_coarse is not an integer") from e
    if n_c < 2:
        raise CurveRefused(
            f"n_coarse={n_c} cannot scan an interval; need at least 2 points"
        )
    try:
        frac = float(cliff_fraction)
    except (TypeError, ValueError) as e:
        raise CurveRefused("cliff_fraction is not numeric") from e
    if not math.isfinite(frac) or frac < 0 or frac > 1:
        raise CurveRefused("cliff_fraction must be in [0, 1]")

    if cache is None:
        cache = {}

    coarse_levels = linspace(lo_f, hi_f, n_c)
    need_new = sum(
        1 for lv in coarse_levels if cache_key(identity, lv) not in cache
    )
    if need_new > bud:
        raise CurveRefused(
            f"coarse scan needs {need_new} new points and budget is {bud}; "
            "refusing to silently shrink the scan"
        )

    n_measured = 0
    n_cache_hits = 0
    points: list[dict[str, Any]] = []
    seen: set[str] = set()

    def pay(level: float, source: str) -> dict[str, Any] | None:
        nonlocal n_measured, n_cache_hits
        key = cache_key(identity, level)
        if key in cache:
            pt = dict(cache[key])
            pt["cached"] = True
            n_cache_hits += 1
            if key not in seen:
                seen.add(key)
                points.append(pt)
            return pt
        if n_measured >= bud:
            return None
        spec = {**identity, "level": float(level)}
        value, extra = extract_value(_call_measure(measure, spec))
        pt = {
            **identity,
            "level": float(level),
            "value": value,
            "source": source,
            "cached": False,
        }
        pt.update(extra)
        cache[key] = {k: v for k, v in pt.items() if k != "cached"}
        n_measured += 1
        seen.add(key)
        points.append(pt)
        return pt

    coarse_pts: list[dict[str, Any]] = []
    for lv in coarse_levels:
        pt = pay(lv, "coarse")
        if pt is None:
            raise CurveRefused(
                "coarse scan ran out of budget mid-grid; the need_new guard "
                "should have refused before any point was taken"
            )
        coarse_pts.append(pt)

    ordered = sorted(coarse_pts, key=lambda p: p["level"])
    values = [float(p["value"]) for p in ordered]
    span = max(values) - min(values)
    best_i = 0
    best_delta = -1.0
    for i in range(len(ordered) - 1):
        delta = abs(float(ordered[i + 1]["value"]) - float(ordered[i]["value"]))
        if delta > best_delta:
            best_delta = delta
            best_i = i

    if span == 0.0 or best_delta < frac * span:
        why = (
            "the curve is flat across the coarse scan: every measured value "
            f"is {values[0]}"
            if span == 0.0
            else (
                f"the largest adjacent jump is {best_delta:g} of span "
                f"{span:g}, below cliff_fraction {frac:g}; the change is "
                "spread across the range rather than concentrated"
            )
        )
        return _no_cliff_result(
            identity=identity,
            lo=lo_f,
            hi=hi_f,
            points=points,
            n_measured=n_measured,
            n_cache_hits=n_cache_hits,
            budget=bud,
            resolution=res,
            n_coarse=n_c,
            coarse_levels=coarse_levels,
            why=why,
        )

    left = ordered[best_i]
    right = ordered[best_i + 1]
    a = float(left["level"])
    b = float(right["level"])
    va = float(left["value"])
    vb = float(right["value"])
    detected = {
        "lo": a,
        "hi": b,
        "delta": best_delta,
        "left_value": va,
        "right_value": vb,
    }

    while (b - a) > res and n_measured < bud:
        mid = (a + b) / 2.0
        if mid <= a or mid >= b:
            break
        pt = pay(mid, "refine")
        if pt is None:
            break
        vm = float(pt["value"])
        # Keep the half that still holds the jump. Ties go left, so the
        # result is deterministic when a ramp splits evenly.
        if abs(vm - va) >= abs(vb - vm):
            b, vb = float(pt["level"]), vm
        else:
            a, va = float(pt["level"]), vm

    bracket = {"lo": a, "hi": b}
    return {
        "cliff_found": True,
        "bracket": bracket,
        "search_range": [lo_f, hi_f],
        "message": f"cliff bracketed in {fmt_interval(a, b)}",
        "why": (
            "largest coarse adjacent jump was "
            f"{best_delta:g} of span {span:g}, then binary-refined"
        ),
        "points": points,
        "n_measured": n_measured,
        "n_cache_hits": n_cache_hits,
        "budget": bud,
        "budget_honoured": n_measured <= bud,
        "resolution": res,
        "resolution_met": (b - a) <= res,
        "n_coarse": n_c,
        "coarse_levels": coarse_levels,
        "detected_interval": detected,
        **dict(identity),
    }


def _step_at(threshold: float, *, pre: float = 1.0, post: float = 0.0) -> Measure:
    def _measure(spec: Mapping[str, Any]) -> float:
        return pre if float(spec["level"]) < threshold else post

    return _measure


def _constant(value: float) -> Measure:
    def _measure(_spec: Mapping[str, Any]) -> float:
        return value

    return _measure


def synthetic_proofs() -> dict[str, Any]:
    """Self-proofs on injected functions. Not a live organ. Not hardware."""
    step_at = 0.37
    step = sweep(
        **SYNTHETIC_IDENTITY,
        lo=0.0,
        hi=1.0,
        resolution=0.02,
        budget=16,
        n_coarse=5,
        measure=_step_at(step_at),
        cache={},
    )
    flat = sweep(
        **SYNTHETIC_IDENTITY,
        lo=0.0,
        hi=1.0,
        resolution=0.02,
        budget=16,
        n_coarse=5,
        measure=_constant(0.5),
        cache={},
    )
    calls: list[float] = []

    def counted(spec: Mapping[str, Any]) -> float:
        calls.append(float(spec["level"]))
        return 1.0 if float(spec["level"]) < step_at else 0.0

    budgeted = sweep(
        **SYNTHETIC_IDENTITY,
        lo=0.0,
        hi=1.0,
        resolution=0.001,
        budget=7,
        n_coarse=5,
        measure=counted,
        cache={},
    )
    step_box = step["bracket"]
    contains = (
        step_box is not None
        and float(step_box["lo"]) <= step_at <= float(step_box["hi"])
    )
    return {
        "kind": "SYNTHETIC",
        "not_a_live_organ": True,
        "identity_used": dict(SYNTHETIC_IDENTITY),
        "step": {
            "true_step": step_at,
            "cliff_found": step["cliff_found"],
            "bracket": step["bracket"],
            "contains_step": contains,
            "n_measured": step["n_measured"],
            "resolution_met": step["resolution_met"],
            "message": step["message"],
        },
        "flat": {
            "cliff_found": flat["cliff_found"],
            "bracket": flat["bracket"],
            "n_measured": flat["n_measured"],
            "message": flat["message"],
        },
        "budget": {
            "budget": 7,
            "n_measured": len(calls),
            "result_n_measured": budgeted["n_measured"],
            "honoured": len(calls) <= 7,
            "resolution_met": budgeted["resolution_met"],
            "cliff_found": budgeted["cliff_found"],
        },
    }


def build() -> dict[str, Any]:
    proofs = synthetic_proofs()
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "lane": "D2",
        "evidence_class": "STATIC_ONLY",
        "gpu_authority": False,
        "question": "Where is the cliff, found without a human typing 10/20/30/40?",
        "what_this_module_is": (
            "search machinery. The caller names the component, layer, axis "
            "and perturbation type; this module finds the bracketing interval "
            "of a behaviour change, or reports that the curve is flat"
        ),
        "what_this_module_does_not_do": (
            "it does not choose which representation or which component to "
            "test next. That is the resident's hypothesis. --build runs only "
            "synthetic self-proofs under the SYNTHETIC identity"
        ),
        "one_point_runner": {
            "module": "tools/future/perturbation_workunit.py",
            "bind": "bind_workunit()",
            "role": "one curve point; this module is the sweep on top of it",
            "missing_is_a_refusal": True,
        },
        "algorithm": {
            "coarse_scan": (
                "n_coarse inclusive linspace points over the caller's [lo, hi], "
                "never a typed percentage list"
            ),
            "detect": (
                "the adjacent coarse pair with the largest |delta|; a cliff "
                "only if that jump is at least cliff_fraction of the value "
                "span. A zero span is flat"
            ),
            "refine": (
                "binary split of the detected interval, keeping the half that "
                "still holds the jump, until width <= resolution or the point "
                "budget is spent"
            ),
            "result": (
                "a bracketing interval [lo, hi] between two samples, or "
                "'no cliff found in [lo, hi]'"
            ),
            "cache": (
                "points are keyed by component, layer, axis, perturbation "
                "type and level; a re-run does not repay"
            ),
            "defaults": {
                "n_coarse": DEFAULT_N_COARSE,
                "cliff_fraction": DEFAULT_CLIFF_FRACTION,
            },
        },
        "synthetic_proofs": proofs,
        "live_sweep": {
            "ran": False,
            "component": None,
            "why": (
                "this module does not choose a component; the resident does. "
                "A live sweep is sweep(..., measure=bind_workunit()) with "
                "the resident's identity and range"
            ),
        },
        "does_not_choose_science": True,
        "claim_boundary": (
            "Static sidecar artifact. No hardware measurement. The numbers "
            "in synthetic_proofs are injected step/flat/budget functions, "
            "not a GPU run and not a chosen organ. A live cliff location "
            "does not exist in this receipt because none was measured."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args(argv)
    doc = build()
    if args.build:
        print(write_receipt(RECEIPT_NAME, doc, RECORDED_BY))
        return 0
    print(json.dumps(
        {k: doc[k] for k in (
            "question", "synthetic_proofs", "does_not_choose_science",
            "live_sweep",
        )},
        indent=1,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
