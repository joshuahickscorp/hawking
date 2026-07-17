#!/usr/bin/env python3.12
"""Empirical runtime + ETA model for the Hawking successor control plane.

Master goal 12.1 / 12.4: fit runtime from real evidence and publish a conservative /
expected / optimistic ETA with confidence, critical path, empirical basis, and the
unknowns that block calibration.

The controlling doctrine here is honesty about what the evidence supports:

  - There is NO single global seconds-per-billion constant. Runtime per billion params
    differs by (branch, phase): a `diagnose` probe on a small parent behaves nothing like
    a full `materialize` on a giant. `fit_runtime` fits a SEPARATE distribution per
    (branch, phase) segment, reports the per-segment sample count and the p10/p50/p90
    spread, and `RuntimeModel.global_seconds_per_billion()` fails closed by design.

  - A segment is only `calibrated` once it has at least `min_samples_calibrated` real
    observations. Predicting on an uncalibrated segment returns an explicit
    `calibrated=False` with no fabricated band.

  - A 120B / giant program, or any program flagged unwired, is NEVER given a calibrated
    ETA even if some lookalike segment happens to be calibrated. `project_eta` marks it
    'uncalibrated - no ETA' and records why in `unknowns`.

This module is additive, default-off scaffolding. It reads observation records handed to
it; it launches no compute, adopts no pids, and writes only under the successor namespace
(`reports/condense/event_horizon_successor/`), never the campaign namespace.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from eco_common import (  # noqa: E402
    EcoError, seal_field, sealed, now_iso, atomic_write_json, read_json_safe,
    is_sha256, canonical_bytes, eco_state_root, repo_root,
)

ETA_MODEL_SCHEMA = "hawking.successor.eta_model.v1"
ETA_PROJECTION_SCHEMA = "hawking.successor.eta_projection.v1"

# Marker string emitted for any program whose segment is not calibrated (or is a giant).
NO_ETA = "uncalibrated - no ETA"


class EtaError(EcoError):
    """Fail-closed error in the empirical ETA model."""


@dataclasses.dataclass(frozen=True)
class Config:
    """Frozen knobs for the empirical runtime fit and the ETA projection."""

    # Minimum real observations before a (branch, phase) segment is trusted for prediction.
    min_samples_calibrated: int = 3
    # Any program at or above this stored-param size is treated as unwired giant execution:
    # never assigned a calibrated ETA, regardless of lookalike segments. 120B >= 100.0.
    giant_params_b: float = 100.0
    # Lower/upper percentiles reported as optimistic / conservative bounds.
    p_low: float = 10.0
    p_mid: float = 50.0
    p_high: float = 90.0
    # A single global seconds-per-billion constant is forbidden. This is never overridden.
    reject_global_constant: bool = True


def default_config() -> Config:
    return Config()


# ── percentile helper (linear interpolation, no numpy dependency) ──────────────────────
def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted, non-empty list."""
    if not sorted_values:
        raise EtaError("percentile of empty sample")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    if pct <= 0:
        return float(sorted_values[0])
    if pct >= 100:
        return float(sorted_values[-1])
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    frac = rank - lo
    if lo + 1 >= len(sorted_values):
        return float(sorted_values[-1])
    return float(sorted_values[lo] + frac * (sorted_values[lo + 1] - sorted_values[lo]))


def _segment_key(branch: str, phase: str) -> str:
    return f"{branch}::{phase}"


def _coerce_pos_float(value: Any, field: str, ctx: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise EtaError(f"observation {ctx}: {field} not a number: {value!r}") from exc
    if out != out or out in (float("inf"), float("-inf")):
        raise EtaError(f"observation {ctx}: {field} not finite: {value!r}")
    if out <= 0:
        raise EtaError(f"observation {ctx}: {field} must be > 0: {value!r}")
    return out


class RuntimeModel:
    """Per-(branch, phase) seconds-per-billion-params model with p10/p50/p90 spread.

    Built by `fit_runtime`. Prediction is segment-scoped; there is deliberately no method
    that collapses all segments into one constant.
    """

    def __init__(self, segments: dict[str, dict[str, Any]], config: Config,
                 n_observations: int):
        self.segments = segments
        self.config = config
        self.n_observations = n_observations

    # -- calibration status --------------------------------------------------------------
    def is_calibrated(self, branch: str, phase: str) -> bool:
        seg = self.segments.get(_segment_key(branch, phase))
        return bool(seg) and seg["count"] >= self.config.min_samples_calibrated

    def calibrated_segments(self) -> list[str]:
        return sorted(k for k, s in self.segments.items()
                      if s["count"] >= self.config.min_samples_calibrated)

    def uncalibrated_segments(self) -> list[str]:
        return sorted(k for k, s in self.segments.items()
                      if s["count"] < self.config.min_samples_calibrated)

    # -- the refusal that keeps this honest ----------------------------------------------
    def global_seconds_per_billion(self) -> float:
        """Refused by design. Runtime per billion params is not one constant across
        differing branch/phase; collapsing segments would fabricate a number the evidence
        does not support."""
        raise EtaError(
            "global seconds-per-billion is refused: fit is per (branch, phase) segment; "
            "use predict(branch, phase, ...) instead")

    # -- prediction ----------------------------------------------------------------------
    def predict(self, branch: str, phase: str, rate: float | None,
                residency: str | None, params_b: float) -> dict[str, Any]:
        """Predict wall-seconds for one program on one segment.

        Returns {p10, p50, p90, calibrated, segment, count, basis}. On an uncalibrated
        segment the band is None and `calibrated` is False (no fabricated estimate).
        """
        key = _segment_key(branch, phase)
        seg = self.segments.get(key)
        pb = _coerce_pos_float(params_b, "params_b", f"predict[{key}]")
        if seg is None:
            return {
                "segment": key, "calibrated": False, "count": 0,
                "p10": None, "p50": None, "p90": None,
                "reason": "no observations for this (branch, phase) segment",
            }
        calibrated = seg["count"] >= self.config.min_samples_calibrated
        result = {
            "segment": key,
            "calibrated": calibrated,
            "count": seg["count"],
            "params_b": pb,
            "rate": rate,
            "residency": residency,
            "spb_p10": seg["spb_p10"],
            "spb_p50": seg["spb_p50"],
            "spb_p90": seg["spb_p90"],
            "p10": round(seg["spb_p10"] * pb, 3),
            "p50": round(seg["spb_p50"] * pb, 3),
            "p90": round(seg["spb_p90"] * pb, 3),
        }
        if not calibrated:
            # keep the arithmetic visible but do not present it as a trusted band
            result["p10"] = result["p50"] = result["p90"] = None
            result["reason"] = (
                f"segment has {seg['count']} < {self.config.min_samples_calibrated} "
                f"observations")
        return result

    # -- serialization -------------------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        body = {
            "schema": ETA_MODEL_SCHEMA,
            "generated_at": now_iso(),
            "n_observations": self.n_observations,
            "config": dataclasses.asdict(self.config),
            "segments": self.segments,
            "calibrated_segments": self.calibrated_segments(),
            "uncalibrated_segments": self.uncalibrated_segments(),
            "global_constant": None,
            "global_constant_note": "refused by design; runtime is per (branch, phase)",
        }
        return seal_field(body, "model_sha256")


# ── fitting ─────────────────────────────────────────────────────────────────────────────
def fit_runtime(observations: list[dict[str, Any]], *, config: Config | None = None) -> RuntimeModel:
    """Fit a per-(branch, phase) seconds-per-billion distribution from real observations.

    Each observation is a record with at least:
        branch, phase, stored_params_b (> 0), wall_seconds (> 0)
    Optional context carried through: rate, residency.

    Refuses to produce a single global constant. Reports, per segment: sample count and the
    p10/p50/p90 of observed seconds-per-billion, plus the min/max wall to expose spread.
    """
    cfg = config or default_config()
    if not isinstance(observations, list):
        raise EtaError("observations must be a list of records")

    buckets: dict[str, list[dict[str, Any]]] = {}
    for idx, obs in enumerate(observations):
        if not isinstance(obs, dict):
            raise EtaError(f"observation {idx} is not an object")
        branch = obs.get("branch")
        phase = obs.get("phase")
        if not isinstance(branch, str) or not branch:
            raise EtaError(f"observation {idx}: branch must be a non-empty string")
        if not isinstance(phase, str) or not phase:
            raise EtaError(f"observation {idx}: phase must be a non-empty string")
        params_b = _coerce_pos_float(obs.get("stored_params_b"), "stored_params_b", str(idx))
        wall = _coerce_pos_float(obs.get("wall_seconds"), "wall_seconds", str(idx))
        spb = wall / params_b
        buckets.setdefault(_segment_key(branch, phase), []).append({
            "branch": branch, "phase": phase, "spb": spb,
            "wall_seconds": wall, "stored_params_b": params_b,
            "rate": obs.get("rate"), "residency": obs.get("residency"),
        })

    segments: dict[str, dict[str, Any]] = {}
    for key, rows in buckets.items():
        spbs = sorted(r["spb"] for r in rows)
        walls = sorted(r["wall_seconds"] for r in rows)
        p50 = _percentile(spbs, cfg.p_mid)
        p10 = _percentile(spbs, cfg.p_low)
        p90 = _percentile(spbs, cfg.p_high)
        # spread ratio: p90/p10, a compact honesty signal on how noisy the segment is
        spread = (p90 / p10) if p10 > 0 else None
        segments[key] = {
            "branch": rows[0]["branch"],
            "phase": rows[0]["phase"],
            "count": len(rows),
            "spb_p10": round(p10, 6),
            "spb_p50": round(p50, 6),
            "spb_p90": round(p90, 6),
            "spb_spread_ratio": round(spread, 4) if spread is not None else None,
            "wall_min": round(walls[0], 3),
            "wall_max": round(walls[-1], 3),
            "rates_seen": sorted({r["rate"] for r in rows if r["rate"] is not None}),
            "residencies_seen": sorted({r["residency"] for r in rows if r["residency"] is not None}),
        }
    return RuntimeModel(segments, cfg, n_observations=len(observations))


# ── projection ──────────────────────────────────────────────────────────────────────────
def _program_is_giant(program: dict[str, Any], cfg: Config) -> bool:
    try:
        pb = float(program.get("params_b", 0.0))
    except (TypeError, ValueError):
        return True  # unparseable size -> treat as giant/unknown, fail closed
    return pb >= cfg.giant_params_b


def project_eta(remaining_programs: list[dict[str, Any]], model: RuntimeModel) -> dict[str, Any]:
    """Project total ETA over the remaining critical-path programs.

    `remaining_programs` are ordered records forming the critical path; each is:
        program_id, branch, phase, params_b, [rate], [residency], [wired]
    Programs are summed sequentially (the critical path is the whole ordered list of
    calibrated programs).

    Returns conservative(p90) / expected(p50) / optimistic(p10) totals over the calibrated
    prefix, a confidence/sensitivity range, the critical path, the empirical basis (which
    segments backed the estimate), the unknowns that block calibration, and an explicit
    per-program breakdown that marks any giant/unwired/uncalibrated program 'uncalibrated
    - no ETA'.
    """
    cfg = model.config
    if not isinstance(remaining_programs, list):
        raise EtaError("remaining_programs must be a list")

    total_p10 = total_p50 = total_p90 = 0.0
    critical_path: list[str] = []
    basis_segments: set[str] = set()
    unknowns: list[dict[str, Any]] = []
    breakdown: list[dict[str, Any]] = []
    any_uncalibrated = False

    for idx, program in enumerate(remaining_programs):
        if not isinstance(program, dict):
            raise EtaError(f"program {idx} is not an object")
        pid = program.get("program_id", f"program-{idx}")
        branch = program.get("branch")
        phase = program.get("phase")
        if not isinstance(branch, str) or not isinstance(phase, str):
            raise EtaError(f"program {pid}: branch and phase must be strings")
        params_b = program.get("params_b")
        rate = program.get("rate")
        residency = program.get("residency")
        wired = program.get("wired", True)
        seg_key = _segment_key(branch, phase)

        giant = _program_is_giant(program, cfg)
        pred = model.predict(branch, phase, rate, residency, params_b)

        # A giant or explicitly unwired program is NEVER given a calibrated ETA, even if a
        # lookalike segment is calibrated. This is the 120B guard.
        forced_no_eta_reason = None
        if giant:
            forced_no_eta_reason = (
                f"giant execution (params_b >= {cfg.giant_params_b}); unwired, no measured "
                f"runtime for {seg_key}")
        elif not wired:
            forced_no_eta_reason = f"program flagged unwired; no measured runtime for {seg_key}"

        calibrated = pred["calibrated"] and forced_no_eta_reason is None

        entry = {
            "program_id": pid,
            "segment": seg_key,
            "params_b": params_b,
            "rate": rate,
            "residency": residency,
        }
        if calibrated:
            entry["eta_status"] = "calibrated"
            entry["p10"] = pred["p10"]
            entry["p50"] = pred["p50"]
            entry["p90"] = pred["p90"]
            total_p10 += pred["p10"]
            total_p50 += pred["p50"]
            total_p90 += pred["p90"]
            critical_path.append(pid)
            basis_segments.add(seg_key)
        else:
            any_uncalibrated = True
            entry["eta_status"] = NO_ETA
            entry["p10"] = entry["p50"] = entry["p90"] = None
            reason = forced_no_eta_reason or pred.get(
                "reason", "segment not calibrated")
            entry["reason"] = reason
            unknowns.append({"program_id": pid, "segment": seg_key, "reason": reason})
        breakdown.append(entry)

    calibrated_count = len(critical_path)
    # Sensitivity range: how far conservative/optimistic straddle the expected total.
    if calibrated_count and total_p50 > 0:
        sensitivity_ratio = round(total_p90 / total_p10, 4) if total_p10 > 0 else None
    else:
        sensitivity_ratio = None

    # Confidence is a coverage statement, not a fabricated probability: the fraction of the
    # requested critical path that the evidence could actually calibrate.
    requested = len(remaining_programs)
    coverage = round(calibrated_count / requested, 4) if requested else 0.0

    body = {
        "schema": ETA_PROJECTION_SCHEMA,
        "generated_at": now_iso(),
        "totals_seconds": {
            "optimistic_p10": round(total_p10, 3) if calibrated_count else None,
            "expected_p50": round(total_p50, 3) if calibrated_count else None,
            "conservative_p90": round(total_p90, 3) if calibrated_count else None,
        },
        "confidence": {
            "calibrated_programs": calibrated_count,
            "requested_programs": requested,
            "coverage": coverage,
            "sensitivity_ratio_p90_over_p10": sensitivity_ratio,
            "complete": (not any_uncalibrated) and requested > 0,
        },
        "critical_path": critical_path,
        "empirical_basis": {
            "calibrated_segments": sorted(basis_segments),
            "min_samples_calibrated": cfg.min_samples_calibrated,
            "segment_sample_counts": {
                k: model.segments[k]["count"]
                for k in sorted(basis_segments) if k in model.segments
            },
        },
        "unknowns": unknowns,
        "programs": breakdown,
    }
    return seal_field(body, "projection_sha256")


def successor_eta_root() -> Path:
    """Successor-owned ETA artifact namespace. Never the campaign namespace."""
    return repo_root() / "reports" / "condense" / "event_horizon_successor" / "eta"


def write_projection(projection: dict[str, Any], *, root: Path | None = None,
                     name: str = "eta_projection.json") -> Path:
    base = root or successor_eta_root()
    if "doctor_v5_ultra" in str(base):
        raise EtaError("refusing to write into the campaign namespace")
    return atomic_write_json(base / name, projection)


# ── selftest ────────────────────────────────────────────────────────────────────────────
def selftest() -> dict[str, Any]:
    import tempfile

    cfg = default_config()

    # Two branches x two phases of synthetic observations. rwkv-state runs slower per
    # billion than qwen-dense; diagnose is much cheaper than materialize. Each calibrated
    # segment gets >= min_samples observations.
    observations: list[dict[str, Any]] = []
    # qwen-dense / diagnose: spb ~ 40 s/B
    for wall, pb in [(80.0, 2.0), (84.0, 2.0), (120.0, 3.0), (160.0, 4.0)]:
        observations.append({"branch": "qwen-dense", "phase": "diagnose",
                             "stored_params_b": pb, "wall_seconds": wall,
                             "rate": 0.5, "residency": "resident"})
    # qwen-dense / materialize: spb ~ 300 s/B (much heavier)
    for wall, pb in [(600.0, 2.0), (630.0, 2.0), (900.0, 3.0), (1200.0, 4.0)]:
        observations.append({"branch": "qwen-dense", "phase": "materialize",
                             "stored_params_b": pb, "wall_seconds": wall,
                             "rate": 0.5, "residency": "resident"})
    # rwkv-state / diagnose: spb ~ 70 s/B (slower branch)
    for wall, pb in [(140.0, 2.0), (150.0, 2.0), (210.0, 3.0), (300.0, 4.0)]:
        observations.append({"branch": "rwkv-state", "phase": "diagnose",
                             "stored_params_b": pb, "wall_seconds": wall,
                             "rate": 0.4, "residency": "resident"})
    # rwkv-state / materialize: spb ~ 520 s/B
    for wall, pb in [(1040.0, 2.0), (1080.0, 2.0), (1560.0, 3.0), (2080.0, 4.0)]:
        observations.append({"branch": "rwkv-state", "phase": "materialize",
                             "stored_params_b": pb, "wall_seconds": wall,
                             "rate": 0.4, "residency": "resident"})
    model = fit_runtime(observations, config=cfg)

    checks: dict[str, Any] = {}

    # 1. Per-segment medians differ between branches (no single global constant).
    qd_diag = model.segments[_segment_key("qwen-dense", "diagnose")]["spb_p50"]
    rw_diag = model.segments[_segment_key("rwkv-state", "diagnose")]["spb_p50"]
    qd_mat = model.segments[_segment_key("qwen-dense", "materialize")]["spb_p50"]
    checks["branch_medians_differ"] = (qd_diag != rw_diag)
    checks["phase_medians_differ"] = (qd_diag != qd_mat)
    if not (checks["branch_medians_differ"] and checks["phase_medians_differ"]):
        raise EtaError("segments failed to differentiate branch/phase runtime")

    # 2. A global constant is refused.
    global_refused = False
    try:
        model.global_seconds_per_billion()
    except EtaError:
        global_refused = True
    checks["global_constant_refused"] = global_refused
    if not global_refused:
        raise EtaError("global seconds-per-billion was not refused")
    # and the serialized model exposes no global constant
    md = model.as_dict()
    checks["model_has_no_global_constant"] = (md.get("global_constant") is None)
    checks["model_sealed"] = sealed(md, "model_sha256")
    if not checks["model_sealed"]:
        raise EtaError("model artifact not self-sealed")

    # 3. Predict on a calibrated segment -> p10 <= p50 <= p90 band.
    pred = model.predict("qwen-dense", "diagnose", rate=0.5, residency="resident", params_b=3.0)
    checks["calibrated_predict"] = pred["calibrated"]
    band_ok = (pred["p10"] is not None and pred["p10"] <= pred["p50"] <= pred["p90"])
    checks["band_monotone"] = band_ok
    if not (pred["calibrated"] and band_ok):
        raise EtaError(f"calibrated predict band wrong: {pred}")

    # 4. Predict on an unseen segment -> uncalibrated, no fabricated band.
    unseen = model.predict("mamba", "materialize", rate=None, residency=None, params_b=7.0)
    checks["unseen_uncalibrated"] = (not unseen["calibrated"] and unseen["p50"] is None)
    if not checks["unseen_uncalibrated"]:
        raise EtaError("unseen segment was not marked uncalibrated")

    # 5. project_eta over a critical path with one calibrated small program and one
    #    120B/giant program -> three totals present, giant marked no-ETA.
    programs = [
        {"program_id": "prog-7b-diagnose", "branch": "qwen-dense", "phase": "diagnose",
         "params_b": 7.0, "rate": 0.5, "residency": "resident"},
        {"program_id": "prog-14b-materialize", "branch": "rwkv-state", "phase": "materialize",
         "params_b": 14.0, "rate": 0.4, "residency": "resident"},
        {"program_id": "prog-120b-materialize", "branch": "qwen-dense", "phase": "materialize",
         "params_b": 120.0, "rate": 0.1, "residency": "spilled"},
    ]
    proj = project_eta(programs, model)
    totals = proj["totals_seconds"]
    checks["three_totals_present"] = all(
        totals[k] is not None for k in ("optimistic_p10", "expected_p50", "conservative_p90"))
    checks["totals_ordered"] = (
        totals["optimistic_p10"] <= totals["expected_p50"] <= totals["conservative_p90"])
    giant_entry = next(p for p in proj["programs"] if p["program_id"] == "prog-120b-materialize")
    checks["giant_marked_no_eta"] = (giant_entry["eta_status"] == NO_ETA
                                     and giant_entry["p50"] is None)
    checks["giant_in_unknowns"] = any(
        u["program_id"] == "prog-120b-materialize" for u in proj["unknowns"])
    checks["projection_incomplete"] = (proj["confidence"]["complete"] is False)
    checks["projection_sealed"] = sealed(proj, "projection_sha256")
    if not (checks["three_totals_present"] and checks["totals_ordered"]
            and checks["giant_marked_no_eta"] and checks["giant_in_unknowns"]
            and checks["projection_sealed"]):
        raise EtaError(f"project_eta guarantees violated: {checks}")

    # 6. Critical path contains only the two calibrated programs, in order.
    checks["critical_path"] = proj["critical_path"]
    if proj["critical_path"] != ["prog-7b-diagnose", "prog-14b-materialize"]:
        raise EtaError(f"critical path wrong: {proj['critical_path']}")

    # 7. Write + read back the projection under the successor namespace only.
    with tempfile.TemporaryDirectory() as d:
        out = write_projection(proj, root=Path(d) / "eta")
        back = read_json_safe(out)
        checks["roundtrip_sealed"] = sealed(back, "projection_sha256")
        if not checks["roundtrip_sealed"]:
            raise EtaError("written projection lost its seal")
        # refuse campaign namespace
        campaign_refused = False
        try:
            write_projection(proj, root=Path(d) / "reports" / "condense" / "doctor_v5_ultra")
        except EtaError:
            campaign_refused = True
        checks["campaign_namespace_refused"] = campaign_refused
        if not campaign_refused:
            raise EtaError("campaign namespace write was not refused")

    return {
        "ok": True,
        "module": "succ_eta",
        "n_observations": len(observations),
        "calibrated_segments": model.calibrated_segments(),
        "uncalibrated_segments": model.uncalibrated_segments(),
        "eta_totals_seconds": totals,
        "checks": checks,
    }


if __name__ == "__main__":
    print(json.dumps(selftest(), indent=2, sort_keys=True))
