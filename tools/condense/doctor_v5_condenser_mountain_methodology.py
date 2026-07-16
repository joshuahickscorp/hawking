#!/usr/bin/env python3.12
"""Empirical, fail-closed condenser methodology for large Doctor tiers.

This tool observes completed Doctor receipts and emits an *unbound* methodology
snapshot.  It never imports the live queue, launches work, reads model payloads,
changes runtime specifications, or grants execution authority.  Its purpose is
to keep the research method responsive to physical evidence while preserving
the active campaign as an immutable experiment.

The central distinction is between:

* a ladder, where a dense resident model can reasonably inherit a calibrated
  execution shape; and
* a mountain, where parameter-pass mass, streaming, source encoding, or
  architecture changes require an architecture-specific qualification funnel.

All quality observations remain provisional until the signed physical release
controller accepts them.  Transfer projections are priors, never claims.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import statistics
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCHEMA = "hawking.doctor_v5_condenser_mountain_methodology.v1"
VERSION = "2026-07-16.2"
DEFAULT_PLAN = ROOT / "reports/condense/doctor_v5_ultra/campaign_plan.json"
DEFAULT_CAMPAIGN = ROOT / "reports/condense/doctor_v5_ultra/campaign.json"
DEFAULT_RESULTS = ROOT / "reports/condense/doctor_v5_ultra/results"
DEFAULT_OUTPUT = (
    ROOT / "reports/condense/doctor_v5_unbound/condenser_methodology/"
    "empirical_mountain_snapshot.json"
)
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_LOG_BYTES = 4 * 1024 * 1024
MOUNTAIN_PARAMETER_PASS_SHARE = 0.10
GOOD_PPL_DELTA_MAX = 0.08
GOOD_CAPABILITY_DELTA_MIN = -0.05
RATE_ORDER = ("4", "3", "2", "1", "0.8", "0.55", "0.5", "0.33", "0.25", "0.1")
BRANCH_ORDER = (
    "codec_control", "doctor_static", "doctor_conditional", "doctor_full",
)


class MethodologyError(RuntimeError):
    """The empirical snapshot is inconsistent or crosses its claim boundary."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != field}


def _read_bytes(path: Path, maximum: int) -> bytes:
    path = Path(path).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise MethodologyError(f"input is not a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size > maximum:
        raise MethodologyError(f"input exceeds read ceiling: {path}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise MethodologyError(f"short read: {path}")
    return raw


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_bytes(path, MAX_JSON_BYTES)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MethodologyError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MethodologyError(f"JSON root is not an object: {path}")
    return value, raw


def _artifact(path: Path, raw: bytes | None = None) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    raw = raw if raw is not None else _read_bytes(path, MAX_JSON_BYTES)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _median(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(finite) if finite else None


def _timestamp(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise MethodologyError("campaign timestamp lacks timezone")
    return parsed


def _phase_logs(cell_root: Path) -> list[tuple[str, float, str]]:
    """Return phase, seconds, file hash for bounded child logs."""
    observations: list[tuple[str, float, str]] = []
    for relative in (Path("strand_ladder/logs"), Path("qwen_treatment/logs")):
        log_root = cell_root / relative
        if not log_root.is_dir():
            continue
        for path in sorted(log_root.glob("*.log")):
            raw = _read_bytes(path, MAX_LOG_BYTES)
            started: dt.datetime | None = None
            for line in raw.decode("utf-8", errors="replace").splitlines()[:3]:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get("event") == "child_start" \
                        and isinstance(event.get("at"), str):
                    started = _timestamp(event["at"])
                    break
            if started is None:
                continue
            ended = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
            seconds = (ended - started).total_seconds()
            if seconds < 0 or not math.isfinite(seconds):
                continue
            phase = path.name.split("-", 1)[0].replace(".log", "")
            observations.append((phase, seconds, hashlib.sha256(raw).hexdigest()))
    return observations


def _result_observation(cell: dict[str, Any], results_root: Path) -> dict[str, Any] | None:
    path = results_root / cell["cell_id"] / "result.json"
    if not path.is_file():
        return None
    result, raw = _read_json(path)
    if result.get("status") != "complete":
        return None
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    quality = metrics.get("quality_observation")
    if not isinstance(quality, dict):
        outcome = metrics.get("treatment_outcome")
        quality = outcome.get("quality_observation") if isinstance(outcome, dict) else {}
    quality = quality if isinstance(quality, dict) else {}
    ppl = quality.get("ppl") if isinstance(quality.get("ppl"), dict) else {}
    capability = (
        quality.get("capability") if isinstance(quality.get("capability"), dict) else {}
    )
    physical = (
        metrics.get("physical_accounting")
        if isinstance(metrics.get("physical_accounting"), dict) else {}
    )
    phases = _phase_logs(path.parent)
    phase_root = _hash_value([
        {"phase": phase, "seconds": seconds, "file_sha256": digest}
        for phase, seconds, digest in phases
    ])
    started = cell.get("started_at")
    completed = cell.get("completed_at")
    wall = None
    if isinstance(started, str) and isinstance(completed, str):
        wall = (_timestamp(completed) - _timestamp(started)).total_seconds()
    child_span = sum(seconds for _phase, seconds, _digest in phases)
    recovery_window_warning = bool(
        wall and cell.get("attempts", 0) > 1 and child_span > wall * 1.25
    )
    return {
        "cell_id": cell["cell_id"],
        "model_label": cell["model_label"],
        "model_family": cell["model_family"],
        "rate_id": str(cell["rate_id"]),
        "branch": cell["branch"],
        "exact_stored_parameter_count": cell["exact_stored_parameter_count"],
        "attempts": cell.get("attempts", 0),
        "receipt_window_wall_seconds": wall,
        "child_phase_seconds": child_span,
        "recovery_window_may_understate_physical_work": recovery_window_warning,
        "actual_model_payload_bpw": physical.get("all_in_model_payload_bpw"),
        "target_bpw": physical.get("target_physical_bpw"),
        "ppl_relative_delta": ppl.get("relative_delta"),
        "capability_absolute_delta": capability.get("absolute_delta"),
        "quality_status": quality.get("status"),
        "quality_claims_permitted": result.get("quality_claims_permitted") is True,
        "phase_count": len(phases),
        "phase_observation_root_sha256": phase_root,
        "result": _artifact(path, raw),
    }


def _tier_progress(cells: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    total_passes = sum(int(cell["exact_stored_parameter_count"]) for cell in cells)
    completed = [cell for cell in cells if cell.get("status") == "complete"]
    completed_passes = sum(int(cell["exact_stored_parameter_count"]) for cell in completed)
    labels = sorted(
        {str(cell["model_label"]) for cell in cells},
        key=lambda label: float(label.rstrip("BT")),
    )
    tiers: list[dict[str, Any]] = []
    for label in labels:
        rows = [cell for cell in cells if cell["model_label"] == label]
        passes = sum(int(cell["exact_stored_parameter_count"]) for cell in rows)
        done = [cell for cell in rows if cell.get("status") == "complete"]
        representative = rows[0]
        admission = representative.get("admission", {})
        triggers: list[str] = []
        share = passes / total_passes
        if share >= MOUNTAIN_PARAMETER_PASS_SHARE:
            triggers.append("parameter_pass_share")
        if admission.get("streaming_required") is True:
            triggers.append("streaming_required")
        if representative.get("model_family") != "qwen2.5-dense":
            triggers.append("architecture_discontinuity")
        manifest = representative.get("parameter_manifest", {})
        if manifest.get("active_moe_parameter_count") is not None:
            triggers.append("stored_active_parameter_discontinuity")
        tiers.append({
            "model_label": label,
            "model_family": representative["model_family"],
            "cell_count": len(rows),
            "complete_cells": len(done),
            "matrix_parameter_passes": passes,
            "parameter_pass_share": share,
            "classification": "mountain" if triggers else "ladder",
            "mountain_triggers": triggers,
        })
    return tiers, {
        "cell_count": len(cells),
        "complete_cells": len(completed),
        "cell_completion_fraction": len(completed) / len(cells),
        "matrix_parameter_passes": total_passes,
        "completed_parameter_passes": completed_passes,
        "parameter_pass_completion_fraction": completed_passes / total_passes,
    }


def _timing(observations: list[dict[str, Any]]) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    for rate in RATE_ORDER:
        for branch in BRANCH_ORDER:
            rows = [row for row in observations
                    if row["rate_id"] == rate and row["branch"] == branch]
            spb = [
                row["receipt_window_wall_seconds"]
                / (row["exact_stored_parameter_count"] / 1_000_000_000)
                for row in rows if row["receipt_window_wall_seconds"] is not None
            ]
            if rows:
                groups.append({
                    "rate_id": rate, "branch": branch, "samples": len(rows),
                    "median_receipt_seconds_per_billion": _median(spb),
                    "recovery_contaminated_samples": sum(
                        row["recovery_window_may_understate_physical_work"] for row in rows
                    ),
                })
    phase_values: dict[str, list[float]] = {}
    # The compact report intentionally carries only per-result phase totals.
    for row in observations:
        if row["child_phase_seconds"]:
            phase_values.setdefault(row["model_label"], []).append(
                row["child_phase_seconds"]
            )
    return {
        "rate_branch_receipt_profiles": groups,
        "model_child_phase_medians": [
            {"model_label": label, "samples": len(values),
             "median_child_phase_seconds": _median(values)}
            for label, values in sorted(
                phase_values.items(), key=lambda item: float(item[0].rstrip("BT"))
            )
        ],
        "recovery_warning_count": sum(
            row["recovery_window_may_understate_physical_work"]
            for row in observations
        ),
        "receipt_window_is_not_operation_timing": True,
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = ("actual_model_payload_bpw", "ppl_relative_delta",
            "capability_absolute_delta")
    if any(left.get(key) is None or right.get(key) is None for key in keys):
        return False
    no_worse = (
        left["actual_model_payload_bpw"] <= right["actual_model_payload_bpw"]
        and left["ppl_relative_delta"] <= right["ppl_relative_delta"]
        and left["capability_absolute_delta"] >= right["capability_absolute_delta"]
    )
    strict = (
        left["actual_model_payload_bpw"] < right["actual_model_payload_bpw"]
        or left["ppl_relative_delta"] < right["ppl_relative_delta"]
        or left["capability_absolute_delta"] > right["capability_absolute_delta"]
    )
    return no_worse and strict


def _rung_branch_efficiency(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Summarize empirical branch dominance only after an exact rung completes."""
    by_branch = {row["branch"]: row for row in rows}
    if set(by_branch) != set(BRANCH_ORDER) or len(rows) != len(BRANCH_ORDER):
        return None
    ordered = [by_branch[branch] for branch in BRANCH_ORDER]
    active = [
        row for row in ordered
        if not any(_dominates(other, row) for other in ordered if other is not row)
    ]
    dominated = [row for row in ordered if row not in active]
    receipt_values = [
        row.get("receipt_window_wall_seconds") for row in ordered
    ]
    complete_receipt_timing = all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value)) and float(value) >= 0
        for value in receipt_values
    )
    total_receipt_seconds = (
        sum(float(value) for value in receipt_values)
        if complete_receipt_timing else None
    )
    dominated_receipt_seconds = (
        sum(float(row["receipt_window_wall_seconds"]) for row in dominated)
        if complete_receipt_timing else None
    )
    best_physical = min(
        ordered,
        key=lambda row: (
            float(row["actual_model_payload_bpw"]),
            float(row["ppl_relative_delta"]),
            -float(row["capability_absolute_delta"]),
            BRANCH_ORDER.index(row["branch"]),
        ),
    )
    physical_passes = [
        row for row in ordered
        if float(row["actual_model_payload_bpw"]) <= float(row["target_bpw"])
    ]
    quality_passes = [
        row for row in ordered
        if float(row["ppl_relative_delta"]) <= GOOD_PPL_DELTA_MAX
        and float(row["capability_absolute_delta"]) >= GOOD_CAPABILITY_DELTA_MIN
    ]
    promotable = [
        row for row in quality_passes
        if float(row["actual_model_payload_bpw"]) <= float(row["target_bpw"])
    ]
    failed_gates: list[str] = []
    if not physical_passes:
        failed_gates.append("physical_density")
    if not quality_passes:
        failed_gates.append("competitive_quality")
    return {
        "model_label": best_physical["model_label"],
        "rate_id": best_physical["rate_id"],
        "promotion_result": "good" if promotable else "bad",
        "evidence_value": "promotable" if promotable else "useful-negative",
        "failed_gates": failed_gates,
        "best_physical_branch": best_physical["branch"],
        "best_actual_model_payload_bpw": best_physical[
            "actual_model_payload_bpw"
        ],
        "pareto_active_branches": [row["branch"] for row in active],
        "dominated_branches": [row["branch"] for row in dominated],
        "frontier_repeat_defer_candidates": [row["branch"] for row in dominated],
        "observed_total_receipt_seconds": total_receipt_seconds,
        "observed_dominated_receipt_seconds": dominated_receipt_seconds,
        "observed_dominated_receipt_fraction": (
            dominated_receipt_seconds / total_receipt_seconds
            if dominated_receipt_seconds is not None and total_receipt_seconds else None
        ),
        "frontier_repeat_pruning_requires_same_model_rate_canary": True,
        "exhaustive_track_retains_all_branches": True,
    }


def _quality(observations: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in observations
              if row.get("actual_model_payload_bpw") is not None
              and row.get("ppl_relative_delta") is not None
              and row.get("capability_absolute_delta") is not None]
    pareto: list[dict[str, Any]] = []
    for label in sorted({row["model_label"] for row in usable},
                        key=lambda value: float(value.rstrip("BT"))):
        rows = [row for row in usable if row["model_label"] == label]
        frontier = sorted(
            [row["cell_id"] for row in rows
             if not any(_dominates(other, row) for other in rows if other is not row)]
        )
        pareto.append({"model_label": label, "observations": len(rows),
                       "pareto_cell_ids": frontier})
    priors: list[dict[str, Any]] = []
    for rate in RATE_ORDER:
        for branch in BRANCH_ORDER:
            rows = [row for row in usable
                    if row["rate_id"] == rate and row["branch"] == branch]
            if rows:
                priors.append({
                    "rate_id": rate, "branch": branch, "samples": len(rows),
                    "median_actual_bpw": _median(
                        row["actual_model_payload_bpw"] for row in rows
                    ),
                    "median_ppl_relative_delta": _median(
                        row["ppl_relative_delta"] for row in rows
                    ),
                    "median_capability_absolute_delta": _median(
                        row["capability_absolute_delta"] for row in rows
                    ),
                    "transfer_status": "dense-qwen-prior-not-gptoss-authority",
                })
    rung_efficiency: list[dict[str, Any]] = []
    model_rate_pairs = sorted(
        {(row["model_label"], row["rate_id"]) for row in usable},
        key=lambda item: (
            float(item[0].rstrip("BT")),
            RATE_ORDER.index(item[1]) if item[1] in RATE_ORDER else len(RATE_ORDER),
        ),
    )
    for label, rate in model_rate_pairs:
        summary = _rung_branch_efficiency([
            row for row in usable
            if row["model_label"] == label and row["rate_id"] == rate
        ])
        if summary is not None:
            rung_efficiency.append(summary)
    compact = [{key: row[key] for key in (
        "cell_id", "model_label", "rate_id", "branch", "target_bpw",
        "actual_model_payload_bpw", "ppl_relative_delta",
        "capability_absolute_delta", "quality_status",
        "quality_claims_permitted", "result",
    )} for row in observations]
    return {
        "observation_count": len(observations),
        "fully_comparable_observation_count": len(usable),
        "observations": compact,
        "per_model_pareto_frontiers": pareto,
        "per_model_rate_branch_efficiency": rung_efficiency,
        "branch_efficiency_semantics": (
            "dominated receipt fractions are summed completed receipt windows, "
            "not projected host wall-time savings; use only to order or defer "
            "frontier repeats after a same-model/rate canary"
        ),
        "rate_branch_transfer_priors": priors,
        "promotion_objectives": [
            "minimize_actual_physical_bpw",
            "minimize_quality_loss_vs_exact_source_baseline",
            "maximize_capability_retention",
        ],
        "nominal_target_bpw_is_not_a_promotion_metric": True,
        "methodology_quality_claims_permitted": False,
        "all_current_quality_is_provisional": all(
            row["quality_claims_permitted"] is not True for row in observations
        ),
        "negative_evidence_is_retained": True,
        "exhaustive_track_retains_empirically_dominated_branches": True,
    }


def _gptoss(cells: list[dict[str, Any]], timing: dict[str, Any],
            quality: dict[str, Any]) -> dict[str, Any]:
    rows = [cell for cell in cells if cell["model_label"] == "120B"]
    if len(rows) != 40:
        raise MethodologyError("exact GPT-OSS 120B 10x4 matrix is absent")
    representative = rows[0]
    manifest = representative["parameter_manifest"]
    stored = int(representative["exact_stored_parameter_count"])
    active = int(manifest["active_moe_parameter_count"])
    source_bytes = int(manifest["source_weight_bytes"])
    profiles = {
        (row["rate_id"], row["branch"]): row
        for row in timing["rate_branch_receipt_profiles"]
    }
    by_rate: list[dict[str, Any]] = []
    total_seconds = 0.0
    all_profiles = True
    priors = {
        (row["rate_id"], row["branch"]): row
        for row in quality["rate_branch_transfer_priors"]
    }
    for rate in RATE_ORDER:
        seconds = 0.0
        candidates: list[dict[str, Any]] = []
        for branch in BRANCH_ORDER:
            profile = profiles.get((rate, branch))
            if not profile or profile["median_receipt_seconds_per_billion"] is None:
                all_profiles = False
            else:
                seconds += profile["median_receipt_seconds_per_billion"] * stored / 1e9
            prior = priors.get((rate, branch))
            if prior:
                projected_bpw = prior["median_actual_bpw"]
                projected_bytes = math.ceil(stored * projected_bpw / 8)
                candidates.append({
                    "branch": branch,
                    "dense_prior_actual_bpw": projected_bpw,
                    "projected_candidate_bytes": projected_bytes,
                    "projected_bytes_delta_vs_mxfp4_source": projected_bytes - source_bytes,
                    "quality_transfer_permitted": False,
                })
        total_seconds += seconds
        by_rate.append({
            "rate_id": rate,
            "naive_serial_seconds_from_receipt_medians": seconds if seconds else None,
            "naive_serial_days_from_receipt_medians": seconds / 86_400 if seconds else None,
            "dense_qwen_transfer_priors": candidates,
        })
    return {
        "classification": "mountain",
        "stored_parameters": stored,
        "active_moe_parameters": active,
        "active_to_stored_ratio": active / stored,
        "mxfp4_source_weight_bytes": source_bytes,
        "mxfp4_source_physical_bpw": source_bytes * 8 / stored,
        "source_units": 615,
        "isolated_jobs": 24_600,
        "matrix_cells": 40,
        "matrix_parameter_passes": stored * 40,
        "receipt_median_projection_complete": all_profiles,
        "naive_serial_seconds_from_receipt_medians": (
            total_seconds if all_profiles else None
        ),
        "naive_serial_days_from_receipt_medians": (
            total_seconds / 86_400 if all_profiles else None
        ),
        "by_rate": by_rate,
        "projection_limits": [
            "receipt windows are not operation-only physical timings",
            "recovery can reset a cell receipt window",
            "dense Qwen timing and quality do not authorize GPT-OSS transfer",
            "shared traversal and preprocessing reuse are not credited before physical receipts",
        ],
    }


def _methodology() -> dict[str, Any]:
    return {
        "ideology": {
            "primary_objective": (
                "beat the current physical-bpw quality frontier without weakening exactness"
            ),
            "secondary_objective": "minimize time-to-frontier and total mountain wall time",
            "forbidden_shortcuts": [
                "counting nominal target bpw instead of physical artifact bpw",
                "promoting quality estimates or small-model transfer priors",
                "skipping exact source-baseline comparisons",
                "sharing mutable evidence or output artifacts between cells",
                "trading parity, reproducibility, or rollback for speed",
            ],
        },
        "mountain_funnel": [
            {
                "stage": 1, "name": "authority-and-source-baseline",
                "requirements": [
                    "exact architecture, tokenizer, source, parameter, and MXFP4 authority",
                    "measure source physical bpw and quality as the transcode baseline",
                    "separate stored-weight transform cost from active-MoE evaluation cost",
                ],
            },
            {
                "stage": 2, "name": "representative-unit-physical-canaries",
                "requirements": [
                    "cover shared tensors, routers, hot and cold experts, large and ragged tails",
                    "calibrate 8, 12, 16, and 20 threads per rate and phase",
                    "measure RAM, swap, thermals, bytes, parity, and operation-only time",
                ],
            },
            {
                "stage": 3, "name": "single-traversal-multi-rate-fanout",
                "requirements": [
                    "read and decode every immutable source unit once per qualified generation",
                    "reuse only immutable branch-independent preprocessing",
                    "keep all rate and branch artifacts, receipts, and failures isolated",
                ],
            },
            {
                "stage": 4, "name": "expert-aware-rate-distortion-search",
                "requirements": [
                    "protect routers, embeddings, shared blocks, and frequently routed experts",
                    "allocate lower bpw to insensitive and rarely routed expert blocks",
                    "optimize a global actual-bpw budget rather than a uniform nominal rate",
                    "adapt residual, outlier, codebook, and side-information budgets per unit",
                ],
            },
            {
                "stage": 5, "name": "progressive-quality-funnel",
                "requirements": [
                    "run exact smoke, routed-expert coverage, medium, then full quality suites",
                    "cache one hash-bound source baseline and never cache candidate conclusions",
                    "promote only statistically stable actual-bpw Pareto improvements",
                    "retain dominated, failed, and negative evidence instead of hiding it",
                ],
            },
            {
                "stage": 6, "name": "direct-quantized-moe-qualification",
                "requirements": [
                    "evaluate the quantized MoE representation without whole-model dense reconstruction",
                    "prove tokenizer, routing, expert coverage, greedy parity, and rollback",
                    "require sealed physical quality and performance receipts before promotion",
                ],
            },
        ],
        "aggressive_research_axes": [
            "mixed-bpw allocation across tensor roles and experts",
            "route-frequency and sensitivity-weighted expert budgets",
            "per-unit residual and sparse-outlier channel selection",
            "shared RHT/statistics with isolated deterministic encodes",
            "progressive precision descent from the source frontier",
            "quality-per-byte and quality-per-joule Pareto ranking",
            "sequential tests that stop dominated candidates early without erasing evidence",
            "native quantized evaluation to avoid reconstruction I/O",
            "ordered read/preprocess/encode/write/attest overlap",
            "Metal preprocessing only after exact CPU parity and physical attribution",
        ],
        "two_track_completion": {
            "frontier_track": (
                "produce the best physically qualified low-bpw artifact as early as evidence allows"
            ),
            "exhaustive_track": (
                "retain the exact 10x4 research matrix and complete it without blocking frontier learning"
            ),
            "automatic_default_promotion_permitted": False,
        },
    }


def build_report(plan_path: Path = DEFAULT_PLAN,
                 campaign_path: Path = DEFAULT_CAMPAIGN,
                 results_root: Path = DEFAULT_RESULTS,
                 *, created_at: str | None = None) -> dict[str, Any]:
    plan, plan_raw = _read_json(plan_path)
    campaign, campaign_raw = _read_json(campaign_path)
    if plan.get("schema") != "hawking.doctor_v5_ultra_campaign_plan.v1":
        raise MethodologyError("campaign plan schema differs")
    if campaign.get("schema") != "hawking.doctor_v5_ultra_campaign.v1":
        raise MethodologyError("campaign observation schema differs")
    if plan.get("plan_sha256") != campaign.get("plan_sha256"):
        raise MethodologyError("plan and campaign observation differ")
    cells = campaign.get("cells")
    if not isinstance(cells, list) or len(cells) != 320:
        raise MethodologyError("campaign observation does not contain the exact 320 cells")
    tiers, progress = _tier_progress(cells)
    observations = [
        row for row in (
            _result_observation(cell, Path(results_root))
            for cell in cells if cell.get("status") == "complete"
        ) if row is not None
    ]
    result_set_root = _hash_value([
        {"cell_id": row["cell_id"],
         "result_sha256": row["result"]["sha256"],
         "phase_observation_root_sha256": row["phase_observation_root_sha256"]}
        for row in sorted(observations, key=lambda item: item["cell_id"])
    ])
    timing = _timing(observations)
    quality = _quality(observations)
    report = {
        "schema": SCHEMA,
        "version": VERSION,
        "created_at": created_at or _now(),
        "status": "sealed-unbound-empirical-methodology-not-executable",
        "claim_boundary": {
            "execution_permitted": False,
            "live_queue_mutation_permitted": False,
            "runtime_spec_mutation_permitted": False,
            "adapter_registry_mutation_permitted": False,
            "completed_evidence_mutation_permitted": False,
            "runtime_default_promotion_permitted": False,
            "quality_claims_permitted": False,
            "source_deletion_permitted": False,
        },
        "input_snapshot": {
            "campaign_plan": _artifact(plan_path, plan_raw),
            "mutable_campaign_observation": _artifact(campaign_path, campaign_raw),
            "mutable_campaign_observation_is_authority": False,
            "results_root": str(Path(results_root).resolve()),
            "completed_result_count": len(observations),
            "completed_result_set_root_sha256": result_set_root,
            "tool": _artifact(Path(__file__)),
        },
        "progress": {**progress, "tiers": tiers},
        "empirical_timing": timing,
        "empirical_quality": quality,
        "gptoss_120b": _gptoss(cells, timing, quality),
        "condenser_methodology": _methodology(),
        "live_campaign_action": "observe-only-no-mutation",
    }
    report["report_sha256"] = _hash_value(report)
    return report


TOP_LEVEL_KEYS = {
    "schema", "version", "created_at", "status", "claim_boundary",
    "input_snapshot", "progress", "empirical_timing", "empirical_quality",
    "gptoss_120b", "condenser_methodology", "live_campaign_action",
    "report_sha256",
}


def validate_report(report: Any) -> list[str]:
    if not isinstance(report, dict):
        return ["methodology report must be an object"]
    errors: list[str] = []
    if set(report) != TOP_LEVEL_KEYS:
        errors.append("methodology report top-level shape differs")
    if report.get("schema") != SCHEMA or report.get("version") != VERSION:
        errors.append("methodology report schema/version differs")
    if report.get("status") != "sealed-unbound-empirical-methodology-not-executable":
        errors.append("methodology report status differs")
    try:
        if report.get("report_sha256") != _hash_value(
                _without(report, "report_sha256")):
            errors.append("methodology report hash differs")
    except (TypeError, ValueError):
        errors.append("methodology report cannot be canonically hashed")
    boundary = report.get("claim_boundary")
    if not isinstance(boundary, dict) or not boundary \
            or any(value is not False for value in boundary.values()):
        errors.append("methodology claim boundary is not fail-closed")
    progress = report.get("progress") if isinstance(report.get("progress"), dict) else {}
    tiers = progress.get("tiers") if isinstance(progress.get("tiers"), list) else []
    if sum(row.get("cell_count", 0) for row in tiers if isinstance(row, dict)) != 320:
        errors.append("tier coverage does not close to 320 cells")
    gptoss = report.get("gptoss_120b") if isinstance(report.get("gptoss_120b"), dict) else {}
    if gptoss.get("classification") != "mountain" \
            or gptoss.get("matrix_cells") != 40 \
            or gptoss.get("source_units") != 615 \
            or gptoss.get("isolated_jobs") != 24_600:
        errors.append("GPT-OSS mountain identity differs")
    try:
        expected_bpw = (
            gptoss["mxfp4_source_weight_bytes"] * 8 / gptoss["stored_parameters"]
        )
        if not math.isclose(gptoss["mxfp4_source_physical_bpw"], expected_bpw,
                            rel_tol=0.0, abs_tol=1e-12):
            errors.append("GPT-OSS source physical bpw differs")
    except (KeyError, TypeError, ZeroDivisionError):
        errors.append("GPT-OSS source accounting is invalid")
    quality = (
        report.get("empirical_quality")
        if isinstance(report.get("empirical_quality"), dict) else {}
    )
    if quality.get("nominal_target_bpw_is_not_a_promotion_metric") is not True \
            or quality.get("methodology_quality_claims_permitted") is not False:
        errors.append("empirical quality claim boundary differs")
    method = (
        report.get("condenser_methodology")
        if isinstance(report.get("condenser_methodology"), dict) else {}
    )
    funnel = method.get("mountain_funnel") if isinstance(method.get("mountain_funnel"), list) else []
    if [row.get("stage") for row in funnel if isinstance(row, dict)] != list(range(1, 7)):
        errors.append("mountain funnel stages differ")
    if report.get("live_campaign_action") != "observe-only-no-mutation":
        errors.append("live campaign action differs")
    return errors


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    progress = report["progress"]
    gptoss = report["gptoss_120b"]
    return {
        "status": report["status"],
        "report_sha256": report["report_sha256"],
        "complete_cells": progress["complete_cells"],
        "cell_completion_fraction": progress["cell_completion_fraction"],
        "parameter_pass_completion_fraction": progress[
            "parameter_pass_completion_fraction"
        ],
        "mountain_tiers": [
            row["model_label"] for row in progress["tiers"]
            if row["classification"] == "mountain"
        ],
        "gptoss_source_physical_bpw": gptoss["mxfp4_source_physical_bpw"],
        "gptoss_naive_serial_days": gptoss[
            "naive_serial_days_from_receipt_medians"
        ],
        "live_campaign_action": report["live_campaign_action"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "build", "verify"))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "verify":
            path = args.report or args.output
            report, _raw = _read_json(path)
            errors = validate_report(report)
            print(json.dumps({"ok": not errors, "errors": errors,
                              "summary": _summary(report) if not errors else None},
                             indent=2, sort_keys=True))
            return 0 if not errors else 2
        report = build_report(args.plan, args.campaign, args.results)
        errors = validate_report(report)
        if errors:
            raise MethodologyError("built report is invalid: " + "; ".join(errors))
        if args.command == "build":
            _atomic_json(args.output, report)
        print(json.dumps(_summary(report), indent=2, sort_keys=True))
        return 0
    except (MethodologyError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
