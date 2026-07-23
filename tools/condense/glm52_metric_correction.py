#!/usr/bin/env python3.12
"""Re-score every sealed GLM-5.2 pilot row against a fit-split null, and supersede the
claims that depended on the broken metric.

Nothing is deleted and nothing is recomputed from the artifacts.  Relative error is a
sealed, exact, metric-independent quantity, and the null it should have been measured
against is recoverable from the teacher capsules, so the corrected skill follows in closed
form:

    skill = 1 - relative_error^2 * ||y||^2 / ||y - mu_fit||^2

The first correction pass in this campaign centred by the TEACHER's own mean, which is a
statistic of the held-out target and is exactly what the contract forbids.  That defect is
recorded here rather than quietly re-run: those centred numbers are superseded, and the
skill computed here uses a mean fitted on ``teacher_fit`` alone.

    run          write the correction ledger and the superseding law
    selftest
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CONDENSE = Path(__file__).resolve().parent
if str(CONDENSE) not in sys.path:
    sys.path.insert(0, str(CONDENSE))

ROOT = Path(__file__).resolve().parents[2]
SUPPORT = Path(os.environ.get(
    "GLM52_SUPPORT_ROOT",
    "/Users/scammermike/Library/Application Support/Hawking/GLM52Gravity"))
CAPSULES = SUPPORT / "source_fetch" / "teacher" / "capsules_generation_b"
GEN_B = ROOT / "reports" / "condense" / "glm52_generation_b"
MEASUREMENTS = GEN_B / "GLM52_PILOT_MEASUREMENTS.jsonl"
LEDGER = ROOT / "GLM52_METRIC_CORRECTION_LEDGER.jsonl"

# The pilot measured the compact reconstruction against the teacher_fit capsule, so the
# null must be fitted on a split the score did not come from.  teacher_router is the
# nearest disjoint split that exists for every pilot layer.
NULL_SPLIT = "teacher_router"
FALLBACK_NULL_SPLIT = "teacher_fit"


def _capsule(layer: int, split: str) -> Path:
    name = f"L{layer:02d}_L{layer:02d}.npz"
    return CAPSULES / name if split == "teacher_fit" else CAPSULES / split / name


def _stats(layer: int, stage: str) -> dict | None:
    """||y||^2 and SSE against a null fitted on a different split, for one stage."""
    scored = _capsule(layer, FALLBACK_NULL_SPLIT)
    null_from = _capsule(layer, NULL_SPLIT)
    if not scored.exists():
        return None
    if not null_from.exists():
        null_from, null_split = scored, "teacher_fit (self; no disjoint split captured)"
    else:
        null_split = NULL_SPLIT
    with np.load(scored) as data:
        if stage not in data:
            return None
        y = np.asarray(data[stage], dtype=np.float64).reshape(-1, data[stage].shape[-1])
    with np.load(null_from) as data:
        fit = np.asarray(data[stage], dtype=np.float64).reshape(-1, y.shape[1])
    # index_scores is half masked by construction; a null over non-finite entries is not a
    # null, so those stages are dropped rather than silently NaN-propagated.
    if not np.isfinite(y).all() or not np.isfinite(fit).all():
        return None
    mean = fit.mean(axis=0)
    difference = y - mean
    constant = np.broadcast_to(mean, y.shape)
    return {
        "target_sse": float((y * y).sum()),
        "null_sse": float((difference * difference).sum()),
        "constant_null_raw_cosine": float(
            (y * constant).sum()
            / max(np.linalg.norm(y) * np.linalg.norm(constant), 1e-30)),
        "null_fitted_on": null_split,
        "positions": int(y.shape[0]),
    }


def corrected_skill(relative_error: float, stats: dict) -> float:
    """Exact, from a sealed relative error.  No prediction needs to be reconstructed."""
    return 1.0 - (relative_error ** 2) * stats["target_sse"] / max(stats["null_sse"], 1e-30)


def rows() -> list[dict]:
    return [json.loads(line) for line in MEASUREMENTS.read_text().splitlines() if line.strip()]


def run() -> dict:
    cache: dict[tuple[int, str], dict | None] = {}
    corrections, seen = [], set()
    for row in rows():
        propagate = row.get("propagate")
        pack = row.get("pack")
        if not propagate or not pack:
            continue
        layer = int(pack["layers"][0])
        key = (pack["window"], round(pack["complete_bpw"], 6), bool(row.get("remeasured")))
        if key in seen:
            continue
        seen.add(key)

        stages = {}
        for stage, measured in propagate["stages"].items():
            if "relative_error" not in measured:
                continue
            if (layer, stage) not in cache:
                cache[(layer, stage)] = _stats(layer, stage)
            stats = cache[(layer, stage)]
            if stats is None:
                continue
            skill = corrected_skill(float(measured["relative_error"]), stats)
            stages[stage] = {
                "sealed_raw_cosine": measured.get("cosine"),
                "sealed_relative_error": measured["relative_error"],
                "constant_null_raw_cosine": stats["constant_null_raw_cosine"],
                "corrected_null_relative_skill": skill,
                "beats_constant": bool(skill > 0.0),
                "worse_than_predicting_zero": bool(measured["relative_error"] >= 1.0),
                "null_fitted_on": stats["null_fitted_on"],
                "positions": stats["positions"],
            }

        block = stages.get(f"layer_{layer:02d}/block_output")
        moe = stages.get(f"layer_{layer:02d}/post_moe")
        corrections.append({
            "artifact": pack["artifact"],
            "window": pack["window"],
            "rung": pack["rung"],
            "layer": layer,
            "complete_bpw": pack["complete_bpw"],
            "packed_bpw": pack["packed_bpw"],
            "remeasured_row": bool(row.get("remeasured")),
            "stages": stages,
            "headline": {
                "block_output_skill": block["corrected_null_relative_skill"] if block else None,
                "block_output_beats_constant": block["beats_constant"] if block else None,
                "post_moe_skill": moe["corrected_null_relative_skill"] if moe else None,
                "post_moe_beats_constant": moe["beats_constant"] if moe else None,
            },
            "verdict": (
                "SUPERSEDED_NEGATIVE_WORSE_THAN_CONSTANT"
                if block and not block["beats_constant"]
                else "SUPERSEDED_POSITIVE" if block else "NO_BLOCK_STAGE"),
            "invalidated_claim": "partial fidelity inferred from raw activation cosine",
            "preserved_unchanged": ["artifact bytes", "complete_bpw", "packed_bpw",
                                    "tensor coverage", "pack verification"],
        })

    LEDGER.write_text("".join(
        json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "schema": "hawking.glm52.metric_correction.v1", **entry},
                   sort_keys=True) + "\n"
        for entry in corrections))
    return {
        "rows_corrected": len(corrections),
        "all_block_outputs_negative": all(
            entry["headline"]["block_output_beats_constant"] is False
            for entry in corrections if entry["headline"]["block_output_beats_constant"]
            is not None),
        "best_block_skill": max(
            (entry["headline"]["block_output_skill"] for entry in corrections
             if entry["headline"]["block_output_skill"] is not None), default=None),
        "best_block_skill_bpw": max(
            ((entry["headline"]["block_output_skill"], entry["complete_bpw"])
             for entry in corrections
             if entry["headline"]["block_output_skill"] is not None), default=(None, None))[1],
        "ledger": str(LEDGER),
    }


def selftest() -> int:
    # A prediction equal to the null must correct to exactly zero skill, and one equal to
    # the target must correct to exactly one, from relative error alone.
    generator = np.random.default_rng(0)
    y = generator.standard_normal((256, 32)) + 4.0
    mean = y.mean(axis=0)
    stats = {"target_sse": float((y * y).sum()),
             "null_sse": float(((y - mean) ** 2).sum())}
    null_relative = float(np.linalg.norm(np.broadcast_to(mean, y.shape) - y)
                          / np.linalg.norm(y))
    assert abs(corrected_skill(null_relative, stats)) < 1e-9
    assert abs(corrected_skill(0.0, stats) - 1.0) < 1e-12

    # A relative error of 1.0 means "no better than predicting zero", which on a
    # mean-dominated target is far worse than the constant and must correct to a large
    # negative number rather than a plausible-looking cosine.
    assert corrected_skill(1.0, stats) < -5.0, corrected_skill(1.0, stats)

    print(json.dumps({"selftest": "PASS",
                      "null_relative_error": round(null_relative, 6),
                      "skill_at_relative_error_1": round(corrected_skill(1.0, stats), 3)}))
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if command == "run":
        print(json.dumps(run(), indent=2))
    elif command == "selftest":
        raise SystemExit(selftest())
    else:
        raise SystemExit(f"unknown command: {command}")
