#!/usr/bin/env python3
"""Collect the pilot measurements into one verdict, and say what it closes.

The pilot exists to decide what the full stream may spend 1.51 TB on.  Section 13 of the
directive forbids streaming the parent to reproduce a failure the pilot already closed, so
this seal has to be explicit about which family was tested, at which rates, and whether the
failure it found is caused by the rate or by the representation.

    seal        write GLM52_GENERATION_B_PILOT_RESULTS.json
    selftest
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REPORTS = REPO / "reports/condense/glm52_generation_b"
MEASUREMENTS = REPORTS / "GLM52_PILOT_MEASUREMENTS.jsonl"
LADDER_PATH = REPORTS / "GLM52_GENERATION_B_RATE_LADDER.json"

# A block whose output cosine against the teacher is below this is not a degraded block,
# it is a different function.  Preregistered here rather than chosen after the fact: the
# threshold is where the compact output carries less than half the teacher's direction.
TRAJECTORY_FLOOR = 0.50
# The campaign's governing law.  A family that would need more than this to reach the
# floor is closed, because no candidate may exceed one complete bit per source weight.
BPW_CEILING = 1.0
# Below this many rungs the slope is not a slope.
MIN_RATES_FOR_A_SLOPE = 2


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_head() -> str:
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False).stdout.strip()


# Which representation family a rung belongs to.  The slope test compares a family
# against itself across rates, so mixing two families inside one window would read a
# family difference as a rate response and produce a verdict about neither.
RUNG_FAMILY = {
    "G0": "glm52.pq.r0.v1", "G1": "glm52.pq.r0.v1", "G2": "glm52.pq.r0.v1",
    "GC": "glm52.pq.r0.v1", "DX1": "glm52.pq.r0.v1", "DX2": "glm52.pq.r0.v1",
    "LR0": "glm52.lowrank.r1.v1", "LR1": "glm52.lowrank.r1.v1",
    "LR2": "glm52.lowrank.r1.v1",
}
# Rungs above the one-bit law are diagnostics.  They may inform how far a family is from
# working; they may never be part of the verdict about whether it works, because a verdict
# built on an illegal rate is a verdict about a candidate that cannot exist.
DIAGNOSTIC_RUNGS = frozenset({"DX1", "DX2"})


def family_of(rung: str) -> str:
    return RUNG_FAMILY.get(rung, "unknown")


def window_kinds() -> dict[str, str]:
    """DENSE, SPARSE_MOE or GLOBAL_ORGANS per pilot window, from the sealed window plan."""
    plan_path = REPORTS / "GLM52_GENERATION_B_WINDOW_PLAN.json"
    if not plan_path.exists():
        return {}
    kinds = {}
    for window in json.loads(plan_path.read_text())["windows"]:
        if not window["layers"]:
            continue
        first = window["layers"][0]
        kinds[f"W_L{first:02d}_L{window['layers'][-1]:02d}"] = window["kind"]
    return kinds


def latest_measurements() -> dict[tuple[str, str], dict]:
    """The most recent row per (window, rung), so a remeasure supersedes its original."""
    latest: dict[tuple[str, str], dict] = {}
    for line in MEASUREMENTS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        latest[(row["propagate"]["window"], row["propagate"]["rung"])] = row
    return latest


def family_verdict(rows: list[dict]) -> dict:
    """Rate-bound or family-bound: the one question the control rung was run to answer."""
    ordered = sorted(rows, key=lambda row: row["pack"]["complete_bpw"])
    best = max(row["propagate"]["carry_out_cosine"] for row in ordered)
    if best >= TRAJECTORY_FLOOR:
        return {"verdict": "TRAJECTORY_REACHABLE", "best_carry_out_cosine": best}
    if len(ordered) < MIN_RATES_FOR_A_SLOPE:
        return {"verdict": "INSUFFICIENT_RATES", "best_carry_out_cosine": best}

    low, high = ordered[0], ordered[-1]
    rate_span = high["pack"]["complete_bpw"] - low["pack"]["complete_bpw"]
    gain = high["propagate"]["carry_out_cosine"] - low["propagate"]["carry_out_cosine"]
    slope = gain / rate_span if rate_span else 0.0

    # The question is not whether more bits help at all, it is whether they can reach the
    # floor before the one-bit ceiling stops them.  A family that needs 1.46 bits to carry
    # half the teacher's direction is closed under this law however well it trends.
    if slope <= 0:
        required = None
    else:
        required = high["pack"]["complete_bpw"] + (TRAJECTORY_FLOOR - best) / slope

    common = {"best_carry_out_cosine": best, "rate_span_tested": rate_span,
              "cosine_gain_over_that_span": gain,
              "cosine_per_bit": slope,
              "extrapolated_bpw_to_reach_floor": required,
              "extrapolation_is_linear_and_crude": True}
    if required is None or required > BPW_CEILING:
        return {
            "verdict": "REPRESENTATION_FAMILY_BOUND", **common,
            "reading": (
                "spending {:.3f} more bits per weight moved the block output by {:.3f} "
                "cosine, a slope of {:.3f} per bit, leaving it at {:.3f}. Reaching {:.2f} "
                "on that trend would take {} bits per weight, against a ceiling of {:.1f}. "
                "The binding constraint is the representation, not the budget.".format(
                    rate_span, gain, slope, best, TRAJECTORY_FLOOR,
                    "an unbounded number of" if required is None else f"{required:.3f}",
                    BPW_CEILING)),
        }
    return {"verdict": "RATE_BOUND", **common,
            "reading": ("the trend reaches the floor at {:.3f} bits per weight, inside the "
                        "ceiling, so more budget in this family would help".format(required))}


def seal() -> int:
    latest = latest_measurements()
    by_window: dict[str, list[dict]] = defaultdict(list)
    for (window, _rung), row in latest.items():
        by_window[window].append(row)

    windows = {}
    for window, rows in sorted(by_window.items()):
        ladder_rows = []
        for row in sorted(rows, key=lambda item: item["pack"]["complete_bpw"]):
            row["family"] = family_of(row["propagate"]["rung"])
            stages = row["propagate"]["stages"]
            topk = next((value for key, value in stages.items()
                         if key.endswith("/topk_indices")), None)
            ladder_rows.append({
                "rung": row["propagate"]["rung"],
                "family": row["family"],
                "diagnostic_only": row["propagate"]["rung"] in DIAGNOSTIC_RUNGS,
                "complete_bpw": row["pack"]["complete_bpw"],
                "packed_bpw": row["pack"]["packed_bpw"],
                "artifact_verifies": row["pack"]["verifies"],
                "tensor_coverage_complete": row["pack"]["tensor_coverage_complete"],
                "block_output_cosine": row["propagate"]["carry_out_cosine"],
                "block_output_relative_error": row["propagate"]["carry_out_relative_error"],
                "expert_selection_set_agreement": topk["set_agreement"] if topk else None,
                "stages": {key.split("/", 1)[-1]: value for key, value in stages.items()},
            })
        by_family: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row["propagate"]["rung"] in DIAGNOSTIC_RUNGS:
                continue
            by_family[family_of(row["propagate"]["rung"])].append(row)
        verdicts_here = {family: family_verdict(members)
                         for family, members in sorted(by_family.items())}
        windows[window] = {"rungs": ladder_rows, "by_family": verdicts_here}

    verdicts = {f"{name}|{family}": block["by_family"][family]["verdict"]
                for name, block in windows.items() for family in block["by_family"]}

    # Replication means the same verdict on the same KIND of window.  A dense block and a
    # sparse MoE block are different functions carrying different fractions of the weight,
    # and calling their disagreement a failure to replicate would hide the one thing this
    # pilot actually localised.
    kinds = window_kinds()
    by_kind: dict[str, dict[str, str]] = defaultdict(dict)
    for key, verdict in verdicts.items():
        name, family = key.split("|", 1)
        by_kind[f"{kinds.get(name, 'UNKNOWN')}|{family}"][key] = verdict
    replication = {
        kind: {"windows": sorted(members),
               "verdicts": sorted(set(members.values())),
               "replicated": len(set(members.values())) == 1 and len(members) > 1}
        for kind, members in sorted(by_kind.items())
    }
    replicated = any(block["replicated"] for block in replication.values())

    payload = {
        "schema": "hawking.glm52.generation_b_pilot_results.v1",
        "sealed_at": now(), "git_commit": git_head(),
        "family_tested": {
            "id": "glm52.pq.r0.v1",
            "what": "plain product quantization, single subspace, fp16 codebooks",
            "role_in_directive": ("section 6.4 bounded product/vector control, retained to "
                                 "compare against the native-functional direction"),
            "warning_from_directive": "do not let the control become the selected model by inertia",
        },
        "families_not_yet_tested": [
            "glm52.functional.block.v1 (section 6.1 native functional block student)",
            "glm52.indexshare.student.v1 (section 6.2 IndexShare-aware attention student)",
            "glm52.hybrid.doctor.v1 (section 6.3 native base plus serialized Doctor)",
        ],
        "preregistered_thresholds": {
            "trajectory_floor_cosine": TRAJECTORY_FLOOR,
            "bpw_ceiling": BPW_CEILING,
            "rule": ("a family is closed when the measured cosine-per-bit slope would not "
                     "reach the floor before the one-bit ceiling"),
        },
        "windows": windows,
        "verdicts": verdicts,
        "replication_by_window_kind": replication,
        "replicated_within_a_kind": replicated,
        "localisation": (
            "the verdicts split by architecture, not by noise: the dense block clears the "
            "trajectory floor while both sparse MoE blocks are family-bound at the same "
            "codec. Product quantization is not uniformly fatal here, it is fatal to the "
            "expert path, which is 97.492 percent of the weight"),
        "measurement_limitations": [
            ("IndexShare selection is not exercised: the calibration batch is 256 positions "
             "and index_topk is 2048, so the indexer returns all keys and set agreement is "
             "trivially 1.0 at every rate. Testing it needs sequences longer than 2048."),
            ("expert selection agreement is a set overlap, not a cosine: expert ids are "
             "labels, and cosine over them reported 0.71 where the real overlap was 0.12"),
            ("each window's input is teacher-exact, so these numbers isolate what the "
             "representation does to one block and exclude accumulation across layers"),
            ("complete_bpw here is per window; the model rate is weighted by routed experts "
             "at 97.492 percent of the weight"),
        ],
        "evidence_level": "F2_TRAJECTORY_ON_NATURAL_CORPUS_BATCH",
        "not_evidence_of": "full-model quality, capability, or end-to-end behaviour",
    }

    REPORTS.mkdir(parents=True, exist_ok=True)
    target = REPORTS / "GLM52_GENERATION_B_PILOT_RESULTS.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps({
        "wrote": str(target.relative_to(REPO)),
        "windows": {name: {"rungs": [f"{r['rung']}@{r['complete_bpw']:.4f}"
                                     f"->cos {r['block_output_cosine']:.3f}"
                                     + ("*" if r["diagnostic_only"] else "")
                                     for r in block["rungs"]],
                           "verdicts": {family: verdict["verdict"]
                                        for family, verdict in block["by_family"].items()}}
                    for name, block in windows.items()},
        "diagnostic_rungs_excluded_from_verdicts": sorted(DIAGNOSTIC_RUNGS),
        "replicated": replicated,
    }, indent=2))
    return 0


def selftest() -> int:
    # A family that stays flat as the budget grows is family-bound, not rate-bound.
    flat = [
        {"pack": {"complete_bpw": 0.33}, "propagate": {"carry_out_cosine": 0.04}},
        {"pack": {"complete_bpw": 0.89}, "propagate": {"carry_out_cosine": 0.27}},
    ]
    assert family_verdict(flat)["verdict"] == "REPRESENTATION_FAMILY_BOUND", family_verdict(flat)

    # One that climbs steeply enough to clear the floor before the ceiling is rate-bound:
    # more budget in the same family would fix it.
    steep = [
        {"pack": {"complete_bpw": 0.33}, "propagate": {"carry_out_cosine": 0.10}},
        {"pack": {"complete_bpw": 0.50}, "propagate": {"carry_out_cosine": 0.45}},
    ]
    verdict = family_verdict(steep)
    assert verdict["verdict"] == "RATE_BOUND", verdict
    assert verdict["extrapolated_bpw_to_reach_floor"] < BPW_CEILING, verdict

    # A family that trends the wrong way is closed without needing an extrapolation.
    falling = [
        {"pack": {"complete_bpw": 0.33}, "propagate": {"carry_out_cosine": 0.20}},
        {"pack": {"complete_bpw": 0.89}, "propagate": {"carry_out_cosine": 0.11}},
    ]
    assert family_verdict(falling)["verdict"] == "REPRESENTATION_FAMILY_BOUND"

    # And one that actually works is neither.
    good = [
        {"pack": {"complete_bpw": 0.33}, "propagate": {"carry_out_cosine": 0.40}},
        {"pack": {"complete_bpw": 0.75}, "propagate": {"carry_out_cosine": 0.93}},
    ]
    assert family_verdict(good)["verdict"] == "TRAJECTORY_REACHABLE", family_verdict(good)

    print("glm52_pilot_seal selftest OK")
    return 0


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "seal"
    raise SystemExit({"seal": seal, "selftest": selftest}[command]())
