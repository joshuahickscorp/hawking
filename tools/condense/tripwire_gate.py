#!/usr/bin/env python3.12
"""Fail-closed capability gate for Hawking's deterministic multi_eval suite.

Policy v1 is intentionally small and interpretable: relative to the same model's f16 baseline,
a condensed candidate may lose at most ONE of 22 total items and at most ONE item inside any
individual QA/cloze/math/code family. The aggregate bound prevents four separate one-item losses;
the per-family bound prevents gains in easy tasks from hiding a localized collapse.
"""
from __future__ import annotations

import json
import math
import sys


TASK_COUNTS = {"qa": 6, "cloze": 5, "math": 6, "code": 5}
TOTAL_ITEMS = sum(TASK_COUNTS.values())
MAX_AGGREGATE_LOSSES = 1
MAX_PER_TASK_LOSSES = 1
# multi_eval rounds rates to four decimals, so express limits at that same resolution.
AGGREGATE_DROP_LIMIT = round(MAX_AGGREGATE_LOSSES / TOTAL_ITEMS, 4)
TASK_DROP_LIMITS = {
    task: round(MAX_PER_TASK_LOSSES / count, 4) for task, count in TASK_COUNTS.items()
}
EPSILON = 1e-6


def policy():
    return {
        "schema": "hawking.tripwire_policy.v1",
        "suite": "hawking.multi_eval.v1",
        "task_counts": dict(TASK_COUNTS),
        "total_items": TOTAL_ITEMS,
        "max_aggregate_losses": MAX_AGGREGATE_LOSSES,
        "max_per_task_losses": MAX_PER_TASK_LOSSES,
        "aggregate_drop_limit": AGGREGATE_DROP_LIMIT,
        "task_drop_limits": dict(TASK_DROP_LIMITS),
        "rule": "candidate loses <=1/22 overall and <=1 item within every task family",
    }


def validate_result(result):
    problems = []
    if not isinstance(result, dict):
        return {"ok": False, "problems": ["result is not an object"]}
    if result.get("n") != TOTAL_ITEMS:
        problems.append(f"n={result.get('n')!r}, expected {TOTAL_ITEMS}")
    declared_counts = result.get("task_n")
    # Legacy multi_eval rows predate task_n but are still identifiable by exact keys, n, and the
    # weighted aggregate consistency check below. New rows always carry task_n.
    if declared_counts is not None and declared_counts != TASK_COUNTS:
        problems.append(f"task_n={declared_counts!r}, expected {TASK_COUNTS!r}")
    per_task = result.get("per_task")
    if not isinstance(per_task, dict) or set(per_task) != set(TASK_COUNTS):
        problems.append(f"per_task keys must be exactly {sorted(TASK_COUNTS)}")
        per_task = {}
    values = {}
    for task in TASK_COUNTS:
        value = per_task.get(task)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            problems.append(f"{task} score is not finite in [0,1]: {value!r}")
        else:
            values[task] = float(value)
    aggregate = result.get("aggregate")
    if isinstance(aggregate, bool) or not isinstance(aggregate, (int, float)) \
            or not math.isfinite(float(aggregate)) or not 0.0 <= float(aggregate) <= 1.0:
        problems.append(f"aggregate is not finite in [0,1]: {aggregate!r}")
    elif len(values) == len(TASK_COUNTS):
        recomputed = sum(values[task] * TASK_COUNTS[task] for task in TASK_COUNTS) / TOTAL_ITEMS
        if abs(float(aggregate) - recomputed) > 0.00011:
            problems.append(
                f"aggregate {float(aggregate):.4f} inconsistent with weighted tasks {recomputed:.4f}"
            )
    return {"ok": not problems, "problems": problems}


def validate_baseline(result, expected_label=None):
    check = validate_result(result)
    problems = list(check["problems"])
    if isinstance(result, dict):
        if result.get("schema") != "hawking.multi_eval.v1" \
                or result.get("suite") != "hawking.multi_eval.v1":
            problems.append("baseline requires hawking.multi_eval.v1 schema/suite")
        if result.get("override") is not None or result.get("adapter") is not None:
            problems.append("f16 baseline must not contain an override or adapter")
        if expected_label is not None and result.get("label") != expected_label:
            problems.append(f"baseline label={result.get('label')!r}, expected {expected_label!r}")
    return {"ok": not problems, "problems": problems}


def compare(baseline, candidate):
    base_check = validate_result(baseline)
    cand_check = validate_result(candidate)
    result = {
        "schema": "hawking.tripwire_gate.v1",
        "status": "fail",
        "policy": policy(),
        "baseline_valid": base_check["ok"],
        "candidate_valid": cand_check["ok"],
        "problems": [
            *[f"baseline: {problem}" for problem in base_check["problems"]],
            *[f"candidate: {problem}" for problem in cand_check["problems"]],
        ],
    }
    if not (base_check["ok"] and cand_check["ok"]):
        return result
    aggregate_drop = round(float(baseline["aggregate"]) - float(candidate["aggregate"]), 4)
    task_drops = {
        task: round(float(baseline["per_task"][task]) - float(candidate["per_task"][task]), 4)
        for task in TASK_COUNTS
    }
    result["aggregate_drop"] = aggregate_drop
    result["task_drops"] = task_drops
    if aggregate_drop > AGGREGATE_DROP_LIMIT + EPSILON:
        result["problems"].append(
            f"aggregate drop {aggregate_drop:.4f} > {AGGREGATE_DROP_LIMIT:.4f} (one item)"
        )
    for task, drop in task_drops.items():
        if drop > TASK_DROP_LIMITS[task] + EPSILON:
            result["problems"].append(
                f"{task} drop {drop:.4f} > {TASK_DROP_LIMITS[task]:.4f} (one item)"
            )
    result["status"] = "pass" if not result["problems"] else "fail"
    return result


def selftest():
    base = {
        "n": 22, "task_n": dict(TASK_COUNTS),
        "per_task": {"qa": 1.0, "cloze": 1.0, "math": 1.0, "code": 1.0},
        "aggregate": 1.0,
    }
    one_loss = {
        "n": 22, "task_n": dict(TASK_COUNTS),
        "per_task": {"qa": round(5 / 6, 4), "cloze": 1.0, "math": 1.0, "code": 1.0},
        "aggregate": round(21 / 22, 4),
    }
    two_losses = {
        "n": 22, "task_n": dict(TASK_COUNTS),
        "per_task": {"qa": round(4 / 6, 4), "cloze": 1.0, "math": 1.0, "code": 1.0},
        "aggregate": round(20 / 22, 4),
    }
    localized_base = {
        "n": 22, "task_n": dict(TASK_COUNTS),
        "per_task": {"qa": 1.0, "cloze": 0.6, "math": 1.0, "code": 1.0},
        "aggregate": round((6 + 3 + 6 + 5) / 22, 4),
    }
    localized = {
        "n": 22, "task_n": dict(TASK_COUNTS),
        "per_task": {"qa": round(4 / 6, 4), "cloze": 1.0, "math": 1.0, "code": 1.0},
        # Two cloze gains hold aggregate flat but cannot hide a two-item QA collapse.
        "aggregate": round((4 + 5 + 6 + 5) / 22, 4),
    }
    assert compare(base, base)["status"] == "pass"
    assert compare(base, one_loss)["status"] == "pass"
    assert compare(base, two_losses)["status"] == "fail"
    localized_gate = compare(localized_base, localized)
    assert localized_gate["aggregate_drop"] == 0.0 and localized_gate["status"] == "fail"
    assert compare(None, one_loss)["status"] == "fail"
    legacy = dict(one_loss); legacy.pop("task_n")
    assert validate_result(legacy)["ok"]
    baseline = {**base, "schema": "hawking.multi_eval.v1", "suite": "hawking.multi_eval.v1",
                "label": "7B-f16", "override": None, "adapter": None}
    assert validate_baseline(baseline, "7B-f16")["ok"]
    assert not validate_baseline({**baseline, "override": "candidate.safetensors"},
                                 "7B-f16")["ok"]
    print("tripwire_gate.py selftest OK")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    print(json.dumps(policy(), indent=2, sort_keys=True))
