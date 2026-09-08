#!/usr/bin/env python3.12
"""G016 role_suitability — a per-role requirements vector, not a grade.

The campaign Pareto frontier carries `role_suitability` with no coverage.
This module fills it from evidence already on disk. It does not invent a
1-10 score, a weighted sum, a letter grade, or any other blend. Different
role, different frontier.

    python3.12 tools/future/role_suitability.py --selftest
    python3.12 tools/future/role_suitability.py --build

HCLI Resident is the one role this campaign can measure today. MAGNETAR
and PULSAR are named only as absent contracts: git grep over HEAD found
no requirements for them, and an invented contract is worse than none.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from tools.future._common import write_receipt  # noqa: E402

RECORDED_BY = "tools/future/role_suitability.py"
RECEIPT_NAME = "G016_ROLE_SUITABILITY.json"
SCHEMA = "hawking.future.role_suitability.v1"
VERSION = 1

RELIABILITY_RECEIPT = "receipts/future/G016_RELIABILITY_AXIS.json"
G007_RECEIPT = "receipts/future/G007_ROUND_PREFILL_ECONOMICS.json"
AUTONOMY_SOURCE = "tools/future/autonomy_ratio.py"

HCLI_RESIDENT = "hcli_resident"
SEALED_MODEL_SUFFIX = "hawking-native.sealed-3.14.json"

# Named requirements for the HCLI Resident role. Order is the contract order.
HCLI_RESIDENT_REQUIREMENTS: tuple[str, ...] = (
    "tool_reliability",
    "instruction_following",
    "repair_behaviour",
    "context_economy",
    "latency",
    "scientific_target_selection",
)

# Keys that would mean we blended the vector into a score. A hit is a REJECT.
BLEND_KEYS = frozenset({
    "score", "blend", "blended_score", "weighted_sum", "weights",
    "weight", "letter_grade", "grade", "gpa", "stars", "suitability_score",
    "overall_score", "role_score", "fitness",
})


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def _git_common_parent() -> Path | None:
    try:
        raw = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=ROOT, capture_output=True, text=True, check=False, timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None
    if not raw:
        return None
    common = Path(raw)
    if not common.is_absolute():
        common = (ROOT / common).resolve()
    else:
        common = common.resolve()
    if common.name == ".git":
        return common.parent
    if common.parent.name == ".git":
        return common.parent.parent
    return common.parent


def hcli_receipts_dir() -> Path:
    """HCLI receipts live next to the main checkout, not in every worktree."""
    env = os.environ.get("HCLI_RECEIPTS") or os.environ.get("HAWKING_HCLI_RECEIPTS")
    if env:
        return Path(env)
    candidates = [ROOT / ".hcli" / "receipts"]
    main = _git_common_parent()
    if main is not None:
        candidates.append(main / ".hcli" / "receipts")
    for cand in candidates:
        if cand.is_dir() and any(cand.glob("*.json")):
            return cand
    return candidates[0]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


# ---------------------------------------------------------------------------
# cells — load-bearing null vs zero
# ---------------------------------------------------------------------------

def measured_or_null(
    value: Any,
    *,
    cause: str | None,
    source_receipt: str,
    source_field: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One requirement cell. A missing measurement is null with a named cause.

    0 is a legal measured value (zero recoveries, zero self-selected targets).
    Unmeasured is None plus `cause`. Conflating those two is how "unmeasured"
    starts looking like "bad".
    """
    extra = dict(extra or {})
    extra.pop("value", None)
    extra.pop("cause", None)
    extra.pop("source_receipt", None)
    extra.pop("source_field", None)
    if value is None:
        if not cause:
            raise ValueError(
                f"null requirement {source_field!r} must name a cause; "
                "silence is not a refusal"
            )
        cell = {
            "value": None,
            "cause": cause,
            "source_receipt": source_receipt,
            "source_field": source_field,
        }
        cell.update(extra)
        return cell  # LOAD_BEARING_NULL
    cell = {
        "value": value,
        "cause": None,
        "source_receipt": source_receipt,
        "source_field": source_field,
    }
    cell.update(extra)
    return cell


def is_covered(cell: Mapping[str, Any]) -> bool:
    """True when the requirement carries a measured value (including 0)."""
    value = cell.get("value")
    if value is None:
        return False
    if isinstance(value, dict):
        return any(v is not None for v in value.values())
    return True


def n_measured_requirements(vector: Mapping[str, Any]) -> int:
    return sum(1 for cell in vector.values() if is_covered(cell))


def covered_names(vector: Mapping[str, Any]) -> list[str]:
    return [name for name, cell in vector.items() if is_covered(cell)]


def uncovered_named(vector: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for name, cell in vector.items():
        if is_covered(cell):
            continue
        out.append({
            "requirement": name,
            "cause": cell.get("cause") or "unmeasured",
            "source_receipt": cell.get("source_receipt"),
            "source_field": cell.get("source_field"),
        })
    return out


def comparable_value_for_axis(vector: Mapping[str, Any]) -> float | None:
    """Named count of measured requirements, or None when the covered set is empty.

    This is coverage of the contract, not a blend of the requirement values.
    An empty covered set is unmeasured (None), not a zero score.
    """
    n = n_measured_requirements(vector)
    if n == 0:
        return None
    return float(n)  # LOAD_BEARING_HARVEST


def blend_keys_found(obj: Any, *, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}.{key}" if path else str(key)
            if str(key) in BLEND_KEYS:
                hits.append(here)
            hits.extend(blend_keys_found(value, path=here))
    elif isinstance(obj, list):
        for i, value in enumerate(obj[:200]):
            hits.extend(blend_keys_found(value, path=f"{path}[{i}]"))
    return hits


# ---------------------------------------------------------------------------
# live evidence (receipts already on disk — no ModelLake, no GPU)
# ---------------------------------------------------------------------------

def _sealed_reliability_row(doc: Mapping[str, Any]) -> dict[str, Any] | None:
    for row in doc.get("models") or []:
        if not isinstance(row, dict):
            continue
        model = str(row.get("model") or "")
        if model.endswith(SEALED_MODEL_SUFFIX) or model.endswith("/" + SEALED_MODEL_SUFFIX):
            return row
    return None


def _rate_payload(row: Mapping[str, Any], field: str) -> dict[str, Any]:
    cell = row.get(field) if isinstance(row.get(field), dict) else {}
    return {
        "value": cell.get("value"),
        "cause": cell.get("missing"),
        "n": cell.get("n"),
        "d": cell.get("d"),
        "predicate": cell.get("predicate"),
        "keys": cell.get("keys"),
    }


def scientific_target_selection_live() -> dict[str, Any]:
    """Counters from autonomy_ratio.report's hcli_autonomy block, not a ratio.

    autonomy_ratio forbids dividing these. We cite them separately.
    """
    import autonomy_ratio as ar

    receipts = hcli_receipts_dir()
    source_field_sel = "report()['hcli_autonomy']['self_selected_target']"
    source_field_meas = "report()['hcli_autonomy']['selected_AND_measured']"
    if not receipts.is_dir():
        return {
            "value": None,
            "cause": f"HCLI receipts directory does not exist: {receipts}",
            "source_receipt": AUTONOMY_SOURCE,
            "source_field": source_field_sel,
            "window_start": getattr(ar, "GOAL_START", None),
            "receipts_dir": str(receipts),
        }
    saved = ar.RECEIPTS
    try:
        ar.RECEIPTS = receipts
        rounds = ar.hcli_rounds(ar._epoch(ar.GOAL_START))
    finally:
        ar.RECEIPTS = saved
    selected = sum(1 for r in rounds if r["open_target"])
    sel_and_meas = sum(1 for r in rounds if r["open_target"] and r["measured"])
    return {
        "value": {
            "self_selected_target": selected,
            "selected_AND_measured": sel_and_meas,
        },
        "cause": None,
        "source_receipt": AUTONOMY_SOURCE,
        "source_field": f"{source_field_sel} and {source_field_meas}",
        "source_fields": {
            "self_selected_target": source_field_sel,
            "selected_AND_measured": source_field_meas,
        },
        "n_rounds": len(rounds),
        "window_start": ar.GOAL_START,
        "receipts_dir": str(receipts),
        "predicate": (
            "self_selected_target = count of engine receipts in the goal window "
            "whose goal contains 'CHOOSE YOUR OWN TARGET'; selected_AND_measured "
            "= that conjunct hcli.engine.is_accepted_measurement. The two are "
            "never divided."
        ),
    }


def live_hcli_resident_evidence() -> dict[str, Any]:
    """Pull each requirement from the receipt/field named in the contract."""
    rel_path = ROOT / RELIABILITY_RECEIPT
    g007_path = ROOT / G007_RECEIPT
    rel = _read_json(rel_path)
    g007 = _read_json(g007_path)
    row = _sealed_reliability_row(rel) if rel else None

    evidence: dict[str, Any] = {
        "model": (row or {}).get("model"),
        "n_runs": (row or {}).get("n_runs"),
        "endpoint": (row or {}).get("model_calls_endpoint") or (row or {}).get("endpoint"),
    }

    if rel is None:
        missing_rel = f"{RELIABILITY_RECEIPT} unreadable or absent"
        for name, field in (
            ("tool_reliability", "tool_failure_rate.value"),
            ("instruction_following", "structured_action_failure_rate.value"),
            ("repair_behaviour", "recovery_rate.value"),
        ):
            evidence[name] = {
                "value": None,
                "cause": missing_rel,
                "source_receipt": RELIABILITY_RECEIPT,
                "source_field": f"models[model ends with {SEALED_MODEL_SUFFIX}].{field}",
            }
    elif row is None:
        missing_row = (
            f"{RELIABILITY_RECEIPT} has no models[] row whose model ends with "
            f"{SEALED_MODEL_SUFFIX}"
        )
        for name, field in (
            ("tool_reliability", "tool_failure_rate.value"),
            ("instruction_following", "structured_action_failure_rate.value"),
            ("repair_behaviour", "recovery_rate.value"),
        ):
            evidence[name] = {
                "value": None,
                "cause": missing_row,
                "source_receipt": RELIABILITY_RECEIPT,
                "source_field": f"models[model ends with {SEALED_MODEL_SUFFIX}].{field}",
            }
    else:
        model = row.get("model")
        prefix = f"models[model={model}]"
        tool = _rate_payload(row, "tool_failure_rate")
        evidence["tool_reliability"] = {
            "value": tool["value"],
            "cause": tool["cause"] or (
                None if tool["value"] is not None
                else "tool_failure_rate.value is null and missing is unnamed"
            ),
            "source_receipt": RELIABILITY_RECEIPT,
            "source_field": f"{prefix}.tool_failure_rate.value",
            "n": tool["n"],
            "d": tool["d"],
            "predicate": tool["predicate"],
            "keys": tool["keys"],
        }
        so = _rate_payload(row, "structured_action_failure_rate")
        evidence["instruction_following"] = {
            "value": so["value"],
            "cause": so["cause"] or (
                None if so["value"] is not None
                else "structured_action_failure_rate.value is null and missing is unnamed"
            ),
            "source_receipt": RELIABILITY_RECEIPT,
            "source_field": f"{prefix}.structured_action_failure_rate.value",
            "n": so["n"],
            "d": so["d"],
            "predicate": so["predicate"],
            "keys": so["keys"],
            "receipt_field_name": "structured_action_failure_rate",
            "also_known_as": "structured_output_failure_rate",
        }
        rec = _rate_payload(row, "recovery_rate")
        evidence["repair_behaviour"] = {
            "value": rec["value"],
            "cause": rec["cause"] or (
                None if rec["value"] is not None
                else "recovery_rate.value is null and missing is unnamed"
            ),
            "source_receipt": RELIABILITY_RECEIPT,
            "source_field": f"{prefix}.recovery_rate.value",
            "n": rec["n"],
            "d": rec["d"],
            "predicate": rec["predicate"],
            "keys": rec["keys"],
        }

    if g007 is None:
        missing_g007 = f"{G007_RECEIPT} unreadable or absent"
        evidence["context_economy"] = {
            "value": None,
            "cause": missing_g007,
            "source_receipt": G007_RECEIPT,
            "source_field": (
                "totals.prompt_tokens_per_generated_token and "
                "totals.prefill_share_of_model_wall"
            ),
        }
        evidence["latency"] = {
            "value": None,
            "cause": missing_g007,
            "source_receipt": G007_RECEIPT,
            "source_field": "totals.prefill_tok_per_s and per_call[].call_wall_s",
        }
    else:
        totals = g007.get("totals") if isinstance(g007.get("totals"), dict) else {}
        per_call = g007.get("per_call") if isinstance(g007.get("per_call"), list) else []
        ppt = totals.get("prompt_tokens_per_generated_token")
        share = totals.get("prefill_share_of_model_wall")
        if ppt is None and share is None:
            evidence["context_economy"] = {
                "value": None,
                "cause": (
                    f"{G007_RECEIPT} totals is missing "
                    "prompt_tokens_per_generated_token and prefill_share_of_model_wall"
                ),
                "source_receipt": G007_RECEIPT,
                "source_field": (
                    "totals.prompt_tokens_per_generated_token and "
                    "totals.prefill_share_of_model_wall"
                ),
            }
        else:
            evidence["context_economy"] = {
                "value": {
                    "prompt_tokens_per_generated_token": ppt,
                    "prefill_share_of_model_wall": share,
                },
                "cause": None,
                "source_receipt": G007_RECEIPT,
                "source_field": (
                    "totals.prompt_tokens_per_generated_token and "
                    "totals.prefill_share_of_model_wall"
                ),
                "source_fields": {
                    "prompt_tokens_per_generated_token": (
                        "totals.prompt_tokens_per_generated_token"
                    ),
                    "prefill_share_of_model_wall": (
                        "totals.prefill_share_of_model_wall"
                    ),
                },
            }
        prefill = totals.get("prefill_tok_per_s")
        walls = [
            c.get("call_wall_s")
            for c in per_call
            if isinstance(c, dict) and c.get("call_wall_s") is not None
        ]
        prefills = [
            c.get("prefill_tok_per_s")
            for c in per_call
            if isinstance(c, dict) and c.get("prefill_tok_per_s") is not None
        ]
        if prefill is None and not walls:
            evidence["latency"] = {
                "value": None,
                "cause": (
                    f"{G007_RECEIPT} is missing totals.prefill_tok_per_s and "
                    "per_call[].call_wall_s"
                ),
                "source_receipt": G007_RECEIPT,
                "source_field": "totals.prefill_tok_per_s and per_call[].call_wall_s",
            }
        else:
            evidence["latency"] = {
                "value": {
                    "prefill_tok_per_s": prefill,
                    "per_call_prefill_tok_per_s": prefills,
                    "per_call_wall_s": walls,
                    "per_call_wall_s_min": min(walls) if walls else None,
                    "per_call_wall_s_max": max(walls) if walls else None,
                },
                "cause": None,
                "source_receipt": G007_RECEIPT,
                "source_field": (
                    "totals.prefill_tok_per_s, per_call[].prefill_tok_per_s, "
                    "per_call[].call_wall_s; min/max are min() and max() of "
                    "per_call[].call_wall_s, not an interpolated spread"
                ),
                "source_fields": {
                    "prefill_tok_per_s": "totals.prefill_tok_per_s",
                    "per_call_prefill_tok_per_s": "per_call[].prefill_tok_per_s",
                    "per_call_wall_s": "per_call[].call_wall_s",
                    "per_call_wall_s_min": "min(per_call[].call_wall_s)",
                    "per_call_wall_s_max": "max(per_call[].call_wall_s)",
                },
            }

    evidence["scientific_target_selection"] = scientific_target_selection_live()
    return evidence


def evaluate_hcli_resident(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    """Requirements vector for the HCLI Resident role.

    `evidence is None` is the no-evidence control: every requirement is null
    with cause 'no evidence provided for this candidate', never 0.
    """
    sources = {
        "tool_reliability": (
            RELIABILITY_RECEIPT,
            f"models[model ends with {SEALED_MODEL_SUFFIX}].tool_failure_rate.value",
        ),
        "instruction_following": (
            RELIABILITY_RECEIPT,
            f"models[model ends with {SEALED_MODEL_SUFFIX}]."
            "structured_action_failure_rate.value",
        ),
        "repair_behaviour": (
            RELIABILITY_RECEIPT,
            f"models[model ends with {SEALED_MODEL_SUFFIX}].recovery_rate.value",
        ),
        "context_economy": (
            G007_RECEIPT,
            "totals.prompt_tokens_per_generated_token and "
            "totals.prefill_share_of_model_wall",
        ),
        "latency": (
            G007_RECEIPT,
            "totals.prefill_tok_per_s and per_call[].call_wall_s",
        ),
        "scientific_target_selection": (
            AUTONOMY_SOURCE,
            "report()['hcli_autonomy']['self_selected_target'] and "
            "report()['hcli_autonomy']['selected_AND_measured']",
        ),
    }
    vector: dict[str, Any] = {}
    for name in HCLI_RESIDENT_REQUIREMENTS:
        src, field = sources[name]
        if evidence is None or name not in evidence:
            vector[name] = measured_or_null(
                None,
                cause="no evidence provided for this candidate",
                source_receipt=src,
                source_field=field,
            )
            continue
        payload = evidence[name]
        if not isinstance(payload, dict):
            vector[name] = measured_or_null(
                payload,
                cause=None,
                source_receipt=src,
                source_field=field,
            )
            continue
        extra = {
            k: v for k, v in payload.items()
            if k not in {"value", "cause", "source_receipt", "source_field"}
        }
        vector[name] = measured_or_null(
            payload.get("value"),
            cause=payload.get("cause") or (
                "no evidence provided for this candidate"
                if payload.get("value") is None else None
            ),
            source_receipt=payload.get("source_receipt") or src,
            source_field=payload.get("source_field") or field,
            extra=extra,
        )
    return vector


def candidate_record(
    *,
    role: str,
    name: str,
    model: Any,
    vector: Mapping[str, Any],
    status: str = "MEASURED",
    refusal: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    comparable = comparable_value_for_axis(vector)
    rec: dict[str, Any] = {
        "role": role,
        "name": name,
        "model": model,
        "status": status if comparable is not None else (
            "REFUSED" if status == "REFUSED" else "UNMEASURED"
        ),
        "requirements": dict(vector),
        "covered": covered_names(vector),
        "uncovered": uncovered_named(vector),
        "n_measured_requirements": n_measured_requirements(vector),
        "n_requirements": len(vector),
        "comparable_value": comparable,
        "comparable_value_is": (
            "n_measured_requirements when the covered set is non-empty; "
            "null (not 0) when nothing was measured. Not a blend of the "
            "requirement magnitudes, not a weighted sum, not a letter grade."
        ),
        "not_a_blend": True,
    }
    if status == "REFUSED" or refusal:
        rec["status"] = "REFUSED"
        rec["comparable_value"] = None
        rec["refusal"] = dict(refusal or {})
    if extra:
        for k, v in extra.items():
            rec.setdefault(k, v)
    return rec


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------

def _failing_evidence() -> dict[str, Any]:
    """Every requirement present and failing. 0 is a measured failure, not null."""
    syn = "synthetic:all_failed"
    return {
        "tool_reliability": {
            "value": 1.0,
            "cause": None,
            "source_receipt": syn,
            "source_field": "tool_failure_rate.value",
            "n": 10,
            "d": 10,
        },
        "instruction_following": {
            "value": 1.0,
            "cause": None,
            "source_receipt": syn,
            "source_field": "structured_action_failure_rate.value",
            "n": 10,
            "d": 10,
        },
        "repair_behaviour": {
            "value": 0.0,
            "cause": None,
            "source_receipt": syn,
            "source_field": "recovery_rate.value",
            "n": 0,
            "d": 10,
        },
        "context_economy": {
            "value": {
                "prompt_tokens_per_generated_token": 999.0,
                "prefill_share_of_model_wall": 1.0,
            },
            "cause": None,
            "source_receipt": syn,
            "source_field": "totals.prompt_tokens_per_generated_token",
        },
        "latency": {
            "value": {
                "prefill_tok_per_s": 0.01,
                "per_call_prefill_tok_per_s": [0.01],
                "per_call_wall_s": [9999.0],
                "per_call_wall_s_min": 9999.0,
                "per_call_wall_s_max": 9999.0,
            },
            "cause": None,
            "source_receipt": syn,
            "source_field": "totals.prefill_tok_per_s",
        },
        "scientific_target_selection": {
            "value": {
                "self_selected_target": 0,
                "selected_AND_measured": 0,
            },
            "cause": None,
            "source_receipt": syn,
            "source_field": "hcli_autonomy.self_selected_target",
        },
    }


def _partial_evidence() -> dict[str, Any]:
    """Instruction following and repair measured; everything else unnamed-missing."""
    syn = "synthetic:partial"
    return {
        "instruction_following": {
            "value": 0.5,
            "cause": None,
            "source_receipt": syn,
            "source_field": "structured_action_failure_rate.value",
        },
        "repair_behaviour": {
            "value": 0.0,
            "cause": None,
            "source_receipt": syn,
            "source_field": "recovery_rate.value",
        },
    }


def negative_control() -> dict[str, Any]:
    """Failing-measured vs no-evidence must be distinguishable. Partial stays partial."""
    failed_vec = evaluate_hcli_resident(_failing_evidence())
    none_vec = evaluate_hcli_resident(None)
    part_vec = evaluate_hcli_resident(_partial_evidence())
    failed = candidate_record(
        role=HCLI_RESIDENT, name="synthetic-all-failed", model=None,
        vector=failed_vec,
    )
    none = candidate_record(
        role=HCLI_RESIDENT, name="synthetic-no-evidence", model=None,
        vector=none_vec,
    )
    part = candidate_record(
        role=HCLI_RESIDENT, name="synthetic-partial", model=None,
        vector=part_vec,
    )
    checks: list[str] = []

    def fail(msg: str) -> None:
        checks.append(msg)

    # 1. all-failed: measured failing values, not nulls. 0 recovery is a value.
    for name, cell in failed_vec.items():
        if cell.get("value") is None:
            fail(f"all-failed {name} was null, expected a measured failing value")
    if failed_vec["repair_behaviour"]["value"] != 0.0:
        fail("all-failed repair_behaviour must be measured 0.0, not null")
    if failed_vec["scientific_target_selection"]["value"]["self_selected_target"] != 0:
        fail("all-failed self_selected_target must be measured 0, not null")
    if failed["comparable_value"] != float(len(HCLI_RESIDENT_REQUIREMENTS)):
        fail(f"all-failed comparable_value {failed['comparable_value']} "
             f"!= {float(len(HCLI_RESIDENT_REQUIREMENTS))}")
    if failed["n_measured_requirements"] != len(HCLI_RESIDENT_REQUIREMENTS):
        fail("all-failed must cover every requirement")

    # 2. no-evidence: nulls with named cause, never zeros.
    for name, cell in none_vec.items():
        if cell.get("value") is not None:
            fail(f"no-evidence {name} was {cell.get('value')!r}, expected null")
        if cell.get("value") == 0 or cell.get("value") == 0.0:
            fail(f"no-evidence {name} used 0 as a stand-in for unmeasured")
        if not cell.get("cause"):
            fail(f"no-evidence {name} is null without a named cause")
    if none["comparable_value"] is not None:
        fail(f"no-evidence comparable_value must be null, got {none['comparable_value']!r}")
    if none["n_measured_requirements"] != 0:
        fail("no-evidence n_measured_requirements must be 0 (the count of measured "
             "cells, recorded next to a null comparable_value)")
    if none["status"] != "UNMEASURED":
        fail(f"no-evidence status {none['status']!r} wanted UNMEASURED")

    # The two controls must not collapse into each other.
    if failed["comparable_value"] == none["comparable_value"]:
        fail("all-failed and no-evidence produced the same comparable_value")
    failed_vals = [failed_vec[n]["value"] for n in HCLI_RESIDENT_REQUIREMENTS]
    none_vals = [none_vec[n]["value"] for n in HCLI_RESIDENT_REQUIREMENTS]
    if failed_vals == none_vals:
        fail("all-failed and no-evidence produced the same requirement values")

    # 3. partial: covered set named, not rounded to 0 or 6.
    expected_covered = ["instruction_following", "repair_behaviour"]
    if part["covered"] != expected_covered:
        fail(f"partial covered {part['covered']} != {expected_covered}")
    if part["n_measured_requirements"] != 2:
        fail(f"partial n_measured={part['n_measured_requirements']} rounded")
    if part["comparable_value"] in (None, 0.0, float(len(HCLI_RESIDENT_REQUIREMENTS))):
        fail(f"partial comparable_value {part['comparable_value']} rounded to an extreme")
    uncovered_names = [u["requirement"] for u in part["uncovered"]]
    for name in ("tool_reliability", "context_economy", "latency",
                 "scientific_target_selection"):
        if name not in uncovered_names:
            fail(f"partial did not name uncovered {name}")
        if not part_vec[name].get("cause"):
            fail(f"partial {name} is uncovered without a cause")
    if part_vec["repair_behaviour"]["value"] != 0.0:
        fail("partial repair_behaviour 0.0 was lost (null/zero collapse)")

    for rec in (failed, none, part):
        hits = blend_keys_found(rec)
        if hits:
            fail(f"{rec['name']} leaked blend keys: {hits}")

    return {
        "passed": not checks,
        "failed_checks": checks,
        "all_failed": failed,
        "no_evidence": none,
        "partial": part,
    }


# ---------------------------------------------------------------------------
# MAGNETAR / PULSAR — search, then refuse to invent
# ---------------------------------------------------------------------------

def roles_without_contracts() -> list[dict[str, Any]]:
    """MAGNETAR and PULSAR get no requirements unless a contract exists on disk."""
    hits: dict[str, list[str]] = {"magnetar": [], "pulsar": []}
    try:
        proc = subprocess.run(
            ["git", "grep", "-i", "-l", "magnetar\\|pulsar", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=False, timeout=60,
        )
        for line in proc.stdout.splitlines():
            low = line.lower()
            if "magnetar" in low:
                hits["magnetar"].append(line)
            if "pulsar" in low:
                hits["pulsar"].append(line)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [
            {
                "role": name,
                "requirements": None,
                "cause": f"git grep failed ({type(exc).__name__}); refusing to invent a contract",
                "search": "git grep -i -l 'magnetar|pulsar' HEAD",
                "hits": [],
            }
            for name in ("magnetar", "pulsar")
        ]
    out = []
    for name, found in hits.items():
        if found:
            out.append({
                "role": name,
                "requirements": None,
                "cause": (
                    f"paths mention {name} but none define a role contract of "
                    "named requirements; refusing to invent one from a name hit"
                ),
                "search": "git grep -i -l 'magnetar|pulsar' HEAD",
                "hits": found[:20],
            })
        else:
            out.append({
                "role": name,
                "requirements": None,
                "cause": (
                    f"no role contract found on disk; git grep -i {name} over "
                    "HEAD returned 0 hits. An invented contract is worse than "
                    "an absent one."
                ),
                "search": "git grep -i -l 'magnetar|pulsar' HEAD",
                "hits": [],
            })
    return out


# ---------------------------------------------------------------------------
# mutation
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tagged_line(path: Path, tag: str) -> tuple[int, str]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    for i, line in enumerate(lines):
        if tag in line:
            return i, line
    raise AssertionError(f"{tag!r} not found in {path}")


@contextmanager
def _mutated_line(path: Path, tag: str, new_line: str) -> Iterator[dict[str, str]]:
    path = path.resolve()
    original = path.read_text(encoding="utf-8")
    before = hashlib.sha256(original.encode("utf-8")).hexdigest()
    idx, line = _tagged_line(path, tag)
    parts = original.splitlines(keepends=True)
    if idx >= len(parts) or tag not in parts[idx]:
        raise AssertionError(f"{tag} not at expected line {idx + 1}")
    newline = "\n" if parts[idx].endswith("\n") else ""
    parts[idx] = new_line.rstrip("\n") + newline
    mutated = "".join(parts)
    if mutated == original:
        raise AssertionError(f"mutation of {tag} did not change the source")
    try:
        path.write_text(mutated, encoding="utf-8")
        after = _sha256_file(path)
        if after == before:
            raise AssertionError("sha256 unchanged after mutation write")
        yield {
            "anchor": f"{path.name}:{idx + 1}:{line.rstrip()}",
            "replacement": new_line.strip(),
            "before_sha256": before,
            "after_sha256": after,
        }
    finally:
        path.write_text(original, encoding="utf-8")
        restored = _sha256_file(path)
        if restored != before:
            path.write_text(original, encoding="utf-8")
            raise AssertionError(f"failed to restore {path} after mutation")


def _probe(script: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    pyc = HERE / "__pycache__"
    for leftover in pyc.glob("role_suitability*.pyc"):
        leftover.unlink(missing_ok=True)
    for leftover in pyc.glob("campaign_pareto*.pyc"):
        leftover.unlink(missing_ok=True)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def mutation_check() -> dict[str, Any]:
    """Revert the load-bearing line, confirm a probe FAILS, restore.

    Null: substituting 0 for None makes the no-evidence candidate look failed.
    Harvest: skipping the axis set leaves n_with_axis['role_suitability'] == 0.
    """
    src = Path(__file__).read_text(encoding="utf-8")
    if "LOAD_BEARING_NULL" not in src or "LOAD_BEARING_HARVEST" not in src:
        raise AssertionError("load-bearing tags missing before mutation")

    null_probe = (
        "import sys\n"
        "sys.dont_write_bytecode = True\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from tools.future.role_suitability import evaluate_hcli_resident, "
        "HCLI_RESIDENT_REQUIREMENTS\n"
        "vec = evaluate_hcli_resident(None)\n"
        "for name in HCLI_RESIDENT_REQUIREMENTS:\n"
        "    cell = vec[name]\n"
        "    assert cell['value'] is None, (name, cell)\n"
        "    assert cell['value'] is not 0 and cell['value'] != 0.0, (name, cell)\n"
    )
    harvest_probe = (
        "import sys\n"
        "sys.dont_write_bytecode = True\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        f"sys.path.insert(0, {str(HERE)!r})\n"
        "from campaign_pareto import _Pool, harvest_role_suitability, has_value\n"
        "pool = _Pool()\n"
        "harvest_role_suitability(pool)\n"
        "n = sum(1 for p in pool.by_id.values() if has_value(p, 'role_suitability'))\n"
        "assert n > 0, {'n': n, 'ids': list(pool.by_id)}\n"
    )

    null_replacement = (
        "        return {\"value\": 0, \"cause\": None, "
        "\"source_receipt\": source_receipt, "
        "\"source_field\": source_field}  # MUTATION_ACTIVE null"
    )
    harvest_replacement = (
        "        pass  # MUTATION_ACTIVE harvest"
    )

    with _mutated_line(Path(__file__), "LOAD_BEARING_NULL", null_replacement) as null_meta:
        null_run = _probe(null_probe)
        if null_run.returncode == 0:
            raise AssertionError(
                "null mutation did not fail the no-evidence probe; "
                f"stdout={null_run.stdout!r} stderr={null_run.stderr!r}"
            )
        null_result = {
            **null_meta,
            "probe_returncode": null_run.returncode,
            "probe_failed_as_required": True,
            "stderr_tail": (null_run.stderr or null_run.stdout)[-400:],
        }

    pareto_path = HERE / "campaign_pareto.py"
    with _mutated_line(pareto_path, "LOAD_BEARING_HARVEST", harvest_replacement) as harv_meta:
        harv_run = _probe(harvest_probe)
        if harv_run.returncode == 0:
            raise AssertionError(
                "harvest mutation did not fail n_with_axis>0; "
                f"stdout={harv_run.stdout!r} stderr={harv_run.stderr!r}"
            )
        harvest_result = {
            **harv_meta,
            "probe_returncode": harv_run.returncode,
            "probe_failed_as_required": True,
            "stderr_tail": (harv_run.stderr or harv_run.stdout)[-400:],
        }

    text = Path(__file__).read_text(encoding="utf-8")
    if "LOAD_BEARING_NULL" not in text:
        raise AssertionError("LOAD_BEARING_NULL missing after restore")
    null_fn = text.split("def measured_or_null")[1].split("def ")[0]
    if "MUTATION_ACTIVE null" in null_fn:
        raise AssertionError("null mutation left in measured_or_null")
    pareto_text = pareto_path.read_text(encoding="utf-8")
    if "LOAD_BEARING_HARVEST" not in pareto_text:
        raise AssertionError("LOAD_BEARING_HARVEST missing after restore")
    harv_fn = pareto_text.split("def harvest_role_suitability")[1].split("def ")[0]
    if "MUTATION_ACTIVE harvest" in harv_fn:
        raise AssertionError("harvest mutation left in harvest_role_suitability")
    return {
        "null_vs_zero": null_result,
        "harvest": harvest_result,
        "restored": True,
        "restored_sha256_role_suitability": _sha256_file(Path(__file__).resolve()),
        "restored_sha256_campaign_pareto": _sha256_file(pareto_path.resolve()),
    }


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def hcli_resident_role_contract() -> dict[str, Any]:
    return {
        "name": HCLI_RESIDENT,
        "why_this_role": (
            "this campaign can measure it today from evidence already on disk"
        ),
        "requirements": [
            {
                "name": "tool_reliability",
                "source_receipt": RELIABILITY_RECEIPT,
                "source_field": (
                    f"models[model ends with {SEALED_MODEL_SUFFIX}]."
                    "tool_failure_rate.value"
                ),
                "direction": "lower_is_better",
                "if_null": "use the named missing cause from that field; never 0",
            },
            {
                "name": "instruction_following",
                "source_receipt": RELIABILITY_RECEIPT,
                "source_field": (
                    f"models[model ends with {SEALED_MODEL_SUFFIX}]."
                    "structured_action_failure_rate.value"
                ),
                "receipt_field_name": "structured_action_failure_rate",
                "also_known_as": "structured_output_failure_rate",
                "direction": "lower_is_better",
            },
            {
                "name": "repair_behaviour",
                "source_receipt": RELIABILITY_RECEIPT,
                "source_field": (
                    f"models[model ends with {SEALED_MODEL_SUFFIX}]."
                    "recovery_rate.value"
                ),
                "direction": "higher_is_better",
            },
            {
                "name": "context_economy",
                "source_receipt": G007_RECEIPT,
                "source_fields": {
                    "prompt_tokens_per_generated_token": (
                        "totals.prompt_tokens_per_generated_token"
                    ),
                    "prefill_share_of_model_wall": (
                        "totals.prefill_share_of_model_wall"
                    ),
                },
                "not_a_blend": True,
                "direction": "lower_is_better_per_component",
            },
            {
                "name": "latency",
                "source_receipt": G007_RECEIPT,
                "source_fields": {
                    "prefill_tok_per_s": "totals.prefill_tok_per_s",
                    "per_call_wall_s": "per_call[].call_wall_s",
                },
                "not_a_blend": True,
                "direction": "components_not_combined",
            },
            {
                "name": "scientific_target_selection",
                "source_receipt": AUTONOMY_SOURCE,
                "source_fields": {
                    "self_selected_target": (
                        "report()['hcli_autonomy']['self_selected_target']"
                    ),
                    "selected_AND_measured": (
                        "report()['hcli_autonomy']['selected_AND_measured']"
                    ),
                },
                "not_a_blend": True,
                "never": "a ratio of the two counters",
            },
        ],
        "not": [
            "a 1-10 score",
            "a weighted sum of the requirements",
            "a letter grade",
            "a blend of MAGNETAR or PULSAR (those roles have no contract on disk)",
        ],
    }


def build_live_candidate() -> dict[str, Any]:
    evidence = live_hcli_resident_evidence()
    vector = evaluate_hcli_resident(evidence)
    model = evidence.get("model")
    if isinstance(model, str) and model:
        name = Path(model).name
    else:
        name = SEALED_MODEL_SUFFIX
    rec = candidate_record(
        role=HCLI_RESIDENT,
        name=name,
        model=model,
        vector=vector,
        extra={
            "n_runs": evidence.get("n_runs"),
            "endpoint": evidence.get("endpoint"),
            "organism": f"serving:{name}",
            "kind": "serving_identity",
        },
    )
    return rec


def build_doc(*, controls: Mapping[str, Any], live: Mapping[str, Any]) -> dict[str, Any]:
    absent = roles_without_contracts()
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "purpose": (
            "Fill the campaign Pareto role_suitability axis from evidence "
            "already on disk. Role suitability is a per-role requirements "
            "vector, not a grade and not a blend. Different role: different "
            "frontier."
        ),
        "claim_boundary": (
            "Static derivation from receipts and counters already on disk. "
            "No hardware measurement. No ModelLake specimen. No interpolated "
            "value. An absent field is null with the missing key named. A "
            "blended score is a reject even if every input to it was measured."
        ),
        "pareto_axis": {
            "name": "role_suitability",
            "higher_is_better": True,
            "comparable_value": "n_measured_requirements",
            "comparable_value_is": (
                "count of contract requirements that carry a measured value "
                "(0 is a value). Harvested onto the matching serving-identity "
                "point. Null, not 0, when the covered set is empty. Not a "
                "blend of the requirement magnitudes."
            ),
            "vector_stays": list(HCLI_RESIDENT_REQUIREMENTS),
            "not_a_blend": True,
        },
        "roles": {HCLI_RESIDENT: hcli_resident_role_contract()},
        "roles_without_contracts": absent,
        "candidates": [live],
        "controls": dict(controls),
        "field_keys": {
            "tool_reliability": {
                "receipt": RELIABILITY_RECEIPT,
                "field": "models[].tool_failure_rate.value",
            },
            "instruction_following": {
                "receipt": RELIABILITY_RECEIPT,
                "field": "models[].structured_action_failure_rate.value",
            },
            "repair_behaviour": {
                "receipt": RELIABILITY_RECEIPT,
                "field": "models[].recovery_rate.value",
            },
            "context_economy": {
                "receipt": G007_RECEIPT,
                "fields": [
                    "totals.prompt_tokens_per_generated_token",
                    "totals.prefill_share_of_model_wall",
                ],
            },
            "latency": {
                "receipt": G007_RECEIPT,
                "fields": [
                    "totals.prefill_tok_per_s",
                    "per_call[].call_wall_s",
                    "per_call[].prefill_tok_per_s",
                ],
            },
            "scientific_target_selection": {
                "receipt": AUTONOMY_SOURCE,
                "fields": [
                    "hcli_autonomy.self_selected_target",
                    "hcli_autonomy.selected_AND_measured",
                ],
            },
        },
    }


def _assert_pareto_role_suitability() -> dict[str, Any]:
    from tools.future.campaign_pareto import harvest as pareto_harvest

    doc = pareto_harvest()
    debt = doc["measurement_debt"]
    n = debt["n_with_axis"].get("role_suitability", 0)
    if n < 1:
        raise AssertionError(
            "campaign_pareto still reports role_suitability unmeasured for everyone: "
            f"{debt}"
        )
    if "role_suitability" in debt["unmeasured_for_everyone"]:
        raise AssertionError(
            "role_suitability remains in unmeasured_for_everyone after harvest"
        )
    points = [
        p for p in doc["scored"]
        if "role_suitability" in (p.get("values") or {})
    ]
    if not points:
        raise AssertionError("harvest produced no comparable role_suitability values")
    for p in points:
        contract = p.get("role_suitability_contract") or {}
        if contract.get("named_value") != "n_measured_requirements":
            raise AssertionError(
                f"harvested a blend or unnamed value: {p['id']} {contract}"
            )
        if contract.get("not_a_blend") is not True:
            raise AssertionError(f"{p['id']} contract missing not_a_blend")
        hits = blend_keys_found(p.get("values") or {})
        if hits:
            raise AssertionError(f"{p['id']} values leaked blend keys: {hits}")
        expected = contract.get("n_measured_requirements")
        got = p["values"]["role_suitability"]
        if expected is not None and abs(float(got) - float(expected)) > 1e-12:
            raise AssertionError(
                f"{p['id']} role_suitability {got} != n_measured_requirements {expected}"
            )
    return {
        "n_with_role_suitability": n,
        "n_points": len(points),
        "unmeasured_for_everyone": debt["unmeasured_for_everyone"],
        "sample": [
            {
                "id": p["id"],
                "role_suitability": p["values"]["role_suitability"],
                "covered": (p.get("role_suitability_contract") or {}).get("covered"),
            }
            for p in points
        ],
    }


def selftest() -> dict[str, Any]:
    neg = negative_control()
    if not neg["passed"]:
        raise AssertionError(f"negative control failed: {neg['failed_checks']}")

    live = build_live_candidate()
    hits = blend_keys_found(live)
    if hits:
        raise AssertionError(f"live candidate leaked blend keys: {hits}")
    if live["role"] != HCLI_RESIDENT:
        raise AssertionError(f"live role {live['role']!r}")
    if not live["covered"]:
        raise AssertionError("live HCLI resident covered set is empty")
    if live["n_measured_requirements"] == live["n_requirements"]:
        # tool_reliability is expected null on disk; a full cover would mean
        # we filled it in. Allowed only if the reliability receipt populated it.
        tool = live["requirements"]["tool_reliability"]
        if tool["value"] is None:
            raise AssertionError(
                "live candidate reports full coverage while tool_reliability is null"
            )
    if live["comparable_value"] in (None, 0.0):
        raise AssertionError(
            f"live comparable_value {live['comparable_value']!r} looks unmeasured"
        )
    if live["comparable_value"] == float(live["n_requirements"]) and live["uncovered"]:
        raise AssertionError("partial coverage rounded to full")

    # Live candidate is itself a partial-coverage case: tool_failure_rate is
    # null with a named cause on the reliability receipt we cite.
    tool = live["requirements"]["tool_reliability"]
    if tool["value"] is None:
        if not tool.get("cause"):
            raise AssertionError("live tool_reliability is null without a cause")
        if "tool_reliability" in live["covered"]:
            raise AssertionError("null tool_reliability listed as covered")
        if not any(u["requirement"] == "tool_reliability" for u in live["uncovered"]):
            raise AssertionError("null tool_reliability not named in uncovered")
    instr = live["requirements"]["instruction_following"]
    if instr["value"] is None:
        raise AssertionError("instruction_following unexpectedly null")
    if not instr.get("source_field"):
        raise AssertionError("instruction_following missing source_field")

    absent = roles_without_contracts()
    for row in absent:
        if row.get("requirements") is not None:
            raise AssertionError(
                f"invented a contract for {row['role']}: {row['requirements']}"
            )
        if not row.get("cause"):
            raise AssertionError(f"{row['role']} missing cause for absent contract")

    controls_preview = {
        "known_result": None,
        "negative_control": {
            "passed": True,
            "failed_checks": [],
            "all_failed": neg["all_failed"],
            "no_evidence": neg["no_evidence"],
            "partial": neg["partial"],
        },
        "mutation_check": None,
        "pareto_harvest": None,
    }
    doc = build_doc(controls=controls_preview, live=live)
    hits = blend_keys_found(doc)
    if hits:
        raise AssertionError(f"receipt draft leaked blend keys: {hits}")
    write_receipt(RECEIPT_NAME, doc, RECORDED_BY)

    mut = mutation_check()
    src = Path(__file__).read_text(encoding="utf-8")
    if "LOAD_BEARING_NULL" not in src:
        raise AssertionError("mutation left LOAD_BEARING_NULL unrestored")
    if "MUTATION_ACTIVE null" in src.split("def measured_or_null")[1].split("def ")[0]:
        raise AssertionError("mutation left measured_or_null unrestored")
    pareto_src = (HERE / "campaign_pareto.py").read_text(encoding="utf-8")
    if "LOAD_BEARING_HARVEST" not in pareto_src:
        raise AssertionError("mutation left LOAD_BEARING_HARVEST unrestored")
    if "MUTATION_ACTIVE harvest" in pareto_src.split("def harvest_role_suitability")[1].split("def ")[0]:
        raise AssertionError("mutation left harvest_role_suitability unrestored")
    if mut["null_vs_zero"]["before_sha256"] == mut["null_vs_zero"]["after_sha256"]:
        raise AssertionError("null mutation did not change sha256")
    if mut["harvest"]["before_sha256"] == mut["harvest"]["after_sha256"]:
        raise AssertionError("harvest mutation did not change sha256")

    pareto = _assert_pareto_role_suitability()
    return {
        "ok": True,
        "negative_control": neg,
        "mutation_check": mut,
        "pareto_harvest": pareto,
        "live": live,
        "roles_without_contracts": absent,
    }


def print_table(doc: Mapping[str, Any]) -> None:
    print()
    print("HCLI RESIDENT role contract (requirements vector, not a score)")
    cands = doc.get("candidates") or []
    if not cands:
        print("  (no candidates)")
        return
    live = cands[0]
    print(f"  candidate: {live.get('name')}  status={live.get('status')}")
    print(f"  covered: {live.get('covered')}")
    print(f"  n_measured_requirements: {live.get('n_measured_requirements')}"
          f"/{live.get('n_requirements')}  comparable_value={live.get('comparable_value')}")
    print()
    hdr = f"{'requirement':<32} {'value':<48} {'source'}"
    print(hdr)
    print("-" * len(hdr))
    for name, cell in (live.get("requirements") or {}).items():
        value = cell.get("value")
        if value is None:
            shown = f"null ({(cell.get('cause') or '')[:40]})"
        elif isinstance(value, dict):
            shown = ", ".join(f"{k}={v}" for k, v in value.items() if not isinstance(v, list))
        else:
            shown = repr(value)
        src = f"{cell.get('source_receipt')} :: {cell.get('source_field')}"
        print(f"{name:<32} {shown:<48} {src}")
    print()
    print("roles without contracts:")
    for row in doc.get("roles_without_contracts") or []:
        print(f"  {row['role']}: requirements=null  cause={row.get('cause')}")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    selftest_only = "--selftest" in args
    build = "--build" in args or not args or selftest_only

    gate = selftest()
    print("selftest: PASS")
    neg = gate["negative_control"]
    print(
        f"  negative control: all-failed n_measured="
        f"{neg['all_failed']['n_measured_requirements']} "
        f"comparable={neg['all_failed']['comparable_value']}; "
        f"no-evidence n_measured={neg['no_evidence']['n_measured_requirements']} "
        f"comparable={neg['no_evidence']['comparable_value']!r}; "
        f"partial covered={neg['partial']['covered']}"
    )
    mut = gate["mutation_check"]
    print(
        f"  mutation null {mut['null_vs_zero']['before_sha256'][:12]}→"
        f"{mut['null_vs_zero']['after_sha256'][:12]} "
        f"anchor={mut['null_vs_zero']['anchor']}"
    )
    print(
        f"  mutation harvest {mut['harvest']['before_sha256'][:12]}→"
        f"{mut['harvest']['after_sha256'][:12]} "
        f"anchor={mut['harvest']['anchor']}"
    )
    print(f"  mutation restored={mut['restored']}")

    if not build:
        return 0

    controls = {
        "negative_control": {
            "passed": True,
            "failed_checks": [],
            "all_failed": {
                "n_measured_requirements": neg["all_failed"]["n_measured_requirements"],
                "comparable_value": neg["all_failed"]["comparable_value"],
                "covered": neg["all_failed"]["covered"],
                "repair_behaviour": neg["all_failed"]["requirements"]["repair_behaviour"]["value"],
                "self_selected_target": (
                    neg["all_failed"]["requirements"]["scientific_target_selection"]["value"]
                    ["self_selected_target"]
                ),
            },
            "no_evidence": {
                "n_measured_requirements": neg["no_evidence"]["n_measured_requirements"],
                "comparable_value": neg["no_evidence"]["comparable_value"],
                "covered": neg["no_evidence"]["covered"],
                "all_values_null": True,
                "causes_named": True,
            },
            "partial": {
                "n_measured_requirements": neg["partial"]["n_measured_requirements"],
                "comparable_value": neg["partial"]["comparable_value"],
                "covered": neg["partial"]["covered"],
                "uncovered": [u["requirement"] for u in neg["partial"]["uncovered"]],
            },
        },
        "mutation_check": mut,
        "pareto_harvest": gate.get("pareto_harvest"),
    }
    doc = build_doc(controls=controls, live=gate["live"])
    path = write_receipt(RECEIPT_NAME, doc, RECORDED_BY)
    print(f"wrote {path}")
    print_table(doc)
    pareto = _assert_pareto_role_suitability()
    print(
        f"campaign_pareto harvest: role_suitability on "
        f"{pareto['n_with_role_suitability']} points; still unmeasured: "
        f"{pareto['unmeasured_for_everyone']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
