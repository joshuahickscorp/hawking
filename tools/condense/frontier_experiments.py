#!/usr/bin/env python3.12
"""frontier_experiments.py - expensive-mode depth checks for frontier claims.

The Studio run is supposed to be maximal, not merely successful once. This module validates the
experiment matrix that proves depth: repeated seeds, ablations, repeated cold/warm cliff runs, and
publishable null results. It does not run heavy work; it makes missing experiment depth visible and
claim-blocking.
"""
from __future__ import annotations

import pathlib
import re
from typing import Any

from frontier_common import read_json as _read_json  # noqa: E402

COND_DIR = pathlib.Path("reports/condense")

EXPERIMENT_REQUIREMENTS = (
    {
        "name": "floor_seeds",
        "description": "At least 3 independent floor-search seeds pass or are fully recorded.",
        "min_count": 3,
        "aliases": ("floor_seed", "seed", "floor-search-seed"),
    },
    {
        "name": "calibration_ablations",
        "description": "Required calibration/recipe ablations are measured.",
        "required_names": ("domain_matched_calib", "mixed_domain_calib", "awq_alpha_sweep", "residual_depth_sweep"),
        "aliases": ("calib", "calibration", "ablation"),
    },
    {
        "name": "bpw_ladder",
        "description": "At least 4 effective-bpw rungs are measured around the chosen frontier recipe.",
        "min_count": 4,
        "aliases": ("bpw", "ladder", "rung"),
    },
    {
        "name": "moe_expert_ablation",
        "description": "MoE per-expert allocation is tested, or dense/N/A is explicitly justified.",
        "aliases": ("moe", "expert", "expert_ablation"),
        "allow_na": True,
    },
    {
        "name": "ramcliff_repeats",
        "description": "At least 3 cold and 3 warm RAM-cliff runs are recorded.",
        "min_cold": 3,
        "min_warm": 3,
        "aliases": ("ramcliff", "cliff_repeat", "cold_warm"),
    },
    {
        "name": "baseline_variants",
        "description": "At least 4 baseline variants or explicit N/A rows are recorded.",
        "min_count": 4,
        "aliases": ("baseline_variant", "baseline", "competitor"),
    },
    {
        "name": "null_certification",
        "description": "At least 2 negative/null results are archived with reasons.",
        "min_count": 2,
        "aliases": ("null", "negative", "kill", "failure"),
    },
    {
        "name": "rebake_or_hash_verify",
        "description": "Artifact rebake, independent verify, or hash-stability proof exists.",
        "aliases": ("rebake", "hash_verify", "verify", "artifact_verify"),
    },
)

PASS_STATUSES = {"pass", "passed", "ok", "done", "complete", "measured", "certified", "verified"}
NA_STATUSES = {"na", "n/a", "n-a", "not-applicable", "not_applicable", "waived", "skip", "skipped"}
SYNTHETIC_MODES = {"synthetic", "mock", "modelled", "modeled", "fake"}


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _status(value: Any) -> str:
    return _norm(value).replace(" ", "-")


def _match(value: Any, req: dict[str, Any]) -> bool:
    hay = _norm(value)
    aliases = (req["name"], *req.get("aliases", ()))
    return any((needle := _norm(alias)) and (needle in hay or hay in needle) for alias in aliases)


def _entry_name(entry: dict[str, Any]) -> str:
    bits = [
        entry.get("name"),
        entry.get("category"),
        entry.get("axis"),
        entry.get("kind"),
        entry.get("domain"),
        entry.get("label"),
    ]
    return " ".join(str(b) for b in bits if b)


def matrix_path(root: pathlib.Path, label: str) -> pathlib.Path:
    return root / COND_DIR / f"{label}_experiment_matrix.json"


def _entries(record: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in ("experiments", "requirements", "matrix", "rows", "ablations", "nulls"):
        val = record.get(key)
        if isinstance(val, dict):
            for name, row in val.items():
                if isinstance(row, list):
                    for item in row:
                        if isinstance(item, dict):
                            copied = dict(item)
                            copied.setdefault("category", name)
                            out.append(copied)
                elif isinstance(row, dict):
                    copied = dict(row)
                    copied.setdefault("category", name)
                    out.append(copied)
        elif isinstance(val, list):
            out.extend(dict(x) for x in val if isinstance(x, dict))
    return out


def _reason(entry: dict[str, Any], record: dict[str, Any]) -> str:
    for key in ("reason", "na_reason", "note", "why", "outcome"):
        if entry.get(key):
            return str(entry[key])
    return str(record.get("note") or "")


def _usable(entry: dict[str, Any], record: dict[str, Any], allow_na: bool = False) -> tuple[bool, str]:
    mode = _status(entry.get("mode") or entry.get("source") or record.get("mode") or record.get("source"))
    if mode in SYNTHETIC_MODES:
        return False, "synthetic/modelled experiment row cannot cover expensive-mode depth"
    st = _status(entry.get("status") or entry.get("verdict") or entry.get("coverage_status"))
    if st in PASS_STATUSES or entry.get("measured") is True:
        return True, ""
    if allow_na and st in NA_STATUSES and _reason(entry, record):
        return True, ""
    if st in NA_STATUSES:
        return False, "N/A row lacks reason or N/A is not allowed for this requirement"
    return False, f"status {st or 'missing'} is not pass/measured/certified"


def _rows_for(entries: list[dict[str, Any]], req: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    aliases = (req["name"], *req.get("aliases", ()))
    norm_aliases = {_norm(alias) for alias in aliases}
    for entry in entries:
        category = entry.get("category")
        if category:
            cat = _norm(category)
            if any(alias and (alias == cat or alias in cat or cat in alias) for alias in norm_aliases):
                rows.append(entry)
            continue
        if _match(_entry_name(entry), req):
            rows.append(entry)
    return rows


def _distinct_count(rows: list[dict[str, Any]], key: str) -> int:
    vals = []
    for row in rows:
        val = row.get(key) if row.get(key) is not None else row.get("name") or _entry_name(row)
        vals.append(str(val))
    return len(set(vals))


def _require_count(record: dict[str, Any], entries: list[dict[str, Any]], req: dict[str, Any]) -> dict[str, Any]:
    rows = _rows_for(entries, req)
    usable = []
    problems = []
    for row in rows:
        ok, problem = _usable(row, record, allow_na=req.get("allow_na", False))
        if ok:
            usable.append(row)
        else:
            problems.append(f"{_entry_name(row) or req['name']}: {problem}")
    count = _distinct_count(usable, "seed") if req["name"] == "floor_seeds" else len(usable)
    needed = req.get("min_count", 1)
    ok = count >= needed
    if not ok:
        problems.append(f"{req['name']} has {count}/{needed} usable row(s)")
    return {
        "requirement": req["name"],
        "ok": ok,
        "covered": count,
        "required": needed,
        "problems": problems,
    }


def _require_named(record: dict[str, Any], entries: list[dict[str, Any]], req: dict[str, Any]) -> dict[str, Any]:
    rows = _rows_for(entries, req)
    missing = []
    problems = []
    for name in req["required_names"]:
        matched = [r for r in rows if _norm(name) in _norm(_entry_name(r)) or _norm(name) in _norm(r.get("name"))]
        good = False
        for row in matched:
            ok, problem = _usable(row, record)
            good = good or ok
            if not ok:
                problems.append(f"{name}: {problem}")
        if not good:
            missing.append(name)
    if missing:
        problems.append("missing named ablation(s): " + ", ".join(missing))
    return {
        "requirement": req["name"],
        "ok": not missing,
        "covered": len(req["required_names"]) - len(missing),
        "required": len(req["required_names"]),
        "problems": problems,
    }


def _require_ramcliff_repeats(record: dict[str, Any], entries: list[dict[str, Any]], req: dict[str, Any]) -> dict[str, Any]:
    rows = _rows_for(entries, req)
    cold = 0
    warm = 0
    problems = []
    for row in rows:
        ok, problem = _usable(row, record)
        if not ok:
            problems.append(f"{_entry_name(row) or req['name']}: {problem}")
            continue
        temp = _status(row.get("temperature") or row.get("run_type") or row.get("phase"))
        if "cold" in temp:
            cold += 1
        elif "warm" in temp:
            warm += 1
    ok = cold >= req["min_cold"] and warm >= req["min_warm"]
    if not ok:
        problems.append(f"RAM-cliff repeats cold={cold}/{req['min_cold']} warm={warm}/{req['min_warm']}")
    return {
        "requirement": req["name"],
        "ok": ok,
        "covered": {"cold": cold, "warm": warm},
        "required": {"cold": req["min_cold"], "warm": req["min_warm"]},
        "problems": problems,
    }


def _require_single(record: dict[str, Any], entries: list[dict[str, Any]], req: dict[str, Any]) -> dict[str, Any]:
    rows = _rows_for(entries, req)
    problems = []
    for row in rows:
        ok, problem = _usable(row, record, allow_na=req.get("allow_na", False))
        if ok:
            return {"requirement": req["name"], "ok": True, "covered": 1, "required": 1, "problems": []}
        problems.append(f"{_entry_name(row) or req['name']}: {problem}")
    problems.append(f"{req['name']} has no usable row")
    return {"requirement": req["name"], "ok": False, "covered": 0, "required": 1, "problems": problems}


def experiment_status(root: pathlib.Path, label: str) -> dict[str, Any]:
    path = matrix_path(root, label)
    record = _read_json(path)
    if not record:
        rows = [
            {
                "requirement": req["name"],
                "ok": False,
                "covered": 0,
                "required": req.get("min_count") or len(req.get("required_names", ())) or 1,
                "problems": [f"{path} is missing or unreadable"],
            }
            for req in EXPERIMENT_REQUIREMENTS
        ]
        return {
            "label": label,
            "path": str(path),
            "exists": False,
            "ok": False,
            "rows": rows,
            "problems": [p for row in rows for p in row["problems"]],
        }
    entries = _entries(record)
    rows = []
    for req in EXPERIMENT_REQUIREMENTS:
        if "required_names" in req:
            rows.append(_require_named(record, entries, req))
        elif req["name"] == "ramcliff_repeats":
            rows.append(_require_ramcliff_repeats(record, entries, req))
        elif "min_count" in req:
            rows.append(_require_count(record, entries, req))
        else:
            rows.append(_require_single(record, entries, req))
    problems = [f"{row['requirement']}: {p}" for row in rows for p in row["problems"]]
    return {
        "label": label,
        "path": str(path),
        "exists": True,
        "schema": record.get("schema"),
        "ok": not problems,
        "passed_count": sum(1 for row in rows if row["ok"]),
        "required_count": len(rows),
        "rows": rows,
        "problems": problems,
    }


def experiment_rollup(root: pathlib.Path, labels: list[str]) -> dict[str, Any]:
    rows = [experiment_status(root, label) for label in labels]
    blocked = [row["label"] for row in rows if not row["ok"]]
    return {
        "schema": "hawking.frontier_experiment_rollup.v1",
        "model_count": len(rows),
        "passed_count": len(rows) - len(blocked),
        "blocked_count": len(blocked),
        "blocked_labels": blocked,
        "requirements": [r["name"] for r in EXPERIMENT_REQUIREMENTS],
        "rows": rows,
        "ok": not blocked,
    }


def _skeleton(label: str) -> dict[str, Any]:
    return {
        "schema": "hawking.frontier_experiment_matrix.v1",
        "model": label,
        "mode": "real",
        "machine_class": "Studio-M3Ultra-96",
        "machine_name": "<exact Studio host label>",
        "same_box": True,
        "same_box_group": "<same machine/session id shared by experiment matrix>",
        "machine_fingerprint_sha256": "<64 hex>",
        "environment_receipt": "<hawking studio environment-capture receipt>",
        "artifact_inventory_receipt": "<artifact inventory receipt>",
        "artifact_inventory_sha256": "<64 hex>",
        "source_provenance_receipt": "<source provenance receipt>",
        "experiment_plan_sha256": "<64 hex>",
        "run_id": "<same-run id>",
        "commands": ["<exact experiment orchestration command>"],
        "experiments": {
            "floor_seeds": [
                {
                    "category": "floor_seed",
                    "seed": i,
                    "status": "TODO pass",
                    "same_box": True,
                    "command": "<exact row command>",
                    "receipt": "<path>",
                    "trace_sha256": "<64 hex>",
                }
                for i in (1, 2, 3)
            ],
            "calibration_ablations": [
                {
                    "category": "calibration_ablations",
                    "name": name,
                    "status": "TODO pass",
                    "same_box": True,
                    "command": "<exact row command>",
                    "receipt": "<path>",
                    "trace_sha256": "<64 hex>",
                }
                for name in ("domain_matched_calib", "mixed_domain_calib", "awq_alpha_sweep", "residual_depth_sweep")
            ],
            "bpw_ladder": [
                {
                    "category": "bpw_ladder",
                    "bpw": bpw,
                    "status": "TODO pass",
                    "same_box": True,
                    "command": "<exact row command>",
                    "receipt": "<path>",
                    "trace_sha256": "<64 hex>",
                }
                for bpw in ("over", "target", "under", "failure_floor")
            ],
            "moe_expert_ablation": [
                {
                    "category": "moe_expert_ablation",
                    "status": "TODO pass|na",
                    "same_box": True,
                    "command": "<exact row command>",
                    "receipt": "<path>",
                    "trace_sha256": "<64 hex>",
                    "reason": "<required if N/A>",
                }
            ],
            "ramcliff_repeats": [
                {
                    "category": "ramcliff_repeats",
                    "run_type": run_type,
                    "status": "TODO pass",
                    "same_box": True,
                    "command": "<exact row command>",
                    "receipt": "<path>",
                    "trace_sha256": "<64 hex>",
                }
                for run_type in ("cold", "cold", "cold", "warm", "warm", "warm")
            ],
            "baseline_variants": [
                {
                    "category": "baseline_variants",
                    "name": name,
                    "status": "TODO pass|na",
                    "same_box": True,
                    "command": "<exact row command>",
                    "receipt": "<path>",
                    "trace_sha256": "<64 hex>",
                    "reason": "<required if N/A>",
                }
                for name in ("llama_q4", "llama_iq2", "mlx_4bit", "unsloth_or_exl3")
            ],
            "null_certification": [
                {
                    "category": "null_certification",
                    "name": "failed_recipe",
                    "status": "TODO certified",
                    "same_box": True,
                    "command": "<exact row command>",
                    "receipt": "<path>",
                    "trace_sha256": "<64 hex>",
                    "reason": "<why it failed>",
                },
                {
                    "category": "null_certification",
                    "name": "baseline_or_quality_loss",
                    "status": "TODO certified",
                    "same_box": True,
                    "command": "<exact row command>",
                    "receipt": "<path>",
                    "trace_sha256": "<64 hex>",
                    "reason": "<why it matters>",
                },
            ],
            "rebake_or_hash_verify": [
                {
                    "category": "rebake_or_hash_verify",
                    "status": "TODO verified",
                    "same_box": True,
                    "command": "<exact row command>",
                    "receipt": "<path>",
                    "trace_sha256": "<64 hex>",
                }
            ],
        },
    }


def experiment_plan(root: pathlib.Path, labels: list[str]) -> dict[str, Any]:
    return {
        "schema": "hawking.frontier_experiment_plan.v1",
        "requirements": [
            {
                "name": req["name"],
                "description": req["description"],
                "min_count": req.get("min_count"),
                "required_names": req.get("required_names"),
            }
            for req in EXPERIMENT_REQUIREMENTS
        ],
        "labels": [
            {
                "label": label,
                "matrix_path": str(matrix_path(root, label)),
                "command_template": (
                    f"hawking studio experiment-receipt draft {label} --sign-draft --force"
                ),
                "skeleton": _skeleton(label),
            }
            for label in labels
        ],
    }
