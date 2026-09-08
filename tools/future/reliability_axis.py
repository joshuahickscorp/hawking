#!/usr/bin/env python3.12
"""G016 reliability axis — rates derived from HCLI receipts, not a grade.

The campaign Pareto frontier carries a `reliability` axis with no coverage.
This module fills it from evidence already on disk: `.hcli/receipts/` (real
HCLI runs) and `tools/drive_rounds.jsonl` (per-round weakness tags).

It does not write `reliability = good`. It does not emit a 1-10 score. It
does not average the rates it reports. Acceptance is
`hcli.engine.is_accepted_work`, imported, never reimplemented: that function's
docstring records why `validation["ok"]` alone scored five fabricated
completions as successes.

    python3.12 tools/future/reliability_axis.py --selftest
    python3.12 tools/future/reliability_axis.py --build
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hcli.engine import is_accepted_work  # noqa: E402  the authority, not a copy
from tools.future._common import write_receipt  # noqa: E402

RECEIPT_NAME = "G016_RELIABILITY_AXIS.json"
SCHEMA = "hawking.future.reliability_axis.v1"
RECORDED_BY = "tools/future/reliability_axis.py"
VERSION = 1

# Same window autonomy_ratio.py uses: receipts *written* since this local stamp.
# File mtime, not timestamps.started_at — "written since" is a filesystem fact.
WINDOW_START = "2026-09-07 02:00:00"
STEER_CITED_NAIVE_OK = 2

# Tags on the load-bearing returns. The mutation check locates them via
# inspect.getsourcelines so a quoted copy of the line cannot steal the rewrite.

# Keys a run-receipt actually carries. Named so a reader can recompute without
# this file. Absence is reported, never filled in.
IDENTITY_KEYS = ("model", "endpoint")
TIME_KEYS = ("timestamps.started_at", "timestamps.finished_at")
ACCEPTANCE_KEYS = ("validation", "result_envelope")
STRUCTURED_KEYS = (
    "structured_output.exhausted",
    "structured_output.errors",
    "structured_output.last_violation",
    "error_type",
)
TOOL_OBS_KEYS = ("observations", "tool_observations")
FAILURE_KEYS = (
    "structured_output.retries",
    "structured_output.exhausted",
    "structured_output.errors",
    "structured_output.last_violation",
    "error",
    "error_type",
    "rolled_back",
    "validation.ok",
    "status",
    "observations",
    "tool_observations",
)
ROLLED_BACK_KEYS = ("rolled_back",)
VARIANCE_KEYS = ("goal", "validation")
DRIVE_ROUND_KEYS = ("round", "elapsed_s", "weaknesses", "stalled", "chars", "tail")


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
    # <repo>/.git  or  <repo>/.git/worktrees/<name>
    if common.name == ".git":
        return common.parent
    if common.parent.name == ".git":
        return common.parent.parent
    return common.parent


def receipts_dir() -> Path:
    """HCLI receipts live next to the main checkout, not in every worktree."""
    env = os.environ.get("HCLI_RECEIPTS") or os.environ.get("HAWKING_HCLI_RECEIPTS")
    if env:
        return Path(env)
    candidates = [
        ROOT / ".hcli" / "receipts",
    ]
    main = _git_common_parent()
    if main is not None:
        candidates.append(main / ".hcli" / "receipts")
    for cand in candidates:
        if cand.is_dir() and any(cand.glob("*.json")):
            return cand
    return candidates[0]


def drive_rounds_path() -> Path | None:
    env = os.environ.get("HCLI_DRIVE_ROUNDS")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    candidates = [ROOT / "tools" / "drive_rounds.jsonl"]
    main = _git_common_parent()
    if main is not None:
        candidates.append(main / "tools" / "drive_rounds.jsonl")
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def is_engine_run(receipt: Mapping[str, Any]) -> bool:
    """True for an HCLI engine run receipt. The research-gate file is not one."""
    if not isinstance(receipt, Mapping):
        return False
    schema = receipt.get("schema")
    if isinstance(schema, str) and schema.startswith("hcli.agentos.research_gate"):
        return False
    return "validation" in receipt and "goal" in receipt


def load_receipts(directory: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (engine_runs, skipped). Unreadable files are skipped, not invented."""
    d = directory if directory is not None else receipts_dir()
    runs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    if not d.is_dir():
        return runs, [{"path": str(d), "reason": "receipts directory does not exist"}]
    for path in sorted(d.glob("*.json")):
        try:
            obj = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            skipped.append({"path": path.name, "reason": f"unreadable: {type(exc).__name__}"})
            continue
        if not isinstance(obj, dict):
            skipped.append({"path": path.name, "reason": f"not an object: {type(obj).__name__}"})
            continue
        obj["_receipt_file"] = path.name
        obj["_receipt_mtime"] = path.stat().st_mtime
        if is_engine_run(obj):
            runs.append(obj)
        else:
            skipped.append({
                "path": path.name,
                "reason": "not an HCLI engine run receipt",
                "schema": obj.get("schema"),
            })
    return runs, skipped


def load_drive_rounds(path: Path | None = None) -> dict[str, Any]:
    p = path if path is not None else drive_rounds_path()
    if p is None or not p.is_file():
        return {
            "path": None,
            "n": 0,
            "keys": list(DRIVE_ROUND_KEYS),
            "weakness_counts": None,
            "missing": "tools/drive_rounds.jsonl",
            "not_joined_to_models": (
                "file is absent, and even when present the records have no "
                "model/endpoint field"
            ),
        }
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        rows.append(row)
        tags = row.get("weaknesses")
        if isinstance(tags, list):
            for tag in tags:
                if isinstance(tag, str):
                    counts[tag] = counts.get(tag, 0) + 1
    return {
        "path": str(p),
        "n": len(rows),
        "keys": list(DRIVE_ROUND_KEYS),
        "weakness_counts": dict(sorted(counts.items())) if counts else {},
        "has_model_field": any("model" in r or "endpoint" in r for r in rows),
        "not_joined_to_models": (
            "drive_rounds.jsonl has no model/endpoint field; weakness tags "
            "cannot be attributed to a model identity and are not a rate"
        ),
    }


# ---------------------------------------------------------------------------
# predicates — the load-bearing lines live here
# ---------------------------------------------------------------------------

def receipt_is_accepted(receipt: Mapping[str, Any]) -> bool:
    """Accepted work is the imported predicate, never validation['ok']."""
    return is_accepted_work(receipt.get("validation"), receipt.get("result_envelope"))  # LOAD_BEARING_ACCEPTANCE


def receipt_naive_ok(receipt: Mapping[str, Any]) -> bool:
    """The predicate that scored fabrications as successes. Control only."""
    validation = receipt.get("validation")
    return isinstance(validation, dict) and validation.get("ok") is True


def _observations(receipt: Mapping[str, Any]) -> list[Any] | None:
    for key in TOOL_OBS_KEYS:
        value = receipt.get(key)
        if isinstance(value, list):
            return value
    return None


def receipt_structured_action_failed(receipt: Mapping[str, Any]) -> bool | None:
    """Terminal schema rejection. None if the run cannot be scored.

    A retry that later produced valid structured output (`retries` > 0 and
    `exhausted` is false) is a recovered malformation, not a terminal failure.
    """
    if receipt.get("error_type") == "StructuredOutputExhausted":
        return True
    so = receipt.get("structured_output")
    if not isinstance(so, dict):
        return None
    if so.get("exhausted") is True:
        return True
    return False


def receipt_hit_failure(receipt: Mapping[str, Any]) -> bool:
    """An intra-run failure visible on the receipt, not 'was not accepted'.

    Fabricated answers (`validation.ok` True, `kind` read_only) did not hit a
    failure; they closed without accepted work. Recovery is failing then still
    closing with accepted work.
    """
    so = receipt.get("structured_output")
    if isinstance(so, dict):
        if so.get("exhausted") is True:
            return True
        retries = so.get("retries")
        if isinstance(retries, int) and retries > 0:
            return True
        errors = so.get("errors")
        if isinstance(errors, list) and errors:
            return True
        if so.get("last_violation"):
            return True
    if receipt.get("error_type") or receipt.get("error"):
        return True
    if receipt.get("rolled_back") is True:
        return True
    validation = receipt.get("validation")
    if isinstance(validation, dict) and validation.get("ok") is False:
        return True
    status = str(receipt.get("status") or "").lower()
    if status in {"failed", "cancelled"}:
        return True
    obs = _observations(receipt)
    if obs is not None:
        for item in obs:
            if isinstance(item, dict) and item.get("ok") is False:
                return True
    return False


def receipt_recovered(receipt: Mapping[str, Any]) -> bool:
    """Failure AND accepted work. Failing is cheap; recovering is the capability."""
    return receipt_hit_failure(receipt) and receipt_is_accepted(receipt)  # LOAD_BEARING_RECOVERY


def receipt_identity(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Group by the receipt's own model/endpoint fields, not inferred substitutes.

    `endpoint` is not a top-level key on these receipts. `model_calls[].endpoint`
    exists on many of them; it is recorded as an observation, not used as the
    grouping key the contract named.
    """
    model = receipt.get("model")
    endpoint = receipt.get("endpoint")  # absent on the on-disk set
    observed = None
    calls = receipt.get("model_calls")
    if isinstance(calls, list):
        eps = [
            c.get("endpoint")
            for c in calls
            if isinstance(c, dict) and c.get("endpoint")
        ]
        if eps:
            observed = eps[0]
    return {
        "model": model,
        "endpoint": endpoint,
        "model_calls_endpoint": observed,
        "identity_keys": list(IDENTITY_KEYS),
        "endpoint_on_receipt": "endpoint" in receipt,
    }


def _identity_key(receipt: Mapping[str, Any]) -> tuple[Any, Any]:
    ident = receipt_identity(receipt)
    return (ident["model"], ident["endpoint"])


# ---------------------------------------------------------------------------
# rates
# ---------------------------------------------------------------------------

def _rate(n: int, d: int, keys: Sequence[str], predicate: str) -> dict[str, Any]:
    return {
        "value": (n / d) if d else None,
        "n": n,
        "d": d,
        "keys": list(keys),
        "predicate": predicate,
    }


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _time_window(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stamps: list[str] = []
    missing = 0
    for receipt in receipts:
        ts = receipt.get("timestamps")
        found = False
        if isinstance(ts, dict):
            for key in ("started_at", "finished_at"):
                value = ts.get(key)
                if isinstance(value, str) and value:
                    stamps.append(value)
                    found = True
        if not found:
            missing += 1
    if not stamps:
        return {
            "first": None,
            "last": None,
            "keys": list(TIME_KEYS),
            "n_missing_timestamps": missing,
            "missing": "timestamps.started_at / timestamps.finished_at",
        }
    return {
        "first": min(stamps),
        "last": max(stamps),
        "keys": list(TIME_KEYS),
        "n_missing_timestamps": missing,
    }


def _tool_failure(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failed = 0
    total = 0
    n_with_field = 0
    for receipt in receipts:
        obs = _observations(receipt)
        if obs is None:
            continue
        n_with_field += 1
        for item in obs:
            total += 1
            if isinstance(item, dict) and item.get("ok") is False:
                failed += 1
            elif not isinstance(item, dict):
                # A non-object observation cannot be scored as ok; count it as
                # an observation that is not a documented success.
                failed += 1
    if n_with_field == 0:
        return {
            "value": None,
            "n": None,
            "d": None,
            "keys": list(TOOL_OBS_KEYS),
            "predicate": "item.ok is False over observations",
            "missing": (
                "receipts do not persist `observations` or `tool_observations`; "
                "the engine records those only in-memory during the tool loop"
            ),
        }
    return _rate(
        failed, total, TOOL_OBS_KEYS,
        "item.ok is False over observations / tool_observations",
    )


def _variance(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Spread of accepted_work (0/1) where the same goal text ran more than once.

    min / median / max. Never a mean.
    """
    by_goal: dict[str, list[int]] = defaultdict(list)
    for receipt in receipts:
        goal = receipt.get("goal")
        if not isinstance(goal, str) or not goal:
            continue
        by_goal[goal].append(1 if receipt_is_accepted(receipt) else 0)
    repeated = {g: outs for g, outs in by_goal.items() if len(outs) > 1}
    if not repeated:
        return {
            "min": None,
            "median": None,
            "max": None,
            "keys": list(VARIANCE_KEYS),
            "outcome": "accepted_work as 0/1",
            "n_unique_goals": len(by_goal),
            "n_goals_run_more_than_once": 0,
            "missing": "no goal text was run more than once for this identity",
        }
    flat = [v for outs in repeated.values() for v in outs]
    mixed = []
    for goal, outs in sorted(repeated.items(), key=lambda kv: -len(kv[1])):
        if min(outs) != max(outs):
            mixed.append({
                "goal_sha256_16": hashlib.sha256(goal.encode()).hexdigest()[:16],
                "goal_head": goal[:120].replace("\n", " / "),
                "n": len(outs),
                "min": min(outs),
                "median": _median(outs),
                "max": max(outs),
                "n_accepted": sum(outs),
            })
    return {
        "min": min(flat),
        "median": _median(flat),
        "max": max(flat),
        "keys": list(VARIANCE_KEYS),
        "outcome": "accepted_work as 0/1",
        "n_unique_goals": len(by_goal),
        "n_goals_run_more_than_once": len(repeated),
        "n_runs_in_repeated_goals": len(flat),
        "n_mixed_goals": len(mixed),
        "mixed_goals": mixed,
        "note": "spread of 0/1 accepted_work over runs of goals executed more than once; a mean is not reported",
    }


def summarize(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One identity's rates. Missing quantities stay null with the field named."""
    n = len(receipts)
    n_accepted = sum(1 for r in receipts if receipt_is_accepted(r))
    n_naive_ok = sum(1 for r in receipts if receipt_naive_ok(r))
    n_rolled = sum(1 for r in receipts if r.get("rolled_back") is True)
    n_rolled_present = sum(1 for r in receipts if "rolled_back" in r)

    scored_so = 0
    n_so_fail = 0
    n_so_absent = 0
    for r in receipts:
        failed = receipt_structured_action_failed(r)
        if failed is None:
            n_so_absent += 1
            continue
        scored_so += 1
        if failed:
            n_so_fail += 1

    n_hit = sum(1 for r in receipts if receipt_hit_failure(r))
    recovered_files = [
        r.get("_receipt_file") for r in receipts if receipt_recovered(r)
    ]
    n_recovered = len(recovered_files)

    ident = receipt_identity(receipts[0]) if receipts else {
        "model": None, "endpoint": None, "model_calls_endpoint": None,
        "identity_keys": list(IDENTITY_KEYS), "endpoint_on_receipt": False,
    }
    so_rate = _rate(
        n_so_fail, scored_so, STRUCTURED_KEYS,
        "error_type == StructuredOutputExhausted OR structured_output.exhausted is True",
    )
    if scored_so == 0:
        so_rate["missing"] = (
            "structured_output (and error_type StructuredOutputExhausted) absent; "
            "the rate is not estimated from other fields"
        )

    rolled_rate = _rate(
        n_rolled, n_rolled_present, ROLLED_BACK_KEYS,
        "rolled_back is True",
    )
    if n_rolled_present == 0:
        rolled_rate["missing"] = "rolled_back"

    return {
        "model": ident["model"],
        "endpoint": ident["endpoint"],
        "model_calls_endpoint": ident["model_calls_endpoint"],
        "identity_keys": list(IDENTITY_KEYS),
        "endpoint_missing": (
            None if ident["endpoint"] is not None or ident["endpoint_on_receipt"]
            else "receipt.endpoint is not a field on these receipts; grouping is by receipt.model"
        ),
        "n_runs": n,
        "window": _time_window(receipts),
        "accepted_rate": _rate(
            n_accepted, n, ACCEPTANCE_KEYS,
            "hcli.engine.is_accepted_work(validation, result_envelope)",
        ),
        "naive_validation_ok_rate": _rate(
            n_naive_ok, n, ("validation.ok",),
            "validation['ok'] is True — control only, not the axis",
        ),
        "structured_action_failure_rate": so_rate,
        "tool_failure_rate": _tool_failure(receipts),
        "recovery_rate": _rate(
            n_recovered, n_hit, FAILURE_KEYS,
            "receipt_hit_failure AND is_accepted_work; denominator is runs that hit a failure",
        ) if n_hit else {
            "value": None, "n": 0, "d": 0, "keys": list(FAILURE_KEYS),
            "predicate": "receipt_hit_failure AND is_accepted_work",
            "missing": "no run hit a failure visible on the receipt; 0/0 is not 0.0",
        },
        "rolled_back_rate": rolled_rate,
        "variance": _variance(receipts),
        "counts": {
            "accepted": n_accepted,
            "naive_validation_ok": n_naive_ok,
            "structured_action_failed": n_so_fail,
            "structured_action_scored": scored_so,
            "structured_output_absent": n_so_absent,
            "hit_failure": n_hit,
            "recovered": n_recovered,
            "recovered_receipts": [f for f in recovered_files if f],
            "rolled_back": n_rolled,
        },
    }


def aggregate(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for receipt in receipts:
        groups[_identity_key(receipt)].append(receipt)
    models = [
        summarize(group)
        for _, group in sorted(groups.items(), key=lambda kv: -len(kv[1]))
    ]
    overall = summarize(list(receipts)) if receipts else summarize([])
    overall["model"] = "*all*"
    overall["endpoint"] = None
    return {
        "n_runs": len(receipts),
        "n_identities": len(models),
        "models": models,
        "overall": overall,
    }


# ---------------------------------------------------------------------------
# known-result window
# ---------------------------------------------------------------------------

def window_runs(receipts: Sequence[Mapping[str, Any]], since: str = WINDOW_START) -> list[Mapping[str, Any]]:
    cutoff = time.mktime(time.strptime(since, "%Y-%m-%d %H:%M:%S"))
    out = []
    for receipt in receipts:
        mtime = receipt.get("_receipt_mtime")
        if not isinstance(mtime, (int, float)):
            continue
        if mtime >= cutoff:
            out.append(receipt)
    return out


def known_result_control(
    receipts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reproduce: is_accepted_work scores 0 in the window; naive ok does not.

    The steer cited naive_ok = 2. The live directory may have grown; the
    control holds if accepted is 0 and naive_ok is at least that cited
    number. Agreeing with naive ok is the bug this axis exists to avoid.
    """
    if receipts is None:
        receipts, _ = load_receipts()
    window = window_runs(receipts)
    accepted = sum(1 for r in window if receipt_is_accepted(r))
    naive = sum(1 for r in window if receipt_naive_ok(r))
    files = [r.get("_receipt_file") for r in window]
    return {
        "window_start": WINDOW_START,
        "window_is": (
            "receipt file mtime, local time, same cutoff as "
            "tools/future/autonomy_ratio.py GOAL_START"
        ),
        "n_receipts": len(window),
        "files": files,
        "accepted_via_is_accepted_work": accepted,
        "naive_validation_ok": naive,
        "steer_cited_naive_ok": STEER_CITED_NAIVE_OK,
        "agrees_with_is_accepted_work": accepted == 0,
        "agrees_with_naive_ok": accepted == naive and naive > 0,
        "note": (
            f"steer cited naive_ok={STEER_CITED_NAIVE_OK}; live disk now has "
            f"{naive}. All window receipts are validation.ok=True kind=read_only "
            "answers. is_accepted_work scores 0 on every one. The axis agrees "
            "with is_accepted_work, not with naive ok. The count grew because "
            ".hcli/receipts is a live directory."
        ),
    }


# ---------------------------------------------------------------------------
# synthetic negative controls
# ---------------------------------------------------------------------------

def _syn_fail(i: int) -> dict[str, Any]:
    return {
        "model": "synthetic-fail",
        "endpoint": None,
        "goal": "synthetic-fail",
        "status": "failed",
        "rolled_back": False,
        "validation": {"ok": False, "checks": [{"kind": "test", "exit_code": 1}]},
        "result_envelope": {},
        "structured_output": {
            "mode": "degraded",
            "exhausted": True,
            "retries": 2,
            "errors": ["not a JSON object"],
            "last_violation": "not a JSON object",
        },
        "error_type": "StructuredOutputExhausted",
        "error": "structured output rejected after 3 attempts",
        "observations": [{"tool": "fs.read", "ok": False, "text": "missing"}],
        "timestamps": {
            "started_at": f"2026-09-07T00:00:0{i}+00:00",
            "finished_at": f"2026-09-07T00:01:0{i}+00:00",
        },
    }


def _syn_pass(i: int) -> dict[str, Any]:
    return {
        "model": "synthetic-pass",
        "endpoint": None,
        "goal": "synthetic-pass",
        "status": "completed",
        "rolled_back": False,
        "validation": {
            "ok": True,
            "checks": [{"kind": "test", "exit_code": 0}],
        },
        "result_envelope": {},
        "structured_output": {
            "mode": "degraded",
            "exhausted": False,
            "retries": 0,
            "errors": [],
        },
        "observations": [{"tool": "fs.read", "ok": True, "text": "{}"}],
        "timestamps": {
            "started_at": f"2026-09-07T00:00:0{i}+00:00",
            "finished_at": f"2026-09-07T00:01:0{i}+00:00",
        },
    }


def _syn_recovered(i: int) -> dict[str, Any]:
    rec = _syn_pass(i)
    rec["model"] = "synthetic-recovered"
    rec["goal"] = "synthetic-recovered"
    rec["structured_output"] = {
        "mode": "degraded",
        "exhausted": False,
        "retries": 1,
        "errors": [],
    }
    rec["observations"] = [
        {"tool": "fs.read", "ok": False, "text": "missing"},
        {"tool": "fs.read", "ok": True, "text": "{}"},
    ]
    return rec


def negative_control() -> dict[str, Any]:
    """Every-run-failed rates are 0.0 not None; every-run-succeeded vice versa."""
    failed = aggregate([_syn_fail(0), _syn_fail(1)])
    passed = aggregate([_syn_pass(0), _syn_pass(1)])
    recovered = aggregate([_syn_recovered(0), _syn_recovered(1)])
    f, p, r = failed["overall"], passed["overall"], recovered["overall"]

    def val(row: Mapping[str, Any], name: str) -> Any:
        cell = row[name]
        return cell["value"] if isinstance(cell, dict) else cell

    checks = {
        "all_failed_accepted_rate_is_0_0": val(f, "accepted_rate") == 0.0,
        "all_failed_recovery_rate_is_0_0": val(f, "recovery_rate") == 0.0,
        "all_failed_structured_action_failure_rate_is_1_0": val(f, "structured_action_failure_rate") == 1.0,
        "all_failed_tool_failure_rate_is_1_0": val(f, "tool_failure_rate") == 1.0,
        "all_failed_rolled_back_rate_is_0_0": val(f, "rolled_back_rate") == 0.0,
        "all_passed_accepted_rate_is_1_0": val(p, "accepted_rate") == 1.0,
        "all_passed_structured_action_failure_rate_is_0_0": val(p, "structured_action_failure_rate") == 0.0,
        "all_passed_tool_failure_rate_is_0_0": val(p, "tool_failure_rate") == 0.0,
        "all_passed_rolled_back_rate_is_0_0": val(p, "rolled_back_rate") == 0.0,
        "all_passed_recovery_rate_is_null": val(p, "recovery_rate") is None,
        "recovered_recovery_rate_is_1_0": val(r, "recovery_rate") == 1.0,
        "recovered_accepted_rate_is_1_0": val(r, "accepted_rate") == 1.0,
        "recovered_structured_action_failure_rate_is_0_0": val(r, "structured_action_failure_rate") == 0.0,
        "no_mean_in_failed_variance": "mean" not in f["variance"],
        "no_mean_in_passed_variance": "mean" not in p["variance"],
        "failed_variance_min_median_max_are_0": (
            f["variance"]["min"] == 0
            and f["variance"]["median"] == 0.0
            and f["variance"]["max"] == 0
        ),
        "passed_variance_min_median_max_are_1": (
            p["variance"]["min"] == 1
            and p["variance"]["median"] == 1.0
            and p["variance"]["max"] == 1
        ),
    }
    failed_names = [k for k, ok in checks.items() if not ok]
    return {
        "passed": not failed_names,
        "failed_checks": failed_names,
        "all_failed": {
            "accepted_rate": val(f, "accepted_rate"),
            "recovery_rate": val(f, "recovery_rate"),
            "structured_action_failure_rate": val(f, "structured_action_failure_rate"),
            "tool_failure_rate": val(f, "tool_failure_rate"),
            "rolled_back_rate": val(f, "rolled_back_rate"),
        },
        "all_passed": {
            "accepted_rate": val(p, "accepted_rate"),
            "recovery_rate": val(p, "recovery_rate"),
            "structured_action_failure_rate": val(p, "structured_action_failure_rate"),
            "tool_failure_rate": val(p, "tool_failure_rate"),
            "rolled_back_rate": val(p, "rolled_back_rate"),
        },
        "recovered": {
            "accepted_rate": val(r, "accepted_rate"),
            "recovery_rate": val(r, "recovery_rate"),
            "structured_action_failure_rate": val(r, "structured_action_failure_rate"),
            "tool_failure_rate": val(r, "tool_failure_rate"),
        },
    }


# ---------------------------------------------------------------------------
# mutation check
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _return_line(func: Any, tag: str) -> tuple[int, str]:
    """0-based line index of `func`'s tagged return, from the file on disk."""
    lines, start = inspect.getsourcelines(func)
    for offset, line in enumerate(lines):
        if tag in line and line.lstrip().startswith("return "):
            return start - 1 + offset, line
    raise AssertionError(f"no tagged return {tag!r} in {func.__name__}")


@contextmanager
def _mutated_return(func: Any, tag: str, new_return: str) -> Iterator[dict[str, str]]:
    path = Path(inspect.getfile(func)).resolve()
    original = path.read_text(encoding="utf-8")
    before = hashlib.sha256(original.encode("utf-8")).hexdigest()
    idx, line = _return_line(func, tag)
    parts = original.splitlines(keepends=True)
    if idx >= len(parts) or tag not in parts[idx]:
        raise AssertionError(f"{tag} not at expected line {idx + 1}")
    newline = "\n" if parts[idx].endswith("\n") else ""
    parts[idx] = new_return.rstrip("\n") + newline
    mutated = "".join(parts)
    if mutated == original:
        raise AssertionError("mutation did not change the source")
    try:
        path.write_text(mutated, encoding="utf-8")
        after = _sha256_file(path)
        if after == before:
            raise AssertionError("sha256 unchanged after mutation write")
        yield {
            "anchor": f"{path.name}:{idx + 1}:{line.strip()}",
            "replacement": new_return.strip(),
            "before_sha256": before,
            "after_sha256": after,
        }
    finally:
        path.write_text(original, encoding="utf-8")
        restored = _sha256_file(path)
        if restored != before:
            path.write_text(original, encoding="utf-8")
            raise AssertionError("failed to restore source after mutation")


def _probe(script: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    pyc = ROOT / "tools" / "future" / "__pycache__"
    for leftover in pyc.glob("reliability_axis*.pyc"):
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
    """Revert the load-bearing line, confirm the test FAILS, restore.

    Acceptance: substituting validation['ok'] makes the window control
    count fabricated answers as accepted, so accepted==0 fails.
    Recovery: dropping the accepted-work conjunct (`return hit_failure`)
    makes the all-failed synthetic report recovery_rate 1.0 instead of 0.0.
    """
    accept_replacement = (
        "    return receipt_naive_ok(receipt)  # MUTATION_ACTIVE acceptance"
    )
    recovery_replacement = (
        "    return receipt_hit_failure(receipt)  # MUTATION_ACTIVE recovery"
    )

    src = Path(__file__).read_text(encoding="utf-8")
    if "LOAD_BEARING_ACCEPTANCE" not in src or "LOAD_BEARING_RECOVERY" not in src:
        raise AssertionError("load-bearing tags missing before mutation")

    accept_probe = (
        "import sys\n"
        "sys.dont_write_bytecode = True\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from tools.future.reliability_axis import known_result_control\n"
        "c = known_result_control()\n"
        "assert c['accepted_via_is_accepted_work'] == 0, c\n"
    )
    recovery_probe = (
        "import sys\n"
        "sys.dont_write_bytecode = True\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from tools.future.reliability_axis import aggregate, _syn_fail\n"
        "row = aggregate([_syn_fail(0), _syn_fail(1)])['overall']['recovery_rate']\n"
        "assert row['value'] == 0.0, row\n"
    )

    with _mutated_return(receipt_is_accepted, "LOAD_BEARING_ACCEPTANCE", accept_replacement) as acc_meta:
        acc_run = _probe(accept_probe)
        if acc_run.returncode == 0:
            raise AssertionError(
                "acceptance mutation did not fail the window control; "
                f"stdout={acc_run.stdout!r} stderr={acc_run.stderr!r}"
            )
        acceptance = {
            **acc_meta,
            "probe_returncode": acc_run.returncode,
            "probe_failed_as_required": True,
            "stderr_tail": (acc_run.stderr or acc_run.stdout)[-400:],
        }

    with _mutated_return(receipt_recovered, "LOAD_BEARING_RECOVERY", recovery_replacement) as rec_meta:
        rec_run = _probe(recovery_probe)
        if rec_run.returncode == 0:
            raise AssertionError(
                "recovery mutation did not fail the all-failed recovery==0 check; "
                f"stdout={rec_run.stdout!r} stderr={rec_run.stderr!r}"
            )
        recovery = {
            **rec_meta,
            "probe_returncode": rec_run.returncode,
            "probe_failed_as_required": True,
            "stderr_tail": (rec_run.stderr or rec_run.stdout)[-400:],
        }

    text = Path(__file__).read_text(encoding="utf-8")
    if "# LOAD_BEARING_ACCEPTANCE" not in text or "# LOAD_BEARING_RECOVERY" not in text:
        raise AssertionError("load-bearing tags missing after restore")
    if "MUTATION_ACTIVE acceptance" in text.split("def receipt_is_accepted")[1].split("def ")[0]:
        raise AssertionError("acceptance mutation left in function")
    return {
        "acceptance": acceptance,
        "recovery": recovery,
        "restored": True,
        "restored_sha256": _sha256_file(Path(__file__).resolve()),
    }


# ---------------------------------------------------------------------------
# build / selftest
# ---------------------------------------------------------------------------

def _imported_predicate() -> dict[str, Any]:
    mod = inspect.getmodule(is_accepted_work)
    src = inspect.getsource(is_accepted_work)
    ours = Path(__file__).read_text(encoding="utf-8")
    # A copy would be a function definition at column 0. Mentions of the name
    # in comments and import lines are not a reimplementation.
    copied = any(line.startswith("def is_accepted_work") for line in ours.splitlines())
    return {
        "name": "is_accepted_work",
        "module": getattr(mod, "__name__", None),
        "file": inspect.getfile(is_accepted_work),
        "imported_not_copied": (getattr(mod, "__name__", None) == "hcli.engine") and not copied,
        "docstring_head": (is_accepted_work.__doc__ or "")[:240],
        "source_sha256_16": hashlib.sha256(src.encode()).hexdigest()[:16],
    }


def _nulls(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    out = []
    for name in (
        "accepted_rate",
        "structured_action_failure_rate",
        "tool_failure_rate",
        "recovery_rate",
        "rolled_back_rate",
        "variance",
    ):
        cell = summary.get(name)
        if not isinstance(cell, dict):
            continue
        if cell.get("value") is None and name != "variance":
            out.append({
                "field": name,
                "missing": cell.get("missing"),
                "keys": cell.get("keys"),
            })
        if name == "variance" and cell.get("min") is None:
            out.append({
                "field": "variance",
                "missing": cell.get("missing"),
                "keys": cell.get("keys"),
            })
    if summary.get("endpoint") is None and summary.get("endpoint_missing"):
        out.append({
            "field": "endpoint",
            "missing": summary.get("endpoint_missing"),
            "keys": ["endpoint"],
        })
    return out


def build_doc(
    *,
    controls: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    skipped: Sequence[Mapping[str, Any]],
    drive: Mapping[str, Any],
) -> dict[str, Any]:
    agg = aggregate(runs)
    overall_nulls = _nulls(agg["overall"])
    # endpoint is missing on every identity; report once at the top, not 12 times
    per_model_nulls = []
    for row in agg["models"]:
        items = [
            n for n in _nulls(row)
            if n["field"] != "endpoint"
        ]
        if items:
            per_model_nulls.append({"model": row["model"], "nulls": items})
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "recorded_by": RECORDED_BY,
        "purpose": (
            "Fill the campaign Pareto reliability axis from HCLI receipts. "
            "Reliability is a vector of rates, not a grade and not a blend."
        ),
        "claim_boundary": (
            "Static derivation from receipts already on disk. No hardware "
            "measurement. No interpolated rate. An absent field is null with "
            "the missing key named."
        ),
        "acceptance_predicate": _imported_predicate(),
        "pareto_axis": {
            "name": "reliability",
            "higher_is_better": True,
            "comparable_value": "accepted_rate",
            "comparable_value_is": (
                "hcli.engine.is_accepted_work / n_runs, harvested onto "
                "serving-identity points. Not a blend of the vector."
            ),
            "vector_stays": (
                "recovery_rate, structured_action_failure_rate, "
                "tool_failure_rate, rolled_back_rate, variance min/median/max"
            ),
        },
        "receipts_dir": str(receipts_dir()),
        "n_engine_runs": len(runs),
        "n_skipped": len(skipped),
        "skipped": list(skipped),
        "drive_rounds": drive,
        "models": agg["models"],
        "overall": agg["overall"],
        "nulls": {
            "overall": overall_nulls,
            "per_model": per_model_nulls,
            "rule": (
                "a null with a named missing field is evidence; a constructed "
                "number is not"
            ),
        },
        "controls": dict(controls),
        "field_keys": {
            "n_runs": "count of engine-run receipts in the identity group",
            "window": list(TIME_KEYS),
            "accepted_rate": {
                "keys": list(ACCEPTANCE_KEYS),
                "predicate": "hcli.engine.is_accepted_work",
                "not": "validation.ok",
            },
            "structured_action_failure_rate": {
                "keys": list(STRUCTURED_KEYS),
                "predicate": (
                    "error_type == StructuredOutputExhausted OR "
                    "structured_output.exhausted is True"
                ),
            },
            "tool_failure_rate": {
                "keys": list(TOOL_OBS_KEYS),
                "predicate": "observation.ok is False / n observations",
            },
            "recovery_rate": {
                "keys": list(FAILURE_KEYS),
                "predicate": "hit_failure AND is_accepted_work",
            },
            "rolled_back_rate": {"keys": list(ROLLED_BACK_KEYS)},
            "variance": {
                "keys": list(VARIANCE_KEYS),
                "reports": "min/median/max of accepted_work 0/1 over repeated goal text",
                "never": "mean",
            },
            "identity": {"keys": list(IDENTITY_KEYS)},
        },
    }


def selftest() -> dict[str, Any]:
    pred = _imported_predicate()
    if not pred["imported_not_copied"]:
        raise AssertionError(f"is_accepted_work must be imported from hcli.engine: {pred}")
    if "fabricated" not in (is_accepted_work.__doc__ or "").lower() and "ANSWER" not in (is_accepted_work.__doc__ or ""):
        raise AssertionError("imported is_accepted_work docstring does not match the known authority")

    # The imported predicate itself still refuses the shapes that fooled counters.
    if is_accepted_work({"ok": True}) is not False:
        raise AssertionError("bare ok=True must not count as accepted work")
    if is_accepted_work({"ok": True, "kind": "read_only",
                         "checks": [{"kind": "test", "exit_code": 0}]}) is not False:
        raise AssertionError("read_only must not count as accepted work")
    if is_accepted_work({"ok": True, "checks": [{"kind": "test", "exit_code": 0}]}) is not True:
        raise AssertionError("a passing test check must count as accepted work")

    runs, skipped = load_receipts()
    if len(runs) < 100:
        raise AssertionError(f"expected hundreds of engine receipts, got {len(runs)} from {receipts_dir()}")

    known = known_result_control(runs)
    if known["accepted_via_is_accepted_work"] != 0:
        raise AssertionError(
            "window is_accepted_work must score 0 accepted; "
            f"got {known}"
        )
    if known["naive_validation_ok"] < STEER_CITED_NAIVE_OK:
        raise AssertionError(
            f"window naive validation.ok should be at least {STEER_CITED_NAIVE_OK}; "
            f"got {known}"
        )
    if known["agrees_with_naive_ok"]:
        raise AssertionError(
            "aggregator agrees with validation['ok']; that is the bug this axis exists to avoid"
        )

    neg = negative_control()
    if not neg["passed"]:
        raise AssertionError(f"negative control failed: {neg['failed_checks']}")

    mut = mutation_check()
    src = Path(__file__).read_text(encoding="utf-8")
    if "# LOAD_BEARING_ACCEPTANCE" not in src or "# LOAD_BEARING_RECOVERY" not in src:
        raise AssertionError("mutation left a load-bearing tag unrestored")
    if mut["acceptance"]["before_sha256"] == mut["acceptance"]["after_sha256"]:
        raise AssertionError("acceptance mutation did not change sha256")
    if mut["recovery"]["before_sha256"] == mut["recovery"]["after_sha256"]:
        raise AssertionError("recovery mutation did not change sha256")

    drive = load_drive_rounds()
    pareto = None
    receipt_path = ROOT / "receipts" / "future" / RECEIPT_NAME
    if receipt_path.is_file():
        pareto = _assert_pareto_reliability()
    return {
        "ok": True,
        "n_engine_runs": len(runs),
        "n_skipped": len(skipped),
        "known_result": known,
        "negative_control": neg,
        "mutation_check": mut,
        "drive_rounds": {"n": drive.get("n"), "missing": drive.get("missing")},
        "acceptance_predicate": pred,
        "pareto_harvest": pareto,
        "runs": runs,
        "skipped": skipped,
        "drive": drive,
    }


def _assert_pareto_reliability() -> dict[str, Any]:
    """G016 verify: campaign_pareto reports a non-null reliability value."""
    from tools.future.campaign_pareto import harvest as pareto_harvest

    doc = pareto_harvest()
    debt = doc["measurement_debt"]
    n = debt["n_with_axis"].get("reliability", 0)
    if n < 1:
        raise AssertionError(
            "campaign_pareto still reports reliability unmeasured for everyone: "
            f"{debt}"
        )
    if "reliability" in debt["unmeasured_for_everyone"]:
        raise AssertionError(
            "reliability remains in unmeasured_for_everyone after harvest"
        )
    points = [
        p for p in doc["scored"]
        if p.get("kind") == "serving_identity" and "reliability" in (p.get("values") or {})
    ]
    if not points:
        raise AssertionError("no serving_identity point carries reliability")
    # The comparable value is accepted_rate, not a blend.
    for p in points:
        contract = p.get("reliability_contract") or {}
        if contract.get("named_rate") != "accepted_rate":
            raise AssertionError(f"harvested a blend or unnamed rate: {p['id']} {contract}")
        if abs(float(p["values"]["reliability"]) - float(contract["accepted_rate"])) > 1e-12:
            raise AssertionError(
                f"{p['id']} reliability {p['values']['reliability']} "
                f"!= accepted_rate {contract['accepted_rate']}"
            )
    return {
        "n_with_reliability": n,
        "n_serving_points": len(points),
        "unmeasured_for_everyone": debt["unmeasured_for_everyone"],
        "sample": [
            {
                "id": p["id"],
                "reliability": p["values"]["reliability"],
                "n_runs": (p.get("reliability_contract") or {}).get("n_runs"),
            }
            for p in sorted(points, key=lambda r: -r["values"]["reliability"])[:5]
        ],
    }


def _fmt_rate(cell: Mapping[str, Any] | None) -> str:
    if not isinstance(cell, dict):
        return "—"
    if cell.get("value") is None:
        missing = str(cell.get("missing") or "null")
        if "observations" in missing:
            return "null (missing: observations)"
        if "structured_output" in missing:
            return "null (missing: structured_output)"
        if "no run hit a failure" in missing:
            return "null (no failures)"
        return f"null ({missing[:48]})"
    return f"{cell['n']}/{cell['d']} = {cell['value']:.4f}"


def _short_model(model: Any) -> str:
    if model is None:
        return "<missing>"
    s = str(model)
    if s.startswith("http"):
        return s
    return Path(s).name


def print_table(doc: Mapping[str, Any]) -> None:
    rows = doc.get("models") or []
    hdr = (
        f"{'model':<44} {'n':>4} {'accepted':<20} {'so_fail':<22} "
        f"{'tool_fail':<28} {'recovery':<20} {'rolled':<18}"
    )
    print()
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        var = row.get("variance") or {}
        print(
            f"{_short_model(row.get('model')):<44} "
            f"{row.get('n_runs', 0):>4} "
            f"{_fmt_rate(row.get('accepted_rate')):<20} "
            f"{_fmt_rate(row.get('structured_action_failure_rate')):<22} "
            f"{_fmt_rate(row.get('tool_failure_rate')):<28} "
            f"{_fmt_rate(row.get('recovery_rate')):<20} "
            f"{_fmt_rate(row.get('rolled_back_rate')):<18}"
        )
        if var.get("n_goals_run_more_than_once"):
            print(
                f"{'':44} variance min/median/max="
                f"{var.get('min')}/{var.get('median')}/{var.get('max')} "
                f"repeated_goals={var.get('n_goals_run_more_than_once')} "
                f"mixed={var.get('n_mixed_goals')}"
            )
    print("-" * len(hdr))
    ov = doc.get("overall") or {}
    print(
        f"{'*all*':<44} "
        f"{ov.get('n_runs', 0):>4} "
        f"{_fmt_rate(ov.get('accepted_rate')):<20} "
        f"{_fmt_rate(ov.get('structured_action_failure_rate')):<22} "
        f"{_fmt_rate(ov.get('tool_failure_rate')):<28} "
        f"{_fmt_rate(ov.get('recovery_rate')):<20} "
        f"{_fmt_rate(ov.get('rolled_back_rate')):<18}"
    )
    rec = ov.get("recovery_rate") or {}
    print()
    print("recovery_rate (overall): "
          f"{_fmt_rate(rec)} — of the runs that hit a failure, "
          "how many still closed with accepted work.")
    print("tool_failure_rate is null: receipts do not persist `observations`.")
    print("endpoint is null: receipt.endpoint is not a field; grouping is by receipt.model.")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    selftest_only = "--selftest" in args
    build = "--build" in args or not args or selftest_only

    gate = selftest()
    print("selftest: PASS")
    print(
        f"  known-result window {gate['known_result']['window_start']}: "
        f"accepted={gate['known_result']['accepted_via_is_accepted_work']} "
        f"naive_ok={gate['known_result']['naive_validation_ok']} "
        f"(steer cited {STEER_CITED_NAIVE_OK})"
    )
    print(
        f"  negative control: all-failed accepted={gate['negative_control']['all_failed']['accepted_rate']} "
        f"recovery={gate['negative_control']['all_failed']['recovery_rate']}; "
        f"all-passed accepted={gate['negative_control']['all_passed']['accepted_rate']} "
        f"tool_fail={gate['negative_control']['all_passed']['tool_failure_rate']}"
    )
    print(
        f"  mutation: acceptance {gate['mutation_check']['acceptance']['before_sha256'][:12]}→"
        f"{gate['mutation_check']['acceptance']['after_sha256'][:12]} "
        f"recovery {gate['mutation_check']['recovery']['before_sha256'][:12]}→"
        f"{gate['mutation_check']['recovery']['after_sha256'][:12]} restored="
        f"{gate['mutation_check']['restored']}"
    )

    if not build:
        return 0

    controls = {
        "known_result": gate["known_result"],
        "negative_control": gate["negative_control"],
        "mutation_check": gate["mutation_check"],
        "pareto_harvest": gate.get("pareto_harvest"),
    }
    doc = build_doc(
        controls=controls,
        runs=gate["runs"],
        skipped=gate["skipped"],
        drive=gate["drive"],
    )
    # Drop the in-memory run list from what we persist — the receipt cites
    # paths and counts, not a copy of every source receipt.
    path = write_receipt(RECEIPT_NAME, doc, RECORDED_BY)
    print(f"wrote {path}")
    print_table(doc)
    pareto = _assert_pareto_reliability()
    print(
        f"campaign_pareto harvest: reliability on {pareto['n_with_reliability']} "
        f"serving identities; still unmeasured: {pareto['unmeasured_for_everyone']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
