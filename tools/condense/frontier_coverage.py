#!/usr/bin/env python3.12
"""frontier_coverage.py - baseline/eval coverage rules for Studio frontier claims.

This module is deliberately receipt-shaped, not benchmark-shaped. It does not measure anything; it
answers one narrow question for each frontier label: do we have enough machine-readable coverage to
let a public claim proceed?

Coverage is satisfied by either measured same-box evidence or an explicit N/A with a reason. Missing
silence never counts as coverage.
"""
from __future__ import annotations

import pathlib
import re
from typing import Any

from frontier_common import is_sha256 as _is_sha256, read_json as _read_json  # noqa: E402

COND_DIR = pathlib.Path("reports/condense")

BASELINE_REQUIREMENTS = (
    {
        "name": "llama.cpp Q4_K_M",
        "aliases": ("llama.cpp q4_k_m", "q4_k_m", "q4_k_s", "iq4_xs"),
    },
    {
        "name": "llama.cpp IQ2_S",
        "aliases": ("llama.cpp iq2_s", "iq2_s", "iq2_xxs", "iq1_s"),
    },
    {
        "name": "llama.cpp mmap OOC",
        "aliases": ("llama.cpp mmap", "mmap ooc", "out-of-core", "ooc"),
    },
    {
        "name": "MLX 4-bit",
        "aliases": ("mlx 4-bit", "mlx-4bit", "mlx_lm mlx-4bit", "mlx_lm 4bit"),
    },
    {
        "name": "Unsloth Dyn 2.0",
        "aliases": ("unsloth", "dyn 2.0", "dynamic gguf", "dynamic-gguf"),
    },
    {
        "name": "EXL3 / PonyExl3",
        "aliases": ("exl3", "ponyexl3", "pony exl3", "aqlm", "qtip"),
    },
)

EVAL_REQUIREMENTS = (
    {
        "name": "ppl_multiwindow",
        "aliases": ("ppl", "perplexity", "multiwindow", "multiwindow_ppl"),
    },
    {
        "name": "capability_qa",
        "aliases": ("capability", "qa", "cloze", "closed_book_qa"),
    },
    {
        "name": "math",
        "aliases": ("math", "gsm", "arithmetic"),
    },
    {
        "name": "coding",
        "aliases": ("coding", "code", "code_completion"),
    },
    {
        "name": "tool_use",
        "aliases": ("tool_use", "tool-use", "function_calling", "agentic"),
    },
    {
        "name": "long_context_recall",
        "aliases": ("long_context", "long-context", "niah", "needle", "recall"),
    },
    {
        "name": "ram_cliff",
        "aliases": ("ram_cliff", "ram-cliff", "cliff", "resident_vs_swap"),
    },
    {
        "name": "native_serve",
        "aliases": ("native_serve", "native-serve", "serve", "tq_serve"),
    },
)

PASS_STATUSES = {"pass", "passed", "ok", "done", "complete", "measured", "allow"}
NA_STATUSES = {"na", "n/a", "n-a", "not-applicable", "not_applicable", "unavailable", "waived", "skip", "skipped"}
FAIL_STATUSES = {"fail", "failed", "error", "blocked", "missing"}
SYNTHETIC_MODES = {"synthetic", "mock", "fake"}


def _norm(s: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _status(s: Any) -> str:
    return _norm(s).replace(" ", "-")


def _truthy(v: Any) -> bool:
    return v is True or _status(v) in PASS_STATUSES


def _reason(entry: dict[str, Any], record: dict[str, Any]) -> str:
    for key in ("reason", "na_reason", "note", "justification", "why"):
        val = entry.get(key)
        if val:
            return str(val)
    return str(record.get("na_reason") or record.get("note") or "")


def _match_requirement(name: str, req: dict[str, Any]) -> bool:
    hay = _norm(name)
    aliases = (req["name"], *req.get("aliases", ()))
    return any((needle := _norm(alias)) and (needle in hay or hay in needle) for alias in aliases)


def _entry_name(entry: dict[str, Any]) -> str:
    bits = [
        entry.get("name"),
        entry.get("baseline"),
        entry.get("domain"),
        entry.get("engine"),
        entry.get("tier"),
        entry.get("task"),
    ]
    return " ".join(str(b) for b in bits if b)


def _path(root: pathlib.Path, label: str, suffix: str) -> pathlib.Path:
    return root / COND_DIR / f"{label}_{suffix}.json"


def baseline_path(root: pathlib.Path, label: str) -> pathlib.Path:
    return _path(root, label, "baselines")


def eval_path(root: pathlib.Path, label: str) -> pathlib.Path:
    return _path(root, label, "eval")


def _baseline_entries(record: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for key in ("baselines", "baseline_receipts", "coverage", "requirements"):
        val = record.get(key)
        if isinstance(val, dict):
            for name, row in val.items():
                if isinstance(row, dict):
                    item = dict(row)
                    item.setdefault("name", name)
                    entries.append(item)
        elif isinstance(val, list):
            entries.extend(dict(x) for x in val if isinstance(x, dict))
    for row in record.get("head_to_head") or []:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        item.setdefault("name", _entry_name(item))
        if "status" not in item:
            item["status"] = "measured" if item.get("available") else "unavailable"
        entries.append(item)
    return entries


def _domain_entries_from_eval_suite(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate eval_suite.py's native shape into coverage-domain rows."""
    if "gate" not in record:
        return []
    gate = record.get("gate") or {}
    task_fails = gate.get("task_fails") if isinstance(gate.get("task_fails"), dict) else {}
    mode = record.get("mode")
    rows = []
    if gate.get("tasks_pass") is True:
        rows.extend([
            {"domain": "capability_qa", "status": "pass", "source": mode},
            {"domain": "math", "status": "pass", "source": mode},
            {"domain": "coding", "status": "pass", "source": mode},
        ])
    else:
        for domain, task in (("capability_qa", "qa"), ("math", "math"), ("coding", "code")):
            status = "fail" if task in task_fails else "missing"
            rows.append({"domain": domain, "status": status, "reason": f"eval_suite task {task} did not pass"})
    rows.append({
        "domain": "long_context_recall",
        "status": "pass" if gate.get("niah_pass") else "fail",
        "reason": gate.get("niah_note") or "NIAH did not pass",
        "source": mode,
    })
    return rows


def _eval_entries(record: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    val = record.get("domains")
    if isinstance(val, dict):
        for name, row in val.items():
            item = dict(row) if isinstance(row, dict) else {"status": row}
            item.setdefault("domain", name)
            entries.append(item)
    elif isinstance(val, list):
        entries.extend(dict(x) for x in val if isinstance(x, dict))
    entries.extend(_domain_entries_from_eval_suite(record))
    return entries


def _classify_baseline(record: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    mode = _status(entry.get("mode") or entry.get("source") or record.get("mode") or record.get("source"))
    status = _status(entry.get("status") or entry.get("coverage_status"))
    available = entry.get("available")
    reason = _reason(entry, record)
    same_box = entry.get("same_box", record.get("same_box"))
    machine = entry.get("machine_class") or record.get("machine_class")

    if mode in SYNTHETIC_MODES:
        return {"status": "invalid", "ok": False, "problem": "synthetic baseline cannot cover a claim"}
    if status in PASS_STATUSES or available is True or _truthy(entry.get("measured")):
        if same_box is not True:
            return {"status": "invalid", "ok": False, "problem": "measured baseline must declare same_box=true"}
        if not machine:
            return {"status": "invalid", "ok": False, "problem": "measured baseline lacks machine_class"}
        return {"status": "measured", "ok": True, "machine_class": machine, "same_box": bool(same_box)}
    if status in NA_STATUSES or available is False:
        if reason:
            return {"status": "na", "ok": True, "reason": reason}
        return {"status": "invalid", "ok": False, "problem": "N/A baseline lacks a reason"}
    if status in FAIL_STATUSES:
        return {"status": "invalid", "ok": False, "problem": reason or f"baseline status={status}"}
    return {"status": "missing", "ok": False, "problem": "baseline entry has no measured/pass or N/A status"}


def _classify_eval(record: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    mode = _status(entry.get("mode") or entry.get("source") or record.get("mode") or record.get("source"))
    status = _status(entry.get("status") or entry.get("coverage_status"))
    reason = _reason(entry, record)
    same_box = entry.get("same_box", record.get("same_box"))
    suite_sha = entry.get("frozen_suite_sha256") or record.get("frozen_suite_sha256")
    score_sha = entry.get("score_set_sha256") or record.get("score_set_sha256")
    if mode in SYNTHETIC_MODES:
        return {"status": "invalid", "ok": False, "problem": "synthetic eval cannot cover a claim"}
    if status in PASS_STATUSES or _truthy(entry.get("pass")) or _truthy(entry.get("measured")):
        if same_box is not True:
            return {"status": "invalid", "ok": False, "problem": "eval domain must declare same_box=true"}
        if not _is_sha256(suite_sha):
            return {"status": "invalid", "ok": False, "problem": "eval domain lacks frozen_suite_sha256"}
        if not _is_sha256(score_sha):
            return {"status": "invalid", "ok": False, "problem": "eval domain lacks score_set_sha256"}
        return {"status": "pass", "ok": True}
    if status in NA_STATUSES:
        if reason:
            return {"status": "na", "ok": True, "reason": reason}
        return {"status": "invalid", "ok": False, "problem": "N/A eval domain lacks a reason"}
    if status in FAIL_STATUSES:
        return {"status": "invalid", "ok": False, "problem": reason or f"eval status={status}"}
    return {"status": "missing", "ok": False, "problem": "eval domain has no pass/measured or N/A status"}


def _coverage_status(root: pathlib.Path, label: str, suffix: str, requirements: tuple[dict[str, Any], ...],
                     entries_fn, classify_fn) -> dict[str, Any]:
    path = _path(root, label, suffix)
    record = _read_json(path)
    rows = []
    if not record:
        for req in requirements:
            rows.append({
                "requirement": req["name"],
                "status": "missing",
                "ok": False,
                "problem": f"{path} is missing or unreadable",
            })
        return {
            "label": label,
            "path": str(path),
            "exists": False,
            "ok": False,
            "covered": 0,
            "required": len(requirements),
            "rows": rows,
            "problems": [r["problem"] for r in rows],
        }
    entries = entries_fn(record)
    for req in requirements:
        entry = next((e for e in entries if _match_requirement(_entry_name(e), req)), None)
        if not entry:
            rows.append({
                "requirement": req["name"],
                "status": "missing",
                "ok": False,
                "problem": "no measured or explicit N/A row",
            })
            continue
        classified = classify_fn(record, entry)
        row = {"requirement": req["name"], "entry": _entry_name(entry), **classified}
        rows.append(row)
    problems = [f"{r['requirement']}: {r.get('problem')}" for r in rows if not r["ok"]]
    return {
        "label": label,
        "path": str(path),
        "exists": True,
        "schema": record.get("schema"),
        "machine_class": record.get("machine_class"),
        "ok": not problems,
        "covered": sum(1 for r in rows if r["ok"]),
        "required": len(requirements),
        "measured": sum(1 for r in rows if r.get("status") in ("measured", "pass")),
        "na": sum(1 for r in rows if r.get("status") == "na"),
        "rows": rows,
        "problems": problems,
    }


def baseline_status(root: pathlib.Path, label: str) -> dict[str, Any]:
    return _coverage_status(root, label, "baselines", BASELINE_REQUIREMENTS,
                            _baseline_entries, _classify_baseline)


def eval_status(root: pathlib.Path, label: str) -> dict[str, Any]:
    return _coverage_status(root, label, "eval", EVAL_REQUIREMENTS,
                            _eval_entries, _classify_eval)


def _rollup(kind: str, labels: list[str], rows: list[dict[str, Any]]) -> dict[str, Any]:
    blocked = [r["label"] for r in rows if not r["ok"]]
    return {
        "schema": "hawking.frontier_coverage_rollup.v1",
        "kind": kind,
        "model_count": len(rows),
        "passed_count": len(rows) - len(blocked),
        "blocked_count": len(blocked),
        "blocked_labels": blocked,
        "requirements": [r["name"] for r in (BASELINE_REQUIREMENTS if kind == "baseline" else EVAL_REQUIREMENTS)],
        "rows": rows,
        "labels": labels,
        "ok": not blocked,
    }


def baseline_rollup(root: pathlib.Path, labels: list[str]) -> dict[str, Any]:
    return _rollup("baseline", labels, [baseline_status(root, label) for label in labels])


def eval_rollup(root: pathlib.Path, labels: list[str]) -> dict[str, Any]:
    return _rollup("eval", labels, [eval_status(root, label) for label in labels])


def _baseline_skeleton(label: str) -> dict[str, Any]:
    return {
        "schema": "hawking.frontier_baselines.v1",
        "model": label,
        "machine_class": "Studio-M3Ultra-96",
        "machine_name": "<exact Studio host label>",
        "same_box": True,
        "same_box_group": "<same machine/session id shared by baselines/evals>",
        "machine_fingerprint_sha256": "<64 hex>",
        "environment_receipt": "<hawking studio environment-capture receipt>",
        "frozen_suite_sha256": "<64 hex>",
        "score_set_sha256": "<64 hex>",
        "score_set_receipt": "<frozen score-set receipt>",
        "baseline_best_effort": False,
        "run_id": "<same-run id>",
        "baselines": [
            {
                "name": req["name"],
                "status": "TODO measured|na",
                "same_box": True,
                "baseline_best_effort": False,
                "command": "<exact command or N/A>",
                "artifact": "<path or N/A>",
                "metrics": {},
                "reason": "<required if status=na>",
            }
            for req in BASELINE_REQUIREMENTS
        ],
    }


def _eval_skeleton(label: str) -> dict[str, Any]:
    return {
        "schema": "hawking.frontier_eval_coverage.v1",
        "model": label,
        "mode": "real",
        "machine_class": "Studio-M3Ultra-96",
        "machine_name": "<exact Studio host label>",
        "same_box": True,
        "same_box_group": "<same machine/session id shared by baselines/evals>",
        "machine_fingerprint_sha256": "<64 hex>",
        "environment_receipt": "<hawking studio environment-capture receipt>",
        "frozen_suite_sha256": "<64 hex>",
        "score_set_sha256": "<64 hex>",
        "score_set_receipt": "<frozen score-set receipt>",
        "run_id": "<same-run id>",
        "domains": [
            {
                "domain": req["name"],
                "status": "TODO pass|na",
                "same_box": True,
                "command": "<exact command or N/A>",
                "receipt": "<path to result or N/A>",
                "reason": "<required if status=na>",
            }
            for req in EVAL_REQUIREMENTS
        ],
    }


def coverage_plan(root: pathlib.Path, labels: list[str]) -> dict[str, Any]:
    return {
        "schema": "hawking.frontier_coverage_plan.v1",
        "baseline_requirements": [r["name"] for r in BASELINE_REQUIREMENTS],
        "eval_requirements": [r["name"] for r in EVAL_REQUIREMENTS],
        "labels": [
            {
                "label": label,
                "baseline_path": str(baseline_path(root, label)),
                "eval_path": str(eval_path(root, label)),
                "baseline_skeleton": _baseline_skeleton(label),
                "eval_skeleton": _eval_skeleton(label),
            }
            for label in labels
        ],
    }
