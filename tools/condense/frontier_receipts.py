#!/usr/bin/env python3.12
"""frontier_receipts.py - strict receipt validation for frontier serve and RAM-cliff claims.

The surrounding tools can produce many useful probe records. This module decides which records are
admissible for public frontier claims. It is intentionally stricter than a casual JSON reader:
synthetic/modelled/gated rows are evidence that work was attempted, but they cannot unlock a claim.
"""
from __future__ import annotations

import json
import pathlib
import re
from typing import Any

COND_DIR = pathlib.Path("reports/condense")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
CLIFF_X_GATE = 10.0


def _read_json(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        return json.load(open(path))
    except Exception:
        return None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.match(value))


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value > 0


def _commands(record: dict[str, Any]) -> list[Any]:
    cmds = record.get("commands")
    if isinstance(cmds, list):
        return cmds
    cmd = record.get("command")
    return [cmd] if cmd else []


def _commit(record: dict[str, Any]) -> Any:
    return record.get("hawking_commit") or record.get("git_commit")


def _machine(record: dict[str, Any]) -> Any:
    return record.get("machine_class") or (record.get("hardware") or {}).get("profile")


def _source(record: dict[str, Any]) -> str:
    return str(record.get("source") or record.get("mode") or "").lower()


def _path(root: pathlib.Path, label: str, suffix: str) -> pathlib.Path:
    return root / COND_DIR / f"{label}_{suffix}.json"


def serve_path(root: pathlib.Path, label: str) -> pathlib.Path:
    return _path(root, label, "serve")


def ramcliff_path(root: pathlib.Path, label: str) -> pathlib.Path:
    return _path(root, label, "ramcliff")


def _base_status(label: str, path: pathlib.Path, record: dict[str, Any] | None,
                 schema: str) -> tuple[list[str], dict[str, Any]]:
    if not record:
        return [f"{path} is missing or unreadable"], {
            "label": label,
            "path": str(path),
            "exists": False,
            "ok": False,
            "problems": [f"{path} is missing or unreadable"],
        }
    problems = []
    if record.get("schema") != schema:
        problems.append(f"schema must be {schema}")
    if (record.get("model") or record.get("label")) != label:
        problems.append("model/label does not match manifest label")
    if _source(record) in {"synthetic", "modeled", "modelled", "mock", "gated"}:
        problems.append("synthetic/modelled/gated record cannot unlock a claim")
    if not _machine(record):
        problems.append("machine_class or hardware.profile is missing")
    if not _commit(record):
        problems.append("git_commit/hawking_commit is missing")
    if not _commands(record):
        problems.append("exact command(s) are missing")
    if not _is_sha256(record.get("artifact_sha256")):
        problems.append("artifact_sha256 is missing or invalid")
    return problems, {
        "label": label,
        "path": str(path),
        "exists": True,
        "schema": record.get("schema"),
        "machine_class": _machine(record),
    }


def serve_status(root: pathlib.Path, label: str) -> dict[str, Any]:
    path = serve_path(root, label)
    record = _read_json(path)
    problems, status = _base_status(label, path, record, "hawking.frontier_serve.v1")
    if record:
        required_true = (
            "native_tq",
            "tq_strict",
            "all_linear",
            "gpu_bitslice",
            "served_forward_pass",
        )
        for key in required_true:
            if record.get(key) is not True:
                problems.append(f"{key} must be true")
        if record.get("rehydrate_f16") is not False:
            problems.append("rehydrate_f16 must be false")
        if record.get("status") != "pass":
            problems.append("status must be pass")
        if not _positive_number(record.get("tok_s")):
            problems.append("tok_s must be positive")
        if record.get("parity_pass") is not True:
            problems.append("parity_pass must be true")
    status.update({
        "ok": not problems,
        "problems": problems,
        "tok_s": record.get("tok_s") if record else None,
        "native_tq": record.get("native_tq") if record else None,
    })
    return status


def ramcliff_status(root: pathlib.Path, label: str) -> dict[str, Any]:
    path = ramcliff_path(root, label)
    record = _read_json(path)
    problems, status = _base_status(label, path, record, "hawking.frontier_ramcliff.v1")
    if record:
        gate = record.get("gate") if isinstance(record.get("gate"), dict) else {}
        for key in (
            "condensed_resident",
            "served_native_tq",
            "q4k_overflows_box",
            "cliff_x_over_gate",
            "resident_lower_energy",
        ):
            if gate.get(key) is not True:
                problems.append(f"gate.{key} must be true")
        if record.get("verdict") != "CLIFF-WIN":
            problems.append("verdict must be CLIFF-WIN")
        if record.get("source") != "measured":
            problems.append("source must be measured")
        if record.get("served_native_tq") is not True:
            problems.append("served_native_tq must be true")
        if not _positive_number(record.get("tok_s_resident")):
            problems.append("tok_s_resident must be positive")
        if not _positive_number(record.get("tok_s_swapping")):
            problems.append("tok_s_swapping must be positive")
        if not _positive_number(record.get("j_per_tok_resident")):
            problems.append("j_per_tok_resident must be positive")
        if not _positive_number(record.get("j_per_tok_swapping")):
            problems.append("j_per_tok_swapping must be positive")
        cliff_x = record.get("cliff_x")
        if not _positive_number(cliff_x) or cliff_x <= CLIFF_X_GATE:
            problems.append(f"cliff_x must be > {CLIFF_X_GATE}")
        if (_positive_number(record.get("j_per_tok_resident"))
                and _positive_number(record.get("j_per_tok_swapping"))
                and record["j_per_tok_resident"] >= record["j_per_tok_swapping"]):
            problems.append("resident J/tok must be lower than swapping J/tok")
    status.update({
        "ok": not problems,
        "problems": problems,
        "cliff_x": record.get("cliff_x") if record else None,
        "j_per_tok_resident": record.get("j_per_tok_resident") if record else None,
    })
    return status


def _rollup(kind: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    blocked = [r["label"] for r in rows if not r["ok"]]
    return {
        "schema": "hawking.frontier_receipt_rollup.v1",
        "kind": kind,
        "model_count": len(rows),
        "passed_count": len(rows) - len(blocked),
        "blocked_count": len(blocked),
        "blocked_labels": blocked,
        "rows": rows,
        "ok": not blocked,
    }


def serve_rollup(root: pathlib.Path, labels: list[str]) -> dict[str, Any]:
    return _rollup("serve", [serve_status(root, label) for label in labels])


def ramcliff_rollup(root: pathlib.Path, labels: list[str]) -> dict[str, Any]:
    return _rollup("ramcliff", [ramcliff_status(root, label) for label in labels])


def receipt_plan(root: pathlib.Path, labels: list[str]) -> dict[str, Any]:
    return {
        "schema": "hawking.frontier_receipt_plan.v1",
        "serve_schema": "hawking.frontier_serve.v1",
        "ramcliff_schema": "hawking.frontier_ramcliff.v1",
        "labels": [
            {
                "label": label,
                "serve_path": str(serve_path(root, label)),
                "ramcliff_path": str(ramcliff_path(root, label)),
                "serve_capture_command_template": (
                    "python3.12 tools/condense/frontier_ops.py serve-capture "
                    f"{label} --artifact <artifact.tq> --bench-json <serve_report.json> "
                    "--command '<exact hawking serve bench command>' "
                    "--served-forward-receipt <served_forward_trace.json> "
                    "--parity-receipt <serve_parity_trace.json> --force"
                ),
                "serve_required": {
                    "schema": "hawking.frontier_serve.v1",
                    "model": label,
                    "status": "pass",
                    "native_tq": True,
                    "rehydrate_f16": False,
                    "tq_strict": True,
                    "all_linear": True,
                    "gpu_bitslice": True,
                    "served_forward_pass": True,
                    "parity_pass": True,
                    "tok_s": ">0",
                    "artifact_sha256": "<64 hex>",
                    "commands": ["<exact command>"],
                    "machine_class": "Studio-M1Ultra-128",
                    "git_commit": "<commit>",
                },
                "ramcliff_required": {
                    "schema": "hawking.frontier_ramcliff.v1",
                    "model": label,
                    "source": "measured",
                    "verdict": "CLIFF-WIN",
                    "served_native_tq": True,
                    "tok_s_resident": ">0",
                    "tok_s_swapping": ">0",
                    "j_per_tok_resident": ">0 and lower than swapping",
                    "j_per_tok_swapping": ">0",
                    "cliff_x": f">{CLIFF_X_GATE}",
                    "artifact_sha256": "<64 hex>",
                    "commands": ["<exact command>"],
                    "machine_class": "Studio-M1Ultra-128",
                    "git_commit": "<commit>",
                },
            }
            for label in labels
        ],
    }
