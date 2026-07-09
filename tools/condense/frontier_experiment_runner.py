#!/usr/bin/env python3.12
"""frontier_experiment_runner.py - draft, sign, and verify expensive-mode experiment matrices.

`frontier_experiments.py` defines the required depth for frontier claims. This file adds the signed
integrity layer: draft blocked envelopes before a run, and final signatures only after the matrix covers
the required seeds, ablations, bpw rungs, RAM-cliff repeats, baselines, nulls, and hash/rebake proof with
row-level traces.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "tools" / "condense"))

from studio_manifest import FRONTIER_MODELS, frontier_by_label  # noqa: E402
import frontier_experiments  # noqa: E402

SIGN_ALG = "sha256-json-v1"
SCHEMA = "hawking.frontier_experiment_matrix.v1"
FINAL_SOURCES = {"real", "measured"}


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _git_commit(root: pathlib.Path = ROOT) -> str:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def _read_json(path: pathlib.Path) -> dict[str, Any] | None:
    try:
        return json.load(open(path))
    except Exception:
        return None


def _write_json(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _canonical_digest(data: dict[str, Any]) -> str:
    unsigned = copy.deepcopy(data)
    unsigned.pop("signature", None)
    return hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _placeholder(value: Any) -> bool:
    return isinstance(value, str) and ("<" in value or "TODO" in value or "..." in value)


def _commands(record: dict[str, Any]) -> list[str]:
    out = []
    cmds = record.get("commands")
    if isinstance(cmds, list):
        out.extend(str(cmd) for cmd in cmds if cmd)
    if record.get("command"):
        out.append(str(record["command"]))
    return out


def _row_commands(row: dict[str, Any]) -> list[str]:
    out = []
    cmds = row.get("commands")
    if isinstance(cmds, list):
        out.extend(str(cmd) for cmd in cmds if cmd)
    if row.get("command"):
        out.append(str(row["command"]))
    return out


def signature_status(record: dict[str, Any]) -> dict[str, Any]:
    sig = record.get("signature") if isinstance(record.get("signature"), dict) else {}
    expected = _canonical_digest(record)
    ok = sig.get("algorithm") == SIGN_ALG and sig.get("digest") == expected
    problems = []
    if sig.get("algorithm") != SIGN_ALG:
        problems.append(f"signature algorithm must be {SIGN_ALG}")
    if sig.get("digest") != expected:
        problems.append("signature digest mismatch")
    return {
        "ok": ok,
        "algorithm": sig.get("algorithm"),
        "digest": sig.get("digest"),
        "expected_digest": expected,
        "problems": problems,
    }


def _requirement_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    entries = frontier_experiments._entries(record)
    rows = []
    for req in frontier_experiments.EXPERIMENT_REQUIREMENTS:
        if "required_names" in req:
            rows.append(frontier_experiments._require_named(record, entries, req))
        elif req["name"] == "ramcliff_repeats":
            rows.append(frontier_experiments._require_ramcliff_repeats(record, entries, req))
        elif "min_count" in req:
            rows.append(frontier_experiments._require_count(record, entries, req))
        else:
            rows.append(frontier_experiments._require_single(record, entries, req))
    return rows


def _entry_label(row: dict[str, Any]) -> str:
    return frontier_experiments._entry_name(row) or row.get("category") or row.get("name") or "experiment row"


def _trace_problems(record: dict[str, Any]) -> list[str]:
    problems = []
    top_cmds = _commands(record)
    if top_cmds and any(_placeholder(cmd) for cmd in top_cmds):
        problems.append("top-level command contains placeholder text")
    entries = frontier_experiments._entries(record)
    for row in entries:
        ok, _ = frontier_experiments._usable(row, record, allow_na=True)
        if not ok:
            continue
        status = frontier_experiments._status(
            row.get("status") or row.get("verdict") or row.get("coverage_status")
        )
        label = _entry_label(row)
        if status in frontier_experiments.NA_STATUSES:
            reason = frontier_experiments._reason(row, record)
            if not reason or _placeholder(reason):
                problems.append(f"{label}: N/A reason missing or placeholder")
            continue
        row_cmds = _row_commands(row)
        trace_values = [
            row.get("receipt"),
            row.get("artifact"),
            row.get("log"),
            row.get("report"),
            row.get("metrics"),
            *row_cmds,
        ]
        has_trace = any(bool(value) for value in trace_values)
        if not has_trace:
            problems.append(f"{label}: receipt/artifact/metrics/command trace missing")
        for value in trace_values:
            if _placeholder(value):
                problems.append(f"{label}: trace contains placeholder text")
                break
        if row_cmds and any(_placeholder(cmd) for cmd in row_cmds):
            problems.append(f"{label}: command contains placeholder text")
    return problems


def record_status(record: dict[str, Any] | None, *, label: str | None = None,
                  require_signature: bool = True) -> dict[str, Any]:
    if not record:
        return {
            "schema": "hawking.frontier_experiment_receipt_status.v1",
            "ok": False,
            "model": label,
            "problems": ["record missing or unreadable"],
        }
    model_label = str(record.get("model") or record.get("label") or label or "")
    problems = []
    if record.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA}")
    if label and model_label != label:
        problems.append("model/label does not match manifest label")
    if not frontier_by_label(model_label):
        problems.append("model must match a frontier manifest label")
    if record.get("receipt_state") != "final":
        problems.append("receipt_state must be final")
    source = frontier_experiments._status(record.get("source") or record.get("mode"))
    if source not in FINAL_SOURCES:
        problems.append("source/mode must be real or measured")
    if not record.get("machine_class"):
        problems.append("machine_class missing")
    if not (record.get("git_commit") or record.get("hawking_commit")):
        problems.append("git_commit/hawking_commit missing")
    rows = _requirement_rows(record)
    for row in rows:
        problems.extend(f"{row['requirement']}: {problem}" for problem in row["problems"])
    problems.extend(_trace_problems(record))
    sig = signature_status(record)
    if require_signature and not sig["ok"]:
        problems.extend(sig["problems"])
    return {
        "schema": "hawking.frontier_experiment_receipt_status.v1",
        "ok": not problems,
        "model": model_label or label,
        "receipt_state": record.get("receipt_state"),
        "passed_count": sum(1 for row in rows if row["ok"]),
        "required_count": len(rows),
        "signature": sig,
        "problems": problems,
    }


def sign_record(record: dict[str, Any], *, label: str | None = None,
                allow_blocked_draft: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    signed = copy.deepcopy(record)
    signed.pop("signature", None)
    signed.setdefault("generated_at", _now())
    signed.setdefault("git_commit", _git_commit(ROOT))
    signed["signed_at"] = _now()
    status = record_status(signed, label=label, require_signature=False)
    if not status["ok"] and not allow_blocked_draft:
        return signed, status
    signed["signature"] = {"algorithm": SIGN_ALG, "digest": _canonical_digest(signed)}
    return signed, record_status(signed, label=label, require_signature=True)


def draft_record(label: str, *, machine_class: str = "Studio-M1Ultra-128") -> dict[str, Any]:
    record = frontier_experiments._skeleton(label)
    record["receipt_state"] = "draft"
    record["generated_at"] = _now()
    record["git_commit"] = _git_commit(ROOT)
    record["machine_class"] = machine_class
    record["commands"] = ["<exact experiment orchestration command>"]
    return record


def _selected_labels(labels: list[str]) -> list[str]:
    if not labels:
        return [m.label for m in FRONTIER_MODELS]
    out = []
    for label in labels:
        model = frontier_by_label(label)
        if not model:
            raise SystemExit(f"unknown frontier label: {label}")
        out.append(model.label)
    return out


def dispatch(args, root: pathlib.Path = ROOT) -> int:
    rows = []
    ok = True
    for label in _selected_labels(args.label):
        path = frontier_experiments.matrix_path(root, label)
        if getattr(args, "out_dir", ""):
            path = pathlib.Path(args.out_dir) / path.name
        if args.experiment_mode == "draft":
            if path.exists() and not args.force:
                rows.append({"label": label, "path": str(path), "ok": False,
                             "problems": ["path exists; use --force to overwrite"]})
                ok = False
                continue
            record = draft_record(label, machine_class=args.machine_class)
            if args.sign_draft:
                record, status = sign_record(record, label=label, allow_blocked_draft=True)
            else:
                status = record_status(record, label=label, require_signature=False)
            _write_json(path, record)
        elif args.experiment_mode == "sign":
            record = _read_json(path)
            record, status = sign_record(record or {}, label=label,
                                         allow_blocked_draft=args.allow_blocked_draft)
            if _read_json(path):
                _write_json(path, record)
        elif args.experiment_mode == "verify":
            record = _read_json(path)
            status = record_status(record, label=label, require_signature=True)
        else:
            raise SystemExit(f"unknown experiment mode: {args.experiment_mode}")
        rows.append({"label": label, "path": str(path), "ok": status["ok"],
                     "problems": status["problems"]})
        ok = ok and status["ok"]
    result = {
        "schema": "hawking.frontier_experiment_receipt_run.v1",
        "mode": args.experiment_mode,
        "ok": ok,
        "rows": rows,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"# frontier experiment receipts {args.experiment_mode}: {'OK' if ok else 'BLOCKED'}")
        for row in rows:
            print(f"{row['label'][:18]:18s} {'OK' if row['ok'] else 'BLOCK':6s} {row['path']}")
            for problem in row["problems"][:6]:
                print(f"  - {problem}")
    return 0 if ok else 1


def _complete_record(label: str) -> dict[str, Any]:
    record = {
        "schema": SCHEMA,
        "model": label,
        "source": "real",
        "receipt_state": "final",
        "machine_class": "Studio-M1Ultra-128",
        "git_commit": "selftest",
        "experiments": {
            "floor_seeds": [
                {"category": "floor_seed", "seed": seed, "status": "pass", "receipt": f"selftest://seed/{seed}"}
                for seed in (1, 2, 3)
            ],
            "calibration_ablations": [
                {"category": "calibration_ablations", "name": name, "status": "pass", "receipt": f"selftest://calib/{name}"}
                for name in (
                    "domain_matched_calib",
                    "mixed_domain_calib",
                    "awq_alpha_sweep",
                    "residual_depth_sweep",
                )
            ],
            "bpw_ladder": [
                {"category": "bpw_ladder", "bpw": bpw, "status": "pass", "metrics": {"ppl": 1.0 + i}}
                for i, bpw in enumerate((1.50, 1.25, 1.00, 0.75))
            ],
            "moe_expert_ablation": [
                {"category": "moe_expert_ablation", "status": "pass", "receipt": "selftest://expert"}
            ],
            "ramcliff_repeats": [
                {"category": "ramcliff_repeats", "run_type": run_type, "status": "pass", "receipt": f"selftest://ram/{i}"}
                for i, run_type in enumerate(("cold", "cold", "cold", "warm", "warm", "warm"))
            ],
            "baseline_variants": [
                {"category": "baseline_variants", "name": name, "status": "pass", "receipt": f"selftest://baseline/{name}"}
                for name in ("llama_q4", "llama_iq2", "mlx_4bit", "unsloth_or_exl3")
            ],
            "null_certification": [
                {"category": "null_certification", "name": name, "status": "certified", "receipt": f"selftest://null/{name}"}
                for name in ("failed_recipe", "baseline_or_quality_loss")
            ],
            "rebake_or_hash_verify": [
                {"category": "rebake_or_hash_verify", "status": "verified", "receipt": "selftest://rebake"}
            ],
        },
    }
    return record


def selftest() -> bool:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    label = FRONTIER_MODELS[0].label
    draft = draft_record(label)
    draft_signed, draft_status = sign_record(draft, label=label, allow_blocked_draft=True)
    check("signed experiment draft stays blocked", not draft_status["ok"] and draft_signed.get("signature"))
    complete_signed, complete_status = sign_record(_complete_record(label), label=label)
    check("complete experiment matrix signs and verifies", complete_status["ok"])
    complete_signed["experiments"]["floor_seeds"][0]["status"] = "fail"
    check("tampered experiment signature fails", not record_status(complete_signed, label=label)["ok"])
    missing_trace = _complete_record(label)
    missing_trace["experiments"]["floor_seeds"][0].pop("receipt")
    _, missing_status = sign_record(missing_trace, label=label)
    check("experiment row without trace is blocked", not missing_status["ok"])
    missing_required = _complete_record(label)
    missing_required["experiments"]["bpw_ladder"] = missing_required["experiments"]["bpw_ladder"][:2]
    _, required_status = sign_record(missing_required, label=label)
    check("missing required experiment depth is blocked", not required_status["ok"])

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        out_dir = root / "reports" / "condense"
        args = argparse.Namespace(experiment_mode="draft", label=[label], out_dir=str(out_dir),
                                  force=True, sign_draft=True,
                                  machine_class="Studio-M1Ultra-128", json=True)
        check("draft command writes blocked experiment receipt", dispatch(args, root=root) == 1)
        check("draft experiment exists", (out_dir / f"{label}_experiment_matrix.json").exists())
    print(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def cmd_selftest(args) -> int:
    return 0 if selftest() else 1


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Draft/sign/verify signed frontier experiment matrices.")
    sub = ap.add_subparsers(dest="experiment_mode")
    for mode in ("draft", "sign", "verify"):
        p = sub.add_parser(mode, help=f"{mode} signed experiment matrices")
        p.add_argument("label", nargs="*", help="frontier label(s); default all")
        p.add_argument("--out-dir", default="")
        p.add_argument("--json", action="store_true")
        if mode == "draft":
            p.add_argument("--force", action="store_true")
            p.add_argument("--sign-draft", action="store_true")
            p.add_argument("--machine-class", default="Studio-M1Ultra-128")
            p.set_defaults(allow_blocked_draft=False)
        else:
            p.set_defaults(force=False, sign_draft=False, machine_class="Studio-M1Ultra-128")
        if mode == "sign":
            p.add_argument("--allow-blocked-draft", action="store_true")
        else:
            p.set_defaults(allow_blocked_draft=False)
        p.set_defaults(func=dispatch)
    p = sub.add_parser("selftest", help="synthetic signed experiment matrix tests")
    p.set_defaults(func=cmd_selftest)
    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()
    if not args.experiment_mode:
        args = ap.parse_args(["verify"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
