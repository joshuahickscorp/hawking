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
import re
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
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
CONTRACT_TEXT_KEYS = (
    "run_id",
    "machine_name",
    "same_box_group",
    "environment_receipt",
    "artifact_inventory_receipt",
    "source_provenance_receipt",
)
CONTRACT_SHA_KEYS = (
    "machine_fingerprint_sha256",
    "artifact_inventory_sha256",
    "experiment_plan_sha256",
)


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
    if value is None:
        return True
    text = str(value).strip()
    return not text or "<" in text or "TODO" in text.upper() or "..." in text


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.match(value))


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


def _trace_ref(row: dict[str, Any]) -> Any:
    for key in ("receipt", "artifact", "log", "report", "result_path", "output_path"):
        value = row.get(key)
        if not _placeholder(value) and str(value).upper() != "N/A":
            return value
    return None


def _trace_sha(row: dict[str, Any]) -> Any:
    for key in ("trace_sha256", "receipt_sha256", "artifact_sha256", "log_sha256", "report_sha256"):
        value = row.get(key)
        if _is_sha256(value):
            return value
    return None


def _contract_problems(record: dict[str, Any]) -> list[str]:
    problems = []
    if record.get("same_box") is not True:
        problems.append("same_box must be true for signed experiment matrices")
    for key in CONTRACT_TEXT_KEYS:
        if _placeholder(record.get(key)):
            problems.append(f"{key} missing or placeholder")
    for key in CONTRACT_SHA_KEYS:
        if not _is_sha256(record.get(key)):
            problems.append(f"{key} missing or invalid")
    cmds = _commands(record)
    if not cmds:
        problems.append("top-level experiment command(s) missing")
    elif any(_placeholder(cmd) for cmd in cmds):
        problems.append("top-level command contains placeholder text")
    return problems


def _row_contract_problems(record: dict[str, Any], row: dict[str, Any],
                           label: str, status: str) -> list[str]:
    problems = []
    if row.get("same_box", record.get("same_box")) is not True:
        problems.append(f"{label}: row must be same_box=true")
    row_run_id = row.get("run_id")
    if row_run_id and row_run_id != record.get("run_id"):
        problems.append(f"{label}: row run_id does not match experiment run_id")
    row_machine = row.get("machine_class")
    if row_machine and row_machine != record.get("machine_class"):
        problems.append(f"{label}: row machine_class does not match experiment machine_class")
    row_cmds = _row_commands(row)
    if not row_cmds:
        problems.append(f"{label}: exact row command(s) missing")
    elif any(_placeholder(cmd) for cmd in row_cmds):
        problems.append(f"{label}: command contains placeholder text")
    if not _trace_ref(row):
        problems.append(f"{label}: receipt/artifact/log/report trace missing")
    if not _trace_sha(row):
        problems.append(f"{label}: trace sha256 missing or invalid")
    if "null" in frontier_experiments._status(row.get("category")) or "null" in frontier_experiments._status(row.get("name")):
        reason = frontier_experiments._reason(row, record)
        if _placeholder(reason):
            problems.append(f"{label}: null certification reason missing or placeholder")
    if status in frontier_experiments.NA_STATUSES:
        reason = frontier_experiments._reason(row, record)
        if _placeholder(reason):
            problems.append(f"{label}: N/A reason missing or placeholder")
    return problems


def _trace_problems(record: dict[str, Any]) -> list[str]:
    problems = []
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
        problems.extend(_row_contract_problems(record, row, label, status))
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
    if _placeholder(record.get("machine_class")):
        problems.append("machine_class missing")
    if not (record.get("git_commit") or record.get("hawking_commit")):
        problems.append("git_commit/hawking_commit missing")
    problems.extend(_contract_problems(record))
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
    record.setdefault("same_box", True)
    record.setdefault("run_id", "<same-run id>")
    record.setdefault("machine_name", "<exact Studio host label>")
    record.setdefault("same_box_group", "<same machine/session id shared by experiment matrix>")
    record.setdefault("machine_fingerprint_sha256", "<64 hex>")
    record.setdefault("environment_receipt", "<hawking studio environment-capture receipt>")
    record.setdefault("artifact_inventory_receipt", "<artifact inventory receipt>")
    record.setdefault("artifact_inventory_sha256", "<64 hex>")
    record.setdefault("source_provenance_receipt", "<source provenance receipt>")
    record.setdefault("experiment_plan_sha256", "<64 hex>")
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


def _complete_contract(label: str) -> dict[str, Any]:
    return {
        "source": "real",
        "receipt_state": "final",
        "machine_class": "Studio-M1Ultra-128",
        "machine_name": "selftest-studio",
        "same_box": True,
        "same_box_group": "selftest-experiment-session",
        "machine_fingerprint_sha256": "a" * 64,
        "environment_receipt": "selftest://environment",
        "artifact_inventory_receipt": "selftest://artifact-inventory",
        "artifact_inventory_sha256": "b" * 64,
        "source_provenance_receipt": "selftest://source-provenance",
        "experiment_plan_sha256": "c" * 64,
        "run_id": f"selftest-experiment-{label}",
        "git_commit": "selftest",
        "commands": ["selftest experiment orchestration"],
    }


def _complete_row(category: str, name: str, *, status: str = "pass",
                  extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "category": category,
        "name": name,
        "status": status,
        "same_box": True,
        "command": f"selftest experiment {category} {name}",
        "receipt": f"selftest://experiment/{category}/{name}",
        "trace_sha256": "d" * 64,
    }
    row.update(extra or {})
    return row


def _complete_record(label: str) -> dict[str, Any]:
    contract = _complete_contract(label)
    record = {
        "schema": SCHEMA,
        "model": label,
        **contract,
        "experiments": {
            "floor_seeds": [
                _complete_row("floor_seed", f"seed_{seed}", extra={"seed": seed})
                for seed in (1, 2, 3)
            ],
            "calibration_ablations": [
                _complete_row("calibration_ablations", name)
                for name in (
                    "domain_matched_calib",
                    "mixed_domain_calib",
                    "awq_alpha_sweep",
                    "residual_depth_sweep",
                )
            ],
            "bpw_ladder": [
                _complete_row("bpw_ladder", f"bpw_{bpw}", extra={"bpw": bpw, "metrics": {"ppl": 1.0 + i}})
                for i, bpw in enumerate((1.50, 1.25, 1.00, 0.75))
            ],
            "moe_expert_ablation": [
                _complete_row("moe_expert_ablation", "expert_sensitivity")
            ],
            "ramcliff_repeats": [
                _complete_row("ramcliff_repeats", f"{run_type}_{i}", extra={"run_type": run_type})
                for i, run_type in enumerate(("cold", "cold", "cold", "warm", "warm", "warm"))
            ],
            "baseline_variants": [
                _complete_row("baseline_variants", name)
                for name in ("llama_q4", "llama_iq2", "mlx_4bit", "unsloth_or_exl3")
            ],
            "null_certification": [
                _complete_row(
                    "null_certification",
                    name,
                    status="certified",
                    extra={"reason": f"selftest archived null result: {name}"},
                )
                for name in ("failed_recipe", "baseline_or_quality_loss")
            ],
            "rebake_or_hash_verify": [
                _complete_row("rebake_or_hash_verify", "artifact_rebake", status="verified")
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
    missing_trace_hash = _complete_record(label)
    missing_trace_hash["experiments"]["floor_seeds"][0].pop("trace_sha256")
    _, missing_hash_status = sign_record(missing_trace_hash, label=label)
    check("experiment row without trace hash is blocked", not missing_hash_status["ok"])
    missing_run = _complete_record(label)
    missing_run.pop("run_id")
    _, missing_run_status = sign_record(missing_run, label=label)
    check("experiment matrix without run id is blocked", not missing_run_status["ok"])
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
