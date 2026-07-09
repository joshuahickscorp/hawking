#!/usr/bin/env python3.12
"""frontier_doctor_recovery.py - signed Doctor recovery receipts for Studio claims.

The Doctor is the recovery stack that must make low-bit condensed artifacts usable. This module does
not train or evaluate a model on the laptop. It defines the signed evidence envelope a Studio-scale
Doctor run must fill before a public frontier claim can cite recovery.
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

from studio_manifest import FRONTIER_MODELS, FrontierModel, frontier_by_label  # noqa: E402

COND_DIR = pathlib.Path("reports/condense")
SCHEMA = "hawking.frontier_doctor_recovery.v1"
SIGN_ALG = "sha256-json-v1"
MIN_PARAMS_B = 7.0
REQUIRED_GATES = (
    "studio_scale_7b_plus",
    "recovery_improves_over_ptq",
    "below_dense_limit",
    "heldout_no_task_collapse",
)
REQUIRED_RECEIPTS = (
    "ptq_receipt",
    "recovered_receipt",
    "heldout_eval_receipt",
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


def _safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label)


def recovery_path(root: pathlib.Path, label: str) -> pathlib.Path:
    return root / COND_DIR / f"{_safe_label(label)}_doctor_recovery.json"


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
    s = str(value)
    return not s.strip() or "<" in s or "TODO" in s or "..." in s


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-fA-F]{64}", value))


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _positive_number(value: Any) -> bool:
    return _number(value) and value > 0


def _commands(record: dict[str, Any]) -> list[str]:
    out = []
    cmds = record.get("commands")
    if isinstance(cmds, list):
        out.extend(str(cmd) for cmd in cmds if cmd)
    if record.get("command"):
        out.append(str(record["command"]))
    return out


def _model_for_record(record: dict[str, Any], model: FrontierModel | None = None) -> FrontierModel | None:
    if model:
        return model
    label = record.get("model") or record.get("label")
    hf_id = record.get("hf_id")
    if label:
        found = frontier_by_label(str(label))
        if found:
            return found
    if hf_id:
        return frontier_by_label(str(hf_id))
    return None


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


def _strict_problems(record: dict[str, Any], model: FrontierModel | None) -> list[str]:
    problems = []
    if record.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA}")
    if not model:
        problems.append("model label/hf_id must match a frontier manifest label")
    else:
        if (record.get("model") or record.get("label")) != model.label:
            problems.append(f"model/label must be {model.label}")
        if record.get("hf_id") != model.hf_id:
            problems.append(f"hf_id must be {model.hf_id}")
    if record.get("receipt_state") != "final":
        problems.append("receipt_state must be final")
    if str(record.get("source") or record.get("mode") or "").lower() != "measured":
        problems.append("source/mode must be measured")
    if record.get("status") != "pass":
        problems.append("status must be pass")
    if not record.get("machine_class"):
        problems.append("machine_class missing")
    if not (record.get("git_commit") or record.get("hawking_commit")):
        problems.append("git_commit/hawking_commit missing")

    params_b = record.get("params_b") or record.get("total_b")
    try:
        params_b_float = float(params_b)
    except (TypeError, ValueError):
        params_b_float = 0.0
    if params_b_float < MIN_PARAMS_B:
        problems.append(f"params_b/total_b must be >= {MIN_PARAMS_B:g}")

    chain = record.get("doctor_chain") or record.get("recovery_chain") or record.get("method")
    if _placeholder(chain):
        problems.append("doctor_chain/recovery_chain/method missing or placeholder")

    gate = record.get("gate") if isinstance(record.get("gate"), dict) else {}
    for key in REQUIRED_GATES:
        if gate.get(key) is not True:
            problems.append(f"gate.{key} must be true")

    for key in REQUIRED_RECEIPTS:
        if _placeholder(record.get(key)):
            problems.append(f"{key} missing or placeholder")
    artifact_hash = record.get("recovered_artifact_sha256") or record.get("artifact_sha256")
    if not _hex64(artifact_hash):
        problems.append("recovered_artifact_sha256/artifact_sha256 is missing or invalid")

    ptq_deg = record.get("ptq_degradation_pct")
    rec_deg = record.get("recovered_degradation_pct")
    if not _number(ptq_deg):
        problems.append("ptq_degradation_pct must be numeric")
    if not _number(rec_deg):
        problems.append("recovered_degradation_pct must be numeric")
    if _number(ptq_deg) and _number(rec_deg) and rec_deg >= ptq_deg:
        problems.append("recovered_degradation_pct must be lower than ptq_degradation_pct")

    dense_limit = record.get("dense_limit_degradation_pct")
    if not _number(dense_limit):
        problems.append("dense_limit_degradation_pct must be numeric")
    if _number(rec_deg) and _number(dense_limit) and rec_deg > dense_limit:
        problems.append("recovered_degradation_pct must be <= dense_limit_degradation_pct")

    heldout = record.get("heldout_task_score_ratio")
    if not _positive_number(heldout):
        problems.append("heldout_task_score_ratio must be positive")
    elif heldout < 0.95:
        problems.append("heldout_task_score_ratio must be >= 0.95")

    commands = _commands(record)
    if not commands:
        problems.append("exact command(s) missing")
    elif any(_placeholder(cmd) for cmd in commands):
        problems.append("command contains placeholder text")
    return problems


def record_status(record: dict[str, Any] | None, *, model: FrontierModel | None = None,
                  require_signature: bool = True) -> dict[str, Any]:
    if not record:
        return {
            "schema": "hawking.frontier_doctor_recovery_status.v1",
            "ok": False,
            "label": model.label if model else None,
            "problems": ["record missing or unreadable"],
        }
    model = _model_for_record(record, model)
    problems = _strict_problems(record, model)
    sig = signature_status(record)
    if require_signature and not sig["ok"]:
        problems.extend(sig["problems"])
    return {
        "schema": "hawking.frontier_doctor_recovery_status.v1",
        "ok": not problems,
        "label": model.label if model else record.get("model") or record.get("label"),
        "hf_id": model.hf_id if model else record.get("hf_id"),
        "receipt_state": record.get("receipt_state"),
        "params_b": record.get("params_b") or record.get("total_b"),
        "signature": sig,
        "problems": problems,
    }


def sign_record(record: dict[str, Any], *, model: FrontierModel | None = None,
                allow_blocked_draft: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    signed = copy.deepcopy(record)
    signed.pop("signature", None)
    signed.setdefault("generated_at", _now())
    signed.setdefault("git_commit", _git_commit(ROOT))
    signed["signed_at"] = _now()
    status = record_status(signed, model=model, require_signature=False)
    if not status["ok"] and not allow_blocked_draft:
        return signed, status
    signed["signature"] = {"algorithm": SIGN_ALG, "digest": _canonical_digest(signed)}
    return signed, record_status(signed, model=model, require_signature=True)


def draft_record(model: FrontierModel, *, machine_class: str = "Studio-M1Ultra-128") -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "model": model.label,
        "hf_id": model.hf_id,
        "generated_at": _now(),
        "git_commit": _git_commit(ROOT),
        "machine_class": machine_class,
        "receipt_state": "draft",
        "source": "TODO measured",
        "status": "TODO pass",
        "params_b": f"TODO >= {MIN_PARAMS_B:g}",
        "doctor_chain": "<AWQ/residual/full-rank/codec-native/LoRA-KD chain>",
        "ptq_degradation_pct": "TODO numeric",
        "recovered_degradation_pct": "TODO numeric and lower than PTQ",
        "dense_limit_degradation_pct": "TODO numeric",
        "heldout_task_score_ratio": "TODO >= 0.95",
        "ptq_receipt": "<path>",
        "recovered_receipt": "<path>",
        "heldout_eval_receipt": "<path>",
        "recovered_artifact_sha256": "<64 hex>",
        "commands": ["<exact doctor/recovery command>", "<exact heldout eval command>"],
        "gate": {key: "TODO true" for key in REQUIRED_GATES},
    }


def complete_record(model: FrontierModel) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "model": model.label,
        "hf_id": model.hf_id,
        "generated_at": _now(),
        "git_commit": "selftest",
        "machine_class": "Studio-M1Ultra-128",
        "receipt_state": "final",
        "source": "measured",
        "status": "pass",
        "params_b": max(model.total_b, MIN_PARAMS_B),
        "doctor_chain": "selftest AWQ -> codec-native recovery -> heldout eval",
        "ptq_degradation_pct": 12.0,
        "recovered_degradation_pct": 4.0,
        "dense_limit_degradation_pct": 8.0,
        "heldout_task_score_ratio": 0.99,
        "ptq_receipt": "selftest://ptq",
        "recovered_receipt": "selftest://recovered",
        "heldout_eval_receipt": "selftest://heldout",
        "recovered_artifact_sha256": "a" * 64,
        "commands": ["selftest doctor recovery", "selftest heldout eval"],
        "gate": {key: True for key in REQUIRED_GATES},
    }


def _selected_models(labels: list[str]) -> list[FrontierModel]:
    if not labels:
        return list(FRONTIER_MODELS)
    out = []
    for label in labels:
        model = frontier_by_label(label)
        if not model:
            raise SystemExit(f"unknown frontier label: {label}")
        out.append(model)
    return out


def recovery_status(root: pathlib.Path, label: str) -> dict[str, Any]:
    model = frontier_by_label(label)
    path = recovery_path(root, label)
    record = _read_json(path)
    status = record_status(record, model=model, require_signature=True)
    status["path"] = str(path)
    status["exists"] = record is not None
    return status


def recovery_rollup(root: pathlib.Path, labels: list[str]) -> dict[str, Any]:
    rows = [recovery_status(root, label) for label in labels]
    blocked = [row.get("label") or label for row, label in zip(rows, labels) if not row["ok"]]
    return {
        "schema": "hawking.frontier_doctor_recovery_rollup.v1",
        "model_count": len(labels),
        "passed_count": len(labels) - len(blocked),
        "blocked_count": len(blocked),
        "blocked_labels": blocked,
        "rows": rows,
        "ok": not blocked,
    }


def recovery_plan(root: pathlib.Path, labels: list[str]) -> dict[str, Any]:
    models = _selected_models(labels)
    return {
        "schema": "hawking.frontier_doctor_recovery_plan.v1",
        "generated_at": _now(),
        "model_count": len(models),
        "requirements": {
            "schema": SCHEMA,
            "min_params_b": MIN_PARAMS_B,
            "required_gates": list(REQUIRED_GATES),
            "required_receipts": list(REQUIRED_RECEIPTS),
            "signature": SIGN_ALG,
        },
        "labels": [
            {
                "label": model.label,
                "hf_id": model.hf_id,
                "params_b": model.total_b,
                "path": str(recovery_path(root, model.label)),
                "command_template": (
                    f"hawking studio doctor-recovery-receipt draft {model.label} --sign-draft --force"
                ),
                "skeleton": draft_record(model),
            }
            for model in models
        ],
        "rollup": recovery_rollup(root, [m.label for m in models]),
    }


def dispatch(args, root: pathlib.Path = ROOT) -> int:
    rows = []
    ok = True
    for model in _selected_models(args.label):
        path = recovery_path(root, model.label)
        if getattr(args, "out_dir", ""):
            path = pathlib.Path(args.out_dir) / path.name
        if args.recovery_mode == "draft":
            if path.exists() and not args.force:
                rows.append({"label": model.label, "path": str(path), "ok": False,
                             "problems": ["path exists; use --force to overwrite"]})
                ok = False
                continue
            record = draft_record(model, machine_class=args.machine_class)
            if args.sign_draft:
                record, status = sign_record(record, model=model, allow_blocked_draft=True)
            else:
                status = record_status(record, model=model, require_signature=False)
            _write_json(path, record)
        elif args.recovery_mode == "sign":
            record = _read_json(path)
            record, status = sign_record(record or {}, model=model,
                                         allow_blocked_draft=args.allow_blocked_draft)
            if _read_json(path):
                _write_json(path, record)
        elif args.recovery_mode == "verify":
            record = _read_json(path)
            status = record_status(record, model=model, require_signature=True)
        else:
            raise SystemExit(f"unknown Doctor recovery mode: {args.recovery_mode}")
        rows.append({"label": model.label, "path": str(path), "ok": status["ok"],
                     "problems": status["problems"]})
        ok = ok and status["ok"]
    result = {
        "schema": "hawking.frontier_doctor_recovery_run.v1",
        "mode": args.recovery_mode,
        "ok": ok,
        "rows": rows,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"# frontier Doctor recovery receipts {args.recovery_mode}: {'OK' if ok else 'BLOCKED'}")
        for row in rows:
            print(f"{row['label'][:18]:18s} {'OK' if row['ok'] else 'BLOCK':6s} {row['path']}")
            for problem in row["problems"][:6]:
                print(f"  - {problem}")
    return 0 if ok else 1


def selftest() -> bool:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    model = FRONTIER_MODELS[0]
    draft = draft_record(model)
    draft_signed, draft_status = sign_record(draft, model=model, allow_blocked_draft=True)
    check("signed Doctor draft stays blocked", not draft_status["ok"] and draft_signed.get("signature"))
    complete_signed, complete_status = sign_record(complete_record(model), model=model)
    check("complete Doctor recovery signs and verifies", complete_status["ok"])
    complete_signed["recovered_degradation_pct"] = 20.0
    check("tampered Doctor signature fails", not record_status(complete_signed, model=model)["ok"])
    weak = complete_record(model)
    weak["heldout_task_score_ratio"] = 0.7
    _, weak_status = sign_record(weak, model=model)
    check("Doctor task collapse is blocked", not weak_status["ok"])
    missing = complete_record(model)
    missing.pop("heldout_eval_receipt")
    _, missing_status = sign_record(missing, model=model)
    check("Doctor receipt without heldout trace is blocked", not missing_status["ok"])

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        out_dir = root / "reports" / "condense"
        args = argparse.Namespace(recovery_mode="draft", label=[model.label], out_dir=str(out_dir),
                                  force=True, sign_draft=True,
                                  machine_class="Studio-M1Ultra-128", json=True)
        check("draft command writes blocked Doctor receipt", dispatch(args, root=root) == 1)
        check("draft Doctor receipt exists", (out_dir / f"{model.label}_doctor_recovery.json").exists())
    print(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def cmd_selftest(args) -> int:
    return 0 if selftest() else 1


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Draft/sign/verify signed frontier Doctor recovery receipts.")
    sub = ap.add_subparsers(dest="recovery_mode")
    p = sub.add_parser("plan", help="print Doctor recovery receipt requirements")
    p.add_argument("label", nargs="*", help="frontier label(s); default all")
    p.add_argument("--out", default="")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_plan)
    for mode in ("draft", "sign", "verify"):
        p = sub.add_parser(mode, help=f"{mode} signed Doctor recovery receipts")
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
    p = sub.add_parser("selftest", help="synthetic signed Doctor recovery receipt tests")
    p.set_defaults(func=cmd_selftest)
    return ap


def cmd_plan(args) -> int:
    data = recovery_plan(ROOT, args.label)
    if args.out:
        _write_json(pathlib.Path(args.out), data)
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0 if data["rollup"]["ok"] else 1
    print("# frontier Doctor recovery plan")
    print(f"# receipts {data['rollup']['passed_count']}/{data['rollup']['model_count']}")
    for row in data["labels"]:
        print(f"{row['label']}: {row['path']}")
    return 0 if data["rollup"]["ok"] else 1


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()
    if not args.recovery_mode:
        args = ap.parse_args(["plan"])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
