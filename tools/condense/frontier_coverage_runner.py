#!/usr/bin/env python3.12
"""frontier_coverage_runner.py - draft, sign, and verify baseline/eval coverage receipts.

This is the non-compute half of the Studio baseline/eval runner. It does not launch llama.cpp, MLX,
or Hawking evals on the laptop. It creates the exact receipt envelopes those runs must fill, then signs
and verifies completed records so claim bundles cannot rely on loose or stale coverage JSON.
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
import frontier_coverage  # noqa: E402

COND_DIR = pathlib.Path("reports/condense")
SIGN_ALG = "sha256-json-v1"
BASELINE_SCHEMA = "hawking.frontier_baselines.v1"
EVAL_SCHEMA = "hawking.frontier_eval_coverage.v1"
KINDS = ("baseline", "eval")


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


def _schema_kind(record: dict[str, Any]) -> str | None:
    schema = record.get("schema")
    if schema == BASELINE_SCHEMA:
        return "baseline"
    if schema == EVAL_SCHEMA:
        return "eval"
    return None


def default_path(root: pathlib.Path, label: str, kind: str) -> pathlib.Path:
    if kind == "baseline":
        return frontier_coverage.baseline_path(root, label)
    if kind == "eval":
        return frontier_coverage.eval_path(root, label)
    raise ValueError(f"unknown coverage receipt kind: {kind}")


def _entries(record: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    if kind == "baseline":
        return frontier_coverage._baseline_entries(record)
    return frontier_coverage._eval_entries(record)


def _requirements(kind: str) -> tuple[dict[str, Any], ...]:
    return frontier_coverage.BASELINE_REQUIREMENTS if kind == "baseline" else frontier_coverage.EVAL_REQUIREMENTS


def _classify(record: dict[str, Any], entry: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "baseline":
        return frontier_coverage._classify_baseline(record, entry)
    return frontier_coverage._classify_eval(record, entry)


def _entry_name(entry: dict[str, Any]) -> str:
    return frontier_coverage._entry_name(entry)


def _commands(row: dict[str, Any], record: dict[str, Any] | None = None) -> list[str]:
    out = []
    for source in (row, record or {}):
        cmds = source.get("commands")
        if isinstance(cmds, list):
            out.extend(str(cmd) for cmd in cmds if cmd)
        cmd = source.get("command")
        if cmd:
            out.append(str(cmd))
    return out


def _placeholder(s: str) -> bool:
    return "<" in s or "TODO" in s or "..." in s


def _trace_present(entry: dict[str, Any], record: dict[str, Any]) -> bool:
    for key in ("artifact", "receipt", "result_path", "output_path", "log_path"):
        val = entry.get(key) or record.get(key)
        if val and str(val).upper() != "N/A":
            return True
    for key in ("metrics", "results", "scores"):
        val = entry.get(key) or record.get(key)
        if isinstance(val, dict) and val:
            return True
    return False


def coverage_rows(record: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    entries = _entries(record, kind)
    rows = []
    for req in _requirements(kind):
        entry = next((e for e in entries
                      if frontier_coverage._match_requirement(_entry_name(e), req)), None)
        if not entry:
            rows.append({
                "requirement": req["name"],
                "status": "missing",
                "ok": False,
                "problem": "no measured/pass or explicit N/A row",
            })
            continue
        classified = _classify(record, entry, kind)
        rows.append({"requirement": req["name"], "entry": _entry_name(entry), **classified})
    return rows


def _trace_problems(record: dict[str, Any], kind: str) -> list[str]:
    problems = []
    entries = _entries(record, kind)
    for req in _requirements(kind):
        entry = next((e for e in entries
                      if frontier_coverage._match_requirement(_entry_name(e), req)), None)
        if not entry:
            continue
        classified = _classify(record, entry, kind)
        if not classified.get("ok") or classified.get("status") == "na":
            continue
        cmds = _commands(entry, record)
        if not cmds:
            problems.append(f"{req['name']}: exact command(s) missing")
        elif any(_placeholder(cmd) for cmd in cmds):
            problems.append(f"{req['name']}: command contains placeholder text")
        if not _trace_present(entry, record):
            problems.append(f"{req['name']}: receipt/artifact/metrics trace missing")
    return problems


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


def record_status(record: dict[str, Any] | None, *, kind: str | None = None,
                  require_signature: bool = True) -> dict[str, Any]:
    if not record:
        return {"ok": False, "kind": kind, "problems": ["record missing or unreadable"]}
    actual_kind = _schema_kind(record)
    kind = kind or actual_kind
    problems = []
    if kind not in KINDS:
        problems.append("kind must be baseline or eval")
    if actual_kind != kind:
        expected = BASELINE_SCHEMA if kind == "baseline" else EVAL_SCHEMA
        problems.append(f"schema must be {expected}")
    if record.get("receipt_state") != "final":
        problems.append("receipt_state must be final")
    rows = coverage_rows(record, kind) if kind in KINDS else []
    problems.extend(f"{row['requirement']}: {row.get('problem')}" for row in rows if not row.get("ok"))
    problems.extend(_trace_problems(record, kind) if kind in KINDS else [])
    sig = signature_status(record)
    if require_signature and not sig["ok"]:
        problems.extend(sig["problems"])
    return {
        "schema": "hawking.frontier_coverage_receipt_status.v1",
        "kind": kind,
        "model": record.get("model") or record.get("label"),
        "receipt_state": record.get("receipt_state"),
        "ok": not problems,
        "covered": sum(1 for row in rows if row.get("ok")),
        "required": len(rows),
        "rows": rows,
        "signature": sig,
        "problems": problems,
    }


def sign_record(record: dict[str, Any], *, kind: str | None = None,
                allow_blocked_draft: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    signed = copy.deepcopy(record)
    signed.pop("signature", None)
    signed.setdefault("generated_at", _now())
    signed.setdefault("git_commit", _git_commit(ROOT))
    signed["signed_at"] = _now()
    status = record_status(signed, kind=kind, require_signature=False)
    if not status["ok"] and not allow_blocked_draft:
        return signed, status
    signed["signature"] = {"algorithm": SIGN_ALG, "digest": _canonical_digest(signed)}
    return signed, record_status(signed, kind=kind, require_signature=True)


def draft_record(label: str, kind: str, *, machine_class: str = "Studio-M1Ultra-128") -> dict[str, Any]:
    if kind == "baseline":
        record = frontier_coverage._baseline_skeleton(label)
    elif kind == "eval":
        record = frontier_coverage._eval_skeleton(label)
    else:
        raise ValueError(f"unknown coverage receipt kind: {kind}")
    record = copy.deepcopy(record)
    record["generated_at"] = _now()
    record["git_commit"] = _git_commit(ROOT)
    record["machine_class"] = machine_class
    record["receipt_state"] = "draft"
    record["source"] = "operator-draft"
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


def _selected_kinds(kind: str) -> list[str]:
    return list(KINDS) if kind == "both" else [kind]


def dispatch(args, root: pathlib.Path = ROOT) -> int:
    labels = _selected_labels(args.label)
    kinds = _selected_kinds(args.kind)
    rows = []
    ok = True
    for label in labels:
        for kind in kinds:
            path = default_path(root, label, kind)
            if getattr(args, "out_dir", ""):
                path = pathlib.Path(args.out_dir) / path.name
            if args.receipt_mode == "draft":
                if path.exists() and not args.force:
                    rows.append({"label": label, "kind": kind, "path": str(path), "ok": False,
                                 "problems": ["path exists; use --force to overwrite"]})
                    ok = False
                    continue
                record = draft_record(label, kind, machine_class=args.machine_class)
                if args.sign_draft:
                    record, status = sign_record(record, kind=kind, allow_blocked_draft=True)
                else:
                    status = record_status(record, kind=kind, require_signature=False)
                _write_json(path, record)
                rows.append({"label": label, "kind": kind, "path": str(path), "ok": status["ok"],
                             "problems": status["problems"]})
                ok = ok and status["ok"]
            elif args.receipt_mode == "sign":
                record = _read_json(path)
                signed, status = sign_record(record or {}, kind=kind,
                                             allow_blocked_draft=args.allow_blocked_draft)
                if record:
                    _write_json(path, signed)
                rows.append({"label": label, "kind": kind, "path": str(path), "ok": status["ok"],
                             "problems": status["problems"]})
                ok = ok and status["ok"]
            elif args.receipt_mode == "verify":
                record = _read_json(path)
                status = record_status(record, kind=kind, require_signature=True)
                rows.append({"label": label, "kind": kind, "path": str(path), "ok": status["ok"],
                             "problems": status["problems"]})
                ok = ok and status["ok"]
            else:
                raise SystemExit(f"unknown receipt mode: {args.receipt_mode}")
    result = {"schema": "hawking.frontier_coverage_receipt_run.v1", "mode": args.receipt_mode,
              "ok": ok, "rows": rows}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"# frontier coverage receipts {args.receipt_mode}: {'OK' if ok else 'BLOCKED'}")
        for row in rows:
            print(f"{row['label'][:18]:18s} {row['kind']:8s} {'OK' if row['ok'] else 'BLOCK':6s} {row['path']}")
            for problem in row["problems"][:6]:
                print(f"  - {problem}")
    return 0 if ok else 1


def _complete_record(label: str, kind: str) -> dict[str, Any]:
    if kind == "baseline":
        return {
            "schema": BASELINE_SCHEMA,
            "model": label,
            "receipt_state": "final",
            "source": "real",
            "machine_class": "Studio-M1Ultra-128",
            "same_box": True,
            "baselines": [
                {
                    "name": req["name"],
                    "status": "measured" if i < 2 else "na",
                    "same_box": True,
                    "command": f"selftest baseline {i}",
                    "artifact": f"selftest://baseline/{i}" if i < 2 else "N/A",
                    "metrics": {"tok_s": 1.0 + i} if i < 2 else {},
                    "reason": "selftest non-runnable baseline" if i >= 2 else "",
                }
                for i, req in enumerate(frontier_coverage.BASELINE_REQUIREMENTS)
            ],
        }
    return {
        "schema": EVAL_SCHEMA,
        "model": label,
        "receipt_state": "final",
        "mode": "real",
        "machine_class": "Studio-M1Ultra-128",
        "domains": [
            {
                "domain": req["name"],
                "status": "pass",
                "command": f"selftest eval {i}",
                "receipt": f"selftest://eval/{i}",
                "metrics": {"score": 1.0},
            }
            for i, req in enumerate(frontier_coverage.EVAL_REQUIREMENTS)
        ],
    }


def selftest() -> bool:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    model = FRONTIER_MODELS[0]
    draft = draft_record(model.label, "baseline")
    draft_signed, draft_status = sign_record(draft, kind="baseline", allow_blocked_draft=True)
    check("signed draft stays blocked", not draft_status["ok"] and draft_signed.get("signature"))

    complete = _complete_record(model.label, "baseline")
    check("complete unsigned baseline is not verified",
          not record_status(complete, kind="baseline", require_signature=True)["ok"])
    signed, status = sign_record(complete, kind="baseline")
    check("complete baseline signs and verifies", status["ok"])
    signed["baselines"][0]["metrics"]["tok_s"] = 0.0
    check("tampered baseline signature fails", not record_status(signed, kind="baseline")["ok"])

    eval_signed, eval_status = sign_record(_complete_record(model.label, "eval"), kind="eval")
    check("complete eval signs and verifies", eval_status["ok"])

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        out_dir = root / COND_DIR
        args = argparse.Namespace(receipt_mode="draft", label=[model.label], kind="both",
                                  out_dir=str(out_dir), force=True, sign_draft=True,
                                  machine_class="Studio-M1Ultra-128", json=True)
        check("draft command writes blocked receipts", dispatch(args, root=root) == 1)
        for kind in KINDS:
            path = out_dir / default_path(root, model.label, kind).name
            check(f"draft {kind} exists", path.exists())
    print(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def cmd_selftest(args) -> int:
    return 0 if selftest() else 1


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Draft/sign/verify signed frontier coverage receipts.")
    sub = ap.add_subparsers(dest="receipt_mode")
    for mode in ("draft", "sign", "verify"):
        p = sub.add_parser(mode, help=f"{mode} frontier baseline/eval coverage receipts")
        p.add_argument("label", nargs="*", help="frontier label(s); default all")
        p.add_argument("--kind", choices=("baseline", "eval", "both"), default="both")
        p.add_argument("--out-dir", default="")
        p.add_argument("--json", action="store_true")
        if mode == "draft":
            p.add_argument("--force", action="store_true")
            p.add_argument("--sign-draft", action="store_true")
            p.add_argument("--machine-class", default="Studio-M1Ultra-128")
        else:
            p.set_defaults(force=False, sign_draft=False, machine_class="Studio-M1Ultra-128")
        if mode == "sign":
            p.add_argument("--allow-blocked-draft", action="store_true")
        else:
            p.set_defaults(allow_blocked_draft=False)
        p.set_defaults(func=dispatch)
    p = sub.add_parser("selftest", help="synthetic signed coverage receipt tests")
    p.set_defaults(func=cmd_selftest)
    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()
    if not args.receipt_mode:
        args = ap.parse_args(["verify"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
