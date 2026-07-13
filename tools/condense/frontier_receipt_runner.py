#!/usr/bin/env python3.12
"""frontier_receipt_runner.py - draft, sign, and verify native serve/RAM-cliff receipts.

This is the receipt integrity layer above `frontier_receipts.py`. It does not serve a model or run a
RAM-cliff benchmark on the laptop. It creates signed but blocked envelopes, and signs final measured
records only after the strict native-serve/RAM-cliff validators pass. Claim bundles can then reject
unsigned or tampered performance evidence before a public win is even considered.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "tools" / "condense"))

from studio_manifest import FRONTIER_MODELS, frontier_by_label  # noqa: E402
from frontier_common import (  # noqa: E402
    commands as _commands,
    git_commit as _git_commit,
    now_utc as _now,
    placeholder as _placeholder,
    read_json as _read_json,
    sign_record as _sign_record,
    signature_status,
    write_json as _write_json,
)
import frontier_receipts  # noqa: E402

SERVE_SCHEMA = "hawking.frontier_serve.v1"
RAMCLIFF_SCHEMA = "hawking.frontier_ramcliff.v1"
KINDS = ("serve", "ramcliff")


def _schema_kind(record: dict[str, Any]) -> str | None:
    schema = record.get("schema")
    if schema == SERVE_SCHEMA:
        return "serve"
    if schema == RAMCLIFF_SCHEMA:
        return "ramcliff"
    return None


def default_path(root: pathlib.Path, label: str, kind: str) -> pathlib.Path:
    if kind == "serve":
        return frontier_receipts.serve_path(root, label)
    if kind == "ramcliff":
        return frontier_receipts.ramcliff_path(root, label)
    raise ValueError(f"unknown receipt kind: {kind}")


def _label(record: dict[str, Any]) -> str:
    return str(record.get("model") or record.get("label") or "")


def _trace_problems(record: dict[str, Any], kind: str) -> list[str]:
    problems = []
    cmds = _commands(record)
    if not cmds:
        problems.append("exact command(s) missing")
    elif any(_placeholder(cmd) for cmd in cmds):
        problems.append("command contains placeholder text")
    if kind == "serve":
        if _placeholder(record.get("served_forward_receipt")):
            problems.append("served_forward_receipt missing or placeholder")
        if _placeholder(record.get("parity_receipt")):
            problems.append("parity_receipt missing or placeholder")
        if _placeholder(record.get("load_receipt")):
            problems.append("load_receipt missing or placeholder")
    if kind == "ramcliff":
        if not record.get("powermetrics_receipt") and not record.get("energy_receipt"):
            problems.append("powermetrics_receipt or energy_receipt missing")
        if not record.get("baseline_receipt"):
            problems.append("baseline_receipt for swapping/Q4_K comparison missing")
    return problems

def _strict_status(record: dict[str, Any], kind: str) -> dict[str, Any]:
    label = _label(record)
    if kind == "serve":
        path = pathlib.Path(f"<{label}_serve.json>")
        problems, status = frontier_receipts._base_status(label, path, record, SERVE_SCHEMA)
        required_true = ("native_tq", "tq_strict", "all_linear", "gpu_bitslice", "served_forward_pass")
        for key in required_true:
            if record.get(key) is not True:
                problems.append(f"{key} must be true")
        if record.get("rehydrate_f16") is not False:
            problems.append("rehydrate_f16 must be false")
        if record.get("status") != "pass":
            problems.append("status must be pass")
        if not frontier_receipts._positive_number(record.get("tok_s")):
            problems.append("tok_s must be positive")
        if record.get("parity_pass") is not True:
            problems.append("parity_pass must be true")
        if not frontier_receipts._positive_number(record.get("memory_peak_gb")):
            problems.append("memory_peak_gb must be positive")
        if not frontier_receipts._positive_number(record.get("memory_resident_gb")):
            problems.append("memory_resident_gb must be positive")
        if not frontier_receipts._positive_number(record.get("unified_memory_gb")):
            problems.append("unified_memory_gb must be positive")
        if record.get("resident_memory_ok") is not True:
            problems.append("resident_memory_ok must be true")
        if (frontier_receipts._positive_number(record.get("memory_peak_gb"))
                and frontier_receipts._positive_number(record.get("unified_memory_gb"))
                and record["memory_peak_gb"] > record["unified_memory_gb"]):
            problems.append("memory_peak_gb must fit within unified_memory_gb")
        status.update({"ok": not problems, "problems": problems})
        return status
    path = pathlib.Path(f"<{label}_ramcliff.json>")
    problems, status = frontier_receipts._base_status(label, path, record, RAMCLIFF_SCHEMA)
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
    if not frontier_receipts._positive_number(record.get("tok_s_resident")):
        problems.append("tok_s_resident must be positive")
    if not frontier_receipts._positive_number(record.get("tok_s_swapping")):
        problems.append("tok_s_swapping must be positive")
    if not frontier_receipts._positive_number(record.get("j_per_tok_resident")):
        problems.append("j_per_tok_resident must be positive")
    if not frontier_receipts._positive_number(record.get("j_per_tok_swapping")):
        problems.append("j_per_tok_swapping must be positive")
    cliff_x = record.get("cliff_x")
    if not frontier_receipts._positive_number(cliff_x) or cliff_x <= frontier_receipts.CLIFF_X_GATE:
        problems.append(f"cliff_x must be > {frontier_receipts.CLIFF_X_GATE}")
    if (frontier_receipts._positive_number(record.get("j_per_tok_resident"))
            and frontier_receipts._positive_number(record.get("j_per_tok_swapping"))
            and record["j_per_tok_resident"] >= record["j_per_tok_swapping"]):
        problems.append("resident J/tok must be lower than swapping J/tok")
    status.update({"ok": not problems, "problems": problems})
    return status


def record_status(record: dict[str, Any] | None, *, kind: str | None = None,
                  require_signature: bool = True) -> dict[str, Any]:
    if not record:
        return {"ok": False, "kind": kind, "problems": ["record missing or unreadable"]}
    actual_kind = _schema_kind(record)
    kind = kind or actual_kind
    problems = []
    if kind not in KINDS:
        problems.append("kind must be serve or ramcliff")
    if actual_kind != kind:
        expected = SERVE_SCHEMA if kind == "serve" else RAMCLIFF_SCHEMA
        problems.append(f"schema must be {expected}")
    if record.get("receipt_state") != "final":
        problems.append("receipt_state must be final")
    strict = _strict_status(record, kind) if kind in KINDS else {"problems": []}
    problems.extend(strict.get("problems", []))
    problems.extend(_trace_problems(record, kind) if kind in KINDS else [])
    sig = signature_status(record)
    if require_signature and not sig["ok"]:
        problems.extend(sig["problems"])
    return {
        "schema": "hawking.frontier_native_receipt_status.v1",
        "kind": kind,
        "model": _label(record),
        "receipt_state": record.get("receipt_state"),
        "ok": not problems,
        "strict_ok": strict.get("ok", False),
        "signature": sig,
        "problems": problems,
    }


def sign_record(record: dict[str, Any], *, kind: str | None = None,
                allow_blocked_draft: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    return _sign_record(
        record,
        record_status,
        root=ROOT,
        status_kwargs={"kind": kind},
        allow_blocked_draft=allow_blocked_draft,
    )


def draft_record(label: str, kind: str, *, machine_class: str = "Studio-M3Ultra-96") -> dict[str, Any]:
    common = {
        "model": label,
        "generated_at": _now(),
        "git_commit": _git_commit(ROOT),
        "machine_class": machine_class,
        "receipt_state": "draft",
        "artifact_sha256": "<64 hex>",
        "commands": ["<exact command>"],
    }
    if kind == "serve":
        return {
            **common,
            "schema": SERVE_SCHEMA,
            "status": "TODO pass",
            "native_tq": "TODO true",
            "rehydrate_f16": "TODO false",
            "tq_strict": "TODO true",
            "all_linear": "TODO true",
            "gpu_bitslice": "TODO true",
            "served_forward_pass": "TODO true",
            "parity_pass": "TODO true",
            "tok_s": "TODO >0",
            "memory_peak_gb": "TODO >0",
            "memory_resident_gb": "TODO >0",
            "unified_memory_gb": "TODO >0",
            "resident_memory_ok": "TODO true",
            "load_receipt": "<path>",
            "served_forward_receipt": "<path>",
            "parity_receipt": "<path>",
        }
    if kind == "ramcliff":
        return {
            **common,
            "schema": RAMCLIFF_SCHEMA,
            "source": "TODO measured",
            "verdict": "TODO CLIFF-WIN",
            "served_native_tq": "TODO true",
            "tok_s_resident": "TODO >0",
            "tok_s_swapping": "TODO >0",
            "j_per_tok_resident": "TODO >0 and lower than swapping",
            "j_per_tok_swapping": "TODO >0",
            "cliff_x": f"TODO >{frontier_receipts.CLIFF_X_GATE}",
            "powermetrics_receipt": "<path>",
            "baseline_receipt": "<path>",
            "gate": {
                "condensed_resident": "TODO true",
                "served_native_tq": "TODO true",
                "q4k_overflows_box": "TODO true",
                "cliff_x_over_gate": "TODO true",
                "resident_lower_energy": "TODO true",
            },
        }
    raise ValueError(f"unknown receipt kind: {kind}")


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
            elif args.receipt_mode == "sign":
                record = _read_json(path)
                record, status = sign_record(record or {}, kind=kind,
                                             allow_blocked_draft=args.allow_blocked_draft)
                if _read_json(path):
                    _write_json(path, record)
            elif args.receipt_mode == "verify":
                record = _read_json(path)
                status = record_status(record, kind=kind, require_signature=True)
            else:
                raise SystemExit(f"unknown receipt mode: {args.receipt_mode}")
            rows.append({"label": label, "kind": kind, "path": str(path), "ok": status["ok"],
                         "problems": status["problems"]})
            ok = ok and status["ok"]
    result = {"schema": "hawking.frontier_native_receipt_run.v1", "mode": args.receipt_mode,
              "ok": ok, "rows": rows}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"# frontier native receipts {args.receipt_mode}: {'OK' if ok else 'BLOCKED'}")
        for row in rows:
            print(f"{row['label'][:18]:18s} {row['kind']:8s} {'OK' if row['ok'] else 'BLOCK':6s} {row['path']}")
            for problem in row["problems"][:6]:
                print(f"  - {problem}")
    return 0 if ok else 1


def _complete_record(label: str, kind: str) -> dict[str, Any]:
    if kind == "serve":
        return {
            "schema": SERVE_SCHEMA,
            "model": label,
            "receipt_state": "final",
            "source": "measured",
            "machine_class": "Studio-M3Ultra-96",
            "status": "pass",
            "native_tq": True,
            "rehydrate_f16": False,
            "tq_strict": True,
            "all_linear": True,
            "gpu_bitslice": True,
            "served_forward_pass": True,
            "parity_pass": True,
            "tok_s": 1.0,
            "memory_peak_gb": 4.0,
            "memory_resident_gb": 3.5,
            "unified_memory_gb": 96.0,
            "resident_memory_ok": True,
            "artifact_sha256": "a" * 64,
            "commands": ["selftest serve"],
            "git_commit": "selftest",
            "load_receipt": "selftest://load",
            "served_forward_receipt": "selftest://served-forward",
            "parity_receipt": "selftest://parity",
        }
    return {
        "schema": RAMCLIFF_SCHEMA,
        "model": label,
        "receipt_state": "final",
        "source": "measured",
        "machine_class": "Studio-M3Ultra-96",
        "verdict": "CLIFF-WIN",
        "served_native_tq": True,
        "tok_s_resident": 20.0,
        "tok_s_swapping": 1.0,
        "j_per_tok_resident": 1.0,
        "j_per_tok_swapping": 3.0,
        "cliff_x": 20.0,
        "artifact_sha256": "b" * 64,
        "commands": ["selftest ramcliff"],
        "git_commit": "selftest",
        "powermetrics_receipt": "selftest://powermetrics",
        "baseline_receipt": "selftest://q4k-swap",
        "gate": {
            "condensed_resident": True,
            "served_native_tq": True,
            "q4k_overflows_box": True,
            "cliff_x_over_gate": True,
            "resident_lower_energy": True,
        },
    }


def selftest() -> bool:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    label = FRONTIER_MODELS[0].label
    draft = draft_record(label, "serve")
    draft_signed, draft_status = sign_record(draft, kind="serve", allow_blocked_draft=True)
    check("signed serve draft stays blocked", not draft_status["ok"] and draft_signed.get("signature"))
    serve_signed, serve_status = sign_record(_complete_record(label, "serve"), kind="serve")
    check("complete serve signs and verifies", serve_status["ok"])
    serve_signed["tok_s"] = 0.0
    check("tampered serve signature fails", not record_status(serve_signed, kind="serve")["ok"])
    missing_memory = _complete_record(label, "serve")
    missing_memory.pop("memory_peak_gb")
    _, missing_memory_status = sign_record(missing_memory, kind="serve")
    check("serve without memory proof is blocked", not missing_memory_status["ok"])
    cliff_signed, cliff_status = sign_record(_complete_record(label, "ramcliff"), kind="ramcliff")
    check("complete RAM-cliff signs and verifies", cliff_status["ok"])
    missing_energy = _complete_record(label, "ramcliff")
    missing_energy.pop("powermetrics_receipt")
    _, missing_status = sign_record(missing_energy, kind="ramcliff")
    check("RAM-cliff without energy trace is blocked", not missing_status["ok"])

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        out_dir = root / "reports" / "condense"
        args = argparse.Namespace(receipt_mode="draft", label=[label], kind="both",
                                  out_dir=str(out_dir), force=True, sign_draft=True,
                                  machine_class="Studio-M3Ultra-96", json=True)
        check("draft command writes blocked native receipts", dispatch(args, root=root) == 1)
        for kind in KINDS:
            path = out_dir / default_path(root, label, kind).name
            check(f"draft {kind} exists", path.exists())
    print(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def cmd_selftest(args) -> int:
    return 0 if selftest() else 1


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Draft/sign/verify signed frontier native serve/RAM-cliff receipts.")
    sub = ap.add_subparsers(dest="receipt_mode")
    for mode in ("draft", "sign", "verify"):
        p = sub.add_parser(mode, help=f"{mode} signed native serve/RAM-cliff receipts")
        p.add_argument("label", nargs="*", help="frontier label(s); default all")
        p.add_argument("--kind", choices=("serve", "ramcliff", "both"), default="both")
        p.add_argument("--out-dir", default="")
        p.add_argument("--json", action="store_true")
        if mode == "draft":
            p.add_argument("--force", action="store_true")
            p.add_argument("--sign-draft", action="store_true")
            p.add_argument("--machine-class", default="Studio-M3Ultra-96")
            p.set_defaults(allow_blocked_draft=False)
        else:
            p.set_defaults(force=False, sign_draft=False, machine_class="Studio-M3Ultra-96")
        if mode == "sign":
            p.add_argument("--allow-blocked-draft", action="store_true")
        else:
            p.set_defaults(allow_blocked_draft=False)
        p.set_defaults(func=dispatch)
    p = sub.add_parser("selftest", help="synthetic signed native receipt tests")
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
