#!/usr/bin/env python3.12
"""frontier_provenance.py - signed source-provenance receipts for frontier checkpoints.

This is a non-compute guard for the Studio run. It does not download models. It records and verifies
where a source checkpoint came from, which exact Hub revision/file manifest was used, and whether the
source is a bf16 parent or a pre-quantized/compressed checkpoint. Claim bundles can then reject wins
whose `.tq` artifact is backed only by a vague model name.
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
SCHEMA = "hawking.frontier_source_provenance.v1"
SIGN_ALG = "sha256-json-v1"


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


def provenance_path(root: pathlib.Path, label: str) -> pathlib.Path:
    return root / COND_DIR / f"{_safe_label(label)}_source_provenance.json"


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


def _commands(record: dict[str, Any]) -> list[str]:
    out = []
    if isinstance(record.get("commands"), list):
        out.extend(str(cmd) for cmd in record["commands"] if cmd)
    if record.get("command"):
        out.append(str(record["command"]))
    if record.get("procurement_command"):
        out.append(str(record["procurement_command"]))
    return out


def _model(label: str | None, hf_id: str | None = None) -> FrontierModel | None:
    if label:
        found = frontier_by_label(label)
        if found:
            return found
    if hf_id:
        return frontier_by_label(hf_id)
    return None


def _has_manifest(record: dict[str, Any]) -> bool:
    files = record.get("files")
    if isinstance(files, list) and files:
        for row in files:
            if not isinstance(row, dict):
                return False
            if _placeholder(row.get("path")):
                return False
            try:
                if int(row.get("bytes", 0)) <= 0:
                    return False
            except (TypeError, ValueError):
                return False
            digest = row.get("sha256")
            etag = row.get("etag")
            if not (_hex64(digest) or (isinstance(etag, str) and etag.strip())):
                return False
        return True
    if _hex64(record.get("file_manifest_sha256")):
        try:
            return int(record.get("file_count", 0)) > 0 and int(record.get("total_bytes", 0)) > 0
        except (TypeError, ValueError):
            return False
    return False


def _format_problems(record: dict[str, Any], model: FrontierModel) -> list[str]:
    problems = []
    kind = model.source_kind.lower()
    source_format = str(record.get("source_format") or record.get("precision") or "").lower()
    prequantized = record.get("source_is_prequantized")
    needs_compressed = any(token in kind for token in ("compressed", "fp4", "fp8", "int4"))
    if needs_compressed:
        if prequantized is not True:
            problems.append("source_is_prequantized must be true for compressed/FP checkpoints")
        if _placeholder(source_format):
            problems.append("source_format must name the compressed/FP source format")
        if not record.get("format_receipt") and not record.get("source_format_receipt"):
            problems.append("format_receipt or source_format_receipt missing for compressed source")
    elif "bf16" in kind:
        if prequantized is not False:
            problems.append("source_is_prequantized must be false for bf16 parents")
        if source_format not in ("bf16", "bfloat16"):
            problems.append("source_format/precision must be bf16 for bf16 parents")
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


def record_status(record: dict[str, Any] | None, *, model: FrontierModel | None = None,
                  require_signature: bool = True) -> dict[str, Any]:
    if not record:
        return {"ok": False, "label": None, "problems": ["record missing or unreadable"]}
    model = model or _model(record.get("label") or record.get("model"), record.get("hf_id"))
    problems = []
    if record.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA}")
    if not model:
        problems.append("model label/hf_id is not in the frontier manifest")
    else:
        if record.get("label") != model.label and record.get("model") != model.label:
            problems.append(f"label/model must be {model.label}")
        if record.get("hf_id") != model.hf_id:
            problems.append(f"hf_id must be {model.hf_id}")
        if record.get("source_kind") != model.source_kind:
            problems.append(f"source_kind must be {model.source_kind}")
    if record.get("receipt_state") != "final":
        problems.append("receipt_state must be final")
    if record.get("source") != "hf-hub":
        problems.append("source must be hf-hub")
    for key in ("revision", "model_card_url", "download_receipt"):
        if _placeholder(record.get(key)):
            problems.append(f"{key} missing or placeholder")
    cmds = _commands(record)
    if not cmds:
        problems.append("exact procurement command(s) missing")
    elif any(_placeholder(cmd) for cmd in cmds):
        problems.append("procurement command contains placeholder text")
    if not _has_manifest(record):
        problems.append("file manifest evidence missing; provide files[] with hashes/etags or file_manifest_sha256+counts")
    if model:
        problems.extend(_format_problems(record, model))
    sig = signature_status(record)
    if require_signature and not sig["ok"]:
        problems.extend(sig["problems"])
    return {
        "schema": "hawking.frontier_source_provenance_status.v1",
        "label": record.get("label") or record.get("model"),
        "hf_id": record.get("hf_id"),
        "receipt_state": record.get("receipt_state"),
        "ok": not problems,
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
    needs_compressed = any(token in model.source_kind.lower() for token in ("compressed", "fp4", "fp8", "int4"))
    return {
        "schema": SCHEMA,
        "generated_at": _now(),
        "git_commit": _git_commit(ROOT),
        "receipt_state": "draft",
        "source": "hf-hub",
        "machine_class": machine_class,
        "label": model.label,
        "hf_id": model.hf_id,
        "source_kind": model.source_kind,
        "source_is_prequantized": "TODO true" if needs_compressed else "TODO false",
        "source_format": "<bf16|compressed-tensors|fp4+fp8|int4>",
        "revision": "<exact HF revision sha>",
        "model_card_url": f"https://huggingface.co/{model.hf_id}",
        "download_receipt": "<reports/condense/frontier_downloads.jsonl row or hf cache verify receipt>",
        "format_receipt": "<model card or config evidence for source format>",
        "procurement_command": f"python3.12 tools/condense/procure.py {model.label} --retries 2 --min-observed-mbs 80 --verify --progress-interval-s 60 --stall-timeout-s 900",
        "file_manifest_sha256": "<sha256 of source file manifest>",
        "file_count": "TODO >0",
        "total_bytes": "TODO >0",
    }


def complete_record(model: FrontierModel) -> dict[str, Any]:
    needs_compressed = any(token in model.source_kind.lower() for token in ("compressed", "fp4", "fp8", "int4"))
    return {
        "schema": SCHEMA,
        "generated_at": _now(),
        "git_commit": "selftest",
        "receipt_state": "final",
        "source": "hf-hub",
        "machine_class": "Studio-M1Ultra-128",
        "label": model.label,
        "hf_id": model.hf_id,
        "source_kind": model.source_kind,
        "source_is_prequantized": bool(needs_compressed),
        "source_format": "compressed-tensors" if needs_compressed else "bf16",
        "revision": "a" * 40,
        "model_card_url": f"https://huggingface.co/{model.hf_id}",
        "download_receipt": "selftest://download-receipt",
        "format_receipt": "selftest://format-receipt",
        "procurement_command": f"python3.12 tools/condense/procure.py {model.label} --verify",
        "file_manifest_sha256": "b" * 64,
        "file_count": 2,
        "total_bytes": 4096,
    }


def provenance_plan(root: pathlib.Path, labels: list[str]) -> dict[str, Any]:
    models = _selected_models(labels)
    rows = []
    for model in models:
        rows.append({
            "label": model.label,
            "hf_id": model.hf_id,
            "source_kind": model.source_kind,
            "path": str(provenance_path(root, model.label)),
            "command_template": (
                f"hawking studio source-provenance-receipt draft {model.label} --sign-draft --force"
            ),
        })
    roll = provenance_rollup(root, [m.label for m in models])
    return {
        "schema": "hawking.frontier_source_provenance_plan.v1",
        "generated_at": _now(),
        "model_count": len(models),
        "rows": rows,
        "rollup": roll,
    }


def provenance_status(root: pathlib.Path, label: str) -> dict[str, Any]:
    model = frontier_by_label(label)
    record = _read_json(provenance_path(root, label))
    status = record_status(record, model=model, require_signature=True)
    status["path"] = str(provenance_path(root, label))
    return status


def provenance_rollup(root: pathlib.Path, labels: list[str]) -> dict[str, Any]:
    rows = [provenance_status(root, label) for label in labels]
    blocked = [row.get("label") or label for row, label in zip(rows, labels) if not row["ok"]]
    return {
        "schema": "hawking.frontier_source_provenance_rollup.v1",
        "model_count": len(labels),
        "passed_count": len(labels) - len(blocked),
        "blocked_count": len(blocked),
        "blocked_labels": blocked,
        "rows": rows,
        "ok": not blocked,
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


def dispatch(args, root: pathlib.Path = ROOT) -> int:
    if args.provenance_mode == "plan":
        labels = [m.label for m in _selected_models(args.label)]
        data = provenance_plan(root, labels)
        if args.out:
            _write_json(pathlib.Path(args.out), data)
        if args.json:
            print(json.dumps(data, indent=2, sort_keys=True))
            return 0 if data["rollup"]["ok"] else 1
        print("# frontier source provenance plan")
        roll = data["rollup"]
        print(f"# provenance {roll['passed_count']}/{roll['model_count']}")
        for row in data["rows"]:
            print(f"{row['label']}: {row['path']}")
            print(f"  {row['command_template']}")
        return 0 if data["rollup"]["ok"] else 1

    rows = []
    ok = True
    for model in _selected_models(args.label):
        path = provenance_path(root, model.label)
        if getattr(args, "out_dir", ""):
            path = pathlib.Path(args.out_dir) / path.name
        if args.provenance_mode == "draft":
            if path.exists() and not args.force:
                status = {"ok": False, "problems": ["path exists; use --force to overwrite"]}
                rows.append({"label": model.label, "path": str(path), "ok": False,
                             "problems": status["problems"]})
                ok = False
                continue
            record = draft_record(model, machine_class=args.machine_class)
            if args.sign_draft:
                record, status = sign_record(record, model=model, allow_blocked_draft=True)
            else:
                status = record_status(record, model=model, require_signature=False)
            _write_json(path, record)
        elif args.provenance_mode == "sign":
            record = _read_json(path)
            record, status = sign_record(record or {}, model=model,
                                         allow_blocked_draft=args.allow_blocked_draft)
            if _read_json(path):
                _write_json(path, record)
        elif args.provenance_mode == "verify":
            status = record_status(_read_json(path), model=model, require_signature=True)
        else:
            raise SystemExit(f"unknown source-provenance mode: {args.provenance_mode}")
        rows.append({"label": model.label, "path": str(path), "ok": status["ok"],
                     "problems": status["problems"]})
        ok = ok and status["ok"]
    result = {"schema": "hawking.frontier_source_provenance_run.v1",
              "mode": args.provenance_mode, "ok": ok, "rows": rows}
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"# frontier source provenance {args.provenance_mode}: {'OK' if ok else 'BLOCKED'}")
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

    bf16 = FRONTIER_MODELS[0]
    draft, draft_status = sign_record(draft_record(bf16), model=bf16, allow_blocked_draft=True)
    check("signed source-provenance draft stays blocked", not draft_status["ok"] and draft.get("signature"))
    final, final_status = sign_record(complete_record(bf16), model=bf16)
    check("complete bf16 source provenance signs and verifies", final_status["ok"])
    final["revision"] = "<placeholder>"
    check("tampered/placeholder provenance is blocked", not record_status(final, model=bf16)["ok"])
    compressed = next(m for m in FRONTIER_MODELS if "compressed" in m.source_kind.lower()
                      or "fp4" in m.source_kind.lower())
    compressed_record = complete_record(compressed)
    compressed_record["source_is_prequantized"] = False
    _, compressed_status = sign_record(compressed_record, model=compressed)
    check("compressed source without prequantized flag is blocked", not compressed_status["ok"])

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        args = argparse.Namespace(provenance_mode="draft", label=[bf16.label], out_dir=str(root),
                                  force=True, sign_draft=True,
                                  machine_class="Studio-M1Ultra-128", json=True)
        check("draft command writes blocked source provenance", dispatch(args, root=root) == 1)
        check("draft file exists", (root / provenance_path(root, bf16.label).name).exists())

    print(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def cmd_selftest(args) -> int:
    return 0 if selftest() else 1


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Draft/sign/verify frontier source-provenance receipts.")
    sub = ap.add_subparsers(dest="provenance_mode")
    p = sub.add_parser("plan", help="print source-provenance receipt paths")
    p.add_argument("label", nargs="*", help="frontier label(s); default all")
    p.add_argument("--out", default="")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=dispatch)
    for mode in ("draft", "sign", "verify"):
        p = sub.add_parser(mode, help=f"{mode} signed source-provenance receipts")
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
    p = sub.add_parser("selftest", help="synthetic source-provenance receipt tests")
    p.set_defaults(func=cmd_selftest)
    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()
    if not args.provenance_mode:
        args = ap.parse_args(["plan"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
