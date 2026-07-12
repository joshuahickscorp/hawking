#!/usr/bin/env python3.12
"""frontier_serve_capture.py - capture native `.tq` serve bench output as a signed receipt.

This is the Studio bridge between a real Hawking serve run and the strict frontier receipt gate. It
does not download or bake models. It reads an existing serve-bench JSON report, hashes the `.tq`
artifact, enforces native TQ/no-f16-rehydrate/all-linear/GPU ownership, and writes a signed
`reports/condense/<LABEL>_serve.json` only when the report is claim-admissible.
"""
from __future__ import annotations

import argparse
import hashlib
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
    git_commit as _git_commit,
    is_sha256 as _is_sha256,
    placeholder as _placeholder,
    read_json as _read_json,
    write_json as _write_json,
)
import frontier_receipt_runner  # noqa: E402
import frontier_receipts  # noqa: E402


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _truthy(report: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        if report.get(key) is True:
            return True
    return False


def _falsey(report: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        if report.get(key) is False:
            return True
    return False


def _number(report: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = report.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                pass
    return None


def _number_gb(report: dict[str, Any], *, gb: tuple[str, ...], mb: tuple[str, ...] = (),
               bytes_: tuple[str, ...] = ()) -> float | None:
    value = _number(report, *gb)
    if value is not None:
        return value
    value = _number(report, *mb)
    if value is not None:
        return value / 1024.0
    value = _number(report, *bytes_)
    if value is not None:
        return value / 1e9
    return None


def _native_tq(report: dict[str, Any]) -> bool:
    mode = str(report.get("decode_mode") or report.get("mode") or "").lower()
    return _truthy(report, "native_tq", "served_native_tq") or mode in {"native_tq", "native-tq"}


def _artifact_report_hash(report: dict[str, Any]) -> str | None:
    for key in ("artifact_sha256", "tq_sha256", "weights_sha256"):
        value = report.get(key)
        if _is_sha256(value):
            return value.lower()
    return None


def _memory_peak_gb(report: dict[str, Any]) -> float | None:
    return _number_gb(
        report,
        gb=("memory_peak_gb", "peak_memory_gb", "peak_unified_memory_gb", "peak_rss_gb"),
        mb=("memory_peak_mb", "peak_memory_mb", "peak_rss_mb"),
        bytes_=("memory_peak_bytes", "peak_memory_bytes", "peak_rss_bytes"),
    )


def _memory_resident_gb(report: dict[str, Any]) -> float | None:
    return _number_gb(
        report,
        gb=("memory_resident_gb", "resident_memory_gb", "loaded_resident_gb", "artifact_resident_gb"),
        mb=("memory_resident_mb", "resident_memory_mb", "loaded_resident_mb"),
        bytes_=("memory_resident_bytes", "resident_memory_bytes", "loaded_resident_bytes"),
    )


def _unified_memory_gb(report: dict[str, Any]) -> float | None:
    return _number_gb(
        report,
        gb=("unified_memory_gb", "total_unified_memory_gb", "ram_gb", "total_ram_gb"),
        mb=("unified_memory_mb", "total_unified_memory_mb", "ram_mb", "total_ram_mb"),
        bytes_=("unified_memory_bytes", "total_unified_memory_bytes", "ram_bytes", "total_ram_bytes"),
    )


def _resident_memory_ok(report: dict[str, Any], peak_gb: float | None,
                        unified_gb: float | None) -> bool:
    explicit = _truthy(
        report,
        "resident_memory_ok",
        "resident_fit",
        "artifact_resident",
        "no_swap",
        "no_memory_pressure",
    )
    return explicit and peak_gb is not None and unified_gb is not None and peak_gb <= unified_gb


def _capture_problems(report: dict[str, Any], artifact_hash: str, command: str,
                      load_receipt: str, served_forward_receipt: str,
                      parity_receipt: str) -> list[str]:
    problems = []
    if not command or _placeholder(command):
        problems.append("exact serve command is missing or contains placeholder text")
    if not load_receipt or _placeholder(load_receipt):
        problems.append("load receipt path is missing or placeholder")
    if not served_forward_receipt or _placeholder(served_forward_receipt):
        problems.append("served-forward receipt path is missing or placeholder")
    if not parity_receipt or _placeholder(parity_receipt):
        problems.append("parity receipt path is missing or placeholder")
    if not _native_tq(report):
        problems.append("serve report does not prove native_tq decode mode")
    if not _falsey(report, "rehydrate_f16", "rehydrated_f16"):
        problems.append("serve report must prove rehydrate_f16/rehydrated_f16 is false")
    if not _truthy(report, "tq_strict", "strict_tq"):
        problems.append("serve report must prove tq_strict=true")
    if not _truthy(report, "all_linear", "all_linear_covered"):
        problems.append("serve report must prove all_linear=true")
    if not _truthy(report, "gpu_bitslice", "gpu_owned", "gpu_ownership"):
        problems.append("serve report must prove gpu_bitslice/GPU ownership")
    if not _truthy(report, "served_forward_pass"):
        problems.append("serve report must prove served_forward_pass=true")
    if not _truthy(report, "parity_pass"):
        problems.append("serve report must prove parity_pass=true")
    tok_s = _number(report, "tok_s", "tokens_per_second", "decode_tok_s")
    if tok_s is None or tok_s <= 0:
        problems.append("serve report tok/s must be positive")
    peak_gb = _memory_peak_gb(report)
    resident_gb = _memory_resident_gb(report)
    unified_gb = _unified_memory_gb(report)
    if peak_gb is None or peak_gb <= 0:
        problems.append("serve report memory_peak_gb must be positive")
    if resident_gb is None or resident_gb <= 0:
        problems.append("serve report memory_resident_gb must be positive")
    if unified_gb is None or unified_gb <= 0:
        problems.append("serve report unified_memory_gb must be positive")
    if not _resident_memory_ok(report, peak_gb, unified_gb):
        problems.append("serve report must prove resident_memory_ok=true and peak memory fits unified memory")
    report_hash = _artifact_report_hash(report)
    if report_hash and report_hash != artifact_hash.lower():
        problems.append("serve report artifact hash does not match the .tq file")
    return problems


def build_record(root: pathlib.Path, label: str, artifact: pathlib.Path, bench_json: pathlib.Path,
                 command: str, load_receipt: str, served_forward_receipt: str,
                 parity_receipt: str,
                 machine_class: str = "Studio-M3Ultra-96") -> tuple[dict[str, Any], dict[str, Any]]:
    model = frontier_by_label(label)
    if not model:
        raise SystemExit(f"unknown frontier label: {label}")
    artifact = artifact.resolve()
    bench_json = bench_json.resolve()
    if not artifact.exists() or not artifact.is_file():
        raise SystemExit(f"artifact not found: {artifact}")
    report = _read_json(bench_json)
    if not report:
        raise SystemExit(f"serve bench JSON missing or unreadable: {bench_json}")
    artifact_hash = _sha256_file(artifact)
    problems = _capture_problems(
        report,
        artifact_hash,
        command,
        load_receipt,
        served_forward_receipt,
        parity_receipt,
    )
    tok_s = _number(report, "tok_s", "tokens_per_second", "decode_tok_s")
    peak_gb = _memory_peak_gb(report)
    resident_gb = _memory_resident_gb(report)
    unified_gb = _unified_memory_gb(report)
    record = {
        "schema": frontier_receipt_runner.SERVE_SCHEMA,
        "model": model.label,
        "hf_id": model.hf_id,
        "receipt_state": "final",
        "source": "measured",
        "machine_class": machine_class,
        "status": "pass" if not problems else "blocked",
        "native_tq": _native_tq(report),
        "rehydrate_f16": False if _falsey(report, "rehydrate_f16", "rehydrated_f16") else None,
        "tq_strict": _truthy(report, "tq_strict", "strict_tq"),
        "all_linear": _truthy(report, "all_linear", "all_linear_covered"),
        "gpu_bitslice": _truthy(report, "gpu_bitslice", "gpu_owned", "gpu_ownership"),
        "served_forward_pass": _truthy(report, "served_forward_pass"),
        "parity_pass": _truthy(report, "parity_pass"),
        "tok_s": tok_s,
        "memory_peak_gb": peak_gb,
        "memory_resident_gb": resident_gb,
        "unified_memory_gb": unified_gb,
        "resident_memory_ok": _resident_memory_ok(report, peak_gb, unified_gb),
        "artifact_path": str(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": artifact_hash,
        "bench_json_path": str(bench_json),
        "bench_json_sha256": _sha256_file(bench_json),
        "commands": [command],
        "git_commit": _git_commit(root),
        "load_receipt": load_receipt,
        "served_forward_receipt": served_forward_receipt,
        "parity_receipt": parity_receipt,
        "serve_report": report,
    }
    if problems:
        status = {"ok": False, "problems": problems}
        return record, status
    signed, status = frontier_receipt_runner.sign_record(record, kind="serve")
    return signed, status


def capture(args, root: pathlib.Path = ROOT) -> int:
    out = pathlib.Path(args.out) if args.out else frontier_receipts.serve_path(root, args.label)
    if out.exists() and not args.force:
        print(f"[frontier-serve-capture] {out} exists; use --force to overwrite", file=sys.stderr)
        return 2
    record, status = build_record(
        root,
        args.label,
        pathlib.Path(args.artifact),
        pathlib.Path(args.bench_json),
        args.command,
        args.load_receipt,
        args.served_forward_receipt,
        args.parity_receipt,
        machine_class=args.machine_class,
    )
    if not status["ok"]:
        if args.json:
            print(json.dumps({"ok": False, "status": status, "record": record}, indent=2, sort_keys=True))
        else:
            print("# frontier native serve capture: BLOCKED")
            for problem in status["problems"]:
                print(f"  - {problem}")
        return 1
    _write_json(out, record)
    if args.json:
        print(json.dumps({"ok": True, "path": str(out), "status": status}, indent=2, sort_keys=True))
    else:
        print(f"# frontier native serve capture: OK -> {out}")
        print(f"# tok_s={record['tok_s']} artifact_sha256={record['artifact_sha256']}")
    return 0


def selftest() -> bool:
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        artifact = root / "tiny.tq"
        artifact.write_bytes(b"tiny-tq")
        ah = _sha256_file(artifact)
        report = {
            "decode_mode": "native_tq",
            "rehydrated_f16": False,
            "tq_strict": True,
            "all_linear": True,
            "gpu_bitslice": True,
            "served_forward_pass": True,
            "parity_pass": True,
            "tok_s": 12.5,
            "memory_peak_gb": 3.5,
            "memory_resident_gb": 3.0,
            "unified_memory_gb": 96.0,
            "resident_memory_ok": True,
            "artifact_sha256": ah,
        }
        report_path = root / "serve_report.json"
        _write_json(report_path, report)
        label = FRONTIER_MODELS[0].label
        record, status = build_record(
            root,
            label,
            artifact,
            report_path,
            "target/release/hawking serve --weights tiny.tq --bench-decode --report-json",
            "selftest://load",
            "selftest://served-forward",
            "selftest://parity",
        )
        check("native serve capture signs strict receipt", status["ok"] and record.get("signature"))
        check("signed capture verifies in strict runner",
              frontier_receipt_runner.record_status(record, kind="serve")["ok"])
        bad = dict(report)
        bad["decode_mode"] = "f16_rehydrate"
        bad["rehydrated_f16"] = True
        bad_path = root / "bad_report.json"
        _write_json(bad_path, bad)
        _, bad_status = build_record(
            root,
            label,
            artifact,
            bad_path,
            "target/release/hawking serve --weights tiny.tq --bench-decode --report-json",
            "selftest://load",
            "selftest://served-forward",
            "selftest://parity",
        )
        check("rehydrate/f16 report is blocked", not bad_status["ok"])
        mismatch = dict(report)
        mismatch["artifact_sha256"] = "0" * 64
        mismatch_path = root / "mismatch_report.json"
        _write_json(mismatch_path, mismatch)
        _, mismatch_status = build_record(
            root,
            label,
            artifact,
            mismatch_path,
            "target/release/hawking serve --weights tiny.tq --bench-decode --report-json",
            "selftest://load",
            "selftest://served-forward",
            "selftest://parity",
        )
        check("artifact hash mismatch is blocked", not mismatch_status["ok"])
        no_memory = dict(report)
        no_memory.pop("memory_peak_gb")
        no_memory_path = root / "no_memory_report.json"
        _write_json(no_memory_path, no_memory)
        _, no_memory_status = build_record(
            root,
            label,
            artifact,
            no_memory_path,
            "target/release/hawking serve --weights tiny.tq --bench-decode --report-json",
            "selftest://load",
            "selftest://served-forward",
            "selftest://parity",
        )
        check("missing memory proof is blocked", not no_memory_status["ok"])
    print(f"\n# SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


def cmd_selftest(args) -> int:
    return 0 if selftest() else 1


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Capture a native `.tq` serve bench JSON as a signed receipt.")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("capture", help="write a signed frontier serve receipt from a bench JSON report")
    p.add_argument("label")
    p.add_argument("--artifact", required=True, help="existing .tq artifact path")
    p.add_argument("--bench-json", required=True, dest="bench_json", help="JSON report emitted by native serve")
    p.add_argument("--command", required=True, help="exact serve command that produced the report")
    p.add_argument("--load-receipt", required=True)
    p.add_argument("--served-forward-receipt", required=True)
    p.add_argument("--parity-receipt", required=True)
    p.add_argument("--machine-class", default="Studio-M3Ultra-96")
    p.add_argument("--out", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=capture)
    p = sub.add_parser("selftest", help="synthetic capture tests; no model serving")
    p.set_defaults(func=cmd_selftest)
    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()
    if not args.cmd:
        args = ap.parse_args(["selftest"])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
