#!/usr/bin/env python3.12
"""Prove serial/parallel equality for the exact Doctor 10-rate x 4-branch matrix."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "reports/condense/doctor_v5_ultra/campaign_plan.json"
SERIAL = ROOT / "build/strand-block-serial/release/quantize-model"
PARALLEL = ROOT / "build/strand-block-parallel/release/quantize-model-block-parallel"
FIXTURE = (ROOT / "build/strand-block-parallel/integration-canary/"
           "block-parallel-canary.safetensors")
WORK = ROOT / "reports/condense/doctor_v5_acceleration/config_matrix"
RECEIPT = WORK / "receipt.json"
SCHEMA = "hawking.doctor_v5_block_parallel_exact_10x4_matrix.v1"
RATES = ("4", "3", "2", "1", "0.8", "0.55", "0.5", "0.33", "0.25", "0.1")
BRANCHES = ("codec_control", "doctor_static", "doctor_conditional", "doctor_full")


class MatrixError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _hash_file(path: Path) -> tuple[str, int]:
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _artifact(path: Path) -> dict[str, Any]:
    digest, size = _hash_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _inventory() -> list[dict[str, Any]]:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    cells = {(row["rate_id"], row["branch"]): row for row in plan["cells"]
             if row["model_label"] == "0.5B"}
    expected = {(rate, branch) for rate in RATES for branch in BRANCHES}
    if set(cells) != expected or len(cells) != 40:
        raise MatrixError("campaign does not contain the exact 0.5B 10x4 matrix")
    rows = []
    for rate in RATES:
        for branch in BRANCHES:
            cell = cells[(rate, branch)]
            path = ROOT / cell["runtime_spec_path"]
            spec = json.loads(path.read_text(encoding="utf-8"))
            binding = spec.get("campaign_binding", {})
            if binding.get("cell_id") != cell["cell_id"] \
                    or binding.get("cell_identity_sha256") != cell["cell_identity_sha256"] \
                    or spec.get("codec", {}).get("rate_id") != rate:
                raise MatrixError(f"runtime spec identity differs: {cell['cell_id']}")
            rows.append({
                "rate_id": rate, "branch": branch, "cell_id": cell["cell_id"],
                "cell_identity_sha256": cell["cell_identity_sha256"],
                "runtime_spec": _artifact(path), "codec": spec["codec"],
            })
    return rows


def _args(codec: dict[str, Any], binary: Path, output: Path,
          *, parallel: bool) -> list[str]:
    argv = [
        str(binary), "--in", str(FIXTURE), "--bits", str(codec["symbol_bits"]),
        "--threads", "1", "--quality", "--rht-cols", "--ragged-v2",
        "--tensor-scope", codec["tensor_scope"], "--block-len",
        str(codec["block_len"]), "--only", "q_proj.weight",
    ]
    if codec["vector_dim"] > 1:
        argv += ["--vec-dim", str(codec["vector_dim"])]
        if codec["learned_codebook"]:
            argv += ["--learned-codebook", "--encode-mem-budget-bytes",
                     str(12 * 1024 * 1024 * 1024)]
    if not codec["adaptive_scales"]:
        argv.append("--no-adaptive-scales")
    if codec["outlier_channel_pct"] > 0:
        argv += ["--outlier-channel", str(codec["outlier_channel_pct"]),
                 "--outlier-bits", str(codec["outlier_bits"])]
    if codec["sdsq_sideinfo"]:
        argv.append("--sdsq-sideinfo")
    if codec["c2f_outl"]:
        argv.append("--c2f-outl")
    argv += ["--packed-v2-out", str(output)]
    if parallel:
        argv += ["--block-threads", "20", "--block-scratch-budget-bytes",
                 str(256 * 1024 * 1024)]
    return argv


def _execute(argv: list[str], log: Path) -> float:
    env = os.environ.copy(); env["STRAND_NO_GPU"] = "1"
    started = time.monotonic()
    with log.open("wb") as handle:
        result = subprocess.run(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise MatrixError(f"matrix invocation failed; see {log}")
    return time.monotonic() - started


def run() -> dict[str, Any]:
    rows = _inventory()
    WORK.mkdir(parents=True, exist_ok=True)
    results = []
    for index, row in enumerate(rows):
        stem = f"{index:02d}-{row['rate_id'].replace('.', 'p')}-{row['branch']}"
        serial_out, parallel_out = WORK / f"{stem}-serial.strand", WORK / f"{stem}-parallel.strand"
        serial_log, parallel_log = WORK / f"{stem}-serial.log", WORK / f"{stem}-parallel.log"
        for path in (serial_out, parallel_out, serial_log, parallel_log):
            path.unlink(missing_ok=True)
        serial_argv = _args(row["codec"], SERIAL, serial_out, parallel=False)
        parallel_argv = _args(row["codec"], PARALLEL, parallel_out, parallel=True)
        serial_seconds = _execute(serial_argv, serial_log)
        parallel_seconds = _execute(parallel_argv, parallel_log)
        serial_artifact, parallel_artifact = _artifact(serial_out), _artifact(parallel_out)
        if serial_artifact["sha256"] != parallel_artifact["sha256"]:
            raise MatrixError(f"serial/parallel bytes differ: {row['cell_id']}")
        results.append({
            **row, "serial_argv": serial_argv, "parallel_argv": parallel_argv,
            "serial_seconds": serial_seconds, "parallel_seconds": parallel_seconds,
            "serial_output": serial_artifact, "parallel_output": parallel_artifact,
            "serial_log": _artifact(serial_log), "parallel_log": _artifact(parallel_log),
            "exact_match": True,
        })
    doc = {
        "schema": SCHEMA, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "pass", "plan": _artifact(PLAN), "fixture": _artifact(FIXTURE),
        "serial_binary": _artifact(SERIAL), "parallel_binary": _artifact(PARALLEL),
        "rates": list(RATES), "branches": list(BRANCHES), "cell_count": 40,
        "cells": results, "all_exact": True, "cpu_only": True,
        "source_deletion_permitted": False,
    }
    doc["receipt_sha256"] = _hash_value(doc)
    _atomic_json(RECEIPT, doc)
    return doc


def validate(doc: Any, *, verify_files: bool = True) -> list[str]:
    errors: list[str] = []
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA \
            or doc.get("status") != "pass":
        return ["config-matrix schema/status is invalid"]
    if doc.get("receipt_sha256") != _hash_value(
            {key: value for key, value in doc.items() if key != "receipt_sha256"}):
        errors.append("config-matrix receipt hash differs")
    cells = doc.get("cells")
    identities = {(row.get("rate_id"), row.get("branch")) for row in cells
                  if isinstance(row, dict)} if isinstance(cells, list) else set()
    expected = {(rate, branch) for rate in RATES for branch in BRANCHES}
    if doc.get("cell_count") != 40 or len(cells or []) != 40 or identities != expected \
            or any(row.get("exact_match") is not True for row in cells or []):
        errors.append("config-matrix is not exact 10x4 coverage")
    try:
        canonical_rows = {(row["rate_id"], row["branch"]): row for row in _inventory()}
    except (OSError, ValueError, MatrixError) as exc:
        errors.append(f"canonical matrix inventory cannot be rebuilt: {exc}")
        canonical_rows = {}
    for cell in cells or []:
        identity = (cell.get("rate_id"), cell.get("branch"))
        expected_row = canonical_rows.get(identity)
        if expected_row is None or any(cell.get(key) != expected_row.get(key) for key in (
                "cell_id", "cell_identity_sha256", "runtime_spec", "codec")):
            errors.append(f"matrix runtime binding differs: {cell.get('cell_id')}")
            continue
        serial_output = cell.get("serial_output", {})
        parallel_output = cell.get("parallel_output", {})
        if serial_output.get("sha256") != parallel_output.get("sha256") \
                or serial_output.get("bytes") != parallel_output.get("bytes"):
            errors.append(f"matrix serial/parallel outputs differ: {cell.get('cell_id')}")
        try:
            expected_serial = _args(cell["codec"], SERIAL,
                                    Path(serial_output["path"]), parallel=False)
            expected_parallel = _args(cell["codec"], PARALLEL,
                                      Path(parallel_output["path"]), parallel=True)
        except (KeyError, TypeError, ValueError):
            errors.append(f"matrix invocation cannot be reconstructed: {cell.get('cell_id')}")
        else:
            if cell.get("serial_argv") != expected_serial \
                    or cell.get("parallel_argv") != expected_parallel:
                errors.append(f"matrix invocation differs from runtime codec: {cell.get('cell_id')}")
    if verify_files:
        for row in [doc.get("plan"), doc.get("fixture"), doc.get("serial_binary"),
                    doc.get("parallel_binary")]:
            try:
                if _artifact(Path(row["path"])) != row:
                    errors.append("a config-matrix bound artifact changed")
            except (OSError, KeyError, TypeError):
                errors.append("a config-matrix bound artifact cannot be verified")
        for cell in cells or []:
            for key in ("runtime_spec", "serial_output", "parallel_output",
                        "serial_log", "parallel_log"):
                row = cell.get(key, {})
                try:
                    if _artifact(Path(row["path"])) != row:
                        errors.append(f"matrix artifact changed: {cell.get('cell_id')}:{key}")
                except (OSError, KeyError, TypeError):
                    errors.append(f"matrix artifact unavailable: {cell.get('cell_id')}:{key}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    args = parser.parse_args()
    doc = run() if args.command == "run" else json.loads(RECEIPT.read_text())
    errors = validate(doc, verify_files=True)
    print(json.dumps({"ok": not errors, "errors": errors,
                      "receipt_sha256": doc.get("receipt_sha256"),
                      "cells": doc.get("cell_count")}, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, MatrixError) as exc:
        print(f"doctor_v5_block_parallel_config_matrix: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
