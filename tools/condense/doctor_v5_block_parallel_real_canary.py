#!/usr/bin/env python3.12
"""Run an exact, real-Qwen block-parallel timing/RSS promotion canary."""
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

from safetensors import safe_open

import doctor_v5_source_seal as source_seal


ROOT = Path(__file__).resolve().parents[2]
SERIAL = ROOT / "build/strand-block-serial/release/quantize-model"
PARALLEL = ROOT / "build/strand-block-parallel/release/quantize-model-block-parallel"
WORK = ROOT / "reports/condense/doctor_v5_acceleration/real_tensor_canary"
RECEIPT = WORK / "receipt.json"
SCHEMA = "hawking.doctor_v5_block_parallel_real_tensor_canary.v1"
LABEL = "3B"
MIN_WEIGHTS = 4_000_000
MAX_WEIGHTS = 16_000_000
MAX_RSS_BYTES = 50_000_000_000
PROJECTIONS = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
               "gate_proj.weight", "up_proj.weight", "down_proj.weight")


class CanaryError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _hash_file(path: Path) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk); size += len(chunk)
    return digest.hexdigest(), size


def _artifact(path: Path) -> dict[str, Any]:
    digest, size = _hash_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_seal() -> dict[str, Any]:
    path = source_seal.default_path(LABEL)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CanaryError(f"cannot read {LABEL} source seal: {exc}") from exc
    errors = source_seal.validate_document(doc, verify_structural=True)
    if errors:
        raise CanaryError("source seal is invalid: " + "; ".join(errors))
    return doc


def _select_tensor(seal: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for shard in seal["shards"]:
        path = Path(shard["path"])
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if not name.endswith(PROJECTIONS):
                    continue
                view = handle.get_slice(name)
                shape = list(view.get_shape())
                weights = 1
                for extent in shape:
                    weights *= int(extent)
                if len(shape) != 2 or weights < MIN_WEIGHTS:
                    continue
                candidates.append({
                    "name": name, "shape": shape, "weights": weights,
                    "dtype": str(view.get_dtype()), "shard": shard,
                })
    if not candidates:
        raise CanaryError("no suitably large real projection tensor was found")
    bounded = [row for row in candidates if row["weights"] <= MAX_WEIGHTS]
    chosen = min(bounded or candidates, key=lambda row: (row["weights"], row["name"]))
    if source_seal.lookup(Path(chosen["shard"]["path"])) != (
            chosen["shard"]["sha256"], chosen["shard"]["bytes"]):
        raise CanaryError("selected source shard no longer matches its seal")
    return chosen


def _rss_bytes(pid: int) -> int:
    result = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                            capture_output=True, text=True, check=False)
    try:
        return int(result.stdout.strip()) * 1024
    except ValueError:
        return 0


def _run(binary: Path, source: Path, tensor: str, output: Path,
         *, parallel: bool) -> dict[str, Any]:
    argv = [
        str(binary), "--in", str(source), "--bits", "4", "--threads", "1",
        "--quality", "--rht-cols", "--ragged-v2", "--tensor-scope", "all-2d",
        "--block-len", "256", "--outlier-channel", "1", "--outlier-bits", "8",
        "--sdsq-sideinfo", "--c2f-outl", "--only", tensor,
        "--packed-v2-out", str(output),
    ]
    if parallel:
        argv.extend(["--block-threads", "20", "--block-scratch-budget-bytes",
                     str(256 * 1024 * 1024)])
    log = output.with_suffix(output.suffix + ".log")
    env = os.environ.copy(); env["STRAND_NO_GPU"] = "1"
    started = time.monotonic(); peak = 0
    with log.open("wb") as handle:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                   stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True)
        while process.poll() is None:
            peak = max(peak, _rss_bytes(process.pid))
            time.sleep(0.2)
        peak = max(peak, _rss_bytes(process.pid))
    elapsed = time.monotonic() - started
    if process.returncode != 0 or not output.is_file():
        raise CanaryError(f"{'parallel' if parallel else 'serial'} canary failed; see {log}")
    return {
        "binary": _artifact(binary), "argv": argv,
        "environment": {"STRAND_NO_GPU": "1"}, "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak, "output": _artifact(output), "log": _artifact(log),
    }


def run() -> dict[str, Any]:
    lease = source_seal._acquire_exclusive_heavy_lease()
    try:
        seal = _read_seal()
        tensor = _select_tensor(seal)
        WORK.mkdir(parents=True, exist_ok=True)
        serial_output = WORK / "serial.strand"
        parallel_output = WORK / "parallel.strand"
        for path in (serial_output, parallel_output,
                     serial_output.with_suffix(".strand.log"),
                     parallel_output.with_suffix(".strand.log")):
            path.unlink(missing_ok=True)
        serial = _run(SERIAL, Path(tensor["shard"]["path"]), tensor["name"],
                      serial_output, parallel=False)
        parallel = _run(PARALLEL, Path(tensor["shard"]["path"]), tensor["name"],
                        parallel_output, parallel=True)
        speedup = serial["elapsed_seconds"] / max(parallel["elapsed_seconds"], 1e-9)
        exact = serial["output"]["sha256"] == parallel["output"]["sha256"]
        if not exact or speedup <= 1.5 or parallel["peak_rss_bytes"] > MAX_RSS_BYTES:
            raise CanaryError("real-tensor exactness, speedup, or RSS gate failed")
        doc = {
            "schema": SCHEMA, "created_at": _now(), "status": "pass",
            "source_seal": _artifact(source_seal.default_path(LABEL)),
            "source_shard": {key: tensor["shard"][key]
                             for key in ("path", "sha256", "bytes", "identity")},
            "tensor": {key: tensor[key] for key in ("name", "shape", "weights", "dtype")},
            "serial": serial, "parallel": parallel, "exact_output": True,
            "speedup": speedup, "max_parallel_rss_bytes": MAX_RSS_BYTES,
            "cpu_only": True, "source_deletion_permitted": False,
        }
        doc["receipt_sha256"] = _hash_value(doc)
        _atomic_json(RECEIPT, doc)
        return doc
    finally:
        lease.close()


def validate(doc: Any, *, verify_files: bool = True) -> list[str]:
    errors: list[str] = []
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA \
            or doc.get("status") != "pass":
        return ["real-tensor canary schema/status is invalid"]
    payload = {key: value for key, value in doc.items() if key != "receipt_sha256"}
    if doc.get("receipt_sha256") != _hash_value(payload):
        errors.append("real-tensor canary receipt hash differs")
    if doc.get("exact_output") is not True or doc.get("cpu_only") is not True \
            or not isinstance(doc.get("speedup"), (int, float)) or doc["speedup"] <= 1.5:
        errors.append("real-tensor exact CPU speedup is not proven")
    if doc.get("parallel", {}).get("peak_rss_bytes", MAX_RSS_BYTES + 1) > MAX_RSS_BYTES:
        errors.append("real-tensor parallel RSS exceeds the gate")
    if doc.get("serial", {}).get("output", {}).get("sha256") \
            != doc.get("parallel", {}).get("output", {}).get("sha256"):
        errors.append("real-tensor output hashes differ")
    if verify_files:
        for section in ("source_seal",):
            row = doc.get(section, {})
            try:
                if _artifact(Path(row["path"])) != row:
                    errors.append(f"{section} artifact changed")
            except (OSError, KeyError, TypeError):
                errors.append(f"{section} artifact cannot be verified")
        for branch in ("serial", "parallel"):
            for section in ("binary", "output", "log"):
                row = doc.get(branch, {}).get(section, {})
                try:
                    if _artifact(Path(row["path"])) != row:
                        errors.append(f"{branch} {section} artifact changed")
                except (OSError, KeyError, TypeError):
                    errors.append(f"{branch} {section} artifact cannot be verified")
        try:
            shard = doc["source_shard"]
            if source_seal.lookup(Path(shard["path"])) != (shard["sha256"], shard["bytes"]):
                errors.append("real-tensor source shard identity changed")
        except (OSError, KeyError, TypeError, source_seal.SourceSealError):
            errors.append("real-tensor source shard cannot be verified")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    args = parser.parse_args(argv)
    doc = run() if args.command == "run" else json.loads(RECEIPT.read_text())
    errors = validate(doc, verify_files=True)
    print(json.dumps({"ok": not errors, "errors": errors,
                      "receipt_sha256": doc.get("receipt_sha256"),
                      "speedup": doc.get("speedup")}, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, CanaryError, source_seal.SourceSealError) as exc:
        print(f"doctor_v5_block_parallel_real_canary: {exc}", file=__import__("sys").stderr)
        raise SystemExit(2)
