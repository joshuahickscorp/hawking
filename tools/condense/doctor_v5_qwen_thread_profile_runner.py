#!/usr/bin/env python3.12
"""Default-off, receipt-gated Qwen production thread-profile qualification.

This program is deliberately separate from the Doctor queue.  ``status`` and
``make-matrix`` are read-only/cheap.  ``run`` is the only physical execution
surface and requires ``--execute-physical``; it then holds the shared heavy
lease, requires an owner-free/green host before every arm, and executes one
real source-bound Qwen projection with the exact codec in its runtime spec.

The serial arm is the canonical oracle.  Candidates 8/12/16/20 use the same
binary, tensor, codec, environment and scratch budget.  A production vendor
receipt is issued only for byte-identical output.  Existing receipts are
skipped only after full validation; invalid existing evidence is never
overwritten.  Runtime defaults and the live queue are never changed.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Iterable

from safetensors import safe_open

import doctor_v5_source_seal as source_seal
import ram_scheduler
import spec_reentry_scaffold


ROOT = Path(__file__).resolve().parents[2]
STAGE_ROOT = ROOT / "reports/condense/doctor_v5_ultra/staged_acceleration/aggressive_v2"
OUTPUT_ROOT = STAGE_ROOT / "profile_qualification"
HEAVY_LOCK = ROOT / "reports/cron/studio_heavy.lock"
CONTRACT_PATH = ROOT / "vendor/strand-quant/tools/thread_profile_contract.py"
PLAN_PATH = ROOT / "reports/condense/doctor_v5_ultra/campaign_plan.json"
EXPECTED_BINARY = ROOT / "build/strand-block-parallel/release/quantize-model-block-parallel"
EXPECTED_SERIAL_BINARY = ROOT / "build/strand-block-serial/release/quantize-model"

RECEIPT_SCHEMA = "hawking.strand.tier-rate-thread-canary.v1"
CANONICAL_SCHEMA = "hawking.doctor_v5_qwen_thread_canonical.v1"
MATRIX_SCHEMA = "hawking.doctor_v5_qwen_thread_profile_matrix.v1"
QUALIFICATION_SCHEMA = "hawking.doctor_v5_qwen_thread_profile_qualification.v1"
FAILURE_SCHEMA = "hawking.doctor_v5_qwen_thread_profile_failure.v1"
VERSION = "2026-07-15.1"
THREADS = (8, 12, 16, 20)
BLOCK_SCRATCH_BUDGET_BYTES = 256 * 1024 * 1024
LEARNED_ENCODE_MEM_BUDGET_BYTES = 12 * 1024**3
RSS_LIMIT_BYTES = 66_000_000_000
MAX_SWAP_MB = 2048.0
MAX_SWAP_GROWTH_MB = 512.0
MIN_DISK_FREE_BYTES = 10_000_000_000
POLL_SECONDS = 0.2
GUARD_SECONDS = 2.0
OWNER_SECONDS = 10.0
THERMAL_SECONDS = 15.0
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
PROJECTIONS = (
    "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
    "gate_proj.weight", "up_proj.weight", "down_proj.weight",
)
CODEC_KEYS = {
    "adaptive_scales", "allow_over_ceiling_control", "artifact_mode", "block_len",
    "c2f_outl", "learned_codebook", "outlier_bits", "outlier_channel_pct",
    "quality", "ragged_v2", "rate_id", "rht_cols", "sdsq_sideinfo",
    "symbol_bits", "tensor_scope", "vector_dim",
}
THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS",
)


class QualificationError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualificationError(f"JSON root is not an object: {path}")
    return value


def _stable_hash(path: Path) -> tuple[str, int]:
    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QualificationError(f"cannot open artifact {path}: {exc}") from exc
    digest, size = hashlib.sha256(), 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise QualificationError(f"artifact is not a single regular file: {path}")
        while True:
            chunk = os.read(descriptor, 16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk); size += len(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev, before.st_ino, before.st_size,
            before.st_mtime_ns, before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev, after.st_ino, after.st_size,
            after.st_mtime_ns, after.st_ctime_ns,
        )
        if identity_before != identity_after or size != after.st_size:
            raise QualificationError(f"artifact changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), size


def _artifact(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve(strict=True)
    digest, size = _stable_hash(resolved)
    return {"path": str(resolved), "sha256": digest, "bytes": size}


def _artifact_matches(value: Any) -> bool:
    try:
        if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}:
            return False
        return _artifact(Path(value["path"])) == value
    except (OSError, QualificationError, TypeError, ValueError):
        return False


def _write_exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise OSError("short receipt write")
            offset += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
    finally:
        os.close(descriptor)


def _load_contract() -> Any:
    path = CONTRACT_PATH.resolve(strict=True)
    spec = importlib.util.spec_from_file_location(
        f"doctor_v5_qwen_profile_contract_{hashlib.sha256(path.read_bytes()).hexdigest()}",
        path,
    )
    if spec is None or spec.loader is None:
        raise QualificationError("cannot load vendor thread-profile contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.DEFAULT_THREADS) != THREADS:
        raise QualificationError("vendor thread candidate set is not exact 8/12/16/20")
    return module


def _runtime_program_payload(document: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "schema", "inputs", "resources", "resource_admission",
        "program_spec_sha256", "resource_admission_sha256",
    }
    return {key: copy.deepcopy(value) for key, value in document.items()
            if key not in excluded}


def _bound_path(value: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise QualificationError("runtime input path is invalid")
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(ROOT.resolve(strict=True))
    except ValueError as exc:
        raise QualificationError(f"runtime input escapes the workspace: {resolved}") from exc
    return resolved


def _validate_codec(value: Any, *, expected_rate: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != CODEC_KEYS:
        raise QualificationError("runtime codec keys differ from the reviewed Qwen schema")
    codec = copy.deepcopy(value)
    if codec.get("rate_id") != expected_rate:
        raise QualificationError("runtime codec rate differs from campaign binding")
    if codec.get("quality") is not True or codec.get("ragged_v2") is not True \
            or codec.get("rht_cols") is not True \
            or codec.get("tensor_scope") != "all-2d" \
            or codec.get("allow_over_ceiling_control") is not True:
        raise QualificationError("runtime codec is not the reviewed production Qwen path")
    for key in ("adaptive_scales", "c2f_outl", "learned_codebook", "sdsq_sideinfo"):
        if not isinstance(codec.get(key), bool):
            raise QualificationError(f"runtime codec {key} is not boolean")
    ints = {"symbol_bits": (1, 4), "vector_dim": (1, 32),
            "block_len": (256, 8192), "outlier_bits": (1, 16)}
    for key, (low, high) in ints.items():
        child = codec.get(key)
        if isinstance(child, bool) or not isinstance(child, int) or not low <= child <= high:
            raise QualificationError(f"runtime codec {key} is outside the reviewed range")
    outlier = codec.get("outlier_channel_pct")
    if isinstance(outlier, bool) or not isinstance(outlier, (int, float)) \
            or not math.isfinite(float(outlier)) or not 0 <= float(outlier) <= 5:
        raise QualificationError("runtime codec outlier percentage is invalid")
    expected_mode = "packed_scalar_control" if codec["vector_dim"] == 1 \
        else "packed_vector_control"
    if codec.get("artifact_mode") != expected_mode:
        raise QualificationError("runtime codec artifact/vector mode is inconsistent")
    return codec


def _spec_identity(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    binding = document.get("campaign_binding")
    if document.get("model_family") != "qwen2.5-dense" or not isinstance(binding, dict):
        raise QualificationError(f"runtime spec is not Qwen source-bound work: {path}")
    tier, rate, branch, cell_id = (
        binding.get("label"), binding.get("target_rate_id"),
        binding.get("branch"), binding.get("cell_id"),
    )
    if any(not isinstance(value, str) or not value for value in (tier, rate, branch, cell_id)):
        raise QualificationError(f"runtime spec campaign identity is invalid: {path}")
    return {"path": path.resolve(strict=True), "document": document,
            "tier": tier, "rate": rate, "branch": branch, "cell_id": cell_id}


def discover_specs(directory: Path) -> list[Path]:
    directory = Path(directory).resolve(strict=True)
    rows: dict[tuple[str, str], Path] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            identity = _spec_identity(path)
        except QualificationError:
            continue
        if identity["branch"] != "codec_control":
            continue
        key = identity["tier"], identity["rate"]
        if key in rows:
            raise QualificationError(f"duplicate codec-control runtime for {key}")
        rows[key] = identity["path"]
    if not rows:
        raise QualificationError("no source-bound Qwen codec-control runtimes were discovered")
    return [rows[key] for key in sorted(rows)]


def build_matrix_manifest(spec_paths: Iterable[Path]) -> dict[str, Any]:
    rows, identities = [], set()
    for original in spec_paths:
        identity = _spec_identity(Path(original))
        key = identity["tier"], identity["rate"]
        if key in identities:
            raise QualificationError(f"matrix repeats exact tier/rate {key}")
        identities.add(key)
        rows.append({
            "tier": key[0], "rate": key[1], "branch": identity["branch"],
            "cell_id": identity["cell_id"], "runtime_spec": _artifact(identity["path"]),
        })
    rows.sort(key=lambda row: (row["tier"], row["rate"]))
    if not rows:
        raise QualificationError("qualification matrix is empty")
    document = {
        "schema": MATRIX_SCHEMA, "version": VERSION,
        "execution_default": "off", "required_threads": list(THREADS),
        "spec_count": len(rows), "specs": rows,
        "runtime_defaults_mutated": False, "live_queue_mutated": False,
    }
    document["matrix_sha256"] = _hash_value(document)
    return document


def validate_matrix(document: Any, *, verify_files: bool = True) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict) or document.get("schema") != MATRIX_SCHEMA \
            or document.get("version") != VERSION:
        return ["matrix schema/version is invalid"]
    if document.get("matrix_sha256") != _hash_value(_without(document, "matrix_sha256")):
        errors.append("matrix self-hash differs")
    rows = document.get("specs")
    if not isinstance(rows, list) or document.get("spec_count") != len(rows) or not rows:
        errors.append("matrix spec inventory is invalid"); return errors
    keys = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
                "tier", "rate", "branch", "cell_id", "runtime_spec"}:
            errors.append("matrix row keys are invalid"); continue
        keys.append((row.get("tier"), row.get("rate")))
        if verify_files and not _artifact_matches(row.get("runtime_spec")):
            errors.append(f"matrix runtime artifact changed: {row.get('cell_id')}")
    if len(keys) != len(set(keys)) or keys != sorted(keys):
        errors.append("matrix exact tier/rate identities are duplicate or unsorted")
    if document.get("required_threads") != list(THREADS) \
            or document.get("execution_default") != "off" \
            or document.get("runtime_defaults_mutated") is not False \
            or document.get("live_queue_mutated") is not False:
        errors.append("matrix safety/qualification contract differs")
    return errors


def _safetensor_inventory(shards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for shard in shards:
        with safe_open(shard["path"], framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if not name.endswith(PROJECTIONS):
                    continue
                view = handle.get_slice(name)
                shape = [int(extent) for extent in view.get_shape()]
                if len(shape) != 2:
                    continue
                elements = math.prod(shape)
                if elements < 4_000_000:
                    continue
                rows.append({
                    "name": name, "shape": shape, "elements": elements,
                    "dtype": str(view.get_dtype()), "source_shard": shard,
                })
    return rows


def _select_tensor(shards: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = _safetensor_inventory(shards)
    if not candidates:
        raise QualificationError("no real production Qwen projection tensor was found")
    bounded = [row for row in candidates if row["elements"] <= 16_000_000]
    return min(bounded or candidates, key=lambda row: (row["elements"], row["name"]))


def load_context(spec_path: Path) -> dict[str, Any]:
    identity = _spec_identity(Path(spec_path))
    path, document = identity["path"], identity["document"]
    plan = _read_json(PLAN_PATH)
    if plan.get("schema") != "hawking.doctor_v5_ultra_campaign_plan.v1" \
            or plan.get("plan_sha256") != _hash_value(_without(plan, "plan_sha256")):
        raise QualificationError("Doctor campaign plan identity is invalid")
    plan_cell = next((row for row in plan.get("cells", [])
                      if isinstance(row, dict) and row.get("cell_id") == identity["cell_id"]), None)
    binding = document.get("campaign_binding", {})
    if not isinstance(plan_cell, dict) or any(
            plan_cell.get(field) != expected for field, expected in (
                ("model_family", "qwen2.5-dense"), ("model_label", identity["tier"]),
                ("rate_id", identity["rate"]), ("branch", identity["branch"]),
                ("cell_identity_sha256", binding.get("cell_identity_sha256")),
            )) or path.name != Path(str(plan_cell.get("runtime_spec_path", ""))).name:
        raise QualificationError("runtime spec is not an exact cell in the sealed Doctor plan")
    if document.get("schema") != "hawking.doctor_v5_strand_ladder_spec.v1" \
            or document.get("quality_claims_permitted") is not False \
            or document.get("source_deletion_permitted") is not False:
        raise QualificationError("runtime spec schema/claim boundary is invalid")
    if document.get("program_spec_sha256") != _hash_value(_runtime_program_payload(document)):
        raise QualificationError("runtime semantic program hash differs")
    resources = document.get("resources")
    if not isinstance(resources, dict) \
            or document.get("resource_admission_sha256") != _hash_value(resources):
        raise QualificationError("runtime resource admission hash differs")
    codec = _validate_codec(document.get("codec"), expected_rate=identity["rate"])
    inputs = document.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise QualificationError("runtime spec has no source-bound input inventory")
    by_role: dict[str, dict[str, Any]] = {}
    for row in inputs:
        if not isinstance(row, dict) or set(row) != {"role", "path", "sha256", "bytes"} \
                or not isinstance(row.get("role"), str) or row["role"] in by_role \
                or HEX64.fullmatch(str(row.get("sha256"))) is None \
                or isinstance(row.get("bytes"), bool) or not isinstance(row.get("bytes"), int):
            raise QualificationError("runtime input inventory is malformed or duplicate")
        by_role[row["role"]] = row
    if "quantizer" not in by_role or "source_seal" not in by_role:
        raise QualificationError("runtime spec omits quantizer or source seal")
    binary_path, seal_path = _bound_path(by_role["quantizer"]["path"]), \
        _bound_path(by_role["source_seal"]["path"])
    if binary_path != EXPECTED_BINARY.resolve(strict=True):
        raise QualificationError("runtime quantizer is not the reviewed block-parallel binary")
    binary, seal_artifact = _artifact(binary_path), _artifact(seal_path)
    serial_binary_path = EXPECTED_SERIAL_BINARY.resolve(strict=True)
    serial_binary = _artifact(serial_binary_path)
    if binary["sha256"] != by_role["quantizer"]["sha256"] \
            or binary["bytes"] != by_role["quantizer"]["bytes"]:
        raise QualificationError("runtime quantizer artifact changed")
    if seal_artifact["sha256"] != by_role["source_seal"]["sha256"] \
            or seal_artifact["bytes"] != by_role["source_seal"]["bytes"]:
        raise QualificationError("runtime source-seal artifact changed")
    seal_document = _read_json(seal_path)
    seal_errors = source_seal.validate_document(seal_document, verify_structural=True)
    if seal_document.get("schema") != source_seal.SCHEMA:
        raise QualificationError("physical qualification requires the reboot-stable v2 source seal")
    if seal_errors:
        raise QualificationError("runtime source seal is invalid: " + "; ".join(seal_errors))
    shard_rows = [row for role, row in sorted(by_role.items())
                  if role.startswith("source_shard:")]
    if not shard_rows or len(shard_rows) != len(seal_document.get("shards", [])):
        raise QualificationError("runtime shard inventory differs from source seal")
    shards = []
    for row in shard_rows:
        shard_path = _bound_path(row["path"])
        try:
            sealed_identity = source_seal.lookup(shard_path)
        except (OSError, source_seal.SourceSealError) as exc:
            raise QualificationError(f"source shard cannot be verified from seal: {exc}") from exc
        if sealed_identity is None:
            raise QualificationError("source shard is absent from the active v2 seal inventory")
        sealed_sha, sealed_bytes = sealed_identity
        if (sealed_sha, sealed_bytes) != (row["sha256"], row["bytes"]):
            raise QualificationError("runtime source shard differs from its v2 seal")
        seal_row = next((child for child in seal_document["shards"]
                         if Path(child["path"]).resolve() == shard_path), None)
        if not isinstance(seal_row, dict):
            raise QualificationError("runtime source shard is absent from exact seal")
        shards.append({key: seal_row[key] for key in ("path", "sha256", "bytes", "identity")})
    tensor = _select_tensor(shards)
    spec_artifact = _artifact(path)
    source_binding = {
        "campaign_plan": _artifact(PLAN_PATH), "plan_sha256": plan["plan_sha256"],
        "qualification_program": {
            "runner": _artifact(Path(__file__)), "vendor_contract": _artifact(CONTRACT_PATH),
            "source_seal_module": _artifact(Path(source_seal.__file__)),
            "resource_module": _artifact(Path(ram_scheduler.__file__)),
            "owner_observer_module": _artifact(Path(spec_reentry_scaffold.__file__)),
        },
        "runtime_spec": spec_artifact,
        "program_spec_sha256": document["program_spec_sha256"],
        "resource_admission_sha256": document["resource_admission_sha256"],
        "cell_id": identity["cell_id"], "tier": identity["tier"],
        "rate": identity["rate"], "branch": identity["branch"],
        "codec": codec, "codec_sha256": _hash_value(codec),
        "source_seal": seal_artifact,
        "serial_canonical_binary": serial_binary,
        "source_seal_schema": seal_document.get("schema"),
        "source_shard": tensor["source_shard"],
        "tensor": {key: tensor[key] for key in ("name", "shape", "elements", "dtype")},
        "selection": "smallest-real-qwen-projection>=4M-elements;prefer<=16M;name-tiebreak",
    }
    return {
        **identity, "runtime_spec": spec_artifact, "codec": codec,
        "binary": binary, "binary_path": binary_path,
        "serial_binary": serial_binary, "serial_binary_path": serial_binary_path,
        "source_seal": seal_artifact, "source_shard": tensor["source_shard"],
        "tensor": source_binding["tensor"], "source_binding": source_binding,
        "source_binding_sha256": _hash_value(source_binding),
    }


def _component(value: str) -> str:
    cleaned = SAFE_COMPONENT.sub("-", value).strip("-.")
    if not cleaned:
        raise QualificationError("empty qualification path component")
    return cleaned


def cell_root(context: dict[str, Any], *, output_root: Path = OUTPUT_ROOT) -> Path:
    return (Path(output_root) / "cells" / _component(context["tier"]) /
            _component(context["rate"]) / context["runtime_spec"]["sha256"][:16])


def canonical_path(context: dict[str, Any], *, output_root: Path = OUTPUT_ROOT) -> Path:
    return cell_root(context, output_root=output_root) / "canonical.json"


def receipt_path(context: dict[str, Any], threads: int,
                 *, output_root: Path = OUTPUT_ROOT) -> Path:
    if threads not in THREADS:
        raise QualificationError(f"thread candidate must be one of {THREADS}")
    return cell_root(context, output_root=output_root) / f"candidate-{threads:02d}.json"


def _launch_env(tmpdir: Path) -> dict[str, str]:
    env = {"STRAND_NO_GPU": "1", "RUST_BACKTRACE": "1", "LC_ALL": "C",
           "TMPDIR": str(tmpdir.resolve())}
    env.update({key: "1" for key in THREAD_ENV_KEYS})
    return env


def _base_argv(context: dict[str, Any], output: Path) -> list[str]:
    codec = context["codec"]
    argv = [
        str(context["binary_path"]), "--in", context["source_shard"]["path"],
        "--bits", str(codec["symbol_bits"]), "--threads", "1",
    ]
    if codec["quality"]:
        argv.append("--quality")
    argv.append("--rht-cols" if codec["rht_cols"] else "--rht-rows")
    argv.append("--ragged-v2" if codec["ragged_v2"] else "--strict-v2")
    argv += ["--tensor-scope", codec["tensor_scope"],
             "--block-len", str(codec["block_len"])]
    if codec["vector_dim"] > 1:
        argv += ["--vec-dim", str(codec["vector_dim"])]
        if codec["learned_codebook"]:
            argv += ["--learned-codebook", "--encode-mem-budget-bytes",
                     str(LEARNED_ENCODE_MEM_BUDGET_BYTES)]
    if not codec["adaptive_scales"]:
        argv.append("--no-adaptive-scales")
    if float(codec["outlier_channel_pct"]) > 0:
        argv += ["--outlier-channel", str(codec["outlier_channel_pct"]),
                 "--outlier-bits", str(codec["outlier_bits"])]
    if codec["sdsq_sideinfo"]:
        argv.append("--sdsq-sideinfo")
    if codec["c2f_outl"]:
        argv.append("--c2f-outl")
    argv += ["--only", context["tensor"]["name"],
             "--packed-v2-out", str(output.resolve())]
    return argv


def build_argv(context: dict[str, Any], output: Path, *, threads: int | None) -> list[str]:
    argv = _base_argv(context, output)
    if threads is None:
        argv[0] = str(context["serial_binary_path"])
    if threads is not None:
        if threads not in THREADS:
            raise QualificationError(f"thread candidate must be one of {THREADS}")
        argv += ["--block-threads", str(threads),
                 "--block-scratch-budget-bytes", str(BLOCK_SCRATCH_BUDGET_BYTES)]
    return argv


def _run_command(argv: list[str], *, timeout: float = 8.0) -> dict[str, Any]:
    try:
        process = subprocess.run(argv, capture_output=True, text=True,
                                 check=False, timeout=timeout)
        return {"argv": argv, "returncode": process.returncode,
                "stdout": process.stdout[-8000:], "stderr": process.stderr[-8000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"argv": argv, "returncode": None, "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc}"[-8000:]}


def _swap_mb(text: str) -> float | None:
    match = re.search(r"used\s*=\s*([0-9.]+)([MGT])", text)
    if match is None:
        return None
    return float(match.group(1)) * {"M": 1.0, "G": 1024.0, "T": 1024.0**2}[match.group(2)]


def _host_cpu() -> tuple[float | None, str | None]:
    argv = ["/bin/ps", "-axo", "%cpu="]
    try:
        process = subprocess.run(argv, capture_output=True, text=True,
                                 check=False, timeout=8)
        probe = {"argv": argv, "returncode": process.returncode,
                 "stdout": process.stdout, "stderr": process.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        probe = {"argv": argv, "returncode": None, "stdout": "",
                 "stderr": f"{type(exc).__name__}: {exc}"}
    if probe["returncode"] != 0:
        return None, _hash_value(probe)
    try:
        cores = sum(float(line.strip()) for line in probe["stdout"].splitlines()
                    if line.strip()) / 100.0
    except ValueError:
        return None, _hash_value(probe)
    return round(cores, 3), _hash_value(probe)


def resource_sample() -> dict[str, Any]:
    pressure = _run_command(["/usr/sbin/sysctl", "-n",
                             "kern.memorystatus_vm_pressure_level"])
    swap = _run_command(["/usr/sbin/sysctl", "-n", "vm.swapusage"])
    thermal = _run_command(["/usr/bin/pmset", "-g", "therm"])
    power = _run_command(["/usr/bin/pmset", "-g", "batt"])
    cpu_cores, cpu_probe_sha = _host_cpu()
    try:
        pressure_level = int(pressure["stdout"].strip()) if pressure["returncode"] == 0 else None
    except ValueError:
        pressure_level = None
    thermal_text = thermal["stdout"] + thermal["stderr"]
    disk = shutil.disk_usage(OUTPUT_ROOT.parent)
    return {
        "sampled_at": _now(), "pressure_level": pressure_level,
        "swap_used_mb": _swap_mb(swap["stdout"]),
        "thermal_nominal": ram_scheduler.thermal_output_ok(
            thermal["returncode"] if thermal["returncode"] is not None else -1,
            thermal_text,
        ),
        "ac_power": "AC Power" in power["stdout"],
        "disk_free_bytes": disk.free, "host_cpu_cores": cpu_cores,
        "logical_cpu_count": os.cpu_count(), "load_average": list(os.getloadavg()),
        "probe_sha256s": {
            "pressure": _hash_value(pressure), "swap": _hash_value(swap),
            "thermal": _hash_value(thermal), "power": _hash_value(power),
            "host_cpu": cpu_probe_sha,
        },
    }


def resource_errors(sample: Any, *, baseline_swap_mb: float | None = None,
                    require_idle: bool = True) -> list[str]:
    if not isinstance(sample, dict):
        return ["resource sample is not an object"]
    errors = []
    if sample.get("pressure_level") != 1:
        errors.append("memory pressure is not normal")
    swap = sample.get("swap_used_mb")
    if isinstance(swap, bool) or not isinstance(swap, (int, float)) \
            or not math.isfinite(float(swap)) or swap > MAX_SWAP_MB:
        errors.append("swap is unknown or above the clean calibration ceiling")
    if isinstance(baseline_swap_mb, (int, float)) and isinstance(swap, (int, float)) \
            and float(swap) - float(baseline_swap_mb) > MAX_SWAP_GROWTH_MB:
        errors.append("swap grew beyond the calibration allowance")
    if sample.get("thermal_nominal") is not True:
        errors.append("thermal state is not explicitly nominal")
    if sample.get("ac_power") is not True:
        errors.append("host is not explicitly on AC power")
    if not isinstance(sample.get("disk_free_bytes"), int) \
            or sample["disk_free_bytes"] < MIN_DISK_FREE_BYTES:
        errors.append("qualification disk headroom is below 10 GB")
    cpu, count = sample.get("host_cpu_cores"), sample.get("logical_cpu_count")
    idle_ceiling = max(2.0, 0.15 * float(count or 1))
    if require_idle and (not isinstance(cpu, (int, float)) or float(cpu) > idle_ceiling):
        errors.append("host is not idle enough for production calibration")
    return errors


def _owner_evidence(owners: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"pid": row.get("pid"), "matched_patterns": row.get("matched_patterns", []),
             "command_sha256": hashlib.sha256(
                 str(row.get("command", "")).encode()).hexdigest()}
            for row in owners]


def _acquire_lease() -> tuple[Any, dict[str, Any]]:
    HEAVY_LOCK.parent.mkdir(parents=True, exist_ok=True)
    handle = HEAVY_LOCK.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise QualificationError("shared heavy lease is already owned") from exc
    info = os.fstat(handle.fileno())
    evidence = {
        "path": str(HEAVY_LOCK.resolve()), "acquired_at": _now(),
        "holder_pid": os.getpid(), "device": int(info.st_dev), "inode": int(info.st_ino),
        "exclusive_nonblocking_flock": True, "inherited_by_child": False,
        "parent_retained_for_entire_child_lifetime": True,
    }
    evidence["lease_evidence_sha256"] = _hash_value(evidence)
    return handle, evidence


def _lease_errors(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return ["lease evidence is not an object"]
    required = {
        "path", "acquired_at", "holder_pid", "device", "inode",
        "exclusive_nonblocking_flock", "inherited_by_child",
        "parent_retained_for_entire_child_lifetime", "lease_evidence_sha256",
    }
    errors = []
    if set(value) != required or value.get("lease_evidence_sha256") != _hash_value(
            _without(value, "lease_evidence_sha256")):
        errors.append("lease evidence keys/self-hash differ")
    if value.get("exclusive_nonblocking_flock") is not True \
            or value.get("inherited_by_child") is not False \
            or value.get("parent_retained_for_entire_child_lifetime") is not True \
            or isinstance(value.get("holder_pid"), bool) \
            or not isinstance(value.get("holder_pid"), int) or value.get("holder_pid", 0) <= 0:
        errors.append("lease ownership contract differs")
    try:
        info = Path(value["path"]).stat()
        if (int(info.st_dev), int(info.st_ino)) != (value.get("device"), value.get("inode")):
            errors.append("lease file identity changed")
    except (OSError, KeyError, TypeError):
        errors.append("lease file cannot be verified")
    return errors


def _tree_rss(pid: int) -> tuple[int, set[int]]:
    argv = ["/bin/ps", "-axo", "pid=,ppid=,rss="]
    try:
        process = subprocess.run(argv, capture_output=True, text=True,
                                 check=False, timeout=8)
        probe = {"returncode": process.returncode, "stdout": process.stdout}
    except (OSError, subprocess.TimeoutExpired):
        probe = {"returncode": None, "stdout": ""}
    if probe["returncode"] != 0:
        return 0, {pid}
    rows = []
    for line in probe["stdout"].splitlines():
        try:
            child, parent, rss = (int(value) for value in line.split())
            rows.append((child, parent, rss * 1024))
        except (ValueError, TypeError):
            continue
    tree, changed = {pid}, True
    while changed:
        changed = False
        for child, parent, _ in rows:
            if parent in tree and child not in tree:
                tree.add(child); changed = True
    return sum(rss for child, _, rss in rows if child in tree), tree


def _fast_guard(baseline_swap_mb: float) -> dict[str, Any]:
    pressure = _run_command(["/usr/sbin/sysctl", "-n",
                             "kern.memorystatus_vm_pressure_level"], timeout=3)
    swap = _run_command(["/usr/sbin/sysctl", "-n", "vm.swapusage"], timeout=3)
    try:
        level = int(pressure["stdout"].strip()) if pressure["returncode"] == 0 else None
    except ValueError:
        level = None
    used = _swap_mb(swap["stdout"])
    ok = level == 1 and isinstance(used, (int, float)) \
        and used <= MAX_SWAP_MB and used - baseline_swap_mb <= MAX_SWAP_GROWTH_MB
    return {"sampled_at": _now(), "pressure_level": level, "swap_used_mb": used,
            "ok": ok, "probe_sha256": _hash_value({"pressure": pressure, "swap": swap})}


def _terminate_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass


def _run_binary(argv: list[str], env: dict[str, str], attempt: Path,
                *, baseline_swap_mb: float) -> dict[str, Any]:
    attempt.parent.mkdir(parents=True, exist_ok=True)
    attempt.mkdir(parents=True, exist_ok=False, mode=0o700)
    (attempt / "tmp").mkdir(mode=0o700)
    output, log_path, monitor_path = attempt / "output.strand", \
        attempt / "process.log", attempt / "monitor.jsonl"
    if str(output.resolve()) not in argv:
        raise QualificationError("execution argv does not bind its anchored output")
    started_wall, started_ns = _now(), time.monotonic_ns()
    peak, samples, trip = 0, 0, None
    last_guard = last_owner = last_thermal = 0.0
    with log_path.open("xb") as log, monitor_path.open("x", encoding="utf-8") as monitor:
        process = subprocess.Popen(
            argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            close_fds=True,
        )
        while process.poll() is None:
            now = time.monotonic()
            rss, tree = _tree_rss(process.pid); peak = max(peak, rss); samples += 1
            event: dict[str, Any] = {"sample": samples, "tree_rss_bytes": rss,
                                     "process_count": len(tree)}
            if now - last_guard >= GUARD_SECONDS:
                guard = _fast_guard(baseline_swap_mb); event["resource_guard"] = guard
                last_guard = now
                if not guard["ok"]:
                    trip = "memory-pressure-or-swap-guard"
            if trip is None and now - last_owner >= OWNER_SECONDS:
                owners = [row for row in spec_reentry_scaffold.active_heavy_owners()
                          if row.get("pid") not in tree]
                event["other_heavy_owners"] = _owner_evidence(owners)
                last_owner = now
                if owners:
                    trip = "another-heavy-owner-appeared"
            if trip is None and now - last_thermal >= THERMAL_SECONDS:
                thermal = _run_command(["/usr/bin/pmset", "-g", "therm"])
                green = ram_scheduler.thermal_output_ok(
                    thermal["returncode"] if thermal["returncode"] is not None else -1,
                    thermal["stdout"] + thermal["stderr"],
                )
                event["thermal"] = {"nominal": green, "probe_sha256": _hash_value(thermal)}
                last_thermal = now
                if not green:
                    trip = "thermal-guard"
            monitor.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
            monitor.flush()
            if trip is not None:
                _terminate_group(process); break
            time.sleep(POLL_SECONDS)
        returncode = process.poll()
        rss, _ = _tree_rss(process.pid); peak = max(peak, rss)
        log.flush(); os.fsync(log.fileno()); monitor.flush(); os.fsync(monitor.fileno())
    elapsed = (time.monotonic_ns() - started_ns) / 1e9
    os.chmod(log_path, 0o400); os.chmod(monitor_path, 0o400)
    result: dict[str, Any] = {
        "started_at": started_wall, "completed_at": _now(),
        "wall_seconds": elapsed, "peak_rss_bytes": peak,
        "rss_sample_count": samples, "returncode": returncode,
        "guard_trip": trip, "argv": argv, "argv_sha256": _hash_value(argv),
        "environment": env, "environment_sha256": _hash_value(env),
        "log": _artifact(log_path), "monitor": _artifact(monitor_path),
    }
    if returncode == 0 and trip is None and output.is_file():
        os.chmod(output, 0o400); result["output"] = _artifact(output)
    return result


def _attempt(root: Path, label: str) -> Path:
    return root / "attempts" / f"{label}-{time.time_ns()}-{secrets.token_hex(6)}"


def _failure(path: Path, *, context: dict[str, Any], kind: str,
             execution: dict[str, Any] | None, blockers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": FAILURE_SCHEMA, "version": VERSION, "created_at": _now(),
        "status": "failed", "kind": kind,
        "cell_id": context["cell_id"], "tier": context["tier"], "rate": context["rate"],
        "runtime_spec": context["runtime_spec"], "blockers": blockers,
        "execution": execution, "production_receipt_issued": False,
        "runtime_defaults_mutated": False, "live_queue_mutated": False,
    }
    doc["failure_sha256"] = _hash_value(doc)
    _write_exclusive_json(path / "failure.json", doc)


def _read_log(row: dict[str, Any]) -> str:
    try:
        return Path(row["path"]).read_text(encoding="utf-8", errors="replace")
    except (OSError, KeyError, TypeError):
        return ""


def validate_canonical(document: Any, context: dict[str, Any],
                       *, verify_files: bool = True) -> list[str]:
    errors = []
    if not isinstance(document, dict) or document.get("schema") != CANONICAL_SCHEMA \
            or document.get("version") != VERSION or document.get("status") != "pass":
        return ["canonical schema/version/status is invalid"]
    if document.get("receipt_sha256") != _hash_value(_without(document, "receipt_sha256")):
        errors.append("canonical self-hash differs")
    if document.get("source_binding_sha256") != context["source_binding_sha256"] \
            or document.get("source_binding") != context["source_binding"]:
        errors.append("canonical source binding differs")
    if document.get("binary") != context.get("serial_binary"):
        errors.append("canonical binary is not the independent hash-bound serial oracle")
    execution = document.get("execution", {})
    output = execution.get("output", {}) if isinstance(execution, dict) else {}
    expected_argv = build_argv(context, Path(str(output.get("path", "missing"))), threads=None)
    if execution.get("argv") != expected_argv \
            or execution.get("argv_sha256") != _hash_value(expected_argv):
        errors.append("canonical argv is not the exact serial oracle")
    env = execution.get("environment")
    if not isinstance(env, dict) or env.get("STRAND_NO_GPU") != "1" \
            or execution.get("environment_sha256") != _hash_value(env) \
            or any(env.get(key) != "1" for key in THREAD_ENV_KEYS):
        errors.append("canonical CPU-only environment differs")
    if not isinstance(execution.get("wall_seconds"), (int, float)) \
            or execution.get("wall_seconds", 0) <= 0 \
            or not isinstance(execution.get("peak_rss_bytes"), int) \
            or execution.get("peak_rss_bytes", 0) <= 0 \
            or execution.get("returncode") != 0 or execution.get("guard_trip") is not None:
        errors.append("canonical physical measurement is invalid")
    if document.get("cpu_only") is not True or document.get("synthetic") is not False \
            or document.get("real_production_tensor") is not True:
        errors.append("canonical is not a real CPU production tensor receipt")
    resources = document.get("resource_evidence", {})
    if resource_errors(resources.get("before"), require_idle=True) \
            or resource_errors(resources.get("after"),
                               baseline_swap_mb=resources.get("before", {}).get("swap_used_mb"),
                               require_idle=True):
        errors.append("canonical resource evidence is not green")
    errors.extend(_lease_errors(document.get("lease_evidence")))
    if verify_files:
        for name in ("runtime_spec", "source_seal", "binary"):
            if not _artifact_matches(document.get(name)):
                errors.append(f"canonical {name} artifact changed")
        for name in ("output", "log", "monitor"):
            if not _artifact_matches(execution.get(name)):
                errors.append(f"canonical execution {name} artifact changed")
        if "Metal encode: ON" in _read_log(execution.get("log", {})):
            errors.append("canonical log reports Metal execution")
    return errors


def validate_receipt(document: Any, context: dict[str, Any], *, threads: int,
                     verify_files: bool = True) -> list[str]:
    errors = []
    if not isinstance(document, dict) or document.get("schema") != RECEIPT_SCHEMA:
        return ["candidate receipt schema is invalid"]
    if document.get("receipt_sha256") != _hash_value(_without(document, "receipt_sha256")):
        errors.append("candidate receipt self-hash differs")
    if document.get("status") != "pass" or document.get("scope") != "production" \
            or document.get("synthetic") is not False \
            or document.get("threads") != threads \
            or document.get("tier") != context["tier"] \
            or document.get("rate") != context["rate"]:
        errors.append("candidate production identity differs")
    if document.get("binary_sha256") != context["binary"]["sha256"] \
            or document.get("source_sha256") != context["source_binding_sha256"] \
            or document.get("source_binding") != context["source_binding"] \
            or document.get("source_binding_sha256") != context["source_binding_sha256"]:
        errors.append("candidate binary/source binding differs")
    execution = document.get("candidate", {})
    output = execution.get("output", {}) if isinstance(execution, dict) else {}
    expected_argv = build_argv(context, Path(str(output.get("path", "missing"))), threads=threads)
    if execution.get("argv") != expected_argv \
            or execution.get("argv_sha256") != _hash_value(expected_argv):
        errors.append("candidate argv differs from exact codec/thread contract")
    env = execution.get("environment")
    if not isinstance(env, dict) or env.get("STRAND_NO_GPU") != "1" \
            or execution.get("environment_sha256") != _hash_value(env) \
            or any(env.get(key) != "1" for key in THREAD_ENV_KEYS):
        errors.append("candidate CPU-only environment differs")
    canonical = document.get("canonical", {})
    if document.get("canonical_output_sha256") != canonical.get("output", {}).get("sha256") \
            or document.get("output_sha256") != output.get("sha256") \
            or document.get("output_sha256") != document.get("canonical_output_sha256") \
            or document.get("exact_output") is not True:
        errors.append("candidate does not prove byte-exact canonical output")
    if document.get("wall_seconds") != execution.get("wall_seconds") \
            or document.get("peak_rss_bytes") != execution.get("peak_rss_bytes") \
            or not isinstance(document.get("wall_seconds"), (int, float)) \
            or document.get("wall_seconds", 0) <= 0 \
            or not isinstance(document.get("peak_rss_bytes"), int) \
            or document.get("peak_rss_bytes", 0) <= 0 \
            or execution.get("returncode") != 0 or execution.get("guard_trip") is not None:
        errors.append("candidate physical measurement is invalid")
    if document.get("scratch_budget_bytes") != BLOCK_SCRATCH_BUDGET_BYTES \
            or document.get("mode") != "block_parallel" \
            or document.get("cpu_only") is not True \
            or document.get("real_production_tensor") is not True \
            or document.get("runtime_defaults_mutated") is not False \
            or document.get("live_queue_mutated") is not False \
            or document.get("source_deletion_permitted") is not False:
        errors.append("candidate execution/safety contract differs")
    resources = document.get("resource_evidence", {})
    if resource_errors(resources.get("before"), require_idle=True) \
            or resource_errors(resources.get("after"),
                               baseline_swap_mb=resources.get("before", {}).get("swap_used_mb"),
                               require_idle=True):
        errors.append("candidate resource evidence is not green")
    errors.extend(_lease_errors(document.get("lease_evidence")))
    if verify_files:
        for name in ("runtime_spec", "source_seal", "binary"):
            if not _artifact_matches(document.get(name)):
                errors.append(f"candidate {name} artifact changed")
        for name in ("output", "log", "monitor"):
            if not _artifact_matches(execution.get(name)):
                errors.append(f"candidate execution {name} artifact changed")
        for name in ("receipt", "output"):
            if not _artifact_matches(canonical.get(name)):
                errors.append(f"candidate canonical {name} artifact changed")
        try:
            canonical_document = _read_json(Path(canonical["receipt"]["path"]))
            canonical_errors = validate_canonical(
                canonical_document, context, verify_files=True,
            )
            if canonical_errors:
                errors.append("bound canonical receipt is invalid: " + "; ".join(canonical_errors))
            elif canonical_document.get("execution", {}).get("output") != canonical.get("output"):
                errors.append("candidate canonical output binding differs from canonical receipt")
        except (KeyError, TypeError, QualificationError) as exc:
            errors.append(f"bound canonical receipt cannot be validated: {exc}")
        log_text = _read_log(execution.get("log", {}))
        phrase = (f"feature-gated block-parallel CPU encode: {threads} block workers, "
                  f"{BLOCK_SCRATCH_BUDGET_BYTES // (1024**2)} MiB aggregate Viterbi scratch cap")
        if phrase not in log_text or "Metal encode: ON" in log_text:
            errors.append("candidate log does not prove the exact CPU block worker count")
    try:
        contract = _load_contract()
        contract.validate_receipt(
            document, expected_binary_sha256=context["binary"]["sha256"],
            allowed_threads=THREADS,
        )
    except Exception as exc:
        errors.append(f"vendor receipt contract refused candidate: {exc}")
    return errors


def _ensure_owner_free() -> list[dict[str, Any]]:
    owners = spec_reentry_scaffold.active_heavy_owners()
    if owners:
        raise QualificationError(
            f"physical calibration requires zero other heavy owners; observed {len(owners)}"
        )
    return owners


def run_canonical(context: dict[str, Any], lease: dict[str, Any],
                  *, output_root: Path = OUTPUT_ROOT) -> tuple[dict[str, Any], bool]:
    final = canonical_path(context, output_root=output_root)
    if final.exists():
        document = _read_json(final)
        errors = validate_canonical(document, context, verify_files=True)
        if errors:
            raise QualificationError("existing canonical is invalid: " + "; ".join(errors))
        return document, True
    _ensure_owner_free()
    before = resource_sample(); blockers = resource_errors(before, require_idle=True)
    if blockers:
        raise QualificationError("canonical admission blocked: " + "; ".join(blockers))
    attempt = _attempt(cell_root(context, output_root=output_root), "serial")
    output = attempt / "output.strand"; env = _launch_env(attempt / "tmp")
    argv = build_argv(context, output, threads=None)
    try:
        execution = _run_binary(argv, env, attempt, baseline_swap_mb=before["swap_used_mb"])
    except OSError as exc:
        _failure(attempt, context=context, kind="canonical-launch", execution=None,
                 blockers=[f"launch failed: {type(exc).__name__}: {exc}"])
        raise QualificationError(f"canonical launch failed: {exc}") from exc
    after = resource_sample()
    blockers = []
    if execution.get("returncode") != 0 or execution.get("guard_trip") is not None \
            or not isinstance(execution.get("output"), dict):
        blockers.append("serial oracle process failed or tripped a guard")
    blockers += resource_errors(after, baseline_swap_mb=before["swap_used_mb"], require_idle=True)
    if blockers:
        _failure(attempt, context=context, kind="canonical", execution=execution,
                 blockers=blockers)
        raise QualificationError("canonical failed: " + "; ".join(blockers))
    document = {
        "schema": CANONICAL_SCHEMA, "version": VERSION, "status": "pass",
        "created_at": _now(), "scope": "production", "synthetic": False,
        "tier": context["tier"], "rate": context["rate"],
        "cell_id": context["cell_id"], "branch": context["branch"],
        "runtime_spec": context["runtime_spec"], "source_seal": context["source_seal"],
        "binary": context["serial_binary"], "source_binding": context["source_binding"],
        "source_binding_sha256": context["source_binding_sha256"],
        "execution": execution, "resource_evidence": {"before": before, "after": after},
        "lease_evidence": lease, "cpu_only": True, "real_production_tensor": True,
        "runtime_defaults_mutated": False, "live_queue_mutated": False,
        "source_deletion_permitted": False,
    }
    document["receipt_sha256"] = _hash_value(document)
    _write_exclusive_json(final, document)
    return document, False


def run_candidate(context: dict[str, Any], canonical: dict[str, Any], threads: int,
                  lease: dict[str, Any], *, output_root: Path = OUTPUT_ROOT) \
        -> tuple[dict[str, Any], bool]:
    final = receipt_path(context, threads, output_root=output_root)
    if final.exists():
        document = _read_json(final)
        errors = validate_receipt(document, context, threads=threads, verify_files=True)
        if errors:
            raise QualificationError("existing candidate is invalid: " + "; ".join(errors))
        return document, True
    canonical_file = canonical_path(context, output_root=output_root)
    canonical_errors = validate_canonical(canonical, context, verify_files=True)
    if canonical_errors:
        raise QualificationError("canonical became invalid: " + "; ".join(canonical_errors))
    _ensure_owner_free()
    before = resource_sample(); blockers = resource_errors(before, require_idle=True)
    if blockers:
        raise QualificationError("candidate admission blocked: " + "; ".join(blockers))
    attempt = _attempt(cell_root(context, output_root=output_root), f"threads-{threads:02d}")
    output = attempt / "output.strand"; env = _launch_env(attempt / "tmp")
    argv = build_argv(context, output, threads=threads)
    try:
        execution = _run_binary(argv, env, attempt, baseline_swap_mb=before["swap_used_mb"])
    except OSError as exc:
        _failure(attempt, context=context, kind=f"candidate-{threads}-launch", execution=None,
                 blockers=[f"launch failed: {type(exc).__name__}: {exc}"])
        raise QualificationError(f"candidate launch failed: {exc}") from exc
    after = resource_sample(); blockers = []
    if execution.get("returncode") != 0 or execution.get("guard_trip") is not None \
            or not isinstance(execution.get("output"), dict):
        blockers.append("candidate process failed or tripped a guard")
    blockers += resource_errors(after, baseline_swap_mb=before["swap_used_mb"], require_idle=True)
    exact = isinstance(execution.get("output"), dict) and \
        execution["output"]["sha256"] == canonical["execution"]["output"]["sha256"]
    if not exact:
        blockers.append("candidate output is not byte-identical to serial canonical")
    log_text = _read_log(execution.get("log", {}))
    phrase = (f"feature-gated block-parallel CPU encode: {threads} block workers, "
              f"{BLOCK_SCRATCH_BUDGET_BYTES // (1024**2)} MiB aggregate Viterbi scratch cap")
    if phrase not in log_text or "Metal encode: ON" in log_text:
        blockers.append("process log does not prove exact CPU block-parallel dispatch")
    if blockers:
        _failure(attempt, context=context, kind=f"candidate-{threads}",
                 execution=execution, blockers=blockers)
        raise QualificationError("candidate failed: " + "; ".join(blockers))
    canonical_binding = {
        "receipt": _artifact(canonical_file),
        "output": canonical["execution"]["output"],
    }
    document = {
        "schema": RECEIPT_SCHEMA, "version": VERSION, "status": "pass",
        "scope": "production", "synthetic": False, "created_at": _now(),
        "tier": context["tier"], "rate": context["rate"], "threads": threads,
        "binary_sha256": context["binary"]["sha256"],
        "source_sha256": context["source_binding_sha256"],
        "canonical_output_sha256": canonical_binding["output"]["sha256"],
        "output_sha256": execution["output"]["sha256"], "exact_output": True,
        "wall_seconds": execution["wall_seconds"],
        "peak_rss_bytes": execution["peak_rss_bytes"],
        "scratch_budget_bytes": BLOCK_SCRATCH_BUDGET_BYTES,
        "mode": "block_parallel", "cell_id": context["cell_id"],
        "branch": context["branch"], "runtime_spec": context["runtime_spec"],
        "source_seal": context["source_seal"], "binary": context["binary"],
        "source_binding": context["source_binding"],
        "source_binding_sha256": context["source_binding_sha256"],
        "canonical": canonical_binding, "candidate": execution,
        "resource_evidence": {"before": before, "after": after},
        "lease_evidence": lease, "cpu_only": True, "real_production_tensor": True,
        "runtime_defaults_mutated": False, "live_queue_mutated": False,
        "source_deletion_permitted": False,
    }
    document["receipt_sha256"] = _hash_value(document)
    _write_exclusive_json(final, document)
    errors = validate_receipt(document, context, threads=threads, verify_files=True)
    if errors:
        raise QualificationError("new candidate failed finalization: " + "; ".join(errors))
    return document, False


def matrix_id(contexts: list[dict[str, Any]]) -> str:
    return _hash_value([{
        "tier": row["tier"], "rate": row["rate"],
        "runtime_spec_sha256": row["runtime_spec"]["sha256"],
    } for row in sorted(contexts, key=lambda row: (row["tier"], row["rate"]))])


def validate_qualification(document: Any, contexts: list[dict[str, Any]],
                           *, profile_artifact: dict[str, Any],
                           receipt_artifacts: list[dict[str, Any]]) -> list[str]:
    errors = []
    expected_matrix = matrix_id(contexts)
    binaries = {row["binary"]["sha256"] for row in contexts}
    if not isinstance(document, dict) or document.get("schema") != QUALIFICATION_SCHEMA \
            or document.get("version") != VERSION or document.get("status") != "qualified":
        return ["qualification schema/version/status is invalid"]
    if document.get("qualification_sha256") != _hash_value(
            _without(document, "qualification_sha256")):
        errors.append("qualification self-hash differs")
    if document.get("matrix_id") != expected_matrix \
            or document.get("exact_tier_rate_count") != len(contexts) \
            or document.get("required_threads") != list(THREADS) \
            or document.get("binary_sha256") != (next(iter(binaries)) if len(binaries) == 1 else None):
        errors.append("qualification matrix/binary identity differs")
    if document.get("profile") != profile_artifact \
            or document.get("receipts") != receipt_artifacts:
        errors.append("qualification profile/receipt inventory differs")
    if document.get("runner") != _artifact(Path(__file__)) \
            or document.get("vendor_contract") != _artifact(CONTRACT_PATH):
        errors.append("qualification program sources changed")
    if document.get("all_receipts_strictly_revalidated") is not True \
            or document.get("automatic_runtime_promotion_permitted") is not False \
            or document.get("runtime_defaults_mutated") is not False \
            or document.get("live_queue_mutated") is not False:
        errors.append("qualification safety contract differs")
    return errors


def build_profile(contexts: list[dict[str, Any]], *, output_root: Path = OUTPUT_ROOT,
                  rss_limit_bytes: int = RSS_LIMIT_BYTES) -> dict[str, Any]:
    contexts = sorted(contexts, key=lambda row: (row["tier"], row["rate"]))
    if len({(row["tier"], row["rate"]) for row in contexts}) != len(contexts):
        raise QualificationError("profile contexts repeat an exact tier/rate")
    receipts = []
    for context in contexts:
        for threads in THREADS:
            path = receipt_path(context, threads, output_root=output_root)
            if not path.is_file():
                raise QualificationError(f"missing exact production receipt: {path}")
            document = _read_json(path)
            errors = validate_receipt(document, context, threads=threads, verify_files=True)
            if errors:
                raise QualificationError(f"invalid production receipt {path}: " + "; ".join(errors))
            receipts.append(path)
    binaries = {row["binary"]["sha256"] for row in contexts}
    if len(binaries) != 1:
        raise QualificationError("matrix runtime specs do not share one exact binary")
    generation = Path(output_root) / "profiles" / matrix_id(contexts)
    profile_path = generation / "thread-profile.json"
    contract = _load_contract(); binary_sha = next(iter(binaries))
    if profile_path.exists():
        profile = _read_json(profile_path)
    else:
        profile = contract.build_profile(
            receipts, expected_binary_sha256=binary_sha,
            rss_limit_bytes=rss_limit_bytes, required_threads=THREADS,
        )
        if profile.get("status") != "qualified" or profile.get("entry_count") != len(contexts):
            raise QualificationError("vendor contract did not qualify the complete matrix")
        _write_exclusive_json(profile_path, profile)
    expected_entry_keys = {
        json.dumps([row["tier"], row["rate"]], separators=(",", ":"), ensure_ascii=False)
        for row in contexts
    }
    if profile.get("status") != "qualified" \
            or profile.get("expected_binary_sha256") != binary_sha \
            or profile.get("rss_limit_bytes") != rss_limit_bytes \
            or profile.get("required_threads") != list(THREADS) \
            or profile.get("entry_count") != len(contexts) \
            or set(profile.get("entries", {})) != expected_entry_keys:
        raise QualificationError("existing vendor profile identity/matrix differs")
    for context in contexts:
        contract.verify_selection(profile, tier=context["tier"], rate=context["rate"],
                                  binary_sha256=binary_sha)
    qualification_path = generation / "qualification.json"
    profile_artifact = _artifact(profile_path)
    receipt_artifacts = [_artifact(path) for path in receipts]
    if qualification_path.exists():
        qualification = _read_json(qualification_path)
        errors = validate_qualification(
            qualification, contexts, profile_artifact=profile_artifact,
            receipt_artifacts=receipt_artifacts,
        )
        if errors:
            raise QualificationError("existing qualification packet is invalid: "
                                     + "; ".join(errors))
        return qualification
    qualification = {
        "schema": QUALIFICATION_SCHEMA, "version": VERSION, "status": "qualified",
        "created_at": _now(), "matrix_id": matrix_id(contexts),
        "exact_tier_rate_count": len(contexts), "required_threads": list(THREADS),
        "profile": profile_artifact, "binary_sha256": binary_sha,
        "receipts": receipt_artifacts,
        "runner": _artifact(Path(__file__)), "vendor_contract": _artifact(CONTRACT_PATH),
        "all_receipts_strictly_revalidated": True,
        "automatic_runtime_promotion_permitted": False,
        "runtime_defaults_mutated": False, "live_queue_mutated": False,
    }
    qualification["qualification_sha256"] = _hash_value(qualification)
    _write_exclusive_json(qualification_path, qualification)
    return qualification


def status(contexts: list[dict[str, Any]], *, output_root: Path = OUTPUT_ROOT) -> dict[str, Any]:
    cells, measured, missing, invalid = [], [], 0, 0
    for context in sorted(contexts, key=lambda row: (row["tier"], row["rate"])):
        row = {"tier": context["tier"], "rate": context["rate"],
               "cell_id": context["cell_id"], "canonical": "missing", "candidates": {}}
        canonical_file = canonical_path(context, output_root=output_root)
        if canonical_file.exists():
            errors = validate_canonical(_read_json(canonical_file), context, verify_files=True)
            row["canonical"] = "valid" if not errors else "invalid"
            invalid += bool(errors)
        else:
            missing += 1
        for threads in THREADS:
            path = receipt_path(context, threads, output_root=output_root)
            if not path.exists():
                row["candidates"][str(threads)] = "missing"; missing += 1; continue
            document = _read_json(path)
            errors = validate_receipt(document, context, threads=threads, verify_files=True)
            row["candidates"][str(threads)] = "valid" if not errors else "invalid"
            if errors:
                invalid += 1
            else:
                measured.append(float(document["wall_seconds"]))
        cells.append(row)
    estimate = None
    if measured:
        ordered = sorted(measured); median = ordered[len(ordered) // 2]
        estimate = round(median * missing, 3)
    return {
        "schema": "hawking.doctor_v5_qwen_thread_profile_status.v1",
        "execution_default": "off", "matrix_id": matrix_id(contexts),
        "exact_tier_rate_count": len(contexts), "required_threads": list(THREADS),
        "arm_count_total": len(contexts) * 5, "arm_count_missing": missing,
        "invalid_evidence_count": invalid, "measured_candidate_count": len(measured),
        "estimated_remaining_seconds": estimate,
        "estimate_basis": ("median-valid-same-runner-candidate-times-missing-arms"
                           if estimate is not None else "unknown-until-first-physical-candidate"),
        "cells": cells,
        "physical_command_required": "run --execute-physical",
        "profile_path": str((Path(output_root) / "profiles" / matrix_id(contexts) /
                             "thread-profile.json").resolve()),
    }


def _paths_from_matrix(path: Path) -> list[Path]:
    document = _read_json(path); errors = validate_matrix(document, verify_files=True)
    if errors:
        raise QualificationError("matrix is invalid: " + "; ".join(errors))
    return [Path(row["runtime_spec"]["path"]) for row in document["specs"]]


def _target_paths(args: argparse.Namespace) -> list[Path]:
    if getattr(args, "runtime_spec", None):
        return [Path(row).resolve(strict=True) for row in args.runtime_spec]
    if getattr(args, "runtime_spec_dir", None):
        return discover_specs(Path(args.runtime_spec_dir))
    if getattr(args, "matrix", None):
        return _paths_from_matrix(Path(args.matrix))
    raise QualificationError("no runtime target was supplied")


def _add_target(parser: argparse.ArgumentParser) -> None:
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--runtime-spec", action="append")
    target.add_argument("--runtime-spec-dir")
    target.add_argument("--matrix")


def _parse_threads(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(child) for child in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("threads must be comma-separated integers") from exc
    if not result or len(result) != len(set(result)) or any(child not in THREADS for child in result):
        raise argparse.ArgumentTypeError(f"threads must be unique values drawn from {THREADS}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("make-matrix"); make.add_argument("--runtime-spec-dir", required=True)
    show = commands.add_parser("status"); _add_target(show)
    run = commands.add_parser("run"); _add_target(run)
    run.add_argument("--threads", type=_parse_threads, required=True)
    run.add_argument("--execute-physical", action="store_true")
    build = commands.add_parser("build-profile"); _add_target(build)
    build.add_argument("--rss-limit-bytes", type=int, default=RSS_LIMIT_BYTES)
    verify = commands.add_parser("verify-receipt")
    verify.add_argument("--runtime-spec", required=True); verify.add_argument("--receipt", required=True)
    args = parser.parse_args(argv)
    if args.command == "make-matrix":
        document = build_matrix_manifest(discover_specs(Path(args.runtime_spec_dir)))
        path = OUTPUT_ROOT / "matrices" / f"{document['matrix_sha256']}.json"
        if path.exists():
            if _read_json(path) != document:
                raise QualificationError("existing matrix path contains different evidence")
        else:
            _write_exclusive_json(path, document)
        print(json.dumps({"status": "ready", "matrix": _artifact(path)}, sort_keys=True)); return 0
    if args.command == "verify-receipt":
        context = load_context(Path(args.runtime_spec)); document = _read_json(Path(args.receipt))
        threads = document.get("threads")
        errors = validate_receipt(document, context, threads=threads, verify_files=True) \
            if threads in THREADS else ["receipt thread identity is invalid"]
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
        return 0 if not errors else 2
    if args.command == "run" and not args.execute_physical:
        raise QualificationError("physical execution is default-off; pass --execute-physical")
    contexts = [load_context(path) for path in _target_paths(args)]
    if len({(row["tier"], row["rate"]) for row in contexts}) != len(contexts):
        raise QualificationError("targets repeat an exact tier/rate")
    if args.command == "status":
        print(json.dumps(status(contexts), indent=2, sort_keys=True)); return 0
    if args.command == "build-profile":
        if args.rss_limit_bytes <= 0:
            raise QualificationError("RSS limit must be positive")
        result = build_profile(contexts, rss_limit_bytes=args.rss_limit_bytes)
        print(json.dumps(result, indent=2, sort_keys=True)); return 0
    handle, lease = _acquire_lease()
    try:
        results = []
        for context in sorted(contexts, key=lambda row: (row["tier"], row["rate"])):
            canonical, skipped = run_canonical(context, lease)
            results.append({"cell_id": context["cell_id"], "arm": "serial",
                            "skipped_valid": skipped})
            for threads in args.threads:
                _, skipped = run_candidate(context, canonical, threads, lease)
                results.append({"cell_id": context["cell_id"], "arm": threads,
                                "skipped_valid": skipped})
        print(json.dumps({"status": "complete", "results": results,
                          "next": "build-profile after all 8/12/16/20 receipts exist"},
                         indent=2, sort_keys=True))
        return 0
    finally:
        handle.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (QualificationError, OSError, ValueError, source_seal.SourceSealError) as exc:
        print(f"doctor_v5_qwen_thread_profile_runner: {exc}", file=sys.stderr)
        raise SystemExit(2)
