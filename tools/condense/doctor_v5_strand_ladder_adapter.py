#!/usr/bin/env python3.12
"""Generic Doctor-v5 ABI adapter for the Qwen2.5 STRAND ladder runtime.

The adapter translates a source-bound campaign cell into the internal sharded
worker dialect, runs it in-process under the inherited heavy-work lease, and
seals the generic result/checkpoint/receipt.  It never promotes a nominal
projection rate to an all-in model-rate claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
from types import ModuleType
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ABI_PATH = HERE / "doctor_v5_adapter_abi.py"
WORKER_PATH = HERE / "doctor_v5_strand_ladder_worker.py"
BASE_HELPER_PATH = HERE / "doctor_v5_pass_b_worker.py"
PARAMETER_MODULE_PATH = HERE / "doctor_v5_parameter_manifest.py"
EVALUATOR_PATH = HERE / "doctor_v5_sharded_eval.py"
PASS_A_INDEX = ROOT / "reports/condense/doctor_v5_scale/index.json"
QUANTIZER = ROOT / "vendor/strand-quant/target/release/quantize-model"
ATTESTOR = ROOT / "vendor/strand-decode-kernel/target/release/attest-strand"
DECODER = ROOT / "vendor/strand-decode-kernel/target/release/archive-to-safetensors"
ADAPTER_ID = "doctor-v5-strand-ladder-qwen25-dense"
ADAPTER_VERSION = "1"
OPERATION = "condense_control"
MODEL_FAMILY = "qwen2.5-dense"
BACKEND = "apple-cpu-strand"
SPEC_SCHEMA = "hawking.doctor_v5_strand_ladder_spec.v1"
INTERNAL_DIR = "strand_ladder"
MAX_JSON_BYTES = 64 * 1024 * 1024
SHA_RE = re.compile(r"[0-9a-f]{64}")
PARAMETER_MANIFEST_DIR = ROOT / "reports/condense/doctor_v5_pass_b/parameter_manifests"


class AdapterError(RuntimeError):
    pass


def _load_module(name: str, path: Path) -> ModuleType:
    if not path.is_file() or path.is_symlink():
        raise AdapterError(f"required module is missing or symlinked: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AdapterError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise AdapterError(f"cannot open bound file {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise AdapterError(f"bound file is not regular: {path}")
        digest, size = hashlib.sha256(), 0
        while True:
            block = os.read(fd, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block); size += len(block)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != \
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) \
                or size != after.st_size:
            raise AdapterError(f"bound file changed while hashing: {path}")
        return digest.hexdigest(), size
    finally:
        os.close(fd)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JSON_BYTES:
            raise AdapterError(f"invalid JSON file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"JSON root must be an object: {path}")
    return value


def _workspace_path(raw: Any, *, must_exist: bool = True) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise AdapterError("path must be a non-empty string")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(ROOT.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise AdapterError(f"path is missing or outside workspace: {raw!r}") from exc
    if must_exist and candidate.is_symlink():
        raise AdapterError(f"symlinked input forbidden: {raw!r}")
    return resolved


def _errors(value: Any, context: str) -> None:
    if not isinstance(value, list) or any(not isinstance(row, str) for row in value):
        raise AdapterError(f"{context} validator returned invalid data")
    if value:
        raise AdapterError(f"{context} validation failed: " + "; ".join(value))


def _entry(registry: dict[str, Any], adapter_id: str) -> dict[str, Any]:
    rows = registry.get("entries")
    matches = [row for row in rows if isinstance(row, dict)
               and row.get("adapter_id") == adapter_id] if isinstance(rows, list) else []
    if len(matches) != 1:
        raise AdapterError("registry does not contain exactly one ladder adapter")
    return matches[0]


def _artifact(path: Path) -> dict[str, Any]:
    digest, size = _hash_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def _tool(path: Path) -> tuple[str, str]:
    resolved = path.resolve(strict=True)
    try: resolved.relative_to(ROOT.resolve())
    except ValueError as exc: raise AdapterError(f"tool is outside workspace: {path}") from exc
    if path.is_symlink():
        raise AdapterError(f"tool is symlinked: {path}")
    return str(resolved), _hash_file(resolved)[0]


def _python() -> tuple[str, str]:
    path = Path(sys.executable)
    if path.is_symlink():
        raise AdapterError("running Python interpreter must not be a symlink")
    resolved = path.resolve(strict=True)
    return str(resolved), _hash_file(resolved)[0]


def _assert_input(request: dict[str, Any], path: Path, digest: str, size: int) -> None:
    rows = request.get("inputs")
    matches = [row for row in rows if isinstance(row, dict)
               and Path(str(row.get("path", ""))).resolve(strict=False) == path]
    if len(matches) != 1 or matches[0].get("sha256") != digest \
            or matches[0].get("bytes") != size:
        raise AdapterError(f"canonical input inventory does not bind {path}")


def _normalized_inputs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize paths after the ABI has already hash-verified every request input."""
    rows = spec.get("inputs")
    if not isinstance(rows, list) or not rows:
        raise AdapterError("typed ladder spec has no inputs")
    result: list[dict[str, Any]] = []
    roles: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"role", "path", "sha256", "bytes"}:
            raise AdapterError("typed ladder input keys are invalid")
        if not isinstance(row["role"], str) or not row["role"] or row["role"] in roles:
            raise AdapterError("typed ladder input role is invalid/duplicate")
        roles.add(row["role"])
        path = _workspace_path(row["path"])
        result.append({**row, "path": str(path)})
    return sorted(result, key=lambda row: row["role"])


def _input_row(role: str, path: Path, *, known_sha: str | None = None,
               known_bytes: int | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if known_sha is None or known_bytes is None:
        digest, size = _hash_file(resolved)
    else:
        digest, size = known_sha, known_bytes
    if not isinstance(digest, str) or SHA_RE.fullmatch(digest) is None \
            or not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise AdapterError(f"invalid known input identity for {resolved}")
    return {"role": role, "path": str(resolved), "sha256": digest, "bytes": size}


def build_spec(*, label: str, rate_id: str, cell_id: str,
               cell_identity_sha256: str, program_spec_sha256: str,
               resource_admission_sha256: str, evaluation_mode: str,
               disk_reserve_bytes: int, scratch_budget_bytes: int,
               threads: int, output_path: Path) -> dict[str, Any]:
    """Materialize one exact runtime spec without hashing model-sized shards again."""
    worker = _load_module("doctor_v5_strand_ladder_worker_spec_builder", WORKER_PATH)
    if label not in worker.SUPPORTED_LABELS or rate_id not in worker.CANONICAL_RATES:
        raise AdapterError("spec builder accepts reviewed Qwen2.5 labels/canonical rates only")
    for name, value in (("cell_identity_sha256", cell_identity_sha256),
                        ("program_spec_sha256", program_spec_sha256),
                        ("resource_admission_sha256", resource_admission_sha256)):
        if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
            raise AdapterError(f"{name} is invalid")
    if not isinstance(cell_id, str) or not cell_id:
        raise AdapterError("cell_id is invalid")
    if evaluation_mode == "auto":
        evaluation_mode = "resident" if label in worker.RESIDENT_LABELS else "deferred"
    if evaluation_mode not in {"resident", "deferred"} \
            or label not in worker.RESIDENT_LABELS and evaluation_mode != "deferred":
        raise AdapterError("evaluation mode violates the dense-reconstruction admission gate")
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
           for v in (disk_reserve_bytes, scratch_budget_bytes, threads)) \
            or disk_reserve_bytes < 50_000_000_000 or not 1 <= threads <= 32:
        raise AdapterError("resource values are invalid")

    parameter_path = PARAMETER_MANIFEST_DIR / f"{label}.json"
    parameter = _load_json(parameter_path)
    census_path = _workspace_path(parameter.get("census_path"))
    census = _load_json(census_path)
    model_dir = _workspace_path(parameter.get("model_dir"))
    if census.get("status") != "complete" or census.get("label") != label:
        raise AdapterError("completed matching census is required")
    inputs = [
        _input_row("adapter_abi", ABI_PATH), _input_row("adapter_source", Path(__file__)),
        _input_row("worker", WORKER_PATH), _input_row("base_helper", BASE_HELPER_PATH),
        _input_row("parameter_module", PARAMETER_MODULE_PATH),
        _input_row("evaluator", EVALUATOR_PATH), _input_row("quantizer", QUANTIZER),
        _input_row("attestor", ATTESTOR), _input_row("decoder", DECODER),
        _input_row("pass_a_index", PASS_A_INDEX),
        _input_row("parameter_manifest", parameter_path),
        _input_row("source_census", census_path),
    ]
    for row in census.get("source", {}).get("shards", []):
        ordinal, name = row.get("ordinal"), row.get("name")
        if not isinstance(ordinal, int) or not isinstance(name, str):
            raise AdapterError("census shard identity is invalid")
        inputs.append(_input_row(
            f"source_shard:{ordinal:05d}", model_dir / name,
            known_sha=row.get("file_sha256"), known_bytes=row.get("bytes"),
        ))
    metadata_names = set(worker.METADATA_NAMES)
    for row in census.get("source", {}).get("auxiliary_files", []):
        name = row.get("name") if isinstance(row, dict) else None
        if name not in metadata_names:
            continue
        inputs.append(_input_row(
            f"model_metadata:{name}", model_dir / name,
            known_sha=row.get("sha256"), known_bytes=row.get("bytes"),
        ))
    paths = [row["path"] for row in inputs]
    if len(paths) != len(set(paths)):
        raise AdapterError("spec input inventory contains duplicate paths")
    geometry = worker._rate_geometry(rate_id)
    vector = geometry["vector_dim"] > 1
    codec = {
        "rate_id": rate_id, "artifact_mode": geometry["artifact_mode"],
        "symbol_bits": geometry["symbol_bits"], "vector_dim": geometry["vector_dim"],
        "block_len": geometry["block_len"], "tensor_scope": "all-2d",
        "quality": True, "rht_cols": True,
        "outlier_channel_pct": 0 if vector else 1,
        "outlier_bits": 8, "sdsq_sideinfo": True,
        "c2f_outl": not vector, "ragged_v2": True,
        "allow_over_ceiling_control": True, "learned_codebook": False,
        "adaptive_scales": geometry["adaptive_scales"],
    }
    target = worker.CANONICAL_RATES[rate_id]
    spec = {
        "schema": SPEC_SCHEMA, "label": label,
        "campaign_binding": {"cell_id": cell_id,
                             "cell_identity_sha256": cell_identity_sha256,
                             "branch": "codec_control", "target_rate_id": rate_id,
                             "target_rate_bpw": float(target), "label": label},
        "adapter_id": ADAPTER_ID, "operation": OPERATION,
        "model_family": MODEL_FAMILY, "backend": BACKEND, "codec": codec,
        "evaluation": {"mode": evaluation_mode, "retain_dense_reconstruction": False},
        "doctor_hook": {"method": "none", "dependent_cells_require_packed_base": True},
        "resources": {"disk_reserve_bytes": disk_reserve_bytes,
                      "scratch_budget_bytes": scratch_budget_bytes, "threads": threads},
        "program_spec_sha256": program_spec_sha256,
        "resource_admission_sha256": resource_admission_sha256,
        "source_deletion_permitted": False, "quality_claims_permitted": False,
        "inputs": sorted(inputs, key=lambda row: row["role"]),
    }
    # Validate the complete shape and every non-model-sized live input.  Source
    # shards retain the completed census identity and are rehashed by execution.
    fake_request = {"model": {"label": label, "family": MODEL_FAMILY},
                    "operation": OPERATION, "backend": BACKEND,
                    "program_spec_sha256": program_spec_sha256,
                    "resource_admission_sha256": resource_admission_sha256}
    _validate_spec(spec, fake_request, worker)
    abi = _load_module("doctor_v5_adapter_abi_spec_writer", ABI_PATH)
    output = _workspace_path(str(output_path), must_exist=False)
    abi.atomic_json(output, spec)
    return spec


def _validate_spec(spec: dict[str, Any], request: dict[str, Any],
                   worker: ModuleType) -> None:
    expected = {
        "schema", "label", "campaign_binding", "adapter_id", "operation",
        "model_family", "backend", "codec", "evaluation", "doctor_hook",
        "resources", "program_spec_sha256", "resource_admission_sha256",
        "source_deletion_permitted", "quality_claims_permitted", "inputs",
    }
    if set(spec) != expected or spec.get("schema") != SPEC_SCHEMA:
        raise AdapterError("typed ladder spec schema/keys are invalid")
    model = request.get("model")
    label = model.get("label") if isinstance(model, dict) else None
    if spec.get("label") != label or spec.get("adapter_id") != ADAPTER_ID \
            or spec.get("operation") != OPERATION or spec.get("model_family") != MODEL_FAMILY \
            or spec.get("backend") != BACKEND:
        raise AdapterError("typed ladder identity differs from canonical request")
    if request.get("operation") != OPERATION or request.get("backend") != BACKEND \
            or not isinstance(model, dict) or model.get("family") != MODEL_FAMILY:
        raise AdapterError("canonical request selected the wrong operation/family/backend")
    if spec.get("program_spec_sha256") != request.get("program_spec_sha256") \
            or spec.get("resource_admission_sha256") != request.get("resource_admission_sha256"):
        raise AdapterError("typed ladder program/resource binding differs from request")
    if not all(isinstance(spec.get(key), str) and SHA_RE.fullmatch(spec[key])
               for key in ("program_spec_sha256", "resource_admission_sha256")):
        raise AdapterError("typed ladder program/resource hash is invalid")
    if spec.get("source_deletion_permitted") is not False \
            or spec.get("quality_claims_permitted") is not False:
        raise AdapterError("typed ladder claim boundary is invalid")
    codec = spec.get("codec")
    if not isinstance(codec, dict) or codec.get("rate_id") not in worker.CANONICAL_RATES:
        raise AdapterError("typed ladder codec rate is invalid")
    campaign = spec.get("campaign_binding")
    worker._validate_campaign(campaign, label, codec["rate_id"])
    geometry = worker._rate_geometry(codec["rate_id"])
    if codec.get("artifact_mode") != geometry["artifact_mode"] \
            or codec.get("symbol_bits") != geometry["symbol_bits"] \
            or codec.get("vector_dim") != geometry["vector_dim"] \
            or codec.get("block_len") != geometry["block_len"] \
            or codec.get("tensor_scope") != "all-2d" \
            or codec.get("adaptive_scales") != geometry["adaptive_scales"]:
        raise AdapterError("typed codec geometry differs from reviewed mapping")


def _paths(request: dict[str, Any], request_path: Path) -> dict[str, Path]:
    raw = request.get("paths")
    if not isinstance(raw, dict):
        raise AdapterError("canonical paths are missing")
    paths = {name: _workspace_path(value, must_exist=False) for name, value in raw.items()}
    required = {"request", "output_dir", "checkpoint", "result", "execution_receipt"}
    if set(paths) != required or paths["request"] != request_path:
        raise AdapterError("canonical paths/request identity differ")
    output = paths["output_dir"]
    if any(paths[name].parent != output for name in (
            "request", "checkpoint", "result", "execution_receipt")):
        raise AdapterError("canonical artifacts must be direct output children")
    if output.exists() and (not output.is_dir() or output.is_symlink()):
        raise AdapterError("canonical output directory is invalid")
    output.mkdir(parents=True, exist_ok=True)
    return paths


def _write_once_or_equal(abi: ModuleType, path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if path.is_symlink() or _load_json(path) != value:
            raise AdapterError(f"resume identity differs at {path}")
        return
    abi.atomic_json(path, value)


def _build_internal(request: dict[str, Any], spec: dict[str, Any],
                    parameter: dict[str, Any], internal_root: Path) -> dict[str, Any]:
    census_path = _workspace_path(parameter.get("census_path"))
    census = _load_json(census_path)
    model_dir = _workspace_path(parameter.get("model_dir"))
    source_rows = census.get("source", {}).get("shards")
    if not isinstance(source_rows, list) or not source_rows:
        raise AdapterError("completed census has no source shards")
    shards: list[dict[str, Any]] = []
    for ordinal, row in enumerate(source_rows):
        name = row.get("name") if isinstance(row, dict) else None
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise AdapterError("census shard name is unsafe")
        path = _workspace_path(str(model_dir / name))
        digest, size = row.get("file_sha256"), row.get("bytes")
        if not isinstance(digest, str) or SHA_RE.fullmatch(digest) is None \
                or not isinstance(size, int) or size <= 0 or row.get("ordinal") != ordinal:
            raise AdapterError(f"live source shard {ordinal} differs from census")
        # The generic ABI hash-verified this exact path immediately before this
        # translation; the worker performs the second, process-bound hash pass.
        _assert_input(request, path, digest, size)
        shards.append({"ordinal": ordinal, "name": name, "path": str(path),
                       "sha256": digest, "bytes": size})

    bindings: dict[str, str] = {}
    for name, path in (
        ("worker", WORKER_PATH), ("base_helper", BASE_HELPER_PATH),
        ("quantizer", QUANTIZER), ("attestor", ATTESTOR), ("decoder", DECODER),
        ("evaluator", EVALUATOR_PATH),
    ):
        bound_path, bound_sha = _tool(path)
        size = _hash_file(Path(bound_path))[1]
        _assert_input(request, Path(bound_path), bound_sha, size)
        bindings[f"{name}_path"] = bound_path; bindings[f"{name}_sha256"] = bound_sha
    python_path, python_sha = _python()
    parameter_path = _workspace_path(request["parameter_manifest"]["path"])
    parameter_sha, parameter_size = _hash_file(parameter_path)
    _assert_input(request, parameter_path, parameter_sha, parameter_size)
    census_sha, census_size = _hash_file(census_path)
    _assert_input(request, census_path, census_sha, census_size)
    outer_sha = request.get("request_sha256")
    if not isinstance(outer_sha, str) or not SHA_RE.fullmatch(outer_sha):
        raise AdapterError("canonical request hash is invalid")
    return {
        "schema": "hawking.doctor_v5_strand_ladder_request.v1",
        "request_id": f"strand-ladder-{spec['label'].lower()}-{spec['codec']['rate_id'].replace('.', 'p')}-{outer_sha[:16]}",
        "label": spec["label"], "model_family": MODEL_FAMILY,
        "campaign_binding": spec["campaign_binding"], "codec": spec["codec"],
        "source": {"model_dir": str(model_dir), "census_path": str(census_path),
                   "census_sha256": census_sha,
                   "source_manifest_sha256": census["source"]["source_manifest_sha256"],
                   "shards": shards},
        "parameter_manifest": {"path": str(parameter_path), "sha256": parameter_sha},
        "execution": {**bindings, "python_path": python_path, "python_sha256": python_sha,
                      "threads": spec["resources"]["threads"]},
        "evaluation": spec["evaluation"], "doctor_hook": spec["doctor_hook"],
        "resources": {"disk_reserve_bytes": spec["resources"]["disk_reserve_bytes"],
                      "scratch_budget_bytes": spec["resources"]["scratch_budget_bytes"]},
        "output_root": str(internal_root),
        "evidence_policy": {"class": "provisional_engineering_evidence",
                            "quality_claims_permitted": False,
                            "dominance_claims_permitted": False,
                            "source_deletion_permitted": False},
    }


def _retained_artifacts(internal_root: Path) -> list[dict[str, Any]]:
    candidates: list[tuple[str, Path]] = [
        ("worker_request", internal_root / "request.json"),
        ("worker_checkpoint", internal_root / "checkpoint.json"),
        ("worker_receipt", internal_root / "execution_receipt.json"),
        ("bundle_manifest", internal_root / "bundle/manifest.json"),
        ("ephemeral_cleanup", internal_root / "evaluation/ephemeral_cleanup.json"),
        ("baseline_ppl", internal_root / "evaluation/baseline_ppl.json"),
        ("reconstruction_ppl", internal_root / "evaluation/reconstruction_ppl.json"),
        ("baseline_capability", internal_root / "evaluation/baseline_capability.json"),
        ("reconstruction_capability", internal_root / "evaluation/reconstruction_capability.json"),
    ]
    shard_root = internal_root / "bundle/shards"
    if shard_root.is_dir():
        for path in sorted(shard_root.iterdir()):
            if path.is_file() and not path.is_symlink():
                candidates.append((f"bundle_shard:{path.name}", path))
    rows = [{"role": role, **_artifact(path)} for role, path in candidates
            if path.is_file() and not path.is_symlink()]
    return sorted(rows, key=lambda row: row["role"])


def _packed_artifact_inventory(retained: list[dict[str, Any]], *, branch: str) \
        -> dict[str, Any]:
    """Exact deletion allowlist for campaign-orchestrated, receipt-bearing GC.

    The adapter never deletes these payloads itself.  Small manifests, results,
    checkpoints, receipts, dependency evidence, and all parent source shards are
    deliberately outside this allowlist.
    """
    rows = [{"role": row["role"], "path": row["path"],
             "sha256": row["sha256"], "bytes": row["bytes"]}
            for row in retained
            if isinstance(row, dict)
            and str(row.get("role", "")).startswith("bundle_shard:")
            and str(row.get("path", "")).endswith(".strand")]
    return {
        "schema": "hawking.doctor_v5_packed_artifact_inventory.v1",
        "branch": branch, "artifacts": rows,
        "artifact_count": len(rows), "total_bytes": sum(row["bytes"] for row in rows),
        "deletion_allowlist_exact": True,
        "gc_requires_successor_or_full_cell_sealed": True,
        "gc_receipt_required": True,
        "dependency_read_requires_packed_payload": False,
        "parent_source_shards_included": False,
        "retained_evidence_excluded_from_gc": [
            "bundle_manifest", "worker_request", "worker_checkpoint", "worker_receipt",
            "dependency_evidence", "outer_result", "outer_execution_receipt",
        ],
    }


def _completed_units(checkpoint: dict[str, Any]) -> list[str]:
    rows = checkpoint.get("completed_units")
    if not isinstance(rows, list) or any(not isinstance(row, str) for row in rows) \
            or len(rows) != len(set(rows)):
        raise AdapterError("internal checkpoint units are invalid")
    return rows


def _checkpoint_artifact_rows(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if set(value) >= {"path", "sha256", "bytes"} \
                and isinstance(value.get("path"), str) \
                and isinstance(value.get("sha256"), str) \
                and isinstance(value.get("bytes"), int):
            rows.append({"path": value["path"], "sha256": value["sha256"],
                         "bytes": value["bytes"]})
        for child in value.values():
            rows.extend(_checkpoint_artifact_rows(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(_checkpoint_artifact_rows(child))
    return rows


def _outer_checkpoint(abi: ModuleType, request: dict[str, Any], registry: dict[str, Any],
                      internal_root: Path, status: str) -> dict[str, Any]:
    request_path = internal_root / "request.json"
    checkpoint_path = internal_root / "checkpoint.json"
    if checkpoint_path.is_file():
        inner = _load_json(checkpoint_path); units = _completed_units(inner)
        cursor = units[-1] if units else "preflight"
        resume_sha = _hash_file(checkpoint_path)[0]
        artifacts = [{"role": "worker_checkpoint", **_artifact(checkpoint_path)},
                     {"role": "worker_request", **_artifact(request_path)}]
        cleanup_path = internal_root / "evaluation/ephemeral_cleanup.json"
        deleted: set[tuple[str, str, int]] = set()
        if cleanup_path.is_file():
            cleanup = _load_json(cleanup_path)
            for row in cleanup.get("deleted_artifacts", []):
                if isinstance(row, dict) and isinstance(row.get("path"), str) \
                        and isinstance(row.get("sha256"), str) \
                        and isinstance(row.get("bytes"), int):
                    deleted.add((row["path"], row["sha256"], row["bytes"]))
        seen = {str(request_path.resolve()), str(checkpoint_path.resolve())}
        for unit in units:
            evidence = inner.get("units", {}).get(unit)
            if not isinstance(evidence, dict):
                raise AdapterError("completed worker unit omits checkpoint evidence")
            for row in _checkpoint_artifact_rows(evidence):
                identity = (row["path"], row["sha256"], row["bytes"])
                if identity in deleted:
                    continue
                path = _workspace_path(row["path"])
                if str(path) in seen:
                    continue
                digest, size = _hash_file(path)
                if digest != row["sha256"] or size != row["bytes"]:
                    raise AdapterError("worker checkpoint artifact changed before outer seal")
                seen.add(str(path))
                artifacts.append({"role": f"worker_artifact:{len(artifacts):05d}",
                                  "path": str(path), "sha256": digest, "bytes": size})
        updated_at = inner.get("updated_at")
    else:
        units, cursor = [], "adapter-bound"
        resume_sha = _hash_file(request_path)[0]
        artifacts = [{"role": "worker_request", **_artifact(request_path)}]
        updated_at = None
    return abi.build_checkpoint(
        request=request, registry=registry, status=status, cursor=cursor,
        completed_units=units, resume_state_sha256=resume_sha,
        artifact_hashes=sorted(artifacts, key=lambda row: row["role"]),
        updated_at=updated_at,
    )


def _lifecycle(internal_root: Path, retained: list[dict[str, Any]]) -> dict[str, Any]:
    cleanup_path = internal_root / "evaluation/ephemeral_cleanup.json"
    deleted = []
    if cleanup_path.is_file():
        cleanup = _load_json(cleanup_path)
        deleted = cleanup.get("deleted_artifacts", [])
        if not isinstance(deleted, list):
            raise AdapterError("ephemeral cleanup receipt is invalid")
    leftovers = list((internal_root / "evaluation/reconstruction").glob("*.safetensors"))
    if leftovers:
        raise AdapterError("dense reconstruction remains after completed worker")
    preserved = [{"role": row.get("role"), "path": row.get("path"),
                  "sha256": row.get("sha256"), "bytes": row.get("bytes")}
                 for row in deleted if isinstance(row, dict)
                 and row.get("role") == "reconstruction"]
    manifest = _load_json(internal_root / "bundle/manifest.json")
    expected_packed = sum(1 for row in manifest.get("shards", [])
                          if isinstance(row, dict) and isinstance(row.get("packed"), dict))
    retained_packed = [row for row in retained
                       if row.get("role", "").startswith("bundle_shard:")
                       and row.get("path", "").endswith(".strand")]
    if expected_packed <= 0 or len(retained_packed) != expected_packed:
        raise AdapterError("complete result does not retain every packed shard")
    return {"dense_reconstruction_ephemeral": True,
            "dense_reconstruction_produced": bool(preserved),
            "dense_reconstruction_removed": True,
            "preserved_deleted_identity": preserved,
            "packed_base_retained": True,
            "packed_shard_count": expected_packed,
            "parent_source_deleted": False}


def _execute(request_path: Path) -> dict[str, Any]:
    abi = _load_module("doctor_v5_adapter_abi_ladder", ABI_PATH)
    parameter_abi = _load_module("doctor_v5_parameter_manifest_ladder", PARAMETER_MODULE_PATH)
    worker = _load_module("doctor_v5_strand_ladder_worker_adapter", WORKER_PATH)
    request_path = _workspace_path(str(request_path))
    request = abi.read_json(request_path)
    registry_path = _workspace_path(str(request_path.parent / "adapter_registry.json"))
    registry = abi.read_json(registry_path)
    _errors(abi.validate_registry(registry, verify_files=True, base_dir=ROOT), "registry")
    _errors(abi.validate_request(request, registry, verify_files=True), "request")
    entry = _entry(registry, ADAPTER_ID)
    if entry.get("adapter_version") != ADAPTER_VERSION \
            or entry.get("operations") != [OPERATION] \
            or entry.get("model_families") != [MODEL_FAMILY] \
            or entry.get("backends") != [BACKEND]:
        raise AdapterError("registry entry differs from reviewed ladder adapter scope")
    source_path = _workspace_path(entry["source_path"])
    if source_path != Path(__file__).resolve() \
            or _hash_file(source_path)[0] != entry["source_sha256"]:
        raise AdapterError("registry source binding differs from this adapter")
    paths = _paths(request, request_path)
    pilot_path = _workspace_path(request["pilot_spec"]["path"])
    if _hash_file(pilot_path)[0] != request["pilot_spec"]["sha256"]:
        raise AdapterError("typed ladder spec identity mismatch")
    spec = _load_json(pilot_path)
    if request["pilot_spec"]["schema"] != SPEC_SCHEMA:
        raise AdapterError("canonical pilot schema is not the ladder schema")
    _validate_spec(spec, request, worker)
    if request["inputs"] != _normalized_inputs(spec):
        raise AdapterError("canonical inputs differ from typed ladder spec")

    parameter_path = _workspace_path(request["parameter_manifest"]["path"])
    if _hash_file(parameter_path)[0] != request["parameter_manifest"]["sha256"]:
        raise AdapterError("parameter manifest identity mismatch")
    parameter = _load_json(parameter_path)
    _errors(parameter_abi.validate_manifest(parameter, verify_files=True), "parameter manifest")
    if parameter.get("census_report_sha256") != request["source_census_sha256"]:
        raise AdapterError("request census binding differs from parameter authority")

    internal_root = paths["output_dir"] / INTERNAL_DIR
    if internal_root.exists() and (not internal_root.is_dir() or internal_root.is_symlink()):
        raise AdapterError("internal ladder output is invalid")
    internal_root.mkdir(parents=True, exist_ok=True)
    inner = _build_internal(request, spec, parameter, internal_root)
    inner_path = internal_root / "request.json"
    _write_once_or_equal(abi, inner_path, inner)
    checkpoint = _outer_checkpoint(abi, request, registry, internal_root, "running")
    abi.atomic_json(paths["checkpoint"], checkpoint)
    try:
        internal_receipt = worker._execute(inner_path, preflight_only=False)
    except (worker.LadderError, worker.BASE.PassBError) as exc:
        checkpoint = _outer_checkpoint(abi, request, registry, internal_root, "checkpointed-stop")
        abi.atomic_json(paths["checkpoint"], checkpoint)
        raise AdapterError(f"ladder worker refused: {exc}") from exc
    if not isinstance(internal_receipt, dict) or internal_receipt.get("status") != "complete" \
            or internal_receipt.get("schema") != worker.RECEIPT_SCHEMA:
        raise AdapterError("ladder worker returned no complete receipt")

    campaign = spec["campaign_binding"]
    artifacts = _retained_artifacts(internal_root)
    metrics = {
        "campaign_cell": {"cell_id": campaign["cell_id"],
                          "cell_identity_sha256": campaign["cell_identity_sha256"],
                          "model_label": campaign["label"],
                          "rate_id": campaign["target_rate_id"],
                          "branch": campaign["branch"]},
        "lifecycle": _lifecycle(internal_root, artifacts),
        "packed_artifact_inventory": _packed_artifact_inventory(
            artifacts, branch=campaign["branch"]),
        "parameter_accounting": internal_receipt.get("parameter_accounting"),
        "physical_accounting": internal_receipt.get("bundle", {}).get("physical_accounting"),
        "quality_observation": internal_receipt.get("quality_observation"),
        "baseline_cache": internal_receipt.get("baseline_cache"),
        "claims": internal_receipt.get("claims"),
        "completed_replicates": 1,
        "replicate_scope": "preliminary_scale_mapping_not_dominance",
    }
    result = abi.build_result(request=request, registry=registry, status="complete",
                              output_artifacts=artifacts, metrics=metrics,
                              completed_at=internal_receipt.get("completed_at"))
    checkpoint = _outer_checkpoint(abi, request, registry, internal_root, "complete")
    command = abi.resolve_command(request, registry, request_path=request_path)
    receipt = abi.build_execution_receipt(
        request=request, result=result, registry=registry, checkpoint=checkpoint,
        command_argv=command, resource_observations=internal_receipt.get("resources", {}),
        phase_resume=internal_receipt.get("resume", {}),
        completed_at=internal_receipt.get("completed_at"),
    )
    _write_once_or_equal(abi, paths["result"], result)
    abi.atomic_json(paths["checkpoint"], checkpoint)
    _write_once_or_equal(abi, paths["execution_receipt"], receipt)
    _errors(abi.validate_result(result, request, registry, verify_files=True), "result")
    _errors(abi.validate_checkpoint(checkpoint, request, registry, verify_files=True), "checkpoint")
    _errors(abi.validate_execution_receipt(
        receipt, request, result, registry, checkpoint=checkpoint, command_argv=command),
        "execution receipt")
    return result


def _selftest() -> None:
    worker = _load_module("doctor_v5_strand_ladder_worker_selftest_adapter", WORKER_PATH)
    abi = _load_module("doctor_v5_adapter_abi_selftest_ladder", ABI_PATH)
    assert worker.REQUEST_SCHEMA == "hawking.doctor_v5_strand_ladder_request.v1"
    assert abi.REQUEST_SCHEMA == "hawking.doctor_v5_adapter_request.v1"
    assert len(worker.CANONICAL_RATES) == 10
    assert worker._rate_geometry("0.33")["candidate_within_payload_ceiling"] is True
    for path in (WORKER_PATH, BASE_HELPER_PATH, PARAMETER_MODULE_PATH,
                 EVALUATOR_PATH, QUANTIZER, ATTESTOR, DECODER):
        _tool(path)
    _python()
    print(json.dumps({"status": "ok", "adapter_id": ADAPTER_ID,
                      "operation": OPERATION, "spec_schema": SPEC_SCHEMA}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run"); run.add_argument("--request", required=True, type=Path)
    build = sub.add_parser("build-spec")
    build.add_argument("--label", required=True)
    build.add_argument("--rate-id", required=True)
    build.add_argument("--cell-id", required=True)
    build.add_argument("--cell-identity-sha256", required=True)
    build.add_argument("--program-spec-sha256", required=True)
    build.add_argument("--resource-admission-sha256", required=True)
    build.add_argument("--evaluation-mode", choices=("auto", "resident", "deferred"),
                       default="auto")
    build.add_argument("--disk-reserve-bytes", type=int, default=50_000_000_000)
    build.add_argument("--scratch-budget-bytes", type=int, required=True)
    build.add_argument("--threads", type=int, default=8)
    build.add_argument("--output", required=True, type=Path)
    sub.add_parser("capabilities"); sub.add_parser("selftest")
    args = parser.parse_args(argv)
    try:
        if args.command == "selftest":
            _selftest(); return 0
        if args.command == "build-spec":
            spec = build_spec(
                label=args.label, rate_id=args.rate_id, cell_id=args.cell_id,
                cell_identity_sha256=args.cell_identity_sha256,
                program_spec_sha256=args.program_spec_sha256,
                resource_admission_sha256=args.resource_admission_sha256,
                evaluation_mode=args.evaluation_mode,
                disk_reserve_bytes=args.disk_reserve_bytes,
                scratch_budget_bytes=args.scratch_budget_bytes,
                threads=args.threads, output_path=args.output,
            )
            print(json.dumps({"status": "ok", "path": str(args.output.resolve()),
                              "schema": spec["schema"], "cell_id": args.cell_id},
                             sort_keys=True))
            return 0
        if args.command == "capabilities":
            worker = _load_module("doctor_v5_strand_ladder_worker_caps", WORKER_PATH)
            print(json.dumps(worker.capabilities(), indent=2, sort_keys=True)); return 0
        print(json.dumps(_execute(args.request), sort_keys=True)); return 0
    except (AdapterError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
