#!/usr/bin/env python3.12
"""Source-bound Pass-B adapter for the 0.5B STRAND Q4 control cell.

The public side of this module speaks :mod:`doctor_v5_adapter_abi`.  Its only
reviewed operation translates that canonical request into the narrower
``doctor_v5_pass_b_worker`` request, executes the worker in this process, and
then seals the generic ABI result, receipt, and resume checkpoint.  There is no
shell, user-supplied argv, model deletion, or quality/dominance promotion path.
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
WORKER_PATH = HERE / "doctor_v5_pass_b_worker.py"
PARAMETER_MODULE_PATH = HERE / "doctor_v5_parameter_manifest.py"
PARAMETER_MANIFEST_DIR = ROOT / "reports/condense/doctor_v5_pass_b/parameter_manifests"
PASS_A_INDEX = ROOT / "reports/condense/doctor_v5_scale/index.json"
QUANTIZER = ROOT / "vendor/strand-quant/target/release/quantize-model"
ATTESTOR = ROOT / "vendor/strand-decode-kernel/target/release/attest-strand"
DECODER = ROOT / "vendor/strand-decode-kernel/target/release/archive-to-safetensors"
PPL_BENCH = HERE / "ppl_bench.py"
MULTI_EVAL = HERE / "multi_eval.py"
ADAPTER_ID = "doctor-v5-strand-q4-control"
ADAPTER_VERSION = "1"
COMMAND_ID = "condense_pilot"
LABEL = "0.5B"
OPERATION = "condense_pilot"
PROFILE = "strand-scalar-quality-rhtcols-v1"
INTERNAL_DIR_NAME = "strand_q4_control"
MAX_JSON_BYTES = 32 * 1024 * 1024
DEFAULT_DISK_RESERVE_BYTES = 150_000_000_000
DEFAULT_SCRATCH_BYTES = 12_000_000_000
HEX64 = re.compile(r"[0-9a-f]{64}")


class AdapterError(RuntimeError):
    """A fail-closed adapter refusal."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _hash_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise AdapterError(f"cannot open bound file {path}: {exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise AdapterError(f"bound path is not a regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(fd, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(fd)
        if not (before.st_dev == after.st_dev and before.st_ino == after.st_ino
                and before.st_size == after.st_size == size
                and before.st_mtime_ns == after.st_mtime_ns):
            raise AdapterError(f"bound file changed while hashing: {path}")
        return digest.hexdigest(), size
    finally:
        os.close(fd)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JSON_BYTES:
            raise AdapterError(f"invalid JSON file type or size: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"JSON root must be an object: {path}")
    return value


def _workspace_path(raw: Any, *, must_exist: bool = True) -> Path:
    if not isinstance(raw, str) or not raw:
        raise AdapterError("bound path must be a non-empty string")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(ROOT.resolve())
    except (OSError, RuntimeError, ValueError) as exc:
        raise AdapterError(f"path is missing or outside the workspace: {raw!r}") from exc
    if must_exist and candidate.is_symlink():
        raise AdapterError(f"symlinked bound input is forbidden: {raw!r}")
    return resolved


def _load_module(name: str, path: Path) -> ModuleType:
    if not path.is_file() or path.is_symlink():
        raise AdapterError(f"required reviewed module is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AdapterError(f"cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _errors(value: Any, context: str) -> None:
    if not isinstance(value, list) or any(not isinstance(row, str) for row in value):
        raise AdapterError(f"{context} validator returned an invalid result")
    if value:
        raise AdapterError(f"{context} validation failed: " + "; ".join(value))


def _input_by_role(request: dict[str, Any], role: str) -> dict[str, Any]:
    rows = request.get("inputs")
    if not isinstance(rows, list):
        raise AdapterError("canonical request inputs are missing")
    matches = [row for row in rows
               if isinstance(row, dict) and row.get("role") == role]
    if len(matches) != 1:
        raise AdapterError(f"canonical request must contain one {role!r} input")
    return matches[0]


def _verified_input(request: dict[str, Any], role: str) -> Path:
    row = _input_by_role(request, role)
    path = _workspace_path(row.get("path"))
    digest, size = _hash_file(path)
    if row.get("sha256") != digest or row.get("bytes") != size:
        raise AdapterError(f"canonical {role} input identity mismatch")
    return path


def _entry(registry: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    rows = registry.get("adapters")
    if not isinstance(rows, list):
        rows = registry.get("entries")
    if not isinstance(rows, list):
        raise AdapterError("reviewed registry has no adapter entries")
    binding = request.get("adapter")
    adapter_id = binding.get("adapter_id") if isinstance(binding, dict) else None
    candidates = [row for row in rows
                  if isinstance(row, dict) and row.get("adapter_id") == adapter_id]
    if len(candidates) != 1:
        raise AdapterError("canonical request does not select one reviewed adapter")
    return candidates[0]


def _artifact(path: Path) -> dict[str, Any]:
    digest, size = _hash_file(path)
    return {"path": str(path.resolve()), "sha256": digest, "bytes": size}


def _tool_binding(path: Path) -> tuple[str, str]:
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise AdapterError(f"reviewed tool is outside workspace: {path}") from exc
    if path.is_symlink():
        raise AdapterError(f"reviewed tool may not be a symlink: {path}")
    return str(resolved), _hash_file(resolved)[0]


def _python_binding() -> tuple[str, str]:
    """Bind the exact interpreter accepted by the in-process worker."""
    path = Path(sys.executable)
    try:
        info = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise AdapterError(f"cannot bind running Python interpreter: {exc}") from exc
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise AdapterError("running Python interpreter must be an exact regular file")
    return str(resolved), _hash_file(resolved)[0]


def _single_source_shard(census: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    source = census.get("source")
    if not isinstance(source, dict) or source.get("model_dir") != str(model_dir):
        raise AdapterError("Pass-A census model directory differs from the canonical request")
    shards = source.get("shards")
    if not isinstance(shards, list) or len(shards) != 1 or not isinstance(shards[0], dict):
        raise AdapterError("the reviewed 0.5B control requires exactly one source shard")
    shard = shards[0]
    if set(("name", "file_sha256", "bytes")) - set(shard):
        raise AdapterError("Pass-A source shard lacks its physical identity")
    name = shard["name"]
    if not isinstance(name, str) or Path(name).name != name:
        raise AdapterError("Pass-A source shard name is unsafe")
    weight = _workspace_path(str(model_dir / name))
    digest, size = _hash_file(weight)
    if digest != shard["file_sha256"] or size != shard["bytes"]:
        raise AdapterError("live 0.5B source shard differs from the completed Pass-A census")
    source_hash = source.get("source_manifest_sha256")
    if not isinstance(source_hash, str) or HEX64.fullmatch(source_hash) is None:
        raise AdapterError("Pass-A census lacks a source-manifest hash")
    return {"path": weight, "sha256": digest, "bytes": size,
            "source_manifest_sha256": source_hash}


def _internal_request(*, outer: dict[str, Any], registry_path: Path,
                      registry: dict[str, Any], entry: dict[str, Any],
                      census_path: Path, parameter_path: Path,
                      model_dir: Path, internal_root: Path,
                      scratch_budget_bytes: int) -> dict[str, Any]:
    """Translate one validated generic request into the frozen worker dialect."""
    model = outer.get("model")
    binding = outer.get("adapter")
    if not isinstance(model, dict) or model.get("label") != LABEL \
            or outer.get("operation") != OPERATION:
        raise AdapterError("adapter accepts only the 0.5B condense_pilot operation")
    if not isinstance(binding, dict) or binding.get("adapter_id") != ADAPTER_ID:
        raise AdapterError("canonical request selected the wrong adapter")
    if entry.get("adapter_version") != ADAPTER_VERSION:
        raise AdapterError("reviewed adapter version differs from this implementation")
    source_path = _workspace_path(entry.get("source_path"))
    source_sha, _ = _hash_file(source_path)
    if source_path != Path(__file__).resolve() or source_sha != entry.get("source_sha256"):
        raise AdapterError("registry source binding does not identify this exact adapter")

    census = _load_json(census_path)
    if census.get("status") != "complete" or census.get("label") != LABEL:
        raise AdapterError("completed 0.5B Pass-A census is required")
    shard = _single_source_shard(census, model_dir)
    census_sha, _ = _hash_file(census_path)
    parameter_sha, _ = _hash_file(parameter_path)
    registry_sha, _ = _hash_file(registry_path)
    pass_a_sha, _ = _hash_file(PASS_A_INDEX)
    worker_path, worker_sha = _tool_binding(WORKER_PATH)
    quantizer_path, quantizer_sha = _tool_binding(QUANTIZER)
    attest_path, attest_sha = _tool_binding(ATTESTOR)
    decoder_path, decoder_sha = _tool_binding(DECODER)
    ppl_path, ppl_sha = _tool_binding(PPL_BENCH)
    multi_path, multi_sha = _tool_binding(MULTI_EVAL)
    python_path, python_sha = _python_binding()

    outer_sha = outer.get("request_sha256")
    if not isinstance(outer_sha, str) or HEX64.fullmatch(outer_sha) is None:
        # The ABI validator is authoritative, but keep the internal identity
        # deterministic if a future ABI names its stamped hash differently.
        outer_sha = _hash_value({key: value for key, value in outer.items()
                                 if key != "request_sha256"})
    if (isinstance(scratch_budget_bytes, bool) or not isinstance(scratch_budget_bytes, int)
            or scratch_budget_bytes < DEFAULT_SCRATCH_BYTES):
        raise AdapterError(
            f"canonical request must reserve at least {DEFAULT_SCRATCH_BYTES} scratch bytes"
        )
    return {
        "schema": "hawking.doctor_v5_pass_b_pilot_request.v1",
        "request_id": f"pass-b-q4-{outer_sha[:20]}",
        "label": LABEL,
        "profile": PROFILE,
        "bits": 4,
        "source": {
            "model_dir": str(model_dir),
            "weight_file": str(shard["path"]),
            "weight_sha256": shard["sha256"],
            "weight_bytes": shard["bytes"],
            "census_report": str(census_path),
            "census_report_sha256": census_sha,
            "pass_a_index_sha256": pass_a_sha,
            "source_manifest_sha256": shard["source_manifest_sha256"],
        },
        "parameter_manifest": {"path": str(parameter_path), "sha256": parameter_sha},
        "adapter": {
            "registry_path": str(registry_path),
            "registry_sha256": registry_sha,
            "adapter_id": ADAPTER_ID,
            "adapter_source_sha256": source_sha,
        },
        "execution": {
            "worker_path": worker_path, "worker_sha256": worker_sha,
            "quantizer_path": quantizer_path, "quantizer_sha256": quantizer_sha,
            "attest_path": attest_path, "attest_sha256": attest_sha,
            "decoder_path": decoder_path, "decoder_sha256": decoder_sha,
            "ppl_bench_path": ppl_path, "ppl_bench_sha256": ppl_sha,
            "multi_eval_path": multi_path, "multi_eval_sha256": multi_sha,
            "python_path": python_path, "python_sha256": python_sha,
            "threads": 8,
        },
        "resources": {
            "disk_reserve_bytes": DEFAULT_DISK_RESERVE_BYTES,
            "scratch_budget_bytes": scratch_budget_bytes,
        },
        "output_root": str(internal_root),
        "evidence_policy": {
            "class": "provisional_engineering_evidence",
            "dominance_claim_permitted": False,
            "sealed_quality_claim_permitted": False,
            "source_deletion_permitted": False,
        },
    }


def _assert_outer_input(request: dict[str, Any], path: Path,
                        digest: str, size: int) -> None:
    rows = request.get("inputs")
    matches = [row for row in rows if isinstance(row, dict)
               and Path(str(row.get("path", ""))).resolve(strict=False) == path]
    if len(matches) != 1 or matches[0].get("sha256") != digest \
            or matches[0].get("bytes") != size:
        raise AdapterError(f"canonical input inventory does not bind {path}")


def _validate_pilot_spec(spec: dict[str, Any], request: dict[str, Any]) -> tuple[int, int]:
    required = {
        "schema": "hawking.doctor_v5_pass_b_strand_control_spec.v1",
        "label": LABEL,
        "adapter_id": ADAPTER_ID,
        "operation": OPERATION,
        "profile": PROFILE,
        "bits": 4,
        "source_deletion_permitted": False,
    }
    for field, expected in required.items():
        if spec.get(field) != expected:
            raise AdapterError(f"typed pilot spec {field} is not the reviewed value")
    for field in ("program_spec_sha256", "resource_admission_sha256"):
        value = spec.get(field)
        if not isinstance(value, str) or HEX64.fullmatch(value) is None:
            raise AdapterError(f"typed pilot spec {field} is invalid")
        if request.get(field) != value:
            raise AdapterError(f"request differs from typed pilot spec {field}")
    model = request.get("model")
    for field, observed in (
        ("model_family", model.get("family") if isinstance(model, dict) else None),
        ("backend", request.get("backend")),
        ("seed", request.get("seed")),
    ):
        if spec.get(field) != observed:
            raise AdapterError(f"request differs from typed pilot spec {field}")
    scratch = spec.get("scratch_budget_bytes")
    reserve = spec.get("disk_reserve_bytes")
    if isinstance(scratch, bool) or not isinstance(scratch, int) \
            or scratch < DEFAULT_SCRATCH_BYTES:
        raise AdapterError("typed pilot scratch budget is below the reviewed floor")
    if isinstance(reserve, bool) or not isinstance(reserve, int) \
            or reserve < DEFAULT_DISK_RESERVE_BYTES:
        raise AdapterError("typed pilot disk reserve is below the reviewed floor")
    # This first adapter is intentionally pinned to the stronger fixed reserve.
    if reserve != DEFAULT_DISK_RESERVE_BYTES:
        raise AdapterError("typed pilot disk reserve is not the frozen control value")
    return scratch, reserve


def _paths(request: dict[str, Any], request_path: Path) -> dict[str, Path]:
    raw = request.get("paths")
    if not isinstance(raw, dict):
        raise AdapterError("canonical output paths are absent")
    paths = {name: _workspace_path(value, must_exist=False)
             for name, value in raw.items()}
    required = {"request", "output_dir", "checkpoint", "result", "execution_receipt"}
    if set(paths) != required or paths["request"] != request_path:
        raise AdapterError("canonical output paths or request identity differ")
    output = paths["output_dir"]
    for name in ("request", "checkpoint", "result", "execution_receipt"):
        if paths[name].parent != output:
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


def _artifact_rows(paths: dict[str, Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role, path in sorted(paths.items()):
        if path.is_file() and not path.is_symlink():
            rows.append({"role": role, **_artifact(path)})
    return rows


def _completed_units(inner_checkpoint: dict[str, Any]) -> list[str]:
    rows = inner_checkpoint.get("completed_phases")
    if not isinstance(rows, list) or any(not isinstance(row, str) for row in rows) \
            or len(rows) != len(set(rows)):
        raise AdapterError("worker checkpoint phase inventory is invalid")
    return rows


def _normalized_spec_inputs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows = spec.get("inputs")
    if not isinstance(rows, list) or not rows:
        raise AdapterError("typed pilot spec has no bound input inventory")
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"role", "path", "sha256", "bytes"}:
            raise AdapterError("typed pilot spec input keys are invalid")
        path = _workspace_path(row["path"])
        normalized.append({**row, "path": str(path)})
    return sorted(normalized, key=lambda row: row["role"])


def _inner_artifact_map(internal_root: Path) -> dict[str, Path]:
    return {
        "worker_request": internal_root / "request.json",
        "worker_checkpoint": internal_root / "checkpoint.json",
        "worker_execution_receipt": internal_root / "execution_receipt.json",
        "packed_projections": internal_root / "bundle/projections.strand",
        "lossless_passthrough": internal_root / "bundle/passthrough.safetensors",
        "bundle_manifest": internal_root / "bundle/manifest.json",
        "archive_attestation": internal_root / "bundle/attestation.json",
        "reconstruction_oracle": internal_root / "evaluation/reconstruction.safetensors",
        "baseline_ppl": internal_root / "evaluation/baseline_ppl.json",
        "reconstruction_ppl": internal_root / "evaluation/reconstruction_ppl.json",
        "baseline_capability": internal_root / "evaluation/baseline_capability.json",
        "reconstruction_capability": internal_root / "evaluation/reconstruction_capability.json",
    }


def _sync_checkpoint(abi: ModuleType, *, request: dict[str, Any],
                     registry: dict[str, Any], outer_path: Path,
                     internal_root: Path, status: str) -> dict[str, Any]:
    inner_paths = _inner_artifact_map(internal_root)
    inner_checkpoint_path = inner_paths["worker_checkpoint"]
    if inner_checkpoint_path.is_file():
        inner_checkpoint = _load_json(inner_checkpoint_path)
        units = _completed_units(inner_checkpoint)
        cursor = units[-1] if units else "preflight"
        resume_sha = _hash_file(inner_checkpoint_path)[0]
        updated_at = inner_checkpoint.get("updated_at")
        selected = {"worker_request": inner_paths["worker_request"],
                    "worker_checkpoint": inner_checkpoint_path}
    else:
        units, cursor = [], "adapter-bound"
        resume_sha = _hash_file(inner_paths["worker_request"])[0]
        updated_at = None
        selected = {"worker_request": inner_paths["worker_request"]}
    artifacts = _artifact_rows(selected)
    checkpoint = abi.build_checkpoint(
        request=request, registry=registry, status=status, cursor=cursor,
        completed_units=units, resume_state_sha256=resume_sha,
        artifact_hashes=artifacts, updated_at=updated_at,
    )
    abi.atomic_json(outer_path, checkpoint)
    return checkpoint


def _execute(request_path: Path) -> dict[str, Any]:
    abi = _load_module("doctor_v5_adapter_abi", ABI_PATH)
    parameter_abi = _load_module("doctor_v5_parameter_manifest", PARAMETER_MODULE_PATH)
    request_path = _workspace_path(str(request_path))
    request = abi.read_json(request_path)
    registry_path = _workspace_path(str(abi.DEFAULT_REGISTRY_PATH))
    registry = abi.read_json(registry_path)
    _errors(abi.validate_registry(registry, verify_files=True, base_dir=ROOT), "registry")
    _errors(abi.validate_request(request, registry, verify_files=True), "request")
    paths = _paths(request, request_path)
    entry = _entry(registry, request)
    if entry.get("adapter_id") != ADAPTER_ID or entry.get("operations") != [OPERATION]:
        raise AdapterError("registry entry is not the single reviewed control operation")

    pilot_binding = request["pilot_spec"]
    pilot_path = _workspace_path(pilot_binding["path"])
    pilot_sha, _ = _hash_file(pilot_path)
    if pilot_sha != pilot_binding["sha256"]:
        raise AdapterError("typed pilot spec file identity mismatch")
    pilot_spec = _load_json(pilot_path)
    if pilot_binding["schema"] != pilot_spec.get("schema"):
        raise AdapterError("typed pilot spec schema binding mismatch")
    scratch_budget, _ = _validate_pilot_spec(pilot_spec, request)
    if request.get("inputs") != _normalized_spec_inputs(pilot_spec):
        raise AdapterError("canonical request inputs differ from the typed pilot spec")

    parameter_binding = request["parameter_manifest"]
    parameter_path = _workspace_path(parameter_binding["path"])
    parameter_sha, _ = _hash_file(parameter_path)
    if parameter_sha != parameter_binding["sha256"]:
        raise AdapterError("parameter manifest file identity mismatch")
    parameter_manifest = _load_json(parameter_path)
    _errors(parameter_abi.validate_manifest(parameter_manifest, verify_files=True),
            "parameter manifest")
    if parameter_manifest.get("census_report_sha256") != request["source_census_sha256"]:
        raise AdapterError("request source census differs from the parameter authority")
    census_path = _workspace_path(parameter_manifest.get("census_path"))
    model_dir = _workspace_path(parameter_manifest.get("model_dir"))

    internal_root = paths["output_dir"] / INTERNAL_DIR_NAME
    if internal_root.exists() and (not internal_root.is_dir() or internal_root.is_symlink()):
        raise AdapterError("internal run root is invalid")
    internal_root.mkdir(parents=True, exist_ok=True)
    inner_request = _internal_request(
        outer=request, registry_path=registry_path, registry=registry, entry=entry,
        census_path=census_path, parameter_path=parameter_path, model_dir=model_dir,
        internal_root=internal_root, scratch_budget_bytes=scratch_budget,
    )
    weight = Path(inner_request["source"]["weight_file"])
    _assert_outer_input(request, weight, inner_request["source"]["weight_sha256"],
                        inner_request["source"]["weight_bytes"])
    inner_request_path = internal_root / "request.json"
    _write_once_or_equal(abi, inner_request_path, inner_request)
    _sync_checkpoint(abi, request=request, registry=registry,
                     outer_path=paths["checkpoint"], internal_root=internal_root,
                     status="running")

    worker = _load_module("doctor_v5_pass_b_worker", WORKER_PATH)
    if worker.REQUEST_SCHEMA != inner_request["schema"] or worker.PROFILE != PROFILE:
        raise AdapterError("loaded worker dialect differs from the frozen adapter")
    try:
        internal_receipt = worker._execute(inner_request_path, preflight_only=False)
    except worker.PassBError as exc:
        _sync_checkpoint(abi, request=request, registry=registry,
                         outer_path=paths["checkpoint"], internal_root=internal_root,
                         status="checkpointed-stop")
        raise AdapterError(f"STRAND control worker refused: {exc}") from exc
    if not isinstance(internal_receipt, dict) or internal_receipt.get("status") != "complete" \
            or internal_receipt.get("schema") != worker.RECEIPT_SCHEMA:
        raise AdapterError("STRAND control worker did not return a complete bound receipt")

    inner_paths = _inner_artifact_map(internal_root)
    missing = [role for role, path in inner_paths.items() if not path.is_file() or path.is_symlink()]
    if missing:
        raise AdapterError("complete worker receipt lacks artifacts: " + ", ".join(missing))
    output_artifacts = _artifact_rows(inner_paths)
    metrics = {
        "parameter_accounting": internal_receipt.get("parameter_accounting"),
        "physical_accounting": internal_receipt.get("bundle", {}).get(
            "physical_accounting"),
        "quality_observation": internal_receipt.get("quality_observation"),
        "claims": internal_receipt.get("claims"),
    }
    result = abi.build_result(
        request=request, registry=registry, status="complete",
        output_artifacts=output_artifacts, metrics=metrics,
        completed_at=internal_receipt.get("completed_at"),
    )
    inner_checkpoint = _load_json(inner_paths["worker_checkpoint"])
    checkpoint = abi.build_checkpoint(
        request=request, registry=registry, status="complete", cursor="receipt",
        completed_units=_completed_units(inner_checkpoint),
        resume_state_sha256=_hash_file(inner_paths["worker_checkpoint"])[0],
        artifact_hashes=output_artifacts,
        updated_at=inner_checkpoint.get("updated_at"),
    )
    command = abi.resolve_command(request, registry, request_path=request_path)
    receipt = abi.build_execution_receipt(
        request=request, result=result, registry=registry, checkpoint=checkpoint,
        command_argv=command,
        resource_observations=internal_receipt.get("resources", {}),
        phase_resume=internal_receipt.get("resume", {}),
        completed_at=internal_receipt.get("completed_at"),
    )
    _write_once_or_equal(abi, paths["result"], result)
    abi.atomic_json(paths["checkpoint"], checkpoint)
    _write_once_or_equal(abi, paths["execution_receipt"], receipt)
    _errors(abi.validate_result(result, request, registry, verify_files=True), "result")
    _errors(abi.validate_checkpoint(checkpoint, request, registry, verify_files=True),
            "checkpoint")
    _errors(abi.validate_execution_receipt(
        receipt, request, result, registry, checkpoint=checkpoint, command_argv=command),
        "execution receipt")
    return result


def _selftest() -> None:
    abi = _load_module("doctor_v5_adapter_abi_selftest", ABI_PATH)
    worker = _load_module("doctor_v5_pass_b_worker_selftest", WORKER_PATH)
    assert abi.REQUEST_SCHEMA == "hawking.doctor_v5_adapter_request.v1"
    assert abi.RESULT_SCHEMA == "hawking.doctor_v5_adapter_result.v1"
    assert worker.REQUEST_SCHEMA == "hawking.doctor_v5_pass_b_pilot_request.v1"
    assert worker.PROFILE == PROFILE and ADAPTER_ID.startswith("doctor-v5-")
    assert _hash_value({"a": 1, "b": 2}) == _hash_value({"b": 2, "a": 1})
    for path in (ABI_PATH, PARAMETER_MODULE_PATH, WORKER_PATH, QUANTIZER,
                 ATTESTOR, DECODER, PPL_BENCH, MULTI_EVAL):
        _tool_binding(path)
    _python_binding()
    print(json.dumps({"status": "ok", "adapter_id": ADAPTER_ID,
                      "operation": OPERATION, "profile": PROFILE}, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--request", required=True, type=Path)
    sub.add_parser("selftest")
    args = parser.parse_args(argv)
    try:
        if args.command == "selftest":
            _selftest()
        else:
            result = _execute(args.request)
            print(json.dumps(result, sort_keys=True))
        return 0
    except (AdapterError, ValueError, KeyError, OSError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
