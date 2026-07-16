#!/usr/bin/env python3.12
"""Default-off GPT-OSS 120B execution/thread contract staging.

This tool is deliberately separate from the live Doctor queue.  It audits the
exact 10-rate x 4-branch GPT-OSS matrix and, only when every cell has a reviewed
typed runtime specification plus real source-bound physical exact-output
evidence, stages the per-cell contract consumed by the aggressive-v2 queue.

No command launches a model, edits the campaign, registry, runtime specs,
results, active markers, or runtime defaults.  ``stage`` writes only to its
explicit non-live contract directory, is compare-and-swap/idempotent, and is
all-or-nothing at the readiness boundary.  GPT-OSS threads come exclusively
from each exact runtime spec and its matching physical receipt; this module has
no Qwen thread-profile fallback.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

import sys
sys.path.insert(0, str(HERE))

import doctor_v5_adapter_abi as adapter_abi
import doctor_v5_gptoss_parallel_scaffold as parallel_scaffold
import doctor_v5_ultra_queue as queue_contract


ULTRA_ROOT = ROOT / "reports/condense/doctor_v5_ultra"
DEFAULT_PLAN = ULTRA_ROOT / "campaign_plan.json"
DEFAULT_REGISTRY = ULTRA_ROOT / "adapter_registry.json"
DEFAULT_RUNTIME_ROOT = ULTRA_ROOT / "runtime_specs"
DEFAULT_PENDING_WIRING = (
    ROOT / "reports/condense/doctor_v5_unbound/gptoss_120b_parallel/pending_wiring.json"
)
DEFAULT_ROOT = (
    ROOT / "reports/condense/doctor_v5_unbound/gptoss_120b_execution_threads"
)
DEFAULT_RECEIPTS_ROOT = DEFAULT_ROOT / "physical_exact_output_receipts"
DEFAULT_CONTRACTS_ROOT = DEFAULT_ROOT / "contracts"
DEFAULT_MANIFEST = DEFAULT_CONTRACTS_ROOT / "manifest.json"

# These two constants are the exact ABI consumed by
# doctor_v5_ultra_aggressive_queue._gptoss_contract_profile.
CONTRACT_SCHEMA = (
    "hawking.doctor_v5_reviewed_source_bound_execution_thread_contract.v1"
)
ABI_VERSION = "2026-07-15.1"
PHYSICAL_RECEIPT_SCHEMA = (
    "hawking.doctor_v5_gptoss_exact_output_physical_receipt.v1"
)
STATUS_SCHEMA = "hawking.doctor_v5_gptoss_execution_thread_contract_status.v1"
MANIFEST_SCHEMA = "hawking.doctor_v5_gptoss_execution_thread_contract_manifest.v1"
GPTOSS_FAMILY = "gpt-oss-moe"
MODEL_LABEL = "120B"
HF_ID = "openai/gpt-oss-120b"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_THREADS = 24
SHA_RE = re.compile(r"[0-9a-f]{64}")
CELL_RE = re.compile(r"gpt-oss-120b__[a-z0-9]+bpw__[a-z-]+")

RATES = ("4", "3", "2", "1", "0.8", "0.55", "0.5", "0.33", "0.25", "0.1")
BRANCHES = {
    "codec_control": (
        "condense_control", "doctor-v5-strand-ladder-gpt-oss-moe",
        "hawking.doctor_v5_strand_ladder_spec.v1",
    ),
    "doctor_static": (
        "doctor_static", "doctor-v5-gpt-oss-static-repair",
        "hawking.doctor_v5_static_spec.v1",
    ),
    "doctor_conditional": (
        "doctor_conditional", "doctor-v5-gpt-oss-conditional-repair",
        "hawking.doctor_v5_conditional_spec.v1",
    ),
    "doctor_full": (
        "doctor_full", "doctor-v5-gpt-oss-full-treatment",
        "hawking.doctor_v5_full_spec.v1",
    ),
}

CONTRACT_KEYS = {
    "schema", "version", "cell_id", "model_family", "runtime_spec_sha256",
    "runtime_inputs_sha256", "selected_threads", "projected_wall_seconds",
    "exclusive_cpu", "review_status", "exact_output_receipts",
    "contract_sha256",
}
PHYSICAL_KEYS = {
    "schema", "version", "cell_id", "model_family", "runtime_spec_sha256",
    "runtime_inputs_sha256", "selected_threads", "wall_seconds",
    "review_status", "physical_execution", "exact_output", "receipt_sha256",
}
PHYSICAL_EXECUTION_KEYS = {
    "mode", "simulated", "fixture", "host_id_sha256", "boot_id_sha256",
    "pid", "started_at", "completed_at", "exit_code",
    "input_content_hashes_verified", "input_count", "argv_sha256",
    "executable", "launch_receipt",
}
EXACT_OUTPUT_KEYS = {
    "status", "comparison", "candidate", "oracle", "tested_cases",
    "skipped_cases", "mismatch_count",
}
MANIFEST_KEYS = {
    "schema", "version", "status", "plan", "registry", "matrix",
    "contract_schema", "aggressive_runtime_rows", "qwen_thread_profiles_used",
    "automatic_activation_permitted", "live_mutation_permitted",
    "runtime_defaults_mutation_permitted", "manifest_sha256",
}


class ContractError(RuntimeError):
    """The exact GPT-OSS contract generation is absent, stale, or unsafe."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _workspace_path(raw: str | Path, *, must_exist: bool = True,
                    regular: bool = False) -> Path:
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    # abspath normalizes '..' without following links; inspecting each existing
    # component below then rejects symlink traversal explicitly.
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(ROOT.resolve(strict=True))
    except ValueError as exc:
        raise ContractError(f"path escapes workspace: {raw}") from exc
    cursor = ROOT.resolve(strict=True)
    for part in relative.parts:
        cursor = cursor / part
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode):
            raise ContractError(f"symlink path component is forbidden: {cursor}")
    if must_exist:
        try:
            info = candidate.lstat()
        except OSError as exc:
            raise ContractError(f"required path is unavailable: {candidate}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ContractError(f"symlink is forbidden: {candidate}")
        if regular and not stat.S_ISREG(info.st_mode):
            raise ContractError(f"regular file required: {candidate}")
    return candidate


def _read_json(path: str | Path) -> dict[str, Any]:
    resolved = _workspace_path(path, regular=True)
    info = resolved.stat()
    if info.st_size > MAX_JSON_BYTES:
        raise ContractError(f"JSON is too large: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON root is not an object: {resolved}")
    return value


def _hash_file(path: str | Path) -> tuple[str, int]:
    resolved = _workspace_path(path, regular=True)
    before = resolved.stat()
    digest, total = hashlib.sha256(), 0
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
    after = resolved.stat()
    identity = lambda row: (
        row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns, row.st_ctime_ns,
    )
    if identity(before) != identity(after) or total != after.st_size:
        raise ContractError(f"file changed while hashing: {resolved}")
    return digest.hexdigest(), total


def _artifact(path: str | Path) -> dict[str, Any]:
    resolved = _workspace_path(path, regular=True)
    digest, size = _hash_file(resolved)
    return {"path": str(resolved), "sha256": digest, "bytes": size}


def _artifact_errors(row: Any, *, expected_path: Path | None = None,
                     nonempty: bool = False,
                     cache: dict[str, tuple[str, int]] | None = None) -> list[str]:
    if not isinstance(row, dict) or set(row) != {"path", "sha256", "bytes"}:
        return ["artifact keys are invalid"]
    if not _valid_sha(row.get("sha256")) \
            or isinstance(row.get("bytes"), bool) \
            or not isinstance(row.get("bytes"), int) or row["bytes"] < 0 \
            or (nonempty and row["bytes"] <= 0):
        return ["artifact identity is invalid"]
    try:
        path = _workspace_path(row["path"], regular=True)
        if expected_path is not None \
                and path != _workspace_path(expected_path, regular=True):
            return ["artifact path differs from its exact target"]
        observed = cache.get(str(path)) if cache is not None else None
        if observed is None:
            observed = _hash_file(path)
            if cache is not None:
                cache[str(path)] = observed
    except (ContractError, KeyError, TypeError) as exc:
        return [f"artifact cannot be verified: {exc}"]
    return [] if observed == (row["sha256"], row["bytes"]) \
        else ["artifact content identity changed"]


def _parse_time(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _plan_cells(plan: Any) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if not isinstance(plan, dict) \
            or plan.get("schema") != "hawking.doctor_v5_ultra_campaign_plan.v1" \
            or plan.get("plan_sha256") != _hash_value(_without(plan, "plan_sha256")):
        return [], ["campaign plan identity is invalid"]
    rows = plan.get("cells")
    if not isinstance(rows, list):
        return [], ["campaign plan cell inventory is invalid"]
    cells = [row for row in rows if isinstance(row, dict)
             and row.get("model_label") == MODEL_LABEL]
    expected_matrix = {(rate, branch) for rate in RATES for branch in BRANCHES}
    observed_matrix = {(row.get("rate_id"), row.get("branch")) for row in cells}
    if len(cells) != 40 or len({row.get("cell_id") for row in cells}) != 40 \
            or observed_matrix != expected_matrix:
        errors.append("campaign is not the exact unique GPT-OSS 10x4 matrix")
    paths: set[str] = set()
    for cell in cells:
        cell_id, branch = cell.get("cell_id"), cell.get("branch")
        expected = BRANCHES.get(branch)
        raw_path = cell.get("runtime_spec_path")
        relative = Path(raw_path) if isinstance(raw_path, str) else Path("")
        if not isinstance(cell_id, str) or CELL_RE.fullmatch(cell_id) is None \
                or expected is None \
                or cell.get("model_family") != GPTOSS_FAMILY \
                or cell.get("hf_id") != HF_ID \
                or (cell.get("command"), cell.get("adapter_id"),
                    cell.get("runtime_spec_schema")) != expected \
                or relative.is_absolute() or ".." in relative.parts \
                or relative.name != f"{cell_id}.json" \
                or str(relative) in paths:
            errors.append(f"GPT-OSS plan cell authority is invalid: {cell_id}")
        paths.add(str(relative))
    return sorted(cells, key=lambda row: row["cell_id"]), errors


def _registry_review(registry_path: Path, cells: list[dict[str, Any]]) \
        -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]], list[str]]:
    try:
        registry = _read_json(registry_path)
    except ContractError as exc:
        return None, {}, [str(exc)]
    errors = adapter_abi.validate_registry(registry, verify_files=True, base_dir=ROOT)
    expected = {
        cell["adapter_id"]: (cell["command"], cell["backend"])
        for cell in cells
    }
    reviewed: dict[str, dict[str, Any]] = {}
    entries = registry.get("entries", []) if isinstance(registry, dict) else []
    for adapter_id, (operation, backend) in sorted(expected.items()):
        row = next((entry for entry in entries if isinstance(entry, dict)
                    and entry.get("adapter_id") == adapter_id), None)
        if row is None or row.get("reviewed") is not True \
                or row.get("execution_only_not_quality_evidence") is not True \
                or row.get("operations") != [operation] \
                or row.get("model_families") != [GPTOSS_FAMILY] \
                or row.get("backends") != [backend]:
            errors.append(f"reviewed exact GPT-OSS adapter is absent: {adapter_id}")
        else:
            reviewed[adapter_id] = row
    return registry, reviewed, errors


def _runtime_path(cell: dict[str, Any], runtime_root: Path) -> Path:
    root = _workspace_path(runtime_root, must_exist=False)
    return root / Path(cell["runtime_spec_path"]).name


def _runtime_record(cell: dict[str, Any], runtime_root: Path,
                    reviewed: dict[str, dict[str, Any]]) \
        -> tuple[dict[str, Any] | None, list[str]]:
    path = _runtime_path(cell, runtime_root)
    if not path.exists():
        return None, ["exact reviewed runtime spec is absent"]
    try:
        spec = _read_json(path)
    except ContractError as exc:
        return None, [str(exc)]
    runtime, _inputs, errors = queue_contract._validate_runtime_spec(
        cell, spec, path, verify_inputs=False,
    )
    errors = list(errors)
    if cell.get("adapter_id") not in reviewed:
        errors.append("runtime adapter is not in the reviewed GPT-OSS registry set")
    inputs = spec.get("inputs")
    roles = {row.get("role") for row in inputs if isinstance(row, dict)} \
        if isinstance(inputs, list) else set()
    if not any(isinstance(role, str) and (
            role.startswith("source_") or role.startswith("source:")
            or role.startswith("source_shard:") or role in {
                "parameter_manifest", "source_census", "source_seal",
                "source_inventory", "source_work_plan",
            }) for role in roles):
        errors.append("runtime has no source-bound input authority")
    threads = spec.get("resources", {}).get("threads")
    if isinstance(threads, bool) or not isinstance(threads, int) \
            or not 1 <= threads <= MAX_THREADS:
        errors.append("GPT-OSS runtime thread count is outside the aggressive ABI")
    if runtime is None or errors:
        return None, errors
    return {
        "cell": cell, "spec": spec, "runtime_path": path,
        "runtime_artifact": _artifact(path),
        "runtime_inputs_sha256": _hash_value(inputs),
        "input_count": len(inputs), "selected_threads": threads,
    }, []


def validate_physical_receipt(receipt: Any, record: dict[str, Any], *,
                              cache: dict[str, tuple[str, int]] | None = None) \
        -> list[str]:
    """Validate genuine, non-fixture, exact-output physical evidence."""
    if not isinstance(receipt, dict) or set(receipt) != PHYSICAL_KEYS:
        return ["physical receipt keys are invalid"]
    errors: list[str] = []
    cell = record["cell"]
    if receipt.get("schema") != PHYSICAL_RECEIPT_SCHEMA \
            or receipt.get("version") != ABI_VERSION \
            or receipt.get("receipt_sha256") != _hash_value(
                _without(receipt, "receipt_sha256")):
        errors.append("physical receipt identity is invalid")
    if receipt.get("cell_id") != cell["cell_id"] \
            or receipt.get("model_family") != GPTOSS_FAMILY \
            or receipt.get("runtime_spec_sha256") \
            != record["runtime_artifact"]["sha256"] \
            or receipt.get("runtime_inputs_sha256") \
            != record["runtime_inputs_sha256"] \
            or receipt.get("selected_threads") != record["selected_threads"]:
        errors.append("physical receipt runtime/source/thread binding differs")
    wall = receipt.get("wall_seconds")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) \
            or not math.isfinite(float(wall)) or float(wall) <= 0:
        errors.append("physical wall time is invalid")
    if receipt.get("review_status") != "approved-source-bound-exact-output":
        errors.append("physical receipt has not passed exact-output review")

    physical = receipt.get("physical_execution")
    if not isinstance(physical, dict) or set(physical) != PHYSICAL_EXECUTION_KEYS:
        errors.append("physical execution envelope is invalid")
        physical = {}
    if physical.get("mode") != "physical-production-host" \
            or physical.get("simulated") is not False \
            or physical.get("fixture") is not False \
            or physical.get("exit_code") != 0 \
            or physical.get("input_content_hashes_verified") is not True \
            or physical.get("input_count") != record["input_count"] \
            or not _valid_sha(physical.get("host_id_sha256")) \
            or not _valid_sha(physical.get("boot_id_sha256")) \
            or not _valid_sha(physical.get("argv_sha256")) \
            or isinstance(physical.get("pid"), bool) \
            or not isinstance(physical.get("pid"), int) or physical.get("pid", 0) <= 1:
        errors.append("execution is not a valid real source-verified physical run")
    started, completed = _parse_time(physical.get("started_at")), \
        _parse_time(physical.get("completed_at"))
    if started is None or completed is None or completed <= started:
        errors.append("physical execution timestamps are invalid")
    elif isinstance(wall, (int, float)) and not isinstance(wall, bool) \
            and math.isfinite(float(wall)) and float(wall) > 0:
        elapsed = (completed - started).total_seconds()
        if abs(elapsed - float(wall)) > max(2.0, float(wall) * 0.2):
            errors.append("physical wall time differs materially from timestamps")
    for name in ("executable", "launch_receipt"):
        errors.extend(f"{name}: {item}" for item in _artifact_errors(
            physical.get(name), nonempty=True, cache=cache,
        ))

    exact = receipt.get("exact_output")
    if not isinstance(exact, dict) or set(exact) != EXACT_OUTPUT_KEYS:
        errors.append("exact-output envelope is invalid")
        exact = {}
    if exact.get("status") != "exact" \
            or exact.get("comparison") != "byte-for-byte" \
            or isinstance(exact.get("tested_cases"), bool) \
            or not isinstance(exact.get("tested_cases"), int) \
            or exact.get("tested_cases", 0) <= 0 \
            or exact.get("skipped_cases") != 0 or exact.get("mismatch_count") != 0:
        errors.append("physical exact-output comparison did not pass without skips")
    candidate, oracle = exact.get("candidate"), exact.get("oracle")
    errors.extend(f"candidate: {item}" for item in _artifact_errors(
        candidate, nonempty=True, cache=cache,
    ))
    errors.extend(f"oracle: {item}" for item in _artifact_errors(
        oracle, nonempty=True, cache=cache,
    ))
    if isinstance(candidate, dict) and isinstance(oracle, dict):
        if (candidate.get("sha256"), candidate.get("bytes")) \
                != (oracle.get("sha256"), oracle.get("bytes")):
            errors.append("candidate and oracle output bytes are not exact")
        try:
            candidate_path = _workspace_path(candidate["path"], regular=True)
            oracle_path = _workspace_path(oracle["path"], regular=True)
            cstat, ostat = candidate_path.stat(), oracle_path.stat()
            if candidate_path == oracle_path \
                    or (cstat.st_dev, cstat.st_ino) == (ostat.st_dev, ostat.st_ino):
                errors.append("candidate and oracle must be independent artifacts")
        except (ContractError, KeyError, TypeError):
            pass
    return errors


def validate_contract(contract: Any, record: dict[str, Any], *,
                      cache: dict[str, tuple[str, int]] | None = None) -> list[str]:
    if not isinstance(contract, dict) or set(contract) != CONTRACT_KEYS:
        return ["aggressive-consumer contract keys are invalid"]
    errors: list[str] = []
    if contract.get("schema") != CONTRACT_SCHEMA \
            or contract.get("version") != ABI_VERSION \
            or contract.get("contract_sha256") != _hash_value(
                _without(contract, "contract_sha256")):
        errors.append("execution/thread contract identity is invalid")
    if contract.get("cell_id") != record["cell"]["cell_id"] \
            or contract.get("model_family") != GPTOSS_FAMILY \
            or contract.get("runtime_spec_sha256") \
            != record["runtime_artifact"]["sha256"] \
            or contract.get("runtime_inputs_sha256") \
            != record["runtime_inputs_sha256"] \
            or contract.get("selected_threads") != record["selected_threads"]:
        errors.append("execution/thread contract binding differs")
    threads = contract.get("selected_threads")
    if contract.get("exclusive_cpu") is not (
            isinstance(threads, int) and not isinstance(threads, bool) and threads >= 20):
        errors.append("execution/thread exclusivity differs from consumer ABI")
    wall = contract.get("projected_wall_seconds")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) \
            or not math.isfinite(float(wall)) or float(wall) <= 0:
        errors.append("execution/thread projected wall time is invalid")
    if contract.get("review_status") != "approved-source-bound-exact-output":
        errors.append("execution/thread contract is not approved")
    receipts = contract.get("exact_output_receipts")
    if not isinstance(receipts, list) or len(receipts) != 1:
        errors.append("contract must bind exactly one per-cell physical receipt")
    else:
        errors.extend(_artifact_errors(receipts[0], cache=cache))
    return errors


def _pending_wiring_audit(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"present": False, "valid": False,
                "errors": ["pending 40-cell wiring packet is absent"],
                "work_plan_present": False, "work_plan_valid": False,
                "work_plan_errors": ["bound GPT-OSS work plan is absent"]}
    try:
        doc = _read_json(path)
        errors = parallel_scaffold.validate_pending_wiring(doc)
        work_binding = doc.get("work_plan")
        if not isinstance(work_binding, dict):
            raise ContractError("pending wiring has no work-plan binding")
        work_path = _workspace_path(work_binding["path"], regular=True)
        work_plan = _read_json(work_path)
        work_errors = parallel_scaffold.validate_work_plan(work_plan)
        if work_binding.get("work_plan_sha256") \
                != work_plan.get("work_plan_sha256"):
            work_errors.append("pending wiring/work-plan semantic hashes differ")
        work_present = True
    except (ContractError, OSError, ValueError) as exc:
        errors = [str(exc)]
        work_errors = [str(exc)]
        work_present = False
    return {
        "present": True, "valid": not errors, "errors": errors,
        "work_plan_present": work_present,
        "work_plan_valid": work_present and not work_errors,
        "work_plan_errors": work_errors,
    }


def _collect(*, plan_path: Path, registry_path: Path, runtime_root: Path,
             receipts_root: Path, contracts_root: Path,
             pending_wiring_path: Path | None) \
        -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    blockers: list[str] = []
    try:
        plan = _read_json(plan_path)
        cells, plan_errors = _plan_cells(plan)
        blockers.extend(plan_errors)
        plan_artifact = _artifact(plan_path)
    except ContractError as exc:
        plan, cells, plan_artifact = {}, [], None
        blockers.append(str(exc))
    registry, reviewed, registry_errors = _registry_review(registry_path, cells)
    blockers.extend(registry_errors)
    cell_rows: list[dict[str, Any]] = []
    records: dict[str, dict[str, Any]] = {}
    missing_specs: list[str] = []
    missing_physical: list[str] = []
    invalid_physical: dict[str, list[str]] = {}
    cache: dict[str, tuple[str, int]] = {}
    for cell in cells:
        cell_id = cell["cell_id"]
        row_blockers: list[str] = []
        record, runtime_errors = _runtime_record(cell, runtime_root, reviewed)
        row_blockers.extend(runtime_errors)
        if record is None:
            missing_specs.append(cell_id)
        receipt_path = _workspace_path(receipts_root, must_exist=False) / f"{cell_id}.json"
        receipt_artifact = None
        if not receipt_path.exists():
            missing_physical.append(cell_id)
            row_blockers.append("exact-output physical receipt is absent")
        elif record is not None:
            try:
                receipt = _read_json(receipt_path)
                receipt_errors = validate_physical_receipt(receipt, record, cache=cache)
                if receipt_errors:
                    invalid_physical[cell_id] = receipt_errors
                    row_blockers.extend(receipt_errors)
                else:
                    receipt_artifact = _artifact(receipt_path)
                    record["receipt"] = receipt
                    record["receipt_artifact"] = receipt_artifact
            except ContractError as exc:
                invalid_physical[cell_id] = [str(exc)]
                row_blockers.append(str(exc))
        contract_path = _workspace_path(contracts_root, must_exist=False) / f"{cell_id}.json"
        contract_valid = False
        if contract_path.exists() and record is not None and receipt_artifact is not None:
            try:
                contract_errors = validate_contract(
                    _read_json(contract_path), record, cache=cache,
                )
                contract_valid = not contract_errors
            except ContractError:
                contract_valid = False
        if not row_blockers and record is not None:
            records[cell_id] = record
        cell_rows.append({
            "cell_id": cell_id, "rate_id": cell.get("rate_id"),
            "branch": cell.get("branch"),
            "runtime_spec_ready": record is not None,
            "selected_threads": record.get("selected_threads") if record else None,
            "runtime_inputs_sha256": (
                record.get("runtime_inputs_sha256") if record else None
            ),
            "physical_exact_output_ready": receipt_artifact is not None,
            "contract_verified": contract_valid,
            "blockers": row_blockers,
        })
    wiring = _pending_wiring_audit(pending_wiring_path)
    ready = len(cells) == 40 and len(records) == 40 and not blockers
    summary: dict[str, Any] = {
        "schema": STATUS_SCHEMA, "version": ABI_VERSION,
        "status": "ready-to-stage-default-off" if ready else "blocked",
        "stage_permitted": ready, "automatic_activation_permitted": False,
        "live_mutation_permitted": False,
        "qwen_thread_profiles_used": False,
        "matrix": {
            "expected_cells": 40, "observed_cells": len(cells),
            "rates": len({row.get("rate_id") for row in cells}),
            "branches": len({row.get("branch") for row in cells}),
        },
        "plan": plan_artifact,
        "plan_sha256": plan.get("plan_sha256") if isinstance(plan, dict) else None,
        "registry": _artifact(registry_path) if registry is not None else None,
        "reviewed_gptoss_adapters": len(reviewed),
        "reviewed_runtime_specs": sum(
            1 for row in cell_rows if row["runtime_spec_ready"]
        ),
        "physical_exact_output_receipts": sum(
            1 for row in cell_rows if row["physical_exact_output_ready"]
        ),
        "verified_contracts": sum(1 for row in cell_rows if row["contract_verified"]),
        "missing_runtime_specs": sorted(missing_specs),
        "missing_physical_evidence": sorted(missing_physical),
        "invalid_physical_evidence": invalid_physical,
        "scaffold_audit": {
            "exact_40_cell_plan": len(cells) == 40 and not any(
                "matrix" in item or "plan cell" in item for item in blockers
            ),
            "pending_wiring": wiring,
            "consumer_contract_schema": CONTRACT_SCHEMA,
            "default_off": True,
        },
        "blockers": blockers,
        "cells": cell_rows,
    }
    summary["status_sha256"] = _hash_value(summary)
    return summary, records


def status(*, plan_path: Path = DEFAULT_PLAN,
           registry_path: Path = DEFAULT_REGISTRY,
           runtime_root: Path = DEFAULT_RUNTIME_ROOT,
           receipts_root: Path = DEFAULT_RECEIPTS_ROOT,
           contracts_root: Path = DEFAULT_CONTRACTS_ROOT,
           pending_wiring_path: Path | None = DEFAULT_PENDING_WIRING) -> dict[str, Any]:
    return _collect(
        plan_path=plan_path, registry_path=registry_path,
        runtime_root=runtime_root, receipts_root=receipts_root,
        contracts_root=contracts_root, pending_wiring_path=pending_wiring_path,
    )[0]


def _contract(record: dict[str, Any]) -> dict[str, Any]:
    receipt = record["receipt"]
    threads = record["selected_threads"]
    doc: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA, "version": ABI_VERSION,
        "cell_id": record["cell"]["cell_id"], "model_family": GPTOSS_FAMILY,
        "runtime_spec_sha256": record["runtime_artifact"]["sha256"],
        "runtime_inputs_sha256": record["runtime_inputs_sha256"],
        "selected_threads": threads,
        "projected_wall_seconds": float(receipt["wall_seconds"]),
        "exclusive_cpu": threads >= 20,
        "review_status": "approved-source-bound-exact-output",
        "exact_output_receipts": [record["receipt_artifact"]],
    }
    doc["contract_sha256"] = _hash_value(doc)
    return doc


def _json_payload(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                      allow_nan=False).encode("utf-8") + b"\n"


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cas_json(path: Path, value: Any) -> None:
    path = _workspace_path(path, must_exist=False)
    payload = _json_payload(value)
    if path.exists():
        if _workspace_path(path, regular=True).read_bytes() != payload:
            raise ContractError(f"CAS conflict; refusing to replace {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _workspace_path(path, regular=True).read_bytes() != payload:
                raise ContractError(f"CAS race; refusing to replace {path}")
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def stage(*, plan_path: Path = DEFAULT_PLAN,
          registry_path: Path = DEFAULT_REGISTRY,
          runtime_root: Path = DEFAULT_RUNTIME_ROOT,
          receipts_root: Path = DEFAULT_RECEIPTS_ROOT,
          contracts_root: Path = DEFAULT_CONTRACTS_ROOT,
          pending_wiring_path: Path | None = DEFAULT_PENDING_WIRING) -> dict[str, Any]:
    """CAS-stage all 40 contracts; never write a partial ready generation."""
    summary, records = _collect(
        plan_path=plan_path, registry_path=registry_path,
        runtime_root=runtime_root, receipts_root=receipts_root,
        contracts_root=contracts_root, pending_wiring_path=pending_wiring_path,
    )
    if summary["stage_permitted"] is not True or len(records) != 40:
        raise ContractError(
            "exact 40-cell contract generation is blocked: "
            f"runtime={summary['reviewed_runtime_specs']}/40 "
            f"physical={summary['physical_exact_output_receipts']}/40"
        )
    root = _workspace_path(contracts_root, must_exist=False)
    try:
        root.relative_to(ULTRA_ROOT.resolve(strict=True))
    except ValueError:
        pass
    else:
        raise ContractError("contract staging beneath the live campaign is forbidden")
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".stage.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        rows: list[dict[str, Any]] = []
        cache: dict[str, tuple[str, int]] = {}
        for cell_id in sorted(records):
            record = records[cell_id]
            contract = _contract(record)
            errors = validate_contract(contract, record, cache=cache)
            if errors:
                raise ContractError(
                    f"generated invalid contract {cell_id}: {'; '.join(errors)}"
                )
            contract_path = root / f"{cell_id}.json"
            _cas_json(contract_path, contract)
            rows.append({
                "cell_id": cell_id, "model_family": GPTOSS_FAMILY,
                "runtime_spec": record["runtime_artifact"],
                "source_bound_execution_contract": _artifact(contract_path),
            })
        plan = _read_json(plan_path)
        registry = _read_json(registry_path)
        manifest: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA, "version": ABI_VERSION,
            "status": "staged-default-off-not-activated",
            "plan": {**_artifact(plan_path), "plan_sha256": plan["plan_sha256"]},
            "registry": {
                **_artifact(registry_path),
                "registry_sha256": registry["registry_sha256"],
            },
            "matrix": {"rates": 10, "branches": 4, "cells": 40},
            "contract_schema": CONTRACT_SCHEMA,
            "aggressive_runtime_rows": rows,
            "qwen_thread_profiles_used": False,
            "automatic_activation_permitted": False,
            "live_mutation_permitted": False,
            "runtime_defaults_mutation_permitted": False,
        }
        manifest["manifest_sha256"] = _hash_value(manifest)
        _cas_json(root / "manifest.json", manifest)
        return manifest


def verify_manifest(manifest_path: Path = DEFAULT_MANIFEST, *,
                    plan_path: Path = DEFAULT_PLAN,
                    registry_path: Path = DEFAULT_REGISTRY,
                    runtime_root: Path = DEFAULT_RUNTIME_ROOT,
                    receipts_root: Path = DEFAULT_RECEIPTS_ROOT,
                    pending_wiring_path: Path | None = DEFAULT_PENDING_WIRING) \
        -> list[str]:
    try:
        manifest = _read_json(manifest_path)
    except ContractError as exc:
        return [str(exc)]
    if set(manifest) != MANIFEST_KEYS:
        return ["contract manifest keys are invalid"]
    errors: list[str] = []
    if manifest.get("schema") != MANIFEST_SCHEMA \
            or manifest.get("version") != ABI_VERSION \
            or manifest.get("status") != "staged-default-off-not-activated" \
            or manifest.get("manifest_sha256") != _hash_value(
                _without(manifest, "manifest_sha256")):
        errors.append("contract manifest identity is invalid")
    if manifest.get("matrix") != {"rates": 10, "branches": 4, "cells": 40} \
            or manifest.get("contract_schema") != CONTRACT_SCHEMA \
            or manifest.get("qwen_thread_profiles_used") is not False \
            or manifest.get("automatic_activation_permitted") is not False \
            or manifest.get("live_mutation_permitted") is not False \
            or manifest.get("runtime_defaults_mutation_permitted") is not False:
        errors.append("contract manifest crosses its default-off boundary")
    try:
        plan = _read_json(plan_path)
        registry = _read_json(registry_path)
        for row, path, semantic, field in (
            (manifest.get("plan"), plan_path, plan.get("plan_sha256"), "plan_sha256"),
            (manifest.get("registry"), registry_path,
             registry.get("registry_sha256"), "registry_sha256"),
        ):
            artifact = {name: row.get(name) for name in ("path", "sha256", "bytes")} \
                if isinstance(row, dict) else None
            errors.extend(_artifact_errors(artifact, expected_path=path))
            if not isinstance(row, dict) or row.get(field) != semantic:
                errors.append(f"manifest {field} semantic binding changed")
    except ContractError as exc:
        errors.append(str(exc))
    _summary, records = _collect(
        plan_path=plan_path, registry_path=registry_path,
        runtime_root=runtime_root, receipts_root=receipts_root,
        contracts_root=manifest_path.parent, pending_wiring_path=pending_wiring_path,
    )
    rows = manifest.get("aggressive_runtime_rows")
    if not isinstance(rows, list) or len(rows) != 40:
        return errors + ["manifest does not contain exactly 40 aggressive runtime rows"]
    seen: set[str] = set()
    cache: dict[str, tuple[str, int]] = {}
    for row in rows:
        required = {"cell_id", "model_family", "runtime_spec",
                    "source_bound_execution_contract"}
        cell_id = row.get("cell_id") if isinstance(row, dict) else None
        record = records.get(cell_id)
        if not isinstance(row, dict) or set(row) != required \
                or cell_id in seen or row.get("model_family") != GPTOSS_FAMILY \
                or record is None:
            errors.append(f"manifest aggressive runtime row is invalid: {cell_id}")
            continue
        seen.add(cell_id)
        errors.extend(f"{cell_id}: runtime: {item}" for item in _artifact_errors(
            row["runtime_spec"], expected_path=record["runtime_path"], cache=cache,
        ))
        contract_path = manifest_path.parent / f"{cell_id}.json"
        errors.extend(f"{cell_id}: contract: {item}" for item in _artifact_errors(
            row["source_bound_execution_contract"], expected_path=contract_path,
            cache=cache,
        ))
        try:
            errors.extend(f"{cell_id}: {item}" for item in validate_contract(
                _read_json(contract_path), record, cache=cache,
            ))
        except ContractError as exc:
            errors.append(f"{cell_id}: {exc}")
    if seen != set(records) or len(seen) != 40:
        errors.append("manifest does not cover the exact ready 40-cell set")
    return errors


def _arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--receipts-root", type=Path, default=DEFAULT_RECEIPTS_ROOT)
    parser.add_argument("--contracts-root", type=Path, default=DEFAULT_CONTRACTS_ROOT)
    parser.add_argument("--pending-wiring", type=Path, default=DEFAULT_PENDING_WIRING)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "stage", "verify"):
        _arguments(sub.add_parser(name))
    args = parser.parse_args(argv)
    common = {
        "plan_path": args.plan, "registry_path": args.registry,
        "runtime_root": args.runtime_root, "receipts_root": args.receipts_root,
        "pending_wiring_path": args.pending_wiring,
    }
    try:
        if args.command == "status":
            document = status(contracts_root=args.contracts_root, **common)
            print(json.dumps(document, indent=2, sort_keys=True))
            return 0
        if args.command == "stage":
            document = stage(contracts_root=args.contracts_root, **common)
            print(json.dumps({
                "status": "ok", "manifest": str(
                    (args.contracts_root / "manifest.json").resolve()
                ), "manifest_sha256": document["manifest_sha256"],
                "contracts": len(document["aggressive_runtime_rows"]),
                "automatic_activation_permitted": False,
            }, indent=2, sort_keys=True))
            return 0
        errors = verify_manifest(
            args.contracts_root / "manifest.json", **common,
        )
        print(json.dumps({
            "status": "ok" if not errors else "invalid", "errors": errors,
        }, indent=2, sort_keys=True))
        return 0 if not errors else 2
    except (ContractError, OSError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)},
                         indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
