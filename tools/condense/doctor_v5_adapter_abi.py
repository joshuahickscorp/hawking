#!/usr/bin/env python3.12
"""Typed, source-bound execution ABI for Doctor-v5 Pass-B adapters."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import sys
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
PASS_B_ROOT = ROOT / "reports/condense/doctor_v5_pass_b"
DEFAULT_REGISTRY_PATH = PASS_B_ROOT / "adapter_registry.json"
REGISTRY_SCHEMA = "hawking.doctor_v5_adapter_registry.v1"
REQUEST_SCHEMA = "hawking.doctor_v5_adapter_request.v1"
RESULT_SCHEMA = "hawking.doctor_v5_adapter_result.v1"
EXECUTION_RECEIPT_SCHEMA = "hawking.doctor_v5_adapter_execution_receipt.v1"
CHECKPOINT_SCHEMA = "hawking.doctor_v5_adapter_exact_resume_checkpoint.v1"
POLICY_VERSION = "2026-07-13.1"
MAX_JSON_BYTES = 64 * 1024 * 1024
SHA_RE = re.compile(r"[0-9a-f]{64}")


REGISTRY_KEYS = {"schema", "policy_version", "created_at", "entries", "registry_sha256"}
ENTRY_KEYS = {
    "adapter_id", "adapter_version", "source_path", "source_sha256",
    "executable_path", "executable_sha256", "entrypoint_argv", "operations",
    "model_families", "backends", "request_schema", "result_schema",
    "checkpoint_schema", "reviewed", "execution_only_not_quality_evidence",
}
REQUEST_KEYS = {
    "schema", "policy_version", "created_at", "request_id",
    "program_spec_sha256", "parameter_manifest", "source_census_sha256",
    "registry_sha256", "adapter", "operation", "model", "backend", "seed",
    "inputs", "pilot_spec", "paths", "authorization",
    "resource_admission_sha256", "quality_claims_permitted", "request_sha256",
}
RESULT_KEYS = {
    "schema", "policy_version", "completed_at", "request_sha256", "adapter",
    "status", "output_artifacts", "metrics", "evidence_class",
    "quality_claims_permitted", "source_deletion_permitted", "result_sha256",
}
CHECKPOINT_KEYS = {
    "schema", "policy_version", "updated_at", "request_sha256", "adapter",
    "status", "cursor", "completed_units", "resume_state_sha256",
    "artifact_hashes", "exact_resume", "checkpoint_sha256",
}
RECEIPT_KEYS = {
    "schema", "policy_version", "completed_at", "request_sha256",
    "result_sha256", "registry_sha256", "adapter", "command_argv_sha256",
    "checkpoint_sha256", "resource_observations", "phase_resume",
    "quality_claims_permitted", "source_deletion_permitted", "receipt_sha256",
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _is_int(value: Any, *, minimum: int | None = None) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) \
        and (minimum is None or value >= minimum)


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(row) for row in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _finite(v) for k, v in value.items())
    return True


def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"JSON too large: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return value


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def _sha_file(path: Path) -> tuple[str, int]:
    if path.is_symlink():
        raise ValueError(f"symlink forbidden: {path}")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"not a regular file: {path}")
    digest, total = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            total += len(block)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != \
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) \
            or total != after.st_size:
        raise ValueError(f"file changed while hashing: {path}")
    return digest.hexdigest(), total


def _resolve(raw: Any, base_dir: Path | None = None, *, must_exist: bool = True) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("path must be a nonempty string")
    path = Path(raw)
    if not path.is_absolute():
        path = (base_dir or ROOT) / path
    return path.resolve(strict=must_exist)


def _entry(registry: dict[str, Any], adapter_id: str) -> dict[str, Any] | None:
    for row in registry.get("entries", []):
        if isinstance(row, dict) and row.get("adapter_id") == adapter_id:
            return row
    return None


def _without(doc: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in doc.items() if name != key}


def build_registry(entries: Iterable[dict[str, Any]], *, created_at: str | None = None,
                   output_path: str | Path | None = None) -> dict[str, Any]:
    rows = sorted((dict(row) for row in entries), key=lambda row: row.get("adapter_id", ""))
    doc: dict[str, Any] = {
        "schema": REGISTRY_SCHEMA, "policy_version": POLICY_VERSION,
        "created_at": created_at or _now(), "entries": rows,
    }
    doc["registry_sha256"] = _hash_value(doc)
    errors = validate_registry(doc)
    if errors:
        raise ValueError("invalid registry: " + "; ".join(errors))
    if output_path is not None:
        atomic_json(output_path, doc)
    return doc


def validate_registry(doc: Any, verify_files: bool = False,
                      base_dir: str | Path | None = None) -> list[str]:
    errors: list[str] = []
    base = Path(base_dir) if base_dir is not None else ROOT
    if not isinstance(doc, dict):
        return ["registry is not an object"]
    if set(doc) != REGISTRY_KEYS:
        return ["registry keys are not exact"]
    if doc.get("schema") != REGISTRY_SCHEMA or doc.get("policy_version") != POLICY_VERSION:
        errors.append("registry schema/policy mismatch")
    if doc.get("registry_sha256") != _hash_value(_without(doc, "registry_sha256")):
        errors.append("registry hash mismatch")
    entries = doc.get("entries")
    if not isinstance(entries, list) or not entries:
        errors.append("registry entries are empty")
        return errors
    ids: set[str] = set()
    if entries != sorted(entries, key=lambda row: row.get("adapter_id", "")
                         if isinstance(row, dict) else ""):
        errors.append("registry entries are not sorted")
    for index, row in enumerate(entries):
        prefix = f"entry[{index}]"
        if not isinstance(row, dict) or set(row) != ENTRY_KEYS:
            errors.append(f"{prefix} keys invalid")
            continue
        adapter_id = row.get("adapter_id")
        if not isinstance(adapter_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{4,127}", adapter_id):
            errors.append(f"{prefix} adapter_id invalid")
        elif adapter_id in ids:
            errors.append(f"{prefix} duplicate adapter_id")
        ids.add(adapter_id)
        if not isinstance(row.get("adapter_version"), str) or not row["adapter_version"]:
            errors.append(f"{prefix} version invalid")
        for field in ("source_sha256", "executable_sha256"):
            if not _is_sha(row.get(field)):
                errors.append(f"{prefix} {field} invalid")
        argv = row.get("entrypoint_argv")
        if not isinstance(argv, list) or not argv or any(
                not isinstance(token, str) or not token or "\x00" in token for token in argv):
            errors.append(f"{prefix} entrypoint argv invalid")
        elif sum(token.count("{request_path}") for token in argv) != 1 \
                or any("{" in token.replace("{request_path}", "")
                       or "}" in token.replace("{request_path}", "") for token in argv):
            errors.append(f"{prefix} entrypoint placeholder invalid")
        for field in ("operations", "model_families", "backends"):
            value = row.get(field)
            if not isinstance(value, list) or not value or value != sorted(set(value)) \
                    or any(not isinstance(v, str) or not v for v in value):
                errors.append(f"{prefix} {field} invalid")
        if row.get("request_schema") != REQUEST_SCHEMA \
                or row.get("result_schema") != RESULT_SCHEMA \
                or row.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
            errors.append(f"{prefix} schema scope invalid")
        if row.get("reviewed") is not True \
                or row.get("execution_only_not_quality_evidence") is not True:
            errors.append(f"{prefix} review boundary invalid")
        if verify_files:
            try:
                source = _resolve(row["source_path"], base)
                executable = _resolve(row["executable_path"], base)
                if _sha_file(source)[0] != row["source_sha256"]:
                    errors.append(f"{prefix} source hash mismatch")
                if _sha_file(executable)[0] != row["executable_sha256"]:
                    errors.append(f"{prefix} executable hash mismatch")
                first = _resolve(argv[0], base)
                if first != executable:
                    errors.append(f"{prefix} argv executable mismatch")
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f"{prefix} file verification failed: {exc}")
    return errors


def build_request(*, registry: dict[str, Any], adapter_id: str, operation: str,
                  program_spec_sha256: str, parameter_manifest_path: str,
                  parameter_manifest_sha256: str, source_census_sha256: str,
                  model_label: str, model_family: str, backend: str, seed: int,
                  inputs: list[dict[str, Any]], pilot_spec_path: str,
                  pilot_spec_sha256: str, pilot_spec_schema: str, request_path: str,
                  output_dir: str, checkpoint_path: str, result_path: str,
                  execution_receipt_path: str, operator_greenlight_sha256: str,
                  resource_admission_sha256: str, created_at: str | None = None) -> dict[str, Any]:
    entry = _entry(registry, adapter_id)
    if entry is None:
        raise ValueError("adapter_id is not registered")
    identity = _hash_value({"adapter": adapter_id, "operation": operation,
                            "label": model_label, "program": program_spec_sha256,
                            "pilot": pilot_spec_sha256})[:16]
    doc: dict[str, Any] = {
        "schema": REQUEST_SCHEMA, "policy_version": POLICY_VERSION,
        "created_at": created_at or _now(),
        "request_id": f"passb-{model_label.lower()}-{identity}",
        "program_spec_sha256": program_spec_sha256,
        "parameter_manifest": {"path": parameter_manifest_path,
                               "sha256": parameter_manifest_sha256},
        "source_census_sha256": source_census_sha256,
        "registry_sha256": registry.get("registry_sha256"),
        "adapter": {"adapter_id": adapter_id, "adapter_version": entry["adapter_version"],
                    "source_sha256": entry["source_sha256"],
                    "executable_sha256": entry["executable_sha256"]},
        "operation": operation,
        "model": {"label": model_label, "family": model_family},
        "backend": backend, "seed": seed,
        "inputs": sorted((dict(row) for row in inputs), key=lambda row: row.get("role", "")),
        "pilot_spec": {"path": pilot_spec_path, "sha256": pilot_spec_sha256,
                       "schema": pilot_spec_schema},
        "paths": {"request": request_path, "output_dir": output_dir,
                  "checkpoint": checkpoint_path, "result": result_path,
                  "execution_receipt": execution_receipt_path},
        "authorization": {"scope": "operator_greenlit_pass_b_pilot",
                          "operator_greenlight_sha256": operator_greenlight_sha256,
                          "quality_evidence": False},
        "resource_admission_sha256": resource_admission_sha256,
        "quality_claims_permitted": False,
    }
    doc["request_sha256"] = _hash_value(doc)
    errors = validate_request(doc, registry)
    if errors:
        raise ValueError("built invalid request: " + "; ".join(errors))
    return doc


def _validate_artifact_rows(rows: Any, *, verify_files: bool,
                            name: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(rows, list):
        return [f"{name} is not a list"]
    roles, paths = set(), set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"role", "path", "sha256", "bytes"}:
            errors.append(f"{name}[{index}] keys invalid")
            continue
        role, path = row.get("role"), row.get("path")
        if not isinstance(role, str) or not role or role in roles:
            errors.append(f"{name}[{index}] role invalid/duplicate")
        if not isinstance(path, str) or not Path(path).is_absolute() or path in paths:
            errors.append(f"{name}[{index}] path invalid/duplicate")
        roles.add(role); paths.add(path)
        if not _is_sha(row.get("sha256")) or not _is_int(row.get("bytes"), minimum=0):
            errors.append(f"{name}[{index}] identity invalid")
        if verify_files and not errors:
            try:
                digest, size = _sha_file(Path(path))
                if digest != row["sha256"] or size != row["bytes"]:
                    errors.append(f"{name}[{index}] live identity mismatch")
            except (OSError, ValueError) as exc:
                errors.append(f"{name}[{index}] file verification failed: {exc}")
    if rows != sorted(rows, key=lambda row: row.get("role", "") if isinstance(row, dict) else ""):
        errors.append(f"{name} is not sorted")
    return errors


def validate_request(doc: Any, registry: dict[str, Any],
                     verify_files: bool = False) -> list[str]:
    errors = validate_registry(registry)
    if not isinstance(doc, dict):
        return errors + ["request is not an object"]
    if set(doc) != REQUEST_KEYS:
        return errors + ["request keys are not exact"]
    if doc.get("schema") != REQUEST_SCHEMA or doc.get("policy_version") != POLICY_VERSION:
        errors.append("request schema/policy mismatch")
    if doc.get("request_sha256") != _hash_value(_without(doc, "request_sha256")):
        errors.append("request hash mismatch")
    for field in ("program_spec_sha256", "source_census_sha256",
                  "registry_sha256", "resource_admission_sha256", "request_sha256"):
        if not _is_sha(doc.get(field)):
            errors.append(f"request {field} invalid")
    if doc.get("registry_sha256") != registry.get("registry_sha256"):
        errors.append("request registry binding mismatch")
    adapter = doc.get("adapter")
    if not isinstance(adapter, dict) or set(adapter) != {
            "adapter_id", "adapter_version", "source_sha256", "executable_sha256"}:
        errors.append("request adapter keys invalid")
        entry = None
    else:
        entry = _entry(registry, adapter.get("adapter_id"))
        if entry is None or any(adapter.get(key) != entry.get(key) for key in (
                "adapter_id", "adapter_version", "source_sha256", "executable_sha256")):
            errors.append("request adapter does not match registry")
    model = doc.get("model")
    if not isinstance(model, dict) or set(model) != {"label", "family"} \
            or not all(isinstance(v, str) and v for v in model.values()):
        errors.append("request model invalid")
    elif entry is not None and model["family"] not in entry["model_families"]:
        errors.append("request model family not allowed")
    if entry is not None and doc.get("operation") not in entry["operations"]:
        errors.append("request operation not allowed")
    if entry is not None and doc.get("backend") not in entry["backends"]:
        errors.append("request backend not allowed")
    if not _is_int(doc.get("seed"), minimum=0):
        errors.append("request seed invalid")
    parameter = doc.get("parameter_manifest")
    pilot = doc.get("pilot_spec")
    if not isinstance(parameter, dict) or set(parameter) != {"path", "sha256"} \
            or not _is_sha(parameter.get("sha256")):
        errors.append("request parameter manifest binding invalid")
    if not isinstance(pilot, dict) or set(pilot) != {"path", "sha256", "schema"} \
            or not _is_sha(pilot.get("sha256")) or not isinstance(pilot.get("schema"), str):
        errors.append("request pilot spec binding invalid")
    paths = doc.get("paths")
    if not isinstance(paths, dict) or set(paths) != {
            "request", "output_dir", "checkpoint", "result", "execution_receipt"}:
        errors.append("request paths invalid")
    else:
        values = list(paths.values())
        if any(not isinstance(v, str) or not Path(v).is_absolute() for v in values) \
                or len(values) != len(set(values)):
            errors.append("request paths must be distinct absolute paths")
        elif any(Path(paths[name]).parent != Path(paths["output_dir"])
                 for name in ("request", "checkpoint", "result", "execution_receipt")):
            errors.append("request artifact paths must be direct children of output_dir")
    auth = doc.get("authorization")
    if not isinstance(auth, dict) or set(auth) != {
            "scope", "operator_greenlight_sha256", "quality_evidence"} \
            or auth.get("scope") != "operator_greenlit_pass_b_pilot" \
            or not _is_sha(auth.get("operator_greenlight_sha256")) \
            or auth.get("quality_evidence") is not False:
        errors.append("request authorization invalid")
    if doc.get("quality_claims_permitted") is not False or not _finite(doc):
        errors.append("request claim/numeric boundary invalid")
    errors.extend(_validate_artifact_rows(doc.get("inputs"), verify_files=verify_files,
                                          name="inputs"))
    if verify_files:
        for name, binding in (("parameter manifest", parameter), ("pilot spec", pilot)):
            if isinstance(binding, dict):
                try:
                    digest, _ = _sha_file(Path(binding["path"]))
                    if digest != binding["sha256"]:
                        errors.append(f"{name} live hash mismatch")
                except (OSError, ValueError, KeyError) as exc:
                    errors.append(f"{name} verification failed: {exc}")
    return errors


def resolve_command(request: dict[str, Any], registry: dict[str, Any],
                    request_path: str | Path | None = None) -> list[str]:
    errors = validate_request(request, registry)
    if errors:
        raise ValueError("cannot resolve invalid request: " + "; ".join(errors))
    expected = Path(request["paths"]["request"]).resolve(strict=False)
    supplied = Path(request_path).resolve(strict=False) if request_path is not None else expected
    if supplied != expected:
        raise ValueError("resolver request_path differs from bound request path")
    entry = _entry(registry, request["adapter"]["adapter_id"])
    assert entry is not None
    return [token.replace("{request_path}", str(expected))
            for token in entry["entrypoint_argv"]]


def build_result(*, request: dict[str, Any], registry: dict[str, Any], status: str,
                 output_artifacts: list[dict[str, Any]], metrics: dict[str, Any],
                 evidence_class: str = "provisional_engineering_evidence",
                 completed_at: str | None = None) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema": RESULT_SCHEMA, "policy_version": POLICY_VERSION,
        "completed_at": completed_at or _now(), "request_sha256": request["request_sha256"],
        "adapter": dict(request["adapter"]), "status": status,
        "output_artifacts": sorted((dict(row) for row in output_artifacts),
                                   key=lambda row: row.get("role", "")),
        "metrics": metrics, "evidence_class": evidence_class,
        "quality_claims_permitted": False, "source_deletion_permitted": False,
    }
    doc["result_sha256"] = _hash_value(doc)
    errors = validate_result(doc, request, registry)
    if errors:
        raise ValueError("built invalid result: " + "; ".join(errors))
    return doc


def validate_result(doc: Any, request: dict[str, Any], registry: dict[str, Any],
                    verify_files: bool = False) -> list[str]:
    errors = validate_request(request, registry)
    if not isinstance(doc, dict):
        return errors + ["result is not an object"]
    if set(doc) != RESULT_KEYS:
        return errors + ["result keys are not exact"]
    if doc.get("schema") != RESULT_SCHEMA or doc.get("policy_version") != POLICY_VERSION:
        errors.append("result schema/policy mismatch")
    if doc.get("result_sha256") != _hash_value(_without(doc, "result_sha256")):
        errors.append("result hash mismatch")
    if doc.get("request_sha256") != request.get("request_sha256") \
            or doc.get("adapter") != request.get("adapter"):
        errors.append("result request/adapter binding mismatch")
    if doc.get("status") != "complete" or not isinstance(doc.get("metrics"), dict) \
            or not _finite(doc.get("metrics")):
        errors.append("result status/metrics invalid")
    if doc.get("evidence_class") != "provisional_engineering_evidence" \
            or doc.get("quality_claims_permitted") is not False \
            or doc.get("source_deletion_permitted") is not False:
        errors.append("result evidence boundary invalid")
    errors.extend(_validate_artifact_rows(doc.get("output_artifacts"),
                                          verify_files=verify_files,
                                          name="output_artifacts"))
    return errors


def build_checkpoint(*, request: dict[str, Any], registry: dict[str, Any], status: str,
                     cursor: str, completed_units: list[str],
                     resume_state_sha256: str,
                     artifact_hashes: list[dict[str, Any]],
                     updated_at: str | None = None) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA, "policy_version": POLICY_VERSION,
        "updated_at": updated_at or _now(), "request_sha256": request["request_sha256"],
        "adapter": dict(request["adapter"]), "status": status, "cursor": cursor,
        "completed_units": list(completed_units), "resume_state_sha256": resume_state_sha256,
        "artifact_hashes": sorted((dict(row) for row in artifact_hashes),
                                  key=lambda row: row.get("role", "")),
        "exact_resume": {"atomic_replace": True, "fsync_file": True,
                         "fsync_parent_directory": True,
                         "validate_all_hashes_before_resume": True,
                         "source_deletion_permitted": False},
    }
    doc["checkpoint_sha256"] = _hash_value(doc)
    errors = validate_checkpoint(doc, request, registry)
    if errors:
        raise ValueError("built invalid checkpoint: " + "; ".join(errors))
    return doc


def validate_checkpoint(doc: Any, request: dict[str, Any], registry: dict[str, Any],
                        verify_files: bool = False) -> list[str]:
    errors = validate_request(request, registry)
    if not isinstance(doc, dict):
        return errors + ["checkpoint is not an object"]
    if set(doc) != CHECKPOINT_KEYS:
        return errors + ["checkpoint keys are not exact"]
    if doc.get("schema") != CHECKPOINT_SCHEMA or doc.get("policy_version") != POLICY_VERSION:
        errors.append("checkpoint schema/policy mismatch")
    if doc.get("checkpoint_sha256") != _hash_value(_without(doc, "checkpoint_sha256")):
        errors.append("checkpoint hash mismatch")
    if doc.get("request_sha256") != request.get("request_sha256") \
            or doc.get("adapter") != request.get("adapter"):
        errors.append("checkpoint request/adapter binding mismatch")
    if doc.get("status") not in {"running", "checkpointed-stop", "complete"} \
            or not isinstance(doc.get("cursor"), str) \
            or not isinstance(doc.get("completed_units"), list) \
            or len(doc["completed_units"]) != len(set(doc["completed_units"])) \
            or not _is_sha(doc.get("resume_state_sha256")):
        errors.append("checkpoint state invalid")
    exact = doc.get("exact_resume")
    if not isinstance(exact, dict) or exact != {
            "atomic_replace": True, "fsync_file": True,
            "fsync_parent_directory": True, "validate_all_hashes_before_resume": True,
            "source_deletion_permitted": False}:
        errors.append("checkpoint exact-resume contract invalid")
    errors.extend(_validate_artifact_rows(doc.get("artifact_hashes"),
                                          verify_files=verify_files,
                                          name="artifact_hashes"))
    return errors


def build_execution_receipt(*, request: dict[str, Any], result: dict[str, Any],
                            registry: dict[str, Any], checkpoint: dict[str, Any],
                            command_argv: list[str], resource_observations: dict[str, Any],
                            phase_resume: dict[str, Any],
                            completed_at: str | None = None) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema": EXECUTION_RECEIPT_SCHEMA, "policy_version": POLICY_VERSION,
        "completed_at": completed_at or _now(), "request_sha256": request["request_sha256"],
        "result_sha256": result["result_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "adapter": dict(request["adapter"]),
        "command_argv_sha256": _hash_value(command_argv),
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "resource_observations": resource_observations, "phase_resume": phase_resume,
        "quality_claims_permitted": False, "source_deletion_permitted": False,
    }
    doc["receipt_sha256"] = _hash_value(doc)
    errors = validate_execution_receipt(doc, request, result, registry,
                                        checkpoint=checkpoint, command_argv=command_argv)
    if errors:
        raise ValueError("built invalid receipt: " + "; ".join(errors))
    return doc


def validate_execution_receipt(doc: Any, request: dict[str, Any],
                               result: dict[str, Any], registry: dict[str, Any],
                               checkpoint: dict[str, Any] | None = None,
                               command_argv: list[str] | None = None) -> list[str]:
    errors = validate_result(result, request, registry)
    if not isinstance(doc, dict):
        return errors + ["execution receipt is not an object"]
    if set(doc) != RECEIPT_KEYS:
        return errors + ["execution receipt keys are not exact"]
    if doc.get("schema") != EXECUTION_RECEIPT_SCHEMA \
            or doc.get("policy_version") != POLICY_VERSION:
        errors.append("receipt schema/policy mismatch")
    if doc.get("receipt_sha256") != _hash_value(_without(doc, "receipt_sha256")):
        errors.append("receipt hash mismatch")
    if doc.get("request_sha256") != request.get("request_sha256") \
            or doc.get("result_sha256") != result.get("result_sha256") \
            or doc.get("registry_sha256") != registry.get("registry_sha256") \
            or doc.get("adapter") != request.get("adapter"):
        errors.append("receipt identity binding mismatch")
    if not _is_sha(doc.get("command_argv_sha256")) \
            or not _is_sha(doc.get("checkpoint_sha256")) \
            or not isinstance(doc.get("resource_observations"), dict) \
            or not isinstance(doc.get("phase_resume"), dict) \
            or not _finite(doc.get("resource_observations")):
        errors.append("receipt execution evidence invalid")
    if doc.get("quality_claims_permitted") is not False \
            or doc.get("source_deletion_permitted") is not False:
        errors.append("receipt claim boundary invalid")
    if checkpoint is not None and doc.get("checkpoint_sha256") != checkpoint.get(
            "checkpoint_sha256"):
        errors.append("receipt checkpoint binding mismatch")
    if command_argv is not None and doc.get("command_argv_sha256") != _hash_value(command_argv):
        errors.append("receipt command binding mismatch")
    return errors


def _selftest() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        source = root / "adapter.py"; source.write_text("# adapter\n")
        executable = Path(sys.executable).resolve()
        source_sha, _ = _sha_file(source); exe_sha, _ = _sha_file(executable)
        entry = {
            "adapter_id": "test-adapter", "adapter_version": "1",
            "source_path": str(source), "source_sha256": source_sha,
            "executable_path": str(executable), "executable_sha256": exe_sha,
            "entrypoint_argv": [str(executable), str(source), "--request", "{request_path}"],
            "operations": ["condense_pilot"], "model_families": ["test"],
            "backends": ["cpu"], "request_schema": REQUEST_SCHEMA,
            "result_schema": RESULT_SCHEMA, "checkpoint_schema": CHECKPOINT_SCHEMA,
            "reviewed": True, "execution_only_not_quality_evidence": True,
        }
        registry = build_registry([entry])
        assert not validate_registry(registry, verify_files=True, base_dir=root)
        inp = root / "input"; inp.write_bytes(b"x"); isha, ibytes = _sha_file(inp)
        spec = root / "spec.json"; spec.write_text("{}\n"); ssha, _ = _sha_file(spec)
        manifest = root / "manifest.json"; manifest.write_text("{}\n"); msha, _ = _sha_file(manifest)
        out = root / "out"; out.mkdir()
        req_path = out / "request.json"
        request = build_request(
            registry=registry, adapter_id="test-adapter", operation="condense_pilot",
            program_spec_sha256="1" * 64, parameter_manifest_path=str(manifest),
            parameter_manifest_sha256=msha, source_census_sha256="2" * 64,
            model_label="tiny", model_family="test", backend="cpu", seed=1,
            inputs=[{"role": "source", "path": str(inp), "sha256": isha, "bytes": ibytes}],
            pilot_spec_path=str(spec), pilot_spec_sha256=ssha, pilot_spec_schema="test.v1",
            request_path=str(req_path), output_dir=str(out), checkpoint_path=str(out/"checkpoint.json"),
            result_path=str(out/"result.json"), execution_receipt_path=str(out/"receipt.json"),
            operator_greenlight_sha256="3" * 64, resource_admission_sha256="4" * 64)
        assert not validate_request(request, registry, verify_files=True)
        assert Path(resolve_command(request, registry)[-1]).resolve() == req_path.resolve()
        artifact = root / "artifact"; artifact.write_bytes(b"result"); asha, abytes = _sha_file(artifact)
        rows = [{"role": "artifact", "path": str(artifact), "sha256": asha, "bytes": abytes}]
        result = build_result(request=request, registry=registry, status="complete",
                              output_artifacts=rows, metrics={"x": 1})
        checkpoint = build_checkpoint(request=request, registry=registry, status="complete",
                                      cursor="done", completed_units=["done"],
                                      resume_state_sha256="5" * 64, artifact_hashes=rows)
        command = resolve_command(request, registry)
        receipt = build_execution_receipt(
            request=request, result=result, registry=registry, checkpoint=checkpoint,
            command_argv=command, resource_observations={"ok": True},
            phase_resume={"completed": True})
        assert not validate_result(result, request, registry, verify_files=True)
        assert not validate_checkpoint(checkpoint, request, registry, verify_files=True)
        assert not validate_execution_receipt(receipt, request, result, registry,
                                              checkpoint=checkpoint, command_argv=command)
    print("doctor_v5_adapter_abi.py selftest OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--kind", required=True,
                          choices=("registry", "request", "result", "receipt", "checkpoint"))
    validate.add_argument("path", type=Path)
    validate.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    validate.add_argument("--request", type=Path)
    validate.add_argument("--result", type=Path)
    validate.add_argument("--verify-files", action="store_true")
    sub.add_parser("selftest")
    args = parser.parse_args()
    if args.command == "selftest":
        _selftest(); return 0
    doc = read_json(args.path)
    if args.kind == "registry":
        errors = validate_registry(doc, verify_files=args.verify_files, base_dir=ROOT)
    else:
        registry = read_json(args.registry)
        if args.request is None:
            raise SystemExit("--request required")
        request = read_json(args.request)
        if args.kind == "request":
            errors = validate_request(doc, registry, verify_files=args.verify_files)
        elif args.kind == "result":
            errors = validate_result(doc, request, registry, verify_files=args.verify_files)
        elif args.kind == "checkpoint":
            errors = validate_checkpoint(doc, request, registry, verify_files=args.verify_files)
        else:
            if args.result is None:
                raise SystemExit("--result required")
            errors = validate_execution_receipt(doc, request, read_json(args.result), registry)
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
