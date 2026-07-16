#!/usr/bin/env python3.12
"""Unbound single-device shared-preprocessing cache and fanout contracts.

The cache is deliberately bounded to one source tensor/expert batch at a time.
It may share immutable source reads, an exact decode, and original-value rank
statistics.  A zeroed-bulk/RHT artifact is shared only by consumers with the
same complete preprocessing signature.  Trellis geometry, codebook training,
adaptive scales, encoding, reconstruction, branch evidence, and scientific
receipts are always per-rate/per-branch.

This module is not imported by the live Doctor queue or workers.  It reads no
model payload and changes no runtime default.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys
import time
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import doctor_v5_gptoss_mxfp4 as mxfp4
import doctor_v5_aggressive_admission_policy as aggressive_admission
import doctor_v5_gptoss_parallel_scaffold as gptoss_parallel
import doctor_v5_gptoss_reuse_fanout as gptoss_fanout
import doctor_v5_higher_tier_scaffold as higher_tier


MANIFEST_SCHEMA = "hawking.doctor_v5_shared_preprocess_consumer_manifest.v1"
QWEN_INVENTORY_SCHEMA = "hawking.doctor_v5_qwen_unbound_tensor_inventory.v1"
PLAN_SCHEMA = "hawking.doctor_v5_shared_preprocess_cache_plan.v1"
CACHE_RECEIPT_SCHEMA = "hawking.doctor_v5_shared_preprocess_cache_receipt.v1"
RESOURCE_RECEIPT_SCHEMA = "hawking.doctor_v5_shared_preprocess_resource_receipt.v1"
SYNTHETIC_RESOURCE_SCOPE = "synthetic-nonpromotable"
PRODUCTION_RESOURCE_SCOPE = "production-local-observer"
TRUSTED_PRODUCTION_RESOURCE_OBSERVER_AVAILABLE = False
OUTPUT_RECEIPT_SCHEMA = "hawking.doctor_v5_shared_preprocess_output_receipt.v1"
STATE_SCHEMA = "hawking.doctor_v5_shared_preprocess_resume_state.v1"
MERGE_SCHEMA = "hawking.doctor_v5_shared_preprocess_merge.v1"
REQUIREMENTS_SCHEMA = "hawking.doctor_v5_shared_preprocess_requirements.v1"
OUTPUT_ROOT = ROOT / "reports/condense/doctor_v5_unbound/shared_preprocess"
DEFAULT_REQUIREMENTS = OUTPUT_ROOT / "requirements.json"
DEFAULT_GPTOSS_MANIFEST = OUTPUT_ROOT / "gptoss_120b_consumer_manifest.json"
DEFAULT_GPTOSS_PLAN = OUTPUT_ROOT / "gptoss_120b_cache_plan.json"
DEFAULT_QWEN_MANIFEST = OUTPUT_ROOT / "qwen_consumer_manifest.json"
DEFAULT_QWEN_PLAN = OUTPUT_ROOT / "qwen_cache_plan.json"
SWAP_CONTROLLER_PATH = HERE / "doctor_v5_aggressive_admission_policy.py"
SHA_RE = re.compile(r"[0-9a-f]{64}")
MAX_JSON_BYTES = 256 * 1024 * 1024
DEFAULT_PROCESS_BUDGET_BYTES = 66_000_000_000
DEFAULT_CACHE_UNIT_LIMIT_BYTES = 8_000_000_000
DEFAULT_DISK_RESERVE_BYTES = 64_000_000_000
DEFAULT_MAX_ACTIVE_CACHE_UNITS = 1


class SharedCacheError(RuntimeError):
    """A cache, fanout, parity, or lifecycle contract is unsafe."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(doc: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in doc.items() if key != field}


def _file_artifact(path: Path) -> dict[str, Any]:
    candidate = Path(path)
    original = candidate.lstat()
    if stat.S_ISLNK(original.st_mode) or not stat.S_ISREG(original.st_mode):
        raise SharedCacheError(f"integration source is not a regular non-symlink: {path}")
    resolved = candidate.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise SharedCacheError(f"integration source is not a regular file: {path}")
    raw = resolved.read_bytes()
    return {"path": str(resolved), "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw)}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) \
                or info.st_size > MAX_JSON_BYTES:
            raise SharedCacheError(f"invalid JSON file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SharedCacheError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SharedCacheError(f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    root = OUTPUT_ROOT.resolve()
    candidate = Path(path).expanduser().absolute()
    resolved = candidate.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise SharedCacheError(f"output must remain inside the unbound root: {path}")
    cursor = candidate.parent
    while cursor != cursor.parent:
        if cursor.exists() and cursor.is_symlink():
            raise SharedCacheError(f"output parent is a symlink: {cursor}")
        if cursor.resolve(strict=False) == root:
            break
        cursor = cursor.parent
    if candidate.exists() and candidate.is_symlink():
        raise SharedCacheError(f"output is a symlink: {candidate}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    mxfp4._atomic_json(candidate, value)


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _artifact_valid(value: Any, *, instance: bool = True) -> bool:
    required = {"sha256", "bytes"} | ({"artifact_instance_id"} if instance else set())
    return isinstance(value, dict) and required <= set(value) \
        and _valid_sha(value.get("sha256")) \
        and not isinstance(value.get("bytes"), bool) \
        and isinstance(value.get("bytes"), int) and value["bytes"] >= 0 \
        and (not instance or isinstance(value.get("artifact_instance_id"), str)
             and bool(value["artifact_instance_id"]))


def shareability_audit() -> dict[str, Any]:
    """Exact audit boundary derived from the current Qwen preprocessing order."""
    return {
        "shareable_with_exact_source_and_implementation_binding": [
            "bounded immutable source-range read",
            "BF16/MXFP4 decode only after exact decoder parity qualification",
            "original-value nonfinite bitmap and absolute-magnitude rank permutation",
        ],
        "conditionally_shareable_by_complete_preprocess_signature": [
            "outlier prefix selection and zeroed bulk keyed by exact outlier percentage",
            "forward RHT keyed by zeroed bulk, orientation, tensor seed, and implementation",
        ],
        "never_cross_rate_or_branch_shared": [
            "outlier value quantization", "trellis geometry", "learned codebook",
            "adaptive scale/min search", "Viterbi symbols", "reconstruction",
            "side-information encoding", "attestation", "quality evaluation",
            "method/scientific evidence",
        ],
        "reason_rht_is_not_globally_shareable": (
            "the current encoder removes branch/rate-selected outliers before RHT; "
            "different outlier percentages produce different transform inputs"
        ),
        "gptoss_limit": (
            "source-range read reuse is structural; decoded/RHT reuse stays blocked until "
            "the missing GPT-OSS numerical decoder and recipe signatures qualify"
        ),
    }


def build_requirements_packet(*, created_at: str | None = None) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema": REQUIREMENTS_SCHEMA, "created_at": created_at or _now(),
        "status": "unbound-default-off-no-live-import",
        "shareability_audit": shareability_audit(),
        "integration_sources": {
            "shared_cache_builder": _file_artifact(Path(__file__)),
            "gptoss_fanout": _file_artifact(Path(gptoss_fanout.__file__)),
            "gptoss_parallel": _file_artifact(Path(gptoss_parallel.__file__)),
            "higher_tier": _file_artifact(Path(higher_tier.__file__)),
            "swap_controller": _file_artifact(SWAP_CONTROLLER_PATH),
        },
        "cache_shape": {
            "whole_model_cache_permitted": False,
            "bounded_tensor_or_expert_batch_only": True,
            "maximum_active_cache_units_default": DEFAULT_MAX_ACTIVE_CACHE_UNITS,
            "maximum_cache_unit_bytes_default": DEFAULT_CACHE_UNIT_LIMIT_BYTES,
            "disk_reserve_bytes_default": DEFAULT_DISK_RESERVE_BYTES,
            "refcounted_ephemeral_release_only_after_all_consumers_validate": True,
        },
        "benchmark_contract": {
            "exact_input_cache_output_scientific_receipt_identities": True,
            "scheduled_executed_validated_counts_required": True,
            "skipped_must_equal_zero": True,
            "serial_output_sha256_equality_required": True,
        },
        "execution_permitted": False, "runtime_defaults_changed": False,
        "source_deletion_permitted": False, "quality_claims_permitted": False,
    }
    doc["requirements_sha256"] = _hash_value(doc)
    return doc


def validate_consumer_manifest(doc: Any) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != MANIFEST_SCHEMA:
        return ["consumer-manifest schema mismatch"]
    errors: list[str] = []
    if doc.get("manifest_sha256") != _hash_value(_without(doc, "manifest_sha256")):
        errors.append("consumer-manifest hash mismatch")
    if doc.get("status") != "unbound-input-only" \
            or doc.get("family") not in {"qwen2.5-dense", "gpt-oss-moe", "higher-tier"} \
            or doc.get("source_deletion_permitted") is not False:
        errors.append("consumer manifest scope/lifecycle is invalid")
    parents = doc.get("parent_bindings")
    if not isinstance(parents, dict) or not parents \
            or any(not isinstance(key, str) or not key or not _valid_sha(value)
                   for key, value in parents.items()):
        errors.append("consumer manifest parent bindings are invalid")
    sources = doc.get("source_units")
    if not isinstance(sources, list) or not sources:
        errors.append("consumer manifest has no bounded source units")
        sources = []
    source_by_id: dict[str, dict[str, Any]] = {}
    for row in sources:
        source_id = row.get("source_unit_id") if isinstance(row, dict) else None
        decode = row.get("decode_contract") if isinstance(row, dict) else None
        numeric = (row.get("logical_parameters"), row.get("estimated_source_read_bytes"),
                   row.get("estimated_decoded_bytes")) if isinstance(row, dict) else ()
        if not isinstance(source_id, str) or not source_id or source_id in source_by_id \
                or not _valid_sha(row.get("source_binding_sha256")) \
                or len(numeric) != 3 or any(isinstance(value, bool)
                                             or not isinstance(value, int) or value < 0
                                             for value in numeric) \
                or not isinstance(decode, dict) \
                or decode.get("status") not in {"qualified", "missing"}:
            errors.append(f"bounded source/decode identity is invalid: {source_id}")
            continue
        if decode["status"] == "qualified" and (
                decode.get("exact_serial_parity") is not True
                or not _valid_sha(decode.get("implementation_sha256"))
                or decode.get("output_dtype") != "F32"
                or not isinstance(decode.get("input_dtype"), str)
                or not _valid_sha(decode.get("layout_sha256"))
                or row["estimated_decoded_bytes"] < 4 * row["logical_parameters"]):
            errors.append(f"qualified decode contract is incomplete: {source_id}")
        if decode["status"] == "missing" and (
                decode.get("exact_serial_parity") is not False
                or not isinstance(decode.get("blocker"), str) or not decode["blocker"]):
            errors.append(f"missing decode contract lacks blocker: {source_id}")
        source_by_id[source_id] = row
    consumers = doc.get("consumers")
    if not isinstance(consumers, list) or not consumers:
        errors.append("consumer manifest has no fanout consumers")
        consumers = []
    seen: set[str] = set()
    source_use: set[str] = set()
    for row in consumers:
        consumer_id = row.get("consumer_id") if isinstance(row, dict) else None
        signature = row.get("preprocess_signature") if isinstance(row, dict) else None
        source_id = row.get("source_unit_id") if isinstance(row, dict) else None
        if not isinstance(consumer_id, str) or not consumer_id or consumer_id in seen \
                or source_id not in source_by_id \
                or not isinstance(row.get("rate_id"), str) or not row["rate_id"] \
                or not isinstance(row.get("branch"), str) or not row["branch"] \
                or not _valid_sha(row.get("cell_identity_sha256")) \
                or not _valid_sha(row.get("rate_parameters_sha256")) \
                or not _valid_sha(row.get("branch_parameters_sha256")) \
                or not isinstance(signature, dict) \
                or signature.get("status") not in {"qualified", "missing"}:
            errors.append(f"fanout consumer identity is invalid: {consumer_id}")
            continue
        if signature["status"] == "qualified":
            pct = signature.get("outlier_pct_decimal")
            try:
                pct_value = float(pct)
            except (TypeError, ValueError):
                pct_value = math.nan
            if not isinstance(pct, str) or not math.isfinite(pct_value) \
                    or not 0 <= pct_value <= 100 or str(pct_value) != pct \
                    or signature.get("use_rht") not in {True, False} \
                    or signature.get("rht_cols") not in {True, False} \
                    or not _valid_sha(signature.get("rht_seed_sha256")) \
                    or not _valid_sha(signature.get("implementation_sha256")) \
                    or any(not _valid_sha(signature.get(field))
                           for field in QWEN_PREPROCESS_AUTHORITY_FIELDS):
                errors.append(f"qualified preprocess signature is incomplete: {consumer_id}")
        elif not isinstance(signature.get("blocker"), str) or not signature["blocker"]:
            errors.append(f"missing preprocess signature lacks blocker: {consumer_id}")
        seen.add(consumer_id); source_use.add(source_id)
    if source_use != set(source_by_id):
        errors.append("one or more cache source units has no consumer")
    coverage = doc.get("coverage")
    if not isinstance(coverage, dict) \
            or coverage.get("source_unit_count") != len(sources) \
            or coverage.get("consumer_count") != len(consumers) \
            or coverage.get("consumer_ids_sha256") \
            != _hash_value(sorted(row.get("consumer_id") for row in consumers
                                  if isinstance(row, dict))):
        errors.append("consumer manifest coverage seal is invalid")
    return errors


def build_gptoss_consumer_manifest(
    work_plan: dict[str, Any], fanout_plan: dict[str, Any], *,
    created_at: str | None = None,
) -> dict[str, Any]:
    work_errors = gptoss_parallel.validate_work_plan(work_plan)
    fanout_errors = gptoss_fanout.validate_fanout_plan(
        fanout_plan, work_plan,
        _read_json(gptoss_parallel.DEFAULT_PENDING_WIRING),
    )
    if work_errors or fanout_errors:
        raise SharedCacheError("GPT-OSS parent scaffold is invalid: "
                               + "; ".join(work_errors + fanout_errors))
    source_units = []
    for source in work_plan["source_units"]:
        staging = source["staging"]
        source_units.append({
            "source_unit_id": source["unit_id"],
            "source_binding_sha256": source["source_binding_sha256"],
            "logical_parameters": source["logical_parameters"],
            "estimated_source_read_bytes": staging["maximum_materialized_bytes"],
            "estimated_decoded_bytes": staging["maximum_materialized_bytes"],
            "bounded_unit_kind": source["kind"],
            "decode_contract": {
                "status": "missing", "exact_serial_parity": False,
                "blocker": (
                    "GPT-OSS numerical MXFP4/BF16 source-unit decoder and exact recipe "
                    "parity are not qualified; only immutable source reads may be shared"
                ),
            },
        })
    consumers = []
    for job in fanout_plan["jobs"]:
        consumers.append({
            "consumer_id": job["job_id"], "source_unit_id": job["source_unit_id"],
            "rate_id": job["rate_id"], "branch": job["branch"],
            "cell_identity_sha256": job["cell_identity_sha256"],
            "rate_parameters_sha256": _hash_value({
                "rate_id": job["rate_id"], "cell_spec_sha256": job["cell_spec_sha256"],
            }),
            "branch_parameters_sha256": _hash_value({
                "branch": job["branch"], "adapter_id": job["adapter_id"],
                "command": job["command"],
            }),
            "preprocess_signature": {
                "status": "missing",
                "blocker": "GPT-OSS rate/branch outlier and RHT recipe is not qualified",
            },
        })
    consumer_ids = sorted(row["consumer_id"] for row in consumers)
    doc: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA, "created_at": created_at or _now(),
        "status": "unbound-input-only", "family": "gpt-oss-moe",
        "parent_bindings": {
            "work_plan_sha256": work_plan["work_plan_sha256"],
            "fanout_plan_sha256": fanout_plan["fanout_plan_sha256"],
            "shared_cache_builder_sha256": _file_artifact(Path(__file__))["sha256"],
        },
        "source_units": source_units, "consumers": consumers,
        "coverage": {"source_unit_count": len(source_units),
                     "consumer_count": len(consumers),
                     "consumer_ids_sha256": _hash_value(consumer_ids)},
        "source_deletion_permitted": False,
    }
    doc["manifest_sha256"] = _hash_value(doc)
    errors = validate_consumer_manifest(doc)
    if errors:
        raise SharedCacheError("generated GPT-OSS manifest invalid: " + "; ".join(errors))
    return doc


QWEN_BRANCHES = ("codec_control", "doctor_static", "doctor_conditional", "doctor_full")
QWEN_PREPROCESS_AUTHORITY_FIELDS = (
    "adapter_artifact_sha256", "runtime_spec_authority_sha256",
    "recipe_authority_sha256",
)


def _canonical_decimal(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return isinstance(value, str) and math.isfinite(parsed) and 0 <= parsed <= 100 \
        and str(parsed) == value


def validate_qwen_inventory(doc: Any) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != QWEN_INVENTORY_SCHEMA:
        return ["Qwen inventory schema mismatch"]
    errors: list[str] = []
    if doc.get("inventory_sha256") != _hash_value(_without(doc, "inventory_sha256")):
        errors.append("Qwen inventory hash mismatch")
    if doc.get("status") != "unbound-input-only" \
            or doc.get("source_deletion_permitted") is not False:
        errors.append("Qwen inventory is not unbound/fail-closed")
    parents = doc.get("parent_bindings")
    if not isinstance(parents, dict) or not parents \
            or any(not isinstance(key, str) or not key or not _valid_sha(value)
                   for key, value in parents.items()):
        errors.append("Qwen parent bindings are invalid")
    recipes = doc.get("branch_preprocess_recipes", [])
    if not isinstance(recipes, list):
        errors.append("Qwen branch preprocess recipe matrix is invalid")
        recipes = []
    expected_cells = {(rate_id, branch) for rate_id in gptoss_parallel.RATES
                      for branch in QWEN_BRANCHES}
    observed_cells: set[tuple[str, str]] = set()
    for row in recipes:
        cell = (row.get("rate_id"), row.get("branch")) if isinstance(row, dict) \
            else (None, None)
        if cell in observed_cells or cell not in expected_cells \
                or row.get("status") not in {"qualified", "missing"}:
            errors.append(f"Qwen preprocess recipe cell is invalid/duplicate: {cell}")
            continue
        if row["status"] == "qualified" and (
                not _canonical_decimal(row.get("outlier_pct_decimal"))
                or row.get("use_rht") not in {True, False}
                or row.get("rht_cols") not in {True, False}
                or any(not _valid_sha(row.get(field))
                       for field in QWEN_PREPROCESS_AUTHORITY_FIELDS)):
            errors.append(f"Qwen qualified preprocess authority is incomplete: {cell}")
        if row["status"] == "missing" and (
                not isinstance(row.get("blocker"), str) or not row["blocker"]):
            errors.append(f"Qwen missing preprocess authority lacks blocker: {cell}")
        observed_cells.add(cell)
    if recipes and observed_cells != expected_cells:
        errors.append("Qwen preprocess authority must cover the exact canonical 10x4 matrix")
    units = doc.get("source_units")
    if not isinstance(units, list) or not units:
        errors.append("Qwen inventory has no bounded tensor units")
        units = []
    seen: set[str] = set()
    for row in units:
        unit_id = row.get("source_unit_id") if isinstance(row, dict) else None
        decode = row.get("decode_contract") if isinstance(row, dict) else None
        integers = tuple(row.get(key) for key in (
            "logical_parameters", "estimated_source_read_bytes",
            "estimated_decoded_bytes")) if isinstance(row, dict) else ()
        if not isinstance(unit_id, str) or not unit_id or unit_id in seen \
                or not _valid_sha(row.get("source_binding_sha256")) \
                or not _valid_sha(row.get("preprocess_implementation_sha256")) \
                or not _valid_sha(row.get("rht_seed_sha256")) \
                or len(integers) != 3 \
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                       for value in integers) or not isinstance(decode, dict):
            errors.append(f"Qwen bounded tensor identity is invalid: {unit_id}")
            continue
        decode_status = decode.get("status")
        if decode_status not in {"qualified", "missing"} \
                or decode_status == "qualified" and (
                    decode.get("exact_serial_parity") is not True
                    or not _valid_sha(decode.get("implementation_sha256"))
                    or decode.get("output_dtype") != "F32"
                    or not isinstance(decode.get("input_dtype"), str)
                    or not _valid_sha(decode.get("layout_sha256"))
                    or row["estimated_decoded_bytes"] < 4 * row["logical_parameters"]) \
                or decode_status == "missing" and (
                    decode.get("exact_serial_parity") is not False
                    or not isinstance(decode.get("blocker"), str)
                    or not decode["blocker"]):
            errors.append(f"Qwen decoder qualification is invalid: {unit_id}")
        seen.add(unit_id)
    return errors


def build_qwen_consumer_manifest(inventory: dict[str, Any], *,
                                 created_at: str | None = None) -> dict[str, Any]:
    """Expand an immutable unbound tensor inventory into the exact 10x4 fanout."""
    errors = validate_qwen_inventory(inventory)
    if errors:
        raise SharedCacheError("cannot build from invalid Qwen inventory: "
                               + "; ".join(errors))
    sources: list[dict[str, Any]] = []
    consumers: list[dict[str, Any]] = []
    recipe_by_cell = {(row["rate_id"], row["branch"]): row
                      for row in inventory.get("branch_preprocess_recipes", [])}
    for source in inventory["source_units"]:
        source_id = source["source_unit_id"]
        sources.append({key: source[key] for key in (
            "source_unit_id", "source_binding_sha256", "logical_parameters",
            "estimated_source_read_bytes", "estimated_decoded_bytes",
            "decode_contract")})
        sources[-1]["bounded_unit_kind"] = "qwen-tensor"
        for rate_id in gptoss_parallel.RATES:
            rate_authority = {"rate_id": rate_id}
            for branch in QWEN_BRANCHES:
                recipe = recipe_by_cell.get((rate_id, branch), {
                    "rate_id": rate_id, "branch": branch, "status": "missing",
                    "blocker": (
                        "no exact adapter/runtime-spec/recipe authority is present in the "
                        "unbound canonical 10x4 inventory"
                    ),
                })
                if recipe["status"] == "qualified":
                    signature = {key: recipe[key] for key in (
                        "status", "outlier_pct_decimal", "use_rht", "rht_cols",
                        *QWEN_PREPROCESS_AUTHORITY_FIELDS)}
                    signature.update({
                        "rht_seed_sha256": source["rht_seed_sha256"],
                        "implementation_sha256": source[
                            "preprocess_implementation_sha256"],
                    })
                else:
                    signature = recipe
                branch_authority = {"branch": branch, "recipe": recipe}
                cell_authority = {"inventory_sha256": inventory["inventory_sha256"],
                                  "source_unit_id": source_id,
                                  "rate": rate_authority, "branch": branch_authority}
                consumers.append({
                    "consumer_id": f"{source_id}/rate={rate_id}/branch={branch}",
                    "source_unit_id": source_id, "rate_id": rate_id, "branch": branch,
                    "cell_identity_sha256": _hash_value(cell_authority),
                    "rate_parameters_sha256": _hash_value(rate_authority),
                    "branch_parameters_sha256": _hash_value(branch_authority),
                    "preprocess_signature": signature,
                })
    ids = sorted(row["consumer_id"] for row in consumers)
    doc: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA, "created_at": created_at or _now(),
        "status": "unbound-input-only", "family": "qwen2.5-dense",
        "parent_bindings": {**inventory["parent_bindings"],
                            "qwen_inventory_sha256": inventory["inventory_sha256"],
                            "shared_cache_builder_sha256": _file_artifact(
                                Path(__file__))["sha256"]},
        "source_units": sources, "consumers": consumers,
        "coverage": {"source_unit_count": len(sources),
                     "consumer_count": len(consumers),
                     "consumer_ids_sha256": _hash_value(ids)},
        "source_deletion_permitted": False,
    }
    doc["manifest_sha256"] = _hash_value(doc)
    manifest_errors = validate_consumer_manifest(doc)
    if manifest_errors:
        raise SharedCacheError("generated Qwen manifest invalid: "
                               + "; ".join(manifest_errors))
    return doc


def _base_cache_id(source_id: str) -> str:
    return f"{source_id}/cache=source-decode-stats"


def _derived_key(source: dict[str, Any], signature: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_binding_sha256": source["source_binding_sha256"],
        "decode_implementation_sha256": source["decode_contract"].get(
            "implementation_sha256"
        ),
        "outlier_pct_decimal": signature["outlier_pct_decimal"],
        "use_rht": signature["use_rht"], "rht_cols": signature["rht_cols"],
        "rht_seed_sha256": signature["rht_seed_sha256"],
        "implementation_sha256": signature["implementation_sha256"],
        **{field: signature[field] for field in QWEN_PREPROCESS_AUTHORITY_FIELDS},
    }


def _derived_cache_id(source_id: str, key: dict[str, Any]) -> str:
    return f"{source_id}/cache=bulk-rht/{_hash_value(key)}"


def _expected_cache_units(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    source_by_id = {row["source_unit_id"]: row for row in manifest["source_units"]}
    consumers_by_source: dict[str, list[dict[str, Any]]] = {}
    for consumer in manifest["consumers"]:
        consumers_by_source.setdefault(consumer["source_unit_id"], []).append(consumer)
    units: list[dict[str, Any]] = []
    for source_id, source in sorted(source_by_id.items()):
        consumers = sorted(row["consumer_id"] for row in consumers_by_source[source_id])
        qualified = source["decode_contract"]["status"] == "qualified"
        stages = ["immutable_source_range_read"]
        if qualified:
            stages += ["exact_decode_to_f32", "nonfinite_bitmap_and_abs_rank"]
        # Conservatively charge source buffer + decoded F32 + abs values/u64 rank/
        # bitmap and allocator headroom.  This intentionally overstates rather than
        # treating a whole-tensor argsort as free.
        rank_statistics_bytes = source["logical_parameters"] * 16 if qualified else 0
        base_resident_bytes = source["estimated_source_read_bytes"]
        base_disk_bytes = source["estimated_source_read_bytes"]
        if qualified:
            base_resident_bytes += source["estimated_decoded_bytes"] \
                + rank_statistics_bytes
            base_disk_bytes += source["estimated_decoded_bytes"] \
                + rank_statistics_bytes
        units.append({
            "cache_unit_id": _base_cache_id(source_id), "source_unit_id": source_id,
            "kind": "base-source-decode-statistics", "stages": stages,
            "source_binding_sha256": source["source_binding_sha256"],
            "estimated_resident_bytes": base_resident_bytes,
            "estimated_disk_bytes": base_disk_bytes,
            "consumer_count": len(consumers),
            "consumer_ids_sha256": _hash_value(consumers),
            "complete_signature_required": True,
        })
        groups: dict[str, dict[str, Any]] = {}
        if qualified:
            for consumer in consumers_by_source[source_id]:
                signature = consumer["preprocess_signature"]
                if signature["status"] != "qualified":
                    continue
                key = _derived_key(source, signature)
                group = groups.setdefault(_hash_value(key), {"key": key, "consumers": []})
                group["consumers"].append(consumer["consumer_id"])
        for group in groups.values():
            consumers = sorted(group["consumers"])
            stages = ["outlier_prefix_and_zeroed_bulk"]
            if group["key"]["use_rht"]:
                stages.append("forward_rht")
            units.append({
                "cache_unit_id": _derived_cache_id(source_id, group["key"]),
                "source_unit_id": source_id, "kind": "exact-bulk-rht-group",
                "stages": stages, "source_binding_sha256": source["source_binding_sha256"],
                "preprocess_key": group["key"],
                # Input F32 + zeroed/RHT output + u64 outlier/rank workspace.
                "estimated_resident_bytes": 2 * source["estimated_decoded_bytes"]
                    + 8 * source["logical_parameters"],
                "estimated_disk_bytes": source["estimated_decoded_bytes"]
                    + 8 * source["logical_parameters"],
                "consumer_count": len(consumers),
                "consumer_ids_sha256": _hash_value(consumers),
                "complete_signature_required": True,
            })
    units.sort(key=lambda row: row["cache_unit_id"])
    return units


def _consumer_route(manifest: dict[str, Any], consumer: dict[str, Any]) -> list[str]:
    source = {row["source_unit_id"]: row for row in manifest["source_units"]}[
        consumer["source_unit_id"]
    ]
    route = [_base_cache_id(source["source_unit_id"])]
    signature = consumer["preprocess_signature"]
    if source["decode_contract"]["status"] == "qualified" \
            and signature["status"] == "qualified":
        route.append(_derived_cache_id(source["source_unit_id"],
                                       _derived_key(source, signature)))
    return route


def build_cache_plan(
    manifest: dict[str, Any], *, process_budget_bytes: int = DEFAULT_PROCESS_BUDGET_BYTES,
    cache_unit_limit_bytes: int = DEFAULT_CACHE_UNIT_LIMIT_BYTES,
    disk_reserve_bytes: int = DEFAULT_DISK_RESERVE_BYTES,
    maximum_active_cache_units: int = DEFAULT_MAX_ACTIVE_CACHE_UNITS,
    created_at: str | None = None,
) -> dict[str, Any]:
    errors = validate_consumer_manifest(manifest)
    if errors:
        raise SharedCacheError("cannot plan invalid consumer manifest: " + "; ".join(errors))
    values = (process_budget_bytes, cache_unit_limit_bytes, disk_reserve_bytes,
              maximum_active_cache_units)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in values) or cache_unit_limit_bytes > process_budget_bytes \
            or maximum_active_cache_units != 1:
        raise SharedCacheError("single-device cache resource envelope is invalid")
    units = _expected_cache_units(manifest)
    consumers = sorted(manifest["consumers"], key=lambda row: row["consumer_id"])
    blocked = [row["consumer_id"] for row in consumers
               if len(_consumer_route(manifest, row)) == 1]
    namespaces = [_hash_value({
        "manifest_sha256": manifest["manifest_sha256"],
        "consumer_id": row["consumer_id"], "source_unit_id": row["source_unit_id"],
        "rate_id": row["rate_id"], "branch": row["branch"],
    }) for row in consumers]
    doc: dict[str, Any] = {
        "schema": PLAN_SCHEMA, "created_at": created_at or _now(),
        "status": "unbound-default-off", "family": manifest["family"],
        "consumer_manifest_sha256": manifest["manifest_sha256"],
        "parent_bindings": manifest["parent_bindings"],
        "shareability_audit_sha256": _hash_value(shareability_audit()),
        "cache_units": units,
        "coverage": {
            "source_unit_count": len(manifest["source_units"]),
            "consumer_count": len(consumers),
            "consumer_ids_sha256": manifest["coverage"]["consumer_ids_sha256"],
            "evidence_namespaces_sha256": _hash_value(namespaces),
            "unique_evidence_namespace_count": len(set(namespaces)),
            "consumers_without_qualified_derived_preprocess": len(blocked),
            "blocked_consumer_ids_sha256": _hash_value(blocked),
        },
        "resources": {
            "process_budget_bytes": process_budget_bytes,
            "cache_unit_limit_bytes": cache_unit_limit_bytes,
            "disk_reserve_bytes": disk_reserve_bytes,
            "maximum_active_cache_units": 1,
            "aggregate_process_tree_rss_required": True,
            "reviewed_swap_admit_decision_required": True,
            "measured_peak_required_before_promotion": True,
            "swap_controller_artifact": _file_artifact(SWAP_CONTROLLER_PATH),
        },
        "lifecycle": {
            "whole_model_cache_permitted": False,
            "cache_units_are_bounded_and_ephemeral": True,
            "cache_release_requires_all_expected_consumers_validated": True,
            "crash_resume_adopts_only_valid_hash_bound_receipts": True,
            "parent_source_deletion_permitted": False,
            "evidence_deletion_permitted": False,
        },
        "serial_equivalence_gate": {
            "every_cache_component_byte_exact": True,
            "every_output_byte_exact_to_serial": True,
            "full_no_skip_coverage_required": True,
            "promotion_currently_permitted": False,
        },
        "execution_permitted": False, "runtime_defaults_changed": False,
        "quality_claims_permitted": False,
    }
    doc["cache_plan_sha256"] = _hash_value(doc)
    plan_errors = validate_cache_plan(doc, manifest)
    if plan_errors:
        raise SharedCacheError("generated cache plan invalid: " + "; ".join(plan_errors))
    return doc


def validate_cache_plan(doc: Any, manifest: dict[str, Any]) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != PLAN_SCHEMA:
        return ["cache-plan schema mismatch"]
    errors = validate_consumer_manifest(manifest)
    if doc.get("cache_plan_sha256") != _hash_value(_without(doc, "cache_plan_sha256")):
        errors.append("cache-plan hash mismatch")
    if doc.get("consumer_manifest_sha256") != manifest.get("manifest_sha256") \
            or doc.get("parent_bindings") != manifest.get("parent_bindings"):
        errors.append("cache plan parent/consumer binding differs")
    if doc.get("shareability_audit_sha256") != _hash_value(shareability_audit()):
        errors.append("cache plan shareability audit seal differs")
    if doc.get("cache_units") != _expected_cache_units(manifest):
        errors.append("cache unit grouping differs from exact shareability signatures")
    consumers = sorted(manifest.get("consumers", []), key=lambda row: row["consumer_id"])
    blocked = [row["consumer_id"] for row in consumers
               if len(_consumer_route(manifest, row)) == 1]
    namespaces = [_hash_value({
        "manifest_sha256": manifest["manifest_sha256"],
        "consumer_id": row["consumer_id"], "source_unit_id": row["source_unit_id"],
        "rate_id": row["rate_id"], "branch": row["branch"],
    }) for row in consumers]
    expected_coverage = {
        "source_unit_count": len(manifest.get("source_units", [])),
        "consumer_count": len(consumers),
        "consumer_ids_sha256": manifest.get("coverage", {}).get("consumer_ids_sha256"),
        "evidence_namespaces_sha256": _hash_value(namespaces),
        "unique_evidence_namespace_count": len(set(namespaces)),
        "consumers_without_qualified_derived_preprocess": len(blocked),
        "blocked_consumer_ids_sha256": _hash_value(blocked),
    }
    if doc.get("coverage") != expected_coverage:
        errors.append("cache plan consumer/evidence coverage differs")
    resources = doc.get("resources")
    if not isinstance(resources, dict) \
            or resources.get("maximum_active_cache_units") != 1 \
            or any(isinstance(resources.get(field), bool)
                   or not isinstance(resources.get(field), int)
                   or resources[field] <= 0 for field in (
                       "process_budget_bytes", "cache_unit_limit_bytes",
                       "disk_reserve_bytes")) \
            or resources.get("cache_unit_limit_bytes", 0) \
            > resources.get("process_budget_bytes", 0):
        errors.append("cache plan bounded resource envelope is invalid")
    if isinstance(resources, dict) and resources.get("swap_controller_artifact") \
            != _file_artifact(SWAP_CONTROLLER_PATH):
        errors.append("cache plan swap controller binding differs")
    lifecycle = doc.get("lifecycle")
    if doc.get("status") != "unbound-default-off" \
            or doc.get("execution_permitted") is not False \
            or doc.get("runtime_defaults_changed") is not False \
            or doc.get("quality_claims_permitted") is not False \
            or not isinstance(lifecycle, dict) \
            or lifecycle.get("whole_model_cache_permitted") is not False \
            or lifecycle.get("parent_source_deletion_permitted") is not False \
            or lifecycle.get("cache_release_requires_all_expected_consumers_validated") \
            is not True:
        errors.append("cache plan execution/lifecycle is not fail closed")
    if doc.get("serial_equivalence_gate") != {
            "every_cache_component_byte_exact": True,
            "every_output_byte_exact_to_serial": True,
            "full_no_skip_coverage_required": True,
            "promotion_currently_permitted": False,
    }:
        errors.append("cache plan serial-equivalence gate differs")
    return errors


def admit_cache_unit(
    plan: dict[str, Any], manifest: dict[str, Any], *, cache_unit_id: str,
    active_cache_units: int, active_reserved_bytes: int,
    aggregate_process_tree_rss_bytes: int, free_disk_bytes: int,
    swap_decision: dict[str, Any], memory_pressure: str, thermal_state: str,
    observed_at_epoch: float, scope: str = SYNTHETIC_RESOURCE_SCOPE,
) -> dict[str, Any]:
    errors = validate_cache_plan(plan, manifest)
    unit = {row["cache_unit_id"]: row for row in plan.get("cache_units", [])}.get(
        cache_unit_id
    )
    integers = (active_cache_units, active_reserved_bytes,
                aggregate_process_tree_rss_bytes, free_disk_bytes)
    if errors or unit is None or any(isinstance(value, bool) or not isinstance(value, int)
                                     or value < 0 for value in integers) \
            or isinstance(observed_at_epoch, bool) \
            or not isinstance(observed_at_epoch, (int, float)) \
            or not math.isfinite(float(observed_at_epoch)) \
            or not isinstance(memory_pressure, str) \
            or not isinstance(thermal_state, str) \
            or not isinstance(scope, str):
        return {"schema": RESOURCE_RECEIPT_SCHEMA, "admitted": False,
                "reason": "invalid plan/unit/resource sample"}
    resources = plan["resources"]
    resource_sample = {
        "active_cache_units": active_cache_units,
        "active_reserved_bytes": active_reserved_bytes,
        "aggregate_process_tree_rss_bytes": aggregate_process_tree_rss_bytes,
        "free_disk_bytes": free_disk_bytes,
        "memory_pressure": memory_pressure,
        "thermal_state": thermal_state,
        "observed_at_epoch": float(observed_at_epoch),
    }
    swap_valid = False
    reviewed_decision: dict[str, Any] = {}
    sealed_swap_decision = False
    if isinstance(swap_decision, dict):
        try:
            sealed_swap_decision = swap_decision.get("receipt_sha256") \
                == _hash_value(_without(swap_decision, "receipt_sha256"))
        except (TypeError, ValueError):
            sealed_swap_decision = False
    if sealed_swap_decision \
            and swap_decision.get("controller_artifact") \
            == resources["swap_controller_artifact"] \
            and swap_decision.get("resource_sample_identity_sha256") \
            == _hash_value(resource_sample) \
            and swap_decision.get("now_epoch") == float(observed_at_epoch) \
            and -2 <= time.time() - float(observed_at_epoch) <= 30:
        try:
            next_state, reviewed_decision = aggressive_admission.advance_swap_state(
                swap_decision.get("previous_state"), swap_decision.get("snapshot"),
                now_epoch=float(observed_at_epoch),
                sealed_baseline_swap_mb=swap_decision.get("sealed_baseline_swap_mb"),
            )
            swap_valid = next_state == swap_decision.get("next_state") \
                and reviewed_decision == swap_decision.get("decision")
        except (aggressive_admission.PolicyError, TypeError, ValueError):
            swap_valid = False
    effective = max(active_reserved_bytes, aggregate_process_tree_rss_bytes)
    admitted = active_cache_units < 1 \
        and unit["estimated_resident_bytes"] <= resources["cache_unit_limit_bytes"] \
        and effective + unit["estimated_resident_bytes"] \
        <= resources["process_budget_bytes"] \
        and free_disk_bytes - unit["estimated_disk_bytes"] \
        >= resources["disk_reserve_bytes"] \
        and swap_valid \
        and reviewed_decision.get("allow_launch") is True \
        and reviewed_decision.get("mode") in {"green", "soft_throttle"} \
        and memory_pressure == "normal" and thermal_state in {"nominal", "fair"} \
        and scope in {SYNTHETIC_RESOURCE_SCOPE, PRODUCTION_RESOURCE_SCOPE} \
        and (scope != PRODUCTION_RESOURCE_SCOPE
             or TRUSTED_PRODUCTION_RESOURCE_OBSERVER_AVAILABLE)
    reason = "inside bounded cache envelope" if admitted else "cache resource gate refused"
    if scope == PRODUCTION_RESOURCE_SCOPE \
            and not TRUSTED_PRODUCTION_RESOURCE_OBSERVER_AVAILABLE:
        reason = "trusted production resource observer is unavailable"
    receipt: dict[str, Any] = {
        "schema": RESOURCE_RECEIPT_SCHEMA, "created_at": _now(),
        "status": "observation-complete", "scope": scope,
        "admitted": admitted, "cache_unit_id": cache_unit_id,
        "cache_plan_sha256": plan["cache_plan_sha256"],
        "consumer_manifest_sha256": manifest["manifest_sha256"],
        "estimated_resident_bytes": unit["estimated_resident_bytes"],
        "estimated_disk_bytes": unit["estimated_disk_bytes"],
        "disk_reserve_bytes": resources["disk_reserve_bytes"],
        "resource_sample": resource_sample,
        "resource_sample_identity_sha256": _hash_value(resource_sample),
        "swap_decision": swap_decision,
        "reviewed_swap_decision": reviewed_decision,
        "swap_decision_valid": swap_valid,
        "production_activation_permitted": False,
        "reason": reason,
    }
    receipt["receipt_sha256"] = _hash_value(receipt)
    return receipt


def _benchmark_bindings(*, input_authority: Any, cache_authority: Any,
                        output_authority: Any, scientific_authority: Any) -> dict[str, Any]:
    return {
        "input_identity_sha256": _hash_value(input_authority),
        "cache_identity_sha256": _hash_value(cache_authority),
        "output_identity_sha256": _hash_value(output_authority),
        "scientific_receipt_identity_sha256": _hash_value(scientific_authority),
        "coverage": {"scheduled": 1, "executed": 1, "validated": 1, "skipped": 0},
    }


def _serial_reference_valid(reference: Any, candidate: Any, *,
                            expected_input_identity_sha256: str | None = None) -> bool:
    """Prove equality to a separately executed, separately materialized oracle."""
    if not isinstance(reference, dict) or not _artifact_valid(candidate) \
            or not _artifact_valid(reference.get("artifact")) \
            or not _artifact_valid(reference.get("program_artifact"), instance=False) \
            or not _valid_sha(reference.get("invocation_sha256")) \
            or not _artifact_valid(reference.get("semantic_receipt")):
        return False
    serial = reference["artifact"]
    invocation = reference.get("invocation")
    if not isinstance(invocation, dict) or invocation != {
            "mode": "independent-serial-oracle",
            "program_sha256": reference["program_artifact"]["sha256"],
            "input_identity_sha256": invocation.get("input_identity_sha256")
            if isinstance(invocation, dict) else None,
    } or not _valid_sha(invocation.get("input_identity_sha256")) \
            or reference["invocation_sha256"] != _hash_value(invocation) \
            or expected_input_identity_sha256 is not None \
            and invocation["input_identity_sha256"] != expected_input_identity_sha256:
        return False
    return serial["sha256"] == candidate["sha256"] \
        and serial["bytes"] == candidate["bytes"] \
        and serial["artifact_instance_id"] != candidate["artifact_instance_id"] \
        and reference["semantic_receipt"]["artifact_instance_id"] \
        not in {serial["artifact_instance_id"], candidate["artifact_instance_id"]}


def _resource_receipt_errors(
    plan: dict[str, Any], manifest: dict[str, Any], unit: dict[str, Any], receipt: Any,
) -> list[str]:
    if not isinstance(receipt, dict) \
            or receipt.get("schema") != RESOURCE_RECEIPT_SCHEMA:
        return ["cache resource receipt schema mismatch"]
    errors: list[str] = []
    try:
        sealed = receipt.get("receipt_sha256") \
            == _hash_value(_without(receipt, "receipt_sha256"))
    except (TypeError, ValueError):
        sealed = False
    if not sealed:
        errors.append("cache resource receipt hash mismatch")
    if receipt.get("status") != "observation-complete" \
            or receipt.get("scope") != SYNTHETIC_RESOURCE_SCOPE \
            or receipt.get("production_activation_permitted") is not False:
        errors.append("cache resource receipt is not synthetic/fail-closed")
    if receipt.get("scope") == PRODUCTION_RESOURCE_SCOPE \
            or TRUSTED_PRODUCTION_RESOURCE_OBSERVER_AVAILABLE:
        errors.append("production cache resource observation is not implemented here")
    if receipt.get("cache_plan_sha256") != plan.get("cache_plan_sha256") \
            or receipt.get("consumer_manifest_sha256") \
            != manifest.get("manifest_sha256") \
            or receipt.get("cache_unit_id") != unit.get("cache_unit_id"):
        errors.append("cache resource receipt plan/unit binding differs")
    sample = receipt.get("resource_sample")
    sample_fields = {
        "active_cache_units", "active_reserved_bytes",
        "aggregate_process_tree_rss_bytes", "free_disk_bytes",
        "memory_pressure", "thermal_state", "observed_at_epoch",
    }
    if not isinstance(sample, dict) or set(sample) != sample_fields:
        errors.append("cache resource sample schema differs")
        return errors
    integer_fields = (
        "active_cache_units", "active_reserved_bytes",
        "aggregate_process_tree_rss_bytes", "free_disk_bytes",
    )
    if any(isinstance(sample.get(field), bool)
           or not isinstance(sample.get(field), int) or sample[field] < 0
           for field in integer_fields) \
            or isinstance(sample.get("observed_at_epoch"), bool) \
            or not isinstance(sample.get("observed_at_epoch"), (int, float)) \
            or not math.isfinite(float(sample["observed_at_epoch"])) \
            or not isinstance(sample.get("memory_pressure"), str) \
            or not isinstance(sample.get("thermal_state"), str):
        errors.append("cache resource sample values are invalid")
        return errors
    decision = receipt.get("swap_decision")
    swap_valid = False
    reviewed: dict[str, Any] = {}
    resources = plan["resources"]
    if isinstance(decision, dict):
        try:
            sealed_decision = decision.get("receipt_sha256") \
                == _hash_value(_without(decision, "receipt_sha256"))
        except (TypeError, ValueError):
            sealed_decision = False
        if sealed_decision \
                and decision.get("controller_artifact") \
                == resources["swap_controller_artifact"] \
                and decision.get("resource_sample_identity_sha256") \
                == _hash_value(sample) \
                and decision.get("now_epoch") == float(sample["observed_at_epoch"]):
            try:
                next_state, reviewed = aggressive_admission.advance_swap_state(
                    decision.get("previous_state"), decision.get("snapshot"),
                    now_epoch=float(sample["observed_at_epoch"]),
                    sealed_baseline_swap_mb=decision.get("sealed_baseline_swap_mb"),
                )
                swap_valid = next_state == decision.get("next_state") \
                    and reviewed == decision.get("decision")
            except (aggressive_admission.PolicyError, TypeError, ValueError):
                swap_valid = False
    effective = max(sample["active_reserved_bytes"],
                    sample["aggregate_process_tree_rss_bytes"])
    expected_admitted = sample["active_cache_units"] < 1 \
        and unit["estimated_resident_bytes"] <= resources["cache_unit_limit_bytes"] \
        and effective + unit["estimated_resident_bytes"] \
        <= resources["process_budget_bytes"] \
        and sample["free_disk_bytes"] - unit["estimated_disk_bytes"] \
        >= resources["disk_reserve_bytes"] \
        and swap_valid and reviewed.get("allow_launch") is True \
        and reviewed.get("mode") in {"green", "soft_throttle"} \
        and sample["memory_pressure"] == "normal" \
        and sample["thermal_state"] in {"nominal", "fair"}
    if receipt.get("resource_sample_identity_sha256") != _hash_value(sample) \
            or receipt.get("estimated_resident_bytes") \
            != unit["estimated_resident_bytes"] \
            or receipt.get("estimated_disk_bytes") != unit["estimated_disk_bytes"] \
            or receipt.get("disk_reserve_bytes") != resources["disk_reserve_bytes"] \
            or receipt.get("swap_decision_valid") is not swap_valid \
            or receipt.get("reviewed_swap_decision") != reviewed \
            or receipt.get("admitted") is not expected_admitted \
            or expected_admitted is not True \
            or receipt.get("reason") != "inside bounded cache envelope":
        errors.append("cache resource receipt admission decision differs")
    return errors


def build_cache_receipt(
    plan: dict[str, Any], manifest: dict[str, Any], *, cache_unit_id: str,
    component_artifacts: list[dict[str, Any]], cache_artifact: dict[str, Any],
    resource_receipt: dict[str, Any],
) -> dict[str, Any]:
    errors = validate_cache_plan(plan, manifest)
    unit = {row["cache_unit_id"]: row for row in plan.get("cache_units", [])}.get(
        cache_unit_id
    )
    if errors or unit is None:
        raise SharedCacheError("cannot receipt invalid cache plan/unit")
    source = {row["source_unit_id"]: row for row in manifest["source_units"]}[
        unit["source_unit_id"]
    ]
    input_authority = {
        "consumer_manifest_sha256": manifest["manifest_sha256"],
        "source_binding_sha256": source["source_binding_sha256"],
        "cache_unit_id": cache_unit_id,
    }
    component_fields = {"stage", "artifact", "serial_reference", "byte_exact"}
    if not isinstance(component_artifacts, list) \
            or len(component_artifacts) != len(unit["stages"]) \
            or any(not isinstance(row, dict) or set(row) != component_fields
                   for row in component_artifacts) \
            or [row.get("stage") for row in component_artifacts] != unit["stages"]:
        raise SharedCacheError("cache component stage coverage differs")
    prior_candidate_sha256s: list[str] = []
    for row in component_artifacts:
        stage_input_sha256 = _hash_value({
            "cache_input_authority": input_authority, "stage": row["stage"],
            "prior_candidate_sha256s": prior_candidate_sha256s,
        })
        if not _artifact_valid(row.get("artifact")) \
                or not _serial_reference_valid(row.get("serial_reference"),
                                               row.get("artifact"),
                                               expected_input_identity_sha256=
                                               stage_input_sha256) \
                or row.get("byte_exact") is not True:
            raise SharedCacheError("cache component lacks exact serial equivalence")
        prior_candidate_sha256s.append(row["artifact"]["sha256"])
    resource_errors = _resource_receipt_errors(
        plan, manifest, unit, resource_receipt
    )
    if not _artifact_valid(cache_artifact) or resource_errors:
        raise SharedCacheError("cache/resource artifact identity is invalid")
    parity_authority = [{"stage": row["stage"], "candidate": row["artifact"],
                         "serial_reference": row["serial_reference"]}
                        for row in component_artifacts]
    doc: dict[str, Any] = {
        "schema": CACHE_RECEIPT_SCHEMA, "created_at": _now(), "status": "complete",
        "cache_plan_sha256": plan["cache_plan_sha256"],
        "consumer_manifest_sha256": manifest["manifest_sha256"],
        "cache_unit_id": cache_unit_id, "source_unit_id": unit["source_unit_id"],
        "source_binding_sha256": unit["source_binding_sha256"],
        "component_artifacts": component_artifacts,
        "cache_artifact": cache_artifact,
        "component_manifest_sha256": _hash_value(parity_authority),
        "resource_receipt": resource_receipt,
        "resource_receipt_sha256": resource_receipt["receipt_sha256"],
        "resource_evidence_scope": SYNTHETIC_RESOURCE_SCOPE,
        "production_activation_permitted": False,
        "consumer_refcount_initial": unit["consumer_count"],
        "benchmark_bindings": _benchmark_bindings(
            input_authority=input_authority, cache_authority=cache_artifact,
            output_authority=cache_artifact, scientific_authority=parity_authority,
        ),
        "parent_sources_deleted": False, "cache_deleted": False,
        "quality_claims_permitted": False,
    }
    doc["receipt_sha256"] = _hash_value(doc)
    receipt_errors = validate_cache_receipt(plan, manifest, doc)
    if receipt_errors:
        raise SharedCacheError("generated cache receipt invalid: "
                               + "; ".join(receipt_errors))
    return doc


def validate_cache_receipt(
    plan: dict[str, Any], manifest: dict[str, Any], receipt: Any,
) -> list[str]:
    if not isinstance(receipt, dict) or receipt.get("schema") != CACHE_RECEIPT_SCHEMA:
        return ["cache receipt schema mismatch"]
    errors: list[str] = []
    if receipt.get("status") != "complete":
        errors.append("cache receipt is not complete")
    if receipt.get("receipt_sha256") != _hash_value(_without(receipt, "receipt_sha256")):
        errors.append("cache receipt hash mismatch")
    unit = {row["cache_unit_id"]: row for row in plan.get("cache_units", [])}.get(
        receipt.get("cache_unit_id")
    )
    if unit is None or receipt.get("cache_plan_sha256") != plan.get("cache_plan_sha256") \
            or receipt.get("consumer_manifest_sha256") != manifest.get("manifest_sha256") \
            or receipt.get("source_unit_id") != (unit or {}).get("source_unit_id") \
            or receipt.get("source_binding_sha256") != (unit or {}).get(
                "source_binding_sha256"):
        errors.append("cache receipt plan/source binding differs")
        return errors
    components = receipt.get("component_artifacts")
    component_fields = {"stage", "artifact", "serial_reference", "byte_exact"}
    if not isinstance(components, list) \
            or len(components) != len(unit["stages"]) \
            or any(not isinstance(row, dict) or set(row) != component_fields
                   for row in components) \
            or [row.get("stage") for row in components] != unit["stages"]:
        errors.append("cache receipt component coverage differs")
        components = []
    parity = []
    prior_candidate_sha256s: list[str] = []
    for row in components:
        stage_input_sha256 = _hash_value({
            "cache_input_authority": {
                "consumer_manifest_sha256": manifest["manifest_sha256"],
                "source_binding_sha256": unit["source_binding_sha256"],
                "cache_unit_id": unit["cache_unit_id"],
            }, "stage": row.get("stage"),
            "prior_candidate_sha256s": prior_candidate_sha256s,
        })
        if not _artifact_valid(row.get("artifact")) \
                or not _serial_reference_valid(row.get("serial_reference"),
                                               row.get("artifact"),
                                               expected_input_identity_sha256=
                                               stage_input_sha256) \
                or row.get("byte_exact") is not True:
            errors.append("cache receipt serial-equivalence component differs")
            continue
        parity.append({"stage": row["stage"], "candidate": row["artifact"],
                       "serial_reference": row["serial_reference"]})
        prior_candidate_sha256s.append(row["artifact"]["sha256"])
    if receipt.get("component_manifest_sha256") != _hash_value(parity) \
            or not _artifact_valid(receipt.get("cache_artifact")) \
            or _resource_receipt_errors(
                plan, manifest, unit, receipt.get("resource_receipt")) \
            or receipt.get("resource_receipt_sha256") \
            != (receipt.get("resource_receipt") or {}).get("receipt_sha256") \
            or receipt.get("resource_evidence_scope") != SYNTHETIC_RESOURCE_SCOPE \
            or receipt.get("production_activation_permitted") is not False \
            or receipt.get("consumer_refcount_initial") != unit["consumer_count"]:
        errors.append("cache receipt artifact/resource/refcount differs")
    source = {row["source_unit_id"]: row for row in manifest["source_units"]}[
        unit["source_unit_id"]
    ]
    expected_benchmark = _benchmark_bindings(
        input_authority={
            "consumer_manifest_sha256": manifest["manifest_sha256"],
            "source_binding_sha256": source["source_binding_sha256"],
            "cache_unit_id": unit["cache_unit_id"],
        }, cache_authority=receipt.get("cache_artifact"),
        output_authority=receipt.get("cache_artifact"), scientific_authority=parity,
    )
    if receipt.get("benchmark_bindings") != expected_benchmark:
        errors.append("cache receipt benchmark identities/no-skip coverage differ")
    if receipt.get("parent_sources_deleted") is not False \
            or receipt.get("cache_deleted") is not False \
            or receipt.get("quality_claims_permitted") is not False:
        errors.append("cache receipt lifecycle/claim boundary differs")
    return errors


def _consumer(plan: dict[str, Any], manifest: dict[str, Any], consumer_id: str) \
        -> dict[str, Any]:
    consumer = {row["consumer_id"]: row for row in manifest["consumers"]}.get(consumer_id)
    if consumer is None or plan.get("consumer_manifest_sha256") != manifest.get(
            "manifest_sha256"):
        raise SharedCacheError(f"consumer is absent from bound manifest: {consumer_id}")
    return consumer


def build_output_receipt(
    plan: dict[str, Any], manifest: dict[str, Any], *, consumer_id: str,
    cache_receipt_refs: list[dict[str, str]], output_artifact: dict[str, Any],
    scientific_receipt: dict[str, Any], serial_reference: dict[str, Any],
) -> dict[str, Any]:
    consumer = _consumer(plan, manifest, consumer_id)
    route = _consumer_route(manifest, consumer)
    ref_fields = {"cache_unit_id", "receipt_sha256"}
    if not isinstance(cache_receipt_refs, list) \
            or len(cache_receipt_refs) != len(route) \
            or any(not isinstance(row, dict) or set(row) != ref_fields
                   for row in cache_receipt_refs) \
            or [row.get("cache_unit_id") for row in cache_receipt_refs] != route \
            or any(not _valid_sha(row.get("receipt_sha256"))
                   for row in cache_receipt_refs):
        raise SharedCacheError("output cache receipt route differs")
    input_authority = {
        "consumer_id": consumer_id,
        "source_binding_sha256": {row["source_unit_id"]: row
                                  for row in manifest["source_units"]}[
                                      consumer["source_unit_id"]
                                  ]["source_binding_sha256"],
        "cache_receipt_refs": cache_receipt_refs,
    }
    if not _artifact_valid(output_artifact) or not _artifact_valid(scientific_receipt) \
            or not _serial_reference_valid(
                serial_reference, output_artifact,
                expected_input_identity_sha256=_hash_value(input_authority)):
        raise SharedCacheError("output/scientific/serial artifact identity differs")
    namespace = _hash_value({
        "manifest_sha256": manifest["manifest_sha256"], "consumer_id": consumer_id,
        "source_unit_id": consumer["source_unit_id"], "rate_id": consumer["rate_id"],
        "branch": consumer["branch"],
    })
    doc: dict[str, Any] = {
        "schema": OUTPUT_RECEIPT_SCHEMA, "created_at": _now(), "status": "complete",
        "cache_plan_sha256": plan["cache_plan_sha256"],
        "consumer_manifest_sha256": manifest["manifest_sha256"],
        "consumer_id": consumer_id, "source_unit_id": consumer["source_unit_id"],
        "rate_id": consumer["rate_id"], "branch": consumer["branch"],
        "cell_identity_sha256": consumer["cell_identity_sha256"],
        "evidence_namespace_sha256": namespace,
        "cache_receipt_refs": cache_receipt_refs,
        "output_artifact": output_artifact,
        "scientific_receipt": scientific_receipt,
        "serial_reference": serial_reference,
        "serial_equivalence": {"serial_output_sha256": output_artifact["sha256"],
                               "byte_exact": True, "zero_skips": True},
        "benchmark_bindings": _benchmark_bindings(
            input_authority=input_authority,
            cache_authority=cache_receipt_refs, output_authority=output_artifact,
            scientific_authority={"scientific_receipt": scientific_receipt,
                                  "serial_reference": serial_reference},
        ),
        "output_artifact_unique": True, "scientific_evidence_unique": True,
        "parent_sources_deleted": False,
        "claims": {"structural_complete": True, "quality": False,
                   "campaign_cell_complete": False},
    }
    doc["receipt_sha256"] = _hash_value(doc)
    errors = validate_output_receipt(plan, manifest, doc)
    if errors:
        raise SharedCacheError("generated output receipt invalid: " + "; ".join(errors))
    return doc


def validate_output_receipt(
    plan: dict[str, Any], manifest: dict[str, Any], receipt: Any,
) -> list[str]:
    if not isinstance(receipt, dict) or receipt.get("schema") != OUTPUT_RECEIPT_SCHEMA:
        return ["output receipt schema mismatch"]
    errors: list[str] = []
    if receipt.get("status") != "complete":
        errors.append("output receipt is not complete")
    if receipt.get("receipt_sha256") != _hash_value(_without(receipt, "receipt_sha256")):
        errors.append("output receipt hash mismatch")
    try:
        consumer = _consumer(plan, manifest, receipt.get("consumer_id"))
    except SharedCacheError as exc:
        return errors + [str(exc)]
    if receipt.get("cache_plan_sha256") != plan.get("cache_plan_sha256") \
            or receipt.get("consumer_manifest_sha256") != manifest.get("manifest_sha256") \
            or any(receipt.get(field) != consumer[field] for field in (
                "source_unit_id", "rate_id", "branch", "cell_identity_sha256")):
        errors.append("output receipt consumer binding differs")
    route = _consumer_route(manifest, consumer)
    refs = receipt.get("cache_receipt_refs")
    ref_fields = {"cache_unit_id", "receipt_sha256"}
    if not isinstance(refs, list) \
            or len(refs) != len(route) \
            or any(not isinstance(row, dict) or set(row) != ref_fields for row in refs) \
            or [row.get("cache_unit_id") for row in refs] != route \
            or any(not _valid_sha(row.get("receipt_sha256")) for row in refs):
        errors.append("output receipt cache route differs")
        refs = []
    namespace = _hash_value({
        "manifest_sha256": manifest["manifest_sha256"],
        "consumer_id": consumer["consumer_id"],
        "source_unit_id": consumer["source_unit_id"],
        "rate_id": consumer["rate_id"], "branch": consumer["branch"],
    })
    output, scientific = receipt.get("output_artifact"), receipt.get("scientific_receipt")
    reference = receipt.get("serial_reference")
    serial = receipt.get("serial_equivalence")
    source = {row["source_unit_id"]: row for row in manifest["source_units"]}[
        consumer["source_unit_id"]
    ]
    input_authority = {"consumer_id": consumer["consumer_id"],
                       "source_binding_sha256": source["source_binding_sha256"],
                       "cache_receipt_refs": refs}
    if receipt.get("evidence_namespace_sha256") != namespace \
            or not _artifact_valid(output) or not _artifact_valid(scientific) \
            or not _serial_reference_valid(
                reference, output,
                expected_input_identity_sha256=_hash_value(input_authority)) \
            or serial != {"serial_output_sha256": output.get("sha256")
                          if isinstance(output, dict) else None,
                          "byte_exact": True, "zero_skips": True}:
        errors.append("output/scientific/serial equivalence differs")
    expected_benchmark = _benchmark_bindings(
        input_authority=input_authority,
        cache_authority=refs, output_authority=output,
        scientific_authority={"scientific_receipt": scientific,
                              "serial_reference": reference},
    )
    if receipt.get("benchmark_bindings") != expected_benchmark:
        errors.append("output benchmark identities/no-skip coverage differ")
    if receipt.get("output_artifact_unique") is not True \
            or receipt.get("scientific_evidence_unique") is not True \
            or receipt.get("parent_sources_deleted") is not False \
            or receipt.get("claims") != {"structural_complete": True, "quality": False,
                                         "campaign_cell_complete": False}:
        errors.append("output receipt isolation/lifecycle/claims differ")
    return errors


def build_resume_state(plan: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    errors = validate_cache_plan(plan, manifest)
    if errors:
        raise SharedCacheError("cannot initialize invalid cache plan")
    doc: dict[str, Any] = {
        "schema": STATE_SCHEMA, "created_at": _now(), "updated_at": _now(),
        "cache_plan_sha256": plan["cache_plan_sha256"],
        "consumer_manifest_sha256": manifest["manifest_sha256"],
        "cache_units": {row["cache_unit_id"]: {"status": "pending"}
                        for row in plan["cache_units"]},
        "outputs": {row["consumer_id"]: {"status": "pending"}
                    for row in manifest["consumers"]},
        "parent_sources_deleted": False, "evidence_deleted": False,
    }
    doc["state_sha256"] = _hash_value(doc)
    return doc


def validate_resume_state(plan: dict[str, Any], manifest: dict[str, Any],
                          state: Any) -> list[str]:
    if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA:
        return ["resume state schema mismatch"]
    errors: list[str] = []
    if state.get("state_sha256") != _hash_value(_without(state, "state_sha256")):
        errors.append("resume state hash mismatch")
    if state.get("cache_plan_sha256") != plan.get("cache_plan_sha256") \
            or state.get("consumer_manifest_sha256") != manifest.get("manifest_sha256"):
        errors.append("resume state plan/manifest binding differs")
    cache_ids = {row["cache_unit_id"] for row in plan.get("cache_units", [])}
    consumer_ids = {row["consumer_id"] for row in manifest.get("consumers", [])}
    if not isinstance(state.get("cache_units"), dict) \
            or set(state["cache_units"]) != cache_ids \
            or not isinstance(state.get("outputs"), dict) \
            or set(state["outputs"]) != consumer_ids:
        errors.append("resume state exact cache/output key coverage differs")
    for table_name in ("cache_units", "outputs"):
        table = state.get(table_name, {})
        if not isinstance(table, dict):
            continue
        for row in table.values():
            if not isinstance(row, dict) or row.get("status") not in {"pending", "complete"} \
                    or row.get("status") == "pending" and set(row) != {"status"} \
                    or row.get("status") == "complete" and (
                        not _valid_sha(row.get("receipt_sha256"))
                        or row.get("adopted_after_restart") is not True):
                errors.append(f"resume state {table_name} row is invalid")
                break
    if state.get("parent_sources_deleted") is not False \
            or state.get("evidence_deleted") is not False:
        errors.append("resume state lifecycle differs")
    return errors


def reconcile_resume_state(
    plan: dict[str, Any], manifest: dict[str, Any], state: dict[str, Any], *,
    cache_receipts: Iterable[dict[str, Any]],
    output_receipts: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    state_errors = validate_resume_state(plan, manifest, state)
    if state_errors:
        raise SharedCacheError("resume state identity is invalid: "
                               + "; ".join(state_errors))
    state = copy.deepcopy(state)
    cache_by_id: dict[str, dict[str, Any]] = {}
    for receipt in cache_receipts:
        errors = validate_cache_receipt(plan, manifest, receipt)
        if errors or receipt["cache_unit_id"] in cache_by_id:
            raise SharedCacheError("crash adoption cache receipt is invalid/duplicate: "
                                   + "; ".join(errors))
        cache_by_id[receipt["cache_unit_id"]] = receipt
    output_by_id: dict[str, dict[str, Any]] = {}
    for receipt in output_receipts:
        errors = validate_output_receipt(plan, manifest, receipt)
        if errors or receipt["consumer_id"] in output_by_id:
            raise SharedCacheError("crash adoption output receipt is invalid/duplicate: "
                                   + "; ".join(errors))
        output_by_id[receipt["consumer_id"]] = receipt
    for cache_id, row in state["cache_units"].items():
        if row["status"] == "complete" and (
                cache_id not in cache_by_id
                or cache_by_id[cache_id]["receipt_sha256"]
                != row["receipt_sha256"]):
            raise SharedCacheError(
                "completed cache state must be reproven by the identical receipt")
    for consumer_id, row in state["outputs"].items():
        if row["status"] == "complete" and (
                consumer_id not in output_by_id
                or output_by_id[consumer_id]["receipt_sha256"]
                != row["receipt_sha256"]):
            raise SharedCacheError(
                "completed output state must be reproven by the identical receipt")
    for cache_id, receipt in cache_by_id.items():
        state["cache_units"][cache_id] = {
            "status": "complete", "receipt_sha256": receipt["receipt_sha256"],
            "adopted_after_restart": True,
        }
    for consumer_id, receipt in output_by_id.items():
        refs = receipt["cache_receipt_refs"]
        if any(cache_by_id.get(row["cache_unit_id"], {}).get("receipt_sha256")
               != row["receipt_sha256"] for row in refs):
            raise SharedCacheError("output adoption references absent/different cache receipts")
        state["outputs"][consumer_id] = {
            "status": "complete", "receipt_sha256": receipt["receipt_sha256"],
            "adopted_after_restart": True,
        }
    state["updated_at"] = _now()
    state["state_sha256"] = _hash_value(_without(state, "state_sha256"))
    final_errors = validate_resume_state(plan, manifest, state)
    if final_errors:
        raise SharedCacheError("reconciled resume state is invalid: "
                               + "; ".join(final_errors))
    return state


def build_merge_manifest(
    plan: dict[str, Any], manifest: dict[str, Any], *,
    cache_receipts: Iterable[dict[str, Any]],
    output_receipts: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    caches: dict[str, dict[str, Any]] = {}
    cache_instances: set[str] = set()
    cache_component_instances: set[str] = set()
    cache_serial_instances: set[str] = set()
    cache_serial_semantic_instances: set[str] = set()
    all_cache_evidence_instances: set[str] = set()
    for receipt in cache_receipts:
        errors = validate_cache_receipt(plan, manifest, receipt)
        if errors:
            raise SharedCacheError("invalid/duplicate cache receipt at merge: "
                                   + "; ".join(errors))
        cache_instance = receipt.get("cache_artifact", {}).get("artifact_instance_id")
        component_instances = [row.get("artifact", {}).get("artifact_instance_id")
                               for row in receipt.get("component_artifacts", [])]
        serial_instances = [row.get("serial_reference", {}).get("artifact", {}).get(
            "artifact_instance_id") for row in receipt.get("component_artifacts", [])]
        semantic_instances = [row.get("serial_reference", {}).get(
            "semantic_receipt", {}).get("artifact_instance_id")
            for row in receipt.get("component_artifacts", [])]
        combined = component_instances + serial_instances + semantic_instances
        receipt_instances = {cache_instance, *combined}
        if receipt["cache_unit_id"] in caches \
                or cache_instance in cache_instances \
                or cache_instance in set(combined) \
                or len(combined) != len(set(combined)) \
                or cache_instance in cache_component_instances \
                or cache_instance in cache_serial_instances \
                or cache_instance in cache_serial_semantic_instances \
                or len(receipt_instances) != 1 + len(combined) \
                or bool(receipt_instances & all_cache_evidence_instances) \
                or set(component_instances) & cache_component_instances \
                or set(serial_instances) & cache_serial_instances \
                or set(semantic_instances) & cache_serial_semantic_instances:
            raise SharedCacheError("invalid/duplicate cache receipt at merge: "
                                   + "; ".join(errors))
        caches[receipt["cache_unit_id"]] = receipt
        cache_instances.add(cache_instance)
        cache_component_instances.update(component_instances)
        cache_serial_instances.update(serial_instances)
        cache_serial_semantic_instances.update(semantic_instances)
        all_cache_evidence_instances.update(receipt_instances)
    outputs: dict[str, dict[str, Any]] = {}
    output_instances: set[str] = set()
    scientific_instances: set[str] = set()
    serial_output_instances: set[str] = set()
    serial_semantic_instances: set[str] = set()
    namespaces: set[str] = set()
    all_output_evidence_instances: set[str] = set()
    cache_owned_instances = (cache_instances | cache_component_instances
                             | cache_serial_instances
                             | cache_serial_semantic_instances)
    for receipt in output_receipts:
        errors = validate_output_receipt(plan, manifest, receipt)
        if errors:
            raise SharedCacheError("invalid/aliased output evidence at merge: "
                                   + "; ".join(errors))
        output_instance = receipt.get("output_artifact", {}).get("artifact_instance_id")
        scientific_instance = receipt.get("scientific_receipt", {}).get(
            "artifact_instance_id")
        serial_instance = receipt.get("serial_reference", {}).get("artifact", {}).get(
            "artifact_instance_id")
        serial_semantic_instance = receipt.get("serial_reference", {}).get(
            "semantic_receipt", {}).get("artifact_instance_id")
        namespace = receipt.get("evidence_namespace_sha256")
        receipt_instances = {output_instance, scientific_instance, serial_instance,
                             serial_semantic_instance}
        if receipt.get("consumer_id") in outputs \
                or output_instance in output_instances \
                or scientific_instance in scientific_instances \
                or serial_instance in serial_output_instances \
                or serial_semantic_instance in serial_semantic_instances \
                or len(receipt_instances) != 4 \
                or bool(receipt_instances & all_output_evidence_instances) \
                or bool(receipt_instances & cache_owned_instances) \
                or namespace in namespaces:
            raise SharedCacheError("invalid/aliased output evidence at merge: "
                                   + "; ".join(errors))
        outputs[receipt["consumer_id"]] = receipt
        output_instances.add(output_instance); scientific_instances.add(scientific_instance)
        serial_output_instances.add(serial_instance)
        serial_semantic_instances.add(serial_semantic_instance)
        all_output_evidence_instances.update(receipt_instances)
        namespaces.add(namespace)
    expected_outputs = {row["consumer_id"] for row in manifest["consumers"]}
    expected_caches = {row["cache_unit_id"] for row in plan["cache_units"]}
    if set(outputs) != expected_outputs or set(caches) != expected_caches:
        raise SharedCacheError("merge has missing/extra cache or output receipt coverage")
    refcounts = {cache_id: 0 for cache_id in caches}
    for consumer_id, receipt in outputs.items():
        for ref in receipt["cache_receipt_refs"]:
            cache = caches.get(ref["cache_unit_id"])
            if cache is None or cache["receipt_sha256"] != ref["receipt_sha256"]:
                raise SharedCacheError(f"output cache receipt reference differs: {consumer_id}")
            refcounts[ref["cache_unit_id"]] += 1
    expected_refcounts = {row["cache_unit_id"]: row["consumer_count"]
                          for row in plan["cache_units"]}
    if refcounts != expected_refcounts:
        raise SharedCacheError("cache consumer refcounts differ; ephemeral GC is forbidden")
    ordered = [{"consumer_id": consumer_id,
                "receipt_sha256": outputs[consumer_id]["receipt_sha256"],
                "output_artifact": outputs[consumer_id]["output_artifact"],
                "scientific_receipt": outputs[consumer_id]["scientific_receipt"]}
               for consumer_id in sorted(outputs)]
    doc: dict[str, Any] = {
        "schema": MERGE_SCHEMA, "created_at": _now(),
        "status": "structural-exact-coverage-quality-deferred",
        "cache_plan_sha256": plan["cache_plan_sha256"],
        "consumer_manifest_sha256": manifest["manifest_sha256"],
        "cache_receipt_count": len(caches), "output_receipt_count": len(outputs),
        "canonical_consumer_order": [row["consumer_id"] for row in ordered],
        "ordered_outputs_sha256": _hash_value(ordered), "ordered_outputs": ordered,
        "coverage": {"scheduled": len(expected_outputs), "executed": len(outputs),
                     "validated": len(outputs), "skipped": 0},
        "cache_refcounts": refcounts,
        "ephemeral_cache_gc_eligible": sorted(caches),
        "parent_sources_deleted": False, "scientific_claims_complete": False,
        "quality_claims_permitted": False,
    }
    doc["merge_sha256"] = _hash_value(doc)
    return doc


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    requirements = sub.add_parser("requirements")
    requirements.add_argument("--output", type=Path, default=DEFAULT_REQUIREMENTS)
    gptoss = sub.add_parser("build-gptoss-plan")
    gptoss.add_argument("--work-plan", type=Path, default=gptoss_parallel.DEFAULT_WORK_PLAN)
    gptoss.add_argument("--fanout-plan", type=Path, default=gptoss_fanout.DEFAULT_FANOUT_PLAN)
    gptoss.add_argument("--manifest-output", type=Path,
                        default=DEFAULT_GPTOSS_MANIFEST)
    gptoss.add_argument("--output", type=Path, default=DEFAULT_GPTOSS_PLAN)
    qwen = sub.add_parser("build-qwen-plan")
    qwen.add_argument("--inventory", type=Path, required=True)
    qwen.add_argument("--manifest-output", type=Path, default=DEFAULT_QWEN_MANIFEST)
    qwen.add_argument("--output", type=Path, default=DEFAULT_QWEN_PLAN)
    args = parser.parse_args(argv)
    if args.command == "requirements":
        doc = build_requirements_packet(); _write_json(args.output, doc)
        print(json.dumps({"status": "ok", "output": str(args.output.resolve()),
                          "requirements_sha256": doc["requirements_sha256"],
                          "execution_permitted": False}, indent=2, sort_keys=True))
        return 0
    if args.command == "build-qwen-plan":
        manifest = build_qwen_consumer_manifest(_read_json(args.inventory))
        plan = build_cache_plan(manifest)
    else:
        work, fanout = _read_json(args.work_plan), _read_json(args.fanout_plan)
        manifest = build_gptoss_consumer_manifest(work, fanout)
        plan = build_cache_plan(manifest)
    _write_json(args.manifest_output, manifest)
    _write_json(args.output, plan)
    persisted_manifest, persisted_plan = (_read_json(args.manifest_output),
                                           _read_json(args.output))
    if persisted_manifest != manifest or persisted_plan != plan \
            or validate_consumer_manifest(persisted_manifest) \
            or validate_cache_plan(persisted_plan, persisted_manifest):
        raise SharedCacheError("persisted manifest/plan reconstruction verification failed")
    print(json.dumps({"status": "ok", "output": str(args.output.resolve()),
                      "manifest_output": str(args.manifest_output.resolve()),
                      "manifest_sha256": manifest["manifest_sha256"],
                      "cache_plan_sha256": plan["cache_plan_sha256"],
                      "cache_units": len(plan["cache_units"]),
                      "consumers": plan["coverage"]["consumer_count"],
                      "derived_preprocess_consumers": (
                          plan["coverage"]["consumer_count"]
                          - plan["coverage"][
                              "consumers_without_qualified_derived_preprocess"]),
                      "execution_permitted": False}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except (SharedCacheError, OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        raise SystemExit(2)
