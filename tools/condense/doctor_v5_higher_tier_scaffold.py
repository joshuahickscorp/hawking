#!/usr/bin/env python3.12
"""Manifest-gated streaming and RAM admission contracts for models above 120B.

This is architecture-neutral scaffolding, not a model claim.  A model becomes
addressable only after an explicit immutable source manifest enumerates every
independent unit, byte range, logical-parameter contribution, staging peak, and
tokenizer authority.  Planning uses dynamic RAM and CPU caps, but execution
remains blocked until measured canary receipts replace estimates.
"""
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
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import doctor_v5_gptoss_mxfp4 as mxfp4
import doctor_v5_aggressive_admission_policy as aggressive_admission
import doctor_v5_higher_tier_authority as higher_authority


SOURCE_MANIFEST_SCHEMA = "hawking.doctor_v5_stream_source_manifest.v1"
ADMISSION_PLAN_SCHEMA = "hawking.doctor_v5_higher_tier_admission_plan.v1"
REQUIREMENTS_SCHEMA = "hawking.doctor_v5_higher_tier_requirements.v1"
DEFAULT_REQUIREMENTS = (
    ROOT / "reports/condense/doctor_v5_unbound/higher_tiers/requirements.json"
)
SHA_RE = re.compile(r"[0-9a-f]{64}")
MAX_JSON_BYTES = 128 * 1024 * 1024
CANONICAL_RATES = tuple(rate_id for rate_id, _rate in mxfp4.CANONICAL_RATES)


class HigherTierError(RuntimeError):
    """A higher-tier source or admission contract is incomplete or unsafe."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(doc: dict[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in doc.items() if key != field}


def _with_hash(doc: dict[str, Any], field: str) -> dict[str, Any]:
    doc[field] = _hash_value(doc)
    return doc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        info = path.stat()
        if path.is_symlink() or not stat.S_ISREG(info.st_mode) \
                or info.st_size > MAX_JSON_BYTES:
            raise HigherTierError(f"invalid JSON file: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HigherTierError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise HigherTierError(f"JSON root is not an object: {path}")
    return value


def _write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mxfp4._atomic_json(path, doc)


def _artifact(path: Path) -> dict[str, Any]:
    lexical = Path(os.path.abspath(os.fspath(path)))
    descriptor = os.open(
        lexical, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise HigherTierError(f"artifact is not a single-link regular file: {path}")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns,
            row.st_ctime_ns, row.st_nlink,
        )
        if identity(before) != identity(after) or size != after.st_size:
            raise HigherTierError(f"artifact changed while hashing: {path}")
        return {"path": str(lexical), "sha256": digest.hexdigest(), "bytes": size}
    finally:
        os.close(descriptor)


def _artifact_binding_errors(
    value: Any, *, label: str, verify_file: bool,
) -> list[str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"} \
            or not isinstance(value.get("path"), str) \
            or not Path(value.get("path", "")).is_absolute() \
            or not isinstance(value.get("sha256"), str) \
            or SHA_RE.fullmatch(value["sha256"]) is None \
            or isinstance(value.get("bytes"), bool) \
            or not isinstance(value.get("bytes"), int) or value["bytes"] <= 0:
        return [f"{label} artifact identity is invalid"]
    if not verify_file:
        return []
    try:
        observed = _artifact(Path(value["path"]))
    except (HigherTierError, OSError) as exc:
        return [f"{label} artifact cannot be verified: {exc}"]
    return [] if observed == value else [f"{label} artifact bytes differ from binding"]


def _bound_json_artifact(
    value: Any, *, label: str, verify_file: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors = _artifact_binding_errors(value, label=label, verify_file=False)
    if errors or not verify_file:
        return None, errors
    assert isinstance(value, dict)
    descriptor = -1
    try:
        descriptor = os.open(
            value["path"], os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
                or before.st_size <= 0 or before.st_size > MAX_JSON_BYTES:
            raise HigherTierError(f"{label} is not a bounded single-link JSON artifact")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity = lambda row: (
            row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns,
            row.st_ctime_ns, row.st_nlink,
        )
        raw = b"".join(chunks)
        if identity(before) != identity(after) or len(raw) != after.st_size \
                or hashlib.sha256(raw).hexdigest() != value["sha256"] \
                or len(raw) != value["bytes"]:
            raise HigherTierError(f"{label} changed or differs from its binding")
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise HigherTierError(f"{label} JSON root is not an object")
        return parsed, []
    except (OSError, UnicodeError, json.JSONDecodeError, HigherTierError) as exc:
        return None, [f"{label} cannot be verified: {exc}"]
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _self_hash_errors(value: Any, *, field: str, label: str) -> list[str]:
    if not isinstance(value, dict) or not isinstance(value.get(field), str) \
            or SHA_RE.fullmatch(value[field]) is None \
            or value[field] != _hash_value(_without(value, field)):
        return [f"{label} self-hash is invalid"]
    return []


def _exact_wiring_binding_errors(doc: dict[str, Any], *, verify_files: bool) -> list[str]:
    """Delegate concrete tensor/adapter/remote closure to operator authority."""
    _bindings, errors = higher_authority.inspect_core_manifest(
        doc, verify_files=verify_files,
    )
    return errors


def aggressive_swap_requirement() -> dict[str, Any]:
    """Exact future-controller contract; no baseline is invented at scaffold time."""
    controller = _artifact(Path(aggressive_admission.__file__))
    requirement: dict[str, Any] = {
        "controller_artifact": controller,
        "controller_overlay_schema": aggressive_admission.SCHEMA,
        "controller_state_schema": aggressive_admission.SWAP_STATE_SCHEMA,
        "controller_version": aggressive_admission.VERSION,
        "aggressive_overlay_artifact_required_for_execution": True,
        "sealed_baseline_swap_mb_required_at_quiescent_promotion": True,
        "baseline_can_ratchet": False,
        "relative_growth_and_rate_policy": aggressive_admission.swap_policy(),
        "unknown_probe_action": "hard_stop_new_launches_without_blind_termination",
        "controller_and_overlay_rehash_required_before_every_production_resume": True,
    }
    requirement["requirement_sha256"] = _hash_value(requirement)
    return requirement


def build_requirements_packet() -> dict[str, Any]:
    """Publish the generic gate without asserting that a particular model works."""
    doc: dict[str, Any] = {
        "schema": REQUIREMENTS_SCHEMA, "created_at": _now(),
        "status": "generic-scaffold-only-no-model-admitted",
        "scope": {"logical_parameters_strictly_greater_than": 120_000_000_000},
        "required_source_manifest": {
            "schema": SOURCE_MANIFEST_SCHEMA,
            "model_identity": [
                "label", "hf_id_or_source_id", "family", "architecture_kind",
                "logical_parameters", "parameter_authority_sha256",
            ],
            "source_identity": [
                "source_id", "transport", "bytes", "sha256",
                "immutable_content", "range_reads_supported",
            ],
            "work_unit_identity": [
                "unit_id", "kind", "source_ranges", "logical_parameters",
                "estimated_peak_resident_bytes", "threads_per_lane",
            ],
            "each_source_range_identity": [
                "source_id", "absolute_byte_range", "range_role",
                "range_sha256", "tensor_keys",
            ],
            "exact_coverage_required": True,
            "tokenizer_binding_required_for_quality": True,
            "exact_matrix_generation_additional_bindings": [
                "file-verified parameter authority",
                "file-verified reviewed architecture adapter",
                "file-verified tokenizer and chat-template binding",
                "file-verified lifecycle manifest",
                "file-verified transport manifest",
                "local payload and per-range hashes, or remote immutable-version and range receipts",
            ],
        },
        "supported_topologies_after_manifest_validation": [
            "local_sharded_dense", "local_sharded_moe",
            "remote_range_sharded_dense", "remote_range_sharded_moe",
        ],
        "parallel_axes": [
            "independent_source_units", "independent_experts",
            "ten_rate_outputs_after_one_source_traversal", "separate_parameter_tiers",
            "content-hash-identical_remote_workers",
        ],
        "admission": {
            "dynamic_memory_cap": True, "dynamic_cpu_cap": True,
            "measured_peak_canary_required": True,
            "normal_memory_pressure_required": True,
            "aggressive_swap_controller": aggressive_swap_requirement(),
            "thermal_stop_required": True, "oom_is_terminal_failure": True,
        },
        "distributed_transport": {
            "contract_schema": "hawking.doctor_v5_transport_plan.v1",
            "optional": True, "remote_host_claimed": False,
            "remote_execution_enabled": False,
            "signed_host_capability_and_expiring_lease_required": True,
            "content_addressed_resume_and_dedup_required": True,
            "host_result_and_coordinator_acceptance_required": True,
            "local_only_fallback": True, "runtime_defaults_changed": False,
        },
        "lifecycle": {
            "full_remote_source_install_required": False,
            "source_deletion_permitted": False,
            "hash_before_ephemeral_gc_required": True,
            "whole_artifact_bytes_not_amortized_across_rates": True,
        },
        "execution_permitted": False, "quality_claims_permitted": False,
        "unsupported_or_negative_outcomes_synthesized": False,
    }
    return _with_hash(doc, "requirements_sha256")


def validate_source_manifest(
    doc: Any, *, require_exact_wiring: bool = False, verify_files: bool = False,
    _operator_authorized_core: bool = False,
) -> list[str]:
    """Validate source graph; exact mode first requires operator SSHSIG seal."""
    if require_exact_wiring and not _operator_authorized_core:
        core, authority_errors = higher_authority.validate_and_unwrap(
            doc, verify_files=verify_files,
        )
        if authority_errors or core is None:
            return list(dict.fromkeys(
                f"higher-tier operator authority: {error}"
                for error in authority_errors
            ))
        doc = core
    if not isinstance(doc, dict) or doc.get("schema") != SOURCE_MANIFEST_SCHEMA:
        return ["source-manifest schema mismatch"]
    errors: list[str] = []
    if doc.get("manifest_sha256") != _hash_value(_without(doc, "manifest_sha256")):
        errors.append("source-manifest hash mismatch")
    if doc.get("status") != "sealed" or doc.get("source_deletion_permitted") is not False:
        errors.append("source manifest is not sealed/read-only")
    model = doc.get("model")
    if not isinstance(model, dict):
        errors.append("source-manifest model identity is missing")
        model = {}
    logical = model.get("logical_parameters")
    if isinstance(logical, bool) or not isinstance(logical, int) \
            or logical <= 120_000_000_000:
        errors.append("higher-tier exact logical parameter count must exceed 120B")
    if model.get("architecture_kind") not in {"dense", "moe"} \
            or any(not isinstance(model.get(field), str) or not model[field]
                   for field in ("label", "hf_id_or_source_id", "family")) \
            or not isinstance(model.get("parameter_authority_sha256"), str) \
            or SHA_RE.fullmatch(model["parameter_authority_sha256"]) is None:
        errors.append("model architecture/identity authority is invalid")
    if require_exact_wiring:
        errors.extend(_exact_wiring_binding_errors(doc, verify_files=verify_files))

    sources = doc.get("sources")
    if not isinstance(sources, list) or not sources:
        errors.append("source manifest has no immutable shards")
        sources = []
    source_by_id: dict[str, dict[str, Any]] = {}
    for row in sources:
        source_id = row.get("source_id") if isinstance(row, dict) else None
        transport = row.get("transport") if isinstance(row, dict) else None
        if not isinstance(source_id, str) or not source_id \
                or source_id in source_by_id or transport not in {
                    "local_file", "https_range", "object_range"
                } or not isinstance(row.get("sha256"), str) \
                or SHA_RE.fullmatch(row["sha256"]) is None \
                or isinstance(row.get("bytes"), bool) or not isinstance(row.get("bytes"), int) \
                or row["bytes"] <= 0 or row.get("immutable_content") is not True \
                or row.get("range_reads_supported") is not True:
            errors.append("source shard identity is invalid or duplicated")
            continue
        locator = row.get("path") if transport == "local_file" else row.get("uri")
        if not isinstance(locator, str) or not locator:
            errors.append(f"source shard locator is missing: {source_id}")
        if require_exact_wiring and transport == "local_file":
            if not isinstance(locator, str) or not Path(locator).is_absolute():
                errors.append(f"local source shard path is not absolute: {source_id}")
            elif verify_files:
                try:
                    observed = _artifact(Path(locator))
                except (HigherTierError, OSError) as exc:
                    errors.append(f"local source shard cannot be verified: {source_id}: {exc}")
                else:
                    expected_identity = {
                        "path": str(Path(os.path.abspath(locator))),
                        "sha256": row.get("sha256"), "bytes": row.get("bytes"),
                    }
                    if observed != expected_identity:
                        errors.append(f"local source shard bytes differ: {source_id}")
        elif require_exact_wiring and transport in {"https_range", "object_range"}:
            if not isinstance(row.get("immutable_version"), str) \
                    or not row["immutable_version"]:
                errors.append(f"remote source lacks immutable version: {source_id}")
            authority_doc, authority_errors = _bound_json_artifact(
                row.get("authority_artifact"),
                label=f"remote source {source_id} authority", verify_file=verify_files,
            )
            errors.extend(authority_errors)
            if verify_files and isinstance(authority_doc, dict):
                unstamped = _without(authority_doc, "authority_sha256")
                if set(authority_doc) != {
                    "schema", "source_id", "uri", "immutable_version", "bytes",
                    "sha256", "reviewed", "authority_sha256",
                } or authority_doc.get("schema") \
                        != "hawking.doctor_v5_remote_source_authority.v1" \
                        or authority_doc.get("source_id") != source_id \
                        or authority_doc.get("uri") != row.get("uri") \
                        or authority_doc.get("immutable_version") != row.get("immutable_version") \
                        or authority_doc.get("bytes") != row.get("bytes") \
                        or authority_doc.get("sha256") != row.get("sha256") \
                        or authority_doc.get("reviewed") is not True \
                        or authority_doc.get("authority_sha256") != _hash_value(unstamped):
                    errors.append(f"remote source authority semantics differ: {source_id}")
        source_by_id[source_id] = row

    units = doc.get("work_units")
    if not isinstance(units, list) or not units:
        errors.append("source manifest has no explicit work units")
        units = []
    unit_ids: set[str] = set()
    parameter_total = 0
    source_segments: dict[str, list[tuple[int, int, str, str]]] = {}
    source_range_rows: dict[str, list[dict[str, Any]]] = {}
    tensor_keys: set[str] = set()
    for unit in units:
        unit_id = unit.get("unit_id") if isinstance(unit, dict) else None
        kind = unit.get("kind") if isinstance(unit, dict) else None
        if not isinstance(unit_id, str) or not unit_id or unit_id in unit_ids \
                or kind not in {"dense_tensor_batch", "expert_batch", "embedding",
                                "output_head", "lossless_sidecar"}:
            errors.append("work unit identity is invalid or duplicated")
            continue
        unit_ids.add(unit_id)
        params = unit.get("logical_parameters")
        peak = unit.get("estimated_peak_resident_bytes")
        threads = unit.get("threads_per_lane")
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (params, peak, threads)) or params < 0 or peak <= 0 \
                or not 1 <= threads <= 128:
            errors.append(f"work-unit resource/parameter declaration invalid: {unit_id}")
        else:
            parameter_total += params
        ranges = unit.get("source_ranges")
        if not isinstance(ranges, list) or not ranges:
            errors.append(f"work unit has no source ranges: {unit_id}")
            continue
        for source_range in ranges:
            source_id = source_range.get("source_id") \
                if isinstance(source_range, dict) else None
            byte_range = source_range.get("absolute_byte_range") \
                if isinstance(source_range, dict) else None
            source = source_by_id.get(source_id)
            if source is None or not isinstance(byte_range, list) or len(byte_range) != 2 \
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           for value in byte_range) or byte_range[0] < 0 \
                    or byte_range[1] <= byte_range[0] \
                    or byte_range[1] > source["bytes"] \
                    or not isinstance(source_range.get("range_role"), str) \
                    or not isinstance(source_range.get("range_sha256"), str) \
                    or SHA_RE.fullmatch(source_range["range_sha256"]) is None \
                    or not isinstance(source_range.get("tensor_keys"), list) \
                    or not source_range["tensor_keys"] \
                    or any(not isinstance(key, str) or not key
                           for key in source_range["tensor_keys"]):
                errors.append(f"work-unit source range invalid: {unit_id}")
                continue
            for key in source_range["tensor_keys"]:
                if key in tensor_keys:
                    errors.append(f"tensor is assigned more than once: {key}")
                tensor_keys.add(key)
            if require_exact_wiring and source.get("transport") in {
                "https_range", "object_range",
            }:
                range_receipt, range_receipt_errors = _bound_json_artifact(
                    source_range.get("range_receipt_artifact"),
                    label=f"remote range {source_id}:{byte_range[0]}-{byte_range[1]}",
                    verify_file=verify_files,
                )
                errors.extend(range_receipt_errors)
                if verify_files and isinstance(range_receipt, dict):
                    unstamped = _without(range_receipt, "receipt_sha256")
                    if set(range_receipt) != {
                        "schema", "source_id", "uri", "immutable_version",
                        "absolute_byte_range", "range_sha256", "range_bytes",
                        "verified", "receipt_sha256",
                    } or range_receipt.get("schema") \
                            != "hawking.doctor_v5_remote_range_receipt.v1" \
                            or range_receipt.get("source_id") != source_id \
                            or range_receipt.get("uri") != source.get("uri") \
                            or range_receipt.get("immutable_version") \
                            != source.get("immutable_version") \
                            or range_receipt.get("absolute_byte_range") != byte_range \
                            or range_receipt.get("range_sha256") \
                            != source_range.get("range_sha256") \
                            or range_receipt.get("range_bytes") != byte_range[1] - byte_range[0] \
                            or range_receipt.get("verified") is not True \
                            or range_receipt.get("receipt_sha256") != _hash_value(unstamped):
                        errors.append(
                            f"remote range receipt semantics differ: "
                            f"{source_id}:{byte_range[0]}-{byte_range[1]}"
                        )
            source_segments.setdefault(source_id, []).append(
                (byte_range[0], byte_range[1], unit_id, source_range["range_sha256"])
            )
            source_range_rows.setdefault(source_id, []).append(source_range)
    if isinstance(logical, int) and parameter_total != logical:
        errors.append("work-unit logical parameters do not exactly close the model authority")
    for source_id, segments in source_segments.items():
        ordered = sorted(segments)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            errors.append(f"overlapping source work ranges: {source_id}")
        source_size = source_by_id[source_id]["bytes"]
        if not ordered or ordered[0][0] != 0 or ordered[-1][1] != source_size \
                or any(left[1] != right[0] for left, right in zip(ordered, ordered[1:])):
            errors.append(f"source work ranges do not fully and contiguously cover: {source_id}")
    if set(source_segments) != set(source_by_id):
        errors.append("one or more immutable sources has no explicit work range")
    coverage = doc.get("coverage")
    if not isinstance(coverage, dict) \
            or coverage.get("work_unit_count") != len(units) \
            or coverage.get("logical_parameters") != logical \
            or coverage.get("all_model_tensors_assigned_exactly_once") is not True \
            or coverage.get("tensor_count") != len(tensor_keys) \
            or not isinstance(coverage.get("tensor_layout_sha256"), str) \
            or SHA_RE.fullmatch(coverage["tensor_layout_sha256"]) is None:
        errors.append("source-manifest exact tensor coverage declaration is invalid")
    tokenizer = doc.get("tokenizer_binding")
    if not require_exact_wiring and tokenizer is not None:
        if isinstance(tokenizer, dict) and "binding_sha256" in tokenizer:
            tokenizer_errors = _self_hash_errors(
                tokenizer, field="binding_sha256", label="tokenizer binding",
            )
            tokenizer_errors.extend(_artifact_binding_errors(
                tokenizer.get("artifact"), label="tokenizer", verify_file=False,
            ))
            if tokenizer.get("tokenizer_sha256") \
                    != (tokenizer.get("artifact") or {}).get("sha256") \
                    or tokenizer.get("reviewed") is not True:
                tokenizer_errors.append("tokenizer binding identity is invalid")
            errors.extend(tokenizer_errors)
        elif not isinstance(tokenizer, dict) \
                or not isinstance(tokenizer.get("sha256"), str) \
                or SHA_RE.fullmatch(tokenizer["sha256"]) is None \
                or not isinstance(tokenizer.get("bytes"), int):
            errors.append("tokenizer binding identity is invalid")

    if require_exact_wiring and verify_files:
        for source_id, source in source_by_id.items():
            if source.get("transport") != "local_file":
                continue
            source_path = source.get("path")
            if not isinstance(source_path, str) or not Path(source_path).is_absolute():
                # The structural pass already recorded the locator error.
                continue
            try:
                descriptor = os.open(
                    source_path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
            except OSError as exc:
                errors.append(f"local source ranges cannot be opened: {source_id}: {exc}")
                continue
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 \
                        or before.st_size != source.get("bytes"):
                    errors.append(f"local source range descriptor is unsafe: {source_id}")
                    continue
                for start, end, _unit_id, expected_sha in sorted(
                    source_segments.get(source_id, [])
                ):
                    if os.lseek(descriptor, start, os.SEEK_SET) != start:
                        errors.append(f"local source range seek failed: {source_id}:{start}-{end}")
                        continue
                    digest = hashlib.sha256()
                    remaining = end - start
                    while remaining:
                        chunk = os.read(descriptor, min(1024 * 1024, remaining))
                        if not chunk:
                            errors.append(
                                f"local source range ended early: {source_id}:{start}-{end}"
                            )
                            break
                        digest.update(chunk)
                        remaining -= len(chunk)
                    if remaining == 0 and digest.hexdigest() != expected_sha:
                        errors.append(f"local source range hash differs: {source_id}:{start}-{end}")
                after = os.fstat(descriptor)
                identity = lambda row: (
                    row.st_dev, row.st_ino, row.st_size, row.st_mtime_ns,
                    row.st_ctime_ns, row.st_nlink,
                )
                if identity(before) != identity(after):
                    errors.append(f"local source changed during range verification: {source_id}")
            finally:
                os.close(descriptor)
    return errors


def unwrap_exact_source_manifest(
    value: Any, *, verify_files: bool = True,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Verify the pinned operator seal and fully validate its exact core."""
    core, errors = higher_authority.validate_and_unwrap(
        value, verify_files=verify_files,
    )
    if errors or core is None:
        return None, list(dict.fromkeys(errors))
    core_errors = validate_source_manifest(
        core, require_exact_wiring=True, verify_files=verify_files,
        _operator_authorized_core=True,
    )
    return (None, core_errors) if core_errors else (core, [])


def build_admission_plan(
    manifest: dict[str, Any], *, total_memory_bytes: int,
    process_budget_bytes: int, control_resident_bytes: int,
    safety_margin_bytes: int, logical_cpu_count: int,
    maximum_lanes: int = 64,
) -> dict[str, Any]:
    errors = validate_source_manifest(manifest)
    if errors:
        raise HigherTierError("source manifest is not admissible: " + "; ".join(errors))
    resource_values = (
        total_memory_bytes, process_budget_bytes, control_resident_bytes,
        safety_margin_bytes, logical_cpu_count, maximum_lanes,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in resource_values) \
            or process_budget_bytes > total_memory_bytes \
            or control_resident_bytes + safety_margin_bytes >= process_budget_bytes:
        raise HigherTierError("host resource envelope is invalid")
    units = manifest["work_units"]
    available = process_budget_bytes - control_resident_bytes - safety_margin_bytes
    largest_peak = max(row["estimated_peak_resident_bytes"] for row in units)
    largest_threads = max(row["threads_per_lane"] for row in units)
    memory_cap = available // largest_peak
    cpu_cap = logical_cpu_count // largest_threads
    proposed = min(maximum_lanes, memory_cap, cpu_cap)
    if proposed < 1:
        raise HigherTierError("no source unit fits the declared host envelope")

    # Deterministic first-fit decreasing waves.  This is a planning upper bound;
    # measured canaries still govern runtime admission.
    pending = sorted(
        units, key=lambda row: (-row["estimated_peak_resident_bytes"], row["unit_id"])
    )
    waves: list[dict[str, Any]] = []
    while pending:
        used_memory = used_threads = 0
        selected: list[dict[str, Any]] = []
        remainder: list[dict[str, Any]] = []
        for unit in pending:
            peak, threads = (unit["estimated_peak_resident_bytes"],
                             unit["threads_per_lane"])
            if len(selected) < maximum_lanes and used_memory + peak <= available \
                    and used_threads + threads <= logical_cpu_count:
                selected.append(unit)
                used_memory += peak
                used_threads += threads
            else:
                remainder.append(unit)
        if not selected:
            raise HigherTierError("deterministic wave planner made no progress")
        waves.append({
            "wave": len(waves), "unit_ids": [row["unit_id"] for row in selected],
            "estimated_peak_resident_bytes": used_memory,
            "declared_threads": used_threads,
        })
        pending = remainder
    source_manifest_hash = manifest["manifest_sha256"]
    output_units = [
        {
            "output_unit_id": f"{unit['unit_id']}/rate={rate_id}",
            "source_unit_id": unit["unit_id"], "rate_id": rate_id,
            "source_manifest_sha256": source_manifest_hash,
            "status": "pending_unmeasured",
        }
        for unit in sorted(units, key=lambda row: row["unit_id"])
        for rate_id in CANONICAL_RATES
    ]
    doc: dict[str, Any] = {
        "schema": ADMISSION_PLAN_SCHEMA, "created_at": _now(),
        "status": "unbound-estimate-only",
        "source_manifest_sha256": source_manifest_hash,
        "model": manifest["model"],
        "host_envelope": {
            "total_memory_bytes": total_memory_bytes,
            "process_budget_bytes": process_budget_bytes,
            "control_resident_bytes": control_resident_bytes,
            "safety_margin_bytes": safety_margin_bytes,
            "available_worker_bytes": available,
            "logical_cpu_count": logical_cpu_count,
            "maximum_lanes": maximum_lanes,
        },
        "estimated_caps": {
            "memory_lane_cap": memory_cap, "cpu_lane_cap": cpu_cap,
            "proposed_lane_cap": proposed,
            "largest_unit_estimated_peak_bytes": largest_peak,
            "largest_unit_threads": largest_threads,
        },
        "waves": waves, "output_units": output_units,
        "transport": {
            "contract_schema": "hawking.doctor_v5_transport_plan.v1",
            "default_mode": "local_only",
            "remote_host_claimed": False,
            "remote_execution_enabled": False,
            "eligible_host_capability_required": True,
            "signed_expiring_lease_required": True,
            "immutable_chunk_manifest_required": True,
            "signed_result_receipt_required": True,
            "local_only_fallback": True,
            "runtime_defaults_changed": False,
        },
        "execution_gate": {
            "executable": False,
            "measured_peak_receipts_complete": False,
            "normal_memory_pressure_observed": False,
            "aggressive_admission": {
                "controller_requirement": aggressive_swap_requirement(),
                "qualified_overlay_artifact_bound": False,
                "sealed_swap_baseline_bound": False,
                "non_ratcheting_controller_state_valid": False,
            },
            "thermal_guard_armed": False,
            "bit_exact_canary_passed": False,
            "source_range_receipts_verified": False,
            "tokenizer_quality_binding_passed": False,
        },
        "source_deletion_permitted": False, "quality_claims_permitted": False,
    }
    return _with_hash(doc, "admission_plan_sha256")


def validate_admission_plan(doc: Any, manifest: dict[str, Any]) -> list[str]:
    if not isinstance(doc, dict) or doc.get("schema") != ADMISSION_PLAN_SCHEMA:
        return ["admission-plan schema mismatch"]
    errors: list[str] = []
    if doc.get("admission_plan_sha256") != _hash_value(
            _without(doc, "admission_plan_sha256")):
        errors.append("admission-plan hash mismatch")
    if doc.get("source_manifest_sha256") != manifest.get("manifest_sha256"):
        errors.append("admission plan binds a different source manifest")
    if doc.get("status") != "unbound-estimate-only" \
            or doc.get("execution_gate", {}).get("executable") is not False \
            or doc.get("source_deletion_permitted") is not False \
            or doc.get("quality_claims_permitted") is not False:
        errors.append("estimated admission plan is not fail closed")
    aggressive = doc.get("execution_gate", {}).get("aggressive_admission")
    if not isinstance(aggressive, dict) \
            or aggressive.get("controller_requirement") != aggressive_swap_requirement() \
            or aggressive.get("qualified_overlay_artifact_bound") is not False \
            or aggressive.get("sealed_swap_baseline_bound") is not False \
            or aggressive.get("non_ratcheting_controller_state_valid") is not False:
        errors.append("higher-tier aggressive admission remains unbound or differs")
    transport = doc.get("transport")
    if not isinstance(transport, dict) or transport.get("default_mode") != "local_only" \
            or transport.get("remote_host_claimed") is not False \
            or transport.get("remote_execution_enabled") is not False \
            or transport.get("local_only_fallback") is not True \
            or transport.get("runtime_defaults_changed") is not False:
        errors.append("higher-tier transport default/fallback boundary is invalid")
    expected = {
        f"{unit['unit_id']}/rate={rate_id}"
        for unit in manifest.get("work_units", []) for rate_id in CANONICAL_RATES
    }
    observed = {row.get("output_unit_id") for row in doc.get("output_units", [])
                if isinstance(row, dict)}
    if observed != expected or len(doc.get("output_units", [])) != len(expected):
        errors.append("admission plan is not the exact unit x ten-rate matrix")
    waves = doc.get("waves")
    flattened = [unit_id for wave in waves for unit_id in wave.get("unit_ids", [])] \
        if isinstance(waves, list) else []
    expected_units = {row["unit_id"] for row in manifest.get("work_units", [])}
    if len(flattened) != len(set(flattened)) or set(flattened) != expected_units:
        errors.append("admission waves do not cover every unit exactly once")
    # The admission document is a deterministic function of the exact manifest
    # and its declared host inputs.  Rebuilding it closes forged cap, wave,
    # output-row, model, and execution-gate fields that a self-hash alone cannot
    # make authoritative.
    host = doc.get("host_envelope") if isinstance(doc.get("host_envelope"), dict) else {}
    try:
        expected_doc = build_admission_plan(
            manifest,
            total_memory_bytes=host["total_memory_bytes"],
            process_budget_bytes=host["process_budget_bytes"],
            control_resident_bytes=host["control_resident_bytes"],
            safety_margin_bytes=host["safety_margin_bytes"],
            logical_cpu_count=host["logical_cpu_count"],
            maximum_lanes=host["maximum_lanes"],
        )
        expected_doc["created_at"] = doc.get("created_at")
        expected_doc = _with_hash(
            _without(expected_doc, "admission_plan_sha256"),
            "admission_plan_sha256",
        )
    except (HigherTierError, KeyError, TypeError, ValueError):
        errors.append("admission plan host envelope cannot deterministically rebuild")
    else:
        if doc != expected_doc:
            errors.append("admission plan differs from deterministic manifest/host rebuild")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    requirements = sub.add_parser("requirements")
    requirements.add_argument("--output", type=Path, default=DEFAULT_REQUIREMENTS)
    validate = sub.add_parser("validate-source")
    validate.add_argument("--manifest", type=Path, required=True)
    build = sub.add_parser("build-admission")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--total-memory-bytes", type=int, required=True)
    build.add_argument("--process-budget-bytes", type=int, required=True)
    build.add_argument("--control-resident-bytes", type=int, required=True)
    build.add_argument("--safety-margin-bytes", type=int, required=True)
    build.add_argument("--logical-cpu-count", type=int, required=True)
    build.add_argument("--maximum-lanes", type=int, default=64)
    args = parser.parse_args(argv)
    try:
        if args.command == "requirements":
            doc = build_requirements_packet()
            _write_json(args.output, doc)
            print(json.dumps({"status": "ok", "output": str(args.output.resolve()),
                              "execution_permitted": False,
                              "requirements_sha256": doc["requirements_sha256"]},
                             indent=2, sort_keys=True))
            return 0
        manifest = _read_json(args.manifest)
        errors = validate_source_manifest(manifest)
        if args.command == "validate-source":
            print(json.dumps({"status": "ok" if not errors else "invalid",
                              "errors": errors}, indent=2, sort_keys=True))
            return 0 if not errors else 2
        if errors:
            raise HigherTierError("source manifest is invalid: " + "; ".join(errors))
        plan = build_admission_plan(
            manifest, total_memory_bytes=args.total_memory_bytes,
            process_budget_bytes=args.process_budget_bytes,
            control_resident_bytes=args.control_resident_bytes,
            safety_margin_bytes=args.safety_margin_bytes,
            logical_cpu_count=args.logical_cpu_count,
            maximum_lanes=args.maximum_lanes,
        )
        plan_errors = validate_admission_plan(plan, manifest)
        if plan_errors:
            raise HigherTierError("generated admission plan invalid: " + "; ".join(plan_errors))
        _write_json(args.output, plan)
        print(json.dumps({
            "status": "ok", "output": str(args.output.resolve()),
            "execution_permitted": False,
            "proposed_lane_cap": plan["estimated_caps"]["proposed_lane_cap"],
            "wave_count": len(plan["waves"]),
            "output_unit_count": len(plan["output_units"]),
            "admission_plan_sha256": plan["admission_plan_sha256"],
        }, indent=2, sort_keys=True))
        return 0
    except (HigherTierError, OSError, TypeError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
