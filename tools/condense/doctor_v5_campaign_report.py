#!/usr/bin/env python3.12
"""Atomic evidence aggregation for the Doctor-v5 Ultra campaign.

The campaign executor and this reporter deliberately have separate authority.
This module never launches, downloads, quantizes, trains, evaluates, or deletes
a model.  It binds an immutable campaign JSON file, records independently
durable cell checkpoints, and materializes transaction-like report snapshots:

* one consolidated report for every model below 120B; and
* one report per model at or above 120B.

Campaign input is intentionally generic.  A top-level ``cells`` or ``jobs``
array is accepted, but every entry must supply ``cell_id``, ``model_label``,
``rate``, and ``branch``.  Unknown entry fields are preserved in the normalized
manifest.  A campaign file is bound by its actual byte SHA-256; it cannot be
silently edited underneath existing checkpoints.

Commands::

    python doctor_v5_campaign_report.py init --campaign CAMPAIGN.json
    python doctor_v5_campaign_report.py record --campaign CAMPAIGN.json \
        --input CELL_CHECKPOINT.json
    python doctor_v5_campaign_report.py sync --campaign CAMPAIGN.json
    python doctor_v5_campaign_report.py aggregate --campaign CAMPAIGN.json
    python doctor_v5_campaign_report.py status --campaign CAMPAIGN.json
    python doctor_v5_campaign_report.py verify --campaign CAMPAIGN.json [--deep]
    python doctor_v5_campaign_report.py selftest

``record`` accepts a small JSON object.  Its required fields are ``cell_id``
and ``status``.  Useful optional fields are ``completed_stages``,
``completed_replicates``, ``evidence_paths``, ``evidence_refs``, ``metrics``,
``provenance``, ``timing``, ``eta``, ``resource_samples``, ``blockers``, and
``notes``.  Local evidence is hashed from a stable file descriptor.  Every
replacement checkpoint also receives an immutable revision copy.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import statistics
import sys
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = ROOT / "reports/condense/doctor_v5_ultra/campaign.json"
DEFAULT_REPORTING_ROOT = ROOT / "reports/condense/doctor_v5_ultra/reporting"

MANIFEST_SCHEMA = "hawking.doctor_v5_campaign_reporting_manifest.v1"
CHECKPOINT_SCHEMA = "hawking.doctor_v5_campaign_cell_checkpoint.v1"
COMPLETENESS_SCHEMA = "hawking.doctor_v5_campaign_completeness.v1"
ETA_SCHEMA = "hawking.doctor_v5_campaign_eta_ledger.v1"
REPORT_SCHEMA = "hawking.doctor_v5_campaign_evidence_report.v1"
INDEX_SCHEMA = "hawking.doctor_v5_campaign_report_index.v1"
RETENTION_SCHEMA = "hawking.doctor_v5_chain_retention_decision.v1"
REPORTER_VERSION = "doctor-v5-campaign-report.2"

DEFAULT_FRONTIER_THRESHOLD_B = 120.0
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_NUMERIC_MEASUREMENTS = 200_000
HEX64 = re.compile(r"[0-9a-f]{64}")
PARAMETER_LABEL = re.compile(r"(?<![0-9.])(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z])")
RATE_TOKEN = re.compile(
    r"^(?:q\s*)?(\d+(?:\.\d+)?)(?:\s*(?:bpw|bits?|b))?$", re.IGNORECASE
)

TERMINAL_COMPLETE = frozenset({"succeeded", "complete_negative", "unsupported"})
STATUSES = frozenset({
    "planned", "queued", "admitted", "running", "checkpointed",
    "succeeded", "complete_negative", "failed_retryable", "failed_terminal",
    "blocked", "invalidated", "skipped", "unsupported",
})
STATUS_ALIASES = {"complete": "succeeded", "negative": "complete_negative"}


class ReportingError(RuntimeError):
    """A fail-closed campaign reporting error."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _identity_payload(document: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    payload = copy.deepcopy(dict(document))
    payload.pop(hash_field, None)
    return payload


def _stamp(document: Mapping[str, Any], hash_field: str) -> dict[str, Any]:
    output = copy.deepcopy(dict(document))
    output[hash_field] = _sha_value(_identity_payload(output, hash_field))
    return output


def _validate_stamp(document: Any, hash_field: str) -> bool:
    return (
        isinstance(document, dict)
        and _is_sha(document.get(hash_field))
        and document[hash_field] == _sha_value(_identity_payload(document, hash_field))
    )


def _validate_json_tree(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReportingError(f"non-finite number at {path}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_tree(child, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ReportingError(f"non-string object key at {path}")
            _validate_json_tree(child, f"{path}.{key}")
        return
    raise ReportingError(f"non-JSON value {type(value).__name__} at {path}")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        st = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(st.st_mode):
            raise ReportingError(f"JSON input is not a regular non-symlink file: {path}")
        if st.st_size > MAX_JSON_BYTES:
            raise ReportingError(f"JSON input exceeds {MAX_JSON_BYTES} bytes: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
    except ReportingError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportingError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportingError(f"JSON root must be an object: {path}")
    _validate_json_tree(value)
    return value


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{os.urandom(4).hex()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, mode)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short atomic write")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    _fsync_dir(path.parent)


def _atomic_json(path: Path, value: Any) -> None:
    _validate_json_tree(value)
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False,
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, payload)


def _hash_stable_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ReportingError(f"cannot open evidence file {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ReportingError(f"evidence is not a regular file: {path}")
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        after = os.fstat(descriptor)
        if not (
            before.st_dev == after.st_dev
            and before.st_ino == after.st_ino
            and before.st_size == after.st_size == total
            and before.st_mtime_ns == after.st_mtime_ns
        ):
            raise ReportingError(f"evidence changed while hashing: {path}")
        return digest.hexdigest(), total
    finally:
        os.close(descriptor)


def _file_ref(path: Path, *, role: str | None = None) -> dict[str, Any]:
    digest, size = _hash_stable_file(path)
    reference: dict[str, Any] = {
        "path": str(path), "sha256": digest, "bytes": size,
    }
    if role is not None:
        reference["role"] = role
    return reference


@contextmanager
def _exclusive_lock(reporting_root: Path) -> Iterator[None]:
    reporting_root.mkdir(parents=True, exist_ok=True)
    path = reporting_root / "reporting.lock"
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0), 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _positive_float(value: Any, field: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ReportingError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportingError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        comparator = "nonnegative" if allow_zero else "positive"
        raise ReportingError(f"{field} must be finite and {comparator}")
    return number


def _positive_int(value: Any, field: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReportingError(f"{field} must be an integer")
    if value < 0 or (value == 0 and not allow_zero):
        comparator = "nonnegative" if allow_zero else "positive"
        raise ReportingError(f"{field} must be {comparator}")
    return value


def _rate(value: Any) -> float:
    if isinstance(value, str):
        match = RATE_TOKEN.fullmatch(value.strip())
        if match is None:
            raise ReportingError(f"rate is not a recognized bit/bpw value: {value!r}")
        value = match.group(1)
    return _positive_float(value, "rate")


def _params_b(entry: Mapping[str, Any]) -> tuple[float, str]:
    for field in (
        "params_b", "parameter_b", "parameter_billions", "parameter_tier_b",
        "model_params_b", "nominal_params_b",
    ):
        if field in entry and entry[field] is not None:
            return _positive_float(entry[field], field), f"declared:{field}"
    label = entry.get("model_label")
    if isinstance(label, str):
        matches = PARAMETER_LABEL.findall(label)
        if len(matches) == 1:
            return _positive_float(matches[0], "model_label parameter tier"), \
                "parsed_unique_B_token_from_model_label"
    raise ReportingError(
        f"cell {entry.get('cell_id')!r} needs params_b or one unique B token in model_label"
    )


def _model_tier_b(entry: Mapping[str, Any], params_b: float) -> tuple[float, str]:
    if entry.get("parameter_tier_b") is not None:
        return _positive_float(entry["parameter_tier_b"], "parameter_tier_b"), \
            "declared:parameter_tier_b"
    label = entry.get("model_label")
    if isinstance(label, str):
        matches = PARAMETER_LABEL.findall(label)
        if len(matches) == 1:
            return _positive_float(matches[0], "model_label parameter tier"), \
                "parsed_unique_B_token_from_model_label"
    return params_b, "fallback_to_params_b"


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportingError(f"{field} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ReportingError(f"{field} must be a list")
    result = [_string(item, f"{field}[]") for item in value]
    if len(result) != len(set(result)):
        raise ReportingError(f"{field} contains duplicates")
    return result


def _expected_replicates(entry: Mapping[str, Any]) -> int:
    if "expected_replicates" in entry:
        return _positive_int(entry["expected_replicates"], "expected_replicates")
    if "replicates" in entry and isinstance(entry["replicates"], int):
        return _positive_int(entry["replicates"], "replicates")
    seeds = entry.get("seeds", entry.get("seed_plan"))
    if isinstance(seeds, list) and seeds:
        return len(seeds)
    return 1


def _estimated_seconds(entry: Mapping[str, Any]) -> float | None:
    for field in ("estimated_seconds", "estimated_duration_s", "eta_seconds"):
        if entry.get(field) is not None:
            return _positive_float(entry[field], field, allow_zero=True)
    return None


def _campaign_arrays(document: Mapping[str, Any]) -> tuple[str, list[Any]]:
    cells = document.get("cells")
    jobs = document.get("jobs")
    if cells is not None and jobs is not None:
        raise ReportingError("campaign must use top-level cells or jobs, not both")
    key, rows = ("cells", cells) if cells is not None else ("jobs", jobs)
    if not isinstance(rows, list) or not rows:
        raise ReportingError("campaign needs a non-empty top-level cells or jobs array")
    return key, rows


def _reporting_policy(document: Mapping[str, Any]) -> tuple[float, int]:
    reporting = document.get("reporting")
    if reporting is None:
        reporting = {}
    if not isinstance(reporting, dict):
        raise ReportingError("campaign.reporting must be an object")
    threshold_raw = reporting.get(
        "frontier_threshold_b",
        document.get("frontier_threshold_b", DEFAULT_FRONTIER_THRESHOLD_B),
    )
    concurrency_raw = reporting.get(
        "max_parallel_cells", document.get("max_parallel_cells", 1)
    )
    threshold = _positive_float(threshold_raw, "frontier_threshold_b")
    concurrency = _positive_int(concurrency_raw, "max_parallel_cells")
    return threshold, concurrency


def normalize_campaign(campaign_path: Path) -> dict[str, Any]:
    """Compile a generic campaign into the reporter's immutable cell manifest."""
    campaign_path = campaign_path.expanduser().resolve(strict=True)
    document = _read_json_object(campaign_path)
    array_key, entries = _campaign_arrays(document)
    threshold, concurrency = _reporting_policy(document)
    campaign_sha, campaign_bytes = _hash_stable_file(campaign_path)

    normalized: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    model_tiers: dict[str, float] = {}
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise ReportingError(f"campaign.{array_key}[{index}] must be an object")
        for required in ("cell_id", "model_label", "branch"):
            if required not in raw:
                raise ReportingError(
                    f"campaign.{array_key}[{index}] is missing required {required}"
                )
        cell_id = _string(raw["cell_id"], "cell_id")
        model_label = _string(raw["model_label"], "model_label")
        branch = _string(raw["branch"], "branch")
        if cell_id in observed_ids:
            raise ReportingError(f"duplicate campaign cell_id: {cell_id}")
        observed_ids.add(cell_id)
        rate_raw = raw.get("rate", raw.get("rate_bpw"))
        if rate_raw is None:
            raise ReportingError(
                f"campaign.{array_key}[{index}] is missing required rate/rate_bpw"
            )
        if raw.get("rate") is not None and raw.get("rate_bpw") is not None \
                and not math.isclose(_rate(raw["rate"]), _rate(raw["rate_bpw"]),
                                     rel_tol=0.0, abs_tol=1e-12):
            raise ReportingError(f"cell {cell_id!r} has inconsistent rate and rate_bpw")
        rate = _rate(rate_raw)
        params_b, params_source = _params_b(raw)
        model_tier_b, model_tier_source = _model_tier_b(raw, params_b)
        prior_tier = model_tiers.setdefault(model_label, params_b)
        if not math.isclose(prior_tier, params_b, rel_tol=0.0, abs_tol=1e-12):
            raise ReportingError(f"model_label {model_label!r} has inconsistent parameter tiers")
        stages_value = raw.get("required_stages", raw.get("stage_path"))
        expected_stages = _string_list(stages_value, "required_stages")
        claim_track = raw.get("claim_track")
        if claim_track is not None:
            claim_track = _string(claim_track, "claim_track")
        group_id = "sub-120B" if model_tier_b < threshold else model_label
        core: dict[str, Any] = {
            "cell_id": cell_id,
            "cell_locator_sha256": hashlib.sha256(cell_id.encode("utf-8")).hexdigest(),
            "model_label": model_label,
            "params_b": params_b,
            "params_b_source": params_source,
            "model_tier_b": model_tier_b,
            "model_tier_b_source": model_tier_source,
            "rate_bpw": rate,
            "branch": branch,
            "claim_track": claim_track,
            "expected_stages": expected_stages,
            "expected_replicates": _expected_replicates(raw),
            "estimated_seconds": _estimated_seconds(raw),
            "report_group_id": group_id,
            "campaign_order": index,
        }
        cell: dict[str, Any] = {
            **core,
            "cell_spec_sha256": _sha_value(core),
            "campaign_entry": copy.deepcopy(raw),
            "campaign_entry_sha256": _sha_value(raw),
        }
        normalized.append(cell)

    groups: list[dict[str, Any]] = []
    sub = [cell for cell in normalized if cell["report_group_id"] == "sub-120B"]
    groups.append({
        "group_id": "sub-120B",
        "kind": "consolidated_sub_frontier_cohort",
        "frontier_threshold_b": threshold,
        "models": list(dict.fromkeys(cell["model_label"] for cell in sub)),
        "cell_ids": [cell["cell_id"] for cell in sub],
    })
    frontier_labels = list(dict.fromkeys(
        cell["model_label"] for cell in normalized if cell["model_tier_b"] >= threshold
    ))
    for model_label in frontier_labels:
        rows = [cell for cell in normalized if cell["model_label"] == model_label]
        groups.append({
            "group_id": model_label,
            "kind": "single_frontier_model",
            "frontier_threshold_b": threshold,
            "models": [model_label],
            "cell_ids": [cell["cell_id"] for cell in rows],
        })

    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "reporter_version": REPORTER_VERSION,
        "campaign": {
            "path": str(campaign_path),
            "sha256": campaign_sha,
            "bytes": campaign_bytes,
            "declared_schema": document.get("schema"),
            "declared_id": document.get("campaign_id", document.get("id")),
            "declared_version": document.get("version"),
            "plan_sha256": document.get("plan_sha256"),
            "cell_array_key": array_key,
        },
        "report_policy": {
            "frontier_threshold_b": threshold,
            "sub_frontier_predicate": "params_b < frontier_threshold_b",
            "frontier_predicate": "params_b >= frontier_threshold_b",
            "one_consolidated_sub_frontier_report": True,
            "one_report_per_frontier_model": True,
            "max_parallel_cells_for_eta_projection": concurrency,
            "missing_metrics_are_null_with_reason_never_zero": True,
        },
        "cells": normalized,
        "report_groups": groups,
        "counts": {
            "cells": len(normalized),
            "models": len(model_tiers),
            "sub_frontier_cells": len(sub),
            "frontier_cells": len(normalized) - len(sub),
            "report_groups": len(groups),
        },
    }
    return _stamp(manifest, "manifest_sha256")


def _manifest_path(reporting_root: Path) -> Path:
    return reporting_root / "manifest.json"


def _cell_spec_payload(cell: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in cell.items()
        if key not in {"cell_spec_sha256", "campaign_entry", "campaign_entry_sha256"}
    }


def _scientific_entry_binding(cell: Mapping[str, Any]) -> dict[str, Any]:
    entry = cell.get("campaign_entry")
    if not isinstance(entry, dict):
        return {}
    keys = (
        "cell_identity_sha256", "cell_spec_sha256", "model_name", "hf_id",
        "model_family", "exact_stored_parameter_count", "parameter_manifest",
        "source_census", "rate_id", "rate_bpw", "rate_fraction", "claim_scope",
        "adapter_id", "command", "backend", "dependencies", "seed_plan",
        "runtime_spec_schema",
    )
    return {key: copy.deepcopy(entry[key]) for key in keys if key in entry}


def _matrix_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "report_policy": copy.deepcopy(manifest.get("report_policy")),
        "plan_sha256": manifest.get("campaign", {}).get("plan_sha256")
            if isinstance(manifest.get("campaign"), dict) else None,
        "cells": [
            {"cell_spec": _cell_spec_payload(cell),
             "cell_spec_sha256": cell.get("cell_spec_sha256"),
             "declared_scientific_binding": _scientific_entry_binding(cell)}
            for cell in manifest.get("cells", []) if isinstance(cell, dict)
        ],
        "report_groups": copy.deepcopy(manifest.get("report_groups")),
    }


def _is_additive_cell_expansion(
    current: Mapping[str, Any], replacement: Mapping[str, Any],
) -> bool:
    """Require a strict cell-id superset; stale pre-run bindings may be regenerated."""
    old = {cell["cell_id"]: cell for cell in current.get("cells", [])
           if isinstance(cell, dict) and isinstance(cell.get("cell_id"), str)}
    new = {cell["cell_id"]: cell for cell in replacement.get("cells", [])
           if isinstance(cell, dict) and isinstance(cell.get("cell_id"), str)}
    return bool(old) and len(new) > len(old) and set(old) < set(new)


def _entry_has_execution_progress(cell: Mapping[str, Any]) -> bool:
    entry = cell.get("campaign_entry")
    if not isinstance(entry, dict):
        return True
    if entry.get("status") not in {None, "planned", "pending", "queued"}:
        return True
    attempts = entry.get("attempts", 0)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts != 0:
        return True
    if entry.get("started_at") is not None or entry.get("completed_at") is not None \
            or entry.get("result_sha256") is not None \
            or entry.get("disposition_sha256") is not None \
            or entry.get("execution_receipt_sha256") is not None \
            or entry.get("request_sha256") is not None:
        return True
    stages = entry.get("completed_stages")
    return stages not in (None, [])


def _reporting_root_has_progress(
    reporting_root: Path, current: Mapping[str, Any], replacement: Mapping[str, Any],
) -> bool:
    if any(_entry_has_execution_progress(cell) for cell in current.get("cells", [])) \
            or any(_entry_has_execution_progress(cell) for cell in replacement.get("cells", [])):
        return True
    if any((reporting_root / "checkpoints").glob("*.json")) \
            or any((reporting_root / "checkpoint_revisions").rglob("*.json")) \
            or any((reporting_root / "retention_decisions").glob("*.json")):
        return True
    index_path = reporting_root / "report_index.json"
    if not index_path.exists():
        return False
    try:
        index = _read_json_object(index_path)
    except ReportingError:
        return True
    if not _validate_stamp(index, "index_sha256"):
        return True
    summary = index.get("summary")
    return not isinstance(summary, dict) \
        or summary.get("complete_cells") not in (None, 0) \
        or bool(index.get("retention_decisions")) \
        or bool(index.get("ultra_report_checkpoints"))


def _validate_manifest(manifest: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        return [f"manifest schema must be {MANIFEST_SCHEMA}"]
    if not _validate_stamp(manifest, "manifest_sha256"):
        errors.append("manifest_sha256 is missing or mismatched")
    campaign = manifest.get("campaign")
    if not isinstance(campaign, dict) or not _is_sha(campaign.get("sha256")):
        errors.append("campaign byte binding is invalid")
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        errors.append("manifest cells are missing")
    else:
        ids: set[str] = set()
        for index, cell in enumerate(cells):
            if not isinstance(cell, dict) or not _is_sha(cell.get("cell_spec_sha256")) \
                    or cell.get("cell_spec_sha256") != _sha_value(_cell_spec_payload(cell)):
                errors.append(f"cells[{index}] spec hash is invalid")
                continue
            cell_id = cell.get("cell_id")
            if not isinstance(cell_id, str) or cell_id in ids:
                errors.append(f"cells[{index}] id is missing or duplicated")
            ids.add(cell_id)
    return errors


def initialize(campaign_path: Path, reporting_root: Path) -> dict[str, Any]:
    manifest = normalize_campaign(campaign_path)
    path = _manifest_path(reporting_root)
    with _exclusive_lock(reporting_root):
        if path.exists():
            current = _read_json_object(path)
            errors = _validate_manifest(current)
            if errors:
                raise ReportingError("existing reporting manifest is invalid: " + "; ".join(errors))
            if _matrix_binding(current) != _matrix_binding(manifest):
                if not _is_additive_cell_expansion(current, manifest) \
                        or _reporting_root_has_progress(reporting_root, current, manifest):
                    raise ReportingError(
                        "campaign scientific matrix changed without a provably zero-progress "
                        "additive expansion"
                    )
                # A pre-execution ladder expansion (for example adding a missed
                # parameter tier) is safe.  Preserve the old manifest as an
                # immutable revision and replace only the zero-progress binding.
                revision = reporting_root / "manifest_revisions" / (
                    f"{current['manifest_sha256']}.json"
                )
                if not revision.exists():
                    _atomic_json(revision, current)
                _atomic_json(path, manifest)
                manifest["_observed_campaign"] = copy.deepcopy(manifest["campaign"])
                manifest["_observed_cells"] = copy.deepcopy(manifest["cells"])
                return manifest
            current["_observed_campaign"] = copy.deepcopy(manifest["campaign"])
            current["_observed_cells"] = copy.deepcopy(manifest["cells"])
            return current
        _atomic_json(path, manifest)
    manifest["_observed_campaign"] = copy.deepcopy(manifest["campaign"])
    manifest["_observed_cells"] = copy.deepcopy(manifest["cells"])
    return manifest


def _bound_manifest(campaign_path: Path, reporting_root: Path) -> dict[str, Any]:
    path = _manifest_path(reporting_root)
    if not path.exists():
        return initialize(campaign_path, reporting_root)
    current = _read_json_object(path)
    errors = _validate_manifest(current)
    if errors:
        raise ReportingError("reporting manifest is invalid: " + "; ".join(errors))
    fresh = normalize_campaign(campaign_path)
    if _matrix_binding(fresh) != _matrix_binding(current):
        return initialize(campaign_path, reporting_root)
    current["_observed_campaign"] = copy.deepcopy(fresh["campaign"])
    current["_observed_cells"] = copy.deepcopy(fresh["cells"])
    return current


def _observed_campaign(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = manifest.get("_observed_campaign", manifest.get("campaign"))
    if not isinstance(value, dict):
        raise ReportingError("observed campaign binding is missing")
    return copy.deepcopy(value)


def _observed_cells(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = manifest.get("_observed_cells", manifest.get("cells"))
    if not isinstance(value, list):
        raise ReportingError("observed campaign cells are missing")
    return copy.deepcopy(value)


def _cell_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {cell["cell_id"]: cell for cell in manifest["cells"]}


def _cell_checkpoint_path(reporting_root: Path, cell: Mapping[str, Any]) -> Path:
    return reporting_root / "checkpoints" / f"{cell['cell_locator_sha256']}.json"


def _resolve_evidence_path(raw: str, campaign_path: Path) -> Path:
    candidate = Path(raw).expanduser()
    candidates = [candidate] if candidate.is_absolute() else [
        campaign_path.parent / candidate, ROOT / candidate,
    ]
    for path in candidates:
        try:
            return path.resolve(strict=True)
        except (OSError, RuntimeError):
            continue
    raise ReportingError(f"evidence path is missing: {raw!r}")


def _numeric_measurements(
    value: Any,
    *,
    source: str,
    prefix: str = "$",
    output: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if output is None:
        output = []
    if len(output) >= MAX_NUMERIC_MEASUREMENTS:
        raise ReportingError("numeric measurement count exceeds safety limit")
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return output
    if isinstance(value, int):
        output.append({"source": source, "path": prefix, "value": value})
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ReportingError(f"non-finite metric at {source}:{prefix}")
        output.append({"source": source, "path": prefix, "value": value})
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _numeric_measurements(
                child, source=source, prefix=f"{prefix}[{index}]", output=output
            )
    elif isinstance(value, dict):
        for key in sorted(value):
            _numeric_measurements(
                value[key], source=source, prefix=f"{prefix}.{key}", output=output
            )
    return output


def _json_evidence_document(path: Path) -> dict[str, Any] | None:
    try:
        st = path.stat()
        if st.st_size > MAX_JSON_BYTES or path.suffix.lower() not in {".json", ".jsonl"}:
            return None
        if path.suffix.lower() == ".jsonl":
            return None
        return _read_json_object(path)
    except (OSError, ReportingError):
        return None


def _nested_output_artifacts(
    document: Mapping[str, Any], *, parent_source: str, campaign_path: Path,
) -> tuple[list[dict[str, Any]], list[tuple[str, Mapping[str, Any]]], list[dict[str, Any]]]:
    """Retain output-artifact ledgers without re-hashing multi-GB payloads.

    The explicit parent evidence file has already been hashed by this reporter.
    Artifact declarations inside it therefore remain content-addressed evidence.
    Small local JSON artifacts are additionally re-hashed and parsed so task-level
    capability and resource observations enter the canonical metric ledger.  Large
    tensor artifacts are not re-read during every checkpoint update.
    """
    rows = document.get("output_artifacts")
    if not isinstance(rows, list):
        return [], [], []
    references: list[dict[str, Any]] = []
    documents: list[tuple[str, Mapping[str, Any]]] = []
    measurements: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        sha, size, raw_path = row.get("sha256"), row.get("bytes"), row.get("path")
        if not _is_sha(sha) or isinstance(size, bool) or not isinstance(size, int) \
                or size < 0 or not isinstance(raw_path, str) or not raw_path:
            continue
        role = str(row.get("role", f"nested_artifact_{index}"))
        reference: dict[str, Any] = {
            "role": role,
            "locator": raw_path,
            "sha256": sha,
            "bytes": size,
            "verification": "content_address_bound_by_hashed_parent_evidence",
            "declared_by": parent_source,
        }
        references.append(reference)
        candidate = Path(raw_path).expanduser()
        candidates = [candidate] if candidate.is_absolute() else [
            campaign_path.parent / candidate, ROOT / candidate,
        ]
        existing = next((path.resolve() for path in candidates if path.is_file()), None)
        if existing is None or existing.suffix.lower() != ".json" or size > MAX_JSON_BYTES:
            continue
        try:
            actual_sha, actual_size = _hash_stable_file(existing)
            if actual_sha != sha or actual_size != size:
                reference["local_verification"] = "mismatch"
                continue
            nested = _read_json_object(existing)
        except ReportingError:
            continue
        reference["local_verification"] = "verified"
        source = f"{parent_source}.output_artifacts[{role}]"
        documents.append((source, nested))
        _numeric_measurements(nested, source=source, output=measurements)
    return references, documents, measurements


def _path_value(document: Any, path: str) -> tuple[bool, Any]:
    current = document
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _observed(value: Any, source_path: str, *, derivation: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {"value": copy.deepcopy(value), "reason": None, "source_path": source_path}
    if derivation is not None:
        row["derivation"] = derivation
    return row


def _missing(reason: str = "not_reported") -> dict[str, Any]:
    return {"value": None, "reason": reason, "source_path": None}


def _find_metric(
    sources: Sequence[tuple[str, Mapping[str, Any]]], candidates: Sequence[str]
) -> dict[str, Any]:
    for source, document in sources:
        for candidate in candidates:
            found, value = _path_value(document, candidate)
            if found and value is not None:
                return _observed(value, f"{source}:{candidate}")
    return _missing()


def _find_role_metric(
    sources: Sequence[tuple[str, Mapping[str, Any]]],
    role_tokens: Sequence[str],
    candidates: Sequence[str],
) -> dict[str, Any]:
    selected = [row for row in sources if any(token in row[0].lower() for token in role_tokens)]
    result = _find_metric(selected, candidates)
    return result if result["value"] is not None else _find_metric(sources, candidates)


def _artifact_role_bytes(evidence: Sequence[Mapping[str, Any]], roles: Sequence[str]) -> dict[str, Any]:
    wanted = {role.lower() for role in roles}
    for row in evidence:
        if str(row.get("role", "")).lower() in wanted:
            return _observed(row.get("bytes"), f"evidence:{row.get('role')}.bytes")
    return _missing()


def _derive_difference(a: dict[str, Any], b: dict[str, Any], *, relative: bool = False) -> dict[str, Any]:
    left, right = a.get("value"), b.get("value")
    if isinstance(left, bool) or isinstance(right, bool):
        return _missing("inputs_not_numeric")
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return _missing("required_inputs_not_reported")
    if relative and left == 0:
        return _missing("baseline_is_zero")
    value = (right / left - 1.0) if relative else (right - left)
    return _observed(
        value,
        f"derived({a.get('source_path')},{b.get('source_path')})",
        derivation="candidate/baseline-1" if relative else "candidate-baseline",
    )


def _derive_bytes_difference(total: dict[str, Any], part: dict[str, Any], label: str) -> dict[str, Any]:
    a, b = total.get("value"), part.get("value")
    if not isinstance(a, int) or isinstance(a, bool) or not isinstance(b, int) or isinstance(b, bool):
        return _missing("required_inputs_not_reported")
    if a < b:
        return _missing("reported_total_is_smaller_than_component")
    return _observed(
        a - b, f"derived({total.get('source_path')},{part.get('source_path')})",
        derivation=label,
    )


def _derive_bpw(
    reported: dict[str, Any], numerator: dict[str, Any], denominator: dict[str, Any],
    numerator_name: str, denominator_name: str,
) -> dict[str, Any]:
    output = copy.deepcopy(reported)
    output["numerator_bytes_field"] = numerator_name
    output["denominator_parameters_field"] = denominator_name
    if output["value"] is None:
        n, d = numerator.get("value"), denominator.get("value")
        if isinstance(n, int) and not isinstance(n, bool) and isinstance(d, int) \
                and not isinstance(d, bool) and d > 0:
            output = _observed(
                n * 8.0 / d,
                f"derived({numerator.get('source_path')},{denominator.get('source_path')})",
                derivation="physical_bytes*8/exact_parameter_denominator",
            )
            output["numerator_bytes_field"] = numerator_name
            output["denominator_parameters_field"] = denominator_name
    return output


def _all_hashes(value: Any, source: str, prefix: str = "$") -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if isinstance(value, str) and _is_sha(value):
        found.append({"source": source, "path": prefix, "sha256": value})
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_all_hashes(child, source, f"{prefix}[{index}]"))
    elif isinstance(value, dict):
        for key in sorted(value):
            found.extend(_all_hashes(value[key], source, f"{prefix}.{key}"))
    return found


def _values_for_keys(value: Any, keys: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys and child is not None:
                found.append(child)
            found.extend(_values_for_keys(child, keys))
    elif isinstance(value, list):
        for child in value:
            found.extend(_values_for_keys(child, keys))
    return found


def _resource_summary(
    samples: Any, *, keys: set[str], mode: str, field: str,
) -> dict[str, Any]:
    values = _values_for_keys(samples, keys)
    if not values:
        return _missing()
    numeric = [float(value) for value in values
               if isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(float(value))]
    if mode == "max" and numeric:
        value: Any = max(numeric)
        if all(isinstance(item, int) for item in values if not isinstance(item, bool)):
            value = int(value)
    elif mode == "min" and numeric:
        value = min(numeric)
        if all(isinstance(item, int) for item in values if not isinstance(item, bool)):
            value = int(value)
    elif mode == "thermal":
        # False nominal flags and non-nominal strings must never be hidden by a
        # later good sample.  Preserve the complete observation vector as well.
        value = {
            "all_nominal": all(item is True or str(item).lower() in {"nominal", "normal"}
                               for item in values),
            "observations": copy.deepcopy(values),
        }
    else:
        value = copy.deepcopy(values[-1])
    return _observed(value, f"checkpoint.resource_samples:{field}",
                     derivation=f"{mode}_over_reported_resource_samples")


def _parse_time(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=dt.timezone.utc) if parsed.tzinfo is None else parsed


def _phase_wall_times(
    sources: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    totals = {"encode_wall_s": 0.0, "decode_wall_s": 0.0,
              "evaluation_wall_s": 0.0, "doctor_wall_s": 0.0}
    observed = {key: False for key in totals}
    provenance: dict[str, list[str]] = {key: [] for key in totals}
    for source, document in sources:
        phases = document.get("phases")
        if not isinstance(phases, dict):
            continue
        ordered: list[tuple[dt.datetime, str]] = []
        for name, phase in phases.items():
            if not isinstance(phase, dict):
                continue
            completed = _parse_time(phase.get("completed_at"))
            if completed is not None:
                ordered.append((completed, str(name)))
        ordered.sort()
        prior = _parse_time(document.get("created_at"))
        for completed, name in ordered:
            if prior is None or completed < prior:
                prior = completed
                continue
            seconds = (completed - prior).total_seconds()
            prior = completed
            lower = name.lower()
            field = None
            if "encode" in lower or "quantiz" in lower or "pack" in lower:
                field = "encode_wall_s"
            elif "decode" in lower or "decompress" in lower:
                field = "decode_wall_s"
            elif any(token in lower for token in ("eval", "ppl", "capability", "benchmark")):
                field = "evaluation_wall_s"
            elif any(token in lower for token in ("doctor", "repair", "treatment", "train")):
                field = "doctor_wall_s"
            if field is not None:
                totals[field] += seconds
                observed[field] = True
                provenance[field].append(f"{source}:phases.{name}.completed_at")
    output: dict[str, dict[str, Any]] = {}
    for field, value in totals.items():
        output[field] = (
            _observed(value, ",".join(provenance[field]),
                      derivation="sum_of_phase_completion_intervals")
            if observed[field] else _missing()
        )
    return output


def _canonical_metric_ledger(
    *,
    payload: Mapping[str, Any],
    cell: Mapping[str, Any],
    status: str,
    evidence: Sequence[Mapping[str, Any]],
    evidence_documents: Sequence[tuple[str, Mapping[str, Any]]],
    elapsed_s: float | None,
) -> dict[str, Any]:
    raw_metrics = payload.get("metrics", {})
    sources: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(raw_metrics, dict):
        sources.append(("checkpoint.metrics", raw_metrics))
    explicit_ledger = payload.get("metric_ledger")
    if isinstance(explicit_ledger, dict):
        sources.insert(0, ("checkpoint.metric_ledger", explicit_ledger))
    sources.extend(evidence_documents)

    stored = _find_metric(sources, (
        "parameters.exact_stored", "exact_stored_parameters",
        "parameter_accounting.stored_parameters",
        "parameter_accounting.stored_parameter_count",
        "metrics.parameter_accounting.stored_parameters",
        "metrics.parameter_accounting.stored_parameter_count",
    ))
    active = _find_metric(sources, (
        "parameters.active", "active_parameters", "active_parameter_count",
        "parameter_accounting.active_parameter_count",
    ))
    quantized = _find_metric(sources, (
        "parameters.quantized", "quantized_parameters",
        "parameter_accounting.quantized_parameter_count",
        "metrics.parameter_accounting.quantized_parameter_count",
    ))
    passthrough_params = _find_metric(sources, (
        "parameters.pass_through", "pass_through_parameters",
        "passthrough_parameters", "parameter_accounting.passthrough_parameter_count",
        "metrics.parameter_accounting.passthrough_parameter_count",
    ))

    source_bytes = _find_metric(sources, (
        "bytes.source", "source_bytes", "source.weight_bytes", "weight_bytes",
    ))
    packed_bytes = _find_metric(sources, (
        "bytes.packed", "packed_bytes", "physical_accounting.packed_bytes",
        "physical_accounting.packed_2d_tensor_bytes",
        "metrics.physical_accounting.packed_bytes",
        "metrics.physical_accounting.packed_2d_tensor_bytes",
    ))
    if packed_bytes["value"] is None:
        packed_bytes = _artifact_role_bytes(
            evidence, ("packed", "packed_projections", "packed_weights", "quantized_artifact")
        )
    passthrough_bytes = _find_metric(sources, (
        "bytes.pass_through", "pass_through_bytes", "passthrough_bytes",
        "physical_accounting.passthrough_bytes",
        "physical_accounting.lossless_non_2d_passthrough_bytes",
        "metrics.physical_accounting.lossless_non_2d_passthrough_bytes",
    ))
    if passthrough_bytes["value"] is None:
        passthrough_bytes = _artifact_role_bytes(
            evidence, ("pass_through", "passthrough", "lossless_passthrough")
        )
    doctor_bytes = _find_metric(sources, (
        "bytes.doctor", "doctor_bytes", "correction_bytes", "treatment_bytes",
        "physical_accounting.doctor_bytes",
    ))
    model_payload_bytes = _find_metric(sources, (
        "bytes.model_payload", "model_payload_bytes",
        "physical_accounting.model_payload_bytes",
        "metrics.physical_accounting.model_payload_bytes",
    ))
    full_bundle_bytes = _find_metric(sources, (
        "bytes.full_bundle", "full_bundle_bytes",
        "physical_accounting.full_bundle_bytes",
        "metrics.physical_accounting.full_bundle_bytes",
    ))
    metadata_bytes = _find_metric(sources, (
        "bytes.metadata", "metadata_bytes", "physical_accounting.metadata_bytes",
    ))
    if metadata_bytes["value"] is None:
        metadata_bytes = _derive_bytes_difference(
            full_bundle_bytes, model_payload_bytes, "full_bundle_bytes-model_payload_bytes"
        )

    packed_bpw = _find_metric(sources, (
        "physical_bpw.packed", "packed_bpw", "physical_accounting.packed_projection_bpw",
        "physical_accounting.packed_2d_tensor_bpw",
        "metrics.physical_accounting.packed_projection_bpw",
        "metrics.physical_accounting.packed_2d_tensor_bpw",
    ))
    payload_bpw = _find_metric(sources, (
        "physical_bpw.model_payload", "model_payload_bpw",
        "physical_accounting.all_in_model_payload_bpw",
        "physical_accounting.model_payload_bpw_over_all_stored_parameters",
        "metrics.physical_accounting.all_in_model_payload_bpw",
        "metrics.physical_accounting.model_payload_bpw_over_all_stored_parameters",
    ))
    bundle_bpw = _find_metric(sources, (
        "physical_bpw.full_bundle", "full_bundle_bpw",
        "physical_accounting.full_bundle_bpw_over_all_stored_parameters",
        "metrics.physical_accounting.full_bundle_bpw_over_all_stored_parameters",
    ))

    baseline_ppl = _find_role_metric(sources, ("baseline",), (
        "quality.baseline_ppl", "baseline_ppl", "ppl.baseline",
        "quality_observation.ppl.baseline", "metrics.quality_observation.ppl.baseline", "ppl",
    ))
    candidate_ppl = _find_role_metric(sources, ("candidate", "reconstruction", "doctor"), (
        "quality.candidate_ppl", "candidate_ppl", "ppl.reconstruction",
        "quality_observation.ppl.reconstruction",
        "metrics.quality_observation.ppl.reconstruction", "ppl",
    ))
    baseline_loss = _find_role_metric(sources, ("baseline",), (
        "quality.baseline_loss", "baseline_loss", "loss.baseline", "loss",
    ))
    candidate_loss = _find_role_metric(sources, ("candidate", "reconstruction", "doctor"), (
        "quality.candidate_loss", "candidate_loss", "loss.reconstruction", "loss",
    ))
    tokens = _find_metric(sources, (
        "quality.evaluated_tokens", "evaluated_tokens", "ntok", "tokens",
    ))
    base_cap = _find_role_metric(sources, ("baseline",), (
        "quality.baseline_capability_aggregate", "baseline_capability_aggregate",
        "capability.baseline", "quality_observation.capability.baseline",
        "metrics.quality_observation.capability.baseline", "aggregate",
    ))
    candidate_cap = _find_role_metric(sources, ("candidate", "reconstruction", "doctor"), (
        "quality.candidate_capability_aggregate", "candidate_capability_aggregate",
        "capability.reconstruction", "quality_observation.capability.reconstruction",
        "metrics.quality_observation.capability.reconstruction", "aggregate",
    ))
    base_tasks = _find_role_metric(sources, ("baseline",), (
        "quality.baseline_capability_per_task", "baseline_capability_per_task", "per_task",
    ))
    candidate_tasks = _find_role_metric(sources, ("candidate", "reconstruction", "doctor"), (
        "quality.candidate_capability_per_task", "candidate_capability_per_task", "per_task",
    ))
    per_task_delta = _missing("required_inputs_not_reported")
    if isinstance(base_tasks["value"], dict) and isinstance(candidate_tasks["value"], dict):
        keys = sorted(set(base_tasks["value"]) | set(candidate_tasks["value"]))
        delta: dict[str, Any] = {}
        for key in keys:
            left, right = base_tasks["value"].get(key), candidate_tasks["value"].get(key)
            delta[key] = right - left if isinstance(left, (int, float)) \
                and not isinstance(left, bool) and isinstance(right, (int, float)) \
                and not isinstance(right, bool) else None
        per_task_delta = _observed(
            delta, f"derived({base_tasks['source_path']},{candidate_tasks['source_path']})",
            derivation="candidate-baseline per task",
        )

    timing_candidates = {
        "encode_wall_s": ("timing.encode_wall_s", "encode_wall_s", "encode_seconds"),
        "decode_wall_s": ("timing.decode_wall_s", "decode_wall_s", "decode_seconds"),
        "evaluation_wall_s": (
            "timing.evaluation_wall_s", "evaluation_wall_s", "eval_seconds"
        ),
        "doctor_wall_s": ("timing.doctor_wall_s", "doctor_wall_s", "treatment_seconds"),
    }
    timings = {key: _find_metric(sources, paths) for key, paths in timing_candidates.items()}
    derived_phase_timings = _phase_wall_times(sources)
    for key in timing_candidates:
        if timings[key]["value"] is None:
            timings[key] = derived_phase_timings[key]
    timings["total_wall_s"] = (
        _observed(elapsed_s, "checkpoint.timing.elapsed_s")
        if elapsed_s is not None else _find_metric(sources, ("timing.total_wall_s", "total_wall_s"))
    )

    resource_fields = {
        "peak_rss_bytes": (
            "resources.peak_rss_bytes", "resources.before.peak_rss_bytes",
            "resources.after.peak_rss_bytes", "peak_rss_bytes",
        ),
        "memory_pressure_level": (
            "resources.memory_pressure_level", "resources.before.memory_pressure_level",
            "resources.after.memory_pressure_level",
            "phases.preflight.resources.memory_pressure_level", "memory_pressure_level"
        ),
        "swap_used_bytes": (
            "resources.swap_used_bytes", "resources.before.swap_used_bytes",
            "resources.after.swap_used_bytes", "phases.preflight.resources.swap_used_bytes",
            "swap_used_bytes",
        ),
        "thermal_state": (
            "resources.thermal_state", "thermal_state", "resources.thermal_nominal",
            "resources.before.thermal_nominal", "resources.after.thermal_nominal",
            "phases.preflight.resources.thermal_nominal", "thermal_nominal",
        ),
        "disk_free_bytes": (
            "resources.disk_free_bytes", "resources.before.disk_free_bytes",
            "resources.after.disk_free_bytes", "phases.preflight.resources.disk_free_bytes",
            "disk_free_bytes",
        ),
    }
    resources = {key: _find_metric(sources, paths) for key, paths in resource_fields.items()}
    samples = payload.get("resource_samples", [])
    sample_derivations = {
        "peak_rss_bytes": _resource_summary(
            samples, keys={"peak_rss_bytes", "rss_peak_bytes", "max_rss_bytes"},
            mode="max", field="peak_rss_bytes",
        ),
        "memory_pressure_level": _resource_summary(
            samples, keys={"memory_pressure_level", "pressure_level"},
            mode="max", field="memory_pressure_level",
        ),
        "swap_used_bytes": _resource_summary(
            samples, keys={"swap_used_bytes"}, mode="max", field="swap_used_bytes",
        ),
        "thermal_state": _resource_summary(
            samples, keys={"thermal_state", "thermal_nominal"},
            mode="thermal", field="thermal_state",
        ),
        "disk_free_bytes": _resource_summary(
            samples, keys={"disk_free_bytes"}, mode="min", field="disk_free_bytes",
        ),
    }
    for key, derived in sample_derivations.items():
        if resources[key]["value"] is None:
            resources[key] = derived
    resources["observations"] = copy.deepcopy(samples) if samples else None
    resources["observations_missing_reason"] = None if samples else "not_reported"

    seeds = {
        "training": _find_metric(sources, ("seeds.training", "training_seeds")),
        "calibration": _find_metric(sources, ("seeds.calibration", "calibration_seeds")),
        "evaluation": _find_metric(sources, ("seeds.evaluation", "evaluation_seeds")),
    }
    hashes: list[dict[str, str]] = []
    for source, document in sources:
        hashes.extend(_all_hashes(document, source))
    for row in evidence:
        if _is_sha(row.get("sha256")):
            hashes.append({
                "source": "evidence", "path": str(row.get("path", row.get("locator"))),
                "sha256": row["sha256"],
            })
    hashes = [dict(row) for row in {
        (row["source"], row["path"], row["sha256"]): row for row in hashes
    }.values()]
    hashes.sort(key=lambda row: (row["source"], row["path"], row["sha256"]))

    blockers = payload.get("blockers")
    blocker_entry = _missing()
    if blockers is not None:
        blocker_entry = _observed(_string_list(blockers, "blockers"), "checkpoint.blockers")
    negative = _observed(status == "complete_negative", "derived(checkpoint.status)",
                         derivation="status==complete_negative")
    eta = payload.get("eta") if isinstance(payload.get("eta"), dict) else {}

    return {
        "schema": "hawking.doctor_v5_metric_ledger.v1",
        "missing_value_policy": "null_with_reason_never_synthetic_zero",
        "parameters": {
            "exact_stored": stored,
            "active": active,
            "quantized": quantized,
            "pass_through": passthrough_params,
        },
        "bytes": {
            "source": source_bytes,
            "packed": packed_bytes,
            "pass_through": passthrough_bytes,
            "doctor": doctor_bytes,
            "metadata": metadata_bytes,
            "model_payload": model_payload_bytes,
            "full_bundle": full_bundle_bytes,
        },
        "physical_bpw": {
            "target_ceiling": _observed(cell["rate_bpw"], "campaign.cell.rate"),
            "packed": _derive_bpw(
                packed_bpw, packed_bytes, quantized, "bytes.packed", "parameters.quantized"
            ),
            "model_payload": _derive_bpw(
                payload_bpw, model_payload_bytes, stored,
                "bytes.model_payload", "parameters.exact_stored",
            ),
            "full_bundle": _derive_bpw(
                bundle_bpw, full_bundle_bytes, stored,
                "bytes.full_bundle", "parameters.exact_stored",
            ),
        },
        "timing": timings,
        "resources": resources,
        "quality": {
            "ppl": {
                "baseline": baseline_ppl,
                "candidate": candidate_ppl,
                "absolute_delta": _derive_difference(baseline_ppl, candidate_ppl),
                "relative_delta": _derive_difference(
                    baseline_ppl, candidate_ppl, relative=True
                ),
            },
            "loss": {
                "baseline": baseline_loss,
                "candidate": candidate_loss,
                "absolute_delta": _derive_difference(baseline_loss, candidate_loss),
            },
            "evaluated_tokens": tokens,
            "capability": {
                "baseline_aggregate": base_cap,
                "candidate_aggregate": candidate_cap,
                "aggregate_absolute_delta": _derive_difference(base_cap, candidate_cap),
                "baseline_per_task": base_tasks,
                "candidate_per_task": candidate_tasks,
                "per_task_delta": per_task_delta,
            },
        },
        "seeds": seeds,
        "provenance_hashes": hashes,
        "outcome": {
            "status": _observed(status, "checkpoint.status"),
            "blockers": blocker_entry,
            "negative_result": negative,
        },
        "eta": {
            "projected_total_s": (
                _observed(eta["projected_total_s"], "checkpoint.eta.projected_total_s")
                if eta.get("projected_total_s") is not None else _missing()
            ),
            "projected_remaining_s": (
                _observed(eta["remaining_s"], "checkpoint.eta.remaining_s")
                if eta.get("remaining_s") is not None else _missing()
            ),
            "measured_elapsed_s": (
                _observed(elapsed_s, "checkpoint.timing.elapsed_s")
                if elapsed_s is not None else _missing()
            ),
            "confidence": (
                _observed(eta["confidence"], "checkpoint.eta.confidence")
                if eta.get("confidence") is not None else _missing()
            ),
        },
        "raw_metrics": copy.deepcopy(raw_metrics),
    }


def _timing(payload: Mapping[str, Any]) -> tuple[dict[str, Any], float | None]:
    raw = payload.get("timing", {})
    if not isinstance(raw, dict):
        raise ReportingError("timing must be an object")
    timing: dict[str, Any] = {}
    for field in ("started_at", "updated_at", "completed_at"):
        if raw.get(field) is not None:
            timing[field] = _string(raw[field], f"timing.{field}")
    elapsed = None
    if raw.get("elapsed_s") is not None:
        elapsed = _positive_float(raw["elapsed_s"], "timing.elapsed_s", allow_zero=True)
        timing["elapsed_s"] = elapsed
    return timing, elapsed


def _eta_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("eta", {})
    if not isinstance(raw, dict):
        raise ReportingError("eta must be an object")
    eta: dict[str, Any] = {}
    for field in ("remaining_s", "projected_total_s"):
        if raw.get(field) is not None:
            eta[field] = _positive_float(raw[field], f"eta.{field}", allow_zero=True)
    if raw.get("confidence") is not None:
        eta["confidence"] = _string(raw["confidence"], "eta.confidence")
    return eta


def _pareto_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("pareto")
    if raw is None:
        return {
            "disposition": "undetermined",
            "decision_receipt_sha256": None,
            "dominated_by_cell_ids": [],
            "reason": "no_explicit_pareto_adjudication",
        }
    if not isinstance(raw, dict):
        raise ReportingError("pareto must be an object")
    disposition = _string(raw.get("disposition"), "pareto.disposition")
    if disposition not in {"retain", "discard_non_pareto", "undetermined"}:
        raise ReportingError("pareto.disposition is invalid")
    receipt = raw.get("decision_receipt_sha256")
    dominators = _string_list(raw.get("dominated_by_cell_ids"),
                              "pareto.dominated_by_cell_ids")
    reason = raw.get("reason")
    if reason is not None:
        reason = _string(reason, "pareto.reason")
    if disposition in {"retain", "discard_non_pareto"} and not _is_sha(receipt):
        raise ReportingError("an adjudicated Pareto disposition requires a receipt SHA-256")
    if disposition == "discard_non_pareto" and not dominators:
        raise ReportingError("discard_non_pareto requires at least one dominating cell")
    if disposition == "undetermined":
        receipt = None
        dominators = []
        reason = reason or "no_explicit_pareto_adjudication"
    return {
        "disposition": disposition,
        "decision_receipt_sha256": receipt,
        "dominated_by_cell_ids": dominators,
        "reason": reason,
    }


def _normalize_evidence(
    payload: Mapping[str, Any], campaign_path: Path,
) -> tuple[list[dict[str, Any]], list[tuple[str, Mapping[str, Any]]], list[dict[str, Any]]]:
    raw_paths = payload.get("evidence_paths", [])
    if not isinstance(raw_paths, list):
        raise ReportingError("evidence_paths must be a list")
    augmented: list[Any] = list(raw_paths)
    for field, role in (
        ("result_path", "result"), ("receipt_path", "receipt"),
        ("worker_checkpoint_path", "worker_checkpoint"),
    ):
        if payload.get(field) is not None:
            augmented.append({"path": payload[field], "role": role})

    evidence: list[dict[str, Any]] = []
    documents: list[tuple[str, Mapping[str, Any]]] = []
    measurements: list[dict[str, Any]] = []
    resolved_seen: set[Path] = set()
    for index, item in enumerate(augmented):
        role = f"evidence_{index}"
        if isinstance(item, str):
            raw_path = item
        elif isinstance(item, dict):
            raw_path = _string(item.get("path"), f"evidence_paths[{index}].path")
            if item.get("role") is not None:
                role = _string(item["role"], f"evidence_paths[{index}].role")
        else:
            raise ReportingError(f"evidence_paths[{index}] must be a path string or object")
        resolved = _resolve_evidence_path(raw_path, campaign_path)
        if resolved in resolved_seen:
            continue
        resolved_seen.add(resolved)
        reference = _file_ref(resolved, role=role)
        reference["declared_path"] = raw_path
        evidence.append(reference)
        document = _json_evidence_document(resolved)
        if document is not None:
            source = f"evidence.{role}"
            documents.append((source, document))
            _numeric_measurements(document, source=source, output=measurements)
            nested_refs, nested_docs, nested_measurements = _nested_output_artifacts(
                document, parent_source=source, campaign_path=campaign_path
            )
            evidence.extend(nested_refs)
            documents.extend(nested_docs)
            measurements.extend(nested_measurements)

    refs = payload.get("evidence_refs", [])
    if not isinstance(refs, list):
        raise ReportingError("evidence_refs must be a list")
    for index, raw in enumerate(refs):
        if not isinstance(raw, dict):
            raise ReportingError(f"evidence_refs[{index}] must be an object")
        sha = raw.get("sha256")
        size = raw.get("bytes")
        locator = raw.get("locator", raw.get("uri", raw.get("path")))
        if not _is_sha(sha):
            raise ReportingError(f"evidence_refs[{index}].sha256 is invalid")
        _positive_int(size, f"evidence_refs[{index}].bytes", allow_zero=True)
        reference = {
            "role": _string(raw.get("role", f"reference_{index}"), "evidence ref role"),
            "locator": _string(locator, "evidence ref locator"),
            "sha256": sha,
            "bytes": size,
            "verification": "caller_supplied_content_address",
        }
        evidence.append(reference)
    return evidence, documents, measurements


def _checkpoint_complete(
    cell: Mapping[str, Any], checkpoint: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    status = checkpoint.get("status")
    if status not in TERMINAL_COMPLETE:
        blockers.append("terminal_scientific_result_missing")
    expected_stages = cell["expected_stages"]
    completed_stages = checkpoint.get("progress", {}).get("completed_stages", [])
    # A reviewed unsupported disposition describes why execution could not
    # begin, so requiring fictional completed stages would corrupt the record.
    # A completed negative did execute its declared path, but is not a
    # scientific replicate and must not invent one merely to close reporting.
    if status != "unsupported" and expected_stages and completed_stages != expected_stages:
        blockers.append("required_stages_incomplete")
    if status == "succeeded" and checkpoint.get("progress", {}).get(
            "completed_replicates", 0) < cell["expected_replicates"]:
        blockers.append("required_replicates_incomplete")
    if not checkpoint.get("evidence"):
        blockers.append("hashed_evidence_missing")
    if status == "succeeded" and not checkpoint.get("numeric_measurements"):
        blockers.append("numeric_measurements_missing")
    return not blockers, blockers


def _validate_checkpoint(
    checkpoint: Any, *, cell: Mapping[str, Any], campaign_sha: str,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        return [f"checkpoint schema must be {CHECKPOINT_SCHEMA}"]
    if checkpoint.get("campaign_sha256") != campaign_sha:
        errors.append("checkpoint campaign binding mismatch")
    if checkpoint.get("cell_id") != cell["cell_id"] \
            or checkpoint.get("cell_spec_sha256") != cell["cell_spec_sha256"]:
        errors.append("checkpoint cell binding mismatch")
    if checkpoint.get("status") not in STATUSES:
        errors.append("checkpoint status is invalid")
    if not _validate_stamp(checkpoint, "checkpoint_sha256"):
        errors.append("checkpoint_sha256 is missing or mismatched")
    expected = cell["expected_stages"]
    completed = checkpoint.get("progress", {}).get("completed_stages")
    if not isinstance(completed, list) or completed != expected[:len(completed)]:
        errors.append("completed stages are not a strict expected-stage prefix")
    return errors


def record_checkpoint(
    campaign_path: Path, reporting_root: Path, payload: Mapping[str, Any],
    *, source_input: Path | None = None,
) -> dict[str, Any]:
    """Record one cell checkpoint and an immutable revision under an exclusive lock."""
    _validate_json_tree(payload)
    manifest = _bound_manifest(campaign_path, reporting_root)
    cell_id = _string(payload.get("cell_id"), "cell_id")
    cells = _cell_map(manifest)
    if cell_id not in cells:
        raise ReportingError(f"unknown campaign cell_id: {cell_id}")
    cell = cells[cell_id]
    raw_status = _string(payload.get("status"), "status")
    status = STATUS_ALIASES.get(raw_status, raw_status)
    if status not in STATUSES:
        raise ReportingError(f"unsupported checkpoint status: {status}")
    completed_stages = _string_list(payload.get("completed_stages"), "completed_stages")
    if completed_stages != cell["expected_stages"][:len(completed_stages)]:
        raise ReportingError("completed_stages must be a strict prefix of required stages")
    completed_replicates = _positive_int(
        payload.get("completed_replicates", 0), "completed_replicates", allow_zero=True
    )
    if completed_replicates > cell["expected_replicates"]:
        raise ReportingError("completed_replicates exceeds campaign expectation")
    attempt = _positive_int(payload.get("attempt", 1), "attempt")
    metrics = payload.get("metrics", {})
    provenance = payload.get("provenance", {})
    resource_samples = payload.get("resource_samples", [])
    if not isinstance(metrics, dict) or not isinstance(provenance, dict):
        raise ReportingError("metrics and provenance must be objects")
    if not isinstance(resource_samples, list):
        raise ReportingError("resource_samples must be a list")
    timing, elapsed = _timing(payload)
    eta = _eta_input(payload)
    evidence, evidence_documents, measurements = _normalize_evidence(payload, campaign_path)
    _numeric_measurements(metrics, source="checkpoint.metrics", output=measurements)
    _numeric_measurements(resource_samples, source="checkpoint.resource_samples", output=measurements)
    notes = _string_list(payload.get("notes"), "notes")
    blockers = payload.get("blockers")
    if blockers is not None:
        blockers = _string_list(blockers, "blockers")

    source_ref = _file_ref(source_input, role="checkpoint_input") if source_input else None
    reporter_ref = _file_ref(Path(__file__).resolve(), role="reporter_source")
    path = _cell_checkpoint_path(reporting_root, cell)
    with _exclusive_lock(reporting_root):
        previous: dict[str, Any] | None = None
        if path.exists():
            previous = _read_json_object(path)
            errors = _validate_checkpoint(
                previous, cell=cell, campaign_sha=manifest["campaign"]["sha256"]
            )
            if errors:
                raise ReportingError("existing checkpoint invalid: " + "; ".join(errors))
            previous_attempt = previous["attempt"]
            if attempt < previous_attempt:
                raise ReportingError("checkpoint attempt cannot decrease")
            if attempt == previous_attempt:
                old_stages = previous["progress"]["completed_stages"]
                old_replicates = previous["progress"]["completed_replicates"]
                if completed_stages[:len(old_stages)] != old_stages:
                    raise ReportingError("checkpoint stage progress cannot shrink or change")
                if completed_replicates < old_replicates:
                    raise ReportingError("checkpoint replicate progress cannot shrink")
                if previous["status"] in TERMINAL_COMPLETE \
                        and status not in {previous["status"], "invalidated"}:
                    raise ReportingError("a completed result may only remain complete or be invalidated")
            elif previous["status"] in TERMINAL_COMPLETE:
                raise ReportingError("completed cells require a new campaign cell for a new treatment")

        revision = 1 if previous is None else int(previous["revision"]) + 1
        recorded_at = _now()
        checkpoint: dict[str, Any] = {
            "schema": CHECKPOINT_SCHEMA,
            "reporter_version": REPORTER_VERSION,
            "campaign_sha256": manifest["campaign"]["sha256"],
            "campaign_observation": _observed_campaign(manifest),
            "manifest_sha256": manifest["manifest_sha256"],
            "cell_id": cell_id,
            "cell_spec_sha256": cell["cell_spec_sha256"],
            "cell_identity": {
                key: copy.deepcopy(cell[key]) for key in (
                    "model_label", "params_b", "rate_bpw", "branch", "claim_track",
                    "report_group_id",
                )
            },
            "attempt": attempt,
            "revision": revision,
            "previous_checkpoint_sha256": (
                previous.get("checkpoint_sha256") if previous is not None else None
            ),
            "recorded_at": recorded_at,
            "status": status,
            "progress": {
                "expected_stages": copy.deepcopy(cell["expected_stages"]),
                "completed_stages": completed_stages,
                "expected_replicates": cell["expected_replicates"],
                "completed_replicates": completed_replicates,
            },
            "evidence": evidence,
            "numeric_measurements": measurements,
            "metric_ledger": {},
            "metrics": copy.deepcopy(metrics),
            "provenance": {
                "campaign_file": _observed_campaign(manifest),
                "reporter_source": reporter_ref,
                "checkpoint_input": source_ref,
                "declared": copy.deepcopy(provenance),
            },
            "resource_samples": copy.deepcopy(resource_samples),
            "timing": timing,
            "eta": eta,
            "pareto": _pareto_input(payload),
            "blockers": copy.deepcopy(blockers),
            "notes": notes,
            "claims": copy.deepcopy(payload.get("claims", {})),
        }
        checkpoint["metric_ledger"] = _canonical_metric_ledger(
            payload=payload, cell=cell, status=status, evidence=evidence,
            evidence_documents=evidence_documents, elapsed_s=elapsed,
        )
        complete, completion_blockers = _checkpoint_complete(cell, checkpoint)
        checkpoint["completeness"] = {
            "complete": complete, "blockers": completion_blockers,
        }
        checkpoint = _stamp(checkpoint, "checkpoint_sha256")
        errors = _validate_checkpoint(
            checkpoint, cell=cell, campaign_sha=manifest["campaign"]["sha256"]
        )
        if errors:
            raise ReportingError("generated checkpoint invalid: " + "; ".join(errors))

        revision_path = (
            reporting_root / "checkpoint_revisions" / cell["cell_locator_sha256"]
            / f"r{revision:08d}-{checkpoint['checkpoint_sha256']}.json"
        )
        if not revision_path.exists():
            _atomic_json(revision_path, checkpoint)
        _atomic_json(path, checkpoint)
    return checkpoint


def _existing_path(raw: Any, campaign_path: Path) -> Path | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return _resolve_evidence_path(raw, campaign_path)
    except ReportingError:
        return None


def _completed_replicates_from_result(result: Mapping[str, Any], expected: int) -> int:
    candidates = (
        "completed_replicates", "metrics.completed_replicates",
        "seed_count", "metrics.seed_count",
    )
    for path in candidates:
        found, value = _path_value(result, path)
        if found and isinstance(value, int) and not isinstance(value, bool) \
                and 0 <= value <= expected:
            return value
    for path in ("completed_seeds", "training_seeds", "seed_results", "metrics.seed_results"):
        found, value = _path_value(result, path)
        if found and isinstance(value, list) and len(value) <= expected:
            return len(value)
    return 0


def _elapsed_between(start: Any, finish: Any) -> float | None:
    left, right = _parse_time(start), _parse_time(finish)
    if left is None or right is None or right < left:
        return None
    return (right - left).total_seconds()


def sync_campaign(campaign_path: Path, reporting_root: Path) -> dict[str, Any]:
    """Import observable live-campaign progress into reporter checkpoints.

    Missing result files remain missing; status alone never fabricates a metric,
    replicate, negative result, or completed scientific cell.
    """
    manifest = _bound_manifest(campaign_path, reporting_root)
    canonical_status = {
        "running": "running",
        "complete": "succeeded",
        "negative": "complete_negative",
        "unsupported": "unsupported",
        "blocked-dependency": "blocked",
        "blocked-execution": "blocked",
    }
    imported = 0
    skipped = 0
    errors: dict[str, str] = {}
    for observed in _observed_cells(manifest):
        entry = observed["campaign_entry"]
        state = entry.get("status")
        if state not in canonical_status:
            skipped += 1
            continue
        cell_id = observed["cell_id"]
        result_paths = entry.get("result_paths") if isinstance(entry.get("result_paths"), dict) else {}
        evidence_paths: list[dict[str, str]] = []
        role_fields = (
            ("result", "result"), ("execution_receipt", "execution_receipt"),
            ("checkpoint", "worker_checkpoint"), ("request", "request"),
        )
        result: dict[str, Any] = {}
        for field, role in role_fields:
            path = _existing_path(result_paths.get(field), campaign_path)
            if path is None:
                continue
            evidence_paths.append({"path": str(path), "role": role})
            if field == "result":
                try:
                    result = _read_json_object(path)
                except ReportingError:
                    result = {}
        disposition = _existing_path(entry.get("disposition_path"), campaign_path)
        if disposition is not None:
            evidence_paths.append({"path": str(disposition), "role": "disposition"})

        current_result_sha = entry.get("result_sha256")
        path = _cell_checkpoint_path(reporting_root, observed)
        if path.exists() and state in {"complete", "negative", "unsupported"}:
            try:
                existing = _read_json_object(path)
                synced = existing.get("provenance", {}).get("declared", {}).get(
                    "campaign_result_sha256"
                )
                if synced == current_result_sha and existing.get("status") \
                        == canonical_status[state]:
                    skipped += 1
                    continue
            except ReportingError:
                pass

        timing: dict[str, Any] = {}
        if entry.get("started_at") is not None:
            timing["started_at"] = entry["started_at"]
        if entry.get("completed_at") is not None:
            timing["completed_at"] = entry["completed_at"]
        elapsed = _elapsed_between(entry.get("started_at"), entry.get("completed_at"))
        if elapsed is not None:
            timing["elapsed_s"] = elapsed
        resources = result.get("resources")
        declared_completed_stages = entry.get("completed_stages")
        if isinstance(declared_completed_stages, list):
            sync_stages = copy.deepcopy(declared_completed_stages)
        elif state in {"complete", "negative", "unsupported"}:
            sync_stages = copy.deepcopy(observed["expected_stages"])
        else:
            sync_stages = []
        payload: dict[str, Any] = {
            "cell_id": cell_id,
            "status": canonical_status[state],
            "completed_stages": sync_stages,
            "completed_replicates": _completed_replicates_from_result(
                result, observed["expected_replicates"]
            ),
            "evidence_paths": evidence_paths,
            "metrics": copy.deepcopy(result.get("metrics", {}))
                if isinstance(result.get("metrics"), dict) else {},
            "provenance": {
                "sync_source": "live_campaign_projection",
                "campaign_result_sha256": current_result_sha,
                "campaign_disposition_sha256": entry.get("disposition_sha256"),
                "cell_identity_sha256": entry.get("cell_identity_sha256"),
            },
            "timing": timing,
            "resource_samples": [copy.deepcopy(resources)]
                if isinstance(resources, dict) else [],
            "blockers": copy.deepcopy(entry.get("blockers"))
                if isinstance(entry.get("blockers"), list) else None,
            "notes": ["synchronized from detached Ultra campaign projection"],
        }
        try:
            record_checkpoint(campaign_path, reporting_root, payload)
            imported += 1
        except ReportingError as exc:
            errors[cell_id] = str(exc)
    index = aggregate(campaign_path, reporting_root)
    return {
        "imported_cells": imported,
        "skipped_cells": skipped,
        "errors": errors,
        "generation_id": index["generation_id"],
        "summary": index["summary"],
    }


def _load_checkpoints(
    manifest: Mapping[str, Any], reporting_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    valid: dict[str, dict[str, Any]] = {}
    invalid: dict[str, list[str]] = {}
    campaign_sha = manifest["campaign"]["sha256"]
    for cell in manifest["cells"]:
        path = _cell_checkpoint_path(reporting_root, cell)
        if not path.exists():
            continue
        try:
            checkpoint = _read_json_object(path)
            errors = _validate_checkpoint(checkpoint, cell=cell, campaign_sha=campaign_sha)
        except ReportingError as exc:
            errors = [str(exc)]
            checkpoint = None
        if errors:
            invalid[cell["cell_id"]] = errors
        elif checkpoint is not None:
            valid[cell["cell_id"]] = checkpoint
    return valid, invalid


def _completeness(
    manifest: Mapping[str, Any], checkpoints: Mapping[str, Mapping[str, Any]],
    invalid: Mapping[str, Sequence[str]], as_of: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    blocker_counts: Counter[str] = Counter()
    dimension_rows: dict[str, dict[str, list[bool]]] = {
        "models": defaultdict(list), "rates_bpw": defaultdict(list),
        "branches": defaultdict(list), "report_groups": defaultdict(list),
    }
    for cell in manifest["cells"]:
        cell_id = cell["cell_id"]
        checkpoint = checkpoints.get(cell_id)
        if cell_id in invalid:
            status = "invalid_checkpoint"
            complete = False
            blockers = [f"invalid_checkpoint:{error}" for error in invalid[cell_id]]
        elif checkpoint is None:
            status = "missing"
            complete = False
            blockers = ["checkpoint_missing"]
        else:
            status = str(checkpoint["status"])
            complete, blockers = _checkpoint_complete(cell, checkpoint)
        status_counts[status] += 1
        blocker_counts.update(blockers)
        dimension_rows["models"][cell["model_label"]].append(complete)
        dimension_rows["rates_bpw"][f"{cell['rate_bpw']:g}"].append(complete)
        dimension_rows["branches"][cell["branch"]].append(complete)
        dimension_rows["report_groups"][cell["report_group_id"]].append(complete)
        row: dict[str, Any] = {
            "campaign_order": cell["campaign_order"],
            "cell_id": cell_id,
            "cell_spec_sha256": cell["cell_spec_sha256"],
            "model_label": cell["model_label"],
            "params_b": cell["params_b"],
            "rate_bpw": cell["rate_bpw"],
            "branch": cell["branch"],
            "claim_track": cell["claim_track"],
            "report_group_id": cell["report_group_id"],
            "status": status,
            "complete": complete,
            "blockers": blockers,
            "expected_stages": cell["expected_stages"],
            "completed_stages": (
                checkpoint["progress"]["completed_stages"] if checkpoint else []
            ),
            "expected_replicates": cell["expected_replicates"],
            "completed_replicates": (
                checkpoint["progress"]["completed_replicates"] if checkpoint else 0
            ),
            "checkpoint_sha256": (
                checkpoint.get("checkpoint_sha256") if checkpoint else None
            ),
        }
        rows.append(row)

    dimensions: dict[str, Any] = {}
    for dimension, values in dimension_rows.items():
        dimensions[dimension] = {
            key: {
                "expected": len(flags), "complete": sum(flags),
                "remaining": len(flags) - sum(flags), "all_complete": all(flags),
            }
            for key, flags in sorted(values.items())
        }
    completed = sum(row["complete"] for row in rows)
    document: dict[str, Any] = {
        "schema": COMPLETENESS_SCHEMA,
        "campaign_sha256": manifest["campaign"]["sha256"],
        "campaign_observation": _observed_campaign(manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "as_of": as_of,
        "summary": {
            "expected_cells": len(rows), "complete_cells": completed,
            "remaining_cells": len(rows) - completed,
            "completion_fraction": completed / len(rows),
            "all_complete": completed == len(rows),
            "status_counts": dict(sorted(status_counts.items())),
            "blocker_counts": dict(sorted(blocker_counts.items())),
        },
        "dimensions": dimensions,
        "matrix": rows,
    }
    return _stamp(document, "completeness_sha256")


def _elapsed(checkpoint: Mapping[str, Any]) -> float | None:
    value = checkpoint.get("timing", {}).get("elapsed_s")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return None


def _median(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values if math.isfinite(float(value)) and value >= 0]
    return float(statistics.median(rows)) if rows else None


def _empirical_estimate(
    target: Mapping[str, Any], completed: Sequence[tuple[Mapping[str, Any], float]],
) -> tuple[float | None, str, str]:
    groupings: list[tuple[str, Any]] = [
        ("same_model_rate_branch", lambda cell: (
            cell["model_label"], cell["rate_bpw"], cell["branch"]
        )),
        ("same_model_rate", lambda cell: (cell["model_label"], cell["rate_bpw"])),
        ("same_rate_branch_scaled_by_params", lambda cell: (cell["rate_bpw"], cell["branch"])),
        ("same_rate_scaled_by_params", lambda cell: cell["rate_bpw"]),
    ]
    for name, key in groupings[:2]:
        durations = [elapsed for cell, elapsed in completed if key(cell) == key(target)]
        estimate = _median(durations)
        if estimate is not None:
            return estimate, f"empirical_{name}_median", "medium"
    for name, key in groupings[2:]:
        rates = [elapsed / cell["params_b"] for cell, elapsed in completed
                 if key(cell) == key(target) and cell["params_b"] > 0]
        estimate = _median(rates)
        if estimate is not None:
            return estimate * target["params_b"], f"empirical_{name}_median", "low"
    rates = [elapsed / cell["params_b"] for cell, elapsed in completed if cell["params_b"] > 0]
    estimate = _median(rates)
    if estimate is not None:
        return estimate * target["params_b"], "empirical_global_seconds_per_billion_median", "low"
    return None, "no_projection_basis", "unknown"


def _iso_plus(timestamp: str, seconds: float) -> str:
    parsed = dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return (parsed + dt.timedelta(seconds=seconds)).astimezone(dt.timezone.utc).isoformat(
        timespec="seconds"
    )


def _eta_ledger(
    manifest: Mapping[str, Any], checkpoints: Mapping[str, Mapping[str, Any]],
    completeness: Mapping[str, Any], as_of: str,
) -> dict[str, Any]:
    cell_by_id = _cell_map(manifest)
    completed_observations = [
        (cell_by_id[cell_id], elapsed)
        for cell_id, checkpoint in checkpoints.items()
        if checkpoint.get("status") in TERMINAL_COMPLETE
        and (elapsed := _elapsed(checkpoint)) is not None
    ]
    complete_by_id = {row["cell_id"]: row["complete"] for row in completeness["matrix"]}
    projections: list[dict[str, Any]] = []
    for cell in manifest["cells"]:
        checkpoint = checkpoints.get(cell["cell_id"])
        observed = _elapsed(checkpoint) if checkpoint else None
        if complete_by_id[cell["cell_id"]]:
            remaining, source, confidence = 0.0, "measured_complete", "measured"
        elif checkpoint and checkpoint.get("eta", {}).get("remaining_s") is not None:
            remaining = float(checkpoint["eta"]["remaining_s"])
            source = "worker_reported_remaining"
            confidence = str(checkpoint["eta"].get("confidence", "reported"))
        elif cell["estimated_seconds"] is not None:
            total = float(cell["estimated_seconds"])
            remaining = max(0.0, total - (observed or 0.0))
            source, confidence = "campaign_declared_total", "declared"
        else:
            total, source, confidence = _empirical_estimate(cell, completed_observations)
            remaining = None if total is None else max(0.0, total - (observed or 0.0))
        projections.append({
            "campaign_order": cell["campaign_order"],
            "cell_id": cell["cell_id"],
            "model_label": cell["model_label"],
            "params_b": cell["params_b"],
            "rate_bpw": cell["rate_bpw"],
            "branch": cell["branch"],
            "report_group_id": cell["report_group_id"],
            "status": checkpoint.get("status") if checkpoint else "missing",
            "observed_elapsed_s": observed,
            "projected_remaining_s": remaining,
            "projection_source": source,
            "confidence": confidence,
        })

    concurrency = manifest["report_policy"]["max_parallel_cells_for_eta_projection"]
    slots = [0.0] * concurrency
    unknown = 0
    schedule: list[dict[str, Any]] = []
    for row in projections:
        remaining = row["projected_remaining_s"]
        if remaining is None:
            unknown += 1
            schedule.append({
                "cell_id": row["cell_id"], "slot": None,
                "projected_start_offset_s": None, "projected_finish_offset_s": None,
                "reason": "duration_unknown",
            })
            continue
        slot = min(range(concurrency), key=lambda index: (slots[index], index))
        start = slots[slot]
        finish = start + remaining
        slots[slot] = finish
        schedule.append({
            "cell_id": row["cell_id"], "slot": slot,
            "projected_start_offset_s": start, "projected_finish_offset_s": finish,
            "projected_start_at": _iso_plus(as_of, start),
            "projected_finish_at": _iso_plus(as_of, finish),
            "reason": None,
        })
    known_remaining = [row["projected_remaining_s"] for row in projections
                       if row["projected_remaining_s"] is not None]
    serial = sum(known_remaining)
    projected = max(slots) if slots else 0.0

    group_summary: dict[str, Any] = {}
    for group in manifest["report_groups"]:
        rows = [row for row in projections if row["report_group_id"] == group["group_id"]]
        values = [row["projected_remaining_s"] for row in rows
                  if row["projected_remaining_s"] is not None]
        missing = sum(row["projected_remaining_s"] is None for row in rows)
        group_summary[group["group_id"]] = {
            "cells": len(rows), "known_projection_cells": len(values),
            "unknown_projection_cells": missing,
            "serial_remaining_s_known_lower_bound": sum(values),
            "total_eta_known": missing == 0,
        }

    throughput = {
        "completed_timed_cells": len(completed_observations),
        "observed_elapsed_s": sum(elapsed for _, elapsed in completed_observations),
        "median_seconds_per_billion_parameters": _median(
            elapsed / cell["params_b"] for cell, elapsed in completed_observations
            if cell["params_b"] > 0
        ),
    }
    summary = {
        "max_parallel_cells": concurrency,
        "known_projection_cells": len(known_remaining),
        "unknown_projection_cells": unknown,
        "serial_remaining_s_known_lower_bound": serial,
        "scheduled_remaining_s_known_lower_bound": projected,
        "total_eta_known": unknown == 0,
        "projected_finish_at": _iso_plus(as_of, projected) if unknown == 0 else None,
        "projected_finish_reason": None if unknown == 0 else "one_or_more_cell_durations_unknown",
    }
    document: dict[str, Any] = {
        "schema": ETA_SCHEMA,
        "campaign_sha256": manifest["campaign"]["sha256"],
        "campaign_observation": _observed_campaign(manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "as_of": as_of,
        "semantics": {
            "projection_not_deadline": True,
            "campaign_order_preserved": True,
            "missing_duration_never_treated_as_zero": True,
            "empirical_scaling_uses_catalogue_parameter_tier": True,
        },
        "summary": summary,
        "throughput": throughput,
        "report_groups": group_summary,
        "cells": projections,
        "schedule": schedule,
    }
    return _stamp(document, "eta_ledger_sha256")


def _safe_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._") or "model"
    return f"{slug[:80]}-{hashlib.sha256(value.encode()).hexdigest()[:12]}"


def _workspace_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError as exc:
        raise ReportingError(f"Ultra report artifact must be inside workspace: {path}") from exc


def _ultra_coverage(
    manifest: Mapping[str, Any], group: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    wanted = set(group["cell_ids"])
    evidence: list[dict[str, Any]] = []
    terminal = True
    for cell in _observed_cells(manifest):
        if cell["cell_id"] not in wanted:
            continue
        entry = cell["campaign_entry"]
        status = entry.get("status")
        evidence.append({
            "cell_id": cell["cell_id"],
            "status": status,
            "result_sha256": entry.get("result_sha256"),
            "disposition_sha256": entry.get("disposition_sha256"),
        })
        terminal = terminal and status in {"complete", "negative", "unsupported"}
    return evidence, terminal and len(evidence) == len(wanted)


def _ultra_report_checkpoint(
    manifest: Mapping[str, Any], group: Mapping[str, Any], report_source_path: Path,
    published_report_path: Path,
) -> dict[str, Any] | None:
    campaign = _observed_campaign(manifest)
    if campaign.get("declared_schema") != "hawking.doctor_v5_ultra_campaign.v1":
        return None
    if not _is_sha(campaign.get("plan_sha256")) \
            or not isinstance(campaign.get("declared_version"), str):
        return None
    coverage, terminal = _ultra_coverage(manifest, group)
    if not terminal:
        return None
    artifact = _file_ref(report_source_path)
    artifact["path"] = _workspace_relative(published_report_path)
    receipt: dict[str, Any] = {
        "schema": "hawking.doctor_v5_ultra_report_checkpoint.v1",
        "version": campaign["declared_version"],
        "plan_sha256": campaign["plan_sha256"],
        "group_id": group["group_id"],
        "covered_cells_sha256": _sha_value(coverage),
        "report_artifact": artifact,
        "verified": True,
        "source_deletion_permitted": False,
    }
    receipt["checkpoint_sha256"] = _sha_value(receipt)
    return receipt


def _ledger_value(checkpoint: Mapping[str, Any], path: str) -> float | None:
    found, raw = _path_value(checkpoint.get("metric_ledger", {}), path)
    value = raw.get("value") if found and isinstance(raw, dict) else None
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)):
        return None
    return float(value)


def _worker_owned_packed_refs(checkpoint: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return only explicit, content-addressed bundle-shard payload references."""
    rows: list[dict[str, Any]] = []
    for raw in checkpoint.get("evidence", []):
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role", ""))
        locator = raw.get("locator", raw.get("path"))
        if not role.startswith("bundle_shard:") or not isinstance(locator, str) \
                or not locator.endswith(".strand") or not _is_sha(raw.get("sha256")) \
                or isinstance(raw.get("bytes"), bool) or not isinstance(raw.get("bytes"), int):
            continue
        candidate = Path(locator).expanduser()
        candidates = [candidate] if candidate.is_absolute() else [ROOT / candidate]
        resolved = next((path.resolve(strict=False) for path in candidates), None)
        if resolved is None:
            continue
        try:
            relative = str(resolved.relative_to(ROOT.resolve()))
        except ValueError:
            continue
        rows.append({
            "role": role, "path": str(resolved), "workspace_relative_path": relative,
            "sha256": raw["sha256"], "bytes": raw["bytes"],
            "verification": raw.get("verification"),
        })
    return sorted(rows, key=lambda row: (row["role"], row["path"]))


def _retention_quality(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for checkpoint in checkpoints:
        capability = _ledger_value(checkpoint, "quality.capability.candidate_aggregate")
        if capability is not None:
            return {"metric": "quality.capability.candidate_aggregate", "value": capability,
                    "objective": "maximize", "cell_id": checkpoint["cell_id"]}
    for checkpoint in checkpoints:
        ppl = _ledger_value(checkpoint, "quality.ppl.candidate")
        if ppl is not None and ppl >= 0:
            return {"metric": "quality.ppl.candidate", "value": ppl,
                    "objective": "minimize", "cell_id": checkpoint["cell_id"]}
    for checkpoint in checkpoints:
        loss = _ledger_value(checkpoint, "quality.loss.candidate")
        if loss is not None:
            return {"metric": "quality.loss.candidate", "value": loss,
                    "objective": "minimize", "cell_id": checkpoint["cell_id"]}
    return {"metric": None, "value": None, "objective": None, "cell_id": None}


def _chain_filename(model_label: str, rate_bpw: float) -> str:
    identity = {"model_label": model_label, "rate_bpw": rate_bpw}
    model = re.sub(r"[^A-Za-z0-9._-]+", "-", model_label).strip("-._") or "model"
    rate = f"{rate_bpw:g}".replace(".", "p")
    return f"{model[:60]}-q{rate}-{_sha_value(identity)[:12]}.json"


def _automatic_retention_decisions(
    manifest: Mapping[str, Any], checkpoints: Mapping[str, Mapping[str, Any]],
    as_of: str, reporting_root: Path,
) -> dict[str, dict[str, Any]]:
    """Adjudicate at most one packed artifact per model from reported metrics.

    A decision exists only after a complete four-branch model×rate chain has
    hashed reporter evidence.  Missing metrics are never converted to a score:
    disk policy requires an explicit ``retain_none`` instead of unknown-retain.
    """
    campaign = _observed_campaign(manifest)
    if campaign.get("declared_schema") != "hawking.doctor_v5_ultra_campaign.v1":
        return {}
    required_branches = (
        "codec_control", "doctor_static", "doctor_conditional", "doctor_full",
    )
    terminal = {"complete", "negative", "unsupported"}
    chains: dict[tuple[str, float], dict[str, Any]] = {}
    for cell in _observed_cells(manifest):
        key = (cell["model_label"], float(cell["rate_bpw"]))
        chains.setdefault(key, {"cells": {}})["cells"][cell["branch"]] = cell

    eligible_chains: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    for (model_label, rate_bpw), chain in sorted(chains.items()):
        cells = chain["cells"]
        if set(cells) != set(required_branches):
            continue
        branch_rows: list[dict[str, Any]] = []
        branch_checkpoints: dict[str, dict[str, Any]] = {}
        ready = True
        for branch in required_branches:
            cell = cells[branch]
            entry = cell["campaign_entry"]
            status = entry.get("status")
            checkpoint = checkpoints.get(cell["cell_id"])
            disposition_required = status in {"negative", "unsupported"}
            if status not in terminal or not isinstance(checkpoint, dict) \
                    or not checkpoint.get("evidence") \
                    or not _is_sha(checkpoint.get("checkpoint_sha256")) \
                    or (status == "complete" and not _is_sha(entry.get("result_sha256"))) \
                    or (disposition_required and not _is_sha(entry.get("disposition_sha256"))):
                ready = False
                break
            branch_checkpoints[branch] = checkpoint
            branch_rows.append({
                "branch": branch, "cell_id": cell["cell_id"], "status": status,
                "cell_spec_sha256": cell["cell_spec_sha256"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "result_sha256": entry.get("result_sha256"),
                "disposition_sha256": entry.get("disposition_sha256"),
                "evidence_sha256": sorted({row["sha256"] for row in checkpoint["evidence"]
                                            if _is_sha(row.get("sha256"))}),
            })
        if not ready:
            continue
        by_branch = branch_checkpoints
        codec = by_branch["codec_control"]
        packed = _worker_owned_packed_refs(codec)
        physical_bpw = _ledger_value(codec, "physical_bpw.model_payload")
        ordered_quality = [by_branch[name] for name in (
            "doctor_full", "doctor_conditional", "doctor_static", "codec_control"
        )]
        quality = _retention_quality(ordered_quality)
        chain_id = _sha_value({
            "campaign_plan_sha256": campaign.get("plan_sha256"),
            "model_label": model_label, "rate_bpw": rate_bpw,
            "cell_ids": [cells[name]["cell_id"] for name in required_branches],
        })
        row = {
            "chain_id": chain_id, "model_label": model_label,
            "params_b": cells["codec_control"]["params_b"], "rate_bpw": rate_bpw,
            "branch_evidence": branch_rows,
            "codec_cell_id": cells["codec_control"]["cell_id"],
            "packed_artifacts": packed, "physical_model_payload_bpw": physical_bpw,
            "quality": quality,
        }
        row["eligible"] = bool(packed) and physical_bpw is not None and physical_bpw > 0 \
            and quality["value"] is not None
        row["ineligibility_reasons"] = [
            reason for condition, reason in (
                (not packed, "no_reported_worker_owned_packed_artifact"),
                (physical_bpw is None or physical_bpw <= 0,
                 "physical_model_payload_bpw_not_reported"),
                (quality["value"] is None, "quality_metric_not_reported"),
            ) if condition
        ]
        closed.append(row)
        if row["eligible"]:
            eligible_chains.append(row)

    selected_by_model: dict[str, str] = {}
    for model_label in sorted({row["model_label"] for row in eligible_chains}):
        candidates = [row for row in eligible_chains if row["model_label"] == model_label]
        metric_priority = {"quality.capability.candidate_aggregate": 0,
                           "quality.ppl.candidate": 1, "quality.loss.candidate": 2}
        def rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
            quality = row["quality"]
            value = float(quality["value"])
            objective_value = -value if quality["objective"] == "maximize" else value
            return (metric_priority[quality["metric"]], objective_value,
                    float(row["physical_model_payload_bpw"]), float(row["rate_bpw"]),
                    row["chain_id"])
        selected_by_model[model_label] = min(candidates, key=rank)["chain_id"]

    decisions: dict[str, dict[str, Any]] = {}
    decision_root = reporting_root / "retention_decisions"
    for row in closed:
        selected_chain_id = selected_by_model.get(row["model_label"])
        retain = row["eligible"] and row["chain_id"] == selected_chain_id
        filename = _chain_filename(row["model_label"], row["rate_bpw"])
        live_path = decision_root / filename
        decision: dict[str, Any] = {
            "schema": RETENTION_SCHEMA, "reporter_version": REPORTER_VERSION,
            "created_at": as_of,
            "campaign_sha256": manifest["campaign"]["sha256"],
            "campaign_observation_sha256": campaign["sha256"],
            "plan_sha256": campaign.get("plan_sha256"),
            "chain": {
                "chain_id": row["chain_id"], "model_label": row["model_label"],
                "params_b": row["params_b"], "rate_bpw": row["rate_bpw"],
                "branch_evidence": row["branch_evidence"],
                "all_four_branches_terminal_and_hashed": True,
            },
            "metric_basis": {
                "eligible": row["eligible"],
                "physical_model_payload_bpw": row["physical_model_payload_bpw"],
                "quality": row["quality"],
                "ineligibility_reasons": row["ineligibility_reasons"],
                "selection_order": (
                    "reported capability descending, else reported PPL ascending, else "
                    "reported loss ascending; then physical bpw, target rate, chain id"
                ),
                "retention_is_not_a_quality_or_dominance_claim": True,
            },
            "decision": {
                "action": "retain_one" if retain else "retain_none",
                "selected_packed_cell_id": row["codec_cell_id"] if retain else None,
                "selected_chain_id_for_model": selected_chain_id,
                "selected_packed_artifacts": row["packed_artifacts"] if retain else [],
                "deletable_worker_owned_packed_artifacts": [] if retain
                    else row["packed_artifacts"],
                "reason": "best_reported_metric_candidate_for_model" if retain else (
                    "another_reported_candidate_selected_for_model" if row["eligible"]
                    and selected_chain_id is not None else
                    "missing_reported_retention_metrics_defaults_to_retain_none"
                ),
            },
            "gc_contract": {
                "scheduler_must_rehash_before_delete": True,
                "deletion_scope": "listed worker-owned bundle_shard .strand payloads only",
                "artifact_liveness_required_when_report_is_built": False,
                "missing_listed_artifact_disposition": "already_gc_noop",
                "rolling_scheduler_lifecycle_receipts_take_precedence": True,
                "reporting_decision_cannot_resurrect_or_require_a_gc_predecessor": True,
                "preserve": ["receipts", "metrics", "specs", "logs", "requests",
                             "checkpoints", "manifests", "dispositions"],
                "maximum_retained_packed_candidates_per_model": 1,
                "unknown_defaults_to_retain": False,
                "parent_source_deletion_permitted": False,
            },
            "live_path": str(live_path),
            "source_deletion_permitted": False,
        }
        decision["receipt_sha256"] = _sha_value(decision)
        decisions[filename] = decision
    return decisions


def _group_report(
    manifest: Mapping[str, Any], group: Mapping[str, Any],
    checkpoints: Mapping[str, Mapping[str, Any]], completeness: Mapping[str, Any],
    eta: Mapping[str, Any], as_of: str,
    retention_decisions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    cell_ids = set(group["cell_ids"])
    comp_rows = [row for row in completeness["matrix"] if row["cell_id"] in cell_ids]
    eta_rows = [row for row in eta["cells"] if row["cell_id"] in cell_ids]
    rows: list[dict[str, Any]] = []
    cells = _cell_map(manifest)
    for comp in comp_rows:
        cell_id = comp["cell_id"]
        checkpoint = checkpoints.get(cell_id)
        rows.append({
            "cell_spec": copy.deepcopy(cells[cell_id]),
            "completeness": copy.deepcopy(comp),
            "eta": copy.deepcopy(next(row for row in eta_rows if row["cell_id"] == cell_id)),
            "checkpoint": copy.deepcopy(checkpoint),
        })
    complete = sum(row["complete"] for row in comp_rows)
    gc_eligible: list[dict[str, Any]] = []
    retained: list[str] = []
    automatic = [copy.deepcopy(decision) for decision in retention_decisions.values()
                 if decision.get("chain", {}).get("model_label") in group["models"]]
    if automatic:
        for decision in automatic:
            selected = decision["decision"]["selected_packed_cell_id"]
            if isinstance(selected, str):
                retained.append(selected)
            artifacts = decision["decision"]["deletable_worker_owned_packed_artifacts"]
            if artifacts:
                gc_eligible.append({
                    "cell_id": decision["chain"]["branch_evidence"][0]["cell_id"],
                    "decision_receipt_sha256": decision["receipt_sha256"],
                    "chain_id": decision["chain"]["chain_id"],
                    "artifacts": copy.deepcopy(artifacts),
                })
    else:
        # Generic campaigns may still supply an explicit external Pareto
        # adjudication.  Undetermined is not silently retained.
        for row in rows:
            checkpoint = row["checkpoint"]
            pareto = checkpoint.get("pareto", {}) if isinstance(checkpoint, dict) else {}
            if pareto.get("disposition") == "retain" \
                    and _is_sha(pareto.get("decision_receipt_sha256")):
                retained.append(row["cell_spec"]["cell_id"])
            elif pareto.get("disposition") == "discard_non_pareto" \
                    and _is_sha(pareto.get("decision_receipt_sha256")) \
                    and pareto.get("dominated_by_cell_ids"):
                gc_eligible.append({
                    "cell_id": row["cell_spec"]["cell_id"],
                    "decision_receipt_sha256": pareto["decision_receipt_sha256"],
                    "dominated_by_cell_ids": copy.deepcopy(pareto["dominated_by_cell_ids"]),
                })
    document: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "reporter_version": REPORTER_VERSION,
        "campaign_sha256": manifest["campaign"]["sha256"],
        "campaign_observation": _observed_campaign(manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "as_of": as_of,
        "group": copy.deepcopy(group),
        "coverage": {
            "expected_cells": len(comp_rows), "complete_cells": complete,
            "remaining_cells": len(comp_rows) - complete,
            "complete": complete == len(comp_rows),
        },
        "claim_guard": {
            "aggregation_is_not_scientific_validation": True,
            "report_complete_only_if_every_expected_cell_is_complete": True,
            "quality_or_dominance_claim_inferred_by_reporter": False,
            "negative_results_retained": True,
        },
        "retention_policy": {
            "unknown_defaults_to_retain": False,
            "averages_never_imply_pareto_disposition": True,
            "gc_scope": "explicitly_listed_worker_owned_bundle_shard_payloads_only",
            "receipts_hashes_metrics_always_retained": True,
            "parent_source_deletion_permitted": False,
            "maximum_retained_packed_candidates_per_model": 1,
            "retained_cell_ids": sorted(set(retained)),
            "gc_eligible_candidates": gc_eligible,
            "automatic_chain_decisions": [{
                "chain_id": decision["chain"]["chain_id"],
                "receipt_sha256": decision["receipt_sha256"],
                "live_path": decision["live_path"],
                "action": decision["decision"]["action"],
            } for decision in automatic],
        },
        "cells": rows,
    }
    return _stamp(document, "report_sha256")


def _as_of(manifest: Mapping[str, Any], checkpoints: Mapping[str, Mapping[str, Any]]) -> str:
    timestamps = [str(checkpoint.get("recorded_at")) for checkpoint in checkpoints.values()
                  if checkpoint.get("recorded_at")]
    if timestamps:
        return max(timestamps)
    # The manifest is intentionally deterministic.  A no-checkpoint aggregate is
    # the sole case where wall-clock time becomes part of a snapshot identity.
    return _now()


def _snapshot_ref(path: Path, snapshot_root: Path, role: str) -> dict[str, Any]:
    reference = _file_ref(path, role=role)
    reference["path"] = str(path.relative_to(snapshot_root))
    return reference


def aggregate(campaign_path: Path, reporting_root: Path) -> dict[str, Any]:
    """Create a durable completeness/ETA/report snapshot, then atomically publish its index."""
    manifest = _bound_manifest(campaign_path, reporting_root)
    with _exclusive_lock(reporting_root):
        checkpoints, invalid = _load_checkpoints(manifest, reporting_root)
        as_of = _as_of(manifest, checkpoints)
        completeness = _completeness(manifest, checkpoints, invalid, as_of)
        eta = _eta_ledger(manifest, checkpoints, completeness, as_of)
        retention_decisions = _automatic_retention_decisions(
            manifest, checkpoints, as_of, reporting_root
        )
        reports = [
            _group_report(
                manifest, group, checkpoints, completeness, eta, as_of,
                retention_decisions,
            )
            for group in manifest["report_groups"]
        ]
        state_binding = {
            "manifest_sha256": manifest["manifest_sha256"],
            "campaign_observation_sha256": _observed_campaign(manifest)["sha256"],
            "checkpoint_sha256_by_cell": {
                cell_id: checkpoint["checkpoint_sha256"]
                for cell_id, checkpoint in sorted(checkpoints.items())
            },
            "invalid_checkpoint_errors": {key: list(value) for key, value in sorted(invalid.items())},
            "retention_receipt_sha256_by_chain": {
                decision["chain"]["chain_id"]: decision["receipt_sha256"]
                for decision in retention_decisions.values()
            },
            "as_of": as_of,
        }
        generation_id = _sha_value(state_binding)
        snapshots = reporting_root / "snapshots"
        snapshots.mkdir(parents=True, exist_ok=True)
        final = snapshots / generation_id
        if not final.exists():
            temporary = Path(tempfile.mkdtemp(prefix=f".{generation_id}.tmp.", dir=snapshots))
            try:
                _atomic_json(temporary / "completeness.json", completeness)
                _atomic_json(temporary / "eta_ledger.json", eta)
                for filename, decision in retention_decisions.items():
                    _atomic_json(temporary / "retention_decisions" / filename, decision)
                for report in reports:
                    group = report["group"]
                    if group["kind"] == "consolidated_sub_frontier_cohort":
                        path = temporary / "reports" / "sub-120B.json"
                    else:
                        path = temporary / "reports" / "frontier" / (
                            _safe_name(group["models"][0]) + ".json"
                        )
                    _atomic_json(path, report)
                    published_path = final / path.relative_to(temporary)
                    receipt = _ultra_report_checkpoint(
                        manifest, group, path, published_path
                    )
                    if receipt is not None:
                        _atomic_json(
                            temporary / "report_checkpoints"
                            / f"{_safe_name(group['group_id'])}.json",
                            receipt,
                        )
                _fsync_dir(temporary)
                os.rename(temporary, final)
                _fsync_dir(snapshots)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise

        completeness_ref = _snapshot_ref(
            final / "completeness.json", final, "completeness_matrix"
        )
        eta_ref = _snapshot_ref(final / "eta_ledger.json", final, "eta_ledger")
        report_refs: list[dict[str, Any]] = []
        for path in sorted((final / "reports").rglob("*.json")):
            report = _read_json_object(path)
            reference = _snapshot_ref(path, final, "evidence_report")
            reference.update({
                "group_id": report["group"]["group_id"],
                "kind": report["group"]["kind"],
                "complete": report["coverage"]["complete"],
            })
            report_refs.append(reference)
        checkpoint_refs = [
            _snapshot_ref(path, final, "ultra_report_checkpoint")
            for path in sorted((final / "report_checkpoints").glob("*.json"))
        ] if (final / "report_checkpoints").is_dir() else []
        retention_refs: list[dict[str, Any]] = []
        for filename, decision in sorted(retention_decisions.items()):
            snapshot_path = final / "retention_decisions" / filename
            live_path = Path(decision["live_path"])
            if not live_path.exists() or _read_json_object(live_path) != decision:
                _atomic_json(live_path, decision)
            reference = _snapshot_ref(snapshot_path, final, "chain_retention_decision")
            reference.update({
                "chain_id": decision["chain"]["chain_id"],
                "model_label": decision["chain"]["model_label"],
                "rate_bpw": decision["chain"]["rate_bpw"],
                "action": decision["decision"]["action"],
                "live_path": str(live_path),
                "receipt_sha256": decision["receipt_sha256"],
            })
            retention_refs.append(reference)

        index_path = reporting_root / "report_index.json"
        previous_generation = None
        previous_index_sha = None
        if index_path.exists():
            previous = _read_json_object(index_path)
            if _validate_stamp(previous, "index_sha256"):
                previous_generation = previous.get("generation_id")
                previous_index_sha = previous.get("index_sha256")
                if previous_generation == generation_id:
                    return previous
        index: dict[str, Any] = {
            "schema": INDEX_SCHEMA,
            "reporter_version": REPORTER_VERSION,
            "campaign": _observed_campaign(manifest),
            "manifest": _file_ref(_manifest_path(reporting_root), role="reporting_manifest"),
            "generation_id": generation_id,
            "snapshot_path": str(final),
            "as_of": as_of,
            "previous_generation_id": (
                previous_generation if previous_generation != generation_id else None
            ),
            "previous_index_sha256": (
                previous_index_sha if previous_generation != generation_id else None
            ),
            "completeness": completeness_ref,
            "eta_ledger": eta_ref,
            "reports": report_refs,
            "ultra_report_checkpoints": checkpoint_refs,
            "retention_decisions": retention_refs,
            "summary": {
                **copy.deepcopy(completeness["summary"]),
                "eta": copy.deepcopy(eta["summary"]),
            },
        }
        index = _stamp(index, "index_sha256")
        _atomic_json(index_path, index)
    return index


def _verify_ref(path: Path, reference: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        digest, size = _hash_stable_file(path)
    except ReportingError as exc:
        return [str(exc)]
    if reference.get("sha256") != digest:
        errors.append(f"hash mismatch: {path}")
    if reference.get("bytes") != size:
        errors.append(f"byte count mismatch: {path}")
    return errors


def verify(campaign_path: Path, reporting_root: Path, *, deep: bool = False) -> list[str]:
    errors: list[str] = []
    try:
        manifest = _bound_manifest(campaign_path, reporting_root)
    except ReportingError as exc:
        return [str(exc)]
    index_path = reporting_root / "report_index.json"
    if not index_path.exists():
        return ["report_index.json is missing"]
    try:
        index = _read_json_object(index_path)
    except ReportingError as exc:
        return [str(exc)]
    if index.get("schema") != INDEX_SCHEMA or not _validate_stamp(index, "index_sha256"):
        errors.append("report index schema/hash is invalid")
    if not _is_sha(index.get("campaign", {}).get("sha256")) \
            or index.get("campaign", {}).get("path") != manifest["campaign"]["path"]:
        errors.append("report index campaign observation binding is invalid")
    snapshot = Path(index.get("snapshot_path", ""))
    if not snapshot.is_dir() or snapshot.name != index.get("generation_id"):
        errors.append("snapshot directory/generation binding is invalid")
        return errors
    for field in ("completeness", "eta_ledger"):
        reference = index.get(field)
        if not isinstance(reference, dict):
            errors.append(f"index {field} reference is invalid")
            continue
        errors.extend(_verify_ref(snapshot / reference["path"], reference))
    for reference in index.get("reports", []):
        if not isinstance(reference, dict) or "path" not in reference:
            errors.append("report reference is invalid")
            continue
        path = snapshot / reference["path"]
        errors.extend(_verify_ref(path, reference))
        try:
            report = _read_json_object(path)
            if not _validate_stamp(report, "report_sha256"):
                errors.append(f"report self-hash mismatch: {path}")
        except ReportingError as exc:
            errors.append(str(exc))
    for reference in index.get("ultra_report_checkpoints", []):
        if not isinstance(reference, dict) or "path" not in reference:
            errors.append("Ultra report-checkpoint reference is invalid")
            continue
        path = snapshot / reference["path"]
        errors.extend(_verify_ref(path, reference))
        try:
            receipt = _read_json_object(path)
            digest = receipt.get("checkpoint_sha256")
            if not _is_sha(digest) or digest != _sha_value(
                    _identity_payload(receipt, "checkpoint_sha256")):
                errors.append(f"Ultra report checkpoint self-hash mismatch: {path}")
        except ReportingError as exc:
            errors.append(str(exc))
    for reference in index.get("retention_decisions", []):
        if not isinstance(reference, dict) or "path" not in reference \
                or "live_path" not in reference:
            errors.append("chain retention-decision reference is invalid")
            continue
        path = snapshot / reference["path"]
        errors.extend(_verify_ref(path, reference))
        try:
            decision = _read_json_object(path)
            if decision.get("schema") != RETENTION_SCHEMA \
                    or not _validate_stamp(decision, "receipt_sha256") \
                    or decision.get("receipt_sha256") != reference.get("receipt_sha256"):
                errors.append(f"retention decision self-hash mismatch: {path}")
            live = Path(reference["live_path"])
            errors.extend(_verify_ref(live, reference))
        except ReportingError as exc:
            errors.append(str(exc))

    completeness_path = snapshot / index["completeness"]["path"]
    eta_path = snapshot / index["eta_ledger"]["path"]
    try:
        if not _validate_stamp(_read_json_object(completeness_path), "completeness_sha256"):
            errors.append("completeness self-hash mismatch")
        if not _validate_stamp(_read_json_object(eta_path), "eta_ledger_sha256"):
            errors.append("ETA ledger self-hash mismatch")
    except ReportingError as exc:
        errors.append(str(exc))

    checkpoints, invalid = _load_checkpoints(manifest, reporting_root)
    for cell_id, cell_errors in invalid.items():
        errors.extend(f"checkpoint {cell_id}: {error}" for error in cell_errors)
    if deep:
        for cell_id, checkpoint in checkpoints.items():
            for reference in checkpoint.get("evidence", []):
                path_value = reference.get("path")
                if path_value is None:
                    continue
                errors.extend(_verify_ref(Path(path_value), reference))
    return errors


def status(reporting_root: Path) -> dict[str, Any]:
    index_path = reporting_root / "report_index.json"
    if not index_path.exists():
        return {
            "initialized": _manifest_path(reporting_root).exists(),
            "aggregated": False,
            "reporting_root": str(reporting_root),
        }
    index = _read_json_object(index_path)
    if not _validate_stamp(index, "index_sha256"):
        raise ReportingError("report index self-hash is invalid")
    return {
        "initialized": True,
        "aggregated": True,
        "reporting_root": str(reporting_root),
        "generation_id": index["generation_id"],
        "as_of": index["as_of"],
        "summary": index["summary"],
        "reports": index["reports"],
        "retention_decisions": index.get("retention_decisions", []),
    }


def _selftest() -> None:
    with tempfile.TemporaryDirectory() as temporary_raw:
        temporary = Path(temporary_raw)
        campaign = temporary / "campaign.json"
        reporting = temporary / "reporting"
        evidence = temporary / "result.json"
        _atomic_json(campaign, {
            "schema": "test.campaign.v1",
            "reporting": {"frontier_threshold_b": 120, "max_parallel_cells": 2},
            "jobs": [
                {"cell_id": "7-q4-control", "model_label": "7B", "rate": "q4",
                 "branch": "control", "required_stages": ["L0", "L1"],
                 "estimated_seconds": 10},
                {"cell_id": "120-q2-doctor", "model_label": "120B", "rate": 2,
                 "branch": "doctor", "required_stages": ["L0"],
                 "estimated_seconds": 20},
            ],
        })
        _atomic_json(evidence, {
            "parameter_accounting": {"stored_parameter_count": 7_000_000_000,
                                     "quantized_parameter_count": 7_000_000_000},
            "physical_accounting": {"packed_bytes": 3_500_000_000,
                                    "full_bundle_bytes": 3_500_000_123},
            "ppl": {"baseline": 10.0, "reconstruction": 10.5},
        })
        manifest = initialize(campaign, reporting)
        assert manifest["counts"]["cells"] == 2
        assert [group["group_id"] for group in manifest["report_groups"]] == [
            "sub-120B", "120B",
        ]
        checkpoint = record_checkpoint(campaign, reporting, {
            "cell_id": "7-q4-control", "status": "succeeded",
            "completed_stages": ["L0", "L1"], "completed_replicates": 1,
            "evidence_paths": [{"path": str(evidence), "role": "result"}],
            "metrics": {"timing": {"encode_wall_s": 8.0}},
            "provenance": {"seed": 7}, "timing": {"elapsed_s": 10.0},
        })
        assert checkpoint["completeness"]["complete"] is True
        assert checkpoint["metric_ledger"]["bytes"]["doctor"]["value"] is None
        assert checkpoint["metric_ledger"]["bytes"]["doctor"]["reason"] == "not_reported"
        assert math.isclose(
            checkpoint["metric_ledger"]["physical_bpw"]["packed"]["value"], 4.0
        )
        index = aggregate(campaign, reporting)
        assert index["summary"]["complete_cells"] == 1
        assert index["summary"]["remaining_cells"] == 1
        assert len(index["reports"]) == 2
        assert verify(campaign, reporting) == []
        try:
            record_checkpoint(campaign, reporting, {
                "cell_id": "7-q4-control", "status": "running",
                "completed_stages": ["L0"], "completed_replicates": 0,
            })
        except ReportingError:
            pass
        else:
            raise AssertionError("checkpoint regression unexpectedly accepted")
    print(json.dumps({"ok": True, "reporter": REPORTER_VERSION}, sort_keys=True))


def _root_argument(arguments: argparse.Namespace) -> Path:
    return Path(arguments.reporting_root).expanduser().resolve()


def _campaign_argument(arguments: argparse.Namespace) -> Path:
    return Path(arguments.campaign).expanduser().resolve()


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--reporting-root", type=Path, default=DEFAULT_REPORTING_ROOT)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "sync", "aggregate", "status", "verify"):
        child = commands.add_parser(name)
        _add_common(child)
        if name == "verify":
            child.add_argument("--deep", action="store_true")
    record = commands.add_parser("record")
    _add_common(record)
    record.add_argument("--input", type=Path, required=True)
    commands.add_parser("selftest")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "selftest":
            _selftest()
            return 0
        campaign = _campaign_argument(arguments)
        reporting_root = _root_argument(arguments)
        if arguments.command == "init":
            result = initialize(campaign, reporting_root)
            print(json.dumps({
                "ok": True, "manifest_sha256": result["manifest_sha256"],
                "counts": result["counts"], "reporting_root": str(reporting_root),
            }, indent=2, sort_keys=True))
            return 0
        if arguments.command == "record":
            input_path = Path(arguments.input).expanduser().resolve(strict=True)
            payload = _read_json_object(input_path)
            result = record_checkpoint(
                campaign, reporting_root, payload, source_input=input_path
            )
            print(json.dumps({
                "ok": True, "cell_id": result["cell_id"], "status": result["status"],
                "revision": result["revision"],
                "checkpoint_sha256": result["checkpoint_sha256"],
                "complete": result["completeness"]["complete"],
            }, indent=2, sort_keys=True))
            return 0
        if arguments.command == "aggregate":
            result = aggregate(campaign, reporting_root)
            print(json.dumps({
                "ok": True, "generation_id": result["generation_id"],
                "summary": result["summary"], "reports": result["reports"],
            }, indent=2, sort_keys=True))
            return 0
        if arguments.command == "sync":
            result = sync_campaign(campaign, reporting_root)
            print(json.dumps({"ok": not result["errors"], **result},
                             indent=2, sort_keys=True))
            return 0 if not result["errors"] else 1
        if arguments.command == "status":
            print(json.dumps(status(reporting_root), indent=2, sort_keys=True))
            return 0
        errors = verify(campaign, reporting_root, deep=arguments.deep)
        print(json.dumps({"ok": not errors, "errors": errors}, indent=2, sort_keys=True))
        return 0 if not errors else 1
    except (OSError, ReportingError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
