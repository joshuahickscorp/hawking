#!/usr/bin/env python3.12
"""Read-only post-120B observer and final-interpretation handoff.

This process never launches model work and never mutates the Doctor V5 campaign,
queue, reporter, runtime specs, or result artifacts.  It snapshots reporter
generations while work is in flight, maintains a RAM-reservation-aware ETA to the
120B boundary, and emits a final interpretation packet only after both report
groups are terminal and their reporter checkpoints have been accepted by the
queue.
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
import statistics
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import doctor_v5_adapter_abi as adapter_abi
import doctor_v5_ultra_queue as queue_contract

ULTRA_ROOT = ROOT / "reports/condense/doctor_v5_ultra"
PLAN = ULTRA_ROOT / "campaign_plan.json"
CAMPAIGN = ULTRA_ROOT / "campaign.json"
REPORT_INDEX = ULTRA_ROOT / "reporting/report_index.json"
CHILD_RESOURCES = ULTRA_ROOT / "child_resources.jsonl"
REGISTRY = ULTRA_ROOT / "adapter_registry.json"
POST_ROOT = ULTRA_ROOT / "post_120b"
STATE = POST_ROOT / "observer_state.json"
SNAPSHOTS = POST_ROOT / "snapshots"
FINAL_PACKET = POST_ROOT / "final_interpretation_packet.json"
FINAL_HANDOFF = POST_ROOT / "CLAUDE_INTERPRETATION_HANDOFF.md"
FINAL_INPUTS = POST_ROOT / "final_inputs"
LOCK = POST_ROOT / "observer.lock"
GPTOSS_ADAPTER = ROOT / "tools/condense/doctor_v5_gptoss_moe_adapter.py"

VERSION = "2026-07-14.3"
PLAN_SCHEMA = "hawking.doctor_v5_ultra_campaign_plan.v1"
CAMPAIGN_SCHEMA = "hawking.doctor_v5_ultra_campaign.v1"
INDEX_SCHEMA = "hawking.doctor_v5_campaign_report_index.v1"
REPORT_CHECKPOINT_SCHEMA = "hawking.doctor_v5_ultra_report_checkpoint.v1"
TERMINAL = {"complete", "negative", "unsupported"}
PROCESS_BUDGET_BYTES = 78_000_000_000
SAFETY_MARGIN_BYTES = 28_000_000_000
MAX_LANES = 8
RESIDENCY_FLOOR_BYTES = 4_000_000_000
RESIDENCY_BASE_WORKING_BYTES = 9_000_000_000
RESIDENCY_DENSE_FACTOR = 2.0
RESIDENCY_SHARD_FACTOR = 2.0
MODEL_ORDER = ("0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B", "120B")

POST_120B_HORIZON = (
    {
        "model": "DeepSeek-V4-Flash",
        "nominal_parameters": "284B",
        "status": "scaffold_only",
        "admission_gate": "architecture adapter, pinned source inventory, streamed receipts",
    },
    {
        "model": "Kimi-K2.6",
        "nominal_parameters": "1.1T",
        "status": "scaffold_only",
        "admission_gate": "adapter plus guarded disk lifecycle after prior source retention closes",
    },
    {
        "model": "DeepSeek-V4-Pro",
        "nominal_parameters": "1.6T",
        "status": "remote_stream_scaffold_only",
        "admission_gate": "remote shard stream-transcode path; never full-installed",
    },
)


class ObserverError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: child for name, child in value.items() if name != key}


def _read_json_identity(path: Path) -> tuple[dict[str, Any], str, int]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObserverError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ObserverError(f"JSON root is not an object: {path}")
    return value, hashlib.sha256(raw).hexdigest(), len(raw)


def _read_json(path: Path) -> dict[str, Any]:
    return _read_json_identity(path)[0]


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n"
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _freeze_json_source(source: Path, target: Path,
                        expected: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = source.read_bytes()
        parsed = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ObserverError(f"cannot freeze JSON source {source}: {exc}") from exc
    if parsed != expected:
        raise ObserverError(f"JSON source changed while freezing: {source}")
    if target.exists():
        try:
            if target.read_bytes() != payload:
                raise ObserverError(f"refusing to replace different frozen input: {target}")
        except OSError as exc:
            raise ObserverError(f"cannot read frozen input {target}: {exc}") from exc
    else:
        _atomic_bytes(target, payload)
    return _artifact_reference(target, base=POST_ROOT)


def _validated_inputs() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], Path, dict[str, Any], bool,
]:
    plan = _read_json(PLAN)
    if plan.get("schema") != PLAN_SCHEMA \
            or plan.get("plan_sha256") != _hash_value(_without(plan, "plan_sha256")):
        raise ObserverError("campaign plan identity is invalid")
    plan_errors = queue_contract.validate_plan(plan, verify_sources=True)
    if plan_errors:
        raise ObserverError("campaign plan semantics are invalid: " + "; ".join(plan_errors))
    campaign, campaign_file_sha, campaign_file_bytes = _read_json_identity(CAMPAIGN)
    if campaign.get("schema") != CAMPAIGN_SCHEMA \
            or campaign.get("plan_sha256") != plan["plan_sha256"] \
            or campaign.get("campaign_sha256") \
            != _hash_value(_without(campaign, "campaign_sha256")):
        raise ObserverError("campaign projection identity is invalid")
    index = _read_json(REPORT_INDEX)
    if index.get("schema") != INDEX_SCHEMA \
            or index.get("index_sha256") != _hash_value(_without(index, "index_sha256")) \
            or index.get("campaign", {}).get("plan_sha256") != plan["plan_sha256"]:
        raise ObserverError("reporter index identity is invalid")
    raw_snapshot = index.get("snapshot_path")
    if not isinstance(raw_snapshot, str):
        raise ObserverError("reporter snapshot path is invalid")
    try:
        snapshot = Path(raw_snapshot).resolve(strict=True)
        snapshot.relative_to((ULTRA_ROOT / "reporting/snapshots").resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ObserverError("reporter snapshot escapes the immutable snapshot root") from exc
    if _parse_time(index.get("as_of")) is None:
        raise ObserverError("reporter observation time is invalid")
    if _parse_time(campaign.get("generated_at")) is None:
        raise ObserverError("campaign projection time is invalid")
    campaign_reference = {
        "path": str(CAMPAIGN.relative_to(ROOT)),
        "sha256": campaign_file_sha,
        "bytes": campaign_file_bytes,
    }
    reporter_campaign = index.get("campaign", {})
    try:
        reporter_campaign_path = Path(reporter_campaign.get("path", "")).resolve(
            strict=True
        )
    except (OSError, TypeError, ValueError):
        reporter_campaign_path = None
    reporter_aligned = (
        reporter_campaign_path == CAMPAIGN.resolve(strict=True)
        and reporter_campaign.get("sha256") == campaign_file_sha
        and reporter_campaign.get("bytes") == campaign_file_bytes
        and reporter_campaign.get("plan_sha256") == plan["plan_sha256"]
    )
    return (
        plan, campaign, index, snapshot, campaign_reference, reporter_aligned,
    )


def _artifact_reference(path: Path, *, base: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    resolved.relative_to(base.resolve(strict=True))
    digest, size = _hash_file(resolved)
    return {
        "path": str(resolved.relative_to(base.resolve(strict=True))),
        "sha256": digest,
        "bytes": size,
    }


def _json_artifact_reference(path: Path, *, base: Path,
                             expected: dict[str, Any]) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    resolved.relative_to(base.resolve(strict=True))
    value, digest, size = _read_json_identity(resolved)
    if value != expected:
        raise ObserverError(f"JSON artifact changed after validation: {path}")
    return {
        "path": str(resolved.relative_to(base.resolve(strict=True))),
        "sha256": digest,
        "bytes": size,
    }


def _report_references(index: dict[str, Any], snapshot: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in index.get("reports", []):
        if not isinstance(row, dict) or row.get("role") != "evidence_report":
            raise ObserverError("reporter evidence report reference is invalid")
        group = row.get("group_id")
        raw = row.get("path")
        if group not in {"sub-120B", "120B"} or not isinstance(raw, str):
            raise ObserverError("reporter group reference is invalid")
        if group in output:
            raise ObserverError("reporter group reference is duplicated")
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ObserverError("reporter group reference escapes snapshot")
        reference = _artifact_reference(snapshot / relative, base=ROOT)
        if reference["sha256"] != row.get("sha256") \
                or reference["bytes"] != row.get("bytes"):
            raise ObserverError("reporter group report identity changed")
        output[group] = {
            **reference,
            "complete": row.get("complete") is True,
            "kind": row.get("kind"),
        }
    if set(output) != {"sub-120B", "120B"}:
        raise ObserverError("both report groups are required")
    return output


def _gptoss_capabilities() -> tuple[dict[str, Any], dict[str, Any]]:
    before = _artifact_reference(GPTOSS_ADAPTER, base=ROOT)
    try:
        process = subprocess.run(
            [sys.executable, str(GPTOSS_ADAPTER), "capabilities"], cwd=ROOT,
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=30, check=False,
        )
        value = json.loads(process.stdout) if process.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        value = None
    after = _artifact_reference(GPTOSS_ADAPTER, base=ROOT)
    if before != after:
        value = None
    if not isinstance(value, dict):
        value = {
            "reviewed_for_live_campaign_execution": False,
            "blockers": [{"id": "capability-probe-unavailable-or-source-changed"}],
        }
    return value, after


def _gptoss_execution_readiness(plan: dict[str, Any], campaign: dict[str, Any],
                                capabilities: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    implemented = capabilities.get("implemented", {})
    required_flags = (
        "typed_spec_builder",
        "header_only_checkpoint_validation",
        "logical_parameter_accounting",
        "bounded_memory_per_expert_mxfp4_to_bf16",
        "source_read_only_byte_range_receipts",
        "full_str2_campaign_execution",
        "quality_evaluation",
        "apple_silicon_moe_runtime",
    )
    if capabilities.get("schema") != "hawking.doctor_v5_gptoss_moe_capabilities.v1" \
            or capabilities.get("model_family") != "gpt-oss-moe" \
            or capabilities.get("labels") != ["120B"] \
            or capabilities.get("rates") != [
                "4", "3", "2", "1", "0.8", "0.55", "0.5", "0.33", "0.25", "0.1"
            ]:
        blockers.append("GPT-OSS capability scope/schema is invalid")
    if capabilities.get("reviewed_for_live_campaign_execution") is not True:
        blockers.append("GPT-OSS capability receipt is not reviewed for live execution")
    if capabilities.get("blockers") != []:
        blockers.append("GPT-OSS capability receipt still declares unresolved blockers")
    for flag in required_flags:
        if not isinstance(implemented, dict) or implemented.get(flag) is not True:
            blockers.append(f"GPT-OSS capability is false: {flag}")

    registry_reference: dict[str, Any] | None = None
    reviewed_adapters: list[dict[str, Any]] = []
    try:
        registry = _read_json(REGISTRY)
        registry_errors = adapter_abi.validate_registry(
            registry, verify_files=True, base_dir=ROOT
        )
        if registry_errors:
            raise ObserverError(
                "adapter registry identity is invalid: " + "; ".join(registry_errors)
            )
        registry_reference = _json_artifact_reference(
            REGISTRY, base=ROOT, expected=registry
        )
    except (ObserverError, OSError, ValueError):
        registry = {"entries": []}
        blockers.append("adapter registry is missing or invalid")

    cells = [cell for cell in plan["cells"] if cell["model_label"] == "120B"]
    if len(cells) != 40 or len({cell["cell_id"] for cell in cells}) != 40 \
            or len({cell["runtime_spec_path"] for cell in cells}) != 40:
        blockers.append("GPT-OSS plan does not contain 40 unique cells and spec paths")
    expected = {
        (cell["adapter_id"], cell["command"], cell["backend"]) for cell in cells
    }
    expected_matrix = {
        (rate, branch) for rate in ("4", "3", "2", "1", "0.8", "0.55",
                                     "0.5", "0.33", "0.25", "0.1")
        for branch in ("codec_control", "doctor_static", "doctor_conditional",
                       "doctor_full")
    }
    if {(cell["rate_id"], cell["branch"]) for cell in cells} != expected_matrix:
        blockers.append("GPT-OSS plan rate/branch matrix is not the canonical 10x4 set")
    entries = registry.get("entries", []) if isinstance(registry, dict) else []
    for adapter_id, operation, backend in sorted(expected):
        entry = next((row for row in entries if isinstance(row, dict)
                      and row.get("adapter_id") == adapter_id
                      and operation in row.get("operations", [])), None)
        if entry is None or entry.get("reviewed") is not True \
                or entry.get("operations") != [operation] \
                or entry.get("model_families") != ["gpt-oss-moe"] \
                or entry.get("backends") != [backend]:
            blockers.append(f"reviewed GPT-OSS registry entry is absent: {adapter_id}")
            continue
        try:
            source = (ROOT / entry["source_path"]).resolve(strict=True)
            source.relative_to(ROOT.resolve(strict=True))
            source_reference = _artifact_reference(source, base=ROOT)
        except (KeyError, OSError, TypeError, ValueError):
            blockers.append(f"registry source is invalid: {adapter_id}")
            continue
        if source_reference["sha256"] != entry.get("source_sha256"):
            blockers.append(f"registry source hash changed: {adapter_id}")
            continue
        reviewed_adapters.append({
            "adapter_id": adapter_id,
            "operation": operation,
            "backend": backend,
            "source": source_reference,
        })

    expected_bindings = sorted((
        (row["adapter_id"], row["operation"], row["backend"],
         row["source"]["sha256"])
        for row in reviewed_adapters
    ))
    declared_bindings = capabilities.get("adapter_bindings")
    normalized_bindings = sorted((
        (row.get("adapter_id"), row.get("operation"), row.get("backend"),
         row.get("source_sha256"))
        for row in declared_bindings if isinstance(row, dict)
    )) if isinstance(declared_bindings, list) else []
    if len(expected_bindings) != 4 or normalized_bindings != expected_bindings:
        blockers.append("GPT-OSS capability receipt does not bind all four registry sources")

    valid_specs = 0
    runtime_spec_references: list[dict[str, Any]] = []
    runtime_scratch_bytes: dict[str, int] = {}
    for cell in cells:
        raw = cell.get("runtime_spec_path")
        try:
            relative = Path(raw)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("runtime spec path escapes workspace")
            path = (ROOT / relative).resolve(strict=True)
            path.relative_to(ROOT.resolve(strict=True))
            spec = _read_json(path)
        except (ObserverError, OSError, TypeError, ValueError):
            continue
        runtime, _, errors = queue_contract._validate_runtime_spec(
            cell, spec, path, verify_inputs=False
        )
        if not errors and runtime is not None:
            valid_specs += 1
            runtime_scratch_bytes[cell["cell_id"]] = runtime["resources"][
                "scratch_budget_bytes"
            ]
            runtime_spec_references.append({
                "cell_id": cell["cell_id"],
                **_json_artifact_reference(path, base=ROOT, expected=spec),
            })
    if valid_specs != len(cells) or len(cells) != 40:
        blockers.append(f"GPT-OSS typed runtime specs are {valid_specs}/40")

    adapter_sources = {
        row["adapter_id"]: row["source"]["sha256"] for row in reviewed_adapters
    }
    spec_hashes = {
        row["cell_id"]: row["sha256"] for row in runtime_spec_references
    }
    expected_preflights = sorted(((
        cell["cell_id"], cell["cell_identity_sha256"], cell["adapter_id"],
        cell["command"], cell["backend"], spec_hashes.get(cell["cell_id"]),
        adapter_sources.get(cell["adapter_id"]), registry.get("registry_sha256"),
    ) for cell in cells), key=lambda row: row[0])
    declared_preflights = capabilities.get("spec_preflights")
    normalized_preflights: list[tuple[Any, ...]] = []
    if isinstance(declared_preflights, list):
        for row in declared_preflights:
            if not isinstance(row, dict) \
                    or row.get("preflight_sha256") \
                    != _hash_value(_without(row, "preflight_sha256")) \
                    or row.get("blockers") != [] \
                    or row.get("execution_contract", {}).get(
                        "codec_execution_executable") is not True:
                continue
            normalized_preflights.append((
                row.get("cell_id"), row.get("cell_identity_sha256"),
                row.get("adapter_id"), row.get("operation"), row.get("backend"),
                row.get("runtime_spec_sha256"), row.get("adapter_source_sha256"),
                row.get("registry_sha256"),
            ))
    if len(expected_preflights) != 40 \
            or sorted(normalized_preflights, key=lambda row: row[0] or "") \
            != expected_preflights:
        blockers.append("GPT-OSS adapter preflight receipts are not valid for all 40 specs")

    try:
        disk = os.statvfs(ROOT)
        disk_free_bytes = disk.f_bavail * disk.f_frsize
    except OSError:
        disk_free_bytes = None
    maximum_required_free = max(
        cell["admission"]["disk_reserve_bytes"]
        + cell["admission"]["recommended_scratch_bytes"]
        + cell["projected_output_bytes"]
        for cell in cells
    )
    campaign_state = {
        cell["cell_id"]: cell for cell in campaign.get("cells", [])
    }
    running_120b = [
        cell for cell in cells
        if campaign_state.get(cell["cell_id"], {}).get("status") == "running"
    ]
    terminal_120b = [
        cell for cell in cells
        if campaign_state.get(cell["cell_id"], {}).get("status") in TERMINAL
    ]
    runnable_heads = [
        cell for cell in cells
        if campaign_state.get(cell["cell_id"], {}).get("status")
        not in TERMINAL | {"running"}
        and all(campaign_state.get(dependency, {}).get("status") == "complete"
                for dependency in cell["dependencies"])
    ]
    disk_heads: list[dict[str, Any]] = []
    for cell in sorted(runnable_heads, key=lambda row: row["priority"]):
        produced = queue_contract._current_output_bytes(cell)
        remaining = max(0, cell["projected_output_bytes"] - produced)
        scratch = runtime_scratch_bytes.get(
            cell["cell_id"], cell["admission"]["recommended_scratch_bytes"]
        )
        required = cell["admission"]["disk_reserve_bytes"] + scratch + remaining
        disk_heads.append({
            "cell_id": cell["cell_id"],
            "observed_current_output_bytes": produced,
            "projected_remaining_output_bytes": remaining,
            "scratch_bytes": scratch,
            "required_free_bytes": required,
            "fits_current_free_space": (
                disk_free_bytes is not None and disk_free_bytes >= required
            ),
        })
    if running_120b or len(terminal_120b) == len(cells):
        currently_disk_admissible: bool | None = True
        disk_admission_reason = "120B work is already running or terminal"
    elif disk_heads:
        currently_disk_admissible = any(
            row["fits_current_free_space"] for row in disk_heads
        )
        disk_admission_reason = (
            "at least one dependency-ready 120B head fits" if currently_disk_admissible
            else "no dependency-ready 120B head fits current free space"
        )
    else:
        currently_disk_admissible = None
        disk_admission_reason = "no 120B head is dependency-ready"

    return {
        "ready": not blockers,
        "structural_execution_ready": not blockers,
        "readiness_semantics": (
            "structurally ready for immediate worker prelaunch; source content hashes "
            "are reverified by the fail-closed worker before execution"
        ),
        "source_content_verification": "deferred_to_worker_prelaunch",
        "blockers": blockers,
        "required_registry_pairs": [
            {"adapter_id": adapter_id, "operation": operation, "backend": backend}
            for adapter_id, operation, backend in sorted(expected)
        ],
        "reviewed_registry_adapters": reviewed_adapters,
        "registry": registry_reference,
        "valid_runtime_specs": valid_specs,
        "expected_runtime_specs": 40,
        "runtime_specs": sorted(runtime_spec_references,
                                key=lambda row: row["cell_id"]),
        "validated_adapter_spec_preflights": len(normalized_preflights),
        "runtime_validator_source": _artifact_reference(
            Path(queue_contract.__file__).resolve(), base=ROOT
        ),
        "adapter_abi_source": _artifact_reference(
            Path(adapter_abi.__file__).resolve(), base=ROOT
        ),
        "disk_free_bytes": disk_free_bytes,
        "maximum_required_free_bytes": maximum_required_free,
        "currently_disk_admissible": currently_disk_admissible,
        "disk_admission_reason": disk_admission_reason,
        "dependency_ready_disk_heads": disk_heads,
    }


def _observed_tier_residency(plan: dict[str, Any]) -> dict[str, int]:
    labels = {cell["cell_id"]: cell["model_label"] for cell in plan["cells"]}
    output: dict[str, int] = {}
    try:
        handle = CHILD_RESOURCES.open("r", encoding="utf-8")
    except (OSError, UnicodeError):
        return output
    with handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) \
                    or row.get("plan_sha256") != plan["plan_sha256"]:
                continue
            label = labels.get(row.get("cell_id"))
            value = row.get("tree_rss_bytes")
            if label is not None and isinstance(value, int) \
                    and not isinstance(value, bool) and value > output.get(label, 0):
                output[label] = value
    return output


def _reservation(cell: dict[str, Any], observed: dict[str, int]) -> int:
    manifest = cell["parameter_manifest"]
    if cell["admission"]["whole_parent_residency_assumed"]:
        projected = RESIDENCY_BASE_WORKING_BYTES + math.ceil(
            manifest["source_weight_bytes"] * RESIDENCY_DENSE_FACTOR
        )
    else:
        projected = RESIDENCY_BASE_WORKING_BYTES + math.ceil(
            manifest["largest_source_shard_bytes"] * RESIDENCY_SHARD_FACTOR
        )
    ceiling = PROCESS_BUDGET_BYTES - SAFETY_MARGIN_BYTES
    projected = max(RESIDENCY_FLOOR_BYTES, min(ceiling, projected))
    return max(RESIDENCY_FLOOR_BYTES, min(ceiling, max(
        projected, observed.get(cell["model_label"], 0)
    )))


def _parse_time(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def _runtime_rates(campaign: dict[str, Any]) -> tuple[
    dict[str, float], dict[str, int], dict[str, float], dict[str, int],
]:
    branch_samples: dict[str, list[float]] = {}
    keyed_samples: dict[str, list[float]] = {}
    for cell in campaign["cells"]:
        if cell["status"] != "complete":
            continue
        started = _parse_time(cell.get("started_at"))
        completed = _parse_time(cell.get("completed_at"))
        parameters = cell.get("exact_stored_parameter_count")
        if started is None or completed is None or completed <= started \
                or not isinstance(parameters, int) or parameters <= 0:
            continue
        rate = (completed - started).total_seconds() / (parameters / 1_000_000_000)
        branch_samples.setdefault(cell["branch"], []).append(rate)
        key = f"{cell['branch']}@{cell['rate_id']}"
        keyed_samples.setdefault(key, []).append(rate)
    all_samples = [value for values in branch_samples.values() for value in values]
    fallback = statistics.median(all_samples) if all_samples else 900.0
    branches = {cell["branch"] for cell in campaign["cells"]}
    branch_rates = {
        branch: statistics.median(branch_samples[branch])
        if branch_samples.get(branch) else fallback
        for branch in branches
    }
    branch_counts = {
        branch: len(branch_samples.get(branch, [])) for branch in branches
    }
    keys = {f"{cell['branch']}@{cell['rate_id']}" for cell in campaign["cells"]}
    keyed_rates = {
        key: statistics.median(keyed_samples[key])
        if keyed_samples.get(key) else branch_rates[key.split("@", 1)[0]]
        for key in keys
    }
    keyed_counts = {key: len(keyed_samples.get(key, [])) for key in keys}
    return branch_rates, branch_counts, keyed_rates, keyed_counts


def _simulate_schedule(plan: dict[str, Any], campaign: dict[str, Any],
                       durations: dict[str, float], reservations: dict[str, int],
                       execution_120b_ready: bool, *,
                       force_large_tiers_to_budget_ceiling: bool = False) \
        -> tuple[float | None, float | None, str | None]:
    plan_cells = {cell["cell_id"]: cell for cell in plan["cells"]}
    state = {cell["cell_id"]: cell for cell in campaign["cells"]}
    allowed = {
        cell_id for cell_id, cell in plan_cells.items()
        if cell["model_label"] != "120B" or execution_120b_ready
    }
    if any(state[cell_id]["status"] == "blocked-execution" for cell_id in allowed):
        return None, None, "one or more admitted cells are blocked-execution"
    terminal_finished = {
        cell_id for cell_id in allowed if state[cell_id]["status"] in TERMINAL
    }
    dependency_complete = {
        cell_id for cell_id in allowed if state[cell_id]["status"] == "complete"
    }
    sub_cells = {
        cell_id for cell_id in allowed
        if plan_cells[cell_id]["model_label"] != "120B"
    }
    pending = {
        cell_id for cell_id in allowed
        if state[cell_id]["status"] not in TERMINAL | {"running"}
    }
    running: dict[str, tuple[float, int]] = {}
    for cell_id in allowed:
        if state[cell_id]["status"] != "running":
            continue
        cell = plan_cells[cell_id]
        reserve = reservations[cell["model_label"]]
        if force_large_tiers_to_budget_ceiling \
                and cell["model_label"] in {"32B", "72B"}:
            reserve = PROCESS_BUDGET_BYTES - SAFETY_MARGIN_BYTES
        running[cell_id] = (durations[cell_id], reserve)

    elapsed = 0.0
    sub_boundary = 0.0 if sub_cells <= terminal_finished else None
    ceiling = PROCESS_BUDGET_BYTES - SAFETY_MARGIN_BYTES
    while pending or running:
        used = sum(reserve for _, reserve in running.values())
        for cell_id in sorted(pending, key=lambda value: plan_cells[value]["priority"]):
            if len(running) >= MAX_LANES:
                break
            cell = plan_cells[cell_id]
            if not set(cell["dependencies"]) <= dependency_complete:
                continue
            reserve = reservations[cell["model_label"]]
            if force_large_tiers_to_budget_ceiling \
                    and cell["model_label"] in {"32B", "72B"}:
                reserve = ceiling
            if used + reserve > ceiling:
                continue
            running[cell_id] = (elapsed + durations[cell_id], reserve)
            pending.remove(cell_id)
            used += reserve
        if not running:
            return None, None, "dependency/resource simulation reached no runnable cell"
        elapsed = min(finish for finish, _ in running.values())
        finished = {
            cell_id for cell_id, (finish, _) in running.items()
            if finish <= elapsed + 1e-9
        }
        for cell_id in finished:
            del running[cell_id]
        terminal_finished.update(finished)
        dependency_complete.update(finished)
        if sub_boundary is None and sub_cells <= terminal_finished:
            sub_boundary = elapsed
    return sub_boundary, elapsed, None


def _eta_projection(plan: dict[str, Any], campaign: dict[str, Any],
                    execution_120b_ready: bool,
                    observation_time: dt.datetime) -> dict[str, Any]:
    branch_rates, branch_counts, keyed_rates, keyed_counts = _runtime_rates(campaign)
    observed = _observed_tier_residency(plan)
    state = {cell["cell_id"]: cell for cell in campaign["cells"]}
    now = observation_time
    tiers: list[dict[str, Any]] = []
    durations: dict[str, float] = {}
    reservations: dict[str, int] = {}
    for label in MODEL_ORDER:
        cells = [cell for cell in plan["cells"] if cell["model_label"] == label]
        representative = cells[0]
        reserve = _reservation(representative, observed)
        reservations[label] = reserve
        lanes = max(1, min(MAX_LANES,
            (PROCESS_BUDGET_BYTES - SAFETY_MARGIN_BYTES) // reserve))
        serial_remaining = 0.0
        for cell in cells:
            row = state[cell["cell_id"]]
            if row["status"] in TERMINAL:
                durations[cell["cell_id"]] = 0.0
                continue
            key = f"{cell['branch']}@{cell['rate_id']}"
            estimate = keyed_rates[key] * (
                cell["exact_stored_parameter_count"] / 1_000_000_000
            )
            if row["status"] == "running":
                active = campaign.get("active_children", {}).get(cell["cell_id"], {})
                started = _parse_time(active.get("started_at")) \
                    if isinstance(active, dict) else None
                if started is None:
                    started = _parse_time(row.get("started_at"))
                if started is not None:
                    elapsed = max(0.0, (now - started).total_seconds())
                    estimate = max(estimate * 0.1, estimate - elapsed)
            durations[cell["cell_id"]] = estimate
            serial_remaining += estimate
        wall = serial_remaining / lanes
        tiers.append({
            "model_label": label,
            "reservation_bytes": reserve,
            "budget_lanes": lanes,
            "serial_remaining_seconds": round(serial_remaining, 3),
            "projected_wall_seconds": round(wall, 3),
            "projection_semantics": (
                "homogeneous serial work divided by budget lanes; diagnostic only and "
                "not an additive list-schedule contribution"
            ),
            "projection_executable": label != "120B" or execution_120b_ready,
        })
    total_to_120b, total_all, simulation_blocker = _simulate_schedule(
        plan, campaign, durations, reservations, execution_120b_ready
    )
    conservative_to_120b, _, conservative_blocker = _simulate_schedule(
        plan, campaign, durations, reservations, execution_120b_ready,
        force_large_tiers_to_budget_ceiling=True,
    )
    all_branches_sampled = all(branch_counts.get(branch, 0) > 0 for branch in (
        "codec_control", "doctor_static", "doctor_conditional", "doctor_full"
    ))
    multiplier_low, multiplier_high = ((0.7, 1.6) if all_branches_sampled
                                       else (0.55, 2.25))
    if total_to_120b is None:
        return {
            "method": "dependency/priority list schedule using per-rate/branch sec/B and RAM reservations",
            "confidence": "blocked",
            "reason": simulation_blocker,
            "to_120b_boundary": {"point_seconds": None, "point_at": None},
            "through_120b": {"point_seconds": None, "point_at": None,
                             "reason": simulation_blocker},
        }
    arrival = now + dt.timedelta(seconds=total_to_120b)
    range_seconds = (total_to_120b * multiplier_low,
                     total_to_120b * multiplier_high)
    return {
        "method": "dependency/priority list schedule using per-rate/branch sec/B and RAM reservations",
        "confidence": "early" if not all_branches_sampled else "provisional",
        "confidence_semantics": "heuristic throughput projection; range is not a calibrated confidence interval",
        "branch_seconds_per_billion": {
            k: round(v, 6) for k, v in sorted(branch_rates.items())
        },
        "branch_complete_samples": dict(sorted(branch_counts.items())),
        "branch_rate_seconds_per_billion": {
            k: round(v, 6) for k, v in sorted(keyed_rates.items())
        },
        "branch_rate_complete_samples": dict(sorted(keyed_counts.items())),
        "tiers": tiers,
        "to_120b_boundary": {
            "point_seconds": round(total_to_120b, 3),
            "range_seconds": [round(value, 3) for value in range_seconds],
            "point_at": arrival.isoformat(timespec="seconds"),
            "range_at": [
                (now + dt.timedelta(seconds=value)).isoformat(timespec="seconds")
                for value in range_seconds
            ],
            "large_tier_50gb_reservation_sensitivity": ({
                "point_seconds": round(conservative_to_120b, 3),
                "point_at": (now + dt.timedelta(
                    seconds=conservative_to_120b
                )).isoformat(timespec="seconds"),
                "meaning": (
                    "worst-case projection with every 32B and 72B cell reserving the "
                    "full 50 GB campaign budget, which also prevents mixed-tier overlap"
                ),
            } if conservative_to_120b is not None else {
                "point_seconds": None,
                "point_at": None,
                "reason": conservative_blocker,
            }),
            "meaning": "projected completion of sub-120B work; not 120B execution readiness",
        },
        "through_120b": ({
            "point_seconds": round(total_all, 3),
            "point_at": (now + dt.timedelta(seconds=total_all)).isoformat(
                timespec="seconds"
            ),
        } if execution_120b_ready and total_all is not None else {
            "point_seconds": None,
            "point_at": None,
            "reason": simulation_blocker or "120B campaign is not fully wired/reviewed/executable",
        }),
    }


def _final_gate(plan: dict[str, Any], campaign: dict[str, Any],
                reports: dict[str, dict[str, Any]],
                checkpoints: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    planned = {cell["cell_id"]: cell for cell in plan.get("cells", [])}
    projected = {cell.get("cell_id"): cell for cell in campaign.get("cells", [])
                 if isinstance(cell, dict) and isinstance(cell.get("cell_id"), str)}
    if len(planned) != 320 or len(projected) != 320 or set(planned) != set(projected):
        blockers.append("campaign cell identities do not exactly cover the 320-cell plan")
    elif any(projected[cell_id].get("cell_identity_sha256")
             != planned[cell_id]["cell_identity_sha256"] for cell_id in planned):
        blockers.append("campaign cell identity hash differs from the plan")
    exact_terminal = sum(
        projected.get(cell_id, {}).get("status") in TERMINAL for cell_id in planned
    )
    counts = campaign.get("counts", {})
    terminal_count = sum(counts.get(status, 0) for status in TERMINAL)
    total = sum(counts.values()) if isinstance(counts, dict) else 0
    if total != 320 or terminal_count != 320 or exact_terminal != 320:
        blockers.append(f"campaign terminal coverage is {exact_terminal}/320")
    if campaign.get("queue_status") != "complete" \
            or campaign.get("active_cells") != [] \
            or campaign.get("active_children") != {}:
        blockers.append("queue is not quiescent and complete")
    groups = {row.get("group_id"): row for row in campaign.get("report_groups", [])}
    for group_id in ("sub-120B", "120B"):
        group = groups.get(group_id, {})
        if group.get("ready_for_verified_report") is not True:
            blockers.append(f"{group_id} queue/report coverage is not terminal")
        accepted = group.get("verified_report_checkpoint")
        checkpoint = checkpoints.get(group_id)
        if accepted is None:
            blockers.append(f"{group_id} checkpoint is not accepted by the queue")
        if reports.get(group_id, {}).get("complete") is not True:
            blockers.append(f"{group_id} reporter artifact is incomplete")
        if checkpoint is None:
            blockers.append(f"{group_id} reporter checkpoint is absent")
        elif accepted is not None and (
            accepted.get("path") != checkpoint.get("path")
            or accepted.get("file_sha256") != checkpoint.get("sha256")
            or accepted.get("checkpoint_sha256")
            != checkpoint.get("checkpoint_sha256")
        ):
            blockers.append(f"{group_id} accepted checkpoint identity does not match reporter")
    return not blockers, blockers


def _checkpoint_references(index: dict[str, Any], snapshot: Path,
                           campaign: dict[str, Any],
                           reports: dict[str, dict[str, Any]]) \
        -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in index.get("ultra_report_checkpoints", []):
        if not isinstance(row, dict) \
                or row.get("role") != "ultra_report_checkpoint":
            raise ObserverError("report checkpoint reference is invalid")
        raw = row.get("path") if isinstance(row, dict) else None
        if not isinstance(raw, str):
            raise ObserverError("report checkpoint reference is invalid")
        path = (snapshot / raw).resolve(strict=True)
        path.relative_to(snapshot)
        reference = _artifact_reference(path, base=ROOT)
        if reference["sha256"] != row.get("sha256") \
                or reference["bytes"] != row.get("bytes"):
            raise ObserverError("report checkpoint identity changed")
        doc = _read_json(path)
        group = doc.get("group_id")
        if group not in {"sub-120B", "120B"}:
            raise ObserverError("report checkpoint group identity is invalid")
        if group in output:
            raise ObserverError("report checkpoint group identity is duplicated")
        required = {
            "schema", "version", "plan_sha256", "group_id",
            "covered_cells_sha256", "report_artifact", "verified",
            "source_deletion_permitted", "checkpoint_sha256",
        }
        report = reports.get(group, {})
        expected_artifact = {
            "path": report.get("path"),
            "sha256": report.get("sha256"),
            "bytes": report.get("bytes"),
        }
        cells = [
            cell for cell in campaign["cells"]
            if (cell["model_label"] != "120B") == (group == "sub-120B")
        ]
        coverage = [{
            "cell_id": cell["cell_id"],
            "status": cell["status"],
            "result_sha256": cell.get("result_sha256"),
            "disposition_sha256": cell.get("disposition_sha256"),
        } for cell in cells]
        if set(doc) != required \
                or doc.get("schema") != REPORT_CHECKPOINT_SCHEMA \
                or doc.get("version") != campaign.get("version") \
                or doc.get("plan_sha256") != campaign.get("plan_sha256") \
                or doc.get("verified") is not True \
                or doc.get("source_deletion_permitted") is not False \
                or doc.get("checkpoint_sha256") \
                != _hash_value(_without(doc, "checkpoint_sha256")) \
                or doc.get("covered_cells_sha256") != _hash_value(coverage) \
                or doc.get("report_artifact") != expected_artifact:
            raise ObserverError(f"{group} report checkpoint contract is invalid")
        output[group] = {
            **reference,
            "checkpoint_sha256": doc["checkpoint_sha256"],
            "covered_cells_sha256": doc["covered_cells_sha256"],
            "report_artifact": expected_artifact,
        }
    return output


def _final_handoff(final_reference: dict[str, Any]) -> str:
    return f"""# Doctor V5 final interpretation handoff

This handoff was emitted only after all 320 campaign cells became terminal and
both reporter checkpoints were accepted by the queue. Treat every earlier
observation as provisional engineering evidence, not a final scientific claim.

## Mandatory input

- Final packet: `{final_reference['path']}`
- SHA-256: `{final_reference['sha256']}`
- Bytes: `{final_reference['bytes']}`

Resolve all paths relative to `{POST_ROOT}` unless the final packet says
otherwise. Verify every recorded byte count and SHA-256 before interpretation.

## Interpretation contract

1. Separate codec fidelity from Doctor treatment effects.
2. Compare physical all-in bpw, never nominal symbols alone.
3. Retain negative and unsupported outcomes; do not survivor-filter them.
4. Analyze scale, rate, branch, and their interactions with uncertainty.
5. Distinguish observed evidence from mechanistic hypotheses.
6. Do not claim sealed dominance from a preliminary one-seed matrix.
7. Report integrity or coverage failures before scientific conclusions.

Produce a concise executive result, a methods-and-integrity section, branch and
scale/rate comparisons, failure/negative-result analysis, limitations, and the
next highest-information experiments. Cite exact result or report artifact IDs
for every material claim.
"""


def _sync() -> dict[str, Any]:
    POST_ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ObserverError("another observer sync owns the lock") from exc
        (plan, campaign, index, reporter_snapshot, campaign_file_reference,
         reporter_aligned) = _validated_inputs()
        reports = _report_references(index, reporter_snapshot)
        checkpoint_references = _checkpoint_references(
            index, reporter_snapshot, campaign, reports
        )
        checkpoint_groups = set(checkpoint_references)
        capabilities, capability_source = _gptoss_capabilities()
        execution_readiness = _gptoss_execution_readiness(
            plan, campaign, capabilities
        )
        execution_ready = execution_readiness["ready"]
        final_ready, final_blockers = _final_gate(
            plan, campaign, reports, checkpoint_references
        )
        observer_source = _artifact_reference(Path(__file__).resolve(), base=ROOT)
        packet: dict[str, Any] = {
            "schema": "hawking.doctor_v5_post_120b_observation.v1",
            "version": VERSION,
            "observed_at": campaign.get("generated_at"),
            "reporter_observed_at": index.get("as_of"),
            "generation_id": index.get("generation_id"),
            "plan_sha256": plan["plan_sha256"],
            "campaign_sha256": campaign["campaign_sha256"],
            "campaign_file": campaign_file_reference,
            "reporter_campaign_aligned": reporter_aligned,
            "report_index_sha256": index["index_sha256"],
            "campaign_counts": campaign["counts"],
            "reporter_snapshot_path": str(reporter_snapshot.relative_to(ROOT)),
            "artifact_path_bases": {
                "workspace": str(ROOT),
                "observation": str(POST_ROOT),
            },
            "observer_source": observer_source,
            "reports": reports,
            "accepted_report_checkpoint_groups": sorted(checkpoint_groups),
            "accepted_report_checkpoints": checkpoint_references,
            "sampling_policy": {
                "provisional_sampling_allowed": True,
                "engineering_improvement_allowed": (
                    "safety/orchestration may improve between samples; completed evidence and "
                    "source-bound scientific recipes remain immutable"
                ),
                "final_scientific_interpretation_allowed": final_ready,
                "intermediate_observations_are_final_claims": False,
            },
            "final_gate": {"ready": final_ready, "blockers": final_blockers},
            "eta": _eta_projection(
                plan, campaign, execution_ready, _parse_time(campaign["generated_at"])
            ),
            "gpt_oss_120b": {
                "execution_ready": execution_ready,
                "execution_readiness": execution_readiness,
                "capabilities_schema": capabilities.get("schema"),
                "capability_source": capability_source,
                "implemented": capabilities.get("implemented", {}),
                "blockers": capabilities.get("blockers", []),
            },
            "post_120b_campaign_horizon": list(POST_120B_HORIZON),
            "source_deletion_permitted": False,
        }
        packet["packet_sha256"] = _hash_value(packet)
        generation = packet.get("generation_id")
        if not isinstance(generation, str) or not generation:
            raise ObserverError("report generation identity is missing")
        observation_id = packet["packet_sha256"]
        snapshot_path = SNAPSHOTS / f"{generation}-{observation_id[:12]}.json"
        existing = _read_json(snapshot_path) if snapshot_path.exists() else None
        if existing is not None and existing != packet:
            raise ObserverError("refusing to replace a different observation generation")
        if existing is None:
            _atomic_json(snapshot_path, packet)
        if final_ready:
            frozen_inputs = {
                "campaign_plan": _freeze_json_source(
                    PLAN, FINAL_INPUTS / "campaign_plan.json", plan
                ),
                "campaign": _freeze_json_source(
                    CAMPAIGN, FINAL_INPUTS / "campaign.json", campaign
                ),
                "report_index": _freeze_json_source(
                    REPORT_INDEX, FINAL_INPUTS / "report_index.json", index
                ),
            }
            final: dict[str, Any] = {
                "schema": "hawking.doctor_v5_final_interpretation_packet.v1",
                "version": VERSION,
                "ready": True,
                "created_at": index["as_of"],
                "plan_sha256": plan["plan_sha256"],
                "campaign_sha256": campaign["campaign_sha256"],
                "source_observation": _artifact_reference(snapshot_path, base=POST_ROOT),
                "artifact_path_bases": {
                    "source_observation": str(POST_ROOT),
                    "frozen_inputs": str(POST_ROOT),
                    "reports_and_checkpoints": str(ROOT),
                },
                "frozen_inputs": frozen_inputs,
                "reports": reports,
                "accepted_report_checkpoint_groups": sorted(checkpoint_groups),
                "accepted_report_checkpoints": checkpoint_references,
                "interpretation_requirements": [
                    "distinguish codec fidelity from Doctor treatment effects",
                    "retain negative and unsupported outcomes",
                    "compare physical all-in bpw rather than nominal symbols",
                    "report scale/rate/branch interactions and uncertainty",
                    "do not infer sealed dominance from this preliminary one-seed matrix",
                ],
                "source_deletion_permitted": False,
            }
            final["packet_sha256"] = _hash_value(final)
            prior_final = _read_json(FINAL_PACKET) if FINAL_PACKET.exists() else None
            if prior_final is not None and prior_final != final:
                raise ObserverError("refusing to replace a different final interpretation packet")
            if prior_final is None:
                _atomic_json(FINAL_PACKET, final)
            final_reference = _artifact_reference(FINAL_PACKET, base=POST_ROOT)
            handoff = _final_handoff(final_reference)
            if FINAL_HANDOFF.exists():
                try:
                    existing_handoff = FINAL_HANDOFF.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as exc:
                    raise ObserverError("cannot read final interpretation handoff") from exc
                if existing_handoff != handoff:
                    raise ObserverError("refusing to replace a different final handoff")
            else:
                FINAL_HANDOFF.parent.mkdir(parents=True, exist_ok=True)
                descriptor, raw = tempfile.mkstemp(
                    prefix=f".{FINAL_HANDOFF.name}.", dir=FINAL_HANDOFF.parent
                )
                temporary = Path(raw)
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                        handle.write(handoff)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, FINAL_HANDOFF)
                    _fsync_dir(FINAL_HANDOFF.parent)
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
        source_sha, source_bytes = _hash_file(Path(__file__).resolve())
        state: dict[str, Any] = {
            "schema": "hawking.doctor_v5_post_120b_observer_state.v1",
            "version": VERSION,
            "updated_at": _now(),
            "observer_source_sha256": source_sha,
            "observer_source_bytes": source_bytes,
            "latest_generation_id": generation,
            "latest_observation_id": observation_id,
            "latest_observation": _artifact_reference(snapshot_path, base=POST_ROOT),
            "final_interpretation_ready": final_ready,
            "final_interpretation_packet": (
                _artifact_reference(FINAL_PACKET, base=POST_ROOT) if final_ready else None
            ),
            "final_interpretation_handoff": (
                _artifact_reference(FINAL_HANDOFF, base=POST_ROOT) if final_ready else None
            ),
            "eta": packet["eta"],
            "gpt_oss_120b_execution_ready": execution_ready,
            "gpt_oss_120b_currently_disk_admissible": execution_readiness[
                "currently_disk_admissible"
            ],
            "final_gate_blockers": final_blockers,
            "source_deletion_permitted": False,
        }
        state["state_sha256"] = _hash_value(state)
        _atomic_json(STATE, state)
        return state


def _selftest() -> None:
    plan_cells = [{
        "cell_id": f"cell-{index:03d}",
        "cell_identity_sha256": hashlib.sha256(str(index).encode()).hexdigest(),
    } for index in range(320)]
    campaign_cells = [{
        **cell,
        "status": "complete" if index < 318 else (
            "negative" if index == 318 else "unsupported"
        ),
    } for index, cell in enumerate(plan_cells)]
    reports = {
        "sub-120B": {"complete": True},
        "120B": {"complete": True},
    }
    checkpoints = {
        "sub-120B": {"path": "sub.json", "sha256": "c" * 64,
                     "checkpoint_sha256": "a" * 64},
        "120B": {"path": "120.json", "sha256": "d" * 64,
                 "checkpoint_sha256": "b" * 64},
    }
    plan = {"cells": plan_cells}
    campaign = {
        "cells": campaign_cells,
        "counts": {"complete": 318, "negative": 1, "unsupported": 1},
        "queue_status": "complete",
        "active_cells": [],
        "active_children": {},
        "report_groups": [
            {"group_id": "sub-120B", "ready_for_verified_report": True,
             "verified_report_checkpoint": {
                 "path": "sub.json", "file_sha256": "c" * 64,
                 "checkpoint_sha256": "a" * 64,
             }},
            {"group_id": "120B", "ready_for_verified_report": True,
             "verified_report_checkpoint": {
                 "path": "120.json", "file_sha256": "d" * 64,
                 "checkpoint_sha256": "b" * 64,
             }},
        ],
    }
    ready, blockers = _final_gate(plan, campaign, reports, checkpoints)
    assert ready and blockers == []
    reports["120B"]["complete"] = False
    ready, blockers = _final_gate(plan, campaign, reports, checkpoints)
    assert not ready and "120B reporter artifact is incomplete" in blockers
    assert _hash_value({"b": 2, "a": 1}) == _hash_value({"a": 1, "b": 2})
    print("doctor_v5_post_120b.py selftest OK")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sync", "status", "selftest"))
    args = parser.parse_args(argv)
    try:
        if args.command == "selftest":
            _selftest()
            return 0
        if args.command == "sync":
            print(json.dumps(_sync(), indent=2, sort_keys=True))
            return 0
        print(json.dumps(_read_json(STATE), indent=2, sort_keys=True))
        return 0
    except ObserverError as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
