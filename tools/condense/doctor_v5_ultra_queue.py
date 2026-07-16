#!/usr/bin/env python3.12
"""Detached, fail-closed Doctor-v5 Ultra experiment campaign.

The immutable campaign is exactly eight fixed models by ten canonical physical
bit ceilings by four treatment branches (320 addressable cells).  A cell can
run only through a source-hashed reviewed adapter registry and an immutable,
typed runtime spec.  Missing or unsupported cells remain visible while the
scheduler continues scanning for other runnable work.

This controller never constructs shell commands, never deletes a parent model,
and never treats the absence of an implementation as an experimental result.
It owns only files below ``reports/condense/doctor_v5_ultra``.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import stat
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, IO


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import doctor_v5_adapter_abi as adapter_abi
import doctor_v5_parameter_manifest as parameter_manifest
import ram_scheduler
import training_ladder_v5


ULTRA_ROOT = ROOT / "reports" / "condense" / "doctor_v5_ultra"
PLAN = ULTRA_ROOT / "campaign_plan.json"
CAMPAIGN = ULTRA_ROOT / "campaign.json"
STATE = ULTRA_ROOT / "queue_state.json"
PID_FILE = ULTRA_ROOT / "queue.pid.json"
QUEUE_LOCK = ULTRA_ROOT / "queue.lock"
CONTROL = ULTRA_ROOT / "control.json"
EVENTS = ULTRA_ROOT / "events.jsonl"
LOG_FILE = ULTRA_ROOT / "queue.log"
REPORTER_SYNC_LOG = ULTRA_ROOT / "reporter_sync.jsonl"
CHILD_RESOURCE_LOG = ULTRA_ROOT / "child_resources.jsonl"
RUNTIME_SPECS = ULTRA_ROOT / "runtime_specs"
DISPOSITIONS = ULTRA_ROOT / "dispositions"
RESULTS = ULTRA_ROOT / "results"
REPORT_CHECKPOINTS = ULTRA_ROOT / "report_checkpoints"
LAUNCH_ARM = ULTRA_ROOT / "launch_armed.json"
LAUNCH_TRIGGER_STATE = ULTRA_ROOT / "launch_trigger_state.json"

LADDER_PATH = ROOT / "reports" / "condense" / "training_ladder_v5.json"
REGISTRY_PATH = ULTRA_ROOT / "adapter_registry.json"
PARAMETER_MANIFEST_ROOT = (
    ROOT / "reports" / "condense" / "doctor_v5_pass_b" / "parameter_manifests"
)
CENSUS_ROOT = ROOT / "reports" / "condense" / "doctor_v5_scale"
HEAVY_LOCK = ROOT / "reports" / "cron" / "studio_heavy.lock"
SCRIPT = Path(__file__).resolve()
STRAND_LADDER_ADAPTER = HERE / "doctor_v5_strand_ladder_adapter.py"
QWEN_TREATMENT_ADAPTER = HERE / "doctor_v5_qwen_treatment_adapter.py"
REPORTER = HERE / "doctor_v5_campaign_report.py"
RAW_3B_MARKER = (ROOT / "reports" / "condense" / "download_state"
                 / "Qwen-Qwen2.5-3B-Instruct.verified.json")
CANONICAL_3B_MARKER = ROOT / "reports" / "condense" / "download_state" / "3B.verified.json"
QWEN_3B_REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"

PLAN_SCHEMA = "hawking.doctor_v5_ultra_campaign_plan.v1"
CAMPAIGN_SCHEMA = "hawking.doctor_v5_ultra_campaign.v1"
STATE_SCHEMA = "hawking.doctor_v5_ultra_queue_state.v1"
PID_SCHEMA = "hawking.doctor_v5_ultra_queue_pid.v1"
CONTROL_SCHEMA = "hawking.doctor_v5_ultra_control.v1"
DISPOSITION_SCHEMA = "hawking.doctor_v5_ultra_disposition.v1"
REPORT_CHECKPOINT_SCHEMA = "hawking.doctor_v5_ultra_report_checkpoint.v1"
PACKED_GC_RECEIPT_SCHEMA = "hawking.doctor_v5_packed_gc_receipt.v2"
PACKED_GC_INTENT_SCHEMA = "hawking.doctor_v5_packed_gc_intent.v2"
VERSION = "2026-07-13.1"

CONTROL_POLL_SECONDS = 5.0
PROCESS_RSS_POLL_SECONDS = 5.0
RESOURCE_POLL_SECONDS = 30.0
PREREQUISITE_POLL_SECONDS = 30.0
REPORTER_SYNC_TIMEOUT_SECONDS = 300.0
DISK_RESERVE_BYTES = 150_000_000_000
MIN_SCRATCH_BYTES = 12_000_000_000
MAX_DECLARED_SCRATCH_BYTES = 140_000_000_000
PROCESS_BUDGET_BYTES = 78_000_000_000
# Concurrent-pool admission (RAM-budget-gated lanes).  The reservation sum plus
# this headroom must stay under PROCESS_BUDGET_BYTES for a new lane to admit.
# Leave room for the co-resident mop process plus the OS so the concurrent pool
# cannot over-commit into swap.
SAFETY_MARGIN_BYTES = 28_000_000_000
# Tolerate a small stale swap remnant; memory pressure is checked separately.
SWAP_TOLERANCE_MB = 1024
# Hard cap on concurrent children (bounds ps/fd cost).  Env-overridable; the
# 28 GB co-resident margin makes RAM reservation, rather than this ceiling, the
# usual binder (about four 0.5B lanes and fewer as model residency grows).
MAX_LANES = max(1, int(os.environ.get("DOCTOR_V5_MAX_LANES", "8")))
RESIDENCY_FLOOR_BYTES = 4_000_000_000            # Lower clamp on a reservation.
RESIDENCY_BASE_WORKING_BYTES = 9_000_000_000     # MEASURED-derived 0.5B floor.
RESIDENCY_DENSE_FACTOR = 2.0                     # INFERRED f32 dense of bf16.
RESIDENCY_SHARD_FACTOR = 2.0                     # INFERRED streaming shard window.
MAX_AUTOMATIC_ATTEMPTS = 3
# Consecutive resource-driven stops for ONE cell before it escalates to
# blocked-execution (so the MAX_AUTOMATIC_ATTEMPTS ceiling applies and the pool
# cannot re-admit + OOM-kill + reset the same cell forever).
MAX_RESOURCE_STOPS = 5
PAUSE_RC = 131
RESOURCE_RC = 75
ADOPT_RC = 132
TERMINAL = {"complete", "negative", "unsupported"}
REPORT_TERMINAL = frozenset(TERMINAL)
CELL_STATUSES = {
    "pending", "running", "complete", "negative", "unsupported",
    "blocked-dependency", "blocked-execution",
}
QUEUE_STATUSES = {
    "compiled", "running", "running-cell", "paused", "drained", "complete",
    "waiting-prerequisites", "waiting-resources", "waiting-heavy-lease",
    "blocked-state",
}
_STOP = False

# MEASURED-in-flight per-tier residency, keyed by model_label.  Seeded once from
# the persisted child-resource log at supervisor start and updated live from each
# tick's sampled tree RSS.  Turns INFERRED reservation projections into MEASURED
# self-calibration as soon as one cell of a tier runs, with no schema change.
_OBSERVED_TIER_RSS: dict[str, int] = {}


@dataclass
class LiveCell:
    """Handle for one concurrently running cell child in the RAM-budget pool."""
    execution: dict[str, Any]
    process: subprocess.Popen[Any]
    process_pgid: int
    spawn_identity: tuple[str, str]
    process_identity: tuple[str, str]
    lease: IO[Any]                 # Per-child HEAVY_LOCK fd, NOT flocked.
    reserved_bytes: int
    max_tree_rss_bytes: int
    last_resource_probe: float


COHORT = (
    {
        "label": "0.5B", "model_name": "qwen2.5-0.5b",
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct", "family": "qwen2.5-dense",
        "model_dir": "scratch/qwen-05b", "nominal_params_b": 0.5,
    },
    {
        "label": "1.5B", "model_name": "qwen2.5-1.5b",
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct", "family": "qwen2.5-dense",
        "model_dir": "scratch/qwen-15b", "nominal_params_b": 1.5,
    },
    {
        "label": "3B", "model_name": "qwen2.5-3b",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct", "family": "qwen2.5-dense",
        "model_dir": "scratch/qwen-3b", "nominal_params_b": 3.0,
    },
    {
        "label": "7B", "model_name": "qwen2.5-7b",
        "hf_id": "Qwen/Qwen2.5-7B-Instruct", "family": "qwen2.5-dense",
        "model_dir": "scratch/qwen-7b", "nominal_params_b": 7.0,
    },
    {
        "label": "14B", "model_name": "qwen2.5-14b",
        "hf_id": "Qwen/Qwen2.5-14B-Instruct", "family": "qwen2.5-dense",
        "model_dir": "scratch/staging/qwen-14b.partial", "nominal_params_b": 14.0,
    },
    {
        "label": "32B", "model_name": "qwen2.5-32b",
        "hf_id": "Qwen/Qwen2.5-32B-Instruct", "family": "qwen2.5-dense",
        "model_dir": "scratch/staging/qwen-32b.partial", "nominal_params_b": 32.0,
    },
    {
        "label": "72B", "model_name": "qwen2.5-72b",
        "hf_id": "Qwen/Qwen2.5-72B-Instruct", "family": "qwen2.5-dense",
        "model_dir": "scratch/staging/qwen-72b.partial", "nominal_params_b": 72.0,
    },
    {
        "label": "120B", "model_name": "gpt-oss-120b",
        "hf_id": "openai/gpt-oss-120b", "family": "gpt-oss-moe",
        "model_dir": "scratch/staging/gpt-oss-120b.partial", "nominal_params_b": 116.8,
    },
)

RATES = (
    {"rate_id": "4", "rate_bpw": 4.0, "numerator": 4, "denominator": 1},
    {"rate_id": "3", "rate_bpw": 3.0, "numerator": 3, "denominator": 1},
    {"rate_id": "2", "rate_bpw": 2.0, "numerator": 2, "denominator": 1},
    {"rate_id": "1", "rate_bpw": 1.0, "numerator": 1, "denominator": 1},
    {"rate_id": "0.8", "rate_bpw": 0.8, "numerator": 4, "denominator": 5},
    {"rate_id": "0.55", "rate_bpw": 0.55, "numerator": 11, "denominator": 20},
    {"rate_id": "0.5", "rate_bpw": 0.5, "numerator": 1, "denominator": 2},
    {"rate_id": "0.33", "rate_bpw": 0.33, "numerator": 33, "denominator": 100},
    {"rate_id": "0.25", "rate_bpw": 0.25, "numerator": 1, "denominator": 4},
    {"rate_id": "0.1", "rate_bpw": 0.1, "numerator": 1, "denominator": 10},
)

BRANCHES = (
    {
        "branch": "codec_control", "claim_scope": "codec_fidelity",
        "operation": "condense_control", "dependencies": (),
        "adapter": {
            "qwen2.5-dense": "doctor-v5-strand-ladder-qwen25-dense",
            "gpt-oss-moe": "doctor-v5-strand-ladder-gpt-oss-moe",
        },
        "runtime_spec_schema": "hawking.doctor_v5_strand_ladder_spec.v1",
    },
    {
        "branch": "doctor_static",
        "claim_scope": "codec_repair_reencoding_experiment",
        "operation": "doctor_static", "dependencies": ("codec_control",),
        "adapter": {
            "qwen2.5-dense": "doctor-v5-static-repair",
            "gpt-oss-moe": "doctor-v5-gpt-oss-static-repair",
        },
        "runtime_spec_schema": "hawking.doctor_v5_static_spec.v1",
    },
    {
        "branch": "doctor_conditional",
        "claim_scope": "codec_repair_reencoding_experiment",
        "operation": "doctor_conditional",
        "dependencies": ("codec_control", "doctor_static"),
        "adapter": {
            "qwen2.5-dense": "doctor-v5-conditional-repair",
            "gpt-oss-moe": "doctor-v5-gpt-oss-conditional-repair",
        },
        "runtime_spec_schema": "hawking.doctor_v5_conditional_spec.v1",
    },
    {
        "branch": "doctor_full",
        "claim_scope": "codec_repair_reencoding_experiment",
        "operation": "doctor_full",
        "dependencies": ("codec_control", "doctor_static", "doctor_conditional"),
        "adapter": {
            "qwen2.5-dense": "doctor-v5-full-treatment",
            "gpt-oss-moe": "doctor-v5-gpt-oss-full-treatment",
        },
        "runtime_spec_schema": "hawking.doctor_v5_full_spec.v1",
    },
)

SEED_PLAN = (20260713, 20260717, 20260719, 20260723, 20260729)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
NONCE_RE = re.compile(r"[0-9a-f]{32}")
CELL_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{8,180}")


class CampaignError(RuntimeError):
    """Campaign state or an execution contract is invalid."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _hash_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _without(value: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: row for name, row in value.items() if name != key}


def _duplicate_safe_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        if path.stat().st_size > adapter_abi.MAX_JSON_BYTES:
            raise ValueError(f"JSON is too large: {path}")
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_duplicate_safe_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return default


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _append_event(kind: str, **payload: Any) -> None:
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    row = {"schema": "hawking.doctor_v5_ultra_event.v1", "at": _now(),
           "kind": kind, **payload}
    row["event_sha256"] = _hash_value(row)
    with EVENTS.open("a", encoding="utf-8") as handle:
        handle.write(_canonical(row).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_dir(EVENTS.parent)


def _sha_file(path: Path) -> tuple[str, int]:
    if path.is_symlink():
        raise CampaignError(f"symlink input is forbidden: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    digest, total = hashlib.sha256(), 0
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise CampaignError(f"not a regular file: {path}")
        while True:
            block = os.read(fd, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
        after = os.fstat(fd)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or total != after.st_size):
            raise CampaignError(f"file changed while hashing: {path}")
    finally:
        os.close(fd)
    return digest.hexdigest(), total


def _relative(path: Path) -> str:
    return str(path.resolve(strict=False).relative_to(ROOT.resolve()))


def _resolve_workspace_path(raw: Any, *, must_exist: bool = True) -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise CampaignError("path must be a nonempty string")
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise CampaignError(f"path must be a safe workspace-relative path: {raw!r}")
    resolved = (ROOT / path).resolve(strict=must_exist)
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise CampaignError(f"path escapes workspace: {raw!r}") from exc
    return resolved


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _cell_id(model_name: str, rate_id: str, branch: str) -> str:
    rate_token = rate_id.replace(".", "p")
    return f"{_slug(model_name)}__{rate_token}bpw__{branch.replace('_', '-')}"


def _manifest_binding(item: dict[str, Any]) -> dict[str, Any]:
    path = PARAMETER_MANIFEST_ROOT / f"{item['label']}.json"
    doc = _read_json(path)
    if not isinstance(doc, dict):
        raise CampaignError(f"parameter manifest missing: {path}")
    errors = parameter_manifest.validate_manifest(doc, verify_files=False)
    if errors:
        raise CampaignError(f"parameter manifest invalid for {item['label']}: "
                            + "; ".join(errors))
    if doc.get("label") != item["label"] or doc.get("hf_id") != item["hf_id"]:
        raise CampaignError(f"parameter manifest identity mismatch for {item['label']}")
    authority = doc.get("parameter_authority", {})
    exact = authority.get("exact_distinct_stored_parameter_count")
    if isinstance(exact, bool) or not isinstance(exact, int) or exact <= 0:
        raise CampaignError(f"exact parameter count missing for {item['label']}")
    active_moe = authority.get("active_moe_parameter_count")
    authority_evidence: dict[str, Any] | None = None
    if item["label"] == "120B":
        review = doc.get("review_boundary")
        if exact != 116_829_156_672 \
                or active_moe != 5_132_852_352 \
                or authority.get("stored_parameter_count") != exact \
                or authority.get("authoritative_for_physical_bpw_denominator") is not True \
                or authority.get("authoritative_for_active_moe_compute_denominator") is not True \
                or not isinstance(authority.get("counting_unit"), str) \
                or "logical_model_weights" not in authority["counting_unit"] \
                or not isinstance(review, dict) \
                or review.get("serialized_safetensors_element_count") != 63_081_444_672 \
                or review.get("serialized_element_count_is_parameter_denominator") is not False \
                or review.get("ue8_scales_are_side_information") is not True:
            raise CampaignError("120B logical/active parameter authority is invalid")
        inventory_raw = review.get("inventory_path")
        if not isinstance(inventory_raw, str):
            raise CampaignError("120B parameter authority inventory path is missing")
        inventory_path = Path(inventory_raw).resolve(strict=True)
        inventory_path.relative_to(ROOT.resolve())
        inventory = _read_json(inventory_path)
        inventory_sha, inventory_bytes = _sha_file(inventory_path)
        accounting = inventory.get("parameter_accounting") \
            if isinstance(inventory, dict) else None
        if not isinstance(inventory, dict) \
                or inventory.get("schema") \
                != "hawking.doctor_v5_gptoss_mxfp4_inventory.v1" \
                or inventory.get("inventory_sha256") \
                != _hash_value(_without(inventory, "inventory_sha256")) \
                or inventory.get("inventory_sha256") != review.get("inventory_sha256") \
                or inventory_sha != review.get("inventory_file_sha256") \
                or inventory_bytes != review.get("inventory_file_bytes") \
                or not isinstance(accounting, dict) \
                or accounting.get("logical_model_parameters") != exact \
                or accounting.get("active_compute_parameter_equivalent") != active_moe \
                or accounting.get("serialized_safetensors_elements_not_a_parameter_denominator") \
                != 63_081_444_672:
            raise CampaignError("120B MXFP4 logical parameter inventory is invalid")
        authority_evidence = {
            "path": _relative(inventory_path), "sha256": inventory_sha,
            "bytes": inventory_bytes,
            "inventory_sha256": inventory["inventory_sha256"],
        }
    digest, size = _sha_file(path)
    return {
        "path": _relative(path), "file_sha256": digest, "bytes": size,
        "manifest_sha256": doc.get("manifest_sha256"),
        "exact_stored_parameter_count": exact,
        "active_moe_parameter_count": active_moe,
        "parameter_counting_unit": authority.get("counting_unit"),
        "parameter_authority_evidence": authority_evidence,
        "source_manifest_sha256": doc.get("source_manifest_sha256"),
        "physical_bpw_denominator_authoritative": authority.get(
            "authoritative_for_physical_bpw_denominator"
        ) is True,
        "source_weight_bytes": sum(
            row.get("bytes", 0) for row in doc.get("source_shards", [])
            if isinstance(row, dict)
        ),
        "largest_source_shard_bytes": max(
            (row.get("bytes", 0) for row in doc.get("source_shards", [])
             if isinstance(row, dict)), default=0
        ),
    }


def _census_binding(item: dict[str, Any]) -> dict[str, Any]:
    path = CENSUS_ROOT / item["label"] / "census.json"
    doc = _read_json(path)
    if not isinstance(doc, dict) or not isinstance(doc.get("report_sha256"), str):
        raise CampaignError(f"source census missing or invalid for {item['label']}")
    digest, size = _sha_file(path)
    return {"path": _relative(path), "file_sha256": digest, "bytes": size,
            "report_sha256": doc["report_sha256"]}


def _scratch_recommendation(item: dict[str, Any], manifest: dict[str, Any]) -> int:
    exact = manifest["exact_stored_parameter_count"]
    shard_window = 3 * int(manifest["largest_source_shard_bytes"])
    if item["nominal_params_b"] > 16:
        # The deferred 32B/72B path never materializes a dense reconstruction.
        # Durable packed output is admitted independently as projected_output_bytes;
        # charging it again as scratch creates a circular disk gate at 72B.
        return min(MAX_DECLARED_SCRATCH_BYTES,
                   max(MIN_SCRATCH_BYTES, shard_window))
    if item["nominal_params_b"] <= 1.5:
        base = 16_000_000_000
    elif item["nominal_params_b"] <= 7:
        base = 32_000_000_000
    else:
        base = 48_000_000_000
    # One dense shard in, one ephemeral reconstruction out, and packed output.
    four_bit_payload = math.ceil(exact * 4 / 8)
    return min(MAX_DECLARED_SCRATCH_BYTES, max(base, shard_window + four_bit_payload))


def _compile_plan(
    manifest_provider: Callable[[dict[str, Any]], dict[str, Any]] = _manifest_binding,
    census_provider: Callable[[dict[str, Any]], dict[str, Any]] = _census_binding,
) -> dict[str, Any]:
    ladder = _read_json(LADDER_PATH)
    if not isinstance(ladder, dict):
        raise CampaignError("training ladder v5 artifact is missing")
    errors = training_ladder_v5.validate_ladder(ladder)
    if errors:
        raise CampaignError("training ladder v5 is invalid: " + "; ".join(errors))
    ladder_sha, ladder_bytes = _sha_file(LADDER_PATH)
    script_sha, script_bytes = _sha_file(SCRIPT)
    abi_sha, abi_bytes = _sha_file(Path(adapter_abi.__file__).resolve())
    scheduler_sha, scheduler_bytes = _sha_file(Path(ram_scheduler.__file__).resolve())
    reporter_sha, reporter_bytes = _sha_file(REPORTER)
    manifest_module_sha, manifest_module_bytes = _sha_file(
        Path(parameter_manifest.__file__).resolve()
    )

    cells: list[dict[str, Any]] = []
    priority = 0
    for item in COHORT:
        manifest = manifest_provider(item)
        census = census_provider(item)
        scratch = _scratch_recommendation(item, manifest)
        for rate in RATES:
            by_branch: dict[str, str] = {}
            for branch in BRANCHES:
                cell_id = _cell_id(item["model_name"], rate["rate_id"], branch["branch"])
                dependencies = [by_branch[name] for name in branch["dependencies"]]
                adapter_id = branch["adapter"][item["family"]]
                nominal_payload = math.ceil(
                    manifest["exact_stored_parameter_count"]
                    * rate["numerator"] / rate["denominator"] / 8
                )
                projected_output = min(
                    manifest["source_weight_bytes"],
                    math.ceil(nominal_payload * 1.10) + 1_000_000_000,
                )
                runtime_spec = RUNTIME_SPECS / f"{cell_id}.json"
                disposition = DISPOSITIONS / f"{cell_id}.json"
                identity = {
                    "model_label": item["label"], "model_name": item["model_name"],
                    "hf_id": item["hf_id"], "model_family": item["family"],
                    "parameter_manifest_sha256": manifest["file_sha256"],
                    "exact_stored_parameter_count": manifest[
                        "exact_stored_parameter_count"
                    ],
                    "rate_id": rate["rate_id"], "rate_bpw": rate["rate_bpw"],
                    "rate_fraction": {
                        "numerator": rate["numerator"],
                        "denominator": rate["denominator"],
                    },
                    "branch": branch["branch"], "claim_scope": branch["claim_scope"],
                    "adapter_id": adapter_id, "command": branch["operation"],
                    "dependencies": dependencies,
                    "seed_plan": list(SEED_PLAN),
                }
                cell_identity_sha256 = _hash_value(identity)
                cell = {
                    "cell_id": cell_id, "cell_identity_sha256": cell_identity_sha256,
                    "priority": priority, "model_label": item["label"],
                    "model_name": item["model_name"], "hf_id": item["hf_id"],
                    "model_family": item["family"], "model_dir": item["model_dir"],
                    "nominal_params_b": item["nominal_params_b"],
                    "exact_stored_parameter_count": manifest[
                        "exact_stored_parameter_count"
                    ],
                    "parameter_manifest": copy.deepcopy(manifest),
                    "source_census": copy.deepcopy(census),
                    "rate_id": rate["rate_id"], "rate_bpw": rate["rate_bpw"],
                    "rate_fraction": identity["rate_fraction"],
                    "nominal_payload_bytes": nominal_payload,
                    "projected_output_bytes": projected_output,
                    "nominal_payload_semantics": (
                        "projection_only_planning_number; measured whole-artifact bytes govern"
                    ),
                    "branch": branch["branch"], "claim_scope": branch["claim_scope"],
                    "adapter_id": adapter_id, "command": branch["operation"],
                    "backend": (
                        "apple-cpu-strand" if item["family"] == "qwen2.5-dense"
                        else "apple-silicon-doctor-v5"
                    ),
                    "runtime_spec_path": _relative(runtime_spec),
                    "runtime_spec_schema": branch["runtime_spec_schema"],
                    "disposition_path": _relative(disposition),
                    "dependencies": dependencies, "seed_plan": list(SEED_PLAN),
                    "expected_replicates": 1,
                    "replicate_scope": "preliminary_scale_mapping_not_dominance",
                    "admission": {
                        "disk_reserve_bytes": DISK_RESERVE_BYTES,
                        "recommended_scratch_bytes": scratch,
                        "maximum_declared_scratch_bytes": MAX_DECLARED_SCRATCH_BYTES,
                        "process_budget_bytes": PROCESS_BUDGET_BYTES,
                        "normal_memory_pressure_required": True,
                        "zero_swap_required": True, "ac_power_required": True,
                        "nominal_thermal_required": True,
                        "streaming_required": item["nominal_params_b"] > 16,
                        "whole_parent_residency_assumed": item["nominal_params_b"] <= 16,
                    },
                    "capability": {
                        "status": "awaiting_typed_runtime_spec_and_reviewed_adapter",
                        "unsupported_must_be_a_hashed_disposition": True,
                        "absence_is_not_a_negative_result": True,
                        "quality_claims_permitted_by_execution_receipt": False,
                        "treatment_is_model_training": False,
                        "conditional_activation_dispatch": (
                            "explicit_negative_proxy_not_implemented"
                            if branch["branch"] == "doctor_conditional" else "not_claimed"
                        ),
                        "quality_observation_expected": (
                            "null" if item["label"] in {"32B", "72B"}
                            else "provisional_measurement_or_explicit_null"
                        ),
                        "quality_null_reason": (
                            "dense_reconstruction_and_resident_quality_eval_deferred_by_96GB_gate"
                            if item["label"] in {"32B", "72B"} else None
                        ),
                    },
                    "lifecycle": {
                        "dense_reconstruction": "ephemeral_hash_and_metrics_only",
                        "packed_base": (
                            "retain_until_self_is_reporter_sealed_and_immediate_"
                            "successor_runtime_spec_is_bound"
                        ),
                        "candidate_gc": "automatic_exact_allowlist_with_durable_receipt",
                        "automatic_payload_gc_scope": (
                            "bundle_shard_colon_star_dot_strand_only; lossless passthrough "
                            "and all evidence remain retained"
                        ),
                        "full_candidate": (
                            "retain_until_before_next_rate_codec_admission; "
                            "final_0.1bpw_full_retained"
                        ),
                        "parent_source_cleanup": "disabled_separate_operator_action_only",
                    },
                    "quality_claims_permitted": False,
                    "source_deletion_permitted": False, "status": "planned",
                }
                cell["cell_spec_sha256"] = _hash_value(cell)
                cells.append(cell)
                by_branch[branch["branch"]] = cell_id
                priority += 1

    plan: dict[str, Any] = {
        "schema": PLAN_SCHEMA, "version": VERSION, "created_at": _now(),
        "matrix": {
            "models": len(COHORT), "rates": len(RATES),
            "branches": len(BRANCHES), "cells": len(cells),
            "formula": "8 models * 10 physical ceilings * 4 treatment branches",
        },
        "cohort": [dict(row) for row in COHORT],
        "rates": [dict(row) for row in RATES],
        "branches": [
            {key: (list(value) if isinstance(value, tuple) else value)
             for key, value in row.items() if key != "adapter"}
            | {"adapter": dict(row["adapter"])}
            for row in BRANCHES
        ],
        "seed_policy": {
            "minimum_independent_seeds": 5, "seeds": list(SEED_PLAN),
            "first_pass_expected_replicates": 1,
            "first_pass_replicate_scope": "preliminary_scale_mapping_not_dominance",
            "five_seed_scope": "future_sealed_proof_rerun",
            "runtime_may_checkpoint_seeds_but_may_not_relabel_repeats_as_independent": True,
        },
        "report_groups": [
            {
                "group_id": "sub-120B", "models": [row["label"] for row in COHORT[:-1]],
                "expected_cells": 280,
                "accepted_terminal_statuses": sorted(REPORT_TERMINAL),
            },
            {
                "group_id": "120B", "models": [COHORT[-1]["label"]],
                "expected_cells": 40,
                "accepted_terminal_statuses": sorted(REPORT_TERMINAL),
            },
        ],
        "execution_policy": {
            "detached": True, "automatic_progression": True,
            "skip_blocked_and_continue_runnable": True,
            "typed_argv_only": True, "shell": False,
            "reviewed_source_hashed_adapter_required": True,
            "immutable_runtime_spec_snapshot_per_cell": True,
            "exact_resume_checkpoint_required": True,
            "shared_heavy_lease_required": True,
            "wait_on_resource_gate_failure": True,
            "control_poll_seconds": CONTROL_POLL_SECONDS,
            "resource_probe_seconds_while_child_runs": RESOURCE_POLL_SECONDS,
            "prerequisite_rescan_seconds": PREREQUISITE_POLL_SECONDS,
            "immediate_prelaunch_resource_gate": True,
            "automatic_reporter_sync": ["queue_start_or_resume", "every_terminal_cell"],
            "reporter_failure_policy": "log_and_continue_without_state_corruption",
            "unbounded_wall_clock": True,
            "source_deletion_permitted": False,
        },
        "source_acquisition_policy": {
            "campaign_launch_requires_all_source_censuses_and_parameter_manifests": True,
            "download_completion_triggers": [
                "verify pinned repository snapshot",
                "hashing safetensors census with exact-resume checkpoint",
                "seal exact stored-parameter manifest",
                "compile and wire typed campaign cells",
            ],
            "download_and_heavy_experiment_overlap": False,
            "prefetch_requires_disk_reserve_bytes": DISK_RESERVE_BYTES,
            "current_campaign_source_horizon": "0.5B_through_120B_inclusive",
            "post_120B_models": "separate_per_model_report_and_admission_campaign",
            "source_deletion_permitted": False,
        },
        "lifecycle_policy": {
            "current_disk_design_point_bytes": 296_000_000_000,
            "immutable_disk_reserve_bytes": DISK_RESERVE_BYTES,
            "dense_reconstructions_are_ephemeral": True,
            "preserve": ["receipts", "hashes", "metrics", "manifests", "logs"],
            "packed_payload_gc_requires_target_reporter_checkpoint": True,
            "packed_payload_gc_requires_bound_successor_runtime_spec": True,
            "packed_payload_gc_occurs_before_successor_resource_admission": True,
            "packed_payload_gc_is_automatic": True,
            "packed_payload_gc_allowlist": "metrics.packed_artifact_inventory.artifacts",
            "rate_chain": [
                "before static admission -> delete reporter-sealed codec payload",
                "before conditional admission -> delete reporter-sealed static payload",
                "before full admission -> delete reporter-sealed conditional payload",
                "before next-rate codec admission -> delete reporter-sealed prior-rate full payload",
                "retain final 0.1bpw full payload per model",
            ],
            "parent_source_cleanup_enabled": False,
            "parent_source_cleanup_requires_separate_operator_authority": True,
        },
        "sources": {
            "orchestrator": {"path": _relative(SCRIPT), "sha256": script_sha,
                             "bytes": script_bytes},
            "adapter_abi": {"path": _relative(Path(adapter_abi.__file__).resolve()),
                            "sha256": abi_sha, "bytes": abi_bytes},
            "ram_scheduler": {"path": _relative(Path(ram_scheduler.__file__).resolve()),
                              "sha256": scheduler_sha, "bytes": scheduler_bytes},
            "campaign_reporter": {"path": _relative(REPORTER),
                                  "sha256": reporter_sha, "bytes": reporter_bytes},
            "parameter_manifest_contract": {
                "path": _relative(Path(parameter_manifest.__file__).resolve()),
                "sha256": manifest_module_sha, "bytes": manifest_module_bytes,
            },
            "training_ladder": {"path": _relative(LADDER_PATH), "sha256": ladder_sha,
                                "bytes": ladder_bytes,
                                "ladder_sha256": ladder.get("ladder_sha256")},
            "adapter_registry_path": _relative(REGISTRY_PATH),
        },
        "cells": cells,
    }
    plan["plan_sha256"] = _hash_value(plan)
    validation = validate_plan(plan, verify_sources=False)
    if validation:
        raise CampaignError("compiler emitted invalid plan: " + "; ".join(validation))
    return plan


def validate_plan(plan: Any, *, verify_sources: bool = True) -> list[str]:
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["plan is not an object"]
    if plan.get("schema") != PLAN_SCHEMA or plan.get("version") != VERSION:
        errors.append("plan schema/version mismatch")
    if plan.get("plan_sha256") != _hash_value(_without(plan, "plan_sha256")):
        errors.append("plan hash mismatch")
    cells = plan.get("cells")
    if not isinstance(cells, list) or len(cells) != 320:
        return errors + ["plan must contain exactly 320 cells"]
    expected = {
        (_cell_id(model["model_name"], rate["rate_id"], branch["branch"]),
         model["label"], rate["rate_id"], branch["branch"])
        for model in COHORT for rate in RATES for branch in BRANCHES
    }
    observed: set[tuple[str, str, str, str]] = set()
    by_id: dict[str, dict[str, Any]] = {}
    model_by_label = {row["label"]: row for row in COHORT}
    rate_by_id = {row["rate_id"]: row for row in RATES}
    branch_by_id = {row["branch"]: row for row in BRANCHES}
    for index, cell in enumerate(cells):
        prefix = f"cells[{index}]"
        if not isinstance(cell, dict):
            errors.append(f"{prefix} is not an object")
            continue
        cell_id = cell.get("cell_id")
        if not isinstance(cell_id, str) or CELL_ID_RE.fullmatch(cell_id) is None:
            errors.append(f"{prefix} cell_id invalid")
            continue
        if cell_id in by_id:
            errors.append(f"duplicate cell_id {cell_id}")
        by_id[cell_id] = cell
        observed.add((cell_id, cell.get("model_label"), cell.get("rate_id"),
                      cell.get("branch")))
        model = model_by_label.get(cell.get("model_label"))
        rate = rate_by_id.get(cell.get("rate_id"))
        branch = branch_by_id.get(cell.get("branch"))
        if model is None or rate is None or branch is None:
            errors.append(f"{cell_id} selects an unknown model/rate/branch")
        else:
            exact_dependencies = [
                _cell_id(model["model_name"], rate["rate_id"], dependency)
                for dependency in branch["dependencies"]
            ]
            exact_fields = {
                "model_name": model["model_name"], "hf_id": model["hf_id"],
                "model_family": model["family"], "model_dir": model["model_dir"],
                "nominal_params_b": model["nominal_params_b"],
                "rate_bpw": rate["rate_bpw"],
                "rate_fraction": {"numerator": rate["numerator"],
                                  "denominator": rate["denominator"]},
                "claim_scope": branch["claim_scope"],
                "adapter_id": branch["adapter"][model["family"]],
                "command": branch["operation"],
                "backend": (
                    "apple-cpu-strand" if model["family"] == "qwen2.5-dense"
                    else "apple-silicon-doctor-v5"
                ),
                "runtime_spec_schema": branch["runtime_spec_schema"],
                "dependencies": exact_dependencies,
                "expected_replicates": 1,
                "replicate_scope": "preliminary_scale_mapping_not_dominance",
            }
            for field, exact in exact_fields.items():
                if cell.get(field) != exact:
                    errors.append(f"{cell_id} canonical field mismatch: {field}")
            if cell_id != _cell_id(model["model_name"], rate["rate_id"],
                                   branch["branch"]):
                errors.append(f"{cell_id} canonical id mismatch")
            identity = {
                "model_label": cell.get("model_label"),
                "model_name": cell.get("model_name"), "hf_id": cell.get("hf_id"),
                "model_family": cell.get("model_family"),
                "parameter_manifest_sha256": (
                    cell.get("parameter_manifest", {}).get("file_sha256")
                    if isinstance(cell.get("parameter_manifest"), dict) else None
                ),
                "exact_stored_parameter_count": cell.get(
                    "exact_stored_parameter_count"
                ),
                "rate_id": cell.get("rate_id"), "rate_bpw": cell.get("rate_bpw"),
                "rate_fraction": cell.get("rate_fraction"),
                "branch": cell.get("branch"), "claim_scope": cell.get("claim_scope"),
                "adapter_id": cell.get("adapter_id"), "command": cell.get("command"),
                "dependencies": cell.get("dependencies"),
                "seed_plan": cell.get("seed_plan"),
            }
            if cell.get("cell_identity_sha256") != _hash_value(identity):
                errors.append(f"{cell_id} identity hash mismatch")
        if cell.get("priority") != index:
            errors.append(f"{cell_id} priority/order mismatch")
        if cell.get("status") != "planned" or cell.get("quality_claims_permitted") is not False \
                or cell.get("source_deletion_permitted") is not False:
            errors.append(f"{cell_id} safety/status boundary invalid")
        if cell.get("cell_spec_sha256") != _hash_value(_without(cell, "cell_spec_sha256")):
            errors.append(f"{cell_id} spec hash mismatch")
        if not isinstance(cell.get("dependencies"), list):
            errors.append(f"{cell_id} dependencies invalid")
        if cell.get("seed_plan") != list(SEED_PLAN):
            errors.append(f"{cell_id} seed plan incomplete")
        fraction = cell.get("rate_fraction")
        if not isinstance(fraction, dict) or set(fraction) != {"numerator", "denominator"}:
            errors.append(f"{cell_id} rate fraction invalid")
        else:
            numerator = fraction.get("numerator")
            denominator = fraction.get("denominator")
            if (isinstance(numerator, bool) or not isinstance(numerator, int)
                    or isinstance(denominator, bool) or not isinstance(denominator, int)
                    or numerator <= 0 or denominator <= 0):
                errors.append(f"{cell_id} rate fraction invalid")
            elif not math.isclose(numerator / denominator, cell.get("rate_bpw", -1),
                                  rel_tol=0, abs_tol=1e-12):
                errors.append(f"{cell_id} rate/fraction mismatch")
    if observed != expected:
        errors.append("cell matrix differs from the exact 8*10*4 campaign")
    for cell_id, cell in by_id.items():
        for dependency in cell.get("dependencies", []):
            if dependency not in by_id:
                errors.append(f"{cell_id} has unknown dependency {dependency}")
            elif by_id[dependency].get("priority", 10**9) >= cell.get("priority", -1):
                errors.append(f"{cell_id} dependency is not earlier in topological order")
    groups = plan.get("report_groups")
    if not isinstance(groups, list) or [row.get("expected_cells") for row in groups
                                       if isinstance(row, dict)] != [280, 40]:
        errors.append("report group cardinalities must be 280 and 40")
    if verify_sources:
        sources = plan.get("sources")
        if not isinstance(sources, dict):
            errors.append("source bindings missing")
        else:
            for name in ("orchestrator", "adapter_abi", "ram_scheduler",
                         "campaign_reporter", "parameter_manifest_contract",
                         "training_ladder"):
                binding = sources.get(name)
                if not isinstance(binding, dict):
                    errors.append(f"source binding missing: {name}")
                    continue
                try:
                    path = _resolve_workspace_path(binding.get("path"))
                    digest, size = _sha_file(path)
                    if digest != binding.get("sha256") or size != binding.get("bytes"):
                        errors.append(f"live source binding changed: {name}")
                except (CampaignError, OSError) as exc:
                    errors.append(f"source binding invalid ({name}): {exc}")
    return errors


def _state_row() -> dict[str, Any]:
    return {
        "status": "pending", "attempts": 0, "started_at": None,
        "completed_at": None, "last_exit_code": None, "blockers": [],
        "runtime_spec_sha256": None, "registry_sha256": None,
        "request_sha256": None, "result_sha256": None,
        "execution_receipt_sha256": None, "disposition_sha256": None,
        "packed_gc_receipt_sha256": None, "payload_released_at": None,
        "released_payload_bytes": 0,
        "error": None,
    }


def _base_state(plan: dict[str, Any]) -> dict[str, Any]:
    now = _now()
    state: dict[str, Any] = {
        "schema": STATE_SCHEMA, "version": VERSION,
        "plan_sha256": plan["plan_sha256"], "created_at": now, "updated_at": now,
        "status": "compiled", "control_mode": "run", "supervisor_pid": None,
        "active_cells": [], "active_children": {}, "last_resource_gate": None,
        "last_child_resource_sample": None, "max_child_tree_rss_bytes": 0,
        "last_resource_stop": None, "resource_stop_counts": {},
        "last_scan": None, "last_reporter_sync": None,
        "cells": {row["cell_id"]: _state_row() for row in plan["cells"]},
        "report_checkpoints": {"sub-120B": None, "120B": None},
        "source_deletion_permitted": False, "error": None,
    }
    state["state_sha256"] = _hash_value(state)
    return state


def _validate_state(state: Any, plan: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(state, dict):
        return ["state is not an object"]
    if state.get("schema") != STATE_SCHEMA or state.get("version") != VERSION:
        errors.append("state schema/version mismatch")
    if state.get("plan_sha256") != plan.get("plan_sha256"):
        errors.append("state plan binding mismatch")
    if state.get("state_sha256") != _hash_value(_without(state, "state_sha256")):
        errors.append("state hash mismatch")
    if state.get("status") not in QUEUE_STATUSES:
        errors.append("state queue status invalid")
    if state.get("control_mode") not in {"run", "pause", "drain"}:
        errors.append("state control mode invalid")
    if state.get("source_deletion_permitted") is not False:
        errors.append("state unexpectedly permits source deletion")
    rows = state.get("cells")
    expected = {cell["cell_id"] for cell in plan["cells"]}
    if not isinstance(rows, dict) or set(rows) != expected:
        return errors + ["state cell set mismatch"]
    for cell_id, row in rows.items():
        if not isinstance(row, dict) or row.get("status") not in CELL_STATUSES:
            errors.append(f"state row invalid: {cell_id}")
            continue
        if set(row) != set(_state_row()):
            errors.append(f"state row keys invalid: {cell_id}")
        if isinstance(row.get("attempts"), bool) or not isinstance(row.get("attempts"), int) \
                or row["attempts"] < 0:
            errors.append(f"attempt counter invalid: {cell_id}")
        for field in ("runtime_spec_sha256", "registry_sha256", "request_sha256",
                      "result_sha256", "execution_receipt_sha256", "disposition_sha256",
                      "packed_gc_receipt_sha256"):
            value = row.get(field)
            if value is not None and (not isinstance(value, str)
                                      or SHA256_RE.fullmatch(value) is None):
                errors.append(f"{cell_id} {field} invalid")
        released = row.get("released_payload_bytes")
        if isinstance(released, bool) or not isinstance(released, int) or released < 0:
            errors.append(f"{cell_id} released payload byte count invalid")
    checkpoints = state.get("report_checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {"sub-120B", "120B"}:
        errors.append("report checkpoint state invalid")
    reporter = state.get("last_reporter_sync")
    if reporter is not None and (not isinstance(reporter, dict)
                                 or reporter.get("ok") not in {True, False}
                                 or not isinstance(reporter.get("at"), str)):
        errors.append("last reporter sync state invalid")
    maximum_rss = state.get("max_child_tree_rss_bytes")
    if isinstance(maximum_rss, bool) or not isinstance(maximum_rss, int) \
            or maximum_rss < 0:
        errors.append("maximum child-tree RSS state is invalid")
    child_sample = state.get("last_child_resource_sample")
    if child_sample is not None and (
            not isinstance(child_sample, dict)
            or child_sample.get("plan_sha256") != plan.get("plan_sha256")
            or isinstance(child_sample.get("tree_rss_bytes"), bool)
            or not isinstance(child_sample.get("tree_rss_bytes"), int)
            or child_sample["tree_rss_bytes"] < 0):
        errors.append("last child resource sample state is invalid")
    active_children = state.get("active_children")
    if not isinstance(active_children, dict):
        errors.append("active children state is invalid")
    else:
        for key, record in active_children.items():
            if not isinstance(record, dict) \
                    or record.get("cell_id") != key or key not in rows \
                    or record.get("process_budget_bytes") != PROCESS_BUDGET_BYTES \
                    or record.get("pgid") != record.get("pid") \
                    or not isinstance(record.get("process_started"), str) \
                    or record.get("handshake_pending") not in {True, False} \
                    or (record.get("handshake_pending") is False
                        and (not isinstance(record.get("process_command_sha256"), str)
                             or SHA256_RE.fullmatch(
                                 record["process_command_sha256"]
                             ) is None)) \
                    or isinstance(record.get("reserved_bytes"), bool) \
                    or not isinstance(record.get("reserved_bytes"), int) \
                    or record["reserved_bytes"] <= 0 \
                    or record["reserved_bytes"] > PROCESS_BUDGET_BYTES:
                errors.append(f"active child state is invalid: {key}")
    active_cells = state.get("active_cells")
    expected_active = sorted(active_children) if isinstance(active_children, dict) else []
    if not isinstance(active_cells, list) or active_cells != expected_active:
        errors.append("active cells state is invalid")
    return errors


def _load_plan(*, verify_sources: bool = True) -> dict[str, Any]:
    plan = _read_json(PLAN)
    errors = validate_plan(plan, verify_sources=verify_sources)
    if errors:
        raise CampaignError("campaign plan invalid: " + "; ".join(errors))
    assert isinstance(plan, dict)
    return plan


def _load_state(plan: dict[str, Any]) -> dict[str, Any]:
    if not STATE.exists():
        state = _base_state(plan)
        _atomic_json(STATE, state)
        _publish_campaign(plan, state)
        return state
    state = _read_json(STATE)
    errors = _validate_state(state, plan)
    if errors:
        raise CampaignError("queue state invalid: " + "; ".join(errors))
    assert isinstance(state, dict)
    return state


def _group_cells(plan: dict[str, Any], group_id: str) -> list[dict[str, Any]]:
    if group_id == "sub-120B":
        return [cell for cell in plan["cells"] if cell["model_label"] != "120B"]
    if group_id == "120B":
        return [cell for cell in plan["cells"] if cell["model_label"] == "120B"]
    raise CampaignError(f"unknown report group: {group_id}")


def _reporter_terminal_complete(cell: dict[str, Any], row: dict[str, Any]) -> bool:
    locator = hashlib.sha256(cell["cell_id"].encode("utf-8")).hexdigest()
    checkpoint = _read_json(
        ULTRA_ROOT / "reporting" / "checkpoints" / f"{locator}.json"
    )
    expected_status = {
        "complete": "succeeded", "negative": "complete_negative",
        "unsupported": "unsupported",
    }.get(row["status"])
    declared = checkpoint.get("provenance", {}).get("declared", {}) \
        if isinstance(checkpoint, dict) else {}
    evidence_matches = (
        declared.get("campaign_result_sha256") == row["result_sha256"]
        if row["status"] == "complete" else
        declared.get("campaign_disposition_sha256") == row["disposition_sha256"]
    )
    return isinstance(checkpoint, dict) \
        and checkpoint.get("cell_id") == cell["cell_id"] \
        and checkpoint.get("status") == expected_status \
        and checkpoint.get("completeness", {}).get("complete") is True \
        and evidence_matches \
        and checkpoint.get("checkpoint_sha256") == _hash_value(
            _without(checkpoint, "checkpoint_sha256")
        )


def _report_groups(plan: dict[str, Any], state: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in plan["report_groups"]:
        group_id = source["group_id"]
        cells = _group_cells(plan, group_id)
        statuses = [state["cells"][cell["cell_id"]]["status"] for cell in cells]
        counts = {name: statuses.count(name) for name in sorted(CELL_STATUSES)}
        queue_terminal = len(cells) == source["expected_cells"] and all(
            status in REPORT_TERMINAL for status in statuses
        )
        reporter_complete = sum(
            _reporter_terminal_complete(cell, state["cells"][cell["cell_id"]])
            for cell in cells
        )
        ready = queue_terminal and reporter_complete == len(cells)
        output.append({
            **source, "counts": counts, "terminal_cells": sum(counts[name]
                for name in REPORT_TERMINAL),
            "queue_terminal": queue_terminal,
            "reporter_complete_cells": reporter_complete,
            "reporter_incomplete_cells": len(cells) - reporter_complete,
            "ready_for_verified_report": ready,
            "verified_report_checkpoint": state["report_checkpoints"].get(group_id),
            "candidate_gc_permitted": ready
            and state["report_checkpoints"].get(group_id) is not None,
            "parent_source_cleanup_permitted": False,
        })
    return output


def _parse_time(value: Any) -> dt.datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def _timing_projection(plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    by_branch: dict[str, list[float]] = {row["branch"]: [] for row in BRANCHES}
    completed_seconds = 0.0
    for cell in plan["cells"]:
        row = state["cells"][cell["cell_id"]]
        if row["status"] != "complete":
            continue
        started, completed = _parse_time(row["started_at"]), _parse_time(row["completed_at"])
        if started is None or completed is None or completed < started:
            continue
        elapsed = (completed - started).total_seconds()
        completed_seconds += elapsed
        billions = cell["exact_stored_parameter_count"] / 1_000_000_000
        if elapsed > 0 and billions > 0:
            by_branch[cell["branch"]].append(elapsed / billions)
    estimates: dict[str, Any] = {}
    total_remaining = 0.0
    missing: list[str] = []
    for branch in by_branch:
        samples = by_branch[branch]
        rate = statistics.median(samples) if samples else None
        remaining_cells = [
            cell for cell in plan["cells"]
            if cell["branch"] == branch
            and state["cells"][cell["cell_id"]]["status"] not in REPORT_TERMINAL
        ]
        estimate = sum(
            rate * cell["exact_stored_parameter_count"] / 1_000_000_000
            for cell in remaining_cells
        ) if rate is not None else None
        if remaining_cells and estimate is None:
            missing.append(branch)
        elif estimate is not None:
            total_remaining += estimate
        estimates[branch] = {
            "observed_complete_samples": len(samples),
            "median_wall_seconds_per_billion_stored_parameters": (
                round(rate, 6) if rate is not None else None
            ),
            "nonterminal_cells": len(remaining_cells),
            "estimated_remaining_wall_seconds": (
                round(estimate, 3) if estimate is not None else None
            ),
        }
    return {
        "method": (
            "empirical median wall-seconds per exact billion stored parameters, by branch; "
            "includes retries/waits between first start and terminal completion"
        ),
        "completed_cell_wall_seconds": round(completed_seconds, 3),
        "branches": estimates, "missing_empirical_branches": missing,
        "total_estimated_remaining_wall_seconds": (
            round(total_remaining, 3) if not missing else None
        ),
        "eta_status": "empirical" if not missing else "underdetermined_until_branch_samples",
        "unsupported_or_negative_cells_contribute_zero_compute": True,
    }


def _campaign_projection(plan: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    cells = []
    for spec in plan["cells"]:
        row = state["cells"][spec["cell_id"]]
        cells.append({
            "cell_id": spec["cell_id"],
            "cell_identity_sha256": spec["cell_identity_sha256"],
            "model_label": spec["model_label"], "model_name": spec["model_name"],
            "hf_id": spec["hf_id"], "model_family": spec["model_family"],
            "nominal_params_b": spec["nominal_params_b"],
            "exact_stored_parameter_count": spec["exact_stored_parameter_count"],
            "parameter_manifest": spec["parameter_manifest"],
            "source_census": spec["source_census"],
            "rate_id": spec["rate_id"], "rate_bpw": spec["rate_bpw"],
            "rate_fraction": spec["rate_fraction"], "branch": spec["branch"],
            "claim_scope": spec["claim_scope"], "adapter_id": spec["adapter_id"],
            "command": spec["command"], "backend": spec["backend"],
            "dependencies": spec["dependencies"], "seed_plan": spec["seed_plan"],
            "expected_replicates": spec["expected_replicates"],
            "replicate_scope": spec["replicate_scope"],
            "nominal_payload_bytes": spec["nominal_payload_bytes"],
            "projected_output_bytes": spec["projected_output_bytes"],
            "runtime_spec_path": spec["runtime_spec_path"],
            "disposition_path": spec["disposition_path"],
            "result_paths": {
                "root": _relative(RESULTS / spec["cell_id"]),
                "request": _relative(RESULTS / spec["cell_id"] / "request.json"),
                "checkpoint": _relative(RESULTS / spec["cell_id"] / "checkpoint.json"),
                "result": _relative(RESULTS / spec["cell_id"] / "result.json"),
                "execution_receipt": _relative(
                    RESULTS / spec["cell_id"] / "execution_receipt.json"
                ),
            },
            "admission": spec["admission"], "capability": spec["capability"],
            "lifecycle": spec["lifecycle"], "status": row["status"],
            "attempts": row["attempts"], "blockers": row["blockers"],
            "runtime_spec_sha256": row["runtime_spec_sha256"],
            "request_sha256": row["request_sha256"],
            "result_sha256": row["result_sha256"],
            "execution_receipt_sha256": row["execution_receipt_sha256"],
            "disposition_sha256": row["disposition_sha256"],
            "packed_gc_receipt_sha256": row["packed_gc_receipt_sha256"],
            "payload_released_at": row["payload_released_at"],
            "released_payload_bytes": row["released_payload_bytes"],
            "started_at": row["started_at"], "completed_at": row["completed_at"],
            "error": row["error"],
        })
    counts = {name: sum(row["status"] == name for row in state["cells"].values())
              for name in sorted(CELL_STATUSES)}
    campaign: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA, "version": VERSION, "generated_at": _now(),
        "plan_sha256": plan["plan_sha256"], "queue_status": state["status"],
        "control_mode": state["control_mode"], "active_cells": state["active_cells"],
        "active_children": state["active_children"], "counts": counts,
        "last_resource_gate": state["last_resource_gate"],
        "last_scan": state["last_scan"], "last_reporter_sync": state["last_reporter_sync"],
        "report_groups": _report_groups(plan, state),
        "timing": _timing_projection(plan, state),
        "lifecycle_policy": plan["lifecycle_policy"],
        "source_deletion_permitted": False, "cells": cells,
    }
    campaign["campaign_sha256"] = _hash_value(campaign)
    return campaign


def _publish_campaign(plan: dict[str, Any], state: dict[str, Any]) -> None:
    _atomic_json(CAMPAIGN, _campaign_projection(plan, state))


def _save_state(plan: dict[str, Any], state: dict[str, Any],
                status: str | None = None, **updates: Any) -> None:
    if status is not None:
        state["status"] = status
    state.update(updates)
    state["updated_at"] = _now()
    state.pop("state_sha256", None)
    state["state_sha256"] = _hash_value(state)
    errors = _validate_state(state, plan)
    if errors:
        raise CampaignError("refusing to save invalid state: " + "; ".join(errors))
    _atomic_json(STATE, state)
    _publish_campaign(plan, state)


def _sync_active_cells(state: dict[str, Any]) -> None:
    """Keep the published active-cell list in lock-step with active_children.

    The invariant active_cells == sorted(active_children) is asserted by
    _validate_state; every mutation of the set funnels through here so any plain
    _save_state persists a crash-consistent plural set with no drift counter.
    """
    state["active_cells"] = sorted(state["active_children"])


def _append_reporter_sync(row: dict[str, Any]) -> None:
    REPORTER_SYNC_LOG.parent.mkdir(parents=True, exist_ok=True)
    with REPORTER_SYNC_LOG.open("a", encoding="utf-8") as handle:
        handle.write(_canonical(row).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_dir(REPORTER_SYNC_LOG.parent)


def _sync_reporter(plan: dict[str, Any], state: dict[str, Any], *, reason: str) -> bool:
    """Run the source-bound reporter without granting it queue authority."""
    command = [
        sys.executable, str(REPORTER), "sync", "--campaign", str(CAMPAIGN),
        "--reporting-root", str(ULTRA_ROOT / "reporting"),
    ]
    row: dict[str, Any] = {
        "schema": "hawking.doctor_v5_ultra_reporter_sync.v1", "at": _now(),
        "reason": reason, "plan_sha256": plan["plan_sha256"],
        "command_argv_sha256": _hash_value(command),
        "reporter_source_sha256": None, "returncode": None, "ok": False,
        "stdout_tail": "", "stderr_tail": "", "error": None,
        "adopted_report_checkpoints": 0,
    }
    try:
        source_sha, _ = _sha_file(REPORTER)
        expected = plan["sources"]["campaign_reporter"]["sha256"]
        row["reporter_source_sha256"] = source_sha
        if source_sha != expected:
            raise CampaignError("reporter source changed after campaign compilation")
        process = subprocess.run(
            command, cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True,
            timeout=REPORTER_SYNC_TIMEOUT_SECONDS, check=False,
        )
        row["returncode"] = process.returncode
        row["stdout_tail"] = process.stdout[-4000:]
        row["stderr_tail"] = process.stderr[-4000:]
        row["ok"] = process.returncode == 0
        if process.returncode != 0:
            row["error"] = f"reporter exited with status {process.returncode}"
        else:
            row["adopted_report_checkpoints"] = _adopt_reporter_checkpoints(
                plan, state
            )
    except Exception as exc:
        row["ok"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["sync_sha256"] = _hash_value(row)
    try:
        _append_reporter_sync(row)
    except OSError:
        pass
    prior = state.get("last_reporter_sync")
    state["last_reporter_sync"] = {
        "at": row["at"], "reason": reason, "ok": row["ok"],
        "returncode": row["returncode"], "sync_sha256": row["sync_sha256"],
        "log_path": _relative(REPORTER_SYNC_LOG), "error": row["error"],
    }
    try:
        _save_state(plan, state)
    except Exception:
        state["last_reporter_sync"] = prior
    try:
        _append_event("reporter-sync", reason=reason, ok=row["ok"],
                      sync_sha256=row["sync_sha256"])
    except OSError:
        pass
    return bool(row["ok"])


def _reporter_checkpoint_ref(target: dict[str, Any],
                             target_row: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve the reporter's immutable, result-bound target checkpoint."""
    locator = hashlib.sha256(target["cell_id"].encode("utf-8")).hexdigest()
    current_path = ULTRA_ROOT / "reporting" / "checkpoints" / f"{locator}.json"
    checkpoint = _read_json(current_path)
    if not isinstance(checkpoint, dict) or checkpoint.get("cell_id") != target["cell_id"] \
            or checkpoint.get("status") != "succeeded" \
            or checkpoint.get("completeness", {}).get("complete") is not True \
            or checkpoint.get("provenance", {}).get("declared", {}).get(
                "campaign_result_sha256") != target_row["result_sha256"] \
            or checkpoint.get("checkpoint_sha256") != _hash_value(
                _without(checkpoint, "checkpoint_sha256")
            ):
        return None
    revision = checkpoint.get("revision")
    digest = checkpoint.get("checkpoint_sha256")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0 \
            or not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        return None
    immutable = (ULTRA_ROOT / "reporting" / "checkpoint_revisions" / locator
                 / f"r{revision:08d}-{digest}.json")
    immutable_doc = _read_json(immutable)
    if immutable_doc != checkpoint:
        return None
    file_sha, size = _sha_file(immutable)
    return {"path": _relative(immutable), "sha256": file_sha, "bytes": size}


def _valid_immutable_reporter_ref(target: dict[str, Any],
                                  target_row: dict[str, Any],
                                  reference: Any) -> bool:
    """Validate a historical target checkpoint for GC crash recovery."""
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "bytes"}:
        return False
    try:
        path = _resolve_workspace_path(reference["path"])
        locator = hashlib.sha256(target["cell_id"].encode("utf-8")).hexdigest()
        expected_root = (ULTRA_ROOT / "reporting" / "checkpoint_revisions"
                         / locator).resolve()
        path.relative_to(expected_root)
        checkpoint = _read_json(path)
        digest, size = _sha_file(path)
    except (CampaignError, OSError, ValueError):
        return False
    declared = checkpoint.get("provenance", {}).get("declared", {}) \
        if isinstance(checkpoint, dict) else {}
    revision = checkpoint.get("revision") if isinstance(checkpoint, dict) else None
    checkpoint_sha = checkpoint.get("checkpoint_sha256") \
        if isinstance(checkpoint, dict) else None
    return isinstance(checkpoint, dict) \
        and checkpoint.get("cell_id") == target["cell_id"] \
        and checkpoint.get("status") == "succeeded" \
        and checkpoint.get("completeness", {}).get("complete") is True \
        and declared.get("campaign_result_sha256") == target_row["result_sha256"] \
        and checkpoint_sha == _hash_value(_without(checkpoint, "checkpoint_sha256")) \
        and isinstance(revision, int) and not isinstance(revision, bool) and revision > 0 \
        and path.parent == expected_root \
        and path.name == f"r{revision:08d}-{checkpoint_sha}.json" \
        and digest == reference.get("sha256") and size == reference.get("bytes")


def _gc_targets(plan: dict[str, Any], successor: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {cell["cell_id"]: cell for cell in plan["cells"]}
    targets: list[dict[str, Any]] = []
    if successor["branch"] == "codec_control":
        rate_index = next(index for index, rate in enumerate(RATES)
                          if rate["rate_id"] == successor["rate_id"])
        if rate_index > 0:
            prior_rate = RATES[rate_index - 1]["rate_id"]
            targets.append(next(
                cell for cell in plan["cells"]
                if cell["model_label"] == successor["model_label"]
                and cell["rate_id"] == prior_rate
                and cell["branch"] == "doctor_full"
            ))
    elif successor["branch"] == "doctor_static":
        targets.append(by_id[successor["dependencies"][0]])
    elif successor["branch"] == "doctor_conditional":
        targets.append(next(by_id[cell_id] for cell_id in successor["dependencies"]
                            if by_id[cell_id]["branch"] == "doctor_static"))
    elif successor["branch"] == "doctor_full":
        targets.append(next(by_id[cell_id] for cell_id in successor["dependencies"]
                            if by_id[cell_id]["branch"] == "doctor_conditional"))
    return targets


def _packed_gc_receipt_path(cell: dict[str, Any]) -> Path:
    return RESULTS / cell["cell_id"] / "packed_gc_receipt.json"


def _successor_runtime_ref(successor: dict[str, Any]) -> dict[str, Any]:
    """Bind GC authority to the already-sealed program that will consume capacity."""
    path = _resolve_workspace_path(successor["runtime_spec_path"])
    spec = _read_json(path)
    runtime, _, errors = _validate_runtime_spec(
        successor, spec, path, verify_inputs=False
    )
    if errors or runtime is None or not isinstance(spec, dict):
        raise CampaignError(
            f"successor runtime spec is not sealed for GC: {successor['cell_id']}: "
            + "; ".join(errors)
        )
    digest, size = _sha_file(path)
    if digest != runtime["sha256"]:
        raise CampaignError("successor runtime spec changed while binding GC authority")
    return {
        "cell_id": successor["cell_id"],
        "cell_identity_sha256": successor["cell_identity_sha256"],
        "runtime_spec_path": _relative(path),
        "runtime_spec_sha256": digest,
        "runtime_spec_bytes": size,
        "program_spec_sha256": spec["program_spec_sha256"],
    }


def _validate_gc_intent(intent: Any, *, plan: dict[str, Any],
                        target: dict[str, Any], successor_ref: dict[str, Any],
                        target_result_sha256: str,
                        deleted: list[dict[str, Any]], reporter_ref: dict[str, Any]) -> bool:
    return isinstance(intent, dict) \
        and set(intent) == {
            "schema", "version", "plan_sha256", "cell_id",
            "cell_identity_sha256", "result_sha256", "successor",
            "reporter_sync", "deleted_artifacts", "created_at",
            "source_deletion_permitted", "intent_sha256",
        } \
        and intent.get("schema") == PACKED_GC_INTENT_SCHEMA \
        and intent.get("version") == VERSION \
        and intent.get("plan_sha256") == plan["plan_sha256"] \
        and intent.get("cell_id") == target["cell_id"] \
        and intent.get("cell_identity_sha256") == target["cell_identity_sha256"] \
        and intent.get("result_sha256") == target_result_sha256 \
        and intent.get("successor") == successor_ref \
        and intent.get("deleted_artifacts") == deleted \
        and intent.get("reporter_sync") == reporter_ref \
        and intent.get("source_deletion_permitted") is False \
        and intent.get("intent_sha256") == _hash_value(
            _without(intent, "intent_sha256")
        )


def _gc_packed_payload(plan: dict[str, Any], state: dict[str, Any],
                       target: dict[str, Any], successor: dict[str, Any],
                       reporter_ref: dict[str, Any]) -> None:
    """Hash, unlink, fsync, and receipt only an adapter-authored packed allowlist."""
    target_row = state["cells"][target["cell_id"]]
    successor_ref = _successor_runtime_ref(successor)
    receipt_path = _packed_gc_receipt_path(target)
    existing = _read_json(receipt_path)
    result_path = RESULTS / target["cell_id"] / "result.json"
    result = _read_json(result_path)
    if not isinstance(result, dict) or result.get("result_sha256") \
            != target_row["result_sha256"]:
        raise CampaignError(f"sealed target result is unavailable for GC: {target['cell_id']}")
    deleted, inventory_errors = _packed_inventory(target, result, verify_live=False)
    if inventory_errors:
        raise CampaignError("refusing packed GC: " + "; ".join(inventory_errors))
    if isinstance(existing, dict):
        expected_keys = {
            "schema", "cell_id", "cell_identity_sha256", "result_sha256",
            "successor", "reporter_sync", "deleted_artifacts",
            "retained_evidence_roles", "parent_source_deleted", "completed_at",
            "receipt_sha256",
        }
        successor_proof = existing.get("successor")
        existing_reporter_ref = existing.get("reporter_sync")
        if set(existing) != expected_keys \
                or existing.get("schema") != PACKED_GC_RECEIPT_SCHEMA \
                or existing.get("cell_id") != target["cell_id"] \
                or existing.get("cell_identity_sha256") != target["cell_identity_sha256"] \
                or existing.get("result_sha256") != target_row["result_sha256"] \
                or successor_proof != successor_ref \
                or not _valid_immutable_reporter_ref(
                    target, target_row, existing_reporter_ref
                ) \
                or existing.get("deleted_artifacts") != deleted \
                or existing.get("parent_source_deleted") is not False \
                or existing.get("receipt_sha256") != _hash_value(
                    _without(existing, "receipt_sha256")
                ):
            raise CampaignError(f"packed GC receipt is invalid: {target['cell_id']}")
        for artifact in deleted:
            if Path(artifact["path"]).exists():
                raise CampaignError(
                    f"packed payload reappeared after GC receipt: {artifact['path']}"
                )
        digest, _ = _sha_file(receipt_path)
        target_row["packed_gc_receipt_sha256"] = digest
        target_row["payload_released_at"] = existing.get("completed_at")
        target_row["released_payload_bytes"] = sum(row["bytes"] for row in deleted)
        return
    intent_path = RESULTS / target["cell_id"] / "packed_gc_intent.json"
    intent = _read_json(intent_path)
    intent_reporter_ref = intent.get("reporter_sync") \
        if isinstance(intent, dict) else None
    recovery_reporter_ref = (
        intent_reporter_ref
        if _valid_immutable_reporter_ref(target, target_row, intent_reporter_ref)
        else reporter_ref
    )
    has_valid_intent = _validate_gc_intent(
        intent, plan=plan, target=target, successor_ref=successor_ref,
        target_result_sha256=target_row["result_sha256"],
        deleted=deleted, reporter_ref=recovery_reporter_ref,
    )
    if intent is not None and not has_valid_intent:
        raise CampaignError(f"packed GC intent is invalid: {target['cell_id']}")
    # Before authority is journaled, every payload must still exist and be
    # content-hashed.  A valid intent permits crash recovery from a partial unlink.
    for artifact in deleted:
        path = Path(artifact["path"])
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise CampaignError(f"packed GC target is not a regular file: {path}")
            digest, size = _sha_file(path)
            if digest != artifact["sha256"] or size != artifact["bytes"]:
                raise CampaignError(f"packed GC target identity changed: {path}")
        elif not has_valid_intent:
            raise CampaignError(f"packed GC target vanished before intent: {path}")
    if not has_valid_intent:
        intent = {
            "schema": PACKED_GC_INTENT_SCHEMA, "version": VERSION,
            "plan_sha256": plan["plan_sha256"], "cell_id": target["cell_id"],
            "cell_identity_sha256": target["cell_identity_sha256"],
            "result_sha256": target_row["result_sha256"],
            "successor": successor_ref,
            "reporter_sync": recovery_reporter_ref, "deleted_artifacts": deleted,
            "created_at": _now(), "source_deletion_permitted": False,
        }
        intent["intent_sha256"] = _hash_value(intent)
        _atomic_json(intent_path, intent)
    for artifact in deleted:
        path = Path(artifact["path"])
        if path.exists():
            path.unlink()
            _fsync_dir(path.parent)
    retained_roles = [
        "bundle_manifest", "worker_request", "worker_checkpoint", "worker_receipt",
        "dependency_evidence", "outer_result", "outer_execution_receipt",
    ]
    receipt: dict[str, Any] = {
        "schema": PACKED_GC_RECEIPT_SCHEMA,
        "cell_id": target["cell_id"],
        "cell_identity_sha256": target["cell_identity_sha256"],
        "result_sha256": target_row["result_sha256"],
        "successor": successor_ref,
        "reporter_sync": recovery_reporter_ref, "deleted_artifacts": deleted,
        "retained_evidence_roles": retained_roles,
        "parent_source_deleted": False, "completed_at": _now(),
    }
    receipt["receipt_sha256"] = _hash_value(receipt)
    _atomic_json(receipt_path, receipt)
    receipt_file_sha, _ = _sha_file(receipt_path)
    target_row["packed_gc_receipt_sha256"] = receipt_file_sha
    target_row["payload_released_at"] = receipt["completed_at"]
    target_row["released_payload_bytes"] = sum(row["bytes"] for row in deleted)
    _append_event(
        "packed-payload-gc", cell_id=target["cell_id"],
        successor_cell_id=successor["cell_id"],
        released_payload_bytes=target_row["released_payload_bytes"],
        packed_gc_receipt_sha256=receipt["receipt_sha256"],
        parent_source_deleted=False,
    )


def _reconcile_lifecycle(plan: dict[str, Any], state: dict[str, Any],
                         *, successor_only: dict[str, Any] | None = None) -> int:
    successors = [successor_only] if successor_only is not None else plan["cells"]
    released = 0
    for successor in successors:
        successor_row = state["cells"][successor["cell_id"]]
        # A terminal negative/unsupported cell will never consume this capacity.
        # All other statuses are eligible so deletion occurs before the successor's
        # resource gate rather than after it has already required duplicate space.
        if successor_row["status"] in {"negative", "unsupported"}:
            continue
        targets = _gc_targets(plan, successor)
        pending = [target for target in targets
                   if state["cells"][target["cell_id"]]["status"] == "complete"
                   and state["cells"][target["cell_id"]]["packed_gc_receipt_sha256"] is None]
        if not pending:
            continue
        for target in pending:
            target_row = state["cells"][target["cell_id"]]
            reporter_ref = _reporter_checkpoint_ref(target, target_row)
            if reporter_ref is None:
                continue
            _gc_packed_payload(plan, state, target, successor, reporter_ref)
            released += 1
    if released:
        _save_state(plan, state)
    return released


def _base_control(plan: dict[str, Any]) -> dict[str, Any]:
    doc = {"schema": CONTROL_SCHEMA, "version": VERSION,
           "plan_sha256": plan["plan_sha256"], "sequence": 0,
           "mode": "run", "updated_at": _now()}
    doc["control_sha256"] = _hash_value(doc)
    return doc


def _load_control(plan: dict[str, Any]) -> dict[str, Any]:
    if not CONTROL.exists():
        doc = _base_control(plan)
        _atomic_json(CONTROL, doc)
        return doc
    doc = _read_json(CONTROL)
    if (not isinstance(doc, dict) or doc.get("schema") != CONTROL_SCHEMA
            or doc.get("version") != VERSION
            or doc.get("plan_sha256") != plan["plan_sha256"]
            or doc.get("mode") not in {"run", "pause", "drain"}
            or isinstance(doc.get("sequence"), bool)
            or not isinstance(doc.get("sequence"), int) or doc["sequence"] < 0
            or doc.get("control_sha256") != _hash_value(_without(doc, "control_sha256"))):
        raise CampaignError("control document is invalid or tampered")
    return doc


def set_control(mode: str) -> int:
    plan = _load_plan()
    old = _load_control(plan)
    doc = {"schema": CONTROL_SCHEMA, "version": VERSION,
           "plan_sha256": plan["plan_sha256"], "sequence": old["sequence"] + 1,
           "mode": mode, "updated_at": _now()}
    doc["control_sha256"] = _hash_value(doc)
    _atomic_json(CONTROL, doc)
    _append_event("control", mode=mode, sequence=doc["sequence"],
                  control_sha256=doc["control_sha256"])
    if mode == "drain":
        owner = _read_json(PID_FILE, {})
        if _owner_alive(owner, plan):
            try:
                os.kill(int(owner["pid"]), signal.SIGTERM)
            except ProcessLookupError:
                pass
    print(json.dumps(doc, indent=2, sort_keys=True))
    return 0


def _thermal() -> dict[str, Any]:
    try:
        process = subprocess.run(["pmset", "-g", "therm"], capture_output=True,
                                 text=True, timeout=5, check=False)
        output = (process.stdout + process.stderr).strip()
        return {"ok": ram_scheduler.thermal_output_ok(process.returncode, output),
                "returncode": process.returncode, "output": output[-1000:]}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _admission_math(*, free_bytes: int | None, scratch_bytes: int,
                    projected_output_bytes: int,
                    resident_payload_bytes: int,
                    resident_predecessor_bytes: int) -> dict[str, Any]:
    """Capacity math that exposes resident predecessors without double counting.

    Files already resident are absent from ``free_bytes``.  The equivalent total
    capacity comparison is therefore (free + resident) against (resident + new
    output + scratch + reserve).  Keeping both sides in the receipt prevents a
    retained predecessor from disappearing from the audit while avoiding a
    mathematically incorrect second subtraction.
    """
    values = (scratch_bytes, projected_output_bytes, resident_payload_bytes,
              resident_predecessor_bytes)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in values):
        raise CampaignError("admission byte inputs must be nonnegative integers")
    required_free = DISK_RESERVE_BYTES + scratch_bytes + projected_output_bytes
    available_total = (free_bytes + resident_payload_bytes
                       if isinstance(free_bytes, int) and free_bytes >= 0 else None)
    required_total = resident_payload_bytes + required_free
    return {
        "required_free_bytes": required_free,
        "available_total_capacity_bytes": available_total,
        "required_total_capacity_bytes": required_total,
        "resident_payload_bytes": resident_payload_bytes,
        "resident_predecessor_bytes": resident_predecessor_bytes,
        "projected_incremental_output_bytes": projected_output_bytes,
        "capacity_ok": free_bytes is not None and free_bytes >= required_free,
    }


def _resource_gate(scratch_bytes: int, *, projected_output_bytes: int = 0,
                   resident_payload_bytes: int = 0,
                   resident_predecessor_bytes: int = 0) -> dict[str, Any]:
    blockers: list[str] = []
    if (isinstance(scratch_bytes, bool) or not isinstance(scratch_bytes, int)
            or scratch_bytes < MIN_SCRATCH_BYTES
            or scratch_bytes > MAX_DECLARED_SCRATCH_BYTES):
        blockers.append("declared scratch is outside the campaign safety envelope")
        scratch_bytes = MAX_DECLARED_SCRATCH_BYTES
    try:
        snapshot = ram_scheduler.resource_snapshot(str(ROOT))
    except Exception as exc:
        snapshot = {"error": f"{type(exc).__name__}: {exc}"}
    thermal = _thermal()
    pressure = snapshot.get("pressure_level")
    swap_mb = snapshot.get("swap_used_mb")
    free_gb = snapshot.get("disk_free_gb")
    power = str(snapshot.get("power_source", ""))
    if pressure != 1:
        blockers.append("memory pressure is not normal")
    if (isinstance(swap_mb, bool) or not isinstance(swap_mb, (int, float))
            or not math.isfinite(float(swap_mb)) or float(swap_mb) > SWAP_TOLERANCE_MB):
        blockers.append("swap exceeds tolerance or is unavailable")
    if "AC Power" not in power:
        blockers.append("AC power is not confirmed")
    if not thermal.get("ok"):
        blockers.append("thermal state is not green")
    free_bytes = int(float(free_gb) * 1_000_000_000) \
        if isinstance(free_gb, (int, float)) and not isinstance(free_gb, bool) \
        and math.isfinite(float(free_gb)) else None
    capacity = _admission_math(
        free_bytes=free_bytes, scratch_bytes=scratch_bytes,
        projected_output_bytes=projected_output_bytes,
        resident_payload_bytes=resident_payload_bytes,
        resident_predecessor_bytes=resident_predecessor_bytes,
    )
    if not capacity["capacity_ok"]:
        blockers.append(
            f"disk free is below {capacity['required_free_bytes'] / 1e9:.3f} GB"
        )
    return {
        "schema": "hawking.doctor_v5_ultra_resource_gate.v1", "sampled_at": _now(),
        "ok": not blockers, "blockers": blockers,
        "disk_reserve_bytes": DISK_RESERVE_BYTES, "scratch_bytes": scratch_bytes,
        **capacity, "resources": snapshot, "thermal": thermal,
    }


def _validate_runtime_inputs(rows: Any, *, verify_files: bool = True) \
        -> tuple[list[dict[str, Any]], list[str]]:
    normalized: list[dict[str, Any]] = []
    errors: list[str] = []
    if not isinstance(rows, list) or not rows:
        return [], ["runtime spec inputs must be a nonempty list"]
    roles: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"role", "path", "sha256", "bytes"}:
            errors.append(f"runtime input[{index}] keys invalid")
            continue
        role = row.get("role")
        if not isinstance(role, str) or not role or role in roles:
            errors.append(f"runtime input[{index}] role invalid or duplicate")
            continue
        roles.add(role)
        try:
            raw_path = row.get("path")
            candidate = Path(raw_path) if isinstance(raw_path, str) else Path("")
            if candidate.is_absolute():
                if "\x00" in str(raw_path):
                    raise CampaignError("input path contains NUL")
                path = candidate.resolve(strict=True)
                path.relative_to(ROOT.resolve())
                if candidate.is_symlink():
                    raise CampaignError("symlink input is forbidden")
            else:
                path = _resolve_workspace_path(raw_path)
            if verify_files:
                digest, size = _sha_file(path)
                if digest != row.get("sha256") or size != row.get("bytes"):
                    errors.append(f"runtime input[{index}] live identity mismatch")
            else:
                info = path.stat()
                digest, size = row.get("sha256"), row.get("bytes")
                if (not stat.S_ISREG(info.st_mode) or not isinstance(digest, str)
                        or SHA256_RE.fullmatch(digest) is None
                        or isinstance(size, bool) or not isinstance(size, int)
                        or size < 0 or info.st_size != size):
                    errors.append(f"runtime input[{index}] structural identity mismatch")
            normalized.append({"role": role, "path": str(path), "sha256": digest,
                               "bytes": size})
        except (CampaignError, OSError, ValueError) as exc:
            errors.append(f"runtime input[{index}] invalid: {exc}")
    normalized.sort(key=lambda row: row["role"])
    return normalized, errors


def _runtime_resource_doc(spec: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
    value = spec.get("resources", spec.get("resource_admission"))
    errors: list[str] = []
    if not isinstance(value, dict):
        return None, ["runtime spec resources are missing"]
    scratch = value.get("scratch_budget_bytes")
    reserve = value.get("disk_reserve_bytes")
    threads = value.get("threads", spec.get("threads"))
    if (isinstance(scratch, bool) or not isinstance(scratch, int)
            or scratch < MIN_SCRATCH_BYTES or scratch > MAX_DECLARED_SCRATCH_BYTES):
        errors.append("runtime scratch budget is outside the safety envelope")
    if (isinstance(reserve, bool) or not isinstance(reserve, int)
            or reserve < DISK_RESERVE_BYTES):
        errors.append("runtime disk reserve is below 150 GB")
    if (isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 32):
        errors.append("runtime thread count is invalid")
    return value, errors


def _runtime_rate(spec: dict[str, Any]) -> tuple[str | None, float | None]:
    binding = spec.get("campaign_binding")
    if isinstance(binding, dict):
        return binding.get("target_rate_id", binding.get("rate_id")), \
            binding.get("target_rate_bpw")
    codec = spec.get("codec")
    if isinstance(codec, dict):
        return codec.get("rate_id"), codec.get("nominal_payload_bpw")
    return spec.get("rate_id"), spec.get("target_rate_bpw")


def _runtime_program_payload(spec: dict[str, Any]) -> dict[str, Any]:
    """Semantic runtime program identity, excluding inputs and resource sizing."""
    excluded = {
        "schema", "inputs", "resources", "resource_admission",
        "program_spec_sha256", "resource_admission_sha256",
    }
    return {key: copy.deepcopy(value) for key, value in spec.items()
            if key not in excluded}


def _validate_runtime_spec(cell: dict[str, Any], spec: Any,
                           spec_path: Path, *, verify_inputs: bool = True) \
        -> tuple[dict[str, Any] | None,
                                                    list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if not isinstance(spec, dict):
        return None, [], ["runtime spec is missing or unreadable"]
    if spec.get("schema") != cell["runtime_spec_schema"]:
        errors.append("runtime spec schema does not match the cell")
    if spec.get("label") != cell["model_label"]:
        errors.append("runtime spec model label mismatch")
    family = spec.get("family", spec.get("model_family"))
    if family != cell["model_family"]:
        errors.append("runtime spec model family mismatch")
    if spec.get("backend") != cell["backend"]:
        errors.append("runtime spec backend mismatch")
    if spec.get("adapter_id") != cell["adapter_id"]:
        errors.append("runtime spec adapter mismatch")
    if spec.get("operation") != cell["command"]:
        errors.append("runtime spec operation mismatch")
    if spec.get("quality_claims_permitted") is not False \
            or spec.get("source_deletion_permitted") is not False:
        errors.append("runtime spec claim/source-deletion boundary invalid")
    rate_id, rate_bpw = _runtime_rate(spec)
    if rate_id != cell["rate_id"]:
        errors.append("runtime spec rate_id mismatch")
    if (isinstance(rate_bpw, bool) or not isinstance(rate_bpw, (int, float))
            or not math.isfinite(float(rate_bpw))
            or float(rate_bpw) > cell["rate_bpw"] + 1e-12):
        errors.append("runtime nominal rate exceeds the cell physical ceiling")
    binding = spec.get("campaign_binding")
    if not isinstance(binding, dict) or set(binding) != {
            "cell_id", "cell_identity_sha256", "branch", "target_rate_id",
            "target_rate_bpw", "label"}:
        errors.append("runtime campaign binding is missing or has invalid keys")
    else:
        for field, expected in (
            ("cell_id", cell["cell_id"]),
            ("cell_identity_sha256", cell["cell_identity_sha256"]),
            ("branch", cell["branch"]),
            ("label", cell["model_label"]),
            ("target_rate_id", cell["rate_id"]),
            ("target_rate_bpw", cell["rate_bpw"]),
        ):
            if binding.get(field) != expected:
                errors.append(f"runtime campaign binding mismatch: {field}")
    for field in ("program_spec_sha256", "resource_admission_sha256"):
        if not isinstance(spec.get(field), str) or SHA256_RE.fullmatch(spec[field]) is None:
            errors.append(f"runtime {field} invalid")
    if isinstance(spec.get("program_spec_sha256"), str) \
            and spec.get("program_spec_sha256") != _hash_value(
                _runtime_program_payload(spec)
            ):
        errors.append("runtime program spec hash mismatch")
    resources, resource_errors = _runtime_resource_doc(spec)
    errors.extend(resource_errors)
    if isinstance(resources, dict) and spec.get("resource_admission_sha256") \
            != _hash_value(resources):
        errors.append("runtime resource admission hash mismatch")
    inputs, input_errors = _validate_runtime_inputs(spec.get("inputs"),
                                                     verify_files=verify_inputs)
    errors.extend(input_errors)
    try:
        digest, _ = _sha_file(spec_path)
    except (CampaignError, OSError) as exc:
        errors.append(f"runtime spec identity failed: {exc}")
        digest = None
    return ({"document": spec, "path": spec_path, "sha256": digest,
             "resources": resources} if digest else None), inputs, errors


def _registry_snapshot(cell: dict[str, Any], output_dir: Path) \
        -> tuple[dict[str, Any] | None, Path, list[str]]:
    snapshot_path = output_dir / "adapter_registry.json"
    source = snapshot_path if snapshot_path.exists() else REGISTRY_PATH
    registry = _read_json(source)
    errors = adapter_abi.validate_registry(registry, verify_files=True, base_dir=ROOT)
    if not errors and isinstance(registry, dict):
        entry = next((row for row in registry.get("entries", [])
                      if row.get("adapter_id") == cell["adapter_id"]), None)
        if entry is None:
            errors.append(f"reviewed adapter is not registered: {cell['adapter_id']}")
        else:
            if cell["command"] not in entry.get("operations", []):
                errors.append("registered adapter does not allow the cell command")
            if cell["model_family"] not in entry.get("model_families", []):
                errors.append("registered adapter does not allow the model family")
            if cell["backend"] not in entry.get("backends", []):
                errors.append("registered adapter does not allow the backend")
    if not errors and not snapshot_path.exists():
        assert isinstance(registry, dict)
        _atomic_json(snapshot_path, registry)
    return registry if isinstance(registry, dict) else None, snapshot_path, errors


def _prepare_execution(cell: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    spec_path = _resolve_workspace_path(cell["runtime_spec_path"], must_exist=False)
    spec_doc = _read_json(spec_path)
    # Structural verification only (presence + size + sha format), NOT a content
    # re-hash of the model-sized source shards.  _scan_runnable_heads prepares
    # EVERY runnable chain-head on EVERY admission scan; content-hashing each
    # cell's multi-GB source here pegged a core and made the pre-admission scan
    # never return (last_scan stayed null, no cell ever launched).  Source-shard
    # content is rehashed at execution by the worker (the strand builder and the
    # GC-authority path both defer shard hashing to execution for the same
    # reason), so the scheduler scan stays O(stat) instead of O(source bytes).
    runtime, inputs, spec_errors = _validate_runtime_spec(cell, spec_doc, spec_path,
                                                          verify_inputs=False)
    blockers.extend(spec_errors)
    output_dir = RESULTS / cell["cell_id"]
    registry_path = output_dir / "adapter_registry.json"
    registry = None
    if not blockers:
        output_dir.mkdir(parents=True, exist_ok=True)
        registry, registry_path, registry_errors = _registry_snapshot(cell, output_dir)
        blockers.extend(registry_errors)
    request_path = output_dir / "request.json"
    result_path = output_dir / "result.json"
    checkpoint_path = output_dir / "checkpoint.json"
    receipt_path = output_dir / "execution_receipt.json"
    request = _read_json(request_path)
    command: list[str] | None = None
    scratch_bytes = cell["admission"]["recommended_scratch_bytes"]
    if runtime is not None and isinstance(runtime.get("resources"), dict):
        scratch_bytes = runtime["resources"].get("scratch_budget_bytes", scratch_bytes)
    if not blockers and isinstance(runtime, dict) and isinstance(registry, dict):
        if not isinstance(request, dict):
            manifest_path = _resolve_workspace_path(cell["parameter_manifest"]["path"])
            census_path = _resolve_workspace_path(cell["source_census"]["path"])
            try:
                request = adapter_abi.build_request(
                    registry=registry, adapter_id=cell["adapter_id"],
                    operation=cell["command"],
                    program_spec_sha256=runtime["document"]["program_spec_sha256"],
                    parameter_manifest_path=str(manifest_path),
                    parameter_manifest_sha256=cell["parameter_manifest"]["file_sha256"],
                    source_census_sha256=cell["source_census"]["report_sha256"],
                    model_label=cell["model_label"], model_family=cell["model_family"],
                    backend=cell["backend"], seed=cell["seed_plan"][0], inputs=inputs,
                    pilot_spec_path=str(runtime["path"]),
                    pilot_spec_sha256=runtime["sha256"],
                    pilot_spec_schema=cell["runtime_spec_schema"],
                    request_path=str(request_path), output_dir=str(output_dir),
                    checkpoint_path=str(checkpoint_path), result_path=str(result_path),
                    execution_receipt_path=str(receipt_path),
                    operator_greenlight_sha256=cell["cell_identity_sha256"],
                    resource_admission_sha256=runtime["document"][
                        "resource_admission_sha256"
                    ],
                )
                _atomic_json(request_path, request)
            except Exception as exc:
                blockers.append(f"request builder failed: {type(exc).__name__}: {exc}")
        if isinstance(request, dict):
            # verify_files=False for the same reason as the runtime-spec check
            # above: the request artifact rows include the model-sized source
            # shards, and content-rehashing them on every admission scan is what
            # busy-spun the pre-admission scan.  Execution rehashes them.
            request_errors = adapter_abi.validate_request(request, registry, verify_files=False)
            blockers.extend(f"request: {row}" for row in request_errors)
            authorization = request.get("authorization", {})
            if authorization.get("operator_greenlight_sha256") != cell["cell_identity_sha256"]:
                blockers.append("request is not bound to this campaign cell")
            if request.get("pilot_spec", {}).get("sha256") != runtime["sha256"]:
                blockers.append("request runtime spec binding mismatch")
            if not blockers:
                try:
                    command = adapter_abi.resolve_command(request, registry,
                                                          request_path=request_path)
                except Exception as exc:
                    blockers.append(f"typed command resolution failed: {exc}")
    if command is not None:
        if (not isinstance(command, list) or not command
                or any(not isinstance(token, str) or not token or "\x00" in token
                       for token in command)):
            blockers.append("adapter resolver returned invalid argv")
        else:
            executable = Path(command[0]).resolve(strict=False)
            if not executable.is_file() or executable.is_symlink():
                blockers.append("resolved executable is absent or symlinked")
    return {
        "cell": cell, "blockers": blockers, "runtime": runtime, "registry": registry,
        "registry_path": registry_path, "request": request, "command": command,
        "output_dir": output_dir, "request_path": request_path,
        "result_path": result_path, "checkpoint_path": checkpoint_path,
        "receipt_path": receipt_path, "scratch_bytes": scratch_bytes,
    }


def _validate_outputs(execution: dict[str, Any]) \
        -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    request, registry = execution["request"], execution["registry"]
    result = _read_json(execution["result_path"])
    checkpoint = _read_json(execution["checkpoint_path"])
    receipt = _read_json(execution["receipt_path"])
    if not isinstance(result, dict):
        errors.append("result is missing")
    else:
        errors.extend(adapter_abi.validate_result(result, request, registry,
                                                  verify_files=True))
    if not isinstance(checkpoint, dict):
        errors.append("exact-resume checkpoint is missing")
    else:
        errors.extend(adapter_abi.validate_checkpoint(checkpoint, request, registry,
                                                      verify_files=True))
    if not isinstance(receipt, dict):
        errors.append("execution receipt is missing")
    elif isinstance(result, dict):
        errors.extend(adapter_abi.validate_execution_receipt(
            receipt, request, result, registry,
            checkpoint=checkpoint if isinstance(checkpoint, dict) else None,
            command_argv=execution.get("command"),
        ))
    if isinstance(result, dict):
        metrics = result.get("metrics", {})
        lifecycle = metrics.get("lifecycle") if isinstance(metrics, dict) else None
        if not isinstance(lifecycle, dict):
            errors.append("result omits the required lifecycle proof")
        else:
            if lifecycle.get("parent_source_deleted") is not False:
                errors.append("result does not prove parent-source preservation")
            if lifecycle.get("dense_reconstruction_ephemeral") is not True:
                errors.append("result does not declare dense reconstruction ephemeral")
            if lifecycle.get("dense_reconstruction_removed") is not True:
                errors.append("ephemeral dense reconstruction was not removed")
            if execution["cell"]["branch"] == "codec_control" \
                    and lifecycle.get("packed_base_retained") is not True:
                errors.append("codec result does not prove packed-base retention")
        if not isinstance(metrics, dict) or metrics.get("completed_replicates") != 1 \
                or metrics.get("replicate_scope") \
                != "preliminary_scale_mapping_not_dominance":
            errors.append("result does not prove the declared preliminary replicate scope")
        inventory_rows, inventory_errors = _packed_inventory(
            execution["cell"], result, verify_live=False
        )
        errors.extend(inventory_errors)
        if not inventory_rows:
            errors.append("result has no packed payload deletion allowlist")
    return (result if isinstance(result, dict) else None,
            receipt if isinstance(receipt, dict) else None, errors)


def _packed_inventory(cell: dict[str, Any], result: dict[str, Any], *,
                      verify_live: bool) -> tuple[list[dict[str, Any]], list[str]]:
    """Return the adapter-authored exact packed-payload deletion allowlist."""
    errors: list[str] = []
    metrics = result.get("metrics")
    inventory = metrics.get("packed_artifact_inventory") \
        if isinstance(metrics, dict) else None
    if not isinstance(inventory, dict) \
            or inventory.get("schema") != "hawking.doctor_v5_packed_artifact_inventory.v1" \
            or inventory.get("branch") != cell["branch"] \
            or inventory.get("deletion_allowlist_exact") is not True \
            or inventory.get("parent_source_shards_included") is not False:
        return [], ["packed artifact inventory policy/identity is invalid"]
    rows = inventory.get("artifacts")
    if not isinstance(rows, list) or not rows:
        return [], ["packed artifact inventory is empty or invalid"]
    output_rows = result.get("output_artifacts")
    output_identities = {
        (row.get("role"), row.get("path"), row.get("sha256"), row.get("bytes"))
        for row in output_rows if isinstance(row, dict)
    } if isinstance(output_rows, list) else set()
    normalized: list[dict[str, Any]] = []
    roles: set[str] = set()
    root = (RESULTS / cell["cell_id"]).resolve()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"role", "path", "sha256", "bytes"}:
            errors.append(f"packed inventory row[{index}] keys invalid")
            continue
        role, raw, digest, size = (row.get("role"), row.get("path"),
                                   row.get("sha256"), row.get("bytes"))
        if not isinstance(role, str) or not role.startswith("bundle_shard:") \
                or role in roles:
            errors.append(f"packed inventory row[{index}] role invalid")
            continue
        roles.add(role)
        if not isinstance(raw, str) or not raw.endswith(".strand") \
                or not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None \
                or isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            errors.append(f"packed inventory row[{index}] identity invalid")
            continue
        try:
            path = Path(raw)
            path = path.resolve(strict=verify_live)
            path.relative_to(root)
            relative = path.relative_to(root).parts
            if len(relative) < 4 or relative[0] not in {"strand_ladder", "qwen_treatment"} \
                    or tuple(relative[1:3]) != ("bundle", "shards"):
                raise CampaignError("packed artifact is outside the exact worker bundle")
            if verify_live:
                info = path.lstat()
                if path.is_symlink() or not stat.S_ISREG(info.st_mode):
                    raise CampaignError("packed artifact is not a regular non-symlink file")
                observed_sha, observed_size = _sha_file(path)
                if observed_sha != digest or observed_size != size:
                    raise CampaignError("packed artifact live identity mismatch")
        except (CampaignError, OSError, ValueError) as exc:
            errors.append(f"packed inventory row[{index}] path invalid: {exc}")
            continue
        identity = (role, raw, digest, size)
        if identity not in output_identities:
            errors.append(f"packed inventory row[{index}] is absent from sealed result")
        normalized.append({"role": role, "path": raw, "sha256": digest, "bytes": size})
    if normalized != sorted(normalized, key=lambda row: row["role"]):
        errors.append("packed artifact inventory is not canonically ordered")
    if inventory.get("artifact_count") != len(normalized) \
            or inventory.get("total_bytes") != sum(row["bytes"] for row in normalized):
        errors.append("packed artifact inventory count/byte total mismatch")
    return normalized, errors


def _live_cell_payload_bytes(cell: dict[str, Any]) -> int:
    result = _read_json(RESULTS / cell["cell_id"] / "result.json")
    if not isinstance(result, dict):
        return 0
    rows, errors = _packed_inventory(cell, result, verify_live=False)
    if errors:
        return 0
    total = 0
    for row in rows:
        path = Path(row["path"])
        try:
            info = path.lstat()
            if not path.is_symlink() and stat.S_ISREG(info.st_mode) \
                    and info.st_size == row["bytes"]:
                total += info.st_size
        except OSError:
            continue
    return total


def _current_output_bytes(cell: dict[str, Any]) -> int:
    root = RESULTS / cell["cell_id"]
    total = 0
    for internal in ("strand_ladder", "qwen_treatment"):
        shard_root = root / internal / "bundle" / "shards"
        if not shard_root.is_dir() or shard_root.is_symlink():
            continue
        for path in shard_root.glob("*.strand"):
            try:
                info = path.lstat()
                if not path.is_symlink() and stat.S_ISREG(info.st_mode):
                    total += info.st_size
            except OSError:
                continue
    return total


def _execution_resource_gate(plan: dict[str, Any], state: dict[str, Any],
                             execution: dict[str, Any]) -> dict[str, Any]:
    cell = execution["cell"]
    residents = [candidate for candidate in plan["cells"]
                 if candidate["model_label"] == cell["model_label"]
                 and state["cells"][candidate["cell_id"]]["status"] == "complete"]
    resident_bytes = sum(_live_cell_payload_bytes(candidate) for candidate in residents)
    predecessor_ids = set(cell["dependencies"])
    predecessor_bytes = sum(_live_cell_payload_bytes(candidate) for candidate in residents
                            if candidate["cell_id"] in predecessor_ids)
    produced = _current_output_bytes(cell)
    projected_remaining = max(0, cell["projected_output_bytes"] - produced)
    gate = _resource_gate(
        execution["scratch_bytes"], projected_output_bytes=projected_remaining,
        resident_payload_bytes=resident_bytes,
        resident_predecessor_bytes=predecessor_bytes,
    )
    gate["cell_id"] = cell["cell_id"]
    gate["projected_whole_output_bytes"] = cell["projected_output_bytes"]
    gate["observed_current_output_bytes"] = produced
    return gate


def _projected_residency_bytes(cell: dict[str, Any]) -> int:
    """Projected peak process-tree residency for one cell (admission estimate).

    Calibration: 0.5B source ~1 GB -> 9 + 2 = 11 GB, matching the MEASURED
    ~10.5 GB peak.  14B/32B/72B/120B clamp near the budget and self-limit to a
    single lane.  Every tier except 0.5B is INFERRED until its first cell runs.
    """
    manifest = cell["parameter_manifest"]
    if cell["admission"]["whole_parent_residency_assumed"]:   # <=16B tiers.
        projected = RESIDENCY_BASE_WORKING_BYTES + math.ceil(
            manifest["source_weight_bytes"] * RESIDENCY_DENSE_FACTOR
        )
    else:                                                     # streaming >16B.
        projected = RESIDENCY_BASE_WORKING_BYTES + math.ceil(
            manifest["largest_source_shard_bytes"] * RESIDENCY_SHARD_FACTOR
        )
    projected = max(RESIDENCY_FLOOR_BYTES, projected)
    # Clamp so a single largest cell ALWAYS admits into an empty pool.
    return min(PROCESS_BUDGET_BYTES - SAFETY_MARGIN_BYTES, projected)


def _seed_observed_tier_residency(plan: dict[str, Any]) -> None:
    """Seed MEASURED per-tier residency once from the persisted child log."""
    _OBSERVED_TIER_RSS.clear()
    if not CHILD_RESOURCE_LOG.exists():
        return
    label_by_cell = {cell["cell_id"]: cell["model_label"] for cell in plan["cells"]}
    try:
        with CHILD_RESOURCE_LOG.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                label = label_by_cell.get(row.get("cell_id"))
                value = row.get("tree_rss_bytes")
                if label is None or isinstance(value, bool) \
                        or not isinstance(value, int) or value < 0:
                    continue
                if value > _OBSERVED_TIER_RSS.get(label, 0):
                    _OBSERVED_TIER_RSS[label] = value
    except OSError:
        return


def _observed_tier_residency(model_label: str) -> int | None:
    """Max tree RSS MEASURED for this tier so far, or None if never measured."""
    value = _OBSERVED_TIER_RSS.get(model_label)
    return value if isinstance(value, int) and value > 0 else None


def _cell_reservation(cell: dict[str, Any]) -> int:
    """RAM the pool charges for a cell: max(projection, MEASURED tier peak).

    Self-calibrating: as soon as one cell of a tier has recorded a tree RSS the
    reservation becomes MEASURED, using data the pipeline already persists.
    """
    projected = _projected_residency_bytes(cell)
    observed = _observed_tier_residency(cell["model_label"])
    if observed is not None:
        projected = max(projected, observed)
    return max(RESIDENCY_FLOOR_BYTES,
               min(PROCESS_BUDGET_BYTES - SAFETY_MARGIN_BYTES, projected))


def _reserved_total(live_cells: dict[str, "LiveCell"]) -> int:
    """Live reservation sum, recomputed on demand so there is no drift counter."""
    return sum(live.reserved_bytes for live in live_cells.values())


def _validate_disposition(cell: dict[str, Any], plan: dict[str, Any]) \
        -> tuple[dict[str, Any] | None, list[str]]:
    path = _resolve_workspace_path(cell["disposition_path"], must_exist=False)
    if not path.exists():
        return None, []
    doc = _read_json(path)
    required = {
        "schema", "version", "plan_sha256", "cell_id", "cell_identity_sha256",
        "status", "reason_code", "detail", "evidence_artifacts", "recorded_at",
        "quality_claims_permitted", "source_deletion_permitted", "disposition_sha256",
    }
    errors: list[str] = []
    if not isinstance(doc, dict) or set(doc) != required:
        return None, ["disposition keys are invalid"]
    if doc.get("schema") != DISPOSITION_SCHEMA or doc.get("version") != VERSION:
        errors.append("disposition schema/version mismatch")
    if doc.get("plan_sha256") != plan["plan_sha256"] \
            or doc.get("cell_id") != cell["cell_id"] \
            or doc.get("cell_identity_sha256") != cell["cell_identity_sha256"]:
        errors.append("disposition campaign/cell binding mismatch")
    if doc.get("status") not in {"negative", "unsupported"}:
        errors.append("disposition status must be negative or unsupported")
    if not isinstance(doc.get("reason_code"), str) or not doc["reason_code"] \
            or not isinstance(doc.get("detail"), str) or not doc["detail"]:
        errors.append("disposition reason is incomplete")
    if doc.get("quality_claims_permitted") is not False \
            or doc.get("source_deletion_permitted") is not False:
        errors.append("disposition safety boundary invalid")
    evidence = doc.get("evidence_artifacts")
    if not isinstance(evidence, list):
        errors.append("disposition evidence list invalid")
    else:
        _, evidence_errors = _validate_runtime_inputs(evidence) if evidence else ([], [])
        errors.extend(evidence_errors)
    if doc.get("disposition_sha256") != _hash_value(_without(doc, "disposition_sha256")):
        errors.append("disposition hash mismatch")
    return doc, errors


def _dependency_state(cell: dict[str, Any], state: dict[str, Any]) \
        -> tuple[bool, list[str]]:
    blockers: list[str] = []
    for dependency in cell["dependencies"]:
        status = state["cells"][dependency]["status"]
        if status != "complete":
            blockers.append(f"dependency {dependency} is {status}")
    return not blockers, blockers


def _scan(plan: dict[str, Any], state: dict[str, Any]) \
        -> tuple[dict[str, Any] | None, dict[str, list[str]]]:
    blockers: dict[str, list[str]] = {}
    for cell in plan["cells"]:
        row = state["cells"][cell["cell_id"]]
        if row["status"] in TERMINAL or row["status"] == "running":
            continue
        disposition, disposition_errors = _validate_disposition(cell, plan)
        if disposition_errors:
            row["blockers"] = disposition_errors
            blockers[cell["cell_id"]] = disposition_errors
            continue
        if isinstance(disposition, dict):
            row.update({
                "status": disposition["status"], "completed_at": _now(),
                "disposition_sha256": disposition["disposition_sha256"],
                "blockers": [], "error": None,
            })
            _append_event("cell-disposition", cell_id=cell["cell_id"],
                          status=disposition["status"],
                          disposition_sha256=disposition["disposition_sha256"])
            _save_state(plan, state, "running")
            _sync_reporter(plan, state, reason=f"terminal:{cell['cell_id']}")
            continue
        if row["status"] == "blocked-execution" \
                and row["attempts"] >= MAX_AUTOMATIC_ATTEMPTS:
            retained = row["blockers"] or [
                f"automatic retry ceiling reached ({MAX_AUTOMATIC_ATTEMPTS})"
            ]
            row["blockers"] = retained
            blockers[cell["cell_id"]] = retained
            continue
        ready, dependency_blockers = _dependency_state(cell, state)
        if not ready:
            row["status"] = "blocked-dependency"
            row["blockers"] = dependency_blockers
            blockers[cell["cell_id"]] = dependency_blockers
            continue
        row["status"] = "pending"
        execution = _prepare_execution(cell)
        if execution["blockers"]:
            row["blockers"] = execution["blockers"]
            blockers[cell["cell_id"]] = execution["blockers"]
            continue
        row["blockers"] = []
        return execution, blockers
    return None, blockers


def _scan_runnable_heads(plan: dict[str, Any], state: dict[str, Any]) \
        -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Generalized _scan: collect EVERY runnable chain-head, not just the first.

    Body is identical to _scan (same disposition sealing, blocked-dependency
    marking, _prepare_execution, per-cell blocker recording) except it appends
    each runnable execution and continues.  Because the linear dependency chain
    makes at most one branch per (model, rate) runnable at once, distinct heads
    come from distinct GC-chains, so the control->static->conditional->full
    serialization and cross-rate independence fall straight out unchanged.
    """
    blockers: dict[str, list[str]] = {}
    heads: list[dict[str, Any]] = []
    for cell in plan["cells"]:
        row = state["cells"][cell["cell_id"]]
        if row["status"] in TERMINAL or row["status"] == "running":
            continue
        if cell["cell_id"] in state["active_children"]:
            continue  # Belt-and-suspenders: a launched lane is never re-admitted.
        disposition, disposition_errors = _validate_disposition(cell, plan)
        if disposition_errors:
            row["blockers"] = disposition_errors
            blockers[cell["cell_id"]] = disposition_errors
            continue
        if isinstance(disposition, dict):
            row.update({
                "status": disposition["status"], "completed_at": _now(),
                "disposition_sha256": disposition["disposition_sha256"],
                "blockers": [], "error": None,
            })
            _append_event("cell-disposition", cell_id=cell["cell_id"],
                          status=disposition["status"],
                          disposition_sha256=disposition["disposition_sha256"])
            _save_state(plan, state, "running")
            _sync_reporter(plan, state, reason=f"terminal:{cell['cell_id']}")
            continue
        if row["status"] == "blocked-execution" \
                and row["attempts"] >= MAX_AUTOMATIC_ATTEMPTS:
            retained = row["blockers"] or [
                f"automatic retry ceiling reached ({MAX_AUTOMATIC_ATTEMPTS})"
            ]
            row["blockers"] = retained
            blockers[cell["cell_id"]] = retained
            continue
        ready, dependency_blockers = _dependency_state(cell, state)
        if not ready:
            row["status"] = "blocked-dependency"
            row["blockers"] = dependency_blockers
            blockers[cell["cell_id"]] = dependency_blockers
            continue
        row["status"] = "pending"
        execution = _prepare_execution(cell)
        if execution["blockers"]:
            row["blockers"] = execution["blockers"]
            blockers[cell["cell_id"]] = execution["blockers"]
            continue
        row["blockers"] = []
        heads.append(execution)
    return heads, blockers


def _acquire_heavy_lease() -> Any | None:
    HEAVY_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lease = HEAVY_LOCK.open("a+")
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        return None
    return lease


def _open_child_lease() -> IO[Any]:
    """Per-child HEAVY_LOCK fd, NOT flocked.

    flock is advisory and per open-file-description, so N children can each hold
    their own non-flocked fd to the same HEAVY_LOCK inode and every worker
    _validate_heavy_lease (which only os.fstats the inherited fd and compares
    st_dev/st_ino) still passes.  The fd VALUE is placed only in the environment,
    never in command_sha256 or any receipt, so the flock becoming pure in-memory
    accounting is invisible to every integrity hash.
    """
    HEAVY_LOCK.parent.mkdir(parents=True, exist_ok=True)
    return HEAVY_LOCK.open("a+")


def _process_group_rows(pgid: int) -> list[dict[str, Any]]:
    try:
        process = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid=,rss=,state="], capture_output=True,
            text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise CampaignError(f"cannot sample child process group: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for raw in process.stdout.splitlines():
        fields = raw.split()
        if len(fields) != 5:
            continue
        try:
            pid, ppid, observed_pgid, rss_kib = map(int, fields[:4])
        except ValueError:
            continue
        if observed_pgid == pgid:
            rows.append({"pid": pid, "ppid": ppid, "pgid": observed_pgid,
                         "rss_bytes": rss_kib * 1024, "state": fields[4]})
    return sorted(rows, key=lambda row: row["pid"])


def _sample_child_tree(root_pid: int, expected_pgid: int,
                       expected_identity: tuple[str, str]) -> dict[str, Any]:
    if expected_pgid != root_pid or _process_identity(root_pid) != expected_identity:
        raise CampaignError("child root process identity changed during resource monitoring")
    try:
        live_pgid = os.getpgid(root_pid)
    except ProcessLookupError as exc:
        raise CampaignError("child root vanished during resource monitoring") from exc
    if live_pgid != expected_pgid:
        raise CampaignError("child process group identity changed")
    rows = _process_group_rows(expected_pgid)
    if not any(row["pid"] == root_pid for row in rows) \
            or any(row["pgid"] != expected_pgid for row in rows):
        raise CampaignError("child process tree sample does not contain the bound root")
    total = sum(row["rss_bytes"] for row in rows)
    return {"sampled_at": _now(), "root_pid": root_pid, "pgid": expected_pgid,
            "process_count": len(rows), "tree_rss_bytes": total, "processes": rows}


def _stable_child_identity(process: subprocess.Popen[Any], *, expected_pgid: int,
                           required_argv_tokens: tuple[str, ...],
                           timeout_seconds: float = 15.0) -> tuple[str, str] | None:
    """Wait past macOS' Python launcher exec transition and bind stable argv/start."""
    if expected_pgid != process.pid or not required_argv_tokens \
            or any(not isinstance(token, str) or not token for token in required_argv_tokens):
        raise CampaignError("child identity handshake inputs are invalid")
    deadline = time.monotonic() + timeout_seconds
    previous: tuple[str, str] | None = None
    stable_samples = 0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return None
        try:
            if os.getpgid(process.pid) != expected_pgid:
                raise CampaignError("spawned adapter changed process group during handshake")
        except ProcessLookupError:
            if process.poll() is not None:
                return None
            raise CampaignError("spawned adapter vanished during identity handshake")
        identity = _process_identity(process.pid)
        if identity is not None \
                and all(token in identity[0] for token in required_argv_tokens):
            if identity == previous:
                stable_samples += 1
            else:
                previous = identity
                stable_samples = 1
            if stable_samples >= 2:
                return identity
        else:
            previous = None
            stable_samples = 0
        time.sleep(0.1)
    raise CampaignError("spawned adapter never reached a stable source-bound identity")


def _append_child_resource_sample(row: dict[str, Any]) -> None:
    CHILD_RESOURCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with CHILD_RESOURCE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(_canonical(row).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_dir(CHILD_RESOURCE_LOG.parent)


def _terminate(process: subprocess.Popen[Any], *, expected_pgid: int,
               expected_identity: tuple[str, str] | None) -> None:
    if expected_pgid != process.pid:
        raise CampaignError("refusing to signal a child with an unbound PGID")
    root_live = process.poll() is None
    if root_live:
        observed_identity = _process_identity(process.pid)
        if expected_identity is not None and observed_identity is not None \
                and observed_identity[1] != expected_identity[1]:
            raise CampaignError("refusing to signal a reused child PID/start identity")
        try:
            if os.getpgid(process.pid) != expected_pgid:
                raise CampaignError("refusing to signal a child with changed PGID")
        except ProcessLookupError:
            root_live = False
    members = [row for row in _process_group_rows(expected_pgid)
               if not str(row.get("state", "")).startswith("Z")]
    if not members:
        if root_live:
            process.wait(timeout=5)
        return
    if any(row["pgid"] != expected_pgid for row in members):
        raise CampaignError("refusing to signal an unverified child process group")
    try:
        os.killpg(expected_pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        members = [row for row in _process_group_rows(expected_pgid)
                   if not str(row.get("state", "")).startswith("Z")]
        if not members:
            break
        time.sleep(0.1)
    members = [row for row in _process_group_rows(expected_pgid)
               if not str(row.get("state", "")).startswith("Z")]
    if members:
        if any(row["pgid"] != expected_pgid for row in members):
            raise CampaignError("refusing SIGKILL without the verified child PGID")
        try:
            os.killpg(expected_pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        kill_deadline = time.monotonic() + 15
        while any(not str(row.get("state", "")).startswith("Z")
                  for row in _process_group_rows(expected_pgid)) \
                and time.monotonic() < kill_deadline:
            time.sleep(0.1)
        if any(not str(row.get("state", "")).startswith("Z")
               for row in _process_group_rows(expected_pgid)):
            raise CampaignError("child process group survived verified SIGKILL")
    if process.poll() is None:
        process.wait(timeout=5)


def _reap_recorded_child(record: Any, state: dict[str, Any]) -> None:
    """Reap ONE process group durably recorded before a prior supervisor died.

    Byte-for-byte the same identity/PGID validation and SIGTERM-then-SIGKILL
    escalation the singular recovery used, factored out so the plural recovery
    can loop it over every active_children entry.
    """
    if not isinstance(record, dict):
        raise CampaignError("recorded active child is invalid")
    pid, pgid, started = record.get("pid"), record.get("pgid"), record.get(
        "process_started"
    )
    cell_id = record.get("cell_id")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1 \
            or pgid != pid or not isinstance(started, str) or not started \
            or cell_id not in state["cells"]:
        raise CampaignError("recorded active child identity is invalid")
    members = [row for row in _process_group_rows(pgid)
               if not str(row.get("state", "")).startswith("Z")]
    if members:
        identity = _process_identity(pid)
        if identity is not None:
            if identity[1] != started:
                raise CampaignError("refusing to reap a reused recorded child PID")
            try:
                if os.getpgid(pid) != pgid:
                    raise CampaignError("recorded active child PGID changed")
            except ProcessLookupError:
                identity = None
        elif any(row["pid"] == pid for row in members):
            raise CampaignError(
                "recorded child root is live but its start identity is unreadable"
            )
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            members = []
        deadline = time.monotonic() + 15
        while members and time.monotonic() < deadline:
            time.sleep(0.1)
            members = [row for row in _process_group_rows(pgid)
                       if not str(row.get("state", "")).startswith("Z")]
        if members:
            os.killpg(pgid, signal.SIGKILL)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                members = [row for row in _process_group_rows(pgid)
                           if not str(row.get("state", "")).startswith("Z")]
                if not members:
                    break
                time.sleep(0.1)
        if members:
            raise CampaignError("recorded active child group survived recovery kill")
    row = state["cells"][cell_id]
    if row["status"] == "running":
        row["status"] = "pending"
    _append_event("recorded-child-reaped", cell_id=cell_id, pid=pid, pgid=pgid,
                  handshake_pending=record.get("handshake_pending"))


def _reap_recorded_children(plan: dict[str, Any], state: dict[str, Any]) -> int:
    """Reap every process group recorded before a prior supervisor died."""
    records = state.get("active_children")
    if not isinstance(records, dict):
        raise CampaignError("recorded active children are invalid")
    reaped = 0
    for record in list(records.values()):
        _reap_recorded_child(record, state)
        reaped += 1
    if records:
        state["active_children"] = {}
        _sync_active_cells(state)
        _save_state(plan, state, "running")
    return reaped


def _reset_orphaned_running_cells(plan: dict[str, Any], state: dict[str, Any]) -> int:
    """One-time startup sweep: reset every cell left status=='running' that has no
    live child recorded in active_children back to 'pending'.

    Admission flips a row to 'running' and PERSISTS it before _launch_cell records
    the child under active_children (which happens only after an up-to-5s spawn
    identity handshake), and a raise on the completion path (e.g. an ADOPT_RC
    revalidation failure, or a _commit_complete save routed through _abort_pool)
    can leave a row 'running' after _release_cell already dropped its child.  In
    both cases the cell is 'running' with nothing alive: _reap_recorded_children
    only revisits active_children entries, and _scan/_scan_runnable_heads skip
    status=='running', so the cell (and every GC-chain successor that depends on
    it) would stall forever.  This runs after the reap, when the supervisor holds
    no live children, so any remaining 'running' row is provably orphaned.
    """
    active = state.get("active_children")
    if not isinstance(active, dict):
        raise CampaignError("recorded active children are invalid")
    reset: list[str] = []
    for cell_id, row in state["cells"].items():
        if row.get("status") == "running" and cell_id not in active:
            row["status"] = "pending"
            row["blockers"] = []
            reset.append(cell_id)
    if reset:
        _sync_active_cells(state)
        for cell_id in reset:
            _append_event("orphaned-running-reset", cell_id=cell_id)
        _save_state(plan, state, "running")
    return len(reset)


def _control_responsive_wait(plan: dict[str, Any], seconds: float) -> None:
    """Wait without delaying a pause/drain observation beyond the control cadence."""
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        time.sleep(min(CONTROL_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
        if _STOP or _load_control(plan)["mode"] != "run":
            return


def _resource_stop_receipt(plan: dict[str, Any], execution: dict[str, Any], *, reason: str,
                           sample: dict[str, Any], max_rss_bytes: int,
                           process_identity: tuple[str, str]) -> dict[str, Any]:
    checkpoint = None
    if execution["checkpoint_path"].is_file():
        digest, size = _sha_file(execution["checkpoint_path"])
        checkpoint = {"path": _relative(execution["checkpoint_path"]),
                      "sha256": digest, "bytes": size}
    receipt: dict[str, Any] = {
        "schema": "hawking.doctor_v5_ultra_resource_stop.v1", "version": VERSION,
        "plan_sha256": plan["plan_sha256"],
        "cell_id": execution["cell"]["cell_id"],
        "cell_identity_sha256": execution["cell"]["cell_identity_sha256"],
        "request_sha256": execution["request"]["request_sha256"],
        "reason": reason, "process_budget_bytes": PROCESS_BUDGET_BYTES,
        "trigger_sample": sample, "max_child_tree_rss_bytes": max_rss_bytes,
        "process_identity": {
            "command_sha256": hashlib.sha256(
                process_identity[0].encode("utf-8")
            ).hexdigest(),
            "started": process_identity[1], "pgid": sample["pgid"],
        },
        "checkpoint": checkpoint,
        "resume_policy": "retry_exact_checkpoint_after_resource_gate_recovers",
        "parent_source_deleted": False, "recorded_at": _now(),
    }
    receipt["receipt_sha256"] = _hash_value(receipt)
    path = execution["output_dir"] / "resource_stop.json"
    _atomic_json(path, receipt)
    digest, size = _sha_file(path)
    return {"path": _relative(path), "sha256": digest, "bytes": size,
            "receipt_sha256": receipt["receipt_sha256"], "reason": reason}


def _launch_cell(plan: dict[str, Any], state: dict[str, Any],
                 execution: dict[str, Any], reserved_bytes: int) -> "LiveCell | int":
    """Non-blocking launch of one cell child; the poll step advances it.

    Returns a LiveCell for a real running child (the caller charges its
    reservation) or an int terminal-at-launch outcome (ADOPT_RC, RESOURCE_RC,
    PAUSE_RC, 130, or an early real returncode) for which nothing is charged.
    The spawned command, environment keys/semantics, and both identity
    handshakes are byte-identical to the former serial _run_external.
    """
    cell = execution["cell"]
    # Mid-flight adopt: complete, source-bound artifacts need no subprocess.
    if execution["result_path"].exists() and execution["receipt_path"].exists():
        _, _, errors = _validate_outputs(execution)
        if not errors:
            return ADOPT_RC
    # Control gate once (NOT a spin loop); the pool loop owns the cadence.
    mode = _load_control(plan)["mode"]
    if _STOP or mode == "drain":
        return 130
    if mode == "pause":
        return PAUSE_RC
    # Disk/pressure/swap/thermal gate stays here, per cell, at launch.
    gate = _execution_resource_gate(plan, state, execution)
    state["last_resource_gate"] = gate
    if not gate["ok"]:
        return RESOURCE_RC
    command = execution["command"]
    if not isinstance(command, list):
        raise CampaignError("typed command invariant failed")
    lease = _open_child_lease()
    process: subprocess.Popen[Any] | None = None
    process_pgid: int | None = None
    spawn_identity: tuple[str, str] | None = None
    process_identity: tuple[str, str] | None = None
    live_pgid: int | None = None
    try:
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment[ram_scheduler.HEAVY_LEASE_FD_ENV] = str(lease.fileno())
        log_path = execution["output_dir"] / "execution.log"
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, shell=False,
                close_fds=True, env=environment, pass_fds=(lease.fileno(),),
            )
        process_pgid = process.pid
        identity_deadline = time.monotonic() + 5
        while time.monotonic() < identity_deadline and process.poll() is None:
            try:
                live_pgid = os.getpgid(process.pid)
            except ProcessLookupError:
                live_pgid = None
            spawn_identity = _process_identity(process.pid)
            if live_pgid == process_pgid and spawn_identity is not None:
                break
            time.sleep(0.05)
        if process.poll() is not None:
            lease.close()
            return int(process.returncode)
        if spawn_identity is None or live_pgid != process_pgid:
            raise CampaignError("spawned adapter process start/PGID is invalid")
        if len(command) < 2 or str(execution["request_path"]) not in command:
            raise CampaignError("typed adapter command omits adapter/request argv binding")
        state["active_children"][cell["cell_id"]] = {
            "cell_id": cell["cell_id"], "pid": process.pid, "pgid": process_pgid,
            "command_sha256": _hash_value(command),
            "request_sha256": execution["request"]["request_sha256"],
            "started_at": _now(),
            "process_started": spawn_identity[1],
            "process_command_sha256": None, "handshake_pending": True,
            "process_budget_bytes": PROCESS_BUDGET_BYTES,
            "max_tree_rss_bytes": 0,
            "reserved_bytes": reserved_bytes,
        }
        _sync_active_cells(state)
        _save_state(plan, state, "running-cell")
        process_identity = _stable_child_identity(
            process, expected_pgid=process_pgid,
            required_argv_tokens=(command[1], str(execution["request_path"])),
        )
        if process_identity is None:
            state["active_children"].pop(cell["cell_id"], None)
            _sync_active_cells(state)
            _save_state(plan, state)
            lease.close()
            return int(process.returncode)
        if process_identity[1] != spawn_identity[1]:
            raise CampaignError("spawned adapter start identity changed during handshake")
        state["active_children"][cell["cell_id"]].update({
            "process_started": process_identity[1],
            "process_command_sha256": hashlib.sha256(
                process_identity[0].encode("utf-8")
            ).hexdigest(),
            "handshake_pending": False,
        })
        _save_state(plan, state, "running-cell")
        return LiveCell(
            execution=execution, process=process, process_pgid=process_pgid,
            spawn_identity=spawn_identity, process_identity=process_identity,
            lease=lease, reserved_bytes=reserved_bytes, max_tree_rss_bytes=0,
            # H3: back-date the first probe by one interval so a fresh lane
            # re-gates on swap/pressure/disk on its FIRST poll tick instead of
            # running unmonitored for a full RESOURCE_POLL_SECONDS.
            last_resource_probe=time.monotonic() - RESOURCE_POLL_SECONDS,
        )
    except BaseException:
        # Fail-closed: terminate and drop THIS child before propagating so a
        # single lane's identity/gate error never leaks a process or a lease.
        try:
            if process is not None and process_pgid is not None:
                _terminate(process, expected_pgid=process_pgid,
                           expected_identity=process_identity or spawn_identity)
        except BaseException:
            pass
        state["active_children"].pop(cell["cell_id"], None)
        _sync_active_cells(state)
        try:
            lease.close()
        except OSError:
            pass
        raise


def _child_tree_from_rows(live: "LiveCell",
                          rows_by_pgid: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
    """Per-child identity guards + tree RSS from one shared ps snapshot.

    Mirrors _sample_child_tree's guards exactly (fail-closed on any mismatch)
    but buckets a pre-taken snapshot instead of shelling out per child.
    """
    root_pid = live.process.pid
    expected_pgid = live.process_pgid
    expected_identity = live.process_identity
    if expected_pgid != root_pid or _process_identity(root_pid) != expected_identity:
        raise CampaignError("child root process identity changed during resource monitoring")
    try:
        observed_pgid = os.getpgid(root_pid)
    except ProcessLookupError as exc:
        raise CampaignError("child root vanished during resource monitoring") from exc
    if observed_pgid != expected_pgid:
        raise CampaignError("child process group identity changed")
    rows = sorted(rows_by_pgid.get(expected_pgid, []), key=lambda row: row["pid"])
    if not any(row["pid"] == root_pid for row in rows) \
            or any(row["pgid"] != expected_pgid for row in rows):
        raise CampaignError("child process tree sample does not contain the bound root")
    total = sum(row["rss_bytes"] for row in rows)
    return {"sampled_at": _now(), "root_pid": root_pid, "pgid": expected_pgid,
            "process_count": len(rows), "tree_rss_bytes": total, "processes": rows}


def _sample_active_children(live_cells: dict[str, "LiveCell"]) \
        -> tuple[dict[str, dict[str, Any]], int]:
    """One shared ps snapshot -> per-cell samples + aggregate tree RSS.

    A single snapshot gives a consistent cross-child instant for the aggregate
    and cuts N ps calls to 1.  Cells whose process is finishing are skipped;
    a live-process identity mismatch fails closed exactly as _sample_child_tree.
    """
    try:
        process = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,pgid=,rss=,state="], capture_output=True,
            text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise CampaignError(f"cannot sample child process group: {exc}") from exc
    rows_by_pgid: dict[int, list[dict[str, Any]]] = {}
    for raw in process.stdout.splitlines():
        fields = raw.split()
        if len(fields) != 5:
            continue
        try:
            pid, ppid, observed_pgid, rss_kib = map(int, fields[:4])
        except ValueError:
            continue
        rows_by_pgid.setdefault(observed_pgid, []).append(
            {"pid": pid, "ppid": ppid, "pgid": observed_pgid,
             "rss_bytes": rss_kib * 1024, "state": fields[4]}
        )
    samples: dict[str, dict[str, Any]] = {}
    aggregate = 0
    for cell_id, live in live_cells.items():
        if live.process.poll() is not None:
            continue  # Finishing; _finalize_cell handles it this tick.
        try:
            sample = _child_tree_from_rows(live, rows_by_pgid)
        except CampaignError:
            if live.process.poll() is not None:
                continue  # Raced a clean exit; finalize, do not fail closed.
            raise
        samples[cell_id] = sample
        aggregate += sample["tree_rss_bytes"]
    return samples, aggregate


def _record_resource_stop(state: dict[str, Any], cell_id: str, *,
                          sole_live: bool) -> str | None:
    """Bound a cell's resource-driven retries so the pool cannot admit -> OOM-kill
    -> reset it forever (which blocks campaign completion).

    A cell whose real residency alone reaches the RAM budget is shed as the SOLE
    live lane; retrying it in isolation can never succeed, so it escalates on that
    first stop.  A cell that merely loses repeated aggregate/pressure contention
    escalates after MAX_RESOURCE_STOPS consecutive resource-stops.  Escalation
    flips it to blocked-execution so the existing MAX_AUTOMATIC_ATTEMPTS ceiling
    surfaces it to a human instead of deadlocking the campaign.  The count is
    tracked in state and cleared when the cell later completes.  Returns the
    blocker string on escalation, else None (caller resets the row to pending).
    """
    counts = state.setdefault("resource_stop_counts", {})
    count = int(counts.get(cell_id, 0)) + 1
    counts[cell_id] = count
    if not (sole_live or count >= MAX_RESOURCE_STOPS):
        return None
    detail = ("its residency alone reaches the process RAM budget as the sole live lane"
              if sole_live else
              f"{count} consecutive resource-stops reached the bound ({MAX_RESOURCE_STOPS})")
    blocker = f"resource-stop ceiling reached: {detail}"
    row = state["cells"][cell_id]
    row["status"] = "blocked-execution"
    row["error"] = blocker
    row["blockers"] = [blocker]
    counts.pop(cell_id, None)
    _append_event("resource-stop-escalated", cell_id=cell_id,
                  consecutive_stops=count, sole_live=sole_live, blocker=blocker)
    return blocker


def _clear_resource_stop(state: dict[str, Any], cell_id: str) -> None:
    """Reset a cell's consecutive resource-stop counter after real progress."""
    counts = state.get("resource_stop_counts")
    if isinstance(counts, dict):
        counts.pop(cell_id, None)


def _enforce_pool_budget(plan: dict[str, Any], state: dict[str, Any],
                         live_cells: dict[str, "LiveCell"],
                         samples_by_cell: dict[str, dict[str, Any]],
                         aggregate: int) -> list[str]:
    """Record each sample, then stop victims while the SUMMED tree RSS is over budget.

    The hard OOM guard: a strict generalization of the old per-child check
    (tree_rss >= budget) to the aggregate, terminating the largest-RSS lane
    first for greatest relief until the sum drops under PROCESS_BUDGET_BYTES.
    """
    for cell_id, sample in samples_by_cell.items():
        live = live_cells.get(cell_id)
        if live is None:
            continue
        execution = live.execution
        sample.update({
            "cell_id": cell_id, "plan_sha256": plan["plan_sha256"],
            "request_sha256": execution["request"]["request_sha256"],
            "process_budget_bytes": PROCESS_BUDGET_BYTES,
        })
        live.max_tree_rss_bytes = max(live.max_tree_rss_bytes, sample["tree_rss_bytes"])
        sample["max_tree_rss_bytes"] = live.max_tree_rss_bytes
        sample["at_or_over_budget"] = aggregate >= PROCESS_BUDGET_BYTES
        _append_child_resource_sample(sample)
        state["last_child_resource_sample"] = sample
        state["max_child_tree_rss_bytes"] = max(
            state["max_child_tree_rss_bytes"], live.max_tree_rss_bytes
        )
        if cell_id in state["active_children"]:
            state["active_children"][cell_id]["max_tree_rss_bytes"] = live.max_tree_rss_bytes
        label = execution["cell"]["model_label"]
        if sample["tree_rss_bytes"] > _OBSERVED_TIER_RSS.get(label, 0):
            _OBSERVED_TIER_RSS[label] = sample["tree_rss_bytes"]  # MEASURED calibration.
    stopped: list[str] = []
    while aggregate >= PROCESS_BUDGET_BYTES and live_cells:
        victim_id = max(
            (cid for cid in samples_by_cell if cid in live_cells),
            key=lambda cid: samples_by_cell[cid]["tree_rss_bytes"], default=None,
        )
        if victim_id is None:
            break
        victim = live_cells[victim_id]
        sample = samples_by_cell[victim_id]
        _terminate(victim.process, expected_pgid=victim.process_pgid,
                   expected_identity=victim.process_identity)
        stop = _resource_stop_receipt(
            plan, victim.execution,
            reason="pool_tree_rss_at_or_over_process_budget",
            sample=sample, max_rss_bytes=victim.max_tree_rss_bytes,
            process_identity=victim.process_identity,
        )
        state["last_resource_stop"] = stop
        _append_event("resource-stop", cell_id=victim_id, **stop)
        # M2: a cell shed as the SOLE live lane (its residency alone reaches the
        # budget) or that keeps losing aggregate contention must not be re-admitted
        # forever; escalate it once the resource-stop ceiling is reached.
        sole_live = len(live_cells) == 1
        if _record_resource_stop(state, victim_id, sole_live=sole_live) is None:
            state["cells"][victim_id]["status"] = "pending"
        _release_cell(state, live_cells, victim)
        aggregate -= sample["tree_rss_bytes"]
        del samples_by_cell[victim_id]
        stopped.append(victim_id)
    # H3: the fixed PROCESS_BUDGET_BYTES aggregate is blind to swap and to
    # non-child residency (a co-resident 'mop' process + the OS share the box).
    # Take ONE shared snapshot per tick from the SAME source the admission/execution
    # gate uses and, under real pressure or swap above the stale-remnant tolerance,
    # shed the single largest-RSS lane for incremental relief.
    # Fail safe: an unreadable snapshot is treated as pressure so the guard never
    # goes blind (a conservative single-lane shed, mirroring the gate's fail-closed
    # swap/pressure fields).
    if live_cells and samples_by_cell:
        try:
            snapshot = ram_scheduler.resource_snapshot(str(ROOT))
        except Exception as exc:
            snapshot = {"error": f"{type(exc).__name__}: {exc}"}
        pressure = snapshot.get("pressure_level")
        swap_mb = snapshot.get("swap_used_mb")
        swap_nonzero = (
            isinstance(swap_mb, bool) or not isinstance(swap_mb, (int, float))
            or not math.isfinite(float(swap_mb))
            or float(swap_mb) > SWAP_TOLERANCE_MB
        )
        under_pressure = "error" in snapshot or pressure != 1 or swap_nonzero
        if under_pressure:
            victim_id = max(
                (cid for cid in samples_by_cell if cid in live_cells),
                key=lambda cid: samples_by_cell[cid]["tree_rss_bytes"], default=None,
            )
            if victim_id is not None:
                victim = live_cells[victim_id]
                sample = samples_by_cell[victim_id]
                _terminate(victim.process, expected_pgid=victim.process_pgid,
                           expected_identity=victim.process_identity)
                stop = _resource_stop_receipt(
                    plan, victim.execution,
                    reason="system_memory_pressure_or_swap",
                    sample=sample, max_rss_bytes=victim.max_tree_rss_bytes,
                    process_identity=victim.process_identity,
                )
                state["last_resource_stop"] = stop
                _append_event("resource-stop", cell_id=victim_id, **stop)
                sole_live = len(live_cells) == 1
                if _record_resource_stop(state, victim_id, sole_live=sole_live) is None:
                    state["cells"][victim_id]["status"] = "pending"
                _release_cell(state, live_cells, victim)
                del samples_by_cell[victim_id]
                stopped.append(victim_id)
    if stopped:
        _save_state(plan, state, "waiting-resources")
    return stopped


def _release_cell(state: dict[str, Any], live_cells: dict[str, "LiveCell"],
                  live: "LiveCell") -> None:
    """Close the (non-flocked) lease and drop the cell from every live index.

    No fcntl: there is no flock to release.  reserved_total is always recomputed
    from live_cells, so a cell MUST leave live_cells in the same step its lease
    closes; every terminate path funnels through here to avoid leaking budget.
    """
    cell_id = live.execution["cell"]["cell_id"]
    try:
        live.lease.close()
    except OSError:
        pass
    state["active_children"].pop(cell_id, None)
    live_cells.pop(cell_id, None)
    _sync_active_cells(state)


def _apply_cell_exit(plan: dict[str, Any], state: dict[str, Any],
                     execution: dict[str, Any], rc: int,
                     live_cells: dict[str, "LiveCell"] | None = None) -> None:
    """Serial-tail rc handling for a cell that produced a terminal outcome.

    live_cells (the OTHER still-running lanes, if any) is threaded through only so
    _commit_complete can keep them monitored across its blocking reporter sync
    (M1).  It is None for callers with no live pool in scope.
    """
    cell_id = execution["cell"]["cell_id"]
    row = state["cells"][cell_id]
    row["last_exit_code"] = rc
    if rc == ADOPT_RC:
        result, receipt, errors = _validate_outputs(execution)
        if errors or not isinstance(result, dict) or not isinstance(receipt, dict):
            raise CampaignError("adoptable artifacts changed during validation")
        _commit_complete(plan, state, execution, result, receipt, live_cells)
        return
    if rc == PAUSE_RC:
        # Pause/drain are re-observed at the loop top, which quiesces the pool.
        row["status"] = "pending"
        _save_state(plan, state)
        return
    if rc == RESOURCE_RC:
        row["status"] = "pending"
        _save_state(plan, state, "waiting-resources")
        return
    if rc == 130:
        row["status"] = "pending"
        _save_state(plan, state)
        return
    if rc != 0:
        row["status"] = "blocked-execution"
        row["error"] = f"typed adapter exited with status {rc}"
        row["blockers"] = [row["error"]]
        _append_event("cell-exit", cell_id=cell_id, exit_code=rc)
        _save_state(plan, state, "running")
        # Do not deadlock the campaign on one failed implementation.
        return
    result, receipt, errors = _validate_outputs(execution)
    if errors or not isinstance(result, dict) or not isinstance(receipt, dict):
        row["status"] = "blocked-execution"
        row["error"] = "invalid output artifacts: " + "; ".join(errors)
        row["blockers"] = list(errors)
        _save_state(plan, state, "running")
        return
    _commit_complete(plan, state, execution, result, receipt, live_cells)


def _finalize_cell(plan: dict[str, Any], state: dict[str, Any],
                   live_cells: dict[str, "LiveCell"], live: "LiveCell",
                   rc: int) -> None:
    """Reap the exited child, release it, then apply serial-tail rc handling."""
    try:
        _terminate(live.process, expected_pgid=live.process_pgid,
                   expected_identity=live.process_identity or live.spawn_identity)
    except CampaignError:
        pass  # Best-effort: the child already exited (poll() was not None).
    _release_cell(state, live_cells, live)
    # live_cells now holds only the OTHER lanes; pass it so a reporter sync on the
    # completion path keeps them budget-monitored + control-responsive (M1).
    _apply_cell_exit(plan, state, live.execution, rc, live_cells)


def _drain_pool(plan: dict[str, Any], state: dict[str, Any],
                live_cells: dict[str, "LiveCell"], *, target_status: str) -> None:
    """Terminate and release EVERY live child (drain/pause quiesce path).

    Each child is reaped and released even if a peer's terminate raises, so the
    pool is fully quiesced; the first error is re-raised afterward (fail-closed).
    """
    _append_event("pool-quiesce", target_status=target_status,
                  child_count=len(live_cells))
    pending_error: BaseException | None = None
    for cell_id in list(live_cells):
        live = live_cells[cell_id]
        try:
            _terminate(live.process, expected_pgid=live.process_pgid,
                       expected_identity=live.process_identity or live.spawn_identity)
        except BaseException as exc:
            if pending_error is None:
                pending_error = exc
        finally:
            row = state["cells"].get(cell_id)
            if row is not None and row["status"] == "running":
                row["status"] = "pending"
            _release_cell(state, live_cells, live)
    if pending_error is not None:
        raise pending_error


def _abort_pool(state: dict[str, Any], live_cells: dict[str, "LiveCell"]) -> None:
    """Best-effort terminate+release of every live child on a fatal error path."""
    for cell_id in list(live_cells):
        live = live_cells[cell_id]
        try:
            _terminate(live.process, expected_pgid=live.process_pgid,
                       expected_identity=live.process_identity or live.spawn_identity)
        except BaseException:
            pass
        finally:
            row = state["cells"].get(cell_id)
            if row is not None and row["status"] == "running":
                row["status"] = "pending"
            _release_cell(state, live_cells, live)


def _guard_live_pool(plan: dict[str, Any], state: dict[str, Any],
                     live_cells: dict[str, "LiveCell"] | None) -> None:
    """Re-monitor the still-live pool around a long BLOCKING call (M1).

    The completion path runs _sync_reporter synchronously (up to
    REPORTER_SYNC_TIMEOUT_SECONDS) while OTHER lanes keep running.  Bracketing
    that call with this guard means the live lanes are neither left over-budget
    (one fresh sample + the OOM/pressure guard sheds a lane that grew during the
    sync) nor left running dark against a stop request (control is re-observed and
    a requested drain/pause quiesces the pool immediately).  A lane that raced a
    clean exit is left for the poll loop to finalize, not failed closed here.
    """
    if not live_cells:
        return
    try:
        samples, aggregate = _sample_active_children(live_cells)
        _enforce_pool_budget(plan, state, live_cells, samples, aggregate)
    except CampaignError:
        pass  # A lane is finishing; the poll loop finalizes it next tick.
    mode = _load_control(plan)["mode"]
    state["control_mode"] = mode
    if (_STOP or mode == "drain") and live_cells:
        _drain_pool(plan, state, live_cells, target_status="drained")
    elif mode == "pause" and live_cells:
        _drain_pool(plan, state, live_cells, target_status="paused")


def _commit_complete(plan: dict[str, Any], state: dict[str, Any],
                     execution: dict[str, Any], result: dict[str, Any],
                     receipt: dict[str, Any],
                     live_cells: dict[str, "LiveCell"] | None = None) -> None:
    cell_id = execution["cell"]["cell_id"]
    row = state["cells"][cell_id]
    row.update({
        "status": "complete", "completed_at": _now(), "blockers": [], "error": None,
        "runtime_spec_sha256": execution["runtime"]["sha256"],
        "registry_sha256": execution["registry"]["registry_sha256"],
        "request_sha256": execution["request"]["request_sha256"],
        "result_sha256": result["result_sha256"],
        "execution_receipt_sha256": receipt["receipt_sha256"],
    })
    _clear_resource_stop(state, cell_id)  # Real progress resets the M2 counter.
    _append_event("cell-complete", cell_id=cell_id,
                  result_sha256=result["result_sha256"],
                  execution_receipt_sha256=receipt["receipt_sha256"])
    # active_cells is maintained by the pool via _sync_active_cells; do not pass
    # the singular active_cell kwarg here so a commit never clobbers the plural
    # bookkeeping of the other concurrently running lanes.
    _save_state(plan, state, "running")
    # M1: bracket the (up to 300s) reporter sync so the OTHER live lanes are not
    # left dark or over budget while it blocks the loop.
    _guard_live_pool(plan, state, live_cells)
    if _sync_reporter(plan, state, reason=f"terminal:{cell_id}"):
        _reconcile_lifecycle(plan, state, successor_only=execution["cell"])
    _guard_live_pool(plan, state, live_cells)


def run_queue(nonce: str) -> int:
    if NONCE_RE.fullmatch(nonce) is None:
        raise CampaignError("ownership nonce is invalid")
    plan = _load_plan()
    QUEUE_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with QUEUE_LOCK.open("a+") as singleton:
        try:
            fcntl.flock(singleton.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError("another Ultra supervisor holds the singleton lease") from exc
        # One supervisor-lifetime exclusive flock on HEAVY_LOCK preserves the
        # machine-wide "one heavy owner" guarantee against OTHER tools (studio_run,
        # processing_queue, audit_ladder).  Children get their own NON-flocked fds
        # (flock is per-OFD) so internal concurrency is governed purely by the RAM
        # budget while cross-tool exclusion is retained.
        heavy_owner = _acquire_heavy_lease()
        if heavy_owner is None:
            raise CampaignError("another heavy tool holds the studio heavy lease")
        live_cells: dict[str, LiveCell] = {}
        state: dict[str, Any] | None = None
        try:
            _atomic_json(PID_FILE, _pid_record(plan, nonce))
            state = _load_state(plan)
            _reap_recorded_children(plan, state)
            # H2: with no live children held, reset any cell orphaned in the
            # 'running' state (a crash in the launch window before it was recorded
            # in active_children, or a raise on the completion path after its child
            # was released) so it and its GC-chain successors do not stall forever.
            _reset_orphaned_running_cells(plan, state)
            state["supervisor_pid"] = os.getpid()
            state["error"] = None
            _save_state(plan, state, "running")
            _append_event("supervisor-start", pid=os.getpid())
            _sync_reporter(plan, state, reason="queue-start-or-resume")
            _reconcile_lifecycle(plan, state)
            _seed_observed_tier_residency(plan)
            while True:
                live_plan = _load_plan()
                if live_plan["plan_sha256"] != plan["plan_sha256"]:
                    raise CampaignError("campaign plan changed while supervisor was running")
                mode = _load_control(plan)["mode"]
                state["control_mode"] = mode
                if _STOP or mode == "drain":
                    _drain_pool(plan, state, live_cells, target_status="drained")
                    _save_state(plan, state, "drained", supervisor_pid=None,
                                active_children={}, active_cells=[])
                    _append_event("supervisor-drained", pid=os.getpid())
                    return 0
                if mode == "pause":
                    _drain_pool(plan, state, live_cells, target_status="paused")
                    _save_state(plan, state, "paused",
                                active_children={}, active_cells=[])
                    _control_responsive_wait(plan, CONTROL_POLL_SECONDS)
                    continue
                # ADVANCE EXISTING CHILDREN: one shared RSS snapshot, aggregate
                # OOM guard, per-lane disk re-gate, then finalize any that exited.
                if live_cells:
                    samples, aggregate = _sample_active_children(live_cells)
                    _enforce_pool_budget(plan, state, live_cells, samples, aggregate)
                    now = time.monotonic()
                    for cell_id in list(live_cells):
                        live = live_cells.get(cell_id)
                        if live is None:
                            continue
                        if now - live.last_resource_probe < RESOURCE_POLL_SECONDS:
                            continue
                        live.last_resource_probe = now
                        if live.process.poll() is not None:
                            continue
                        gate = _execution_resource_gate(plan, state, live.execution)
                        state["last_resource_gate"] = gate
                        if not gate["ok"]:
                            sample = samples.get(cell_id) or {"pgid": live.process_pgid}
                            _terminate(live.process, expected_pgid=live.process_pgid,
                                       expected_identity=live.process_identity)
                            stop = _resource_stop_receipt(
                                plan, live.execution,
                                reason="pressure_swap_power_thermal_or_disk_gate",
                                sample=sample, max_rss_bytes=live.max_tree_rss_bytes,
                                process_identity=live.process_identity,
                            )
                            state["last_resource_stop"] = stop
                            _append_event("resource-stop", cell_id=cell_id, **stop)
                            state["cells"][cell_id]["status"] = "pending"
                            _release_cell(state, live_cells, live)
                        else:
                            _save_state(plan, state)
                    for cell_id in list(live_cells):
                        live = live_cells.get(cell_id)
                        if live is None:
                            continue
                        rc = live.process.poll()
                        if rc is not None:
                            _finalize_cell(plan, state, live_cells, live, int(rc))
                # LIFECYCLE GC each tick (preserves GC-chain ordering + source
                # binding).  A missing reporter checkpoint retains payloads; it
                # never authorizes deletion or blocks independent cells.
                last_sync = state.get("last_reporter_sync")
                if _reconcile_lifecycle(plan, state) == 0 \
                        and isinstance(last_sync, dict) and last_sync.get("ok") is False:
                    if _sync_reporter(plan, state, reason="lifecycle-reconcile"):
                        _reconcile_lifecycle(plan, state)
                # CAMPAIGN COMPLETE only when the pool is empty AND all terminal.
                if not live_cells \
                        and all(row["status"] in TERMINAL for row in state["cells"].values()):
                    groups = _report_groups(plan, state)
                    reports_sealed = all(
                        group["ready_for_verified_report"]
                        and group["verified_report_checkpoint"] is not None
                        for group in groups
                    )
                    if not reports_sealed:
                        _sync_reporter(plan, state, reason="terminal-completeness-retry")
                        groups = _report_groups(plan, state)
                        reports_sealed = all(
                            group["ready_for_verified_report"]
                            and group["verified_report_checkpoint"] is not None
                            for group in groups
                        )
                    if not reports_sealed:
                        _save_state(plan, state, "waiting-prerequisites",
                                    active_children={}, active_cells=[])
                        _control_responsive_wait(plan, PREREQUISITE_POLL_SECONDS)
                        continue
                    _save_state(plan, state, "complete", supervisor_pid=None,
                                active_children={}, active_cells=[])
                    _append_event("campaign-complete", plan_sha256=plan["plan_sha256"])
                    return 0
                # ADMISSION: greedily fill free lanes with runnable chain-heads
                # (plan priority order = small models first) while the live
                # reservation sum plus the safety margin stays under budget.
                admitted_any = False
                if len(live_cells) < MAX_LANES:
                    heads, blockers = _scan_runnable_heads(plan, state)
                    state["last_scan"] = {
                        "at": _now(), "blocked_cell_count": len(blockers),
                        "runnable_cell": heads[0]["cell"]["cell_id"] if heads else None,
                        "runnable_cell_count": len(heads),
                        "blockers_sha256": _hash_value(blockers),
                    }
                    for execution in heads:
                        if len(live_cells) >= MAX_LANES:
                            break
                        cell = execution["cell"]
                        cell_id = cell["cell_id"]
                        if cell_id in state["active_children"] or cell_id in live_cells:
                            continue
                        reserve = _cell_reservation(cell)
                        if _reserved_total(live_cells) + reserve + SAFETY_MARGIN_BYTES \
                                > PROCESS_BUDGET_BYTES:
                            continue  # Try the next (smaller) head; no head-of-line block.
                        # Adopt complete, source-bound artifacts before spawning.
                        if execution["result_path"].exists() \
                                and execution["receipt_path"].exists():
                            result, receipt, errors = _validate_outputs(execution)
                            if not errors and isinstance(result, dict) \
                                    and isinstance(receipt, dict):
                                _commit_complete(plan, state, execution, result,
                                                 receipt, live_cells)
                                admitted_any = True
                                continue
                        row = state["cells"][cell_id]
                        row["status"] = "running"
                        row["attempts"] += 1
                        row["started_at"] = row["started_at"] or _now()
                        row["runtime_spec_sha256"] = execution["runtime"]["sha256"]
                        row["registry_sha256"] = execution["registry"]["registry_sha256"]
                        row["request_sha256"] = execution["request"]["request_sha256"]
                        _save_state(plan, state, "running-cell")
                        outcome = _launch_cell(plan, state, execution, reserve)
                        if isinstance(outcome, LiveCell):
                            live_cells[cell_id] = outcome
                            admitted_any = True
                        else:
                            _apply_cell_exit(plan, state, execution, int(outcome),
                                             live_cells)
                            admitted_any = True
                # TICK CADENCE: a live pool samples every control cadence (fast,
                # keeps RSS sampling + drain/pause responsive).  H1: whenever NO
                # child is live the loop must back off at least one poll interval
                # even when admitted_any is True, otherwise a terminal-at-launch
                # admission (e.g. _launch_cell returning RESOURCE_RC before spawn,
                # which _apply_cell_exit resets to pending) re-enters with zero
                # delay and busy-spins at 100% CPU under disk/thermal/swap
                # pressure.  _control_responsive_wait re-observes control each
                # inner tick so drain/pause stay responsive during the backoff.
                if live_cells:
                    _control_responsive_wait(plan, CONTROL_POLL_SECONDS)
                else:
                    _save_state(plan, state, "waiting-prerequisites",
                                active_children={}, active_cells=[])
                    _control_responsive_wait(plan, PREREQUISITE_POLL_SECONDS)
        except BaseException:
            # Fail-closed: terminate every live child before propagating, then
            # best-effort persist the quiesced set so recovery is clean.
            if state is not None:
                _abort_pool(state, live_cells)
                try:
                    _save_state(plan, state)
                except BaseException:
                    pass
            raise
        finally:
            try:
                fcntl.flock(heavy_owner.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            heavy_owner.close()


def _process_identity(pid: Any) -> tuple[str, str] | None:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 1:
        return None
    try:
        command = subprocess.run(["ps", "-ww", "-p", str(pid), "-o", "command="],
                                 capture_output=True, text=True, check=True).stdout.strip()
        started = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="],
                                 capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return (command, started) if command and started else None


def _pid_record(plan: dict[str, Any], nonce: str) -> dict[str, Any]:
    identity = _process_identity(os.getpid())
    if identity is None:
        raise CampaignError("could not capture supervisor process identity")
    command, started = identity
    payload = {
        "schema": PID_SCHEMA, "version": VERSION, "pid": os.getpid(),
        "process_started": started, "process_command_sha256": hashlib.sha256(
            command.encode("utf-8")
        ).hexdigest(),
        "ownership_nonce": nonce, "plan_sha256": plan["plan_sha256"],
        "recorded_at": _now(),
    }
    payload["pid_record_sha256"] = _hash_value(payload)
    return payload


def _owner_alive(record: Any, plan: dict[str, Any]) -> bool:
    if not isinstance(record, dict) or record.get("schema") != PID_SCHEMA \
            or record.get("version") != VERSION \
            or record.get("plan_sha256") != plan.get("plan_sha256") \
            or record.get("pid_record_sha256") != _hash_value(
                _without(record, "pid_record_sha256")
            ):
        return False
    nonce = record.get("ownership_nonce")
    identity = _process_identity(record.get("pid"))
    if identity is None or not isinstance(nonce, str) or NONCE_RE.fullmatch(nonce) is None:
        return False
    command, started = identity
    return (started == record.get("process_started")
            and hashlib.sha256(command.encode("utf-8")).hexdigest()
            == record.get("process_command_sha256")
            and "doctor_v5_ultra_queue.py run" in command
            and f"--nonce {nonce}" in command)


def start_queue() -> int:
    plan = _load_plan()
    owner = _read_json(PID_FILE, {})
    if _owner_alive(owner, plan):
        print(f"[doctor-v5-ultra] already active pid={owner['pid']}")
        return 0
    control = _load_control(plan)
    if control["mode"] != "run":
        set_control("run")
    nonce = secrets.token_hex(16)
    command = [sys.executable, str(SCRIPT), "run", "--nonce", nonce]
    if shutil.which("caffeinate"):
        command = ["caffeinate", "-dimsu", *command]
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, close_fds=True, shell=False)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        record = _read_json(PID_FILE, {})
        if record.get("ownership_nonce") == nonce and _owner_alive(record, plan):
            print(f"[doctor-v5-ultra] detached pid={record['pid']} log={LOG_FILE}")
            return 0
        if process.poll() is not None:
            break
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    raise CampaignError("detached ownership handshake failed")


def compile_campaign() -> int:
    plan = _compile_plan()
    if PLAN.exists():
        existing = _read_json(PLAN)
        # Created timestamps are not authority.  Refuse to replace a campaign with
        # completed work unless the semantic matrix is identical.
        comparable_existing = copy.deepcopy(existing)
        comparable_new = copy.deepcopy(plan)
        for value in (comparable_existing, comparable_new):
            if isinstance(value, dict):
                value.pop("plan_sha256", None)
                value.pop("created_at", None)
        if comparable_existing == comparable_new:
            errors = validate_plan(existing)
            if errors:
                raise CampaignError("existing semantically identical plan is invalid: "
                                    + "; ".join(errors))
            assert isinstance(existing, dict)
            plan = existing
            state = _load_state(plan)
            _publish_campaign(plan, state)
            print(json.dumps({
                "schema": "hawking.doctor_v5_ultra_compile_receipt.v1",
                "plan": _relative(PLAN), "campaign": _relative(CAMPAIGN),
                "plan_sha256": plan["plan_sha256"], "cells": len(plan["cells"]),
                "sub_120_cells": 280, "120B_cells": 40,
                "idempotent": True, "state_preserved": True,
                "source_deletion_permitted": False,
            }, indent=2, sort_keys=True))
            return 0
        if STATE.exists():
            state = _read_json(STATE, {})
            if any(row.get("status") in TERMINAL for row in state.get("cells", {}).values()
                   if isinstance(row, dict)):
                raise CampaignError("refusing to replace a campaign with terminal cells")
    _atomic_json(PLAN, plan)
    state = _base_state(plan)
    _atomic_json(STATE, state)
    _atomic_json(CONTROL, _base_control(plan))
    _publish_campaign(plan, state)
    _append_event("campaign-compiled", plan_sha256=plan["plan_sha256"], cells=320)
    print(json.dumps({
        "schema": "hawking.doctor_v5_ultra_compile_receipt.v1",
        "plan": _relative(PLAN), "campaign": _relative(CAMPAIGN),
        "plan_sha256": plan["plan_sha256"], "cells": len(plan["cells"]),
        "sub_120_cells": 280, "120B_cells": 40,
        "idempotent": False, "state_preserved": False,
        "source_deletion_permitted": False,
    }, indent=2, sort_keys=True))
    return 0


def _canonical_3b_marker() -> dict[str, Any]:
    """Promote the pinned raw-HF receipt into the campaign's canonical 3B label."""
    raw = _read_json(RAW_3B_MARKER)
    if not isinstance(raw, dict):
        raise CampaignError("raw Qwen 3B verified marker is not available")
    source_sha, source_bytes = _sha_file(RAW_3B_MARKER)
    verification = raw.get("verification")
    local_dir = Path(str(raw.get("local_dir", "")))
    if not local_dir.is_absolute():
        local_dir = ROOT / local_dir
    if raw.get("schema") != "hawking.frontier_download_verified.v1" \
            or raw.get("status") != "verified" \
            or raw.get("verified_complete") is not True \
            or raw.get("hf_download_returncode") != 0 \
            or not isinstance(verification, dict) \
            or verification.get("requested") is not True \
            or verification.get("returncode") != 0 \
            or raw.get("hf_id") != "Qwen/Qwen2.5-3B-Instruct" \
            or raw.get("label") != "Qwen/Qwen2.5-3B-Instruct" \
            or raw.get("revision") != QWEN_3B_REVISION \
            or raw.get("include_patterns") != [] \
            or local_dir.resolve(strict=True) != (ROOT / "scratch/qwen-3b").resolve(strict=True):
        raise CampaignError("raw Qwen 3B verified marker identity/policy is invalid")
    existing = _read_json(CANONICAL_3B_MARKER)
    if isinstance(existing, dict):
        if existing.get("promotion_receipt_sha256") != _hash_value(
            _without(existing, "promotion_receipt_sha256")
        ) or existing.get("source_verified_marker_sha256") != source_sha:
            raise CampaignError("existing canonical 3B marker is invalid or stale")
        return existing
    canonical = copy.deepcopy(raw)
    canonical.update({
        "label": "3B", "source_kind": "bf16 parent",
        "source_verified_marker_path": _relative(RAW_3B_MARKER),
        "source_verified_marker_sha256": source_sha,
        "source_verified_marker_bytes": source_bytes,
        "promotion_schema": "hawking.doctor_v5_download_marker_promotion.v1",
        "promotion_completed_at": _now(), "source_deletion_permitted": False,
    })
    canonical["promotion_receipt_sha256"] = _hash_value(canonical)
    _atomic_json(CANONICAL_3B_MARKER, canonical)
    return canonical


def finalize_3b_prerequisite() -> int:
    marker = _canonical_3b_marker()
    census_root = CENSUS_ROOT / "3B"
    census_path = census_root / "census.json"
    checkpoint_path = census_root / "census.checkpoint.json"
    manifest_path = PARAMETER_MANIFEST_ROOT / "3B.json"
    receipt_path = ULTRA_ROOT / "3B_source_prerequisite.json"
    existing = _read_json(receipt_path)
    if isinstance(existing, dict) \
            and existing.get("receipt_sha256") == _hash_value(
                _without(existing, "receipt_sha256")
            ):
        try:
            census_sha, census_bytes = _sha_file(census_path)
            manifest_sha, manifest_bytes = _sha_file(manifest_path)
            if existing.get("census") == {
                "path": _relative(census_path), "sha256": census_sha,
                "bytes": census_bytes,
            } and existing.get("parameter_manifest") == {
                "path": _relative(manifest_path), "sha256": manifest_sha,
                "bytes": manifest_bytes,
            } and existing.get("source_marker_sha256") \
                    == marker["source_verified_marker_sha256"]:
                print(json.dumps({"ok": True, "label": "3B",
                                  "receipt": _relative(receipt_path),
                                  "idempotent": True}, indent=2, sort_keys=True))
                return 0
        except (CampaignError, OSError):
            pass
    gate = _resource_gate(MIN_SCRATCH_BYTES)
    if not gate["ok"]:
        raise CampaignError("3B finalization resource gate refused: "
                            + "; ".join(gate["blockers"]))
    commands = [
        [sys.executable, str(HERE / "doctor_v5_census.py"), "run",
         "--label", "3B", "--hf-id", "Qwen/Qwen2.5-3B-Instruct",
         "--model-dir", str(ROOT / "scratch/qwen-3b"),
         "--output", str(census_path), "--checkpoint", str(checkpoint_path),
         "--expected-download-marker", str(CANONICAL_3B_MARKER)],
        [sys.executable, str(HERE / "doctor_v5_census.py"), "validate",
         str(census_path)],
        [sys.executable, str(HERE / "doctor_v5_parameter_manifest.py"), "build",
         "--census", str(census_path), "--output", str(manifest_path)],
        [sys.executable, str(HERE / "doctor_v5_parameter_manifest.py"), "validate",
         "--verify-files", str(manifest_path)],
    ]
    lease = _acquire_heavy_lease()
    if lease is None:
        raise CampaignError("shared heavy lease is busy; 3B finalization is resumable")
    try:
        for command in commands:
            process = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     capture_output=True, text=True, check=False)
            if process.returncode != 0:
                detail = (process.stderr or process.stdout).strip()[-4000:]
                raise CampaignError(
                    f"3B prerequisite command failed ({command[2]}): {detail}"
                )
    finally:
        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        lease.close()
    census_sha, census_bytes = _sha_file(census_path)
    manifest_sha, manifest_bytes = _sha_file(manifest_path)
    receipt: dict[str, Any] = {
        "schema": "hawking.doctor_v5_ultra_source_prerequisite.v1",
        "version": VERSION, "completed_at": _now(), "label": "3B",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct", "revision": QWEN_3B_REVISION,
        "verified_marker": {"path": _relative(CANONICAL_3B_MARKER),
                            "sha256": _sha_file(CANONICAL_3B_MARKER)[0]},
        "source_marker_sha256": marker["source_verified_marker_sha256"],
        "census": {"path": _relative(census_path), "sha256": census_sha,
                   "bytes": census_bytes},
        "parameter_manifest": {"path": _relative(manifest_path),
                               "sha256": manifest_sha, "bytes": manifest_bytes},
        "source_deletion_permitted": False,
    }
    receipt["receipt_sha256"] = _hash_value(receipt)
    _atomic_json(receipt_path, receipt)
    print(json.dumps({
        "ok": True, "label": "3B", "receipt": _relative(receipt_path),
        "census": receipt["census"], "parameter_manifest": receipt["parameter_manifest"],
        "source_deletion_permitted": False,
    }, indent=2, sort_keys=True))
    return 0


def await_3b_prerequisite() -> int:
    """Low-duty completion trigger: no download polling beyond one stat/30 s."""
    while not RAW_3B_MARKER.is_file():
        if _STOP:
            return 130
        time.sleep(30)
    result = finalize_3b_prerequisite()
    if result != 0:
        return result
    return await_arm_and_launch()


def _prerequisite_owner_alive(record: Any) -> bool:
    expected = {
        "schema", "pid", "process_started", "process_command_sha256",
        "launcher_command_sha256", "started_at", "log", "record_sha256",
    }
    if not isinstance(record, dict) or set(record) != expected \
            or record.get("schema") != "hawking.doctor_v5_ultra_prerequisite_pid.v2" \
            or record.get("record_sha256") != _hash_value(
                _without(record, "record_sha256")
            ):
        return False
    identity = _process_identity(record.get("pid"))
    if identity is None:
        return False
    command, started = identity
    return started == record.get("process_started") \
        and hashlib.sha256(command.encode("utf-8")).hexdigest() \
        == record.get("process_command_sha256") \
        and "doctor_v5_ultra_queue.py await-3b-prerequisite" in command


def start_3b_prerequisite_trigger() -> int:
    pid_path = ULTRA_ROOT / "3B_prerequisite.pid.json"
    existing = _read_json(pid_path, {})
    if _prerequisite_owner_alive(existing):
        print(json.dumps({"active": True, "pid": existing["pid"],
                          "idempotent": True}, indent=2, sort_keys=True))
        return 0
    log_path = ULTRA_ROOT / "3B_prerequisite.log"
    command = [sys.executable, str(SCRIPT), "await-3b-prerequisite"]
    if shutil.which("caffeinate"):
        command = ["caffeinate", "-dimsu", *command]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
            shell=False,
        )
    deadline = time.monotonic() + 15
    identity = None
    while time.monotonic() < deadline and process.poll() is None:
        first = _process_identity(process.pid)
        time.sleep(0.1)
        second = _process_identity(process.pid)
        if first is not None and first == second:
            identity = first
            break
    if identity is None:
        raise CampaignError("detached 3B prerequisite identity handshake failed")
    record = {"schema": "hawking.doctor_v5_ultra_prerequisite_pid.v2",
              "pid": process.pid, "process_started": identity[1],
              "process_command_sha256": hashlib.sha256(
                  identity[0].encode("utf-8")
              ).hexdigest(),
              "launcher_command_sha256": _hash_value(command),
              "started_at": _now(), "log": _relative(log_path)}
    record["record_sha256"] = _hash_value(record)
    _atomic_json(pid_path, record)
    if process.poll() is not None or not _prerequisite_owner_alive(record):
        raise CampaignError("detached 3B prerequisite trigger exited during handshake")
    print(json.dumps({"active": True, "pid": process.pid, "idempotent": False,
                      "log": _relative(log_path)}, indent=2, sort_keys=True))
    return 0


def _launch_source_bindings() -> dict[str, dict[str, Any]]:
    paths = {
        "orchestrator": SCRIPT,
        "reporter": REPORTER,
        "adapter_abi": Path(adapter_abi.__file__).resolve(),
        "codec_adapter": STRAND_LADDER_ADAPTER,
        "treatment_adapter": QWEN_TREATMENT_ADAPTER,
        "shared_worker": HERE / "doctor_v5_strand_ladder_worker.py",
        "pass_b_worker": HERE / "doctor_v5_pass_b_worker.py",
        "sharded_evaluator": HERE / "doctor_v5_sharded_eval.py",
        "gptoss_mxfp4_authority_builder": HERE / "doctor_v5_gptoss_mxfp4.py",
        "gptoss_fail_closed_adapter": HERE / "doctor_v5_gptoss_moe_adapter.py",
        "gptoss_mxfp4_inventory": (
            ULTRA_ROOT / "gpt_oss_120b_mxfp4_inventory.json"
        ),
        "ram_scheduler": Path(ram_scheduler.__file__).resolve(),
        "parameter_manifest_contract": Path(parameter_manifest.__file__).resolve(),
        "training_ladder_contract": Path(training_ladder_v5.__file__).resolve(),
        "training_ladder": LADDER_PATH,
        "strand_quantizer": ROOT / "vendor/strand-quant/target/release/quantize-model",
        "strand_attestor": (
            ROOT / "vendor/strand-decode-kernel/target/release/attest-strand"
        ),
        "strand_decoder": (
            ROOT / "vendor/strand-decode-kernel/target/release/archive-to-safetensors"
        ),
    }
    for model in COHORT:
        paths[f"parameter_manifest_{model['label']}"] = (
            PARAMETER_MANIFEST_ROOT / f"{model['label']}.json"
        )
        paths[f"source_census_{model['label']}"] = (
            CENSUS_ROOT / model["label"] / "census.json"
        )
    output: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        digest, size = _sha_file(path)
        output[name] = {"path": _relative(path), "sha256": digest, "bytes": size}
    return output


def _validated_independent_audit(path: Path) -> tuple[dict[str, Any], str, int]:
    """Require a semantic launch recommendation, not merely an arbitrary file hash."""
    path = path.resolve(strict=True)
    path.relative_to(ROOT.resolve())
    doc = _read_json(path)
    expected_keys = {
        "schema", "version", "audited_at", "scope", "source_bindings",
        "tests", "findings", "blockers", "counts", "launch_recommended",
        "source_deletion_permitted", "audit_sha256",
    }
    expected_counts = {
        "total_cells": 320, "sub_120b_cells": 280, "cells_120b": 40,
        "structurally_wired_cells": 280, "intentionally_blocked_cells": 40,
        "wired_120b_cells": 0, "blocked_120b_cells": 40,
    }
    if not isinstance(doc, dict) or set(doc) != expected_keys \
            or doc.get("schema") \
            != "hawking.doctor_v5_ultra_independent_launch_audit.v1" \
            or doc.get("version") != VERSION \
            or not isinstance(doc.get("audited_at"), str) or not doc["audited_at"] \
            or doc.get("source_bindings") != _launch_source_bindings() \
            or doc.get("counts") != expected_counts \
            or doc.get("blockers") != [] \
            or doc.get("launch_recommended") is not True \
            or doc.get("source_deletion_permitted") is not False \
            or doc.get("audit_sha256") != _hash_value(
                _without(doc, "audit_sha256")
            ):
        raise CampaignError("independent launch audit recommendation is invalid")
    scope = doc.get("scope")
    if not isinstance(scope, list) or not scope \
            or any(not isinstance(row, str) or not row for row in scope):
        raise CampaignError("independent launch audit scope is invalid")
    tests = doc.get("tests")
    if not isinstance(tests, list) or not tests:
        raise CampaignError("independent launch audit has no passing tests")
    for row in tests:
        if not isinstance(row, dict) or set(row) != {"name", "status", "evidence"} \
                or row.get("status") != "pass" \
                or any(not isinstance(row.get(key), str) or not row[key]
                       for key in ("name", "evidence")):
            raise CampaignError("independent launch audit test evidence is invalid")
    required_test_names = {
        "matrix_and_120b_parameter_authority",
        "pre_admission_gc_v2_handoff_and_recovery",
        "reporter_migration_and_auto_adoption",
        "process_tree_rss_enforcement",
        "runtime_unit_regression",
        "source_and_disk_readiness",
    }
    test_names = [row["name"] for row in tests]
    if len(test_names) != len(set(test_names)) \
            or not required_test_names.issubset(test_names):
        raise CampaignError("independent launch audit test coverage is incomplete")
    findings = doc.get("findings")
    if not isinstance(findings, list):
        raise CampaignError("independent launch audit findings are invalid")
    for row in findings:
        if not isinstance(row, dict) or set(row) != {"severity", "code", "detail"} \
                or row.get("severity") not in {"info", "limitation", "warning"} \
                or any(not isinstance(row.get(key), str) or not row[key]
                       for key in ("code", "detail")):
            raise CampaignError("independent launch audit finding is invalid")
    digest, size = _sha_file(path)
    return doc, digest, size


def arm_launch(args: argparse.Namespace) -> int:
    audit_path = Path(args.audit_receipt).resolve(strict=True)
    _, audit_sha, audit_bytes = _validated_independent_audit(audit_path)
    arm: dict[str, Any] = {
        "schema": "hawking.doctor_v5_ultra_launch_arm.v1", "version": VERSION,
        "armed": True, "armed_at": _now(), "expected_cells": 320,
        "expected_structurally_wired_cells": 280,
        "sources": _launch_source_bindings(),
        "independent_audit_receipt": {"path": _relative(audit_path),
                                      "sha256": audit_sha, "bytes": audit_bytes},
        "source_deletion_permitted": False,
    }
    arm["arm_sha256"] = _hash_value(arm)
    existing = _read_json(LAUNCH_ARM)
    if existing is not None and existing != arm:
        raise CampaignError("refusing to replace an existing different launch arm")
    _atomic_json(LAUNCH_ARM, arm)
    print(json.dumps(arm, indent=2, sort_keys=True))
    return 0


def _validated_launch_arm() -> dict[str, Any] | None:
    arm = _read_json(LAUNCH_ARM)
    expected_keys = {
        "schema", "version", "armed", "armed_at", "expected_cells",
        "expected_structurally_wired_cells", "sources", "independent_audit_receipt",
        "source_deletion_permitted", "arm_sha256",
    }
    if arm is None:
        return None
    if not isinstance(arm, dict) or set(arm) != expected_keys \
            or arm.get("schema") != "hawking.doctor_v5_ultra_launch_arm.v1" \
            or arm.get("version") != VERSION or arm.get("armed") is not True \
            or arm.get("expected_cells") != 320 \
            or arm.get("expected_structurally_wired_cells") != 280 \
            or arm.get("source_deletion_permitted") is not False \
            or arm.get("arm_sha256") != _hash_value(_without(arm, "arm_sha256")):
        raise CampaignError("launch arm is invalid")
    if arm.get("sources") != _launch_source_bindings():
        raise CampaignError("launch source hashes changed after arming")
    audit = arm.get("independent_audit_receipt")
    if not isinstance(audit, dict) or set(audit) != {"path", "sha256", "bytes"}:
        raise CampaignError("launch arm audit binding is invalid")
    path = _resolve_workspace_path(audit["path"])
    digest, size = _sha_file(path)
    if digest != audit["sha256"] or size != audit["bytes"]:
        raise CampaignError("launch arm audit receipt changed after arming")
    _validated_independent_audit(path)
    return arm


def _append_launch_log(row: dict[str, Any]) -> None:
    path = ULTRA_ROOT / "launch_trigger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_canonical(row).decode("utf-8") + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_dir(path.parent)


def _save_launch_trigger_state(state: dict[str, Any]) -> None:
    state.pop("state_sha256", None)
    state["state_sha256"] = _hash_value(state)
    _atomic_json(LAUNCH_TRIGGER_STATE, state)


def _run_armed_launch() -> int:
    arm = _validated_launch_arm()
    if arm is None:
        raise CampaignError("launch is not armed")
    arm_sha = arm["arm_sha256"]
    state = _read_json(LAUNCH_TRIGGER_STATE)
    if not isinstance(state, dict) or state.get("arm_sha256") != arm_sha:
        state = {"schema": "hawking.doctor_v5_ultra_launch_trigger_state.v1",
                 "version": VERSION, "arm_sha256": arm_sha, "status": "running",
                 "started_at": _now(), "updated_at": _now(),
                 "completed_phases": [], "active_phase": None, "error": None,
                 "source_deletion_permitted": False}
        _save_launch_trigger_state(state)
    phases = [
        ("selftest", [sys.executable, str(SCRIPT), "selftest"]),
        ("compile", [sys.executable, str(SCRIPT), "compile"]),
        ("wire-codec", [sys.executable, str(SCRIPT), "wire-codec"]),
        ("wire-doctor", [sys.executable, str(SCRIPT), "wire-doctor"]),
        ("reporter-init", [sys.executable, str(REPORTER), "init", "--campaign",
                           str(CAMPAIGN), "--reporting-root",
                           str(ULTRA_ROOT / "reporting")]),
        ("reporter-sync", [sys.executable, str(REPORTER), "sync", "--campaign",
                           str(CAMPAIGN), "--reporting-root",
                           str(ULTRA_ROOT / "reporting")]),
        ("reporter-deep-verify", [sys.executable, str(REPORTER), "verify",
                                  "--campaign", str(CAMPAIGN), "--reporting-root",
                                  str(ULTRA_ROOT / "reporting"), "--deep"]),
        ("readiness", [sys.executable, str(SCRIPT), "readiness"]),
        ("start", [sys.executable, str(SCRIPT), "start"]),
    ]
    completed = set(state.get("completed_phases", []))
    for name, command in phases:
        if name in completed:
            continue
        _validated_launch_arm()
        state.update({"status": "running", "active_phase": name,
                      "updated_at": _now(), "error": None})
        _save_launch_trigger_state(state)
        started = _now()
        process = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                 capture_output=True, text=True, check=False)
        row = {"at": _now(), "started_at": started, "phase": name,
               "command_sha256": _hash_value(command), "returncode": process.returncode,
               "stdout_tail": process.stdout[-12000:], "stderr_tail": process.stderr[-12000:]}
        _append_launch_log(row)
        if process.returncode != 0:
            state.update({"status": "failed", "active_phase": None,
                          "updated_at": _now(),
                          "error": f"{name} exited with status {process.returncode}"})
            _save_launch_trigger_state(state)
            return 2
        if name == "readiness":
            try:
                ready = json.loads(process.stdout)
            except json.JSONDecodeError as exc:
                raise CampaignError("readiness output is not JSON") from exc
            if ready.get("total_cells") != 320 \
                    or ready.get("structurally_wired_cells") != 280 \
                    or ready.get("blocked_cells") != 40 \
                    or ready.get("by_model", {}).get("120B") != {"wired": 0, "blocked": 40} \
                    or ready.get("registry_errors") != [] \
                    or ready.get("first_resource_gate", {}).get("ok") is not True:
                state.update({"status": "failed", "active_phase": None,
                              "updated_at": _now(),
                              "error": "readiness exact acceptance failed"})
                _save_launch_trigger_state(state)
                return 2
        completed.add(name)
        state["completed_phases"] = [phase for phase, _ in phases if phase in completed]
        state.update({"active_phase": None, "updated_at": _now()})
        _save_launch_trigger_state(state)
    state.update({"status": "complete", "active_phase": None,
                  "completed_at": _now(), "updated_at": _now(), "error": None})
    _save_launch_trigger_state(state)
    return 0


def await_arm_and_launch() -> int:
    while not LAUNCH_ARM.is_file():
        if _STOP:
            return 130
        time.sleep(30)
    return _run_armed_launch()


def _install_codec_registry() -> dict[str, Any]:
    if not STRAND_LADDER_ADAPTER.is_file() or STRAND_LADDER_ADAPTER.is_symlink():
        raise CampaignError("reviewed STRAND ladder adapter source is unavailable")
    adapter_sha, _ = _sha_file(STRAND_LADDER_ADAPTER)
    executable = Path(sys.executable)
    if executable.is_symlink():
        raise CampaignError("Python interpreter is a symlink")
    executable = executable.resolve(strict=True)
    executable_sha, _ = _sha_file(executable)
    entry = {
        "adapter_id": "doctor-v5-strand-ladder-qwen25-dense",
        "adapter_version": "1",
        "source_path": _relative(STRAND_LADDER_ADAPTER),
        "source_sha256": adapter_sha,
        "executable_path": str(executable),
        "executable_sha256": executable_sha,
        "entrypoint_argv": [str(executable), _relative(STRAND_LADDER_ADAPTER),
                            "run", "--request", "{request_path}"],
        "operations": ["condense_control"],
        "model_families": ["qwen2.5-dense"],
        "backends": ["apple-cpu-strand"],
        "request_schema": adapter_abi.REQUEST_SCHEMA,
        "result_schema": adapter_abi.RESULT_SCHEMA,
        "checkpoint_schema": adapter_abi.CHECKPOINT_SCHEMA,
        "reviewed": True, "execution_only_not_quality_evidence": True,
    }
    existing = _read_json(REGISTRY_PATH)
    entries: list[dict[str, Any]] = []
    if isinstance(existing, dict):
        errors = adapter_abi.validate_registry(existing, verify_files=False, base_dir=ROOT)
        if errors:
            raise CampaignError("existing Ultra adapter registry is invalid: "
                                + "; ".join(errors))
        entries = [dict(row) for row in existing["entries"]
                   if row.get("adapter_id") != entry["adapter_id"]]
    entries.append(entry)
    created_at = existing.get("created_at") if isinstance(existing, dict) else None
    registry = adapter_abi.build_registry(entries, created_at=created_at)
    errors = adapter_abi.validate_registry(registry, verify_files=True, base_dir=ROOT)
    if errors:
        raise CampaignError("refusing invalid Ultra adapter registry: " + "; ".join(errors))
    _atomic_json(REGISTRY_PATH, registry)
    return registry


def _install_treatment_registry() -> dict[str, Any]:
    if not QWEN_TREATMENT_ADAPTER.is_file() or QWEN_TREATMENT_ADAPTER.is_symlink():
        raise CampaignError("reviewed Qwen treatment adapter source is unavailable")
    adapter_sha, _ = _sha_file(QWEN_TREATMENT_ADAPTER)
    executable = Path(sys.executable)
    if executable.is_symlink():
        raise CampaignError("Python interpreter is a symlink")
    executable = executable.resolve(strict=True)
    executable_sha, _ = _sha_file(executable)
    definitions = (
        ("doctor-v5-static-repair", "doctor_static"),
        ("doctor-v5-conditional-repair", "doctor_conditional"),
        ("doctor-v5-full-treatment", "doctor_full"),
    )
    new_entries = [{
        "adapter_id": adapter_id, "adapter_version": "1",
        "source_path": _relative(QWEN_TREATMENT_ADAPTER),
        "source_sha256": adapter_sha,
        "executable_path": str(executable),
        "executable_sha256": executable_sha,
        "entrypoint_argv": [str(executable), _relative(QWEN_TREATMENT_ADAPTER),
                            "run", "--request", "{request_path}"],
        "operations": [operation], "model_families": ["qwen2.5-dense"],
        "backends": ["apple-cpu-strand"],
        "request_schema": adapter_abi.REQUEST_SCHEMA,
        "result_schema": adapter_abi.RESULT_SCHEMA,
        "checkpoint_schema": adapter_abi.CHECKPOINT_SCHEMA,
        "reviewed": True, "execution_only_not_quality_evidence": True,
    } for adapter_id, operation in definitions]
    replaced = {row[0] for row in definitions}
    existing = _read_json(REGISTRY_PATH)
    entries: list[dict[str, Any]] = []
    if isinstance(existing, dict):
        errors = adapter_abi.validate_registry(existing, verify_files=False, base_dir=ROOT)
        if errors:
            raise CampaignError("existing Ultra adapter registry is invalid: "
                                + "; ".join(errors))
        entries = [dict(row) for row in existing["entries"]
                   if row.get("adapter_id") not in replaced]
    entries.extend(new_entries)
    created_at = existing.get("created_at") if isinstance(existing, dict) else None
    registry = adapter_abi.build_registry(entries, created_at=created_at)
    errors = adapter_abi.validate_registry(registry, verify_files=True, base_dir=ROOT)
    if errors:
        raise CampaignError("refusing invalid Ultra treatment registry: "
                            + "; ".join(errors))
    _atomic_json(REGISTRY_PATH, registry)
    return registry


def _clear_unstarted_bindings(cells: list[dict[str, Any]],
                              state: dict[str, Any]) -> None:
    """Remove only readiness-created request/snapshot files before source rewiring."""
    for cell in cells:
        row = state["cells"][cell["cell_id"]]
        if row["attempts"] > 0 or row["status"] in TERMINAL | {"running"}:
            raise CampaignError(
                f"refusing to rewrite runtime identity after execution began: {cell['cell_id']}"
            )
        root = RESULTS / cell["cell_id"]
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise CampaignError(f"unstarted result root is unsafe: {root}")
        allowed = {"request.json", "adapter_registry.json"}
        unexpected = [path.name for path in root.iterdir() if path.name not in allowed]
        if unexpected:
            raise CampaignError(
                f"refusing to clear non-readiness artifacts for {cell['cell_id']}: "
                + ", ".join(sorted(unexpected))
            )
        for name in sorted(allowed):
            path = root / name
            if path.exists():
                if path.is_symlink() or not path.is_file():
                    raise CampaignError(f"readiness binding is unsafe: {path}")
                path.unlink()
        _fsync_dir(root)


def _run_codec_spec_builder(cell: dict[str, Any], program_sha256: str,
                            resource_sha256: str, scratch_bytes: int) -> None:
    command = [
        sys.executable, str(STRAND_LADDER_ADAPTER), "build-spec",
        "--label", cell["model_label"], "--rate-id", cell["rate_id"],
        "--cell-id", cell["cell_id"],
        "--cell-identity-sha256", cell["cell_identity_sha256"],
        "--program-spec-sha256", program_sha256,
        "--resource-admission-sha256", resource_sha256,
        "--evaluation-mode", "auto", "--disk-reserve-bytes", str(DISK_RESERVE_BYTES),
        "--scratch-budget-bytes", str(scratch_bytes), "--threads", "20",
        "--output", str(_resolve_workspace_path(cell["runtime_spec_path"],
                                                must_exist=False)),
    ]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                             timeout=300, check=False)
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip()[-2000:]
        raise CampaignError(f"codec spec builder refused {cell['cell_id']}: {detail}")


def _run_treatment_spec_builder(cell: dict[str, Any], program_sha256: str,
                                resource_sha256: str, scratch_bytes: int,
                                by_id: dict[str, dict[str, Any]]) -> None:
    command = [
        sys.executable, str(QWEN_TREATMENT_ADAPTER), "build-spec",
        "--operation", cell["command"], "--label", cell["model_label"],
        "--rate-id", cell["rate_id"], "--cell-id", cell["cell_id"],
        "--cell-identity-sha256", cell["cell_identity_sha256"],
        "--program-spec-sha256", program_sha256,
        "--resource-admission-sha256", resource_sha256,
    ]
    for dependency_id in cell["dependencies"]:
        dependency = by_id[dependency_id]
        command.extend([
            "--dependency",
            f"{dependency['branch']}:{dependency_id}:{dependency['cell_identity_sha256']}",
        ])
    command.extend([
        "--evaluation-mode", "auto", "--disk-reserve-bytes", str(DISK_RESERVE_BYTES),
        "--scratch-budget-bytes", str(scratch_bytes), "--threads", "20",
        "--output", str(_resolve_workspace_path(cell["runtime_spec_path"],
                                                 must_exist=False)),
    ])
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                             timeout=300, check=False)
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip()[-2000:]
        raise CampaignError(
            f"treatment spec builder refused {cell['cell_id']}: {detail}"
        )


def wire_codec() -> int:
    """Install the reviewed codec adapter and materialize all 70 Qwen specs."""
    plan = _load_plan()
    state = _load_state(plan)
    receipts: list[dict[str, Any]] = []
    codec_cells = [
        cell for cell in plan["cells"]
        if cell["branch"] == "codec_control" and cell["model_family"] == "qwen2.5-dense"
    ]
    if len(codec_cells) != 70:
        raise CampaignError("codec wiring requires exactly 70 Qwen cells")
    touched = [cell["cell_id"] for cell in codec_cells
               if state["cells"][cell["cell_id"]]["attempts"] > 0
               or state["cells"][cell["cell_id"]]["status"] in TERMINAL | {"running"}]
    if touched:
        raise CampaignError(
            "refusing to rewrite runtime identities after codec execution began: "
            + ", ".join(touched[:5])
        )
    _clear_unstarted_bindings(codec_cells, state)
    registry = _install_codec_registry()
    for cell in codec_cells:
        scratch = cell["admission"]["recommended_scratch_bytes"]
        resources = {"disk_reserve_bytes": DISK_RESERVE_BYTES,
                     "scratch_budget_bytes": scratch, "threads": 20}
        resource_sha = _hash_value(resources)
        # The builder needs the final hash inside the file.  First materialize the
        # reviewed geometry, hash its semantic program payload, then seal it.
        _run_codec_spec_builder(cell, "0" * 64, resource_sha, scratch)
        path = _resolve_workspace_path(cell["runtime_spec_path"])
        provisional = _read_json(path)
        if not isinstance(provisional, dict):
            raise CampaignError(f"codec builder emitted no spec: {cell['cell_id']}")
        program_sha = _hash_value(_runtime_program_payload(provisional))
        _run_codec_spec_builder(cell, program_sha, resource_sha, scratch)
        spec = _read_json(path)
        if not isinstance(spec, dict):
            raise CampaignError(f"sealed codec spec is unreadable: {cell['cell_id']}")
        # Avoid rehashing model-sized shards during wiring.  The scheduler and
        # adapter both verify them immediately before execution.
        if spec.get("program_spec_sha256") != _hash_value(_runtime_program_payload(spec)) \
                or spec.get("resource_admission_sha256") != _hash_value(spec.get("resources")) \
                or spec.get("campaign_binding", {}).get("cell_identity_sha256") \
                != cell["cell_identity_sha256"]:
            raise CampaignError(f"sealed codec spec identity failed: {cell['cell_id']}")
        digest, size = _sha_file(path)
        receipts.append({"cell_id": cell["cell_id"], "path": cell["runtime_spec_path"],
                         "sha256": digest, "bytes": size})
    receipt: dict[str, Any] = {
        "schema": "hawking.doctor_v5_ultra_codec_wiring_receipt.v1",
        "version": VERSION, "created_at": _now(), "plan_sha256": plan["plan_sha256"],
        "registry": {"path": _relative(REGISTRY_PATH),
                     "registry_sha256": registry["registry_sha256"],
                     "file_sha256": _sha_file(REGISTRY_PATH)[0]},
        "spec_count": len(receipts), "specs": receipts,
        "unsupported_not_omitted": {
            "120B_codec_cells": 10,
            "status": "remain_addressable_and_waiting_for_gpt_oss_typed_adapter",
        },
        "quality_claims_permitted": False, "source_deletion_permitted": False,
    }
    receipt["receipt_sha256"] = _hash_value(receipt)
    _atomic_json(ULTRA_ROOT / "codec_wiring_receipt.json", receipt)
    print(json.dumps({
        "schema": receipt["schema"], "plan_sha256": plan["plan_sha256"],
        "registry_sha256": registry["registry_sha256"], "spec_count": len(receipts),
        "120B_codec_waiting": 10,
        "receipt": _relative(ULTRA_ROOT / "codec_wiring_receipt.json"),
        "source_deletion_permitted": False,
    }, indent=2, sort_keys=True))
    return 0


def wire_doctor() -> int:
    """Install three reviewed Doctor adapters and seal all 210 Qwen specs."""
    plan = _load_plan()
    state = _load_state(plan)
    cells = [cell for cell in plan["cells"]
             if cell["branch"] != "codec_control"
             and cell["model_family"] == "qwen2.5-dense"]
    if len(cells) != 210:
        raise CampaignError("Doctor wiring requires exactly 210 Qwen cells")
    touched = [cell["cell_id"] for cell in cells
               if state["cells"][cell["cell_id"]]["attempts"] > 0
               or state["cells"][cell["cell_id"]]["status"] in TERMINAL | {"running"}]
    if touched:
        raise CampaignError(
            "refusing to rewrite runtime identities after Doctor execution began: "
            + ", ".join(touched[:5])
        )
    _clear_unstarted_bindings(cells, state)
    registry = _install_treatment_registry()
    by_id = {cell["cell_id"]: cell for cell in plan["cells"]}
    receipts: list[dict[str, Any]] = []
    for cell in cells:
        scratch = cell["admission"]["recommended_scratch_bytes"]
        resources = {"disk_reserve_bytes": DISK_RESERVE_BYTES,
                     "scratch_budget_bytes": scratch, "threads": 20}
        resource_sha = _hash_value(resources)
        _run_treatment_spec_builder(cell, "0" * 64, resource_sha, scratch, by_id)
        path = _resolve_workspace_path(cell["runtime_spec_path"])
        provisional = _read_json(path)
        if not isinstance(provisional, dict):
            raise CampaignError(f"Doctor builder emitted no spec: {cell['cell_id']}")
        program_sha = _hash_value(_runtime_program_payload(provisional))
        _run_treatment_spec_builder(cell, program_sha, resource_sha, scratch, by_id)
        spec = _read_json(path)
        if not isinstance(spec, dict) \
                or spec.get("program_spec_sha256") != _hash_value(
                    _runtime_program_payload(spec)
                ) \
                or spec.get("resource_admission_sha256") != _hash_value(
                    spec.get("resources")
                ) \
                or spec.get("campaign_binding", {}).get("cell_identity_sha256") \
                != cell["cell_identity_sha256"]:
            raise CampaignError(f"sealed Doctor spec identity failed: {cell['cell_id']}")
        digest, size = _sha_file(path)
        receipts.append({"cell_id": cell["cell_id"], "branch": cell["branch"],
                         "path": cell["runtime_spec_path"], "sha256": digest,
                         "bytes": size})
    receipt: dict[str, Any] = {
        "schema": "hawking.doctor_v5_ultra_doctor_wiring_receipt.v1",
        "version": VERSION, "created_at": _now(),
        "plan_sha256": plan["plan_sha256"],
        "registry": {"path": _relative(REGISTRY_PATH),
                     "registry_sha256": registry["registry_sha256"],
                     "file_sha256": _sha_file(REGISTRY_PATH)[0]},
        "spec_count": len(receipts),
        "branch_counts": {
            branch: sum(row["branch"] == branch for row in receipts)
            for branch in ("doctor_static", "doctor_conditional", "doctor_full")
        },
        "specs": receipts,
        "unsupported_not_omitted": {
            "120B_doctor_cells": 30,
            "status": "remain_addressable_and_waiting_for_gpt_oss_typed_adapters",
        },
        "quality_claims_permitted": False, "source_deletion_permitted": False,
    }
    receipt["receipt_sha256"] = _hash_value(receipt)
    path = ULTRA_ROOT / "doctor_wiring_receipt.json"
    _atomic_json(path, receipt)
    print(json.dumps({
        "schema": receipt["schema"], "plan_sha256": plan["plan_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "spec_count": len(receipts), "branch_counts": receipt["branch_counts"],
        "120B_doctor_waiting": 30, "receipt": _relative(path),
        "source_deletion_permitted": False,
    }, indent=2, sort_keys=True))
    return 0


def record_disposition(args: argparse.Namespace) -> int:
    plan = _load_plan()
    cell = next((row for row in plan["cells"] if row["cell_id"] == args.cell_id), None)
    if cell is None:
        raise CampaignError("unknown cell_id")
    evidence: list[dict[str, Any]] = []
    for raw in args.evidence:
        path = Path(raw).resolve(strict=True)
        try:
            relative = _relative(path)
        except ValueError as exc:
            raise CampaignError("disposition evidence must be inside the workspace") from exc
        digest, size = _sha_file(path)
        evidence.append({"role": f"evidence-{len(evidence):03d}", "path": relative,
                         "sha256": digest, "bytes": size})
    doc = {
        "schema": DISPOSITION_SCHEMA, "version": VERSION,
        "plan_sha256": plan["plan_sha256"], "cell_id": cell["cell_id"],
        "cell_identity_sha256": cell["cell_identity_sha256"], "status": args.status,
        "reason_code": args.reason_code, "detail": args.detail,
        "evidence_artifacts": evidence, "recorded_at": _now(),
        "quality_claims_permitted": False, "source_deletion_permitted": False,
    }
    doc["disposition_sha256"] = _hash_value(doc)
    path = _resolve_workspace_path(cell["disposition_path"], must_exist=False)
    if path.exists() and _read_json(path) != doc:
        raise CampaignError("refusing to replace an existing different disposition")
    _atomic_json(path, doc)
    print(json.dumps(doc, indent=2, sort_keys=True))
    return 0


def _validated_report_checkpoint(plan: dict[str, Any], state: dict[str, Any],
                                 group_id: str, path: Path) -> dict[str, Any]:
    groups = {row["group_id"]: row for row in _report_groups(plan, state)}
    if group_id not in groups:
        raise CampaignError(f"unknown report group: {group_id}")
    group = groups[group_id]
    if not group["ready_for_verified_report"]:
        raise CampaignError("report group is not terminal; verified checkpoint is premature")
    path = path.resolve(strict=True)
    path.relative_to(ROOT.resolve())
    doc = _read_json(path)
    required = {
        "schema", "version", "plan_sha256", "group_id", "covered_cells_sha256",
        "report_artifact", "verified", "source_deletion_permitted", "checkpoint_sha256",
    }
    if not isinstance(doc, dict) or set(doc) != required \
            or doc.get("schema") != REPORT_CHECKPOINT_SCHEMA \
            or doc.get("version") != VERSION or doc.get("plan_sha256") != plan["plan_sha256"] \
            or doc.get("group_id") != group_id or doc.get("verified") is not True \
            or doc.get("source_deletion_permitted") is not False \
            or doc.get("checkpoint_sha256") != _hash_value(
                _without(doc, "checkpoint_sha256")
            ):
        raise CampaignError("report checkpoint receipt is invalid")
    cell_evidence = [
        {"cell_id": cell["cell_id"], "status": state["cells"][cell["cell_id"]]["status"],
         "result_sha256": state["cells"][cell["cell_id"]]["result_sha256"],
         "disposition_sha256": state["cells"][cell["cell_id"]]["disposition_sha256"]}
        for cell in _group_cells(plan, group_id)
    ]
    if doc.get("covered_cells_sha256") != _hash_value(cell_evidence):
        raise CampaignError("report checkpoint does not cover the exact terminal cell set")
    artifact = doc.get("report_artifact")
    if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256", "bytes"}:
        raise CampaignError("report artifact binding invalid")
    report_path = _resolve_workspace_path(artifact["path"])
    digest, size = _sha_file(report_path)
    if digest != artifact["sha256"] or size != artifact["bytes"]:
        raise CampaignError("report artifact live identity mismatch")
    receipt_sha, _ = _sha_file(path)
    return {
        "path": _relative(path), "file_sha256": receipt_sha,
        "checkpoint_sha256": doc["checkpoint_sha256"], "accepted_at": _now(),
    }


def _adopt_reporter_checkpoints(plan: dict[str, Any], state: dict[str, Any]) -> int:
    """Adopt reporter-authored terminal group receipts without operator polling."""
    index_path = ULTRA_ROOT / "reporting" / "report_index.json"
    index = _read_json(index_path)
    if not isinstance(index, dict) \
            or index.get("schema") != "hawking.doctor_v5_campaign_report_index.v1" \
            or index.get("index_sha256") != _hash_value(
                _without(index, "index_sha256")
            ) \
            or index.get("campaign", {}).get("plan_sha256") != plan["plan_sha256"]:
        raise CampaignError("reporter index is invalid after successful sync")
    snapshot_raw = index.get("snapshot_path")
    if not isinstance(snapshot_raw, str):
        raise CampaignError("reporter index snapshot path is invalid")
    snapshot = Path(snapshot_raw).resolve(strict=True)
    snapshot.relative_to((ULTRA_ROOT / "reporting" / "snapshots").resolve())
    adopted = 0
    for reference in index.get("ultra_report_checkpoints", []):
        if not isinstance(reference, dict) \
                or reference.get("role") != "ultra_report_checkpoint" \
                or not isinstance(reference.get("path"), str):
            raise CampaignError("reporter group-checkpoint reference is invalid")
        relative = Path(reference["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise CampaignError("reporter group-checkpoint path escapes its snapshot")
        path = (snapshot / relative).resolve(strict=True)
        path.relative_to(snapshot)
        digest, size = _sha_file(path)
        if digest != reference.get("sha256") or size != reference.get("bytes"):
            raise CampaignError("reporter group-checkpoint reference changed")
        doc = _read_json(path)
        group_id = doc.get("group_id") if isinstance(doc, dict) else None
        if not isinstance(group_id, str):
            raise CampaignError("reporter group-checkpoint omits group identity")
        accepted = _validated_report_checkpoint(plan, state, group_id, path)
        prior = state["report_checkpoints"].get(group_id)
        if prior is None or prior.get("checkpoint_sha256") != accepted["checkpoint_sha256"]:
            state["report_checkpoints"][group_id] = accepted
            adopted += 1
    return adopted


def accept_report(args: argparse.Namespace) -> int:
    plan = _load_plan()
    state = _load_state(plan)
    path = Path(args.receipt)
    state["report_checkpoints"][args.group] = _validated_report_checkpoint(
        plan, state, args.group, path
    )
    _save_state(plan, state)
    print(json.dumps(state["report_checkpoints"][args.group], indent=2, sort_keys=True))
    return 0


def status() -> int:
    plan = _load_plan()
    state = _load_state(plan)
    owner = _read_json(PID_FILE, {})
    campaign = _campaign_projection(plan, state)
    summary = {
        "schema": "hawking.doctor_v5_ultra_status.v1", "generated_at": _now(),
        "active": _owner_alive(owner, plan),
        "pid": owner.get("pid") if isinstance(owner, dict) else None,
        "plan_sha256": plan["plan_sha256"], "queue_status": state["status"],
        "control_mode": _load_control(plan)["mode"], "active_cells": state["active_cells"],
        "counts": campaign["counts"], "report_groups": campaign["report_groups"],
        "timing": campaign["timing"],
        "last_resource_gate": state["last_resource_gate"],
        "last_scan": state["last_scan"], "last_reporter_sync": state["last_reporter_sync"],
        "monitor_cadence": {
            "control_poll_seconds": CONTROL_POLL_SECONDS,
            "resource_probe_seconds": RESOURCE_POLL_SECONDS,
            "prerequisite_rescan_seconds": PREREQUISITE_POLL_SECONDS,
        },
        "campaign_path": _relative(CAMPAIGN),
        "restart_command": "python3.12 tools/condense/doctor_v5_ultra_queue.py resume",
        "source_deletion_permitted": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def readiness() -> int:
    """Audit every addressable cell without rehashing model-sized shards."""
    plan = _load_plan()
    registry = _read_json(REGISTRY_PATH)
    registry_errors = adapter_abi.validate_registry(
        registry, verify_files=True, base_dir=ROOT
    )
    entries = {
        row.get("adapter_id"): row for row in registry.get("entries", [])
        if isinstance(row, dict)
    } if isinstance(registry, dict) and not registry_errors else {}
    wired: list[str] = []
    blockers: dict[str, list[str]] = {}
    by_branch = {row["branch"]: {"wired": 0, "blocked": 0} for row in BRANCHES}
    by_model = {row["label"]: {"wired": 0, "blocked": 0} for row in COHORT}
    for cell in plan["cells"]:
        reasons: list[str] = []
        path = _resolve_workspace_path(cell["runtime_spec_path"], must_exist=False)
        if not path.exists():
            reasons.append("typed runtime spec missing")
        else:
            spec = _read_json(path)
            _, _, reasons = _validate_runtime_spec(
                cell, spec, path, verify_inputs=False
            )
        entry = entries.get(cell["adapter_id"])
        if entry is None:
            reasons.append("reviewed adapter registry entry missing")
        else:
            if cell["command"] not in entry.get("operations", []):
                reasons.append("registry operation capability missing")
            if cell["model_family"] not in entry.get("model_families", []):
                reasons.append("registry model-family capability missing")
            if cell["backend"] not in entry.get("backends", []):
                reasons.append("registry backend capability missing")
        if reasons:
            blockers[cell["cell_id"]] = sorted(set(reasons))
            by_branch[cell["branch"]]["blocked"] += 1
            by_model[cell["model_label"]]["blocked"] += 1
        else:
            wired.append(cell["cell_id"])
            by_branch[cell["branch"]]["wired"] += 1
            by_model[cell["model_label"]]["wired"] += 1
    first = next((cell for cell in plan["cells"] if cell["cell_id"] in set(wired)), None)
    gate = _resource_gate(
        first["admission"]["recommended_scratch_bytes"],
        projected_output_bytes=first["projected_output_bytes"],
    ) \
        if first is not None else None
    payload = {
        "schema": "hawking.doctor_v5_ultra_readiness.v1", "generated_at": _now(),
        "plan_sha256": plan["plan_sha256"], "total_cells": len(plan["cells"]),
        "structurally_wired_cells": len(wired), "blocked_cells": len(blockers),
        "by_branch": by_branch, "by_model": by_model,
        "registry_errors": registry_errors,
        "first_structurally_runnable_cell": first["cell_id"] if first else None,
        "first_resource_gate": gate,
        "blocker_summary": {
            reason: sum(reason in rows for rows in blockers.values())
            for reason in sorted({reason for rows in blockers.values() for reason in rows})
        },
        "sample_blocked_cells": [
            {"cell_id": cell_id, "blockers": rows}
            for cell_id, rows in list(blockers.items())[:20]
        ],
        "full_input_hashing_deferred_to_immediate_prelaunch": True,
        "source_deletion_permitted": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not registry_errors and wired else 1


def validate_command(path: str | None) -> int:
    document = _read_json(Path(path)) if path else _read_json(PLAN)
    errors = validate_plan(document)
    print(json.dumps({"ok": not errors, "errors": errors,
                      "path": path or _relative(PLAN)}, indent=2, sort_keys=True))
    return 0 if not errors else 1


def selftest() -> int:
    # Pure compiler/validator coverage uses real reviewed manifests but no model bytes.
    if (PARAMETER_MANIFEST_ROOT / "3B.json").is_file():
        plan = _compile_plan()
    else:
        def manifest_provider(item: dict[str, Any]) -> dict[str, Any]:
            if item["label"] != "3B":
                return _manifest_binding(item)
            synthetic = copy.deepcopy(_manifest_binding(COHORT[1]))
            synthetic["exact_stored_parameter_count"] = 3_000_000_000
            synthetic["source_weight_bytes"] = 6_000_000_000
            return synthetic

        def census_provider(item: dict[str, Any]) -> dict[str, Any]:
            return _census_binding(COHORT[1] if item["label"] == "3B" else item)

        plan = _compile_plan(manifest_provider, census_provider)
    assert plan["matrix"]["cells"] == 320
    assert len([row for row in plan["cells"] if row["model_label"] != "120B"]) == 280
    assert len([row for row in plan["cells"] if row["model_label"] == "120B"]) == 40
    assert {row["rate_id"] for row in plan["cells"]} == {row["rate_id"] for row in RATES}
    assert {row["branch"] for row in plan["cells"]} == {row["branch"] for row in BRANCHES}
    assert all(row["expected_replicates"] == 1 for row in plan["cells"])
    assert all(row["replicate_scope"] == "preliminary_scale_mapping_not_dominance"
               for row in plan["cells"])
    assert not validate_plan(plan, verify_sources=False)
    damaged = copy.deepcopy(plan)
    damaged["cells"] = damaged["cells"][:-1]
    damaged["plan_sha256"] = _hash_value(_without(damaged, "plan_sha256"))
    assert any("320" in row for row in validate_plan(damaged, verify_sources=False))
    state = _base_state(plan)
    assert not _validate_state(state, plan)
    first_codec = plan["cells"][0]
    first_static = plan["cells"][1]
    ready, reasons = _dependency_state(first_static, state)
    assert not ready and reasons
    state["cells"][first_codec["cell_id"]]["status"] = "complete"
    ready, reasons = _dependency_state(first_static, state)
    assert ready and not reasons
    groups = _report_groups(plan, state)
    assert [row["expected_cells"] for row in groups] == [280, 40]
    assert not any(row["ready_for_verified_report"] for row in groups)
    terminal_state = _base_state(plan)
    for cell in plan["cells"]:
        if cell["model_label"] != "120B":
            terminal_state["cells"][cell["cell_id"]]["status"] = "unsupported"
    groups = _report_groups(plan, terminal_state)
    assert groups[0]["queue_terminal"] is True
    assert groups[0]["ready_for_verified_report"] is False
    assert groups[0]["reporter_incomplete_cells"] == 280
    assert groups[0]["terminal_cells"] == 280
    assert groups[1]["ready_for_verified_report"] is False
    # A blocked first cell must not deadlock a later runnable cell.
    scan_state = _base_state(plan)
    original_prepare = globals()["_prepare_execution"]
    original_disposition = globals()["_validate_disposition"]
    codec_seen = 0

    def fake_prepare(cell: dict[str, Any]) -> dict[str, Any]:
        nonlocal codec_seen
        codec_seen += 1
        if codec_seen == 1:
            return {"cell": cell, "blockers": ["adapter absent"]}
        return {"cell": cell, "blockers": []}

    try:
        globals()["_prepare_execution"] = fake_prepare
        globals()["_validate_disposition"] = lambda cell, current: (None, [])
        runnable, blocked = _scan(plan, scan_state)
        assert runnable is not None
        assert runnable["cell"]["branch"] == "codec_control"
        assert runnable["cell"]["cell_id"] != first_codec["cell_id"]
        assert first_codec["cell_id"] in blocked
    finally:
        globals()["_prepare_execution"] = original_prepare
        globals()["_validate_disposition"] = original_disposition
    # A sealed disposition remains authoritative after the automatic retry
    # ceiling, in both the serial and concurrent scan paths.
    saved = {name: globals()[name] for name in (
        "_validate_disposition", "_prepare_execution", "_append_event",
        "_save_state", "_sync_reporter",
    )}
    disposition_sha = "d" * 64
    try:
        globals()["_validate_disposition"] = lambda cell, current: (
            ({"status": "unsupported", "disposition_sha256": disposition_sha}, [])
            if cell["cell_id"] == first_codec["cell_id"] else (None, [])
        )
        globals()["_prepare_execution"] = lambda cell: {
            "cell": cell, "blockers": ["stop after disposition assertion"],
        }
        for name in ("_append_event", "_save_state", "_sync_reporter"):
            globals()[name] = lambda *args, **kwargs: None
        for scan in (_scan, _scan_runnable_heads):
            exhausted = _base_state(plan)
            row = exhausted["cells"][first_codec["cell_id"]]
            row.update({"status": "blocked-execution",
                        "attempts": MAX_AUTOMATIC_ATTEMPTS,
                        "blockers": ["retry ceiling"], "error": "exit 2"})
            scan(plan, exhausted)
            assert row["status"] == "unsupported"
            assert row["disposition_sha256"] == disposition_sha
            assert row["blockers"] == [] and row["error"] is None
    finally:
        globals().update(saved)
    control = _base_control(plan)
    assert control["control_sha256"] == _hash_value(_without(control, "control_sha256"))
    # The strict ceiling must not accept a one-third nominal rate as 0.33.
    cell_033 = next(row for row in plan["cells"] if row["rate_id"] == "0.33")
    fake = {
        "schema": cell_033["runtime_spec_schema"], "label": cell_033["model_label"],
        "family": cell_033["model_family"], "adapter_id": cell_033["adapter_id"],
        "operation": cell_033["command"],
        "codec": {"rate_id": "0.33", "nominal_payload_bpw": 1 / 3},
        "program_spec_sha256": "0" * 64, "resource_admission_sha256": "1" * 64,
        "resources": {"disk_reserve_bytes": DISK_RESERVE_BYTES,
                      "scratch_budget_bytes": MIN_SCRATCH_BYTES, "threads": 1},
        "inputs": [],
    }
    with tempfile.TemporaryDirectory(dir=ROOT / "scratch") as raw:
        fake_path = Path(raw) / "spec.json"
        _atomic_json(fake_path, fake)
        _, _, errors = _validate_runtime_spec(cell_033, fake, fake_path)
        assert any("exceeds" in row for row in errors)
    gate = _resource_gate(MIN_SCRATCH_BYTES)
    assert gate["required_free_bytes"] == DISK_RESERVE_BYTES + MIN_SCRATCH_BYTES
    low_disk = _admission_math(
        free_bytes=DISK_RESERVE_BYTES + MIN_SCRATCH_BYTES + 999,
        scratch_bytes=MIN_SCRATCH_BYTES, projected_output_bytes=1_000,
        resident_payload_bytes=36_000_000_000,
        resident_predecessor_bytes=18_000_000_000,
    )
    assert low_disk["capacity_ok"] is False
    assert low_disk["available_total_capacity_bytes"] \
        < low_disk["required_total_capacity_bytes"]
    first_full = next(row for row in plan["cells"]
                      if row["model_label"] == "0.5B" and row["rate_id"] == "4"
                      and row["branch"] == "doctor_full")
    second_full = next(row for row in plan["cells"]
                       if row["model_label"] == "0.5B" and row["rate_id"] == "3"
                       and row["branch"] == "doctor_full")
    second_codec = next(row for row in plan["cells"]
                        if row["model_label"] == "0.5B" and row["rate_id"] == "3"
                        and row["branch"] == "codec_control")
    assert [row["branch"] for row in _gc_targets(plan, first_full)] \
        == ["doctor_conditional"]
    assert [row["branch"] for row in _gc_targets(plan, second_full)] \
        == ["doctor_conditional"]
    assert [row["cell_id"] for row in _gc_targets(plan, second_codec)] \
        == [first_full["cell_id"]]
    # Full projected Qwen chain at the audited free-space snapshot: every target
    # is released before its successor gate, while each model's final 0.1 full
    # candidate remains durable.  This catches the 72B codec->static circularity.
    simulated_free = 277_069_541_376
    resident: dict[str, int] = {}
    minimum_margin: int | None = None
    minimum_cell: str | None = None
    simulated_margins: dict[str, int] = {}
    for cell in (row for row in plan["cells"] if row["model_label"] != "120B"):
        for target in _gc_targets(plan, cell):
            released_bytes = resident.pop(target["cell_id"], 0)
            simulated_free += released_bytes
        required = (DISK_RESERVE_BYTES
                    + cell["admission"]["recommended_scratch_bytes"]
                    + cell["projected_output_bytes"])
        margin = simulated_free - required
        simulated_margins[cell["cell_id"]] = margin
        if minimum_margin is None or margin < minimum_margin:
            minimum_margin, minimum_cell = margin, cell["cell_id"]
        assert margin >= 0, (cell["cell_id"], margin)
        simulated_free -= cell["projected_output_bytes"]
        resident[cell["cell_id"]] = cell["projected_output_bytes"]
    assert minimum_margin is not None and minimum_margin > 20_000_000_000
    assert minimum_cell is not None
    static_72_4 = next(row for row in plan["cells"]
                       if row["model_label"] == "72B" and row["rate_id"] == "4"
                       and row["branch"] == "doctor_static")
    assert simulated_margins[static_72_4["cell_id"]] > 20_000_000_000
    deferred = [row for row in plan["cells"]
                if row["model_label"] in {"32B", "72B"}]
    assert all(row["admission"]["recommended_scratch_bytes"] == MIN_SCRATCH_BYTES
               for row in deferred)
    # Audit authority is semantic: a complete source-bound all-pass recommendation
    # succeeds, while a freshly rehashed negative recommendation still fails.
    with tempfile.TemporaryDirectory(dir=ROOT / "scratch") as raw:
        audit_path = Path(raw) / "audit.json"
        audit = {
            "schema": "hawking.doctor_v5_ultra_independent_launch_audit.v1",
            "version": VERSION, "audited_at": _now(), "scope": ["selftest"],
            "source_bindings": _launch_source_bindings(),
            "tests": [
                {"name": name, "status": "pass", "evidence": "selftest"}
                for name in (
                    "matrix_and_120b_parameter_authority",
                    "pre_admission_gc_v2_handoff_and_recovery",
                    "reporter_migration_and_auto_adoption",
                    "process_tree_rss_enforcement",
                    "runtime_unit_regression",
                    "source_and_disk_readiness",
                )
            ],
            "findings": [], "blockers": [],
            "counts": {
                "total_cells": 320, "sub_120b_cells": 280, "cells_120b": 40,
                "structurally_wired_cells": 280,
                "intentionally_blocked_cells": 40,
                "wired_120b_cells": 0, "blocked_120b_cells": 40,
            },
            "launch_recommended": True, "source_deletion_permitted": False,
        }
        audit["audit_sha256"] = _hash_value(audit)
        _atomic_json(audit_path, audit)
        _validated_independent_audit(audit_path)
        audit["launch_recommended"] = False
        audit["audit_sha256"] = _hash_value(_without(audit, "audit_sha256"))
        _atomic_json(audit_path, audit)
        try:
            _validated_independent_audit(audit_path)
            raise AssertionError("negative audit recommendation was accepted")
        except CampaignError:
            pass
    # A live but unrelated/reused PID cannot suppress the detached arm watcher.
    current_identity = _process_identity(os.getpid())
    assert current_identity is not None
    fake_owner = {
        "schema": "hawking.doctor_v5_ultra_prerequisite_pid.v2",
        "pid": os.getpid(), "process_started": current_identity[1],
        "process_command_sha256": hashlib.sha256(
            current_identity[0].encode("utf-8")
        ).hexdigest(),
        "launcher_command_sha256": "0" * 64, "started_at": _now(),
        "log": "reports/condense/doctor_v5_ultra/3B_prerequisite.log",
    }
    fake_owner["record_sha256"] = _hash_value(fake_owner)
    assert _prerequisite_owner_alive(fake_owner) is False
    # Historical immutable reporter revisions remain valid GC recovery authority
    # even after the reporter's mutable current checkpoint advances.
    with tempfile.TemporaryDirectory(dir=ROOT / "scratch") as raw:
        temporary_ultra = Path(raw) / "ultra"
        recovery_target = {"cell_id": "recovery-cell"}
        recovery_row = {"result_sha256": "7" * 64}
        locator = hashlib.sha256(b"recovery-cell").hexdigest()
        checkpoint = {
            "cell_id": "recovery-cell", "status": "succeeded", "revision": 1,
            "completeness": {"complete": True},
            "provenance": {"declared": {
                "campaign_result_sha256": recovery_row["result_sha256"]
            }},
        }
        checkpoint["checkpoint_sha256"] = _hash_value(checkpoint)
        checkpoint_path = (temporary_ultra / "reporting" / "checkpoint_revisions"
                           / locator / (
                               f"r00000001-{checkpoint['checkpoint_sha256']}.json"
                           ))
        _atomic_json(checkpoint_path, checkpoint)
        recovery_ref = {
            "path": _relative(checkpoint_path),
            "sha256": _sha_file(checkpoint_path)[0],
            "bytes": _sha_file(checkpoint_path)[1],
        }
        original_ultra = ULTRA_ROOT
        globals()["ULTRA_ROOT"] = temporary_ultra
        try:
            assert _valid_immutable_reporter_ref(
                recovery_target, recovery_row, recovery_ref
            )
        finally:
            globals()["ULTRA_ROOT"] = original_ultra
    child_code = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c','import time; x=bytearray(8000000); time.sleep(30)']); "
        "x=bytearray(4000000); time.sleep(30)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_code], stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        time.sleep(0.5)
        identity = _process_identity(child.pid)
        assert identity is not None
        sample = _sample_child_tree(child.pid, child.pid, identity)
        assert sample["process_count"] >= 2 and sample["tree_rss_bytes"] > 8_000_000
        try:
            _sample_child_tree(child.pid, child.pid + 1, identity)
            raise AssertionError("wrong PGID was accepted")
        except CampaignError:
            pass
        _terminate(child, expected_pgid=child.pid, expected_identity=identity)
    finally:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)
    # A blocked first head must not stop _scan_runnable_heads from collecting a
    # later runnable head from a different (model, rate) GC-chain concurrently.
    heads_state = _base_state(plan)
    original_prepare = globals()["_prepare_execution"]
    original_disposition = globals()["_validate_disposition"]
    heads_codec_seen = 0

    def fake_prepare_heads(cell: dict[str, Any]) -> dict[str, Any]:
        nonlocal heads_codec_seen
        heads_codec_seen += 1
        if heads_codec_seen == 1:
            return {"cell": cell, "blockers": ["adapter absent"]}
        return {"cell": cell, "blockers": []}

    try:
        globals()["_prepare_execution"] = fake_prepare_heads
        globals()["_validate_disposition"] = lambda cell, current: (None, [])
        runnable_heads, blocked_heads = _scan_runnable_heads(plan, heads_state)
        assert len(runnable_heads) >= 2
        assert all(head["cell"]["branch"] == "codec_control" for head in runnable_heads)
        assert first_codec["cell_id"] in blocked_heads
        head_cells = {head["cell"]["cell_id"] for head in runnable_heads}
        assert first_codec["cell_id"] not in head_cells
        # Distinct heads are all chain-heads from distinct (model, rate) chains.
        assert len({(head["cell"]["model_label"], head["cell"]["rate_id"])
                    for head in runnable_heads}) == len(runnable_heads)
    finally:
        globals()["_prepare_execution"] = original_prepare
        globals()["_validate_disposition"] = original_disposition
    # RAM-budget reservation math.  Every tier's reservation stays within
    # [FLOOR, ceiling] and always admits into an empty pool (deadlock guard).
    # The 0.5B tier packs several lanes; a whole-residency large tier (14B)
    # self-limits to one; a MEASURED tier peak overrides a lower projection and
    # is itself clamped so the single-cell admission guarantee still holds.
    ceiling = PROCESS_BUDGET_BYTES - SAFETY_MARGIN_BYTES
    small_cell = next(row for row in plan["cells"] if row["model_label"] == "0.5B")
    whole_large = next(row for row in plan["cells"] if row["model_label"] == "14B")
    _OBSERVED_TIER_RSS.clear()
    for probe in (small_cell, whole_large):
        reserve = _cell_reservation(probe)
        assert RESIDENCY_FLOOR_BYTES <= reserve <= ceiling
        assert reserve + SAFETY_MARGIN_BYTES <= PROCESS_BUDGET_BYTES  # Admits empty.
    small_reserve = _cell_reservation(small_cell)
    assert 2 * small_reserve + SAFETY_MARGIN_BYTES <= PROCESS_BUDGET_BYTES  # >=2 lanes.
    large_reserve = _cell_reservation(whole_large)
    assert 2 * large_reserve + SAFETY_MARGIN_BYTES > PROCESS_BUDGET_BYTES    # 1 lane.
    _OBSERVED_TIER_RSS[small_cell["model_label"]] = PROCESS_BUDGET_BYTES  # Above ceiling.
    assert _cell_reservation(small_cell) == ceiling
    _OBSERVED_TIER_RSS.clear()
    # Exercise the concurrent pool primitives at a low synthetic RSS ceiling.
    # _launch_cell must spawn one lane and record it under active_children
    # (plural); a single enforcement tick over the aggregate must stop and reap
    # the bound group, durable-receipt the pool cutoff, empty active_children,
    # and release the per-child lease so the shared inode is immediately free.
    with tempfile.TemporaryDirectory(dir=ROOT / "scratch") as raw:
        runtime_root = Path(raw)
        output_dir = runtime_root / "result"
        output_dir.mkdir()
        request_path = output_dir / "request.json"
        _atomic_json(request_path, {"request_sha256": "8" * 64})
        memory_script = runtime_root / "memory_child.py"
        memory_script.write_text(
            "import time\nx = bytearray(50_000_000)\ntime.sleep(30)\n",
            encoding="utf-8",
        )
        cutoff_cell = plan["cells"][0]
        cutoff_state = _base_state(plan)
        execution = {
            "cell": cutoff_cell, "output_dir": output_dir,
            "request_path": request_path,
            "request": {"request_sha256": "8" * 64},
            "command": [sys.executable, str(memory_script), str(request_path)],
            "result_path": output_dir / "result.json",
            "receipt_path": output_dir / "execution_receipt.json",
            "checkpoint_path": output_dir / "checkpoint.json",
        }
        replaced_globals = {
            "STATE": STATE, "CAMPAIGN": CAMPAIGN, "CONTROL": CONTROL,
            "EVENTS": EVENTS, "CHILD_RESOURCE_LOG": CHILD_RESOURCE_LOG,
            "HEAVY_LOCK": HEAVY_LOCK, "CONTROL_POLL_SECONDS": CONTROL_POLL_SECONDS,
            "PROCESS_BUDGET_BYTES": PROCESS_BUDGET_BYTES,
            "_execution_resource_gate": globals()["_execution_resource_gate"],
        }
        globals().update({
            "STATE": runtime_root / "state.json",
            "CAMPAIGN": runtime_root / "campaign.json",
            "CONTROL": runtime_root / "control.json",
            "EVENTS": runtime_root / "events.jsonl",
            "CHILD_RESOURCE_LOG": runtime_root / "child_resources.jsonl",
            "HEAVY_LOCK": runtime_root / "heavy.lock",
            "CONTROL_POLL_SECONDS": 0.1,
            "PROCESS_BUDGET_BYTES": 20_000_000,
            "_execution_resource_gate": lambda *_args, **_kwargs: {"ok": True},
        })
        # Pin a healthy system snapshot so the H3 swap/pressure shed cannot fire on
        # a host that is itself under memory pressure; the aggregate tree-RSS guard
        # is the deterministic shedder exercised here.
        _saved_snapshot = ram_scheduler.resource_snapshot
        ram_scheduler.resource_snapshot = lambda *_a, **_k: {
            "pressure_level": 1, "swap_used_mb": 0,
        }
        try:
            live_cells: dict[str, LiveCell] = {}
            outcome = _launch_cell(plan, cutoff_state, execution, 12_000_000)
            assert isinstance(outcome, LiveCell)
            live_cells[cutoff_cell["cell_id"]] = outcome
            assert cutoff_cell["cell_id"] in cutoff_state["active_children"]
            assert cutoff_state["active_cells"] == [cutoff_cell["cell_id"]]
            assert _reserved_total(live_cells) == 12_000_000
            stopped: list[str] = []
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not stopped:
                samples, aggregate = _sample_active_children(live_cells)
                stopped = _enforce_pool_budget(
                    plan, cutoff_state, live_cells, samples, aggregate
                )
                if not stopped:
                    time.sleep(0.1)
            assert stopped == [cutoff_cell["cell_id"]]
            assert not live_cells
            assert cutoff_state["active_children"] == {}
            assert cutoff_state["active_cells"] == []
            assert _reserved_total(live_cells) == 0
            # M2: this lane's residency alone reaches the budget and it is the SOLE
            # live cell when shed, so it escalates to blocked-execution (retrying it
            # in isolation can never fit) instead of looping pending forever.  Its
            # consecutive resource-stop counter is cleared on escalation.
            assert cutoff_state["cells"][cutoff_cell["cell_id"]]["status"] \
                == "blocked-execution"
            assert cutoff_state["resource_stop_counts"].get(
                cutoff_cell["cell_id"]) is None
            assert cutoff_state["last_resource_stop"]["reason"] \
                == "pool_tree_rss_at_or_over_process_budget"
            cutoff_receipt = _read_json(output_dir / "resource_stop.json")
            assert cutoff_receipt["process_budget_bytes"] == 20_000_000
            stopped_pgid = cutoff_receipt["trigger_sample"]["pgid"]
            assert _process_identity(stopped_pgid) is None
            assert not any(not str(row.get("state", "")).startswith("Z")
                           for row in _process_group_rows(stopped_pgid))
            reacquired = _acquire_heavy_lease()
            assert reacquired is not None
            fcntl.flock(reacquired.fileno(), fcntl.LOCK_UN)
            reacquired.close()
            # If the root execs again after the stable handshake, per-tick
            # identity re-verification must fail closed; the pool then reaps the
            # start-time/PGID-bound group and empties active_children (plural).
            globals()["PROCESS_BUDGET_BYTES"] = 1_000_000_000
            exec_output = runtime_root / "exec-change-result"
            exec_output.mkdir()
            exec_request = exec_output / "request.json"
            _atomic_json(exec_request, {"request_sha256": "9" * 64})
            exec_pid_file = runtime_root / "exec-change.pid"
            exec_script = runtime_root / "exec_change.py"
            exec_script.write_text(
                "import os,sys,time\n"
                "open(sys.argv[2], 'w').write(str(os.getpid()))\n"
                "time.sleep(0.6)\n"
                "os.execv(sys.executable, [sys.executable, '-c', "
                "'import time; x=bytearray(30000000); time.sleep(30)', sys.argv[1]])\n",
                encoding="utf-8",
            )
            exec_state = _base_state(plan)
            exec_execution = {
                "cell": cutoff_cell, "output_dir": exec_output,
                "request_path": exec_request,
                "request": {"request_sha256": "9" * 64},
                "command": [sys.executable, str(exec_script), str(exec_request),
                            str(exec_pid_file)],
                "result_path": exec_output / "result.json",
                "receipt_path": exec_output / "execution_receipt.json",
                "checkpoint_path": exec_output / "checkpoint.json",
            }
            exec_cells: dict[str, LiveCell] = {}
            exec_outcome = _launch_cell(plan, exec_state, exec_execution, 12_000_000)
            assert isinstance(exec_outcome, LiveCell)
            exec_cells[cutoff_cell["cell_id"]] = exec_outcome
            raised = False
            try:
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    try:
                        _sample_active_children(exec_cells)
                    except CampaignError as exc:
                        assert "identity" in str(exc)
                        raised = True
                        break
                    time.sleep(0.1)
            finally:
                # Fail-closed cleanup: terminate + release the bound group.
                for cid in list(exec_cells):
                    live = exec_cells[cid]
                    try:
                        _terminate(live.process, expected_pgid=live.process_pgid,
                                   expected_identity=live.process_identity
                                   or live.spawn_identity)
                    except CampaignError:
                        pass
                    _release_cell(exec_state, exec_cells, live)
            assert raised
            exec_pid = int(exec_pid_file.read_text(encoding="utf-8"))
            assert exec_state["active_children"] == {}
            assert exec_state["active_cells"] == []
            assert _process_identity(exec_pid) is None
            assert not any(not str(row.get("state", "")).startswith("Z")
                           for row in _process_group_rows(exec_pid))
            reacquired = _acquire_heavy_lease()
            assert reacquired is not None
            fcntl.flock(reacquired.fileno(), fcntl.LOCK_UN)
            reacquired.close()
        finally:
            ram_scheduler.resource_snapshot = _saved_snapshot
            globals().update(replaced_globals)
    print("doctor_v5_ultra_queue.py selftest OK")
    return 0


def _signal_stop(_signal: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("finalize-3b-prerequisite")
    commands.add_parser("await-3b-prerequisite")
    commands.add_parser("start-3b-prerequisite-trigger")
    commands.add_parser("await-arm-and-launch")
    arm = commands.add_parser("arm-launch")
    arm.add_argument("--audit-receipt", required=True)
    commands.add_parser("compile")
    commands.add_parser("wire-codec")
    commands.add_parser("wire-doctor")
    validate = commands.add_parser("validate")
    validate.add_argument("path", nargs="?")
    commands.add_parser("status")
    commands.add_parser("readiness")
    commands.add_parser("start")
    run = commands.add_parser("run")
    run.add_argument("--nonce", required=True)
    commands.add_parser("pause")
    commands.add_parser("resume")
    commands.add_parser("drain")
    disposition = commands.add_parser("record-disposition")
    disposition.add_argument("--cell-id", required=True)
    disposition.add_argument("--status", choices=("negative", "unsupported"), required=True)
    disposition.add_argument("--reason-code", required=True)
    disposition.add_argument("--detail", required=True)
    disposition.add_argument("--evidence", action="append", default=[])
    report = commands.add_parser("accept-report")
    report.add_argument("--group", choices=("sub-120B", "120B"), required=True)
    report.add_argument("--receipt", required=True)
    commands.add_parser("selftest")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, _signal_stop)
    signal.signal(signal.SIGINT, _signal_stop)
    try:
        if args.command == "finalize-3b-prerequisite":
            return finalize_3b_prerequisite()
        if args.command == "await-3b-prerequisite":
            return await_3b_prerequisite()
        if args.command == "start-3b-prerequisite-trigger":
            return start_3b_prerequisite_trigger()
        if args.command == "await-arm-and-launch":
            return await_arm_and_launch()
        if args.command == "arm-launch":
            return arm_launch(args)
        if args.command == "compile":
            return compile_campaign()
        if args.command == "wire-codec":
            return wire_codec()
        if args.command == "wire-doctor":
            return wire_doctor()
        if args.command == "validate":
            return validate_command(args.path)
        if args.command == "status":
            return status()
        if args.command == "readiness":
            return readiness()
        if args.command == "start":
            return start_queue()
        if args.command == "run":
            return run_queue(args.nonce)
        if args.command == "pause":
            return set_control("pause")
        if args.command == "resume":
            set_control("run")
            return start_queue()
        if args.command == "drain":
            return set_control("drain")
        if args.command == "record-disposition":
            return record_disposition(args)
        if args.command == "accept-report":
            return accept_report(args)
        if args.command == "selftest":
            return selftest()
    except Exception as exc:
        print(f"[doctor-v5-ultra] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
